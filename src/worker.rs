//! Runs [`Task`]s on a [`Stream`] from a dedicated host thread.
//!
//! Callers move tasks into a bounded queue and get a [`TaskHandle`] back. The worker thread
//! launches each task on the stream, keeps it until its kernel has finished, then sends the
//! task's arguments back through the handle.
//!
//! ```text
//! CALLER THREAD(S)
//! +----------------------------------------------------------------------+
//! |                                                                      |
//! |  Task<A> ---> unsafe worker.submit(task) ---> TaskHandle<A>          |
//! |                           |                         |                |
//! |                           | wraps                   | .wait()        |
//! |                           v                         v                |
//! |             Submitted<A> { task, done }     Result<A>: args back     |
//! |                           |                         ^                |
//! +---------------------------|-------------------------|----------------+
//!                             | Box<dyn Job>            |
//!                             v                         |
//!                   +-------------------+               |  one channel per task
//!                   |   bounded queue   |               |  (done -> result)
//!                   | `capacity` slots  |               |
//!                   +-------------------+               |
//!                             |                         |
//! WORKER THREAD               v                         |
//! +---------------------------|-------------------------|----------------+
//! |  WorkerThread        jobs.recv()                    |                |
//! |                           |                         |                |
//! |                           v                         |                |
//! |                      launch(job) --> in_flight      |                |
//! |                           |      [oldest..newest]   |                |
//! |                           |              |          |                |
//! |                           |              v          |                |
//! |                           |    finish_completed() --+                |
//! |                           |              ^                           |
//! +---------------------------|--------------|---------------------------+
//!                             | launch       | poll end event
//!                             v              |
//! GPU      stream:  [start|kernel|end] [start|kernel|end] ...
//! ```
//!
//! [`Worker`] is the handle callers hold and share. Everything the loop uses (the stream, the
//! receiving end of the queue and the in-flight tasks) is owned by the worker thread alone.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, channel, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::{Error, KernelArgs, Result, Stream, Task, TaskState};

/// How long the worker waits for a new task before checking in-flight tasks again.
const POLL_INTERVAL: Duration = Duration::from_micros(50);

/// A submitted task of any argument type, as seen by the worker thread.
trait Job: Send {
    /// Launches the task on `stream`. On failure the error is sent to the task's handle and
    /// `None` is returned.
    ///
    /// # Safety
    /// Same contract as [`Task::launch`].
    unsafe fn launch(self: Box<Self>, stream: &Stream) -> Option<Box<dyn Job>>;

    /// Returns whether the kernel has finished, or faulted, without waiting.
    fn is_finished(&self) -> bool;

    /// Waits for the kernel to finish and sends the arguments to the task's handle.
    fn finish(self: Box<Self>);
}

struct Submitted<A: KernelArgs> {
    task: Task<A>,
    done: Sender<Result<A>>,
}

impl<A: KernelArgs + Send + 'static> Job for Submitted<A> {
    unsafe fn launch(mut self: Box<Self>, stream: &Stream) -> Option<Box<dyn Job>> {
        // SAFETY: the caller upholds the argument contract.
        match unsafe { self.task.launch(stream) } {
            Ok(()) => Some(self),
            Err(err) => {
                // The handle may have been dropped, in which case nobody wants the error.
                let _ = self.done.send(Err(err));
                None
            }
        }
    }

    fn is_finished(&self) -> bool {
        match self.task.state() {
            Ok(TaskState::Queued | TaskState::Running) => false,
            // A fault is reported to the handle by `finish`.
            Ok(TaskState::Pending | TaskState::Completed) | Err(_) => true,
        }
    }

    fn finish(self: Box<Self>) {
        let Submitted { task, done } = *self;
        let _ = done.send(task.wait());
    }
}

/// A host thread that launches submitted [`Task`]s on one [`Stream`], in submission order.
///
/// Tasks wait in a bounded queue until the thread launches them. Once launched, the thread
/// keeps each task until its kernel has finished, then hands its arguments back through the
/// [`TaskHandle`] returned by [`Worker::submit`]. Dropping the worker blocks until every
/// submitted task has finished and been handed back.
pub struct Worker {
    /// `None` only during drop, so the thread sees the queue close.
    queue: Option<SyncSender<Box<dyn Job>>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    /// Starts a worker thread that runs tasks on `stream`.
    ///
    /// At most `capacity` tasks wait to be launched; with a capacity of 0, `submit` waits until
    /// the thread takes the task.
    pub fn new(stream: Stream, capacity: usize) -> Self {
        let (queue, jobs) = sync_channel(capacity);
        let thread = thread::spawn(move || WorkerThread::new(stream, jobs).run());
        Self {
            queue: Some(queue),
            thread: Some(thread),
        }
    }

    /// Queues `task` to be launched on the worker's stream, blocking while the queue is full.
    ///
    /// Fails with `WorkerStopped` if the worker thread has exited. Launch and kernel errors
    /// are reported by [`TaskHandle::wait`].
    ///
    /// # Safety
    /// - The task's arguments must match the kernel's parameter list in number, order and type.
    /// - Every buffer in the task's arguments must live on the stream's device.
    pub unsafe fn submit<A: KernelArgs + Send + 'static>(
        &self,
        task: Task<A>,
    ) -> Result<TaskHandle<A>> {
        let (done, result) = channel();
        let queue = self.queue.as_ref().expect("the queue is open until drop");
        queue
            .send(Box::new(Submitted { task, done }))
            .map_err(|_| Error::WorkerStopped)?;
        Ok(TaskHandle { result })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.queue = None;
        if let Some(thread) = self.thread.take() {
            // A panic on the worker thread has already been reported to every handle as
            // `WorkerStopped`.
            let _ = thread.join();
        }
    }
}

/// The state owned by the worker thread.
struct WorkerThread {
    stream: Stream,
    jobs: Receiver<Box<dyn Job>>,
    /// Launched tasks, oldest first.
    in_flight: VecDeque<Box<dyn Job>>,
}

impl WorkerThread {
    fn new(stream: Stream, jobs: Receiver<Box<dyn Job>>) -> Self {
        Self {
            stream,
            jobs,
            in_flight: VecDeque::new(),
        }
    }

    /// Launches queued tasks and hands back finished ones until the queue closes, then waits
    /// for whatever is still running.
    fn run(mut self) {
        loop {
            let next = if self.in_flight.is_empty() {
                self.jobs.recv().map_err(|_| RecvTimeoutError::Disconnected)
            } else {
                self.jobs.recv_timeout(POLL_INTERVAL)
            };
            match next {
                Ok(job) => self.launch(job),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            self.finish_completed();
        }
        for job in self.in_flight {
            job.finish();
        }
    }

    fn launch(&mut self, job: Box<dyn Job>) {
        // SAFETY: `submit`'s caller upheld the argument contract for this stream.
        if let Some(job) = unsafe { job.launch(&self.stream) } {
            self.in_flight.push_back(job);
        }
    }

    /// Hands back every task at the front of the in-flight queue whose kernel has finished.
    fn finish_completed(&mut self) {
        // The stream runs tasks in order, so nothing behind an unfinished task has finished.
        while self.in_flight.front().is_some_and(|job| job.is_finished()) {
            self.in_flight.pop_front().expect("front was checked").finish();
        }
    }
}

/// Receives a submitted task's arguments once its kernel has finished.
pub struct TaskHandle<A> {
    result: Receiver<Result<A>>,
}

impl<A> TaskHandle<A> {
    /// Blocks until the kernel has finished, then hands its arguments back.
    ///
    /// Fails if the task could not be launched or the kernel faulted, or with `WorkerStopped`
    /// if the worker thread exited before handing the task back.
    pub fn wait(self) -> Result<A> {
        self.result.recv().map_err(|_| Error::WorkerStopped)?
    }
}
