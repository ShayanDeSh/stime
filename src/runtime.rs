//! Spreads [`Task`]s over several [`Worker`]s, each running on its own [`Stream`].
//!
//! ```text
//!                         +--> Worker 0 --> stream 0
//! Task<A> --> submit() ---+--> Worker 1 --> stream 1
//!        (round-robin)    +--> Worker 2 --> stream 2
//!                         +--> ...
//! ```
//!
//! Tasks sent to the same worker run in submission order. Tasks sent to different workers run
//! on different streams, so the GPU is free to overlap them.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{Device, KernelArgs, Result, Stream, Task, TaskHandle, Worker};

/// A fixed set of [`Worker`]s, one per stream, that share submitted tasks round-robin.
///
/// The first task goes to worker 0, the second to worker 1, and so on, wrapping around after
/// the last worker. Dropping the runtime drops every worker, which blocks until each has
/// finished and handed back its submitted tasks.
pub struct Runtime {
    workers: Vec<Worker>,
    /// Counts submissions; the worker for a submission is this value modulo `workers.len()`.
    next: AtomicUsize,
}

impl Runtime {
    /// Starts `workers` workers, each on its own new stream on `device`.
    ///
    /// Every worker queues at most `capacity` tasks waiting to be launched, as in
    /// [`Worker::new`].
    pub fn new(device: &Device, workers: NonZeroUsize, capacity: usize) -> Result<Self> {
        let workers = (0..workers.get())
            .map(|_| Ok(Worker::new(Stream::new(device)?, capacity)))
            .collect::<Result<_>>()?;
        Ok(Self {
            workers,
            next: AtomicUsize::new(0),
        })
    }

    /// Returns the number of workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Queues `task` on the next worker in round-robin order, blocking while that worker's
    /// queue is full.
    ///
    /// Fails with `WorkerStopped` if that worker's thread has exited. Launch and kernel errors,
    /// including a buffer on the wrong device, are reported by [`TaskHandle::wait`].
    ///
    /// # Safety
    /// The task's arguments must match the kernel's parameter list in number, order and type.
    pub unsafe fn submit<A: KernelArgs + Send + 'static>(
        &self,
        task: Task<A>,
    ) -> Result<TaskHandle<A>> {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        // SAFETY: the caller upholds the argument contract.
        unsafe { self.workers[index].submit(task) }
    }
}
