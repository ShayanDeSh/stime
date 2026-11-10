use crate::{Device, Error, Event, Kernel, KernelArg, LaunchConfig, Result, Stream};

/// Where a [`Task`] is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Created but not yet launched.
    Pending,
    /// Launched onto a stream, waiting behind earlier work on that stream.
    Queued,
    /// The work ahead of it has finished, so the kernel is executing.
    Running,
    /// The kernel has finished.
    Completed,
}

/// The full parameter list of a kernel, in parameter order.
///
/// Implemented for tuples of [`KernelArg`]s, up to 12 elements.
pub trait KernelArgs {
    fn as_args(&self) -> Vec<&dyn KernelArg>;
}

macro_rules! impl_kernel_args {
    ($($arg:ident),*) => {
        impl<$($arg: KernelArg),*> KernelArgs for ($($arg,)*) {
            #[allow(non_snake_case)]
            fn as_args(&self) -> Vec<&dyn KernelArg> {
                let ($($arg,)*) = self;
                vec![$($arg as &dyn KernelArg),*]
            }
        }
    };
}

impl_kernel_args!();
impl_kernel_args!(A1);
impl_kernel_args!(A1, A2);
impl_kernel_args!(A1, A2, A3);
impl_kernel_args!(A1, A2, A3, A4);
impl_kernel_args!(A1, A2, A3, A4, A5);
impl_kernel_args!(A1, A2, A3, A4, A5, A6);
impl_kernel_args!(A1, A2, A3, A4, A5, A6, A7);
impl_kernel_args!(A1, A2, A3, A4, A5, A6, A7, A8);
impl_kernel_args!(A1, A2, A3, A4, A5, A6, A7, A8, A9);
impl_kernel_args!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
impl_kernel_args!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);
impl_kernel_args!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12);

/// Events recorded on the stream immediately before and after the kernel.
struct Markers {
    start: Event,
    end: Event,
}

/// One kernel launch: the kernel, its launch shape, and the arguments it runs with.
///
/// The task owns its arguments, so no buffer can be read, written or freed by the host while
/// the kernel may be using it. They are handed back by [`Task::wait`] once the kernel has
/// finished. Dropping a launched task blocks until the kernel has finished.
pub struct Task<A: KernelArgs> {
    kernel: Kernel,
    config: LaunchConfig,
    /// `None` only once `wait` has handed the arguments back.
    args: Option<A>,
    /// `None` until the task is launched.
    markers: Option<Markers>,
}

impl<A: KernelArgs> Task<A> {
    /// Creates a pending task that will run `kernel` with `args`.
    pub fn new(kernel: Kernel, config: LaunchConfig, args: A) -> Self {
        Self {
            kernel,
            config,
            args: Some(args),
            markers: None,
        }
    }

    /// Queues the kernel on `stream` and returns without waiting for it to run.
    ///
    /// Fails with `TaskAlreadyLaunched` if the task has been launched before. On any other
    /// failure the task stays pending, though the kernel may have run if it was already queued.
    ///
    /// # Safety
    /// - `args` must match the kernel's parameter list in number, order and type.
    /// - Every buffer in `args` must live on the stream's device.
    pub unsafe fn launch(&mut self, stream: &Stream) -> Result<()> {
        if self.markers.is_some() {
            return Err(Error::TaskAlreadyLaunched);
        }
        let args = self.args.as_ref().expect("arguments are held until `wait`");
        let device = Device::get(stream.inner().context().ordinal())?;
        let start = Event::new(&device)?;
        let end = Event::new(&device)?;
        start.record(stream)?;
        // SAFETY: the caller upholds the argument contract; the task keeps the arguments alive
        // until the kernel has finished.
        unsafe { self.kernel.launch(stream, self.config, &args.as_args()) }?;
        if let Err(err) = end.record(stream) {
            // The kernel is already queued but nothing tracks it. Wait for it here so the
            // arguments cannot be freed while it is still running.
            let _ = stream.synchronize();
            return Err(err);
        }
        self.markers = Some(Markers { start, end });
        Ok(())
    }

    /// Reports where the task is in its lifecycle, without waiting.
    ///
    /// Fails if the kernel, or work queued ahead of it, faulted on the device.
    pub fn state(&self) -> Result<TaskState> {
        let Some(markers) = &self.markers else {
            return Ok(TaskState::Pending);
        };
        if markers.end.is_complete()? {
            Ok(TaskState::Completed)
        } else if markers.start.is_complete()? {
            Ok(TaskState::Running)
        } else {
            Ok(TaskState::Queued)
        }
    }

    /// Blocks until the kernel has finished, then hands its arguments back.
    ///
    /// Fails with `TaskNotLaunched` if the task was never launched.
    pub fn wait(mut self) -> Result<A> {
        let markers = self.markers.as_ref().ok_or(Error::TaskNotLaunched)?;
        markers.end.synchronize()?;
        Ok(self.args.take().expect("arguments are held until `wait`"))
    }
}

impl<A: KernelArgs> Drop for Task<A> {
    fn drop(&mut self) {
        // The arguments must outlive the kernel. Errors cannot be reported from `drop`.
        if let Some(markers) = &self.markers {
            let _ = markers.end.synchronize();
        }
    }
}
