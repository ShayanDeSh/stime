use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent};

use crate::{Device, Result, Stream};

/// A marker in a [`Stream`] that completes once all work queued before it has finished.
///
/// Events can be used to time work, to poll for completion, or to make one stream wait for
/// another (see [`Stream::wait_event`]). The event is destroyed on drop.
pub struct Event {
    inner: CudaEvent,
}

impl Event {
    /// Creates a new event on `device` with timing enabled.
    pub fn new(device: &Device) -> Result<Self> {
        let ctx = CudaContext::new(device.ordinal())?;
        Ok(Self {
            inner: ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        })
    }

    pub(crate) fn inner(&self) -> &CudaEvent {
        &self.inner
    }

    /// Captures the work currently queued on `stream`.
    ///
    /// The event completes once that work has finished. Recording again replaces the captured
    /// work. Fails if `stream` is on a different device than the event.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        Ok(self.inner.record(stream.inner())?)
    }

    /// Returns whether all captured work has finished, without waiting.
    ///
    /// An event that has never been recorded is complete.
    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.inner.try_is_complete()?)
    }

    /// Blocks until all captured work has finished.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }

    /// Milliseconds elapsed between `self` and `end`, blocking until both have completed.
    ///
    /// Both events must have been recorded, and must be on the same device.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        Ok(self.inner.elapsed_ms(end.inner())?)
    }
}
