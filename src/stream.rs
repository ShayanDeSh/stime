use std::sync::Arc;

use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaContext, CudaStream};

use crate::{Device, Event, Result};

/// An ordered queue of work on a device.
///
/// The stream is destroyed on drop.
pub struct Stream {
    inner: Arc<CudaStream>,
}

impl Stream {
    /// Creates a new stream on `device`.
    pub fn new(device: &Device) -> Result<Self> {
        let ctx = CudaContext::new(device.ordinal())?;
        Ok(Self {
            inner: ctx.new_stream()?,
        })
    }

    pub(crate) fn raw(&self) -> CUstream {
        self.inner.cu_stream()
    }

    pub(crate) fn inner(&self) -> &CudaStream {
        &self.inner
    }

    /// Makes all work queued on the stream after this call wait until the work captured by
    /// `event` has finished. Returns without blocking the host.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        Ok(self.inner.wait(event.inner())?)
    }

    /// Blocks until all work queued on the stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }
}
