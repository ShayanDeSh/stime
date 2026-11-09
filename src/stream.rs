use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use crate::{Device, Result};

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

    /// Blocks until all work queued on the stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }
}
