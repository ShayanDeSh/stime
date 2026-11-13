use std::num::NonZeroU64;
use std::sync::Arc;

use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, DriverError, result};

use crate::{Allocator, Device, Result};

/// Allocates directly from the CUDA driver without caching.
pub struct SystemAllocator {
    device: Device,
    ctx: Arc<CudaContext>,
}

impl SystemAllocator {
    /// Creates a non-pooling driver allocator on `device`.
    pub fn new(device: &Device) -> Result<Self> {
        let ctx = CudaContext::new(device.ordinal())?;
        Ok(Self {
            device: *device,
            ctx,
        })
    }
}

// SAFETY: `malloc_sync` returns driver-aligned allocations on `self.device` that stay live
// until `free_sync`.
unsafe impl Allocator for SystemAllocator {
    fn device(&self) -> Device {
        self.device
    }

    unsafe fn alloc(&self, bytes: usize) -> Result<NonZeroU64> {
        debug_assert!(bytes > 0);
        self.ctx.bind_to_thread()?;
        let ptr = unsafe { result::malloc_sync(bytes) }?;
        Ok(NonZeroU64::new(ptr).ok_or(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY))?)
    }

    unsafe fn free(&self, ptr: NonZeroU64, _bytes: usize) {
        // Errors cannot be reported from `drop`.
        if self.ctx.bind_to_thread().is_ok() {
            let _ = unsafe { result::free_sync(ptr.get()) };
        }
    }
}
