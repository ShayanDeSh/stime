use std::sync::Arc;

use cudarc::driver::{CudaContext, DeviceRepr, ValidAsZeroBits};

use crate::{Allocator, Device, DeviceBuffer, Result, SystemAllocator};

/// Shared handle to an [`Allocator`] that produces typed [`DeviceBuffer`]s.
///
/// Cloning a pool shares the allocator. Buffers return their memory to it on drop.
#[derive(Clone)]
pub struct MemoryPool {
    allocator: Arc<dyn Allocator>,
    ctx: Arc<CudaContext>,
}

impl MemoryPool {
    /// Wraps `allocator` in a shareable pool handle.
    pub fn new<A: Allocator + 'static>(allocator: A) -> Result<Self> {
        let ctx = CudaContext::new(allocator.device().ordinal())?;
        Ok(Self {
            allocator: Arc::new(allocator),
            ctx,
        })
    }

    /// Creates a pool backed by [`SystemAllocator`] on `device`.
    pub fn system(device: &Device) -> Result<Self> {
        Self::new(SystemAllocator::new(device)?)
    }

    /// Device on which this pool allocates memory.
    pub fn device(&self) -> Device {
        self.allocator.device()
    }

    /// Releases any cached unused memory in the underlying allocator back to the device.
    pub fn trim(&self) -> Result<()> {
        self.allocator.trim()
    }

    /// Allocates a buffer of `len` elements with unspecified contents from this pool.
    ///
    /// # Safety
    /// The contents are uninitialized; every element must be written before it is read back.
    pub unsafe fn uninit<T: DeviceRepr>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        // SAFETY: the caller upholds initialization before readback.
        unsafe { DeviceBuffer::uninit_with(Arc::clone(&self.ctx), &self.allocator, len) }
    }

    /// Allocates a buffer of `len` elements from this pool, all bytes set to zero.
    pub fn zeroed<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        // SAFETY: the memory is zeroed below, which is a valid bit pattern for `T`.
        let mut buf = unsafe { self.uninit(len)? };
        buf.zero_bytes()?;
        Ok(buf)
    }

    /// Allocates a buffer from this pool and copies `data` into it.
    pub fn from_slice<T: DeviceRepr>(&self, data: &[T]) -> Result<DeviceBuffer<T>> {
        // SAFETY: every element is overwritten by the copy below.
        let mut buf = unsafe { self.uninit(data.len())? };
        buf.copy_from_host(data)?;
        Ok(buf)
    }
}
