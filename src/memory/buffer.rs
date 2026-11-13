use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::Arc;

use cudarc::driver::sys::{CUdeviceptr, CUresult};
use cudarc::driver::{CudaContext, DeviceRepr, DriverError, ValidAsZeroBits, result};

use crate::{Allocator, Device, Error, MemoryPool, Result};

/// A fixed-size, owned allocation of `len` values of `T` in device memory.
///
/// The memory is returned to its [`Allocator`] on drop. Host copies are synchronous.
pub struct DeviceBuffer<T: DeviceRepr> {
    device: Device,
    ctx: Arc<CudaContext>,
    allocator: Option<Arc<dyn Allocator>>,
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    pub(super) unsafe fn uninit_with(
        ctx: Arc<CudaContext>,
        allocator: &Arc<dyn Allocator>,
        len: usize,
    ) -> Result<Self> {
        let device = allocator.device();
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;
        // Empty buffers hold a null pointer, since the driver rejects zero-sized allocations.
        let (ptr, allocator) = if bytes == 0 {
            (0, None)
        } else {
            // SAFETY: `bytes` is non-zero.
            let ptr = unsafe { allocator.alloc(bytes)? };
            (ptr.get(), Some(Arc::clone(allocator)))
        };
        Ok(Self {
            device,
            ctx,
            allocator,
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Device on which the buffer lives.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Address of the device pointer, for use as a kernel parameter.
    pub(crate) fn device_ptr_ref(&self) -> *const CUdeviceptr {
        &self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_in_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Copies `src` into the buffer. Fails with `LengthMismatch` if `src.len() != self.len()`.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        self.check_len(src.len())?;
        if self.size_in_bytes() == 0 {
            return Ok(());
        }
        self.ctx.bind_to_thread()?;
        // SAFETY: `self.ptr` is a live allocation of `self.len` values of `T`.
        unsafe { result::memcpy_htod_sync(self.ptr, src) }?;
        self.sync_default_stream()
    }

    /// Copies the buffer into `dst`. Fails with `LengthMismatch` if `dst.len() != self.len()`.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        self.check_len(dst.len())?;
        if self.size_in_bytes() == 0 {
            return Ok(());
        }
        self.ctx.bind_to_thread()?;
        // SAFETY: `self.ptr` is a live allocation of `self.len` values of `T`.
        Ok(unsafe { result::memcpy_dtoh_sync(dst, self.ptr) }?)
    }

    /// Waits for host copies and memsets on the legacy default stream to finish. Non-blocking
    /// streams do not order themselves after it, and a pageable copy can return early.
    fn sync_default_stream(&self) -> Result<()> {
        // SAFETY: a null stream is the legacy default stream.
        Ok(unsafe { result::stream::synchronize(std::ptr::null_mut()) }?)
    }

    fn check_len(&self, actual: usize) -> Result<()> {
        if actual == self.len {
            Ok(())
        } else {
            Err(Error::LengthMismatch {
                expected: self.len,
                actual,
            })
        }
    }
}

impl<T: DeviceRepr + ValidAsZeroBits> DeviceBuffer<T> {
    /// Allocates a buffer of `len` elements, all bytes set to zero.
    pub fn zeroed(device: &Device, len: usize) -> Result<Self> {
        MemoryPool::system(device)?.zeroed(len)
    }

    pub(super) fn zero_bytes(&mut self) -> Result<()> {
        if self.size_in_bytes() != 0 {
            self.ctx.bind_to_thread()?;
            // SAFETY: `self.ptr` is a live allocation of `self.size_in_bytes()` bytes.
            unsafe { result::memset_d8_sync(self.ptr, 0, self.size_in_bytes()) }?;
            self.sync_default_stream()?;
        }
        Ok(())
    }
}

impl<T: DeviceRepr + Default + Clone> DeviceBuffer<T> {
    /// Copies the buffer into a new `Vec`.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut out = vec![T::default(); self.len];
        self.copy_to_host(&mut out)?;
        Ok(out)
    }
}

impl<T: DeviceRepr> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if let (Some(allocator), Some(ptr)) = (self.allocator.take(), NonZeroU64::new(self.ptr)) {
            // SAFETY: `ptr` was allocated with `self.size_in_bytes()` from `allocator` and will
            // not be used again.
            unsafe { allocator.free(ptr, self.size_in_bytes()) };
        }
    }
}
