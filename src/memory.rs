use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaContext, DeviceRepr, ValidAsZeroBits, result};

use crate::{Device, Error, Result};

/// A fixed-size, owned allocation of `len` values of `T` in device memory.
///
/// The memory is freed on drop. Copies to and from the host are synchronous.
pub struct DeviceBuffer<T: DeviceRepr> {
    ctx: Arc<CudaContext>,
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Allocates a buffer of `len` elements with unspecified contents.
    ///
    /// # Safety
    /// The contents are uninitialized; every element must be written before it is read back.
    pub unsafe fn uninit(device: &Device, len: usize) -> Result<Self> {
        let ctx = CudaContext::new(device.ordinal())?;
        let bytes = len * size_of::<T>();
        // The driver rejects zero-sized allocations, so empty buffers hold a null pointer.
        let ptr = if bytes == 0 {
            0
        } else {
            unsafe { result::malloc_sync(bytes) }?
        };
        Ok(Self {
            ctx,
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Allocates a buffer and copies `data` into it.
    pub fn from_slice(device: &Device, data: &[T]) -> Result<Self> {
        // SAFETY: every element is overwritten by the copy below.
        let mut buf = unsafe { Self::uninit(device, data.len())? };
        buf.copy_from_host(data)?;
        Ok(buf)
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
        Ok(unsafe { result::memcpy_htod_sync(self.ptr, src) }?)
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
        // SAFETY: the memory is zeroed below, which is a valid bit pattern for `T`.
        let buf = unsafe { Self::uninit(device, len)? };
        if buf.size_in_bytes() != 0 {
            buf.ctx.bind_to_thread()?;
            unsafe { result::memset_d8_sync(buf.ptr, 0, buf.size_in_bytes()) }?;
        }
        Ok(buf)
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
        if self.ptr == 0 {
            return;
        }
        // Errors cannot be reported from `drop`; the driver reclaims the memory with the context.
        if self.ctx.bind_to_thread().is_ok() {
            let _ = unsafe { result::free_sync(self.ptr) };
        }
    }
}
