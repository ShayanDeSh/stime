use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::Arc;

use cudarc::driver::sys::{CUdeviceptr, CUresult};
use cudarc::driver::{CudaContext, DeviceRepr, DriverError, ValidAsZeroBits, result};

use crate::{Device, Error, Result};

/// A strategy for allocating and freeing untyped blocks of device memory.
///
/// `free` runs on the host as soon as a buffer is dropped. A freed block can be reused
/// immediately because a [`Task`](crate::Task) owns its buffers until its kernel has finished.
///
/// # Safety
/// - `alloc` must return a live pointer on `self.device()`, valid for `bytes` bytes, aligned for
///   any [`DeviceRepr`] type, and exclusively owned by the caller until passed to `free`.
/// - `free` is called with the exact `(ptr, bytes)` returned by `alloc`.
pub unsafe trait Allocator: Send + Sync {
    /// Device on which this allocator owns memory.
    fn device(&self) -> Device;

    /// Allocates `bytes` of uninitialized device memory.
    ///
    /// # Safety
    /// `bytes` must be non-zero.
    unsafe fn alloc(&self, bytes: usize) -> Result<NonZeroU64>;

    /// Releases a block returned by `alloc`.
    ///
    /// # Safety
    /// `ptr` must have been returned by `self.alloc(bytes)` and not freed since.
    unsafe fn free(&self, ptr: NonZeroU64, bytes: usize);

    /// Releases any cached unused memory back to the device.
    fn trim(&self) -> Result<()> {
        Ok(())
    }
}

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
    unsafe fn uninit_with(
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

    fn zero_bytes(&mut self) -> Result<()> {
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
