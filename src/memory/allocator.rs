use std::num::NonZeroU64;

use crate::{Device, Result};

/// A strategy for allocating and freeing untyped blocks of device memory.
///
/// `free` runs on the host as soon as a buffer is dropped. A freed block can be reused
/// immediately because a [`Task`](crate::Task) owns its buffers until its kernel has finished.
///
/// # Safety
/// - `alloc` must return a live pointer on `self.device()`, valid for `bytes` bytes, aligned for
///   any [`DeviceRepr`](cudarc::driver::DeviceRepr) type, and exclusively owned by the caller
///   until passed to `free`.
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
