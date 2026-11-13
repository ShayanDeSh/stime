use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cudarc::driver::DriverError;
use cudarc::driver::sys::CUresult;

use crate::{Allocator, Device, Error, Result};

/// Smallest block the caching allocator hands out. Smaller requests are rounded up to this.
const MIN_BLOCK_BYTES: usize = 512;

/// Number of size classes per power of two (as a shift): `1 << 2` = 4.
const CLASS_SHIFT: u32 = 2;

/// Rounds `bytes` up to its size class, or `None` if that overflows `usize`.
///
/// Size classes split every power-of-two range into four equal steps (512, 640, 768, 896, 1024,
/// 1280, ...), so a block is at most 25% larger than the request while requests of similar size
/// still share a bucket.
fn bucket_size(bytes: usize) -> Option<usize> {
    let bytes = bytes.max(MIN_BLOCK_BYTES);
    let msb = usize::BITS - 1 - (bytes - 1).leading_zeros();
    let step = 1usize << (msb - CLASS_SHIFT);
    bytes.checked_add(step - 1).map(|n| n / step * step)
}

/// Counters describing the activity of a [`CachingAllocator`].
///
/// Byte counts are in terms of blocks (rounded-up bucket sizes), not requested sizes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocatorStats {
    /// Successful `alloc` calls.
    pub allocations: u64,
    /// `free` calls.
    pub frees: u64,
    /// Allocations served from a cached block.
    pub cache_hits: u64,
    /// Allocations that had to go to the underlying allocator.
    pub cache_misses: u64,
    /// Blocks returned to the underlying allocator, by a trim, an over-budget free, or an
    /// out-of-memory retry.
    pub driver_frees: u64,
    /// Bytes in blocks currently handed out.
    pub bytes_in_use: usize,
    /// Highest value `bytes_in_use` has reached.
    pub peak_bytes_in_use: usize,
    /// Bytes in idle blocks held for reuse.
    pub bytes_cached: usize,
}

#[derive(Default)]
struct State {
    /// Idle blocks keyed by block size.
    buckets: BTreeMap<usize, Vec<NonZeroU64>>,
    stats: AllocatorStats,
}

impl State {
    fn take_cached(&mut self) -> Vec<(NonZeroU64, usize)> {
        let blocks: Vec<_> = std::mem::take(&mut self.buckets)
            .into_iter()
            .flat_map(|(size, ptrs)| ptrs.into_iter().map(move |ptr| (ptr, size)))
            .collect();
        self.stats.bytes_cached = 0;
        self.stats.driver_frees += blocks.len() as u64;
        blocks
    }

    fn record_alloc(&mut self, block: usize) {
        let stats = &mut self.stats;
        stats.allocations += 1;
        stats.bytes_in_use += block;
        stats.peak_bytes_in_use = stats.peak_bytes_in_use.max(stats.bytes_in_use);
    }
}

struct Inner<A: Allocator> {
    allocator: A,
    cache_budget: usize,
    state: Mutex<State>,
}

impl<A: Allocator> Inner<A> {
    fn state(&self) -> MutexGuard<'_, State> {
        // The state is only updated by short, non-panicking sections, so it is consistent
        // even if another thread panicked while holding the lock.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns every cached block to the underlying allocator and reports how many there were.
    fn release_cached(&self) -> usize {
        let blocks = self.state().take_cached();
        for &(ptr, size) in &blocks {
            // SAFETY: each block came from `self.allocator.alloc(size)` and is idle.
            unsafe { self.allocator.free(ptr, size) };
        }
        blocks.len()
    }
}

impl<A: Allocator> Drop for Inner<A> {
    fn drop(&mut self) {
        self.release_cached();
    }
}

/// An [`Allocator`] that keeps freed blocks in size buckets and reuses them.
///
/// Requests are rounded up to a size class (512 bytes minimum, four classes per power of two)
/// and a freed block is reused by any later request that rounds to the same class. At most `cache_budget` bytes of idle blocks
/// are kept; a block freed past that limit goes straight back to the underlying allocator.
/// When the underlying allocator runs out of memory, the whole cache is released and the
/// allocation is retried once.
///
/// Cloning shares the same cache, so a clone can be kept to read [`stats`](Self::stats) after
/// the allocator is moved into a [`MemoryPool`](crate::MemoryPool). Cached blocks are released
/// when the last clone is dropped.
pub struct CachingAllocator<A: Allocator> {
    inner: Arc<Inner<A>>,
}

impl<A: Allocator> Clone for CachingAllocator<A> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<A: Allocator> CachingAllocator<A> {
    /// Caches blocks freed to `allocator`, holding at most `cache_budget` idle bytes.
    ///
    /// A budget of zero disables caching.
    pub fn new(allocator: A, cache_budget: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                allocator,
                cache_budget,
                state: Mutex::default(),
            }),
        }
    }

    /// Maximum number of idle bytes kept for reuse.
    pub fn cache_budget(&self) -> usize {
        self.inner.cache_budget
    }

    /// Returns a snapshot of the allocation counters.
    pub fn stats(&self) -> AllocatorStats {
        self.inner.state().stats
    }
}

// SAFETY: blocks come either from `self.inner.allocator` or from an earlier `free` of this
// allocator, and a cached block is removed from its bucket when handed out. Every block is at
// least as large as the request and keeps the alignment of the underlying allocator.
unsafe impl<A: Allocator> Allocator for CachingAllocator<A> {
    fn device(&self) -> Device {
        self.inner.allocator.device()
    }

    unsafe fn alloc(&self, bytes: usize) -> Result<NonZeroU64> {
        let inner = &*self.inner;
        let block = bucket_size(bytes)
            .ok_or(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY))?;

        {
            let mut state = inner.state();
            let cached = state.buckets.get_mut(&block).and_then(Vec::pop);
            if let Some(ptr) = cached {
                state.stats.bytes_cached -= block;
                state.stats.cache_hits += 1;
                state.record_alloc(block);
                return Ok(ptr);
            }
        }

        // SAFETY: `block` is non-zero.
        let ptr = match unsafe { inner.allocator.alloc(block) } {
            Ok(ptr) => ptr,
            Err(err @ Error::Driver(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY))) => {
                if inner.release_cached() == 0 {
                    return Err(err);
                }
                // SAFETY: `block` is non-zero.
                unsafe { inner.allocator.alloc(block)? }
            }
            Err(err) => return Err(err),
        };

        let mut state = inner.state();
        state.stats.cache_misses += 1;
        state.record_alloc(block);
        Ok(ptr)
    }

    unsafe fn free(&self, ptr: NonZeroU64, bytes: usize) {
        let inner = &*self.inner;
        // `alloc(bytes)` succeeded, so the size class exists.
        let block = bucket_size(bytes).expect("freed size was allocated");

        {
            let mut state = inner.state();
            state.stats.frees += 1;
            state.stats.bytes_in_use -= block;
            if state.stats.bytes_cached + block <= inner.cache_budget {
                state.buckets.entry(block).or_default().push(ptr);
                state.stats.bytes_cached += block;
                return;
            }
            state.stats.driver_frees += 1;
        }

        // SAFETY: `ptr` is a block of `block` bytes from `inner.allocator`, not used again.
        unsafe { inner.allocator.free(ptr, block) };
    }

    fn trim(&self) -> Result<()> {
        self.inner.release_cached();
        self.inner.allocator.trim()
    }
}
