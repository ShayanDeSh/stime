use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use cudarc::driver::DriverError;
use cudarc::driver::sys::CUresult;
use stime::{Allocator, AllocatorStats, CachingAllocator, Device, MemoryPool, Result};

/// Hands out fake addresses up to a byte capacity and records what is live.
#[derive(Default)]
struct FakeState {
    next: u64,
    live: HashMap<NonZeroU64, usize>,
    allocs: usize,
    frees: usize,
}

#[derive(Clone)]
struct FakeAllocator {
    capacity: usize,
    state: Arc<Mutex<FakeState>>,
}

impl FakeAllocator {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Arc::default(),
        }
    }

    fn allocs(&self) -> usize {
        self.state.lock().unwrap().allocs
    }

    fn frees(&self) -> usize {
        self.state.lock().unwrap().frees
    }

    fn live_bytes(&self) -> usize {
        self.state.lock().unwrap().live.values().sum()
    }
}

// SAFETY: addresses are unique and tracked until freed with the same size; no memory is touched.
unsafe impl Allocator for FakeAllocator {
    fn device(&self) -> Device {
        Device::get(0).unwrap()
    }

    unsafe fn alloc(&self, bytes: usize) -> Result<NonZeroU64> {
        let mut state = self.state.lock().unwrap();
        if state.live.values().sum::<usize>() + bytes > self.capacity {
            return Err(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY).into());
        }
        state.next += 0x1000;
        let ptr = NonZeroU64::new(state.next).unwrap();
        state.live.insert(ptr, bytes);
        state.allocs += 1;
        Ok(ptr)
    }

    unsafe fn free(&self, ptr: NonZeroU64, bytes: usize) {
        let mut state = self.state.lock().unwrap();
        assert_eq!(state.live.remove(&ptr), Some(bytes), "free of unknown block");
        state.frees += 1;
    }
}

const MB: usize = 1 << 20;

#[test]
fn freed_block_is_reused_for_the_same_size() {
    // Given a caching allocator
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), MB);

    // When allocating, freeing, and allocating the same size again
    let first = unsafe { cache.alloc(1000).unwrap() };
    unsafe { cache.free(first, 1000) };
    let second = unsafe { cache.alloc(1000).unwrap() };

    // Then the block is reused without another underlying allocation
    assert_eq!(first, second);
    assert_eq!(fake.allocs(), 1);
    assert_eq!(cache.stats().cache_hits, 1);
    assert_eq!(cache.stats().cache_misses, 1);
}

#[test]
fn sizes_in_the_same_bucket_share_blocks() {
    // Given a freed 513-byte allocation, which rounds up to a 640-byte block
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), MB);
    let first = unsafe { cache.alloc(513).unwrap() };
    unsafe { cache.free(first, 513) };

    // When allocating a different size that rounds to the same block
    let second = unsafe { cache.alloc(600).unwrap() };

    // Then the cached block is reused
    assert_eq!(first, second);
    assert_eq!(fake.allocs(), 1);
}

#[test]
fn sizes_in_different_buckets_do_not_share_blocks() {
    // Given a freed 600-byte allocation (640-byte block)
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), MB);
    let first = unsafe { cache.alloc(600).unwrap() };
    unsafe { cache.free(first, 600) };

    // When allocating 700 bytes (768-byte block)
    let second = unsafe { cache.alloc(700).unwrap() };

    // Then a new block is taken from the underlying allocator
    assert_ne!(first, second);
    assert_eq!(fake.allocs(), 2);
    assert_eq!(cache.stats().cache_hits, 0);
}

#[test]
fn small_requests_round_up_to_the_minimum_block() {
    // Given a caching allocator
    let cache = CachingAllocator::new(FakeAllocator::new(usize::MAX), MB);

    // When allocating one byte
    let ptr = unsafe { cache.alloc(1).unwrap() };

    // Then a 512-byte block is in use
    assert_eq!(cache.stats().bytes_in_use, 512);
    unsafe { cache.free(ptr, 1) };
}

#[test]
fn blocks_over_the_budget_are_released() {
    // Given a budget that fits one 1024-byte block
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), 1024);
    let a = unsafe { cache.alloc(1024).unwrap() };
    let b = unsafe { cache.alloc(1024).unwrap() };

    // When freeing both
    unsafe { cache.free(a, 1024) };
    unsafe { cache.free(b, 1024) };

    // Then the first is cached and the second is released
    let stats = cache.stats();
    assert_eq!(stats.bytes_cached, 1024);
    assert_eq!(stats.driver_frees, 1);
    assert_eq!(fake.frees(), 1);
    assert_eq!(fake.live_bytes(), 1024);
}

#[test]
fn zero_budget_disables_caching() {
    // Given an allocator with no cache budget
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), 0);
    assert_eq!(cache.cache_budget(), 0);

    // When allocating and freeing
    let ptr = unsafe { cache.alloc(4096).unwrap() };
    unsafe { cache.free(ptr, 4096) };

    // Then the block goes straight back to the underlying allocator
    assert_eq!(fake.live_bytes(), 0);
    assert_eq!(cache.stats().bytes_cached, 0);
}

#[test]
fn trim_releases_cached_blocks() {
    // Given cached blocks in two buckets
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), MB);
    let a = unsafe { cache.alloc(1024).unwrap() };
    let b = unsafe { cache.alloc(4096).unwrap() };
    unsafe { cache.free(a, 1024) };
    unsafe { cache.free(b, 4096) };
    assert_eq!(cache.stats().bytes_cached, 1024 + 4096);

    // When trimming
    cache.trim().unwrap();

    // Then everything is returned to the underlying allocator
    assert_eq!(cache.stats().bytes_cached, 0);
    assert_eq!(cache.stats().driver_frees, 2);
    assert_eq!(fake.live_bytes(), 0);
}

#[test]
fn dropping_the_allocator_releases_cached_blocks() {
    // Given a cached block, with a clone of the allocator kept alive
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), MB);
    let clone = cache.clone();
    let ptr = unsafe { cache.alloc(1024).unwrap() };
    unsafe { cache.free(ptr, 1024) };

    // When dropping one handle
    drop(cache);

    // Then the cache is kept for the remaining handle
    assert_eq!(fake.live_bytes(), 1024);

    // When dropping the last handle
    drop(clone);

    // Then the cached block is released
    assert_eq!(fake.live_bytes(), 0);
}

#[test]
fn out_of_memory_releases_the_cache_and_retries() {
    // Given an underlying allocator filled by a cached 1024-byte block
    let fake = FakeAllocator::new(1024);
    let cache = CachingAllocator::new(fake.clone(), MB);
    let small = unsafe { cache.alloc(1024).unwrap() };
    unsafe { cache.free(small, 1024) };

    // When allocating a different size that does not fit alongside the cache
    let big = unsafe { cache.alloc(768).unwrap() };

    // Then the cache is released to make room
    assert_eq!(cache.stats().bytes_cached, 0);
    assert_eq!(cache.stats().driver_frees, 1);
    assert_eq!(fake.live_bytes(), 768);
    unsafe { cache.free(big, 768) };
}

#[test]
fn out_of_memory_with_nothing_cached_fails() {
    // Given an allocator whose capacity is fully in use
    let cache = CachingAllocator::new(FakeAllocator::new(1024), MB);
    let _held = unsafe { cache.alloc(1024).unwrap() };

    // When allocating more
    let result = unsafe { cache.alloc(1024) };

    // Then it fails and the counters are unchanged
    assert!(result.is_err());
    assert_eq!(cache.stats().allocations, 1);
}

#[test]
fn unrepresentable_size_fails() {
    // Given a caching allocator
    let cache = CachingAllocator::new(FakeAllocator::new(usize::MAX), MB);

    // When allocating a size whose block size overflows
    let result = unsafe { cache.alloc(usize::MAX) };

    // Then it fails
    assert!(result.is_err());
}

#[test]
fn stats_track_usage_and_peak() {
    // Given a caching allocator
    let cache = CachingAllocator::new(FakeAllocator::new(usize::MAX), MB);
    assert_eq!(cache.stats(), AllocatorStats::default());

    // When allocating two blocks, freeing one, and reusing it
    let a = unsafe { cache.alloc(1024).unwrap() };
    let b = unsafe { cache.alloc(2048).unwrap() };
    unsafe { cache.free(a, 1024) };
    let c = unsafe { cache.alloc(1024).unwrap() };
    unsafe { cache.free(b, 2048) };
    unsafe { cache.free(c, 1024) };

    // Then the counters reflect each step
    assert_eq!(
        cache.stats(),
        AllocatorStats {
            allocations: 3,
            frees: 3,
            cache_hits: 1,
            cache_misses: 2,
            driver_frees: 0,
            bytes_in_use: 0,
            peak_bytes_in_use: 1024 + 2048,
            bytes_cached: 1024 + 2048,
        }
    );
}

#[test]
fn concurrent_use_keeps_counters_consistent() {
    // Given a caching allocator shared by several threads
    let fake = FakeAllocator::new(usize::MAX);
    let cache = CachingAllocator::new(fake.clone(), MB);

    // When each thread repeatedly allocates and frees
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..200 {
                    let ptr = unsafe { cache.alloc(1000).unwrap() };
                    unsafe { cache.free(ptr, 1000) };
                }
            });
        }
    });

    // Then every allocation was freed and hits plus misses account for all of them
    let stats = cache.stats();
    assert_eq!(stats.allocations, 800);
    assert_eq!(stats.frees, 800);
    assert_eq!(stats.cache_hits + stats.cache_misses, 800);
    assert_eq!(stats.bytes_in_use, 0);
    assert_eq!(stats.cache_misses as usize, fake.allocs());
}

#[test]
fn memory_pool_reuses_device_memory() {
    // Given a pool backed by a caching allocator over the CUDA driver
    let device = Device::get(0).unwrap();
    let cache = CachingAllocator::new(stime::SystemAllocator::new(&device).unwrap(), 64 * MB);
    let pool = MemoryPool::new(cache.clone()).unwrap();

    // When creating and dropping buffers of similar size
    let first = pool.from_slice(&[1u32, 2, 3, 4]).unwrap();
    assert_eq!(first.to_vec().unwrap(), [1, 2, 3, 4]);
    drop(first);
    let second = pool.from_slice(&[5u32, 6, 7, 8]).unwrap();

    // Then the second buffer reuses the first's memory and holds its own data
    assert_eq!(second.to_vec().unwrap(), [5, 6, 7, 8]);
    assert_eq!(cache.stats().cache_hits, 1);
    drop(second);

    // And trimming the pool releases the cache
    pool.trim().unwrap();
    assert_eq!(cache.stats().bytes_cached, 0);
}
