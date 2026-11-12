use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, Mutex};

use stime::{
    Allocator, Device, DeviceBuffer, Error, MemoryPool, Module, Result, Runtime, SystemAllocator,
    Task,
};

mod common;

use common::{ADD_PTX, config};

fn device() -> Device {
    Device::get(0).unwrap()
}

#[test]
fn from_slice_round_trips_data() {
    // Given host data
    let data = [1.0f32, 2.0, 3.0, 4.0];

    // When copying it to the device and back
    let buffer = MemoryPool::system(&device()).unwrap().from_slice(&data).unwrap();
    let result = buffer.to_vec().unwrap();

    // Then the data is unchanged
    assert_eq!(result, data);
}

#[test]
fn zeroed_buffer_reads_back_as_zeros() {
    // Given a zeroed buffer
    let buffer = DeviceBuffer::<u32>::zeroed(&device(), 8).unwrap();

    // When reading it back
    let result = buffer.to_vec().unwrap();

    // Then every element is zero
    assert_eq!(result, vec![0u32; 8]);
}

#[test]
fn buffer_reports_its_size() {
    // Given a buffer of five u32 values
    let buffer = DeviceBuffer::<u32>::zeroed(&device(), 5).unwrap();

    // When querying its size
    // Then the length is in elements and the size is in bytes
    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.size_in_bytes(), 20);
    assert!(!buffer.is_empty());
}

#[test]
fn copy_from_host_overwrites_contents() {
    // Given a zeroed buffer
    let mut buffer = DeviceBuffer::<i32>::zeroed(&device(), 3).unwrap();

    // When copying new data into it
    buffer.copy_from_host(&[7, 8, 9]).unwrap();

    // Then reading it back returns the new data
    assert_eq!(buffer.to_vec().unwrap(), [7, 8, 9]);
}

#[test]
fn copy_to_host_fills_the_destination() {
    // Given a buffer with known contents
    let buffer = MemoryPool::system(&device()).unwrap().from_slice(&[10u8, 20, 30]).unwrap();

    // When copying it into a host slice
    let mut dst = [0u8; 3];
    buffer.copy_to_host(&mut dst).unwrap();

    // Then the slice holds the buffer's contents
    assert_eq!(dst, [10, 20, 30]);
}

#[test]
fn empty_buffer_is_usable() {
    // Given an empty buffer
    let buffer = DeviceBuffer::<u8>::zeroed(&device(), 0).unwrap();

    // When reading it back
    let result = buffer.to_vec().unwrap();

    // Then it is empty and the copy succeeds
    assert!(buffer.is_empty());
    assert!(result.is_empty());
}

#[test]
fn copy_from_host_with_wrong_length_fails() {
    // Given a buffer of three elements
    let mut buffer = DeviceBuffer::<u8>::zeroed(&device(), 3).unwrap();

    // When copying a slice of a different length
    let result = buffer.copy_from_host(&[1, 2]);

    // Then it fails with a length mismatch
    assert_eq!(
        result,
        Err(Error::LengthMismatch {
            expected: 3,
            actual: 2
        })
    );
}

#[test]
fn copy_to_host_with_wrong_length_fails() {
    // Given a buffer of three elements
    let buffer = DeviceBuffer::<u8>::zeroed(&device(), 3).unwrap();

    // When copying into a slice of a different length
    let result = buffer.copy_to_host(&mut [0u8; 4]);

    // Then it fails with a length mismatch
    assert_eq!(
        result,
        Err(Error::LengthMismatch {
            expected: 3,
            actual: 4
        })
    );
}

#[test]
fn oversized_allocation_fails() {
    // Given a size far beyond the device's memory
    let len = usize::MAX / 2;

    // When allocating a buffer of that size
    let result = DeviceBuffer::<u8>::zeroed(&device(), len);

    // Then it returns an error
    assert!(result.is_err());
}

#[test]
fn buffers_are_freed_on_drop() {
    // Given repeated allocations of a large buffer
    // When each is dropped before the next
    // Then allocation keeps succeeding, so memory is released
    let device = device();
    let free_bytes = device.total_memory().unwrap() / 2;
    for _ in 0..8 {
        let buffer = DeviceBuffer::<u8>::zeroed(&device, free_bytes).unwrap();
        assert_eq!(buffer.len(), free_bytes);
    }
}

#[test]
fn buffer_can_be_used_from_another_thread() {
    // Given a buffer created on this thread
    let buffer = MemoryPool::system(&device()).unwrap().from_slice(&[1u32, 2, 3]).unwrap();

    // When reading it back from a different thread
    let result = std::thread::spawn(move || buffer.to_vec().unwrap())
        .join()
        .unwrap();

    // Then the data is intact
    assert_eq!(result, [1, 2, 3]);
}

#[derive(Default)]
struct PoolStats {
    driver_allocs: usize,
    driver_frees: usize,
    pool_hits: usize,
    allocated_ptrs: Vec<NonZeroU64>,
    freed_blocks: Vec<(NonZeroU64, usize)>,
}

struct RecyclingAllocator {
    inner: SystemAllocator,
    stats: Arc<Mutex<PoolStats>>,
}

impl RecyclingAllocator {
    fn new(device: &Device) -> (Self, Arc<Mutex<PoolStats>>) {
        let stats = Arc::new(Mutex::new(PoolStats::default()));
        let alloc = Self {
            inner: SystemAllocator::new(device).unwrap(),
            stats: Arc::clone(&stats),
        };
        (alloc, stats)
    }
}

// SAFETY: delegates device memory ownership to `SystemAllocator` and only hands out blocks
// currently sitting in `freed_blocks` with a matching byte size.
unsafe impl Allocator for RecyclingAllocator {
    fn device(&self) -> Device {
        self.inner.device()
    }

    unsafe fn alloc(&self, bytes: usize) -> Result<NonZeroU64> {
        let mut stats = self.stats.lock().unwrap();
        if let Some(pos) = stats.freed_blocks.iter().position(|&(_, b)| b == bytes) {
            let (ptr, _) = stats.freed_blocks.swap_remove(pos);
            stats.pool_hits += 1;
            stats.allocated_ptrs.push(ptr);
            return Ok(ptr);
        }
        // SAFETY: `bytes` is non-zero.
        let ptr = unsafe { self.inner.alloc(bytes)? };
        stats.driver_allocs += 1;
        stats.allocated_ptrs.push(ptr);
        Ok(ptr)
    }

    unsafe fn free(&self, ptr: NonZeroU64, bytes: usize) {
        self.stats.lock().unwrap().freed_blocks.push((ptr, bytes));
    }

    fn trim(&self) -> Result<()> {
        let mut stats = self.stats.lock().unwrap();
        let blocks = std::mem::take(&mut stats.freed_blocks);
        for (ptr, bytes) in blocks {
            // SAFETY: each `(ptr, bytes)` came from `self.inner.alloc(bytes)` and is idle.
            unsafe { self.inner.free(ptr, bytes) };
            stats.driver_frees += 1;
        }
        Ok(())
    }
}

impl Drop for RecyclingAllocator {
    fn drop(&mut self) {
        let _ = self.trim();
    }
}

#[test]
fn system_pool_allocates_round_trips_and_trims() {
    // Given a system-backed memory pool
    let dev = device();
    let pool = MemoryPool::system(&dev).unwrap();
    assert_eq!(pool.device(), dev);

    // When allocating initialized, zeroed, uninit, and empty buffers from it
    let from_slice = pool.from_slice(&[1.5f32, -2.5, 3.5]).unwrap();
    let zeroed = pool.zeroed::<u64>(6).unwrap();
    // SAFETY: every element is written via `copy_from_host` before being read back.
    let mut uninit = unsafe { pool.uninit::<i32>(4).unwrap() };
    uninit.copy_from_host(&[10, 20, 30, 40]).unwrap();
    let empty = pool.zeroed::<u8>(0).unwrap();

    // Then each buffer holds the expected contents and trimming succeeds
    assert_eq!(from_slice.to_vec().unwrap(), [1.5, -2.5, 3.5]);
    assert_eq!(zeroed.to_vec().unwrap(), [0u64; 6]);
    assert_eq!(uninit.to_vec().unwrap(), [10, 20, 30, 40]);
    assert!(empty.is_empty());
    assert!(empty.to_vec().unwrap().is_empty());
    pool.trim().unwrap();
}

#[test]
fn custom_allocator_reuses_dropped_blocks_and_trims() {
    // Given a pool backed by a custom recycling allocator
    let dev = device();
    let (alloc, stats) = RecyclingAllocator::new(&dev);
    let pool = MemoryPool::new(alloc).unwrap();

    // When creating and dropping zero-sized buffers
    {
        let _empty_zeroed = pool.zeroed::<u32>(0).unwrap();
        let _empty_slice = pool.from_slice::<u32>(&[]).unwrap();
        // SAFETY: empty buffer has no elements to initialize.
        let _empty_uninit = unsafe { pool.uninit::<u32>(0).unwrap() };
    }

    // Then the allocator is never called for zero-byte allocations or drops
    {
        let s = stats.lock().unwrap();
        assert_eq!(s.driver_allocs, 0);
        assert_eq!(s.pool_hits, 0);
        assert!(s.freed_blocks.is_empty());
    }

    // When allocating a 4-element u32 buffer and dropping it
    let buf1 = pool.from_slice(&[1u32, 2, 3, 4]).unwrap();
    assert_eq!(buf1.to_vec().unwrap(), [1, 2, 3, 4]);
    let first_ptr = stats.lock().unwrap().allocated_ptrs[0];
    drop(buf1);

    // Then `free` receives the exact pointer and byte count instead of freeing to the driver
    {
        let s = stats.lock().unwrap();
        assert_eq!(s.driver_allocs, 1);
        assert_eq!(s.driver_frees, 0);
        assert_eq!(s.freed_blocks, vec![(first_ptr, 16)]);
    }

    // When allocating matching-size buffers with `zeroed` and `from_slice`
    let buf2 = pool.zeroed::<u32>(4).unwrap();
    assert_eq!(buf2.to_vec().unwrap(), [0, 0, 0, 0]);
    drop(buf2);

    let buf3 = pool.from_slice(&[9u32, 8, 7, 6]).unwrap();
    assert_eq!(buf3.to_vec().unwrap(), [9, 8, 7, 6]);
    drop(buf3);

    // Then both reuse the cached device pointer without additional driver allocations
    {
        let s = stats.lock().unwrap();
        assert_eq!(s.driver_allocs, 1);
        assert_eq!(s.pool_hits, 2);
        assert_eq!(s.allocated_ptrs, vec![first_ptr, first_ptr, first_ptr]);
        assert_eq!(s.freed_blocks, vec![(first_ptr, 16)]);
    }

    // When trimming the pool
    pool.trim().unwrap();

    // Then the cached block is released back to the underlying driver allocator
    {
        let s = stats.lock().unwrap();
        assert!(s.freed_blocks.is_empty());
        assert_eq!(s.driver_frees, 1);
    }
}

#[test]
fn pooled_buffers_execute_in_runtime_and_return_on_drop() {
    // Given a custom-pooled allocator, a runtime, and the `add` kernel
    let dev = device();
    let (alloc, stats) = RecyclingAllocator::new(&dev);
    let pool = MemoryPool::new(alloc).unwrap();
    let runtime = Runtime::new(&dev, NonZeroUsize::new(2).unwrap(), 4).unwrap();
    let add = Module::from_ptx(&dev, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();

    let len = 256;
    let n = len as u32;
    let bytes = len * size_of::<f32>();
    let a_host: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..len).map(|i| (10 * i) as f32).collect();

    // When running a task with pooled buffers and waiting for completion
    let a = pool.from_slice(&a_host).unwrap();
    let b = pool.from_slice(&b_host).unwrap();
    let out = pool.zeroed::<f32>(len).unwrap();
    let task = Task::new(add.clone(), config(n), (a, b, out, n));
    // SAFETY: the arguments match `add`'s signature and live on the runtime's device.
    let handle = unsafe { runtime.submit(task) }.unwrap();
    let (a, b, out, _) = handle.wait().unwrap();

    // Then the buffers are still owned by the caller until dropped, and hold the sum
    assert!(stats.lock().unwrap().freed_blocks.is_empty());
    let expected: Vec<f32> = (0..len).map(|i| (11 * i) as f32).collect();
    assert_eq!(out.to_vec().unwrap(), expected);

    // When dropping the returned buffers and submitting a second task of the same shape
    drop((a, b, out));
    {
        let s = stats.lock().unwrap();
        assert_eq!(s.driver_allocs, 3);
        assert_eq!(s.freed_blocks.len(), 3);
        assert!(s.freed_blocks.iter().all(|&(_, b)| b == bytes));
    }

    let a2 = pool.from_slice(&vec![2.0f32; len]).unwrap();
    let b2 = pool.from_slice(&vec![3.0f32; len]).unwrap();
    let out2 = pool.zeroed::<f32>(len).unwrap();
    let task2 = Task::new(add, config(n), (a2, b2, out2, n));
    // SAFETY: the arguments match `add`'s signature and live on the runtime's device.
    let (_, _, out2, _) = unsafe { runtime.submit(task2) }.unwrap().wait().unwrap();

    // Then the second task reuses all three pooled blocks and computes the right result
    assert_eq!(out2.to_vec().unwrap(), vec![5.0f32; len]);
    drop(out2);
    {
        let s = stats.lock().unwrap();
        assert_eq!(s.driver_allocs, 3);
        assert_eq!(s.pool_hits, 3);
        assert_eq!(s.freed_blocks.len(), 3);
    }
}
