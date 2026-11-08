use stime::{Device, DeviceBuffer, Error};

fn device() -> Device {
    Device::get(0).unwrap()
}

#[test]
fn from_slice_round_trips_data() {
    // Given host data
    let data = [1.0f32, 2.0, 3.0, 4.0];

    // When copying it to the device and back
    let buffer = DeviceBuffer::from_slice(&device(), &data).unwrap();
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
    let buffer = DeviceBuffer::from_slice(&device(), &[10u8, 20, 30]).unwrap();

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
    let buffer = DeviceBuffer::from_slice(&device(), &[1u32, 2, 3]).unwrap();

    // When reading it back from a different thread
    let result = std::thread::spawn(move || buffer.to_vec().unwrap())
        .join()
        .unwrap();

    // Then the data is intact
    assert_eq!(result, [1, 2, 3]);
}
