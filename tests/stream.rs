use stime::{Device, Stream};

#[test]
fn new_stream_can_be_created_and_synchronized() {
    // Given a device
    let device = Device::get(0).unwrap();

    // When creating a stream and synchronizing it with no queued work
    let stream = Stream::new(&device).unwrap();
    let result = stream.synchronize();

    // Then synchronizing succeeds
    assert!(result.is_ok());
}

#[test]
fn multiple_streams_can_coexist_on_one_device() {
    // Given a device
    let device = Device::get(0).unwrap();

    // When creating several streams
    let streams: Vec<Stream> = (0..4).map(|_| Stream::new(&device).unwrap()).collect();

    // Then each one can be synchronized
    for stream in &streams {
        assert!(stream.synchronize().is_ok());
    }
}

#[test]
fn stream_can_be_used_from_another_thread() {
    // Given a stream created on this thread
    let stream = Stream::new(&Device::get(0).unwrap()).unwrap();

    // When synchronizing it from a different thread
    let result = std::thread::spawn(move || stream.synchronize())
        .join()
        .unwrap();

    // Then synchronizing succeeds
    assert!(result.is_ok());
}
