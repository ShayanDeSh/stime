use stime::{Device, Event, Stream};

#[test]
fn unrecorded_event_is_complete() {
    // Given a freshly created event
    let device = Device::get(0).unwrap();
    let event = Event::new(&device).unwrap();

    // When querying whether it is complete
    let complete = event.is_complete();

    // Then it reports complete
    assert!(complete.unwrap());
}

#[test]
fn event_recorded_on_idle_stream_completes() {
    // Given an event recorded on a stream with no queued work
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let event = Event::new(&device).unwrap();
    event.record(&stream).unwrap();

    // When synchronizing the event
    event.synchronize().unwrap();

    // Then it reports complete
    assert!(event.is_complete().unwrap());
}

#[test]
fn event_can_be_recorded_repeatedly() {
    // Given an event that has already been recorded and completed
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let event = Event::new(&device).unwrap();
    event.record(&stream).unwrap();
    event.synchronize().unwrap();

    // When recording it again
    let result = event.record(&stream);

    // Then recording succeeds and the event completes again
    assert!(result.is_ok());
    event.synchronize().unwrap();
    assert!(event.is_complete().unwrap());
}

#[test]
fn elapsed_time_between_recorded_events_is_non_negative() {
    // Given two events recorded in order on a stream
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let start = Event::new(&device).unwrap();
    let end = Event::new(&device).unwrap();
    start.record(&stream).unwrap();
    end.record(&stream).unwrap();

    // When measuring the time between them
    let elapsed = start.elapsed_ms(&end).unwrap();

    // Then the elapsed time is not negative
    assert!(elapsed >= 0.0);
}

#[test]
fn elapsed_time_with_unrecorded_event_fails() {
    // Given one recorded event and one that was never recorded
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let start = Event::new(&device).unwrap();
    let end = Event::new(&device).unwrap();
    start.record(&stream).unwrap();

    // When measuring the time between them
    let result = start.elapsed_ms(&end);

    // Then measuring fails
    assert!(result.is_err());
}

#[test]
fn stream_can_wait_on_event_from_another_stream() {
    // Given an event recorded on one stream
    let device = Device::get(0).unwrap();
    let producer = Stream::new(&device).unwrap();
    let consumer = Stream::new(&device).unwrap();
    let event = Event::new(&device).unwrap();
    event.record(&producer).unwrap();

    // When another stream waits on the event
    let result = consumer.wait_event(&event);

    // Then waiting succeeds and the waiting stream can be synchronized
    assert!(result.is_ok());
    assert!(consumer.synchronize().is_ok());
}

#[test]
fn event_can_be_used_from_another_thread() {
    // Given an event recorded on this thread
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let event = Event::new(&device).unwrap();
    event.record(&stream).unwrap();

    // When synchronizing it from a different thread
    let result = std::thread::spawn(move || event.synchronize())
        .join()
        .unwrap();

    // Then synchronizing succeeds
    assert!(result.is_ok());
}
