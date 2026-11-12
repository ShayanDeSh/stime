use stime::{Device, Error, Stream, TaskState};

mod common;

use common::add_task;

#[test]
fn new_task_is_pending() {
    // Given a task that has not been launched
    let device = Device::get(0).unwrap();
    let task = add_task(&device, &[1.0], &[2.0]);

    // When querying its state
    let state = task.state().unwrap();

    // Then it is pending
    assert_eq!(state, TaskState::Pending);
}

#[test]
fn launched_task_returns_its_results() {
    // Given a task over two input buffers
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1000).map(|i| 2.0 * i as f32).collect();
    let mut task = add_task(&device, &a, &b);

    // When launching it and waiting for it
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { task.launch(&stream) }.unwrap();
    let (_, _, out, _) = task.wait().unwrap();

    // Then the returned output holds the element-wise sums
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn launched_task_completes() {
    // Given a launched task
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let mut task = add_task(&device, &[1.0], &[2.0]);
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { task.launch(&stream) }.unwrap();

    // When the stream has finished its work
    stream.synchronize().unwrap();

    // Then the task reports completed
    assert_eq!(task.state().unwrap(), TaskState::Completed);
}

#[test]
fn launching_twice_fails() {
    // Given a task that has already been launched
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let mut task = add_task(&device, &[1.0], &[2.0]);
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { task.launch(&stream) }.unwrap();

    // When launching it again
    // SAFETY: as above.
    let result = unsafe { task.launch(&stream) };

    // Then launching fails
    assert_eq!(result, Err(Error::TaskAlreadyLaunched));
}

#[test]
fn waiting_on_unlaunched_task_fails() {
    // Given a task that has not been launched
    let device = Device::get(0).unwrap();
    let task = add_task(&device, &[1.0], &[2.0]);

    // When waiting on it
    let result = task.wait();

    // Then waiting fails
    assert!(matches!(result, Err(Error::TaskNotLaunched)));
}

#[test]
fn task_can_be_waited_on_from_another_thread() {
    // Given a task launched on this thread
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let mut task = add_task(&device, &[1.0], &[2.0]);
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { task.launch(&stream) }.unwrap();

    // When waiting on it from a different thread
    let (_, _, out, _) = std::thread::spawn(move || task.wait())
        .join()
        .unwrap()
        .unwrap();

    // Then its results are handed back
    assert_eq!(out.to_vec().unwrap(), vec![3.0]);
}
