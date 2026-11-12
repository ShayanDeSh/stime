use std::sync::Arc;

use stime::{Device, DeviceBuffer, Error, LaunchConfig, Module, Stream, Task, Worker};

mod common;

use common::{ADD_PTX, FILL_PTX};

type AddArgs = (DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>, u32);
type FillArgs = (DeviceBuffer<f32>, f32, u32);

fn config(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A task that adds `a` and `b` element-wise into a zeroed output buffer.
fn add_task(device: &Device, a: &[f32], b: &[f32]) -> Task<AddArgs> {
    let kernel = Module::from_ptx(device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let n = a.len() as u32;
    let args = (
        DeviceBuffer::from_slice(device, a).unwrap(),
        DeviceBuffer::from_slice(device, b).unwrap(),
        DeviceBuffer::zeroed(device, a.len()).unwrap(),
        n,
    );
    Task::new(kernel, config(n), args)
}

/// A task that fills a zeroed buffer of `len` elements with `value`.
fn fill_task(device: &Device, len: usize, value: f32) -> Task<FillArgs> {
    let kernel = Module::from_ptx(device, FILL_PTX)
        .unwrap()
        .kernel(c"fill")
        .unwrap();
    let n = len as u32;
    let args = (DeviceBuffer::zeroed(device, len).unwrap(), value, n);
    Task::new(kernel, config(n), args)
}

fn worker(device: &Device, capacity: usize) -> Worker {
    Worker::new(Stream::new(device).unwrap(), capacity)
}

#[test]
fn submitted_task_returns_its_results() {
    // Given a worker and a task over two input buffers
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 4);
    let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1000).map(|i| 2.0 * i as f32).collect();
    let task = add_task(&device, &a, &b);

    // When submitting it and waiting on its handle
    // SAFETY: the arguments match `add`'s signature and live on the worker's device.
    let handle = unsafe { worker.submit(task) }.unwrap();
    let (_, _, out, _) = handle.wait().unwrap();

    // Then the returned output holds the element-wise sums
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn many_queued_tasks_each_return_their_own_results() {
    // Given a worker with a small queue
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 2);

    // When submitting far more tasks than the queue holds
    let handles: Vec<_> = (0..50)
        .map(|i| {
            let task = fill_task(&device, 1024, i as f32);
            // SAFETY: the arguments match `fill`'s signature and live on the worker's device.
            unsafe { worker.submit(task) }.unwrap()
        })
        .collect();

    // Then every handle receives the output of its own task
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 1024]);
    }
}

#[test]
fn tasks_with_different_arguments_share_a_worker() {
    // Given a worker
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 4);

    // When submitting tasks whose argument types differ
    // SAFETY: the arguments match each kernel's signature and live on the worker's device.
    let add = unsafe { worker.submit(add_task(&device, &[1.0, 2.0], &[3.0, 4.0])) }.unwrap();
    let fill = unsafe { worker.submit(fill_task(&device, 3, 7.0)) }.unwrap();

    // Then each handle returns its own argument type with the right results
    let (_, _, sums, _) = add.wait().unwrap();
    let (filled, _, _) = fill.wait().unwrap();
    assert_eq!(sums.to_vec().unwrap(), vec![4.0, 6.0]);
    assert_eq!(filled.to_vec().unwrap(), vec![7.0; 3]);
}

#[test]
fn zero_capacity_worker_runs_tasks() {
    // Given a worker whose queue holds no waiting tasks
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 0);

    // When submitting several tasks
    // SAFETY: the arguments match `fill`'s signature and live on the worker's device.
    let handles: Vec<_> = (0..5)
        .map(|i| unsafe { worker.submit(fill_task(&device, 16, i as f32)) }.unwrap())
        .collect();

    // Then all of them complete with the right results
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 16]);
    }
}

#[test]
fn tasks_can_be_submitted_from_many_threads() {
    // Given a worker shared between threads
    let device = Device::get(0).unwrap();
    let worker = Arc::new(worker(&device, 4));

    // When several threads submit tasks at once
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let worker = Arc::clone(&worker);
            std::thread::spawn(move || {
                let device = Device::get(0).unwrap();
                let handles: Vec<_> = (0..25)
                    .map(|i| {
                        let value = (t * 100 + i) as f32;
                        // SAFETY: the arguments match `fill`'s signature and live on the
                        // worker's device.
                        unsafe { worker.submit(fill_task(&device, 512, value)) }.unwrap()
                    })
                    .collect();
                for (i, handle) in handles.into_iter().enumerate() {
                    let (out, _, _) = handle.wait().unwrap();
                    assert_eq!(out.to_vec().unwrap(), vec![(t * 100 + i) as f32; 512]);
                }
            })
        })
        .collect();

    // Then every thread gets back its own results
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn handle_can_be_waited_on_from_another_thread() {
    // Given a submitted task
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 1);
    // SAFETY: the arguments match `add`'s signature and live on the worker's device.
    let handle = unsafe { worker.submit(add_task(&device, &[1.0], &[2.0])) }.unwrap();

    // When waiting on its handle from a different thread
    let (_, _, out, _) = std::thread::spawn(move || handle.wait())
        .join()
        .unwrap()
        .unwrap();

    // Then its results are handed back
    assert_eq!(out.to_vec().unwrap(), vec![3.0]);
}

#[test]
fn already_launched_task_reports_the_error_to_its_handle() {
    // Given a task that has already been launched elsewhere
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let mut task = add_task(&device, &[1.0], &[2.0]);
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { task.launch(&stream) }.unwrap();
    let worker = worker(&device, 1);

    // When submitting it to a worker
    // SAFETY: as above.
    let handle = unsafe { worker.submit(task) }.unwrap();

    // Then waiting on its handle reports the failed launch
    assert!(matches!(handle.wait(), Err(Error::TaskAlreadyLaunched)));
}

#[test]
fn failed_task_does_not_stop_later_tasks() {
    // Given a worker and a task that cannot be launched
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let mut launched = add_task(&device, &[1.0], &[2.0]);
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { launched.launch(&stream) }.unwrap();
    let worker = worker(&device, 4);
    // SAFETY: as above.
    let failed = unsafe { worker.submit(launched) }.unwrap();

    // When submitting another task behind it
    let next = unsafe { worker.submit(add_task(&device, &[5.0], &[6.0])) }.unwrap();

    // Then the later task still completes
    assert!(failed.wait().is_err());
    let (_, _, out, _) = next.wait().unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![11.0]);
}

#[test]
fn dropping_the_worker_hands_back_queued_tasks() {
    // Given a worker with tasks still queued or running
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 8);
    // SAFETY: the arguments match `fill`'s signature and live on the worker's device.
    let handles: Vec<_> = (0..8)
        .map(|i| unsafe { worker.submit(fill_task(&device, 4096, i as f32)) }.unwrap())
        .collect();

    // When dropping the worker
    drop(worker);

    // Then every handle still receives its results
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 4096]);
    }
}

#[test]
fn dropping_a_handle_does_not_stop_the_worker() {
    // Given a worker with a task whose handle is dropped before it finishes
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 4);
    // SAFETY: the arguments match `fill`'s signature and live on the worker's device.
    drop(unsafe { worker.submit(fill_task(&device, 1024, 1.0)) }.unwrap());

    // When submitting another task
    let handle = unsafe { worker.submit(fill_task(&device, 1024, 2.0)) }.unwrap();

    // Then the later task completes normally
    let (out, _, _) = handle.wait().unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![2.0; 1024]);
}

#[test]
fn fill_outputs_feed_add_tasks_in_a_pipeline() {
    // Given a worker and the `add` kernel loaded once for every round
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 4);
    let add = Module::from_ptx(&device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let rounds = 40;
    let len = 2048;
    let n = len as u32;

    // When every round's two `fill` tasks are submitted, then each round's filled buffers
    // are moved into an `add` task submitted to the same worker
    // SAFETY (all submits): the arguments match each kernel's signature and live on the
    // worker's device.
    let fills: Vec<_> = (0..rounds)
        .map(|i| {
            let x = unsafe { worker.submit(fill_task(&device, len, i as f32)) }.unwrap();
            let y = unsafe { worker.submit(fill_task(&device, len, (2 * i + 1) as f32)) }.unwrap();
            (x, y)
        })
        .collect();
    let adds: Vec<_> = fills
        .into_iter()
        .map(|(x, y)| {
            let (x, _, _) = x.wait().unwrap();
            let (y, _, _) = y.wait().unwrap();
            let out = DeviceBuffer::<f32>::zeroed(&device, len).unwrap();
            let task = Task::new(add.clone(), config(n), (x, y, out, n));
            unsafe { worker.submit(task) }.unwrap()
        })
        .collect();

    // Then each sum is the two values its round was filled with
    for (i, handle) in adds.into_iter().enumerate() {
        let (_, _, out, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![(3 * i + 1) as f32; len]);
    }
}

#[test]
fn buffer_passed_through_a_chain_of_tasks_accumulates() {
    // Given a worker, the `add` kernel, and a buffer of ones to add at every stage
    let device = Device::get(0).unwrap();
    let worker = worker(&device, 2);
    let add = Module::from_ptx(&device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let len = 1024;
    let n = len as u32;
    let stages = 25;
    let mut acc = DeviceBuffer::<f32>::zeroed(&device, len).unwrap();
    let mut ones = DeviceBuffer::from_slice(&device, &vec![1.0f32; len]).unwrap();

    // When each stage's output buffer becomes the next stage's accumulator input
    for _ in 0..stages {
        let out = DeviceBuffer::<f32>::zeroed(&device, len).unwrap();
        let task = Task::new(add.clone(), config(n), (acc, ones, out, n));
        // SAFETY: the arguments match `add`'s signature and live on the worker's device.
        let handle = unsafe { worker.submit(task) }.unwrap();
        let (_, returned_ones, out, _) = handle.wait().unwrap();
        acc = out;
        ones = returned_ones;
    }

    // Then the buffer holds the number of stages in every element
    assert_eq!(acc.to_vec().unwrap(), vec![stages as f32; len]);
}
