use std::num::NonZeroUsize;
use std::sync::Arc;

use stime::{Device, DeviceBuffer, Error, Module, Runtime, Stream, Task};

mod common;

use common::{ADD_PTX, add_task, config, fill_task};

fn runtime(device: &Device, workers: usize, capacity: usize) -> Runtime {
    Runtime::new(device, NonZeroUsize::new(workers).unwrap(), capacity).unwrap()
}

#[test]
fn reports_the_number_of_workers_it_was_created_with() {
    // Given a device
    let device = Device::get(0).unwrap();

    // When creating runtimes with different worker counts
    // Then each reports the count it was created with
    for workers in [1, 3, 8] {
        assert_eq!(runtime(&device, workers, 4).worker_count(), workers);
    }
}

#[test]
fn submitted_task_returns_its_results() {
    // Given a runtime and a task over two input buffers
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 3, 4);
    let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1000).map(|i| 2.0 * i as f32).collect();
    let task = add_task(&device, &a, &b);

    // When submitting it and waiting on its handle
    // SAFETY: the arguments match `add`'s signature and live on the runtime's device.
    let handle = unsafe { runtime.submit(task) }.unwrap();
    let (_, _, out, _) = handle.wait().unwrap();

    // Then the returned output holds the element-wise sums
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(out.to_vec().unwrap(), expected);
}

#[test]
fn many_tasks_spread_over_workers_each_return_their_own_results() {
    // Given a runtime with several workers and small queues
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 4, 2);

    // When submitting far more tasks than the workers' queues hold
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let task = fill_task(&device, 1024, i as f32);
            // SAFETY: the arguments match `fill`'s signature and live on the runtime's device.
            unsafe { runtime.submit(task) }.unwrap()
        })
        .collect();

    // Then every handle receives the output of its own task
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 1024]);
    }
}

#[test]
fn single_worker_runtime_runs_tasks() {
    // Given a runtime with one worker
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 1, 4);

    // When submitting several tasks
    // SAFETY: the arguments match `fill`'s signature and live on the runtime's device.
    let handles: Vec<_> = (0..10)
        .map(|i| unsafe { runtime.submit(fill_task(&device, 64, i as f32)) }.unwrap())
        .collect();

    // Then all of them complete with the right results
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 64]);
    }
}

#[test]
fn zero_capacity_workers_run_tasks() {
    // Given a runtime whose workers' queues hold no waiting tasks
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 3, 0);

    // When submitting several tasks
    // SAFETY: the arguments match `fill`'s signature and live on the runtime's device.
    let handles: Vec<_> = (0..9)
        .map(|i| unsafe { runtime.submit(fill_task(&device, 16, i as f32)) }.unwrap())
        .collect();

    // Then all of them complete with the right results
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 16]);
    }
}

#[test]
fn tasks_with_different_arguments_share_a_runtime() {
    // Given a runtime
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 2, 4);

    // When submitting tasks whose argument types differ
    // SAFETY: the arguments match each kernel's signature and live on the runtime's device.
    let add = unsafe { runtime.submit(add_task(&device, &[1.0, 2.0], &[3.0, 4.0])) }.unwrap();
    let fill = unsafe { runtime.submit(fill_task(&device, 3, 7.0)) }.unwrap();

    // Then each handle returns its own argument type with the right results
    let (_, _, sums, _) = add.wait().unwrap();
    let (filled, _, _) = fill.wait().unwrap();
    assert_eq!(sums.to_vec().unwrap(), vec![4.0, 6.0]);
    assert_eq!(filled.to_vec().unwrap(), vec![7.0; 3]);
}

#[test]
fn tasks_can_be_submitted_from_many_threads() {
    // Given a runtime shared between threads
    let device = Device::get(0).unwrap();
    let runtime = Arc::new(runtime(&device, 3, 4));

    // When several threads submit tasks at once
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let runtime = Arc::clone(&runtime);
            std::thread::spawn(move || {
                let device = Device::get(0).unwrap();
                let handles: Vec<_> = (0..25)
                    .map(|i| {
                        let value = (t * 100 + i) as f32;
                        // SAFETY: the arguments match `fill`'s signature and live on the
                        // runtime's device.
                        unsafe { runtime.submit(fill_task(&device, 512, value)) }.unwrap()
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
    let runtime = runtime(&device, 2, 1);
    // SAFETY: the arguments match `add`'s signature and live on the runtime's device.
    let handle = unsafe { runtime.submit(add_task(&device, &[1.0], &[2.0])) }.unwrap();

    // When waiting on its handle from a different thread
    let (_, _, out, _) = std::thread::spawn(move || handle.wait())
        .join()
        .unwrap()
        .unwrap();

    // Then its results are handed back
    assert_eq!(out.to_vec().unwrap(), vec![3.0]);
}

#[test]
fn failed_task_does_not_stop_later_tasks() {
    // Given a runtime and a task that cannot be launched
    let device = Device::get(0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let mut launched = add_task(&device, &[1.0], &[2.0]);
    // SAFETY: the arguments match `add`'s signature and live on the stream's device.
    unsafe { launched.launch(&stream) }.unwrap();
    let runtime = runtime(&device, 2, 4);
    // SAFETY: as above.
    let failed = unsafe { runtime.submit(launched) }.unwrap();

    // When submitting more tasks behind it, enough to reach every worker
    let next: Vec<_> = (0..4)
        .map(|i| {
            let task = add_task(&device, &[i as f32], &[1.0]);
            unsafe { runtime.submit(task) }.unwrap()
        })
        .collect();

    // Then the failure is reported to its own handle and every later task still completes
    assert!(matches!(failed.wait(), Err(Error::TaskAlreadyLaunched)));
    for (i, handle) in next.into_iter().enumerate() {
        let (_, _, out, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32 + 1.0]);
    }
}

#[test]
fn dropping_the_runtime_hands_back_queued_tasks() {
    // Given a runtime with tasks still queued or running on its workers
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 3, 8);
    // SAFETY: the arguments match `fill`'s signature and live on the runtime's device.
    let handles: Vec<_> = (0..24)
        .map(|i| unsafe { runtime.submit(fill_task(&device, 4096, i as f32)) }.unwrap())
        .collect();

    // When dropping the runtime
    drop(runtime);

    // Then every handle still receives its results
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; 4096]);
    }
}

#[test]
fn dropping_a_handle_does_not_stop_the_runtime() {
    // Given a runtime with a task whose handle is dropped before it finishes
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 2, 4);
    // SAFETY: the arguments match `fill`'s signature and live on the runtime's device.
    drop(unsafe { runtime.submit(fill_task(&device, 1024, 1.0)) }.unwrap());

    // When submitting more tasks, enough to reach every worker
    let handles: Vec<_> = (0..4)
        .map(|i| unsafe { runtime.submit(fill_task(&device, 1024, 2.0 + i as f32)) }.unwrap())
        .collect();

    // Then the later tasks complete normally
    for (i, handle) in handles.into_iter().enumerate() {
        let (out, _, _) = handle.wait().unwrap();
        assert_eq!(out.to_vec().unwrap(), vec![2.0 + i as f32; 1024]);
    }
}

#[test]
fn fill_outputs_feed_add_tasks_across_workers() {
    // Given a runtime and the `add` kernel loaded once for every round
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 3, 4);
    let add = Module::from_ptx(&device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let rounds = 30;
    let len = 2048;
    let n = len as u32;

    // When every round's two `fill` tasks are submitted, then each round's filled buffers
    // are moved into an `add` task submitted to the runtime
    // SAFETY (all submits): the arguments match each kernel's signature and live on the
    // runtime's device.
    let fills: Vec<_> = (0..rounds)
        .map(|i| {
            let x = unsafe { runtime.submit(fill_task(&device, len, i as f32)) }.unwrap();
            let y = unsafe { runtime.submit(fill_task(&device, len, (2 * i + 1) as f32)) }.unwrap();
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
            unsafe { runtime.submit(task) }.unwrap()
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
    // Given a runtime, the `add` kernel, and a buffer of ones to add at every stage
    let device = Device::get(0).unwrap();
    let runtime = runtime(&device, 3, 2);
    let add = Module::from_ptx(&device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let len = 1024;
    let n = len as u32;
    let stages = 25;
    let mut acc = DeviceBuffer::<f32>::zeroed(&device, len).unwrap();
    let mut ones = DeviceBuffer::from_slice(&device, &vec![1.0f32; len]).unwrap();

    // When each stage's output buffer becomes the next stage's accumulator input, so the
    // stages land on different workers in turn
    for _ in 0..stages {
        let out = DeviceBuffer::<f32>::zeroed(&device, len).unwrap();
        let task = Task::new(add.clone(), config(n), (acc, ones, out, n));
        // SAFETY: the arguments match `add`'s signature and live on the runtime's device.
        let handle = unsafe { runtime.submit(task) }.unwrap();
        let (_, returned_ones, out, _) = handle.wait().unwrap();
        acc = out;
        ones = returned_ones;
    }

    // Then the buffer holds the number of stages in every element
    assert_eq!(acc.to_vec().unwrap(), vec![stages as f32; len]);
}
