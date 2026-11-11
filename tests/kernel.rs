use stime::{Device, DeviceBuffer, LaunchConfig, Module, Stream};

mod common;

use common::ADD_PTX;

#[test]
fn kernel_adds_two_buffers() {
    // Given a device, a loaded kernel and two input buffers
    let device = Device::get(0).unwrap();
    let kernel = Module::from_ptx(&device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let stream = Stream::new(&device).unwrap();
    let n = 1000u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
    let a_dev = DeviceBuffer::from_slice(&device, &a).unwrap();
    let b_dev = DeviceBuffer::from_slice(&device, &b).unwrap();
    let out_dev = DeviceBuffer::<f32>::zeroed(&device, n as usize).unwrap();

    // When launching the kernel and waiting for it to finish
    let config = LaunchConfig {
        grid_dim: (n.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the arguments match `add`'s signature and outlive the synchronized launch.
    unsafe { kernel.launch(&stream, config, &[&a_dev, &b_dev, &out_dev, &n]) }.unwrap();
    stream.synchronize().unwrap();

    // Then the output holds the element-wise sums
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(out_dev.to_vec().unwrap(), expected);
}

#[test]
fn loading_invalid_ptx_fails() {
    // Given a device
    let device = Device::get(0).unwrap();

    // When loading text that is not PTX
    let result = Module::from_ptx(&device, c"not ptx");

    // Then loading fails
    assert!(result.is_err());
}

#[test]
fn looking_up_missing_kernel_fails() {
    // Given a loaded module
    let device = Device::get(0).unwrap();
    let module = Module::from_ptx(&device, ADD_PTX).unwrap();

    // When looking up a kernel that does not exist
    let result = module.kernel(c"missing");

    // Then the lookup fails
    assert!(result.is_err());
}

#[test]
fn kernel_outlives_its_module_handle() {
    // Given a kernel whose module handle has been dropped
    let device = Device::get(0).unwrap();
    let kernel = Module::from_ptx(&device, ADD_PTX)
        .unwrap()
        .kernel(c"add")
        .unwrap();
    let stream = Stream::new(&device).unwrap();
    let a = DeviceBuffer::from_slice(&device, &[1.0f32]).unwrap();
    let b = DeviceBuffer::from_slice(&device, &[2.0f32]).unwrap();
    let out = DeviceBuffer::<f32>::zeroed(&device, 1).unwrap();

    // When launching it
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the arguments match `add`'s signature and outlive the synchronized launch.
    unsafe { kernel.launch(&stream, config, &[&a, &b, &out, &1u32]) }.unwrap();
    stream.synchronize().unwrap();

    // Then it still runs
    assert_eq!(out.to_vec().unwrap(), vec![3.0]);
}
