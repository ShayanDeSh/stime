use stime::{Device, DeviceBuffer, LaunchConfig, Module, Stream};

/// PTX for `add(const float* a, const float* b, float* out, unsigned n)`: `out[i] = a[i] + b[i]`.
const ADD_PTX: &std::ffi::CStr = cr#"
.version 7.0
.target sm_52
.address_size 64

.visible .entry add(
	.param .u64 add_param_0,
	.param .u64 add_param_1,
	.param .u64 add_param_2,
	.param .u32 add_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;

	ld.param.u64 	%rd1, [add_param_0];
	ld.param.u64 	%rd2, [add_param_1];
	ld.param.u64 	%rd3, [add_param_2];
	ld.param.u32 	%r2, [add_param_3];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.u32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd1;
	mul.wide.u32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd2;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.f32 	%f1, [%rd8];
	ld.global.f32 	%f2, [%rd6];
	add.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd5;
	st.global.f32 	[%rd10], %f3;

$L__BB0_2:
	ret;
}
"#;

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
