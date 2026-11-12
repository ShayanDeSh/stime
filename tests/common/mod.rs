// Each test binary uses only some of these items.
#![allow(dead_code)]

use stime::{Device, DeviceBuffer, LaunchConfig, Module, Task};

/// PTX for `add(const float* a, const float* b, float* out, unsigned n)`: `out[i] = a[i] + b[i]`.
pub const ADD_PTX: &std::ffi::CStr = cr#"
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

/// PTX for `fill(float* out, float value, unsigned n)`: `out[i] = value`.
pub const FILL_PTX: &std::ffi::CStr = cr#"
.version 7.0
.target sm_52
.address_size 64

.visible .entry fill(
	.param .u64 fill_param_0,
	.param .f32 fill_param_1,
	.param .u32 fill_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<5>;

	ld.param.u64 	%rd1, [fill_param_0];
	ld.param.f32 	%f1, [fill_param_1];
	ld.param.u32 	%r2, [fill_param_2];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.u32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.u32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f1;

$L__BB0_2:
	ret;
}
"#;

pub type AddArgs = (DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>, u32);
pub type FillArgs = (DeviceBuffer<f32>, f32, u32);

/// A launch shape covering `n` elements with 128-thread blocks.
pub fn config(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A task that adds `a` and `b` element-wise into a zeroed output buffer.
pub fn add_task(device: &Device, a: &[f32], b: &[f32]) -> Task<AddArgs> {
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
pub fn fill_task(device: &Device, len: usize, value: f32) -> Task<FillArgs> {
    let kernel = Module::from_ptx(device, FILL_PTX)
        .unwrap()
        .kernel(c"fill")
        .unwrap();
    let n = len as u32;
    let args = (DeviceBuffer::zeroed(device, len).unwrap(), value, n);
    Task::new(kernel, config(n), args)
}
