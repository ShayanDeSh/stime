use std::ffi::{CStr, c_void};
use std::sync::Arc;

use cudarc::driver::sys::{CUfunction, CUmodule};
use cudarc::driver::{CudaContext, DeviceRepr, result};

use crate::{Device, DeviceBuffer, Result, Stream};

struct ModuleInner {
    ctx: Arc<CudaContext>,
    raw: CUmodule,
}

// SAFETY: driver module handles may be used from any thread once the context is bound.
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        // Errors cannot be reported from `drop`; the driver reclaims the module with the context.
        if self.ctx.bind_to_thread().is_ok() {
            let _ = unsafe { result::module::unload(self.raw) };
        }
    }
}

/// A compiled GPU module loaded onto a device, from which kernels are looked up.
///
/// The module is unloaded once it and every [`Kernel`] obtained from it are dropped.
#[derive(Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

impl Module {
    /// Loads a module from NUL-terminated PTX text onto `device`.
    pub fn from_ptx(device: &Device, ptx: &CStr) -> Result<Self> {
        let ctx = CudaContext::new(device.ordinal())?;
        ctx.bind_to_thread()?;
        // SAFETY: `ptx` is a valid NUL-terminated string, which the driver accepts as PTX text.
        let raw = unsafe { result::module::load_data(ptx.as_ptr().cast()) }?;
        Ok(Self {
            inner: Arc::new(ModuleInner { ctx, raw }),
        })
    }

    /// Looks up the kernel called `name` in the module.
    pub fn kernel(&self, name: &CStr) -> Result<Kernel> {
        self.inner.ctx.bind_to_thread()?;
        // SAFETY: `self.inner.raw` is a live module.
        let raw = unsafe { result::module::get_function(self.inner.raw, name.to_owned()) }?;
        Ok(Kernel {
            raw,
            module: self.clone(),
        })
    }
}

/// Grid and block dimensions for a kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Number of blocks in each dimension.
    pub grid_dim: (u32, u32, u32),
    /// Number of threads per block in each dimension.
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory per block, in bytes.
    pub shared_mem_bytes: u32,
}

/// A value that can be passed as a kernel parameter.
///
/// Implemented for [`DeviceBuffer`] (passed as a device pointer) and for plain scalar values.
pub trait KernelArg {
    /// Pointer to the value of the parameter, as expected by `cuLaunchKernel`.
    fn as_param(&self) -> *mut c_void;
}

impl<T: DeviceRepr> KernelArg for DeviceBuffer<T> {
    fn as_param(&self) -> *mut c_void {
        self.device_ptr_ref().cast_mut().cast()
    }
}

impl<T: DeviceRepr> KernelArg for T {
    fn as_param(&self) -> *mut c_void {
        std::ptr::from_ref(self).cast_mut().cast()
    }
}

/// A function in a [`Module`] that can be launched on a [`Stream`].
#[derive(Clone)]
pub struct Kernel {
    raw: CUfunction,
    /// Keeps the module loaded for as long as the kernel exists.
    module: Module,
}

// SAFETY: driver function handles may be used from any thread once the context is bound.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Queues the kernel on `stream` and returns without waiting for it to finish.
    ///
    /// # Safety
    /// - `args` must match the kernel's parameter list in number, order and type.
    /// - Every buffer in `args` must live on the stream's device and stay alive until the stream
    ///   has been synchronized. The kernel may read and write any of them.
    /// - Results must not be read back until the stream has been synchronized.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        config: LaunchConfig,
        args: &[&dyn KernelArg],
    ) -> Result<()> {
        self.module.inner.ctx.bind_to_thread()?;
        let mut params: Vec<*mut c_void> = args.iter().map(|arg| arg.as_param()).collect();
        // SAFETY: `self.raw` belongs to a live module; the caller upholds the argument contract.
        Ok(unsafe {
            result::launch_kernel(
                self.raw,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream.raw(),
                &mut params,
            )
        }?)
    }
}
