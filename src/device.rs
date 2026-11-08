use std::sync::OnceLock;

use cudarc::driver::result;
use cudarc::driver::sys::{CUdevice, CUdevice_attribute};

use crate::{Error, Result};

/// Calls `cuInit` once per process.
fn init() -> Result<()> {
    static INIT: OnceLock<Result<()>> = OnceLock::new();
    *INIT.get_or_init(|| result::init().map_err(Error::from))
}

/// A CUDA-capable device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    raw: CUdevice,
}

impl Device {
    /// Number of CUDA devices in the system.
    pub fn count() -> Result<usize> {
        init()?;
        Ok(result::device::get_count()? as usize)
    }

    /// Returns the device with the given ordinal.
    pub fn get(ordinal: usize) -> Result<Self> {
        init()?;
        Ok(Self {
            raw: result::device::get(ordinal as i32)?,
        })
    }

    /// Returns all devices in the system.
    pub fn all() -> Result<Vec<Self>> {
        (0..Self::count()?).map(Self::get).collect()
    }

    /// Device ordinal as used by the driver.
    pub fn ordinal(&self) -> usize {
        self.raw as usize
    }

    pub fn name(&self) -> Result<String> {
        Ok(result::device::get_name(self.raw)?)
    }

    /// Total device memory in bytes.
    pub fn total_memory(&self) -> Result<usize> {
        // SAFETY: `self.raw` was returned by `result::device::get`.
        Ok(unsafe { result::device::total_mem(self.raw) }?)
    }

    /// Compute capability as `(major, minor)`.
    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        Ok((
            self.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            self.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ))
    }

    fn attribute(&self, attrib: CUdevice_attribute) -> Result<u32> {
        // SAFETY: `self.raw` was returned by `result::device::get`.
        Ok(unsafe { result::device::get_attribute(self.raw, attrib) }? as u32)
    }
}
