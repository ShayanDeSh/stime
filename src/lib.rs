mod device;

pub use cudarc::driver::DriverError as Error;
pub use device::Device;

pub type Result<T> = std::result::Result<T, Error>;
