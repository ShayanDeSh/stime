use std::fmt;

use cudarc::driver::DriverError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The CUDA driver returned an error.
    Driver(DriverError),
    /// A host slice did not have the same length as the device buffer.
    LengthMismatch { expected: usize, actual: usize },
    /// A task argument lives on a different device than the stream the task was launched on.
    DeviceMismatch,
    /// A task that has already been launched was launched again.
    TaskAlreadyLaunched,
    /// A task was waited on before being launched.
    TaskNotLaunched,
    /// The worker thread exited before it could take or hand back a task.
    WorkerStopped,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Error::LengthMismatch { expected, actual } => write!(
                f,
                "length mismatch: buffer has {expected} elements, host slice has {actual}"
            ),
            Error::DeviceMismatch => write!(f, "argument is on a different device than the stream"),
            Error::TaskAlreadyLaunched => write!(f, "task has already been launched"),
            Error::TaskNotLaunched => write!(f, "task has not been launched"),
            Error::WorkerStopped => write!(f, "worker thread has stopped"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Driver(e) => Some(e),
            Error::LengthMismatch { .. }
            | Error::DeviceMismatch
            | Error::TaskAlreadyLaunched
            | Error::TaskNotLaunched
            | Error::WorkerStopped => None,
        }
    }
}

impl From<DriverError> for Error {
    fn from(e: DriverError) -> Self {
        Error::Driver(e)
    }
}
