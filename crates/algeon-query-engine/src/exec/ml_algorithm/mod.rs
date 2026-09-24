#[cfg(feature = "cuml")]
mod assembly;
#[cfg(feature = "cuml")]
mod enabled;

#[cfg(feature = "cuml")]
pub(in crate::exec) use enabled::{execute_ml_algorithm, validate_ml_algorithm_execution};

#[cfg(not(feature = "cuml"))]
mod disabled;

#[cfg(not(feature = "cuml"))]
pub(in crate::exec) use disabled::ml_algorithm_execution_unavailable;
