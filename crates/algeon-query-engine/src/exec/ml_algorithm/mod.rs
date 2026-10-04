#[cfg(feature = "cuml")]
mod assembly;
#[cfg(feature = "cuml")]
mod enabled;

#[cfg(feature = "cuml")]
pub(in crate::exec) use enabled::{execute_ml_algorithm, validate_ml_algorithm_execution};
#[cfg(all(feature = "cuml", feature = "test-hooks"))]
pub use enabled::{set_forest_after_import_hook_for_tests, set_forest_batch_element_limit_for_tests};

#[cfg(not(feature = "cuml"))]
mod disabled;

#[cfg(not(feature = "cuml"))]
pub(in crate::exec) use disabled::ml_algorithm_execution_unavailable;
