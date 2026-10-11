//! Vector-algorithm execution boundary.
//!
//! The feature-enabled path runs the vector algorithms (kNN, pairwise distance,
//! Top-K, K-means and PCA) through the runtime-bound safe cuVS crate. The disabled
//! implementation is intentionally GPU-free.

#[cfg(feature = "cuvs")]
mod assembly;
#[cfg(not(feature = "cuvs"))]
mod disabled;
#[cfg(feature = "cuvs")]
mod enabled;

#[cfg(not(feature = "cuvs"))]
pub(in crate::exec) use disabled::vector_algorithm_execution_unavailable;
#[cfg(feature = "cuvs")]
pub(in crate::exec) use enabled::{execute_vector_algorithm, validate_vector_algorithm_execution};
#[cfg(all(feature = "cuvs", feature = "test-hooks"))]
pub use enabled::{set_vector_after_index_build_hook_for_tests, set_vector_pool_coverage_hook_for_tests};
