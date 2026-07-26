//! Vector-algorithm execution boundary.
//!
//! The feature-enabled path runs exact kNN through the runtime-bound safe cuVS
//! crate. The disabled implementation is intentionally GPU-free.

#[cfg(feature = "cuvs")]
mod assembly;
#[cfg(not(feature = "cuvs"))]
mod disabled;
#[cfg(feature = "cuvs")]
mod enabled;

#[cfg(not(feature = "cuvs"))]
pub(in crate::exec) use disabled::execute_vector_algorithm;
#[cfg(feature = "cuvs")]
pub(in crate::exec) use enabled::{execute_vector_algorithm, validate_vector_algorithm_execution};
