//! Feature-selected graph-algorithm execution entrypoints.
//!
//! The public-to-`exec` boundary remains available without cuGraph so generic
//! plan dispatch can return a structured unsupported-operator error.  The
//! full cuGraph implementation is compiled only when its optional dependency
//! is enabled.

#[cfg(feature = "cugraph")]
#[path = "cugraph.rs"]
mod implementation;

#[cfg(not(feature = "cugraph"))]
#[path = "disabled.rs"]
mod implementation;

#[cfg(all(test, feature = "cugraph"))]
pub(crate) use implementation::force_next_graph_statistics_optional_resource_exhausted_for_test;
#[cfg(not(feature = "cugraph"))]
pub(in crate::exec) use implementation::graph_algorithm_feature_disabled;
#[cfg(feature = "cugraph")]
pub(in crate::exec) use implementation::{
    bounded_betweenness, diagnostics, execute_graph_algorithm, execute_graph_algorithm_sanitized,
    graph_empty_input_prevalidated, memory, statistics,
};
#[cfg(all(test, feature = "cugraph"))]
pub(crate) use implementation::{
    force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test,
};
