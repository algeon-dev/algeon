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

pub(in crate::exec) use implementation::execute_graph_algorithm;
#[cfg(feature = "cugraph")]
pub(in crate::exec) use implementation::{
    adaptive_betweenness, diagnostics, execute_graph_algorithm_cache_hit, execute_graph_algorithm_sanitized, memory,
    resident_cache, statistics,
};
#[cfg(feature = "cugraph")]
pub use implementation::{
    force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test,
    force_next_graph_statistics_optional_resource_exhausted_for_test,
};
