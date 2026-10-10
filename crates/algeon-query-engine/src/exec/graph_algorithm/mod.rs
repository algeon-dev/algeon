//! Graph-algorithm execution entrypoints.
//!
//! The cuGraph implementation is compiled only when its optional dependency is
//! enabled. A build without it rejects graph tasks in the compiled executor.

#[cfg(feature = "cugraph")]
mod enabled;

#[cfg(feature = "cugraph")]
pub(in crate::exec) use enabled::{
    bounded_betweenness, execute_graph_algorithm_sanitized, graph_empty_input_prevalidated, statistics,
};
