use super::*;

include!("graph_algorithm_execution/support.rs");
// Durable graph-cache scenarios are P5 substrate, not P2 validation.  P2
// graph tests execute with cache injection disabled.
include!("graph_algorithm_execution/memory_and_statistics.rs");
include!("graph_algorithm_execution/betweenness.rs");
include!("graph_algorithm_execution/bfs.rs");
include!("graph_algorithm_execution/similarity.rs");
