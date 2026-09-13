use query_engine::plan::NodeId;

use super::*;

include!("graph_algorithm_execution/support.rs");

include!("graph_algorithm_execution/betweenness.rs");
include!("graph_algorithm_execution/bfs.rs");
include!("graph_algorithm_execution/cache.rs");
include!("graph_algorithm_execution/empty_inputs.rs");
include!("graph_algorithm_execution/reproducible_results.rs");
include!("graph_algorithm_execution/similarity.rs");
