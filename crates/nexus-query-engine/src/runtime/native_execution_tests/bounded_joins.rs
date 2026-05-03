use super::*;

#[path = "bounded_joins/aggregate_pipelines.rs"]
mod aggregate_pipelines;
// The cache-backed multi-join scenario belongs to its dedicated cache suite.
#[path = "bounded_joins/root_joins.rs"]
mod root_joins;
