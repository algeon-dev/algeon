use super::*;

#[path = "bounded_joins/aggregate_pipelines.rs"]
mod aggregate_pipelines;
// The cache-backed multi-join scenario is retained as private P5 substrate.
#[path = "bounded_joins/root_joins.rs"]
mod root_joins;
