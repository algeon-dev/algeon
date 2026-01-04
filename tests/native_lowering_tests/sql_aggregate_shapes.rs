use super::*;

#[path = "sql_aggregate_shapes/default_distributed.rs"]
mod default_distributed;
#[path = "sql_aggregate_shapes/direct_global.rs"]
mod direct_global;
#[path = "sql_aggregate_shapes/direct_grouped.rs"]
mod direct_grouped;
#[path = "sql_aggregate_shapes/pruning.rs"]
mod pruning;
