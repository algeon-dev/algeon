//! DataFusion-free native query planning, proof, and admission contracts.

pub mod admission;
pub mod capability;
pub mod chunk;
pub mod contract_report;
#[doc(hidden)]
pub mod correlation;
pub mod datatypes;
pub mod dynamic_filter;
pub mod error;
#[doc(hidden)]
pub mod execution_contracts;
pub mod expr;
#[doc(hidden)]
pub mod expression_lowering_policy;
pub mod graph_observation;
#[doc(hidden)]
pub mod materialization_sizing;
pub mod mode;
pub mod naming;
pub mod optimizer;
pub mod parquet_stats;
pub mod pipeline;
pub mod plan;
#[doc(hidden)]
pub mod prepared;
#[doc(hidden)]
pub mod schedule_projection;
pub mod source;
pub mod triage;
#[doc(hidden)]
pub mod util;
#[doc(hidden)]
pub mod validation;
mod vector_validation;

pub use validation::validate_query_plan;
