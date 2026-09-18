//! Algeon native-rs query engine contracts.
//!
//! This crate does not depend on any query-engine adapter. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `algeon-datafusion`.
//!
//! Public APIs fall into four groups:
//!
//! - [`plan`] and [`expr`] define the native query IR.
//! - [`capability`], [`admission`], and [`pipeline`] prove and compile bounded-execution
//!   contracts without assigning plans to named execution families.
//! - [`runtime`] owns device, memory, boundary, and scheduler policy.
//! - [`exec`] executes a validated [`plan::QueryPlan`] and returns
//!   either host Arrow output or GPU-resident handles according to its
//!   [`plan::ResultSink`].
//!
//! Byte counts in native reports and runtime contracts are device bytes unless
//! explicitly labeled as host or Arrow bytes. Native ids and cache keys are
//! local identities for one plan or runtime; callers should not treat them as
//! stable global handles.
//!
//! The `cuvs` feature enables vector-algorithm execution; the `cuml` feature
//! enables query-local TSVD, HDBSCAN, UMAP, logistic-regression, and
//! random-forest execution. Their planning contracts remain
//! available when either feature is disabled.

#[cfg(test)]
extern crate self as query_engine;

#[cfg(test)]
#[path = "execution_integration_tests/runtime_observability.rs"]
mod runtime_observability_integration_tests;
#[cfg(test)]
#[path = "../tests/native_contract_tests/yielded_frame.rs"]
mod yielded_frame;

pub use query_planning::admission;
/// Backend capability contract plan validation asks the execution owner.
pub use query_planning::backend;
/// Capability-analysis model and admission helpers for native execution.
pub use query_planning::capability;
pub use query_runtime::attempt;
pub mod chunk;
/// Arrow-to-cuDF datatype support and cast policy helpers.
pub mod datatypes;
/// Local-filesystem implementation of the planning dimension-pruning resolver contract.
pub mod dimension_pruning;
/// Native dynamic-filter producer/consumer route IR.
pub use query_planning::dynamic_filter;
pub use query_planning::{contract_report, error};
pub mod exec;
/// Expression IR used by native query plans.
pub use query_planning::expr;
pub(crate) use query_planning::expression_lowering_policy;
/// Measured graph-execution observation rows.
pub use query_planning::graph_observation;
/// Stable helper primitives shared by report and benchmark harnesses.
pub mod harness;
/// Stable name-generation helpers shared across native planning code.
pub use query_planning::naming;
/// Native error projection: cuDF from the runtime boundary, cuVS from here.
pub mod native_error;
#[doc(hidden)]
pub mod object_store_error;
/// RAPIDS implementation of the planning backend capability contract.
pub mod rapids_backend;
/// Shared Parquet footer/statistics helpers for native pruning and sizing.
pub use query_planning::parquet_stats;
/// Native query plan IR and graph operators.
pub use query_planning::plan;
/// Backend materialized-output sizing, applying the component's formula.
#[doc(hidden)]
pub mod materialization_sizing;
pub use query_planning::{mode, optimizer, pipeline, source, triage, validate_query_plan};
/// Native cuDF expression, aggregate, and sort lowering.
pub(crate) mod prepared;
#[doc(hidden)]
pub use query_planning::{util, validation};
pub(crate) mod parquet_metadata;
pub use query_runtime::{metrics, observability, runtime};

pub mod shadow;
pub(crate) mod source_chunk;
