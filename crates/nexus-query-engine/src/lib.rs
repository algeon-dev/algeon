//! Native cudf-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-nexus`.
//!
//! Public APIs fall into four groups:
//!
//! - [`plan`] and [`expr`] define the DataFusion-free query IR.
//! - [`capability`] and [`pipeline`] prove and compile bounded-execution
//!   contracts without naming legacy execution families.
//! - [`runtime`] owns device, memory, boundary, and scheduler policy.
//! - [`exec`] executes a validated [`plan::QueryPlan`] and returns
//!   either host Arrow output or GPU-resident handles according to its
//!   [`plan::ResultSink`].
//!
//! Byte counts in native reports and runtime contracts are device bytes unless
//! explicitly labeled as host or Arrow bytes. Native ids and cache keys are
//! local identities for one plan or runtime; callers should not treat them as
//! stable global handles.

pub mod adaptive_report;
pub mod adaptive_runtime;
pub(crate) mod bounded;
/// Capability-analysis model and admission helpers for native execution.
pub mod capability;
pub mod chunk;
pub mod contract_report;
/// Arrow-to-cuDF datatype support and cast policy helpers.
pub mod datatypes;
/// DataFusion-free dynamic-filter producer/consumer route IR.
pub mod dynamic_filter;
pub mod error;
pub mod exec;
/// DataFusion-free expression IR used by native query plans.
pub mod expr;
/// Approximate cuGraph memory pricing outside strict bounded admission.
pub mod graph_memory;
/// Stable helper primitives shared by report and benchmark harnesses.
pub mod harness;
/// Runtime metrics, counters, and memory-reservation event records.
pub mod metrics;
pub mod mode;
/// Stable name-generation helpers shared across native planning code.
pub mod naming;
pub mod optimizer;
pub(crate) mod parquet_metadata;
/// Shared Parquet footer/statistics helpers for native pruning and sizing.
pub mod parquet_stats;
pub mod pipeline;
/// DataFusion-free query plan IR and graph operators.
pub mod plan;
pub mod runtime;
pub mod source;
pub mod triage;
pub(crate) mod util;
