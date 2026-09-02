//! Nexus native-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-nexus`.
//!
//! Public APIs fall into four groups:
//!
//! - [`plan`] and [`expr`] define the DataFusion-free query IR.
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

#[cfg(test)]
extern crate self as nexus_query_engine;

// These focused contract sources exercise crate-private reservation and frame
// ownership hooks.  They remain outside the production facade while compiling
// with the engine's unit-test crate.
#[cfg(test)]
#[path = "../tests/native_contract_tests/reservation_tracker_contracts.rs"]
mod reservation_tracker_contracts;
#[cfg(test)]
#[path = "execution_integration_tests/runtime_observability.rs"]
mod runtime_observability_integration_tests;
#[cfg(test)]
#[path = "../tests/native_contract_tests/yielded_frame.rs"]
mod yielded_frame;

pub use nexus_query_planning::admission;
/// Capability-analysis model and admission helpers for native execution.
pub use nexus_query_planning::capability;
pub use nexus_query_runtime::attempt;
pub mod chunk;
/// Arrow-to-cuDF datatype support and cast policy helpers.
pub use nexus_query_planning::datatypes;
/// DataFusion-free dynamic-filter producer/consumer route IR.
pub use nexus_query_planning::dynamic_filter;
pub use nexus_query_planning::{contract_report, error};
pub mod exec;
/// DataFusion-free expression IR used by native query plans.
pub use nexus_query_planning::expr;
/// Approximate cuGraph memory pricing outside bounded admission.
pub use nexus_query_planning::graph_memory;
pub(crate) use nexus_query_planning::{execution_contracts, expression_lowering_policy};
/// Stable helper primitives shared by report and benchmark harnesses.
pub mod harness;
/// Stable name-generation helpers shared across native planning code.
pub use nexus_query_planning::naming;
/// Shared Parquet footer/statistics helpers for native pruning and sizing.
pub use nexus_query_planning::parquet_stats;
/// DataFusion-free query plan IR and graph operators.
pub use nexus_query_planning::plan;
pub(crate) use nexus_query_planning::{materialization_sizing, prepared, util, validation};
pub use nexus_query_planning::{mode, optimizer, pipeline, source, triage};
pub(crate) use nexus_query_runtime::parquet_metadata;
pub use nexus_query_runtime::{metrics, observability, runtime};
