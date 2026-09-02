//! DataFusion-free native query runtime, lifecycle, ledger, and observation owners.

#[cfg(test)]
extern crate self as nexus_query_engine;

pub mod attempt;
pub mod metrics;
pub mod observability;
pub mod runtime;

pub(crate) use nexus_query_planning::{
    admission, capability, chunk, correlation, dynamic_filter, error, execution_contracts, graph_memory, mode, naming,
    plan, schedule_projection, source, triage, util,
};

pub(crate) mod harness {
    pub use nexus_query_planning::util::max_option;
}

#[doc(hidden)]
pub mod parquet_metadata;
mod triage_metrics;
