//! DataFusion-free native query runtime, lifecycle, ledger, and observation owners.

#[cfg(test)]
extern crate self as nexus_query_engine;

pub mod attempt;
pub mod metrics;
pub mod observability;
pub mod runtime;

pub(crate) mod harness {
    pub use nexus_query_planning::util::max_option;
}

#[doc(hidden)]
pub mod parquet_metadata;
mod triage_metrics;
