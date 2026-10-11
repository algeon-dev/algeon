//! Native query runtime, lifecycle, ledger, and observation owners.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion) and its re-exports.
//! See the [API documentation boundary](https://algeon.dev/design#api-documentation).

pub mod attempt;
pub mod metrics;
pub mod native_error;
pub mod observability;
pub mod runtime;

mod triage_metrics;
