//! Backend-neutral native query planning, proof, and admission contracts.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion) and its re-exports.
//! See the [API documentation boundary](https://algeon.dev/design#api-documentation).
//!
//! This crate contains no dependency on DataFusion or RAPIDS components.
//! The resolved dependency closure across all feature combinations excludes both,
//! as validated by `scripts/ci/check_dependency_direction.py`.
//! Capabilities that depend on backend execution semantics, such as type
//! representation, expression binding, or kernel output type inference, are
//! mediated through the [`backend::BackendCapabilities`] trait and supplied
//! via [`backend::AnalysisContext`].
//!
//! The crate performs no file or network I/O. File metadata and decoded statistics
//! are provided externally through [`source::SourceFooterFacts`] and the
//! pruning resolvers in [`optimizer`].
//!
//! While dependency direction is structurally enforced by CI scripts, build
//! behavior on hosts lacking RAPIDS or CUDA toolchains is not validated by the
//! workspace test harness.

pub mod admission;
pub mod backend;
pub mod capability;
pub mod chunk;
pub mod contract_report;
pub mod correlation;
pub mod datatypes;
pub mod dynamic_filter;
pub mod error;
pub mod execution_contracts;
pub mod expr;
#[doc(hidden)]
pub mod expression_lowering_policy;
mod ml_validation;
pub mod mode;
pub mod naming;
pub mod optimizer;
pub mod parquet_stats;
pub mod pipeline;
pub mod plan;
pub mod policy;
#[doc(hidden)]
pub mod prepared;
pub mod source;
pub mod triage;
#[doc(hidden)]
pub mod util;
#[doc(hidden)]
pub mod validation;
mod vector_validation;

pub use validation::validate_query_plan;
