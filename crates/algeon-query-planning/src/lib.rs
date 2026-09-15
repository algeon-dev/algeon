//! Backend-neutral native query planning, proof, and admission contracts.
//!
//! This crate depends on neither DataFusion nor any RAPIDS component: its
//! resolved dependency closure under the union of every feature contains
//! neither, which `scripts/ci/check_dependency_direction.py` enforces.
//! Questions it cannot answer from Arrow and IR semantics alone; whether a
//! backend can represent a type, bind an expression, or infer a kernel's
//! output type; are asked through [`backend::BackendCapabilities`], which the
//! execution engine implements and passes in explicitly as a
//! [`backend::AnalysisContext`].
//!
//! It also performs no file or network I/O. Footers and decoded rows reach it
//! from whoever read them, through [`source::SourceFooterFacts`] and the
//! pruning resolvers in [`optimizer`].
//!
//! Note that a native-free dependency closure is a structural property, not a
//! demonstration: this workspace has no build or test lane that runs without a
//! CUDA toolchain installed, so "builds on a host without RAPIDS" remains
//! unproven here.

pub mod admission;
pub mod backend;
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
mod ml_validation;
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
