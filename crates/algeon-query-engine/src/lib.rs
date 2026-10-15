//! Algeon native query engine executing query plans over RAPIDS.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion) and its re-exports.
//! See the [API documentation boundary](https://algeon.dev/design#api-documentation).
//!
//! Native IR definitions, validation, capability and admission contracts, and the
//! optimizer belong to `query_planning`. Execution attempts, device ledgers, contexts,
//! and observation belong to `query_runtime`. This crate composes those components:
//! [`exec`] executes validated `query_planning::plan::QueryPlan` trees, yielding either
//! host Arrow batches or GPU-resident handles according to the configured result sink,
//! while [`rapids_backend`] implements the planning backend capability contract for RAPIDS.
//!
//! This crate maintains no dependencies on specific query-engine adapters. DataFusion
//! physical plan matching, fallback reporting, and adapter wrappers reside in
//! `algeon-datafusion`.
//!
//! Byte quantities reported in native metrics and runtime contracts represent device bytes
//! unless explicitly qualified as host or Arrow bytes. Native identifiers and cache keys
//! serve as plan-local or runtime-local identities rather than globally stable handles.
//!
//! The `cuvs` feature enables vector-algorithm execution. The `cuml` feature
//! enables query-local TSVD, HDBSCAN, UMAP, logistic-regression, and
//! random-forest execution. Respective planning contracts remain available when
//! either feature is disabled.

pub mod chunk;
/// Arrow-to-cuDF datatype support and cast policy helpers.
pub(crate) mod datatypes;
/// Local-filesystem implementation of the planning dimension-pruning resolver contract.
pub mod dimension_pruning;
pub mod exec;
/// Backend materialized-output sizing, applying the component's formula.
pub(crate) mod materialization_sizing;
/// Native error projection for cuDF and cuVS.
pub mod native_error;
#[doc(hidden)]
pub mod object_store_error;
pub(crate) mod parquet_metadata;
/// Native cuDF expression, aggregate, and sort lowering.
pub(crate) mod prepared;
/// RAPIDS implementation of the planning backend capability contract.
pub mod rapids_backend;

pub mod shadow;
pub(crate) mod source_chunk;
