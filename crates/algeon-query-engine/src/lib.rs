//! Algeon native query engine: execution composition over RAPIDS.
//!
//! The native IR, validation, capability and admission contracts, and the
//! optimizer belong to `query_planning`; attempts, device ledgers, contexts,
//! and observation belong to `query_runtime`. Callers import those owners
//! directly. This crate composes them: [`exec`] runs a validated
//! `query_planning::plan::QueryPlan` and returns host Arrow output or
//! GPU-resident handles according to its result sink, and [`rapids_backend`]
//! answers planning's backend capability questions for RAPIDS.
//!
//! This crate does not depend on any query-engine adapter. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `algeon-datafusion`.
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

pub mod chunk;
/// Arrow-to-cuDF datatype support and cast policy helpers.
pub(crate) mod datatypes;
/// Local-filesystem implementation of the planning dimension-pruning resolver contract.
pub mod dimension_pruning;
pub mod exec;
/// Backend materialized-output sizing, applying the component's formula.
pub(crate) mod materialization_sizing;
/// Native error projection: cuDF from the runtime boundary, cuVS from here.
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
