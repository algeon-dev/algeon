//! DataFusion-free Iceberg catalog and scan-planning contracts.
//!
//! This crate deliberately stops at Iceberg semantics. It knows catalog
//! configuration, resolved snapshots, source-access policy, and file/delete
//! facts, but it does not know DataFusion plans, native capability, GPU I/O,
//! cache strategy, or credentials in scan candidates.

mod error;
mod scan;
mod source_access;

#[cfg(feature = "catalog")]
mod catalog;
#[cfg(feature = "catalog")]
mod planner;
#[cfg(feature = "catalog")]
mod resolved;

#[cfg(feature = "glue")]
pub use catalog::GlueCatalogBackend;
#[cfg(feature = "rest")]
pub use catalog::RestCatalogBackend;
#[cfg(feature = "catalog")]
pub use catalog::{CatalogBackend, CatalogKind, Namespace};
pub use error::{CatalogErrorKind, Error, ErrorCode, ErrorContext, ErrorStatus, Operation, Result};
#[cfg(feature = "catalog")]
pub use planner::ScanPlanningOptions;
#[cfg(feature = "catalog")]
pub use resolved::ResolvedTable;
pub use scan::{
    ColumnBinding, ColumnRole, ColumnSource, DataFile, DefaultValue, DeleteContent, DeleteFile, FileFormat, FileRange,
    FileSequence, NameMapping, NameMappingField, PartitionTransform, PartitionValue, PredicateEnforcement,
    PredicateIntent, Projection, PruningIntent, ResidualPredicate, ResolvedSnapshot, ResolvedTableFacts, ScalarValue,
    ScanCandidate, ScanRequest, Schema, SchemaField, SourceLimit, TableIdentifier,
};
pub use source_access::{
    GlueCatalogConfig, RestCatalogConfig, S3CredentialMode, S3CredentialRef, S3FileIoConfigVerification,
    S3SourceAccessPolicy, S3SourceAccessRef, SourceAccessPolicy,
};
