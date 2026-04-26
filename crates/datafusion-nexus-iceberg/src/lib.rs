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
pub use catalog::GlueIcebergCatalogBackend;
#[cfg(feature = "rest")]
pub use catalog::RestIcebergCatalogBackend;
#[cfg(feature = "catalog")]
pub use catalog::{IcebergCatalogBackend, IcebergCatalogKind, IcebergNamespace};
pub use error::{
    IcebergCatalogErrorKind, IcebergError, IcebergErrorCode, IcebergErrorContext, IcebergErrorStatus, IcebergOperation,
    IcebergResult, classify_error_message,
};
#[cfg(feature = "catalog")]
pub use planner::{IcebergScanPlanner, IcebergScanPlanningOptions};
#[cfg(feature = "catalog")]
pub use resolved::ResolvedIcebergTable;
pub use scan::{
    IcebergColumnBinding, IcebergColumnRole, IcebergColumnSource, IcebergDataFile, IcebergDefaultValue,
    IcebergDeleteContent, IcebergDeleteFile, IcebergFileFormat, IcebergFileRange, IcebergFileSequence,
    IcebergNameMapping, IcebergNameMappingField, IcebergPartitionTransform, IcebergPartitionValue,
    IcebergPredicateEnforcement, IcebergPredicateIntent, IcebergProjection, IcebergPruningIntent,
    IcebergResidualPredicate, IcebergScalarValue, IcebergScanCandidate, IcebergScanRequest, IcebergSchema,
    IcebergSchemaField, IcebergSourceLimit, IcebergTableIdentifier, ResolvedIcebergSnapshot, ResolvedIcebergTableFacts,
};
pub use source_access::{
    GlueIcebergCatalogConfig, RestIcebergCatalogConfig, S3CredentialMode, S3CredentialRef, S3FileIoConfigVerification,
    S3SourceAccessPolicy, S3SourceAccessRef, SourceAccessPolicy,
};
