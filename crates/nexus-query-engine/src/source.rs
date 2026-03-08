//! Native query source contracts and source diagnostics.
//!
//! This module contains frontend-neutral source metadata. File locations,
//! schema mapping, partition values, missing columns, row-group metadata, and
//! reader predicate support are represented explicitly so native execution does
//! not depend on DataFusion scan internals.

mod diagnostics;
mod model;

pub use diagnostics::{
    ByteRangeDisposition, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    ScanLimitDisposition, SchemaMappingDisposition, SelectedRowGroupStats, SourceDiagnostics, SourceFormat,
    ZeroColumnCountDisposition,
};
pub(crate) use diagnostics::{redacted_object_store_path_diagnostic, redacted_path_diagnostic};
pub use model::{
    CredentialMode, GpuResidentSource, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus,
    LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn, ParquetFieldIdReadBinding,
    ParquetReadStrategy, ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, PartitionColumn,
    RemoteParquetReadMode, ResolvedParquetSource, SourceColumnRef, SourceDecodeInput, SourceNode,
    ZERO_COLUMN_COUNT_COLUMN,
};
