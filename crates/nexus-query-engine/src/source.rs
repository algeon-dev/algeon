//! Native query source contracts and source diagnostics.
//!
//! This module contains frontend-neutral source metadata. File locations,
//! schema mapping, partition values, missing columns, row-group metadata, and
//! reader predicate support are represented explicitly so native execution does
//! not depend on DataFusion scan internals.

mod diagnostics;
mod model;
mod reader_predicate_literals;

pub use diagnostics::{
    ByteRangeDisposition, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    ScanLimitDisposition, SchemaMappingDisposition, SelectedRowGroupStats, SourceDiagnostics, SourceFormat,
    ZeroColumnCountDisposition,
};
pub(crate) use diagnostics::{
    local_parquet_file_row_group_metadata_count, parquet_file_path_fact, parquet_file_stable_id_fact,
    parquet_source_path_fact, parquet_source_stable_id_fact, redacted_object_store_path_diagnostic,
    redacted_path_diagnostic,
};
pub use model::{
    CredentialMode, GpuResidentSource, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus,
    LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn, ParquetEqualityDeleteFile,
    ParquetFieldIdReadBinding, ParquetPositionDeleteFile, ParquetReadStrategy, ParquetReaderDecimalType,
    ParquetReaderDecimalTypeId, ParquetSourceLocation, PartitionColumn, RemoteParquetReadMode, ResolvedParquetSource,
    SourceColumnRef, SourceDecodeInput, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
};
pub(crate) use reader_predicate_literals::{
    ReaderPredicateDecimalScaleError, reader_predicate_decimal_raw_for_integer_value,
    reader_predicate_expr_references_decimal_column, reader_predicate_integer_literal_as,
    reader_predicate_integer_literal_as_decimal_raw, reader_predicate_integer_literal_raw,
    reader_predicate_rescale_decimal_raw, reader_predicate_unsigned_literal_as, reader_predicate_unsigned_literal_raw,
};
