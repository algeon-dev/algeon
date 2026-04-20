//! Native query source contracts and source diagnostics.
//!
//! This module contains frontend-neutral source metadata. File locations,
//! schema mapping, partition values, missing columns, row-group metadata, and
//! reader predicate support are represented explicitly so native execution does
//! not depend on DataFusion scan internals.

use std::sync::Arc;

use arrow::{
    array::new_empty_array,
    datatypes::{Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};

mod deletes;
mod diagnostics;
mod location;
mod parquet;
mod predicate;
mod read_policy;
mod reader_predicate_literals;
mod validation;

pub use deletes::{ParquetEqualityDeleteFile, ParquetPositionDeleteFile};
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
pub use location::ParquetSourceLocation;
pub use parquet::{
    LocalParquetFile, LocalParquetSource, MissingColumn, ParquetFieldIdReadBinding, ParquetReaderDecimalType,
    ParquetReaderDecimalTypeId, PartitionColumn, SourceColumnRef, ZERO_COLUMN_COUNT_COLUMN,
};
pub use read_policy::{
    CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, ParquetReadStrategy,
    RemoteParquetReadMode, ResolvedParquetSource, SourceDecodeInput,
};
pub(crate) use reader_predicate_literals::{
    ReaderPredicateDecimalScaleError, reader_predicate_decimal_raw_for_integer_value,
    reader_predicate_expr_references_decimal_column, reader_predicate_integer_literal_as,
    reader_predicate_integer_literal_as_decimal_raw, reader_predicate_integer_literal_raw,
    reader_predicate_rescale_decimal_raw, reader_predicate_unsigned_literal_as, reader_predicate_unsigned_literal_raw,
};
pub(crate) use validation::validate_literal_batch_source;
pub use validation::validate_local_parquet_source;

/// Native query-engine source description.
///
/// Frontends lower scan nodes into this enum after they have resolved file
/// locations, emitted source schema, schema mapping, projected column roles,
/// filter-only column roles, and source diagnostics. The executor should not
/// need frontend-specific scan metadata.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum SourceNode {
    /// Local filesystem Parquet input.
    LocalParquet(LocalParquetSource),
    /// Already-materialized GPU input identified by a frontend-provided name.
    GpuResident(GpuResidentSource),
    /// Host Arrow literal batches imported into GPU memory at execution time.
    LiteralBatch(LiteralBatchSource),
}

impl SourceNode {
    /// Returns the native output schema exposed by this source node.
    pub fn output_schema(&self) -> &SchemaRef {
        match self {
            Self::LocalParquet(source) => source.output_schema(),
            Self::GpuResident(source) => &source.schema,
            Self::LiteralBatch(source) => &source.schema,
        }
    }
}

/// Literal host Arrow batches used as a native source.
///
/// This source is for SQL roots such as `VALUES`, `SELECT 1`, and typed empty
/// inputs. Execution imports the batches into the query runtime explicitly; it
/// does not read files or call back into a frontend executor.
#[derive(Debug, Clone, PartialEq)]
pub struct LiteralBatchSource {
    /// Schema exposed by this source.
    pub schema: SchemaRef,
    /// Host Arrow batches in source order.
    pub batches: Vec<RecordBatch>,
}

impl LiteralBatchSource {
    /// Creates a literal source from already-typed host Arrow batches.
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self { schema, batches }
    }

    /// Creates a one-row, zero-column placeholder source.
    pub fn placeholder_row() -> Self {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("one-row empty RecordBatch is valid");
        Self::new(schema, vec![batch])
    }

    /// Creates a zero-row source with the provided schema.
    pub fn empty(schema: SchemaRef) -> Self {
        let columns = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(0)),
        )
        .expect("zero-row RecordBatch must match its schema");
        Self::new(schema, vec![batch])
    }

    /// Total row count across all batches.
    pub fn row_count(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// Source for data that already resides on the GPU.
///
/// `input_name` is resolved by the embedding runtime. `schema` is the native
/// output schema that downstream query IR nodes consume.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuResidentSource {
    /// Runtime input handle supplied by the frontend or embedding application.
    pub input_name: String,
    /// Schema of the GPU-resident input as seen by the native plan.
    pub schema: SchemaRef,
}

impl GpuResidentSource {
    /// Creates a GPU-resident source with the provided runtime name and schema.
    pub fn new(input_name: impl Into<String>, schema: SchemaRef) -> Self {
        Self { input_name: input_name.into(), schema }
    }
}
