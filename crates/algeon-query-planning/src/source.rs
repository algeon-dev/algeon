//! Native query source contracts and source diagnostics.
//!
//! This module contains frontend-neutral source metadata. File locations,
//! schema mapping, partition values, missing columns, row-group metadata, and
//! reader predicate support are represented explicitly so native execution does
//! not depend on any query-engine adapter's scan internals.

use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use arrow::{
    array::new_empty_array,
    datatypes::{Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};

use crate::error::{QueryError, QueryErrorCode, Result};

mod diagnostics;
mod footer_facts;
mod location;
mod parquet;
mod predicate;
mod read_policy;
mod reader_predicate_literals;
mod validation;

pub(crate) use diagnostics::parquet_source_stable_id_fact;
pub use diagnostics::{
    ByteRangeDisposition, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    ScanLimitDisposition, SchemaMappingDisposition, SelectedRowGroupStats, SourceDiagnostics,
    ZeroColumnCountDisposition,
};
#[doc(hidden)]
pub use diagnostics::{
    local_parquet_file_row_group_metadata_count, parquet_file_path_fact, parquet_file_stable_id_fact,
    parquet_source_path_fact, redacted_object_store_path_diagnostic, redacted_path_diagnostic,
};
pub use footer_facts::SourceFooterFacts;
pub use location::ParquetSourceLocation;
pub use parquet::{
    GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY, GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY,
    GraphSimpleDirectedEdgeTopology, LocalParquetFile, LocalParquetSource, MissingColumn, ParquetReaderDecimalType,
    ParquetReaderDecimalTypeId, PartitionColumn, SourceColumnRef, ZERO_COLUMN_COUNT_COLUMN,
};
pub use read_policy::{
    CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, ParquetReadStrategy,
    ResolvedParquetInput, ResolvedParquetSource,
};
#[doc(hidden)]
pub use reader_predicate_literals::{
    ReaderPredicateDecimalScaleError, reader_predicate_expr_references_decimal_column,
    reader_predicate_integer_literal_as, reader_predicate_integer_literal_as_decimal_raw,
    reader_predicate_rescale_decimal_raw, reader_predicate_unsigned_literal_as,
};
pub(crate) use reader_predicate_literals::{
    reader_predicate_decimal_raw_for_integer_value, reader_predicate_integer_literal_raw,
    reader_predicate_unsigned_literal_raw,
};
#[doc(hidden)]
pub use validation::validate_literal_batch_source;
pub use validation::validate_local_parquet_source;

/// Names of the columns in a local Parquet source's execution frame. A frame
/// column missing from the source schema is still named here; source
/// validation reports the missing field.
pub(crate) fn local_parquet_execution_frame_columns(source: &LocalParquetSource) -> BTreeSet<String> {
    local_parquet_execution_frame_schema(source).map_or_else(
        |_| source.projected_output_columns.iter().chain(&source.pure_filter_columns).cloned().collect(),
        |schema| schema.fields().iter().map(|field| field.name().clone()).collect(),
    )
}

#[doc(hidden)]
pub fn local_parquet_execution_frame_schema(source: &LocalParquetSource) -> Result<SchemaRef> {
    if source.is_zero_column_count_source() {
        return Ok(Arc::clone(&source.schema));
    }
    if source.is_count_star_probe_source() {
        return Ok(Arc::clone(source.output_schema()));
    }

    let mut fields = Vec::with_capacity(source.projected_output_columns.len() + source.pure_filter_columns.len());
    let mut seen = HashSet::with_capacity(fields.capacity());
    for column in source.projected_output_columns.iter().chain(&source.pure_filter_columns) {
        if !seen.insert(column.as_str()) {
            continue;
        }
        let field = source.schema.field_with_name(column).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.local_parquet_source.execution_frame_schema",
                "local parquet execution frame column is missing from the source schema",
            )
            .with_fact("column", column.as_str())
            .with_fact("source_fields", source.schema.fields().len())
            .with_source(error)
        })?;
        fields.push(field.clone());
    }

    Ok(Arc::new(Schema::new(fields)))
}

/// Native query-engine source description.
///
/// Frontends lower scan nodes into this enum after they have resolved file
/// locations, emitted source schema, schema mapping, projected column roles,
/// filter-only column roles, and source diagnostics. The executor should not
/// need frontend-specific scan metadata.
#[derive(Debug, Clone, PartialEq)]
#[expect(clippy::large_enum_variant, reason = "SourceNode variants represent disjoint planning sources")]
pub enum SourceNode {
    /// Local filesystem Parquet input.
    LocalParquet(LocalParquetSource),
    /// Already-materialized GPU input identified by a frontend-provided name.
    GpuResident(GpuResidentSource),
    /// Host Arrow literal batches imported into GPU memory at execution time.
    LiteralBatch(LiteralBatchSource),
}

impl SourceNode {
    pub(crate) fn output_schema(&self) -> &SchemaRef {
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
    /// A frontend-proved hard upper bound on the number of input rows.
    ///
    /// This evidence is optional because a generic device-resident handle does
    /// not intrinsically have a planning-time cardinality. When present, it
    /// can admit bounded operators without inspecting or materializing the
    /// handle first.
    pub row_count_upper_bound: Option<u64>,
    /// A source-owned upper bound on graph vertices for this exact edge rowset.
    /// Unlike row cardinality, this metadata is only meaningful to graph
    /// planning and remains absent for generic resident inputs.
    pub graph_vertex_upper_bound: Option<u64>,
    /// One source-owned key whose values are unique in this exact resident
    /// result, expressed in native schema names.
    pub(crate) unique_key_columns: Option<BTreeSet<String>>,
}

impl GpuResidentSource {
    /// Creates a GPU-resident source with the provided runtime name and schema.
    pub fn new(input_name: impl Into<String>, schema: SchemaRef) -> Self {
        Self {
            input_name: input_name.into(),
            schema,
            row_count_upper_bound: None,
            graph_vertex_upper_bound: None,
            unique_key_columns: None,
        }
    }

    /// Attaches a frontend-proved hard upper bound on this input's row count.
    pub fn with_row_count_upper_bound(mut self, row_count_upper_bound: u64) -> Self {
        self.row_count_upper_bound = Some(row_count_upper_bound);
        self
    }

    /// Attaches a source-owned graph vertex upper bound for this exact edge rowset.
    pub fn with_graph_vertex_upper_bound(mut self, graph_vertex_upper_bound: u64) -> Self {
        self.graph_vertex_upper_bound = Some(graph_vertex_upper_bound);
        self
    }

    /// Attaches a key proven unique by the producer that owns this resident
    /// result.
    pub fn with_unique_key_columns(mut self, unique_key_columns: BTreeSet<String>) -> Self {
        self.unique_key_columns = Some(unique_key_columns);
        self
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};

    use super::GpuResidentSource;

    #[test]
    fn gpu_resident_source_keeps_row_and_graph_vertex_bounds_distinct() {
        let source = GpuResidentSource::new(
            "edges",
            std::sync::Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false)])),
        )
        .with_row_count_upper_bound(7)
        .with_graph_vertex_upper_bound(11);

        assert_eq!(source.row_count_upper_bound, Some(7));
        assert_eq!(source.graph_vertex_upper_bound, Some(11));
    }
}
