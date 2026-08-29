//! Chunk assembly from row-group metadata.

use std::collections::BTreeSet;

use super::{SourceChunk, SourceChunkPlanningPolicy, contiguous_row_group_position_range, row_group::RowGroupMetadata};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::NodeId,
    source::{LocalParquetFile, LocalParquetSource, ParquetReaderDecimalType},
};

pub(super) fn planned_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    metadata: Option<&RowGroupMetadata>,
) -> Result<Option<Vec<usize>>> {
    if let Some(row_groups) = &file.row_groups {
        validate_selected_row_groups(file, file_index, row_groups, metadata)?;
        validate_position_delete_planned_row_groups(file, file_index, row_groups, metadata)?;
        return Ok(Some(row_groups.clone()));
    }

    let Some(metadata) = metadata else {
        return Ok(None);
    };

    let row_groups = (0..metadata.row_group_count()).collect::<Vec<_>>();
    validate_position_delete_planned_row_groups(file, file_index, &row_groups, Some(metadata))?;
    Ok(Some(row_groups))
}

pub(super) fn validate_selected_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
) -> Result<()> {
    let Some(metadata) = metadata else {
        return Ok(());
    };

    for (row_group_position, row_group) in row_groups.iter().enumerate() {
        if *row_group >= metadata.row_group_count() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.plan",
                "selected parquet row group is outside file metadata",
            )
            .with_fact("path", file.location.diagnostic_location())
            .with_fact("stable_id", crate::source::parquet_file_stable_id_fact(file))
            .with_fact("file_index", file_index)
            .with_fact("row_group_position", row_group_position)
            .with_fact("row_group", row_group)
            .with_fact("max_row_group", metadata.row_group_count().saturating_sub(1)));
        }
    }

    Ok(())
}

pub(super) fn validate_position_delete_planned_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
) -> Result<()> {
    if !file.has_position_delete_files() || row_groups.is_empty() {
        return Ok(());
    }
    let row_group_rows = metadata.and_then(RowGroupMetadata::row_group_rows_vec).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "Iceberg position-delete row-group planning requires row-count metadata",
        )
        .with_fact("reason", "position_deletes_require_row_group_rows")
        .with_fact("path", file.location.diagnostic_location())
        .with_fact("stable_id", crate::source::parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
    })?;
    contiguous_row_group_position_range(row_groups, &row_group_rows).map(|_range| ()).map_err(|error| {
        error
            .with_fact("path", file.location.diagnostic_location())
            .with_fact("stable_id", crate::source::parquet_file_stable_id_fact(file))
            .with_fact("file_index", file_index)
    })
}

pub(super) fn append_row_group_chunks(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
    chunks: &mut Vec<SourceChunk>,
) {
    let max_row_groups = policy.effective_max_row_groups_per_chunk();
    let row_group_count_basis = metadata.map_or(row_groups.len(), RowGroupMetadata::row_group_count).max(1);
    let mut current = PendingChunk::default();

    for row_group in row_groups {
        let row_group_bytes = metadata.and_then(|metadata| metadata.row_group_bytes(*row_group));
        let estimated_row_group_bytes =
            row_group_bytes.or_else(|| proportional_bytes(file.file_size_bytes, 1, row_group_count_basis));
        let projected_compressed_bytes = metadata.and_then(|metadata| metadata.projected_compressed_bytes(*row_group));
        let projected_uncompressed_bytes =
            metadata.and_then(|metadata| metadata.projected_uncompressed_bytes(*row_group));
        let pure_filter_compressed_bytes =
            metadata.and_then(|metadata| metadata.pure_filter_compressed_bytes(*row_group));
        let pure_filter_uncompressed_bytes =
            metadata.and_then(|metadata| metadata.pure_filter_uncompressed_bytes(*row_group));
        let row_group_rows = metadata.and_then(|metadata| metadata.row_group_rows(*row_group));
        let retained_output_bytes = projected_uncompressed_bytes.or(estimated_row_group_bytes);
        let source_read_planning_bytes =
            source_read_uncompressed_bytes(source, projected_uncompressed_bytes, pure_filter_uncompressed_bytes)
                .or(retained_output_bytes);
        if current.should_flush_before(source_read_planning_bytes, max_row_groups, policy.max_source_chunk_bytes) {
            push_pending_chunk(
                source_node_id,
                source,
                file,
                file_index,
                row_groups.len(),
                metadata,
                policy,
                reader_decimal_types,
                reader_decimal_types_footer_verified,
                &mut current,
                chunks,
            );
        }
        current.push(
            *row_group,
            row_group_bytes,
            source_read_planning_bytes,
            projected_compressed_bytes,
            projected_uncompressed_bytes,
            pure_filter_compressed_bytes,
            pure_filter_uncompressed_bytes,
            row_group_rows,
        );
    }

    push_pending_chunk(
        source_node_id,
        source,
        file,
        file_index,
        row_groups.len(),
        metadata,
        policy,
        reader_decimal_types,
        reader_decimal_types_footer_verified,
        &mut current,
        chunks,
    );
}
pub(super) fn push_pending_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    planned_row_group_count: usize,
    metadata: Option<&RowGroupMetadata>,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
    pending: &mut PendingChunk,
    chunks: &mut Vec<SourceChunk>,
) {
    if pending.row_groups.is_empty() {
        return;
    }

    let known_row_group_bytes = pending.known_bytes();
    let known_projected_compressed_bytes = pending.known_projected_compressed_bytes();
    let known_projected_uncompressed_bytes = pending.known_projected_uncompressed_bytes();
    let known_pure_filter_compressed_bytes = pending.known_pure_filter_compressed_bytes();
    let known_pure_filter_uncompressed_bytes = pending.known_pure_filter_uncompressed_bytes();
    let known_row_count = pending.known_rows();
    let estimated_chunk_bytes = known_projected_uncompressed_bytes
        .or(known_row_group_bytes)
        .or_else(|| proportional_bytes(file.file_size_bytes, pending.row_groups.len(), planned_row_group_count));
    let estimated_source_read_bytes = source_read_uncompressed_bytes(
        source,
        known_projected_uncompressed_bytes,
        known_pure_filter_uncompressed_bytes,
    )
    .or(known_row_group_bytes)
    .or(estimated_chunk_bytes);
    let exact_source_read_preflight_bytes =
        exact_source_read_preflight_bytes(source, known_row_count, source.is_zero_column_count_source());
    chunks.push(SourceChunk {
        source_node_id,
        chunk_index: chunks.len(),
        file_index,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: std::mem::take(&mut pending.row_groups),
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes,
        known_projected_compressed_bytes,
        known_projected_uncompressed_bytes,
        known_pure_filter_compressed_bytes,
        known_pure_filter_uncompressed_bytes,
        estimated_chunk_bytes,
        estimated_source_read_bytes,
        exact_source_read_preflight_bytes,
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: metadata.and_then(RowGroupMetadata::row_group_bytes_vec),
        source_file_projected_compressed_bytes: metadata.and_then(RowGroupMetadata::projected_compressed_bytes_vec),
        source_file_projected_uncompressed_bytes: metadata.and_then(RowGroupMetadata::projected_uncompressed_bytes_vec),
        source_file_pure_filter_compressed_bytes: metadata.and_then(RowGroupMetadata::pure_filter_compressed_bytes_vec),
        source_file_pure_filter_uncompressed_bytes: metadata
            .and_then(RowGroupMetadata::pure_filter_uncompressed_bytes_vec),
        source_file_row_group_rows: metadata.and_then(RowGroupMetadata::row_group_rows_vec),
        reader_decimal_types: reader_decimal_types.to_vec(),
        reader_decimal_types_footer_verified,
        read_columns: source.read_columns.clone(),
        partition_values: file.partition_values.clone(),
        zero_column_count: source.is_zero_column_count_source(),
        known_row_count,
        scan_limit_rows: None,
        size_estimate_based: known_row_group_bytes.is_none(),
        source_batching_reason: None,
        source_chunk_planning_mode: policy.source_chunk_planning_mode(),
        source_chunk_planning_reason: policy.source_chunk_planning_reason().to_string(),
    });
    pending.clear();
}

pub(super) fn source_read_uncompressed_bytes(
    source: &LocalParquetSource,
    projected_uncompressed_bytes: Option<u64>,
    pure_filter_uncompressed_bytes: Option<u64>,
) -> Option<u64> {
    match (projected_uncompressed_bytes, pure_filter_uncompressed_bytes) {
        (Some(projected), Some(filter)) if projected_and_pure_filter_file_columns_overlap(source) => {
            Some(projected.max(filter))
        },
        (Some(projected), Some(filter)) => projected.checked_add(filter),
        (Some(projected), None) => Some(projected),
        (None, Some(filter)) => Some(filter),
        (None, None) => None,
    }
}

pub(super) fn exact_source_read_preflight_bytes(
    source: &LocalParquetSource,
    known_row_count: Option<u64>,
    zero_column_count: bool,
) -> Option<u64> {
    let row_count = if zero_column_count { 1 } else { known_row_count? };
    if zero_column_count {
        return exact_source_read_required_budget_bytes(exact_fixed_width_column_upper_bound_bytes(
            &arrow::datatypes::DataType::Int64,
            false,
            row_count,
        )?);
    }
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for column in source.projected_output_columns.iter().chain(&source.pure_filter_columns) {
        if !seen.insert(column.as_str()) {
            continue;
        }
        let field = source.schema.field_with_name(column).ok()?;
        let column_bytes =
            exact_fixed_width_column_upper_bound_bytes(field.data_type(), field.is_nullable(), row_count)?;
        total = total.checked_add(column_bytes)?;
    }
    exact_source_read_required_budget_bytes(total)
}

pub(super) fn exact_source_read_required_budget_bytes(retained_output_bytes: u64) -> Option<u64> {
    let minimum_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).ok()?;
    Some(retained_output_bytes.max(minimum_budget))
}

pub(super) fn exact_fixed_width_column_upper_bound_bytes(
    data_type: &arrow::datatypes::DataType,
    nullable: bool,
    row_count: u64,
) -> Option<u64> {
    crate::datatypes::cudf_dtype_from_arrow_type(data_type)?;
    let element_size = crate::datatypes::arrow_fixed_width_byte_size(data_type)?;
    let data_bytes = row_count.checked_mul(element_size)?;
    let mask_bytes = if nullable { null_mask_allocation_size_bytes(row_count) } else { 0 };
    data_bytes.checked_add(mask_bytes)
}

pub(super) fn null_mask_allocation_size_bytes(row_count: u64) -> u64 {
    if row_count == 0 { 0 } else { row_count.div_ceil(8).div_ceil(64) * 64 }
}

pub(super) fn source_file_has_row_count_metadata(file: &LocalParquetFile) -> bool {
    let Some(rows) = &file.row_group_rows else {
        return false;
    };
    if let Some(selected) = &file.row_groups {
        selected.iter().all(|row_group| *row_group < rows.len())
    } else {
        !rows.is_empty()
    }
}

pub(super) fn projected_and_pure_filter_file_columns_overlap(source: &LocalParquetSource) -> bool {
    let projected = source
        .projected_output_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            crate::source::SourceColumnRef::File(index) => Some(*index),
            crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();
    source.pure_filter_column_source_refs.iter().any(|source_ref| match source_ref {
        crate::source::SourceColumnRef::File(index) => projected.contains(index),
        crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => false,
    })
}

pub(super) fn single_all_row_groups_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
) -> SourceChunk {
    SourceChunk {
        source_node_id,
        chunk_index,
        file_index,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: Vec::new(),
        reads_all_row_groups: true,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: None,
        known_projected_compressed_bytes: None,
        known_projected_uncompressed_bytes: None,
        known_pure_filter_compressed_bytes: None,
        known_pure_filter_uncompressed_bytes: None,
        estimated_chunk_bytes: file.file_size_bytes,
        estimated_source_read_bytes: file.file_size_bytes,
        exact_source_read_preflight_bytes: None,
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: None,
        source_file_projected_compressed_bytes: None,
        source_file_projected_uncompressed_bytes: None,
        source_file_pure_filter_compressed_bytes: None,
        source_file_pure_filter_uncompressed_bytes: None,
        source_file_row_group_rows: None,
        reader_decimal_types: reader_decimal_types.to_vec(),
        reader_decimal_types_footer_verified,
        read_columns: source.read_columns.clone(),
        partition_values: file.partition_values.clone(),
        zero_column_count: source.is_zero_column_count_source(),
        known_row_count: None,
        scan_limit_rows: None,
        size_estimate_based: true,
        source_batching_reason: None,
        source_chunk_planning_mode: policy.source_chunk_planning_mode(),
        source_chunk_planning_reason: policy.source_chunk_planning_reason().to_string(),
    }
}

/// Builds the synthetic zero-weight chunk for a represented empty COUNT(*) source.
///
/// A metadata-only count source models each selected file as one count weight.
/// When no row groups survive planning, that source still needs the additive
/// identity so the lowered `SUM(weight)` yields `COUNT(*) = 0` rather than
/// SQL `SUM(empty) = NULL`.
pub(super) fn zero_column_count_identity_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    policy: SourceChunkPlanningPolicy,
    reader_decimal_types: &[ParquetReaderDecimalType],
    reader_decimal_types_footer_verified: bool,
) -> SourceChunk {
    let mut chunk = single_all_row_groups_chunk(
        source_node_id,
        source,
        file,
        file_index,
        chunk_index,
        policy,
        reader_decimal_types,
        reader_decimal_types_footer_verified,
    );
    chunk.reads_all_row_groups = false;
    chunk.known_row_count = Some(0);
    chunk.estimated_chunk_bytes = Some(0);
    chunk.estimated_source_read_bytes = Some(0);
    chunk.exact_source_read_preflight_bytes = exact_source_read_preflight_bytes(source, chunk.known_row_count, true);
    chunk.size_estimate_based = false;
    chunk
}

pub(super) fn proportional_bytes(
    total_bytes: Option<u64>,
    child_row_groups: usize,
    parent_row_groups: usize,
) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

#[derive(Debug, Default)]
pub(super) struct PendingChunk {
    row_groups: Vec<usize>,
    known_bytes: u64,
    known_planning_bytes: u64,
    known_projected_compressed_bytes: u64,
    known_projected_uncompressed_bytes: u64,
    known_pure_filter_compressed_bytes: u64,
    known_pure_filter_uncompressed_bytes: u64,
    known_rows: u64,
    has_unknown_bytes: bool,
    has_unknown_projected_compressed_bytes: bool,
    has_unknown_projected_uncompressed_bytes: bool,
    has_unknown_pure_filter_compressed_bytes: bool,
    has_unknown_pure_filter_uncompressed_bytes: bool,
    has_unknown_rows: bool,
}

impl PendingChunk {
    fn push(
        &mut self,
        row_group: usize,
        row_group_bytes: Option<u64>,
        planning_bytes: Option<u64>,
        projected_compressed_bytes: Option<u64>,
        projected_uncompressed_bytes: Option<u64>,
        pure_filter_compressed_bytes: Option<u64>,
        pure_filter_uncompressed_bytes: Option<u64>,
        row_group_rows: Option<u64>,
    ) {
        self.row_groups.push(row_group);
        if let Some(row_group_bytes) = row_group_bytes {
            self.known_bytes = self.known_bytes.saturating_add(row_group_bytes);
        } else {
            self.has_unknown_bytes = true;
        }
        if let Some(planning_bytes) = planning_bytes {
            self.known_planning_bytes = self.known_planning_bytes.saturating_add(planning_bytes);
        }
        if let Some(projected_compressed_bytes) = projected_compressed_bytes {
            self.known_projected_compressed_bytes =
                self.known_projected_compressed_bytes.saturating_add(projected_compressed_bytes);
        } else {
            self.has_unknown_projected_compressed_bytes = true;
        }
        if let Some(projected_uncompressed_bytes) = projected_uncompressed_bytes {
            self.known_projected_uncompressed_bytes =
                self.known_projected_uncompressed_bytes.saturating_add(projected_uncompressed_bytes);
        } else {
            self.has_unknown_projected_uncompressed_bytes = true;
        }
        if let Some(pure_filter_compressed_bytes) = pure_filter_compressed_bytes {
            self.known_pure_filter_compressed_bytes =
                self.known_pure_filter_compressed_bytes.saturating_add(pure_filter_compressed_bytes);
        } else {
            self.has_unknown_pure_filter_compressed_bytes = true;
        }
        if let Some(pure_filter_uncompressed_bytes) = pure_filter_uncompressed_bytes {
            self.known_pure_filter_uncompressed_bytes =
                self.known_pure_filter_uncompressed_bytes.saturating_add(pure_filter_uncompressed_bytes);
        } else {
            self.has_unknown_pure_filter_uncompressed_bytes = true;
        }
        if let Some(row_group_rows) = row_group_rows {
            self.known_rows = self.known_rows.saturating_add(row_group_rows);
        } else {
            self.has_unknown_rows = true;
        }
    }

    fn should_flush_before(
        &self,
        next_row_group_bytes: Option<u64>,
        max_row_groups: usize,
        max_source_chunk_bytes: Option<u64>,
    ) -> bool {
        if self.row_groups.is_empty() {
            return false;
        }
        if self.row_groups.len() >= max_row_groups {
            return true;
        }

        let Some(max_source_chunk_bytes) = max_source_chunk_bytes else {
            return false;
        };
        let Some(next_row_group_bytes) = next_row_group_bytes else {
            return false;
        };

        self.known_planning_bytes.saturating_add(next_row_group_bytes) > max_source_chunk_bytes
    }

    fn known_bytes(&self) -> Option<u64> {
        (!self.has_unknown_bytes).then_some(self.known_bytes)
    }

    fn known_projected_compressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_projected_compressed_bytes).then_some(self.known_projected_compressed_bytes)
    }

    fn known_projected_uncompressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_projected_uncompressed_bytes).then_some(self.known_projected_uncompressed_bytes)
    }

    fn known_pure_filter_compressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_pure_filter_compressed_bytes).then_some(self.known_pure_filter_compressed_bytes)
    }

    fn known_pure_filter_uncompressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_pure_filter_uncompressed_bytes).then_some(self.known_pure_filter_uncompressed_bytes)
    }

    fn known_rows(&self) -> Option<u64> {
        (!self.has_unknown_rows).then_some(self.known_rows)
    }

    fn clear(&mut self) {
        self.row_groups.clear();
        self.known_bytes = 0;
        self.known_planning_bytes = 0;
        self.known_projected_compressed_bytes = 0;
        self.known_projected_uncompressed_bytes = 0;
        self.known_pure_filter_compressed_bytes = 0;
        self.known_pure_filter_uncompressed_bytes = 0;
        self.known_rows = 0;
        self.has_unknown_bytes = false;
        self.has_unknown_projected_compressed_bytes = false;
        self.has_unknown_projected_uncompressed_bytes = false;
        self.has_unknown_pure_filter_compressed_bytes = false;
        self.has_unknown_pure_filter_uncompressed_bytes = false;
        self.has_unknown_rows = false;
    }
}
