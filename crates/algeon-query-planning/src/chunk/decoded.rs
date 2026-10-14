//! Decoded device-size estimates for the read columns of Parquet row groups.
//!
//! A footer's uncompressed column size is still Parquet-encoded: a
//! dictionary-encoded string column stores only its dictionary indexes there.
//! Chunk planning compares these decoded estimates with the source-byte
//! target, and the engine's source-adjacent batch model reads the same figure
//! through `SourceChunk::estimated_decoded_bytes`.

use arrow::datatypes::{DataType as ArrowDataType, Schema};
use parquet::file::metadata::{ColumnChunkMetaData, ParquetMetaData};

use super::assembly::null_mask_allocation_size_bytes;
use crate::{
    datatypes::arrow_fixed_width_byte_size, parquet_stats::root_column_leaf_indices, source::LocalParquetSource,
};

/// Per-row payload assumed for a variable-width column whose footer has no
/// unencoded byte-array size (Parquet `SizeStatistics`), when its
/// uncompressed bytes are smaller.
const COLD_VARIABLE_WIDTH_PAYLOAD_ROW_BYTES: u64 = 16;

/// Offset bytes per row of a decoded variable-width column. cuDF switches a
/// strings column to 64-bit offsets once its characters exceed 2 GiB; the
/// estimate always uses that width, which overstates a smaller column by
/// 4 bytes per row.
const VARIABLE_WIDTH_OFFSET_ROW_BYTES: u64 = 8;

/// Decoded widths of a source's read columns, from its Arrow schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ReadColumnWidths {
    /// Sum of the Arrow byte widths of the fixed-width read columns.
    pub(crate) fixed_width_row_bytes: u64,
    /// Read columns without a fixed width, or absent from the schema.
    pub(crate) variable_width_columns: u64,
    /// Nullable read columns, each decoded with a validity mask.
    pub(crate) nullable_columns: u64,
}

impl ReadColumnWidths {
    /// Classifies `read_columns` by their types in `schema`.
    pub(crate) fn of(schema: &Schema, read_columns: &[String]) -> Self {
        read_columns.iter().fold(Self::default(), |mut widths, column| {
            let field = schema.field_with_name(column).ok();
            match field.and_then(|field| arrow_fixed_width_byte_size(field.data_type())) {
                Some(width) => widths.fixed_width_row_bytes = widths.fixed_width_row_bytes.saturating_add(width),
                None => widths.variable_width_columns += 1,
            }
            if field.is_none_or(|field| field.is_nullable()) {
                widths.nullable_columns += 1;
            }
            widths
        })
    }

    /// Decoded estimate for `rows` rows whose read columns have
    /// `uncompressed_bytes` in the footer, without per-column footer detail.
    ///
    /// Fixed-width columns take their Arrow width. The variable-width payload
    /// is the larger of the whole uncompressed size, which also counts the
    /// encoded fixed-width columns, and the cold per-row payload. `None`
    /// without read columns.
    pub(crate) fn decoded_bytes(self, rows: u64, uncompressed_bytes: u64) -> Option<u64> {
        if self == Self::default() {
            return None;
        }
        let variable_payload = if self.variable_width_columns == 0 {
            0
        } else {
            uncompressed_bytes
                .max(rows.saturating_mul(self.variable_width_columns * COLD_VARIABLE_WIDTH_PAYLOAD_ROW_BYTES))
        };
        rows.saturating_mul(
            self.fixed_width_row_bytes
                .saturating_add(self.variable_width_columns.saturating_mul(VARIABLE_WIDTH_OFFSET_ROW_BYTES)),
        )
        .saturating_add(variable_payload)
        .saturating_add(self.nullable_columns.saturating_mul(null_mask_allocation_size_bytes(rows)))
        .into()
    }
}

/// Decoded estimate of the source's read columns for every physical row group
/// of `metadata`.
///
/// `read_column_file_indices` are root column indexes; a nested read column
/// sums the column chunks of every leaf beneath it. Each read column
/// contributes its Arrow width per row when the source schema gives it one. A
/// variable-width column contributes 64-bit offsets plus its payload: per
/// leaf, the footer's unencoded byte-array size when present, otherwise the
/// larger of the leaf's uncompressed size and the cold per-row payload.
/// Nullable columns add a validity mask. `None` without read columns, since a
/// zero-column count decodes no column, or when a read column has no column
/// chunk.
pub fn row_group_decoded_bytes_from_footer(
    metadata: &ParquetMetaData,
    schema: &Schema,
    read_columns: &[String],
    read_column_file_indices: &[usize],
) -> Option<Vec<u64>> {
    if read_columns.is_empty() || read_columns.len() != read_column_file_indices.len() {
        return None;
    }
    let footer_schema = metadata.file_metadata().schema_descr();
    let read_column_leaves = read_column_file_indices
        .iter()
        .map(|root_index| {
            Some(root_column_leaf_indices(footer_schema, *root_index)).filter(|leaves| !leaves.is_empty())
        })
        .collect::<Option<Vec<_>>>()?;
    metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            let rows = u64::try_from(row_group.num_rows()).ok()?;
            read_columns.iter().zip(&read_column_leaves).try_fold(0u64, |total, (column, leaves)| {
                let chunks = leaves.iter().map(|leaf| row_group.columns().get(*leaf)).collect::<Option<Vec<_>>>()?;
                let field = schema.field_with_name(column).ok();
                let bytes = decoded_column_bytes(
                    field.map(|field| field.data_type()),
                    field.is_none_or(|field| field.is_nullable()),
                    rows,
                    &chunks,
                )?;
                total.checked_add(bytes)
            })
        })
        .collect()
}

/// [`row_group_decoded_bytes_from_footer`] over `source`'s schema and read columns.
pub(crate) fn source_row_group_decoded_bytes_from_footer(
    metadata: &ParquetMetaData,
    source: &LocalParquetSource,
) -> Option<Vec<u64>> {
    row_group_decoded_bytes_from_footer(
        metadata,
        &source.schema,
        &source.read_columns,
        &source.read_column_file_indices,
    )
}

fn decoded_column_bytes(
    data_type: Option<&ArrowDataType>,
    nullable: bool,
    rows: u64,
    leaf_chunks: &[&ColumnChunkMetaData],
) -> Option<u64> {
    let mask_bytes = if nullable { null_mask_allocation_size_bytes(rows) } else { 0 };
    if let Some(width) = data_type.and_then(arrow_fixed_width_byte_size) {
        return rows.checked_mul(width)?.checked_add(mask_bytes);
    }
    let payload = leaf_chunks.iter().try_fold(0u64, |total, chunk| {
        let leaf_payload = match chunk.unencoded_byte_array_data_bytes() {
            Some(bytes) => u64::try_from(bytes).ok()?,
            None => u64::try_from(chunk.uncompressed_size())
                .ok()?
                .max(rows.saturating_mul(COLD_VARIABLE_WIDTH_PAYLOAD_ROW_BYTES)),
        };
        total.checked_add(leaf_payload)
    })?;
    rows.checked_mul(VARIABLE_WIDTH_OFFSET_ROW_BYTES)?.checked_add(payload)?.checked_add(mask_bytes)
}
