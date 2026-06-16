//! Shared Parquet footer/statistics helpers used by native pruning and sizing.

use std::collections::{BTreeMap, BTreeSet};

use parquet::{
    file::{
        metadata::{ParquetMetaData, RowGroupMetaData},
        statistics::Statistics,
    },
    schema::types::SchemaDescriptor,
};

/// Returns the only primitive leaf column for a top-level root column.
///
/// Nested roots and roots that map to zero or multiple leaves return `None`.
pub fn single_leaf_primitive_root_column_index(schema: &SchemaDescriptor, root_index: usize) -> Option<usize> {
    if !schema.root_schema().get_fields().get(root_index)?.is_primitive() {
        return None;
    }
    let mut leaf_indices =
        (0..schema.num_columns()).filter(|leaf_index| schema.get_column_root_idx(*leaf_index) == root_index);
    let leaf_index = leaf_indices.next()?;
    leaf_indices.next().is_none().then_some(leaf_index)
}

/// Returns statistics for a top-level primitive root column represented by one leaf.
pub fn leaf_column_statistics<'a>(
    row_group: &'a RowGroupMetaData,
    schema: &SchemaDescriptor,
    root_column_index: usize,
) -> Option<&'a Statistics> {
    let leaf_column_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    row_group.columns().get(leaf_column_index)?.statistics()
}

/// Which encoded byte count to read from a Parquet column chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnByteKind {
    /// Compressed bytes reported by the Parquet column metadata.
    Compressed,
    /// Uncompressed bytes reported by the Parquet column metadata.
    Uncompressed,
}

/// Sums compressed or uncompressed bytes for the requested leaf column indexes.
pub fn row_group_column_bytes(
    row_group: &RowGroupMetaData,
    column_indexes: &BTreeSet<usize>,
    kind: ColumnByteKind,
) -> Option<u64> {
    column_indexes.iter().try_fold(0u64, |total, column_index| {
        let column = row_group.columns().get(*column_index)?;
        let bytes = match kind {
            ColumnByteKind::Compressed => column.compressed_size(),
            ColumnByteKind::Uncompressed => column.uncompressed_size(),
        };
        let bytes = u64::try_from(bytes).ok()?;
        total.checked_add(bytes)
    })
}

/// Returns per-row-group null counts for a primitive root represented by one leaf.
pub fn row_group_null_counts_for_root_column(
    metadata: &ParquetMetaData,
    root_column_index: usize,
) -> Option<Vec<Option<u64>>> {
    let schema = metadata.file_metadata().schema_descr();
    let leaf_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    Some(
        metadata
            .row_groups()
            .iter()
            .map(|row_group| {
                row_group
                    .columns()
                    .get(leaf_index)
                    .and_then(|column| column.statistics())
                    .and_then(|statistics| statistics.null_count_opt())
            })
            .collect(),
    )
}

/// Returns per-row-group null counts for all primitive roots represented by one leaf.
pub fn row_group_null_counts_by_root_column(metadata: &ParquetMetaData) -> BTreeMap<usize, Vec<Option<u64>>> {
    let schema = metadata.file_metadata().schema_descr();
    let root_field_count = schema.root_schema().get_fields().len();
    (0..root_field_count)
        .filter_map(|root_index| Some((root_index, row_group_null_counts_for_root_column(metadata, root_index)?)))
        .collect()
}

/// Inclusive `i64` range decoded from integer Parquet statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I64StatisticsRange {
    /// Minimum integer value recorded by Parquet statistics.
    pub min: i64,
    /// Maximum integer value recorded by Parquet statistics.
    pub max: i64,
}

/// Decodes Int32/Int64 Parquet statistics into an inclusive `i64` range.
pub fn statistics_i64_range(statistics: &Statistics) -> Option<I64StatisticsRange> {
    match statistics {
        Statistics::Int32(stats) => {
            Some(I64StatisticsRange { min: i64::from(*stats.min_opt()?), max: i64::from(*stats.max_opt()?) })
        },
        Statistics::Int64(stats) => Some(I64StatisticsRange { min: *stats.min_opt()?, max: *stats.max_opt()? }),
        _ => None,
    }
}

/// Decodes Int32/Int64 statistics for a leaf column in a row group.
pub fn row_group_i64_range(row_group: &RowGroupMetaData, leaf_column_index: usize) -> Option<I64StatisticsRange> {
    statistics_i64_range(row_group.columns().get(leaf_column_index)?.statistics()?)
}

/// Decodes Int32/Int64 statistics for a primitive root represented by one leaf.
pub fn row_group_i64_range_for_root_column(
    row_group: &RowGroupMetaData,
    schema: &SchemaDescriptor,
    root_column_index: usize,
) -> Option<I64StatisticsRange> {
    let leaf_column_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    row_group_i64_range(row_group, leaf_column_index)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parquet::schema::{parser::parse_message_type, types::SchemaDescriptor};

    use super::single_leaf_primitive_root_column_index;

    #[test]
    fn single_leaf_mapping_refuses_nested_roots_and_maps_later_primitives() {
        let parquet_type = parse_message_type(
            "
            message schema {
              optional group payload {
                optional int32 x;
                optional int32 y;
              }
              optional int32 fact_key;
            }
            ",
        )
        .expect("test Parquet schema should parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));

        assert_eq!(single_leaf_primitive_root_column_index(&schema, 0), None);
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 1), Some(2));
    }

    #[test]
    fn single_leaf_mapping_uses_parquet_root_indexes_not_names() {
        let parquet_type = parse_message_type(
            "
            message schema {
              optional int32 duplicate_name;
              optional int64 duplicate_name;
            }
            ",
        )
        .expect("test Parquet schema should parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));

        assert_eq!(single_leaf_primitive_root_column_index(&schema, 0), Some(0));
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 1), Some(1));
    }
}
