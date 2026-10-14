//! Shared Parquet footer/statistics helpers used by native pruning and sizing.

use std::collections::BTreeSet;

use parquet::{
    basic::SortOrder,
    file::{
        metadata::{ColumnChunkMetaData, ParquetMetaData, RowGroupMetaData},
        statistics::Statistics,
    },
    schema::types::SchemaDescriptor,
};

/// Returns the leaf column indexes of a top-level root column, in footer
/// order. A primitive root has one leaf; a nested root has one per primitive
/// field beneath it.
pub(crate) fn root_column_leaf_indices(schema: &SchemaDescriptor, root_index: usize) -> Vec<usize> {
    (0..schema.num_columns()).filter(|leaf_index| schema.get_column_root_idx(*leaf_index) == root_index).collect()
}

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

/// Returns the column chunk metadata for a top-level primitive root column represented by one leaf.
pub fn leaf_column_chunk<'a>(
    row_group: &'a RowGroupMetaData,
    schema: &SchemaDescriptor,
    root_column_index: usize,
) -> Option<&'a ColumnChunkMetaData> {
    let leaf_column_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    row_group.columns().get(leaf_column_index)
}

/// Returns statistics for a top-level primitive root column represented by one leaf.
pub fn leaf_column_statistics<'a>(
    row_group: &'a RowGroupMetaData,
    schema: &SchemaDescriptor,
    root_column_index: usize,
) -> Option<&'a Statistics> {
    leaf_column_chunk(row_group, schema, root_column_index)?.statistics()
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

/// Inclusive integer range decoded from Parquet Int32/Int64 statistics, wide
/// enough to hold both the signed and unsigned physical-slot domains.
///
/// Parquet stores `Integer{is_signed:false}` / converted `UINT_8/16/32/64`
/// column min/max in the same physical INT32/INT64 statistics slot as signed
/// values, ordered by unsigned comparison. Decoding that slot as plain `i32`
/// / `i64` sign-extends values above the signed range (e.g. a `UInt32` max of
/// `3_000_000_000` decodes as a negative `i32`). `i128` is wide enough to hold
/// an unsigned `u64` without truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerStatisticsRange {
    /// Minimum integer value recorded by Parquet statistics.
    pub min: i128,
    /// Maximum integer value recorded by Parquet statistics.
    pub max: i128,
}

/// Decodes Int32/Int64 Parquet statistics into an inclusive range, reading
/// unsigned-sorted columns as unsigned. Other physical types return `None`.
pub fn integer_statistics_range(column: &ColumnChunkMetaData) -> Option<IntegerStatisticsRange> {
    let statistics = column.statistics()?;
    let unsigned = column.column_descr().sort_order() == SortOrder::UNSIGNED;
    match statistics {
        Statistics::Int32(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            Some(if unsigned {
                IntegerStatisticsRange { min: i128::from(min as u32), max: i128::from(max as u32) }
            } else {
                IntegerStatisticsRange { min: i128::from(min), max: i128::from(max) }
            })
        },
        Statistics::Int64(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            Some(if unsigned {
                IntegerStatisticsRange { min: i128::from(min as u64), max: i128::from(max as u64) }
            } else {
                IntegerStatisticsRange { min: i128::from(min), max: i128::from(max) }
            })
        },
        _ => None,
    }
}

/// Inclusive `i64` range decoded from integer Parquet statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I64StatisticsRange {
    /// Minimum integer value recorded by Parquet statistics.
    pub min: i64,
    /// Maximum integer value recorded by Parquet statistics.
    pub max: i64,
}

/// Decodes Int32/Int64 statistics for a leaf column in a row group, narrowed
/// to `i64`. Returns `None` (unknown stats; callers retain the row group)
/// when the logical-type-aware range does not fit `i64`, e.g. a `UINT_64`
/// column with a value above `i64::MAX`.
pub(crate) fn row_group_i64_range(
    row_group: &RowGroupMetaData,
    leaf_column_index: usize,
) -> Option<I64StatisticsRange> {
    let range = integer_statistics_range(row_group.columns().get(leaf_column_index)?)?;
    Some(I64StatisticsRange { min: i64::try_from(range.min).ok()?, max: i64::try_from(range.max).ok()? })
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

    use parquet::{
        file::{
            metadata::{ColumnChunkMetaData, RowGroupMetaData},
            statistics::Statistics,
        },
        schema::{parser::parse_message_type, types::SchemaDescriptor},
    };

    use super::{I64StatisticsRange, row_group_i64_range, single_leaf_primitive_root_column_index};

    fn row_group_with_statistics(schema_body: &str, statistics: Statistics) -> RowGroupMetaData {
        let parquet_type = parse_message_type(schema_body).expect("test Parquet schema should parse");
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(parquet_type)));
        let column = ColumnChunkMetaData::builder(Arc::clone(&schema.columns()[0]))
            .set_statistics(statistics)
            .build()
            .expect("test column metadata should build");
        RowGroupMetaData::builder(Arc::clone(&schema))
            .set_num_rows(2)
            .add_column_metadata(column)
            .build()
            .expect("test row-group metadata should build")
    }

    #[test]
    fn row_group_i64_range_decodes_unsigned_int32_statistics_as_unsigned() {
        let row_group = row_group_with_statistics(
            "message schema { required int32 k (INTEGER(32,false)); }",
            Statistics::int32(Some(100), Some(3_000_000_000u32 as i32), None, None, false),
        );
        let range = row_group_i64_range(&row_group, 0).expect("unsigned int32 stats should decode");
        assert_eq!(range, I64StatisticsRange { min: 100, max: 3_000_000_000 });
    }

    #[test]
    fn row_group_i64_range_is_none_for_unsigned_int64_beyond_i64_max() {
        let row_group = row_group_with_statistics(
            "message schema { required int64 k (INTEGER(64,false)); }",
            Statistics::int64(Some(0), Some(u64::MAX as i64), None, None, false),
        );
        assert_eq!(row_group_i64_range(&row_group, 0), None);
    }

    #[test]
    fn row_group_i64_range_keeps_signed_int32_statistics_unchanged() {
        let row_group = row_group_with_statistics(
            "message schema { required int32 k; }",
            Statistics::int32(Some(-5), Some(42), None, None, false),
        );
        let range = row_group_i64_range(&row_group, 0).expect("signed int32 stats should decode");
        assert_eq!(range, I64StatisticsRange { min: -5, max: 42 });
    }

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
    fn single_leaf_mapping_refuses_list_roots_with_one_required_leaf() {
        let parquet_type = parse_message_type(
            "
            message schema {
              optional group items (LIST) {
                repeated group list {
                  required int32 element;
                }
              }
              optional int32 fact_key;
            }
            ",
        )
        .expect("test Parquet schema should parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));

        // The list has one leaf whose own null count is zero while the nullable
        // parent still holds nulls, so its footer cannot prove non-null rows.
        assert_eq!(schema.num_columns(), 2);
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 0), None);
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 1), Some(1));
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
