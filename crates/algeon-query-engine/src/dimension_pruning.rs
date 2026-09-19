//! Local-filesystem implementation of the planning dimension-pruning resolver contract.
//!
//! `query-planning` proves candidates, admission limits, and row-group
//! selections without ever opening a file; this module supplies the local
//! Parquet footer and row-batch reads those proofs ask for. The equivalent
//! object-store reads live in the DataFusion adapter behind
//! [`optimizer::LakehouseFooterPruningResolver`].

use std::fs::File;

use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::reader::{FileReader, SerializedFileReader},
};
use query_planning::{
    expr::Expr,
    optimizer::{
        DimensionKeyAccumulator, DimensionKeyFilter, I64ExprRangeAccumulator, KeyRange, LocalParquetPruningResolver,
    },
    parquet_stats::row_group_i64_range_for_root_column,
    source::{LocalParquetFile, LocalParquetSource},
};

/// Local-filesystem [`LocalParquetPruningResolver`] used by the native engine
/// and the DataFusion adapter for local Parquet sources.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalDimensionPruningResolver;

/// The resolver the native engine and adapter use for local Parquet sources.
pub const fn local_dimension_pruning_resolver() -> &'static LocalDimensionPruningResolver {
    &LocalDimensionPruningResolver
}

impl LocalParquetPruningResolver for LocalDimensionPruningResolver {
    fn dimension_scan_admitted(&self, source: &LocalParquetSource, max_dimension_rows: u64) -> bool {
        let mut total_rows = 0u64;
        for file in &source.files {
            let Some(path) = file.location.as_local_path() else {
                return false;
            };
            let Ok(reader) = SerializedFileReader::try_from(path) else {
                return false;
            };
            let metadata = reader.metadata();
            let row_groups = selected_or_all_row_groups(file, metadata.num_row_groups());
            for row_group in row_groups {
                let Some(row_group_rows) = metadata.row_group(row_group).num_rows().try_into().ok() else {
                    return false;
                };
                total_rows = total_rows.saturating_add(row_group_rows);
                if total_rows > max_dimension_rows {
                    return false;
                }
            }
        }
        true
    }

    fn dimension_key_filter(
        &self,
        source: &LocalParquetSource,
        key_column: &str,
        predicate: &Expr,
        projection_indices: &[usize],
    ) -> Option<Option<DimensionKeyFilter>> {
        let mut accumulator = DimensionKeyAccumulator::default();
        for file in &source.files {
            let path = file.location.as_local_path()?;
            let opened = File::open(path).ok()?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(opened).ok()?;
            let row_group_count = builder.metadata().num_row_groups();
            let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices.iter().copied());
            let row_groups = selected_or_all_row_groups(file, row_group_count);
            let reader = builder.with_projection(mask).with_row_groups(row_groups).build().ok()?;
            for batch in reader {
                let batch = batch.ok()?;
                accumulator.observe_batch(&batch, key_column, predicate)?;
            }
        }
        Some(accumulator.finish())
    }

    fn target_row_group_key_ranges(
        &self,
        source: &LocalParquetSource,
        key_column_index: usize,
    ) -> Option<Vec<Vec<Option<KeyRange>>>> {
        let mut per_file_row_group_key_ranges = Vec::with_capacity(source.files.len());
        for file in &source.files {
            let path = file.location.as_local_path()?;
            let reader = SerializedFileReader::try_from(path).ok()?;
            let metadata = reader.metadata();
            let schema = metadata.file_metadata().schema_descr();
            let selected_row_groups = selected_or_all_row_groups(file, metadata.num_row_groups());
            let ranges = selected_row_groups
                .into_iter()
                .map(|row_group_index| {
                    row_group_i64_range_for_root_column(metadata.row_group(row_group_index), schema, key_column_index)
                        .map(|range| KeyRange { min: range.min, max: range.max })
                })
                .collect();
            per_file_row_group_key_ranges.push(ranges);
        }
        Some(per_file_row_group_key_ranges)
    }

    fn filtered_i64_expr_range(
        &self,
        source: &LocalParquetSource,
        expr: &Expr,
        predicate: Option<&Expr>,
        projection_indices: &[usize],
    ) -> Option<Option<KeyRange>> {
        let mut accumulator = I64ExprRangeAccumulator::default();
        for file in &source.files {
            let path = file.location.as_local_path()?;
            let opened = File::open(path).ok()?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(opened).ok()?;
            let row_group_count = builder.metadata().num_row_groups();
            let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices.iter().copied());
            let row_groups = selected_or_all_row_groups(file, row_group_count);
            let reader = builder.with_projection(mask).with_row_groups(row_groups).build().ok()?;
            for batch in reader {
                let batch = batch.ok()?;
                accumulator.observe_batch(&batch, expr, predicate)?;
            }
        }
        Some(accumulator.finish())
    }
}

// Matches `query_planning::optimizer::dimension_pruning`'s local
// selection rule: an explicit but empty `row_groups` is treated the same as
// "unset" (every physical row group), not as an empty selection.
fn selected_or_all_row_groups(file: &LocalParquetFile, row_group_count: usize) -> Vec<usize> {
    if let Some(row_groups) = &file.row_groups
        && !row_groups.is_empty()
    {
        return row_groups.clone();
    }
    (0..row_group_count).collect()
}
