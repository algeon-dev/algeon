//! Tests for high-level local Parquet source execution.

use std::process::Command;

use cudf::{
    BoundaryKind, DataFrame, DataType, ErrorCode, TypeId,
    core::{Column, Table},
    io::source::{
        LocalParquetFile, LocalParquetSource, PartitionColumn, PartitionValue, SourceColumn, SourceColumnRef,
        SourceColumnRole, SourceOutputColumn, SourceReadMode, SourceRequiredColumn, SourceSchema,
        ZERO_COLUMN_COUNT_COLUMN, plan_local_parquet_source_chunks,
    },
    runtime::QueryStageKind,
};
use tempfile::TempDir;

use crate::common::{column_from_slice, error_context_value, sample_runtime};

fn dtype(id: TypeId) -> DataType {
    DataType::new(id)
}

fn write_parquet_df(path: &std::path::Path, column_names: &[&str], columns: Vec<Column>, row_group_size_rows: i32) {
    let table = Table::from_columns(columns).unwrap();
    let write_opts = cudf::io::parquet::ParquetTableWriteOptions::builder()
        .filepath(path)
        .column_names(column_names)
        .row_group_size_rows(row_group_size_rows)
        .build();
    cudf::io::parquet::write_parquet(&table, &write_opts).unwrap();
}

fn write_pyarrow_i32_parquet_with_row_groups(path: &std::path::Path, values: &[i32], row_group_size: usize) -> bool {
    let vals = values.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
    let script = format!(
        "import pyarrow as pa, pyarrow.parquet as pq; \
         t = pa.table({{'id': pa.array([{}], type=pa.int32())}}); \
         pq.write_table(t, '{}', row_group_size={})",
        vals,
        path.display(),
        row_group_size
    );
    match Command::new("python3").args(["-c", &script]).output() {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            eprintln!("PyArrow row-group fixture write failed: {}", String::from_utf8_lossy(&output.stderr));
            false
        },
        Err(_) => false,
    }
}

#[test]
fn read_parquet_source_single_file_projects_and_records_boundary() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("source.parquet");
    let rt = sample_runtime();

    write_parquet_df(
        &path,
        &["id", "value", "filter_only"],
        vec![
            column_from_slice(&[1i32, 2, 3]).unwrap(),
            column_from_slice(&[10i64, 20, 30]).unwrap(),
            column_from_slice(&[7i32, 8, 9]).unwrap(),
        ],
        0,
    );

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "part-0", []).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("id_alias", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("filter_only", dtype(TypeId::Int32), true).unwrap(),
        ])
        .unwrap(),
        ["id", "filter_only"],
        [SourceOutputColumn::new("id_alias", SourceColumnRef::ReadColumn(0)).unwrap()],
        [SourceRequiredColumn::new("filter_only", SourceColumnRef::ReadColumn(1), SourceColumnRole::PureFilter)
            .unwrap()],
        [],
    )
    .unwrap();

    cudf::testing::boundary::reset_boundary_observation_for_test();
    let final_frame = DataFrame::read_parquet_source(&rt, &source).unwrap();
    let observation = cudf::testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(final_frame.column_names(), vec!["id_alias"]);
    assert_eq!(final_frame.num_rows(), 3);
    assert_eq!(final_frame.series("id_alias").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(observation.kind, BoundaryKind::HostToDevice);
    assert_eq!(observation.query_stage, Some(QueryStageKind::SourceRead));
    assert_eq!(observation.rows, 3);
    assert_eq!(observation.columns, 1);

    let execution_frame =
        DataFrame::read_parquet_source_with_mode(&rt, &source, SourceReadMode::ExecutionFrame).unwrap();
    assert_eq!(execution_frame.column_names(), vec!["id_alias", "filter_only"]);
    assert_eq!(execution_frame.series("filter_only").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9]);
}

#[test]
fn read_parquet_source_materializes_string_partition_columns_across_files() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("first.parquet");
    let second = dir.path().join("second.parquet");
    let rt = sample_runtime();

    write_parquet_df(&first, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()], 0);
    write_parquet_df(&second, &["id"], vec![column_from_slice(&[3i32]).unwrap()], 0);

    let source = LocalParquetSource::data(
        [
            LocalParquetFile::new(&first, "first", [PartitionValue::String("AMER".to_string())]).unwrap(),
            LocalParquetFile::new(&second, "second", [PartitionValue::String("APAC".to_string())]).unwrap(),
        ],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("region", dtype(TypeId::String), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("region", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("region", dtype(TypeId::String), false).unwrap()],
    )
    .unwrap();

    let df = DataFrame::read_parquet_source(&rt, &source).unwrap();

    assert_eq!(df.column_names(), vec!["id", "region"]);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(
        df.series("region").unwrap().to_strings_strict().unwrap(),
        vec!["AMER".to_string(), "AMER".to_string(), "APAC".to_string()]
    );
}

#[test]
fn read_parquet_source_chunk_honors_selected_row_groups() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("row_groups.parquet");
    let rt = sample_runtime();

    if !write_pyarrow_i32_parquet_with_row_groups(&path, &[1, 2, 3, 4], 2) {
        return;
    }

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "rg", [])
            .unwrap()
            .with_row_groups(Some(vec![1]))
            .with_row_group_metadata(vec![128, 128], vec![2, 2])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();

    let df =
        DataFrame::read_parquet_source_chunk(&rt, &source, &plan.chunks()[0], SourceReadMode::FinalOutput).unwrap();

    assert_eq!(df.column_names(), vec!["id"]);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![3, 4]);
}

#[test]
fn read_parquet_source_empty_row_group_selection_returns_typed_empty_output() {
    let rt = sample_runtime();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("does-not-need-to-exist.parquet", "empty", []).unwrap().with_row_groups(Some(vec![]))],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("filter_only", dtype(TypeId::Int64), true).unwrap(),
        ])
        .unwrap(),
        ["id", "filter_only"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [SourceRequiredColumn::new("filter_only", SourceColumnRef::ReadColumn(1), SourceColumnRole::PureFilter)
            .unwrap()],
        [],
    )
    .unwrap();

    let final_frame = DataFrame::read_parquet_source(&rt, &source).unwrap();
    assert_eq!(final_frame.column_names(), vec!["id"]);
    assert_eq!(final_frame.num_rows(), 0);
    assert_eq!(final_frame.series("id").unwrap().dtype().unwrap(), dtype(TypeId::Int32));

    let execution_frame =
        DataFrame::read_parquet_source_with_mode(&rt, &source, SourceReadMode::ExecutionFrame).unwrap();
    assert_eq!(execution_frame.column_names(), vec!["id", "filter_only"]);
    assert_eq!(execution_frame.num_rows(), 0);
    assert_eq!(execution_frame.series("filter_only").unwrap().dtype().unwrap(), dtype(TypeId::Int64));
}

#[test]
fn read_parquet_zero_column_count_uses_metadata_without_reading_file() {
    let rt = sample_runtime();
    let source = LocalParquetSource::zero_column_count([
        LocalParquetFile::new("missing-count-a.parquet", "count-a", [])
            .unwrap()
            .with_row_group_metadata(vec![128, 256], vec![2, 3])
            .unwrap(),
        LocalParquetFile::new("missing-count-b.parquet", "count-b", [])
            .unwrap()
            .with_row_group_metadata(vec![64], vec![7])
            .unwrap(),
    ])
    .unwrap();

    let df = DataFrame::read_parquet_source(&rt, &source).unwrap();

    assert_eq!(df.column_names(), vec![ZERO_COLUMN_COUNT_COLUMN]);
    assert_eq!(df.series(ZERO_COLUMN_COUNT_COLUMN).unwrap().to_vec_strict::<i64>().unwrap(), vec![5, 7]);
    assert_eq!(df.series(ZERO_COLUMN_COUNT_COLUMN).unwrap().sum_value::<i64>().unwrap(), 12);
}

#[test]
fn read_parquet_zero_column_count_empty_selection_returns_zero_row_count_frame() {
    let rt = sample_runtime();
    let source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("empty-count.parquet", "empty-count", [])
            .unwrap()
            .with_row_groups(Some(vec![]))])
        .unwrap();

    let df = DataFrame::read_parquet_source(&rt, &source).unwrap();

    assert_eq!(df.column_names(), vec![ZERO_COLUMN_COUNT_COLUMN]);
    assert_eq!(df.num_rows(), 0);
    assert_eq!(df.series(ZERO_COLUMN_COUNT_COLUMN).unwrap().dtype().unwrap(), dtype(TypeId::Int64));
}

#[test]
fn read_parquet_zero_column_count_requires_metadata_and_checked_i64_counts() {
    let rt = sample_runtime();
    let missing_source =
        LocalParquetSource::zero_column_count([
            LocalParquetFile::new("missing-metadata.parquet", "missing", []).unwrap()
        ])
        .unwrap();
    let missing_err = DataFrame::read_parquet_source(&rt, &missing_source).unwrap_err();
    assert_eq!(missing_err.code(), ErrorCode::ForeignInvalidArgument);
    assert_eq!(error_context_value(&missing_err, "reason").as_deref(), Some("missing_row_group_row_count_metadata"));

    let overflow_source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("overflow-count.parquet", "overflow", [])
            .unwrap()
            .with_row_group_metadata(vec![1], vec![i64::MAX as u64 + 1])
            .unwrap()])
        .unwrap();
    let overflow_err = DataFrame::read_parquet_source(&rt, &overflow_source).unwrap_err();
    assert_eq!(overflow_err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(error_context_value(&overflow_err, "reason").as_deref(), Some("row_count_exceeds_i64"));
}
