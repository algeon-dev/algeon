//! Tests for high-level local Parquet source execution.

use std::{collections::BTreeMap, fs, process::Command, sync::Arc};

use arrow::{
    array::{Array, AsArray, Int32Array, Int64Array, LargeListArray, ListArray, new_null_array},
    datatypes::{
        DataType as ArrowDataType, Date32Type, Decimal128Type, DurationNanosecondType, Field, Int32Type, Int64Type,
        Schema, TimeUnit, TimestampMicrosecondType,
    },
    record_batch::RecordBatch,
};
use cudf::{
    BoundaryKind, DataFrame, DataType, ErrorCode, ExecutionBudgetEnforcement, ExecutionBudgetRequest,
    TimestampResolution, TypeId,
    core::{Column, Table},
    frame::dataframe::{GpuRecordBatch, InfoDetail, ParquetSourceChunkReadOutput},
    io::{
        parquet::{ParquetTableWriteOptions, RemoteParquetProfile, RemoteReadMode, write_parquet},
        source::{
            LocalParquetFile, LocalParquetSource, PartitionColumn, PartitionValue, PredicateDisposition, SourceChunk,
            SourceChunkPlanningPolicy, SourceColumn, SourceColumnRef, SourceColumnRole, SourceOutputColumn,
            SourceReadMode, SourceReaderPredicate, SourceReaderPredicateColumn, SourceReaderPredicateLiteral,
            SourceRequiredColumn, SourceSchema, ZERO_COLUMN_COUNT_COLUMN, build_local_parquet_reader_batch_predicate,
            build_local_parquet_reader_predicate, plan_local_parquet_source_chunks,
        },
    },
    ops::ast::AstExpression,
    runtime::{QueryStageKind, Runtime},
    testing::{
        boundary::{last_boundary_observation_for_test, reset_boundary_observation_for_test},
        prelude::*,
    },
};
use parquet::{
    arrow::ArrowWriter,
    file::properties::{EnabledStatistics, WriterProperties},
};
use tempfile::TempDir;

use crate::common::{
    column_from_slice, column_from_strs, error_context_value, last_exec_mr_value, reset_exec_observation,
    sample_runtime,
};

fn dtype(id: TypeId) -> DataType {
    DataType::new(id)
}

fn dataframe_device_allocated_size_bytes(df: &DataFrame) -> usize {
    df.info_with(InfoDetail::DEVICE_ALLOCATED).unwrap().memory_accounting.unwrap().device_allocated_bytes.unwrap()
}

fn assert_budget_report_high_water(report: &cudf::ExecutionBudgetReport, retained_or_output_bytes: usize) {
    let high_water = report.allocator_high_water_bytes().expect("budget report should include allocator high-water");
    assert!(high_water > 0, "allocator high-water should be positive: {report:?}");
    assert!(
        high_water >= retained_or_output_bytes,
        "allocator high-water should cover retained/output bytes: high_water={high_water}, retained_or_output={retained_or_output_bytes}, report={report:?}",
    );
    assert!(
        high_water <= report.request().budget_bytes(),
        "allocator high-water should stay within budget: high_water={high_water}, report={report:?}",
    );
    assert!(
        report.allocator_peak_aligned_bytes().is_some_and(|bytes| bytes <= report.request().budget_bytes()),
        "aligned allocator peak should stay within budget: {report:?}",
    );
}

#[test]
fn parquet_source_chunk_exact_preflight_reports_fixed_width_output_bound() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("/tmp/preflight-fixed.parquet", "preflight-fixed", [])
            .unwrap()
            .with_row_group_metadata(vec![1_024], vec![10])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();

    let preflight =
        DataFrame::parquet_source_chunk_exact_preflight(&source, &plan.chunks()[0], SourceReadMode::FinalOutput)
            .unwrap()
            .expect("fixed-width source with row-count metadata should have exact preflight");

    assert_eq!(preflight.retained_output_bytes(), 40);
    assert_eq!(preflight.required_budget_bytes(), DataFrame::minimum_parquet_source_read_budget_bytes());
}

#[test]
fn parquet_source_chunk_exact_preflight_returns_none_for_variable_width_output() {
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("/tmp/preflight-string.parquet", "preflight-string", [])
            .unwrap()
            .with_row_group_metadata(vec![1_024], vec![10])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("name", dtype(TypeId::String), true).unwrap()]).unwrap(),
        ["name"],
        [SourceOutputColumn::new("name", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();

    let preflight =
        DataFrame::parquet_source_chunk_exact_preflight(&source, &plan.chunks()[0], SourceReadMode::FinalOutput)
            .unwrap();

    assert_eq!(preflight, None);
}

fn write_parquet_df(path: &std::path::Path, column_names: &[&str], columns: Vec<Column>, row_group_size_rows: i32) {
    let table = Table::from_columns(columns).unwrap();
    let write_opts = ParquetTableWriteOptions::builder()
        .filepath(path)
        .column_names(column_names)
        .row_group_size_rows(row_group_size_rows)
        .build();
    write_parquet(&table, &write_opts).unwrap();
}

fn write_decimal_parquet_df(path: &std::path::Path, storage_type: TypeId) {
    let rt = sample_runtime();
    let dtype = DataType::fixed_point(storage_type, -2).unwrap();
    let amount = rt.series(&[-5i64, 0, 1, 2]).unwrap().astype(dtype).unwrap();
    DataFrame::from_series([("amount", amount)]).unwrap().write_parquet(path).unwrap();
}

fn decimal_source(path: &std::path::Path) -> LocalParquetSource {
    decimal_source_from_paths([path])
}

fn decimal_source_from_paths<'a>(paths: impl IntoIterator<Item = &'a std::path::Path>) -> LocalParquetSource {
    let decimal128 = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();
    LocalParquetSource::data(
        paths
            .into_iter()
            .enumerate()
            .map(|(index, path)| LocalParquetFile::new(path, format!("decimal-{index}"), []).unwrap()),
        SourceSchema::new([SourceColumn::new("amount", decimal128, false).unwrap()]).unwrap(),
        ["amount"],
        [SourceOutputColumn::new("amount", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap()
}

fn decimal_predicate_source(
    rt: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: SourceReaderPredicate,
) -> LocalParquetSource {
    let predicate = build_local_parquet_reader_predicate(rt, source, chunk, &predicate).unwrap();
    source.clone().with_reader_predicate(std::sync::Arc::new(predicate)).unwrap()
}

fn assert_decimal_amount_raws(df: &DataFrame, expected: &[i128]) {
    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Decimal128(38, 2));
    let amount = batch.column(0).as_primitive::<Decimal128Type>();
    assert_eq!(amount.scale(), 2);
    assert_eq!(amount.values().as_ref(), expected);
}

fn write_i32_parquet_with_row_groups(path: &std::path::Path, values: &[i32], row_group_size: usize) -> bool {
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

fn single_i32_source(path: &std::path::Path, stable_id: &str) -> LocalParquetSource {
    LocalParquetSource::data(
        [LocalParquetFile::new(path, stable_id, []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap()
}

fn read_single_chunk_with_evidence(rt: &Runtime, source: &LocalParquetSource) -> ParquetSourceChunkReadOutput {
    let plan = plan_local_parquet_source_chunks(source, Default::default()).unwrap();
    assert_eq!(plan.chunks().len(), 1);
    DataFrame::read_parquet_source_chunk_with_evidence(rt, source, &plan.chunks()[0], SourceReadMode::FinalOutput)
        .unwrap()
}

#[test]
fn read_parquet_source_materializes_typed_missing_list_columns() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("typed_missing_lists.parquet");
    let physical_schema = Arc::new(Schema::new(vec![Field::new("id", ArrowDataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&physical_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    let file = fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, physical_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let boolean_list = ArrowDataType::List(Arc::new(Field::new("flag", ArrowDataType::Boolean, true)));
    let integer_list = ArrowDataType::List(Arc::new(Field::new("port", ArrowDataType::Int64, false)));
    let string_list = ArrowDataType::LargeList(Arc::new(Field::new("tag", ArrowDataType::Utf8, true)));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int32, false),
        Field::new("flags", boolean_list.clone(), true),
        Field::new("ports", integer_list.clone(), true),
        Field::new("tags", string_list.clone(), true),
    ]));
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "typed-missing-lists", []).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new_list("flags", boolean_list, true).unwrap(),
            SourceColumn::new_list("ports", integer_list, true).unwrap(),
            SourceColumn::new_list("tags", string_list, true).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("flags", SourceColumnRef::MissingNull).unwrap(),
            SourceOutputColumn::new("ports", SourceColumnRef::MissingNull).unwrap(),
            SourceOutputColumn::new("tags", SourceColumnRef::MissingNull).unwrap(),
        ],
        [],
        [],
    )
    .unwrap();
    let output = read_single_chunk_with_evidence(&sample_runtime(), &source);
    let host =
        GpuRecordBatch::new(output.into_frame(), Arc::clone(&output_schema)).unwrap().to_host_record_batch().unwrap();

    assert_eq!(host.schema(), output_schema);
    for column in 1..=3 {
        assert_eq!(host.column(column).null_count(), 3);
    }
    assert_eq!(host.column(1).as_any().downcast_ref::<ListArray>().unwrap().values().len(), 0);
    assert_eq!(host.column(2).as_any().downcast_ref::<ListArray>().unwrap().values().len(), 0);
    assert_eq!(host.column(3).as_any().downcast_ref::<LargeListArray>().unwrap().values().len(), 0);
}

#[test]
fn read_parquet_source_materializes_typed_all_null_and_empty_list_columns() {
    let dir = TempDir::new().unwrap();
    let all_null_path = dir.path().join("typed_all_null_lists.parquet");
    let empty_path = dir.path().join("typed_empty_lists.parquet");
    let list_types = [
        ArrowDataType::List(Arc::new(Field::new("flag", ArrowDataType::Boolean, true))),
        ArrowDataType::List(Arc::new(Field::new("port", ArrowDataType::Int64, false))),
        ArrowDataType::LargeList(Arc::new(Field::new("tag", ArrowDataType::Utf8, true))),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("flags", list_types[0].clone(), true),
        Field::new("ports", list_types[1].clone(), true),
        Field::new("tags", list_types[2].clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        list_types.iter().map(|data_type| new_null_array(data_type, 3)).collect(),
    )
    .unwrap();
    let file = fs::File::create(&all_null_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let file = fs::File::create(&empty_path).unwrap();
    ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap().close().unwrap();

    let source = |path: &std::path::Path, stable_id: &str| {
        LocalParquetSource::data(
            [LocalParquetFile::new(path, stable_id, []).unwrap()],
            SourceSchema::new([
                SourceColumn::new_list("flags", list_types[0].clone(), true).unwrap(),
                SourceColumn::new_list("ports", list_types[1].clone(), true).unwrap(),
                SourceColumn::new_list("tags", list_types[2].clone(), true).unwrap(),
            ])
            .unwrap(),
            ["flags", "ports", "tags"],
            [
                SourceOutputColumn::new("flags", SourceColumnRef::ReadColumn(0)).unwrap(),
                SourceOutputColumn::new("ports", SourceColumnRef::ReadColumn(1)).unwrap(),
                SourceOutputColumn::new("tags", SourceColumnRef::ReadColumn(2)).unwrap(),
            ],
            [],
            [],
        )
        .unwrap()
    };
    let runtime = sample_runtime();
    let all_null = read_single_chunk_with_evidence(&runtime, &source(&all_null_path, "typed-all-null-lists"));
    let all_null =
        GpuRecordBatch::new(all_null.into_frame(), Arc::clone(&schema)).unwrap().to_host_record_batch().unwrap();
    assert_eq!(all_null.schema(), schema);
    assert_eq!(all_null.num_rows(), 3);
    assert!(all_null.columns().iter().all(|column| column.null_count() == 3));
    assert_eq!(all_null.column(0).as_any().downcast_ref::<ListArray>().unwrap().values().len(), 0);
    assert_eq!(all_null.column(1).as_any().downcast_ref::<ListArray>().unwrap().values().len(), 0);
    assert_eq!(all_null.column(2).as_any().downcast_ref::<LargeListArray>().unwrap().values().len(), 0);

    let empty = DataFrame::read_parquet_source(&runtime, &source(&empty_path, "typed-empty-lists")).unwrap();
    let empty = GpuRecordBatch::new(empty, Arc::clone(&schema)).unwrap().to_host_record_batch().unwrap();
    assert_eq!(empty.schema(), schema);
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty.column(0).as_any().downcast_ref::<ListArray>().unwrap().value_type(), ArrowDataType::Boolean);
    assert_eq!(empty.column(1).as_any().downcast_ref::<ListArray>().unwrap().value_type(), ArrowDataType::Int64);
    assert_eq!(empty.column(2).as_any().downcast_ref::<LargeListArray>().unwrap().value_type(), ArrowDataType::Utf8);
}

#[test]
fn read_parquet_source_chunk_reads_arrow_writer_all_null_optional_column_row_group() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("all_null_optional.parquet");
    let rt = sample_runtime();
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", ArrowDataType::Int64, false),
        Field::new("m00", ArrowDataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4])), Arc::new(Int64Array::from(vec![None, None, None, None]))],
    )
    .unwrap();
    // Without footer statistics, eliding `m00` can only come from the null
    // counts attached to the planned source file.
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    let file = fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "all-null-optional", [])
            .unwrap()
            .with_row_group_metadata(vec![88, 88], vec![2, 2])
            .unwrap()
            .with_row_group_null_counts(BTreeMap::from([("m00".to_string(), vec![Some(2), Some(2)])]))],
        SourceSchema::new([
            SourceColumn::new("payload", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("m00", dtype(TypeId::Int64), true).unwrap(),
        ])
        .unwrap(),
        ["payload", "m00"],
        [
            SourceOutputColumn::new("payload", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("m00", SourceColumnRef::ReadColumn(1)).unwrap(),
        ],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::new(None, 1).unwrap()).unwrap();

    assert_eq!(plan.chunks().len(), 2);
    let first =
        DataFrame::read_parquet_source_chunk(&rt, &source, &plan.chunks()[0], SourceReadMode::FinalOutput).unwrap();
    let second =
        DataFrame::read_parquet_source_chunk(&rt, &source, &plan.chunks()[1], SourceReadMode::FinalOutput).unwrap();

    assert_eq!(first.num_rows(), 2);
    assert_eq!(first.series("payload").unwrap().to_vec_strict::<i64>().unwrap(), vec![1, 2]);
    assert_eq!(first.series("m00").unwrap().null_count(), 2);
    assert_eq!(second.num_rows(), 2);
    assert_eq!(second.series("payload").unwrap().to_vec_strict::<i64>().unwrap(), vec![3, 4]);
    assert_eq!(second.series("m00").unwrap().null_count(), 2);

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("all_null_batch_read"))
        .unwrap();
    let batch = DataFrame::read_parquet_source_chunks_with_budget(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    assert_eq!(batch.value().num_rows(), 4);
    assert_eq!(batch.value().series("payload").unwrap().to_vec_strict::<i64>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(batch.value().series("m00").unwrap().null_count(), 4);
    drop(batch);
    drop(token);

    let null_only_source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "all-null-only", [])
            .unwrap()
            .with_row_group_metadata(vec![88, 88], vec![2, 2])
            .unwrap()
            .with_row_group_null_counts(BTreeMap::from([("m00".to_string(), vec![Some(2), Some(2)])]))],
        SourceSchema::new([SourceColumn::new("m00", dtype(TypeId::Int64), true).unwrap()]).unwrap(),
        ["m00"],
        [SourceOutputColumn::new("m00", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let null_only_plan =
        plan_local_parquet_source_chunks(&null_only_source, SourceChunkPlanningPolicy::new(None, 1).unwrap()).unwrap();
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("all_null_only_read"))
        .unwrap();
    let null_only = DataFrame::read_parquet_source_chunk_with_budget_with_evidence(
        &rt,
        &null_only_source,
        &null_only_plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();
    assert_eq!(null_only.value().frame().series("m00").unwrap().null_count(), 2);
    let evidence = null_only.value().read_evidence().unwrap();
    assert_eq!(evidence.input_kind, "local_path");
    assert_eq!(evidence.backend, "parquet_footer_materialized");
    drop(null_only);
    drop(token);

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("all_null_only_batch"))
        .unwrap();
    let null_only_batch = DataFrame::read_parquet_source_chunks_with_budget_with_evidence(
        &rt,
        &null_only_source,
        null_only_plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();
    assert_eq!(null_only_batch.value().frame().series("m00").unwrap().null_count(), 4);
    let evidence = null_only_batch.value().read_evidence().unwrap();
    assert_eq!(evidence.input_kind, "local_path_multi");
    assert_eq!(evidence.backend, "parquet_footer_materialized");
}

#[test]
fn read_parquet_source_chunks_handles_mixed_all_null_optional_column_row_groups() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mixed_all_null_optional.parquet");
    let rt = sample_runtime();
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", ArrowDataType::Int64, false),
        Field::new("m00", ArrowDataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![None, None, Some(30), Some(40)])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "mixed-all-null-optional", [])
            .unwrap()
            .with_row_group_metadata(vec![88, 88], vec![2, 2])
            .unwrap()
            .with_row_group_null_counts(BTreeMap::from([("m00".to_string(), vec![Some(2), Some(0)])]))],
        SourceSchema::new([
            SourceColumn::new("payload", dtype(TypeId::Int64), false).unwrap(),
            SourceColumn::new("m00", dtype(TypeId::Int64), true).unwrap(),
        ])
        .unwrap(),
        ["payload", "m00"],
        [
            SourceOutputColumn::new("payload", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("m00", SourceColumnRef::ReadColumn(1)).unwrap(),
        ],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::new(None, 1).unwrap()).unwrap();
    assert_eq!(plan.chunks().len(), 2);

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("mixed_all_null_batch"))
        .unwrap();
    let outcome = DataFrame::read_parquet_source_chunks_with_budget_with_evidence(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    let df = outcome.value().frame();
    assert_eq!(df.series("payload").unwrap().to_vec_strict::<i64>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(df.series("m00").unwrap().null_count(), 2);
    let batch = df.to_record_batch().unwrap();
    assert!(batch.column(1).is_null(0));
    assert!(batch.column(1).is_null(1));
    assert_eq!(&batch.column(1).as_primitive::<Int64Type>().values().as_ref()[2..], &[30, 40]);
    let evidence = outcome.value().read_evidence().unwrap();
    assert_eq!(evidence.input_kind, "local_path_multi");
    assert_eq!(evidence.backend, "parquet_footer_materialized");
}

#[test]
fn parquet_read_evidence_reports_no_predicate_pruning_absent() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no_predicate_pruning.parquet");
    let rt = sample_runtime();
    assert!(write_i32_parquet_with_row_groups(&path, &[1, 2, 3, 4], 2));

    let output = read_single_chunk_with_evidence(&rt, &single_i32_source(&path, "no-predicate-pruning"));
    let pruning = output.read_evidence().and_then(|evidence| evidence.pruning).unwrap();

    assert_eq!(output.frame().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(pruning.input_row_groups, 2);
    assert_eq!(pruning.row_groups_after_stats_filter, None);
    assert_eq!(pruning.row_groups_after_bloom_filter, None);
}

#[test]
fn parquet_read_evidence_reports_stats_pruned_row_groups() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("stats_pruning.parquet");
    let rt = sample_runtime();
    assert!(write_i32_parquet_with_row_groups(&path, &[1, 2, 100, 101, 200, 201], 2));

    let predicate = AstExpression::greater(
        AstExpression::left_column(0).unwrap(),
        AstExpression::literal(rt.scalar(150i32).unwrap()).unwrap(),
    )
    .unwrap();
    let source =
        single_i32_source(&path, "stats-pruning").with_reader_predicate(std::sync::Arc::new(predicate)).unwrap();
    let output = read_single_chunk_with_evidence(&rt, &source);
    let pruning = output.read_evidence().and_then(|evidence| evidence.pruning).unwrap();

    assert_eq!(output.frame().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![200, 201]);
    assert_eq!(pruning.input_row_groups, 3);
    assert!(pruning.row_groups_after_stats_filter.is_some_and(|after| after < pruning.input_row_groups));
}

#[test]
fn parquet_reader_predicate_bare_boolean_column_root_does_not_crash() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bare_boolean_column_root.parquet");
    let rt = sample_runtime();
    write_parquet_df(
        &path,
        &["id", "flag"],
        vec![
            column_from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap(),
            column_from_slice(&[true, false, true, false, true, false]).unwrap(),
        ],
        2,
    );
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "bare-boolean-column-root", []).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("flag", dtype(TypeId::Bool8), false).unwrap(),
        ])
        .unwrap(),
        ["id", "flag"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("flag", SourceColumnRef::ReadColumn(1)).unwrap(),
        ],
        [],
        [],
    )
    .unwrap()
    .with_reader_predicate(Arc::new(AstExpression::left_column(1).unwrap()))
    .unwrap();

    let output = read_single_chunk_with_evidence(&rt, &source);
    let pruning = output.read_evidence().and_then(|evidence| evidence.pruning).unwrap();

    assert_eq!(output.frame().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 5]);
    assert_eq!(output.frame().series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
    assert!(pruning.input_row_groups > 0);
    assert_eq!(pruning.row_groups_after_stats_filter, None);
}

#[test]
fn parquet_reader_predicate_bare_boolean_literal_root_does_not_crash() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bare_boolean_literal_root.parquet");
    let rt = sample_runtime();
    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2, 3]).unwrap()], 2);
    let source = single_i32_source(&path, "bare-boolean-literal-root")
        .with_reader_predicate(Arc::new(AstExpression::literal(rt.scalar(true).unwrap()).unwrap()))
        .unwrap();

    let output = read_single_chunk_with_evidence(&rt, &source);
    let pruning = output.read_evidence().and_then(|evidence| evidence.pruning).unwrap();

    assert_eq!(output.frame().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert!(pruning.input_row_groups > 0);
    assert_eq!(pruning.row_groups_after_stats_filter, None);
}

#[test]
fn parquet_reader_predicate_bare_boolean_false_literal_root_filters_all_rows() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bare_boolean_false_literal_root.parquet");
    let rt = sample_runtime();
    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2, 3]).unwrap()], 2);
    let source = single_i32_source(&path, "bare-boolean-false-literal-root")
        .with_reader_predicate(Arc::new(AstExpression::literal(rt.scalar(false).unwrap()).unwrap()))
        .unwrap();

    let output = read_single_chunk_with_evidence(&rt, &source);
    let pruning = output.read_evidence().and_then(|evidence| evidence.pruning).unwrap();

    assert_eq!(output.frame().num_rows(), 0);
    assert!(pruning.input_row_groups > 0);
    assert_eq!(pruning.row_groups_after_stats_filter, None);
}

#[test]
fn parquet_read_evidence_reports_bloom_pruned_row_groups() {
    let rt = sample_runtime();
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cudf/python/cudf/cudf/tests/data/parquet/mixed_card_ndv_100_bf_fpp0.1_nostats.snappy.parquet");
    let predicate = AstExpression::equal(
        AstExpression::left_column(0).unwrap(),
        AstExpression::literal(rt.string_scalar("FINDME").unwrap()).unwrap(),
    )
    .unwrap();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "bloom-pruning", []).unwrap()],
        SourceSchema::new([SourceColumn::new("str", dtype(TypeId::String), false).unwrap()]).unwrap(),
        ["str"],
        [SourceOutputColumn::new("str", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap()
    .with_reader_predicate(std::sync::Arc::new(predicate))
    .unwrap();

    let output = read_single_chunk_with_evidence(&rt, &source);
    let pruning = output.read_evidence().and_then(|evidence| evidence.pruning).unwrap();

    assert_eq!(output.frame().num_rows(), 2);
    assert_eq!(pruning.input_row_groups, 5);
    // The fixture carries no usable row-group statistics, so 26.08 skips the stats stage
    // (upstream #22664) and reports no survivor count for it.
    assert_eq!(pruning.row_groups_after_stats_filter, None);
    assert!(pruning.row_groups_after_bloom_filter.is_some_and(|after| after < pruning.input_row_groups));
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

    reset_boundary_observation_for_test();
    let final_frame = DataFrame::read_parquet_source(&rt, &source).unwrap();
    let observation = last_boundary_observation_for_test().unwrap();

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

    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let output = DataFrame::read_parquet_source_chunk_with_evidence(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
    )
    .unwrap();
    let evidence = output.read_evidence().expect("local path reads should report Parquet read evidence");
    assert_eq!(evidence.input_kind, "local_path");
    assert_eq!(evidence.requested_remote_read_mode, None);
    assert_eq!(evidence.backend, "libcudf_path_reader");
    assert_eq!(evidence.fallback_reason, "not_applicable");
}

#[test]
fn decimal_reader_predicate_matches_decimal32_backed_source_column() {
    assert_decimal_reader_predicate_matches_backed_source_column(TypeId::Decimal32);
}

#[test]
fn decimal_reader_predicate_matches_decimal64_backed_source_column() {
    assert_decimal_reader_predicate_matches_backed_source_column(TypeId::Decimal64);
}

#[test]
fn decimal_reader_predicate_matches_decimal128_backed_source_column() {
    assert_decimal_reader_predicate_matches_backed_source_column(TypeId::Decimal128);
}

fn assert_decimal_reader_predicate_matches_backed_source_column(storage_type: TypeId) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(format!("decimal_reader_predicate_{storage_type}.parquet"));
    let rt = sample_runtime();

    write_decimal_parquet_df(&path, storage_type);
    let source = decimal_source(&path);
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let chunk = &plan.chunks()[0];

    let eq_source = decimal_predicate_source(
        &rt,
        &source,
        chunk,
        SourceReaderPredicate::equal(
            SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
            SourceReaderPredicateLiteral::Decimal128 { raw: -500, scale: -2 },
        ),
    );
    let eq_frame = DataFrame::read_parquet_source_chunk(&rt, &eq_source, chunk, SourceReadMode::FinalOutput).unwrap();
    assert_decimal_amount_raws(&eq_frame, &[-500]);

    let ge_source = decimal_predicate_source(
        &rt,
        &source,
        chunk,
        SourceReaderPredicate::greater_equal(
            SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
            SourceReaderPredicateLiteral::Decimal128 { raw: 100, scale: -2 },
        ),
    );
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("decimal_source_read"))
        .unwrap();
    let ge_outcome =
        DataFrame::read_parquet_source_chunk_with_budget(&rt, &ge_source, chunk, SourceReadMode::FinalOutput, &token)
            .unwrap();
    assert_decimal_amount_raws(ge_outcome.value(), &[100, 200]);
}

#[test]
fn decimal_reader_batch_predicate_reads_decimal32_backed_chunks() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("decimal_reader_batch_predicate_first.parquet");
    let second = dir.path().join("decimal_reader_batch_predicate_second.parquet");
    let rt = sample_runtime();

    write_decimal_parquet_df(&first, TypeId::Decimal32);
    write_decimal_parquet_df(&second, TypeId::Decimal32);
    let source = decimal_source_from_paths([first.as_path(), second.as_path()]);
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    assert_eq!(plan.chunks().len(), 2);
    let predicate = SourceReaderPredicate::greater_equal(
        SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
        SourceReaderPredicateLiteral::Decimal128 { raw: 100, scale: -2 },
    );
    let predicate = build_local_parquet_reader_batch_predicate(&rt, &source, plan.chunks(), &predicate).unwrap();
    let source = source.with_reader_predicate(std::sync::Arc::new(predicate)).unwrap();
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("decimal_batch_read"))
        .unwrap();

    let outcome = DataFrame::read_parquet_source_chunks_with_budget(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    assert_decimal_amount_raws(outcome.value(), &[100, 200, 100, 200]);
}

#[test]
fn decimal_reader_batch_predicate_rejects_mixed_decimal_storage_types() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("decimal_reader_batch_predicate_decimal32.parquet");
    let second = dir.path().join("decimal_reader_batch_predicate_decimal64.parquet");
    let rt = sample_runtime();

    write_decimal_parquet_df(&first, TypeId::Decimal32);
    write_decimal_parquet_df(&second, TypeId::Decimal64);
    let source = decimal_source_from_paths([first.as_path(), second.as_path()]);
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let predicate = SourceReaderPredicate::greater_equal(
        SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
        SourceReaderPredicateLiteral::Decimal128 { raw: 100, scale: -2 },
    );

    let err = build_local_parquet_reader_batch_predicate(&rt, &source, plan.chunks(), &predicate)
        .expect_err("mixed decimal physical types must not share one batch reader predicate");

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("batch_reader_predicate_dtype_mismatch"));
}

#[test]
fn decimal_reader_predicate_supports_integer_and_in_list_literals() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("decimal_reader_predicate_in_list.parquet");
    let rt = sample_runtime();

    write_decimal_parquet_df(&path, TypeId::Decimal32);
    let source = decimal_source(&path);
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let chunk = &plan.chunks()[0];
    let predicate = SourceReaderPredicate::in_list(
        SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
        vec![SourceReaderPredicateLiteral::Int64(-5), SourceReaderPredicateLiteral::Decimal128 { raw: 200, scale: -2 }],
        false,
    )
    .unwrap();
    let source = decimal_predicate_source(&rt, &source, chunk, predicate);

    let frame = DataFrame::read_parquet_source_chunk(&rt, &source, chunk, SourceReadMode::FinalOutput).unwrap();
    assert_decimal_amount_raws(&frame, &[-500, 200]);
}

#[test]
fn decimal_reader_predicate_rejects_inexact_literal_rescale_before_execution() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("decimal_reader_predicate_inexact_rescale.parquet");
    let rt = sample_runtime();

    write_decimal_parquet_df(&path, TypeId::Decimal32);
    let source = decimal_source(&path);
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let err = build_local_parquet_reader_predicate(
        &rt,
        &source,
        &plan.chunks()[0],
        &SourceReaderPredicate::equal(
            SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
            SourceReaderPredicateLiteral::Decimal128 { raw: 1, scale: -3 },
        ),
    )
    .expect_err("0.001 cannot be represented exactly at source reader scale -2");

    assert_eq!(err.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("decimal_literal_rescale_not_exact"));
    assert_eq!(error_context_value(&err, "source_scale").as_deref(), Some("-3"));
    assert_eq!(error_context_value(&err, "target_scale").as_deref(), Some("-2"));
}

#[test]
fn decimal_reader_predicate_rejects_source_schema_scale_mismatch_before_execution() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("decimal_reader_predicate_scale_mismatch.parquet");
    let rt = sample_runtime();

    write_decimal_parquet_df(&path, TypeId::Decimal32);
    let decimal128_scale_three = DataType::fixed_point(TypeId::Decimal128, -3).unwrap();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "decimal", []).unwrap()],
        SourceSchema::new([SourceColumn::new("amount", decimal128_scale_three, false).unwrap()]).unwrap(),
        ["amount"],
        [SourceOutputColumn::new("amount", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();

    let err = build_local_parquet_reader_predicate(
        &rt,
        &source,
        &plan.chunks()[0],
        &SourceReaderPredicate::equal(
            SourceReaderPredicateColumn::read_column("amount", 0).unwrap(),
            SourceReaderPredicateLiteral::Decimal128 { raw: 1000, scale: -3 },
        ),
    )
    .expect_err("source schema scale must match the Parquet footer decimal scale");

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("decimal_scale_mismatch"));
    assert_eq!(error_context_value(&err, "logical_scale").as_deref(), Some("-3"));
    assert_eq!(error_context_value(&err, "reader_scale").as_deref(), Some("-2"));
}

#[test]
fn read_parquet_source_chunk_applies_reader_predicate() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("reader_predicate.parquet");
    let rt = sample_runtime();

    write_parquet_df(
        &path,
        &["id", "value"],
        vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_slice(&[10i64, 20, 30]).unwrap()],
        0,
    );

    let predicate = AstExpression::greater(
        AstExpression::left_column(0).unwrap(),
        AstExpression::literal(rt.scalar(1i32).unwrap()).unwrap(),
    )
    .unwrap();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "part-0", []).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("value", dtype(TypeId::Int64), false).unwrap(),
        ])
        .unwrap(),
        ["id", "value"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("value", SourceColumnRef::ReadColumn(1)).unwrap(),
        ],
        [],
        [],
    )
    .unwrap()
    .with_reader_predicate(std::sync::Arc::new(predicate))
    .unwrap();

    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let frame =
        DataFrame::read_parquet_source_chunk(&rt, &source, &plan.chunks()[0], SourceReadMode::FinalOutput).unwrap();

    assert_eq!(source.diagnostics().predicate_disposition(), PredicateDisposition::ReaderPushdown);
    assert_eq!(frame.column_names(), vec!["id", "value"]);
    assert_eq!(frame.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![2, 3]);
    assert_eq!(frame.series("value").unwrap().to_vec_strict::<i64>().unwrap(), vec![20, 30]);
}

#[test]
fn read_parquet_source_chunks_with_budget_reads_one_file_through_one_reader_input() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("batch-fan-in-first.parquet");
    let second = dir.path().join("batch-fan-in-second.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&first, &[1, 2, 3, 4, 5, 6], 2)
        || !write_i32_parquet_with_row_groups(&second, &[10, 20, 30, 40], 2)
    {
        return;
    }

    let source = LocalParquetSource::data(
        [
            LocalParquetFile::new(&first, "first", [])
                .unwrap()
                .with_row_group_metadata(vec![128, 128, 128], vec![2, 2, 2])
                .unwrap(),
            LocalParquetFile::new(&second, "second", [])
                .unwrap()
                .with_row_group_metadata(vec![128, 128], vec![2, 2])
                .unwrap(),
        ],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    // One row group per chunk is the shape that opened one file handle and
    // reparsed one footer per row group before same-file chunks were merged.
    let plan = plan_local_parquet_source_chunks(&source, SourceChunkPlanningPolicy::new(None, 1).unwrap()).unwrap();
    assert_eq!(plan.chunks().len(), 5);

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_batch_read"))
        .unwrap();
    let outcome = DataFrame::read_parquet_source_chunks_with_budget_with_evidence(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    assert_eq!(
        outcome.value().frame().series("id").unwrap().to_vec_strict::<i32>().unwrap(),
        vec![1, 2, 3, 4, 5, 6, 10, 20, 30, 40],
    );
    assert_eq!(outcome.report().context_facts().get("chunk_count").map(String::as_str), Some("5"));
    assert_eq!(outcome.report().context_facts().get("reader_input_count").map(String::as_str), Some("2"));
    assert_eq!(outcome.report().output_rows(), Some(10));
}

#[test]
fn read_parquet_source_chunks_with_budget_preserves_row_group_selection_and_file_order() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("batch-first.parquet");
    let second = dir.path().join("batch-second.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&first, &[1, 2, 3, 4], 2)
        || !write_i32_parquet_with_row_groups(&second, &[10, 20, 30, 40], 2)
    {
        return;
    }

    let source = LocalParquetSource::data(
        [
            LocalParquetFile::new(&first, "first", [])
                .unwrap()
                .with_row_groups(Some(vec![0]))
                .with_row_group_metadata(vec![128, 128], vec![2, 2])
                .unwrap(),
            LocalParquetFile::new(&second, "second", [])
                .unwrap()
                .with_row_groups(Some(vec![1]))
                .with_row_group_metadata(vec![128, 128], vec![2, 2])
                .unwrap(),
        ],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    assert_eq!(plan.chunks().len(), 2);

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_batch_read"))
        .unwrap();
    let outcome = DataFrame::read_parquet_source_chunks_with_budget_with_evidence(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    let evidence = outcome.value().read_evidence().expect("local batch reads should report Parquet read evidence");
    assert_eq!(evidence.input_kind, "local_path_multi");
    assert_eq!(evidence.requested_remote_read_mode, None);
    assert_eq!(evidence.backend, "libcudf_path_reader");
    assert_eq!(evidence.fallback_reason, "not_applicable");
    assert_eq!(outcome.value().frame().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 30, 40],);
    assert_eq!(outcome.report().request().boundary(), "source_batch_read");
    assert_eq!(outcome.report().output_rows(), Some(4));
    assert_eq!(outcome.report().context_facts().get("chunk_count").map(String::as_str), Some("2"));
    assert_eq!(outcome.report().context_facts().get("known_row_count").map(String::as_str), Some("4"));
}

#[test]
fn read_parquet_source_chunks_with_budget_applies_reader_predicate_to_batch() {
    let dir = TempDir::new().unwrap();
    let first = dir.path().join("batch-predicate-first.parquet");
    let second = dir.path().join("batch-predicate-second.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&first, &[1, 2, 3, 4], 2)
        || !write_i32_parquet_with_row_groups(&second, &[10, 20, 30, 40], 2)
    {
        return;
    }

    let predicate = AstExpression::greater(
        AstExpression::left_column(0).unwrap(),
        AstExpression::literal(rt.scalar(12i32).unwrap()).unwrap(),
    )
    .unwrap();
    let source = LocalParquetSource::data(
        [
            LocalParquetFile::new(&first, "first", [])
                .unwrap()
                .with_row_groups(Some(vec![0, 1]))
                .with_row_group_metadata(vec![128, 128], vec![2, 2])
                .unwrap(),
            LocalParquetFile::new(&second, "second", [])
                .unwrap()
                .with_row_groups(Some(vec![0, 1]))
                .with_row_group_metadata(vec![128, 128], vec![2, 2])
                .unwrap(),
        ],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap()
    .with_reader_predicate(std::sync::Arc::new(predicate))
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    assert_eq!(plan.chunks().len(), 2);
    assert!(plan.chunks().iter().all(SourceChunk::selects_all_known_file_row_groups));

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_batch_read"))
        .unwrap();
    let outcome = DataFrame::read_parquet_source_chunks_with_budget(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    assert_eq!(source.diagnostics().predicate_disposition(), PredicateDisposition::ReaderPushdown);
    assert_eq!(outcome.value().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30, 40]);
    assert_eq!(outcome.report().context_facts().get("chunk_count").map(String::as_str), Some("2"));
}

#[test]
fn read_parquet_source_chunks_with_budget_validates_remote_sources_at_multi_reader_boundary() {
    let rt = sample_runtime();
    let remote_profile = RemoteParquetProfile::shared_process_env().with_endpoint("http://local\0host:9000");
    let source = LocalParquetSource::data(
        [
            LocalParquetFile::object_store_uri("s3://bucket/table/first.parquet", "first", [])
                .unwrap()
                .with_remote_parquet_contract(remote_profile.clone(), RemoteReadMode::RequireDirect)
                .unwrap()
                .with_row_group_metadata(vec![128], vec![2])
                .unwrap(),
            LocalParquetFile::object_store_uri("s3://bucket/table/second.parquet", "second", [])
                .unwrap()
                .with_remote_parquet_contract(remote_profile, RemoteReadMode::RequireDirect)
                .unwrap()
                .with_row_group_metadata(vec![128], vec![2])
                .unwrap(),
        ],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    assert_eq!(plan.chunks().len(), 2);

    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("remote_batch_read"))
        .unwrap();
    let error = DataFrame::read_parquet_source_chunks_with_budget(
        &rt,
        &source,
        plan.chunks(),
        SourceReadMode::FinalOutput,
        &token,
    )
    .expect_err("remote multi-source profile with an interior null should be rejected before libcudf opens S3");

    assert_eq!(error.code(), ErrorCode::NullUnexpected);
    assert_eq!(error_context_value(&error, "resource").as_deref(), Some("aws_endpoint_url"));
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

    if !write_i32_parquet_with_row_groups(&path, &[1, 2, 3, 4], 2) {
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
fn read_parquet_source_chunk_from_host_buffer_honors_row_groups_and_reader_predicate() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("host_buffer_row_groups_predicate.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&path, &[1, 2, 3, 4], 2) {
        return;
    }
    let filebytes = fs::read(&path).unwrap();

    let predicate = AstExpression::greater(
        AstExpression::left_column(0).unwrap(),
        AstExpression::literal(rt.scalar(3i32).unwrap()).unwrap(),
    )
    .unwrap();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "host-buffer-rg", [])
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
    .unwrap()
    .with_reader_predicate(std::sync::Arc::new(predicate))
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();

    let df = DataFrame::read_parquet_source_chunk_from_host_buffer(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &filebytes,
    )
    .unwrap();

    assert_eq!(df.column_names(), vec!["id"]);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![4]);
}

#[test]
fn read_parquet_source_chunk_with_budget_uses_token_allocator_and_reports_observed_output() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("budgeted_source.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2, 3, 4]).unwrap()], 0);

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "budgeted", []).unwrap().with_row_group_metadata(vec![128], vec![4]).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read")).unwrap();

    reset_exec_observation();
    let outcome = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();
    let df = outcome.value();
    let report = outcome.report();
    let expected_output_bytes = dataframe_device_allocated_size_bytes(df);
    let manager_mr = token.memory_resource().testing_raw_handle_value();

    assert_eq!(df.column_names(), vec!["id"]);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(df.testing_execution_domain_mr_value(), manager_mr);
    assert_eq!(last_exec_mr_value(), manager_mr);
    assert_eq!(report.request().boundary(), "source_read");
    assert_eq!(report.request().budget_bytes(), 64 * 1024 * 1024);
    assert_eq!(report.enforcement(), ExecutionBudgetEnforcement::StreamBoundReservation);
    assert_eq!(report.output_rows(), Some(4));
    assert_eq!(report.retained_bytes(), Some(expected_output_bytes));
    assert_eq!(report.output_bytes(), Some(expected_output_bytes));
    assert_budget_report_high_water(report, expected_output_bytes);
    assert_eq!(
        report.context_facts().get("source_read_preflight_contract").map(String::as_str),
        Some("exact_retained_output_upper_bound")
    );
    assert_eq!(
        report.context_facts().get("source_read_preflight_retained_output_bytes").map(String::as_str),
        Some("16")
    );
    let expected_required_budget = DataFrame::minimum_parquet_source_read_budget_bytes().to_string();
    assert_eq!(
        report.context_facts().get("source_read_preflight_required_budget_bytes").map(String::as_str),
        Some(expected_required_budget.as_str())
    );
    assert_eq!(report.context_facts().get("source_kind").map(String::as_str), Some("data"));
    assert_eq!(report.context_facts().get("source_read_mode").map(String::as_str), Some("final_output"));
    assert_eq!(report.context_facts().get("stable_id").map(String::as_str), Some("budgeted"));
    assert_eq!(report.context_facts().get("chunk_index").map(String::as_str), Some("0"));
    assert_eq!(report.context_facts().get("file_index").map(String::as_str), Some("0"));
    assert_eq!(report.context_facts().get("reads_all_row_groups").map(String::as_str), Some("true"));
    assert_eq!(report.context_facts().get("selected_row_groups").map(String::as_str), Some("all"));
    assert_eq!(report.context_facts().get("read_columns").map(String::as_str), Some("id"));
    assert_eq!(report.context_facts().get("read_column_count").map(String::as_str), Some("1"));
    assert_eq!(report.context_facts().get("projected_columns").map(String::as_str), Some("id"));
    assert_eq!(report.context_facts().get("projected_column_count").map(String::as_str), Some("1"));
    assert!(report.context_facts().get("path").is_some_and(|path| path.ends_with("budgeted_source.parquet")));
    assert!(expected_output_bytes > 0);
}

#[test]
fn read_parquet_source_chunk_with_budget_from_host_buffer_uses_token_allocator() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("budgeted_host_buffer_source.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2, 3, 4]).unwrap()], 0);
    let filebytes = fs::read(&path).unwrap();

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "budgeted-host-buffer", [])
            .unwrap()
            .with_row_group_metadata(vec![128], vec![4])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("host_buffer_source_read"))
        .unwrap();

    reset_exec_observation();
    let outcome = DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer_with_evidence(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &filebytes,
        &token,
    )
    .unwrap();
    let evidence = outcome.value().read_evidence().expect("host-buffer reads should report Parquet read evidence");
    assert_eq!(evidence.input_kind, "host_buffer");
    assert_eq!(evidence.requested_remote_read_mode, None);
    assert_eq!(evidence.backend, "host_buffer");
    assert_eq!(evidence.fallback_reason, "not_applicable");
    let df = outcome.value().frame();
    let report = outcome.report();
    let expected_output_bytes = dataframe_device_allocated_size_bytes(df);
    let manager_mr = token.memory_resource().testing_raw_handle_value();

    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(df.testing_execution_domain_mr_value(), manager_mr);
    assert_eq!(last_exec_mr_value(), manager_mr);
    assert_eq!(report.request().boundary(), "host_buffer_source_read");
    assert_eq!(report.output_rows(), Some(4));
    assert_eq!(report.retained_bytes(), Some(expected_output_bytes));
    assert_eq!(report.output_bytes(), Some(expected_output_bytes));
    assert_budget_report_high_water(report, expected_output_bytes);
    assert_eq!(
        report.context_facts().get("source_read_preflight_contract").map(String::as_str),
        Some("exact_retained_output_upper_bound")
    );
}

#[test]
fn read_parquet_source_chunk_rejects_cross_source_chunk_with_matching_partition_shape() {
    let dir = TempDir::new().unwrap();
    let source_a_path = dir.path().join("source-a.parquet");
    let source_b_path = dir.path().join("source-b.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&source_a_path, &[1, 2, 3, 4], 2)
        || !write_i32_parquet_with_row_groups(&source_b_path, &[10, 20, 30, 40], 2)
    {
        return;
    }

    let source_a = LocalParquetSource::data(
        [LocalParquetFile::new(&source_a_path, "source-a", [PartitionValue::Int32(9)])
            .unwrap()
            .with_row_groups(Some(vec![0]))
            .with_row_group_metadata(vec![128, 128], vec![2, 2])
            .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("part", dtype(TypeId::Int32), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("part", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("part", dtype(TypeId::Int32), false).unwrap()],
    )
    .unwrap();
    let source_b = LocalParquetSource::data(
        [LocalParquetFile::new(&source_b_path, "source-b", [PartitionValue::Int32(9)])
            .unwrap()
            .with_row_groups(Some(vec![1]))
            .with_row_group_metadata(vec![128, 128], vec![2, 2])
            .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("part", dtype(TypeId::Int32), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("part", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("part", dtype(TypeId::Int32), false).unwrap()],
    )
    .unwrap();
    let source_a_plan = plan_local_parquet_source_chunks(&source_a, Default::default()).unwrap();
    let source_a_chunk = &source_a_plan.chunks()[0];

    let err = DataFrame::read_parquet_source_chunk(&rt, &source_b, source_a_chunk, SourceReadMode::FinalOutput)
        .expect_err("chunk planned from source A must not be accepted for source B");
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("source_chunk_path_mismatch"));
    assert_eq!(error_context_value(&err, "expected_path").as_deref(), Some(source_b_path.to_str().unwrap()));
    assert_eq!(error_context_value(&err, "actual_path").as_deref(), Some(source_a_path.to_str().unwrap()));

    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read")).unwrap();
    let budget_err = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source_b,
        source_a_chunk,
        SourceReadMode::FinalOutput,
        &token,
    )
    .expect_err("budgeted chunk read must reject a chunk planned from another source");
    assert_eq!(budget_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&budget_err, "reason").as_deref(), Some("source_chunk_path_mismatch"));
}

#[test]
fn read_parquet_source_chunk_rejects_cross_source_subset_row_groups_with_same_file_identity() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("same-file-row-groups.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&path, &[1, 2, 3, 4], 2) {
        return;
    }

    let schema = || {
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("part", dtype(TypeId::Int32), false).unwrap(),
        ])
        .unwrap()
    };
    let outputs = || {
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("part", SourceColumnRef::Partition(0)).unwrap(),
        ]
    };
    let partitions = || [PartitionColumn::new("part", dtype(TypeId::Int32), false).unwrap()];

    let source_a = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "same-file", [PartitionValue::Int32(9)])
            .unwrap()
            .with_row_groups(Some(vec![0]))
            .with_row_group_metadata(vec![128, 128], vec![2, 2])
            .unwrap()],
        schema(),
        ["id"],
        outputs(),
        [],
        partitions(),
    )
    .unwrap();
    let source_b = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "same-file", [PartitionValue::Int32(9)])
            .unwrap()
            .with_row_groups(Some(vec![0, 1]))
            .with_row_group_metadata(vec![128, 128], vec![2, 2])
            .unwrap()],
        schema(),
        ["id"],
        outputs(),
        [],
        partitions(),
    )
    .unwrap();
    let source_a_plan = plan_local_parquet_source_chunks(&source_a, Default::default()).unwrap();
    let source_a_chunk = &source_a_plan.chunks()[0];

    let err = DataFrame::read_parquet_source_chunk(&rt, &source_b, source_a_chunk, SourceReadMode::FinalOutput)
        .expect_err("chunk planned from source A must not be accepted for a superset source B");
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("source_chunk_source_row_groups_mismatch"));

    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read")).unwrap();
    let budget_err = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source_b,
        source_a_chunk,
        SourceReadMode::FinalOutput,
        &token,
    )
    .expect_err("budgeted chunk read must reject a chunk planned from a narrower row-group source");
    assert_eq!(budget_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&budget_err, "reason").as_deref(), Some("source_chunk_source_row_groups_mismatch"));
}

#[test]
fn parquet_source_chunk_preflight_rejects_stale_remote_contract_with_same_file_identity() {
    let schema = || SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap();
    let outputs = || [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()];
    let uri = "s3://bucket/table/part-0.parquet";
    let source_a = LocalParquetSource::data(
        [LocalParquetFile::object_store_uri(uri, "same-file", [])
            .unwrap()
            .with_remote_parquet_contract(
                RemoteParquetProfile::shared_process_env().with_region("us-west-2"),
                RemoteReadMode::AllowHostFallback,
            )
            .unwrap()
            .with_row_group_metadata(vec![128], vec![2])
            .unwrap()],
        schema(),
        ["id"],
        outputs(),
        [],
        [],
    )
    .unwrap();
    let source_b = LocalParquetSource::data(
        [LocalParquetFile::object_store_uri(uri, "same-file", [])
            .unwrap()
            .with_row_group_metadata(vec![128], vec![2])
            .unwrap()],
        schema(),
        ["id"],
        outputs(),
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source_a, Default::default()).unwrap();

    let err =
        DataFrame::parquet_source_chunk_exact_preflight(&source_b, &plan.chunks()[0], SourceReadMode::FinalOutput)
            .expect_err("chunk planned with a different remote contract must not be accepted");

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("source_chunk_location_mismatch"));
}

#[test]
fn read_parquet_source_chunk_rejects_null_partition_chunk_for_non_nullable_source() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("partition_identity.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()], 0);

    let nullable_source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "nullable", [PartitionValue::Null(dtype(TypeId::Int32))]).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("part", dtype(TypeId::Int32), true).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("part", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("part", dtype(TypeId::Int32), true).unwrap()],
    )
    .unwrap();
    let non_nullable_source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "non-nullable", [PartitionValue::Int32(7)]).unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("part", dtype(TypeId::Int32), false).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("part", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [],
        [PartitionColumn::new("part", dtype(TypeId::Int32), false).unwrap()],
    )
    .unwrap();
    let nullable_plan = plan_local_parquet_source_chunks(&nullable_source, Default::default()).unwrap();
    let nullable_chunk = &nullable_plan.chunks()[0];

    let err =
        DataFrame::read_parquet_source_chunk(&rt, &non_nullable_source, nullable_chunk, SourceReadMode::FinalOutput)
            .unwrap_err();
    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("partition_null_for_non_nullable_column"));
    assert_eq!(error_context_value(&err, "partition_column").as_deref(), Some("part"));

    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read")).unwrap();
    let budget_err = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &non_nullable_source,
        nullable_chunk,
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap_err();
    assert_eq!(budget_err.code(), ErrorCode::NullUnexpected);
    assert_eq!(error_context_value(&budget_err, "reason").as_deref(), Some("partition_null_for_non_nullable_column"));
    assert_eq!(error_context_value(&budget_err, "partition_column").as_deref(), Some("part"));
}

#[test]
fn read_parquet_source_chunk_with_budget_materializes_extended_partition_values() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("extended_partitions.parquet");
    let rt = sample_runtime();
    let decimal_dtype = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[1i32, 2]).unwrap()], 0);

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(
            &path,
            "extended",
            [
                PartitionValue::Decimal128 { raw: 12345, scale: -2 },
                PartitionValue::Date32(19_481),
                PartitionValue::Timestamp { raw: 1_700_000_123_456, resolution: TimestampResolution::Microseconds },
                PartitionValue::Duration { raw: -42_000_000, resolution: TimestampResolution::Nanoseconds },
                PartitionValue::Null(dtype(TypeId::Int32)),
                PartitionValue::Null(decimal_dtype),
                PartitionValue::Null(dtype(TypeId::TimestampDays)),
                PartitionValue::Null(dtype(TypeId::TimestampMicroseconds)),
                PartitionValue::Null(dtype(TypeId::DurationNanoseconds)),
            ],
        )
        .unwrap()],
        SourceSchema::new([
            SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap(),
            SourceColumn::new("amount", decimal_dtype, false).unwrap(),
            SourceColumn::new("date", dtype(TypeId::TimestampDays), false).unwrap(),
            SourceColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap(),
            SourceColumn::new("duration", dtype(TypeId::DurationNanoseconds), false).unwrap(),
            SourceColumn::new("maybe_id", dtype(TypeId::Int32), true).unwrap(),
            SourceColumn::new("maybe_amount", decimal_dtype, true).unwrap(),
            SourceColumn::new("maybe_date", dtype(TypeId::TimestampDays), true).unwrap(),
            SourceColumn::new("maybe_ts", dtype(TypeId::TimestampMicroseconds), true).unwrap(),
            SourceColumn::new("maybe_duration", dtype(TypeId::DurationNanoseconds), true).unwrap(),
        ])
        .unwrap(),
        ["id"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("amount", SourceColumnRef::Partition(0)).unwrap(),
            SourceOutputColumn::new("date", SourceColumnRef::Partition(1)).unwrap(),
            SourceOutputColumn::new("ts", SourceColumnRef::Partition(2)).unwrap(),
            SourceOutputColumn::new("duration", SourceColumnRef::Partition(3)).unwrap(),
            SourceOutputColumn::new("maybe_id", SourceColumnRef::Partition(4)).unwrap(),
            SourceOutputColumn::new("maybe_amount", SourceColumnRef::Partition(5)).unwrap(),
            SourceOutputColumn::new("maybe_date", SourceColumnRef::Partition(6)).unwrap(),
            SourceOutputColumn::new("maybe_ts", SourceColumnRef::Partition(7)).unwrap(),
            SourceOutputColumn::new("maybe_duration", SourceColumnRef::Partition(8)).unwrap(),
        ],
        [],
        [
            PartitionColumn::new("amount", decimal_dtype, false).unwrap(),
            PartitionColumn::new("date", dtype(TypeId::TimestampDays), false).unwrap(),
            PartitionColumn::new("ts", dtype(TypeId::TimestampMicroseconds), false).unwrap(),
            PartitionColumn::new("duration", dtype(TypeId::DurationNanoseconds), false).unwrap(),
            PartitionColumn::new("maybe_id", dtype(TypeId::Int32), true).unwrap(),
            PartitionColumn::new("maybe_amount", decimal_dtype, true).unwrap(),
            PartitionColumn::new("maybe_date", dtype(TypeId::TimestampDays), true).unwrap(),
            PartitionColumn::new("maybe_ts", dtype(TypeId::TimestampMicroseconds), true).unwrap(),
            PartitionColumn::new("maybe_duration", dtype(TypeId::DurationNanoseconds), true).unwrap(),
        ],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read")).unwrap();

    let outcome = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();
    let df = outcome.value();

    assert_eq!(
        df.column_names(),
        vec![
            "id",
            "amount",
            "date",
            "ts",
            "duration",
            "maybe_id",
            "maybe_amount",
            "maybe_date",
            "maybe_ts",
            "maybe_duration"
        ]
    );
    assert_eq!(df.series("amount").unwrap().dtype().unwrap(), decimal_dtype);
    assert_eq!(df.series("date").unwrap().dtype().unwrap(), dtype(TypeId::TimestampDays));
    assert_eq!(df.series("ts").unwrap().dtype().unwrap(), dtype(TypeId::TimestampMicroseconds));
    assert_eq!(df.series("duration").unwrap().dtype().unwrap(), dtype(TypeId::DurationNanoseconds));
    assert_eq!(df.series("maybe_id").unwrap().null_count(), 2);
    assert_eq!(df.series("maybe_amount").unwrap().dtype().unwrap(), decimal_dtype);
    assert_eq!(df.series("maybe_date").unwrap().dtype().unwrap(), dtype(TypeId::TimestampDays));
    assert_eq!(df.series("maybe_ts").unwrap().dtype().unwrap(), dtype(TypeId::TimestampMicroseconds));
    assert_eq!(df.series("maybe_duration").unwrap().dtype().unwrap(), dtype(TypeId::DurationNanoseconds));
    assert_eq!(df.series("maybe_amount").unwrap().null_count(), 2);
    assert_eq!(df.series("maybe_date").unwrap().null_count(), 2);
    assert_eq!(df.series("maybe_ts").unwrap().null_count(), 2);
    assert_eq!(df.series("maybe_duration").unwrap().null_count(), 2);

    let batch = df.to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(1).data_type(), ArrowDataType::Decimal128(38, 2));
    assert_eq!(*batch.schema().field(2).data_type(), ArrowDataType::Date32);
    assert_eq!(*batch.schema().field(3).data_type(), ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(*batch.schema().field(4).data_type(), ArrowDataType::Duration(TimeUnit::Nanosecond));
    assert_eq!(*batch.schema().field(5).data_type(), ArrowDataType::Int32);
    assert!(batch.schema().field(5).is_nullable());
    assert_eq!(*batch.schema().field(6).data_type(), ArrowDataType::Decimal128(38, 2));
    assert_eq!(*batch.schema().field(7).data_type(), ArrowDataType::Date32);
    assert_eq!(*batch.schema().field(8).data_type(), ArrowDataType::Timestamp(TimeUnit::Microsecond, None));
    assert_eq!(*batch.schema().field(9).data_type(), ArrowDataType::Duration(TimeUnit::Nanosecond));
    assert!(batch.schema().field(6).is_nullable());
    assert!(batch.schema().field(7).is_nullable());
    assert!(batch.schema().field(8).is_nullable());
    assert!(batch.schema().field(9).is_nullable());

    assert_eq!(batch.column(0).as_primitive::<Int32Type>().values().as_ref(), &[1, 2]);
    assert_eq!(batch.column(1).as_primitive::<Decimal128Type>().values().as_ref(), &[12345, 12345]);
    assert_eq!(batch.column(2).as_primitive::<Date32Type>().values().as_ref(), &[19_481, 19_481]);
    assert_eq!(
        batch.column(3).as_primitive::<TimestampMicrosecondType>().values().as_ref(),
        &[1_700_000_123_456, 1_700_000_123_456]
    );
    assert_eq!(batch.column(4).as_primitive::<DurationNanosecondType>().values().as_ref(), &[-42_000_000, -42_000_000]);
    assert!(batch.column(5).is_null(0));
    assert!(batch.column(5).is_null(1));
    for column_index in 6..=9 {
        assert!(batch.column(column_index).is_null(0));
        assert!(batch.column(column_index).is_null(1));
    }
}

#[test]
fn read_parquet_source_chunk_with_budget_reports_exceeded_budget_facts() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("budget_too_small.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[10i32, 20, 30, 40]).unwrap()], 0);

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "too-small", []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token = rt.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("source_read")).unwrap();

    let err = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &token,
    )
    .expect_err("one-byte budget should not cover parquet decoded output allocations");

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("source_read"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "budget_enforcement").as_deref(), Some("stream_bound_reservation"));
    assert_eq!(error_context_value(&err, "budget_status").as_deref(), Some("exceeded_or_lower_layer_rejected"));
    assert_eq!(error_context_value(&err, "budget_peak_bytes").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "budget_current_bytes").as_deref(), Some("0"));
    assert_eq!(error_context_value(&err, "budget_failed_allocation_count").as_deref(), Some("1"));
    assert!(error_context_value(&err, "budget_largest_failed_allocation_bytes").is_some());
    let budget = err.budget_exhaustion().expect("budgeted source read should expose typed budget detail");
    assert_eq!(budget.hard_budget_bytes(), 1);
    assert_eq!(budget.peak_bytes(), 0);
    assert_eq!(budget.peak_aligned_bytes(), 0);
    assert_eq!(budget.current_bytes(), 0);
    assert_eq!(budget.failed_allocation_count(), 1);
    assert!(budget.largest_failed_allocation_bytes() > 0);
    assert_eq!(budget.boundary(), "source_read");
    assert_eq!(budget.operation(), "dataframe::read_parquet_source_chunk_with_budget");
}

#[test]
fn read_parquet_source_chunk_with_budget_from_host_buffer_reports_exceeded_budget_facts() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("host_buffer_budget_too_small.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[10i32, 20, 30, 40]).unwrap()], 0);
    let filebytes = fs::read(&path).unwrap();

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "host-buffer-too-small", []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("host_buffer_source_read")).unwrap();

    let err = DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &filebytes,
        &token,
    )
    .expect_err("one-byte budget should not cover host-buffer parquet decoded output allocations");

    assert_eq!(err.code(), ErrorCode::OutOfMemory);
    assert_eq!(error_context_value(&err, "budget_boundary").as_deref(), Some("host_buffer_source_read"));
    assert_eq!(error_context_value(&err, "budget_bytes").as_deref(), Some("1"));
    assert_eq!(
        error_context_value(&err, "budget_operation").as_deref(),
        Some("dataframe::read_parquet_source_chunk_with_budget_from_host_buffer")
    );
}

#[test]
fn read_parquet_source_chunk_minimum_hard_budget_covers_tiny_data_read() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("minimum_budget.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["id"], vec![column_from_slice(&[10i32, 20]).unwrap()], 0);

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "minimum-budget", []).unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token = rt
        .execution_budget_token(
            ExecutionBudgetRequest::new(DataFrame::minimum_parquet_source_read_budget_bytes())
                .with_boundary("source_read_minimum"),
        )
        .unwrap();

    let outcome = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();

    assert_eq!(outcome.value().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20]);
    assert_eq!(outcome.report().request().boundary(), "source_read_minimum");
    assert_eq!(outcome.report().request().budget_bytes(), DataFrame::minimum_parquet_source_read_budget_bytes());
}

#[test]
fn read_parquet_source_chunk_with_budget_string_output_keeps_preflight_unresolved() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("string_budgeted_source.parquet");
    let rt = sample_runtime();

    write_parquet_df(&path, &["name"], vec![column_from_strs(&["ALPHA", "BETA", "GAMMA"]).unwrap()], 0);

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "strings", []).unwrap()],
        SourceSchema::new([SourceColumn::new("name", dtype(TypeId::String), false).unwrap()]).unwrap(),
        ["name"],
        [SourceOutputColumn::new("name", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read_strings"))
        .unwrap();

    let outcome = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source,
        &plan.chunks()[0],
        SourceReadMode::FinalOutput,
        &token,
    )
    .unwrap();
    let df = outcome.value();
    let report = outcome.report();

    assert_eq!(
        df.series("name").unwrap().to_strings_strict().unwrap(),
        vec!["ALPHA".to_string(), "BETA".to_string(), "GAMMA".to_string()]
    );
    assert_eq!(report.output_rows(), Some(3));
    assert!(report.output_bytes().unwrap() > 0);
}

#[test]
fn source_contract_reports_exact_preflight_with_known_fixed_width_source_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("source_contract.parquet");
    let rt = sample_runtime();

    if !write_i32_parquet_with_row_groups(&path, &[10, 20, 30, 40], 2) {
        return;
    }

    let source = LocalParquetSource::data(
        [LocalParquetFile::new(&path, "source-contract", [])
            .unwrap()
            .with_file_size_bytes(4_096)
            .with_row_groups(Some(vec![1]))
            .with_row_group_metadata(vec![1_024, 2_048], vec![2, 2])
            .unwrap()],
        SourceSchema::new([SourceColumn::new("id", dtype(TypeId::Int32), false).unwrap()]).unwrap(),
        ["id"],
        [SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap()],
        [],
        [],
    )
    .unwrap();
    let plan = plan_local_parquet_source_chunks(&source, Default::default()).unwrap();
    let chunk = &plan.chunks()[0];
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("estimated_source_read"))
        .unwrap();

    assert!(plan.row_group_metadata_available());
    assert!(!plan.size_estimate_based());
    assert_eq!(chunk.row_group_indexes(), &[1]);
    assert_eq!(chunk.known_file_bytes(), Some(4_096));
    assert_eq!(chunk.known_row_group_bytes(), Some(2_048));
    assert_eq!(chunk.estimated_chunk_bytes(), Some(2_048));
    assert_eq!(chunk.known_row_count(), Some(2));

    let outcome =
        DataFrame::read_parquet_source_chunk_with_budget(&rt, &source, chunk, SourceReadMode::FinalOutput, &token)
            .unwrap();

    assert_eq!(outcome.value().series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![30, 40]);
    assert_eq!(outcome.report().request().boundary(), "estimated_source_read");
    assert_eq!(
        outcome.report().context_facts().get("source_read_preflight_contract").map(String::as_str),
        Some("exact_retained_output_upper_bound")
    );
    assert_eq!(
        outcome.report().context_facts().get("source_read_preflight_retained_output_bytes").map(String::as_str),
        Some("8")
    );
    assert_eq!(outcome.report().context_facts().get("reads_all_row_groups").map(String::as_str), Some("false"));
    assert_eq!(outcome.report().context_facts().get("selected_row_groups").map(String::as_str), Some("1"));
    assert_eq!(outcome.report().context_facts().get("selected_row_group_count").map(String::as_str), Some("1"));
    assert_eq!(outcome.report().context_facts().get("known_file_bytes").map(String::as_str), Some("4096"));
    assert_eq!(outcome.report().context_facts().get("known_row_group_bytes").map(String::as_str), Some("2048"));
    assert_eq!(outcome.report().context_facts().get("estimated_chunk_bytes").map(String::as_str), Some("2048"));
    assert_eq!(outcome.report().context_facts().get("known_row_count").map(String::as_str), Some("2"));
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

#[test]
fn read_parquet_source_chunk_rejects_stale_zero_column_count_metadata() {
    let rt = sample_runtime();
    let source_a =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("same-count.parquet", "same-count", [])
            .unwrap()
            .with_row_groups(Some(vec![0]))
            .with_row_group_metadata(vec![128, 256], vec![2, 3])
            .unwrap()])
        .unwrap();
    let source_b =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("same-count.parquet", "same-count", [])
            .unwrap()
            .with_row_groups(Some(vec![0]))
            .with_row_group_metadata(vec![128, 256], vec![9, 3])
            .unwrap()])
        .unwrap();
    let source_a_plan = plan_local_parquet_source_chunks(&source_a, Default::default()).unwrap();
    let source_a_chunk = &source_a_plan.chunks()[0];

    let err = DataFrame::read_parquet_source_chunk(&rt, &source_b, source_a_chunk, SourceReadMode::FinalOutput)
        .expect_err("zero-column count chunk planned from stale row-count metadata must be rejected");
    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("source_chunk_row_group_rows_metadata_mismatch"));

    let token =
        rt.execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("source_read")).unwrap();
    let budget_err = DataFrame::read_parquet_source_chunk_with_budget(
        &rt,
        &source_b,
        source_a_chunk,
        SourceReadMode::FinalOutput,
        &token,
    )
    .expect_err("budgeted zero-column count chunk read must reject stale row-count metadata");
    assert_eq!(budget_err.code(), ErrorCode::TypeMismatch);
    assert_eq!(
        error_context_value(&budget_err, "reason").as_deref(),
        Some("source_chunk_row_group_rows_metadata_mismatch")
    );
}
