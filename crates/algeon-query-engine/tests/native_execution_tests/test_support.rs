use std::{path::Path, sync::Arc};

use query_engine::{error::Result as QueryEngineResult, plan::QueryPlan};

use super::*;

pub(super) fn projection_outputs(expressions: Vec<Expr>, schema: Arc<Schema>) -> Vec<ProjectionOutput> {
    assert_eq!(expressions.len(), schema.fields().len(), "projection outputs must match schema");
    expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| ProjectionOutput::new(expr, field.as_ref().clone()))
        .collect()
}

pub(super) fn partition_column_query_plan(
    first_path: &Path,
    second_path: &Path,
    file_schema: Arc<Schema>,
) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(
        file_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new("part", DataType::Int32, false)])
            .collect::<Vec<_>>(),
    ));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("part", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("part", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(first_path)
                .with_file_size_bytes(fs::metadata(first_path).unwrap().len())
                .with_partition_values(vec![LiteralValue::Int32(1)]),
            LocalParquetFile::new(second_path)
                .with_file_size_bytes(fs::metadata(second_path).unwrap().len())
                .with_partition_values(vec![LiteralValue::Int32(2)]),
        ],
        Arc::clone(&source_schema),
        vec!["m00".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)])
    .with_column_roles(vec!["part".to_string(), "m00".to_string()], vec!["part".to_string()]);
    let filter = Expr::binary(Expr::column("part"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2)));
    let projection = vec![Expr::column("part"), Expr::column("m00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("part")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("part"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, source_schema);
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    builder.finish_with_host_sink(sort)
}

pub(super) fn partition_passthrough_query_plan(
    paths: &[&Path],
    file_schema: Arc<Schema>,
    partition_columns: Vec<PartitionColumn>,
    partition_values: Vec<Vec<LiteralValue>>,
) -> QueryPlan {
    assert_eq!(paths.len(), partition_values.len());
    let partition_names = partition_columns.iter().map(|column| column.name.clone()).collect::<Vec<_>>();
    let source_schema = Arc::new(Schema::new(
        file_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain(
                partition_columns
                    .iter()
                    .map(|column| Field::new(column.name.as_str(), column.data_type.clone(), column.nullable)),
            )
            .collect::<Vec<_>>(),
    ));
    let aggregated_schema = Arc::new(Schema::new(
        source_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new("sum_id", DataType::Int64, true)])
            .collect::<Vec<_>>(),
    ));
    let files = paths
        .iter()
        .zip(partition_values)
        .map(|(path, values)| {
            LocalParquetFile::new(*path)
                .with_file_size_bytes(fs::metadata(path).unwrap().len())
                .with_partition_values(values)
        })
        .collect::<Vec<_>>();
    let mut projected_columns = vec!["id".to_string()];
    projected_columns.extend(partition_names.iter().cloned());
    let source = LocalParquetSource::from_files(files, Arc::clone(&source_schema), vec!["id".to_string()])
        .with_partition_columns(partition_columns)
        .with_column_roles(projected_columns.clone(), Vec::new());
    let mut group_by = vec![Expr::column("id")];
    group_by.extend(partition_names.iter().map(Expr::column));
    let aggregate =
        AggregateSpec::new(group_by, vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("id"), "sum_id")]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("id"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    builder.finish_with_host_sink(sort)
}

pub(super) fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array")
}

pub(super) fn int64_array(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch.column(index).as_any().downcast_ref::<Int64Array>().expect("expected Int64Array")
}

pub(super) fn uint64_array(batch: &RecordBatch, index: usize) -> &UInt64Array {
    batch.column(index).as_any().downcast_ref::<UInt64Array>().expect("expected UInt64Array")
}

pub(super) fn boolean_array(batch: &RecordBatch, index: usize) -> &BooleanArray {
    batch.column(index).as_any().downcast_ref::<BooleanArray>().expect("expected BooleanArray")
}

pub(super) fn float64_array(batch: &RecordBatch, index: usize) -> &Float64Array {
    batch.column(index).as_any().downcast_ref::<Float64Array>().expect("expected Float64Array")
}

pub(super) fn float32_array(batch: &RecordBatch, index: usize) -> &Float32Array {
    batch.column(index).as_any().downcast_ref::<Float32Array>().expect("expected Float32Array")
}

pub(super) fn decimal128_array(batch: &RecordBatch, index: usize) -> &Decimal128Array {
    batch.column(index).as_any().downcast_ref::<Decimal128Array>().expect("expected Decimal128Array")
}

pub(super) fn date32_array(batch: &RecordBatch, index: usize) -> &Date32Array {
    batch.column(index).as_any().downcast_ref::<Date32Array>().expect("expected Date32Array")
}

pub(super) fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

pub(super) fn timestamp_ns_array(batch: &RecordBatch, index: usize) -> &TimestampNanosecondArray {
    batch.column(index).as_any().downcast_ref::<TimestampNanosecondArray>().expect("expected TimestampNanosecondArray")
}

pub(super) fn duration_ns_array(batch: &RecordBatch, index: usize) -> &DurationNanosecondArray {
    batch.column(index).as_any().downcast_ref::<DurationNanosecondArray>().expect("expected DurationNanosecondArray")
}

pub(super) fn string_array(batch: &RecordBatch, index: usize) -> &StringArray {
    batch.column(index).as_any().downcast_ref::<StringArray>().expect("expected StringArray")
}

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

// Runs the plan on an admitted context so the query domain's real peak is
// observable after the attempt.
pub(super) fn execute_admitted_with_peak(
    context: &BackendQueryContext,
    plan: &QueryPlan,
) -> (QueryEngineResult<QueryOutput>, QueryMetrics, u64) {
    let admitted = context.admit_plan(plan).unwrap();
    let admitted_plan =
        admit_query_plan(plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let usage = admitted
        .attempt_allocation_domain_usage()
        .expect("allocation-domain usage samples")
        .expect("admitted context owns one allocation domain");
    admitted.finish();
    (result, metrics.snapshot(), usage.peak_bytes)
}

pub(super) fn assert_attempt_domain_released(context: &ExecutionContext) {
    let usage = context
        .attempt_allocation_domain_usage()
        .expect("allocation-domain usage samples")
        .expect("admitted context owns one allocation domain");
    assert_eq!(usage.current_bytes, 0, "failed work must be released inside the attempt domain: {usage:?}");
    assert_eq!(usage.current_allocations, 0, "no allocation may outlive the failed work: {usage:?}");
}

pub(super) fn wait_for_ledger_release(policy: BackendTestPolicy) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let reports =
            suite_backend_with_policy(policy).session_capability().device_ledger_reports().expect("ledger reports");
        let report = reports.iter().find(|report| report.device_ordinal == 0).expect("device 0 ledger");
        if report.active_attempts == 0 && report.grants.is_empty() && report.live_grants_bytes == 0 {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "finished attempt must release its grant: {report:?}");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
