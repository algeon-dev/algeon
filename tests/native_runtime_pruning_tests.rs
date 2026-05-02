use std::{
    fs::{self, File},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use cudf_query_engine::{ExecutionContext, QueryOutput, execute_query_plan};
use datafusion::{
    parquet::{
        arrow::ArrowWriter,
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
    },
    physical_plan::{ExecutionPlan, collect},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::lower_physical_plan_to_native;

static ARROW_HOST_OBSERVATION_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn native_runtime_default_sql_sum_expression_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-sum");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_min_max_count_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-min-max-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_MIN_MAX_COUNT_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_count_star_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-count-star");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_COUNT_STAR_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_prunes_metadata_row_groups_for_pushed_parquet_predicate() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn assert_default_sql_native_parity(
    path: &std::path::Path,
    sql: &str,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    let cpu_batches = cpu_sql_output(path, sql).await;
    let native_physical_plan = sql_plan(path, SessionConfig::new(), sql).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(lowering.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(runtime);
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();

    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(total_row_groups, 4);
        assert_eq!(metrics.row_groups_selected, selected_row_groups as u64);
        assert!(metrics.row_groups_selected < total_row_groups as u64);
    }
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.host_to_device_uploaded_rows, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));
}

async fn cpu_sql_output(path: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = sql_plan(path, SessionConfig::new(), sql).await;
    collect(plan, Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap()
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

const SQL_SUM_EXPRESSION_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_MIN_MAX_COUNT_QUERY: &str = "
SELECT key, MIN(m00) AS min_m00, MAX(m01) AS max_m01, COUNT(payload) AS count_payload
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

const SQL_COUNT_STAR_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00, COUNT(*) AS count_rows
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

fn write_row_group_pruning_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 10, 11, 20, 21])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn assert_batches_equal(left: &[RecordBatch], right: &[RecordBatch]) {
    assert!(!left.is_empty());
    assert!(!right.is_empty());
    assert_schema_names_and_types_equal(&left[0], &right[0]);
    assert_eq!(batch_rows(left), batch_rows(right));
}

fn assert_schema_names_and_types_equal(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_columns(), right.num_columns());
    for index in 0..left.num_columns() {
        assert_eq!(left.schema().field(index).name(), right.schema().field(index).name());
        assert_eq!(left.schema().field(index).data_type(), right.schema().field(index).data_type());
    }
}

fn batch_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());
            for column in 0..batch.num_columns() {
                values.push(array_value(batch.column(column), row));
            }
            rows.push(values);
        }
    }
    rows
}

fn array_value(array: &ArrayRef, row: usize) -> String {
    array_value_to_string(array.as_ref(), row).unwrap()
}

fn parquet_row_group_count(path: &std::path::Path) -> usize {
    SerializedFileReader::try_from(path).unwrap().metadata().num_row_groups()
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
