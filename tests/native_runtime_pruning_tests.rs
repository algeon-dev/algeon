use std::{
    fs::{self, File},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
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
async fn native_runtime_prunes_metadata_row_groups_for_pushed_parquet_predicate() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    let cpu_batches = cpu_sql_output(&path).await;
    let native_physical_plan = sql_plan(&path, direct_friendly_sql_session_config()).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1);
    assert_eq!(lowering.rejected_count(), 0);
    assert_eq!(lowering.plans.len(), 1);
    assert_eq!(lowering.source_diagnostics[0].selected_row_groups, Some(2));

    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(runtime);
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();

    assert_eq!(total_row_groups, 4);
    assert_eq!(metrics.row_groups_selected, 2);
    assert!(metrics.row_groups_selected < total_row_groups as u64);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_eq!(cpu_batches.len(), 1);
    assert_batches_equal(&cpu_batches[0], native_batch);

    fs::remove_dir_all(dir).unwrap();
}

async fn cpu_sql_output(path: &std::path::Path) -> Vec<RecordBatch> {
    let plan = sql_plan(path, SessionConfig::new().with_target_partitions(1)).await;
    collect(plan, Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap()
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(SQL_BENCHMARK_QUERY).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

fn direct_friendly_sql_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

const SQL_BENCHMARK_QUERY: &str = "
SELECT key, MIN(m00) AS min_m00, MAX(m01) AS max_m01, COUNT(payload) AS count_payload
FROM bench
WHERE payload >= 10
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

fn assert_batches_equal(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_columns(), 4);
    assert_eq!(right.num_columns(), 4);
    for index in 0..4 {
        assert_eq!(left.schema().field(index).name(), right.schema().field(index).name());
        assert_eq!(left.schema().field(index).data_type(), right.schema().field(index).data_type());
    }
    assert_eq!(left.num_rows(), right.num_rows());
    assert_eq!(int32_values(left, 0), int32_values(right, 0));
    assert_eq!(int64_values(left, 1), int64_values(right, 1));
    assert_eq!(int64_values(left, 2), int64_values(right, 2));
    assert_eq!(int64_values(left, 3), int64_values(right, 3));
}

fn int32_values(batch: &RecordBatch, index: usize) -> Vec<i32> {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array").values().to_vec()
}

fn int64_values(batch: &RecordBatch, index: usize) -> Vec<i64> {
    batch.column(index).as_any().downcast_ref::<Int64Array>().expect("expected Int64Array").values().to_vec()
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
