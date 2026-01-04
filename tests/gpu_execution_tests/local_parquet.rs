use std::{fs, sync::Arc};

use datafusion::{
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::collect,
    prelude::{ParquetReadOptions, SessionContext},
};
use datafusion_cudf::{CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuFallbackPolicy};

use super::support::{ARROW_HOST_OBSERVATION_LOCK, input_batch, int32_values, unique_temp_dir, write_parquet_file};

#[test]
fn gpu_island_reads_local_parquet_without_record_batch_upload() {
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().unwrap();
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let dir = unique_temp_dir("datafusion-cudf-local-parquet");
    let path = dir.join("sample.parquet");
    write_parquet_file(&path, &input_batch());

    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(1);
    let state = SessionStateBuilder::new().with_cudf_gpu_islands(config).build();
    let ctx = SessionContext::new_with_state(state);
    let df = futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default()))
        .unwrap()
        .select_columns(&["id", "value"])
        .unwrap();
    let plan = futures::executor::block_on(df.create_physical_plan()).unwrap();

    assert_eq!(plan.name(), "CudfGpuIslandExec");
    let batches = futures::executor::block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(int32_values(&batches[0], 0), vec![1, 2, 3, 4]);
    assert_eq!(int32_values(&batches[0], 1), vec![10, 20, 30, 40]);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = plan.metrics().expect("GPU island should expose execution metrics");
    assert_eq!(metrics.sum_by_name("parquet_files").unwrap().as_usize(), 1);
    assert!(metrics.sum_by_name("parquet_bytes_known").unwrap().as_usize() > 0);
    assert_eq!(metrics.sum_by_name("read_columns").unwrap().as_usize(), 2);
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);

    fs::remove_dir_all(dir).unwrap();
}
