use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{record_batch::RecordBatch, util::display::array_value_to_string};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, utils::memory::get_record_batch_memory_size},
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect, metrics::MetricsSet},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV,
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error as AdapterError,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome,
    NativeDataFusionPlanningReport, NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath,
    NativeDataFusionRuntimeOutcome, NativePipelineExecutionMode, TPCH_QUERIES, TPCH_TABLES, TpchDatasetMetadata,
    TpchQuery, TpchWorkloadError, describe_tpch_dataset_with_content_fingerprint, prepare_tpch_report_dataset,
    tpch_report_session_config,
};

#[path = "native_tpch_benchmark/case_runner.rs"]
mod case_runner;
#[path = "native_tpch_benchmark/config.rs"]
mod config;
#[path = "native_tpch_benchmark/dataset.rs"]
mod dataset;
#[path = "native_tpch_benchmark/errors.rs"]
mod errors;
#[path = "native_tpch_benchmark/execution.rs"]
mod execution;
#[path = "native_tpch_benchmark/reports.rs"]
mod reports;
#[path = "native_tpch_benchmark/validation.rs"]
mod validation;

use case_runner::*;
use config::*;
use errors::*;
use reports::*;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let config = NativeTpchBenchmarkConfig::from_env()?;
    let work_dir = unique_temp_dir("datafusion-cudf-native-tpch-benchmark");
    let report = run_native_tpch_benchmark(&config, &work_dir).await?;
    print!("{}", report.to_tsv());
    fs::remove_dir_all(work_dir).ok();
    Ok(())
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
#[path = "native_tpch_benchmark/tests.rs"]
mod tests;
