use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::{NativeDataFusionExecutionReport, NativePipelineExecutionMode};
use datafusion_nexus_bench::{
    TpchDatasetMetadata, prepare_tpch_report_dataset, tpch_datafusion_execution_workload_items_with_optimizer_config,
    tpch_native_optimizer_config, tpch_report_session_config,
};

const CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-tpch-execution");
    let TpchExecutionReport { report, metadata } = tpch_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpch_report(dir: &std::path::Path) -> datafusion_nexus_bench::TpchWorkloadResult<TpchExecutionReport> {
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let optimizer_config =
        tpch_native_optimizer_config().with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env());
    let items = tpch_datafusion_execution_workload_items_with_optimizer_config(
        &root,
        tpch_report_session_config(),
        optimizer_config,
    )
    .await?;
    let report = NativeDataFusionExecutionReport::collect(items);
    Ok(TpchExecutionReport { report, metadata })
}

struct TpchExecutionReport {
    report: NativeDataFusionExecutionReport,
    metadata: TpchDatasetMetadata,
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn native_pipeline_execution_mode_from_env() -> NativePipelineExecutionMode {
    let mode = std::env::var(CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV)
        .unwrap_or_else(|_| NativePipelineExecutionMode::StrictBounded.as_str().to_string());
    mode.parse::<NativePipelineExecutionMode>().unwrap_or_else(|error| {
        panic!(
            "unsupported {CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV}={}; expected strict_bounded or adaptive",
            error.value()
        )
    })
}
