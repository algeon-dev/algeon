use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::native_report::NativeDataFusionExecutionReport;
use datafusion_nexus_bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        gpu_fallback_policy_label, parse_gpu_fallback_policy,
        tpch::{
            tpch_datafusion_execution_workload_items_with_optimizer_config, tpch_native_optimizer_config,
            tpch_report_session_config,
        },
    },
};
use datafusion_nexus_workloads::tpch::{TpchDatasetMetadata, TpchWorkloadResult, prepare_tpch_report_dataset};
use nexus_query_engine::mode::NativePipelineExecutionMode;

const CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";
const CUDF_TPCH_NATIVE_FALLBACK_POLICY_ENV: &str = "CUDF_TPCH_NATIVE_FALLBACK_POLICY";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-tpch-execution");
    let TpchExecutionReport { report, metadata } = tpch_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpch_report(dir: &std::path::Path) -> TpchWorkloadResult<TpchExecutionReport> {
    let fallback_policy = native_fallback_policy_from_env();
    let native_pipeline_execution_mode = native_pipeline_execution_mode_from_env();
    eprintln!(
        "[native_tpch_execution_report] stage=run_config fallback_policy={} native_pipeline_execution_mode={}",
        gpu_fallback_policy_label(fallback_policy),
        native_pipeline_execution_mode.as_str(),
    );
    let (root, metadata) = prepare_tpch_report_dataset(dir)?;
    let optimizer_config = tpch_native_optimizer_config()
        .with_fallback_policy(fallback_policy)
        .with_native_pipeline_execution_mode(native_pipeline_execution_mode);
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

fn native_fallback_policy_from_env() -> datafusion_nexus::policy::GpuFallbackPolicy {
    match std::env::var(CUDF_TPCH_NATIVE_FALLBACK_POLICY_ENV) {
        Ok(value) => parse_gpu_fallback_policy(&value).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => datafusion_nexus::policy::GpuFallbackPolicy::GpuPreferred,
        Err(error) => panic!("failed to read {CUDF_TPCH_NATIVE_FALLBACK_POLICY_ENV}: {error}"),
    }
}
