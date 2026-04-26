use std::{
    fs,
    io::{self, Write as _},
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::native_report::NativeDataFusionExecutionReport;
use datafusion_nexus_bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        gpu_fallback_policy_label, parse_gpu_fallback_policy,
        tpcds::{
            TpcdsExecutionProgressEvent, tpcds_datafusion_execution_workload_items_filtered,
            tpcds_native_analytics_optimizer_config, tpcds_report_session_config,
        },
    },
};
use datafusion_nexus_workloads::tpcds::{TpcdsDatasetMetadata, TpcdsWorkloadResult, prepare_tpcds_report_dataset};
use nexus_query_engine::mode::NativePipelineExecutionMode;

const NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "NEXUS_NATIVE_PIPELINE_EXECUTION_MODE";
const NEXUS_TPCDS_NATIVE_FALLBACK_POLICY_ENV: &str = "NEXUS_TPCDS_NATIVE_FALLBACK_POLICY";
const NEXUS_TPCDS_EXECUTION_QUERY_IDS_ENV: &str = "NEXUS_TPCDS_EXECUTION_QUERY_IDS";
const NEXUS_TPCDS_EXECUTION_PROGRESS_ENV: &str = "NEXUS_TPCDS_EXECUTION_PROGRESS";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-tpcds-execution");
    let TpcdsExecutionReport { report, metadata } = tpcds_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpcds_report(dir: &std::path::Path) -> TpcdsWorkloadResult<TpcdsExecutionReport> {
    let fallback_policy = native_fallback_policy_from_env();
    let native_pipeline_execution_mode = native_pipeline_execution_mode_from_env();
    eprintln!(
        "[native_tpcds_execution_report] stage=run_config fallback_policy={} native_pipeline_execution_mode={}",
        gpu_fallback_policy_label(fallback_policy),
        native_pipeline_execution_mode.as_str(),
    );
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let optimizer_config = tpcds_native_analytics_optimizer_config()
        .with_fallback_policy(fallback_policy)
        .with_native_pipeline_execution_mode(native_pipeline_execution_mode);
    let statistical_aggregate_execution_mode = optimizer_config.statistical_aggregate_execution_mode();
    let query_filter = query_filter_from_env();
    let progress_enabled = progress_enabled_from_env();
    let mut progress_callback = |event: TpcdsExecutionProgressEvent<'_>| {
        let mut stderr = io::stderr().lock();
        match event {
            TpcdsExecutionProgressEvent::QueryStarted { query_index, total, query_id } => {
                let _ = writeln!(stderr, "[tpcds-exec] start  {query_index}/{total} {query_id}");
            },
            TpcdsExecutionProgressEvent::QueryFinished {
                query_index,
                total,
                query_id,
                elapsed_ms,
                runtime_outcome,
            } => {
                let _ = writeln!(
                    stderr,
                    "[tpcds-exec] finish {query_index}/{total} {query_id} elapsed_ms={elapsed_ms} outcome={runtime_outcome}"
                );
            },
        }
    };
    let progress = if progress_enabled {
        Some(&mut progress_callback as &mut dyn FnMut(TpcdsExecutionProgressEvent<'_>))
    } else {
        None
    };
    let query_ids: Option<Vec<&str>> = query_filter.as_ref().map(|ids| ids.iter().map(String::as_str).collect());
    let items = tpcds_datafusion_execution_workload_items_filtered(
        &root,
        tpcds_report_session_config(),
        optimizer_config,
        query_ids.as_deref(),
        progress,
    )
    .await?;
    let report = NativeDataFusionExecutionReport::collect(items)
        .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode);
    Ok(TpcdsExecutionReport { report, metadata })
}

struct TpcdsExecutionReport {
    report: NativeDataFusionExecutionReport,
    metadata: TpcdsDatasetMetadata,
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn native_pipeline_execution_mode_from_env() -> NativePipelineExecutionMode {
    let mode = std::env::var(NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV)
        .unwrap_or_else(|_| NativePipelineExecutionMode::StrictBounded.as_str().to_string());
    mode.parse::<NativePipelineExecutionMode>().unwrap_or_else(|error| {
        panic!(
            "unsupported {NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV}={}; expected strict_bounded or adaptive",
            error.value()
        )
    })
}

fn query_filter_from_env() -> Option<Vec<String>> {
    let raw = std::env::var(NEXUS_TPCDS_EXECUTION_QUERY_IDS_ENV).ok()?;
    let filtered: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if filtered.is_empty() { None } else { Some(filtered) }
}

fn progress_enabled_from_env() -> bool {
    match std::env::var(NEXUS_TPCDS_EXECUTION_PROGRESS_ENV) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

fn native_fallback_policy_from_env() -> datafusion_nexus::policy::GpuFallbackPolicy {
    match std::env::var(NEXUS_TPCDS_NATIVE_FALLBACK_POLICY_ENV) {
        Ok(value) => parse_gpu_fallback_policy(&value).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => datafusion_nexus::policy::GpuFallbackPolicy::GpuPreferred,
        Err(error) => panic!("failed to read {NEXUS_TPCDS_NATIVE_FALLBACK_POLICY_ENV}: {error}"),
    }
}
