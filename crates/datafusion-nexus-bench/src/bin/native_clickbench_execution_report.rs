use std::{
    io::{self, Write as _},
    process::ExitCode,
};

use datafusion_nexus::native_report::NativeDataFusionExecutionReport;
use datafusion_nexus_bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        clickbench::{
            ClickbenchExecutionProgressEvent, clickbench_datafusion_execution_workload_items_filtered,
            clickbench_native_optimizer_config, clickbench_native_profile_settings_from_env,
            clickbench_report_session_config,
        },
        gpu_fallback_policy_label, parse_gpu_fallback_policy,
    },
};
use datafusion_nexus_workloads::clickbench::{
    ClickbenchDatasetMetadata, ClickbenchWorkloadError, ClickbenchWorkloadResult, prepare_clickbench_report_dataset,
};
use nexus_query_engine::mode::NativePipelineExecutionMode;

const NEXUS_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "NEXUS_NATIVE_PIPELINE_EXECUTION_MODE";
const NEXUS_CLICKBENCH_NATIVE_FALLBACK_POLICY_ENV: &str = "NEXUS_CLICKBENCH_NATIVE_FALLBACK_POLICY";
const NEXUS_CLICKBENCH_EXECUTION_QUERY_IDS_ENV: &str = "NEXUS_CLICKBENCH_EXECUTION_QUERY_IDS";
const NEXUS_CLICKBENCH_EXECUTION_PROGRESS_ENV: &str = "NEXUS_CLICKBENCH_EXECUTION_PROGRESS";

#[tokio::main]
async fn main() -> ExitCode {
    match clickbench_report().await {
        Ok(ClickbenchExecutionReport { report, metadata }) => {
            print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
            ExitCode::SUCCESS
        },
        Err(error) => {
            let mut stderr = io::stderr().lock();
            let _ = writeln!(stderr, "native_clickbench_execution_report: {error}");
            ExitCode::FAILURE
        },
    }
}

async fn clickbench_report() -> ClickbenchWorkloadResult<ClickbenchExecutionReport> {
    let fallback_policy = native_fallback_policy_from_env();
    let native_pipeline_execution_mode = native_pipeline_execution_mode_from_env();
    let native_profile = clickbench_native_profile_settings_from_env();
    eprintln!(
        "[native_clickbench_execution_report] stage=run_config fallback_policy={} native_profile={} native_pipeline_execution_mode={} device_budget_bytes={} max_source_chunk_bytes={} max_row_groups_per_chunk={} source_chunk_planning_mode={}",
        gpu_fallback_policy_label(fallback_policy),
        native_profile.profile_name,
        native_pipeline_execution_mode.as_str(),
        native_profile.device_budget_bytes,
        native_profile.max_source_chunk_bytes,
        native_profile.max_row_groups_per_chunk,
        native_profile.source_chunk_planning_mode.as_str(),
    );
    let (root, metadata) = prepare_clickbench_report_dataset()?.ok_or_else(|| {
        ClickbenchWorkloadError::InvalidDataset(
            "no ClickBench fixture found; set NEXUS_PARQUET_DIR or stage \
             fixture/clickbench/data/{hits.parquet, partitioned/} before running this report"
                .to_string(),
        )
    })?;
    let optimizer_config = clickbench_native_optimizer_config()
        .with_fallback_policy(fallback_policy)
        .with_native_pipeline_execution_mode(native_pipeline_execution_mode);
    let query_filter = query_filter_from_env();
    let progress_enabled = progress_enabled_from_env();
    let mut progress_callback = |event: ClickbenchExecutionProgressEvent<'_>| {
        let mut stderr = io::stderr().lock();
        match event {
            ClickbenchExecutionProgressEvent::QueryStarted { query_index, total, query_id } => {
                let _ = writeln!(stderr, "[clickbench-exec] start  {query_index}/{total} {query_id}");
            },
            ClickbenchExecutionProgressEvent::QueryFinished {
                query_index,
                total,
                query_id,
                elapsed_ms,
                runtime_outcome,
            } => {
                let _ = writeln!(
                    stderr,
                    "[clickbench-exec] finish {query_index}/{total} {query_id} elapsed_ms={elapsed_ms} outcome={runtime_outcome}"
                );
            },
        }
    };
    let progress = if progress_enabled {
        Some(&mut progress_callback as &mut dyn FnMut(ClickbenchExecutionProgressEvent<'_>))
    } else {
        None
    };
    let query_ids: Option<Vec<&str>> = query_filter.as_ref().map(|ids| ids.iter().map(String::as_str).collect());
    let items = clickbench_datafusion_execution_workload_items_filtered(
        &root,
        clickbench_report_session_config(),
        optimizer_config,
        query_ids.as_deref(),
        progress,
    )
    .await?;
    let report = NativeDataFusionExecutionReport::collect(items);
    Ok(ClickbenchExecutionReport { report, metadata })
}

struct ClickbenchExecutionReport {
    report: NativeDataFusionExecutionReport,
    metadata: ClickbenchDatasetMetadata,
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
    let raw = std::env::var(NEXUS_CLICKBENCH_EXECUTION_QUERY_IDS_ENV).ok()?;
    let filtered: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if filtered.is_empty() { None } else { Some(filtered) }
}

fn progress_enabled_from_env() -> bool {
    match std::env::var(NEXUS_CLICKBENCH_EXECUTION_PROGRESS_ENV) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

fn native_fallback_policy_from_env() -> datafusion_nexus::policy::GpuFallbackPolicy {
    match std::env::var(NEXUS_CLICKBENCH_NATIVE_FALLBACK_POLICY_ENV) {
        Ok(value) => parse_gpu_fallback_policy(&value).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => datafusion_nexus::policy::GpuFallbackPolicy::GpuPreferred,
        Err(error) => panic!("failed to read {NEXUS_CLICKBENCH_NATIVE_FALLBACK_POLICY_ENV}: {error}"),
    }
}
