use std::{
    fs,
    io::{self, Write as _},
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_cudf::{
    NativeDataFusionExecutionReport, NativePipelineExecutionMode, TpcdsDatasetMetadata, TpcdsExecutionProgressEvent,
    prepare_tpcds_report_dataset, tpcds_datafusion_execution_workload_items_filtered, tpcds_native_optimizer_config,
    tpcds_report_session_config,
};

const CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";
const CUDF_TPCDS_EXECUTION_QUERY_IDS_ENV: &str = "CUDF_TPCDS_EXECUTION_QUERY_IDS";
const CUDF_TPCDS_EXECUTION_PROGRESS_ENV: &str = "CUDF_TPCDS_EXECUTION_PROGRESS";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-cudf-native-tpcds-execution");
    let TpcdsExecutionReport { report, metadata } = tpcds_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn tpcds_report(dir: &std::path::Path) -> datafusion_cudf::TpcdsWorkloadResult<TpcdsExecutionReport> {
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let optimizer_config =
        tpcds_native_optimizer_config().with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env());
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
    let report = NativeDataFusionExecutionReport::collect(items);
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
    let mode = std::env::var(CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV)
        .unwrap_or_else(|_| NativePipelineExecutionMode::StrictBounded.as_str().to_string());
    mode.parse::<NativePipelineExecutionMode>().unwrap_or_else(|error| {
        panic!(
            "unsupported {CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV}={}; expected strict_bounded or adaptive",
            error.value()
        )
    })
}

fn query_filter_from_env() -> Option<Vec<String>> {
    let raw = std::env::var(CUDF_TPCDS_EXECUTION_QUERY_IDS_ENV).ok()?;
    let filtered: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if filtered.is_empty() { None } else { Some(filtered) }
}

fn progress_enabled_from_env() -> bool {
    match std::env::var(CUDF_TPCDS_EXECUTION_PROGRESS_ENV) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}
