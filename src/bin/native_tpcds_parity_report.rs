use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_cudf::{
    NativePipelineExecutionMode, NativeTpcdsParityReport, ParityFloatTolerance, TpcdsDatasetMetadata,
    prepare_tpcds_report_dataset, tpcds_native_optimizer_config, tpcds_parity_workload_items,
    tpcds_report_session_config,
};

const CUDF_TPCDS_PARITY_QUERY_IDS_ENV: &str = "CUDF_TPCDS_PARITY_QUERY_IDS";
const CUDF_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV: &str = "CUDF_TPCDS_PARITY_FLOAT_ABS_TOLERANCE";
const CUDF_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV: &str = "CUDF_TPCDS_PARITY_FLOAT_REL_TOLERANCE";
const CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-cudf-native-tpcds-parity");
    let TpcdsParityReport { report, metadata } = parity_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn parity_report(dir: &std::path::Path) -> datafusion_cudf::TpcdsWorkloadResult<TpcdsParityReport> {
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let optimizer_config =
        tpcds_native_optimizer_config().with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env());
    let query_filter = query_filter_from_env();
    let items =
        tpcds_parity_workload_items(&root, tpcds_report_session_config(), optimizer_config, query_filter.as_deref())
            .await?;
    let tolerance = parity_float_tolerance_from_env();
    let report = NativeTpcdsParityReport::collect(items, tolerance);
    Ok(TpcdsParityReport { report, metadata })
}

struct TpcdsParityReport {
    report: NativeTpcdsParityReport,
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
    let raw = std::env::var(CUDF_TPCDS_PARITY_QUERY_IDS_ENV).ok()?;
    let filtered: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if filtered.is_empty() { None } else { Some(filtered) }
}

fn parity_float_tolerance_from_env() -> ParityFloatTolerance {
    let absolute = std::env::var(CUDF_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let relative = std::env::var(CUDF_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    ParityFloatTolerance::new(absolute, relative)
}
