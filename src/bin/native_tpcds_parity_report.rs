use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::{
    CudfOptimizerConfig, GpuFallbackPolicy, NativePipelineExecutionMode, NativeTpcdsParityReport, ParityFloatTolerance,
    StatisticalAggregateExecutionMode, TpcdsDatasetMetadata, TpcdsExternalEngineConfig, TpcdsExternalOracleConfig,
    prepare_tpcds_report_dataset, tpcds_native_optimizer_config, tpcds_parity_workload_items_with_external_oracles,
    tpcds_report_session_config,
};

const CUDF_TPCDS_PARITY_QUERY_IDS_ENV: &str = "CUDF_TPCDS_PARITY_QUERY_IDS";
const CUDF_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV: &str = "CUDF_TPCDS_PARITY_FLOAT_ABS_TOLERANCE";
const CUDF_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV: &str = "CUDF_TPCDS_PARITY_FLOAT_REL_TOLERANCE";
const CUDF_TPCDS_PARITY_DUCKDB_BIN_ENV: &str = "CUDF_TPCDS_PARITY_DUCKDB_BIN";
const CUDF_TPCDS_PARITY_external-engine_BIN_ENV: &str = "CUDF_TPCDS_PARITY_external-engine_BIN";
const CUDF_TPCDS_PARITY_external-engine_CONFIG_FILE_ENV: &str = "CUDF_TPCDS_PARITY_external-engine_CONFIG_FILE";
const CUDF_TPCDS_PARITY_SKIP_NATIVE_ENV: &str = "CUDF_TPCDS_PARITY_SKIP_NATIVE";
const CUDF_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("datafusion-nexus-native-tpcds-parity");
    let TpcdsParityReport { report, metadata } = parity_report(&dir).await.unwrap();
    print!("{}", report.with_workload_dataset_metadata(metadata.to_workload_dataset_metadata()).to_tsv());
    fs::remove_dir_all(dir).unwrap();
}

async fn parity_report(dir: &std::path::Path) -> datafusion_nexus::TpcdsWorkloadResult<TpcdsParityReport> {
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let optimizer_config = native_optimizer_config_from_env();
    let statistical_aggregate_execution_mode = optimizer_config.statistical_aggregate_execution_mode();
    let query_filter = query_filter_from_env();
    let external_oracles = external_oracle_config_from_env(dir.join("external_oracles"));
    let tolerance = parity_float_tolerance_from_env();
    validate_statistical_aggregate_parity_contract(statistical_aggregate_execution_mode, tolerance);
    let items = tpcds_parity_workload_items_with_external_oracles(
        &root,
        tpcds_report_session_config(),
        optimizer_config,
        query_filter.as_deref(),
        external_oracles,
    )
    .await?;
    let report = NativeTpcdsParityReport::collect(items, tolerance)
        .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode);
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

fn native_optimizer_config_from_env() -> CudfOptimizerConfig {
    let base = if bool_env(CUDF_TPCDS_PARITY_SKIP_NATIVE_ENV) {
        CudfOptimizerConfig::new(GpuFallbackPolicy::CpuOnly)
    } else {
        tpcds_native_optimizer_config()
    };
    base.with_native_pipeline_execution_mode(native_pipeline_execution_mode_from_env())
}

fn bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
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

fn validate_statistical_aggregate_parity_contract(
    mode: StatisticalAggregateExecutionMode,
    tolerance: ParityFloatTolerance,
) {
    if mode.tolerance_contract() == datafusion_nexus::StatisticalAggregateToleranceContract::FloatTolerant
        && !tolerance.is_enabled()
    {
        panic!(
            "invalid TPC-DS parity configuration: {mode} requires non-zero {CUDF_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV} or {CUDF_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV}"
        );
    }
}

fn external_oracle_config_from_env(scratch_dir: PathBuf) -> Option<TpcdsExternalOracleConfig> {
    let duckdb = std::env::var_os(CUDF_TPCDS_PARITY_DUCKDB_BIN_ENV).map(|binary| TpcdsExternalEngineConfig {
        binary: PathBuf::from(binary),
        external-engine_config_file: None,
        disable_external-engine: true,
    });
    let external-engine = std::env::var_os(CUDF_TPCDS_PARITY_external-engine_BIN_ENV).map(|binary| TpcdsExternalEngineConfig {
        binary: PathBuf::from(binary),
        external-engine_config_file: std::env::var_os(CUDF_TPCDS_PARITY_external-engine_CONFIG_FILE_ENV).map(PathBuf::from),
        disable_external-engine: false,
    });
    if duckdb.is_none() && external-engine.is_none() {
        None
    } else {
        Some(TpcdsExternalOracleConfig { scratch_dir, duckdb, external-engine })
    }
}
