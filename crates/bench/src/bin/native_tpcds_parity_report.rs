use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::native_report::bench_support::{NativeTpcdsParityReport, ParityFloatTolerance};
use bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        RelationalExecutionSelection,
        tpcds::{
            TpcdsExternalEngineConfig, TpcdsExternalOracleConfig, TpcdsParityProgressEvent,
            tpcds_native_optimizer_config_with_profile_settings,
            tpcds_native_profile_settings_and_device_memory_limit_from_env,
            tpcds_parity_report_rows_with_external_oracles_and_device_memory_limit, tpcds_report_session_config,
        },
    },
};
use query_planning::mode::{StatisticalAggregateExecutionMode, StatisticalAggregateToleranceContract};
use workloads::tpcds::{WorkloadResult, prepare_tpcds_report_dataset};

const ALGEON_TPCDS_PARITY_QUERY_IDS_ENV: &str = "ALGEON_TPCDS_PARITY_QUERY_IDS";
const ALGEON_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV: &str = "ALGEON_TPCDS_PARITY_FLOAT_ABS_TOLERANCE";
const ALGEON_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV: &str = "ALGEON_TPCDS_PARITY_FLOAT_REL_TOLERANCE";
const ALGEON_TPCDS_PARITY_DUCKDB_BIN_ENV: &str = "ALGEON_TPCDS_PARITY_DUCKDB_BIN";

#[tokio::main]
async fn main() {
    let dir = unique_temp_dir("algeon-datafusion-native-tpcds-parity");
    let stdout = io::stdout();
    let mut output = stdout.lock();
    write_parity_report(&dir, &mut output).await.unwrap();
    fs::remove_dir_all(dir).unwrap();
}

async fn write_parity_report(dir: &std::path::Path, output: &mut impl Write) -> WorkloadResult<()> {
    let (root, metadata) = prepare_tpcds_report_dataset(dir)?;
    let (profile, device_memory_limit) = tpcds_native_profile_settings_and_device_memory_limit_from_env();
    let optimizer_config = RelationalExecutionSelection::strict_native()
        .configure_native_optimizer(tpcds_native_optimizer_config_with_profile_settings(&profile))
        .expect("TPC-DS parity installs the strict native rule");
    let statistical_aggregate_execution_mode = optimizer_config.statistical_aggregate_execution_mode();
    let query_filter = query_filter_from_env();
    let external_oracles = external_oracle_config_from_env(dir.join("external_oracles"));
    let tolerance = parity_float_tolerance_from_env();
    validate_statistical_aggregate_parity_contract(statistical_aggregate_execution_mode, tolerance);

    let header = NativeTpcdsParityReport::from_rows(Vec::new(), tolerance)
        .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode)
        .with_workload_dataset_metadata(metadata.to_workload_dataset_metadata())
        .to_tsv();
    output.write_all(header.as_bytes())?;
    output.flush()?;

    let mut progress = |event: TpcdsParityProgressEvent<'_>| {
        if let TpcdsParityProgressEvent::RowReady { row, .. } = event {
            writeln!(output, "{}", row.to_tsv_line()).expect("failed to write TPC-DS parity TSV row");
            output.flush().expect("failed to flush TPC-DS parity TSV row");
        }
    };
    tpcds_parity_report_rows_with_external_oracles_and_device_memory_limit(
        &root,
        tpcds_report_session_config(),
        optimizer_config,
        device_memory_limit,
        query_filter.as_deref(),
        external_oracles,
        tolerance,
        Some(&mut progress),
    )
    .await?;
    Ok(())
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn query_filter_from_env() -> Option<Vec<String>> {
    let raw = std::env::var(ALGEON_TPCDS_PARITY_QUERY_IDS_ENV).ok()?;
    let filtered: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if filtered.is_empty() { None } else { Some(filtered) }
}

fn parity_float_tolerance_from_env() -> ParityFloatTolerance {
    let absolute = std::env::var(ALGEON_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let relative = std::env::var(ALGEON_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    ParityFloatTolerance::new(absolute, relative)
}

fn validate_statistical_aggregate_parity_contract(
    mode: StatisticalAggregateExecutionMode,
    tolerance: ParityFloatTolerance,
) {
    if mode.tolerance_contract() == StatisticalAggregateToleranceContract::FloatTolerant && !tolerance.is_enabled() {
        panic!(
            "invalid TPC-DS parity configuration: {mode} requires non-zero {ALGEON_TPCDS_PARITY_FLOAT_ABS_TOLERANCE_ENV} or {ALGEON_TPCDS_PARITY_FLOAT_REL_TOLERANCE_ENV}"
        );
    }
}

fn external_oracle_config_from_env(scratch_dir: PathBuf) -> Option<TpcdsExternalOracleConfig> {
    let duckdb = std::env::var_os(ALGEON_TPCDS_PARITY_DUCKDB_BIN_ENV)
        .map(|binary| TpcdsExternalEngineConfig { binary: PathBuf::from(binary) });
    if duckdb.is_none() { None } else { Some(TpcdsExternalOracleConfig { scratch_dir, duckdb }) }
}
