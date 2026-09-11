use std::{
    collections::BTreeSet,
    error::Error as StdError,
    fmt,
    io::{self, Write},
    process::Stdio,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    native_report::{
        CandidateOutcomeKind, NativeDataFusionRuntimeOutcome,
        bench_support::{ParityComparisonMode, ParityStatus},
    },
    table_format::{TableFormat, TableFormatScanPlan},
};
use tokio::process::Command;
use workloads::tpch::TPCH_QUERIES;

use crate::iceberg_smoke::{
    IcebergNativeSmokeConfig, IcebergNativeSmokeFixture, IcebergNativeSmokeReport, IcebergNativeSmokeReportRow,
    ObjectStoreStagingMode, ObjectStoreStagingStatus, ObjectStoreStagingSummary, SmokeQueryExecutionMode, run_report,
    run_single_tpch_query_report_with_execution_mode, stable_tsv_field,
};

type BenchmarkResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;

pub const REPORT_HEADER: &str = "query_id\titeration\tfixture_mode\tcatalog_type\ttable_ident\ttpch_scale_factor\ttable_format\tsnapshot_id\tplanned_data_file_count\tplanned_bytes\tadmitted_native_source_file_count\tnative_source_read_capability\tsource_read_strategy\tcredential_mode\tlakehouse_cache_status\tlakehouse_cache_read_mode\tlakehouse_cache_fill_mode\tdirect_device_read_disposition\tobject_store_staging_mode\tobject_store_staging_status\tstaged_data_file_count\tstaged_bytes\tnative_planning_outcome\tnative_runtime_outcome\tnative_runtime_elapsed_ms\tnative_output_rows\tnative_output_batches\tnative_source_read_runtime_rows\tlakehouse_cache_fill_event_snapshot_rows\tcpu_runtime_outcome\tcpu_runtime_elapsed_ms\tcpu_output_rows\tcpu_output_batches\tcomparison_mode\tparity_status\tparity_detail\ttimeout_category\tnative_output_fingerprint\tcpu_output_fingerprint\treport_elapsed_ms\twarmup_iterations\tmeasured_iterations\tcredential_source";
pub const REPORT_COLUMNS: &[&str] = &[
    "query_id",
    "iteration",
    "fixture_mode",
    "catalog_type",
    "table_ident",
    "tpch_scale_factor",
    "table_format",
    "snapshot_id",
    "planned_data_file_count",
    "planned_bytes",
    "admitted_native_source_file_count",
    "native_source_read_capability",
    "source_read_strategy",
    "credential_mode",
    "lakehouse_cache_status",
    "lakehouse_cache_read_mode",
    "lakehouse_cache_fill_mode",
    "direct_device_read_disposition",
    "object_store_staging_mode",
    "object_store_staging_status",
    "staged_data_file_count",
    "staged_bytes",
    "native_planning_outcome",
    "native_runtime_outcome",
    "native_runtime_elapsed_ms",
    "native_output_rows",
    "native_output_batches",
    "native_source_read_runtime_rows",
    "lakehouse_cache_fill_event_snapshot_rows",
    "cpu_runtime_outcome",
    "cpu_runtime_elapsed_ms",
    "cpu_output_rows",
    "cpu_output_batches",
    "comparison_mode",
    "parity_status",
    "parity_detail",
    "timeout_category",
    "native_output_fingerprint",
    "cpu_output_fingerprint",
    "report_elapsed_ms",
    "warmup_iterations",
    "measured_iterations",
    "credential_source",
];

const QUERY_IDS_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_QUERY_IDS";
const WARMUPS_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_WARMUPS";
const ITERATIONS_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_ITERATIONS";
const OBJECT_STORE_STAGING_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_OBJECT_STORE_STAGING";
const QUERY_TIMEOUT_SECS_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_QUERY_TIMEOUT_SECS";
const ITERATION_TIMEOUT_SECS_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_ITERATION_TIMEOUT_SECS";
const CPU_BASELINE_TIMEOUT_SECS_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_CPU_BASELINE_TIMEOUT_SECS";
const CPU_CHILD_ENV: &str = "ALGEON_ICEBERG_TPCH_BENCH_CPU_CHILD";
const DEFAULT_TIMEOUT_ENABLED_CPU_BASELINE_TIMEOUT_SECS: u64 = 60;
const DEFAULT_QUERY_IDS: &[&str] = &["q01", "q03", "q06", "q10"];

/// Run the native Iceberg TPC-H benchmark from environment configuration.
///
/// When `ALGEON_ICEBERG_TPCH_BENCH_CPU_CHILD=1`, executes the CPU-baseline child path.
pub async fn run_main() {
    if std::env::var(CPU_CHILD_ENV).as_deref() == Ok("1") {
        match run_cpu_child_from_env().await {
            Ok(()) => return,
            Err(error) => {
                eprintln!("[native_iceberg_tpch_benchmark] cpu_child_error={}", stable_tsv_field(&error.to_string()));
                std::process::exit(1);
            },
        }
    }
    match run_from_env_with_config().await {
        Ok((config, report)) => {
            let gate_result = validate_full_sf1_staged_s3_gate_for_config(&report, &config);
            if !config.stream_rows_to_stdout() {
                print!("{}", report.to_tsv());
            }
            match gate_result {
                Ok(Some(summary)) => eprintln!("{}", summary.progress_line()),
                Ok(None) => {},
                Err(error) => {
                    eprintln!(
                        "[native_iceberg_tpch_benchmark] stage=full_sf1_staged_s3_gate outcome=failed error={}",
                        stable_tsv_field(&error.to_string())
                    );
                    std::process::exit(1);
                },
            }
            if let Some(error) = report.timeout_failure_message() {
                eprintln!(
                    "[native_iceberg_tpch_benchmark] stage=timeout_gate outcome=failed error={}",
                    stable_tsv_field(&error)
                );
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("[native_iceberg_tpch_benchmark] error={}", stable_tsv_field(&error.to_string()));
            std::process::exit(1);
        },
    }
}

async fn run_from_env_with_config() -> BenchmarkResult<(IcebergTpchBenchmarkConfig, IcebergTpchBenchmarkReport)> {
    let config = IcebergTpchBenchmarkConfig::from_env()?;
    let report = run_benchmark(config.clone()).await?;
    Ok((config, report))
}

async fn run_cpu_child_from_env() -> BenchmarkResult<()> {
    let config = IcebergTpchBenchmarkConfig::from_env()?;
    config.validate()?;
    if config.query_ids.len() != 1 {
        return Err(benchmark_error(format!(
            "{CPU_CHILD_ENV}=1 requires exactly one {QUERY_IDS_ENV}, got {}",
            config.query_ids.join(",")
        )));
    }
    let query_id = config.query_ids[0].clone();
    let started = Instant::now();
    let smoke_row = run_single_tpch_query_report_with_execution_mode(
        config.smoke_config.clone(),
        query_id,
        SmokeQueryExecutionMode::CpuBaselineOnly,
    )
    .await?;
    let row = IcebergTpchBenchmarkRow {
        iteration: 1,
        tpch_scale_factor: config.smoke_config.tpch_scale_factor.clone(),
        warmup_iterations: 0,
        measured_iterations: 1,
        report_elapsed_ms: started.elapsed().as_millis(),
        timeout_category: benchmark_timeout_category_from_smoke_row(&smoke_row),
        smoke_row,
    };
    print!("{}", IcebergTpchBenchmarkReport { rows: vec![row] }.to_tsv());
    Ok(())
}

pub async fn run_benchmark(config: IcebergTpchBenchmarkConfig) -> BenchmarkResult<IcebergTpchBenchmarkReport> {
    config.validate()?;
    let stream_rows_to_stdout = config.stream_rows_to_stdout();
    if stream_rows_to_stdout {
        write_benchmark_tsv_header_to_stdout()?;
    }
    for warmup_index in 0..config.warmup_iterations {
        eprintln!(
            "[native_iceberg_tpch_benchmark] stage=warmup_start index={}/{} query_ids={}",
            warmup_index + 1,
            config.warmup_iterations,
            config.query_ids.join(","),
        );
        run_report_with_timeout(
            config.smoke_config.clone(),
            config.iteration_timeout_secs,
            BenchmarkTimeoutCategory::Iteration,
        )
        .await?;
    }

    let mut rows = Vec::new();
    for iteration_index in 0..config.iterations {
        let iteration_started = Instant::now();
        let iteration_row_start = rows.len();
        eprintln!(
            "[native_iceberg_tpch_benchmark] stage=iteration_start index={}/{} query_ids={}",
            iteration_index + 1,
            config.iterations,
            config.query_ids.join(","),
        );
        if !config.timeouts_enabled() {
            let smoke_report = run_report(config.smoke_config.clone()).await?;
            let report_elapsed_ms = iteration_started.elapsed().as_millis();
            rows.extend(smoke_report.rows.into_iter().map(|row| IcebergTpchBenchmarkRow {
                iteration: iteration_index + 1,
                tpch_scale_factor: config.smoke_config.tpch_scale_factor.clone(),
                warmup_iterations: config.warmup_iterations,
                measured_iterations: config.iterations,
                report_elapsed_ms,
                timeout_category: None,
                smoke_row: row,
            }));
            continue;
        }
        for (query_index, query_id) in config.query_ids.iter().enumerate() {
            let timeout = match query_timeout_for_iteration(&config, iteration_started) {
                Ok(timeout) => timeout,
                Err(error) => {
                    return Err(benchmark_error(format!("{error}; selected_query_ids={}", config.query_ids.join(","))));
                },
            };
            let query_started = Instant::now();
            eprintln!(
                "[native_iceberg_tpch_benchmark] stage=query_start iteration={}/{} query_index={}/{} query_id={} timeout_category={} timeout_secs={}",
                iteration_index + 1,
                config.iterations,
                query_index + 1,
                config.query_ids.len(),
                query_id,
                timeout.category.as_str(),
                timeout.duration.map(|duration| duration.as_secs().to_string()).unwrap_or_else(|| "none".to_string()),
            );
            match run_single_query_report_with_timeout(&config, query_id, timeout).await {
                Ok(smoke_row) => {
                    let report_elapsed_ms = query_started.elapsed().as_millis();
                    eprintln!(
                        "[native_iceberg_tpch_benchmark] stage=query_finish iteration={}/{} query_id={} elapsed_ms={} native_runtime_outcome={} cpu_runtime_outcome={} parity_status={}",
                        iteration_index + 1,
                        config.iterations,
                        query_id,
                        report_elapsed_ms,
                        smoke_row.native_runtime_outcome.as_str(),
                        smoke_row.cpu_runtime_outcome.as_str(),
                        smoke_row.parity_status.as_str(),
                    );
                    let timeout_category = benchmark_timeout_category_from_smoke_row(&smoke_row);
                    let row = IcebergTpchBenchmarkRow {
                        iteration: iteration_index + 1,
                        tpch_scale_factor: config.smoke_config.tpch_scale_factor.clone(),
                        warmup_iterations: config.warmup_iterations,
                        measured_iterations: config.iterations,
                        report_elapsed_ms: if stream_rows_to_stdout { report_elapsed_ms } else { 0 },
                        timeout_category,
                        smoke_row,
                    };
                    if stream_rows_to_stdout {
                        write_benchmark_tsv_row_to_stdout(&row)?;
                    }
                    rows.push(row);
                },
                Err(BenchmarkQueryExecutionError::TimedOut { category, timeout_secs }) => {
                    let report_elapsed_ms = query_started.elapsed().as_millis();
                    eprintln!(
                        "[native_iceberg_tpch_benchmark] stage=query_timeout iteration={}/{} query_id={} timeout_category={} timeout_secs={} elapsed_ms={}",
                        iteration_index + 1,
                        config.iterations,
                        query_id,
                        category.as_str(),
                        timeout_secs,
                        report_elapsed_ms,
                    );
                    let row = IcebergTpchBenchmarkRow {
                        iteration: iteration_index + 1,
                        tpch_scale_factor: config.smoke_config.tpch_scale_factor.clone(),
                        warmup_iterations: config.warmup_iterations,
                        measured_iterations: config.iterations,
                        report_elapsed_ms: if stream_rows_to_stdout { report_elapsed_ms } else { 0 },
                        timeout_category: Some(category),
                        smoke_row: timeout_smoke_row(&config, query_id, category, timeout_secs),
                    };
                    if stream_rows_to_stdout {
                        write_benchmark_tsv_row_to_stdout(&row)?;
                    }
                    rows.push(row);
                    if !stream_rows_to_stdout {
                        set_iteration_report_elapsed_ms(
                            &mut rows,
                            iteration_row_start,
                            iteration_started.elapsed().as_millis(),
                        );
                    }
                    return Ok(IcebergTpchBenchmarkReport { rows });
                },
                Err(BenchmarkQueryExecutionError::Failed(error)) => return Err(error),
            }
        }
        if !stream_rows_to_stdout {
            set_iteration_report_elapsed_ms(&mut rows, iteration_row_start, iteration_started.elapsed().as_millis());
        }
    }

    Ok(IcebergTpchBenchmarkReport { rows })
}

pub fn set_iteration_report_elapsed_ms(
    rows: &mut [IcebergTpchBenchmarkRow],
    iteration_row_start: usize,
    report_elapsed_ms: u128,
) {
    for row in &mut rows[iteration_row_start..] {
        row.report_elapsed_ms = report_elapsed_ms;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkTimeoutCategory {
    NativeRuntime,
    CpuS3Baseline,
    Iteration,
}

impl BenchmarkTimeoutCategory {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeRuntime => "native_runtime_timeout",
            Self::CpuS3Baseline => "cpu_s3_baseline_timeout",
            Self::Iteration => "iteration_timeout",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryTimeout {
    duration: Option<Duration>,
    category: BenchmarkTimeoutCategory,
}

enum BenchmarkQueryExecutionError {
    TimedOut { category: BenchmarkTimeoutCategory, timeout_secs: u64 },
    Failed(BenchmarkErrorBox),
}

type BenchmarkErrorBox = Box<dyn StdError + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuChildResult {
    pub elapsed_ms: Option<u128>,
    pub output_rows: Option<usize>,
    pub output_batches: Option<usize>,
    pub output_fingerprint: Option<String>,
    pub comparison_mode: ParityComparisonMode,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuChildFailure {
    pub elapsed_ms: u128,
    pub detail: String,
    pub timeout: bool,
}

async fn run_cpu_child(query_id: &str, timeout_secs: Option<u64>) -> Result<CpuChildResult, CpuChildFailure> {
    let started = Instant::now();
    let executable = std::env::current_exe().map_err(|error| CpuChildFailure {
        elapsed_ms: started.elapsed().as_millis(),
        detail: format!("cpu_baseline_child_current_exe_error={}", stable_tsv_field(&error.to_string())),
        timeout: false,
    })?;
    let mut child = Command::new(executable);
    child
        .env(CPU_CHILD_ENV, "1")
        .env(QUERY_IDS_ENV, query_id)
        .env(WARMUPS_ENV, "0")
        .env(ITERATIONS_ENV, "1")
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = child.spawn().map_err(|error| CpuChildFailure {
        elapsed_ms: started.elapsed().as_millis(),
        detail: format!("cpu_baseline_child_spawn_error={}", stable_tsv_field(&error.to_string())),
        timeout: false,
    })?;
    let output_future = child.wait_with_output();
    let output = match timeout_secs {
        Some(timeout_secs) => match tokio::time::timeout(Duration::from_secs(timeout_secs), output_future).await {
            Ok(output) => output,
            Err(_) => {
                return Err(CpuChildFailure {
                    elapsed_ms: started.elapsed().as_millis(),
                    detail: format!("cpu_baseline_child_timeout_secs={timeout_secs}"),
                    timeout: true,
                });
            },
        },
        None => output_future.await,
    }
    .map_err(|error| CpuChildFailure {
        elapsed_ms: started.elapsed().as_millis(),
        detail: format!("cpu_baseline_child_wait_error={}", stable_tsv_field(&error.to_string())),
        timeout: false,
    })?;
    let elapsed_ms = started.elapsed().as_millis();
    if !output.status.success() {
        return Err(CpuChildFailure {
            elapsed_ms,
            detail: format!(
                "cpu_baseline_child_exit_status={};stderr={}",
                child_exit_status_text(output.status),
                stable_tsv_field(&String::from_utf8_lossy(&output.stderr))
            ),
            timeout: false,
        });
    }
    parse_cpu_child_result(&String::from_utf8_lossy(&output.stdout)).map_err(|error| CpuChildFailure {
        elapsed_ms,
        detail: format!("cpu_baseline_child_parse_error={}", stable_tsv_field(&error.to_string())),
        timeout: false,
    })
}

fn merge_cpu_child_result(row: &mut IcebergNativeSmokeReportRow, child: CpuChildResult) {
    row.cpu_runtime_started = true;
    row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
    row.cpu_runtime_elapsed_ms = child.elapsed_ms;
    row.cpu_output_rows = child.output_rows;
    row.cpu_output_batches = child.output_batches;
    row.cpu_output_fingerprint = child.output_fingerprint;
    row.comparison_mode = child.comparison_mode;
    row.parity_detail = child.detail;
    row.parity_status = match (row.cpu_output_fingerprint.as_deref(), row.native_output_fingerprint.as_deref()) {
        (Some(cpu), Some(native)) if cpu == native => ParityStatus::Matched,
        (Some(_), Some(_)) => {
            if row.parity_detail.is_none() {
                row.parity_detail = Some("child CPU fingerprint differs from native fingerprint".to_string());
            }
            ParityStatus::Mismatched
        },
        _ => {
            if row.parity_detail.is_none() {
                row.parity_detail = Some("child CPU or native fingerprint missing".to_string());
            }
            ParityStatus::NotCompared
        },
    };
}

pub fn apply_cpu_child_failure(row: &mut IcebergNativeSmokeReportRow, failure: CpuChildFailure) {
    row.cpu_runtime_started = true;
    row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
    row.cpu_runtime_elapsed_ms = Some(failure.elapsed_ms);
    row.parity_status = ParityStatus::CpuFailed;
    let timeout_category = if failure.timeout { "timeout_category=cpu_s3_baseline_timeout;" } else { "" };
    row.parity_detail = Some(format!("{timeout_category}{}", stable_tsv_field(&failure.detail)));
    row.comparison_mode = ParityComparisonMode::NotApplicable;
}

pub fn parse_cpu_child_result(output: &str) -> BenchmarkResult<CpuChildResult> {
    let mut lines = output.lines().filter(|line| !line.trim().is_empty() && !line.starts_with('#'));
    let header = lines.next().ok_or_else(|| benchmark_error("CPU child TSV missing header"))?;
    let row = lines.next().ok_or_else(|| benchmark_error("CPU child TSV missing row"))?;
    if lines.next().is_some() {
        return Err(benchmark_error("CPU child TSV contained multiple rows"));
    }
    let headers = header.split('\t').collect::<Vec<_>>();
    let fields = row.split('\t').collect::<Vec<_>>();
    if headers.len() != fields.len() {
        return Err(benchmark_error(format!(
            "CPU child TSV field count mismatch: header={}, row={}",
            headers.len(),
            fields.len()
        )));
    }
    let field = |name: &str| -> BenchmarkResult<&str> {
        headers
            .iter()
            .position(|header| *header == name)
            .and_then(|index| fields.get(index).copied())
            .ok_or_else(|| benchmark_error(format!("CPU child TSV missing field {name}")))
    };
    if field("cpu_runtime_outcome")? != "succeeded" {
        return Err(benchmark_error(format!(
            "CPU child runtime outcome={}; detail={}",
            field("cpu_runtime_outcome")?,
            field("parity_detail").unwrap_or("none")
        )));
    }
    Ok(CpuChildResult {
        elapsed_ms: parse_optional_u128_field(field("cpu_runtime_elapsed_ms")?)?,
        output_rows: parse_optional_usize_field(field("cpu_output_rows")?)?,
        output_batches: parse_optional_usize_field(field("cpu_output_batches")?)?,
        output_fingerprint: parse_optional_string_field(field("cpu_output_fingerprint")?),
        comparison_mode: parse_comparison_mode(field("comparison_mode")?)?,
        detail: parse_optional_string_field(field("parity_detail")?),
    })
}

fn parse_optional_u128_field(value: &str) -> BenchmarkResult<Option<u128>> {
    if value == "none" {
        Ok(None)
    } else {
        value.parse::<u128>().map(Some).map_err(|error| benchmark_error(format!("invalid u128 field {value}: {error}")))
    }
}

fn parse_optional_usize_field(value: &str) -> BenchmarkResult<Option<usize>> {
    if value == "none" {
        Ok(None)
    } else {
        value
            .parse::<usize>()
            .map(Some)
            .map_err(|error| benchmark_error(format!("invalid usize field {value}: {error}")))
    }
}

fn parse_optional_string_field(value: &str) -> Option<String> {
    (value != "none").then(|| value.to_string())
}

fn parse_comparison_mode(value: &str) -> BenchmarkResult<ParityComparisonMode> {
    match value {
        "ordered_exact" => Ok(ParityComparisonMode::OrderedExact),
        "unordered_exact" => Ok(ParityComparisonMode::UnorderedExact),
        "ordered_float_tolerant" => Ok(ParityComparisonMode::OrderedFloatTolerant),
        "unordered_float_tolerant" => Ok(ParityComparisonMode::UnorderedFloatTolerant),
        "not_applicable" => Ok(ParityComparisonMode::NotApplicable),
        other => Err(benchmark_error(format!("unknown comparison mode {other}"))),
    }
}

fn child_exit_status_text(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("signal:{signal}");
        }
    }
    status.code().map_or_else(|| "unknown".to_string(), |code| format!("code:{code}"))
}

async fn run_report_with_timeout(
    smoke_config: IcebergNativeSmokeConfig,
    timeout_secs: Option<u64>,
    category: BenchmarkTimeoutCategory,
) -> BenchmarkResult<IcebergNativeSmokeReport> {
    let future = run_report(smoke_config);
    match timeout_secs {
        Some(timeout_secs) => tokio::time::timeout(Duration::from_secs(timeout_secs), future)
            .await
            .map_err(|_| benchmark_error(format!("{} after {timeout_secs}s", category.as_str())))?,
        None => future.await,
    }
}

async fn run_single_query_report_with_timeout(
    config: &IcebergTpchBenchmarkConfig,
    query_id: &str,
    timeout: QueryTimeout,
) -> Result<IcebergNativeSmokeReportRow, BenchmarkQueryExecutionError> {
    if config.uses_cpu_child_baseline() {
        return run_single_query_report_with_child_cpu(config, query_id, timeout).await;
    }
    let mut smoke_config = config.smoke_config.clone();
    smoke_config.tpch_query_ids = vec![query_id.to_string()];
    let future = run_report(smoke_config);
    let report = match timeout.duration {
        Some(duration) => tokio::time::timeout(duration, future)
            .await
            .map_err(|_| BenchmarkQueryExecutionError::TimedOut {
                category: timeout.category,
                timeout_secs: timeout_secs_for_report(duration),
            })?
            .map_err(BenchmarkQueryExecutionError::Failed)?,
        None => future.await.map_err(BenchmarkQueryExecutionError::Failed)?,
    };
    let mut rows = report.rows.into_iter();
    let row = rows.next().ok_or_else(|| {
        BenchmarkQueryExecutionError::Failed(benchmark_error(format!(
            "native Iceberg TPC-H benchmark produced no smoke row for {query_id}"
        )))
    })?;
    if rows.next().is_some() {
        return Err(BenchmarkQueryExecutionError::Failed(benchmark_error(format!(
            "native Iceberg TPC-H benchmark produced multiple smoke rows for single query {query_id}"
        ))));
    }
    Ok(row)
}

async fn run_single_query_report_with_child_cpu(
    config: &IcebergTpchBenchmarkConfig,
    query_id: &str,
    timeout: QueryTimeout,
) -> Result<IcebergNativeSmokeReportRow, BenchmarkQueryExecutionError> {
    let native_future = run_single_tpch_query_report_with_execution_mode(
        config.smoke_config.clone(),
        query_id.to_string(),
        SmokeQueryExecutionMode::NativeOnly,
    );
    let mut native_row = match timeout.duration {
        Some(duration) => tokio::time::timeout(duration, native_future)
            .await
            .map_err(|_| BenchmarkQueryExecutionError::TimedOut {
                category: timeout.category,
                timeout_secs: timeout_secs_for_report(duration),
            })?
            .map_err(BenchmarkQueryExecutionError::Failed)?,
        None => native_future.await.map_err(BenchmarkQueryExecutionError::Failed)?,
    };
    if native_row.native_runtime_outcome != NativeDataFusionRuntimeOutcome::Succeeded {
        return Ok(native_row);
    }
    let timeout_secs =
        config.smoke_config.cpu_baseline_timeout_secs.or(config.query_timeout_secs).or(config.iteration_timeout_secs);
    match run_cpu_child(query_id, timeout_secs).await {
        Ok(child) => {
            merge_cpu_child_result(&mut native_row, child);
            Ok(native_row)
        },
        Err(error) => {
            apply_cpu_child_failure(&mut native_row, error);
            Ok(native_row)
        },
    }
}

fn timeout_secs_for_report(duration: Duration) -> u64 {
    duration.as_secs().max(1)
}

fn query_timeout_for_iteration(
    config: &IcebergTpchBenchmarkConfig,
    iteration_started: Instant,
) -> BenchmarkResult<QueryTimeout> {
    let query_timeout = config.query_timeout_secs.map(Duration::from_secs);
    let iteration_remaining = if let Some(timeout_secs) = config.iteration_timeout_secs {
        let timeout = Duration::from_secs(timeout_secs);
        let elapsed = iteration_started.elapsed();
        Some(timeout.checked_sub(elapsed).ok_or_else(|| {
            benchmark_error(format!(
                "{} before starting next query after {}s",
                BenchmarkTimeoutCategory::Iteration.as_str(),
                timeout_secs
            ))
        })?)
    } else {
        None
    };

    let timeout = match (query_timeout, iteration_remaining) {
        (Some(query_timeout), Some(iteration_remaining)) if iteration_remaining < query_timeout => {
            QueryTimeout { duration: Some(iteration_remaining), category: BenchmarkTimeoutCategory::Iteration }
        },
        (Some(query_timeout), _) => {
            QueryTimeout { duration: Some(query_timeout), category: BenchmarkTimeoutCategory::NativeRuntime }
        },
        (None, Some(iteration_remaining)) => {
            QueryTimeout { duration: Some(iteration_remaining), category: BenchmarkTimeoutCategory::Iteration }
        },
        (None, None) => QueryTimeout { duration: None, category: BenchmarkTimeoutCategory::NativeRuntime },
    };
    Ok(timeout)
}

fn timeout_smoke_row(
    config: &IcebergTpchBenchmarkConfig,
    query_id: &str,
    category: BenchmarkTimeoutCategory,
    timeout_secs: u64,
) -> IcebergNativeSmokeReportRow {
    let query_sql = TPCH_QUERIES
        .iter()
        .find(|query| query.id == query_id)
        .map(|query| query.sql.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let diagnostics = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_catalog_type(config.smoke_config.fixture.catalog_type())
        .diagnostics();
    let mut row = IcebergNativeSmokeReportRow::new(
        query_id.to_string(),
        config.smoke_config.fixture.mode_code().to_string(),
        config.smoke_config.fixture.catalog_type().to_string(),
        config.smoke_config.fixture.tpch_all_tables_ident_display(&config.smoke_config.tpch_scale_factor),
        query_sql,
        diagnostics,
        None,
        Some(category.as_str().to_string()),
        None,
        ObjectStoreStagingSummary::new(config.smoke_config.object_store_staging, ObjectStoreStagingStatus::NotNeeded),
        None,
        config.smoke_config.fixture.credential_source().to_string(),
    );
    row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
    row.parity_status = ParityStatus::NativeFailed;
    row.comparison_mode = ParityComparisonMode::NotApplicable;
    row.parity_detail = Some(format!("timeout_category={};timeout_secs={timeout_secs}", category.as_str()));
    row
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergTpchBenchmarkConfig {
    pub smoke_config: IcebergNativeSmokeConfig,
    pub query_ids: Vec<String>,
    pub warmup_iterations: usize,
    pub iterations: usize,
    pub query_timeout_secs: Option<u64>,
    pub iteration_timeout_secs: Option<u64>,
}

impl IcebergTpchBenchmarkConfig {
    pub fn from_env() -> BenchmarkResult<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> BenchmarkResult<Self> {
        let mut smoke_config = IcebergNativeSmokeConfig::from_env_get(|name| get(name))?;
        if !matches!(
            smoke_config.fixture,
            IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }
        ) {
            return Err(benchmark_error(
                "native Iceberg TPC-H benchmark requires ALGEON_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
            ));
        }

        let query_ids = match get(QUERY_IDS_ENV) {
            Some(value) => parse_query_ids(&value)?,
            None => DEFAULT_QUERY_IDS.iter().map(|query_id| (*query_id).to_string()).collect(),
        };
        let warmup_iterations = parse_usize_env(get(WARMUPS_ENV), WARMUPS_ENV, 0)?;
        let iterations = parse_usize_env(get(ITERATIONS_ENV), ITERATIONS_ENV, 1)?;
        if iterations == 0 {
            return Err(benchmark_error(format!("{ITERATIONS_ENV} must be greater than zero")));
        }
        let query_timeout_secs = parse_optional_u64_env(get(QUERY_TIMEOUT_SECS_ENV), QUERY_TIMEOUT_SECS_ENV)?;
        let iteration_timeout_secs =
            parse_optional_u64_env(get(ITERATION_TIMEOUT_SECS_ENV), ITERATION_TIMEOUT_SECS_ENV)?;
        let cpu_baseline_timeout_secs =
            parse_optional_u64_env(get(CPU_BASELINE_TIMEOUT_SECS_ENV), CPU_BASELINE_TIMEOUT_SECS_ENV)?;
        if smoke_config.cpu_baseline_timeout_secs.is_none() {
            smoke_config.cpu_baseline_timeout_secs = cpu_baseline_timeout_secs.or_else(|| {
                (query_timeout_secs.is_some() || iteration_timeout_secs.is_some())
                    .then_some(DEFAULT_TIMEOUT_ENABLED_CPU_BASELINE_TIMEOUT_SECS)
            });
        }
        if let Some(value) = get(OBJECT_STORE_STAGING_ENV) {
            smoke_config.object_store_staging = parse_object_store_staging_mode(&value)?;
        }
        smoke_config.tpch_query_ids = query_ids.clone();

        Ok(Self { smoke_config, query_ids, warmup_iterations, iterations, query_timeout_secs, iteration_timeout_secs })
    }

    fn validate(&self) -> BenchmarkResult<()> {
        if self.query_ids.is_empty() {
            return Err(benchmark_error(format!("{QUERY_IDS_ENV} must select at least one query")));
        }
        if self.iterations == 0 {
            return Err(benchmark_error(format!("{ITERATIONS_ENV} must be greater than zero")));
        }
        if self.query_timeout_secs == Some(0) {
            return Err(benchmark_error(format!("{QUERY_TIMEOUT_SECS_ENV} must be greater than zero when set")));
        }
        if self.iteration_timeout_secs == Some(0) {
            return Err(benchmark_error(format!("{ITERATION_TIMEOUT_SECS_ENV} must be greater than zero when set")));
        }
        Ok(())
    }

    fn timeouts_enabled(&self) -> bool {
        self.query_timeout_secs.is_some() || self.iteration_timeout_secs.is_some()
    }

    fn stream_rows_to_stdout(&self) -> bool {
        self.timeouts_enabled()
    }

    fn uses_cpu_child_baseline(&self) -> bool {
        self.timeouts_enabled()
    }
}

fn benchmark_timeout_category_from_smoke_row(row: &IcebergNativeSmokeReportRow) -> Option<BenchmarkTimeoutCategory> {
    row.parity_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("timeout_category=cpu_s3_baseline_timeout"))
        .then_some(BenchmarkTimeoutCategory::CpuS3Baseline)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergTpchBenchmarkReport {
    pub rows: Vec<IcebergTpchBenchmarkRow>,
}

impl IcebergTpchBenchmarkReport {
    pub fn to_tsv(&self) -> String {
        let rows = self.rows.iter().map(IcebergTpchBenchmarkRow::to_tsv_row).collect::<Vec<_>>().join("\n");
        debug_assert_eq!(REPORT_COLUMNS.join("\t"), REPORT_HEADER);
        format!("{REPORT_HEADER}\n{rows}\n")
    }

    fn timeout_failure_message(&self) -> Option<String> {
        let failures = self
            .rows
            .iter()
            .filter_map(|row| {
                row.timeout_category.map(|category| format!("{}:{}", row.smoke_row.query_id, category.as_str()))
            })
            .collect::<Vec<_>>();
        if failures.is_empty() { None } else { Some(format!("benchmark timeout failures={}", failures.join(","))) }
    }
}

pub fn write_benchmark_tsv_header(writer: &mut impl Write) -> io::Result<()> {
    debug_assert_eq!(REPORT_COLUMNS.join("\t"), REPORT_HEADER);
    writeln!(writer, "{REPORT_HEADER}")?;
    writer.flush()
}

pub fn write_benchmark_tsv_row(writer: &mut impl Write, row: &IcebergTpchBenchmarkRow) -> io::Result<()> {
    writeln!(writer, "{}", row.to_tsv_row())?;
    writer.flush()
}

fn write_benchmark_tsv_header_to_stdout() -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    write_benchmark_tsv_header(&mut stdout)
}

fn write_benchmark_tsv_row_to_stdout(row: &IcebergTpchBenchmarkRow) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    write_benchmark_tsv_row(&mut stdout, row)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergTpchBenchmarkRow {
    pub iteration: usize,
    pub tpch_scale_factor: String,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub report_elapsed_ms: u128,
    pub timeout_category: Option<BenchmarkTimeoutCategory>,
    pub smoke_row: IcebergNativeSmokeReportRow,
}

impl IcebergTpchBenchmarkRow {
    fn to_tsv_row(&self) -> String {
        let row = &self.smoke_row;
        let fields = [
            row.query_id.clone(),
            self.iteration.to_string(),
            row.fixture_mode.clone(),
            row.catalog_type.clone(),
            row.table_ident.clone(),
            self.tpch_scale_factor.clone(),
            row.diagnostics.table_format.clone(),
            row.diagnostics.snapshot_id.clone().unwrap_or_else(|| "none".to_string()),
            row.diagnostics.planned_data_file_count.to_string(),
            optional_u64(row.diagnostics.planned_bytes),
            row.diagnostics.admitted_native_source_file_count.to_string(),
            empty_as_none(&row.diagnostics.native_source_read_capability),
            empty_as_none(&row.diagnostics.source_read_strategy),
            empty_as_none(&row.diagnostics.credential_mode),
            empty_as_none(&row.diagnostics.lakehouse_cache_status),
            empty_as_none(&row.diagnostics.lakehouse_cache_read_mode),
            empty_as_none(&row.diagnostics.lakehouse_cache_fill_mode),
            row.diagnostics.direct_device_read_disposition.code().to_string(),
            object_store_staging_mode_code(row.staging_summary.mode).to_string(),
            object_store_staging_status_code(row.staging_summary.status).to_string(),
            row.staging_summary.staged_data_file_count.to_string(),
            row.staging_summary.staged_bytes.to_string(),
            row.native_planning_outcome.map_or("none", CandidateOutcomeKind::as_str).to_string(),
            row.native_runtime_outcome.as_str().to_string(),
            optional_u128(row.native_runtime_elapsed_ms),
            optional_usize(row.native_output_rows),
            optional_usize(row.native_output_batches),
            row.native_source_read_runtime_rows.clone(),
            row.lakehouse_cache_fill_event_snapshot_rows.clone(),
            row.cpu_runtime_outcome.as_str().to_string(),
            optional_u128(row.cpu_runtime_elapsed_ms),
            optional_usize(row.cpu_output_rows),
            optional_usize(row.cpu_output_batches),
            row.comparison_mode.as_str().to_string(),
            row.parity_status.as_str().to_string(),
            row.parity_detail.clone().unwrap_or_else(|| "none".to_string()),
            self.timeout_category.map(|category| category.as_str()).unwrap_or("none").to_string(),
            row.native_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            row.cpu_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.report_elapsed_ms.to_string(),
            self.warmup_iterations.to_string(),
            self.measured_iterations.to_string(),
            row.credential_source.clone(),
        ];
        debug_assert_eq!(fields.len(), REPORT_COLUMNS.len());
        fields.into_iter().map(|field| stable_tsv_field(&field)).collect::<Vec<_>>().join("\t")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullSf1DirectS3GateSummary {
    pub query_count: usize,
    pub measured_iterations: usize,
    pub native_runtime_elapsed_ms_total: u128,
    pub cpu_runtime_elapsed_ms_total: u128,
    pub report_elapsed_ms: u128,
}

impl FullSf1DirectS3GateSummary {
    fn progress_line(&self) -> String {
        format!(
            "[native_iceberg_tpch_benchmark] stage=full_sf1_staged_s3_gate outcome=passed query_count={} measured_iterations={} native_runtime_elapsed_ms_total={} cpu_runtime_elapsed_ms_total={} report_elapsed_ms={}",
            self.query_count,
            self.measured_iterations,
            self.native_runtime_elapsed_ms_total,
            self.cpu_runtime_elapsed_ms_total,
            self.report_elapsed_ms,
        )
    }
}

pub fn validate_full_sf1_staged_s3_gate_for_config(
    report: &IcebergTpchBenchmarkReport,
    config: &IcebergTpchBenchmarkConfig,
) -> BenchmarkResult<Option<FullSf1DirectS3GateSummary>> {
    if !full_sf1_staged_s3_gate_applies(config) {
        return Ok(None);
    }

    let expected_query_ids = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    let observed_query_ids = report.rows.iter().map(|row| row.smoke_row.query_id.as_str()).collect::<BTreeSet<_>>();
    let missing = expected_query_ids
        .iter()
        .filter(|query_id| !observed_query_ids.contains(**query_id))
        .copied()
        .collect::<Vec<_>>();

    let expected_row_count = expected_query_ids.len() * config.iterations;

    let mut timeout_failures = Vec::new();
    let mut failures = Vec::new();
    if !missing.is_empty() {
        failures.push(format!("missing_query_rows={}", missing.join(",")));
    }
    if report.rows.len() != expected_row_count {
        failures.push(format!("expected_row_count={expected_row_count},observed_row_count={}", report.rows.len()));
    }
    let mut native_runtime_elapsed_ms_total = 0_u128;
    let mut cpu_runtime_elapsed_ms_total = 0_u128;
    let mut report_elapsed_ms = 0_u128;

    for row in &report.rows {
        let query_id = row.smoke_row.query_id.as_str();
        if let Some(category) = row.timeout_category {
            timeout_failures.push(format!("{query_id}:{}", category.as_str()));
        }
        let diagnostics = &row.smoke_row.diagnostics;
        if diagnostics.native_source_read_capability != "local_file_system" {
            failures.push(format!(
                "{query_id}:native_source_read_capability={}",
                empty_as_none(&diagnostics.native_source_read_capability)
            ));
        }
        if diagnostics.direct_device_read_disposition.code() != "eligible" {
            failures.push(format!(
                "{query_id}:direct_device_read_disposition={}",
                diagnostics.direct_device_read_disposition.code()
            ));
        }
        if row.smoke_row.staging_summary.status != ObjectStoreStagingStatus::StagedLocal {
            failures.push(format!(
                "{query_id}:object_store_staging_status={}",
                object_store_staging_status_code(row.smoke_row.staging_summary.status)
            ));
        }
        if row.smoke_row.staging_summary.staged_data_file_count == 0 {
            failures.push(format!(
                "{query_id}:staged_data_file_count={}",
                row.smoke_row.staging_summary.staged_data_file_count
            ));
        }
        if row.smoke_row.native_planning_outcome != Some(CandidateOutcomeKind::Selected) {
            failures.push(format!(
                "{query_id}:native_planning_outcome={}",
                row.smoke_row.native_planning_outcome.map_or("none", CandidateOutcomeKind::as_str)
            ));
        }
        if row.smoke_row.native_runtime_outcome.as_str() != "succeeded" {
            failures
                .push(format!("{query_id}:native_runtime_outcome={}", row.smoke_row.native_runtime_outcome.as_str()));
        }
        if row.smoke_row.cpu_runtime_outcome.as_str() != "succeeded" {
            failures.push(format!("{query_id}:cpu_runtime_outcome={}", row.smoke_row.cpu_runtime_outcome.as_str()));
        }
        if row.smoke_row.parity_status.as_str() != "matched" {
            failures.push(format!("{query_id}:parity_status={}", row.smoke_row.parity_status.as_str()));
        }
        native_runtime_elapsed_ms_total += row.smoke_row.native_runtime_elapsed_ms.unwrap_or(0);
        cpu_runtime_elapsed_ms_total += row.smoke_row.cpu_runtime_elapsed_ms.unwrap_or(0);
        report_elapsed_ms = report_elapsed_ms.max(row.report_elapsed_ms);
    }

    if !timeout_failures.is_empty() {
        let semantic = if failures.is_empty() { "none".to_string() } else { failures.join(",") };
        return Err(benchmark_error(format!(
            "full SF1 staged-S3 gate timeout failures={}; semantic_failures={semantic}",
            timeout_failures.join(",")
        )));
    }

    if !failures.is_empty() {
        return Err(benchmark_error(format!(
            "full SF1 staged-S3 gate requires required staging, native execution, and CPU parity; failures={}",
            failures.join(",")
        )));
    }

    Ok(Some(FullSf1DirectS3GateSummary {
        query_count: expected_query_ids.len(),
        measured_iterations: config.iterations,
        native_runtime_elapsed_ms_total,
        cpu_runtime_elapsed_ms_total,
        report_elapsed_ms,
    }))
}

fn full_sf1_staged_s3_gate_applies(config: &IcebergTpchBenchmarkConfig) -> bool {
    matches!(config.smoke_config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. })
        && config.smoke_config.tpch_scale_factor == "sf1"
        && config.query_ids.iter().map(String::as_str).collect::<BTreeSet<_>>()
            == TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>()
}

fn parse_query_ids(value: &str) -> BenchmarkResult<Vec<String>> {
    let query_ids = value.split(',').map(str::trim).filter(|query_id| !query_id.is_empty()).collect::<Vec<_>>();
    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in &query_ids {
        if !known.contains(query_id) {
            return Err(benchmark_error(format!("unknown TPC-H query id {query_id:?} in {QUERY_IDS_ENV}")));
        }
    }
    Ok(query_ids.into_iter().map(str::to_string).collect())
}

fn parse_usize_env(value: Option<String>, name: &'static str, default: usize) -> BenchmarkResult<usize> {
    match value {
        Some(value) if !value.trim().is_empty() => value.trim().parse::<usize>().map_err(|error| {
            benchmark_error(format!("invalid {name}={}: expected unsigned integer: {error}", value.trim()))
        }),
        _ => Ok(default),
    }
}

fn parse_optional_u64_env(value: Option<String>, name: &'static str) -> BenchmarkResult<Option<u64>> {
    match value {
        Some(value) if !value.trim().is_empty() => {
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                benchmark_error(format!("invalid {name}={}: expected unsigned integer seconds: {error}", value.trim()))
            })?;
            if parsed == 0 {
                return Err(benchmark_error(format!("{name} must be greater than zero when set")));
            }
            Ok(Some(parsed))
        },
        _ => Ok(None),
    }
}

fn parse_object_store_staging_mode(value: &str) -> BenchmarkResult<ObjectStoreStagingMode> {
    match value.trim() {
        "" | "auto" => Ok(ObjectStoreStagingMode::Auto),
        "disabled" | "false" | "0" | "off" => Ok(ObjectStoreStagingMode::Disabled),
        "required" | "true" | "1" | "on" => Ok(ObjectStoreStagingMode::Required),
        other => Err(benchmark_error(format!(
            "unsupported {OBJECT_STORE_STAGING_ENV}={other}; expected auto, disabled, or required"
        ))),
    }
}

fn object_store_staging_mode_code(mode: ObjectStoreStagingMode) -> &'static str {
    match mode {
        ObjectStoreStagingMode::Auto => "auto",
        ObjectStoreStagingMode::Disabled => "disabled",
        ObjectStoreStagingMode::Required => "required",
    }
}

fn object_store_staging_status_code(status: ObjectStoreStagingStatus) -> &'static str {
    status.code()
}

fn optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn optional_u128(value: Option<u128>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn empty_as_none(value: &str) -> String {
    if value.is_empty() { "none".to_string() } else { value.to_string() }
}

fn benchmark_error(message: impl Into<String>) -> Box<dyn StdError + Send + Sync> {
    Box::new(BenchmarkError(message.into()))
}

#[derive(Debug)]
struct BenchmarkError(String);

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl StdError for BenchmarkError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_header_matches_columns() {
        assert_eq!(REPORT_COLUMNS.first(), Some(&"query_id"));
        assert_eq!(REPORT_COLUMNS.last(), Some(&"credential_source"));
        assert_eq!(REPORT_COLUMNS.join("\t"), REPORT_HEADER);
    }
}
