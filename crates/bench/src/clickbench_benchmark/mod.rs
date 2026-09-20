use std::{
    fmt::Write as _,
    fs::{self, File},
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
    time::Instant,
};

use algeon_datafusion::{planner::plan_with_native_diagnostics, session::AlgeonSessionStateBuilderExt};
use arrow::{array::RecordBatch, error::ArrowError, util::display::array_value_to_string};
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use workloads::clickbench::{
    CLICKBENCH_QUERIES, ClickbenchDatasetMetadata, ClickbenchLayout, ClickbenchQuery, WorkloadError, WorkloadResult,
    discover_clickbench_layout, prepare_clickbench_report_dataset, register_clickbench_table,
};

use crate::report_workload::{
    RelationalExecutionSelection,
    clickbench::{
        ClickbenchNativeProfileSettings, clickbench_native_optimizer_config_with_profile_settings,
        clickbench_native_profile_settings_from_env, clickbench_report_session_config,
        clickbench_tuned_cpu_session_config,
    },
};
const TSV_COLUMNS: &[&str] = &[
    "query_id",
    "parity_status",
    "speedup_vs_cpu",
    "native_rewritten",
    "cpu_outcome",
    "cpu_iteration_failures",
    "cpu_wall_ms_mean",
    "cpu_wall_ms_min",
    "cpu_wall_ms_median",
    "cpu_wall_ms_max",
    "cpu_output_batches",
    "cpu_output_rows",
    "cpu_output_bytes",
    "cpu_output_fingerprint",
    "cpu_failure",
    "native_outcome",
    "native_iteration_failures",
    "native_wall_ms_mean",
    "native_wall_ms_min",
    "native_wall_ms_median",
    "native_wall_ms_max",
    "native_output_batches",
    "native_output_rows",
    "native_output_bytes",
    "native_output_fingerprint",
    "native_failure",
];

/// ClickBench queries where byte-for-byte fingerprint parity is waived.
///
/// These compare row counts only and emit
/// `parity_status=row_count_matched_nondeterministic`; measured-iteration
/// consistency is also disabled for nondeterministic row selection.
///
/// Reason categories:
/// - `float_accumulation_variance`: deterministic result set, but different
///   valid Float64 accumulation orders produce last-bit AVG differences.
///   q04 is `AVG("UserID")` over large-magnitude int64 values; q10 is
///   `AVG("ResolutionWidth")` in a grouped query whose rank-10/rank-11
///   `COUNT(*)` boundary is not tied.
/// - `result_set_nondeterministic`: selected rows are engine-defined because
///   the SQL has `LIMIT`/`OFFSET` without a deterministic total ordering and
///   the ClickBench fixture has ties at the boundary.
///
/// q23 and q10 are intentionally excluded because their prior DISTINCT-count
/// bugs are fixed and they must report `matched`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParityWaiverReason {
    FloatAccumulationVariance,
    ResultSetNondeterministic,
}

impl ParityWaiverReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::FloatAccumulationVariance => "float_accumulation_variance",
            Self::ResultSetNondeterministic => "result_set_nondeterministic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParityWaiver {
    query_id: &'static str,
    reason: ParityWaiverReason,
}

const PARITY_WAIVERS: &[ParityWaiver] = &[
    ParityWaiver { query_id: "q04", reason: ParityWaiverReason::FloatAccumulationVariance },
    ParityWaiver { query_id: "q18", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q25", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q26", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q31", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q32", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q33", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q39", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q40", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q41", reason: ParityWaiverReason::ResultSetNondeterministic },
    ParityWaiver { query_id: "q42", reason: ParityWaiverReason::ResultSetNondeterministic },
];

fn parity_waiver_reason(query_id: &str) -> Option<ParityWaiverReason> {
    PARITY_WAIVERS.iter().find(|waiver| waiver.query_id == query_id).map(|waiver| waiver.reason)
}

fn parity_waived_query_ids_label() -> String {
    PARITY_WAIVERS.iter().map(|waiver| waiver.query_id).collect::<Vec<_>>().join(",")
}

fn parity_waiver_reasons_label() -> String {
    PARITY_WAIVERS
        .iter()
        .map(|waiver| format!("{}={}", waiver.query_id, waiver.reason.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

const QUERY_IDS_ENV: &str = "ALGEON_CLICKBENCH_BENCH_QUERY_IDS";
const WARMUPS_ENV: &str = "ALGEON_CLICKBENCH_BENCH_WARMUPS";
const ITERATIONS_ENV: &str = "ALGEON_CLICKBENCH_BENCH_ITERATIONS";
const SKIP_CPU_ENV: &str = "ALGEON_CLICKBENCH_BENCH_SKIP_CPU";
const SKIP_NATIVE_ENV: &str = "ALGEON_CLICKBENCH_BENCH_SKIP_NATIVE";
const CPU_PROFILE_ENV: &str = "ALGEON_CLICKBENCH_BENCH_CPU_PROFILE";
const PROGRESS_ENV: &str = "ALGEON_CLICKBENCH_BENCH_PROGRESS";
const DUMP_ROWS_DIR_ENV: &str = "ALGEON_CLICKBENCH_BENCH_DUMP_ROWS_DIR";
pub(crate) const KVIKIO_COMPAT_MODE_ENV: &str = "KVIKIO_COMPAT_MODE";

pub async fn run_main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr().lock(), "native_clickbench_benchmark: {error}");
            ExitCode::FAILURE
        },
    }
}

async fn run() -> WorkloadResult<()> {
    let config = BenchmarkConfig::from_env()?;
    let (root, metadata) = prepare_clickbench_report_dataset()?.ok_or_else(|| {
        WorkloadError::InvalidDataset(
            "no ClickBench fixture found; set ALGEON_PARQUET_DIR or stage \
             fixture/clickbench/data/{hits.parquet, partitioned/} before running this benchmark"
                .into(),
        )
    })?;
    let layout = discover_clickbench_layout(&root)?;
    let report = run_benchmark(&config, &metadata, &layout).await?;
    print!("{}", report.to_tsv());
    Ok(())
}

#[derive(Clone)]
struct BenchmarkConfig {
    query_ids: Vec<String>,
    warmup_iterations: usize,
    iterations: usize,
    skip_cpu: bool,
    skip_native: bool,
    cpu_profile: CpuProfile,
    native_execution: RelationalExecutionSelection,
    native_profile: ClickbenchNativeProfileSettings,
    progress: bool,
    dump_rows_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
enum CpuProfile {
    Tuned,
    Report,
}

impl CpuProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tuned => "tuned",
            Self::Report => "report",
        }
    }

    fn session_config(self) -> SessionConfig {
        match self {
            Self::Tuned => clickbench_tuned_cpu_session_config(),
            Self::Report => clickbench_report_session_config(),
        }
    }
}

impl BenchmarkConfig {
    fn from_env() -> WorkloadResult<Self> {
        let warmup_iterations = env_usize(WARMUPS_ENV, 0)?;
        let iterations = env_usize(ITERATIONS_ENV, 1)?;
        if iterations == 0 {
            return Err(invalid_env(format!("{ITERATIONS_ENV} must be >= 1")));
        }
        let skip_cpu = env_truthy(SKIP_CPU_ENV);
        let skip_native = env_truthy(SKIP_NATIVE_ENV);
        if skip_cpu && skip_native {
            return Err(invalid_env(format!(
                "cannot set both {SKIP_CPU_ENV} and {SKIP_NATIVE_ENV}; one engine must run"
            )));
        }
        let cpu_profile = match std::env::var(CPU_PROFILE_ENV).unwrap_or_else(|_| "tuned".into()).as_str() {
            "tuned" => CpuProfile::Tuned,
            "report" => CpuProfile::Report,
            other => {
                return Err(invalid_env(format!(
                    "unsupported {CPU_PROFILE_ENV}={other}; expected 'tuned' or 'report'"
                )));
            },
        };
        let native_profile = clickbench_native_profile_settings_from_env();
        let query_ids = query_ids_from_env(QUERY_IDS_ENV)?;
        Ok(Self {
            query_ids,
            warmup_iterations,
            iterations,
            skip_cpu,
            skip_native,
            cpu_profile,
            native_execution: RelationalExecutionSelection::strict_native(),
            native_profile,
            progress: env_truthy(PROGRESS_ENV),
            dump_rows_dir: env_path(DUMP_ROWS_DIR_ENV),
        })
    }

    fn selected_queries(&self) -> Vec<&'static ClickbenchQuery> {
        if self.query_ids.is_empty() {
            CLICKBENCH_QUERIES.iter().collect()
        } else {
            CLICKBENCH_QUERIES.iter().filter(|q| self.query_ids.iter().any(|id| id == q.id)).collect()
        }
    }
}

async fn run_benchmark(
    config: &BenchmarkConfig,
    metadata: &ClickbenchDatasetMetadata,
    layout: &ClickbenchLayout,
) -> WorkloadResult<BenchmarkReport> {
    if let Some(dir) = &config.dump_rows_dir {
        fs::create_dir_all(dir).map_err(|error| {
            WorkloadError::InvalidDataset(format!("failed to create {DUMP_ROWS_DIR_ENV} {}: {error}", dir.display()))
        })?;
    }
    let queries = config.selected_queries();
    let cpu_ctx = if config.skip_cpu { None } else { Some(build_cpu_ctx(config, layout).await?) };
    let native_ctx = if config.skip_native { None } else { Some(build_native_ctx(config, layout).await?) };
    let total = queries.len();
    let mut stderr = io::stderr().lock();
    let mut rows = Vec::with_capacity(total);
    for (idx, query) in queries.iter().enumerate() {
        let qpos = idx + 1;
        let cpu = match &cpu_ctx {
            Some(ctx) => Some(measure_query(ctx, query, "cpu", qpos, total, config, false, &mut stderr).await),
            None => None,
        };
        let native = match &native_ctx {
            Some(ctx) => Some(measure_query(ctx, query, "native", qpos, total, config, true, &mut stderr).await),
            None => None,
        };
        rows.push(BenchmarkRow::build(query.id, cpu, native));
    }
    Ok(BenchmarkReport { config: config.clone(), metadata: metadata.clone(), rows })
}

async fn build_cpu_ctx(config: &BenchmarkConfig, layout: &ClickbenchLayout) -> WorkloadResult<SessionContext> {
    let ctx = SessionContext::new_with_config(config.cpu_profile.session_config());
    register_clickbench_table(&ctx, layout).await?;
    Ok(ctx)
}

async fn build_native_ctx(config: &BenchmarkConfig, layout: &ClickbenchLayout) -> WorkloadResult<SessionContext> {
    let optimizer_config = config
        .native_execution
        .configure_native_optimizer(clickbench_native_optimizer_config_with_profile_settings(&config.native_profile))
        .expect("native ClickBench session installs the native rule");
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(clickbench_report_session_config())
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_clickbench_table(&ctx, layout).await?;
    Ok(ctx)
}

#[expect(clippy::too_many_arguments, reason = "benchmark harness parameters are independent measurement axes")]
async fn measure_query(
    ctx: &SessionContext,
    query: &ClickbenchQuery,
    engine: &str,
    qpos: usize,
    total: usize,
    config: &BenchmarkConfig,
    native_rule_installed: bool,
    stderr: &mut io::StderrLock<'_>,
) -> EngineMeasurement {
    for w in 0..config.warmup_iterations {
        let res = run_once(ctx, query, native_rule_installed, None).await;
        if config.progress {
            let _ = writeln!(
                stderr,
                "[bench {engine}] warmup {}/{} q={} ({qpos}/{total}) elapsed_ms={} outcome={} rewritten={:?}",
                w + 1,
                config.warmup_iterations,
                query.id,
                res.elapsed_ms,
                res.outcome.as_str(),
                res.native_rewritten
            );
        }
    }
    let mut samples_ms: Vec<u128> = Vec::with_capacity(config.iterations);
    let mut runs: Vec<RunResult> = Vec::with_capacity(config.iterations);
    for it in 0..config.iterations {
        let dump = config.dump_rows_dir.as_deref().map(|dir| DumpRowsSpec {
            dir,
            engine,
            query_id: query.id,
            iteration: it + 1,
        });
        let res = run_once(ctx, query, native_rule_installed, dump).await;
        if config.progress {
            let _ = writeln!(
                stderr,
                "[bench {engine}] iter {}/{} q={} ({qpos}/{total}) elapsed_ms={} outcome={} rewritten={:?}",
                it + 1,
                config.iterations,
                query.id,
                res.elapsed_ms,
                res.outcome.as_str(),
                res.native_rewritten
            );
        }
        samples_ms.push(res.elapsed_ms);
        runs.push(res);
    }
    let check_iteration_consistency = parity_waiver_reason(query.id).is_none();
    aggregate_measurement(runs, samples_ms, check_iteration_consistency)
}

/// Collapse N per-iteration runs into a single `EngineMeasurement`. Policy:
///
/// 1. Any failed measured iteration → outcome `Failed`, first failure wins
///    for the message.
/// 2. If all iterations succeeded but their canonical fingerprints disagree,
///    outcome is forced to `Failed` with `inconsistent_iterations`. This
///    catches flaky correctness bugs that would otherwise be hidden when a
///    later iteration happens to match CPU.
/// 3. If all iterations succeeded and agree, the last iteration's
///    rows/fingerprint represent the canonical output.
///
/// All elapsed_ms samples are retained so consumers can still see jitter;
/// speedup/parity gates on the aggregated outcome.
fn aggregate_measurement(
    runs: Vec<RunResult>,
    samples_ms: Vec<u128>,
    check_iteration_consistency: bool,
) -> EngineMeasurement {
    assert!(!runs.is_empty(), "iterations >= 1");
    let iteration_failure_count = runs.iter().filter(|r| r.outcome == RunOutcome::Failed).count();
    let native_rewritten = runs.iter().find_map(|r| r.native_rewritten).map(|_| {
        // Treat as rewritten only if EVERY iteration was rewritten; mixed
        // behavior would mean the rule's decision is flipping, which is a
        // bug we want to surface rather than smooth over.
        runs.iter().all(|r| r.native_rewritten == Some(true))
    });
    if iteration_failure_count > 0 {
        let first_failure = runs.iter().find(|r| r.outcome == RunOutcome::Failed).expect("at least one failure");
        return EngineMeasurement {
            outcome: RunOutcome::Failed,
            failure: first_failure.failure.clone(),
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            output_fingerprint: None,
            native_rewritten,
            samples_ms,
            iteration_failure_count,
        };
    }
    if check_iteration_consistency && let Some(inconsistent) = first_inconsistent_iteration(&runs) {
        return EngineMeasurement {
            outcome: RunOutcome::Failed,
            failure: Some(format!(
                "inconsistent_iterations: iteration 1 fingerprint={} (rows={:?}) differs from iteration {} fingerprint={} (rows={:?})",
                runs[0].output_fingerprint.as_deref().unwrap_or("none"),
                runs[0].output_rows,
                inconsistent + 1,
                runs[inconsistent].output_fingerprint.as_deref().unwrap_or("none"),
                runs[inconsistent].output_rows,
            )),
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            output_fingerprint: None,
            native_rewritten,
            samples_ms,
            iteration_failure_count,
        };
    }
    let canonical = runs.last().expect("iterations >= 1");
    EngineMeasurement {
        outcome: RunOutcome::Succeeded,
        failure: None,
        output_batches: canonical.output_batches,
        output_rows: canonical.output_rows,
        output_bytes: canonical.output_bytes,
        output_fingerprint: canonical.output_fingerprint.clone(),
        native_rewritten,
        samples_ms,
        iteration_failure_count,
    }
}

/// Index of the first iteration whose output disagrees with iteration 0,
/// or `None` if all iterations agree. Compares fingerprint and row count
/// (both should match across iterations for a deterministic query).
fn first_inconsistent_iteration(runs: &[RunResult]) -> Option<usize> {
    let first = runs.first()?;
    let fp = first.output_fingerprint.as_deref();
    let rows = first.output_rows;
    runs.iter()
        .enumerate()
        .skip(1)
        .find(|(_, r)| r.output_fingerprint.as_deref() != fp || r.output_rows != rows)
        .map(|(idx, _)| idx)
}

struct RunResult {
    elapsed_ms: u128,
    outcome: RunOutcome,
    failure: Option<String>,
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
    output_fingerprint: Option<String>,
    native_rewritten: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunOutcome {
    Succeeded,
    Failed,
}

impl RunOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

async fn run_once(
    ctx: &SessionContext,
    query: &ClickbenchQuery,
    native_rule_installed: bool,
    dump: Option<DumpRowsSpec<'_>>,
) -> RunResult {
    let started = Instant::now();
    let (result, native_rewritten) = match ctx.sql(query.sql).await {
        Ok(df) => {
            let planned = if native_rule_installed {
                let (state, logical) = df.into_parts();
                plan_with_native_diagnostics(&state, &logical)
                    .await
                    .map(|(plan, diagnostics)| (plan, Some(diagnostics.report.selected_count() > 0)))
            } else {
                df.create_physical_plan().await.map(|plan| (plan, None))
            };
            match planned {
                Ok((plan, rewritten)) => (execute_plan(plan, ctx, dump).await, rewritten),
                Err(error) => (Err(format!("physical_plan: {error}")), None),
            }
        },
        Err(error) => (Err(format!("sql: {error}")), None),
    };
    let elapsed_ms = started.elapsed().as_millis();
    match result {
        Ok(success) => RunResult {
            elapsed_ms,
            outcome: RunOutcome::Succeeded,
            failure: None,
            output_batches: Some(success.batches),
            output_rows: Some(success.rows),
            output_bytes: Some(success.bytes),
            output_fingerprint: Some(success.fingerprint),
            native_rewritten,
        },
        Err(failure) => RunResult {
            elapsed_ms,
            outcome: RunOutcome::Failed,
            failure: Some(sanitize(failure)),
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            output_fingerprint: None,
            native_rewritten,
        },
    }
}

struct PlanSuccess {
    batches: usize,
    rows: usize,
    bytes: usize,
    fingerprint: String,
}

async fn execute_plan(
    plan: Arc<dyn ExecutionPlan>,
    ctx: &SessionContext,
    dump: Option<DumpRowsSpec<'_>>,
) -> Result<PlanSuccess, String> {
    let batches = collect(plan, ctx.task_ctx()).await.map_err(|error| format!("collect: {error}"))?;
    if let Some(dump) = dump {
        dump_record_batches_to_tsv(&batches, dump.path()).map_err(|error| format!("dump_rows: {error}"))?;
    }
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let bytes = batches.iter().map(get_record_batch_memory_size).sum();
    let fingerprint = fingerprint_batches(&batches).map_err(|error| format!("fingerprint: {error}"))?;
    Ok(PlanSuccess { batches: batches.len(), rows, bytes, fingerprint })
}

#[derive(Clone, Copy)]
struct DumpRowsSpec<'a> {
    dir: &'a Path,
    engine: &'a str,
    query_id: &'a str,
    iteration: usize,
}

impl DumpRowsSpec<'_> {
    fn path(&self) -> PathBuf {
        self.dir.join(format!("{}_{}_iter{}.tsv", self.query_id, self.engine, self.iteration))
    }
}

fn dump_record_batches_to_tsv(batches: &[RecordBatch], path: PathBuf) -> Result<(), String> {
    let mut file = File::create(&path).map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    let Some(first) = batches.first() else {
        return Ok(());
    };
    let schema = first.schema();
    let header = schema.fields().iter().map(|field| sanitize_tsv_field(field.name())).collect::<Vec<_>>().join("\t");
    writeln!(file, "{header}").map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    for batch in batches {
        for row_idx in 0..batch.num_rows() {
            for col_idx in 0..batch.num_columns() {
                if col_idx > 0 {
                    write!(file, "\t").map_err(|error| format!("failed to write {}: {error}", path.display()))?;
                }
                let column = batch.column(col_idx);
                if column.is_null(row_idx) {
                    write!(file, "NULL").map_err(|error| format!("failed to write {}: {error}", path.display()))?;
                } else {
                    let value = array_value_to_string(column, row_idx).map_err(|error| {
                        format!("failed to format {} row {row_idx} column {col_idx}: {error}", path.display())
                    })?;
                    write!(file, "{}", sanitize_tsv_field(&value))
                        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
                }
            }
            writeln!(file).map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

/// Order-insensitive row fingerprint for parity comparisons.
///
/// Rows are serialized, sorted, then hashed so untied `GROUP BY ... LIMIT`
/// results may differ in row order. Column indexes remain encoded.
///
/// Values are tagged so `NULL` and `""` do not collide in Arrow string output.
fn fingerprint_batches(batches: &[RecordBatch]) -> Result<String, ArrowError> {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };
    let mut rows: Vec<String> = Vec::new();
    for batch in batches {
        let num_cols = batch.num_columns();
        for row_idx in 0..batch.num_rows() {
            let mut encoded = String::new();
            for col_idx in 0..num_cols {
                if col_idx > 0 {
                    encoded.push('\u{1f}');
                }
                let column = batch.column(col_idx);
                if column.is_null(row_idx) {
                    encoded.push('\0');
                } else {
                    encoded.push('\u{1}');
                    encoded.push_str(&array_value_to_string(column, row_idx)?);
                }
            }
            rows.push(encoded);
        }
    }
    rows.sort();
    let mut hasher = DefaultHasher::new();
    rows.len().hash(&mut hasher);
    for row in &rows {
        row.hash(&mut hasher);
    }
    Ok(format!("{:016x}", hasher.finish()))
}

struct EngineMeasurement {
    outcome: RunOutcome,
    failure: Option<String>,
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
    output_fingerprint: Option<String>,
    /// `None` on the CPU side. On the native side, `Some(true)` when the
    /// optimizer rule actually replaced a sub-plan with a native node, else
    /// `Some(false)`; in which case the recorded wall time is the retained
    /// DataFusion baseline, not GPU execution. Used to gate `speedup_vs_cpu`
    /// so the report cannot publish DataFusion-baseline time as native time.
    native_rewritten: Option<bool>,
    samples_ms: Vec<u128>,
    /// Count of measured iterations that did not produce a successful result.
    /// `> 0` forces the aggregated outcome to `Failed` even when other
    /// iterations succeeded, so flaky runs aren't laundered into "succeeded".
    iteration_failure_count: usize,
}

impl EngineMeasurement {
    fn mean_ms(&self) -> Option<f64> {
        if self.samples_ms.is_empty() {
            return None;
        }
        let total: u128 = self.samples_ms.iter().sum();
        Some(total as f64 / self.samples_ms.len() as f64)
    }

    fn median_ms(&self) -> Option<u128> {
        if self.samples_ms.is_empty() {
            return None;
        }
        let mut sorted = self.samples_ms.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        Some(if n.is_multiple_of(2) { (sorted[n / 2 - 1] + sorted[n / 2]) / 2 } else { sorted[n / 2] })
    }

    fn min_ms(&self) -> Option<u128> {
        self.samples_ms.iter().copied().min()
    }

    fn max_ms(&self) -> Option<u128> {
        self.samples_ms.iter().copied().max()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParityStatus {
    /// Both engines succeeded and produced byte-equal canonicalized outputs.
    Matched,
    /// Query is covered by a reason-coded ClickBench parity waiver. Row counts
    /// agree; content is not compared.
    RowCountMatchedNondeterministic,
    /// CPU vs native disagree on row count.
    RowCountMismatch,
    /// Row counts agree but order-insensitive fingerprints differ.
    ContentMismatch,
    /// Both engines succeeded but the native rule did not rewrite this plan,
    /// so the recorded native timing is the retained DataFusion baseline, not
    /// native execution.
    /// Parity is trivially equal here; the row is flagged so consumers
    /// don't conflate it with a real native run.
    NativeNotRewritten,
    CpuOnlyFailed,
    NativeOnlyFailed,
    BothFailed,
    NotApplicable,
}

impl ParityStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::RowCountMatchedNondeterministic => "row_count_matched_nondeterministic",
            Self::RowCountMismatch => "row_count_mismatch",
            Self::ContentMismatch => "content_mismatch",
            Self::NativeNotRewritten => "native_not_rewritten",
            Self::CpuOnlyFailed => "cpu_only_failed",
            Self::NativeOnlyFailed => "native_only_failed",
            Self::BothFailed => "both_failed",
            Self::NotApplicable => "n/a",
        }
    }
}

fn compute_parity(query_id: &str, cpu: Option<&EngineMeasurement>, native: Option<&EngineMeasurement>) -> ParityStatus {
    let (cpu, native) = match (cpu, native) {
        (Some(c), Some(n)) => (c, n),
        _ => return ParityStatus::NotApplicable,
    };
    match (cpu.outcome, native.outcome) {
        (RunOutcome::Failed, RunOutcome::Failed) => ParityStatus::BothFailed,
        (RunOutcome::Failed, RunOutcome::Succeeded) => ParityStatus::CpuOnlyFailed,
        (RunOutcome::Succeeded, RunOutcome::Failed) => ParityStatus::NativeOnlyFailed,
        (RunOutcome::Succeeded, RunOutcome::Succeeded) => {
            if native.native_rewritten == Some(false) {
                return ParityStatus::NativeNotRewritten;
            }
            match (cpu.output_rows, native.output_rows) {
                (Some(cr), Some(nr)) if cr != nr => ParityStatus::RowCountMismatch,
                _ => {
                    if parity_waiver_reason(query_id).is_some() {
                        return ParityStatus::RowCountMatchedNondeterministic;
                    }
                    match (cpu.output_fingerprint.as_deref(), native.output_fingerprint.as_deref()) {
                        (Some(cf), Some(nf)) if cf != nf => ParityStatus::ContentMismatch,
                        _ => ParityStatus::Matched,
                    }
                },
            }
        },
    }
}

struct BenchmarkRow {
    query_id: String,
    cpu: Option<EngineMeasurement>,
    native: Option<EngineMeasurement>,
    parity_status: ParityStatus,
}

impl BenchmarkRow {
    fn build(query_id: &str, cpu: Option<EngineMeasurement>, native: Option<EngineMeasurement>) -> Self {
        let parity_status = compute_parity(query_id, cpu.as_ref(), native.as_ref());
        Self { query_id: query_id.to_string(), cpu, native, parity_status }
    }

    /// `speedup_vs_cpu` is only meaningful when both sides succeeded **and**
    /// the native rule actually rewrote the plan; otherwise the "native"
    /// wall time is a retained DataFusion baseline and a 1.0× ratio would be
    /// misleading.
    fn speedup_vs_cpu(&self) -> Option<f64> {
        let cpu = self.cpu.as_ref()?;
        let native = self.native.as_ref()?;
        if cpu.outcome != RunOutcome::Succeeded || native.outcome != RunOutcome::Succeeded {
            return None;
        }
        if native.native_rewritten == Some(false) {
            return None;
        }
        let cpu_mean = cpu.mean_ms()?;
        let native_mean = native.mean_ms()?;
        if native_mean <= 0.0 { None } else { Some(cpu_mean / native_mean) }
    }
}

struct BenchmarkReport {
    config: BenchmarkConfig,
    metadata: ClickbenchDatasetMetadata,
    rows: Vec<BenchmarkRow>,
}

impl BenchmarkReport {
    fn to_tsv(&self) -> String {
        let mut output = String::new();
        writeln!(output, "# workload=clickbench").unwrap();
        writeln!(output, "# dataset_root={}", self.metadata.root.display()).unwrap();
        writeln!(output, "# dataset_layout={}", self.metadata.layout_label()).unwrap();
        writeln!(output, "# dataset_provenance={}", self.metadata.provenance).unwrap();
        writeln!(output, "# dataset_generation_command={}", self.metadata.generation_command).unwrap();
        writeln!(output, "# dataset_file_count={}", self.metadata.file_count).unwrap();
        writeln!(output, "# dataset_total_rows={}", self.metadata.total_rows).unwrap();
        writeln!(output, "# dataset_total_bytes={}", self.metadata.total_bytes).unwrap();
        writeln!(output, "# dataset_metadata_fingerprint={}", self.metadata.metadata_fingerprint).unwrap();
        if let Some(fp) = &self.metadata.content_fingerprint {
            writeln!(output, "# dataset_content_fingerprint={fp}").unwrap();
        }
        let query_ids_label =
            if self.config.query_ids.is_empty() { "all".to_string() } else { self.config.query_ids.join(",") };
        writeln!(output, "# clickbench_benchmark_query_ids={query_ids_label}").unwrap();
        writeln!(output, "# clickbench_benchmark_warmups={}", self.config.warmup_iterations).unwrap();
        writeln!(output, "# clickbench_benchmark_iterations={}", self.config.iterations).unwrap();
        writeln!(output, "# clickbench_benchmark_wall_time_fields=per_measured_iteration_mean").unwrap();
        writeln!(output, "# clickbench_benchmark_cpu_profile={}", self.config.cpu_profile.as_str()).unwrap();
        writeln!(output, "# clickbench_benchmark_skip_cpu={}", self.config.skip_cpu).unwrap();
        writeln!(output, "# clickbench_benchmark_skip_native={}", self.config.skip_native).unwrap();
        writeln!(output, "# clickbench_benchmark_relational_execution={}", self.config.native_execution.as_str())
            .unwrap();
        writeln!(output, "# clickbench_benchmark_native_profile_name={}", self.config.native_profile.profile_name)
            .unwrap();
        writeln!(output, "# clickbench_benchmark_native_profile_source={}", self.config.native_profile.profile_source)
            .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_native_device_budget_bytes={}",
            self.config.native_profile.device_budget_bytes
        )
        .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_native_max_source_chunk_bytes={}",
            self.config.native_profile.max_source_chunk_bytes
        )
        .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_native_max_row_groups_per_chunk={}",
            self.config.native_profile.max_row_groups_per_chunk
        )
        .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_native_max_retry_attempts={}",
            self.config.native_profile.max_retry_attempts
        )
        .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_native_min_retry_chunk_row_groups={}",
            self.config.native_profile.min_retry_chunk_row_groups
        )
        .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_native_source_chunk_planning_mode={}",
            self.config.native_profile.source_chunk_planning_mode.as_str()
        )
        .unwrap();
        writeln!(
            output,
            "# clickbench_benchmark_kvikio_compat_mode={}",
            std::env::var(KVIKIO_COMPAT_MODE_ENV).unwrap_or_else(|_| "unset".to_string())
        )
        .unwrap();
        writeln!(output, "# clickbench_benchmark_parity_fingerprint=order_insensitive").unwrap();
        writeln!(output, "# clickbench_benchmark_nondeterministic_queries={}", parity_waived_query_ids_label())
            .unwrap();
        writeln!(output, "# clickbench_benchmark_parity_waivers={}", parity_waiver_reasons_label()).unwrap();
        if let Some(dir) = &self.config.dump_rows_dir {
            writeln!(output, "# clickbench_benchmark_dump_rows_dir={}", dir.display()).unwrap();
        }
        writeln!(output, "# clickbench_benchmark_iteration_failure_policy=any_failed_iteration_fails_row").unwrap();
        writeln!(output, "{}", TSV_COLUMNS.join("\t")).unwrap();
        for row in &self.rows {
            writeln!(output, "{}", row.to_tsv_line()).unwrap();
        }
        output
    }
}

impl BenchmarkRow {
    fn to_tsv_line(&self) -> String {
        let mut fields: Vec<String> = Vec::with_capacity(TSV_COLUMNS.len());
        fields.push(self.query_id.clone());
        fields.push(self.parity_status.as_str().to_string());
        fields.push(format_opt_f64(self.speedup_vs_cpu()));
        let native_rewritten_label = match self.native.as_ref().and_then(|m| m.native_rewritten) {
            Some(true) => "true",
            Some(false) => "false",
            None => "n/a",
        };
        fields.push(native_rewritten_label.to_string());
        push_engine_columns(&mut fields, self.cpu.as_ref());
        push_engine_columns(&mut fields, self.native.as_ref());
        debug_assert_eq!(fields.len(), TSV_COLUMNS.len());
        fields.join("\t")
    }
}

fn push_engine_columns(fields: &mut Vec<String>, engine: Option<&EngineMeasurement>) {
    match engine {
        Some(m) => {
            fields.push(m.outcome.as_str().to_string());
            fields.push(m.iteration_failure_count.to_string());
            fields.push(format_opt_f64(m.mean_ms()));
            fields.push(format_opt_u128(m.min_ms()));
            fields.push(format_opt_u128(m.median_ms()));
            fields.push(format_opt_u128(m.max_ms()));
            fields.push(format_opt_usize(m.output_batches));
            fields.push(format_opt_usize(m.output_rows));
            fields.push(format_opt_usize(m.output_bytes));
            fields.push(m.output_fingerprint.clone().unwrap_or_else(|| "none".into()));
            fields.push(m.failure.clone().unwrap_or_else(|| "none".into()));
        },
        None => {
            for _ in 0..11 {
                fields.push("skipped".into());
            }
        },
    }
}

fn format_opt_f64(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.3}")).unwrap_or_else(|| "none".into())
}

fn format_opt_u128(value: Option<u128>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "none".into())
}

fn format_opt_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "none".into())
}

fn sanitize(s: String) -> String {
    s.replace(['\t', '\n'], " ")
}

fn sanitize_tsv_field(s: &str) -> String {
    s.replace(['\t', '\r', '\n'], " ")
}

fn env_usize(name: &str, default: usize) -> WorkloadResult<usize> {
    match std::env::var(name) {
        Ok(value) => value.parse().map_err(|_| invalid_env(format!("{name}={value} is not a non-negative integer"))),
        Err(_) => Ok(default),
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from)
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref().map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn query_ids_from_env(name: &str) -> WorkloadResult<Vec<String>> {
    match std::env::var(name) {
        Ok(value) => {
            let mut ids: Vec<String> =
                value.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            for id in &ids {
                if !CLICKBENCH_QUERIES.iter().any(|q| q.id == id) {
                    return Err(invalid_env(format!("{name} references unknown query id {id}")));
                }
            }
            ids.dedup();
            Ok(ids)
        },
        Err(_) => Ok(Vec::new()),
    }
}

fn invalid_env(message: String) -> WorkloadError {
    WorkloadError::InvalidDataset(message)
}

#[cfg(test)]
mod tests {
    use query_engine::runtime::SourceChunkPlanningMode;

    use super::*;

    #[test]
    fn parity_waiver_surface_is_reason_coded_and_ordered() {
        let ids = PARITY_WAIVERS.iter().map(|waiver| waiver.query_id).collect::<Vec<_>>();
        assert_eq!(ids, vec!["q04", "q18", "q25", "q26", "q31", "q32", "q33", "q39", "q40", "q41", "q42"]);

        assert_eq!(parity_waiver_reason("q04"), Some(ParityWaiverReason::FloatAccumulationVariance));
        assert_eq!(parity_waiver_reason("q10"), None);
        assert_eq!(parity_waiver_reason("q23"), None);
        for query_id in ["q18", "q25", "q26", "q31", "q32", "q33", "q39", "q40", "q41", "q42"] {
            assert_eq!(parity_waiver_reason(query_id), Some(ParityWaiverReason::ResultSetNondeterministic));
        }

        for query_id in ids {
            assert!(CLICKBENCH_QUERIES.iter().any(|query| query.id == query_id), "unknown ClickBench query {query_id}");
        }
    }

    #[test]
    fn parity_waivers_compare_row_counts_only_for_all_reason_categories() {
        for query_id in ["q04", "q39", "q40", "q41", "q42"] {
            let cpu = succeeded_measurement(10, "cpu-fingerprint", None);
            let native = succeeded_measurement(10, "native-fingerprint", Some(true));
            assert_eq!(
                compute_parity(query_id, Some(&cpu), Some(&native)),
                ParityStatus::RowCountMatchedNondeterministic,
                "{query_id}"
            );
        }

        let cpu = succeeded_measurement(10, "cpu-fingerprint", None);
        let native = succeeded_measurement(10, "native-fingerprint", Some(true));
        assert_eq!(compute_parity("q23", Some(&cpu), Some(&native)), ParityStatus::ContentMismatch);
        let cpu = succeeded_measurement(10, "cpu-fingerprint", None);
        let native = succeeded_measurement(10, "native-fingerprint", Some(true));
        assert_eq!(compute_parity("q43", Some(&cpu), Some(&native)), ParityStatus::ContentMismatch);
    }

    #[test]
    fn tsv_header_exposes_backward_compatible_list_and_reason_metadata() {
        let report = BenchmarkReport { config: test_config(), metadata: test_metadata(), rows: Vec::new() };
        let tsv = report.to_tsv();
        assert_eq!(TSV_COLUMNS.first(), Some(&"query_id"));
        assert_eq!(TSV_COLUMNS.last(), Some(&"native_failure"));
        assert_eq!(TSV_COLUMNS.join("\t"), tsv.lines().find(|line| !line.starts_with('#')).unwrap());
        assert!(
            tsv.contains(
                "# clickbench_benchmark_nondeterministic_queries=q04,q18,q25,q26,q31,q32,q33,q39,q40,q41,q42\n"
            )
        );
        assert!(tsv.contains("q04=float_accumulation_variance"));
        assert!(!tsv.contains("q10="));
        assert!(!tsv.contains("q23="));
        assert!(tsv.contains("q39=result_set_nondeterministic"));
        assert!(tsv.contains("q42=result_set_nondeterministic"));
        assert!(tsv.contains("# clickbench_benchmark_relational_execution=native_no_datafusion_cpu\n"));
        assert!(tsv.contains("# clickbench_benchmark_native_profile_name=clickbench_gpu_throughput\n"));
        assert!(tsv.contains("# clickbench_benchmark_native_profile_source=env_profile\n"));
        assert!(tsv.contains("# clickbench_benchmark_native_max_source_chunk_bytes=34359738368\n"));
        assert!(tsv.contains("# clickbench_benchmark_native_max_row_groups_per_chunk=1024\n"));
        assert!(tsv.contains("# clickbench_benchmark_native_source_chunk_planning_mode=local_parquet_throughput\n"));
        assert!(tsv.contains("# clickbench_benchmark_kvikio_compat_mode="));
    }

    #[test]
    fn nondeterministic_iterations_are_not_failed_for_waived_queries() {
        let runs = vec![run_result(10, "first"), run_result(10, "second")];
        let measurement = aggregate_measurement(runs, vec![1, 2], parity_waiver_reason("q40").is_none());
        assert_eq!(measurement.outcome, RunOutcome::Succeeded);
        assert_eq!(measurement.output_fingerprint.as_deref(), Some("second"));

        let runs = vec![run_result(10, "first"), run_result(10, "second")];
        let measurement = aggregate_measurement(runs, vec![1, 2], parity_waiver_reason("q43").is_none());
        assert_eq!(measurement.outcome, RunOutcome::Failed);
        assert!(measurement.failure.as_deref().unwrap_or_default().contains("inconsistent_iterations"));
    }

    fn succeeded_measurement(rows: usize, fingerprint: &str, native_rewritten: Option<bool>) -> EngineMeasurement {
        EngineMeasurement {
            outcome: RunOutcome::Succeeded,
            failure: None,
            output_batches: Some(1),
            output_rows: Some(rows),
            output_bytes: Some(128),
            output_fingerprint: Some(fingerprint.to_string()),
            native_rewritten,
            samples_ms: vec![10],
            iteration_failure_count: 0,
        }
    }

    fn run_result(rows: usize, fingerprint: &str) -> RunResult {
        RunResult {
            elapsed_ms: 1,
            outcome: RunOutcome::Succeeded,
            failure: None,
            output_batches: Some(1),
            output_rows: Some(rows),
            output_bytes: Some(128),
            output_fingerprint: Some(fingerprint.to_string()),
            native_rewritten: Some(true),
        }
    }

    fn test_config() -> BenchmarkConfig {
        use algeon_datafusion::native_exec::AlgeonNativeExecutionConfig;

        use crate::report_workload::clickbench::{
            CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES, CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES, CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT,
        };

        BenchmarkConfig {
            query_ids: Vec::new(),
            warmup_iterations: 0,
            iterations: 1,
            skip_cpu: false,
            skip_native: false,
            cpu_profile: CpuProfile::Tuned,
            native_execution: RelationalExecutionSelection::strict_native(),
            native_profile: ClickbenchNativeProfileSettings {
                profile_name: CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT,
                profile_source: "env_profile",
                device_budget_bytes: CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES,
                max_source_chunk_bytes: CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES,
                max_row_groups_per_chunk: CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
                max_retry_attempts: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                min_retry_chunk_row_groups: AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
                source_chunk_planning_mode: SourceChunkPlanningMode::LocalParquetThroughput,
            },
            progress: false,
            dump_rows_dir: None,
        }
    }

    fn test_metadata() -> ClickbenchDatasetMetadata {
        ClickbenchDatasetMetadata {
            root: PathBuf::from("/tmp/clickbench"),
            layout: ClickbenchLayout::Single { file: PathBuf::from("/tmp/clickbench/hits.parquet") },
            provenance: "test".to_string(),
            generation_command: "test".to_string(),
            file_count: 1,
            total_rows: 0,
            total_bytes: 0,
            schema: "schema".to_string(),
            row_group_count: 0,
            row_group_rows: Vec::new(),
            row_group_bytes: Vec::new(),
            metadata_fingerprint: "metadata-fnv1a64:test".to_string(),
            content_fingerprint: None,
        }
    }
}
