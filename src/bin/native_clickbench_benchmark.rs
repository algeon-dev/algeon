use std::{
    fmt::Write as _,
    io::{self, Write as _},
    process::ExitCode,
    sync::Arc,
    time::Instant,
};

use arrow::{array::RecordBatch, util::display::array_value_to_string};
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CLICKBENCH_QUERIES, ClickbenchDatasetMetadata, ClickbenchLayout, ClickbenchQuery, ClickbenchWorkloadError,
    ClickbenchWorkloadResult, CudfNativePhysicalOptimizerRule, CudfSessionStateBuilderExt, NativePipelineExecutionMode,
    clickbench_native_optimizer_config, clickbench_report_session_config, clickbench_tuned_cpu_session_config,
    discover_clickbench_layout, prepare_clickbench_report_dataset, register_clickbench_table,
};

const SCHEMA_VERSION: u32 = 1;

/// Queries whose result *set* — not just row order — is engine-defined,
/// typically because they use `LIMIT` without a deterministic `ORDER BY`
/// tiebreaker. For these queries CPU and native may legally pick different
/// groups, so the benchmark compares row counts only and emits
/// `parity_status=row_count_matched_nondeterministic` instead of running the
/// content fingerprint check. The iteration-consistency check is also
/// disabled for these queries (the engine itself may return different rows
/// across runs).
///
/// This list is curated. Add a query here when it satisfies BOTH:
/// 1. The SQL has `LIMIT` without a deterministic total ordering at the
///    boundary (no `ORDER BY`, or `ORDER BY` columns that admit ties), AND
/// 2. The fixture in practice exhibits ties at the LIMIT boundary.
///
/// Current entries:
/// - q18: `GROUP BY ... LIMIT 10` with no `ORDER BY`.
/// - q25: `ORDER BY EventTime LIMIT 10` — timestamp ties at the boundary.
/// - q26: `ORDER BY SearchPhrase LIMIT 10` — many duplicate phrases at the
///   boundary even with `WHERE SearchPhrase <> ''`.
const NONDETERMINISTIC_QUERIES: &[&str] = &["q18", "q25", "q26"];

const QUERY_IDS_ENV: &str = "CUDF_CLICKBENCH_BENCH_QUERY_IDS";
const WARMUPS_ENV: &str = "CUDF_CLICKBENCH_BENCH_WARMUPS";
const ITERATIONS_ENV: &str = "CUDF_CLICKBENCH_BENCH_ITERATIONS";
const SKIP_CPU_ENV: &str = "CUDF_CLICKBENCH_BENCH_SKIP_CPU";
const SKIP_NATIVE_ENV: &str = "CUDF_CLICKBENCH_BENCH_SKIP_NATIVE";
const CPU_PROFILE_ENV: &str = "CUDF_CLICKBENCH_BENCH_CPU_PROFILE";
const PIPELINE_MODE_ENV: &str = "CUDF_NATIVE_PIPELINE_EXECUTION_MODE";
const PROGRESS_ENV: &str = "CUDF_CLICKBENCH_BENCH_PROGRESS";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr().lock(), "native_clickbench_benchmark: {error}");
            ExitCode::FAILURE
        },
    }
}

async fn run() -> ClickbenchWorkloadResult<()> {
    let config = BenchmarkConfig::from_env()?;
    let (root, metadata) = prepare_clickbench_report_dataset()?.ok_or_else(|| {
        ClickbenchWorkloadError::InvalidDataset(
            "no ClickBench fixture found; set CUDF_CLICKBENCH_PARQUET_DIR or stage \
             tests/data/clickbench/{hits.parquet, partitioned/} before running this benchmark"
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
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    progress: bool,
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
    fn from_env() -> ClickbenchWorkloadResult<Self> {
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
        let native_pipeline_execution_mode = match std::env::var(PIPELINE_MODE_ENV) {
            Ok(value) => value
                .parse::<NativePipelineExecutionMode>()
                .map_err(|error| invalid_env(format!("unsupported {PIPELINE_MODE_ENV}: {error}")))?,
            Err(_) => NativePipelineExecutionMode::StrictBounded,
        };
        let query_ids = query_ids_from_env(QUERY_IDS_ENV)?;
        Ok(Self {
            query_ids,
            warmup_iterations,
            iterations,
            skip_cpu,
            skip_native,
            cpu_profile,
            native_pipeline_execution_mode,
            progress: env_truthy(PROGRESS_ENV),
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
) -> ClickbenchWorkloadResult<BenchmarkReport> {
    let queries = config.selected_queries();
    let cpu_ctx = if config.skip_cpu { None } else { Some(build_cpu_ctx(config, layout).await?) };
    let native_setup = if config.skip_native { None } else { Some(build_native_ctx(config, layout).await?) };
    let total = queries.len();
    let mut stderr = io::stderr().lock();
    let mut rows = Vec::with_capacity(total);
    for (idx, query) in queries.iter().enumerate() {
        let qpos = idx + 1;
        let cpu = match &cpu_ctx {
            Some(ctx) => Some(measure_query(ctx, query, "cpu", qpos, total, config, None, &mut stderr).await),
            None => None,
        };
        let native = match &native_setup {
            Some((ctx, rule)) => {
                Some(measure_query(ctx, query, "native", qpos, total, config, Some(rule.as_ref()), &mut stderr).await)
            },
            None => None,
        };
        rows.push(BenchmarkRow::build(query.id, cpu, native));
    }
    Ok(BenchmarkReport { config: config.clone(), metadata: metadata.clone(), rows })
}

async fn build_cpu_ctx(
    config: &BenchmarkConfig,
    layout: &ClickbenchLayout,
) -> ClickbenchWorkloadResult<SessionContext> {
    let ctx = SessionContext::new_with_config(config.cpu_profile.session_config());
    register_clickbench_table(&ctx, layout).await?;
    Ok(ctx)
}

async fn build_native_ctx(
    config: &BenchmarkConfig,
    layout: &ClickbenchLayout,
) -> ClickbenchWorkloadResult<(SessionContext, Arc<CudfNativePhysicalOptimizerRule>)> {
    let optimizer_config =
        clickbench_native_optimizer_config().with_native_pipeline_execution_mode(config.native_pipeline_execution_mode);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(clickbench_report_session_config())
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_clickbench_table(&ctx, layout).await?;
    Ok((ctx, native_rule))
}

async fn measure_query(
    ctx: &SessionContext,
    query: &ClickbenchQuery,
    engine: &str,
    qpos: usize,
    total: usize,
    config: &BenchmarkConfig,
    native_rule: Option<&CudfNativePhysicalOptimizerRule>,
    stderr: &mut io::StderrLock<'_>,
) -> EngineMeasurement {
    for w in 0..config.warmup_iterations {
        let res = run_once(ctx, query, native_rule).await;
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
        let res = run_once(ctx, query, native_rule).await;
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
    let check_iteration_consistency = !NONDETERMINISTIC_QUERIES.contains(&query.id);
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
    native_rule: Option<&CudfNativePhysicalOptimizerRule>,
) -> RunResult {
    let started = Instant::now();
    let (result, native_rewritten) = match ctx.sql(query.sql).await {
        Ok(df) => match df.create_physical_plan().await {
            Ok(plan) => {
                let rewritten = native_rule
                    .map(|rule| rule.last_planning_diagnostics().is_some_and(|diag| diag.report.rewritten_count() > 0));
                (execute_plan(plan, ctx).await, rewritten)
            },
            Err(error) => (Err(format!("physical_plan: {error}")), None),
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

async fn execute_plan(plan: Arc<dyn ExecutionPlan>, ctx: &SessionContext) -> Result<PlanSuccess, String> {
    let batches = collect(plan, ctx.task_ctx()).await.map_err(|error| format!("collect: {error}"))?;
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let bytes = batches.iter().map(get_record_batch_memory_size).sum();
    let fingerprint = fingerprint_batches(&batches).map_err(|error| format!("fingerprint: {error}"))?;
    Ok(PlanSuccess { batches: batches.len(), rows, bytes, fingerprint })
}

/// Order-insensitive row fingerprint: serialize each row to a `\u{1f}`-joined
/// string, sort lexicographically, then hash. Used for parity comparisons
/// because many ClickBench queries are GROUP BY ... LIMIT without an
/// ORDER BY tiebreaker (e.g. q18), so CPU and native may legally return rows
/// in different orders. The column index is included in each row's encoding
/// so column reorderings would still register as a mismatch.
///
/// Null vs empty-string distinction: Arrow's `array_value_to_string` renders
/// both `NULL` and `""` as an empty string, which would let one engine's
/// `NULL` collide with the other engine's `""`. ClickBench string columns
/// (e.g. `SearchPhrase`, `URL`, `Referer`) commonly mix the two, so we
/// prefix each value with a one-byte tag: `\0` for null, `\u{1}` for a real
/// value. The tag bytes are also separators so column reorderings still
/// register as a mismatch.
fn fingerprint_batches(batches: &[RecordBatch]) -> Result<String, arrow::error::ArrowError> {
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
    /// `Some(false)` — in which case the recorded wall time is CPU fallback,
    /// not GPU execution. Used to gate `speedup_vs_cpu` so the report cannot
    /// publish CPU-fallback time as native time.
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
    /// Query is in [`NONDETERMINISTIC_QUERIES`] (engine-defined result set
    /// because of `LIMIT` without `ORDER BY`). Row counts agree; content is
    /// not compared.
    RowCountMatchedNondeterministic,
    /// CPU vs native disagree on row count.
    RowCountMismatch,
    /// Row counts agree but order-insensitive fingerprints differ.
    ContentMismatch,
    /// Both engines succeeded but the native rule did not rewrite this plan,
    /// so the recorded native timing is CPU fallback, not native execution.
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
                    if NONDETERMINISTIC_QUERIES.contains(&query_id) {
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
    /// the native rule actually rewrote the plan — otherwise the "native"
    /// wall time is CPU fallback and a 1.0× ratio would be misleading.
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
        writeln!(output, "# native_clickbench_benchmark_tsv_schema_version={SCHEMA_VERSION}").unwrap();
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
        writeln!(
            output,
            "# clickbench_benchmark_native_pipeline_execution_mode={}",
            self.config.native_pipeline_execution_mode.as_str()
        )
        .unwrap();
        writeln!(output, "# clickbench_benchmark_parity_fingerprint=order_insensitive").unwrap();
        writeln!(output, "# clickbench_benchmark_nondeterministic_queries={}", NONDETERMINISTIC_QUERIES.join(","))
            .unwrap();
        writeln!(output, "# clickbench_benchmark_iteration_failure_policy=any_failed_iteration_fails_row").unwrap();
        writeln!(
            output,
            "query_id\tparity_status\tspeedup_vs_cpu\tnative_rewritten\
             \tcpu_outcome\tcpu_iteration_failures\tcpu_wall_ms_mean\tcpu_wall_ms_min\tcpu_wall_ms_median\tcpu_wall_ms_max\
             \tcpu_output_batches\tcpu_output_rows\tcpu_output_bytes\tcpu_output_fingerprint\tcpu_failure\
             \tnative_outcome\tnative_iteration_failures\tnative_wall_ms_mean\tnative_wall_ms_min\tnative_wall_ms_median\tnative_wall_ms_max\
             \tnative_output_batches\tnative_output_rows\tnative_output_bytes\tnative_output_fingerprint\tnative_failure"
        )
        .unwrap();
        for row in &self.rows {
            writeln!(output, "{}", row.to_tsv_line()).unwrap();
        }
        output
    }
}

impl BenchmarkRow {
    fn to_tsv_line(&self) -> String {
        let mut fields: Vec<String> = Vec::with_capacity(24);
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

fn env_usize(name: &str, default: usize) -> ClickbenchWorkloadResult<usize> {
    match std::env::var(name) {
        Ok(value) => value.parse().map_err(|_| invalid_env(format!("{name}={value} is not a non-negative integer"))),
        Err(_) => Ok(default),
    }
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref().map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn query_ids_from_env(name: &str) -> ClickbenchWorkloadResult<Vec<String>> {
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

fn invalid_env(message: String) -> ClickbenchWorkloadError {
    ClickbenchWorkloadError::InvalidDataset(message)
}
