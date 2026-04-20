use std::{
    any::Any,
    fs,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
};

use datafusion::{
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_report::{
        NativePlanningFailure, NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput,
        NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityRow,
        NativeTpcdsParityWorkloadItem, ParityFloatTolerance,
    },
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
    session::CudfSessionStateBuilderExt,
};
use datafusion_nexus_workloads::tpcds::TpcdsQuery;
use futures::FutureExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{
    TPCDS_QUERIES, TPCDS_TABLES, TpcdsWorkloadResult, discover_tpcds_table_layout, native_tpcds_session_config,
    register_tpcds_tables_with_native_metadata,
};

/// External oracle configuration for TPC-DS parity runs.
#[derive(Debug, Clone)]
pub struct TpcdsExternalOracleConfig {
    /// Directory used for generated SQL and Parquet oracle outputs.
    pub scratch_dir: PathBuf,
    /// Optional DuckDB-compatible oracle process.
    pub duckdb: Option<TpcdsExternalEngineConfig>,
}

/// One external engine process used by the TPC-DS parity oracle surface.
#[derive(Debug, Clone)]
pub struct TpcdsExternalEngineConfig {
    /// Executable invoked with SQL on stdin and an in-memory database argument.
    pub binary: PathBuf,
}

/// Per-query lifecycle events emitted by the TPC-DS parity row builder.
/// `query_index` is 1-based and counts only the queries selected for this run.
pub enum TpcdsParityProgressEvent<'a> {
    /// A selected query is about to execute CPU and native parity work.
    QueryStarted {
        /// One-based index among selected queries.
        query_index: usize,
        /// Total selected query count.
        total: usize,
        /// Stable TPC-DS query id.
        query_id: &'a str,
    },
    /// A selected query has been reduced to its report row. Heavy CPU/native
    /// Arrow batches have already been consumed by row construction.
    RowReady {
        /// One-based index among selected queries.
        query_index: usize,
        /// Total selected query count.
        total: usize,
        /// Stable TPC-DS query id.
        query_id: &'a str,
        /// Completed row for this query.
        row: &'a NativeTpcdsParityRow,
    },
}

/// Build parity workload items for the TPC-DS workload. For each selected query
/// the runner executes both CPU DataFusion and native-rule DataFusion and
/// captures schema/batches/diagnostics/metrics needed by the parity report.
pub async fn tpcds_parity_workload_items(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    tpcds_parity_workload_items_with_external_oracles(root, session_config, optimizer_config, query_filter, None).await
}

/// Build parity workload items using distinct CPU and native session configs.
///
/// This is useful for tests that want a more parallel CPU oracle while keeping
/// native physical-plan coverage pinned to the report surface's shape contract.
pub async fn tpcds_parity_workload_items_with_session_configs(
    root: &Path,
    cpu_session_config: SessionConfig,
    native_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    tpcds_parity_workload_items_with_session_configs_and_external_oracles(
        root,
        cpu_session_config,
        native_session_config,
        optimizer_config,
        query_filter,
        None,
    )
    .await
}

/// Build TPC-DS parity items and optionally run external oracles.
///
/// `query_filter` matches exact query ids and silently skips unknown ids.
/// External engines write one SQL file and one Parquet result per selected
/// query under `scratch_dir`; failures are captured in the workload item rather
/// than aborting the whole run.
pub async fn tpcds_parity_workload_items_with_external_oracles(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
    external_oracles: Option<TpcdsExternalOracleConfig>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    tpcds_parity_workload_items_with_session_configs_and_external_oracles(
        root,
        session_config.clone(),
        session_config,
        optimizer_config,
        query_filter,
        external_oracles,
    )
    .await
}

/// Build TPC-DS parity report rows and optionally run external oracles.
///
/// Unlike [`tpcds_parity_workload_items_with_external_oracles`], this converts
/// each query into a stable report row before starting the next query so CPU and
/// native output batches do not stay retained for the full selected workload.
pub async fn tpcds_parity_report_rows_with_external_oracles(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
    external_oracles: Option<TpcdsExternalOracleConfig>,
    float_tolerance: ParityFloatTolerance,
    progress: Option<&mut dyn FnMut(TpcdsParityProgressEvent<'_>)>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityRow>> {
    tpcds_parity_report_rows_with_session_configs_and_external_oracles(
        root,
        session_config.clone(),
        session_config,
        optimizer_config,
        query_filter,
        external_oracles,
        float_tolerance,
        progress,
    )
    .await
}

async fn tpcds_parity_workload_items_with_session_configs_and_external_oracles(
    root: &Path,
    cpu_session_config: SessionConfig,
    native_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
    external_oracles: Option<TpcdsExternalOracleConfig>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    let runner = TpcdsParityRunner::new(root, cpu_session_config, native_session_config, optimizer_config).await?;
    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
        if !is_query_selected(query.id, query_filter) {
            continue;
        }

        items.push(runner.run_query(root, query, external_oracles.as_ref()).await);
    }
    Ok(items)
}

async fn tpcds_parity_report_rows_with_session_configs_and_external_oracles(
    root: &Path,
    cpu_session_config: SessionConfig,
    native_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
    external_oracles: Option<TpcdsExternalOracleConfig>,
    float_tolerance: ParityFloatTolerance,
    mut progress: Option<&mut dyn FnMut(TpcdsParityProgressEvent<'_>)>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityRow>> {
    let runner = TpcdsParityRunner::new(root, cpu_session_config, native_session_config, optimizer_config).await?;
    let selected: Vec<_> = TPCDS_QUERIES.iter().filter(|query| is_query_selected(query.id, query_filter)).collect();
    let total = selected.len();
    let mut rows = Vec::with_capacity(total);
    for (index, query) in selected.into_iter().enumerate() {
        let query_index = index + 1;
        if let Some(cb) = progress.as_deref_mut() {
            cb(TpcdsParityProgressEvent::QueryStarted { query_index, total, query_id: query.id });
        }

        let item = runner.run_query(root, query, external_oracles.as_ref()).await;
        let row = NativeTpcdsParityRow::from_workload_item(item, float_tolerance);
        if let Some(cb) = progress.as_deref_mut() {
            cb(TpcdsParityProgressEvent::RowReady { query_index, total, query_id: query.id, row: &row });
        }
        rows.push(row);
    }
    Ok(rows)
}

struct TpcdsParityRunner {
    cpu_ctx: SessionContext,
    cpu_stats_fallback_ctx: Option<SessionContext>,
    native_ctx: SessionContext,
    native_rule: Arc<CudfNativePhysicalOptimizerRule>,
}

impl TpcdsParityRunner {
    async fn new(
        root: &Path,
        cpu_session_config: SessionConfig,
        native_session_config: SessionConfig,
        optimizer_config: CudfOptimizerConfig,
    ) -> TpcdsWorkloadResult<Self> {
        let cpu_stats_fallback_ctx = if cpu_session_config.options().execution.collect_statistics {
            let mut fallback_config = cpu_session_config.clone();
            fallback_config.options_mut().execution.collect_statistics = false;
            let fallback_ctx = SessionContext::new_with_config(fallback_config);
            register_tpcds_tables_with_native_metadata(&fallback_ctx, root).await?;
            Some(fallback_ctx)
        } else {
            None
        };
        let cpu_ctx = SessionContext::new_with_config(cpu_session_config);
        register_tpcds_tables_with_native_metadata(&cpu_ctx, root).await?;

        let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
        let native_state = SessionStateBuilder::new_with_default_features()
            .with_config(native_tpcds_session_config(native_session_config))
            .with_cudf_native_rule(Arc::clone(&native_rule))
            .build();
        let native_ctx = SessionContext::new_with_state(native_state);
        register_tpcds_tables_with_native_metadata(&native_ctx, root).await?;

        Ok(Self { cpu_ctx, cpu_stats_fallback_ctx, native_ctx, native_rule })
    }

    async fn run_query(
        &self,
        root: &Path,
        query: &TpcdsQuery,
        external_oracles: Option<&TpcdsExternalOracleConfig>,
    ) -> NativeTpcdsParityWorkloadItem {
        let cpu_outcome = run_cpu_query(&self.cpu_ctx, self.cpu_stats_fallback_ctx.as_ref(), query.sql).await;
        let (native_planning, native_runtime) = run_native_query(&self.native_ctx, &self.native_rule, query.sql).await;
        let duckdb_output = external_oracles
            .and_then(|config| config.duckdb.as_ref().map(|engine| (config, engine)))
            .map_or(NativeTpcdsParityExternalOutput::NotConfigured, |(config, engine)| {
                run_external_oracle(root, query.id, query.sql, "duckdb", engine, &config.scratch_dir)
            });
        NativeTpcdsParityWorkloadItem {
            query_id: query.id.to_string(),
            sql: query.sql.to_string(),
            cpu_outcome,
            native_planning,
            native_runtime,
            duckdb_output,
        }
    }
}

fn is_query_selected(query_id: &str, filter: Option<&[String]>) -> bool {
    match filter {
        Some(allowed) => allowed.iter().any(|allowed_id| allowed_id == query_id),
        None => true,
    }
}

async fn run_cpu_query(
    ctx: &SessionContext,
    stats_fallback_ctx: Option<&SessionContext>,
    sql: &str,
) -> NativeTpcdsParityCpuOutcome {
    let primary = run_cpu_query_once(ctx, sql).await;
    let fallback = match (&primary, stats_fallback_ctx) {
        (CpuQueryAttempt::PhysicalPlanningPanicked(_), Some(fallback_ctx)) => {
            Some(run_cpu_query_once(fallback_ctx, sql).await)
        },
        _ => None,
    };
    cpu_query_outcome_from_attempts(primary, fallback)
}

fn cpu_query_outcome_from_attempts(
    primary: CpuQueryAttempt,
    stats_fallback: Option<CpuQueryAttempt>,
) -> NativeTpcdsParityCpuOutcome {
    match primary {
        CpuQueryAttempt::Outcome(outcome) => outcome,
        CpuQueryAttempt::PhysicalPlanningPanicked(message) => match stats_fallback {
            Some(CpuQueryAttempt::Outcome(outcome)) => outcome,
            Some(CpuQueryAttempt::PhysicalPlanningPanicked(fallback_message)) => physical_planning_panic_failure(
                format!("initial physical planning panicked: {message}; stats-off retry panicked: {fallback_message}"),
            ),
            None => physical_planning_panic_failure(message),
        },
    }
}

enum CpuQueryAttempt {
    Outcome(NativeTpcdsParityCpuOutcome),
    PhysicalPlanningPanicked(String),
}

async fn run_cpu_query_once(ctx: &SessionContext, sql: &str) -> CpuQueryAttempt {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: error.to_string(),
            }));
        },
    };
    let plan = match AssertUnwindSafe(df.create_physical_plan()).catch_unwind().await {
        Ok(Ok(plan)) => plan,
        Ok(Err(error)) => {
            return CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_failed".to_string(),
                message: error.to_string(),
            }));
        },
        Err(payload) => return CpuQueryAttempt::PhysicalPlanningPanicked(panic_payload_to_string(payload.as_ref())),
    };
    let schema = plan.schema();
    match collect(plan, Arc::new(TaskContext::default())).await {
        Ok(batches) => CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::Succeeded { schema, batches }),
        Err(error) => {
            CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::ExecutionFailed { message: error.to_string() })
        },
    }
}

fn physical_planning_panic_failure(message: String) -> NativeTpcdsParityCpuOutcome {
    NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
        stage: "physical_planning".to_string(),
        reason_code: "physical_planning_panicked".to_string(),
        message,
    })
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "unknown panic payload".to_string()
    }
}

async fn run_native_query(
    ctx: &SessionContext,
    native_rule: &Arc<CudfNativePhysicalOptimizerRule>,
    sql: &str,
) -> (NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome) {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return (
                NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "sql_to_logical_plan".to_string(),
                    reason_code: "sql_planning_failed".to_string(),
                    message: error.to_string(),
                }),
                NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
            );
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            return (
                NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "physical_planning".to_string(),
                    reason_code: "physical_planning_failed".to_string(),
                    message: error.to_string(),
                }),
                NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
            );
        },
    };
    let Some(diagnostics) = native_rule.last_planning_diagnostics() else {
        return (
            NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                stage: "native_diagnostics".to_string(),
                reason_code: "missing_native_diagnostics".to_string(),
                message: "native optimizer rule did not retain planning diagnostics".to_string(),
            }),
            NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        );
    };
    let accepted = diagnostics.report.rewritten_count() > 0;
    let planning_outcome = NativeTpcdsParityNativePlanningOutcome::Planned {
        accepted,
        planned_plan: Arc::clone(&plan),
        diagnostics: Box::new(diagnostics),
    };
    let runtime_outcome =
        if accepted { execute_native(plan).await } else { NativeTpcdsParityNativeRuntimeOutcome::NotStarted };
    (planning_outcome, runtime_outcome)
}

async fn execute_native(plan: Arc<dyn ExecutionPlan>) -> NativeTpcdsParityNativeRuntimeOutcome {
    let schema = plan.schema();
    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => {
            let metrics = plan.metrics().unwrap_or_default();
            NativeTpcdsParityNativeRuntimeOutcome::Succeeded { schema, batches, metrics }
        },
        Err(error) => NativeTpcdsParityNativeRuntimeOutcome::Failed { message: error.to_string() },
    }
}

fn run_external_oracle(
    root: &Path,
    query_id: &str,
    sql: &str,
    engine_name: &str,
    engine: &TpcdsExternalEngineConfig,
    scratch_dir: &Path,
) -> NativeTpcdsParityExternalOutput {
    if !engine.binary.is_file() {
        return NativeTpcdsParityExternalOutput::Failed {
            message: format!("{engine_name} binary not found: {}", engine.binary.display()),
        };
    }
    let engine_dir = scratch_dir.join(engine_name);
    if let Err(error) = fs::create_dir_all(&engine_dir) {
        return NativeTpcdsParityExternalOutput::Failed {
            message: format!("failed to create {engine_name} scratch dir {}: {error}", engine_dir.display()),
        };
    }
    let output_path = engine_dir.join(format!("{query_id}.parquet"));
    let sql_path = engine_dir.join(format!("{query_id}.sql"));
    let command_sql = match external_oracle_sql(root, sql, &output_path) {
        Ok(sql) => sql,
        Err(error) => return NativeTpcdsParityExternalOutput::Failed { message: error.to_string() },
    };
    if let Err(error) = fs::write(&sql_path, command_sql) {
        return NativeTpcdsParityExternalOutput::Failed {
            message: format!("failed to write {engine_name} SQL file {}: {error}", sql_path.display()),
        };
    }
    let stdin_file = match fs::File::open(&sql_path) {
        Ok(file) => file,
        Err(error) => {
            return NativeTpcdsParityExternalOutput::Failed {
                message: format!("failed to open {engine_name} SQL file {}: {error}", sql_path.display()),
            };
        },
    };
    let mut command = Command::new(&engine.binary);
    command.arg(":memory:").stdin(Stdio::from(stdin_file));
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => {
            return NativeTpcdsParityExternalOutput::Failed {
                message: format!("failed to run {engine_name} binary {}: {error}", engine.binary.display()),
            };
        },
    };
    if !output.status.success() {
        return NativeTpcdsParityExternalOutput::Failed {
            message: format!(
                "{engine_name} exited with status {}; stderr={}",
                output.status,
                stable_command_output(&output.stderr)
            ),
        };
    }
    read_external_oracle_parquet(&output_path).unwrap_or_else(|message| NativeTpcdsParityExternalOutput::Failed {
        message: format!("{engine_name} output read failed: {message}"),
    })
}

fn external_oracle_sql(root: &Path, query_sql: &str, output_path: &Path) -> TpcdsWorkloadResult<String> {
    let mut sql = String::new();
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        let files = layout
            .files
            .iter()
            .map(|path| format!("'{}'", escape_sql_string(path.to_string_lossy().as_ref())))
            .collect::<Vec<_>>()
            .join(",");
        sql.push_str(&format!("CREATE VIEW {table} AS SELECT * FROM read_parquet([{files}]);\n"));
    }
    sql.push_str(&format!(
        "COPY ({}) TO '{}' (FORMAT PARQUET);\n",
        query_sql.trim().trim_end_matches(';'),
        escape_sql_string(output_path.to_string_lossy().as_ref())
    ));
    Ok(sql)
}

fn read_external_oracle_parquet(path: &Path) -> Result<NativeTpcdsParityExternalOutput, String> {
    let file = fs::File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| error.to_string())?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(|error| error.to_string())?;
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
    Ok(NativeTpcdsParityExternalOutput::Succeeded { schema, batches })
}

fn escape_sql_string(value: &str) -> String {
    value.replace('\'', "''")
}

fn stable_command_output(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    text.lines().take(20).collect::<Vec<_>>().join("\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_query_physical_planning_panic_uses_stats_fallback_success() {
        let outcome = cpu_query_outcome_from_attempts(
            CpuQueryAttempt::PhysicalPlanningPanicked("attempt to multiply with overflow".to_string()),
            Some(CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::ExecutionFailed {
                message: "fallback execution reached".to_string(),
            })),
        );

        match outcome {
            NativeTpcdsParityCpuOutcome::ExecutionFailed { message } => {
                assert_eq!(message, "fallback execution reached");
            },
            other => panic!("expected fallback outcome, got {other:?}"),
        }
    }

    #[test]
    fn cpu_query_physical_planning_panic_reports_both_failed_attempts() {
        let outcome = cpu_query_outcome_from_attempts(
            CpuQueryAttempt::PhysicalPlanningPanicked("primary overflow".to_string()),
            Some(CpuQueryAttempt::PhysicalPlanningPanicked("fallback overflow".to_string())),
        );

        match outcome {
            NativeTpcdsParityCpuOutcome::PlanningFailed(failure) => {
                assert_eq!(failure.stage, "physical_planning");
                assert_eq!(failure.reason_code, "physical_planning_panicked");
                assert!(failure.message.contains("primary overflow"));
                assert!(failure.message.contains("fallback overflow"));
            },
            other => panic!("expected planning failure, got {other:?}"),
        }
    }
}
