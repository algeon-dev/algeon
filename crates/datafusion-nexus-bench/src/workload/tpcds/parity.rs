use std::{
    fs,
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
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, NativePlanningFailure,
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput, NativeTpcdsParityNativePlanningOutcome,
    NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityWorkloadItem,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{
    TPCDS_QUERIES, TPCDS_TABLES, TpcdsWorkloadResult, discover_tpcds_table_layout, native_tpcds_session_config,
    register_tpcds_tables,
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

async fn tpcds_parity_workload_items_with_session_configs_and_external_oracles(
    root: &Path,
    cpu_session_config: SessionConfig,
    native_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
    external_oracles: Option<TpcdsExternalOracleConfig>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    let cpu_ctx = SessionContext::new_with_config(cpu_session_config);
    register_tpcds_tables(&cpu_ctx, root).await?;

    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let native_state = SessionStateBuilder::new_with_default_features()
        .with_config(native_tpcds_session_config(native_session_config))
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let native_ctx = SessionContext::new_with_state(native_state);
    register_tpcds_tables(&native_ctx, root).await?;

    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
        if !is_query_selected(query.id, query_filter) {
            continue;
        }

        let cpu_outcome = run_cpu_query(&cpu_ctx, query.sql).await;
        let (native_planning, native_runtime) = run_native_query(&native_ctx, &native_rule, query.sql).await;
        let duckdb_output = external_oracles
            .as_ref()
            .and_then(|config| config.duckdb.as_ref().map(|engine| (config, engine)))
            .map_or(NativeTpcdsParityExternalOutput::NotConfigured, |(config, engine)| {
                run_external_oracle(root, query.id, query.sql, "duckdb", engine, &config.scratch_dir)
            });
        items.push(NativeTpcdsParityWorkloadItem {
            query_id: query.id.to_string(),
            sql: query.sql.to_string(),
            cpu_outcome,
            native_planning,
            native_runtime,
            duckdb_output,
        });
    }
    Ok(items)
}

fn is_query_selected(query_id: &str, filter: Option<&[String]>) -> bool {
    match filter {
        Some(allowed) => allowed.iter().any(|allowed_id| allowed_id == query_id),
        None => true,
    }
}

async fn run_cpu_query(ctx: &SessionContext, sql: &str) -> NativeTpcdsParityCpuOutcome {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: error.to_string(),
            });
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            return NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_failed".to_string(),
                message: error.to_string(),
            });
        },
    };
    let schema = plan.schema();
    match collect(plan, Arc::new(TaskContext::default())).await {
        Ok(batches) => NativeTpcdsParityCpuOutcome::Succeeded { schema, batches },
        Err(error) => NativeTpcdsParityCpuOutcome::ExecutionFailed { message: error.to_string() },
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
