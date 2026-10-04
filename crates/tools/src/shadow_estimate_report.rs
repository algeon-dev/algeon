//! Explicit estimate-vs-observation shadow report (not a Cargo test target).
//!
//! For each selected TPC-H/TPC-DS query this tool plans through the normal
//! native optimizer, computes a pre-execution shadow memory estimate over the
//! compiled bounded pipeline, and then (unless `--no-execute`) runs the same
//! query for real through the public backend + attempt API. It joins the two
//! views into one comparison record per query. This tool never feeds its
//! estimate back into planning, admission, grants, or execution: it only
//! reads already-public planning/admission/shadow/attempt surfaces.

use std::{
    collections::HashMap,
    error::Error,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    attempt::{AttemptOptions, QueryAttemptReport},
    backend::AlgeonGpuBackend,
    native_report::{NativePlanTaskMappingMetricRow, NativeTaskRuntimeMetricRow},
    planner::AlgeonNativeOptimizerConfig,
    session::AlgeonSessionStateBuilderExt,
};
use bench::report_workload::{
    RelationalExecutionSelection,
    tpcds::{
        register_tpcds_tables_with_native_metadata, tpcds_native_analytics_optimizer_config_with_profile_settings,
        tpcds_native_profile_settings_and_device_memory_limit_from_env, tpcds_native_report_session_config,
    },
    tpch::{TPCH_NATIVE_DEVICE_BUDGET_BYTES, tpch_native_optimizer_config, tpch_report_session_config},
};
use datafusion::{
    execution::SessionStateBuilder,
    physical_plan::ExecutionPlan,
    prelude::{SessionConfig, SessionContext},
};
use futures::TryStreamExt;
use query_engine::shadow::{estimate_compiled_pipeline, frontend_output_records};
use query_planning::{
    pipeline::{ShadowBytes, ShadowPipelineEstimate, ShadowTaskEnvelope},
    plan::QueryPlan,
};
use query_runtime::{
    observability::AttemptFinalDomainUsage,
    runtime::{
        GpuBackendDeviceConfig, GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
};
use serde::Serialize;
use serde_json::Value;
use workloads::{
    tpcds::{TPCDS_QUERIES, TpcdsDatasetMetadata, prepare_tpcds_report_dataset},
    tpch::{TPCH_QUERIES, TpchDatasetMetadata, prepare_tpch_report_dataset, register_tpch_tables},
};

use crate::native_plan_peel::peel_native_admitted_plan;

type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const MANIFEST_SCHEMA: &str = "algeon_datafusion.shadow_estimate_report.manifest";
const COMPARE_RECORD_SCHEMA: &str = "algeon_datafusion.shadow_estimate_report.compare";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Workload {
    Tpch,
    Tpcds,
}

impl Workload {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Tpch => "tpch",
            Self::Tpcds => "tpcds",
        }
    }
}

#[derive(Debug, Clone)]
struct ReportConfig {
    workload: Workload,
    query_ids: Vec<String>,
    artifact_dir: PathBuf,
    no_execute: bool,
}

pub async fn run(args: Vec<String>) -> ToolResult<()> {
    if args.iter().any(|arg| matches!(arg.as_str(), "-h" | "--help")) {
        print_usage();
        return Ok(());
    }
    let config = parse_args(args)?;
    fs::create_dir_all(&config.artifact_dir)?;
    let queries = resolve_workload_queries(config.workload, &config.query_ids)?;
    let tiny_root = config.artifact_dir.join("tiny-fixture");

    let (root, dataset_json, base_optimizer_config, device_memory_limit, device_budget_bytes) = match config.workload {
        Workload::Tpch => {
            let (root, metadata) = prepare_tpch_report_dataset(&tiny_root)?;
            let optimizer_config = RelationalExecutionSelection::strict_native()
                .configure_native_optimizer(tpch_native_optimizer_config())
                .ok_or("strict-native selection must install the native optimizer rule")?;
            (
                root,
                tpch_dataset_metadata_json(&metadata),
                optimizer_config,
                GpuDeviceMemoryLimit::Explicit { limit_bytes: TPCH_NATIVE_DEVICE_BUDGET_BYTES },
                TPCH_NATIVE_DEVICE_BUDGET_BYTES,
            )
        },
        Workload::Tpcds => {
            let (root, metadata) = prepare_tpcds_report_dataset(&tiny_root)?;
            let (profile, device_memory_limit) = tpcds_native_profile_settings_and_device_memory_limit_from_env();
            let optimizer_config = RelationalExecutionSelection::strict_native()
                .configure_native_optimizer(tpcds_native_analytics_optimizer_config_with_profile_settings(&profile))
                .ok_or("strict-native selection must install the native optimizer rule")?;
            let device_budget_bytes = profile.device_budget_bytes;
            (root, tpcds_dataset_metadata_json(&metadata), optimizer_config, device_memory_limit, device_budget_bytes)
        },
    };
    let session_config = match config.workload {
        Workload::Tpch => tpch_report_session_config(),
        Workload::Tpcds => tpcds_native_report_session_config(),
    };

    let manifest = build_manifest(&config, &queries, dataset_json, &base_optimizer_config, device_budget_bytes);
    fs::write(config.artifact_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;

    let attempt_backend = if config.no_execute {
        None
    } else {
        Some(build_attempt_backend(base_optimizer_config.clone(), device_memory_limit)?)
    };
    let query_context = QueryProcessingContext {
        workload: config.workload,
        root: &root,
        session_config,
        optimizer_config: &base_optimizer_config,
        attempt_backend: attempt_backend.as_ref(),
    };

    let compare_path = config.artifact_dir.join("shadow_estimate_compare.jsonl");
    let mut compare_file = fs::File::create(&compare_path)?;
    for (query_id, sql) in &queries {
        let query_dir = config.artifact_dir.join(query_id);
        let record = match process_query(&query_context, query_id, sql, &query_dir).await {
            Ok(record) => record,
            Err(error) => CompareRecord::error(config.workload, query_id, error.to_string()),
        };
        eprintln!(
            "[shadow-estimate-report] query={} pipeline={} observation={} join_mismatches={}",
            record.query_id,
            record.pipeline.status_label(),
            record.observation.status_label(),
            record.join_mismatches.len(),
        );
        serde_json::to_writer(&mut compare_file, &record)?;
        compare_file.write_all(b"\n")?;
    }
    eprintln!("[shadow-estimate-report] wrote {}", compare_path.display());
    Ok(())
}

fn resolve_workload_queries(workload: Workload, query_ids: &[String]) -> ToolResult<Vec<(String, &'static str)>> {
    let mut resolved = Vec::with_capacity(query_ids.len());
    for id in query_ids {
        let sql = match workload {
            Workload::Tpch => TPCH_QUERIES.iter().find(|query| query.id == id).map(|query| query.sql),
            Workload::Tpcds => TPCDS_QUERIES.iter().find(|query| query.id == id).map(|query| query.sql),
        };
        let sql = sql.ok_or_else(|| format!("unknown {} query id: {id}", workload.as_str()))?;
        resolved.push((id.clone(), sql));
    }
    Ok(resolved)
}

fn build_attempt_backend(
    optimizer_config: AlgeonNativeOptimizerConfig,
    device_memory_limit: GpuDeviceMemoryLimit,
) -> ToolResult<AlgeonGpuBackend> {
    let device = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(0)
            .with_memory_limit(device_memory_limit)
            .with_max_active_attempts(1)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(2, 2, 1)),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    Ok(AlgeonGpuBackend::builder().with_optimizer_config(optimizer_config).devices([device]).build()?)
}

struct QueryProcessingContext<'a> {
    workload: Workload,
    root: &'a Path,
    session_config: SessionConfig,
    optimizer_config: &'a AlgeonNativeOptimizerConfig,
    attempt_backend: Option<&'a AlgeonGpuBackend>,
}

async fn process_query(
    ctx: &QueryProcessingContext<'_>,
    query_id: &str,
    sql: &str,
    query_dir: &Path,
) -> ToolResult<CompareRecord> {
    // Groups (a) + (b): plan without a live backend so this half works
    // without a GPU (see `--no-execute`).
    let estimate_execution_config = ctx.optimizer_config.native_execution_config().clone().with_query_id(query_id);
    let estimate_optimizer_config =
        ctx.optimizer_config.clone().with_native_execution_config(estimate_execution_config);
    let physical = plan_workload_physical(
        ctx.workload,
        query_id,
        ctx.root,
        sql,
        ctx.session_config.clone(),
        estimate_optimizer_config,
    )
    .await?;
    let Some(admitted) = peel_native_admitted_plan(physical.as_ref()) else {
        return Ok(CompareRecord::error(
            ctx.workload,
            query_id,
            "physical plan did not lower to native execution (CPU fallback)".to_string(),
        ));
    };
    // The node's artifact is the plan in the deferred host sink form it
    // executes, admitted under the estimate config's mode and policy.
    let gpu_plan = admitted.plan();
    let execution_scope = format!("blake3:{}", gpu_plan.execution_fingerprint());
    let frontend_outputs = frontend_output_records(gpu_plan);
    let frontend_coverage = frontend_output_coverage(gpu_plan);

    let estimate = estimate_compiled_pipeline(gpu_plan, admitted.pipeline())?;
    let pipeline = PipelineGroup::Present {
        task_workspace_coverage: shadow_bytes_coverage(estimate.pipeline.task_workspace.iter()),
        resource_coverage: shadow_bytes_coverage(
            estimate.pipeline.resources.iter().map(|resource| &resource.model.bytes),
        ),
        envelope_coverage: shadow_bytes_coverage(
            estimate.pipeline.envelopes.iter().map(|envelope| &envelope.query_owned_bytes),
        ),
        collection_elapsed_ns: estimate.collection_elapsed_ns,
        estimate: Box::new(estimate.pipeline),
    };

    // Group (c): real execution through the public backend + attempt API,
    // independent of whether a compiled schedule exists for group (b).
    let (observation, sealed_report) = match ctx.attempt_backend {
        None => (ObservationGroup::SkippedNoExecute, None),
        Some(backend) => {
            let (observation, report) =
                execute_for_real(ctx.workload, query_id, ctx.root, sql, ctx.session_config.clone(), backend).await?;
            (observation, Some(report))
        },
    };

    let (task_deltas, join_mismatches, query_peak_delta) = match (&pipeline, &observation) {
        (
            PipelineGroup::Present { estimate, .. },
            ObservationGroup::Present { task_runtime_rows, final_domain_usage, .. },
        ) => {
            let (deltas, mismatches) = join_task_runtime_rows(&estimate.envelopes, task_runtime_rows);
            let observed_peak = final_domain_usage.as_ref().map(|usage| usage.peak_bytes);
            let (signed_delta_bytes, classification) = classify_delta(estimate.query_owned_peak.value(), observed_peak);
            (
                deltas,
                mismatches,
                Some(PeakDelta {
                    estimate_query_owned_peak_bytes: estimate.query_owned_peak.clone(),
                    observed_peak_bytes: observed_peak,
                    signed_delta_bytes,
                    classification,
                }),
            )
        },
        _ => (Vec::new(), Vec::new(), None),
    };

    if let Some(report) = &sealed_report {
        fs::create_dir_all(query_dir)?;
        fs::write(query_dir.join("attempt_report.json"), serde_json::to_vec_pretty(report.as_ref())?)?;
    }

    let collection_elapsed_ns = match &pipeline {
        PipelineGroup::Present { collection_elapsed_ns, .. } => Some(*collection_elapsed_ns),
        PipelineGroup::Absent { .. } => None,
    };
    let (oom_requested_bytes, cleanup_disposition) = match &observation {
        ObservationGroup::Present { allocator_event_requested_bytes, cleanup_disposition, .. } => {
            (*allocator_event_requested_bytes, Some(cleanup_disposition.clone()))
        },
        _ => (None, None),
    };

    Ok(CompareRecord {
        schema: COMPARE_RECORD_SCHEMA,
        workload: ctx.workload.as_str(),
        query_id: query_id.to_string(),
        execution_scope,
        frontend_outputs,
        frontend_coverage,
        pipeline,
        observation,
        task_deltas,
        query_peak_delta,
        join_mismatches,
        collection_elapsed_ns,
        oom_requested_bytes,
        cleanup_disposition,
        error: None,
    })
}

async fn plan_workload_physical(
    workload: Workload,
    query_id: &str,
    root: &Path,
    sql: &str,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> ToolResult<Arc<dyn ExecutionPlan>> {
    let state = SessionStateBuilder::new()
        .with_config(session_config)
        .with_default_features()
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_workload_tables(workload, &ctx, root).await?;
    let framed = format!("-- query_id={query_id}\n{sql}");
    let df = ctx.sql(&framed).await?;
    Ok(df.create_physical_plan().await?)
}

/// Plans through the given backend's own installed native rule so the
/// resulting physical plan's admission ticket donor is bound to that
/// backend's device ledger, matching whatever backend begins the executing
/// attempt (mirrors `clickbench_stage_probe::plan_native_physical_for_attempt`).
async fn plan_workload_physical_for_attempt(
    workload: Workload,
    query_id: &str,
    root: &Path,
    sql: &str,
    session_config: SessionConfig,
    backend: &AlgeonGpuBackend,
) -> ToolResult<(SessionContext, Arc<dyn ExecutionPlan>)> {
    let builder = backend.install_on(SessionStateBuilder::new().with_config(session_config).with_default_features())?;
    let ctx = SessionContext::new_with_state(builder.build());
    register_workload_tables(workload, &ctx, root).await?;
    let framed = format!("-- query_id={query_id}\n{sql}");
    let df = ctx.sql(&framed).await?;
    let physical = df.create_physical_plan().await?;
    Ok((ctx, physical))
}

async fn register_workload_tables(workload: Workload, ctx: &SessionContext, root: &Path) -> ToolResult<()> {
    match workload {
        Workload::Tpch => register_tpch_tables(ctx, root).await?,
        Workload::Tpcds => register_tpcds_tables_with_native_metadata(ctx, root).await?,
    }
    Ok(())
}

async fn execute_for_real(
    workload: Workload,
    query_id: &str,
    root: &Path,
    sql: &str,
    session_config: SessionConfig,
    backend: &AlgeonGpuBackend,
) -> ToolResult<(ObservationGroup, Arc<QueryAttemptReport>)> {
    let (ctx, physical) =
        plan_workload_physical_for_attempt(workload, query_id, root, sql, session_config, backend).await?;
    let attempt =
        backend.begin_attempt(AttemptOptions { query_id: Some(query_id.to_string()), ..AttemptOptions::default() })?;
    // Obtained before execution starts so it observes every terminal outcome,
    // per `AlgeonAttempt::completion`'s documented contract.
    let completion = attempt.completion();
    let stream_result = match attempt.execute_physical_plan(physical, &ctx) {
        Ok(stream) => stream
            .try_collect::<Vec<_>>()
            .await
            .map(|batches| batches.iter().map(|batch| batch.num_rows()).sum::<usize>())
            .map_err(|error| error.to_string()),
        Err(error) => Err(error.to_string()),
    };
    // The attempt seals (and publishes its native execution snapshot) as part
    // of the same terminal commit that resolves `completion`, regardless of
    // stream success or failure; awaiting it first is the documented,
    // race-free way to read either afterward.
    let report = completion.await;
    let native_snapshot = attempt.sealed_native_execution_snapshot();

    let observation = match (stream_result, native_snapshot) {
        (Ok(output_rows), Some(snapshot)) => ObservationGroup::Present {
            output_rows: Some(output_rows as u64),
            task_runtime_rows: snapshot.metrics.task_runtime_rows.iter().map(ObservedTaskRuntimeRow::from).collect(),
            plan_task_mapping_rows: snapshot
                .metrics
                .plan_task_mapping_rows
                .iter()
                .map(ObservedPlanTaskMappingRow::from)
                .collect(),
            final_domain_usage: report.final_domain_usage,
            allocator_event_requested_bytes: report
                .resource_evidence
                .as_ref()
                .and_then(|evidence| evidence.allocator_event.value.as_ref())
                .map(|event| event.requested_bytes),
            cleanup_disposition: format!("{:?}", report.cleanup_disposition),
        },
        (Ok(_), None) => {
            ObservationGroup::Error { message: "attempt sealed without a native execution snapshot".to_string() }
        },
        (Err(message), _) => ObservationGroup::Error { message },
    };
    Ok((observation, report))
}

// ---------------------------------------------------------------------------
// Comparison record shape
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct CompareRecord {
    schema: &'static str,
    workload: &'static str,
    query_id: String,
    execution_scope: String,
    frontend_outputs: Vec<Value>,
    frontend_coverage: Coverage,
    pipeline: PipelineGroup,
    observation: ObservationGroup,
    task_deltas: Vec<TaskDelta>,
    query_peak_delta: Option<PeakDelta>,
    join_mismatches: Vec<JoinMismatch>,
    /// Estimate collection overhead; `None` when group (b) is absent.
    collection_elapsed_ns: Option<u64>,
    oom_requested_bytes: Option<u64>,
    cleanup_disposition: Option<String>,
    /// Set only when this query could not be planned, admitted, or estimated
    /// at all; the query still appears in the JSONL rather than being dropped.
    error: Option<String>,
}

impl CompareRecord {
    fn error(workload: Workload, query_id: &str, message: String) -> Self {
        Self {
            schema: COMPARE_RECORD_SCHEMA,
            workload: workload.as_str(),
            query_id: query_id.to_string(),
            execution_scope: String::new(),
            frontend_outputs: Vec::new(),
            frontend_coverage: Coverage::default(),
            pipeline: PipelineGroup::Absent { reason: "query_processing_error" },
            observation: ObservationGroup::Error { message: message.clone() },
            task_deltas: Vec::new(),
            query_peak_delta: None,
            join_mismatches: Vec::new(),
            collection_elapsed_ns: None,
            oom_requested_bytes: None,
            cleanup_disposition: None,
            error: Some(message),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
struct Coverage {
    total: usize,
    known: usize,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum PipelineGroup {
    Present {
        estimate: Box<ShadowPipelineEstimate>,
        task_workspace_coverage: Coverage,
        resource_coverage: Coverage,
        envelope_coverage: Coverage,
        collection_elapsed_ns: u64,
    },
    /// No compiled bounded-pipeline schedule exists for this plan.
    /// `reason` is `"query_processing_error"` for a planning failure; a plan
    /// whose candidate planning rejected has no native node and is recorded
    /// as a CPU fallback before this group is built.
    Absent { reason: &'static str },
}

impl PipelineGroup {
    const fn status_label(&self) -> &'static str {
        match self {
            Self::Present { .. } => "present",
            Self::Absent { .. } => "absent",
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ObservationGroup {
    SkippedNoExecute,
    Error {
        message: String,
    },
    Present {
        output_rows: Option<u64>,
        task_runtime_rows: Vec<ObservedTaskRuntimeRow>,
        plan_task_mapping_rows: Vec<ObservedPlanTaskMappingRow>,
        final_domain_usage: Option<AttemptFinalDomainUsage>,
        allocator_event_requested_bytes: Option<u64>,
        cleanup_disposition: String,
    },
}

impl ObservationGroup {
    const fn status_label(&self) -> &'static str {
        match self {
            Self::SkippedNoExecute => "skipped_no_execute",
            Self::Error { .. } => "error",
            Self::Present { .. } => "present",
        }
    }
}

/// Local `Serialize` mirror of `NativeTaskRuntimeMetricRow`, which does not
/// derive `Serialize` itself.
#[derive(Debug, Clone, Serialize)]
struct ObservedTaskRuntimeRow {
    compiled_task_id: String,
    operator_kind: String,
    elapsed_ns: usize,
    max_observed_peak_bytes: Option<usize>,
    max_retained_bytes: Option<usize>,
}

impl From<&NativeTaskRuntimeMetricRow> for ObservedTaskRuntimeRow {
    fn from(row: &NativeTaskRuntimeMetricRow) -> Self {
        Self {
            compiled_task_id: row.compiled_task_id.clone(),
            operator_kind: row.operator_kind.clone(),
            elapsed_ns: row.elapsed_ns,
            max_observed_peak_bytes: row.max_observed_peak_bytes,
            max_retained_bytes: row.max_retained_bytes,
        }
    }
}

/// Local `Serialize` mirror of `NativePlanTaskMappingMetricRow`, which does
/// not derive `Serialize` itself.
#[derive(Debug, Clone, Serialize)]
struct ObservedPlanTaskMappingRow {
    query_id: String,
    plan_node_id: Option<String>,
    compiled_task_id: Option<String>,
    operator_kind: Option<String>,
    datafusion_shape: Option<String>,
}

impl From<&NativePlanTaskMappingMetricRow> for ObservedPlanTaskMappingRow {
    fn from(row: &NativePlanTaskMappingMetricRow) -> Self {
        Self {
            query_id: row.query_id.clone(),
            plan_node_id: row.plan_node_id.clone(),
            compiled_task_id: row.compiled_task_id.clone(),
            operator_kind: row.operator_kind.clone(),
            datafusion_shape: row.datafusion_shape.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DeltaClassification {
    /// The estimate under-predicted: observed bytes exceeded the estimate.
    Under,
    /// The estimate over-predicted: observed bytes fell short of the estimate.
    Over,
    Exact,
    /// Either side lacked a known value; never coerced to zero.
    Unknown,
}

#[derive(Debug, Serialize)]
struct TaskDelta {
    task: usize,
    compiled_task_id: String,
    estimate_query_owned_bytes: ShadowBytes,
    observed_max_observed_peak_bytes: Option<usize>,
    observed_max_retained_bytes: Option<usize>,
    signed_delta_bytes: Option<i64>,
    classification: DeltaClassification,
}

#[derive(Debug, Serialize)]
struct PeakDelta {
    estimate_query_owned_peak_bytes: ShadowBytes,
    observed_peak_bytes: Option<u64>,
    signed_delta_bytes: Option<i64>,
    classification: DeltaClassification,
}

#[derive(Debug, Serialize)]
struct JoinMismatch {
    compiled_task_id: String,
    reason: &'static str,
    detail: String,
}

/// Computes `signed = observed - estimate` and classifies it. `Unknown`
/// whenever either side is absent; an absent estimate never becomes an
/// invented zero.
const fn classify_delta(estimate: Option<u64>, observed: Option<u64>) -> (Option<i64>, DeltaClassification) {
    match (estimate, observed) {
        (Some(estimate), Some(observed)) => {
            let signed = observed as i64 - estimate as i64;
            let classification = if signed > 0 {
                DeltaClassification::Under
            } else if signed < 0 {
                DeltaClassification::Over
            } else {
                DeltaClassification::Exact
            };
            (Some(signed), classification)
        },
        _ => (None, DeltaClassification::Unknown),
    }
}

fn shadow_bytes_coverage<'a>(values: impl Iterator<Item = &'a ShadowBytes>) -> Coverage {
    let mut coverage = Coverage::default();
    for value in values {
        coverage.total += 1;
        if value.value().is_some() {
            coverage.known += 1;
        }
    }
    coverage
}

/// Parses the canonical `t<index>` shape produced by `PipelineTaskId`'s
/// `Display` impl (`crates/algeon-query-planning/src/pipeline/display.rs`).
fn parse_compiled_task_index(compiled_task_id: &str) -> Option<usize> {
    compiled_task_id.strip_prefix('t').and_then(|rest| rest.parse::<usize>().ok())
}

/// Joins compiled-pipeline task envelopes (index space) against observed
/// `compiled_task_id` rows (string space). The two spaces can legitimately
/// diverge; for example when a retry grows the runtime task graph beyond
/// what was compiled for the estimate; so every row that cannot be placed
/// unambiguously is reported as a `JoinMismatch` instead of guessed.
fn join_task_runtime_rows(
    envelopes: &[ShadowTaskEnvelope],
    observed: &[ObservedTaskRuntimeRow],
) -> (Vec<TaskDelta>, Vec<JoinMismatch>) {
    let mut by_index: HashMap<usize, &ObservedTaskRuntimeRow> = HashMap::new();
    let mut mismatches = Vec::new();
    for row in observed {
        match parse_compiled_task_index(&row.compiled_task_id) {
            Some(index) if index < envelopes.len() => {
                if let Some(previous) = by_index.insert(index, row) {
                    mismatches.push(JoinMismatch {
                        compiled_task_id: row.compiled_task_id.clone(),
                        reason: "duplicate_compiled_task_id",
                        detail: format!(
                            "task index {index} already claimed by operator_kind={}",
                            previous.operator_kind
                        ),
                    });
                }
            },
            Some(index) => mismatches.push(JoinMismatch {
                compiled_task_id: row.compiled_task_id.clone(),
                reason: "task_index_out_of_range",
                detail: format!("observed task index {index} exceeds compiled task count {}", envelopes.len()),
            }),
            None => mismatches.push(JoinMismatch {
                compiled_task_id: row.compiled_task_id.clone(),
                reason: "unparseable_compiled_task_id",
                detail: "compiled task id did not match the canonical `t<index>` shape".to_string(),
            }),
        }
    }
    let deltas = envelopes
        .iter()
        .map(|envelope| {
            let observed_row = by_index.get(&envelope.task).copied();
            let estimate_bytes = envelope.query_owned_bytes.value();
            let observed_bytes = observed_row.and_then(|row| row.max_observed_peak_bytes).map(|value| value as u64);
            let (signed_delta_bytes, classification) = classify_delta(estimate_bytes, observed_bytes);
            TaskDelta {
                task: envelope.task,
                compiled_task_id: format!("t{}", envelope.task),
                estimate_query_owned_bytes: envelope.query_owned_bytes.clone(),
                observed_max_observed_peak_bytes: observed_row.and_then(|row| row.max_observed_peak_bytes),
                observed_max_retained_bytes: observed_row.and_then(|row| row.max_retained_bytes),
                signed_delta_bytes,
                classification,
            }
        })
        .collect();
    (deltas, mismatches)
}

fn frontend_output_coverage(plan: &QueryPlan) -> Coverage {
    let outputs = plan.advisory_outputs();
    let known = outputs.iter().filter(|binding| binding.statistics.bytes.value().is_some()).count();
    Coverage { total: outputs.len(), known }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Manifest {
    schema: &'static str,
    workload: &'static str,
    query_ids: Vec<String>,
    no_execute: bool,
    artifact_dir: String,
    dataset: Value,
    capacity: Value,
    git: Value,
    gpu: Value,
    created_unix_ns: u128,
}

fn build_manifest(
    config: &ReportConfig,
    queries: &[(String, &'static str)],
    dataset_json: Value,
    optimizer_config: &AlgeonNativeOptimizerConfig,
    device_budget_bytes: u64,
) -> Manifest {
    let execution_config = optimizer_config.native_execution_config();
    let capacity = serde_json::json!({
        "device_budget_bytes": device_budget_bytes,
        "max_source_chunk_bytes": execution_config.source_chunking().max_source_chunk_bytes.get(),
        "max_row_groups_per_chunk": execution_config.source_chunking().max_row_groups_per_chunk.get(),
        "max_retry_attempts": execution_config.source_chunking().max_retry_attempts.get(),
        "source_chunk_planning_mode": format!("{:?}", execution_config.source_chunk_planning_mode()),
        "statistical_aggregate_execution_mode": format!("{:?}", optimizer_config.statistical_aggregate_execution_mode()),
        // Every attempt in this run uses the backend's configured device
        // ledger cap rather than an explicit per-query request.
        "query_cap_bytes": Value::Null,
    });
    Manifest {
        schema: MANIFEST_SCHEMA,
        workload: config.workload.as_str(),
        query_ids: queries.iter().map(|(id, _)| id.clone()).collect(),
        no_execute: config.no_execute,
        artifact_dir: config.artifact_dir.display().to_string(),
        dataset: dataset_json,
        capacity,
        git: git_revision_json(),
        gpu: gpu_driver_identity_json(),
        created_unix_ns: SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_nanos()).unwrap_or(0),
    }
}

fn tpch_dataset_metadata_json(metadata: &TpchDatasetMetadata) -> Value {
    serde_json::json!({
        "workload": "tpch",
        "root": metadata.root.display().to_string(),
        "scale_factor": metadata.scale_factor,
        "provenance": metadata.provenance,
        "generation_command": metadata.generation_command,
        "table_count": metadata.table_count,
        "file_count": metadata.file_count,
        "total_rows": metadata.total_rows,
        "total_bytes": metadata.total_bytes,
        "metadata_fingerprint": metadata.metadata_fingerprint,
        "content_fingerprint": metadata.content_fingerprint,
    })
}

fn tpcds_dataset_metadata_json(metadata: &TpcdsDatasetMetadata) -> Value {
    serde_json::json!({
        "workload": "tpcds",
        "root": metadata.root.display().to_string(),
        "scale_factor": metadata.scale_factor,
        "provenance": metadata.provenance,
        "generation_command": metadata.generation_command,
        "table_count": metadata.table_count,
        "file_count": metadata.file_count,
        "total_rows": metadata.total_rows,
        "total_bytes": metadata.total_bytes,
        "metadata_fingerprint": metadata.metadata_fingerprint,
        "content_fingerprint": metadata.content_fingerprint,
    })
}

fn git_revision_json() -> Value {
    let sha = command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = command_output("git", &["status", "--porcelain"]).is_some_and(|value| !value.is_empty());
    serde_json::json!({ "sha": sha, "dirty": dirty })
}

/// Best-effort GPU/driver identity via `nvidia-smi`. This tool intentionally
/// does not depend on the `nvml` feature (unlike
/// `single-node-multi-gpu-validate-run`), so identity here is opportunistic
/// and degrades to `"unavailable"` rather than failing the run.
fn gpu_driver_identity_json() -> Value {
    match command_output("nvidia-smi", &["--query-gpu=name,driver_version,uuid", "--format=csv,noheader"]) {
        Some(text) if !text.is_empty() => {
            let devices: Vec<Value> = text
                .lines()
                .map(|line| {
                    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                    serde_json::json!({
                        "name": parts.first().copied().unwrap_or("unknown"),
                        "driver_version": parts.get(1).copied().unwrap_or("unknown"),
                        "uuid": parts.get(2).copied().unwrap_or("unknown"),
                    })
                })
                .collect();
            serde_json::json!({ "source": "nvidia-smi", "devices": devices })
        },
        _ => serde_json::json!({ "source": "unavailable", "devices": [] }),
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

fn parse_args(args: Vec<String>) -> ToolResult<ReportConfig> {
    let mut workload = None;
    let mut query_ids = Vec::new();
    let mut artifact_dir = None;
    let mut no_execute = false;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--workload" => {
                let value = iter.next().ok_or("--workload requires a value")?;
                workload = Some(match value.as_str() {
                    "tpch" => Workload::Tpch,
                    "tpcds" => Workload::Tpcds,
                    other => return Err(format!("unsupported --workload {other}").into()),
                });
            },
            "--query" | "--queries" => {
                let value = iter.next().ok_or("--queries requires a value")?;
                for part in value.split(',') {
                    let part = part.trim();
                    if !part.is_empty() {
                        query_ids.push(part.to_string());
                    }
                }
            },
            "--artifact-dir" => {
                artifact_dir = Some(PathBuf::from(iter.next().ok_or("--artifact-dir requires a value")?));
            },
            "--no-execute" => no_execute = true,
            other => return Err(format!("unknown shadow-estimate-report argument: {other}").into()),
        }
    }

    let workload = workload.ok_or("--workload is required (tpch|tpcds)")?;
    if query_ids.is_empty() {
        query_ids = match workload {
            Workload::Tpch => TPCH_QUERIES.iter().map(|query| query.id.to_string()).collect(),
            Workload::Tpcds => TPCDS_QUERIES.iter().map(|query| query.id.to_string()).collect(),
        };
    }
    let artifact_dir = artifact_dir.unwrap_or_else(default_artifact_dir);
    Ok(ReportConfig { workload, query_ids, artifact_dir, no_execute })
}

fn default_artifact_dir() -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_secs()).unwrap_or(0);
    PathBuf::from(format!("target/shadow-estimate/run_{stamp}"))
}

fn print_usage() {
    eprintln!(
        "shadow-estimate-report\n\
         \n\
         USAGE:\n\
           flock /tmp/cudf-gpu.lock cargo run -p tools --features shadow-estimate-report -- \\\n\
             shadow-estimate-report --workload tpch|tpcds [--queries q01,q06] \\\n\
             [--artifact-dir DIR] [--no-execute]\n\
         \n\
         Produces target/shadow-estimate/<run-id>/manifest.json, \\\n\
         shadow_estimate_compare.jsonl, and a per-query attempt_report.json. \\\n\
         Never influences planning, admission, grants, or execution: it only reads \\\n\
         already-public planning/admission/shadow/attempt surfaces. --no-execute skips \\\n\
         the real attempt (useful without a GPU) and leaves the observation group absent."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_requires_workload() {
        let error = parse_args(vec![]).expect_err("workload is required");
        assert!(error.to_string().contains("--workload"));
    }

    #[test]
    fn parse_args_defaults_to_the_full_catalog() {
        let config = parse_args(vec!["--workload".into(), "tpch".into()]).expect("parse");
        assert_eq!(config.workload, Workload::Tpch);
        assert_eq!(config.query_ids.len(), TPCH_QUERIES.len());
        assert!(!config.no_execute);
    }

    #[test]
    fn parse_args_reads_explicit_queries_and_no_execute() {
        let config = parse_args(vec![
            "--workload".into(),
            "tpcds".into(),
            "--queries".into(),
            "q01,q06".into(),
            "--no-execute".into(),
        ])
        .expect("parse");
        assert_eq!(config.query_ids, ["q01", "q06"]);
        assert!(config.no_execute);
    }

    #[test]
    fn shadow_bytes_coverage_counts_known_values_without_treating_unknown_as_zero() {
        let values =
            [ShadowBytes::estimated(10, "model"), ShadowBytes::unknown("reason"), ShadowBytes::estimated(0, "model")];
        let coverage = shadow_bytes_coverage(values.iter());
        assert_eq!(coverage, Coverage { total: 3, known: 2 });
    }

    #[test]
    fn classify_delta_distinguishes_under_over_exact_and_unknown() {
        assert_eq!(classify_delta(Some(100), Some(150)), (Some(50), DeltaClassification::Under));
        assert_eq!(classify_delta(Some(150), Some(100)), (Some(-50), DeltaClassification::Over));
        assert_eq!(classify_delta(Some(100), Some(100)), (Some(0), DeltaClassification::Exact));
        assert_eq!(classify_delta(None, Some(100)), (None, DeltaClassification::Unknown));
        assert_eq!(classify_delta(Some(100), None), (None, DeltaClassification::Unknown));
    }

    fn envelope(task: usize, bytes: ShadowBytes) -> ShadowTaskEnvelope {
        ShadowTaskEnvelope {
            task,
            plan_node: None,
            live_resources: Vec::new(),
            overlapping_workspaces: Vec::new(),
            query_owned_bytes: bytes,
            borrowed_resident_bytes: ShadowBytes::estimated(0, "fixture"),
        }
    }

    fn observed_row(compiled_task_id: &str, max_observed_peak_bytes: Option<usize>) -> ObservedTaskRuntimeRow {
        ObservedTaskRuntimeRow {
            compiled_task_id: compiled_task_id.to_string(),
            operator_kind: "fixture_operator".to_string(),
            elapsed_ns: 1,
            max_observed_peak_bytes,
            max_retained_bytes: None,
        }
    }

    #[test]
    fn join_task_runtime_rows_matches_by_parsed_task_index() {
        let envelopes = vec![envelope(0, ShadowBytes::estimated(10, "model"))];
        let observed = vec![observed_row("t0", Some(12))];
        let (deltas, mismatches) = join_task_runtime_rows(&envelopes, &observed);
        assert!(mismatches.is_empty());
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].signed_delta_bytes, Some(2));
        assert_eq!(deltas[0].classification, DeltaClassification::Under);
    }

    #[test]
    fn join_task_runtime_rows_never_guesses_alignment_for_out_of_range_or_unparseable_ids() {
        let envelopes =
            vec![envelope(0, ShadowBytes::estimated(10, "model")), envelope(1, ShadowBytes::unknown("reason"))];
        let observed =
            vec![observed_row("t0", Some(12)), observed_row("t9", Some(1)), observed_row("not-a-task", None)];
        let (deltas, mismatches) = join_task_runtime_rows(&envelopes, &observed);

        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].classification, DeltaClassification::Under);
        assert_eq!(
            deltas[1].classification,
            DeltaClassification::Unknown,
            "an unknown estimate must never be treated as a fabricated zero"
        );
        assert!(deltas[1].signed_delta_bytes.is_none());

        assert_eq!(mismatches.len(), 2);
        assert!(mismatches.iter().any(|m| m.reason == "task_index_out_of_range" && m.compiled_task_id == "t9"));
        assert!(
            mismatches.iter().any(|m| m.reason == "unparseable_compiled_task_id" && m.compiled_task_id == "not-a-task")
        );
    }

    #[test]
    fn join_task_runtime_rows_flags_duplicate_claims_on_the_same_task_index() {
        let envelopes = vec![envelope(0, ShadowBytes::estimated(10, "model"))];
        let observed = vec![observed_row("t0", Some(5)), observed_row("t0", Some(9))];
        let (deltas, mismatches) = join_task_runtime_rows(&envelopes, &observed);
        assert_eq!(deltas.len(), 1);
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].reason, "duplicate_compiled_task_id");
    }
}
