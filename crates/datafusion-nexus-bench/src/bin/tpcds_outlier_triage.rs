// Per-query triage dump for TPC-DS outliers where CPU and GPU diverge.
//
// For each requested query, emits one directory under the output root with:
//   - The SQL
//   - Tuned-CPU and native-GPU session configs (and a unified diff)
//   - Unoptimized and optimized DataFusion logical plans (CPU and GPU session)
//   - CPU final physical plan (post-physical-opt, the one CPU actually runs)
//   - GPU physical plan that arrives at `CudfNativePhysicalOptimizerRule`
//     (post-physical-opt under GPU session config, WITHOUT our rule)
//   - GPU physical plan after our rule runs (NativeCudfExec wrappers visible)
//   - Actual retained planning diagnostics from `CudfNativePhysicalOptimizerRule`
//   - Lowered native IR (`nexus_query_engine::QueryPlan` per candidate) from
//     calling `lower_physical_plan_to_native` on the pre-rule plan
//   - The `PlanningReport` from the lowering call (rewritten / rejected
//     candidates with `FallbackReason::code()`)
//   - A per-query SUMMARY.md that lists every file and quotes the suggested
//     `diff` commands for cross-engine and pre/post-rule comparisons
//
// Inputs (env vars):
//   CUDF_TPCDS_PARQUET_DIR              - required, path to TPC-DS parquet root
//   CUDF_TPCDS_OUTLIER_TRIAGE_QUERIES   - required, comma-separated query ids
//   CUDF_TPCDS_OUTLIER_TRIAGE_OUT_DIR   - optional, default ./tpcds_outlier_triage_<unix_ns>
//
// This binary does not execute the plans, so it does not need the GPU lock
// and finishes in seconds even for SF100 inputs.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion::{
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, displayable},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt,
    NativeDataFusionPlanningReport, lower_physical_plan_to_native,
};
use datafusion_nexus_bench::{
    TPCDS_QUERIES, TpcdsQuery, register_tpcds_tables, tpcds_native_optimizer_config, tpcds_native_report_session_config,
};
use nexus_query_engine::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
    DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JOIN_FILTER_REASSOCIATION_RULE_NAME,
    LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, OUTER_JOIN_NULL_REJECTION_RULE_NAME, PROJECTION_SIMPLIFICATION_RULE_NAME,
    SEMIJOIN_REDUCTION_RULE_NAME, SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME,
};

const CURRENT_STATE_TSV_SCHEMA_VERSION: u32 = 1;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn Error>> {
    let cfg = TriageConfig::from_env()?;
    eprintln!(
        "[tpcds_outlier_triage] root={} queries={} out={}",
        cfg.root.display(),
        cfg.queries.iter().map(|q| q.id).collect::<Vec<_>>().join(","),
        cfg.out_dir.display(),
    );
    fs::create_dir_all(&cfg.out_dir)?;

    let ctx_cpu = build_cpu_context(&cfg.root).await?;
    let optimizer_config = tpcds_native_optimizer_config();
    let (ctx_gpu_pre, _) = build_gpu_context(&cfg.root, false, optimizer_config.clone()).await?;
    let (ctx_gpu_post, gpu_native_rule) = build_gpu_context(&cfg.root, true, optimizer_config.clone()).await?;

    let mut summary_rows: Vec<String> = Vec::new();
    let mut current_state_rows: Vec<CurrentStateRow> = Vec::new();
    for query in &cfg.queries {
        match triage_query(
            query,
            &cfg.out_dir,
            &ctx_cpu,
            &ctx_gpu_pre,
            &ctx_gpu_post,
            gpu_native_rule.as_ref(),
            &optimizer_config,
        )
        .await
        {
            Ok(summary) => {
                summary_rows.push(summary.markdown_row);
                current_state_rows.push(summary.current_state);
            },
            Err(error) => {
                summary_rows.push(format!("| {} | error: {} |", query.id, error));
                eprintln!("[tpcds_outlier_triage] {} failed: {}", query.id, error);
            },
        }
    }

    write_current_state_tsv(&cfg.out_dir, &current_state_rows)?;
    write_top_summary(&cfg, &summary_rows)?;
    eprintln!("[tpcds_outlier_triage] done -> {}", cfg.out_dir.display());
    Ok(())
}

struct TriageConfig {
    root: PathBuf,
    queries: Vec<&'static TpcdsQuery>,
    out_dir: PathBuf,
}

impl TriageConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let root = std::env::var("CUDF_TPCDS_PARQUET_DIR")
            .map_err(|_| "CUDF_TPCDS_PARQUET_DIR must be set to the TPC-DS parquet root")?;
        let root = PathBuf::from(root);
        let queries_csv = std::env::var("CUDF_TPCDS_OUTLIER_TRIAGE_QUERIES")
            .map_err(|_| "CUDF_TPCDS_OUTLIER_TRIAGE_QUERIES must list query ids, e.g. q08,q21,q39")?;
        let query_ids: Vec<String> =
            queries_csv.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        if query_ids.is_empty() {
            return Err("CUDF_TPCDS_OUTLIER_TRIAGE_QUERIES is empty".into());
        }
        let queries: Vec<&'static TpcdsQuery> = query_ids
            .iter()
            .map(|id| TPCDS_QUERIES.iter().find(|q| q.id == id).ok_or_else(|| format!("unknown TPC-DS query id: {id}")))
            .collect::<Result<_, _>>()?;
        let out_dir = match std::env::var("CUDF_TPCDS_OUTLIER_TRIAGE_OUT_DIR") {
            Ok(value) => PathBuf::from(value),
            Err(_) => PathBuf::from(format!(
                "tpcds_outlier_triage_{}",
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
            )),
        };
        Ok(Self { root, queries, out_dir })
    }
}

async fn build_cpu_context(root: &Path) -> Result<SessionContext, Box<dyn Error>> {
    let ctx = SessionContext::new_with_config(tuned_cpu_session_config());
    register_tpcds_tables(&ctx, root).await?;
    Ok(ctx)
}

async fn build_gpu_context(
    root: &Path,
    with_native_rule: bool,
    optimizer_config: CudfOptimizerConfig,
) -> Result<(SessionContext, Option<Arc<CudfNativePhysicalOptimizerRule>>), Box<dyn Error>> {
    let mut builder =
        SessionStateBuilder::new_with_default_features().with_config(tpcds_native_report_session_config());
    let mut retained_rule = None;
    if with_native_rule {
        let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
        retained_rule = Some(Arc::clone(&rule));
        builder = builder.with_cudf_native_rule(rule);
    }
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, root).await?;
    Ok((ctx, retained_rule))
}

fn tuned_cpu_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_target_partitions(16)
        .with_batch_size(8192)
        .with_repartition_file_scans(true)
        .with_repartition_aggregations(true)
        .with_repartition_joins(true)
        .with_repartition_sorts(true);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_join_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = true;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config
}

async fn triage_query(
    query: &TpcdsQuery,
    out_root: &Path,
    ctx_cpu: &SessionContext,
    ctx_gpu_pre: &SessionContext,
    ctx_gpu_post: &SessionContext,
    gpu_native_rule: Option<&Arc<CudfNativePhysicalOptimizerRule>>,
    optimizer_config: &CudfOptimizerConfig,
) -> Result<QueryTriageSummary, Box<dyn Error>> {
    let qid = query.id;
    let dir = out_root.join(qid);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("query.sql"), query.sql.trim_start())?;

    fs::write(dir.join("session_config.cpu.txt"), describe_session_config(&ctx_cpu.copied_config()))?;
    fs::write(dir.join("session_config.gpu.txt"), describe_session_config(&ctx_gpu_pre.copied_config()))?;

    let (cpu_unopt, cpu_opt, cpu_physical) = dump_engine_plans(ctx_cpu, query.sql).await?;
    fs::write(dir.join("logical_plan.unopt.cpu.txt"), &cpu_unopt)?;
    fs::write(dir.join("logical_plan.opt.cpu.txt"), &cpu_opt)?;
    fs::write(dir.join("physical_plan.cpu.txt"), &cpu_physical)?;

    let (gpu_unopt, gpu_opt, gpu_pre_rule_text) = dump_engine_plans(ctx_gpu_pre, query.sql).await?;
    fs::write(dir.join("logical_plan.unopt.gpu.txt"), &gpu_unopt)?;
    fs::write(dir.join("logical_plan.opt.gpu.txt"), &gpu_opt)?;
    fs::write(dir.join("physical_plan.gpu.preRule.txt"), &gpu_pre_rule_text)?;

    let gpu_pre_rule_plan = ctx_gpu_pre.sql(query.sql).await?.create_physical_plan().await?;
    let lowering = lower_physical_plan_to_native(Arc::clone(&gpu_pre_rule_plan));
    fs::write(dir.join("native_ir.gpu.txt"), format_native_ir(&lowering))?;
    fs::write(dir.join("native_ir.gpu.optimized.txt"), format_optimized_native_ir(&lowering, optimizer_config))?;
    fs::write(dir.join("planning_report.gpu.txt"), format_planning_report(&lowering))?;

    let gpu_post_rule_plan = ctx_gpu_post.sql(query.sql).await?.create_physical_plan().await?;
    let retained_planning_tsv = gpu_native_rule
        .map(|rule| NativeDataFusionPlanningReport::from_retained_rule(qid, Arc::clone(&gpu_post_rule_plan), rule))
        .map(|report| report.to_tsv())
        .unwrap_or_else(|| "planning_report=<none: retained native rule diagnostics unavailable>\n".to_string());
    fs::write(dir.join("planning_report.gpu.postRule.tsv"), &retained_planning_tsv)?;
    let retained_diagnostics = gpu_native_rule.and_then(|rule| rule.last_planning_diagnostics());
    let retained_planning_report = retained_diagnostics
        .as_ref()
        .map(|diagnostics| format_planning_report_from_report(&diagnostics.report))
        .unwrap_or_else(|| "planning_report=<none: retained native rule diagnostics unavailable>\n".to_string());
    fs::write(dir.join("planning_report.gpu.postRule.txt"), retained_planning_report)?;
    let native_optimizer = retained_diagnostics
        .as_ref()
        .map(|diagnostics| format_native_optimizer_diagnostics(&diagnostics.native_optimizer))
        .unwrap_or_else(|| "native_optimizer=<none: retained native rule diagnostics unavailable>\n".to_string());
    fs::write(dir.join("native_optimizer.gpu.postRule.txt"), native_optimizer)?;
    let strict_contract =
        retained_diagnostics.as_ref().map(format_strict_bounded_contract_diagnostics).unwrap_or_else(|| {
            "strict_bounded_contract=<none: retained native rule diagnostics unavailable>\n".to_string()
        });
    fs::write(dir.join("strict_bounded_contract.gpu.postRule.txt"), strict_contract)?;
    let adaptive_pipeline = retained_diagnostics
        .as_ref()
        .map(format_adaptive_pipeline_diagnostics)
        .unwrap_or_else(|| "adaptive_pipeline=<none: retained native rule diagnostics unavailable>\n".to_string());
    fs::write(dir.join("adaptive_pipeline.gpu.postRule.txt"), adaptive_pipeline)?;
    let gpu_post_rule_text = displayable(gpu_post_rule_plan.as_ref()).indent(true).to_string();
    fs::write(dir.join("physical_plan.gpu.postRule.txt"), &gpu_post_rule_text)?;
    let post_rule_native_plans = collect_native_exec_plans(gpu_post_rule_plan.as_ref());
    fs::write(dir.join("native_ir.gpu.postRule.txt"), format_post_rule_native_ir(&post_rule_native_plans))?;
    fs::write(
        dir.join("compiled_pipeline.gpu.postRule.txt"),
        format_post_rule_compiled_pipelines(&post_rule_native_plans),
    )?;

    fs::write(dir.join("SUMMARY.md"), render_query_summary(query, &lowering))?;
    let current_state = current_state_row(
        query,
        &lowering,
        retained_diagnostics.as_ref(),
        &retained_planning_tsv,
        &post_rule_native_plans,
        optimizer_config,
    );

    Ok(QueryTriageSummary {
        markdown_row: format!(
            "| {} | candidates={} rewritten={} rejected={} | [link](./{}/SUMMARY.md) |",
            qid,
            lowering.candidate_count(),
            lowering.rewritten_count(),
            lowering.rejected_count(),
            qid,
        ),
        current_state,
    })
}

struct QueryTriageSummary {
    markdown_row: String,
    current_state: CurrentStateRow,
}

struct CurrentStateRow {
    query_id: String,
    direct_lowering_outcome: String,
    direct_lowering_first_rejection: String,
    native_optimizer_rules_applied: String,
    first_capability_rejection_projection: String,
    strict_bounded_envelope: String,
    compiled_pipeline_status: String,
    compiled_pipeline_task_count: String,
    compiled_pipeline_edge_count: String,
    compiled_pipeline_barrier_count: String,
    compiled_pipeline_resource_count: String,
    compiled_pipeline_gate_count: String,
    post_native_execution_path: String,
    native_plan_fragment_count: String,
    host_datafusion_boundary_count: String,
    post_rule_planning_outcomes: String,
    bundle: String,
}

impl CurrentStateRow {
    const COLUMNS: &'static [&'static str] = &[
        "query_id",
        "direct_lowering_outcome",
        "direct_lowering_first_rejection",
        "native_optimizer_rules_applied",
        "first_capability_rejection_projection",
        "strict_bounded_envelope",
        "compiled_pipeline_status",
        "compiled_pipeline_task_count",
        "compiled_pipeline_edge_count",
        "compiled_pipeline_barrier_count",
        "compiled_pipeline_resource_count",
        "compiled_pipeline_gate_count",
        "post_native_execution_path",
        "native_plan_fragment_count",
        "host_datafusion_boundary_count",
        "post_rule_planning_outcomes",
        "bundle",
    ];

    fn to_tsv_line(&self) -> String {
        let fields = [
            self.query_id.as_str(),
            self.direct_lowering_outcome.as_str(),
            self.direct_lowering_first_rejection.as_str(),
            self.native_optimizer_rules_applied.as_str(),
            self.first_capability_rejection_projection.as_str(),
            self.strict_bounded_envelope.as_str(),
            self.compiled_pipeline_status.as_str(),
            self.compiled_pipeline_task_count.as_str(),
            self.compiled_pipeline_edge_count.as_str(),
            self.compiled_pipeline_barrier_count.as_str(),
            self.compiled_pipeline_resource_count.as_str(),
            self.compiled_pipeline_gate_count.as_str(),
            self.post_native_execution_path.as_str(),
            self.native_plan_fragment_count.as_str(),
            self.host_datafusion_boundary_count.as_str(),
            self.post_rule_planning_outcomes.as_str(),
            self.bundle.as_str(),
        ];
        debug_assert_eq!(fields.len(), Self::COLUMNS.len());
        fields.into_iter().map(stable_tsv_field).collect::<Vec<_>>().join("\t")
    }
}

struct OptimizedNativePlanEvidence {
    first_capability_rejection_projection: String,
    strict_bounded_envelope: String,
    compiled_pipeline_status: String,
    compiled_pipeline_task_count: String,
    compiled_pipeline_edge_count: String,
    compiled_pipeline_barrier_count: String,
    compiled_pipeline_resource_count: String,
    compiled_pipeline_gate_count: String,
}

fn current_state_row(
    query: &TpcdsQuery,
    lowering: &datafusion_nexus::NativeLoweringResult,
    retained_diagnostics: Option<&datafusion_nexus::NativePlanningDiagnostics>,
    retained_planning_tsv: &str,
    post_rule_native_plans: &[&nexus_query_engine::QueryPlan],
    optimizer_config: &CudfOptimizerConfig,
) -> CurrentStateRow {
    let optimized_evidence = optimized_native_plan_evidence(lowering, post_rule_native_plans, optimizer_config);
    let planning = summarize_retained_planning_tsv(retained_planning_tsv);
    CurrentStateRow {
        query_id: query.id.to_string(),
        direct_lowering_outcome: direct_lowering_outcome(lowering),
        direct_lowering_first_rejection: direct_lowering_first_rejection(lowering),
        native_optimizer_rules_applied: retained_diagnostics
            .map(|diagnostics| native_optimizer_applied_rules(&diagnostics.native_optimizer))
            .unwrap_or_else(|| "none".to_string()),
        first_capability_rejection_projection: optimized_evidence.first_capability_rejection_projection,
        strict_bounded_envelope: optimized_evidence.strict_bounded_envelope,
        compiled_pipeline_status: optimized_evidence.compiled_pipeline_status,
        compiled_pipeline_task_count: optimized_evidence.compiled_pipeline_task_count,
        compiled_pipeline_edge_count: optimized_evidence.compiled_pipeline_edge_count,
        compiled_pipeline_barrier_count: optimized_evidence.compiled_pipeline_barrier_count,
        compiled_pipeline_resource_count: optimized_evidence.compiled_pipeline_resource_count,
        compiled_pipeline_gate_count: optimized_evidence.compiled_pipeline_gate_count,
        post_native_execution_path: planning.post_native_execution_path,
        native_plan_fragment_count: planning.native_plan_fragment_count,
        host_datafusion_boundary_count: planning.host_datafusion_boundary_count,
        post_rule_planning_outcomes: planning.post_rule_planning_outcomes,
        bundle: format!("{}/SUMMARY.md", query.id),
    }
}

fn optimized_native_plan_evidence(
    lowering: &datafusion_nexus::NativeLoweringResult,
    post_rule_native_plans: &[&nexus_query_engine::QueryPlan],
    optimizer_config: &CudfOptimizerConfig,
) -> OptimizedNativePlanEvidence {
    let plan = if let Some(plan) = lowering.plans.first() {
        let options = native_optimizer_options_for_execution(optimizer_config);
        nexus_query_engine::optimize_query_plan(plan.clone(), &options).plan
    } else if let Some(plan) = post_rule_native_plans.first() {
        (*plan).clone()
    } else {
        return OptimizedNativePlanEvidence {
            first_capability_rejection_projection: "none:no_lowered_native_plan".to_string(),
            strict_bounded_envelope: "none:no_lowered_native_plan".to_string(),
            compiled_pipeline_status: "not_available:no_lowered_native_plan".to_string(),
            compiled_pipeline_task_count: "none".to_string(),
            compiled_pipeline_edge_count: "none".to_string(),
            compiled_pipeline_barrier_count: "none".to_string(),
            compiled_pipeline_resource_count: "none".to_string(),
            compiled_pipeline_gate_count: "none".to_string(),
        };
    };

    let statistical_mode = optimizer_config.statistical_aggregate_execution_mode();
    let admission =
        nexus_query_engine::strict_bounded_pipeline_admission_with_statistical_aggregate_mode(&plan, statistical_mode);
    let first_capability_rejection_projection =
        match nexus_query_engine::strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
            &plan,
            statistical_mode,
        ) {
            Ok(Some(projection)) => format_capability_rejection_projection(&projection),
            Ok(None) => "none".to_string(),
            Err(error) => format!("error:{}:{}", error.code().as_str(), stable_inline(error.message())),
        };

    match admission {
        Ok(Some(admission)) => OptimizedNativePlanEvidence {
            first_capability_rejection_projection,
            strict_bounded_envelope: admission.envelope().as_str().to_string(),
            compiled_pipeline_status: "compiled".to_string(),
            compiled_pipeline_task_count: admission.pipeline().tasks().len().to_string(),
            compiled_pipeline_edge_count: admission.pipeline().edges().len().to_string(),
            compiled_pipeline_barrier_count: admission.pipeline().barriers().len().to_string(),
            compiled_pipeline_resource_count: admission.pipeline().resources().len().to_string(),
            compiled_pipeline_gate_count: admission.pipeline().gates().len().to_string(),
        },
        Ok(None) => {
            let analysis = plan.capability_analysis();
            let strict_bounded_envelope = match analysis.strict_bounded_envelope_admission() {
                nexus_query_engine::capability::CapabilityEnvelopeAdmission::Admitted { envelope } => {
                    format!("not_admitted_for_mode:{},raw={}", statistical_mode.as_str(), envelope.as_str())
                },
                nexus_query_engine::capability::CapabilityEnvelopeAdmission::Rejected { projection } => {
                    format!("missing:{}", projection.reason)
                },
            };
            OptimizedNativePlanEvidence {
                first_capability_rejection_projection,
                strict_bounded_envelope,
                compiled_pipeline_status: "not_available:not_admitted_for_mode".to_string(),
                compiled_pipeline_task_count: "none".to_string(),
                compiled_pipeline_edge_count: "none".to_string(),
                compiled_pipeline_barrier_count: "none".to_string(),
                compiled_pipeline_resource_count: "none".to_string(),
                compiled_pipeline_gate_count: "none".to_string(),
            }
        },
        Err(error) => OptimizedNativePlanEvidence {
            first_capability_rejection_projection,
            strict_bounded_envelope: format!("error:{}", statistical_mode.as_str()),
            compiled_pipeline_status: format!(
                "compile_failed:{}:{}",
                error.code().as_str(),
                stable_inline(error.message())
            ),
            compiled_pipeline_task_count: "none".to_string(),
            compiled_pipeline_edge_count: "none".to_string(),
            compiled_pipeline_barrier_count: "none".to_string(),
            compiled_pipeline_resource_count: "none".to_string(),
            compiled_pipeline_gate_count: "none".to_string(),
        },
    }
}

fn native_optimizer_options_for_execution(config: &CudfOptimizerConfig) -> nexus_query_engine::NativeOptimizerOptions {
    let mut options = config.native_optimizer_options().clone();
    options.semijoin_reduction.admission_mode = match config.native_pipeline_execution_mode() {
        nexus_query_engine::NativePipelineExecutionMode::StrictBounded => {
            nexus_query_engine::SemijoinReductionAdmissionMode::StrictBounded
        },
        nexus_query_engine::NativePipelineExecutionMode::Adaptive => {
            nexus_query_engine::SemijoinReductionAdmissionMode::Adaptive
        },
    };
    options
}

fn direct_lowering_outcome(lowering: &datafusion_nexus::NativeLoweringResult) -> String {
    match (lowering.rewritten_count(), lowering.rejected_count()) {
        (0, 0) => "no_candidates".to_string(),
        (rewritten, 0) => format!("rewritten:{rewritten}"),
        (0, rejected) => format!("rejected:{rejected}"),
        (rewritten, rejected) => format!("mixed:rewritten={rewritten},rejected={rejected}"),
    }
}

fn direct_lowering_first_rejection(lowering: &datafusion_nexus::NativeLoweringResult) -> String {
    lowering
        .report
        .outcomes()
        .iter()
        .find_map(|outcome| match outcome {
            CandidateOutcome::Rejected { candidate, reason } => {
                Some(format!("shape={},reason={}", candidate.shape, reason.code()))
            },
            CandidateOutcome::Rewritten { .. } => None,
        })
        .unwrap_or_else(|| "none".to_string())
}

fn native_optimizer_applied_rules(diagnostics: &nexus_query_engine::NativeOptimizerDiagnostics) -> String {
    let mut rules = Vec::new();
    if diagnostics.projection_simplification.applied_count > 0 {
        rules.push(PROJECTION_SIMPLIFICATION_RULE_NAME);
    }
    if diagnostics.semijoin_reduction.applied_count > 0 {
        rules.push(SEMIJOIN_REDUCTION_RULE_NAME);
    }
    if diagnostics.join_build_side_selection.applied_count > 0 {
        rules.push(JOIN_BUILD_SIDE_SELECTION_RULE_NAME);
    }
    if diagnostics.outer_join_null_rejection.applied_count > 0 {
        rules.push(OUTER_JOIN_NULL_REJECTION_RULE_NAME);
    }
    if diagnostics.join_filter_reassociation.applied_count > 0 {
        rules.push(JOIN_FILTER_REASSOCIATION_RULE_NAME);
    }
    if diagnostics.dimension_derived_row_group_pruning.applied_count > 0 {
        rules.push(DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME);
    }
    if diagnostics.stream_fanout_replay.applied_count > 0 {
        rules.push(STREAM_FANOUT_REPLAY_RULE_NAME);
    }
    if diagnostics.local_parquet_source_reuse.applied_count > 0 {
        rules.push(LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME);
    }
    if diagnostics.shared_subplan_materialization.applied_count > 0 {
        rules.push(SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME);
    }
    if diagnostics.dynamic_filter_route_discovery.route_count > 0 {
        rules.push(DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME);
    }
    if diagnostics.aggregate_join_pushdown.applied_count > 0 {
        rules.push(AGGREGATE_JOIN_PUSHDOWN_RULE_NAME);
    }
    if rules.is_empty() { "none".to_string() } else { rules.join(",") }
}

fn format_capability_rejection_projection(
    projection: &nexus_query_engine::capability::CapabilityRejectionProjection,
) -> String {
    let facts = if projection.facts.is_empty() {
        "none".to_string()
    } else {
        projection
            .facts
            .iter()
            .map(|(key, value)| format!("{key}={}", stable_inline(value)))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "category={},scope={},memory_boundary={},reason={},facts={}",
        projection.category.as_str(),
        projection.contract_scope.as_str(),
        projection.memory_boundary,
        projection.reason,
        facts
    )
}

struct RetainedPlanningSummary {
    post_native_execution_path: String,
    native_plan_fragment_count: String,
    host_datafusion_boundary_count: String,
    post_rule_planning_outcomes: String,
}

fn summarize_retained_planning_tsv(tsv: &str) -> RetainedPlanningSummary {
    let Some(header) = tsv.lines().find(|line| !line.starts_with('#') && line.contains('\t')) else {
        return RetainedPlanningSummary {
            post_native_execution_path: "none".to_string(),
            native_plan_fragment_count: "none".to_string(),
            host_datafusion_boundary_count: "none".to_string(),
            post_rule_planning_outcomes: "none".to_string(),
        };
    };
    let columns = header.split('\t').collect::<Vec<_>>();
    let index = |name: &str| columns.iter().position(|column| *column == name);
    let path_index = index("post_native_execution_path");
    let fragments_index = index("native_plan_fragment_count");
    let boundaries_index = index("host_datafusion_boundary_count");
    let outcome_index = index("outcome");
    let reason_index = index("reason_code");

    let rows = tsv
        .lines()
        .skip_while(|line| *line != header)
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let first_field = |idx: Option<usize>| {
        rows.first()
            .and_then(|fields| idx.and_then(|idx| fields.get(idx)))
            .map(|value| (*value).to_string())
            .unwrap_or_else(|| "none".to_string())
    };
    let outcomes = rows
        .iter()
        .filter_map(|fields| {
            let outcome = outcome_index.and_then(|idx| fields.get(idx))?;
            let reason = reason_index.and_then(|idx| fields.get(idx)).copied().unwrap_or("none");
            Some(if reason == "none" { (*outcome).to_string() } else { format!("{outcome}:{reason}") })
        })
        .collect::<Vec<_>>();

    RetainedPlanningSummary {
        post_native_execution_path: first_field(path_index),
        native_plan_fragment_count: first_field(fragments_index),
        host_datafusion_boundary_count: first_field(boundaries_index),
        post_rule_planning_outcomes: if outcomes.is_empty() { "none".to_string() } else { outcomes.join(";") },
    }
}

fn write_current_state_tsv(out_dir: &Path, rows: &[CurrentStateRow]) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str(&format!(
        "# tpcds_strict_bounded_phase0_current_state_schema_version={CURRENT_STATE_TSV_SCHEMA_VERSION}\n"
    ));
    out.push_str(&CurrentStateRow::COLUMNS.join("\t"));
    out.push('\n');
    for row in rows {
        out.push_str(&row.to_tsv_line());
        out.push('\n');
    }
    fs::write(out_dir.join("current_state.tsv"), out)?;
    Ok(())
}

fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn stable_inline(value: &str) -> String {
    stable_tsv_field(value).replace(',', ";")
}

async fn dump_engine_plans(ctx: &SessionContext, sql: &str) -> Result<(String, String, String), Box<dyn Error>> {
    let df = ctx.sql(sql).await?;
    let unopt_logical = df.logical_plan().display_indent().to_string();
    let opt_logical = df.clone().into_optimized_plan()?.display_indent().to_string();
    let physical_plan: Arc<dyn ExecutionPlan> = df.create_physical_plan().await?;
    let physical_text = displayable(physical_plan.as_ref()).indent(true).to_string();
    Ok((unopt_logical, opt_logical, physical_text))
}

fn collect_native_exec_plans(plan: &dyn ExecutionPlan) -> Vec<&nexus_query_engine::QueryPlan> {
    let mut plans = Vec::new();
    collect_native_exec_plans_inner(plan, &mut plans);
    plans
}

fn collect_native_exec_plans_inner<'a>(
    plan: &'a dyn ExecutionPlan,
    plans: &mut Vec<&'a nexus_query_engine::QueryPlan>,
) {
    if let Some(native) = plan.as_any().downcast_ref::<datafusion_nexus::NativeCudfExec>() {
        plans.push(native.native_plan());
    }
    if let Some(native) = plan.as_any().downcast_ref::<datafusion_nexus::NativeCudfComposedExec>() {
        plans.push(native.native_plan());
    }
    for child in plan.children() {
        collect_native_exec_plans_inner(child.as_ref(), plans);
    }
}

fn describe_session_config(config: &SessionConfig) -> String {
    let options = config.options();
    let mut out = String::new();
    out.push_str(&format!("target_partitions={}\n", config.target_partitions()));
    out.push_str(&format!("batch_size={}\n", config.batch_size()));
    out.push_str(&format!("repartition_file_scans={}\n", options.optimizer.repartition_file_scans));
    out.push_str(&format!("repartition_aggregations={}\n", options.optimizer.repartition_aggregations));
    out.push_str(&format!("repartition_joins={}\n", options.optimizer.repartition_joins));
    out.push_str(&format!("repartition_sorts={}\n", options.optimizer.repartition_sorts));
    out.push_str(&format!("repartition_windows={}\n", options.optimizer.repartition_windows));
    out.push_str(&format!(
        "enable_topk_dynamic_filter_pushdown={}\n",
        options.optimizer.enable_topk_dynamic_filter_pushdown,
    ));
    out.push_str(&format!(
        "enable_aggregate_dynamic_filter_pushdown={}\n",
        options.optimizer.enable_aggregate_dynamic_filter_pushdown,
    ));
    out.push_str(&format!(
        "enable_join_dynamic_filter_pushdown={}\n",
        options.optimizer.enable_join_dynamic_filter_pushdown,
    ));
    out.push_str(&format!("enable_dynamic_filter_pushdown={}\n", options.optimizer.enable_dynamic_filter_pushdown,));
    out.push_str(&format!("parquet_pushdown_filters={}\n", options.execution.parquet.pushdown_filters));
    out.push_str(&format!("parquet_reorder_filters={}\n", options.execution.parquet.reorder_filters));
    out.push_str(&format!("collect_statistics={}\n", options.execution.collect_statistics));
    out
}

fn format_native_ir(lowering: &datafusion_nexus::NativeLoweringResult) -> String {
    let mut out = String::new();
    out.push_str(&format!("candidate_count={}\n", lowering.candidate_count()));
    out.push_str(&format!("rewritten_count={}\n", lowering.rewritten_count()));
    out.push_str(&format!("rejected_count={}\n", lowering.rejected_count()));
    out.push_str(&format!("source_diagnostic_count={}\n", lowering.source_diagnostics.len()));
    out.push('\n');
    if lowering.plans.is_empty() {
        out.push_str("native_ir=<none: no candidates lowered>\n");
    }
    for (index, plan) in lowering.plans.iter().enumerate() {
        out.push_str(&format!("native_plan[{index}]=\n{plan}\n\n"));
    }
    out
}

fn format_optimized_native_ir(
    lowering: &datafusion_nexus::NativeLoweringResult,
    optimizer_config: &CudfOptimizerConfig,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("candidate_count={}\n", lowering.candidate_count()));
    out.push_str(&format!("rewritten_count={}\n", lowering.rewritten_count()));
    out.push_str(&format!("rejected_count={}\n", lowering.rejected_count()));
    out.push('\n');
    if lowering.plans.is_empty() {
        out.push_str("native_ir=<none: no candidates lowered>\n");
    }
    let options = native_optimizer_options_for_execution(optimizer_config);
    let statistical_mode = optimizer_config.statistical_aggregate_execution_mode();
    for (index, plan) in lowering.plans.iter().enumerate() {
        let optimized = nexus_query_engine::optimize_query_plan(plan.clone(), &options);
        let analysis = optimized.plan.capability_analysis();
        let mode_aware_envelope =
            match nexus_query_engine::strict_bounded_pipeline_admission_with_statistical_aggregate_mode(
                &optimized.plan,
                statistical_mode,
            ) {
                Ok(Some(admission)) => admission.envelope().as_str().to_string(),
                Ok(None) => "none:not_admitted_for_mode".to_string(),
                Err(error) => format!("error:{}:{}", error.code().as_str(), stable_inline(error.message())),
            };
        out.push_str(&format!(
            "native_plan[{index}] changed={} applied_rules={}\nstrict_bounded_envelope_admission={:?}\nmode_aware_strict_bounded_envelope={}\nstatistical_aggregate_execution_mode={}\nrollup_rank_topk_after_aggregate_join_pushdown_admission={:?}\nfirst_rejection_projection={:?}\n{}\n\n",
            optimized.changed(),
            native_optimizer_applied_rules(&optimized.diagnostics),
            analysis.strict_bounded_envelope_admission(),
            mode_aware_envelope,
            statistical_mode.as_str(),
            analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission,
            analysis.first_rejection_projection(),
            optimized.plan
        ));
    }
    out
}

fn format_post_rule_native_ir(plans: &[&nexus_query_engine::QueryPlan]) -> String {
    let mut out = String::new();
    out.push_str(&format!("native_exec_count={}\n\n", plans.len()));
    if plans.is_empty() {
        out.push_str("native_ir=<none: post-rule plan has no NativeCudfExec or NativeCudfComposedExec>\n");
    }
    for (index, plan) in plans.iter().enumerate() {
        out.push_str(&format!("native_plan[{index}]=\n{plan}\n\n"));
    }
    out
}

fn format_post_rule_compiled_pipelines(plans: &[&nexus_query_engine::QueryPlan]) -> String {
    let mut out = String::new();
    out.push_str(&format!("native_exec_count={}\n\n", plans.len()));
    if plans.is_empty() {
        out.push_str("compiled_pipeline=<none: post-rule plan has no NativeCudfExec or NativeCudfComposedExec>\n");
    }
    for (index, plan) in plans.iter().enumerate() {
        let analysis = plan.capability_analysis();
        match nexus_query_engine::compile_pipeline(plan, &analysis) {
            Ok(pipeline) => out.push_str(&format!("compiled_pipeline[{index}]=\n{}\n", pipeline.debug_render())),
            Err(error) => {
                out.push_str(&format!("compiled_pipeline[{index}]=<compile failed: {error}>\n\n"));
            },
        }
    }
    out
}

fn format_native_optimizer_diagnostics(diagnostics: &nexus_query_engine::NativeOptimizerDiagnostics) -> String {
    let mut out = String::new();
    out.push_str(&format!("is_empty={}\n", diagnostics.is_empty()));
    out.push_str(&format!("{diagnostics:#?}\n"));
    out
}

fn format_strict_bounded_contract_diagnostics(diagnostics: &datafusion_nexus::NativePlanningDiagnostics) -> String {
    let contract = &diagnostics.strict_bounded_contract;
    let mut out = String::new();
    out.push_str(&format!("accepted_count={}\n", contract.accepted_candidates.len()));
    out.push_str(&format!("rejected_count={}\n", contract.rejected_candidates.len()));
    out.push_str(&format!("{contract:#?}\n"));
    out
}

fn format_adaptive_pipeline_diagnostics(diagnostics: &datafusion_nexus::NativePlanningDiagnostics) -> String {
    let adaptive = &diagnostics.adaptive_pipeline;
    let rejected_count = adaptive
        .candidates
        .iter()
        .filter(|candidate| {
            candidate.report.row().candidate_status == nexus_query_engine::AdaptiveCandidateStatus::Rejected
        })
        .count();
    let mut out = String::new();
    out.push_str(&format!("candidate_count={}\n", adaptive.candidates.len()));
    out.push_str(&format!("rejected_count={rejected_count}\n"));
    out.push_str(&format!("{adaptive:#?}\n"));
    out
}

fn format_planning_report(lowering: &datafusion_nexus::NativeLoweringResult) -> String {
    let mut out = String::new();
    out.push_str("# Lowering CandidateOutcome list (one per candidate found before execution-boundary admission)\n\n");
    out.push_str(&format_planning_report_from_report(&lowering.report));
    out
}

fn format_planning_report_from_report(report: &datafusion_nexus::PlanningReport) -> String {
    let mut out = String::new();
    out.push_str("# CandidateOutcome list\n\n");
    for (index, outcome) in report.outcomes().iter().enumerate() {
        match outcome {
            CandidateOutcome::Rewritten { candidate } => {
                out.push_str(&format!(
                    "[{}] REWRITTEN shape={} operator_count={} output_schema_fields={}\n",
                    index, candidate.shape, candidate.operator_count, candidate.output_schema_fields,
                ));
            },
            CandidateOutcome::Rejected { candidate, reason } => {
                out.push_str(&format!(
                    "[{}] REJECTED  shape={} operator_count={} output_schema_fields={}\n           reason={} detail={:?}\n",
                    index, candidate.shape, candidate.operator_count, candidate.output_schema_fields,
                    reason.code(), reason,
                ));
            },
        }
    }
    if report.outcomes().is_empty() {
        out.push_str("(no candidates)\n");
    }
    out
}

fn render_query_summary(query: &TpcdsQuery, lowering: &datafusion_nexus::NativeLoweringResult) -> String {
    let mut out = String::new();
    out.push_str(&format!("# TPC-DS {} triage bundle\n\n", query.id));
    out.push_str("## Files\n\n");
    out.push_str("| file | what it is |\n|---|---|\n");
    out.push_str("| query.sql | the SQL submitted to both engines |\n");
    out.push_str("| session_config.cpu.txt | tuned CPU DataFusion session config |\n");
    out.push_str(
        "| session_config.gpu.txt | native-GPU report session config (target_partitions=1, no repartition) |\n",
    );
    out.push_str("| logical_plan.unopt.cpu.txt | CPU `df.logical_plan()` (post-analyzer, pre-logical-optimizer) |\n");
    out.push_str("| logical_plan.opt.cpu.txt | CPU `df.into_optimized_plan()` (post-logical-optimizer) |\n");
    out.push_str("| physical_plan.cpu.txt | CPU final physical plan (post-physical-optimizer, executed) |\n");
    out.push_str("| logical_plan.unopt.gpu.txt | GPU `df.logical_plan()` under GPU session config |\n");
    out.push_str("| logical_plan.opt.gpu.txt | GPU `df.into_optimized_plan()` under GPU session config |\n");
    out.push_str("| physical_plan.gpu.preRule.txt | GPU session physical plan WITHOUT `CudfNativePhysicalOptimizerRule` (the input the rule sees) |\n");
    out.push_str("| physical_plan.gpu.postRule.txt | GPU session physical plan WITH the rule (NativeCudfExec wrappers visible) |\n");
    out.push_str("| planning_report.gpu.postRule.txt | actual retained rule report after execution-boundary admission and fallback policy |\n");
    out.push_str(
        "| planning_report.gpu.postRule.tsv | canonical native DataFusion planning row(s) for the post-rule plan |\n",
    );
    out.push_str("| native_optimizer.gpu.postRule.txt | retained native optimizer diagnostics from the post-rule GPU physical optimizer |\n");
    out.push_str("| strict_bounded_contract.gpu.postRule.txt | strict-bounded admission evidence retained by the post-rule optimizer |\n");
    out.push_str(
        "| adaptive_pipeline.gpu.postRule.txt | adaptive admission evidence retained by the post-rule optimizer |\n",
    );
    out.push_str("| native_ir.gpu.txt | lowered `nexus_query_engine::QueryPlan` per rewritten candidate |\n");
    out.push_str(
        "| native_ir.gpu.optimized.txt | direct lowered native IR after native optimizer rules, with strict envelope evidence |\n",
    );
    out.push_str(
        "| native_ir.gpu.postRule.txt | optimized native IR extracted from post-rule `NativeCudfExec` wrappers |\n",
    );
    out.push_str("| compiled_pipeline.gpu.postRule.txt | compiled strict/adaptive-neutral pipeline for each optimized post-rule native IR |\n");
    out.push_str("| planning_report.gpu.txt | direct lowering outcomes before execution-boundary admission |\n\n");

    out.push_str("## Lowering snapshot\n\n");
    out.push_str(&format!("- candidate_count = {}\n", lowering.candidate_count()));
    out.push_str(&format!("- rewritten_count = {}\n", lowering.rewritten_count()));
    out.push_str(&format!("- rejected_count = {}\n", lowering.rejected_count()));
    out.push_str(&format!("- source_diagnostic_count = {}\n\n", lowering.source_diagnostics.len()));

    out.push_str("## Suggested diff commands\n\n");
    out.push_str("```sh\n");
    out.push_str("# Has DataFusion logical optimization diverged between CPU and GPU sessions?\n");
    out.push_str("diff -u logical_plan.opt.cpu.txt logical_plan.opt.gpu.txt\n\n");
    out.push_str("# Has DataFusion physical planning produced different shapes (config-driven)?\n");
    out.push_str("# CPU has 16 partitions + repartition, GPU has 1 + none — significant structural diffs expected\n");
    out.push_str("diff -u physical_plan.cpu.txt physical_plan.gpu.preRule.txt\n\n");
    out.push_str("# What did our rule rewrite? (which subtrees got NativeCudfExec'd)\n");
    out.push_str("diff -u physical_plan.gpu.preRule.txt physical_plan.gpu.postRule.txt\n");
    out.push_str("```\n\n");

    out.push_str("## Decision tree\n\n");
    out.push_str("- If `logical_plan.opt.*` diff is non-trivial → root cause in DataFusion logical optimizer (session config), not in our crates.\n");
    out.push_str("- If `physical_plan.cpu.txt` vs `physical_plan.gpu.preRule.txt` differs in **shape** (e.g. join order, agg split, repartition placement) → DataFusion physical planning is choosing differently for the GPU session config; fix in datafusion-nexus session config or align partitioning settings.\n");
    out.push_str("- If `physical_plan.gpu.postRule.txt` shows large `DataFusion` subtrees not wrapped in `NativeCudfExec` → our rule rejected them; read `planning_report.gpu.postRule.txt` and `strict_bounded_contract.gpu.postRule.txt` to see the actual post-admission reason. `planning_report.gpu.txt` is the direct lowering report before execution-boundary admission.\n");
    out.push_str("- If `physical_plan.gpu.postRule.txt` is fully native AND `native_ir.gpu.txt` looks reasonable → root cause is at execution time; combine this bundle with the runtime metrics from `tpcds_three_way_bench` (operator class breakdown, source read, materialization, peak memory) and look at **nexus-query-engine** (operator implementation, reservation) or **cudf-rs** (libcudf primitive performance / API gap).\n");

    out
}

fn write_top_summary(cfg: &TriageConfig, rows: &[String]) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("# TPC-DS outlier triage\n\n");
    out.push_str(&format!("- parquet root: `{}`\n", cfg.root.display()));
    out.push_str(&format!("- output dir: `{}`\n", cfg.out_dir.display()));
    out.push_str("- current state rows: [`current_state.tsv`](./current_state.tsv)\n");
    out.push_str("\n## Queries\n\n");
    out.push_str("| query | lowering | bundle |\n|---|---|---|\n");
    for row in rows {
        out.push_str(row);
        out.push('\n');
    }
    out.push_str("\n## Triage decision tree\n\n");
    out.push_str("Open any query's `SUMMARY.md` for diff commands and the decision tree (logical-divergence → physical-divergence → rule-rejection → nexus-query-engine / cudf-rs).\n");
    fs::write(cfg.out_dir.join("SUMMARY.md"), out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_state_tsv_row_contract_matches_columns() {
        let row = CurrentStateRow {
            query_id: "q01".to_string(),
            direct_lowering_outcome: "rewritten".to_string(),
            direct_lowering_first_rejection: "none".to_string(),
            native_optimizer_rules_applied: "none".to_string(),
            first_capability_rejection_projection: "none".to_string(),
            strict_bounded_envelope: "ready".to_string(),
            compiled_pipeline_status: "compiled".to_string(),
            compiled_pipeline_task_count: "1".to_string(),
            compiled_pipeline_edge_count: "0".to_string(),
            compiled_pipeline_barrier_count: "0".to_string(),
            compiled_pipeline_resource_count: "0".to_string(),
            compiled_pipeline_gate_count: "0".to_string(),
            post_native_execution_path: "native".to_string(),
            native_plan_fragment_count: "1".to_string(),
            host_datafusion_boundary_count: "0".to_string(),
            post_rule_planning_outcomes: "accepted_replaced".to_string(),
            bundle: "q01".to_string(),
        };
        assert_eq!(CURRENT_STATE_TSV_SCHEMA_VERSION, 1);
        assert_eq!(CurrentStateRow::COLUMNS.first(), Some(&"query_id"));
        assert_eq!(CurrentStateRow::COLUMNS.last(), Some(&"bundle"));
        assert_eq!(row.to_tsv_line().split('\t').count(), CurrentStateRow::COLUMNS.len());
    }
}
