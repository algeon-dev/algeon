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
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfSessionStateBuilderExt, lower_physical_plan_to_native,
};
use datafusion_nexus_bench::{
    TPCDS_QUERIES, TPCDS_TABLES, TpcdsQuery, discover_tpcds_table_layout, tpcds_native_optimizer_config,
    tpcds_native_report_session_config,
};

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
    let (ctx_gpu_pre, _) = build_gpu_context(&cfg.root, false).await?;
    let (ctx_gpu_post, gpu_native_rule) = build_gpu_context(&cfg.root, true).await?;

    let mut summary_rows: Vec<String> = Vec::new();
    for query in &cfg.queries {
        match triage_query(query, &cfg.out_dir, &ctx_cpu, &ctx_gpu_pre, &ctx_gpu_post, gpu_native_rule.as_ref()).await {
            Ok(summary) => summary_rows.push(summary),
            Err(error) => {
                summary_rows.push(format!("| {} | error: {} |", query.id, error));
                eprintln!("[tpcds_outlier_triage] {} failed: {}", query.id, error);
            },
        }
    }

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
) -> Result<(SessionContext, Option<Arc<CudfNativePhysicalOptimizerRule>>), Box<dyn Error>> {
    let mut builder =
        SessionStateBuilder::new_with_default_features().with_config(tpcds_native_report_session_config());
    let mut retained_rule = None;
    if with_native_rule {
        let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(tpcds_native_optimizer_config()));
        retained_rule = Some(Arc::clone(&rule));
        builder = builder.with_cudf_native_rule(rule);
    }
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, root).await?;
    Ok((ctx, retained_rule))
}

async fn register_tpcds_tables(ctx: &SessionContext, root: &Path) -> Result<(), Box<dyn Error>> {
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        let path = layout.anchor.to_str().ok_or_else(|| format!("non-utf8 path: {}", layout.anchor.display()))?;
        ctx.register_parquet(table, path, ParquetReadOptions::default()).await?;
    }
    Ok(())
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
) -> Result<String, Box<dyn Error>> {
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
    fs::write(dir.join("planning_report.gpu.txt"), format_planning_report(&lowering))?;

    let gpu_post_rule_plan = ctx_gpu_post.sql(query.sql).await?.create_physical_plan().await?;
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

    Ok(format!(
        "| {} | candidates={} rewritten={} rejected={} | [link](./{}/SUMMARY.md) |",
        qid,
        lowering.candidate_count(),
        lowering.rewritten_count(),
        lowering.rejected_count(),
        qid,
    ))
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
    out.push_str("| native_optimizer.gpu.postRule.txt | retained native optimizer diagnostics from the post-rule GPU physical optimizer |\n");
    out.push_str("| strict_bounded_contract.gpu.postRule.txt | strict-bounded admission evidence retained by the post-rule optimizer |\n");
    out.push_str(
        "| adaptive_pipeline.gpu.postRule.txt | adaptive admission evidence retained by the post-rule optimizer |\n",
    );
    out.push_str("| native_ir.gpu.txt | lowered `nexus_query_engine::QueryPlan` per rewritten candidate |\n");
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
