//! Empirical DataFusion-baseline contract matrix validator.
//!
//! Proves (with structured PlanningReport / plan node / result-parity evidence)
//! how algeon-datafusion retains executable DataFusion plans, including:
//! - default native planning retaining an executable DataFusion plan
//! - an explicit completed-plan requirement that rejects DataFusion execution
//! - a DataFusion baseline that deliberately installs no relational native rule
//! - Parquet scans that expose executable DataFusion plans
//! - the frozen fact that native runtime failures do not replay through DataFusion
//!
//! This is an explicit developer tool (`cargo run`); it is not a Cargo test target.

use std::{
    collections::BTreeMap,
    env,
    error::Error as StdError,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking, below_native_admission},
    native_report::{FinalPlanDisposition, NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadItem},
    planner::{AlgeonGpuExecutionMode, NativePlanningDiagnostics, plan_with_native_diagnostics},
    report::{CandidateOutcome, PlanningReport},
    session::{ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME, AlgeonSessionStateBuilderExt},
};
use arrow::{
    array::{Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::pretty::pretty_format_batches,
};
use datafusion::{
    error::{DataFusionError, Result as DataFusionResult},
    execution::{SessionStateBuilder, session_state::SessionState},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::{ExecutionPlan, displayable},
    prelude::{DataFrame, ParquetReadOptions, SessionConfig, SessionContext},
};
use serde::Serialize;
use tokio::task::spawn_blocking;

type ToolResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Session-level relational execution selection for this contract matrix.
///
/// A DataFusion baseline deliberately has no native rule. The strict case still
/// uses native planning, but rejects a completed plan that retains DataFusion
/// execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelationalExecution {
    NativeDefault,
    RequireNoDataFusionCpu,
    DataFusionBaseline,
}

impl RelationalExecution {
    const fn installs_native_rule(self) -> bool {
        !matches!(self, Self::DataFusionBaseline)
    }

    const fn requires_no_datafusion_cpu(self) -> bool {
        matches!(self, Self::RequireNoDataFusionCpu)
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::NativeDefault => "native_default",
            Self::RequireNoDataFusionCpu => "no_datafusion_cpu_requirement",
            Self::DataFusionBaseline => "datafusion_baseline",
        }
    }
}

#[derive(Debug, Serialize)]
struct CaseResult {
    id: String,
    category: String,
    ok: bool,
    detail: String,
    path: Option<String>,
    fragments: Option<usize>,
    host_boundaries: Option<usize>,
    plan_root: Option<String>,
    contains_native: Option<bool>,
    contains_data_source: Option<bool>,
    rewritten: Option<usize>,
    rejected_reasons: Vec<String>,
    parity_ok: Option<bool>,
    error_facts: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct MatrixReport {
    generated_at_unix_ms: u128,
    fixture_dir: String,
    conclusions: Vec<String>,
    cases: Vec<CaseResult>,
    pass_count: usize,
    fail_count: usize,
}

pub async fn run(args: Vec<String>) -> ToolResult<()> {
    let out_dir = args
        .iter()
        .position(|a| a == "--out-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/report"));
    fs::create_dir_all(&out_dir)?;

    let work = env::var("ALGEON_DATAFUSION_BASELINE_CONTRACT_WORK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/dlami/nvme/fixture/data/datafusion_baseline_contract_validation"));
    fs::create_dir_all(&work)?;
    let fixture = work.join("synthetic");
    fs::create_dir_all(&fixture)?;

    let bench = fixture.join("bench.parquet");
    let bench2 = fixture.join("bench2.parquet");
    write_synth_parquet(&bench, 0, 256)?;
    write_synth_parquet(&bench2, 1, 256)?;

    // Optional real TPC-H SF1 path for larger-data corner case.
    let tpch = PathBuf::from("/opt/dlami/nvme/fixture/data/tpch/sf1/parquet");

    let empty = fixture.join("empty.parquet");
    write_empty_parquet(&empty)?;
    let nullable = fixture.join("nullable.parquet");
    write_nullable_parquet(&nullable, 128)?;

    let backend = build_backend(64 * 1024 * 1024)?;

    let mut cases: Vec<CaseResult> = Vec::new();

    cases.push(
        run_parquet_case(
            "parquet_native_default_control_supported",
            RelationalExecution::NativeDefault,
            &bench,
            &bench2,
            "SELECT key, SUM(m00) AS s, COUNT(*) AS c FROM bench GROUP BY key ORDER BY key",
            Some(FinalPlanDisposition::Native),
            Some(1),
            true,
            false,
            &backend,
        )
        .await,
    );

    cases.push(
        run_parquet_case(
            "parquet_native_default_unsupported_approx_distinct_retains_datafusion",
            RelationalExecution::NativeDefault,
            &bench,
            &bench2,
            "SELECT t.key, APPROX_DISTINCT(t.name) AS d \
             FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
             GROUP BY t.key ORDER BY t.key",
            Some(FinalPlanDisposition::DataFusion),
            Some(0),
            true,
            false,
            &backend,
        )
        .await,
    );

    cases.push(
        run_parquet_case(
            "parquet_native_default_two_unsupported_branches_retain_datafusion",
            RelationalExecution::NativeDefault,
            &bench,
            &bench2,
            "SELECT * FROM ( \
               SELECT 'a' AS src, t.key, APPROX_DISTINCT(t.name) AS d \
               FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
               GROUP BY t.key \
               UNION ALL \
               SELECT 'b' AS src, u.key, APPROX_DISTINCT(u.name) AS d \
               FROM (SELECT b2.key, b2.name FROM bench2 b2 JOIN bench b3 ON b2.key = b3.key WHERE b3.m00 >= 2.0) u \
               GROUP BY u.key) v ORDER BY src, key",
            Some(FinalPlanDisposition::DataFusion),
            Some(0),
            true,
            false,
            &backend,
        )
        .await,
    );

    cases.push(
        run_parquet_case(
            "parquet_native_default_unsupported_expression_retains_datafusion",
            RelationalExecution::NativeDefault,
            &bench,
            &bench2,
            "SELECT t.key, SUM(TRY_CAST(t.name AS DOUBLE)) AS s \
             FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
             GROUP BY t.key ORDER BY t.key",
            Some(FinalPlanDisposition::DataFusion),
            Some(0),
            true,
            false,
            &backend,
        )
        .await,
    );

    cases.push(
        run_parquet_case(
            "parquet_native_default_median_retains_executable_plan",
            RelationalExecution::NativeDefault,
            &bench,
            &bench2,
            // MEDIAN is often unsupported on native in distributed shapes; the
            // default native rule must retain an executable plan.
            "SELECT key, approx_percentile_cont(m00, 0.5) AS p50 FROM bench GROUP BY key ORDER BY key",
            None,
            None,
            true,
            false,
            &backend,
        )
        .await,
    );

    cases.push(
        run_parquet_case(
            "parquet_no_datafusion_cpu_requirement_rejects_datafusion_plan",
            RelationalExecution::RequireNoDataFusionCpu,
            &bench,
            &bench2,
            "SELECT t.key, SUM(TRY_CAST(t.name AS DOUBLE)) AS s \
             FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
             GROUP BY t.key ORDER BY t.key",
            None,
            None,
            false,
            false,
            &backend,
        )
        .await,
    );

    cases.push(
        run_parquet_case(
            "parquet_datafusion_baseline_supported_without_native_rule",
            RelationalExecution::DataFusionBaseline,
            &bench,
            &bench2,
            "SELECT key, SUM(m00) AS s, COUNT(*) AS c FROM bench GROUP BY key ORDER BY key",
            None,
            None,
            true,
            false,
            &backend,
        )
        .await,
    );

    // Real TPC-H SF1 unsupported-expression case.
    if tpch.join("lineitem.parquet").is_file() {
        cases.push(run_tpch_case(&tpch, &backend).await);
    } else {
        cases.push(CaseResult {
            id: "tpch_sf1_unavailable".into(),
            category: "parquet_fixture".into(),
            ok: false,
            detail: format!("missing {}", tpch.join("lineitem.parquet").display()),
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        });
    }

    // Preserve the report ordering while deferring the distinct 1-byte backend
    // until the ordinary process backend has closed completely.
    let runtime_failure_case_index = cases.len();

    // Corner: empty parquet file
    cases.push(run_empty_parquet_corner(&empty, &backend).await);

    // Corner: null-heavy + stringy keys
    cases.push(
        run_single_table_case(
            "parquet_native_default_nullable_approx_distinct",
            RelationalExecution::NativeDefault,
            "n",
            &nullable,
            "SELECT k, APPROX_DISTINCT(s) AS d FROM n GROUP BY k ORDER BY k",
            true,
            &backend,
        )
        .await,
    );

    cases = finish_backend_scope(backend, Ok(cases)).await?;
    let runtime_failure_case = run_runtime_no_rescue(&bench).await;
    cases.insert(runtime_failure_case_index, runtime_failure_case);

    let pass_count = cases.iter().filter(|c| c.ok).count();
    let fail_count = cases.len() - pass_count;

    let conclusions = vec![
        "DataFusion retention is a planning-time completed-plan outcome, never a native-failure replay.".into(),
        "Default native planning may retain an executable DataFusion plan when a relational candidate is unsupported.".into(),
        "The no_datafusion_cpu requirement rejects a completed plan with DataFusion execution as FinalPlanRequirementUnsatisfied.".into(),
        "A datafusion_baseline session deliberately installs no relational native optimizer rule and executes through DataFusion.".into(),
        "Local Parquet scans expose executable DataFusion plans when native lowering is not selected.".into(),
        "After native execution starts, cpu_fallback_after_native_start=false means that attempt is terminal and is not restarted through DataFusion.".into(),
    ];

    let report = MatrixReport {
        generated_at_unix_ms: now_ms(),
        fixture_dir: fixture.display().to_string(),
        conclusions,
        cases,
        pass_count,
        fail_count,
    };

    let json_path = out_dir.join("datafusion_baseline_contract_matrix.json");
    let md_path = out_dir.join("datafusion_baseline_contract_matrix.md");
    fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
    fs::write(&md_path, render_markdown(&report))?;

    println!("wrote {}", json_path.display());
    println!("wrote {}", md_path.display());
    println!("pass={pass_count} fail={fail_count} total={}", report.cases.len());
    for c in &report.cases {
        let mark = if c.ok { "PASS" } else { "FAIL" };
        println!(
            "  [{mark}] {} path={:?} frag={:?} host={:?} | {}",
            c.id, c.path, c.fragments, c.host_boundaries, c.detail
        );
    }

    if fail_count > 0 { Err(format!("{fail_count} case(s) failed; see {}", md_path.display()).into()) } else { Ok(()) }
}

fn render_markdown(report: &MatrixReport) -> String {
    let mut out = String::new();
    out.push_str("# DataFusion baseline contract validation matrix\n\n");
    out.push_str(&format!(
        "- generated_at_unix_ms: {}\n- fixture_dir: `{}`\n- pass: **{}** / fail: **{}** / total: {}\n\n",
        report.generated_at_unix_ms,
        report.fixture_dir,
        report.pass_count,
        report.fail_count,
        report.cases.len()
    ));
    out.push_str("## Conclusions\n\n");
    for c in &report.conclusions {
        out.push_str(&format!("- {c}\n"));
    }
    out.push_str("\n## Case matrix\n\n");
    out.push_str("| id | category | ok | path | fragments | host_boundaries | plan_root | parity | detail |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for c in &report.cases {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            c.id,
            c.category,
            c.ok,
            c.path.as_deref().unwrap_or("-"),
            c.fragments.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
            c.host_boundaries.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
            c.plan_root.as_deref().unwrap_or("-"),
            match c.parity_ok {
                Some(true) => "ok",
                Some(false) => "MISMATCH",
                None => "-",
            },
            c.detail.replace('|', "\\|").replace('\n', " "),
        ));
    }
    out.push_str("\n## Structured facts (failures / leaf-guard / runtime)\n\n");
    for c in &report.cases {
        if c.error_facts.is_empty() {
            continue;
        }
        out.push_str(&format!("### `{}`\n\n```json\n", c.id));
        out.push_str(
            &serde_json::to_string_pretty(&c.error_facts).expect("structured error facts must be serializable"),
        );
        out.push_str("\n```\n\n");
    }
    out.push_str("## Terminology mapping ( empirically observed )\n\n");
    out.push_str("| Term | Observed meaning |\n|---|---|\n");
    out.push_str(
        "| `native_default` | Install the default native rule. Unsupported relational candidates retain their executable DataFusion plan. |\n",
    );
    out.push_str(
        "| `no_datafusion_cpu_requirement` | Install the native rule and reject the completed plan unless it has no DataFusion CPU execution. |\n",
    );
    out.push_str(
        "| `datafusion_baseline` | Install no relational native rule; plan and execute solely with DataFusion. |\n",
    );
    out.push_str(
        "| Native runtime replay | Does **not** exist (`cpu_fallback_after_native_start=false` means the started native attempt is terminal). |\n",
    );
    out
}

#[expect(clippy::too_many_arguments, reason = "validation harness parameters are independent measurement axes")]
async fn run_parquet_case(
    id: &str,
    selection: RelationalExecution,
    bench: &Path,
    bench2: &Path,
    sql: &str,
    expected_path: Option<FinalPlanDisposition>,
    expected_min_fragments: Option<usize>,
    expect_execute_ok: bool,
    require_host_boundary_if_partial: bool,
    backend: &AlgeonGpuBackend,
) -> CaseResult {
    match run_parquet_case_inner(
        id,
        selection,
        bench,
        bench2,
        sql,
        expected_path,
        expected_min_fragments,
        expect_execute_ok,
        require_host_boundary_if_partial,
        backend,
    )
    .await
    {
        Ok(mut r) => {
            r.category = selection.as_str().into();
            r
        },
        Err(e) => CaseResult {
            id: id.into(),
            category: selection.as_str().into(),
            ok: false,
            detail: format!("internal error: {e:#}"),
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        },
    }
}

#[expect(clippy::too_many_arguments, reason = "validation harness parameters are independent measurement axes")]
async fn run_parquet_case_inner(
    id: &str,
    selection: RelationalExecution,
    bench: &Path,
    bench2: &Path,
    sql: &str,
    expected_path: Option<FinalPlanDisposition>,
    expected_min_fragments: Option<usize>,
    expect_execute_ok: bool,
    require_host_boundary_if_partial: bool,
    backend: &AlgeonGpuBackend,
) -> ToolResult<CaseResult> {
    let gpu_ctx = relational_ctx(selection, backend)?;
    register_pair(&gpu_ctx, bench, bench2).await?;

    let plan_result = plan_query(gpu_ctx.sql(sql).await?).await;

    if let Err(error) = plan_result {
        let facts = algeon_facts_from_df_error(&error);
        let is_strict_requirement_rejection =
            matches!(error.find_algeon_error().map(|e| e.code()), Some(ErrorCode::FinalPlanRequirementUnsatisfied))
                && facts.get("requirement").map(String::as_str) == Some("no_datafusion_cpu");
        let ok = !expect_execute_ok && selection.requires_no_datafusion_cpu() && is_strict_requirement_rejection;
        return Ok(CaseResult {
            id: id.into(),
            category: String::new(),
            ok,
            detail: if ok {
                format!("planning rejected as expected: {error}")
            } else {
                format!("unexpected planning error: {error}")
            },
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: facts.get("fallback_reason").cloned().into_iter().collect(),
            parity_ok: None,
            error_facts: facts,
        });
    }

    let (plan, diagnostics) = plan_result?;
    let native_rule_installed = diagnostics.is_some();
    let (rewritten, rejected_reasons) =
        diagnostics.as_ref().map_or((0, Vec::new()), |diagnostics| outcome_stats(&diagnostics.report));
    let report = diagnostics.map(|diagnostics| planning_report(id, &plan, diagnostics));
    let row = report.as_ref().and_then(|r| r.rows.first());

    let path = row.and_then(|r| r.final_disposition);
    let fragments = row.and_then(|r| r.native_plan_fragment_count);
    let host_boundaries = row.and_then(|r| r.host_datafusion_boundary_count);
    let plan_root = Some(below_native_admission(&plan).name().to_string());
    let contains_native = Some(plan_contains_native(plan.as_ref()));
    let contains_data_source = Some(plan_contains_name(plan.as_ref(), "DataSourceExec"));

    let mut detail = format!(
        "plan_root={} path={path:?} frag={fragments:?} host={host_boundaries:?} rewritten={rewritten} rejected={rejected_reasons:?}",
        below_native_admission(&plan).name()
    );
    let mut ok = true;

    if selection.installs_native_rule() != native_rule_installed {
        ok = false;
        detail.push_str("; native rule installation did not match the selected execution mode");
    }
    if selection == RelationalExecution::DataFusionBaseline
        && (contains_native == Some(true) || contains_data_source != Some(true))
    {
        ok = false;
        detail.push_str("; DataFusion baseline must have no native plan and retain DataSourceExec");
    }

    if let Some(expected) = expected_path
        && path != Some(expected)
    {
        ok = false;
        detail.push_str(&format!("; expected path {expected:?}"));
    }
    if let Some(min_f) = expected_min_fragments
        && fragments.unwrap_or(0) < min_f
    {
        ok = false;
        detail.push_str(&format!("; fragments {} < {min_f}", fragments.unwrap_or(0)));
    }
    if require_host_boundary_if_partial && path == Some(FinalPlanDisposition::Mixed) && host_boundaries.unwrap_or(0) < 1
    {
        ok = false;
        detail.push_str("; mixed plan without a DataFusion boundary");
    }
    // A DataFusion plan retained by default native planning cannot contain native fragments.
    if path == Some(FinalPlanDisposition::DataFusion) && fragments.unwrap_or(0) != 0 {
        ok = false;
        detail.push_str("; DataFusion path must have zero fragments");
    }

    let mut parity_ok = None;
    if expect_execute_ok {
        let gpu_batches = match gpu_ctx.sql(sql).await?.collect().await {
            Ok(b) => b,
            Err(e) => {
                ok = false;
                detail.push_str(&format!("; gpu collect failed: {e}"));
                return Ok(CaseResult {
                    id: id.into(),
                    category: String::new(),
                    ok,
                    detail,
                    path: path.map(|p| format!("{p:?}")),
                    fragments,
                    host_boundaries,
                    plan_root,
                    contains_native,
                    contains_data_source,
                    rewritten: Some(rewritten),
                    rejected_reasons,
                    parity_ok: Some(false),
                    error_facts: BTreeMap::new(),
                });
            },
        };
        let datafusion_ctx = plain_datafusion_ctx();
        register_pair(&datafusion_ctx, bench, bench2).await?;
        let datafusion_batches = datafusion_ctx.sql(sql).await?.collect().await?;
        let gpu_text = pretty_format_batches(&gpu_batches)?.to_string();
        let datafusion_text = pretty_format_batches(&datafusion_batches)?.to_string();
        let equals = gpu_text == datafusion_text;
        parity_ok = Some(equals);
        if !equals {
            ok = false;
            detail.push_str("; result parity mismatch vs plain DataFusion");
        } else {
            detail.push_str("; parity=ok");
        }
    }

    Ok(CaseResult {
        id: id.into(),
        category: String::new(),
        ok,
        detail,
        path: path.map(|p| format!("{p:?}")),
        fragments,
        host_boundaries,
        plan_root,
        contains_native,
        contains_data_source,
        rewritten: Some(rewritten),
        rejected_reasons,
        parity_ok,
        error_facts: BTreeMap::new(),
    })
}

async fn run_single_table_case(
    id: &str,
    selection: RelationalExecution,
    table: &str,
    path: &Path,
    sql: &str,
    expect_execute_ok: bool,
    backend: &AlgeonGpuBackend,
) -> CaseResult {
    let result = async {
        let gpu_ctx = relational_ctx(selection, backend)?;
        gpu_ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default()).await?;
        let (plan, diagnostics) = plan_query(gpu_ctx.sql(sql).await?).await?;
        let (rewritten, rejected_reasons) =
            diagnostics.as_ref().map_or((0, Vec::new()), |diagnostics| outcome_stats(&diagnostics.report));
        let report = diagnostics.map(|diagnostics| planning_report(id, &plan, diagnostics));
        let row = report.as_ref().and_then(|r| r.rows.first());
        let mut detail = format!(
            "path={:?} frag={:?} host={:?} rewritten={rewritten} rejected={rejected_reasons:?}",
            row.and_then(|r| r.final_disposition),
            row.and_then(|r| r.native_plan_fragment_count),
            row.and_then(|r| r.host_datafusion_boundary_count),
        );
        let mut ok = true;
        let mut parity_ok = None;
        if expect_execute_ok {
            let gpu_batches = gpu_ctx.sql(sql).await?.collect().await?;
            let datafusion_ctx = plain_datafusion_ctx();
            datafusion_ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default()).await?;
            let datafusion_batches = datafusion_ctx.sql(sql).await?.collect().await?;
            let equals = pretty_format_batches(&gpu_batches)?.to_string()
                == pretty_format_batches(&datafusion_batches)?.to_string();
            parity_ok = Some(equals);
            if !equals {
                ok = false;
                detail.push_str("; parity mismatch");
            } else {
                detail.push_str("; parity=ok");
            }
        }
        ToolResult::Ok(CaseResult {
            id: id.into(),
            category: selection.as_str().into(),
            ok,
            detail,
            path: row.map(|r| format!("{:?}", r.final_disposition)),
            fragments: row.and_then(|r| r.native_plan_fragment_count),
            host_boundaries: row.and_then(|r| r.host_datafusion_boundary_count),
            plan_root: Some(below_native_admission(&plan).name().to_string()),
            contains_native: Some(plan_contains_native(plan.as_ref())),
            contains_data_source: Some(plan_contains_name(plan.as_ref(), "DataSourceExec")),
            rewritten: Some(rewritten),
            rejected_reasons,
            parity_ok,
            error_facts: BTreeMap::new(),
        })
    }
    .await;

    match result {
        Ok(c) => c,
        Err(e) => CaseResult {
            id: id.into(),
            category: selection.as_str().into(),
            ok: false,
            detail: format!("{e:#}"),
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        },
    }
}

async fn run_tpch_case(tpch: &Path, backend: &AlgeonGpuBackend) -> CaseResult {
    let id = "tpch_sf1_native_default_approx_distinct_join";
    // Highly selective filter keeps SF1 manageable while exercising default
    // native planning's executable-plan retention.
    let sql = "SELECT l.l_returnflag, APPROX_DISTINCT(o.o_orderkey) AS d                FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey                WHERE l.l_shipdate = DATE '1998-01-01'                GROUP BY l.l_returnflag ORDER BY l.l_returnflag";
    let result = async {
        let gpu_ctx = relational_ctx(RelationalExecution::NativeDefault, backend)?;
        for t in ["lineitem", "orders"] {
            let p = tpch.join(format!("{t}.parquet"));
            gpu_ctx.register_parquet(t, p.to_str().unwrap(), ParquetReadOptions::default()).await?;
        }
        let (state, logical) = gpu_ctx.sql(sql).await?.into_parts();
        let (plan, diagnostics) = plan_with_native_diagnostics(&state, &logical).await?;
        let (rewritten, rejected_reasons) = outcome_stats(&diagnostics.report);
        let report = planning_report(id, &plan, diagnostics);
        let row = report.rows.first().ok_or("no planning row")?;
        let dump_dir = PathBuf::from("/tmp/datafusion_baseline_contract_validation/plans");
        fs::create_dir_all(&dump_dir)?;
        fs::write(dump_dir.join(format!("{id}.plan.txt")), displayable(plan.as_ref()).indent(false).to_string())?;

        let path = row.final_disposition;
        let planning_ok = matches!(path, Some(FinalPlanDisposition::DataFusion) | Some(FinalPlanDisposition::Native));

        // Prefer full parity; if GPU collect fails, retain planning evidence and classify.
        let mut detail = format!(
            "path={path:?} frag={:?} host={:?} rewritten={rewritten} rejected={rejected_reasons:?}",
            row.native_plan_fragment_count, row.host_datafusion_boundary_count
        );
        let (ok, parity_ok, error_facts) = match gpu_ctx.sql(sql).await?.collect().await {
            Ok(gpu_batches) => {
                let datafusion_ctx = plain_datafusion_ctx();
                for t in ["lineitem", "orders"] {
                    let p = tpch.join(format!("{t}.parquet"));
                    datafusion_ctx.register_parquet(t, p.to_str().unwrap(), ParquetReadOptions::default()).await?;
                }
                let datafusion_batches = datafusion_ctx.sql(sql).await?.collect().await?;
                let equals = pretty_format_batches(&gpu_batches)?.to_string()
                    == pretty_format_batches(&datafusion_batches)?.to_string();
                detail.push_str(&format!("; parity={equals}"));
                (planning_ok && equals, Some(equals), BTreeMap::new())
            },
            Err(e) => {
                detail.push_str(&format!("; collect_failed={e}"));
                (false, None, algeon_facts_from_df_error(&e))
            },
        };

        ToolResult::Ok(CaseResult {
            id: id.into(),
            category: "parquet_tpch_fixture".into(),
            ok,
            detail,
            path: Some(format!("{path:?}")),
            fragments: row.native_plan_fragment_count,
            host_boundaries: row.host_datafusion_boundary_count,
            plan_root: Some(below_native_admission(&plan).name().to_string()),
            contains_native: Some(plan_contains_native(plan.as_ref())),
            contains_data_source: Some(plan_contains_name(plan.as_ref(), "DataSourceExec")),
            rewritten: Some(rewritten),
            rejected_reasons,
            parity_ok,
            error_facts,
        })
    }
    .await;
    match result {
        Ok(c) => c,
        Err(e) => CaseResult {
            id: id.into(),
            category: "parquet_tpch_fixture".into(),
            ok: false,
            detail: format!("{e:#}"),
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        },
    }
}

async fn run_runtime_no_rescue(bench: &Path) -> CaseResult {
    let id = "runtime_failure_no_datafusion_replay";
    match run_runtime_no_rescue_constrained(bench).await {
        Ok(c) => c,
        Err(e) => CaseResult {
            id: id.into(),
            category: "runtime_failure".into(),
            ok: false,
            detail: format!("{e:#}"),
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        },
    }
}

async fn run_runtime_no_rescue_constrained(bench: &Path) -> ToolResult<CaseResult> {
    let id = "runtime_failure_no_datafusion_replay";
    // default_query_limit_bytes=1 forces a memory reservation failure after native start.
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
        .with_default_query_limit_bytes(1);
    let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder()
        .with_native_execution_config(AlgeonNativeExecutionConfig::new(SourceChunking::try_new(1024 * 1024, 1, 1, 1)?))
        .queue_policy(AlgeonGpuQueuePolicy::default())
        .devices([device])
        .build()?;
    let work = async {
    let state = SessionStateBuilder::new_with_default_features().with_config(direct_cfg());
    let state = backend.install_on(state)?.build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", bench.to_str().unwrap(), ParquetReadOptions::default()).await?;
    let sql = "SELECT key, SUM(m00) AS s, SUM(payload) AS p FROM bench GROUP BY key";
    let (state, logical) = ctx.sql(sql).await?.into_parts();
    let (plan, diagnostics) = plan_with_native_diagnostics(&state, &logical).await?;
    let report = planning_report(id, &plan, diagnostics);
    let row = report.rows.first();
    if !plan_contains_native(plan.as_ref()) {
        return Ok(CaseResult {
            id: id.into(),
            category: "runtime_failure".into(),
            ok: false,
            detail: format!(
                "constrained plan not native: root={} path={:?}",
                below_native_admission(&plan).name(),
                row.map(|r| r.final_disposition)
            ),
            path: row.map(|r| format!("{:?}", r.final_disposition)),
            fragments: row.and_then(|r| r.native_plan_fragment_count),
            host_boundaries: row.and_then(|r| r.host_datafusion_boundary_count),
            plan_root: Some(below_native_admission(&plan).name().to_string()),
            contains_native: Some(false),
            contains_data_source: Some(plan_contains_name(plan.as_ref(), "DataSourceExec")),
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        });
    }
    let batches = match ctx.sql(sql).await?.collect().await {
        Ok(batches) => batches,
        Err(err) => {
            let facts = algeon_facts_from_df_error(&err);
            let no_rescue = facts.get("cpu_fallback_after_native_start").map(String::as_str) == Some("false");
            let stage = facts.get("native_datafusion_stage").map(String::as_str).unwrap_or("");
            let ok = no_rescue && (stage.contains("runtime") || facts.contains_key("error_code"));
            return Ok(CaseResult {
                id: id.into(),
                category: "runtime_failure".into(),
                ok,
                detail: format!(
                    "native plan started then failed without DataFusion replay; no_rescue={no_rescue} stage={stage:?} err={err}"
                ),
                path: row.map(|r| format!("{:?}", r.final_disposition)),
                fragments: row.and_then(|r| r.native_plan_fragment_count),
                host_boundaries: row.and_then(|r| r.host_datafusion_boundary_count),
                plan_root: Some(below_native_admission(&plan).name().to_string()),
                contains_native: Some(true),
                contains_data_source: Some(false),
                rewritten: None,
                rejected_reasons: vec![],
                parity_ok: None,
                error_facts: facts,
            });
        },
    };
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let ok = !batches.is_empty() && rows > 0;
    Ok(CaseResult {
        id: id.into(),
        category: "runtime_failure".into(),
        ok,
        detail: format!(
            "constrained admitted execution completed natively with no proof failure or replay; batches={} rows={rows}",
            batches.len()
        ),
        path: row.map(|r| format!("{:?}", r.final_disposition)),
        fragments: row.and_then(|r| r.native_plan_fragment_count),
        host_boundaries: row.and_then(|r| r.host_datafusion_boundary_count),
        plan_root: Some(below_native_admission(&plan).name().to_string()),
        contains_native: Some(true),
        contains_data_source: Some(false),
        rewritten: None,
        rejected_reasons: vec![],
        parity_ok: None,
        error_facts: BTreeMap::new(),
    })
    }
    .await;
    finish_backend_scope(backend, work).await
}

async fn run_empty_parquet_corner(path: &Path, backend: &AlgeonGpuBackend) -> CaseResult {
    let id = "parquet_empty_file_count_star";
    let sql = "SELECT COUNT(*) AS c FROM empty";
    let result = async {
        let gpu_ctx = relational_ctx(RelationalExecution::NativeDefault, backend)?;
        gpu_ctx.register_parquet("empty", path.to_str().unwrap(), ParquetReadOptions::default()).await?;
        let (plan, diagnostics) = plan_query(gpu_ctx.sql(sql).await?).await?;
        let (rewritten, rejected_reasons) =
            diagnostics.as_ref().map_or((0, vec![]), |diagnostics| outcome_stats(&diagnostics.report));
        let report = diagnostics.map(|diagnostics| planning_report(id, &plan, diagnostics));
        let row = report.as_ref().and_then(|r| r.rows.first());
        let mut detail = format!(
            "path={:?} frag={:?} host={:?} rewritten={rewritten} rejected={rejected_reasons:?}",
            row.and_then(|r| r.final_disposition),
            row.and_then(|r| r.native_plan_fragment_count),
            row.and_then(|r| r.host_datafusion_boundary_count)
        );
        let (ok, parity_ok, error_facts) = match gpu_ctx.sql(sql).await?.collect().await {
            Ok(gpu_batches) => {
                let datafusion_ctx = plain_datafusion_ctx();
                datafusion_ctx.register_parquet("empty", path.to_str().unwrap(), ParquetReadOptions::default()).await?;
                let datafusion_batches = datafusion_ctx.sql(sql).await?.collect().await?;
                let equals = pretty_format_batches(&gpu_batches)?.to_string()
                    == pretty_format_batches(&datafusion_batches)?.to_string();
                detail.push_str(&format!("; parity={equals}"));
                (equals, Some(equals), BTreeMap::new())
            },
            Err(e) => {
                let error_facts = algeon_facts_from_df_error(&e);
                // Empty-input native execution may fail closed. That is valid evidence as
                // long as it does not silently replay through DataFusion after native start
                // of a different plan id, and the planning report is populated.
                let no_rescue = error_facts.get("cpu_fallback_after_native_start").map(String::as_str) == Some("false");
                let ok = if row.is_some() && (no_rescue || error_facts.contains_key("error_code")) {
                    detail.push_str(&format!(
                        "; empty native collect failed without DataFusion replay (contractual): {e}"
                    ));
                    true
                } else if row.is_some()
                    && row.and_then(|r| r.final_disposition) == Some(FinalPlanDisposition::DataFusion)
                {
                    detail.push_str(&format!("; unexpected DataFusion-path collect failure: {e}"));
                    false
                } else {
                    detail.push_str(&format!("; collect failed: {e}"));
                    false
                };
                (ok, None, error_facts)
            },
        };
        ToolResult::Ok(CaseResult {
            id: id.into(),
            category: "parquet_corner".into(),
            ok,
            detail,
            path: row.map(|r| format!("{:?}", r.final_disposition)),
            fragments: row.and_then(|r| r.native_plan_fragment_count),
            host_boundaries: row.and_then(|r| r.host_datafusion_boundary_count),
            plan_root: Some(below_native_admission(&plan).name().to_string()),
            contains_native: Some(plan_contains_native(plan.as_ref())),
            contains_data_source: Some(plan_contains_name(plan.as_ref(), "DataSourceExec")),
            rewritten: Some(rewritten),
            rejected_reasons,
            parity_ok,
            error_facts,
        })
    }
    .await;
    match result {
        Ok(c) => c,
        Err(e) => CaseResult {
            id: id.into(),
            category: "parquet_corner".into(),
            ok: false,
            detail: format!("{e:#}"),
            path: None,
            fragments: None,
            host_boundaries: None,
            plan_root: None,
            contains_native: None,
            contains_data_source: None,
            rewritten: None,
            rejected_reasons: vec![],
            parity_ok: None,
            error_facts: BTreeMap::new(),
        },
    }
}

fn build_backend(default_query_limit_bytes: u64) -> ToolResult<AlgeonGpuBackend> {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
        .with_default_query_limit_bytes(default_query_limit_bytes);
    let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
    Ok(AlgeonGpuBackend::builder()
        .with_native_execution_config(AlgeonNativeExecutionConfig::default())
        .queue_policy(AlgeonGpuQueuePolicy::default())
        .devices([device])
        .build()?)
}

fn relational_ctx(selection: RelationalExecution, backend: &AlgeonGpuBackend) -> ToolResult<SessionContext> {
    let builder = SessionStateBuilder::new_with_default_features().with_config(direct_cfg());
    let state = match selection {
        RelationalExecution::DataFusionBaseline => builder.build(),
        RelationalExecution::NativeDefault => backend.install_on(builder)?.build(),
        RelationalExecution::RequireNoDataFusionCpu => builder
            .with_algeon_native(
                backend.optimizer_config().clone().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
            )
            .build(),
    };
    Ok(SessionContext::new_with_state(state))
}

async fn finish_backend_scope<T>(backend: AlgeonGpuBackend, work: ToolResult<T>) -> ToolResult<T> {
    let cleanup = spawn_blocking(move || backend.close_and_wait(Duration::from_secs(30))).await;
    let cleanup: ToolResult<()> = match cleanup {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(Box::new(error)),
        Err(error) => Err(format!("backend cleanup task failed to join: {error}").into()),
    };
    match (work, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(work_error), Err(cleanup_error)) => {
            Err(format!("validation work failed: {work_error:#}; backend cleanup also failed: {cleanup_error:#}")
                .into())
        },
    }
}

fn direct_cfg() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn plain_datafusion_ctx() -> SessionContext {
    SessionContext::new_with_state(SessionStateBuilder::new_with_default_features().with_config(direct_cfg()).build())
}

async fn register_pair(ctx: &SessionContext, bench: &Path, bench2: &Path) -> ToolResult<()> {
    ctx.register_parquet("bench", bench.to_str().unwrap(), ParquetReadOptions::default()).await?;
    ctx.register_parquet("bench2", bench2.to_str().unwrap(), ParquetReadOptions::default()).await?;
    Ok(())
}

fn has_final_native_rule(state: &SessionState) -> bool {
    state.physical_optimizers().iter().any(|rule| rule.name() == ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME)
}

/// Plans `df`; the diagnostics are `None` for a session without the native rule.
async fn plan_query(df: DataFrame) -> DataFusionResult<(Arc<dyn ExecutionPlan>, Option<NativePlanningDiagnostics>)> {
    let (state, logical) = df.into_parts();
    if !has_final_native_rule(&state) {
        return Ok((state.create_physical_plan(&logical).await?, None));
    }
    let (plan, diagnostics) = plan_with_native_diagnostics(&state, &logical).await?;
    Ok((plan, Some(diagnostics)))
}

fn planning_report(
    id: &str,
    plan: &Arc<dyn ExecutionPlan>,
    diagnostics: NativePlanningDiagnostics,
) -> NativeDataFusionPlanningReport {
    NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        id,
        Arc::clone(plan),
        diagnostics,
    )])
}

fn outcome_stats(report: &PlanningReport) -> (usize, Vec<String>) {
    let mut selected = 0;
    let mut not_selected = Vec::new();
    for outcome in report.outcomes() {
        match outcome {
            CandidateOutcome::Selected { .. } => selected += 1,
            CandidateOutcome::NotSupported { capability_reason, .. } => {
                not_selected.push(capability_reason.code().to_string())
            },
            CandidateOutcome::NotSelectedByCost { cost_reason, .. } => {
                not_selected.push(cost_reason.code().to_string())
            },
        }
    }
    (selected, not_selected)
}

fn plan_contains_native(plan: &dyn ExecutionPlan) -> bool {
    let name = plan.name();
    name.starts_with("AlgeonNativeExec")
        || name.starts_with("AlgeonNativeComposedExec")
        || name.starts_with("AlgeonNativeGpuProducerExec")
        || name.starts_with("AlgeonNative")
        || plan.children().iter().any(|c| plan_contains_native(c.as_ref()))
}

fn plan_contains_name(plan: &dyn ExecutionPlan, name: &str) -> bool {
    plan.name() == name
        || plan.name().strip_prefix(name).is_some_and(|s| s.is_empty() || s.starts_with('('))
        || plan.children().iter().any(|c| plan_contains_name(c.as_ref(), name))
}

fn algeon_facts_from_df_error(error: &DataFusionError) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("display".into(), error.to_string());
    let debug = format!("{error:?}");
    out.insert("debug".into(), debug.chars().take(2000).collect());
    if let Some(algeon) = error.find_algeon_error() {
        out.insert("error_code".into(), format!("{:?}", algeon.code()));
        out.insert("source_error_status".into(), algeon.status().as_str().to_string());
        out.insert("retry_advice".into(), "unavailable_no_sealed_attempt_report".to_string());
        out.insert("status".into(), format!("{:?}", algeon.status()));
        out.insert("kind".into(), format!("{:?}", algeon.kind()));
        out.insert("operation".into(), algeon.operation().to_string());
        out.insert("message".into(), algeon.message().to_string());
        for (k, v) in algeon.facts().iter() {
            out.insert(k.to_string(), v.to_string());
        }
        // Contract markers must also be discoverable via fact_value on the typed error.
        if let Some(v) = algeon.fact_value("cpu_fallback_after_native_start") {
            out.insert("cpu_fallback_after_native_start".into(), v.to_string());
        }
    } else {
        // Walk External/Context wrappers manually for stubborn chains.
        walk_df_error(error, &mut out, 0);
    }
    out
}

fn walk_df_error(error: &DataFusionError, out: &mut BTreeMap<String, String>, depth: usize) {
    if depth > 8 {
        return;
    }
    match error {
        DataFusionError::Context(ctx, inner) => {
            out.entry(format!("context_{depth}")).or_insert_with(|| ctx.clone());
            walk_df_error(inner, out, depth + 1);
        },
        DataFusionError::External(boxed) => {
            if let Some(algeon) = boxed.downcast_ref::<Error>() {
                out.insert("error_code".into(), format!("{:?}", algeon.code()));
                out.insert("source_error_status".into(), algeon.status().as_str().to_string());
                out.insert("retry_advice".into(), "unavailable_no_sealed_attempt_report".to_string());
                out.insert("status".into(), format!("{:?}", algeon.status()));
                out.insert("kind".into(), format!("{:?}", algeon.kind()));
                out.insert("operation".into(), algeon.operation().to_string());
                out.insert("message".into(), algeon.message().to_string());
                for (k, v) in algeon.facts().iter() {
                    out.insert(k.to_string(), v.to_string());
                }
            } else {
                out.insert(format!("external_{depth}"), boxed.to_string());
                // try source chain
                let mut src = (boxed.as_ref() as &dyn StdError).source();
                let mut i = 0;
                while let Some(s) = src {
                    out.insert(format!("source_{depth}_{i}"), s.to_string());
                    // string-scan for contract fact when typed downcast fails
                    let text = s.to_string();
                    if text.contains("cpu_fallback_after_native_start") {
                        out.insert("cpu_fallback_after_native_start_mentioned".into(), "true".into());
                    }
                    src = s.source();
                    i += 1;
                    if i > 6 {
                        break;
                    }
                }
            }
        },
        other => {
            let text = format!("{other:?}");
            out.entry(format!("other_{depth}")).or_insert_with(|| text.chars().take(500).collect());
        },
    }
}

fn write_synth_parquet(path: &Path, salt: i64, rows: i64) -> ToolResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let keys: Vec<i32> = (0..rows).map(|i| (i % 16) as i32).collect();
    let payloads: Vec<i64> = (0..rows).map(|i| (i * 3 + salt * 7) % 40).collect();
    let m00: Vec<f64> = (0..rows).map(|i| ((i + salt) as f64) / 4.0).collect();
    let names: Vec<String> = (0..rows).map(|i| format!("name_{}", (i + salt * 11) % 23)).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int64Array::from(payloads)),
            Arc::new(Float64Array::from(m00)),
            Arc::new(StringArray::from(names)),
        ],
    )?;
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_empty_parquet(path: &Path) -> ToolResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let batch = RecordBatch::new_empty(Arc::clone(&schema));
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_nullable_parquet(path: &Path, rows: i64) -> ToolResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("x", DataType::Float64, true),
    ]));
    let ks: Vec<Option<i32>> = (0..rows).map(|i| if i % 7 == 0 { None } else { Some((i % 5) as i32) }).collect();
    let ss: Vec<Option<String>> =
        (0..rows).map(|i| if i % 11 == 0 { None } else { Some(format!("s{}", i % 13)) }).collect();
    let xs: Vec<Option<f64>> = (0..rows).map(|i| if i % 9 == 0 { None } else { Some(i as f64 * 0.25) }).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(ks)), Arc::new(StringArray::from(ss)), Arc::new(Float64Array::from(xs))],
    )?;
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}
