//! Substrait logical-plan ingress experiment for
//! docs/issue/substrait_limitation_issue.md and
//! docs/research/substrait-adoption-assessment.md (§4.2).
//!
//! Tests the hypothesis that Substrait plans can enter Nexus at the LOGICAL
//! level (`datafusion-substrait` consumer -> DataFusion logical plan ->
//! standard physical planning -> `CudfNativePhysicalOptimizerRule`) while
//! preserving native GPU rewrites, per-fragment CPU fallback, and result
//! parity with the SQL entry path.
//!
//! For every TPC-H query (plus one mixed supported/unsupported query) it runs:
//!   path A: SQL text -> physical plan -> planning report -> results
//!   path B: SQL -> optimized logical plan -> Substrait producer -> Substrait
//!           consumer -> physical plan -> planning report -> results
//! and compares planning outcomes and result batches.
//!
//! Run under the GPU lock:
//! `flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-bench \
//!    --features substrait --bin substrait_ingress_experiment`

use std::{
    error::Error,
    fs,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::util::pretty::pretty_format_batches;
use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use datafusion_nexus::{
    native_report::{NativeDataFusionPlanningReport, NativeDataFusionPostNativeExecutionPath},
    planner::CudfNativePhysicalOptimizerRule,
    session::CudfSessionStateBuilderExt,
};
use datafusion_nexus_bench::{
    TPCH_QUERIES, TPCH_TABLES, discover_tpch_table_layout, prepare_tpch_report_dataset, register_tpch_table_layout,
    tpch_native_optimizer_config, tpch_report_session_config,
};
use datafusion_substrait::logical_plan::{consumer::from_substrait_plan, producer::to_substrait_plan};

const MIXED_QUERY_ID: &str = "mixed_approx_distinct";
const MIXED_QUERY_SQL: &str = "SELECT o_orderpriority, APPROX_DISTINCT(l_comment) AS d \
     FROM lineitem JOIN orders ON l_orderkey = o_orderkey \
     WHERE o_orderdate >= DATE '1995-01-01' \
     GROUP BY o_orderpriority ORDER BY o_orderpriority";

struct RowResult {
    id: String,
    roundtrip: &'static str,
    sql_path: String,
    substrait_path: String,
    parity: &'static str,
    note: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = unique_temp_dir("substrait-ingress-experiment");
    let (root, _metadata) = prepare_tpch_report_dataset(&dir)?;

    let mut rows = Vec::new();
    let mut cases: Vec<(String, String)> = TPCH_QUERIES.iter().map(|q| (q.id.to_string(), q.sql.to_string())).collect();
    cases.push((MIXED_QUERY_ID.to_string(), MIXED_QUERY_SQL.to_string()));
    if let Ok(filter) = std::env::var("SUBSTRAIT_EXPERIMENT_QUERY_IDS") {
        let wanted: Vec<&str> = filter.split(',').map(str::trim).collect();
        cases.retain(|(id, _)| wanted.contains(&id.as_str()));
    }

    for (id, sql) in &cases {
        let row = run_case(id, sql, &root).await;
        println!(
            "{}\t roundtrip={}\t sql={}\t substrait={}\t parity={}\t {}",
            row.id, row.roundtrip, row.sql_path, row.substrait_path, row.parity, row.note
        );
        rows.push(row);
    }

    let total = rows.len();
    let roundtrip_ok = rows.iter().filter(|r| r.roundtrip == "ok").count();
    let parity_ok = rows.iter().filter(|r| r.parity == "match").count();
    let same_path = rows.iter().filter(|r| r.roundtrip == "ok" && r.sql_path == r.substrait_path).count();
    println!("\nSUMMARY: {total} cases; substrait roundtrip ok {roundtrip_ok}/{total}; ");
    println!(
        "identical planning path (sql vs substrait) {same_path}/{roundtrip_ok} of surviving; result parity {parity_ok}/{roundtrip_ok}"
    );

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

async fn run_case(id: &str, sql: &str, root: &std::path::Path) -> RowResult {
    // Path A: SQL entry.
    let (ctx_a, rule_a) = match session(root).await {
        Ok(v) => v,
        Err(e) => return error_row(id, "session", e),
    };
    eprintln!("[{id}] path A (sql entry) starting");
    let (sql_path, sql_batches) = match plan_and_run(&ctx_a, &rule_a, id, sql, SqlEntry::Sql).await {
        Ok(v) => v,
        Err(e) => return error_row(id, "sql-entry", e),
    };
    eprintln!("[{id}] path A done ({sql_path}); path B (substrait entry) starting");

    // Path B: Substrait entry (fresh session so reports don't mix).
    let (ctx_b, rule_b) = match session(root).await {
        Ok(v) => v,
        Err(e) => return error_row(id, "session", e),
    };
    let logical = match ctx_b.sql(sql).await.and_then(|df| df.into_optimized_plan()) {
        Ok(plan) => plan,
        Err(e) => return error_row(id, "logical-plan", Box::new(e)),
    };
    let state = ctx_b.state();
    let substrait_plan = match to_substrait_plan(&logical, &state) {
        Ok(plan) => plan,
        Err(e) => {
            return RowResult {
                id: id.to_string(),
                roundtrip: "producer_error",
                sql_path,
                substrait_path: "-".into(),
                parity: "-",
                note: first_line(&e.to_string()),
            };
        },
    };
    let consumed = match from_substrait_plan(&state, &substrait_plan).await {
        Ok(plan) => plan,
        Err(e) => {
            return RowResult {
                id: id.to_string(),
                roundtrip: "consumer_error",
                sql_path,
                substrait_path: "-".into(),
                parity: "-",
                note: first_line(&e.to_string()),
            };
        },
    };
    let (substrait_path, substrait_batches) =
        match plan_and_run(&ctx_b, &rule_b, id, sql, SqlEntry::Logical(Box::new(consumed))).await {
            Ok(v) => v,
            Err(e) => {
                return RowResult {
                    id: id.to_string(),
                    roundtrip: "exec_error",
                    sql_path,
                    substrait_path: "-".into(),
                    parity: "-",
                    note: first_line(&e.to_string()),
                };
            },
        };

    let sql_text = pretty_format_batches(&sql_batches).map(|t| t.to_string()).unwrap_or_default();
    let substrait_text = pretty_format_batches(&substrait_batches).map(|t| t.to_string()).unwrap_or_default();
    let parity = if sql_text == substrait_text { "match" } else { "MISMATCH" };
    RowResult { id: id.to_string(), roundtrip: "ok", sql_path, substrait_path, parity, note: String::new() }
}

enum SqlEntry {
    Sql,
    Logical(Box<datafusion::logical_expr::LogicalPlan>),
}

async fn plan_and_run(
    ctx: &SessionContext,
    rule: &Arc<CudfNativePhysicalOptimizerRule>,
    id: &str,
    sql: &str,
    entry: SqlEntry,
) -> Result<(String, Vec<arrow::record_batch::RecordBatch>), Box<dyn Error>> {
    let df = match entry {
        SqlEntry::Sql => ctx.sql(sql).await?,
        SqlEntry::Logical(plan) => ctx.execute_logical_plan(*plan).await?,
    };
    if std::env::var("SUBSTRAIT_EXPERIMENT_SINGLE_PLAN").is_ok() {
        // Minimal-repro mode: one planning pass only (plain user-facing flow).
        let batches = df.collect().await?;
        return Ok(("unreported".to_string(), batches));
    }
    let physical = df.clone().create_physical_plan().await?;
    let report = NativeDataFusionPlanningReport::from_retained_rule(id, Arc::clone(&physical), rule.as_ref());
    let row = report.rows.first().ok_or("planning report had no rows")?;
    let path = match row.post_native_execution_path {
        NativeDataFusionPostNativeExecutionPath::Native => "native".to_string(),
        NativeDataFusionPostNativeExecutionPath::PartialNative => {
            format!("partial({} frags)", row.native_plan_fragment_count)
        },
        _ => format!("{:?}", row.post_native_execution_path).to_lowercase(),
    };
    let batches = df.collect().await?;
    Ok((path, batches))
}

async fn session(
    root: &std::path::Path,
) -> Result<(SessionContext, Arc<CudfNativePhysicalOptimizerRule>), Box<dyn Error>> {
    let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(tpch_native_optimizer_config()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_native_rule(Arc::clone(&rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    for table in TPCH_TABLES {
        let layout = discover_tpch_table_layout(root, table)?;
        register_tpch_table_layout(&ctx, table, &layout).await?;
    }
    Ok((ctx, rule))
}

fn error_row(id: &str, stage: &'static str, e: Box<dyn Error>) -> RowResult {
    RowResult {
        id: id.to_string(),
        roundtrip: stage,
        sql_path: "-".into(),
        substrait_path: "-".into(),
        parity: "-",
        note: first_line(&e.to_string()),
    }
}

fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default();
    let mut short = line.chars().take(160).collect::<String>();
    if short.len() < line.len() {
        short.push('…');
    }
    short
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
