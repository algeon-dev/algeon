//! C1 empirical experiment for docs/issue/substrait_limitation_issue.md.
//!
//! Verifies the intro.mdx claim that under `GpuPreferred` a query mixing
//! supported and unsupported operators keeps supported fragments on the GPU
//! while the unsupported remainder stays on the DataFusion CPU path, with
//! correct results — instead of falling back the whole query.
//!
//! Run under the GPU lock:
//! `flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-bench --bin c1_mixed_fallback_experiment`

use std::{
    error::Error,
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::pretty::pretty_format_batches,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::displayable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionPlanningReport,
    NativeDataFusionPostNativeExecutionPath,
};

struct Case {
    id: &'static str,
    sql: &'static str,
    expected_path: NativeDataFusionPostNativeExecutionPath,
    expected_min_fragments: usize,
}

const CASES: &[Case] = &[
    // Control: fully supported query should be rewritten whole.
    Case {
        id: "control_supported",
        sql: "SELECT key, SUM(m00) AS s, COUNT(*) AS c FROM bench GROUP BY key ORDER BY key",
        expected_path: NativeDataFusionPostNativeExecutionPath::Native,
        expected_min_fragments: 1,
    },
    // Mixed: APPROX_DISTINCT (unsupported operator) over a supported join+filter
    // subtree. Expect the join fragment on GPU, the aggregate on CPU.
    Case {
        id: "mixed_unsupported_aggregate",
        sql: "SELECT t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: NativeDataFusionPostNativeExecutionPath::PartialNative,
        expected_min_fragments: 1,
    },
    // Mixed with two disjoint supported fragments under a CPU union of two
    // unsupported aggregates.
    Case {
        id: "mixed_two_disjoint_fragments",
        sql: "SELECT * FROM ( \
              SELECT 'a' AS src, t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key \
              UNION ALL \
              SELECT 'b' AS src, u.key, APPROX_DISTINCT(u.name) AS d \
              FROM (SELECT b2.key, b2.name FROM bench2 b2 JOIN bench b3 ON b2.key = b3.key WHERE b3.m00 >= 2.0) u \
              GROUP BY u.key) v ORDER BY src, key",
        expected_path: NativeDataFusionPostNativeExecutionPath::PartialNative,
        expected_min_fragments: 2,
    },
    // Negative control: unsupported *expression* rejections intentionally block
    // fragment extraction -> whole query stays on CPU.
    Case {
        id: "negative_unsupported_expression",
        sql: "SELECT t.key, SUM(TRY_CAST(t.name AS DOUBLE)) AS s \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        expected_min_fragments: 0,
    },
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = unique_temp_dir("c1-mixed-fallback-experiment");
    let bench = dir.join("bench.parquet");
    let bench2 = dir.join("bench2.parquet");
    write_experiment_parquet(&bench, 0);
    write_experiment_parquet(&bench2, 1);

    let mut failures = Vec::new();
    for case in CASES {
        let outcome = run_case(case, &bench, &bench2).await?;
        println!("{outcome}");
        if !outcome.passed {
            failures.push(case.id);
        }
    }

    fs::remove_dir_all(&dir).ok();
    if failures.is_empty() {
        println!("\nC1 VERDICT: all {} cases behaved as expected", CASES.len());
        Ok(())
    } else {
        Err(format!("C1 experiment failures: {failures:?}").into())
    }
}

struct CaseOutcome {
    id: &'static str,
    passed: bool,
    detail: String,
}

impl std::fmt::Display for CaseOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "==== case {} => {} ====", self.id, if self.passed { "PASS" } else { "FAIL" })?;
        write!(f, "{}", self.detail)
    }
}

async fn run_case(case: &Case, bench: &std::path::Path, bench2: &std::path::Path) -> Result<CaseOutcome, Box<dyn Error>> {
    let mut detail = String::new();

    // GPU-preferred session with a retained rule for report capture.
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1));
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(experiment_session_config())
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let gpu_ctx = SessionContext::new_with_state(state);
    register_tables(&gpu_ctx, bench, bench2).await?;

    let df = gpu_ctx.sql(case.sql).await?;
    let plan = df.create_physical_plan().await?;
    let report = NativeDataFusionPlanningReport::from_retained_rule(case.id, Arc::clone(&plan), native_rule.as_ref());
    let row = report.rows.first().ok_or("planning report produced no rows")?;

    let planning_report = native_rule.last_report().ok_or("rule retained no PlanningReport")?;
    let mut rewritten = 0usize;
    let mut rejected_reasons = Vec::new();
    for outcome in planning_report.outcomes() {
        match outcome {
            CandidateOutcome::Rewritten { .. } => rewritten += 1,
            CandidateOutcome::Rejected { reason, .. } => rejected_reasons.push(reason.code().to_string()),
        }
    }

    use std::fmt::Write as _;
    writeln!(detail, "sql: {}", case.sql)?;
    writeln!(
        detail,
        "planning: outcome={:?} path={:?} fragments={} host_boundaries={} reason_code={:?}",
        row.outcome, row.post_native_execution_path, row.native_plan_fragment_count, row.host_datafusion_boundary_count,
        row.reason_code
    )?;
    writeln!(detail, "candidates: rewritten={rewritten} rejected_reasons={rejected_reasons:?}")?;
    writeln!(detail, "optimized physical plan:\n{}", displayable(plan.as_ref()).indent(false))?;

    // Execute on GPU-preferred session.
    let gpu_batches = gpu_ctx.sql(case.sql).await?.collect().await?;
    // Execute on a plain CPU DataFusion session (no nexus rule at all).
    let cpu_ctx = SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features().with_config(experiment_session_config()).build(),
    );
    register_tables(&cpu_ctx, bench, bench2).await?;
    let cpu_batches = cpu_ctx.sql(case.sql).await?.collect().await?;

    let gpu_text = pretty_format_batches(&gpu_batches)?.to_string();
    let cpu_text = pretty_format_batches(&cpu_batches)?.to_string();
    let parity = gpu_text == cpu_text;
    writeln!(detail, "results parity vs plain-CPU DataFusion: {}", if parity { "MATCH" } else { "MISMATCH" })?;
    if !parity {
        writeln!(detail, "gpu:\n{gpu_text}\ncpu:\n{cpu_text}")?;
    } else {
        writeln!(detail, "result:\n{gpu_text}")?;
    }

    let path_ok = row.post_native_execution_path == case.expected_path;
    let fragments_ok = row.native_plan_fragment_count >= case.expected_min_fragments;
    let mixed_ok = match case.expected_path {
        NativeDataFusionPostNativeExecutionPath::PartialNative => {
            rewritten >= 1 && !rejected_reasons.is_empty() && row.host_datafusion_boundary_count >= 1
        },
        NativeDataFusionPostNativeExecutionPath::Cpu => rewritten == 0,
        _ => true,
    };
    if !path_ok {
        writeln!(detail, "EXPECTATION FAILED: path {:?} != expected {:?}", row.post_native_execution_path, case.expected_path)?;
    }
    if !fragments_ok {
        writeln!(
            detail,
            "EXPECTATION FAILED: fragments {} < expected min {}",
            row.native_plan_fragment_count, case.expected_min_fragments
        )?;
    }
    if !mixed_ok {
        writeln!(detail, "EXPECTATION FAILED: mixed-candidate evidence missing (rewritten={rewritten}, rejected={rejected_reasons:?})")?;
    }

    Ok(CaseOutcome { id: case.id, passed: parity && path_ok && fragments_ok && mixed_ok, detail })
}

async fn register_tables(
    ctx: &SessionContext,
    bench: &std::path::Path,
    bench2: &std::path::Path,
) -> Result<(), Box<dyn Error>> {
    ctx.register_parquet("bench", bench.to_str().unwrap(), ParquetReadOptions::default()).await?;
    ctx.register_parquet("bench2", bench2.to_str().unwrap(), ParquetReadOptions::default()).await?;
    Ok(())
}

fn experiment_session_config() -> SessionConfig {
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

fn write_experiment_parquet(path: &std::path::Path, salt: i64) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let keys: Vec<i32> = (0..64).map(|i| i % 8).collect();
    let payloads: Vec<i64> = (0..64).map(|i| (i * 3 + salt * 7) % 40).collect();
    let m00: Vec<f64> = (0..64).map(|i| ((i + salt) as f64) / 4.0).collect();
    let names: Vec<String> = (0..64).map(|i| format!("name_{}", (i + salt * 11) % 13)).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int64Array::from(payloads)),
            Arc::new(Float64Array::from(m00)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
