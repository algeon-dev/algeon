//! DataFusion-baseline execution coverage migrated from
//! `native_mixed_fallback_report` (formerly `c1_mixed_fallback_experiment`).
//!
//! Asserts whole-plan native selection under the normal session configuration:
//! - post-native execution path
//! - exact native fragment count
//! - retained DataFusion baseline for unsupported relational parents
//! - candidate rewritten/rejected outcomes
//! - session results vs plain DataFusion parity

use std::fs::File;

use algeon_datafusion::{planner::plan_with_native_diagnostics, report::CandidateOutcome};
use arrow::{
    array::{Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::pretty::pretty_format_batches,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};

use super::*;

struct DataFusionBaselineCase {
    id: &'static str,
    sql: &'static str,
    expected_path: FinalPlanDisposition,
    expected_native_fragment_count: usize,
}

const DATAFUSION_BASELINE_CASES: &[DataFusionBaselineCase] = &[
    // Control: fully supported query should be rewritten whole.
    DataFusionBaselineCase {
        id: "control_supported",
        sql: "SELECT key, SUM(m00) AS s, COUNT(*) AS c FROM bench GROUP BY key ORDER BY key",
        expected_path: FinalPlanDisposition::Native,
        expected_native_fragment_count: 1,
    },
    // APPROX_DISTINCT is an unsupported relational parent. Its supported join
    // subtree must not be extracted into a separate native fragment.
    DataFusionBaselineCase {
        id: "datafusion_baseline_unsupported_aggregate",
        sql: "SELECT t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: FinalPlanDisposition::DataFusion,
        expected_native_fragment_count: 0,
    },
    // A union of unsupported relational parents remains a single DataFusion
    // plan; no disjoint native child fragments are producerized.
    DataFusionBaselineCase {
        id: "datafusion_baseline_two_disjoint_rejections",
        sql: "SELECT * FROM ( \
              SELECT 'a' AS src, t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key \
              UNION ALL \
              SELECT 'b' AS src, u.key, APPROX_DISTINCT(u.name) AS d \
              FROM (SELECT b2.key, b2.name FROM bench2 b2 JOIN bench b3 ON b2.key = b3.key WHERE b3.m00 >= 2.0) u \
              GROUP BY u.key) v ORDER BY src, key",
        expected_path: FinalPlanDisposition::DataFusion,
        expected_native_fragment_count: 0,
    },
    // Unsupported expressions likewise retain the whole query on DataFusion.
    DataFusionBaselineCase {
        id: "negative_unsupported_expression",
        sql: "SELECT t.key, SUM(TRY_CAST(t.name AS DOUBLE)) AS s \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: FinalPlanDisposition::DataFusion,
        expected_native_fragment_count: 0,
    },
];

#[tokio::test]
async fn datafusion_baseline_execution_coverage_preserves_datafusion_plans() {
    let dir = unique_temp_dir("algeon-datafusion-datafusion-baseline");
    let bench = dir.join("bench.parquet");
    let bench2 = dir.join("bench2.parquet");
    write_datafusion_baseline_input_parquet(&bench, 0);
    write_datafusion_baseline_input_parquet(&bench2, 1);

    for case in DATAFUSION_BASELINE_CASES {
        run_datafusion_baseline_case(case, &bench, &bench2).await;
    }

    fs::remove_dir_all(&dir).ok();
}

async fn run_datafusion_baseline_case(
    case: &DataFusionBaselineCase,
    bench: &std::path::Path,
    bench2: &std::path::Path,
) {
    // Share the execution suite's process-owned backend across cases. A second
    // whole-device backend must not be installed after the first case runs.
    let backend = test_gpu_backend();
    // install_on freezes session config for QueryService admission authority.
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features().with_config(direct_friendly_sql_session_config()))
        .unwrap_or_else(|error| panic!("{}: backend install failed: {error}", case.id))
        .build();
    let gpu_ctx = SessionContext::new_with_state(state);
    register_datafusion_baseline_tables(&gpu_ctx, bench, bench2).await;

    let df = gpu_ctx.sql(case.sql).await.expect(case.id);
    let (session_state, logical) = df.into_parts();
    let (plan, diagnostics) = plan_with_native_diagnostics(&session_state, &logical).await.expect(case.id);
    let planning_report = diagnostics.report.clone();
    let report = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        case.id,
        Arc::clone(&plan),
        diagnostics,
    )]);
    let row = report.rows.first().unwrap_or_else(|| panic!("{}: planning report produced no rows", case.id));

    let mut rewritten = 0usize;
    let mut rejected_reasons = Vec::new();
    for outcome in planning_report.outcomes() {
        match outcome {
            CandidateOutcome::Selected { .. } | CandidateOutcome::Deferred { .. } => rewritten += 1,
            CandidateOutcome::NotSupported { capability_reason, .. } => {
                rejected_reasons.push(capability_reason.code().to_string());
            },
            CandidateOutcome::NotSelectedByCost { cost_reason, .. } => {
                rejected_reasons.push(cost_reason.code().to_string());
            },
        }
    }
    drop(session_state);

    let gpu_batches = match gpu_ctx.sql(case.sql).await.expect(case.id).collect().await {
        Ok(batches) => batches,
        Err(error) => panic!(
            "{}: GPU collect failed: {error:#}\npath={:?} fragments={:?} host_boundaries={:?} rewritten={rewritten} rejected={rejected_reasons:?}",
            case.id, row.final_disposition, row.native_plan_fragment_count, row.host_datafusion_boundary_count,
        ),
    };
    let cpu_ctx = SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features().with_config(direct_friendly_sql_session_config()).build(),
    );
    register_datafusion_baseline_tables(&cpu_ctx, bench, bench2).await;
    let cpu_batches = cpu_ctx.sql(case.sql).await.expect(case.id).collect().await.expect(case.id);

    let gpu_text = pretty_format_batches(&gpu_batches).expect(case.id).to_string();
    let cpu_text = pretty_format_batches(&cpu_batches).expect(case.id).to_string();
    assert_eq!(
        gpu_text, cpu_text,
        "{}: GPU-vs-plain-DataFusion result parity mismatch\ngpu:\n{gpu_text}\ncpu:\n{cpu_text}",
        case.id
    );

    assert_eq!(
        row.final_disposition,
        Some(case.expected_path),
        "{}: unexpected final_disposition (fragments={:?}, host_boundaries={:?}, rewritten={rewritten}, rejected={rejected_reasons:?})",
        case.id,
        row.native_plan_fragment_count,
        row.host_datafusion_boundary_count
    );
    assert_eq!(
        row.native_plan_fragment_count,
        Some(case.expected_native_fragment_count),
        "{}: unexpected native fragment count",
        case.id,
    );

    match case.expected_path {
        FinalPlanDisposition::DataFusion => {
            assert_eq!(
                rewritten, 0,
                "{}: DataFusion baseline should have zero rewritten candidates, got {rewritten}",
                case.id
            );
            assert!(
                !rejected_reasons.is_empty(),
                "{}: DataFusion baseline should retain its native rejection evidence",
                case.id
            );
            assert_eq!(
                row.host_datafusion_boundary_count,
                Some(0),
                "{}: ordinary relational DataFusion baseline must not create a host/native boundary",
                case.id
            );
        },
        FinalPlanDisposition::Native => {
            assert_eq!(rewritten, 1, "{}: supported whole plan should have one rewrite", case.id);
            assert!(rejected_reasons.is_empty(), "{}: supported whole plan should not have rejections", case.id);
            assert_eq!(
                row.host_datafusion_boundary_count,
                Some(0),
                "{}: whole native plan should have no host boundary",
                case.id
            );
        },
        unexpected => panic!("{}: DataFusion-baseline coverage does not admit {unexpected:?}", case.id),
    }
}

async fn register_datafusion_baseline_tables(ctx: &SessionContext, bench: &std::path::Path, bench2: &std::path::Path) {
    ctx.register_parquet("bench", bench.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.register_parquet("bench2", bench2.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
}

fn direct_friendly_sql_session_config() -> SessionConfig {
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

fn write_datafusion_baseline_input_parquet(path: &std::path::Path, salt: i64) {
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
