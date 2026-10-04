//! Execution coverage of native selection under the normal session
//! configuration. A fully supported query runs as one native plan. A query
//! whose relational parent or expression has no native contract keeps that
//! operator on DataFusion and runs the supported subtrees below it as native
//! fragments. Each case checks the final disposition, the fragment count, that
//! the report selects exactly the fragments in the final plan after the root's
//! rejection, and parity with plain DataFusion.

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

struct ExecutionCase {
    id: &'static str,
    sql: &'static str,
    expected_path: FinalPlanDisposition,
    expected_native_fragment_count: usize,
}

const EXECUTION_CASES: &[ExecutionCase] = &[
    ExecutionCase {
        id: "control_supported",
        sql: "SELECT key, SUM(m00) AS s, COUNT(*) AS c FROM bench GROUP BY key ORDER BY key",
        expected_path: FinalPlanDisposition::Native,
        expected_native_fragment_count: 1,
    },
    // APPROX_DISTINCT has no native aggregate; its join input runs as one
    // native fragment.
    ExecutionCase {
        id: "unsupported_aggregate_over_join_fragment",
        sql: "SELECT t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: FinalPlanDisposition::Mixed,
        expected_native_fragment_count: 1,
    },
    // Each union branch keeps its APPROX_DISTINCT on DataFusion over its own
    // join fragment.
    ExecutionCase {
        id: "union_of_unsupported_aggregates_over_two_fragments",
        sql: "SELECT * FROM ( \
              SELECT 'a' AS src, t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key \
              UNION ALL \
              SELECT 'b' AS src, u.key, APPROX_DISTINCT(u.name) AS d \
              FROM (SELECT b2.key, b2.name FROM bench2 b2 JOIN bench b3 ON b2.key = b3.key WHERE b3.m00 >= 2.0) u \
              GROUP BY u.key) v ORDER BY src, key",
        expected_path: FinalPlanDisposition::Mixed,
        expected_native_fragment_count: 2,
    },
    // ILIKE has no native expression. DataFusion copies the filter's predicate
    // into the Parquet scan for pruning; the fragment reads the scan without
    // the copy while the filter evaluates it.
    ExecutionCase {
        id: "ilike_filter_over_scan_fragment",
        sql: "SELECT key, name FROM bench WHERE name ILIKE 'NAME_1%' ORDER BY key, name",
        expected_path: FinalPlanDisposition::Mixed,
        expected_native_fragment_count: 1,
    },
    // TRY_CAST has no native expression; the join below its aggregate input
    // runs as a native fragment.
    ExecutionCase {
        id: "unsupported_expression_over_join_fragment",
        sql: "SELECT t.key, SUM(TRY_CAST(t.name AS DOUBLE)) AS s \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: FinalPlanDisposition::Mixed,
        expected_native_fragment_count: 1,
    },
];

#[tokio::test]
async fn native_selection_execution_coverage_matches_datafusion() {
    let dir = unique_temp_dir("algeon-datafusion-native-selection-execution");
    let bench = dir.join("bench.parquet");
    let bench2 = dir.join("bench2.parquet");
    write_execution_input_parquet(&bench, 0);
    write_execution_input_parquet(&bench2, 1);

    for case in EXECUTION_CASES {
        run_execution_case(case, &bench, &bench2).await;
    }

    fs::remove_dir_all(&dir).ok();
}

async fn run_execution_case(case: &ExecutionCase, bench: &std::path::Path, bench2: &std::path::Path) {
    // Share the execution suite's process-owned backend across cases. A second
    // whole-device backend must not be installed after the first case runs.
    let backend = test_gpu_backend();
    // install_on freezes session config for QueryService admission authority.
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features().with_config(direct_friendly_sql_session_config()))
        .unwrap_or_else(|error| panic!("{}: backend install failed: {error}", case.id))
        .build();
    let gpu_ctx = SessionContext::new_with_state(state);
    register_execution_tables(&gpu_ctx, bench, bench2).await;

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
            CandidateOutcome::Selected { .. } => rewritten += 1,
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
    register_execution_tables(&cpu_ctx, bench, bench2).await;
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
    assert_eq!(
        rewritten, case.expected_native_fragment_count,
        "{}: the report selects exactly the native fragments of the final plan",
        case.id
    );

    match case.expected_path {
        FinalPlanDisposition::Mixed => {
            assert!(
                matches!(planning_report.outcomes().first(), Some(CandidateOutcome::NotSupported { .. })),
                "{}: the root's rejection leads the report: {rejected_reasons:?}",
                case.id
            );
            assert!(
                row.host_datafusion_boundary_count.is_some_and(|count| count > 0),
                "{}: DataFusion operators read the native fragments' host output",
                case.id
            );
        },
        FinalPlanDisposition::Native => {
            assert!(rejected_reasons.is_empty(), "{}: supported whole plan should not have rejections", case.id);
            assert_eq!(
                row.host_datafusion_boundary_count,
                Some(0),
                "{}: whole native plan should have no host boundary",
                case.id
            );
        },
        unexpected => panic!("{}: execution coverage does not admit {unexpected:?}", case.id),
    }
}

/// A fragment whose single partition would break the co-partitioning of a
/// partitioned hash join is abandoned with that reason, and the fragments
/// below each hash repartition keep the join's partition layout.
#[tokio::test]
async fn fragment_that_breaks_a_co_partitioned_join_is_abandoned_for_fragments_below_it() {
    let dir = unique_temp_dir("algeon-datafusion-co-partitioned-join-fragments");
    // Four files per table give each scan four partitions.
    let bench = dir.join("bench/");
    let bench2 = dir.join("bench2/");
    for (table, salt) in [(&bench, 0), (&bench2, 1)] {
        fs::create_dir_all(table).unwrap();
        for file in 0..4 {
            write_execution_input_parquet(&table.join(format!("part-{file}.parquet")), salt + 2 * file);
        }
    }
    // The ILIKE join filter has no native expression, so the join stays on
    // DataFusion; without a size threshold DataFusion partitions it by key.
    let sql = "SELECT a.key, a.name, b.payload FROM bench a JOIN bench2 b \
               ON a.key = b.key AND a.name ILIKE b.name ORDER BY a.key, a.name, b.payload";
    let mut session_config = SessionConfig::new().with_target_partitions(4);
    session_config.options_mut().optimizer.hash_join_single_partition_threshold = 0;
    session_config.options_mut().optimizer.hash_join_single_partition_threshold_rows = 0;
    let state = test_gpu_backend()
        .install_on(SessionStateBuilder::new_with_default_features().with_config(session_config.clone()))
        .expect("backend install")
        .build();
    let gpu_ctx = SessionContext::new_with_state(state);
    register_execution_tables(&gpu_ctx, &bench, &bench2).await;

    let (session_state, logical) = gpu_ctx.sql(sql).await.unwrap().into_parts();
    let (plan, diagnostics) = plan_with_native_diagnostics(&session_state, &logical).await.unwrap();
    drop(session_state);

    assert!(
        matches!(
            diagnostics.report.outcomes(),
            [
                CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. },
                CandidateOutcome::NotSupported {
                    capability_reason: CapabilityReason::NativeFragmentParentContractUnsupported,
                    ..
                },
                CandidateOutcome::Selected { .. },
                CandidateOutcome::NotSupported {
                    capability_reason: CapabilityReason::NativeFragmentParentContractUnsupported,
                    ..
                },
                CandidateOutcome::Selected { .. },
            ]
        ),
        "{:?}",
        diagnostics.report.outcomes()
    );
    let mut join = below_native_admission(&plan);
    while join.name() != "HashJoinExec" {
        join = join.children()[0];
    }
    for side in join.children() {
        assert_eq!(side.name(), "RepartitionExec");
        assert!(matches!(side.properties().output_partitioning(), Partitioning::Hash(_, 4)));
        assert!(side.children()[0].downcast_ref::<AlgeonNativeExec>().is_some());
    }

    let gpu_batches = collect(Arc::clone(&plan), gpu_ctx.task_ctx()).await.unwrap();
    let cpu_ctx = SessionContext::new_with_config(session_config);
    register_execution_tables(&cpu_ctx, &bench, &bench2).await;
    let cpu_batches = cpu_ctx.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(
        pretty_format_batches(&gpu_batches).unwrap().to_string(),
        pretty_format_batches(&cpu_batches).unwrap().to_string()
    );

    fs::remove_dir_all(&dir).ok();
}

async fn register_execution_tables(ctx: &SessionContext, bench: &std::path::Path, bench2: &std::path::Path) {
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

fn write_execution_input_parquet(path: &std::path::Path, salt: i64) {
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
