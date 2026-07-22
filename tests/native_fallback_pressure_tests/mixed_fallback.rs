//! Four-case mixed-fallback execution contract migrated from
//! `native_mixed_fallback_report` (formerly `c1_mixed_fallback_experiment`).
//!
//! Asserts under `GpuPreferred`:
//! - post-native execution path
//! - min native fragment count
//! - host DataFusion boundary evidence
//! - candidate rewritten/rejected outcomes
//! - GPU-preferred session results vs plain DataFusion parity

use arrow::{array::StringArray, util::pretty::pretty_format_batches};
use datafusion_nexus::{
    backend::{GrantBand, MemoryShare, NexusAdmissionPolicy, NexusGpuBackend, NexusMemoryPolicy},
    report::CandidateOutcome,
    session::NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
};
use nexus_query_engine::runtime::SlotGuarantee;

use super::*;

struct MixedFallbackCase {
    id: &'static str,
    sql: &'static str,
    expected_path: NativeDataFusionPostNativeExecutionPath,
    expected_min_fragments: usize,
}

const MIXED_FALLBACK_CASES: &[MixedFallbackCase] = &[
    // Control: fully supported query should be rewritten whole.
    MixedFallbackCase {
        id: "control_supported",
        sql: "SELECT key, SUM(m00) AS s, COUNT(*) AS c FROM bench GROUP BY key ORDER BY key",
        expected_path: NativeDataFusionPostNativeExecutionPath::Native,
        expected_min_fragments: 1,
    },
    // Mixed: APPROX_DISTINCT (unsupported operator) over a supported join+filter
    // subtree. Expect the join fragment on GPU, the aggregate on CPU.
    MixedFallbackCase {
        id: "mixed_unsupported_aggregate",
        sql: "SELECT t.key, APPROX_DISTINCT(t.name) AS d \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: NativeDataFusionPostNativeExecutionPath::PartialNative,
        expected_min_fragments: 1,
    },
    // Mixed with two disjoint supported fragments under a CPU union of two
    // unsupported aggregates.
    MixedFallbackCase {
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
    MixedFallbackCase {
        id: "negative_unsupported_expression",
        sql: "SELECT t.key, SUM(TRY_CAST(t.name AS DOUBLE)) AS s \
              FROM (SELECT b.key, b.name FROM bench b JOIN bench2 s ON b.key = s.key WHERE s.payload >= 10) t \
              GROUP BY t.key ORDER BY t.key",
        expected_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        expected_min_fragments: 0,
    },
];

#[tokio::test]
async fn mixed_fallback_execution_contract_four_cases() {
    let dir = unique_temp_dir("datafusion-nexus-native-mixed-fallback");
    let bench = dir.join("bench.parquet");
    let bench2 = dir.join("bench2.parquet");
    write_mixed_fallback_input_parquet(&bench, 0);
    write_mixed_fallback_input_parquet(&bench2, 1);

    for case in MIXED_FALLBACK_CASES {
        run_mixed_fallback_case(case, &bench, &bench2).await;
    }

    fs::remove_dir_all(&dir).ok();
}

async fn run_mixed_fallback_case(case: &MixedFallbackCase, bench: &std::path::Path, bench2: &std::path::Path) {
    // Multi-fragment PartialNative plans need shared QueryService admission; use
    // NexusGpuBackend so concurrent native fragments are not standalone-unadmitted.
    let backend = NexusGpuBackend::builder()
        .fallback_policy(GpuFallbackPolicy::GpuPreferred)
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 256 * 1024 * 1024,
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .admission(
            NexusAdmissionPolicy::default()
                .with_slot_guarantee(SlotGuarantee::Optimistic)
                // Two concurrent native fragments share one admission handle/grant; size the
                // race-free grant high enough for both join fragments under the 256MiB budget.
                .with_grant_band(GrantBand {
                    min: MemoryShare::Bytes(64 * 1024 * 1024),
                    want: MemoryShare::Bytes(128 * 1024 * 1024),
                }),
        )
        .build()
        .unwrap_or_else(|error| panic!("{}: backend build failed: {error}", case.id));
    // install_on freezes session config for QueryService admission authority and
    // wraps multi-fragment plans with a shared admission ticket.
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features().with_config(direct_friendly_sql_session_config()))
        .unwrap_or_else(|error| panic!("{}: backend install failed: {error}", case.id))
        .build();
    let gpu_ctx = SessionContext::new_with_state(state);
    register_mixed_fallback_tables(&gpu_ctx, bench, bench2).await;

    let df = gpu_ctx.sql(case.sql).await.expect(case.id);
    let plan = df.create_physical_plan().await.expect(case.id);
    // state() clones SessionState; Arc optimizer rules still share retained diagnostics.
    let session_state = gpu_ctx.state();
    let native_rule = installed_final_native_rule(&session_state)
        .unwrap_or_else(|| panic!("{}: installed final native rule missing", case.id));
    let report = NativeDataFusionPlanningReport::from_retained_rule(case.id, Arc::clone(&plan), native_rule);
    let row = report.rows.first().unwrap_or_else(|| panic!("{}: planning report produced no rows", case.id));

    let planning_report =
        native_rule.last_report().unwrap_or_else(|| panic!("{}: rule retained no PlanningReport", case.id));
    let mut rewritten = 0usize;
    let mut rejected_reasons = Vec::new();
    for outcome in planning_report.outcomes() {
        match outcome {
            CandidateOutcome::Rewritten { .. } => rewritten += 1,
            CandidateOutcome::Rejected { reason, .. } => rejected_reasons.push(reason.code().to_string()),
        }
    }
    drop(session_state);

    let gpu_batches = match gpu_ctx.sql(case.sql).await.expect(case.id).collect().await {
        Ok(batches) => batches,
        Err(error) => panic!(
            "{}: GPU collect failed: {error:#}\npath={:?} fragments={} host_boundaries={} rewritten={rewritten} rejected={rejected_reasons:?}",
            case.id, row.post_native_execution_path, row.native_plan_fragment_count, row.host_datafusion_boundary_count,
        ),
    };
    let cpu_ctx = SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features().with_config(direct_friendly_sql_session_config()).build(),
    );
    register_mixed_fallback_tables(&cpu_ctx, bench, bench2).await;
    let cpu_batches = cpu_ctx.sql(case.sql).await.expect(case.id).collect().await.expect(case.id);

    let gpu_text = pretty_format_batches(&gpu_batches).expect(case.id).to_string();
    let cpu_text = pretty_format_batches(&cpu_batches).expect(case.id).to_string();
    assert_eq!(
        gpu_text, cpu_text,
        "{}: GPU-vs-plain-DataFusion result parity mismatch\ngpu:\n{gpu_text}\ncpu:\n{cpu_text}",
        case.id
    );

    assert_eq!(
        row.post_native_execution_path, case.expected_path,
        "{}: unexpected post_native_execution_path (fragments={}, host_boundaries={}, rewritten={rewritten}, rejected={rejected_reasons:?})",
        case.id, row.native_plan_fragment_count, row.host_datafusion_boundary_count
    );
    assert!(
        row.native_plan_fragment_count >= case.expected_min_fragments,
        "{}: fragments {} < expected min {}",
        case.id,
        row.native_plan_fragment_count,
        case.expected_min_fragments
    );

    match case.expected_path {
        NativeDataFusionPostNativeExecutionPath::PartialNative => {
            assert!(rewritten >= 1, "{}: expected at least one rewritten candidate, got {rewritten}", case.id);
            assert!(!rejected_reasons.is_empty(), "{}: expected non-empty rejected reasons for mixed path", case.id);
            assert!(
                row.host_datafusion_boundary_count >= 1,
                "{}: expected host_datafusion_boundary_count >= 1, got {}",
                case.id,
                row.host_datafusion_boundary_count
            );
        },
        NativeDataFusionPostNativeExecutionPath::Cpu => {
            assert_eq!(rewritten, 0, "{}: CPU path should have zero rewritten candidates, got {rewritten}", case.id);
        },
        NativeDataFusionPostNativeExecutionPath::Native => {},
    }
}

fn installed_final_native_rule(
    state: &datafusion::execution::session_state::SessionState,
) -> Option<&NexusNativePhysicalOptimizerRule> {
    state.physical_optimizers().iter().find_map(|rule| {
        if rule.name() != NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME {
            return None;
        }
        (rule.as_ref() as &dyn std::any::Any).downcast_ref::<NexusNativePhysicalOptimizerRule>()
    })
}

async fn register_mixed_fallback_tables(ctx: &SessionContext, bench: &std::path::Path, bench2: &std::path::Path) {
    ctx.register_parquet("bench", bench.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.register_parquet("bench2", bench2.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
}

fn write_mixed_fallback_input_parquet(path: &std::path::Path, salt: i64) {
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
