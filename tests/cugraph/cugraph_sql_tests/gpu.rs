use datafusion::{common::config::ConfigOptions, physical_optimizer::PhysicalOptimizerRule};
use datafusion_nexus::planner::{
    FinalPlanDisposition, HostExecutionQosConfig, NexusFinalPlanRequirementRule, NexusNativeOptimizerConfig,
    NexusNativePhysicalOptimizerRule, classify_final_plan,
};

use super::{support::*, *};

const CPU_INPUT_BFS_SQL: &str = "SELECT vertex, distance, predecessor \
     FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
          '{\"construction_policy\":\"raw_libcugraph\",\
             \"include_edges_table\":\"allowed_edges\",\
             \"include_edge_src_col\":\"allowed_src\",\
             \"include_edge_dst_col\":\"allowed_dst\"}') \
     ORDER BY vertex";

fn cugraph_input_is_runtime_native(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> bool {
    if plan.downcast_ref::<NexusNativeExec>().is_some() {
        return true;
    }
    let children = plan.children();
    matches!(
        children.as_slice(),
        [child] if plan.name() == "CooperativeExec" && child.downcast_ref::<NexusNativeExec>().is_some()
    )
}

fn plan_contains_named_node(plan: &dyn datafusion::physical_plan::ExecutionPlan, name: &str) -> bool {
    plan.name() == name || plan.children().into_iter().any(|child| plan_contains_named_node(child.as_ref(), name))
}

fn register_edges_with_upstream_only_list(ctx: &SessionContext, table_name: &str) {
    let tag_field = Arc::new(Field::new("item", DataType::Int64, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("tags", DataType::List(Arc::clone(&tag_field)), false),
    ]));
    let tags = arrow::array::ListArray::new(
        tag_field,
        arrow::buffer::OffsetBuffer::new(vec![0_i32, 1, 2, 3].into()),
        Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            Arc::new(tags) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(table_name, Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
}

#[tokio::test]
async fn cugraph_sql_stages_cpu_edge_and_side_inputs_for_cpu_downstream() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_allowed_bfs_edges(&ctx, "allowed_edges");

    let df = ctx.sql(CPU_INPUT_BFS_SQL).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(find_composed_exec(plan.as_ref()).is_none(), "standalone SQL should keep the downstream plan on CPU");
    let cugraph = find_cugraph_algo_exec(plan.as_ref()).expect("physical plan should contain CugraphAlgoExec");
    assert!(!plan_contains_native_exec(cugraph), "MemTable edge and side inputs should remain CPU sources");

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await.unwrap();
    assert!(cudf::testing::arrow::arrow_host_import_call_count_for_test() >= 2);
    assert!(cudf::testing::arrow::arrow_host_export_call_count_for_test() >= 1);
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), None);
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
}

#[tokio::test]
async fn cugraph_sql_cpu_to_gpu_to_cpu_projects_away_unsupported_upstream_type() {
    let state = SessionStateBuilder::new_with_default_features().with_cugraph_sql(CugraphSqlConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    register_edges_with_upstream_only_list(&ctx, "raw_edges");
    ctx.sql(
        "CREATE VIEW filtered_edges AS \
         SELECT src, dst FROM raw_edges WHERE array_length(tags) > 0",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, out_degree \
             FROM cugraph_degrees_all('filtered_edges', 'src', 'dst') \
             WHERE out_degree > 0 ORDER BY vertex",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let cugraph =
        find_cugraph_algo_exec(plan.as_ref()).expect("physical plan should contain the explicit cuGraph island");
    let cugraph_children = cugraph.children();
    let [edge_input] = cugraph_children.as_slice() else {
        panic!("degrees should have one edge input");
    };

    assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Mixed);
    assert_ne!(
        plan.name(),
        "CugraphAlgoExec",
        "the public SQL filter/order must retain a CPU operator downstream of the graph island"
    );
    assert!(
        plan_contains_named_node(edge_input.as_ref(), "FilterExec"),
        "the unsupported list must be consumed by a CPU filter before the graph island"
    );
    assert!(
        edge_input.schema().field_with_name("tags").is_err(),
        "the optimized CPU projection must remove the upstream-only list before the cuGraph boundary"
    );
    assert!(!plan_contains_native_exec(edge_input.as_ref()), "the upstream view must execute through DataFusion CPU");

    let native_rule = NexusNativePhysicalOptimizerRule::new(NexusNativeOptimizerConfig::default());
    let selected = native_rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();
    assert_eq!(classify_final_plan(selected.as_ref()), FinalPlanDisposition::Mixed);
    assert!(
        find_cugraph_algo_exec(selected.as_ref()).is_some(),
        "ordinary relational selection must retain the explicit cuGraph island"
    );
    let report = native_rule.last_report().expect("native selection must retain a planning report");
    assert_eq!(
        report.selected_count(),
        1,
        "the post-island native consumer is an explicit declared GPU-island binding"
    );
    assert!(
        report
            .outcomes()
            .iter()
            .any(|outcome| matches!(outcome, datafusion_nexus::report::CandidateOutcome::Selected { .. })),
        "the report must retain the explicit-island selection: {report:?}"
    );

    let requirement =
        NexusFinalPlanRequirementRule::new(NexusNativeOptimizerConfig::default().with_no_datafusion_cpu_requirement());
    let error = requirement
        .optimize(selected, &ConfigOptions::default())
        .expect_err("NoDataFusionCpu must reject this exact already-planned mixed island");
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(nexus.fact_value("observed_final_plan_disposition"), Some("mixed"));

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert!(
        cudf::testing::arrow::arrow_host_import_call_count_for_test() >= 1,
        "the CPU edge view must cross the explicit host-to-GPU boundary"
    );
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 1, "the explicit island must record one terminal GPU observation");
    let observation = &metrics.graph_runtime_rows[0];
    assert_eq!(observation.algorithm, "degrees_all");
    assert_eq!(observation.runtime_outcome, "succeeded");
}

#[tokio::test]
async fn cugraph_nested_planning_uses_the_outer_host_execution_qos_permit_once() {
    let state = SessionStateBuilder::new_with_default_features()
        .with_nexus_native(NexusNativeOptimizerConfig::default())
        .with_host_execution_qos(HostExecutionQosConfig::new(std::num::NonZeroUsize::new(1).unwrap()))
        .with_cugraph_sql(CugraphSqlConfig::default())
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_edges_with_upstream_only_list(&ctx, "raw_edges");
    ctx.sql(
        "CREATE VIEW filtered_edges AS \
         SELECT src, dst FROM raw_edges WHERE array_length(tags) > 0",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let plan = ctx
        .sql(
            "SELECT vertex, out_degree \
             FROM cugraph_degrees_all('filtered_edges', 'src', 'dst') \
             WHERE out_degree > 0 ORDER BY vertex",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Mixed);

    let batches =
        tokio::time::timeout(Duration::from_secs(5), datafusion::physical_plan::collect(plan, ctx.task_ctx()))
            .await
            .expect("nested cuGraph planning must not reacquire the outer host QoS permit")
            .expect("mixed CPU -> cuGraph -> CPU statement should execute");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
}

#[tokio::test]
async fn cugraph_sql_keeps_native_inputs_device_resident_for_native_downstream() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_allowed_bfs_edges(&ctx, "allowed_edges");

    let df = ctx.sql(CPU_INPUT_BFS_SQL).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(find_composed_exec(plan.as_ref()).is_some(), "backend SQL should compose the downstream native plan");
    let cugraph = find_cugraph_algo_exec(plan.as_ref()).expect("physical plan should contain CugraphAlgoExec");
    let cugraph_display = displayable(cugraph).indent(false).to_string();
    let cugraph_children = cugraph.children();
    assert_eq!(cugraph_children.len(), 2, "BFS should retain its edge and side-input children");
    assert!(
        cugraph_children.iter().all(|child| cugraph_input_is_runtime_native(child.as_ref())),
        "backend-managed edge and side inputs should remain native or transparently cooperative:\n{cugraph_display}"
    );

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(
        cudf::testing::arrow::arrow_host_import_call_count_for_test(),
        2,
        "the two native literal sources should import once each without an extra cuGraph boundary"
    );
    assert_eq!(
        cudf::testing::arrow::arrow_host_export_call_count_for_test(),
        1,
        "cuGraph output must stay device-resident until the composed root materializes"
    );
    let batch = single_batch(&batches);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), None);
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
}

// Per-algorithm Python-oracle GPU tests for PageRank, BFS, and Louvain
// previously lived here. They have been subsumed by the registry-driven
// `tests/cugraph/cugraph_parity.rs` harness, which reads local oracle JSON
// produced by `fixture/graph/cugraph_oracle.py`. The composition test below
// stays in place and will move into the harness as a Milestone 2
// composition entry.

#[tokio::test]
async fn cugraph_sql_backend_session_records_query_service_admission_metrics() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-backend-admission");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let backend = NexusGpuBackend::builder()
        .admission(NexusAdmissionPolicy::default().with_slot_guarantee(SlotGuarantee::Optimistic))
        .build()
        .unwrap();
    let cugraph = CugraphSqlConfig::default()
        .with_native_execution_config(backend.optimizer_config().native_execution_config().clone());
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(config).with_default_features())
        .unwrap()
        .with_cugraph_sql(cugraph)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let df = ctx.sql("SELECT * FROM cugraph_degrees_all('tiny_directed_edges', 'src', 'dst')").await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    assert!(!batches.is_empty());

    let cugraph_plan = find_cugraph_algo_exec(plan.as_ref()).expect("physical plan should contain CugraphAlgoExec");
    let cugraph_display = displayable(cugraph_plan).indent(false).to_string();
    assert!(
        plan_contains_native_exec(cugraph_plan),
        "backend-managed cuGraph plan should have a native edge-source child:\n{cugraph_display}"
    );
    let metrics = plan.metrics().expect("backend-managed cuGraph plan should expose native metrics");

    assert_eq!(metric_usize(&metrics, "admission_sequence_observed"), 1);
    assert!(metric_usize(&metrics, "admission_sequence") > 0);
    assert_eq!(metric_usize(&metrics, "admission_wait_time_observed"), 1);
    assert_eq!(metric_usize(&metrics, "device_ordinal_observed"), 1);
    assert_eq!(metric_usize(&metrics, "query_grant_bytes_observed"), 1);
    assert!(metric_usize(&metrics, "query_grant_bytes") > 0);
    assert_eq!(plan_metric_usize(plan.as_ref(), "admission_sequence_observed"), 1);
    assert_eq!(plan_metric_usize(plan.as_ref(), "admission_wait_time_observed"), 1);
    assert_eq!(plan_metric_usize(plan.as_ref(), "query_grant_bytes_observed"), 1);
    tokio::time::timeout(Duration::from_secs(5), async {
        while backend.query_service().admission_snapshot().unwrap().active_queries != 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("ordered query-service cleanup should release admission");
    assert_eq!(backend.query_service().admission_snapshot().unwrap().active_queries, 0);

    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cugraph_sql_gpu_betweenness_reservation_reports_request_calibration_and_seed_mode() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    ctx.sql("SET datafusion_nexus.request.graph_raw_input_edges = '45564149'").await.unwrap().collect().await.unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_betweenness_centrality('raw_edges', 'src', 'dst', NULL, '{\"k\": 2}') \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    assert!(!batches.is_empty());

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    let graph_reservations = metrics.graph_reservation_rows;
    assert_eq!(graph_reservations.len(), 1);
    let row = &graph_reservations[0];
    assert_eq!(row.algorithm, "betweenness_centrality");
    assert_eq!(row.outcome, "admitted");
    assert_eq!(row.raw_input_edges, Some(45_564_149));
    assert_eq!(row.seed_count, Some(2));
    assert_eq!(row.seed_mode, "runtime_k_selection");
}

/// Runs the provisioned Python-cuGraph PageRank acceptance check.
///
/// This is an explicit tools command rather than a Cargo test because it
/// requires an external Python/cuGraph installation.
#[cfg_attr(test, allow(dead_code))]
pub async fn run_pagerank_filter_sort_limit_acceptance() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-pagerank-compose");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let oracle = python_pagerank_oracle(&path)
        .expect("Python cuGraph oracle is required for the selected PageRank composition acceptance test");
    eprintln!("Python cuGraph oracle versions: {}", oracle.versions);

    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = backend_cugraph_sql_state(SessionStateBuilder::new().with_config(config).with_default_features());
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql(
        "CREATE VIEW graph_edges AS \
         SELECT src, dst, weight FROM tiny_directed_edges WHERE active",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let df = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_pagerank('graph_edges', 'src', 'dst', 'weight') \
             WHERE value > 0 \
             ORDER BY value DESC, vertex ASC \
             LIMIT 3",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("NexusNativeComposedExec"),
        "PageRank post-algorithm filter/sort/limit should be a composed native GPU handoff:\n{plan_display}"
    );
    assert!(
        plan_display.contains("CugraphAlgoExec"),
        "composed native plan should keep cuGraph as the GPU-resident producer child:\n{plan_display}"
    );
    let composed =
        find_composed_exec(plan.as_ref()).expect("PageRank composition plan should contain NexusNativeComposedExec");
    let native_plan = composed.native_plan().to_string();
    for expected in ["GpuResidentSource", "Filter", "Sort", "Limit"] {
        assert!(
            native_plan.contains(expected),
            "composed native consumer should include {expected} over the cuGraph result:\n{native_plan}"
        );
    }

    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let actual = pagerank_rows(&batches);
    let mut expected = oracle.rows.clone();
    expected.retain(|row| row.value > 0.0);
    expected.sort_by(|left, right| {
        right.value.partial_cmp(&left.value).unwrap().then_with(|| left.vertex.cmp(&right.vertex))
    });
    expected.truncate(3);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.vertex, expected.vertex);
        let delta = (actual.value - expected.value).abs();
        assert!(
            delta <= 1.0e-6,
            "vertex {} composed PageRank mismatch: actual={} expected={} delta={}",
            actual.vertex,
            actual.value,
            expected.value,
            delta
        );
    }
    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cugraph_sql_gpu_personalized_pagerank_executes_with_side_input() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 2, 0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 1.0, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("ppr_edges", Arc::new(MemTable::try_new(edge_schema, vec![vec![edge_batch]]).unwrap())).unwrap();

    let seed_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let seed_batch = RecordBatch::try_new(
        Arc::clone(&seed_schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("ppr_seeds", Arc::new(MemTable::try_new(seed_schema, vec![vec![seed_batch]]).unwrap())).unwrap();

    let batches = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_personalized_pagerank(\
               'ppr_edges', 'src', 'dst', 'weight', \
               '{\"personalization_table\":\"ppr_seeds\",\
                 \"personalization_vertex_col\":\"vertex\",\
                 \"personalization_value_col\":\"value\"}'\
             ) \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let rows = pagerank_rows(&batches);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows.iter().map(|row| row.vertex).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert!(rows.iter().all(|row| row.value.is_finite() && row.value >= 0.0));
    let sum = rows.iter().map(|row| row.value).sum::<f64>();
    assert!((sum - 1.0).abs() <= 1.0e-6, "personalized PageRank scores should be normalized, got {sum}");
}

#[tokio::test]
async fn cugraph_sql_gpu_force_atlas2_executes_weighted_layout() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1, 2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 0, 3])) as ArrayRef,
            Arc::new(Float32Array::from(vec![1.0_f32, 1.5, 1.0, 0.5])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("layout_edges", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();

    let batches = ctx
        .sql(
            "SELECT vertex, x, y \
             FROM cugraph_force_atlas2('layout_edges', 'src', 'dst', 'weight', \
                  '{\"max_iter\":5,\"seed\":42}') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2, 3]);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Float32);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Float32);
    let x = batch.column(1).as_any().downcast_ref::<Float32Array>().unwrap();
    let y = batch.column(2).as_any().downcast_ref::<Float32Array>().unwrap();
    for row in 0..batch.num_rows() {
        assert!(x.value(row).is_finite(), "ForceAtlas2 x coordinate must be finite at row {row}");
        assert!(y.value(row).is_finite(), "ForceAtlas2 y coordinate must be finite at row {row}");
    }
}

#[tokio::test]
async fn cugraph_sql_gpu_similarity_rejects_actual_null_pair_values() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0_i32, 0, 1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1_i32, 2, 2, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("similarity_edges", Arc::new(MemTable::try_new(edge_schema, vec![vec![edge_batch]]).unwrap()))
        .unwrap();

    let pair_schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Int32, true),
        Field::new("second", DataType::Int32, true),
    ]));
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(0_i32), None])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1_i32), Some(2_i32)])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("candidate_pairs", Arc::new(MemTable::try_new(pair_schema, vec![vec![pair_batch]]).unwrap()))
        .unwrap();

    let error = ctx
        .sql(
            "SELECT * FROM cugraph_jaccard(\
             'similarity_edges', 'src', 'dst', NULL, \
             '{\"vertex_pairs_table\":\"candidate_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.code(), ErrorCode::GraphInputInvalid);
    assert_eq!(nexus.fact_value("algorithm"), Some("jaccard"));
    assert_eq!(nexus.fact_value("input_role"), Some("vertex_pairs"));
}

#[tokio::test]
async fn cugraph_sql_gpu_similarity_executes_weighted_jaccard_for_nullable_int64_pairs() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_weighted_similarity_edges_i64(&ctx, "similarity_edges");
    // A nullable Arrow schema is admitted when the materialized pair values are
    // all present. This is distinct from the null-rejection coverage above.
    register_i64_similarity_pairs(&ctx, "candidate_pairs", true, vec![Some(0)], vec![Some(1)]);

    let sql = "SELECT first, second, similarity \
               FROM cugraph_jaccard(\
                 'similarity_edges', 'src', 'dst', 'weight', \
                 '{\"vertex_pairs_table\":\"candidate_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}'\
               ) ORDER BY first, second";
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let batch = single_batch(&batches);
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Int64);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Float64);
    assert_eq!(int64_column(batch, 0), vec![0]);
    assert_eq!(int64_column(batch, 1), vec![1]);
    let score = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    // The fixture has weighted-intersection sum 2 and weighted-union max sum
    // 5 for pair (0, 1), so the hand-verifiable score is 2 / 5.
    assert!((score - 0.4).abs() <= 1.0e-6, "weighted Jaccard score was {score}");

    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 1);
    assert!(metrics.graph_runtime_rows[0].algorithm_metadata.contains("use_weight=true"));
}

#[tokio::test]
async fn cugraph_sql_gpu_similarity_short_circuits_empty_pairs_and_rejects_absent_endpoints() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_weighted_similarity_edges_i64(&ctx, "similarity_edges");
    register_i64_similarity_pairs(&ctx, "empty_pairs", false, vec![], vec![]);
    register_i64_similarity_pairs(&ctx, "absent_pairs", false, vec![Some(99)], vec![Some(0)]);

    let empty_sql = "SELECT * FROM cugraph_jaccard(\
                     'similarity_edges', 'src', 'dst', NULL, \
                     '{\"vertex_pairs_table\":\"empty_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}'\
                   )";
    let empty_df = ctx.sql(empty_sql).await.unwrap();
    let empty_plan = empty_df.create_physical_plan().await.unwrap();
    let empty_batches = datafusion::physical_plan::collect(Arc::clone(&empty_plan), ctx.task_ctx()).await.unwrap();
    let empty = single_batch(&empty_batches);
    assert_eq!(empty.schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(empty.schema().field(1).data_type(), &DataType::Int64);
    assert_eq!(empty.schema().field(2).data_type(), &DataType::Float64);
    assert_eq!(empty.num_rows(), 0);
    let empty_metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(empty_plan.as_ref()));
    assert_eq!(empty_metrics.graph_runtime_rows.len(), 1);
    assert!(empty_metrics.graph_runtime_rows[0].algorithm_metadata.contains("vertex_pairs_empty"));

    let error = ctx
        .sql(
            "SELECT * FROM cugraph_jaccard(\
               'similarity_edges', 'src', 'dst', NULL, \
               '{\"vertex_pairs_table\":\"absent_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.code(), ErrorCode::CudfInvalidInput);
    assert_eq!(nexus.fact_value("algorithm"), Some("jaccard"));
    assert_eq!(nexus.fact_value("input_role"), Some("vertex_pairs"));
}

#[tokio::test]
async fn cugraph_sql_gpu_similarity_reuses_a_keyable_graph_cache_between_cold_and_warm_queries() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-similarity-cache");
    let path = dir.join("similarity_edges.parquet");
    write_tiny_directed_edges(&path);

    let backend = NexusGpuBackend::builder()
        .admission(NexusAdmissionPolicy::default().with_slot_guarantee(SlotGuarantee::Optimistic))
        .query_service_resident_graph_cache_max_retained_bytes(Some(64 * 1024 * 1024))
        .build()
        .unwrap();
    let cugraph = CugraphSqlConfig::default()
        .with_native_execution_config(backend.optimizer_config().native_execution_config().clone());
    let state =
        backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().with_cugraph_sql(cugraph).build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("similarity_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    register_i64_similarity_pairs(&ctx, "candidate_pairs", false, vec![Some(0)], vec![Some(1)]);

    let sql = "SELECT * FROM cugraph_jaccard(\
               'similarity_edges', 'src', 'dst', 'weight', \
               '{\"vertex_pairs_table\":\"candidate_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}')";
    let cold_df = ctx.sql(sql).await.unwrap();
    let cold_plan = cold_df.create_physical_plan().await.unwrap();
    let cold_batches = datafusion::physical_plan::collect(Arc::clone(&cold_plan), ctx.task_ctx()).await.unwrap();
    assert_eq!(cold_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let cold_metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(cold_plan.as_ref()));
    assert_eq!(cold_metrics.graph_runtime_rows.len(), 1);
    assert!(
        cold_metrics.graph_runtime_rows[0].algorithm_metadata.contains("graph_cache=miss_inserted"),
        "{}",
        cold_metrics.graph_runtime_rows[0].algorithm_metadata
    );
    drop(cold_batches);
    drop(cold_plan);

    let warm_df = ctx.sql(sql).await.unwrap();
    let warm_plan = warm_df.create_physical_plan().await.unwrap();
    let warm_batches = datafusion::physical_plan::collect(Arc::clone(&warm_plan), ctx.task_ctx()).await.unwrap();
    assert_eq!(warm_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let warm_metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(warm_plan.as_ref()));
    assert_eq!(warm_metrics.graph_runtime_rows.len(), 1);
    assert!(
        warm_metrics.graph_runtime_rows[0].algorithm_metadata.contains("graph_cache=hit;"),
        "{}",
        warm_metrics.graph_runtime_rows[0].algorithm_metadata
    );

    tokio::time::timeout(Duration::from_secs(5), async {
        while backend.query_service().admission_snapshot().unwrap().active_queries != 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("ordered query-service cleanup should release admission");
    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_predicate_executes_end_to_end() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_allowed_bfs_edges(&ctx, "allowed_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), None);
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_numeric_source_list_executes_multi_source() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_disconnected_bfs_edges(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, reachable \
             FROM cugraph_bfs('raw_edges', [0, 10], 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
    assert_eq!(bfs_distance_for(batch, 10), Some(0));
    assert_eq!(bfs_distance_for(batch, 11), Some(1));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_source_list_executes_multi_source() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_disconnected_string_bfs_edges(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs('raw_edges', ['a', 'x'], 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "b"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(2));
    assert_eq!(string_bfs_distance_for(batch, "x"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "y"), Some(1));
    assert_eq!(string_bfs_predecessor_for(batch, "c"), Some("b".to_string()));
    assert_eq!(string_bfs_predecessor_for(batch, "y"), Some("x".to_string()));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_source_table_combines_with_target_path() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_disconnected_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0, 10]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}') \
             ORDER BY path_index",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2]);
    assert_eq!(int64_column(batch, 1), vec![0, 0, 0]);
    assert_eq!(int64_column(batch, 2), vec![3, 3, 3]);
    assert_eq!(int64_column(batch, 3), vec![0, 2, 3]);
    assert_eq!(int64_column(batch, 4), vec![0, 1, 2]);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_absent_source_returns_empty_batch() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![99]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    assert_empty_i64_path_batch(single_batch(&batches));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_empty_source_table_returns_empty_batch() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", Vec::new());
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    assert_empty_i64_path_batch(single_batch(&batches));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_empty_source_with_bad_include_vertices_is_rejected() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", Vec::new());
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);
    register_i64_vertex_values(&ctx, "included", "vertex_id", vec![123]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\",\
                     \"include_vertices_table\":\"included\",\
                     \"include_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(nexus.fact_value("predicate_role"), Some("include_vertices"));
    let native_message =
        nexus.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_absent_source_with_bad_include_vertices_is_rejected() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![99]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);
    register_i64_vertex_values(&ctx, "included", "vertex_id", vec![123]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\",\
                     \"include_vertices_table\":\"included\",\
                     \"include_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(nexus.fact_value("predicate_role"), Some("include_vertices"));
    let native_message =
        nexus.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_partial_absent_sources_are_rejected() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0, 99]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.fact_value("reason"), Some("partial_source_vertices_absent"));
    assert_eq!(nexus.fact_value("predicate_role"), Some("source_vertices"));
    let native_message =
        nexus.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_absent_target_returns_empty_batch() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![99]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    assert_empty_i64_path_batch(single_batch(&batches));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_empty_edge_source_returns_empty_batch() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_empty_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    assert_empty_i64_path_batch(single_batch(&batches));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_empty_edge_source_with_bad_include_vertices_is_rejected() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_empty_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);
    register_i64_vertex_values(&ctx, "included", "vertex_id", vec![123]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\",\
                     \"include_vertices_table\":\"included\",\
                     \"include_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(nexus.fact_value("predicate_role"), Some("include_vertices"));
    let native_message =
        nexus.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_source_table_maps_logical_ids_end_to_end() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_vertex_memtable(&ctx, "bfs_sources", "vertex_id", &["a"]);

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs('raw_edges', NULL, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"source_vertices_table\":\"bfs_sources\",\
                     \"source_vertex_col\":\"vertex_id\",\
                     \"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "b"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "d"), Some(2));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_include_edges_maps_logical_ids_end_to_end() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs('raw_edges', 'a', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\",\
                     \"output_mode\":\"normalized\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "b"), None);
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "d"), Some(2));
    assert_eq!(string_bfs_predecessor_for(batch, "d"), Some("c".to_string()));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_nullable_edges_drop_actual_null_rows() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_nullable_string_bfs_edges_with_nulls(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs('raw_edges', 'a', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"normalized\"}') \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "b"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(2));
    assert_eq!(string_bfs_distance_for(batch, "d"), Some(3));
    assert_eq!(string_bfs_distance_for(batch, "x"), None);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_reconstructs_utf8_target_path_end_to_end() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");
    register_string_vertex_memtable(&ctx, "targets", "vertex_id", &["d"]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs('raw_edges', 'a', 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"output_mode\":\"path\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}') \
             ORDER BY path_index",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    let batch = single_batch(&batches);
    assert_eq!(string_path_column(batch, 1), vec!["a", "a", "a"]);
    assert_eq!(string_path_column(batch, 2), vec!["d", "d", "d"]);
    assert_eq!(string_path_column(batch, 3), vec!["a", "c", "d"]);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2]);
    assert_eq!(int64_column(batch, 4), vec![0, 1, 2]);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_drops_actual_null_predicate_rows() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_nullable_allowed_bfs_edges_with_nulls(&ctx, "allowed_edges");

    // The whitelist enumerates 0->2 and 2->3 plus a null row that cannot name
    // any edge. The null row is dropped (not rejected), leaving the same
    // policy as the clean include_edges test: reachable {0,2,3}, 1 unreachable.
    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), None);
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_target_vertices_with_null_values_is_rejected() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_nullable_vertex_memtable_with_nulls(&ctx, "targets", "vertex_id");

    let error = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"target_vertices_table\":\"targets\",\
                     \"target_vertex_col\":\"vertex_id\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    // Vertex-set predicate values that are actually null are rejected at
    // execution with the value-level diagnostic code carrying the role/column.
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.fact_value("reason"), Some("predicate_column_contains_nulls"));
    assert_eq!(nexus.fact_value("predicate_role"), Some("target_vertices"));
    assert_eq!(nexus.fact_value("column"), Some("vertex_id"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_empty_after_drop_is_explicit_error() {
    let state = default_backend_cugraph_sql_state();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_all_null_allowed_bfs_edges(&ctx, "allowed_edges");

    let error = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
                  '{\"construction_policy\":\"raw_libcugraph\",\
                     \"include_edges_table\":\"allowed_edges\",\
                     \"include_edge_src_col\":\"allowed_src\",\
                     \"include_edge_dst_col\":\"allowed_dst\"}')",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    // A whitelist that collapses to empty after dropping null rows is a
    // deliberate error, never a silent empty BFS.
    let nexus = typed_nexus_error(&error);
    assert_eq!(nexus.fact_value("predicate_role"), Some("include_edges"));
    assert!(
        nexus.fact_value("native_message").is_some_and(|message| message.contains("include_edges whitelist is empty")),
        "unexpected typed Nexus error: {nexus:?}"
    );
}
