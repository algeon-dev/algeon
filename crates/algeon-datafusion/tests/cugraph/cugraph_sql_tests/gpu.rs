use std::{fs, sync::Arc, time::Duration};

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuDeviceInstallationBounds, AlgeonGpuDeviceProfile, GpuDeviceMemoryLimit},
    gpu_coverage::validate_query,
    native_exec::below_native_admission,
    planner::{
        AlgeonGpuExecutionMode, AlgeonNativePhysicalOptimizerRule, DataFusionCpuExecutionPolicy, FinalPlanDisposition,
        classify_final_plan,
    },
};
use arrow::{array::ListArray, buffer::OffsetBuffer};
use cudf::{
    frame::dataframe::HostBatchCursorConfig,
    testing::arrow::{
        arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
        reset_arrow_host_interop_observation_for_test,
    },
};
use datafusion::{
    common::{Constraint, Constraints, config::ConfigOptions},
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::SessionStateBuilder,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, collect},
};
use futures::StreamExt;
use query_engine::observability::{AttemptCapacityDisposition, AttemptCleanupDisposition, RuntimeTerminalOutcome};
use tokio::{
    sync::Barrier,
    task::yield_now,
    time::{sleep, timeout},
};

use super::{support::*, *};

const CPU_INPUT_BFS_SQL: &str = "SELECT vertex, distance, predecessor \
     FROM cugraph_bfs(\
          edges => (SELECT src, dst FROM raw_edges), source_vertex => 0, \
          construction_policy => 'raw_libcugraph') \
     ORDER BY vertex";

const RAW_BFS_TOP_K_SQL: &str = "SELECT vertex, distance, predecessor \
     FROM cugraph_bfs(\
          edges => (SELECT src, dst FROM phase_c_top_k_edges), source_vertex => 0, \
          construction_policy => 'raw_libcugraph') \
     ORDER BY distance \
     LIMIT 100";

const PAGERANK_TOP_K_SQL: &str = "SELECT vertex, value \
     FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM phase_c_top_k_edges)) \
     ORDER BY value DESC \
     LIMIT 100";

const PAGERANK_FLOAT32_TOP_K_SQL: &str = "SELECT vertex, value \
     FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM phase_c_top_k_edges_float32)) \
     ORDER BY value DESC \
     LIMIT 100";

const PAGERANK_FORCEATLAS2_JOIN_SQL: &str = "WITH pagerank AS (\n\
        SELECT vertex, value AS pagerank_value\n\
        FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM graph_edges))\n\
    ), layout AS (\n\
        SELECT vertex, x, y\n\
        FROM cugraph_force_atlas2(edges => (SELECT src, dst, weight FROM graph_edges))\n\
    )\n\
    SELECT p.vertex, p.pagerank_value, l.x, l.y\n\
    FROM layout l JOIN pagerank p ON p.vertex = l.vertex\n\
    WHERE p.pagerank_value > 0";

fn cugraph_input_is_runtime_native(plan: &dyn ExecutionPlan) -> bool {
    if plan.downcast_ref::<AlgeonNativeExec>().is_some() {
        return true;
    }
    let children = plan.children();
    matches!(
        children.as_slice(),
        [child] if plan.name() == "CooperativeExec" && child.downcast_ref::<AlgeonNativeExec>().is_some()
    )
}

fn identity_value(array: &dyn Array, row: usize) -> i64 {
    if array.is_null(row) {
        return i64::MIN;
    }
    match array.data_type() {
        DataType::Utf8 => {
            i64::from(array.as_any().downcast_ref::<StringArray>().unwrap().value(row).as_bytes()[0] - b'a' + 1)
        },
        DataType::Int64 => {
            let value = array.as_any().downcast_ref::<Int64Array>().unwrap().value(row);
            if value < 0 { i64::MIN } else { value }
        },
        other => panic!("unexpected vertex identity dtype {other:?}"),
    }
}

async fn identity_rows(ctx: &SessionContext, sql: &str, identity_columns: usize) -> Vec<Vec<i64>> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push((0..identity_columns).map(|column| identity_value(batch.column(column).as_ref(), row)).collect());
        }
    }
    rows.sort_unstable();
    rows
}

fn plan_contains_named_node(plan: &dyn ExecutionPlan, name: &str) -> bool {
    plan.name() == name || plan.children().into_iter().any(|child| plan_contains_named_node(child.as_ref(), name))
}

fn register_edges_with_upstream_only_list(ctx: &SessionContext, table_name: &str) {
    let tag_field = Arc::new(Field::new("item", DataType::Int64, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("tags", DataType::List(Arc::clone(&tag_field)), false),
    ]));
    let tags = ListArray::new(
        tag_field,
        OffsetBuffer::new(vec![0_i32, 1, 2, 3].into()),
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

fn register_phase_c_top_k_edges(ctx: &SessionContext, table_name: &str, weight_type: DataType) {
    let sources = (0_i64..=256).collect::<Vec<_>>();
    let mut destinations = (1_i64..=256).collect::<Vec<_>>();
    destinations.push(0);
    let weights = match &weight_type {
        DataType::Float32 => Arc::new(Float32Array::from(vec![1.0_f32; sources.len()])) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(vec![1.0_f64; sources.len()])) as ArrayRef,
        other => panic!("PageRank fixture does not support {other:?}"),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", weight_type, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(sources)) as ArrayRef,
            Arc::new(Int64Array::from(destinations)) as ArrayRef,
            weights,
        ],
    )
    .unwrap();
    ctx.register_table(table_name, Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
}

#[tokio::test]
async fn cugraph_sql_cpu_to_gpu_to_cpu_projects_away_unsupported_upstream_type() {
    let (backend, ctx) = backend_cugraph_sql_context_with_mode(
        SessionStateBuilder::new_with_default_features(),
        AlgeonGpuExecutionMode::FunctionsOnly,
    );
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
             FROM cugraph_degrees_all(edges => (SELECT src, dst FROM filtered_edges)) \
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

    let reject_rule = AlgeonNativePhysicalOptimizerRule::new(
        backend.optimizer_config().clone().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
    );
    let error = reject_rule
        .optimize(Arc::clone(&plan), &ConfigOptions::default())
        .expect_err("NoDataFusionCpu must reject this exact already-planned mixed island");
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(algeon.fact_value("observed_final_plan_disposition"), Some("mixed"));

    reset_arrow_host_interop_observation_for_test();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert!(
        arrow_host_import_call_count_for_test() >= 1,
        "the CPU edge view must cross the explicit host-to-GPU boundary"
    );
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 1, "the explicit island must record one terminal GPU observation");
    let observation = &metrics.graph_runtime_rows[0];
    assert_eq!(observation.algorithm, "degrees_all");
    assert_eq!(observation.runtime_outcome, "succeeded");
}

#[tokio::test]
async fn functions_only_keeps_sql_limit_outside_the_cugraph_function() {
    let (_backend, ctx) = backend_cugraph_sql_context_with_mode(
        SessionStateBuilder::new_with_default_features(),
        AlgeonGpuExecutionMode::FunctionsOnly,
    );
    register_edge_memtable(&ctx, "limit_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    let plan = ctx
        .sql("SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM limit_edges)) LIMIT 2")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    assert!(
        plan_contains_named_node(plan.as_ref(), "GlobalLimitExec")
            || plan_contains_named_node(plan.as_ref(), "LocalLimitExec"),
        "the SQL LIMIT must remain a DataFusion operator outside the cuGraph function"
    );
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
}

async fn assert_repeated_cugraph_cte_executes(mode: AlgeonGpuExecutionMode, expected_gpu_executions: usize) {
    let (_backend, ctx) = backend_cugraph_sql_context_with_mode(SessionStateBuilder::new_with_default_features(), mode);
    register_edge_memtable(&ctx, "shared_cte_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    let plan = ctx
        .sql(
            "WITH graph_result AS (\
                 SELECT vertex, in_degree, out_degree \
                 FROM cugraph_degrees_all(edges => (SELECT src, dst, weight FROM shared_cte_edges))\
             ) \
             SELECT a.vertex, a.out_degree, b.in_degree \
             FROM graph_result a JOIN graph_result b ON a.vertex = b.vertex \
             ORDER BY a.vertex",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(
        rendered.matches("CugraphAlgoExec: algorithm=DegreesAll").count(),
        expected_gpu_executions,
        "the execution policy must preserve its declared CTE sharing behavior:\n{rendered}"
    );
    let batches = collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .expect("both references to one cuGraph table provider must execute");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(
        metrics.graph_runtime_rows.len(),
        expected_gpu_executions,
        "each unshared graph occurrence must execute once"
    );
}

#[tokio::test]
async fn functions_only_executes_repeated_cugraph_cte_occurrences() {
    assert_repeated_cugraph_cte_executes(AlgeonGpuExecutionMode::FunctionsOnly, 2).await;
}

#[tokio::test]
async fn native_preferred_executes_repeated_cugraph_cte_occurrences() {
    assert_repeated_cugraph_cte_executes(AlgeonGpuExecutionMode::NativePreferred, 1).await;
}

#[tokio::test]
async fn cugraph_sql_keeps_native_inputs_device_resident_for_native_downstream() {
    let (backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges", DataType::Float64);

    let df = ctx.sql(CPU_INPUT_BFS_SQL).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let cugraph = find_cugraph_algo_exec(plan.as_ref()).expect("physical plan should contain CugraphAlgoExec");
    let cugraph_display = displayable(cugraph).indent(false).to_string();
    let cugraph_children = cugraph.children();
    assert_eq!(cugraph_children.len(), 1, "BFS should retain its edge child");
    assert!(
        cugraph_children.iter().all(|child| cugraph_input_is_runtime_native(child.as_ref())),
        "backend-managed edge and side inputs should remain native or transparently cooperative:\n{cugraph_display}"
    );

    reset_arrow_host_interop_observation_for_test();
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(
        arrow_host_import_call_count_for_test(),
        1,
        "the native edge source should import once without an extra cuGraph boundary"
    );
    assert_eq!(
        arrow_host_export_call_count_for_test(),
        1,
        "cuGraph output must stay device-resident until the composed root materializes"
    );
    let batch = single_batch(&batches);
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 1), Some(1));
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));

    let top_k_attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let top_k_completion = top_k_attempt.completion();
    let top_k_plan = top_k_attempt.plan_sql(&ctx, RAW_BFS_TOP_K_SQL).await.unwrap();
    assert_eq!(schema_field_names(top_k_plan.schema().as_ref()), vec!["vertex", "distance", "predecessor"]);
    assert!(top_k_plan.schema().field_with_name("predecessor").unwrap().is_nullable());
    let mut top_k_stream = top_k_attempt.execute_physical_plan(Arc::clone(&top_k_plan), &ctx).unwrap();
    let mut top_k_batches = Vec::new();
    while let Some(batch) = top_k_stream.next().await {
        top_k_batches.push(batch.expect("raw BFS top-k must execute through the admitted composed schedule"));
    }
    let report = top_k_completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some(), "raw BFS top-k must execute under one admitted fixed cap");
    assert_eq!(top_k_batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 100);
    let distances = top_k_batches
        .iter()
        .flat_map(|batch| {
            batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap().values().iter().copied().collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(distances, (0_i64..100).collect::<Vec<_>>());
}

#[tokio::test]
async fn cugraph_sql_gpu_pagerank_forceatlas2_join_executes_as_one_scheduled_composed_attempt() {
    let (backend, ctx) = default_backend_cugraph_sql_context();
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges", DataType::Float64);
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges_float32", DataType::Float32);
    let plan = ctx.sql(PAGERANK_FORCEATLAS2_JOIN_SQL).await.unwrap().create_physical_plan().await.unwrap();
    let composed =
        find_composed_exec(plan.as_ref()).expect("the canonical two-producer topology lowers to one composed root");
    assert_eq!(composed.children().len(), 2, "the composed root keeps both scheduled graph occurrences");

    reset_arrow_host_interop_observation_for_test();
    let batches = collect(plan, ctx.task_ctx())
        .await
        .expect("the complete composed schedule must execute both graph producers and their join");
    let batch = single_batch(&batches);
    assert_eq!(schema_field_names(batch.schema().as_ref()), vec!["vertex", "pagerank_value", "x", "y"]);
    assert_eq!(batch.num_rows(), 2);
    let mut vertices = int64_column(batch, 0);
    vertices.sort_unstable();
    assert_eq!(vertices, vec![0, 1]);
    let pagerank = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    let x = batch.column(2).as_any().downcast_ref::<Float32Array>().unwrap();
    let y = batch.column(3).as_any().downcast_ref::<Float32Array>().unwrap();
    for row in 0..batch.num_rows() {
        assert!(pagerank.value(row).is_finite() && pagerank.value(row) > 0.0);
        assert!(x.value(row).is_finite());
        assert!(y.value(row).is_finite());
    }

    for (_weight_type, sql) in
        [(DataType::Float32, PAGERANK_FLOAT32_TOP_K_SQL), (DataType::Float64, PAGERANK_TOP_K_SQL)]
    {
        let top_k_attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
        let top_k_completion = top_k_attempt.completion();
        let top_k_plan = top_k_attempt.plan_sql(&ctx, sql).await.unwrap();
        assert_eq!(schema_field_names(top_k_plan.schema().as_ref()), vec!["vertex", "value"]);
        assert_eq!(top_k_plan.schema().field_with_name("value").unwrap().data_type(), &DataType::Float64);
        let mut top_k_stream = top_k_attempt.execute_physical_plan(Arc::clone(&top_k_plan), &ctx).unwrap();
        let mut top_k_batches = Vec::new();
        while let Some(batch) = top_k_stream.next().await {
            top_k_batches.push(batch.expect("PageRank top-k must execute through the admitted composed schedule"));
        }
        let report = top_k_completion.await;
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        assert!(report.admission.is_some(), "PageRank top-k must execute under one admitted fixed cap");
        let rows = pagerank_rows(&top_k_batches);
        assert_eq!(rows.len(), 100);
        assert!(rows.windows(2).all(|pair| pair[0].value >= pair[1].value));
    }

    for (algorithm, sql) in [
        (
            "PageRank",
            "SELECT vertex, value FROM cugraph_pagerank(\
             edges => (SELECT src, dst, weight FROM phase_c_top_k_edges)) LIMIT 3",
        ),
        (
            "Eigenvector",
            "SELECT vertex, value FROM cugraph_eigenvector_centrality(\
             edges => (SELECT src, dst, weight FROM phase_c_top_k_edges)) LIMIT 3",
        ),
    ] {
        let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
        let batches = collect(plan, ctx.task_ctx())
            .await
            .unwrap_or_else(|error| panic!("{algorithm} bare limit must execute: {error}"));
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }
}

#[tokio::test]
async fn cugraph_sql_gpu_composed_attempt_retains_context_grant_and_gpu_results_until_consumer_output_drains() {
    let (backend, ctx) = default_backend_cugraph_sql_context();
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    reset_arrow_host_interop_observation_for_test();
    let plan = attempt.plan_sql(&ctx, PAGERANK_FORCEATLAS2_JOIN_SQL).await.unwrap();
    let mut stream = attempt.execute_physical_plan(Arc::clone(&plan), &ctx).unwrap();

    // The first result batch is materialized only after both graph producer
    // handles feed the native join consumer. The deferred cursor still owns
    // the final GPU result and its admitted context until the stream reaches
    // EOF, so admission must remain live here.
    let first = stream.next().await.expect("the composed consumer must produce a host batch").unwrap();
    assert_eq!(first.num_rows(), 2);
    assert!(!completion.is_ready(), "the attempt cannot seal while its deferred native cursor remains live");

    let admission = backend.admission_snapshot().unwrap();
    assert_eq!(admission.active_queries, 1, "the context lease must survive the first deferred host batch");
    let live_ledgers = backend.device_ledger_reports().unwrap();
    let [live_ledger] = live_ledgers.as_slice() else {
        panic!("one-device fixture must expose exactly one ledger: {live_ledgers:#?}");
    };
    assert_eq!(live_ledger.active_attempts, 1);
    assert_eq!(live_ledger.grants.len(), 1, "the immutable attempt grant remains owned by the deferred cursor");
    let live_grant = &live_ledger.grants[0];
    assert!(live_grant.grant_bytes > 0);
    let live_ledger_sequence = live_ledger.sequence;

    while let Some(batch) = stream.next().await {
        batch.expect("the composed consumer stream must complete without a second execution path");
    }
    // The deferred consumer's metrics reach DataFusion once its cursor ends.
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 2, "the composed attempt must record both graph producers");
    assert!(metrics.graph_runtime_rows.iter().all(|row| {
        row.runtime_outcome == "succeeded"
            && row.input_boundary == "gpu_resident"
            && row.output_boundary == "gpu_resident"
            && row.device_to_host_materializations == 0
    }));

    let dynamic_filter_producers =
        metrics.dynamic_filter_runtime_rows.iter().filter(|row| row.event == "produced").collect::<Vec<_>>();
    assert_eq!(
        metrics.dynamic_filter_produced_count, 1,
        "the filter predicate must produce one bounded control payload"
    );
    assert_eq!(dynamic_filter_producers.len(), 1, "exactly one dynamic-filter producer may export a key payload");
    assert_eq!(dynamic_filter_producers[0].producer_kind, "hash_join_build");
    assert!(dynamic_filter_producers[0].producer_rows.is_some_and(|rows| rows > 0));
    assert_eq!(
        plan_metric_usize(plan.as_ref(), "compiled_pipeline_host_arrow_sinks"),
        1,
        "the composed consumer must retain one deferred host-output sink",
    );
    assert_eq!(
        metrics.device_to_host_materializations,
        dynamic_filter_producers.len() + 1,
        "one dynamic-filter key export plus the deferred composed consumer output must be the only D2H boundaries",
    );
    assert_eq!(
        arrow_host_export_call_count_for_test(),
        metrics.device_to_host_materializations,
        "the cuDF interop observation must match the categorized native D2H boundaries",
    );

    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert_eq!(
        report.cleanup_disposition,
        AttemptCleanupDisposition::Completed,
        "the terminal owner must complete one context/grant cleanup before it seals",
    );
    assert_eq!(
        report.capacity_disposition,
        AttemptCapacityDisposition::Released,
        "the successful attempt must release rather than retain its immutable grant",
    );
    assert_eq!(report.cleanup_failure_count, 0);
    assert!(report.cleanup_failure_count_complete);

    let admission_fact = report.admission.as_ref().expect("the composed query must be admitted");
    assert_eq!(report.device_ordinal, Some(live_ledger.device_ordinal));
    assert_eq!(admission_fact.device_ordinal, live_ledger.device_ordinal);
    assert_eq!(admission_fact.grant_bytes, live_grant.grant_bytes);

    timeout(Duration::from_secs(5), async {
        loop {
            let admission = backend.admission_snapshot().unwrap();
            let released_ledgers = backend.device_ledger_reports().unwrap();
            if admission.active_queries == 0
                && released_ledgers.iter().all(|ledger| ledger.active_attempts == 0 && ledger.grants.is_empty())
            {
                break;
            }
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("stream EOF must release the context lease and immutable grant exactly once");

    let released_ledgers = backend.device_ledger_reports().unwrap();
    let [released_ledger] = released_ledgers.as_slice() else {
        panic!("one-device fixture must expose exactly one ledger after cleanup: {released_ledgers:#?}");
    };
    assert_eq!(released_ledger.active_attempts, 0);
    assert_eq!(released_ledger.live_grants_bytes, 0);
    assert!(released_ledger.grants.is_empty());
    assert!(
        released_ledger.sequence > live_ledger_sequence,
        "the live grant must transition to one completed release after the cursor drains",
    );
}

#[tokio::test]
async fn cugraph_sql_gpu_composed_attempt_returns_cache_checkouts_before_deferred_output_and_preserves_cache_affinity()
{
    const CACHE_CAP_BYTES: u64 = 16 * 1024 * 1024;

    let dir = unique_temp_dir("algeon-datafusion-cugraph-composed-cache-lifecycle");
    let path = dir.join("graph_edges.parquet");
    write_tiny_directed_edges(&path);

    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_default_query_limit_bytes(128 * 1024 * 1024)
        .with_cache_cap_bytes(CACHE_CAP_BYTES);
    let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let cugraph = CugraphSqlConfig::default();
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(config).with_default_features())
        .unwrap()
        .try_with_cugraph_sql(cugraph)
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("graph_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    // Prime the graph cache from the production two-producer SQL shape. The
    // local Parquet source gives PageRank a stable cacheable identity;
    // ForceAtlas2 currently stays on its owning uncached graph-builder path.
    let primer = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let primer_completion = primer.completion();
    let mut primer_stream = primer.sql(&ctx, PAGERANK_FORCEATLAS2_JOIN_SQL).await.unwrap();
    while let Some(batch) = primer_stream.next().await {
        batch.expect("the cache-primer attempt must complete");
    }
    let primer_report = primer_completion.await;
    assert!(matches!(primer_report.outcome, RuntimeTerminalOutcome::Succeeded));

    let primed_ledgers = backend.device_ledger_reports().unwrap();
    let [primed_ledger] = primed_ledgers.as_slice() else {
        panic!("one-device fixture must expose one cache-backed ledger: {primed_ledgers:#?}");
    };
    let primed_caches = backend.device_cache_reports().unwrap();
    let [primed_cache] = primed_caches.as_slice() else {
        panic!("one cache-enabled device must expose one cache report: {primed_caches:#?}");
    };
    assert_eq!(primed_cache.device_ordinal, primed_ledger.device_ordinal);
    assert_eq!(primed_cache.cache_cap_bytes, CACHE_CAP_BYTES);
    assert!(primed_cache.ledger_sequence >= primed_ledger.sequence);
    assert_eq!(primed_cache.ledger_evictable_resident_bytes, primed_ledger.cache_evictable_resident_bytes);
    assert_eq!(primed_cache.ledger_pinned_in_use_bytes, primed_ledger.cache_pinned_in_use_bytes);
    assert!(primed_cache.ledger_evictable_resident_bytes > 0, "the primed graphs must remain cache-owned");
    assert_eq!(primed_cache.ledger_pinned_in_use_bytes, 0, "the primer must not strand a graph checkout pin");
    assert_eq!(primed_cache.graph.insertions, 1, "the cacheable PageRank producer must retain one parked graph");
    assert_eq!(primed_cache.graph.available_entries, 1);
    assert_eq!(primed_cache.graph.checked_out_entries, 0);
    assert_eq!(primed_cache.graph.preparation_pins, 0);

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan = attempt.plan_sql(&ctx, PAGERANK_FORCEATLAS2_JOIN_SQL).await.unwrap();
    let mut stream = attempt.execute_physical_plan(Arc::clone(&plan), &ctx).unwrap();
    let first =
        stream.next().await.expect("the composed cache-hit attempt must yield its deferred host output").unwrap();
    assert!(first.num_rows() > 0);
    assert!(!completion.is_ready(), "the deferred host cursor must still own the live attempt after its first batch");

    let live_ledgers = backend.device_ledger_reports().unwrap();
    let [live_ledger] = live_ledgers.as_slice() else {
        panic!("one-device fixture must expose one live cache-backed ledger: {live_ledgers:#?}");
    };
    assert_eq!(live_ledger.active_attempts, 1);
    assert_eq!(live_ledger.grants.len(), 1);
    let live_caches = backend.device_cache_reports().unwrap();
    let [live_cache] = live_caches.as_slice() else {
        panic!("one cache-enabled device must expose one live cache report: {live_caches:#?}");
    };
    assert_eq!(live_cache.device_ordinal, live_ledger.device_ordinal);
    assert_eq!(live_cache.allocation_domain_id, primed_cache.allocation_domain_id);
    assert_eq!(live_cache.cache_cap_bytes, CACHE_CAP_BYTES);
    assert!(live_cache.ledger_sequence >= live_ledger.sequence);
    assert_eq!(live_cache.ledger_evictable_resident_bytes, live_ledger.cache_evictable_resident_bytes);
    assert_eq!(live_cache.ledger_pinned_in_use_bytes, live_ledger.cache_pinned_in_use_bytes);
    assert_eq!(live_cache.graph.lookup_hits, primed_cache.graph.lookup_hits + 1);
    assert_eq!(live_cache.graph.available_entries, 1);
    assert_eq!(
        live_cache.graph.checked_out_entries, 0,
        "the PageRank graph checkout must return before the composed host cursor drains"
    );
    assert_eq!(live_cache.graph.preparation_pins, 0);
    assert_eq!(
        live_cache.ledger_pinned_in_use_bytes, 0,
        "the retained consumer output must not retain a graph-cache pin"
    );
    assert!(
        live_cache.ledger_evictable_resident_bytes > 0,
        "the returned graphs must remain owned by the cache domain"
    );

    while let Some(batch) = stream.next().await {
        batch.expect("the cache-hit composed consumer stream must complete");
    }
    let report = completion.await;
    let admission = report.admission.as_ref().expect("cache-hit attempt must retain its admission fact");
    assert_eq!(
        admission.device_ordinal, live_cache.device_ordinal,
        "the cache domain and admitted attempt must retain device affinity"
    );
    let evidence = report.resource_evidence.expect("cache-backed attempt must seal resource evidence");
    assert_eq!(evidence.cache_evictable_resident_bytes.value, Some(live_ledger.cache_evictable_resident_bytes));
    assert_eq!(evidence.cache_pinned_in_use_bytes.value, Some(0));

    let released_caches = backend.device_cache_reports().unwrap();
    let [released_cache] = released_caches.as_slice() else {
        panic!("one cache-enabled device must retain one report after attempt cleanup: {released_caches:#?}");
    };
    assert_eq!(released_cache.allocation_domain_id, live_cache.allocation_domain_id);
    assert_eq!(released_cache.graph.lookup_hits, live_cache.graph.lookup_hits);
    assert_eq!(released_cache.graph.available_entries, 1);
    assert_eq!(released_cache.graph.checked_out_entries, 0);
    assert_eq!(released_cache.graph.preparation_pins, 0);

    fs::remove_dir_all(dir).ok();
}

// Per-algorithm Python-oracle GPU tests for PageRank, BFS, and Louvain
// previously lived here. They have been subsumed by the registry-driven
// `tests/cugraph/cugraph_parity.rs` harness, which reads local oracle JSON
// produced by `fixture/graph/cugraph_oracle.py`. The composition test below
// stays in place until the harness has a composition entry for it.

#[tokio::test]
async fn cugraph_sql_backend_session_records_query_service_admission_metrics() {
    let dir = unique_temp_dir("algeon-datafusion-cugraph-backend-admission");
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);

    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let cugraph = CugraphSqlConfig::default();
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(config).with_default_features())
        .unwrap()
        .try_with_cugraph_sql(cugraph)
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("tiny_directed_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan = attempt
        .plan_sql(&ctx, "SELECT * FROM cugraph_degrees_all(edges => (SELECT src, dst FROM tiny_directed_edges))")
        .await
        .unwrap();
    let mut stream = attempt.execute_physical_plan(Arc::clone(&plan), &ctx).unwrap();
    let mut batches = Vec::new();
    while let Some(batch) = stream.next().await {
        batches.push(batch.unwrap());
    }
    let report = completion.await;
    assert!(!batches.is_empty());
    let physical = report.timestamps.physical_execution_started_ns.expect("cuGraph attempt records physical start");
    let gpu = report.timestamps.gpu_execution_started_ns.expect("cuGraph attempt records GPU start");
    assert!(gpu >= physical);

    let cugraph_plan = find_cugraph_algo_exec(plan.as_ref()).expect("physical plan should contain CugraphAlgoExec");
    let cugraph_display = displayable(cugraph_plan).indent(false).to_string();
    assert!(
        plan_contains_native_exec(cugraph_plan),
        "backend-managed cuGraph plan should have a native edge-source child:\n{cugraph_display}"
    );
    let metrics =
        below_native_admission(&plan).metrics().expect("backend-managed cuGraph plan should expose native metrics");

    assert_eq!(metric_usize(&metrics, "admission_sequence_observed"), 1);
    assert!(metric_usize(&metrics, "admission_sequence") > 0);
    assert_eq!(metric_usize(&metrics, "admission_wait_time_observed"), 1);
    assert_eq!(metric_usize(&metrics, "device_ordinal_observed"), 1);
    assert_eq!(metric_usize(&metrics, "query_grant_bytes_observed"), 1);
    assert!(metric_usize(&metrics, "query_grant_bytes") > 0);
    assert_eq!(plan_metric_usize(plan.as_ref(), "admission_sequence_observed"), 1);
    assert_eq!(plan_metric_usize(plan.as_ref(), "admission_wait_time_observed"), 1);
    assert_eq!(plan_metric_usize(plan.as_ref(), "query_grant_bytes_observed"), 1);
    timeout(Duration::from_secs(5), async {
        while backend.admission_snapshot().unwrap().active_queries != 0 {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("ordered query-service cleanup should release admission");
    assert_eq!(backend.admission_snapshot().unwrap().active_queries, 0);

    fs::remove_dir_all(dir).ok();
}

async fn filtered_local_parquet_context(
    path: &std::path::Path,
    device_capacity_bytes: u64,
) -> (AlgeonGpuBackend, SessionContext) {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: device_capacity_bytes });
    let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let cugraph = CugraphSqlConfig::default();
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    let state = backend
        .install_on(SessionStateBuilder::new().with_config(config).with_default_features())
        .unwrap()
        .try_with_cugraph_sql(cugraph)
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("large_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    ctx.sql("CREATE VIEW filtered_edges AS SELECT src, dst FROM large_edges WHERE active")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    (backend, ctx)
}

#[tokio::test]
async fn cugraph_sql_gpu_filtered_local_parquet_hits_and_bfs_use_materialized_edge_rows() {
    const DEVICE_CAPACITY_BYTES: u64 = 128 * 1024 * 1024;

    let dir = unique_temp_dir("algeon-datafusion-cugraph-filtered-local-parquet-phase");
    let path = dir.join("large_filtered_edges.parquet");
    write_large_filtered_directed_edges(&path);
    let (_backend, ctx) = filtered_local_parquet_context(&path, DEVICE_CAPACITY_BYTES).await;

    // One million source rows would exceed this device capacity under the
    // static graph model; only the four rows surviving Filter should price the
    // deferred graph phase.
    let hits_batches = ctx
        .sql("SELECT * FROM cugraph_hits(edges => (SELECT src, dst FROM filtered_edges))")
        .await
        .unwrap()
        .collect()
        .await
        .expect("HITS must execute after runtime filtered-row phase admission");
    assert_eq!(
        hits_batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        FILTERED_GRAPH_ACTIVE_ROWS,
        "HITS must see only the active materialized edge set"
    );

    let bfs_batches = ctx
        .sql("SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM filtered_edges), source_vertex => 0)")
        .await
        .unwrap()
        .collect()
        .await
        .expect("BFS must execute after runtime filtered-row phase admission");
    assert_eq!(
        bfs_batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        FILTERED_GRAPH_ACTIVE_ROWS,
        "BFS must see only the active materialized edge set"
    );

    drop_session_context_on_blocking_thread(ctx).await;
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn cugraph_sql_gpu_overlapping_same_key_bfs_share_one_traversal() {
    const DEVICE_CAPACITY_BYTES: u64 = 256 * 1024 * 1024;
    const VERTICES: i64 = 60_000;
    const BFS_SQL: &str = "SELECT COUNT(*) AS reached, SUM(distance) AS total \
         FROM cugraph_bfs(edges => (SELECT src, dst FROM overlap_edges), source_vertex => 0, output_mode => 'normalized')";

    // Parquet edges (not a MemTable) so the planner attaches graph residency
    // and mints a singleflight key: a directed cycle plus antipodal chords
    // where every vertex is reachable from 0, large enough that two
    // barrier-released queries overlap inside one generation.
    let dir = unique_temp_dir("algeon-datafusion-cugraph-overlap-same-key-bfs");
    let path = dir.join("overlap_edges.parquet");
    write_overlap_cycle_chord_edges(&path, VERTICES);

    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: DEVICE_CAPACITY_BYTES })
        .with_max_active_attempts(2)
        .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(8, 8, 2));
    let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .unwrap()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("overlap_edges", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();

    // Sequential baseline fixes the expected aggregate.
    let baseline = ctx.sql(BFS_SQL).await.unwrap().collect().await.expect("sequential BFS baseline succeeds");

    // Release both queries at the same barrier instant so the second joins
    // the first's singleflight generation instead of leading its own. Each
    // round is an independent race between the follower copy and leader
    // teardown; repeating rounds keeps a single lost race from escaping as a
    // pass, and the first lost race quarantines the device so later rounds
    // fail fast instead of passing independently.
    for round in 0..8 {
        let barrier = Arc::new(Barrier::new(3));
        let leader_barrier = Arc::clone(&barrier);
        let follower_barrier = Arc::clone(&barrier);
        let leader_ctx = ctx.clone();
        let follower_ctx = ctx.clone();
        let leader_task = tokio::spawn(async move {
            leader_barrier.wait().await;
            leader_ctx.sql(BFS_SQL).await.unwrap().collect().await
        });
        let follower_task = tokio::spawn(async move {
            follower_barrier.wait().await;
            follower_ctx.sql(BFS_SQL).await.unwrap().collect().await
        });
        barrier.wait().await;
        let leader_batches = leader_task
            .await
            .expect("leader task joins")
            .unwrap_or_else(|error| panic!("overlapped BFS leader succeeds in round {round}: {error}"));
        let follower_batches = follower_task
            .await
            .expect("follower task joins")
            .unwrap_or_else(|error| panic!("overlapped BFS follower succeeds in round {round}: {error}"));
        assert_eq!(leader_batches, baseline, "overlapped leader matches its sequential baseline");
        assert_eq!(follower_batches, baseline, "singleflight follower matches its sequential baseline");
    }

    // Engagement is asserted, not assumed: every round must coalesce exactly
    // one traversal and one follower copy, otherwise the race above never ran
    // and the parity assertions prove nothing.
    let snapshot = backend.cugraph_bfs_singleflight_snapshot();
    assert_eq!(snapshot.eligible_requests, 17, "baseline plus eight rounds join singleflight");
    assert_eq!(snapshot.physical_traversals, 9, "each round shares one traversal");
    assert_eq!(snapshot.follower_copies, 8, "each round produces one follower copy");
    assert!(snapshot.peak_width >= 2, "barrier-released queries join one generation");

    drop_session_context_on_blocking_thread(ctx).await;
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn cugraph_sql_gpu_filtered_local_parquet_executes_under_the_fixed_cap_and_releases_the_grant() {
    const DEVICE_CAPACITY_BYTES: u64 = 128 * 1024 * 1024;
    const ACTIVE_EDGE_ROWS: usize = 600_000;

    let dir = unique_temp_dir("algeon-datafusion-cugraph-filtered-local-parquet-phase-rejection");
    let path = dir.join("large_filtered_edges.parquet");
    write_large_filtered_directed_edges_with_active_rows(&path, ACTIVE_EDGE_ROWS);
    let (backend, ctx) = filtered_local_parquet_context(&path, DEVICE_CAPACITY_BYTES).await;

    let batches = ctx
        .sql("SELECT * FROM cugraph_hits(edges => (SELECT src, dst FROM filtered_edges))")
        .await
        .unwrap()
        .collect()
        .await
        .expect("modeled graph memory must not reject an executable workload before actual allocation");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1024);

    drop_session_context_on_blocking_thread(ctx).await;
    timeout(Duration::from_secs(5), async {
        loop {
            let reports = backend.device_ledger_reports().unwrap();
            if reports.iter().all(|report| report.active_attempts == 0 && report.grants.is_empty()) {
                break;
            }
            yield_now().await;
        }
    })
    .await
    .expect("graph execution must release the attempt and immutable grant");
    let ledgers = backend.device_ledger_reports().unwrap();
    let [ledger] = ledgers.as_slice() else {
        panic!("one-device fixture must expose exactly one ledger after rejection: {ledgers:#?}");
    };
    assert_eq!(ledger.active_attempts, 0);
    assert_eq!(ledger.live_grants_bytes, 0);
    assert_eq!(ledger.resident_owner_bytes, 0);
    assert!(ledger.grants.is_empty());
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn cugraph_sql_gpu_actual_allocation_exceeding_query_cap_reports_memory_limit() {
    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 1024 }),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .unwrap()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_directed_bfs_edges(&ctx, "raw_edges");

    let error = ctx
        .sql(
            "SELECT vertex, value \
             FROM cugraph_betweenness_centrality(edges => (SELECT src, dst FROM raw_edges), k => 2) \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("actual graph allocation must fail against the one-kibibyte query cap");
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.code(), ErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(algeon.fact_value("native_code"), Some("query_memory_limit_exceeded"));
}

/// Runs the provisioned Python-cuGraph PageRank acceptance check.
///
/// This is an explicit tools command rather than a Cargo test because it
/// requires an external Python/cuGraph installation.
#[cfg_attr(test, expect(dead_code, reason = "explicit tools runner entry point"))]
pub async fn run_pagerank_filter_sort_limit_acceptance() {
    let dir = unique_temp_dir("algeon-datafusion-cugraph-pagerank-compose");
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
    let (_backend, ctx) =
        backend_cugraph_sql_context(SessionStateBuilder::new().with_config(config).with_default_features());
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
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM graph_edges)) \
             WHERE value > 0 \
             ORDER BY value DESC, vertex ASC \
             LIMIT 3",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let plan_display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        plan_display.contains("AlgeonNativeComposedExec"),
        "PageRank post-algorithm filter/sort/limit should be a composed native GPU handoff:\n{plan_display}"
    );
    assert!(
        plan_display.contains("CugraphAlgoExec"),
        "composed native plan should keep cuGraph as the GPU-resident producer child:\n{plan_display}"
    );

    reset_arrow_host_interop_observation_for_test();
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
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
               edges => (SELECT src, dst, weight FROM ppr_edges), \
               personalization => (SELECT vertex, value FROM ppr_seeds)\
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
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
             FROM cugraph_force_atlas2(\
                  edges => (SELECT src, dst, weight FROM layout_edges), max_iter => 5, seed => 42) \
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
async fn cugraph_sql_gpu_force_atlas2_executes_with_vertex_side_inputs() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1, 2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 0, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("layout_edges", Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    for ddl in [
        "CREATE VIEW layout_vertices AS SELECT * FROM (VALUES (CAST(0 AS BIGINT)), (1), (2), (3)) AS v(vertex)",
        "CREATE VIEW layout_attributes AS \
         SELECT vertex, CAST(0.75 AS REAL) AS radius, CAST(1.0 AS REAL) AS mass FROM layout_vertices",
        "CREATE VIEW layout_positions AS \
         SELECT vertex, CAST(vertex AS REAL) AS x, CAST(-vertex AS REAL) AS y FROM layout_vertices",
    ] {
        ctx.sql(ddl).await.unwrap().collect().await.unwrap();
    }

    // Side-input validation runs while the cuGraph handle routes the query stream, so it must
    // use the query runtime rather than cuDF's default execution context.
    let batches = ctx
        .sql(
            "SELECT vertex, x, y \
             FROM cugraph_force_atlas2(\
                  edges => (SELECT src, dst FROM layout_edges), max_iter => 5, seed => 42, \
                  prevent_overlapping => true, \
                  vertex_attributes => (SELECT vertex, radius, mass FROM layout_attributes), \
                  initial_positions => (SELECT vertex, x, y FROM layout_positions)) \
             ORDER BY vertex",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2, 3]);
    let x = batch.column(1).as_any().downcast_ref::<Float32Array>().unwrap();
    let y = batch.column(2).as_any().downcast_ref::<Float32Array>().unwrap();
    for row in 0..batch.num_rows() {
        assert!(x.value(row).is_finite() && y.value(row).is_finite(), "coordinates must be finite at row {row}");
    }
}

#[tokio::test]
async fn cugraph_sql_gpu_string_jaccard_70_by_70_executes_with_bounded_producers() {
    let dir = unique_temp_dir("algeon-datafusion-cugraph-string-jaccard-70-by-70");
    let edge_path = dir.join("edges.parquet");
    let pair_path = dir.join("pairs.parquet");
    let edge_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let pair_schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Utf8, false),
        Field::new("second", DataType::Utf8, false),
    ]));
    let vertices = (0..10).map(|vertex| format!("vertex-{vertex}")).collect::<Vec<_>>();
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from((0..70).map(|row| vertices[row % vertices.len()].as_str()).collect::<Vec<_>>()))
                as ArrayRef,
            Arc::new(StringArray::from(
                (0..70).map(|row| vertices[(row + 1) % vertices.len()].as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![
            Arc::new(StringArray::from((0..70).map(|row| vertices[row % vertices.len()].as_str()).collect::<Vec<_>>()))
                as ArrayRef,
            Arc::new(StringArray::from(
                (0..70).map(|row| vertices[(row + 2) % vertices.len()].as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    for (path, schema, batch) in
        [(&edge_path, Arc::clone(&edge_schema), edge_batch), (&pair_path, Arc::clone(&pair_schema), pair_batch)]
    {
        let mut writer = ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    let (_backend, ctx) = default_backend_cugraph_sql_context();
    ctx.register_parquet("string_similarity_edges", edge_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet("string_candidate_pairs", pair_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    let df = ctx
        .sql(
            "SELECT first, second, similarity \
             FROM cugraph_jaccard(\
               edges => (SELECT src, dst FROM string_similarity_edges), \
               vertex_pairs => (SELECT first, second FROM string_candidate_pairs)\
             ) ORDER BY similarity DESC LIMIT 70",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Native);
    assert!(find_composed_exec(plan.as_ref()).is_some(), "string Jaccard must remain one native composed GPU island");
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let output_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    assert_eq!(output_rows, 70, "explicit pair multiplicity must be preserved without exceeding the 70-row bound");
    let schema = plan.schema();
    assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
    assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
    assert_eq!(schema.field(2).data_type(), &DataType::Float64);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 1);

    ctx.sql(
        "CREATE VIEW string_candidate_pairs_values AS \
         SELECT first, second FROM (VALUES \
           ('vertex-0', 'vertex-2'), \
           ('vertex-0', 'vertex-3'), \
           ('vertex-0', 'vertex-4'), \
           ('vertex-1', 'vertex-3'), \
           ('vertex-1', 'vertex-4'), \
           ('vertex-2', 'vertex-4') \
         ) AS pairs(first, second)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let values_df = ctx
        .sql(
            "SELECT first, second, similarity \
             FROM cugraph_jaccard(\
               edges => (SELECT src, dst FROM string_similarity_edges), \
               vertex_pairs => (SELECT first, second FROM string_candidate_pairs_values)\
             ) ORDER BY similarity DESC",
        )
        .await
        .unwrap();
    let values_plan = values_df.create_physical_plan().await.unwrap();
    assert_eq!(classify_final_plan(values_plan.as_ref()), FinalPlanDisposition::Native);
    assert!(
        find_composed_exec(values_plan.as_ref()).is_some(),
        "VALUES-backed string Jaccard must remain one native composed GPU island",
    );
    let values_batches = collect(Arc::clone(&values_plan), ctx.task_ctx()).await.unwrap();
    let values_batch = single_batch(&values_batches);
    assert_eq!(values_batch.num_rows(), 6);
    assert_eq!(values_batch.schema().field(0).data_type(), &DataType::Utf8);
    assert_eq!(values_batch.schema().field(1).data_type(), &DataType::Utf8);
    assert_eq!(values_batch.schema().field(2).data_type(), &DataType::Float64);
    let similarities = values_batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!((1..similarities.len()).all(|row| similarities.value(row - 1) >= similarities.value(row)));

    drop_session_context_on_blocking_thread(ctx).await;
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn cugraph_sql_gpu_similarity_rejects_actual_null_pair_values() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
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
             edges => (SELECT src, dst FROM similarity_edges), \
             vertex_pairs => (SELECT first, second FROM candidate_pairs))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.code(), ErrorCode::GraphInputInvalid);
    assert_eq!(algeon.fact_value("algorithm"), Some("jaccard"));
    assert_eq!(algeon.fact_value("input_role"), Some("vertex_pairs"));
}

#[tokio::test]
async fn cugraph_sql_gpu_similarity_executes_weighted_jaccard_for_nullable_int64_pairs() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_weighted_similarity_edges_i64(&ctx, "similarity_edges");
    // A nullable Arrow schema is admitted when the materialized pair values are
    // all present. This is distinct from the null-rejection coverage above.
    register_i64_similarity_pairs(&ctx, "candidate_pairs", true, vec![Some(0)], vec![Some(1)]);

    let sql = "SELECT first, second, similarity \
               FROM cugraph_jaccard(\
                 edges => (SELECT src, dst, weight FROM similarity_edges), \
                 vertex_pairs => (SELECT first, second FROM candidate_pairs)\
               )";
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_weighted_similarity_edges_i64(&ctx, "similarity_edges");
    register_i64_similarity_pairs(&ctx, "empty_pairs", false, vec![], vec![]);
    register_i64_similarity_pairs(&ctx, "absent_pairs", false, vec![Some(99)], vec![Some(0)]);

    let empty_sql = "SELECT * FROM cugraph_jaccard(\
                     edges => (SELECT src, dst FROM similarity_edges), \
                     vertex_pairs => (SELECT first, second FROM empty_pairs)\
                   )";
    let empty_df = ctx.sql(empty_sql).await.unwrap();
    let empty_plan = empty_df.create_physical_plan().await.unwrap();
    let empty_batches = collect(Arc::clone(&empty_plan), ctx.task_ctx()).await.unwrap();
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
               edges => (SELECT src, dst FROM similarity_edges), \
               vertex_pairs => (SELECT first, second FROM absent_pairs))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.code(), ErrorCode::CudfInvalidInput);
    assert_eq!(algeon.fact_value("algorithm"), Some("jaccard"));
    assert_eq!(algeon.fact_value("input_role"), Some("vertex_pairs"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_predicate_executes_end_to_end() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_allowed_bfs_edges(&ctx, "allowed_edges");

    let df = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 0, \
                  construction_policy => 'raw_libcugraph', \
                  include_edges => (SELECT allowed_src AS src, allowed_dst AS dst FROM allowed_edges))",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let batch = single_batch(&batches);
    assert_eq!(bfs_distance_for(batch, 0), Some(0));
    assert_eq!(bfs_distance_for(batch, 2), Some(1));
    assert_eq!(bfs_distance_for(batch, 3), Some(2));
    assert_eq!(bfs_distance_for(batch, 1), None);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&plan_metrics_recursive(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 1);
}

#[tokio::test]
async fn cugraph_sql_gpu_camel_case_side_inputs_execute_every_shared_relation_shape() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    let register = |table_name: &str, fields: Vec<Field>, columns: Vec<ArrayRef>| {
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        ctx.register_table(table_name, Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())).unwrap();
    };
    register(
        "camelEdges",
        vec![
            Field::new("edgeSrc", DataType::Int64, false),
            Field::new("edgeDst", DataType::Int64, false),
            Field::new("edgeWeight", DataType::Float64, false),
            Field::new("edgeKey", DataType::Int64, false),
        ],
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 2, 0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64, 1.0, 1.0, 1.0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10_i64, 11, 12, 13])) as ArrayRef,
        ],
    );
    register(
        "camelSources",
        vec![Field::new("sourceVertex", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef],
    );
    register(
        "camelIncluded",
        vec![Field::new("includeVertex", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![0_i64, 1, 2])) as ArrayRef],
    );
    register(
        "camelExcluded",
        vec![Field::new("excludeVertex", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![2_i64])) as ArrayRef],
    );
    register(
        "camelTargets",
        vec![Field::new("targetVertex", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![2_i64])) as ArrayRef],
    );
    register(
        "camelAllowedEdges",
        vec![
            Field::new("edgeSource", DataType::Int64, false),
            Field::new("edgeDestination", DataType::Int64, false),
            Field::new("edgeId", DataType::Int64, false),
        ],
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 2, 0])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10_i64, 11, 12, 13])) as ArrayRef,
        ],
    );
    register(
        "camelSeeds",
        vec![Field::new("seedVertex", DataType::Int64, false), Field::new("seedValue", DataType::Float64, false)],
        vec![
            Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64])) as ArrayRef,
        ],
    );
    register(
        "camelPairs",
        vec![Field::new("pairLeft", DataType::Int64, false), Field::new("pairRight", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef, Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef],
    );

    let cases = [
        (
            "source_vertices",
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst FROM camelEdges), \
             sources => (SELECT \"sourceVertex\" AS vertex FROM camelSources))",
        ),
        (
            "include_vertices",
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst FROM camelEdges), source_vertex => 0, \
             include_vertices => (SELECT \"includeVertex\" AS vertex FROM camelIncluded))",
        ),
        (
            "exclude_vertices",
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst FROM camelEdges), source_vertex => 0, \
             exclude_vertices => (SELECT \"excludeVertex\" AS vertex FROM camelExcluded))",
        ),
        (
            "target_vertices",
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst FROM camelEdges), source_vertex => 0, \
             target_vertices => (SELECT \"targetVertex\" AS vertex FROM camelTargets))",
        ),
        (
            "include_edges",
            "SELECT * FROM cugraph_bfs(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst, \"edgeKey\" AS edge_id FROM camelEdges), source_vertex => 0, \
             include_edges => (SELECT \"edgeSource\" AS src, \"edgeDestination\" AS dst, \"edgeId\" AS edge_id FROM camelAllowedEdges))",
        ),
        (
            "personalization",
            "SELECT * FROM cugraph_personalized_pagerank(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst, \"edgeWeight\" AS weight FROM camelEdges), \
             personalization => (SELECT \"seedVertex\" AS vertex, \"seedValue\" AS value FROM camelSeeds))",
        ),
        (
            "vertex_pairs",
            "SELECT * FROM cugraph_jaccard(\
             edges => (SELECT \"edgeSrc\" AS src, \"edgeDst\" AS dst, \"edgeWeight\" AS weight FROM camelEdges), \
             vertex_pairs => (SELECT \"pairLeft\" AS first, \"pairRight\" AS second FROM camelPairs))",
        ),
    ];
    for (role, sql) in cases {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap_or_else(|error| panic!("{role}: {error:?}"));
        assert!(batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0, "{role} returned no rows");
    }
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_numeric_source_list_executes_multi_source() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_disconnected_bfs_edges(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, reachable \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertices => [0, 10], \
                  construction_policy => 'raw_libcugraph', output_mode => 'normalized')",
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_disconnected_string_bfs_edges(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertices => ['a', 'x'], \
                  construction_policy => 'raw_libcugraph', output_mode => 'normalized')",
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
async fn cugraph_sql_bfs_path_ordering_satisfies_strict_final_plan_without_sort_exec() {
    let dir = unique_temp_dir("algeon-datafusion-cugraph-bfs-path-metadata");
    let metadata_path = dir.join("metadata.parquet");
    let (backend, ctx) = default_backend_cugraph_sql_context();
    let vertices = [
        "bd2f79216e59522ab2ce48a860bcbfdf",
        "9c369bda5e2752df7dda44ad8bffd4209afa2324",
        "0506fd4818344024781bd1ccd573e9cf1f0f9dd8",
        "80010341c28f2c670107a85423d7108b",
        "78e4713e48a0fad2bec00cb8ace0a7481568b54a",
        "1563260218f042627fff19494703c405613aa573",
        "f24db1a10764c275d5cd2e3ab17bca31",
    ];
    let edge_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(StringArray::from(vertices[..6].to_vec())) as ArrayRef,
            Arc::new(StringArray::from(vertices[1..].to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("raw_edges", Arc::new(MemTable::try_new(edge_schema, vec![vec![edge_batch]]).unwrap())).unwrap();
    register_string_vertex_memtable(&ctx, "targets", "vertex_id", &vertices[6..]);
    let metadata_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![
            Arc::new(StringArray::from(vertices.to_vec())) as ArrayRef,
            Arc::new(StringArray::from((0..vertices.len()).map(|index| format!("node-{index}")).collect::<Vec<_>>()))
                as ArrayRef,
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(fs::File::create(&metadata_path).unwrap(), metadata_schema, None).unwrap();
    writer.write(&metadata_batch).unwrap();
    writer.close().unwrap();
    ctx.register_parquet("path_metadata", metadata_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();

    let plan = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 'bd2f79216e59522ab2ce48a860bcbfdf',\
                  construction_policy => 'raw_libcugraph', depth_limit => 6, output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets)) \
             ORDER BY path_index",
        )
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    assert!(find_cugraph_algo_exec(plan.as_ref()).is_some());
    assert!(!plan_contains_named_node(plan.as_ref(), "SortExec"));
    let reject_rule = AlgeonNativePhysicalOptimizerRule::new(
        backend.optimizer_config().clone().with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
    );
    reject_rule
        .optimize(Arc::clone(&plan), &ConfigOptions::default())
        .expect("path_index ordering must satisfy NoDataFusionCpu without a DataFusion sort");
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let batch = single_batch(&batches);
    assert_eq!(int64_column(batch, 0), (0_i64..=6).collect::<Vec<_>>());
    let path = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!((0..batch.num_rows()).map(|row| path.value(row)).collect::<Vec<_>>(), vertices);

    let joined = ctx
        .sql(
            "WITH path AS (\
                 SELECT path_index, source, target, vertex, distance \
                 FROM cugraph_bfs(\
                      edges => (SELECT src, dst FROM raw_edges), source_vertex => 'bd2f79216e59522ab2ce48a860bcbfdf',\
                      construction_policy => 'raw_libcugraph', depth_limit => 6, output_mode => 'path',\
                      target_vertices => (SELECT vertex_id AS vertex FROM targets))\
             ) \
             SELECT path.vertex, metadata.label \
             FROM path LEFT JOIN path_metadata metadata ON path.vertex = metadata.vertex",
        )
        .await
        .unwrap();
    let joined_plan = joined.create_physical_plan().await.unwrap();
    assert_eq!(classify_final_plan(joined_plan.as_ref()), FinalPlanDisposition::Native);
    assert!(find_composed_exec(joined_plan.as_ref()).is_some(), "BFS path join must form one composed GPU island");
    let joined_batches = collect(joined_plan, ctx.task_ctx()).await.unwrap();
    let joined = single_batch(&joined_batches);
    assert_eq!(joined.num_rows(), 7);
    let labels = joined.column(1);
    let mut labels = match labels.data_type() {
        DataType::Utf8 => labels
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap().to_string())
            .collect::<Vec<_>>(),
        DataType::LargeUtf8 => labels
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap().to_string())
            .collect::<Vec<_>>(),
        DataType::Utf8View => labels
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap().to_string())
            .collect::<Vec<_>>(),
        data_type => panic!("unexpected label type: {data_type}"),
    };
    labels.sort_unstable();
    assert_eq!(labels, (0..7).map(|index| format!("node-{index}")).collect::<Vec<_>>());

    drop_session_context_on_blocking_thread(ctx).await;
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_depth_zero_returns_the_source_target_vertex() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![0]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 0,\
                  construction_policy => 'raw_libcugraph', depth_limit => 0, output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batch = single_batch(&batches);
    for column in 0..5 {
        assert_eq!(int64_column(batch, column), vec![0]);
    }
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_source_table_combines_with_target_path() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_disconnected_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0, 10]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets)) \
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![99]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", Vec::new());
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", Vec::new());
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);
    register_i64_vertex_values(&ctx, "included", "vertex_id", vec![123]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets),\
                  include_vertices => (SELECT vertex_id AS vertex FROM included))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(algeon.fact_value("predicate_role"), Some("include_vertices"));
    let native_message =
        algeon.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_absent_source_with_bad_include_vertices_is_rejected() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![99]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);
    register_i64_vertex_values(&ctx, "included", "vertex_id", vec![123]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets),\
                  include_vertices => (SELECT vertex_id AS vertex FROM included))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(algeon.fact_value("predicate_role"), Some("include_vertices"));
    let native_message =
        algeon.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_partial_absent_sources_are_rejected() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0, 99]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.fact_value("reason"), Some("partial_source_vertices_absent"));
    assert_eq!(algeon.fact_value("predicate_role"), Some("source_vertices"));
    let native_message =
        algeon.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_path_absent_target_returns_empty_batch() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![99]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_empty_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);

    let batches = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_empty_directed_bfs_edges(&ctx, "raw_edges");
    register_i64_vertex_values(&ctx, "bfs_sources", "vertex_id", vec![0]);
    register_i64_vertex_values(&ctx, "targets", "vertex_id", vec![3]);
    register_i64_vertex_values(&ctx, "included", "vertex_id", vec![123]);

    let error = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets),\
                  include_vertices => (SELECT vertex_id AS vertex FROM included))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    drop_session_context_on_blocking_thread(ctx).await;
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.fact_value("reason"), Some("predicate_vertices_absent"));
    assert_eq!(algeon.fact_value("predicate_role"), Some("include_vertices"));
    let native_message =
        algeon.fact_value("native_message").expect("native diagnostic must be preserved as a typed fact");
    assert!(!native_message.contains("Found invalid vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_m1_string_algorithms_preserve_numeric_vertex_identity_coverage() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_mixed_string_algorithm_parity_edges(&ctx, "string_edges", "numeric_edges");

    let cases = [
        ("louvain", "SELECT vertex FROM cugraph_louvain(edges => (SELECT src, dst FROM {table}))"),
        ("eigenvector", "SELECT vertex FROM cugraph_eigenvector_centrality(edges => (SELECT src, dst FROM {table}))"),
        ("katz", "SELECT vertex FROM cugraph_katz_centrality(edges => (SELECT src, dst FROM {table}))"),
        ("hits", "SELECT vertex FROM cugraph_hits(edges => (SELECT src, dst FROM {table}))"),
        ("degrees", "SELECT vertex FROM cugraph_degrees_all(edges => (SELECT src, dst FROM {table}))"),
        ("in-degrees", "SELECT vertex FROM cugraph_in_degrees_all(edges => (SELECT src, dst FROM {table}))"),
        ("out-degrees", "SELECT vertex FROM cugraph_out_degrees_all(edges => (SELECT src, dst FROM {table}))"),
        (
            "weak components",
            "SELECT vertex FROM cugraph_weakly_connected_components(edges => (SELECT src, dst FROM {table}))",
        ),
        (
            "strong components",
            "SELECT vertex FROM cugraph_strongly_connected_components(edges => (SELECT src, dst FROM {table}))",
        ),
        ("triangle count", "SELECT vertex FROM cugraph_triangle_count_all(edges => (SELECT src, dst FROM {table}))"),
        ("core number", "SELECT vertex FROM cugraph_core_number(edges => (SELECT src, dst FROM {table}))"),
        ("leiden", "SELECT vertex FROM cugraph_leiden(edges => (SELECT src, dst FROM {table}))"),
        ("ecg", "SELECT vertex FROM cugraph_ecg(edges => (SELECT src, dst FROM {table}))"),
        (
            "force atlas 2",
            "SELECT vertex FROM cugraph_force_atlas2(edges => (SELECT src, dst FROM {table}), max_iter => 5, seed => 42)",
        ),
    ];

    for (label, template) in cases {
        let string_rows = identity_rows(&ctx, &template.replace("{table}", "string_edges"), 1).await;
        let numeric_rows = identity_rows(&ctx, &template.replace("{table}", "numeric_edges"), 1).await;
        assert_eq!(string_rows, numeric_rows, "{label} logical string vertex coverage");
        assert_eq!(string_rows, vec![vec![1], vec![2], vec![3], vec![4]], "{label} vertex set");
    }
    drop_session_context_on_blocking_thread(ctx).await;
}

#[tokio::test]
async fn cugraph_sql_gpu_m2_string_side_inputs_and_edge_subset_match_numeric_identity_rows() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_mixed_string_algorithm_parity_edges(&ctx, "string_edges", "numeric_edges");
    register_string_algorithm_parity_side_inputs(&ctx);

    let string_ppr = "SELECT vertex FROM cugraph_personalized_pagerank(\
        edges => (SELECT src, dst FROM string_edges), \
        personalization => (SELECT vertex, value FROM string_ppr_seeds))";
    let numeric_ppr =
        string_ppr.replace("string_edges", "numeric_edges").replace("string_ppr_seeds", "numeric_ppr_seeds");
    assert_eq!(identity_rows(&ctx, string_ppr, 1).await, identity_rows(&ctx, &numeric_ppr, 1).await);

    for function in ["cugraph_jaccard", "cugraph_sorensen", "cugraph_overlap", "cugraph_cosine"] {
        let string_sql = format!(
            "SELECT first, second FROM {function}(\
             edges => (SELECT src, dst FROM string_edges), \
             vertex_pairs => (SELECT first, second FROM string_candidate_pairs))"
        );
        let numeric_sql = string_sql
            .replace("string_edges", "numeric_edges")
            .replace("string_candidate_pairs", "numeric_candidate_pairs");
        assert_eq!(
            identity_rows(&ctx, &string_sql, 2).await,
            identity_rows(&ctx, &numeric_sql, 2).await,
            "{function} candidate-pair multiplicity and identity"
        );
    }

    let string_k_core = "SELECT src, dst FROM cugraph_k_core(edges => (SELECT src, dst FROM string_edges))";
    let numeric_k_core = "SELECT src, dst FROM cugraph_k_core(edges => (SELECT src, dst FROM numeric_edges))";
    assert_eq!(identity_rows(&ctx, string_k_core, 2).await, identity_rows(&ctx, numeric_k_core, 2).await);
    drop_session_context_on_blocking_thread(ctx).await;
}

#[tokio::test]
async fn cugraph_graph_outputs_feed_an_outer_graph_and_personalization_side_input_on_device() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_mixed_string_algorithm_parity_edges(&ctx, "string_edges", "numeric_edges");
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 0, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table("graph_edges_i32", Arc::new(MemTable::try_new(edge_schema, vec![vec![edge_batch]]).unwrap()))
        .unwrap();
    ctx.sql(
        "CREATE VIEW core_edges AS \
         SELECT src, dst FROM cugraph_k_core(edges => (SELECT src, dst FROM graph_edges_i32), k => 1)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let outer = ctx
        .sql("SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM core_edges)) ORDER BY vertex")
        .await
        .unwrap();
    assert_eq!(outer.schema().field(0).data_type(), &DataType::Int32);
    let outer_plan = outer.create_physical_plan().await.unwrap();
    assert_eq!(
        classify_final_plan(outer_plan.as_ref()),
        FinalPlanDisposition::Native,
        "{}",
        displayable(outer_plan.as_ref()).indent(false)
    );
    let outer_batches = collect(outer_plan, ctx.task_ctx()).await.unwrap();
    assert!(outer_batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);

    ctx.sql(
        "CREATE VIEW top_seeds AS \
         SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM graph_edges_i32)) \
         ORDER BY value DESC, vertex LIMIT 2",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let personalized = ctx
        .sql(
            "SELECT vertex, value FROM cugraph_personalized_pagerank(\
                 edges => (SELECT src, dst FROM graph_edges_i32), \
                 personalization => (SELECT vertex, value FROM top_seeds)) \
             ORDER BY vertex",
        )
        .await
        .unwrap();
    assert_eq!(personalized.schema().field(0).data_type(), &DataType::Int32);
    let personalized_plan = personalized.create_physical_plan().await.unwrap();
    assert_eq!(
        classify_final_plan(personalized_plan.as_ref()),
        FinalPlanDisposition::Native,
        "{}",
        displayable(personalized_plan.as_ref()).indent(false)
    );
    let personalized_batches = collect(personalized_plan, ctx.task_ctx()).await.unwrap();
    assert!(personalized_batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);

    ctx.sql(
        "CREATE VIEW string_top_seeds AS \
         SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) \
         ORDER BY value DESC, vertex LIMIT 2",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let string_personalized = ctx
        .sql(
            "SELECT vertex, value FROM cugraph_personalized_pagerank(\
                 edges => (SELECT src, dst FROM string_edges), \
                 personalization => (SELECT vertex, value FROM string_top_seeds)) \
             ORDER BY value DESC, vertex LIMIT 5",
        )
        .await
        .unwrap();
    assert_eq!(string_personalized.schema().field(0).data_type(), &DataType::Utf8);
    let string_personalized_plan = string_personalized.create_physical_plan().await.unwrap();
    assert_eq!(
        classify_final_plan(string_personalized_plan.as_ref()),
        FinalPlanDisposition::Native,
        "{}",
        displayable(string_personalized_plan.as_ref()).indent(false)
    );
    let string_personalized_batches = collect(string_personalized_plan, ctx.task_ctx()).await.unwrap();
    assert!(string_personalized_batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);
    drop_session_context_on_blocking_thread(ctx).await;
}

#[tokio::test]
async fn graph_results_support_aggregate_provenance_union_and_global_row_number_composition() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_edge_memtable(&ctx, "ordinary_graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    register_phase_c_top_k_edges(&ctx, "row_number_limit_edges", DataType::Float64);
    let queries = [
        (
            "aggregate",
            "SELECT COUNT(value) AS count_value, SUM(value) AS sum_value, \
                    MIN(value) AS min_value, MAX(value) AS max_value \
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM ordinary_graph_edges))",
        ),
        (
            "provenance union",
            "SELECT vertex, 'pagerank' AS provenance \
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM ordinary_graph_edges)) \
             UNION ALL \
             SELECT vertex, 'katz' AS provenance \
             FROM cugraph_katz_centrality(edges => (SELECT src, dst, weight FROM ordinary_graph_edges)) \
             ORDER BY provenance",
        ),
        (
            "global row number",
            "SELECT vertex, value, \
                    ROW_NUMBER() OVER (ORDER BY value DESC, vertex) AS row_number \
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM ordinary_graph_edges)) \
             ORDER BY row_number",
        ),
        (
            "global row number limit",
            "SELECT vertex, value, \
                    ROW_NUMBER() OVER (ORDER BY value DESC, vertex) AS row_number \
             FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM row_number_limit_edges)) \
             ORDER BY row_number \
             LIMIT 3",
        ),
    ];
    for (label, sql) in queries {
        let dataframe = ctx.sql(sql).await.unwrap_or_else(|error| panic!("{label} SQL must plan: {error}"));
        let plan = dataframe.create_physical_plan().await.unwrap();
        assert_eq!(
            classify_final_plan(plan.as_ref()),
            FinalPlanDisposition::Native,
            "{label}\n{}",
            displayable(plan.as_ref()).indent(false),
        );
        let batches =
            collect(plan, ctx.task_ctx()).await.unwrap_or_else(|error| panic!("{label} must execute: {error}"));
        let row_count = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        if label == "global row number limit" {
            assert_eq!(row_count, 3);
        } else {
            assert!(row_count > 0, "{label}");
        }
    }
    drop_session_context_on_blocking_thread(ctx).await;
}

#[tokio::test]
async fn cugraph_sql_gpu_m3_string_selectors_match_numeric_identity_rows() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_mixed_string_algorithm_parity_edges(&ctx, "string_edges", "numeric_edges");

    let selectors = [
        (
            "sssp",
            "SELECT vertex, predecessor FROM cugraph_sssp(edges => (SELECT src, dst, weight FROM string_edges), source_vertex => 'a')",
            "SELECT vertex, predecessor FROM cugraph_sssp(edges => (SELECT src, dst, weight FROM numeric_edges), source_vertex => 1)",
            2,
        ),
        (
            "betweenness",
            "SELECT vertex FROM cugraph_betweenness_centrality(edges => (SELECT src, dst FROM string_edges), seeds => ['a', 'c'])",
            "SELECT vertex FROM cugraph_betweenness_centrality(edges => (SELECT src, dst FROM numeric_edges), seeds => [1, 3])",
            1,
        ),
        (
            "edge betweenness",
            "SELECT source, destination FROM cugraph_edge_betweenness_centrality(edges => (SELECT src, dst FROM string_edges), seeds => ['a', 'c'])",
            "SELECT source, destination FROM cugraph_edge_betweenness_centrality(edges => (SELECT src, dst FROM numeric_edges), seeds => [1, 3])",
            2,
        ),
    ];
    for (label, string_sql, numeric_sql, identity_columns) in selectors {
        let numeric_rows = identity_rows(&ctx, numeric_sql, identity_columns).await;
        let string_rows = identity_rows(&ctx, string_sql, identity_columns).await;
        assert_eq!(string_rows, numeric_rows, "{label} encoded selector and decoded output identity");
    }
    drop_session_context_on_blocking_thread(ctx).await;
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_source_table_maps_logical_ids_end_to_end() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_vertex_memtable(&ctx, "bfs_sources", "vertex_id", &["a"]);

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges),\
                  sources => (SELECT vertex_id AS vertex FROM bfs_sources), output_mode => 'normalized',\
                  construction_policy => 'raw_libcugraph')",
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
async fn cugraph_sql_gpu_pagerank_utf8_is_native_and_matches_dense_numeric_ids() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_string_pagerank_parity_edges(&ctx, "string_edges", "numeric_edges");
    register_string_vertex_memtable(&ctx, "node_universe", "vertex", &["a", "b", "c", "d", "isolated"]);
    let large_limit = HostBatchCursorConfig::DEFAULT_MAX_ROWS + 1;
    register_large_string_pagerank_edges(&ctx, "large_string_edges", "large_numeric_edges", large_limit);
    let string_sql = "SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges))";

    let coverage = validate_query(&ctx, string_sql).await.unwrap();
    assert_eq!(coverage.gpu_path(), "native");
    let plan_cov = coverage.final_plan_coverage().expect("final plan coverage");
    assert_eq!(plan_cov.disposition(), FinalPlanDisposition::Native);
    assert_eq!(plan_cov.datafusion_boundary_count(), 0);

    let string_df = ctx.sql(string_sql).await.unwrap();
    assert_eq!(string_df.schema().field(0).data_type(), &DataType::Utf8);
    let string_plan = string_df.create_physical_plan().await.unwrap();
    assert_eq!(classify_final_plan(string_plan.as_ref()), FinalPlanDisposition::Native);
    let string_batches = collect(Arc::clone(&string_plan), ctx.task_ctx()).await.unwrap();
    let algo = find_cugraph_algo_exec(string_plan.as_ref()).expect("string PageRank runs a cuGraph producer");
    assert_eq!(
        metric_usize(&algo.metrics().expect("cuGraph producer metrics"), "compiled_pipeline_attempts"),
        2,
        "the edge source and the logical-mapping graph plan must each run as one compiled pipeline",
    );

    let numeric_batches = ctx
        .sql("SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM numeric_edges))")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let limited_df = ctx
        .sql("SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) LIMIT 2")
        .await
        .unwrap();
    let limited_plan = limited_df.create_physical_plan().await.unwrap();
    let limited_rows = collect(limited_plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(limited_rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    let top_k_df = ctx
        .sql(
            "SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) \
             ORDER BY value DESC LIMIT 2",
        )
        .await
        .unwrap();
    let top_k_plan = top_k_df.create_physical_plan().await.unwrap();
    let top_k_rows = collect(top_k_plan, ctx.task_ctx()).await.unwrap();
    let top_k_batch = single_batch(&top_k_rows);
    let top_k_values = top_k_batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(top_k_batch.num_rows(), 2);
    assert!(top_k_values.value(0) >= top_k_values.value(1));

    let direct_large_sql = format!(
        "SELECT vertex, value \
         FROM cugraph_pagerank(edges => (SELECT src, dst FROM large_string_edges)) \
         ORDER BY value DESC \
         LIMIT {large_limit}"
    );
    let nested_large_sql = format!(
        "SELECT vertex, value \
         FROM (SELECT vertex, value \
               FROM cugraph_pagerank(edges => (SELECT src, dst FROM large_string_edges)) \
               ORDER BY value DESC \
               LIMIT {large_limit}) AS ranked"
    );
    let outer_large_sql = format!(
        "SELECT vertex, value \
         FROM (SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM large_string_edges))) AS ranked \
         ORDER BY value DESC \
         LIMIT {large_limit}"
    );
    let mut large_values = Vec::new();
    for (shape, sql) in [
        ("direct", direct_large_sql.as_str()),
        ("nested", nested_large_sql.as_str()),
        ("outer", outer_large_sql.as_str()),
    ] {
        let df = ctx.sql(sql).await.unwrap();
        assert_eq!(df.schema().field(0).data_type(), &DataType::Utf8, "{shape} PageRank schema");
        let plan = df.create_physical_plan().await.unwrap();
        assert_eq!(classify_final_plan(plan.as_ref()), FinalPlanDisposition::Native, "{shape} PageRank plan");
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        assert!(batches.len() > 1, "{shape} string PageRank top-k must span multiple host batches");
        let values = pagerank_values(&batches);
        assert_eq!(values.len(), large_limit, "{shape} string PageRank top-k row count");
        assert!(values.iter().all(|value| value.is_finite()), "{shape} PageRank values must be finite");
        assert!(values.windows(2).all(|pair| pair[0] >= pair[1]), "{shape} string PageRank top-k must be descending");
        large_values.push((shape, values));
    }
    let direct_values = &large_values[0].1;
    for (shape, values) in large_values.iter().skip(1) {
        assert!(
            values.iter().zip(direct_values).all(|(actual, expected)| (actual - expected).abs() <= 1.0e-12),
            "{shape} PageRank top-k must agree with the direct shape"
        );
    }

    let numeric_large_sql = format!(
        "SELECT vertex, value \
         FROM cugraph_pagerank(edges => (SELECT src, dst FROM large_numeric_edges)) \
         ORDER BY value DESC \
         LIMIT {large_limit}"
    );
    let numeric_large_df = ctx.sql(&numeric_large_sql).await.unwrap();
    let numeric_large_plan = numeric_large_df.create_physical_plan().await.unwrap();
    let numeric_large_batches = collect(numeric_large_plan, ctx.task_ctx()).await.unwrap();
    assert!(numeric_large_batches.len() > 1, "numeric PageRank top-k must span multiple host batches");
    let numeric_large_values = pagerank_values(&numeric_large_batches);
    assert_eq!(numeric_large_values.len(), large_limit);
    assert!(numeric_large_values.windows(2).all(|pair| pair[0] >= pair[1]));
    drop_session_context_on_blocking_thread(ctx).await;

    let string_batch = single_batch(&string_batches);
    let numeric_batch = single_batch(&numeric_batches);
    let vertices = string_batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let string_values = string_batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    let numeric_vertices = numeric_batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let numeric_values = numeric_batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

    let mut vertex_names = (0..vertices.len()).map(|row| vertices.value(row)).collect::<Vec<_>>();
    vertex_names.sort_unstable();
    assert_eq!(vertex_names, vec!["a", "b", "c", "d"]);
    for row in 0..string_values.len() {
        let expected_id = i64::from(vertices.value(row).as_bytes()[0] - b'a' + 1);
        let numeric_row = (0..numeric_vertices.len())
            .find(|numeric_row| numeric_vertices.value(*numeric_row) == expected_id)
            .expect("dense numeric fixture must contain every string endpoint");
        assert!(
            (string_values.value(row) - numeric_values.value(numeric_row)).abs() <= 1.0e-12,
            "PageRank mismatch for {}: string={} numeric={}",
            vertices.value(row),
            string_values.value(row),
            numeric_values.value(numeric_row),
        );
    }
}

#[tokio::test]
async fn cugraph_string_pagerank_and_bfs_bounded_metadata_joins_execute_natively() {
    let dir = unique_temp_dir("algeon-datafusion-cugraph-string-metadata-enrichment");
    let metadata_path = dir.join("metadata.parquet");
    let unique_metadata_path = dir.join("unique-metadata.parquet");
    let ranked_metadata_path = dir.join("ranked-metadata.parquet");
    let metadata_schema = Arc::new(Schema::new(vec![
        Field::new("vertexId", DataType::Utf8, false),
        Field::new("vertexName", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "c", "d", "isolated"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["A", "A2", "B", "C", "D", "Isolated"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![9, 10, 8, 7, 6, 5])) as ArrayRef,
        ],
    )
    .unwrap();
    let unique_metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "isolated"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["A", "B", "C", "D", "Isolated"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![9, 8, 7, 6, 5])) as ArrayRef,
        ],
    )
    .unwrap();
    let ranked_metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["A", "A2", "B", "C"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![9, 10, 8, 7])) as ArrayRef,
        ],
    )
    .unwrap();
    for (path, batch) in [
        (&metadata_path, metadata_batch),
        (&unique_metadata_path, unique_metadata_batch),
        (&ranked_metadata_path, ranked_metadata_batch),
    ] {
        let mut writer =
            ArrowWriter::try_new(fs::File::create(path).unwrap(), Arc::clone(&metadata_schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    let (backend, ctx) = default_backend_cugraph_sql_context();
    register_string_pagerank_parity_edges(&ctx, "string_edges", "numeric_edges");
    ctx.register_parquet("vertex_metadata", metadata_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    ctx.register_parquet("ranked_metadata", ranked_metadata_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    let metadata_url = ListingTableUrl::parse(unique_metadata_path.to_str().unwrap()).unwrap();
    let metadata_options = ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
    let metadata_config =
        ListingTableConfig::new(metadata_url).with_listing_options(metadata_options).with_schema(metadata_schema);
    let unique_metadata = ListingTable::try_new(metadata_config)
        .unwrap()
        .with_constraints(Constraints::new_unverified(vec![Constraint::Unique(vec![0])]));
    ctx.register_table("vertex_metadata_unique", Arc::new(unique_metadata)).unwrap();

    let queries = [
        (
            "PageRank",
            "SELECT g.vertex, g.value, m.\"vertexName\", m.\"vertexPriority\" \
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) g \
             JOIN vertex_metadata m ON g.vertex = m.\"vertexId\" \
             ORDER BY g.value DESC LIMIT 3",
        ),
        (
            "BFS",
            "SELECT g.vertex, g.distance, m.\"vertexName\", m.\"vertexPriority\" \
             FROM cugraph_bfs(edges => (SELECT src, dst FROM string_edges), source_vertex => 'a') g \
             JOIN vertex_metadata m ON g.vertex = m.\"vertexId\" \
             ORDER BY g.distance, g.vertex LIMIT 10",
        ),
        (
            "PageRank non-unique Left",
            "SELECT g.vertex, g.value, m.\"vertexName\", m.\"vertexPriority\" \
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) g \
             LEFT JOIN vertex_metadata m ON g.vertex = m.\"vertexId\"",
        ),
        (
            "BFS non-unique Left",
            "SELECT g.vertex, g.distance, m.\"vertexName\", m.\"vertexPriority\" \
             FROM cugraph_bfs(edges => (SELECT src, dst FROM string_edges), source_vertex => 'a') g \
             LEFT JOIN vertex_metadata m ON g.vertex = m.\"vertexId\"",
        ),
        (
            "PageRank Left pre-enrichment Top-K",
            "SELECT g.vertex, g.value, m.\"vertexName\", m.\"vertexPriority\" \
             FROM (SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) \
                   ORDER BY value DESC LIMIT 3) g \
             LEFT JOIN vertex_metadata_unique m ON g.vertex = m.\"vertexId\"",
        ),
        (
            "PageRank Left post-enrichment Top-K",
            "SELECT g.vertex, g.value, m.\"vertexName\", m.\"vertexPriority\" \
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) g \
             LEFT JOIN vertex_metadata_unique m ON g.vertex = m.\"vertexId\" \
             ORDER BY g.value DESC LIMIT 3",
        ),
        (
            "BFS Left pre-enrichment Top-K",
            "SELECT g.vertex, g.distance, m.\"vertexName\", m.\"vertexPriority\" \
             FROM (SELECT vertex, distance FROM cugraph_bfs(edges => (SELECT src, dst FROM string_edges), source_vertex => 'a') \
                   ORDER BY distance, vertex LIMIT 3) g \
             LEFT JOIN vertex_metadata_unique m ON g.vertex = m.\"vertexId\"",
        ),
        (
            "BFS Left post-enrichment Top-K",
            "SELECT g.vertex, g.distance, m.\"vertexName\", m.\"vertexPriority\" \
             FROM cugraph_bfs(edges => (SELECT src, dst FROM string_edges), source_vertex => 'a') g \
             LEFT JOIN vertex_metadata_unique m ON g.vertex = m.\"vertexId\" \
             ORDER BY g.distance, g.vertex LIMIT 3",
        ),
    ];
    for (algorithm, sql) in queries {
        let df = ctx.sql(sql).await.unwrap_or_else(|error| panic!("{algorithm} metadata SQL must plan: {error}"));
        let plan = df.create_physical_plan().await.unwrap();
        assert_eq!(
            classify_final_plan(plan.as_ref()),
            FinalPlanDisposition::Native,
            "{algorithm} metadata join\n{}",
            displayable(plan.as_ref()).indent(false)
        );
        let batches = collect(plan, ctx.task_ctx())
            .await
            .unwrap_or_else(|error| panic!("{algorithm} metadata join must execute: {error}"));
        assert!(batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);
    }

    ctx.sql(
        "CREATE VIEW role_pairs AS \
         SELECT first, second FROM (VALUES ('a', 'b'), ('c', 'd')) AS pairs(first, second)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    let filtered_roles_sql = "WITH scores AS (\
             SELECT first, second, similarity \
             FROM cugraph_jaccard(\
               edges => (SELECT src, dst FROM string_edges),\
               vertex_pairs => (SELECT first, second FROM role_pairs)\
             )\
         ), roles AS (\
             SELECT \"vertexId\", \"vertexName\" \
             FROM vertex_metadata \
             WHERE \"vertexId\" IN ('a', 'b', 'c')\
         ) \
         SELECT a.\"vertexName\" AS role_a, b.\"vertexName\" AS role_b, s.similarity \
         FROM scores s \
         LEFT JOIN roles a ON a.\"vertexId\" = s.first \
         LEFT JOIN roles b ON b.\"vertexId\" = s.second \
         ORDER BY s.similarity DESC, s.first, s.second";
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan = attempt.plan_sql(&ctx, filtered_roles_sql).await.unwrap();
    assert_eq!(
        classify_final_plan(plan.as_ref()),
        FinalPlanDisposition::Native,
        "filtered two-alias role enrichment\n{}",
        displayable(plan.as_ref()).indent(false)
    );
    let mut stream = attempt.execute_physical_plan(plan, &ctx).unwrap();
    let mut batches = Vec::new();
    while let Some(batch) = stream.next().await {
        batches.push(batch.expect("filtered two-alias role enrichment must execute"));
    }
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some(), "the retained filtered producers must execute under one admitted fixed cap");
    let batch = single_batch(&batches);
    assert_eq!(batch.num_rows(), 3, "duplicate role metadata must preserve left-join multiplicity");
    assert_eq!(batch.column(0).null_count(), 0);
    assert_eq!(batch.column(1).null_count(), 1, "missing role metadata must preserve the graph result row");

    let ranked_metadata_sql = "WITH ranked AS (\
             SELECT vertex, value \
             FROM cugraph_pagerank(edges => (SELECT src, dst FROM string_edges)) \
             ORDER BY value DESC, vertex \
             LIMIT 5\
         ), metadata AS (\
             SELECT \"vertexId\", \"vertexName\" \
             FROM ranked_metadata \
             WHERE \"vertexId\" IN (SELECT vertex FROM ranked)\
         ) \
         SELECT r.vertex, r.value, a.\"vertexName\" \
         FROM ranked r \
         LEFT JOIN metadata a ON a.\"vertexId\" = r.vertex \
         ORDER BY r.value DESC, r.vertex";
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan = attempt.plan_sql(&ctx, ranked_metadata_sql).await.unwrap();
    assert_eq!(
        classify_final_plan(plan.as_ref()),
        FinalPlanDisposition::Native,
        "shared graph Top-K metadata filter and enrichment\n{}",
        displayable(plan.as_ref()).indent(false)
    );
    find_composed_exec(plan.as_ref()).expect("shared graph Top-K must use one composed execution");
    let mut stream = attempt.execute_physical_plan(plan, &ctx).unwrap();
    let first = stream.next().await.expect("ranked metadata enrichment must produce a batch").unwrap();
    let active = backend.device_ledger_reports().unwrap();
    assert_eq!(active.iter().map(|report| report.grants.len()).sum::<usize>(), 1);
    let mut batches = vec![first];
    while let Some(batch) = stream.next().await {
        batches.push(batch.expect("ranked metadata enrichment must finish"));
    }
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some(), "shared graph Top-K must execute under one admitted fixed cap");
    let batch = single_batch(&batches);
    assert_eq!(batch.num_rows(), 5, "duplicate metadata must preserve LEFT JOIN multiplicity");
    assert_eq!(batch.column(2).null_count(), 1, "missing metadata must preserve the ranked graph row");

    drop_session_context_on_blocking_thread(ctx).await;
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_metadata_join_limit_and_distinct_keep_the_same_empty_and_nonempty_plan_contract() {
    let dir = unique_temp_dir("algeon-datafusion-empty-metadata-join-contract");
    let relationship_path = dir.join("relationships.parquet");
    let empty_metadata_path = dir.join("empty-metadata.parquet");
    let nonempty_metadata_path = dir.join("nonempty-metadata.parquet");
    let relationship_schema = Arc::new(Schema::new(vec![
        Field::new("semantic", DataType::Utf8, false),
        Field::new("targetId", DataType::Utf8, false),
        Field::new("sourceId", DataType::Utf8, false),
    ]));
    let relationship_batch = RecordBatch::try_new(
        Arc::clone(&relationship_schema),
        vec![
            Arc::new(StringArray::from(vec!["depends_on"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["vertex-1"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["subject-1"])) as ArrayRef,
        ],
    )
    .unwrap();
    let metadata_schema = Arc::new(Schema::new(vec![
        Field::new("vertexId", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let empty_metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![Arc::new(StringArray::from(vec!["vertex-2"])) as ArrayRef, Arc::new(Int64Array::from(vec![7])) as ArrayRef],
    )
    .unwrap();
    let nonempty_metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![Arc::new(StringArray::from(vec!["vertex-1"])) as ArrayRef, Arc::new(Int64Array::from(vec![7])) as ArrayRef],
    )
    .unwrap();
    for (path, schema, batch) in [
        (&relationship_path, relationship_schema, relationship_batch),
        (&empty_metadata_path, Arc::clone(&metadata_schema), empty_metadata_batch),
        (&nonempty_metadata_path, metadata_schema, nonempty_metadata_batch),
    ] {
        let mut writer = ArrowWriter::try_new(fs::File::create(path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    let (_backend, ctx) = default_backend_cugraph_sql_context();
    for (table, path) in [
        ("relationships", &relationship_path),
        ("metadata_empty", &empty_metadata_path),
        ("metadata_nonempty", &nonempty_metadata_path),
    ] {
        ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }

    let select = "e.\"semantic\", n.\"vertexId\", n.\"vertexPriority\"";
    for (metadata, expected_rows) in [("metadata_empty", 0), ("metadata_nonempty", 1)] {
        let from = format!(
            "FROM relationships e JOIN {metadata} n ON e.\"targetId\" = n.\"vertexId\" \
             WHERE e.\"sourceId\" = 'subject-1' AND n.\"vertexPriority\" >= 7"
        );
        let queries = [
            format!("SELECT {select} {from} LIMIT 100"),
            format!("SELECT DISTINCT {select} {from} LIMIT 100"),
            format!("SELECT DISTINCT {select} {from} ORDER BY n.\"vertexPriority\" DESC, n.\"vertexId\" LIMIT 100"),
        ];
        for sql in queries {
            let df = ctx.sql(&sql).await.unwrap();
            let plan = df.create_physical_plan().await.unwrap();
            assert_eq!(
                classify_final_plan(plan.as_ref()),
                FinalPlanDisposition::Native,
                "{sql}\n{}",
                displayable(plan.as_ref()).indent(false)
            );
            let batches = collect(plan, ctx.task_ctx()).await.unwrap();
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), expected_rows, "{sql}");
        }
    }

    drop_session_context_on_blocking_thread(ctx).await;
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_include_edges_maps_logical_ids_end_to_end() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");

    let df = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 'a',\
                  construction_policy => 'raw_libcugraph', output_mode => 'normalized',\
                  include_edges => (SELECT allowed_src AS src, allowed_dst AS dst FROM allowed_edges))",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    let batch = single_batch(&batches);
    assert_eq!(string_bfs_distance_for(batch, "a"), Some(0));
    assert_eq!(string_bfs_distance_for(batch, "c"), Some(1));
    assert_eq!(string_bfs_distance_for(batch, "d"), Some(2));
    assert_eq!(string_bfs_distance_for(batch, "b"), None);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_utf8_nullable_edges_drop_actual_null_rows() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_nullable_string_bfs_edges_with_nulls(&ctx, "raw_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor, reachable \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 'a',\
                  construction_policy => 'raw_libcugraph', output_mode => 'normalized')",
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
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_string_bfs_edges(&ctx, "raw_edges");
    register_string_allowed_bfs_edges(&ctx, "allowed_edges");
    register_string_vertex_memtable(&ctx, "targets", "vertex_id", &["d"]);

    let df = ctx
        .sql(
            "SELECT path_index, source, target, vertex, distance \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 'a',\
                  construction_policy => 'raw_libcugraph', output_mode => 'path',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets),\
                  include_edges => (SELECT allowed_src AS src, allowed_dst AS dst FROM allowed_edges)) \
             ORDER BY path_index",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    drop_session_context_on_blocking_thread(ctx).await;
    let batch = single_batch(&batches);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2]);
    let vertices = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!((0..batch.num_rows()).map(|row| vertices.value(row)).collect::<Vec<_>>(), vec!["a", "c", "d"]);
    assert_eq!(int64_column(batch, 4), vec![0, 1, 2]);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_drops_actual_null_predicate_rows() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_nullable_allowed_bfs_edges_with_nulls(&ctx, "allowed_edges");

    let batches = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 0,\
                  construction_policy => 'raw_libcugraph',\
                  include_edges => (SELECT allowed_src AS src, allowed_dst AS dst FROM allowed_edges))",
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
    assert_eq!(bfs_distance_for(batch, 1), None);
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_target_vertices_with_null_values_is_rejected() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_nullable_vertex_memtable_with_nulls(&ctx, "targets", "vertex_id");

    let error = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 0,\
                  construction_policy => 'raw_libcugraph',\
                  target_vertices => (SELECT vertex_id AS vertex FROM targets))",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    // Vertex-set predicate values that are actually null are rejected at
    // execution with the value-level diagnostic code carrying the role/column.
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.fact_value("reason"), Some("predicate_column_contains_nulls"));
    assert_eq!(algeon.fact_value("predicate_role"), Some("target_vertices"));
    assert_eq!(algeon.fact_value("column"), Some("vertex"));
}

#[tokio::test]
async fn cugraph_sql_gpu_bfs_include_edges_empty_after_drop_is_explicit_error() {
    let (_backend, ctx) = default_backend_cugraph_sql_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_all_null_allowed_bfs_edges(&ctx, "allowed_edges");

    let df = ctx
        .sql(
            "SELECT vertex, distance, predecessor \
             FROM cugraph_bfs(\
                  edges => (SELECT src, dst FROM raw_edges), source_vertex => 0,\
                  construction_policy => 'raw_libcugraph',\
                  include_edges => (SELECT allowed_src AS src, allowed_dst AS dst FROM allowed_edges))",
        )
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let error = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap_err();
    let algeon = typed_algeon_error(&error);
    assert_eq!(algeon.fact_value("predicate_role"), Some("include_edges"));
    assert_eq!(algeon.fact_value("input_rows"), Some("2"));
    assert_eq!(algeon.fact_value("null_rows_dropped"), Some("2"));
}
