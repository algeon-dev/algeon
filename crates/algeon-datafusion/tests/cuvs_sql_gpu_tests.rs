#![cfg(feature = "cuvs")]

use std::{fs::File, sync::Arc, time::Duration};

use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    },
    cuvs_sql::CuvsAlgoExec,
    native_exec::AlgeonNativeExecutionConfig,
    planner::AlgeonGpuExecutionMode,
    session::AlgeonSessionStateBuilderExt,
};
#[cfg(feature = "cugraph")]
use algeon_datafusion::{
    cugraph_sql::CugraphSqlConfig,
    native_report::NativeDataFusionExecutionMetrics,
    planner::{FinalPlanDisposition, classify_final_plan},
};
use arrow::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array, ListArray, StringArray},
    buffer::OffsetBuffer,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::testing::arrow::{
    arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
    reset_arrow_host_interop_observation_for_test,
};
#[cfg(feature = "cugraph")]
use datafusion::physical_plan::{execute_stream, metrics::MetricsSet};
use datafusion::{
    datasource::MemTable,
    execution::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect, displayable},
    prelude::{ParquetReadOptions, SessionContext},
};
#[cfg(feature = "cugraph")]
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
#[cfg(feature = "cugraph")]
use tokio::{task::yield_now, time::timeout};

const KNN_OPTIONS: &str = r#"{
  "dataset":{"id":"item_id","vector":{"columns":["d0","d1"]}},
  "queries":{"id":"query_id","vector":{"columns":["d0","d1"]}},
  "k":1,
  "metric":"l2_sqrt_expanded"
}"#;

const KMEANS_OPTIONS: &str = r#"{
  "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
  "n_clusters":2,
  "init":"kmeans++"
}"#;

const PCA_OPTIONS: &str = r#"{
  "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
  "n_components":1
}"#;

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn native_required_accepts_a_raw_cugraph_function_root() {
    let backend = cuvs_test_backend_with_mode(AlgeonGpuExecutionMode::NativeRequired);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install strict backend")
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .try_with_cuvs_sql()
        .expect("install cuVS SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tables(&ctx);

    let graph = ctx
        .sql("SELECT * FROM cugraph_degrees_all('edges', 'src', 'dst', 'weight')")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .expect("strict mode accepts a raw cuGraph function root");
    assert_eq!(classify_final_plan(graph.as_ref()), FinalPlanDisposition::Native);
    assert_eq!(collect(graph, ctx.task_ctx()).await.unwrap().iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
    close_cuvs_test_backend(backend, ctx);
}

#[tokio::test]
async fn functions_only_keeps_native_supported_sql_on_cpu_without_gpu_activity() {
    let backend = cuvs_test_backend_with_mode(AlgeonGpuExecutionMode::FunctionsOnly);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install functions-only backend")
        .try_with_cuvs_sql()
        .expect("install cuVS SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tables(&ctx);

    let plan = ctx
        .sql("SELECT SUM(item_id) AS total FROM dataset_vectors")
        .await
        .expect("plan ordinary SQL")
        .create_physical_plan()
        .await
        .expect("physical ordinary SQL plan");
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    assert!(rendered.contains("AggregateExec"), "ordinary SQL must remain in DataFusion:\n{rendered}");
    assert!(!rendered.contains("AlgeonNative"), "ordinary SQL must not be rewritten in functions-only:\n{rendered}");

    reset_arrow_host_interop_observation_for_test();
    let batches = collect(plan, ctx.task_ctx()).await.expect("ordinary CPU SQL executes");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 0);
    let admission = backend.admission_snapshot().expect("post-query admission snapshot");
    assert_eq!(admission.active_queries, 0);
    let device = backend.backend_metrics_snapshot().device(0).expect("device metrics");
    assert_eq!(device.admission.active_attempts, 0);
    assert_eq!(device.admission.live_grant_bytes, 0);
    close_cuvs_test_backend(backend, ctx);
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn functions_only_executes_cugraph_through_cpu_join_into_cuvs_under_one_admission() {
    let backend = cuvs_test_backend_with_mode(AlgeonGpuExecutionMode::FunctionsOnly);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .try_with_cuvs_sql()
        .expect("install cuGraph and cuVS SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);

    register_tables(&ctx);
    let sql = format!(
        "WITH candidates AS (\
             SELECT e.item_id, e.d0, e.d1 \
             FROM cugraph_force_atlas2(\
                 'edges', 'src', 'dst', 'weight', '{{\"max_iter\":5,\"seed\":42}}'\
             ) AS g \
             JOIN dataset_vectors AS e ON e.item_id = g.vertex \
             WHERE g.x IS NOT NULL\
         ) \
         SELECT * FROM cuvs_brute_force_knn(\
             (SELECT item_id, d0, d1 FROM candidates), \
             (SELECT * FROM query_vectors), \
             '{}')",
        sql_literal(KNN_OPTIONS)
    );

    let plan = ctx
        .sql(&sql)
        .await
        .expect("plan combined cuGraph/cuVS SQL")
        .create_physical_plan()
        .await
        .expect("physical plan");
    find_cuvs_exec(plan.as_ref()).expect("CuvsAlgoExec");
    let rendered_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        rendered_plan.contains("CugraphAlgoExec: algorithm=ForceAtlas2"),
        "the cuVS relation must retain its planned cuGraph producer:\n{rendered_plan}"
    );
    assert!(
        rendered_plan.contains("HashJoinExec"),
        "the SQL join must remain a DataFusion CPU operator:\n{rendered_plan}"
    );
    assert!(
        !rendered_plan.contains("AlgeonNativeComposedExec:"),
        "functions-only must not absorb ordinary relational SQL into a native consumer:\n{rendered_plan}"
    );

    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.expect("GPU -> CPU -> GPU composition must execute");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&recursive_plan_metrics(plan.as_ref()));
    assert!(metrics.admission_sequence.is_some(), "both GPU functions must retain one admission sequence");
    assert_eq!(metrics.graph_runtime_rows.len(), 1, "the graph occurrence must execute exactly once");
    close_cuvs_test_backend(backend, ctx);
}

#[cfg(feature = "cugraph")]
async fn assert_shared_cugraph_cte_lineage_to_two_cuvs_consumers(mode: AlgeonGpuExecutionMode) {
    let backend = cuvs_test_backend_with_mode(mode);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .try_with_cuvs_sql()
        .expect("install cuGraph and cuVS SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tables(&ctx);
    let sql = format!(
        "WITH graph_result AS (\
             SELECT vertex, x, y FROM cugraph_force_atlas2(\
                 'edges', 'src', 'dst', 'weight', '{{\"max_iter\":5,\"seed\":42}}'\
             )\
         ), left_queries AS (\
             SELECT d.item_id AS query_id, d.d0, d.d1 \
             FROM graph_result g JOIN dataset_vectors d ON d.item_id = g.vertex \
             WHERE g.x IS NOT NULL\
         ), right_queries AS (\
             SELECT d.item_id AS query_id, d.d0, d.d1 \
             FROM graph_result g JOIN dataset_vectors d ON d.item_id = g.vertex \
             WHERE g.y IS NOT NULL\
         ), left_knn AS (\
             SELECT query_id, neighbor_id FROM cuvs_brute_force_knn(\
                 (SELECT item_id, d0, d1 FROM dataset_vectors), \
                 (SELECT * FROM left_queries), '{}')\
         ), right_knn AS (\
             SELECT query_id, neighbor_id FROM cuvs_brute_force_knn(\
                 (SELECT item_id, d0, d1 FROM dataset_vectors), \
                 (SELECT * FROM right_queries), '{}')\
         ) \
         SELECT query_id, neighbor_id FROM left_knn \
         UNION ALL \
         SELECT query_id, neighbor_id FROM right_knn",
        sql_literal(KNN_OPTIONS),
        sql_literal(KNN_OPTIONS),
    );
    let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(rendered.matches("CugraphAlgoExec: algorithm=ForceAtlas2").count(), 2, "{rendered}");
    assert_eq!(rendered.matches("CuvsAlgoExec").count(), 2, "{rendered}");

    let batches = collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .expect("both shared-producer branches must satisfy their own downstream occurrence");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&recursive_plan_metrics(plan.as_ref()));
    assert_eq!(metrics.graph_runtime_rows.len(), 2, "each unshared CTE occurrence executes once");
    close_cuvs_test_backend(backend, ctx);
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn functions_only_binds_shared_cugraph_cte_lineage_to_two_cuvs_consumers() {
    assert_shared_cugraph_cte_lineage_to_two_cuvs_consumers(AlgeonGpuExecutionMode::FunctionsOnly).await;
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn native_preferred_binds_nested_host_lineage_inside_composed_consumers() {
    assert_shared_cugraph_cte_lineage_to_two_cuvs_consumers(AlgeonGpuExecutionMode::NativePreferred).await;
}

#[tokio::test]
async fn functions_only_executes_cuvs_through_cpu_limit_into_cuvs() {
    let (backend, ctx) = cuvs_bounded_context_with_mode(AlgeonGpuExecutionMode::FunctionsOnly);
    register_tables(&ctx);
    let sql = format!(
        "WITH first_neighbors AS (\
             SELECT neighbor_id \
             FROM cuvs_brute_force_knn(\
                 (SELECT item_id, d0, d1 FROM dataset_vectors), \
                 (SELECT * FROM query_vectors), \
                 '{}') \
             LIMIT 1\
         ), second_queries AS (\
             SELECT d.item_id AS query_id, d.d0, d.d1 \
             FROM first_neighbors n \
             JOIN dataset_vectors d ON d.item_id = n.neighbor_id\
         ) \
         SELECT * FROM cuvs_brute_force_knn(\
             (SELECT item_id, d0, d1 FROM dataset_vectors), \
             (SELECT query_id, d0, d1 FROM second_queries), \
             '{}')",
        sql_literal(KNN_OPTIONS),
        sql_literal(KNN_OPTIONS),
    );
    let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(
        rendered.matches("CuvsAlgoExec").count(),
        2,
        "both cuVS function occurrences must remain explicit:\n{rendered}"
    );
    assert!(
        rendered.contains("GlobalLimitExec") || rendered.contains("LocalLimitExec"),
        "the intermediate LIMIT must remain a DataFusion CPU boundary:\n{rendered}"
    );

    let batches = collect(plan, ctx.task_ctx())
        .await
        .expect("a CPU LIMIT may stop the first host cursor before EOF and still feed the second cuVS occurrence");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    close_cuvs_test_backend(backend, ctx);
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn functions_only_executes_two_gpu_branches_through_a_cpu_join_under_one_admission() {
    let backend = cuvs_test_backend_with_mode(AlgeonGpuExecutionMode::FunctionsOnly);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .expect("install cuGraph SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tables(&ctx);

    let sql = "WITH left_layout AS (\
                   SELECT vertex, x FROM cugraph_force_atlas2(\
                       'edges', 'src', 'dst', 'weight', '{\"max_iter\":5,\"seed\":42}')\
               ), right_layout AS (\
                   SELECT vertex, y FROM cugraph_force_atlas2(\
                       'edges', 'src', 'dst', 'weight', '{\"max_iter\":5,\"seed\":43}')\
               ) \
               SELECT l.vertex, l.x, r.y \
               FROM left_layout l JOIN right_layout r ON l.vertex = r.vertex \
               ORDER BY l.vertex";
    let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
    let rendered = displayable(plan.as_ref()).indent(false).to_string();
    assert!(rendered.contains("HashJoinExec"), "the branch join must remain on DataFusion CPU:\n{rendered}");

    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.expect("two GPU branches must execute");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
    let metrics = NativeDataFusionExecutionMetrics::from_metrics(&recursive_plan_metrics(plan.as_ref()));
    assert!(metrics.admission_sequence.is_some());
    assert_eq!(metrics.graph_runtime_rows.len(), 2, "each graph occurrence must execute once");
    close_cuvs_test_backend(backend, ctx);
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn functions_only_stream_drop_releases_admission_and_the_plan_can_run_again() {
    let backend = cuvs_test_backend_with_mode(AlgeonGpuExecutionMode::FunctionsOnly);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .try_with_cuvs_sql()
        .expect("install cuVS SQL")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tables(&ctx);
    let sql = format!(
        "SELECT * FROM cuvs_brute_force_knn(\
             (SELECT item_id, d0, d1 FROM dataset_vectors), \
             (SELECT * FROM query_vectors), \
             '{}')",
        sql_literal(KNN_OPTIONS)
    );
    let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();

    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx()).unwrap();
    assert!(stream.next().await.transpose().unwrap().is_some());
    drop(stream);
    timeout(Duration::from_secs(5), async {
        while backend.admission_snapshot().unwrap().active_queries != 0 {
            yield_now().await;
        }
    })
    .await
    .expect("dropping the stream must release query admission");
    assert_eq!(backend.admission_snapshot().unwrap().active_queries, 0);

    let batches =
        collect(plan, ctx.task_ctx()).await.expect("the same physical plan must start a fresh invocation round");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    timeout(Duration::from_secs(5), async {
        while backend.admission_snapshot().unwrap().active_queries != 0 {
            yield_now().await;
        }
    })
    .await
    .expect("completed reuse must release query admission");
    assert_eq!(backend.admission_snapshot().unwrap().active_queries, 0);
    close_cuvs_test_backend(backend, ctx);
}

#[tokio::test]
async fn cuvs_sql_executes_uniform_list_float32_vectors_with_bounded_memory() {
    let (backend, ctx) = cuvs_bounded_context();
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let dataset_schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("embedding", DataType::List(Arc::clone(&item)), false),
    ]));
    let dataset = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![
            Arc::new(Int64Array::from(vec![10_i64, 20])) as ArrayRef,
            Arc::new(uniform_list_f32_array(Arc::clone(&item), &[0.0, 0.0, 3.0, 3.0], 2)) as ArrayRef,
        ],
    )
    .expect("List<Float32> dataset fixture");
    ctx.register_table(
        "list_dataset_vectors",
        Arc::new(MemTable::try_new(dataset_schema, vec![vec![dataset]]).expect("List dataset table")),
    )
    .expect("register List dataset");

    let query_item = Arc::new(Field::new("item", DataType::Float32, false));
    let query_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Int64, false),
        Field::new("embedding", DataType::List(Arc::clone(&query_item)), false),
    ]));
    let queries = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![
            Arc::new(Int64Array::from(vec![99_i64])) as ArrayRef,
            Arc::new(uniform_list_f32_array(query_item, &[0.0, 0.0], 2)) as ArrayRef,
        ],
    )
    .expect("List<Float32> query fixture");
    ctx.register_table(
        "list_query_vectors",
        Arc::new(MemTable::try_new(query_schema, vec![vec![queries]]).expect("List query table")),
    )
    .expect("register List queries");

    let options = r#"{
      "dataset":{"id":"item_id","vector":{"column":"embedding"}},
      "queries":{"id":"query_id","vector":{"column":"embedding"}},
      "k":1,
      "metric":"l2_expanded"
    }"#;
    let sql = format!(
        "SELECT query_id, neighbor_id, rank \
         FROM cuvs_brute_force_knn((SELECT * FROM list_dataset_vectors), (SELECT * FROM list_query_vectors), '{}')",
        sql_literal(options)
    );
    let plan = ctx
        .sql(&sql)
        .await
        .expect("plan List<Float32> cuVS SQL")
        .create_physical_plan()
        .await
        .expect("physical List plan");
    find_cuvs_exec(plan.as_ref()).expect("List<Float32> inputs must plan a CuvsAlgoExec");
    assert_eq!(plan.schema().field(0).name(), "query_id");
    assert_eq!(plan.schema().field(1).name(), "neighbor_id");
    assert_eq!(plan.schema().field(2).name(), "rank");
    assert_cuvs_results(plan, &ctx).await;
    close_cuvs_test_backend(backend, ctx);
}

#[tokio::test]
async fn cuvs_sql_executes_multi_partition_input_and_relational_operators_with_bounded_memory() {
    let (backend, ctx) = cuvs_bounded_context();
    let dataset_schema = wide_vector_schema("item_id");
    let first_partition = wide_vector_batch(Arc::clone(&dataset_schema), "item_id", &[1], &[0.0], &[0.0]);
    let second_partition = wide_vector_batch(Arc::clone(&dataset_schema), "item_id", &[2], &[5.0], &[5.0]);
    ctx.register_table(
        "partitioned_dataset_vectors",
        Arc::new(
            MemTable::try_new(dataset_schema, vec![vec![first_partition], vec![second_partition]])
                .expect("multi-partition dataset table"),
        ),
    )
    .expect("register multi-partition dataset");

    let query_schema = wide_vector_schema("query_id");
    let queries = wide_vector_batch(Arc::clone(&query_schema), "query_id", &[11, 12], &[0.0, 5.0], &[0.0, 5.0]);
    ctx.register_table(
        "partitioned_query_vectors",
        Arc::new(MemTable::try_new(query_schema, vec![vec![queries]]).expect("query table")),
    )
    .expect("register queries");

    let labels_schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let labels = RecordBatch::try_new(
        Arc::clone(&labels_schema),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["nearest-one", "nearest-two"])) as ArrayRef,
        ],
    )
    .expect("labels fixture");
    ctx.register_table("labels", Arc::new(MemTable::try_new(labels_schema, vec![vec![labels]]).expect("labels table")))
        .expect("register labels");

    let plan = plan_sql(
        &ctx,
        &format!(
            "SELECT knn.query_id, labels.label \
             FROM (\
                 SELECT * FROM cuvs_brute_force_knn(\
                     (SELECT * FROM partitioned_dataset_vectors), \
                     (SELECT * FROM partitioned_query_vectors), \
                     '{}')\
                 WHERE rank = 0\
             ) AS knn \
             JOIN labels ON knn.neighbor_id = labels.item_id \
             WHERE knn.neighbor_id = 1 \
             ORDER BY knn.query_id \
             LIMIT 1",
            sql_literal(KNN_OPTIONS)
        ),
    )
    .await;
    find_cuvs_exec(plan.as_ref()).expect("relational composition must retain CuvsAlgoExec");
    assert_eq!(plan.schema().field(0).name(), "query_id");
    assert_eq!(plan.schema().field(1).name(), "label");
    assert_cuvs_results(plan, &ctx).await;
    close_cuvs_test_backend(backend, ctx);
}

#[tokio::test]
async fn cuvs_sql_executes_parquet_relations_with_bounded_memory() {
    let (backend, ctx) = cuvs_bounded_context_with_mode(AlgeonGpuExecutionMode::FunctionsOnly);
    let temp_dir = tempfile::tempdir().expect("temporary Parquet directory");
    let dataset_path = temp_dir.path().join("dataset.parquet");
    let dataset_schema = wide_vector_schema("item_id");
    let dataset = wide_vector_batch(Arc::clone(&dataset_schema), "item_id", &[7, 8], &[1.0, 9.0], &[1.0, 9.0]);
    let file = File::create(&dataset_path).expect("create Parquet fixture");
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&dataset_schema), None).expect("create Parquet writer");
    writer.write(&dataset).expect("write Parquet fixture");
    writer.close().expect("finish Parquet fixture");
    ctx.register_parquet(
        "parquet_dataset_vectors",
        dataset_path.to_str().expect("UTF-8 Parquet path"),
        ParquetReadOptions::default(),
    )
    .await
    .expect("register Parquet dataset");

    let query_schema = wide_vector_schema("query_id");
    let queries = wide_vector_batch(Arc::clone(&query_schema), "query_id", &[77], &[1.0], &[1.0]);
    ctx.register_table(
        "parquet_query_vectors",
        Arc::new(MemTable::try_new(query_schema, vec![vec![queries]]).expect("query table")),
    )
    .expect("register queries");

    let plan = plan_sql(
        &ctx,
        &format!(
            "SELECT query_id, neighbor_id \
             FROM cuvs_brute_force_knn(\
                 (SELECT item_id, d0, d1 FROM parquet_dataset_vectors), \
                 (SELECT * FROM parquet_query_vectors), \
                 '{}')",
            sql_literal(KNN_OPTIONS)
        ),
    )
    .await;
    find_cuvs_exec(plan.as_ref()).expect("Parquet relation must plan a CuvsAlgoExec");
    assert_eq!(plan.schema().field(0).name(), "query_id");
    assert_eq!(plan.schema().field(1).name(), "neighbor_id");
    assert_cuvs_results(plan, &ctx).await;
    close_cuvs_test_backend(backend, ctx);
}

#[tokio::test]
async fn cuvs_sql_executes_fixed_size_list_float32_parquet_relations_with_bounded_memory() {
    let (backend, ctx) = cuvs_bounded_context();
    let temp_dir = tempfile::tempdir().expect("temporary Parquet directory");
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let dataset_schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("embedding", DataType::FixedSizeList(Arc::clone(&item), 2), false),
    ]));
    let dataset = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![
            Arc::new(Int64Array::from(vec![10_i64, 20])) as ArrayRef,
            Arc::new(fixed_size_list_f32_array(Arc::clone(&item), &[0.0, 0.0, 3.0, 3.0], 2)) as ArrayRef,
        ],
    )
    .expect("FixedSizeList<Float32> Parquet dataset fixture");
    let dataset_path = temp_dir.path().join("fixed_dataset.parquet");
    let file = File::create(&dataset_path).expect("create fixed-size dataset Parquet fixture");
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&dataset_schema), None).expect("create fixed-size dataset writer");
    writer.write(&dataset).expect("write fixed-size dataset fixture");
    writer.close().expect("finish fixed-size dataset fixture");
    ctx.register_parquet(
        "fixed_parquet_dataset_vectors",
        dataset_path.to_str().expect("UTF-8 Parquet path"),
        ParquetReadOptions::default(),
    )
    .await
    .expect("register fixed-size dataset Parquet table");

    let query_item = Arc::new(Field::new("item", DataType::Float32, false));
    let query_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Int64, false),
        Field::new("embedding", DataType::FixedSizeList(Arc::clone(&query_item), 2), false),
    ]));
    let query = RecordBatch::try_new(
        Arc::clone(&query_schema),
        vec![
            Arc::new(Int64Array::from(vec![99_i64])) as ArrayRef,
            Arc::new(fixed_size_list_f32_array(query_item, &[0.0, 0.0], 2)) as ArrayRef,
        ],
    )
    .expect("FixedSizeList<Float32> Parquet query fixture");
    let query_path = temp_dir.path().join("fixed_query.parquet");
    let file = File::create(&query_path).expect("create fixed-size query Parquet fixture");
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&query_schema), None).expect("create fixed-size query writer");
    writer.write(&query).expect("write fixed-size query fixture");
    writer.close().expect("finish fixed-size query fixture");
    ctx.register_parquet(
        "fixed_parquet_query_vectors",
        query_path.to_str().expect("UTF-8 Parquet path"),
        ParquetReadOptions::default(),
    )
    .await
    .expect("register fixed-size query Parquet table");

    let options = r#"{
      "dataset":{"id":"item_id","vector":{"column":"embedding"}},
      "queries":{"id":"query_id","vector":{"column":"embedding"}},
      "k":1,
      "metric":"l2_expanded"
    }"#;
    let plan = plan_sql(
        &ctx,
        &format!(
            "SELECT query_id, neighbor_id, rank \
             FROM cuvs_brute_force_knn(\
                 (SELECT * FROM fixed_parquet_dataset_vectors), \
                 (SELECT * FROM fixed_parquet_query_vectors), \
                 '{}')",
            sql_literal(options)
        ),
    )
    .await;
    find_cuvs_exec(plan.as_ref()).expect("FixedSizeList<Float32> Parquet inputs must plan a CuvsAlgoExec");
    assert_eq!(plan.schema().field(0).name(), "query_id");
    assert_eq!(plan.schema().field(1).name(), "neighbor_id");
    assert_eq!(plan.schema().field(2).name(), "rank");
    assert_cuvs_results(plan, &ctx).await;
    close_cuvs_test_backend(backend, ctx);
}

#[tokio::test]
async fn cuvs_kmeans_and_pca_execute_over_parquet_with_bounded_memory() {
    let (backend, ctx) = cuvs_bounded_context();
    let temp_dir = tempfile::tempdir().expect("temporary Parquet directory");
    let input_path = temp_dir.path().join("vector_input.parquet");
    let input_schema = wide_vector_schema("item_id");
    let input = wide_vector_batch(
        Arc::clone(&input_schema),
        "item_id",
        &[1, 2, 3, 4],
        &[0.0, 0.2, 10.0, 10.2],
        &[0.0, 0.1, 10.0, 10.1],
    );
    let file = File::create(&input_path).expect("create Parquet fixture");
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&input_schema), None).expect("create Parquet writer");
    writer.write(&input).expect("write Parquet fixture");
    writer.close().expect("finish Parquet fixture");
    ctx.register_parquet(
        "parquet_vector_input",
        input_path.to_str().expect("UTF-8 Parquet path"),
        ParquetReadOptions::default(),
    )
    .await
    .expect("register Parquet vector input");

    let kmeans = plan_sql(
        &ctx,
        &format!(
            "SELECT id, cluster_id \
             FROM cuvs_kmeans((SELECT item_id, d0, d1 FROM parquet_vector_input), '{}') \
             ORDER BY row_ordinal",
            sql_literal(KMEANS_OPTIONS)
        ),
    )
    .await;
    find_cuvs_exec(kmeans.as_ref()).expect("K-means Parquet input must plan a CuvsAlgoExec");
    assert_eq!(kmeans.schema().field(0).name(), "id");
    assert_eq!(kmeans.schema().field(1).name(), "cluster_id");
    assert_cuvs_results(kmeans, &ctx).await;

    let pca = plan_sql(
        &ctx,
        &format!(
            "SELECT id, pc_0 \
             FROM cuvs_pca((SELECT item_id, d0, d1 FROM parquet_vector_input), '{}') \
             ORDER BY row_ordinal",
            sql_literal(PCA_OPTIONS)
        ),
    )
    .await;
    find_cuvs_exec(pca.as_ref()).expect("PCA Parquet input must plan a CuvsAlgoExec");
    assert_eq!(pca.schema().field(0).name(), "id");
    assert_eq!(pca.schema().field(1).name(), "pc_0");
    assert_cuvs_results(pca, &ctx).await;
    close_cuvs_test_backend(backend, ctx);
}

fn cuvs_test_backend_with_mode(mode: AlgeonGpuExecutionMode) -> AlgeonGpuBackend {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 512 * 1024 * 1024 });
    let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
    AlgeonGpuBackend::builder()
        .with_execution_mode(mode)
        .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            512 * 1024 * 1024,
            256 * 1024 * 1024,
            1,
            3,
            1,
        ))
        .devices([device])
        .build()
        .expect("cuVS SQL tests require a backend-owned GPU")
}

fn cuvs_bounded_context() -> (AlgeonGpuBackend, SessionContext) {
    cuvs_bounded_context_with_mode(AlgeonGpuExecutionMode::NativePreferred)
}

fn cuvs_bounded_context_with_mode(mode: AlgeonGpuExecutionMode) -> (AlgeonGpuBackend, SessionContext) {
    let backend = cuvs_test_backend_with_mode(mode);
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("install backend admission on the cuVS SQL session")
        .try_with_cuvs_sql()
        .expect("install cuVS SQL")
        .build();
    (backend, SessionContext::new_with_state(state))
}

fn close_cuvs_test_backend(backend: AlgeonGpuBackend, ctx: SessionContext) {
    drop(ctx);
    let close = backend.close_and_wait(Duration::from_secs(10)).expect("drain cuVS SQL test backend");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

fn wide_vector_schema(id: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(id, DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]))
}

fn wide_vector_batch(schema: Arc<Schema>, id: &str, ids: &[i64], d0: &[f32], d1: &[f32]) -> RecordBatch {
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(Float32Array::from(d0.to_vec())) as ArrayRef,
            Arc::new(Float32Array::from(d1.to_vec())) as ArrayRef,
        ],
    )
    .unwrap_or_else(|error| panic!("wide vector batch for {id} must be valid: {error}"))
}

fn uniform_list_f32_array(item: Arc<Field>, values: &[f32], dimension: usize) -> ListArray {
    assert!(dimension > 0 && values.len().is_multiple_of(dimension), "uniform List fixture shape must be valid");
    let offsets = (0..=values.len() / dimension)
        .map(|row| i32::try_from(row * dimension).expect("small test List offset"))
        .collect::<Vec<_>>();
    ListArray::new(item, OffsetBuffer::new(offsets.into()), Arc::new(Float32Array::from(values.to_vec())), None)
}

fn fixed_size_list_f32_array(item: Arc<Field>, values: &[f32], dimension: i32) -> FixedSizeListArray {
    assert!(
        dimension > 0 && values.len().is_multiple_of(dimension as usize),
        "fixed-size List fixture shape must be valid"
    );
    FixedSizeListArray::new(item, dimension, Arc::new(Float32Array::from(values.to_vec())), None)
}

async fn plan_sql(ctx: &SessionContext, sql: &str) -> Arc<dyn ExecutionPlan> {
    ctx.sql(sql).await.expect("plan cuVS SQL").create_physical_plan().await.expect("physical cuVS plan")
}

async fn assert_cuvs_results(plan: Arc<dyn ExecutionPlan>, ctx: &SessionContext) {
    let batches = collect(plan, ctx.task_ctx()).await.expect("bounded cuVS execution");
    assert!(batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0, "cuVS execution must return rows");
}

fn register_tables(ctx: &SessionContext) {
    #[cfg(feature = "cugraph")]
    {
        let edges_schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float32, false),
        ]));
        let edges = RecordBatch::try_new(
            Arc::clone(&edges_schema),
            vec![
                Arc::new(Int64Array::from(vec![0_i64, 1, 2, 2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1_i64, 2, 0, 3])) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.0_f32, 1.5, 1.0, 0.5])) as ArrayRef,
            ],
        )
        .expect("edge fixture");
        ctx.register_table("edges", Arc::new(MemTable::try_new(edges_schema, vec![vec![edges]]).expect("edge table")))
            .expect("register edge table");
    }

    let dataset_schema = Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let dataset = RecordBatch::try_new(
        Arc::clone(&dataset_schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 1, 2, 3])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.0_f32, 2.0, 0.0, 3.0])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.0_f32, 0.0, 2.0, 3.0])) as ArrayRef,
        ],
    )
    .expect("dataset fixture");
    ctx.register_table(
        "dataset_vectors",
        Arc::new(MemTable::try_new(dataset_schema, vec![vec![dataset]]).expect("dataset table")),
    )
    .expect("register dataset table");

    let queries_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let queries = RecordBatch::try_new(
        Arc::clone(&queries_schema),
        vec![
            Arc::new(Int64Array::from(vec![99_i64])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.0_f32])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.0_f32])) as ArrayRef,
        ],
    )
    .expect("query fixture");
    ctx.register_table(
        "query_vectors",
        Arc::new(MemTable::try_new(queries_schema, vec![vec![queries]]).expect("query table")),
    )
    .expect("register query table");
}

fn find_cuvs_exec(plan: &dyn ExecutionPlan) -> Option<&CuvsAlgoExec> {
    if let Some(exec) = plan.downcast_ref::<CuvsAlgoExec>() {
        return Some(exec);
    }
    plan.children().into_iter().find_map(|child| find_cuvs_exec(child.as_ref()))
}

#[cfg(feature = "cugraph")]
fn recursive_plan_metrics(plan: &dyn ExecutionPlan) -> MetricsSet {
    fn append(metrics: &mut MetricsSet, plan: &dyn ExecutionPlan) {
        if let Some(plan_metrics) = plan.metrics() {
            for metric in plan_metrics.iter() {
                metrics.push(Arc::clone(metric));
            }
        }
        for child in plan.children() {
            append(metrics, child.as_ref());
        }
    }

    let mut metrics = MetricsSet::new();
    append(&mut metrics, plan);
    metrics
}

fn sql_literal(json: &str) -> String {
    json.replace('\'', "''")
}
