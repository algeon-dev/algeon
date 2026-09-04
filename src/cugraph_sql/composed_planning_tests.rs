use std::{fs, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use datafusion::{
    catalog::MemTable,
    execution::SessionStateBuilder,
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionContext},
};
use nexus_query_engine::{
    expr::Expr,
    plan::{PlanNode, QueryPlanBuilder},
    source::{GpuResidentSource, LiteralBatchSource, SourceNode},
};
use parquet::arrow::ArrowWriter;

use super::{CugraphAlgoExec, CugraphSqlConfig};
use crate::{
    backend::{GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceConfig, NexusGpuDeviceProfile},
    native_exec::{NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig, NexusNativeGpuProducerExec},
    session::NexusSessionStateBuilderExt,
};

const CPU_INPUT_BFS_SQL: &str = "SELECT vertex, distance, predecessor \
     FROM cugraph_bfs('raw_edges', 0, 'src', 'dst', NULL, \
          '{\"construction_policy\":\"raw_libcugraph\"}') \
     ORDER BY vertex";

const RAW_BFS_TOP_K_SQL: &str = "SELECT vertex, distance, predecessor \
     FROM cugraph_bfs('phase_c_top_k_edges', 0, 'src', 'dst', NULL, \
          '{\"construction_policy\":\"raw_libcugraph\"}') \
     ORDER BY distance \
     LIMIT 100";

const PAGERANK_TOP_K_SQL: &str = "SELECT vertex, value \
     FROM cugraph_pagerank('phase_c_top_k_edges', 'src', 'dst', 'weight') \
     ORDER BY value DESC \
     LIMIT 100";

const PAGERANK_FLOAT32_TOP_K_SQL: &str = "SELECT vertex, value \
     FROM cugraph_pagerank('phase_c_top_k_edges_float32', 'src', 'dst', 'weight') \
     ORDER BY value DESC \
     LIMIT 100";

fn backend_context() -> (NexusGpuBackend, SessionContext) {
    let device = NexusGpuDeviceConfig::new(NexusGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = NexusGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .unwrap()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap()
        .build();
    (backend, SessionContext::new_with_state(state))
}

fn register_edge_memtable(
    ctx: &SessionContext,
    table_name: &str,
    src_type: DataType,
    dst_type: DataType,
    weight_type: DataType,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", src_type.clone(), false),
        Field::new("dst", dst_type.clone(), false),
        Field::new("weight", weight_type.clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![edge_column(&src_type), edge_column(&dst_type), edge_column(&weight_type)],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn edge_column(data_type: &DataType) -> ArrayRef {
    match data_type {
        DataType::Int64 => Arc::new(Int64Array::from(vec![0_i64, 1])) as ArrayRef,
        DataType::Float32 => Arc::new(Float32Array::from(vec![1.0_f32, 2.0])) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(vec![1.0_f64, 2.0])) as ArrayRef,
        other => panic!("test helper does not support {other:?}"),
    }
}

fn register_directed_bfs_edges(ctx: &SessionContext, table_name: &str) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0_i64, 0, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    ctx.register_table(table_name, provider).unwrap();
}

fn register_phase_c_top_k_edges(ctx: &SessionContext, table_name: &str, weight_type: DataType) {
    let sources = (0_i64..=256).collect::<Vec<_>>();
    let mut destinations = (1_i64..=256).collect::<Vec<_>>();
    destinations.push(0);
    let weights = match &weight_type {
        DataType::Float32 => Arc::new(Float32Array::from(vec![1.0_f32; sources.len()])) as ArrayRef,
        DataType::Float64 => Arc::new(Float64Array::from(vec![1.0_f64; sources.len()])) as ArrayRef,
        other => panic!("phase C PageRank fixture does not support {other:?}"),
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

fn find_composed_exec(plan: &dyn ExecutionPlan) -> Option<&NexusNativeComposedExec> {
    if let Some(composed) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(composed);
    }
    plan.children().into_iter().find_map(|child| find_composed_exec(child.as_ref()))
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()))
}

fn register_string_pagerank_parity_edges(ctx: &SessionContext, string_table: &str, numeric_table: &str) {
    let string_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, true), Field::new("dst", DataType::Utf8, true)]));
    let string_batch = RecordBatch::try_new(
        Arc::clone(&string_schema),
        vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("a"), Some("a"), Some("b"), Some("c"), None, Some("d")]))
                as ArrayRef,
            Arc::new(StringArray::from(vec![Some("b"), Some("b"), Some("c"), Some("d"), Some("d"), Some("a"), None]))
                as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(string_table, Arc::new(MemTable::try_new(string_schema, vec![vec![string_batch]]).unwrap()))
        .unwrap();

    let numeric_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, true), Field::new("dst", DataType::Int64, true)]));
    let numeric_batch = RecordBatch::try_new(
        Arc::clone(&numeric_schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(1), Some(2), Some(3), None, Some(4)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2), Some(2), Some(3), Some(4), Some(4), Some(1), None])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(numeric_table, Arc::new(MemTable::try_new(numeric_schema, vec![vec![numeric_batch]]).unwrap()))
        .unwrap();
}

fn register_large_string_pagerank_edges(
    ctx: &SessionContext,
    string_table: &str,
    numeric_table: &str,
    vertex_count: usize,
) {
    assert!(vertex_count > 1);
    let edge_count = vertex_count + (vertex_count.saturating_sub(1) / 2);
    let mut string_sources = Vec::with_capacity(edge_count);
    let mut string_destinations = Vec::with_capacity(edge_count);
    let mut numeric_sources = Vec::with_capacity(edge_count);
    let mut numeric_destinations = Vec::with_capacity(edge_count);
    for source in 0..vertex_count {
        let source_id = i64::try_from(source + 1).unwrap();
        let next_id = i64::try_from((source + 1) % vertex_count + 1).unwrap();
        string_sources.push(format!("vertex-{source:05}"));
        string_destinations.push(format!("vertex-{next:05}", next = (source + 1) % vertex_count));
        numeric_sources.push(source_id);
        numeric_destinations.push(next_id);
        if source > 0 && source % 2 == 0 {
            string_sources.push(format!("vertex-{source:05}"));
            string_destinations.push("vertex-00000".to_string());
            numeric_sources.push(source_id);
            numeric_destinations.push(1);
        }
    }

    let string_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Utf8, false), Field::new("dst", DataType::Utf8, false)]));
    let string_batch = RecordBatch::try_new(
        Arc::clone(&string_schema),
        vec![
            Arc::new(StringArray::from(string_sources)) as ArrayRef,
            Arc::new(StringArray::from(string_destinations)) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(string_table, Arc::new(MemTable::try_new(string_schema, vec![vec![string_batch]]).unwrap()))
        .unwrap();

    let numeric_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let numeric_batch = RecordBatch::try_new(
        Arc::clone(&numeric_schema),
        vec![
            Arc::new(Int64Array::from(numeric_sources)) as ArrayRef,
            Arc::new(Int64Array::from(numeric_destinations)) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(numeric_table, Arc::new(MemTable::try_new(numeric_schema, vec![vec![numeric_batch]]).unwrap()))
        .unwrap();
}

fn write_tiny_directed_edges(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 0, 1, 2, 3, 0, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 3, 4, 1, 5])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, true, true, true, true, true, false])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn schema_field_names(schema: &arrow::datatypes::Schema) -> Vec<&str> {
    schema.fields().iter().map(|field| field.name().as_str()).collect()
}

fn producer_plan(schema: &arrow::datatypes::SchemaRef) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(schema))));
    builder.finish_gpu_resident(source)
}

#[tokio::test]
async fn phase0_pagerank_forceatlas2_join_lowers_to_one_composed_consumer() {
    let (_backend, ctx) = backend_context();
    register_edge_memtable(&ctx, "graph_edges", DataType::Int64, DataType::Int64, DataType::Float64);
    let sql = "WITH pagerank AS (\n\
        SELECT vertex, value AS pagerank_value\n\
        FROM cugraph_pagerank('graph_edges', 'src', 'dst', 'weight')\n\
    ), layout AS (\n\
        SELECT vertex, x, y\n\
        FROM cugraph_force_atlas2('graph_edges', 'src', 'dst', 'weight')\n\
    )\n\
    SELECT p.vertex, p.pagerank_value, l.x, l.y\n\
    FROM layout l JOIN pagerank p ON p.vertex = l.vertex\n\
    WHERE p.pagerank_value > 0";
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    let composed = plan
        .downcast_ref::<NexusNativeComposedExec>()
        .expect("the canonical graph topology must lower at the root to one composed native consumer");
    let producers = composed.children();
    assert_eq!(producers.len(), 2, "the composed boundary must retain both graph producer occurrences");
    assert!(producers.iter().all(|producer| producer.downcast_ref::<CugraphAlgoExec>().is_some()));
    assert_eq!(
        producers
            .iter()
            .map(|producer| producer.schema().fields().iter().map(|field| field.name().to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![
            vec!["vertex".to_string(), "value".to_string()],
            vec!["vertex".to_string(), "x".to_string(), "y".to_string()]
        ],
        "producer order must be PageRank followed by ForceAtlas2"
    );

    let graph_edge_inputs = producers
        .iter()
        .map(|producer| {
            assert_eq!(producer.children().len(), 1, "each graph producer must have one shared edge relation input");
            let edge_input = producer.children()[0]
                .downcast_ref::<NexusNativeExec>()
                .expect("graph edge input must be lowered through the native source boundary");
            assert_eq!(
                edge_input.schema().fields().iter().map(|field| field.name().to_string()).collect::<Vec<_>>(),
                vec!["src".to_string(), "dst".to_string(), "weight".to_string()]
            );
            edge_input.native_plan()
        })
        .collect::<Vec<_>>();
    assert_eq!(graph_edge_inputs[0], graph_edge_inputs[1], "both graph producers must use the same edge-source plan");

    let native_consumer = composed.native_plan();
    assert!(native_consumer.sink().materializes_host_boundary());
    assert_eq!(
        native_consumer
            .nodes()
            .iter()
            .filter(|node| matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_))))
            .count(),
        2,
        "the native consumer must declare one GPU input for each graph producer"
    );
    assert!(native_consumer.contains_join(), "the native consumer must retain the vertex join");
    assert_eq!(
        native_consumer.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Filter(_))).count(),
        1,
        "the native consumer must retain the relational PageRank filter"
    );
    assert_eq!(
        native_consumer.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Projection(_))).count(),
        1,
        "the native consumer must retain the relational projection"
    );
    assert!(matches!(&native_consumer.root_node().kind, PlanNode::Join(_)));
}

#[test]
fn phase0_synthetic_n3_composed_fixture_preserves_equal_producer_occurrences() {
    let schema =
        Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new("vertex", DataType::Int64, false)]));
    let producers = ["producer-0", "producer-1", "producer-2"]
        .into_iter()
        .map(|name| {
            Arc::new(
                NexusNativeGpuProducerExec::try_new(name, producer_plan(&schema))
                    .expect("synthetic GPU producer must satisfy the production constructor contract"),
            ) as Arc<dyn ExecutionPlan>
        })
        .collect::<Vec<_>>();

    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_inputs = ["producer-0", "producer-1", "producer-2"]
        .into_iter()
        .map(|name| {
            consumer_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(name, Arc::clone(&schema))))
        })
        .collect::<Vec<_>>();
    let union = consumer_builder.add_union(consumer_inputs, Arc::clone(&schema));
    let consumer = consumer_builder.finish_with_host_sink(union);
    let composed = NexusNativeComposedExec::try_new(consumer, producers, NexusNativeExecutionConfig::bounded_default())
        .expect("synthetic N=3 topology must reach the composed production boundary");

    let children = composed.children();
    assert_eq!(children.len(), 3, "equal native producer shapes must not be structurally deduplicated");
    assert_eq!(
        children
            .iter()
            .map(|child| child
                .downcast_ref::<NexusNativeGpuProducerExec>()
                .expect("synthetic children must remain native GPU producers")
                .input_name())
            .collect::<Vec<_>>(),
        vec!["producer-0", "producer-1", "producer-2"]
    );
    let native_producer_plans = children
        .iter()
        .map(|child| child.downcast_ref::<NexusNativeGpuProducerExec>().unwrap().native_plan())
        .collect::<Vec<_>>();
    assert_eq!(native_producer_plans[0], native_producer_plans[1]);
    assert_eq!(native_producer_plans[1], native_producer_plans[2]);

    let native_consumer = composed.native_plan();
    assert_eq!(
        native_consumer
            .nodes()
            .iter()
            .filter(|node| matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_))))
            .count(),
        3,
        "the native consumer must retain all three runtime input occurrences"
    );
    assert_eq!(native_consumer.nodes().iter().filter(|node| matches!(&node.kind, PlanNode::Union(_))).count(), 1);
}

#[tokio::test]
async fn cpu_input_bfs_orders_by_vertex_through_composed_native_plan() {
    let (_backend, ctx) = backend_context();
    register_directed_bfs_edges(&ctx, "raw_edges");
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges", DataType::Float64);

    let df = ctx.sql(CPU_INPUT_BFS_SQL).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let composed = find_composed_exec(plan.as_ref()).expect("backend SQL should compose the downstream native plan");
    let native_plan = composed.native_plan();
    assert!(native_plan.unreachable_node_ids().is_empty(), "the composed native plan must have no detached nodes");
    assert_eq!(native_plan.nodes().len(), 2, "BFS ORDER BY should lower to one source and one sort");
    let [source_node, sort_node] = native_plan.nodes() else {
        panic!("BFS ORDER BY should lower to exactly GpuResidentSource -> Sort");
    };
    let PlanNode::Source(source) = &source_node.kind else {
        panic!("BFS ORDER BY native input must be a source node");
    };
    let SourceNode::GpuResident(source) = source.as_ref() else {
        panic!("BFS ORDER BY native input must remain GPU-resident");
    };
    assert!(!source.input_name.is_empty(), "the graph result source must retain its runtime identity");
    let PlanNode::Sort(sort) = &sort_node.kind else {
        panic!("BFS ORDER BY native consumer must be a Sort node");
    };
    assert_eq!(sort.input, source_node.id, "the Sort must consume the graph-result source directly");
    assert_eq!(sort.spec.expressions.len(), 1, "BFS ORDER BY must use one supported sort key");
    assert_eq!(sort.spec.expressions[0].expr, Expr::column("vertex"));
    assert_eq!(native_plan.root_id(), sort_node.id, "the native sort must be the composed consumer root");
}

#[tokio::test]
async fn raw_bfs_top_k_consumer_keeps_sort_limit_structure() {
    let (_backend, ctx) = backend_context();
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges", DataType::Float64);

    let top_k_plan = ctx.sql(RAW_BFS_TOP_K_SQL).await.unwrap().create_physical_plan().await.unwrap();
    assert_eq!(schema_field_names(top_k_plan.schema().as_ref()), vec!["vertex", "distance", "predecessor"]);
    assert!(top_k_plan.schema().field_with_name("predecessor").unwrap().is_nullable());
    let top_k_composed =
        find_composed_exec(top_k_plan.as_ref()).expect("raw BFS top-k must lower to one composed native consumer");
    let top_k_native_plan = top_k_composed.native_plan().to_string();
    for expected in ["GpuResidentSource", "Sort", "Limit"] {
        assert!(
            top_k_native_plan.contains(expected),
            "raw BFS top-k consumer must include {expected}:\n{top_k_native_plan}"
        );
    }
}

#[tokio::test]
async fn pagerank_top_k_consumer_keeps_source_sort_limit_structure() {
    let (_backend, ctx) = backend_context();
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges", DataType::Float64);
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges_float32", DataType::Float32);

    for (weight_type, sql) in [(DataType::Float32, PAGERANK_FLOAT32_TOP_K_SQL), (DataType::Float64, PAGERANK_TOP_K_SQL)]
    {
        let top_k_plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
        assert_eq!(schema_field_names(top_k_plan.schema().as_ref()), vec!["vertex", "value"]);
        assert_eq!(top_k_plan.schema().field_with_name("value").unwrap().data_type(), &DataType::Float64);
        let top_k_composed =
            find_composed_exec(top_k_plan.as_ref()).expect("PageRank top-k must lower to one composed native consumer");
        let top_k_native_plan = top_k_composed.native_plan();
        assert!(top_k_native_plan.unreachable_node_ids().is_empty());
        assert_eq!(
            top_k_native_plan.nodes().len(),
            3,
            "{weight_type:?} PageRank top-k must lower to GpuResidentSource -> Sort -> Limit"
        );
        let [source_node, sort_node, limit_node] = top_k_native_plan.nodes() else {
            panic!("{weight_type:?} PageRank top-k must lower to GpuResidentSource -> Sort -> Limit");
        };
        assert!(
            matches!(&source_node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_)))
        );
        let PlanNode::Sort(sort) = &sort_node.kind else {
            panic!("{weight_type:?} PageRank top-k consumer must sort its GPU-resident input");
        };
        assert_eq!(sort.input, source_node.id);
        let PlanNode::Limit(limit) = &limit_node.kind else {
            panic!("{weight_type:?} PageRank top-k consumer must apply a limit after sorting");
        };
        assert_eq!(limit.input, sort_node.id);
        assert_eq!(limit.spec.skip, 0);
        assert_eq!(limit.spec.fetch, Some(100));
        assert_eq!(top_k_native_plan.root_id(), limit_node.id);
    }
}

#[tokio::test]
async fn bare_limit_plans_keep_source_limit_structure_per_algorithm() {
    let (_backend, ctx) = backend_context();
    register_phase_c_top_k_edges(&ctx, "phase_c_top_k_edges", DataType::Float64);

    for (algorithm, sql) in [
        (
            "PageRank",
            "SELECT vertex, value FROM cugraph_pagerank(\
             'phase_c_top_k_edges', 'src', 'dst', 'weight') LIMIT 3",
        ),
        (
            "Eigenvector",
            "SELECT vertex, value FROM cugraph_eigenvector_centrality(\
             'phase_c_top_k_edges', 'src', 'dst', 'weight') LIMIT 3",
        ),
    ] {
        let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
        let composed = find_composed_exec(plan.as_ref())
            .unwrap_or_else(|| panic!("{algorithm} bare limit must lower to one composed native consumer"));
        let [source, limit] = composed.native_plan().nodes() else {
            panic!("{algorithm} bare limit must lower to GpuResidentSource -> Limit");
        };
        assert!(
            matches!(&source.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.row_count_upper_bound == Some(3)))
        );
        assert!(
            matches!(&limit.kind, PlanNode::Limit(limit) if limit.input == source.id && limit.spec.fetch == Some(3))
        );
        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap_or_else(|error| panic!("{algorithm} bare limit must execute: {error}"));
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }
}

#[tokio::test]
async fn string_vertex_pagerank_limits_stay_single_source_consumers() {
    let (_backend, ctx) = backend_context();
    register_string_pagerank_parity_edges(&ctx, "string_edges", "numeric_edges");

    let limited_df =
        ctx.sql("SELECT vertex, value FROM cugraph_pagerank('string_edges', 'src', 'dst') LIMIT 2").await.unwrap();
    let limited_plan = limited_df.create_physical_plan().await.unwrap();
    let limited_consumer =
        find_composed_exec(limited_plan.as_ref()).expect("pushed string PageRank limit stays native");
    assert_eq!(limited_consumer.native_plan().nodes().len(), 1);
    assert!(matches!(limited_consumer.native_plan().root_node().kind, PlanNode::Source(_)));

    let top_k_df = ctx
        .sql(
            "SELECT vertex, value FROM cugraph_pagerank('string_edges', 'src', 'dst') \
             ORDER BY value DESC LIMIT 2",
        )
        .await
        .unwrap();
    let top_k_plan = top_k_df.create_physical_plan().await.unwrap();
    let top_k_consumer = find_composed_exec(top_k_plan.as_ref()).expect("string PageRank top-k stays native");
    assert_eq!(top_k_consumer.native_plan().nodes().len(), 1);
    assert!(matches!(top_k_consumer.native_plan().root_node().kind, PlanNode::Source(_)));
}

#[tokio::test]
async fn large_numeric_pagerank_top_k_keeps_source_sort_limit_structure() {
    let large_limit = cudf::frame::dataframe::HostBatchCursorConfig::DEFAULT_MAX_ROWS + 1;
    let (_backend, ctx) = backend_context();
    register_large_string_pagerank_edges(&ctx, "large_string_edges", "large_numeric_edges", large_limit);

    let numeric_large_sql = format!(
        "SELECT vertex, value \
         FROM cugraph_pagerank('large_numeric_edges', 'src', 'dst') \
         ORDER BY value DESC \
         LIMIT {large_limit}"
    );
    let numeric_large_df = ctx.sql(&numeric_large_sql).await.unwrap();
    let numeric_large_plan = numeric_large_df.create_physical_plan().await.unwrap();
    let numeric_consumer =
        find_composed_exec(numeric_large_plan.as_ref()).expect("numeric PageRank top-k must remain composed");
    let numeric_native_plan = numeric_consumer.native_plan().to_string();
    for expected in ["GpuResidentSource", "Sort", "Limit"] {
        assert!(
            numeric_native_plan.contains(expected),
            "numeric PageRank top-k consumer must retain {expected} in Source -> Sort -> Limit path:\n{numeric_native_plan}"
        );
    }
}

#[tokio::test]
async fn ranked_metadata_enrichment_shares_one_top_k_occurrence() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-ranked-structural");
    std::fs::create_dir_all(&dir).unwrap();
    let ranked_metadata_path = dir.join("ranked-metadata.parquet");
    let metadata_schema = Arc::new(Schema::new(vec![
        Field::new("vertexId", DataType::Utf8, false),
        Field::new("vertexName", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let ranked_metadata_batch = RecordBatch::try_new(
        Arc::clone(&metadata_schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "c"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["A", "A2", "B", "C"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![9, 10, 8, 7])) as ArrayRef,
        ],
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(fs::File::create(&ranked_metadata_path).unwrap(), Arc::clone(&metadata_schema), None)
            .unwrap();
    writer.write(&ranked_metadata_batch).unwrap();
    writer.close().unwrap();

    let (_backend, ctx) = backend_context();
    register_string_pagerank_parity_edges(&ctx, "string_edges", "numeric_edges");
    ctx.register_parquet("ranked_metadata", ranked_metadata_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();

    let ranked_metadata_sql = "WITH ranked AS (\
             SELECT vertex, value \
             FROM cugraph_pagerank('string_edges', 'src', 'dst') \
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
    let plan = ctx.sql(ranked_metadata_sql).await.unwrap().create_physical_plan().await.unwrap();
    let composed = find_composed_exec(plan.as_ref()).expect("shared graph Top-K must use one composed execution");
    let native_plan = composed.native_plan();
    let graph_sources = native_plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some((node.id, source.input_name.clone())),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(graph_sources.len(), 1, "ranked must have one authoritative graph result occurrence");
    let shared_limits = native_plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Limit(limit) if limit.spec.fetch == Some(5) => Some(node.id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(shared_limits.len(), 1, "ranked must have one authoritative Top-K occurrence");
    let shared_limit = shared_limits[0];
    assert_eq!(
        native_plan.nodes().iter().filter(|node| node.input_ids().contains(&shared_limit)).count(),
        2,
        "metadata filtering and enrichment must consume the same ranked Top-K"
    );
    assert!(
        native_plan.nodes().iter().any(|node| {
            matches!(
                &node.kind,
                PlanNode::Join(join) if join.spec.join_type == nexus_query_engine::plan::JoinType::LeftSemi
            )
        }),
        "the IN subquery must lower to a bounded native LeftSemi join"
    );
}

#[tokio::test]
async fn pagerank_composition_consumer_covers_filter_sort_limit() {
    let dir = unique_temp_dir("datafusion-nexus-cugraph-compose-structural");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny_directed_edges.parquet");
    write_tiny_directed_edges(&path);
    let (_backend, ctx) = backend_context();
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
    let composed =
        find_composed_exec(plan.as_ref()).expect("PageRank composition plan should contain NexusNativeComposedExec");
    let native_plan = composed.native_plan().to_string();
    for expected in ["GpuResidentSource", "Filter", "Sort", "Limit"] {
        assert!(
            native_plan.contains(expected),
            "composed native consumer should include {expected} over the cuGraph result:\n{native_plan}"
        );
    }
}
