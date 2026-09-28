fn graph_residency_bfs_spec(edge_columns: GraphEdgeColumnBindings) -> GraphAlgorithmSpec {
    GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        edge_columns,
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(7, None))),
    )
}

fn graph_keyable_descriptor(plan: &QueryPlan, spec: &GraphAlgorithmSpec) -> GraphResidencyDescriptor {
    let GraphResidencyPlanAnalysis::Keyable(descriptor) = graph_residency_descriptor_for_edge_plan(plan, spec) else {
        panic!("graph edge source should be keyable");
    };
    descriptor
}

fn graph_edge_source_plan(read_columns: &[&str], projection: Option<&[&str]>) -> QueryPlan {
    graph_edge_source_plan_from_source(graph_local_parquet_source(read_columns), projection)
}

fn graph_edge_source_plan_from_source(source: LocalParquetSource, projection: Option<&[&str]>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let root = match projection {
        Some(columns) => builder
            .add_projection(
                source,
                columns
                    .iter()
                    .map(|column| {
                        ProjectionOutput::new(
                            Expr::column(*column),
                            graph_edge_field(column).expect("test projection column must exist"),
                        )
                    })
                    .collect(),
            )
            .unwrap(),
        None => source,
    };
    builder.finish_gpu_resident(root).unwrap()
}

fn graph_local_parquet_source(read_columns: &[&str]) -> LocalParquetSource {
    graph_local_parquet_source_with_file(read_columns, graph_local_parquet_file())
}

fn graph_local_parquet_source_with_file(read_columns: &[&str], file: LocalParquetFile) -> LocalParquetSource {
    let fields = read_columns
        .iter()
        .map(|column| graph_edge_field(column).expect("test source column must exist"))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    LocalParquetSource::from_files(
        vec![file],
        schema,
        read_columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn graph_local_parquet_file() -> LocalParquetFile {
    LocalParquetFile::new("/tmp/graph-residency.parquet")
        .with_stable_id("local:/tmp/graph-residency.parquet:1024:mtime=2026-07-12T00:00:00Z")
        .with_file_size_bytes(1024)
        .with_row_groups(vec![0, 2])
}

fn graph_object_store_parquet_file(content_identity: &str) -> LocalParquetFile {
    let resolved =
        ResolvedParquetSource::remote_kvikio("s3://bucket/graph-residency.parquet", CredentialMode::SharedProcessEnv)
            .with_content_identity(content_identity);
    LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("object-store graph source should be valid")
        .with_file_size_bytes(1024)
        .with_row_groups(vec![0, 2])
}

fn graph_edge_field(column: &str) -> Option<Field> {
    match column {
        "src" | "dst" => Some(Field::new(column, DataType::Int64, false)),
        "weight" => Some(Field::new(column, DataType::Float64, false)),
        "edge_id" => Some(Field::new(column, DataType::Int64, false)),
        _ => None,
    }
}

fn graph_plan(algorithm: GraphAlgorithm, params: GraphAlgorithmParams, input_schema: Arc<Schema>) -> QueryPlan {
    graph_plan_with_options(algorithm, params, GraphConstructionOptions::default(), input_schema)
}

fn graph_plan_with_options(
    algorithm: GraphAlgorithm,
    params: GraphAlgorithmParams,
    graph_options: GraphConstructionOptions,
    input_schema: Arc<Schema>,
) -> QueryPlan {
    let vertex_type = input_schema.field_with_name("src").expect("test edge schema has src").data_type().clone();
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", input_schema)));
    let output_schema = if matches!(vertex_type, DataType::Int32 | DataType::Int64) {
        graph_result_schema_for_vertex_type(algorithm, &params, &vertex_type)
    } else {
        stable_graph_result_schema(algorithm)
    };
    let spec = GraphAlgorithmSpec::new(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        graph_options,
        params,
        output_schema,
    );
    let graph = builder.add_graph_algorithm(input, Vec::new(), spec).unwrap();
    builder.finish_with_host_sink(graph).unwrap()
}

fn edge_schema(vertex_type: DataType, weight_type: DataType, weight_nullable: bool) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_type.clone(), false),
        Field::new("dst", vertex_type, false),
        Field::new("weight", weight_type, weight_nullable),
    ]))
}

fn edge_schema_no_weight() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)]))
}

fn edge_schema_with_edge_id(vertex_type: DataType, weight_type: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_type.clone(), false),
        Field::new("dst", vertex_type.clone(), false),
        Field::new("weight", weight_type, false),
        Field::new("edge_id", vertex_type, false),
    ]))
}

// ═══════════════════════════════════════════════════════════════════════════
// cuGraph public-contract gate
// ═══════════════════════════════════════════════════════════════════════════

/// The native registry names that require the cuGraph graph implementation.
///
/// This is deliberately a contract over our public dependency and descriptor
/// layer, rather than a parser for its implementation source. API
/// changes in the local `cugraph` crate are caught by the compile-time type references
/// below and by the execution adapters that call those APIs.
#[cfg(feature = "cugraph")]
const CUGRAPH_REGISTRY_NAMES: &[&str] = &[
    "pagerank",
    "personalized_pagerank",
    "bfs",
    "louvain",
    "eigenvector_centrality",
    "katz_centrality",
    "hits",
    "sssp",
    "degrees_all",
    "in_degrees_all",
    "out_degrees_all",
    "weakly_connected_components",
    "strongly_connected_components",
    "triangle_count_all",
    "core_number",
    "k_core",
    "minimum_spanning_tree",
    "leiden",
    "ecg",
    "spectral_modularity_maximization",
    "force_atlas2",
];
