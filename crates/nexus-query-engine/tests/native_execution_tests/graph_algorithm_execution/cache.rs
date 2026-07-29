#[test]
fn source_alias_residency_does_not_preserve_graph_residency_descriptor() {
    let runtime = backend_runtime();
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let dir = unique_temp_dir("shared-residency-graph-descriptor");
    let edge_path = dir.join("edges.parquet");
    DataFrame::from_record_batch(&runtime, &edge_batch).unwrap().write_parquet(&edge_path).unwrap();
    let edge_file_size = fs::metadata(&edge_path).unwrap().len();
    let local_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&edge_path).with_file_size_bytes(edge_file_size).with_stable_id(format!(
            "local:{}:{}:mtime=test",
            edge_path.display(),
            edge_file_size
        ))],
        Arc::clone(&edge_schema),
        vec!["src".to_string(), "dst".to_string()],
    );
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_source));
    let descriptor_plan = builder.finish_with_host_sink(source);
    let GraphResidencyPlanAnalysis::Keyable(descriptor) =
        graph_residency_descriptor_for_edge_plan(&descriptor_plan, &spec)
    else {
        panic!("local parquet edge source should be keyable for graph residency");
    };

    let source = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &edge_batch).unwrap())
        .with_graph_residency(descriptor.clone());
    let selected = source.frame().select(["src", "dst"]).unwrap();
    let derived = GpuDataFrame::try_new_with_source_residency(selected, &source).unwrap();

    assert!(
        derived.graph_residency().is_none(),
        "shared-residency derived frames must not blindly preserve graph residency"
    );
}
