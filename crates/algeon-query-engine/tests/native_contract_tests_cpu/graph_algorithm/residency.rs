use super::*;

#[test]
fn graph_residency_descriptor_sees_through_edge_projection() {
    let plan = graph_edge_source_plan(&["src", "dst", "weight"], Some(&["src", "dst", "weight"]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );

    let GraphResidencyPlanAnalysis::Keyable(descriptor) = graph_residency_descriptor_for_edge_plan(&plan, &spec) else {
        panic!("projected local parquet graph edge source should be keyable");
    };
    assert!(descriptor.source_fingerprint.as_str().starts_with("graph-source-v1-fnv64-"));
    assert_eq!(descriptor.vertex_data_type, DataType::Int64);
    assert_eq!(descriptor.weight_data_type, Some(DataType::Float64));
    assert_eq!(descriptor.graph_vertex_upper_bound(), None);
    assert_eq!(descriptor.graph_raw_edge_row_count(), None);

    let forward = descriptor.resident_key(&spec, GraphResidencyOrientation::Forward);
    let transposed = descriptor.resident_key(&spec, GraphResidencyOrientation::Transposed);
    assert_ne!(forward, transposed);
    assert_eq!(forward.weight_column.as_deref(), Some("weight"));
}

#[test]
fn graph_residency_descriptor_for_content_identity_requires_a_stable_nonempty_identity() {
    let first = GraphResidencyDescriptor::for_content_identity(
        "sha256:edge-rowset-a",
        DataType::Int32,
        Some(DataType::Float64),
        None,
    )
    .unwrap();
    let same = GraphResidencyDescriptor::for_content_identity(
        "sha256:edge-rowset-a",
        DataType::Int32,
        Some(DataType::Float64),
        None,
    )
    .unwrap();
    let different = GraphResidencyDescriptor::for_content_identity(
        "sha256:edge-rowset-b",
        DataType::Int32,
        Some(DataType::Float64),
        None,
    )
    .unwrap();

    assert_eq!(first.source_fingerprint, same.source_fingerprint);
    assert_ne!(first.source_fingerprint, different.source_fingerprint);
    assert_eq!(first.vertex_data_type, DataType::Int32);
    assert_eq!(first.weight_data_type, Some(DataType::Float64));
    assert_eq!(first.graph_vertex_upper_bound(), None);
    assert_eq!(first.graph_raw_edge_row_count(), None);

    let error = GraphResidencyDescriptor::for_content_identity(" ", DataType::Int32, None, None).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("missing_content_identity"));
}

#[test]
fn graph_residency_descriptor_carries_a_footer_vertex_bound_only_for_the_complete_source_rowset() {
    let source = graph_local_parquet_source_with_file(
        &["src", "dst"],
        LocalParquetFile::new("/tmp/graph-residency-footer.parquet")
            .with_stable_id("local:/tmp/graph-residency-footer.parquet:1024:mtime=2026-08-05T00:00:00Z")
            .with_file_size_bytes(1024)
            .with_row_group_rows(vec![45_564_149])
            .with_graph_vertex_upper_bound(4_894_081),
    );
    let plan = graph_edge_source_plan_from_source(source, Some(&["src", "dst"]));
    let spec = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>));

    let GraphResidencyPlanAnalysis::Keyable(descriptor) = graph_residency_descriptor_for_edge_plan(&plan, &spec) else {
        panic!("complete local parquet source should retain its footer graph bound");
    };
    assert_eq!(descriptor.graph_vertex_upper_bound(), Some(4_894_081));
    assert_eq!(descriptor.graph_raw_edge_row_count(), Some(45_564_149));

    let partial_source = graph_local_parquet_source_with_file(
        &["src", "dst"],
        LocalParquetFile::new("/tmp/graph-residency-partial.parquet")
            .with_stable_id("local:/tmp/graph-residency-partial.parquet:1024:mtime=2026-08-05T00:00:00Z")
            .with_file_size_bytes(1024)
            .with_row_group_rows(vec![7, 11])
            .with_row_groups(vec![0])
            .with_graph_vertex_upper_bound(18),
    );
    let partial_plan = graph_edge_source_plan_from_source(partial_source, Some(&["src", "dst"]));
    let GraphResidencyPlanAnalysis::Keyable(partial_descriptor) =
        graph_residency_descriptor_for_edge_plan(&partial_plan, &spec)
    else {
        panic!("partial local parquet source remains cache-keyable but cannot retain full-source facts");
    };
    assert_eq!(partial_descriptor.graph_vertex_upper_bound(), None);
    assert_eq!(partial_descriptor.graph_raw_edge_row_count(), None);
}

#[test]
fn simple_directed_topology_proof_survives_only_column_preserving_native_projections() {
    let topology = GraphSimpleDirectedEdgeTopology::new(0, 1).expect("source and destination columns differ");
    let source = graph_local_parquet_source_with_file(
        &["src", "dst"],
        LocalParquetFile::new("/tmp/graph-residency-simple-directed.parquet")
            .with_stable_id("local:/tmp/graph-residency-simple-directed.parquet:1024:mtime=2026-08-06T00:00:00Z")
            .with_file_size_bytes(1024)
            .with_graph_simple_directed_edge_topology(topology),
    );
    let projected_plan = graph_edge_source_plan_from_source(source, Some(&["src", "dst"]));

    assert!(edge_plan_has_simple_directed_edge_topology(&projected_plan, "src", "dst"));
    assert!(!edge_plan_has_simple_directed_edge_topology(&projected_plan, "dst", "src"));

    let source = graph_local_parquet_source_with_file(
        &["src", "dst"],
        LocalParquetFile::new("/tmp/graph-residency-simple-directed-filtered.parquet")
            .with_stable_id(
                "local:/tmp/graph-residency-simple-directed-filtered.parquet:1024:mtime=2026-08-06T00:00:00Z",
            )
            .with_file_size_bytes(1024)
            .with_graph_simple_directed_edge_topology(topology),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(source, Expr::literal(LiteralValue::Boolean(true))).unwrap();
    let filtered_plan = builder.finish_gpu_resident(filtered).unwrap();

    assert!(
        !edge_plan_has_simple_directed_edge_topology(&filtered_plan, "src", "dst"),
        "a file footer cannot prove a rowset after a row-changing native node"
    );
}

#[test]
fn graph_residency_descriptor_ignores_unconsumed_weight_without_python_normalization() {
    let with_extra_weight = graph_edge_source_plan(&["src", "dst", "weight"], Some(&["src", "dst"]));
    let without_extra_weight = graph_edge_source_plan(&["src", "dst"], Some(&["src", "dst"]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(7, None))),
    );

    let GraphResidencyPlanAnalysis::Keyable(with_extra_weight) =
        graph_residency_descriptor_for_edge_plan(&with_extra_weight, &spec)
    else {
        panic!("extra unconsumed weight source should still be keyable");
    };
    let GraphResidencyPlanAnalysis::Keyable(without_extra_weight) =
        graph_residency_descriptor_for_edge_plan(&without_extra_weight, &spec)
    else {
        panic!("unweighted source should be keyable");
    };

    assert_eq!(with_extra_weight.source_fingerprint, without_extra_weight.source_fingerprint);
    assert_eq!(with_extra_weight.weight_data_type, None);
}

#[test]
fn graph_residency_descriptor_includes_weight_used_by_python_normalization() {
    let plan = graph_edge_source_plan(&["src", "dst", "weight"], Some(&["src", "dst", "weight"]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(7, None))),
    )
    .with_edge_normalize_inserted(true);

    let GraphResidencyPlanAnalysis::Keyable(descriptor) = graph_residency_descriptor_for_edge_plan(&plan, &spec) else {
        panic!("python-normalized weighted source should be keyable");
    };

    assert_eq!(descriptor.weight_data_type, Some(DataType::Float64));
}

#[test]
fn planned_resident_graph_key_matches_cached_execution_or_bypasses() {
    let plan = graph_edge_source_plan(&["src", "dst", "weight"], Some(&["src", "dst", "weight"]));
    let pagerank = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let pagerank_descriptor = graph_keyable_descriptor(&plan, &pagerank);
    assert_eq!(
        planned_resident_graph_key(&pagerank_descriptor, &pagerank),
        Some(pagerank_descriptor.resident_key(&pagerank, GraphResidencyOrientation::Transposed))
    );

    let bfs = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>));
    let bfs_descriptor =
        graph_keyable_descriptor(&graph_edge_source_plan(&["src", "dst"], Some(&["src", "dst"])), &bfs);
    assert_eq!(
        planned_resident_graph_key(&bfs_descriptor, &bfs),
        Some(bfs_descriptor.resident_key(&bfs, GraphResidencyOrientation::Forward))
    );

    let mut include_edges_params = GraphBfsParams::new(7, None);
    include_edges_params.predicates.include_edges = Some(GraphEdgePredicateInput::new(1, "src", "dst", None::<String>));
    let bfs_include_edges = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<String>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(include_edges_params)),
    );
    let include_edges_descriptor =
        graph_keyable_descriptor(&graph_edge_source_plan(&["src", "dst"], Some(&["src", "dst"])), &bfs_include_edges);
    assert!(planned_resident_graph_key(&include_edges_descriptor, &bfs_include_edges).is_none());

    let bfs_edge_id =
        graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>).with_edge_id("edge_id"));
    let edge_id_descriptor = graph_keyable_descriptor(
        &graph_edge_source_plan(&["src", "dst", "edge_id"], Some(&["src", "dst", "edge_id"])),
        &bfs_edge_id,
    );
    assert!(planned_resident_graph_key(&edge_id_descriptor, &bfs_edge_id).is_none());
}

#[test]
fn graph_residency_fingerprint_distinguishes_rowset_and_preparation_contracts() {
    let raw_spec = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>));
    let base_plan = graph_edge_source_plan(&["src", "dst"], Some(&["src", "dst"]));
    let base = graph_keyable_descriptor(&base_plan, &raw_spec);

    let swapped_spec = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("dst", "src", None::<String>));
    assert_ne!(base.source_fingerprint, graph_keyable_descriptor(&base_plan, &swapped_spec).source_fingerprint);

    let mtime_plan = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(
            &["src", "dst"],
            graph_local_parquet_file()
                .with_stable_id("local:/tmp/graph-residency.parquet:1024:mtime=2026-07-12T00:00:01Z"),
        ),
        Some(&["src", "dst"]),
    );
    assert_ne!(base.source_fingerprint, graph_keyable_descriptor(&mtime_plan, &raw_spec).source_fingerprint);

    let row_group_plan = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(&["src", "dst"], graph_local_parquet_file().with_row_groups(vec![1])),
        Some(&["src", "dst"]),
    );
    assert_ne!(base.source_fingerprint, graph_keyable_descriptor(&row_group_plan, &raw_spec).source_fingerprint);

    let predicate = Expr::binary(Expr::column("src"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(7)));
    let predicate_plan = graph_edge_source_plan_from_source(
        graph_local_parquet_source(&["src", "dst"]).with_reader_predicate(predicate),
        Some(&["src", "dst"]),
    );
    assert_ne!(base.source_fingerprint, graph_keyable_descriptor(&predicate_plan, &raw_spec).source_fingerprint);

    let python_spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<String>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(7, None))),
    )
    .with_edge_normalize_inserted(true);
    assert_ne!(
        base.resident_key(&raw_spec, GraphResidencyOrientation::Forward),
        graph_keyable_descriptor(&base_plan, &python_spec)
            .resident_key(&python_spec, GraphResidencyOrientation::Forward),
    );
}

#[test]
fn graph_residency_descriptor_distinguishes_object_store_content_identity() {
    let spec = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>));
    let content_a = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(&["src", "dst"], graph_object_store_parquet_file("etag-a")),
        Some(&["src", "dst"]),
    );
    let content_b = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(&["src", "dst"], graph_object_store_parquet_file("etag-b")),
        Some(&["src", "dst"]),
    );
    assert_ne!(
        graph_keyable_descriptor(&content_a, &spec).source_fingerprint,
        graph_keyable_descriptor(&content_b, &spec).source_fingerprint,
    );

    let weak_identity = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(
            &["src", "dst"],
            graph_object_store_parquet_file("weak_size_1024_rows_10"),
        ),
        Some(&["src", "dst"]),
    );
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&weak_identity, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::UnstableSourceIdentity),
    );

    let weak_etag = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(&["src", "dst"], graph_object_store_parquet_file("W/\"etag\"")),
        Some(&["src", "dst"]),
    );
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&weak_etag, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::UnstableSourceIdentity),
    );

    let empty_identity = graph_edge_source_plan_from_source(
        graph_local_parquet_source_with_file(&["src", "dst"], graph_object_store_parquet_file("  ")),
        Some(&["src", "dst"]),
    );
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&empty_identity, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::UnstableSourceIdentity),
    );
}

#[test]
fn graph_residency_descriptor_bypasses_unstable_identity_and_scan_limit() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<String>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(7, None))),
    );

    let unstable_source = graph_local_parquet_source_with_file(
        &["src", "dst"],
        LocalParquetFile::new("/tmp/graph-unstable.parquet").with_file_size_bytes(1024),
    );
    let unstable_plan = graph_edge_source_plan_from_source(unstable_source, Some(&["src", "dst"]));
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&unstable_plan, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::UnstableSourceIdentity),
    );

    let scan_limit_plan = graph_edge_source_plan_from_source(
        graph_local_parquet_source(&["src", "dst"]).with_scan_limit(Some(10)),
        Some(&["src", "dst"]),
    );
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&scan_limit_plan, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::UnsupportedTransformedInput),
    );
}

#[test]
fn graph_residency_descriptor_bypasses_transforms_and_gpu_inputs() {
    let spec = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>));

    let local_source = graph_local_parquet_source(&["src", "dst"]);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_source));
    let filter = builder.add_filter(source, Expr::literal(LiteralValue::Boolean(true))).unwrap();
    let filtered_plan = builder.finish_gpu_resident(filter).unwrap();
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&filtered_plan, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::UnsupportedTransformedInput),
    );

    let mut builder = QueryPlanBuilder::new();
    let gpu = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", edge_schema_no_weight())));
    let gpu_plan = builder.finish_gpu_resident(gpu).unwrap();
    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&gpu_plan, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::NonLocalParquetSource),
    );
}

#[test]
fn graph_residency_descriptor_reports_vertex_dtype_mismatch() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![graph_local_parquet_file()],
        schema,
        vec!["src".to_string(), "dst".to_string()],
    );
    let plan = graph_edge_source_plan_from_source(source, Some(&["src", "dst"]));
    let spec = graph_residency_bfs_spec(GraphEdgeColumnBindings::new("src", "dst", None::<String>));

    assert_eq!(
        graph_residency_descriptor_for_edge_plan(&plan, &spec).bypass_reason(),
        Some(GraphSourceFingerprintBypassReason::EdgeVertexDataTypeMismatch),
    );
}
