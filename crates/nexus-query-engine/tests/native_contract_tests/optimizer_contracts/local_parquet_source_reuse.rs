#[test]
fn native_optimizer_reuses_compatible_small_local_parquet_source_variants() {
    let plan = compatible_small_local_parquet_source_variants_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    let candidate = &result.diagnostics.local_parquet_source_reuse.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.reuse_count, 2);
    assert_eq!(candidate.branch_filter_count, 1);
    assert_eq!(candidate.superset_read_columns, vec!["id", "state", "zip"]);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address"), 1);
    assert!(
        result.plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Filter(_))),
        "reader predicate should be replayed as a branch-local native filter"
    );
}

#[test]
fn gpu_local_parquet_source_cache_key_isolates_resolved_source_contracts() {
    let direct_source = contract_sensitive_customer_address_source(remote_direct_customer_address_contract());
    let host_buffer_source = contract_sensitive_customer_address_source(host_buffer_customer_address_contract());
    let decimal_source = contract_sensitive_customer_address_source(remote_direct_customer_address_contract())
        .with_reader_decimal_contract("id", ParquetReaderDecimalTypeId::Decimal64, -2);

    let direct_source_key = GpuLocalParquetSourceCacheKey::from_source(&direct_source);
    let host_buffer_source_key = GpuLocalParquetSourceCacheKey::from_source(&host_buffer_source);
    let decimal_source_key = GpuLocalParquetSourceCacheKey::from_source(&decimal_source);
    assert_ne!(
        direct_source_key, host_buffer_source_key,
        "whole-source GPU cache keys must include resolved read/cache/credential contract fields"
    );
    assert_ne!(
        direct_source_key, decimal_source_key,
        "whole-source GPU cache keys must include per-file Parquet reader decimal contracts"
    );
    assert!(direct_source_key.as_str().contains("file.0.resolved.read_strategy=remote_direct"));
    assert!(direct_source_key.as_str().contains("file.0.resolved.remote_read_mode=require_direct"));
    assert!(direct_source_key.as_str().contains("file.0.resolved.credential_mode=shared_process_env"));
    assert!(direct_source_key.as_str().contains("file.0.resolved.storage_profile_id=remote-profile-a"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.read_strategy=remote_host_buffer"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.decode_input=host_buffer"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.lakehouse_cache_status=miss"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.lakehouse_cache_read_mode=enabled"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.lakehouse_cache_fill_mode=background"));
    assert!(decimal_source_key.as_str().contains("file.0.reader_decimal_types.0.type_id=decimal64"));
    assert!(!direct_source_key.as_str().contains("query_variant=a"));
    assert!(!host_buffer_source_key.as_str().contains("/tmp/nexus-lakehouse-cache"));

    let mut builder = QueryPlanBuilder::new();
    let direct_source_id = builder.add_source(SourceNode::LocalParquet(direct_source.clone()));
    let host_buffer_source_id = builder.add_source(SourceNode::LocalParquet(host_buffer_source.clone()));
    let decimal_source_id = builder.add_source(SourceNode::LocalParquet(decimal_source.clone()));
    let direct_chunk_plan =
        plan_local_parquet_source_chunks(direct_source_id, &direct_source, SourceChunkPlanningPolicy::new(None, 1))
            .unwrap();
    let host_buffer_chunk_plan = plan_local_parquet_source_chunks(
        host_buffer_source_id,
        &host_buffer_source,
        SourceChunkPlanningPolicy::new(None, 1),
    )
    .unwrap();
    let decimal_chunk_plan =
        plan_local_parquet_source_chunks(decimal_source_id, &decimal_source, SourceChunkPlanningPolicy::new(None, 1))
            .unwrap();
    let direct_chunk_key =
        GpuLocalParquetSourceCacheKey::from_source_chunk(&direct_source, &direct_chunk_plan.chunks[0]);
    let host_buffer_chunk_key =
        GpuLocalParquetSourceCacheKey::from_source_chunk(&host_buffer_source, &host_buffer_chunk_plan.chunks[0]);
    let decimal_chunk_key =
        GpuLocalParquetSourceCacheKey::from_source_chunk(&decimal_source, &decimal_chunk_plan.chunks[0]);

    assert_ne!(
        direct_chunk_key, host_buffer_chunk_key,
        "bounded chunk GPU cache keys must include resolved read/cache/credential contract fields"
    );
    assert_ne!(
        direct_chunk_key, decimal_chunk_key,
        "bounded chunk GPU cache keys must include per-chunk Parquet reader decimal contracts"
    );
    assert!(direct_chunk_key.as_str().contains("chunk.resolved.read_strategy=remote_direct"));
    assert!(host_buffer_chunk_key.as_str().contains("chunk.resolved.read_strategy=remote_host_buffer"));
    assert!(decimal_chunk_key.as_str().contains("chunk.reader_decimal_types.0.type_id=decimal64"));
    assert!(decimal_chunk_key.as_str().contains("chunk.reader_decimal_types_footer_verified=false"));
}

#[test]
fn native_optimizer_keeps_local_parquet_source_reuse_contract_isolated() {
    let plan = contract_mismatched_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(
        result.diagnostics.local_parquet_source_reuse.considered_count, 0,
        "sources with different resolved read/cache/credential contracts must not enter the same reuse group"
    );
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 0);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address"), 2);
}

#[test]
fn native_optimizer_local_parquet_source_reuse_diagnostics_redact_stable_file_ids() {
    let plan = compatible_sensitive_stable_id_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    let stable_file_ids = &result.diagnostics.local_parquet_source_reuse.candidates[0].stable_file_ids;
    assert_eq!(stable_file_ids.len(), 1);
    assert!(stable_file_ids[0].starts_with("object_store:s3:fnv64:"));
    for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
        assert!(
            !stable_file_ids[0].contains(sensitive),
            "{sensitive} leaked in source reuse diagnostic: {:?}",
            stable_file_ids
        );
    }
}

#[test]
fn native_optimizer_keeps_decimal_contract_source_reuse_isolated() {
    let plan = decimal_contract_mismatched_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(
        result.diagnostics.local_parquet_source_reuse.considered_count, 0,
        "sources with different per-file decimal reader contracts must not enter the same reuse group"
    );
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 0);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address"), 2);
}

#[test]
fn native_optimizer_rejects_local_parquet_source_reuse_with_different_file_order() {
    let plan = compatible_local_parquet_source_variants_with_different_file_order_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };
    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 0);
    assert_eq!(
        result.diagnostics.local_parquet_source_reuse.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address_a"), 2);
}

#[test]
fn native_optimizer_rejects_source_reuse_with_conflicting_file_index_column_names() {
    let plan = conflicting_file_index_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
}

#[test]
fn native_optimizer_rejects_source_reuse_with_conflicting_column_name_file_indices() {
    let plan = conflicting_column_name_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
}

#[test]
fn native_optimizer_does_not_lift_variant_max_rows_to_shared_local_parquet_source() {
    let plan = compatible_small_local_parquet_source_variants_with_max_rows_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    let sources = local_parquet_sources_for_stable_id(&result.plan, "customer_address");
    let [shared_source] = sources.as_slice() else {
        panic!("source reuse should produce exactly one shared customer_address source, got {}", sources.len());
    };
    assert_eq!(
        shared_source.max_rows, None,
        "branch-local max_rows is valid only after the branch predicate and must not be lifted to the shared source"
    );
}

#[test]
fn native_optimizer_rewrites_graph_algorithm_bfs_side_input_references_with_source_reuse() {
    let plan = graph_bfs_local_source_reuse_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    assert_all_node_inputs_exist(&result.plan);

    let PlanNode::GraphAlgorithm(graph) = &result.plan.root_node().kind else {
        panic!("optimized root should remain a graph algorithm");
    };
    let GraphAlgorithmParams::Bfs(params) = &graph.spec.params else {
        panic!("expected BFS graph params");
    };
    let target_vertices = params.predicates.target_vertices.as_ref().expect("BFS target side input should exist");

    assert_eq!(graph.inputs.first().copied(), Some(graph.input));
    assert_eq!(graph.inputs.get(1).copied(), Some(target_vertices.input));
    assert!(
        result.plan.node(target_vertices.input).is_some(),
        "BFS target side-input binding must be remapped to an existing rewritten node"
    );
    assert_eq!(result.plan.root_node().input_ids(), graph.inputs.as_slice());
}

#[test]
fn native_optimizer_reuses_large_projection_only_local_parquet_source_variants() {
    let plan = large_local_parquet_source_variants_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(candidate.superset_read_columns, vec!["inv_item_sk", "inv_date_sk", "inv_store_sk"]);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
}

#[test]
fn native_optimizer_reuses_large_existing_superset_with_smaller_local_parquet_source_variant() {
    let plan = large_superset_with_smaller_local_parquet_source_variant_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.superset_read_columns, vec!["inv_item_sk", "inv_date_sk"]);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
}

#[test]
fn native_optimizer_rejects_large_superset_reuse_when_variant_feeds_grouped_aggregate() {
    let plan = large_superset_with_grouped_aggregate_source_variant_plan();
    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::VariantTooLarge)
    );
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 2);
}

#[test]
fn native_optimizer_reuses_large_superset_when_grouped_aggregate_variant_is_not_widened() {
    let plan = large_grouped_aggregate_superset_with_smaller_source_variant_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert!(diagnostics.candidates[0].applied);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
}

#[test]
fn native_optimizer_reuses_large_filtered_existing_superset_outside_source_adjacent_grouped_aggregate() {
    let plan = large_filtered_superset_with_smaller_local_parquet_source_variant_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };
    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert!(diagnostics.candidates[0].applied);
    assert_eq!(diagnostics.candidates[0].branch_filter_count, 1);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
    let branch_filters = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Filter(filter) => Some(&filter.predicate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(branch_filters.len(), 1);
    assert!(
        branch_filters[0].to_string().contains(" OR "),
        "test predicate should cover recursive replay normalization"
    );
    assert!(
        branch_filters[0].to_string().contains(" AND "),
        "test predicate should cover recursive replay normalization"
    );
    assert_no_non_decimal_binary_output_types(
        branch_filters[0],
        "replayed reader predicate should strip declared boolean output type before native filter validation",
    );
}

fn compatible_small_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let full = builder.add_source(customer_address_source(&["id", "state"], &[0, 1], None));
    let filtered = builder.add_source(customer_address_source(
        &["id", "zip"],
        &[0, 2],
        Some(Expr::binary(Expr::column("zip"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(30301)))),
    ));
    let full_id = builder.add_projection(
        full,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let filtered_id = builder.add_projection(
        filtered,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![full_id, filtered_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn contract_mismatched_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let direct = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        remote_direct_customer_address_contract(),
    )));
    let host_buffer = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        host_buffer_customer_address_contract(),
    )));
    let direct_id = builder.add_projection(
        direct,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let host_buffer_id = builder.add_projection(
        host_buffer,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![direct_id, host_buffer_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_sensitive_stable_id_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let full = builder.add_source(sensitive_stable_id_customer_address_source(&["id", "state"], &[0, 1], None));
    let filtered = builder.add_source(sensitive_stable_id_customer_address_source(
        &["id", "zip"],
        &[0, 2],
        Some(Expr::binary(Expr::column("zip"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(30301)))),
    ));
    let full_id = builder.add_projection(
        full,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let filtered_id = builder.add_projection(
        filtered,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![full_id, filtered_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn decimal_contract_mismatched_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let plain = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        remote_direct_customer_address_contract(),
    )));
    let decimal = builder.add_source(SourceNode::LocalParquet(
        contract_sensitive_customer_address_source(remote_direct_customer_address_contract())
            .with_reader_decimal_contract("id", ParquetReaderDecimalTypeId::Decimal64, -2),
    ));
    let plain_id = builder.add_projection(
        plain,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let decimal_id = builder.add_projection(
        decimal,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![plain_id, decimal_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_local_parquet_source_variants_with_different_file_order_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let first = builder.add_source(customer_address_two_file_source(&["id", "state"], &[0, 1], false));
    let second = builder.add_source(customer_address_two_file_source(&["id", "zip"], &[0, 2], true));
    let first_id = builder.add_projection(
        first,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let second_id = builder.add_projection(
        second,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![first_id, second_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_small_local_parquet_source_variants_with_max_rows_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let full = builder.add_source(customer_address_source(&["id", "state"], &[0, 1], None));
    let filtered = builder.add_source(customer_address_source_with_max_rows(
        &["id", "zip"],
        &[0, 2],
        Some(Expr::binary(Expr::column("zip"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(30301)))),
        Some(1),
    ));
    let full_id = builder.add_projection(
        full,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let filtered_id = builder.add_projection(
        filtered,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![full_id, filtered_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn conflicting_file_index_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let id_source = builder.add_source(customer_address_source(&["id"], &[0], None));
    let zip_source = builder.add_source(customer_address_source(&["zip"], &[0], None));
    let id = builder.add_projection(
        id_source,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let zip_as_id = builder.add_projection(
        zip_source,
        vec![ProjectionOutput::new(Expr::column("zip").alias("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![id, zip_as_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn conflicting_column_name_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let first = builder.add_source(customer_address_source(&["id"], &[0], None));
    let second = builder.add_source(customer_address_source(&["id"], &[1], None));
    let first = builder.add_projection(
        first,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let second = builder.add_projection(
        second,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![first, second], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn graph_bfs_local_source_reuse_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(graph_reuse_source(&["src", "dst", "weight"], &[0, 1, 2]));
    let target_vertices = builder.add_source(graph_reuse_source(&["vertex_id"], &[3]));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            target_vertices: Some(GraphPredicateInput::new(target_vertices, "vertex_id")),
            ..Default::default()
        },
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    builder.finish_gpu_resident(graph)
}

fn graph_reuse_source(columns: &[&str], file_indices: &[usize]) -> SourceNode {
    let fields = columns.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let file = LocalParquetFile::new("/tmp/graph_inputs.parquet")
        .with_stable_id("graph_inputs")
        .with_row_group_bytes(vec![1024 * 1024])
        .with_row_group_rows(vec![4096]);
    let read_columns = columns.iter().map(|column| (*column).to_string()).collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(vec![file], schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    ))
}

fn large_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let right = builder.add_source(estimated_local_source_with_stable_id_and_file_indices(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_store_sk"],
        &[0, 2],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let left_item = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let right_item = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left_item, right_item], schema(&["inv_item_sk"]));
    builder.finish_gpu_resident(union)
}

fn large_superset_with_smaller_local_parquet_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        600 * 1024 * 1024,
        10_000_000,
    ));
    let subset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let superset_item = builder.add_projection(
        superset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let subset_item = builder.add_projection(
        subset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![superset_item, subset_item], schema(&["inv_item_sk"]));
    builder.finish_gpu_resident(union)
}

fn large_superset_with_grouped_aggregate_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity", "inv_price"],
        600 * 1024 * 1024,
        10_000_000,
    ));
    let aggregate_input = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let aggregate_schema = schema(&["inv_item_sk", "sum_qty"]);
    let aggregate = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            vec![Expr::column("inv_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("inv_quantity"), "sum_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let superset_item = builder.add_projection(
        superset,
        vec![
            ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("inv_quantity"), Field::new("sum_qty", DataType::Int32, false)),
        ],
    );
    let union = builder.add_union(vec![aggregate, superset_item], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn large_grouped_aggregate_superset_with_smaller_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_input = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity", "inv_price"],
        600 * 1024 * 1024,
        10_000_000,
    ));
    let subset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let aggregate_schema = schema(&["inv_item_sk", "sum_qty"]);
    let aggregate = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            vec![Expr::column("inv_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("inv_quantity"), "sum_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let subset_item = builder.add_projection(
        subset,
        vec![
            ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::literal(LiteralValue::Int32(0)), Field::new("sum_qty", DataType::Int32, false)),
        ],
    );
    let union = builder.add_union(vec![aggregate, subset_item], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn large_filtered_superset_with_smaller_local_parquet_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        600 * 1024 * 1024,
        10_000_000,
        nested_reader_predicate_with_declared_boolean_output_types(),
    ));
    let subset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let superset_item = builder.add_projection(
        superset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let subset_item = builder.add_projection(
        subset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![superset_item, subset_item], schema(&["inv_item_sk"]));
    builder.finish_gpu_resident(union)
}

fn customer_address_source(columns: &[&str], file_indices: &[usize], predicate: Option<Expr>) -> SourceNode {
    customer_address_source_with_max_rows(columns, file_indices, predicate, None)
}

fn sensitive_stable_id_customer_address_source(
    columns: &[&str],
    file_indices: &[usize],
    predicate: Option<Expr>,
) -> SourceNode {
    let SourceNode::LocalParquet(mut source) = customer_address_source(columns, file_indices, predicate) else {
        unreachable!("customer address source helper always builds local Parquet")
    };
    source.files[0].stable_id = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc".to_string();
    source.refresh_diagnostics();
    SourceNode::LocalParquet(source)
}

fn contract_sensitive_customer_address_source(resolved_source: ResolvedParquetSource) -> LocalParquetSource {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
        .unwrap()
        .with_stable_id("customer_address")
        .with_row_group_bytes(vec![2 * 1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    LocalParquetSource::from_files(vec![file], schema, vec!["id".to_string()])
}

trait LocalParquetSourceTestExt {
    fn with_reader_decimal_contract(
        self,
        column_name: &str,
        type_id: ParquetReaderDecimalTypeId,
        cudf_scale: i32,
    ) -> Self;
}

impl LocalParquetSourceTestExt for LocalParquetSource {
    fn with_reader_decimal_contract(
        mut self,
        column_name: &str,
        type_id: ParquetReaderDecimalTypeId,
        cudf_scale: i32,
    ) -> Self {
        self.files[0].reader_decimal_types =
            vec![ParquetReaderDecimalType::new(column_name.to_string(), type_id, cudf_scale)];
        self.refresh_diagnostics();
        self
    }
}

fn remote_direct_customer_address_contract() -> ResolvedParquetSource {
    ResolvedParquetSource::remote_direct(
        "s3://warehouse/customer_address.parquet?query_variant=a",
        CredentialMode::SharedProcessEnv,
    )
    .with_remote_read_mode(RemoteParquetReadMode::RequireDirect)
    .with_storage_profile_id("remote-profile-a")
    .with_storage_domain("warehouse@s3")
    .with_content_identity("snapshot=42,file=customer_address,etag=abc")
}

fn host_buffer_customer_address_contract() -> ResolvedParquetSource {
    ResolvedParquetSource::remote_host_buffer(
        "s3://warehouse/customer_address.parquet?query_variant=a",
        "/tmp/nexus-lakehouse-cache/customer_address.parquet",
        CredentialMode::SharedProcessEnv,
    )
    .with_storage_profile_id("remote-profile-a")
    .with_storage_domain("warehouse@s3")
    .with_content_identity("snapshot=42,file=customer_address,etag=abc")
    .with_lakehouse_cache_config("/tmp/nexus-lakehouse-cache", Some(128 * 1024 * 1024))
    .with_lakehouse_cache_path("/tmp/nexus-lakehouse-cache/customer_address.parquet")
    .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Enabled)
    .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background)
    .with_cache_status(LakehouseCacheStatus::Miss)
}

fn customer_address_two_file_source(columns: &[&str], file_indices: &[usize], reversed: bool) -> SourceNode {
    let fields = columns.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let mut files = vec![
        LocalParquetFile::new("/tmp/customer_address_a.parquet")
            .with_stable_id("customer_address_a")
            .with_row_group_bytes(vec![2 * 1024 * 1024])
            .with_row_group_rows(vec![100_000]),
        LocalParquetFile::new("/tmp/customer_address_b.parquet")
            .with_stable_id("customer_address_b")
            .with_row_group_bytes(vec![2 * 1024 * 1024])
            .with_row_group_rows(vec![100_000]),
    ];
    if reversed {
        files.reverse();
    }
    let read_columns = columns.iter().map(|column| (*column).to_string()).collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(files, schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    );
    SourceNode::LocalParquet(source)
}

fn customer_address_source_with_max_rows(
    columns: &[&str],
    file_indices: &[usize],
    predicate: Option<Expr>,
    max_rows: Option<usize>,
) -> SourceNode {
    let fields = columns.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let file = LocalParquetFile::new("/tmp/customer_address.parquet")
        .with_stable_id("customer_address")
        .with_row_group_bytes(vec![2 * 1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    let read_columns = columns.iter().map(|column| (*column).to_string()).collect::<Vec<_>>();
    let mut source = LocalParquetSource::from_files(vec![file], schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    );
    if let Some(predicate) = predicate {
        source =
            source.with_column_roles(vec!["id".to_string()], vec!["zip".to_string()]).with_reader_predicate(predicate);
    }
    source = source.with_max_rows(max_rows);
    SourceNode::LocalParquet(source)
}

fn gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
}

fn local_parquet_source_count_for_stable_id(plan: &nexus_query_engine::QueryPlan, stable_id: &str) -> usize {
    local_parquet_sources_for_stable_id(plan, stable_id).len()
}

fn local_parquet_sources_for_stable_id<'a>(
    plan: &'a nexus_query_engine::QueryPlan,
    stable_id: &str,
) -> Vec<&'a LocalParquetSource> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) if source.files.iter().any(|file| file.stable_id == stable_id) => {
                    Some(source)
                },
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn assert_source_input(plan: &nexus_query_engine::QueryPlan, node_id: nexus_query_engine::NodeId, name: &str) {
    let node = plan.node(node_id).expect("node should exist");
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(source) => assert_eq!(source.input_name, name),
            other => panic!("expected GPU-resident source {name}, got {other:?}"),
        },
        other => panic!("expected source {name}, got {other:?}"),
    }
}
