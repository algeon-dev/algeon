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
    let sources = local_parquet_sources_for_stable_id(&result.plan, "customer_address");
    assert_eq!(sources.len(), 1);
    validate_local_parquet_source(sources[0]).expect("shared source must retain a valid schema mapping contract");
    assert_eq!(sources[0].diagnostics.schema_mapping_disposition, SchemaMappingDisposition::DirectByName);
    assert!(sources[0].diagnostics.rejected_features.is_empty());
    assert!(sources[0].unique_key_columns.is_empty(), "source reuse must not leak branch-local proof");
    assert_eq!(sources[0].projected_output_columns.len(), sources[0].projected_output_column_source_refs.len());
    assert_eq!(sources[0].pure_filter_columns.len(), sources[0].pure_filter_column_source_refs.len());
    assert!(
        sources[0]
            .projected_output_column_source_refs
            .iter()
            .chain(sources[0].pure_filter_column_source_refs.iter())
            .all(|source_ref| matches!(source_ref, SourceColumnRef::File(_)))
    );
    assert!(
        result.plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Filter(_))),
        "reader predicate should be replayed as a branch-local native filter"
    );
}

#[test]
fn native_optimizer_rejects_source_reuse_with_branch_specific_dynamic_filter_routes() {
    let mut builder = QueryPlanBuilder::new();
    let sales_for_date_join = builder.add_source(estimated_local_source_with_stable_id_and_file_indices(
        "web_sales",
        "web_sales",
        &["ws_sold_date_sk", "ws_item_sk"],
        &[0, 1],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let sales_for_other_branch = builder.add_source(estimated_local_source_with_stable_id_and_file_indices(
        "web_sales",
        "web_sales",
        &["ws_item_sk"],
        &[1],
        128 * 1024 * 1024,
        10_000_000,
    ));
    let date_dim = builder.add_source(estimated_local_source("date_dim", "d_date_sk", 16 * 1024 * 1024, 100_000));
    let branch_schema = schema(&["date_sk"]);
    let sales_dates = builder.add_projection(
        sales_for_date_join,
        vec![ProjectionOutput::new(Expr::column("ws_sold_date_sk"), Field::new("date_sk", DataType::Int32, false))],
    );
    let date_join = builder.add_join(
        sales_dates,
        date_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let other_branch = builder.add_projection(
        sales_for_other_branch,
        vec![ProjectionOutput::new(
            Expr::column("ws_item_sk").alias("date_sk"),
            Field::new("date_sk", DataType::Int32, false),
        )],
    );
    let union = builder.add_union(vec![date_join, other_branch], branch_schema);
    let plan = builder.finish_gpu_resident(union);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::DynamicFilterConsumerConflict)
    );
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "web_sales"), 2);
    assert!(result.plan.dynamic_filter_routes().iter().any(|route| {
        route.consumer_columns == vec!["ws_sold_date_sk".to_string()]
            && matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == sales_for_date_join)
    }));
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
}

#[test]
fn native_optimizer_rejects_count_star_probe_source_reuse() {
    let plan = count_star_probe_source_reuse_plan();
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
    assert_eq!(count_star_probe_source_count(&result.plan), 2);
}

#[test]
fn native_optimizer_rejects_position_delete_source_reuse_contract() {
    let plan = position_delete_source_reuse_plan();
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
}

#[test]
fn native_optimizer_rejects_field_id_output_schema_override_source_reuse_contract() {
    let plan = field_id_output_schema_override_source_reuse_plan();
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
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
        Some(nexus_query_engine::optimizer::LocalParquetSourceReuseRejectedReason::VariantTooLarge)
    );
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 2);
}

#[test]
fn native_optimizer_uses_full_file_size_for_source_reuse_variant_size_evidence() {
    let plan = file_size_only_superset_with_grouped_aggregate_source_variant_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions {
            max_variant_selected_bytes: 1024 * 1024 * 1024,
            ..LocalParquetSourceReuseOptions::default()
        },
        ..NativeOptimizerOptions::disabled()
    };
    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert!(diagnostics.candidates[0].applied);
    assert_eq!(diagnostics.candidates[0].rejected_reason, None);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
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

fn compatible_small_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let SourceNode::LocalParquet(full_source) = customer_address_source(&["id", "state"], &[0, 1], None) else {
        unreachable!("customer address source helper always builds local Parquet")
    };
    let full = builder.add_source(SourceNode::LocalParquet(
        full_source.with_unique_key_columns(vec![BTreeSet::from(["id".to_string()])]),
    ));
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

fn contract_mismatched_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let direct = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        remote_direct_customer_address_contract(),
    )));
    let cache_policy = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        cache_policy_customer_address_contract(),
    )));
    let direct_id = builder.add_projection(
        direct,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let cache_policy_id = builder.add_projection(
        cache_policy,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![direct_id, cache_policy_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn count_star_probe_source_reuse_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(count_star_probe_customer_address_source("id", 0));
    let right = builder.add_source(count_star_probe_customer_address_source("zip", 2));
    let output_schema = Arc::new(Schema::new(vec![Field::new(ZERO_COLUMN_COUNT_COLUMN, DataType::Int64, false)]));
    let union = builder.add_union(vec![left, right], output_schema);
    builder.finish_gpu_resident(union)
}

fn position_delete_source_reuse_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(
        contract_sensitive_customer_address_source(remote_direct_customer_address_contract())
            .with_position_delete_file("delete-a", "s3://warehouse/customer_address-delete-a.parquet"),
    ));
    let right = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        remote_direct_customer_address_contract(),
    )));
    let left_id = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let right_id = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left_id, right_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn field_id_output_schema_override_source_reuse_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(field_id_output_schema_override_source(&[("old_id", "id")]));
    let right = builder.add_source(field_id_output_schema_override_source(&[("old_id", "id"), ("old_state", "state")]));
    let left_id = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let right_id = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left_id, right_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_sensitive_stable_id_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn decimal_contract_mismatched_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn compatible_local_parquet_source_variants_with_different_file_order_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn compatible_small_local_parquet_source_variants_with_max_rows_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn conflicting_file_index_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn conflicting_column_name_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn graph_bfs_local_source_reuse_plan() -> nexus_query_engine::plan::QueryPlan {
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
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
        path_output: None,
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

fn large_local_parquet_source_variants_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn large_superset_with_smaller_local_parquet_source_variant_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn large_superset_with_grouped_aggregate_source_variant_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn file_size_only_superset_with_grouped_aggregate_source_variant_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(file_size_only_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity", "inv_price"],
        600 * 1024 * 1024,
    ));
    let aggregate_input = builder.add_source(file_size_only_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity"],
        600 * 1024 * 1024,
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

fn file_size_only_local_source_with_stable_id(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    file_size_bytes: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(stable_id)
        .with_file_size_bytes(file_size_bytes);
    let read_columns = field_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    let file_indices = (0..field_names.len()).collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(vec![file], source_schema, read_columns).with_column_file_indices(
        file_indices.clone(),
        file_indices,
        Vec::new(),
    );
    SourceNode::LocalParquet(source)
}

fn large_grouped_aggregate_superset_with_smaller_source_variant_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn large_filtered_superset_with_smaller_local_parquet_source_variant_plan() -> nexus_query_engine::plan::QueryPlan {
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

fn field_id_output_schema_override_source(field_names: &[(&str, &str)]) -> SourceNode {
    let physical_names = field_names.iter().map(|(physical, _logical)| *physical).collect::<Vec<_>>();
    let logical_names = field_names.iter().map(|(_physical, logical)| *logical).collect::<Vec<_>>();
    let source_schema = schema(&physical_names);
    let file = LocalParquetFile::new("/tmp/customer_status.parquet")
        .with_stable_id("customer_status")
        .with_row_group_bytes(vec![2 * 1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    let read_columns = physical_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    let mut source = LocalParquetSource::from_files(vec![file], source_schema, read_columns);
    source.files[0].field_id_read_bindings = field_names
        .iter()
        .enumerate()
        .map(|(index, (physical, logical))| {
            ParquetFieldIdReadBinding::new(index as i32 + 1, *logical, *physical, index, false, DataType::Int32)
        })
        .collect();
    source = source.with_output_schema_override(schema(&logical_names));
    SourceNode::LocalParquet(source)
}

trait LocalParquetSourceTestExt {
    fn with_reader_decimal_contract(
        self,
        column_name: &str,
        type_id: ParquetReaderDecimalTypeId,
        cudf_scale: i32,
    ) -> Self;
    fn with_position_delete_file(self, id: &str, location: &str) -> Self;
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

    fn with_position_delete_file(mut self, id: &str, location: &str) -> Self {
        let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
            id,
            ResolvedParquetSource::remote_kvikio(location, CredentialMode::SharedProcessEnv),
        )
        .unwrap();
        self.files[0].position_delete_files = vec![delete_file];
        self.refresh_diagnostics();
        self
    }
}

fn remote_direct_customer_address_contract() -> ResolvedParquetSource {
    ResolvedParquetSource::remote_kvikio(
        "s3://warehouse/customer_address.parquet?query_variant=a",
        CredentialMode::SharedProcessEnv,
    )
    .with_storage_profile_id("remote-profile-a")
    .with_storage_domain("warehouse@s3")
    .with_content_identity("snapshot=42,file=customer_address,etag=abc")
}

fn cache_policy_customer_address_contract() -> ResolvedParquetSource {
    ResolvedParquetSource::remote_kvikio(
        "s3://warehouse/customer_address.parquet?query_variant=a",
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

fn count_star_probe_customer_address_source(column: &str, file_index: usize) -> SourceNode {
    let schema = Arc::new(Schema::new(vec![Field::new(column, DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/customer_address.parquet").with_stable_id("customer_address");
    let source = LocalParquetSource::count_star_probe(vec![file], schema, vec![column.to_string()])
        .with_column_roles(Vec::new(), vec![column.to_string()])
        .with_column_file_indices(vec![file_index], Vec::new(), vec![file_index]);
    SourceNode::LocalParquet(source)
}

fn count_star_probe_source_count(plan: &nexus_query_engine::plan::QueryPlan) -> usize {
    plan.nodes()
        .iter()
        .filter(|node| match &node.kind {
            PlanNode::Source(source) => {
                matches!(source.as_ref(), SourceNode::LocalParquet(source) if source.is_count_star_probe_source())
            },
            _ => false,
        })
        .count()
}

fn gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
}

fn local_parquet_source_count_for_stable_id(plan: &nexus_query_engine::plan::QueryPlan, stable_id: &str) -> usize {
    local_parquet_sources_for_stable_id(plan, stable_id).len()
}

fn local_parquet_sources_for_stable_id<'a>(
    plan: &'a nexus_query_engine::plan::QueryPlan,
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

fn assert_source_input(
    plan: &nexus_query_engine::plan::QueryPlan,
    node_id: nexus_query_engine::plan::NodeId,
    name: &str,
) {
    let node = plan.node(node_id).expect("node should exist");
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(source) => assert_eq!(source.input_name, name),
            other => panic!("expected GPU-resident source {name}, got {other:?}"),
        },
        other => panic!("expected source {name}, got {other:?}"),
    }
}
