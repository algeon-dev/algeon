#[test]
fn dynamic_filter_runtime_metrics_record_produced_skipped_and_applied_events() {
    let plan = optimize_query_plan(
        estimated_inner_join_plan(JoinSide::Right, 512 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000),
        &NativeOptimizerOptions::default(), rapids_analysis_context(), local_dimension_pruning_resolver())
    .plan;
    let route = &plan.dynamic_filter_routes()[0];
    let filter = DynamicFilter::SmallInListI64 { values: vec![10, 20, 30], min: 10, max: 30 };
    let mut metrics = QueryMetrics::default();

    metrics.record_dynamic_filter_produced(route, &filter, 3);
    metrics.record_dynamic_filter_applied(
        route,
        &filter,
        DynamicFilterSourceSnapshot { row_groups: 5, source_bytes: Some(8192) },
        DynamicFilterSourceSnapshot { row_groups: 2, source_bytes: Some(4096) },
    );
    metrics.record_dynamic_filter_reader_predicate_source_reduction(Some(1000), Some(250));
    metrics.record_dynamic_filter_post_decode_rows(route, &filter, Some(80), Some(25));
    metrics.record_dynamic_filter_skipped(route, "unsupported_producer_key_type");

    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert_eq!(metrics.dynamic_filter_skipped_count, 1);
    assert_eq!(metrics.dynamic_filter_row_groups_pruned, 3);
    assert_eq!(metrics.dynamic_filter_source_bytes_pruned, 4096);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_before, 1000);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_after, 250);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_filtered, 750);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_before, 80);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_after, 25);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_filtered, 55);
    assert_eq!(metrics.dynamic_filter_source_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_filter_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_join_probe_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_runtime_rows.len(), 4);
    assert!(metrics.dynamic_filter_runtime_rows[0].to_stable_kv_row().contains("event=produced"));
    assert!(metrics.dynamic_filter_runtime_rows[1].to_stable_kv_row().contains("event=applied"));
    assert!(metrics.dynamic_filter_runtime_rows[2].to_stable_kv_row().contains("event=post_decode_rows"));
    assert!(
        metrics.dynamic_filter_runtime_rows[0].to_stable_kv_row().contains("consumer_reduction_stage=source_reader")
    );
    assert!(
        metrics.dynamic_filter_runtime_rows[2]
            .to_stable_kv_row()
            .contains("post_decode_rows_before=80,post_decode_rows_after=25,post_decode_rows_filtered=55")
    );
    assert!(
        metrics.dynamic_filter_runtime_rows[3]
            .to_stable_kv_row()
            .contains("skipped_reason=unsupported_producer_key_type")
    );
}

#[test]
fn dynamic_filter_keeps_large_integer_keysets_as_hash_set_contains_payloads() {
    let filter = DynamicFilter::from_i64_values([Some(5), Some(1), None, Some(5), Some(3)], 2);

    assert_eq!(filter, DynamicFilter::HashSetContainsI64 { values: vec![1, 3, 5], min: 1, max: 5 });
    assert_eq!(filter.kind(), DynamicFilterKind::HashSetContains);
    assert_eq!(filter.value_count(), Some(3));
    assert_eq!(filter.i64_range(), Some((1, 5)));
}

#[test]
fn dynamic_filter_compacts_dense_integer_keysets_to_range_payloads() {
    let filter = DynamicFilter::from_i64_values([Some(3), Some(1), Some(2), Some(2), None], 2);

    assert_eq!(filter, DynamicFilter::RangeI64 { min: 1, max: 3 });
    assert_eq!(filter.kind(), DynamicFilterKind::Range);
    assert_eq!(filter.value_count(), None);
    assert_eq!(filter.i64_range(), Some((1, 3)));
}

#[test]
fn aggregate_group_key_dynamic_filter_uses_aggregate_output_column_before_projection_alias() {
    let mut builder = QueryPlanBuilder::new();
    let aggregate = aggregate_keyset_source(&mut builder, "returns", "raw_customer_sk");
    let projected = builder.add_projection(
        aggregate,
        vec![ProjectionOutput::new(
            Expr::column("raw_customer_sk"),
            Field::new("ctr_customer_sk", DataType::Int32, false),
        )],
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let join = builder.add_join(
        projected,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            "c_customer_sk",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["c_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let route = &result.plan.dynamic_filter_routes()[0];
    assert!(
        matches!(route.producer, DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } if aggregate_node_id == aggregate)
    );
    assert_eq!(route.producer_columns, vec!["raw_customer_sk".to_string()]);
    assert_eq!(route.consumer_columns, vec!["c_customer_sk".to_string()]);
}

#[test]
fn dynamic_filter_uses_hash_join_build_when_join_input_narrows_aggregate_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let returns = builder.add_source(gpu_source("returns", &["raw_customer_sk", "raw_store_sk"]));
    let aggregate = builder.add_aggregate(
        returns,
        AggregateSpec::new(vec![Expr::column("raw_customer_sk"), Expr::column("raw_store_sk")], Vec::new()),
        schema(&["raw_customer_sk", "raw_store_sk"]),
    );
    let projected = builder.add_projection(
        aggregate,
        vec![
            ProjectionOutput::new(
                Expr::column("raw_customer_sk"),
                Field::new("ctr_customer_sk", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("raw_store_sk"), Field::new("ctr_store_sk", DataType::Int32, false)),
        ],
    );
    let store = builder.add_source(estimated_local_source("store", "s_store_sk", 1024, 10));
    let narrowed = builder.add_join(
        store,
        projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "s_store_sk", "s_store_sk"),
                ColumnRef::input(1, "ctr_store_sk", "ctr_store_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "ctr_customer_sk", "ctr_customer_sk"),
            "ctr_customer_sk",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["ctr_customer_sk"]),
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let final_join = builder.add_join(
        narrowed,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            "c_customer_sk",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["c_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(final_join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    let route = result
        .plan
        .dynamic_filter_routes()
        .iter()
        .find(|route| route.consumer_columns == ["c_customer_sk".to_string()])
        .expect("final join should route a customer key filter");
    assert!(matches!(
        route.producer,
        DynamicFilterProducer::HashJoinBuild {
            join_node_id,
            build_side: JoinSide::Left,
            ..
        } if join_node_id == final_join
    ));
    assert_eq!(route.producer_columns, vec!["ctr_customer_sk".to_string()]);
}
