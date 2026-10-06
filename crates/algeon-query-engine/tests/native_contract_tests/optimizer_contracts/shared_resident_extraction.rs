/// A filtered Parquet scan joined with itself and with a resident input. The
/// returned ids are the repeated subplan root and the scan below it.
fn shared_resident_join_plan(repeated_aggregate: bool) -> (QueryPlan, NodeId, NodeId) {
    let schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false), Field::new("name", DataType::Utf8, false)]));
    let file = LocalParquetFile::new("/tmp/shared-resident-not-read.parquet")
        .with_file_size_bytes(64)
        .with_row_group_bytes(vec![64])
        .with_projected_row_group_bytes(vec![64], vec![64])
        .with_row_group_rows(vec![8]);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["id".to_string(), "name".to_string()],
    )));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::Literal(LiteralValue::Utf8("a".to_string())))).unwrap();
    let repeated = if repeated_aggregate {
        builder.add_aggregate(
            filtered,
            AggregateSpec::new(
                vec![Expr::column("id")],
                vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("name"), "name")],
            ),
            Arc::clone(&schema),
        ).unwrap()
    } else {
        filtered
    };
    let output = Arc::new(Schema::new(vec![
        Field::new("left_name", DataType::Utf8, false),
        Field::new("right_name", DataType::Utf8, false),
    ]));
    let join = builder.add_join(
        repeated,
        repeated,
        JoinSpec::new(JoinType::Inner, vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))])
            .with_output_projection(vec![
                JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("name"), "left_name"),
                JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("name"), "right_name"),
            ]),
        Arc::clone(&output),
    ).unwrap();
    let resident = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("existing-resident", Arc::clone(&schema)).with_row_count_upper_bound(8),
    ));
    let outer = builder.add_join(
        join,
        resident,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::unqualified("left_name"), ColumnRef::unqualified("name"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("left_name"), "left_name"),
            JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("right_name"), "right_name"),
        ]),
        Arc::clone(&output),
    ).unwrap();
    // The outer join's build side narrows the scan inside the repeated
    // subplan, which every consumer of a shared producer would then see.
    let plan = builder.finish_with_host_sink_schema(outer, output).unwrap().with_dynamic_filter_routes_for_test(vec![
        DynamicFilterRoute::new(
            DynamicFilterId::new(0),
            DynamicFilterProducer::HashJoinBuild { join_node_id: outer, build_side: JoinSide::Right, build_input: resident },
            DynamicFilterConsumer::LocalParquetSource { source_node_id: source },
            vec!["name".to_string()],
            vec!["name".to_string()],
            vec![DynamicFilterKind::Range],
            DynamicFilterConsumerLowering::ReaderPredicate,
        ),
    ]);
    (plan, repeated, source)
}

#[test]
fn repeated_aggregate_becomes_one_resident_producer_without_routes_into_it() {
    let (plan, aggregate, _) = shared_resident_join_plan(true);
    let considered_count = plan.node_count();

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::disabled(), rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver()).unwrap();
    let split = SharedResidentPlan {
        consumer: optimized.plan,
        producers: optimized.shared_resident_producers,
        diagnostics: optimized.diagnostics.shared_resident_extraction,
    };

    assert_eq!(split.producers.len(), 1);
    assert!(matches!(split.producers[0].plan.root_node().kind, PlanNode::Aggregate(_)));
    assert_eq!(split.diagnostics.considered_count, considered_count);
    assert_eq!(split.diagnostics.extracted_count, 1);
    let extracted = split
        .diagnostics
        .candidates
        .iter()
        .find(|candidate| candidate.decision == SharedResidentExtractionDecision::Extracted)
        .expect("one repeated closed subplan must be extracted");
    assert_eq!(extracted.node_id, aggregate);
    assert_eq!(extracted.attempt_fragment, None);
    assert_eq!(extracted.reason, SharedResidentExtractionReason::Extracted);
    assert_eq!(extracted.consumer_occurrences.len(), 2);
    assert_eq!(extracted.resident_input_name.as_deref(), Some(split.producers[0].input_name.as_str()));
    assert_eq!(extracted.subplan_fingerprint, split.producers[0].plan.execution_fingerprint());
    // A route from a consumer into the shared subplan would narrow the
    // result every other consumer reads, so neither plan keeps it.
    assert!(split.consumer.dynamic_filter_routes().is_empty());
    assert!(split.producers[0].plan.dynamic_filter_routes().is_empty());
    let resident_sources = split
        .consumer
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) if source.input_name.starts_with("__algeon_shared_resident_") => {
                    Some(source)
                },
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(resident_sources.len(), 1);
    assert_eq!(resident_sources[0].row_count_upper_bound, Some(8));
    assert_eq!(
        split.consumer.nodes().iter().find(|node| matches!(node.kind, PlanNode::Join(_))).unwrap().input_ids(),
        &[resident_sources_node_id(&split.consumer), resident_sources_node_id(&split.consumer)]
    );
}

#[test]
fn repeated_filtered_scan_is_recomputed_by_each_consumer() {
    let (plan, filtered, _) = shared_resident_join_plan(false);

    let optimized = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled(), rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver()).unwrap();

    assert!(optimized.shared_resident_producers.is_empty());
    assert_eq!(optimized.plan.nodes(), plan.nodes());
    let rejected = optimized
        .diagnostics
        .shared_resident_extraction
        .candidates
        .iter()
        .find(|candidate| candidate.node_id == filtered)
        .expect("the repeated filtered scan must retain one decision");
    assert_eq!(rejected.decision, SharedResidentExtractionDecision::Rejected);
    assert_eq!(rejected.reason, SharedResidentExtractionReason::RootDoesNotReduceRows);
    assert_eq!(optimized.diagnostics.shared_resident_extraction.rejected_count, 1);
}

fn resident_sources_node_id(plan: &QueryPlan) -> NodeId {
    plan.nodes()
        .iter()
        .find(|node| {
            matches!(
                &node.kind,
                PlanNode::Source(source)
                    if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name.starts_with("__algeon_shared_resident_"))
            )
        })
        .expect("consumer must contain the extracted resident source")
        .id
}
