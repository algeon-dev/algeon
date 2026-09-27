#[test]
fn repeated_filtered_source_becomes_one_resident_producer() {
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
        Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::Literal(LiteralValue::Utf8("a".to_string()))),
        Arc::clone(&schema),
    );
    let output = Arc::new(Schema::new(vec![
        Field::new("left_name", DataType::Utf8, false),
        Field::new("right_name", DataType::Utf8, false),
    ]));
    let join = builder.add_join(
        filtered,
        filtered,
        JoinSpec::new(JoinType::Inner, vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))])
            .with_output_projection(vec![
                JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("name"), "left_name"),
                JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("name"), "right_name"),
            ]),
        Arc::clone(&output),
    );
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
    );
    let plan = builder.finish_with_host_sink_schema(outer, output);
    let considered_count = plan.node_count();
    let routed = plan.clone().with_dynamic_filter_routes(vec![DynamicFilterRoute::new(
        DynamicFilterId::new(0),
        DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id: join },
        DynamicFilterConsumer::LocalParquetSource { source_node_id: source },
        vec!["id".to_string()],
        vec!["id".to_string()],
        vec![DynamicFilterKind::Range],
        DynamicFilterConsumerLowering::ReaderPredicate,
    )]);
    let routed_split = optimize_shared_resident_extraction(routed);
    assert!(routed_split.producers.is_empty());
    let routed_candidate = routed_split
        .diagnostics
        .candidates
        .iter()
        .find(|candidate| candidate.node_id == filtered)
        .expect("the repeated routed subplan must retain one decision");
    assert_eq!(routed_candidate.decision, SharedResidentExtractionDecision::Skipped);
    assert_eq!(routed_candidate.reason, SharedResidentExtractionReason::DynamicFilterRoutesUnremappable);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::disabled(), rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());
    let split = SharedResidentPlan {
        consumer: optimized.plan,
        producers: optimized.shared_resident_producers,
        diagnostics: optimized.diagnostics.shared_resident_extraction,
    };

    assert_eq!(split.producers.len(), 1);
    assert_eq!(split.diagnostics.considered_count, considered_count);
    assert_eq!(split.diagnostics.extracted_count, 1);
    let extracted = split
        .diagnostics
        .candidates
        .iter()
        .find(|candidate| candidate.decision == SharedResidentExtractionDecision::Extracted)
        .expect("one repeated closed subplan must be extracted");
    assert_eq!(extracted.reason, SharedResidentExtractionReason::Extracted);
    assert_eq!(extracted.consumer_occurrences.len(), 2);
    assert_eq!(extracted.resident_input_name.as_deref(), Some(split.producers[0].input_name.as_str()));
    assert_eq!(extracted.subplan_fingerprint, split.producers[0].plan.execution_fingerprint());
    assert_eq!(split.producers[0].plan.root_node().output_schema(), &schema);
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
