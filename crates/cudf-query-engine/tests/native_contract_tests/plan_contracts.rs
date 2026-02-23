use super::*;

#[test]
fn native_plan_can_represent_proven_benchmark_shape() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("x00", DataType::Float64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("sum_x00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_file_size_bytes(3_235_196_281)],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, true),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(100), Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(limit);

    assert_eq!(plan.node_count(), 6);
    assert_eq!(plan.output_schema().fields().len(), 2);
    assert_eq!(plan.root_id(), limit);
    assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == limit));
    assert!(plan.sink().materializes_host_boundary());
    assert!(plan.nodes().iter().all(|node| node.facts.placement == DataPlacement::Device));
    assert_eq!(plan.node(source).unwrap().facts.schema_behavior, SchemaBehavior::SourceOutput);
    assert_eq!(plan.node(filter).unwrap().facts.schema_behavior, SchemaBehavior::PreservesInput);
    assert_eq!(plan.node(projection).unwrap().facts.schema_behavior, SchemaBehavior::Projection);
    assert_eq!(plan.node(aggregate).unwrap().facts.schema_behavior, SchemaBehavior::Aggregate);
    assert_eq!(plan.node(sort).unwrap().facts.output_ordering.len(), 2);
    assert_eq!(plan.node(limit).unwrap().facts.output_ordering.len(), 2);
    assert!(matches!(plan.root_node().kind, PlanNode::Limit(_)));
    assert!(plan.to_string().contains("LocalParquetSource"));
    assert!(plan.to_string().contains("Limit input=n4 skip=0 fetch=100"));
    assert!(plan.to_string().contains("Sink id=sink input=n5 mode=host_arrow materialization=device_to_host"));
}

#[test]
fn native_plan_can_represent_offset_only_limit() {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
    let limit = builder.add_limit(source, LimitSpec::offset(2), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(limit);

    assert!(
        matches!(&plan.root_node().kind, PlanNode::Limit(node) if node.spec.skip == 2 && node.spec.fetch.is_none())
    );
    assert!(plan.to_string().contains("LiteralBatchSource batches=1 rows=3 schema_fields=1"));
    assert!(plan.to_string().contains("Limit input=n0 skip=2 fetch=none"));
    assert!(
        plan.capability_analysis().nodes.iter().all(|node| matches!(node.executable, CapabilityDecision::Satisfied))
    );
}

#[test]
fn native_plan_can_represent_literal_batch_source() {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![7, 8]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
    let plan = builder.finish_with_host_sink(source);

    assert_eq!(plan.output_schema().as_ref(), schema.as_ref());
    assert!(matches!(
        &plan.root_node().kind,
        PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LiteralBatch(source) if source.row_count() == 2)
    ));
    assert_eq!(
        plan.capability_analysis().nodes[0].operator,
        NodeOperator::Source { format: SourceFormatCapability::LiteralBatch }
    );
}

#[test]
fn projection_plan_facts_schema_is_derived_from_typed_outputs() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let output_field = Field::new("payload_copy", DataType::Int64, false);

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
    let projection = builder.add_projection(
        source,
        vec![ProjectionOutput::new(Expr::column("payload").alias("payload_copy"), output_field.clone())],
    );
    let plan = builder.finish_gpu_resident(projection);
    let node = plan.node(projection).unwrap();
    let PlanNode::Projection(projection_node) = &node.kind else {
        panic!("expected projection node");
    };

    assert_eq!(projection_node.outputs.len(), 1);
    assert_eq!(projection_node.outputs[0].field, output_field);
    assert_eq!(projection_node.output_schema().as_ref(), node.facts.output_schema.as_ref());
    assert_eq!(plan.output_schema().as_ref(), projection_node.output_schema().as_ref());
}

#[test]
fn native_plan_can_represent_ordered_union() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);

    assert_eq!(plan.node_count(), 3);
    assert_eq!(plan.output_schema().as_ref(), schema.as_ref());
    assert_eq!(plan.root_id(), union);
    assert_eq!(plan.node(union).unwrap().facts.schema_behavior, SchemaBehavior::Union);
    assert!(matches!(plan.root_node().kind, PlanNode::Union(_)));
    assert!(plan.to_string().contains("Union inputs=n0,n1 input_count=2"));
}

#[test]
#[should_panic(expected = "query plan union node must have at least one input")]
fn native_plan_builder_rejects_empty_union() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    builder.add_union(Vec::new(), schema);
}

#[test]
fn native_plan_can_represent_gpu_resident_fragment_input() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("sum_x00", DataType::Int64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "partial_aggregate_input",
        Arc::clone(&input_schema),
    )));
    let projection = builder.add_projection(source, projection_outputs(vec![Expr::column("sum_x00")], output_schema));
    let plan = builder.finish_with_host_sink(projection);

    assert_eq!(plan.node_count(), 2);
    assert_eq!(plan.output_schema().fields().len(), 1);
    assert_eq!(plan.node(source).unwrap().facts.placement, DataPlacement::Device);
    assert_eq!(plan.node(source).unwrap().facts.schema_behavior, SchemaBehavior::SourceOutput);
    assert!(matches!(plan.node(source).unwrap().kind, PlanNode::Source(_)));
    assert!(plan.to_string().contains("GpuResidentSource input_name=partial_aggregate_input"));
    assert!(plan.to_string().contains("Sink id=sink input=n1 mode=host_arrow materialization=device_to_host"));
}

#[test]
fn native_plan_graph_represents_binary_join_with_same_named_inputs() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_id", DataType::Int32, false),
    ]));
    let left_source = LocalParquetSource::new("/tmp/left.parquet", Arc::clone(&left_schema), vec!["id".into()]);
    let right_source = LocalParquetSource::new("/tmp/right.parquet", Arc::clone(&right_schema), vec!["id".into()]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "id", "right_id"), "right_id"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let PlanNode::Join(join_node) = &plan.root_node().kind else {
        panic!("root node should be a native join contract");
    };
    assert_eq!(plan.root_node().input_ids(), &[left, right]);
    assert_eq!(join_node.spec.keys[0].left.name(), "id");
    assert_eq!(join_node.spec.keys[0].left.display_name(), "left_id");
    assert_eq!(join_node.spec.keys[0].right.name(), "id");
    assert_eq!(join_node.spec.keys[0].right.display_name(), "right_id");
    assert_eq!(plan.root_node().facts.schema_behavior, SchemaBehavior::Join);
    assert!(plan.to_string().contains("Join left=n0 right=n1 type=inner key_count=1"));
}

#[test]
fn native_query_validation_rejects_unreachable_dag_nodes() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let root = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    )));
    let _unreachable = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/unreachable.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    )));
    let plan = builder.finish_with_host_sink(root);
    let unreachable = plan.unreachable_node_ids().iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(unreachable, vec!["n1"]);
}
