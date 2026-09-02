use std::sync::Arc;

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
        vec![LocalParquetFile::new("/tmp/source-contract.parquet").with_file_size_bytes(3_235_196_281)],
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
fn aggregate_strategy_capability_selects_encoded_key_partial_after_contract_proof() {
    let input_schema = Schema::new(vec![
        Field::new("returnflag", DataType::Utf8, true),
        Field::new("linestatus", DataType::Utf8, true),
        Field::new("quantity", DataType::Int64, false),
    ]);
    let aggregate = AggregateSpec::new(
        vec![Expr::column("returnflag"), Expr::column("linestatus")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("quantity"), "sum_qty")],
    );

    let decision = aggregate_strategy_capability_decision(&aggregate, &input_schema, AggregateExecutionStage::Partial);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::EncodedKeyPartial);
    assert_eq!(decision.facts.get("selected_strategy").map(String::as_str), Some("encoded_key_partial"));
    assert_eq!(decision.facts.get("execution_stage").map(String::as_str), Some("partial"));
    assert_eq!(decision.facts.get("group_key_dtypes").map(String::as_str), Some("Utf8,Utf8"));
    assert_eq!(decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str), Some("none"));
    assert_eq!(decision.facts.get("encoded_key_partial_contract_status").map(String::as_str), Some("available"));
    assert_eq!(
        decision.facts.get("partitioned_merge_rejected_reason").map(String::as_str),
        Some("partitioned_merge_requires_final_merge_stage")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_runtime_cardinality_evidence")
    );
    assert_eq!(decision.facts.get("prepared_sort_groupby_contract_status").map(String::as_str), Some("available"));
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("pending_runtime_evidence")
    );
    assert_eq!(
        decision.rejected_candidate_summary(),
        "prepared_sort_groupby:prepared_sort_groupby_requires_runtime_cardinality_evidence;partitioned_merge:partitioned_merge_requires_final_merge_stage"
    );
}

#[test]
fn aggregate_strategy_capability_selects_partitioned_merge_for_final_merge_stage() {
    let input_schema = Schema::new(vec![
        Field::new("returnflag", DataType::Utf8, true),
        Field::new("linestatus", DataType::Utf8, true),
        Field::new("sum_qty", DataType::Int64, true),
    ]);
    let aggregate = AggregateSpec::new(
        vec![Expr::column("returnflag"), Expr::column("linestatus")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("quantity"), "sum_qty")],
    );

    let decision =
        aggregate_strategy_capability_decision(&aggregate, &input_schema, AggregateExecutionStage::FinalMerge);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::PartitionedMerge);
    assert_eq!(decision.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));
    assert_eq!(decision.facts.get("execution_stage").map(String::as_str), Some("final_merge"));
    assert_eq!(decision.facts.get("partitioned_merge_rejected_reason").map(String::as_str), Some("none"));
    assert_eq!(decision.facts.get("partitioned_merge_contract_status").map(String::as_str), Some("available"));
    assert_eq!(
        decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_requires_partial_stage")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_partial_or_single_stage")
    );
    assert_eq!(
        decision.rejected_candidate_summary(),
        "encoded_key_partial:encoded_key_partial_requires_partial_stage;prepared_sort_groupby:prepared_sort_groupby_requires_partial_or_single_stage"
    );
}

#[test]
fn aggregate_strategy_capability_analysis_records_grouped_aggregate_without_node_rejection() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("returnflag", DataType::Utf8, true),
        Field::new("quantity", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("returnflag", DataType::Utf8, true),
        Field::new("sum_qty", DataType::Int64, true),
    ]));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("returnflag")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("quantity"), "sum_qty")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("lineitem", Arc::clone(&input_schema))));
    let aggregate_id = builder.add_aggregate(source, aggregate, aggregate_schema);
    let plan = builder.finish_gpu_resident(aggregate_id);
    let analysis = plan.capability_analysis();

    let node = analysis.nodes.iter().find(|node| node.node_id == aggregate_id).expect("aggregate node capability");
    assert_eq!(node.executable, CapabilityDecision::Satisfied);
    assert_eq!(analysis.aggregate_strategies.len(), 1);
    let strategy = &analysis.aggregate_strategies[0];
    assert_eq!(strategy.node_id, aggregate_id);
    assert_eq!(strategy.stage, AggregateExecutionStage::Partial);
    assert_eq!(strategy.decision.selected_strategy, AggregateExecutionStrategy::EncodedKeyPartial);
    assert_eq!(strategy.decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str), Some("none"));
    assert_eq!(
        strategy.decision.facts.get("encoded_key_partial_contract_status").map(String::as_str),
        Some("available")
    );
    assert_eq!(
        strategy.decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_runtime_cardinality_evidence")
    );

    let report = CapabilityContractReport::from_capability_analysis(&analysis)
        .expect("modeled aggregate capability should produce a admission contract report");
    assert!(report.gate_rows().iter().any(|row| {
        row.required_contract == "aggregate_contract" && row.contract_status == ContractStatus::Modeled
    }));
    assert!(
        report.gate_rows().iter().all(|row| row.contract_status != ContractStatus::RejectedByCapability),
        "optimized strategy candidate rejections are local strategy facts, not bounded capability failures"
    );
}

#[test]
fn aggregate_strategy_capability_rejects_encoded_key_for_non_string_keys_locally() {
    let input_schema =
        Schema::new(vec![Field::new("key", DataType::Int32, false), Field::new("value", DataType::Int64, false)]);
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );

    let decision = aggregate_strategy_capability_decision(&aggregate, &input_schema, AggregateExecutionStage::Partial);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_requires_string_group_key")
    );
    assert_eq!(
        decision.facts.get("rejected_candidate_strategies").map(String::as_str),
        Some(
            "encoded_key_partial:encoded_key_partial_requires_string_group_key;prepared_sort_groupby:prepared_sort_groupby_requires_runtime_cardinality_evidence;partitioned_merge:partitioned_merge_requires_final_merge_stage"
        )
    );
}

#[test]
fn aggregate_strategy_capability_rejects_encoded_key_for_filter_and_distinct_locally() {
    let input_schema = Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
        Field::new("include", DataType::Boolean, true),
    ]);
    let filtered = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")
                .with_filter(Some(Expr::column("include"))),
        ],
    );
    let distinct = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "distinct_values").with_distinct(true),
        ],
    );

    let filtered_decision =
        aggregate_strategy_capability_decision(&filtered, &input_schema, AggregateExecutionStage::Partial);
    let distinct_decision =
        aggregate_strategy_capability_decision(&distinct, &input_schema, AggregateExecutionStage::Partial);

    assert_eq!(filtered_decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        filtered_decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_aggregate_filter_unsupported")
    );
    assert_eq!(distinct_decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        distinct_decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_distinct_aggregate_unsupported")
    );
}

#[test]
fn aggregate_strategy_capability_treats_median_as_single_stage_only() {
    let input_schema =
        Schema::new(vec![Field::new("key", DataType::Utf8, false), Field::new("value", DataType::Int64, false)]);
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Median, Expr::column("value"), "median_value")],
    );

    let partial = aggregate_strategy_capability_decision(&aggregate, &input_schema, AggregateExecutionStage::Partial);
    assert_eq!(partial.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        partial.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_median_unsupported")
    );
    assert_eq!(
        partial.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_median_requires_single_stage")
    );
    assert_eq!(
        partial.facts.get("partitioned_merge_rejected_reason").map(String::as_str),
        Some("partitioned_merge_requires_final_merge_stage")
    );

    let final_merge =
        aggregate_strategy_capability_decision(&aggregate, &input_schema, AggregateExecutionStage::FinalMerge);
    assert_eq!(final_merge.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        final_merge.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_requires_partial_stage")
    );
    assert_eq!(
        final_merge.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_partial_or_single_stage")
    );
    assert_eq!(
        final_merge.facts.get("partitioned_merge_rejected_reason").map(String::as_str),
        Some("partitioned_merge_median_unsupported")
    );
}

#[test]
fn aggregate_node_capability_rejects_mixed_median_and_floating_statistical_functions() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float64, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("stdev_value", DataType::Float64, true),
        Field::new("median_value", DataType::Float64, true),
    ]));
    let aggregate = AggregateSpec::new(
        vec![],
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("value"), "stdev_value"),
            AggregateExpr::new(AggregateFunction::Median, Expr::column("value"), "median_value"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
    let aggregate_id = builder.add_aggregate(source, aggregate, output_schema);
    let plan = builder.finish_gpu_resident(aggregate_id);
    let analysis = plan.capability_analysis();

    let node = analysis.nodes.iter().find(|node| node.node_id == aggregate_id).expect("aggregate node capability");
    let CapabilityDecision::Rejected { reason, facts } = &node.executable else {
        panic!("mixed median/statistical aggregate should reject, got {:?}", node.executable);
    };
    assert_eq!(*reason, "unsupported_aggregate_expression_capability");
    assert_eq!(facts.get("reason").map(String::as_str), Some("median_floating_statistical_mixed_unsupported"));
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("aggregate"));
}

#[test]
fn aggregate_node_capability_rejects_decimal_median_input_type() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 2), false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("median_amount", DataType::Decimal128(15, 2), true)]));
    let aggregate = AggregateSpec::new(
        vec![],
        vec![AggregateExpr::new(AggregateFunction::Median, Expr::column("amount"), "median_amount")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
    let aggregate_id = builder.add_aggregate(source, aggregate, output_schema);
    let plan = builder.finish_gpu_resident(aggregate_id);
    let analysis = plan.capability_analysis();

    let node = analysis.nodes.iter().find(|node| node.node_id == aggregate_id).expect("aggregate node capability");
    let CapabilityDecision::Rejected { reason, facts } = &node.executable else {
        panic!("Decimal median aggregate should reject, got {:?}", node.executable);
    };
    assert_eq!(*reason, "unsupported_aggregate_expression_capability");
    assert_eq!(facts.get("reason").map(String::as_str), Some("unsupported_median_input_type"));
    assert_eq!(facts.get("data_type").map(String::as_str), Some("Decimal128(15, 2)"));
}

#[test]
fn aggregate_strategy_capability_admits_single_stage_prepared_sort_groupby() {
    let input_schema =
        Schema::new(vec![Field::new("key", DataType::Utf8, false), Field::new("value", DataType::Int64, false)]);
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );

    let decision = aggregate_strategy_capability_decision(&aggregate, &input_schema, AggregateExecutionStage::Single);

    assert_eq!(decision.selected_strategy, AggregateExecutionStrategy::LibcudfGroupby);
    assert_eq!(
        decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_requires_partial_stage")
    );
    assert_eq!(
        decision.facts.get("partitioned_merge_rejected_reason").map(String::as_str),
        Some("partitioned_merge_requires_final_merge_stage")
    );
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str),
        Some("prepared_sort_groupby_requires_runtime_cardinality_evidence")
    );
    assert_eq!(decision.facts.get("prepared_sort_groupby_contract_status").map(String::as_str), Some("available"));
    assert_eq!(
        decision.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("pending_runtime_evidence")
    );
}

#[test]
fn aggregate_strategy_capability_rejects_partitioned_merge_for_unsupported_final_merge_shapes() {
    let input_schema =
        Schema::new(vec![Field::new("key", DataType::Utf8, false), Field::new("value", DataType::Int64, false)]);
    let scalar = AggregateSpec::new(
        vec![],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );
    let distinct = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "distinct_values").with_distinct(true),
        ],
    );
    let stddev = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("value"), "stddev_value")],
    );

    for (spec, reason) in [
        (&scalar, "partitioned_merge_requires_grouped_aggregate"),
        (&distinct, "partitioned_merge_distinct_aggregate_unsupported"),
        (&stddev, "partitioned_merge_floating_statistical_aggregate_unsupported"),
    ] {
        let decision = aggregate_strategy_capability_decision(spec, &input_schema, AggregateExecutionStage::FinalMerge);
        assert_ne!(decision.selected_strategy, AggregateExecutionStrategy::PartitionedMerge);
        assert_eq!(decision.facts.get("partitioned_merge_rejected_reason").map(String::as_str), Some(reason));
        assert_eq!(
            decision.facts.get("partitioned_merge_contract_status").map(String::as_str),
            Some("blocked_by_shape_or_stage")
        );
    }
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
fn regexp_replace_projection_declares_bounded_output_contract() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("referer", DataType::Utf8, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("domain", DataType::Utf8, true)]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("hits", Arc::clone(&input_schema))));
    let projection = builder.add_projection(
        source,
        projection_outputs(
            vec![
                Expr::regexp_replace(
                    Expr::column("referer"),
                    "^https?://(?:www\\.)?([^/]+)/.*$",
                    "\\1",
                    DataType::Utf8,
                )
                .alias("domain"),
            ],
            output_schema,
        ),
    );
    let plan = builder.finish_gpu_resident(projection);
    let analysis = plan.capability_analysis();
    let node = analysis.nodes.iter().find(|node| node.node_id == projection).expect("projection node capability");
    assert_eq!(node.executable, CapabilityDecision::Satisfied);
    assert_eq!(node.local_requirements.len(), 1);
    let requirement = &node.local_requirements[0];
    assert_eq!(requirement.kind, ReservationKind::OperatorOutput);
    assert_eq!(requirement.memory_boundary, "regexp_replace_expression_output");
    assert_eq!(requirement.required_contract, "regexp_replace_output_size_preflight");
    assert_eq!(requirement.quality, ContractQuality::ExactPreflight);
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
fn with_host_sink_schema_attaches_metadata_above_frontend_named_sink() {
    // The executable root may run on synthetic unique names while the host
    // sink restores the frontend-visible schema (including duplicate names).
    // Replacing the host sink schema attaches schema-level metadata without
    // constraining the executable root.
    let executable_schema = Arc::new(Schema::new(vec![
        Field::new("__nexus_native_field_0_id", DataType::Int32, false),
        Field::new("__nexus_native_field_1_id", DataType::Int32, false),
    ]));
    let frontend_schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("id", DataType::Int32, false)]));
    let source = LocalParquetSource::new(
        "/tmp/frontend-named.parquet",
        Arc::clone(&executable_schema),
        vec!["__nexus_native_field_0_id".into(), "__nexus_native_field_1_id".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink_schema(source, Arc::clone(&frontend_schema));

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("owner".to_string(), "frontend".to_string());
    let annotated_schema = Arc::new(Schema::new_with_metadata(frontend_schema.fields().clone(), metadata.clone()));
    let plan = plan.with_host_sink_schema(Arc::clone(&annotated_schema));

    let ResultSink::HostArrow { schema, .. } = plan.sink() else {
        panic!("host sink schema replacement requires a host Arrow sink");
    };
    assert_eq!(schema.as_ref(), annotated_schema.as_ref());
    assert_eq!(plan.root_node().output_schema().fields(), executable_schema.fields());
}

#[test]
#[should_panic(expected = "host sink schema replacement must preserve the existing host sink fields")]
fn with_host_sink_schema_rejects_field_shape_changes() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let renamed = Arc::new(Schema::new(vec![Field::new("renamed", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rename-reject.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    )));
    let plan = builder.finish_with_host_sink(source);
    let _ = plan.with_host_sink_schema(renamed);
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
