use super::*;

pub(super) fn assert_rejection_reason_is_family_free(reason: &str, label: &str) {
    for token in ["envelope", "_shape", "family", "topology", "recognizer", "catalogue", "whitelist"] {
        assert!(
            !reason.contains(token),
            "{label}: rejection reason {reason:?} must name the operator or edge contract, not a plan family"
        );
    }
}

#[test]
fn bounded_preflight_rejection_reports_caller_visible_sink() {
    // A rejected join plan reports the caller-visible sink as a diagnostic
    // fact only; the fact never selects a second decision.
    let plan = compositional_join_row_number_plan(false);
    let error = admit_query_plan(&plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())
        .expect_err("a row number over unordered join output must not preflight");
    assert_eq!(error.fact_value("caller_visible_sink"), Some("host_arrow"));
}

#[test]
fn bounded_compiled_pipeline_rejects_duplicate_fused_topk_limit_claims() {
    // `applied_limit` is set by exactly one producer and dropped by exactly
    // one limit pass-through: two sorts claiming `FusedTopK` for the same
    // limit would apply fetch/skip twice.
    let mut builder = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut builder);
    let inner_sort =
        builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)])).unwrap();
    let inner_limit = builder.add_limit(inner_sort, LimitSpec::fetch(10)).unwrap();
    let outer_sort = builder
        .add_sort(inner_limit, SortSpec::new(vec![SortExpr::new(Expr::column("left_id"), true, false)]))
        .unwrap();
    let outer_limit = builder.add_limit(outer_sort, LimitSpec::fetch(1)).unwrap();
    let plan = builder.finish_with_host_sink(outer_limit).unwrap();

    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("stacked sort limits should compile");
    pipeline.validate().expect("compiler output satisfies the fused top-k binding");

    let sort_task = |node| {
        pipeline
            .tasks()
            .iter()
            .find(|task| task.kind() == PipelineTaskKind::FinalSort && task.plan_node() == Some(node))
            .expect("each sort compiles to a final sort task")
    };
    let (inner_sort_task, outer_sort_task) = (sort_task(inner_sort), sort_task(outer_sort));
    let inner_claim = inner_sort_task.sort_strategy().expect("sort feeding a single limit fuses");
    let SortStrategy::FusedTopK { limit_task, limit_node } = inner_claim else {
        panic!("sort feeding a single limit must fuse, got {inner_claim:?}")
    };
    // Point the outer sort's claim at the inner limit. The structure is
    // unchanged, so only the uniqueness clause can reject the rebuilt pipeline.
    let mut tasks = pipeline.tasks().to_vec();
    let outer_index = tasks.iter().position(|task| task.id() == outer_sort_task.id()).expect("outer sort task exists");
    tasks[outer_index] = PipelineTask::new(outer_sort_task.id(), TaskPhysical::FinalSort(inner_claim))
        .with_input_resources(outer_sort_task.input_resources().to_vec())
        .with_output_resources(outer_sort_task.output_resources().to_vec());
    let rebuilt = CompiledPipeline::from_parts_unvalidated(
        tasks,
        pipeline.edges().to_vec(),
        pipeline.barriers().to_vec(),
        pipeline.resources().to_vec(),
        pipeline.sink().clone(),
    );
    let error = rebuilt.validate().expect_err("two producers must not claim one fused limit");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("first_sort_task"), Some(inner_sort_task.id().to_string()).as_deref());
    assert_eq!(error.fact_value("second_sort_task"), Some(outer_sort_task.id().to_string()).as_deref());
    assert_eq!(error.fact_value("limit_task"), Some(limit_task.to_string()).as_deref());
    assert_eq!(error.fact_value("limit_node"), Some(limit_node.to_string()).as_deref());
}

#[test]
fn bounded_offset_only_limit_runs_below_a_full_sort() {
    // Finite top-k planning needs a fetch bound: an offset-only limit cannot
    // fuse, so the sort stays a full sort and the limit applies the offset
    // over one complete frame.
    let mut builder = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut builder);
    let sort =
        builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::offset(1)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("join sort offset should compile");
    let sort_task =
        pipeline.tasks().iter().find(|task| task.kind() == PipelineTaskKind::FinalSort).expect("one final sort task");
    assert!(
        matches!(sort_task.sort_strategy(), Some(SortStrategy::FullSort)),
        "offset-only limit must not fuse into top-k, got {:?}",
        sort_task.sort_strategy()
    );
    bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("offset-only sort limit stays executable");
}

#[test]
fn bounded_sort_strategy_records_fused_binding_once() {
    // The fused top-k decision is recorded in exactly one place; the sort
    // task's strategy; with the consuming limit task and plan node bound to
    // it; readers use the recorded value and do not re-derive fusion.
    let mut builder = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut builder);
    let sort =
        builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(1)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("join sort limit should compile");
    let limit_task =
        pipeline.tasks().iter().find(|task| task.kind() == PipelineTaskKind::Limit).expect("one limit task");
    let strategies = pipeline
        .tasks()
        .iter()
        .filter_map(|task| task.sort_strategy().map(|strategy| (task.id(), strategy)))
        .collect::<Vec<_>>();
    assert_eq!(
        strategies,
        vec![(
            pipeline
                .tasks()
                .iter()
                .find(|task| task.kind() == PipelineTaskKind::FinalSort)
                .expect("one final sort task")
                .id(),
            SortStrategy::FusedTopK { limit_task: limit_task.id(), limit_node: limit },
        )],
        "exactly the sort task records the fused binding"
    );
}

#[test]
fn bounded_resource_ordering_tracks_sort_keys_through_frames() {
    // A final sort establishes the global key order; filters and limits pass
    // it through, a projection that drops the key column ends it, and an
    // aggregate stream never carries one.
    let mut builder = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut builder);
    let sort =
        builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(1)).unwrap();
    let filter = builder
        .add_filter(
            limit,
            Expr::binary(Expr::column("right_value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(i32::MIN))),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(filter).unwrap();
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("join sort limit filter should compile");
    let task_id = |kind| pipeline.tasks().iter().find(|task| task.kind() == kind).expect("task exists").id();
    let ordering_of = |kind| {
        let producer = task_id(kind);
        pipeline
            .resources()
            .iter()
            .find(|resource| resource.producer() == producer)
            .expect("producer output exists")
            .ordering()
    };
    for kind in [PipelineTaskKind::FinalSort, PipelineTaskKind::Limit, PipelineTaskKind::Unary] {
        let ordering = ordering_of(kind).expect("sorted frames stay ordered");
        assert_eq!(ordering.scope(), OrderingScope::Global);
        assert_eq!(ordering.keys().len(), 1);
        assert!(matches!(
            &ordering.keys()[0].expr,
            Expr::Column(column) if column.name() == "right_value"
        ));
    }

    let mut dropped = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut dropped);
    let sort =
        dropped.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)])).unwrap();
    let projection = dropped
        .add_projection(
            sort,
            vec![ProjectionOutput::new(Expr::column("left_id"), Field::new("left_id", DataType::Int32, false))],
        )
        .unwrap();
    let dropped_plan = dropped.finish_with_host_sink(projection).unwrap();
    let dropped_analysis = dropped_plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let dropped_pipeline =
        compile_pipeline(&dropped_plan, &dropped_analysis).expect("key-dropping projection should compile");
    let projection_task = dropped_pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Unary)
        .expect("one projection task");
    assert_eq!(
        dropped_pipeline
            .resources()
            .iter()
            .find(|resource| resource.producer() == projection_task.id())
            .expect("projection output exists")
            .ordering(),
        None,
        "dropping the key column ends the ordering"
    );

    let mut aggregated = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut aggregated);
    let aggregate = aggregated
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("left_id")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("left_value"), "count")],
            ),
            Arc::new(Schema::new(vec![
                Field::new("left_id", DataType::Int32, false),
                Field::new("count", DataType::Int64, false),
            ])),
        )
        .unwrap();
    let aggregate_plan = aggregated.finish_with_host_sink(aggregate).unwrap();
    let aggregate_analysis = aggregate_plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let aggregate_pipeline =
        compile_pipeline(&aggregate_plan, &aggregate_analysis).expect("grouped aggregate should compile");
    assert!(
        aggregate_pipeline.resources().iter().all(|resource| resource.ordering().is_none()),
        "aggregate outputs carry no ordering"
    );
}

#[test]
fn bounded_aggregate_state_and_max_rows_are_node_facts() {
    use query_planning::capability::AggregateStateContract;

    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/s3-node-facts.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let dedup = builder
        .add_aggregate(source, AggregateSpec::new(vec![Expr::column("k")], Vec::new()), Arc::clone(&key_schema))
        .unwrap();
    let scalar = builder
        .add_aggregate(
            dedup,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "count")],
            ),
            Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)])),
        )
        .unwrap();
    let limit = builder.add_limit(scalar, LimitSpec::fetch(1)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

    let node_fact = |id| analysis.nodes.iter().find(|node| node.node_id == id).expect("node capability exists");
    assert_eq!(node_fact(dedup).aggregate_state, Some(AggregateStateContract::DedupGroupOnly));
    assert_eq!(node_fact(dedup).max_output_rows, None);
    assert_eq!(node_fact(scalar).aggregate_state, Some(AggregateStateContract::Plain));
    assert_eq!(node_fact(scalar).max_output_rows, Some(1));
    assert_eq!(node_fact(limit).max_output_rows, Some(1));
    assert_eq!(node_fact(source).max_output_rows, None);
}

#[test]
fn bounded_left_anti_build_edge_requires_dedup_aggregate() {
    // EXCEPT ALL-shaped: the build side is a grouped aggregate with measures,
    // which the bounded engine cannot do with multiplicity.
    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/s3-anti-build-left.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/s3-anti-build-right.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let grouped = builder
        .add_aggregate(
            right,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "count")],
            ),
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int32, false),
                Field::new("count", DataType::Int64, false),
            ])),
        )
        .unwrap();
    let anti = builder
        .add_join(
            left,
            grouped,
            JoinSpec::new(
                JoinType::LEFT_ANTI,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
            .with_build_side(JoinSide::Right),
            Arc::clone(&key_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(anti).unwrap();
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

    let build_edge = analysis
        .edges
        .iter()
        .find(|edge| edge.producer == grouped && edge.consumer == EdgeConsumer::Node(anti))
        .expect("aggregate build edge exists");
    let CapabilityDecision::Rejected { reason, facts } = &build_edge.decision else {
        panic!("grouped-measure anti build should reject the build edge");
    };
    assert_eq!(*reason, Reason::LeftAntiBuildRequiresDedupAggregate);
    assert_eq!(facts.get("producer_aggregate_state").map(String::as_str), Some("plain"));
}

#[test]
fn bounded_left_anti_probe_edge_requires_dedup_proven_input() {
    // EXCEPT DISTINCT-shaped but the probe side is a raw source: a trailing
    // scalar count would count duplicates.
    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let one_schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
    let scalar_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/s3-anti-probe-left.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/s3-anti-probe-right.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let dedup = builder
        .add_aggregate(right, AggregateSpec::new(vec![Expr::column("k")], Vec::new()), Arc::clone(&key_schema))
        .unwrap();
    let anti = builder
        .add_join(
            left,
            dedup,
            JoinSpec::new(
                JoinType::LEFT_ANTI,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
            .with_build_side(JoinSide::Right),
            Arc::clone(&key_schema),
        )
        .unwrap();
    let count_input = builder
        .add_projection(anti, projection_outputs(vec![Expr::literal(LiteralValue::Int64(1)).alias("one")], one_schema))
        .unwrap();
    let count = builder
        .add_aggregate(
            count_input,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("one"), "count")],
            ),
            Arc::clone(&scalar_schema),
        )
        .unwrap();
    let projection =
        builder.add_projection(count, projection_outputs(vec![Expr::column("count")], scalar_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

    let probe_edge = analysis
        .edges
        .iter()
        .find(|edge| edge.producer == left && edge.consumer == EdgeConsumer::Node(anti))
        .expect("probe edge exists");
    let CapabilityDecision::Rejected { reason, facts } = &probe_edge.decision else {
        panic!("non-dedup anti probe should reject the probe edge");
    };
    assert_eq!(*reason, Reason::LeftAntiProbeRequiresDedupProvenInput);
    assert_eq!(facts.get("producer_aggregate_state").map(String::as_str), Some("none"));
}

#[test]
fn bounded_admission_refuses_a_plan_with_an_unreachable_node_as_malformed() {
    // A detached aggregate is not part of the plan the sink reads. `finish_with_host_sink` rejects
    // the graph with `query_engine.plan.graph` before any plan exists. Admission never runs, and no
    // capability decision is recorded.
    let mut builder = QueryPlanBuilder::new();
    let join = compositional_inner_join_base(&mut builder);
    let detached_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("metric", DataType::Int64, false),
    ]));
    let detached = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("left_id")],
                vec![
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("left_value"), "metric")
                        .with_distinct(true),
                ],
            ),
            detached_schema,
        )
        .unwrap();
    let error = builder.finish_with_host_sink(join).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.plan.graph");
    assert_eq!(error.fact_value("node"), Some(detached.to_string().as_str()));
}

pub(super) fn assert_rejection_facts_are_family_free(
    facts: &std::collections::BTreeMap<&'static str, String>,
    label: &str,
) {
    for retired_key in [
        "bounded_join_execution_family",
        "bounded_join_blocked_join_family",
        "bounded_join_specific_blocker",
        "bounded_join_shape_blocker",
        "bounded_join_lower_level_blockers",
        "bounded_join_required_shape_contract",
        "bounded_join_required_lower_level_contract",
    ] {
        assert_eq!(facts.get(retired_key), None, "{label}: retired rejection fact {retired_key} must not be emitted");
    }
    for (key, value) in facts {
        assert!(
            !value.contains("topology_unsupported")
                && !value.contains("BoundedJoinExecutionFamily")
                && !value.contains("BoundedJoinExecutionPath")
                && !key.ends_with("_pipeline_capability"),
            "{label}: {key}={value:?} must not carry retired family/path vocabulary"
        );
    }
}
