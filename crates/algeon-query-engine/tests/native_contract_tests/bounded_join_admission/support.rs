fn assert_rejection_reason_is_family_free(reason: &str, label: &str) {
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
    let (plan, _) = compositional_edge_normalize_plan();
    let error = query_engine::exec::preflight_bounded_query_plan(
        &plan,
        query_engine::mode::StatisticalAggregateExecutionMode::default(),
    )
    .expect_err("edge normalization must not preflight");
    assert_eq!(error.fact_value("caller_visible_sink"), Some("host_arrow"));
}

fn admitted_artifact_for(plan: &query_engine::plan::QueryPlan) -> query_engine::admission::BoundedPipelineAdmission {
    bounded_pipeline_admission(plan)
        .expect("artifact fixture should not fail admission")
        .expect("artifact fixture should be admitted")
}

#[test]
fn bounded_artifact_binding_matches_decided_plan_and_mode() {
    // The artifact binds the exact plan, mode, and sink it was decided for;
    // execution and reporting read this same binding.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let plan = builder.finish_with_host_sink(join);
    let mode = query_engine::mode::StatisticalAggregateExecutionMode::default();

    let admission = admitted_artifact_for(&plan);
    assert!(admission.matches(&plan, mode));
    assert_eq!(admission.sink_kind(), query_engine::pipeline::SinkBoundaryKind::HostArrowMaterialization);
    assert_eq!(admission.statistical_aggregate_execution_mode(), mode);
    assert!(admission.scope().matches(&plan));
}

#[test]
fn bounded_artifact_binding_rejects_mismatched_plan_and_mode() {
    // A plan, mode, or sink the artifact does not bind to is a caller error
    // with a typed reason, never a silent re-decision.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let plan = builder.finish_with_host_sink(join);
    let mode = query_engine::mode::StatisticalAggregateExecutionMode::default();
    let admission = admitted_artifact_for(&plan);

    let aliased = compositional_alias_join_plan();
    assert!(!admission.matches(&aliased, mode));
    let projection =
        admission.binding_rejection(&aliased, mode).expect("mismatched plan must name its binding clause");
    assert_eq!(projection.reason, "compiled_pipeline_sink_output_plan_node_mismatch");
    assert_rejection_reason_is_family_free(projection.reason, "artifact binding rejection");

    let other_mode = query_engine::mode::StatisticalAggregateExecutionMode::DataFusionExactHostWelford;
    assert!(mode != other_mode, "binding test needs two distinct execution modes");
    assert!(!admission.matches(&plan, other_mode));
    let projection =
        admission.binding_rejection(&plan, other_mode).expect("mismatched mode must name its binding clause");
    assert_eq!(projection.reason, "compiled_pipeline_artifact_mode_mismatch");
}

#[test]
fn bounded_compiled_pipeline_rejects_duplicate_fused_topk_limit_claims() {
    // `applied_limit` is set by exactly one producer and dropped by exactly
    // one limit pass-through: two tasks claiming `FusedTopK` for the same
    // limit would apply fetch/skip twice.
    let mut builder = QueryPlanBuilder::new();
    let (join, output_schema) = compositional_inner_join_base(&mut builder);
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(limit);

    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let pipeline = query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("join sort limit should compile");
    pipeline.validate().expect("compiler output satisfies the fused top-k binding");

    let sort_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::FinalSort)
        .expect("one final sort task");
    let fused = sort_task.sort_strategy().expect("sort feeding a single limit fuses");
    let (limit_task_id, limit_node_id) = match fused {
        query_engine::pipeline::SortStrategy::FusedTopK { limit_task, limit_node } => {
            (limit_task, limit_node)
        },
        query_engine::pipeline::SortStrategy::FullSort => {
            panic!("sort feeding a single limit must fuse, got {fused:?}")
        },
    };
    // Duplicate the claim onto the limit task itself: kind, plan node, and
    // sort-output consumption still hold for both claimants, so only the
    // uniqueness clause can reject the rebuilt pipeline.
    let mut tasks = pipeline.tasks().to_vec();
    let limit_index =
        tasks.iter().position(|task| task.id() == limit_task_id).expect("fused limit task exists");
    tasks[limit_index] = tasks[limit_index].clone().with_sort_strategy(fused);
    let rebuilt = query_engine::pipeline::CompiledPipeline::new(
        tasks,
        pipeline.edges().to_vec(),
        pipeline.barriers().to_vec(),
        pipeline.resources().to_vec(),
        pipeline.sink().clone(),
    );
    let error = rebuilt.validate().expect_err("two producers must not claim one fused limit");
    assert_eq!(error.code(), query_engine::error::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("first_sort_task"), Some(sort_task.id().to_string()).as_deref());
    assert_eq!(error.fact_value("second_sort_task"), Some(limit_task_id.to_string()).as_deref());
    assert_eq!(error.fact_value("limit_task"), Some(limit_task_id.to_string()).as_deref());
    assert_eq!(error.fact_value("limit_node"), Some(limit_node_id.to_string()).as_deref());
}

#[test]
fn bounded_offset_only_limit_runs_below_a_full_sort() {
    // Finite top-k planning needs a fetch bound: an offset-only limit cannot
    // fuse, so the sort stays a full sort and the limit applies the offset
    // over one complete frame.
    let mut builder = QueryPlanBuilder::new();
    let (join, output_schema) = compositional_inner_join_base(&mut builder);
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::offset(1), Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(limit);

    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let pipeline = query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("join sort offset should compile");
    let sort_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::FinalSort)
        .expect("one final sort task");
    assert!(
        matches!(sort_task.sort_strategy(), Some(query_engine::pipeline::SortStrategy::FullSort)),
        "offset-only limit must not fuse into top-k, got {:?}",
        sort_task.sort_strategy()
    );
    bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("offset-only sort limit stays executable");
}

#[test]
fn bounded_sort_strategy_records_fused_binding_once() {
    // The fused top-k decision is recorded in exactly one place — the sort
    // task's strategy — with the consuming limit task and plan node bound to
    // it; readers use the recorded value instead of re-deriving fusion.
    let mut builder = QueryPlanBuilder::new();
    let (join, output_schema) = compositional_inner_join_base(&mut builder);
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(limit);

    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let pipeline = query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("join sort limit should compile");
    let limit_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::Limit)
        .expect("one limit task");
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
                .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::FinalSort)
                .expect("one final sort task")
                .id(),
            query_engine::pipeline::SortStrategy::FusedTopK { limit_task: limit_task.id(), limit_node: limit },
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
    let (join, output_schema) = compositional_inner_join_base(&mut builder);
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), Arc::clone(&output_schema));
    let filter = builder.add_filter(
        limit,
        Expr::binary(
            Expr::column("right_value"),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::Int32(i32::MIN)),
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(filter);
    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let pipeline = query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("join sort limit filter should compile");
    let task_id = |kind| {
        pipeline.tasks().iter().find(|task| task.kind() == kind).expect("task exists").id()
    };
    let ordering_of = |kind| {
        let producer = task_id(kind);
        pipeline
            .resources()
            .iter()
            .find(|resource| resource.producer() == producer)
            .expect("producer output exists")
            .ordering()
    };
    for kind in [
        query_engine::pipeline::PipelineTaskKind::FinalSort,
        query_engine::pipeline::PipelineTaskKind::Limit,
        query_engine::pipeline::PipelineTaskKind::Unary,
    ] {
        let ordering = ordering_of(kind).expect("sorted frames stay ordered");
        assert_eq!(ordering.scope(), query_engine::pipeline::OrderingScope::Global);
        assert_eq!(ordering.keys().len(), 1);
        assert!(matches!(
            &ordering.keys()[0].expr,
            query_engine::expr::Expr::Column(column) if column.name() == "right_value"
        ));
    }

    let mut dropped = QueryPlanBuilder::new();
    let (join, output_schema) = compositional_inner_join_base(&mut dropped);
    let sort = dropped.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let projection = dropped.add_projection(
        sort,
        vec![ProjectionOutput::new(
            Expr::column("left_id"),
            Field::new("left_id", DataType::Int32, false),
        )],
    );
    let dropped_plan = dropped.finish_with_host_sink(projection);
    let dropped_analysis = dropped_plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let dropped_pipeline =
        query_engine::pipeline::compile_pipeline(&dropped_plan, &dropped_analysis)
            .expect("key-dropping projection should compile");
    let projection_task = dropped_pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::Unary)
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
    let (join, _) = compositional_inner_join_base(&mut aggregated);
    let aggregate = aggregated.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("left_id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("left_value"), "count")],
        ),
        Arc::new(Schema::new(vec![
            Field::new("left_id", DataType::Int32, false),
            Field::new("count", DataType::Int64, false),
        ])),
    );
    let aggregate_plan = aggregated.finish_with_host_sink(aggregate);
    let aggregate_analysis = aggregate_plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let aggregate_pipeline =
        query_engine::pipeline::compile_pipeline(&aggregate_plan, &aggregate_analysis)
            .expect("grouped aggregate should compile");
    assert!(
        aggregate_pipeline.resources().iter().all(|resource| resource.ordering().is_none()),
        "aggregate outputs carry no ordering"
    );
}

#[test]
fn bounded_aggregate_task_records_single_stage() {
    // Compiled aggregate tasks always run single-shot: partial state never
    // crosses a compiled edge, it is merged inside the task.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("left_id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("left_value"), "count")],
        ),
        Arc::new(Schema::new(vec![
            Field::new("left_id", DataType::Int32, false),
            Field::new("count", DataType::Int64, false),
        ])),
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let pipeline = query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("grouped aggregate should compile");
    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::Aggregate)
        .expect("one aggregate task");
    assert_eq!(
        aggregate_task.aggregate_stage(),
        Some(query_engine::pipeline::AggregateExecutionStage::Single)
    );
    assert!(
        pipeline.tasks().iter().all(|task| !matches!(
            task.aggregate_stage(),
            Some(query_engine::pipeline::AggregateExecutionStage::Partial)
                | Some(query_engine::pipeline::AggregateExecutionStage::FinalMerge)
        )),
        "no compiled edge carries partial state"
    );
}

#[test]
fn bounded_aggregate_state_and_max_rows_are_node_facts() {
    use query_engine::capability::AggregateStateContract;

    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/s3-node-facts.parquet",
        Arc::clone(&key_schema),
        vec!["k".into()],
    )));
    let dedup = builder.add_aggregate(
        source,
        AggregateSpec::new(vec![Expr::column("k")], Vec::new()),
        Arc::clone(&key_schema),
    );
    let scalar = builder.add_aggregate(
        dedup,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "count")],
        ),
        Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)])),
    );
    let limit = builder.add_limit(
        scalar,
        LimitSpec::fetch(1),
        Arc::new(Schema::new(vec![Field::new("count", DataType::Int64, false)])),
    );
    let plan = builder.finish_with_host_sink(limit);
    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());

    let node_fact = |id| {
        analysis.nodes.iter().find(|node| node.node_id == id).expect("node capability exists")
    };
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
    let grouped = builder.add_aggregate(
        right,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "count")],
        ),
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("count", DataType::Int64, false),
        ])),
    );
    let anti = builder.add_join(
        left,
        grouped,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&key_schema),
    );
    let plan = builder.finish_with_host_sink(anti);
    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());

    let build_edge = analysis
        .edges
        .iter()
        .find(|edge| {
            edge.producer == grouped
                && edge.consumer == query_engine::capability::EdgeConsumer::Node(anti)
        })
        .expect("aggregate build edge exists");
    let query_engine::capability::CapabilityDecision::Rejected { reason, facts } = &build_edge.decision else {
        panic!("grouped-measure anti build should reject the build edge");
    };
    assert_eq!(*reason, "left_anti_build_requires_dedup_aggregate");
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
    let dedup = builder.add_aggregate(
        right,
        AggregateSpec::new(vec![Expr::column("k")], Vec::new()),
        Arc::clone(&key_schema),
    );
    let anti = builder.add_join(
        left,
        dedup,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&key_schema),
    );
    let count_input = builder.add_projection(
        anti,
        projection_outputs(vec![Expr::literal(LiteralValue::Int64(1)).alias("one")], one_schema),
    );
    let count = builder.add_aggregate(
        count_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("one"), "count")],
        ),
        Arc::clone(&scalar_schema),
    );
    let projection = builder.add_projection(count, projection_outputs(vec![Expr::column("count")], scalar_schema));
    let plan = builder.finish_with_host_sink(projection);
    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());

    let probe_edge = analysis
        .edges
        .iter()
        .find(|edge| {
            edge.producer == left && edge.consumer == query_engine::capability::EdgeConsumer::Node(anti)
        })
        .expect("probe edge exists");
    let query_engine::capability::CapabilityDecision::Rejected { reason, facts } = &probe_edge.decision else {
        panic!("non-dedup anti probe should reject the probe edge");
    };
    assert_eq!(*reason, "left_anti_probe_requires_dedup_proven_input");
    assert_eq!(facts.get("producer_aggregate_state").map(String::as_str), Some("none"));
}

#[test]
fn bounded_admission_ignores_unreachable_node_rejection() {
    // A detached aggregate that fails node capability must not veto a plan
    // the compiler never instantiates it for.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let detached_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("metric", DataType::Int64, false),
    ]));
    let detached = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("left_id")],
            vec![
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("left_value"), "metric")
                    .with_distinct(true),
            ],
        ),
        detached_schema,
    );
    let plan = builder.finish_with_host_sink(join);

    let unreachable = plan.unreachable_node_ids();
    assert!(unreachable.contains(&detached), "fixture must keep the bad projection detached: {unreachable:?}");
    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    assert!(
        analysis.nodes.iter().any(|node| node.node_id == detached
            && matches!(
                node.executable,
                query_engine::capability::CapabilityDecision::Rejected { .. }
            )),
        "fixture must carry a detached node rejection"
    );
    bounded_pipeline_admission(&plan)
        .expect("admission should not fail")
        .expect("an unreachable node rejection must not veto the reachable pipeline");
}

fn assert_rejection_facts_are_family_free(facts: &std::collections::BTreeMap<&'static str, String>, label: &str) {
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

#[test]
fn bounded_compiled_aggregate_task_runs_single_stage_without_partial_edges() {
    // Compiled aggregate tasks run single-shot: partial/final state never
    // crosses a compiled edge (it is merged inside the task), so the task
    // records `Single` and no edge carries a partial-state label.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("line_count", DataType::Int64, false),
    ]));
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("left_id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("right_value"), "line_count")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let analysis = plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context());
    let pipeline =
        query_engine::pipeline::compile_pipeline(&plan, &analysis).expect("join grouped aggregate compiles");
    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == query_engine::pipeline::PipelineTaskKind::Aggregate)
        .expect("one aggregate task");
    assert_eq!(aggregate_task.aggregate_stage(), Some(AggregateExecutionStage::Single));
    assert!(
        pipeline.edges().iter().all(|edge| !format!("{:?}", edge.kind()).contains("Partial")),
        "no compiled edge may carry partial aggregate state"
    );
}
