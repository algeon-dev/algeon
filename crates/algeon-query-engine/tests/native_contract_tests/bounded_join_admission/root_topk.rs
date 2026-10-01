#[test]
fn bounded_root_topk_join_aggregate_full_scan_below_grouped_aggregate_is_not_a_support_gap() {
    // Capacity is the allocator's: planning proves executability from node, edge,
    // join-capability, and subcontract facts, and the admitted grant's allocator
    // enforces the service-configured cap at runtime, reporting a typed
    // `query_memory_limit_exceeded` failure at its source. A missing, late, or
    // skipped dynamic filter must never change admission or results, so these
    // fixtures, which differ from admitted shapes only in scan-reduction evidence
    // (full scans or large exact sizes below the grouped aggregate), stay admitted.
    for variant in [
        RootTopKJoinAggregateVariant::MissingSourceReduction,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "{variant:?}: a full scan is a capacity question, not a support gap"
        );
        assert!(unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).is_none(), "{variant:?}");
    }
}

#[test]
fn bounded_root_topk_join_aggregate_reduction_on_unrelated_branch_does_not_gate_admission() {
    // Capacity is the allocator's, enforced as a typed cap failure at runtime.
    // Which branch a dynamic filter happens to reduce (or whether one runs at
    // all) must never change admission or results.
    let plan =
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "unrelated-branch reduction root TopK pipeline should be admitted"
    );
}

#[test]
fn bounded_root_topk_join_aggregate_allows_shared_source_reuse() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SharedSourceFanout);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "source reuse is a modeled source-read contract, not computed producer fanout"
    );
}

#[test]
fn bounded_root_topk_join_aggregate_allows_shared_computed_fanout_with_explicit_subplan_evidence() {
    for (variant, expected_shared_operator) in [
        (RootTopKJoinAggregateVariant::SharedComputedFanout, "join"),
        (RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout, "aggregate"),
        (RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout, "projection"),
        (RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout, "projection"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis(rapids_analysis_context());

        let has_expected_shared_producer = analysis.invariants.iter().any(|invariant| {
            if !matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
                || invariant.decision != CapabilityDecision::Satisfied
            {
                return false;
            }
            let Some(shared_producer) = invariant.nodes.first().copied() else {
                return false;
            };
            plan.node(shared_producer).is_some_and(|node| {
                matches!(
                    (&node.kind, expected_shared_operator),
                    (PlanNode::Aggregate(_), "aggregate")
                        | (PlanNode::Join(_), "join")
                        | (PlanNode::Projection(_), "projection")
                )
            })
        });
        assert!(has_expected_shared_producer, "{variant:?}: expected shared {expected_shared_operator} producer");

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "{variant:?} root TopK join/aggregate pipeline should be admitted"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_distinct_and_keyless_subcontracts() {
    for variant in [
        RootTopKJoinAggregateVariant::DistinctAggregate,
        RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin,
        RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin,
        RootTopKJoinAggregateVariant::KeylessJoin,
        // An unrelated scalar-aggregate branch (over an unreduced fact scan) does
        // not break the keyless subcontract: capacity is the allocator's, enforced
        // as a typed cap failure at runtime, so a missing, late, or skipped
        // dynamic filter must never change admission or results.
        RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate,
        RootTopKJoinAggregateVariant::ProjectedCrossJoin,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "{variant:?} root TopK join/aggregate pipeline should be admitted"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_direct_grouped_sources_feeding_keyless_conditionals() {
    let plan = root_topk_keyless_conditional_direct_source_aggregates_plan(false);

    assert!(modeled_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).is_some());
}

#[test]
fn bounded_root_topk_join_aggregate_admits_unreduced_direct_grouped_keyless_branches() {
    // Capacity is the allocator's, enforced as a typed cap failure at runtime; a
    // missing, late, or skipped dynamic filter on any direct grouped branch must
    // never change admission or results, so a branch scanning its source in full
    // is not a support gap.
    let plan = root_topk_keyless_conditional_direct_source_aggregates_plan(true);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "unreduced direct grouped keyless branches should be admitted"
    );
}

#[test]
fn bounded_root_topk_join_aggregate_allows_sort_fetch_hash_join_input_subcontract() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchHashJoinInput,
        RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "{variant:?} root TopK pipeline should be admitted"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_keyed_left_semi_join_carrying_a_residual_predicate() {
    let plan =
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchLeftSemiResidualJoinInput);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "a keyed left-semi join with a residual predicate emits left-only output and is executable on the \
         mixed keyed/residual path, so the root TopK pipeline must be admitted"
    );
}

#[test]
fn bounded_root_topk_allows_one_left_join_after_grouped_aggregate_topk() {
    let plan = root_topk_grouped_aggregate_input_left_join_plan();

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "grouped aggregate TopK left join should be admitted"
    );
}

#[test]
fn bounded_root_topk_join_aggregate_allows_whole_partition_window_filter_before_final_sort() {
    // Pins change B: a whole-partition aggregate window sitting directly between the
    // grouped-aggregate fragment and the final top-k sort/limit, with no projection
    // reinserted before the final sort, is the same composition already admitted
    // whenever an ORDER BY expression happens to force a projection into that chain
    // (q47/q57/q89). q53/q63 hit this path because their ORDER BY uses plain columns.
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::AggregateWindowFilterTopK);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "whole-partition window filter root TopK pipeline should be admitted"
    );
}

#[test]
fn bounded_root_topk_join_aggregate_allows_cumulative_window_full_outer_before_final_sort() {
    // Pins change B: q51's two cumulative-window branches, combined by a FULL OUTER
    // join and followed by a second cumulative window/filter with no projection
    // before the final sort, is admitted through the same window candidacy path.
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::CumulativeWindowFullOuterTopK);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "cumulative window full-outer root TopK pipeline should be admitted"
    );
}

#[test]
fn bounded_root_topk_join_aggregate_sort_fetch_join_input_subcontract_excludes_keyless_and_cross() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput,
        RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "{variant:?}"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_sort_fetch_join_aggregate_subcontracts() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
        RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput,
        RootTopKJoinAggregateVariant::SortFetchLeftMarkJoinInput,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "{variant:?} root TopK pipeline should be admitted"
        );
    }
}

#[test]
fn bounded_root_topk_join_aggregate_allows_null_filtered_aggregate_fan_in_subcontract() {
    for variant in [
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild,
    ] {
        let plan = optimize_query_plan(
            root_topk_join_aggregate_variant_plan(variant),
            &NativeOptimizerOptions::default(), rapids_analysis_context(),local_dimension_pruning_resolver())
        .plan;

        assert!(
            bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
            "null-filtered left-join aggregate fan-in should be admitted after rewriting marker filters to LeftAnti ({variant:?})"
        );
    }
}

/// One Inner keyed join over dummy paths. Admission never reads files, so the
/// compositional tests below share this fragment instead of writing parquet.
fn compositional_inner_join_base(builder: &mut QueryPlanBuilder) -> (NodeId, Arc<Schema>) {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-left.parquet",
        Arc::clone(&left_schema),
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-right.parquet",
        Arc::clone(&right_schema),
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    (join, output_schema)
}

fn compositional_join_build_input_form(
    plan: &QueryPlan,
) -> (
    AcceptedInputForm,
    ResidentDataForm,
    ResourceSharing,
) {
    let analysis = plan.capability_analysis(rapids_analysis_context());
    let pipeline = compile_pipeline(plan, &analysis)
        .expect("compositional fixture should compile");
    let build = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::JoinBuild)
        .expect("one join build task");
    let [input] = build.input_resources() else {
        panic!("join build task should consume exactly one resource");
    };
    let resource =
        pipeline.resources().iter().find(|resource| resource.id() == *input).expect("build input resource exists");
    (build.accepted_input_form(), resource.data_form(), resource.sharing())
}

#[test]
fn bounded_compositional_admission_accepts_join_sort_limit_filter() {
    // Join → Sort → Limit → Filter is decided by the Limit/Unary task contracts;
    // the retired post-limit projection allowlist must not reject the filter.
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
        Expr::binary(Expr::column("right_value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(100))),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(filter);

    let analysis = plan.capability_analysis(rapids_analysis_context());
    let pipeline = compile_pipeline(&plan, &analysis)
        .expect("join sort limit filter should compile");
    let limit_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Limit)
        .expect("one limit task");
    assert_eq!(limit_task.accepted_input_form(), AcceptedInputForm::Frame);
    let sort_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::FinalSort)
        .expect("one final sort task");
    assert!(
        matches!(sort_task.sort_strategy(), Some(SortStrategy::FusedTopK { .. })),
        "sort feeding a single limit fuses into top-k, got {:?}",
        sort_task.sort_strategy()
    );
    let filter_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Unary)
        .expect("one filter task");
    assert!(limit_task.id().index() < filter_task.id().index(), "the limit must stay below the filter");
    let [filter_input] = filter_task.input_resources() else {
        panic!("filter task should consume exactly one resource");
    };
    let filter_resource = pipeline
        .resources()
        .iter()
        .find(|resource| resource.id() == *filter_input)
        .expect("filter input resource exists");
    assert_eq!(filter_resource.data_form(), ResidentDataForm::Frame);

    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("join sort limit filter should be admitted");
    let _ = admission;
    assert!(unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).is_none(), "no post-limit rejection may remain");
}

#[test]
fn bounded_compositional_admission_accepts_join_limit_filter_without_sort() {
    // A Limit without an ORDER BY below it is still an executable composition;
    // the retired post-root rule demanding a sort input must not reject it.
    let mut builder = QueryPlanBuilder::new();
    let (join, output_schema) = compositional_inner_join_base(&mut builder);
    let limit = builder.add_limit(join, LimitSpec::fetch(10), Arc::clone(&output_schema));
    let filter = builder.add_filter(
        limit,
        Expr::binary(Expr::column("right_value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(100))),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(filter);

    bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("join limit filter should be admitted");
    assert!(unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).is_none(), "no post-root limit rejection may remain");
}

#[test]
fn bounded_compositional_admission_ignores_identity_projection_variants() {
    // Inserting identity projections over the join inputs must not change native
    // selection or the recorded contracts: both variants are admitted and the
    // join build input carries the same form and sharing.
    let mut plain_builder = QueryPlanBuilder::new();
    let (plain_join, _) = compositional_inner_join_base(&mut plain_builder);
    let plain = plain_builder.finish_with_host_sink(plain_join);

    // Rebuild with identity projections spliced over both join inputs.
    let mut spliced = QueryPlanBuilder::new();
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));
    let left = spliced.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-left.parquet",
        Arc::clone(&left_schema),
        vec!["left_id".into(), "left_value".into()],
    )));
    let left_projected = spliced.add_projection(
        left,
        vec![
            ProjectionOutput::new(Expr::column("left_id"), Field::new("left_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("left_value"), Field::new("left_value", DataType::Int32, false)),
        ],
    );
    let right = spliced.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-right.parquet",
        Arc::clone(&right_schema),
        vec!["right_id".into(), "right_value".into()],
    )));
    let right_projected = spliced.add_projection(
        right,
        vec![
            ProjectionOutput::new(Expr::column("right_id"), Field::new("right_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("right_value"), Field::new("right_value", DataType::Int32, false)),
        ],
    );
    let spliced_join = spliced.add_join(
        left_projected,
        right_projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    let projected = spliced.finish_with_host_sink(spliced_join);

    for (label, plan) in [("plain", &plain), ("projected", &projected)] {
        bounded_pipeline_admission(plan)
            .unwrap_or_else(|error| panic!("{label} join variant should not fail admission: {error}"))
            .unwrap_or_else(|| panic!("{label} join variant should be admitted"));
    }
    assert_eq!(
        compositional_join_build_input_form(&plain),
        compositional_join_build_input_form(&projected),
        "identity projections must not change the recorded join input contract"
    );
}

/// The [`compositional_inner_join_base`] fragment with an alias-only projection
/// renaming the join output. Same executable pipeline plus one unary stage.
fn compositional_alias_join_plan() -> QueryPlan {
    let mut aliased_builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut aliased_builder);
    let aliased = aliased_builder.add_projection(
        join,
        vec![
            ProjectionOutput::new(Expr::column("left_id"), Field::new("id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("right_value"), Field::new("right_value", DataType::Int32, true)),
        ],
    );
    aliased_builder.finish_with_host_sink(aliased)
}

#[test]
fn bounded_compositional_admission_accepts_alias_projection_above_join() {
    // Renaming the join output through an alias-only projection is the same
    // executable pipeline plus one unary stage.
    let mut plain_builder = QueryPlanBuilder::new();
    let (plain_join, _) = compositional_inner_join_base(&mut plain_builder);
    let plain = plain_builder.finish_with_host_sink(plain_join);
    let plain_tasks = compile_pipeline(&plain, &plain.capability_analysis(rapids_analysis_context()))
        .expect("plain join should compile")
        .tasks()
        .len();

    let aliased_plan = compositional_alias_join_plan();

    bounded_pipeline_admission(&aliased_plan)
        .expect("bounded admission should not fail")
        .expect("alias projection above join should be admitted");
    let aliased_pipeline =
        compile_pipeline(&aliased_plan, &aliased_plan.capability_analysis(rapids_analysis_context()))
            .expect("aliased join should compile");
    assert_eq!(aliased_pipeline.tasks().len(), plain_tasks + 1, "aliasing adds exactly one unary task");
    assert_eq!(
        compositional_join_build_input_form(&plain),
        compositional_join_build_input_form(&aliased_plan),
        "aliasing the output must not change the recorded join input contract"
    );
}

/// A join whose output flows through edge normalization to the sink.
fn compositional_edge_normalize_plan() -> (QueryPlan, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let key_schema = Arc::new(Schema::new(vec![
        Field::new("join_key", DataType::Int32, false),
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("join_key", DataType::Int32, false)]));
    let join_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int32, false), Field::new("dst", DataType::Int32, false)]));
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-edge-left.parquet",
        Arc::clone(&key_schema),
        vec!["join_key".into(), "src".into(), "dst".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-edge-right.parquet",
        Arc::clone(&right_schema),
        vec!["join_key".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "join_key", "join_key"),
                ColumnRef::input(1, "join_key", "join_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "src", "src"), "src"),
            JoinOutputColumn::left(ColumnRef::input(0, "dst", "dst"), "dst"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let edge_normalize = builder.add_edge_normalize(
        join,
        EdgeNormalizeSpec {
            edge_columns: GraphEdgeColumnBindings::new("src", "dst", None::<String>),
            add_reverse: false,
            dedup: EdgeDedupStrategy::KeepLast,
            weight_combiner: EdgeWeightCombiner::None,
        },
        join_schema,
    );
    let plan = builder.finish_with_host_sink(edge_normalize);
    (plan, edge_normalize)
}

#[test]
fn bounded_compositional_support_admits_graph_task_on_sink_path() {
    // A graph or edge-normalize task hands its one resident frame on to the
    // sink like any other stage output.
    let (plan, _) = compositional_edge_normalize_plan();

    let analysis = plan.capability_analysis(rapids_analysis_context());
    let pipeline = compile_pipeline(&plan, &analysis).expect("join edge-normalize plan should compile");
    let support = compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail");
    assert!(matches!(support, CompiledBoundedPipelineSupport::Supported));

    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded decision should not fail")
        .expect("join plus edge normalization is admitted by the task contracts");
    assert!(admission.pipeline().tasks().iter().any(|task| task.kind() == PipelineTaskKind::EdgeNormalize));
}

/// A row-number window over the join output, sorted on the window's order key
/// first when `sorted`.
fn compositional_join_row_number_plan(sorted: bool) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let (join, join_schema) = compositional_inner_join_base(&mut builder);
    let order_key = SortExpr::new(Expr::column("left_id"), false, false);
    let window_input = if sorted {
        builder.add_sort(join, SortSpec::new(vec![order_key.clone()]), Arc::clone(&join_schema))
    } else {
        join
    };
    let window_schema = Arc::new(Schema::new(
        join_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new("row_number", DataType::Int64, false)])
            .collect::<Vec<_>>(),
    ));
    let window = builder.add_window(
        window_input,
        WindowSpec::new(Vec::new(), vec![order_key], vec![WindowFunction::row_number_i64("row_number")]),
        window_schema,
    );
    builder.finish_with_host_sink(window)
}

#[test]
fn bounded_compositional_admission_accepts_row_number_over_ordered_join_output() {
    let plan = compositional_join_row_number_plan(true);

    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded decision should not fail")
        .expect("a row number over the sorted join output is admitted");
    assert!(admission.pipeline().tasks().iter().any(|task| task.kind() == PipelineTaskKind::Window));
}

#[test]
fn bounded_compositional_compile_rejects_shared_exclusive_consumer() {
    // A resource shared with an edge-normalize consumer cannot be handed to
    // both readers: edge normalization moves its whole input out of the
    // schedule, and the compiler has no conversion that serves both.
    let triple_schema = Arc::new(Schema::new(vec![
        Field::new("join_key", DataType::Int32, false),
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-shared-left.parquet",
        Arc::clone(&triple_schema),
        vec!["join_key".into(), "src".into(), "dst".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compositional-shared-right.parquet",
        Arc::clone(&triple_schema),
        vec!["join_key".into(), "src".into(), "dst".into()],
    )));
    let first_join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "join_key", "join_key"),
                ColumnRef::input(1, "join_key", "join_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "join_key", "join_key"), "join_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "src", "src"), "src"),
            JoinOutputColumn::left(ColumnRef::input(0, "dst", "dst"), "dst"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&triple_schema),
    );
    let edge_schema =
        Arc::new(Schema::new(vec![Field::new("src", DataType::Int32, false), Field::new("dst", DataType::Int32, false)]));
    let shared = builder.add_projection(
        first_join,
        vec![
            ProjectionOutput::new(Expr::column("src"), Field::new("src", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("dst"), Field::new("dst", DataType::Int32, false)),
        ],
    );
    let edge_normalize = builder.add_edge_normalize(
        shared,
        EdgeNormalizeSpec {
            edge_columns: GraphEdgeColumnBindings::new("src", "dst", None::<String>),
            add_reverse: false,
            dedup: EdgeDedupStrategy::KeepLast,
            weight_combiner: EdgeWeightCombiner::None,
        },
        Arc::clone(&edge_schema),
    );
    let passthrough = builder.add_filter(
        shared,
        Expr::binary(Expr::column("src"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&edge_schema),
    );
    let union = builder.add_union(vec![edge_normalize, passthrough], edge_schema);
    let plan = builder.finish_with_host_sink(union);

    let analysis = plan.capability_analysis(rapids_analysis_context());
    let error =
        compile_pipeline(&plan, &analysis).expect_err("shared exclusive input must not compile");
    let reason = error.fact_value("capability_reason").expect("compile rejection names its contract");
    assert_eq!(reason, "consumer_requires_exclusive_input");
    assert_rejection_reason_is_family_free(reason, "exclusive consumer rejection");

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("bounded decision should not fail");
    assert!(decision.admission().is_none(), "shared edge-normalize input must not be admitted");
}
