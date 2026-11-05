use query_planning::optimizer::{SharedResidentFragments, extract_shared_resident_fragments};

use super::*;

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
    let filtered = builder
        .add_filter(
            source,
            Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::Literal(LiteralValue::Utf8("a".to_string()))),
        )
        .unwrap();
    let repeated = if repeated_aggregate {
        builder
            .add_aggregate(
                filtered,
                AggregateSpec::new(
                    vec![Expr::column("id")],
                    vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("name"), "name")],
                ),
                Arc::clone(&schema),
            )
            .unwrap()
    } else {
        filtered
    };
    let output = Arc::new(Schema::new(vec![
        Field::new("left_name", DataType::Utf8, false),
        Field::new("right_name", DataType::Utf8, false),
    ]));
    let join = builder
        .add_join(
            repeated,
            repeated,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("name"), "left_name"),
                JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("name"), "right_name"),
            ]),
            Arc::clone(&output),
        )
        .unwrap();
    let resident = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("existing-resident", Arc::clone(&schema)).with_row_count_upper_bound(8),
    ));
    let outer = builder
        .add_join(
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
        )
        .unwrap();
    // The outer join's build side narrows the scan inside the repeated
    // subplan, which every consumer of a shared producer would then see.
    let plan = builder.finish_with_host_sink_schema(outer, output).unwrap().with_dynamic_filter_routes_for_test(vec![
        DynamicFilterRoute::new(
            DynamicFilterId::new(0),
            DynamicFilterProducer::HashJoinBuild {
                join_node_id: outer,
                build_side: JoinSide::Right,
                build_input: resident,
            },
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

    let optimized = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::disabled(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
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

    let optimized = optimize_query_plan(
        plan.clone(),
        &NativeOptimizerOptions::disabled(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

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

/// A `likes(s, j, k, v, w)` scan reading `row_groups` of four; `adjust` sets
/// reader-level facts such as a reader predicate or a scan limit.
fn likes_scan(
    builder: &mut QueryPlanBuilder,
    row_groups: Option<Vec<usize>>,
    adjust: impl FnOnce(LocalParquetSource) -> LocalParquetSource,
) -> NodeId {
    table_scan(builder, "likes", row_groups, adjust)
}

/// [`likes_scan`] over the file of `table`. The builder merges equal source
/// nodes, so two tables of one plan need two files.
fn table_scan(
    builder: &mut QueryPlanBuilder,
    table: &str,
    row_groups: Option<Vec<usize>>,
    adjust: impl FnOnce(LocalParquetSource) -> LocalParquetSource,
) -> NodeId {
    let schema = Arc::new(Schema::new(vec![
        Field::new("s", DataType::Int64, false),
        Field::new("j", DataType::Int32, false),
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("w", DataType::Float64, false),
    ]));
    let mut file = LocalParquetFile::new(format!("/tmp/shared-resident-residual-{table}-not-read.parquet"))
        .with_file_size_bytes(256)
        .with_row_group_bytes(vec![64; 4])
        .with_projected_row_group_bytes(vec![64; 4], vec![64; 4])
        .with_row_group_rows(vec![8; 4]);
    file.row_groups = row_groups;
    builder.add_source(SourceNode::LocalParquet(adjust(LocalParquetSource::from_files(
        vec![file],
        schema,
        ["s", "j", "k", "v", "w"].map(str::to_string).to_vec(),
    ))))
}

/// The scan of [`likes_scan`] under a filter directly above the source.
fn filtered_likes_source(builder: &mut QueryPlanBuilder, row_groups: Option<Vec<usize>>, predicate: Expr) -> NodeId {
    let source = likes_scan(builder, row_groups, |source| source);
    builder.add_filter(source, predicate).unwrap()
}

fn compare(column: &str, op: BinaryOp, value: LiteralValue) -> Expr {
    Expr::binary(Expr::column(column), op, Expr::Literal(value))
}

fn all_of(mut conjuncts: Vec<Expr>) -> Expr {
    let first = conjuncts.remove(0);
    conjuncts.into_iter().fold(first, |left, right| Expr::binary(left, BinaryOp::And, right))
}

/// `SELECT DISTINCT s AS src, CAST(j AS BIGINT) AS dst, <late> AS late` over
/// the node `input` builds, as a fragment ending at a resident sink.
fn distinct_fragment(input: impl FnOnce(&mut QueryPlanBuilder) -> NodeId, late: Expr) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = input(&mut builder);
    let projected = builder
        .add_projection(
            input,
            vec![
                ProjectionOutput::new(Expr::column("s"), Field::new("src", DataType::Int64, false)),
                ProjectionOutput::new(
                    Expr::Cast { expr: Box::new(Expr::column("j")), target_type: DataType::Int64 },
                    Field::new("dst", DataType::Int64, false),
                ),
                ProjectionOutput::new(late, Field::new("late", DataType::Boolean, false)),
            ],
        )
        .unwrap();
    let distinct = builder
        .add_aggregate(
            projected,
            AggregateSpec::new(vec![Expr::column("src"), Expr::column("dst"), Expr::column("late")], Vec::new()),
            Arc::new(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
                Field::new("late", DataType::Boolean, false),
            ])),
        )
        .unwrap();
    builder.finish_gpu_resident(distinct).unwrap()
}

fn late_from_k() -> Expr {
    compare("k", BinaryOp::GtEq, LiteralValue::Int64(10))
}

/// [`distinct_fragment`] with `late = k >= 10` over the filtered scan.
fn distinct_likes_fragment(row_groups: Option<Vec<usize>>, predicate: Expr) -> QueryPlan {
    distinct_fragment(|builder| filtered_likes_source(builder, row_groups, predicate), late_from_k())
}

fn residual_and_resident_input(fragment: &QueryPlan) -> (Expr, String) {
    let PlanNode::Filter(filter) = &fragment.root_node().kind else {
        panic!("the fragment reads the shared result under a residual filter: {fragment:?}");
    };
    let PlanNode::Source(source) = &fragment.node(filter.input).unwrap().kind else {
        panic!("the residual filter reads the resident source");
    };
    let SourceNode::GpuResident(source) = source.as_ref() else {
        panic!("the residual filter reads a resident source")
    };
    (filter.predicate.clone(), source.input_name.clone())
}

#[test]
fn period_copies_and_an_all_period_copy_share_one_producer_under_residual_filters() {
    let k = |op, value| compare("k", op, LiteralValue::Int64(value));
    let before = all_of(vec![k(BinaryOp::GtEq, 0), k(BinaryOp::Lt, 10)]);
    // The lower bound is implied by `k >= 10`, as DataFusion drops it.
    let after = all_of(vec![k(BinaryOp::GtEq, 10), k(BinaryOp::Lt, 20)]);
    let all =
        all_of(vec![k(BinaryOp::GtEq, 0), k(BinaryOp::Lt, 20), compare("j", BinaryOp::Gt, LiteralValue::Int32(0))]);
    let fragments = vec![
        distinct_likes_fragment(Some(vec![0, 1]), before),
        distinct_likes_fragment(Some(vec![1, 2]), after),
        distinct_likes_fragment(None, all),
    ];

    let shared = extract_shared_resident_fragments(fragments).unwrap();

    let [producer] = shared.producers.as_slice() else { panic!("one producer for the three copies: {shared:?}") };
    assert_eq!(producer.consumer_fragments, [0, 1, 2]);
    let residuals = shared.fragments.iter().map(residual_and_resident_input).collect::<Vec<_>>();
    assert_eq!(
        residuals,
        [
            (Expr::Not(Box::new(Expr::column("late"))), producer.input_name.clone()),
            (Expr::column("late"), producer.input_name.clone()),
            (compare("dst", BinaryOp::Gt, LiteralValue::Int64(0)), producer.input_name.clone()),
        ]
    );
    // The producer scans the copy whose row groups cover the others.
    let PlanNode::Source(source) = &producer.plan.nodes()[0].kind else { panic!("the producer starts at its scan") };
    let SourceNode::LocalParquet(source) = source.as_ref() else { panic!("the producer scans Parquet") };
    assert_eq!(source.files[0].row_groups, None);
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Extracted, SharedResidentExtractionReason::ExtractedUnderResidualFilter);
            3]
    );
    assert_eq!(shared.diagnostics.extracted_count, 1);
    assert_eq!(shared.diagnostics.rejected_count, 0);
}

/// Decisions for each fragment's root, the only node without a consumer.
fn root_reasons(
    shared: &SharedResidentFragments,
) -> Vec<(SharedResidentExtractionDecision, SharedResidentExtractionReason)> {
    shared
        .diagnostics
        .candidates
        .iter()
        .filter(|candidate| candidate.consumer_occurrences.is_empty())
        .map(|candidate| (candidate.decision, candidate.reason))
        .collect()
}

#[test]
fn copies_whose_row_groups_no_copy_covers_stay_separate() {
    let k = |op, value| compare("k", op, LiteralValue::Int64(value));
    let shared = extract_shared_resident_fragments(vec![
        distinct_likes_fragment(Some(vec![0, 1]), k(BinaryOp::Lt, 10)),
        distinct_likes_fragment(Some(vec![2, 3]), k(BinaryOp::GtEq, 10)),
    ])
    .unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::WidenedScanNotCovered); 2]
    );
}

/// `SELECT k, SUM(v) FROM likes WHERE <predicate> GROUP BY k`.
fn grouped_sum_fragment(predicate: Expr) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let filtered = filtered_likes_source(&mut builder, None, predicate);
    let projected = builder
        .add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, false)),
                ProjectionOutput::new(Expr::column("s"), Field::new("s", DataType::Int64, false)),
            ],
        )
        .unwrap();
    let grouped = builder
        .add_aggregate(
            projected,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")],
            ),
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, true),
            ])),
        )
        .unwrap();
    builder.finish_gpu_resident(grouped).unwrap()
}

#[test]
fn copies_differing_in_an_aggregated_column_stay_separate() {
    let v = |op, value| compare("v", op, LiteralValue::Int64(value));
    let shared = extract_shared_resident_fragments(vec![
        grouped_sum_fragment(v(BinaryOp::Lt, 0)),
        grouped_sum_fragment(v(BinaryOp::Gt, 0)),
    ])
    .unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::ResidualNotMappedToOutput); 2]
    );
}

fn k_compare(op: BinaryOp, value: i64) -> Expr {
    compare("k", op, LiteralValue::Int64(value))
}

#[test]
fn a_copy_filtered_on_the_null_supplying_side_of_a_left_join_stays_separate() {
    // `j` comes from the right input of a left join, so filtering it before
    // the join keeps the left rows with null `j` that a filter on the group
    // key `j` would drop.
    let fragment = |predicate| {
        let mut builder = QueryPlanBuilder::new();
        let left = table_scan(&mut builder, "users", None, |source| source);
        let right = filtered_likes_source(&mut builder, None, predicate);
        let joined = builder
            .add_join(
                left,
                right,
                JoinSpec::new(
                    JoinType::Left,
                    vec![JoinKey::new(ColumnRef::unqualified("k"), ColumnRef::unqualified("k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("s"), "s"),
                    JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("j"), "j"),
                ]),
                Arc::new(Schema::new(vec![
                    Field::new("s", DataType::Int64, false),
                    Field::new("j", DataType::Int32, true),
                ])),
            )
            .unwrap();
        let distinct = builder
            .add_aggregate(
                joined,
                AggregateSpec::new(vec![Expr::column("s"), Expr::column("j")], Vec::new()),
                Arc::new(Schema::new(vec![
                    Field::new("s", DataType::Int64, false),
                    Field::new("j", DataType::Int32, true),
                ])),
            )
            .unwrap();
        builder.finish_gpu_resident(distinct).unwrap()
    };
    let j = |value| compare("j", BinaryOp::Gt, LiteralValue::Int32(value));

    let shared = extract_shared_resident_fragments(vec![fragment(j(0)), fragment(j(5))]).unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::ResidualNotMappedToOutput); 2]
    );
}

#[test]
fn copies_split_on_a_floating_column_stay_separate() {
    // NaN fails both `w < 1` and `w >= 1`, so `NOT late` is not `w < 1`.
    let w = |op| compare("w", op, LiteralValue::Float64(1.0));
    let fragment =
        |predicate| distinct_fragment(|builder| filtered_likes_source(builder, None, predicate), w(BinaryOp::GtEq));

    let shared =
        extract_shared_resident_fragments(vec![fragment(w(BinaryOp::Lt)), fragment(w(BinaryOp::GtEq))]).unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::ResidualNotMappedToOutput); 2]
    );
}

/// `items` joined to `events` on `k` and projected for [`distinct_fragment`],
/// with the join's build keys routed to the `events` reader. `items` is
/// filtered only when `items_predicate` is given.
fn joined_likes_fragment(items_predicate: Option<Expr>, events_predicate: Expr) -> QueryPlan {
    let mut source_ids = None;
    let mut plan = distinct_fragment(
        |builder| {
            let items = table_scan(builder, "items", None, |source| source);
            let build = match items_predicate {
                Some(predicate) => builder.add_filter(items, predicate).unwrap(),
                None => items,
            };
            let events = likes_scan(builder, None, |source| source);
            let filtered_events = builder.add_filter(events, events_predicate).unwrap();
            let joined = builder
                .add_join(
                    build,
                    filtered_events,
                    JoinSpec::new(
                        JoinType::Inner,
                        vec![JoinKey::new(ColumnRef::unqualified("k"), ColumnRef::unqualified("k"))],
                    )
                    .with_output_projection(vec![
                        JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("s"), "s"),
                        JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("j"), "j"),
                        JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("k"), "k"),
                    ]),
                    Arc::new(Schema::new(vec![
                        Field::new("s", DataType::Int64, false),
                        Field::new("j", DataType::Int32, false),
                        Field::new("k", DataType::Int64, false),
                    ])),
                )
                .unwrap();
            source_ids = Some((joined, build, events));
            joined
        },
        late_from_k(),
    );
    let (joined, build, events) = source_ids.unwrap();
    plan = plan.with_dynamic_filter_routes_for_test(vec![DynamicFilterRoute::new(
        DynamicFilterId::new(0),
        DynamicFilterProducer::HashJoinBuild { join_node_id: joined, build_side: JoinSide::Left, build_input: build },
        DynamicFilterConsumer::LocalParquetSource { source_node_id: events },
        vec!["k".to_string()],
        vec!["k".to_string()],
        vec![DynamicFilterKind::Range],
        DynamicFilterConsumerLowering::ReaderPredicate,
    )]);
    plan
}

#[test]
fn a_filter_only_one_copy_applies_leaves_the_producer_and_its_route_moves_to_the_scan() {
    let items = compare("j", BinaryOp::Gt, LiteralValue::Int32(0));
    let shared = extract_shared_resident_fragments(vec![
        joined_likes_fragment(Some(items), k_compare(BinaryOp::Lt, 10)),
        joined_likes_fragment(None, k_compare(BinaryOp::GtEq, 10)),
    ])
    .unwrap();

    let [producer] = shared.producers.as_slice() else { panic!("one producer for both copies: {shared:?}") };
    let plan = &producer.plan;
    let [route] = plan.dynamic_filter_routes() else { panic!("the join's route survives: {plan:?}") };
    let DynamicFilterProducer::HashJoinBuild { build_input, .. } = route.producer else {
        panic!("the route is a join build");
    };
    let DynamicFilterConsumer::LocalParquetSource { source_node_id } = route.consumer else {
        panic!("the route feeds the events reader");
    };
    // The items filter the first copy applies is gone; the join builds from
    // the scan, which is where the route now starts.
    assert!(matches!(plan.node(build_input).unwrap().kind, PlanNode::Source(_)), "{plan:?}");
    assert!(matches!(plan.node(source_node_id).unwrap().kind, PlanNode::Source(_)));
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Filter(_))).count(), 1);
    let residuals = shared.fragments.iter().map(|fragment| residual_and_resident_input(fragment).0).collect::<Vec<_>>();
    assert_eq!(
        residuals,
        [
            Expr::binary(
                compare("dst", BinaryOp::Gt, LiteralValue::Int64(0)),
                BinaryOp::And,
                Expr::Not(Box::new(Expr::column("late")))
            ),
            Expr::column("late"),
        ]
    );
}

/// [`distinct_fragment`] with `late = k >= 10` over a scan that selects by
/// reader predicate only.
fn reader_selected_fragment(row_groups: Option<Vec<usize>>, predicate: Expr, max_rows: Option<usize>) -> QueryPlan {
    distinct_fragment(
        |builder| {
            likes_scan(builder, row_groups, |source| source.with_reader_predicate(predicate).with_max_rows(max_rows))
        },
        late_from_k(),
    )
}

#[test]
fn copies_whose_reader_predicates_all_differ_have_no_covering_copy() {
    // With no filter above the scan, the producer reads one copy's reader
    // predicate unchanged, and neither `s = 5` nor `s = 7` selects both rows.
    let fragment = |value| reader_selected_fragment(None, compare("s", BinaryOp::Eq, LiteralValue::Int64(value)), None);

    let shared = extract_shared_resident_fragments(vec![fragment(5), fragment(7)]).unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::WidenedScanNotCovered); 2]
    );
}

#[test]
fn a_reader_selected_copy_holding_only_shared_conjuncts_covers_the_period_copies() {
    let range = |low, high| all_of(vec![k_compare(BinaryOp::GtEq, low), k_compare(BinaryOp::Lt, high)]);
    let covering = range(0, 20);
    let shared = extract_shared_resident_fragments(vec![
        reader_selected_fragment(Some(vec![0, 1]), range(0, 10), Some(32)),
        reader_selected_fragment(Some(vec![1, 2]), range(10, 20), Some(32)),
        reader_selected_fragment(None, covering.clone(), Some(32)),
    ])
    .unwrap();

    let [producer] = shared.producers.as_slice() else { panic!("one producer for the three copies: {shared:?}") };
    let PlanNode::Source(source) = &producer.plan.nodes()[0].kind else { panic!("the producer starts at its scan") };
    let SourceNode::LocalParquet(source) = source.as_ref() else { panic!("the producer scans Parquet") };
    assert_eq!((source.files[0].row_groups.as_ref(), source.reader_predicate.as_ref()), (None, Some(&covering)));
    assert_eq!(source.max_rows, Some(32));
    let residuals =
        shared.fragments[..2].iter().map(|fragment| residual_and_resident_input(fragment).0).collect::<Vec<_>>();
    assert_eq!(residuals, [Expr::Not(Box::new(Expr::column("late"))), Expr::column("late")]);
    assert!(matches!(shared.fragments[2].root_node().kind, PlanNode::Source(_)));
}

#[test]
fn a_selection_read_by_both_inputs_of_a_self_join_stays_separate() {
    // The builder interns the filtered scan, so both join inputs read one
    // selection. A residual on the group key `k` would restate the period for
    // the left input only, while the right input would read the widened rows.
    let fragment = |predicate: Option<Expr>| {
        let mut builder = QueryPlanBuilder::new();
        let scan = likes_scan(&mut builder, None, |source| source);
        let selected = match predicate {
            Some(predicate) => builder.add_filter(scan, predicate).unwrap(),
            None => scan,
        };
        let joined = builder
            .add_join(
                selected,
                selected,
                JoinSpec::new(
                    JoinType::Inner,
                    vec![JoinKey::new(ColumnRef::unqualified("v"), ColumnRef::unqualified("s"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("k"), "k"),
                    JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("s"), "s"),
                ]),
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, false),
                    Field::new("s", DataType::Int64, false),
                ])),
            )
            .unwrap();
        let distinct = builder
            .add_aggregate(
                joined,
                AggregateSpec::new(vec![Expr::column("k"), Expr::column("s")], Vec::new()),
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, false),
                    Field::new("s", DataType::Int64, false),
                ])),
            )
            .unwrap();
        builder.finish_gpu_resident(distinct).unwrap()
    };

    let shared = extract_shared_resident_fragments(vec![
        fragment(Some(k_compare(BinaryOp::Lt, 10))),
        fragment(Some(k_compare(BinaryOp::GtEq, 10))),
        fragment(None),
    ])
    .unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::ResidualNotMappedToOutput); 3]
    );
}

#[test]
fn a_covering_copy_whose_reader_predicate_drops_widened_rows_does_not_cover() {
    // Only the first copy's row groups cover the second's, but its reader
    // predicate `k < 10` drops the rows the second copy selects.
    let first = distinct_fragment(
        |builder| {
            let source = likes_scan(builder, None, |source| source.with_reader_predicate(k_compare(BinaryOp::Lt, 10)));
            builder.add_filter(source, compare("s", BinaryOp::Gt, LiteralValue::Int64(0))).unwrap()
        },
        late_from_k(),
    );
    let second = distinct_likes_fragment(
        Some(vec![0]),
        all_of(vec![compare("s", BinaryOp::Gt, LiteralValue::Int64(0)), k_compare(BinaryOp::GtEq, 10)]),
    );

    let shared = extract_shared_resident_fragments(vec![first, second]).unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::WidenedScanNotCovered); 2]
    );
}

#[test]
fn copies_under_a_scan_limit_stay_separate() {
    // The limit keeps the first rows the selection admits, so the first rows
    // of the widened selection are not each copy's first rows.
    let fragment = |predicate| {
        distinct_fragment(
            |builder| {
                let source = likes_scan(builder, None, |source| source.with_scan_limit(Some(4)));
                builder.add_filter(source, predicate).unwrap()
            },
            late_from_k(),
        )
    };

    let shared = extract_shared_resident_fragments(vec![
        fragment(k_compare(BinaryOp::Lt, 10)),
        fragment(k_compare(BinaryOp::GtEq, 10)),
    ])
    .unwrap();

    assert!(shared.producers.is_empty());
    assert_eq!(
        root_reasons(&shared),
        [(SharedResidentExtractionDecision::Rejected, SharedResidentExtractionReason::WidenedScanNotCovered); 2]
    );
}
