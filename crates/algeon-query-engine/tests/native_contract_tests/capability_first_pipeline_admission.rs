//! Capability-first bounded-pipeline admission contract tests, bound to the
//! engine's [`RapidsBackend`](query_engine::rapids_backend::RapidsBackend)
//! through [`rapids_analysis_context`].
//!
//! These exercise the same admission behavior as
//! `query_planning::admission::shared`'s former unit tests, through the
//! crate-public [`admit`] entry point: the tested
//! function (`capability_first_bounded_pipeline_admission`) and its result
//! type are `pub(super)` to planning's `admission` module and require a
//! backend, which planning does not implement.

use query_engine::{
    admission::{admit, bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    capability::CapabilityRejectionOwner,
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
        DynamicFilterProducer, DynamicFilterRoute,
    },
    expr::GroupingSetsSpec,
    mode::StatisticalAggregateExecutionMode,
    pipeline::{AcceptedInputForm, PipelineTaskKind, ResidentDataForm, SinkBoundaryKind},
    plan::QueryPlan,
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn capability_admission_accepts_root_full_sort_left_semi_grouped_aggregate() {
    let plan = root_full_sort_left_semi_grouped_aggregate_plan();

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("left-semi full-sort aggregate admission should not error");

    assert!(decision.admission().is_some(), "join plan should produce capability admission evidence");
}

#[test]
fn capability_admission_accepts_gpu_resident_sink_for_join_pipeline() {
    let plan = root_full_sort_left_semi_grouped_aggregate_plan().clone_with_gpu_resident_sink();

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("gpu-resident sink admission should not error");
    let admission = decision.admission().expect("join plan should produce capability admission evidence");
    let pipeline = admission.pipeline();
    assert_eq!(pipeline.sink().kind(), SinkBoundaryKind::GpuResidentOutput);
    assert!(
        pipeline.tasks().iter().all(|task| task.kind() != PipelineTaskKind::SinkMaterialize),
        "gpu-resident sink must not compile a host materialization task"
    );
}

#[test]
fn capability_admission_reports_dynamic_filter_control_cycle_as_a_dependency_cycle() {
    let plan = dynamic_filter_control_cycle_plan();

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("admission should classify the control cycle");

    assert!(decision.admission().is_none());
    let projection = decision.rejection_projection().expect("a rejected candidate carries its rejection projection");
    assert_eq!(projection.reason, "compiled_pipeline_dependency_cycle");
    assert_eq!(projection.owner, CapabilityRejectionOwner::Admission("compiled_pipeline_dependency_cycle"));
    assert_eq!(projection.boundary, "compiled_pipeline");
    assert!(projection.facts.contains_key("task_id"), "the cycle names the task it closed on: {:?}", projection);

    assert!(
        bounded_join_modeled_facts(&plan, &decision, rapids_analysis_context()).is_none(),
        "a rejected compiled pipeline must not emit modeled bounded-join facts"
    );
    let unsupported_facts = bounded_join_unsupported_shape_facts(&plan, &decision, rapids_analysis_context())
        .expect("a rejected compiled pipeline emits unsupported bounded-join facts");
    assert_eq!(
        unsupported_facts.get("bounded_join_rejection_reason").map(String::as_str),
        Some("compiled_pipeline_dependency_cycle")
    );
    assert_eq!(unsupported_facts.get("bounded_join_rejection_boundary").map(String::as_str), Some("compiled_pipeline"));
    assert_eq!(
        unsupported_facts.get("bounded_join_rejection_category").map(String::as_str),
        Some("unsupported_compiled_pipeline_envelope")
    );
}

#[test]
fn capability_admission_accepts_independent_source_unary_branches_behind_union() {
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("left literal batch");
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3])) as ArrayRef])
        .expect("right literal batch");
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let left = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let right = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("union-scope admission should not error");
    {
        let admission =
            decision.admission().expect("independent source/unary branches behind union use compiled admission");
        let pipeline = admission.pipeline();
        assert!(pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::Union));
        assert_eq!(
            pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::LiteralBatchSource).count(),
            2,
            "both independent branches compile into the same pipeline"
        );
    }
    assert!(decision.admission().is_some(), "admitted union-scope pipelines reach the compiled executor");
}

#[test]
fn capability_admission_accepts_source_unary_without_requiring_a_union() {
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef])
        .expect("literal batch");
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
    let projection = builder.add_projection(
        source,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let plan = builder.finish_with_host_sink(projection);

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("converged admission should not error");
    {
        let admission =
            decision.admission().expect("admission follows operator capability, not the surrounding plan shape");
        assert!(
            admission.pipeline().tasks().iter().all(|task| task.kind() != PipelineTaskKind::Union),
            "no union is required to enter the compiled pipeline"
        );
    }
    assert!(decision.admission().is_some(), "admitted source/unary pipelines reach the compiled executor");
}

#[test]
fn capability_admission_accepts_non_join_scan_filter_grouped_aggregate_topk() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("measure", DataType::Int32, false),
    ]));
    let batches = vec![
        RecordBatch::try_new(
            Arc::clone(&input_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                Arc::new(Int32Array::from(vec![4, 7, 9])) as ArrayRef,
            ],
        )
        .expect("literal aggregate batch"),
    ];
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("row_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), batches)));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("measure"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&input_schema),
    );
    let aggregate = builder.add_aggregate(
        filtered,
        AggregateSpec::new(
            vec![Expr::column("group_key")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("measure"), "row_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("group_key"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), Arc::clone(&aggregate_schema));
    let plan = builder.finish_with_host_sink(limit);

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("non-join grouped aggregate admission should not error");
    {
        let admission =
            decision.admission().expect("supported scan/filter/grouped aggregate should use compiled admission");
        let pipeline = admission.pipeline();
        assert_eq!(pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::Aggregate).count(), 1);
        assert_eq!(pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::FinalSort).count(), 1);
        assert_eq!(pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::Limit).count(), 1);
        assert!(pipeline.tasks().iter().all(|task| {
            !matches!(
                task.kind(),
                PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe | PipelineTaskKind::JoinMaterialize
            )
        }));
    }
    assert!(decision.admission().is_some(), "grouped aggregate reaches the compiled executor");
}

#[test]
fn capability_admission_accepts_non_join_distinct_aggregate_with_source_stream_contract() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/capability-first-distinct.parquet",
        Arc::clone(&input_schema),
        vec!["value".to_string()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "distinct_values")
                    .with_distinct(true),
            ],
        ),
        Arc::new(Schema::new(vec![Field::new("distinct_values", DataType::Int64, false)])),
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("whole-input aggregate admission should not error");
    {
        let admission = decision.admission().expect("supported whole-input aggregate should be admitted");
        let pipeline = admission.pipeline();
        let aggregate_task = pipeline
            .tasks()
            .iter()
            .find(|task| task.kind() == PipelineTaskKind::Aggregate)
            .expect("compiled pipeline has an aggregate task");
        assert_eq!(aggregate_task.accepted_input_form(), AcceptedInputForm::FrameStreamOrSourceStream);
        let source_task = pipeline
            .tasks()
            .iter()
            .find(|task| task.kind() == PipelineTaskKind::SourceRead)
            .expect("compiled pipeline has a source task");
        let source_resource = pipeline
            .resources()
            .iter()
            .find(|resource| resource.producer() == source_task.id())
            .expect("source output resource");
        assert_eq!(source_resource.data_form(), ResidentDataForm::SourceStream);
    }
    assert!(
        decision.admission().is_some(),
        "whole-input aggregate reaches the compiled executor while retaining its source-stream contract"
    );
}

/// A row-number window over a literal source, ordered by `value` when `order_by`.
fn literal_row_number_plan(order_by: bool) -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("literal window batch");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![batch])));
    let order_by = if order_by { vec![SortExpr::new(Expr::column("value"), false, false)] } else { Vec::new() };
    let window = builder.add_window(
        source,
        WindowSpec::new(Vec::new(), order_by, vec![WindowFunction::row_number("row_number")]),
        Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("row_number", DataType::UInt64, false),
        ])),
    );
    builder.finish_with_host_sink(window)
}

#[test]
fn capability_admission_admits_window_that_depends_on_no_input_order() {
    let plan = literal_row_number_plan(false);

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("window admission should not error");
    let admission = decision.admission().expect("a row number without ORDER BY is admitted");
    assert!(admission.pipeline().tasks().iter().any(|task| task.kind() == PipelineTaskKind::Window));
}

#[test]
fn capability_admission_rejects_row_number_over_unordered_input() {
    // A literal source records no row ordering, so ranks under ORDER BY would
    // follow arrival order.
    let plan = literal_row_number_plan(true);

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("window admission should not error");
    let projection = decision.rejection_projection().expect("a row number over unordered input is rejected");
    assert_eq!(projection.reason, "window_input_ordering_not_established");
}

#[test]
fn capability_admission_rejects_rank_over_a_materialized_partition_key() {
    // DataFusion sorts a TPC-DS q86 rank input on the partition expression.
    // Lowering materializes it once as a sort key, drops that key after the
    // sort, and materializes it again as the window key, so no recorded
    // ordering covers the column the rank partitions on.
    let plan = rollup_rank_over_materialized_partition_key_plan();

    let decision = admit(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("window admission should not error");
    let projection = decision.rejection_projection().expect("no compiled contract proves the rank input order");
    assert_eq!(projection.reason, "window_input_ordering_not_established");
}

/// TPC-DS q86 after aggregate-join pushdown: a rollup over a joined detail
/// pre-aggregate, ranked within `CASE WHEN __grouping_id = 0 THEN category END`.
fn rollup_rank_over_materialized_partition_key_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("amount", DataType::Int64, true),
    ]));
    let detail_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("detail_sales", DataType::Int64, true),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("detail_sales", DataType::Int64, true),
    ]));
    let aggregate_fields = vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
    ];
    let aggregate_schema = Arc::new(Schema::new(aggregate_fields.clone()));
    let mut sort_keyed_fields = aggregate_fields.clone();
    sort_keyed_fields.push(Field::new("__algeon_native_sort_key_0", DataType::Utf8, true));
    let sort_keyed_schema = Arc::new(Schema::new(sort_keyed_fields));
    let mut keyed_fields = aggregate_fields;
    keyed_fields.push(Field::new("__algeon_native_window_key_0", DataType::Utf8, true));
    let keyed_schema = Arc::new(Schema::new(keyed_fields.clone()));
    let mut ranked_fields = keyed_fields;
    ranked_fields.push(Field::new("rk", DataType::UInt64, false));
    let ranked_schema = Arc::new(Schema::new(ranked_fields));
    let partition_key = || {
        Expr::case(
            vec![CaseWhen::new(
                Expr::binary(Expr::column("__grouping_id"), BinaryOp::Eq, Expr::literal(LiteralValue::UInt32(0))),
                Expr::column("category"),
            )],
            Expr::literal(LiteralValue::Null(DataType::Utf8)),
        )
    };

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rollup-rank-materialized-key-fact.parquet",
        fact_schema,
        vec!["id".to_string(), "category".to_string(), "amount".to_string()],
    )));
    let detail = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            vec![Expr::column("id"), Expr::column("category")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "detail_sales")],
        ),
        detail_schema,
    );
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rollup-rank-materialized-key-dim.parquet",
        dim_schema,
        vec!["id".to_string()],
    )));
    let join = builder.add_join(
        detail,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "category", "category"), "category"),
            JoinOutputColumn::left(ColumnRef::input(0, "detail_sales", "detail_sales"), "detail_sales"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let rollup = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("category")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("detail_sales"), "sumsales")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
        Arc::clone(&aggregate_schema),
    );
    let sort_keyed = builder.add_projection(
        rollup,
        projection_outputs(
            vec![
                Expr::column("category"),
                Expr::column("__grouping_id"),
                Expr::column("sumsales"),
                partition_key().alias("__algeon_native_sort_key_0"),
            ],
            Arc::clone(&sort_keyed_schema),
        ),
    );
    let rank_input_sort = builder.add_sort(
        sort_keyed,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("__algeon_native_sort_key_0"), false, false),
            SortExpr::new(Expr::column("sumsales"), true, true),
        ]),
        sort_keyed_schema,
    );
    let sorted = builder.add_projection(
        rank_input_sort,
        projection_outputs(
            vec![Expr::column("category"), Expr::column("__grouping_id"), Expr::column("sumsales")],
            aggregate_schema,
        ),
    );
    let keyed = builder.add_projection(
        sorted,
        projection_outputs(
            vec![
                Expr::column("category"),
                Expr::column("__grouping_id"),
                Expr::column("sumsales"),
                partition_key().alias("__algeon_native_window_key_0"),
            ],
            keyed_schema,
        ),
    );
    let window = builder.add_window(
        keyed,
        WindowSpec::new(
            vec![Expr::column("__algeon_native_window_key_0")],
            vec![SortExpr::new(Expr::column("sumsales"), true, true)],
            vec![WindowFunction::rank("rk")],
        ),
        Arc::clone(&ranked_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))),
        Arc::clone(&ranked_schema),
    );
    let final_sort = builder.add_sort(
        filter,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category"), false, true),
            SortExpr::new(Expr::column("sumsales"), true, false),
        ]),
        Arc::clone(&ranked_schema),
    );
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100), ranked_schema);
    builder.finish_with_host_sink(limit)
}

#[test]
fn admitted_plan_accepts_footer_pruning_that_only_narrows_row_groups() {
    let ctx = rapids_analysis_context();
    let decision = admit(&footer_pruned_scan_plan(vec![0, 1, 2], 5), StatisticalAggregateExecutionMode::default(), ctx)
        .expect("a filtered scan is decidable");
    let admitted = decision.admission().expect("a filtered scan is admitted");

    let pruned = footer_pruned_scan_plan(vec![1], 5);
    let refined =
        admitted.with_pruned_sources(pruned.clone(), ctx).expect("dropping row groups keeps the plan admitted");
    assert_eq!(refined.plan(), &pruned);

    let changed = footer_pruned_scan_plan(vec![1], 6);
    let error = admitted.with_pruned_sources(changed, ctx).expect_err("a changed filter is not a footer refinement");
    assert_eq!(error.code(), QueryErrorCode::InternalInvariantViolation);
    assert_eq!(error.fact_value("changed"), Some("node:n1"));
}

/// A scan of one Parquet file reading `row_groups`, filtered on `value > threshold`.
fn footer_pruned_scan_plan(row_groups: Vec<usize>, threshold: i32) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let file =
        LocalParquetFile::new("/tmp/footer-pruned-scan.parquet").with_file_size_bytes(4096).with_row_groups(row_groups);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["value".to_string()],
    )));
    let filter = builder.add_filter(
        source,
        Expr::binary(Expr::column("value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(threshold))),
        schema,
    );
    builder.finish_with_host_sink(filter)
}

fn root_full_sort_left_semi_grouped_aggregate_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_key", DataType::Int64, false),
        Field::new("priority", DataType::Utf8, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int64, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("priority", DataType::Utf8, false),
        Field::new("order_count", DataType::UInt64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/orders.parquet",
        Arc::clone(&left_schema),
        vec!["order_key".to_string(), "priority".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/lineitem.parquet",
        Arc::clone(&right_schema),
        vec!["order_key".to_string()],
    )));
    let semi = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_key", "order_key"),
                ColumnRef::input(1, "order_key", "order_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_key", "order_key"), "order_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "priority", "priority"), "priority"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&left_schema),
    );
    let aggregate = builder.add_aggregate(
        semi,
        AggregateSpec::new(
            vec![Expr::column("priority")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_key"), "order_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        vec![
            ProjectionOutput::new(Expr::column("priority"), Field::new("priority", DataType::Utf8, false)),
            ProjectionOutput::new(Expr::column("order_count"), Field::new("order_count", DataType::UInt64, false)),
        ],
    );
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("priority"), false, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
}

fn dynamic_filter_control_cycle_plan() -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![Field::new("shared_key", DataType::Int32, false)]));
    let build_schema = Arc::new(Schema::new(vec![Field::new("build_key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let shared_probe = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/shared-probe.parquet",
        Arc::clone(&source_schema),
        vec!["shared_key".to_string()],
    )));
    let build_seed = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/build-seed.parquet",
        Arc::clone(&build_schema),
        vec!["build_key".to_string()],
    )));
    let build_subtree = builder.add_join(
        build_seed,
        shared_probe,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "build_key", "build_key"),
                ColumnRef::input(1, "shared_key", "shared_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "build_key", "build_key"),
            "build_key",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&build_schema),
    );
    let root = builder.add_join(
        shared_probe,
        build_subtree,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "shared_key", "shared_key"),
                ColumnRef::input(1, "build_key", "build_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "shared_key", "shared_key"),
            "shared_key",
        )])
        .with_build_side(JoinSide::Right),
        source_schema,
    );
    let plan = builder.finish_with_host_sink(root);
    plan.with_dynamic_filter_routes(vec![DynamicFilterRoute::new(
        DynamicFilterId::new(0),
        DynamicFilterProducer::HashJoinBuild {
            join_node_id: root,
            build_side: JoinSide::Right,
            build_input: build_subtree,
        },
        DynamicFilterConsumer::LocalParquetSource { source_node_id: shared_probe },
        vec!["build_key".to_string()],
        vec!["shared_key".to_string()],
        vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains],
        DynamicFilterConsumerLowering::ReaderPredicate,
    )])
}
