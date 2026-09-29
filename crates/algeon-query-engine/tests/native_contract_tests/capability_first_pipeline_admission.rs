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

use arrow::array::Int64Array;
use query_engine::{
    admission::{AdmissionDecision, admit, bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    capability::{Reason, RejectionOwner, UnsupportedPart},
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
        DynamicFilterProducer, DynamicFilterRoute,
    },
    error::Result as QueryEngineResult,
    expr::GroupingSetsSpec,
    mode::StatisticalAggregateExecutionMode,
    pipeline::{
        AcceptedInputForm, AggregateInputHints, PipelineTaskKind, ResidentDataForm, SinkBoundaryKind, SourceOutput,
        SourceReadPlan, TaskPhysical,
    },
    plan::QueryPlan,
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn capability_admission_accepts_root_full_sort_left_semi_grouped_aggregate() {
    let plan = root_full_sort_left_semi_grouped_aggregate_plan();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("left-semi full-sort aggregate admission should not error");

    assert!(decision.admission().is_some(), "join plan should produce capability admission evidence");
}

#[test]
fn capability_admission_accepts_gpu_resident_sink_for_join_pipeline() {
    let plan = root_full_sort_left_semi_grouped_aggregate_plan().clone_with_gpu_resident_sink();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
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

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("admission should classify the control cycle");

    assert!(decision.admission().is_none());
    let projection = decision.rejection().expect("a rejected candidate carries its rejection");
    assert_eq!(projection.reason, Reason::CompiledPipelineDependencyCycle);
    assert_eq!(projection.owner, RejectionOwner::Plan);
    assert_eq!(projection.boundary(), "compiled_pipeline");
    assert!(projection.facts.contains_key("task_id"), "the cycle names the task it closed on: {:?}", projection);

    assert!(
        bounded_join_modeled_facts(&plan, &decision, rapids_analysis_context(PlanningPolicy::default())).is_none(),
        "a rejected compiled pipeline must not emit modeled bounded-join facts"
    );
    let unsupported_facts =
        bounded_join_unsupported_shape_facts(&plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
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
    let left = builder
        .add_projection(
            left,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let right = builder
        .add_projection(
            right,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let plan = builder.finish_with_host_sink(union).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
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
    let projection = builder
        .add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
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
    let filtered = builder
        .add_filter(
            source,
            Expr::binary(Expr::column("measure"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            filtered,
            AggregateSpec::new(
                vec![Expr::column("group_key")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("measure"), "row_count")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("group_key"), false, false)]))
        .unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(10)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
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
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "distinct_values")
                        .with_distinct(true),
                ],
            ),
            Arc::new(Schema::new(vec![Field::new("distinct_values", DataType::Int64, false)])),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
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
        let no_stages = AggregateInputHints { adjacent_stages: false, grouped_string_key: false };
        assert_eq!(
            source_task.physical(),
            &TaskPhysical::SourceRead(SourceReadPlan {
                output: SourceOutput::AggregateInputStream { adjacent_stages: false, chunk_hints: no_stages },
                dynamic_filter_routes: false,
                prefetch_eligible: false,
            }),
            "the aggregate pulls the deferred source read chunk by chunk"
        );
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
    let window = builder
        .add_window(source, WindowSpec::new(Vec::new(), order_by, vec![WindowFunction::row_number("row_number")]))
        .unwrap();
    builder.finish_with_host_sink(window).unwrap()
}

#[test]
fn capability_admission_admits_window_that_depends_on_no_input_order() {
    let plan = literal_row_number_plan(false);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("window admission should not error");
    let admission = decision.admission().expect("a row number without ORDER BY is admitted");
    assert!(admission.pipeline().tasks().iter().any(|task| task.kind() == PipelineTaskKind::Window));
}

#[test]
fn capability_admission_rejects_row_number_over_unordered_input() {
    // A literal source records no row ordering, so ranks under ORDER BY would
    // follow arrival order.
    let plan = literal_row_number_plan(true);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("window admission should not error");
    let projection = decision.rejection().expect("a row number over unordered input is rejected");
    assert_eq!(projection.reason, Reason::WindowInputOrderingNotEstablished);
}

/// A row-number window over a literal `(p, o)` source, partitioned by `p`,
/// ordered by `o` when `order_by`, over input sorted by `input_sort` when given.
fn partitioned_row_number_plan(input_sort: Option<Vec<SortExpr>>, order_by: bool) -> QueryPlan {
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("p", DataType::Int32, false), Field::new("o", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
        ],
    )
    .expect("literal window batch");
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![batch])));
    let input = match input_sort {
        Some(keys) => builder.add_sort(source, SortSpec::new(keys)).unwrap(),
        None => source,
    };
    let order_by = if order_by { vec![SortExpr::new(Expr::column("o"), false, false)] } else { Vec::new() };
    let window = builder
        .add_window(
            input,
            WindowSpec::new(vec![Expr::column("p")], order_by, vec![WindowFunction::row_number("row_number")]),
        )
        .unwrap();
    builder.finish_with_host_sink(window).unwrap()
}

fn admit_default(plan: &QueryPlan) -> QueryEngineResult<AdmissionDecision> {
    admit(
        plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
}

#[test]
fn capability_admission_rejects_row_number_whose_input_is_not_ordered_by_its_partition_key() {
    // Sorted by `o` alone, the rows of each `p` partition are interleaved, so
    // positional numbering would give [1, 1, 3, 3, 5, 5].
    let plan = partitioned_row_number_plan(Some(vec![SortExpr::new(Expr::column("o"), false, false)]), true);

    let decision = admit_default(&plan).expect("window admission should not error");
    let projection = decision.rejection().expect("an input not grouped by partition is rejected");
    assert_eq!(projection.reason, Reason::WindowInputOrderingNotEstablished);

    let plan = partitioned_row_number_plan(
        Some(vec![SortExpr::new(Expr::column("p"), false, false), SortExpr::new(Expr::column("o"), false, false)]),
        true,
    );
    assert!(admit_default(&plan).expect("window admission should not error").admission().is_some());
}

#[test]
fn capability_admission_rejects_partitioned_row_number_without_order_over_unordered_input() {
    // Without ORDER BY any numbering within a partition is valid, but it is
    // still positional: the partition's rows must be adjacent.
    let plan = partitioned_row_number_plan(None, false);

    let decision = admit_default(&plan).expect("window admission should not error");
    let projection = decision.rejection().expect("an ungrouped partition input is rejected");
    assert_eq!(projection.reason, Reason::WindowInputOrderingNotEstablished);
}

#[test]
fn capability_admission_refuses_a_plan_reading_a_column_its_input_does_not_produce() {
    // The scalar aggregate outputs only `v`; the projection above it reads `k`.
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef, Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef],
    )
    .expect("literal batch");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![batch])));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")]),
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])),
        )
        .unwrap();
    let projection_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let projection =
        builder.add_projection(aggregate, projection_outputs(vec![Expr::column("k")], projection_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = admit_default(&plan).expect_err("a plan reading a missing column has no admission decision");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
}

#[test]
fn capability_admission_rejects_a_type_the_backend_does_not_support_with_its_validation_code() {
    // A timezone-aware timestamp is a valid Arrow group key that the native
    // aggregate cannot key on. Validation reports it as unsupported, which is
    // a rejection a frontend may fall back from, not a malformed plan.
    let timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("ts", timestamp.clone(), false),
        Field::new("v", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/capability-first-timezone-group-key.parquet",
        Arc::clone(&input_schema),
        vec!["ts".to_string(), "v".to_string()],
    )));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("ts")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            Arc::new(Schema::new(vec![Field::new("ts", timestamp, false), Field::new("total", DataType::Int64, true)])),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let decision = admit_default(&plan).expect("an unsupported group key type is a decision");
    let projection = decision.rejection().expect("an unsupported group key type is rejected");
    assert_eq!(projection.reason, Reason::PlanValidation(UnsupportedPart::DataType));
    assert_eq!(projection.facts.get("validation_code").map(String::as_str), Some("unsupported_type"));
    let AdmissionDecision::Rejected(rejection) = &decision else { unreachable!("checked above") };
    let error = rejection.execution_error(&plan);
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(Some(error.operation()), projection.facts.get("validation_operation").map(String::as_str));
}

#[test]
fn capability_admission_reports_a_malformed_node_behind_an_unsupported_one() {
    // The aggregate keys on a timezone-aware timestamp the backend cannot run;
    // the projection above it reads a column the aggregate does not produce.
    // The plan is malformed, so it has no admission decision even though the
    // unsupported node comes first.
    let timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("ts", timestamp.clone(), false),
        Field::new("v", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/capability-first-malformed-behind-unsupported.parquet",
        Arc::clone(&input_schema),
        vec!["ts".to_string(), "v".to_string()],
    )));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("ts")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            Arc::new(Schema::new(vec![Field::new("ts", timestamp, false), Field::new("total", DataType::Int64, true)])),
        )
        .unwrap();
    let projection_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let projection =
        builder.add_projection(aggregate, projection_outputs(vec![Expr::column("v")], projection_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = admit_default(&plan).expect_err("a malformed plan has no admission decision");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("column"), Some("v"));
}

#[test]
fn capability_admission_rejects_rank_over_a_materialized_partition_key() {
    // DataFusion sorts a TPC-DS q86 rank input on the partition expression.
    // Lowering materializes it once as a sort key, drops that key after the
    // sort, and materializes it again as the window key, so no recorded
    // ordering covers the column the rank partitions on.
    let plan = rollup_rank_over_materialized_partition_key_plan();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("window admission should not error");
    let projection = decision.rejection().expect("no compiled contract proves the rank input order");
    assert_eq!(projection.reason, Reason::WindowInputOrderingNotEstablished);
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
    let detail = builder
        .add_aggregate(
            fact,
            AggregateSpec::new(
                vec![Expr::column("id"), Expr::column("category")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "detail_sales")],
            ),
            detail_schema,
        )
        .unwrap();
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rollup-rank-materialized-key-dim.parquet",
        dim_schema,
        vec!["id".to_string()],
    )));
    let join = builder
        .add_join(
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
        )
        .unwrap();
    let rollup = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("category")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("detail_sales"), "sumsales")],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort_keyed = builder
        .add_projection(
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
        )
        .unwrap();
    let rank_input_sort = builder
        .add_sort(
            sort_keyed,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("__algeon_native_sort_key_0"), false, false),
                SortExpr::new(Expr::column("sumsales"), true, true),
            ]),
        )
        .unwrap();
    let sorted = builder
        .add_projection(
            rank_input_sort,
            projection_outputs(
                vec![Expr::column("category"), Expr::column("__grouping_id"), Expr::column("sumsales")],
                aggregate_schema,
            ),
        )
        .unwrap();
    let keyed = builder
        .add_projection(
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
        )
        .unwrap();
    let window = builder
        .add_window(
            keyed,
            WindowSpec::new(
                vec![Expr::column("__algeon_native_window_key_0")],
                vec![SortExpr::new(Expr::column("sumsales"), true, true)],
                vec![WindowFunction::rank("rk")],
            ),
        )
        .unwrap();
    let filter = builder
        .add_filter(window, Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))))
        .unwrap();
    let final_sort = builder
        .add_sort(
            filter,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("category"), false, true),
                SortExpr::new(Expr::column("sumsales"), true, false),
            ]),
        )
        .unwrap();
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

#[test]
fn admitted_plan_accepts_footer_pruning_that_only_narrows_row_groups() {
    let ctx = rapids_analysis_context(PlanningPolicy::default());
    let decision = admit(
        &footer_pruned_scan_plan(vec![0, 1, 2], 5),
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        ctx,
    )
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

#[test]
fn admitted_plan_refuses_a_pruned_source_selecting_a_row_group_its_file_does_not_have() {
    let ctx = rapids_analysis_context(PlanningPolicy::default());
    let decision = admit(
        &footer_pruned_scan_plan(vec![0, 1, 2], 5),
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        ctx,
    )
    .expect("a filtered scan is decidable");
    let admitted = decision.admission().expect("a filtered scan is admitted");

    // Footer metadata reports one row group, yet the pruned selection keeps
    // row group 2: the resolved source is malformed, not unsupported.
    let broken = file_scan_plan(
        LocalParquetFile::new("/tmp/footer-pruned-scan.parquet")
            .with_file_size_bytes(4096)
            .with_row_groups(vec![2])
            .with_row_group_rows(vec![8]),
        5,
    );

    let error = admitted.with_pruned_sources(broken, ctx).expect_err("a row group past the footer is malformed");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
}

/// A scan of one Parquet file reading `row_groups`, filtered on `value > threshold`.
fn footer_pruned_scan_plan(row_groups: Vec<usize>, threshold: i32) -> QueryPlan {
    file_scan_plan(
        LocalParquetFile::new("/tmp/footer-pruned-scan.parquet").with_file_size_bytes(4096).with_row_groups(row_groups),
        threshold,
    )
}

/// A scan of `file` filtered on `value > threshold`.
fn file_scan_plan(file: LocalParquetFile, threshold: i32) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["value".to_string()],
    )));
    let filter = builder
        .add_filter(
            source,
            Expr::binary(Expr::column("value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(threshold))),
        )
        .unwrap();
    builder.finish_with_host_sink(filter).unwrap()
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
    let semi = builder
        .add_join(
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
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            semi,
            AggregateSpec::new(
                vec![Expr::column("priority")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_key"), "order_count")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let projection = builder
        .add_projection(
            aggregate,
            vec![
                ProjectionOutput::new(Expr::column("priority"), Field::new("priority", DataType::Utf8, false)),
                ProjectionOutput::new(Expr::column("order_count"), Field::new("order_count", DataType::UInt64, false)),
            ],
        )
        .unwrap();
    let sort = builder
        .add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column("priority"), false, false)]))
        .unwrap();
    builder.finish_with_host_sink(sort).unwrap()
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
    let build_subtree = builder
        .add_join(
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
        )
        .unwrap();
    let root = builder
        .add_join(
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
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(root).unwrap();
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
