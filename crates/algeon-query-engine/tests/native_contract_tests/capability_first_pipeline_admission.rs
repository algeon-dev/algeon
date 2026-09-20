//! Capability-first bounded-pipeline admission contract tests, bound to the
//! engine's [`RapidsBackend`](query_engine::rapids_backend::RapidsBackend)
//! through [`rapids_analysis_context`].
//!
//! These exercise the same admission behavior as
//! `query_planning::admission::shared`'s former unit tests, through the
//! crate-public [`bounded_pipeline_decision`] entry point: the tested
//! function (`capability_first_bounded_pipeline_admission`) and its result
//! type are `pub(super)` to planning's `admission` module and require a
//! backend, which planning does not implement.

use query_engine::{
    admission::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts, bounded_pipeline_decision},
    capability::{CapabilityEnvelope, CapabilityRejectionOwner},
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
        DynamicFilterProducer, DynamicFilterRoute,
    },
    mode::StatisticalAggregateExecutionMode,
    pipeline::{AcceptedInputForm, PipelineTaskKind, ResidentDataForm, SinkBoundaryKind},
    plan::QueryPlan,
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn capability_admission_accepts_root_full_sort_left_semi_grouped_aggregate() {
    let plan = root_full_sort_left_semi_grouped_aggregate_plan();

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .expect("left-semi full-sort aggregate admission should not error");
    let admission = decision.admission().expect("join plan should produce capability admission evidence");

    assert_eq!(admission.envelope(), CapabilityEnvelope::RootFullSortJoinAggregate);
}

#[test]
fn capability_admission_accepts_gpu_resident_sink_for_join_pipeline() {
    let plan = root_full_sort_left_semi_grouped_aggregate_plan().clone_with_gpu_resident_sink();

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
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

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .expect("admission should classify the control cycle");

    assert!(decision.admission().is_none());
    let projection = decision.rejection_projection().expect("a rejected candidate carries its rejection projection");
    assert_eq!(projection.reason, "compiled_pipeline_dependency_cycle");
    assert_eq!(projection.owner, CapabilityRejectionOwner::Admission("compiled_pipeline_dependency_cycle"));
    assert_eq!(projection.boundary, "compiled_pipeline");
    assert!(projection.facts.contains_key("task_id"), "the cycle names the task it closed on: {:?}", projection);

    assert!(
        bounded_join_modeled_facts(&plan, rapids_analysis_context()).is_none(),
        "a rejected compiled pipeline must not emit modeled bounded-join facts"
    );
    let unsupported_facts = bounded_join_unsupported_shape_facts(&plan, rapids_analysis_context())
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

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
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
    assert!(
        decision.into_execution_admission().expect("newly admitted shape must not hard-error").is_some(),
        "admitted union-scope pipelines reach the compiled executor"
    );
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

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .expect("converged admission should not error");
    {
        let admission =
            decision.admission().expect("admission follows operator capability, not the surrounding plan shape");
        assert!(
            admission.pipeline().tasks().iter().all(|task| task.kind() != PipelineTaskKind::Union),
            "no union is required to enter the compiled pipeline"
        );
    }
    assert!(
        decision.into_execution_admission().expect("admitted shape must not hard-error").is_some(),
        "admitted source/unary pipelines reach the compiled executor"
    );
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

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .expect("non-join grouped aggregate admission should not error");
    {
        let admission =
            decision.admission().expect("supported scan/filter/grouped aggregate should use compiled admission");
        assert_eq!(admission.envelope(), CapabilityEnvelope::ComposedPipeline);
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
    assert!(
        decision.into_execution_admission().expect("admitted grouped aggregate must not hard-error").is_some(),
        "grouped aggregate reaches the compiled executor"
    );
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

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
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
        decision.into_execution_admission().expect("whole-input aggregate must not hard-error").is_some(),
        "whole-input aggregate reaches the compiled executor while retaining its source-stream contract"
    );
}

#[test]
fn capability_admission_keeps_window_composition_on_existing_path() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("literal window batch");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![batch])));
    let window = builder.add_window(
        source,
        WindowSpec::new(Vec::new(), Vec::new(), vec![WindowFunction::row_number("row_number")]),
        Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("row_number", DataType::UInt64, false),
        ])),
    );
    let plan = builder.finish_with_host_sink(window);

    let decision =
        bounded_pipeline_decision(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .expect("unsupported window composition should classify without an error");
    assert!(decision.admission().is_none(), "window composition remains outside the migrated contract");
    assert!(
        decision.into_execution_admission().expect("unsupported window composition must remain a fallback").is_none(),
        "unsupported non-join composition keeps the existing executor path"
    );
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
