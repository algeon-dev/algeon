//! Capability-first bounded-pipeline admission contract tests, bound to the
//! engine's [`RapidsBackend`](nexus_query_engine::rapids_backend::RapidsBackend)
//! through [`rapids_analysis_context`].
//!
//! These exercise the same admission behavior as
//! `nexus_query_planning::admission::shared`'s former unit tests, through the
//! crate-public [`bounded_pipeline_decision`] entry point: the tested
//! function (`capability_first_bounded_pipeline_admission`) and its result
//! type are `pub(super)` to planning's `admission` module and require a
//! backend, which planning does not implement.

use nexus_query_engine::{
    admission::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts, bounded_pipeline_decision},
    capability::{CapabilityEnvelope, CapabilityRejectionOwner},
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
        DynamicFilterProducer, DynamicFilterRoute,
    },
    mode::StatisticalAggregateExecutionMode,
    pipeline::{PipelineTaskKind, SinkBoundaryKind},
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
