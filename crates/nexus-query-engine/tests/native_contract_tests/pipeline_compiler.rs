use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use nexus_query_engine::{
    capability::{
        CapabilityAnalysis, CapabilityDecision, ContractQuality, ContractStatus, ReservationRequirement,
        ReservationRequirementScope, SubgraphInvariantKind,
    },
    contract_report::CapabilityContractReport,
    error::QueryError,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
    pipeline::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
        ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
        compile_pipeline,
    },
    plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanNode, QueryPlan, QueryPlanBuilder},
    runtime::ReservationKind,
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

use crate::bounded_join_admission::{
    AggregateOverJoinSubplanInputsVariant, ResidualResidentGroupedAggregateInputVariant,
    keyless_conditional_aggregate_over_join_subplan_inputs_plan,
    plans::{
        KeyedHashJoinResidentSubplanPipelineVariant, KeylessInputSubplanVariant,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant, aggregate_over_join_subplan_topk_plan,
        keyed_hash_join_resident_subplan_pipeline_plan, keyless_conditional_join_input_subplan_plan,
        resident_grouped_aggregate_input_post_join_aggregate_topk_plan,
        resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan,
        resident_grouped_aggregate_subquery_filter_plan, serial_aggregate_multi_join_spine_plan,
        serial_aggregate_single_join_spine_plan, single_join_post_root_aggregate_sort_limit_plan,
    },
    residual_resident_grouped_aggregate_input_plan,
};

#[test]
fn pipeline_model_renders_symbolic_tasks_edges_barriers_resources_and_sink() {
    let left_source = PipelineTaskId::new(0);
    let right_source = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let probe = PipelineTaskId::new(3);
    let materialize = PipelineTaskId::new(4);
    let sink_task = PipelineTaskId::new(5);

    let left_frame = ResidentResourceId::new(0);
    let right_frame = ResidentResourceId::new(1);
    let join_output = ResidentResourceId::new(2);

    let left_source_gate = ReservationGateId::new(0);
    let build_gate = ReservationGateId::new(1);
    let probe_gate = ReservationGateId::new(2);
    let join_output_gate = ReservationGateId::new(3);
    let sink_gate = ReservationGateId::new(4);

    let tasks = vec![
        PipelineTask::new(left_source, PipelineTaskKind::SourceRead)
            .with_output_resources(vec![left_frame])
            .with_gates(vec![left_source_gate]),
        PipelineTask::new(right_source, PipelineTaskKind::GpuResidentSource).with_output_resources(vec![right_frame]),
        PipelineTask::new(build, PipelineTaskKind::JoinBuild)
            .with_input_resources(vec![left_frame])
            .with_gates(vec![build_gate]),
        PipelineTask::new(probe, PipelineTaskKind::JoinProbe)
            .with_input_resources(vec![right_frame])
            .with_gates(vec![probe_gate]),
        PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize)
            .with_output_resources(vec![join_output])
            .with_gates(vec![join_output_gate]),
        PipelineTask::new(sink_task, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
    ];

    let edges = vec![
        PipelineEdge::new(PipelineEdgeId::new(0), left_source, build, PipelineEdgeKind::JoinBuildInput)
            .with_resources(vec![left_frame])
            .with_gates(vec![build_gate]),
        PipelineEdge::new(PipelineEdgeId::new(1), right_source, probe, PipelineEdgeKind::JoinProbeInput)
            .with_resources(vec![right_frame])
            .with_gates(vec![probe_gate]),
        PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![join_output])
            .with_gates(vec![join_output_gate]),
    ];

    let barriers = vec![
        PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::JoinBuildBeforeProbe)
            .with_upstream_tasks(vec![build])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
            .with_upstream_tasks(vec![sink_task])
            .with_gates(vec![sink_gate]),
    ];

    let resources = vec![
        ResidentResource::new(
            left_frame,
            left_source,
            vec![build],
            ResidentLifetime::BetweenTasks { producer: left_source },
        ),
        ResidentResource::new(
            right_frame,
            right_source,
            vec![probe],
            ResidentLifetime::BetweenTasks { producer: right_source },
        ),
        ResidentResource::new(
            join_output,
            materialize,
            vec![sink_task],
            ResidentLifetime::BetweenTasks { producer: materialize },
        ),
    ];

    let gates = vec![
        ReservationGate::new(
            left_source_gate,
            ReservationGateOwner::Task(left_source),
            ReservationKind::SourceRead,
            ReservationRequirementScope::Edge,
            "source_read",
            "source_contract",
            ContractStatus::Modeled,
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ),
        ReservationGate::new(
            build_gate,
            ReservationGateOwner::Edge(PipelineEdgeId::new(0)),
            ReservationKind::BuildSide,
            ReservationRequirementScope::Edge,
            "join_build_side",
            "query_engine_build_side_source_reservation",
            ContractStatus::Modeled,
            ContractQuality::ExactPreflight,
        ),
        ReservationGate::new(
            probe_gate,
            ReservationGateOwner::Edge(PipelineEdgeId::new(1)),
            ReservationKind::ProbeSide,
            ReservationRequirementScope::Edge,
            "join_probe_side",
            "query_engine_probe_side_source_reservation",
            ContractStatus::Modeled,
            ContractQuality::QueryEngineOwnedOrdering,
        ),
        ReservationGate::new(
            join_output_gate,
            ReservationGateOwner::Edge(PipelineEdgeId::new(4)),
            ReservationKind::JoinOutput,
            ReservationRequirementScope::Edge,
            "join_output",
            "dataframe_gather_output_reservation_size_bytes",
            ContractStatus::Modeled,
            ContractQuality::ExactPreflight,
        ),
        ReservationGate::new(
            sink_gate,
            ReservationGateOwner::SinkBoundary,
            ReservationKind::Materialization,
            ReservationRequirementScope::Edge,
            "host_materialization",
            "query_engine_explicit_sink_materialization_reservation",
            ContractStatus::Modeled,
            ContractQuality::QueryEngineOwnedBoundary,
        ),
    ];

    let pipeline = CompiledPipeline::new(
        tasks,
        edges,
        barriers,
        resources,
        gates,
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink_task).with_gates(vec![sink_gate]),
    );

    assert_eq!(
        pipeline.debug_render(),
        "CompiledPipeline\n  tasks:\n    t0 kind=source_read node=none inputs=none outputs=r0 gates=g0\n    t1 kind=gpu_resident_source node=none inputs=none outputs=r1 gates=none\n    t2 kind=join_build node=none inputs=r0 outputs=none gates=g1\n    t3 kind=join_probe node=none inputs=r1 outputs=none gates=g2\n    t4 kind=join_materialize node=none inputs=none outputs=r2 gates=g3\n    t5 kind=sink_materialize node=none inputs=r2 outputs=none gates=none\n  edges:\n    e0 t0 -> t2 kind=join_build_input resources=r0 gates=g1\n    e1 t1 -> t3 kind=join_probe_input resources=r1 gates=g2\n    e2 t2 -> t3 kind=control resources=none gates=none\n    e3 t3 -> t4 kind=control resources=none gates=none\n    e4 t4 -> t5 kind=sink_input resources=r2 gates=g3\n  barriers:\n    b0 kind=join_build_before_probe upstream=t2 downstream=t3 gates=none\n    b1 kind=sink_boundary upstream=t5 downstream=none gates=g4\n  resources:\n    r0 producer=t0 consumers=t2 lifetime=between_tasks(producer=t0,release=after_all_consumers)\n    r1 producer=t1 consumers=t3 lifetime=between_tasks(producer=t1,release=after_all_consumers)\n    r2 producer=t4 consumers=t5 lifetime=between_tasks(producer=t4,release=after_all_consumers)\n  gates:\n    g0 owner=task:t0 kind=source_read boundary=source_read contract=source_contract status=modeled quality=hard_budgeted_report_with_unresolved_preflight\n    g1 owner=edge:e0 kind=build_side boundary=join_build_side contract=query_engine_build_side_source_reservation status=modeled quality=exact_preflight\n    g2 owner=edge:e1 kind=probe_side boundary=join_probe_side contract=query_engine_probe_side_source_reservation status=modeled quality=query_engine_owned_ordering\n    g3 owner=edge:e4 kind=join_output boundary=join_output contract=dataframe_gather_output_reservation_size_bytes status=modeled quality=exact_preflight\n    g4 owner=sink_boundary kind=materialization boundary=host_materialization contract=query_engine_explicit_sink_materialization_reservation status=modeled quality=query_engine_owned_boundary\n  sink: kind=host_arrow_materialization input=t5 gates=g4\n"
    );
}

#[test]
fn reservation_gate_constructor_preserves_status_and_quality_as_separate_contract_fields() {
    let gate = ReservationGate::new(
        ReservationGateId::new(7),
        ReservationGateOwner::Barrier(PipelineBarrierId::new(3)),
        ReservationKind::FinalSort,
        ReservationRequirementScope::Edge,
        "join_pipeline_final_sort",
        "final_sort_contract",
        ContractStatus::Modeled,
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
    );

    assert_eq!(gate.id().to_string(), "g7");
    assert_eq!(gate.owner().to_string(), "barrier:b3");
    assert_eq!(gate.reservation_kind(), ReservationKind::FinalSort);
    assert_eq!(gate.status(), ContractStatus::Modeled);
    assert_eq!(gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
    assert_eq!(gate.requirement().scope, ReservationRequirementScope::Edge);
    assert_eq!(gate.requirement().memory_boundary, "join_pipeline_final_sort");
    assert_eq!(gate.requirement().required_contract, "final_sort_contract");
}

#[test]
fn reservation_gate_can_wrap_existing_capability_requirement_without_losing_blocked_status() {
    let requirement = ReservationRequirement {
        kind: ReservationKind::HashTable,
        scope: ReservationRequirementScope::Node,
        memory_boundary: "join_hash_table",
        required_contract: "hash_join_pre_build_reservation_size_bytes",
        status: ContractStatus::BlockedByLowerLevel,
        quality: ContractQuality::ExactPreflight,
    };

    let gate = ReservationGate::from_requirement(
        ReservationGateId::new(2),
        ReservationGateOwner::Task(PipelineTaskId::new(4)),
        requirement,
    );

    assert_eq!(gate.owner().to_string(), "task:t4");
    assert_eq!(gate.status(), ContractStatus::BlockedByLowerLevel);
    assert_eq!(gate.quality(), ContractQuality::ExactPreflight);
    assert_eq!(gate.requirement().required_contract, "hash_join_pre_build_reservation_size_bytes");
}

#[test]
fn compile_pipeline_root_join_uses_build_side_and_not_append_order_as_schedule_proof() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();

    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    pipeline.validate().expect("compiled root join pipeline should validate");
    assert_no_retired_execution_selection_terms("root join", &pipeline);
    assert_join_build_probe_dependencies("root join", &plan, &analysis, &pipeline);

    assert_eq!(
        pipeline.tasks().iter().map(|task| task.kind()).collect::<Vec<_>>(),
        vec![
            PipelineTaskKind::SourceRead,
            PipelineTaskKind::SourceRead,
            PipelineTaskKind::JoinBuild,
            PipelineTaskKind::JoinProbe,
            PipelineTaskKind::JoinMaterialize,
            PipelineTaskKind::SinkMaterialize,
        ]
    );

    let build = &pipeline.tasks()[2];
    let probe = &pipeline.tasks()[3];
    assert_eq!(build.input_resources(), &[ResidentResourceId::new(1)]);
    assert_eq!(probe.input_resources(), &[ResidentResourceId::new(0)]);

    assert!(pipeline.edges().iter().any(|edge| {
        edge.from() == PipelineTaskId::new(1)
            && edge.to() == PipelineTaskId::new(2)
            && edge.kind() == PipelineEdgeKind::JoinBuildInput
    }));
    assert!(pipeline.edges().iter().any(|edge| {
        edge.from() == PipelineTaskId::new(0)
            && edge.to() == PipelineTaskId::new(3)
            && edge.kind() == PipelineEdgeKind::JoinProbeInput
    }));
    assert!(pipeline.edges().iter().any(|edge| {
        edge.from() == PipelineTaskId::new(2)
            && edge.to() == PipelineTaskId::new(3)
            && edge.kind() == PipelineEdgeKind::Control
    }));

    let build_before_probe = pipeline
        .barriers()
        .iter()
        .find(|barrier| barrier.kind() == PipelineBarrierKind::JoinBuildBeforeProbe)
        .expect("join should compile an explicit build-before-probe barrier");
    assert_eq!(build_before_probe.upstream_tasks(), &[PipelineTaskId::new(2)]);
    assert_eq!(build_before_probe.downstream_tasks(), &[PipelineTaskId::new(3)]);
}

#[test]
fn compile_pipeline_non_join_aggregate_sort_limit_plan_has_blocking_barriers_in_dependency_order() {
    let plan = aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();

    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate sort limit plan should compile");
    pipeline.validate().expect("compiled aggregate pipeline should validate");

    assert_eq!(
        pipeline.tasks().iter().map(|task| task.kind()).collect::<Vec<_>>(),
        vec![
            PipelineTaskKind::SourceRead,
            PipelineTaskKind::Aggregate,
            PipelineTaskKind::FinalSort,
            PipelineTaskKind::Limit,
            PipelineTaskKind::SinkMaterialize,
        ]
    );

    let aggregate = pipeline
        .barriers()
        .iter()
        .find(|barrier| barrier.kind() == PipelineBarrierKind::AggregateBoundary)
        .expect("aggregate should compile a blocking boundary");
    assert_eq!(aggregate.upstream_tasks(), &[PipelineTaskId::new(1)]);
    assert_eq!(aggregate.downstream_tasks(), &[PipelineTaskId::new(2)]);

    let sort = pipeline
        .barriers()
        .iter()
        .find(|barrier| barrier.kind() == PipelineBarrierKind::SortBoundary)
        .expect("sort should compile a final-sort boundary");
    assert_eq!(sort.upstream_tasks(), &[PipelineTaskId::new(2)]);
    assert_eq!(sort.downstream_tasks(), &[PipelineTaskId::new(3)]);

    assert_eq!(pipeline.sink().kind(), SinkBoundaryKind::HostArrowMaterialization);
    assert_eq!(pipeline.sink().input(), PipelineTaskId::new(4));
}

#[test]
fn compile_pipeline_root_join_attaches_edge_task_and_sink_reservation_gates() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();

    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");

    let build_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::JoinBuild)
        .expect("join build task should exist");
    assert_gate(
        &pipeline,
        build_task.gates(),
        ReservationGateOwner::Task(build_task.id()),
        ReservationKind::HashTable,
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );

    let build_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.kind() == PipelineEdgeKind::JoinBuildInput)
        .expect("join build input edge should exist");
    assert_gate(
        &pipeline,
        build_edge.gates(),
        ReservationGateOwner::Edge(build_edge.id()),
        ReservationKind::SourceRead,
        ContractStatus::Modeled,
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
    );
    assert_gate(
        &pipeline,
        build_edge.gates(),
        ReservationGateOwner::Edge(build_edge.id()),
        ReservationKind::BuildSide,
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );

    let probe_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.kind() == PipelineEdgeKind::JoinProbeInput)
        .expect("join probe input edge should exist");
    assert_gate(
        &pipeline,
        probe_edge.gates(),
        ReservationGateOwner::Edge(probe_edge.id()),
        ReservationKind::ProbeSide,
        ContractStatus::Modeled,
        ContractQuality::QueryEngineOwnedOrdering,
    );

    let sink_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.kind() == PipelineEdgeKind::SinkInput)
        .expect("host sink input edge should exist");
    assert_gate(
        &pipeline,
        sink_edge.gates(),
        ReservationGateOwner::Edge(sink_edge.id()),
        ReservationKind::GatherMap,
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );
    assert_gate(
        &pipeline,
        sink_edge.gates(),
        ReservationGateOwner::Edge(sink_edge.id()),
        ReservationKind::JoinOutput,
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );
    assert_gate(
        &pipeline,
        pipeline.sink().gates(),
        ReservationGateOwner::SinkBoundary,
        ReservationKind::Materialization,
        ContractStatus::Modeled,
        ContractQuality::QueryEngineOwnedBoundary,
    );
}

#[test]
fn compile_pipeline_aggregate_and_sort_preserve_unresolved_preflight_gate_quality() {
    let plan = aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();

    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate sort limit plan should compile");

    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Aggregate)
        .expect("aggregate task should exist");
    assert_gate(
        &pipeline,
        aggregate_task.gates(),
        ReservationGateOwner::Task(aggregate_task.id()),
        ReservationKind::OperatorOutput,
        ContractStatus::Modeled,
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
    );

    let source_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.from() == PipelineTaskId::new(0) && edge.to() == aggregate_task.id())
        .expect("source-to-aggregate edge should exist");
    assert_gate(
        &pipeline,
        source_edge.gates(),
        ReservationGateOwner::Edge(source_edge.id()),
        ReservationKind::SourceRead,
        ContractStatus::Modeled,
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
    );

    let sort_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::FinalSort)
        .expect("final sort task should exist");
    let sort_edge =
        pipeline.edges().iter().find(|edge| edge.to() == sort_task.id()).expect("sort input edge should exist");
    assert_gate(
        &pipeline,
        sort_edge.gates(),
        ReservationGateOwner::Edge(sort_edge.id()),
        ReservationKind::FinalSort,
        ContractStatus::Modeled,
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
    );
}

#[test]
fn compile_pipeline_residual_join_preserves_observed_retained_state_quality() {
    let plan = residual_root_join_plan();
    let analysis = plan.capability_analysis();

    let pipeline = compile_pipeline(&plan, &analysis).expect("residual root join should compile");

    let materialize_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::JoinMaterialize)
        .expect("join materialization task should exist");
    assert_gate(
        &pipeline,
        materialize_task.gates(),
        ReservationGateOwner::Task(materialize_task.id()),
        ReservationKind::MixedSizeData,
        ContractStatus::Modeled,
        ContractQuality::ObservedRetainedState,
    );
}

#[test]
fn compile_pipeline_single_join_post_root_pipeline_has_post_join_aggregate_and_sort_barriers() {
    let (_plan, _analysis, pipeline) = compile_admitted_join_case(
        "single-join post-root",
        serial_aggregate_single_join_spine_plan(),
        CapabilityAnalysis::admits_single_join_post_root_pipeline,
    );

    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 1);
    assert_task_kind_count_at_least(&pipeline, PipelineTaskKind::Aggregate, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_barrier_kind_count_at_least(&pipeline, PipelineBarrierKind::AggregateBoundary, 2);
    assert_barrier_kind_count(&pipeline, PipelineBarrierKind::SortBoundary, 1);
    assert_barrier_appears_before(&pipeline, PipelineBarrierKind::AggregateBoundary, PipelineBarrierKind::SortBoundary);
}

#[test]
fn compile_pipeline_single_join_post_root_aggregate_sort_limit_graph_is_visible_before_d2_execution() {
    let (_plan, _analysis, pipeline) = compile_admitted_join_case(
        "single-join post-root aggregate sort limit",
        single_join_post_root_aggregate_sort_limit_plan(),
        CapabilityAnalysis::admits_single_join_post_root_pipeline,
    );

    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_barrier_kind_count(&pipeline, PipelineBarrierKind::AggregateBoundary, 1);
    assert_barrier_kind_count(&pipeline, PipelineBarrierKind::SortBoundary, 1);
    assert_barrier_appears_before(
        &pipeline,
        PipelineBarrierKind::JoinMaterialization,
        PipelineBarrierKind::AggregateBoundary,
    );
    assert_barrier_appears_before(&pipeline, PipelineBarrierKind::AggregateBoundary, PipelineBarrierKind::SortBoundary);

    let rendered = pipeline.debug_render();
    for expected in [
        "kind=aggregate",
        "kind=final_sort",
        "kind=limit",
        "kind=aggregate_boundary",
        "kind=sort_boundary",
        "kind=sink_boundary",
    ] {
        assert!(rendered.contains(expected), "compiled graph rendering should contain {expected}");
    }
}

#[test]
fn compile_pipeline_multi_join_paths_make_every_build_probe_constraint_explicit() {
    for (label, plan) in [
        ("serial aggregate multi-join", serial_aggregate_multi_join_spine_plan()),
        ("mixed residual multi-join", crate::bounded_join_admission::admitted_mixed_residual_multi_join_plan()),
    ] {
        let (plan, analysis, pipeline) =
            compile_admitted_join_case(label, plan, CapabilityAnalysis::admits_closed_multi_join_subgraph);

        assert!(join_node_ids(&plan).len() >= 2, "{label} should exercise more than one join");
        assert_eq!(
            analysis
                .invariants
                .iter()
                .filter(|invariant| matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { .. }))
                .count(),
            join_node_ids(&plan).len(),
            "{label} should solve one build-before-probe invariant for each join"
        );
        assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, join_node_ids(&plan).len());
        assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, join_node_ids(&plan).len());
        assert_barrier_kind_count(&pipeline, PipelineBarrierKind::JoinBuildBeforeProbe, join_node_ids(&plan).len());
    }
}

#[test]
fn compile_pipeline_keyless_join_subplan_cases_keep_subplan_work_as_dependencies() {
    let (_plan, _analysis, input_subplan) = compile_admitted_join_case(
        "keyless input-subplan",
        keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted),
        CapabilityAnalysis::admits_keyless_conditional_join_input_subplan_pipeline,
    );
    assert_task_kind_count_at_least(&input_subplan, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count_at_least(&input_subplan, PipelineTaskKind::Unary, 1);
    assert_task_kind_count_at_least(&input_subplan, PipelineTaskKind::JoinBuild, 2);

    let (_plan, _analysis, aggregate_over_inputs) = compile_admitted_join_case(
        "keyless aggregate-over-join-subplan inputs",
        keyless_conditional_aggregate_over_join_subplan_inputs_plan(AggregateOverJoinSubplanInputsVariant::Admitted),
        CapabilityAnalysis::admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
    );
    assert_task_kind_count_at_least(&aggregate_over_inputs, PipelineTaskKind::Aggregate, 2);
    assert_task_kind_count_at_least(&aggregate_over_inputs, PipelineTaskKind::JoinBuild, 5);
    assert_barrier_kind_count_at_least(&aggregate_over_inputs, PipelineBarrierKind::JoinBuildBeforeProbe, 5);
}

#[test]
fn compile_pipeline_topk_bounded_paths_compile_aggregate_sort_limit_structure() {
    for (label, plan, admits, expected_aggregates) in [
        (
            "aggregate-over-join TopK",
            aggregate_over_join_subplan_topk_plan(),
            CapabilityAnalysis::admits_multi_join_aggregate_over_join_subplan_topk_pipeline
                as fn(&CapabilityAnalysis) -> bool,
            1,
        ),
        (
            "resident grouped-aggregate TopK",
            resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate TopK direct aggregate input",
            resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::DirectPostJoinAggregateInput,
            ),
            CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate TopK filter-wrapped direct aggregate input",
            resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::FilterWrappedGroupedInputDirectPostJoinAggregateInput,
            ),
            CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate TopK preaggregated filter-wrapped direct aggregate input",
            resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput,
            ),
            CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&CapabilityAnalysis) -> bool,
            3,
        ),
    ] {
        let (_plan, _analysis, pipeline) = compile_admitted_join_case(label, plan, admits);

        assert_task_kind_count_at_least(&pipeline, PipelineTaskKind::Aggregate, expected_aggregates);
        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_barrier_kind_count_at_least(&pipeline, PipelineBarrierKind::AggregateBoundary, expected_aggregates);
        assert_barrier_kind_count(&pipeline, PipelineBarrierKind::SortBoundary, 1);
        assert_barrier_appears_before(
            &pipeline,
            PipelineBarrierKind::AggregateBoundary,
            PipelineBarrierKind::SortBoundary,
        );
    }
}

#[test]
fn compile_pipeline_resident_grouped_aggregate_subquery_filter_keeps_join_and_aggregate_boundaries() {
    let (_plan, _analysis, pipeline) = compile_admitted_join_case(
        "resident grouped-aggregate subquery filter",
        resident_grouped_aggregate_subquery_filter_plan(),
        CapabilityAnalysis::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
    );

    assert_task_kind_count_at_least(&pipeline, PipelineTaskKind::JoinBuild, 3);
    assert_task_kind_count_at_least(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_barrier_kind_count_at_least(&pipeline, PipelineBarrierKind::JoinMaterialization, 3);
    assert_barrier_kind_count_at_least(&pipeline, PipelineBarrierKind::AggregateBoundary, 1);
}

#[test]
fn compile_pipeline_mixed_residual_resident_grouped_aggregate_branch_feeds_root_build_side() {
    let (plan, _analysis, pipeline) = compile_admitted_join_case(
        "mixed residual resident grouped aggregate input",
        residual_resident_grouped_aggregate_input_plan(ResidualResidentGroupedAggregateInputVariant::BlockedFamily),
        CapabilityAnalysis::admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
    );

    let root_join = residual_join_node_id(&plan);
    let build_task = task_for_plan_node(&pipeline, root_join, PipelineTaskKind::JoinBuild);
    let probe_task = task_for_plan_node(&pipeline, root_join, PipelineTaskKind::JoinProbe);
    let aggregate_branch = pipeline
        .edges()
        .iter()
        .find(|edge| edge.to() == build_task.id() && edge.kind() == PipelineEdgeKind::JoinBuildInput)
        .expect("root residual join build side should be fed by the grouped aggregate branch");
    let producer = pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == aggregate_branch.from())
        .expect("aggregate branch producer task should exist");
    assert!(
        matches!(producer.kind(), PipelineTaskKind::Aggregate | PipelineTaskKind::Unary),
        "grouped aggregate branch should reach the root build side as resident pipeline work"
    );
    assert!(pipeline.edges().iter().any(|edge| {
        edge.from() == build_task.id() && edge.to() == probe_task.id() && edge.kind() == PipelineEdgeKind::Control
    }));
}

#[test]
fn compile_pipeline_keyed_hash_resident_subplan_uses_shared_resource_identity_for_fanout() {
    let (_plan, _analysis, pipeline) = compile_admitted_join_case(
        "keyed hash resident subplan",
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily),
        CapabilityAnalysis::admits_keyed_hash_join_resident_subplan_pipeline,
    );

    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
        .collect::<Vec<_>>();
    assert!(!shared_edges.is_empty(), "shared resident subplan should compile explicit shared-producer edges");
    let shared_resource =
        shared_edges[0].resources().first().copied().expect("shared producer edge should carry the producer resource");
    assert!(
        shared_edges.iter().all(|edge| edge.resources() == [shared_resource]),
        "shared fan-out should be represented by the same resident resource identity on all shared edges"
    );
    let resource = pipeline
        .resources()
        .iter()
        .find(|resource| resource.id() == shared_resource)
        .expect("shared resident resource should exist");
    assert!(
        resource.consumers().len() >= 2,
        "shared resident producer should dominate multiple consumers through one resource"
    );
}

#[test]
fn compiled_pipeline_validate_allows_shared_resident_resource_to_feed_sink_boundary() {
    let producer = PipelineTaskId::new(0);
    let first_consumer = PipelineTaskId::new(1);
    let second_consumer = PipelineTaskId::new(2);
    let sink = PipelineTaskId::new(3);
    let frame = ResidentResourceId::new(0);
    let pipeline = CompiledPipeline::new(
        vec![
            PipelineTask::new(producer, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(first_consumer, PipelineTaskKind::JoinBuild).with_input_resources(vec![frame]),
            PipelineTask::new(second_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
            PipelineTask::new(sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), producer, first_consumer, PipelineEdgeKind::JoinBuildInput)
                .with_resources(vec![frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(1),
                producer,
                second_consumer,
                PipelineEdgeKind::SharedResidentProducer,
            )
            .with_resources(vec![frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), producer, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![producer])
                .with_downstream_tasks(vec![first_consumer, second_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![ResidentResource::new(
            frame,
            producer,
            vec![first_consumer, second_consumer, sink],
            ResidentLifetime::BetweenTasks { producer },
        )],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    );

    pipeline.validate().expect("sink input edge should cover a shared resource sink consumer");
}

#[test]
fn compile_pipeline_rejects_local_capability_rejection_before_building_graph() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("", schema)));
    let plan = builder.finish_gpu_resident(source);
    let analysis = plan.capability_analysis();

    let error = compile_pipeline(&plan, &analysis).expect_err("rejected source capability should not compile");

    assert_eq!(error.code().as_str(), "unsupported_plan_shape");
    assert_eq!(error.fact_value("capability_scope"), Some("node"));
    assert_eq!(error.fact_value("capability_reason"), Some("unsupported_gpu_resident_source_capability"));
    assert_eq!(error.fact_value("node_id"), Some("n0"));
}

#[test]
fn compile_pipeline_reports_exact_source_read_preflight_when_lower_layer_covers_source() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/exact-source-preflight.parquet").with_row_group_rows(vec![10])],
        Arc::clone(&schema),
        vec!["payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("exact source preflight plan should compile");

    let source_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("compiled pipeline should carry a source-read gate");
    assert_eq!(source_gate.required_contract(), "parquet_source_read_exact_preflight");
    assert_eq!(source_gate.quality(), ContractQuality::ExactPreflight);

    let report = CapabilityContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .expect("exact source preflight should be valid strict contract evidence");
    let source_row = report
        .gate_rows()
        .iter()
        .find(|row| row.reservation_kind == ReservationKind::SourceRead)
        .expect("strict report should contain source-read row");
    assert_eq!(source_row.required_contract, "parquet_source_read_exact_preflight");
    assert_eq!(source_row.contract_quality, ContractQuality::ExactPreflight);
    assert_eq!(source_row.unresolved_preflight_gap, None);
}

#[test]
fn compile_pipeline_keeps_source_read_fallback_contract_for_uncovered_source() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/fallback-source-preflight.parquet").with_row_group_rows(vec![10])],
        Arc::clone(&schema),
        vec!["payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("fallback source preflight plan should compile");

    let source_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("compiled pipeline should carry a source-read gate");
    assert_eq!(source_gate.required_contract(), "source_contract");
    assert_eq!(source_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
}

#[test]
fn compiled_pipeline_validate_rejects_resource_consumer_without_carrying_edge() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.edges[4] = PipelineEdge::new(
            PipelineEdgeId::new(4),
            PipelineTaskId::new(4),
            PipelineTaskId::new(5),
            PipelineEdgeKind::SinkInput,
        );
    });

    let error = pipeline.validate().expect_err("resource consumer without a carrying edge should be invalid");

    assert_pipeline_validation_error(&error, "pipeline resident resource consumer has no carrying dependency edge");
    assert_eq!(error.fact_value("resource_id"), Some("r2"));
    assert_eq!(error.fact_value("producer_task"), Some("t4"));
    assert_eq!(error.fact_value("consumer_task"), Some("t5"));
}

#[test]
fn compiled_pipeline_validate_rejects_task_input_resource_without_recorded_consumer() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.edges[4] = PipelineEdge::new(
            PipelineEdgeId::new(4),
            PipelineTaskId::new(4),
            PipelineTaskId::new(5),
            PipelineEdgeKind::Control,
        );
        parts.resources[2] = ResidentResource::new(
            ResidentResourceId::new(2),
            PipelineTaskId::new(4),
            vec![],
            ResidentLifetime::TaskLocal { task: PipelineTaskId::new(4) },
        );
    });

    let error = pipeline.validate().expect_err("task input resource should be a recorded consumer");

    assert_pipeline_validation_error(&error, "pipeline task input resource consumer is not recorded on resource");
    assert_eq!(error.fact_value("resource_id"), Some("r2"));
    assert_eq!(error.fact_value("producer_task"), Some("t4"));
    assert_eq!(error.fact_value("consumer_task"), Some("t5"));
}

#[test]
fn compiled_pipeline_validate_rejects_join_probe_without_matching_build_dependency() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.edges[2] = PipelineEdge::new(
            PipelineEdgeId::new(2),
            PipelineTaskId::new(0),
            PipelineTaskId::new(3),
            PipelineEdgeKind::Control,
        );
    });

    let error = pipeline.validate().expect_err("join probe must depend on its build task");

    assert_pipeline_validation_error(&error, "pipeline join probe task does not depend on its build task");
    assert_eq!(error.fact_value("task_id"), Some("t3"));
}

#[test]
fn compiled_pipeline_validate_rejects_barrier_without_upstream_dominance() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.barriers.push(
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![PipelineTaskId::new(0)])
                .with_downstream_tasks(vec![PipelineTaskId::new(2)]),
        );
    });

    let error = pipeline.validate().expect_err("barrier upstream should dominate downstream tasks");

    assert_pipeline_validation_error(&error, "pipeline barrier upstream tasks do not dominate downstream task");
    assert_eq!(error.fact_value("barrier_id"), Some("b3"));
    assert_eq!(error.fact_value("downstream_task"), Some("t2"));
}

#[test]
fn compiled_pipeline_validate_rejects_unresolved_preflight_gate_mislabeled_as_exact() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.gates[0] = ReservationGate::new(
            ReservationGateId::new(0),
            ReservationGateOwner::Edge(PipelineEdgeId::new(0)),
            ReservationKind::SourceRead,
            ReservationRequirementScope::Edge,
            "source_read",
            "source_contract",
            ContractStatus::Modeled,
            ContractQuality::ExactPreflight,
        );
    });

    let error = pipeline.validate().expect_err("hard-budgeted source read gate should keep unresolved quality");

    assert_pipeline_validation_error(
        &error,
        "pipeline unresolved-preflight gate is not labeled with unresolved-preflight quality",
    );
    assert_eq!(error.fact_value("gate_id"), Some("g0"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("contract_quality"), Some("exact_preflight"));
}

#[test]
fn compiled_pipeline_validate_rejects_hard_budgeted_quality_without_tracked_gap() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.gates[0] = ReservationGate::new(
            ReservationGateId::new(0),
            ReservationGateOwner::Edge(PipelineEdgeId::new(0)),
            ReservationKind::SourceRead,
            ReservationRequirementScope::Edge,
            "source_read",
            "libcudf_decoded_output_preflight_contract_after_api_lands",
            ContractStatus::Modeled,
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        );
    });

    let error = pipeline
        .validate()
        .expect_err("F.6 should reject unresolved quality after the placeholder contract is replaced");

    assert_pipeline_validation_error(
        &error,
        "pipeline hard-budgeted gate is missing lower-level unresolved preflight gap",
    );
    assert_eq!(error.fact_value("gate_id"), Some("g0"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(
        error.fact_value("required_contract"),
        Some("libcudf_decoded_output_preflight_contract_after_api_lands")
    );
}

#[test]
fn compiled_pipeline_validate_rejects_gate_owner_scope_mismatch() {
    let pipeline = strict_validation_fixture_with(|parts| {
        parts.tasks[1] = PipelineTask::new(PipelineTaskId::new(1), PipelineTaskKind::SourceRead)
            .with_output_resources(vec![ResidentResourceId::new(1)])
            .with_gates(vec![ReservationGateId::new(0)]);
        parts.edges[0] = PipelineEdge::new(
            PipelineEdgeId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(2),
            PipelineEdgeKind::JoinBuildInput,
        )
        .with_resources(vec![ResidentResourceId::new(1)]);
        parts.gates[0] = ReservationGate::new(
            ReservationGateId::new(0),
            ReservationGateOwner::Task(PipelineTaskId::new(1)),
            ReservationKind::SourceRead,
            ReservationRequirementScope::Edge,
            "source_read",
            "source_contract",
            ContractStatus::Modeled,
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        );
    });

    let error = pipeline.validate().expect_err("edge-scoped gate should not be owned by a task");

    assert_pipeline_validation_error(&error, "pipeline reservation gate owner does not match requirement scope");
    assert_eq!(error.fact_value("gate_id"), Some("g0"));
    assert_eq!(error.fact_value("gate_owner"), Some("task:t1"));
    assert_eq!(error.fact_value("requirement_scope"), Some("edge"));
}

fn assert_gate(
    pipeline: &CompiledPipeline,
    gate_ids: &[ReservationGateId],
    owner: ReservationGateOwner,
    kind: ReservationKind,
    status: ContractStatus,
    quality: ContractQuality,
) {
    let gate = gate_ids
        .iter()
        .filter_map(|gate_id| pipeline.gates().iter().find(|gate| gate.id() == *gate_id))
        .find(|gate| gate.owner() == owner && gate.reservation_kind() == kind)
        .unwrap_or_else(|| panic!("missing {kind:?} gate for owner {owner:?}"));

    assert_eq!(gate.status(), status);
    assert_eq!(gate.quality(), quality);
    assert_eq!(gate.requirement().status, status);
    assert_eq!(gate.requirement().quality, quality);
}

fn assert_pipeline_validation_error(error: &QueryError, expected_message: &str) {
    assert_eq!(error.code().as_str(), "invalid_plan");
    assert_eq!(error.operation(), "nexus_query_engine.pipeline.validate");
    assert_eq!(error.message(), expected_message);
    assert!(!error.to_string().contains("BoundedJoinExecutionFamily"));
    assert!(!error.to_string().contains("BoundedJoinExecutionPath"));
}

#[derive(Debug, Clone)]
struct StrictValidationFixtureParts {
    tasks: Vec<PipelineTask>,
    edges: Vec<PipelineEdge>,
    barriers: Vec<PipelineBarrier>,
    resources: Vec<ResidentResource>,
    gates: Vec<ReservationGate>,
    sink: SinkBoundary,
}

fn strict_validation_fixture_with(update: impl FnOnce(&mut StrictValidationFixtureParts)) -> CompiledPipeline {
    let mut parts = strict_validation_fixture_parts();
    update(&mut parts);
    CompiledPipeline::new(parts.tasks, parts.edges, parts.barriers, parts.resources, parts.gates, parts.sink)
}

fn strict_validation_fixture_parts() -> StrictValidationFixtureParts {
    let left_source = PipelineTaskId::new(0);
    let right_source = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let probe = PipelineTaskId::new(3);
    let materialize = PipelineTaskId::new(4);
    let sink_task = PipelineTaskId::new(5);

    let left_frame = ResidentResourceId::new(0);
    let right_frame = ResidentResourceId::new(1);
    let join_output = ResidentResourceId::new(2);
    let source_gate = ReservationGateId::new(0);
    let sink_gate = ReservationGateId::new(1);

    let tasks = vec![
        PipelineTask::new(left_source, PipelineTaskKind::SourceRead).with_output_resources(vec![left_frame]),
        PipelineTask::new(right_source, PipelineTaskKind::SourceRead).with_output_resources(vec![right_frame]),
        PipelineTask::new(build, PipelineTaskKind::JoinBuild).with_input_resources(vec![right_frame]),
        PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![left_frame]),
        PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
        PipelineTask::new(sink_task, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
    ];

    let edges = vec![
        PipelineEdge::new(PipelineEdgeId::new(0), right_source, build, PipelineEdgeKind::JoinBuildInput)
            .with_resources(vec![right_frame])
            .with_gates(vec![source_gate]),
        PipelineEdge::new(PipelineEdgeId::new(1), left_source, probe, PipelineEdgeKind::JoinProbeInput)
            .with_resources(vec![left_frame]),
        PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![join_output]),
    ];

    let barriers = vec![
        PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::JoinBuildBeforeProbe)
            .with_upstream_tasks(vec![build])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::JoinMaterialization)
            .with_upstream_tasks(vec![probe])
            .with_downstream_tasks(vec![materialize]),
        PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::SinkBoundary)
            .with_upstream_tasks(vec![sink_task]),
    ];

    let resources = vec![
        ResidentResource::new(
            left_frame,
            left_source,
            vec![probe],
            ResidentLifetime::BetweenTasks { producer: left_source },
        ),
        ResidentResource::new(
            right_frame,
            right_source,
            vec![build],
            ResidentLifetime::BetweenTasks { producer: right_source },
        ),
        ResidentResource::new(
            join_output,
            materialize,
            vec![sink_task],
            ResidentLifetime::BetweenTasks { producer: materialize },
        ),
    ];

    let gates = vec![
        ReservationGate::new(
            source_gate,
            ReservationGateOwner::Edge(PipelineEdgeId::new(0)),
            ReservationKind::SourceRead,
            ReservationRequirementScope::Edge,
            "source_read",
            "source_contract",
            ContractStatus::Modeled,
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ),
        ReservationGate::new(
            sink_gate,
            ReservationGateOwner::SinkBoundary,
            ReservationKind::Materialization,
            ReservationRequirementScope::Edge,
            "host_materialization",
            "query_engine_explicit_sink_materialization_reservation",
            ContractStatus::Modeled,
            ContractQuality::QueryEngineOwnedBoundary,
        ),
    ];

    let sink = SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink_task).with_gates(vec![sink_gate]);
    StrictValidationFixtureParts { tasks, edges, barriers, resources, gates, sink }
}

fn compile_admitted_join_case(
    label: &str,
    plan: QueryPlan,
    admits: fn(&CapabilityAnalysis) -> bool,
) -> (QueryPlan, CapabilityAnalysis, CompiledPipeline) {
    let analysis = plan.capability_analysis();
    assert!(!analysis.has_rejections(), "{label}: capability analysis should not contain local rejections");
    assert!(admits(&analysis), "{label}: expected capability admission to be satisfied");

    let pipeline =
        compile_pipeline(&plan, &analysis).unwrap_or_else(|error| panic!("{label}: compile failed: {error}"));
    pipeline.validate().unwrap_or_else(|error| panic!("{label}: compiled pipeline should validate: {error}"));
    assert_no_retired_execution_selection_terms(label, &pipeline);
    assert_join_build_probe_dependencies(label, &plan, &analysis, &pipeline);
    (plan, analysis, pipeline)
}

fn assert_no_retired_execution_selection_terms(label: &str, pipeline: &CompiledPipeline) {
    let rendered = pipeline.debug_render();
    for forbidden in ["BoundedJoinExecutionPath", "BoundedJoinExecutionFamily"] {
        assert!(
            !rendered.contains(forbidden),
            "{label}: compiled pipeline rendering must not name retired execution selection type {forbidden}"
        );
    }
}

fn assert_join_build_probe_dependencies(
    label: &str,
    plan: &QueryPlan,
    analysis: &CapabilityAnalysis,
    pipeline: &CompiledPipeline,
) {
    for join_id in join_node_ids(plan) {
        let invariant = analysis
            .invariants
            .iter()
            .find(|invariant| {
                matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { join_id: candidate } if candidate == join_id)
            })
            .unwrap_or_else(|| panic!("{label}: join {join_id} should have a build-before-probe invariant"));
        assert_eq!(
            invariant.decision,
            CapabilityDecision::Satisfied,
            "{label}: build-before-probe invariant for {join_id} should be satisfied"
        );

        let build = task_for_plan_node(pipeline, join_id, PipelineTaskKind::JoinBuild);
        let probe = task_for_plan_node(pipeline, join_id, PipelineTaskKind::JoinProbe);
        let materialize = task_for_plan_node(pipeline, join_id, PipelineTaskKind::JoinMaterialize);
        assert!(
            pipeline.edges().iter().any(|edge| {
                edge.from() == build.id() && edge.to() == probe.id() && edge.kind() == PipelineEdgeKind::Control
            }),
            "{label}: join {join_id} should have explicit build -> probe control dependency"
        );
        assert!(
            pipeline.edges().iter().any(|edge| {
                edge.from() == probe.id() && edge.to() == materialize.id() && edge.kind() == PipelineEdgeKind::Control
            }),
            "{label}: join {join_id} should have explicit probe -> materialize control dependency"
        );
        assert!(
            pipeline.barriers().iter().any(|barrier| {
                barrier.kind() == PipelineBarrierKind::JoinBuildBeforeProbe
                    && barrier.upstream_tasks() == [build.id()]
                    && barrier.downstream_tasks() == [probe.id()]
            }),
            "{label}: join {join_id} should have explicit build-before-probe barrier"
        );
    }
}

fn task_for_plan_node(pipeline: &CompiledPipeline, node_id: NodeId, kind: PipelineTaskKind) -> &PipelineTask {
    pipeline
        .tasks()
        .iter()
        .find(|task| task.plan_node() == Some(node_id) && task.kind() == kind)
        .unwrap_or_else(|| panic!("missing {kind:?} task for plan node {node_id}"))
}

fn join_node_ids(plan: &QueryPlan) -> Vec<NodeId> {
    plan.nodes().iter().filter_map(|node| matches!(node.kind, PlanNode::Join(_)).then_some(node.id)).collect()
}

fn residual_join_node_id(plan: &QueryPlan) -> NodeId {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) if join.spec.residual_predicate.is_some() => Some(node.id),
            _ => None,
        })
        .expect("plan should contain a residual join")
}

fn assert_task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    assert_eq!(task_kind_count(pipeline, kind), expected, "unexpected count for {kind:?} tasks");
}

fn assert_task_kind_count_at_least(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    let actual = task_kind_count(pipeline, kind);
    assert!(actual >= expected, "expected at least {expected} {kind:?} tasks, found {actual}");
}

fn task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> usize {
    pipeline.tasks().iter().filter(|task| task.kind() == kind).count()
}

fn assert_barrier_kind_count(pipeline: &CompiledPipeline, kind: PipelineBarrierKind, expected: usize) {
    assert_eq!(barrier_kind_count(pipeline, kind), expected, "unexpected count for {kind:?} barriers");
}

fn assert_barrier_kind_count_at_least(pipeline: &CompiledPipeline, kind: PipelineBarrierKind, expected: usize) {
    let actual = barrier_kind_count(pipeline, kind);
    assert!(actual >= expected, "expected at least {expected} {kind:?} barriers, found {actual}");
}

fn barrier_kind_count(pipeline: &CompiledPipeline, kind: PipelineBarrierKind) -> usize {
    pipeline.barriers().iter().filter(|barrier| barrier.kind() == kind).count()
}

fn assert_barrier_appears_before(pipeline: &CompiledPipeline, before: PipelineBarrierKind, after: PipelineBarrierKind) {
    let last_before = pipeline
        .barriers()
        .iter()
        .rposition(|barrier| barrier.kind() == before)
        .unwrap_or_else(|| panic!("missing {before:?} barrier"));
    let first_after = pipeline
        .barriers()
        .iter()
        .position(|barrier| barrier.kind() == after)
        .unwrap_or_else(|| panic!("missing {after:?} barrier"));
    assert!(
        last_before < first_after,
        "{before:?} barriers should appear before the first {after:?} barrier in dependency order"
    );
}

fn root_join_plan() -> QueryPlan {
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

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/pipeline-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/pipeline-right.parquet",
        right_schema,
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
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn residual_root_join_plan() -> QueryPlan {
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

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/pipeline-residual-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/pipeline-residual-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_value", "left_value"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_value", "right_value"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn aggregate_sort_limit_plan() -> QueryPlan {
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/pipeline-aggregate.parquet",
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("v_count"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}
