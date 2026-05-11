use cudf_query_engine::{
    CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
    PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
    ReservationGateOwner, ReservationKind, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary,
    SinkBoundaryKind,
    capability::{ContractQuality, ContractStatus, ReservationRequirement, ReservationRequirementScope},
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
            ResidentLifetime::BetweenTasks { producer: left_source, last_consumer: build },
        ),
        ResidentResource::new(
            right_frame,
            right_source,
            vec![probe],
            ResidentLifetime::BetweenTasks { producer: right_source, last_consumer: probe },
        ),
        ResidentResource::new(
            join_output,
            materialize,
            vec![sink_task],
            ResidentLifetime::BetweenTasks { producer: materialize, last_consumer: sink_task },
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
        "CompiledPipeline\n  tasks:\n    t0 kind=source_read node=none inputs=none outputs=r0 gates=g0\n    t1 kind=gpu_resident_source node=none inputs=none outputs=r1 gates=none\n    t2 kind=join_build node=none inputs=r0 outputs=none gates=g1\n    t3 kind=join_probe node=none inputs=r1 outputs=none gates=g2\n    t4 kind=join_materialize node=none inputs=none outputs=r2 gates=g3\n    t5 kind=sink_materialize node=none inputs=r2 outputs=none gates=none\n  edges:\n    e0 t0 -> t2 kind=join_build_input resources=r0 gates=g1\n    e1 t1 -> t3 kind=join_probe_input resources=r1 gates=g2\n    e2 t2 -> t3 kind=control resources=none gates=none\n    e3 t3 -> t4 kind=control resources=none gates=none\n    e4 t4 -> t5 kind=sink_input resources=r2 gates=g3\n  barriers:\n    b0 kind=join_build_before_probe upstream=t2 downstream=t3 gates=none\n    b1 kind=sink_boundary upstream=t5 downstream=none gates=g4\n  resources:\n    r0 producer=t0 consumers=t2 lifetime=between_tasks(producer=t0,last_consumer=t2)\n    r1 producer=t1 consumers=t3 lifetime=between_tasks(producer=t1,last_consumer=t3)\n    r2 producer=t4 consumers=t5 lifetime=between_tasks(producer=t4,last_consumer=t5)\n  gates:\n    g0 owner=task:t0 kind=source_read boundary=source_read contract=source_contract status=modeled quality=hard_budgeted_report_with_unresolved_preflight\n    g1 owner=edge:e0 kind=build_side boundary=join_build_side contract=query_engine_build_side_source_reservation status=modeled quality=exact_preflight\n    g2 owner=edge:e1 kind=probe_side boundary=join_probe_side contract=query_engine_probe_side_source_reservation status=modeled quality=query_engine_owned_ordering\n    g3 owner=edge:e4 kind=join_output boundary=join_output contract=dataframe_gather_output_reservation_size_bytes status=modeled quality=exact_preflight\n    g4 owner=sink_boundary kind=materialization boundary=host_materialization contract=query_engine_explicit_sink_materialization_reservation status=modeled quality=query_engine_owned_boundary\n  sink: kind=host_arrow_materialization input=t5 gates=g4\n"
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
