use super::*;

fn pipeline_with_extra_sink_candidate(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let mut tasks = pipeline.tasks().to_vec();
    let mut edges = pipeline.edges().to_vec();
    let sink_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.kind() == PipelineEdgeKind::SinkInput)
        .expect("fixture should have a sink input edge");
    let [sink_resource] = sink_edge.resources() else {
        panic!("sink input edge should carry one resource");
    };
    let extra_sink = PipelineTaskId::new(tasks.len());
    tasks.push(PipelineTask::new(extra_sink, TaskPhysical::SinkMaterialize).with_input_resources(vec![*sink_resource]));
    edges.push(
        PipelineEdge::new(PipelineEdgeId::new(edges.len()), sink_edge.from(), extra_sink, PipelineEdgeKind::SinkInput)
            .with_resources(vec![*sink_resource]),
    );

    let resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let mut consumers = resource.consumers().to_vec();
            if resource.id() == *sink_resource {
                consumers.push(extra_sink);
                consumers.sort_unstable();
                consumers.dedup();
            }
            ResidentResource::new(resource.id(), resource.producer(), consumers, resource.lifetime())
        })
        .collect::<Vec<_>>();
    let sink = SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input());
    CompiledPipeline::from_parts_unvalidated(tasks, edges, pipeline.barriers().to_vec(), resources, sink)
}

fn pipeline_with_post_root_fanout(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let mut tasks = pipeline.tasks().to_vec();
    let mut edges = pipeline.edges().to_vec();
    let fanout_edge = pipeline
        .edges()
        .iter()
        .find(|edge| {
            edge.kind() == PipelineEdgeKind::ResidentFrame
                && pipeline
                    .tasks()
                    .iter()
                    .find(|task| task.id() == edge.from())
                    .is_some_and(|task| task.kind() == PipelineTaskKind::JoinMaterialize)
        })
        .expect("fixture should have a post-root resident-frame edge");
    let [fanout_resource] = fanout_edge.resources() else {
        panic!("post-root resident-frame edge should carry one resource");
    };
    let extra_task = PipelineTaskId::new(tasks.len());
    let extra_resource = ResidentResourceId::new(pipeline.resources().len());
    tasks.push(
        PipelineTask::new(extra_task, TaskPhysical::Unary)
            .with_input_resources(vec![*fanout_resource])
            .with_output_resources(vec![extra_resource]),
    );
    edges.push(
        PipelineEdge::new(
            PipelineEdgeId::new(edges.len()),
            fanout_edge.from(),
            extra_task,
            PipelineEdgeKind::ResidentFrame,
        )
        .with_resources(vec![*fanout_resource]),
    );

    let mut resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let mut consumers = resource.consumers().to_vec();
            if resource.id() == *fanout_resource {
                consumers.push(extra_task);
                consumers.sort_unstable();
                consumers.dedup();
            }
            ResidentResource::new(resource.id(), resource.producer(), consumers, resource.lifetime())
        })
        .collect::<Vec<_>>();
    resources.push(ResidentResource::new(
        extra_resource,
        extra_task,
        vec![],
        ResidentLifetime::TaskLocal { task: extra_task },
    ));
    let sink = SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input());
    CompiledPipeline::from_parts_unvalidated(tasks, edges, pipeline.barriers().to_vec(), resources, sink)
}

fn simple_unary_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let unary = PipelineTaskId::new(1);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::from_parts_unvalidated(
        vec![
            PipelineTask::new(source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![frame]),
            PipelineTask::new(unary, TaskPhysical::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, unary, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![unary]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![unary]),
        ],
        vec![ResidentResource::new(frame, source, vec![unary], ResidentLifetime::BetweenTasks { producer: source })],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, unary),
    )
}

fn independent_two_source_pipeline() -> CompiledPipeline {
    let first_source = PipelineTaskId::new(0);
    let second_source = PipelineTaskId::new(1);
    let first_consumer = PipelineTaskId::new(2);
    let second_consumer = PipelineTaskId::new(3);
    let first_frame = ResidentResourceId::new(0);
    let second_frame = ResidentResourceId::new(1);

    CompiledPipeline::from_parts_unvalidated(
        vec![
            PipelineTask::new(first_source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![first_frame]),
            PipelineTask::new(second_source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![second_frame]),
            PipelineTask::new(first_consumer, TaskPhysical::Unary).with_input_resources(vec![first_frame]),
            PipelineTask::new(second_consumer, TaskPhysical::Unary).with_input_resources(vec![second_frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), first_source, first_consumer, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![first_frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), second_source, second_consumer, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![second_frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![first_source])
                .with_downstream_tasks(vec![first_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![second_source])
                .with_downstream_tasks(vec![second_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![first_consumer, second_consumer]),
        ],
        vec![
            ResidentResource::new(
                first_frame,
                first_source,
                vec![first_consumer],
                ResidentLifetime::BetweenTasks { producer: first_source },
            ),
            ResidentResource::new(
                second_frame,
                second_source,
                vec![second_consumer],
                ResidentLifetime::BetweenTasks { producer: second_source },
            ),
        ],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, second_consumer),
    )
}

fn out_of_order_join_barrier_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let probe = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let materialize = PipelineTaskId::new(3);
    let sink = PipelineTaskId::new(4);
    let source_frame = ResidentResourceId::new(0);
    let join_output = ResidentResourceId::new(1);

    CompiledPipeline::from_parts_unvalidated(
        vec![
            PipelineTask::new(source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![source_frame]),
            PipelineTask::new(probe, TaskPhysical::JoinProbe(fixture_join_plan()))
                .with_input_resources(vec![source_frame]),
            PipelineTask::new(build, TaskPhysical::JoinBuild(fixture_join_plan()))
                .with_input_resources(vec![source_frame]),
            PipelineTask::new(materialize, TaskPhysical::JoinMaterialize).with_output_resources(vec![join_output]),
            PipelineTask::new(sink, TaskPhysical::SinkMaterialize).with_input_resources(vec![join_output]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, probe, PipelineEdgeKind::JoinProbeInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), source, build, PipelineEdgeKind::JoinBuildInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![join_output]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![build, probe]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![build])
                .with_downstream_tasks(vec![probe]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![probe])
                .with_downstream_tasks(vec![materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![
            ResidentResource::new(
                source_frame,
                source,
                vec![build, probe],
                ResidentLifetime::BetweenTasks { producer: source },
            ),
            ResidentResource::new(
                join_output,
                materialize,
                vec![sink],
                ResidentLifetime::BetweenTasks { producer: materialize },
            ),
        ],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    )
}

fn downstream_build_branch_after_completed_upstream_build_pipeline() -> CompiledPipeline {
    let first_build_source = PipelineTaskId::new(0);
    let first_build = PipelineTaskId::new(1);
    let first_probe_source = PipelineTaskId::new(2);
    let second_build_source = PipelineTaskId::new(3);
    let first_probe = PipelineTaskId::new(4);
    let second_build = PipelineTaskId::new(5);
    let first_materialize = PipelineTaskId::new(6);
    let second_probe = PipelineTaskId::new(7);
    let second_materialize = PipelineTaskId::new(8);
    let sink = PipelineTaskId::new(9);
    let first_build_frame = ResidentResourceId::new(0);
    let first_probe_frame = ResidentResourceId::new(1);
    let second_build_frame = ResidentResourceId::new(2);
    let first_join_output = ResidentResourceId::new(3);
    let second_join_output = ResidentResourceId::new(4);

    CompiledPipeline::from_parts_unvalidated(
        vec![
            PipelineTask::new(first_build_source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![first_build_frame]),
            PipelineTask::new(first_build, TaskPhysical::JoinBuild(fixture_join_plan()))
                .with_input_resources(vec![first_build_frame]),
            PipelineTask::new(first_probe_source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![first_probe_frame]),
            PipelineTask::new(second_build_source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![second_build_frame]),
            PipelineTask::new(first_probe, TaskPhysical::JoinProbe(fixture_join_plan()))
                .with_input_resources(vec![first_probe_frame]),
            PipelineTask::new(second_build, TaskPhysical::JoinBuild(fixture_join_plan()))
                .with_input_resources(vec![second_build_frame]),
            PipelineTask::new(first_materialize, TaskPhysical::JoinMaterialize)
                .with_output_resources(vec![first_join_output]),
            PipelineTask::new(second_probe, TaskPhysical::JoinProbe(fixture_join_plan()))
                .with_input_resources(vec![first_join_output]),
            PipelineTask::new(second_materialize, TaskPhysical::JoinMaterialize)
                .with_output_resources(vec![second_join_output]),
            PipelineTask::new(sink, TaskPhysical::SinkMaterialize).with_input_resources(vec![second_join_output]),
        ],
        vec![
            PipelineEdge::new(
                PipelineEdgeId::new(0),
                first_build_source,
                first_build,
                PipelineEdgeKind::JoinBuildInput,
            )
            .with_resources(vec![first_build_frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(1),
                first_probe_source,
                first_probe,
                PipelineEdgeKind::JoinProbeInput,
            )
            .with_resources(vec![first_probe_frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), first_build, first_probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(3), first_probe, first_materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(
                PipelineEdgeId::new(4),
                second_build_source,
                second_build,
                PipelineEdgeKind::JoinBuildInput,
            )
            .with_resources(vec![second_build_frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(5),
                first_materialize,
                second_probe,
                PipelineEdgeKind::JoinProbeInput,
            )
            .with_resources(vec![first_join_output]),
            PipelineEdge::new(PipelineEdgeId::new(6), second_build, second_probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(7), second_probe, second_materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(8), second_materialize, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![second_join_output]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![first_build_source])
                .with_downstream_tasks(vec![first_build]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![first_probe_source])
                .with_downstream_tasks(vec![first_probe]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![second_build_source])
                .with_downstream_tasks(vec![second_build]),
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![first_build])
                .with_downstream_tasks(vec![first_probe]),
            PipelineBarrier::new(PipelineBarrierId::new(4), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![first_probe])
                .with_downstream_tasks(vec![first_materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(5), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![second_build])
                .with_downstream_tasks(vec![second_probe]),
            PipelineBarrier::new(PipelineBarrierId::new(6), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![second_probe])
                .with_downstream_tasks(vec![second_materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(7), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![
            ResidentResource::new(
                first_build_frame,
                first_build_source,
                vec![first_build],
                ResidentLifetime::BetweenTasks { producer: first_build_source },
            ),
            ResidentResource::new(
                first_probe_frame,
                first_probe_source,
                vec![first_probe],
                ResidentLifetime::BetweenTasks { producer: first_probe_source },
            ),
            ResidentResource::new(
                second_build_frame,
                second_build_source,
                vec![second_build],
                ResidentLifetime::BetweenTasks { producer: second_build_source },
            ),
            ResidentResource::new(
                first_join_output,
                first_materialize,
                vec![second_probe],
                ResidentLifetime::BetweenTasks { producer: first_materialize },
            ),
            ResidentResource::new(
                second_join_output,
                second_materialize,
                vec![sink],
                ResidentLifetime::BetweenTasks { producer: second_materialize },
            ),
        ],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    )
}

fn shared_resident_fanout_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let first_consumer = PipelineTaskId::new(1);
    let second_consumer = PipelineTaskId::new(2);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::from_parts_unvalidated(
        vec![
            PipelineTask::new(source, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                .with_output_resources(vec![frame]),
            PipelineTask::new(first_consumer, TaskPhysical::Unary).with_input_resources(vec![frame]),
            PipelineTask::new(second_consumer, TaskPhysical::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, first_consumer, PipelineEdgeKind::SharedResidentProducer)
                .with_resources(vec![frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(1),
                source,
                second_consumer,
                PipelineEdgeKind::SharedResidentProducer,
            )
            .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![first_consumer, second_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![second_consumer]),
        ],
        vec![ResidentResource::new(
            frame,
            source,
            vec![first_consumer, second_consumer],
            ResidentLifetime::BetweenTasks { producer: source },
        )],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, second_consumer),
    )
}

/// A keyed equi-join plan for scheduler fixtures whose join tasks reference no
/// plan node.
fn fixture_join_plan() -> JoinPlan {
    JoinPlan::new(JoinForm::KeyedEqui, JoinStrategy::OneShotEqui, JoinForm::KeyedEqui.strategies().to_vec())
        .expect("a keyed equi-join may run every keyed equi strategy")
}
