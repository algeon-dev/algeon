use super::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskOutput,
    admission::{JoinBuildStreamEstimate, settle_partitioned_join_build},
};
use crate::{
    capability::keyed_hash_join_capable,
    error::Result,
    exec::bounded::{
        BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedExecutionAttempt,
        BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
        BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame,
        compiled::{
            ResidentFrameStreamPayload,
            helpers::{
                attach_task_facts, invalid, join_node, plan_node, single_input_resource, single_output_resource,
            },
            resource::CompiledGpuResourcePayload,
        },
        consume_bounded_join_build_input_stream, execute_bounded_compiled_join_build_from_frame,
        execute_bounded_compiled_join_build_from_resident_frame, execute_bounded_compiled_join_materialize,
        execute_bounded_compiled_join_materialize_preserving_stream, execute_bounded_compiled_join_probe_from_frame,
        execute_bounded_compiled_join_probe_from_frame_stream, execute_bounded_compiled_join_probe_from_resident_frame,
        execute_bounded_compiled_join_probe_from_stream,
    },
    pipeline::{PipelineTask, ResidentDataForm, ResourceSharing},
    plan::{JoinNode, NodeId},
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_join_build_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        if self.join_builds.contains_key(&node_id) || self.join_probes.contains_key(&node_id) {
            return Err(invalid("compiled join build task found existing state for its join node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input = single_input_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let build = if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            match input_payload.payload {
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                    let stream = self.apply_dynamic_filters_to_source_stream(*stream)?;
                    self.execute_join_build_from_leaf_stream(task, node_id, node, stream)
                },
                CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                    let frame = prefetch.finish(&mut self.execution)?;
                    self.execute_join_build_from_resident_frame(task, node_id, node, frame)
                },
                CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                    self.execute_join_build_from_resident_frame(task, node_id, node, *frame)
                },
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                    self.execute_join_build_from_frame_stream(task, node_id, node, stream)
                },
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let stream = self.apply_dynamic_filters_to_source_stream(stream)?;
                self.execute_join_build_from_leaf_stream(task, node_id, node, stream)
            } else if let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &input_payload.payload {
                let stream = stream.alias(node_id, "compiled_join_build_stream_alias")?;
                self.execute_join_build_from_frame_stream(task, node_id, node, stream)
            } else if matches!(input_payload.payload, CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_))
            {
                Err(invalid("compiled join build cannot share a pending source read")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("resident_resource_id", input))
            } else {
                self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                let input_payload = state.resource_payload(input)?;
                let input_frame = input_payload.clone_frame();
                self.produce_hash_join_build_dynamic_filters(node_id, node, &[input_frame.frame()])?;
                let strategy = self.select_join_build_strategy(
                    task,
                    node_id,
                    node,
                    &input_frame,
                    input_payload.reservation_bytes(),
                )?;
                execute_bounded_compiled_join_build_from_frame(
                    &mut self.execution,
                    node_id,
                    node,
                    strategy.strategy,
                    self.plan.keyed_hash_join_strategy_options(),
                    input_frame,
                )
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.join_builds.insert(node_id, build);
        Ok(CompiledTaskOutput::empty())
    }

    fn execute_join_build_from_resident_frame(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        input_frame: ResidentGpuFrame,
    ) -> Result<BoundedCompiledJoinBuildState> {
        self.produce_hash_join_build_dynamic_filters(node_id, node, &[input_frame.frame.frame()])?;
        let strategy =
            self.select_join_build_strategy(task, node_id, node, &input_frame.frame, input_frame.device_bytes())?;
        execute_bounded_compiled_join_build_from_resident_frame(
            &mut self.execution,
            node_id,
            node,
            strategy.strategy,
            self.plan.keyed_hash_join_strategy_options(),
            input_frame,
        )
    }

    fn execute_join_build_from_leaf_stream(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        stream: BoundedJoinLeafLocalParquetSourceStream,
    ) -> Result<BoundedCompiledJoinBuildState> {
        let estimate = JoinBuildStreamEstimate {
            rows: stream.estimated_source_rows(),
            bytes: stream.estimated_source_read_bytes(),
        };
        self.execute_join_build_from_input_stream(
            task,
            node_id,
            node,
            BoundedJoinBuildInputStream::Leaf(Box::new(stream)),
            estimate,
        )
    }

    fn execute_join_build_from_frame_stream(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        stream: ResidentFrameStreamPayload,
    ) -> Result<BoundedCompiledJoinBuildState> {
        let estimate = JoinBuildStreamEstimate {
            rows: Some(stream.num_rows()).filter(|rows| *rows > 0),
            bytes: Some(stream.aggregate_reservation_bytes()).filter(|bytes| *bytes > 0),
        };
        self.execute_join_build_from_input_stream(
            task,
            node_id,
            node,
            BoundedJoinBuildInputStream::Frames(stream),
            estimate,
        )
    }

    // Consumes the build input chunk by chunk. The strategy owner decides while
    // chunks arrive whether the rest can go straight into partitions; otherwise
    // the chunks are concatenated and the resident-frame path selects as before.
    fn execute_join_build_from_input_stream(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        input: BoundedJoinBuildInputStream,
        estimate: JoinBuildStreamEstimate,
    ) -> Result<BoundedCompiledJoinBuildState> {
        let plan = self.plan;
        let keyed_hash_inputs = keyed_hash_join_capable(node);
        let mut settle = |execution: &mut BoundedExecutionAttempt<'_>,
                          observation: Option<BoundedJoinBuildStreamObservation>| {
            settle_partitioned_join_build(plan, task, node_id, node, estimate, execution, observation)
        };
        let outcome = consume_bounded_join_build_input_stream(
            &mut self.execution,
            node_id,
            node,
            input,
            keyed_hash_inputs,
            &mut settle,
        )?;
        match outcome {
            BoundedJoinBuildStreamOutcome::Materialized(frame) => {
                self.execute_join_build_from_resident_frame(task, node_id, node, frame)
            },
            BoundedJoinBuildStreamOutcome::Partitioned(build) => {
                let partitions = build.partitions().iter().collect::<Vec<_>>();
                self.produce_hash_join_build_dynamic_filters(node_id, node, &partitions)?;
                Ok(BoundedCompiledJoinBuildState::Partitioned(build))
            },
        }
    }

    pub(super) fn execute_join_probe_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let build = self.join_builds.remove(&node_id).ok_or_else(|| {
            invalid("compiled join probe task ran before join build state was available")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        if self.join_probes.contains_key(&node_id) {
            return Err(invalid("compiled join probe task found existing probe state for its join node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let lazy_output = self.join_probe_output_consumed_as_stream(node_id);
        let probe = if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            match input_payload.payload {
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                    let stream = self.apply_dynamic_filters_to_join_leaf_stream(node_id, *stream)?;
                    execute_bounded_compiled_join_probe_from_stream(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        stream,
                        lazy_output,
                    )
                },
                CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                    let frame = prefetch.finish(&mut self.execution)?;
                    let frame = self.apply_dynamic_filters_to_join_probe_frame(node_id, node, frame)?;
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        frame,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                    let frame = self.apply_dynamic_filters_to_join_probe_frame(node_id, node, *frame)?;
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        frame,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                    let stream = self.apply_dynamic_filters_to_join_probe_stream(node_id, node, stream)?;
                    execute_bounded_compiled_join_probe_from_frame_stream(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        stream,
                        self.aggregate_dispatch.as_ref(),
                    )
                },
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let stream = self.apply_dynamic_filters_to_join_leaf_stream(node_id, stream)?;
                execute_bounded_compiled_join_probe_from_stream(
                    &mut self.execution,
                    node_id,
                    node,
                    build,
                    stream,
                    lazy_output,
                )
            } else if let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &input_payload.payload {
                if self.resource_data_form(input) != ResidentDataForm::FrameStream {
                    return Err(invalid("compiled join probe cannot share this resident frame stream")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id)
                        .with_fact("resident_resource_id", input));
                }
                let stream = match stream.alias(node_id, "compiled_join_probe_stream_alias") {
                    Ok(stream) => stream,
                    Err(error) => {
                        return Err(error);
                    },
                };
                let stream = self.apply_dynamic_filters_to_join_probe_stream(node_id, node, stream)?;
                execute_bounded_compiled_join_probe_from_frame_stream(
                    &mut self.execution,
                    node_id,
                    node,
                    build,
                    stream,
                    self.aggregate_dispatch.as_ref(),
                )
            } else if matches!(input_payload.payload, CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_))
            {
                return Err(invalid("compiled join probe cannot share a pending source read")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("resident_resource_id", input));
            } else {
                self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                let input_payload = state.resource_payload(input)?;
                if let Some(frame) =
                    self.apply_dynamic_filters_to_join_probe_shared_frame(node_id, node, input_payload.clone_frame())?
                {
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        frame,
                    )
                } else {
                    execute_bounded_compiled_join_probe_from_frame(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        input_payload.clone_frame(),
                    )
                }
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.join_probes.insert(node_id, probe);
        Ok(CompiledTaskOutput::empty())
    }

    pub(super) fn execute_join_materialize_task(
        &mut self,
        task: &PipelineTask,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let probe = self.join_probes.remove(&node_id).ok_or_else(|| {
            invalid("compiled join materialize task ran before join probe state was available")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        let resource = if self.resource_data_form(output) == ResidentDataForm::FrameStream {
            match execute_bounded_compiled_join_materialize_preserving_stream(&mut self.execution, node_id, node, probe)
                .map_err(|error| attach_task_facts(error, task, node_id))?
            {
                BoundedCompiledJoinMaterializeOutput::Frame(frame) => CompiledGpuResource::new(*frame),
                BoundedCompiledJoinMaterializeOutput::Stream(frames) => CompiledGpuResource::frame_stream(frames),
                BoundedCompiledJoinMaterializeOutput::LazyStream(stream) => {
                    if self.resource_sharing(output) != ResourceSharing::SingleConsume {
                        CompiledGpuResource::frame_stream(stream.into_remaining_frames_owned(&mut self.execution)?)
                    } else {
                        CompiledGpuResource::frame_stream_payload(stream)
                    }
                },
            }
        } else {
            let frame = execute_bounded_compiled_join_materialize(
                &mut self.execution,
                node_id,
                node,
                probe,
                self.aggregate_dispatch.as_ref(),
            )
            .map_err(|error| attach_task_facts(error, task, node_id))?;
            CompiledGpuResource::new(frame)
        };
        Ok(CompiledTaskOutput::new(vec![(output, resource)]))
    }
}
