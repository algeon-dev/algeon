use query_planning::{
    capability::JoinForm,
    error::Result,
    pipeline::{PipelineTask, ResidentDataForm, ResidentResourceId},
    plan::{JoinNode, NodeId},
};

use super::{
    CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState,
    admission::{JoinBuildStreamEstimate, settle_partitioned_join_build},
};
use crate::exec::bounded::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedExecutionAttempt,
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    BoundedJoinLeafLocalParquetSourceStream, ResidentGpuFrame,
    compiled::{
        ResidentFrameStreamPayload,
        helpers::{
            attach_task_facts, invalid, join_node, plan_node, single_input_resource, single_output_resource,
            task_join_plan,
        },
        resource::CompiledGpuResourcePayload,
    },
    consume_bounded_join_build_input_stream, execute_bounded_compiled_join_build_from_resident_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
    execute_bounded_compiled_join_probe_from_frame_stream, execute_bounded_compiled_join_probe_from_resident_frame,
    execute_bounded_compiled_join_probe_from_stream,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_join_build_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        if self.join_builds.contains_key(&node_id) || self.join_probes.contains_key(&node_id) {
            return Err(invalid("compiled join build task found existing state for its join node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input = single_input_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        let build = match self.take_task_input(task, state, input)? {
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => self
                .apply_dynamic_filters_to_source_stream(*stream)
                .and_then(|stream| self.execute_join_build_from_leaf_stream(task, node_id, node, stream)),
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => prefetch
                .finish(&mut self.execution)
                .and_then(|frame| self.execute_join_build_from_resident_frame(task, node_id, node, frame)),
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                self.execute_join_build_from_resident_frame(task, node_id, node, *frame)
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                self.execute_join_build_from_frame_stream(task, node_id, node, stream)
            },
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        // Build state is not a declared resource: the probe task takes it by
        // join node id, and the compiled control edge from build to probe is
        // what orders the two.
        self.join_builds.insert(node_id, build);
        Ok(Vec::new())
    }

    fn execute_join_build_from_resident_frame(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        input_frame: ResidentGpuFrame,
    ) -> Result<BoundedCompiledJoinBuildState> {
        self.produce_hash_join_build_dynamic_filters(node_id, node, &[input_frame.frame.frame()])?;
        let choice =
            self.select_join_build_strategy(task, node_id, node, &input_frame.frame, input_frame.device_bytes())?;
        execute_bounded_compiled_join_build_from_resident_frame(
            &mut self.execution,
            node_id,
            node,
            task_join_plan(task, node_id)?.form(),
            choice.strategy(),
            self.policy.keyed_hash_join,
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
            bytes: stream.total_bytes_hint().filter(|bytes| *bytes > 0),
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
        let options = self.policy.keyed_hash_join;
        let keyed_hash_inputs = task_join_plan(task, node_id)?.form() == JoinForm::KeyedEqui;
        let mut settle = |execution: &mut BoundedExecutionAttempt<'_>,
                          observation: Option<BoundedJoinBuildStreamObservation>| {
            settle_partitioned_join_build(plan, options, task, node_id, estimate, execution, observation)
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
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let node = join_node(self.plan, task, node_id)?;
        // The build moves into the probe state; after this point a failed
        // probe is released through `join_probes` in `cleanup_after_error`,
        // or through the probe function's own error path before insertion.
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
        // Dynamic filters that target this probe apply to the probe input
        // before it reaches the hash table, so the filtered rows never pay for
        // a probe; the resident-frame arm filters in place, the stream arms
        // wrap lazily.
        let lazy_output = self.join_probe_output_consumed_as_stream(node_id);
        let probe = match self.take_task_input(task, state, input)? {
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                self.apply_dynamic_filters_to_join_leaf_stream(node_id, *stream).and_then(|stream| {
                    execute_bounded_compiled_join_probe_from_stream(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        stream,
                        lazy_output,
                    )
                })
            },
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => prefetch
                .finish(&mut self.execution)
                .and_then(|frame| self.apply_dynamic_filters_to_join_probe_frame(node_id, node, frame))
                .and_then(|frame| {
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        frame,
                    )
                }),
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                self.apply_dynamic_filters_to_join_probe_frame(node_id, node, *frame).and_then(|frame| {
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        frame,
                    )
                })
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                self.apply_dynamic_filters_to_join_probe_stream(node_id, node, stream).and_then(|stream| {
                    execute_bounded_compiled_join_probe_from_frame_stream(
                        &mut self.execution,
                        node_id,
                        node,
                        build,
                        stream,
                        self.aggregate_dispatch.as_ref(),
                    )
                })
            },
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.join_probes.insert(node_id, probe);
        Ok(Vec::new())
    }

    pub(super) fn execute_join_materialize_task(
        &mut self,
        task: &PipelineTask,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
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
                    CompiledGpuResource::frame_stream_payload(stream)
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
        Ok(vec![(output, resource)])
    }
}
