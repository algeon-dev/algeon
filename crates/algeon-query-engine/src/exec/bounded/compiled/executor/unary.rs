use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
use crate::{
    error::Result,
    exec::{
        bounded::{
            compiled::{
                frame_stream::{map_filter_resident_frame_stream, map_projection_resident_frame_stream},
                helpers::{attach_task_facts, invalid, plan_node, single_output_resource},
                resource::CompiledGpuResourcePayload,
            },
            execute_bounded_projection, execute_bounded_projection_from_frame, run_budgeted,
        },
        operators::execute_filter_with_budget,
    },
    pipeline::{PipelineTask, ResidentResourceId, ResourceSharing},
    plan::PlanNode,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_unary_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let input = *task.input_resources().first().ok_or_else(|| {
            invalid("compiled unary task has no input resource")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
        })?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled unary task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let frame = match &node.kind {
            PlanNode::Filter(filter) => {
                if state.can_consume_task_input_resource(task, input) {
                    let input_payload = state.consume_task_input_resource(task, input)?;
                    match input_payload.payload {
                        CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                                stream.with_filter_stage(node_id, filter.clone()),
                            ))
                        },
                        CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                            let frame = self.apply_dynamic_filters_to_filter_frame(node_id, filter, *frame)?;
                            run_budgeted(
                                &mut self.execution,
                                node_id,
                                "compiled_filter",
                                frame.frame,
                                |context, input, token, metrics| {
                                    execute_filter_with_budget(context, filter, input, token, metrics)
                                },
                            )
                            .map(CompiledGpuResource::new)
                        },
                        CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                            let stream = self.apply_dynamic_filters_to_filter_stream(node_id, filter, stream)?;
                            let stream = map_filter_resident_frame_stream(node_id, stream, filter.clone());
                            let stream = if self.resource_sharing(output) != ResourceSharing::SingleConsume {
                                stream.into_shared(&mut self.execution)?
                            } else {
                                stream
                            };
                            Ok(CompiledGpuResource::frame_stream_payload(stream))
                        },
                        CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                            Err(invalid("compiled filter task cannot consume a pending source read")
                                .with_fact("compiled_task_id", task.id())
                                .with_fact("plan_node_id", node_id))
                        },
                    }
                } else {
                    let input_payload = state.resource_payload(input)?;
                    if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                        Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                            stream.with_filter_stage(node_id, filter.clone()),
                        ))
                    } else if matches!(
                        input_payload.payload,
                        CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_)
                    ) {
                        Err(invalid("compiled filter task cannot share a pending source read")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", node_id)
                            .with_fact("resident_resource_id", input))
                    } else if let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &input_payload.payload {
                        let stream = stream.alias(node_id, "compiled_filter_stream_alias")?;
                        let stream = self.apply_dynamic_filters_to_filter_stream(node_id, filter, stream)?;
                        let stream = map_filter_resident_frame_stream(node_id, stream, filter.clone());
                        let stream = if self.resource_sharing(output) != ResourceSharing::SingleConsume {
                            stream.into_shared(&mut self.execution)?
                        } else {
                            stream
                        };
                        Ok(CompiledGpuResource::frame_stream_payload(stream))
                    } else {
                        self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                        let input_payload = state.resource_payload(input)?;
                        if let Some(frame) = self.apply_dynamic_filters_to_filter_shared_frame(
                            node_id,
                            filter,
                            input_payload.clone_frame(),
                        )? {
                            run_budgeted(
                                &mut self.execution,
                                node_id,
                                "compiled_filter",
                                frame.frame,
                                |context, input, token, metrics| {
                                    execute_filter_with_budget(context, filter, input, token, metrics)
                                },
                            )
                            .map(CompiledGpuResource::new)
                        } else {
                            run_budgeted(
                                &mut self.execution,
                                node_id,
                                "compiled_filter",
                                input_payload.clone_frame(),
                                |context, input, token, metrics| {
                                    execute_filter_with_budget(context, filter, input, token, metrics)
                                },
                            )
                            .map(CompiledGpuResource::new)
                        }
                    }
                }
            },
            PlanNode::Projection(projection) => {
                if state.can_consume_task_input_resource(task, input) {
                    let input_payload = state.consume_task_input_resource(task, input)?;
                    match input_payload.payload {
                        CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                                stream.with_projection_stage(node_id, projection.clone()),
                            ))
                        },
                        CompiledGpuResourcePayload::ResidentFrame { frame, .. } => execute_bounded_projection(
                            &mut self.execution,
                            node_id,
                            "compiled_projection",
                            projection,
                            *frame,
                        )
                        .map(CompiledGpuResource::new),
                        CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                            let stream = map_projection_resident_frame_stream(node_id, stream, projection.clone());
                            let stream = if self.resource_sharing(output) != ResourceSharing::SingleConsume {
                                stream.into_shared(&mut self.execution)?
                            } else {
                                stream
                            };
                            Ok(CompiledGpuResource::frame_stream_payload(stream))
                        },
                        CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                            Err(invalid("compiled projection task cannot consume a pending source read")
                                .with_fact("compiled_task_id", task.id())
                                .with_fact("plan_node_id", node_id))
                        },
                    }
                } else {
                    let input_payload = state.resource_payload(input)?;
                    if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                        Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                            stream.with_projection_stage(node_id, projection.clone()),
                        ))
                    } else if matches!(
                        input_payload.payload,
                        CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_)
                    ) {
                        Err(invalid("compiled projection task cannot share a pending source read")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", node_id)
                            .with_fact("resident_resource_id", input))
                    } else if let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &input_payload.payload {
                        let stream = stream.alias(node_id, "compiled_projection_stream_alias")?;
                        let stream = map_projection_resident_frame_stream(node_id, stream, projection.clone());
                        let stream = if self.resource_sharing(output) != ResourceSharing::SingleConsume {
                            stream.into_shared(&mut self.execution)?
                        } else {
                            stream
                        };
                        Ok(CompiledGpuResource::frame_stream_payload(stream))
                    } else {
                        self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                        let input_payload = state.resource_payload(input)?;
                        execute_bounded_projection_from_frame(
                            &mut self.execution,
                            node_id,
                            "compiled_projection",
                            projection,
                            input_payload.clone_frame(),
                        )
                        .map(CompiledGpuResource::new)
                    }
                }
            },
            _ => Err(invalid("compiled unary task does not reference a filter or projection plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, frame)])
    }
}
