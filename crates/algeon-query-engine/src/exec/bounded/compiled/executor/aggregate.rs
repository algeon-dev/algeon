use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskOutput};
use crate::{
    error::Result,
    exec::bounded::{
        ResidentGpuFrame,
        compiled::{
            helpers::{
                attach_task_facts, invalid, limit_node, plan_node, single_input_resource, single_output_resource,
                sort_node,
            },
            resource::CompiledGpuResourcePayload,
        },
        execute_bounded_generic_unary_from_frame, execute_compiled_bounded_aggregate_from_frame,
        execute_compiled_bounded_aggregate_from_source_stream, execute_compiled_bounded_aggregate_from_stream,
        execute_final_sort_node_from_committed_with_retry, execute_final_sort_node_from_frame_with_retry,
        execute_final_sort_node_from_stream_with_retry,
    },
    pipeline::{PipelineTask, ResidentDataForm, SortStrategy},
    plan::PlanNode,
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_aggregate_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let preserve_output_stream = self.resource_data_form(output) == ResidentDataForm::FrameStream;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled aggregate task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(invalid("compiled aggregate task does not reference an aggregate plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let topk = compiled_aggregate_topk(self.plan, self.pipeline, task);
        let input_is_stream =
            matches!(state.resource_payload(input)?.payload, CompiledGpuResourcePayload::ResidentFrameStream(_));
        let can_consume_join_leaf_source = state.can_consume_task_input_resource(task, input)
            && matches!(
                state.resource_payload(input)?.payload,
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            );
        let frame = if input_is_stream {
            let stream = if state.can_consume_task_input_resource(task, input) {
                let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
                let CompiledGpuResourcePayload::ResidentFrameStream(stream) = input_payload.payload else {
                    unreachable!("aggregate stream input was checked");
                };
                stream
            } else {
                let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &state.resource_payload(input)?.payload
                else {
                    unreachable!("aggregate stream input was checked");
                };
                stream.alias(node_id, "compiled_aggregate_stream_alias")?
            };
            execute_compiled_bounded_aggregate_from_stream(
                &mut self.execution,
                node_id,
                aggregate,
                self.plan
                    .node(aggregate.input)
                    .ok_or_else(|| {
                        invalid("compiled aggregate task references a missing aggregate input node")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", aggregate.input)
                    })?
                    .output_schema(),
                node.output_schema(),
                stream,
                preserve_output_stream,
            )
        } else if can_consume_join_leaf_source {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) = input_payload.payload else {
                unreachable!("join leaf source payload was checked above")
            };
            let stream = self.apply_dynamic_filters_to_source_stream(*stream)?;
            execute_compiled_bounded_aggregate_from_source_stream(
                &mut self.execution,
                node_id,
                aggregate,
                self.plan
                    .node(aggregate.input)
                    .ok_or_else(|| {
                        invalid("compiled aggregate task references a missing aggregate input node")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", aggregate.input)
                    })?
                    .output_schema(),
                node.output_schema(),
                stream,
                preserve_output_stream,
                self.aggregate_dispatch,
                topk.as_ref(),
            )
        } else if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = input_payload.payload else {
                return Err(invalid("compiled aggregate task expected a resident frame input")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            };
            let ResidentGpuFrame { frame: input_frame, device_bytes: input_reserved_bytes } = *frame;
            let result = execute_compiled_bounded_aggregate_from_frame(
                &mut self.execution,
                node_id,
                aggregate,
                self.plan
                    .node(aggregate.input)
                    .ok_or_else(|| {
                        invalid("compiled aggregate task references a missing aggregate input node")
                            .with_fact("compiled_task_id", task.id())
                            .with_fact("plan_node_id", aggregate.input)
                    })?
                    .output_schema(),
                node.output_schema(),
                input_frame,
                input_reserved_bytes,
            );
            match result.map(CompiledGpuResource::new) {
                Ok(output) => Ok(output),
                Err(error) => Err(error),
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let stream = self.apply_dynamic_filters_to_source_stream(stream)?;
                execute_compiled_bounded_aggregate_from_source_stream(
                    &mut self.execution,
                    node_id,
                    aggregate,
                    self.plan
                        .node(aggregate.input)
                        .ok_or_else(|| {
                            invalid("compiled aggregate task references a missing aggregate input node")
                                .with_fact("compiled_task_id", task.id())
                                .with_fact("plan_node_id", aggregate.input)
                        })?
                        .output_schema(),
                    node.output_schema(),
                    stream,
                    preserve_output_stream,
                    self.aggregate_dispatch,
                    topk.as_ref(),
                )
            } else if matches!(input_payload.payload, CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_))
            {
                Err(invalid("compiled aggregate task cannot share a pending source read")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("resident_resource_id", input))
            } else {
                self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                let input_payload = state.resource_payload(input)?;
                execute_compiled_bounded_aggregate_from_frame(
                    &mut self.execution,
                    node_id,
                    aggregate,
                    self.plan
                        .node(aggregate.input)
                        .ok_or_else(|| {
                            invalid("compiled aggregate task references a missing aggregate input node")
                                .with_fact("compiled_task_id", task.id())
                                .with_fact("plan_node_id", aggregate.input)
                        })?
                        .output_schema(),
                    node.output_schema(),
                    input_payload.clone_frame(),
                    input_payload.reservation_bytes(),
                )
                .map(CompiledGpuResource::new)
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        let group_key_state = match &frame.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame: resident, .. } => Some(vec![resident.frame.clone()]),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.complete_group_key_state(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => None,
        };
        self.produce_aggregate_group_key_dynamic_filters(node_id, node.output_schema(), group_key_state.as_deref())?;
        Ok(CompiledTaskOutput::new(vec![(output, frame)]))
    }

    pub(super) fn execute_window_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled window task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let PlanNode::Window(window) = &node.kind else {
            return Err(invalid("compiled window task does not reference a window plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let input_payload = state.resource_payload(input)?;
        let input_schema = self
            .plan
            .node(window.input)
            .ok_or_else(|| {
                invalid("compiled window task references a missing window input node")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", window.input)
            })?
            .output_schema()
            .clone();
        let output_schema = node.output_schema().clone();
        let frame = execute_bounded_generic_unary_from_frame(
            &mut self.execution,
            node_id,
            "compiled_window",
            input_payload.clone_frame(),
            false,
            |context, input, _token, metrics| {
                crate::exec::operators::execute_window(context, window, &input_schema, input, &output_schema, metrics)
            },
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    pub(super) fn execute_final_sort_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let sort = sort_node(self.plan, task, node_id)?.clone();
        if let Some(SortStrategy::FusedTopK { limit_node: limit_node_id, .. }) = task.sort_strategy()
            && (state.can_consume_task_input_resource(task, input)
                || matches!(state.resource_payload(input)?.payload, CompiledGpuResourcePayload::ResidentFrameStream(_)))
        {
            let limit = limit_node(self.plan, task, limit_node_id)?.clone();
            let input_payload = if state.can_consume_task_input_resource(task, input) {
                state.consume_task_input_resource(task, input)?.into_payload()
            } else {
                let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &state.resource_payload(input)?.payload
                else {
                    unreachable!("shared sort input is a stream");
                };
                CompiledGpuResource::frame_stream_payload(stream.alias(node_id, "compiled_final_sort_stream_alias")?)
            };
            let frame = match input_payload.payload {
                CompiledGpuResourcePayload::ResidentFrame { frame: input_frame, .. } => {
                    execute_final_sort_node_from_committed_with_retry(
                        &mut self.execution,
                        node_id,
                        &sort,
                        Some(&limit),
                        *input_frame,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                    execute_final_sort_node_from_stream_with_retry(&mut self.execution, node_id, &sort, &limit, stream)
                },
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                    Err(invalid("compiled final sort cannot consume a deferred join leaf source")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
                CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                    Err(invalid("compiled final sort cannot consume a pending source read")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
            }
            .map_err(|error| attach_task_facts(error, task, node_id))?;
            return Ok(CompiledTaskOutput::new(vec![(
                output,
                CompiledGpuResource::frame_with_applied_limit(frame, limit_node_id),
            )]));
        }

        let input_payload = state.resource_payload(input)?;
        let frame = execute_final_sort_node_from_frame_with_retry(
            &mut self.execution,
            node_id,
            &sort,
            input_payload.clone_frame(),
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }
}

/// Propagates the compiled sort's existing pruning contract through an
/// exclusively consumed projection chain. Shared aggregate facts must remain
/// complete for every consumer, including dynamic-filter producers.
fn compiled_aggregate_topk<'a>(
    plan: &'a crate::plan::QueryPlan,
    pipeline: &crate::pipeline::CompiledPipeline,
    task: &PipelineTask,
) -> Option<crate::exec::bounded::SourceAggregateTopK<'a>> {
    let node_id = task.plan_node()?;
    if plan.dynamic_filter_routes().iter().any(|route| matches!(
        route.producer,
        crate::dynamic_filter::DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } if aggregate_node_id == node_id
    )) {
        return None;
    }
    let mut current = task;
    let mut stages = Vec::new();
    loop {
        let [output] = current.output_resources() else {
            return None;
        };
        let resource = pipeline.resources().iter().find(|resource| resource.id() == *output)?;
        let [consumer] = resource.consumers() else {
            return None;
        };
        current = pipeline.tasks().iter().find(|task| task.id() == *consumer)?;
        let node_id = current.plan_node()?;
        let node = plan.node(node_id)?;
        match &node.kind {
            PlanNode::Projection(projection) => stages.push(crate::exec::bounded::BoundedUnaryStage::Projection {
                node_id,
                node: projection,
                output_schema: node.output_schema(),
            }),
            PlanNode::Sort(sort) => {
                let SortStrategy::FusedTopK { limit_node, .. } = current.sort_strategy()? else {
                    return None;
                };
                let PlanNode::Limit(limit) = &plan.node(limit_node)?.kind else {
                    return None;
                };
                return Some(crate::exec::bounded::SourceAggregateTopK {
                    post_aggregate_stages: stages,
                    sort,
                    limit: (limit_node, limit),
                });
            },
            _ => return None,
        }
    }
}
