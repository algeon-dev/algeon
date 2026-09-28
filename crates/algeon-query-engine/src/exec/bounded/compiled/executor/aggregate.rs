use std::sync::Arc;

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState};
use crate::{
    dynamic_filter::DynamicFilterProducer,
    error::Result,
    exec::{
        bounded::{
            BoundedUnaryStage, ResidentGpuFrame, SourceAggregateTopK,
            compiled::{
                helpers::{
                    attach_task_facts, invalid, limit_node, plan_node, single_input_resource, single_output_resource,
                    sort_node,
                },
                resource::CompiledGpuResourcePayload,
            },
            execute_bounded_unary, execute_compiled_bounded_aggregate_from_frame,
            execute_compiled_bounded_aggregate_from_source_stream, execute_compiled_bounded_aggregate_from_stream,
            execute_final_sort_node_from_committed_with_retry, execute_final_sort_node_from_frame_with_retry,
            execute_final_sort_node_from_stream_with_retry,
        },
        operators::execute_window,
    },
    pipeline::{CompiledPipeline, PipelineTask, ResidentDataForm, ResidentResourceId, SortStrategy},
    plan::{PlanNode, QueryPlan},
};

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_aggregate_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
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
        let input_schema = self
            .plan
            .node(aggregate.input)
            .ok_or_else(|| {
                invalid("compiled aggregate task references a missing aggregate input node")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", aggregate.input)
            })?
            .output_schema();
        let frame = match self.take_task_input(task, state, input)? {
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => execute_compiled_bounded_aggregate_from_stream(
                &mut self.execution,
                node_id,
                aggregate,
                input_schema,
                node.output_schema(),
                stream,
                preserve_output_stream,
            ),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                self.apply_dynamic_filters_to_source_stream(*stream).and_then(|stream| {
                    execute_compiled_bounded_aggregate_from_source_stream(
                        &mut self.execution,
                        node_id,
                        aggregate,
                        input_schema,
                        node.output_schema(),
                        stream,
                        preserve_output_stream,
                        self.aggregate_dispatch.as_ref(),
                        topk.as_ref(),
                    )
                })
            },
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                let ResidentGpuFrame { frame: input_frame, device_bytes: input_device_bytes } = *frame;
                execute_compiled_bounded_aggregate_from_frame(
                    &mut self.execution,
                    node_id,
                    aggregate,
                    input_schema,
                    node.output_schema(),
                    input_frame,
                    input_device_bytes,
                )
                .map(CompiledGpuResource::new)
            },
            payload => Err(self.reject_task_input(task, input, payload)?),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        let group_key_state = match &frame.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame: resident, .. } => Some(vec![resident.frame.clone()]),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.complete_group_key_state(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => None,
        };
        self.produce_aggregate_group_key_dynamic_filters(node_id, node.output_schema(), group_key_state.as_deref())?;
        Ok(vec![(output, frame)])
    }

    pub(super) fn execute_window_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
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
        let input_frame = self.take_frame_input(task, state, input)?;
        let input_schema = Arc::clone(
            self.plan
                .node(window.input)
                .ok_or_else(|| {
                    invalid("compiled window task references a missing window input node")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", window.input)
                })?
                .output_schema(),
        );
        let output_schema = Arc::clone(node.output_schema());
        let frame =
            execute_bounded_unary(&mut self.execution, node_id, input_frame.frame, |context, input, metrics| {
                execute_window(context, window, &input_schema, input, &output_schema, metrics)
            })
            .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, CompiledGpuResource::new(frame))])
    }

    pub(super) fn execute_final_sort_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<Vec<(ResidentResourceId, CompiledGpuResource)>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let sort = sort_node(self.plan, task, node_id)?.clone();
        if let Some(SortStrategy::FusedTopK { limit_node: limit_node_id, .. }) = task.sort_strategy() {
            let limit = limit_node(self.plan, task, limit_node_id)?.clone();
            let frame = match self.take_task_input(task, state, input)? {
                CompiledGpuResourcePayload::ResidentFrame { frame: input_frame, .. } => {
                    execute_final_sort_node_from_committed_with_retry(
                        &mut self.execution,
                        node_id,
                        &sort,
                        Some(&limit),
                        *input_frame,
                        self.aggregate_dispatch.as_ref(),
                    )
                },
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                    execute_final_sort_node_from_stream_with_retry(
                        &mut self.execution,
                        node_id,
                        &sort,
                        &limit,
                        stream,
                        self.aggregate_dispatch.as_ref(),
                    )
                },
                payload => Err(self.reject_task_input(task, input, payload)?),
            }
            .map_err(|error| attach_task_facts(error, task, node_id))?;
            return Ok(vec![(output, CompiledGpuResource::frame_with_applied_limit(frame, limit_node_id))]);
        }

        let input_frame = self.take_frame_input(task, state, input)?;
        let frame =
            execute_final_sort_node_from_frame_with_retry(&mut self.execution, node_id, &sort, input_frame.frame)
                .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(vec![(output, CompiledGpuResource::new(frame))])
    }
}

/// Propagates the compiled sort's existing pruning contract through an
/// exclusively consumed projection chain. Shared aggregate facts must remain
/// complete for every consumer, including dynamic-filter producers.
fn compiled_aggregate_topk<'a>(
    plan: &'a QueryPlan,
    pipeline: &CompiledPipeline,
    task: &PipelineTask,
) -> Option<SourceAggregateTopK<'a>> {
    let node_id = task.plan_node()?;
    if plan.dynamic_filter_routes().iter().any(|route| {
        matches!(
            route.producer,
            DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } if aggregate_node_id == node_id
        )
    }) {
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
            PlanNode::Projection(projection) => stages.push(BoundedUnaryStage::Projection { node: projection }),
            PlanNode::Sort(sort) => {
                let SortStrategy::FusedTopK { limit_node, .. } = current.sort_strategy()? else {
                    return None;
                };
                let PlanNode::Limit(limit) = &plan.node(limit_node)?.kind else {
                    return None;
                };
                return Some(SourceAggregateTopK { post_aggregate_stages: stages, sort, limit: (limit_node, limit) });
            },
            _ => return None,
        }
    }
}
