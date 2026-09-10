use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskOutput};
use crate::{
    error::Result,
    exec::bounded::{
        ResidentGpuFrame,
        compiled::{
            helpers::{
                attach_task_facts, invalid, plan_node, single_input_resource, single_output_resource, sort_node,
            },
            resource::CompiledGpuResourcePayload,
        },
        execute_bounded_generic_unary_from_frame, execute_compiled_bounded_aggregate_from_frame,
        execute_compiled_bounded_aggregate_from_source_stream, execute_compiled_bounded_aggregate_from_stream,
        execute_final_sort_node_from_committed_with_retry, execute_final_sort_node_from_frame_with_retry,
        execute_final_sort_node_from_stream_with_retry,
    },
    pipeline::{PipelineTask, PipelineTaskKind, ResidentResourceId},
    plan::{LimitNode, NodeId, PlanNode, QueryPlan, SortNode},
    source::SourceNode,
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
        let can_consume_stream = state.can_consume_task_input_resource(task, input)
            && matches!(state.resource_payload(input)?.payload, CompiledGpuResourcePayload::ResidentFrameStream(_));
        let can_consume_join_leaf_source = state.can_consume_task_input_resource(task, input)
            && matches!(
                state.resource_payload(input)?.payload,
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            );
        let frame = if can_consume_stream {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            match input_payload.payload {
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
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
                    )
                },
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                    Err(invalid("compiled aggregate task cannot consume a deferred join leaf source stream")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
                CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                    Err(invalid("compiled aggregate task cannot consume a pending source read")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
                CompiledGpuResourcePayload::ResidentFrame { .. } => {
                    unreachable!("only frame streams are consumed here")
                },
            }
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
            match result {
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
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.produce_aggregate_group_key_dynamic_filters(node_id, node.output_schema(), &frame.frame)?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
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
        if let Some((limit_node_id, limit)) = self.downstream_fused_limit(task, output)?
            && state.can_consume_task_input_resource(task, input)
        {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
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

    pub(super) fn downstream_fused_limit(
        &self,
        task: &PipelineTask,
        output: ResidentResourceId,
    ) -> Result<Option<(NodeId, LimitNode)>> {
        let Some(resource) = self.pipeline.resources().iter().find(|resource| resource.id() == output) else {
            return Ok(None);
        };
        let [consumer] = resource.consumers() else {
            return Ok(None);
        };
        let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer) else {
            return Err(invalid("compiled final sort output references a missing consumer task")
                .with_fact("compiled_task_id", task.id())
                .with_fact("resident_resource_id", output));
        };
        if consumer_task.kind() != PipelineTaskKind::Limit {
            return Ok(None);
        }
        let limit_node_id = plan_node(consumer_task)?;
        let Some(node) = self.plan.node(limit_node_id) else {
            return Err(invalid("compiled final sort downstream limit references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("limit_task_id", consumer_task.id())
                .with_fact("plan_node_id", limit_node_id));
        };
        let PlanNode::Limit(limit) = &node.kind else {
            return Err(invalid("compiled final sort downstream limit task does not reference a limit plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("limit_task_id", consumer_task.id())
                .with_fact("plan_node_id", limit_node_id));
        };
        if Some(limit.input) != task.plan_node() {
            return Ok(None);
        }
        let sort_node_id = plan_node(task)?;
        let sort = sort_node(self.plan, task, sort_node_id)?;
        if !final_sort_limit_fusion_allowed(self.plan, sort) {
            return Ok(None);
        }
        Ok(Some((limit_node_id, limit.clone())))
    }
}

fn final_sort_limit_fusion_allowed(plan: &QueryPlan, sort: &SortNode) -> bool {
    !matches!(
        plan.node(sort.input).map(|node| &node.kind),
        Some(PlanNode::Source(source)) if matches!(source.as_ref(), SourceNode::GpuResident(_))
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::final_sort_limit_fusion_allowed;
    use crate::{
        expr::{Expr, LimitSpec, SortExpr, SortSpec},
        plan::{PlanNode, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetSource, SourceNode},
    };

    #[test]
    fn final_sort_limit_fusion_excludes_only_direct_gpu_resident_sources() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));

        let mut resident_builder = QueryPlanBuilder::new();
        let resident = resident_builder
            .add_source(SourceNode::GpuResident(GpuResidentSource::new("resident", Arc::clone(&schema))));
        let resident_sort = resident_builder.add_sort(
            resident,
            SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]),
            Arc::clone(&schema),
        );
        let resident_limit =
            resident_builder.add_limit(resident_sort, LimitSpec::new(0, Some(100)), Arc::clone(&schema));
        let resident_plan = resident_builder.finish_with_host_sink(resident_limit);
        let PlanNode::Sort(resident_sort_node) = &resident_plan.node(resident_sort).unwrap().kind else {
            panic!("resident sort node must remain a sort")
        };
        assert!(!final_sort_limit_fusion_allowed(&resident_plan, resident_sort_node));

        let mut parquet_builder = QueryPlanBuilder::new();
        let parquet = parquet_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/final-sort-limit-fusion.parquet",
            Arc::clone(&schema),
            vec!["value".to_owned()],
        )));
        let parquet_sort = parquet_builder.add_sort(
            parquet,
            SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]),
            Arc::clone(&schema),
        );
        let parquet_limit = parquet_builder.add_limit(parquet_sort, LimitSpec::new(0, Some(100)), schema);
        let parquet_plan = parquet_builder.finish_with_host_sink(parquet_limit);
        let PlanNode::Sort(parquet_sort_node) = &parquet_plan.node(parquet_sort).unwrap().kind else {
            panic!("Parquet sort node must remain a sort")
        };
        assert!(final_sort_limit_fusion_allowed(&parquet_plan, parquet_sort_node));
    }
}
