use super::*;

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_limit_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let input = single_input_resource(task)?;
        let output = single_output_resource(task)?;
        let limit = limit_node(self.plan, task, node_id)?;
        if state.resource_payload(input)?.applied_limit() == Some(node_id) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = input_payload.payload else {
                return Err(invalid("compiled fused limit can only pass through resident frame input")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id));
            };
            return Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(*frame))]));
        }

        let input_payload = state.resource_payload(input)?;
        let frame = execute_bounded_generic_unary_from_frame_with_reservation_kind(
            self.context,
            node_id,
            "compiled_limit",
            "compiled_limit_output",
            input_payload.clone_frame(),
            input_payload.reservation_bytes(),
            ReservationKind::FinalSort,
            &mut self.reservations,
            self.metrics,
            false,
            |context, input, _token, metrics| crate::exec::operators::execute_limit(context, limit, input, metrics),
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    pub(super) fn execute_union_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let Some(node) = self.plan.node(node_id) else {
            return Err(invalid("compiled union task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        if !matches!(node.kind, PlanNode::Union(_)) {
            return Err(invalid("compiled union task does not reference a union plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let inputs = self.union_task_inputs(task, state, node_id)?;
        let frame = execute_bounded_generic_union(self.context, node_id, inputs, &mut self.reservations, self.metrics)
            .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    pub(super) fn union_task_inputs(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
        node_id: NodeId,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let mut inputs = Vec::new();
        for resource in task.input_resources() {
            let result = if state.can_consume_task_input_resource(task, *resource) {
                let payload = state.consume_task_input_resource(task, *resource)?.into_payload();
                self.union_payload_to_inputs(task, node_id, *resource, payload, true, &mut inputs)
            } else {
                let payload = state.resource_payload(*resource)?;
                self.union_payload_ref_to_inputs(task, node_id, *resource, payload, &mut inputs)
            };
            if let Err(error) = result {
                release_reservations_reverse(
                    &mut self.reservations,
                    inputs.into_iter().map(|input| input.reservation).collect(),
                    self.metrics,
                );
                return Err(error);
            }
        }
        Ok(inputs)
    }

    pub(super) fn union_payload_to_inputs(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        resource: ResidentResourceId,
        payload: CompiledGpuResource,
        consumed: bool,
        inputs: &mut Vec<ResidentGpuFrame>,
    ) -> Result<()> {
        match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => inputs.push(*frame),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                inputs.extend(stream.into_remaining_frames_owned(&mut self.reservations, self.metrics)?)
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                if !consumed {
                    return Err(invalid("compiled union task cannot share a deferred join leaf source")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id)
                        .with_fact("resident_resource_id", resource));
                }
                let stream = self.apply_dynamic_filters_to_source_stream(*stream)?;
                inputs.push(materialize_bounded_join_leaf_local_parquet_source_stream(
                    self.context,
                    stream,
                    &mut self.reservations,
                    self.metrics,
                )?);
            },
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                if !consumed {
                    return Err(invalid("compiled union task cannot share a pending source read")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id)
                        .with_fact("resident_resource_id", resource));
                }
                inputs.push(prefetch.finish(self.context, &mut self.reservations, self.metrics)?);
            },
        }
        Ok(())
    }

    pub(super) fn union_payload_ref_to_inputs(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        resource: ResidentResourceId,
        payload: &CompiledGpuResource,
        inputs: &mut Vec<ResidentGpuFrame>,
    ) -> Result<()> {
        match &payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                inputs.push(ResidentGpuFrame {
                    frame: frame.frame.clone(),
                    reservation: self.union_alias_reservation(node_id, &frame.reservation),
                });
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                for frame in stream.frames() {
                    inputs.push(ResidentGpuFrame {
                        frame: frame.frame.clone(),
                        reservation: self.union_alias_reservation(node_id, &frame.reservation),
                    });
                }
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {
                return Err(invalid("compiled union task cannot share a deferred join leaf source")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("resident_resource_id", resource));
            },
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => {
                return Err(invalid("compiled union task cannot share a pending source read")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("resident_resource_id", resource));
            },
        }
        Ok(())
    }

    pub(super) fn union_alias_reservation(&mut self, node_id: NodeId, source: &ActiveReservation) -> ActiveReservation {
        self.reservations.alias_reservation(
            source,
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, source.bytes(), self.reservations.budget())
                .with_memory_boundary("compiled_union_input_alias")
                .with_node_id(node_id)
                .with_stage("compiled_union_input_alias")
                .with_attempt(1, 1),
            self.metrics,
        )
    }

    pub(super) fn execute_sink_materialize_task(
        &mut self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        if self.pipeline.sink().kind() != SinkBoundaryKind::HostArrowMaterialization
            || self.pipeline.sink().input() != task.id()
        {
            return Err(invalid("compiled sink materialize task does not match the pipeline sink boundary")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str())
                .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
        }
        if self.host_output.is_some() {
            return Err(invalid("compiled host sink materialized more than once")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        }
        let input = single_input_resource(task)?;
        let input_payload = state.resource_payload(input)?;
        let ResultSink::HostArrow { input: final_node_id, schema } = self.plan.sink() else {
            return Err(invalid("compiled sink materialize task requires a host Arrow sink")
                .with_fact("compiled_task_id", task.id())
                .with_fact("pipeline_task_kind", task.kind().as_str()));
        };
        let executable_schema = self
            .plan
            .node(*final_node_id)
            .ok_or_else(|| {
                invalid("compiled sink task references a missing final plan node")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", final_node_id)
            })?
            .output_schema();
        let output = materialize_bounded_frame_to_host(
            self.context,
            input_payload.clone_frame(),
            input_payload.reservation_bytes(),
            *final_node_id,
            executable_schema,
            schema,
            &mut self.reservations,
            self.metrics,
        )
        .map_err(|error| {
            error
                .with_fact_if_absent("compiled_task_id", task.id())
                .with_fact_if_absent("pipeline_task_kind", task.kind().as_str())
                .with_fact_if_absent("resident_resource_id", input)
                .with_fact_if_absent("plan_node_id", final_node_id)
                .with_fact_if_absent("sink_boundary_kind", self.pipeline.sink().kind().as_str())
        })?;
        self.host_output = Some(output);
        Ok(CompiledTaskOutput::empty())
    }
}
