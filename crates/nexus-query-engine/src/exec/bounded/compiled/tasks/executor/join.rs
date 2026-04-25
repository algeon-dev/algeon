use super::*;

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
            let _resource_observation_scope = self.reservations.enter_resource_observation_scope(input);
            let input_frame = match input_payload.payload {
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                    let stream = self.apply_dynamic_filters_to_source_stream(*stream)?;
                    materialize_bounded_join_leaf_local_parquet_source_stream(
                        self.context,
                        stream,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                    prefetch.finish(self.context, &mut self.reservations, self.metrics)
                },
                CompiledGpuResourcePayload::ResidentFrame { frame, .. } => Ok(*frame),
                CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                    Err(invalid("compiled join build task cannot consume a resident frame stream")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id))
                },
            }?;
            self.produce_hash_join_build_dynamic_filters(node_id, node, &input_frame.frame)?;
            let strategy = self.select_join_build_strategy(
                task,
                node_id,
                node,
                &input_frame.frame,
                input_frame.reservation.bytes(),
            )?;
            execute_bounded_compiled_join_build_from_resident_frame(
                self.context,
                node_id,
                node,
                strategy.strategy,
                self.plan.keyed_hash_join_strategy_options(),
                input_frame,
                &mut self.reservations,
                self.metrics,
            )
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let stream = self.apply_dynamic_filters_to_source_stream(stream)?;
                let input_frame = materialize_bounded_join_leaf_local_parquet_source_stream(
                    self.context,
                    stream,
                    &mut self.reservations,
                    self.metrics,
                )?;
                self.produce_hash_join_build_dynamic_filters(node_id, node, &input_frame.frame)?;
                let strategy = self.select_join_build_strategy(
                    task,
                    node_id,
                    node,
                    &input_frame.frame,
                    input_frame.reservation.bytes(),
                )?;
                execute_bounded_compiled_join_build_from_resident_frame(
                    self.context,
                    node_id,
                    node,
                    strategy.strategy,
                    self.plan.keyed_hash_join_strategy_options(),
                    input_frame,
                    &mut self.reservations,
                    self.metrics,
                )
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
                self.produce_hash_join_build_dynamic_filters(node_id, node, &input_frame)?;
                let strategy = self.select_join_build_strategy(
                    task,
                    node_id,
                    node,
                    &input_frame,
                    input_payload.reservation_bytes(),
                )?;
                execute_bounded_compiled_join_build_from_frame(
                    self.context,
                    node_id,
                    node,
                    strategy.strategy,
                    self.plan.keyed_hash_join_strategy_options(),
                    input_frame,
                    input_payload.reservation(),
                    &mut self.reservations,
                    self.metrics,
                )
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.join_builds.insert(node_id, build);
        Ok(CompiledTaskOutput::empty())
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
        let probe = if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let _resource_observation_scope = self.reservations.enter_resource_observation_scope(input);
            match input_payload.payload {
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                    let stream = self.apply_dynamic_filters_to_join_leaf_stream(node_id, *stream)?;
                    execute_bounded_compiled_join_probe_from_stream(
                        self.context,
                        node_id,
                        node,
                        build,
                        stream,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                    let frame = prefetch.finish(self.context, &mut self.reservations, self.metrics)?;
                    let frame = self.apply_dynamic_filters_to_join_probe_frame(node_id, node, frame)?;
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        self.context,
                        node_id,
                        node,
                        build,
                        frame,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                    let frame = self.apply_dynamic_filters_to_join_probe_frame(node_id, node, *frame)?;
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        self.context,
                        node_id,
                        node,
                        build,
                        frame,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                    let stream = self.apply_dynamic_filters_to_join_probe_stream(node_id, node, stream)?;
                    execute_bounded_compiled_join_probe_from_frame_stream(
                        self.context,
                        node_id,
                        node,
                        build,
                        stream,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let stream = self.apply_dynamic_filters_to_join_leaf_stream(node_id, stream)?;
                execute_bounded_compiled_join_probe_from_stream(
                    self.context,
                    node_id,
                    node,
                    build,
                    stream,
                    &mut self.reservations,
                    self.metrics,
                )
            } else if let CompiledGpuResourcePayload::ResidentFrameStream(stream) = &input_payload.payload {
                if !self.join_probe_task_can_consume_frame_stream(task) {
                    release_bounded_compiled_join_build_state(build, &mut self.reservations, self.metrics);
                    return Err(invalid("compiled join probe cannot share this resident frame stream")
                        .with_fact("compiled_task_id", task.id())
                        .with_fact("plan_node_id", node_id)
                        .with_fact("resident_resource_id", input));
                }
                let stream = match stream.alias_eager(
                    node_id,
                    ReservationKind::ProbeSide,
                    "compiled_join_probe_stream_alias",
                    &mut self.reservations,
                    self.metrics,
                ) {
                    Ok(stream) => stream,
                    Err(error) => {
                        release_bounded_compiled_join_build_state(build, &mut self.reservations, self.metrics);
                        return Err(error);
                    },
                };
                let stream = self.apply_dynamic_filters_to_join_probe_stream(node_id, node, stream)?;
                execute_bounded_compiled_join_probe_from_frame_stream(
                    self.context,
                    node_id,
                    node,
                    build,
                    stream,
                    &mut self.reservations,
                    self.metrics,
                )
            } else if matches!(input_payload.payload, CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_))
            {
                release_bounded_compiled_join_build_state(build, &mut self.reservations, self.metrics);
                return Err(invalid("compiled join probe cannot share a pending source read")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("resident_resource_id", input));
            } else {
                self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                let input_payload = state.resource_payload(input)?;
                if let Some(frame) = self.apply_dynamic_filters_to_join_probe_shared_frame(
                    node_id,
                    node,
                    input_payload.clone_frame(),
                    input_payload.reservation_bytes(),
                )? {
                    execute_bounded_compiled_join_probe_from_resident_frame(
                        self.context,
                        node_id,
                        node,
                        build,
                        frame,
                        &mut self.reservations,
                        self.metrics,
                    )
                } else {
                    execute_bounded_compiled_join_probe_from_frame(
                        self.context,
                        node_id,
                        node,
                        build,
                        input_payload.clone_frame(),
                        input_payload.reservation(),
                        &mut self.reservations,
                        self.metrics,
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
        let resource = if self.join_materialize_output_can_preserve_stream(task.id(), output) {
            match execute_bounded_compiled_join_materialize_preserving_stream(
                self.context,
                node_id,
                node,
                probe,
                &mut self.reservations,
                self.metrics,
            )
            .map_err(|error| attach_task_facts(error, task, node_id))?
            {
                BoundedCompiledJoinMaterializeOutput::Frame(frame) => CompiledGpuResource::new(*frame),
                BoundedCompiledJoinMaterializeOutput::Stream(frames) => CompiledGpuResource::frame_stream(frames),
                BoundedCompiledJoinMaterializeOutput::LazyStream(stream) => {
                    if self.resource_has_multiple_consumers(output) {
                        CompiledGpuResource::frame_stream(
                            stream.into_remaining_frames_owned(&mut self.reservations, self.metrics)?,
                        )
                    } else {
                        CompiledGpuResource::frame_stream_payload(stream)
                    }
                },
            }
        } else {
            let frame = execute_bounded_compiled_join_materialize(
                self.context,
                node_id,
                node,
                probe,
                &mut self.reservations,
                self.metrics,
            )
            .map_err(|error| attach_task_facts(error, task, node_id))?;
            CompiledGpuResource::new(frame)
        };
        Ok(CompiledTaskOutput::new(vec![(output, resource)]))
    }
}
