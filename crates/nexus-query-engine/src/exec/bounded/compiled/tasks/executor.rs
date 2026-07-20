use super::*;

mod dynamic_filters;

fn join_build_key_uniqueness_probe_can_fallback(error: &QueryError) -> bool {
    matches!(
        error.code(),
        QueryErrorCode::CudfResourceExhausted
            | QueryErrorCode::CudfTemporaryFailure
            | QueryErrorCode::MemoryReservationFailed
            | QueryErrorCode::ServiceOverloaded
    )
}

/// Compiled task interpreter for bounded GPU work.
#[doc(hidden)]
pub struct CompiledGpuTaskExecutor<'a> {
    context: &'a ExecutionContext,
    plan: &'a QueryPlan,
    pipeline: &'a CompiledPipeline,
    gpu_inputs: &'a GpuResidentInputMap,
    metrics: &'a mut QueryMetrics,
    reservations: ReservationTracker,
    dynamic_filters: DynamicFilterSet,
    source_post_decode_applied_routes: BTreeMap<ResidentResourceId, BTreeSet<DynamicFilterId>>,
    join_builds: BTreeMap<NodeId, BoundedCompiledJoinBuildState>,
    join_probes: BTreeMap<NodeId, BoundedCompiledJoinProbeState>,
    host_output: Option<RecordBatch>,
    cooperative_shrink_applied_since_last_cycle: bool,
}

impl<'a> CompiledGpuTaskExecutor<'a> {
    pub fn new(
        context: &'a ExecutionContext,
        plan: &'a QueryPlan,
        pipeline: &'a CompiledPipeline,
        gpu_inputs: &'a GpuResidentInputMap,
        metrics: &'a mut QueryMetrics,
        observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
            return Err(invalid("compiled bounded task execution requires bounded memory control")
                .with_fact("memory_control_policy", context.memory_control_policy().as_str()));
        };
        let reservations = ReservationTracker::new_for_compiled_context(
            context,
            plan,
            pipeline,
            device_budget_bytes,
            metrics,
            observation_scope,
        )?;
        Ok(Self {
            context,
            plan,
            pipeline,
            gpu_inputs,
            metrics,
            reservations,
            dynamic_filters: DynamicFilterSet::new(),
            source_post_decode_applied_routes: BTreeMap::new(),
            join_builds: BTreeMap::new(),
            join_probes: BTreeMap::new(),
            host_output: None,
            cooperative_shrink_applied_since_last_cycle: false,
        })
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reservations.reserved_bytes()
    }

    pub fn remaining_bytes(&self) -> u64 {
        self.reservations.remaining_bytes()
    }

    fn acknowledge_requested_grant_shrink(&mut self) -> Result<bool> {
        let budget_before = self.reservations.budget();
        let request_satisfied = self.reservations.acknowledge_requested_grant_shrink()?;
        if self.reservations.budget() < budget_before {
            self.cooperative_shrink_applied_since_last_cycle = true;
        }
        Ok(request_satisfied)
    }

    fn select_join_build_strategy(
        &mut self,
        task: &PipelineTask,
        node_id: NodeId,
        node: &JoinNode,
        build_frame: &GpuDataFrame,
        build_side_bytes: u64,
    ) -> Result<CompiledJoinStrategySelection> {
        let observed_build_rows = build_frame.num_rows() as u64;
        let mut build_key_uniqueness_probe_status = None;
        let mut build_key_uniqueness_probe_error_code = None;
        let build_key_unique = if join_strategy_needs_build_key_uniqueness_probe(
            self.plan,
            node,
            node_id,
            observed_build_rows,
            build_side_bytes,
        ) {
            match measure_join_build_key_unique(self.context, node, build_frame, &self.reservations, self.metrics) {
                Ok(unique) => {
                    build_key_uniqueness_probe_status = Some("measured");
                    Some(unique)
                },
                Err(error) if join_build_key_uniqueness_probe_can_fallback(&error) => {
                    build_key_uniqueness_probe_status = Some("fallback_temporary_failure");
                    build_key_uniqueness_probe_error_code = Some(error.code().as_str().to_string());
                    None
                },
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let mut strategy =
            join_strategy(self.plan, node, task, node_id, observed_build_rows, build_side_bytes, build_key_unique)?;
        if let Some(facts) = strategy.facts.as_mut() {
            facts.insert(
                "build_key_unique",
                build_key_unique.map_or_else(|| "none".to_string(), |unique| unique.to_string()),
            );
            facts.insert(
                "build_key_uniqueness_probe_status",
                build_key_uniqueness_probe_status.unwrap_or("none").to_string(),
            );
            facts.insert(
                "build_key_uniqueness_probe_error_code",
                build_key_uniqueness_probe_error_code.unwrap_or_else(|| "none".to_string()),
            );
        }
        if let Some(facts) = strategy.facts.as_ref() {
            self.metrics.record_join_strategy_decision(node_id, facts);
        }
        Ok(strategy)
    }

    pub fn take_host_output(&mut self) -> Result<RecordBatch> {
        self.host_output.take().ok_or_else(|| {
            invalid("compiled host sink did not produce a host Arrow output")
                .with_fact("compiled_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str())
        })
    }

    pub fn take_gpu_output(
        &mut self,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<GpuResultHandle> {
        if self.pipeline.sink().kind() != SinkBoundaryKind::GpuResidentOutput {
            return Err(invalid("compiled GPU output requires a GPU-resident sink boundary")
                .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
        }
        let output = state.take_sink_output_resource(self.pipeline)?;
        let resource = output.id();
        let payload = output.into_payload();
        let frame = match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame,
            other => {
                <Self as CompiledTaskExecutor>::release_resource(
                    self,
                    resource,
                    CompiledGpuResource { payload: other },
                )?;
                return Err(invalid("compiled GPU sink output must be a resident frame")
                    .with_fact("sink_boundary_task_id", self.pipeline.sink().input())
                    .with_fact("resident_resource_id", resource)
                    .with_fact("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
            },
        };
        let ResidentGpuFrame { frame, reservation } = *frame;
        let live_reservation = reservation.retain_grant_live_reservation();
        let handle = match GpuResultHandle::new_with_live_reservation(
            self.context,
            frame,
            self.plan.output_schema().clone(),
            live_reservation,
        ) {
            Ok(handle) => handle,
            Err(error) => {
                self.reservations.release(reservation, self.metrics);
                return Err(error
                    .with_fact_if_absent("sink_boundary_task_id", self.pipeline.sink().input())
                    .with_fact_if_absent("resident_resource_id", resource)
                    .with_fact_if_absent("sink_boundary_kind", self.pipeline.sink().kind().as_str()));
            },
        };
        self.reservations.release(reservation, self.metrics);
        self.acknowledge_requested_grant_shrink()?;
        Ok(handle)
    }

    fn can_admit_with_current_reservations(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskAdmission> {
        let remaining_bytes = self.remaining_bytes();
        let estimated_required_bytes = self.estimated_task_admission_bytes(task, state)?;
        let reclaimable_input_bytes = self.reclaimable_input_bytes(task, state)?;
        let available_bytes = remaining_bytes.saturating_add(reclaimable_input_bytes);
        if estimated_required_bytes <= available_bytes || self.task_has_source_retry_path(task, state)? {
            return Ok(CompiledTaskAdmission::admitted_with_bytes(estimated_required_bytes, remaining_bytes));
        }
        Ok(CompiledTaskAdmission::not_ready(
            estimated_required_bytes,
            remaining_bytes,
            "estimated_task_reservation_exceeds_remaining_bytes",
        ))
    }

    fn estimated_task_admission_bytes(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<u64> {
        let input_estimate = task.input_resources().iter().try_fold(0_u64, |total, resource| {
            let payload = state.resource_payload(*resource)?;
            Ok(total.saturating_add(self.resource_admission_estimate_bytes(payload)))
        })?;
        let operator_overlap = self.stream_consumer_operator_admission_bytes(task, state)?;
        let join_materialize_floor = self.join_materialize_admission_floor_bytes(task)?;
        Ok(input_estimate.saturating_add(operator_overlap).saturating_add(join_materialize_floor))
    }

    fn join_materialize_admission_floor_bytes(&self, task: &PipelineTask) -> Result<u64> {
        if task.kind() != PipelineTaskKind::JoinMaterialize {
            return Ok(0);
        }
        let node_id = plan_node(task)?;
        let Some(probe) = self.join_probes.get(&node_id) else {
            return Ok(0);
        };
        let Some(node) = self.plan.node(node_id) else {
            return Err(invalid("compiled join materialize task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id));
        };
        let output_columns = node
            .output_schema()
            .fields()
            .iter()
            .map(|field| crate::exec::JoinOutputColumnUpperBoundInput {
                data_type: field.data_type().clone(),
                input_max_string_length_bytes: string_floor_max_bytes(field.data_type()),
                input_total_string_bytes: None,
            })
            .collect::<Vec<_>>();
        let output_floor = crate::exec::join_output_upper_bound_bytes(1, &output_columns)?;
        let gather_map_floor =
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(1).map(|bytes| bytes as u64).unwrap_or(64);
        let stream_target_bytes = reusable_hash_join_stream_target_output_bytes(self.context, &self.reservations);
        let stream_quantum_bytes = BoundedPartitionTargetPolicy::default().join_materialize_stream_quantum_bytes();
        Ok(output_floor
            .saturating_add(gather_map_floor)
            .max(stream_quantum_bytes)
            .max(probe.materialize_admission_bytes(stream_target_bytes)))
    }

    fn stream_consumer_operator_admission_bytes(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<u64> {
        if !matches!(task.kind(), PipelineTaskKind::Aggregate | PipelineTaskKind::FinalSort) {
            return Ok(0);
        }
        task.input_resources().iter().try_fold(0_u64, |total, resource| {
            let payload = state.resource_payload(*resource)?;
            match &payload.payload {
                CompiledGpuResourcePayload::ResidentFrameStream(_) => {
                    Ok(total.saturating_add(self.resource_admission_estimate_bytes(payload)))
                },
                CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
                    if task.kind() == PipelineTaskKind::Aggregate =>
                {
                    Ok(total.saturating_add(self.resource_admission_estimate_bytes(payload)))
                },
                CompiledGpuResourcePayload::ResidentFrame { .. }
                | CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
                | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => Ok(total),
            }
        })
    }

    fn reclaimable_input_bytes(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<u64> {
        task.input_resources().iter().try_fold(0_u64, |total, resource| {
            if state.can_consume_task_input_resource(task, *resource) {
                let payload = state.resource_payload(*resource)?;
                Ok(total.saturating_add(payload.reservation_bytes()))
            } else {
                Ok(total)
            }
        })
    }

    fn resource_admission_estimate_bytes(&self, resource: &CompiledGpuResource) -> u64 {
        // Per Milestone 4.B / design v5: scheduler admission consumes the
        // worst-case-per-step footprint, not the aggregate. For
        // Vec-backed payloads this equals `aggregate_reservation_bytes`
        // because the consumer holds every frame simultaneously; for
        // lazy producers (Milestone 4.E.3a onward) this is the max
        // per-partition upper bound.
        match &resource.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => frame.reservation.bytes(),
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => stream.max_per_frame_reservation_bytes(),
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => prefetch.reservation_bytes(),
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => stream
                .estimated_source_read_bytes()
                .unwrap_or_else(|| self.coarse_reservation_slot_bytes())
                .min(self.coarse_reservation_slot_bytes()),
        }
    }

    fn coarse_reservation_slot_bytes(&self) -> u64 {
        match self.context.memory_control_policy() {
            MemoryControlPolicy::Bounded { device_budget_bytes, max_source_chunk_bytes, .. } => {
                max_source_chunk_bytes.min(device_budget_bytes).max(1)
            },
            MemoryControlPolicy::Unbounded => 0,
        }
    }

    fn task_has_source_retry_path(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<bool> {
        if task.kind() == PipelineTaskKind::SourceRead {
            return Ok(true);
        }
        task.input_resources().iter().try_fold(false, |has_stream, resource| {
            let payload = state.resource_payload(*resource)?;
            Ok(has_stream
                || matches!(
                    &payload.payload,
                    CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
                        | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_)
                ))
        })
    }

    fn execute_source_task(&mut self, task: &PipelineTask) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
        let node_id = plan_node(task)?;
        let output = single_output_resource(task)?;
        let node = self.plan.node(node_id).ok_or_else(|| {
            invalid("compiled source task references a missing plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
        })?;
        let resource = match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) if !source.has_position_delete_visible_scan_limit() => {
                    let has_multiple_consumers = self.resource_has_multiple_consumers(output);
                    let can_defer_source = !has_multiple_consumers
                        || self.shared_source_consumers_can_clone_source_stream(task.id(), output);
                    let feeds_direct_join = self.task_feeds_direct_join_input(task.id());
                    let feeds_join_through_unary = self.task_feeds_any_join_input_through_unary_chain(task.id());
                    let feeds_aggregate = self.task_feeds_aggregate_input(task.id());
                    let has_aggregate_source_adjacent_stages =
                        self.task_feeds_aggregate_input_through_unary_chain(task.id());
                    let has_join_source_adjacent_stages = feeds_join_through_unary;
                    let feeds_join = feeds_direct_join || feeds_join_through_unary;
                    let has_dynamic_filter_routes = self.source_has_dynamic_filter_routes(node_id);
                    if !can_defer_source && feeds_join {
                        let stream = plan_bounded_join_leaf_local_parquet_source_stream(
                            self.context,
                            source,
                            node_id,
                            has_join_source_adjacent_stages,
                            self.metrics,
                        )?;
                        materialize_bounded_join_leaf_local_parquet_source_stream(
                            self.context,
                            stream,
                            &mut self.reservations,
                            self.metrics,
                        )
                        .map(CompiledGpuResource::new)
                    } else if !can_defer_source || (!feeds_join && !feeds_aggregate && !has_dynamic_filter_routes) {
                        execute_bounded_generic_local_parquet_source(
                            self.context,
                            source,
                            node_id,
                            &mut self.reservations,
                            self.metrics,
                        )
                        .map(CompiledGpuResource::new)
                    } else {
                        let has_source_adjacent_stages =
                            has_join_source_adjacent_stages || has_aggregate_source_adjacent_stages;
                        let stream = if feeds_aggregate && (feeds_direct_join || feeds_join_through_unary) {
                            plan_bounded_shared_join_aggregate_local_parquet_source_stream(
                                self.context,
                                source,
                                node_id,
                                has_source_adjacent_stages,
                                self.metrics,
                            )?
                        } else if feeds_aggregate {
                            plan_bounded_aggregate_input_local_parquet_source_stream(
                                self.context,
                                source,
                                node_id,
                                has_aggregate_source_adjacent_stages,
                                self.metrics,
                            )?
                        } else {
                            plan_bounded_join_leaf_local_parquet_source_stream(
                                self.context,
                                source,
                                node_id,
                                has_join_source_adjacent_stages,
                                self.metrics,
                            )?
                        };
                        if feeds_direct_join
                            && !has_multiple_consumers
                            && !has_dynamic_filter_routes
                            && let Some(prefetch) = try_start_bounded_join_leaf_local_parquet_source_prefetch(
                                self.context,
                                &stream,
                                &mut self.reservations,
                                self.metrics,
                            )?
                        {
                            return Ok(CompiledTaskOutput::new(vec![(
                                output,
                                CompiledGpuResource::pending_join_leaf_local_parquet_source(prefetch),
                            )]));
                        }
                        if feeds_aggregate
                            || has_multiple_consumers
                            || has_join_source_adjacent_stages
                            || stream.chunk_count() > 1
                            || has_dynamic_filter_routes
                        {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(stream))
                        } else {
                            materialize_bounded_join_leaf_local_parquet_source_stream(
                                self.context,
                                stream,
                                &mut self.reservations,
                                self.metrics,
                            )
                            .map(CompiledGpuResource::new)
                        }
                    }
                },
                source if self.task_feeds_join_input(task.id()) => execute_bounded_join_leaf_source(
                    self.context,
                    source,
                    node_id,
                    self.gpu_inputs,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
                SourceNode::LocalParquet(source) => execute_bounded_generic_local_parquet_source(
                    self.context,
                    source,
                    node_id,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
                SourceNode::GpuResident(source) => execute_bounded_generic_gpu_resident_source(
                    self.context,
                    source,
                    node_id,
                    self.gpu_inputs,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
                SourceNode::LiteralBatch(source) => execute_bounded_generic_literal_batch_source(
                    self.context,
                    source,
                    node_id,
                    &mut self.reservations,
                    self.metrics,
                )
                .map(CompiledGpuResource::new),
            },
            _ => Err(invalid("compiled source task does not reference a source plan node")
                .with_fact("compiled_task_id", task.id())
                .with_fact("plan_node_id", node_id)
                .with_fact("pipeline_task_kind", task.kind().as_str())),
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, resource)]))
    }

    fn execute_unary_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
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
                    let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
                    match input_payload.payload {
                        CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                                stream.with_filter_stage(node_id, filter.clone()),
                            ))
                        },
                        CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                            let frame = self.apply_dynamic_filters_to_filter_frame(node_id, filter, *frame)?;
                            execute_bounded_generic_unary(
                                self.context,
                                node_id,
                                "compiled_filter",
                                "compiled_filter_output",
                                frame,
                                &mut self.reservations,
                                self.metrics,
                                true,
                                |context, input, token, metrics| {
                                    crate::exec::operators::execute_filter_with_budget(
                                        context,
                                        filter,
                                        input,
                                        token.expect("budget token is created for compiled bounded filter stages"),
                                        metrics,
                                    )
                                },
                            )
                            .map(CompiledGpuResource::new)
                        },
                        CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                            let stream = self.apply_dynamic_filters_to_filter_stream(node_id, filter, stream)?;
                            let stream =
                                map_filter_resident_frame_stream(self.context, node_id, stream, filter.clone());
                            if self.resource_has_multiple_consumers(output) {
                                Ok(CompiledGpuResource::frame_stream(
                                    stream.into_remaining_frames_owned(&mut self.reservations, self.metrics)?,
                                ))
                            } else {
                                Ok(CompiledGpuResource::frame_stream_payload(stream))
                            }
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
                    } else {
                        self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                        let input_payload = state.resource_payload(input)?;
                        if let Some(frame) = self.apply_dynamic_filters_to_filter_shared_frame(
                            node_id,
                            filter,
                            input_payload.clone_frame(),
                            input_payload.reservation_bytes(),
                        )? {
                            execute_bounded_generic_unary(
                                self.context,
                                node_id,
                                "compiled_filter",
                                "compiled_filter_output",
                                frame,
                                &mut self.reservations,
                                self.metrics,
                                true,
                                |context, input, token, metrics| {
                                    crate::exec::operators::execute_filter_with_budget(
                                        context,
                                        filter,
                                        input,
                                        token.expect("budget token is created for compiled bounded filter stages"),
                                        metrics,
                                    )
                                },
                            )
                            .map(CompiledGpuResource::new)
                        } else {
                            execute_bounded_generic_unary_from_frame(
                                self.context,
                                node_id,
                                "compiled_filter",
                                "compiled_filter_output",
                                input_payload.clone_frame(),
                                input_payload.reservation_bytes(),
                                &mut self.reservations,
                                self.metrics,
                                true,
                                |context, input, token, metrics| {
                                    crate::exec::operators::execute_filter_with_budget(
                                        context,
                                        filter,
                                        input,
                                        token.expect("budget token is created for compiled bounded filter stages"),
                                        metrics,
                                    )
                                },
                            )
                            .map(CompiledGpuResource::new)
                        }
                    }
                }
            },
            PlanNode::Projection(projection) => {
                if state.can_consume_task_input_resource(task, input) {
                    let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
                    match input_payload.payload {
                        CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream) => {
                            Ok(CompiledGpuResource::join_leaf_local_parquet_source(
                                stream.with_projection_stage(node_id, projection.clone()),
                            ))
                        },
                        CompiledGpuResourcePayload::ResidentFrame { frame, .. } => execute_bounded_generic_unary(
                            self.context,
                            node_id,
                            "compiled_projection",
                            "compiled_projection_output",
                            *frame,
                            &mut self.reservations,
                            self.metrics,
                            true,
                            |context, input, token, metrics| {
                                crate::exec::operators::execute_projection_with_budget(
                                    context,
                                    projection,
                                    input,
                                    token.expect("budget token is created for compiled bounded projection stages"),
                                    metrics,
                                )
                            },
                        )
                        .map(CompiledGpuResource::new),
                        CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                            let stream =
                                map_projection_resident_frame_stream(self.context, node_id, stream, projection.clone());
                            if self.resource_has_multiple_consumers(output) {
                                Ok(CompiledGpuResource::frame_stream(
                                    stream.into_remaining_frames_owned(&mut self.reservations, self.metrics)?,
                                ))
                            } else {
                                Ok(CompiledGpuResource::frame_stream_payload(stream))
                            }
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
                    } else {
                        self.apply_ready_dynamic_filters_to_shared_source_resource(state, input)?;
                        let input_payload = state.resource_payload(input)?;
                        execute_bounded_generic_unary_from_frame(
                            self.context,
                            node_id,
                            "compiled_projection",
                            "compiled_projection_output",
                            input_payload.clone_frame(),
                            input_payload.reservation_bytes(),
                            &mut self.reservations,
                            self.metrics,
                            true,
                            |context, input, token, metrics| {
                                crate::exec::operators::execute_projection_with_budget(
                                    context,
                                    projection,
                                    input,
                                    token.expect("budget token is created for compiled bounded projection stages"),
                                    metrics,
                                )
                            },
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
        Ok(CompiledTaskOutput::new(vec![(output, frame)]))
    }

    fn execute_join_build_task(
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

    fn execute_join_probe_task(
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

    fn execute_join_materialize_task(
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

    fn execute_aggregate_task(
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
                        self.context,
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
                        &mut self.reservations,
                        self.metrics,
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
                self.context,
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
                &mut self.reservations,
                self.metrics,
            )
        } else if state.can_consume_task_input_resource(task, input) {
            let input_payload = state.consume_task_input_resource(task, input)?.into_payload();
            let CompiledGpuResourcePayload::ResidentFrame { frame, .. } = input_payload.payload else {
                return Err(invalid("compiled aggregate task expected a resident frame input")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("plan_node_id", node_id)
                    .with_fact("pipeline_task_kind", task.kind().as_str()));
            };
            let ResidentGpuFrame { frame: input_frame, reservation: input_reservation } = *frame;
            let input_reserved_bytes = input_reservation.bytes();
            let result = execute_compiled_bounded_aggregate_from_frame(
                self.context,
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
                &mut self.reservations,
                self.metrics,
            );
            self.reservations.release(input_reservation, self.metrics);
            match result {
                Ok(output) => {
                    if let Err(error) = self.acknowledge_requested_grant_shrink() {
                        let ResidentGpuFrame { frame, reservation } = output;
                        drop(frame);
                        self.reservations.release(reservation, self.metrics);
                        return Err(error);
                    }
                    Ok(output)
                },
                Err(error) => Err(error),
            }
        } else {
            let input_payload = state.resource_payload(input)?;
            if let Some(stream) = input_payload.as_join_leaf_local_parquet_source().cloned() {
                let stream = self.apply_dynamic_filters_to_source_stream(stream)?;
                execute_compiled_bounded_aggregate_from_source_stream(
                    self.context,
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
                    &mut self.reservations,
                    self.metrics,
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
                    self.context,
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
                    &mut self.reservations,
                    self.metrics,
                )
            }
        }
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        self.produce_aggregate_group_key_dynamic_filters(node_id, node.output_schema(), &frame.frame)?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn execute_window_task(
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
            self.context,
            node_id,
            "compiled_window",
            "compiled_window_output",
            input_payload.clone_frame(),
            input_payload.reservation_bytes(),
            &mut self.reservations,
            self.metrics,
            false,
            |context, input, _token, metrics| {
                crate::exec::operators::execute_window(context, window, &input_schema, input, &output_schema, metrics)
            },
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn execute_final_sort_task(
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
                        self.context,
                        node_id,
                        &sort,
                        Some(&limit),
                        *input_frame,
                        &mut self.reservations,
                        self.metrics,
                    )
                },
                CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                    execute_final_sort_node_from_stream_with_retry(
                        self.context,
                        node_id,
                        &sort,
                        &limit,
                        stream,
                        &mut self.reservations,
                        self.metrics,
                    )
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
            self.context,
            node_id,
            &sort,
            input_payload.clone_frame(),
            &mut self.reservations,
            self.metrics,
        )
        .map_err(|error| attach_task_facts(error, task, node_id))?;
        Ok(CompiledTaskOutput::new(vec![(output, CompiledGpuResource::new(frame))]))
    }

    fn downstream_fused_limit(
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
        Ok(Some((limit_node_id, limit.clone())))
    }

    fn execute_limit_task(
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

    fn execute_union_task(
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

    fn union_task_inputs(
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

    fn union_payload_to_inputs(
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

    fn union_payload_ref_to_inputs(
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

    fn union_alias_reservation(&mut self, node_id: NodeId, source: &ActiveReservation) -> ActiveReservation {
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

    fn execute_sink_materialize_task(
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

    fn task_feeds_join_input(&self, task: PipelineTaskId) -> bool {
        let mut pending = vec![task];
        let mut visited = BTreeSet::new();

        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }

            for edge in self.pipeline.edges().iter().filter(|edge| edge.from() == current) {
                match edge.kind() {
                    PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => return true,
                    PipelineEdgeKind::ResidentFrame => pending.push(edge.to()),
                    PipelineEdgeKind::Control
                    | PipelineEdgeKind::SharedResidentProducer
                    | PipelineEdgeKind::BarrierOutput
                    | PipelineEdgeKind::SinkInput => {},
                }
            }
        }

        false
    }

    fn task_feeds_direct_join_input(&self, task: PipelineTaskId) -> bool {
        self.pipeline.edges().iter().any(|edge| {
            edge.from() == task
                && matches!(edge.kind(), PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput)
        })
    }

    fn task_feeds_aggregate_input(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_aggregate_input_path(task).is_some()
    }

    fn task_feeds_aggregate_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_aggregate_input_path(task).unwrap_or(false)
    }

    fn task_feeds_aggregate_input_path(&self, task: PipelineTaskId) -> Option<bool> {
        let mut pending = vec![(task, false)];
        let mut visited = BTreeSet::new();

        while let Some((current, saw_unary)) = pending.pop() {
            if !visited.insert((current, saw_unary)) {
                continue;
            }

            for edge in self.pipeline.edges().iter().filter(|edge| edge.from() == current) {
                if !matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer) {
                    continue;
                }
                let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == edge.to())
                else {
                    continue;
                };
                match consumer_task.kind() {
                    PipelineTaskKind::Aggregate => return Some(saw_unary),
                    PipelineTaskKind::Unary => pending.push((edge.to(), true)),
                    _ => {},
                }
            }
        }

        None
    }

    fn resource_has_multiple_consumers(&self, resource: ResidentResourceId) -> bool {
        self.pipeline
            .resources()
            .iter()
            .find(|candidate| candidate.id() == resource)
            .is_some_and(|resource| resource.consumers().len() > 1)
    }

    fn shared_source_consumers_can_clone_source_stream(
        &self,
        task: PipelineTaskId,
        resource: ResidentResourceId,
    ) -> bool {
        let Some(resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource) else {
            return false;
        };
        if resource.consumers().len() <= 1 {
            return true;
        }
        let mut feeds_aggregate = false;
        let all_consumers_can_clone = resource.consumers().iter().all(|consumer| {
            let Some(edge) = self.pipeline.edges().iter().find(|edge| {
                edge.from() == task && edge.to() == *consumer && edge.resources().contains(&resource.id())
            }) else {
                return false;
            };
            match edge.kind() {
                PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => true,
                PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer => {
                    self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer).is_some_and(
                        |consumer_task| match consumer_task.kind() {
                            PipelineTaskKind::Aggregate => {
                                feeds_aggregate = true;
                                true
                            },
                            PipelineTaskKind::Unary => {
                                let unary_feeds_aggregate = self.task_feeds_aggregate_input(consumer_task.id());
                                feeds_aggregate |= unary_feeds_aggregate;
                                self.task_feeds_join_input(consumer_task.id()) || unary_feeds_aggregate
                            },
                            _ => false,
                        },
                    )
                },
                PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput => false,
            }
        });
        all_consumers_can_clone && feeds_aggregate
    }

    fn local_parquet_source_node_for_resource(&self, resource: ResidentResourceId) -> Option<NodeId> {
        let producer = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource)?.producer();
        let source_node_id = self.pipeline.tasks().iter().find(|candidate| candidate.id() == producer)?.plan_node()?;
        self.plan.node(source_node_id).and_then(|node| match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Some(source_node_id),
            _ => None,
        })
    }

    fn join_materialize_output_can_preserve_stream(&self, task: PipelineTaskId, resource: ResidentResourceId) -> bool {
        let Some(first_resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == resource) else {
            return false;
        };
        if first_resource.producer() != task {
            return false;
        }
        if first_resource.consumers().len() > 1 {
            return first_resource.consumers().iter().all(|consumer| {
                self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer).is_some_and(
                    |consumer_task| {
                        consumer_task.kind() == PipelineTaskKind::JoinProbe
                            && self.join_probe_task_can_consume_frame_stream(consumer_task)
                    },
                )
            });
        }
        let mut current_resource = resource;
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current_resource) {
                return false;
            }
            let Some(resource) = self.pipeline.resources().iter().find(|candidate| candidate.id() == current_resource)
            else {
                return false;
            };
            if resource.consumers().len() > 1 {
                return resource.consumers().iter().all(|consumer| {
                    self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer).is_some_and(
                        |consumer_task| {
                            consumer_task.kind() == PipelineTaskKind::JoinProbe
                                && self.join_probe_task_can_consume_frame_stream(consumer_task)
                        },
                    )
                });
            }
            let [consumer] = resource.consumers() else {
                return false;
            };
            let Some(consumer_task) = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer) else {
                return false;
            };
            match consumer_task.kind() {
                PipelineTaskKind::Aggregate => return true,
                PipelineTaskKind::JoinProbe => return self.join_probe_task_can_consume_frame_stream(consumer_task),
                PipelineTaskKind::FinalSort => return self.final_sort_task_can_consume_frame_stream(consumer_task),
                PipelineTaskKind::Unary => {
                    let [next_resource] = consumer_task.output_resources() else {
                        return false;
                    };
                    current_resource = *next_resource;
                },
                _ => return false,
            }
        }
    }

    fn final_sort_task_can_consume_frame_stream(&self, task: &PipelineTask) -> bool {
        let [output] = task.output_resources() else {
            return false;
        };
        self.downstream_fused_limit(task, *output).ok().flatten().is_some()
    }

    fn join_probe_task_can_consume_frame_stream(&self, task: &PipelineTask) -> bool {
        let Some(node_id) = task.plan_node() else {
            return false;
        };
        let Some(node) = self.plan.node(node_id) else {
            return false;
        };
        let PlanNode::Join(join) = &node.kind else {
            return false;
        };
        if join.spec.residual_predicate.is_some() {
            return matches!(
                CompiledJoinExecutionStrategy::from_plan_join_node(self.plan, node_id, join),
                Ok(Some(CompiledJoinExecutionStrategy::OneShotEqui))
            );
        }
        match join.spec.join_type {
            JoinType::Inner | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => true,
            JoinType::Left => join.spec.build_side == JoinSide::Right,
            JoinType::Full | JoinType::Cross => false,
        }
    }

    fn task_feeds_any_join_input_through_unary_chain(&self, task: PipelineTaskId) -> bool {
        self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinBuildInput).is_some()
            || self.task_feeds_join_input_through_unary_chain(task, PipelineEdgeKind::JoinProbeInput).is_some()
    }

    fn task_feeds_join_input_through_unary_chain(
        &self,
        task: PipelineTaskId,
        target_kind: PipelineEdgeKind,
    ) -> Option<PipelineTaskId> {
        let mut current = task;
        let mut saw_unary = false;
        loop {
            let current_task = self.pipeline.tasks().iter().find(|candidate| candidate.id() == current)?;
            let [output_resource] = current_task.output_resources() else {
                return None;
            };
            let resource = self.pipeline.resources().iter().find(|resource| resource.id() == *output_resource)?;
            let [consumer] = resource.consumers() else {
                return None;
            };
            let edge = self.pipeline.edges().iter().find(|edge| edge.from() == current && edge.to() == *consumer)?;
            match edge.kind() {
                kind if kind == target_kind && saw_unary => return Some(*consumer),
                kind if kind == target_kind => return None,
                PipelineEdgeKind::ResidentFrame => {
                    let consumer_task = self.pipeline.tasks().iter().find(|candidate| candidate.id() == *consumer)?;
                    if consumer_task.kind() != PipelineTaskKind::Unary {
                        return None;
                    }
                    saw_unary = true;
                    current = *consumer;
                },
                PipelineEdgeKind::Control
                | PipelineEdgeKind::SharedResidentProducer
                | PipelineEdgeKind::BarrierOutput
                | PipelineEdgeKind::JoinBuildInput
                | PipelineEdgeKind::JoinProbeInput
                | PipelineEdgeKind::SinkInput => return None,
            }
        }
    }
}

fn string_floor_max_bytes(data_type: &arrow::datatypes::DataType) -> Option<u64> {
    match data_type {
        arrow::datatypes::DataType::Utf8
        | arrow::datatypes::DataType::LargeUtf8
        | arrow::datatypes::DataType::Utf8View => Some(0),
        _ => None,
    }
}

impl CompiledTaskExecutor for CompiledGpuTaskExecutor<'_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        let attempt_id = crate::observability::task_attempt_id(task.id().to_string(), 1);
        let span_id = crate::observability::RuntimeSpanId::new(
            u64::try_from(task.id().index()).unwrap_or(u64::MAX).saturating_add(2),
        );
        self.execute_task_attempt(task, state, &attempt_id, span_id)
    }

    fn execute_task_attempt(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
        attempt_id: &crate::observability::TaskAttemptId,
        span_id: crate::observability::RuntimeSpanId,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        self.context.check_cancellation("nexus_query_engine.exec.bounded.compiled.task")?;
        validate_task_runtime_gates(self.pipeline, task)?;
        self.metrics.record_compiled_pipeline_task_mapping(task.id(), task.kind().as_str(), task.plan_node());
        self.metrics.record_compiled_pipeline_task_execution(task.id(), task.kind().as_str());
        let _task_observation_scope = self.reservations.enter_task_observation_scope(
            attempt_id.clone(),
            task.kind().as_str(),
            span_id,
            task_diagnostic_resident_resource(task),
        );
        let budget_event_start = self.metrics.budgeted_execution_event_count();
        let timer = Instant::now();
        let result = match task.kind() {
            PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::LiteralBatchSource => self.execute_source_task(task),
            PipelineTaskKind::Unary => self.execute_unary_task(task, state),
            PipelineTaskKind::JoinBuild => self.execute_join_build_task(task, state),
            PipelineTaskKind::JoinProbe => self.execute_join_probe_task(task, state),
            PipelineTaskKind::JoinMaterialize => self.execute_join_materialize_task(task),
            PipelineTaskKind::Aggregate => self.execute_aggregate_task(task, state),
            PipelineTaskKind::Window => self.execute_window_task(task, state),
            PipelineTaskKind::FinalSort => self.execute_final_sort_task(task, state),
            PipelineTaskKind::Limit => self.execute_limit_task(task, state),
            PipelineTaskKind::Union => self.execute_union_task(task, state),
            PipelineTaskKind::SinkMaterialize => self.execute_sink_materialize_task(task, state),
        };
        self.metrics.record_task_budget_observations_since(task.id(), budget_event_start);
        self.metrics.record_task_elapsed(task.id(), task.kind().as_str(), timer.elapsed());
        result
    }

    fn release_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        let _resource_observation_scope = self.reservations.enter_resource_observation_scope(resource);
        match payload.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => {
                self.reservations.release(frame.reservation, self.metrics);
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                stream.release_all(&mut self.reservations, self.metrics);
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_) => {},
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch) => {
                prefetch.abandon(&mut self.reservations, self.metrics);
            },
        }
        self.acknowledge_requested_grant_shrink()?;
        Ok(())
    }

    fn cleanup_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        let _cleanup_observation_scope = self
            .reservations
            .enter_release_observation_scope(crate::observability::RuntimeReservationDisposition::CleanupRelease);
        self.release_resource(resource, payload)
    }

    fn cleanup_after_error(&mut self) -> Result<()> {
        let _cleanup_observation_scope = self
            .reservations
            .enter_release_observation_scope(crate::observability::RuntimeReservationDisposition::CleanupRelease);
        while let Some((_, probe)) = self.join_probes.pop_first() {
            release_bounded_compiled_join_probe_state(probe, &mut self.reservations, self.metrics);
        }
        while let Some((_, build)) = self.join_builds.pop_first() {
            release_bounded_compiled_join_build_state(build, &mut self.reservations, self.metrics);
        }
        self.acknowledge_requested_grant_shrink()?;
        Ok(())
    }

    fn before_scheduling_cycle(&mut self) -> Result<CompiledSchedulingCycle> {
        self.acknowledge_requested_grant_shrink()?;
        if self.cooperative_shrink_applied_since_last_cycle {
            self.cooperative_shrink_applied_since_last_cycle = false;
            Ok(CompiledSchedulingCycle::after_cooperative_shrink_applied())
        } else {
            Ok(CompiledSchedulingCycle::default())
        }
    }

    fn live_reserved_bytes(&self) -> Option<u64> {
        Some(self.reserved_bytes())
    }

    fn remaining_reserved_bytes(&self) -> Option<u64> {
        Some(self.remaining_bytes())
    }

    fn can_admit_task(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskAdmission> {
        self.can_admit_with_current_reservations(task, state)
    }
}
