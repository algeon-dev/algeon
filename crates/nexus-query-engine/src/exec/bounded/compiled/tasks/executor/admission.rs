use super::*;

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn select_join_build_strategy(
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
            match measure_join_build_key_unique(&mut self.execution, node, build_frame) {
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
            self.execution.metrics_mut().record_join_strategy_decision(node_id, facts);
        }
        Ok(strategy)
    }

    pub(super) fn can_admit_with_current_reservations(
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

    pub(super) fn estimated_task_admission_bytes(
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

    pub(super) fn join_materialize_admission_floor_bytes(&self, task: &PipelineTask) -> Result<u64> {
        const LIST_ONE_ROW_OUTPUT_FLOOR_BYTES: u64 = 72;

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
        let mut list_output_floor = 0u64;
        let output_columns = node
            .output_schema()
            .fields()
            .iter()
            .filter_map(|field| {
                if crate::datatypes::is_supported_list_carry_type(field.data_type()) {
                    list_output_floor = list_output_floor.saturating_add(LIST_ONE_ROW_OUTPUT_FLOOR_BYTES);
                    return None;
                }
                Some(crate::exec::JoinOutputColumnUpperBoundInput {
                    data_type: field.data_type().clone(),
                    input_max_string_length_bytes: string_floor_max_bytes(field.data_type()),
                    input_total_string_bytes: None,
                })
            })
            .collect::<Vec<_>>();
        let output_floor =
            crate::exec::join_output_upper_bound_bytes(1, &output_columns)?.saturating_add(list_output_floor);
        let gather_map_floor =
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(1).map(|bytes| bytes as u64).unwrap_or(64);
        let stream_target_bytes = reusable_hash_join_stream_target_output_bytes(&self.execution);
        let stream_quantum_bytes = BoundedPartitionTargetPolicy::default().join_materialize_stream_quantum_bytes();
        Ok(output_floor
            .saturating_add(gather_map_floor)
            .max(stream_quantum_bytes)
            .max(probe.materialize_admission_bytes(stream_target_bytes)))
    }

    pub(super) fn stream_consumer_operator_admission_bytes(
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

    pub(super) fn reclaimable_input_bytes(
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

    pub(super) fn resource_admission_estimate_bytes(&self, resource: &CompiledGpuResource) -> u64 {
        // Scheduler admission consumes the
        // worst-case-per-step footprint, not the aggregate. For
        // Vec-backed payloads this equals `aggregate_reservation_bytes`
        // because the consumer holds every frame simultaneously; for
        // lazy producers this is the max
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

    pub(super) fn coarse_reservation_slot_bytes(&self) -> u64 {
        match self.execution.context().memory_control_policy() {
            MemoryControlPolicy::Bounded { device_budget_bytes, max_source_chunk_bytes, .. } => {
                max_source_chunk_bytes.min(device_budget_bytes).max(1)
            },
            MemoryControlPolicy::Unbounded => 0,
        }
    }

    pub(super) fn task_has_source_retry_path(
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
}
