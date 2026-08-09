use super::*;

impl CompiledGpuTaskExecutor<'_> {
    pub(super) fn execute_source_task(
        &mut self,
        task: &PipelineTask,
    ) -> Result<CompiledTaskOutput<CompiledGpuResource>> {
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
                        let aggregate_chunk_cap_hints = self.source_adjacent_aggregate_chunk_cap_hints(task.id());
                        let stream = if feeds_aggregate && (feeds_direct_join || feeds_join_through_unary) {
                            plan_bounded_shared_join_aggregate_local_parquet_source_stream(
                                self.context,
                                source,
                                node_id,
                                has_source_adjacent_stages,
                                aggregate_chunk_cap_hints,
                                self.metrics,
                            )?
                        } else if feeds_aggregate {
                            plan_bounded_aggregate_input_local_parquet_source_stream(
                                self.context,
                                source,
                                node_id,
                                has_aggregate_source_adjacent_stages,
                                aggregate_chunk_cap_hints,
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
                    &self.gpu_inputs,
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
                    &self.gpu_inputs,
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
}
