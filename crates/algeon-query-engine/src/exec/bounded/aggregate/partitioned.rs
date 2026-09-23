use super::{stream::execute_compiled_bounded_partial_aggregate_from_frame, *};

/// Group-key hash partitioning of one aggregate input.
///
/// Every row of a group lands in exactly one partition, so each partition is a
/// self-contained aggregate input.
struct PartitionedAggregateInput {
    frame: cudf::DataFrame,
    offsets: Vec<usize>,
    partition_count: usize,
}

fn hash_partition_aggregate_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input: GpuDataFrame,
    partition_count: usize,
) -> Result<PartitionedAggregateInput> {
    let mut lowered_keys = lower_aggregate_spec_keys(&aggregate.spec)?;
    if aggregate.spec.grouping_sets.is_some() {
        lowered_keys.key_columns.truncate(aggregate.spec.group_exprs.len());
    }
    let frame = input.into_frame();
    let partitioned = {
        let token = match execution_budget_token_for_boundary(
            execution.context(),
            execution.device_budget_bytes(),
            "compiled_aggregate_hash_partition",
            "query_engine.bounded.aggregate.compiled_partitioned_hash.budget_token",
            "failed to create bounded native compiled partitioned aggregate budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                return Err(error);
            },
        };

        let input_rows = frame.num_rows() as u64;
        let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(execution.context());
        let stage_guard = RuntimeOperatorStageGuard::start(
            observation,
            plan_execution_id,
            parent_span_id,
            RuntimeOperatorStageKind::HashPartition,
            RuntimeOperatorStageIdentity::new().with_node_id(node_id.to_string()).with_input_rows(input_rows),
        );
        match frame
            .with_execution(&token.execution_overrides())
            .hash_partition_by(lowered_keys.key_columns.iter().map(String::as_str), partition_count)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    cudf_query_error(
                        "query_engine.bounded.aggregate.hash_partition",
                        "failed to hash-partition compiled bounded aggregate input",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("key_count", lowered_keys.key_columns.len())
                    .with_fact("partition_count", partition_count),
                    &token,
                )
            }) {
            Ok(partitioned) => {
                stage_guard.finish(Some(input_rows), None);
                Ok(partitioned)
            },
            Err(error) => {
                stage_guard.fail(&error);
                Err(error)
            },
        }
    };
    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    observe_device_memory_bytes(&partitioned_gpu, execution.metrics())?;
    Ok(PartitionedAggregateInput { frame: partitioned_frame, offsets, partition_count })
}

pub(super) fn execute_compiled_bounded_partitioned_aggregate_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    output_schema: &ArrowSchemaRef,
    input: GpuDataFrame,
    input_device_bytes: u64,
) -> Result<ResidentGpuFrame> {
    let partition_count = partitioned_aggregate_partition_count(execution.context(), input_device_bytes);
    let PartitionedAggregateInput { frame: partitioned_frame, offsets, partition_count } =
        hash_partition_aggregate_input(execution, node_id, aggregate, input, partition_count)?;

    let mut partials = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let partition_frame =
            match slice_compiled_aggregate_partition(execution, node_id, &partitioned_frame, &offsets, partition) {
                Ok(frame) => frame,
                Err(error) => {
                    return Err(error);
                },
            };
        match execute_compiled_bounded_partial_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partition_frame,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                return Err(error);
            },
        }
    }

    execute_compiled_bounded_final_aggregate_merge(execution, node_id, aggregate, input_schema, output_schema, partials)
}

fn slice_compiled_aggregate_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    partition: usize,
) -> Result<ResidentGpuFrame> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.aggregate.hash_partition_slice",
            "failed to slice compiled bounded hash-partitioned aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(gpu, bytes))
}

pub(super) fn should_partition_compiled_grouped_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_device_bytes: u64,
) -> bool {
    !aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec)
        && !aggregate_spec_has_median(&aggregate.spec)
        && !aggregate.spec.group_exprs.is_empty()
        && partitioned_aggregate_partition_count(context, input_device_bytes) > 1
}

pub(super) fn partitioned_aggregate_partition_count(context: &ExecutionContext, input_bytes: u64) -> usize {
    let MemoryControlPolicy { device_budget_bytes, .. } = context.memory_control_policy();
    let target = BoundedPartitionTargetPolicy::default().aggregate_partition_target_bytes(device_budget_bytes);
    let partitions = input_bytes.max(1).saturating_add(target.saturating_sub(1)) / target.max(1);
    partitions.clamp(1, 1024) as usize
}
