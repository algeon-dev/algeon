use super::{
    final_merge::concatenate_partitioned_final_merge_outputs,
    stream::execute_compiled_bounded_partial_aggregate_from_frame, *,
};

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
        let (observation, plan_execution_id, parent_span_id) =
            crate::observability::stage_context_from_execution(execution.context());
        let stage_guard = crate::observability::RuntimeOperatorStageGuard::start(
            observation,
            plan_execution_id,
            parent_span_id,
            crate::observability::RuntimeOperatorStageKind::HashPartition,
            crate::observability::RuntimeOperatorStageIdentity::new()
                .with_node_id(node_id.to_string())
                .with_input_rows(input_rows),
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
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
) -> Result<ResidentGpuFrame> {
    let partition_count = partitioned_aggregate_partition_count(execution.context(), input_reserved_bytes);
    let PartitionedAggregateInput { frame: partitioned_frame, offsets, partition_count } =
        hash_partition_aggregate_input(execution, node_id, aggregate, input, partition_count)?;

    let mut partials = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let partition_frame =
            match slice_compiled_aggregate_partition(execution, node_id, &partitioned_frame, &offsets, partition) {
                Ok(frame) => frame,
                Err(error) => {
                    release_partials_reverse(partials);
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
                release_partials_reverse(partials);
                return Err(error);
            },
        }
    }

    execute_compiled_bounded_final_aggregate_merge(execution, node_id, aggregate, input_schema, output_schema, partials)
}

/// Runs a whole-input grouped aggregate without holding every raw chunk at once.
///
/// STDDEV, MEDIAN, and exact DISTINCT cannot merge chunk-local partial states,
/// so they need each group's rows together. A group's rows all hash to the same
/// partition, so concatenating one partition across chunks rebuilds a complete
/// input for those kernels without rebuilding the whole query input, and the
/// per-partition outputs are disjoint by group.
pub(super) fn execute_bounded_partitioned_whole_input_aggregate_from_raw_chunks(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    raw_frames: Vec<CommittedPartial>,
    partition_count: usize,
) -> Result<CommittedPartial> {
    let node_id = bounded.aggregate_id;
    let aggregate = bounded.aggregate.0;
    let mut partition_inputs: Vec<Vec<CommittedPartial>> = (0..partition_count).map(|_| Vec::new()).collect();

    for raw in raw_frames {
        let partitioned =
            match hash_partition_aggregate_input(execution, node_id, aggregate, raw.frame, partition_count) {
                Ok(partitioned) => partitioned,
                Err(error) => {
                    release_nested_partials_reverse(partition_inputs);
                    return Err(error);
                },
            };
        let PartitionedAggregateInput { frame: partitioned_frame, offsets, .. } = partitioned;
        for partition in 0..partition_count {
            let slice =
                match slice_compiled_aggregate_partition(execution, node_id, &partitioned_frame, &offsets, partition) {
                    Ok(slice) => slice,
                    Err(error) => {
                        release_nested_partials_reverse(partition_inputs);
                        return Err(error);
                    },
                };
            if slice.frame.num_rows() == 0 {
                continue;
            }
            partition_inputs[partition].push(CommittedPartial { frame: slice.frame, device_bytes: slice.device_bytes });
        }
    }

    let mut outputs: Vec<CommittedPartial> = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let inputs = std::mem::take(&mut partition_inputs[partition]);
        if inputs.is_empty() {
            continue;
        }
        let partition_input = match concatenate_partitioned_final_merge_outputs(execution, node_id, inputs) {
            Ok(input) => input,
            Err(error) => {
                release_nested_partials_reverse(partition_inputs);
                release_partials_reverse(outputs);
                return Err(error);
            },
        };
        let input_bytes = partition_input.device_bytes();
        match execute_compiled_bounded_aggregate_from_frame(
            execution,
            node_id,
            aggregate,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            partition_input.frame,
            input_bytes,
        ) {
            Ok(output) => {
                outputs.push(CommittedPartial { frame: output.frame, device_bytes: output.device_bytes });
            },
            Err(error) => {
                release_nested_partials_reverse(partition_inputs);
                release_partials_reverse(outputs);
                return Err(error);
            },
        }
    }

    if outputs.is_empty() {
        let frame = execute_compiled_bounded_aggregate_from_empty_input(
            execution,
            node_id,
            aggregate,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
        )?;
        return Ok(CommittedPartial { frame: frame.frame, device_bytes: frame.device_bytes });
    }
    concatenate_partitioned_final_merge_outputs(execution, node_id, outputs)
}

/// Whether a whole-input aggregate can run per group-key partition instead of
/// over one concatenated frame.
///
/// Requires real group keys: a global aggregate has a single implicit group
/// whose rows cannot be split across partitions. Grouping sets are excluded
/// because their keys are lowered to a superset that does not partition the
/// rows each set actually groups by.
pub(super) fn should_partition_whole_input_grouped_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_reserved_bytes: u64,
) -> bool {
    !aggregate.spec.group_exprs.is_empty()
        && aggregate.spec.grouping_sets.is_none()
        && partitioned_aggregate_partition_count(context, input_reserved_bytes) > 1
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
    input_reserved_bytes: u64,
) -> bool {
    !aggregate_spec_has_floating_statistical_aggregate(&aggregate.spec)
        && !aggregate_spec_has_median(&aggregate.spec)
        && !aggregate.spec.group_exprs.is_empty()
        && partitioned_aggregate_partition_count(context, input_reserved_bytes) > 1
}

pub(super) fn partitioned_aggregate_partition_count(context: &ExecutionContext, input_bytes: u64) -> usize {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        return 1;
    };
    let target = BoundedPartitionTargetPolicy::default().aggregate_partition_target_bytes(device_budget_bytes);
    let partitions = input_bytes.max(1).saturating_add(target.saturating_sub(1)) / target.max(1);
    partitions.clamp(1, 1024) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::GroupingSetsSpec;

    #[test]
    fn should_partition_whole_input_grouped_aggregate_requires_real_group_keys() {
        // Above the 128 MiB default partition target at this budget, so only the
        // group/grouping-set shape can decide the outcome.
        const INPUT_BYTES: u64 = 1024 * 1024 * 1024;

        let context = ExecutionContext::builder()
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 4 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 32 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .build()
            .expect("execution context builds");
        assert!(partitioned_aggregate_partition_count(&context, INPUT_BYTES) > 1);

        let grouped = AggregateNode {
            input: NodeId::from_index_for_internal_use(1),
            spec: AggregateSpec::new(vec![Expr::Literal(LiteralValue::Utf8("k".to_string()))], Vec::new()),
        };
        assert!(should_partition_whole_input_grouped_aggregate(&context, &grouped, INPUT_BYTES));

        // One implicit group: splitting its rows would aggregate each partition
        // independently and give wrong STDDEV/MEDIAN/exact-DISTINCT results.
        let global = AggregateNode {
            input: NodeId::from_index_for_internal_use(1),
            spec: AggregateSpec::new(Vec::new(), Vec::new()),
        };
        assert!(!should_partition_whole_input_grouped_aggregate(&context, &global, INPUT_BYTES));

        let grouping_sets = AggregateNode {
            input: NodeId::from_index_for_internal_use(1),
            spec: AggregateSpec::new(vec![Expr::Literal(LiteralValue::Utf8("k".to_string()))], Vec::new())
                .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false]], "grouping_id")),
        };
        assert!(!should_partition_whole_input_grouped_aggregate(&context, &grouping_sets, INPUT_BYTES));
    }
}
