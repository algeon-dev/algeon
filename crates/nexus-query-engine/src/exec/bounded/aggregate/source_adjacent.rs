use super::{partitioned::partitioned_aggregate_partition_count, *};

struct SourceAdjacentPartitionInput {
    chunks: Vec<SourceChunk>,
    partial: CommittedPartial,
    source_row_group_count: u64,
}

#[derive(Clone, Debug)]
enum SourceAdjacentAggregatePartitioner {
    RowBalanced,
    GroupKey { key_columns: Vec<String> },
}

impl SourceAdjacentAggregatePartitioner {
    fn label(&self) -> &'static str {
        match self {
            Self::RowBalanced => "row_balanced",
            Self::GroupKey { .. } => "group_key",
        }
    }

    fn key_count(&self) -> Option<u64> {
        match self {
            Self::RowBalanced => None,
            Self::GroupKey { key_columns } => Some(key_columns.len() as u64),
        }
    }

    fn retain_empty_partition(&self, partition: usize) -> bool {
        match self {
            Self::RowBalanced => true,
            Self::GroupKey { .. } => partition == 0,
        }
    }
}

pub(in crate::exec::bounded) fn execute_bounded_partitioned_source_adjacent_aggregate_from_raw_chunks(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let estimated_input_bytes = chunks
        .iter()
        .filter_map(|chunk| chunk.estimated_chunk_bytes.or(chunk.estimated_source_read_bytes))
        .sum::<u64>()
        .max(1);
    let partition_count = partitioned_aggregate_partition_count(context, estimated_input_bytes)
        .max(source_adjacent_partitioned_aggregate_min_partition_count(chunks.len()));
    let partitioner = source_adjacent_aggregate_partitioner(context, bounded, estimated_input_bytes, chunks.len());
    let mut partition_inputs =
        (0..partition_count).map(|_| Vec::new()).collect::<Vec<Vec<SourceAdjacentPartitionInput>>>();
    let mut aggregate_outputs = Vec::with_capacity(partition_count);

    let source_batches = source_adjacent_raw_source_batches(context, bounded, chunks);
    let mut committed_source_batches = Vec::with_capacity(source_batches.len());
    for batch in source_batches {
        let raw_inputs = match execute_source_adjacent_raw_source_batch(context, bounded, batch, reservations, metrics)
        {
            Ok(raw_inputs) => raw_inputs,
            Err(error) => {
                release_nested_partition_inputs_reverse(reservations, partition_inputs, metrics);
                release_partials_reverse(reservations, aggregate_outputs, metrics);
                return Err(error);
            },
        };
        for (batch_chunks, raw) in raw_inputs {
            let chunk_index = batch_chunks.first().expect("raw source batch has at least one chunk").chunk_index;
            let partitions = match partition_source_adjacent_raw_chunk(
                context,
                bounded.aggregate_id,
                chunk_index,
                raw.partial,
                partition_count,
                &partitioner,
                reservations,
                metrics,
            ) {
                Ok(partitions) => partitions,
                Err(error) => {
                    release_nested_partition_inputs_reverse(reservations, partition_inputs, metrics);
                    release_partials_reverse(reservations, aggregate_outputs, metrics);
                    return Err(error);
                },
            };
            for (partition_index, partition) in partitions.into_iter().enumerate() {
                if let Some(partition) = partition {
                    partition_inputs[partition_index].push(SourceAdjacentPartitionInput {
                        chunks: batch_chunks.clone(),
                        partial: partition,
                        source_row_group_count: raw.source_row_group_count,
                    });
                }
            }
            committed_source_batches.push((batch_chunks.len() as u64, raw.source_row_group_count));
        }
        for (partition_index, inputs) in partition_inputs.iter_mut().enumerate() {
            if source_adjacent_partition_batch_ready(context, inputs) {
                match execute_source_adjacent_partition_aggregate_batch(
                    context,
                    bounded,
                    partition_index,
                    partition_count,
                    &partitioner,
                    inputs,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => aggregate_outputs.push(output),
                    Err(error) => {
                        release_nested_partition_inputs_reverse(reservations, partition_inputs, metrics);
                        release_partials_reverse(reservations, aggregate_outputs, metrics);
                        return Err(error);
                    },
                }
            }
        }
    }

    for (partition_index, mut inputs) in partition_inputs.into_iter().enumerate() {
        if inputs.is_empty() {
            continue;
        }
        match execute_source_adjacent_partition_aggregate_batch(
            context,
            bounded,
            partition_index,
            partition_count,
            &partitioner,
            &mut inputs,
            reservations,
            metrics,
        ) {
            Ok(output) => aggregate_outputs.push(output),
            Err(error) => {
                release_partition_inputs_reverse(reservations, inputs, metrics);
                release_partials_reverse(reservations, aggregate_outputs, metrics);
                return Err(error);
            },
        }
    }
    for (chunk_count, row_group_count) in committed_source_batches {
        metrics.record_source_chunk_commit_count(chunk_count, row_group_count);
    }

    execute_final_aggregate_merge(context, bounded, aggregate_outputs, reservations, metrics)
}

fn execute_source_adjacent_raw_source_batch(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    batch: SourceChunkBatch,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<(Vec<SourceChunk>, RawSourceInput)>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.source_adjacent_raw_source_batch_retry")?;
    if batch.is_singleton() {
        let chunk = batch.chunks.into_iter().next().expect("singleton source batch has one child");
        let raw = execute_bounded_raw_source_chunk(context, bounded, &chunk, reservations, metrics)?;
        return Ok(vec![(vec![chunk], raw)]);
    }
    match execute_bounded_raw_source_chunk_batch(context, bounded, &batch, reservations, metrics) {
        Ok(raw) => Ok(vec![(batch.chunks, raw)]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, metrics);
            metrics.record_source_chunk_retry();
            let Some((left, right)) = batch.split_for_retry() else {
                return Err(error);
            };
            let mut raw_inputs =
                execute_source_adjacent_raw_source_batch(context, bounded, left, reservations, metrics)?;
            match execute_source_adjacent_raw_source_batch(context, bounded, right, reservations, metrics) {
                Ok(mut right_inputs) => {
                    raw_inputs.append(&mut right_inputs);
                    Ok(raw_inputs)
                },
                Err(error) => {
                    release_raw_source_inputs_reverse(reservations, raw_inputs, metrics);
                    Err(error)
                },
            }
        },
        Err(error) => Err(error),
    }
}

fn release_raw_source_inputs_reverse(
    reservations: &mut ReservationTracker,
    mut inputs: Vec<(Vec<SourceChunk>, RawSourceInput)>,
    metrics: &mut QueryMetrics,
) {
    while let Some((_chunks, raw)) = inputs.pop() {
        drop(raw.partial.frame);
        reservations.release(raw.partial.reservation, metrics);
    }
}

fn source_adjacent_raw_source_batches(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
) -> Vec<SourceChunkBatch> {
    if context.local_parquet_source_cache().is_some()
        || !bounded.source.partition_columns.is_empty()
        || bounded.source.files.iter().any(LocalParquetFile::has_position_delete_files)
        || bounded.source.scan_limit.is_some()
        || bounded
            .source
            .reader_predicate
            .as_ref()
            .is_some_and(|predicate| !bounded.source.reader_predicate_is_supported(predicate))
    {
        return singleton_source_chunk_batches(chunks);
    }
    let max_batch_source_read_bytes = BoundedSourceBatchingPolicy::default()
        .source_adjacent_raw_source_batch_target_bytes(context.memory_control_policy(), false, false);
    plan_source_chunk_batches(chunks, max_batch_source_read_bytes)
}

fn singleton_source_chunk_batches(chunks: &[SourceChunk]) -> Vec<SourceChunkBatch> {
    chunks
        .iter()
        .enumerate()
        .map(|(batch_index, chunk)| SourceChunkBatch {
            batch_index,
            chunks: vec![chunk.clone()],
            estimated_source_read_bytes: chunk
                .estimated_source_read_bytes
                .or(chunk.known_row_group_bytes)
                .or(chunk.estimated_chunk_bytes)
                .or(chunk.known_file_bytes),
            source_batching_reason: "source_gate:source_adjacent_batching_disabled".to_string(),
        })
        .collect()
}

fn source_adjacent_partitioned_aggregate_input_evidence(
    chunks: &[SourceChunk],
    source_row_group_count: u64,
    partitioner: &SourceAdjacentAggregatePartitioner,
    partition_count: usize,
    non_empty_partition_count: usize,
    max_partition_fan_in: usize,
) -> PartialAggregateInputEvidence {
    PartialAggregateInputEvidence {
        source_chunk_fingerprint: partial_aggregate_source_chunk_fingerprint(chunks),
        source_adjacent_partial: true,
        source_chunk_row_group_count: source_row_group_count,
        source_chunk_reads_all_row_groups: chunks.iter().all(|chunk| chunk.reads_all_row_groups),
        source_chunk_known_row_count: sum_optional_u64(chunks.iter().map(|chunk| chunk.known_row_count)),
        source_chunk_estimated_source_read_bytes: sum_optional_u64(
            chunks.iter().map(|chunk| chunk.estimated_source_read_bytes),
        ),
        source_chunk_size_estimate_based: chunks.iter().any(|chunk| chunk.size_estimate_based),
        source_adjacent_partition_strategy: partitioner.label(),
        source_adjacent_partition_count: Some(partition_count as u64),
        source_adjacent_non_empty_partition_count: Some(non_empty_partition_count as u64),
        source_adjacent_partition_key_count: partitioner.key_count(),
        source_adjacent_max_partition_fan_in: Some(max_partition_fan_in as u64),
    }
}

fn source_adjacent_partition_batch_ready(context: &ExecutionContext, inputs: &[SourceAdjacentPartitionInput]) -> bool {
    BoundedSourceBatchingPolicy::default().source_adjacent_partition_batch_ready(
        inputs.len(),
        inputs.iter().map(|input| input.source_row_group_count).sum(),
        source_adjacent_partition_batch_reserved_bytes(inputs),
        bounded_device_budget_bytes(context),
    )
}

fn source_adjacent_partition_batch_reserved_bytes(inputs: &[SourceAdjacentPartitionInput]) -> u64 {
    if inputs.len() <= 1 {
        return 0;
    }
    inputs
        .iter()
        .map(|input| input.partial.reservation.bytes())
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .unwrap_or(u64::MAX)
}

fn bounded_device_budget_bytes(context: &ExecutionContext) -> Option<u64> {
    match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { device_budget_bytes, .. } => Some(device_budget_bytes),
        _ => None,
    }
}

#[cfg(test)]
fn source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes: Option<u64>) -> u64 {
    BoundedSourceBatchingPolicy::default().source_adjacent_partition_batch_byte_limit_for_budget(device_budget_bytes)
}

fn execute_source_adjacent_partition_aggregate_batch(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
    inputs: &mut Vec<SourceAdjacentPartitionInput>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let batch_inputs = std::mem::take(inputs);
    let non_empty_partition_count = usize::from(batch_inputs.iter().any(|input| input.partial.frame.num_rows() > 0));
    let max_partition_fan_in = batch_inputs.len();
    let source_row_group_count = batch_inputs.iter().map(|input| input.source_row_group_count).sum::<u64>();
    let batch_chunks = batch_inputs.iter().flat_map(|input| input.chunks.iter().cloned()).collect::<Vec<_>>();
    let input_evidence = source_adjacent_partitioned_aggregate_input_evidence(
        &batch_chunks,
        source_row_group_count,
        partitioner,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
    );
    let partials = batch_inputs.into_iter().map(|input| input.partial).collect::<Vec<_>>();
    let partition_input =
        concatenate_partition_inputs(context, bounded.aggregate_id, partition_index, partials, reservations, metrics)?;
    execute_source_adjacent_partition_aggregate(
        context,
        bounded,
        partition_index,
        partition_input,
        input_evidence,
        reservations,
        metrics,
    )
}

fn source_adjacent_aggregate_partitioner(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    _estimated_input_bytes: u64,
    chunk_count: usize,
) -> SourceAdjacentAggregatePartitioner {
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&bounded.aggregate.0.spec) else {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    };
    if lowered_keys.key_columns.is_empty() {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    let mut has_string_key = false;
    for key_column in &lowered_keys.key_columns {
        let Ok(field) = bounded.aggregate_input_schema.field_with_name(key_column) else {
            return SourceAdjacentAggregatePartitioner::RowBalanced;
        };
        has_string_key |=
            matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View);
    }
    if !has_string_key {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    if matches!(context.memory_control_policy(), MemoryControlPolicy::Bounded { .. })
        && !BoundedPartitionRowPolicy::default().source_adjacent_group_key_partitioning_fits_chunk_count(chunk_count)
    {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    SourceAdjacentAggregatePartitioner::GroupKey { key_columns: lowered_keys.key_columns }
}

fn sum_optional_u64(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    values.into_iter().try_fold(0u64, |total, value| value.map(|value| total.saturating_add(value)))
}

fn partition_source_adjacent_raw_chunk(
    context: &ExecutionContext,
    node_id: NodeId,
    chunk_index: usize,
    raw: CommittedPartial,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<Option<CommittedPartial>>> {
    let input_reserved_bytes = raw.reservation.bytes();
    let hash_task_id = source_adjacent_aggregate_task_id(
        node_id,
        "source_adjacent_aggregate_hash_partition",
        format!("chunk{chunk_index}"),
    );
    let mut partition_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_aggregate_hash_partition")
            .with_node_id(node_id)
            .with_stage("source_adjacent_aggregate_hash_partition")
            .with_compiled_task_id(hash_task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(raw.frame);
            reservations.release(raw.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&partition_reservation),
        "source_adjacent_aggregate_hash_partition",
        "nexus_query_engine.bounded.aggregate.source_adjacent_partitioned_hash.budget_token",
        "failed to create source-adjacent partitioned aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            drop(raw.frame);
            reservations.release(raw.reservation, metrics);
            return Err(error);
        },
    };

    let raw_frame = raw.frame.into_frame();
    let original_columns = matches!(partitioner, SourceAdjacentAggregatePartitioner::RowBalanced)
        .then(|| raw_frame.column_names().into_iter().map(str::to_string).collect::<Vec<_>>());
    let partitioned = match partitioner {
        SourceAdjacentAggregatePartitioner::RowBalanced => {
            let partition_column = unique_source_adjacent_partition_column_name(&raw_frame);
            let row_count = raw_frame.num_rows();
            let with_partition_column = raw_frame
                .with_execution(&token.execution_overrides())
                .with_position_column(&partition_column)
                .map_err(|error| {
                    attach_budget_token_failure_facts(
                        QueryError::from_cudf(
                            "nexus_query_engine.bounded.aggregate.source_adjacent_partition_column",
                            "failed to add row-balanced source-adjacent aggregate partition column",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("partition_count", partition_count)
                        .with_fact("input_rows", row_count),
                        &token,
                    )
                });
            let with_partition_column = match with_partition_column {
                Ok(frame) => frame,
                Err(error) => {
                    drop(raw_frame);
                    reservations.release(raw.reservation, metrics);
                    reservations.release(partition_reservation, metrics);
                    return Err(error);
                },
            };
            let partitioned = with_partition_column
                .with_execution(&token.execution_overrides())
                .hash_partition_by([partition_column.as_str()], partition_count)
                .map_err(|error| {
                    attach_budget_token_failure_facts(
                        QueryError::from_cudf(
                            "nexus_query_engine.bounded.aggregate.source_adjacent_hash_partition",
                            "failed to hash-partition source-adjacent aggregate input",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("partition_key", partition_column.as_str())
                        .with_fact("partition_count", partition_count),
                        &token,
                    )
                });
            drop(with_partition_column);
            partitioned
        },
        SourceAdjacentAggregatePartitioner::GroupKey { key_columns } => raw_frame
            .with_execution(&token.execution_overrides())
            .hash_partition_by(key_columns.iter().map(String::as_str), partition_count)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "nexus_query_engine.bounded.aggregate.source_adjacent_group_key_hash_partition",
                        "failed to hash-partition source-adjacent aggregate input by group keys",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("partition_keys", key_columns.join(","))
                    .with_fact("partition_count", partition_count),
                    &token,
                )
            }),
    };
    drop(raw_frame);
    reservations.release(raw.reservation, metrics);

    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_gpu, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("source_adjacent_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("source_adjacent_aggregate_hash_partition")
        .with_compiled_task_id(hash_task_id.clone());
        if let Err(error) = reservations.try_grow(&mut partition_reservation, grow_request, metrics) {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut partition_reservation, partitioned_bytes, metrics);
    }

    let mut partitions = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        match slice_source_adjacent_aggregate_partition(
            context,
            node_id,
            &partitioned_frame,
            &offsets,
            original_columns.as_deref(),
            chunk_index,
            partition,
            partitioner.retain_empty_partition(partition),
            reservations,
            metrics,
        ) {
            Ok(frame) => partitions.push(frame),
            Err(error) => {
                release_optional_partials_reverse(reservations, partitions, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        }
    }
    drop(partitioned_frame);
    reservations.release(partition_reservation, metrics);
    Ok(partitions)
}

fn slice_source_adjacent_aggregate_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    output_columns: Option<&[String]>,
    chunk_index: usize,
    partition: usize,
    retain_empty: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<CommittedPartial>> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "source-adjacent aggregate partition is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "source-adjacent aggregate partition is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    if len == 0 && !retain_empty {
        return Ok(None);
    }
    let overrides = execution_overrides(context);
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.aggregate.source_adjacent_hash_partition_slice",
            "failed to slice source-adjacent hash-partitioned aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let sliced = if let Some(output_columns) = output_columns {
        let output_refs = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
        sliced.select(output_refs).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.aggregate.source_adjacent_hash_partition_select",
                "failed to remove source-adjacent hash-partition helper column",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("partition", partition)
            .with_fact("partition_rows", len)
        })?
    } else {
        sliced
    };
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("source_adjacent_aggregate_partition")
            .with_compiled_task_id(source_adjacent_aggregate_task_id(
                node_id,
                "source_adjacent_aggregate_partition",
                format!("chunk{chunk_index}.partition{partition}"),
            )),
        metrics,
    )?;
    Ok(Some(CommittedPartial { frame: gpu, reservation }))
}

fn concatenate_partition_inputs(
    context: &ExecutionContext,
    node_id: NodeId,
    partition_index: usize,
    mut inputs: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if inputs.len() == 1 {
        return Ok(inputs.pop().expect("single input checked"));
    }
    concatenate_committed_frames(
        context,
        node_id,
        partition_index,
        inputs,
        ReservationKind::OperatorOutput,
        "source_adjacent_aggregate_partition_concat",
        "source_adjacent_aggregate_partition_concat_output",
        "nexus_query_engine.bounded.aggregate.source_adjacent_partition_concat.budget_token",
        "failed to create source-adjacent aggregate partition concat budget token",
        "nexus_query_engine.bounded.aggregate.source_adjacent_partition_concat",
        "failed to concatenate source-adjacent aggregate partition inputs",
        reservations,
        metrics,
    )
}

fn execute_source_adjacent_partition_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    input: CommittedPartial,
    input_evidence: PartialAggregateInputEvidence,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let input_reserved_bytes = input.reservation.bytes();
    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let task_id = source_adjacent_aggregate_task_id(
        bounded.aggregate_id,
        "source_adjacent_partitioned_partial_aggregate",
        format!("partition{partition_index}"),
    );
    let mut work_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::PartialState, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_partitioned_partial_aggregate")
            .with_node_id(bounded.aggregate_id)
            .with_stage("source_adjacent_partitioned_partial_aggregate")
            .with_attempt(1, 1)
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input_frame);
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        "source_adjacent_partitioned_partial_aggregate",
        "nexus_query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            drop(input_frame);
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", bounded.aggregate_id));
        },
    };
    let chunk_operator_timer = Instant::now();
    metrics.set_current_task(task_id.clone());
    let result = execute_partial_aggregate_with_budget(
        context,
        Some(bounded.aggregate_id),
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        input_frame,
        &token,
        Some(input_evidence),
        metrics,
    );
    metrics.clear_current_task();
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error
                .with_fact_if_absent("node_id", bounded.aggregate_id)
                .with_fact_if_absent("partition", partition_index));
        },
    };
    reservations.release(input_reservation, metrics);
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error
                .with_fact_if_absent("node_id", bounded.aggregate_id)
                .with_fact_if_absent("partition", partition_index));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, output_bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_partitioned_partial_aggregate_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("source_adjacent_partitioned_partial_aggregate")
            .with_attempt(1, 1)
            .with_compiled_task_id(task_id);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        drop(frame);
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    let output = CommittedPartial { frame, reservation: output_reservation };
    metrics.record_partial_aggregate(input_rows, output.frame.num_rows() as u64, output.reservation.bytes());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn concatenate_committed_frames(
    context: &ExecutionContext,
    node_id: NodeId,
    partition_index: usize,
    mut inputs: Vec<CommittedPartial>,
    reservation_kind: ReservationKind,
    stage: &'static str,
    output_boundary: &'static str,
    token_operation: &'static str,
    token_message: &'static str,
    concat_operation: &'static str,
    concat_message: &'static str,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            concat_operation,
            "source-adjacent partitioned aggregate concat requires at least one input frame",
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition_index));
    }
    let requested_bytes = inputs.iter().map(|input| input.reservation.bytes()).sum::<u64>();
    let task_id = source_adjacent_aggregate_task_id(node_id, stage, format!("partition{partition_index}"));
    let mut concat_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(reservation_kind, requested_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&concat_reservation),
        stage,
        token_operation,
        token_message,
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };

    let first = inputs.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let other_frames = inputs.iter().map(|input| input.frame.frame().clone()).collect::<Vec<_>>();
        let other_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(concat_operation, concat_message, error)
                    .with_fact("node_id", node_id)
                    .with_fact("partition", partition_index)
                    .with_fact("input_count", inputs.len() + 1),
                &token,
            )
        })
    };
    drop(first_frame);
    reservations.release(first_reservation, metrics);
    release_partials_reverse(reservations, inputs, metrics);

    let concatenated = match concatenated {
        Ok(frame) => GpuDataFrame::new(frame),
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&concatenated, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            return Err(error);
        },
    };
    let output_request = MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_compiled_task_id(task_id.clone());
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut concat_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        drop(concatenated);
        reservations.release(concat_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(concat_reservation, output_request, metrics);
    Ok(CommittedPartial { frame: concatenated, reservation: output_reservation })
}

fn release_nested_partition_inputs_reverse(
    reservations: &mut ReservationTracker,
    mut partitions: Vec<Vec<SourceAdjacentPartitionInput>>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partition) = partitions.pop() {
        release_partition_inputs_reverse(reservations, partition, metrics);
    }
}

fn release_partition_inputs_reverse(
    reservations: &mut ReservationTracker,
    mut inputs: Vec<SourceAdjacentPartitionInput>,
    metrics: &mut QueryMetrics,
) {
    while let Some(input) = inputs.pop() {
        drop(input.partial.frame);
        reservations.release(input.partial.reservation, metrics);
    }
}

fn release_optional_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partials: Vec<Option<CommittedPartial>>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partial) = partials.pop() {
        if let Some(partial) = partial {
            drop(partial.frame);
            reservations.release(partial.reservation, metrics);
        }
    }
}

fn unique_source_adjacent_partition_column_name(frame: &cudf::DataFrame) -> String {
    let base = "__cudf_source_adjacent_partition_row_id";
    crate::naming::unique_alias_with_suffix(base, |candidate| frame.contains(candidate))
}

fn source_adjacent_aggregate_task_id(node_id: NodeId, stage: &'static str, suffix: impl std::fmt::Display) -> String {
    format!("legacy_bounded.aggregate.node{node_id}.{stage}.{suffix}")
}

fn source_adjacent_partitioned_aggregate_min_partition_count(chunk_count: usize) -> usize {
    BoundedPartitionRowPolicy::default().source_adjacent_partitioned_aggregate_min_partition_count(chunk_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_adjacent_partition_batch_byte_limit_tracks_device_budget_with_bounds() {
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(Some(16 * 1024 * 1024 * 1024)),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(Some(4 * 1024 * 1024 * 1024)),
            512 * 1024 * 1024
        );
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(Some(256 * 1024 * 1024)),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MIN_BATCH_BYTES
        );
        assert_eq!(
            source_adjacent_partition_batch_byte_limit_for_budget(None),
            BoundedSourceBatchingPolicy::DEFAULT_SOURCE_ADJACENT_PARTITIONED_AGGREGATE_MAX_BATCH_BYTES
        );
    }
}
