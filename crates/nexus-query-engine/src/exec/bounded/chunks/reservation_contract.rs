use super::*;

pub(in crate::exec::bounded) fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    source_read_reservation_contract(policy, chunk).requested_bytes
}

pub(in crate::exec) fn source_read_batch_reservation_contract(
    policy: MemoryControlPolicy,
    batch: &SourceChunkBatch,
) -> SourceReadReservationContract {
    let mut requested_bytes = 0_u64;
    let mut planning_bytes = Some(0_u64);
    let mut estimated_bytes = Some(0_u64);
    let mut all_exact = true;
    let mut all_exact_or_projected = true;

    for chunk in &batch.chunks {
        let contract = source_read_reservation_contract(policy, chunk);
        requested_bytes = requested_bytes.saturating_add(contract.requested_bytes);
        planning_bytes = add_optional_contract_bytes(planning_bytes, contract.planning_bytes);
        estimated_bytes = add_optional_contract_bytes(estimated_bytes, contract.estimated_bytes);
        all_exact &= contract.quality == SourceReadReservationContractQuality::ExactLowerLevelContract;
        all_exact_or_projected &= matches!(
            contract.quality,
            SourceReadReservationContractQuality::ExactLowerLevelContract
                | SourceReadReservationContractQuality::ProjectedUncompressedEstimate
        );
    }

    SourceReadReservationContract {
        requested_bytes,
        planning_bytes,
        estimated_bytes: estimated_bytes.or(batch.estimated_source_read_bytes),
        quality: if all_exact {
            SourceReadReservationContractQuality::ExactLowerLevelContract
        } else if all_exact_or_projected {
            SourceReadReservationContractQuality::ProjectedUncompressedEstimate
        } else {
            SourceReadReservationContractQuality::ConservativeEstimate
        },
    }
}

pub(in crate::exec::bounded::chunks) fn add_optional_contract_bytes(
    left: Option<u64>,
    right: Option<u64>,
) -> Option<u64> {
    Some(left?.saturating_add(right?))
}

pub(in crate::exec::bounded) fn prefetch_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let contract = source_read_reservation_contract(policy, chunk);
    match (policy, contract.quality) {
        (MemoryControlPolicy::Unbounded, _) => contract.requested_bytes,
        (
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. },
            SourceReadReservationContractQuality::ConservativeEstimate,
        ) => contract.requested_bytes.max(max_source_chunk_bytes),
        (MemoryControlPolicy::Bounded { .. }, _) => contract.requested_bytes,
    }
}

pub(in crate::exec::bounded) fn join_leaf_prefetch_requested_bytes(
    policy: MemoryControlPolicy,
    chunk: &SourceChunk,
) -> Option<u64> {
    let requested_bytes = prefetch_requested_bytes(policy, chunk);
    let MemoryControlPolicy::Bounded { device_budget_bytes, max_source_chunk_bytes, .. } = policy else {
        return Some(requested_bytes);
    };
    if chunk.zero_column_count {
        return Some(requested_bytes);
    }
    let minimum_read_budget = minimum_parquet_source_read_budget_bytes();
    if minimum_read_budget > max_source_chunk_bytes || minimum_read_budget > device_budget_bytes {
        return None;
    }
    Some(requested_bytes.max(minimum_read_budget))
}

pub(in crate::exec::bounded::chunks) fn minimum_parquet_source_read_budget_bytes() -> u64 {
    cudf::DataFrame::minimum_parquet_source_read_budget_bytes() as u64
}

/// Precision of the byte contract used when reserving memory for one bounded
/// source-read chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum SourceReadReservationContractQuality {
    /// A lower layer supplied an exact preflight byte contract, so bounded
    /// source reads can reserve the required bytes before decoding starts.
    ExactLowerLevelContract,
    /// Parquet footer metadata gave projected uncompressed byte counts for the
    /// columns that the source reader must decode.
    ProjectedUncompressedEstimate,
    /// The read is admitted with an initial reservation and then reconciled to
    /// observed device bytes by growing or shrinking after decoding.
    ObservedGrowShrink,
    /// A failed lower-level hard budget reported typed allocation counters and
    /// the retry reservation was corrected from those counters.
    CorrectedLowerLevelBudget,
    /// The reservation is based on file statistics or configured chunk caps;
    /// it may be retried with smaller row-group chunks if execution exceeds it.
    ConservativeEstimate,
}

impl SourceReadReservationContractQuality {
    pub(in crate::exec) const fn as_str(self) -> &'static str {
        match self {
            Self::ExactLowerLevelContract => "exact_lower_level_contract",
            Self::ProjectedUncompressedEstimate => "projected_uncompressed_estimate",
            Self::ObservedGrowShrink => "observed_grow_shrink",
            Self::CorrectedLowerLevelBudget => "corrected_lower_level_budget",
            Self::ConservativeEstimate => "conservative_estimate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct SourceReadBudgetCorrection {
    previous_budget_bytes: u64,
    corrected_budget_bytes: u64,
    peak_aligned_bytes: u64,
    largest_failed_allocation_bytes: u64,
    failed_allocation_count: u64,
    cap_bytes: u64,
}

impl SourceReadBudgetCorrection {
    pub(in crate::exec) fn attach_to_error(self, error: QueryError) -> QueryError {
        error
            .with_fact_if_absent("budget_retry_previous_budget_bytes", self.previous_budget_bytes)
            .with_fact_if_absent("budget_retry_corrected_budget_bytes", self.corrected_budget_bytes)
            .with_fact_if_absent("budget_retry_peak_aligned_bytes", self.peak_aligned_bytes)
            .with_fact_if_absent("budget_retry_largest_failed_allocation_bytes", self.largest_failed_allocation_bytes)
            .with_fact_if_absent("budget_retry_failed_allocation_count", self.failed_allocation_count)
            .with_fact_if_absent(
                "budget_retry_slack_bytes",
                BoundedSourceReservationPolicy::default().source_read_budget_correction_slack_bytes(),
            )
            .with_fact_if_absent("budget_retry_cap_bytes", self.cap_bytes)
    }

    pub(in crate::exec) fn corrected_budget_bytes(self) -> u64 {
        self.corrected_budget_bytes
    }

    pub(in crate::exec) fn previous_budget_bytes(self) -> u64 {
        self.previous_budget_bytes
    }
}

pub(in crate::exec) fn source_read_budget_correction_from_error(
    error: &QueryError,
    reservations: &ReservationTracker,
    reservation: &ActiveReservation,
) -> Option<SourceReadBudgetCorrection> {
    if error.code() != QueryErrorCode::CudfResourceExhausted {
        return None;
    }
    let detail = error.cudf_budget_exhaustion()?;
    if detail.failed_allocation_count() == 0 || detail.largest_failed_allocation_bytes() == 0 {
        return None;
    }
    let previous_budget_bytes = u64::try_from(detail.hard_budget_bytes()).ok()?;
    let peak_aligned_bytes = u64::try_from(detail.peak_aligned_bytes()).ok()?;
    let largest_failed_allocation_bytes = u64::try_from(detail.largest_failed_allocation_bytes()).ok()?;
    let failed_allocation_count = u64::try_from(detail.failed_allocation_count()).ok()?;
    let lower_bound = BoundedSourceReservationPolicy::default().source_read_budget_correction_lower_bound(
        previous_budget_bytes,
        peak_aligned_bytes,
        largest_failed_allocation_bytes,
    );
    let cap_bytes = reservations.available_bytes_including(reservation);
    let corrected_budget_bytes = lower_bound.min(cap_bytes);
    if corrected_budget_bytes <= previous_budget_bytes {
        return None;
    }
    Some(SourceReadBudgetCorrection {
        previous_budget_bytes,
        corrected_budget_bytes,
        peak_aligned_bytes,
        largest_failed_allocation_bytes,
        failed_allocation_count,
        cap_bytes,
    })
}

pub(in crate::exec) fn try_grow_source_read_reservation_for_budget_correction(
    reservations: &mut ReservationTracker,
    reservation: &mut ActiveReservation,
    correction: SourceReadBudgetCorrection,
    mut request: MemoryReservationRequest,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    if correction.corrected_budget_bytes <= reservation.bytes() {
        return Ok(());
    }
    let delta = correction.corrected_budget_bytes.saturating_sub(reservation.bytes());
    request = request.with_requested_bytes(delta).with_source_read_contract(
        SourceReadReservationContractQuality::CorrectedLowerLevelBudget.as_str(),
        Some(correction.previous_budget_bytes),
        Some(correction.corrected_budget_bytes),
    );
    reservations.try_grow(reservation, request, metrics).map_err(|error| correction.attach_to_error(error))
}

pub(in crate::exec::bounded::chunks) fn try_read_local_parquet_chunk_with_corrected_budget_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    error: &QueryError,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    context.check_cancellation("nexus_query_engine.exec.bounded.source_chunk_budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(error, reservations, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, metrics);
    metrics.record_source_chunk_retry();
    crate::exec::retry::backoff_before_retry(context, "nexus_query_engine.exec.bounded.source_chunk_budget_retry")?;
    let request = chunk_reservation_request(
        ReservationKind::SourceRead,
        memory_boundary,
        0,
        reservations.budget(),
        chunk,
        attempt + 1,
        max_attempts,
    );
    try_grow_source_read_reservation_for_budget_correction(reservations, reservation, correction, request, metrics)?;
    metrics.record_source_chunk_attempt();
    read_local_parquet_chunk(context, source, chunk, correction.corrected_budget_bytes(), memory_boundary, metrics)
        .map(Some)
        .map_err(|error| correction.attach_to_error(error))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct SourceReadReservationContract {
    pub(in crate::exec) requested_bytes: u64,
    pub(in crate::exec) planning_bytes: Option<u64>,
    pub(in crate::exec) estimated_bytes: Option<u64>,
    pub(in crate::exec) quality: SourceReadReservationContractQuality,
}

pub(in crate::exec::bounded) fn source_read_reservation_contract(
    policy: MemoryControlPolicy,
    chunk: &SourceChunk,
) -> SourceReadReservationContract {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    if let Some(exact_bytes) = chunk.exact_source_read_preflight_bytes {
        return SourceReadReservationContract {
            requested_bytes: exact_bytes,
            planning_bytes: Some(exact_bytes),
            estimated_bytes: chunk.estimated_source_read_bytes,
            quality: SourceReadReservationContractQuality::ExactLowerLevelContract,
        };
    }
    let has_projected_source_read_estimate = chunk.estimated_source_read_bytes.is_some()
        && (chunk.known_projected_uncompressed_bytes.is_some() || chunk.known_pure_filter_uncompressed_bytes.is_some());
    let planning_bytes = if has_projected_source_read_estimate {
        chunk.estimated_source_read_bytes
    } else {
        chunk
            .estimated_chunk_bytes
            .or(chunk.known_projected_uncompressed_bytes)
            .or(chunk.known_row_group_bytes)
            .or(chunk.known_file_bytes)
    };
    let estimated_bytes = chunk
        .estimated_source_read_bytes
        .or(chunk.known_row_group_bytes)
        .or(chunk.estimated_chunk_bytes)
        .or(chunk.known_file_bytes);
    let conservative_bytes = estimated_bytes.unwrap_or(fallback);
    let requested_bytes = match policy {
        MemoryControlPolicy::Unbounded => conservative_bytes,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => conservative_bytes.min(max_source_chunk_bytes),
    };
    SourceReadReservationContract {
        requested_bytes,
        planning_bytes,
        estimated_bytes,
        quality: if has_projected_source_read_estimate {
            SourceReadReservationContractQuality::ProjectedUncompressedEstimate
        } else {
            SourceReadReservationContractQuality::ConservativeEstimate
        },
    }
}

pub(in crate::exec::bounded) fn chunk_reservation_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = source_read_request(
        kind,
        memory_boundary,
        requested_bytes,
        budget_bytes,
        chunk.source_node_id,
        chunk,
        attempt,
        max_attempts,
    );
    if let Some(operator_kind) = legacy_bounded_chunk_operator_kind(kind, memory_boundary) {
        request = request.with_task_registration(
            legacy_bounded_chunk_task_id(operator_kind, memory_boundary, chunk, attempt),
            u32::try_from(attempt).unwrap_or(u32::MAX),
            operator_kind,
        );
    }
    request
}

pub(in crate::exec) fn source_chunk_batch_reservation_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return chunk_reservation_request(
            kind,
            memory_boundary,
            requested_bytes,
            budget_bytes,
            first_chunk,
            attempt,
            max_attempts,
        );
    }

    source_read_batch_request(
        kind,
        memory_boundary,
        requested_bytes,
        budget_bytes,
        first_chunk.source_node_id,
        batch,
        attempt,
        max_attempts,
    )
}

pub(in crate::exec::bounded::chunks) fn legacy_bounded_chunk_operator_kind(
    kind: ReservationKind,
    memory_boundary: &str,
) -> Option<&'static str> {
    match kind {
        ReservationKind::SourceRead => Some("source_read"),
        ReservationKind::ChunkOperator | ReservationKind::PartialState
            if memory_boundary.contains("filter") || memory_boundary.contains("projection") =>
        {
            Some("unary")
        },
        ReservationKind::ChunkOperator | ReservationKind::PartialState => Some("aggregate"),
        _ => None,
    }
}

pub(in crate::exec::bounded::chunks) fn legacy_bounded_chunk_task_id(
    operator_kind: &str,
    memory_boundary: &str,
    chunk: &SourceChunk,
    attempt: usize,
) -> String {
    format!(
        "legacy_bounded.{operator_kind}.{}.node{}.file{}.chunk{}.attempt{}",
        legacy_bounded_task_id_component(memory_boundary),
        chunk.source_node_id,
        chunk.file_index,
        chunk.chunk_index,
        attempt
    )
}

pub(in crate::exec::bounded::chunks) fn legacy_bounded_task_id_component(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' }).collect()
}

pub(in crate::exec::bounded) fn source_read_reservation_request(
    policy: MemoryControlPolicy,
    kind: ReservationKind,
    memory_boundary: &'static str,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let contract = source_read_reservation_contract(policy, chunk);
    chunk_reservation_request(
        kind,
        memory_boundary,
        contract.requested_bytes,
        budget_bytes,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

pub(in crate::exec) fn source_read_batch_reservation_request(
    policy: MemoryControlPolicy,
    kind: ReservationKind,
    memory_boundary: &'static str,
    budget_bytes: u64,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let contract = source_read_batch_reservation_contract(policy, batch);
    source_chunk_batch_reservation_request(
        kind,
        memory_boundary,
        contract.requested_bytes,
        budget_bytes,
        batch,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

pub(in crate::exec::bounded) fn observed_source_read_reservation_request(
    memory_boundary: &'static str,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    chunk_reservation_request(
        ReservationKind::SourceRead,
        memory_boundary,
        0,
        budget_bytes,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        chunk.estimated_chunk_bytes,
        chunk.estimated_source_read_bytes,
    )
}

pub(in crate::exec) fn observed_source_read_batch_reservation_request(
    memory_boundary: &'static str,
    budget_bytes: u64,
    batch: &SourceChunkBatch,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let first_chunk = batch.chunks.first().expect("source chunk batch is non-empty");
    if batch.is_singleton() {
        return observed_source_read_reservation_request(
            memory_boundary,
            budget_bytes,
            first_chunk,
            attempt,
            max_attempts,
        );
    }
    let contract = source_read_batch_reservation_contract(MemoryControlPolicy::Unbounded, batch);
    source_chunk_batch_reservation_request(
        ReservationKind::SourceRead,
        memory_boundary,
        0,
        budget_bytes,
        batch,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        contract.planning_bytes,
        contract.estimated_bytes,
    )
}
