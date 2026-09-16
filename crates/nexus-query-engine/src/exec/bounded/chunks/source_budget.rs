use super::{
    BoundedExecutionAttempt, GpuDataFrame, LocalParquetSource, MemoryControlPolicy, QueryError, QueryErrorCode, Result,
    SourceChunk, SourceReadBudgetPolicy, cudf_budget_exhaustion, read_local_parquet_chunk, record_chunk_retry_error,
};

pub(in crate::exec::bounded) fn source_read_budget_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    source_read_sizing(policy, chunk).requested_bytes
}

pub(in crate::exec::bounded) fn prefetch_budget_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let sizing = source_read_sizing(policy, chunk);
    match (policy, sizing.quality) {
        (MemoryControlPolicy::Unbounded, _) => sizing.requested_bytes,
        (
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. },
            SourceReadSizingQuality::ConservativeEstimate,
        ) => sizing.requested_bytes.max(max_source_chunk_bytes),
        (MemoryControlPolicy::Bounded { .. }, _) => sizing.requested_bytes,
    }
}

pub(in crate::exec::bounded) fn join_leaf_prefetch_budget_bytes(
    policy: MemoryControlPolicy,
    chunk: &SourceChunk,
) -> Option<u64> {
    let requested_bytes = prefetch_budget_bytes(policy, chunk);
    let MemoryControlPolicy::Bounded { device_budget_bytes, max_source_chunk_bytes, .. } = policy else {
        return Some(requested_bytes);
    };
    if chunk.zero_column_count {
        return Some(requested_bytes);
    }
    let minimum_read_budget = cudf::DataFrame::minimum_parquet_source_read_budget_bytes() as u64;
    if minimum_read_budget > max_source_chunk_bytes || minimum_read_budget > device_budget_bytes {
        return None;
    }
    Some(requested_bytes.max(minimum_read_budget))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceReadSizingQuality {
    ExactLowerLevelContract,
    ProjectedUncompressedEstimate,
    ConservativeEstimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceReadSizing {
    requested_bytes: u64,
    quality: SourceReadSizingQuality,
}

fn source_read_sizing(policy: MemoryControlPolicy, chunk: &SourceChunk) -> SourceReadSizing {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    if let Some(exact_bytes) = chunk.exact_source_read_preflight_bytes {
        return SourceReadSizing {
            requested_bytes: exact_bytes,
            quality: SourceReadSizingQuality::ExactLowerLevelContract,
        };
    }
    let projected = chunk.estimated_source_read_bytes.is_some()
        && (chunk.known_projected_uncompressed_bytes.is_some() || chunk.known_pure_filter_uncompressed_bytes.is_some());
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
    SourceReadSizing {
        requested_bytes,
        quality: if projected {
            SourceReadSizingQuality::ProjectedUncompressedEstimate
        } else {
            SourceReadSizingQuality::ConservativeEstimate
        },
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
                SourceReadBudgetPolicy::default().source_read_budget_correction_slack_bytes(),
            )
            .with_fact_if_absent("budget_retry_cap_bytes", self.cap_bytes)
    }

    pub(in crate::exec) const fn corrected_budget_bytes(self) -> u64 {
        self.corrected_budget_bytes
    }
}

pub(in crate::exec) fn source_read_budget_correction_from_error(
    execution: &BoundedExecutionAttempt<'_>,
    error: &QueryError,
) -> Option<SourceReadBudgetCorrection> {
    if error.code() != QueryErrorCode::CudfResourceExhausted {
        return None;
    }
    let detail = cudf_budget_exhaustion(error)?;
    if detail.failed_allocation_count() == 0 || detail.largest_failed_allocation_bytes() == 0 {
        return None;
    }
    let previous_budget_bytes = u64::try_from(detail.hard_budget_bytes()).ok()?;
    let peak_aligned_bytes = u64::try_from(detail.peak_aligned_bytes()).ok()?;
    let largest_failed_allocation_bytes = u64::try_from(detail.largest_failed_allocation_bytes()).ok()?;
    let failed_allocation_count = u64::try_from(detail.failed_allocation_count()).ok()?;
    let lower_bound = SourceReadBudgetPolicy::default().source_read_budget_correction_lower_bound(
        previous_budget_bytes,
        peak_aligned_bytes,
        largest_failed_allocation_bytes,
    );
    let cap_bytes = execution.device_budget_bytes();
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

pub(in crate::exec::bounded::chunks) fn try_read_local_parquet_chunk_with_corrected_budget_after_error(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    error: &QueryError,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.source_chunk_budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(execution, error) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, execution.metrics_mut());
    execution.metrics_mut().record_source_chunk_retry();
    crate::exec::retry::backoff_before_retry(
        execution.context(),
        "nexus_query_engine.exec.bounded.source_chunk_budget_retry",
    )?;
    execution.metrics_mut().record_source_chunk_attempt();
    read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        correction.corrected_budget_bytes(),
        memory_boundary,
        execution.metrics_mut(),
    )
    .map(Some)
    .map_err(|error| correction.attach_to_error(error))
}
