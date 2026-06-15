use std::time::Instant;

use super::{
    object_store_safety::{SourceFetchLocation, object_store_bucket_fact, source_fetch_location},
    ranged_parquet::RangedParquetFetchDiagnostics,
    *,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::S3ObjectStoreKey,
};

pub(in crate::exec::source) struct ScheduledHostParquetFetch {
    pub(in crate::exec::source::host_buffer) receiver: std::sync::mpsc::Receiver<Result<HostParquetBuffer>>,
}

pub(in crate::exec::source::host_buffer) struct HostParquetBuffer {
    pub(in crate::exec::source::host_buffer) bytes: bytes::Bytes,
    pub(in crate::exec::source::host_buffer) permit: Option<crate::runtime::HostFetchPermit>,
    pub(in crate::exec::source::host_buffer) actual_bytes: u64,
    pub(in crate::exec::source::host_buffer) ranged_fetch: Option<RangedParquetFetchDiagnostics>,
    pub(in crate::exec::source::host_buffer) host_budget_wait_elapsed: std::time::Duration,
    pub(in crate::exec::source::host_buffer) concurrency_wait_elapsed: std::time::Duration,
    pub(in crate::exec::source::host_buffer) fetch_elapsed: std::time::Duration,
}

impl ScheduledHostParquetFetch {
    pub(super) fn finish(self) -> Result<HostParquetBuffer> {
        self.receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.source_fetch.receive",
                "object-store source fetch worker stopped before returning a host buffer",
            )
            .with_source(error)
        })?
    }
}

impl HostParquetBuffer {
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn runtime_metrics(&self) -> HostBufferFetchRuntimeMetrics {
        if let Some(ranged_fetch) = self.ranged_fetch.as_ref() {
            HostBufferFetchRuntimeMetrics {
                fetch_mode: "ranged_parquet".to_string(),
                payload_bytes: self.actual_bytes,
                requested_range_count: ranged_fetch.requested_range_count as u64,
                requested_range_bytes: ranged_fetch.requested_range_bytes,
                original_file_bytes: ranged_fetch.original_file_bytes,
                mini_parquet_footer_bytes: ranged_fetch.footer_bytes,
            }
        } else {
            HostBufferFetchRuntimeMetrics {
                fetch_mode: "full_object".to_string(),
                payload_bytes: self.actual_bytes,
                requested_range_count: 0,
                requested_range_bytes: 0,
                original_file_bytes: self.actual_bytes,
                mini_parquet_footer_bytes: 0,
            }
        }
    }

    pub(super) fn release_after_decode(&mut self) {
        self.permit.take();
    }
}

pub(super) fn start_host_parquet_buffer_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
) -> Result<ScheduledHostParquetFetch> {
    start_host_parquet_buffer_fetch_with_budget_mode(
        context,
        location,
        resolved_source,
        estimated_fetch_bytes,
        HostFetchBudgetAcquireMode::Wait,
        HostFetchPayloadAccounting::GrowToActual,
    )?
    .ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.source_fetch.host_buffer",
            "blocking host-buffer fetch budget acquisition unexpectedly returned no permit",
        )
    })
}

pub(super) fn start_host_parquet_buffer_pipeline_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: u64,
) -> Result<ScheduledHostParquetFetch> {
    start_host_parquet_buffer_fetch_with_budget_mode(
        context,
        location,
        resolved_source,
        Some(estimated_fetch_bytes),
        HostFetchBudgetAcquireMode::Wait,
        HostFetchPayloadAccounting::ReservedExact,
    )?
    .ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.source_fetch.host_buffer",
            "blocking pipeline host-buffer fetch budget acquisition unexpectedly returned no permit",
        )
    })
}

pub(super) fn try_start_host_parquet_buffer_pipeline_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: u64,
) -> Result<Option<ScheduledHostParquetFetch>> {
    start_host_parquet_buffer_fetch_with_budget_mode(
        context,
        location,
        resolved_source,
        Some(estimated_fetch_bytes),
        HostFetchBudgetAcquireMode::Try,
        HostFetchPayloadAccounting::ReservedExact,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostFetchBudgetAcquireMode {
    Wait,
    Try,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostFetchPayloadAccounting {
    GrowToActual,
    ReservedExact,
}

fn start_host_parquet_buffer_fetch_with_budget_mode(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
    budget_mode: HostFetchBudgetAcquireMode,
    payload_accounting: HostFetchPayloadAccounting,
) -> Result<Option<ScheduledHostParquetFetch>> {
    context.check_cancellation("nexus_query_engine.source_fetch.host_buffer")?;
    let fetch_location = source_fetch_location(location, resolved_source)?;
    let host_budget = context.host_fetch_budget();
    let host_budget_bytes = host_budget.budget_bytes();
    let host_fetch_acquire_timeout = context.host_fetch_acquire_timeout();
    let reserve_bytes = host_fetch_reserve_bytes(
        host_budget_bytes,
        context.source_scheduler_policy().max_source_reads_in_flight(),
        estimated_fetch_bytes,
    );
    let (mut permit, operation_permit, mut host_budget_wait_elapsed, concurrency_wait_elapsed) = match budget_mode {
        HostFetchBudgetAcquireMode::Wait => {
            let host_budget_wait_timer = Instant::now();
            let permit = host_budget
                .acquire_with_timeout(
                    context.cancellation(),
                    reserve_bytes,
                    host_fetch_acquire_timeout,
                    "nexus_query_engine.source_fetch.host_buffer_budget",
                )
                .map_err(|error| attach_fetch_budget_facts(error, &fetch_location, reserve_bytes))?
                .ok_or_else(|| host_fetch_budget_timeout_error(&fetch_location, reserve_bytes))?;
            let host_budget_wait_elapsed = host_budget_wait_timer.elapsed();
            let concurrency_wait_timer = Instant::now();
            let operation_permit = context
                .acquire_source_fetch_operation_with_timeout(
                    host_fetch_acquire_timeout,
                    "nexus_query_engine.source_fetch.host_buffer_concurrency",
                )
                .map_err(|error| attach_fetch_budget_facts(error, &fetch_location, reserve_bytes))?
                .ok_or_else(|| host_fetch_concurrency_timeout_error(&fetch_location, reserve_bytes))?;
            (permit, operation_permit, host_budget_wait_elapsed, concurrency_wait_timer.elapsed())
        },
        HostFetchBudgetAcquireMode::Try => {
            let concurrency_wait_timer = Instant::now();
            let Some(operation_permit) = context
                .try_acquire_source_fetch_operation()
                .map_err(|error| attach_fetch_budget_facts(error, &fetch_location, reserve_bytes))?
            else {
                return Ok(None);
            };
            let concurrency_wait_elapsed = concurrency_wait_timer.elapsed();
            let host_budget_wait_timer = Instant::now();
            let Some(permit) = host_budget
                .try_acquire(reserve_bytes)
                .map_err(|error| attach_fetch_budget_facts(error, &fetch_location, reserve_bytes))?
            else {
                return Ok(None);
            };
            (permit, operation_permit, host_budget_wait_timer.elapsed(), concurrency_wait_elapsed)
        },
    };
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
        storage_profile_id: fetch_location.storage_profile_id.clone(),
    })?;
    let (sender, receiver) = std::sync::mpsc::channel();
    let context_host_budget_bytes = host_budget_bytes;
    context.spawn_object_store_fetch(store, fetch_location.key, move |fetch_result| {
        let fetch_elapsed = fetch_result.elapsed;
        let result = fetch_result
            .bytes
            .map_err(|error| {
                error
                    .with_fact_if_absent("source_fetch_substrate", "object_store")
                    .with_fact_if_absent("source_fetch_scheme", "s3")
                    .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
                    .with_fact_if_absent("region", fetch_location.region.as_str())
            })
            .and_then(|bytes| {
                let actual_bytes = u64::try_from(bytes.len()).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.source_fetch.host_buffer_size",
                        "object-store source fetch payload length exceeds u64 range",
                    )
                })?;
                host_budget_wait_elapsed += account_host_fetch_payload_bytes(
                    &mut permit,
                    payload_accounting,
                    actual_bytes,
                    context_host_budget_bytes,
                )
                .map_err(|error| {
                    error
                        .with_fact_if_absent("source_fetch_substrate", "object_store")
                        .with_fact_if_absent("source_fetch_scheme", "s3")
                        .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
                        .with_fact_if_absent("actual_bytes", actual_bytes)
                })?;
                Ok(HostParquetBuffer {
                    bytes,
                    permit: Some(permit),
                    actual_bytes,
                    ranged_fetch: None,
                    host_budget_wait_elapsed,
                    concurrency_wait_elapsed,
                    fetch_elapsed,
                })
            });
        let _ = sender.send(result);
        drop(operation_permit);
    });
    Ok(Some(ScheduledHostParquetFetch { receiver }))
}

fn host_fetch_reserve_bytes(
    budget_bytes: u64,
    _max_source_reads_in_flight: usize,
    estimated_fetch_bytes: Option<u64>,
) -> u64 {
    estimated_fetch_bytes.unwrap_or(budget_bytes).max(1)
}

fn account_host_fetch_payload_bytes(
    permit: &mut crate::runtime::HostFetchPermit,
    payload_accounting: HostFetchPayloadAccounting,
    actual_bytes: u64,
    budget_bytes: u64,
) -> Result<Duration> {
    let charge_bytes = actual_bytes.max(1);
    match payload_accounting {
        HostFetchPayloadAccounting::GrowToActual => {
            let grow_wait_timer = Instant::now();
            permit.grow_to(charge_bytes)?;
            Ok(grow_wait_timer.elapsed())
        },
        HostFetchPayloadAccounting::ReservedExact => {
            if charge_bytes > permit.bytes() {
                return Err(QueryError::new(
                    QueryErrorCode::MemoryReservationFailed,
                    "nexus_query_engine.source_fetch.host_buffer_pipeline_size",
                    "pipelined object-store source fetch exceeded reserved host-buffer bytes",
                )
                .with_fact("reserved_bytes", permit.bytes())
                .with_fact("actual_bytes", actual_bytes)
                .with_fact("budget_bytes", budget_bytes));
            }
            Ok(Duration::ZERO)
        },
    }
}

fn attach_fetch_budget_facts(
    error: QueryError,
    fetch_location: &SourceFetchLocation,
    reserve_bytes: u64,
) -> QueryError {
    error
        .with_fact_if_absent("source_fetch_substrate", "object_store")
        .with_fact_if_absent("source_fetch_scheme", "s3")
        .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
        .with_fact_if_absent("reserved_bytes", reserve_bytes)
}

fn host_fetch_budget_timeout_error(fetch_location: &SourceFetchLocation, reserve_bytes: u64) -> QueryError {
    attach_fetch_budget_facts(
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.source_fetch.host_buffer_budget",
            "timed out waiting for host fetch budget capacity",
        ),
        fetch_location,
        reserve_bytes,
    )
}

fn host_fetch_concurrency_timeout_error(fetch_location: &SourceFetchLocation, reserve_bytes: u64) -> QueryError {
    attach_fetch_budget_facts(
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.source_fetch.host_buffer_concurrency",
            "timed out waiting for object-store source fetch operation capacity",
        ),
        fetch_location,
        reserve_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_fetch_size_reserves_full_budget() {
        assert_eq!(host_fetch_reserve_bytes(100, 4, None), 100);
        assert_eq!(host_fetch_reserve_bytes(3, 4, None), 3);
        assert_eq!(host_fetch_reserve_bytes(100, 0, None), 100);
    }

    #[test]
    fn known_fetch_size_uses_estimate() {
        assert_eq!(host_fetch_reserve_bytes(100, 4, Some(7)), 7);
        assert_eq!(host_fetch_reserve_bytes(100, 4, Some(0)), 1);
    }

    #[test]
    fn pipeline_payload_accounting_rejects_underestimated_payload_without_growing() {
        let budget = crate::runtime::HostFetchBudget::new(10).expect("budget should be valid");
        let mut permit = budget.acquire(5).expect("initial reservation should fit");

        let error = account_host_fetch_payload_bytes(
            &mut permit,
            HostFetchPayloadAccounting::ReservedExact,
            6,
            budget.budget_bytes(),
        )
        .expect_err("pipeline accounting should reject underestimated payloads");

        assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
        assert_eq!(permit.bytes(), 5);
        assert!(budget.try_acquire(5).expect("remaining budget should still be available").is_some());
    }
}
