use std::time::Instant;

use super::{
    object_store_safety::{object_store_bucket_fact, source_fetch_location},
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
    context.check_cancellation("nexus_query_engine.source_fetch.host_buffer")?;
    let fetch_location = source_fetch_location(location, resolved_source)?;
    let reserve_bytes = estimated_fetch_bytes.unwrap_or_else(|| context.host_fetch_budget().budget_bytes()).max(1);
    let host_budget_wait_timer = Instant::now();
    let mut permit = context.host_fetch_budget().acquire(reserve_bytes).map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
    })?;
    let mut host_budget_wait_elapsed = host_budget_wait_timer.elapsed();
    let concurrency_wait_timer = Instant::now();
    let operation_permit = context.acquire_source_fetch_operation().map_err(|error| {
        error
            .with_fact_if_absent("source_fetch_substrate", "object_store")
            .with_fact_if_absent("source_fetch_scheme", "s3")
            .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
    })?;
    let concurrency_wait_elapsed = concurrency_wait_timer.elapsed();
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
        storage_profile_id: fetch_location.storage_profile_id.clone(),
    })?;
    let (sender, receiver) = std::sync::mpsc::channel();
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
                let grow_wait_timer = Instant::now();
                permit.grow_to(actual_bytes.max(1)).map_err(|error| {
                    error
                        .with_fact_if_absent("source_fetch_substrate", "object_store")
                        .with_fact_if_absent("source_fetch_scheme", "s3")
                        .with_fact_if_absent("bucket", object_store_bucket_fact(fetch_location.bucket.as_str()))
                        .with_fact_if_absent("actual_bytes", actual_bytes)
                })?;
                host_budget_wait_elapsed += grow_wait_timer.elapsed();
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
    Ok(ScheduledHostParquetFetch { receiver })
}
