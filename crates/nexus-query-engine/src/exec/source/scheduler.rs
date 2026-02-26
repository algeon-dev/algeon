use std::{
    thread::{self, JoinHandle},
    time::Instant,
};

use super::*;
use crate::{
    exec::bounded::{
        ActiveReservation, ReservationTracker,
        chunks::{
            SourceReadReservationContractQuality, record_chunk_retry_error, source_read_budget_correction_from_error,
            try_grow_source_read_reservation_for_budget_correction,
        },
    },
    runtime::MemoryReservationRequest,
};

pub(in crate::exec) struct ScheduledLocalParquetRead {
    chunk: SourceChunk,
    state: ScheduledLocalParquetReadState,
    budget_boundary: &'static str,
    prefetch: bool,
}

enum ScheduledLocalParquetReadState {
    Ready(Box<ScheduledLocalParquetReadOutput>),
    Running {
        reservation: Box<ActiveReservation>,
        handle: JoinHandle<ScheduledLocalParquetReadResult>,
        source_read_lease: Box<crate::runtime::SourceReadStreamLease>,
        source: Box<LocalParquetSource>,
        attempt: usize,
        max_attempts: usize,
    },
}

pub(in crate::exec) struct ScheduledLocalParquetReadOutput {
    pub(in crate::exec) chunk: SourceChunk,
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
    pub(in crate::exec) observed_bytes: u64,
}

struct ScheduledLocalParquetReadResult {
    frame: Result<GpuDataFrame>,
    metrics: QueryMetrics,
}

impl ScheduledLocalParquetRead {
    pub(in crate::exec) fn spawn(
        context: ExecutionContext,
        source: LocalParquetSource,
        chunk: SourceChunk,
        reservation: ActiveReservation,
        budget_bytes: u64,
        budget_boundary: &'static str,
        prefetch: bool,
        attempt: usize,
        max_attempts: usize,
    ) -> Result<Self> {
        let thread_chunk = chunk.clone();
        let source_read_lease = context.acquire_source_read_lease()?;
        let thread_context = source_read_lease.context().clone();
        let thread_source = source.clone();
        let handle = thread::spawn(move || {
            let mut metrics = QueryMetrics::default();
            let frame = read_local_parquet_chunk(
                &thread_context,
                &thread_source,
                &thread_chunk,
                budget_bytes,
                budget_boundary,
                &mut metrics,
            )
            .and_then(|frame| {
                thread_context.runtime().synchronize().map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.source_scheduler.worker_sync",
                        "failed to synchronize scheduled source read worker stream",
                        error,
                    )
                })?;
                Ok(frame)
            });
            ScheduledLocalParquetReadResult { frame, metrics }
        });
        Ok(Self {
            chunk,
            state: ScheduledLocalParquetReadState::Running {
                reservation: Box::new(reservation),
                handle,
                source_read_lease: Box::new(source_read_lease),
                source: Box::new(source),
                attempt,
                max_attempts,
            },
            budget_boundary,
            prefetch,
        })
    }

    pub(in crate::exec) fn ready(
        output: ScheduledLocalParquetReadOutput,
        budget_boundary: &'static str,
        prefetch: bool,
    ) -> Self {
        Self {
            chunk: output.chunk.clone(),
            state: ScheduledLocalParquetReadState::Ready(Box::new(output)),
            budget_boundary,
            prefetch,
        }
    }

    pub(in crate::exec) fn is_prefetch(&self) -> bool {
        self.prefetch
    }

    pub(in crate::exec) fn budget_boundary(&self) -> &'static str {
        self.budget_boundary
    }

    pub(in crate::exec) fn finish(
        self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Result<ScheduledLocalParquetReadOutput> {
        let ScheduledLocalParquetRead { chunk, state, budget_boundary, prefetch: _ } = self;
        let (mut reservation, handle, retry_context) = match state {
            ScheduledLocalParquetReadState::Ready(output) => return Ok(*output),
            ScheduledLocalParquetReadState::Running {
                reservation,
                handle,
                source_read_lease,
                source,
                attempt,
                max_attempts,
            } => (*reservation, handle, Some((*source_read_lease, *source, attempt, max_attempts))),
        };
        let wait_timer = Instant::now();
        let result = handle.join().map_err(|_panic| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.source_scheduler.join",
                "native source scheduler read worker panicked",
            )
            .with_fact("chunk_index", chunk.chunk_index)
            .with_fact("file_index", chunk.file_index)
            .with_fact("path", chunk.path.display())
            .with_fact("row_groups", format_row_groups(&chunk))
        })?;
        metrics.record_source_read_queue_wait(wait_timer.elapsed());
        metrics.merge_from(result.metrics);
        match result.frame {
            Ok(frame) => Ok(ScheduledLocalParquetReadOutput { chunk, frame, reservation, observed_bytes: 0 }),
            Err(error) => {
                if let Some((source_read_lease, source, attempt, max_attempts)) = retry_context
                    && let Some(frame) = try_finish_scheduled_source_read_with_corrected_budget(
                        source_read_lease.context(),
                        &source,
                        &chunk,
                        &mut reservation,
                        reservations,
                        metrics,
                        budget_boundary,
                        &error,
                        attempt,
                        max_attempts,
                    )?
                {
                    return Ok(ScheduledLocalParquetReadOutput { chunk, frame, reservation, observed_bytes: 0 });
                }
                reservations.release(reservation, metrics);
                Err(error)
            },
        }
    }

    pub(in crate::exec) fn abandon(self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        let ScheduledLocalParquetRead { state, .. } = self;
        let result = match state {
            ScheduledLocalParquetReadState::Ready(output) => {
                let output = *output;
                reservations.release(output.reservation, metrics);
                metrics.record_source_read_prefetch_abandoned();
                return;
            },
            ScheduledLocalParquetReadState::Running { reservation, handle, .. } => {
                let result = handle.join();
                reservations.release(*reservation, metrics);
                result
            },
        };
        metrics.record_source_read_prefetch_abandoned();
        if let Ok(result) = result {
            metrics.merge_from(result.metrics);
        }
    }
}

fn try_finish_scheduled_source_read_with_corrected_budget(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    budget_boundary: &'static str,
    error: &QueryError,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    context.check_cancellation("nexus_query_engine.source_scheduler.budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(error, reservations, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, metrics);
    metrics.record_source_chunk_retry();
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, 0, reservations.budget())
        .with_memory_boundary(budget_boundary)
        .with_node_id(chunk.source_node_id)
        .with_stage(budget_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt + 1, max_attempts)
        .with_source_read_contract(
            SourceReadReservationContractQuality::CorrectedLowerLevelBudget.as_str(),
            Some(correction.previous_budget_bytes()),
            Some(correction.corrected_budget_bytes()),
        );
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    try_grow_source_read_reservation_for_budget_correction(reservations, reservation, correction, request, metrics)?;
    metrics.record_source_chunk_attempt();
    read_local_parquet_chunk(context, source, chunk, correction.corrected_budget_bytes(), budget_boundary, metrics)
        .map(Some)
        .map_err(|error| correction.attach_to_error(error))
}
