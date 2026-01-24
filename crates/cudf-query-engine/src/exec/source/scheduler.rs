use std::{
    thread::{self, JoinHandle},
    time::Instant,
};

use super::*;
use crate::exec::bounded::{ActiveReservation, ReservationTracker};

pub(in crate::exec) struct ScheduledLocalParquetRead {
    chunk: SourceChunk,
    state: ScheduledLocalParquetReadState,
    budget_boundary: &'static str,
    prefetch: bool,
}

enum ScheduledLocalParquetReadState {
    Ready(Box<ScheduledLocalParquetReadOutput>),
    Running { reservation: ActiveReservation, handle: JoinHandle<ScheduledLocalParquetReadResult> },
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
    #[allow(clippy::too_many_arguments)]
    pub(in crate::exec) fn spawn(
        context: ExecutionContext,
        source: LocalParquetSource,
        chunk: SourceChunk,
        reservation: ActiveReservation,
        budget_bytes: u64,
        budget_boundary: &'static str,
        prefetch: bool,
    ) -> Self {
        let thread_chunk = chunk.clone();
        let handle = thread::spawn(move || {
            let mut metrics = QueryMetrics::default();
            let frame =
                read_local_parquet_chunk(&context, &source, &thread_chunk, budget_bytes, budget_boundary, &mut metrics);
            ScheduledLocalParquetReadResult { frame, metrics }
        });
        Self {
            chunk,
            state: ScheduledLocalParquetReadState::Running { reservation, handle },
            budget_boundary,
            prefetch,
        }
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
        let ScheduledLocalParquetRead { chunk, state, budget_boundary: _, prefetch: _ } = self;
        let (reservation, handle) = match state {
            ScheduledLocalParquetReadState::Ready(output) => return Ok(*output),
            ScheduledLocalParquetReadState::Running { reservation, handle } => (reservation, handle),
        };
        let wait_timer = Instant::now();
        let result = handle.join().map_err(|_panic| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.source_scheduler.join",
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
                reservations.release(reservation);
                Err(error)
            },
        }
    }

    pub(in crate::exec) fn abandon(self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        let ScheduledLocalParquetRead { state, .. } = self;
        let result = match state {
            ScheduledLocalParquetReadState::Ready(output) => {
                let output = *output;
                reservations.release(output.reservation);
                metrics.record_source_read_prefetch_abandoned();
                return;
            },
            ScheduledLocalParquetReadState::Running { reservation, handle } => {
                let result = handle.join();
                reservations.release(reservation);
                result
            },
        };
        metrics.record_source_read_prefetch_abandoned();
        if let Ok(result) = result {
            metrics.merge_from(result.metrics);
        }
    }
}
