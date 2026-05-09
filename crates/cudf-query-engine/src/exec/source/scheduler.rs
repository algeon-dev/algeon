use std::{
    thread::{self, JoinHandle},
    time::Instant,
};

use super::*;
use crate::exec::bounded::{ActiveReservation, ReservationTracker};

pub(in crate::exec) struct ScheduledLocalParquetRead {
    chunk: SourceChunk,
    reservation: ActiveReservation,
    handle: JoinHandle<ScheduledLocalParquetReadResult>,
    budget_boundary: &'static str,
    prefetch: bool,
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
        Self { chunk, reservation, handle, budget_boundary, prefetch }
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
        let wait_timer = Instant::now();
        let result = self.handle.join().map_err(|_panic| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.source_scheduler.join",
                "native source scheduler read worker panicked",
            )
            .with_fact("chunk_index", self.chunk.chunk_index)
            .with_fact("file_index", self.chunk.file_index)
            .with_fact("path", self.chunk.path.display())
            .with_fact("row_groups", format_row_groups(&self.chunk))
        })?;
        metrics.record_source_read_queue_wait(wait_timer.elapsed());
        metrics.merge_from(result.metrics);
        match result.frame {
            Ok(frame) => Ok(ScheduledLocalParquetReadOutput {
                chunk: self.chunk,
                frame,
                reservation: self.reservation,
                observed_bytes: 0,
            }),
            Err(error) => {
                reservations.release(self.reservation);
                Err(error)
            },
        }
    }

    pub(in crate::exec) fn abandon(self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        let result = self.handle.join();
        reservations.release(self.reservation);
        metrics.record_source_read_prefetch_abandoned();
        if let Ok(result) = result {
            metrics.merge_from(result.metrics);
        }
    }
}
