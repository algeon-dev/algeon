use super::*;
use crate::runtime::{ExecutionContext, MemorySpaceId};

pub(in crate::exec) struct ActiveReservation {
    memory_space: MemorySpaceId,
    reserved_bytes: Rc<Cell<u64>>,
    bytes: u64,
    charged_bytes: u64,
    released: bool,
}

impl ActiveReservation {
    pub(in crate::exec) fn new(memory_space: MemorySpaceId, reserved_bytes: Rc<Cell<u64>>, bytes: u64) -> Self {
        Self { memory_space, reserved_bytes, bytes, charged_bytes: bytes, released: false }
    }

    pub(in crate::exec) fn borrowed(memory_space: MemorySpaceId, reserved_bytes: Rc<Cell<u64>>, bytes: u64) -> Self {
        Self { memory_space, reserved_bytes, bytes, charged_bytes: 0, released: false }
    }

    pub(in crate::exec) fn memory_space(&self) -> MemorySpaceId {
        self.memory_space
    }

    pub(in crate::exec) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(in crate::exec) fn charged_bytes(&self) -> u64 {
        self.charged_bytes
    }

    pub(in crate::exec) fn release(&mut self) {
        if self.released {
            return;
        }
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(self.charged_bytes));
        self.released = true;
    }

    fn grow(&mut self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        assert!(!self.released, "cannot grow a released reservation");
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_add(bytes));
        self.bytes = self.bytes.saturating_add(bytes);
        self.charged_bytes = self.charged_bytes.saturating_add(bytes);
    }

    fn shrink_to(&mut self, bytes: u64) {
        if bytes >= self.bytes {
            return;
        }
        assert!(!self.released, "cannot shrink a released reservation");
        let refunded = (self.bytes - bytes).min(self.charged_bytes);
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(refunded));
        self.bytes = bytes;
        self.charged_bytes = self.charged_bytes.saturating_sub(refunded);
    }

    fn promote_charged_bytes_to_baseline(&mut self, promoted_bytes: u64) {
        if promoted_bytes == 0 {
            return;
        }
        assert!(!self.released, "cannot promote a released reservation");
        let promoted_charged_bytes = promoted_bytes.min(self.charged_bytes);
        self.charged_bytes = self.charged_bytes.saturating_sub(promoted_charged_bytes);
    }
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug)]
pub(in crate::exec) struct ReservationTracker {
    memory_space: MemorySpaceId,
    budget_bytes: u64,
    reserved_bytes: Rc<Cell<u64>>,
    source_cache_baseline_bytes: u64,
}

impl ReservationTracker {
    pub(in crate::exec) fn new(memory_space: MemorySpaceId, budget_bytes: u64) -> Self {
        Self { memory_space, budget_bytes, reserved_bytes: Rc::new(Cell::new(0)), source_cache_baseline_bytes: 0 }
    }

    pub(in crate::exec) fn new_for_context(
        context: &ExecutionContext,
        budget_bytes: u64,
        metrics: &mut QueryMetrics,
    ) -> Result<Self> {
        let mut tracker = Self::new(context.device_memory_space(), budget_bytes);
        if let Some(cache) = context.local_parquet_source_cache() {
            let retained_bytes = cache.retained_device_bytes()?;
            if retained_bytes > 0 {
                tracker.try_reserve_source_cache_baseline(retained_bytes, metrics)?;
                metrics.record_source_cache_retained_bytes(retained_bytes);
            }
        }
        Ok(tracker)
    }

    pub(in crate::exec) fn budget(&self) -> u64 {
        self.budget_bytes
    }

    pub(in crate::exec) fn reserved_bytes(&self) -> u64 {
        self.raw_reserved_bytes().saturating_sub(self.source_cache_baseline_bytes)
    }

    pub(in crate::exec) fn remaining_bytes(&self) -> u64 {
        self.budget_bytes.saturating_sub(self.raw_reserved_bytes())
    }

    pub(in crate::exec) fn can_reserve(&self, requested_bytes: u64) -> bool {
        requested_bytes <= self.remaining_bytes()
    }

    pub(in crate::exec) fn available_bytes_including(&self, reservation: &ActiveReservation) -> u64 {
        self.assert_reservation_belongs_to_tracker(reservation, "available_bytes_including");
        reservation.charged_bytes().saturating_add(self.remaining_bytes())
    }

    pub(in crate::exec) fn try_reserve(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        let request = self.stamp_request(request);
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_event(&request, remaining);
            return Err(request.denied_error(remaining));
        }

        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(request.requested_bytes));
        metrics.record_reservation_granted_event(&request, request.requested_bytes);
        Ok(ActiveReservation::new(self.memory_space, Rc::clone(&self.reserved_bytes), request.requested_bytes))
    }

    pub(in crate::exec) fn borrow_accounted(&self, bytes: u64) -> ActiveReservation {
        ActiveReservation::borrowed(self.memory_space, Rc::clone(&self.reserved_bytes), bytes)
    }

    pub(in crate::exec) fn release(&mut self, mut reservation: ActiveReservation) {
        self.assert_reservation_belongs_to_tracker(&reservation, "release");
        reservation.release();
    }

    pub(in crate::exec) fn try_grow(
        &mut self,
        reservation: &mut ActiveReservation,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
        self.validate_reservation_belongs_to_tracker(reservation, "try_grow")?;
        let request = self.stamp_request(request);
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_event(&request, remaining);
            return Err(request.denied_error(remaining));
        }

        reservation.grow(request.requested_bytes);
        metrics.record_reservation_granted_event(&request, request.requested_bytes);
        Ok(())
    }

    pub(in crate::exec) fn shrink_to_observed(&mut self, reservation: &mut ActiveReservation, observed_bytes: u64) {
        self.assert_reservation_belongs_to_tracker(reservation, "shrink_to_observed");
        reservation.shrink_to(observed_bytes);
    }

    pub(in crate::exec) fn promote_source_cache_insert(
        &mut self,
        reservation: &mut ActiveReservation,
        retained_bytes: u64,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
        self.validate_reservation_belongs_to_tracker(reservation, "promote_source_cache_insert")?;
        if retained_bytes == 0 {
            return Ok(());
        }

        let request = self.stamp_request(request);
        let additional_bytes = retained_bytes.saturating_sub(reservation.charged_bytes());
        let remaining = self.remaining_bytes();
        if additional_bytes > remaining {
            metrics.record_reservation_denied_event(&request, remaining);
            return Err(request.denied_error(remaining));
        }

        if additional_bytes > 0 {
            self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(additional_bytes));
        }
        reservation.promote_charged_bytes_to_baseline(retained_bytes);
        self.source_cache_baseline_bytes = self.source_cache_baseline_bytes.saturating_add(retained_bytes);
        metrics.record_reservation_granted_event(&request, retained_bytes);
        Ok(())
    }

    fn try_reserve_source_cache_baseline(&mut self, retained_bytes: u64, metrics: &mut QueryMetrics) -> Result<()> {
        let request =
            MemoryReservationRequest::new(ReservationKind::SourceCacheRetained, retained_bytes, self.budget_bytes)
                .with_memory_boundary("source_cache_retained")
                .with_stage("source_cache_retained");
        let request = self.stamp_request(request);
        let remaining = self.remaining_bytes();
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_event(&request, remaining);
            return Err(request.denied_error(remaining));
        }
        self.reserved_bytes.set(self.raw_reserved_bytes().saturating_add(request.requested_bytes));
        self.source_cache_baseline_bytes = self.source_cache_baseline_bytes.saturating_add(request.requested_bytes);
        metrics.record_reservation_granted_event(&request, request.requested_bytes);
        Ok(())
    }

    fn stamp_request(&self, mut request: MemoryReservationRequest) -> MemoryReservationRequest {
        request.memory_space = self.memory_space;
        request.budget_bytes = self.budget_bytes;
        request
    }

    fn raw_reserved_bytes(&self) -> u64 {
        self.reserved_bytes.get()
    }

    fn reservation_belongs_to_tracker(&self, reservation: &ActiveReservation) -> bool {
        reservation.memory_space() == self.memory_space && Rc::ptr_eq(&reservation.reserved_bytes, &self.reserved_bytes)
    }

    fn assert_reservation_belongs_to_tracker(&self, reservation: &ActiveReservation, operation: &'static str) {
        assert!(
            self.reservation_belongs_to_tracker(reservation),
            "reservation tracker invariant failed during {operation}: reservation memory_space={} tracker memory_space={}",
            reservation.memory_space(),
            self.memory_space
        );
    }

    fn validate_reservation_belongs_to_tracker(
        &self,
        reservation: &ActiveReservation,
        operation: &'static str,
    ) -> Result<()> {
        if self.reservation_belongs_to_tracker(reservation) {
            return Ok(());
        }

        Err(QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.bounded.reservation_tracker",
            "active reservation does not belong to this reservation tracker",
        )
        .with_fact("operation", operation)
        .with_fact("reservation_memory_space", reservation.memory_space())
        .with_fact("tracker_memory_space", self.memory_space))
    }
}

pub(in crate::exec) struct CommittedPartial {
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
}

pub(in crate::exec) type ResidentGpuFrame = CommittedPartial;
