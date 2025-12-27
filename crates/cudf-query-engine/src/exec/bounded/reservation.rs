use super::*;

pub(in crate::exec) struct ActiveReservation {
    reserved_bytes: Rc<Cell<u64>>,
    bytes: u64,
    released: bool,
}

impl ActiveReservation {
    pub(in crate::exec) fn new(reserved_bytes: Rc<Cell<u64>>, bytes: u64) -> Self {
        Self { reserved_bytes, bytes, released: false }
    }

    pub(in crate::exec) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(in crate::exec) fn release(&mut self) {
        if self.released {
            return;
        }
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(self.bytes));
        self.released = true;
    }
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug)]
pub(in crate::exec) struct ReservationTracker {
    budget_bytes: u64,
    reserved_bytes: Rc<Cell<u64>>,
}

impl ReservationTracker {
    pub(in crate::exec) fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes, reserved_bytes: Rc::new(Cell::new(0)) }
    }

    pub(in crate::exec) fn budget(&self) -> u64 {
        self.budget_bytes
    }

    pub(in crate::exec) fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes.get()
    }

    pub(in crate::exec) fn try_reserve(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        let remaining = self.budget_bytes.saturating_sub(self.reserved_bytes());
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied_event(&request, remaining);
            return Err(request.denied_error(remaining));
        }

        self.reserved_bytes.set(self.reserved_bytes().saturating_add(request.requested_bytes));
        metrics.record_reservation_granted_event(&request, request.requested_bytes);
        Ok(ActiveReservation::new(Rc::clone(&self.reserved_bytes), request.requested_bytes))
    }

    pub(in crate::exec) fn release(&mut self, mut reservation: ActiveReservation) {
        reservation.release();
    }
}

pub(in crate::exec) struct CommittedPartial {
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
}

pub(in crate::exec) type ResidentGpuFrame = CommittedPartial;
