use super::*;

/// One admitted bounded execution attempt: the execution context, reservation
/// ledger, and metrics authority that every operator in the attempt shares.
///
/// The attempt binds exactly what one admitted execution owns for its whole
/// lifetime. Operator-occurrence data — node identity, specs, schemas, frames,
/// strategies, and allocation facts — stays explicit at each call site instead
/// of accumulating here. Reservation mutations go through the paired
/// operations on this owner so the metrics authority can never be mispaired
/// with a different ledger.
pub(in crate::exec) struct BoundedExecutionAttempt<'a> {
    context: &'a ExecutionContext,
    reservations: ReservationTracker,
    metrics: &'a mut QueryMetrics,
}

impl<'a> BoundedExecutionAttempt<'a> {
    /// The only construction point is a bounded entry that owns the attempt's
    /// device budget and observation scope.
    pub(in crate::exec::bounded) fn new(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a mut QueryMetrics,
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        let reservations =
            ReservationTracker::new_for_context(context, device_budget_bytes, metrics, observation_scope)?;
        Ok(Self { context, reservations, metrics })
    }

    pub(in crate::exec) fn context(&self) -> &'a ExecutionContext {
        self.context
    }

    pub(in crate::exec) fn metrics(&self) -> &QueryMetrics {
        self.metrics
    }

    pub(in crate::exec) fn metrics_mut(&mut self) -> &mut QueryMetrics {
        self.metrics
    }

    /// Read-only ledger view for budget queries and observation scopes.
    pub(in crate::exec) fn reservations(&self) -> &ReservationTracker {
        &self.reservations
    }

    pub(in crate::exec::bounded) fn into_reservations(self) -> ReservationTracker {
        self.reservations
    }

    /// Wraps a test-owned ledger; production construction is bounded-entry-only.
    #[cfg(test)]
    pub(in crate::exec) fn for_test_tracker(
        context: &'a ExecutionContext,
        reservations: ReservationTracker,
        metrics: &'a mut QueryMetrics,
    ) -> Self {
        Self { context, reservations, metrics }
    }

    pub(in crate::exec) fn try_reserve(&mut self, request: MemoryReservationRequest) -> Result<ActiveReservation> {
        self.reservations.try_reserve(request, self.metrics)
    }

    #[cfg(test)]
    pub(in crate::exec) fn try_reserve_owned_resident(
        &mut self,
        request: MemoryReservationRequest,
    ) -> Result<ActiveReservation> {
        self.reservations.try_reserve_owned_resident(request, self.metrics)
    }

    pub(in crate::exec) fn try_reserve_work(&mut self, request: MemoryReservationRequest) -> Result<ActiveReservation> {
        self.reservations.try_reserve_work(request, self.metrics)
    }

    pub(in crate::exec) fn borrow_resident(&mut self, request: MemoryReservationRequest) -> ActiveReservation {
        self.reservations.borrow_resident(request, self.metrics)
    }

    pub(in crate::exec) fn alias_reservation(
        &mut self,
        source: &ActiveReservation,
        request: MemoryReservationRequest,
    ) -> ActiveReservation {
        self.reservations.alias_reservation(source, request, self.metrics)
    }

    pub(in crate::exec) fn relabel_alias_reservation(
        &mut self,
        reservation: ActiveReservation,
        request: MemoryReservationRequest,
    ) -> ActiveReservation {
        self.reservations.relabel_alias_reservation(reservation, request, self.metrics)
    }

    pub(in crate::exec) fn transfer(
        &mut self,
        reservation: ActiveReservation,
        request: MemoryReservationRequest,
    ) -> ActiveReservation {
        self.reservations.transfer(reservation, request, self.metrics)
    }

    pub(in crate::exec) fn release(&mut self, reservation: ActiveReservation) {
        self.reservations.release(reservation, self.metrics);
    }

    pub(in crate::exec) fn try_grow(
        &mut self,
        reservation: &mut ActiveReservation,
        request: MemoryReservationRequest,
    ) -> Result<()> {
        self.reservations.try_grow(reservation, request, self.metrics)
    }

    pub(in crate::exec) fn shrink_to_observed(&mut self, reservation: &mut ActiveReservation, observed_bytes: u64) {
        self.reservations.shrink_to_observed(reservation, observed_bytes, self.metrics);
    }

    pub(in crate::exec) fn shrink_to_observed_with_request(
        &mut self,
        reservation: &mut ActiveReservation,
        observed_bytes: u64,
        request: MemoryReservationRequest,
    ) {
        self.reservations.shrink_to_observed_with_request(reservation, observed_bytes, request, self.metrics);
    }

    pub(in crate::exec) fn reconcile_work_reservation_to_output(
        &mut self,
        work_reservation: &mut ActiveReservation,
        output_bytes: u64,
        output_request: &MemoryReservationRequest,
    ) -> Result<()> {
        reconcile_work_reservation_to_output(
            &mut self.reservations,
            work_reservation,
            output_bytes,
            output_request,
            self.metrics,
        )
    }
}
