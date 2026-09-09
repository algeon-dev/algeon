use std::{cell::Cell, rc::Rc};

use super::{
    ActiveReservationEventContext, FrozenResidentLease, GpuDataFrame, MemoryGrant, MemoryReservationEvent,
    MemoryReservationEventAction, MemoryReservationRequest, MemorySpaceId, QueryMetrics, ReservationObservation,
    ReservationProjectionLedgerHandle, ReservationTracker, Result, RuntimeReservationAction,
    RuntimeReservationDisposition, RuntimeSpanId, TaskAttemptId,
};

/// Controls whether an active reservation charges capacity itself or only
/// keeps already-charged bytes alive through another reservation lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum ActiveReservationSemantics {
    /// Charged bytes backing a resident frame or source read. Aliases may keep
    /// those bytes reserved after the owner is released.
    OwnedResident,
    /// Charged scratch or intermediate work space. It can be transferred to a
    /// later reservation without creating a second capacity charge.
    Work,
    /// Non-charging view of existing bytes. Releasing the last alias also
    /// releases any owner charge that was deferred while aliases were alive.
    Alias,
}

#[derive(Debug)]
pub(super) struct ActiveReservationLineage {
    pub(super) alias_count: Cell<usize>,
    pub(super) deferred_charged_release_bytes: Cell<u64>,
    /// Portion of a deferred charged release whose ledger ownership was
    /// transferred to an escaping result handle. The tracker must still emit
    /// the historical release event, but that portion must not decrement the
    /// grant until the result handle drops.
    pub(super) deferred_transferred_live_charge_bytes: Cell<u64>,
}

impl ActiveReservationLineage {
    pub(super) fn new() -> Self {
        Self {
            alias_count: Cell::new(0),
            deferred_charged_release_bytes: Cell::new(0),
            deferred_transferred_live_charge_bytes: Cell::new(0),
        }
    }
}

pub(in crate::exec) struct ActiveReservation {
    pub(super) memory_space: MemorySpaceId,
    pub(super) reserved_bytes: Rc<Cell<u64>>,
    pub(super) live_accounting: Option<MemoryGrant>,
    pub(super) event_context: ActiveReservationEventContext,
    pub(super) reservation_projection_ledger: ReservationProjectionLedgerHandle,
    pub(super) observation: Option<ReservationObservation>,
    pub(super) semantics: ActiveReservationSemantics,
    pub(super) lineage: Option<Rc<ActiveReservationLineage>>,
    pub(super) release_action: MemoryReservationEventAction,
    pub(super) bytes: u64,
    pub(super) charged_bytes: u64,
    /// Bytes in `charged_bytes` whose ledger charge now belongs to a
    /// [`FrozenResidentLease`].
    pub(super) transferred_live_charge_bytes: u64,
    pub(super) released: bool,
}

impl ActiveReservation {
    pub(super) fn owned_resident(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            live_accounting,
            bytes,
            request,
            reservation_projection_ledger,
            observation,
            ActiveReservationSemantics::OwnedResident,
        )
    }

    pub(super) fn work(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        Self::charged(
            memory_space,
            reserved_bytes,
            live_accounting,
            bytes,
            request,
            reservation_projection_ledger,
            observation,
            ActiveReservationSemantics::Work,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
    )]
    pub(super) fn charged(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
        semantics: ActiveReservationSemantics,
    ) -> Self {
        Self {
            memory_space,
            reserved_bytes,
            live_accounting,
            event_context: ActiveReservationEventContext::from_request(request),
            reservation_projection_ledger,
            observation,
            semantics,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::Release,
            bytes,
            charged_bytes: bytes,
            transferred_live_charge_bytes: 0,
            released: false,
        }
    }

    pub(super) fn borrowed_resident(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        Self {
            memory_space,
            reserved_bytes,
            live_accounting: None,
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger,
            observation,
            semantics: ActiveReservationSemantics::Alias,
            lineage: Some(Rc::new(ActiveReservationLineage::new())),
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            transferred_live_charge_bytes: 0,
            released: false,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
    )]
    pub(super) fn alias(
        memory_space: MemorySpaceId,
        reserved_bytes: Rc<Cell<u64>>,
        live_accounting: Option<MemoryGrant>,
        lineage: Rc<ActiveReservationLineage>,
        bytes: u64,
        request: &MemoryReservationRequest,
        reservation_projection_ledger: ReservationProjectionLedgerHandle,
        observation: Option<ReservationObservation>,
    ) -> Self {
        lineage.alias_count.set(lineage.alias_count.get().saturating_add(1));
        Self {
            memory_space,
            reserved_bytes,
            live_accounting,
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger,
            observation,
            semantics: ActiveReservationSemantics::Alias,
            lineage: Some(lineage),
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes,
            charged_bytes: 0,
            transferred_live_charge_bytes: 0,
            released: false,
        }
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

    /// Moves this reservation's existing grant charge into an escaping result
    /// lifetime. The ledger remains charged exactly once throughout the
    /// transfer; releasing this tracker still emits its normal historical
    /// release event but leaves the transferred bytes for the result handle.
    pub(in crate::exec) fn transfer_grant_live_reservation(&mut self) -> Option<FrozenResidentLease> {
        if self.released || self.charged_bytes == 0 || self.transferred_live_charge_bytes != 0 {
            return None;
        }
        let retained =
            self.live_accounting.as_ref().and_then(|grant| grant.transfer_live_reservation(self.charged_bytes))?;
        self.transferred_live_charge_bytes = self.charged_bytes;
        Some(retained)
    }

    /// Retains the exact escaping output owner when P1 cap-only execution did
    /// not create a projection reservation for it. Borrowed aliases remain
    /// owned by their input lease and must not be charged to this attempt.
    pub(in crate::exec) fn retain_exact_grant_output_if_uncharged(
        &self,
        output_bytes: u64,
    ) -> Result<Option<FrozenResidentLease>> {
        if self.released || self.is_alias() || self.charged_bytes != 0 || output_bytes == 0 {
            return Ok(None);
        }
        self.live_accounting.as_ref().map_or(Ok(None), |grant| grant.retain_live_reservation(output_bytes))
    }

    /// Retains the escaping GPU-resident output owner for this reservation.
    ///
    /// Every `GpuResident` sink uses this one path so an output escapes with
    /// a lease whether or not its stage created a projection reservation.
    /// The lease's byte count is executor-side evidence: the frozen resident
    /// charge is reconciled to the allocator's own measurement at freeze.
    pub(in crate::exec) fn retain_escaping_resident_output(
        &mut self,
        output_bytes: u64,
    ) -> Result<Option<FrozenResidentLease>> {
        match self.transfer_grant_live_reservation() {
            Some(reservation) => Ok(Some(reservation)),
            None => self.retain_exact_grant_output_if_uncharged(output_bytes),
        }
    }

    pub(in crate::exec) fn is_alias(&self) -> bool {
        matches!(self.semantics, ActiveReservationSemantics::Alias)
    }

    pub(super) fn transferred(mut self, request: &MemoryReservationRequest) -> Self {
        assert!(!self.released, "cannot transfer a released reservation");
        assert!(!self.is_alias(), "cannot transfer an alias reservation");
        let charged_bytes = self.charged_bytes;
        let transferred_live_charge_bytes = self.transferred_live_charge_bytes;
        let semantics = self.semantics;
        self.charged_bytes = 0;
        self.transferred_live_charge_bytes = 0;
        self.released = true;
        Self {
            memory_space: self.memory_space,
            reserved_bytes: Rc::clone(&self.reserved_bytes),
            live_accounting: self.live_accounting.clone(),
            event_context: ActiveReservationEventContext::from_request(request),
            reservation_projection_ledger: self.reservation_projection_ledger.clone(),
            observation: self.observation.clone(),
            semantics,
            lineage: self.lineage.clone(),
            release_action: MemoryReservationEventAction::Release,
            bytes: request.requested_bytes,
            charged_bytes,
            transferred_live_charge_bytes,
            released: false,
        }
    }

    pub(super) fn relabeled_alias(mut self, request: &MemoryReservationRequest) -> Self {
        assert!(!self.released, "cannot relabel a released reservation");
        assert!(self.is_alias(), "cannot relabel a non-alias reservation as an alias");
        let lineage = self.lineage.clone();
        self.released = true;
        Self {
            memory_space: self.memory_space,
            reserved_bytes: Rc::clone(&self.reserved_bytes),
            live_accounting: self.live_accounting.clone(),
            event_context: ActiveReservationEventContext::from_request(request).with_alias(),
            reservation_projection_ledger: self.reservation_projection_ledger.clone(),
            observation: self.observation.clone(),
            semantics: ActiveReservationSemantics::Alias,
            lineage,
            release_action: MemoryReservationEventAction::AliasRelease,
            bytes: request.requested_bytes,
            charged_bytes: 0,
            transferred_live_charge_bytes: 0,
            released: false,
        }
    }

    pub(super) fn release(&mut self, disposition: RuntimeReservationDisposition) -> Option<MemoryReservationEvent> {
        if self.released {
            return None;
        }
        match (self.semantics, self.lineage.as_ref()) {
            (ActiveReservationSemantics::Alias, Some(lineage)) => {
                lineage.alias_count.set(lineage.alias_count.get().saturating_sub(1));
                let (deferred_release, deferred_transferred_live_charge) = if lineage.alias_count.get() == 0 {
                    let bytes = lineage.deferred_charged_release_bytes.get();
                    lineage.deferred_charged_release_bytes.set(0);
                    let transferred = lineage.deferred_transferred_live_charge_bytes.get().min(bytes);
                    lineage.deferred_transferred_live_charge_bytes.set(0);
                    (bytes, transferred)
                } else {
                    (0, 0)
                };
                if deferred_release > 0 {
                    self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(deferred_release));
                    let ledger_release = deferred_release.saturating_sub(deferred_transferred_live_charge);
                    self.release_live_bytes(ledger_release);
                    self.released = true;
                    if let Some(observation) = &self.observation {
                        observation.record(
                            &self.event_context,
                            runtime_release_action(disposition),
                            disposition,
                            self.bytes,
                            0,
                            -i128::from(deferred_release),
                            self.reserved_bytes.get(),
                        );
                    }
                    return Some(self.event_context.event(
                        self.release_action,
                        self.bytes,
                        -i128::from(deferred_release),
                        self.reserved_bytes.get(),
                    ));
                }
            },
            (_, Some(lineage)) => {
                if lineage.alias_count.get() > 0 {
                    let transferred_live_charge = self.transferred_live_charge_bytes.min(self.charged_bytes);
                    lineage
                        .deferred_charged_release_bytes
                        .set(lineage.deferred_charged_release_bytes.get().saturating_add(self.charged_bytes));
                    lineage.deferred_transferred_live_charge_bytes.set(
                        lineage.deferred_transferred_live_charge_bytes.get().saturating_add(transferred_live_charge),
                    );
                    self.charged_bytes = 0;
                    self.transferred_live_charge_bytes = 0;
                }
            },
            (_, None) => {},
        }
        let charged_delta_bytes = -i128::from(self.charged_bytes);
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(self.charged_bytes));
        let ledger_release = self.charged_bytes.saturating_sub(self.transferred_live_charge_bytes);
        self.release_live_bytes(ledger_release);
        self.released = true;
        if let Some(observation) = &self.observation {
            let disposition = if charged_delta_bytes == 0
                && !self.is_alias()
                && self.lineage.as_ref().is_some_and(|lineage| lineage.alias_count.get() > 0)
            {
                RuntimeReservationDisposition::AliasDeferred
            } else {
                disposition
            };
            observation.record(
                &self.event_context,
                runtime_release_action(disposition),
                disposition,
                self.bytes,
                0,
                charged_delta_bytes,
                self.reserved_bytes.get(),
            );
        }
        Some(self.event_context.event(self.release_action, self.bytes, charged_delta_bytes, self.reserved_bytes.get()))
    }

    pub(super) fn retarget_observation(
        &mut self,
        task_attempt_id: Option<TaskAttemptId>,
        span_id: Option<RuntimeSpanId>,
    ) {
        if let Some(observation) = &mut self.observation {
            observation.retarget(task_attempt_id, span_id);
        }
    }

    pub(super) fn grow(&mut self, bytes: u64) -> u64 {
        if bytes == 0 {
            return 0;
        }
        assert!(!self.released, "cannot grow a released reservation");
        self.bytes = self.bytes.saturating_add(bytes);
        self.charged_bytes = self.charged_bytes.saturating_add(bytes);
        bytes
    }

    pub(super) fn shrink_to(&mut self, bytes: u64) -> u64 {
        if bytes >= self.bytes {
            return 0;
        }
        assert!(!self.released, "cannot shrink a released reservation");
        let refunded = (self.bytes - bytes).min(self.charged_bytes);
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(refunded));
        let transferred_refund = refunded.min(self.transferred_live_charge_bytes);
        self.release_live_bytes(refunded.saturating_sub(transferred_refund));
        self.bytes = bytes;
        self.charged_bytes = self.charged_bytes.saturating_sub(refunded);
        self.transferred_live_charge_bytes = self.transferred_live_charge_bytes.saturating_sub(transferred_refund);
        refunded
    }

    pub(super) fn release_live_bytes(&self, bytes: u64) {
        if let Some(grant) = &self.live_accounting {
            grant.release_live_bytes(bytes);
        }
    }
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        let disposition = if std::thread::panicking() {
            RuntimeReservationDisposition::UnwindRelease
        } else {
            RuntimeReservationDisposition::DropRelease
        };
        if let Some(event) = self.release(disposition) {
            self.reservation_projection_ledger.record_implicit_release(event, self.event_context.task_kind.clone());
        }
    }
}

pub(super) fn runtime_release_action(disposition: RuntimeReservationDisposition) -> RuntimeReservationAction {
    if disposition == RuntimeReservationDisposition::RollbackRelease {
        RuntimeReservationAction::Rollback
    } else {
        RuntimeReservationAction::Release
    }
}

pub(in crate::exec) struct ResidentGpuFrame {
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
}

pub(in crate::exec) type CommittedPartial = ResidentGpuFrame;

pub(in crate::exec::bounded) fn reconcile_work_reservation_to_output(
    reservations: &mut ReservationTracker,
    work_reservation: &mut ActiveReservation,
    output_bytes: u64,
    output_request: &MemoryReservationRequest,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    if output_bytes > work_reservation.bytes() {
        let delta = output_bytes.saturating_sub(work_reservation.bytes());
        reservations.try_grow(work_reservation, output_request.clone().with_requested_bytes(delta), metrics)?;
    } else {
        reservations.shrink_to_observed_with_request(work_reservation, output_bytes, output_request.clone(), metrics);
    }
    Ok(())
}
