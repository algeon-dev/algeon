use super::*;

/// Cancel-safe queued admission request.
///
/// Dropping this value before admission removes the queued request from the
/// service queue. This lets async servers wait without parking a blocking
/// thread inside [`QueryService::admit_query`].
#[derive(Debug)]
pub(crate) struct QueryAdmissionWaiter {
    pub(super) service: QueryService,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) requested_at: Instant,
    pub(super) deadline: Instant,
    pub(super) device_entries: DeviceAdmissionEntries,
    pub(super) execution_cancellation: ExecutionCancellation,
    pub(super) terminal: QueuedAdmissionTerminal,
    pub(super) queued: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AdmissionTicketPhase {
    Provisional,
    Active,
    /// Cleanup for an unpublished envelope. Its grant has never represented
    /// an admitted handle, so public admission snapshots must not count it.
    ProvisionalCleanup,
    /// Cleanup for a handle that was fully published. Its immutable grant is
    /// still held until the report commit completes, so it remains active in
    /// public admission snapshots.
    ActiveCleanup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct AdmissionLifecycleSnapshot {
    pub(super) provisional: usize,
    pub(super) active: usize,
    pub(super) published_cleanup: usize,
    pub(super) cleanup: usize,
}

#[derive(Debug, Default)]
struct AdmissionLifecycleTrackerState {
    next_ticket_id: u64,
    tickets: BTreeMap<u64, AdmissionTicketPhase>,
}

/// The single RAII source for admission lifecycle quiescence.
///
/// A ticket is created when a ledger grant becomes provisional, transitions to
/// `Active` only after the complete envelope is ready, and transitions to
/// `Cleanup` before rollback or terminal resource cleanup.  Capacity remains
/// owned by the per-device ledgers; these phases are lifecycle diagnostics and
/// a close/drain fence only.  Every transition is checked against the ticket's
/// current phase so underflow and double-release bugs cannot be hidden by
/// saturating arithmetic.
#[derive(Debug, Clone, Default)]
pub(super) struct AdmissionLifecycleTracker {
    state: Arc<Mutex<AdmissionLifecycleTrackerState>>,
}

impl AdmissionLifecycleTracker {
    pub(super) fn begin_provisional(&self, service: Weak<QueryServiceInner>) -> AdmissionTicket {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state.next_ticket_id;
        state.next_ticket_id = state.next_ticket_id.checked_add(1).expect("admission lifecycle ticket id overflow");
        let previous = state.tickets.insert(id, AdmissionTicketPhase::Provisional);
        debug_assert!(previous.is_none(), "admission lifecycle ticket ids must be unique");
        AdmissionTicket { tracker: self.clone(), service, id, phase: AdmissionTicketPhase::Provisional }
    }

    fn transition(&self, id: u64, expected: AdmissionTicketPhase, next: AdmissionTicketPhase) -> Result<()> {
        let mut state = self.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission.lifecycle",
                "admission lifecycle tracker mutex was poisoned",
            )
            .with_fact("reason", "lifecycle_tracker_poisoned")
        })?;
        let Some(current) = state.tickets.get_mut(&id) else {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket is missing",
            )
            .with_fact("reason", "lifecycle_ticket_missing")
            .with_fact("ticket_id", id));
        };
        if *current != expected {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket transition conflicts with its current phase",
            )
            .with_fact("reason", "lifecycle_transition_conflict")
            .with_fact("ticket_id", id)
            .with_fact("expected_phase", format!("{expected:?}"))
            .with_fact("actual_phase", format!("{current:?}")));
        }
        *current = next;
        Ok(())
    }

    fn finish_cleanup(&self, id: u64) -> Result<()> {
        let mut state = self.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission.lifecycle",
                "admission lifecycle tracker mutex was poisoned",
            )
            .with_fact("reason", "lifecycle_tracker_poisoned")
        })?;
        match state.tickets.get(&id).copied() {
            Some(actual) if actual.is_cleanup() => {
                state.tickets.remove(&id);
                Ok(())
            },
            Some(actual) => Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket dropped before cleanup completed",
            )
            .with_fact("reason", "lifecycle_drop_conflict")
            .with_fact("ticket_id", id)
            .with_fact("expected_phase", "cleanup")
            .with_fact("actual_phase", format!("{actual:?}"))),
            None => Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket was already dropped",
            )
            .with_fact("reason", "lifecycle_ticket_missing")
            .with_fact("ticket_id", id)),
        }
    }

    pub(super) fn snapshot(&self) -> AdmissionLifecycleSnapshot {
        let state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut snapshot = AdmissionLifecycleSnapshot::default();
        for phase in state.tickets.values() {
            match phase {
                AdmissionTicketPhase::Provisional => snapshot.provisional += 1,
                AdmissionTicketPhase::Active => snapshot.active += 1,
                AdmissionTicketPhase::ProvisionalCleanup => snapshot.cleanup += 1,
                AdmissionTicketPhase::ActiveCleanup => {
                    snapshot.published_cleanup += 1;
                    snapshot.cleanup += 1;
                },
            }
        }
        snapshot
    }

    pub(super) fn is_quiescent(&self) -> bool {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).tickets.is_empty()
    }
}

/// One linear lifecycle token.  It is intentionally non-cloneable: ownership
/// moves from admission transaction to handle and finally to cleanup.
#[derive(Debug)]
pub(super) struct AdmissionTicket {
    tracker: AdmissionLifecycleTracker,
    service: Weak<QueryServiceInner>,
    id: u64,
    phase: AdmissionTicketPhase,
}

impl AdmissionTicket {
    pub(super) fn is_cleanup(&self) -> bool {
        self.phase.is_cleanup()
    }

    pub(super) fn transition_to_active(&mut self) -> Result<()> {
        self.tracker.transition(self.id, AdmissionTicketPhase::Provisional, AdmissionTicketPhase::Active)?;
        self.phase = AdmissionTicketPhase::Active;
        Ok(())
    }

    pub(super) fn transition_to_cleanup(&mut self) -> Result<()> {
        let cleanup_phase = match self.phase {
            AdmissionTicketPhase::Provisional => AdmissionTicketPhase::ProvisionalCleanup,
            AdmissionTicketPhase::Active => AdmissionTicketPhase::ActiveCleanup,
            AdmissionTicketPhase::ProvisionalCleanup | AdmissionTicketPhase::ActiveCleanup => return Ok(()),
        };
        self.tracker.transition(self.id, self.phase, cleanup_phase)?;
        self.phase = cleanup_phase;
        Ok(())
    }
}

impl AdmissionTicketPhase {
    const fn is_cleanup(self) -> bool {
        matches!(self, Self::ProvisionalCleanup | Self::ActiveCleanup)
    }
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        match self.transition_to_cleanup().and_then(|()| self.tracker.finish_cleanup(self.id)) {
            Ok(()) => {
                if let Some(service) = self.service.upgrade() {
                    service.notify_admission_available();
                }
            },
            Err(error) => {
                eprintln!("[nexus-query-engine] warning: admission lifecycle ticket drop invariant failed: {error}");
            },
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct AdmissionState {
    pub(super) lifecycle: AdmissionLifecycleTracker,
    pub(super) queued: VecDeque<AdmissionRequest>,
    /// Largest queue depth observed by this service since construction.
    ///
    /// The admission mutex owns both the queue and this value, so snapshots
    /// cannot miss an enqueue that raced with a later drain.  It deliberately
    /// never decreases while requests leave the queue.
    queue_high_watermark: usize,
    pub(super) closed: bool,
}

impl AdmissionState {
    /// Inserts a request and records the resulting queue high-water mark.
    pub(super) fn enqueue_request(&mut self, request: AdmissionRequest) {
        self.queued.push_back(request);
        self.record_queue_depth();
    }

    /// Removes one request while retaining the historical queue peak.
    pub(super) fn remove_queued_at(&mut self, position: usize) -> Option<AdmissionRequest> {
        let request = self.queued.remove(position);
        self.record_queue_depth();
        request
    }

    /// Takes the queue for a service-owned sweep. The subsequent replacement
    /// records the post-sweep transition before another snapshot can observe
    /// the state.
    pub(super) fn take_queued_requests(&mut self) -> VecDeque<AdmissionRequest> {
        std::mem::take(&mut self.queued)
    }

    /// Reinstalls the queue after a service-owned sweep and retains its peak.
    pub(super) fn replace_queued_requests(&mut self, queued: VecDeque<AdmissionRequest>) {
        self.queued = queued;
        self.record_queue_depth();
    }

    /// Returns the service-owned queue high-water mark.
    pub(super) const fn queue_high_watermark(&self) -> usize {
        self.queue_high_watermark
    }

    fn record_queue_depth(&mut self) {
        self.queue_high_watermark = self.queue_high_watermark.max(self.queued.len());
    }
}

/// Shared terminal result for a queued admission owner.
///
/// Queue maintenance is deliberately allowed to remove an expired, cancelled,
/// or health-rejected request on behalf of a different caller.  The owning
/// blocking/async waiter still needs the precise terminal result rather than
/// discovering that its request vanished and waiting until its ordinary queue
/// deadline. Store the one owned error before releasing the admission mutex.
/// A queue sweep separately retains the immutable completion disposition, so
/// handing this error to the waiter cannot suppress terminal completion.
#[derive(Debug, Default)]
struct QueuedAdmissionTerminalState {
    error: Option<QueryError>,
    primary_failure: Option<crate::observability::AttemptFailureSummary>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct QueuedAdmissionTerminal {
    state: Arc<Mutex<QueuedAdmissionTerminalState>>,
}

impl QueuedAdmissionTerminal {
    /// Records the waiter-visible error and returns the independent
    /// disposition that must finish the removed request's completion payload.
    pub(super) fn record(&self, error: QueryError) -> RuntimeTerminalOutcome {
        let disposition = RuntimeTerminalOutcome::from_query_error(&error);
        let primary_failure = RuntimeTerminalOutcome::primary_failure_from_query_error(&error);
        let mut terminal = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if terminal.error.is_none() {
            terminal.primary_failure = primary_failure;
            terminal.error = Some(error);
        }
        disposition
    }

    pub(super) fn take(&self) -> Option<QueryError> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).error.take()
    }

    /// Transfers the borrowed-error summary captured by queue maintenance to
    /// the completion path. The waiter may already have taken its full error,
    /// so the stable report must not reconstruct a generic fallback.
    pub(super) fn take_primary_failure(&self) -> Option<crate::observability::AttemptFailureSummary> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).primary_failure.take()
    }
}

#[derive(Debug)]
pub(super) struct AdmissionRequest {
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    /// Engine-minted identity retained from request submission through the
    /// selected ledger grant and allocator-event correlation.
    pub(super) attempt_id: crate::observability::AttemptId,
    pub(super) completion: QueryCompletion,
    /// Optional plan proof submitted by planning.  This is distinct from the
    /// always-resolved placement requirement below: no proof means execution
    /// coverage validation remains unavailable, rather than asserting an
    /// empty proof.
    pub(super) submitted_plan_capability_evidence: Option<AttemptRequirement>,
    /// Policy-resolved, device-specific entries retained from submission
    /// through the execution context that consumes the selected grant.
    pub(super) device_entries: DeviceAdmissionEntries,
    /// Monotonic FIFO identity retained independently from admission order.
    pub(super) submission_seq: u64,
    /// Device ordinals established at submission; capability filtering belongs
    /// here, and each configured device is considered independently.
    pub(super) compatible_devices: Vec<usize>,
    /// Finite number of compatible-device bypasses before slot protection.
    pub(super) overtakes_remaining: u32,
    /// Ordinal currently holding this waiter's liveness protection.
    pub(super) protected_device: Option<usize>,
    pub(super) deadline: Instant,
    /// The attempt-wide token also used by transactional envelope acquisition.
    pub(super) execution_cancellation: ExecutionCancellation,
    /// Terminal result recorded by queue maintenance before it drops this
    /// request's completion payload outside the admission mutex.
    pub(super) terminal: QueuedAdmissionTerminal,
}

/// One immutable requirement/envelope pair resolved for one installed device.
/// Placement must never derive a second pair after submission.
#[derive(Debug, Clone)]
pub(super) struct DeviceAdmissionEntry {
    pub(super) requirement: AttemptRequirement,
    pub(super) admitted_reservation_envelope: Option<AttemptReservationEnvelope>,
}

pub(super) type DeviceAdmissionEntries = BTreeMap<usize, DeviceAdmissionEntry>;

impl AdmissionRequest {
    pub(super) fn entry_for_device(&self, device_ordinal: usize) -> Result<&DeviceAdmissionEntry> {
        self.device_entries.get(&device_ordinal).ok_or_else(|| missing_device_admission_entry_error(device_ordinal))
    }

    pub(super) fn requirement_for_device(&self, device_ordinal: usize) -> Result<&AttemptRequirement> {
        Ok(&self.entry_for_device(device_ordinal)?.requirement)
    }

    pub(super) fn admitted_reservation_envelope_for_device(
        &self,
        device_ordinal: usize,
    ) -> Result<&Option<AttemptReservationEnvelope>> {
        Ok(&self.entry_for_device(device_ordinal)?.admitted_reservation_envelope)
    }

    /// Captures the controller's immutable finite-overtake state at the
    /// moment this queued request is selected for placement. The request is
    /// removed immediately afterwards, so later snapshots cannot recover
    /// this fact truthfully.
    pub(super) fn admission_wait_fact(&self, overtake_limit: u32) -> crate::observability::AttemptAdmissionWaitFact {
        crate::observability::AttemptAdmissionWaitFact {
            submission_sequence: self.submission_seq,
            overtake_limit,
            overtakes_consumed: overtake_limit.saturating_sub(self.overtakes_remaining),
            overtakes_remaining: self.overtakes_remaining,
            protection_device_ordinal: self.protected_device,
        }
    }
}

pub(super) fn missing_device_admission_entry_error(device_ordinal: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        "nexus_query_engine.query_service.admission",
        "admission request is missing the selected device-specific requirement",
    )
    .with_fact("reason", "missing_device_specific_admission_resolution")
    .with_fact("device_ordinal", device_ordinal)
}

/// The one ownership boundary for terminal admission completion.
///
/// This payload starts at backend admission, remains with a queued request,
/// and, once admitted, stays with the query handle until every GPU-bound
/// resource has been torn down.  It deliberately owns the opaque activity
/// lease so activity callbacks cannot run while admission or resource locks
/// are held.
#[derive(Debug)]
pub(crate) struct QueryCompletion {
    observation_owner: Option<RuntimeObservationOwner>,
    attempt_finalizer_slot: Option<crate::attempt::AttemptFinalizerSlot>,
    attempt_finalizer: Option<crate::attempt::AttemptFinalizer>,
    observation_permit: Option<ObservationPermit>,
    observation_activity: Option<QueryActivityLease>,
    completion_dispatcher: Option<ObservationCompletionDispatcher>,
    gpu_resource_permit: Option<GpuResourcePermit>,
}

impl QueryCompletion {
    pub(super) fn new(
        observation_owner: Option<RuntimeObservationOwner>,
        attempt_finalizer_slot: Option<crate::attempt::AttemptFinalizerSlot>,
        observation_permit: Option<ObservationPermit>,
        observation_activity: Option<QueryActivityLease>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
        gpu_resource_permit: Option<GpuResourcePermit>,
    ) -> Self {
        Self {
            observation_owner,
            attempt_finalizer_slot,
            attempt_finalizer: None,
            observation_permit,
            observation_activity,
            completion_dispatcher,
            gpu_resource_permit,
        }
    }

    /// Publishes the completion payload with its GPU-generation permit retained
    /// by the admission envelope. The envelope is the one owner of every
    /// admission resource through terminal cleanup; observer completion may
    /// only run after that owner has released the permit.
    pub(super) fn into_admitted(
        mut self,
        envelope: &mut super::service::transaction::AdmissionEnvelope,
    ) -> Result<Self> {
        if let Some(slot) = self.attempt_finalizer_slot.as_ref() {
            let finalizer = slot.take_for_admission().map_err(|error| {
                QueryError::new(
                    crate::error::QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.attempt_finalizer",
                    "attempt finalizer was unavailable while publishing admission",
                )
                .with_fact("reason", "attempt_finalizer_unavailable")
                .with_fact("lifecycle_error", error.to_string())
            })?;
            self.attempt_finalizer_slot.take();
            self.attempt_finalizer = Some(finalizer);
        }
        envelope.retain_gpu_resource_permit(self.gpu_resource_permit.take());
        Ok(self)
    }

    pub(super) fn observation_owner(&self) -> Option<RuntimeObservationOwner> {
        self.observation_owner.clone()
    }

    /// Samples the query-relative attempt clock only when this completion
    /// owns the admitted attempt finalizer. A missing binding stays
    /// unavailable in stable resource evidence rather than receiving a
    /// fabricated wall-clock timestamp.
    pub(super) fn attempt_monotonic_ns(&self) -> Option<u64> {
        self.attempt_finalizer.as_ref().map(crate::attempt::AttemptFinalizer::monotonic_ns)
    }

    pub(super) fn begin_query_cleanup(&self) {
        if let Some(owner) = &self.observation_owner {
            owner.begin_query_cleanup();
        }
    }

    pub(super) fn finish_error(self, error: &QueryError) {
        // Retain the bounded engine-owned summary while the original error is
        // still borrowed. `finish` deliberately consumes only its closed
        // terminal projection, so doing this afterwards would lose operation,
        // allowlisted facts, and redacted text from an attempt report.
        if let Some(summary) = RuntimeTerminalOutcome::primary_failure_from_query_error(error) {
            self.record_primary_failure(summary);
        }
        self.finish(RuntimeTerminalOutcome::from_query_error(error));
    }

    fn record_primary_failure(&self, summary: crate::observability::AttemptFailureSummary) {
        if let Some(finalizer) = &self.attempt_finalizer {
            finalizer.record_primary_failure(summary);
        } else if let Some(slot) = &self.attempt_finalizer_slot {
            let _ = slot.record_primary_failure(summary);
        }
    }

    pub(super) fn finish_timeout(self, error: &QueryError) {
        self.finish_error(error);
    }

    pub(super) fn finish_cancelled(self) {
        self.finish(RuntimeTerminalOutcome::Cancelled);
    }

    /// Finishes a queue-removed request from the disposition captured by its
    /// sweep. This must not consult the waiter-facing terminal error: the
    /// waiter may already have consumed that error before the sweep runs.
    pub(super) fn finish_terminal_disposition(
        self,
        disposition: RuntimeTerminalOutcome,
        primary_failure: Option<crate::observability::AttemptFailureSummary>,
    ) {
        if let Some(summary) = primary_failure {
            self.record_primary_failure(summary);
        }
        self.finish(disposition);
    }

    /// Resolves the terminal outcome before physical cleanup begins.
    ///
    /// An admitted attempt that reaches release without an execution terminal
    /// boundary must establish its fallback before CUDA cleanup is scheduled.
    /// The resolved outcome is then carried through cleanup into the sole
    /// report/capacity commit, so a late mechanical fallback cannot race an
    /// already-resolved attempt terminal state.
    pub(super) fn resolve_terminal_outcome_for_cleanup(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> RuntimeTerminalOutcome {
        let observed_outcome = self
            .observation_owner
            .as_ref()
            .map_or_else(|| fallback.clone(), |owner| owner.terminal_outcome_for_cleanup(fallback.clone()));
        self.attempt_finalizer.as_ref().map_or(observed_outcome.clone(), |attempt_finalizer| {
            match attempt_finalizer.terminalize_for_release(observed_outcome.clone()) {
                Ok(outcome) => outcome,
                Err(error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to establish attempt terminal outcome before GPU cleanup: {error}"
                    );
                    observed_outcome
                },
            }
        })
    }

    /// Seals the observation inline and returns the deferred activity tail.
    ///
    /// The release commit seals the report before capacity reuse, so the seal
    /// cannot be deferred to the dispatcher lane.
    /// The opaque activity lease still runs last on the dispatcher tail lane:
    /// its destructor may reenter backend close and must observe the released
    /// backend GPU permit.
    pub(super) fn seal_after_gpu_cleanup(mut self, outcome: RuntimeTerminalOutcome) -> QueryCompletionActivityTail {
        debug_assert!(self.gpu_resource_permit.is_none(), "GPU permit must leave before observer completion");
        debug_assert!(self.attempt_finalizer.is_none(), "attempt finalizer must be consumed before report seal");
        debug_assert!(
            self.attempt_finalizer_slot.is_none(),
            "admitted completion cannot retain owner finalizer routing"
        );
        let dispatcher = self.completion_dispatcher.take();
        let owner = self.observation_owner.take();
        let permit = self.observation_permit.take();
        let activity = self.observation_activity.take();
        // The seal publishes the report inline so capacity reuse always
        // follows report-ready. A seal panic already completed degraded
        // delivery inside the recorder; suppress it like the combined path.
        let delivery = owner.and_then(|owner| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.seal_query_cleanup(outcome))).ok()
        });
        QueryCompletionActivityTail { activity, dispatcher, delivery, permit }
    }

    /// Drops an admitted attempt's unpublishable observation after finalizer
    /// failure. The capacity owner has already selected fail-closed health;
    /// fabricating a second finalization or sealing an incomplete disposition would be
    /// less truthful than explicit publication loss. Keep the activity tail's
    /// ordering identical to normal sealing.
    pub(super) fn discard_unfinalized_after_gpu_cleanup(mut self) -> QueryCompletionActivityTail {
        debug_assert!(self.gpu_resource_permit.is_none(), "GPU permit must leave before observer completion");
        self.attempt_finalizer.take();
        self.attempt_finalizer_slot.take();
        drop(self.observation_owner.take());
        let dispatcher = self.completion_dispatcher.take();
        let permit = self.observation_permit.take();
        let activity = self.observation_activity.take();
        QueryCompletionActivityTail { activity, dispatcher, delivery: None, permit }
    }

    /// Consumes the linear post-cleanup finalizer before the report seal.
    pub(super) fn finalize_attempt_after_gpu_cleanup(
        &mut self,
        terminal_commit: crate::observability::AttemptTerminalCommit,
    ) -> std::result::Result<(), crate::attempt::AttemptLifecycleError> {
        match self.attempt_finalizer.take() {
            Some(finalizer) => finalizer.finalize_admitted(terminal_commit),
            None if self.attempt_finalizer_slot.is_some() => {
                Err(crate::attempt::AttemptLifecycleError::FinalizationUnavailable)
            },
            None => Ok(()),
        }
    }

    fn finish(mut self, fallback: RuntimeTerminalOutcome) {
        // A rejected or cancelled request has no handle cleanup owner. Release
        // its backend generation before observer delivery so enqueue fallback
        // cannot synchronously reenter backend close while retaining GPU state.
        drop(self.gpu_resource_permit.take());
        self.dispatch_after_gpu_cleanup(fallback);
    }

    fn dispatch_after_gpu_cleanup(mut self, fallback: RuntimeTerminalOutcome) {
        debug_assert!(self.gpu_resource_permit.is_none(), "GPU permit must leave before observer completion");
        if let Some(dispatcher) = self.completion_dispatcher.take() {
            let activity_tail_dispatcher = dispatcher.clone();
            dispatcher.dispatch(move || {
                let activity = self.finish_observation(fallback);
                if let Some(activity) = activity {
                    activity_tail_dispatcher.dispatch_activity_tail(move || Self::finish_activity(Some(activity)));
                }
            });
        } else {
            self.finish_inline(fallback);
        }
    }

    fn finish_inline(mut self, fallback: RuntimeTerminalOutcome) {
        // Recorder sealing may invoke observer callbacks. It must happen after
        // every GPU reference has been released and outside admission/resource
        // locks. The recorder itself preserves a prior structured execution
        // outcome over this fallback.
        let activity = self.finish_observation(fallback);
        Self::finish_activity(activity);
    }

    /// Seals and delivers the observation while its permit still fences that
    /// work, then relinquishes the permit before returning the opaque activity
    /// lease. The caller decides the activity lane so observer delivery cannot
    /// be starved by a reentrant activity close.
    fn finish_observation(&mut self, fallback: RuntimeTerminalOutcome) -> Option<QueryActivityLease> {
        let owner = self.observation_owner.take();
        let permit = self.observation_permit.take();
        let activity = self.observation_activity.take();
        let finalized = if self.attempt_finalizer.is_some() {
            // This completion was already admitted but reached a recovery path
            // before its post-cleanup finalizer ran. Publishing would invent
            // admitted cleanup/capacity facts, so fail publication instead.
            eprintln!(
                "[nexus-query-engine] warning: admitted completion dropped before finalization; suppressing report publication"
            );
            false
        } else if let Some(slot) = self.attempt_finalizer_slot.take() {
            match slot.finalize_not_admitted(fallback.clone()) {
                Ok(_) => true,
                Err(crate::attempt::AttemptLifecycleError::FinalizationUnavailable)
                    if owner.as_ref().is_some_and(|owner| owner.sealed_snapshot().is_some()) =>
                {
                    true
                },
                Err(error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to finalize unadmitted attempt before seal; suppressing report publication: {error}"
                    );
                    false
                },
            }
        } else {
            true
        };
        if finalized && let Some(owner) = owner {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = owner.finish_query_cleanup(fallback);
            }));
        }
        // The permit fences observer delivery but cannot include an opaque
        // activity destructor: that destructor may synchronously reenter
        // backend close and wait for unrelated observation permits.
        drop(permit);
        activity
    }

    fn finish_activity(activity: Option<QueryActivityLease>) {
        // Keep activity last even if a recorder observer panicked. A panicking
        // activity destructor is isolated here so Drop never double-panics.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(activity)));
    }
}

/// The opaque activity lease deferred until after capacity release. Running
/// it can reenter backend close, so it always observes the released GPU
/// permit and never gates reuse.
pub(super) struct QueryCompletionActivityTail {
    activity: Option<QueryActivityLease>,
    dispatcher: Option<ObservationCompletionDispatcher>,
    delivery: Option<crate::observability::QueryCleanupSealDelivery>,
    permit: Option<ObservationPermit>,
}

impl QueryCompletionActivityTail {
    /// Runs deferred observer delivery, then the activity lease. Delivery and
    /// consumption both happen after capacity release and never gate reuse.
    pub(super) fn finish(self) {
        let QueryCompletionActivityTail { activity, dispatcher, delivery, permit } = self;
        match dispatcher {
            Some(dispatcher) => {
                // Delivery and the activity tail ride the dispatcher lane so
                // a blocking observer or reentrant activity destructor never
                // stalls the stream-cleanup worker. The observation permit
                // still fences delivery, and the activity lease runs last.
                let tail_dispatcher = dispatcher.clone();
                dispatcher.dispatch(move || {
                    if let Some(delivery) = delivery {
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| delivery.deliver()));
                    }
                    drop(permit);
                    tail_dispatcher.dispatch_activity_tail(move || QueryCompletion::finish_activity(activity));
                });
            },
            None => {
                if let Some(delivery) = delivery {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| delivery.deliver()));
                }
                // The permit fences observer delivery but cannot include an
                // opaque activity destructor: that destructor may reenter
                // backend close and wait for unrelated observation permits.
                drop(permit);
                QueryCompletion::finish_activity(activity);
            },
        }
    }
}

impl Drop for QueryCompletion {
    fn drop(&mut self) {
        if self.observation_owner.is_none()
            && self.attempt_finalizer_slot.is_none()
            && self.attempt_finalizer.is_none()
            && self.observation_permit.is_none()
            && self.observation_activity.is_none()
            && self.gpu_resource_permit.is_none()
        {
            return;
        }

        // Recover every terminal path that unwinds before it can explicitly
        // finish. Do not let a cleanup panic turn an already-unwinding path
        // into an abort; recorder sealing receives an explicit Panicked
        // fallback in that case.
        let fallback =
            if std::thread::panicking() { RuntimeTerminalOutcome::Panicked } else { RuntimeTerminalOutcome::Dropped };
        let recovered = Self {
            observation_owner: self.observation_owner.take(),
            attempt_finalizer_slot: self.attempt_finalizer_slot.take(),
            attempt_finalizer: self.attempt_finalizer.take(),
            observation_permit: self.observation_permit.take(),
            observation_activity: self.observation_activity.take(),
            completion_dispatcher: self.completion_dispatcher.take(),
            gpu_resource_permit: self.gpu_resource_permit.take(),
        };
        // `finish` consumes the payload. Keep a local mutable binding only so
        // its fields are emptied before this Drop implementation returns.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recovered.finish(fallback)));
    }
}

#[derive(Debug)]
pub(super) struct PlacedQuery {
    pub(super) device_ordinal: usize,
    pub(super) memory_grant: MemoryGrant,
}

impl QueryAdmissionWaiter {
    /// Attempts to complete this queued admission without blocking.
    ///
    /// Returns `Ok(None)` while the request is still queued behind active work
    /// or a compatible device's current capacity/health constraint. Queue
    /// timeout and service close remove the request and return a structured
    /// error.
    pub fn try_admit(&mut self) -> Result<Option<QueryHandle>> {
        if let Some(error) = self.terminal.take() {
            self.queued = false;
            return Err(error);
        }
        if !self.queued {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "query admission waiter has already completed",
            )
            .with_fact("admission_request_id", self.admission_request_id)
            .with_fact("reason", "admission_waiter_completed"));
        }

        if self.execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&self.execution_cancellation);
            let removed = match self.service.inner.admission.lock() {
                Ok(mut state) => {
                    let removed = self
                        .service
                        .remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            AdmissionLifecycleState::CancellationRequested,
                            state.queued.len(),
                            None,
                            None,
                        );
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            AdmissionLifecycleState::Cancelled,
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    self.service.inner.notify_admission_available();
                    removed
                },
                Err(error) => {
                    self.queued = false;
                    return Err(self
                        .service
                        .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
                },
            };
            finish_removed_request_cancelled(removed);
            return Err(error);
        }

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = match self.service.inner.admission.lock() {
            Ok(state) => state,
            Err(error) => {
                self.queued = false;
                return Err(self
                    .service
                    .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
            },
        };
        let sweep = self.service.sweep_expired_queued_requests_locked(&mut state);
        if !sweep.is_empty() {
            drop(state);
            sweep.finish();
            state = match self.service.inner.admission.lock() {
                Ok(state) => state,
                Err(error) => {
                    self.queued = false;
                    return Err(self
                        .service
                        .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
                },
            };
        }
        // This invocation may have swept its own cancellation, expiry, or
        // health-rejection from the queue. The terminal result is recorded
        // before the sweep drops its completion payload, so consume it before
        // falling through to placement and returning a misleading `Ok(None)`.
        if let Some(error) = self.terminal.take() {
            self.queued = false;
            drop(state);
            return Err(error);
        }
        if let Some(error) = self.service.admission_terminal_rejection(&state) {
            let removed =
                self.service.remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
            self.queued = false;
            self.service.inner.notify_admission_available();
            if removed.is_some() {
                record_admission_lifecycle(
                    &self.observation,
                    self.admission_request_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
            }
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }
        let selected = match self.service.select_next_runnable_locked(&mut state) {
            Ok(selected) => match selected
                .filter(|(position, _)| state.queued[*position].admission_request_id == self.admission_request_id)
            {
                Some(selected) => Ok(Some(selected)),
                None => self.service.fallback_driver_selection_locked(&state, self.admission_request_id),
            },
            Err(error) => Err(error),
        };
        let selected = match selected {
            Ok(selected) => selected,
            Err(error) => {
                let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                self.queued = false;
                if removed.is_some() {
                    record_admission_lifecycle(
                        &self.observation,
                        self.admission_request_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            },
        };
        if let Some((position, device_ordinal)) = selected {
            let selected_entry = match self.device_entries.get(&device_ordinal) {
                Some(entry) => entry,
                None => {
                    let error = missing_device_admission_entry_error(device_ordinal);
                    let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            admission_rejection(&error),
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                },
            };
            let placed_query = match self.service.try_place_on_device(
                device_ordinal,
                self.admission_request_id,
                state.queued[position].attempt_id,
                state.queued[position].requirement_for_device(device_ordinal)?,
                &mut slot_notifications,
            ) {
                Ok(placed_query) => placed_query,
                Err(error) => {
                    let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            admission_rejection(&error),
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                },
            };
            let Some(placed_query) = placed_query else {
                if Instant::now() >= self.deadline {
                    let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            AdmissionLifecycleState::TimedOut,
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    let error =
                        self.service.memory_grant_wait_timeout_error(&selected_entry.requirement, state.queued.len());
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                }
                return Ok(None);
            };
            let admitted = state.remove_queued_at(position).expect("admissible request exists");
            let admitted_submission_seq = admitted.submission_seq;
            let admission_wait_fact =
                admitted.admission_wait_fact(self.service.inner.queue_bounds.max_overtakes_per_waiter);
            if let Some(error) = self.service.admission_terminal_rejection(&state) {
                record_admission_lifecycle(
                    &admitted.observation,
                    self.admission_request_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.queued = false;
                self.service.inner.notify_admission_available();
                drop(state);
                drop(placed_query);
                finish_removed_request_error(Some(admitted), &error);
                return Err(error);
            }
            let lifecycle_ticket = state.lifecycle.begin_provisional(Arc::downgrade(&self.service.inner));
            self.queued = false;
            self.service.inner.notify_admission_available();
            drop(state);
            return self
                .service
                .commit_placed_admission(super::service::PlacedAdmissionCommit {
                    placed_query,
                    admitted_device_requirements: admitted
                        .device_entries
                        .iter()
                        .map(|(&ordinal, entry)| (ordinal, entry.requirement.clone()))
                        .collect(),
                    submitted_plan_capability_evidence: admitted.submitted_plan_capability_evidence.clone(),
                    admitted_reservation_envelope: admitted
                        .admitted_reservation_envelope_for_device(device_ordinal)?
                        .clone(),
                    admitted_compatible_devices: admitted.compatible_devices.clone(),
                    observation: admitted.observation,
                    attempt_id: admitted.attempt_id,
                    completion: admitted.completion,
                    admission_request_id: self.admission_request_id,
                    admission_wait: self.requested_at.elapsed(),
                    deadline: self.deadline,
                    execution_cancellation: admitted.execution_cancellation,
                    admitted_submission_seq: Some(admitted_submission_seq),
                    admission_wait_fact: Some(admission_wait_fact),
                    lifecycle_ticket,
                })
                .map(Some);
        }

        if Instant::now() >= self.deadline {
            let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
            self.queued = false;
            if removed.is_some() {
                record_admission_lifecycle(
                    &self.observation,
                    self.admission_request_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
            let error = service_overloaded_error(
                "queue_wait_elapsed",
                state.queued.len(),
                self.service.inner.queue_bounds.max_waiters,
            );
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }

        Ok(None)
    }

    /// Remaining time before this queued admission reaches the service timeout.
    pub fn remaining_wait(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

impl Drop for QueryAdmissionWaiter {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        let removed = match self.service.inner.admission.lock() {
            Ok(mut state) => {
                let removed =
                    self.service.remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
                self.service.inner.notify_admission_available();
                if removed.is_some() {
                    record_admission_lifecycle(
                        &self.observation,
                        self.admission_request_id,
                        AdmissionLifecycleState::CancellationRequested,
                        state.queued.len(),
                        None,
                        None,
                    );
                    record_admission_lifecycle(
                        &self.observation,
                        self.admission_request_id,
                        AdmissionLifecycleState::Cancelled,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                removed
            },
            Err(error) => {
                self.queued = false;
                let error = self
                    .service
                    .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id);
                eprintln!(
                    "[nexus-query-engine] warning: recovered queued query service admission request after {error}"
                );
                return;
            },
        };
        finish_removed_request_cancelled(removed);
    }
}

pub(super) fn remove_queued_request(
    state: &mut AdmissionState,
    admission_request_id: AdmissionRequestId,
) -> Option<AdmissionRequest> {
    if let Some(position) = state.queued.iter().position(|request| request.admission_request_id == admission_request_id)
    {
        state.remove_queued_at(position)
    } else {
        None
    }
}

pub(super) fn finish_removed_request_error(request: Option<AdmissionRequest>, error: &QueryError) {
    if let Some(request) = request {
        request.completion.finish_error(error);
    }
}

pub(super) fn admission_cancellation_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "nexus_query_engine.query_service.admission",
        "query admission was cancelled before a complete envelope was published",
    )
    .with_fact("reason", "admission_cancelled")
    .with_fact("cancellation_requested", true)
}

/// Preserves an attempt-wide deadline's typed cancellation reason when a
/// queue boundary observes an effective cancellation. Explicit caller
/// cancellation retains the stable admission-specific reason.
pub(super) fn admission_cancellation_or_deadline_error(cancellation: &ExecutionCancellation) -> QueryError {
    match cancellation.check("nexus_query_engine.query_service.admission") {
        Err(error) if error.fact_value("reason") == Some("deadline_expired") => error,
        _ => admission_cancellation_error(),
    }
}

pub(super) fn finish_removed_request_cancelled(request: Option<AdmissionRequest>) {
    if let Some(request) = request {
        request.completion.finish_cancelled();
    }
}

pub(super) fn finish_expired_queued_requests(expired: Vec<(AdmissionRequest, RuntimeTerminalOutcome)>) {
    for (request, disposition) in expired {
        let primary_failure = request.terminal.take_primary_failure();
        request.completion.finish_terminal_disposition(disposition, primary_failure);
    }
}

pub(super) fn finish_closed_queued_requests(closed: Vec<AdmissionRequest>) {
    for request in closed {
        let error = service_closed_error();
        record_admission_lifecycle(
            &request.observation,
            request.admission_request_id,
            AdmissionLifecycleState::Rejected { reason: "admission_closed".to_string() },
            0,
            None,
            None,
        );
        finish_removed_request_error(Some(request), &error);
    }
}

pub(super) fn service_overloaded_error(reason: &'static str, queued_queries: usize, max_waiters: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.query_service.admission",
        "query service admission is overloaded",
    )
    .with_fact("reason", reason)
    .with_fact("queued_queries", queued_queries)
    .with_fact("max_waiters", max_waiters)
}

pub(super) fn service_closed_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "nexus_query_engine.query_service.admission",
        "query service admission is closed",
    )
    .with_fact("reason", "admission_closed")
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use super::*;
    use crate::{
        attempt::{AttemptBackendCounters, AttemptOwner, BeginAttemptRequest},
        observability::{
            AttemptFailureCode, AttemptFailureFactKey, AttemptFailureFactValue, AttemptFailureOperation,
            RuntimeObservationConfig,
        },
        runtime::BackendLifecycle,
    };

    #[test]
    fn dropping_provisional_admission_ticket_restores_quiescence() {
        let tracker = AdmissionLifecycleTracker::default();
        let ticket = tracker.begin_provisional(Weak::new());

        assert_eq!(tracker.snapshot().provisional, 1);
        drop(ticket);

        assert!(tracker.is_quiescent());
    }

    #[test]
    fn dropping_active_admission_ticket_restores_quiescence() {
        let tracker = AdmissionLifecycleTracker::default();
        let mut ticket = tracker.begin_provisional(Weak::new());
        ticket.transition_to_active().expect("provisional ticket activates");

        assert_eq!(tracker.snapshot().active, 1);
        drop(ticket);

        assert!(tracker.is_quiescent());
    }

    #[test]
    fn queued_admission_terminal_preserves_borrowed_primary_failure_after_waiter_consumes_error() {
        let backend_lifecycle = BackendLifecycle::new();
        let attempt = AttemptOwner::begin(
            BeginAttemptRequest {
                observation: RuntimeObservationConfig::bounded(8, 8 * 1024),
                ..BeginAttemptRequest::default()
            },
            backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
            Arc::new(AttemptBackendCounters::default()),
            None,
        );
        let owner = attempt.owner;
        let completion = attempt.completion;
        let (observation_owner, finalizer_slot) = owner.admission_binding().into_parts();
        let query_completion =
            QueryCompletion::new(Some(observation_owner), Some(finalizer_slot), None, None, None, None);
        let error = QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.query_service.admission.test",
            "admission failure with retained primary evidence",
        )
        .with_fact("reason", "backend_closing");

        let queued_terminal = QueuedAdmissionTerminal::default();
        let disposition = queued_terminal.record(error);
        let waiter_error = queued_terminal.take().expect("queue waiter receives its full terminal error first");

        // The waiter keeps the full error, including the original source
        // chain. Queue maintenance then consumes only the independently
        // captured bounded summary for the attempt completion.
        assert_eq!(waiter_error.fact_value("reason"), Some("backend_closing"));
        query_completion.finish_terminal_disposition(disposition, queued_terminal.take_primary_failure());

        let report = completion.wait_timeout(Duration::from_secs(1)).expect("attempt report publishes");
        let summary = report.primary_failure.as_ref().expect("borrowed primary failure is retained");
        assert_eq!(summary.code(), AttemptFailureCode::MemoryReservationFailed);
        assert_eq!(summary.operation(), AttemptFailureOperation::Admission);
        assert_eq!(
            summary.facts().iter().map(|fact| (fact.key(), fact.value())).collect::<Vec<_>>(),
            vec![(AttemptFailureFactKey::Reason, AttemptFailureFactValue::BackendClosing)]
        );
    }
}
