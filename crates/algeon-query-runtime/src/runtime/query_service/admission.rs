//! Queued admission: the request, its placement step, and the terminal
//! completion payload every outcome funnels through.
//!
//! A request enters the controller's queue as an `AdmissionRequest` and is
//! driven by its owner's `AdmissionWaiter`. Each `try_place` is one step under
//! the admission mutex that either places the request on a device (a ledger
//! grant plus a provisional lifecycle ticket, returned as `PlacedAdmission`
//! for the service to build an envelope around; `service/transaction.rs`
//! defines the envelope), removes it with a terminal
//! error, or leaves it queued. Placement ends only queue membership; the
//! attempt's grant is held until its terminal commit, as
//! `website/docs/design/admission.mdx` specifies.
//!
//! Three lock rules hold throughout. The admission mutex
//! (`AdmissionController::state`) protects the queue, the pending-preflight
//! set, the closed flag, and the queue high-water mark. The lifecycle tracker
//! has its own mutex and is taken inside the admission mutex when both are
//! needed, never the other way round. No `MemoryGrant` is dropped and no
//! `QueryCompletion` is finished while the admission mutex is held, because
//! the grant's release observer and the completion's observers both take it.
//!
//! `AdmissionLifecycleTracker` is the drain fence: a ticket exists from the
//! moment a grant is provisional until terminal cleanup has released every
//! admission resource, and shutdown is quiescent only when no ticket remains.
//! `QueryCompletion` is the one owner of the backend permits and the attempt
//! finalizer for a request, from backend admission through the sealed report.

#[cfg(any(test, feature = "test-hooks"))]
use std::pin::pin;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    mem,
    sync::{Arc, Mutex, PoisonError, Weak},
    time::{Duration, Instant},
};

use query_planning::{
    admission::AttemptCapabilityEvidence,
    error::{QueryError, QueryErrorCode, Result},
};
use tokio::time;

use super::{QueryActivityLease, admission_rejection, controller::AdmissionController, record_admission_lifecycle};
#[cfg(any(test, feature = "test-hooks"))]
use super::{QueryHandle, QueryService};
#[cfg(any(test, feature = "test-hooks"))]
use crate::{
    attempt::AttemptOwner,
    runtime::{deadline::block_until, query_service::service::transaction::AdmissionTransactionProbe},
};
use crate::{
    attempt::{AdmittedGrantRelease, AttemptAdmission, AttemptHandle, AttemptLifecycleError, AttemptReportPublisher},
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, AttemptAdmissionFact, AttemptAdmissionWaitFact,
        AttemptFailureOperation, AttemptFailureSummary, AttemptId, AttemptTerminalCommit, QueryCleanupSealDelivery,
        RuntimeObservationHandle, RuntimeTerminalOutcome,
    },
    runtime::{
        ExecutionCancellation, MemoryGrant,
        backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
    },
};

/// One request's membership in the admission queue.
///
/// Placing the request ends only its own membership; dropping this value
/// before placement removes the queued request. It owns no CUDA object.
#[derive(Debug)]
pub(crate) struct AdmissionWaiter {
    pub(super) controller: Arc<AdmissionController>,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) requested_at: Instant,
    /// Queue deadline: the earlier of `QueueBounds::max_wait` after submission
    /// and the caller's own deadline. It bounds only the wait for placement;
    /// the envelope build that follows is bounded by `execution_cancellation`.
    pub(super) deadline: Option<Instant>,
    pub(super) execution_cancellation: ExecutionCancellation,
    /// Shared with the queued request so a sweep that removes it on another
    /// thread can leave this waiter its exact terminal error.
    pub(super) terminal: QueuedAdmissionTerminal,
    /// True while this waiter is responsible for a request in the queue.
    /// Cleared once, on the step that places or removes the request, after
    /// which `Drop` has nothing to remove.
    pub(super) queued: bool,
}

/// A queued admission on a query service: placement through the service's
/// admission controller, then the envelope commit that publishes a handle.
///
/// Dropping this value before admission removes the queued request.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug)]
pub(crate) struct QueryAdmissionWaiter {
    pub(super) service: QueryService,
    pub(super) waiter: AdmissionWaiter,
    /// The owner of a test attempt, handed to the handle this waiter admits.
    pub(super) test_attempt: Option<AttemptOwner>,
    pub(super) probe: Option<Arc<AdmissionTransactionProbe>>,
}

/// Lifecycle phase of one admitted grant, from placement to the end of its
/// terminal cleanup.
///
/// Transitions are `Provisional -> Active -> ActiveCleanup` for a published
/// handle and `Provisional -> ProvisionalCleanup` for a rollback; each cleanup
/// phase ends when the ticket drops. Any other transition is an invariant
/// error, which is how a double release or a drop before cleanup surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AdmissionTicketPhase {
    /// A ledger grant exists but no handle has been published for it.
    Provisional,
    /// The envelope validated and the handle is published.
    Active,
    /// Cleanup for an unpublished envelope. Its grant has never represented
    /// an admitted handle, so public admission snapshots must not count it.
    ProvisionalCleanup,
    /// Cleanup for a handle that was fully published. Its immutable grant is
    /// still held until the report commit completes, so it remains active in
    /// public admission snapshots.
    ActiveCleanup,
}

/// Ticket counts by phase. `cleanup` counts both cleanup phases;
/// `published_cleanup` counts only tickets whose handle was published, which
/// public admission snapshots still report as active.
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

/// A submission's place in the queue before its request exists.
///
/// Submission resolves device entries outside the admission mutex. Without
/// this reservation a younger submission could be queued, selected, and
/// placed during that window. The reservation records the submission sequence
/// in `AdmissionState::pending_preflight`, counts against `max_waiters`, and
/// acts as a selection barrier: no queued request with a later sequence is
/// selected while an earlier reservation is pending. It is converted into the
/// queued request under the mutex, or released if submission fails.
#[derive(Debug)]
pub(super) struct PendingPreflightReservation {
    controller: Weak<AdmissionController>,
    submission_seq: u64,
    /// False once converted or released; `Drop` then does nothing.
    armed: bool,
}

impl PendingPreflightReservation {
    pub(super) fn new(controller: Weak<AdmissionController>, submission_seq: u64) -> Self {
        Self { controller, submission_seq, armed: true }
    }

    pub(super) const fn submission_seq(&self) -> u64 {
        self.submission_seq
    }

    pub(super) fn release(mut self) {
        self.release_if_armed();
    }

    /// Replaces the reservation with the queued request. The caller holds the
    /// admission mutex and inserts the request in the same critical section,
    /// so the barrier lifts and the request appears atomically to selectors.
    pub(super) fn convert_locked(mut self, state: &mut AdmissionState) {
        let removed = state.pending_preflight.remove(&self.submission_seq);
        self.armed = false;
        assert!(removed, "pending preflight sequence must be reserved before conversion");
    }

    fn release_if_armed(&mut self) {
        if !self.armed {
            return;
        }
        let Some(controller) = self.controller.upgrade() else {
            self.armed = false;
            return;
        };
        let removed = match controller.state.lock() {
            Ok(mut state) => state.pending_preflight.remove(&self.submission_seq),
            Err(error) => error.into_inner().pending_preflight.remove(&self.submission_seq),
        };
        self.armed = false;
        // Lifting the barrier can make a younger queued request selectable,
        // and shutdown waits for the pending set to empty.
        if removed {
            controller.notify_admission_available();
        }
    }
}

impl Drop for PendingPreflightReservation {
    fn drop(&mut self) {
        self.release_if_armed();
    }
}

impl AdmissionLifecycleTracker {
    /// Mints a provisional ticket for a grant the caller placed. The
    /// caller holds the admission mutex, which fixes the lock order admission
    /// mutex, then tracker mutex, for the one place both are nested.
    pub(super) fn begin_provisional(&self, controller: Weak<AdmissionController>) -> AdmissionTicket {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let id = state.next_ticket_id;
        state.next_ticket_id = state.next_ticket_id.checked_add(1).expect("admission lifecycle ticket id overflow");
        let previous = state.tickets.insert(id, AdmissionTicketPhase::Provisional);
        debug_assert!(previous.is_none(), "admission lifecycle ticket ids must be unique");
        AdmissionTicket { tracker: self.clone(), controller, id, phase: AdmissionTicketPhase::Provisional }
    }

    fn transition(&self, id: u64, expected: AdmissionTicketPhase, next: AdmissionTicketPhase) -> Result<()> {
        // Each phase change is one map write, so a recovered guard holds
        // complete phases; the checks below still reject a conflicting one.
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(current) = state.tickets.get_mut(&id) else {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket is missing",
            )
            .with_fact("reason", "lifecycle_ticket_missing")
            .with_fact("ticket_id", id));
        };
        if *current != expected {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.admission.lifecycle",
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
        // Each phase change is one map write, so a recovered guard holds
        // complete phases; the checks below still reject a conflicting one.
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match state.tickets.get(&id).copied() {
            Some(actual) if actual.is_cleanup() => {
                state.tickets.remove(&id);
                Ok(())
            },
            Some(actual) => Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket dropped before cleanup completed",
            )
            .with_fact("reason", "lifecycle_drop_conflict")
            .with_fact("ticket_id", id)
            .with_fact("expected_phase", "cleanup")
            .with_fact("actual_phase", format!("{actual:?}"))),
            None => Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.admission.lifecycle",
                "admission lifecycle ticket was already dropped",
            )
            .with_fact("reason", "lifecycle_ticket_missing")
            .with_fact("ticket_id", id)),
        }
    }

    /// Phase counts at one instant. Taken under the tracker mutex only, so a
    /// caller that also holds the admission mutex sees tickets whose grant
    /// placement it has already observed.
    pub(super) fn snapshot(&self) -> AdmissionLifecycleSnapshot {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
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

    /// Whether no admitted grant, published or not, still exists anywhere
    /// between placement and the end of its terminal cleanup.
    pub(super) fn is_quiescent(&self) -> bool {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).tickets.is_empty()
    }
}

/// One linear lifecycle token.  It is intentionally non-cloneable: ownership
/// moves from admission transaction to handle and finally to cleanup.
///
/// The ticket is the last stage of the envelope's terminal teardown, so its
/// drop is the point at which an attempt's admission resources are all gone.
/// `Drop` records that fact in the tracker and wakes shutdown waiters.
#[derive(Debug)]
pub(super) struct AdmissionTicket {
    tracker: AdmissionLifecycleTracker,
    controller: Weak<AdmissionController>,
    id: u64,
    /// Local mirror of the tracker's phase, kept so a transition can be
    /// chosen without taking the tracker mutex first.
    phase: AdmissionTicketPhase,
}

impl AdmissionTicket {
    pub(super) fn is_cleanup(&self) -> bool {
        self.phase.is_cleanup()
    }

    /// Publishes the admission once the envelope validated and the handle
    /// exists. Fails if the ticket is not provisional, which no caller should
    /// be able to arrange.
    pub(super) fn transition_to_active(&mut self) -> Result<()> {
        self.tracker.transition(self.id, AdmissionTicketPhase::Provisional, AdmissionTicketPhase::Active)?;
        self.phase = AdmissionTicketPhase::Active;
        Ok(())
    }

    /// Enters the cleanup phase matching the current one. Idempotent, because
    /// rollback and terminal release can both reach it on the same ticket.
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
                if let Some(controller) = self.controller.upgrade() {
                    // The tracker has its own mutex, so removing the ticket does not touch the admission mutex.
                    // Locking it before the wakeup keeps the notify from landing between a waiter's quiescence
                    // check and its wait. No ticket drops on a thread holding this mutex: creators release it
                    // before moving the ticket on, and envelope teardown drops the grant, whose release observer
                    // also locks it, before the ticket.
                    let _admission = controller.state.lock().unwrap_or_else(PoisonError::into_inner);
                    controller.notify_admission_available();
                }
            },
            Err(error) => {
                eprintln!("[query-engine] warning: admission lifecycle ticket drop invariant failed: {error}");
            },
        }
    }
}

/// Everything the admission mutex protects.
///
/// `lifecycle` carries its own mutex and is only stored here so a holder of
/// the admission mutex can mint tickets and read the drain fence in one
/// critical section.
#[derive(Debug, Default)]
pub(super) struct AdmissionState {
    pub(super) lifecycle: AdmissionLifecycleTracker,
    /// Ordered by `submission_seq`; `enqueue_request` keeps the order when a
    /// pending submission is queued after a younger one.
    pub(super) queued: VecDeque<AdmissionRequest>,
    /// Submission sequences reserved but not yet queued. The smallest one is
    /// the selection barrier; the set's size counts against `max_waiters`.
    pub(super) pending_preflight: BTreeSet<u64>,
    /// Largest queue depth observed by this service since construction.
    ///
    /// The admission mutex owns both the queue and this value, so snapshots
    /// cannot miss an enqueue that raced with a later drain.  It deliberately
    /// never decreases while requests leave the queue.
    queue_high_watermark: usize,
    /// The queue-level close flag, set under the mutex. The controller's
    /// atomic `admission_open` gate is the one close can set without waiting;
    /// commit paths check both.
    pub(super) closed: bool,
}

impl AdmissionState {
    /// Inserts a request and records the resulting queue high-water mark.
    pub(super) fn enqueue_request(&mut self, request: AdmissionRequest) {
        let position = self
            .queued
            .iter()
            .position(|queued| queued.submission_seq > request.submission_seq)
            .unwrap_or(self.queued.len());
        self.queued.insert(position, request);
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
        mem::take(&mut self.queued)
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
    primary_failure: Option<AttemptFailureSummary>,
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
        let primary_failure =
            RuntimeTerminalOutcome::primary_failure_from_query_error(&error, AttemptFailureOperation::Admission);
        let mut terminal = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if terminal.error.is_none() {
            terminal.primary_failure = primary_failure;
            terminal.error = Some(error);
        }
        disposition
    }

    pub(super) fn take(&self) -> Option<QueryError> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).error.take()
    }

    /// Transfers the borrowed-error summary captured by queue maintenance to
    /// the completion path. The waiter may already have taken its full error,
    /// so the stable report must not reconstruct a generic fallback.
    pub(super) fn take_primary_failure(&self) -> Option<AttemptFailureSummary> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).primary_failure.take()
    }
}

#[derive(Debug)]
pub(super) struct AdmissionRequest {
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    /// Engine-minted identity retained from request submission through the
    /// selected ledger grant and allocator-event correlation.
    pub(super) attempt_id: AttemptId,
    pub(super) completion: QueryCompletion,
    /// Optional plan proof submitted by planning.  This is distinct from the
    /// always-resolved placement requirement below: no proof means execution
    /// coverage validation remains unavailable, rather than asserting an
    /// empty proof.
    pub(super) submitted_plan_capability_evidence: Option<AttemptCapabilityEvidence>,
    /// Device-specific entries retained from submission
    /// through the execution context that consumes the selected grant.
    pub(super) device_entries: DeviceAdmissionEntries,
    /// Monotonic FIFO identity retained independently from admission order.
    pub(super) submission_seq: u64,
    /// Device ordinals established at submission; capability filtering belongs
    /// here, and each configured device is considered independently.
    pub(super) compatible_devices: Vec<usize>,
    /// Exact request cap, or `None` until the selected ledger resolves its
    /// configured default inside the admission transaction.
    pub(super) requested_query_cap_bytes: Option<u64>,
    pub(super) deadline: Option<Instant>,
    /// The attempt-wide token also used by transactional envelope acquisition.
    pub(super) execution_cancellation: ExecutionCancellation,
    /// Terminal result recorded by queue maintenance before it drops this
    /// request's completion payload outside the admission mutex.
    pub(super) terminal: QueuedAdmissionTerminal,
    /// The revalidation generation this request's device state was last
    /// checked at; `None` until its first check.
    pub(super) validated_generation: Option<u64>,
}

/// One immutable capability/executability requirement resolved for one
/// installed device, keyed by device ordinal. Placement must never derive a
/// second requirement after submission.
pub(super) type DeviceAdmissionEntries = BTreeMap<usize, AttemptCapabilityEvidence>;

impl AdmissionRequest {
    pub(super) fn requirement_for_device(&self, device_ordinal: usize) -> Result<&AttemptCapabilityEvidence> {
        self.device_entries.get(&device_ordinal).ok_or_else(|| missing_device_admission_entry_error(device_ordinal))
    }

    /// Whether this waiter still belongs to the queue at `now`: neither
    /// cancelled nor past its deadline. A non-member waits only for its own
    /// waiter or a sweep to remove it.
    pub(super) fn is_queue_member(&self, now: Instant) -> bool {
        !self.execution_cancellation.is_cancelled() && self.deadline.is_none_or(|deadline| deadline > now)
    }

    /// Captures FIFO queue identity when the waiter is selected.
    pub(super) fn admission_wait_fact(&self) -> AttemptAdmissionWaitFact {
        AttemptAdmissionWaitFact { submission_sequence: self.submission_seq }
    }
}

pub(super) fn missing_device_admission_entry_error(device_ordinal: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        "query_engine.query_service.admission",
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
    /// The attempt's admission claim. Consumed by the terminal commit on the
    /// admitted path, or by `finish_not_admitted` on every other path.
    attempt_admission: Option<AttemptAdmission>,
    // Outlives `attempt_admission`, which is consumed before the seal: it
    // seals the attempt's observation with the attempt report.
    attempt_report: Option<AttemptReportPublisher>,
    /// Holds the backend observation scope open until the report is sealed
    /// and delivered. Released before the activity lease runs.
    observation_permit: Option<ObservationPermit>,
    /// Opaque caller lease whose destructor may reenter backend close. It is
    /// always the last thing a completion releases.
    observation_activity: Option<QueryActivityLease>,
    completion_dispatcher: Option<ObservationCompletionDispatcher>,
    /// Retains the backend resource generation. Moved into the envelope at
    /// publication; on every other path it is the first thing released.
    gpu_resource_permit: Option<GpuResourcePermit>,
}

impl QueryCompletion {
    pub(super) fn new(
        attempt_admission: AttemptAdmission,
        observation_permit: Option<ObservationPermit>,
        observation_activity: Option<QueryActivityLease>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
        gpu_resource_permit: Option<GpuResourcePermit>,
    ) -> Self {
        Self {
            attempt_report: Some(attempt_admission.report_publisher()),
            attempt_admission: Some(attempt_admission),
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
    pub(super) fn into_admitted(mut self, envelope: &mut super::service::transaction::AdmissionEnvelope) -> Self {
        envelope.retain_gpu_resource_permit(self.gpu_resource_permit.take());
        self
    }

    /// Records the published grant on the claimed attempt, under the
    /// admission mutex.
    pub(super) fn record_attempt_admitted(
        &self,
        fact: AttemptAdmissionFact,
        wait: Option<AttemptAdmissionWaitFact>,
    ) -> std::result::Result<(), AttemptLifecycleError> {
        self.attempt_admission.as_ref().map_or(Ok(()), |admission| admission.record_admitted(fact, wait))
    }

    /// The attempt this completion finishes, until its report is sealed.
    pub(super) fn attempt(&self) -> Option<AttemptHandle> {
        self.attempt_report.as_ref().map(AttemptReportPublisher::attempt)
    }

    /// Ties the grant's release to the claimed attempt's terminal outcome.
    /// Returns whether the outcome is already terminal, so the caller
    /// requests the release itself.
    pub(super) fn bind_grant_release(&self, release: Arc<dyn AdmittedGrantRelease>) -> bool {
        self.attempt_admission.as_ref().is_some_and(|admission| admission.bind_grant_release(release))
    }

    /// Samples the query-relative attempt clock only when this completion
    /// owns the admitted attempt finalizer. A missing binding stays
    /// unavailable in stable resource evidence rather than receiving a
    /// fabricated wall-clock timestamp.
    pub(super) fn attempt_monotonic_ns(&self) -> Option<u64> {
        self.attempt_admission.as_ref().map(AttemptAdmission::monotonic_ns)
    }

    /// Marks the attempt's observation as entering cleanup. Called once when
    /// release begins so the report's cleanup timing starts before any GPU
    /// cleanup job is submitted.
    pub(super) fn begin_query_cleanup(&self) {
        if let Some(attempt_report) = &self.attempt_report {
            attempt_report.begin_cleanup();
        }
    }

    /// Finishes a request that never published a handle with `error` as its
    /// terminal cause. Runs the observation seal and activity drop, so it must
    /// not be called while the admission mutex is held.
    pub(super) fn finish_error(self, error: &QueryError) {
        // Retain the bounded engine-owned summary while the original error is
        // still borrowed. `finish` deliberately consumes only its closed
        // terminal projection, so doing this afterwards would lose operation,
        // allowlisted facts, and redacted text from an attempt report.
        if let Some(summary) =
            RuntimeTerminalOutcome::primary_failure_from_query_error(error, AttemptFailureOperation::Admission)
        {
            self.record_primary_failure(summary);
        }
        self.finish(RuntimeTerminalOutcome::from_query_error(error));
    }

    fn record_primary_failure(&self, summary: AttemptFailureSummary) {
        if let Some(admission) = &self.attempt_admission {
            admission.record_primary_failure(summary);
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
        primary_failure: Option<AttemptFailureSummary>,
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
        self.attempt_admission.as_ref().map_or(fallback.clone(), |admission| {
            match admission.terminalize_for_release(fallback.clone()) {
                Ok(outcome) => outcome,
                Err(error) => {
                    eprintln!(
                        "[query-engine] warning: failed to establish attempt terminal outcome before GPU cleanup: {error}"
                    );
                    fallback
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
        debug_assert!(self.attempt_admission.is_none(), "attempt admission must be consumed before report seal");
        let dispatcher = self.completion_dispatcher.take();
        let permit = self.observation_permit.take();
        let activity = self.observation_activity.take();
        // The seal publishes the report inline so capacity reuse always
        // follows report-ready. A seal panic already completed degraded
        // delivery inside the recorder; suppress it like the combined path.
        let delivery = self.attempt_report.take().and_then(|attempt_report| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| attempt_report.seal(outcome))).ok()
        });
        QueryCompletionActivityTail { activity, dispatcher, delivery, permit }
    }

    /// Consumes the linear post-cleanup finalizer before the report seal.
    pub(super) fn finalize_attempt_after_gpu_cleanup(
        &mut self,
        terminal_commit: AttemptTerminalCommit,
    ) -> std::result::Result<(), AttemptLifecycleError> {
        self.attempt_admission.take().map_or(Ok(()), |admission| admission.finish_admitted(terminal_commit))
    }

    fn finish(mut self, fallback: RuntimeTerminalOutcome) {
        // A rejected or cancelled request has no handle cleanup owner. Release
        // its backend generation before observer delivery so enqueue fallback
        // cannot synchronously reenter backend close while retaining GPU state.
        drop(self.gpu_resource_permit.take());
        self.dispatch_after_gpu_cleanup(fallback);
    }

    /// Seals and delivers on the observation executor when the completion
    /// has one, otherwise inline. The split keeps an observer that blocks or
    /// an activity destructor that reenters close off the thread that
    /// finished the request, which may be a pool worker or the admission
    /// caller.
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
        let permit = self.observation_permit.take();
        let activity = self.observation_activity.take();
        // `Some(outcome)` seals the attempt's observation with its report;
        // `None` suppresses publication.
        let attempt_outcome = match self.attempt_admission.take() {
            // This completion was already admitted but reached a recovery path
            // before its post-cleanup commit ran. Publishing would invent
            // admitted cleanup/capacity facts, so fail publication instead.
            Some(admission) if admission.is_admitted() => {
                eprintln!(
                    "[query-engine] warning: admitted completion dropped before finalization; suppressing report publication"
                );
                None
            },
            Some(admission) => match admission.finish_not_admitted(fallback) {
                Ok(outcome) => Some(outcome),
                Err(error) => {
                    eprintln!(
                        "[query-engine] warning: failed to finalize unadmitted attempt before seal; suppressing report publication: {error}"
                    );
                    None
                },
            },
            // The admitted path already committed its cleanup facts with the
            // admission; a recovery that reaches here before the seal
            // publishes them.
            None => Some(fallback),
        };
        let attempt_report = self.attempt_report.take();
        if let (Some(outcome), Some(attempt_report)) = (attempt_outcome, attempt_report) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| attempt_report.seal(outcome).deliver()));
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
    delivery: Option<QueryCleanupSealDelivery>,
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
    /// Recovery for a completion that was neither finished nor sealed. Every
    /// explicit path empties the fields first, so reaching the body below
    /// means an unwind or a dropped future abandoned the request.
    fn drop(&mut self) {
        if self.attempt_report.is_none()
            && self.attempt_admission.is_none()
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
            attempt_admission: self.attempt_admission.take(),
            attempt_report: self.attempt_report.take(),
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

/// A provisionally placed admission and every value its publication commits.
///
/// Holding a ledger grant and a provisional lifecycle ticket but no CUDA
/// object, it is what placement hands from the controller to the service's
/// envelope commit. Keeping these values together prevents the admission
/// paths from publishing different request metadata.
pub(in crate::runtime::query_service) struct PlacedAdmission {
    pub(in crate::runtime::query_service) placed_query: PlacedQuery,
    pub(in crate::runtime::query_service) admitted_device_requirements: BTreeMap<usize, AttemptCapabilityEvidence>,
    pub(in crate::runtime::query_service) submitted_plan_capability_evidence: Option<AttemptCapabilityEvidence>,
    pub(in crate::runtime::query_service) admitted_compatible_devices: Vec<usize>,
    pub(in crate::runtime::query_service) observation: RuntimeObservationHandle,
    pub(in crate::runtime::query_service) attempt_id: AttemptId,
    pub(in crate::runtime::query_service) completion: QueryCompletion,
    pub(in crate::runtime::query_service) admission_request_id: AdmissionRequestId,
    pub(in crate::runtime::query_service) admission_wait: Duration,
    pub(in crate::runtime::query_service) execution_cancellation: ExecutionCancellation,
    pub(in crate::runtime::query_service) admission_wait_fact: Option<AttemptAdmissionWaitFact>,
    pub(in crate::runtime::query_service) lifecycle_ticket: AdmissionTicket,
}

/// The ledger side of a placement: the device and its committed grant.
/// Dropping it before an envelope exists returns the grant through the
/// ledger's release observer, which takes the admission mutex.
#[derive(Debug)]
pub(super) struct PlacedQuery {
    pub(super) device_ordinal: usize,
    pub(super) memory_grant: MemoryGrant,
}

impl AdmissionWaiter {
    /// Places this queued admission if it is runnable, returning the commit
    /// that builds and publishes its envelope.
    ///
    /// One call is one step under the admission mutex. `Ok(None)` means the
    /// request is still queued: another waiter is older and runnable, or every
    /// compatible device is at capacity. Every `Err` has removed the request
    /// and finished its completion; a waiter that sees an error has nothing
    /// left to clean up. The removal and the completion are separated by a
    /// `drop(state)` on each path because the completion seals the attempt's
    /// observation, and its observers may take the admission mutex or reenter
    /// backend close.
    ///
    /// The ledger transaction in `try_place_on_device` can decline a device
    /// the queue selected, because selection ranks a snapshot and the ledger
    /// decides on current state. A declined placement at the queue deadline
    /// is reported as `memory_grant_wait_timeout` for that device instead of
    /// the plain `queue_wait_elapsed`, so a memory-bound wait keeps its own
    /// taxonomy.
    pub(crate) fn try_place(&mut self) -> Result<Option<PlacedAdmission>> {
        if let Some(error) = self.terminal.take() {
            self.queued = false;
            return Err(error);
        }
        if !self.queued {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.query_service.admission",
                "query admission waiter has already completed",
            )
            .with_fact("admission_request_id", self.admission_request_id)
            .with_fact("reason", "admission_waiter_completed"));
        }

        if self.execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&self.execution_cancellation);
            let removed = match self.controller.state.lock() {
                Ok(mut state) => {
                    let removed = self
                        .controller
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
                    self.controller.notify_admission_available();
                    removed
                },
                Err(error) => {
                    self.queued = false;
                    return Err(self
                        .controller
                        .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
                },
            };
            finish_removed_request_cancelled(removed);
            return Err(error);
        }

        let mut state = match self.controller.state.lock() {
            Ok(state) => state,
            Err(error) => {
                self.queued = false;
                return Err(self
                    .controller
                    .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
            },
        };
        // A step ends only this request. Other waiters' cancellation and
        // expiry are removed by their own steps or by submission sweeps;
        // until then they are not queue members and claim no device.
        // A sweep that ran since the check above may have recorded this
        // request's terminal result before dropping its completion payload.
        if let Some(error) = self.terminal.take() {
            self.queued = false;
            return Err(error);
        }
        if let Err(error) = self.controller.reconcile_native_admission_health() {
            eprintln!("[query-engine] warning: native admission-health reconciliation failed: {error}");
        }
        let device_state_rejection =
            match state.queued.iter_mut().find(|request| request.admission_request_id == self.admission_request_id) {
                Some(request) => self.controller.revalidate_queued_request_device_state(request),
                None => None,
            };
        if let Some(error) = device_state_rejection {
            let removed = self.controller.abandon_queued_request_locked(&mut state, self.admission_request_id);
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
            // Completion observers may take the admission mutex; release it first.
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }
        if let Some(error) = self.controller.admission_terminal_rejection(&state) {
            let removed =
                self.controller.remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
            self.queued = false;
            self.controller.notify_admission_available();
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
            // Completion observers may take the admission mutex; release it first.
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }
        let selected = match self.controller.select_next_runnable_locked(&mut state) {
            Ok(selected) => match selected
                .filter(|(position, _)| state.queued[*position].admission_request_id == self.admission_request_id)
            {
                Some(selected) => Ok(Some(selected)),
                None => self.controller.fallback_driver_selection_locked(&state, self.admission_request_id),
            },
            Err(error) => Err(error),
        };
        let selected = match selected {
            Ok(selected) => selected,
            Err(error) => {
                let removed = self.controller.abandon_queued_request_locked(&mut state, self.admission_request_id);
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
                // Completion observers may take the admission mutex; release it first.
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            },
        };
        if let Some((position, device_ordinal)) = selected {
            let placed_query = match self.controller.try_place_on_device(
                device_ordinal,
                self.admission_request_id,
                state.queued[position].attempt_id,
                state.queued[position].requirement_for_device(device_ordinal)?,
                state.queued[position].requested_query_cap_bytes,
            ) {
                Ok(placed_query) => placed_query,
                Err(error) => {
                    let removed = self.controller.abandon_queued_request_locked(&mut state, self.admission_request_id);
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
                    // Completion observers may take the admission mutex; release it first.
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                },
            };
            let Some(placed_query) = placed_query else {
                if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    // Read the timed-out waiter's cap before abandoning it: the
                    // removal shifts every later entry, so `position` no longer
                    // names this request afterwards.
                    let requested_query_cap_bytes = state.queued[position].requested_query_cap_bytes;
                    let removed = self.controller.abandon_queued_request_locked(&mut state, self.admission_request_id);
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
                    let error = self.controller.memory_grant_wait_timeout_error(
                        device_ordinal,
                        requested_query_cap_bytes,
                        self.controller.queue_member_count_locked(&state),
                    );
                    // Completion observers may take the admission mutex; release it first.
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                }
                return Ok(None);
            };
            let admitted = state.remove_queued_at(position).expect("admissible request exists");
            let admission_wait_fact = admitted.admission_wait_fact();
            if let Some(error) = self.controller.admission_terminal_rejection(&state) {
                record_admission_lifecycle(
                    &admitted.observation,
                    self.admission_request_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.queued = false;
                self.controller.notify_admission_available();
                // The grant's release observer and the completion's observers both
                // take the admission mutex; neither may run under the guard.
                drop(state);
                drop(placed_query);
                finish_removed_request_error(Some(admitted), &error);
                return Err(error);
            }
            let lifecycle_ticket = state.lifecycle.begin_provisional(Arc::downgrade(&self.controller));
            self.queued = false;
            self.controller.notify_admission_available();
            drop(state);
            return Ok(Some(PlacedAdmission {
                placed_query,
                admitted_device_requirements: admitted
                    .device_entries
                    .iter()
                    .map(|(&ordinal, requirement)| (ordinal, requirement.clone()))
                    .collect(),
                submitted_plan_capability_evidence: admitted.submitted_plan_capability_evidence.clone(),
                admitted_compatible_devices: admitted.compatible_devices.clone(),
                observation: admitted.observation,
                attempt_id: admitted.attempt_id,
                completion: admitted.completion,
                admission_request_id: self.admission_request_id,
                admission_wait: self.requested_at.elapsed(),
                execution_cancellation: admitted.execution_cancellation,
                admission_wait_fact: Some(admission_wait_fact),
                lifecycle_ticket,
            }));
        }

        if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let removed = self.controller.abandon_queued_request_locked(&mut state, self.admission_request_id);
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
                self.controller.queue_member_count_locked(&state),
                self.controller.queue_bounds.max_waiters,
            );
            // Completion observers may take the admission mutex; release it first.
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }

        Ok(None)
    }

    /// Remaining time before this queued admission reaches its queue
    /// deadline, or `None` when the wait has no deadline.
    pub(crate) fn remaining_wait(&self) -> Option<Duration> {
        self.deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// Waits until this request is placed or rejected. A ledger transition,
    /// cancellation, and the queue deadline each end one wait.
    ///
    /// Dropping the returned future removes the request from the queue
    /// through the waiter's `Drop`; the caller's runtime is never blocked.
    pub(crate) async fn place_async(&mut self) -> Result<PlacedAdmission> {
        let controller = Arc::clone(&self.controller);
        loop {
            // `Notify::notify_waiters` wakes only futures already registered,
            // so the `Notified` is created and enabled before the step. A
            // wakeup that lands between the step and the await is then kept
            // instead of lost.
            let notified = controller.admission_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(placed) = self.try_place()? {
                return Ok(placed);
            }
            let wake = async {
                tokio::select! {
                    () = &mut notified => {},
                    () = self.execution_cancellation.cancelled_or_expired() => {},
                }
            };
            match self.remaining_wait() {
                // The deadline has passed; the next step reports the timeout
                // and removes the request rather than this loop deciding it.
                Some(remaining_wait) if remaining_wait.is_zero() => continue,
                Some(remaining_wait) => {
                    let _ = time::timeout(remaining_wait, wake).await;
                },
                None => wake.await,
            }
        }
    }

    /// Waits on the calling thread until this request is placed or rejected.
    ///
    /// Each pass is one [`Self::try_place`]; between passes the thread parks
    /// until an admission notification, cancellation, or the earlier of this
    /// request's and its cancellation's deadlines, the same events that end
    /// an async wait.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn place_blocking(&mut self) -> Result<PlacedAdmission> {
        let controller = Arc::clone(&self.controller);
        loop {
            let notified = controller.admission_notify.notified();
            let mut notified = pin!(notified);
            notified.as_mut().enable();
            if let Some(placed) = self.try_place()? {
                return Ok(placed);
            }
            let wake_deadline = match (self.execution_cancellation.deadline(), self.deadline) {
                (Some(attempt), Some(queue)) => Some(attempt.min(queue)),
                (attempt, queue) => attempt.or(queue),
            };
            let _ = block_until(
                async {
                    tokio::select! {
                        biased;
                        () = notified.as_mut() => {},
                        () = self.execution_cancellation.cancel_requested() => {},
                    }
                },
                wake_deadline,
            );
        }
    }
}

#[cfg(any(test, feature = "test-hooks"))]
impl QueryAdmissionWaiter {
    /// Attempts to complete this queued admission without blocking.
    ///
    /// Returns `Ok(None)` while the request is still queued behind active work
    /// or a compatible device's current capacity/health constraint. Queue
    /// timeout and service close remove the request and return a structured
    /// error.
    pub(crate) fn try_admit(&mut self) -> Result<Option<QueryHandle>> {
        let Some(placed) = self.waiter.try_place()? else {
            return Ok(None);
        };
        let handle = self.service.commit_placed_admission(placed, self.probe.clone())?;
        handle.own_test_attempt(self.test_attempt.take());
        Ok(Some(handle))
    }

    /// Completes this queued admission on the calling thread.
    pub(crate) fn admit_blocking(mut self) -> Result<QueryHandle> {
        let placed = self.waiter.place_blocking()?;
        let handle = self.service.commit_placed_admission(placed, self.probe.take())?;
        handle.own_test_attempt(self.test_attempt.take());
        Ok(handle)
    }
}

impl Drop for AdmissionWaiter {
    /// A waiter dropped while its request is still queued is the caller's
    /// cancellation: the request is removed and finished as `Cancelled`. A
    /// waiter whose step already placed or removed the request does nothing.
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        let removed = match self.controller.state.lock() {
            Ok(mut state) => {
                let removed = self
                    .controller
                    .remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
                self.controller.notify_admission_available();
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
                // The recovery records this request's rejection and finishes
                // its completion with the poison error.
                self.controller
                    .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id);
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

/// Finishes a request removed from the queue with `error`. Must run without
/// the admission mutex held; see `QueryCompletion::finish_error`.
pub(super) fn finish_removed_request_error(request: Option<AdmissionRequest>, error: &QueryError) {
    if let Some(request) = request {
        request.completion.finish_error(error);
    }
}

pub(super) fn admission_cancellation_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "query_engine.query_service.admission",
        "query admission was cancelled before a complete envelope was published",
    )
    .with_fact("reason", "admission_cancelled")
    .with_fact("cancellation_requested", true)
}

/// Preserves an attempt-wide deadline's typed cancellation reason when a
/// queue boundary observes an effective cancellation. Explicit caller
/// cancellation retains the stable admission-specific reason.
pub(super) fn admission_cancellation_or_deadline_error(cancellation: &ExecutionCancellation) -> QueryError {
    match cancellation.check("query_engine.query_service.admission") {
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

/// Finishes every request a service close took from the queue. The recorded
/// queue depth is zero because close empties the queue in one take.
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
        "query_engine.query_service.admission",
        "query service admission is overloaded",
    )
    .with_fact("reason", reason)
    .with_fact("queued_queries", queued_queries)
    .with_fact("max_waiters", max_waiters)
}

pub(super) fn admission_queue_full_error(
    queued_queries: usize,
    max_waiters: usize,
    pending_preflight: usize,
) -> QueryError {
    service_overloaded_error("admission_queue_full", queued_queries, max_waiters)
        .with_fact("pending_preflight", pending_preflight)
}

pub(super) fn service_closed_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "query_engine.query_service.admission",
        "query service admission is closed",
    )
    .with_fact("reason", "admission_closed")
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use query_planning::error::AttemptFailureCode;

    use super::*;
    use crate::{
        attempt::{AttemptBackendCounters, AttemptOwner, BeginAttemptRequest},
        observability::{
            AttemptDisposition, AttemptFailureFactKey, AttemptFailureFactValue, AttemptFailureOperation,
            RuntimeObservationConfig,
        },
        runtime::BackendLifecycle,
    };

    #[test]
    fn a_ticket_leaves_a_poisoned_lifecycle_tracker() {
        let tracker = AdmissionLifecycleTracker::default();
        let ticket = tracker.begin_provisional(Weak::new());
        let poisoner = tracker.clone();
        std::thread::spawn(move || {
            let _state = poisoner.state.lock().unwrap();
            panic!("poison the lifecycle tracker");
        })
        .join()
        .expect_err("the poisoning thread panics while holding the tracker");

        drop(ticket);

        assert!(tracker.is_quiescent(), "the dropped ticket leaves the lifecycle drain fence");
    }

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
        let completion = attempt.completion.completion();
        owner.handle().set_disposition(AttemptDisposition::Native).unwrap();
        let binding = owner.admission_binding();
        let admission = binding.claim().unwrap();
        let query_completion = QueryCompletion::new(admission, None, None, None, None);
        let error = QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "query_engine.query_service.admission.test",
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
        assert_eq!(summary.code(), AttemptFailureCode::Query(QueryErrorCode::MemoryReservationFailed));
        assert_eq!(summary.operation(), AttemptFailureOperation::Admission);
        assert_eq!(
            summary.facts().iter().map(|fact| (fact.key(), fact.value())).collect::<Vec<_>>(),
            vec![(AttemptFailureFactKey::Reason, AttemptFailureFactValue::BackendClosing)]
        );
    }
}
