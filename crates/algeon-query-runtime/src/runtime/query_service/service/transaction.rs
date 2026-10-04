//! Transactional admission envelope.
//!
//! One admitted attempt owns its ledger grant, attempt lane, root stream
//! (Root permit class), allocation domain with the root stream member, and
//! device worker scope as a single transaction. The envelope is built after
//! the ledger transaction commits (inside the admission mutex) but while the
//! admission mutex is released, so a bounded stream wait never blocks queue
//! maintenance or cleanup. Admission is published; the `Admitted` lifecycle
//! fact and the [`QueryHandle`]; only after every component validates.
//!
//! Rollback drops components in reverse acquisition order (member → domain →
//! stream → lane → grant), preserves the primary error, and restores every
//! admission counter exactly. Fault injection via
//! [`AdmissionTransactionProbe`] proves both properties at every step.

#[cfg(test)]
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    super::admission::{AdmissionTicket, PlacedQuery},
    QueryServiceInner,
};
use crate::{
    observability::{AttemptId, RuntimeObservationHandle, RuntimeStreamLeaseScope},
    runtime::{
        ExecutionCancellation, StreamLease, StreamPermitClass,
        allocator_correlation::{AllocatorDomainAttempt, discard_admitted_domain, register_admitted_domain},
        attempt_domain::AttemptAllocationDomain,
        backend_core::GpuResourcePermit,
        gpu::{DeviceLanePermit, DeviceWorkerScope, MemoryGrant},
        query_service::admission::AdmissionState,
    },
};

/// Ordered acquisition points of the admission transaction.
///
/// Fault injection fires before and after every acquisition so tests can
/// prove exact rollback at each point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AdmissionStep {
    BeforeLanePermit,
    AfterLanePermit,
    BeforeRootStream,
    AfterRootStream,
    BeforeAllocationDomain,
    AfterAllocationDomain,
    BeforeRootMember,
    AfterRootMember,
    BeforeWorkerScope,
    AfterWorkerScope,
}

impl AdmissionStep {
    #[cfg(test)]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::BeforeLanePermit => "before_lane_permit",
            Self::AfterLanePermit => "after_lane_permit",
            Self::BeforeRootStream => "before_root_stream",
            Self::AfterRootStream => "after_root_stream",
            Self::BeforeAllocationDomain => "before_allocation_domain",
            Self::AfterAllocationDomain => "after_allocation_domain",
            Self::BeforeRootMember => "before_root_member",
            Self::AfterRootMember => "after_root_member",
            Self::BeforeWorkerScope => "before_worker_scope",
            Self::AfterWorkerScope => "after_worker_scope",
        }
    }

    #[cfg(test)]
    pub(crate) const ALL: [AdmissionStep; 10] = [
        Self::BeforeLanePermit,
        Self::AfterLanePermit,
        Self::BeforeRootStream,
        Self::AfterRootStream,
        Self::BeforeAllocationDomain,
        Self::AfterAllocationDomain,
        Self::BeforeRootMember,
        Self::AfterRootMember,
        Self::BeforeWorkerScope,
        Self::AfterWorkerScope,
    ];
}

/// Fault-injection probe for the admission transaction.
///
/// The mechanism is test-only: production installs no probe, so the struct
/// carries no state there and every hook is an inlined no-op that takes no
/// locks on the admission path.
pub(crate) struct AdmissionTransactionProbe {
    #[cfg(test)]
    fail_at: Mutex<Option<AdmissionStep>>,
    #[cfg(test)]
    pause_at: Mutex<Option<(AdmissionStep, Arc<AdmissionTransactionPause>)>>,
    #[cfg(test)]
    cancel_before_publication: Mutex<bool>,
    #[cfg(test)]
    rollback_pause: Mutex<Option<Arc<AdmissionTransactionPause>>>,
    /// Name of the thread that ran the transaction's first probed step.
    #[cfg(test)]
    transaction_thread: Mutex<Option<String>>,
}

#[cfg(test)]
struct AdmissionTransactionPause {
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl std::fmt::Debug for AdmissionTransactionProbe {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("AdmissionTransactionProbe").finish_non_exhaustive()
    }
}

impl AdmissionTransactionProbe {
    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self {
            fail_at: Mutex::new(None),
            pause_at: Mutex::new(None),
            cancel_before_publication: Mutex::new(false),
            rollback_pause: Mutex::new(None),
            transaction_thread: Mutex::new(None),
        }
    }

    #[cfg(test)]
    pub(crate) fn failing_at(step: AdmissionStep) -> Self {
        Self {
            fail_at: Mutex::new(Some(step)),
            pause_at: Mutex::new(None),
            cancel_before_publication: Mutex::new(false),
            rollback_pause: Mutex::new(None),
            transaction_thread: Mutex::new(None),
        }
    }

    /// Pauses a transaction after it acquires the resources preceding `step`.
    /// The bounded entry notification tells the test that the provisional
    /// grant exists; the release channel lets it resume without publishing it
    /// early.
    #[cfg(test)]
    pub(crate) fn pausing_at(step: AdmissionStep) -> (Self, super::TestPauseHandle) {
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (release_sender, release_receiver) = mpsc::sync_channel(1);
        let pause =
            Arc::new(AdmissionTransactionPause { entered: entered_sender, release: Mutex::new(release_receiver) });
        (
            Self {
                fail_at: Mutex::new(None),
                pause_at: Mutex::new(Some((step, pause))),
                cancel_before_publication: Mutex::new(false),
                rollback_pause: Mutex::new(None),
                transaction_thread: Mutex::new(None),
            },
            super::TestPauseHandle::new("admission transaction", entered_receiver, release_sender),
        )
    }

    /// Pauses after an unpublished envelope has atomically transferred its
    /// lifecycle ticket into rollback cleanup, but before its drop starts
    /// physical cleanup. This exposes the exact shutdown-quiescence boundary.
    #[cfg(test)]
    pub(crate) fn pausing_after_rollback_accounting() -> (Self, super::TestPauseHandle) {
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (release_sender, release_receiver) = mpsc::sync_channel(1);
        let pause =
            Arc::new(AdmissionTransactionPause { entered: entered_sender, release: Mutex::new(release_receiver) });
        (
            Self {
                fail_at: Mutex::new(None),
                pause_at: Mutex::new(None),
                cancel_before_publication: Mutex::new(true),
                rollback_pause: Mutex::new(Some(pause)),
                transaction_thread: Mutex::new(None),
            },
            super::TestPauseHandle::new("admission rollback cleanup", entered_receiver, release_sender),
        )
    }

    /// Name of the thread that ran the transaction's first probed step.
    #[cfg(test)]
    pub(crate) fn transaction_thread(&self) -> Option<String> {
        self.transaction_thread.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    #[cfg(test)]
    fn forces_cancellation_before_publication(&self) -> bool {
        *self.cancel_before_publication.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn forces_cancellation_before_publication(&self) -> bool {
        false
    }

    #[cfg(test)]
    fn pause_after_rollback_accounting(&self) -> Result<()> {
        let pause = self
            .rollback_pause
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.admission_transaction",
                    "admission rollback pause-probe mutex was poisoned",
                )
            })?
            .as_ref()
            .map(Arc::clone);
        if let Some(pause) = pause {
            pause.entered.send(()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.admission_transaction",
                    "admission rollback pause observer disconnected",
                )
            })?;
            pause
                .release
                .lock()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "query_engine.query_service.admission_transaction",
                        "admission rollback pause release mutex was poisoned",
                    )
                })?
                .recv()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "query_engine.query_service.admission_transaction",
                        "admission rollback pause releaser disconnected",
                    )
                })?;
        }
        Ok(())
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn pause_after_rollback_accounting(&self) -> Result<()> {
        Ok(())
    }

    #[cfg(test)]
    fn check(&self, step: AdmissionStep) -> Result<()> {
        self.transaction_thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(|| std::thread::current().name().unwrap_or_default().to_string());
        let fail_at = self.fail_at.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.admission_transaction",
                "admission transaction probe mutex was poisoned",
            )
        })?;
        if *fail_at == Some(step) {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.admission_transaction",
                "admission transaction step failed by injected fault",
            )
            .with_fact("reason", "injected_admission_step_fault")
            .with_fact("admission_step", step.as_str()));
        }
        // Release the failure-probe lock before acquiring the pause-probe lock.
        drop(fail_at);
        if let Some(pause) = self
            .pause_at
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.admission_transaction",
                    "admission transaction pause-probe mutex was poisoned",
                )
            })?
            .as_ref()
            .filter(|(paused_step, _)| *paused_step == step)
            .map(|(_, pause)| Arc::clone(pause))
        {
            pause.entered.send(()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.admission_transaction",
                    "admission transaction pause observer disconnected",
                )
                .with_fact("admission_step", step.as_str())
            })?;
            pause
                .release
                .lock()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "query_engine.query_service.admission_transaction",
                        "admission transaction pause release mutex was poisoned",
                    )
                    .with_fact("admission_step", step.as_str())
                })?
                .recv()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "query_engine.query_service.admission_transaction",
                        "admission transaction pause releaser disconnected",
                    )
                    .with_fact("admission_step", step.as_str())
                })?;
        }
        Ok(())
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn check(&self, _step: AdmissionStep) -> Result<()> {
        Ok(())
    }
}

/// The root stream binding consumed by the first context acquisition.
pub(crate) struct RootStreamBinding {
    pub(crate) stream_lease: StreamLease,
    pub(crate) member_lease: cudf::StreamReservationLease,
}

/// GPU-bound components of an admitted attempt. Fields are declared in
/// reverse acquisition order so dropping releases member → domain →
/// stream/warm reservation.
struct EnvelopePhysical {
    worker_scope: DeviceWorkerScope,
    root_member: Mutex<Option<cudf::StreamReservationLease>>,
    domain: AttemptAllocationDomain,
    root_stream: Mutex<Option<StreamLease>>,
}

/// One stage in a typed teardown sequence.
///
/// Each stage owns exactly one resource and the only valid continuation after
/// that resource is released. Its destructor explicitly releases the current
/// stage before Rust can descend into `then`, so changing field declaration
/// order cannot weaken the sequence.
struct TeardownStep<T, Next> {
    release: Option<T>,
    then: Next,
}

impl<T, Next> Drop for TeardownStep<T, Next> {
    fn drop(&mut self) {
        // This release must precede descent into `then`: admission capacity
        // is reusable only after every earlier stage has completed.
        drop(self.release.take());
    }
}

type AdmissionTerminalSequence = TeardownStep<
    DeviceLanePermit,
    TeardownStep<MemoryGrant, TeardownStep<GpuResourcePermit, TeardownStep<AdmissionTicket, ()>>>,
>;

/// Terminal admission resources encoded as a release chain rather than
/// independent envelope fields. The type fixes lane → grant → backend permit
/// → lifecycle-ticket destruction regardless of enclosing field order.
struct AdmissionTerminalTeardown {
    sequence: AdmissionTerminalSequence,
}

impl AdmissionTerminalTeardown {
    fn new(
        lane_permit: Option<DeviceLanePermit>,
        memory_grant: MemoryGrant,
        gpu_resource_permit: Option<GpuResourcePermit>,
        lifecycle_ticket: Option<AdmissionTicket>,
    ) -> Self {
        Self {
            sequence: TeardownStep {
                release: lane_permit,
                then: TeardownStep {
                    release: Some(memory_grant),
                    then: TeardownStep {
                        release: gpu_resource_permit,
                        then: TeardownStep { release: lifecycle_ticket, then: () },
                    },
                },
            },
        }
    }

    fn memory_grant(&self) -> &MemoryGrant {
        self.sequence.then.release.as_ref().expect("admission teardown retains an immutable memory grant")
    }

    fn lane_index(&self) -> Option<usize> {
        self.sequence.release.as_ref().map(DeviceLanePermit::index)
    }

    fn lifecycle_ticket_mut(&mut self) -> Option<&mut AdmissionTicket> {
        self.sequence.then.then.then.release.as_mut()
    }

    fn retain_gpu_resource_permit(&mut self, gpu_resource_permit: Option<GpuResourcePermit>) {
        let slot = &mut self.sequence.then.then.release;
        debug_assert!(slot.is_none(), "published admission envelope can retain only one GPU resource permit");
        *slot = gpu_resource_permit;
    }
}

/// GPU-bound and terminal admission resources. Physical resources are released
/// before the typed terminal sequence; a rollback can move the terminal
/// sequence intact behind a stream cleanup fence.
struct AdmissionTeardown {
    physical: Option<EnvelopePhysical>,
    terminal: Option<AdmissionTerminalTeardown>,
}

impl AdmissionTeardown {
    fn new(
        physical: EnvelopePhysical,
        lane_permit: DeviceLanePermit,
        memory_grant: MemoryGrant,
        lifecycle_ticket: Option<AdmissionTicket>,
    ) -> Self {
        Self {
            physical: Some(physical),
            terminal: Some(AdmissionTerminalTeardown::new(Some(lane_permit), memory_grant, None, lifecycle_ticket)),
        }
    }

    fn terminal(&self) -> &AdmissionTerminalTeardown {
        self.terminal.as_ref().expect("admission envelope terminal teardown is present")
    }

    fn terminal_mut(&mut self) -> &mut AdmissionTerminalTeardown {
        self.terminal.as_mut().expect("admission envelope terminal teardown is present")
    }

    fn take_terminal(&mut self) -> AdmissionTerminalTeardown {
        self.terminal.take().expect("admission envelope terminal teardown is present")
    }
}

impl Drop for AdmissionTeardown {
    fn drop(&mut self) {
        // Physical CUDA state must disappear before the terminal release
        // chain can return capacity or clear lifecycle quiescence.
        drop(self.physical.take());
        drop(self.terminal.take());
    }
}

/// One admitted attempt's resource envelope.
pub(crate) struct AdmissionEnvelope {
    /// Retained so a pre-publication drop can account for deferred rollback
    /// cleanup until its stream callback has released the terminal sequence.
    service: Arc<QueryServiceInner>,
    device_ordinal: usize,
    teardown: AdmissionTeardown,
    /// Present only for an engine-owned attempt after the native domain and
    /// its post-install ring watermark have both been observed.
    allocator_domain_attempt: Option<AllocatorDomainAttempt>,
    /// Idempotence guard for the provisional-to-cleanup transition.
    rollback_cleanup_started: bool,
}

impl std::fmt::Debug for AdmissionEnvelope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdmissionEnvelope")
            .field("device_ordinal", &self.device_ordinal)
            .field("grant_id", &self.teardown.terminal().memory_grant().grant_id())
            .field("holds_gpu_resources", &self.teardown.physical.is_some())
            .finish_non_exhaustive()
    }
}

impl Drop for AdmissionEnvelope {
    fn drop(&mut self) {
        if let Some(binding) = self.allocator_domain_attempt.take() {
            // This is either an unpublished rollback or a recovery path that
            // could not prove cleanup completion. Neither may retain a window
            // that could attribute a later callback to this attempt.
            discard_admitted_domain(&self.service.allocator_event_correlation, binding);
        }
        let Some(physical) = self.teardown.physical.take() else {
            self.begin_terminal_cleanup();
            return;
        };
        let EnvelopePhysical { worker_scope: _, root_member, domain, root_stream } = physical;
        // Release stream reservation lease and domain before handling stream quiescence.
        drop(root_member.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner));
        drop(domain);
        let root_stream = root_stream.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);

        if let Some(root_stream) = root_stream {
            // Any envelope that reaches Drop with its root stream still
            // attached is a rollback path. Move its terminal owners behind
            // the stream quiescence fence before releasing capacity.
            self.begin_terminal_cleanup();
            defer_rollback_capacity_release(&self.service, root_stream, self.teardown.take_terminal());
        } else {
            self.begin_terminal_cleanup();
        }
    }
}

/// Builder whose drop releases partially acquired components in reverse
/// order. Field declaration order is the required drop order.
struct EnvelopeBuilder {
    service: Arc<QueryServiceInner>,
    lifecycle_ticket: Option<AdmissionTicket>,
    root_member: Option<cudf::StreamReservationLease>,
    domain: Option<AttemptAllocationDomain>,
    root_stream: Option<StreamLease>,
    lane_permit: Option<DeviceLanePermit>,
    memory_grant: Option<MemoryGrant>,
}

fn defer_rollback_capacity_release(
    service: &Arc<QueryServiceInner>,
    root_stream: StreamLease,
    terminal: AdmissionTerminalTeardown,
) {
    let service_for_notification = Arc::clone(service);
    let retention = Arc::new(Mutex::new(Some(terminal)));
    let release_after_quiescence = Arc::clone(&retention);
    let installed = root_stream.install_cleanup_completion(
        |_| {},
        move || {
            drop(release_after_quiescence.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take());
            // Notify only after the typed terminal sequence has dropped: its
            // lifecycle ticket is the final stage, making this the precise
            // quiescence edge.
            service_for_notification.admission.notify_admission_available();
        },
    );
    if !installed {
        eprintln!("[query-engine] warning: provisional admission stream cleanup completion was already installed");
        if let Some(retention) = retention.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            // Releasing capacity without a cleanup fence would permit a
            // replacement to overlap unknown GPU work. Leak closed on this
            // unreachable invariant violation instead.
            std::mem::forget(retention);
        }
    }
}

impl Drop for EnvelopeBuilder {
    fn drop(&mut self) {
        let mut lifecycle_ticket = self.lifecycle_ticket.take();
        if let Some(ticket) = lifecycle_ticket.as_mut() {
            let _ = ticket.transition_to_cleanup();
        }
        // Release the stream reservation lease before reclaiming the allocation domain.
        drop(self.root_member.take());
        drop(self.domain.take());
        let root_stream = self.root_stream.take();
        let Some(memory_grant) = self.memory_grant.take() else {
            // A successfully built envelope has moved every terminal owner
            // out of this builder; its destructor must therefore be a no-op.
            debug_assert!(root_stream.is_none(), "transferred envelope leaves no root stream in its builder");
            debug_assert!(self.lane_permit.is_none(), "transferred envelope leaves no lane permit in its builder");
            debug_assert!(lifecycle_ticket.is_none(), "transferred envelope leaves no lifecycle ticket in its builder");
            return;
        };
        let terminal = AdmissionTerminalTeardown::new(self.lane_permit.take(), memory_grant, None, lifecycle_ticket);

        let Some(root_stream) = root_stream else {
            // No stream cleanup fence exists yet. Release capacity explicitly
            // before the local lifecycle ticket drops; otherwise a concurrent
            // close could observe lifecycle quiescence while this builder
            // still holds an attempt lane or immutable grant.
            drop(terminal);
            return;
        };
        defer_rollback_capacity_release(&self.service, root_stream, terminal);
    }
}

/// The queue deadline ended at placement; from there the envelope is bounded
/// by the attempt's cancellation and its deadline.
fn check_admission_boundary(cancellation: &ExecutionCancellation) -> Result<()> {
    cancellation.check("query_engine.query_service.admission_transaction")
}

/// The attempt's cancellation, checked again once publication holds the
/// admission mutex.
pub(super) fn check_publication_cancellation(
    cancellation: &ExecutionCancellation,
    probe: Option<&AdmissionTransactionProbe>,
) -> Result<()> {
    const OPERATION: &str = "query_engine.query_service.admission_transaction.publish";
    if probe.is_some_and(AdmissionTransactionProbe::forces_cancellation_before_publication) {
        let forced = ExecutionCancellation::new();
        forced.cancel();
        return forced.check(OPERATION);
    }
    cancellation.check(OPERATION)
}

/// Optional rendezvous for the interval after rollback ownership is counted
/// and the admission mutex is released, but before dropping the envelope can
/// begin physical cleanup. Production's default probe makes this a no-op.
pub(super) fn pause_after_rollback_accounting(probe: Option<&AdmissionTransactionProbe>) {
    if let Some(probe) = probe
        && let Err(error) = probe.pause_after_rollback_accounting()
    {
        eprintln!("[query-engine] warning: rollback accounting probe failed: {error}");
    }
}

impl AdmissionEnvelope {
    /// Rolls a placed grant back before any envelope component exists, on
    /// the same path as a failed build.
    pub(crate) fn abandon(service: &Arc<QueryServiceInner>, placed: PlacedQuery, lifecycle_ticket: AdmissionTicket) {
        let _rollback = EnvelopeBuilder {
            service: Arc::clone(service),
            lifecycle_ticket: Some(lifecycle_ticket),
            root_member: None,
            domain: None,
            root_stream: None,
            lane_permit: None,
            memory_grant: Some(placed.memory_grant),
        };
    }

    /// Builds the envelope for a placed query while the admission mutex is
    /// released. The root stream acquisition is the only blocking step and
    /// observes the attempt-wide cancellation and deadline; the queue
    /// deadline ended at placement.
    pub(crate) fn acquire(
        service: &Arc<QueryServiceInner>,
        placed: PlacedQuery,
        lifecycle_ticket: AdmissionTicket,
        cancellation: &ExecutionCancellation,
        observation: &RuntimeObservationHandle,
        probe: Option<&AdmissionTransactionProbe>,
    ) -> Result<Self> {
        let operation = "query_engine.query_service.admission_transaction";
        let PlacedQuery { device_ordinal, memory_grant } = placed;
        let mut builder = EnvelopeBuilder {
            service: Arc::clone(service),
            lifecycle_ticket: Some(lifecycle_ticket),
            root_member: None,
            domain: None,
            root_stream: None,
            lane_permit: None,
            memory_grant: Some(memory_grant),
        };
        check_admission_boundary(cancellation)?;
        let resources = &service.resources;
        let device = resources.devices.context_by_ordinal(device_ordinal)?;
        let grant_ref = builder.memory_grant.as_ref().expect("grant present during envelope build");

        // (1) Attempt lane: non-blocking by construction (active <= A_d).
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeLanePermit))?;
        check_admission_boundary(cancellation)?;
        builder.lane_permit = Some(device.lane_permits().acquire()?);
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterLanePermit))?;
        check_admission_boundary(cancellation)?;

        // (2) Root stream (Root permit class).
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeRootStream))?;
        check_admission_boundary(cancellation)?;
        let stream_lease = device.stream_pool().acquire_observed_with_cancellation_and_class(
            cancellation,
            "query_engine.query_service.admission_transaction.stream_acquire",
            observation.clone(),
            RuntimeStreamLeaseScope::default(),
            StreamPermitClass::Root,
        )?;
        if stream_lease.device().ordinal() != device_ordinal {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                operation,
                "admission envelope root stream device does not match the ledger grant device",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("stream_device_ordinal", stream_lease.device().ordinal())
            .with_fact("reason", "envelope_identity_mismatch"));
        }
        builder.root_stream = Some(stream_lease);
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterRootStream))?;
        check_admission_boundary(cancellation)?;

        // (3) Allocation domain: grant == ceiling, domain_id == grant_id.
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeAllocationDomain))?;
        check_admission_boundary(cancellation)?;
        let runtime = builder.root_stream.as_ref().expect("root stream present").shared_runtime()?;
        let domain = AttemptAllocationDomain::create(runtime.as_ref(), grant_ref)?;
        if domain.device_ordinal() != device_ordinal {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                operation,
                "admission envelope domain device does not match the ledger grant device",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("domain_device_ordinal", domain.device_ordinal())
            .with_fact("reason", "envelope_identity_mismatch"));
        }
        builder.domain = Some(domain);
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterAllocationDomain))?;
        check_admission_boundary(cancellation)?;

        // (4) Root stream member of the attempt domain.
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeRootMember))?;
        check_admission_boundary(cancellation)?;
        builder.root_member = Some(builder.domain.as_ref().expect("domain present").create_member(runtime.stream())?);
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterRootMember))?;
        check_admission_boundary(cancellation)?;

        // (5) Device worker scope: identity evidence, non-blocking.
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeWorkerScope))?;
        check_admission_boundary(cancellation)?;
        let pool = resources.gpu_execution_pool_for_device(device_ordinal)?;
        let worker_scope = DeviceWorkerScope::new(
            device_ordinal,
            pool.snapshot().pool_id,
            builder.lane_permit.as_ref().expect("lane present").index(),
        );
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterWorkerScope))?;
        check_admission_boundary(cancellation)?;
        check_admission_boundary(cancellation)?;
        Ok(Self {
            service: Arc::clone(&builder.service),
            device_ordinal,
            teardown: AdmissionTeardown::new(
                EnvelopePhysical {
                    worker_scope,
                    root_member: Mutex::new(builder.root_member.take()),
                    domain: builder.domain.take().expect("domain present"),
                    root_stream: Mutex::new(builder.root_stream.take()),
                },
                builder.lane_permit.take().expect("lane present"),
                builder.memory_grant.take().expect("grant present"),
                builder.lifecycle_ticket.take(),
            ),
            allocator_domain_attempt: None,
            rollback_cleanup_started: false,
        })
    }

    pub(crate) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn memory_grant(&self) -> &MemoryGrant {
        self.teardown.terminal().memory_grant()
    }

    /// Atomically hands an unpublished envelope's lifecycle ticket from
    /// provisional admission to rollback cleanup. The caller holds the
    /// service admission mutex while installing the idempotence guard.
    pub(crate) fn preaccount_rollback_cleanup(&mut self, _state: &mut AdmissionState) {
        if !self.rollback_cleanup_started {
            if let Some(ticket) = self.teardown.terminal_mut().lifecycle_ticket_mut()
                && let Err(error) = ticket.transition_to_cleanup()
            {
                eprintln!(
                    "[query-engine] warning: failed to transition provisional admission ticket to cleanup: {error}"
                );
            }
            self.rollback_cleanup_started = true;
        }
    }

    pub(crate) fn activate_lifecycle_ticket(&mut self) -> Result<()> {
        self.teardown
            .terminal_mut()
            .lifecycle_ticket_mut()
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.admission_transaction",
                    "admission envelope has no lifecycle ticket",
                )
            })?
            .transition_to_active()
    }

    /// Records the exact native domain identity after its admission envelope
    /// is complete but before the query can execute. Diagnostic registration
    /// failure leaves later events explicitly unattributed and never changes
    /// admission or the immutable grant.
    pub(crate) fn register_allocator_domain_attempt(&mut self, attempt_id: AttemptId) {
        if self.allocator_domain_attempt.is_some() {
            return;
        }
        let Some(domain) = self.domain() else {
            return;
        };
        self.allocator_domain_attempt =
            register_admitted_domain(&self.service.allocator_event_correlation, domain.as_cudf(), attempt_id);
    }

    /// Moves the binding to the post-cleanup report commit. The binding is
    /// consumed exactly once, so envelope destruction cannot discard an
    /// already-retired attempt window.
    pub(crate) fn take_allocator_domain_attempt(&mut self) -> Option<AllocatorDomainAttempt> {
        self.allocator_domain_attempt.take()
    }

    /// Records the active-to-cleanup transition while retaining the ticket
    /// inside this envelope through all terminal resource destruction.
    pub(crate) fn begin_terminal_cleanup(&mut self) {
        if let Some(ticket) = self.teardown.terminal_mut().lifecycle_ticket_mut()
            && !ticket.is_cleanup()
            && let Err(error) = ticket.transition_to_cleanup()
        {
            eprintln!("[query-engine] warning: failed to transition admission ticket to cleanup: {error}");
        }
    }

    /// Adds the backend-generation permit at publication. It is intentionally
    /// not exposed again: the envelope releases it together with the admitted
    /// attempt's other terminal resources.
    pub(crate) fn retain_gpu_resource_permit(&mut self, gpu_resource_permit: Option<GpuResourcePermit>) {
        self.teardown.terminal_mut().retain_gpu_resource_permit(gpu_resource_permit);
    }

    /// The attempt allocation domain, until the unconsumed GPU-bound
    /// components are released.
    pub(crate) fn domain(&self) -> Option<&AttemptAllocationDomain> {
        self.teardown.physical.as_ref().map(|physical| &physical.domain)
    }

    /// Device worker scope identity evidence, until the unconsumed GPU-bound
    /// components are released.
    pub(crate) fn worker_scope(&self) -> Option<DeviceWorkerScope> {
        self.teardown.physical.as_ref().map(|physical| physical.worker_scope)
    }

    /// Attempt lane index while the terminal teardown is retained.
    pub(crate) fn lane_index(&self) -> Option<usize> {
        self.teardown.terminal().lane_index()
    }

    /// Whether this envelope still owns GPU-bound components. Once the root
    /// context consumed the stream and member, the remaining domain keeps the
    /// envelope GPU-bound until release.
    pub(crate) fn holds_gpu_resources(&self) -> bool {
        self.teardown.physical.is_some()
    }

    /// Takes the root stream binding for the first context acquisition.
    /// Exactly one call succeeds per envelope.
    pub(crate) fn take_root_binding(&self) -> Result<RootStreamBinding> {
        let operation = "query_engine.query_service.admission_transaction";
        let physical = self.teardown.physical.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "admission envelope already released its GPU-bound components",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "envelope_physical_released")
        })?;
        let stream_lease = physical
            .root_stream
            .lock()
            .map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, operation, "admission envelope stream mutex was poisoned")
            })?
            .take();
        let member_lease = physical
            .root_member
            .lock()
            .map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, operation, "admission envelope member mutex was poisoned")
            })?
            .take()
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "admission envelope root stream binding was already consumed",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "root_binding_consumed")
            })?;
        stream_lease.map(|stream_lease| RootStreamBinding { stream_lease, member_lease }).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "admission envelope root stream binding was already consumed",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "root_binding_consumed")
        })
    }

    /// Releases the unconsumed GPU-bound components for the no-context
    /// release path and returns the root stream lease whose cleanup
    /// completion fences the release. The member detaches and the domain is
    /// destroyed before the stream returns to its pool. Returns `None` when
    /// the root context already consumed the binding.
    pub(crate) fn release_unconsumed_physical(&mut self) -> Option<StreamLease> {
        let physical = self.teardown.physical.take()?;
        let member = physical.root_member.lock().ok().and_then(|mut slot| slot.take());
        // Release the stream reservation lease before reclaiming the allocation domain.
        drop(member);
        let domain = physical.domain;
        // Release the allocation domain before returning the unconsumed root stream lease.
        drop(domain);
        physical.root_stream.lock().ok().and_then(|mut slot| slot.take())
    }
}

#[cfg(test)]
mod teardown_sequence_tests {
    use std::sync::{Arc, Mutex};

    use super::TeardownStep;

    struct RecordedRelease {
        name: &'static str,
        releases: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for RecordedRelease {
        fn drop(&mut self) {
            self.releases.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(self.name);
        }
    }

    #[test]
    fn typed_teardown_releases_each_stage_before_its_successor() {
        let releases = Arc::new(Mutex::new(Vec::new()));
        let record = |name| RecordedRelease { name, releases: Arc::clone(&releases) };
        let teardown = TeardownStep {
            release: Some(record("lane")),
            then: TeardownStep {
                release: Some(record("grant")),
                then: TeardownStep {
                    release: Some(record("backend_permit")),
                    then: TeardownStep { release: Some(record("lifecycle_ticket")), then: () },
                },
            },
        };

        drop(teardown);

        assert_eq!(
            *releases.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            ["lane", "grant", "backend_permit", "lifecycle_ticket"]
        );
    }
}
