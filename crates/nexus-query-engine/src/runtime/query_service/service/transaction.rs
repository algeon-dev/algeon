//! Transactional admission envelope.
//!
//! One admitted attempt owns its ledger grant, attempt lane, root stream
//! (Root permit class), allocation domain with the root stream member, and
//! device worker scope as a single transaction. The envelope is built after
//! the ledger transaction commits (inside the admission mutex) but while the
//! admission mutex is released, so a bounded stream wait never blocks queue
//! maintenance or cleanup. Admission is published — the `Admitted` lifecycle
//! fact and the [`QueryHandle`] — only after every component validates.
//!
//! Rollback drops components in reverse acquisition order (member → domain →
//! stream → lane → grant), preserves the primary error, and restores every
//! admission counter exactly. Fault injection via
//! [`AdmissionTransactionProbe`] proves both properties at every step.
//!
//! Services constructed without runtime resources (admission-only tests and
//! tools) admit ledger-only envelopes: the grant with no physical components.
//! Every attempt admitted against a real device always carries the complete
//! physical envelope.

#[cfg(test)]
use std::time::Duration;
use std::{
    sync::{Arc, Mutex, mpsc},
    time::Instant,
};

use super::{
    super::admission::{AdmissionTicket, PlacedQuery},
    QueryServiceInner,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    observability::{AttemptId, RuntimeObservationHandle, RuntimeStreamLeaseScope},
    runtime::{
        ExecutionCancellation, StreamLease, StreamPermitClass,
        allocator_correlation::{AllocatorDomainAttempt, discard_admitted_domain, register_admitted_domain},
        attempt_domain::AttemptAllocationDomain,
        backend_core::GpuResourcePermit,
        gpu::{DeviceLanePermit, DeviceWorkerScope, MemoryGrant},
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

/// Fault-injection probe for the admission transaction. The default is no-op.
pub(crate) struct AdmissionTransactionProbe {
    fail_at: Mutex<Option<AdmissionStep>>,
    pause_at: Mutex<Option<(AdmissionStep, Arc<AdmissionTransactionPause>)>>,
    force_deadline_at: Mutex<Option<ForcedAdmissionDeadline>>,
    rollback_pause: Mutex<Option<Arc<AdmissionTransactionPause>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedAdmissionDeadline {
    Acquisition(AdmissionStep),
    BeforePublication,
}

struct AdmissionTransactionPause {
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}

/// Test-side half of a transaction pause. Channel rendezvous are bounded so a
/// setup failure before the requested admission step fails the test instead of
/// hanging the whole nextest process on an unbounded barrier.
#[cfg(test)]
pub(crate) struct AdmissionTransactionPauseHandle {
    entered: Mutex<mpsc::Receiver<()>>,
    release: Mutex<Option<mpsc::SyncSender<()>>>,
}

#[cfg(test)]
impl AdmissionTransactionPauseHandle {
    pub(crate) fn wait_until_entered(&self, timeout: Duration) {
        self.entered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(timeout)
            .expect("admission transaction did not reach the requested pause step before the deadline");
    }

    pub(crate) fn release(&self) {
        let release = self.release.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(release) = release {
            release.send(()).expect("paused admission transaction remains live until release");
        }
    }
}

#[cfg(test)]
impl Drop for AdmissionTransactionPauseHandle {
    fn drop(&mut self) {
        if let Some(release) = self.release.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            let _ = release.try_send(());
        }
    }
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
            force_deadline_at: Mutex::new(None),
            rollback_pause: Mutex::new(None),
        }
    }

    #[cfg(test)]
    pub(crate) fn failing_at(step: AdmissionStep) -> Self {
        Self {
            fail_at: Mutex::new(Some(step)),
            pause_at: Mutex::new(None),
            force_deadline_at: Mutex::new(None),
            rollback_pause: Mutex::new(None),
        }
    }

    /// Pauses a transaction after it acquires the resources preceding `step`.
    /// The bounded entry notification tells the test that the provisional
    /// grant exists; the release channel lets it resume without publishing it
    /// early.
    #[cfg(test)]
    pub(crate) fn pausing_at(step: AdmissionStep) -> (Self, AdmissionTransactionPauseHandle) {
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (release_sender, release_receiver) = mpsc::sync_channel(1);
        let pause =
            Arc::new(AdmissionTransactionPause { entered: entered_sender, release: Mutex::new(release_receiver) });
        (
            Self {
                fail_at: Mutex::new(None),
                pause_at: Mutex::new(Some((step, pause))),
                force_deadline_at: Mutex::new(None),
                rollback_pause: Mutex::new(None),
            },
            AdmissionTransactionPauseHandle {
                entered: Mutex::new(entered_receiver),
                release: Mutex::new(Some(release_sender)),
            },
        )
    }

    /// Forces the boundary immediately after `step` to behave as if its
    /// deadline elapsed.  This keeps deadline coverage deterministic without
    /// turning a scheduler delay into a test precondition.
    #[cfg(test)]
    pub(crate) fn force_deadline_after_acquisition_step(&self, step: AdmissionStep) {
        *self.force_deadline_at.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(ForcedAdmissionDeadline::Acquisition(step));
    }

    /// Forces only the publication recheck to expire.  Acquisition still
    /// completes, allowing tests to prove the post-mutex publication gate.
    #[cfg(test)]
    pub(crate) fn force_deadline_before_publication(&self) {
        *self.force_deadline_at.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(ForcedAdmissionDeadline::BeforePublication);
    }

    /// Pauses after an unpublished envelope has atomically transferred its
    /// lifecycle ticket into rollback cleanup, but before its drop starts
    /// physical cleanup. This exposes the exact shutdown-quiescence boundary.
    #[cfg(test)]
    pub(crate) fn pausing_after_rollback_accounting() -> (Self, AdmissionTransactionPauseHandle) {
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (release_sender, release_receiver) = mpsc::sync_channel(1);
        let pause =
            Arc::new(AdmissionTransactionPause { entered: entered_sender, release: Mutex::new(release_receiver) });
        (
            Self {
                fail_at: Mutex::new(None),
                pause_at: Mutex::new(None),
                force_deadline_at: Mutex::new(Some(ForcedAdmissionDeadline::BeforePublication)),
                rollback_pause: Mutex::new(Some(pause)),
            },
            AdmissionTransactionPauseHandle {
                entered: Mutex::new(entered_receiver),
                release: Mutex::new(Some(release_sender)),
            },
        )
    }

    fn forces_deadline_at_acquisition_stage(&self, stage: &'static str) -> bool {
        matches!(
            *self.force_deadline_at.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            Some(ForcedAdmissionDeadline::Acquisition(step)) if step.as_str() == stage
        )
    }

    fn forces_deadline_before_publication(&self) -> bool {
        matches!(
            *self.force_deadline_at.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            Some(ForcedAdmissionDeadline::BeforePublication)
        )
    }

    fn pause_after_rollback_accounting(&self) -> Result<()> {
        let pause = self
            .rollback_pause
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission_transaction",
                    "admission rollback pause-probe mutex was poisoned",
                )
            })?
            .as_ref()
            .map(Arc::clone);
        if let Some(pause) = pause {
            pause.entered.send(()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission_transaction",
                    "admission rollback pause observer disconnected",
                )
            })?;
            pause
                .release
                .lock()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.query_service.admission_transaction",
                        "admission rollback pause release mutex was poisoned",
                    )
                })?
                .recv()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.query_service.admission_transaction",
                        "admission rollback pause releaser disconnected",
                    )
                })?;
        }
        Ok(())
    }

    fn check(&self, step: AdmissionStep) -> Result<()> {
        let fail_at = self.fail_at.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission_transaction",
                "admission transaction probe mutex was poisoned",
            )
        })?;
        if *fail_at == Some(step) {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission_transaction",
                "admission transaction step failed by injected fault",
            )
            .with_fact("reason", "injected_admission_step_fault")
            .with_fact("admission_step", step.as_str()));
        }
        drop(fail_at);
        if let Some(pause) = self
            .pause_at
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission_transaction",
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
                    "nexus_query_engine.query_service.admission_transaction",
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
                        "nexus_query_engine.query_service.admission_transaction",
                        "admission transaction pause release mutex was poisoned",
                    )
                    .with_fact("admission_step", step.as_str())
                })?
                .recv()
                .map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.query_service.admission_transaction",
                        "admission transaction pause releaser disconnected",
                    )
                    .with_fact("admission_step", step.as_str())
                })?;
        }
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

/// One admitted attempt's resource envelope.
///
/// Field declaration order is the required drop order: physical components,
/// attempt lane, immutable ledger grant, backend generation permit, and
/// finally the admission lifecycle ticket.
pub(crate) struct AdmissionEnvelope {
    /// Retained so a pre-publication drop can account for deferred rollback
    /// cleanup until its stream callback has released the lane and grant.
    service: Arc<QueryServiceInner>,
    device_ordinal: usize,
    physical: Option<EnvelopePhysical>,
    lane_permit: Option<DeviceLanePermit>,
    memory_grant: MemoryGrant,
    /// Retained by the published envelope until every CUDA-bound component
    /// and the immutable grant have been released.
    gpu_resource_permit: Option<GpuResourcePermit>,
    /// Final close/drain fence. This remains with the envelope so it cannot
    /// outlive or be released before the admission resources it represents.
    lifecycle_ticket: Option<AdmissionTicket>,
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
            .field("grant_id", &self.memory_grant.grant_id())
            .field("ledger_only", &self.physical.is_none())
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
        let Some(physical) = self.physical.take() else {
            // Keep the ticket in its field. Rust drops fields after this
            // destructor in declaration order, so the grant and generation
            // permit release before the ticket clears lifecycle quiescence.
            self.begin_terminal_cleanup();
            return;
        };
        let EnvelopePhysical { worker_scope: _, root_member, domain, root_stream } = physical;
        drop(root_member.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner));
        drop(domain);
        let root_stream = root_stream.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);

        if let Some(root_stream) = root_stream {
            // Any envelope that reaches Drop with its root stream still
            // attached is a rollback path. Move its terminal owners behind
            // the stream quiescence fence before releasing capacity.
            self.begin_terminal_cleanup();
            defer_rollback_capacity_release(
                &self.service,
                root_stream,
                self.lane_permit.take(),
                Some(self.memory_grant.clone()),
                self.gpu_resource_permit.take(),
                self.lifecycle_ticket.take(),
            );
        } else {
            // The root binding was already consumed. Keep both terminal
            // owners in their fields so normal struct-field destruction
            // releases them only after lane and grant cleanup.
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

/// Capacity acquired for a provisional admission remains unavailable until
/// the root stream's deferred cleanup has crossed its quiescence boundary.
/// The field order preserves lane-before-grant release at that boundary.
struct RollbackCapacityRetention {
    _lane_permit: Option<DeviceLanePermit>,
    _memory_grant: Option<MemoryGrant>,
    _gpu_resource_permit: Option<GpuResourcePermit>,
    /// Dropped last, after lane/grant release has crossed the stream cleanup
    /// fence and the generation permit has been released, so lifecycle
    /// quiescence cannot be observed before capacity is mechanically reusable.
    _lifecycle_ticket: Option<AdmissionTicket>,
}

fn defer_rollback_capacity_release(
    service: &Arc<QueryServiceInner>,
    root_stream: StreamLease,
    lane_permit: Option<DeviceLanePermit>,
    memory_grant: Option<MemoryGrant>,
    gpu_resource_permit: Option<GpuResourcePermit>,
    lifecycle_ticket: Option<AdmissionTicket>,
) {
    let service_for_notification = Arc::clone(service);
    let retention = Arc::new(Mutex::new(Some(RollbackCapacityRetention {
        _lane_permit: lane_permit,
        _memory_grant: memory_grant,
        _gpu_resource_permit: gpu_resource_permit,
        _lifecycle_ticket: lifecycle_ticket,
    })));
    let release_after_quiescence = Arc::clone(&retention);
    let installed = root_stream.install_cleanup_completion(
        |_| {},
        move || {
            drop(release_after_quiescence.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take());
            // Notify only after the retention fields have dropped: the
            // lifecycle ticket is declared last and therefore removed after
            // the ledger grant/lane and backend-generation permit, making
            // this the precise quiescence edge.
            service_for_notification.notify_admission_available();
        },
    );
    if !installed {
        eprintln!(
            "[nexus-query-engine] warning: provisional admission stream cleanup completion was already installed"
        );
        if let Some(retention) = retention.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            // Releasing capacity without a cleanup fence would permit a
            // replacement to overlap unknown GPU work. Leak closed on this
            // unreachable invariant violation instead.
            std::mem::forget(retention);
        }
    }
    drop(root_stream);
}

impl Drop for EnvelopeBuilder {
    fn drop(&mut self) {
        let mut lifecycle_ticket = self.lifecycle_ticket.take();
        if let Some(ticket) = lifecycle_ticket.as_mut() {
            let _ = ticket.transition_to_cleanup();
        }
        drop(self.root_member.take());
        drop(self.domain.take());
        let root_stream = self.root_stream.take();

        let Some(root_stream) = root_stream else {
            // No stream cleanup fence exists yet. Release capacity explicitly
            // before the local lifecycle ticket drops; otherwise a concurrent
            // close could observe lifecycle quiescence while this builder
            // still holds an attempt lane or immutable grant.
            drop(self.lane_permit.take());
            drop(self.memory_grant.take());
            drop(lifecycle_ticket);
            return;
        };
        defer_rollback_capacity_release(
            &self.service,
            root_stream,
            self.lane_permit.take(),
            self.memory_grant.take(),
            None,
            lifecycle_ticket,
        );
    }
}

fn check_admission_boundary(
    cancellation: &ExecutionCancellation,
    deadline: Instant,
    acquisition_stage: &'static str,
    probe: Option<&AdmissionTransactionProbe>,
) -> Result<()> {
    const OPERATION: &str = "nexus_query_engine.query_service.admission_transaction";
    cancellation.check(OPERATION)?;
    let deadline_elapsed = Instant::now() >= deadline
        || probe.is_some_and(|probe| probe.forces_deadline_at_acquisition_stage(acquisition_stage));
    if !deadline_elapsed {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        OPERATION,
        "query service admission deadline elapsed while acquiring the resource envelope",
    )
    .with_fact("reason", "queue_wait_elapsed")
    .with_fact("acquisition_stage", acquisition_stage))
}

pub(super) fn publication_deadline_error(
    deadline: Instant,
    probe: Option<&AdmissionTransactionProbe>,
) -> Option<QueryError> {
    (Instant::now() >= deadline || probe.is_some_and(AdmissionTransactionProbe::forces_deadline_before_publication))
        .then(|| {
            QueryError::new(
                QueryErrorCode::ServiceOverloaded,
                "nexus_query_engine.query_service.admission_transaction.publish",
                "query service admission deadline elapsed before publication",
            )
            .with_fact("reason", "queue_wait_elapsed")
            .with_fact("acquisition_stage", "before_publication")
        })
}

/// Optional rendezvous for the interval after rollback ownership is counted
/// and the admission mutex is released, but before dropping the envelope can
/// begin physical cleanup. Production's default probe makes this a no-op.
pub(super) fn pause_after_rollback_accounting(probe: Option<&AdmissionTransactionProbe>) {
    if let Some(probe) = probe
        && let Err(error) = probe.pause_after_rollback_accounting()
    {
        eprintln!("[nexus-query-engine] warning: rollback accounting probe failed: {error}");
    }
}

impl AdmissionEnvelope {
    /// Builds the envelope for a placed query while the admission mutex is
    /// released. The root stream acquisition is the only blocking step and
    /// observes the attempt-wide cancellation and queue deadline.
    pub(crate) fn acquire(
        service: &Arc<QueryServiceInner>,
        placed: PlacedQuery,
        lifecycle_ticket: AdmissionTicket,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        observation: &RuntimeObservationHandle,
        probe: Option<&AdmissionTransactionProbe>,
    ) -> Result<Self> {
        let operation = "nexus_query_engine.query_service.admission_transaction";
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
        check_admission_boundary(cancellation, deadline, "transaction_entry", probe)?;
        let Some(resources) = service.resources.as_ref() else {
            // Admission-only service: the ledger grant is the entire envelope.
            check_admission_boundary(cancellation, deadline, "transaction_complete", probe)?;
            return Ok(Self {
                service: Arc::clone(&builder.service),
                device_ordinal,
                physical: None,
                lane_permit: None,
                memory_grant: builder.memory_grant.take().expect("grant present"),
                gpu_resource_permit: None,
                lifecycle_ticket: builder.lifecycle_ticket.take(),
                allocator_domain_attempt: None,
                rollback_cleanup_started: false,
            });
        };
        let device = resources.devices.context_by_ordinal(device_ordinal)?;
        let grant_ref = builder.memory_grant.as_ref().expect("grant present during envelope build");

        // (1) Attempt lane: non-blocking by construction (active <= A_d).
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeLanePermit))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::BeforeLanePermit.as_str(), probe)?;
        builder.lane_permit = Some(device.lane_permits().acquire()?);
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterLanePermit))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::AfterLanePermit.as_str(), probe)?;

        // (2) Root stream (Root permit class).
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeRootStream))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::BeforeRootStream.as_str(), probe)?;
        let stream_lease = device.stream_pool().acquire_observed_with_cancellation_deadline_and_class(
            cancellation,
            deadline,
            "nexus_query_engine.query_service.admission_transaction.stream_acquire",
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
        check_admission_boundary(cancellation, deadline, AdmissionStep::AfterRootStream.as_str(), probe)?;

        // (3) Allocation domain: grant == ceiling, domain_id == grant_id.
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeAllocationDomain))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::BeforeAllocationDomain.as_str(), probe)?;
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
        check_admission_boundary(cancellation, deadline, AdmissionStep::AfterAllocationDomain.as_str(), probe)?;

        // (4) Root stream member of the attempt domain.
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeRootMember))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::BeforeRootMember.as_str(), probe)?;
        builder.root_member = Some(builder.domain.as_ref().expect("domain present").create_member(runtime.stream())?);
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterRootMember))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::AfterRootMember.as_str(), probe)?;

        // (5) Device worker scope: identity evidence, non-blocking.
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::BeforeWorkerScope))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::BeforeWorkerScope.as_str(), probe)?;
        let pool = resources.gpu_execution_pool_for_device(device_ordinal)?;
        let worker_scope = DeviceWorkerScope::new(
            device_ordinal,
            pool.snapshot().pool_id,
            builder.lane_permit.as_ref().expect("lane present").index(),
        );
        probe.map_or(Ok(()), |probe| probe.check(AdmissionStep::AfterWorkerScope))?;
        check_admission_boundary(cancellation, deadline, AdmissionStep::AfterWorkerScope.as_str(), probe)?;
        check_admission_boundary(cancellation, deadline, "transaction_complete", probe)?;
        Ok(Self {
            service: Arc::clone(&builder.service),
            device_ordinal,
            physical: Some(EnvelopePhysical {
                worker_scope,
                root_member: Mutex::new(builder.root_member.take()),
                domain: builder.domain.take().expect("domain present"),
                root_stream: Mutex::new(builder.root_stream.take()),
            }),
            lane_permit: builder.lane_permit.take(),
            memory_grant: builder.memory_grant.take().expect("grant present"),
            gpu_resource_permit: None,
            lifecycle_ticket: builder.lifecycle_ticket.take(),
            allocator_domain_attempt: None,
            rollback_cleanup_started: false,
        })
    }

    pub(crate) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn memory_grant(&self) -> &MemoryGrant {
        &self.memory_grant
    }

    /// Atomically hands an unpublished envelope's lifecycle ticket from
    /// provisional admission to rollback cleanup. The caller holds the
    /// service admission mutex while installing the idempotence guard.
    pub(crate) fn preaccount_rollback_cleanup(&mut self, _state: &mut super::super::admission::AdmissionState) {
        if !self.rollback_cleanup_started {
            if let Some(ticket) = self.lifecycle_ticket.as_mut()
                && let Err(error) = ticket.transition_to_cleanup()
            {
                eprintln!(
                    "[nexus-query-engine] warning: failed to transition provisional admission ticket to cleanup: {error}"
                );
            }
            self.rollback_cleanup_started = true;
        }
    }

    pub(crate) fn activate_lifecycle_ticket(&mut self) -> Result<()> {
        self.lifecycle_ticket
            .as_mut()
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission_transaction",
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
        if let Some(ticket) = self.lifecycle_ticket.as_mut()
            && !ticket.is_cleanup()
            && let Err(error) = ticket.transition_to_cleanup()
        {
            eprintln!("[nexus-query-engine] warning: failed to transition admission ticket to cleanup: {error}");
        }
    }

    /// Adds the backend-generation permit at publication. It is intentionally
    /// not exposed again: the envelope releases it together with the admitted
    /// attempt's other terminal resources.
    pub(crate) fn retain_gpu_resource_permit(&mut self, gpu_resource_permit: Option<GpuResourcePermit>) {
        debug_assert!(
            self.gpu_resource_permit.is_none(),
            "published admission envelope can retain only one GPU resource permit"
        );
        self.gpu_resource_permit = gpu_resource_permit;
    }

    /// The attempt allocation domain. Always present for device-backed
    /// admissions; `None` only on ledger-only envelopes.
    pub(crate) fn domain(&self) -> Option<&AttemptAllocationDomain> {
        self.physical.as_ref().map(|physical| &physical.domain)
    }

    /// Device worker scope identity evidence for device-backed admissions.
    pub(crate) fn worker_scope(&self) -> Option<DeviceWorkerScope> {
        self.physical.as_ref().map(|physical| physical.worker_scope)
    }

    /// Attempt lane index for device-backed admissions.
    pub(crate) fn lane_index(&self) -> Option<usize> {
        self.lane_permit.as_ref().map(DeviceLanePermit::index)
    }

    /// Whether this envelope still owns GPU-bound components. Once the root
    /// context consumed the stream and member, the remaining domain keeps the
    /// envelope GPU-bound until release.
    pub(crate) fn holds_gpu_resources(&self) -> bool {
        self.physical.is_some()
    }

    /// Takes the root stream binding for the first context acquisition.
    /// Exactly one call succeeds per envelope.
    pub(crate) fn take_root_binding(&self) -> Result<RootStreamBinding> {
        let operation = "nexus_query_engine.query_service.admission_transaction";
        let physical = self.physical.as_ref().ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, operation, "ledger-only admission envelope has no root binding")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "ledger_only_envelope")
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
        let physical = self.physical.take()?;
        let member = physical.root_member.lock().ok().and_then(|mut slot| slot.take());
        drop(member);
        let domain = physical.domain;
        drop(domain);
        physical.root_stream.lock().ok().and_then(|mut slot| slot.take())
    }
}
