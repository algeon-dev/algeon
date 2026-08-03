use super::{
    super::context::{
        AttemptAuxiliaryPermit, AttemptAuxiliarySlots, QueryResourceAnchor, QueryResourceEpoch,
        QueryServiceResourceEpoch,
    },
    *,
};
#[cfg(test)]
use crate::runtime::attempt_domain::AttemptAllocationDomain;
use crate::{
    observability::{
        AttemptAllocatorEvent, AttemptAllocatorEventAction, AttemptAllocatorEventSource, AttemptAllocatorPoolUsage,
        AttemptEvidenceField, AttemptFaultIsolation, AttemptResourceEvidence, AttemptTimedDomainUsage,
        AttemptUnattributedAllocatorEvent,
    },
    runtime::{
        GpuExecutorLease,
        allocator_correlation::{
            AllocatorDomainAttempt, AllocatorEventEvidenceRead, discard_admitted_domain, retire_admitted_domain,
        },
    },
};

const CONTEXT_SETUP_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextSetupClaim {
    Root,
    Auxiliary,
    WaitForRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootContextSetupState {
    Unclaimed,
    InFlight,
    Ready,
}

#[derive(Debug)]
struct RootContextSetupCoordinator {
    state: Mutex<RootContextSetupState>,
    changed: Condvar,
    async_changed: tokio::sync::Notify,
}

impl RootContextSetupCoordinator {
    fn new() -> Self {
        Self {
            state: Mutex::new(RootContextSetupState::Unclaimed),
            changed: Condvar::new(),
            async_changed: tokio::sync::Notify::new(),
        }
    }
}

/// Owns a root-context setup claim until setup becomes ready or the work is
/// abandoned. The claim is moved into the device-pool future so cancellation
/// while queued drops it at the same boundary that drops the queued work.
#[derive(Debug)]
struct RootContextSetupClaimGuard {
    lifetime: Arc<QueryHandleLifetime>,
}

impl RootContextSetupClaimGuard {
    fn new(lifetime: Arc<QueryHandleLifetime>) -> Self {
        Self { lifetime }
    }
}

impl Drop for RootContextSetupClaimGuard {
    fn drop(&mut self) {
        self.lifetime.rollback_root_context_setup();
    }
}

#[derive(Debug)]
pub(super) struct QueryHandleLifetime {
    service: Mutex<Option<Arc<QueryServiceInner>>>,
    pub(super) admission_request_id: AdmissionRequestId,
    observation: RuntimeObservationHandle,
    completion: Mutex<Option<QueryCompletion>>,
    service_epoch_user: Mutex<Option<QueryServiceEpochUser>>,
    device_ordinal: usize,
    /// Immutable admission inputs retained only until the post-cleanup report
    /// commit can determine whether a quarantined device has a truthful
    /// alternate retry target. They never drive a second placement.
    admitted_requirement: AttemptRequirement,
    compatible_devices: Vec<usize>,
    release_requested: AtomicBool,
    released: AtomicBool,
    query_epoch: QueryResourceEpoch,
    root_context: Mutex<Option<ExecutionContext>>,
    root_context_setup: RootContextSetupCoordinator,
    attempt_auxiliary_slots: Arc<AttemptAuxiliarySlots>,
    envelope: Mutex<Option<super::service::transaction::AdmissionEnvelope>>,
    #[cfg(test)]
    release_commit_gate: Mutex<Option<Arc<ReleaseCommitGate>>>,
    #[cfg(test)]
    root_context_binding_gate: Mutex<Option<Arc<RootContextBindingGate>>>,
    terminal_domain_usage: Mutex<Option<cudf::QueryMemoryDomainUsage>>,
}

impl QueryHandleLifetime {
    pub(super) fn new(
        service: Arc<QueryServiceInner>,
        admission_request_id: AdmissionRequestId,
        observation: RuntimeObservationHandle,
        completion: QueryCompletion,
        envelope: super::service::transaction::AdmissionEnvelope,
        admitted_requirement: AttemptRequirement,
        compatible_devices: Vec<usize>,
    ) -> Arc<Self> {
        let device_ordinal = envelope.device_ordinal();
        let attempt_auxiliary_capacity = service
            .resources
            .as_ref()
            .and_then(|resources| resources.devices.resource_profile(device_ordinal))
            .map_or_else(
                || service.config.source_scheduler_policy.max_source_reads_in_flight().max(1),
                |profile| profile.source_work_per_attempt,
            );
        let attempt_auxiliary_slots = AttemptAuxiliarySlots::new(device_ordinal, attempt_auxiliary_capacity);
        let service_epoch_user = service.service_epoch.retain_existing_user();
        Arc::new_cyclic(|weak: &Weak<QueryHandleLifetime>| {
            let weak_lifetime = weak.clone();
            let on_query_idle = Arc::new(move || {
                if let Some(lifetime) = weak_lifetime.upgrade() {
                    lifetime.release_if_ready();
                }
            });
            Self {
                service: Mutex::new(Some(service)),
                admission_request_id,
                observation,
                completion: Mutex::new(Some(completion)),
                service_epoch_user: Mutex::new(Some(service_epoch_user)),
                device_ordinal,
                admitted_requirement,
                compatible_devices,
                release_requested: AtomicBool::new(false),
                released: AtomicBool::new(false),
                query_epoch: QueryResourceEpoch::new(Some(on_query_idle)),
                root_context: Mutex::new(None),
                root_context_setup: RootContextSetupCoordinator::new(),
                attempt_auxiliary_slots,
                envelope: Mutex::new(Some(envelope)),
                #[cfg(test)]
                release_commit_gate: Mutex::new(None),
                #[cfg(test)]
                root_context_binding_gate: Mutex::new(None),
                terminal_domain_usage: Mutex::new(None),
            }
        })
    }

    fn lock_envelope(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<super::service::transaction::AdmissionEnvelope>>> {
        self.envelope.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.context",
                "query service admission envelope mutex was poisoned",
            )
        })
    }

    /// Attempt allocation domain clone for disposition tests.
    #[cfg(test)]
    pub(super) fn envelope_domain_for_test(&self) -> Option<AttemptAllocationDomain> {
        self.lock_envelope().ok()?.as_ref().and_then(|envelope| envelope.domain().cloned())
    }

    /// Admission-envelope identity evidence for transaction tests:
    /// (domain id, grant id, worker scope).
    #[cfg(test)]
    pub(super) fn envelope_evidence(
        &self,
    ) -> Option<(Option<u64>, u64, Option<crate::runtime::gpu::DeviceWorkerScope>)> {
        self.lock_envelope().ok()?.as_ref().map(|envelope| {
            (
                envelope.domain().map(|domain| domain.domain_id()),
                envelope.memory_grant().grant_id(),
                envelope.worker_scope(),
            )
        })
    }

    /// Arms a barrier inside this handle's release commit, after the health
    /// commit and before the observation seal. The barrier fires exactly once.
    #[cfg(test)]
    pub(super) fn arm_release_commit_gate(&self) -> Arc<ReleaseCommitGate> {
        let gate = Arc::new(ReleaseCommitGate {
            entered: std::sync::atomic::AtomicUsize::new(0),
            gate: std::sync::Barrier::new(2),
        });
        *self.release_commit_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&gate));
        gate
    }

    /// Pauses root-context setup after it has consumed the root binding while
    /// the admission envelope remains locked. This makes release/setup races
    /// deterministic without exposing a production lifecycle seam.
    #[cfg(test)]
    pub(super) fn arm_root_context_binding_gate(&self) -> Arc<RootContextBindingGate> {
        let gate = Arc::new(RootContextBindingGate {
            entered: std::sync::atomic::AtomicUsize::new(0),
            gate: std::sync::Barrier::new(2),
        });
        *self.root_context_binding_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Arc::clone(&gate));
        gate
    }

    #[cfg(test)]
    fn wait_at_root_context_binding_gate(&self) {
        let gate = self
            .root_context_binding_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(Arc::clone);
        if let Some(gate) = gate {
            gate.entered.fetch_add(1, Ordering::AcqRel);
            gate.gate.wait();
        }
    }

    /// Overrides this handle's terminal attempt-domain sample exactly once.
    #[cfg(test)]
    pub(super) fn inject_terminal_domain_usage_for_test(&self, usage: cudf::QueryMemoryDomainUsage) {
        *self.terminal_domain_usage.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(usage);
    }

    /// Installs an unrelated root-stream completion to exercise the
    /// fail-closed duplicate-handoff path. Normal query-service ownership
    /// never permits this; a later release must commit `CallbackAbandoned`
    /// before it returns admission capacity.
    #[cfg(test)]
    pub(super) fn install_foreign_root_stream_cleanup_completion_for_test(&self) {
        let root_context = self.root_context.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let root_context = root_context.as_ref().expect("root context is acquired before injecting cleanup collision");
        assert!(root_context.install_root_stream_cleanup_completion(|_| {}, || {}));
    }

    pub(super) fn memory_grant(&self) -> Result<MemoryGrant> {
        self.lock_envelope()?
            .as_ref()
            .map(|envelope| envelope.memory_grant().clone())
            .ok_or_else(|| self.released_error())
    }

    fn acquire_resource_anchor(self: &Arc<Self>, operation: &'static str) -> Result<Arc<QueryResourceAnchor>> {
        if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
            return Err(self.released_error());
        }
        let owner: Arc<dyn Send + Sync> = self.clone();
        self.query_epoch.acquire(owner, operation).map_err(|_error| self.released_error())
    }

    fn lock_root_context_setup(
        &self,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, RootContextSetupState>> {
        self.root_context_setup.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "query service root-context setup mutex was poisoned")
                .with_fact("reason", "root_context_setup_poisoned")
        })
    }

    fn reserve_context_setup(&self, operation: &'static str) -> Result<ContextSetupClaim> {
        if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
            return Err(self.released_error());
        }
        let mut state = self.lock_root_context_setup(operation)?;
        Ok(match *state {
            RootContextSetupState::Unclaimed => {
                *state = RootContextSetupState::InFlight;
                ContextSetupClaim::Root
            },
            RootContextSetupState::InFlight => ContextSetupClaim::WaitForRoot,
            RootContextSetupState::Ready => ContextSetupClaim::Auxiliary,
        })
    }

    fn wait_for_root_context_setup(&self, cancellation: &ExecutionCancellation, operation: &'static str) -> Result<()> {
        loop {
            cancellation.check(operation)?;
            if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
                return Err(self.released_error());
            }
            let state = self.lock_root_context_setup(operation)?;
            if *state != RootContextSetupState::InFlight {
                return Ok(());
            }
            let (_state, _timeout) =
                self.root_context_setup.changed.wait_timeout(state, CONTEXT_SETUP_WAIT_POLL_INTERVAL).map_err(
                    |_error| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            operation,
                            "query service root-context setup mutex was poisoned while waiting",
                        )
                        .with_fact("reason", "root_context_setup_poisoned")
                    },
                )?;
        }
    }

    async fn wait_for_root_context_setup_async(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<()> {
        loop {
            cancellation.check(operation)?;
            if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
                return Err(self.released_error());
            }
            // Register before looking at the state so a transition cannot be
            // lost between the observation and the await.
            let changed = self.root_context_setup.async_changed.notified();
            let root_setup_in_flight = {
                let state = self.lock_root_context_setup(operation)?;
                *state == RootContextSetupState::InFlight
            };
            if !root_setup_in_flight {
                return Ok(());
            }
            let _ = tokio::time::timeout(CONTEXT_SETUP_WAIT_POLL_INTERVAL, changed).await;
        }
    }

    fn reserve_context_setup_sync(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<ContextSetupClaim> {
        loop {
            match self.reserve_context_setup(operation)? {
                ContextSetupClaim::WaitForRoot => self.wait_for_root_context_setup(cancellation, operation)?,
                claim => return Ok(claim),
            }
        }
    }

    async fn reserve_context_setup_async(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<ContextSetupClaim> {
        loop {
            match self.reserve_context_setup(operation)? {
                ContextSetupClaim::WaitForRoot => {
                    self.wait_for_root_context_setup_async(cancellation, operation).await?
                },
                claim => return Ok(claim),
            }
        }
    }

    fn complete_root_context_setup(&self) {
        let mut state = self.root_context_setup.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(*state, RootContextSetupState::InFlight);
        *state = RootContextSetupState::Ready;
        drop(state);
        self.root_context_setup.changed.notify_all();
        self.root_context_setup.async_changed.notify_waiters();
    }

    fn rollback_root_context_setup(&self) {
        let mut state = self.root_context_setup.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if *state == RootContextSetupState::InFlight {
            *state = RootContextSetupState::Unclaimed;
        }
        drop(state);
        self.root_context_setup.changed.notify_all();
        self.root_context_setup.async_changed.notify_waiters();
    }

    fn wake_root_context_setup_waiters(&self) {
        self.root_context_setup.changed.notify_all();
        self.root_context_setup.async_changed.notify_waiters();
    }

    fn request_release(&self) {
        self.release_requested.store(true, Ordering::Release);
        self.query_epoch.invalidate();
        self.wake_root_context_setup_waiters();
        self.release_if_ready();
    }

    fn release_if_ready(&self) {
        if !self.release_requested.load(Ordering::Acquire) || !self.query_epoch.is_idle() {
            return;
        }
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let root_context = take_owned_for_cleanup(&self.root_context, "root context");
        let envelope = take_owned_for_cleanup(&self.envelope, "admission envelope");
        let completion = take_owned_for_cleanup(&self.completion, "query completion");
        let service_epoch_user = take_owned_for_cleanup(&self.service_epoch_user, "service epoch user");
        #[cfg(test)]
        let release_commit_gate = take_owned_for_cleanup(&self.release_commit_gate, "release commit gate");
        let terminal_domain_usage = take_owned_for_cleanup(&self.terminal_domain_usage, "terminal domain usage");
        let service = take_owned_for_cleanup(&self.service, "query service")
            .expect("query handle lifetime retains its service until the one release path");
        let mut release = QueryHandleRelease {
            service: Some(service),
            admission_request_id: self.admission_request_id,
            observation: self.observation.clone(),
            completion,
            service_epoch_user,
            device_ordinal: self.device_ordinal,
            admitted_requirement: self.admitted_requirement.clone(),
            compatible_devices: self.compatible_devices.clone(),
            root_context,
            envelope,
            #[cfg(test)]
            release_commit_gate,
            terminal_domain_usage,
            outcome_terminal_domain_usage: None,
            terminal_outcome: RuntimeTerminalOutcome::Dropped,
            release_started: false,
            internal_cleanup_finished: false,
        };
        release.begin_query_cleanup();
        release.resolve_terminal_outcome_before_gpu_cleanup();
        let requires_gpu_cleanup = release.requires_gpu_cleanup();
        let pool = if requires_gpu_cleanup {
            release
                .service
                .as_ref()
                .expect("query handle release retains its service until cleanup completion")
                .resources
                .as_ref()
                .and_then(|resources| resources.gpu_execution_pool_for_device(self.device_ordinal).ok())
        } else {
            None
        };
        let release = QueryHandleReleaseGuard::new(release);
        // An admitted query that never acquired a runtime context owns no
        // CUDA-bound state. Its grant and device slot are ledger resources,
        // so returning them must not wait behind unrelated GPU execution.
        // This is important for callers that reserve admission before
        // execution: releasing such a reservation is itself the event that
        // unblocks the next query.
        if !requires_gpu_cleanup {
            release.finish();
            return;
        }
        match pool {
            Some(pool) => {
                let observation = self.observation.clone();
                if let Err(error) = pool.spawn_release_cleanup_observed_with_completion(
                    observation,
                    async move { release.finish() },
                    || {},
                ) {
                    // The rejected task drops its release guard, which drives
                    // the same consuming cleanup path synchronously. Do not
                    // create an unbounded cleanup queue merely to hide a
                    // saturated or closing bounded executor.
                    eprintln!(
                        "[nexus-query-engine] warning: GPU cleanup submission was rejected; releasing inline: {error}"
                    );
                }
            },
            None => {
                release.finish();
            },
        }
    }

    fn released_error(&self) -> QueryError {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.context",
            "query service handle was released before context acquisition",
        )
        .with_fact("admission_request_id", self.admission_request_id)
        .with_fact("reason", "query_handle_released")
    }

    fn service(&self) -> Result<Arc<QueryServiceInner>> {
        self.service
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.context",
                    "query service handle mutex was poisoned",
                )
                .with_fact("reason", "query_service_handle_poisoned")
            })?
            .as_ref()
            .cloned()
            .ok_or_else(|| self.released_error())
    }

    fn observation_owner(&self) -> Option<RuntimeObservationOwner> {
        match self.completion.lock() {
            Ok(completion) => completion.as_ref().and_then(QueryCompletion::observation_owner),
            Err(error) => error.into_inner().as_ref().and_then(QueryCompletion::observation_owner),
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ReleaseCommitGate {
    entered: std::sync::atomic::AtomicUsize,
    gate: std::sync::Barrier,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct RootContextBindingGate {
    entered: std::sync::atomic::AtomicUsize,
    gate: std::sync::Barrier,
}

#[cfg(test)]
impl RootContextBindingGate {
    pub(crate) fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire) > 0
    }

    pub(crate) fn open(&self) {
        self.gate.wait();
    }
}

#[cfg(test)]
impl ReleaseCommitGate {
    /// Whether the release commit has reached the pre-seal barrier.
    pub(crate) fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire) > 0
    }

    /// Releases the commit to run the seal and the capacity release.
    pub(crate) fn open(&self) {
        self.gate.wait();
    }
}

fn take_owned_for_cleanup<T>(owned: &Mutex<Option<T>>, resource: &'static str) -> Option<T> {
    match owned.lock() {
        Ok(mut owned) => owned.take(),
        Err(error) => {
            eprintln!("[nexus-query-engine] warning: recovering poisoned query service {resource}");
            error.into_inner().take()
        },
    }
}

struct QueryHandleRelease {
    service: Option<Arc<QueryServiceInner>>,
    admission_request_id: AdmissionRequestId,
    observation: RuntimeObservationHandle,
    completion: Option<QueryCompletion>,
    service_epoch_user: Option<QueryServiceEpochUser>,
    device_ordinal: usize,
    admitted_requirement: AttemptRequirement,
    compatible_devices: Vec<usize>,
    root_context: Option<ExecutionContext>,
    envelope: Option<super::service::transaction::AdmissionEnvelope>,
    #[cfg(test)]
    release_commit_gate: Option<Arc<ReleaseCommitGate>>,
    terminal_domain_usage: Option<cudf::QueryMemoryDomainUsage>,
    /// Sampled after the terminal outcome is resolved but before GPU cleanup
    /// begins. The timestamp is obtained only from an admitted attempt clock.
    outcome_terminal_domain_usage: Option<AttemptTimedDomainUsage>,
    terminal_outcome: RuntimeTerminalOutcome,
    release_started: bool,
    internal_cleanup_finished: bool,
}

/// The P2.7 report commit: disposition decision, health commit, and
/// observation seal, always before capacity release.
struct QueryHandleReportCommit {
    service: Option<Arc<QueryServiceInner>>,
    admission_request_id: AdmissionRequestId,
    observation: RuntimeObservationHandle,
    device_ordinal: usize,
    admitted_requirement: AttemptRequirement,
    compatible_devices: Vec<usize>,
    domain: Option<super::super::attempt_domain::AttemptAllocationDomain>,
    memory_grant: Option<MemoryGrant>,
    allocator_domain_attempt: Option<AllocatorDomainAttempt>,
    outcome_terminal_domain_usage: Option<AttemptTimedDomainUsage>,
    #[cfg(test)]
    release_commit_gate: Option<Arc<ReleaseCommitGate>>,
    terminal_domain_usage: Option<cudf::QueryMemoryDomainUsage>,
}

impl QueryHandleReportCommit {
    fn run(
        self,
        mut external_completion: Option<QueryCompletion>,
        terminal_outcome: RuntimeTerminalOutcome,
        evidence: super::service::disposition::CleanupEvidence,
    ) -> Option<super::admission::QueryCompletionActivityTail> {
        use super::service::disposition::{CapacityCommitDisposition, CleanupEvidence, decide_capacity_commit};
        let allocator_event_evidence = self.finish_allocator_domain_correlation(evidence);
        let decision =
            decide_capacity_commit(self.device_ordinal, self.domain.as_ref(), evidence, self.terminal_domain_usage);
        let super::service::disposition::CapacityCommitDecision { disposition, usage, native_fault_evidence } =
            decision;
        if let Some(usage) = usage.as_ref() {
            debug_assert_eq!(
                usage.domain_id,
                self.domain.as_ref().map(|domain| domain.domain_id()).unwrap_or_default()
            );
            let _ = usage.peak_bytes;
        }
        let cleanup_failure = match evidence {
            CleanupEvidence::Completed => None,
            CleanupEvidence::StreamSynchronizationFailed => {
                Some(crate::observability::CleanupFailureSummary::stream_synchronization_failed())
            },
            CleanupEvidence::CallbackAbandoned => {
                Some(crate::observability::CleanupFailureSummary::callback_abandoned())
            },
        };
        if let Some(summary) = cleanup_failure {
            // Record the canonical lifecycle fact before the bounded timeline
            // projection. Dropped timeline records cannot affect the exact
            // cleanup count or its first summary.
            self.observation.record_attempt_cleanup_failure(summary.clone());
            self.observation.record(|| {
                crate::observability::RuntimeObservationPayload::CleanupFailure(Box::new(
                    crate::observability::RuntimeCleanupFailureObservation { summary },
                ))
            });
        }
        let final_domain_usage = usage.map(crate::observability::AttemptFinalDomainUsage::from);
        // The post-cleanup usage is intentionally timed independently from
        // the pre-cleanup outcome-terminal sample. No timestamp is invented
        // for a release without an admitted attempt-clock binding.
        let post_cleanup_usage = external_completion
            .as_ref()
            .and_then(QueryCompletion::attempt_monotonic_ns)
            .zip(final_domain_usage)
            .map(|(sampled_ns, usage)| AttemptTimedDomainUsage { sampled_ns, usage });
        // Commit any device or backend health disposition before taking the
        // atomic ledger snapshot. The resource evidence sequence therefore
        // reflects the state which governed report publication and reuse.
        let capacity_health_committed = self.commit_capacity_health(&disposition);
        let alternate_device_eligibility = self.alternate_device_eligibility_after_quarantine(
            &disposition,
            native_fault_evidence,
            capacity_health_committed,
        );
        let resource_evidence = self.resource_evidence(
            external_completion.as_ref().and_then(QueryCompletion::attempt_monotonic_ns),
            post_cleanup_usage,
            native_fault_evidence,
            allocator_event_evidence.as_ref(),
        );
        // This is the sole P2 projection of the exact final usage sample and
        // health transition above. P3 extends this commit with bounded typed
        // fault/retry evidence; it must not resample these terminal facts.
        let terminal_commit = match (&disposition, evidence) {
            (CapacityCommitDisposition::HealthyReuse, CleanupEvidence::Completed) => {
                crate::observability::AttemptTerminalCommit::new(
                    crate::observability::AttemptCapacityDisposition::Released,
                    final_domain_usage,
                    crate::observability::AttemptHealthTransition::HealthyReuse,
                )
            },
            (CapacityCommitDisposition::HealthyReuse, _) => crate::observability::AttemptTerminalCommit::new(
                crate::observability::AttemptCapacityDisposition::Released,
                final_domain_usage,
                crate::observability::AttemptHealthTransition::HealthyReuse,
            ),
            (
                CapacityCommitDisposition::DeviceQuarantine { .. } | CapacityCommitDisposition::BackendUnhealthy { .. },
                CleanupEvidence::Completed,
            ) => crate::observability::AttemptTerminalCommit::new(
                crate::observability::AttemptCapacityDisposition::RetainedQuarantine,
                final_domain_usage,
                match disposition {
                    CapacityCommitDisposition::DeviceQuarantine { .. } => {
                        crate::observability::AttemptHealthTransition::DeviceQuarantined
                    },
                    CapacityCommitDisposition::BackendUnhealthy { .. } => {
                        crate::observability::AttemptHealthTransition::BackendUnhealthy
                    },
                    CapacityCommitDisposition::HealthyReuse => unreachable!(),
                },
            ),
            (
                CapacityCommitDisposition::DeviceQuarantine { .. } | CapacityCommitDisposition::BackendUnhealthy { .. },
                _,
            ) => crate::observability::AttemptTerminalCommit::new(
                crate::observability::AttemptCapacityDisposition::RetainedQuarantine,
                final_domain_usage,
                match disposition {
                    CapacityCommitDisposition::DeviceQuarantine { .. } => {
                        crate::observability::AttemptHealthTransition::DeviceQuarantined
                    },
                    CapacityCommitDisposition::BackendUnhealthy { .. } => {
                        crate::observability::AttemptHealthTransition::BackendUnhealthy
                    },
                    CapacityCommitDisposition::HealthyReuse => unreachable!(),
                },
            ),
        };
        let terminal_commit = native_fault_evidence
            .map(|evidence| terminal_commit.with_native_fault_evidence(evidence))
            .unwrap_or(terminal_commit);
        let terminal_commit = resource_evidence
            .map(|evidence| terminal_commit.with_resource_evidence(evidence))
            .unwrap_or(terminal_commit);
        let terminal_commit = terminal_commit.with_alternate_device_eligibility(alternate_device_eligibility);
        if let Some(error) = external_completion
            .as_mut()
            .and_then(|completion| completion.finalize_attempt_after_gpu_cleanup(terminal_commit).err())
        {
            // A linear finalizer can never be replaced with a second set of
            // facts. Close admission, suppress the untruthful report, and let
            // the normal post-accounting tail release activity after capacity.
            if let Some(service) = self.service.as_ref() {
                service.close_unknown_isolation_admission(format!("attempt finalization failed: {error}"));
            }
            eprintln!(
                "[nexus-query-engine] warning: failed to consume attempt finalizer after cleanup; suppressing report publication: {error}"
            );
            return external_completion.map(QueryCompletion::discard_unfinalized_after_gpu_cleanup);
        }
        // The release is committed: record the terminal admission lifecycle
        // event before the seal so the sealed report retains it. The
        // mechanical capacity release still happens only after the seal.
        if let Some(service) = self.service.as_ref() {
            let queue_depth = service.admission.lock().map(|state| state.queued.len()).unwrap_or_else(|error| {
                eprintln!("[nexus-query-engine] warning: failed to read poisoned admission queue depth");
                error.into_inner().queued.len()
            });
            record_admission_lifecycle(
                &self.observation,
                self.admission_request_id,
                AdmissionLifecycleState::Released,
                queue_depth,
                Some(self.device_ordinal),
                None,
            );
        }
        // Deterministic-barrier hook: tests prove cleanup-complete without
        // report-ready cannot admit a replacement.
        #[cfg(test)]
        if let Some(gate) = self.release_commit_gate {
            gate.entered.fetch_add(1, Ordering::AcqRel);
            gate.gate.wait();
        }
        external_completion.map(|completion| completion.seal_after_gpu_cleanup(terminal_outcome))
    }

    /// Commits the selected health disposition before the resource evidence is
    /// sampled or the attempt finalizer can publish a report.
    fn commit_capacity_health(&self, disposition: &super::service::disposition::CapacityCommitDisposition) -> bool {
        use super::service::disposition::CapacityCommitDisposition;

        match disposition {
            CapacityCommitDisposition::HealthyReuse => true,
            CapacityCommitDisposition::DeviceQuarantine { reason } => {
                // The device closes BEFORE the resource snapshot and report
                // publication. Its ledger sequence therefore records the
                // closed health state that governed capacity reuse.
                let Some(service) = self.service.as_ref() else {
                    return false;
                };
                let Some(resources) = service.resources.as_ref() else {
                    return false;
                };
                let Ok(device) = resources.devices.context_by_ordinal(self.device_ordinal) else {
                    return false;
                };
                let ledger = device.ledger();
                if let Err(error) = ledger.close_admission(reason.clone()) {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to close device admission during release commit: {error}"
                    );
                    return false;
                }
                let committed =
                    matches!(ledger.health(), Ok(crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. }));
                if !committed {
                    eprintln!(
                        "[nexus-query-engine] warning: device admission closure was not observable during release commit"
                    );
                    return false;
                }
                service.notify_admission_available();
                true
            },
            CapacityCommitDisposition::BackendUnhealthy { reason } => {
                if let Some(service) = self.service.as_ref() {
                    service.close_unknown_isolation_admission(reason);
                }
                true
            },
        }
    }

    /// Checks the frozen compatible-device set after the failed device has
    /// been quarantined. This is a bounded proof for report advice only: a
    /// later admission must independently re-check placement and capacity.
    fn alternate_device_eligibility_after_quarantine(
        &self,
        disposition: &super::service::disposition::CapacityCommitDisposition,
        native_fault_evidence: Option<crate::observability::AttemptNativeFaultEvidence>,
        capacity_health_committed: bool,
    ) -> crate::observability::AttemptAlternateDeviceEligibility {
        use super::service::disposition::CapacityCommitDisposition;
        use crate::observability::{AttemptAlternateDeviceEligibility, has_proven_device_local_fault_evidence};

        let proven_device_local_fault = matches!(disposition, CapacityCommitDisposition::DeviceQuarantine { .. })
            && native_fault_evidence.as_ref().is_some_and(has_proven_device_local_fault_evidence);
        if !proven_device_local_fault {
            return AttemptAlternateDeviceEligibility::NotApplicable;
        }
        if !capacity_health_committed {
            // A device-local fault is insufficient by itself: retrying on an
            // alternate also requires proof that the failed device was
            // actually removed from admission before this report seals.
            return AttemptAlternateDeviceEligibility::Unavailable;
        }

        let Some(service) = self.service.as_ref() else {
            return AttemptAlternateDeviceEligibility::Unavailable;
        };
        for ordinal in self.compatible_devices.iter().copied().filter(|ordinal| *ordinal != self.device_ordinal) {
            match &service.resources {
                Some(resources) => {
                    let Ok(device) = resources.devices.context_by_ordinal(ordinal) else {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    };
                    // The ledger is the scheduler's health authority, but it
                    // must first fold any sibling-owned persistent CUDA fault
                    // into that authority before advice calls an alternate
                    // currently healthy.
                    if device.synchronize_native_admission_health().is_err() {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    }
                    let ledger = device.ledger();
                    let Ok(required_floor) = ledger.required_floor_bytes_for(&self.admitted_requirement) else {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    };
                    let Ok(can_ever_fit) = ledger.can_ever_fit_required_floor_bytes(required_floor) else {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    };
                    if can_ever_fit && matches!(ledger.health(), Ok(crate::runtime::gpu::DeviceAdmissionHealth::Open)) {
                        return AttemptAlternateDeviceEligibility::Eligible;
                    }
                },
                None => {
                    let Some(ledger) = service.fallback_device_ledger.as_ref() else {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    };
                    if ledger.device_ordinal() != ordinal {
                        return AttemptAlternateDeviceEligibility::NoEligibleAlternate;
                    }
                    let Ok(required_floor) = ledger.required_floor_bytes_for(&self.admitted_requirement) else {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    };
                    let Ok(can_ever_fit) = ledger.can_ever_fit_required_floor_bytes(required_floor) else {
                        return AttemptAlternateDeviceEligibility::Unavailable;
                    };
                    if can_ever_fit && matches!(ledger.health(), Ok(crate::runtime::gpu::DeviceAdmissionHealth::Open)) {
                        return AttemptAlternateDeviceEligibility::Eligible;
                    }
                },
            }
        }
        AttemptAlternateDeviceEligibility::NoEligibleAlternate
    }

    /// Constructs the one bounded P3.4 resource block. Immutable grant/domain
    /// facts remain attempt-exact even if cleanup lost device isolation; every
    /// device-local snapshot and allocator record stays unavailable in that
    /// case instead of being associated with a guessed device.
    fn resource_evidence(
        &self,
        device_sample_ns: Option<u64>,
        post_cleanup_usage: Option<AttemptTimedDomainUsage>,
        native_fault_evidence: Option<crate::observability::AttemptNativeFaultEvidence>,
        allocator_event_evidence: Option<&AllocatorEventEvidenceRead>,
    ) -> Option<AttemptResourceEvidence> {
        let (memory_grant, domain) = self.memory_grant.as_ref().zip(self.domain.as_ref())?;
        let mut evidence = AttemptResourceEvidence::unavailable();
        evidence.grant_id = AttemptEvidenceField::attempt_exact(memory_grant.grant_id());
        evidence.grant_cap_bytes = AttemptEvidenceField::attempt_exact(memory_grant.budget_bytes());
        evidence.domain_id = AttemptEvidenceField::attempt_exact(domain.domain_id());
        evidence.domain_cap_bytes = AttemptEvidenceField::attempt_exact(domain.cap_bytes());
        if let Some(usage) = self.outcome_terminal_domain_usage {
            evidence.outcome_terminal_usage = AttemptEvidenceField::sampled_after_failure(usage);
        }
        if let Some(usage) = post_cleanup_usage {
            evidence.post_cleanup_usage = AttemptEvidenceField::sampled_after_failure(usage);
        }

        // Unknown isolation deliberately retains no device-local ledger,
        // profile, or ring sample. Immutable attempt facts above remain useful
        // without inventing a device association.
        if matches!(
            native_fault_evidence,
            Some(crate::observability::AttemptNativeFaultEvidence { isolation: AttemptFaultIsolation::Unknown, .. })
        ) {
            return Some(evidence);
        }
        let Some(service) = self.service.as_ref() else {
            return Some(evidence);
        };
        let Some(resources) = service.resources.as_ref() else {
            return Some(evidence);
        };
        let Ok(device) = resources.devices.context_by_ordinal(self.device_ordinal) else {
            return Some(evidence);
        };
        let profile = device.ledger().profile();
        evidence.configured_limit_bytes = AttemptEvidenceField::attempt_exact(profile.configured_limit_bytes);
        evidence.headroom_bytes = AttemptEvidenceField::attempt_exact(profile.headroom_bytes);
        evidence.backend_reserve_bytes = AttemptEvidenceField::attempt_exact(profile.backend_reserve_bytes);
        evidence.managed_capacity_bytes = AttemptEvidenceField::attempt_exact(profile.managed_capacity_bytes);
        if let (Some(sampled_ns), Ok(snapshot)) = (device_sample_ns, device.ledger().snapshot()) {
            evidence.ledger_sequence = AttemptEvidenceField::device_snapshot(snapshot.sequence);
            evidence.ledger_sample_ns = AttemptEvidenceField::device_snapshot(sampled_ns);
            evidence.live_grants_bytes = AttemptEvidenceField::device_snapshot(snapshot.live_grants_bytes);
            evidence.live_reserved_bytes = AttemptEvidenceField::device_snapshot(snapshot.live_reserved_bytes);
            evidence.protected_slot = AttemptEvidenceField::device_snapshot(snapshot.protected_slot);
            evidence.protected_reserve_bytes = AttemptEvidenceField::device_snapshot(snapshot.protected_reserve_bytes);
            evidence.cache_evictable_resident_bytes =
                AttemptEvidenceField::device_snapshot(snapshot.cache_evictable_resident_bytes);
            evidence.cache_pinned_in_use_bytes =
                AttemptEvidenceField::device_snapshot(snapshot.cache_pinned_in_use_bytes);
            evidence.active_attempt_count =
                AttemptEvidenceField::device_snapshot(u64::try_from(snapshot.active_attempts).unwrap_or(u64::MAX));
        }
        if let Some(allocator_event_evidence) = allocator_event_evidence {
            if let Some(event) = allocator_event_evidence.exact {
                evidence.allocator_event = AttemptEvidenceField::attempt_exact(Self::allocator_event(event, true));
            } else if let (Some(sampled_ns), Some(event)) = (device_sample_ns, allocator_event_evidence.unattributed) {
                evidence.unattributed_device_allocator_event =
                    AttemptEvidenceField::device_snapshot(AttemptUnattributedAllocatorEvent {
                        device_ordinal: self.device_ordinal,
                        sampled_ns,
                        event: Self::allocator_event(event, false),
                    });
            }
        }
        Some(evidence)
    }

    fn allocator_event(event: cudf::AllocatorEvent, attempt_exact: bool) -> AttemptAllocatorEvent {
        let source = match event.source {
            cudf::DeviceMemoryFailureSource::Physical => AttemptAllocatorEventSource::Physical,
            cudf::DeviceMemoryFailureSource::ReservationLimit => AttemptAllocatorEventSource::ReservationLimit,
            cudf::DeviceMemoryFailureSource::Unknown(_) => AttemptAllocatorEventSource::Unknown,
        };
        let action = event.action.map(|action| match action {
            cudf::AllocatorEventAction::Decline => AttemptAllocatorEventAction::Decline,
            cudf::AllocatorEventAction::Retry => AttemptAllocatorEventAction::Retry,
            cudf::AllocatorEventAction::TrimAndRetry => AttemptAllocatorEventAction::TrimAndRetry,
            cudf::AllocatorEventAction::CallbackPanicked => AttemptAllocatorEventAction::CallbackPanicked,
            cudf::AllocatorEventAction::Unknown(_) => AttemptAllocatorEventAction::Unknown,
        });
        let pool_usage = event.pool_usage.map(|usage| AttemptAllocatorPoolUsage {
            reserved_current_bytes: usage.reserved_current_bytes,
            reserved_high_bytes: usage.reserved_high_bytes,
            used_current_bytes: usage.used_current_bytes,
            used_high_bytes: usage.used_high_bytes,
        });
        AttemptAllocatorEvent {
            sequence: event.sequence,
            monotonic_time_ns: Self::allocator_event_field(event.monotonic_time_ns, attempt_exact),
            requested_bytes: event.requested_bytes,
            source,
            action: Self::allocator_event_field(action, attempt_exact),
            callback_attempt: event.callback_attempt,
            trim_retry_count: event.trim_retry_count,
            pool_usage: Self::allocator_event_field(pool_usage, attempt_exact),
        }
    }

    fn allocator_event_field<T: Copy>(value: Option<T>, attempt_exact: bool) -> AttemptEvidenceField<T> {
        match (value, attempt_exact) {
            (Some(value), true) => AttemptEvidenceField::attempt_exact(value),
            (Some(value), false) => AttemptEvidenceField::device_snapshot(value),
            (None, _) => AttemptEvidenceField::unavailable(),
        }
    }

    /// Keeps a closed, bounded interval only after the stream/domain cleanup
    /// callback proves completion. Any failed or abandoned cleanup is an
    /// explicit unattributed state rather than a guessed retired watermark.
    fn finish_allocator_domain_correlation(
        &self,
        evidence: super::service::disposition::CleanupEvidence,
    ) -> Option<AllocatorEventEvidenceRead> {
        let service = self.service.as_ref()?;
        let binding = self.allocator_domain_attempt?;
        if evidence == super::service::disposition::CleanupEvidence::Completed
            && let Some(domain) = self.domain.as_ref()
        {
            Some(retire_admitted_domain(&service.allocator_event_correlation, binding, domain.as_cudf()))
        } else {
            discard_admitted_domain(&service.allocator_event_correlation, binding);
            None
        }
    }
}

/// Owns a release until either CUDA cleanup starts or a submission/cancellation
/// path drops the task. The fallback is deliberately consuming: a rejected GPU
/// executor submission cannot strand an admitted handle in the active count.
struct QueryHandleReleaseGuard {
    release: Option<QueryHandleRelease>,
}

impl QueryHandleReleaseGuard {
    fn new(release: QueryHandleRelease) -> Self {
        Self { release: Some(release) }
    }

    fn finish(mut self) {
        if let Some(release) = self.release.take() {
            release.finish();
        }
    }
}

impl Drop for QueryHandleReleaseGuard {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| release.finish()));
        }
    }
}

impl QueryHandleRelease {
    fn requires_gpu_cleanup(&self) -> bool {
        if self.root_context.is_some() {
            return true;
        }
        // A device-backed admission envelope owns the root stream and
        // allocation domain from admission time (P2.6). Their destruction
        // must run on the device executor even when execution never reached
        // context acquisition. Ledger-only envelopes own no CUDA state.
        self.envelope.as_ref().is_some_and(super::service::transaction::AdmissionEnvelope::holds_gpu_resources)
    }

    fn begin_query_cleanup(&self) {
        if let Some(completion) = &self.completion {
            completion.begin_query_cleanup();
        }
    }

    /// Establish an unterminalized attempt's fallback before any GPU cleanup
    /// task is submitted. The stored value is the exact terminal outcome used
    /// after cleanup for report publication and capacity accounting.
    fn resolve_terminal_outcome_before_gpu_cleanup(&mut self) {
        self.terminal_outcome = self.completion.as_ref().map_or(RuntimeTerminalOutcome::Dropped, |completion| {
            completion.resolve_terminal_outcome_for_cleanup(RuntimeTerminalOutcome::Dropped)
        });
        self.capture_outcome_terminal_domain_usage();
    }

    /// Captures the outcome-terminal usage after the attempt terminal boundary
    /// is fixed and before the release path can start GPU cleanup. A failed
    /// usage read remains unavailable; no synthetic all-zero sample is kept.
    fn capture_outcome_terminal_domain_usage(&mut self) {
        let Some(sampled_ns) = self.completion.as_ref().and_then(QueryCompletion::attempt_monotonic_ns) else {
            return;
        };
        let Some(domain) = self.envelope.as_ref().and_then(|envelope| envelope.domain()) else {
            return;
        };
        self.outcome_terminal_domain_usage =
            domain.usage().ok().map(|usage| AttemptTimedDomainUsage { sampled_ns, usage: usage.into() });
    }

    fn finish(mut self) {
        self.start_release();
    }

    fn start_release(&mut self) {
        if self.release_started {
            self.complete_without_stream_cleanup();
            return;
        }
        self.release_started = true;
        if let Some(envelope) = self.envelope.as_mut() {
            envelope.begin_terminal_cleanup();
        }
        let root_context = self.root_context.take();
        if root_context.as_ref().is_some_and(ExecutionContext::has_root_stream_lease) {
            let root_context = root_context.expect("root context with a stream lease is present");
            let external_completion = self.take_external_completion();
            let commit = self.prepare_report_commit();
            let terminal_outcome = self.terminal_outcome.clone();
            let internal_cleanup =
                QueryHandleReleaseGuard::new(std::mem::replace(self, Self::empty_recovered_release()));
            // P2.7 report-before-reuse: the health commit, usage sample, and
            // observation seal run BEFORE the capacity release callback.
            let activity_tail = Arc::new(Mutex::new(None));
            let activity_tail_slot = Arc::clone(&activity_tail);
            let installed = root_context.install_root_stream_cleanup_completion(
                move |outcome| {
                    *activity_tail_slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = commit.run(
                        external_completion,
                        terminal_outcome,
                        super::service::disposition::CleanupEvidence::from(outcome),
                    )
                },
                move || {
                    internal_cleanup.finish();
                    if let Some(tail) = activity_tail.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
                        tail.finish();
                    }
                },
            );
            if !installed {
                eprintln!("[nexus-query-engine] warning: root stream cleanup completion was already installed");
            }
            drop_recovering(Some(root_context), "root execution context");
            return;
        }
        drop(root_context);
        // An admitted attempt whose context was never acquired still owns the
        // envelope's root stream. Destroy the member and domain, then use the
        // root stream cleanup
        // completion as the same post-accounting fence used by the
        // root-context path. Publishing external activity from the GPU job
        // itself would let a reentrant backend close wait on that
        // still-active job.
        let commit = self.prepare_report_commit();
        if let Some(stream_lease) = self.envelope.as_mut().and_then(|envelope| envelope.release_unconsumed_physical()) {
            let external_completion = self.take_external_completion();
            let terminal_outcome = self.terminal_outcome.clone();
            let internal_cleanup =
                QueryHandleReleaseGuard::new(std::mem::replace(self, Self::empty_recovered_release()));
            let activity_tail = Arc::new(Mutex::new(None));
            let activity_tail_slot = Arc::clone(&activity_tail);
            let installed = stream_lease.install_cleanup_completion(
                move |outcome| {
                    *activity_tail_slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = commit.run(
                        external_completion,
                        terminal_outcome,
                        super::service::disposition::CleanupEvidence::from(outcome),
                    )
                },
                move || {
                    internal_cleanup.finish();
                    if let Some(tail) = activity_tail.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
                        tail.finish();
                    }
                },
            );
            if !installed {
                eprintln!("[nexus-query-engine] warning: root stream cleanup completion was already installed");
            }
            drop_recovering(Some(stream_lease), "admission envelope root stream lease");
            return;
        }
        // Ledger-only or context-consumed release: no stream fence remains.
        // The report commit still precedes the capacity release.
        let external_completion = self.take_external_completion();
        let activity_tail = commit.run(
            external_completion,
            self.terminal_outcome.clone(),
            super::service::disposition::CleanupEvidence::Completed,
        );
        self.complete_without_stream_cleanup();
        if let Some(tail) = activity_tail {
            tail.finish();
        }
    }

    fn empty_recovered_release() -> Self {
        Self {
            service: None,
            admission_request_id: AdmissionRequestId::new(0),
            observation: RuntimeObservationHandle::disabled(),
            completion: None,
            service_epoch_user: None,
            device_ordinal: 0,
            admitted_requirement: AttemptRequirement::empty(),
            compatible_devices: Vec::new(),
            root_context: None,
            envelope: None,
            #[cfg(test)]
            release_commit_gate: None,
            terminal_domain_usage: None,
            outcome_terminal_domain_usage: None,
            terminal_outcome: RuntimeTerminalOutcome::Dropped,
            release_started: true,
            internal_cleanup_finished: true,
        }
    }

    fn take_external_completion(&mut self) -> Option<QueryCompletion> {
        self.completion.take()
    }

    /// Captures everything the report commit needs after GPU cleanup
    /// completes: the service for health plumbing, the device, and the
    /// attempt domain clone for the final usage sample.
    fn prepare_report_commit(&mut self) -> QueryHandleReportCommit {
        let (domain, memory_grant, allocator_domain_attempt) =
            self.envelope.as_mut().map_or((None, None, None), |envelope| {
                let domain = envelope.domain().cloned();
                let memory_grant = Some(envelope.memory_grant().clone());
                let allocator_domain_attempt = envelope.take_allocator_domain_attempt();
                (domain, memory_grant, allocator_domain_attempt)
            });
        QueryHandleReportCommit {
            service: self.service.clone(),
            admission_request_id: self.admission_request_id,
            observation: self.observation.clone(),
            device_ordinal: self.device_ordinal,
            admitted_requirement: self.admitted_requirement.clone(),
            compatible_devices: self.compatible_devices.clone(),
            domain,
            memory_grant,
            allocator_domain_attempt,
            outcome_terminal_domain_usage: self.outcome_terminal_domain_usage.take(),
            #[cfg(test)]
            release_commit_gate: self.release_commit_gate.take(),
            terminal_domain_usage: self.terminal_domain_usage.take(),
        }
    }

    fn complete_without_stream_cleanup(&mut self) {
        self.complete_internal_cleanup();
    }

    fn complete_internal_cleanup(&mut self) {
        if self.internal_cleanup_finished {
            return;
        }
        // The envelope drops physical remnants, the attempt lane, immutable
        // grant, backend-generation permit, and lifecycle ticket in that
        // order. The final MemoryGrant drop invokes the ledger release hook
        // that reacquires the admission mutex, so it must never happen while
        // that mutex is held.
        drop_recovering(self.envelope.take(), "admission envelope");
        // Publish query quiescence only after this handle no longer retains
        // the service epoch. `close_and_wait_for_quiescence` uses the active
        // query count as its completion fence, so waking it earlier could let
        // a replacement race the final governor-claim release.
        drop_recovering(self.service_epoch_user.take(), "query service epoch user");
        let Some(service) = self.service.take() else {
            self.internal_cleanup_finished = true;
            return;
        };
        // The Released lifecycle event was recorded by the report commit
        // before the seal; only mechanical capacity accounting remains here.
        // Finished handles retain only projections and the service reference.
        // The envelope already released the admission resources and its final
        // lifecycle fence before this notification can publish quiescence.
        notify_shared_device_admission_available(self.device_ordinal);
        service.notify_admission_available();
        drop_recovering(Some(service), "query service");
        self.internal_cleanup_finished = true;
    }
}

impl Drop for QueryHandleRelease {
    fn drop(&mut self) {
        if self.internal_cleanup_finished {
            return;
        }

        // A panic may reach the recovery destructor before `start_release`.
        // Preserve the envelope-owned cleanup fence instead of leaking an
        // Active ticket in the tracker.
        if let Some(envelope) = self.envelope.as_mut() {
            envelope.begin_terminal_cleanup();
        }

        // A cancellation or panic can drop a queued GPU/stream callback before
        // it invokes its body: cleanup completion is unproven and fault
        // isolation is unknown, so the commit marks the backend unhealthy
        // before the seal and only then releases capacity.
        let commit = self.prepare_report_commit();
        let activity_tail = commit.run(
            self.take_external_completion(),
            self.terminal_outcome.clone(),
            super::service::disposition::CleanupEvidence::CallbackAbandoned,
        );
        self.complete_internal_cleanup();
        if let Some(tail) = activity_tail {
            tail.finish();
        }
    }
}

fn drop_recovering<T>(value: Option<T>, resource: &'static str) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value))).is_err() {
        eprintln!("[nexus-query-engine] warning: query service {resource} drop panicked during cleanup recovery");
    }
}

/// RAII admission handle for one native query.
#[derive(Debug)]
pub struct QueryHandle {
    pub(super) config: QueryExecutionConfig,
    pub(super) queue_bounds: QueueBounds,
    /// The proof supplied by planning, retained for execution-time coverage
    /// validation.  It remains absent when admission resolved a requirement
    /// solely for placement/accounting.
    pub(super) submitted_plan_capability_evidence: Option<crate::admission::AttemptRequirement>,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) admission_sequence: u64,
    pub(super) admission_wait: Duration,
    pub(super) device_budget_bytes: u64,
    pub(super) device_ordinal: usize,
    pub(super) lifetime: Arc<QueryHandleLifetime>,
}

impl QueryHandle {
    /// Stable service-local request id assigned for this admission attempt.
    pub fn admission_request_id(&self) -> AdmissionRequestId {
        self.admission_request_id
    }

    /// Monotonic service-local admission sequence assigned when this handle was admitted.
    pub fn admission_sequence(&self) -> u64 {
        self.admission_sequence
    }

    /// Time spent waiting in the service admission queue.
    pub fn admission_wait(&self) -> Duration {
        self.admission_wait
    }

    /// Device-memory budget slice assigned to this query.
    pub fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// CUDA device ordinal selected for this query.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Bounded memory policy this handle will use for native execution.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.config.memory_policy_for_admitted_query(self.device_budget_bytes)
    }

    /// Boundary policy copied into the admitted execution context.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.config.boundary_policy
    }

    /// Source scheduler policy copied into the admitted execution context.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.config.source_scheduler_policy
    }

    /// Source chunk planning mode copied into the admitted execution context.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.config.source_chunk_planning_mode
    }

    /// Lakehouse cache-fill resource policy assigned to this admitted query.
    pub fn lakehouse_cache_fill_config(&self) -> LakehouseCacheFillConfig {
        self.config.lakehouse_cache_fill.clone()
    }

    /// Floating statistical aggregate execution contract assigned to this admitted query.
    pub fn statistical_aggregate_execution_mode(&self) -> StddevSampExecutionMode {
        self.config.stddev_samp_execution_mode
    }

    /// Floating statistical aggregate contract returned through the legacy
    /// STDDEV_SAMP accessor.
    pub fn stddev_samp_execution_mode(&self) -> StddevSampExecutionMode {
        self.statistical_aggregate_execution_mode()
    }

    /// Dynamic-filter execution policy assigned to this admitted query.
    pub fn dynamic_filter_execution_options(&self) -> DynamicFilterExecutionOptions {
        self.config.dynamic_filter_execution_options
    }

    /// Keyed-hash join strategy policy assigned to this admitted query.
    pub fn keyed_hash_join_strategy_options(&self) -> KeyedHashJoinStrategyOptions {
        self.config.keyed_hash_join_strategy_options
    }

    /// Grouped aggregate strategy execution policy assigned to this admitted query.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.config.aggregate_strategy_execution_options
    }

    /// Expert source chunk planning policy assigned to this admitted query.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.config.source_chunk_planning_options
    }

    fn context_stream_acquire_deadline(&self) -> Instant {
        Instant::now().checked_add(self.queue_bounds.max_wait).unwrap_or_else(Instant::now)
    }

    /// Acquires an admitted execution context from the service stream pool.
    pub fn acquire_context(&self) -> Result<AdmittedQueryContext> {
        self.acquire_context_with_cancellation(ExecutionCancellation::new())
    }

    /// Acquires an admitted execution context without blocking the caller's async runtime.
    pub async fn acquire_context_async(self: &Arc<Self>) -> Result<AdmittedQueryContext> {
        self.acquire_context_with_cancellation_async(ExecutionCancellation::new()).await
    }

    /// Acquires an admitted execution context on this query's device executor.
    pub async fn acquire_context_with_cancellation_async(
        self: &Arc<Self>,
        cancellation: ExecutionCancellation,
    ) -> Result<AdmittedQueryContext> {
        let operation = "nexus_query_engine.query_service.context";
        let setup_claim = self.lifetime.reserve_context_setup_async(&cancellation, operation).await?;
        let root_setup_claim = (setup_claim == ContextSetupClaim::Root)
            .then(|| RootContextSetupClaimGuard::new(Arc::clone(&self.lifetime)));
        let stream_acquire_deadline = self.context_stream_acquire_deadline();
        // Claim the attempt-owned share before submitting an auxiliary job.
        // Otherwise excess work from one attempt can fill every global
        // auxiliary executor permit while it waits for its own local limit.
        let attempt_auxiliary_permit = match setup_claim {
            ContextSetupClaim::Auxiliary => Some(
                self.lifetime
                    .attempt_auxiliary_slots
                    .acquire_with_cancellation_deadline_async(
                        &cancellation,
                        stream_acquire_deadline,
                        "nexus_query_engine.query_service.context.attempt_auxiliary_acquire",
                    )
                    .await?,
            ),
            ContextSetupClaim::Root => None,
            ContextSetupClaim::WaitForRoot => {
                unreachable!("context setup reservation resolves waits before scheduling")
            },
        };
        let pool = self.gpu_execution_pool()?;
        let job_class = match setup_claim {
            ContextSetupClaim::Root => super::super::gpu_execution_pool::GpuExecutionJobClass::Root,
            ContextSetupClaim::Auxiliary => super::super::gpu_execution_pool::GpuExecutionJobClass::Auxiliary,
            ContextSetupClaim::WaitForRoot => {
                unreachable!("context setup reservation resolves waits before scheduling")
            },
        };
        let observation = self.lifetime.observation.clone();
        let handle = Arc::clone(self);
        let result = pool
            .run_observed_scoped_in_class(
                GpuExecutionJobKind::ContextSetup,
                job_class,
                observation,
                move |stream_scope| async move {
                    let _root_setup_claim = root_setup_claim;
                    handle.acquire_context_with_cancellation_and_stream_scope(
                        cancellation,
                        stream_scope,
                        setup_claim,
                        stream_acquire_deadline,
                        attempt_auxiliary_permit,
                    )
                },
            )
            .await
            .and_then(|result| result);
        result
    }

    /// Opaque lifecycle-bound executor for synchronous GPU work on this
    /// query's selected device.
    pub fn gpu_executor(&self) -> Result<GpuExecutorLease> {
        self.gpu_execution_pool().map(GpuExecutorLease::from_pool)
    }

    pub(crate) fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let operation = "nexus_query_engine.query_service.gpu_execution_pool";
        let query_anchor = self.lifetime.acquire_resource_anchor(operation)?;
        let service = self.lifetime.service()?;
        let Some(resources) = &service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "query service was constructed without runtime resources",
            ));
        };
        let pool = resources.gpu_execution_pool_for_device(self.device_ordinal)?;
        let resource_epoch =
            QueryServiceResourceEpoch::acquire(&service.service_epoch, Some(&query_anchor), operation)?;
        Ok(pool.with_resource_epoch(resource_epoch))
    }

    /// Acquires an admitted execution context that observes caller-supplied
    /// cooperative cancellation.
    pub fn acquire_context_with_cancellation(
        &self,
        cancellation: ExecutionCancellation,
    ) -> Result<AdmittedQueryContext> {
        let operation = "nexus_query_engine.query_service.context";
        let setup_claim = self.lifetime.reserve_context_setup_sync(&cancellation, operation)?;
        let _root_setup_claim = (setup_claim == ContextSetupClaim::Root)
            .then(|| RootContextSetupClaimGuard::new(Arc::clone(&self.lifetime)));
        let stream_acquire_deadline = self.context_stream_acquire_deadline();
        self.acquire_context_with_cancellation_and_stream_scope(
            cancellation,
            crate::observability::RuntimeStreamLeaseScope::default(),
            setup_claim,
            stream_acquire_deadline,
            None,
        )
    }

    fn acquire_context_with_cancellation_and_stream_scope(
        &self,
        cancellation: ExecutionCancellation,
        stream_scope: crate::observability::RuntimeStreamLeaseScope,
        setup_claim: ContextSetupClaim,
        stream_acquire_deadline: Instant,
        attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
    ) -> Result<AdmittedQueryContext> {
        let operation = "nexus_query_engine.query_service.context";
        let query_anchor = self.lifetime.acquire_resource_anchor(operation)?;
        let service = self.lifetime.service()?;
        let Some(resources) = &service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.context",
                "query service was constructed without runtime resources",
            ));
        };
        let device = resources.devices.context_by_ordinal(self.device_ordinal)?;
        let mut root_context = self.lifetime.root_context.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.context",
                "query service root context mutex was poisoned",
            )
        })?;
        if let Some(template) = root_context.as_ref().cloned() {
            if setup_claim != ContextSetupClaim::Auxiliary {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "root context was ready without an auxiliary context-setup reservation",
                )
                .with_fact("reason", "context_setup_claim_mismatch")
                .with_fact("context_setup_claim", "root"));
            }
            // Concurrent fragments under one admitted query must not share the
            // root stream: cuDF rejects a second live query-domain budget token
            // on the same stream. Bind each secondary acquire to a dedicated
            // service stream while preserving the shared grant/domain.
            drop(root_context);
            let attempt_auxiliary_permit = match attempt_auxiliary_permit {
                Some(permit) => permit,
                None => self.lifetime.attempt_auxiliary_slots.acquire_with_cancellation_deadline(
                    &cancellation,
                    stream_acquire_deadline,
                    "nexus_query_engine.query_service.context.attempt_auxiliary_acquire",
                )?,
            };
            let stream_lease = device.stream_pool().acquire_observed_with_cancellation_deadline_and_class(
                &cancellation,
                stream_acquire_deadline,
                "nexus_query_engine.query_service.context.stream_acquire",
                self.lifetime.observation.clone(),
                stream_scope,
                crate::runtime::StreamPermitClass::Auxiliary,
            )?;
            validate_service_stream_pool_device(device.stream_pool(), stream_lease.device())?;
            let context = template
                .with_owned_stream_lease(stream_lease)?
                .into_context()
                .with_auxiliary_stream_permit_guard(attempt_auxiliary_permit)
                .with_query_resource_anchor(query_anchor);
            self.lifetime.query_epoch.check_open(operation).map_err(|_error| self.lifetime.released_error())?;
            return Ok(AdmittedQueryContext { context });
        }
        if setup_claim != ContextSetupClaim::Root {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "auxiliary context setup ran before the admitted root context was ready",
            )
            .with_fact("reason", "context_setup_claim_mismatch")
            .with_fact("context_setup_claim", "auxiliary"));
        }
        debug_assert!(attempt_auxiliary_permit.is_none(), "root context setup cannot carry an auxiliary permit");
        // The admission transaction already owns the root stream, allocation
        // domain, and root stream member (P2.6). Consume that binding; no
        // stream or domain is created after admission.
        let gpu_execution_pool = resources.gpu_execution_pool_for_device(self.device_ordinal)?;
        let (stream_lease_guard, memory_grant, runtime, attempt_allocation_domain, attempt_member_lease) = {
            let envelope = self.lifetime.lock_envelope()?;
            let envelope = envelope.as_ref().ok_or_else(|| self.lifetime.released_error())?;
            // The envelope's worker scope is the admission-time identity
            // evidence: same device, same worker pool, and the lane the
            // attempt has held since the ledger transaction committed.
            if let Some(scope) = envelope.worker_scope() {
                debug_assert_eq!(scope.device_ordinal(), self.device_ordinal);
                debug_assert_eq!(scope.lane_index(), envelope.lane_index().expect("lane matches its scope"));
                debug_assert_eq!(scope.pool_id(), gpu_execution_pool.snapshot().pool_id);
            }
            let binding = envelope.take_root_binding()?;
            validate_service_stream_pool_device(device.stream_pool(), binding.stream_lease.device())?;
            let stream_lease_guard = Arc::new(binding.stream_lease);
            let memory_grant = envelope.memory_grant().clone();
            let runtime = stream_lease_guard.shared_runtime()?;
            let attempt_allocation_domain = envelope.domain().cloned().ok_or_else(|| self.lifetime.released_error())?;
            #[cfg(test)]
            self.lifetime.wait_at_root_context_binding_gate();
            (stream_lease_guard, memory_grant, runtime, attempt_allocation_domain, binding.member_lease)
        };
        // The root execution stream binds the attempt's root domain member
        // once for the whole attempt lifetime.
        let observation_owner = self.lifetime.observation_owner();
        let query_id = self.lifetime.observation.query_id().map(|query_id| QueryId::new(query_id.as_str()));
        let trace_id =
            self.lifetime.observation.trace_id().map(|trace_id| crate::metrics::TraceId::new(trace_id.as_str()));
        let runtime_observation = self.lifetime.observation.clone();
        let context = ExecutionContext::from_parts(ExecutionContextParts {
            service_epoch: Some(service.service_epoch.clone()),
            query_epoch: Some(self.lifetime.query_epoch.clone()),
            query_anchor: None,
            io_lifecycle: None,
            query_id,
            trace_id,
            runtime_observation,
            runtime_observation_owner: observation_owner,
            runtime_execution_scope: None,
            runtime_stream_lease_scope: stream_scope,
            runtime,
            gpu_execution_pool,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy(),
            memory_grant: Some(memory_grant),
            grant_reservation_arena: None,
            grant_memory_domain: Some(Arc::clone(attempt_allocation_domain.as_cudf())),
            attempt_allocation_domain: Some(attempt_allocation_domain),
            attempt_auxiliary_slots: Some(Arc::clone(&self.lifetime.attempt_auxiliary_slots)),
            admitted_capability_evidence: self.submitted_plan_capability_evidence.clone(),
            attempt_member_lease: Some(attempt_member_lease),
            boundary_policy: service.config.boundary_policy,
            source_scheduler_policy: service.config.source_scheduler_policy,
            source_chunk_planning_mode: service.config.source_chunk_planning_mode,
            lakehouse_cache_fill_config: service.config.lakehouse_cache_fill.clone(),
            service_stream_pool: device.stream_pool().clone(),
            lakehouse_cache_fill_budget: resources.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: resources.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: super::LakehouseCacheFillEventSink::default(),
            object_store_control_runtime: Arc::clone(&resources.object_store_control_runtime),
            source_object_stores: Arc::clone(&resources.source_object_stores),
            deployment_io_capability: resources.deployment_io_capability.clone(),
            cancellation,
            stddev_samp_execution_mode: service.config.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: service.config.aggregate_strategy_execution_options,
            source_chunk_planning_options: service.config.source_chunk_planning_options,
            graph_memory_calibration_override: service.config.graph_memory_calibration_override,
            stream_lease_guard: Some(stream_lease_guard),
            auxiliary_stream_permit_guard: None,
            query_service_admitted: true,
        });
        *root_context = Some(context.clone());
        drop(root_context);
        self.lifetime.complete_root_context_setup();
        let context = context.with_query_resource_anchor(query_anchor);
        self.lifetime.query_epoch.check_open(operation).map_err(|_error| self.lifetime.released_error())?;
        Ok(AdmittedQueryContext { context })
    }

    /// Requests admission release before dropping the handle.
    ///
    /// Runtime-backed services complete GPU resource destruction on the
    /// selected device executor before waking queued admissions.
    pub fn finish(&self) {
        self.lifetime.request_release();
    }

    fn release_once(&self) {
        self.lifetime.request_release();
    }
}

impl Drop for QueryHandle {
    fn drop(&mut self) {
        self.release_once();
    }
}

/// Execution context borrowed from a [`QueryHandle`].
#[derive(Debug)]
pub struct AdmittedQueryContext {
    context: ExecutionContext,
}

impl AdmittedQueryContext {
    /// Native execution context for this admitted query.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Arc, Mutex},
    };

    use super::take_owned_for_cleanup;

    #[test]
    fn poisoned_owned_resource_is_recovered_for_ordered_cleanup() {
        let owned = Arc::new(Mutex::new(Some(7_u8)));
        let poison = Arc::clone(&owned);
        let _ = catch_unwind(AssertUnwindSafe(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison test resource");
        }));

        assert_eq!(take_owned_for_cleanup(owned.as_ref(), "test resource"), Some(7));
        assert_eq!(take_owned_for_cleanup(owned.as_ref(), "test resource"), None);
    }
}
