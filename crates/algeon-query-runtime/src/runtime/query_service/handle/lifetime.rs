//! Admitted attempt lifetime and root-context setup coordination.

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Condvar, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    super::{
        super::context::{AttemptAuxiliarySlots, QueryResourceAnchor, QueryResourceEpoch},
        *,
    },
    release::{QueryHandleRelease, QueryHandleReleaseGuard, take_owned_for_cleanup},
};
#[cfg(test)]
use crate::runtime::attempt_domain::AttemptAllocationDomain;
use crate::runtime::{AttemptStreamProviderRoute, allocator_correlation::AttemptCacheScopeAllocatorCorrelation};

const CONTEXT_SETUP_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContextSetupClaim {
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
pub(super) struct RootContextSetupClaimGuard {
    lifetime: Arc<QueryHandleLifetime>,
}

impl RootContextSetupClaimGuard {
    pub(super) fn new(lifetime: Arc<QueryHandleLifetime>) -> Self {
        Self { lifetime }
    }
}

impl Drop for RootContextSetupClaimGuard {
    fn drop(&mut self) {
        self.lifetime.rollback_root_context_setup();
    }
}

#[derive(Debug)]
pub(in crate::runtime::query_service) struct QueryHandleLifetime {
    service: Mutex<Option<Arc<QueryServiceInner>>>,
    pub(in crate::runtime::query_service) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) attempt_id: crate::observability::AttemptId,
    /// The admitted attempt owns the collector until every outward execution
    /// context has drained. Report commit then consumes its one unambiguous
    /// cache-scope event after CUDA cleanup is fenced.
    cache_scope_allocator_correlation: Mutex<AttemptCacheScopeAllocatorCorrelation>,
    completion: Mutex<Option<QueryCompletion>>,
    service_epoch_user: Mutex<Option<QueryServiceEpochUser>>,
    device_ordinal: usize,
    /// Immutable admission inputs retained only until the post-cleanup report
    /// commit can determine whether a quarantined device has a truthful
    /// alternate retry target. They never drive a second placement.
    admitted_device_requirements: BTreeMap<usize, AttemptRequirement>,
    compatible_devices: Vec<usize>,
    release_requested: AtomicBool,
    released: AtomicBool,
    pub(super) query_epoch: QueryResourceEpoch,
    pub(super) root_context: Mutex<Option<ExecutionContext>>,
    root_context_setup: RootContextSetupCoordinator,
    pub(super) attempt_auxiliary_slots: Arc<AttemptAuxiliarySlots>,
    pub(super) attempt_stream_provider_route: Mutex<Option<Arc<AttemptStreamProviderRoute>>>,
    envelope: Mutex<Option<super::super::service::transaction::AdmissionEnvelope>>,
    pub(super) admitted_capability_evidence: Option<Arc<AttemptRequirement>>,
    #[cfg(test)]
    release_commit_gate: Mutex<Option<Arc<ReleaseCommitGate>>>,
    #[cfg(test)]
    envelope_release_gate: Mutex<Option<Arc<EnvelopeReleaseGate>>>,
    #[cfg(test)]
    root_context_binding_gate: Mutex<Option<Arc<RootContextBindingGate>>>,
    terminal_domain_usage: Mutex<Option<cudf::QueryMemoryDomainUsage>>,
}

impl QueryHandleLifetime {
    #[cfg(test)]
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn admitted_device_requirement_for_test(
        &self,
        ordinal: usize,
    ) -> Option<&AttemptRequirement> {
        self.admitted_device_requirements.get(&ordinal)
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn admitted_capability_evidence_for_test(
        &self,
    ) -> Option<&AttemptRequirement> {
        self.admitted_capability_evidence.as_deref()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "service construction inputs are distinct admission and ownership values"
    )]
    pub(in crate::runtime::query_service) fn new(
        service: Arc<QueryServiceInner>,
        admission_request_id: AdmissionRequestId,
        observation: RuntimeObservationHandle,
        attempt_id: crate::observability::AttemptId,
        completion: QueryCompletion,
        envelope: super::super::service::transaction::AdmissionEnvelope,
        admitted_capability_evidence: Option<Arc<AttemptRequirement>>,
        admitted_device_requirements: BTreeMap<usize, AttemptRequirement>,
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
            let weak_lifetime = Weak::clone(weak);
            let on_query_idle = Arc::new(move || {
                if let Some(lifetime) = weak_lifetime.upgrade() {
                    lifetime.release_if_ready();
                }
            });
            Self {
                service: Mutex::new(Some(service)),
                admission_request_id,
                observation,
                attempt_id,
                cache_scope_allocator_correlation: Mutex::new(
                    AttemptCacheScopeAllocatorCorrelation::for_admitted_attempt(Some(attempt_id)),
                ),
                completion: Mutex::new(Some(completion)),
                service_epoch_user: Mutex::new(Some(service_epoch_user)),
                device_ordinal,
                admitted_device_requirements,
                compatible_devices,
                release_requested: AtomicBool::new(false),
                released: AtomicBool::new(false),
                query_epoch: QueryResourceEpoch::new(Some(on_query_idle)),
                root_context: Mutex::new(None),
                root_context_setup: RootContextSetupCoordinator::new(),
                attempt_auxiliary_slots,
                attempt_stream_provider_route: Mutex::new(None),
                envelope: Mutex::new(Some(envelope)),
                admitted_capability_evidence,
                #[cfg(test)]
                release_commit_gate: Mutex::new(None),
                #[cfg(test)]
                envelope_release_gate: Mutex::new(None),
                #[cfg(test)]
                root_context_binding_gate: Mutex::new(None),
                terminal_domain_usage: Mutex::new(None),
            }
        })
    }

    pub(super) fn lock_envelope(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<super::super::service::transaction::AdmissionEnvelope>>> {
        self.envelope.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.context",
                "query service admission envelope mutex was poisoned",
            )
        })
    }

    pub(super) fn cache_scope_allocator_correlation(&self) -> AttemptCacheScopeAllocatorCorrelation {
        self.cache_scope_allocator_correlation.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    fn take_cache_scope_allocator_correlation(&self) -> AttemptCacheScopeAllocatorCorrelation {
        match self.cache_scope_allocator_correlation.lock() {
            Ok(mut correlation) => std::mem::take(&mut *correlation),
            Err(error) => {
                eprintln!(
                    "[query-engine] warning: recovering poisoned query service cache scope allocator correlation"
                );
                std::mem::take(&mut *error.into_inner())
            },
        }
    }

    /// Attempt allocation domain clone for disposition tests.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn envelope_domain_for_test(&self) -> Option<AttemptAllocationDomain> {
        self.lock_envelope().ok()?.as_ref().and_then(|envelope| envelope.domain().cloned())
    }

    /// Admission-envelope identity evidence for transaction tests:
    /// (domain id, grant id, worker scope).
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn envelope_evidence(
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
    pub(in crate::runtime::query_service) fn arm_release_commit_gate(&self) -> Arc<ReleaseCommitGate> {
        let gate = Arc::new(ReleaseCommitGate {
            entered: std::sync::atomic::AtomicUsize::new(0),
            gate: std::sync::Barrier::new(2),
        });
        *self.release_commit_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&gate));
        gate
    }

    /// Arms a barrier that holds the attempt at the point where it still owns
    /// its admission envelope, after the report seal. The barrier fires once.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn arm_envelope_release_gate(&self) -> Arc<EnvelopeReleaseGate> {
        let gate = Arc::new(EnvelopeReleaseGate {
            entered: std::sync::atomic::AtomicUsize::new(0),
            gate: std::sync::Barrier::new(2),
        });
        *self.envelope_release_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&gate));
        gate
    }

    /// Pauses root-context setup after it has consumed the root binding while
    /// the admission envelope remains locked. This makes release/setup races
    /// deterministic without exposing a production lifecycle seam.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn arm_root_context_binding_gate(&self) -> Arc<RootContextBindingGate> {
        let gate = Arc::new(RootContextBindingGate {
            entered: std::sync::atomic::AtomicUsize::new(0),
            gate: std::sync::Barrier::new(2),
        });
        *self.root_context_binding_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Arc::clone(&gate));
        gate
    }

    #[cfg(test)]
    pub(super) fn wait_at_root_context_binding_gate(&self) {
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
    pub(in crate::runtime::query_service) fn inject_terminal_domain_usage_for_test(
        &self,
        usage: cudf::QueryMemoryDomainUsage,
    ) {
        *self.terminal_domain_usage.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(usage);
    }

    /// Installs an unrelated root-stream completion to exercise the
    /// fail-closed duplicate-handoff path. Normal query-service ownership
    /// never permits this; a later release must commit `CallbackAbandoned`
    /// before it returns admission capacity.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn install_foreign_root_stream_cleanup_completion_for_test(&self) {
        let root_context = self.root_context.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let root_context = root_context.as_ref().expect("root context is acquired before injecting cleanup collision");
        assert!(root_context.install_root_stream_cleanup_completion(|_| {}, || {}));
    }

    #[cfg(test)]
    pub(in crate::runtime::query_service) fn memory_grant(&self) -> Result<MemoryGrant> {
        self.lock_envelope()?
            .as_ref()
            .map(|envelope| envelope.memory_grant().clone())
            .ok_or_else(|| self.released_error())
    }

    pub(super) fn acquire_resource_anchor(
        self: &Arc<Self>,
        operation: &'static str,
    ) -> Result<Arc<QueryResourceAnchor>> {
        if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
            return Err(self.released_error());
        }
        let owner: Arc<dyn Send + Sync> = Arc::clone(self) as _;
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

    pub(super) fn reserve_context_setup_sync(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<ContextSetupClaim> {
        loop {
            // Context setup can wait in the device worker queue after its
            // first preflight. Recheck the attempt-wide token when that work
            // begins so an elapsed caller deadline cannot consume the root
            // binding or start a native context.
            cancellation.check(operation)?;
            match self.reserve_context_setup(operation)? {
                ContextSetupClaim::WaitForRoot => self.wait_for_root_context_setup(cancellation, operation)?,
                claim => return Ok(claim),
            }
        }
    }

    pub(super) async fn reserve_context_setup_async(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<ContextSetupClaim> {
        loop {
            // See the synchronous path: a queued GPU worker must recheck the
            // one absolute attempt deadline before it consumes context state.
            cancellation.check(operation)?;
            match self.reserve_context_setup(operation)? {
                ContextSetupClaim::WaitForRoot => {
                    self.wait_for_root_context_setup_async(cancellation, operation).await?
                },
                claim => return Ok(claim),
            }
        }
    }

    pub(super) fn complete_root_context_setup(&self) {
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

    pub(super) fn request_release(&self) {
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
        let attempt_stream_provider_route =
            take_owned_for_cleanup(&self.attempt_stream_provider_route, "attempt stream provider route");
        let envelope = take_owned_for_cleanup(&self.envelope, "admission envelope");
        let completion = take_owned_for_cleanup(&self.completion, "query completion");
        let service_epoch_user = take_owned_for_cleanup(&self.service_epoch_user, "service epoch user");
        let cache_scope_allocator_correlation = self.take_cache_scope_allocator_correlation();
        #[cfg(test)]
        let release_commit_gate = take_owned_for_cleanup(&self.release_commit_gate, "release commit gate");
        #[cfg(test)]
        let envelope_release_gate = take_owned_for_cleanup(&self.envelope_release_gate, "envelope release gate");
        let terminal_domain_usage = take_owned_for_cleanup(&self.terminal_domain_usage, "terminal domain usage");
        let service = take_owned_for_cleanup(&self.service, "query service")
            .expect("query handle lifetime retains its service until the one release path");
        let mut release = QueryHandleRelease {
            service: Some(service),
            admission_request_id: self.admission_request_id,
            observation: self.observation.clone(),
            completion,
            service_epoch_user,
            cache_scope_allocator_correlation,
            device_ordinal: self.device_ordinal,
            admitted_device_requirements: self.admitted_device_requirements.clone(),
            compatible_devices: self.compatible_devices.clone(),
            root_context,
            attempt_stream_provider_route,
            envelope,
            pending_frozen_resident_publish: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            release_commit_gate,
            #[cfg(test)]
            envelope_release_gate,
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
                    eprintln!("[query-engine] warning: GPU cleanup submission was rejected; releasing inline: {error}");
                }
            },
            None => {
                release.finish();
            },
        }
    }

    pub(super) fn released_error(&self) -> QueryError {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.query_service.context",
            "query service handle was released before context acquisition",
        )
        .with_fact("admission_request_id", self.admission_request_id)
        .with_fact("reason", "query_handle_released")
    }

    pub(super) fn service(&self) -> Result<Arc<QueryServiceInner>> {
        self.service
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.context",
                    "query service handle mutex was poisoned",
                )
                .with_fact("reason", "query_service_handle_poisoned")
            })?
            .as_ref()
            .cloned()
            .ok_or_else(|| self.released_error())
    }

    pub(super) fn observation_owner(&self) -> Option<RuntimeObservationOwner> {
        match self.completion.lock() {
            Ok(completion) => completion.as_ref().and_then(QueryCompletion::observation_owner),
            Err(error) => error.into_inner().as_ref().and_then(QueryCompletion::observation_owner),
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
#[doc(hidden)]
pub(in crate::runtime::query_service) struct ReleaseCommitGate {
    pub(super) entered: std::sync::atomic::AtomicUsize,
    pub(super) gate: std::sync::Barrier,
}

/// Barrier held while the attempt still owns its admission envelope, so a
/// test can observe exactly what a concurrent admission sees before the lane,
/// grant, and permit are released.
#[cfg(test)]
#[derive(Debug)]
#[doc(hidden)]
pub(in crate::runtime::query_service) struct EnvelopeReleaseGate {
    pub(super) entered: std::sync::atomic::AtomicUsize,
    pub(super) gate: std::sync::Barrier,
}

#[cfg(test)]
impl EnvelopeReleaseGate {
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire) > 0
    }

    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn open(&self) {
        self.gate.wait();
    }
}

#[cfg(test)]
#[derive(Debug)]
#[doc(hidden)]
pub(in crate::runtime::query_service) struct RootContextBindingGate {
    pub(super) entered: std::sync::atomic::AtomicUsize,
    pub(super) gate: std::sync::Barrier,
}

#[cfg(test)]
impl RootContextBindingGate {
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire) > 0
    }

    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn open(&self) {
        self.gate.wait();
    }
}

#[cfg(test)]
impl ReleaseCommitGate {
    /// Whether the release commit has reached the pre-seal barrier.
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire) > 0
    }

    /// Releases the commit to run the seal and the capacity release.
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn open(&self) {
        self.gate.wait();
    }
}
