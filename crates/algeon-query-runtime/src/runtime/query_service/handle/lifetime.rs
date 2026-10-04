//! Admitted attempt lifetime and root-context setup coordination.

#[cfg(any(test, feature = "cugraph"))]
use std::sync;
#[cfg(test)]
use std::sync::Barrier;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::BTreeMap,
    mem,
    sync::{Arc, MutexGuard, Weak, atomic::AtomicBool},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::sync::watch;

use super::{
    super::{
        super::context::{AdmittedPlanIdentities, AttemptAuxiliarySlots, QueryResourceAnchor, QueryResourceEpoch},
        *,
    },
    release::{QueryHandleRelease, QueryHandleReleaseGuard},
};
#[cfg(test)]
use crate::runtime::attempt_domain::AttemptAllocationDomain;
#[cfg(test)]
use crate::runtime::gpu::DeviceWorkerScope;
use crate::{
    attempt::{AdmittedGrantRelease, AttemptHandle},
    observability::AttemptId,
    runtime::{
        AttemptStreamProviderRoute, allocator_correlation::AttemptCacheScopeAllocatorCorrelation,
        deadline::block_until, query_service::service::transaction::AdmissionEnvelope,
    },
};

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
    /// Release was requested; no context starts. Being a value rather than a
    /// bare wakeup, a waiter that read the state just before the release still
    /// sees it on its next read.
    Released,
}

/// The root-context setup state. Each state change wakes the waiting
/// receivers.
#[derive(Debug)]
struct RootContextSetupCoordinator {
    state: watch::Sender<RootContextSetupState>,
}

impl RootContextSetupCoordinator {
    fn new() -> Self {
        Self { state: watch::Sender::new(RootContextSetupState::Unclaimed) }
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

/// The admitted attempt's resources until the one release takes them.
#[derive(Debug)]
pub(super) struct LifetimeResources {
    service: Arc<QueryServiceInner>,
    completion: QueryCompletion,
    service_epoch_user: QueryServiceEpochUser,
    pub(super) envelope: AdmissionEnvelope,
    pub(super) root_context: Option<ExecutionContext>,
    pub(super) attempt_stream_provider_route: Option<Arc<AttemptStreamProviderRoute>>,
    terminal_domain_usage: Option<cudf::QueryMemoryDomainUsage>,
}

impl LifetimeResources {
    pub(super) fn attempt(&self) -> Option<AttemptHandle> {
        self.completion.attempt()
    }
}

#[derive(Debug)]
enum LifetimeState {
    /// Admitted; execution contexts may be acquired.
    Active(Box<LifetimeResources>),
    /// Release was requested. Contexts already in flight keep the resources
    /// until the query epoch goes idle; no new context starts.
    Releasing(Box<LifetimeResources>),
    /// The one release took the resources.
    Released,
}

impl LifetimeState {
    fn resources(&mut self) -> Option<&mut LifetimeResources> {
        match self {
            Self::Active(resources) | Self::Releasing(resources) => Some(resources),
            Self::Released => None,
        }
    }
}

#[derive(Debug)]
pub(in crate::runtime::query_service) struct QueryHandleLifetime {
    state: Mutex<LifetimeState>,
    pub(in crate::runtime::query_service) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) attempt_id: AttemptId,
    /// The admitted attempt owns the collector until every outward execution
    /// context has drained. Report commit then consumes its one unambiguous
    /// cache-scope event after CUDA cleanup is fenced.
    cache_scope_allocator_correlation: Mutex<AttemptCacheScopeAllocatorCorrelation>,
    device_ordinal: usize,
    /// Immutable admission inputs retained only until the post-cleanup report
    /// commit can determine whether a quarantined device has a truthful
    /// alternate retry target. They never drive a second placement.
    admitted_device_requirements: BTreeMap<usize, AttemptCapabilityEvidence>,
    compatible_devices: Vec<usize>,
    pub(super) query_epoch: QueryResourceEpoch,
    root_context_setup: RootContextSetupCoordinator,
    pub(super) attempt_auxiliary_slots: Arc<AttemptAuxiliarySlots>,
    /// Placement was fixed when the claim was placed; these identities grow
    /// as fragments planned after the claim join the grant.
    pub(super) admitted_plan_identities: Arc<AdmittedPlanIdentities>,
    #[cfg(test)]
    release_commit_gate: Mutex<Option<Arc<ReleaseCommitGate>>>,
    #[cfg(test)]
    envelope_release_gate: Mutex<Option<Arc<EnvelopeReleaseGate>>>,
    #[cfg(test)]
    root_context_binding_gate: Mutex<Option<Arc<RootContextBindingGate>>>,
}

impl QueryHandleLifetime {
    #[cfg(test)]
    #[doc(hidden)]
    pub(in crate::runtime::query_service) fn admitted_device_requirement_for_test(
        &self,
        ordinal: usize,
    ) -> Option<&AttemptCapabilityEvidence> {
        self.admitted_device_requirements.get(&ordinal)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "service construction inputs are distinct admission and ownership values"
    )]
    pub(in crate::runtime::query_service) fn new(
        service: Arc<QueryServiceInner>,
        admission_request_id: AdmissionRequestId,
        observation: RuntimeObservationHandle,
        attempt_id: AttemptId,
        completion: QueryCompletion,
        envelope: AdmissionEnvelope,
        admitted_capability_evidence: Option<AttemptCapabilityEvidence>,
        admitted_device_requirements: BTreeMap<usize, AttemptCapabilityEvidence>,
        compatible_devices: Vec<usize>,
    ) -> Arc<Self> {
        let device_ordinal = envelope.device_ordinal();
        let attempt_auxiliary_capacity = service.admission.devices.ledger(device_ordinal).map_or_else(
            |_| service.config.source_scheduler_policy.max_source_reads_in_flight().max(1),
            |ledger| ledger.profile().source_work_per_attempt,
        );
        let attempt_auxiliary_slots = AttemptAuxiliarySlots::new(device_ordinal, attempt_auxiliary_capacity);
        let service_epoch_user = service.service_epoch.retain_existing_user();
        let lifetime = Arc::new_cyclic(|weak: &Weak<QueryHandleLifetime>| {
            let weak_lifetime = Weak::clone(weak);
            let on_query_idle = Arc::new(move || {
                if let Some(lifetime) = weak_lifetime.upgrade() {
                    lifetime.release_if_ready();
                }
            });
            Self {
                state: Mutex::new(LifetimeState::Active(Box::new(LifetimeResources {
                    service,
                    completion,
                    service_epoch_user,
                    envelope,
                    root_context: None,
                    attempt_stream_provider_route: None,
                    terminal_domain_usage: None,
                }))),
                admission_request_id,
                observation,
                attempt_id,
                cache_scope_allocator_correlation: Mutex::new(
                    AttemptCacheScopeAllocatorCorrelation::for_admitted_attempt(Some(attempt_id)),
                ),
                device_ordinal,
                admitted_device_requirements,
                compatible_devices,
                query_epoch: QueryResourceEpoch::new(Some(on_query_idle)),
                root_context_setup: RootContextSetupCoordinator::new(),
                attempt_auxiliary_slots,
                admitted_plan_identities: Arc::new(AdmittedPlanIdentities::new(admitted_capability_evidence)),
                #[cfg(test)]
                release_commit_gate: Mutex::new(None),
                #[cfg(test)]
                envelope_release_gate: Mutex::new(None),
                #[cfg(test)]
                root_context_binding_gate: Mutex::new(None),
            }
        });
        // The attempt, not this handle, holds the release: only its terminal
        // commit returns the grant.
        let release: Arc<dyn AdmittedGrantRelease> = Arc::clone(&lifetime) as Arc<dyn AdmittedGrantRelease>;
        let attempt_already_terminal = match &*lifetime.lock_state() {
            LifetimeState::Active(resources) => resources.completion.bind_grant_release(release),
            LifetimeState::Releasing(_) | LifetimeState::Released => unreachable!("a new handle lifetime is active"),
        };
        if attempt_already_terminal {
            lifetime.request_release();
        }
        lifetime
    }

    fn lock_state(&self) -> MutexGuard<'_, LifetimeState> {
        self.state.lock().unwrap_or_else(|error| {
            eprintln!("[query-engine] warning: recovering poisoned query service handle lifetime state");
            error.into_inner()
        })
    }

    /// Runs `with` on the attempt's resources while they exist: before the
    /// one release takes them, including while a requested release waits for
    /// in-flight contexts.
    pub(super) fn with_resources<T>(&self, with: impl FnOnce(&mut LifetimeResources) -> Result<T>) -> Result<T> {
        let mut state = self.lock_state();
        with(state.resources().ok_or_else(|| self.released_error())?)
    }

    /// Whether the handle still accepts new execution contexts.
    fn is_active(&self) -> bool {
        matches!(*self.lock_state(), LifetimeState::Active(_))
    }

    #[cfg(feature = "cugraph")]
    pub(super) fn cache_scope_allocator_correlation(&self) -> AttemptCacheScopeAllocatorCorrelation {
        self.cache_scope_allocator_correlation.lock().unwrap_or_else(sync::PoisonError::into_inner).clone()
    }

    fn take_cache_scope_allocator_correlation(&self) -> AttemptCacheScopeAllocatorCorrelation {
        match self.cache_scope_allocator_correlation.lock() {
            Ok(mut correlation) => mem::take(&mut *correlation),
            Err(error) => {
                eprintln!(
                    "[query-engine] warning: recovering poisoned query service cache scope allocator correlation"
                );
                mem::take(&mut *error.into_inner())
            },
        }
    }

    /// Attempt allocation domain clone for disposition tests.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn envelope_domain_for_test(&self) -> Option<AttemptAllocationDomain> {
        self.with_resources(|resources| Ok(resources.envelope.domain().cloned())).ok().flatten()
    }

    /// Admission-envelope identity evidence for transaction tests:
    /// (domain id, grant id, worker scope).
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn envelope_evidence(
        &self,
    ) -> Option<(Option<u64>, u64, Option<DeviceWorkerScope>)> {
        self.with_resources(|resources| {
            let envelope = &resources.envelope;
            Ok((
                envelope.domain().map(|domain| domain.domain_id()),
                envelope.memory_grant().grant_id(),
                envelope.worker_scope(),
            ))
        })
        .ok()
    }

    /// Arms a barrier inside this handle's release commit, after the health
    /// commit and before the observation seal. The barrier fires exactly once.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn arm_release_commit_gate(&self) -> Arc<ReleaseCommitGate> {
        let gate = Arc::new(ReleaseCommitGate {
            entered: sync::atomic::AtomicUsize::new(0),
            gate: sync::Barrier::new(2),
            drop_completion: false,
        });
        *self.release_commit_gate.lock().unwrap_or_else(sync::PoisonError::into_inner) = Some(Arc::clone(&gate));
        gate
    }

    /// Arms this handle's release commit to drop its completion payload after
    /// the attempt is finalized and before the observation seal, so the
    /// payload's recovery path seals instead.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn arm_release_commit_completion_drop(&self) {
        let gate = ReleaseCommitGate {
            entered: sync::atomic::AtomicUsize::new(0),
            gate: sync::Barrier::new(1),
            drop_completion: true,
        };
        *self.release_commit_gate.lock().unwrap_or_else(sync::PoisonError::into_inner) = Some(Arc::new(gate));
    }

    /// Arms a barrier that holds the attempt at the point where it still owns
    /// its admission envelope, after the report seal. The barrier fires once.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn arm_envelope_release_gate(&self) -> Arc<EnvelopeReleaseGate> {
        let gate =
            Arc::new(EnvelopeReleaseGate { entered: sync::atomic::AtomicUsize::new(0), gate: sync::Barrier::new(2) });
        *self.envelope_release_gate.lock().unwrap_or_else(sync::PoisonError::into_inner) = Some(Arc::clone(&gate));
        gate
    }

    /// Pauses root-context setup after it has consumed the root binding and
    /// before the root context exists. This makes release/setup races
    /// deterministic without exposing a production lifecycle seam.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn arm_root_context_binding_gate(&self) -> Arc<RootContextBindingGate> {
        let gate = Arc::new(RootContextBindingGate {
            entered: sync::atomic::AtomicUsize::new(0),
            gate: sync::Barrier::new(2),
        });
        *self.root_context_binding_gate.lock().unwrap_or_else(sync::PoisonError::into_inner) = Some(Arc::clone(&gate));
        gate
    }

    #[cfg(test)]
    pub(super) fn wait_at_root_context_binding_gate(&self) {
        let gate = self
            .root_context_binding_gate
            .lock()
            .unwrap_or_else(sync::PoisonError::into_inner)
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
        let _ = self.with_resources(|resources| {
            resources.terminal_domain_usage = Some(usage);
            Ok(())
        });
    }

    /// Installs an unrelated root-stream completion to exercise the
    /// fail-closed duplicate-handoff path. Normal query-service ownership
    /// never permits this; a later release must commit `CallbackAbandoned`
    /// before it returns admission capacity.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn install_foreign_root_stream_cleanup_completion_for_test(&self) {
        let installed = self.with_resources(|resources| {
            let root_context =
                resources.root_context.as_ref().expect("root context is acquired before injecting cleanup collision");
            Ok(root_context.install_root_stream_cleanup_completion(|_| {}, || {}))
        });
        assert!(installed.expect("handle is not released"));
    }

    #[cfg(test)]
    pub(in crate::runtime::query_service) fn memory_grant(&self) -> Result<MemoryGrant> {
        self.with_resources(|resources| Ok(resources.envelope.memory_grant().clone()))
    }

    pub(super) fn acquire_resource_anchor(
        self: &Arc<Self>,
        operation: &'static str,
    ) -> Result<Arc<QueryResourceAnchor>> {
        if !self.is_active() {
            return Err(self.released_error());
        }
        let owner: Arc<dyn Send + Sync> = Arc::clone(self) as _;
        self.query_epoch.acquire(owner, operation).map_err(|_error| self.released_error())
    }

    fn reserve_context_setup(&self) -> Result<ContextSetupClaim> {
        if !self.is_active() {
            return Err(self.released_error());
        }
        // `None` once release was requested: no context starts after it.
        let mut claim = Some(ContextSetupClaim::Auxiliary);
        self.root_context_setup.state.send_if_modified(|state| match *state {
            RootContextSetupState::Unclaimed => {
                *state = RootContextSetupState::InFlight;
                claim = Some(ContextSetupClaim::Root);
                true
            },
            RootContextSetupState::InFlight => {
                claim = Some(ContextSetupClaim::WaitForRoot);
                false
            },
            RootContextSetupState::Ready => false,
            RootContextSetupState::Released => {
                claim = None;
                false
            },
        });
        claim.ok_or_else(|| self.released_error())
    }

    fn wait_for_root_context_setup(&self, cancellation: &ExecutionCancellation, operation: &'static str) -> Result<()> {
        let mut setup = self.root_context_setup.state.subscribe();
        loop {
            cancellation.check(operation)?;
            if !self.is_active() {
                return Err(self.released_error());
            }
            match *setup.borrow_and_update() {
                RootContextSetupState::InFlight => {},
                RootContextSetupState::Released => return Err(self.released_error()),
                RootContextSetupState::Unclaimed | RootContextSetupState::Ready => return Ok(()),
            }
            let _ = block_until(
                async {
                    tokio::select! {
                        biased;
                        _ = setup.changed() => {},
                        () = cancellation.cancel_requested() => {},
                    }
                },
                cancellation.deadline(),
            );
        }
    }

    async fn wait_for_root_context_setup_async(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<()> {
        let mut setup = self.root_context_setup.state.subscribe();
        loop {
            cancellation.check(operation)?;
            if !self.is_active() {
                return Err(self.released_error());
            }
            match *setup.borrow_and_update() {
                RootContextSetupState::InFlight => {},
                RootContextSetupState::Released => return Err(self.released_error()),
                RootContextSetupState::Unclaimed | RootContextSetupState::Ready => return Ok(()),
            }
            tokio::select! {
                biased;
                _ = setup.changed() => {},
                () = cancellation.cancelled_or_expired() => {},
            }
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
            match self.reserve_context_setup()? {
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
            match self.reserve_context_setup()? {
                ContextSetupClaim::WaitForRoot => {
                    self.wait_for_root_context_setup_async(cancellation, operation).await?
                },
                claim => return Ok(claim),
            }
        }
    }

    pub(super) fn complete_root_context_setup(&self) {
        self.root_context_setup.state.send_if_modified(|state| {
            debug_assert!(matches!(*state, RootContextSetupState::InFlight | RootContextSetupState::Released));
            let in_flight = *state == RootContextSetupState::InFlight;
            if in_flight {
                *state = RootContextSetupState::Ready;
            }
            in_flight
        });
    }

    fn rollback_root_context_setup(&self) {
        self.root_context_setup.state.send_if_modified(|state| {
            let in_flight = *state == RootContextSetupState::InFlight;
            if in_flight {
                *state = RootContextSetupState::Unclaimed;
            }
            in_flight
        });
    }

    fn release_root_context_setup(&self) {
        self.root_context_setup.state.send_replace(RootContextSetupState::Released);
    }

    pub(super) fn request_release(&self) {
        {
            let mut state = self.lock_state();
            if let LifetimeState::Active(_) = &*state {
                let LifetimeState::Active(resources) = mem::replace(&mut *state, LifetimeState::Released) else {
                    unreachable!("the state was just matched as active");
                };
                *state = LifetimeState::Releasing(resources);
            }
        }
        self.query_epoch.invalidate();
        self.release_root_context_setup();
        self.release_if_ready();
    }

    fn release_if_ready(&self) {
        if !self.query_epoch.is_idle() {
            return;
        }
        let resources = {
            let mut state = self.lock_state();
            if !matches!(*state, LifetimeState::Releasing(_)) {
                return;
            }
            match mem::replace(&mut *state, LifetimeState::Released) {
                LifetimeState::Releasing(resources) => *resources,
                LifetimeState::Active(_) | LifetimeState::Released => {
                    unreachable!("the state was just matched as releasing")
                },
            }
        };
        let LifetimeResources {
            service,
            completion,
            service_epoch_user,
            envelope,
            root_context,
            attempt_stream_provider_route,
            terminal_domain_usage,
        } = resources;
        let cache_scope_allocator_correlation = self.take_cache_scope_allocator_correlation();
        #[cfg(test)]
        let release_commit_gate = self.release_commit_gate.lock().unwrap_or_else(sync::PoisonError::into_inner).take();
        #[cfg(test)]
        let envelope_release_gate =
            self.envelope_release_gate.lock().unwrap_or_else(sync::PoisonError::into_inner).take();
        let mut release = QueryHandleRelease {
            service: Some(service),
            admission_request_id: self.admission_request_id,
            observation: self.observation.clone(),
            completion: Some(completion),
            service_epoch_user: Some(service_epoch_user),
            cache_scope_allocator_correlation,
            device_ordinal: self.device_ordinal,
            admitted_device_requirements: self.admitted_device_requirements.clone(),
            compatible_devices: self.compatible_devices.clone(),
            root_context,
            attempt_stream_provider_route,
            envelope: Some(envelope),
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
                .gpu_execution_pool_for_device(self.device_ordinal)
                .ok()
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
        self.with_resources(|resources| Ok(Arc::clone(&resources.service)))
    }
}

impl AdmittedGrantRelease for QueryHandleLifetime {
    fn request_release(&self) {
        QueryHandleLifetime::request_release(self);
    }
}

#[cfg(test)]
#[derive(Debug)]
#[doc(hidden)]
pub(in crate::runtime::query_service) struct ReleaseCommitGate {
    pub(super) entered: AtomicUsize,
    pub(super) gate: Barrier,
    // Drops the completion payload at the barrier instead of sealing it, as
    // a release that unwinds between finalization and the seal would.
    pub(super) drop_completion: bool,
}

/// Barrier held while the attempt still owns its admission envelope, so a
/// test can observe exactly what a concurrent admission sees before the lane,
/// grant, and permit are released.
#[cfg(test)]
#[derive(Debug)]
#[doc(hidden)]
pub(in crate::runtime::query_service) struct EnvelopeReleaseGate {
    pub(super) entered: AtomicUsize,
    pub(super) gate: Barrier,
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
    pub(super) entered: AtomicUsize,
    pub(super) gate: Barrier,
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
