//! Terminal GPU cleanup and admission capacity release.

use std::{
    collections::BTreeMap,
    mem, panic,
    panic::AssertUnwindSafe,
    sync,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[cfg(test)]
use super::lifetime::{EnvelopeReleaseGate, ReleaseCommitGate};
use super::{
    super::*,
    report_commit::{QueryHandleReportCommit, project_pool_usage},
};
use crate::{
    observability::AttemptTimedDomainUsage,
    runtime::{
        AttemptStreamProviderRoute,
        allocator_correlation::AttemptCacheScopeAllocatorCorrelation,
        query_service::service::{disposition::CleanupEvidence, transaction::AdmissionEnvelope},
        service_stream::StreamCleanupOutcome,
    },
};

pub(super) fn take_owned_for_cleanup<T>(owned: &Mutex<Option<T>>, resource: &'static str) -> Option<T> {
    match owned.lock() {
        Ok(mut owned) => owned.take(),
        Err(error) => {
            eprintln!("[query-engine] warning: recovering poisoned query service {resource}");
            error.into_inner().take()
        },
    }
}

pub(super) struct QueryHandleRelease {
    pub(super) service: Option<Arc<QueryServiceInner>>,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) completion: Option<QueryCompletion>,
    pub(super) service_epoch_user: Option<QueryServiceEpochUser>,
    pub(super) cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    pub(super) device_ordinal: usize,
    pub(super) admitted_device_requirements: BTreeMap<usize, AttemptRequirement>,
    pub(super) compatible_devices: Vec<usize>,
    pub(super) root_context: Option<ExecutionContext>,
    pub(super) attempt_stream_provider_route: Option<Arc<AttemptStreamProviderRoute>>,
    pub(super) envelope: Option<AdmissionEnvelope>,
    /// Set by the report commit when the sealed report published a frozen
    /// resident disposition. The B-to-R ledger transition is applied only
    /// after the envelope releases this attempt's lane and root stream, so a
    /// consumer that passes the ledger always finds a free lane.
    pub(super) pending_frozen_resident_publish: Arc<AtomicBool>,
    #[cfg(test)]
    pub(super) release_commit_gate: Option<Arc<ReleaseCommitGate>>,
    #[cfg(test)]
    pub(super) envelope_release_gate: Option<Arc<EnvelopeReleaseGate>>,
    pub(super) terminal_domain_usage: Option<cudf::QueryMemoryDomainUsage>,
    /// Sampled after the terminal outcome is resolved but before GPU cleanup
    /// begins. The timestamp is obtained only from an admitted attempt clock.
    pub(super) outcome_terminal_domain_usage: Option<AttemptTimedDomainUsage>,
    pub(super) terminal_outcome: RuntimeTerminalOutcome,
    pub(super) release_started: bool,
    pub(super) internal_cleanup_finished: bool,
}

/// Owns a release until either CUDA cleanup starts or a submission/cancellation
/// path drops the task. The fallback is deliberately consuming: a rejected GPU
/// executor submission cannot strand an admitted handle in the active count.
pub(super) struct QueryHandleReleaseGuard {
    release: Option<QueryHandleRelease>,
}

impl QueryHandleReleaseGuard {
    pub(super) fn new(release: QueryHandleRelease) -> Self {
        Self { release: Some(release) }
    }

    pub(super) fn finish(mut self) {
        if let Some(release) = self.release.take() {
            release.finish();
        }
    }
}

impl Drop for QueryHandleReleaseGuard {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = panic::catch_unwind(AssertUnwindSafe(|| release.finish()));
        }
    }
}

impl QueryHandleRelease {
    pub(super) fn requires_gpu_cleanup(&self) -> bool {
        if self.root_context.is_some() {
            return true;
        }
        // A device-backed admission envelope owns the root stream and
        // allocation domain from admission time. Their destruction must run on
        // the device executor even when execution never reached
        // context acquisition. Ledger-only envelopes own no CUDA state.
        self.envelope.as_ref().is_some_and(AdmissionEnvelope::holds_gpu_resources)
    }

    pub(super) fn begin_query_cleanup(&self) {
        if let Some(completion) = &self.completion {
            completion.begin_query_cleanup();
        }
    }

    /// Establish an unterminalized attempt's fallback before any GPU cleanup
    /// task is submitted. The stored value is the exact terminal outcome used
    /// after cleanup for report publication and capacity accounting.
    pub(super) fn resolve_terminal_outcome_before_gpu_cleanup(&mut self) {
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
        let usage = domain.usage().ok();
        let pool_usage = domain.pool_usage().ok().map(project_pool_usage);
        self.outcome_terminal_domain_usage =
            usage.map(|usage| AttemptTimedDomainUsage { sampled_ns, usage: usage.into(), pool_usage });
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
            let stream_provider_route = self.attempt_stream_provider_route.take();
            let external_completion = self.take_external_completion();
            let commit = self.prepare_report_commit();
            let terminal_outcome = self.terminal_outcome.clone();
            let internal_cleanup = QueryHandleReleaseGuard::new(mem::replace(self, Self::empty_recovered_release()));
            // The health commit, usage sample, and observation seal run before
            // the capacity release callback.
            let activity_tail = Arc::new(Mutex::new(None));
            let activity_tail_slot = Arc::clone(&activity_tail);
            let installed = root_context.install_root_stream_cleanup_completion(
                move |outcome| {
                    let outcome = if stream_provider_route
                        .as_ref()
                        .is_some_and(|route| route.synchronize_auxiliary_for_teardown().is_err())
                    {
                        StreamCleanupOutcome::StreamSynchronizationFailed
                    } else {
                        outcome
                    };
                    drop_recovering(stream_provider_route, "attempt stream provider route");
                    *activity_tail_slot.lock().unwrap_or_else(sync::PoisonError::into_inner) =
                        commit.run(external_completion, terminal_outcome, CleanupEvidence::from(outcome))
                },
                move || {
                    internal_cleanup.finish();
                    if let Some(tail) = activity_tail.lock().unwrap_or_else(sync::PoisonError::into_inner).take() {
                        tail.finish();
                    }
                },
            );
            if !installed {
                eprintln!("[query-engine] warning: root stream cleanup completion was already installed");
            }
            drop_recovering(Some(root_context), "root execution context");
            return;
        }
        drop(root_context);
        // Fence accounting on root-stream cleanup; publishing from the GPU job can deadlock reentrant backend close.
        let commit = self.prepare_report_commit();
        if let Some(stream_lease) = self.envelope.as_mut().and_then(|envelope| envelope.release_unconsumed_physical()) {
            let external_completion = self.take_external_completion();
            let terminal_outcome = self.terminal_outcome.clone();
            let internal_cleanup = QueryHandleReleaseGuard::new(mem::replace(self, Self::empty_recovered_release()));
            let activity_tail = Arc::new(Mutex::new(None));
            let activity_tail_slot = Arc::clone(&activity_tail);
            let installed = stream_lease.install_cleanup_completion(
                move |outcome| {
                    *activity_tail_slot.lock().unwrap_or_else(sync::PoisonError::into_inner) =
                        commit.run(external_completion, terminal_outcome, CleanupEvidence::from(outcome))
                },
                move || {
                    internal_cleanup.finish();
                    if let Some(tail) = activity_tail.lock().unwrap_or_else(sync::PoisonError::into_inner).take() {
                        tail.finish();
                    }
                },
            );
            if !installed {
                eprintln!("[query-engine] warning: root stream cleanup completion was already installed");
            }
            drop_recovering(Some(stream_lease), "admission envelope root stream lease");
            return;
        }
        // Ledger-only or context-consumed release: no stream fence remains.
        // The report commit still precedes the capacity release.
        let external_completion = self.take_external_completion();
        let activity_tail = commit.run(external_completion, self.terminal_outcome.clone(), CleanupEvidence::Completed);
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
            cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation::default(),
            device_ordinal: 0,
            admitted_device_requirements: BTreeMap::new(),
            compatible_devices: Vec::new(),
            root_context: None,
            attempt_stream_provider_route: None,
            envelope: None,
            pending_frozen_resident_publish: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            release_commit_gate: None,
            #[cfg(test)]
            envelope_release_gate: None,
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
            admitted_device_requirements: self.admitted_device_requirements.clone(),
            compatible_devices: self.compatible_devices.clone(),
            domain,
            memory_grant,
            allocator_domain_attempt,
            cache_scope_allocator_correlation: mem::take(&mut self.cache_scope_allocator_correlation),
            outcome_terminal_domain_usage: self.outcome_terminal_domain_usage.take(),
            pending_frozen_resident_publish: Arc::clone(&self.pending_frozen_resident_publish),
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
        // Unregister the native fork provider before dropping the envelope's
        // physical roots. The route owns the auxiliary stream members, and
        // its native registration must outlive every forked stream submission.
        drop_recovering(self.attempt_stream_provider_route.take(), "attempt stream provider route");
        // A frozen resident grant outlives this attempt through its output
        // lease, so its ledger transition needs a handle that survives the
        // envelope.
        let frozen_resident_grant = self
            .pending_frozen_resident_publish
            .load(Ordering::Acquire)
            .then(|| self.envelope.as_ref().map(|envelope| envelope.memory_grant().clone()))
            .flatten();
        // Deterministic-barrier hook: tests observe what a concurrent
        // admission sees while this attempt still owns its lane.
        #[cfg(test)]
        if let Some(gate) = self.envelope_release_gate.take() {
            gate.entered.fetch_add(1, Ordering::AcqRel);
            gate.gate.wait();
        }
        // The envelope drops physical remnants, the attempt lane, immutable
        // grant, backend-generation permit, and lifecycle ticket in that
        // order. The final MemoryGrant drop invokes the ledger release hook
        // that reacquires the admission mutex, so it must never happen while
        // that mutex is held.
        drop_recovering(self.envelope.take(), "admission envelope");
        // Seal, then release the lane and root stream, then publish B to R.
        // Only now can the freed `B - R` become visible to admission, and by
        // then this attempt's lane is already free.
        if let Some(grant) = frozen_resident_grant {
            self.pending_frozen_resident_publish.store(false, Ordering::Release);
            if let Err(error) = grant.publish_frozen_resident() {
                if let Some(service) = self.service.as_ref() {
                    service.close_unknown_isolation_admission(format!(
                        "failed to publish prepared frozen resident accounting: {error}"
                    ));
                }
                eprintln!("[query-engine] warning: failed to publish frozen resident accounting after report: {error}");
            }
        }
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
            CleanupEvidence::CallbackAbandoned,
        );
        self.complete_internal_cleanup();
        if let Some(tail) = activity_tail {
            tail.finish();
        }
    }
}

fn drop_recovering<T>(value: Option<T>, resource: &'static str) {
    if panic::catch_unwind(AssertUnwindSafe(|| drop(value))).is_err() {
        eprintln!("[query-engine] warning: query service {resource} drop panicked during cleanup recovery");
    }
}
