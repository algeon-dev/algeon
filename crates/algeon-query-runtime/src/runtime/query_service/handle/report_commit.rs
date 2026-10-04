//! Terminal report commit: disposition, health, and observation seal.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use query_planning::error::{QueryError, QueryErrorCode};

#[cfg(test)]
use super::lifetime::ReleaseCommitGate;
use crate::{
    attempt::AttemptLifecycleError,
    observability::{
        AttemptAllocatorEvent, AttemptAllocatorEventAction, AttemptAllocatorEventSource, AttemptAllocatorPoolUsage,
        AttemptAlternateDeviceEligibility, AttemptCapacityDisposition, AttemptEvidenceField, AttemptFaultIsolation,
        AttemptFinalDomainUsage, AttemptHealthTransition, AttemptNativeFaultEvidence, AttemptResourceEvidence,
        AttemptTerminalCommit, AttemptTimedDomainUsage, AttemptUnattributedAllocatorEvent, CleanupFailurePhase,
        CleanupFailureSummary, RuntimeCleanupFailureObservation, RuntimeObservationPayload,
        has_proven_device_local_fault_evidence,
    },
    runtime::{
        allocator_correlation::{
            AllocatorDomainAttempt, AllocatorEventEvidenceRead, AttemptCacheScopeAllocatorCorrelation,
            discard_admitted_domain, retire_admitted_domain,
        },
        attempt_domain::AttemptAllocationDomain,
        gpu::DeviceAdmissionHealth,
        query_service::{
            admission::QueryCompletionActivityTail,
            service::disposition::{
                CapacityCommitDecision, CapacityCommitDisposition, CleanupEvidence, decide_capacity_commit,
            },
            *,
        },
    },
};

pub(super) fn project_pool_usage(usage: cudf::DeviceMemoryPoolUsage) -> AttemptAllocatorPoolUsage {
    AttemptAllocatorPoolUsage {
        reserved_current_bytes: usage.reserved_current_bytes,
        reserved_high_bytes: usage.reserved_high_bytes,
        used_current_bytes: usage.used_current_bytes,
        used_high_bytes: usage.used_high_bytes,
    }
}

/// The report commit: disposition decision, health commit, and observation
/// seal, always before capacity release.
pub(super) struct QueryHandleReportCommit {
    pub(super) service: Option<Arc<QueryServiceInner>>,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) device_ordinal: usize,
    pub(super) admitted_device_requirements: BTreeMap<usize, AttemptRequirement>,
    pub(super) compatible_devices: Vec<usize>,
    pub(super) domain: Option<AttemptAllocationDomain>,
    pub(super) memory_grant: Option<MemoryGrant>,
    pub(super) allocator_domain_attempt: Option<AllocatorDomainAttempt>,
    pub(super) cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    pub(super) outcome_terminal_domain_usage: Option<AttemptTimedDomainUsage>,
    pub(super) pending_frozen_resident_publish: Arc<AtomicBool>,
    #[cfg(test)]
    pub(super) release_commit_gate: Option<Arc<ReleaseCommitGate>>,
    pub(super) terminal_domain_usage: Option<cudf::QueryMemoryDomainUsage>,
}

enum FrozenResidentCommit {
    Frozen(cudf::QueryMemoryDomainUsage),
    Failed { error: QueryError, usage: Option<cudf::QueryMemoryDomainUsage> },
}

struct SettledCapacityCommit {
    disposition: CapacityCommitDisposition,
    usage: Option<cudf::QueryMemoryDomainUsage>,
    native_fault_evidence: Option<AttemptNativeFaultEvidence>,
    cache_scope_allocator_event: Option<cudf::AllocatorEvent>,
    allocator_event_evidence: Option<AllocatorEventEvidenceRead>,
    resident_cleanup_failure: Option<CleanupFailureSummary>,
}

impl QueryHandleReportCommit {
    pub(super) fn run(
        self,
        mut external_completion: Option<QueryCompletion>,
        terminal_outcome: RuntimeTerminalOutcome,
        evidence: CleanupEvidence,
    ) -> Option<QueryCompletionActivityTail> {
        let settled = self.settle_capacity_commit(evidence, terminal_outcome.clone());
        let SettledCapacityCommit {
            disposition,
            usage,
            native_fault_evidence,
            cache_scope_allocator_event,
            allocator_event_evidence,
            resident_cleanup_failure,
        } = settled;
        self.record_terminal_cleanup_failure(evidence, resident_cleanup_failure);
        let committed = self.commit_terminal_report(
            &mut external_completion,
            &disposition,
            usage,
            native_fault_evidence,
            allocator_event_evidence.as_ref(),
            cache_scope_allocator_event,
        );
        // Publication records the admission facts under the admission mutex,
        // so an admitted claim always accepts its one terminal commit.
        debug_assert!(committed.is_ok(), "admitted attempt rejected its terminal commit: {committed:?}");
        // The release is committed: record the terminal admission lifecycle
        // event before the seal so the sealed report retains it. The
        // mechanical capacity release still happens only after the seal.
        if let Some(service) = self.service.as_ref() {
            let queue_depth = service.admission.state.lock().map(|state| state.queued.len()).unwrap_or_else(|error| {
                eprintln!("[query-engine] warning: failed to read poisoned admission queue depth");
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
            if gate.drop_completion {
                drop(external_completion.take());
            }
        }
        // The B-to-R transition is recorded here and applied by the release
        // path after the attempt lane and root stream are gone. Publishing
        // inline would expose reusable ledger capacity while this attempt
        // still holds its lane, so a consumer could pass the ledger and then
        // fail to acquire a lane.
        if matches!(disposition, CapacityCommitDisposition::FrozenResident { .. }) {
            self.pending_frozen_resident_publish.store(true, Ordering::Release);
        }
        external_completion.map(|completion| completion.seal_after_gpu_cleanup(terminal_outcome))
    }

    fn settle_capacity_commit(
        &self,
        evidence: CleanupEvidence,
        terminal_outcome: RuntimeTerminalOutcome,
    ) -> SettledCapacityCommit {
        // Seal attempt-owned cache-scope windows before this report chooses
        // any allocator evidence. A context clone cannot append to a report
        // in flight, and an unclosed checkout remains unavailable rather than
        // racing its private ring against the terminal commit.
        let cache_scope_allocator_event = self.cache_scope_allocator_correlation.seal_and_take_exact_event();
        let allocator_event_evidence = self.finish_allocator_domain_correlation(evidence);
        // The admitted-domain event remains the canonical source when it is
        // exact. Resident cache scopes are independent allocation domains and
        // only fill that exact-evidence gap; they never become a device-wide
        // snapshot or a second event store.
        let frozen_resident_commit = if terminal_outcome == RuntimeTerminalOutcome::Succeeded
            && evidence == CleanupEvidence::Completed
            && self.memory_grant.as_ref().is_some_and(|grant| grant.published_resident_owner_count() > 0)
        {
            Some(self.freeze_published_resident_domain())
        } else {
            None
        };
        let resident_cleanup_failure = match &frozen_resident_commit {
            Some(FrozenResidentCommit::Failed { error, .. }) => {
                Some(CleanupFailureSummary::from_native_error(CleanupFailurePhase::FinalValidation, error))
            },
            Some(FrozenResidentCommit::Frozen(_)) | None => None,
        };
        let decision = match frozen_resident_commit {
            Some(FrozenResidentCommit::Frozen(usage)) => CapacityCommitDecision {
                disposition: CapacityCommitDisposition::FrozenResident { resident_bytes: usage.current_aligned_bytes },
                usage: Some(usage),
                native_fault_evidence: None,
            },
            Some(FrozenResidentCommit::Failed { error, usage }) => {
                let mut decision = decide_capacity_commit(self.device_ordinal, self.domain.as_ref(), evidence, usage);
                if matches!(decision.disposition, CapacityCommitDisposition::HealthyReuse) {
                    decision.disposition = CapacityCommitDisposition::DeviceQuarantine {
                        reason: format!("resident owner attestation failed on device {}: {error}", self.device_ordinal),
                    };
                }
                decision
            },
            None => {
                decide_capacity_commit(self.device_ordinal, self.domain.as_ref(), evidence, self.terminal_domain_usage)
            },
        };
        let CapacityCommitDecision { disposition, usage, native_fault_evidence } = decision;
        if let Some(usage) = usage.as_ref() {
            debug_assert_eq!(
                usage.domain_id,
                self.domain.as_ref().map(|domain| domain.domain_id()).unwrap_or_default()
            );
            let _ = usage.peak_bytes;
        }
        SettledCapacityCommit {
            disposition,
            usage,
            native_fault_evidence,
            cache_scope_allocator_event,
            allocator_event_evidence,
            resident_cleanup_failure,
        }
    }

    fn record_terminal_cleanup_failure(
        &self,
        evidence: CleanupEvidence,
        resident_cleanup_failure: Option<CleanupFailureSummary>,
    ) {
        let cleanup_failure = match evidence {
            CleanupEvidence::Completed => resident_cleanup_failure,
            CleanupEvidence::StreamSynchronizationFailed => {
                Some(CleanupFailureSummary::stream_synchronization_failed())
            },
            CleanupEvidence::CallbackAbandoned => Some(CleanupFailureSummary::callback_abandoned()),
        };
        if let Some(summary) = cleanup_failure {
            // Record the exact tally before the bounded timeline projection.
            // Dropped timeline records cannot affect the exact cleanup count
            // or its first summary.
            self.observation.record_attempt_cleanup_failure(summary.clone());
            self.observation.record(|| {
                RuntimeObservationPayload::CleanupFailure(Box::new(RuntimeCleanupFailureObservation { summary }))
            });
        }
    }

    fn commit_terminal_report(
        &self,
        external_completion: &mut Option<QueryCompletion>,
        disposition: &CapacityCommitDisposition,
        usage: Option<cudf::QueryMemoryDomainUsage>,
        native_fault_evidence: Option<AttemptNativeFaultEvidence>,
        allocator_event_evidence: Option<&AllocatorEventEvidenceRead>,
        cache_scope_allocator_event: Option<cudf::AllocatorEvent>,
    ) -> std::result::Result<(), AttemptLifecycleError> {
        let final_domain_usage = usage.map(AttemptFinalDomainUsage::from);
        let post_cleanup_pool_usage =
            self.domain.as_ref().and_then(|domain| domain.pool_usage().ok()).map(project_pool_usage);
        // The post-cleanup usage is intentionally timed independently from
        // the pre-cleanup outcome-terminal sample. No timestamp is invented
        // for a release without an admitted attempt-clock binding.
        let post_cleanup_usage =
            external_completion.as_ref().and_then(QueryCompletion::attempt_monotonic_ns).zip(final_domain_usage).map(
                |(sampled_ns, usage)| AttemptTimedDomainUsage {
                    sampled_ns,
                    usage,
                    pool_usage: post_cleanup_pool_usage,
                },
            );
        // Commit any device or backend health disposition before taking the
        // atomic ledger snapshot. The resource evidence sequence therefore
        // reflects the state which governed report publication and reuse.
        let capacity_health_committed = self.commit_capacity_health(disposition);
        let alternate_device_eligibility = self.alternate_device_eligibility_after_quarantine(
            disposition,
            native_fault_evidence,
            capacity_health_committed,
        );
        let resource_evidence = self.resource_evidence(
            external_completion.as_ref().and_then(QueryCompletion::attempt_monotonic_ns),
            post_cleanup_usage,
            native_fault_evidence,
            allocator_event_evidence,
            cache_scope_allocator_event,
        );
        // This is the sole projection of the exact final usage sample and
        // health transition above. Typed fault and retry evidence shares this
        // commit and must not resample terminal facts.
        let terminal_commit = match &disposition {
            CapacityCommitDisposition::FrozenResident { .. } => AttemptTerminalCommit::new(
                AttemptCapacityDisposition::FrozenResident,
                final_domain_usage,
                AttemptHealthTransition::HealthyReuse,
            ),
            CapacityCommitDisposition::HealthyReuse => AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                final_domain_usage,
                AttemptHealthTransition::HealthyReuse,
            ),
            CapacityCommitDisposition::DeviceQuarantine { .. } => AttemptTerminalCommit::new(
                AttemptCapacityDisposition::RetainedQuarantine,
                final_domain_usage,
                AttemptHealthTransition::DeviceQuarantined,
            ),
            CapacityCommitDisposition::BackendUnhealthy { .. } => AttemptTerminalCommit::new(
                AttemptCapacityDisposition::RetainedQuarantine,
                final_domain_usage,
                AttemptHealthTransition::BackendUnhealthy,
            ),
        };
        let terminal_commit = native_fault_evidence
            .map(|evidence| terminal_commit.with_native_fault_evidence(evidence))
            .unwrap_or(terminal_commit);
        let terminal_commit = resource_evidence
            .map(|evidence| terminal_commit.with_resource_evidence(evidence))
            .unwrap_or(terminal_commit);
        let terminal_commit = terminal_commit.with_alternate_device_eligibility(alternate_device_eligibility);
        external_completion
            .as_mut()
            .and_then(|completion| completion.finalize_attempt_after_gpu_cleanup(terminal_commit).err())
            .map_or(Ok(()), Err)
    }

    fn freeze_published_resident_domain(&self) -> FrozenResidentCommit {
        let domain = self.domain.as_ref().expect("resident freeze requires an admitted allocation domain");
        let grant = self.memory_grant.as_ref().expect("resident freeze requires an admitted memory grant");
        let usage = match domain.freeze() {
            Ok(usage) => usage,
            Err(error) => return FrozenResidentCommit::Failed { error, usage: None },
        };
        // Owner attestation reads the allocation registry, not two
        // independently derived byte totals. A frozen domain that still owns
        // allocations must agree with its own registry; a residual the
        // registry does not account for is an unowned allocation and
        // therefore a cleanup failure rather than a resident result.
        let live = match domain.live_allocations() {
            Ok(live) => live,
            Err(error) => return FrozenResidentCommit::Failed { error, usage: Some(usage) },
        };
        let registry_aligned_bytes =
            live.iter().try_fold(0u64, |total, allocation| total.checked_add(allocation.aligned_bytes));
        let owner_allocations = grant.published_resident_allocation_pointers();
        let unowned_allocation = live.iter().find(|allocation| !owner_allocations.contains(&allocation.device_pointer));
        let attestation_failure = if registry_aligned_bytes != Some(usage.current_aligned_bytes) {
            Some("frozen domain usage does not match its own allocation registry")
        } else if unowned_allocation.is_some() {
            Some("frozen domain retains a backing allocation no published output owner identifies")
        } else if usage.current_aligned_bytes > grant.budget_bytes() {
            Some("frozen resident bytes exceed the immutable query cap")
        } else {
            None
        };
        if let Some(message) = attestation_failure {
            let error = QueryError::new(
                QueryErrorCode::CleanupFailed,
                "query_engine.runtime.query_service.freeze_resident",
                message,
            )
            .with_fact("domain_id", usage.domain_id)
            .with_fact("current_bytes", usage.current_bytes)
            .with_fact("current_aligned_bytes", usage.current_aligned_bytes)
            .with_fact("registry_allocations", live.len() as u64)
            .with_fact("registry_aligned_bytes", registry_aligned_bytes.unwrap_or(u64::MAX))
            .with_fact("published_owner_allocations", owner_allocations.len() as u64)
            .with_fact("first_unowned_allocation", unowned_allocation.map_or(0, |allocation| allocation.device_pointer))
            .with_fact("query_cap_bytes", grant.budget_bytes());
            return FrozenResidentCommit::Failed { error, usage: Some(usage) };
        }
        if let Err(error) = grant.freeze_resident_with_domain(usage.current_aligned_bytes, Arc::clone(domain.as_cudf()))
        {
            return FrozenResidentCommit::Failed { error, usage: Some(usage) };
        }
        // The publish runs after this report is sealed, where a failure can no
        // longer be recorded as a cleanup disposition. Prove it can succeed
        // while the disposition is still open.
        match grant.validate_frozen_resident_publish() {
            Ok(()) => FrozenResidentCommit::Frozen(usage),
            Err(error) => FrozenResidentCommit::Failed { error, usage: Some(usage) },
        }
    }

    /// Commits the selected health disposition before the resource evidence is
    /// sampled or the attempt finalizer can publish a report.
    fn commit_capacity_health(&self, disposition: &CapacityCommitDisposition) -> bool {
        match disposition {
            CapacityCommitDisposition::HealthyReuse | CapacityCommitDisposition::FrozenResident { .. } => true,
            CapacityCommitDisposition::DeviceQuarantine { reason } => {
                // The device closes BEFORE the resource snapshot and report
                // publication. Its ledger sequence therefore records the
                // closed health state that governed capacity reuse.
                let Some(service) = self.service.as_ref() else {
                    return false;
                };
                let Ok(ledger) = service.admission.devices.ledger(self.device_ordinal) else {
                    return false;
                };
                if let Err(error) = ledger.close_admission(reason.clone()) {
                    eprintln!(
                        "[query-engine] warning: failed to close device admission during release commit: {error}"
                    );
                    return false;
                }
                let committed = matches!(ledger.health(), Ok(DeviceAdmissionHealth::Closed { .. }));
                if !committed {
                    eprintln!(
                        "[query-engine] warning: device admission closure was not observable during release commit"
                    );
                    return false;
                }
                service.admission.notify_admission_available();
                true
            },
            CapacityCommitDisposition::BackendUnhealthy { reason } => {
                if let Some(service) = self.service.as_ref() {
                    service.admission.close_unknown_isolation_admission(reason);
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
        disposition: &CapacityCommitDisposition,
        native_fault_evidence: Option<AttemptNativeFaultEvidence>,
        capacity_health_committed: bool,
    ) -> AttemptAlternateDeviceEligibility {
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
        let Some(query_cap_bytes) = self.memory_grant.as_ref().map(MemoryGrant::budget_bytes) else {
            return AttemptAlternateDeviceEligibility::Unavailable;
        };
        for ordinal in self.compatible_devices.iter().copied().filter(|ordinal| *ordinal != self.device_ordinal) {
            // The ledger is the scheduler's health authority, but it must
            // first fold any sibling-owned persistent CUDA fault into that
            // authority before advice calls an alternate currently healthy.
            if service.admission.devices.synchronize_native_health(ordinal).is_err() {
                return AttemptAlternateDeviceEligibility::Unavailable;
            }
            let Ok(ledger) = service.admission.devices.ledger(ordinal) else {
                return AttemptAlternateDeviceEligibility::Unavailable;
            };
            if !self.admitted_device_requirements.contains_key(&ordinal) {
                return AttemptAlternateDeviceEligibility::Unavailable;
            }
            let can_ever_fit =
                ledger.snapshot().is_ok_and(|snapshot| snapshot.can_ever_fit_query_cap(Some(query_cap_bytes)));
            if can_ever_fit && matches!(ledger.health(), Ok(DeviceAdmissionHealth::Open)) {
                return AttemptAlternateDeviceEligibility::Eligible;
            }
        }
        AttemptAlternateDeviceEligibility::NoEligibleAlternate
    }

    /// Constructs the one bounded resource block. Immutable grant/domain facts
    /// remain attempt-exact even if cleanup lost device isolation; every
    /// device-local snapshot and allocator record stays unavailable in that
    /// case instead of being associated with a guessed device.
    fn resource_evidence(
        &self,
        device_sample_ns: Option<u64>,
        post_cleanup_usage: Option<AttemptTimedDomainUsage>,
        native_fault_evidence: Option<AttemptNativeFaultEvidence>,
        allocator_event_evidence: Option<&AllocatorEventEvidenceRead>,
        cache_scope_allocator_event: Option<cudf::AllocatorEvent>,
    ) -> Option<AttemptResourceEvidence> {
        let (memory_grant, domain) = self.memory_grant.as_ref().zip(self.domain.as_ref())?;
        let mut evidence = AttemptResourceEvidence::unavailable();
        evidence.grant_id = AttemptEvidenceField::attempt_exact(memory_grant.grant_id());
        evidence.grant_cap_bytes = AttemptEvidenceField::attempt_exact(memory_grant.budget_bytes());
        evidence.domain_id = AttemptEvidenceField::attempt_exact(domain.domain_id());
        evidence.domain_cap_bytes = AttemptEvidenceField::attempt_exact(domain.cap_bytes());
        if let Some(usage) = self.outcome_terminal_domain_usage {
            evidence.outcome_terminal_usage = AttemptEvidenceField::sampled_at_terminal_boundary(usage);
        }
        if let Some(usage) = post_cleanup_usage {
            evidence.post_cleanup_usage = AttemptEvidenceField::sampled_at_terminal_boundary(usage);
        }

        // Unknown isolation deliberately retains no device-local ledger,
        // profile, or ring sample. Immutable attempt facts above remain useful
        // without inventing a device association.
        if matches!(
            native_fault_evidence,
            Some(AttemptNativeFaultEvidence { isolation: AttemptFaultIsolation::Unknown, .. })
        ) {
            return Some(evidence);
        }
        let Some(service) = self.service.as_ref() else {
            return Some(evidence);
        };
        let Ok(ledger) = service.admission.devices.ledger(self.device_ordinal) else {
            return Some(evidence);
        };
        let profile = ledger.profile();
        evidence.configured_limit_bytes = AttemptEvidenceField::attempt_exact(profile.configured_limit_bytes);
        evidence.headroom_bytes = AttemptEvidenceField::attempt_exact(profile.headroom_bytes);
        evidence.backend_reserve_bytes = AttemptEvidenceField::attempt_exact(profile.backend_reserve_bytes);
        evidence.managed_capacity_bytes = AttemptEvidenceField::attempt_exact(profile.managed_capacity_bytes);
        if let (Some(sampled_ns), Ok(snapshot)) = (device_sample_ns, ledger.snapshot()) {
            evidence.ledger_sequence = AttemptEvidenceField::device_snapshot(snapshot.sequence);
            evidence.ledger_sample_ns = AttemptEvidenceField::device_snapshot(sampled_ns);
            evidence.live_grants_bytes = AttemptEvidenceField::device_snapshot(snapshot.live_grants_bytes);
            evidence.resident_owner_bytes = AttemptEvidenceField::device_snapshot(snapshot.resident_owner_bytes);
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
            } else if let Some(event) = cache_scope_allocator_event {
                evidence.allocator_event = AttemptEvidenceField::attempt_exact(Self::allocator_event(event, true));
            } else if let (Some(sampled_ns), Some(event)) = (device_sample_ns, allocator_event_evidence.unattributed) {
                evidence.unattributed_device_allocator_event =
                    AttemptEvidenceField::device_snapshot(AttemptUnattributedAllocatorEvent {
                        device_ordinal: self.device_ordinal,
                        sampled_ns,
                        event: Self::allocator_event(event, false),
                    });
            }
        } else if let Some(event) = cache_scope_allocator_event {
            evidence.allocator_event = AttemptEvidenceField::attempt_exact(Self::allocator_event(event, true));
        }
        Some(evidence)
    }

    fn allocator_event(event: cudf::AllocatorEvent, attempt_exact: bool) -> AttemptAllocatorEvent {
        let source = match event.source {
            cudf::DeviceMemoryFailureSource::Physical => AttemptAllocatorEventSource::Physical,
            cudf::DeviceMemoryFailureSource::ReservationLimit => AttemptAllocatorEventSource::ReservationLimit,
            cudf::DeviceMemoryFailureSource::AllocationLimit => AttemptAllocatorEventSource::AllocationLimit,
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
    fn finish_allocator_domain_correlation(&self, evidence: CleanupEvidence) -> Option<AllocatorEventEvidenceRead> {
        let service = self.service.as_ref()?;
        let binding = self.allocator_domain_attempt?;
        if evidence == CleanupEvidence::Completed
            && let Some(domain) = self.domain.as_ref()
        {
            Some(retire_admitted_domain(&service.allocator_event_correlation, binding, domain.as_cudf()))
        } else {
            discard_admitted_domain(&service.allocator_event_correlation, binding);
            None
        }
    }
}
