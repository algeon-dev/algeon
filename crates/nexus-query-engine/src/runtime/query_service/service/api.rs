//! Public request/snapshot DTOs for query-service admission.

use super::*;
use crate::{
    admission::{AttemptRequirement, QueryPlanCapabilityProof},
    attempt::{AttemptAdmissionBinding, AttemptReleaseCommit},
};

#[derive(Debug)]
enum RuntimeObservationAttachment {
    Disabled,
    StandaloneOwner(RuntimeObservationOwner),
    Handle(RuntimeObservationHandle),
    Attempt(AttemptAdmissionBinding),
}

impl Default for RuntimeObservationAttachment {
    fn default() -> Self {
        Self::Disabled
    }
}

/// Point-in-time service admission state for bounded-queue and overload reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceAdmissionSnapshot {
    /// Number of currently admitted query handles.
    pub active_queries: usize,
    /// Number of admissions currently waiting in the service queue.
    pub queued_queries: usize,
    /// Devices currently withholding one future slot and liveness bytes for
    /// an overtake-limited waiter.
    pub protected_devices: Vec<usize>,
}

/// Process-wide active-query slots currently held on one service device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryServiceDeviceAdmissionSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Slots held across every `QueryService` in this process.
    pub process_active_query_slots: usize,
    /// Whether this device currently protects a future slot for a waiter.
    pub has_protection: bool,
}

/// Complete request-scoped metadata for one query-service admission attempt.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct QueryAdmissionRequest {
    /// Opaque capability proof emitted by planning.
    ///
    /// Request callers provide lifecycle and correlation metadata only;
    /// raw floors, working-set hints, placement choices, and capability sets are not a public
    /// admission input.  When absent, admission uses empty evidence only when
    /// every installed profile accepts every modeled capability.
    pub(in crate::runtime::query_service) attempt_requirement: Option<AttemptRequirement>,
    /// Public proof retained for diagnostic inspection until submission.
    planned_capability_proof: Option<QueryPlanCapabilityProof>,
    /// Caller-provided bound for queueing this attempt. The service combines it
    /// with its configured queue bound, choosing the earlier instant.
    pub queue_deadline: Option<Instant>,
    /// Attempt-wide cancellation shared by queueing and transactional resource
    /// acquisition. The same token must remain attached until the admission
    /// either publishes a complete envelope or rolls every provisional
    /// resource back.
    pub execution_cancellation: ExecutionCancellation,
    runtime_observation: RuntimeObservationAttachment,
    runtime_observation_activity: Option<Result<QueryActivityLease>>,
}

pub(in crate::runtime::query_service::service) struct AdmissionObservation {
    pub(in crate::runtime::query_service::service) handle: RuntimeObservationHandle,
    pub(in crate::runtime::query_service::service) owner: Option<RuntimeObservationOwner>,
    pub(in crate::runtime::query_service::service) activity: Option<QueryActivityLease>,
    pub(in crate::runtime::query_service::service) activity_error: Option<QueryError>,
    pub(in crate::runtime::query_service::service) attempt_release_commit: Option<AttemptReleaseCommit>,
}

impl AdmissionObservation {
    pub(in crate::runtime::query_service::service) fn as_ref(&self) -> &RuntimeObservationHandle {
        &self.handle
    }

    pub(in crate::runtime::query_service::service) fn reject_activity_error(
        &mut self,
        admission_request_id: AdmissionRequestId,
    ) -> Result<()> {
        let Some(error) = self.activity_error.take() else {
            return Ok(());
        };
        record_admission_lifecycle(&self.handle, admission_request_id, admission_rejection(&error), 0, None, None);
        Err(error)
    }
}

/// Opaque RAII token retained until query-service cleanup is complete.
///
/// The native engine never invokes the value. Dropping it returns activity
/// ownership to the adapter coordinator after queued admission, contexts,
/// grants, device slots, and cleanup work have all been released.
pub struct QueryActivityLease {
    _lease: Box<dyn Send + Sync>,
}

impl QueryActivityLease {
    pub fn new(lease: impl Send + Sync + 'static) -> Self {
        Self { _lease: Box::new(lease) }
    }
}

impl std::fmt::Debug for QueryActivityLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("QueryActivityLease").finish_non_exhaustive()
    }
}

impl QueryAdmissionRequest {
    /// Creates a request with no explicit capability requirement.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrows the engine-issued capability proof, if planning attached one.
    /// This is diagnostic/read-only and cannot manufacture a new proof.
    pub fn planned_capability_proof(&self) -> Option<&QueryPlanCapabilityProof> {
        self.planned_capability_proof.as_ref()
    }

    /// Attaches an opaque proof computed by capability-first planning.
    ///
    /// A public request may prove required device capabilities, but it cannot
    /// carry a model byte target into grant selection. The installed device
    /// profile therefore remains the sole authority for the immutable grant.
    pub fn with_query_plan_capability_proof(mut self, proof: QueryPlanCapabilityProof) -> Self {
        self.planned_capability_proof = Some(proof);
        self
    }

    /// Crate-test counterpart that preserves a synthetic planner hint.
    #[cfg(test)]
    pub(crate) fn with_test_planned_capability_evidence(mut self, attempt_requirement: AttemptRequirement) -> Self {
        self.attempt_requirement = Some(attempt_requirement);
        self
    }

    /// Internal controller/test constructor for synthetic capability evidence.
    pub(crate) fn with_attempt_requirement(mut self, attempt_requirement: AttemptRequirement) -> Self {
        self.attempt_requirement = Some(attempt_requirement);
        self
    }

    pub(in crate::runtime::query_service) fn resolved_attempt_requirement(&self) -> Option<AttemptRequirement> {
        self.planned_capability_proof
            .as_ref()
            .map(|proof| proof.as_requirement().clone())
            .or_else(|| self.attempt_requirement.clone())
    }

    /// Sets an attempt-bound queue deadline.
    pub fn with_queue_deadline(mut self, queue_deadline: Instant) -> Self {
        self.queue_deadline = Some(queue_deadline);
        self
    }

    /// Carries the attempt-wide cancellation token into admission.
    pub fn with_execution_cancellation(mut self, execution_cancellation: ExecutionCancellation) -> Self {
        self.execution_cancellation = execution_cancellation;
        self
    }

    /// Attaches the query observation owner created by the adapter before enqueue.
    pub fn with_runtime_observation_owner(mut self, owner: RuntimeObservationOwner) -> Self {
        self.runtime_observation = RuntimeObservationAttachment::StandaloneOwner(owner);
        self
    }

    /// Attaches the one engine-minted observation/cleanup binding for an
    /// execution attempt.
    ///
    /// The binding is the only request path that carries attempt cleanup
    /// authority. It is minted atomically by [`crate::attempt::AttemptOwner`],
    /// so request callers cannot create incomplete or cross-attempt pairings.
    pub fn with_attempt_binding(mut self, binding: AttemptAdmissionBinding) -> Self {
        self.runtime_observation = RuntimeObservationAttachment::Attempt(binding);
        self
    }

    /// Attaches coordinator-owned recording access without query-seal authority.
    pub fn with_runtime_observation_handle(mut self, observation: RuntimeObservationHandle) -> Self {
        self.runtime_observation = RuntimeObservationAttachment::Handle(observation);
        self
    }

    /// Retains an adapter-owned activity token through delayed native cleanup.
    pub fn with_runtime_observation_activity(mut self, activity: QueryActivityLease) -> Self {
        self.runtime_observation_activity = Some(Ok(activity));
        self
    }

    /// Carries fallible adapter activity acquisition into native admission.
    ///
    /// Coordinated callers must pass acquisition failure through this field so
    /// admission rejects structurally instead of silently omitting the lease.
    pub fn with_runtime_observation_activity_result(mut self, activity: Result<QueryActivityLease>) -> Self {
        self.runtime_observation_activity = Some(activity);
        self
    }

    pub(in crate::runtime::query_service::service) fn into_runtime_observation(self) -> AdmissionObservation {
        let (activity, activity_error) = match self.runtime_observation_activity {
            Some(Ok(activity)) => (Some(activity), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        let (handle, owner, attempt_release_commit) = match self.runtime_observation {
            RuntimeObservationAttachment::Disabled => (RuntimeObservationHandle::disabled(), None, None),
            RuntimeObservationAttachment::StandaloneOwner(owner) => (owner.handle(), Some(owner), None),
            RuntimeObservationAttachment::Handle(handle) => (handle, None, None),
            RuntimeObservationAttachment::Attempt(binding) => {
                let (owner, release_commit) = binding.into_parts();
                (owner.handle(), Some(owner), Some(release_commit))
            },
        };
        AdmissionObservation { handle, owner, activity, activity_error, attempt_release_commit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::RuntimeObservationConfig;

    #[test]
    fn public_capability_attachment_retains_only_an_engine_issued_proof() {
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution();
        let request = QueryAdmissionRequest::new().with_query_plan_capability_proof(proof.clone());

        assert_eq!(request.planned_capability_proof(), Some(&proof));
        assert_eq!(request.resolved_attempt_requirement(), Some(proof.into_requirement()));
    }

    #[cfg(test)]
    #[test]
    fn explicit_test_hook_retains_synthetic_working_set_hint() {
        let request = QueryAdmissionRequest::new().with_test_planned_capability_evidence(
            AttemptRequirement::from_parts(0, 0, Some(4096), crate::admission::EstimateQuality::Modeled, Vec::new()),
        );
        let evidence = request.resolved_attempt_requirement().expect("test evidence is retained");

        assert_eq!(evidence.soft_working_set_hint_bytes(), Some(4096));
        assert_eq!(evidence.estimate_quality(), crate::admission::EstimateQuality::Modeled);
    }

    #[test]
    fn standalone_observation_owner_carries_no_attempt_release_capability() {
        let owner = RuntimeObservationOwner::new(
            "standalone-observation",
            None,
            RuntimeObservationConfig::bounded(8, 8 * 1024),
        );
        let observation = QueryAdmissionRequest::new().with_runtime_observation_owner(owner).into_runtime_observation();

        assert!(observation.owner.is_some());
        assert!(observation.attempt_release_commit.is_none());
    }

    #[test]
    fn attempt_binding_attaches_observation_and_release_together() {
        let lifecycle = crate::runtime::BackendLifecycle::new();
        let attempt = crate::attempt::AttemptOwner::begin(
            crate::attempt::BeginAttemptRequest {
                observation: RuntimeObservationConfig::bounded(8, 8 * 1024),
                ..crate::attempt::BeginAttemptRequest::default()
            },
            lifecycle.try_begin_observation().expect("attempt observation permit"),
            std::sync::Arc::new(crate::attempt::AttemptBackendCounters::default()),
            None,
        );
        let observation = QueryAdmissionRequest::new()
            .with_attempt_binding(attempt.owner.admission_binding())
            .into_runtime_observation();

        assert!(observation.owner.is_some());
        assert!(observation.attempt_release_commit.is_some());
    }
}
