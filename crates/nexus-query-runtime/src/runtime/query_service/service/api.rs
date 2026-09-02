//! Public request/snapshot DTOs for query-service admission.

use std::time::Duration;

use serde::Serialize;

use super::*;

/// Result of a controlled device-quarantine injection.
///
/// The injection changes only the engine's admission ledger. It deliberately
/// does not manufacture a CUDA error or mutate cuDF's process-wide device
/// health; controlled validation uses it to exercise the same device-local
/// scheduler transition that a proven native fault would perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceQuarantineInjection {
    device_ordinal: usize,
    device_health_generation: u64,
    newly_quarantined: bool,
}

impl DeviceQuarantineInjection {
    #[doc(hidden)]
    pub fn new(device_ordinal: usize, device_health_generation: u64, newly_quarantined: bool) -> Self {
        Self { device_ordinal, device_health_generation, newly_quarantined }
    }

    /// Ordinal of the admission ledger that was targeted.
    pub fn device_ordinal(self) -> usize {
        self.device_ordinal
    }

    /// Generation of the targeted ledger's closed admission state.
    pub fn device_health_generation(self) -> u64 {
        self.device_health_generation
    }

    /// Whether this call performed the open-to-closed transition.
    pub fn newly_quarantined(self) -> bool {
        self.newly_quarantined
    }
}
use crate::{
    admission::{AttemptRequirement, QueryPlanCapabilityProof},
    attempt::{AttemptAdmissionBinding, AttemptFinalizerSlot},
};

#[derive(Debug, Default)]
enum RuntimeObservationAttachment {
    #[default]
    Disabled,
    StandaloneOwner(RuntimeObservationOwner),
    Handle(RuntimeObservationHandle),
    Attempt(AttemptAdmissionBinding),
}

/// Point-in-time service admission state for bounded-queue and overload reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QueryServiceAdmissionSnapshot {
    /// Number of currently admitted query handles.
    pub active_queries: usize,
    /// Number of admissions currently waiting in the service queue.
    pub queued_queries: usize,
    /// Largest number of admissions queued concurrently since this service
    /// was constructed. This is maintained by queue transitions, rather than
    /// inferred from snapshot polling, and therefore remains available after
    /// the queue drains.
    pub queue_high_watermark: usize,
    /// Devices currently withholding one future slot and liveness bytes for
    /// an overtake-limited waiter.
    pub protected_devices: Vec<usize>,
}

/// Process-wide active-query slots currently held on one service device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct QueryServiceDeviceAdmissionSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Slots held across every `QueryService` in this process.
    pub process_active_query_slots: usize,
    /// Whether this device currently protects a future slot for a waiter.
    pub has_protection: bool,
}

/// Caller-facing classification for a failed native admission or execution
/// attempt. This projection is intentionally policy-neutral: applications
/// decide whether a capacity result should be queued, shed, or surfaced.
///
/// Admission and execution are separate caller boundaries. Use
/// [`Self::from_error`] for an error returned while an admission is being
/// attempted, and [`Self::from_execution_error`] after a handle has been
/// published. The engine does not infer that boundary from a diagnostic
/// reason or message.
///
/// A frontend can keep its own protocol policy at the edge without teaching
/// the engine about HTTP:
///
/// ```text
/// match QueryAdmissionOutcome::from_error_with_queue_bounds(&error, &queue_bounds) {
///     Capacity { queued_queries, queue_capacity, max_queue_wait } => {
///         // Enqueue when the application's wait budget allows it. If its
///         // own bounded queue is full, return 429. `max_queue_wait` is only
///         // the engine's rejection bound; it is not a relief-time ETA.
///         if should_enqueue(queued_queries, queue_capacity, max_queue_wait) {
///             AppAction::Enqueue
///         } else {
///             AppAction::Http429
///         }
///     }
///     CapabilityRejected | InvalidInput => AppAction::Shed,
///     Cancelled => AppAction::Shed,
///     ExecutionFailure => AppAction::Http503,
/// }
/// ```
///
/// `AppAction` and `should_enqueue` belong to the application. The original
/// [`QueryError`] remains the structured diagnostic source; this enum is only
/// its stable policy-selection projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QueryAdmissionOutcome {
    /// Bounded capacity is temporarily unavailable.  Queue facts are present
    /// when the service attached them to the structured error; the configured
    /// wait bound is supplied by [`Self::from_error_with_queue_bounds`].
    Capacity { queued_queries: Option<usize>, queue_capacity: Option<usize>, max_queue_wait: Option<Duration> },
    /// The plan or operator is permanently outside the supported capability
    /// catalogue.
    CapabilityRejected,
    /// The request, plan, or materialized input is invalid for this operation.
    InvalidInput,
    /// Caller cancellation or deadline stopped the attempt before completion.
    Cancelled,
    /// A dependency, invariant, or native execution failure occurred after
    /// admission.  No retry policy is implied by this label.
    ExecutionFailure,
}

impl QueryAdmissionOutcome {
    /// Classifies an error returned while admission is being attempted.
    ///
    /// This method does not inspect diagnostic reason strings. A caller that
    /// has already published a handle must use [`Self::from_execution_error`]
    /// instead, even when the execution error has a resource-exhaustion code.
    pub fn from_error(error: &QueryError) -> Self {
        Self::from_error_with_optional_queue_bounds(error, None)
    }

    /// Classifies a structured native error and attaches the service-owned
    /// queue wait bound when the caller has the corresponding configuration.
    pub fn from_error_with_queue_bounds(error: &QueryError, queue_bounds: &QueueBounds) -> Self {
        Self::from_error_with_optional_queue_bounds(error, Some(queue_bounds))
    }

    /// Classifies an error observed after admission published a handle.
    ///
    /// Execution-time resource errors are deliberately not projected as
    /// admission capacity: the service cannot ask the caller to queue an
    /// attempt that has already started. Cancellation remains cancellation;
    /// every other post-admission error is an execution failure.
    pub fn from_execution_error(error: &QueryError) -> Self {
        if error.code() == QueryErrorCode::Cancelled { Self::Cancelled } else { Self::ExecutionFailure }
    }

    fn from_error_with_optional_queue_bounds(error: &QueryError, queue_bounds: Option<&QueueBounds>) -> Self {
        match error.code() {
            QueryErrorCode::ServiceOverloaded
            | QueryErrorCode::MemoryReservationFailed
            | QueryErrorCode::CudfResourceExhausted => Self::Capacity {
                queued_queries: fact_usize(error, "queued_queries"),
                queue_capacity: fact_usize(error, "max_waiters")
                    .or_else(|| queue_bounds.map(|bounds| bounds.max_waiters)),
                max_queue_wait: queue_bounds.map(|bounds| bounds.max_wait),
            },
            QueryErrorCode::UnsupportedPlanShape
            | QueryErrorCode::UnsupportedOperator
            | QueryErrorCode::UnsupportedExpression
            | QueryErrorCode::UnsupportedType
            | QueryErrorCode::CudfUnsupported => Self::CapabilityRejected,
            QueryErrorCode::InvalidPlan
            | QueryErrorCode::InvalidAttemptCorrelation
            | QueryErrorCode::GraphInputInvalid
            | QueryErrorCode::CudfInvalidInput => Self::InvalidInput,
            QueryErrorCode::Cancelled => Self::Cancelled,
            QueryErrorCode::CudfTemporaryFailure
            | QueryErrorCode::CudfPersistentFailure
            | QueryErrorCode::AlgorithmDidNotConverge
            | QueryErrorCode::CudfDependencyFailure
            | QueryErrorCode::SourceAccessDenied
            | QueryErrorCode::SourceUnauthenticated
            | QueryErrorCode::SourceObjectNotFound
            | QueryErrorCode::SourceTemporaryFailure
            | QueryErrorCode::CudfInvariantViolation
            | QueryErrorCode::SizeOverflow
            | QueryErrorCode::HostMaterializationLimitExceeded
            | QueryErrorCode::Internal => Self::ExecutionFailure,
        }
    }

    /// Returns the queue depth attached to a capacity result, if known.
    pub fn queued_queries(&self) -> Option<usize> {
        match self {
            Self::Capacity { queued_queries, .. } => *queued_queries,
            _ => None,
        }
    }

    /// Returns the configured queue capacity attached to a capacity result,
    /// if the rejection carried it.
    pub fn queue_capacity(&self) -> Option<usize> {
        match self {
            Self::Capacity { queue_capacity, .. } => *queue_capacity,
            _ => None,
        }
    }

    /// Returns the configured maximum queue wait when supplied by the caller.
    pub fn max_queue_wait(&self) -> Option<Duration> {
        match self {
            Self::Capacity { max_queue_wait, .. } => *max_queue_wait,
            _ => None,
        }
    }
}

fn fact_usize(error: &QueryError, key: &str) -> Option<usize> {
    error.fact_value(key)?.parse().ok()
}

#[cfg(test)]
mod outcome_tests {
    use super::*;

    fn error(code: QueryErrorCode) -> QueryError {
        QueryError::new(code, "nexus_query_engine.test", "test error")
    }

    #[test]
    fn capacity_projection_preserves_queue_facts_and_configured_wait_bound() {
        let error = error(QueryErrorCode::ServiceOverloaded)
            .with_fact("reason", "admission_queue_full")
            .with_fact("queued_queries", 7)
            .with_fact("max_waiters", 8);
        let bounds = QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(11), max_overtakes_per_waiter: 2 };

        let outcome = QueryAdmissionOutcome::from_error_with_queue_bounds(&error, &bounds);
        assert_eq!(outcome.queued_queries(), Some(7));
        assert_eq!(outcome.queue_capacity(), Some(8));
        assert_eq!(outcome.max_queue_wait(), Some(Duration::from_secs(11)));
        assert!(matches!(outcome, QueryAdmissionOutcome::Capacity { .. }));
    }

    #[test]
    fn admission_projection_is_exhaustive_and_does_not_parse_reason_strings() {
        let cases = [
            (QueryErrorCode::MemoryReservationFailed, "capacity"),
            (QueryErrorCode::UnsupportedPlanShape, "capability"),
            (QueryErrorCode::InvalidPlan, "invalid"),
            (QueryErrorCode::Cancelled, "cancelled"),
            (QueryErrorCode::CudfInvariantViolation, "execution"),
        ];
        for (code, expected) in cases {
            let outcome = QueryAdmissionOutcome::from_error(&error(code).with_fact("reason", "unrelated_reason_name"));
            let actual = match outcome {
                QueryAdmissionOutcome::Capacity { .. } => "capacity",
                QueryAdmissionOutcome::CapabilityRejected => "capability",
                QueryAdmissionOutcome::InvalidInput => "invalid",
                QueryAdmissionOutcome::Cancelled => "cancelled",
                QueryAdmissionOutcome::ExecutionFailure => "execution",
            };
            assert_eq!(actual, expected, "classification for {code:?}");
        }
    }

    #[test]
    fn capacity_projection_leaves_unknown_queue_metrics_unknown() {
        let outcome = QueryAdmissionOutcome::from_error(&error(QueryErrorCode::CudfResourceExhausted));
        assert!(matches!(
            outcome,
            QueryAdmissionOutcome::Capacity { queued_queries: None, queue_capacity: None, max_queue_wait: None }
        ));

        let bounds = QueueBounds { max_waiters: 4, max_wait: Duration::from_secs(3), max_overtakes_per_waiter: 1 };
        let outcome =
            QueryAdmissionOutcome::from_error_with_queue_bounds(&error(QueryErrorCode::CudfResourceExhausted), &bounds);
        assert_eq!(outcome.queued_queries(), None);
        assert_eq!(outcome.queue_capacity(), Some(4));
        assert_eq!(outcome.max_queue_wait(), Some(Duration::from_secs(3)));
    }

    #[test]
    fn execution_projection_does_not_reclassify_post_admission_resource_errors() {
        for code in [QueryErrorCode::MemoryReservationFailed, QueryErrorCode::CudfResourceExhausted] {
            assert_eq!(
                QueryAdmissionOutcome::from_execution_error(&error(code)),
                QueryAdmissionOutcome::ExecutionFailure,
                "post-admission {code:?} is an execution failure"
            );
        }
        assert_eq!(
            QueryAdmissionOutcome::from_execution_error(&error(QueryErrorCode::Cancelled)),
            QueryAdmissionOutcome::Cancelled
        );
    }
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
    #[cfg(any(test, feature = "test-hooks"))]
    test_placement_requirement: Option<AttemptRequirement>,
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
    /// The owner-held finalizer routing handle. It remains here while the
    /// request is queued or rejected and moves its unique token into the
    /// admitted envelope only after admission succeeds.
    pub(in crate::runtime::query_service::service) attempt_finalizer_slot: Option<AttemptFinalizerSlot>,
    /// Present only for an engine-minted attempt binding. This is the identity
    /// that may receive a post-callback allocator correlation after a physical
    /// domain has been admitted.
    pub(in crate::runtime::query_service::service) attempt_id: Option<crate::observability::AttemptId>,
}

impl AdmissionObservation {
    pub(in crate::runtime::query_service::service) fn as_ref(&self) -> &RuntimeObservationHandle {
        &self.handle
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

    /// Crate-test constructor for an explicit placement requirement.
    ///
    /// This sizes an admission only; it is intentionally not an execution
    /// plan proof and therefore cannot authorize a later-built plan.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn with_test_placement_requirement(mut self, attempt_requirement: AttemptRequirement) -> Self {
        AttemptRequirement::empty()
            .with_test_placement_overlay(attempt_requirement.clone())
            .expect("test placement evidence is finite");
        self.test_placement_requirement = Some(attempt_requirement);
        self
    }

    /// Internal controller/test constructor for synthetic capability evidence.
    ///
    /// Kept as crate-private API surface: the public-API boundary UI tests
    /// pin that callers outside the crate cannot attach their own requirement.
    #[allow(dead_code)]
    #[doc(hidden)]
    pub fn with_attempt_requirement(mut self, attempt_requirement: AttemptRequirement) -> Self {
        self.attempt_requirement = Some(attempt_requirement);
        self
    }

    pub(in crate::runtime::query_service) fn resolved_attempt_requirement(&self) -> Option<AttemptRequirement> {
        let requirement = self
            .planned_capability_proof
            .as_ref()
            .map(|proof| proof.as_requirement().clone())
            .or_else(|| self.attempt_requirement.clone());
        #[cfg(any(test, feature = "test-hooks"))]
        if let Some(test_placement_requirement) = &self.test_placement_requirement {
            return requirement
                .map(|requirement| {
                    requirement
                        .with_test_placement_overlay(test_placement_requirement.clone())
                        .expect("test placement evidence is finite")
                })
                .or_else(|| Some(test_placement_requirement.clone()));
        }
        requirement
    }

    pub(in crate::runtime::query_service::service) fn apply_test_placement_overlay(
        &self,
        requirement: AttemptRequirement,
    ) -> AttemptRequirement {
        #[cfg(any(test, feature = "test-hooks"))]
        if let Some(test_placement_requirement) = &self.test_placement_requirement {
            return requirement
                .with_test_placement_overlay(test_placement_requirement.clone())
                .expect("test placement evidence is finite");
        }
        requirement
    }

    /// Returns the proof that was actually submitted by planning for the
    /// execution-time coverage check.  Internal requirements still size and
    /// place an admission, but must not impersonate a proof for an arbitrary
    /// later-built plan.
    pub(in crate::runtime::query_service) fn submitted_plan_capability_evidence(&self) -> Option<AttemptRequirement> {
        self.planned_capability_proof.as_ref().map(|proof| proof.as_requirement().clone())
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

    #[cfg(test)]
    pub(in crate::runtime::query_service::service) fn into_runtime_observation(self) -> AdmissionObservation {
        let mut request = self;
        request.take_runtime_observation()
    }

    pub(in crate::runtime::query_service::service) fn take_runtime_observation(&mut self) -> AdmissionObservation {
        let (activity, activity_error) = match self.runtime_observation_activity.take() {
            Some(Ok(activity)) => (Some(activity), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        let (handle, owner, attempt_finalizer_slot, attempt_id) = match std::mem::take(&mut self.runtime_observation) {
            RuntimeObservationAttachment::Disabled => (RuntimeObservationHandle::disabled(), None, None, None),
            RuntimeObservationAttachment::StandaloneOwner(owner) => (owner.handle(), Some(owner), None, None),
            RuntimeObservationAttachment::Handle(handle) => (handle, None, None, None),
            RuntimeObservationAttachment::Attempt(binding) => {
                let attempt_id = binding.attempt_id();
                let (owner, finalizer_slot) = binding.into_parts();
                (owner.handle(), Some(owner), Some(finalizer_slot), Some(attempt_id))
            },
        };
        AdmissionObservation { handle, owner, activity, activity_error, attempt_finalizer_slot, attempt_id }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        observability::RuntimeObservationConfig,
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, SourceNode},
    };

    fn explicit_sink_capability_proof() -> QueryPlanCapabilityProof {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let plan = builder.finish_with_host_sink(source);
        QueryPlanCapabilityProof::for_query_plan(&plan)
            .expect("an explicit host sink must expose a capability requirement")
    }

    #[test]
    fn public_capability_attachment_retains_only_an_engine_issued_proof() {
        let proof = explicit_sink_capability_proof();
        let request = QueryAdmissionRequest::new().with_query_plan_capability_proof(proof.clone());

        assert_eq!(request.planned_capability_proof(), Some(&proof));
        assert_eq!(request.resolved_attempt_requirement(), Some(proof.into_requirement()));
        assert!(request.submitted_plan_capability_evidence().is_some());
    }

    #[cfg(test)]
    #[test]
    fn explicit_test_placement_requirement_retains_working_set_hint_without_submitted_proof() {
        let request = QueryAdmissionRequest::new().with_test_placement_requirement(AttemptRequirement::from_parts(
            0,
            0,
            Some(4096),
            crate::admission::EstimateQuality::Modeled,
            Vec::new(),
        ));
        let evidence = request.resolved_attempt_requirement().expect("test evidence is retained");

        assert_eq!(evidence.soft_working_set_hint_bytes(), Some(4096));
        assert_eq!(evidence.estimate_quality(), crate::admission::EstimateQuality::Modeled);
        assert!(request.submitted_plan_capability_evidence().is_none());
    }

    #[test]
    fn test_placement_overlay_is_independent_of_proof_attachment_order() {
        let proof = explicit_sink_capability_proof();
        let original_proof = proof.clone();
        let placement =
            AttemptRequirement::from_parts(0, 0, Some(4096), crate::admission::EstimateQuality::Measured, Vec::new());

        let proof_first = QueryAdmissionRequest::new()
            .with_query_plan_capability_proof(proof.clone())
            .with_test_placement_requirement(placement.clone());
        let placement_first = QueryAdmissionRequest::new()
            .with_test_placement_requirement(placement)
            .with_query_plan_capability_proof(proof);

        assert_eq!(proof_first.resolved_attempt_requirement(), placement_first.resolved_attempt_requirement());
        assert_eq!(
            proof_first.submitted_plan_capability_evidence(),
            placement_first.submitted_plan_capability_evidence()
        );
        assert_eq!(proof_first.planned_capability_proof(), Some(&original_proof));
        assert_eq!(placement_first.planned_capability_proof(), Some(&original_proof));
        assert_eq!(proof_first.submitted_plan_capability_evidence(), Some(original_proof.as_requirement().clone()));
    }

    #[test]
    fn test_placement_overlay_preserves_plan_scope_and_capability_evidence() {
        let proof = explicit_sink_capability_proof();
        let before = proof.clone();
        let request = QueryAdmissionRequest::new()
            .with_query_plan_capability_proof(proof)
            .with_test_placement_requirement(AttemptRequirement::from_parts(
                0,
                0,
                Some(4096),
                crate::admission::EstimateQuality::Measured,
                Vec::new(),
            ));
        assert_eq!(request.planned_capability_proof(), Some(&before));
        assert_eq!(request.submitted_plan_capability_evidence(), Some(before.as_requirement().clone()));
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
        assert!(observation.attempt_finalizer_slot.is_none());
        assert!(observation.attempt_id.is_none());
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
        let attempt_id = attempt.owner.attempt_id();
        let observation = QueryAdmissionRequest::new()
            .with_attempt_binding(attempt.owner.admission_binding())
            .into_runtime_observation();

        assert!(observation.owner.is_some());
        assert!(observation.attempt_finalizer_slot.is_some());
        assert_eq!(observation.attempt_id, Some(attempt_id));
    }
}
