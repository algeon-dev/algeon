//! Public request/snapshot DTOs for query-service admission.

#[cfg(test)]
use std::time::Duration;
use std::{fmt, time::Instant};

#[cfg(test)]
use query_planning::error::QueryErrorCode;
use query_planning::error::{QueryError, Result};
use serde::Serialize;

/// Result of a controlled device-quarantine injection.
///
/// The injection changes only the engine's admission ledger. It deliberately
/// does not manufacture a CUDA error or mutate cuDF's process-wide device
/// health; controlled validation uses it to exercise the same device-local
/// scheduler transition that a proven native fault would perform.
#[cfg(feature = "qualification")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceQuarantineInjection {
    device_ordinal: usize,
    device_health_generation: u64,
    newly_quarantined: bool,
}

#[cfg(feature = "qualification")]
impl DeviceQuarantineInjection {
    pub(crate) fn new(device_ordinal: usize, device_health_generation: u64, newly_quarantined: bool) -> Self {
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
use query_planning::admission::AttemptRequirement;

#[cfg(any(test, feature = "test-hooks"))]
use crate::attempt::AttemptOwner;
#[cfg(test)]
use crate::runtime::query_service::config::QueueBounds;
use crate::{
    attempt::{AttemptAdmission, AttemptAdmissionBinding, AttemptLifecycleError},
    observability::{AttemptId, RuntimeObservationHandle},
    runtime::ExecutionCancellation,
};

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
    pub(crate) protected_devices: Vec<usize>,
}

/// Process-wide active-query slots currently held on one service device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct QueryServiceDeviceAdmissionSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Slots held across every `QueryService` in this process.
    pub process_active_query_slots: usize,
    /// Whether this device currently protects a future slot for a waiter.
    pub(crate) has_protection: bool,
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
///     BudgetUnavailable { .. } => AppAction::ShedWithNewCapOrDevice,
///     CapabilityRejected | InvalidInput => AppAction::Shed,
///     Cancelled => AppAction::Shed,
///     // Returned both for pre-admission errors outside the categories
///     // above and for every post-admission error except cancellation.
///     ExecutionFailure => AppAction::Http503,
/// }
/// ```
///
/// `AppAction` and `should_enqueue` belong to the application. The original
/// [`QueryError`] remains the structured diagnostic source; this enum is only
/// its stable policy-selection projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg(test)]
pub(crate) enum QueryAdmissionOutcome {
    /// Bounded capacity is temporarily unavailable.  Queue facts are present
    /// when the service attached them to the structured error; the configured
    /// wait bound is supplied by [`Self::from_error_with_queue_bounds`].
    Capacity { queued_queries: Option<usize>, queue_capacity: Option<usize>, max_queue_wait: Option<Duration> },
    /// The requested query cap does not fit any eligible device after
    /// resident ownership is deducted.
    ///
    /// This is deliberately not [`Self::Capacity`]: queueing and backoff
    /// cannot make an impossible cap fit, so the caller must release resident
    /// owners or change the cap, device, or query.
    BudgetUnavailable { requested_cap_bytes: Option<u64>, largest_eligible_cap_bytes: Option<u64> },
    /// The plan or operator is permanently outside the supported capability
    /// catalogue.
    CapabilityRejected,
    /// The request, plan, or materialized input is invalid for this operation.
    InvalidInput,
    /// Caller cancellation or deadline stopped the attempt before completion.
    Cancelled,
    /// A failure that is not admission capacity, an impossible cap, a
    /// capability rejection, invalid input, or cancellation. [`Self::from_error`]
    /// returns it for any such pre-admission error; [`Self::from_execution_error`]
    /// returns it for every post-admission error except cancellation. The
    /// label therefore cannot prove execution started, and no retry policy
    /// is implied by it.
    ExecutionFailure,
}

#[cfg(test)]
impl QueryAdmissionOutcome {
    /// Classifies an error returned while admission is being attempted.
    ///
    /// This method does not inspect diagnostic reason strings. A caller that
    /// has already published a handle must use [`Self::from_execution_error`]
    /// instead, even when the execution error has a resource-exhaustion code.
    #[cfg(test)]
    pub(crate) fn from_error(error: &QueryError) -> Self {
        Self::from_error_with_optional_queue_bounds(error, None)
    }

    /// Classifies a structured native error and attaches the service-owned
    /// queue wait bound when the caller has the corresponding configuration.
    #[cfg(test)]
    pub(crate) fn from_error_with_queue_bounds(error: &QueryError, queue_bounds: &QueueBounds) -> Self {
        Self::from_error_with_optional_queue_bounds(error, Some(queue_bounds))
    }

    /// Classifies an error observed after admission published a handle.
    ///
    /// Execution-time resource errors are deliberately not projected as
    /// admission capacity: the service cannot ask the caller to queue an
    /// attempt that has already started. Cancellation remains cancellation;
    /// every other post-admission error is an execution failure.
    #[cfg(test)]
    pub(crate) fn from_execution_error(error: &QueryError) -> Self {
        if error.code() == QueryErrorCode::Cancelled { Self::Cancelled } else { Self::ExecutionFailure }
    }

    #[cfg(test)]
    fn from_error_with_optional_queue_bounds(error: &QueryError, queue_bounds: Option<&QueueBounds>) -> Self {
        match error.code() {
            QueryErrorCode::QueryBudgetUnavailable => Self::BudgetUnavailable {
                requested_cap_bytes: fact_u64(error, "requested_cap_bytes"),
                largest_eligible_cap_bytes: fact_u64(error, "largest_eligible_cap_bytes"),
            },
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
            | QueryErrorCode::QueryMemoryLimitExceeded
            | QueryErrorCode::DeviceOutOfMemory
            | QueryErrorCode::DeviceAllocationFailed
            | QueryErrorCode::DeviceFault
            | QueryErrorCode::CleanupFailed
            | QueryErrorCode::InternalInvariantViolation
            | QueryErrorCode::SizeOverflow
            | QueryErrorCode::HostMaterializationLimitExceeded
            | QueryErrorCode::Internal => Self::ExecutionFailure,
        }
    }

    /// Returns the queue depth attached to a capacity result, if known.
    #[cfg(test)]
    pub(crate) fn queued_queries(&self) -> Option<usize> {
        match self {
            Self::Capacity { queued_queries, .. } => *queued_queries,
            _ => None,
        }
    }

    /// Returns the configured queue capacity attached to a capacity result,
    /// if the rejection carried it.
    #[cfg(test)]
    pub(crate) fn queue_capacity(&self) -> Option<usize> {
        match self {
            Self::Capacity { queue_capacity, .. } => *queue_capacity,
            _ => None,
        }
    }

    /// Returns the configured maximum queue wait when supplied by the caller.
    #[cfg(test)]
    pub(crate) fn max_queue_wait(&self) -> Option<Duration> {
        match self {
            Self::Capacity { max_queue_wait, .. } => *max_queue_wait,
            _ => None,
        }
    }
}

#[cfg(test)]
fn fact_usize(error: &QueryError, key: &str) -> Option<usize> {
    error.fact_value(key)?.parse().ok()
}

#[cfg(test)]
fn fact_u64(error: &QueryError, key: &str) -> Option<u64> {
    error.fact_value(key)?.parse().ok()
}

#[cfg(test)]
mod outcome_tests {
    use super::*;

    fn error(code: QueryErrorCode) -> QueryError {
        QueryError::new(code, "query_engine.test", "test error")
    }

    #[test]
    fn capacity_projection_preserves_queue_facts_and_configured_wait_bound() {
        let error = error(QueryErrorCode::ServiceOverloaded)
            .with_fact("reason", "admission_queue_full")
            .with_fact("queued_queries", 7)
            .with_fact("max_waiters", 8);
        let bounds = QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(11) };

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
            // An impossible cap is not queue pressure: the caller must change
            // the cap or release resident owners, never simply back off.
            (QueryErrorCode::QueryBudgetUnavailable, "budget_unavailable"),
            (QueryErrorCode::UnsupportedPlanShape, "capability"),
            (QueryErrorCode::InvalidPlan, "invalid"),
            (QueryErrorCode::Cancelled, "cancelled"),
            (QueryErrorCode::CudfInvariantViolation, "execution"),
        ];
        for (code, expected) in cases {
            let outcome = QueryAdmissionOutcome::from_error(&error(code).with_fact("reason", "unrelated_reason_name"));
            let actual = match outcome {
                QueryAdmissionOutcome::Capacity { .. } => "capacity",
                QueryAdmissionOutcome::BudgetUnavailable { .. } => "budget_unavailable",
                QueryAdmissionOutcome::CapabilityRejected => "capability",
                QueryAdmissionOutcome::InvalidInput => "invalid",
                QueryAdmissionOutcome::Cancelled => "cancelled",
                QueryAdmissionOutcome::ExecutionFailure => "execution",
            };
            assert_eq!(actual, expected, "classification for {code:?}");
        }
        // The budget projection never reports queue metrics, so a backend
        // cannot mistake it for queue pressure with a wait bound.
        let bounds = QueueBounds { max_waiters: 8, max_wait: Duration::from_secs(11) };
        let budget = QueryAdmissionOutcome::from_error_with_queue_bounds(
            &error(QueryErrorCode::QueryBudgetUnavailable).with_fact("queued_queries", 7).with_fact("max_waiters", 8),
            &bounds,
        );
        assert_eq!(budget.queued_queries(), None);
        assert_eq!(budget.queue_capacity(), None);
        assert_eq!(budget.max_queue_wait(), None);
    }

    #[test]
    fn capacity_projection_leaves_unknown_queue_metrics_unknown() {
        let outcome = QueryAdmissionOutcome::from_error(&error(QueryErrorCode::CudfResourceExhausted));
        assert!(matches!(
            outcome,
            QueryAdmissionOutcome::Capacity { queued_queries: None, queue_capacity: None, max_queue_wait: None }
        ));

        let bounds = QueueBounds { max_waiters: 4, max_wait: Duration::from_secs(3) };
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
#[derive(Debug)]
#[non_exhaustive]
pub struct QueryAdmissionRequest {
    /// Explicit immutable query cap. When absent, the selected device resolves
    /// its configured default inside the admission transaction.
    requested_query_cap_bytes: Option<u64>,
    /// Opaque capability evidence emitted by planning.
    ///
    /// Request callers provide lifecycle and correlation metadata only;
    /// raw floors, working-set hints, placement choices, and capability sets are not a public
    /// admission input.  When absent, admission uses empty evidence only when
    /// every installed profile accepts every modeled capability.
    pub(in crate::runtime::query_service) attempt_requirement: Option<AttemptRequirement>,
    /// Public evidence retained for diagnostic inspection until submission.
    planned_capability_evidence: Option<AttemptRequirement>,
    /// Caller-provided bound for queueing this attempt. The service combines it
    /// with its configured queue bound, choosing the earlier instant.
    pub(crate) queue_deadline: Option<Instant>,
    /// Attempt-wide cancellation shared by queueing and transactional resource
    /// acquisition. The same token must remain attached until the admission
    /// either publishes a complete envelope or rolls every provisional
    /// resource back.
    pub(crate) execution_cancellation: ExecutionCancellation,
    /// The admitted attempt's observation and cleanup attachment.
    attempt: AttemptAdmissionBinding,
    runtime_observation_activity: Option<Result<QueryActivityLease>>,
    /// The owner of a [`Self::test_attempt`] request, handed to the admitted
    /// query handle.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(in crate::runtime::query_service) test_attempt: Option<AttemptOwner>,
}

pub(in crate::runtime::query_service::service) struct AdmissionObservation {
    pub(in crate::runtime::query_service::service) handle: RuntimeObservationHandle,
    pub(in crate::runtime::query_service::service) activity: Option<QueryActivityLease>,
    pub(in crate::runtime::query_service::service) activity_error: Option<QueryError>,
    /// The engine-minted attempt identity, which may receive a post-callback
    /// allocator correlation after a physical domain has been admitted.
    pub(in crate::runtime::query_service::service) attempt_id: AttemptId,
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

impl fmt::Debug for QueryActivityLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("QueryActivityLease").finish_non_exhaustive()
    }
}

impl QueryAdmissionRequest {
    /// Creates the admission request of one execution attempt, with no
    /// explicit capability requirement.
    ///
    /// The binding is the request's only observation and cleanup attachment.
    /// [`crate::attempt::AttemptOwner`] mints it as one unit, so a request
    /// cannot pair one attempt's observation with another attempt's cleanup.
    pub fn new(attempt: AttemptAdmissionBinding) -> Self {
        Self {
            requested_query_cap_bytes: None,
            attempt_requirement: None,
            planned_capability_evidence: None,
            queue_deadline: None,
            execution_cancellation: ExecutionCancellation::default(),
            attempt,
            runtime_observation_activity: None,
            #[cfg(any(test, feature = "test-hooks"))]
            test_attempt: None,
        }
    }

    /// Creates the request of a fresh attempt classified as native, for
    /// tests that admit without a planning frontend.
    ///
    /// The admitted query handle owns the attempt: only an attempt's terminal
    /// commit releases its grant, so dropping that handle ends the attempt as
    /// `Dropped` and the release follows.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn test_attempt() -> Self {
        let owner = AttemptOwner::native_for_test();
        let mut request = Self::new(owner.admission_binding());
        request.test_attempt = Some(owner);
        request
    }

    /// Requests an exact immutable query cap. Admission never clamps it.
    pub fn with_query_cap_bytes(mut self, query_cap_bytes: u64) -> Self {
        self.requested_query_cap_bytes = Some(query_cap_bytes);
        self
    }

    pub(in crate::runtime::query_service) const fn requested_query_cap_bytes(&self) -> Option<u64> {
        self.requested_query_cap_bytes
    }

    /// Attaches an opaque evidence computed by capability-first planning.
    ///
    /// A public request may prove required device capabilities and plan
    /// scope, but it cannot carry a model byte target into grant selection.
    /// The installed device profile therefore remains the sole authority for
    /// the immutable grant.
    pub fn with_query_plan_capability_evidence(mut self, evidence: AttemptRequirement) -> Self {
        self.planned_capability_evidence = Some(evidence);
        self
    }

    /// Internal controller/test constructor for synthetic capability evidence.
    ///
    /// Kept as crate-private API surface: the public-API boundary UI tests
    /// pin that callers outside the crate cannot attach their own requirement.
    #[cfg(test)]
    pub(crate) fn with_attempt_requirement(mut self, attempt_requirement: AttemptRequirement) -> Self {
        self.attempt_requirement = Some(attempt_requirement);
        self
    }

    pub(in crate::runtime::query_service) fn resolved_attempt_requirement(&self) -> Option<AttemptRequirement> {
        self.planned_capability_evidence.clone().or_else(|| self.attempt_requirement.clone())
    }

    /// Returns the evidence that was actually submitted by planning for the
    /// execution-time coverage check.  Internal requirements still size and
    /// place an admission, but must not impersonate a evidence for an arbitrary
    /// later-built plan.
    pub(in crate::runtime::query_service) fn submitted_plan_capability_evidence(&self) -> Option<AttemptRequirement> {
        self.planned_capability_evidence.clone()
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

    /// Retains an adapter-owned activity token through delayed native cleanup.
    #[cfg(test)]
    pub(crate) fn with_runtime_observation_activity(mut self, activity: QueryActivityLease) -> Self {
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

    pub(in crate::runtime::query_service::service) fn into_admission_observation(self) -> AdmissionObservation {
        let (activity, activity_error) = match self.runtime_observation_activity {
            Some(Ok(activity)) => (Some(activity), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        AdmissionObservation {
            handle: self.attempt.observation_handle(),
            activity,
            activity_error,
            attempt_id: self.attempt.attempt_id(),
        }
    }

    /// Claims the bound attempt's only admission right.
    pub(in crate::runtime::query_service::service) fn claim_attempt_admission(
        &self,
    ) -> std::result::Result<AttemptAdmission, AttemptLifecycleError> {
        self.attempt.claim()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use query_planning::{
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, SourceNode},
    };

    use super::*;
    use crate::{
        attempt::{AttemptBackendCounters, BeginAttemptRequest},
        observability::{AttemptDisposition, RuntimeObservationConfig},
        runtime::BackendLifecycle,
    };

    fn explicit_sink_capability_evidence() -> AttemptRequirement {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let plan = builder.finish_with_host_sink(source);
        AttemptRequirement::for_query_plan(&plan)
    }

    #[test]
    fn public_capability_attachment_retains_only_an_engine_issued_evidence() {
        let evidence = explicit_sink_capability_evidence();
        let request = QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence.clone());

        assert_eq!(request.submitted_plan_capability_evidence(), Some(evidence.clone()));
        assert_eq!(request.resolved_attempt_requirement(), Some(evidence));
    }

    #[test]
    fn attempt_binding_attaches_observation_and_release_together() {
        let lifecycle = BackendLifecycle::new();
        let attempt = AttemptOwner::begin(
            BeginAttemptRequest {
                observation: RuntimeObservationConfig::bounded(8, 8 * 1024),
                ..BeginAttemptRequest::default()
            },
            lifecycle.try_begin_observation().expect("attempt observation permit"),
            Arc::new(AttemptBackendCounters::default()),
            None,
        );
        let attempt_id = attempt.owner.attempt_id();
        attempt.owner.handle().set_disposition(AttemptDisposition::Native).expect("native disposition");
        let request = QueryAdmissionRequest::new(attempt.owner.admission_binding());

        assert!(request.claim_attempt_admission().is_ok());
        assert!(matches!(request.claim_attempt_admission(), Err(AttemptLifecycleError::AdmissionAlreadyClaimed)));
        let observation = request.into_admission_observation();
        assert_eq!(observation.attempt_id, attempt_id);
    }
}
