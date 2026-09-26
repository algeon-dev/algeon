//! Bounded native query-service admission harness.

use std::sync::Mutex;

#[cfg(test)]
use query_planning::policy::PlanningPolicy;
use query_planning::{
    admission::AttemptRequirement,
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    mode::StatisticalAggregateExecutionMode,
};
#[cfg(test)]
use tokio::time;

#[cfg(test)]
use super::LakehouseCacheFillConfig;
#[cfg(test)]
use super::{DeviceLedgerReport, GpuBackendDeviceConfig, GpuDeviceMemoryLimit, GpuMemoryOwnership};
use super::{
    DeviceRegistry, ExecutionCancellation, ExecutionContext, GpuExecutionJobKind, GpuExecutionPool,
    MemoryControlPolicy, MemoryGrant, SourceChunkPlanningMode, SourceSchedulerPolicy,
    context::{
        AdmittedAttempt, PlanScope, QueryServiceEpochUser, StreamBinding, memory_control_policy_for_grant,
        validate_service_stream_pool_device,
    },
};
use crate::{
    metrics::QueryId,
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, RuntimeAdmissionObservation, RuntimeObservationHandle,
        RuntimeObservationPayload, RuntimeTerminalOutcome,
    },
};

mod admission;
mod config;
mod device_waiters;
mod handle;
mod resources;
mod service;
#[cfg(test)]
mod tests;

#[cfg(test)]
use admission::QueryAdmissionWaiter;
use admission::QueryCompletion;
#[cfg(test)]
use admission::{AdmissionState, DeviceAdmissionEntries};
pub use config::{QueryExecutionConfig, QueueBounds};
use device_waiters::{PendingDeviceSlotNotifications, notify_shared_device_admission_available};
pub use handle::{AdmittedQueryContext, QueryHandle};
#[cfg(feature = "qualification")]
pub use service::DeviceQuarantineInjection;
#[cfg(test)]
pub(crate) use service::QueryAdmissionOutcome;
#[doc(hidden)]
pub use service::QueryService;
use service::QueryServiceInner;
pub use service::{
    QueryActivityLease, QueryAdmissionRequest, QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryServiceDrainPhase {
    StreamPoolClose,
    AdmissionQuiescence,
    ExecutionPoolQuiescence,
    StreamPoolQuiescence,
    ResourceEpoch,
    FinalCompletion,
}

impl QueryServiceDrainPhase {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StreamPoolClose => "stream_pool_close",
            Self::AdmissionQuiescence => "admission_quiescence",
            Self::ExecutionPoolQuiescence => "execution_pool_quiescence",
            Self::StreamPoolQuiescence => "stream_pool_quiescence",
            Self::ResourceEpoch => "resource_epoch",
            Self::FinalCompletion => "final_completion",
        }
    }
}

#[derive(Debug)]
pub(crate) enum QueryServiceDrainPhaseOutcome {
    Quiescent {
        phase: QueryServiceDrainPhase,
    },
    DeadlineElapsed {
        phase: QueryServiceDrainPhase,
        #[cfg_attr(
            not(any(test, feature = "test-hooks")),
            expect(dead_code, reason = "read only by into_result under test-hooks")
        )]
        error: QueryError,
    },
    PermanentFailure {
        phase: QueryServiceDrainPhase,
        error: QueryError,
    },
}

impl QueryServiceDrainPhaseOutcome {
    fn from_result(phase: QueryServiceDrainPhase, result: Result<()>) -> Self {
        match result {
            Ok(()) => Self::Quiescent { phase },
            Err(error) if error.is_cleanup_deadline_elapsed() => Self::DeadlineElapsed { phase, error },
            Err(error) => Self::PermanentFailure { phase, error },
        }
    }

    pub(crate) const fn phase(&self) -> QueryServiceDrainPhase {
        match self {
            Self::Quiescent { phase } | Self::DeadlineElapsed { phase, .. } | Self::PermanentFailure { phase, .. } => {
                *phase
            },
        }
    }

    pub(crate) const fn is_quiescent(&self) -> bool {
        matches!(self, Self::Quiescent { .. })
    }
}

/// One phase-total cleanup pass.
///
/// Every owned component contributes a typed outcome. A phase that depends on
/// unfinished earlier cleanup is reported as deadline-pending; in particular,
/// the cleanup executors must remain available for anchors released after an
/// earlier bounded admission wait times out.
#[derive(Debug)]
pub(crate) struct QueryServiceDrainOutcome {
    phases: Vec<QueryServiceDrainPhaseOutcome>,
}

impl QueryServiceDrainOutcome {
    fn new(phases: Vec<QueryServiceDrainPhaseOutcome>) -> Self {
        Self { phases }
    }

    pub(crate) fn phases(&self) -> &[QueryServiceDrainPhaseOutcome] {
        &self.phases
    }

    #[cfg(any(test, feature = "test-hooks"))]
    fn into_result(self) -> Result<()> {
        let mut first_deadline = None;
        for outcome in self.phases {
            match outcome {
                QueryServiceDrainPhaseOutcome::PermanentFailure { error, .. } => return Err(error),
                QueryServiceDrainPhaseOutcome::DeadlineElapsed { error, .. } if first_deadline.is_none() => {
                    first_deadline = Some(error);
                },
                QueryServiceDrainPhaseOutcome::Quiescent { .. }
                | QueryServiceDrainPhaseOutcome::DeadlineElapsed { .. } => {},
            }
        }
        first_deadline.map_or(Ok(()), Err)
    }
}

fn record_admission_lifecycle(
    observation: &RuntimeObservationHandle,
    admission_request_id: AdmissionRequestId,
    lifecycle: AdmissionLifecycleState,
    queue_depth: usize,
    device_ordinal: Option<usize>,
    grant_bytes: Option<u64>,
) {
    if !observation.is_enabled() {
        return;
    }
    observation.record(|| {
        RuntimeObservationPayload::Admission(Box::new(RuntimeAdmissionObservation {
            admission_request_id,
            lifecycle,
            queue_depth,
            device_ordinal: device_ordinal.and_then(|value| u32::try_from(value).ok()),
            grant_bytes,
        }))
    });
}

fn admission_rejection(error: &QueryError) -> AdmissionLifecycleState {
    AdmissionLifecycleState::Rejected {
        reason: error.fact_value("reason").unwrap_or_else(|| error.code().as_str()).to_string(),
    }
}

#[cfg(test)]
mod drain_outcome_tests {
    use super::*;

    #[test]
    fn later_permanent_failure_wins_over_earlier_deadline() {
        let deadline = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.query_service.test_cleanup",
            "injected cleanup deadline",
        )
        .with_fact("reason", "injected_deadline")
        .with_cleanup_deadline_elapsed();
        let permanent = QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.query_service.test_cleanup",
            "injected permanent cleanup failure",
        )
        .with_fact("reason", "injected_permanent_failure");
        let outcome = QueryServiceDrainOutcome::new(vec![
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::StreamPoolClose, Err(deadline)),
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::StreamPoolQuiescence, Err(permanent)),
        ]);

        let error = outcome.into_result().unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("injected_permanent_failure"));
    }

    #[test]
    fn non_deadline_service_overloaded_is_permanent() {
        let overloaded = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.query_service.test_cleanup",
            "ordinary overload is not a cleanup deadline",
        )
        .with_fact("reason", "ordinary_service_overload");
        let outcome =
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::AdmissionQuiescence, Err(overloaded));

        assert!(matches!(outcome, QueryServiceDrainPhaseOutcome::PermanentFailure { .. }));
    }
}
