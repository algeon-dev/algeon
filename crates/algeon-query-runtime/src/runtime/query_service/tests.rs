use std::{
    collections::BTreeMap,
    sync::{
        Arc, Barrier, PoisonError, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DeviceId, ErrorKind as CudfErrorKind, Exec, QueryMemoryDomainUsage, Runtime as CudfRuntime,
    runtime::{DeviceFaultKind, DeviceFaultNativeStatus, report_device_fault},
    testing::device_safety::{DeviceSafetyTestScope, injected_device_id},
};
use grant_requests::explicit_sink_plan;
use observation::observed_attempt;
use query_planning::{
    admission::{AttemptCapabilityEvidence, ExecutionCapability},
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    error::{AttemptFailureCode, QueryErrorKind, QueryErrorStatus},
    plan::QueryPlan,
};
use support::{
    AdmissionTestController, AdmissionTestHandle, admission_test_service_with_profile, distinct_capability_requirement,
    first_memory_governor_report, observed_admission_events, requirement, runtime_test_device_profile,
    runtime_test_device_profile_with_limits, runtime_test_service, runtime_test_service_with_profiles, test_ledger,
    test_ledger_for_ordinal, test_service, try_guaranteed_runtime_test_service,
    try_guaranteed_runtime_test_service_with_concurrency, wait_for_exact_queued_len, wait_for_pending_preflight_len,
    wait_for_queued_len, wait_for_sealed_observation,
};
use tokio::{
    sync::{Barrier as TokioBarrier, Semaphore, mpsc::unbounded_channel, oneshot},
    task::yield_now,
    time::{Instant as TokioInstant, sleep_until, timeout},
};

use super::{
    admission::{AdmissionLifecycleSnapshot, AdmissionTicket, AdmissionWaiter, PlacedAdmission, PlacedQuery},
    controller::{AdmissionController, AdmissionDevice, AdmissionDevices, PendingSubmission},
    *,
};
use crate::{
    attempt::{AttemptBackendCounters, AttemptCompletion, AttemptLifecycleError, AttemptOwner, BeginAttemptRequest},
    metrics::QueryMetrics,
    native_error::cudf_query_error,
    observability::{
        AdmissionLifecycleState, AttemptAllocatorEventAction, AttemptAllocatorEventSource, AttemptCapacityDisposition,
        AttemptCleanupDisposition, AttemptDeviceHealthStatus, AttemptDisposition, AttemptEvidenceField,
        AttemptEvidenceProvenance, AttemptFailureFactKey, AttemptFailureFactValue, AttemptFailureOperation,
        AttemptFailureStatus, AttemptFailureSummary, AttemptFaultIsolation, AttemptHealthTransition,
        AttemptNativeFaultKind, AttemptNativeStatus, AttemptQueryId, AttemptRetryAdvice, AttemptRetryPrerequisite,
        AttemptRetryRecovery, AttemptTeardownHealth, AttemptTraceContext, CleanupFailurePhase, CleanupFailureSummary,
        GpuJobLifecycleState, LakehouseCacheFillEvent, LakehouseCacheFillOutcome, PlanExecutionId, QueryBoundaryKind,
        RuntimeObservationConfig, RuntimeObservationPayload, RuntimePlanExecutionLifecycle, RuntimeTerminalOutcome,
        RuntimeTimelineEvent, SealedRuntimeObservationSnapshot, StreamLeaseLifecycleState,
    },
    runtime::{
        BackendLifecycle, DeviceCapabilities, DeviceResourceEnvelope, DeviceResourceProfile,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, StandaloneConfig,
        context::AdmittedPlanIdentities,
        gpu::{DeviceAdmissionHealth, DeviceLedger, GrantIdentity},
        gpu_execution_pool::RESERVED_LIFECYCLE_WORKERS,
        query_service::service::{
            disposition::{
                CapacityCommitDisposition, CleanupEvidence, decide_capacity_commit, decide_capacity_disposition,
            },
            transaction::{AdmissionStep, AdmissionTransactionProbe},
        },
        resident::{GpuDataFrame, GpuResultHandle},
        service_stream::{StreamCleanupOutcome, block_stream_cleanup_thread_for_test},
    },
};

mod config_resources;
mod contexts;
mod grant_requests;
mod observation;
mod placement;
mod queue;
mod release_commit;
mod support;
mod transaction;

// This exercises registry-backed admission as a child of the private service
// module. Keeping the state-machine coverage here prevents the integration
// test crate from becoming an alternative public construction client.
mod multi_device_registry;
