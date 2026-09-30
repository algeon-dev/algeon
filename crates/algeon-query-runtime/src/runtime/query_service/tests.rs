use std::{
    sync::{
        Arc, Barrier, Weak,
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
    DataFrame, DeviceId, ErrorKind as CudfErrorKind, QueryMemoryDomainUsage, Runtime as CudfRuntime,
    runtime::{DeviceFaultKind, DeviceFaultNativeStatus, report_device_fault},
    testing::device_safety::DeviceSafetyTestScope,
};
use query_planning::admission::AttemptCapabilityEvidence;
use tokio::{
    sync::{Barrier as TokioBarrier, Semaphore, oneshot},
    task::yield_now,
    time::{Instant as TokioInstant, sleep_until, timeout},
};

use super::{admission::AdmissionTicket, *};
use crate::{
    attempt::{AttemptBackendCounters, AttemptLifecycleError, AttemptOwner, BeginAttemptRequest},
    observability::{
        AdmissionLifecycleState, AttemptAdmissionFact, AttemptAllocatorEventAction, AttemptAllocatorEventSource,
        AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDeviceHealthStatus, AttemptDisposition,
        AttemptEvidenceField, AttemptEvidenceProvenance, AttemptFailureCode, AttemptFailureStatus,
        AttemptFaultIsolation, AttemptHealthTransition, AttemptNativeFaultKind, AttemptNativeStatus,
        AttemptRetryAdvice, AttemptRetryPrerequisite, AttemptRetryRecovery, AttemptTeardownHealth, CleanupFailurePhase,
        CleanupFailureSummary, GpuJobLifecycleState, PlanExecutionId, QueryBoundaryKind, RuntimeObservationConfig,
        RuntimeObservationOwner, RuntimeObservationPayload, RuntimePlanExecutionLifecycle, RuntimeTerminalOutcome,
        RuntimeTraceId, StreamLeaseLifecycleState,
    },
    runtime::{
        BackendLifecycle, DeviceResourceEnvelope, DeviceResourceProfile, GpuBackendDeviceInstallationBounds,
        GpuBackendDeviceProfile, LakehouseCacheFillEvent, LakehouseCacheFillOutcome,
        gpu::DeviceAdmissionHealth,
        resident::{GpuDataFrame, GpuResultHandle},
        service_stream::{StreamCleanupOutcome, block_stream_cleanup_thread_for_test},
    },
};

include!("tests/support.rs");
include!("tests/observation.rs");
include!("tests/grant_requests.rs");
include!("tests/config_resources.rs");
include!("tests/contexts.rs");
include!("tests/queue.rs");
include!("tests/transaction.rs");
include!("tests/release_commit.rs");
include!("tests/placement.rs");
// This exercises registry-backed admission as a child of the private service
// module. Keeping the state-machine coverage here prevents the integration
// test crate from becoming an alternative public construction client.
#[path = "tests/multi_device_registry.rs"]
mod multi_device_registry;
