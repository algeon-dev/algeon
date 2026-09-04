use std::{
    collections::BTreeMap,
    sync::{
        Arc, Barrier, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use nexus_query_planning::graph_memory::GraphMemoryCalibrationOverride;

use super::{admission::AdmissionTicket, *};
use crate::{
    observability::{
        AdmissionLifecycleState, AttemptFailureCode, AttemptFailureStatus, GpuJobLifecycleState, PlanExecutionId,
        QueryBoundaryKind, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeObservationPayload,
        RuntimePlanExecutionLifecycle, RuntimeTerminalOutcome, RuntimeTraceId, StreamLeaseLifecycleState,
    },
    runtime::{
        DeviceResourceEnvelope, DeviceResourceProfile, GdsDriverProbeStatus, GpuBackendDeviceInstallationBounds,
        GpuBackendDeviceProfile, KvikioCompatModeStatus, NvidiaFsProbeStatus,
    },
};

include!("tests/support.rs");
include!("tests/observation.rs");
include!("tests/grant_requests.rs");
include!("tests/config_resources.rs");
include!("tests/contexts.rs");
include!("tests/queue.rs");
include!("tests/overtake.rs");
include!("tests/transaction.rs");
include!("tests/release_commit.rs");
include!("tests/placement.rs");
// This exercises registry-backed admission as a child of the private service
// module. Keeping the state-machine coverage here prevents the integration
// test crate from becoming an alternative public construction client.
#[path = "tests/multi_device_registry.rs"]
mod multi_device_registry;
