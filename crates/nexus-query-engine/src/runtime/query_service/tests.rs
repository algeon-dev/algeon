use std::sync::{
    Barrier,
    atomic::{AtomicUsize, Ordering},
};

use super::{super::memory_domain::grant_memory_domain_creation_locks, *};
use crate::{
    graph_memory::GraphMemoryCalibrationOverride,
    observability::{
        AdmissionLifecycleState, GpuJobLifecycleState, PlanExecutionId, QueryBoundaryKind, RuntimeObservationConfig,
        RuntimeObservationOwner, RuntimeObservationPayload, RuntimePlanExecutionLifecycle, RuntimeTerminalOutcome,
        RuntimeTraceId, StreamLeaseLifecycleState,
    },
    runtime::{GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus},
};

include!("tests/support.rs");
include!("tests/observation.rs");
include!("tests/grant_requests.rs");
include!("tests/config_resources.rs");
include!("tests/contexts.rs");
include!("tests/queue.rs");
include!("tests/adaptive.rs");
include!("tests/placement.rs");
