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
#[cfg(feature = "cugraph")]
include!("tests/warm_graph_pricing.rs");

// This exercises registry-backed admission as a child of the private service
// module. Keeping the state-machine coverage here prevents the integration
// test crate from becoming an alternative public construction client.
#[path = "../../../tests/native_contract_tests/multi_device_registry.rs"]
mod multi_device_registry;
