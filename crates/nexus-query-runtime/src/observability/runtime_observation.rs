//! Typed runtime observations, topology, and bounded retention records.

use std::fmt;

use nexus_query_planning::{
    correlation::QueryId,
    execution_contracts::{RuntimeTaskId, TaskAttemptId},
};
use serde::{Deserialize, Serialize};

use super::attempt_report::{CleanupFailureSummary, QueryAttemptReport, RuntimeTerminalOutcome};

/// JSONL schema emitted by sealed runtime observation snapshots.
pub const RUNTIME_OBSERVATION_JSONL_SCHEMA: &str = "nexus_query_engine.runtime_observation.v2";

macro_rules! string_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Creates an identity from its stable, scope-local label.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Returns the stable label.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

string_id!(PlanExecutionId, "Identity for one invocation of a native plan.");
string_id!(RuntimeTraceId, "Optional parent trace identity for one observation session.");
string_id!(RuntimePlanNodeId, "Plan-node identity scoped to one plan execution.");
string_id!(RuntimeReservationId, "Reservation identity scoped to one plan execution.");
string_id!(RuntimeResourceId, "Resident-resource identity scoped to one plan execution.");
string_id!(RuntimeJobId, "Executor-job identity scoped to one query.");
string_id!(RuntimeWorkerId, "Worker identity scoped to one query.");
string_id!(RuntimePipelineEdgeId, "Compiled-pipeline edge identity scoped to one plan execution.");
string_id!(RuntimeBarrierId, "Compiled pipeline-barrier identity scoped to one plan execution.");
string_id!(RuntimeSinkId, "Compiled pipeline-sink identity scoped to one plan execution.");
string_id!(RuntimeReservationGateId, "Reservation-gate identity scoped to one plan execution.");

/// Service-local identity for one admission attempt.
///
/// This value is never a query identity. It is meaningful only for joining
/// admission lifecycle events inside one process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AdmissionRequestId(u64);

impl AdmissionRequestId {
    /// Creates a service-local admission request identity.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the service-local numeric value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AdmissionRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Opaque query-scoped identity for one source location.
///
/// Construction is restricted to the observation session's keyed digest path;
/// callers can inspect or deserialize an exported identity but cannot turn raw
/// source text into an apparently valid identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeSourceLocationId(String);

impl RuntimeSourceLocationId {
    #[doc(hidden)]
    pub fn from_keyed_digest(value: String) -> Self {
        Self(value)
    }

    /// Returns the opaque query-scoped label.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RuntimeSourceLocationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Immutable canonical query identity for one observation session.
///
/// This delegates to [`QueryId`] so manifests and execution contexts cannot
/// retain different representations of blank or whitespace-only identities.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct RuntimeQueryId(String);

impl RuntimeQueryId {
    /// Creates an identity using the native query-id canonicalization policy.
    pub fn new(value: impl Into<String>) -> Self {
        Self(QueryId::new(value).to_string())
    }

    /// Returns the canonical query identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<QueryId> for RuntimeQueryId {
    fn from(value: QueryId) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for RuntimeQueryId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RuntimeQueryId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for RuntimeQueryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RuntimeQueryId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        String::deserialize(deserializer).map(Self::new)
    }
}

/// Total recorder order assigned to every attempted record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationSequence(u64);

impl ObservationSequence {
    #[doc(hidden)]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the one-based sequence value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Span identity scoped to one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeSpanId(u64);

impl RuntimeSpanId {
    /// Creates a span identity from a query-local value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the query-local value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Recorder limits copied into the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationLimits {
    /// Maximum retained topology rows and timeline events.
    pub max_retained_records: u64,
    /// Maximum deterministic logical bytes charged to retained records.
    pub max_retained_bytes: u64,
}

/// Clock domain for host runtime observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeClockDomain {
    /// Query-relative time sampled from a monotonic host clock.
    HostMonotonic,
}

/// First JSONL record describing immutable query identity and capture limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationManifest {
    /// Versioned JSONL schema label.
    pub schema: String,
    /// Query-wide diagnostic identity.
    pub query_id: RuntimeQueryId,
    /// Optional parent trace identity.
    pub trace_id: Option<RuntimeTraceId>,
    /// Optional wall-clock anchor captured once when the session is created.
    pub unix_epoch_ns: Option<u64>,
    /// Clock domain used by timeline offsets.
    pub clock_domain: RuntimeClockDomain,
    /// Deterministic per-query retention limits.
    pub limits: RuntimeObservationLimits,
}

/// One native plan node in the static topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanNodeRow {
    pub plan_execution_id: PlanExecutionId,
    pub node_id: RuntimePlanNodeId,
    pub operator_kind: String,
}

/// One directed edge in the static native plan topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanEdgeRow {
    pub plan_execution_id: PlanExecutionId,
    pub from_node_id: RuntimePlanNodeId,
    pub to_node_id: RuntimePlanNodeId,
    pub input_ordinal: u32,
}

/// One compiled task definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskDefinitionRow {
    pub plan_execution_id: PlanExecutionId,
    pub task_id: RuntimeTaskId,
    pub plan_node_id: Option<RuntimePlanNodeId>,
    pub task_kind: String,
}

/// One compiled dependency edge in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompiledEdgeRow {
    pub plan_execution_id: PlanExecutionId,
    pub edge_id: RuntimePipelineEdgeId,
    pub from_task_id: RuntimeTaskId,
    pub to_task_id: RuntimeTaskId,
    pub edge_kind: String,
    pub resource_ids: Vec<RuntimeResourceId>,
    pub gate_ids: Vec<RuntimeReservationGateId>,
}

/// One compiled barrier definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BarrierDefinitionRow {
    pub plan_execution_id: PlanExecutionId,
    pub barrier_id: RuntimeBarrierId,
    pub barrier_kind: String,
    pub upstream_task_ids: Vec<RuntimeTaskId>,
    pub downstream_task_ids: Vec<RuntimeTaskId>,
    pub gate_ids: Vec<RuntimeReservationGateId>,
}

/// The compiled result-sink definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SinkDefinitionRow {
    pub plan_execution_id: PlanExecutionId,
    pub sink_id: RuntimeSinkId,
    pub sink_kind: String,
    pub input_task_id: RuntimeTaskId,
    pub gate_ids: Vec<RuntimeReservationGateId>,
}

/// One reservation gate and its owner in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReservationGateRow {
    pub plan_execution_id: PlanExecutionId,
    pub gate_id: RuntimeReservationGateId,
    pub owner_kind: String,
    pub owner_id: String,
    pub reservation_kind: String,
    pub requirement_scope: String,
    pub memory_boundary: String,
    pub required_contract: String,
    pub contract_status: String,
    pub contract_quality: String,
}

/// One GPU executor job definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JobDefinitionRow {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub job_id: RuntimeJobId,
    pub job_kind: String,
    pub device_ordinal: u32,
}

/// One host worker definition used by GPU or source execution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerDefinitionRow {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub worker_id: RuntimeWorkerId,
    pub worker_kind: String,
    pub device_ordinal: Option<u32>,
}

/// Resource domain carried by a static resource definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeResourceDomain {
    Host,
    Device { device_ordinal: u32 },
    Storage,
}

/// One resident resource in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuntimeResourceRow {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub resource_id: RuntimeResourceId,
    pub resource_kind: String,
    pub domain: RuntimeResourceDomain,
    pub producer_task_id: Option<RuntimeTaskId>,
    pub consumer_task_ids: Vec<RuntimeTaskId>,
    pub lifetime: String,
}

/// Static topology accepted by the bounded recorder.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuntimeTopologyRecord {
    PlanNode(PlanNodeRow),
    PlanEdge(PlanEdgeRow),
    TaskDefinition(TaskDefinitionRow),
    CompiledEdge(CompiledEdgeRow),
    BarrierDefinition(BarrierDefinitionRow),
    SinkDefinition(SinkDefinitionRow),
    ReservationGate(ReservationGateRow),
    JobDefinition(JobDefinitionRow),
    WorkerDefinition(WorkerDefinitionRow),
    Resource(RuntimeResourceRow),
}

/// Query-level observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryBoundaryKind {
    Started,
    CleanupStarted,
    CleanupFinished,
    Succeeded,
    Failed,
    Cancelled,
    Dropped,
    Panicked,
}

/// Plan-execution observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanExecutionBoundaryKind {
    Started,
    Finished,
    Failed,
    Cancelled,
}

/// Exactly-once lifecycle state for one compiled task attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TaskLifecycleState {
    Ready,
    Started,
    Finished,
    Failed { error_code: String },
    Cancelled,
    Dropped,
    Panicked,
}

/// Reservation operation recorded at the ownership boundary that applied it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReservationAction {
    Reserve,
    Grow,
    Shrink,
    Transfer,
    Alias,
    Relabel,
    SourceCachePromote,
    SourceCacheEvict,
    Release,
    Deny,
    Rollback,
}

/// Terminal or nonterminal ownership disposition for a reservation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReservationDisposition {
    Applied,
    Granted,
    Denied,
    ExplicitRelease,
    DropRelease,
    UnwindRelease,
    CleanupRelease,
    RollbackRelease,
    AliasDeferred,
    CancelledRelease,
}

/// Query-service admission lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AdmissionLifecycleState {
    Enqueued,
    Admitted,
    Rejected { reason: String },
    TimedOut,
    CancellationRequested,
    Cancelled,
    Released,
    Dropped,
}

/// GPU execution-pool job lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GpuJobLifecycleState {
    Enqueued,
    Started,
    Finished,
    Failed { error_code: String },
    CancellationRequested,
    Cancelled,
    Dropped,
    Panicked,
}

/// How a service stream lease was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamLeaseAcquisition {
    Reused,
    Created,
}

/// Service stream-lease lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StreamLeaseLifecycleState {
    Enqueued,
    Acquired {
        acquisition: StreamLeaseAcquisition,
        host_wait_ns: u64,
    },
    CancellationRequested,
    Cancelled,
    Failed {
        error_code: String,
    },
    /// Query ownership transferred synchronously to pool-managed cleanup.
    Released,
}

/// Explicit correlation available when a service-stream lease is acquired.
///
/// Callers compose this value at execution boundaries; stream acquisition does
/// not consult thread-local or task-local ambient state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStreamLeaseScope {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub task_id: Option<RuntimeTaskId>,
    pub source_worker_id: Option<RuntimeWorkerId>,
    pub gpu_job_id: Option<RuntimeJobId>,
    pub gpu_worker_id: Option<RuntimeWorkerId>,
}

impl RuntimeStreamLeaseScope {
    pub fn for_plan(plan_execution_id: PlanExecutionId) -> Self {
        Self { plan_execution_id: Some(plan_execution_id), ..Self::default() }
    }

    pub fn with_source_worker(mut self, task_id: RuntimeTaskId, worker_id: RuntimeWorkerId) -> Self {
        self.task_id = Some(task_id);
        self.source_worker_id = Some(worker_id);
        self
    }

    pub fn with_gpu_worker(mut self, job_id: RuntimeJobId, worker_id: RuntimeWorkerId) -> Self {
        self.gpu_job_id = Some(job_id);
        self.gpu_worker_id = Some(worker_id);
        self
    }
}

/// Source-read scheduler lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SourceReadLifecycleState {
    Enqueued,
    WorkerStarted { queue_wait_ns: u64 },
    ReadFinished { read_duration_ns: u64 },
    ReadFailed { error_code: String, read_duration_ns: u64 },
    ConsumerJoinStarted,
    ConsumerJoinFinished { join_duration_ns: u64 },
    Cancelled,
    Panicked,
}

/// Scope of one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMeasurementScope {
    Query,
    Task,
    Process,
    DeviceGlobal,
}

/// Provenance of one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMeasurementSource {
    ReservationLedger,
    FrameResidency,
    AllocatorHighWater,
    DeviceSampler,
    StreamPool,
}

/// Evidence quality of one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMeasurementQuality {
    Exact,
    Observed,
    Estimated,
}

macro_rules! runtime_observation_body {
    ($(#[$meta:meta])* $name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $name {
            $(pub $field: $ty),*
        }
    };
}

runtime_observation_body!(
    /// Task-lifecycle payload body retained out of line.
    RuntimeTaskLifecycleObservation {
        attempt_id: TaskAttemptId,
        task_kind: String,
        lifecycle: TaskLifecycleState,
    }
);

runtime_observation_body!(
    /// Admission-lifecycle payload body retained out of line.
    RuntimeAdmissionObservation {
        admission_request_id: AdmissionRequestId,
        lifecycle: AdmissionLifecycleState,
        queue_depth: usize,
        device_ordinal: Option<u32>,
        grant_bytes: Option<u64>,
    }
);

runtime_observation_body!(
    /// GPU-job payload body retained out of line.
    RuntimeGpuJobObservation {
        job_id: RuntimeJobId,
        job_kind: String,
        device_ordinal: u32,
        worker_id: Option<RuntimeWorkerId>,
        lifecycle: GpuJobLifecycleState,
    }
);

runtime_observation_body!(
    /// Stream-lease payload body retained out of line.
    RuntimeStreamLeaseObservation {
        lease_id: RuntimeResourceId,
        device_ordinal: u32,
        scope: RuntimeStreamLeaseScope,
        lifecycle: StreamLeaseLifecycleState,
    }
);

runtime_observation_body!(
    /// Source-read payload body retained out of line.
    RuntimeSourceReadObservation {
        task_id: RuntimeTaskId,
        worker_id: RuntimeWorkerId,
        chunk_index: usize,
        file_index: usize,
        lifecycle: SourceReadLifecycleState,
    }
);

runtime_observation_body!(
    /// Resource-measurement payload body retained out of line.
    RuntimeResourceMeasurementObservation {
        resource_id: RuntimeResourceId,
        scope: RuntimeMeasurementScope,
        source: RuntimeMeasurementSource,
        quality: RuntimeMeasurementQuality,
        measurement_kind: String,
        current_value: u64,
        delta: i128,
        limit: Option<u64>,
        device_ordinal: Option<u32>,
        grant_bytes: Option<u64>,
    }
);

/// Context-rich reservation payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReservationObservation {
    pub reservation_id: RuntimeReservationId,
    pub lineage_id: RuntimeReservationId,
    pub task_attempt_id: Option<TaskAttemptId>,
    pub task_kind: Option<String>,
    pub action: RuntimeReservationAction,
    pub denied_action: Option<RuntimeReservationAction>,
    pub disposition: RuntimeReservationDisposition,
    pub alias: bool,
    pub reservation_kind: String,
    pub memory_space: String,
    pub memory_boundary: String,
    pub requested_bytes: u64,
    pub granted_bytes: u64,
    pub charged_delta_bytes: i128,
    pub current_reserved_bytes: u64,
    pub budget_bytes: u64,
    pub node_id: Option<String>,
    pub stage: Option<String>,
    pub chunk_index: Option<usize>,
    pub file_index: Option<usize>,
    pub source_location_id: Option<RuntimeSourceLocationId>,
    pub row_groups: Option<Vec<usize>>,
    pub attempt: Option<usize>,
    pub max_attempts: Option<usize>,
    pub resident_resource_id: Option<String>,
    pub source_read_contract_quality: Option<String>,
    pub source_read_planning_bytes: Option<u64>,
    pub source_read_estimated_bytes: Option<u64>,
}

runtime_observation_body!(
    /// Secondary-cleanup-failure payload body retained out of line.
    RuntimeCleanupFailureObservation {
        summary: CleanupFailureSummary,
    }
);

/// Operator-substage category for blocking native primitives.
///
/// Pair events by [`RuntimeTimelineEvent::span_id`]. Kinds must stay short and
/// free of SQL, paths, or unbounded diagnostic text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOperatorStageKind {
    RegexReplacePreflight,
    RegexReplaceExecute,
    ExactCardinalityProbe,
    PreparedGroupbyPrepare,
    PreparedGroupbyAggregate,
    LibcudfGroupby,
    HashPartition,
    PartialAggregate,
    FinalMerge,
    PartitionLocalTopK,
    FinalSort,
    VectorPack,
    VectorNative,
    VectorAssemble,
}

impl RuntimeOperatorStageKind {
    /// Stable snake_case label used in facts and JSON.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RegexReplacePreflight => "regex_replace_preflight",
            Self::RegexReplaceExecute => "regex_replace_execute",
            Self::ExactCardinalityProbe => "exact_cardinality_probe",
            Self::PreparedGroupbyPrepare => "prepared_groupby_prepare",
            Self::PreparedGroupbyAggregate => "prepared_groupby_aggregate",
            Self::LibcudfGroupby => "libcudf_groupby",
            Self::HashPartition => "hash_partition",
            Self::PartialAggregate => "partial_aggregate",
            Self::FinalMerge => "final_merge",
            Self::PartitionLocalTopK => "partition_local_top_k",
            Self::FinalSort => "final_sort",
            Self::VectorPack => "vector_pack",
            Self::VectorNative => "vector_native",
            Self::VectorAssemble => "vector_assemble",
        }
    }
}

impl fmt::Display for RuntimeOperatorStageKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Lifecycle state for one operator-substage span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RuntimeOperatorStageState {
    Started,
    Finished,
    Failed { error_code: String },
    Cancelled,
}

/// Operator-substage payload body retained out of line.
///
/// Safe numerical facts only: no SQL, raw paths, or unbounded strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOperatorStageObservation {
    pub stage_kind: RuntimeOperatorStageKind,
    pub lifecycle: RuntimeOperatorStageState,
    pub node_id: Option<String>,
    pub task_id: Option<RuntimeTaskId>,
    pub chunk_index: Option<usize>,
    pub partition_index: Option<usize>,
    pub attempt: Option<u32>,
    pub strategy_label: Option<String>,
    pub input_rows: Option<u64>,
    pub output_rows: Option<u64>,
    pub input_bytes: Option<u64>,
    pub output_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_relation_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_dimension: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix_layout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_binding_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_ordinal: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_version_major: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_version_minor: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_version_patch: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounded_control_scalar_events: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounded_control_scalar_bytes: Option<u64>,
}

impl RuntimeOperatorStageObservation {
    /// Creates a stage observation with only the required identity fields set.
    pub fn new(stage_kind: RuntimeOperatorStageKind, lifecycle: RuntimeOperatorStageState) -> Self {
        Self {
            stage_kind,
            lifecycle,
            node_id: None,
            task_id: None,
            chunk_index: None,
            partition_index: None,
            attempt: None,
            strategy_label: None,
            input_rows: None,
            output_rows: None,
            input_bytes: None,
            output_bytes: None,
            reserved_bytes: None,
            vector_algorithm: None,
            vector_relation_role: None,
            vector_dimension: None,
            matrix_layout: None,
            execution_binding_id: None,
            device_ordinal: None,
            native_version_major: None,
            native_version_minor: None,
            native_version_patch: None,
            bounded_control_scalar_events: None,
            bounded_control_scalar_bytes: None,
        }
    }

    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    pub fn with_task_id(mut self, task_id: RuntimeTaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }

    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    pub fn with_partition_index(mut self, partition_index: usize) -> Self {
        self.partition_index = Some(partition_index);
        self
    }

    pub fn with_attempt(mut self, attempt: u32) -> Self {
        self.attempt = Some(attempt);
        self
    }

    pub fn with_strategy_label(mut self, strategy_label: impl Into<String>) -> Self {
        self.strategy_label = Some(strategy_label.into());
        self
    }

    pub fn with_input_rows(mut self, input_rows: u64) -> Self {
        self.input_rows = Some(input_rows);
        self
    }

    pub fn with_output_rows(mut self, output_rows: u64) -> Self {
        self.output_rows = Some(output_rows);
        self
    }

    pub fn with_input_bytes(mut self, input_bytes: u64) -> Self {
        self.input_bytes = Some(input_bytes);
        self
    }

    pub fn with_output_bytes(mut self, output_bytes: u64) -> Self {
        self.output_bytes = Some(output_bytes);
        self
    }

    pub fn with_reserved_bytes(mut self, reserved_bytes: u64) -> Self {
        self.reserved_bytes = Some(reserved_bytes);
        self
    }

    pub fn with_vector_algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.vector_algorithm = Some(algorithm.into());
        self
    }

    pub fn with_vector_relation_role(mut self, role: impl Into<String>) -> Self {
        self.vector_relation_role = Some(role.into());
        self
    }

    pub fn with_vector_dimension(mut self, dimension: usize) -> Self {
        self.vector_dimension = Some(dimension);
        self
    }

    pub fn with_matrix_layout(mut self, layout: impl Into<String>) -> Self {
        self.matrix_layout = Some(layout.into());
        self
    }

    pub fn with_execution_binding_id(mut self, binding_id: u64) -> Self {
        self.execution_binding_id = Some(binding_id);
        self
    }

    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    pub fn with_native_version(mut self, major: u16, minor: u16, patch: u16) -> Self {
        self.native_version_major = Some(major);
        self.native_version_minor = Some(minor);
        self.native_version_patch = Some(patch);
        self
    }

    pub fn with_bounded_control_scalars(mut self, events: u64, bytes: u64) -> Self {
        self.bounded_control_scalar_events = Some(events);
        self.bounded_control_scalar_bytes = Some(bytes);
        self
    }

    #[doc(hidden)]
    pub fn logical_retained_bytes(&self) -> u64 {
        string_bytes(self.stage_kind.as_str())
            + match &self.lifecycle {
                RuntimeOperatorStageState::Failed { error_code } => string_bytes(error_code),
                _ => 1,
            }
            + self.node_id.as_ref().map_or(0, |value| string_bytes(value))
            + self.task_id.as_ref().map_or(0, |value| string_bytes(value.as_str()))
            + self.strategy_label.as_ref().map_or(0, |value| string_bytes(value))
            + 48
    }
}

/// Typed dynamic event payload. Later phases extend this enum at authoritative boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeObservationPayload {
    QueryBoundary {
        boundary: QueryBoundaryKind,
    },
    PlanExecutionBoundary {
        boundary: PlanExecutionBoundaryKind,
    },
    TaskLifecycle(Box<RuntimeTaskLifecycleObservation>),
    Admission(Box<RuntimeAdmissionObservation>),
    GpuJob(Box<RuntimeGpuJobObservation>),
    StreamLease(Box<RuntimeStreamLeaseObservation>),
    SourceRead(Box<RuntimeSourceReadObservation>),
    ResourceMeasurement(Box<RuntimeResourceMeasurementObservation>),
    Reservation(Box<RuntimeReservationObservation>),
    /// Secondary cleanup failure retained without replacing a primary query error.
    CleanupFailure(Box<RuntimeCleanupFailureObservation>),
    /// Operator-substage lifecycle for blocking native primitives.
    OperatorStage(Box<RuntimeOperatorStageObservation>),
}

/// One dynamic event in recorder total order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimelineEvent {
    pub sequence: ObservationSequence,
    pub monotonic_ns: u64,
    pub plan_execution_id: Option<PlanExecutionId>,
    pub span_id: Option<RuntimeSpanId>,
    pub parent_span_id: Option<RuntimeSpanId>,
    pub payload: RuntimeObservationPayload,
}

/// Retention class used by explicit loss accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObservationRecordClass {
    PlanNode,
    PlanEdge,
    TaskDefinition,
    CompiledEdge,
    BarrierDefinition,
    SinkDefinition,
    ReservationGate,
    JobDefinition,
    WorkerDefinition,
    Resource,
    TimelineEvent,
}

/// Reason an attempted record was not retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObservationLossReason {
    RecordLimit,
    ByteLimit,
    Oversized,
    SourceLocationKeyUnavailable,
}

/// Count of dropped records in one class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationClassLoss {
    pub class: RuntimeObservationRecordClass,
    pub count: u64,
}

/// Count of dropped records for one reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationReasonLoss {
    pub reason: RuntimeObservationLossReason,
    pub count: u64,
}

fn serialize_optional_attempt_report<S>(
    report: &Option<std::sync::Arc<QueryAttemptReport>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match report {
        Some(report) => serializer.serialize_some(report.as_ref()),
        None => serializer.serialize_none(),
    }
}

fn deserialize_optional_attempt_report<'de, D>(
    deserializer: D,
) -> Result<Option<std::sync::Arc<QueryAttemptReport>>, D::Error>
where D: serde::Deserializer<'de> {
    Option::<QueryAttemptReport>::deserialize(deserializer).map(|report| report.map(std::sync::Arc::new))
}

/// Final loss, completeness, and terminal metadata.
///
/// An attempt-owned snapshot may also carry the exact immutable terminal
/// report. The `Arc` is shared with attempt completion; it does not create a
/// second report or event store. Keeping it in the exported footer lets the
/// retained JSONL carry the same sealed facts as the other projections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationFooter {
    pub complete: bool,
    pub terminal: Option<RuntimeTerminalOutcome>,
    pub retained_records: u64,
    pub retained_bytes: u64,
    pub first_dropped_sequence: Option<ObservationSequence>,
    pub last_attempted_sequence: Option<ObservationSequence>,
    pub dropped_by_class: Vec<RuntimeObservationClassLoss>,
    pub dropped_by_reason: Vec<RuntimeObservationReasonLoss>,
    pub truncated: bool,
    pub degraded: bool,
    pub poison_recoveries: u64,
    /// The exact sealed attempt report, when this observation belongs to an
    /// execution attempt. This is optional so non-attempt observations retain
    /// their existing JSONL shape.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_attempt_report",
        deserialize_with = "deserialize_optional_attempt_report"
    )]
    pub attempt_report: Option<std::sync::Arc<QueryAttemptReport>>,
    /// At least one recorder sequence or accounting counter saturated.
    ///
    /// Saturated values remain monotonic, while this marker prevents exported
    /// observations from presenting them as exact.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub counter_overflow: bool,
}

/// One independently parseable JSONL record.
// Timeline records retain the stable public enum shape even though reservation
// payloads are deliberately context-rich.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "record_type", content = "data", rename_all = "snake_case")]
pub enum RuntimeObservationRecord {
    Manifest(RuntimeObservationManifest),
    PlanNode { sequence: ObservationSequence, row: PlanNodeRow },
    PlanEdge { sequence: ObservationSequence, row: PlanEdgeRow },
    TaskDefinition { sequence: ObservationSequence, row: TaskDefinitionRow },
    CompiledEdge { sequence: ObservationSequence, row: CompiledEdgeRow },
    BarrierDefinition { sequence: ObservationSequence, row: BarrierDefinitionRow },
    SinkDefinition { sequence: ObservationSequence, row: SinkDefinitionRow },
    ReservationGate { sequence: ObservationSequence, row: ReservationGateRow },
    JobDefinition { sequence: ObservationSequence, row: JobDefinitionRow },
    WorkerDefinition { sequence: ObservationSequence, row: WorkerDefinitionRow },
    Resource { sequence: ObservationSequence, row: RuntimeResourceRow },
    TimelineEvent(RuntimeTimelineEvent),
    Footer(RuntimeObservationFooter),
}

#[doc(hidden)]
pub trait LogicalRetainedBytes {
    fn logical_retained_bytes(&self) -> u64;
}

fn string_bytes(value: &str) -> u64 {
    value.len() as u64
}

impl RuntimeTopologyRecord {
    #[doc(hidden)]
    pub const fn class(&self) -> RuntimeObservationRecordClass {
        match self {
            Self::PlanNode(_) => RuntimeObservationRecordClass::PlanNode,
            Self::PlanEdge(_) => RuntimeObservationRecordClass::PlanEdge,
            Self::TaskDefinition(_) => RuntimeObservationRecordClass::TaskDefinition,
            Self::CompiledEdge(_) => RuntimeObservationRecordClass::CompiledEdge,
            Self::BarrierDefinition(_) => RuntimeObservationRecordClass::BarrierDefinition,
            Self::SinkDefinition(_) => RuntimeObservationRecordClass::SinkDefinition,
            Self::ReservationGate(_) => RuntimeObservationRecordClass::ReservationGate,
            Self::JobDefinition(_) => RuntimeObservationRecordClass::JobDefinition,
            Self::WorkerDefinition(_) => RuntimeObservationRecordClass::WorkerDefinition,
            Self::Resource(_) => RuntimeObservationRecordClass::Resource,
        }
    }
}

impl LogicalRetainedBytes for RuntimeTopologyRecord {
    fn logical_retained_bytes(&self) -> u64 {
        const ENVELOPE: u64 = 24;
        ENVELOPE
            + match self {
                Self::PlanNode(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.node_id.as_str())
                        + string_bytes(&row.operator_kind)
                },
                Self::PlanEdge(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.from_node_id.as_str())
                        + string_bytes(row.to_node_id.as_str())
                        + 4
                },
                Self::TaskDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.task_id.as_str())
                        + row.plan_node_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(&row.task_kind)
                        + 1
                },
                Self::CompiledEdge(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.edge_id.as_str())
                        + string_bytes(row.from_task_id.as_str())
                        + string_bytes(row.to_task_id.as_str())
                        + string_bytes(&row.edge_kind)
                        + row.resource_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.gate_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::BarrierDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.barrier_id.as_str())
                        + string_bytes(&row.barrier_kind)
                        + row.upstream_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.downstream_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.gate_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::SinkDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.sink_id.as_str())
                        + string_bytes(&row.sink_kind)
                        + string_bytes(row.input_task_id.as_str())
                        + row.gate_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::ReservationGate(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.gate_id.as_str())
                        + string_bytes(&row.owner_kind)
                        + string_bytes(&row.owner_id)
                        + string_bytes(&row.reservation_kind)
                        + string_bytes(&row.requirement_scope)
                        + string_bytes(&row.memory_boundary)
                        + string_bytes(&row.required_contract)
                        + string_bytes(&row.contract_status)
                        + string_bytes(&row.contract_quality)
                },
                Self::JobDefinition(row) => {
                    row.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(row.job_id.as_str())
                        + string_bytes(&row.job_kind)
                        + 4
                },
                Self::WorkerDefinition(row) => {
                    row.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(row.worker_id.as_str())
                        + string_bytes(&row.worker_kind)
                        + 5
                },
                Self::Resource(row) => {
                    row.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(row.resource_id.as_str())
                        + string_bytes(&row.resource_kind)
                        + row.producer_task_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + row.consumer_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + string_bytes(&row.lifetime)
                        + match row.domain {
                            RuntimeResourceDomain::Host | RuntimeResourceDomain::Storage => 1,
                            RuntimeResourceDomain::Device { .. } => 5,
                        }
                },
            }
    }
}

impl LogicalRetainedBytes for RuntimeTimelineEvent {
    fn logical_retained_bytes(&self) -> u64 {
        const ENVELOPE: u64 = 42;
        let plan_execution_bytes = self.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()));
        let payload_bytes = match &self.payload {
            RuntimeObservationPayload::QueryBoundary { .. } => 2,
            RuntimeObservationPayload::PlanExecutionBoundary { .. } => 2,
            RuntimeObservationPayload::TaskLifecycle(observation) => {
                string_bytes(observation.attempt_id.task_id.as_str())
                    + 4
                    + string_bytes(&observation.task_kind)
                    + match &observation.lifecycle {
                        TaskLifecycleState::Failed { error_code } => string_bytes(error_code),
                        _ => 1,
                    }
            },
            RuntimeObservationPayload::Admission(observation) => {
                (match &observation.lifecycle {
                    AdmissionLifecycleState::Rejected { reason } => string_bytes(reason),
                    _ => 1,
                }) + 29
            },
            RuntimeObservationPayload::GpuJob(observation) => {
                string_bytes(observation.job_id.as_str())
                    + string_bytes(&observation.job_kind)
                    + observation.worker_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + match &observation.lifecycle {
                        GpuJobLifecycleState::Failed { error_code } => string_bytes(error_code),
                        _ => 1,
                    }
                    + 5
            },
            RuntimeObservationPayload::StreamLease(observation) => {
                string_bytes(observation.lease_id.as_str())
                    + observation.scope.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.task_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.source_worker_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.gpu_job_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.gpu_worker_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + match &observation.lifecycle {
                        StreamLeaseLifecycleState::Failed { error_code } => string_bytes(error_code),
                        StreamLeaseLifecycleState::Acquired { .. } => 9,
                        _ => 1,
                    }
                    + 4
            },
            RuntimeObservationPayload::SourceRead(observation) => {
                string_bytes(observation.task_id.as_str())
                    + string_bytes(observation.worker_id.as_str())
                    + match &observation.lifecycle {
                        SourceReadLifecycleState::ReadFailed { error_code, .. } => string_bytes(error_code) + 8,
                        SourceReadLifecycleState::WorkerStarted { .. }
                        | SourceReadLifecycleState::ReadFinished { .. }
                        | SourceReadLifecycleState::ConsumerJoinFinished { .. } => 8,
                        _ => 1,
                    }
                    + 16
            },
            RuntimeObservationPayload::ResourceMeasurement(observation) => {
                string_bytes(observation.resource_id.as_str()) + string_bytes(&observation.measurement_kind) + 47
            },
            RuntimeObservationPayload::Reservation(observation) => {
                let RuntimeReservationObservation {
                    reservation_id,
                    lineage_id,
                    task_attempt_id,
                    task_kind,
                    reservation_kind,
                    memory_space,
                    memory_boundary,
                    node_id,
                    stage,
                    source_location_id,
                    row_groups,
                    resident_resource_id,
                    source_read_contract_quality,
                    ..
                } = observation.as_ref();
                string_bytes(reservation_id.as_str())
                    + string_bytes(lineage_id.as_str())
                    + task_attempt_id.as_ref().map_or(0, |attempt| string_bytes(attempt.task_id.as_str()) + 4)
                    + task_kind.as_ref().map_or(0, |value| string_bytes(value))
                    + string_bytes(reservation_kind)
                    + string_bytes(memory_space)
                    + string_bytes(memory_boundary)
                    + node_id.as_ref().map_or(0, |value| string_bytes(value))
                    + stage.as_ref().map_or(0, |value| string_bytes(value))
                    + source_location_id.as_ref().map_or(0, |value| string_bytes(value.as_str()))
                    + row_groups.as_ref().map_or(0, |values| values.len() as u64 * 8)
                    + resident_resource_id.as_ref().map_or(0, |value| string_bytes(value))
                    + source_read_contract_quality.as_ref().map_or(0, |value| string_bytes(value))
                    + 66
            },
            RuntimeObservationPayload::CleanupFailure(observation) => {
                let summary = &observation.summary;
                let failure = summary.failure();
                let text_bytes = failure.text().map_or(0, |text| string_bytes(text.value()));
                64 + text_bytes + u64::try_from(failure.facts().len()).unwrap_or(u64::MAX) * 16
            },
            RuntimeObservationPayload::OperatorStage(observation) => observation.logical_retained_bytes(),
        };
        ENVELOPE + plan_execution_bytes + payload_bytes
    }
}

impl QueryAttemptReport {
    /// Projects the sealed observation footer completeness facts into this attempt report.
    #[doc(hidden)]
    pub fn set_observation_completeness(&mut self, footer: &RuntimeObservationFooter) {
        self.observation_lost_records = footer.dropped_by_class.iter().map(|loss| loss.count).sum();
        self.observation_retained_records = footer.retained_records;
        self.observation_retained_bytes = footer.retained_bytes;
        self.observation_truncated = footer.truncated;
        self.observation_degraded = footer.degraded;
    }
}
