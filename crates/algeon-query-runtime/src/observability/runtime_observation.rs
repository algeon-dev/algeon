//! Typed runtime observations, topology, and bounded retention records.

use std::fmt;

use query_planning::correlation::{QueryId, TraceId};
use serde::{Deserialize, Serialize};

use super::attempt_report::{CleanupFailureSummary, QueryAttemptReport, RuntimeTerminalOutcome};
use crate::observability::{RuntimeTaskId, TaskAttemptId};

/// JSONL schema emitted by sealed runtime observation snapshots.
pub const RUNTIME_OBSERVATION_JSONL_SCHEMA: &str = "query_engine.runtime_observation";

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
string_id!(RuntimePlanNodeId, "Plan-node identity scoped to one plan execution.");
string_id!(RuntimeReservationId, "Reservation identity scoped to one plan execution.");
string_id!(RuntimeResourceId, "Resident-resource identity scoped to one plan execution.");
string_id!(RuntimeJobId, "Executor-job identity scoped to one query.");
string_id!(RuntimeWorkerId, "Worker identity scoped to one query.");
string_id!(RuntimePipelineEdgeId, "Compiled-pipeline edge identity scoped to one plan execution.");
string_id!(RuntimeBarrierId, "Compiled pipeline-barrier identity scoped to one plan execution.");
string_id!(RuntimeSinkId, "Compiled pipeline-sink identity scoped to one plan execution.");

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
    pub(crate) fn from_keyed_digest(value: String) -> Self {
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

/// Total recorder order assigned to every attempted record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationSequence(u64);

impl ObservationSequence {
    pub(crate) const fn new(value: u64) -> Self {
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
    /// Maximum retained topology rows, timeline events and metrics detail rows.
    pub max_retained_records: u64,
    /// Maximum deterministic logical bytes charged to retained records.
    pub max_retained_bytes: u64,
}

/// Clock domain for host runtime observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RuntimeClockDomain {
    /// Query-relative time sampled from a monotonic host clock.
    HostMonotonic,
}

/// First JSONL record describing immutable query identity and capture limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationManifest {
    /// JSONL schema label.
    pub schema: String,
    /// Query-wide diagnostic identity.
    pub query_id: QueryId,
    /// Optional parent trace identity.
    pub trace_id: Option<TraceId>,
    /// Optional wall-clock anchor captured once when the session is created.
    pub unix_epoch_ns: Option<u64>,
    /// Clock domain used by timeline offsets.
    pub(crate) clock_domain: RuntimeClockDomain,
    /// Deterministic per-query retention limits.
    pub(crate) limits: RuntimeObservationLimits,
}

/// One native plan node in the static topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanNodeRow {
    /// Plan invocation containing this node.
    pub plan_execution_id: PlanExecutionId,
    /// Node identity within that plan invocation.
    pub node_id: RuntimePlanNodeId,
    /// Stable native operator category used for diagnostic grouping.
    pub operator_kind: String,
}

/// One directed edge in the static native plan topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanEdgeRow {
    /// Plan invocation containing this dependency.
    pub plan_execution_id: PlanExecutionId,
    /// Input-producing plan node.
    pub from_node_id: RuntimePlanNodeId,
    /// Plan node that consumes the input.
    pub to_node_id: RuntimePlanNodeId,
    /// Zero-based position of the input in the consuming node.
    pub input_ordinal: u32,
}

/// One compiled task definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskDefinitionRow {
    /// Plan invocation whose compiled schedule defines this task.
    pub plan_execution_id: PlanExecutionId,
    /// Compiled task identity within the plan invocation.
    pub task_id: RuntimeTaskId,
    /// Originating plan node, or `None` for a task without a direct plan node.
    pub plan_node_id: Option<RuntimePlanNodeId>,
    /// Stable compiled task category used for diagnostic grouping.
    pub task_kind: String,
}

/// One compiled dependency edge in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompiledEdgeRow {
    /// Plan invocation containing this compiled dependency.
    pub plan_execution_id: PlanExecutionId,
    /// Dependency edge identity within the plan invocation.
    pub edge_id: RuntimePipelineEdgeId,
    /// Compiled task that produces the dependency.
    pub from_task_id: RuntimeTaskId,
    /// Compiled task that consumes the dependency.
    pub to_task_id: RuntimeTaskId,
    /// Stable dependency category.
    pub edge_kind: String,
    /// Resident resources carried by this dependency.
    pub resource_ids: Vec<RuntimeResourceId>,
}

/// One compiled barrier definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BarrierDefinitionRow {
    /// Plan invocation containing this barrier.
    pub plan_execution_id: PlanExecutionId,
    /// Barrier identity within the compiled schedule.
    pub barrier_id: RuntimeBarrierId,
    /// Stable synchronization-barrier category.
    pub barrier_kind: String,
    /// Tasks whose completion precedes this barrier.
    pub upstream_task_ids: Vec<RuntimeTaskId>,
    /// Tasks whose scheduling follows this barrier.
    pub downstream_task_ids: Vec<RuntimeTaskId>,
}

/// The compiled result-sink definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SinkDefinitionRow {
    /// Plan invocation containing this result sink.
    pub plan_execution_id: PlanExecutionId,
    /// Result-sink identity within the plan invocation.
    pub sink_id: RuntimeSinkId,
    /// Stable result materialization category.
    pub sink_kind: String,
    /// Compiled task that produces the sink input.
    pub input_task_id: RuntimeTaskId,
}

/// One GPU executor job definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JobDefinitionRow {
    pub(crate) plan_execution_id: Option<PlanExecutionId>,
    pub(crate) job_id: RuntimeJobId,
    pub(crate) job_kind: String,
    pub(crate) device_ordinal: u32,
}

/// One host worker definition used by GPU or source execution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerDefinitionRow {
    /// Associated plan invocation, or `None` for query-wide work.
    pub plan_execution_id: Option<PlanExecutionId>,
    /// Worker identity within this query observation.
    pub worker_id: RuntimeWorkerId,
    /// Stable source or GPU worker category.
    pub worker_kind: String,
    /// Associated CUDA ordinal, or `None` for a worker without a device binding.
    pub device_ordinal: Option<u32>,
}

/// Resource domain carried by a static resource definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeResourceDomain {
    /// Resource stored in host memory.
    Host,
    /// Resource stored on the specified CUDA device.
    Device {
        /// CUDA device ordinal containing the resource.
        device_ordinal: u32,
    },
    /// Resource retained in source or cache storage.
    Storage,
}

/// One resident resource in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuntimeResourceRow {
    /// Associated plan invocation, or `None` for query-wide ownership.
    pub plan_execution_id: Option<PlanExecutionId>,
    /// Resident-resource identity within this observation.
    pub resource_id: RuntimeResourceId,
    /// Stable resident-resource category.
    pub resource_kind: String,
    /// Memory or storage domain containing the resource.
    pub domain: RuntimeResourceDomain,
    /// Producing task, or `None` when no compiled task produces the resource.
    pub producer_task_id: Option<RuntimeTaskId>,
    /// Compiled tasks that consume the resource.
    pub consumer_task_ids: Vec<RuntimeTaskId>,
    /// Diagnostic label describing the resource's scheduled lifetime.
    pub lifetime: String,
}

/// Static topology accepted by the bounded recorder.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuntimeTopologyRecord {
    /// Native plan node definition.
    PlanNode(PlanNodeRow),
    /// Directed input dependency between plan nodes.
    PlanEdge(PlanEdgeRow),
    /// Compiled executable task definition.
    TaskDefinition(TaskDefinitionRow),
    /// Resource dependency between compiled tasks.
    CompiledEdge(CompiledEdgeRow),
    /// Compiled task synchronization barrier.
    BarrierDefinition(BarrierDefinitionRow),
    /// Compiled result materialization sink.
    SinkDefinition(SinkDefinitionRow),
    /// GPU executor job definition.
    JobDefinition(JobDefinitionRow),
    /// Host worker definition for source or GPU execution.
    WorkerDefinition(WorkerDefinitionRow),
    /// Resident-resource ownership and lifetime definition.
    Resource(RuntimeResourceRow),
}

/// Query-level observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryBoundaryKind {
    /// Query observation began.
    Started,
    /// Terminal query cleanup began.
    CleanupStarted,
    /// Terminal query cleanup completed.
    CleanupFinished,
    /// Query execution completed successfully.
    Succeeded,
    /// An execution error ended the query.
    Failed,
    /// Caller cancellation or deadline expiry ended the query.
    Cancelled,
    /// The query owner was dropped before another terminal outcome.
    Dropped,
    /// An unwind ended the query.
    Panicked,
}

/// Plan-execution observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanExecutionBoundaryKind {
    /// One native plan invocation began.
    Started,
    /// The plan invocation completed successfully.
    Finished,
    /// The plan invocation ended with an error, drop or panic.
    Failed,
    /// Caller cancellation or deadline expiry ended the plan invocation.
    Cancelled,
}

/// Exactly-once lifecycle state for one compiled task attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TaskLifecycleState {
    /// The task attempt became runnable.
    Ready,
    /// The task attempt began execution.
    Started,
    /// The task attempt completed successfully.
    Finished,
    /// A classified error ended the task attempt.
    Failed {
        /// Stable machine code of the classified error.
        error_code: String,
    },
    /// Cooperative cancellation ended the task attempt.
    Cancelled,
    /// The task lifecycle was dropped before another terminal state.
    Dropped,
    /// An unwind ended the task attempt.
    Panicked,
}

/// Reservation operation recorded at the ownership boundary that applied it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReservationAction {
    /// Request an initial byte reservation.
    Reserve,
    /// Increase an existing reservation.
    Grow,
    /// Decrease an existing reservation.
    Shrink,
    /// Move a reservation charge to another ownership boundary.
    Transfer,
    /// Create a reference to existing reservation ownership.
    Alias,
    /// Change the reservation's diagnostic ownership label.
    Relabel,
    /// Transfer a source result into resident-cache ownership.
    SourceCachePromote,
    /// Remove resident source-cache ownership.
    SourceCacheEvict,
    /// Return charged reservation bytes.
    Release,
    /// Record an operation rejected against a reservation limit.
    Deny,
    /// Undo an uncommitted ownership operation.
    Rollback,
}

/// Terminal or nonterminal ownership disposition for a reservation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReservationDisposition {
    /// A nonterminal reservation operation changed ownership state.
    Applied,
    /// The requested reservation charge was accepted.
    Granted,
    /// The requested operation did not acquire a charge.
    Denied,
    /// An explicit release returned the charge.
    ExplicitRelease,
    /// Normal destruction returned the charge.
    DropRelease,
    /// Destruction during unwinding returned the charge.
    UnwindRelease,
    /// Terminal cleanup returned the charge.
    CleanupRelease,
    /// Rollback returned an uncommitted charge.
    RollbackRelease,
    /// An alias keeps the underlying charge retained.
    AliasDeferred,
    /// Cancellation cleanup returned the charge.
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
    /// The pool accepted the job before a worker began it.
    Enqueued,
    /// A worker began the job body.
    Started,
    /// The job body completed successfully.
    Finished,
    /// A classified error ended the job body.
    Failed {
        /// Stable machine code of the classified error.
        error_code: String,
    },
    /// Cooperative cancellation was requested for the accepted job.
    CancellationRequested,
    /// Cancellation ended the job.
    Cancelled,
    /// The job was dropped before another terminal state.
    Dropped,
    /// An unwind ended the job body.
    Panicked,
}

/// How a service stream lease was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamLeaseAcquisition {
    /// An existing pooled CUDA stream was leased.
    Reused,
    /// A new CUDA stream was created for the lease.
    Created,
}

/// Service stream-lease lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StreamLeaseLifecycleState {
    /// Stream acquisition began waiting for a lease.
    Enqueued,
    /// A CUDA stream lease was acquired after the recorded host wait.
    Acquired {
        /// Whether the lease reused a stream or created one.
        acquisition: StreamLeaseAcquisition,
        /// Elapsed host acquisition wait in nanoseconds.
        host_wait_ns: u64,
    },
    /// Cancellation was requested while acquiring a lease.
    CancellationRequested,
    /// Cancellation ended stream acquisition.
    Cancelled,
    /// A classified error ended stream acquisition.
    Failed {
        /// Stable machine code of the stream acquisition error.
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
    pub(crate) plan_execution_id: Option<PlanExecutionId>,
    /// Associated compiled task, or `None` when acquisition has no task binding.
    pub task_id: Option<RuntimeTaskId>,
    /// Source worker requesting the stream, or `None` outside source work.
    pub source_worker_id: Option<RuntimeWorkerId>,
    /// GPU executor job requesting the stream, or `None` outside a job.
    pub gpu_job_id: Option<RuntimeJobId>,
    /// GPU executor worker requesting the stream, or `None` when unavailable.
    pub gpu_worker_id: Option<RuntimeWorkerId>,
}

impl RuntimeStreamLeaseScope {
    /// Creates a lease scope bound to one plan invocation with no worker or task correlation.
    pub fn for_plan(plan_execution_id: PlanExecutionId) -> Self {
        Self { plan_execution_id: Some(plan_execution_id), ..Self::default() }
    }

    pub(crate) fn with_source_worker(mut self, task_id: RuntimeTaskId, worker_id: RuntimeWorkerId) -> Self {
        self.task_id = Some(task_id);
        self.source_worker_id = Some(worker_id);
        self
    }

    pub(crate) fn with_gpu_worker(mut self, job_id: RuntimeJobId, worker_id: RuntimeWorkerId) -> Self {
        self.gpu_job_id = Some(job_id);
        self.gpu_worker_id = Some(worker_id);
        self
    }
}

/// Source-read scheduler lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SourceReadLifecycleState {
    /// A source-read job entered the scheduler.
    Enqueued,
    /// The source worker began after the recorded queue wait.
    WorkerStarted {
        /// Nanoseconds from source job enqueue to worker start.
        queue_wait_ns: u64,
    },
    /// The source worker completed the read.
    ReadFinished {
        /// Nanoseconds spent executing the source read.
        read_duration_ns: u64,
    },
    /// A classified error ended the source read.
    ReadFailed {
        /// Stable machine code of the source-read error.
        error_code: String,
        /// Nanoseconds from read start to its error.
        read_duration_ns: u64,
    },
    /// The consumer began waiting for the source worker result.
    ConsumerJoinStarted,
    /// The consumer finished joining the source worker result.
    ConsumerJoinFinished {
        /// Nanoseconds spent waiting for the source worker result.
        join_duration_ns: u64,
    },
    /// Cooperative cancellation ended the source read.
    Cancelled,
    /// An unwind ended the source read.
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

/// Task-lifecycle payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTaskLifecycleObservation {
    /// Compiled task identity paired with its execution attempt number.
    pub attempt_id: TaskAttemptId,
    pub(crate) task_kind: String,
    /// Task boundary recorded by the task lifecycle owner.
    pub lifecycle: TaskLifecycleState,
}

/// Admission-lifecycle payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAdmissionObservation {
    pub(crate) admission_request_id: AdmissionRequestId,
    pub(crate) lifecycle: AdmissionLifecycleState,
    pub(crate) queue_depth: usize,
    pub(crate) device_ordinal: Option<u32>,
    /// Immutable admitted query capacity in bytes, or `None` before a grant.
    pub grant_bytes: Option<u64>,
}

/// GPU-job payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGpuJobObservation {
    pub(crate) job_id: RuntimeJobId,
    pub(crate) job_kind: String,
    pub(crate) device_ordinal: u32,
    pub(crate) worker_id: Option<RuntimeWorkerId>,
    /// Executor-job boundary recorded by the pool owner.
    pub lifecycle: GpuJobLifecycleState,
}

/// Stream-lease payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStreamLeaseObservation {
    pub(crate) lease_id: RuntimeResourceId,
    pub(crate) device_ordinal: u32,
    /// Explicit task, source-worker and GPU-job correlation attached at acquisition.
    pub scope: RuntimeStreamLeaseScope,
    /// Stream lease boundary recorded by the pool owner.
    pub lifecycle: StreamLeaseLifecycleState,
}

/// Source-read payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSourceReadObservation {
    /// Compiled task that owns this source read.
    pub task_id: RuntimeTaskId,
    /// Source worker that performed this read.
    pub worker_id: RuntimeWorkerId,
    /// Zero-based source chunk index within this execution.
    pub chunk_index: usize,
    /// Zero-based source file index within the source contract.
    pub file_index: usize,
    /// Read, worker or consumer-join boundary recorded by the source scheduler.
    pub lifecycle: SourceReadLifecycleState,
}

/// Resource-measurement payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeResourceMeasurementObservation {
    pub(crate) resource_id: RuntimeResourceId,
    pub(crate) scope: RuntimeMeasurementScope,
    pub(crate) source: RuntimeMeasurementSource,
    pub(crate) quality: RuntimeMeasurementQuality,
    pub(crate) measurement_kind: String,
    pub(crate) current_value: u64,
    pub(crate) delta: i128,
    pub(crate) limit: Option<u64>,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) grant_bytes: Option<u64>,
}

/// Context-rich reservation payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReservationObservation {
    /// Identity of the reservation operated on at this boundary.
    pub reservation_id: RuntimeReservationId,
    /// Identity shared by reservations derived from the same ownership lineage.
    pub lineage_id: RuntimeReservationId,
    /// Associated task attempt, or `None` outside task execution.
    pub task_attempt_id: Option<TaskAttemptId>,
    /// Associated compiled task category, or `None` when unavailable.
    pub task_kind: Option<String>,
    /// Reservation operation attempted at this boundary.
    pub action: RuntimeReservationAction,
    /// Operation that was denied, or `None` when this record is not a denial.
    pub denied_action: Option<RuntimeReservationAction>,
    /// Ownership result after the operation.
    pub disposition: RuntimeReservationDisposition,
    /// Whether the reservation aliases existing ownership without a second charge.
    pub alias: bool,
    /// Stable category of the reservation.
    pub reservation_kind: String,
    /// Diagnostic memory-domain label.
    pub memory_space: String,
    /// Diagnostic execution-boundary label for the charge.
    pub memory_boundary: String,
    /// Bytes requested by the recorded operation.
    pub requested_bytes: u64,
    /// Bytes granted by the recorded operation.
    pub granted_bytes: u64,
    /// Signed change in charged reservation bytes at this boundary.
    pub charged_delta_bytes: i128,
    /// Reservation bytes charged after this operation.
    pub current_reserved_bytes: u64,
    /// Byte limit against which the reservation was checked.
    pub budget_bytes: u64,
    /// Associated native plan node label, or `None` when unavailable.
    pub node_id: Option<String>,
    /// Associated execution-stage label, or `None` when unavailable.
    pub stage: Option<String>,
    /// Associated source chunk index, or `None` outside chunk work.
    pub chunk_index: Option<usize>,
    /// Associated source file index, or `None` outside file work.
    pub file_index: Option<usize>,
    /// Opaque query-scoped source-location digest, or `None` when unavailable.
    /// The digest excludes raw source text from the observation.
    pub source_location_id: Option<RuntimeSourceLocationId>,
    /// Selected row-group ordinals, or `None` when the operation has no selection.
    pub row_groups: Option<Vec<usize>>,
    /// Source operation attempt number, or `None` outside bounded retries.
    pub attempt: Option<usize>,
    /// Source retry bound, or `None` when unavailable.
    pub max_attempts: Option<usize>,
    /// Associated resident-resource label, or `None` without a resident result.
    pub resident_resource_id: Option<String>,
    /// Source sizing evidence label, or `None` outside source reservation.
    pub source_read_contract_quality: Option<String>,
    /// Source contract planning byte charge, or `None` when unavailable.
    pub source_read_planning_bytes: Option<u64>,
    /// Source contract estimated bytes, or `None` when unavailable.
    pub source_read_estimated_bytes: Option<u64>,
}

/// Secondary-cleanup-failure payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCleanupFailureObservation {
    /// Bounded secondary cleanup failure; the primary outcome remains separate.
    pub summary: CleanupFailureSummary,
}

/// Operator-substage category for blocking native primitives.
///
/// Pair events by [`RuntimeTimelineEvent::span_id`]. Kinds must stay short and
/// free of SQL, paths, or unbounded diagnostic text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOperatorStageKind {
    /// Regular-expression replacement size preflight.
    RegexReplacePreflight,
    /// Regular-expression replacement native execution.
    RegexReplaceExecute,
    /// Native exact-cardinality measurement used for strategy selection.
    ExactCardinalityProbe,
    /// Preparation of reusable groupby state.
    PreparedGroupbyPrepare,
    /// Aggregation using prepared groupby state.
    PreparedGroupbyAggregate,
    /// Direct libcudf groupby execution.
    LibcudfGroupby,
    /// Hash partitioning of a native frame.
    HashPartition,
    /// Aggregation of one partial input.
    PartialAggregate,
    /// Merge of partial aggregate results.
    FinalMerge,
    /// Top-k selection within one native partition.
    PartitionLocalTopK,
    /// Final native sort of retained inputs.
    FinalSort,
    /// Preparation of dense vector inputs for a native algorithm.
    VectorPack,
    /// Native vector algorithm execution.
    VectorNative,
    /// Assembly of vector algorithm output into query results.
    VectorAssemble,
    /// Preparation of matrix inputs for a native ML algorithm.
    MlPack,
    /// Native machine-learning algorithm execution.
    MlNative,
    /// Assembly of ML algorithm output into query results.
    MlAssemble,
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
            Self::MlPack => "ml_pack",
            Self::MlNative => "ml_native",
            Self::MlAssemble => "ml_assemble",
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
    /// The native operator stage began.
    Started,
    /// The native operator stage completed successfully.
    Finished,
    /// A classified error ended the operator stage.
    Failed {
        /// Stable machine code of the classified error.
        error_code: String,
    },
    /// Cooperative cancellation ended the operator stage.
    Cancelled,
}

/// Operator-substage payload body retained out of line.
///
/// The payload carries numeric facts and bounded string labels, such as
/// `node_id` and `strategy_label`. It excludes SQL, raw paths, and unbounded
/// strings.
/// Constructors preserve caller labels without bounding or validating them;
/// callers supply labels that satisfy the observation contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOperatorStageObservation {
    /// Native primitive or preparation stage measured by this span.
    pub stage_kind: RuntimeOperatorStageKind,
    /// Start, completion or failure boundary within the stage span.
    pub lifecycle: RuntimeOperatorStageState,
    pub(crate) node_id: Option<String>,
    pub(crate) task_id: Option<RuntimeTaskId>,
    /// Associated source chunk index, or `None` outside chunk work.
    pub chunk_index: Option<usize>,
    pub(crate) partition_index: Option<usize>,
    pub(crate) attempt: Option<u32>,
    /// Bounded execution strategy label, or `None` when unavailable.
    pub strategy_label: Option<String>,
    /// Observed input row count, or `None` without a row observation.
    pub input_rows: Option<u64>,
    /// Observed output row count, or `None` without a row observation.
    pub output_rows: Option<u64>,
    pub(crate) input_bytes: Option<u64>,
    pub(crate) output_bytes: Option<u64>,
    pub(crate) reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) vector_algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Machine-learning algorithm label, or `None` outside ML stages.
    pub ml_algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Vector input role label, or `None` outside vector stages.
    pub vector_relation_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Number of coordinates in a dense vector, or `None` when unavailable.
    pub vector_dimension: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Native matrix layout label, or `None` when unavailable.
    pub matrix_layout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) execution_binding_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) device_ordinal: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Native component major version, or `None` without version evidence.
    pub native_version_major: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Native component minor version, or `None` without version evidence.
    pub native_version_minor: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Native component patch version, or `None` without version evidence.
    pub native_version_patch: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bounded_control_scalar_events: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bounded_control_scalar_bytes: Option<u64>,
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
            ml_algorithm: None,
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

    /// Attaches the bounded native plan-node label for this stage.
    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Attaches the compiled task associated with this stage.
    pub fn with_task_id(mut self, task_id: RuntimeTaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Attaches the source chunk index associated with this stage.
    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    /// Attaches the native partition index associated with this stage.
    pub fn with_partition_index(mut self, partition_index: usize) -> Self {
        self.partition_index = Some(partition_index);
        self
    }

    /// Attaches the operation attempt number associated with this stage.
    pub fn with_attempt(mut self, attempt: u32) -> Self {
        self.attempt = Some(attempt);
        self
    }

    /// Attaches a bounded execution strategy label without validating its content.
    pub fn with_strategy_label(mut self, strategy_label: impl Into<String>) -> Self {
        self.strategy_label = Some(strategy_label.into());
        self
    }

    /// Attaches the observed input row count.
    pub fn with_input_rows(mut self, input_rows: u64) -> Self {
        self.input_rows = Some(input_rows);
        self
    }

    pub(crate) fn with_output_rows(mut self, output_rows: u64) -> Self {
        self.output_rows = Some(output_rows);
        self
    }

    /// Attaches the observed input size in bytes.
    pub fn with_input_bytes(mut self, input_bytes: u64) -> Self {
        self.input_bytes = Some(input_bytes);
        self
    }

    pub(crate) fn with_output_bytes(mut self, output_bytes: u64) -> Self {
        self.output_bytes = Some(output_bytes);
        self
    }

    /// Attaches the reservation charge in bytes associated with this stage.
    pub fn with_reserved_bytes(mut self, reserved_bytes: u64) -> Self {
        self.reserved_bytes = Some(reserved_bytes);
        self
    }

    /// Attaches the native vector algorithm label.
    pub fn with_vector_algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.vector_algorithm = Some(algorithm.into());
        self
    }

    /// Attaches the native machine-learning algorithm label.
    pub fn with_ml_algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.ml_algorithm = Some(algorithm.into());
        self
    }

    /// Attaches the vector input role used by the native call.
    pub fn with_vector_relation_role(mut self, role: impl Into<String>) -> Self {
        self.vector_relation_role = Some(role.into());
        self
    }

    /// Attaches the number of coordinates in each dense vector.
    pub fn with_vector_dimension(mut self, dimension: usize) -> Self {
        self.vector_dimension = Some(dimension);
        self
    }

    /// Attaches the native matrix layout label.
    pub fn with_matrix_layout(mut self, layout: impl Into<String>) -> Self {
        self.matrix_layout = Some(layout.into());
        self
    }

    /// Attaches the native execution binding identity for correlation.
    pub fn with_execution_binding_id(mut self, binding_id: u64) -> Self {
        self.execution_binding_id = Some(binding_id);
        self
    }

    /// Attaches the CUDA ordinal on which the stage executes.
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Attaches the native component version as major, minor and patch values.
    pub fn with_native_version(mut self, major: u16, minor: u16, patch: u16) -> Self {
        self.native_version_major = Some(major);
        self.native_version_minor = Some(minor);
        self.native_version_patch = Some(patch);
        self
    }

    pub(crate) fn with_bounded_control_scalars(mut self, events: u64, bytes: u64) -> Self {
        self.bounded_control_scalar_events = Some(events);
        self.bounded_control_scalar_bytes = Some(bytes);
        self
    }

    pub(crate) fn logical_retained_bytes(&self) -> u64 {
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

/// Typed dynamic event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeObservationPayload {
    /// Query lifecycle boundary stored with the event timestamp.
    QueryBoundary {
        /// Query lifecycle transition represented by this event.
        boundary: QueryBoundaryKind,
    },
    /// Native plan invocation boundary stored with the event timestamp.
    PlanExecutionBoundary {
        /// Native plan invocation transition represented by this event.
        boundary: PlanExecutionBoundaryKind,
    },
    /// Compiled task attempt lifecycle evidence.
    TaskLifecycle(Box<RuntimeTaskLifecycleObservation>),
    /// Admission lifecycle and immutable grant evidence.
    Admission(Box<RuntimeAdmissionObservation>),
    /// GPU executor job lifecycle evidence.
    GpuJob(Box<RuntimeGpuJobObservation>),
    /// Stream acquisition and release lifecycle evidence.
    StreamLease(Box<RuntimeStreamLeaseObservation>),
    /// Source worker and consumer-join lifecycle evidence.
    SourceRead(Box<RuntimeSourceReadObservation>),
    /// Resource sample with its owning scope and provenance.
    ResourceMeasurement(Box<RuntimeResourceMeasurementObservation>),
    /// Reservation ownership transition and byte charge evidence.
    Reservation(Box<RuntimeReservationObservation>),
    /// Secondary cleanup failure retained without replacing a primary query error.
    CleanupFailure(Box<RuntimeCleanupFailureObservation>),
    /// Operator-substage lifecycle for blocking native primitives.
    OperatorStage(Box<RuntimeOperatorStageObservation>),
}

/// One dynamic event in recorder total order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimelineEvent {
    /// Recorder sequence that orders this event with topology record attempts.
    pub sequence: ObservationSequence,
    /// Nanoseconds elapsed from the query observation's monotonic clock origin.
    pub monotonic_ns: u64,
    /// Associated native plan invocation, or `None` for query-wide events.
    pub plan_execution_id: Option<PlanExecutionId>,
    /// Query-scoped span used to pair lifecycle events, or `None` without a span.
    pub span_id: Option<RuntimeSpanId>,
    pub(crate) parent_span_id: Option<RuntimeSpanId>,
    /// Typed lifecycle or resource evidence retained by this event.
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
    JobDefinition,
    WorkerDefinition,
    Resource,
    TimelineEvent,
    /// A source-read detail row offered for bounded retention.
    SourceReadDetail,
    /// An operator occurrence row offered for bounded retention.
    OperatorDetail,
    /// A join strategy decision row offered for bounded retention.
    JoinStrategyDetail,
    /// An aggregate strategy decision row offered for bounded retention.
    AggregateStrategyDetail,
    /// A dynamic-filter runtime row offered for bounded retention.
    DynamicFilterDetail,
    /// A lower-level budget report row offered for bounded retention.
    BudgetDetail,
    /// A background lakehouse cache-fill completion offered for bounded
    /// retention.
    LakehouseCacheFillDetail,
}

/// Reason an attempted record was not retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObservationLossReason {
    /// The configured retained-record limit was exhausted.
    RecordLimit,
    /// The configured retained logical-byte budget was exhausted.
    ByteLimit,
    /// A single record exceeded the entire configured logical-byte budget.
    Oversized,
    /// A keyed source digest was unavailable; raw source text was not retained.
    SourceLocationKeyUnavailable,
    /// No plan execution was left unfrozen to carry the record when the
    /// session sealed.
    Unassigned,
}

/// Count of dropped records in one class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuntimeObservationClassLoss {
    pub(crate) class: RuntimeObservationRecordClass,
    pub(crate) count: u64,
}

/// Count of dropped records for one reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationReasonLoss {
    /// Condition that prevented retention.
    pub reason: RuntimeObservationLossReason,
    /// Records dropped for this condition; consult footer overflow metadata for exactness.
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
    /// Whether the observation session sealed. Retention losses remain described separately.
    pub complete: bool,
    /// Committed query outcome, or `None` when no outcome was supplied at seal.
    pub terminal: Option<RuntimeTerminalOutcome>,
    /// Retained topology rows, timeline events and bounded detail records.
    pub retained_records: u64,
    /// Deterministic logical bytes charged to retained records.
    pub retained_bytes: u64,
    /// First record attempt declined for retention, or `None` when none was dropped.
    pub first_dropped_sequence: Option<ObservationSequence>,
    /// Last attempted record sequence, or `None` when no record was attempted.
    pub last_attempted_sequence: Option<ObservationSequence>,
    pub(crate) dropped_by_class: Vec<RuntimeObservationClassLoss>,
    /// Loss counts grouped by their closed retention reason.
    pub dropped_by_reason: Vec<RuntimeObservationReasonLoss>,
    /// Whether at least one attempted record could not be retained.
    pub truncated: bool,
    /// Whether recording encountered degradation, including recovered recorder poisoning.
    pub degraded: bool,
    pub(crate) poison_recoveries: u64,
    /// The exact sealed attempt report, when this observation belongs to an
    /// execution attempt. A footer without an attempt report omits this field
    /// when serialized.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_attempt_report",
        deserialize_with = "deserialize_optional_attempt_report"
    )]
    pub(crate) attempt_report: Option<std::sync::Arc<QueryAttemptReport>>,
    /// At least one recorder sequence or accounting counter saturated.
    ///
    /// Saturated values remain monotonic, while this marker prevents exported
    /// observations from presenting them as exact.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub counter_overflow: bool,
}

/// One independently parseable JSONL record.
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
    JobDefinition { sequence: ObservationSequence, row: JobDefinitionRow },
    WorkerDefinition { sequence: ObservationSequence, row: WorkerDefinitionRow },
    Resource { sequence: ObservationSequence, row: RuntimeResourceRow },
    TimelineEvent(RuntimeTimelineEvent),
    Footer(RuntimeObservationFooter),
}

fn string_bytes(value: &str) -> u64 {
    value.len() as u64
}

impl RuntimeTopologyRecord {
    pub(crate) const fn class(&self) -> RuntimeObservationRecordClass {
        match self {
            Self::PlanNode(_) => RuntimeObservationRecordClass::PlanNode,
            Self::PlanEdge(_) => RuntimeObservationRecordClass::PlanEdge,
            Self::TaskDefinition(_) => RuntimeObservationRecordClass::TaskDefinition,
            Self::CompiledEdge(_) => RuntimeObservationRecordClass::CompiledEdge,
            Self::BarrierDefinition(_) => RuntimeObservationRecordClass::BarrierDefinition,
            Self::SinkDefinition(_) => RuntimeObservationRecordClass::SinkDefinition,
            Self::JobDefinition(_) => RuntimeObservationRecordClass::JobDefinition,
            Self::WorkerDefinition(_) => RuntimeObservationRecordClass::WorkerDefinition,
            Self::Resource(_) => RuntimeObservationRecordClass::Resource,
        }
    }

    pub(super) fn logical_retained_bytes(&self) -> u64 {
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
                },
                Self::BarrierDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.barrier_id.as_str())
                        + string_bytes(&row.barrier_kind)
                        + row.upstream_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.downstream_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::SinkDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.sink_id.as_str())
                        + string_bytes(&row.sink_kind)
                        + string_bytes(row.input_task_id.as_str())
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

impl RuntimeTimelineEvent {
    pub(super) fn logical_retained_bytes(&self) -> u64 {
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
    pub(crate) fn set_observation_completeness(&mut self, footer: &RuntimeObservationFooter) {
        self.observation_lost_records = footer.dropped_by_class.iter().map(|loss| loss.count).sum();
        self.observation_retained_records = footer.retained_records;
        self.observation_retained_bytes = footer.retained_bytes;
        self.observation_truncated = footer.truncated;
        self.observation_degraded = footer.degraded;
    }
}
