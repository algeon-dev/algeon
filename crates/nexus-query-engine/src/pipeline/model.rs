//! Mode-agnostic compiled pipeline graph model.
//!
//! Ids in this module are local to one [`CompiledPipeline`] and are expected to
//! match their position in the corresponding vector. Builders do not validate
//! cross references; call [`CompiledPipeline::validate`] before execution.

use crate::{
    capability::{ContractQuality, ContractStatus, ReservationRequirement, ReservationRequirementScope},
    plan::NodeId,
    runtime::ReservationKind,
};

/// Mode-agnostic compiled graph for a native query DAG.
///
/// This structure records task order, resource lifetimes, barriers, sink
/// boundaries, and reservation gates. Strict and adaptive executors may consume
/// it, but mode-specific admission and runtime-action policy live outside it.
/// Task, edge, barrier, resource, and gate ids are local index spaces owned by
/// one compiled graph and are validated for topological consistency before use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledPipeline {
    tasks: Vec<PipelineTask>,
    edges: Vec<PipelineEdge>,
    barriers: Vec<PipelineBarrier>,
    resources: Vec<ResidentResource>,
    gates: Vec<ReservationGate>,
    sink: SinkBoundary,
}

impl CompiledPipeline {
    /// Create a compiled graph from already assigned local id spaces.
    pub fn new(
        tasks: Vec<PipelineTask>,
        edges: Vec<PipelineEdge>,
        barriers: Vec<PipelineBarrier>,
        resources: Vec<ResidentResource>,
        gates: Vec<ReservationGate>,
        sink: SinkBoundary,
    ) -> Self {
        Self { tasks, edges, barriers, resources, gates, sink }
    }

    /// Symbolic tasks in local id order.
    pub fn tasks(&self) -> &[PipelineTask] {
        &self.tasks
    }

    /// Dependency edges in local id order.
    pub fn edges(&self) -> &[PipelineEdge] {
        &self.edges
    }

    /// Blocking/materialization barriers in local id order.
    pub fn barriers(&self) -> &[PipelineBarrier] {
        &self.barriers
    }

    /// Resident resources in local id order.
    pub fn resources(&self) -> &[ResidentResource] {
        &self.resources
    }

    /// Reservation gates in local id order.
    pub fn gates(&self) -> &[ReservationGate] {
        &self.gates
    }

    /// Final sink boundary for the compiled graph.
    pub fn sink(&self) -> &SinkBoundary {
        &self.sink
    }

    /// Render a human-readable graph summary for diagnostics.
    pub fn debug_render(&self) -> String {
        self.to_string()
    }

    /// Validate id positions, cross references, resource flow, and gate contracts.
    pub fn validate(&self) -> crate::error::Result<()> {
        super::validate::validate_pipeline(self)
    }
}

/// Local identifier for a symbolic pipeline task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineTaskId(usize);

#[allow(missing_docs)]
impl PipelineTaskId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Local identifier for a symbolic pipeline edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineEdgeId(usize);

#[allow(missing_docs)]
impl PipelineEdgeId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Local identifier for a symbolic pipeline barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineBarrierId(usize);

#[allow(missing_docs)]
impl PipelineBarrierId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Local identifier for a resident device resource in the compiled graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentResourceId(usize);

#[allow(missing_docs)]
impl ResidentResourceId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Local identifier for a reservation gate in the compiled graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReservationGateId(usize);

#[allow(missing_docs)]
impl ReservationGateId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// One symbolic unit of native work.
///
/// A task names the operator work to perform and the resident resources it
/// consumes or produces. The scheduler uses task ids and dependencies; the
/// operator interpreter owns the actual cuDF work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineTask {
    id: PipelineTaskId,
    kind: PipelineTaskKind,
    plan_node: Option<NodeId>,
    input_resources: Vec<ResidentResourceId>,
    output_resources: Vec<ResidentResourceId>,
    gates: Vec<ReservationGateId>,
}

impl PipelineTask {
    /// Create a task that is not tied to one native plan node.
    pub fn new(id: PipelineTaskId, kind: PipelineTaskKind) -> Self {
        Self { id, kind, plan_node: None, input_resources: Vec::new(), output_resources: Vec::new(), gates: Vec::new() }
    }

    /// Create a task associated with a native plan node.
    pub fn for_plan_node(id: PipelineTaskId, kind: PipelineTaskKind, plan_node: NodeId) -> Self {
        Self { plan_node: Some(plan_node), ..Self::new(id, kind) }
    }

    /// Attach resident resources consumed by this task.
    pub fn with_input_resources(mut self, input_resources: Vec<ResidentResourceId>) -> Self {
        self.input_resources = input_resources;
        self
    }

    /// Attach resident resources produced by this task.
    pub fn with_output_resources(mut self, output_resources: Vec<ResidentResourceId>) -> Self {
        self.output_resources = output_resources;
        self
    }

    /// Attach reservation gates owned by this task.
    pub fn with_gates(mut self, gates: Vec<ReservationGateId>) -> Self {
        self.gates = gates;
        self
    }

    /// Local task id.
    pub fn id(&self) -> PipelineTaskId {
        self.id
    }

    /// Symbolic work kind.
    pub fn kind(&self) -> PipelineTaskKind {
        self.kind
    }

    /// Native plan node that produced this task, when applicable.
    pub fn plan_node(&self) -> Option<NodeId> {
        self.plan_node
    }

    /// Resident resources consumed by this task.
    pub fn input_resources(&self) -> &[ResidentResourceId] {
        &self.input_resources
    }

    /// Resident resources produced by this task.
    pub fn output_resources(&self) -> &[ResidentResourceId] {
        &self.output_resources
    }

    /// Reservation gates owned by this task.
    pub fn gates(&self) -> &[ReservationGateId] {
        &self.gates
    }
}

/// Symbolic native work categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineTaskKind {
    /// Read a bounded source chunk into a resident frame.
    SourceRead,
    /// Import an already GPU-resident input frame.
    GpuResidentSource,
    /// Import host Arrow literal batches into a resident frame.
    LiteralBatchSource,
    /// Apply pipelineable unary work such as filter or projection.
    Unary,
    /// Execute a blocking aggregate.
    Aggregate,
    /// Execute a blocking window operator.
    Window,
    /// Prepare the resident build side of a join.
    JoinBuild,
    /// Execute the probe stage against a prepared join build side.
    JoinProbe,
    /// Materialize join indices or prepared join state into an output frame.
    JoinMaterialize,
    /// Execute the final ordering boundary.
    FinalSort,
    /// Apply a bounded limit.
    Limit,
    /// Concatenate ordered union inputs.
    Union,
    /// Materialize the explicit result sink.
    SinkMaterialize,
}

impl PipelineTaskKind {
    /// Stable diagnostic label for this task kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceRead => "source_read",
            Self::GpuResidentSource => "gpu_resident_source",
            Self::LiteralBatchSource => "literal_batch_source",
            Self::Unary => "unary",
            Self::Aggregate => "aggregate",
            Self::Window => "window",
            Self::JoinBuild => "join_build",
            Self::JoinProbe => "join_probe",
            Self::JoinMaterialize => "join_materialize",
            Self::FinalSort => "final_sort",
            Self::Limit => "limit",
            Self::Union => "union",
            Self::SinkMaterialize => "sink_materialize",
        }
    }
}

/// A dependency edge between symbolic tasks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineEdge {
    id: PipelineEdgeId,
    from: PipelineTaskId,
    to: PipelineTaskId,
    kind: PipelineEdgeKind,
    resources: Vec<ResidentResourceId>,
    gates: Vec<ReservationGateId>,
}

impl PipelineEdge {
    /// Create an edge between two tasks.
    pub fn new(id: PipelineEdgeId, from: PipelineTaskId, to: PipelineTaskId, kind: PipelineEdgeKind) -> Self {
        Self { id, from, to, kind, resources: Vec::new(), gates: Vec::new() }
    }

    /// Attach resident resources that flow across this edge.
    pub fn with_resources(mut self, resources: Vec<ResidentResourceId>) -> Self {
        self.resources = resources;
        self
    }

    /// Attach reservation gates owned by this edge.
    pub fn with_gates(mut self, gates: Vec<ReservationGateId>) -> Self {
        self.gates = gates;
        self
    }

    /// Local edge id.
    pub fn id(&self) -> PipelineEdgeId {
        self.id
    }

    /// Producer task id.
    pub fn from(&self) -> PipelineTaskId {
        self.from
    }

    /// Consumer task id.
    pub fn to(&self) -> PipelineTaskId {
        self.to
    }

    /// Symbolic dependency kind.
    pub fn kind(&self) -> PipelineEdgeKind {
        self.kind
    }

    /// Resident resources transferred or referenced by this edge.
    pub fn resources(&self) -> &[ResidentResourceId] {
        &self.resources
    }

    /// Reservation gates owned by this edge.
    pub fn gates(&self) -> &[ReservationGateId] {
        &self.gates
    }
}

/// Symbolic dependency classes in the compiled graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineEdgeKind {
    /// Pure ordering dependency with no resident payload transfer.
    Control,
    /// A resident frame flows from producer to consumer.
    ResidentFrame,
    /// A resident frame is consumed as a join build input.
    JoinBuildInput,
    /// A resident frame is consumed as a join probe input.
    JoinProbeInput,
    /// One resident producer is intentionally reused by multiple consumers.
    SharedResidentProducer,
    /// Output of a blocking boundary flows downstream.
    BarrierOutput,
    /// The final resident resource feeds the sink.
    SinkInput,
}

impl PipelineEdgeKind {
    /// Stable diagnostic label for this edge kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::ResidentFrame => "resident_frame",
            Self::JoinBuildInput => "join_build_input",
            Self::JoinProbeInput => "join_probe_input",
            Self::SharedResidentProducer => "shared_resident_producer",
            Self::BarrierOutput => "barrier_output",
            Self::SinkInput => "sink_input",
        }
    }
}

/// A blocking or materialization boundary that downstream tasks must respect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineBarrier {
    id: PipelineBarrierId,
    kind: PipelineBarrierKind,
    upstream_tasks: Vec<PipelineTaskId>,
    downstream_tasks: Vec<PipelineTaskId>,
    gates: Vec<ReservationGateId>,
}

impl PipelineBarrier {
    /// Create a barrier with no attached tasks or gates.
    pub fn new(id: PipelineBarrierId, kind: PipelineBarrierKind) -> Self {
        Self { id, kind, upstream_tasks: Vec::new(), downstream_tasks: Vec::new(), gates: Vec::new() }
    }

    /// Attach tasks that must complete before this barrier.
    pub fn with_upstream_tasks(mut self, upstream_tasks: Vec<PipelineTaskId>) -> Self {
        self.upstream_tasks = upstream_tasks;
        self
    }

    /// Attach tasks dominated by this barrier.
    pub fn with_downstream_tasks(mut self, downstream_tasks: Vec<PipelineTaskId>) -> Self {
        self.downstream_tasks = downstream_tasks;
        self
    }

    /// Attach reservation gates owned by this barrier.
    pub fn with_gates(mut self, gates: Vec<ReservationGateId>) -> Self {
        self.gates = gates;
        self
    }

    /// Local barrier id.
    pub fn id(&self) -> PipelineBarrierId {
        self.id
    }

    /// Symbolic barrier kind.
    pub fn kind(&self) -> PipelineBarrierKind {
        self.kind
    }

    /// Tasks that must complete before the barrier is satisfied.
    pub fn upstream_tasks(&self) -> &[PipelineTaskId] {
        &self.upstream_tasks
    }

    /// Tasks that run after the barrier is satisfied.
    pub fn downstream_tasks(&self) -> &[PipelineTaskId] {
        &self.downstream_tasks
    }

    /// Reservation gates owned by this barrier.
    pub fn gates(&self) -> &[ReservationGateId] {
        &self.gates
    }
}

/// Blocking and ordering boundaries in the pipeline graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineBarrierKind {
    /// Boundary between planned source chunks.
    SourceChunkBoundary,
    /// Aggregate must complete before downstream consumers run.
    AggregateBoundary,
    /// Join build state must exist before probing begins.
    JoinBuildBeforeProbe,
    /// Join probe state must be materialized before downstream consumers run.
    JoinMaterialization,
    /// Sort/order output is a blocking boundary.
    SortBoundary,
    /// Final output materialization boundary.
    SinkBoundary,
}

impl PipelineBarrierKind {
    /// Stable diagnostic label for this barrier kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceChunkBoundary => "source_chunk_boundary",
            Self::AggregateBoundary => "aggregate_boundary",
            Self::JoinBuildBeforeProbe => "join_build_before_probe",
            Self::JoinMaterialization => "join_materialization",
            Self::SortBoundary => "sort_boundary",
            Self::SinkBoundary => "sink_boundary",
        }
    }
}

/// A resident device resource and its symbolic lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentResource {
    id: ResidentResourceId,
    producer: PipelineTaskId,
    consumers: Vec<PipelineTaskId>,
    lifetime: ResidentLifetime,
}

impl ResidentResource {
    /// Create a resident resource with its producer, consumers, and lifetime.
    pub fn new(
        id: ResidentResourceId,
        producer: PipelineTaskId,
        consumers: Vec<PipelineTaskId>,
        lifetime: ResidentLifetime,
    ) -> Self {
        Self { id, producer, consumers, lifetime }
    }

    /// Local resident-resource id.
    pub fn id(&self) -> ResidentResourceId {
        self.id
    }

    /// Task that produces the resource.
    pub fn producer(&self) -> PipelineTaskId {
        self.producer
    }

    /// Tasks that consume the resource.
    pub fn consumers(&self) -> &[PipelineTaskId] {
        &self.consumers
    }

    /// Symbolic lifetime for release planning.
    pub fn lifetime(&self) -> ResidentLifetime {
        self.lifetime
    }
}

/// Symbolic resident lifetime class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentLifetime {
    /// Produced and released during the same task.
    TaskLocal {
        /// Task that owns the resource for its local execution window.
        task: PipelineTaskId,
    },
    /// Produced by one task and retained until the last recorded consumer.
    BetweenTasks {
        /// Task that produced the resident resource.
        producer: PipelineTaskId,
    },
    /// Retained as the final sink output.
    SinkOutput {
        /// Task that produced the final sink resource.
        producer: PipelineTaskId,
    },
}

/// Memory or execution-boundary gate attached to its most local pipeline owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationGate {
    id: ReservationGateId,
    owner: ReservationGateOwner,
    requirement: ReservationRequirement,
}

impl ReservationGate {
    /// Create a gate from individual reservation requirement fields.
    pub fn new(
        id: ReservationGateId,
        owner: ReservationGateOwner,
        kind: ReservationKind,
        scope: ReservationRequirementScope,
        memory_boundary: &'static str,
        required_contract: &'static str,
        status: ContractStatus,
        quality: ContractQuality,
    ) -> Self {
        let requirement = ReservationRequirement { kind, scope, memory_boundary, required_contract, status, quality };
        Self { id, owner, requirement }
    }

    /// Create a gate from an existing reservation requirement.
    pub fn from_requirement(
        id: ReservationGateId,
        owner: ReservationGateOwner,
        requirement: ReservationRequirement,
    ) -> Self {
        Self { id, owner, requirement }
    }

    /// Local reservation-gate id.
    pub fn id(&self) -> ReservationGateId {
        self.id
    }

    /// Graph object that owns the gate.
    pub fn owner(&self) -> ReservationGateOwner {
        self.owner
    }

    /// Reservation requirement enforced or reported by this gate.
    pub fn requirement(&self) -> &ReservationRequirement {
        &self.requirement
    }

    /// Reservation kind associated with the gate.
    pub fn reservation_kind(&self) -> ReservationKind {
        self.requirement.kind
    }

    /// Stable memory-boundary diagnostic string.
    pub fn memory_boundary(&self) -> &'static str {
        self.requirement.memory_boundary
    }

    /// Required contract diagnostic string.
    pub fn required_contract(&self) -> &'static str {
        self.requirement.required_contract
    }

    /// Modeling status for the gate requirement.
    pub fn status(&self) -> ContractStatus {
        self.requirement.status
    }

    /// Evidence quality for the gate requirement.
    pub fn quality(&self) -> ContractQuality {
        self.requirement.quality
    }
}

/// The graph object that owns a reservation gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationGateOwner {
    /// Gate is owned by a task-local operation.
    Task(PipelineTaskId),
    /// Gate is owned by a producer-consumer edge.
    Edge(PipelineEdgeId),
    /// Gate is owned by a blocking or ordering barrier.
    Barrier(PipelineBarrierId),
    /// Gate is owned by the final sink boundary.
    SinkBoundary,
}

/// Explicit result boundary for a compiled pipeline graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkBoundary {
    kind: SinkBoundaryKind,
    input: PipelineTaskId,
    gates: Vec<ReservationGateId>,
}

impl SinkBoundary {
    /// Create a sink boundary fed by `input`.
    pub fn new(kind: SinkBoundaryKind, input: PipelineTaskId) -> Self {
        Self { kind, input, gates: Vec::new() }
    }

    /// Attach reservation gates owned by the sink boundary.
    pub fn with_gates(mut self, gates: Vec<ReservationGateId>) -> Self {
        self.gates = gates;
        self
    }

    /// Sink boundary kind.
    pub fn kind(&self) -> SinkBoundaryKind {
        self.kind
    }

    /// Task that produces the sink input resource.
    pub fn input(&self) -> PipelineTaskId {
        self.input
    }

    /// Reservation gates owned by the sink boundary.
    pub fn gates(&self) -> &[ReservationGateId] {
        &self.gates
    }
}

/// The final result boundary class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SinkBoundaryKind {
    /// Materialize the result as host Arrow.
    HostArrowMaterialization,
    /// Keep the result as a GPU-resident frame.
    GpuResidentOutput,
}

impl SinkBoundaryKind {
    /// Stable diagnostic label for this sink boundary kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostArrowMaterialization => "host_arrow_materialization",
            Self::GpuResidentOutput => "gpu_resident_output",
        }
    }
}
