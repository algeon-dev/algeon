//! Mode-agnostic compiled pipeline graph model.
//!
//! Ids in this module are local to one [`CompiledPipeline`] and are expected to
//! match their position in the corresponding vector. Builders do not validate
//! cross-references; call [`CompiledPipeline::validate`] before execution.

pub use crate::capability::AggregateExecutionStage;
use crate::{capability::CompiledJoinExecutionStrategy, error::Result, expr::SortExpr, plan::NodeId};

/// Mode-agnostic compiled graph for a native query DAG.
///
/// This structure records task order, resource lifetimes, barriers, sink
/// boundaries. The capability-first bounded executor consumes it. Task, edge,
/// barrier, and resource ids are local index spaces owned by one compiled graph
/// and are validated for topological consistency before use.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPipeline {
    tasks: Vec<PipelineTask>,
    edges: Vec<PipelineEdge>,
    barriers: Vec<PipelineBarrier>,
    resources: Vec<ResidentResource>,
    sink: SinkBoundary,
}

impl CompiledPipeline {
    /// Create a compiled graph from already assigned local id spaces.
    pub fn new(
        tasks: Vec<PipelineTask>,
        edges: Vec<PipelineEdge>,
        barriers: Vec<PipelineBarrier>,
        resources: Vec<ResidentResource>,
        sink: SinkBoundary,
    ) -> Self {
        Self { tasks, edges, barriers, resources, sink }
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

    /// Final sink boundary for the compiled graph.
    pub fn sink(&self) -> &SinkBoundary {
        &self.sink
    }

    /// Render a human-readable graph summary for diagnostics.
    pub fn debug_render(&self) -> String {
        self.to_string()
    }

    /// Validate id positions, cross references, and resource flow.
    pub fn validate(&self) -> Result<()> {
        super::validate::validate_pipeline(self)
    }
}

/// Local identifier for a symbolic pipeline task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineTaskId(usize);

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

impl ResidentResourceId {
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
    accepted_input_form: AcceptedInputForm,
    sort_strategy: Option<SortStrategy>,
    plan_join_strategy: Option<CompiledJoinExecutionStrategy>,
    aggregate_stage: Option<AggregateExecutionStage>,
}

impl PipelineTask {
    /// Create a task that is not tied to one native plan node.
    ///
    /// The execution contract defaults to the narrowest form a task can
    /// require: one complete resident frame, no fused sort, no join strategy.
    /// [`compile_pipeline`](crate::pipeline::compile_pipeline) replaces it with
    /// the contract it resolved for the task.
    pub fn new(id: PipelineTaskId, kind: PipelineTaskKind) -> Self {
        Self {
            id,
            kind,
            plan_node: None,
            input_resources: Vec::new(),
            output_resources: Vec::new(),
            accepted_input_form: AcceptedInputForm::Frame,
            sort_strategy: None,
            plan_join_strategy: None,
            aggregate_stage: None,
        }
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

    /// Record the strongest payload form this task accepts on a data input.
    pub fn with_accepted_input_form(mut self, accepted_input_form: AcceptedInputForm) -> Self {
        self.accepted_input_form = accepted_input_form;
        self
    }

    /// Record the final-sort execution strategy selected for this task.
    pub fn with_sort_strategy(mut self, sort_strategy: SortStrategy) -> Self {
        self.sort_strategy = Some(sort_strategy);
        self
    }

    /// Record the plan-derived join strategy selected for this join task.
    pub fn with_plan_join_strategy(mut self, plan_join_strategy: CompiledJoinExecutionStrategy) -> Self {
        self.plan_join_strategy = Some(plan_join_strategy);
        self
    }

    /// Record the aggregate execution stage this task runs.
    pub fn with_aggregate_stage(mut self, aggregate_stage: AggregateExecutionStage) -> Self {
        self.aggregate_stage = Some(aggregate_stage);
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

    /// Strongest payload form this task accepts on a data input.
    pub fn accepted_input_form(&self) -> AcceptedInputForm {
        self.accepted_input_form
    }

    /// Final-sort execution strategy, set on [`PipelineTaskKind::FinalSort`].
    pub fn sort_strategy(&self) -> Option<SortStrategy> {
        self.sort_strategy
    }

    /// Plan-derived join strategy, set on join build and probe tasks.
    ///
    /// This is the strategy the plan alone determines. The executor may still
    /// refine it from observed build facts before running the kernel.
    pub fn plan_join_strategy(&self) -> Option<CompiledJoinExecutionStrategy> {
        self.plan_join_strategy
    }

    /// Aggregate execution stage, set on [`PipelineTaskKind::Aggregate`].
    ///
    /// Compiled aggregate tasks always run [`AggregateExecutionStage::Single`]:
    /// partial state never crosses a compiled edge, it is merged inside the
    /// task.
    pub fn aggregate_stage(&self) -> Option<AggregateExecutionStage> {
        self.aggregate_stage
    }
}

/// Strongest payload form a task accepts on its data inputs.
///
/// The variants are ordered: each one accepts every payload the previous one
/// accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AcceptedInputForm {
    /// One complete resident frame only.
    Frame,
    /// A complete resident frame or a resident frame stream.
    FrameOrStream,
    /// Also a deferred source stream the task re-stages and pulls itself.
    FrameStreamOrSourceStream,
    /// Also an in-flight source prefetch the task finishes itself.
    AnyProducedForm,
}

impl AcceptedInputForm {
    /// Stable diagnostic label for this accepted input form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frame => "frame",
            Self::FrameOrStream => "frame_or_stream",
            Self::FrameStreamOrSourceStream => "frame_stream_or_source_stream",
            Self::AnyProducedForm => "any_produced_form",
        }
    }
}

/// Final-sort execution strategy, selected once at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortStrategy {
    /// Sort the whole input, then hand one frame downstream.
    FullSort,
    /// Fuse the single downstream limit into a top-k over the sort input.
    FusedTopK {
        /// Task that applies the fused limit.
        limit_task: PipelineTaskId,
        /// Plan node of the fused limit.
        limit_node: NodeId,
    },
}

impl SortStrategy {
    /// Stable diagnostic label for this sort strategy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullSort => "full_sort",
            Self::FusedTopK { .. } => "fused_topk",
        }
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
    /// Normalize an edge relation before graph execution.
    EdgeNormalize,
    /// Execute one graph algorithm after its edge and side-input frames are resident.
    GraphAlgorithm,
    /// Execute one vector algorithm after its relation input frames are resident.
    VectorAlgorithm,
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
            Self::EdgeNormalize => "edge_normalize",
            Self::GraphAlgorithm => "graph_algorithm",
            Self::VectorAlgorithm => "vector_algorithm",
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
}

impl PipelineEdge {
    /// Create an edge between two tasks.
    pub fn new(id: PipelineEdgeId, from: PipelineTaskId, to: PipelineTaskId, kind: PipelineEdgeKind) -> Self {
        Self { id, from, to, kind, resources: Vec::new() }
    }

    /// Attach resident resources that flow across this edge.
    pub fn with_resources(mut self, resources: Vec<ResidentResourceId>) -> Self {
        self.resources = resources;
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
}

impl PipelineBarrier {
    /// Create a barrier with no attached tasks.
    pub fn new(id: PipelineBarrierId, kind: PipelineBarrierKind) -> Self {
        Self { id, kind, upstream_tasks: Vec::new(), downstream_tasks: Vec::new() }
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
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentResource {
    id: ResidentResourceId,
    producer: PipelineTaskId,
    consumers: Vec<PipelineTaskId>,
    lifetime: ResidentLifetime,
    data_form: ResidentDataForm,
    sharing: ResourceSharing,
    ordering: Option<ResourceOrdering>,
}

impl ResidentResource {
    /// Create a resident resource with its producer, consumers, and lifetime.
    ///
    /// The consumption contract defaults to the narrowest form a consumer can
    /// require: one complete resident frame, moved into a single consumer or
    /// read as a shared immutable frame.
    /// [`compile_pipeline`](crate::pipeline::compile_pipeline) replaces it with
    /// the contract it resolved for the resource.
    pub fn new(
        id: ResidentResourceId,
        producer: PipelineTaskId,
        consumers: Vec<PipelineTaskId>,
        lifetime: ResidentLifetime,
    ) -> Self {
        let sharing =
            if consumers.len() > 1 { ResourceSharing::SharedImmutableFrame } else { ResourceSharing::SingleConsume };
        Self { id, producer, consumers, lifetime, data_form: ResidentDataForm::Frame, sharing, ordering: None }
    }

    /// Record the payload form and sharing mode resolved for this resource.
    pub fn with_consumption(mut self, data_form: ResidentDataForm, sharing: ResourceSharing) -> Self {
        self.data_form = data_form;
        self.sharing = sharing;
        self
    }

    /// Record the row ordering this resource's rows are known to carry.
    pub fn with_ordering(mut self, ordering: Option<ResourceOrdering>) -> Self {
        self.ordering = ordering;
        self
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

    /// Strongest payload form the producer may hand this resource's consumers.
    ///
    /// The producer may hand over a weaker form than this: a deferred source
    /// read that observes a single chunk materializes it, and a shared stream
    /// is collected eagerly. It never hands over a stronger one.
    pub fn data_form(&self) -> ResidentDataForm {
        self.data_form
    }

    /// How this resource's consumers read it.
    pub fn sharing(&self) -> ResourceSharing {
        self.sharing
    }

    /// Row ordering this resource's rows are known to carry, if any.
    pub fn ordering(&self) -> Option<&ResourceOrdering> {
        self.ordering.as_ref()
    }
}

/// Ordering a resource's rows are known to carry, and over what extent.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceOrdering {
    keys: Vec<SortExpr>,
    scope: OrderingScope,
}

impl ResourceOrdering {
    /// Ordering keys in precedence order, with direction and null placement.
    pub fn keys(&self) -> &[SortExpr] {
        &self.keys
    }

    /// Extent the key ordering holds over.
    pub fn scope(&self) -> OrderingScope {
        self.scope
    }

    /// Ordering that holds over every row of the producer.
    pub fn global(keys: Vec<SortExpr>) -> Self {
        Self { keys, scope: OrderingScope::Global }
    }

    /// Ordering that holds only within each emitted chunk.
    pub fn per_chunk(keys: Vec<SortExpr>) -> Self {
        Self { keys, scope: OrderingScope::PerChunk }
    }
}

/// Extent a recorded row ordering holds over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingScope {
    /// Rows are ordered only within each emitted chunk; chunks in sequence
    /// are not ordered relative to each other.
    PerChunk,
    /// Every row of the producer is ordered relative to every other row.
    Global,
}

impl OrderingScope {
    /// Stable diagnostic label for this ordering scope.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PerChunk => "per_chunk",
            Self::Global => "global",
        }
    }
}

/// Payload form a compiled resident resource may carry between tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentDataForm {
    /// One complete resident frame.
    Frame,
    /// A resident frame stream the consumers pull chunk by chunk.
    FrameStream,
    /// A deferred source stream the consumers re-stage and pull themselves.
    SourceStream,
}

impl ResidentDataForm {
    /// Stable diagnostic label for this payload form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frame => "frame",
            Self::FrameStream => "frame_stream",
            Self::SourceStream => "source_stream",
        }
    }
}

/// How several consumers read one resident resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceSharing {
    /// Exactly one consumer; the payload moves into it.
    SingleConsume,
    /// Several consumers read the same immutable resident frame, or the same
    /// eagerly collected chunk list.
    SharedImmutableFrame,
    /// Several consumers each take an independent cursor over the producer.
    IndependentCursorStream,
}

impl ResourceSharing {
    /// Stable diagnostic label for this sharing mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SingleConsume => "single_consume",
            Self::SharedImmutableFrame => "shared_immutable_frame",
            Self::IndependentCursorStream => "independent_cursor_stream",
        }
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

/// Explicit result boundary for a compiled pipeline graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkBoundary {
    kind: SinkBoundaryKind,
    input: PipelineTaskId,
}

impl SinkBoundary {
    /// Create a sink boundary fed by `input`.
    pub fn new(kind: SinkBoundaryKind, input: PipelineTaskId) -> Self {
        Self { kind, input }
    }

    /// Sink boundary kind.
    pub fn kind(&self) -> SinkBoundaryKind {
        self.kind
    }

    /// Task that produces the sink input resource.
    pub fn input(&self) -> PipelineTaskId {
        self.input
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
