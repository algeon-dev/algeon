//! Capability-analysis model for native query admission.
//!
//! This module is the public vocabulary for local node, edge, and subgraph
//! capabilities. Rejection reasons and fact keys are report-facing stable strings;
//! add new labels instead of reusing old whole-plan family names.
use std::collections::{BTreeMap, BTreeSet};

use super::{
    AggregateFunction, AggregateStrategyCapability, GraphAlgorithm, JoinSide, JoinType, NodeId,
    aggregate::AggregateStateContract,
};
use crate::plan::{EdgeDedupStrategy, EdgeWeightCombiner, VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams};

/// The most local graph scope that owns a capability decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityScope {
    /// Decision owned by one native plan node.
    Node,
    /// Decision owned by one producer-consumer boundary.
    Edge,
    /// Decision owned by a non-local graph constraint or compiled pipeline.
    Subgraph,
}

impl CapabilityScope {
    /// Stable snake-case label emitted in contract diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
            Self::Subgraph => "subgraph",
        }
    }
}

/// Rejection code for precision-sensitive floating statistical aggregates over
/// join pipelines whose final output is a full sort. The aggregate itself may
/// have a DataFusion-exact host Welford implementation, but capability-envelope
/// admission still needs a plan-level order/batch and full-sort bound evidence.
pub const STATISTICAL_JOIN_FINAL_SORT_REQUIRES_EXACT_CONTRACTS: &str =
    "statistical_join_final_sort_requires_exact_contracts";

/// A symbolic dependency that a conditional capability is waiting for.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintRef {
    /// Graph scope that owns the prerequisite.
    pub scope: ConstraintScope,
    /// Stable local constraint name within that scope.
    pub name: &'static str,
}

impl ConstraintRef {
    /// Creates a prerequisite reference owned by a node.
    pub const fn node(node_id: NodeId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Node(node_id), name }
    }

    /// Creates a prerequisite reference owned by a producer-consumer edge.
    pub const fn edge(producer: NodeId, consumer: EdgeConsumer, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Edge { producer, consumer }, name }
    }

    /// Creates a prerequisite reference owned by a subgraph invariant.
    pub const fn invariant(invariant_id: InvariantId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Invariant(invariant_id), name }
    }
}

/// The graph scope where a constraint belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintScope {
    /// Constraint attached to one native plan node.
    Node(NodeId),
    /// Constraint attached to one producer-consumer boundary.
    Edge {
        /// Producer node on the dependency edge.
        producer: NodeId,
        /// Consumer endpoint on the dependency edge.
        consumer: EdgeConsumer,
    },
    /// Constraint attached to a non-local subgraph invariant.
    Invariant(InvariantId),
}

/// Capability result for a node, edge, or invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecision {
    /// The capability is fully proven at its owning scope.
    Satisfied,
    /// The capability failed at this scope; facts should name the local cause.
    Rejected {
        /// Stable rejection code for reports and tests.
        reason: &'static str,
        /// Stable fact-key map with values formatted for diagnostics.
        facts: BTreeMap<&'static str, String>,
    },
    /// The capability depends on other graph-scope constraints.
    Conditional {
        /// Prerequisites that must be satisfied before this decision is final.
        waiting_on: Vec<ConstraintRef>,
    },
}

impl CapabilityDecision {
    pub(super) fn rejected(reason: &'static str, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self::Rejected { reason, facts: facts.into_iter().collect() }
    }

    /// Returns the stable rejection code for rejected decisions.
    pub const fn rejection_reason(&self) -> Option<&'static str> {
        match self {
            Self::Rejected { reason, .. } => Some(reason),
            Self::Satisfied | Self::Conditional { .. } => None,
        }
    }

    /// Returns the stable rejection fact map for rejected decisions.
    pub fn rejection_facts(&self) -> Option<&BTreeMap<&'static str, String>> {
        match self {
            Self::Rejected { facts, .. } => Some(facts),
            Self::Satisfied | Self::Conditional { .. } => None,
        }
    }
}

/// Node-local executable capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCapability {
    /// The native plan node this capability describes.
    pub node_id: NodeId,
    /// The node's operator-level shape, independent of whole-query families.
    pub operator: NodeOperator,
    /// Whether this node is executable before considering downstream edges.
    pub executable: CapabilityDecision,
    /// Properties this node makes available to consumers.
    pub produced_properties: Vec<ProducedProperty>,
    /// Aggregate state contract, set on aggregate nodes with a classifiable
    /// spec; edges consult it instead of re-deriving setop/dedup proofs.
    pub aggregate_state: Option<AggregateStateContract>,
    /// Rows this node emits at most, when the operator alone bounds it
    /// (scalar aggregate, ≤1 fetch limit, ≤1 bounded source).
    pub max_output_rows: Option<u64>,
}

/// Native operator shape captured without naming a whole bounded family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOperator {
    /// Local or resident source scan.
    Source {
        /// Physical input format and residency source.
        format: SourceFormatCapability,
    },
    /// Predicate filter preserving row order and residency when admitted.
    Filter,
    /// Expression projection over a resident input frame.
    Projection,
    /// Aggregate operator with grouping and aggregate-function capability facts.
    Aggregate {
        /// Grouping shape used for capability admission.
        grouping: AggregateGrouping,
        /// Aggregate functions that must be supported by the native engine.
        functions: Vec<AggregateFunction>,
    },
    /// Window operator over a resident input.
    Window {
        /// Number of window functions that participate in output sizing.
        function_count: usize,
    },
    /// Sort or top-k ordering barrier.
    Sort {
        /// Whether the plan shape requires the operator to establish ordering.
        ordering: OrderingRequirement,
    },
    /// Row-count limit over a resident input.
    Limit,
    /// Join operator with local join-shape facts.
    Join {
        /// Logical join type requested by the plan.
        join_type: JoinType,
        /// Number of equality key pairs.
        key_count: usize,
        /// Whether an additional residual predicate must be evaluated.
        residual: ResidualRequirement,
        /// Input side chosen as the resident build side.
        build_side: JoinSide,
    },
    /// Union over multiple resident inputs.
    Union {
        /// Number of input edges expected by the union node.
        input_count: usize,
    },
    /// Graph edge normalization operator before graph construction.
    EdgeNormalize {
        /// Whether reverse edges are inserted for undirected processing.
        add_reverse: bool,
        /// Duplicate-edge handling requested by the graph operation.
        dedup: EdgeDedupStrategy,
        /// Weight-combining rule applied when duplicate edges are collapsed.
        weight_combiner: EdgeWeightCombiner,
    },
    /// cuGraph algorithm execution over a constructed graph.
    GraphAlgorithm {
        /// Algorithm family whose parameter contract must be supported.
        algorithm: GraphAlgorithm,
    },
    /// cuVS vector algorithm execution over one or more resident relations.
    VectorAlgorithm {
        /// Algorithm family whose parameter contract must be supported.
        algorithm: VectorAlgorithm,
    },
}

/// Source format capability known by the native engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormatCapability {
    /// Local Parquet files decoded through the native source reader.
    LocalParquet,
    /// Already resident cudf table supplied by an upstream native node.
    GpuResident,
    /// Host Arrow literal batches imported by native execution.
    LiteralBatch,
}

/// Aggregate grouping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateGrouping {
    /// Scalar aggregate with no grouping keys.
    Scalar,
    /// Standard grouped aggregate with explicit grouping keys.
    Grouped,
    /// Grouping sets, rollups, or cubes, which require separate support.
    GroupingSets,
}

/// Whether a node establishes an ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingRequirement {
    /// The operator or edge must establish an ordered output.
    Required,
    /// No ordering evidence is required.
    None,
}

/// Whether a join has a residual predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualRequirement {
    /// Join has no residual predicate beyond equality keys.
    None,
    /// Join includes a residual predicate that must be proven supported.
    Required,
}

/// Properties produced by a node for downstream edge contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProducedProperty {
    /// Output is a resident cudf frame available to downstream consumers.
    ResidentFrame,
    /// Output is a grouped aggregate frame.
    GroupedFrame,
    /// Output is a one-row scalar aggregate frame.
    ScalarFrame,
    /// Output ordering is established for downstream order-sensitive consumers.
    OrderedFrame,
}

/// Producer-consumer capability for one dependency edge.
///
/// Every edge requires its producer to yield a GPU-resident cudf frame;
/// `decision` is rejected with `missing_edge_required_property` otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeContract {
    /// Producer node for this dependency.
    pub producer: NodeId,
    /// Consumer node or final sink for this dependency.
    pub consumer: EdgeConsumer,
    /// Contract classes carried by this producer-consumer relationship.
    pub contract_kinds: Vec<EdgeContractKind>,
    /// Semantic input slot on the consumer.
    pub input_role: InputRole,
    /// Whether this edge satisfies its consumer contract.
    pub decision: CapabilityDecision,
}

/// Producer-consumer contract classes attached to one edge.
///
/// A single graph edge can carry more than one contract. For example, a
/// local-Parquet source feeding a join build side owns both the source-read
/// resident input contract and the build-side residency contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeContractKind {
    /// Source output must be resident before the consumer runs.
    ResidentSourceInput {
        /// Source format that supplies the resident input.
        format: SourceFormatCapability,
    },
    /// Aggregate input must be a resident frame.
    ResidentAggregateInput,
    /// Window input must be a resident frame.
    WindowInput,
    /// Join build input residency and ordering contract.
    JoinBuildInput {
        /// Join side consumed as the build input.
        side: JoinSide,
    },
    /// Join probe input residency contract.
    JoinProbeInput {
        /// Join side consumed as the probe input.
        side: JoinSide,
    },
    /// Union input position contract.
    UnionInput {
        /// Zero-based union input index.
        input_index: usize,
    },
    /// Multiple consumers reuse one resident producer because
    /// common-subexpression elimination preserved the same producer node id.
    SharedProducer,
    /// Generic resident-frame flow between native nodes.
    ResidentFrameFlow,
    /// Materialized join output required by a downstream consumer.
    JoinMaterialization,
    /// Sort input residency contract.
    SortInput,
    /// Graph algorithm input residency contract.
    GraphAlgorithmInput,
    /// Vector algorithm relation-input residency contract.
    VectorAlgorithmInput {
        /// Semantic relation role within the vector algorithm.
        role: VectorInputRole,
    },
    /// Final result materialization boundary.
    ExplicitSinkMaterialization,
}

/// Consumer endpoint for an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeConsumer {
    /// Another native plan node consumes the edge.
    Node(NodeId),
    /// Final result sink consumes the edge.
    Sink(ResultSinkKind),
}

/// Result sink kinds as capability consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResultSinkKind {
    /// Host Arrow batches are materialized for the caller.
    HostArrow,
    /// Output remains resident on the GPU for a native downstream consumer.
    GpuResident,
}

/// Semantic role of an input edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputRole {
    /// Single ordinary input for unary operators.
    UnaryInput,
    /// Input consumed by an aggregate node.
    AggregateInput,
    /// Input consumed by a window node.
    WindowInput,
    /// Input consumed by a sort or top-k node.
    SortInput,
    /// Input consumed at a specific union position.
    UnionInput {
        /// Zero-based union input index.
        input_index: usize,
    },
    /// Input consumed by a graph algorithm node.
    GraphAlgorithmInput,
    /// Input consumed by a vector algorithm node.
    VectorAlgorithmInput {
        /// Semantic relation role within the vector algorithm.
        role: VectorInputRole,
    },
    /// Input consumed by a limit node.
    LimitInput,
    /// Build-side input consumed by a join node.
    JoinBuildInput {
        /// Join side consumed as build.
        side: JoinSide,
    },
    /// Probe-side input consumed by a join node.
    JoinProbeInput {
        /// Join side consumed as probe.
        side: JoinSide,
    },
    /// Final result sink input.
    SinkInput,
}

/// Stable semantic role for a relation consumed by a vector algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorInputRole {
    /// Indexed dataset for exact brute-force kNN.
    Dataset,
    /// Query vectors for exact brute-force kNN.
    Queries,
    /// Training/transform input for K-means or PCA.
    Input,
    /// First axis of a pairwise distance result.
    X,
    /// Second axis of a pairwise distance result.
    Y,
    /// Relation-backed initial K-means centroids.
    InitialCentroids,
}

impl VectorInputRole {
    #[doc(hidden)]
    pub fn for_vector_input(vector: &VectorAlgorithmNode, input_index: usize) -> Self {
        match (&vector.spec.params, input_index) {
            (VectorAlgorithmParams::BruteForceKnn(_) | VectorAlgorithmParams::IvfFlat(_), 0) => Self::Dataset,
            (VectorAlgorithmParams::BruteForceKnn(_) | VectorAlgorithmParams::IvfFlat(_), 1) => Self::Queries,
            (VectorAlgorithmParams::PairwiseDistance(_), 0) => Self::X,
            (VectorAlgorithmParams::PairwiseDistance(_), 1) => Self::Y,
            (VectorAlgorithmParams::TopK(_), 0) => Self::Input,
            (VectorAlgorithmParams::KMeans(_), 0) | (VectorAlgorithmParams::Pca(_), 0) => Self::Input,
            (VectorAlgorithmParams::KMeans(_), 1) => Self::InitialCentroids,
            _ => Self::Input,
        }
    }

    /// Stable snake-case diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Queries => "queries",
            Self::Input => "input",
            Self::X => "x",
            Self::Y => "y",
            Self::InitialCentroids => "initial_centroids",
        }
    }
}

/// Stable identifier for an invariant inside one analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InvariantId(pub usize);

impl InvariantId {
    /// Returns the zero-based index within one capability analysis.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A non-local statement that cannot be proven from one node or edge alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphInvariant {
    /// Stable identifier within this capability analysis result.
    pub invariant_id: InvariantId,
    /// The kind of non-local graph property being proven.
    pub kind: SubgraphInvariantKind,
    /// Nodes covered by the invariant.
    pub nodes: Vec<NodeId>,
    /// Scheduling property required from the compiled pipeline.
    pub required_schedule: ScheduleConstraint,
    /// Resident resources whose lifetime participates in the evidence.
    pub required_resources: Vec<ResidentResource>,
    /// Whether the invariant is satisfied for this graph.
    pub decision: CapabilityDecision,
}

/// Subgraph invariant vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgraphInvariantKind {
    /// The join's build input can be made resident before the probe side needs
    /// it, without a graph cycle through the join itself.
    BuildBeforeProbe {
        /// Join whose build side must be scheduled before its probe side.
        join_id: NodeId,
    },
    /// The reachable join region below the root is closed over supported join,
    /// projection, filter, and source nodes.
    ///
    /// Retired in Step S3: multi-join aggregate scope is decided by the
    /// compiled `Aggregate`/`JoinBuild`/`JoinProbe`/`JoinMaterialize` task
    /// contracts, not by a plan-shape invariant. The variant stays so
    /// downstream exhaustive matches keep compiling.
    ClosedJoinSubtree {
        /// Root join of the closed resident join region.
        root_join_id: NodeId,
    },
    /// The compiled graph can schedule all reachable joins in dependency order
    /// so build/probe residency requirements are not violated by join nesting.
    TopologicalBuildProbeSchedule,
    /// One resident producer is shared by multiple consumers and each consumer
    /// edge preserves that producer identity.
    SharedResidentSubplan,
    /// A scalar aggregate can consume a shared resident producer only when every
    /// aggregate expression derives from columns produced by that same shared
    /// node after transparent projections are peeled.
    AggregateDerivesFromSharedProducer,
    /// A final ordering barrier consumes the resident input after all upstream
    /// producers are complete and does not feed back into its input.
    FinalBarrierOrder,
    /// Post-root stages above a root join remain in the specific order required
    /// by the admitted envelope.
    PostRootStageOrder {
        /// Root join whose post-root stages are being ordered.
        root_join_id: NodeId,
    },
    /// A projection wrapper can be ignored for residency only when both the
    /// peeled producer and the projection output are resident frames.
    ProjectionTransparentResidency,
    /// A filter/projection chain preserves residency only when every node in
    /// the connected unary chain consumes the previous resident output and also
    /// produces a resident frame.
    ResidentUnaryChainPreservesResidency,
    /// The residual predicate on the join references both join inputs, so the
    /// residual is a true join condition rather than a single-side filter.
    ResidualUsesBothInputs {
        /// Join whose residual predicate is being checked.
        join_id: NodeId,
    },
    /// A root projection is bounded by its immediate fetch-bearing limit input.
    ///
    /// Retired in Step S1: the limit/projection hand-off is decided by the
    /// compiled `Limit`/`Unary` task contracts, not by a plan-shape invariant.
    /// The variant stays so downstream exhaustive matches keep compiling.
    LimitBoundedProjection,
    /// A floating statistical aggregate participates in a join pipeline whose
    /// final output is a full sort; capability-envelope admission needs an explicit
    /// exactness envelope before this can replace the DataFusion baseline.
    StatisticalJoinFinalSortExactness,
    /// Every reachable resident producer remains valid for all of its admitted
    /// consumers, and no rejected producer-consumer edge is hidden by the
    /// broader compiled subgraph admission.
    ResidentLifetimeDominatesConsumers,
}

/// Scheduling evidence required by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleConstraint {
    /// No non-local scheduling constraint is required.
    None,
    /// Build-side work must complete before probe-side work.
    BuildBeforeProbe,
    /// Nodes must be schedulable in topological dependency order.
    Topological,
    /// A final barrier must wait for all upstream producers.
    Barrier,
}

/// Resident resources whose lifetime or ordering is governed by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentResource {
    /// Resident output table produced by one node.
    NodeOutput(NodeId),
    /// Resident join build input retained for a join.
    JoinBuildInput {
        /// Join that owns the resident build input.
        join_id: NodeId,
        /// Producer node supplying the build input.
        input: NodeId,
    },
    /// Resident join probe input retained for a join.
    JoinProbeInput {
        /// Join that owns the resident probe input.
        join_id: NodeId,
        /// Producer node supplying the probe input.
        input: NodeId,
    },
}

/// Report-facing projection of the first capability rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRejectionProjection {
    /// Local owner whose rejection is projected into diagnostics.
    pub owner: CapabilityRejectionOwner,
    /// Stable rejection category for report/fact projections.
    pub category: CapabilityRejectionCategory,
    /// Contract scope used by capability-envelope reports.
    pub contract_scope: CapabilityScope,
    /// Stable diagnostic boundary label.
    pub boundary: &'static str,
    /// Stable rejection code.
    pub reason: &'static str,
    /// Stable fact-key map copied from the rejected capability decision.
    pub facts: BTreeMap<&'static str, String>,
}

/// Stable category for rejected capability or envelope admission projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityRejectionCategory {
    /// An expression family is not supported by the native engine.
    UnsupportedExpressionFamily,
    /// An Arrow data type is not supported by the native engine.
    UnsupportedArrowDtype,
    /// A native plan node operator or local operator shape is unsupported.
    UnsupportedNodeOperator,
    /// A producer-consumer edge contract is missing.
    MissingEdgeContract,
    /// A subgraph invariant or envelope evidence is missing.
    MissingSubgraphInvariant,
    /// The final sink boundary is outside the compiled bounded runtime envelope.
    UnsupportedSinkBoundary,
    /// The compiled pipeline graph is outside the capability-envelope.
    UnsupportedCompiledPipelineEnvelope,
}

impl CapabilityRejectionCategory {
    /// Stable snake-case label emitted in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedExpressionFamily => "unsupported_expression_family",
            Self::UnsupportedArrowDtype => "unsupported_arrow_dtype",
            Self::UnsupportedNodeOperator => "unsupported_node_operator",
            Self::MissingEdgeContract => "missing_edge_contract",
            Self::MissingSubgraphInvariant => "missing_subgraph_invariant",
            Self::UnsupportedSinkBoundary => "unsupported_sink_boundary",
            Self::UnsupportedCompiledPipelineEnvelope => "unsupported_compiled_pipeline_envelope",
        }
    }
}

/// Local capability owner projected into capability-envelope contract diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityRejectionOwner {
    /// Rejection owned by a node-local capability.
    Node(NodeId),
    /// Rejection owned by a producer-consumer edge.
    Edge {
        /// Producer node for the rejected edge.
        producer: NodeId,
        /// Consumer endpoint for the rejected edge.
        consumer: EdgeConsumer,
    },
    /// Rejection owned by a subgraph invariant.
    Invariant(InvariantId),
    /// Rejection owned by a compiled-envelope admission gate.
    Admission(&'static str),
}

/// Capability-envelope execution envelope selected by native capability admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityEnvelope {
    /// A single join is the sink root.
    RootJoin,
    /// A single join feeds admitted post-root operators.
    SingleJoinPostRootPipeline,
    /// A closed multi-join subgraph feeds the sink.
    ClosedMultiJoinSubgraph,
    /// A multi-join subplan feeds aggregate/final top-k stages.
    MultiJoinAggregateOverJoinSubplanTopK,
    /// Root top-k over keyed join and grouped-aggregate fragments.
    RootTopKJoinAggregate,
    /// Root full sort over join and grouped-aggregate fragments.
    RootFullSortJoinAggregate,
    /// Exact floating statistical aggregate over join pipelines with a final full sort.
    ExactStatisticalJoinFinalFullSort,
    /// GPU-native tolerant floating statistical aggregate over join pipelines with a final full sort.
    TolerantStatisticalJoinFinalFullSort,
    /// Projection root bounded by a global limit over join/aggregate work.
    ProjectionGlobalLimit,
    /// Projection root over scalar fanout or setop/dedup join work.
    ProjectionScalarSetop,
    /// Mixed keyed/residual join output feeds a resident grouped aggregate input.
    MixedKeyedResidualJoinResidentGroupedAggregateInput,
    /// Hash join resident grouped aggregate feeds post-join aggregate/top-k.
    HashJoinResidentGroupedAggregateInputPostJoinAggregateTopK,
    /// Rollup/rank/top-k after aggregate-join pushdown reduced the source side.
    RollupRankTopKAfterAggregateJoinPushdown,
    /// Keyed hash join consumes a shared resident subplan.
    KeyedHashJoinResidentSubplan,
    /// Keyless conditional join consumes an input subplan.
    KeylessConditionalJoinInputSubplan,
    /// Keyless conditional join consumes aggregate-over-join subplan inputs.
    KeylessConditionalJoinAggregateOverJoinSubplanInputs,
    /// Multi-join resident grouped aggregate participates in a subquery filter.
    MultiJoinResidentGroupedAggregateSubqueryFilter,
    /// Native union pipeline over admitted branch-local work.
    UnionPipeline,
    /// A pipeline the compiled task and resource contracts connect without
    /// matching any named plan family.
    ComposedPipeline,
}

impl CapabilityEnvelope {
    /// Stable snake-case label emitted in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RootJoin => "root_join",
            Self::SingleJoinPostRootPipeline => "single_join_post_root_pipeline",
            Self::ClosedMultiJoinSubgraph => "closed_multi_join_subgraph",
            Self::MultiJoinAggregateOverJoinSubplanTopK => "multi_join_aggregate_over_join_subplan_topk",
            Self::RootTopKJoinAggregate => "root_topk_join_aggregate",
            Self::RootFullSortJoinAggregate => "root_full_sort_join_aggregate",
            Self::ExactStatisticalJoinFinalFullSort => "exact_statistical_join_final_full_sort",
            Self::TolerantStatisticalJoinFinalFullSort => "tolerant_statistical_join_final_full_sort",
            Self::ProjectionGlobalLimit => "projection_global_limit",
            Self::ProjectionScalarSetop => "projection_scalar_setop",
            Self::MixedKeyedResidualJoinResidentGroupedAggregateInput => {
                "mixed_keyed_residual_join_resident_grouped_aggregate_input"
            },
            Self::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopK => {
                "hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk"
            },
            Self::RollupRankTopKAfterAggregateJoinPushdown => "rollup_rank_topk_after_aggregate_join_pushdown",
            Self::KeyedHashJoinResidentSubplan => "keyed_hash_join_resident_subplan",
            Self::KeylessConditionalJoinInputSubplan => "keyless_conditional_join_input_subplan",
            Self::KeylessConditionalJoinAggregateOverJoinSubplanInputs => {
                "keyless_conditional_join_aggregate_over_join_subplan_inputs"
            },
            Self::MultiJoinResidentGroupedAggregateSubqueryFilter => {
                "multi_join_resident_grouped_aggregate_subquery_filter"
            },
            Self::UnionPipeline => "union_pipeline",
            Self::ComposedPipeline => "composed_pipeline",
        }
    }
}

/// Capability authority result for capability-envelope selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityEnvelopeAdmission {
    /// The analysis selected a capability-envelope.
    Admitted {
        /// Selected envelope.
        envelope: CapabilityEnvelope,
    },
    /// The analysis rejected capability-envelope admission with one projection.
    Rejected {
        /// Structured report/fact projection for the rejection.
        projection: CapabilityRejectionProjection,
    },
}

/// Capability analysis for one lowered native query plan.
///
/// The node, edge, and invariant vectors are the primary evidence surface.
/// The `*_admission` decisions select compiled-executor envelopes from those
/// facts; they must not become report compatibility families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    /// Node-local executable facts.
    pub nodes: Vec<NodeCapability>,
    /// Producer-consumer edge facts.
    pub edges: Vec<EdgeContract>,
    /// Non-local scheduling, lifetime, and shared-resource facts.
    pub invariants: Vec<SubgraphInvariant>,
    /// Node-local aggregate strategy capability facts.
    pub aggregate_strategies: Vec<AggregateStrategyCapability>,
    /// Capability root-join compiled-envelope admission.
    pub root_join_admission: CapabilityDecision,
    /// Capability multi-join aggregate/top-k compiled-envelope admission.
    pub multi_join_aggregate_over_join_subplan_topk_admission: CapabilityDecision,
    /// Capability root top-k join/aggregate envelope admission.
    pub root_topk_join_aggregate_admission: CapabilityDecision,
    /// Capability root full-sort join/aggregate envelope admission.
    pub root_full_sort_join_aggregate_admission: CapabilityDecision,
    /// Capability exact statistical join/final-full-sort envelope admission.
    pub exact_statistical_join_final_full_sort_admission: CapabilityDecision,
    /// Capability mixed keyed/residual resident-aggregate compiled-envelope admission.
    pub mixed_keyed_residual_join_resident_grouped_aggregate_input_admission: CapabilityDecision,
    /// Capability hash-join resident-aggregate post-join/top-k envelope admission.
    pub hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission: CapabilityDecision,
    /// Capability rollup/rank/top-k post aggregate-join-pushdown envelope admission.
    pub rollup_rank_topk_after_aggregate_join_pushdown_admission: CapabilityDecision,
    /// Capability shared resident-subplan compiled-envelope admission.
    pub keyed_hash_join_resident_subplan_admission: CapabilityDecision,
    /// Capability keyless conditional input-subplan envelope admission.
    pub keyless_conditional_join_input_subplan_admission: CapabilityDecision,
    /// Capability keyless aggregate-over-join-inputs envelope admission.
    pub keyless_conditional_join_aggregate_over_join_subplan_inputs_admission: CapabilityDecision,
    /// Capability multi-join resident aggregate subquery-filter envelope admission.
    pub multi_join_resident_grouped_aggregate_subquery_filter_admission: CapabilityDecision,
    /// Capability union compiled-envelope admission.
    pub union_pipeline_admission: CapabilityDecision,
    /// Sink-reachable node ids of the analyzed plan.
    ///
    /// Local rejections on nodes, edges, and invariants outside this set never
    /// veto admission: the compiler emits no task for them, so they cannot
    /// break execution. A malformed plan whose reachability walk fails records
    /// every node here, preserving the unfiltered behavior for that case.
    pub(crate) reachable_node_ids: BTreeSet<NodeId>,
}

impl CapabilityAnalysis {
    /// Returns true when any node, edge, or invariant has a local rejection.
    pub fn has_rejections(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
            || self.edges.iter().any(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
            || self.invariants.iter().any(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
    }

    /// Selects the one capability-envelope admitted by this analysis, or a
    /// structured projection explaining why no capability-envelope is admissible.
    pub fn capability_envelope_admission(&self) -> CapabilityEnvelopeAdmission {
        if let Some(projection) = self.first_local_rejection_projection() {
            return CapabilityEnvelopeAdmission::Rejected { projection };
        }
        if let Some(envelope) = self.selected_specialized_capability_envelope() {
            return CapabilityEnvelopeAdmission::Admitted { envelope };
        }
        if let Some(envelope) = self.selected_generic_capability_envelope() {
            return CapabilityEnvelopeAdmission::Admitted { envelope };
        }
        if let Some(projection) = self.first_raw_admission_rejection_projection() {
            return CapabilityEnvelopeAdmission::Rejected { projection };
        }
        CapabilityEnvelopeAdmission::Rejected {
            projection: CapabilityRejectionProjection {
                owner: CapabilityRejectionOwner::Admission("capability_envelope_admission"),
                category: CapabilityRejectionCategory::MissingSubgraphInvariant,
                contract_scope: CapabilityScope::Subgraph,
                boundary: "capability_envelope_admission",
                reason: "capability_envelope_not_satisfied",
                facts: BTreeMap::from([("capability_envelope_admission", "false".to_string())]),
            },
        }
    }

    /// Returns whether the generic root-join capability evidence is satisfied.
    ///
    /// This inspection helper reports generic coverage only. Executable
    /// capability-envelope admission still goes through `capability_envelope_admission()`.
    pub fn admits_root_join(&self) -> bool {
        admission_is_satisfied(&self.root_join_admission)
    }

    /// Returns whether the capability multi-join aggregate/top-k envelope is admitted.
    pub fn admits_multi_join_aggregate_over_join_subplan_topk_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(CapabilityEnvelope::MultiJoinAggregateOverJoinSubplanTopK)
    }

    /// Returns whether the capability root top-k join/aggregate envelope is admitted.
    pub fn admits_root_topk_join_aggregate_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(CapabilityEnvelope::RootTopKJoinAggregate)
    }

    /// Returns whether the capability root full-sort join/aggregate envelope is admitted.
    pub fn admits_root_full_sort_join_aggregate_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(CapabilityEnvelope::RootFullSortJoinAggregate)
    }

    /// Returns whether the capability mixed keyed/residual aggregate-input envelope is admitted.
    pub fn admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(
            CapabilityEnvelope::MixedKeyedResidualJoinResidentGroupedAggregateInput,
        )
    }

    /// Returns whether the capability hash-join aggregate post-join/top-k envelope is admitted.
    pub fn admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(
            CapabilityEnvelope::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopK,
        )
    }

    /// Returns whether the capability rollup/rank/top-k post-pushdown envelope is admitted.
    pub fn admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(CapabilityEnvelope::RollupRankTopKAfterAggregateJoinPushdown)
    }

    /// Returns whether the capability shared resident-subplan recognizer is locally
    /// satisfied, independent of broader selected-envelope priority.
    pub fn admits_keyed_hash_join_resident_subplan_pipeline(&self) -> bool {
        admission_is_satisfied(&self.keyed_hash_join_resident_subplan_admission)
    }

    /// Returns whether the capability keyless conditional input-subplan envelope is admitted.
    pub fn admits_keyless_conditional_join_input_subplan_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(CapabilityEnvelope::KeylessConditionalJoinInputSubplan)
    }

    /// Returns whether the capability keyless aggregate-over-inputs envelope is admitted.
    pub fn admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(
            CapabilityEnvelope::KeylessConditionalJoinAggregateOverJoinSubplanInputs,
        )
    }

    /// Returns whether the capability resident aggregate subquery-filter envelope is admitted.
    pub fn admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(&self) -> bool {
        self.admits_selected_capability_envelope(CapabilityEnvelope::MultiJoinResidentGroupedAggregateSubqueryFilter)
    }

    /// Returns true when any capability-envelope join compiled envelope is admitted.
    pub fn admits_bounded_join(&self) -> bool {
        matches!(self.capability_envelope_admission(), CapabilityEnvelopeAdmission::Admitted { .. })
    }

    /// Names the plan family this analysis recognizes, for diagnostics only.
    ///
    /// A pipeline the compiled contracts connect without matching a named
    /// family reports [`CapabilityEnvelope::ComposedPipeline`]; the label never
    /// decides whether the pipeline is admitted.
    pub fn capability_envelope_label(&self) -> CapabilityEnvelope {
        self.selected_specialized_capability_envelope()
            .or_else(|| self.selected_generic_capability_envelope())
            .unwrap_or(CapabilityEnvelope::ComposedPipeline)
    }

    /// Projects the first local rejection or lower-level block into report diagnostics.
    pub fn first_rejection_projection(&self) -> Option<CapabilityRejectionProjection> {
        match self.capability_envelope_admission() {
            CapabilityEnvelopeAdmission::Admitted { .. } => None,
            CapabilityEnvelopeAdmission::Rejected { projection } => Some(projection),
        }
    }

    fn admits_selected_capability_envelope(&self, expected: CapabilityEnvelope) -> bool {
        matches!(
            self.capability_envelope_admission(),
            CapabilityEnvelopeAdmission::Admitted { envelope } if envelope == expected
        )
    }

    fn selected_specialized_capability_envelope(&self) -> Option<CapabilityEnvelope> {
        // This order is part of admission semantics. Prefer narrow, shape-specific
        // envelopes before broader projection/scalar-setop contracts, since those
        // contracts can also prove cardinality for several specialized join shapes.
        [
            (
                CapabilityEnvelope::MultiJoinAggregateOverJoinSubplanTopK,
                &self.multi_join_aggregate_over_join_subplan_topk_admission,
            ),
            (
                CapabilityEnvelope::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopK,
                &self.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
            ),
            (CapabilityEnvelope::RootTopKJoinAggregate, &self.root_topk_join_aggregate_admission),
            (
                CapabilityEnvelope::ExactStatisticalJoinFinalFullSort,
                &self.exact_statistical_join_final_full_sort_admission,
            ),
            (CapabilityEnvelope::RootFullSortJoinAggregate, &self.root_full_sort_join_aggregate_admission),
            (
                CapabilityEnvelope::MixedKeyedResidualJoinResidentGroupedAggregateInput,
                &self.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
            ),
            (
                CapabilityEnvelope::RollupRankTopKAfterAggregateJoinPushdown,
                &self.rollup_rank_topk_after_aggregate_join_pushdown_admission,
            ),
            (CapabilityEnvelope::KeyedHashJoinResidentSubplan, &self.keyed_hash_join_resident_subplan_admission),
            (
                CapabilityEnvelope::KeylessConditionalJoinInputSubplan,
                &self.keyless_conditional_join_input_subplan_admission,
            ),
            (
                CapabilityEnvelope::KeylessConditionalJoinAggregateOverJoinSubplanInputs,
                &self.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
            ),
            (
                CapabilityEnvelope::MultiJoinResidentGroupedAggregateSubqueryFilter,
                &self.multi_join_resident_grouped_aggregate_subquery_filter_admission,
            ),
            (CapabilityEnvelope::UnionPipeline, &self.union_pipeline_admission),
        ]
        .into_iter()
        .find_map(|(envelope, decision)| (*decision == CapabilityDecision::Satisfied).then_some(envelope))
    }

    fn selected_generic_capability_envelope(&self) -> Option<CapabilityEnvelope> {
        [(CapabilityEnvelope::RootJoin, &self.root_join_admission)]
            .into_iter()
            .find_map(|(envelope, decision)| (*decision == CapabilityDecision::Satisfied).then_some(envelope))
    }

    pub(crate) fn first_local_rejection_projection(&self) -> Option<CapabilityRejectionProjection> {
        self.nodes
            .iter()
            .filter(|node| self.reachable_node_ids.contains(&node.node_id))
            .find_map(|node| {
                Some(capability_rejection_projection(
                    CapabilityRejectionOwner::Node(node.node_id),
                    CapabilityScope::Node,
                    "node_capability",
                    node.executable.rejection_reason()?,
                    node.executable.rejection_facts()?,
                ))
            })
            .or_else(|| {
                self.edges
                    .iter()
                    .filter(|edge| {
                        self.reachable_node_ids.contains(&edge.producer)
                            && match edge.consumer {
                                EdgeConsumer::Node(consumer) => self.reachable_node_ids.contains(&consumer),
                                EdgeConsumer::Sink(_) => true,
                            }
                    })
                    .find_map(|edge| {
                        Some(capability_rejection_projection(
                            CapabilityRejectionOwner::Edge { producer: edge.producer, consumer: edge.consumer },
                            CapabilityScope::Edge,
                            "edge_capability",
                            edge.decision.rejection_reason()?,
                            edge.decision.rejection_facts()?,
                        ))
                    })
            })
            .or_else(|| {
                self.invariants
                    .iter()
                    .filter(|invariant| invariant.nodes.iter().all(|node| self.reachable_node_ids.contains(node)))
                    .find_map(|invariant| {
                        Some(capability_rejection_projection(
                            CapabilityRejectionOwner::Invariant(invariant.invariant_id),
                            CapabilityScope::Subgraph,
                            "subgraph_capability",
                            invariant.decision.rejection_reason()?,
                            invariant.decision.rejection_facts()?,
                        ))
                    })
            })
    }

    fn first_raw_admission_rejection_projection(&self) -> Option<CapabilityRejectionProjection> {
        if root_topk_join_aggregate_rejection_projects_before_root_join(&self.root_topk_join_aggregate_admission) {
            return Some(capability_rejection_projection(
                CapabilityRejectionOwner::Admission("root_topk_join_aggregate_admission"),
                CapabilityScope::Subgraph,
                "admission_capability",
                self.root_topk_join_aggregate_admission.rejection_reason()?,
                self.root_topk_join_aggregate_admission.rejection_facts()?,
            ));
        }
        [
            ("root_join_admission", &self.root_join_admission),
            (
                "multi_join_aggregate_over_join_subplan_topk_admission",
                &self.multi_join_aggregate_over_join_subplan_topk_admission,
            ),
            (
                "mixed_keyed_residual_join_resident_grouped_aggregate_input_admission",
                &self.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
            ),
            (
                "hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission",
                &self.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
            ),
            (
                "rollup_rank_topk_after_aggregate_join_pushdown_admission",
                &self.rollup_rank_topk_after_aggregate_join_pushdown_admission,
            ),
            ("keyed_hash_join_resident_subplan_admission", &self.keyed_hash_join_resident_subplan_admission),
            (
                "keyless_conditional_join_input_subplan_admission",
                &self.keyless_conditional_join_input_subplan_admission,
            ),
            (
                "keyless_conditional_join_aggregate_over_join_subplan_inputs_admission",
                &self.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
            ),
            (
                "multi_join_resident_grouped_aggregate_subquery_filter_admission",
                &self.multi_join_resident_grouped_aggregate_subquery_filter_admission,
            ),
            ("union_pipeline_admission", &self.union_pipeline_admission),
        ]
        .into_iter()
        .find_map(|(admission, decision)| {
            Some(capability_rejection_projection(
                CapabilityRejectionOwner::Admission(admission),
                CapabilityScope::Subgraph,
                "admission_capability",
                decision.rejection_reason()?,
                decision.rejection_facts()?,
            ))
        })
    }
}

fn root_topk_join_aggregate_rejection_projects_before_root_join(decision: &CapabilityDecision) -> bool {
    matches!(
        decision.rejection_reason(),
        Some(
            "root_topk_join_aggregate_requires_fetch_limit"
                | "root_topk_join_aggregate_rejects_distinct_subcontract"
                | "root_topk_join_aggregate_rejects_rollup_subcontract"
                | "root_topk_join_aggregate_rejects_window_subcontract"
                | "root_topk_join_aggregate_requires_grouped_aggregate_fragment"
                | "root_topk_join_aggregate_requires_ordered_window_input"
                | "root_topk_join_aggregate_rejects_nested_blocking_or_branch_node"
                | "root_topk_join_aggregate_rejects_shared_or_fanout_subplan"
                | "root_topk_join_aggregate_rejects_sort_fetch_hash_join_input_subcontract"
                | "root_topk_join_aggregate_rejects_sort_fetch_hash_join_aggregate_subcontract"
                | "root_topk_join_aggregate_requires_keyed_inner_hash_join"
                | "root_topk_join_aggregate_requires_supported_join_state"
                | "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join"
                | "root_topk_join_aggregate_rejects_keyless_outer_cross_subcontract"
                | "root_topk_join_aggregate_rejects_unsupported_distinct_state"
        )
    )
}

fn admission_is_satisfied(decision: &CapabilityDecision) -> bool {
    *decision == CapabilityDecision::Satisfied
}

fn capability_rejection_projection(
    owner: CapabilityRejectionOwner,
    contract_scope: CapabilityScope,
    boundary: &'static str,
    reason: &'static str,
    facts: &BTreeMap<&'static str, String>,
) -> CapabilityRejectionProjection {
    let category = capability_rejection_category(&owner, reason, facts);
    CapabilityRejectionProjection { owner, category, contract_scope, boundary, reason, facts: facts.clone() }
}

fn capability_rejection_category(
    owner: &CapabilityRejectionOwner,
    reason: &str,
    facts: &BTreeMap<&'static str, String>,
) -> CapabilityRejectionCategory {
    if facts.contains_key("expression_family") {
        return CapabilityRejectionCategory::UnsupportedExpressionFamily;
    }
    if facts.contains_key("data_type")
        || facts.contains_key("dtype")
        || reason.contains("data_type")
        || reason.contains("dtype")
    {
        return CapabilityRejectionCategory::UnsupportedArrowDtype;
    }
    match owner {
        CapabilityRejectionOwner::Node(_) => CapabilityRejectionCategory::UnsupportedNodeOperator,
        CapabilityRejectionOwner::Edge { .. } => CapabilityRejectionCategory::MissingEdgeContract,
        CapabilityRejectionOwner::Invariant(_) | CapabilityRejectionOwner::Admission(_) => {
            CapabilityRejectionCategory::MissingSubgraphInvariant
        },
    }
}
