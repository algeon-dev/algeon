//! Capability-analysis model for native query admission.
//!
//! This module is the public vocabulary for local node, edge, and subgraph
//! capabilities. Rejection reasons and fact keys are report-facing stable strings.
//! New cases get new labels, and old whole-plan family names are not reused.
use std::collections::BTreeMap;

use super::{
    AggregateFunction, GraphAlgorithm, JoinSide, JoinType, NodeId,
    aggregate::AggregateStateContract,
    rejection::{Reason, Rejection, RejectionOwner},
};
use crate::{
    plan::{
        EdgeDedupStrategy, EdgeWeightCombiner, MlAlgorithm, MlRelationRole, VectorAlgorithm, VectorAlgorithmNode,
        VectorAlgorithmParams,
    },
    policy::PlanningPolicy,
};

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
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
            Self::Subgraph => "subgraph",
        }
    }
}

/// Capability result for a node, edge, or invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecision {
    /// The capability is fully proven at its owning scope.
    Satisfied,
    /// The capability failed at this scope; facts should name the local cause.
    Rejected {
        /// Typed rejection reason; its label is the report code.
        reason: Reason,
        /// Stable fact-key map with values formatted for diagnostics.
        facts: BTreeMap<&'static str, String>,
    },
}

impl CapabilityDecision {
    pub(super) fn rejected(reason: Reason, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self::Rejected { reason, facts: facts.into_iter().collect() }
    }

    /// Returns the rejection reason for rejected decisions.
    pub(crate) const fn rejection_reason(&self) -> Option<Reason> {
        match self {
            Self::Rejected { reason, .. } => Some(*reason),
            Self::Satisfied => None,
        }
    }

    /// Returns the stable rejection fact map for rejected decisions.
    pub(crate) fn rejection_facts(&self) -> Option<&BTreeMap<&'static str, String>> {
        match self {
            Self::Rejected { facts, .. } => Some(facts),
            Self::Satisfied => None,
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
    /// Aggregate state contract, set on aggregate nodes with a classifiable
    /// spec. Edges consult it, and no edge re-derives setop/dedup proofs.
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
    /// cuML dimensionality reduction over resident relations.
    MlAlgorithm {
        /// Algorithm family whose parameter contract must be supported.
        algorithm: MlAlgorithm,
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

/// Whether the plan requires an operator or edge to establish an ordering.
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

/// Producer-consumer capability for one dependency edge.
///
/// Every edge requires its producer to exist in the plan and to yield a
/// GPU-resident cudf frame. `decision` carries `missing_edge_producer` when the
/// producer is absent, which plan construction already rules out; every other
/// edge is satisfied here and judged by the compiled pipeline contracts.
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
    /// cuML relation-input residency contract.
    MlAlgorithmInput {
        /// Relation role identifying the algorithm input.
        role: MlRelationRole,
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
    /// A deferred host sink is analyzed as this kind, because execution treats
    /// it as GPU-resident and the frontend materializes the host result.
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
    /// Input consumed by a cuML node.
    MlAlgorithmInput {
        /// Relation role identifying the algorithm input.
        role: MlRelationRole,
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
    /// Name of the relation role in SQL and diagnostics.
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

    /// Role of the relation at `input_index` among the node's inputs.
    pub fn for_vector_input(vector: &VectorAlgorithmNode, input_index: usize) -> Self {
        match (&vector.spec.params, input_index) {
            (
                VectorAlgorithmParams::BruteForceKnn(_)
                | VectorAlgorithmParams::IvfFlat(_)
                | VectorAlgorithmParams::IvfPq(_)
                | VectorAlgorithmParams::Cagra(_),
                0,
            ) => Self::Dataset,
            (
                VectorAlgorithmParams::BruteForceKnn(_)
                | VectorAlgorithmParams::IvfFlat(_)
                | VectorAlgorithmParams::IvfPq(_)
                | VectorAlgorithmParams::Cagra(_),
                1,
            ) => Self::Queries,
            (VectorAlgorithmParams::PairwiseDistance(_), 0) => Self::X,
            (VectorAlgorithmParams::PairwiseDistance(_), 1) => Self::Y,
            (VectorAlgorithmParams::TopK(_), 0) => Self::Input,
            (VectorAlgorithmParams::KMeans(_), 0) | (VectorAlgorithmParams::Pca(_), 0) => Self::Input,
            (VectorAlgorithmParams::KMeans(_), 1) => Self::InitialCentroids,
            _ => Self::Input,
        }
    }
}

/// Stable identifier for an invariant inside one analysis result. Its index is
/// the invariant's position in `CapabilityAnalysis::invariants`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InvariantId(
    /// Zero-based position in the capability analysis invariant list.
    pub usize,
);

impl InvariantId {
    /// Returns the zero-based index within one capability analysis.
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

/// A non-local statement that cannot be proven from one node or edge alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphInvariant {
    /// Stable identifier within this capability analysis result.
    pub(crate) invariant_id: InvariantId,
    /// The kind of non-local graph property being proven.
    pub kind: SubgraphInvariantKind,
    /// Nodes covered by the invariant.
    pub nodes: Vec<NodeId>,
    /// Scheduling property required from the compiled pipeline.
    pub(crate) required_schedule: ScheduleConstraint,
    /// Resident resources whose lifetime participates in the evidence.
    pub(crate) required_resources: Vec<ResidentResource>,
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
    /// The compiled graph can schedule all reachable joins in dependency order
    /// so build/probe residency requirements are not violated by join nesting.
    TopologicalBuildProbeSchedule,
    /// One resident producer is shared by multiple consumers and each consumer
    /// edge preserves that producer identity.
    SharedResidentSubplan,
    /// A scalar aggregate can consume a shared resident producer only when every
    /// aggregate expression derives from columns produced by that same shared
    /// node after transparent projections are peeled. An expression computed
    /// from those columns, in the aggregate or in a peeled projection, derives
    /// from them; a literal argument such as the `1` of `COUNT(*)` reads none.
    AggregateDerivesFromSharedProducer,
    /// A final ordering barrier consumes the resident input after all upstream
    /// producers are complete and does not feed back into its input.
    ///
    /// Capability analysis does not emit this kind; the compiled pipeline's
    /// barrier dominance validation owns the property. The variant stays so
    /// the report vocabulary keeps its name.
    FinalBarrierOrder,
    /// A projection wrapper can be ignored for residency only when both the
    /// peeled producer and the projection output are resident frames.
    ProjectionTransparentResidency,
    /// A filter/projection chain preserves residency only when every node in
    /// the connected unary chain consumes the previous resident output and also
    /// produces a resident frame.
    ResidentUnaryChainPreservesResidency,
    /// The residual predicate on the join references both join inputs, so the
    /// residual is a join condition. A residual that reads one input is a
    /// single-side filter.
    ResidualUsesBothInputs {
        /// Join whose residual predicate is being checked.
        join_id: NodeId,
    },
    /// Every reachable resident producer remains valid for all of its admitted
    /// consumers, and no rejected producer-consumer edge is hidden by the
    /// broader compiled subgraph admission.
    ResidentLifetimeDominatesConsumers,
}

/// Scheduling evidence required by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ScheduleConstraint {
    /// No non-local scheduling constraint is required.
    None,
    /// Build-side work must complete before probe-side work.
    BuildBeforeProbe,
    /// Nodes must be schedulable in topological dependency order.
    Topological,
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
    pub(crate) const fn as_str(self) -> &'static str {
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

/// Capability analysis for one lowered native query plan.
///
/// The node, edge, and invariant vectors are the evidence surface that the
/// compiled task and resource contracts consume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    /// Node-local executable facts.
    pub nodes: Vec<NodeCapability>,
    /// Producer-consumer edge facts.
    pub edges: Vec<EdgeContract>,
    /// Non-local scheduling, lifetime, and shared-resource facts.
    pub invariants: Vec<SubgraphInvariant>,
    /// Execution policy the plan was analyzed under; compile and the compiled
    /// contracts read the join strategy thresholds from it.
    pub(crate) policy: PlanningPolicy,
}

impl CapabilityAnalysis {
    /// Returns true when any node, edge, or invariant has a local rejection.
    pub fn has_rejections(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
            || self.edges.iter().any(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
            || self.invariants.iter().any(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
    }

    /// The first local rejection of a node, edge, or invariant.
    ///
    /// Nodes are searched before edges and edges before invariants, each in
    /// plan order, so the most local owner is reported. A rejected node often
    /// makes a dependent invariant fail as well; reporting the node names the
    /// operator, type, or expression that blocked the candidate, and planning
    /// reports show that node as the blocking element.
    pub fn first_rejection(&self) -> Option<Rejection> {
        self.nodes
            .iter()
            .find_map(|node| {
                Some(Rejection {
                    owner: RejectionOwner::Node(node.node_id),
                    reason: node.executable.rejection_reason()?,
                    facts: node.executable.rejection_facts()?.clone(),
                })
            })
            .or_else(|| {
                self.edges.iter().find_map(|edge| {
                    Some(Rejection {
                        owner: RejectionOwner::Edge { producer: edge.producer, consumer: edge.consumer },
                        reason: edge.decision.rejection_reason()?,
                        facts: edge.decision.rejection_facts()?.clone(),
                    })
                })
            })
            .or_else(|| {
                self.invariants.iter().find_map(|invariant| {
                    Some(Rejection {
                        owner: RejectionOwner::Invariant(invariant.invariant_id),
                        reason: invariant.decision.rejection_reason()?,
                        facts: invariant.decision.rejection_facts()?.clone(),
                    })
                })
            })
    }
}
