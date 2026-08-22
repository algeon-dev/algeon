//! Core query-plan IR types: [`QueryPlan`], [`QueryNode`], [`NodeId`], and
//! [`PlanNode`]. The DAG is append-only and ends at an explicit
//! [`ResultSink`]; native execution is GPU-resident until the sink chooses
//! otherwise.

use super::*;
use crate::{
    capability::KeyedHashJoinStrategyOptions,
    dynamic_filter::{DynamicFilterExecutionOptions, DynamicFilterRoute},
};

/// DataFusion-free query plan consumed by the native cuDF execution engine.
///
/// A plan is an append-only DAG of [`QueryNode`]s. [`NodeId`] values are indexes
/// into `nodes` and are stable once emitted by [`QueryPlanBuilder`]; callers must
/// not infer any execution scheduling guarantee beyond the dependency edges
/// recorded by each node.
///
/// The plan always ends at an explicit [`ResultSink`]. Native execution keeps
/// intermediate and final data GPU-resident by default; host materialization is a
/// sink choice, not an implicit side effect of reaching the root.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    pub(super) nodes: Vec<QueryNode>,
    pub(super) root: NodeId,
    pub(super) sink: ResultSink,
    /// Caller-visible host boundary retained when an embedding frontend takes
    /// ownership of incremental host materialization after native execution.
    ///
    /// `sink` remains the physical GPU-resident execution boundary. This
    /// contract is consulted only when projecting caller-visible runtime
    /// rejection diagnostics, so a rejected plan does not silently lose its
    /// modeled host-materialization boundary merely because the copy is
    /// deferred to the frontend.
    pub(super) deferred_host_sink_schema: Option<SchemaRef>,
    pub(super) dynamic_filter_routes: Vec<DynamicFilterRoute>,
    pub(super) dynamic_filter_execution_options: DynamicFilterExecutionOptions,
    pub(super) keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions,
}

impl QueryPlan {
    /// Rebuilds a plan from already-validated nodes after an internal IR rewrite.
    ///
    /// Optimizers use this instead of [`QueryPlanBuilder`] when they need to
    /// clone an existing DAG while preserving node facts from lowered IR.
    #[doc(hidden)]
    pub fn from_rewritten_parts(nodes: Vec<QueryNode>, root: NodeId, sink: ResultSink) -> Self {
        assert!(nodes.get(root.index()).is_some(), "query plan root node must exist");
        assert_eq!(sink.input(), root, "query plan sink must consume the root node");
        Self {
            nodes,
            root,
            sink,
            deferred_host_sink_schema: None,
            dynamic_filter_routes: Vec::new(),
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default(),
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default(),
        }
    }

    /// Returns the node consumed by the plan sink.
    pub fn root_id(&self) -> NodeId {
        self.root
    }

    /// Returns the root node.
    ///
    /// Plans built through [`QueryPlanBuilder`] guarantee that the root exists.
    pub fn root_node(&self) -> &QueryNode {
        self.node(self.root).expect("query plan root node must exist")
    }

    /// Returns the explicit execution boundary for this plan.
    pub fn sink(&self) -> &ResultSink {
        &self.sink
    }

    /// Dynamic-filter producer/consumer routes attached to this plan.
    pub fn dynamic_filter_routes(&self) -> &[DynamicFilterRoute] {
        &self.dynamic_filter_routes
    }

    /// Dynamic-filter execution policy captured when routes were discovered.
    pub fn dynamic_filter_execution_options(&self) -> DynamicFilterExecutionOptions {
        self.dynamic_filter_execution_options
    }

    /// Keyed-hash join strategy policy used by capability admission and runtime
    /// strategy selection.
    pub fn keyed_hash_join_strategy_options(&self) -> KeyedHashJoinStrategyOptions {
        self.keyed_hash_join_strategy_options
    }

    /// Returns this plan with a replacement dynamic-filter route list.
    #[doc(hidden)]
    pub fn with_dynamic_filter_routes(mut self, routes: Vec<DynamicFilterRoute>) -> Self {
        self.dynamic_filter_routes = routes;
        self
    }

    /// Returns this plan with replacement dynamic-filter execution options.
    #[doc(hidden)]
    pub fn with_dynamic_filter_execution_options(mut self, options: DynamicFilterExecutionOptions) -> Self {
        self.dynamic_filter_execution_options = options;
        self
    }

    /// Returns this plan with replacement keyed-hash join strategy options.
    #[doc(hidden)]
    pub fn with_keyed_hash_join_strategy_options(mut self, options: KeyedHashJoinStrategyOptions) -> Self {
        self.keyed_hash_join_strategy_options = options;
        self
    }

    /// Returns this plan with dynamic-filter routes and execution options copied
    /// from another plan.
    #[doc(hidden)]
    pub fn with_dynamic_filter_state_from(mut self, source: &Self) -> Self {
        self.dynamic_filter_routes = source.dynamic_filter_routes.clone();
        self.dynamic_filter_execution_options = source.dynamic_filter_execution_options;
        self
    }

    /// Returns this plan with rewrite-preserved execution policy state copied
    /// from another plan.
    #[doc(hidden)]
    pub fn with_rewrite_state_from(self, source: &Self) -> Self {
        let mut rewritten = self
            .with_dynamic_filter_state_from(source)
            .with_keyed_hash_join_strategy_options(source.keyed_hash_join_strategy_options);
        rewritten.deferred_host_sink_schema = source.deferred_host_sink_schema.clone();
        rewritten
    }

    /// Returns this plan with keyed-hash join strategy options copied from
    /// another plan.
    #[doc(hidden)]
    pub fn with_keyed_hash_join_strategy_options_from(self, source: &Self) -> Self {
        self.with_keyed_hash_join_strategy_options(source.keyed_hash_join_strategy_options)
    }

    /// Return a copy of this plan with the same root and a GPU-resident sink.
    ///
    /// This is used by embedding frontends that need to feed a native plan's
    /// output into another GPU consumer instead of materializing the existing
    /// host-arrow boundary.
    pub fn clone_with_gpu_resident_sink(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            root: self.root,
            sink: ResultSink::GpuResident { input: self.root },
            deferred_host_sink_schema: None,
            dynamic_filter_routes: self.dynamic_filter_routes.clone(),
            dynamic_filter_execution_options: self.dynamic_filter_execution_options,
            keyed_hash_join_strategy_options: self.keyed_hash_join_strategy_options,
        }
    }

    /// Return a copy that executes to a GPU-resident result while retaining
    /// the existing host sink as the caller-visible result contract.
    ///
    /// Embedding frontends use this when they implement bounded incremental
    /// device-to-host materialization themselves. Keeping this distinct from
    /// [`Self::clone_with_gpu_resident_sink`] prevents a true GPU-to-GPU
    /// consumer from acquiring a host-materialization contract it does not
    /// own.
    pub fn clone_with_deferred_host_materialization_sink(&self) -> Self {
        let deferred_host_sink_schema = match (&self.deferred_host_sink_schema, &self.sink) {
            (Some(schema), _) | (None, ResultSink::HostArrow { schema, .. }) => Some(Arc::clone(schema)),
            (None, ResultSink::GpuResident { .. }) => None,
        };
        Self {
            nodes: self.nodes.clone(),
            root: self.root,
            sink: ResultSink::GpuResident { input: self.root },
            deferred_host_sink_schema,
            dynamic_filter_routes: self.dynamic_filter_routes.clone(),
            dynamic_filter_execution_options: self.dynamic_filter_execution_options,
            keyed_hash_join_strategy_options: self.keyed_hash_join_strategy_options,
        }
    }

    /// Reconstruct the caller-visible sink view used by runtime rejection
    /// diagnostics. Successful execution continues to use the physical sink.
    #[doc(hidden)]
    pub fn clone_with_caller_visible_sink_contract(&self) -> Self {
        let mut plan = self.clone();
        if let Some(schema) = plan.deferred_host_sink_schema.take() {
            plan.sink = ResultSink::HostArrow { input: plan.root, schema };
        }
        plan
    }

    /// Replace the externally materialized host schema while preserving the
    /// executable root. This keeps Arrow schema metadata owned by an embedding
    /// frontend at the host boundary without injecting frontend state into the
    /// DataFusion-independent IR nodes.
    ///
    /// The replacement must retain the exact output fields and types of the
    /// host sink schema it replaces. Only schema-level metadata may differ.
    /// The host sink schema itself may differ from the executable root schema
    /// by field names (see
    /// [`QueryPlanBuilder::finish_with_host_sink_schema`]), so the executable
    /// root is not a valid reference for this check.
    ///
    /// [`QueryPlanBuilder::finish_with_host_sink_schema`]: crate::plan::QueryPlanBuilder::finish_with_host_sink_schema
    pub fn with_host_sink_schema(mut self, schema: SchemaRef) -> Self {
        match &mut self.sink {
            ResultSink::HostArrow { schema: host_schema, .. } => {
                assert_eq!(
                    host_schema.fields(),
                    schema.fields(),
                    "host sink schema replacement must preserve the existing host sink fields"
                );
                *host_schema = schema;
            },
            ResultSink::GpuResident { .. } => {
                panic!("cannot replace the host sink schema of a GPU-resident plan")
            },
        }
        self
    }

    /// Returns this plan with a native sort appended above the current root.
    ///
    /// The existing DAG is preserved and the result sink is retargeted to the
    /// new root. This is used by frontends that need to make an ordering
    /// contract explicit after lowering an order-breaking operator.
    pub fn with_appended_sort(mut self, spec: SortSpec) -> Self {
        let input = self.root;
        let output_schema = Arc::clone(self.root_node().output_schema());
        let root = NodeId(self.nodes.len());
        let output_ordering = spec.expressions.clone();
        self.nodes.push(QueryNode {
            id: root,
            kind: PlanNode::Sort(SortNode { input, spec }),
            facts: PlanFacts::device(output_schema, SchemaBehavior::PreservesInput)
                .with_output_ordering(output_ordering),
        });
        self.root = root;
        self.sink = match self.sink {
            ResultSink::GpuResident { .. } => ResultSink::GpuResident { input: root },
            ResultSink::HostArrow { schema, .. } => ResultSink::HostArrow { input: root, schema },
        };
        self
    }

    /// Returns all nodes in append order.
    ///
    /// The slice order is also the [`NodeId::index`] address space. Existing
    /// node positions are part of the plan contract and must remain stable.
    pub fn nodes(&self) -> &[QueryNode] {
        &self.nodes
    }

    /// Returns the node for `id` when it is still present at that stable index.
    pub fn node(&self, id: NodeId) -> Option<&QueryNode> {
        self.nodes.get(id.index()).filter(|node| node.id == id)
    }

    /// Returns the schema visible at the plan boundary.
    ///
    /// GPU-resident plans derive this from the sink input. Host sinks carry an
    /// explicit schema because materialization is a boundary contract.
    pub fn output_schema(&self) -> &SchemaRef {
        match &self.sink {
            ResultSink::GpuResident { input } => {
                self.node(*input).expect("query plan sink input must exist").output_schema()
            },
            ResultSink::HostArrow { schema, .. } => schema,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains_join(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::Join(_)))
    }

    pub fn contains_graph_algorithm(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_)))
    }

    pub fn contains_edge_normalize(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::EdgeNormalize(_)))
    }

    /// Whether this plan contains a vector algorithm node.
    pub fn contains_vector_algorithm(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::VectorAlgorithm(_)))
    }

    /// Returns nodes that are not reachable from the explicit sink input.
    ///
    /// Builders normally produce sink-reachable plans, but this is useful for
    /// validation and diagnostics when composing partial DAGs.
    pub fn unreachable_node_ids(&self) -> Vec<NodeId> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![self.sink.input()];
        while let Some(id) = stack.pop() {
            let Some(node) = self.node(id) else {
                continue;
            };
            if reachable[id.index()] {
                continue;
            }
            reachable[id.index()] = true;
            stack.extend(node.input_ids().iter().copied());
        }
        self.nodes.iter().filter(|node| !reachable[node.id.index()]).map(|node| node.id).collect()
    }
}

/// Stable identifier for a node inside one [`QueryPlan`].
///
/// `NodeId` is an index into [`QueryPlan::nodes`]. It is intentionally small and
/// copyable, but it is only meaningful for the plan that emitted it. Once a
/// builder returns an id, the referenced node must keep that index for the
/// lifetime of the plan so expressions, join inputs, sinks, and diagnostics can
/// refer to it durably.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(super) usize);

impl NodeId {
    /// Returns the stable index into [`QueryPlan::nodes`].
    pub const fn index(self) -> usize {
        self.0
    }

    /// Builds a `NodeId` from a raw index for crate-internal tests and validators.
    ///
    /// Frontends should obtain ids from [`QueryPlanBuilder`] so existence and
    /// append-order invariants are checked as the plan is constructed.
    #[doc(hidden)]
    pub const fn from_index_for_internal_use(index: usize) -> Self {
        Self(index)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// One node in the native query DAG.
///
/// The node id records its stable address, `kind` records the operation and
/// input edges, and `facts` records planner/executor facts about the node's
/// output. Facts are descriptive contracts used by validators and diagnostics;
/// they are not hidden knobs that rewrite execution behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryNode {
    /// Stable id whose index matches this node's position in [`QueryPlan::nodes`].
    pub id: NodeId,
    /// Operation represented by this node.
    pub kind: PlanNode,
    /// Output facts established when the node is lowered into native IR.
    pub facts: PlanFacts,
}

impl QueryNode {
    /// Returns the schema produced by this node.
    pub fn output_schema(&self) -> &SchemaRef {
        match &self.kind {
            PlanNode::Projection(projection) => projection.output_schema(),
            PlanNode::GraphAlgorithm(graph) => graph.output_schema(),
            PlanNode::VectorAlgorithm(vector) => vector.output_schema(),
            _ => &self.facts.output_schema,
        }
    }

    /// Returns input node ids referenced by this operation.
    pub fn input_ids(&self) -> &[NodeId] {
        self.kind.input_ids()
    }
}

/// Operation payload for a [`QueryNode`].
///
/// Variants describe native IR operations only; they do not carry DataFusion
/// execution plans. Input edges are represented by [`NodeId`]s so non-DataFusion
/// frontends can construct the same IR directly.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode {
    /// Reads data from a native source such as local Parquet or a GPU-resident
    /// input supplied by the caller.
    Source(Box<SourceNode>),
    /// Applies a boolean predicate without changing the input schema.
    Filter(FilterNode),
    /// Evaluates expressions and produces a projection schema.
    Projection(ProjectionNode),
    /// Groups and aggregates rows according to an [`AggregateSpec`].
    Aggregate(AggregateNode),
    /// Appends native window function outputs to each input row.
    Window(WindowNode),
    /// Establishes a full output ordering for the node.
    Sort(SortNode),
    /// Applies skip/fetch row limiting while preserving input ordering facts.
    Limit(LimitNode),
    /// Combines two inputs using separated equality keys, residual predicates,
    /// output projection, and build-side/cache choices.
    Join(JoinNode),
    /// Concatenates N schema-compatible inputs in order.
    Union(UnionNode),
    /// Canonicalizes an edge-list before cuGraph raw construction.
    EdgeNormalize(EdgeNormalizeNode),
    /// Runs a graph algorithm over an edge-list input.
    GraphAlgorithm(GraphAlgorithmNode),
    /// Runs a vector search or vector-ML algorithm over relation inputs.
    VectorAlgorithm(VectorAlgorithmNode),
}

impl PlanNode {
    /// Returns input node ids in the semantic order used by this operation.
    ///
    /// Joins always return `[left, right]`; graph algorithms return the primary
    /// edge-list input followed by side inputs; unary operators return their
    /// single input; sources return an empty slice.
    pub fn input_ids(&self) -> &[NodeId] {
        match self {
            Self::Source(_) => &[],
            Self::Filter(node) => std::slice::from_ref(&node.input),
            Self::Projection(node) => std::slice::from_ref(&node.input),
            Self::Aggregate(node) => std::slice::from_ref(&node.input),
            Self::Window(node) => std::slice::from_ref(&node.input),
            Self::Sort(node) => std::slice::from_ref(&node.input),
            Self::Limit(node) => std::slice::from_ref(&node.input),
            Self::Join(node) => &node.inputs,
            Self::Union(node) => &node.inputs,
            Self::EdgeNormalize(node) => std::slice::from_ref(&node.input),
            Self::GraphAlgorithm(node) => &node.inputs,
            Self::VectorAlgorithm(node) => &node.inputs,
        }
    }
}
