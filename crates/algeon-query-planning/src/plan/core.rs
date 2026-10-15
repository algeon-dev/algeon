//! Core query-plan IR types: [`QueryPlan`], [`QueryNode`], [`NodeId`], and
//! [`PlanNode`]. The DAG is append-only and ends at an explicit
//! [`ResultSink`]; native execution is GPU-resident until the sink chooses
//! otherwise.
use std::{collections::BTreeMap, fmt, sync::Arc};

use super::{
    AggregateNode, EdgeNormalizeNode, FilterNode, GraphAlgorithmNode, JoinNode, LimitNode, MlAlgorithmNode, NodeRemap,
    PlanFacts, ProjectionNode, ResultSink, SchemaRef, SortNode, SortSpec, SourceNode, UnionNode, VectorAlgorithmNode,
    WindowNode, compact_reachable_nodes,
    graph::{check_node_order, validated_plan_parts},
};
use crate::{
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterProducer, DynamicFilterRoute},
    error::{QueryError, QueryErrorCode, Result},
};

/// Query plan consumed by the native cuDF execution engine.
///
/// A plan is an append-only DAG of [`QueryNode`]s. [`NodeId`] values are indexes
/// into `nodes` and are stable once emitted by [`crate::plan::QueryPlanBuilder`]; callers must
/// not infer any execution scheduling guarantee beyond the dependency edges
/// recorded by each node.
///
/// Every plan value holds these invariants, checked when it is built: each
/// node id equals its position, each input precedes its consumer, the sink
/// reaches every node, and each node's output schema has one source (see
/// [`PlanFacts`]).
///
/// The plan always ends at an explicit [`ResultSink`]. Native execution keeps
/// intermediate and final data GPU-resident by default. Host materialization is
/// a sink choice; reaching the root does not perform it.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    pub(super) nodes: Vec<QueryNode>,
    pub(super) sink: ResultSink,
    pub(super) annotations: Annotations,
}

/// Facts bound to node ids that follow the plan through
/// [`QueryPlan::rewrite`] without taking part in execution.
///
/// Rules never edit these directly. A rule returns nodes only, and `rewrite`
/// decides which bindings survive from how the rule described its output
/// (see `RewriteOrigin`). The execution fingerprint excludes advisory
/// bindings, so dropping or invalidating one changes no plan identity.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Annotations {
    /// Dynamic-filter producer/consumer routes between plan nodes.
    pub(crate) dynamic_filter_routes: Vec<DynamicFilterRoute>,
    /// Frontend volume signals for plan output occurrences.
    pub(crate) advisory: Vec<super::AdvisoryOutputBinding>,
}

/// One optimizer rule's output: the new node list and how its nodes relate to
/// the nodes of the plan it rewrote.
pub(crate) struct RewrittenNodes {
    nodes: Vec<QueryNode>,
    root: NodeId,
    origin: RewriteOrigin,
}

/// How a rule's output nodes relate to the nodes it rewrote. `rewrite` reads
/// this to decide which side state it can carry over without re-deriving it.
enum RewriteOrigin {
    /// Every source node kept its id; new nodes may be appended.
    InPlace,
    /// Every source node kept its id and operator; only what source nodes
    /// read (row-group selection, reader predicate) was narrowed.
    RefinedSources,
    /// Source node id to the id of its image in the new list.
    Rebuilt(NodeRemap),
}

impl RewrittenNodes {
    /// Source nodes of a copy of the source node list narrowed to read fewer
    /// rows. The plan keeps its structure, so dynamic-filter routes keep their
    /// endpoints.
    pub(crate) fn refined_sources(nodes: Vec<QueryNode>, root: NodeId) -> Self {
        Self { nodes, root, origin: RewriteOrigin::RefinedSources }
    }

    /// Nodes edited in place in a copy of the source node list.
    pub(crate) fn in_place(nodes: Vec<QueryNode>, root: NodeId) -> Self {
        Self { nodes, root, origin: RewriteOrigin::InPlace }
    }

    /// Nodes rebuilt into a new list; `origin` maps each carried-over source
    /// node to its image. A source node without an entry has no image, and
    /// source nodes that share one image (a merge) have none either.
    pub(crate) fn rebuilt(nodes: Vec<QueryNode>, root: NodeId, origin: BTreeMap<NodeId, NodeId>) -> Self {
        let mut sources_per_image = BTreeMap::<NodeId, usize>::new();
        for image in origin.values() {
            *sources_per_image.entry(*image).or_default() += 1;
        }
        // Two sources mapped to one image were merged. The merged node stands
        // for neither source alone, so a route endpoint or advisory bound to
        // one of them must not move onto it; without an entry, `rewrite`
        // discards that state.
        let origin = origin.into_iter().filter(|(_, image)| sources_per_image[image] == 1).collect();
        Self { nodes, root, origin: RewriteOrigin::Rebuilt(NodeRemap::new(origin)) }
    }
}

impl QueryPlan {
    /// Validates plan invariants and derives output schemas for all nodes.
    pub(crate) fn from_parts(nodes: Vec<QueryNode>, sink: ResultSink, annotations: Annotations) -> Result<Self> {
        let nodes = validated_plan_parts(nodes, &sink)?;
        Ok(Self { nodes, sink, annotations })
    }

    /// Returns the node consumed by the plan sink.
    pub fn root_id(&self) -> NodeId {
        self.sink.input()
    }

    /// Returns the root node.
    pub fn root_node(&self) -> &QueryNode {
        &self.nodes[self.root_id().index()]
    }

    /// Returns the explicit execution boundary for this plan.
    pub fn sink(&self) -> &ResultSink {
        &self.sink
    }

    /// Dynamic-filter producer/consumer routes attached to this plan.
    pub fn dynamic_filter_routes(&self) -> &[DynamicFilterRoute] {
        &self.annotations.dynamic_filter_routes
    }

    pub(crate) fn with_dynamic_filter_routes(mut self, routes: Vec<DynamicFilterRoute>) -> Self {
        self.annotations.dynamic_filter_routes = routes;
        self
    }

    /// Returns this plan with a replacement dynamic-filter route list, for
    /// tests that state routes without running route discovery.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn with_dynamic_filter_routes_for_test(self, routes: Vec<DynamicFilterRoute>) -> Self {
        self.with_dynamic_filter_routes(routes)
    }

    /// Produces a rewritten plan rooted at `new_root` with unreachable nodes pruned.
    ///
    /// Nodes unreachable from `new_root` are pruned and remaining nodes renumbered.
    /// Side state follows the composed source-to-result node mapping: a
    /// dynamic-filter route survives only when every endpoint maps to an
    /// unchanged node (route discovery runs late and finds rewritten ones
    /// again), except that narrowing what a source reads keeps its routes; an
    /// advisory binding moves with its node and is invalidated when that node
    /// or any dependency changed. The sink keeps its kind and host contract at
    /// the new root.
    ///
    /// The rewritten node list is checked before compaction reads it by id,
    /// and the result is checked like any other plan.
    pub(crate) fn rewrite(&self, rewritten: RewrittenNodes) -> Result<Self> {
        let RewrittenNodes { nodes, root, origin } = rewritten;
        let sink = |root| self.sink.with_input(root);
        match origin {
            RewriteOrigin::InPlace => self.rebuilt(nodes, root, &NodeRemap::identity(self.node_count()), sink),
            RewriteOrigin::RefinedSources => {
                debug_assert_eq!(nodes.len(), self.node_count(), "refining sources keeps every node");
                let mut refined = self.rebuilt(nodes, root, &NodeRemap::identity(self.node_count()), sink)?;
                refined.annotations.dynamic_filter_routes.clone_from(&self.annotations.dynamic_filter_routes);
                Ok(refined)
            },
            RewriteOrigin::Rebuilt(origin) => self.rebuilt(nodes, root, &origin, sink),
        }
    }

    /// The subplan rooted at `root`, ending at a GPU-resident sink, with the
    /// side state of the nodes it keeps.
    pub(crate) fn resident_subplan(&self, root: NodeId) -> Result<Self> {
        let origin = NodeRemap::identity(self.node_count());
        self.rebuilt(self.nodes.clone(), root, &origin, |input| ResultSink::GpuResident { input })
    }

    /// Shared tail of every rewrite: checks the new node list, compacts it to
    /// the nodes `root` reaches, constructs the plan through
    /// [`Self::from_parts`], and carries side state across `origin` composed
    /// with the compaction. `check_node_order` runs first because compaction
    /// indexes the list by id and would follow the wrong node for an id that
    /// is out of position.
    fn rebuilt(
        &self,
        nodes: Vec<QueryNode>,
        root: NodeId,
        origin: &NodeRemap,
        sink: impl FnOnce(NodeId) -> ResultSink,
    ) -> Result<Self> {
        check_node_order(&nodes, root)?;
        let (nodes, root, compaction) = compact_reachable_nodes(nodes, root);
        let remap = origin.then(&compaction);
        let mut rewritten = Self::from_parts(nodes, sink(root), Annotations::default())?;
        rewritten.annotations.dynamic_filter_routes = self
            .annotations
            .dynamic_filter_routes
            .iter()
            .filter_map(|route| {
                let producer = remap_route_producer(&route.producer, &remap, self, &rewritten)?;
                let consumer = remap_route_consumer(&route.consumer, &remap, self, &rewritten)?;
                Some(DynamicFilterRoute { producer, consumer, ..route.clone() })
            })
            .collect();
        rewritten.annotations.advisory = self.remapped_advisory_outputs(&rewritten, &remap);
        Ok(rewritten)
    }

    /// Moves each advisory binding to its node's image and invalidates it
    /// when that node or any dependency changed. A binding whose node has no
    /// image is dropped.
    fn remapped_advisory_outputs(&self, rewritten: &Self, remap: &NodeRemap) -> Vec<super::AdvisoryOutputBinding> {
        let mut unchanged = vec![false; rewritten.node_count()];
        for node in rewritten.nodes() {
            unchanged[node.id.index()] =
                remap.old_id(node.id).and_then(|old_id| self.node(old_id)).is_some_and(|old_node| {
                    remap.try_translate_inputs(&old_node.kind).as_ref() == Some(&node.kind)
                        && old_node.facts == node.facts
                        && node.input_ids().iter().all(|input| unchanged[input.index()])
                });
        }
        self.annotations
            .advisory
            .iter()
            .filter_map(|binding| {
                let node = rewritten.node(remap.remap(binding.node)?)?;
                let mut binding = binding.clone();
                binding.node = node.id;
                if !unchanged[node.id.index()] {
                    binding.invalidate(super::AdvisoryUnknownReason::RewrittenSubtree);
                    binding.schema = Arc::clone(node.output_schema());
                }
                Some(binding)
            })
            .collect()
    }

    /// Replaces this plan's nodes with an in-place edit of them, keeping its
    /// side state through [`Self::rewrite`].
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn with_nodes_for_test(&self, nodes: Vec<QueryNode>) -> Result<Self> {
        self.rewrite(RewrittenNodes::in_place(nodes, self.root_id()))
    }

    /// Return a copy of this plan with the same root and a GPU-resident sink.
    ///
    /// This is used by embedding frontends that need to feed a native plan's
    /// output into another GPU consumer. The existing host-arrow boundary is
    /// replaced by the GPU-resident sink.
    pub fn clone_with_gpu_resident_sink(&self) -> Self {
        let mut plan = self.clone();
        plan.sink = ResultSink::GpuResident { input: self.root_id() };
        plan
    }

    /// Return a copy that executes to a GPU-resident result while keeping the
    /// host sink as the caller-visible result contract
    /// ([`ResultSink::DeferredHostArrow`]).
    ///
    /// Embedding frontends use this when they implement bounded incremental
    /// device-to-host materialization themselves. Keeping this distinct from
    /// [`Self::clone_with_gpu_resident_sink`] prevents a GPU-to-GPU
    /// consumer from acquiring a host-materialization contract it does not
    /// own.
    pub fn clone_with_deferred_host_materialization_sink(&self) -> Self {
        let mut plan = self.clone();
        if let ResultSink::HostArrow { input, schema } = &plan.sink {
            plan.sink = ResultSink::DeferredHostArrow { input: *input, schema: Arc::clone(schema) };
        }
        plan
    }

    /// Replace the externally materialized host schema while preserving the
    /// executable root. This keeps Arrow schema metadata owned by an embedding
    /// frontend at the host boundary without injecting frontend state into the
    /// native IR nodes.
    ///
    /// The replacement must retain the exact output fields and types of the
    /// host sink schema it replaces. Only schema-level metadata may differ.
    /// The host sink schema itself may differ from the executable root schema
    /// by field names (see
    /// [`QueryPlanBuilder::finish_with_host_sink_schema`]), so the executable
    /// root is not a valid reference for this check.
    ///
    /// [`QueryPlanBuilder::finish_with_host_sink_schema`]: crate::plan::QueryPlanBuilder::finish_with_host_sink_schema
    pub fn with_host_sink_schema(mut self, schema: SchemaRef) -> Result<Self> {
        match &mut self.sink {
            ResultSink::HostArrow { schema: host_schema, .. }
            | ResultSink::DeferredHostArrow { schema: host_schema, .. } => {
                if host_schema.fields() != schema.fields() {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.plan.host_sink_schema",
                        "host sink schema replacement must preserve the existing host sink fields",
                    ));
                }
                *host_schema = schema;
            },
            ResultSink::GpuResident { .. } => {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.plan.host_sink_schema",
                    "a GPU-resident plan has no host sink schema to replace",
                ));
            },
        }
        Ok(self)
    }

    /// Returns this plan with a native sort appended above the current root.
    ///
    /// The existing DAG is preserved and the result sink is retargeted to the
    /// new root. This is used by frontends that need to make an ordering
    /// contract explicit after lowering an order-breaking operator.
    pub fn with_appended_sort(mut self, spec: SortSpec) -> Self {
        let input = self.root_id();
        let output_schema = Arc::clone(self.root_node().output_schema());
        let root = NodeId(self.nodes.len());
        // Appending keeps every invariant `from_parts` checks: the new id is
        // its position, its only input precedes it, and the retargeted sink
        // reaches it and, through it, everything the old root reached. The
        // node is pushed without re-validating the plan for that reason.
        let output_ordering = spec.expressions.clone();
        self.nodes.push(QueryNode {
            id: root,
            kind: PlanNode::Sort(SortNode { input, spec }),
            facts: PlanFacts::new(output_schema).with_output_ordering(output_ordering),
        });
        self.sink = self.sink.with_input(root);
        self
    }

    /// Frontend volume signals scoped to this plan's native output occurrences.
    pub fn advisory_outputs(&self) -> &[super::AdvisoryOutputBinding] {
        &self.annotations.advisory
    }

    /// Returns all nodes in append order.
    ///
    /// The slice order is also the [`NodeId::index`] address space. Existing
    /// node positions are part of the plan contract and must remain stable.
    pub fn nodes(&self) -> &[QueryNode] {
        &self.nodes
    }

    /// Returns the node for `id` when it is in range.
    ///
    /// [`NodeId`] is a plan-local index: an id from another plan that happens
    /// to be in range returns this plan's node at that index. Callers that
    /// carry ids across plans bind them to the plan's scope and fingerprint.
    pub fn node(&self, id: NodeId) -> Option<&QueryNode> {
        self.nodes.get(id.index())
    }

    /// Returns the schema visible at the plan boundary.
    ///
    /// GPU-resident and deferred host sinks derive this from the sink input. A
    /// host Arrow sink carries an explicit schema because materialization is a
    /// boundary contract.
    pub fn output_schema(&self) -> &SchemaRef {
        match &self.sink {
            ResultSink::GpuResident { input } | ResultSink::DeferredHostArrow { input, .. } => {
                self.nodes[input.index()].output_schema()
            },
            ResultSink::HostArrow { schema, .. } => schema,
        }
    }

    /// Returns the schema the caller receives: the host schema of a host or
    /// deferred host sink, otherwise the GPU-resident output schema.
    pub fn caller_visible_output_schema(&self) -> &SchemaRef {
        self.sink.caller_visible_host_schema().unwrap_or_else(|| self.output_schema())
    }

    /// Number of entries in this plan's node arena.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether any node in the arena is a join.
    pub fn contains_join(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::Join(_)))
    }

    pub(crate) fn contains_graph_algorithm(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_)))
    }

    /// Whether any node performs graph edge normalization.
    pub fn contains_edge_normalize(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::EdgeNormalize(_)))
    }

    /// Whether this plan contains a vector algorithm node.
    pub fn contains_vector_algorithm(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::VectorAlgorithm(_)))
    }

    /// Whether any node executes a machine-learning algorithm.
    pub fn contains_ml_algorithm(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::MlAlgorithm(_)))
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
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// Returns the stable index into [`QueryPlan::nodes`].
    pub const fn index(self) -> usize {
        self.0
    }

    /// A raw id for tests that need an id without building a plan.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub const fn from_index_for_test(index: usize) -> Self {
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
/// output. Facts are descriptive contracts used by validators and diagnostics,
/// and they do not change how execution runs.
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
        &self.facts.output_schema
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
    /// Groups and aggregates rows according to an [`crate::expr::AggregateSpec`].
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
    /// Runs a cuML operation on dense relation inputs.
    MlAlgorithm(MlAlgorithmNode),
}

impl PlanNode {
    /// Returns input node ids in the semantic order used by this operation.
    ///
    /// Joins always return `[left, right]`; graph algorithms return the primary
    /// edge-list input followed by side inputs; unary operators return their
    /// single input; sources return an empty slice.
    pub(crate) fn input_ids(&self) -> &[NodeId] {
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
            Self::MlAlgorithm(node) => &node.inputs,
        }
    }
}

/// Translates one route endpoint through a renumbering rewrite.
///
/// Returns the new id only when the node is kept and still matches its source
/// occurrence; anything else drops the route.
fn remapped_route_endpoint(
    old: NodeId,
    remap: &NodeRemap,
    source: &QueryPlan,
    rewritten: &QueryPlan,
) -> Option<NodeId> {
    let new = remap.remap(old)?;
    let (Some(old_node), Some(new_node)) = (source.node(old), rewritten.node(new)) else {
        return None;
    };
    (remap.try_translate_inputs(&old_node.kind).as_ref() == Some(&new_node.kind) && old_node.facts == new_node.facts)
        .then_some(new)
}

fn remap_route_producer(
    producer: &DynamicFilterProducer,
    remap: &NodeRemap,
    source: &QueryPlan,
    rewritten: &QueryPlan,
) -> Option<DynamicFilterProducer> {
    match producer {
        DynamicFilterProducer::HashJoinBuild { join_node_id, build_side, build_input } => {
            Some(DynamicFilterProducer::HashJoinBuild {
                join_node_id: remapped_route_endpoint(*join_node_id, remap, source, rewritten)?,
                build_side: *build_side,
                build_input: remapped_route_endpoint(*build_input, remap, source, rewritten)?,
            })
        },
        DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } => {
            Some(DynamicFilterProducer::AggregateGroupKeys {
                aggregate_node_id: remapped_route_endpoint(*aggregate_node_id, remap, source, rewritten)?,
            })
        },
    }
}

fn remap_route_consumer(
    consumer: &DynamicFilterConsumer,
    remap: &NodeRemap,
    source: &QueryPlan,
    rewritten: &QueryPlan,
) -> Option<DynamicFilterConsumer> {
    match consumer {
        DynamicFilterConsumer::LocalParquetSource { source_node_id } => {
            Some(DynamicFilterConsumer::LocalParquetSource {
                source_node_id: remapped_route_endpoint(*source_node_id, remap, source, rewritten)?,
            })
        },
        DynamicFilterConsumer::HashJoinProbe { join_node_id, probe_side } => {
            Some(DynamicFilterConsumer::HashJoinProbe {
                join_node_id: remapped_route_endpoint(*join_node_id, remap, source, rewritten)?,
                probe_side: *probe_side,
            })
        },
        DynamicFilterConsumer::Filter { filter_node_id } => Some(DynamicFilterConsumer::Filter {
            filter_node_id: remapped_route_endpoint(*filter_node_id, remap, source, rewritten)?,
        }),
    }
}
