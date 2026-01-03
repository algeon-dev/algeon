use super::*;

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
}

impl QueryPlan {
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

    /// Returns the number of nodes in the append-only DAG.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true when the plan contains at least one native join node.
    pub fn contains_join(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.kind, PlanNode::Join(_)))
    }

    /// Returns true when the plan matches the bounded single-root join executor.
    pub fn has_bounded_executable_root_join(&self) -> bool {
        matches!(analyze_bounded_join_plan_support(self), BoundedJoinPlanSupport::RootJoin)
    }

    /// Returns true when the plan matches a supported bounded join pipeline.
    pub fn has_bounded_executable_join_pipeline(&self) -> bool {
        analyze_bounded_join_pipeline(self).is_ok()
    }

    /// Returns true when the plan matches the supported keyless conditional join
    /// input-subplan pipeline.
    pub fn has_bounded_executable_keyless_conditional_join_input_subplan_pipeline(&self) -> bool {
        analyze_bounded_keyless_conditional_join_input_subplan_pipeline(self).is_ok()
    }

    /// Returns true when the plan matches the supported aggregate-over-keyless
    /// conditional join subplan pipeline.
    pub fn has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(&self) -> bool {
        analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(self).is_ok()
    }

    /// Returns true when the plan matches the supported bounded multi-join pipeline.
    pub fn has_bounded_executable_multi_join_pipeline(&self) -> bool {
        crate::bounded_support::analyze_bounded_multi_join_pipeline(self).is_ok()
    }

    /// Returns true when the plan matches the supported keyed-residual join
    /// with a resident grouped local-Parquet aggregate input.
    pub fn has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(&self) -> bool {
        analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(self).is_ok()
    }

    /// Returns diagnostic facts for join plans rejected by every bounded join path.
    ///
    /// These facts are explanatory planning/execution diagnostics. They do not
    /// toggle executor behavior and should not be treated as optimizer hints.
    pub fn bounded_join_unsupported_shape_facts(
        &self,
        boundary_policy: ExecutionBoundaryPolicy,
    ) -> Option<BTreeMap<&'static str, String>> {
        if self.contains_join()
            && !self.has_bounded_executable_root_join()
            && !self.has_bounded_executable_join_pipeline()
            && !self.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline()
            && !self.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
            && !self.has_bounded_executable_multi_join_pipeline()
            && !self.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline()
        {
            Some(bounded_join_unsupported_shape_facts(self, boundary_policy))
        } else {
            None
        }
    }

    /// Returns modeled bounded-join facts for executable bounded join plans.
    pub fn bounded_join_modeled_facts(
        &self,
        boundary_policy: ExecutionBoundaryPolicy,
    ) -> Option<BTreeMap<&'static str, String>> {
        self.contains_join().then(|| bounded_join_modeled_facts(self, boundary_policy))
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
    pub(crate) const fn from_index_for_internal_use(index: usize) -> Self {
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
    /// Establishes a full output ordering for the node.
    Sort(SortNode),
    /// Applies skip/fetch row limiting while preserving input ordering facts.
    Limit(LimitNode),
    /// Combines two inputs using separated equality keys, residual predicates,
    /// output projection, and build-side/cache choices.
    Join(JoinNode),
}

impl PlanNode {
    /// Returns input node ids in the semantic order used by this operation.
    ///
    /// Joins always return `[left, right]`; unary operators return their single
    /// input; sources return an empty slice.
    pub fn input_ids(&self) -> &[NodeId] {
        match self {
            Self::Source(_) => &[],
            Self::Filter(node) => std::slice::from_ref(&node.input),
            Self::Projection(node) => std::slice::from_ref(&node.input),
            Self::Aggregate(node) => std::slice::from_ref(&node.input),
            Self::Sort(node) => std::slice::from_ref(&node.input),
            Self::Limit(node) => std::slice::from_ref(&node.input),
            Self::Join(node) => &node.inputs,
        }
    }
}
