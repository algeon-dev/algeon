use super::*;

/// Append-only builder for native query plans.
///
/// The builder assigns [`NodeId`]s by appending to `nodes`. Methods validate that
/// referenced inputs already exist, which keeps ids stable and prevents forward
/// references. Frontends should use this builder instead of constructing
/// [`QueryPlan`] directly.
#[derive(Debug, Clone, Default)]
pub struct QueryPlanBuilder {
    nodes: Vec<QueryNode>,
}

impl QueryPlanBuilder {
    /// Creates an empty plan builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a source node and returns its stable id.
    ///
    /// Source nodes establish their output schema and currently default to
    /// device placement when lowered into this IR.
    pub fn add_source(&mut self, source: SourceNode) -> NodeId {
        let output_schema = source.output_schema().clone();
        self.push(PlanNode::Source(Box::new(source)), PlanFacts::device(output_schema, SchemaBehavior::SourceOutput))
    }

    /// Adds an order-preserving filter node.
    ///
    /// The caller supplies the output schema produced by lowering. It should
    /// match the input schema unless a future filter form explicitly changes it.
    pub fn add_filter(&mut self, input: NodeId, predicate: Expr, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = self.node(input).facts.output_ordering.clone();
        self.push(
            PlanNode::Filter(FilterNode { input, predicate }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        )
    }

    /// Adds a projection node from explicit typed outputs.
    pub fn add_projection(&mut self, input: NodeId, outputs: Vec<ProjectionOutput>) -> NodeId {
        self.assert_input_exists(input);
        let node = ProjectionNode::new(input, outputs);
        let output_schema = Arc::clone(node.output_schema());
        self.push(PlanNode::Projection(node), PlanFacts::device(output_schema, SchemaBehavior::Projection))
    }

    /// Adds an aggregate node.
    ///
    /// Aggregate semantics, including partial/final state representation, must
    /// be modeled explicitly in [`AggregateSpec`] and the expression IR before a
    /// frontend lowers them here.
    pub fn add_aggregate(&mut self, input: NodeId, spec: AggregateSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        self.push(
            PlanNode::Aggregate(AggregateNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::Aggregate),
        )
    }

    /// Adds a sort node and records its output ordering fact.
    pub fn add_sort(&mut self, input: NodeId, spec: SortSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = spec.expressions.clone();
        self.push(
            PlanNode::Sort(SortNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        )
    }

    /// Adds a limit node that preserves input ordering facts.
    pub fn add_limit(&mut self, input: NodeId, spec: LimitSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = self.node(input).facts.output_ordering.clone();
        self.push(
            PlanNode::Limit(LimitNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        )
    }

    /// Adds a join node with explicit left and right inputs.
    ///
    /// Equality keys, residual predicate, output projection, null matching, and
    /// build-side/cache behavior live in [`JoinSpec`] so frontends do not need to
    /// encode those contracts through expression conventions.
    pub fn add_join(&mut self, left: NodeId, right: NodeId, spec: JoinSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(left);
        self.assert_input_exists(right);
        self.push(
            PlanNode::Join(JoinNode { inputs: [left, right], spec }),
            PlanFacts::device(output_schema, SchemaBehavior::Join),
        )
    }

    /// Finishes the plan with a host Arrow materialization sink.
    pub fn finish_with_host_sink(self, root: NodeId) -> QueryPlan {
        let schema = self.node(root).output_schema().clone();
        self.finish(root, ResultSink::HostArrow { input: root, schema })
    }

    /// Finishes the plan with a host Arrow materialization sink whose external
    /// schema may differ from the executable root schema by field names.
    pub fn finish_with_host_sink_schema(self, root: NodeId, schema: SchemaRef) -> QueryPlan {
        self.finish(root, ResultSink::HostArrow { input: root, schema })
    }

    /// Finishes the plan with a GPU-resident result sink.
    pub fn finish_gpu_resident(self, root: NodeId) -> QueryPlan {
        self.finish(root, ResultSink::GpuResident { input: root })
    }

    fn finish(self, root: NodeId, sink: ResultSink) -> QueryPlan {
        assert!(self.nodes.get(root.index()).is_some(), "query plan root node must exist");
        assert_eq!(sink.input(), root, "query plan sink must consume the root node");
        QueryPlan { nodes: self.nodes, root, sink }
    }

    fn push(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }

    fn assert_input_exists(&self, input: NodeId) {
        assert!(self.nodes.get(input.index()).is_some(), "query plan input node must exist: {input}");
    }

    fn node(&self, id: NodeId) -> &QueryNode {
        self.nodes.get(id.index()).expect("query plan node must exist")
    }
}
