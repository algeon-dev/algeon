//! Append-only builder for [`QueryPlan`].
//!
//! Frontends construct plans through [`QueryPlanBuilder`] so node ids stay
//! stable and forward references are impossible. Equivalent nodes (same kind
//! and facts) are interned as a small CSE step.
use std::sync::Arc;

use super::{
    AggregateNode, AggregateSpec, EdgeNormalizeNode, EdgeNormalizeSpec, Expr, Field, FilterNode, GraphAlgorithmNode,
    GraphAlgorithmSpec, GraphConstructionPolicy, JoinNode, JoinSpec, LimitNode, LimitSpec, NodeId, PlanFacts, PlanNode,
    ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, ResultSink, Schema, SchemaBehavior, SchemaRef, SortNode,
    SortSpec, SourceNode, UnionNode, VectorAlgorithmNode, VectorAlgorithmSpec, WindowNode, WindowSpec,
};
use crate::error::{QueryError, QueryErrorCode, Result};

/// Builder for native query plans.
///
/// The builder assigns stable [`NodeId`]s while interning equivalent nodes as a
/// local common-subexpression-elimination step. Methods validate that referenced
/// inputs already exist, which keeps ids stable and prevents forward references.
/// Frontends should use this builder instead of constructing [`QueryPlan`]
/// directly.
#[derive(Debug, Clone, Default)]
pub struct QueryPlanBuilder {
    nodes: Vec<QueryNode>,
    advisory_outputs: Vec<super::AdvisoryOutputBinding>,
}

impl QueryPlanBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a diagnostic frontend volume to an existing native occurrence.
    /// Conflicting frontend occurrences interned to one native node become unknown.
    pub fn set_advisory_output(&mut self, binding: super::AdvisoryOutputBinding) {
        assert_eq!(
            self.node(binding.node).output_schema(),
            &binding.schema,
            "advisory output schema must match its native occurrence"
        );
        if let Some(existing) = self.advisory_outputs.iter_mut().find(|value| value.node == binding.node) {
            if existing != &binding {
                existing.invalidate(super::AdvisoryUnknownReason::ConflictingOccurrences);
            }
        } else {
            self.advisory_outputs.push(binding);
        }
    }

    /// Returns the diagnostic frontend volume attached to an occurrence.
    pub fn advisory_output(&self, node: NodeId) -> Option<&super::AdvisoryOutputBinding> {
        self.advisory_outputs.iter().find(|binding| binding.node == node)
    }

    /// Adds a source node and returns its stable id.
    ///
    /// Source nodes establish their output schema and currently default to
    /// device placement when lowered into this IR.
    pub fn add_source(&mut self, source: SourceNode) -> NodeId {
        let output_schema = Arc::clone(source.output_schema());
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

    /// Adds a window node that appends function outputs while preserving input row identity.
    pub fn add_window(&mut self, input: NodeId, spec: WindowSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = self.node(input).facts.output_ordering.clone();
        self.push(
            PlanNode::Window(WindowNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::Projection).with_output_ordering(output_ordering),
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
        if spec.skip == 0
            && self.node(input).output_schema() == &output_schema
            && matches!(&self.node(input).kind, PlanNode::Limit(limit) if limit.spec == spec)
        {
            return input;
        }
        let output_ordering = self.node(input).facts.output_ordering.clone();
        let node = self.push(
            PlanNode::Limit(LimitNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        );
        if let Some(mut binding) = self.advisory_output(input).cloned() {
            binding.node = node;
            binding.schema = Arc::clone(self.node(node).output_schema());
            let limit_rows = |rows: u64| {
                let remaining = rows.saturating_sub(spec.skip as u64);
                spec.fetch.map_or(remaining, |fetch| remaining.min(fetch as u64))
            };
            binding.statistics.rows = match binding.statistics.rows {
                super::AdvisoryStatistic::Exact(rows) => super::AdvisoryStatistic::Exact(limit_rows(rows)),
                super::AdvisoryStatistic::Inexact(rows) => super::AdvisoryStatistic::Inexact(limit_rows(rows)),
                super::AdvisoryStatistic::Absent => super::AdvisoryStatistic::Absent,
            };
            binding.statistics.bytes = super::AdvisoryStatistic::Absent;
            if binding.statistics.rows.value().is_some() {
                binding.unknown_reason = Some(super::AdvisoryUnknownReason::LimitBytesUnavailable);
            }
            self.set_advisory_output(binding);
        }
        node
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

    /// Adds an ordered, schema-compatible union node.
    pub fn add_union(&mut self, inputs: Vec<NodeId>, output_schema: SchemaRef) -> NodeId {
        assert!(!inputs.is_empty(), "query plan union node must have at least one input");
        for input in &inputs {
            self.assert_input_exists(*input);
        }
        self.push(PlanNode::Union(UnionNode { inputs }), PlanFacts::device(output_schema, SchemaBehavior::Union))
    }

    /// Adds edge-list normalization before cuGraph raw graph construction.
    pub fn add_edge_normalize(&mut self, input: NodeId, spec: EdgeNormalizeSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        self.push(
            PlanNode::EdgeNormalize(EdgeNormalizeNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::EdgeNormalize),
        )
    }

    /// Adds a graph algorithm node over an edge-list input.
    ///
    /// This source-compatible wrapper is for callers that have already
    /// validated graph edge columns. Production lowerers should use
    /// [`Self::try_add_graph_algorithm`] so schema misses become query errors.
    pub fn add_graph_algorithm(&mut self, input: NodeId, spec: GraphAlgorithmSpec) -> NodeId {
        self.try_add_graph_algorithm(input, spec).expect("graph algorithm edge columns must exist")
    }

    /// Adds a graph algorithm node over an edge-list input, returning a
    /// structured error when implicit edge normalization cannot derive its
    /// schema.
    pub fn try_add_graph_algorithm(&mut self, input: NodeId, mut spec: GraphAlgorithmSpec) -> Result<NodeId> {
        self.assert_input_exists(input);
        let side_inputs = graph_algorithm_side_inputs(&spec);
        for side_input in &side_inputs {
            self.assert_input_exists(*side_input);
        }
        let input = if spec.graph_options.policy == GraphConstructionPolicy::PythonCugraph {
            let normalize_spec = EdgeNormalizeSpec::python_cugraph(spec.edge_columns.clone(), &spec.graph_options);
            spec.graph_options.policy = GraphConstructionPolicy::RawLibcugraph;
            spec.edge_normalize_inserted = true;
            let output_schema = edge_normalize_output_schema(self.node(input).output_schema(), &normalize_spec)?;
            self.add_edge_normalize(input, normalize_spec, output_schema)
        } else {
            input
        };
        let mut inputs = Vec::with_capacity(1 + side_inputs.len());
        inputs.push(input);
        inputs.extend(side_inputs);
        let output_schema = Arc::clone(&spec.output_schema);
        Ok(self.push(
            PlanNode::GraphAlgorithm(GraphAlgorithmNode { input, inputs, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::GraphAlgorithm),
        ))
    }

    /// Adds a typed vector algorithm node over its relation bindings.
    pub fn add_vector_algorithm(&mut self, spec: VectorAlgorithmSpec) -> NodeId {
        self.try_add_vector_algorithm(spec).expect("vector algorithm relation inputs must exist")
    }

    /// Returns whether the engine-owned output facts prove uniqueness on the
    /// requested physical columns for the current partial plan.
    pub fn output_is_unique_on(&self, node: NodeId, columns: &std::collections::BTreeSet<String>) -> Result<bool> {
        self.assert_input_exists(node);
        Ok(super::column_evidence::partial_plan_node_is_unique_on(&self.nodes, node, columns))
    }

    /// Adds a typed vector algorithm node and validates all relation references.
    pub fn try_add_vector_algorithm(&mut self, spec: VectorAlgorithmSpec) -> Result<NodeId> {
        let inputs = spec.params.relation_bindings().map(|binding| binding.input).collect::<Vec<_>>();
        if inputs.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.plan.builder.vector_algorithm",
                "vector algorithm must reference at least one relation input",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        }
        for input in &inputs {
            if self.nodes.get(input.index()).is_none() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.plan.builder.vector_algorithm",
                    "vector algorithm references a missing or forward relation input",
                )
                .with_fact("algorithm", spec.algorithm.as_str())
                .with_fact("input", input));
            }
        }
        let output_schema = Arc::clone(&spec.output_schema);
        Ok(self.push(
            PlanNode::VectorAlgorithm(VectorAlgorithmNode { inputs, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::VectorAlgorithm),
        ))
    }

    /// Finishes the plan with a host Arrow materialization sink.
    pub fn finish_with_host_sink(self, root: NodeId) -> QueryPlan {
        let schema = Arc::clone(self.node(root).output_schema());
        self.finish(root, ResultSink::HostArrow { input: root, schema })
    }

    /// Finishes the plan with a host Arrow materialization sink whose external
    /// field names, nullability, and metadata may differ from the executable
    /// root schema. Field count, order, and Arrow types remain exact.
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
        QueryPlan {
            nodes: self.nodes,
            advisory_outputs: self.advisory_outputs,
            root,
            sink,
            deferred_host_sink_schema: None,
            dynamic_filter_routes: Vec::new(),
            dynamic_filter_execution_options: Default::default(),
            keyed_hash_join_strategy_options: Default::default(),
        }
    }

    fn push(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return existing.id;
        }
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

fn graph_algorithm_side_inputs(spec: &GraphAlgorithmSpec) -> Vec<NodeId> {
    spec.params.side_input_node_ids().collect()
}

fn edge_normalize_output_schema(input_schema: &SchemaRef, spec: &EdgeNormalizeSpec) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(
        2 + usize::from(spec.edge_columns.weight.is_some()) + usize::from(spec.edge_columns.edge_id.is_some()),
    );
    fields.push(edge_normalize_field(input_schema, &spec.edge_columns.source, "source")?);
    fields.push(edge_normalize_field(input_schema, &spec.edge_columns.destination, "destination")?);
    if let Some(weight) = &spec.edge_columns.weight {
        fields.push(edge_normalize_field(input_schema, weight, "weight")?);
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        fields.push(edge_normalize_field(input_schema, edge_id, "edge_id")?);
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn edge_normalize_field(input_schema: &SchemaRef, column: &str, role: &'static str) -> Result<Field> {
    input_schema.field_with_name(column).cloned().map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.graph_algorithm.edge_normalize_schema",
            "edge-normalize column is not present in the input schema",
        )
        .with_fact("column", column)
        .with_fact("role", role)
        .with_source(error)
    })
}
