//! Append-only builder for [`QueryPlan`].
//!
//! Frontends construct plans through [`QueryPlanBuilder`] so node ids stay
//! stable and forward references are impossible. Equivalent nodes (same kind
//! and facts) are interned as a small CSE step. Every method that takes an
//! input id returns an error when the id names no node of this builder, and
//! finishing rejects nodes the sink does not reach.
use std::{collections::BTreeSet, sync::Arc};

use super::{
    AggregateNode, AggregateSpec, Annotations, EdgeNormalizeNode, EdgeNormalizeSpec, Expr, FilterNode,
    GraphAlgorithmNode, GraphAlgorithmSpec, GraphConstructionPolicy, JoinNode, JoinSpec, LimitNode, LimitSpec,
    MlAlgorithmNode, MlAlgorithmSpec, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode,
    QueryPlan, ResultSink, SchemaRef, SortNode, SortSpec, SourceNode, UnionNode, VectorAlgorithmNode,
    VectorAlgorithmSpec, WindowNode, WindowSpec, graph::derive_output, uniqueness::partial_plan_node_is_unique_on,
};
use crate::error::{QueryError, QueryErrorCode, Result};

const OPERATION: &str = "query_engine.plan.builder";

/// Builder for native query plans.
///
/// The builder assigns stable [`NodeId`]s while interning equivalent nodes as a
/// local common-subexpression-elimination step. Methods validate that referenced
/// inputs already exist, which keeps ids stable and prevents forward references.
/// A [`NodeId`] is a plan-local index: an id taken from another builder or plan
/// that happens to be in range is not detected. Frontends should use this
/// builder and avoid constructing [`QueryPlan`] directly.
#[derive(Debug, Clone, Default)]
pub struct QueryPlanBuilder {
    /// Every node already validated against its predecessors; the output
    /// schema of each is derived at insertion, so later `add_*` calls and
    /// `node_output_schema` read it without a second pass.
    nodes: Vec<QueryNode>,
    advisory_outputs: Vec<super::AdvisoryOutputBinding>,
}

impl QueryPlanBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a diagnostic frontend volume to an existing native occurrence.
    /// Conflicting frontend occurrences interned to one native node become unknown.
    pub fn set_advisory_output(&mut self, binding: super::AdvisoryOutputBinding) -> Result<()> {
        if self.node_output_schema(binding.node) != Some(&binding.schema) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "advisory output schema must match its native occurrence",
            )
            .with_fact("node", binding.node));
        }
        if let Some(existing) = self.advisory_outputs.iter_mut().find(|value| value.node == binding.node) {
            if existing != &binding {
                existing.invalidate(super::AdvisoryUnknownReason::ConflictingOccurrences);
            }
        } else {
            self.advisory_outputs.push(binding);
        }
        Ok(())
    }

    /// Returns the diagnostic frontend volume attached to an occurrence.
    pub(crate) fn advisory_output(&self, node: NodeId) -> Option<&super::AdvisoryOutputBinding> {
        self.advisory_outputs.iter().find(|binding| binding.node == node)
    }

    /// The output schema of `node`, when this builder holds it.
    pub fn node_output_schema(&self, node: NodeId) -> Option<&SchemaRef> {
        self.nodes.get(node.index()).map(QueryNode::output_schema)
    }

    /// Adds a source node and returns its stable id.
    pub fn add_source(&mut self, source: SourceNode) -> NodeId {
        let output_schema = Arc::clone(source.output_schema());
        let id = NodeId(self.nodes.len());
        let kind = PlanNode::Source(Box::new(source));
        let facts = PlanFacts::new(output_schema);
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return existing.id;
        }
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }

    /// Adds an order-preserving filter node whose output is its input's schema.
    pub fn add_filter(&mut self, input: NodeId, predicate: Expr) -> Result<NodeId> {
        let ordering = self.input_ordering(input)?;
        self.push(PlanNode::Filter(FilterNode { input, predicate }), None, ordering)
    }

    /// Adds a projection node from explicit typed outputs.
    pub fn add_projection(&mut self, input: NodeId, outputs: Vec<ProjectionOutput>) -> Result<NodeId> {
        self.push(PlanNode::Projection(ProjectionNode::new(input, outputs)), None, Vec::new())
    }

    /// Adds an aggregate node with its declared output schema.
    ///
    /// Aggregate semantics, including partial/final state representation, must
    /// be modeled explicitly in [`AggregateSpec`] and the expression IR before a
    /// frontend lowers them here.
    pub fn add_aggregate(&mut self, input: NodeId, spec: AggregateSpec, output_schema: SchemaRef) -> Result<NodeId> {
        self.push(PlanNode::Aggregate(AggregateNode { input, spec }), Some(output_schema), Vec::new())
    }

    /// Adds a window node that appends one output field per function to its
    /// input's fields while preserving input row identity.
    pub fn add_window(&mut self, input: NodeId, spec: WindowSpec) -> Result<NodeId> {
        let ordering = self.input_ordering(input)?;
        self.push(PlanNode::Window(WindowNode { input, spec }), None, ordering)
    }

    /// Adds a sort node and records its output ordering fact.
    pub fn add_sort(&mut self, input: NodeId, spec: SortSpec) -> Result<NodeId> {
        let ordering = spec.expressions.clone();
        self.push(PlanNode::Sort(SortNode { input, spec }), None, ordering)
    }

    /// Adds a limit node that preserves input ordering facts.
    ///
    /// A fetch-only limit over an identical limit returns the input id, because
    /// the second limit would pass every row through. An advisory binding on the
    /// input moves to the new node with its row estimate clipped by the limit;
    /// its byte estimate is dropped because the frontend's byte model gives no
    /// per-row figure to scale it by.
    pub fn add_limit(&mut self, input: NodeId, spec: LimitSpec) -> Result<NodeId> {
        let ordering = self.input_ordering(input)?;
        if spec.skip == 0 && matches!(&self.nodes[input.index()].kind, PlanNode::Limit(limit) if limit.spec == spec) {
            return Ok(input);
        }
        let node = self.push(PlanNode::Limit(LimitNode { input, spec }), None, ordering)?;
        if let Some(mut binding) = self.advisory_output(input).cloned() {
            binding.node = node;
            binding.schema = Arc::clone(self.nodes[node.index()].output_schema());
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
            self.set_advisory_output(binding)?;
        }
        Ok(node)
    }

    /// Adds a join node with explicit left and right inputs and its declared
    /// output schema.
    ///
    /// Equality keys, residual predicate, output projection, null matching, and
    /// build-side/cache behavior live in [`JoinSpec`] so frontends do not need to
    /// encode those contracts through expression conventions.
    pub fn add_join(
        &mut self,
        left: NodeId,
        right: NodeId,
        spec: JoinSpec,
        output_schema: SchemaRef,
    ) -> Result<NodeId> {
        self.push(PlanNode::Join(JoinNode { inputs: [left, right], spec }), Some(output_schema), Vec::new())
    }

    /// Adds an ordered union node. `output_schema` is the frontend's canonical
    /// schema: it names the output, and each input matches it by position in
    /// Arrow type, with no nullable input under a non-nullable output field.
    pub fn add_union(&mut self, inputs: Vec<NodeId>, output_schema: SchemaRef) -> Result<NodeId> {
        self.push(PlanNode::Union(UnionNode { inputs }), Some(output_schema), Vec::new())
    }

    /// Adds edge-list normalization before cuGraph raw graph construction.
    pub fn add_edge_normalize(&mut self, input: NodeId, spec: EdgeNormalizeSpec) -> Result<NodeId> {
        self.push(PlanNode::EdgeNormalize(EdgeNormalizeNode { input, spec }), None, Vec::new())
    }

    /// Adds a graph algorithm node over an edge-list input and its side
    /// inputs; a side-input binding in `spec` names its relation by position,
    /// counting the edge-list input as position 0. A
    /// [`GraphConstructionPolicy::PythonCugraph`] spec gets an explicit edge
    /// normalization below it.
    pub fn add_graph_algorithm(
        &mut self,
        input: NodeId,
        side_inputs: Vec<NodeId>,
        mut spec: GraphAlgorithmSpec,
    ) -> Result<NodeId> {
        let input = if spec.graph_options.policy == GraphConstructionPolicy::PythonCugraph {
            let normalize_spec = EdgeNormalizeSpec::python_cugraph(spec.edge_columns.clone(), &spec.graph_options);
            spec.graph_options.policy = GraphConstructionPolicy::RawLibcugraph;
            spec.edge_normalize_inserted = true;
            self.add_edge_normalize(input, normalize_spec)?
        } else {
            input
        };
        let inputs = std::iter::once(input).chain(side_inputs).collect();
        self.push(PlanNode::GraphAlgorithm(GraphAlgorithmNode { inputs, spec }), None, Vec::new())
    }

    /// Returns whether the engine-owned output facts prove uniqueness on the
    /// requested physical columns for the current partial plan. An id this
    /// builder does not hold proves nothing.
    pub fn output_is_unique_on(&self, node: NodeId, columns: &BTreeSet<String>) -> bool {
        partial_plan_node_is_unique_on(&self.nodes, node, columns)
    }

    /// Adds a typed vector algorithm node over its relation inputs; each
    /// relation binding in `spec` names its relation by position in `inputs`.
    pub fn add_vector_algorithm(&mut self, inputs: Vec<NodeId>, spec: VectorAlgorithmSpec) -> Result<NodeId> {
        self.push(PlanNode::VectorAlgorithm(VectorAlgorithmNode { inputs, spec }), None, Vec::new())
    }

    /// Adds a cuML node over its relation inputs; each relation binding in
    /// `spec` names its relation by position in `inputs`.
    pub fn add_ml_algorithm(&mut self, inputs: Vec<NodeId>, spec: MlAlgorithmSpec) -> Result<NodeId> {
        self.push(PlanNode::MlAlgorithm(MlAlgorithmNode { inputs, spec }), None, Vec::new())
    }

    /// Finishes the plan with a host Arrow materialization sink.
    pub fn finish_with_host_sink(self, root: NodeId) -> Result<QueryPlan> {
        let schema = Arc::clone(self.node_output_schema(root).ok_or_else(|| missing_input(root))?);
        self.finish(ResultSink::HostArrow { input: root, schema })
    }

    /// Finishes the plan with a host Arrow materialization sink whose external
    /// field names, nullability, and metadata may differ from the executable
    /// root schema. Field count, order, and Arrow types remain exact.
    pub fn finish_with_host_sink_schema(self, root: NodeId, schema: SchemaRef) -> Result<QueryPlan> {
        self.finish(ResultSink::HostArrow { input: root, schema })
    }

    /// Finishes the plan with a GPU-resident result sink.
    pub fn finish_gpu_resident(self, root: NodeId) -> Result<QueryPlan> {
        self.finish(ResultSink::GpuResident { input: root })
    }

    fn finish(self, sink: ResultSink) -> Result<QueryPlan> {
        QueryPlan::from_parts(
            self.nodes,
            sink,
            Annotations { dynamic_filter_routes: Vec::new(), advisory: self.advisory_outputs },
        )
    }

    fn input_ordering(&self, input: NodeId) -> Result<Vec<super::SortExpr>> {
        self.nodes.get(input.index()).map(|node| node.facts.output_ordering.clone()).ok_or_else(|| missing_input(input))
    }

    /// Validates and appends one node, or returns the id of an equal node
    /// already held.
    ///
    /// The input check precedes `derive_output` because derivation indexes
    /// `self.nodes` by input id and would panic on a forward reference. The
    /// interning compares the full kind and facts, so two nodes that differ
    /// only in a declared schema or ordering fact stay distinct.
    fn push(&mut self, kind: PlanNode, declared: Option<SchemaRef>, ordering: Vec<super::SortExpr>) -> Result<NodeId> {
        let id = NodeId(self.nodes.len());
        if let Some(input) = kind.input_ids().iter().find(|input| input.index() >= self.nodes.len()) {
            return Err(missing_input(*input));
        }
        let output_schema = derive_output(id, &kind, declared.as_ref(), &self.nodes)?;
        let facts = PlanFacts::new(output_schema).with_output_ordering(ordering);
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return Ok(existing.id);
        }
        self.nodes.push(QueryNode { id, kind, facts });
        Ok(id)
    }
}

fn missing_input(input: NodeId) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        OPERATION,
        "query plan builder input does not name a node of this builder",
    )
    .with_fact("input", input)
}
