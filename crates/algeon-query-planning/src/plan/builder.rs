//! Append-only builder for [`QueryPlan`].
//!
//! Frontends construct plans through [`QueryPlanBuilder`] so node ids stay
//! stable and forward references are impossible. Equivalent nodes (same kind
//! and facts) are interned as a small CSE step. Every method that takes an
//! input id returns an error when the id names no node of this builder, and
//! finishing rejects nodes the sink does not reach.
use std::{collections::BTreeSet, sync::Arc};

use super::{
    AggregateNode, AggregateSpec, Annotations, DataType, EdgeNormalizeNode, EdgeNormalizeSpec, Expr, Field, FilterNode,
    GraphAlgorithmNode, GraphAlgorithmSpec, GraphConstructionPolicy, JoinNode, JoinSpec, LimitNode, LimitSpec,
    MlAlgorithmNode, MlAlgorithmSpec, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode,
    QueryPlan, ResultSink, SchemaRef, SortNode, SortSpec, SourceNode, UnionNode, VectorAlgorithmNode,
    VectorAlgorithmSpec, WindowNode, WindowSpec, graph::derive_output, uniqueness::partial_plan_node_is_unique_on,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::alias_wrapped_column,
    naming::first_available_numeric_suffix_from,
};

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
    ///
    /// A native projection evaluates `REGEXP_REPLACE` only as a whole output
    /// over an input column. When an output uses it anywhere else, such as
    /// inside `LOWER`, a `CASE` branch or over a computed input, earlier
    /// projections compute each replacement input and each replacement as a
    /// named column. The returned id names the last projection, whose outputs
    /// are `outputs` reading those columns.
    pub fn add_projection(&mut self, input: NodeId, outputs: Vec<ProjectionOutput>) -> Result<NodeId> {
        let input_schema = Arc::clone(self.node_output_schema(input).ok_or_else(|| missing_input(input))?);
        let staged = stage_regexp_replace(&input_schema, outputs);
        let mut node = input;
        for stage in staged.stages {
            node = self.push(PlanNode::Projection(ProjectionNode::new(node, stage)), None, Vec::new())?;
        }
        self.push(PlanNode::Projection(ProjectionNode::new(node, staged.outputs)), None, Vec::new())
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

const REPLACEMENT_COLUMN_BASE: &str = "__algeon_native_regexp_replace";
const REPLACEMENT_INPUT_COLUMN_BASE: &str = "__algeon_native_regexp_replace_input";

/// A projection split into the projections that run before it and its own
/// outputs.
struct StagedProjection {
    /// Outputs of the projections that run first, in order.
    stages: Vec<Vec<ProjectionOutput>>,
    outputs: Vec<ProjectionOutput>,
}

/// Moves every `REGEXP_REPLACE` that `outputs` place where a native projection
/// does not evaluate it into earlier projections over `input_schema`.
///
/// Outputs a projection already evaluates are returned unchanged with no
/// stages, as are outputs naming a column `input_schema` lacks; plan
/// validation reports that column against the projection that names it. When
/// staging is needed, an output that is a replacement over a column and equals
/// a staged replacement becomes a rename of the staged column, so the
/// replacement runs once.
fn stage_regexp_replace(input_schema: &SchemaRef, mut outputs: Vec<ProjectionOutput>) -> StagedProjection {
    let unresolved_column = || {
        outputs
            .iter()
            .flat_map(|output| output.expr.columns())
            .any(|column| input_schema.field_with_name(column.name()).is_err())
    };
    if outputs.iter().all(|output| output.expr.projection_evaluates_regexp_replace()) || unresolved_column() {
        return StagedProjection { stages: Vec::new(), outputs };
    }
    let mut staging = Staging { input_schema, stages: Vec::new(), computed: Vec::new() };
    for output in &mut outputs {
        if !output.expr.projection_evaluates_regexp_replace() {
            staging.extract(&mut output.expr);
        }
    }
    for output in &mut outputs {
        staging.reuse_staged_replacement(&mut output.expr);
    }
    StagedProjection { stages: staging.finish(&outputs), outputs }
}

struct Staging<'a> {
    input_schema: &'a SchemaRef,
    /// Columns each stage computes. Level `n` owns stage `2n`, which computes
    /// replacement inputs, and stage `2n + 1`, which computes replacements.
    stages: Vec<Vec<ProjectionOutput>>,
    computed: Vec<ComputedColumn>,
}

struct ComputedColumn {
    field: Field,
    stage: usize,
    /// The replacement or replacement input this column holds, so that an equal
    /// one of the same type reads it. An input never contains a replacement,
    /// because extraction replaces those first, so the two kinds cannot be equal.
    source: Expr,
}

impl Staging<'_> {
    /// Replaces each `REGEXP_REPLACE` in `expr`, innermost first, with the
    /// column that computes it.
    fn extract(&mut self, expr: &mut Expr) {
        for child in expr.children_mut() {
            self.extract(child);
        }
        if matches!(expr, Expr::RegexpReplace { .. }) {
            *expr = Expr::column(self.replacement_column(expr));
        }
    }

    /// Makes an output that is itself a replacement over a column read the
    /// column that already computes an equal replacement.
    fn reuse_staged_replacement(&self, expr: &mut Expr) {
        if let Expr::Alias { expr, .. } = expr {
            self.reuse_staged_replacement(expr);
        } else if let Expr::RegexpReplace { output_type, .. } = &*expr
            && let Some(column) = self.computed_column(expr, output_type)
        {
            *expr = Expr::column(column);
        }
    }

    fn replacement_column(&mut self, replacement: &Expr) -> String {
        let Expr::RegexpReplace { expr: input, pattern, replacement: template, output_type } = replacement else {
            unreachable!("only REGEXP_REPLACE expressions are staged");
        };
        let latest_read = input.columns().filter_map(|column| self.stage_of(column.name())).max();
        let input_stage = latest_read.map_or(0, |stage| stage + 2 - stage % 2);
        let input = if alias_wrapped_column(input).is_some() {
            input.as_ref().clone()
        } else {
            // A replacement input is a string. String representations share
            // one cuDF type and validation admits the implicit cast between
            // them, so the input column takes the replacement's string type.
            Expr::column(self.compute(
                input_stage,
                REPLACEMENT_INPUT_COLUMN_BASE,
                output_type,
                input,
                input.as_ref().clone(),
            ))
        };
        self.compute(
            input_stage + 1,
            REPLACEMENT_COLUMN_BASE,
            output_type,
            replacement,
            Expr::regexp_replace(input, pattern.clone(), template.clone(), output_type.clone()),
        )
    }

    /// The column that computes `expr` as `source`, adding it to `stage` unless
    /// an equal `source` of the same type already has one.
    fn compute(&mut self, stage: usize, base: &str, data_type: &DataType, source: &Expr, expr: Expr) -> String {
        if let Some(existing) = self.computed_column(source, data_type) {
            return existing;
        }
        let suffix = first_available_numeric_suffix_from(0, |suffix| self.is_taken(&format!("{base}_{suffix}")));
        let field = Field::new(format!("{base}_{suffix}"), data_type.clone(), true);
        if self.stages.len() <= stage {
            self.stages.resize_with(stage + 1, Vec::new);
        }
        self.stages[stage].push(ProjectionOutput::new(expr, field.clone()));
        let name = field.name().clone();
        self.computed.push(ComputedColumn { field, stage, source: source.clone() });
        name
    }

    fn computed_column(&self, source: &Expr, data_type: &DataType) -> Option<String> {
        self.computed
            .iter()
            .find(|column| column.source == *source && column.field.data_type() == data_type)
            .map(|column| column.field.name().clone())
    }

    fn is_taken(&self, name: &str) -> bool {
        self.input_schema.field_with_name(name).is_ok()
            || self.computed.iter().any(|column| column.field.name() == name)
    }

    fn stage_of(&self, name: &str) -> Option<usize> {
        self.computed.iter().find(|column| column.field.name() == name).map(|column| column.stage)
    }

    /// The non-empty stages, each passing through the columns that later
    /// stages and `outputs` read ahead of the columns it computes. Passed
    /// columns keep input-schema order, then computation order.
    fn finish(self, outputs: &[ProjectionOutput]) -> Vec<Vec<ProjectionOutput>> {
        let known = self
            .input_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain(self.computed.iter().map(|column| column.field.clone()))
            .collect::<Vec<_>>();
        let mut read = outputs
            .iter()
            .flat_map(|output| output.expr.columns())
            .map(|column| column.name().to_string())
            .collect::<BTreeSet<_>>();
        let mut stages = Vec::new();
        for computed in self.stages.into_iter().rev().filter(|stage| !stage.is_empty()) {
            let defined = computed.iter().map(|output| output.field.name().as_str()).collect::<BTreeSet<_>>();
            let mut stage = known
                .iter()
                .filter(|field| read.contains(field.name()) && !defined.contains(field.name().as_str()))
                .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.clone()))
                .collect::<Vec<_>>();
            read = stage
                .iter()
                .map(|output| output.field.name().clone())
                .chain(computed.iter().flat_map(|output| output.expr.columns()).map(|column| column.name().to_string()))
                .collect();
            stage.extend(computed);
            stages.push(stage);
        }
        stages.reverse();
        stages
    }
}
