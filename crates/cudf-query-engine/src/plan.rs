use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};

use crate::{
    bounded_support::{
        BoundedJoinPlanSupport, analyze_bounded_join_pipeline, analyze_bounded_join_plan_support,
        analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
        analyze_bounded_keyless_conditional_join_input_subplan_pipeline,
        analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
        bounded_join_modeled_facts, bounded_join_unsupported_shape_facts, bounded_root_filtered_join_compatible,
        bounded_root_hash_join_compatible, bounded_root_join_compatible,
        bounded_root_keyless_conditional_join_compatible, bounded_root_mixed_join_compatible,
        bounded_root_projected_cross_join_compatible,
    },
    expr::{AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
    runtime::ExecutionBoundaryPolicy,
    source::SourceNode,
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
    nodes: Vec<QueryNode>,
    root: NodeId,
    sink: ResultSink,
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

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QueryPlan root={} sink={}", self.root, self.sink.kind_name())?;
        for node in &self.nodes {
            node.fmt_with_indent(f, 1)?;
        }
        match &self.sink {
            ResultSink::GpuResident { input } => {
                writeln!(f, "  Sink id=sink input={input} mode=gpu_resident")
            },
            ResultSink::HostArrow { input, schema } => writeln!(
                f,
                "  Sink id=sink input={input} mode=host_arrow materialization=device_to_host schema_fields={}",
                schema.fields().len()
            ),
        }
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
pub struct NodeId(usize);

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

    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let padding = "  ".repeat(indent);
        match &self.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    writeln!(
                        f,
                        "{padding}{} LocalParquetSource files={} path={} format={:?} direct_device_read={} known_bytes={} selected_row_groups={} read_columns={} projected_output_columns={} pure_filter_columns={} predicate={} byte_range={} partition_columns={} zero_column_count={} schema_mapping={} rejected_features={} placement={} schema_behavior={}",
                        self.id,
                        source.diagnostics.file_count,
                        source.files.first().map_or_else(|| "none".to_string(), |file| file.path.display().to_string()),
                        source.diagnostics.format,
                        source.diagnostics.direct_device_read,
                        source.diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
                        source
                            .diagnostics
                            .selected_row_groups
                            .map_or_else(|| "all".to_string(), |row_groups| row_groups.to_string()),
                        source.diagnostics.read_column_count,
                        source.diagnostics.projected_output_column_count,
                        source.diagnostics.pure_filter_column_count,
                        source.diagnostics.predicate_disposition.as_str(),
                        source.diagnostics.byte_range_disposition.as_str(),
                        source.diagnostics.partition_column_disposition.as_str(),
                        source.diagnostics.zero_column_count_disposition.as_str(),
                        source.diagnostics.schema_mapping_disposition.as_str(),
                        source
                            .diagnostics
                            .rejected_features
                            .iter()
                            .map(|feature| feature.as_str())
                            .collect::<Vec<_>>()
                            .join(","),
                        self.facts.placement,
                        self.facts.schema_behavior
                    )
                },
                SourceNode::GpuResident(source) => writeln!(
                    f,
                    "{padding}{} GpuResidentSource input_name={} schema_fields={} placement={} schema_behavior={}",
                    self.id,
                    source.input_name,
                    source.schema.fields().len(),
                    self.facts.placement,
                    self.facts.schema_behavior
                ),
            },
            PlanNode::Filter(node) => writeln!(
                f,
                "{padding}{} Filter input={} predicate={} placement={} schema_behavior={}",
                self.id, node.input, node.predicate, self.facts.placement, self.facts.schema_behavior
            ),
            PlanNode::Projection(node) => writeln!(
                f,
                "{padding}{} Projection input={} output_count={} expressions={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.outputs.len(),
                format_exprs(&node.expressions().cloned().collect::<Vec<_>>()),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Aggregate(node) => writeln!(
                f,
                "{padding}{} Aggregate input={} group_count={} aggr_count={} groups={} aggrs={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.group_exprs.len(),
                node.spec.aggr_exprs.len(),
                format_exprs(&node.spec.group_exprs),
                format_aggregate_exprs(&node.spec.aggr_exprs),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Sort(node) => writeln!(
                f,
                "{padding}{} Sort input={} expr_count={} expressions={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.expressions.len(),
                format_sort_exprs(&node.spec.expressions),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Limit(node) => writeln!(
                f,
                "{padding}{} Limit input={} skip={} fetch={} placement={} schema_behavior={}",
                self.id, node.input, node.spec.skip, node.spec.fetch, self.facts.placement, self.facts.schema_behavior
            ),
            PlanNode::Join(node) => writeln!(
                f,
                "{padding}{} Join left={} right={} type={} key_count={} residual={} null_equality={} build_side={} build_artifact={} output_projection={} placement={} schema_behavior={}",
                self.id,
                node.inputs[0],
                node.inputs[1],
                node.spec.join_type,
                node.spec.keys.len(),
                node.spec.residual_predicate.as_ref().map_or("none".to_string(), ToString::to_string),
                node.spec.null_equality,
                node.spec.build_side,
                node.spec.build_artifact_id.as_ref().map_or("none", HashJoinBuildArtifactId::as_str),
                format_join_output_projection(&node.spec.output_projection),
                self.facts.placement,
                self.facts.schema_behavior
            ),
        }
    }
}

fn format_exprs(expressions: &[Expr]) -> String {
    if expressions.is_empty() {
        "none".to_string()
    } else {
        expressions.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}

fn format_aggregate_exprs(expressions: &[crate::expr::AggregateExpr]) -> String {
    if expressions.is_empty() {
        "none".to_string()
    } else {
        expressions
            .iter()
            .map(|expr| format!("{:?}({}) AS {}", expr.function, expr.expr, expr.alias))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn format_sort_exprs(expressions: &[SortExpr]) -> String {
    if expressions.is_empty() {
        "none".to_string()
    } else {
        expressions
            .iter()
            .map(|expr| {
                format!(
                    "{} {} NULLS {}",
                    expr.expr,
                    if expr.descending { "DESC" } else { "ASC" },
                    if expr.nulls_first { "FIRST" } else { "LAST" }
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn format_join_output_projection(projection: &[JoinOutputColumn]) -> String {
    if projection.is_empty() {
        "none".to_string()
    } else {
        projection.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
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

/// Planner/executor facts attached to a node's output.
///
/// These fields describe contracts already established by lowering. They are
/// not optimizer hints, and changing them must not be used to request hidden
/// executor behavior. If behavior needs to change, add an explicit IR field or
/// operator instead.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanFacts {
    /// Schema produced by the node.
    pub output_schema: SchemaRef,
    /// Known ordering of the node output.
    ///
    /// Empty means no ordering contract is known. Sort nodes populate this
    /// directly, and order-preserving operators may carry it forward.
    pub output_ordering: Vec<SortExpr>,
    /// Physical residency of the node output.
    ///
    /// The native engine's normal boundary is GPU memory. Host residency should
    /// be introduced by an explicit source or sink contract, not inferred.
    pub placement: DataPlacement,
    /// Coarse reason this node has its output schema.
    pub schema_behavior: SchemaBehavior,
}

impl PlanFacts {
    fn device(output_schema: SchemaRef, schema_behavior: SchemaBehavior) -> Self {
        Self { output_schema, output_ordering: Vec::new(), placement: DataPlacement::Device, schema_behavior }
    }

    fn with_output_ordering(mut self, output_ordering: Vec<SortExpr>) -> Self {
        self.output_ordering = output_ordering;
        self
    }
}

/// Physical residency for a node output.
///
/// Native query execution defaults to GPU-resident data. Host placement is a
/// boundary fact and should correspond to an explicit source/sink or future
/// transfer node, not an implicit materialization point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataPlacement {
    /// Output lives in device/GPU memory.
    Device,
    /// Output lives in host memory.
    Host,
}

impl fmt::Display for DataPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Device => "device",
            Self::Host => "host",
        })
    }
}

/// Describes how a node's output schema relates to its inputs.
///
/// This is a diagnostic and validation fact. It is intentionally coarse: the
/// concrete schema is still carried by [`PlanFacts::output_schema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaBehavior {
    /// Schema is established by a source.
    SourceOutput,
    /// Schema is expected to match the input schema.
    PreservesInput,
    /// Schema is produced by projection expressions.
    Projection,
    /// Schema is produced by aggregate group and aggregate expressions.
    Aggregate,
    /// Schema is produced by join type and output projection.
    Join,
}

impl fmt::Display for SchemaBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SourceOutput => "source_output",
            Self::PreservesInput => "preserves_input",
            Self::Projection => "projection",
            Self::Aggregate => "aggregate",
            Self::Join => "join",
        })
    }
}

/// Explicit result boundary for a native query plan.
///
/// A sink is part of the IR so callers can choose whether the final result stays
/// GPU-resident or crosses back to Arrow on the host. Operators before the sink
/// should not materialize host output merely because they are the plan root.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultSink {
    /// Return a GPU-resident handle for the sink input.
    GpuResident { input: NodeId },
    /// Materialize the sink input to host Arrow batches with the given schema.
    HostArrow { input: NodeId, schema: SchemaRef },
}

impl ResultSink {
    /// Returns the node consumed by this sink.
    pub fn input(&self) -> NodeId {
        match self {
            Self::GpuResident { input } | Self::HostArrow { input, .. } => *input,
        }
    }

    /// Returns true when executing this sink requires a device-to-host boundary.
    pub fn materializes_host_boundary(&self) -> bool {
        matches!(self, Self::HostArrow { .. })
    }

    fn kind_name(&self) -> &'static str {
        match self {
            Self::GpuResident { .. } => "gpu_resident",
            Self::HostArrow { .. } => "host_arrow",
        }
    }
}

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

/// Boolean filtering operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterNode {
    /// Input whose rows are tested.
    pub input: NodeId,
    /// Predicate evaluated against the input schema.
    pub predicate: Expr,
}

/// Expression projection over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionNode {
    /// Input whose columns and computed expressions are projected.
    pub input: NodeId,
    /// Output expressions and fields in result schema order.
    pub outputs: Vec<ProjectionOutput>,
    output_schema: SchemaRef,
}

impl ProjectionNode {
    pub fn new(input: NodeId, outputs: Vec<ProjectionOutput>) -> Self {
        let output_schema = projection_output_schema(&outputs);
        Self { input, outputs, output_schema }
    }

    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    pub fn expressions(&self) -> impl Iterator<Item = &Expr> {
        self.outputs.iter().map(|output| &output.expr)
    }

    pub fn output_fields(&self) -> impl Iterator<Item = &Field> {
        self.outputs.iter().map(|output| &output.field)
    }
}

/// One typed output of a projection node.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionOutput {
    /// Expression evaluated for this output column.
    pub expr: Expr,
    /// Arrow field that names and types the executable output column.
    pub field: Field,
}

impl ProjectionOutput {
    pub fn new(expr: Expr, field: Field) -> Self {
        Self { expr, field }
    }
}

fn projection_output_schema(outputs: &[ProjectionOutput]) -> SchemaRef {
    Arc::new(Schema::new(outputs.iter().map(|output| output.field.clone()).collect::<Vec<_>>()))
}

/// Aggregate operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateNode {
    /// Input whose rows are grouped and aggregated.
    pub input: NodeId,
    /// Group and aggregate expressions. Distributed or partial/final aggregate
    /// states must be represented explicitly in this spec.
    pub spec: AggregateSpec,
}

/// Sort operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct SortNode {
    /// Input whose rows are sorted.
    pub input: NodeId,
    /// Sort expressions and optional fetch.
    pub spec: SortSpec,
}

/// Limit operation over one input.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitNode {
    /// Input whose rows are limited.
    pub input: NodeId,
    /// Skip/fetch contract.
    pub spec: LimitSpec,
}

/// Binary join operation.
///
/// `inputs[0]` is the semantic left input and `inputs[1]` is the semantic right
/// input. Physical build/probe choices are recorded separately in
/// [`JoinSpec::build_side`] so join semantics remain independent from executor
/// strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinNode {
    /// Semantic join inputs in `[left, right]` order.
    pub inputs: [NodeId; 2],
    /// Join semantics, predicates, projection, and build/cache choices.
    pub spec: JoinSpec,
}

impl JoinNode {
    /// The single source of truth for whether this join can be executed by the
    /// bounded root reusable hash-join path. The `Left → build_side == Right`
    /// constraint is load-bearing: `cudf-rs::DataFrameHashJoin::left_probe`
    /// treats the probe input as the left side of the outer join, so a
    /// `JoinType::Left` with `build_side == Left` would keep right-side rows
    /// for unmatched build rows and produce the wrong DataFusion semantics.
    pub fn is_bounded_root_hash_join_compatible(&self) -> bool {
        bounded_root_hash_join_compatible(self)
    }

    /// Returns true when this join can use the bounded keyless conditional root path.
    pub fn is_bounded_root_keyless_conditional_join_compatible(&self) -> bool {
        bounded_root_keyless_conditional_join_compatible(self)
    }

    /// Returns true when this join can use the bounded filtered-join root path.
    pub fn is_bounded_root_filtered_join_compatible(&self) -> bool {
        bounded_root_filtered_join_compatible(self)
    }

    /// Returns true when this join can use the bounded projected cross-join root path.
    pub fn is_bounded_root_projected_cross_join_compatible(&self) -> bool {
        bounded_root_projected_cross_join_compatible(self)
    }

    /// Returns true when this join can use the bounded mixed-predicate root path.
    pub fn is_bounded_root_mixed_join_compatible(&self) -> bool {
        bounded_root_mixed_join_compatible(self)
    }

    /// Returns true when this join is accepted by any bounded root join path.
    pub fn is_bounded_root_join_compatible(&self) -> bool {
        bounded_root_join_compatible(self)
    }
}

/// Complete semantic and physical contract for a native join node.
///
/// Joins intentionally keep four concerns separate: equality keys in
/// [`JoinSpec::keys`], an optional non-equi residual predicate in
/// [`JoinSpec::residual_predicate`], output column selection/order in
/// [`JoinSpec::output_projection`], and physical build/cache choices in
/// [`JoinSpec::build_side`] plus [`JoinSpec::build_artifact_id`].
///
/// A frontend should not encode residual predicates as fake keys, rely on the
/// executor's default output ordering when column order matters, or treat
/// build-side selection as changing join semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinSpec {
    /// Logical join type.
    pub join_type: JoinType,
    /// Equality predicates matched by column pairs.
    ///
    /// Empty keys are valid for cross joins and keyless conditional joins when
    /// the selected executor path supports them.
    pub keys: Vec<JoinKey>,
    /// Optional predicate evaluated after equality matching.
    ///
    /// This is the place for non-equality join conditions. Keeping it separate
    /// lets the planner reason about hash keys and post-join filters
    /// independently.
    pub residual_predicate: Option<Expr>,
    /// Explicit output projection in result schema order.
    ///
    /// Executable native joins require this to be populated explicitly and to
    /// match the result schema order. An empty projection is only an incomplete
    /// builder default before a frontend sets the projection; execution
    /// validation rejects it.
    pub output_projection: Vec<JoinOutputColumn>,
    /// Null matching behavior for equality keys.
    pub null_equality: JoinNullEquality,
    /// Input chosen to build reusable/hash state.
    ///
    /// This is a physical execution choice. It does not redefine which input is
    /// semantic left or right; those remain [`JoinNode::inputs`].
    pub build_side: JoinSide,
    /// Durable cache identifier for reusable hash-join build state.
    ///
    /// When present, the executor may store and reuse the build artifact under
    /// this id for compatible joins in the same execution context. The id is a
    /// contract between frontend and executor: it must be non-empty and stable
    /// for the logical build input, key set, null-equality behavior, and other
    /// build-affecting choices.
    pub build_artifact_id: Option<HashJoinBuildArtifactId>,
}

impl JoinSpec {
    /// Creates a join spec with default physical choices.
    ///
    /// Defaults are: no residual predicate, empty output projection,
    /// `NullsDoNotMatch`, right-side build, and no reusable build artifact id.
    /// The empty output projection is an incomplete builder default; frontends
    /// must set an explicit projection before the join can execute.
    pub fn new(join_type: JoinType, keys: Vec<JoinKey>) -> Self {
        Self {
            join_type,
            keys,
            residual_predicate: None,
            output_projection: Vec::new(),
            null_equality: JoinNullEquality::NullsDoNotMatch,
            build_side: JoinSide::Right,
            build_artifact_id: None,
        }
    }

    /// Sets the non-equality join predicate.
    pub fn with_residual_predicate(mut self, predicate: Expr) -> Self {
        self.residual_predicate = Some(predicate);
        self
    }

    /// Sets an explicit result projection for the join.
    pub fn with_output_projection(mut self, output_projection: Vec<JoinOutputColumn>) -> Self {
        self.output_projection = output_projection;
        self
    }

    /// Sets null matching behavior for equality keys.
    pub fn with_null_equality(mut self, null_equality: JoinNullEquality) -> Self {
        self.null_equality = null_equality;
        self
    }

    /// Sets the physical build side without changing semantic left/right inputs.
    pub fn with_build_side(mut self, build_side: JoinSide) -> Self {
        self.build_side = build_side;
        self
    }

    /// Assigns a durable reusable-build cache id.
    ///
    /// The value must identify the complete build artifact contract, not just a
    /// table name. Validation rejects empty ids.
    pub fn with_build_artifact_id(mut self, id: impl Into<String>) -> Self {
        self.build_artifact_id = Some(HashJoinBuildArtifactId::new(id));
        self
    }
}

/// Durable identifier for reusable hash-join build state.
///
/// The executor uses this as a cache key for GPU build artifacts within an
/// execution context. Frontends are responsible for choosing ids that remain
/// stable for equivalent build inputs and change when any build-affecting
/// property changes. This is the native IR's reusable-join-build handle; there
/// is deliberately no DataFusion dependency in the identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HashJoinBuildArtifactId(String);

impl HashJoinBuildArtifactId {
    /// Creates a new cache identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the raw identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns true when this id would be invalid for reusable build caching.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for HashJoinBuildArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One equality-key pair for a join.
///
/// `left` is resolved against the semantic left input and `right` against the
/// semantic right input. Non-equality conditions belong in
/// [`JoinSpec::residual_predicate`], not in this list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinKey {
    /// Column from the semantic left input.
    pub left: ColumnRef,
    /// Column from the semantic right input.
    pub right: ColumnRef,
}

impl JoinKey {
    /// Creates one left/right equality key pair.
    pub fn new(left: ColumnRef, right: ColumnRef) -> Self {
        Self { left, right }
    }
}

/// One column selected by an explicit join output projection.
///
/// The projection list order is the result schema order. `output_name` is the
/// name exposed at the join output boundary and may differ from the input column
/// name to resolve duplicates or preserve frontend naming rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinOutputColumn {
    /// Which semantic input side provides the column.
    pub side: JoinSide,
    /// Column selected from that side.
    pub column: ColumnRef,
    /// Name assigned to the output field.
    pub output_name: String,
}

impl JoinOutputColumn {
    /// Creates a projected column from either semantic input side.
    pub fn new(side: JoinSide, column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self { side, column, output_name: output_name.into() }
    }

    /// Creates a projected column from the semantic left input.
    pub fn left(column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self::new(JoinSide::Left, column, output_name)
    }

    /// Creates a projected column from the semantic right input.
    pub fn right(column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self::new(JoinSide::Right, column, output_name)
    }
}

impl fmt::Display for JoinOutputColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{} AS {}", self.side, self.column.name(), self.output_name)
    }
}

/// Logical join type represented by the native IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Keep rows that match on the join condition from both sides.
    Inner,
    /// Keep every left row, with nullable right columns when unmatched.
    Left,
    /// Keep every row from both sides, with nullable columns for unmatched side.
    Full,
    /// Keep left rows that have a match; output only left-side columns.
    LeftSemi,
    /// Keep left rows that do not have a match; output only left-side columns.
    LeftAnti,
    /// Cartesian join; equality keys should be empty.
    Cross,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Inner => "inner",
            Self::Left => "left",
            Self::Full => "full",
            Self::LeftSemi => "left_semi",
            Self::LeftAnti => "left_anti",
            Self::Cross => "cross",
        })
    }
}

/// Null matching policy for join equality keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinNullEquality {
    /// Null key values do not match other nulls.
    NullsDoNotMatch,
    /// Null key values match other nulls.
    NullsMatch,
}

impl fmt::Display for JoinNullEquality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NullsDoNotMatch => "nulls_do_not_match",
            Self::NullsMatch => "nulls_match",
        })
    }
}

/// Semantic input side for a binary join.
///
/// In [`JoinNode::inputs`], left is index 0 and right is index 1. The same enum
/// is also used for physical build-side selection; in that context it names
/// which semantic side is built, not a new logical orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinSide {
    /// Semantic left input.
    Left,
    /// Semantic right input.
    Right,
}

impl fmt::Display for JoinSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Left => "left",
            Self::Right => "right",
        })
    }
}
