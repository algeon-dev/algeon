use std::fmt;

use arrow::datatypes::SchemaRef;

use crate::{
    bounded_support::{
        BoundedJoinPlanSupport, analyze_bounded_join_pipeline, analyze_bounded_join_plan_support,
        bounded_root_hash_join_compatible,
    },
    expr::{AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
    source::SourceNode,
};

#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    nodes: Vec<QueryNode>,
    root: NodeId,
    sink: ResultSink,
}

impl QueryPlan {
    pub fn root_id(&self) -> NodeId {
        self.root
    }

    pub fn root_node(&self) -> &QueryNode {
        self.node(self.root).expect("query plan root node must exist")
    }

    pub fn sink(&self) -> &ResultSink {
        &self.sink
    }

    pub fn nodes(&self) -> &[QueryNode] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> Option<&QueryNode> {
        self.nodes.get(id.index()).filter(|node| node.id == id)
    }

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

    pub fn has_bounded_executable_root_join(&self) -> bool {
        matches!(analyze_bounded_join_plan_support(self), BoundedJoinPlanSupport::Root)
    }

    pub fn has_bounded_executable_join_pipeline(&self) -> bool {
        analyze_bounded_join_pipeline(self).is_ok()
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub const fn index(self) -> usize {
        self.0
    }

    pub(crate) const fn from_index_for_internal_use(index: usize) -> Self {
        Self(index)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryNode {
    pub id: NodeId,
    pub kind: PlanNode,
    pub facts: PlanFacts,
}

impl QueryNode {
    pub fn output_schema(&self) -> &SchemaRef {
        &self.facts.output_schema
    }

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
                "{padding}{} Projection input={} expr_count={} expressions={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.expressions.len(),
                format_exprs(&node.expressions),
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

#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode {
    Source(Box<SourceNode>),
    Filter(FilterNode),
    Projection(ProjectionNode),
    Aggregate(AggregateNode),
    Sort(SortNode),
    Limit(LimitNode),
    Join(JoinNode),
}

impl PlanNode {
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

#[derive(Debug, Clone, PartialEq)]
pub struct PlanFacts {
    pub output_schema: SchemaRef,
    pub output_ordering: Vec<SortExpr>,
    pub placement: DataPlacement,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataPlacement {
    Device,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaBehavior {
    SourceOutput,
    PreservesInput,
    Projection,
    Aggregate,
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

#[derive(Debug, Clone, PartialEq)]
pub enum ResultSink {
    GpuResident { input: NodeId },
    HostArrow { input: NodeId, schema: SchemaRef },
}

impl ResultSink {
    pub fn input(&self) -> NodeId {
        match self {
            Self::GpuResident { input } | Self::HostArrow { input, .. } => *input,
        }
    }

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

#[derive(Debug, Clone, Default)]
pub struct QueryPlanBuilder {
    nodes: Vec<QueryNode>,
}

impl QueryPlanBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_source(&mut self, source: SourceNode) -> NodeId {
        let output_schema = source.output_schema().clone();
        self.push(PlanNode::Source(Box::new(source)), PlanFacts::device(output_schema, SchemaBehavior::SourceOutput))
    }

    pub fn add_filter(&mut self, input: NodeId, predicate: Expr, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = self.node(input).facts.output_ordering.clone();
        self.push(
            PlanNode::Filter(FilterNode { input, predicate }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        )
    }

    pub fn add_projection(&mut self, input: NodeId, expressions: Vec<Expr>, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        self.push(
            PlanNode::Projection(ProjectionNode { input, expressions }),
            PlanFacts::device(output_schema, SchemaBehavior::Projection),
        )
    }

    pub fn add_aggregate(&mut self, input: NodeId, spec: AggregateSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        self.push(
            PlanNode::Aggregate(AggregateNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::Aggregate),
        )
    }

    pub fn add_sort(&mut self, input: NodeId, spec: SortSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = spec.expressions.clone();
        self.push(
            PlanNode::Sort(SortNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        )
    }

    pub fn add_limit(&mut self, input: NodeId, spec: LimitSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(input);
        let output_ordering = self.node(input).facts.output_ordering.clone();
        self.push(
            PlanNode::Limit(LimitNode { input, spec }),
            PlanFacts::device(output_schema, SchemaBehavior::PreservesInput).with_output_ordering(output_ordering),
        )
    }

    pub fn add_join(&mut self, left: NodeId, right: NodeId, spec: JoinSpec, output_schema: SchemaRef) -> NodeId {
        self.assert_input_exists(left);
        self.assert_input_exists(right);
        self.push(
            PlanNode::Join(JoinNode { inputs: [left, right], spec }),
            PlanFacts::device(output_schema, SchemaBehavior::Join),
        )
    }

    pub fn finish_with_host_sink(self, root: NodeId) -> QueryPlan {
        let schema = self.node(root).facts.output_schema.clone();
        self.finish(root, ResultSink::HostArrow { input: root, schema })
    }

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

#[derive(Debug, Clone, PartialEq)]
pub struct FilterNode {
    pub input: NodeId,
    pub predicate: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionNode {
    pub input: NodeId,
    pub expressions: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateNode {
    pub input: NodeId,
    pub spec: AggregateSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SortNode {
    pub input: NodeId,
    pub spec: SortSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LimitNode {
    pub input: NodeId,
    pub spec: LimitSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinNode {
    pub inputs: [NodeId; 2],
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinSpec {
    pub join_type: JoinType,
    pub keys: Vec<JoinKey>,
    pub residual_predicate: Option<Expr>,
    pub output_projection: Vec<JoinOutputColumn>,
    pub null_equality: JoinNullEquality,
    pub build_side: JoinSide,
    pub build_artifact_id: Option<HashJoinBuildArtifactId>,
}

impl JoinSpec {
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

    pub fn with_residual_predicate(mut self, predicate: Expr) -> Self {
        self.residual_predicate = Some(predicate);
        self
    }

    pub fn with_output_projection(mut self, output_projection: Vec<JoinOutputColumn>) -> Self {
        self.output_projection = output_projection;
        self
    }

    pub fn with_null_equality(mut self, null_equality: JoinNullEquality) -> Self {
        self.null_equality = null_equality;
        self
    }

    pub fn with_build_side(mut self, build_side: JoinSide) -> Self {
        self.build_side = build_side;
        self
    }

    pub fn with_build_artifact_id(mut self, id: impl Into<String>) -> Self {
        self.build_artifact_id = Some(HashJoinBuildArtifactId::new(id));
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HashJoinBuildArtifactId(String);

impl HashJoinBuildArtifactId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for HashJoinBuildArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinKey {
    pub left: ColumnRef,
    pub right: ColumnRef,
}

impl JoinKey {
    pub fn new(left: ColumnRef, right: ColumnRef) -> Self {
        Self { left, right }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinOutputColumn {
    pub side: JoinSide,
    pub column: ColumnRef,
    pub output_name: String,
}

impl JoinOutputColumn {
    pub fn new(side: JoinSide, column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self { side, column, output_name: output_name.into() }
    }

    pub fn left(column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self::new(JoinSide::Left, column, output_name)
    }

    pub fn right(column: ColumnRef, output_name: impl Into<String>) -> Self {
        Self::new(JoinSide::Right, column, output_name)
    }
}

impl fmt::Display for JoinOutputColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{} AS {}", self.side, self.column.name(), self.output_name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    Inner,
    Left,
    Full,
    LeftSemi,
    LeftAnti,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinNullEquality {
    NullsDoNotMatch,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinSide {
    Left,
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
