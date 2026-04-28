use std::fmt;

use arrow::datatypes::SchemaRef;

use crate::{
    expr::{AggregateSpec, Expr, LimitSpec, SortExpr, SortSpec},
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
            PlanNode::Source(SourceNode::LocalParquet(source)) => {
                writeln!(
                    f,
                    "{padding}{} LocalParquetSource path={} read_columns={} placement={} schema_behavior={}",
                    self.id,
                    source.path.display(),
                    source.read_columns.len(),
                    self.facts.placement,
                    self.facts.schema_behavior
                )
            },
            PlanNode::Filter(node) => writeln!(
                f,
                "{padding}{} Filter input={} predicate={} placement={} schema_behavior={}",
                self.id, node.input, node.predicate, self.facts.placement, self.facts.schema_behavior
            ),
            PlanNode::Projection(node) => writeln!(
                f,
                "{padding}{} Projection input={} expr_count={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.expressions.len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Aggregate(node) => writeln!(
                f,
                "{padding}{} Aggregate input={} group_count={} aggr_count={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.group_exprs.len(),
                node.spec.aggr_exprs.len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Sort(node) => writeln!(
                f,
                "{padding}{} Sort input={} expr_count={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.expressions.len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Limit(node) => writeln!(
                f,
                "{padding}{} Limit input={} skip={} fetch={} placement={} schema_behavior={}",
                self.id, node.input, node.spec.skip, node.spec.fetch, self.facts.placement, self.facts.schema_behavior
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode {
    Source(SourceNode),
    Filter(FilterNode),
    Projection(ProjectionNode),
    Aggregate(AggregateNode),
    Sort(SortNode),
    Limit(LimitNode),
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
}

impl fmt::Display for SchemaBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SourceOutput => "source_output",
            Self::PreservesInput => "preserves_input",
            Self::Projection => "projection",
            Self::Aggregate => "aggregate",
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
        self.push(PlanNode::Source(source), PlanFacts::device(output_schema, SchemaBehavior::SourceOutput))
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
