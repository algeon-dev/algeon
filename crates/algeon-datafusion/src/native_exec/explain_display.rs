//! Display-only projection of native execution wrappers into per-node trees.
//!
//! Native operators are never DataFusion `ExecutionPlan::children()` during
//! optimization or execution. During `EXPLAIN` and server diagnostics, this
//! projection wraps `AlgeonNativeExec` nodes in non-executable display nodes
//! whose children mirror the native IR graph so DataFusion's indent and tree
//! renderers format every native operator at its true tree depth.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, Statistics, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, StatisticsArgs,
        StatisticsContext, metrics::MetricsSet,
    },
};
use query_planning::{
    dynamic_filter::DynamicFilterRoute,
    expr::{AggregateExpr, Expr, GroupingSetsSpec, SortExpr},
    plan::{JoinOutputColumn, NodeId, PlanNode, QueryNode, QueryPlan, ResultSink, ShiftDirection, WindowFunction},
    source::{LocalParquetSource, SourceNode},
};

use super::{AlgeonNativeExec, common::conservative_properties, producer::gpu_resident_producer_input_name};

/// Projects `plan` into a display-only `ExecutionPlan` tree where each
/// [`AlgeonNativeExec`] expands its native IR nodes as child nodes.
///
/// Plans that contain no [`AlgeonNativeExec`] are returned unchanged.
pub(crate) fn project_plan_for_display(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if let Some(native_exec) = plan.downcast_ref::<AlgeonNativeExec>() {
        return project_native_exec(native_exec);
    }

    let original_children = plan.children();
    if original_children.is_empty() {
        return Arc::clone(plan);
    }

    let mut changed = false;
    let mut projected_children = Vec::with_capacity(original_children.len());
    for child in original_children {
        let projected = project_plan_for_display(child);
        if !Arc::ptr_eq(&projected, child) {
            changed = true;
        }
        projected_children.push(projected);
    }

    if !changed {
        return Arc::clone(plan);
    }

    Arc::new(DisplayForwardNode { inner: Arc::clone(plan), children: projected_children })
}

/// Projects a borrowed `ExecutionPlan` when it contains an [`AlgeonNativeExec`].
///
/// Returns `None` when no [`AlgeonNativeExec`] appears in the subtree so
/// callers can render `plan` directly without extra allocation.
pub(crate) fn project_plan_for_display_ref(plan: &dyn ExecutionPlan) -> Option<Arc<dyn ExecutionPlan>> {
    if let Some(native_exec) = plan.downcast_ref::<AlgeonNativeExec>() {
        return Some(project_native_exec(native_exec));
    }

    let original_children = plan.children();
    if original_children.is_empty() {
        return None;
    }

    let mut changed = false;
    let mut projected_children = Vec::with_capacity(original_children.len());
    for child in original_children {
        let projected = project_plan_for_display(child);
        if !Arc::ptr_eq(&projected, child) {
            changed = true;
        }
        projected_children.push(projected);
    }

    if !changed {
        return None;
    }

    Some(Arc::new(DisplaySnapshotNode::from_plan(plan, projected_children)))
}

fn project_native_exec(native_exec: &AlgeonNativeExec) -> Arc<dyn ExecutionPlan> {
    let plan = native_exec.native_plan();
    let mut gpu_inputs_by_name = HashMap::<String, Arc<dyn ExecutionPlan>>::with_capacity(native_exec.gpu_inputs.len());
    let mut unmatched_gpu_inputs = Vec::new();
    for child in &native_exec.gpu_inputs {
        let projected = project_plan_for_display(child);
        if let Some(input_name) = gpu_resident_producer_input_name(child.as_ref()) {
            gpu_inputs_by_name.insert(input_name, projected);
        } else {
            unmatched_gpu_inputs.push(projected);
        }
    }

    let mut visited_nodes = HashSet::new();
    let mut expanded_gpu_inputs = HashSet::new();
    let root_node =
        build_ir_display_node(plan, plan.root_id(), &mut visited_nodes, &gpu_inputs_by_name, &mut expanded_gpu_inputs);

    let sink_node: Arc<dyn ExecutionPlan> = Arc::new(NativeSinkDisplayNode::new(plan, root_node));
    let mut wrapper_children = vec![sink_node];

    // Preserve any GPU input whose name did not match a GpuResidentSource in
    // the IR so diagnostics never hide a child plan.
    for child in &native_exec.gpu_inputs {
        if let Some(input_name) = gpu_resident_producer_input_name(child.as_ref())
            && !expanded_gpu_inputs.contains(&input_name)
            && let Some(projected) = gpu_inputs_by_name.get(&input_name)
        {
            wrapper_children.push(Arc::clone(projected));
        }
    }
    wrapper_children.extend(unmatched_gpu_inputs);

    Arc::new(NativeWrapperDisplayNode {
        properties: Arc::clone(native_exec.properties()),
        metrics: native_exec.metrics(),
        statistics: StatisticsContext::new().compute(native_exec, &StatisticsArgs::new()).ok(),
        name: native_exec.name().to_owned(),
        summary: native_exec.summary_fields(),
        tree_fields: native_exec.tree_render_fields(),
        children: wrapper_children,
    })
}

fn build_ir_display_node(
    plan: &QueryPlan,
    node_id: NodeId,
    visited: &mut HashSet<NodeId>,
    gpu_inputs_by_name: &HashMap<String, Arc<dyn ExecutionPlan>>,
    expanded_gpu_inputs: &mut HashSet<String>,
) -> Arc<dyn ExecutionPlan> {
    let Some(node) = plan.node(node_id) else {
        return Arc::new(NativeMissingNodeDisplay {
            node_id,
            properties: conservative_properties(Arc::clone(plan.output_schema())),
        });
    };

    if !visited.insert(node_id) {
        return Arc::new(NativeSharedRefDisplayNode::new(node));
    }

    let mut children = Vec::new();
    if let PlanNode::Source(source) = &node.kind
        && let SourceNode::GpuResident(gpu_source) = source.as_ref()
        && let Some(producer) = gpu_inputs_by_name.get(&gpu_source.input_name)
    {
        if expanded_gpu_inputs.insert(gpu_source.input_name.clone()) {
            children.push(Arc::clone(producer));
        } else {
            children.push(Arc::new(NativeSharedProducerRefDisplayNode {
                input_name: gpu_source.input_name.clone(),
                properties: conservative_properties(Arc::clone(node.output_schema())),
            }));
        }
    } else {
        for &input_id in node.input_ids() {
            children.push(build_ir_display_node(plan, input_id, visited, gpu_inputs_by_name, expanded_gpu_inputs));
        }
    }

    Arc::new(NativeIrDisplayNode::new(node, children))
}

/// Renders a compact, ordered summary of `plan` for DataFusion's single-box
/// `TreeRender` fallback when `AlgeonNativeExec` is formatted without
/// projection.
///
/// DataFusion parses `TreeRender` lines with `split_once('=')` into a map and
/// renders keys in alphabetical order. Emitting one key with a multiline value
/// preserves node order and avoids splitting `key=value` attributes across
/// unrelated keys.
pub(super) fn write_compact_native_plan_tree(f: &mut fmt::Formatter<'_>, plan: &QueryPlan) -> fmt::Result {
    let mut visited = HashSet::new();
    let mut lines = Vec::new();
    collect_compact_tree_lines(plan, plan.root_id(), 0, &mut visited, &mut lines);
    writeln!(f, "ir={}", lines.join("\n"))
}

fn collect_compact_tree_lines(
    plan: &QueryPlan,
    node_id: NodeId,
    depth: usize,
    visited: &mut HashSet<NodeId>,
    lines: &mut Vec<String>,
) {
    let Some(node) = plan.node(node_id) else {
        return;
    };
    let indent = " ".repeat(depth);
    if !visited.insert(node_id) {
        lines.push(format!("{indent}{} {} (ref)", node.id, operator_kind_name(&node.kind)));
        return;
    }
    lines.push(format!("{indent}{} {}", node.id, operator_kind_name(&node.kind)));
    for &input_id in node.input_ids() {
        collect_compact_tree_lines(plan, input_id, depth + 1, visited, lines);
    }
}

fn display_only_execute_error() -> DataFusionError {
    DataFusionError::Internal("display-only native explain projection node cannot be executed".to_owned())
}

/// Forwards an existing DataFusion `ExecutionPlan`'s display, schema, stats,
/// and metrics while substituting projected children.
///
/// Avoids calling `ExecutionPlan::with_new_children` on real operators whose
/// invariants check child schemas, partitioning, or producer types.
#[derive(Debug)]
struct DisplayForwardNode {
    inner: Arc<dyn ExecutionPlan>,
    children: Vec<Arc<dyn ExecutionPlan>>,
}

impl DisplayAs for DisplayForwardNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt_as(t, f)
    }
}

impl ExecutionPlan for DisplayForwardNode {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.inner.properties()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self { inner: Arc::clone(&self.inner), children }))
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.inner.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        StatisticsContext::new().compute(self.inner.as_ref(), &StatisticsArgs::new().with_partition(partition))
    }
}

/// Snapshot wrapper used when projecting a borrowed `&dyn ExecutionPlan` root
/// (such as inside `DefaultPhysicalPlanner::optimize_physical_plan` observers).
#[derive(Debug)]
struct DisplaySnapshotNode {
    name: String,
    default_text: String,
    verbose_text: String,
    tree_text: String,
    properties: Arc<PlanProperties>,
    metrics: Option<MetricsSet>,
    statistics: Option<Arc<Statistics>>,
    children: Vec<Arc<dyn ExecutionPlan>>,
}

impl DisplaySnapshotNode {
    fn from_plan(plan: &dyn ExecutionPlan, children: Vec<Arc<dyn ExecutionPlan>>) -> Self {
        Self {
            name: plan.name().to_owned(),
            default_text: render_display_as(plan, DisplayFormatType::Default),
            verbose_text: render_display_as(plan, DisplayFormatType::Verbose),
            tree_text: render_display_as(plan, DisplayFormatType::TreeRender),
            properties: Arc::clone(plan.properties()),
            metrics: plan.metrics(),
            statistics: StatisticsContext::new().compute(plan, &StatisticsArgs::new()).ok(),
            children,
        }
    }
}

fn render_display_as(plan: &dyn ExecutionPlan, format: DisplayFormatType) -> String {
    struct Wrapper<'a>(&'a dyn ExecutionPlan, DisplayFormatType);
    impl fmt::Display for Wrapper<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt_as(self.1, f)
        }
    }
    Wrapper(plan, format).to_string()
}

impl DisplayAs for DisplaySnapshotNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default => f.write_str(&self.default_text),
            DisplayFormatType::Verbose => f.write_str(&self.verbose_text),
            DisplayFormatType::TreeRender => f.write_str(&self.tree_text),
        }
    }
}

impl ExecutionPlan for DisplaySnapshotNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            name: self.name.clone(),
            default_text: self.default_text.clone(),
            verbose_text: self.verbose_text.clone(),
            tree_text: self.tree_text.clone(),
            properties: Arc::clone(&self.properties),
            metrics: self.metrics.clone(),
            statistics: self.statistics.clone(),
            children,
        }))
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.metrics.clone()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        if partition.is_none()
            && let Some(statistics) = &self.statistics
        {
            return Ok(Arc::clone(statistics));
        }
        Ok(Arc::new(Statistics::new_unknown(self.schema().as_ref())))
    }
}

/// Display-only wrapper for [`AlgeonNativeExec`] whose children are the
/// projected native IR nodes.
#[derive(Debug)]
struct NativeWrapperDisplayNode {
    properties: Arc<PlanProperties>,
    metrics: Option<MetricsSet>,
    statistics: Option<Arc<Statistics>>,
    name: String,
    summary: String,
    tree_fields: Vec<(&'static str, String)>,
    children: Vec<Arc<dyn ExecutionPlan>>,
}

impl DisplayAs for NativeWrapperDisplayNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "{}: {}", self.name, self.summary)
            },
            DisplayFormatType::TreeRender => {
                for (key, value) in &self.tree_fields {
                    writeln!(f, "{key}={value}")?;
                }
                Ok(())
            },
        }
    }
}

impl ExecutionPlan for NativeWrapperDisplayNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            properties: Arc::clone(&self.properties),
            metrics: self.metrics.clone(),
            statistics: self.statistics.clone(),
            name: self.name.clone(),
            summary: self.summary.clone(),
            tree_fields: self.tree_fields.clone(),
            children,
        }))
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.metrics.clone()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        if partition.is_none()
            && let Some(statistics) = &self.statistics
        {
            return Ok(Arc::clone(statistics));
        }
        Ok(Arc::new(Statistics::new_unknown(self.schema().as_ref())))
    }
}

/// Display-only node representing a native plan's [`ResultSink`].
#[derive(Debug)]
struct NativeSinkDisplayNode {
    default_detail: String,
    verbose_detail: String,
    tree_fields: Vec<(&'static str, String)>,
    properties: Arc<PlanProperties>,
    children: Vec<Arc<dyn ExecutionPlan>>,
}

impl NativeSinkDisplayNode {
    fn new(plan: &QueryPlan, root_child: Arc<dyn ExecutionPlan>) -> Self {
        let routes = plan.dynamic_filter_routes();
        let (mode, materialization, schema_fields) = match plan.sink() {
            ResultSink::GpuResident { .. } => ("gpu_resident", None, None),
            ResultSink::HostArrow { schema, .. } => ("host_arrow", Some("device_to_host"), Some(schema.fields().len())),
            ResultSink::DeferredHostArrow { schema, .. } => {
                ("deferred_host_arrow", Some("frontend"), Some(schema.fields().len()))
            },
        };

        let mut default_detail = format!("mode={mode}");
        if !routes.is_empty() {
            default_detail.push_str(&format!(", dynamic_filter_routes={}", routes.len()));
        }

        let mut verbose_detail = format!("id=sink, input={}, mode={mode}", plan.root_id());
        if let Some(materialization) = materialization {
            verbose_detail.push_str(&format!(", materialization={materialization}"));
        }
        if let Some(fields) = schema_fields {
            verbose_detail.push_str(&format!(", schema_fields={fields}"));
        }
        if !routes.is_empty() {
            let formatted = routes.iter().map(format_dynamic_filter_route_verbose).collect::<Vec<_>>().join("; ");
            verbose_detail.push_str(&format!(", dynamic_filter_routes=[{formatted}]"));
        }

        let mut tree_fields = vec![("mode", mode.to_owned())];
        if let Some(fields) = schema_fields {
            tree_fields.push(("fields", fields.to_string()));
        }
        if !routes.is_empty() {
            tree_fields.push(("dynamic_filters", routes.len().to_string()));
        }

        let properties = conservative_properties(Arc::clone(plan.caller_visible_output_schema()));
        Self { default_detail, verbose_detail, tree_fields, properties, children: vec![root_child] }
    }
}

impl DisplayAs for NativeSinkDisplayNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default => write!(f, "Sink: {}", self.default_detail),
            DisplayFormatType::Verbose => write!(f, "Sink: {}", self.verbose_detail),
            DisplayFormatType::TreeRender => {
                for (key, value) in &self.tree_fields {
                    writeln!(f, "{key}={value}")?;
                }
                Ok(())
            },
        }
    }
}

impl ExecutionPlan for NativeSinkDisplayNode {
    fn name(&self) -> &str {
        "Sink"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            default_detail: self.default_detail.clone(),
            verbose_detail: self.verbose_detail.clone(),
            tree_fields: self.tree_fields.clone(),
            properties: Arc::clone(&self.properties),
            children,
        }))
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }
}

/// Display-only node representing one [`QueryNode`] in a native IR plan.
#[derive(Debug)]
struct NativeIrDisplayNode {
    name: String,
    default_detail: String,
    verbose_detail: String,
    tree_fields: Vec<(&'static str, String)>,
    properties: Arc<PlanProperties>,
    children: Vec<Arc<dyn ExecutionPlan>>,
}

impl NativeIrDisplayNode {
    fn new(node: &QueryNode, children: Vec<Arc<dyn ExecutionPlan>>) -> Self {
        let name = format!("{} {}", node.id, operator_kind_name(&node.kind));
        let default_detail = sanitize_single_line(&format_node_default(node));
        let verbose_detail = sanitize_single_line(&format_node_verbose(node));
        let tree_fields =
            format_node_tree_fields(node).into_iter().map(|(key, value)| (key, sanitize_single_line(&value))).collect();
        let properties = conservative_properties(Arc::clone(node.output_schema()));
        Self { name, default_detail, verbose_detail, tree_fields, properties, children }
    }
}

impl DisplayAs for NativeIrDisplayNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default => {
                if self.default_detail.is_empty() {
                    write!(f, "{}", self.name)
                } else {
                    write!(f, "{}: {}", self.name, self.default_detail)
                }
            },
            DisplayFormatType::Verbose => {
                if self.verbose_detail.is_empty() {
                    write!(f, "{}", self.name)
                } else {
                    write!(f, "{}: {}", self.name, self.verbose_detail)
                }
            },
            DisplayFormatType::TreeRender => {
                for (key, value) in &self.tree_fields {
                    writeln!(f, "{key}={value}")?;
                }
                Ok(())
            },
        }
    }
}

impl ExecutionPlan for NativeIrDisplayNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            name: self.name.clone(),
            default_detail: self.default_detail.clone(),
            verbose_detail: self.verbose_detail.clone(),
            tree_fields: self.tree_fields.clone(),
            properties: Arc::clone(&self.properties),
            children,
        }))
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }
}

/// Leaf placeholder when a DAG node is referenced by more than one consumer
/// within the same native plan.
#[derive(Debug)]
struct NativeSharedRefDisplayNode {
    name: String,
    target_id: NodeId,
    properties: Arc<PlanProperties>,
}

impl NativeSharedRefDisplayNode {
    fn new(node: &QueryNode) -> Self {
        Self {
            name: format!("{} {}", node.id, operator_kind_name(&node.kind)),
            target_id: node.id,
            properties: conservative_properties(Arc::clone(node.output_schema())),
        }
    }
}

impl DisplayAs for NativeSharedRefDisplayNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "{}: shared_ref={}", self.name, self.target_id)
            },
            DisplayFormatType::TreeRender => writeln!(f, "shared_ref={}", self.target_id),
        }
    }
}

impl ExecutionPlan for NativeSharedRefDisplayNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }
}

/// Leaf placeholder when two `GpuResidentSource` nodes inside one fragment
/// reference the same GPU-resident producer input.
#[derive(Debug)]
struct NativeSharedProducerRefDisplayNode {
    input_name: String,
    properties: Arc<PlanProperties>,
}

impl DisplayAs for NativeSharedProducerRefDisplayNode {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "SharedGpuProducerRef: input_name={}", self.input_name)
            },
            DisplayFormatType::TreeRender => writeln!(f, "input={}", self.input_name),
        }
    }
}

impl ExecutionPlan for NativeSharedProducerRefDisplayNode {
    fn name(&self) -> &str {
        "SharedGpuProducerRef"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }
}

/// Defensive fallback if a plan references an out-of-range node id.
#[derive(Debug)]
struct NativeMissingNodeDisplay {
    node_id: NodeId,
    properties: Arc<PlanProperties>,
}

impl DisplayAs for NativeMissingNodeDisplay {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "{} MissingNode", self.node_id)
            },
            DisplayFormatType::TreeRender => writeln!(f, "missing={}", self.node_id),
        }
    }
}

impl ExecutionPlan for NativeMissingNodeDisplay {
    fn name(&self) -> &str {
        "MissingNode"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(&self, _partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(display_only_execute_error())
    }
}

fn operator_kind_name(kind: &PlanNode) -> &'static str {
    match kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(_) => "LocalParquetSource",
            SourceNode::GpuResident(_) => "GpuResidentSource",
            SourceNode::LiteralBatch(_) => "LiteralBatchSource",
        },
        PlanNode::Filter(_) => "Filter",
        PlanNode::Projection(_) => "Projection",
        PlanNode::Aggregate(_) => "Aggregate",
        PlanNode::Window(_) => "Window",
        PlanNode::Sort(_) => "Sort",
        PlanNode::Limit(_) => "Limit",
        PlanNode::Join(_) => "Join",
        PlanNode::Union(_) => "Union",
        PlanNode::EdgeNormalize(_) => "EdgeNormalize",
        PlanNode::GraphAlgorithm(_) => "GraphAlgorithm",
        PlanNode::VectorAlgorithm(_) => "VectorAlgorithm",
        PlanNode::MlAlgorithm(_) => "MlAlgorithm",
    }
}

fn format_node_default(node: &QueryNode) -> String {
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                let row_groups = format_row_group_ratio(source);
                format!(
                    "files={}, row_groups={row_groups}, read_columns={}, predicate={}",
                    source.diagnostics.file_count,
                    source.diagnostics.read_column_count,
                    source.diagnostics.predicate_disposition.as_str(),
                )
            },
            SourceNode::GpuResident(source) => {
                format!("input_name={}, schema_fields={}", source.input_name, source.schema.fields().len())
            },
            SourceNode::LiteralBatch(source) => {
                format!("batches={}, rows={}", source.batches.len(), source.row_count())
            },
        },
        PlanNode::Filter(filter) => filter.predicate.to_string(),
        PlanNode::Projection(projection) => format_exprs_spaced(&projection.expressions().cloned().collect::<Vec<_>>()),
        PlanNode::Aggregate(aggregate) => {
            let mut parts = Vec::new();
            if !aggregate.spec.group_exprs.is_empty() {
                parts.push(format!("groups=[{}]", format_exprs_spaced(&aggregate.spec.group_exprs)));
            }
            if !aggregate.spec.aggr_exprs.is_empty() {
                parts.push(format!("aggrs=[{}]", format_aggregate_exprs_spaced(&aggregate.spec.aggr_exprs)));
            }
            if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
                parts.push(format!("grouping_sets={}", grouping_sets.groups.len()));
            }
            parts.join(", ")
        },
        PlanNode::Window(window) => {
            let mut parts = Vec::new();
            if !window.spec.partition_by.is_empty() {
                parts.push(format!("partitions=[{}]", format_exprs_spaced(&window.spec.partition_by)));
            }
            if !window.spec.order_by.is_empty() {
                parts.push(format!("order=[{}]", format_sort_exprs_spaced(&window.spec.order_by)));
            }
            parts.push(format!("functions=[{}]", format_window_functions_spaced(&window.spec.functions)));
            parts.join(", ")
        },
        PlanNode::Sort(sort) => format!("[{}]", format_sort_exprs_spaced(&sort.spec.expressions)),
        PlanNode::Limit(limit) => format!(
            "skip={}, fetch={}",
            limit.spec.skip,
            limit.spec.fetch.map_or_else(|| "none".to_owned(), |fetch| fetch.to_string()),
        ),
        PlanNode::Join(join) => {
            let mut parts = vec![
                format!("type={}", join.spec.join_type),
                format!("keys={}", join.spec.keys.len()),
                format!("build_side={}", join.spec.build_side),
            ];
            if let Some(residual) = &join.spec.residual_predicate {
                parts.push(format!("residual={residual}"));
            }
            parts.join(", ")
        },
        PlanNode::Union(union) => format!("inputs={}", union.inputs.len()),
        PlanNode::EdgeNormalize(edge) => format!(
            "src_col={}, dst_col={}, add_reverse={}, dedup={}",
            edge.spec.edge_columns.source, edge.spec.edge_columns.destination, edge.spec.add_reverse, edge.spec.dedup,
        ),
        PlanNode::GraphAlgorithm(graph) => {
            format!("algorithm={}, directed={}", graph.spec.algorithm, graph.spec.graph_options.directed)
        },
        PlanNode::VectorAlgorithm(vector) => format!("algorithm={}", vector.spec.algorithm),
        PlanNode::MlAlgorithm(ml) => format!("algorithm={}", ml.spec.algorithm),
    }
}

fn format_node_verbose(node: &QueryNode) -> String {
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                let row_group_stats = source.diagnostics.selected_row_group_stats.as_ref().map_or_else(
                    || "unknown".to_owned(),
                    |stats| {
                        format!(
                            "{}/{} selected_bytes={} total_bytes={}",
                            stats.selected_row_groups,
                            stats.total_row_groups,
                            stats
                                .selected_source_bytes()
                                .map_or_else(|| "unknown".to_owned(), |bytes| bytes.to_string()),
                            stats.total_source_bytes().map_or_else(|| "unknown".to_owned(), |bytes| bytes.to_string()),
                        )
                    },
                );
                format!(
                    "files={}, path={}, format=Parquet, direct_device_read={}, known_bytes={}, selected_row_groups={}, row_group_stats={}, read_columns={}, projected_output_columns={}, pure_filter_columns={}, predicate={}, byte_range={}, partition_columns={}, zero_column_count={}, scan_limit={}, max_rows={}, schema_mapping={}, rejected_features={}",
                    source.diagnostics.file_count,
                    source.files.first().map_or_else(|| "none".to_owned(), |file| file.location.diagnostic_location()),
                    source.diagnostics.direct_device_read,
                    source.diagnostics.known_bytes.map_or_else(|| "unknown".to_owned(), |bytes| bytes.to_string()),
                    source
                        .diagnostics
                        .selected_row_groups
                        .map_or_else(|| "all".to_owned(), |row_groups| row_groups.to_string()),
                    row_group_stats,
                    source.diagnostics.read_column_count,
                    source.diagnostics.projected_output_column_count,
                    source.diagnostics.pure_filter_column_count,
                    source.diagnostics.predicate_disposition.as_str(),
                    source.diagnostics.byte_range_disposition.as_str(),
                    source.diagnostics.partition_column_disposition.as_str(),
                    source.diagnostics.zero_column_count_disposition.as_str(),
                    source.diagnostics.scan_limit_disposition.as_str(),
                    source.max_rows.map_or_else(|| "unknown".to_owned(), |rows| rows.to_string()),
                    source.diagnostics.schema_mapping_disposition.as_str(),
                    source
                        .diagnostics
                        .rejected_features
                        .iter()
                        .map(|feature| feature.as_str())
                        .collect::<Vec<_>>()
                        .join(","),
                )
            },
            SourceNode::GpuResident(source) => {
                format!("input_name={}, schema_fields={}", source.input_name, source.schema.fields().len())
            },
            SourceNode::LiteralBatch(source) => {
                format!(
                    "batches={}, rows={}, schema_fields={}",
                    source.batches.len(),
                    source.row_count(),
                    source.schema.fields().len(),
                )
            },
        },
        PlanNode::Filter(filter) => format!("input={}, predicate={}", filter.input, filter.predicate),
        PlanNode::Projection(projection) => format!(
            "input={}, output_count={}, expressions=[{}]",
            projection.input,
            projection.outputs.len(),
            format_exprs_spaced(&projection.expressions().cloned().collect::<Vec<_>>()),
        ),
        PlanNode::Aggregate(aggregate) => format!(
            "input={}, group_count={}, aggr_count={}, grouping_set_count={}, grouping_id_alias={}, grouping_sets={}, groups=[{}], aggrs=[{}]",
            aggregate.input,
            aggregate.spec.group_exprs.len(),
            aggregate.spec.aggr_exprs.len(),
            aggregate.spec.grouping_sets.as_ref().map_or(0, |grouping_sets| grouping_sets.groups.len()),
            aggregate
                .spec
                .grouping_sets
                .as_ref()
                .map_or("none", |grouping_sets| grouping_sets.grouping_id_alias.as_str()),
            format_grouping_sets(aggregate.spec.grouping_sets.as_ref()),
            format_exprs_spaced(&aggregate.spec.group_exprs),
            format_aggregate_exprs_spaced(&aggregate.spec.aggr_exprs),
        ),
        PlanNode::Window(window) => format!(
            "input={}, partition_count={}, order_count={}, function_count={}, partitions=[{}], order=[{}], functions=[{}]",
            window.input,
            window.spec.partition_by.len(),
            window.spec.order_by.len(),
            window.spec.functions.len(),
            format_exprs_spaced(&window.spec.partition_by),
            format_sort_exprs_spaced(&window.spec.order_by),
            format_window_functions_spaced(&window.spec.functions),
        ),
        PlanNode::Sort(sort) => format!(
            "input={}, expr_count={}, expressions=[{}]",
            sort.input,
            sort.spec.expressions.len(),
            format_sort_exprs_spaced(&sort.spec.expressions),
        ),
        PlanNode::Limit(limit) => format!(
            "input={}, skip={}, fetch={}",
            limit.input,
            limit.spec.skip,
            limit.spec.fetch.map_or_else(|| "none".to_owned(), |fetch| fetch.to_string()),
        ),
        PlanNode::Join(join) => format!(
            "left={}, right={}, type={}, key_count={}, residual={}, null_equality={}, build_side={}, build_artifact={}, output_projection={}",
            join.inputs[0],
            join.inputs[1],
            join.spec.join_type,
            join.spec.keys.len(),
            join.spec.residual_predicate.as_ref().map_or_else(|| "none".to_owned(), ToString::to_string),
            join.spec.null_equality,
            join.spec.build_side,
            join.spec.build_artifact_id.as_ref().map_or_else(|| "none".to_owned(), ToString::to_string),
            format_join_output_projection(&join.spec.output_projection),
        ),
        PlanNode::Union(union) => {
            format!("inputs={}, input_count={}", format_node_ids(&union.inputs), union.inputs.len(),)
        },
        PlanNode::EdgeNormalize(edge) => format!(
            "input={}, src_col={}, dst_col={}, weight_col={}, edge_id_col={}, add_reverse={}, dedup={}, weight_combiner={}, distinct_edge_pairs={}",
            edge.input,
            edge.spec.edge_columns.source,
            edge.spec.edge_columns.destination,
            edge.spec.edge_columns.weight.as_deref().unwrap_or("none"),
            edge.spec.edge_columns.edge_id.as_deref().unwrap_or("none"),
            edge.spec.add_reverse,
            edge.spec.dedup,
            edge.spec.weight_combiner,
            edge.spec.distinct_edge_pairs,
        ),
        PlanNode::GraphAlgorithm(graph) => format!(
            "input={}, side_inputs={}, algorithm={}, src_col={}, dst_col={}, weight_col={}, edge_id_col={}, directed={}, renumber={}, construction_policy={}, output_schema_fields={}",
            graph.edge_input(),
            format_node_ids(&graph.inputs[1..]),
            graph.spec.algorithm,
            graph.spec.edge_columns.source,
            graph.spec.edge_columns.destination,
            graph.spec.edge_columns.weight.as_deref().unwrap_or("none"),
            graph.spec.edge_columns.edge_id.as_deref().unwrap_or("none"),
            graph.spec.graph_options.directed,
            graph.spec.graph_options.renumber,
            graph.spec.graph_options.policy,
            graph.output_schema().fields().len(),
        ),
        PlanNode::VectorAlgorithm(vector) => format!(
            "inputs={}, algorithm={}, output_schema_fields={}",
            format_node_ids(&vector.inputs),
            vector.spec.algorithm,
            vector.output_schema().fields().len(),
        ),
        PlanNode::MlAlgorithm(ml) => format!(
            "inputs={}, algorithm={}, output_schema_fields={}",
            format_node_ids(&ml.inputs),
            ml.spec.algorithm,
            ml.output_schema().fields().len(),
        ),
    }
}

fn format_node_tree_fields(node: &QueryNode) -> Vec<(&'static str, String)> {
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => vec![
                ("files", source.diagnostics.file_count.to_string()),
                ("row_groups", format_row_group_ratio(source)),
                ("cols", source.diagnostics.read_column_count.to_string()),
                ("pred", source.diagnostics.predicate_disposition.as_str().to_owned()),
            ],
            SourceNode::GpuResident(source) => {
                vec![("input", source.input_name.clone()), ("fields", source.schema.fields().len().to_string())]
            },
            SourceNode::LiteralBatch(source) => {
                vec![("batches", source.batches.len().to_string()), ("rows", source.row_count().to_string())]
            },
        },
        PlanNode::Filter(filter) => vec![("pred", filter.predicate.to_string())],
        PlanNode::Projection(projection) => vec![
            ("outputs", projection.outputs.len().to_string()),
            ("exprs", format_exprs_spaced(&projection.expressions().cloned().collect::<Vec<_>>())),
        ],
        PlanNode::Aggregate(aggregate) => {
            let mut fields = Vec::new();
            if !aggregate.spec.group_exprs.is_empty() {
                fields.push(("groups", format_exprs_spaced(&aggregate.spec.group_exprs)));
            }
            if !aggregate.spec.aggr_exprs.is_empty() {
                fields.push(("aggrs", format_aggregate_exprs_spaced(&aggregate.spec.aggr_exprs)));
            }
            fields
        },
        PlanNode::Window(window) => {
            let mut fields = Vec::new();
            if !window.spec.partition_by.is_empty() {
                fields.push(("partitions", format_exprs_spaced(&window.spec.partition_by)));
            }
            if !window.spec.order_by.is_empty() {
                fields.push(("order", format_sort_exprs_spaced(&window.spec.order_by)));
            }
            fields.push(("funcs", window.spec.functions.len().to_string()));
            fields
        },
        PlanNode::Sort(sort) => vec![("order", format_sort_exprs_spaced(&sort.spec.expressions))],
        PlanNode::Limit(limit) => vec![
            ("skip", limit.spec.skip.to_string()),
            ("fetch", limit.spec.fetch.map_or_else(|| "none".to_owned(), |fetch| fetch.to_string())),
        ],
        PlanNode::Join(join) => {
            let mut fields = vec![
                ("type", join.spec.join_type.to_string()),
                ("keys", join.spec.keys.len().to_string()),
                ("build", join.spec.build_side.to_string()),
            ];
            if let Some(residual) = &join.spec.residual_predicate {
                fields.push(("residual", residual.to_string()));
            }
            fields
        },
        PlanNode::Union(union) => vec![("inputs", union.inputs.len().to_string())],
        PlanNode::EdgeNormalize(edge) => vec![
            ("src", edge.spec.edge_columns.source.clone()),
            ("dst", edge.spec.edge_columns.destination.clone()),
            ("dedup", edge.spec.dedup.to_string()),
        ],
        PlanNode::GraphAlgorithm(graph) => vec![
            ("algo", graph.spec.algorithm.to_string()),
            ("directed", graph.spec.graph_options.directed.to_string()),
        ],
        PlanNode::VectorAlgorithm(vector) => vec![("algo", vector.spec.algorithm.to_string())],
        PlanNode::MlAlgorithm(ml) => vec![("algo", ml.spec.algorithm.to_string())],
    }
}

fn format_row_group_ratio(source: &LocalParquetSource) -> String {
    if let Some(stats) = &source.diagnostics.selected_row_group_stats {
        format!("{}/{}", stats.selected_row_groups, stats.total_row_groups)
    } else if let Some(selected) = source.diagnostics.selected_row_groups {
        selected.to_string()
    } else {
        "all".to_owned()
    }
}

fn format_dynamic_filter_route_verbose(route: &DynamicFilterRoute) -> String {
    format!(
        "{}: {}({}) -> {}({}) cols={}->{} lowering={} stage={}",
        route.id,
        route.producer.kind(),
        route.producer.node_id(),
        route.consumer.kind(),
        route.consumer.node_id(),
        route.producer_columns.join(","),
        route.consumer_columns.join(","),
        route.consumer_lowering,
        route.consumer_reduction_stage(),
    )
}

fn format_node_ids(ids: &[NodeId]) -> String {
    if ids.is_empty() { "none".to_owned() } else { ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",") }
}

fn format_exprs_spaced(expressions: &[Expr]) -> String {
    if expressions.is_empty() {
        "none".to_owned()
    } else {
        expressions.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    }
}

fn format_aggregate_exprs_spaced(expressions: &[AggregateExpr]) -> String {
    if expressions.is_empty() {
        "none".to_owned()
    } else {
        expressions
            .iter()
            .map(|expr| {
                let distinct = if expr.distinct { "DISTINCT " } else { "" };
                let filter = expr.filter.as_ref().map(|filter| format!(" FILTER ({filter})")).unwrap_or_default();
                format!("{:?}({distinct}{}){filter} AS {}", expr.function, expr.expr, expr.alias)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn format_grouping_sets(grouping_sets: Option<&GroupingSetsSpec>) -> String {
    let Some(grouping_sets) = grouping_sets else {
        return "none".to_owned();
    };
    if grouping_sets.groups.is_empty() {
        return "empty".to_owned();
    }
    grouping_sets
        .groups
        .iter()
        .map(|group| group.iter().map(|masked| if *masked { "1" } else { "0" }).collect::<Vec<_>>().join(""))
        .collect::<Vec<_>>()
        .join("|")
}

fn format_window_functions_spaced(functions: &[WindowFunction]) -> String {
    if functions.is_empty() {
        "none".to_owned()
    } else {
        functions
            .iter()
            .map(|function| match function {
                WindowFunction::Aggregate { function, expr, output, frame } => {
                    format!("{function:?}({expr}) {frame:?} AS {}", output.name())
                },
                WindowFunction::Rank { output } => format!("RANK() AS {}", output.name()),
                WindowFunction::RowNumber { output } => {
                    format!("ROW_NUMBER()::{:?} AS {}", output.data_type(), output.name())
                },
                WindowFunction::LagLead { expr, offset, default, output, direction } => {
                    let alias = output.name();
                    let name = match direction {
                        ShiftDirection::Lag => "LAG",
                        ShiftDirection::Lead => "LEAD",
                    };
                    match default {
                        Some(value) => format!("{name}({expr}, {offset}, {value}) AS {alias}"),
                        None => format!("{name}({expr}, {offset}) AS {alias}"),
                    }
                },
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn format_sort_exprs_spaced(expressions: &[SortExpr]) -> String {
    if expressions.is_empty() {
        "none".to_owned()
    } else {
        expressions
            .iter()
            .map(|expr| {
                format!(
                    "{} {} NULLS {}",
                    expr.expr,
                    if expr.descending { "DESC" } else { "ASC" },
                    if expr.nulls_first { "FIRST" } else { "LAST" },
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn format_join_output_projection(projection: &[JoinOutputColumn]) -> String {
    if projection.is_empty() {
        "none".to_owned()
    } else {
        projection.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}

fn sanitize_single_line(text: &str) -> String {
    if !text.contains(['\n', '\r']) {
        return text.to_owned();
    }
    text.replace('\r', "\\r").replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        path::{Path, PathBuf},
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    use arrow::{
        array::{Date32Array, Int32Array, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::{
        catalog::Session,
        common::Result as DataFusionResult,
        execution::{
            SessionStateBuilder,
            context::{QueryPlanner, TaskContext},
        },
        logical_expr::LogicalPlan,
        physical_plan::{ExecutionPlan, collect, displayable, limit::GlobalLimitExec},
        physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
        prelude::{ParquetReadOptions, SessionContext},
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
    use query_planning::{
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnRef, Expr, LiteralValue},
        plan::{JoinKey, JoinOutputColumn, JoinSpec, JoinType, ProjectionOutput, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetSource, SourceNode},
    };

    use super::project_plan_for_display;
    use crate::{
        native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
        planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
        session::AlgeonSessionStateBuilderExt,
    };

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let seq = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{nanos}-{seq}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_execution_config() -> AlgeonNativeExecutionConfig {
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).unwrap())
    }

    fn orders_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("customer_id", DataType::Int32, false),
            Field::new("amount", DataType::Int64, false),
            Field::new("order_date", DataType::Date32, false),
        ]))
    }

    fn write_orders_parquet(path: &Path) {
        let schema = orders_schema();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 1, 3])),
                Arc::new(Int64Array::from(vec![100, 200, 150, 300])),
                Arc::new(Date32Array::from(vec![20_500, 20_501, 20_640, 20_650])),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn linear_native_exec(path: &Path) -> Arc<dyn ExecutionPlan> {
        let schema = orders_schema();
        let source = LocalParquetSource::new(
            path.to_path_buf(),
            Arc::clone(&schema),
            vec!["customer_id".to_owned(), "amount".to_owned(), "order_date".to_owned()],
        );
        let mut builder = QueryPlanBuilder::new();
        let scan = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder
            .add_filter(
                scan,
                Expr::Binary {
                    left: Box::new(Expr::Column(ColumnRef::unqualified("order_date"))),
                    op: BinaryOp::GtEq,
                    right: Box::new(Expr::Literal(LiteralValue::Date32(20_635))),
                    output_type: None,
                },
            )
            .unwrap();
        let aggregate_schema = Arc::new(Schema::new(vec![
            Field::new("customer_id", DataType::Int32, false),
            Field::new("sum(orders.amount)", DataType::Int64, true),
        ]));
        let aggregate = builder
            .add_aggregate(
                filter,
                AggregateSpec::new(
                    vec![Expr::Column(ColumnRef::unqualified("customer_id"))],
                    vec![AggregateExpr::new(
                        AggregateFunction::Sum,
                        Expr::Column(ColumnRef::unqualified("amount")),
                        "sum(orders.amount)",
                    )],
                ),
                aggregate_schema,
            )
            .unwrap();
        let projection = builder
            .add_projection(
                aggregate,
                vec![
                    ProjectionOutput::new(
                        Expr::Column(ColumnRef::unqualified("customer_id")),
                        Field::new("customer_id", DataType::Int32, false),
                    ),
                    ProjectionOutput::new(
                        Expr::Alias {
                            expr: Box::new(Expr::Column(ColumnRef::unqualified("sum(orders.amount)"))),
                            alias: "total".to_owned(),
                        },
                        Field::new("total", DataType::Int64, true),
                    ),
                ],
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap().clone_with_deferred_host_materialization_sink();
        Arc::new(super::AlgeonNativeExec::try_new_host_for_test(plan, test_execution_config()).unwrap())
    }

    fn line_indent(rendered: &str, needle: &str) -> usize {
        let line = rendered
            .lines()
            .find(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("expected line containing `{needle}` in:\n{rendered}"));
        line.len() - line.trim_start().len()
    }

    #[test]
    fn projected_display_indents_linear_native_plan_and_renders_separate_tree_boxes() {
        let dir = unique_temp_dir("algeon-explain-display-linear");
        let path = dir.join("orders.parquet");
        write_orders_parquet(&path);

        let native = linear_native_exec(&path);
        let wrapped: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(native, 0, Some(10)));
        let projected = project_plan_for_display(&wrapped);

        let default_text = displayable(projected.as_ref()).indent(false).to_string();
        assert_eq!(line_indent(&default_text, "GlobalLimitExec:"), 0, "{default_text}");
        assert_eq!(line_indent(&default_text, "AlgeonNativeExec:"), 2, "{default_text}");
        assert_eq!(line_indent(&default_text, "Sink: mode=deferred_host_arrow"), 4, "{default_text}");
        assert_eq!(line_indent(&default_text, "n3 Projection:"), 6, "{default_text}");
        assert_eq!(line_indent(&default_text, "n2 Aggregate:"), 8, "{default_text}");
        assert_eq!(line_indent(&default_text, "n1 Filter:"), 10, "{default_text}");
        assert_eq!(line_indent(&default_text, "n0 LocalParquetSource:"), 12, "{default_text}");
        assert!(default_text.contains("sink=deferred_host_arrow"), "{default_text}");
        assert!(!default_text.contains("native_plan="), "{default_text}");

        let verbose_text = displayable(projected.as_ref()).indent(true).to_string();
        assert!(verbose_text.contains("Sink: id=sink, input=n3, mode=deferred_host_arrow"), "{verbose_text}");
        assert!(verbose_text.contains("n3 Projection: input=n2, output_count=2"), "{verbose_text}");
        assert!(verbose_text.contains("n2 Aggregate: input=n1, group_count=1, aggr_count=1"), "{verbose_text}");
        assert!(verbose_text.contains("n1 Filter: input=n0, predicate="), "{verbose_text}");
        assert!(verbose_text.contains("n0 LocalParquetSource: files=1, path="), "{verbose_text}");
        assert!(!verbose_text.contains("native_plan=QueryPlan"), "{verbose_text}");

        let tree_text = displayable(projected.as_ref()).tree_render().to_string();
        let expected_boxes =
            ["AlgeonNativeExec", "Sink", "n3 Projection", "n2 Aggregate", "n1 Filter", "n0 LocalParquetSource"];
        let mut cursor = 0usize;
        for box_name in expected_boxes {
            let found = tree_text[cursor..]
                .find(box_name)
                .unwrap_or_else(|| panic!("expected `{box_name}` after offset {cursor} in:\n{tree_text}"));
            cursor += found + box_name.len();
        }
        assert!(!tree_text.contains("..."), "compact tree boxes should not truncate:\n{tree_text}");

        let execute_err = match projected.execute(0, Arc::new(TaskContext::default())) {
            Err(error) => error,
            Ok(_) => panic!("display-only projection must reject execute()"),
        };
        assert!(execute_err.to_string().contains("display-only"), "{execute_err}");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn projected_display_handles_join_branches_and_dag_shared_nodes() {
        let dir = unique_temp_dir("algeon-explain-display-dag");
        let path = dir.join("orders.parquet");
        write_orders_parquet(&path);

        let schema = orders_schema();
        let source = LocalParquetSource::new(
            path,
            Arc::clone(&schema),
            vec!["customer_id".to_owned(), "amount".to_owned(), "order_date".to_owned()],
        );
        let join_schema = Arc::new(Schema::new(vec![
            Field::new("left_customer_id", DataType::Int32, false),
            Field::new("right_customer_id", DataType::Int32, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let scan = builder.add_source(SourceNode::LocalParquet(source));
        let join = builder
            .add_join(
                scan,
                scan,
                JoinSpec::new(
                    JoinType::Inner,
                    vec![JoinKey::new(
                        ColumnRef::input(0, "customer_id", "customer_id"),
                        ColumnRef::input(1, "customer_id", "customer_id"),
                    )],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "left_customer_id"),
                    JoinOutputColumn::right(ColumnRef::input(1, "customer_id", "customer_id"), "right_customer_id"),
                ]),
                join_schema,
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap().clone_with_deferred_host_materialization_sink();
        let native: Arc<dyn ExecutionPlan> =
            Arc::new(super::AlgeonNativeExec::try_new_host_for_test(plan, test_execution_config()).unwrap());
        let projected = project_plan_for_display(&native);

        let default_text = displayable(projected.as_ref()).indent(false).to_string();
        assert_eq!(line_indent(&default_text, "n1 Join:"), 4, "{default_text}");
        let scan_lines: Vec<&str> =
            default_text.lines().filter(|line| line.contains("n0 LocalParquetSource:")).collect();
        assert_eq!(scan_lines.len(), 2, "{default_text}");
        assert_eq!(scan_lines[0].len() - scan_lines[0].trim_start().len(), 6, "{default_text}");
        assert_eq!(scan_lines[1].len() - scan_lines[1].trim_start().len(), 6, "{default_text}");
        assert!(scan_lines[0].contains("files=1"), "{default_text}");
        assert!(scan_lines[1].contains("shared_ref=n0"), "{default_text}");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn projected_display_attaches_composed_gpu_resident_producers_under_gpu_resident_source() {
        let dir = unique_temp_dir("algeon-explain-display-composed");
        let path = dir.join("orders.parquet");
        write_orders_parquet(&path);

        let schema = orders_schema();
        let producer_plan = {
            let mut builder = QueryPlanBuilder::new();
            let scan = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                path,
                Arc::clone(&schema),
                vec!["customer_id".to_owned(), "amount".to_owned(), "order_date".to_owned()],
            )));
            builder.finish_gpu_resident(scan).unwrap()
        };
        let consumer_plan = {
            let mut builder = QueryPlanBuilder::new();
            let scan =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("frag_in", Arc::clone(&schema))));
            builder.finish_with_host_sink(scan).unwrap().clone_with_deferred_host_materialization_sink()
        };
        let producer: Arc<dyn ExecutionPlan> =
            Arc::new(super::AlgeonNativeExec::try_new_producer_for_test("frag_in", producer_plan).unwrap());
        let composed: Arc<dyn ExecutionPlan> = Arc::new(
            super::AlgeonNativeExec::try_new_composed_for_test(consumer_plan, vec![producer], test_execution_config())
                .unwrap(),
        );

        let projected = project_plan_for_display(&composed);
        let default_text = displayable(projected.as_ref()).indent(false).to_string();
        assert_eq!(line_indent(&default_text, "AlgeonNativeComposedExec:"), 0, "{default_text}");
        assert_eq!(line_indent(&default_text, "Sink: mode=deferred_host_arrow"), 2, "{default_text}");
        assert_eq!(line_indent(&default_text, "n0 GpuResidentSource: input_name=frag_in"), 4, "{default_text}");
        assert_eq!(line_indent(&default_text, "AlgeonNativeGpuProducerExec: input_name=frag_in"), 6, "{default_text}");
        assert_eq!(line_indent(&default_text, "Sink: mode=gpu_resident"), 8, "{default_text}");
        assert_eq!(line_indent(&default_text, "n0 LocalParquetSource:"), 10, "{default_text}");

        fs::remove_dir_all(dir).unwrap();
    }

    async fn collect_explain_rows(ctx: &SessionContext, sql: &str) -> Vec<(String, String)> {
        let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let mut rows = Vec::new();
        for batch in batches {
            let plan_types = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let plans = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for row in 0..batch.num_rows() {
                rows.push((plan_types.value(row).to_owned(), plans.value(row).to_owned()));
            }
        }
        rows
    }

    #[tokio::test]
    async fn sql_explain_formats_native_plan_across_default_verbose_and_tree() {
        let dir = unique_temp_dir("algeon-sql-explain-layout");
        let path = dir.join("orders.parquet");
        write_orders_parquet(&path);

        let state = SessionStateBuilder::new_with_default_features()
            .with_algeon_native(
                AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
            )
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_parquet("orders", path.to_string_lossy().as_ref(), ParquetReadOptions::default()).await.unwrap();

        let query = "SELECT customer_id, SUM(amount) AS total FROM orders WHERE order_date >= DATE '2026-07-01' GROUP BY customer_id";

        let default_rows = collect_explain_rows(&ctx, &format!("EXPLAIN {query}")).await;
        let physical = default_rows
            .iter()
            .find(|(kind, _)| kind == "physical_plan")
            .map(|(_, plan)| plan.as_str())
            .expect("EXPLAIN must emit physical_plan");
        assert!(physical.contains("AlgeonNativeExec:"), "{physical}");
        assert!(physical.contains("Sink: mode=deferred_host_arrow"), "{physical}");
        assert!(physical.contains("Projection:"), "{physical}");
        assert!(physical.contains("Aggregate:"), "{physical}");
        assert!(physical.contains("Filter:"), "{physical}");
        assert!(physical.contains("LocalParquetSource:"), "{physical}");
        assert!(
            line_indent(physical, "Sink:") < line_indent(physical, "Projection:")
                && line_indent(physical, "Projection:") < line_indent(physical, "Aggregate:")
                && line_indent(physical, "Aggregate:") < line_indent(physical, "Filter:")
                && line_indent(physical, "Filter:") < line_indent(physical, "LocalParquetSource:"),
            "{physical}"
        );

        let verbose_rows = collect_explain_rows(&ctx, &format!("EXPLAIN VERBOSE {query}")).await;
        let verbose_physical = verbose_rows
            .iter()
            .find(|(kind, _)| kind == "physical_plan")
            .map(|(_, plan)| plan.as_str())
            .expect("EXPLAIN VERBOSE must emit physical_plan");
        assert!(verbose_physical.contains("Sink: id=sink"), "{verbose_physical}");
        assert!(verbose_physical.contains("LocalParquetSource: files=1, path="), "{verbose_physical}");
        assert!(!verbose_physical.contains("native_plan=QueryPlan"), "{verbose_physical}");
        assert!(
            verbose_rows.iter().any(|(kind, _)| kind.contains("algeon_datafusion_native")),
            "EXPLAIN VERBOSE must retain optimizer rule traces"
        );

        let tree_rows = collect_explain_rows(&ctx, &format!("EXPLAIN FORMAT TREE {query}")).await;
        let tree_plan = tree_rows
            .iter()
            .find(|(kind, _)| kind == "physical_plan")
            .map(|(_, plan)| plan.as_str())
            .expect("EXPLAIN FORMAT TREE must emit physical_plan");
        assert!(tree_plan.contains("AlgeonNativeExec"), "{tree_plan}");
        assert!(tree_plan.contains("Sink"), "{tree_plan}");
        assert!(tree_plan.contains("LocalParquetSource"), "{tree_plan}");
        assert!(tree_plan.contains("Aggregate"), "{tree_plan}");
        assert!(!tree_plan.contains("Sink id:"), "tree renderer must not split raw IR lines on `=`:\n{tree_plan}");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn with_algeon_native_preserves_caller_query_planner() {
        #[derive(Debug)]
        struct CustomQueryPlanner;

        #[async_trait::async_trait]
        impl QueryPlanner for CustomQueryPlanner {
            async fn create_physical_plan(
                &self,
                logical_plan: &LogicalPlan,
                session: &dyn Session,
            ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
                DefaultPhysicalPlanner::default().create_physical_plan(logical_plan, session).await
            }
        }

        let state = SessionStateBuilder::new()
            .with_query_planner(Arc::new(CustomQueryPlanner))
            .with_algeon_native(AlgeonNativeOptimizerConfig::default())
            .build();
        let any = state.query_planner().as_ref() as &dyn std::any::Any;
        assert!(any.is::<CustomQueryPlanner>());
    }
}
