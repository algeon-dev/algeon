//! Adaptive-only runtime evidence stores.
//!
//! These structures are descriptive inputs for adaptive admission. They do not
//! participate in strict bounded contract validation and are intentionally kept
//! outside the mode-agnostic pipeline graph.

use std::{collections::BTreeMap, sync::RwLock};

use arrow::datatypes::DataType;

use crate::{
    expr::{AggregateFunction, BinaryOp, Expr, LiteralValue},
    metrics::{QueryMetrics, TaskRuntimeMetrics},
    pipeline::{CompiledPipeline, PipelineTask},
    plan::{PlanNode, QueryPlan, ResultSink},
    source::SourceNode,
};

/// Structural key used to look up prior adaptive memory observations.
///
/// The key is derived from native IR shape and boundary metadata only. It
/// deliberately avoids SQL text, table names, file paths, literal values, and
/// benchmark query ids so history can inform adaptive evidence without becoming
/// query-specific routing.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdaptiveMemoryHistoryKey(String);

impl AdaptiveMemoryHistoryKey {
    pub fn for_query_plan(plan: &QueryPlan) -> Self {
        let mut parts = vec![
            format!("node_count={}", plan.node_count()),
            format!("root={}", plan.root_id().index()),
            format!("sink={}", sink_signature(plan.sink())),
            format!("output_schema={}", schema_signature(plan.output_schema().fields())),
        ];
        parts.extend(plan.nodes().iter().map(|node| {
            format!(
                "node{}:{}:inputs={}:output_schema={}",
                node.id.index(),
                node_signature(&node.kind),
                node.input_ids().len(),
                schema_signature(node.output_schema().fields())
            )
        }));
        Self(parts.join("|"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Structural key used to look up prior adaptive task memory observations.
///
/// The key is scoped to operator class, a coarse input-basis byte signal, output
/// schema, and expression/operator shape. It intentionally excludes table names,
/// file paths, column names, literals, physical ids, and benchmark query ids.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdaptiveTaskMemoryHistoryKey(String);

impl AdaptiveTaskMemoryHistoryKey {
    pub fn for_plan_task(plan: &QueryPlan, task: &PipelineTask, input_basis_bytes: u64) -> Self {
        Self::for_task_shape(&AdaptiveTaskMemoryHistoryShape::for_plan_task(plan, task), input_basis_bytes)
    }

    pub fn for_task_shape(shape: &AdaptiveTaskMemoryHistoryShape, input_basis_bytes: u64) -> Self {
        Self(format!(
            "operator_kind={}:input_basis_bytes={}:schema={}:expression={}",
            shape.operator_kind, input_basis_bytes, shape.schema_signature, shape.expression_signature
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Reusable structural task shape used to build task memory-history keys.
///
/// This lets runtime code precompute task shape without keeping references to
/// the query plan or compiled pipeline in the reservation tracker.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdaptiveTaskMemoryHistoryShape {
    operator_kind: String,
    schema_signature: String,
    expression_signature: String,
}

impl AdaptiveTaskMemoryHistoryShape {
    pub fn for_plan_task(plan: &QueryPlan, task: &PipelineTask) -> Self {
        let (schema, expression) = task.plan_node().and_then(|node_id| plan.node(node_id)).map_or_else(
            || ("none".to_string(), "none".to_string()),
            |node| (schema_signature(node.output_schema().fields()), node_signature(&node.kind)),
        );
        Self {
            operator_kind: task.kind().as_str().to_string(),
            schema_signature: schema,
            expression_signature: expression,
        }
    }
}

/// One query observation that can be folded into adaptive memory history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveMemoryObservation {
    pub resident_dataframe_peak_bytes: Option<u64>,
    pub reservation_granted_bytes: u64,
    pub source_chunks_committed: u64,
}

/// One compiled task observation that can be folded into adaptive memory history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveTaskMemoryObservation {
    pub reservation_requested_bytes: u64,
    pub reservation_granted_bytes: u64,
    pub observed_peak_bytes: Option<u64>,
    pub retained_bytes: Option<u64>,
    pub reservation_released_bytes: u64,
}

impl AdaptiveTaskMemoryObservation {
    pub fn from_task_metrics(metrics: &TaskRuntimeMetrics) -> Self {
        Self {
            reservation_requested_bytes: metrics.reservation_requested_bytes,
            reservation_granted_bytes: metrics.reservation_granted_bytes,
            observed_peak_bytes: metrics.max_observed_peak_bytes,
            retained_bytes: metrics.max_retained_bytes,
            reservation_released_bytes: metrics.reservation_released_bytes,
        }
    }
}

impl AdaptiveMemoryObservation {
    pub fn from_query_metrics(metrics: &QueryMetrics) -> Self {
        Self {
            resident_dataframe_peak_bytes: metrics.resident_dataframe_peak_bytes,
            reservation_granted_bytes: metrics.reservation_granted_bytes,
            source_chunks_committed: metrics.source_chunks_committed,
        }
    }
}

/// Aggregated adaptive memory evidence for one structural query key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveMemoryHistorySummary {
    pub observation_count: u64,
    pub max_resident_dataframe_peak_bytes: Option<u64>,
    pub max_reservation_granted_bytes: u64,
    pub max_source_chunks_committed: u64,
}

impl AdaptiveMemoryHistorySummary {
    fn record(&mut self, observation: &AdaptiveMemoryObservation) {
        self.observation_count = self.observation_count.saturating_add(1);
        self.max_resident_dataframe_peak_bytes =
            match (self.max_resident_dataframe_peak_bytes, observation.resident_dataframe_peak_bytes) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            };
        self.max_reservation_granted_bytes =
            self.max_reservation_granted_bytes.max(observation.reservation_granted_bytes);
        self.max_source_chunks_committed = self.max_source_chunks_committed.max(observation.source_chunks_committed);
    }
}

/// Aggregated adaptive memory evidence for one structural task key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveTaskMemoryHistorySummary {
    pub observation_count: u64,
    pub max_reservation_requested_bytes: u64,
    pub max_reservation_granted_bytes: u64,
    pub max_observed_peak_bytes: Option<u64>,
    pub max_retained_bytes: Option<u64>,
    pub max_reservation_released_bytes: u64,
}

impl AdaptiveTaskMemoryHistorySummary {
    fn record(&mut self, observation: &AdaptiveTaskMemoryObservation) {
        self.observation_count = self.observation_count.saturating_add(1);
        self.max_reservation_requested_bytes =
            self.max_reservation_requested_bytes.max(observation.reservation_requested_bytes);
        self.max_reservation_granted_bytes =
            self.max_reservation_granted_bytes.max(observation.reservation_granted_bytes);
        self.max_observed_peak_bytes = max_optional(self.max_observed_peak_bytes, observation.observed_peak_bytes);
        self.max_retained_bytes = max_optional(self.max_retained_bytes, observation.retained_bytes);
        self.max_reservation_released_bytes =
            self.max_reservation_released_bytes.max(observation.reservation_released_bytes);
    }

    pub fn recommended_reservation_bytes(&self, conservative_bytes: u64) -> Option<u64> {
        let observed_bytes = max_optional(self.max_observed_peak_bytes, self.max_retained_bytes)?;
        if observed_bytes == 0 || observed_bytes >= conservative_bytes {
            return None;
        }
        Some(add_adaptive_reservation_headroom(observed_bytes).min(conservative_bytes))
    }
}

/// In-memory adaptive history store.
#[derive(Debug, Default)]
pub struct AdaptiveMemoryHistory {
    summaries: RwLock<BTreeMap<AdaptiveMemoryHistoryKey, AdaptiveMemoryHistorySummary>>,
    task_summaries: RwLock<BTreeMap<AdaptiveTaskMemoryHistoryKey, AdaptiveTaskMemoryHistorySummary>>,
}

impl AdaptiveMemoryHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_observation(&self, key: AdaptiveMemoryHistoryKey, observation: AdaptiveMemoryObservation) {
        let mut summaries = self.summaries.write().expect("adaptive memory history lock poisoned");
        summaries.entry(key).or_default().record(&observation);
    }

    pub fn lookup(&self, key: &AdaptiveMemoryHistoryKey) -> Option<AdaptiveMemoryHistorySummary> {
        self.summaries.read().expect("adaptive memory history lock poisoned").get(key).cloned()
    }

    pub fn lookup_query_plan(&self, plan: &QueryPlan) -> Option<AdaptiveMemoryHistorySummary> {
        self.lookup(&AdaptiveMemoryHistoryKey::for_query_plan(plan))
    }

    pub fn record_task_observation(
        &self,
        key: AdaptiveTaskMemoryHistoryKey,
        observation: AdaptiveTaskMemoryObservation,
    ) {
        let mut summaries = self.task_summaries.write().expect("adaptive task memory history lock poisoned");
        summaries.entry(key).or_default().record(&observation);
    }

    pub fn lookup_task(&self, key: &AdaptiveTaskMemoryHistoryKey) -> Option<AdaptiveTaskMemoryHistorySummary> {
        self.task_summaries.read().expect("adaptive task memory history lock poisoned").get(key).cloned()
    }

    pub fn recommended_task_reservation_bytes_for_shape(
        &self,
        shape: &AdaptiveTaskMemoryHistoryShape,
        input_basis_bytes: u64,
        conservative_bytes: u64,
    ) -> Option<u64> {
        self.lookup_task(&AdaptiveTaskMemoryHistoryKey::for_task_shape(shape, input_basis_bytes))?
            .recommended_reservation_bytes(conservative_bytes)
    }

    pub fn recommended_task_reservation_bytes_for_plan_task(
        &self,
        plan: &QueryPlan,
        task: &PipelineTask,
        input_basis_bytes: u64,
        conservative_bytes: u64,
    ) -> Option<u64> {
        self.recommended_task_reservation_bytes_for_shape(
            &AdaptiveTaskMemoryHistoryShape::for_plan_task(plan, task),
            input_basis_bytes,
            conservative_bytes,
        )
    }

    pub fn record_task_observations_from_query_metrics(
        &self,
        plan: &QueryPlan,
        pipeline: &CompiledPipeline,
        metrics: &QueryMetrics,
    ) {
        for task in pipeline.tasks() {
            let task_id = task.id().to_string();
            let Some(task_metrics) = metrics.task_runtime_metrics.get(&task_id) else {
                continue;
            };
            let input_basis_bytes = task_metrics
                .reservation_requested_bytes
                .max(task_metrics.reservation_granted_bytes)
                .max(task_metrics.max_observed_peak_bytes.unwrap_or(0))
                .max(task_metrics.max_retained_bytes.unwrap_or(0));
            self.record_task_observation(
                AdaptiveTaskMemoryHistoryKey::for_plan_task(plan, task, input_basis_bytes),
                AdaptiveTaskMemoryObservation::from_task_metrics(task_metrics),
            );
        }
    }
}

fn max_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn add_adaptive_reservation_headroom(observed_bytes: u64) -> u64 {
    observed_bytes.saturating_add((observed_bytes / 8).max(1))
}

fn sink_signature(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
    }
}

fn node_signature(node: &PlanNode) -> String {
    match node {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => format!(
                "source:local_parquet:format={}:read_columns={}:projected_columns={}:zero_column_count={}",
                match source.diagnostics.format {
                    crate::source::SourceFormat::Parquet => "parquet",
                },
                source.diagnostics.read_column_count,
                source.diagnostics.projected_output_column_count,
                source.diagnostics.zero_column_count_disposition.as_str()
            ),
            SourceNode::GpuResident(_) => "source:gpu_resident".to_string(),
        },
        PlanNode::Filter(node) => format!("filter:predicate={}", expr_signature(&node.predicate)),
        PlanNode::Projection(node) => {
            let outputs = node.outputs.iter().map(|output| {
                format!(
                    "expr={}:type={}:nullable={}",
                    expr_signature(&output.expr),
                    data_type_signature(output.field.data_type()),
                    output.field.is_nullable()
                )
            });
            format!("projection:outputs={}", join_signatures(outputs))
        },
        PlanNode::Aggregate(node) => {
            let groups = node.spec.group_exprs.iter().map(expr_signature);
            let aggregates = node.spec.aggr_exprs.iter().map(|aggregate| {
                format!(
                    "function={}:expr={}",
                    aggregate_function_signature(aggregate.function),
                    expr_signature(&aggregate.expr)
                )
            });
            format!("aggregate:groups={}:aggregates={}", join_signatures(groups), join_signatures(aggregates))
        },
        PlanNode::Window(node) => {
            let partitions = node.spec.partition_by.iter().map(expr_signature);
            let orderings = node.spec.order_by.iter().map(|expr| {
                format!(
                    "expr={}:descending={}:nulls_first={}",
                    expr_signature(&expr.expr),
                    expr.descending,
                    expr.nulls_first
                )
            });
            let functions = node.spec.functions.iter().map(|function| match function {
                crate::plan::WindowFunction::Aggregate { function, expr, frame, .. } => {
                    format!(
                        "aggregate:function={}:expr={}:frame={frame:?}",
                        aggregate_function_signature(*function),
                        expr_signature(expr)
                    )
                },
                crate::plan::WindowFunction::Rank { .. } => "rank".to_string(),
            });
            format!(
                "window:partitions={}:orderings={}:functions={}",
                join_signatures(partitions),
                join_signatures(orderings),
                join_signatures(functions)
            )
        },
        PlanNode::Sort(node) => {
            let expressions = node.spec.expressions.iter().map(|expr| {
                format!(
                    "expr={}:descending={}:nulls_first={}",
                    expr_signature(&expr.expr),
                    expr.descending,
                    expr.nulls_first
                )
            });
            format!("sort:exprs={}", join_signatures(expressions))
        },
        PlanNode::Limit(node) => format!("limit:skip={}:fetch={}", node.spec.skip > 0, node.spec.fetch > 0),
        PlanNode::Join(node) => format!(
            "join:type={}:keys={}:residual={}:null_equality={}:build_side={}:outputs={}",
            node.spec.join_type,
            node.spec.keys.len(),
            node.spec.residual_predicate.as_ref().map_or_else(|| "none".to_string(), expr_signature),
            node.spec.null_equality,
            node.spec.build_side,
            node.spec.output_projection.len()
        ),
        PlanNode::Union(node) => format!("union:inputs={}", node.inputs.len()),
        PlanNode::EdgeNormalize(node) => format!(
            "edge_normalize:add_reverse={}:dedup={}:weight_combiner={}",
            node.spec.add_reverse, node.spec.dedup, node.spec.weight_combiner
        ),
        PlanNode::GraphAlgorithm(node) => format!(
            "graph_algorithm:algorithm={}:directed={}:renumber={}:weight={}:schema_fields={}",
            node.spec.algorithm,
            node.spec.graph_options.directed,
            node.spec.graph_options.renumber,
            node.spec.edge_columns.weight.is_some(),
            node.output_schema().fields().len()
        ),
    }
}

fn schema_signature<'a>(fields: impl IntoIterator<Item = &'a arrow::datatypes::FieldRef>) -> String {
    let fields = fields.into_iter().map(field_signature);
    join_signatures(fields)
}

fn field_signature(field: &arrow::datatypes::FieldRef) -> String {
    format!("type={}:nullable={}", data_type_signature(field.data_type()), field.is_nullable())
}

fn data_type_signature(data_type: &DataType) -> String {
    match data_type {
        DataType::Null => "null".to_string(),
        DataType::Boolean => "bool".to_string(),
        DataType::Int8 => "i8".to_string(),
        DataType::Int16 => "i16".to_string(),
        DataType::Int32 => "i32".to_string(),
        DataType::Int64 => "i64".to_string(),
        DataType::UInt8 => "u8".to_string(),
        DataType::UInt16 => "u16".to_string(),
        DataType::UInt32 => "u32".to_string(),
        DataType::UInt64 => "u64".to_string(),
        DataType::Float16 => "f16".to_string(),
        DataType::Float32 => "f32".to_string(),
        DataType::Float64 => "f64".to_string(),
        DataType::Binary => "binary".to_string(),
        DataType::FixedSizeBinary(bytes) => format!("fixed_size_binary:bytes={bytes}"),
        DataType::LargeBinary => "large_binary".to_string(),
        DataType::BinaryView => "binary_view".to_string(),
        DataType::Utf8 => "utf8".to_string(),
        DataType::LargeUtf8 => "large_utf8".to_string(),
        DataType::Utf8View => "utf8_view".to_string(),
        DataType::Date32 => "date32".to_string(),
        DataType::Date64 => "date64".to_string(),
        DataType::Time32(unit) => format!("time32:{unit:?}"),
        DataType::Time64(unit) => format!("time64:{unit:?}"),
        DataType::Timestamp(unit, timezone) => format!("timestamp:{unit:?}:timezone={}", timezone.is_some()),
        DataType::Duration(unit) => format!("duration:{unit:?}"),
        DataType::Interval(unit) => format!("interval:{unit:?}"),
        DataType::List(field) => format!("list:field={}", field_signature(field)),
        DataType::ListView(field) => format!("list_view:field={}", field_signature(field)),
        DataType::FixedSizeList(field, length) => {
            format!("fixed_size_list:length={length}:field={}", field_signature(field))
        },
        DataType::LargeList(field) => format!("large_list:field={}", field_signature(field)),
        DataType::LargeListView(field) => format!("large_list_view:field={}", field_signature(field)),
        DataType::Struct(fields) => format!("struct:fields={}", schema_signature(fields)),
        DataType::Union(fields, mode) => {
            let fields = fields.iter().map(|(type_id, field)| format!("type_id={type_id}:{}", field_signature(field)));
            format!("union:mode={mode:?}:fields={}", join_signatures(fields))
        },
        DataType::Dictionary(key_type, value_type) => {
            format!("dictionary:key={}:value={}", data_type_signature(key_type), data_type_signature(value_type))
        },
        DataType::Decimal32(precision, scale) => format!("decimal32:precision={precision}:scale={scale}"),
        DataType::Decimal64(precision, scale) => format!("decimal64:precision={precision}:scale={scale}"),
        DataType::Decimal128(precision, scale) => format!("decimal128:precision={precision}:scale={scale}"),
        DataType::Decimal256(precision, scale) => format!("decimal256:precision={precision}:scale={scale}"),
        DataType::Map(field, sorted) => format!("map:sorted={sorted}:field={}", field_signature(field)),
        DataType::RunEndEncoded(run_ends, values) => {
            format!("run_end_encoded:run_ends={}:values={}", field_signature(run_ends), field_signature(values))
        },
    }
}

fn expr_signature(expr: &Expr) -> String {
    match expr {
        Expr::Column(column) => format!("column:qualified={}", column.is_qualified()),
        Expr::Literal(value) => format!("literal:type={}", data_type_signature(&literal_data_type(value))),
        Expr::Not(expr) => format!("not({})", expr_signature(expr)),
        Expr::IsNull(expr) => format!("is_null({})", expr_signature(expr)),
        Expr::IsNotNull(expr) => format!("is_not_null({})", expr_signature(expr)),
        Expr::Cast { expr, target_type } => {
            format!("cast({}:target={})", expr_signature(expr), data_type_signature(target_type))
        },
        Expr::Binary { left, op, right, output_type } => format!(
            "binary:op={}:left={}:right={}:output={}",
            binary_op_signature(*op),
            expr_signature(left),
            expr_signature(right),
            output_type.as_ref().map(data_type_signature).unwrap_or_else(|| "none".to_string())
        ),
        Expr::Like { expr, escape, negated, .. } => {
            format!("like:expr={}:escape={}:negated={}", expr_signature(expr), escape.is_some(), negated)
        },
        Expr::Substring { expr, length, .. } => {
            format!("substring:expr={}:length_present={}", expr_signature(expr), length.is_some())
        },
        Expr::Lower(expr) => format!("lower({})", expr_signature(expr)),
        Expr::Upper(expr) => format!("upper({})", expr_signature(expr)),
        Expr::Concat(exprs) => format!("concat({})", join_signatures(exprs.iter().map(expr_signature))),
        Expr::Abs(expr) => format!("abs({})", expr_signature(expr)),
        Expr::Round { expr, decimal_places, output_type } => format!(
            "round:expr={}:decimal_places={}:output={}",
            expr_signature(expr),
            decimal_places,
            data_type_signature(output_type)
        ),
        Expr::InList { expr, list, negated } => {
            let list = list.iter().map(expr_signature);
            format!("in_list:expr={}:list={}:negated={}", expr_signature(expr), join_signatures(list), negated)
        },
        Expr::Case { branches, else_expr } => {
            let branches = branches
                .iter()
                .map(|branch| format!("when={}:then={}", expr_signature(&branch.when), expr_signature(&branch.then)));
            format!("case:branches={}:else={}", join_signatures(branches), expr_signature(else_expr))
        },
        Expr::Extract { field, expr } => format!("extract:field={field}:expr={}", expr_signature(expr)),
        Expr::Alias { expr, .. } => format!("alias({})", expr_signature(expr)),
    }
}

fn literal_data_type(value: &LiteralValue) -> DataType {
    value.data_type()
}

fn binary_op_signature(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Subtract => "subtract",
        BinaryOp::Multiply => "multiply",
        BinaryOp::Divide => "divide",
        BinaryOp::Modulo => "modulo",
        BinaryOp::BitwiseAnd => "bitwise_and",
        BinaryOp::BitwiseOr => "bitwise_or",
        BinaryOp::BitwiseXor => "bitwise_xor",
        BinaryOp::ShiftLeft => "shift_left",
        BinaryOp::ShiftRight => "shift_right",
        BinaryOp::Eq => "eq",
        BinaryOp::NotEq => "not_eq",
        BinaryOp::Lt => "lt",
        BinaryOp::LtEq => "lt_eq",
        BinaryOp::Gt => "gt",
        BinaryOp::GtEq => "gt_eq",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
    }
}

fn aggregate_function_signature(function: AggregateFunction) -> &'static str {
    match function {
        AggregateFunction::Sum => "sum",
        AggregateFunction::Min => "min",
        AggregateFunction::Max => "max",
        AggregateFunction::Count => "count",
        AggregateFunction::Avg => "avg",
        AggregateFunction::StddevSamp => "stddev_samp",
    }
}

fn join_signatures(signatures: impl IntoIterator<Item = String>) -> String {
    let signatures = signatures.into_iter().collect::<Vec<_>>();
    if signatures.is_empty() { "none".to_string() } else { signatures.join(",") }
}
