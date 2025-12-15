use std::time::Instant;

use arrow::{array::Array, record_batch::RecordBatch};
use cudf::{
    AggSpec as CudfAggSpec, DataFrame, ExecutionOverrides, ExecutionPlacement, GroupByAgg,
    SortOptions as CudfSortOptions,
    frame::dataframe::ParquetReadOptions,
    ops::{col, lit},
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortSpec},
    metrics::QueryMetrics,
    plan::{AggregateNode, FilterNode, LimitNode, NodeId, PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode},
    runtime::ExecutionContext,
    source::LocalParquetSource,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    GpuResident,
    MaterializeToHost,
}

impl ExecutionMode {
    pub fn for_sink(sink: &ResultSink) -> Self {
        match sink {
            ResultSink::GpuResident { .. } => Self::GpuResident,
            ResultSink::HostArrow { .. } => Self::MaterializeToHost,
        }
    }
}

pub struct GpuDataFrame {
    frame: DataFrame,
    placement: ExecutionPlacement,
}

impl GpuDataFrame {
    pub fn new(frame: DataFrame) -> Self {
        let placement = frame.placement();
        Self { frame, placement }
    }

    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    pub fn into_frame(self) -> DataFrame {
        debug_assert_eq!(self.placement, self.frame.placement());
        self.frame
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.placement
    }

    pub fn num_rows(&self) -> usize {
        self.frame.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.frame.num_columns()
    }
}

pub enum QueryOutput {
    GpuResident(GpuDataFrame),
    HostArrow(RecordBatch),
}

impl QueryOutput {
    pub fn mode(&self) -> ExecutionMode {
        match self {
            Self::GpuResident(_) => ExecutionMode::GpuResident,
            Self::HostArrow(_) => ExecutionMode::MaterializeToHost,
        }
    }

    pub fn as_host_arrow(&self) -> Option<&RecordBatch> {
        match self {
            Self::HostArrow(batch) => Some(batch),
            Self::GpuResident(_) => None,
        }
    }

    pub fn as_gpu_resident(&self) -> Option<&GpuDataFrame> {
        match self {
            Self::GpuResident(frame) => Some(frame),
            Self::HostArrow(_) => None,
        }
    }
}

pub struct QueryExecutionResult {
    output: QueryOutput,
    metrics: QueryMetrics,
}

impl QueryExecutionResult {
    pub fn new(output: QueryOutput, metrics: QueryMetrics) -> Self {
        Self { output, metrics }
    }

    pub fn output(&self) -> &QueryOutput {
        &self.output
    }

    pub fn metrics(&self) -> &QueryMetrics {
        &self.metrics
    }

    pub fn into_parts(self) -> (QueryOutput, QueryMetrics) {
        (self.output, self.metrics)
    }
}

pub fn execute_query_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<QueryExecutionResult> {
    let mut metrics = QueryMetrics::default();
    let frame = execute_node(context, plan, plan.sink().input(), &mut metrics)?;
    let output = match plan.sink() {
        ResultSink::GpuResident { .. } => QueryOutput::GpuResident(frame),
        ResultSink::HostArrow { .. } => QueryOutput::HostArrow(materialize_to_host(frame, &mut metrics)?),
    };
    Ok(QueryExecutionResult::new(output, metrics))
}

pub fn validate_local_parquet_source(source: &LocalParquetSource) -> Result<()> {
    if source.path.as_os_str().is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source path must not be empty",
        ));
    }

    if source.read_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source must name at least one required read column",
        )
        .with_fact("path", source.path.display())
        .with_fact("reason", "empty_read_columns"));
    }

    for (index, row_group) in source.row_groups.iter().flatten().enumerate() {
        if i32::try_from(*row_group).is_err() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet row group index exceeds the cudf reader range",
            )
            .with_fact("path", source.path.display())
            .with_fact("row_group_position", index)
            .with_fact("row_group", row_group)
            .with_fact("max_row_group", i32::MAX));
        }
    }

    Ok(())
}

pub fn read_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_local_parquet_source(source)?;

    let columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let row_groups = cudf_row_groups(source)?;
    let read_timer = Instant::now();
    let options = if let Some(row_groups) = &row_groups {
        ParquetReadOptions::builder()
            .filepath(&source.path)
            .columns(&columns)
            .row_groups(row_groups)
            .with_runtime(context.runtime())
            .build()
    } else {
        ParquetReadOptions::builder().filepath(&source.path).columns(&columns).with_runtime(context.runtime()).build()
    };

    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.read",
            "failed to read native local parquet source with cudf",
            error,
        )
        .with_fact("path", source.path.display())
        .with_fact("read_columns", source.read_column_count())
        .with_fact("row_groups_selected", source.selected_row_group_count())
    })?;
    metrics.parquet_read_time += read_timer.elapsed();
    metrics.record_parquet_source(
        source.file_size_bytes,
        source.selected_row_group_count() as u64,
        source.read_column_count() as u64,
    );
    metrics.record_input_batch(frame.num_rows() as u64);

    Ok(GpuDataFrame::new(frame))
}

fn cudf_row_groups(source: &LocalParquetSource) -> Result<Option<Vec<i32>>> {
    let Some(row_groups) = &source.row_groups else {
        return Ok(None);
    };

    if row_groups.is_empty() {
        return Ok(None);
    }

    row_groups
        .iter()
        .map(|row_group| {
            i32::try_from(*row_group).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.local_parquet_source.row_groups",
                    "local parquet row group index exceeds the cudf reader range",
                )
                .with_fact("path", source.path.display())
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", i32::MAX)
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn execute_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let node = plan.node(id).ok_or_else(|| invalid_node_reference(id, "query plan references a missing node"))?;

    match &node.kind {
        PlanNode::Source(source) => match source {
            crate::source::SourceNode::LocalParquet(source) => read_local_parquet_source(context, source, metrics),
        },
        PlanNode::Filter(filter) => {
            let input = execute_node(context, plan, filter.input, metrics)?;
            execute_filter(context, filter, input, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_node(context, plan, projection.input, metrics)?;
            execute_projection(context, projection, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Aggregate(aggregate) => {
            let input = execute_node(context, plan, aggregate.input, metrics)?;
            execute_aggregate(context, aggregate, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Sort(sort) => {
            let input = execute_node(context, plan, sort.input, metrics)?;
            execute_sort(context, sort, input, metrics)
        },
        PlanNode::Limit(limit) => {
            let input = execute_node(context, plan, limit.input, metrics)?;
            execute_limit(context, limit, input, metrics)
        },
    }
}

fn execute_filter(
    context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let predicate = lower_expr(&node.predicate)?;
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).filter_expr(predicate).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.filter.execute", "failed to execute native cudf filter", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_projection(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let output_field_count = output_schema.fields().len();
    if node.expressions.len() != output_field_count {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection expression count must match output schema field count",
        )
        .with_fact("expression_count", node.expressions.len())
        .with_fact("output_field_count", output_field_count));
    }

    let frame = input.into_frame();
    let output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let exprs = output_names
        .iter()
        .zip(&node.expressions)
        .map(|(name, expr)| lower_expr(expr).map(|expr| (name.as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let output_columns = output_names.iter().map(String::as_str).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output =
        frame.with_execution(&overrides).with_columns(exprs).and_then(|frame| frame.select(output_columns)).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.projection.execute",
                    "failed to execute native cudf projection",
                    error,
                )
            },
        )?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let aggregate = lower_aggregate_spec(&node.spec)?;
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if aggregate.output_names != expected_output_names {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.aggregate.validate",
            "aggregate output names must match output schema fields",
        )
        .with_fact("aggregate_output_names", aggregate.output_names.join(","))
        .with_fact("schema_output_names", expected_output_names.join(",")));
    }

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .with_execution(&overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.execute",
                "failed to execute native cudf aggregate",
                error,
            )
        })?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.rename_keys",
                "failed to rename native cudf aggregate keys",
                error,
            )
        })?;
    }
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_sort(
    context: &ExecutionContext,
    node: &SortNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&node.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).sort_values(sort.columns, options).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.sort.execute", "failed to execute native cudf sort", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_limit(
    context: &ExecutionContext,
    node: &LimitNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (offset, length) = limit_slice(input.num_rows(), node.spec)?;
    if offset == 0 && length == input.num_rows() {
        return Ok(input);
    }

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).slice(offset, length).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.limit.execute", "failed to execute native cudf limit", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn materialize_to_host(input: GpuDataFrame, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    let timer = Instant::now();
    let output = input.into_frame().to_record_batch().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.host_arrow.materialize",
            "failed to materialize native cudf output to host Arrow",
            error,
        )
    })?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

fn execution_overrides(context: &ExecutionContext) -> ExecutionOverrides<'_> {
    ExecutionOverrides::new().with_stream(context.stream()).with_memory_resource(context.memory_resource())
}

fn lower_expr(expr: &Expr) -> Result<cudf::Expr> {
    match expr {
        Expr::Column(name) => Ok(col(name)),
        Expr::Literal(value) => lower_literal(value),
        Expr::Binary { left, op, right } => lower_binary(lower_expr(left)?, *op, lower_expr(right)?),
        Expr::Alias { expr, alias } => Ok(lower_expr(expr)?.alias(alias)),
    }
}

fn lower_literal(value: &LiteralValue) -> Result<cudf::Expr> {
    match value {
        LiteralValue::Boolean(value) => Ok(lit(*value)),
        LiteralValue::Int64(value) => Ok(lit(*value)),
        LiteralValue::UInt64(value) => Ok(lit(*value)),
        LiteralValue::Float64(value) => Ok(lit(*value)),
        LiteralValue::String(_) | LiteralValue::Null => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_literal",
            "native execution supports only non-null numeric and boolean literals",
        )
        .with_fact("literal", value)),
    }
}

fn lower_binary(left: cudf::Expr, op: BinaryOp, right: cudf::Expr) -> Result<cudf::Expr> {
    match op {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Subtract => Ok(left - right),
        BinaryOp::Multiply => Ok(left * right),
        BinaryOp::Divide => Ok(left / right),
        BinaryOp::Eq => Ok(left.eq(right)),
        BinaryOp::NotEq => Ok(left.ne(right)),
        BinaryOp::Lt => Ok(left.lt(right)),
        BinaryOp::LtEq => Ok(left.le(right)),
        BinaryOp::Gt => Ok(left.gt(right)),
        BinaryOp::GtEq => Ok(left.ge(right)),
        BinaryOp::And => Ok(left & right),
        BinaryOp::Or => Ok(left | right),
    }
}

struct LoweredAggregate {
    key_columns: Vec<String>,
    output_key_names: Vec<String>,
    aggregations: Vec<CudfAggSpec>,
    output_names: Vec<String>,
}

fn lower_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    if spec.group_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution currently requires at least one group key",
        ));
    }
    if spec.aggr_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution requires at least one aggregate expression",
        ));
    }

    let mut key_columns = Vec::with_capacity(spec.group_exprs.len());
    let mut output_key_names = Vec::with_capacity(spec.group_exprs.len());
    for expr in &spec.group_exprs {
        let (input_column, output_name) = lower_named_column_expr(expr, "cudf_query_engine.aggregate.lower_key")?;
        key_columns.push(input_column);
        output_key_names.push(output_name);
    }

    let aggregations = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.aggregate.lower_expr")?;
            let function = match expr.function {
                AggregateFunction::Sum => GroupByAgg::Sum,
            };
            Ok(CudfAggSpec::new(input_column, function).with_alias(expr.alias.as_str()))
        })
        .collect::<Result<Vec<_>>>()?;

    let output_names = output_key_names.iter().chain(spec.aggr_exprs.iter().map(|expr| &expr.alias)).cloned().collect();

    Ok(LoweredAggregate { key_columns, output_key_names, aggregations, output_names })
}

struct LoweredSort {
    columns: Vec<String>,
    descending: Vec<bool>,
    nulls_first: Vec<bool>,
}

fn lower_sort_spec(spec: &SortSpec) -> Result<LoweredSort> {
    if spec.expressions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.sort.lower",
            "native execution requires at least one sort expression",
        ));
    }

    let mut columns = Vec::with_capacity(spec.expressions.len());
    let mut descending = Vec::with_capacity(spec.expressions.len());
    let mut nulls_first = Vec::with_capacity(spec.expressions.len());
    for sort_expr in &spec.expressions {
        columns.push(lower_column_expr(&sort_expr.expr, "cudf_query_engine.sort.lower")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    Ok(LoweredSort { columns, descending, nulls_first })
}

fn lower_named_column_expr(expr: &Expr, operation: &'static str) -> Result<(String, String)> {
    match expr {
        Expr::Column(name) => Ok((name.clone(), name.clone())),
        Expr::Alias { expr, alias } => Ok((lower_column_expr(expr, operation)?, alias.clone())),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_column_expr(expr: &Expr, operation: &'static str) -> Result<String> {
    match expr {
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias { expr, .. } => lower_column_expr(expr, operation),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

fn limit_slice(row_count: usize, spec: LimitSpec) -> Result<(usize, usize)> {
    let Some(end) = spec.skip.checked_add(spec.fetch) else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.limit.validate",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", spec.fetch));
    };
    let offset = spec.skip.min(row_count);
    let length = end.min(row_count).saturating_sub(offset);
    Ok((offset, length))
}

fn record_batch_memory_size(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(|array| array.get_array_memory_size()).sum()
}

fn invalid_node_reference(id: NodeId, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "cudf_query_engine.plan.execute", message).with_fact("node", id)
}
