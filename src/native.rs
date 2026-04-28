use std::{collections::BTreeSet, path::PathBuf, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use cudf_query_engine::{
    AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
    AggregateSpec as NativeAggregateSpec, BinaryOp as NativeBinaryOp, Expr as NativeExpr, LimitSpec as NativeLimitSpec,
    LiteralValue as NativeLiteralValue, LocalParquetSource as NativeLocalParquetSource, QueryPlan as NativeQueryPlan,
    QueryPlanBuilder as NativeQueryPlanBuilder, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec,
    SourceNode as NativeSourceNode,
};
use datafusion::{
    common::ScalarValue,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{FileScanConfig, FileSource, ParquetSource},
        source::DataSourceExec,
    },
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, PhysicalSortExpr,
        expressions::{BinaryExpr, Column, Literal},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};

use crate::{
    policy::FallbackReason,
    report::{GpuCandidate, PlanningReport},
};

const SLICE1_SHAPE: &str = "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))";

#[derive(Debug, Clone)]
pub struct NativeLoweringResult {
    pub plans: Vec<NativeQueryPlan>,
    pub report: PlanningReport,
}

impl NativeLoweringResult {
    pub fn candidate_count(&self) -> usize {
        self.report.outcomes().len()
    }

    pub fn rewritten_count(&self) -> usize {
        self.report.rewritten_count()
    }

    pub fn rejected_count(&self) -> usize {
        self.report.rejected_count()
    }
}

pub fn lower_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
    let candidate = GpuCandidate::with_operator_count(SLICE1_SHAPE, plan.schema().fields().len(), 5);
    match lower_slice1_plan(plan.as_ref()) {
        Ok(native_plan) => {
            let mut report = PlanningReport::new();
            report.push_rewritten(candidate);
            NativeLoweringResult { plans: vec![native_plan], report }
        },
        Err(reason) => {
            let mut report = PlanningReport::new();
            report.push_rejected(candidate, reason);
            NativeLoweringResult { plans: Vec::new(), report }
        },
    }
}

fn lower_slice1_plan(plan: &dyn ExecutionPlan) -> Result<NativeQueryPlan, FallbackReason> {
    let limit = plan.as_any().downcast_ref::<LocalLimitExec>().ok_or(FallbackReason::UnsupportedOperator)?;
    let sort = limit.input().as_any().downcast_ref::<SortExec>().ok_or(FallbackReason::UnsupportedOperator)?;
    let aggregate = sort.input().as_any().downcast_ref::<AggregateExec>().ok_or(FallbackReason::UnsupportedOperator)?;
    let projection =
        aggregate.input().as_any().downcast_ref::<ProjectionExec>().ok_or(FallbackReason::UnsupportedOperator)?;
    let filter = projection.input().as_any().downcast_ref::<FilterExec>().ok_or(FallbackReason::UnsupportedOperator)?;
    let source_exec =
        filter.input().as_any().downcast_ref::<DataSourceExec>().ok_or(FallbackReason::UnsupportedOperator)?;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;

    validate_schema(source_exec.schema())?;
    validate_schema(filter.schema())?;
    validate_schema(projection.schema())?;
    validate_schema(aggregate.schema())?;
    validate_schema(sort.schema())?;
    validate_schema(limit.schema())?;

    validate_slice1_parquet_scan(file_scan, parquet_source)?;
    validate_aggregate(aggregate)?;
    validate_sort(sort)?;

    let read_columns = required_parquet_read_columns(file_scan, filter, projection, aggregate)?;
    let file = single_full_local_file(file_scan)?;

    let source = NativeLocalParquetSource::new(
        local_object_path_to_filesystem(file.object_meta.location.as_ref()),
        source_exec.schema(),
        read_columns,
    )
    .with_file_size_bytes(file.object_meta.size);

    let mut builder = NativeQueryPlanBuilder::new();
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    let filter_node = builder.add_filter(
        source_node,
        lower_physical_expr(filter.predicate(), &source_exec.schema())?,
        filter.schema(),
    );
    let projection_node = builder.add_projection(filter_node, lower_projection_exprs(projection)?, projection.schema());
    let aggregate_node = builder.add_aggregate(
        projection_node,
        lower_aggregate_spec(aggregate, &projection.schema())?,
        aggregate.schema(),
    );
    let sort_node = builder.add_sort(aggregate_node, lower_sort_spec(sort)?, sort.schema());
    let limit_node = builder.add_limit(sort_node, NativeLimitSpec::fetch(limit.fetch()), limit.schema());

    Ok(builder.finish_with_host_sink(limit_node))
}

fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
) -> Result<(), FallbackReason> {
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if !file_scan.table_partition_cols().is_empty() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if file_scan.expr_adapter_factory.is_some() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if file_scan.limit.is_some() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if file_scan.preserve_order {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if !file_scan.output_ordering.is_empty() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if parquet_source.filter().is_some() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    single_full_local_file(file_scan)?;
    Ok(())
}

fn single_full_local_file(file_scan: &FileScanConfig) -> Result<&PartitionedFile, FallbackReason> {
    let [group] = file_scan.file_groups.as_slice() else {
        return Err(FallbackReason::PlanPropertyMismatch);
    };
    let [file] = group.files() else {
        return Err(FallbackReason::PlanPropertyMismatch);
    };
    if !file.partition_values.is_empty() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if file.range.is_some() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if file.ordering.is_some() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    Ok(file)
}

fn validate_schema(schema: SchemaRef) -> Result<(), FallbackReason> {
    for field in schema.fields() {
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
    }
    Ok(())
}

fn is_supported_native_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn validate_aggregate(aggregate: &AggregateExec) -> Result<(), FallbackReason> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let group_by = aggregate.group_expr();
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    for aggregate_expr in aggregate.aggr_expr() {
        if aggregate_expr.is_distinct() || !aggregate_expr.order_bys().is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        if !aggregate_expr.fun().name().eq_ignore_ascii_case("sum") {
            return Err(FallbackReason::UnsupportedOperator);
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        if arg.as_any().downcast_ref::<Column>().is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }
    for (group_expr, _alias) in group_by.expr() {
        if group_expr.as_any().downcast_ref::<Column>().is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }
    Ok(())
}

fn validate_sort(sort: &SortExec) -> Result<(), FallbackReason> {
    if sort.fetch().is_some() {
        return Err(FallbackReason::PlanPropertyMismatch);
    }
    if sort.expr().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if sort.expr().iter().any(|sort_expr| sort_expr.expr.as_any().downcast_ref::<Column>().is_none()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(())
}

fn lower_projection_exprs(projection: &ProjectionExec) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .expr()
        .iter()
        .map(|projection_expr| lower_projection_expr(projection_expr, &projection.input().schema()))
        .collect()
}

fn lower_projection_expr(
    projection_expr: &ProjectionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let expr = lower_physical_expr(&projection_expr.expr, input_schema)?;
    Ok(alias_if_needed(expr, projection_expr.alias.as_str()))
}

fn lower_aggregate_spec(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
) -> Result<NativeAggregateSpec, FallbackReason> {
    let group_exprs = aggregate
        .group_expr()
        .expr()
        .iter()
        .map(|(expr, alias)| {
            let native_expr = lower_physical_expr(expr, input_schema)?;
            Ok(alias_if_needed(native_expr, alias))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let aggr_exprs = aggregate
        .aggr_expr()
        .iter()
        .map(|expr| {
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            Ok(NativeAggregateExpr::new(
                NativeAggregateFunction::Sum,
                lower_physical_expr(arg, input_schema)?,
                expr.name(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(NativeAggregateSpec::new(group_exprs, aggr_exprs))
}

fn lower_sort_spec(sort: &SortExec) -> Result<NativeSortSpec, FallbackReason> {
    let expressions = sort.expr().iter().map(lower_sort_expr).collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

fn lower_sort_expr(sort_expr: &PhysicalSortExpr) -> Result<NativeSortExpr, FallbackReason> {
    let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(NativeSortExpr::new(
        NativeExpr::column(column.name()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}

fn lower_physical_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        validate_column_type(column, input_schema)?;
        return Ok(NativeExpr::column(column.name()));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_physical_expr(binary.left(), input_schema)?;
        let right = lower_physical_expr(binary.right(), input_schema)?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let field = input_schema.field_with_name(column.name()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(field.data_type()) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, FallbackReason> {
    let literal = match value {
        ScalarValue::Boolean(Some(value)) => NativeLiteralValue::Boolean(*value),
        ScalarValue::Int8(Some(value)) => NativeLiteralValue::Int64(i64::from(*value)),
        ScalarValue::Int16(Some(value)) => NativeLiteralValue::Int64(i64::from(*value)),
        ScalarValue::Int32(Some(value)) => NativeLiteralValue::Int64(i64::from(*value)),
        ScalarValue::Int64(Some(value)) => NativeLiteralValue::Int64(*value),
        ScalarValue::UInt8(Some(value)) => NativeLiteralValue::UInt64(u64::from(*value)),
        ScalarValue::UInt16(Some(value)) => NativeLiteralValue::UInt64(u64::from(*value)),
        ScalarValue::UInt32(Some(value)) => NativeLiteralValue::UInt64(u64::from(*value)),
        ScalarValue::UInt64(Some(value)) => NativeLiteralValue::UInt64(*value),
        ScalarValue::Float32(Some(value)) => NativeLiteralValue::Float64(f64::from(*value)),
        ScalarValue::Float64(Some(value)) => NativeLiteralValue::Float64(*value),
        _ => return Err(FallbackReason::UnsupportedExpression),
    };
    Ok(NativeExpr::literal(literal))
}

fn lower_binary_operator(op: &Operator) -> Result<NativeBinaryOp, FallbackReason> {
    match op {
        Operator::Plus => Ok(NativeBinaryOp::Add),
        Operator::Minus => Ok(NativeBinaryOp::Subtract),
        Operator::Multiply => Ok(NativeBinaryOp::Multiply),
        Operator::Divide => Ok(NativeBinaryOp::Divide),
        Operator::Eq => Ok(NativeBinaryOp::Eq),
        Operator::NotEq => Ok(NativeBinaryOp::NotEq),
        Operator::Lt => Ok(NativeBinaryOp::Lt),
        Operator::LtEq => Ok(NativeBinaryOp::LtEq),
        Operator::Gt => Ok(NativeBinaryOp::Gt),
        Operator::GtEq => Ok(NativeBinaryOp::GtEq),
        Operator::And => Ok(NativeBinaryOp::And),
        Operator::Or => Ok(NativeBinaryOp::Or),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn alias_if_needed(expr: NativeExpr, alias: &str) -> NativeExpr {
    match &expr {
        NativeExpr::Column(name) if name == alias => expr,
        _ => expr.alias(alias),
    }
}

fn required_parquet_read_columns(
    file_scan: &FileScanConfig,
    filter: &FilterExec,
    projection: &ProjectionExec,
    aggregate: &AggregateExec,
) -> Result<Vec<String>, FallbackReason> {
    let file_schema = file_scan.file_schema();
    let mut required = BTreeSet::new();

    if let Some(projection) = file_scan.file_source().projection() {
        for index in projection.column_indices() {
            if index >= file_schema.fields().len() {
                return Err(FallbackReason::PlanPropertyMismatch);
            }
            required.insert(file_schema.field(index).name().clone());
        }
    }

    collect_required_columns(filter.predicate(), &mut required)?;
    for projection_expr in projection.expr() {
        collect_required_columns(&projection_expr.expr, &mut required)?;
    }
    for (group_expr, _alias) in aggregate.group_expr().expr() {
        collect_required_columns(group_expr, &mut required)?;
    }

    for column in &required {
        if file_schema.index_of(column).is_err() {
            return Err(FallbackReason::PlanPropertyMismatch);
        }
    }

    Ok(required.into_iter().collect())
}

fn collect_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        required.insert(column.name().to_string());
        return Ok(());
    }
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), required)?;
        collect_required_columns(binary.right(), required)?;
        return Ok(());
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn local_object_path_to_filesystem(location: &str) -> PathBuf {
    PathBuf::from("/").join(location)
}
