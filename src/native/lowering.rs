use std::sync::Arc;

use arrow::datatypes::{Field, Schema, SchemaRef};
use cudf_query_engine::{
    AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
    AggregateSpec as NativeAggregateSpec, BinaryOp as NativeBinaryOp, Expr as NativeExpr, LimitSpec as NativeLimitSpec,
    LiteralValue as NativeLiteralValue, LocalParquetSource as NativeLocalParquetSource, QueryPlan as NativeQueryPlan,
    QueryPlanBuilder as NativeQueryPlanBuilder, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec,
    SourceNode as NativeSourceNode, ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
};
use datafusion::{
    common::ScalarValue,
    datasource::physical_plan::ParquetSource,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, PhysicalSortExpr,
        expressions::{BinaryExpr, CastExpr, Column, Literal, NotExpr, TryCastExpr},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};

use super::{
    normalize::NormalizedNativePlan,
    source::{
        local_parquet_files, local_parquet_source_diagnostics, required_parquet_column_requirements,
        validate_slice1_parquet_scan,
    },
    validate::{
        aggregate_is_count_star_only, is_supported_native_cast, is_supported_native_type,
        native_zero_column_count_source_is_safe, validate_aggregate, validate_schema, validate_sort,
    },
};
use crate::policy::FallbackReason;

pub(super) fn lower_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    lower_slice1_plan(normalized)
}

fn lower_slice1_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    let source_exec = normalized.source_exec;
    let filter = normalized.filter;
    let pre_aggregate_projection = normalized.pre_aggregate_projection;
    let aggregate = normalized.aggregate;
    let post_aggregate_projection = normalized.post_aggregate_projection;
    let sort = normalized.sort;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;

    validate_schema(source_exec.schema())?;
    if let Some(filter) = filter {
        validate_schema(filter.schema())?;
    }
    if let Some(projection) = pre_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    validate_schema(aggregate.schema())?;
    validate_schema(Arc::clone(&normalized.aggregate_output_schema))?;
    if let Some(projection) = post_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    if let Some(sort) = sort {
        validate_schema(sort.schema())?;
    }

    validate_slice1_parquet_scan(file_scan, parquet_source, filter, source_exec.schema())?;
    validate_aggregate(aggregate, pre_aggregate_projection.is_some(), normalized.aggregate_allows_partial)?;
    if let Some(sort) = sort {
        validate_sort(sort)?;
    } else if !native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate) {
        return Err(FallbackReason::UnsupportedOperator);
    }

    let column_requirements =
        required_parquet_column_requirements(file_scan, filter, pre_aggregate_projection, aggregate)?;
    let native_files = local_parquet_files(file_scan, parquet_source, filter, source_exec.schema())?;
    let zero_column_count_safe = native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate);
    let mut source = if zero_column_count_safe && column_requirements.read_columns.is_empty() {
        NativeLocalParquetSource::zero_column_count(native_files)
    } else {
        NativeLocalParquetSource::from_files(
            native_files,
            source_exec.schema(),
            column_requirements.read_columns.clone(),
        )
        .with_column_roles(
            column_requirements.projected_output_columns.clone(),
            column_requirements.pure_filter_columns.clone(),
        )
        .with_column_source_refs(
            column_requirements.read_column_file_indices.clone(),
            column_requirements.projected_output_column_source_refs.clone(),
            column_requirements.pure_filter_column_source_refs.clone(),
        )
        .with_partition_columns(column_requirements.partition_columns.clone())
    };
    source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        source_exec.schema(),
        Some(&column_requirements),
        zero_column_count_safe,
    );

    let mut builder = NativeQueryPlanBuilder::new();
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    let (filter_node, filter_output_schema) = if let Some(filter) = filter {
        (
            builder.add_filter(
                source_node,
                lower_physical_expr(filter.predicate(), &source_exec.schema())?,
                filter.schema(),
            ),
            filter.schema(),
        )
    } else {
        (source_node, source_exec.schema())
    };
    let (aggregate_input_node, aggregate_input_schema, synthetic_aggregate_args) = if zero_column_count_safe {
        (filter_node, filter_output_schema, None)
    } else if let Some(projection) = pre_aggregate_projection {
        let projection_node =
            builder.add_projection(filter_node, lower_projection_exprs(projection)?, projection.schema());
        if let Some(synthetic_projection) =
            synthetic_aggregate_input_projection_if_needed(aggregate, &projection.schema())?
        {
            let synthetic_projection_node = builder.add_projection(
                projection_node,
                synthetic_projection.expressions,
                Arc::clone(&synthetic_projection.schema),
            );
            (synthetic_projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
        } else {
            (projection_node, projection.schema(), None)
        }
    } else {
        let synthetic_projection = synthetic_aggregate_input_projection(aggregate, &filter_output_schema)?;
        let projection_node = builder.add_projection(
            filter_node,
            synthetic_projection.expressions,
            Arc::clone(&synthetic_projection.schema),
        );
        (projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
    };
    let aggregate_spec = if zero_column_count_safe {
        lower_zero_column_count_aggregate_spec(aggregate)?
    } else {
        lower_aggregate_spec(aggregate, &aggregate_input_schema, synthetic_aggregate_args.as_deref())?
    };
    let aggregate_node =
        builder.add_aggregate(aggregate_input_node, aggregate_spec, Arc::clone(&normalized.aggregate_output_schema));
    let sort_input_node = if let Some(projection) = post_aggregate_projection {
        builder.add_projection(aggregate_node, lower_projection_exprs(projection)?, projection.schema())
    } else {
        aggregate_node
    };
    let sorted_or_projected_node = if let Some(sort) = sort {
        builder.add_sort(sort_input_node, lower_sort_spec(sort)?, sort.schema())
    } else {
        sort_input_node
    };
    let output_node = if let Some(limit_fetch) = normalized.limit_fetch {
        let output_schema = normalized.sort.expect("native limit requires preceding sort").schema();
        builder.add_limit(sorted_or_projected_node, NativeLimitSpec::fetch(limit_fetch), output_schema)
    } else {
        sorted_or_projected_node
    };

    Ok(builder.finish_with_host_sink(output_node))
}

struct SyntheticAggregateInputProjection {
    expressions: Vec<NativeExpr>,
    schema: SchemaRef,
    aggregate_arg_aliases: Vec<String>,
}

fn synthetic_aggregate_input_projection(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
) -> Result<SyntheticAggregateInputProjection, FallbackReason> {
    let mut expressions = Vec::new();
    let mut fields = Vec::new();

    for (group_expr, alias) in aggregate.group_expr().expr() {
        expressions.push(alias_if_needed(lower_physical_expr(group_expr, input_schema)?, alias));
        fields.push(physical_expr_field(group_expr, input_schema, alias)?);
    }

    let mut aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
    for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
        let expressions_for_aggregate = aggregate_expr.expressions();
        let [arg] = expressions_for_aggregate.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        // cuDF aggregate lowering consumes columns, so SQL aggregate arguments
        // such as SUM(m00 + m01) are first materialized as synthetic projection columns.
        let alias = unique_synthetic_aggregate_arg_alias(index, input_schema, &fields);
        expressions.push(alias_if_needed(lower_physical_expr(arg, input_schema)?, alias.as_str()));
        fields.push(physical_expr_field(arg, input_schema, alias.as_str())?);
        aggregate_arg_aliases.push(alias);
    }

    Ok(SyntheticAggregateInputProjection { expressions, schema: Arc::new(Schema::new(fields)), aggregate_arg_aliases })
}

fn synthetic_aggregate_input_projection_if_needed(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
) -> Result<Option<SyntheticAggregateInputProjection>, FallbackReason> {
    let needs_projection = aggregate.aggr_expr().iter().any(|aggregate_expr| {
        let expressions = aggregate_expr.expressions();
        !matches!(expressions.as_slice(), [arg] if arg.as_any().downcast_ref::<Column>().is_some())
    });
    if !needs_projection {
        return Ok(None);
    }

    let mut expressions =
        input_schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();

    let mut aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
    for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
        let expressions_for_aggregate = aggregate_expr.expressions();
        let [arg] = expressions_for_aggregate.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        let alias = unique_synthetic_aggregate_arg_alias(index, input_schema, &fields);
        expressions.push(alias_if_needed(lower_physical_expr(arg, input_schema)?, alias.as_str()));
        fields.push(physical_expr_field(arg, input_schema, alias.as_str())?);
        aggregate_arg_aliases.push(alias);
    }

    Ok(Some(SyntheticAggregateInputProjection {
        expressions,
        schema: Arc::new(Schema::new(fields)),
        aggregate_arg_aliases,
    }))
}

fn unique_synthetic_aggregate_arg_alias(index: usize, input_schema: &SchemaRef, fields: &[Field]) -> String {
    let base = format!("__cudf_aggregate_arg_{index}");
    if synthetic_alias_is_available(base.as_str(), input_schema, fields) {
        return base;
    }

    for suffix in 1usize.. {
        let alias = format!("{base}_{suffix}");
        if synthetic_alias_is_available(alias.as_str(), input_schema, fields) {
            return alias;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn synthetic_alias_is_available(alias: &str, input_schema: &SchemaRef, fields: &[Field]) -> bool {
    input_schema.index_of(alias).is_err() && fields.iter().all(|field| field.name() != alias)
}

fn physical_expr_field(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    name: &str,
) -> Result<Field, FallbackReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&data_type) {
        return Err(FallbackReason::UnsupportedType);
    }
    let nullable = expr.nullable(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    Ok(Field::new(name, data_type, nullable))
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
    synthetic_aggregate_args: Option<&[String]>,
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
        .enumerate()
        .map(|(index, expr)| {
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            let lowered_arg = synthetic_aggregate_args
                .and_then(|aliases| aliases.get(index))
                .map_or_else(|| lower_physical_expr(arg, input_schema), |alias| Ok(NativeExpr::column(alias)))?;
            Ok(NativeAggregateExpr::new(lower_native_aggregate_function(expr.fun().name())?, lowered_arg, expr.name()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(NativeAggregateSpec::new(group_exprs, aggr_exprs))
}

fn lower_zero_column_count_aggregate_spec(aggregate: &AggregateExec) -> Result<NativeAggregateSpec, FallbackReason> {
    if !aggregate_is_count_star_only(aggregate) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let expr = &aggregate.aggr_expr()[0];
    Ok(NativeAggregateSpec::new(
        Vec::new(),
        vec![NativeAggregateExpr::new(
            NativeAggregateFunction::Sum,
            NativeExpr::column(NATIVE_ZERO_COLUMN_COUNT_COLUMN),
            expr.name(),
        )],
    ))
}

pub(super) fn lower_native_aggregate_function(name: &str) -> Result<NativeAggregateFunction, FallbackReason> {
    if name.eq_ignore_ascii_case("sum") {
        Ok(NativeAggregateFunction::Sum)
    } else if name.eq_ignore_ascii_case("min") {
        Ok(NativeAggregateFunction::Min)
    } else if name.eq_ignore_ascii_case("max") {
        Ok(NativeAggregateFunction::Max)
    } else if name.eq_ignore_ascii_case("count") {
        Ok(NativeAggregateFunction::Count)
    } else if name.eq_ignore_ascii_case("avg") {
        Ok(NativeAggregateFunction::Avg)
    } else {
        Err(FallbackReason::UnsupportedOperator)
    }
}

fn lower_sort_spec(sort: &SortExec) -> Result<NativeSortSpec, FallbackReason> {
    let expressions = sort.expr().iter().map(lower_sort_expr).collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

pub(super) fn lower_sort_expr(sort_expr: &PhysicalSortExpr) -> Result<NativeSortExpr, FallbackReason> {
    let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(NativeSortExpr::new(
        NativeExpr::column(column.name()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}

pub(super) fn lower_physical_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
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
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(NativeExpr::not(lower_physical_expr(not.arg(), input_schema)?));
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        return Ok(NativeExpr::cast(lower_physical_expr(cast.expr(), input_schema)?, cast.cast_type().clone()));
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
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
        ScalarValue::Int8(Some(value)) => NativeLiteralValue::Int8(*value),
        ScalarValue::Int16(Some(value)) => NativeLiteralValue::Int16(*value),
        ScalarValue::Int32(Some(value)) => NativeLiteralValue::Int32(*value),
        ScalarValue::Int64(Some(value)) => NativeLiteralValue::Int64(*value),
        ScalarValue::UInt8(Some(value)) => NativeLiteralValue::UInt8(*value),
        ScalarValue::UInt16(Some(value)) => NativeLiteralValue::UInt16(*value),
        ScalarValue::UInt32(Some(value)) => NativeLiteralValue::UInt32(*value),
        ScalarValue::UInt64(Some(value)) => NativeLiteralValue::UInt64(*value),
        ScalarValue::Float32(Some(value)) => NativeLiteralValue::Float32(*value),
        ScalarValue::Float64(Some(value)) => NativeLiteralValue::Float64(*value),
        ScalarValue::Utf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => NativeLiteralValue::Utf8(value.clone()),
        ScalarValue::LargeUtf8(Some(value)) => NativeLiteralValue::LargeUtf8(value.clone()),
        ScalarValue::Decimal128(Some(raw), precision, scale) => {
            NativeLiteralValue::Decimal128 { raw: *raw, precision: *precision, scale: *scale }
        },
        ScalarValue::Date32(Some(value)) => NativeLiteralValue::Date32(*value),
        ScalarValue::TimestampSecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: arrow::datatypes::TimeUnit::Second }
        },
        ScalarValue::TimestampMillisecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: arrow::datatypes::TimeUnit::Millisecond }
        },
        ScalarValue::TimestampMicrosecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: arrow::datatypes::TimeUnit::Microsecond }
        },
        ScalarValue::TimestampNanosecond(Some(raw), None) => {
            NativeLiteralValue::Timestamp { raw: *raw, unit: arrow::datatypes::TimeUnit::Nanosecond }
        },
        ScalarValue::DurationSecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: arrow::datatypes::TimeUnit::Second }
        },
        ScalarValue::DurationMillisecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: arrow::datatypes::TimeUnit::Millisecond }
        },
        ScalarValue::DurationMicrosecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: arrow::datatypes::TimeUnit::Microsecond }
        },
        ScalarValue::DurationNanosecond(Some(raw)) => {
            NativeLiteralValue::Duration { raw: *raw, unit: arrow::datatypes::TimeUnit::Nanosecond }
        },
        ScalarValue::Boolean(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Boolean),
        ScalarValue::Int8(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Int8),
        ScalarValue::Int16(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Int16),
        ScalarValue::Int32(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Int32),
        ScalarValue::Int64(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Int64),
        ScalarValue::UInt8(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::UInt8),
        ScalarValue::UInt16(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::UInt16),
        ScalarValue::UInt32(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::UInt32),
        ScalarValue::UInt64(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::UInt64),
        ScalarValue::Float32(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Float32),
        ScalarValue::Float64(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Float64),
        ScalarValue::Utf8(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Utf8),
        ScalarValue::Utf8View(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Utf8View),
        ScalarValue::LargeUtf8(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::LargeUtf8),
        ScalarValue::Decimal128(None, precision, scale) => {
            NativeLiteralValue::Null(arrow::datatypes::DataType::Decimal128(*precision, *scale))
        },
        ScalarValue::Date32(None) => NativeLiteralValue::Null(arrow::datatypes::DataType::Date32),
        ScalarValue::TimestampSecond(None, None) => {
            NativeLiteralValue::Null(arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None))
        },
        ScalarValue::TimestampMillisecond(None, None) => NativeLiteralValue::Null(
            arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
        ),
        ScalarValue::TimestampMicrosecond(None, None) => NativeLiteralValue::Null(
            arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
        ),
        ScalarValue::TimestampNanosecond(None, None) => NativeLiteralValue::Null(
            arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
        ),
        ScalarValue::DurationSecond(None) => {
            NativeLiteralValue::Null(arrow::datatypes::DataType::Duration(arrow::datatypes::TimeUnit::Second))
        },
        ScalarValue::DurationMillisecond(None) => {
            NativeLiteralValue::Null(arrow::datatypes::DataType::Duration(arrow::datatypes::TimeUnit::Millisecond))
        },
        ScalarValue::DurationMicrosecond(None) => {
            NativeLiteralValue::Null(arrow::datatypes::DataType::Duration(arrow::datatypes::TimeUnit::Microsecond))
        },
        ScalarValue::DurationNanosecond(None) => {
            NativeLiteralValue::Null(arrow::datatypes::DataType::Duration(arrow::datatypes::TimeUnit::Nanosecond))
        },
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
