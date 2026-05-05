use std::sync::Arc;

use arrow::datatypes::{Field, Schema, SchemaRef};
use cudf_query_engine::{
    AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
    AggregateSpec as NativeAggregateSpec, BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen,
    DatePart as NativeDatePart, Expr as NativeExpr, LimitSpec as NativeLimitSpec,
    LocalParquetSource as NativeLocalParquetSource, QueryPlan as NativeQueryPlan,
    QueryPlanBuilder as NativeQueryPlanBuilder, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec,
    SourceNode as NativeSourceNode, ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
};
use datafusion::{
    common::ScalarValue,
    datasource::physical_plan::ParquetSource,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
            TryCastExpr,
        },
        projection::ProjectionExprs,
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};

use super::{
    literal::{NativeLiteralContext, native_literal_value},
    normalize::NormalizedNativePlan,
    source::{
        local_parquet_files, local_parquet_source_diagnostics, native_source_output_schema,
        required_parquet_column_requirements, validate_slice1_parquet_scan,
    },
    validate::{
        aggregate_is_count_star_only, is_supported_native_cast, is_supported_native_type,
        native_zero_column_count_source_is_safe, validate_aggregate, validate_physical_expr, validate_schema,
        validate_sort,
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
    let column_requirements =
        required_parquet_column_requirements(file_scan, filter, pre_aggregate_projection, aggregate)?;
    let zero_column_count_safe = native_zero_column_count_source_is_safe(filter, pre_aggregate_projection, aggregate);
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let source_output_schema = if zero_column_count_safe {
        source_exec.schema()
    } else {
        native_source_output_schema(file_scan, &column_requirements)?
    };

    validate_schema(Arc::clone(&source_output_schema))?;
    validate_required_operator_schemas(normalized, filter, pre_aggregate_projection, post_aggregate_projection, sort)?;
    validate_required_scalar_expressions(
        scan_projection,
        &table_schema,
        source_exec.schema(),
        &column_requirements.scan_output_columns,
        filter,
        pre_aggregate_projection,
        aggregate,
        post_aggregate_projection,
    )?;
    validate_slice1_parquet_scan(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    validate_aggregate(aggregate, pre_aggregate_projection.is_some(), normalized.aggregate_allows_partial)?;
    if let Some(sort) = sort {
        validate_sort(sort)?;
    }

    let native_files =
        local_parquet_files(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    let mut source = if zero_column_count_safe && column_requirements.read_columns.is_empty() {
        NativeLocalParquetSource::zero_column_count(native_files)
    } else {
        NativeLocalParquetSource::from_files(
            native_files,
            Arc::clone(&source_output_schema),
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
        Arc::clone(&table_schema),
        Some(&column_requirements),
        zero_column_count_safe,
    );

    let mut builder = NativeQueryPlanBuilder::new();
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection
        && !zero_column_count_safe
    {
        let required_scan_projection =
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns);
        if required_scan_projection_requires_native_projection(&required_scan_projection, &table_schema) {
            let expressions = lower_required_file_source_projection_exprs(&required_scan_projection, &table_schema)?;
            let schema = required_scan_projection_schema(&required_scan_projection, source_exec.schema())?;
            (builder.add_projection(source_node, expressions, Arc::clone(&schema)), schema)
        } else {
            (source_node, Arc::clone(&source_output_schema))
        }
    } else {
        (source_node, Arc::clone(&source_output_schema))
    };
    let filter_input_schema = scan_projection.map_or_else(|| source_exec.schema(), |_| source_exec.schema());
    let (filter_node, filter_output_schema) = if let Some(filter) = filter {
        let filtered_node = builder.add_filter(
            scan_output_node,
            lower_physical_expr(filter.predicate(), &scan_output_schema)?,
            Arc::clone(&scan_output_schema),
        );
        if let Some(projection) = filter.projection() {
            (
                builder.add_projection(
                    filtered_node,
                    lower_filter_projection_exprs(projection, &filter_input_schema)?,
                    filter.schema(),
                ),
                filter.schema(),
            )
        } else {
            (filtered_node, scan_output_schema)
        }
    } else {
        (scan_output_node, scan_output_schema)
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

fn validate_required_operator_schemas(
    normalized: &NormalizedNativePlan<'_>,
    _filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    post_aggregate_projection: Option<&ProjectionExec>,
    sort: Option<&SortExec>,
) -> Result<(), FallbackReason> {
    if let Some(projection) = pre_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    validate_schema(normalized.aggregate.schema())?;
    validate_schema(Arc::clone(&normalized.aggregate_output_schema))?;
    if let Some(projection) = post_aggregate_projection {
        validate_schema(projection.schema())?;
    }
    if let Some(sort) = sort {
        validate_schema(sort.schema())?;
    }
    Ok(())
}

fn validate_required_scalar_expressions(
    scan_projection: Option<&ProjectionExprs>,
    table_schema: &SchemaRef,
    scan_output_schema: SchemaRef,
    required_scan_output_columns: &[String],
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    pre_aggregate_projection: Option<&ProjectionExec>,
    aggregate: &AggregateExec,
    post_aggregate_projection: Option<&ProjectionExec>,
) -> Result<(), FallbackReason> {
    if let Some(scan_projection) = scan_projection {
        for projection_expr in
            required_scan_projection(scan_projection, scan_output_schema, required_scan_output_columns)
        {
            validate_physical_expr(&projection_expr.expr, table_schema)?;
        }
    }
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }
    if let Some(projection) = pre_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    for aggregate_expr in aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr(&expr, &aggregate.input().schema())?;
        }
    }
    if let Some(projection) = post_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    Ok(())
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

struct RequiredScanProjectionExpr<'a> {
    expr: &'a Arc<dyn PhysicalExpr>,
    alias: &'a str,
}

fn required_scan_projection<'a>(
    projection: &'a ProjectionExprs,
    scan_output_schema: SchemaRef,
    required_scan_output_columns: &[String],
) -> Vec<RequiredScanProjectionExpr<'a>> {
    projection
        .iter()
        .filter(|projection_expr| {
            required_scan_output_columns.iter().any(|column| column == &projection_expr.alias)
                || scan_output_schema.index_of(projection_expr.alias.as_str()).is_ok_and(|index| {
                    scan_output_schema
                        .fields()
                        .get(index)
                        .is_some_and(|field| required_scan_output_columns.iter().any(|column| column == field.name()))
                })
        })
        .map(|projection_expr| RequiredScanProjectionExpr {
            expr: &projection_expr.expr,
            alias: projection_expr.alias.as_str(),
        })
        .collect()
}

fn lower_required_file_source_projection_exprs(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .iter()
        .map(|projection_expr| {
            let expr = lower_physical_expr(projection_expr.expr, input_schema)?;
            Ok(alias_if_needed(expr, projection_expr.alias))
        })
        .collect()
}

fn required_scan_projection_requires_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> bool {
    projection.iter().any(|projection_expr| {
        let Some(column) = projection_expr.expr.as_any().downcast_ref::<Column>() else {
            return true;
        };
        input_schema.fields().get(column.index()).is_none_or(|field| field.name() != projection_expr.alias)
    })
}

fn required_scan_projection_schema(
    projection: &[RequiredScanProjectionExpr<'_>],
    scan_output_schema: SchemaRef,
) -> Result<SchemaRef, FallbackReason> {
    let fields = projection
        .iter()
        .map(|projection_expr| {
            scan_output_schema
                .field_with_name(projection_expr.alias)
                .map(|field| field.clone())
                .map_err(|_error| FallbackReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

fn lower_filter_projection_exprs(
    projection: &[usize],
    input_schema: &SchemaRef,
) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .iter()
        .map(|index| {
            let field = input_schema.fields().get(*index).ok_or(FallbackReason::UnsupportedExpression)?;
            Ok(NativeExpr::column(field.name()))
        })
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
        return Ok(NativeExpr::column(resolved_column_name(column, input_schema)?));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if matches!(binary.op(), Operator::Modulo) {
            validate_modulo_operand_type(binary.left(), input_schema)?;
            validate_modulo_operand_type(binary.right(), input_schema)?;
        }
        let left = lower_physical_expr(binary.left(), input_schema)?;
        let right = lower_physical_expr(binary.right(), input_schema)?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(NativeExpr::not(lower_physical_expr(not.arg(), input_schema)?));
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_physical_expr(is_null.arg(), input_schema)?));
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_physical_expr(is_not_null.arg(), input_schema)?));
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        let source_type =
            cast.expr().data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
        if !is_supported_native_cast(&source_type, cast.cast_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
        return Ok(NativeExpr::cast(lower_physical_expr(cast.expr(), input_schema)?, cast.cast_type().clone()));
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        return lower_like_expr(like, input_schema);
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        return lower_in_list_expr(in_list, input_schema);
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        return lower_case_expr(case, input_schema);
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        return lower_scalar_function_expr(function, input_schema);
    }
    if expr.as_any().downcast_ref::<TryCastExpr>().is_some() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_like_expr(like: &LikeExpr, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    if like.case_insensitive() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let pattern = literal_string(like.pattern())?;
    Ok(if like.negated() {
        NativeExpr::not_like_escape(lower_physical_expr(like.expr(), input_schema)?, pattern, "\\")
    } else {
        NativeExpr::like_escape(lower_physical_expr(like.expr(), input_schema)?, pattern, "\\")
    })
}

fn lower_in_list_expr(in_list: &InListExpr, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    if in_list.is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let expr = lower_physical_expr(in_list.expr(), input_schema)?;
    let list = in_list
        .list()
        .iter()
        .map(|expr| {
            let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            if literal.value().is_null() {
                return Err(FallbackReason::UnsupportedExpression);
            }
            lower_physical_expr(expr, input_schema)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if in_list.negated() { NativeExpr::not_in_list(expr, list) } else { NativeExpr::in_list(expr, list) })
}

fn lower_case_expr(case: &CaseExpr, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    let base_expr = case.expr().map(|expr| lower_physical_expr(expr, input_schema)).transpose()?;
    let branches = case
        .when_then_expr()
        .iter()
        .map(|(when, then)| {
            let when = match &base_expr {
                Some(base_expr) => {
                    NativeExpr::binary(base_expr.clone(), NativeBinaryOp::Eq, lower_physical_expr(when, input_schema)?)
                },
                None => lower_physical_expr(when, input_schema)?,
            };
            Ok(NativeCaseWhen::new(when, lower_physical_expr(then, input_schema)?))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let else_expr = match case.else_expr() {
        Some(else_expr) => lower_physical_expr(else_expr, input_schema)?,
        None => {
            let data_type =
                case.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
            NativeExpr::literal(cudf_query_engine::LiteralValue::Null(data_type))
        },
    };
    Ok(NativeExpr::case(branches, else_expr))
}

fn lower_scalar_function_expr(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    match function.name().to_ascii_lowercase().as_str() {
        "substring" | "substr" => lower_substring_function(function, input_schema),
        "date_part" | "extract" => lower_extract_function(function, input_schema),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn lower_substring_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let start = literal_i32(&args[1])?;
    if start < 1 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let length = if args.len() == 3 { Some(literal_i32(&args[2])?) } else { None };
    if let Some(length) = length
        && (length < 0 || (start - 1).checked_add(length).is_none())
    {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(NativeExpr::substring(lower_physical_expr(&args[0], input_schema)?, start, length))
}

fn lower_extract_function(
    function: &ScalarFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let [field, expr] = function.args() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let field = match literal_string(field)?.to_ascii_lowercase().as_str() {
        "year" => NativeDatePart::Year,
        "month" => NativeDatePart::Month,
        "day" => NativeDatePart::Day,
        _ => return Err(FallbackReason::UnsupportedExpression),
    };
    Ok(NativeExpr::extract(field, lower_physical_expr(expr, input_schema)?))
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let data_type = input_schema.field_with_name(column.name()).map(|field| field.data_type()).or_else(|_error| {
        input_schema
            .fields()
            .get(column.index())
            .map(|field| field.data_type())
            .ok_or(FallbackReason::UnsupportedExpression)
    })?;
    if is_supported_native_type(data_type) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn validate_modulo_operand_type(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if matches!(
        data_type,
        arrow::datatypes::DataType::Int8
            | arrow::datatypes::DataType::Int16
            | arrow::datatypes::DataType::Int32
            | arrow::datatypes::DataType::Int64
            | arrow::datatypes::DataType::UInt8
            | arrow::datatypes::DataType::UInt16
            | arrow::datatypes::DataType::UInt32
            | arrow::datatypes::DataType::UInt64
    ) {
        Ok(())
    } else {
        Err(FallbackReason::UnsupportedType)
    }
}

fn resolved_column_name<'a>(column: &Column, input_schema: &'a SchemaRef) -> Result<&'a str, FallbackReason> {
    input_schema.field_with_name(column.name()).map(|field| field.name().as_str()).or_else(|_error| {
        input_schema
            .fields()
            .get(column.index())
            .map(|field| field.name().as_str())
            .ok_or(FallbackReason::UnsupportedExpression)
    })
}

fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, FallbackReason> {
    Ok(NativeExpr::literal(native_literal_value(value, NativeLiteralContext::Expression)?))
}

fn lower_binary_operator(op: &Operator) -> Result<NativeBinaryOp, FallbackReason> {
    match op {
        Operator::Plus => Ok(NativeBinaryOp::Add),
        Operator::Minus => Ok(NativeBinaryOp::Subtract),
        Operator::Multiply => Ok(NativeBinaryOp::Multiply),
        Operator::Divide => Ok(NativeBinaryOp::Divide),
        Operator::Modulo => Ok(NativeBinaryOp::Modulo),
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

fn literal_string(expr: &Arc<dyn PhysicalExpr>) -> Result<String, FallbackReason> {
    let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal.value() {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Ok(value.clone())
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn literal_i32(expr: &Arc<dyn PhysicalExpr>) -> Result<i32, FallbackReason> {
    let Some(literal) = expr.as_any().downcast_ref::<Literal>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    match literal.value() {
        ScalarValue::Int8(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::Int32(Some(value)) => Ok(*value),
        ScalarValue::Int64(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        ScalarValue::UInt8(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        ScalarValue::UInt64(Some(value)) => {
            i32::try_from(*value).map_err(|_error| FallbackReason::UnsupportedExpression)
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn alias_if_needed(expr: NativeExpr, alias: &str) -> NativeExpr {
    match &expr {
        NativeExpr::Column(name) if name == alias => expr,
        _ => expr.alias(alias),
    }
}
