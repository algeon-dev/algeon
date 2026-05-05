use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};
use cudf_query_engine::{
    AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
    AggregateSpec as NativeAggregateSpec, BinaryOp as NativeBinaryOp, CaseWhen as NativeCaseWhen,
    ColumnRef as NativeColumnRef, DatePart as NativeDatePart, Expr as NativeExpr, JoinKey as NativeJoinKey,
    JoinNullEquality as NativeJoinNullEquality, JoinOutputColumn as NativeJoinOutputColumn, JoinSide as NativeJoinSide,
    JoinSpec as NativeJoinSpec, JoinType as NativeJoinType, LimitSpec as NativeLimitSpec,
    LocalParquetSource as NativeLocalParquetSource, NodeId as NativeNodeId, QueryPlan as NativeQueryPlan,
    QueryPlanBuilder as NativeQueryPlanBuilder, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec,
    SourceNode as NativeSourceNode, ZERO_COLUMN_COUNT_COLUMN as NATIVE_ZERO_COLUMN_COUNT_COLUMN,
};
use datafusion::{
    common::{JoinSide as DataFusionJoinSide, NullEquality, ScalarValue},
    datasource::physical_plan::ParquetSource,
    logical_expr::{JoinType, Operator},
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
        joins::{NestedLoopJoinExec, utils::JoinFilter},
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};

use super::{
    literal::{NativeLiteralContext, native_literal_value},
    normalize::{
        NormalizedNativeHashJoin, NormalizedNativeInput, NormalizedNativeNestedLoopJoin, NormalizedNativePlan,
        NormalizedNativeSource,
    },
    source::{
        local_parquet_files, local_parquet_source_diagnostics, native_source_output_schema,
        required_parquet_column_requirements, required_parquet_column_requirements_for_scan_columns,
        validate_slice1_parquet_scan,
    },
    validate::{
        aggregate_is_count_star_only, collect_required_columns, is_supported_native_cast, is_supported_native_type,
        native_zero_column_count_source_is_safe, validate_aggregate, validate_physical_expr, validate_schema,
        validate_sort,
    },
};
use crate::policy::FallbackReason;

pub(super) fn lower_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    if normalized.single_source_input().is_some() {
        lower_slice1_plan(normalized)
    } else {
        lower_joined_aggregate_plan(normalized)
    }
}

struct LoweredNativeInput {
    node: NativeNodeId,
    schema: SchemaRef,
}

#[derive(Debug, Clone, Copy)]
struct CanonicalJoin {
    join_type: NativeJoinType,
    swap_inputs: bool,
}

impl CanonicalJoin {
    const fn new(join_type: NativeJoinType, swap_inputs: bool) -> Self {
        Self { join_type, swap_inputs }
    }

    fn order_inputs(
        self,
        left: LoweredNativeInput,
        right: LoweredNativeInput,
    ) -> (LoweredNativeInput, LoweredNativeInput) {
        if self.swap_inputs { (right, left) } else { (left, right) }
    }

    fn native_input_index_for_datafusion_side(self, side: DataFusionJoinSide) -> Result<usize, FallbackReason> {
        match (self.swap_inputs, side) {
            (false, DataFusionJoinSide::Left) | (true, DataFusionJoinSide::Right) => Ok(0),
            (false, DataFusionJoinSide::Right) | (true, DataFusionJoinSide::Left) => Ok(1),
            (_, DataFusionJoinSide::None) => Err(FallbackReason::UnsupportedExpression),
        }
    }

    fn native_side_for_datafusion_side(self, side: DataFusionJoinSide) -> Result<NativeJoinSide, FallbackReason> {
        match self.native_input_index_for_datafusion_side(side)? {
            0 => Ok(NativeJoinSide::Left),
            1 => Ok(NativeJoinSide::Right),
            _ => unreachable!("canonical native join input index is always 0 or 1"),
        }
    }

    const fn datafusion_side_for_native_side(self, side: NativeJoinSide) -> DataFusionJoinSide {
        match (self.swap_inputs, side) {
            (false, NativeJoinSide::Left) | (true, NativeJoinSide::Right) => DataFusionJoinSide::Left,
            (false, NativeJoinSide::Right) | (true, NativeJoinSide::Left) => DataFusionJoinSide::Right,
        }
    }

    const fn build_side(self) -> NativeJoinSide {
        match self.join_type {
            NativeJoinType::Left | NativeJoinType::LeftSemi | NativeJoinType::LeftAnti => NativeJoinSide::Right,
            NativeJoinType::Inner | NativeJoinType::Full => NativeJoinSide::Left,
            NativeJoinType::Cross => NativeJoinSide::Left,
        }
    }
}

fn lower_slice1_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    let source = normalized.single_source_input().ok_or(FallbackReason::UnsupportedOperator)?;
    let source_exec = source.source_exec;
    let filter = source.filter;
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
    validate_required_scalar_expressions(RequiredScalarExpressions {
        scan_projection,
        table_schema: &table_schema,
        scan_output_schema: source_exec.schema(),
        required_scan_output_columns: &column_requirements.scan_output_columns,
        filter,
        pre_aggregate_projection,
        aggregate,
        post_aggregate_projection,
    })?;
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

fn lower_joined_aggregate_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    let pre_aggregate_projection = normalized.pre_aggregate_projection;
    let aggregate = normalized.aggregate;
    let post_aggregate_projection = normalized.post_aggregate_projection;
    let sort = normalized.sort;

    validate_required_operator_schemas(normalized, None, pre_aggregate_projection, post_aggregate_projection, sort)?;
    validate_aggregate(aggregate, false, normalized.aggregate_allows_partial)?;
    validate_joined_aggregate_scalar_expressions(normalized)?;
    if let Some(sort) = sort {
        validate_sort(sort)?;
    }

    let required_input_columns = required_aggregate_input_columns(aggregate, pre_aggregate_projection)?;
    let mut builder = NativeQueryPlanBuilder::new();
    let lowered_input = lower_native_input(&normalized.input, &mut builder, &required_input_columns)?;
    let (aggregate_input_node, aggregate_input_schema, synthetic_aggregate_args) = if let Some(projection) =
        pre_aggregate_projection
    {
        let projection_node =
            builder.add_projection(lowered_input.node, lower_projection_exprs(projection)?, projection.schema());
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
        let synthetic_projection = synthetic_aggregate_input_projection(aggregate, &lowered_input.schema)?;
        let projection_node = builder.add_projection(
            lowered_input.node,
            synthetic_projection.expressions,
            Arc::clone(&synthetic_projection.schema),
        );
        (projection_node, synthetic_projection.schema, Some(synthetic_projection.aggregate_arg_aliases))
    };
    let aggregate_spec = lower_aggregate_spec(aggregate, &aggregate_input_schema, synthetic_aggregate_args.as_deref())?;
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

fn validate_joined_aggregate_scalar_expressions(normalized: &NormalizedNativePlan<'_>) -> Result<(), FallbackReason> {
    if let Some(projection) = normalized.pre_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    for aggregate_expr in normalized.aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr(&expr, &normalized.aggregate.input().schema())?;
        }
    }
    for (group_expr, _alias) in normalized.aggregate.group_expr().expr() {
        validate_physical_expr(group_expr, &normalized.aggregate.input().schema())?;
    }
    if let Some(projection) = normalized.post_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    Ok(())
}

fn required_aggregate_input_columns(
    aggregate: &AggregateExec,
    pre_aggregate_projection: Option<&ProjectionExec>,
) -> Result<Vec<String>, FallbackReason> {
    let mut required = BTreeSet::new();
    if let Some(projection) = pre_aggregate_projection {
        for projection_expr in projection.expr() {
            collect_required_columns(&projection_expr.expr, &projection.input().schema(), &mut required)?;
        }
    } else {
        for aggregate_expr in aggregate.aggr_expr() {
            for expr in aggregate_expr.expressions() {
                collect_required_columns(&expr, &aggregate.input().schema(), &mut required)?;
            }
        }
        for (group_expr, _alias) in aggregate.group_expr().expr() {
            collect_required_columns(group_expr, &aggregate.input().schema(), &mut required)?;
        }
    }
    Ok(required.into_iter().collect())
}

fn lower_native_input(
    input: &NormalizedNativeInput<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    match input {
        NormalizedNativeInput::Source(source) => lower_source_input(source, builder, required_output_columns),
        NormalizedNativeInput::HashJoin(join) => lower_hash_join_input(join, builder, required_output_columns),
        NormalizedNativeInput::NestedLoopJoin(join) => {
            lower_nested_loop_join_input(join, builder, required_output_columns)
        },
    }
}

fn lower_source_input(
    source: &NormalizedNativeSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let source_exec = source.source_exec;
    let filter = source.filter;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;
    let column_requirements =
        required_parquet_column_requirements_for_scan_columns(file_scan, filter, required_output_columns)?;
    let table_schema = Arc::clone(file_scan.file_source().table_schema().table_schema());
    let scan_projection = file_scan.file_source().projection();
    let source_output_schema = native_source_output_schema(file_scan, &column_requirements)?;

    validate_schema(Arc::clone(&source_output_schema))?;
    if let Some(scan_projection) = scan_projection {
        for projection_expr in
            required_scan_projection(scan_projection, source_exec.schema(), &column_requirements.scan_output_columns)
        {
            validate_physical_expr(projection_expr.expr, &table_schema)?;
        }
    }
    if let Some(filter) = filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }
    validate_slice1_parquet_scan(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;

    let native_files =
        local_parquet_files(file_scan, parquet_source, filter, Arc::clone(&table_schema), &column_requirements)?;
    let mut native_source = NativeLocalParquetSource::from_files(
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
    .with_partition_columns(column_requirements.partition_columns.clone());
    native_source.diagnostics = local_parquet_source_diagnostics(
        file_scan,
        parquet_source,
        filter,
        Arc::clone(&table_schema),
        Some(&column_requirements),
        false,
    );

    let source_node = builder.add_source(NativeSourceNode::LocalParquet(native_source));
    let (scan_output_node, scan_output_schema) = if let Some(scan_projection) = scan_projection {
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
    let filter_input_schema = source_exec.schema();
    let (node, schema) = if let Some(filter) = filter {
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
    Ok(LoweredNativeInput { node, schema })
}

fn lower_hash_join_input(
    join: &NormalizedNativeHashJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let canonical = canonical_join(join.join.join_type())?;
    let output_columns = hash_join_output_columns(join, required_output_columns, canonical)?;
    let residual = join
        .join
        .filter()
        .map(|filter| lower_join_filter(filter, &join.join.left().schema(), &join.join.right().schema(), canonical))
        .transpose()?;
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let keys = join
        .join
        .on()
        .iter()
        .map(|(left, right)| {
            let left_schema = join.join.left().schema();
            let right_schema = join.join.right().schema();
            let left_name = join_key_column_name(left, &left_schema)?;
            let right_name = join_key_column_name(right, &right_schema)?;
            left_required.insert(left_name.to_string());
            right_required.insert(right_name.to_string());
            if canonical.swap_inputs {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, right_name, right_name),
                    NativeColumnRef::input(1, left_name, left_name),
                ))
            } else {
                Ok(NativeJoinKey::new(
                    NativeColumnRef::input(0, left_name, left_name),
                    NativeColumnRef::input(1, right_name, right_name),
                ))
            }
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    for output in &output_columns.projection {
        match canonical.datafusion_side_for_native_side(output.side) {
            DataFusionJoinSide::Left => {
                left_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::None => {},
        }
    }
    if let Some(residual) = &residual {
        left_required.extend(residual.left_required.iter().cloned());
        right_required.extend(residual.right_required.iter().cloned());
    }
    let left_required = left_required.into_iter().collect::<Vec<_>>();
    let right_required = right_required.into_iter().collect::<Vec<_>>();
    let left = lower_native_input(&join.left, builder, &left_required)?;
    let right = lower_native_input(&join.right, builder, &right_required)?;
    let (native_left, native_right) = canonical.order_inputs(left, right);
    let mut spec = NativeJoinSpec::new(canonical.join_type, keys)
        .with_output_projection(output_columns.projection)
        .with_null_equality(lower_join_null_equality(join.join.null_equality()))
        .with_build_side(canonical.build_side());
    if let Some(residual) = residual {
        spec = spec.with_residual_predicate(residual.predicate);
    }
    let node = builder.add_join(native_left.node, native_right.node, spec, Arc::clone(&output_columns.schema));
    Ok(LoweredNativeInput { node, schema: output_columns.schema })
}

fn lower_nested_loop_join_input(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_columns: &[String],
) -> Result<LoweredNativeInput, FallbackReason> {
    let (canonical, residual) = lower_nested_loop_join_contract(join.join)?;
    let output_columns = nested_loop_join_output_columns(join, required_output_columns, canonical)?;
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    for output in &output_columns.projection {
        match canonical.datafusion_side_for_native_side(output.side) {
            DataFusionJoinSide::Left => {
                left_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::Right => {
                right_required.insert(output.column.name().to_string());
            },
            DataFusionJoinSide::None => {},
        }
    }
    if let Some(residual) = &residual {
        left_required.extend(residual.left_required.iter().cloned());
        right_required.extend(residual.right_required.iter().cloned());
    }
    let left_required = left_required.into_iter().collect::<Vec<_>>();
    let right_required = right_required.into_iter().collect::<Vec<_>>();
    let left = lower_native_input(&join.left, builder, &left_required)?;
    let right = lower_native_input(&join.right, builder, &right_required)?;
    let (native_left, native_right) = canonical.order_inputs(left, right);
    let mut spec = NativeJoinSpec::new(canonical.join_type, Vec::new())
        .with_output_projection(output_columns.projection)
        .with_build_side(canonical.build_side());
    if let Some(residual) = residual {
        spec = spec.with_residual_predicate(residual.predicate);
    }
    let node = builder.add_join(native_left.node, native_right.node, spec, Arc::clone(&output_columns.schema));
    Ok(LoweredNativeInput { node, schema: output_columns.schema })
}

fn lower_nested_loop_join_contract(
    join: &NestedLoopJoinExec,
) -> Result<(CanonicalJoin, Option<LoweredJoinResidual>), FallbackReason> {
    let canonical = canonical_join(join.join_type())?;
    if let Some(filter) = join.filter() {
        let residual = lower_join_filter(filter, &join.left().schema(), &join.right().schema(), canonical)?;
        return Ok((canonical, Some(residual)));
    }
    if matches!(join.join_type(), JoinType::Inner) {
        return Ok((CanonicalJoin::new(NativeJoinType::Cross, false), None));
    }
    Err(FallbackReason::UnsupportedOperator)
}

struct LoweredJoinResidual {
    predicate: NativeExpr,
    left_required: BTreeSet<String>,
    right_required: BTreeSet<String>,
}

fn lower_join_filter(
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    canonical: CanonicalJoin,
) -> Result<LoweredJoinResidual, FallbackReason> {
    let mut left_required = BTreeSet::new();
    let mut right_required = BTreeSet::new();
    let predicate = lower_join_filter_expr(
        filter.expression(),
        filter,
        left_schema,
        right_schema,
        canonical,
        &mut left_required,
        &mut right_required,
    )?;
    if left_required.is_empty() || right_required.is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(LoweredJoinResidual { predicate, left_required, right_required })
}

fn lower_join_filter_expr(
    expr: &Arc<dyn PhysicalExpr>,
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    canonical: CanonicalJoin,
    left_required: &mut BTreeSet<String>,
    right_required: &mut BTreeSet<String>,
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return lower_join_filter_column(
            column,
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        );
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_join_filter_expr(
            binary.left(),
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        )?;
        let right = lower_join_filter_expr(
            binary.right(),
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        )?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(!lower_join_filter_expr(
            not.arg(),
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        )?);
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return Ok(NativeExpr::is_null(lower_join_filter_expr(
            is_null.arg(),
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        )?));
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return Ok(NativeExpr::is_not_null(lower_join_filter_expr(
            is_not_null.arg(),
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        )?));
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        let expr = lower_join_filter_expr(
            in_list.expr(),
            filter,
            left_schema,
            right_schema,
            canonical,
            left_required,
            right_required,
        )?;
        let list = in_list
            .list()
            .iter()
            .map(|item| {
                let Some(literal) = item.as_any().downcast_ref::<Literal>() else {
                    return Err(FallbackReason::UnsupportedExpression);
                };
                if literal.value().is_null() {
                    return Err(FallbackReason::UnsupportedExpression);
                }
                lower_join_filter_expr(
                    item,
                    filter,
                    left_schema,
                    right_schema,
                    canonical,
                    left_required,
                    right_required,
                )
            })
            .collect::<Result<Vec<_>, FallbackReason>>()?;
        return Ok(if in_list.negated() {
            NativeExpr::not_in_list(expr, list)
        } else {
            NativeExpr::in_list(expr, list)
        });
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn lower_join_filter_column(
    column: &Column,
    filter: &JoinFilter,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    canonical: CanonicalJoin,
    left_required: &mut BTreeSet<String>,
    right_required: &mut BTreeSet<String>,
) -> Result<NativeExpr, FallbackReason> {
    let column_index = filter.column_indices().get(column.index()).ok_or(FallbackReason::UnsupportedExpression)?;
    let (datafusion_side, schema, required) = match column_index.side {
        DataFusionJoinSide::Left => (DataFusionJoinSide::Left, left_schema, left_required),
        DataFusionJoinSide::Right => (DataFusionJoinSide::Right, right_schema, right_required),
        DataFusionJoinSide::None => return Err(FallbackReason::UnsupportedExpression),
    };
    let field = schema.fields().get(column_index.index).ok_or(FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(field.data_type()) {
        return Err(FallbackReason::UnsupportedType);
    }
    required.insert(field.name().clone());
    let native_input_index = canonical.native_input_index_for_datafusion_side(datafusion_side)?;
    Ok(NativeExpr::input_column(native_input_index, field.name(), field.name()))
}

struct JoinOutputColumns {
    projection: Vec<NativeJoinOutputColumn>,
    schema: SchemaRef,
}

fn hash_join_output_columns(
    join: &NormalizedNativeHashJoin<'_>,
    required_output_columns: &[String],
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    join_output_columns(
        required_output_columns,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        join.join.projection.as_deref(),
        join.join.join_type(),
        canonical,
    )
}

fn nested_loop_join_output_columns(
    join: &NormalizedNativeNestedLoopJoin<'_>,
    required_output_columns: &[String],
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    join_output_columns(
        required_output_columns,
        &join.join.left().schema(),
        &join.join.right().schema(),
        &join.join.schema(),
        join.join.projection().as_deref(),
        join.join.join_type(),
        canonical,
    )
}

fn join_output_columns(
    required_output_columns: &[String],
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection_indices: Option<&[usize]>,
    datafusion_join_type: &JoinType,
    canonical: CanonicalJoin,
) -> Result<JoinOutputColumns, FallbackReason> {
    let required = required_output_columns.iter().collect::<BTreeSet<_>>();
    let mut output_fields = Vec::new();
    let mut projection = Vec::new();
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if !required.is_empty() && !required.contains(field.name()) {
            continue;
        }
        let join_schema_index = match projection_indices {
            Some(projection) => *projection.get(output_index).ok_or(FallbackReason::UnsupportedExpression)?,
            None => output_index,
        };
        let (datafusion_side, input_index) =
            datafusion_join_output_side(datafusion_join_type, join_schema_index, left_schema)?;
        let side = canonical.native_side_for_datafusion_side(datafusion_side)?;
        let input_schema = if datafusion_side == DataFusionJoinSide::Left { left_schema } else { right_schema };
        let input_field = input_schema.fields().get(input_index).ok_or(FallbackReason::UnsupportedExpression)?;
        let output_name = field.name().clone();
        let column = NativeColumnRef::input(
            match side {
                NativeJoinSide::Left => 0,
                NativeJoinSide::Right => 1,
            },
            input_field.name(),
            output_name.clone(),
        );
        projection.push(NativeJoinOutputColumn::new(side, column, output_name));
        output_fields.push(field.as_ref().clone());
    }
    Ok(JoinOutputColumns { projection, schema: Arc::new(Schema::new(output_fields)) })
}

fn join_key_column_name<'a>(
    expr: &'a Arc<dyn PhysicalExpr>,
    input_schema: &'a SchemaRef,
) -> Result<&'a str, FallbackReason> {
    let Some(column) = expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    resolved_column_name(column, input_schema)
}

fn canonical_join(join_type: &JoinType) -> Result<CanonicalJoin, FallbackReason> {
    match join_type {
        JoinType::Inner => Ok(CanonicalJoin::new(NativeJoinType::Inner, false)),
        JoinType::Left => Ok(CanonicalJoin::new(NativeJoinType::Left, false)),
        JoinType::Right => Ok(CanonicalJoin::new(NativeJoinType::Left, true)),
        JoinType::Full => Ok(CanonicalJoin::new(NativeJoinType::Full, false)),
        JoinType::LeftSemi => Ok(CanonicalJoin::new(NativeJoinType::LeftSemi, false)),
        JoinType::RightSemi => Ok(CanonicalJoin::new(NativeJoinType::LeftSemi, true)),
        JoinType::LeftAnti => Ok(CanonicalJoin::new(NativeJoinType::LeftAnti, false)),
        JoinType::RightAnti => Ok(CanonicalJoin::new(NativeJoinType::LeftAnti, true)),
        _ => Err(FallbackReason::UnsupportedOperator),
    }
}

fn datafusion_join_output_side(
    join_type: &JoinType,
    join_schema_index: usize,
    left_schema: &SchemaRef,
) -> Result<(DataFusionJoinSide, usize), FallbackReason> {
    match join_type {
        JoinType::RightSemi | JoinType::RightAnti => Ok((DataFusionJoinSide::Right, join_schema_index)),
        JoinType::LeftSemi | JoinType::LeftAnti => Ok((DataFusionJoinSide::Left, join_schema_index)),
        _ if join_schema_index < left_schema.fields().len() => Ok((DataFusionJoinSide::Left, join_schema_index)),
        _ => Ok((DataFusionJoinSide::Right, join_schema_index - left_schema.fields().len())),
    }
}

fn lower_join_null_equality(null_equality: NullEquality) -> NativeJoinNullEquality {
    match null_equality {
        NullEquality::NullEqualsNothing => NativeJoinNullEquality::NullsDoNotMatch,
        NullEquality::NullEqualsNull => NativeJoinNullEquality::NullsMatch,
    }
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

struct RequiredScalarExpressions<'a> {
    scan_projection: Option<&'a ProjectionExprs>,
    table_schema: &'a SchemaRef,
    scan_output_schema: SchemaRef,
    required_scan_output_columns: &'a [String],
    filter: Option<&'a datafusion::physical_plan::filter::FilterExec>,
    pre_aggregate_projection: Option<&'a ProjectionExec>,
    aggregate: &'a AggregateExec,
    post_aggregate_projection: Option<&'a ProjectionExec>,
}

fn validate_required_scalar_expressions(inputs: RequiredScalarExpressions<'_>) -> Result<(), FallbackReason> {
    if let Some(scan_projection) = inputs.scan_projection {
        for projection_expr in
            required_scan_projection(scan_projection, inputs.scan_output_schema, inputs.required_scan_output_columns)
        {
            validate_physical_expr(projection_expr.expr, inputs.table_schema)?;
        }
    }
    if let Some(filter) = inputs.filter {
        validate_physical_expr(filter.predicate(), &filter.input().schema())?;
    }
    if let Some(projection) = inputs.pre_aggregate_projection {
        for projection_expr in projection.expr() {
            validate_physical_expr(&projection_expr.expr, &projection.input().schema())?;
        }
    }
    for aggregate_expr in inputs.aggregate.aggr_expr() {
        for expr in aggregate_expr.expressions() {
            validate_physical_expr(&expr, &inputs.aggregate.input().schema())?;
        }
    }
    for (group_expr, _alias) in inputs.aggregate.group_expr().expr() {
        validate_physical_expr(group_expr, &inputs.aggregate.input().schema())?;
    }
    if let Some(projection) = inputs.post_aggregate_projection {
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
                .cloned()
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
        return Ok(!lower_physical_expr(not.arg(), input_schema)?);
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
        && (length < 0 || substring_stop_position(start, length).is_none())
    {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(NativeExpr::substring(lower_physical_expr(&args[0], input_schema)?, start, length))
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
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
        NativeExpr::Column(column) if column.display_name() == alias || column.name() == alias => expr,
        _ => expr.alias(alias),
    }
}
