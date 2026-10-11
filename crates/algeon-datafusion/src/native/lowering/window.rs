use std::{any::Any, collections::BTreeSet, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use cudf::interop::arrow_host::is_supported_arrow_type;
use datafusion::{
    common::ScalarValue,
    functions_window::{
        lead_lag::{WindowShift, WindowShiftKind},
        rank::Rank,
        row_number::RowNumber,
    },
    logical_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits},
    physical_expr::{
        PhysicalExpr,
        expressions::{Column, Literal},
        window::{PlainAggregateWindowExpr, SlidingAggregateWindowExpr, StandardWindowExpr, WindowExpr},
    },
    physical_plan::windows::WindowUDFExpr,
};
use query_planning::{
    datatypes::{is_supported_list_carry_type, is_timestamp_with_timezone},
    expr::{
        AggregateFunction as NativeAggregateFunction, Expr as NativeExpr, LimitSpec as NativeLimitSpec,
        LiteralValue as NativeLiteralValue, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec,
    },
    plan::{
        QueryPlanBuilder as NativeQueryPlanBuilder, ShiftDirection as NativeShiftDirection,
        WindowFrame as NativeWindowFrame, WindowFunction as NativeWindowFunction, WindowSpec as NativeWindowSpec,
    },
};

use super::{
    Demand, FieldBinding, LoweredNativeInput, RequiredFields,
    aggregate::{lower_native_aggregate_function, project_required_output_columns},
    expr::{collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input,
    naming::{sanitize_field_name, unique_name_with_suffix},
    sort::{SortedNativeInput, lower_sort_ordering_unprojected_with_input, project_sort_output},
};
use crate::{
    capability::CapabilityReason,
    native::{
        ExpressionRejection, LoweringFailure, LoweringRejection, ParquetReadStrategyConfig,
        expression_contract::validate_physical_expr_contract,
        literal::{NativeLiteralContext, native_literal_value},
        normalize::{NormalizedNativeInput, NormalizedNativeSort, NormalizedNativeWindow},
        validate::{aggregate_blocked, validate_schema, validate_sort_ordering},
    },
    report::{AggregateDefect, PlanBlocker, WindowDefect},
};

/// A rejection of `reason` that names the window function `function` and what
/// is unsupported about its use.
fn window_blocked(reason: CapabilityReason, function: &str, defect: WindowDefect) -> LoweringFailure {
    LoweringRejection::blocked_by(reason, PlanBlocker::Window { function: function.to_string(), defect }).into()
}

pub(super) fn lower_sort_input(
    sort: &NormalizedNativeSort<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let sorted = lower_unprojected_sort_input(sort, builder, required_output_fields, source_reads)?;
    project_sort_output(builder, sort.sort.output_schema(), sorted.input, required_output_fields)
}

fn lower_unprojected_sort_input(
    sort: &NormalizedNativeSort<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<SortedNativeInput, LoweringFailure> {
    validate_schema(Arc::clone(sort.sort.output_schema()))?;
    validate_sort_ordering(sort.sort.ordering(), sort.sort.input_schema())?;
    let mut child_required = BTreeSet::new();
    for sort_expr in sort.sort.ordering() {
        collect_required_field_indices(&sort_expr.expr, sort.sort.input_schema(), &mut child_required)?;
    }
    child_required.extend(required_output_fields.requested_indices(sort.sort.output_schema().fields().len())?);
    let input =
        lower_native_input(&sort.input, builder, &Demand::fields(RequiredFields::only(child_required)), source_reads)?;
    let sorted =
        lower_sort_ordering_unprojected_with_input(builder, sort.sort.ordering(), sort.sort.input_schema(), input)?;
    Ok(SortedNativeInput {
        input: apply_sort_fetch_limit(builder, sorted.input, sort.fetch)?,
        key_columns: sorted.key_columns,
    })
}

fn apply_sort_fetch_limit(
    builder: &mut NativeQueryPlanBuilder,
    sorted: LoweredNativeInput,
    fetch: Option<NativeLimitSpec>,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let Some(fetch) = fetch else {
        return Ok(sorted);
    };
    let node = builder.add_limit(sorted.node, fetch)?;
    Ok(LoweredNativeInput::with_bindings(node, sorted.schema, sorted.bindings))
}

pub(super) fn lower_window_input(
    window: &NormalizedNativeWindow<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let window_schema = window.window.schema();
    let window_input_schema = window.window.input_schema();
    validate_schema(Arc::clone(&window_schema))?;
    validate_window_exprs(window.window.window_expr(), &window_input_schema)?;

    // A window key that repeats an expression key of the sort below it reads
    // the column the sort ordered by. Recomputing the expression above the
    // sort yields a new column that carries none of the sort's ordering.
    let (input, sorted_keys) = match window.input.as_ref() {
        NormalizedNativeInput::Sort(sort) => {
            let sorted = lower_unprojected_sort_input(sort, builder, &RequiredFields::all(), source_reads)?;
            let keys = sort
                .sort
                .ordering()
                .iter()
                .map(|sort_expr| &sort_expr.expr)
                .zip(sorted.key_columns)
                .collect::<Vec<_>>();
            (sorted.input, keys)
        },
        // The native window reads its input in order.
        input => {
            (lower_native_input(input, builder, &Demand { ordered: true, ..Demand::all() }, source_reads)?, Vec::new())
        },
    };
    let mut prepared = materialize_window_key_expressions(
        builder,
        window.window.window_expr(),
        &window_input_schema,
        input,
        &sorted_keys,
    )?;
    if window_needs_ordered_input(window.window.window_expr(), &prepared.keys)
        && !input_keeps_row_order(window.input.as_ref())
    {
        prepared = sort_window_input(builder, prepared)?;
    }
    let (output_schema, output_bindings) = native_window_output_schema_and_bindings(&prepared.input, &window_schema)?;
    let spec = lower_window_spec(
        window.window.window_expr(),
        &window_input_schema,
        &output_bindings,
        &output_schema,
        &prepared.keys,
    )?;
    let window_node = builder.add_window(prepared.input.node, spec)?;
    project_required_output_columns(builder, window_node, output_schema, output_bindings, required_output_fields)
}

/// Whether the window reads rows in window order: it orders them, or numbers
/// them by position inside each partition. The other window functions only
/// return rows in the order they read them, which DataFusion reports as the
/// plan's ordering.
fn window_needs_ordered_input(window_exprs: &[Arc<dyn WindowExpr>], keys: &[PreparedWindowKeys]) -> bool {
    let Some(keys) = keys.first() else {
        return false;
    };
    let numbers_rows = window_exprs.iter().any(|window_expr| {
        window_expr.as_any().downcast_ref::<StandardWindowExpr>().is_some_and(|standard| {
            standard_window_kind(standard).is_ok_and(|kind| kind == StandardWindowKind::RowNumber)
        })
    });
    !keys.order_by.is_empty() || (numbers_rows && !keys.partition_by.is_empty())
}

/// Whether the native plan lowered from `input` returns the rows in the order
/// DataFusion reports for it. A sort, or a source read in its declared order,
/// establishes the order; filters, projections and windows keep it; joins,
/// aggregates, unions and GPU-resident inputs do not, and a limit keeps it
/// only over an input that establishes or keeps it without a limit of its
/// own. `native_plan_reproduces_declared_output_ordering` applies the same
/// rule to the optimized native plan.
fn input_keeps_row_order(input: &NormalizedNativeInput<'_>) -> bool {
    match input {
        NormalizedNativeInput::Source(_) | NormalizedNativeInput::LiteralSource(_) | NormalizedNativeInput::Sort(_) => {
            true
        },
        NormalizedNativeInput::Filter(filter) => input_keeps_row_order(&filter.input),
        NormalizedNativeInput::Projection(projection) => input_keeps_row_order(&projection.input),
        NormalizedNativeInput::Window(window) => input_keeps_row_order(&window.input),
        NormalizedNativeInput::Limit(limit) => match limit.input.as_ref() {
            NormalizedNativeInput::Limit(_) => false,
            input => input_keeps_row_order(input),
        },
        NormalizedNativeInput::GpuResidentSource(_)
        | NormalizedNativeInput::Aggregate(_)
        | NormalizedNativeInput::HashJoin(_)
        | NormalizedNativeInput::CrossJoin(_)
        | NormalizedNativeInput::NestedLoopJoin(_)
        | NormalizedNativeInput::Union(_) => false,
    }
}

/// Orders the window input by the partition keys, then the order keys, which
/// is the ordering the compiled window contract recognizes. An order key that
/// repeats a partition key is constant inside a partition and adds nothing.
fn sort_window_input(
    builder: &mut NativeQueryPlanBuilder,
    prepared: PreparedWindowInput,
) -> Result<PreparedWindowInput, LoweringFailure> {
    let keys = prepared.keys.first().ok_or(CapabilityReason::UnsupportedExpression)?;
    let expressions = keys
        .partition_by
        .iter()
        .map(|key| NativeSortExpr::new(key.clone(), false, false))
        .chain(keys.order_by.iter().filter(|key| !keys.partition_by.contains(&key.expr)).cloned())
        .collect::<Vec<_>>();
    let node = builder.add_sort(prepared.input.node, NativeSortSpec::new(expressions))?;
    let input = LoweredNativeInput::with_bindings(node, prepared.input.schema, prepared.input.bindings);
    Ok(PreparedWindowInput { input, keys: prepared.keys })
}

fn validate_window_exprs(
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
) -> Result<(), LoweringFailure> {
    if window_exprs.is_empty() {
        return Err(CapabilityReason::UnsupportedOperator.into());
    }
    for window_expr in window_exprs {
        for expr in window_expr.partition_by() {
            validate_window_value_expr(expr, input_schema)?;
        }
        for sort_expr in window_expr.order_by() {
            validate_window_value_expr(&sort_expr.expr, input_schema)?;
        }
        if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
            validate_plain_aggregate_window(plain, input_schema)?;
        } else if let Some(standard) = window_expr.as_any().downcast_ref::<StandardWindowExpr>() {
            validate_standard_window(standard, input_schema)?;
        } else {
            return Err(unsupported_window_expr(window_expr));
        }
    }
    Ok(())
}

/// The rejection for a window expression that is neither a plain aggregate nor
/// a standard function. A sliding aggregate window is the one DataFusion plans
/// for a bounded frame.
fn unsupported_window_expr(window_expr: &Arc<dyn WindowExpr>) -> LoweringFailure {
    match window_expr.as_any().downcast_ref::<SlidingAggregateWindowExpr>() {
        Some(sliding) => window_blocked(
            CapabilityReason::UnsupportedOperator,
            sliding.get_aggregate_expr().fun().name(),
            WindowDefect::UnsupportedFrame(Box::new(WindowFrame::clone(sliding.get_window_frame()))),
        ),
        None => {
            window_blocked(CapabilityReason::UnsupportedOperator, window_expr.name(), WindowDefect::UnsupportedFunction)
        },
    }
}

fn validate_plain_aggregate_window(
    plain: &PlainAggregateWindowExpr,
    input_schema: &SchemaRef,
) -> Result<(), LoweringFailure> {
    let aggregate = plain.get_aggregate_expr();
    let function = aggregate.fun().name();
    let frame = plain.get_window_frame();
    let whole_partition = is_whole_partition_frame(frame);
    let cumulative = is_rows_unbounded_preceding_to_current_frame(frame);
    let unsupported_frame = || {
        window_blocked(
            CapabilityReason::UnsupportedOperator,
            function,
            WindowDefect::UnsupportedFrame(Box::new(WindowFrame::clone(frame))),
        )
    };
    if !whole_partition && !cumulative {
        return Err(unsupported_frame());
    }
    if !plain.order_by().is_empty() && !cumulative {
        return Err(unsupported_frame());
    }
    if plain.order_by().is_empty() && !whole_partition {
        return Err(unsupported_frame());
    }
    if aggregate.is_distinct() {
        return Err(
            aggregate_blocked(CapabilityReason::UnsupportedOperator, function, AggregateDefect::Distinct).into()
        );
    }
    if !aggregate.order_bys().is_empty() {
        return Err(aggregate_blocked(CapabilityReason::UnsupportedOperator, function, AggregateDefect::OrderBy).into());
    }
    // The cumulative kernel is a grouped scan, which exists for these three.
    let native_function = lower_native_aggregate_function(aggregate.fun())?;
    if cumulative
        && !matches!(
            native_function,
            NativeAggregateFunction::Sum | NativeAggregateFunction::Min | NativeAggregateFunction::Max
        )
    {
        return Err(unsupported_frame());
    }
    let expressions = aggregate.expressions();
    let [arg] = expressions.as_slice() else {
        return Err(aggregate_blocked(
            CapabilityReason::UnsupportedExpression,
            function,
            AggregateDefect::ArgumentCount(expressions.len()),
        )
        .into());
    };
    validate_window_value_expr(arg, input_schema)
}

/// The SQL function name of a standard window function, such as `rank`.
fn standard_function_name(standard: &StandardWindowExpr) -> &str {
    let function = standard.get_standard_func_expr();
    function.as_any().downcast_ref::<WindowUDFExpr>().map_or_else(|| function.name(), |expr| expr.fun().name())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StandardWindowKind {
    Rank,
    RowNumber,
    Lag,
    Lead,
}

fn validate_standard_window(standard: &StandardWindowExpr, input_schema: &SchemaRef) -> Result<(), LoweringFailure> {
    let kind = standard_window_kind(standard)?;
    let missing_order_by = || {
        window_blocked(
            CapabilityReason::UnsupportedOperator,
            standard_function_name(standard),
            WindowDefect::MissingOrderBy,
        )
    };
    match kind {
        StandardWindowKind::Rank if standard.order_by().is_empty() => Err(missing_order_by()),
        StandardWindowKind::Rank | StandardWindowKind::RowNumber => Ok(()),
        StandardWindowKind::Lag | StandardWindowKind::Lead => {
            if standard.order_by().is_empty() {
                return Err(missing_order_by());
            }
            // Parsing the offset and default literals here keeps validation and
            // lowering in sync. A parse failure returns the structured window
            // rejection that lowering would return for the same arguments.
            let _ = parse_lag_lead_args(standard, input_schema)?;
            Ok(())
        },
    }
}

fn standard_window_kind(standard: &StandardWindowExpr) -> Result<StandardWindowKind, LoweringFailure> {
    let unsupported_function =
        |name: &str| window_blocked(CapabilityReason::UnsupportedOperator, name, WindowDefect::UnsupportedFunction);
    let udf = standard
        .get_standard_func_expr()
        .as_any()
        .downcast_ref::<WindowUDFExpr>()
        .ok_or_else(|| unsupported_function(standard_function_name(standard)))?;
    let implementation = udf.fun().inner().as_ref() as &dyn Any;
    // `dense_rank()` and `percent_rank()` are the same `Rank` type with a
    // private rank kind; only `rank()` has a native kernel.
    if implementation.downcast_ref::<Rank>().is_some_and(|rank| *rank == Rank::basic()) {
        Ok(StandardWindowKind::Rank)
    } else if implementation.is::<RowNumber>() {
        Ok(StandardWindowKind::RowNumber)
    } else if let Some(shift) = implementation.downcast_ref::<WindowShift>() {
        match shift.kind() {
            WindowShiftKind::Lag => Ok(StandardWindowKind::Lag),
            WindowShiftKind::Lead => Ok(StandardWindowKind::Lead),
        }
    } else {
        Err(unsupported_function(udf.fun().name()))
    }
}

#[derive(Debug, Clone)]
struct LagLeadArgs {
    value_expr: Arc<dyn PhysicalExpr>,
    offset: u32,
    default: Option<NativeLiteralValue>,
}

/// Parse the (`value`, `offset`, `default`) argument triple for a
/// `lag` / `lead` window UDF.
///
/// DataFusion expands missing `lag`/`lead` arguments to typed null literals.
/// `offset` must be a non-negative integer literal no greater than `i32::MAX`, or
/// null for SQL's default `1`; `default` must be a scalar literal.
///
/// Use `WindowUDFExpr::args()` because
/// [`StandardWindowFunctionExpr::expressions`](datafusion::physical_expr::window::StandardWindowFunctionExpr::expressions)
/// hides `offset` and `default`.
fn parse_lag_lead_args(
    standard: &StandardWindowExpr,
    input_schema: &SchemaRef,
) -> Result<LagLeadArgs, LoweringFailure> {
    let udf = standard.get_standard_func_expr().as_any().downcast_ref::<WindowUDFExpr>().ok_or_else(|| {
        window_blocked(
            CapabilityReason::UnsupportedOperator,
            standard_function_name(standard),
            WindowDefect::UnsupportedFunction,
        )
    })?;
    let function = udf.fun().name();
    let non_constant_arguments = |failure: LoweringFailure| {
        failure.blocked_by_if_unnamed(|| PlanBlocker::Window {
            function: function.to_string(),
            defect: WindowDefect::Arguments,
        })
    };
    let args = udf.args();
    if args.is_empty() || args.len() > 3 {
        return Err(window_blocked(CapabilityReason::UnsupportedExpression, function, WindowDefect::Arguments));
    }
    let value_expr = Arc::clone(&args[0]);
    validate_physical_expr_contract(&value_expr, input_schema)?;
    let value_type =
        value_expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if !is_supported_arrow_type(&value_type)
        || is_timestamp_with_timezone(&value_type)
        || is_supported_list_carry_type(&value_type)
    {
        return Err(ExpressionRejection::unsupported_value_type(&value_expr, value_type).into());
    }
    let offset = match args.get(1) {
        None => 1u32,
        Some(expr) => parse_lag_lead_offset(expr).map_err(non_constant_arguments)?,
    };
    let default = match args.get(2) {
        None => None,
        Some(expr) => parse_lag_lead_default(expr, &value_type).map_err(non_constant_arguments)?,
    };
    Ok(LagLeadArgs { value_expr, offset, default })
}

fn validate_window_value_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), LoweringFailure> {
    validate_physical_expr_contract(expr, input_schema)?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if is_supported_arrow_type(&data_type)
        && !is_timestamp_with_timezone(&data_type)
        && !is_supported_list_carry_type(&data_type)
    {
        Ok(())
    } else {
        Err(ExpressionRejection::unsupported_value_type(expr, data_type).into())
    }
}

fn parse_lag_lead_offset(expr: &Arc<dyn PhysicalExpr>) -> Result<u32, LoweringFailure> {
    let literal = expr.downcast_ref::<Literal>().ok_or(CapabilityReason::UnsupportedExpression)?;
    match literal.value() {
        ScalarValue::Null => Ok(1u32),
        ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None) => Ok(1u32),
        ScalarValue::Int8(Some(value)) => offset_from_i64(i64::from(*value)),
        ScalarValue::Int16(Some(value)) => offset_from_i64(i64::from(*value)),
        ScalarValue::Int32(Some(value)) => offset_from_i64(i64::from(*value)),
        ScalarValue::Int64(Some(value)) => offset_from_i64(*value),
        ScalarValue::UInt8(Some(value)) => Ok(u32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(u32::from(*value)),
        ScalarValue::UInt32(Some(value)) => offset_from_u32(*value),
        ScalarValue::UInt64(Some(value)) => {
            offset_from_i64(i64::try_from(*value).map_err(|_| CapabilityReason::UnsupportedExpression)?)
        },
        _ => Err(CapabilityReason::UnsupportedExpression.into()),
    }
}

/// Reject offsets outside libcudf's signed shift range.
///
/// libcudf's `groupby::shift` uses `cudf::size_type` (`int32`), and the
/// engine negates the offset for `LEAD`. The supported range is therefore
/// `[0, i32::MAX]`. A larger offset is rejected here and the plan falls back to
/// CPU DataFusion; admitting it would produce a native plan that errors at
/// execute time.
fn offset_from_i64(value: i64) -> Result<u32, LoweringFailure> {
    if value < 0 {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    let value = u64::try_from(value).map_err(|_| CapabilityReason::UnsupportedExpression)?;
    if value > i32::MAX as u64 {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    Ok(value as u32)
}

fn offset_from_u32(value: u32) -> Result<u32, LoweringFailure> {
    if value > i32::MAX as u32 {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    Ok(value)
}

fn parse_lag_lead_default(
    expr: &Arc<dyn PhysicalExpr>,
    value_type: &DataType,
) -> Result<Option<NativeLiteralValue>, LoweringFailure> {
    let literal = expr.downcast_ref::<Literal>().ok_or(CapabilityReason::UnsupportedExpression)?;
    let scalar = literal.value();
    // Match DataFusion's lag/lead constructor, which encodes the missing
    // `default` argument as `ScalarValue::Null`. Treat that as the no-default
    // case so the engine fills with NULL of the value column's type.
    if matches!(scalar, ScalarValue::Null) {
        return Ok(None);
    }
    let lowered = native_literal_value(scalar, NativeLiteralContext::Expression)?;
    if matches!(&lowered, NativeLiteralValue::Null(_)) {
        // Typed nulls (e.g. `ScalarValue::Int32(None)`) collapse to the
        // no-default case so the engine derives the fill from the output
        // schema, matching DataFusion's "null default of expression type".
        return Ok(None);
    }
    if &lowered.data_type() != value_type {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }
    // These literal families match those the engine's `build_lag_lead_value_fill_scalar`
    // can lower to a cudf Scalar. Decimal, date, timestamp, duration, and interval
    // literals need a cudf fill-scalar contract before the adapter can admit them.
    // They are rejected and the query falls back to CPU; admitting them would fail
    // at execute time.
    match &lowered {
        NativeLiteralValue::Boolean(_)
        | NativeLiteralValue::Int8(_)
        | NativeLiteralValue::Int16(_)
        | NativeLiteralValue::Int32(_)
        | NativeLiteralValue::Int64(_)
        | NativeLiteralValue::UInt8(_)
        | NativeLiteralValue::UInt16(_)
        | NativeLiteralValue::UInt32(_)
        | NativeLiteralValue::UInt64(_)
        | NativeLiteralValue::Float32(_)
        | NativeLiteralValue::Float64(_)
        | NativeLiteralValue::Utf8(_)
        | NativeLiteralValue::LargeUtf8(_) => Ok(Some(lowered)),
        NativeLiteralValue::Decimal128 { .. }
        | NativeLiteralValue::Date32(_)
        | NativeLiteralValue::Date64(_)
        | NativeLiteralValue::Time32 { .. }
        | NativeLiteralValue::Time64 { .. }
        | NativeLiteralValue::Timestamp { .. }
        | NativeLiteralValue::Duration { .. }
        | NativeLiteralValue::IntervalMonthDayNano { .. }
        | NativeLiteralValue::Null(_) => Err(CapabilityReason::UnsupportedExpression.into()),
    }
}

fn is_whole_partition_frame(frame: &WindowFrame) -> bool {
    matches!(frame.units, WindowFrameUnits::Rows | WindowFrameUnits::Range)
        && matches!(&frame.start_bound, WindowFrameBound::Preceding(value) if scalar_is_unbounded(value))
        && matches!(&frame.end_bound, WindowFrameBound::Following(value) if scalar_is_unbounded(value))
}

fn is_rows_unbounded_preceding_to_current_frame(frame: &WindowFrame) -> bool {
    frame.units == WindowFrameUnits::Rows
        && matches!(&frame.start_bound, WindowFrameBound::Preceding(value) if scalar_is_unbounded(value))
        && matches!(frame.end_bound, WindowFrameBound::CurrentRow)
}

fn scalar_is_unbounded(value: &ScalarValue) -> bool {
    value.is_null()
}

fn native_window_output_schema_and_bindings(
    input: &LoweredNativeInput,
    window_schema: &SchemaRef,
) -> Result<(SchemaRef, Vec<FieldBinding>), LoweringFailure> {
    if window_schema.fields().len() < input.bindings.len() {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }

    let frontend_names = window_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut used_native_names = input.schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut bindings = input.bindings.clone();

    for output_index in input.bindings.len()..window_schema.fields().len() {
        let field = window_schema.fields().get(output_index).ok_or(CapabilityReason::UnsupportedExpression)?;
        let native_name = unique_window_output_name(output_index, field.name(), &frontend_names, &used_native_names);
        used_native_names.insert(native_name.clone());
        fields.push(rename_field(field.as_ref(), native_name.as_str()));
        bindings.push(FieldBinding { frontend_index: output_index, frontend_name: field.name().clone(), native_name });
    }

    Ok((Arc::new(Schema::new(fields)), bindings))
}

struct PreparedWindowInput {
    input: LoweredNativeInput,
    keys: Vec<PreparedWindowKeys>,
}

struct PreparedWindowKeys {
    partition_by: Vec<NativeExpr>,
    order_by: Vec<NativeSortExpr>,
    /// The column a window aggregate or LAG / LEAD reads; `None` for ranking
    /// functions.
    argument: Option<NativeExpr>,
}

/// Expression keys of the sort directly below a window, each with the column
/// that holds it.
type SortedWindowKeys<'a> = [(&'a Arc<dyn PhysicalExpr>, String)];

fn materialize_window_key_expressions(
    builder: &mut NativeQueryPlanBuilder,
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
    input: LoweredNativeInput,
    sorted_keys: &SortedWindowKeys<'_>,
) -> Result<PreparedWindowInput, LoweringFailure> {
    let mut used_names = input.schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut materialized_keys = Vec::<MaterializedWindowKey>::new();
    let mut prepared_keys = Vec::with_capacity(window_exprs.len());

    for window_expr in window_exprs {
        let partition_by = window_expr
            .partition_by()
            .iter()
            .map(|expr| {
                window_key_expr(expr, input_schema, &input, sorted_keys, &mut used_names, &mut materialized_keys)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let order_by = window_expr
            .order_by()
            .iter()
            .map(|sort_expr| {
                Ok(NativeSortExpr::new(
                    window_key_expr(
                        &sort_expr.expr,
                        input_schema,
                        &input,
                        sorted_keys,
                        &mut used_names,
                        &mut materialized_keys,
                    )?,
                    sort_expr.options.descending,
                    sort_expr.options.nulls_first,
                ))
            })
            .collect::<Result<Vec<_>, LoweringFailure>>()?;
        let argument = window_function_argument(window_expr.as_ref(), input_schema)?
            .map(|argument| {
                window_key_expr(&argument, input_schema, &input, sorted_keys, &mut used_names, &mut materialized_keys)
            })
            .transpose()?;
        prepared_keys.push(PreparedWindowKeys { partition_by, order_by, argument });
    }

    if materialized_keys.is_empty() {
        return Ok(PreparedWindowInput { input, keys: prepared_keys });
    }

    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut expressions = input
        .schema
        .fields()
        .iter()
        .map(|field| NativeExpr::column(field.name().as_str()).alias(field.name().as_str()))
        .collect::<Vec<_>>();
    for key in materialized_keys {
        fields.push(key.field);
        expressions.push(key.expr.alias(key.name.as_str()));
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input.node, super::projection_outputs(expressions, &schema)?)?;
    Ok(PreparedWindowInput {
        input: LoweredNativeInput::with_bindings(node, schema, input.bindings),
        keys: prepared_keys,
    })
}

/// The expression a window aggregate or LAG / LEAD reads for each row. The
/// native window kernel reads columns, so a computed argument such as the
/// `1` of `COUNT(*)` or `x + 1` is materialized like a computed key.
fn window_function_argument(
    window_expr: &dyn WindowExpr,
    input_schema: &SchemaRef,
) -> Result<Option<Arc<dyn PhysicalExpr>>, LoweringFailure> {
    if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
        return match plain.get_aggregate_expr().expressions().as_slice() {
            [argument] => Ok(Some(Arc::clone(argument))),
            _ => Ok(None),
        };
    }
    if let Some(standard) = window_expr.as_any().downcast_ref::<StandardWindowExpr>()
        && matches!(standard_window_kind(standard)?, StandardWindowKind::Lag | StandardWindowKind::Lead)
    {
        return Ok(Some(parse_lag_lead_args(standard, input_schema)?.value_expr));
    }
    Ok(None)
}

struct MaterializedWindowKey {
    signature: String,
    name: String,
    field: Field,
    expr: NativeExpr,
}

fn window_key_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
    input: &LoweredNativeInput,
    sorted_keys: &SortedWindowKeys<'_>,
    used_names: &mut BTreeSet<String>,
    materialized_keys: &mut Vec<MaterializedWindowKey>,
) -> Result<NativeExpr, LoweringFailure> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let binding = input.binding_for_column(column)?;
        return Ok(NativeExpr::column(binding.native_name.as_str()));
    }
    if let Some((_, column)) = sorted_keys.iter().find(|(sort_key, _)| sort_key.as_ref() == expr.as_ref()) {
        return Ok(NativeExpr::column(column.as_str()));
    }

    let lowered = lower_physical_expr_with_bindings(expr, input_schema, Some(&input.bindings))?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    let nullable = expr.nullable(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    let signature = format!("{lowered}::{data_type:?}:nullable={nullable}");
    if let Some(existing) = materialized_keys.iter().find(|key| key.signature == signature) {
        return Ok(NativeExpr::column(existing.name.as_str()));
    }

    let name = unique_window_key_name(materialized_keys.len(), used_names);
    used_names.insert(name.clone());
    materialized_keys.push(MaterializedWindowKey {
        signature,
        name: name.clone(),
        field: Field::new(name.as_str(), data_type, nullable),
        expr: lowered,
    });
    Ok(NativeExpr::column(name.as_str()))
}

fn unique_window_key_name(index: usize, used_names: &BTreeSet<String>) -> String {
    let base = format!("__algeon_native_window_key_{index}");
    unique_name_with_suffix(base.as_str(), |candidate| !used_names.contains(candidate))
}

fn lower_window_spec(
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
    output_bindings: &[FieldBinding],
    output_schema: &SchemaRef,
    prepared_keys: &[PreparedWindowKeys],
) -> Result<NativeWindowSpec, LoweringFailure> {
    let input_field_count = input_schema.fields().len();
    if output_bindings.len() != input_field_count + window_exprs.len() || prepared_keys.len() != window_exprs.len() {
        return Err(CapabilityReason::UnsupportedExpression.into());
    }

    let first_keys = prepared_keys.first().ok_or(CapabilityReason::UnsupportedExpression)?;
    let expected_partition_by = first_keys.partition_by.clone();
    let expected_order_by = first_keys.order_by.clone();
    let mut functions = Vec::with_capacity(window_exprs.len());
    for (function_index, window_expr) in window_exprs.iter().enumerate() {
        let keys = prepared_keys.get(function_index).ok_or(CapabilityReason::UnsupportedExpression)?;
        let partition_by = keys.partition_by.clone();
        let order_by = keys.order_by.clone();
        if expected_partition_by != partition_by {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }
        if expected_order_by != order_by {
            return Err(CapabilityReason::UnsupportedOperator.into());
        }

        // The native output schema appends the window fields, renamed to their
        // native names, after the native input fields.
        let output = output_schema
            .fields()
            .len()
            .checked_sub(window_exprs.len() - function_index)
            .and_then(|index| output_schema.fields().get(index))
            .ok_or(CapabilityReason::UnsupportedExpression)?
            .as_ref()
            .clone();
        if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
            let aggregate = plain.get_aggregate_expr();
            let argument_count = aggregate.expressions().len();
            let expr = match &keys.argument {
                Some(argument) if argument_count == 1 => argument.clone(),
                _ => {
                    return Err(aggregate_blocked(
                        CapabilityReason::UnsupportedExpression,
                        aggregate.fun().name(),
                        AggregateDefect::ArgumentCount(argument_count),
                    )
                    .into());
                },
            };
            let frame = if is_whole_partition_frame(plain.get_window_frame()) {
                NativeWindowFrame::WholePartition
            } else if is_rows_unbounded_preceding_to_current_frame(plain.get_window_frame()) {
                NativeWindowFrame::RowsUnboundedPrecedingToCurrent
            } else {
                return Err(window_blocked(
                    CapabilityReason::UnsupportedOperator,
                    aggregate.fun().name(),
                    WindowDefect::UnsupportedFrame(Box::new(WindowFrame::clone(plain.get_window_frame()))),
                ));
            };
            functions.push(NativeWindowFunction::Aggregate {
                function: lower_native_aggregate_function(aggregate.fun())?,
                expr,
                output,
                frame,
            });
        } else if let Some(standard) = window_expr.as_any().downcast_ref::<StandardWindowExpr>() {
            let kind = standard_window_kind(standard)?;
            match kind {
                StandardWindowKind::Rank => functions.push(NativeWindowFunction::Rank { output }),
                StandardWindowKind::RowNumber => functions.push(NativeWindowFunction::RowNumber { output }),
                StandardWindowKind::Lag | StandardWindowKind::Lead => {
                    let direction = if kind == StandardWindowKind::Lag {
                        NativeShiftDirection::Lag
                    } else {
                        NativeShiftDirection::Lead
                    };
                    let args = parse_lag_lead_args(standard, input_schema)?;
                    let lowered_value = keys.argument.clone().ok_or(CapabilityReason::UnsupportedExpression)?;
                    functions.push(NativeWindowFunction::LagLead {
                        expr: lowered_value,
                        offset: args.offset,
                        default: args.default,
                        output,
                        direction,
                    });
                },
            }
        } else {
            return Err(unsupported_window_expr(window_expr));
        }
    }

    Ok(NativeWindowSpec::new(expected_partition_by, expected_order_by, functions))
}

fn rename_field(field: &Field, name: &str) -> Field {
    field.clone().with_name(name)
}

fn unique_window_output_name(
    ordinal: usize,
    frontend_name: &str,
    frontend_names: &BTreeSet<String>,
    used_native_names: &BTreeSet<String>,
) -> String {
    if !used_native_names.contains(frontend_name) {
        return frontend_name.to_string();
    }
    let sanitized = sanitize_field_name(frontend_name);
    let base = format!("__algeon_native_window_{ordinal}_{sanitized}");
    unique_name_with_suffix(base.as_str(), |candidate| {
        !frontend_names.contains(candidate) && !used_native_names.contains(candidate)
    })
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::FieldRef;
    use datafusion::{
        common::Result as DataFusionResult,
        logical_expr::{
            PartitionEvaluator, Signature, WindowUDF, WindowUDFImpl,
            function::{PartitionEvaluatorArgs, WindowUDFFieldArgs},
        },
        physical_expr::window::StandardWindowExpr,
        physical_plan::windows::create_udwf_window_expr,
    };

    use super::*;

    #[derive(Debug, PartialEq, Eq, Hash)]
    struct ShadowingRankFunc {
        inner: Rank,
    }

    impl WindowUDFImpl for ShadowingRankFunc {
        fn name(&self) -> &str {
            "rank"
        }

        fn signature(&self) -> &Signature {
            self.inner.signature()
        }

        fn partition_evaluator(&self, args: PartitionEvaluatorArgs) -> DataFusionResult<Box<dyn PartitionEvaluator>> {
            self.inner.partition_evaluator(args)
        }

        fn field(&self, args: WindowUDFFieldArgs) -> DataFusionResult<FieldRef> {
            self.inner.field(args)
        }
    }

    #[test]
    fn window_function_shadowed_by_name_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let shadowing = Arc::new(WindowUDF::new_from_impl(ShadowingRankFunc { inner: Rank::basic() }));
        let func_expr = create_udwf_window_expr(&shadowing, &[], schema.as_ref(), "rank".to_string(), false).unwrap();
        let standard = StandardWindowExpr::new(func_expr, &[], &[], Arc::new(WindowFrame::new(None)));
        assert!(standard_window_kind(&standard).is_err());
    }
}
