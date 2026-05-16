use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::{
    common::ScalarValue,
    logical_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits},
    physical_expr::{
        PhysicalExpr,
        expressions::{Column, Literal},
        window::{PlainAggregateWindowExpr, StandardWindowExpr, WindowExpr},
    },
    physical_plan::windows::WindowUDFExpr,
};
use nexus_query_engine::{
    datatypes::{is_supported_list_carry_type, is_timestamp_with_timezone},
    expr::{
        Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralValue as NativeLiteralValue,
        SortExpr as NativeSortExpr,
    },
    plan::{
        QueryPlanBuilder as NativeQueryPlanBuilder, ShiftDirection as NativeShiftDirection,
        WindowFrame as NativeWindowFrame, WindowFunction as NativeWindowFunction, WindowSpec as NativeWindowSpec,
    },
};

use super::{
    FieldBinding, LoweredNativeInput, RequiredFields,
    aggregate::{lower_native_aggregate_function, project_required_output_columns},
    expr::{collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input,
    naming::{sanitize_field_name, unique_name_with_suffix},
    sort::lower_sort_ordering_with_input,
};
use crate::{
    capability::CapabilityReason,
    native::{
        expression_contract::{is_supported_native_type, validate_physical_expr},
        literal::{NativeLiteralContext, native_literal_value},
        normalize::{NormalizedNativeSort, NormalizedNativeWindow},
        validate::{validate_schema, validate_sort_ordering},
    },
};

pub(super) fn lower_sort_input(
    sort: &NormalizedNativeSort<'_>,
    builder: &mut NativeQueryPlanBuilder,
    _required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, CapabilityReason> {
    validate_schema(Arc::clone(sort.sort.output_schema()))?;
    validate_sort_ordering(sort.sort.ordering(), sort.sort.input_schema())?;
    let mut child_required = BTreeSet::new();
    for sort_expr in sort.sort.ordering() {
        collect_required_field_indices(&sort_expr.expr, sort.sort.input_schema(), &mut child_required)?;
    }
    child_required.extend(_required_output_fields.requested_indices(sort.sort.output_schema().fields().len())?);
    let input = lower_native_input(&sort.input, builder, &RequiredFields::only(child_required))?;
    let sorted = lower_sort_ordering_with_input(
        builder,
        sort.sort.ordering(),
        sort.sort.input_schema(),
        sort.sort.output_schema(),
        input,
        _required_output_fields,
    )?;
    apply_sort_fetch_limit(builder, sorted, sort.fetch)
}

fn apply_sort_fetch_limit(
    builder: &mut NativeQueryPlanBuilder,
    sorted: LoweredNativeInput,
    fetch: Option<NativeLimitSpec>,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let Some(fetch) = fetch else {
        return Ok(sorted);
    };
    let node = builder.add_limit(sorted.node, fetch, Arc::clone(&sorted.schema));
    Ok(LoweredNativeInput::with_bindings(node, sorted.schema, sorted.bindings))
}

pub(super) fn lower_window_input(
    window: &NormalizedNativeWindow<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, CapabilityReason> {
    let window_schema = window.window.schema();
    let window_input_schema = window.window.input_schema();
    validate_schema(Arc::clone(&window_schema))?;
    validate_window_exprs(window.window.window_expr(), &window_input_schema)?;

    let input = lower_native_input(&window.input, builder, &RequiredFields::all())?;
    let prepared =
        materialize_window_key_expressions(builder, window.window.window_expr(), &window_input_schema, input)?;
    let (output_schema, output_bindings) = native_window_output_schema_and_bindings(&prepared.input, &window_schema)?;
    let spec = lower_window_spec(
        window.window.window_expr(),
        &window_input_schema,
        &prepared.input,
        &output_bindings,
        &prepared.keys,
    )?;
    let window_node = builder.add_window(prepared.input.node, spec, Arc::clone(&output_schema));
    project_required_output_columns(builder, window_node, output_schema, output_bindings, required_output_fields)
}

fn validate_window_exprs(
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
) -> Result<(), CapabilityReason> {
    if window_exprs.is_empty() {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    for window_expr in window_exprs {
        for expr in window_expr.partition_by() {
            validate_window_key_expr(expr, input_schema)?;
        }
        for sort_expr in window_expr.order_by() {
            validate_window_key_expr(&sort_expr.expr, input_schema)?;
        }
        if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
            validate_plain_aggregate_window(plain, input_schema)?;
        } else if let Some(standard) = window_expr.as_any().downcast_ref::<StandardWindowExpr>() {
            validate_standard_window(standard, input_schema)?;
        } else {
            return Err(CapabilityReason::UnsupportedOperator);
        }
    }
    Ok(())
}

fn validate_window_key_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), CapabilityReason> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if is_supported_native_type(&data_type)
        && !is_timestamp_with_timezone(&data_type)
        && !is_supported_list_carry_type(&data_type)
    {
        Ok(())
    } else {
        Err(CapabilityReason::UnsupportedType)
    }
}

fn validate_plain_aggregate_window(
    plain: &PlainAggregateWindowExpr,
    input_schema: &SchemaRef,
) -> Result<(), CapabilityReason> {
    let whole_partition = is_whole_partition_frame(plain.get_window_frame());
    let cumulative = is_rows_unbounded_preceding_to_current_frame(plain.get_window_frame());
    if !whole_partition && !cumulative {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    if !plain.order_by().is_empty() && !cumulative {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    if plain.order_by().is_empty() && !whole_partition {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    let aggregate = plain.get_aggregate_expr();
    if aggregate.is_distinct() || !aggregate.order_bys().is_empty() {
        return Err(CapabilityReason::UnsupportedOperator);
    }
    lower_native_aggregate_function(aggregate.fun().name())?;
    let expressions = aggregate.expressions();
    let [arg] = expressions.as_slice() else {
        return Err(CapabilityReason::UnsupportedExpression);
    };
    validate_window_value_expr(arg, input_schema)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StandardWindowKind {
    Rank,
    RowNumber,
    Lag,
    Lead,
}

fn validate_standard_window(standard: &StandardWindowExpr, input_schema: &SchemaRef) -> Result<(), CapabilityReason> {
    match standard_window_kind(standard)? {
        StandardWindowKind::Rank if standard.order_by().is_empty() => Err(CapabilityReason::UnsupportedOperator),
        StandardWindowKind::Rank | StandardWindowKind::RowNumber => Ok(()),
        StandardWindowKind::Lag | StandardWindowKind::Lead => {
            if standard.order_by().is_empty() {
                return Err(CapabilityReason::UnsupportedOperator);
            }
            // Parsing the offset / default literals here keeps validation
            // and lowering in sync. We do not commit a fallback reason for
            // parse failures because lowering will surface a structured
            // rejection if the literal arguments are not constant.
            let _ = parse_lag_lead_args(standard, input_schema)?;
            Ok(())
        },
    }
}

fn standard_window_kind(standard: &StandardWindowExpr) -> Result<StandardWindowKind, CapabilityReason> {
    match standard_window_function_name(standard).to_ascii_lowercase().as_str() {
        "rank" => Ok(StandardWindowKind::Rank),
        "row_number" => Ok(StandardWindowKind::RowNumber),
        "lag" => Ok(StandardWindowKind::Lag),
        "lead" => Ok(StandardWindowKind::Lead),
        _ => Err(CapabilityReason::UnsupportedOperator),
    }
}

fn standard_window_function_name(standard: &StandardWindowExpr) -> &str {
    let function = standard.get_standard_func_expr();
    function.as_any().downcast_ref::<WindowUDFExpr>().map_or_else(|| function.name(), |expr| expr.fun().name())
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
/// `offset` must be a non-negative integer literal fitting `u32`, or null for
/// SQL's default `1`; `default` must be a scalar literal.
///
/// Use `WindowUDFExpr::args()` because
/// [`StandardWindowFunctionExpr::expressions`] hides `offset` and `default`.
fn parse_lag_lead_args(
    standard: &StandardWindowExpr,
    input_schema: &SchemaRef,
) -> Result<LagLeadArgs, CapabilityReason> {
    let udf = standard
        .get_standard_func_expr()
        .as_any()
        .downcast_ref::<WindowUDFExpr>()
        .ok_or(CapabilityReason::UnsupportedOperator)?;
    let args = udf.args();
    if args.is_empty() || args.len() > 3 {
        return Err(CapabilityReason::UnsupportedExpression);
    }
    let value_expr = args[0].clone();
    validate_physical_expr(&value_expr, input_schema)?;
    let value_type =
        value_expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if !is_supported_native_type(&value_type)
        || is_timestamp_with_timezone(&value_type)
        || is_supported_list_carry_type(&value_type)
    {
        return Err(CapabilityReason::UnsupportedType);
    }
    let offset = match args.get(1) {
        None => 1u32,
        Some(expr) => parse_lag_lead_offset(expr)?,
    };
    let default = match args.get(2) {
        None => None,
        Some(expr) => parse_lag_lead_default(expr, &value_type)?,
    };
    Ok(LagLeadArgs { value_expr, offset, default })
}

fn validate_window_value_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), CapabilityReason> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| CapabilityReason::UnsupportedExpression)?;
    if is_supported_native_type(&data_type)
        && !is_timestamp_with_timezone(&data_type)
        && !is_supported_list_carry_type(&data_type)
    {
        Ok(())
    } else {
        Err(CapabilityReason::UnsupportedType)
    }
}

fn parse_lag_lead_offset(expr: &Arc<dyn PhysicalExpr>) -> Result<u32, CapabilityReason> {
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
        _ => Err(CapabilityReason::UnsupportedExpression),
    }
}

/// Reject offsets outside libcudf's signed shift range.
///
/// libcudf's `groupby::shift` uses `cudf::size_type` (`int32`), and the
/// engine negates the offset for `LEAD`, so the supported window is
/// `[0, i32::MAX]`. Rejecting larger offsets here means the plan falls back
/// to CPU DataFusion instead of admitting a native plan that would error at
/// execute time.
fn offset_from_i64(value: i64) -> Result<u32, CapabilityReason> {
    if value < 0 {
        return Err(CapabilityReason::UnsupportedExpression);
    }
    let value = u64::try_from(value).map_err(|_| CapabilityReason::UnsupportedExpression)?;
    if value > i32::MAX as u64 {
        return Err(CapabilityReason::UnsupportedExpression);
    }
    Ok(value as u32)
}

fn offset_from_u32(value: u32) -> Result<u32, CapabilityReason> {
    if value > i32::MAX as u32 {
        return Err(CapabilityReason::UnsupportedExpression);
    }
    Ok(value)
}

fn parse_lag_lead_default(
    expr: &Arc<dyn PhysicalExpr>,
    value_type: &arrow::datatypes::DataType,
) -> Result<Option<NativeLiteralValue>, CapabilityReason> {
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
        return Err(CapabilityReason::UnsupportedExpression);
    }
    // Mirror the literal families that the engine's
    // `build_lag_lead_value_fill_scalar` can lower to a cudf Scalar today.
    // Decimal, date, timestamp, duration, and interval literals still need a
    // cudf fill-scalar contract before the adapter can admit them; until
    // then we reject so the query falls back to CPU instead of failing at
    // execute time.
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
        | NativeLiteralValue::Null(_) => Err(CapabilityReason::UnsupportedExpression),
    }
}

fn is_whole_partition_frame(frame: &WindowFrame) -> bool {
    frame.units == WindowFrameUnits::Rows
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
) -> Result<(SchemaRef, Vec<FieldBinding>), CapabilityReason> {
    if window_schema.fields().len() < input.bindings.len() {
        return Err(CapabilityReason::UnsupportedExpression);
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
}

fn materialize_window_key_expressions(
    builder: &mut NativeQueryPlanBuilder,
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
    input: LoweredNativeInput,
) -> Result<PreparedWindowInput, CapabilityReason> {
    let mut used_names = input.schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut materialized_keys = Vec::<MaterializedWindowKey>::new();
    let mut prepared_keys = Vec::with_capacity(window_exprs.len());

    for window_expr in window_exprs {
        let partition_by = window_expr
            .partition_by()
            .iter()
            .map(|expr| window_key_expr(expr, input_schema, &input, &mut used_names, &mut materialized_keys))
            .collect::<Result<Vec<_>, _>>()?;
        let order_by = window_expr
            .order_by()
            .iter()
            .map(|sort_expr| {
                Ok(NativeSortExpr::new(
                    window_key_expr(&sort_expr.expr, input_schema, &input, &mut used_names, &mut materialized_keys)?,
                    sort_expr.options.descending,
                    sort_expr.options.nulls_first,
                ))
            })
            .collect::<Result<Vec<_>, CapabilityReason>>()?;
        prepared_keys.push(PreparedWindowKeys { partition_by, order_by });
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
    let node = builder.add_projection(input.node, super::projection_outputs(expressions, &schema)?);
    Ok(PreparedWindowInput {
        input: LoweredNativeInput::with_bindings(node, schema, input.bindings),
        keys: prepared_keys,
    })
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
    used_names: &mut BTreeSet<String>,
    materialized_keys: &mut Vec<MaterializedWindowKey>,
) -> Result<NativeExpr, CapabilityReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let binding = input.binding_for_column(column)?;
        return Ok(NativeExpr::column(binding.native_name.as_str()));
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
    let base = format!("__nexus_native_window_key_{index}");
    unique_name_with_suffix(base.as_str(), |candidate| !used_names.contains(candidate))
}

fn lower_window_spec(
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
    input: &LoweredNativeInput,
    output_bindings: &[FieldBinding],
    prepared_keys: &[PreparedWindowKeys],
) -> Result<NativeWindowSpec, CapabilityReason> {
    let input_field_count = input_schema.fields().len();
    if output_bindings.len() != input_field_count + window_exprs.len() || prepared_keys.len() != window_exprs.len() {
        return Err(CapabilityReason::UnsupportedExpression);
    }

    let first_keys = &prepared_keys[0];
    let expected_partition_by = first_keys.partition_by.clone();
    let expected_order_by = first_keys.order_by.clone();
    let mut functions = Vec::with_capacity(window_exprs.len());
    for (function_index, window_expr) in window_exprs.iter().enumerate() {
        let keys = prepared_keys.get(function_index).ok_or(CapabilityReason::UnsupportedExpression)?;
        let partition_by = keys.partition_by.clone();
        let order_by = keys.order_by.clone();
        if expected_partition_by != partition_by {
            return Err(CapabilityReason::UnsupportedOperator);
        }
        if expected_order_by != order_by {
            return Err(CapabilityReason::UnsupportedOperator);
        }

        let alias = output_bindings
            .get(input_field_count + function_index)
            .ok_or(CapabilityReason::UnsupportedExpression)?
            .native_name
            .clone();
        if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
            let aggregate = plain.get_aggregate_expr();
            let expressions = aggregate.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(CapabilityReason::UnsupportedExpression);
            };
            let expr = lower_physical_expr_with_bindings(arg, input_schema, Some(&input.bindings))?;
            let frame = if is_whole_partition_frame(plain.get_window_frame()) {
                NativeWindowFrame::WholePartition
            } else if is_rows_unbounded_preceding_to_current_frame(plain.get_window_frame()) {
                NativeWindowFrame::RowsUnboundedPrecedingToCurrent
            } else {
                return Err(CapabilityReason::UnsupportedOperator);
            };
            functions.push(NativeWindowFunction::Aggregate {
                function: lower_native_aggregate_function(aggregate.fun().name())?,
                expr,
                alias,
                frame,
            });
        } else if let Some(standard) = window_expr.as_any().downcast_ref::<StandardWindowExpr>() {
            match standard_window_kind(standard)? {
                StandardWindowKind::Rank => functions.push(NativeWindowFunction::Rank { alias }),
                StandardWindowKind::RowNumber => functions.push(NativeWindowFunction::RowNumber { alias }),
                StandardWindowKind::Lag | StandardWindowKind::Lead => {
                    let direction = match standard_window_kind(standard)? {
                        StandardWindowKind::Lag => NativeShiftDirection::Lag,
                        StandardWindowKind::Lead => NativeShiftDirection::Lead,
                        _ => unreachable!(),
                    };
                    let args = parse_lag_lead_args(standard, input_schema)?;
                    let lowered_value =
                        lower_physical_expr_with_bindings(&args.value_expr, input_schema, Some(&input.bindings))?;
                    functions.push(NativeWindowFunction::LagLead {
                        expr: lowered_value,
                        offset: args.offset,
                        default: args.default,
                        alias,
                        direction,
                    });
                },
            }
        } else {
            return Err(CapabilityReason::UnsupportedOperator);
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
    let base = format!("__nexus_native_window_{ordinal}_{sanitized}");
    unique_name_with_suffix(base.as_str(), |candidate| {
        !frontend_names.contains(candidate) && !used_native_names.contains(candidate)
    })
}
