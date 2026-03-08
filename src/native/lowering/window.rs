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
    physical_plan::{ExecutionPlan, windows::WindowUDFExpr},
};
use nexus_query_engine::{
    Expr as NativeExpr, LimitSpec as NativeLimitSpec, LiteralValue as NativeLiteralValue,
    QueryPlanBuilder as NativeQueryPlanBuilder, ShiftDirection as NativeShiftDirection, SortExpr as NativeSortExpr,
    WindowFrame as NativeWindowFrame, WindowFunction as NativeWindowFunction, WindowSpec as NativeWindowSpec,
};

use super::{
    FieldBinding, LoweredNativeInput, RequiredFields,
    aggregate::{lower_native_aggregate_function, project_required_output_columns},
    expr::{collect_required_field_indices, lower_physical_expr_with_bindings},
    lower_native_input,
    sort::lower_sort_with_input,
};
use crate::{
    native::{
        literal::{NativeLiteralContext, native_literal_value},
        normalize::{NormalizedNativeSort, NormalizedNativeWindow},
        validate::{is_supported_native_type, validate_physical_expr, validate_schema, validate_sort},
    },
    policy::FallbackReason,
};

pub(super) fn lower_sort_input(
    sort: &NormalizedNativeSort<'_>,
    builder: &mut NativeQueryPlanBuilder,
    _required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    validate_schema(sort.sort.schema())?;
    validate_sort(sort.sort)?;
    let mut child_required = BTreeSet::new();
    for sort_expr in sort.sort.expr() {
        collect_required_field_indices(&sort_expr.expr, &sort.sort.input().schema(), &mut child_required)?;
    }
    child_required.extend(_required_output_fields.requested_indices(sort.sort.schema().fields().len())?);
    let input = lower_native_input(&sort.input, builder, &RequiredFields::only(child_required))?;
    let sorted = lower_sort_with_input(builder, sort.sort, input, _required_output_fields)?;
    apply_sort_fetch_limit(builder, sorted, sort.fetch)
}

fn apply_sort_fetch_limit(
    builder: &mut NativeQueryPlanBuilder,
    sorted: LoweredNativeInput,
    fetch: Option<NativeLimitSpec>,
) -> Result<LoweredNativeInput, FallbackReason> {
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
) -> Result<LoweredNativeInput, FallbackReason> {
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

fn validate_window_exprs(window_exprs: &[Arc<dyn WindowExpr>], input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    if window_exprs.is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
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
            return Err(FallbackReason::UnsupportedOperator);
        }
    }
    Ok(())
}

fn validate_window_key_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    validate_physical_expr(expr, input_schema)?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(&data_type) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn validate_plain_aggregate_window(
    plain: &PlainAggregateWindowExpr,
    input_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let whole_partition = is_whole_partition_frame(plain.get_window_frame());
    let cumulative = is_rows_unbounded_preceding_to_current_frame(plain.get_window_frame());
    if !whole_partition && !cumulative {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if !plain.order_by().is_empty() && !cumulative {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if plain.order_by().is_empty() && !whole_partition {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let aggregate = plain.get_aggregate_expr();
    if aggregate.is_distinct() || !aggregate.order_bys().is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    lower_native_aggregate_function(aggregate.fun().name())?;
    let expressions = aggregate.expressions();
    let [arg] = expressions.as_slice() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    validate_physical_expr(arg, input_schema)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StandardWindowKind {
    Rank,
    RowNumber,
    Lag,
    Lead,
}

fn validate_standard_window(standard: &StandardWindowExpr, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    match standard_window_kind(standard)? {
        StandardWindowKind::Rank if standard.order_by().is_empty() => Err(FallbackReason::UnsupportedOperator),
        StandardWindowKind::Rank | StandardWindowKind::RowNumber => Ok(()),
        StandardWindowKind::Lag | StandardWindowKind::Lead => {
            if standard.order_by().is_empty() {
                return Err(FallbackReason::UnsupportedOperator);
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

fn standard_window_kind(standard: &StandardWindowExpr) -> Result<StandardWindowKind, FallbackReason> {
    match standard_window_function_name(standard).to_ascii_lowercase().as_str() {
        "rank" => Ok(StandardWindowKind::Rank),
        "row_number" => Ok(StandardWindowKind::RowNumber),
        "lag" => Ok(StandardWindowKind::Lag),
        "lead" => Ok(StandardWindowKind::Lead),
        _ => Err(FallbackReason::UnsupportedOperator),
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
fn parse_lag_lead_args(standard: &StandardWindowExpr, input_schema: &SchemaRef) -> Result<LagLeadArgs, FallbackReason> {
    let udf = standard
        .get_standard_func_expr()
        .as_any()
        .downcast_ref::<WindowUDFExpr>()
        .ok_or(FallbackReason::UnsupportedOperator)?;
    let args = udf.args();
    if args.is_empty() || args.len() > 3 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let value_expr = args[0].clone();
    validate_physical_expr(&value_expr, input_schema)?;
    let value_type =
        value_expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if !is_supported_native_type(&value_type) {
        return Err(FallbackReason::UnsupportedType);
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

fn parse_lag_lead_offset(expr: &Arc<dyn PhysicalExpr>) -> Result<u32, FallbackReason> {
    let literal = expr.downcast_ref::<Literal>().ok_or(FallbackReason::UnsupportedExpression)?;
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
            offset_from_i64(i64::try_from(*value).map_err(|_| FallbackReason::UnsupportedExpression)?)
        },
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

/// Reject offsets outside libcudf's signed shift range.
///
/// libcudf's `groupby::shift` uses `cudf::size_type` (`int32`), and the
/// engine negates the offset for `LEAD`, so the supported window is
/// `[0, i32::MAX]`. Rejecting larger offsets here means the plan falls back
/// to CPU DataFusion instead of admitting a native plan that would error at
/// execute time.
fn offset_from_i64(value: i64) -> Result<u32, FallbackReason> {
    if value < 0 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    let value = u64::try_from(value).map_err(|_| FallbackReason::UnsupportedExpression)?;
    if value > i32::MAX as u64 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(value as u32)
}

fn offset_from_u32(value: u32) -> Result<u32, FallbackReason> {
    if value > i32::MAX as u32 {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(value)
}

fn parse_lag_lead_default(
    expr: &Arc<dyn PhysicalExpr>,
    value_type: &arrow::datatypes::DataType,
) -> Result<Option<NativeLiteralValue>, FallbackReason> {
    let literal = expr.downcast_ref::<Literal>().ok_or(FallbackReason::UnsupportedExpression)?;
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
        return Err(FallbackReason::UnsupportedExpression);
    }
    // Mirror the literal families that the engine's
    // `build_lag_lead_value_fill_scalar` can lower to a cudf-rs Scalar today.
    // Decimal, date, timestamp, duration, and interval literals still need a
    // cudf-rs fill-scalar contract before the adapter can admit them; until
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
        | NativeLiteralValue::Timestamp { .. }
        | NativeLiteralValue::Duration { .. }
        | NativeLiteralValue::IntervalMonthDayNano { .. }
        | NativeLiteralValue::Null(_) => Err(FallbackReason::UnsupportedExpression),
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
) -> Result<(SchemaRef, Vec<FieldBinding>), FallbackReason> {
    if window_schema.fields().len() < input.bindings.len() {
        return Err(FallbackReason::UnsupportedExpression);
    }

    let frontend_names = window_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut used_native_names = input.schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut bindings = input.bindings.clone();

    for output_index in input.bindings.len()..window_schema.fields().len() {
        let field = window_schema.fields().get(output_index).ok_or(FallbackReason::UnsupportedExpression)?;
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
) -> Result<PreparedWindowInput, FallbackReason> {
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
            .collect::<Result<Vec<_>, FallbackReason>>()?;
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
) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let binding = input.binding_for_column(column)?;
        return Ok(NativeExpr::column(binding.native_name.as_str()));
    }

    let lowered = lower_physical_expr_with_bindings(expr, input_schema, Some(&input.bindings))?;
    let data_type = expr.data_type(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    let nullable = expr.nullable(input_schema.as_ref()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
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
    let base = format!("__cudf_window_key_{index}");
    if !used_names.contains(base.as_str()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if !used_names.contains(candidate.as_str()) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn lower_window_spec(
    window_exprs: &[Arc<dyn WindowExpr>],
    input_schema: &SchemaRef,
    input: &LoweredNativeInput,
    output_bindings: &[FieldBinding],
    prepared_keys: &[PreparedWindowKeys],
) -> Result<NativeWindowSpec, FallbackReason> {
    let input_field_count = input_schema.fields().len();
    if output_bindings.len() != input_field_count + window_exprs.len() || prepared_keys.len() != window_exprs.len() {
        return Err(FallbackReason::UnsupportedExpression);
    }

    let mut expected_partition_by = None;
    let mut expected_order_by = None;
    let mut functions = Vec::with_capacity(window_exprs.len());
    for (function_index, window_expr) in window_exprs.iter().enumerate() {
        let keys = prepared_keys.get(function_index).ok_or(FallbackReason::UnsupportedExpression)?;
        let partition_by = keys.partition_by.clone();
        let order_by = keys.order_by.clone();
        if let Some(expected) = &expected_partition_by {
            if expected != &partition_by {
                return Err(FallbackReason::UnsupportedOperator);
            }
        } else {
            expected_partition_by = Some(partition_by.clone());
        }
        if let Some(expected) = &expected_order_by {
            if expected != &order_by {
                return Err(FallbackReason::UnsupportedOperator);
            }
        } else {
            expected_order_by = Some(order_by.clone());
        }

        let alias = output_bindings
            .get(input_field_count + function_index)
            .ok_or(FallbackReason::UnsupportedExpression)?
            .native_name
            .clone();
        if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
            let aggregate = plain.get_aggregate_expr();
            let expressions = aggregate.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            let expr = lower_physical_expr_with_bindings(arg, input_schema, Some(&input.bindings))?;
            let frame = if is_whole_partition_frame(plain.get_window_frame()) {
                NativeWindowFrame::WholePartition
            } else if is_rows_unbounded_preceding_to_current_frame(plain.get_window_frame()) {
                NativeWindowFrame::RowsUnboundedPrecedingToCurrent
            } else {
                return Err(FallbackReason::UnsupportedOperator);
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
            return Err(FallbackReason::UnsupportedOperator);
        }
    }

    Ok(NativeWindowSpec::new(
        expected_partition_by.unwrap_or_default(),
        expected_order_by.unwrap_or_default(),
        functions,
    ))
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
    let base = format!("__cudf_window_{ordinal}_{sanitized}");
    if !frontend_names.contains(base.as_str()) && !used_native_names.contains(base.as_str()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if !frontend_names.contains(candidate.as_str()) && !used_native_names.contains(candidate.as_str()) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn sanitize_field_name(name: &str) -> String {
    let sanitized =
        name.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' }).collect::<String>();
    if sanitized.is_empty() { "field".to_string() } else { sanitized }
}
