use std::{collections::BTreeSet, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};
use cudf_query_engine::{
    Expr as NativeExpr, QueryPlanBuilder as NativeQueryPlanBuilder, SortExpr as NativeSortExpr,
    WindowFrame as NativeWindowFrame, WindowFunction as NativeWindowFunction, WindowSpec as NativeWindowSpec,
};
use datafusion::{
    common::ScalarValue,
    logical_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits},
    physical_expr::{
        PhysicalExpr,
        expressions::Column,
        window::{PlainAggregateWindowExpr, StandardWindowExpr, WindowExpr},
    },
    physical_plan::{ExecutionPlan, windows::WindowUDFExpr},
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
    lower_sort_with_input(builder, sort.sort, input, _required_output_fields)
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
            validate_standard_window(standard)?;
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

fn validate_standard_window(standard: &StandardWindowExpr) -> Result<(), FallbackReason> {
    if !standard_window_function_name(standard).eq_ignore_ascii_case("rank") || standard.order_by().is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    Ok(())
}

fn standard_window_function_name(standard: &StandardWindowExpr) -> &str {
    let function = standard.get_standard_func_expr();
    function.as_any().downcast_ref::<WindowUDFExpr>().map_or_else(|| function.name(), |expr| expr.fun().name())
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
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
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
        } else if window_expr.as_any().downcast_ref::<StandardWindowExpr>().is_some() {
            functions.push(NativeWindowFunction::Rank { alias });
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
