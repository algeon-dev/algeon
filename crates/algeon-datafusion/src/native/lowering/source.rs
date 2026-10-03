use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    sync::Arc,
};

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::{
    common::Constraint,
    physical_expr::{PhysicalExpr, expressions::Column, projection::ProjectionExprs},
    physical_plan::{ExecutionPlan, filter::FilterExec},
};
use query_engine::{
    expr::Expr as NativeExpr,
    plan::{NodeId, QueryPlanBuilder as NativeQueryPlanBuilder},
    source::SourceDiagnostics as NativeSourceDiagnostics,
};

mod local_parquet;
mod reader_pushdown;

use local_parquet::lower_local_parquet_source_input;
pub(super) use reader_pushdown::{
    reader_predicate_proves_local_parquet_at_most_one_row, source_reader_pushdown_predicate,
};

use super::{
    LoweredNativeInput, RequiredFields, executable_schema_and_bindings,
    expr::{alias_if_needed, lower_physical_expr, lower_physical_expr_with_bindings},
    projection_outputs,
};
use crate::{
    capability::CapabilityReason,
    native::{LoweringFailure, normalize::NormalizedNativeSource},
};

pub(super) fn lower_source_input(
    source: &NormalizedNativeSource<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
    pushed_scan_limit: Option<usize>,
) -> Result<LoweredNativeInput, LoweringFailure> {
    lower_local_parquet_source_input(source.scan, source.filter, builder, required_output_fields, pushed_scan_limit)
}

pub(super) fn combine_scan_limits(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(limit), None) | (None, Some(limit)) => Some(limit),
        (None, None) => None,
    }
}

pub(super) fn preserve_resolved_source_diagnostics(
    diagnostics: &mut NativeSourceDiagnostics,
    resolved_source_diagnostics: &NativeSourceDiagnostics,
) {
    diagnostics.source_read_strategy = resolved_source_diagnostics.source_read_strategy.clone();
    diagnostics.credential_mode = resolved_source_diagnostics.credential_mode.clone();
    diagnostics.lakehouse_cache_status = resolved_source_diagnostics.lakehouse_cache_status.clone();
    diagnostics.lakehouse_cache_read_mode = resolved_source_diagnostics.lakehouse_cache_read_mode.clone();
    diagnostics.lakehouse_cache_fill_mode = resolved_source_diagnostics.lakehouse_cache_fill_mode.clone();
    diagnostics.direct_device_read =
        diagnostics.rejected_features.is_empty() && resolved_source_diagnostics.direct_device_read;
}

pub(super) enum SourceUnprojectedFilterOutput {
    PreserveScanOutput,
    ProjectRequiredOutput,
}

pub(super) fn project_scan_output_columns(
    builder: &mut NativeQueryPlanBuilder,
    input_node: NodeId,
    input_schema: SchemaRef,
    input_bindings: Vec<super::FieldBinding>,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let selected_bindings = selected_source_bindings_in_frontend_order(&input_bindings, required_output_fields)?;
    if selected_bindings.len() == input_schema.fields().len()
        && selected_bindings.iter().zip(input_bindings.iter()).all(|(selected, input)| {
            selected.frontend_index == input.frontend_index && selected.native_name == input.native_name
        })
    {
        return Ok(LoweredNativeInput::with_bindings(input_node, input_schema, input_bindings));
    }

    let mut fields_by_name = HashMap::with_capacity(input_schema.fields().len());
    for field in input_schema.fields() {
        fields_by_name.entry(field.name().as_str()).or_insert(field);
    }
    let mut expressions = Vec::with_capacity(selected_bindings.len());
    let mut fields = Vec::with_capacity(selected_bindings.len());
    for binding in &selected_bindings {
        expressions.push(NativeExpr::column(binding.native_name.as_str()));
        let field = fields_by_name.get(binding.native_name.as_str()).ok_or(CapabilityReason::UnsupportedExpression)?;
        fields.push(Arc::clone(field));
    }
    let schema = Arc::new(Schema::new(fields));
    let node = builder.add_projection(input_node, projection_outputs(expressions, &schema)?)?;
    Ok(LoweredNativeInput::with_bindings(node, schema, selected_bindings))
}

fn selected_source_bindings_in_frontend_order(
    input_bindings: &[super::FieldBinding],
    required_output_fields: &RequiredFields,
) -> Result<Vec<super::FieldBinding>, LoweringFailure> {
    let mut selected_bindings = match required_output_fields {
        RequiredFields::All => input_bindings.to_vec(),
        RequiredFields::Only { .. } => {
            let frontend_field_count =
                input_bindings.iter().map(|binding| binding.frontend_index).max().map_or(0, |index| index + 1);
            required_output_fields
                .requested_indices(frontend_field_count)?
                .into_iter()
                .map(|frontend_index| {
                    input_bindings
                        .iter()
                        .find(|binding| binding.frontend_index == frontend_index)
                        .cloned()
                        .ok_or(CapabilityReason::UnsupportedExpression)
                })
                .collect::<Result<Vec<_>, _>>()?
        },
    };
    selected_bindings.sort_by_key(|binding| binding.frontend_index);
    Ok(selected_bindings)
}

pub(super) fn source_bindings_for_schema(schema: &SchemaRef, frontend_schema: SchemaRef) -> Vec<super::FieldBinding> {
    // Each native field binds the lowest unused frontend index with its name, so duplicate names
    // consume their frontend positions in order.
    let mut frontend_indices_by_name = HashMap::<&str, VecDeque<usize>>::new();
    for (index, frontend_field) in frontend_schema.fields().iter().enumerate() {
        frontend_indices_by_name.entry(frontend_field.name().as_str()).or_default().push_back(index);
    }
    schema
        .fields()
        .iter()
        .filter_map(|field| {
            let frontend_index = frontend_indices_by_name.get_mut(field.name().as_str())?.pop_front()?;
            Some(super::FieldBinding {
                frontend_index,
                frontend_name: field.name().clone(),
                native_name: field.name().clone(),
            })
        })
        .collect()
}

pub(super) fn source_unique_key_columns(
    source: &dyn ExecutionPlan,
    bindings: &[super::FieldBinding],
) -> Vec<BTreeSet<String>> {
    let schema = source.schema();
    source
        .properties()
        .equivalence_properties()
        .constraints()
        .iter()
        .filter_map(|constraint| {
            let indices = match constraint {
                Constraint::PrimaryKey(indices) | Constraint::Unique(indices) => indices,
            };
            if indices.is_empty() || indices.iter().copied().collect::<BTreeSet<_>>().len() != indices.len() {
                return None;
            }
            let fields = indices.iter().map(|index| schema.fields().get(*index)).collect::<Option<Vec<_>>>()?;
            if fields.iter().any(|field| field.is_nullable()) {
                return None;
            }
            indices
                .iter()
                .map(|index| {
                    bindings
                        .iter()
                        .find(|binding| binding.frontend_index == *index)
                        .map(|binding| binding.native_name.clone())
                })
                .collect::<Option<BTreeSet<_>>>()
        })
        .collect()
}

pub(super) fn required_source_scan_output_indices(
    source_schema: SchemaRef,
    filter: Option<&FilterExec>,
    required_output_fields: &RequiredFields,
) -> Result<Vec<usize>, CapabilityReason> {
    if let Some(filter) = filter
        && let Some(projection) = filter.projection()
    {
        return required_output_fields
            .requested_indices(filter.schema().fields().len())?
            .into_iter()
            .map(|output_index| projection.get(output_index).copied().ok_or(CapabilityReason::UnsupportedExpression))
            .collect();
    }
    required_output_fields.requested_indices(source_schema.fields().len())
}

pub(super) fn add_source_filter_projection(
    builder: &mut NativeQueryPlanBuilder,
    filtered_node: NodeId,
    scan_output: &LoweredNativeInput,
    filter_schema: SchemaRef,
    projection: &[usize],
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, LoweringFailure> {
    let requested_output_indices = required_output_fields.requested_indices(filter_schema.fields().len())?;
    let requested = RequiredFields::only(requested_output_indices.iter().copied().collect());
    let (schema, bindings) = executable_schema_and_bindings(&filter_schema, &requested)?;
    let expressions = requested_output_indices
        .iter()
        .zip(bindings.iter())
        .map(|(output_index, output_binding)| {
            let input_index = projection.get(*output_index).copied().ok_or(CapabilityReason::UnsupportedExpression)?;
            let input_binding = scan_output.binding_for_frontend_index(input_index)?;
            Ok(alias_if_needed(
                NativeExpr::column(input_binding.native_name.as_str()),
                output_binding.native_name.as_str(),
            ))
        })
        .collect::<Result<Vec<_>, LoweringFailure>>()?;
    let node = builder.add_projection(filtered_node, projection_outputs(expressions, &schema)?)?;
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

pub(super) fn lower_source_filter_and_projection(
    builder: &mut NativeQueryPlanBuilder,
    scan_output: LoweredNativeInput,
    filter: &FilterExec,
    reader_pushdown_covers_filter: bool,
    required_output_fields: &RequiredFields,
    unprojected_output: SourceUnprojectedFilterOutput,
) -> Result<LoweredNativeInput, LoweringFailure> {
    if reader_pushdown_covers_filter {
        if let Some(projection) = filter.projection() {
            return add_source_filter_projection(
                builder,
                scan_output.node,
                &scan_output,
                filter.schema(),
                projection,
                required_output_fields,
            );
        }
        return unprojected_source_filter_output(builder, scan_output, required_output_fields, unprojected_output);
    }

    let filtered_node = builder.add_filter(
        scan_output.node,
        lower_physical_expr_with_bindings(filter.predicate(), &filter.input().schema(), Some(&scan_output.bindings))?,
    )?;
    if let Some(projection) = filter.projection() {
        add_source_filter_projection(
            builder,
            filtered_node,
            &scan_output,
            filter.schema(),
            projection,
            required_output_fields,
        )
    } else if matches!(unprojected_output, SourceUnprojectedFilterOutput::ProjectRequiredOutput) {
        project_scan_output_columns(
            builder,
            filtered_node,
            scan_output.schema,
            scan_output.bindings,
            required_output_fields,
        )
    } else {
        Ok(LoweredNativeInput::with_bindings(filtered_node, scan_output.schema, scan_output.bindings))
    }
}

fn unprojected_source_filter_output(
    builder: &mut NativeQueryPlanBuilder,
    scan_output: LoweredNativeInput,
    required_output_fields: &RequiredFields,
    unprojected_output: SourceUnprojectedFilterOutput,
) -> Result<LoweredNativeInput, LoweringFailure> {
    match unprojected_output {
        SourceUnprojectedFilterOutput::PreserveScanOutput => Ok(scan_output),
        SourceUnprojectedFilterOutput::ProjectRequiredOutput => project_scan_output_columns(
            builder,
            scan_output.node,
            scan_output.schema,
            scan_output.bindings,
            required_output_fields,
        ),
    }
}

/// Scan projection expression retained because downstream native nodes need it.
pub(super) struct RequiredScanProjectionExpr<'a> {
    pub(super) output_index: usize,
    pub(super) expr: &'a Arc<dyn PhysicalExpr>,
    pub(super) alias: &'a str,
}

pub(super) fn required_scan_projection<'a>(
    projection: &'a ProjectionExprs,
    required_scan_output_columns: &[String],
) -> Vec<RequiredScanProjectionExpr<'a>> {
    let required_scan_output_columns = required_scan_output_columns.iter().map(String::as_str).collect::<HashSet<_>>();
    projection
        .iter()
        .enumerate()
        .filter(|(_output_index, projection_expr)| {
            required_scan_output_columns.contains(projection_expr.alias.as_str())
        })
        .map(|(output_index, projection_expr)| RequiredScanProjectionExpr {
            output_index,
            expr: &projection_expr.expr,
            alias: projection_expr.alias.as_str(),
        })
        .collect()
}

pub(super) fn lower_required_file_source_projection_exprs(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> Result<Vec<NativeExpr>, LoweringFailure> {
    projection
        .iter()
        .map(|projection_expr| {
            let expr = lower_physical_expr(projection_expr.expr, input_schema)?;
            Ok(alias_if_needed(expr, projection_expr.alias))
        })
        .collect()
}

pub(super) fn required_scan_projection_requires_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
) -> bool {
    projection.iter().any(|projection_expr| {
        let Some(column) = projection_expr.expr.downcast_ref::<Column>() else {
            return true;
        };
        input_schema.fields().get(column.index()).is_none_or(|field| field.name() != projection_expr.alias)
    })
}

pub(super) fn required_scan_projection_needs_native_projection(
    projection: &[RequiredScanProjectionExpr<'_>],
    input_schema: &SchemaRef,
    current_source_schema: &SchemaRef,
    scan_output_schema: SchemaRef,
) -> Result<bool, LoweringFailure> {
    if required_scan_projection_requires_native_projection(projection, input_schema) {
        return Ok(true);
    }
    let projected_schema = required_scan_projection_schema(projection, scan_output_schema)?;
    Ok(projected_schema.as_ref() != current_source_schema.as_ref())
}

pub(super) fn required_scan_projection_schema(
    projection: &[RequiredScanProjectionExpr<'_>],
    scan_output_schema: SchemaRef,
) -> Result<SchemaRef, LoweringFailure> {
    let fields = projection
        .iter()
        .map(|projection_expr| {
            scan_output_schema
                .fields()
                .get(projection_expr.output_index)
                .cloned()
                .ok_or(CapabilityReason::UnsupportedExpression)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{RequiredFields, selected_source_bindings_in_frontend_order};
    use crate::native::lowering::FieldBinding;

    fn binding(frontend_index: usize, name: &str) -> FieldBinding {
        FieldBinding { frontend_index, frontend_name: name.to_string(), native_name: name.to_string() }
    }

    #[test]
    fn selected_source_bindings_all_uses_frontend_schema_order() {
        let input = vec![binding(1, "dst"), binding(0, "src"), binding(2, "weight")];

        let selected = selected_source_bindings_in_frontend_order(&input, &RequiredFields::all()).unwrap();

        assert_eq!(
            selected.iter().map(|binding| binding.native_name.as_str()).collect::<Vec<_>>(),
            vec!["src", "dst", "weight"],
        );
    }

    #[test]
    fn selected_source_bindings_only_uses_requested_frontend_order() {
        let input = vec![binding(1, "dst"), binding(0, "src"), binding(2, "weight")];
        let selected =
            selected_source_bindings_in_frontend_order(&input, &RequiredFields::only(BTreeSet::from([0, 2]))).unwrap();

        assert_eq!(
            selected.iter().map(|binding| binding.native_name.as_str()).collect::<Vec<_>>(),
            vec!["src", "weight"],
        );
    }
}
