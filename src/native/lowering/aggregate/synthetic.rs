use std::{collections::HashMap, sync::Arc};

use arrow::datatypes::Field;
use datafusion::physical_expr::PhysicalExpr;

use super::*;
use crate::native::lowering::{
    expr::physical_expr_field,
    naming::{sanitize_field_name, unique_name_with_suffix},
};

// Projection that materializes aggregate inputs into executable native columns.
pub(super) struct SyntheticAggregateInputProjection {
    pub(super) expressions: Vec<NativeExpr>,
    pub(super) schema: SchemaRef,
    pub(super) bindings: Vec<FieldBinding>,
    pub(super) group_expr_aliases: Option<Vec<String>>,
    pub(super) aggregate_arg_aliases: Vec<String>,
    pub(super) filter_aliases: Vec<Option<String>>,
}

pub(super) fn synthetic_aggregate_input_projection(
    aggregate: &AggregateExec,
    input: &LoweredNativeInput,
) -> Result<SyntheticAggregateInputProjection, CapabilityReason> {
    let frontend_schema = aggregate.input().schema();
    let mut expressions = Vec::new();
    let mut fields = Vec::new();
    let group_expr_aliases =
        append_synthetic_group_expr_projections(aggregate, input, &frontend_schema, &mut expressions, &mut fields)?;

    let mut aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
    let mut filter_aliases = vec![None; aggregate.aggr_expr().len()];
    let mut filters_by_expression = HashMap::<String, String>::new();
    for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
        let expressions_for_aggregate = aggregate_expr.expressions();
        let [arg] = expressions_for_aggregate.as_slice() else {
            return Err(CapabilityReason::UnsupportedExpression);
        };
        // cuDF aggregate lowering consumes columns, so SQL aggregate arguments
        // such as SUM(m00 + m01) are first materialized as synthetic projection columns.
        let alias = unique_synthetic_aggregate_arg_alias(index, &input.schema, &fields);
        expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(arg, &frontend_schema, Some(&input.bindings))?,
            alias.as_str(),
        ));
        fields.push(physical_expr_field(arg, &frontend_schema, alias.as_str())?);
        aggregate_arg_aliases.push(alias);
        if let Some(filter) = aggregate.filter_expr().get(index).and_then(Option::as_ref) {
            append_synthetic_filter_projection(
                index,
                filter,
                input,
                &frontend_schema,
                &mut expressions,
                &mut fields,
                &mut filter_aliases,
                &mut filters_by_expression,
            )?;
        }
    }

    Ok(SyntheticAggregateInputProjection {
        expressions,
        schema: Arc::new(Schema::new(fields)),
        bindings: Vec::new(),
        group_expr_aliases: Some(group_expr_aliases),
        aggregate_arg_aliases,
        filter_aliases,
    })
}

pub(super) fn synthetic_aggregate_input_projection_if_needed(
    aggregate: &AggregateExec,
    input: &LoweredNativeInput,
) -> Result<Option<SyntheticAggregateInputProjection>, CapabilityReason> {
    let frontend_schema = aggregate.input().schema();
    let needs_group_projection =
        aggregate.group_expr().expr().iter().any(|(group_expr, _alias)| group_expr.downcast_ref::<Column>().is_none());
    let needs_aggregate_arg_projection = aggregate.aggr_expr().iter().any(|aggregate_expr| {
        let expressions = aggregate_expr.expressions();
        !matches!(expressions.as_slice(), [arg] if arg.downcast_ref::<Column>().is_some())
    });
    let needs_filter_projection = aggregate.filter_expr().iter().any(Option::is_some);
    let needs_projection = needs_group_projection || needs_aggregate_arg_projection || needs_filter_projection;
    if !needs_projection {
        return Ok(None);
    }

    let mut expressions =
        input.schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();

    let group_expr_aliases = if needs_group_projection {
        Some(append_synthetic_group_expr_projections(
            aggregate,
            input,
            &frontend_schema,
            &mut expressions,
            &mut fields,
        )?)
    } else {
        None
    };

    let mut aggregate_arg_aliases = Vec::new();
    if needs_aggregate_arg_projection {
        aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
        for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
            let expressions_for_aggregate = aggregate_expr.expressions();
            let [arg] = expressions_for_aggregate.as_slice() else {
                return Err(CapabilityReason::UnsupportedExpression);
            };
            let alias = unique_synthetic_aggregate_arg_alias(index, &input.schema, &fields);
            expressions.push(alias_if_needed(
                lower_physical_expr_with_bindings(arg, &frontend_schema, Some(&input.bindings))?,
                alias.as_str(),
            ));
            fields.push(physical_expr_field(arg, &frontend_schema, alias.as_str())?);
            aggregate_arg_aliases.push(alias);
        }
    }
    let mut filter_aliases = vec![None; aggregate.aggr_expr().len()];
    if needs_filter_projection {
        append_synthetic_filter_projections(
            aggregate,
            input,
            &frontend_schema,
            &mut expressions,
            &mut fields,
            &mut filter_aliases,
        )?;
    }
    let mut bindings = input.bindings.clone();
    bindings.extend(aggregate_arg_aliases.iter().enumerate().map(|(index, alias)| FieldBinding {
        frontend_index: usize::MAX - index,
        frontend_name: alias.clone(),
        native_name: alias.clone(),
    }));

    Ok(Some(SyntheticAggregateInputProjection {
        expressions,
        schema: Arc::new(Schema::new(fields)),
        bindings,
        group_expr_aliases,
        aggregate_arg_aliases,
        filter_aliases,
    }))
}

fn append_synthetic_group_expr_projections(
    aggregate: &AggregateExec,
    input: &LoweredNativeInput,
    frontend_schema: &SchemaRef,
    expressions: &mut Vec<NativeExpr>,
    fields: &mut Vec<Field>,
) -> Result<Vec<String>, CapabilityReason> {
    let mut aliases = Vec::with_capacity(aggregate.group_expr().expr().len());
    for (index, (group_expr, alias)) in aggregate.group_expr().expr().iter().enumerate() {
        let executable_alias = unique_synthetic_aggregate_group_alias(index, alias, fields);
        expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(group_expr, frontend_schema, Some(&input.bindings))?,
            executable_alias.as_str(),
        ));
        fields.push(physical_expr_field(group_expr, frontend_schema, executable_alias.as_str())?);
        aliases.push(executable_alias);
    }
    Ok(aliases)
}

fn append_synthetic_filter_projections(
    aggregate: &AggregateExec,
    input: &LoweredNativeInput,
    frontend_schema: &SchemaRef,
    expressions: &mut Vec<NativeExpr>,
    fields: &mut Vec<Field>,
    filter_aliases: &mut [Option<String>],
) -> Result<(), CapabilityReason> {
    let mut filters_by_expression = HashMap::<String, String>::new();
    for (index, filter) in aggregate
        .filter_expr()
        .iter()
        .enumerate()
        .filter_map(|(index, filter)| filter.as_ref().map(|filter| (index, filter)))
    {
        append_synthetic_filter_projection(
            index,
            filter,
            input,
            frontend_schema,
            expressions,
            fields,
            filter_aliases,
            &mut filters_by_expression,
        )?;
    }
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "inputs are distinct domain values that share no owner")]
fn append_synthetic_filter_projection(
    index: usize,
    filter: &Arc<dyn PhysicalExpr>,
    input: &LoweredNativeInput,
    frontend_schema: &SchemaRef,
    expressions: &mut Vec<NativeExpr>,
    fields: &mut Vec<Field>,
    filter_aliases: &mut [Option<String>],
    filters_by_expression: &mut HashMap<String, String>,
) -> Result<(), CapabilityReason> {
    let key = filter.to_string();
    let alias = if let Some(alias) = filters_by_expression.get(&key) {
        alias.clone()
    } else {
        let alias = unique_synthetic_aggregate_filter_alias(filters_by_expression.len(), &input.schema, fields);
        expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(filter, frontend_schema, Some(&input.bindings))?,
            alias.as_str(),
        ));
        fields.push(physical_expr_field(filter, frontend_schema, alias.as_str())?);
        filters_by_expression.insert(key, alias.clone());
        alias
    };
    let slot = filter_aliases.get_mut(index).ok_or(CapabilityReason::UnsupportedExpression)?;
    *slot = Some(alias);
    Ok(())
}

fn unique_synthetic_aggregate_group_alias(index: usize, alias: &str, fields: &[Field]) -> String {
    if fields.iter().all(|field| field.name() != alias) {
        return alias.to_string();
    }

    let sanitized = sanitize_field_name(alias);
    let base = format!("__nexus_native_group_expr_{index}_{sanitized}");
    unique_name_with_suffix(base.as_str(), |candidate| fields.iter().all(|field| field.name() != candidate))
}

fn unique_synthetic_aggregate_arg_alias(index: usize, input_schema: &SchemaRef, fields: &[Field]) -> String {
    let base = format!("__nexus_native_aggregate_arg_{index}");
    unique_name_with_suffix(base.as_str(), |alias| synthetic_alias_is_available(alias, input_schema, fields))
}

fn unique_synthetic_aggregate_filter_alias(index: usize, input_schema: &SchemaRef, fields: &[Field]) -> String {
    let base = format!("__nexus_native_aggregate_filter_{index}");
    unique_name_with_suffix(base.as_str(), |alias| synthetic_alias_is_available(alias, input_schema, fields))
}

fn synthetic_alias_is_available(alias: &str, input_schema: &SchemaRef, fields: &[Field]) -> bool {
    input_schema.index_of(alias).is_err() && fields.iter().all(|field| field.name() != alias)
}

pub(in crate::native::lowering) struct PreparedAggregateInput {
    pub(in crate::native::lowering) input: LoweredNativeInput,
    pub(super) group_expr_aliases: Option<Vec<String>>,
    pub(super) aggregate_arg_aliases: Option<Vec<String>>,
    pub(super) filter_aliases: Option<Vec<Option<String>>>,
}

impl PreparedAggregateInput {
    pub(in crate::native::lowering) fn new(input: LoweredNativeInput) -> Self {
        Self { input, group_expr_aliases: None, aggregate_arg_aliases: None, filter_aliases: None }
    }
}

pub(in crate::native::lowering) fn prepare_aggregate_input_with_full_synthetic_projection(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    aggregate: &AggregateExec,
) -> Result<PreparedAggregateInput, CapabilityReason> {
    let synthetic_projection = synthetic_aggregate_input_projection(aggregate, &input)?;
    let projection_node = builder.add_projection(
        input.node,
        projection_outputs(synthetic_projection.expressions, &synthetic_projection.schema)?,
    );
    Ok(PreparedAggregateInput {
        input: LoweredNativeInput::with_bindings(
            projection_node,
            synthetic_projection.schema,
            synthetic_projection.bindings,
        ),
        group_expr_aliases: synthetic_projection.group_expr_aliases,
        aggregate_arg_aliases: Some(synthetic_projection.aggregate_arg_aliases),
        filter_aliases: Some(synthetic_projection.filter_aliases),
    })
}

pub(in crate::native::lowering) fn prepare_aggregate_input_with_optional_synthetic_args(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    aggregate: &AggregateExec,
) -> Result<PreparedAggregateInput, CapabilityReason> {
    if let Some(synthetic_projection) = synthetic_aggregate_input_projection_if_needed(aggregate, &input)? {
        let synthetic_projection_node = builder.add_projection(
            input.node,
            projection_outputs(synthetic_projection.expressions, &synthetic_projection.schema)?,
        );
        Ok(PreparedAggregateInput {
            input: LoweredNativeInput::with_bindings(
                synthetic_projection_node,
                synthetic_projection.schema,
                synthetic_projection.bindings,
            ),
            group_expr_aliases: synthetic_projection.group_expr_aliases,
            aggregate_arg_aliases: Some(synthetic_projection.aggregate_arg_aliases),
            filter_aliases: Some(synthetic_projection.filter_aliases),
        })
    } else {
        Ok(PreparedAggregateInput::new(input))
    }
}
