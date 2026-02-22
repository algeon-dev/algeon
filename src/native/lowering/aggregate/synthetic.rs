use arrow::datatypes::Field;

use super::*;
use crate::native::lowering::expr::physical_expr_field;

pub(super) struct SyntheticAggregateInputProjection {
    pub(super) expressions: Vec<NativeExpr>,
    pub(super) schema: SchemaRef,
    pub(super) bindings: Vec<FieldBinding>,
    pub(super) group_expr_aliases: Option<Vec<String>>,
    pub(super) aggregate_arg_aliases: Vec<String>,
}

pub(super) fn synthetic_aggregate_input_projection(
    aggregate: &AggregateExec,
    input: &LoweredNativeInput,
) -> Result<SyntheticAggregateInputProjection, FallbackReason> {
    let frontend_schema = aggregate.input().schema();
    let mut expressions = Vec::new();
    let mut fields = Vec::new();
    let mut group_expr_aliases = Vec::with_capacity(aggregate.group_expr().expr().len());

    for (index, (group_expr, alias)) in aggregate.group_expr().expr().iter().enumerate() {
        let executable_alias = unique_synthetic_aggregate_group_alias(index, alias, &fields);
        expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(group_expr, &frontend_schema, Some(&input.bindings))?,
            executable_alias.as_str(),
        ));
        fields.push(physical_expr_field(group_expr, &frontend_schema, executable_alias.as_str())?);
        group_expr_aliases.push(executable_alias);
    }

    let mut aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
    for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
        let expressions_for_aggregate = aggregate_expr.expressions();
        let [arg] = expressions_for_aggregate.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
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
    }

    Ok(SyntheticAggregateInputProjection {
        expressions,
        schema: Arc::new(Schema::new(fields)),
        bindings: Vec::new(),
        group_expr_aliases: Some(group_expr_aliases),
        aggregate_arg_aliases,
    })
}

pub(super) fn synthetic_aggregate_input_projection_if_needed(
    aggregate: &AggregateExec,
    input: &LoweredNativeInput,
) -> Result<Option<SyntheticAggregateInputProjection>, FallbackReason> {
    let frontend_schema = aggregate.input().schema();
    let needs_group_projection = aggregate
        .group_expr()
        .expr()
        .iter()
        .any(|(group_expr, _alias)| group_expr.as_any().downcast_ref::<Column>().is_none());
    let needs_aggregate_arg_projection = aggregate.aggr_expr().iter().any(|aggregate_expr| {
        let expressions = aggregate_expr.expressions();
        !matches!(expressions.as_slice(), [arg] if arg.as_any().downcast_ref::<Column>().is_some())
    });
    let needs_projection = needs_group_projection || needs_aggregate_arg_projection;
    if !needs_projection {
        return Ok(None);
    }

    let mut expressions =
        input.schema.fields().iter().map(|field| NativeExpr::column(field.name())).collect::<Vec<_>>();
    let mut fields = input.schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();

    let group_expr_aliases = if needs_group_projection {
        let mut aliases = Vec::with_capacity(aggregate.group_expr().expr().len());
        for (index, (group_expr, alias)) in aggregate.group_expr().expr().iter().enumerate() {
            let executable_alias = unique_synthetic_aggregate_group_alias(index, alias, &fields);
            expressions.push(alias_if_needed(
                lower_physical_expr_with_bindings(group_expr, &frontend_schema, Some(&input.bindings))?,
                executable_alias.as_str(),
            ));
            fields.push(physical_expr_field(group_expr, &frontend_schema, executable_alias.as_str())?);
            aliases.push(executable_alias);
        }
        Some(aliases)
    } else {
        None
    };

    let mut aggregate_arg_aliases = Vec::new();
    if needs_aggregate_arg_projection {
        aggregate_arg_aliases = Vec::with_capacity(aggregate.aggr_expr().len());
        for (index, aggregate_expr) in aggregate.aggr_expr().iter().enumerate() {
            let expressions_for_aggregate = aggregate_expr.expressions();
            let [arg] = expressions_for_aggregate.as_slice() else {
                return Err(FallbackReason::UnsupportedExpression);
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
    }))
}

fn unique_synthetic_aggregate_group_alias(index: usize, alias: &str, fields: &[Field]) -> String {
    if fields.iter().all(|field| field.name() != alias) {
        return alias.to_string();
    }

    let sanitized = sanitize_synthetic_alias_component(alias);
    let base = format!("__cudf_group_expr_{index}_{sanitized}");
    if fields.iter().all(|field| field.name() != base.as_str()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if fields.iter().all(|field| field.name() != candidate.as_str()) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
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

fn sanitize_synthetic_alias_component(name: &str) -> String {
    let sanitized =
        name.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' }).collect::<String>();
    if sanitized.is_empty() { "field".to_string() } else { sanitized }
}

pub(in crate::native::lowering) struct PreparedAggregateInput {
    pub(in crate::native::lowering) input: LoweredNativeInput,
    pub(super) group_expr_aliases: Option<Vec<String>>,
    pub(super) aggregate_arg_aliases: Option<Vec<String>>,
}

impl PreparedAggregateInput {
    pub(in crate::native::lowering) fn new(input: LoweredNativeInput) -> Self {
        Self { input, group_expr_aliases: None, aggregate_arg_aliases: None }
    }
}

pub(in crate::native::lowering) fn prepare_aggregate_input_with_full_synthetic_projection(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    aggregate: &AggregateExec,
) -> Result<PreparedAggregateInput, FallbackReason> {
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
    })
}

pub(in crate::native::lowering) fn prepare_aggregate_input_with_optional_synthetic_args(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    aggregate: &AggregateExec,
) -> Result<PreparedAggregateInput, FallbackReason> {
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
        })
    } else {
        Ok(PreparedAggregateInput::new(input))
    }
}
