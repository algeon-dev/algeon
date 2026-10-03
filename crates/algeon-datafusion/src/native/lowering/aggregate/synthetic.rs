use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use arrow::datatypes::Field;
use datafusion::physical_expr::PhysicalExpr;

use super::{
    AggregateExec, CapabilityReason, Column, FieldBinding, LoweredNativeInput, LoweringFailure, NativeExpr,
    NativeQueryPlanBuilder, Schema, SchemaRef, alias_if_needed, lower_physical_expr_with_bindings, projection_outputs,
};
use crate::native::lowering::{
    expr::physical_expr_field,
    naming::{sanitize_field_name, unique_name_with_suffix},
};

/// An aggregate's input with every group key, argument and filter resolved
/// to a column. `None` reads the input column through the input bindings;
/// `Some` names a column the preparing projection computed.
pub(in crate::native::lowering) struct PreparedAggregateInput {
    pub(in crate::native::lowering) input: LoweredNativeInput,
    pub(super) group_aliases: Vec<Option<String>>,
    pub(super) arg_aliases: Vec<Option<String>>,
    pub(super) filter_aliases: Vec<Option<String>>,
}

/// Projects the columns an aggregate reads when it must compute a group key,
/// argument or filter, because the native aggregate reads columns: the
/// projection keeps the input columns the aggregate reads under their own
/// names and adds the computed ones. When every key, argument and filter is
/// an input column, the aggregate reads its input directly, so aggregates
/// over the same input read the same columns.
pub(in crate::native::lowering) fn prepare_aggregate_input(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    aggregate: &AggregateExec,
) -> Result<PreparedAggregateInput, LoweringFailure> {
    let frontend_schema = aggregate.input().schema();
    let args = aggregate
        .aggr_expr()
        .iter()
        .map(|aggregate_expr| match aggregate_expr.expressions().as_slice() {
            [arg] => Ok(Arc::clone(arg)),
            _ => Err(LoweringFailure::from(CapabilityReason::UnsupportedExpression)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_columns = aggregate
        .group_expr()
        .expr()
        .iter()
        .map(|(group_expr, _alias)| group_expr)
        .chain(args.iter())
        .map(|expr| input_column(&input, expr))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .map(|binding| binding.native_name.clone())
        .collect::<BTreeSet<_>>();

    let mut projection = AggregateInputProjection::default();
    let mut group_aliases = Vec::with_capacity(aggregate.group_expr().expr().len());
    for (index, (group_expr, alias)) in aggregate.group_expr().expr().iter().enumerate() {
        if let Some(binding) = input_column(&input, group_expr)? {
            projection.keep(&input, binding)?;
            group_aliases.push(None);
        } else {
            let alias = unique_synthetic_aggregate_group_alias(index, alias, &input_columns, &projection.fields);
            projection.compute(&input, group_expr, &frontend_schema, alias.as_str())?;
            group_aliases.push(Some(alias));
        }
    }
    let mut arg_aliases = Vec::with_capacity(args.len());
    let mut filter_aliases = vec![None; args.len()];
    let mut filters_by_expression = HashMap::<String, String>::new();
    for (index, arg) in args.iter().enumerate() {
        if let Some(binding) = input_column(&input, arg)? {
            projection.keep(&input, binding)?;
            arg_aliases.push(None);
        } else {
            // cuDF aggregate lowering consumes columns, so SQL aggregate
            // arguments such as SUM(m00 + m01) are first materialized as
            // synthetic projection columns.
            let alias = unique_synthetic_aggregate_arg_alias(index, &input.schema, &projection.fields);
            projection.compute(&input, arg, &frontend_schema, alias.as_str())?;
            arg_aliases.push(Some(alias));
        }
        if let Some(filter) = aggregate.filter_expr().get(index).and_then(Option::as_ref) {
            let key = filter.to_string();
            let alias = match filters_by_expression.get(&key) {
                Some(alias) => alias.clone(),
                None => {
                    let alias = unique_synthetic_aggregate_filter_alias(
                        filters_by_expression.len(),
                        &input.schema,
                        &projection.fields,
                    );
                    projection.compute(&input, filter, &frontend_schema, alias.as_str())?;
                    filters_by_expression.insert(key, alias.clone());
                    alias
                },
            };
            filter_aliases[index] = Some(alias);
        }
    }

    let input = if projection.computes_nothing() {
        input
    } else {
        let schema = Arc::new(Schema::new(projection.fields));
        let node = builder.add_projection(input.node, projection_outputs(projection.expressions, &schema)?)?;
        LoweredNativeInput::with_bindings(node, schema, projection.bindings)
    };
    Ok(PreparedAggregateInput { input, group_aliases, arg_aliases, filter_aliases })
}

fn input_column<'i>(
    input: &'i LoweredNativeInput,
    expr: &Arc<dyn PhysicalExpr>,
) -> Result<Option<&'i FieldBinding>, LoweringFailure> {
    Ok(expr.downcast_ref::<Column>().map(|column| input.binding_for_column(column)).transpose()?)
}

#[derive(Default)]
struct AggregateInputProjection {
    expressions: Vec<NativeExpr>,
    fields: Vec<Field>,
    bindings: Vec<FieldBinding>,
}

impl AggregateInputProjection {
    fn keep(&mut self, input: &LoweredNativeInput, binding: &FieldBinding) -> Result<(), LoweringFailure> {
        if self.bindings.iter().any(|kept| kept.native_name == binding.native_name) {
            return Ok(());
        }
        let field = input
            .schema
            .field_with_name(binding.native_name.as_str())
            .map_err(|_error| CapabilityReason::UnsupportedExpression)?;
        self.expressions.push(NativeExpr::column(binding.native_name.as_str()));
        self.fields.push(field.clone());
        self.bindings.push(binding.clone());
        Ok(())
    }

    fn compute(
        &mut self,
        input: &LoweredNativeInput,
        expr: &Arc<dyn PhysicalExpr>,
        frontend_schema: &SchemaRef,
        alias: &str,
    ) -> Result<(), LoweringFailure> {
        self.expressions.push(alias_if_needed(
            lower_physical_expr_with_bindings(expr, frontend_schema, Some(&input.bindings))?,
            alias,
        ));
        self.fields.push(physical_expr_field(expr, frontend_schema, alias)?);
        Ok(())
    }

    /// Every projected column is an input column, which the aggregate reads
    /// by name without a projection.
    fn computes_nothing(&self) -> bool {
        self.expressions.len() == self.bindings.len()
    }
}

/// A computed group key keeps its output name unless an input column the
/// aggregate reads, or an earlier projected column, already has it.
fn unique_synthetic_aggregate_group_alias(
    index: usize,
    alias: &str,
    input_columns: &BTreeSet<String>,
    fields: &[Field],
) -> String {
    let available =
        |candidate: &str| !input_columns.contains(candidate) && fields.iter().all(|field| field.name() != candidate);
    if available(alias) {
        return alias.to_string();
    }
    let sanitized = sanitize_field_name(alias);
    let base = format!("__algeon_native_group_expr_{index}_{sanitized}");
    unique_name_with_suffix(base.as_str(), available)
}

fn unique_synthetic_aggregate_arg_alias(index: usize, input_schema: &SchemaRef, fields: &[Field]) -> String {
    let base = format!("__algeon_native_aggregate_arg_{index}");
    unique_name_with_suffix(base.as_str(), |alias| synthetic_alias_is_available(alias, input_schema, fields))
}

fn unique_synthetic_aggregate_filter_alias(index: usize, input_schema: &SchemaRef, fields: &[Field]) -> String {
    let base = format!("__algeon_native_aggregate_filter_{index}");
    unique_name_with_suffix(base.as_str(), |alias| synthetic_alias_is_available(alias, input_schema, fields))
}

fn synthetic_alias_is_available(alias: &str, input_schema: &SchemaRef, fields: &[Field]) -> bool {
    input_schema.index_of(alias).is_err() && fields.iter().all(|field| field.name() != alias)
}
