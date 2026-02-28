use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;

pub(in crate::exec) fn partial_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        spec.aggr_exprs
            .iter()
            .enumerate()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Avg => {
                    let state = AvgStateAliases::for_aggregate_index(index);
                    vec![
                        AggregateExpr::new(AggregateFunction::Sum, expr.expr.clone(), state.sum().to_string()),
                        AggregateExpr::new(AggregateFunction::Count, expr.expr.clone(), state.count().to_string()),
                    ]
                },
                AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => vec![
                    expr.clone(),
                    AggregateExpr::new(
                        AggregateFunction::Count,
                        expr.expr.clone(),
                        global_valid_count_state_alias(expr.alias.as_str()),
                    ),
                ],
                AggregateFunction::Count | AggregateFunction::NUnique => {
                    vec![expr.clone()]
                },
                AggregateFunction::StddevSamp
                | AggregateFunction::StddevPop
                | AggregateFunction::VarSamp
                | AggregateFunction::VarPop => vec![expr.clone()],
            })
            .collect(),
    )
}

pub(in crate::exec) fn partial_global_aggregate_output_schema(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
) -> Result<Arc<Schema>> {
    let mut fields = Vec::with_capacity(spec.aggr_exprs.len());
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                let input_type = global_aggregate_input_data_type(&expr.expr, input_schema)?;
                let output_field = global_aggregate_output_field(expr.alias.as_str(), aggregate_output_schema)?;
                let sum_state_type = avg_sum_state_data_type(&input_type, output_field.data_type())?;
                fields.push(Field::new(state.sum(), sum_state_type, true));
                fields.push(Field::new(state.count(), ArrowDataType::Int64, false));
            },
            AggregateFunction::Count | AggregateFunction::NUnique => {
                fields.push(Field::new(expr.alias.as_str(), ArrowDataType::Int64, false));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                fields.push(global_aggregate_output_field(expr.alias.as_str(), aggregate_output_schema)?);
                fields.push(Field::new(
                    global_valid_count_state_alias(expr.alias.as_str()),
                    ArrowDataType::Int64,
                    false,
                ));
            },
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop => {
                fields.push(global_aggregate_output_field(expr.alias.as_str(), aggregate_output_schema)?);
            },
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub(in crate::exec) fn final_global_aggregate_state_output_schema(
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
) -> Result<Arc<Schema>> {
    partial_global_aggregate_output_schema(spec, input_schema, aggregate_output_schema)
}

pub(in crate::exec) fn final_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        spec.aggr_exprs
            .iter()
            .enumerate()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Count => vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(expr.alias.clone()),
                    expr.alias.clone(),
                )],
                AggregateFunction::NUnique => vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(expr.alias.clone()),
                    expr.alias.clone(),
                )],
                AggregateFunction::Avg => {
                    let state = AvgStateAliases::for_aggregate_index(index);
                    vec![
                        AggregateExpr::new(AggregateFunction::Sum, state.sum_expr(), state.sum().to_string()),
                        AggregateExpr::new(AggregateFunction::Sum, state.count_expr(), state.count().to_string()),
                    ]
                },
                AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                    let valid_count_alias = global_valid_count_state_alias(expr.alias.as_str());
                    vec![
                        AggregateExpr::new(expr.function, Expr::column(expr.alias.clone()), expr.alias.clone()),
                        AggregateExpr::new(
                            AggregateFunction::Sum,
                            Expr::column(valid_count_alias.clone()),
                            valid_count_alias,
                        ),
                    ]
                },
                AggregateFunction::StddevSamp
                | AggregateFunction::StddevPop
                | AggregateFunction::VarSamp
                | AggregateFunction::VarPop => vec![expr.clone()],
            })
            .collect(),
    )
}

fn global_aggregate_input_data_type(expr: &Expr, input_schema: &ArrowSchemaRef) -> Result<ArrowDataType> {
    let input_column = lower_column_expr(expr, "nexus_query_engine.global_aggregate.state_schema")?;
    let field = input_schema.field_with_name(input_column.as_str()).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.global_aggregate.state_schema",
            "global aggregate state references a column missing from its input schema",
        )
        .with_fact("column", input_column.as_str())
        .with_fact("arrow_error", error.to_string())
    })?;
    Ok(field.data_type().clone())
}

fn global_aggregate_output_field(alias: &str, output_schema: &ArrowSchemaRef) -> Result<Field> {
    output_schema.field_with_name(alias).cloned().map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.global_aggregate.state_schema",
            "global aggregate state references a column missing from its output schema",
        )
        .with_fact("alias", alias)
        .with_fact("arrow_error", error.to_string())
    })
}

pub(in crate::exec) fn global_valid_count_state_alias(alias: &str) -> String {
    format!("{GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX}{alias}")
}

const GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX: &str = "__cudf_global_valid_count_";

pub(in crate::exec) fn validate_global_valid_count_state_alias_collisions(
    spec: &AggregateSpec,
    operation: &'static str,
) -> Result<()> {
    let reserved_aliases = spec
        .aggr_exprs
        .iter()
        .filter(|expr| {
            matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
        })
        .map(|expr| global_valid_count_state_alias(expr.alias.as_str()))
        .collect::<Vec<_>>();
    if reserved_aliases.is_empty() {
        return Ok(());
    }

    let visible_aliases = spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, operation).map(|(_, alias)| alias))
        .chain(spec.aggr_exprs.iter().map(|expr| Ok(expr.alias.clone())))
        .collect::<Result<Vec<_>>>()?;
    for alias in visible_aliases {
        if reserved_aliases.iter().any(|reserved| reserved == &alias) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "visible aggregate output alias conflicts with reserved internal global aggregate state column",
            )
            .with_fact("alias", alias)
            .with_fact("reserved_aliases", reserved_aliases.join(",")));
        }
    }

    Ok(())
}
