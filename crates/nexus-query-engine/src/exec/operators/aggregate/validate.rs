use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;

pub(in crate::exec) fn validate_aggregate_capability(
    node: &AggregateNode,
    output_schema: &ArrowSchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        validate_global_valid_count_state_alias_collisions(
            &node.spec,
            "nexus_query_engine.global_aggregate.validate_valid_count_state_alias",
        )?;
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_floating_statistical_aggregate(&node.spec) {
        if aggregate_spec_has_distinct(&node.spec) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "nexus_query_engine.aggregate.validate_statistical_exact",
                "DataFusion-exact statistical aggregate path does not support distinct aggregate expressions",
            ));
        }
        let expected_output_names = node
            .spec
            .group_exprs
            .iter()
            .map(host_group_expr_output_name)
            .chain(node.spec.aggr_exprs.iter().map(|expr| expr.alias.clone()))
            .collect::<Vec<_>>();
        let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
        if expected_output_names != schema_output_names {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.aggregate.validate_statistical_exact",
                "DataFusion-exact statistical aggregate output names must match output schema fields",
            )
            .with_fact("aggregate_output_names", expected_output_names.join(","))
            .with_fact("schema_output_names", schema_output_names.join(",")));
        }
        return Ok(());
    }
    let validation_spec = aggregate_validation_spec(&node.spec);
    let aggregate = lower_aggregate_spec(&validation_spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "nexus_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    Ok(())
}

fn aggregate_validation_spec(spec: &AggregateSpec) -> AggregateSpec {
    let mut validation = spec.clone();
    for (index, expr) in validation.aggr_exprs.iter_mut().enumerate() {
        if expr.filter.is_some()
            && expr.function == AggregateFunction::Count
            && !expr.distinct
            && global_count_star_expr(&expr.expr)
        {
            expr.expr = Expr::column(format!("__nexus_native_filtered_count_star_validation_{index}"));
        }
    }
    validation
}

pub(in crate::exec) fn validate_final_aggregate_capability(
    node: &AggregateNode,
    output_schema: &ArrowSchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        validate_global_valid_count_state_alias_collisions(
            &node.spec,
            "nexus_query_engine.global_aggregate.validate_valid_count_state_alias",
        )?;
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_floating_statistical_aggregate(&node.spec) {
        if aggregate_spec_has_distinct(&node.spec) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "nexus_query_engine.final_merge.validate_statistical_exact",
                "DataFusion-exact statistical aggregate path does not support distinct aggregate expressions",
            ));
        }
        return validate_aggregate_capability(node, output_schema);
    }
    let aggregate = lower_final_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_state_alias_collisions(&node.spec, "nexus_query_engine.final_merge.validate_avg_state_alias")?;
        validate_avg_output_fields(&node.spec, output_schema)?;
        return Ok(());
    }
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "nexus_query_engine.final_merge.validate",
        "final aggregate output names must match output schema fields",
    )
}

pub(in crate::exec) fn validate_global_aggregate_capability(
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
) -> Result<()> {
    if aggregate_spec_has_avg(spec) {
        validate_avg_state_alias_collisions(spec, "nexus_query_engine.global_aggregate.validate_avg_state_alias")?;
    }
    let expected_output_names = spec.aggr_exprs.iter().map(|expr| expr.alias.clone()).collect::<Vec<_>>();
    let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if expected_output_names != schema_output_names {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.global_aggregate.validate",
            "global aggregate output names must match output schema fields",
        )
        .with_fact("aggregate_output_names", expected_output_names.join(","))
        .with_fact("schema_output_names", schema_output_names.join(",")));
    }
    for expr in &spec.aggr_exprs {
        if expr.function == AggregateFunction::Avg {
            validate_avg_output_fields(spec, output_schema)?;
        }
        if expr.function.is_floating_statistical() {
            if expr.distinct {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedOperator,
                    "nexus_query_engine.global_aggregate.validate_statistical_exact",
                    "DataFusion-exact statistical aggregate path does not support distinct aggregate expressions",
                )
                .with_fact("alias", expr.alias.as_str()));
            }
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.global_aggregate.validate_statistical_exact",
                    "floating statistical aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            if !matches!(field.data_type(), ArrowDataType::Float64) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.global_aggregate.validate_statistical_exact",
                    "floating statistical aggregate output must be Float64",
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("function", expr.function.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type())));
            }
        }
        validate_global_aggregate_input_expr(expr)?;
    }
    Ok(())
}

fn validate_global_aggregate_input_expr(expr: &AggregateExpr) -> Result<()> {
    match expr.function {
        AggregateFunction::Count if !expr.distinct && global_count_star_expr(&expr.expr) => Ok(()),
        _ => lower_column_expr(&expr.expr, "nexus_query_engine.global_aggregate.validate_expr").map(|_| ()),
    }
}

pub(in crate::exec) fn validate_aggregate_output_names(
    aggregate: &LoweredAggregate,
    output_schema: &ArrowSchemaRef,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if aggregate.output_names != expected_output_names {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, operation, message)
            .with_fact("aggregate_output_names", aggregate.output_names.join(","))
            .with_fact("schema_output_names", expected_output_names.join(",")));
    }

    Ok(())
}
