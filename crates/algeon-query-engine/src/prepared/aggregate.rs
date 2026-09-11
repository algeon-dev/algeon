use super::{
    AggregateExpr, AggregateFunction, AggregateSpec, AvgStateAliases, CudfAggSpec, GroupByAgg, QueryError,
    QueryErrorCode, Result, aggregate_expr_is_exact_nunique, lower_column_expr, lower_named_column_expr,
};

#[doc(hidden)]
pub struct LoweredAggregate {
    #[doc(hidden)]
    pub key_columns: Vec<String>,
    #[doc(hidden)]
    pub output_key_names: Vec<String>,
    #[doc(hidden)]
    pub aggregations: Vec<CudfAggSpec>,
    #[doc(hidden)]
    pub output_names: Vec<String>,
}

#[doc(hidden)]
pub fn lower_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let aggregations = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            let input_column = lower_column_expr(&expr.expr, "query_engine.aggregate.lower_expr")?;
            let function = lower_aggregate_function(expr);
            Ok(CudfAggSpec::new(input_column, function).with_alias(expr.alias.as_str()))
        })
        .collect::<Result<Vec<_>>>()?;

    aggregate.aggregations = aggregations;
    aggregate.output_names =
        aggregate.output_key_names.iter().chain(spec.aggr_exprs.iter().map(|expr| &expr.alias)).cloned().collect();
    Ok(aggregate)
}

#[doc(hidden)]
pub fn lower_single_stage_grouped_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(2));
    let mut output_names = aggregate.output_key_names.clone();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column = lower_column_expr(&expr.expr, "query_engine.aggregate.lower_single_expr")?;
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(input_column.as_str(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(
                    CudfAggSpec::new(input_column.as_str(), GroupByAgg::Count { include_nulls: false })
                        .with_alias(state.count()),
                );
                output_names.extend(state.output_names());
            },
            _ => {
                aggregations.push(
                    CudfAggSpec::new(input_column, lower_aggregate_function(expr)).with_alias(expr.alias.as_str()),
                );
                output_names.push(expr.alias.clone());
            },
        }
    }

    aggregate.aggregations = aggregations;
    aggregate.output_names = output_names;
    Ok(aggregate)
}

#[doc(hidden)]
pub fn lower_partial_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(2));
    let mut output_names = aggregate.output_key_names.clone();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column = lower_column_expr(&expr.expr, "query_engine.aggregate.lower_partial_expr")?;
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(input_column.as_str(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(
                    CudfAggSpec::new(input_column.as_str(), GroupByAgg::Count { include_nulls: false })
                        .with_alias(state.count()),
                );
                output_names.extend(state.output_names());
            },
            AggregateFunction::Median => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedOperator,
                    "query_engine.aggregate.lower_partial",
                    "median aggregate does not have native partial/final state",
                )
                .with_fact("alias", expr.alias.as_str()));
            },
            _ => {
                aggregations.push(
                    CudfAggSpec::new(input_column, lower_aggregate_function(expr)).with_alias(expr.alias.as_str()),
                );
                output_names.push(expr.alias.clone());
            },
        }
    }

    aggregate.aggregations = aggregations;
    aggregate.output_names = output_names;
    Ok(aggregate)
}

#[doc(hidden)]
pub fn lower_final_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let keys = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(2));
    let mut output_names = keys.output_key_names.clone();
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(state.sum(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(CudfAggSpec::new(state.count(), GroupByAgg::Sum).with_alias(state.count()));
                output_names.extend(state.output_names());
            },
            AggregateFunction::Median => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedOperator,
                    "query_engine.aggregate.lower_final",
                    "median aggregate does not have native partial/final state",
                )
                .with_fact("alias", expr.alias.as_str()));
            },
            _ => {
                aggregations.push(
                    CudfAggSpec::new(expr.alias.as_str(), lower_final_aggregate_function(expr.function))
                        .with_alias(expr.alias.as_str()),
                );
                output_names.push(expr.alias.clone());
            },
        }
    }

    Ok(LoweredAggregate {
        key_columns: keys.output_key_names.clone(),
        output_key_names: keys.output_key_names,
        aggregations,
        output_names,
    })
}

#[doc(hidden)]
pub fn lower_aggregate_function(expr: &AggregateExpr) -> GroupByAgg {
    if aggregate_expr_is_exact_nunique(expr) {
        return GroupByAgg::NUnique;
    }
    match expr.function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Count { include_nulls: false },
        AggregateFunction::NUnique => GroupByAgg::NUnique,
        AggregateFunction::Avg => GroupByAgg::Mean,
        AggregateFunction::Median => GroupByAgg::Median,
        AggregateFunction::StddevSamp => GroupByAgg::Std { ddof: 1 },
        AggregateFunction::StddevPop => GroupByAgg::Std { ddof: 0 },
        AggregateFunction::VarSamp => GroupByAgg::Variance { ddof: 1 },
        AggregateFunction::VarPop => GroupByAgg::Variance { ddof: 0 },
    }
}

#[doc(hidden)]
pub fn lower_final_aggregate_function(function: AggregateFunction) -> GroupByAgg {
    match function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Sum,
        AggregateFunction::NUnique => GroupByAgg::Sum,
        AggregateFunction::Avg => unreachable!("avg final aggregate is lowered as explicit sum/count state"),
        AggregateFunction::Median => unreachable!("median aggregate does not have native partial/final state"),
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarSamp
        | AggregateFunction::VarPop => {
            unreachable!("floating statistical final aggregate uses the whole-input statistical aggregate path")
        },
    }
}

#[doc(hidden)]
pub fn lower_aggregate_spec_keys(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    if spec.group_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.aggregate.lower",
            "native execution currently requires at least one group key",
        ));
    }
    let mut key_columns = Vec::with_capacity(spec.group_exprs.len());
    let mut output_key_names = Vec::with_capacity(spec.group_exprs.len());
    for expr in &spec.group_exprs {
        let (input_column, output_name) = lower_named_column_expr(expr, "query_engine.aggregate.lower_key")?;
        key_columns.push(input_column);
        output_key_names.push(output_name);
    }
    if let Some(grouping_sets) = &spec.grouping_sets {
        key_columns.push(grouping_sets.grouping_id_alias.clone());
        output_key_names.push(grouping_sets.grouping_id_alias.clone());
    }

    Ok(LoweredAggregate { key_columns, output_key_names, aggregations: Vec::new(), output_names: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    fn median_spec() -> AggregateSpec {
        AggregateSpec::new(
            vec![Expr::column("key")],
            vec![AggregateExpr::new(AggregateFunction::Median, Expr::column("value"), "median_value")],
        )
    }

    #[test]
    fn median_rejects_partial_and_final_state_lowering() {
        let partial = match lower_partial_aggregate_spec(&median_spec()) {
            Ok(_) => panic!("median partial state must reject"),
            Err(error) => error,
        };
        assert_eq!(partial.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(partial.operation(), "query_engine.aggregate.lower_partial");
        assert_eq!(partial.fact_value("alias"), Some("median_value"));

        let final_merge = match lower_final_aggregate_spec(&median_spec()) {
            Ok(_) => panic!("median final state must reject"),
            Err(error) => error,
        };
        assert_eq!(final_merge.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(final_merge.operation(), "query_engine.aggregate.lower_final");
        assert_eq!(final_merge.fact_value("alias"), Some("median_value"));
    }

    #[test]
    fn median_lowers_for_single_stage_grouped_aggregate() {
        let lowered =
            lower_single_stage_grouped_aggregate_spec(&median_spec()).expect("median single-stage aggregate lowers");
        assert_eq!(lowered.output_names, vec!["key", "median_value"]);
        assert_eq!(lowered.aggregations.len(), 1);
    }

    #[test]
    fn mixed_avg_and_median_lower_for_single_stage_grouped_aggregate() {
        let spec = AggregateSpec::new(
            vec![Expr::column("key")],
            vec![
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("value"), "avg_value"),
                AggregateExpr::new(AggregateFunction::Median, Expr::column("value"), "median_value"),
            ],
        );

        let lowered = lower_single_stage_grouped_aggregate_spec(&spec)
            .expect("mixed avg and median single-stage aggregate lowers");

        assert_eq!(
            lowered.output_names,
            vec!["key", "__algeon_native_avg_0_sum", "__algeon_native_avg_0_count", "median_value"]
        );
        assert_eq!(lowered.aggregations.len(), 3);
    }
}
