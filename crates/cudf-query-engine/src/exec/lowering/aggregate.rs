use super::*;

pub(in crate::exec) struct LoweredAggregate {
    pub(in crate::exec) key_columns: Vec<String>,
    pub(in crate::exec) output_key_names: Vec<String>,
    pub(in crate::exec) aggregations: Vec<CudfAggSpec>,
    pub(in crate::exec) output_names: Vec<String>,
}

pub(in crate::exec) fn lower_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let aggregations = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.aggregate.lower_expr")?;
            let function = lower_aggregate_function(expr);
            Ok(CudfAggSpec::new(input_column, function).with_alias(expr.alias.as_str()))
        })
        .collect::<Result<Vec<_>>>()?;

    aggregate.aggregations = aggregations;
    aggregate.output_names =
        aggregate.output_key_names.iter().chain(spec.aggr_exprs.iter().map(|expr| &expr.alias)).cloned().collect();
    Ok(aggregate)
}

pub(in crate::exec) fn lower_partial_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len().saturating_mul(2));
    let mut output_names = aggregate.output_key_names.clone();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.aggregate.lower_partial_expr")?;
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

pub(in crate::exec) fn lower_final_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let partial = lower_partial_aggregate_spec(spec)?;
    let mut aggregations = Vec::with_capacity(partial.aggregations.len());
    let mut output_names = partial.output_key_names.clone();
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                aggregations.push(CudfAggSpec::new(state.sum(), GroupByAgg::Sum).with_alias(state.sum()));
                aggregations.push(CudfAggSpec::new(state.count(), GroupByAgg::Sum).with_alias(state.count()));
                output_names.extend(state.output_names());
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
        key_columns: partial.output_key_names.clone(),
        output_key_names: partial.output_key_names,
        aggregations,
        output_names,
    })
}

pub(in crate::exec) fn lower_aggregate_function(expr: &AggregateExpr) -> GroupByAgg {
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
        AggregateFunction::StddevSamp => GroupByAgg::Std { ddof: 1 },
        AggregateFunction::StddevPop => GroupByAgg::Std { ddof: 0 },
        AggregateFunction::VarSamp => GroupByAgg::Variance { ddof: 1 },
        AggregateFunction::VarPop => GroupByAgg::Variance { ddof: 0 },
    }
}

pub(in crate::exec) fn lower_final_aggregate_function(function: AggregateFunction) -> GroupByAgg {
    match function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Sum,
        AggregateFunction::NUnique => GroupByAgg::Sum,
        AggregateFunction::Avg => unreachable!("avg final aggregate is lowered as explicit sum/count state"),
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarSamp
        | AggregateFunction::VarPop => {
            unreachable!("floating statistical final aggregate uses the whole-input statistical aggregate path")
        },
    }
}

pub(in crate::exec) fn lower_aggregate_spec_keys(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    if spec.group_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution currently requires at least one group key",
        ));
    }
    let mut key_columns = Vec::with_capacity(spec.group_exprs.len());
    let mut output_key_names = Vec::with_capacity(spec.group_exprs.len());
    for expr in &spec.group_exprs {
        let (input_column, output_name) = lower_named_column_expr(expr, "cudf_query_engine.aggregate.lower_key")?;
        key_columns.push(input_column);
        output_key_names.push(output_name);
    }
    if let Some(grouping_sets) = &spec.grouping_sets {
        key_columns.push(grouping_sets.grouping_id_alias.clone());
        output_key_names.push(grouping_sets.grouping_id_alias.clone());
    }

    Ok(LoweredAggregate { key_columns, output_key_names, aggregations: Vec::new(), output_names: Vec::new() })
}

pub(in crate::exec) fn aggregate_spec_has_avg(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::Avg)
}

pub(in crate::exec) fn aggregate_spec_has_floating_statistical_aggregate(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical())
}

pub(in crate::exec) fn aggregate_spec_has_distinct(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(aggregate_expr_is_exact_nunique)
}

pub(in crate::exec) fn aggregate_expr_is_exact_nunique(expr: &AggregateExpr) -> bool {
    expr.function == AggregateFunction::NUnique || expr.distinct && expr.function == AggregateFunction::Count
}

pub(in crate::exec) fn aggregate_spec_has_filters(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.filter.is_some())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::exec) struct AvgStateAliases {
    sum_alias: String,
    count_alias: String,
}

impl AvgStateAliases {
    /// Stable internal columns for logical AVG partial/final state.
    ///
    /// The public aggregate output keeps the logical AVG alias. These names
    /// exist only while carrying SUM and non-null COUNT state between partial
    /// and final aggregate stages.
    pub(in crate::exec) fn for_aggregate_index(index: usize) -> Self {
        Self { sum_alias: format!("__cudf_avg_{index}_sum"), count_alias: format!("__cudf_avg_{index}_count") }
    }

    pub(in crate::exec) fn sum(&self) -> &str {
        self.sum_alias.as_str()
    }

    pub(in crate::exec) fn count(&self) -> &str {
        self.count_alias.as_str()
    }

    pub(in crate::exec) fn sum_expr(&self) -> Expr {
        Expr::column(self.sum_alias.clone())
    }

    pub(in crate::exec) fn count_expr(&self) -> Expr {
        Expr::column(self.count_alias.clone())
    }

    pub(in crate::exec) fn output_names(&self) -> [String; 2] {
        [self.sum_alias.clone(), self.count_alias.clone()]
    }
}
