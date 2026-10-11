//! The final aggregate that merges pre-aggregated states back into the
//! original outputs.
//!
//! A merged `COUNT` over a global aggregate is nullable: a join that matched
//! no rows feeds the merge no state, and `SUM` of nothing is null where SQL
//! `COUNT` returns zero. The projection above the merge reads that null as
//! zero and finishes `AVG` as sum over count.

use std::sync::Arc;

use super::{
    AggregateExpr, AggregateFunction, AggregateJoinPushdownRejectedReason, AggregateNode, AggregateSpec, Expr,
    PreAggregateMeasureState, ProjectionOutput, Schema, SchemaRef, append_group_projection_outputs, avg_final_expr,
    avg_merge_aggregates, merged_count_field, simple_projection_column, zero_filled_count_expr,
};

/// The merge aggregate and, when a measure needs finishing, the projection
/// that restores the original output expressions and names.
pub(in crate::optimizer::aggregate_join_pushdown) struct FinalAggregateRewrite {
    pub(in crate::optimizer::aggregate_join_pushdown) spec: AggregateSpec,
    pub(in crate::optimizer::aggregate_join_pushdown) output_schema: SchemaRef,
    pub(in crate::optimizer::aggregate_join_pushdown) projection_outputs: Option<Vec<ProjectionOutput>>,
}

pub(in crate::optimizer::aggregate_join_pushdown) fn final_aggregate_rewrite_after_preagg(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<FinalAggregateRewrite, AggregateJoinPushdownRejectedReason> {
    if aggregate.spec.aggr_exprs.len() != preagg_measure_states.len() {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let mut fields = Vec::with_capacity(
        aggregate.spec.group_exprs.len()
            + preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>(),
    );
    for group in &aggregate.spec.group_exprs {
        let column =
            simple_projection_column(group).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
        fields.push(
            aggregate_output_schema
                .field_with_name(column.name())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
                .as_ref()
                .clone(),
        );
    }
    if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
        fields.push(
            aggregate_output_schema
                .field_with_name(grouping_sets.grouping_id_alias.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
                .as_ref()
                .clone(),
        );
    }

    // A global merge over an empty join sums no partial counts.
    let global_merge = aggregate.spec.group_exprs.is_empty();
    let mut aggr_exprs = Vec::new();
    for measure in preagg_measure_states {
        match measure.original_function {
            AggregateFunction::Sum | AggregateFunction::Count => {
                let state = measure
                    .state_columns
                    .first()
                    .ok_or(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
                aggr_exprs.push(AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(state.alias.as_str()),
                    measure.original_alias.as_str(),
                ));
                fields.push(merged_count_field(
                    measure.final_output_field.clone(),
                    global_merge && measure.original_function == AggregateFunction::Count,
                ));
            },
            AggregateFunction::Avg => {
                let [sum, count] = measure.state_columns.as_slice() else {
                    return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
                };
                aggr_exprs.extend(avg_merge_aggregates(sum.alias.as_str(), count.alias.as_str()));
                fields.extend([sum.field.clone(), merged_count_field(count.field.clone(), global_merge)]);
            },
            _ => return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        }
    }

    let spec = AggregateSpec {
        group_exprs: aggregate.spec.group_exprs.clone(),
        aggr_exprs,
        grouping_sets: aggregate.spec.grouping_sets.clone(),
    };
    let output_schema = Arc::new(Schema::new(fields));
    let projection_outputs = preagg_measure_states
        .iter()
        .any(|measure| {
            measure.original_function == AggregateFunction::Avg
                || global_merge && measure.original_function == AggregateFunction::Count
        })
        .then(|| final_projection_outputs(aggregate, aggregate_output_schema, preagg_measure_states))
        .transpose()?;
    Ok(FinalAggregateRewrite { spec, output_schema, projection_outputs })
}

pub(in crate::optimizer::aggregate_join_pushdown) fn final_projection_outputs(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<Vec<ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
    let global_merge = aggregate.spec.group_exprs.is_empty();
    let grouping_id_output_count = usize::from(aggregate.spec.grouping_sets.is_some());
    let mut outputs = Vec::with_capacity(
        aggregate.spec.group_exprs.len() + grouping_id_output_count + aggregate.spec.aggr_exprs.len(),
    );
    append_group_projection_outputs(&mut outputs, aggregate, aggregate_output_schema)?;
    if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
        let field = aggregate_output_schema
            .field_with_name(grouping_sets.grouping_id_alias.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
            .as_ref()
            .clone();
        outputs.push(ProjectionOutput::new(Expr::column(grouping_sets.grouping_id_alias.as_str()), field));
    }

    for measure in preagg_measure_states {
        let field = measure.final_output_field.clone();
        let expr = match measure.original_function {
            AggregateFunction::Sum => Expr::column(measure.original_alias.as_str()),
            AggregateFunction::Count if global_merge => {
                zero_filled_count_expr(Expr::column(measure.original_alias.as_str()), field.data_type())
                    .map_err(|_| AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?
                    .alias(measure.original_alias.as_str())
            },
            AggregateFunction::Count => Expr::column(measure.original_alias.as_str()),
            AggregateFunction::Avg => final_avg_projection_expr(measure)?,
            _ => return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        };
        outputs.push(ProjectionOutput::new(expr, field));
    }
    Ok(outputs)
}

pub(in crate::optimizer::aggregate_join_pushdown) fn final_avg_projection_expr(
    measure: &PreAggregateMeasureState,
) -> Result<Expr, AggregateJoinPushdownRejectedReason> {
    let [sum, count] = measure.state_columns.as_slice() else {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    };
    let avg = avg_final_expr(
        Expr::column(sum.alias.as_str()),
        Expr::column(count.alias.as_str()),
        measure.final_output_field.data_type(),
    )
    .map_err(|_| AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
    Ok(avg.alias(measure.original_alias.as_str()))
}
