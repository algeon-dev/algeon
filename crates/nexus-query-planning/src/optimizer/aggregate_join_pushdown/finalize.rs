use std::sync::Arc;

use super::{
    AggregateExpr, AggregateFunction, AggregateJoinPushdownRejectedReason, AggregateNode, AggregateSpec, BinaryOp,
    CaseWhen, DataType, Expr, LiteralValue, PreAggregateMeasureState, ProjectionOutput, Schema, SchemaRef,
    append_group_projection_outputs, simple_projection_column,
};

pub(in crate::optimizer::aggregate_join_pushdown) struct FinalAggregateRewrite {
    pub(in crate::optimizer::aggregate_join_pushdown) spec: AggregateSpec,
    pub(in crate::optimizer::aggregate_join_pushdown) output_schema: SchemaRef,
    pub(in crate::optimizer::aggregate_join_pushdown) projection_outputs: Option<Vec<crate::plan::ProjectionOutput>>,
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
                fields.push(measure.final_output_field.clone());
            },
            AggregateFunction::Avg => {
                if measure.state_columns.len() != 2 {
                    return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
                }
                for state in &measure.state_columns {
                    aggr_exprs.push(AggregateExpr::new(
                        AggregateFunction::Sum,
                        Expr::column(state.alias.as_str()),
                        state.alias.as_str(),
                    ));
                    fields.push(state.field.clone());
                }
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
        .any(|measure| measure.original_function == AggregateFunction::Avg)
        .then(|| final_avg_projection_outputs(aggregate, aggregate_output_schema, preagg_measure_states))
        .transpose()?;
    Ok(FinalAggregateRewrite { spec, output_schema, projection_outputs })
}

pub(in crate::optimizer::aggregate_join_pushdown) fn final_avg_projection_outputs(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<Vec<ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
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
            AggregateFunction::Sum | AggregateFunction::Count => Expr::column(measure.original_alias.as_str()),
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
    if measure.state_columns.len() != 2 {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let sum_state = &measure.state_columns[0];
    let count_state = &measure.state_columns[1];
    let division_type =
        avg_pushdown_division_data_type(sum_state.field.data_type(), measure.final_output_field.data_type())?;
    let sum = Expr::cast(Expr::column(sum_state.alias.as_str()), division_type.clone());
    let count =
        Expr::cast(Expr::column(count_state.alias.as_str()), avg_pushdown_count_projection_data_type(&division_type));
    let count_is_zero =
        Expr::binary(Expr::column(count_state.alias.as_str()), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(0)));
    let avg_divide = if matches!(division_type, DataType::Decimal128(_, _)) {
        Expr::binary_with_output_type(sum, BinaryOp::Divide, count, division_type)
    } else {
        Expr::binary(sum, BinaryOp::Divide, count)
    };
    let avg = Expr::cast(avg_divide, measure.final_output_field.data_type().clone());
    Ok(Expr::case(
        vec![CaseWhen::new(
            count_is_zero,
            Expr::literal(LiteralValue::Null(measure.final_output_field.data_type().clone())),
        )],
        avg,
    )
    .alias(measure.original_alias.as_str()))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn avg_pushdown_sum_state_data_type(
    input_type: &DataType,
    avg_output_type: &DataType,
) -> Result<DataType, AggregateJoinPushdownRejectedReason> {
    match avg_output_type {
        DataType::Float32 | DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(_, output_scale) => match input_type {
            DataType::Decimal128(_, input_scale) => {
                Ok(DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *input_scale))
            },
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Ok(DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *output_scale)),
            _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        },
        _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) fn avg_pushdown_division_data_type(
    state_sum_type: &DataType,
    avg_output_type: &DataType,
) -> Result<DataType, AggregateJoinPushdownRejectedReason> {
    match avg_output_type {
        DataType::Float32 | DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(_, output_scale) => match state_sum_type {
            DataType::Decimal128(_, _)
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Ok(DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *output_scale)),
            _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        },
        _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) fn avg_pushdown_count_projection_data_type(
    avg_output_type: &DataType,
) -> DataType {
    match avg_output_type {
        DataType::Decimal128(_, _) => DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, 0),
        _ => avg_output_type.clone(),
    }
}
