use super::*;

#[derive(Clone)]
pub(in crate::optimizer::aggregate_join_pushdown) struct PreAggregateColumn {
    pub(in crate::optimizer::aggregate_join_pushdown) source_name: String,
    pub(in crate::optimizer::aggregate_join_pushdown) output_name: String,
}

#[derive(Clone)]
pub(in crate::optimizer::aggregate_join_pushdown) struct PreAggregateMeasureState {
    pub(in crate::optimizer::aggregate_join_pushdown) original_function: AggregateFunction,
    pub(in crate::optimizer::aggregate_join_pushdown) original_alias: String,
    pub(in crate::optimizer::aggregate_join_pushdown) final_output_field: Field,
    pub(in crate::optimizer::aggregate_join_pushdown) state_columns: Vec<PreAggregateStateColumn>,
}

#[derive(Clone)]
pub(in crate::optimizer::aggregate_join_pushdown) struct PreAggregateStateColumn {
    pub(in crate::optimizer::aggregate_join_pushdown) alias: String,
    pub(in crate::optimizer::aggregate_join_pushdown) preaggregate_function: AggregateFunction,
    pub(in crate::optimizer::aggregate_join_pushdown) field: Field,
}

pub(in crate::optimizer::aggregate_join_pushdown) fn push_unique_preagg_group_column(
    columns: &mut Vec<PreAggregateColumn>,
    source_name: &str,
    output_name: &str,
    input_schema: &SchemaRef,
) -> Result<(), AggregateJoinPushdownRejectedReason> {
    if let Some(existing) = columns.iter().find(|column| column.output_name == output_name) {
        if existing.source_name == source_name {
            return Ok(());
        }
        return Err(AggregateJoinPushdownRejectedReason::ConflictingPreAggregateOutputName);
    }
    input_schema.field_with_name(source_name).map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
    columns.push(PreAggregateColumn { source_name: source_name.to_string(), output_name: output_name.to_string() });
    Ok(())
}

pub(in crate::optimizer::aggregate_join_pushdown) fn preaggregate_schema(
    input_schema: &SchemaRef,
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    group_columns: &[PreAggregateColumn],
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<SchemaRef, AggregateJoinPushdownRejectedReason> {
    let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
    let mut fields = Vec::with_capacity(group_columns.len() + state_count);
    for column in group_columns {
        let input_field = input_schema
            .field_with_name(column.source_name.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        let mut field = input_field.as_ref().clone();
        if field.name() != &column.output_name {
            field = Field::new(column.output_name.as_str(), field.data_type().clone(), field.is_nullable());
        }
        fields.push(field);
    }
    if aggregate.spec.aggr_exprs.len() != preagg_measure_states.len()
        || preagg_measure_states
            .iter()
            .any(|measure| aggregate_output_schema.field_with_name(measure.original_alias.as_str()).is_err())
    {
        return Err(AggregateJoinPushdownRejectedReason::MissingInputField);
    }
    for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
        fields.push(state.field.clone());
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn preaggregate_spec(
    aggregate: &AggregateNode,
    group_columns: &[PreAggregateColumn],
    preagg_measure_states: &[PreAggregateMeasureState],
    measure_input_aliases: &[String],
) -> Result<AggregateSpec, AggregateJoinPushdownRejectedReason> {
    // Both group keys and measure inputs are read from the materialized
    // measure-input projection above the join's measure-side input. The
    // projection preserves group/key column names by source name, so the
    // pre-aggregate's group_exprs still alias from source name to output name.
    let group_exprs = group_columns
        .iter()
        .map(|column| Expr::column(column.source_name.as_str()).alias(column.output_name.as_str()))
        .collect::<Vec<_>>();
    if measure_input_aliases.len() != preagg_measure_states.len()
        || aggregate.spec.aggr_exprs.len() != preagg_measure_states.len()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let aggr_exprs = measure_input_aliases
        .iter()
        .zip(preagg_measure_states)
        .flat_map(|(measure_input_alias, measure)| {
            measure.state_columns.iter().map(|state| {
                AggregateExpr::new(
                    state.preaggregate_function,
                    Expr::column(measure_input_alias.as_str()),
                    state.alias.as_str(),
                )
            })
        })
        .collect::<Vec<_>>();
    Ok(AggregateSpec::new(group_exprs, aggr_exprs))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn preaggregate_measure_states(
    input_schema: &SchemaRef,
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    state_aliases: &[Vec<String>],
    measure_input_aliases: &[String],
) -> Result<Vec<PreAggregateMeasureState>, AggregateJoinPushdownRejectedReason> {
    if aggregate.spec.aggr_exprs.len() != state_aliases.len()
        || aggregate.spec.aggr_exprs.len() != measure_input_aliases.len()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    aggregate
        .spec
        .aggr_exprs
        .iter()
        .zip(state_aliases)
        .zip(measure_input_aliases)
        .map(|((expr, aliases), measure_input_alias)| {
            let output_field = aggregate_output_schema
                .field_with_name(expr.alias.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
                .as_ref()
                .clone();
            let input_field = input_schema
                .field_with_name(measure_input_alias.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
            let state_columns = match expr.function {
                AggregateFunction::Sum | AggregateFunction::Count => {
                    let alias =
                        aliases.first().ok_or(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
                    vec![PreAggregateStateColumn {
                        alias: alias.clone(),
                        preaggregate_function: expr.function,
                        field: Field::new(alias.as_str(), output_field.data_type().clone(), output_field.is_nullable()),
                    }]
                },
                AggregateFunction::Avg => {
                    if aliases.len() != 2 {
                        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
                    }
                    let sum_alias = aliases[0].clone();
                    let count_alias = aliases[1].clone();
                    let sum_type = avg_pushdown_sum_state_data_type(input_field.data_type(), output_field.data_type())?;
                    vec![
                        PreAggregateStateColumn {
                            alias: sum_alias.clone(),
                            preaggregate_function: AggregateFunction::Sum,
                            field: Field::new(sum_alias.as_str(), sum_type, true),
                        },
                        PreAggregateStateColumn {
                            alias: count_alias.clone(),
                            preaggregate_function: AggregateFunction::Count,
                            field: Field::new(count_alias.as_str(), DataType::Int64, false),
                        },
                    ]
                },
                _ => return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
            };
            Ok(PreAggregateMeasureState {
                original_function: expr.function,
                original_alias: expr.alias.clone(),
                final_output_field: output_field,
                state_columns,
            })
        })
        .collect()
}

/// Builds the projection outputs for the materializing projection inserted
/// above the join's measure-side input. Group/key columns are passed through
/// by source name; each measure expression is materialized once under a fresh
/// alias so the pre-aggregate that consumes the projection can reference it
/// as a plain column.
pub(in crate::optimizer::aggregate_join_pushdown) fn measure_input_projection_outputs(
    measure_input_schema: &SchemaRef,
    aggregate_input_schema: &SchemaRef,
    aggregate: &AggregateNode,
    group_columns: &[PreAggregateColumn],
    lineage: &AggregateJoinLineage,
    measure_input_aliases: &[String],
) -> Result<Vec<crate::plan::ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
    if lineage.measure_columns.len() != measure_input_aliases.len()
        || aggregate.spec.aggr_exprs.len() != measure_input_aliases.len()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let mut outputs = Vec::with_capacity(group_columns.len() + measure_input_aliases.len());
    for column in group_columns {
        let field = measure_input_schema
            .field_with_name(column.source_name.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
            .clone();
        outputs.push(crate::plan::ProjectionOutput::new(Expr::column(column.source_name.as_str()), field));
    }
    for ((measure, alias), aggr_expr) in
        lineage.measure_columns.iter().zip(measure_input_aliases).zip(aggregate.spec.aggr_exprs.iter())
    {
        // Resolve the measure-input value type from the aggregate's input
        // schema, using the original aggr-expr column reference. The
        // projection chain preserved this name at the aggregate boundary, so
        // its arrow type matches what the inlined side-input expression
        // evaluates to.
        let original_column = simple_projection_column(&aggr_expr.expr)
            .ok_or(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
        let aggregate_input_field = aggregate_input_schema
            .field_with_name(original_column.name())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        let field =
            Field::new(alias.as_str(), aggregate_input_field.data_type().clone(), aggregate_input_field.is_nullable());
        outputs.push(crate::plan::ProjectionOutput::new(
            measure.side_input_aggregate.expr.clone().alias(alias.as_str()),
            field,
        ));
    }
    Ok(outputs)
}

pub(in crate::optimizer::aggregate_join_pushdown) fn unique_measure_input_alias(
    index: usize,
    aggregate_output_schema: &Schema,
    group_columns: &[PreAggregateColumn],
    preagg_aliases: &[String],
) -> String {
    let mut used = aggregate_output_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    used.extend(group_columns.iter().map(|column| column.output_name.clone()));
    used.extend(group_columns.iter().map(|column| column.source_name.clone()));
    used.extend(preagg_aliases.iter().cloned());
    for suffix in 0usize.. {
        let candidate = if suffix == 0 {
            format!("__nexus_native_preagg_arg_{index}")
        } else {
            format!("__nexus_native_preagg_arg_{index}_{suffix}")
        };
        if !used.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("infinite suffix search must find a measure-input alias")
}

pub(in crate::optimizer::aggregate_join_pushdown) fn decomposable_union_partial_aggregate_expr(
    expr: &AggregateExpr,
) -> bool {
    matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Count)
        && !expr.distinct
        && expr.filter.is_none()
}

pub(in crate::optimizer::aggregate_join_pushdown) fn join_with_replaced_input(
    join: &JoinNode,
    side: JoinSide,
    replacement: NodeId,
) -> JoinNode {
    let mut rewritten = join.clone();
    rewritten.inputs[join_side_index(side)] = replacement;
    rewritten
}

pub(in crate::optimizer::aggregate_join_pushdown) fn final_aggregate_spec_after_union_preagg(
    preagg_spec: &AggregateSpec,
    preagg_schema: &SchemaRef,
) -> Option<AggregateSpec> {
    if preagg_spec.grouping_sets.is_some() || preagg_schema.fields().len() < preagg_spec.group_exprs.len() {
        return None;
    }
    let group_exprs = preagg_schema
        .fields()
        .iter()
        .take(preagg_spec.group_exprs.len())
        .map(|field| Expr::column(field.name()))
        .collect::<Vec<_>>();
    let aggr_exprs = preagg_spec
        .aggr_exprs
        .iter()
        .map(|expr| AggregateExpr::new(AggregateFunction::Sum, Expr::column(expr.alias.as_str()), expr.alias.as_str()))
        .collect::<Vec<_>>();
    Some(AggregateSpec::new(group_exprs, aggr_exprs))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn new_join_output_schema(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_schema: &SchemaRef,
    preagg_measure_states: &[PreAggregateMeasureState],
    preagg_side: JoinSide,
    join_type: JoinType,
) -> Result<SchemaRef, AggregateJoinPushdownRejectedReason> {
    let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
    let mut fields = Vec::with_capacity(aggregate.spec.group_exprs.len() + state_count);
    for group in &aggregate.spec.group_exprs {
        let column =
            simple_projection_column(group).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
        let field = aggregate_output_schema
            .field_with_name(column.name())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        fields.push(field.as_ref().clone());
    }
    for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
        let field = preagg_schema
            .field_with_name(state.alias.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        let mut field = field.as_ref().clone();
        if join_side_introduces_nulls(join_type, preagg_side) && !field.is_nullable() {
            field = Field::new(field.name().as_str(), field.data_type().clone(), true);
        }
        fields.push(field);
    }
    Ok(Arc::new(Schema::new(fields)))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn coalesce_preagg_projection_outputs(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<Vec<ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
    let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
    let mut outputs = Vec::with_capacity(aggregate.spec.group_exprs.len() + state_count);
    append_group_projection_outputs(&mut outputs, aggregate, aggregate_output_schema)?;

    for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
        let zero = zero_literal_for_data_type(state.field.data_type())?;
        let field = Field::new(state.alias.as_str(), state.field.data_type().clone(), state.field.is_nullable());
        let coalesced = Expr::case(
            vec![CaseWhen::new(Expr::is_null(Expr::column(state.alias.as_str())), Expr::literal(zero))],
            Expr::column(state.alias.as_str()),
        )
        .alias(state.alias.as_str());
        outputs.push(ProjectionOutput::new(coalesced, field));
    }
    Ok(outputs)
}

pub(in crate::optimizer::aggregate_join_pushdown) fn append_group_projection_outputs(
    outputs: &mut Vec<ProjectionOutput>,
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
) -> Result<(), AggregateJoinPushdownRejectedReason> {
    for group in &aggregate.spec.group_exprs {
        let column =
            simple_projection_column(group).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
        let field = aggregate_output_schema
            .field_with_name(column.name())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
            .as_ref()
            .clone();
        outputs.push(ProjectionOutput::new(Expr::column(column.name()), field));
    }
    Ok(())
}

pub(in crate::optimizer::aggregate_join_pushdown) fn zero_literal_for_data_type(
    data_type: &arrow::datatypes::DataType,
) -> Result<crate::expr::LiteralValue, AggregateJoinPushdownRejectedReason> {
    match data_type {
        arrow::datatypes::DataType::Int8 => Ok(crate::expr::LiteralValue::Int8(0)),
        arrow::datatypes::DataType::Int16 => Ok(crate::expr::LiteralValue::Int16(0)),
        arrow::datatypes::DataType::Int32 => Ok(crate::expr::LiteralValue::Int32(0)),
        arrow::datatypes::DataType::Int64 => Ok(crate::expr::LiteralValue::Int64(0)),
        arrow::datatypes::DataType::UInt8 => Ok(crate::expr::LiteralValue::UInt8(0)),
        arrow::datatypes::DataType::UInt16 => Ok(crate::expr::LiteralValue::UInt16(0)),
        arrow::datatypes::DataType::UInt32 => Ok(crate::expr::LiteralValue::UInt32(0)),
        arrow::datatypes::DataType::UInt64 => Ok(crate::expr::LiteralValue::UInt64(0)),
        _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) fn preaggregate_state_aliases(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    group_columns: &[PreAggregateColumn],
) -> Vec<Vec<String>> {
    let mut used = aggregate_output_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    used.extend(group_columns.iter().map(|column| column.output_name.clone()));
    used.extend(group_columns.iter().map(|column| column.source_name.clone()));
    aggregate
        .spec
        .aggr_exprs
        .iter()
        .enumerate()
        .map(|(index, expr)| match expr.function {
            AggregateFunction::Avg => vec![
                unique_preagg_state_alias(format!("__nexus_native_preagg_avg_{index}_sum"), &mut used),
                unique_preagg_state_alias(format!("__nexus_native_preagg_avg_{index}_count"), &mut used),
            ],
            _ => vec![unique_preagg_state_alias(format!("__nexus_native_preagg_sum_{index}"), &mut used)],
        })
        .collect()
}

pub(in crate::optimizer::aggregate_join_pushdown) fn unique_preagg_state_alias(
    base: String,
    used: &mut BTreeSet<String>,
) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("infinite suffix search must find a preaggregate state alias")
}
