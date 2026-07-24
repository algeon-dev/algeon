use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    physical_expr::expressions::Column,
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
    },
};

use super::{NativePlanRejection, peel_repartition};
use crate::{
    capability::CapabilityReason, native::expression_contract::is_supported_native_type, report::GpuCandidate,
};

#[derive(Debug, Clone)]
pub(in crate::native) struct DistributedAggregatePeel<'a> {
    pub(in crate::native) aggregate: &'a AggregateExec,
    pub(in crate::native) aggregate_allows_partial: bool,
    pub(in crate::native) has_distributed_aggregate: bool,
    pub(in crate::native) aggregate_output_schema: SchemaRef,
}

pub(super) fn peel_distributed_aggregate_pair_if_present(
    aggregate: &AggregateExec,
    candidate: GpuCandidate,
) -> Result<DistributedAggregatePeel<'_>, NativePlanRejection> {
    // DataFusion distributed aggregates appear as final aggregate over a
    // repartition/coalesce boundary over a partial aggregate. Lowering keeps the
    // partial node as the executable aggregate but preserves the final schema.
    if !matches!(aggregate.mode(), AggregateMode::Final | AggregateMode::FinalPartitioned) {
        return Ok(DistributedAggregatePeel {
            aggregate,
            aggregate_allows_partial: false,
            has_distributed_aggregate: false,
            aggregate_output_schema: aggregate.schema(),
        });
    }

    let partial =
        peel_repartition(aggregate.input().as_ref()).downcast_ref::<AggregateExec>().ok_or(NativePlanRejection {
            candidate: candidate.clone(),
            reason: CapabilityReason::NativeDistributedAggregateUnsupported,
        })?;
    if !distributed_aggregate_pair_is_safe(aggregate, partial) {
        return Err(NativePlanRejection { candidate, reason: CapabilityReason::NativeDistributedAggregateUnsupported });
    }

    Ok(DistributedAggregatePeel {
        aggregate: partial,
        aggregate_allows_partial: true,
        has_distributed_aggregate: true,
        aggregate_output_schema: aggregate.schema(),
    })
}

fn distributed_aggregate_pair_is_safe(final_aggregate: &AggregateExec, partial_aggregate: &AggregateExec) -> bool {
    // Safety is structural: group columns must be the partial output columns,
    // aggregate functions and state fields must match, and no distinct/order
    // by semantics may be hidden in the pair. Per-aggregate filters are
    // carried by the partial aggregate, where input rows are still available.
    matches!(partial_aggregate.mode(), AggregateMode::Partial)
        && distributed_group_expr_layout_is_native_safe(final_aggregate, partial_aggregate)
        && final_aggregate.group_expr().expr().len() == partial_group_output_count(partial_aggregate)
        && final_aggregate.aggr_expr().len() == partial_aggregate.aggr_expr().len()
        && distributed_aggregate_filters_are_native_safe(final_aggregate, partial_aggregate)
        && distributed_group_exprs_are_native_safe(final_aggregate, partial_aggregate)
        && final_aggregate.aggr_expr().iter().zip(partial_aggregate.aggr_expr()).all(|(final_expr, partial_expr)| {
            final_expr.fun().name().eq_ignore_ascii_case(partial_expr.fun().name())
                && distributed_aggregate_function_is_native_safe(final_expr.fun().name())
                && distributed_final_output_is_native_safe(final_aggregate, final_expr.name(), final_expr.fun().name())
                && !final_expr.is_distinct()
                && !partial_expr.is_distinct()
                && final_expr.order_bys().is_empty()
                && partial_expr.order_bys().is_empty()
        })
        && distributed_aggregate_state_exprs_are_native_safe(final_aggregate, partial_aggregate)
}

fn distributed_aggregate_filters_are_native_safe(
    final_aggregate: &AggregateExec,
    partial_aggregate: &AggregateExec,
) -> bool {
    // SQL aggregate filters must be evaluated while the original input rows are
    // still available. DataFusion may keep matching filter metadata on the
    // final aggregate expression, but the state-schema proof below lets native
    // lowering execute the partial aggregate directly and use the final node
    // only as the output-schema contract. Mismatched final/partial filters are
    // rejected because they would no longer prove the same SQL aggregate.
    final_aggregate.filter_expr().len() == final_aggregate.aggr_expr().len()
        && partial_aggregate.filter_expr().len() == partial_aggregate.aggr_expr().len()
        && final_aggregate.filter_expr().iter().zip(partial_aggregate.filter_expr()).all(
            |(final_filter, partial_filter)| match (final_filter, partial_filter) {
                (Some(final_filter), Some(partial_filter)) => final_filter.eq(partial_filter),
                (None, Some(_partial_filter)) => true,
                (None, None) => true,
                _ => false,
            },
        )
}

fn partial_group_output_count(partial_aggregate: &AggregateExec) -> usize {
    partial_group_output_names(partial_aggregate).len()
}

fn distributed_group_expr_layout_is_native_safe(
    final_aggregate: &AggregateExec,
    partial_aggregate: &AggregateExec,
) -> bool {
    let final_group_by = final_aggregate.group_expr();
    let partial_group_by = partial_aggregate.group_expr();
    if partial_group_by.has_grouping_set() {
        let partial_group_expr_count = partial_group_by.expr().len();
        final_group_by.is_single()
            && !final_group_by.has_grouping_set()
            && final_group_by.expr().len() == partial_group_expr_count + 1
            && partial_group_by.null_expr().len() == partial_group_expr_count
            && partial_group_by.groups().iter().all(|group| group.len() == partial_group_expr_count)
    } else {
        final_group_by.expr().len() == partial_group_by.expr().len()
    }
}

fn distributed_group_exprs_are_native_safe(final_aggregate: &AggregateExec, partial_aggregate: &AggregateExec) -> bool {
    final_aggregate.group_expr().expr().iter().zip(partial_group_output_names(partial_aggregate)).all(
        |((final_expr, _final_alias), partial_alias)| {
            final_expr.downcast_ref::<Column>().is_some_and(|final_column| final_column.name() == partial_alias)
        },
    )
}

fn partial_group_output_names(partial_aggregate: &AggregateExec) -> Vec<String> {
    partial_aggregate
        .group_expr()
        .expr()
        .iter()
        .map(|(_expr, alias)| alias.clone())
        .chain(partial_aggregate.group_expr().has_grouping_set().then(|| "__grouping_id".to_string()))
        .collect()
}

fn distributed_aggregate_function_is_native_safe(name: &str) -> bool {
    // AVG is allowed only because the state schema checks below require
    // DataFusion's explicit sum/count state layout and the final output check
    // requires a floating result. It is not an average-of-averages shortcut.
    name.eq_ignore_ascii_case("sum")
        || name.eq_ignore_ascii_case("min")
        || name.eq_ignore_ascii_case("max")
        || name.eq_ignore_ascii_case("count")
        || name.eq_ignore_ascii_case("avg")
}

fn distributed_aggregate_state_exprs_are_native_safe(
    final_aggregate: &AggregateExec,
    partial_aggregate: &AggregateExec,
) -> bool {
    // The final aggregate input must expose exactly the partial state fields in
    // schema order. This is the provenance check that lets lowering execute the
    // partial aggregate while reporting the final aggregate output schema.
    let mut state_index_base = final_aggregate.group_expr().expr().len();
    let final_input_schema = final_aggregate.input().schema();
    let partial_output_schema = partial_aggregate.schema();
    for (final_expr, partial_expr) in final_aggregate.aggr_expr().iter().zip(partial_aggregate.aggr_expr()) {
        let Ok(state_fields) = partial_expr.state_fields() else {
            return false;
        };
        let Ok(final_state_fields) = final_expr.state_fields() else {
            return false;
        };
        if final_state_fields != state_fields
            || !distributed_state_fields_match_schema(&state_fields, &partial_output_schema, state_index_base)
            || !distributed_state_fields_match_schema(&state_fields, &final_input_schema, state_index_base)
        {
            return false;
        }
        state_index_base += state_fields.len();
    }
    true
}

fn distributed_state_fields_match_schema(
    state_fields: &[arrow::datatypes::FieldRef],
    schema: &SchemaRef,
    state_index_base: usize,
) -> bool {
    state_index_base.checked_add(state_fields.len()).is_some_and(|end| end <= schema.fields().len())
        && state_fields
            .iter()
            .enumerate()
            .all(|(state_offset, field)| schema.field(state_index_base + state_offset) == field.as_ref())
}

fn distributed_final_output_is_native_safe(
    final_aggregate: &AggregateExec,
    aggregate_name: &str,
    function_name: &str,
) -> bool {
    // AVG finalization consumes numeric state and produces a floating value;
    // other distributed functions use the normal native type whitelist.
    let schema = final_aggregate.schema();
    let Ok(field) = schema.field_with_name(aggregate_name) else {
        return false;
    };
    if function_name.eq_ignore_ascii_case("avg") {
        matches!(field.data_type(), DataType::Float32 | DataType::Float64 | DataType::Decimal128(_, _))
    } else {
        is_supported_native_type(field.data_type())
    }
}
