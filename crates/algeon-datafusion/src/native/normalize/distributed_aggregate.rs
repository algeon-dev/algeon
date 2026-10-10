use std::sync::Arc;

use arrow::datatypes::{DataType, FieldRef, SchemaRef};
use cudf::interop::arrow_host::is_supported_arrow_type;
use datafusion::{
    physical_expr::{PhysicalExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
        sorts::sort_preserving_merge::SortPreservingMergeExec,
    },
};
use query_planning::expr::AggregateFunction as NativeAggregateFunction;

use super::{NativePlanRejection, peel_repartition, reads_union_branches};
use crate::{
    capability::CapabilityReason,
    native::{expression_contract::native_aggregate_function_kind, validate::validate_aggregate_for_native_lowering},
    report::{AggregateDefect, GpuCandidate, PlanBlocker, SinglePartitionOutcome},
};

#[derive(Debug, Clone)]
pub(in crate::native) struct DistributedAggregatePeel<'a> {
    pub(in crate::native) aggregate: &'a AggregateExec,
    pub(in crate::native) aggregate_allows_partial: bool,
    /// The repartition, coalesce or merge of sorted partitions between the
    /// final and partial aggregate.
    pub(in crate::native) exchange: Option<&'a str>,
    pub(in crate::native) aggregate_output_schema: SchemaRef,
}

pub(super) fn peel_distributed_aggregate_pair_if_present(
    aggregate: &AggregateExec,
    candidate: GpuCandidate,
) -> Result<DistributedAggregatePeel<'_>, NativePlanRejection> {
    // DataFusion distributed aggregates appear as final aggregate over a
    // repartition/coalesce/merge boundary over a partial aggregate. Lowering
    // keeps the partial node as the executable aggregate but preserves the
    // final schema.
    if !matches!(aggregate.mode(), AggregateMode::Final | AggregateMode::FinalPartitioned) {
        return Ok(DistributedAggregatePeel {
            aggregate,
            aggregate_allows_partial: false,
            exchange: None,
            aggregate_output_schema: aggregate.schema(),
        });
    }

    let below_exchange = peel_aggregate_exchange(aggregate.input().as_ref());
    let rejection = |candidate, blocker, single_stage: SinglePartitionOutcome| NativePlanRejection {
        candidate,
        reason: CapabilityReason::NativeDistributedAggregateUnsupported,
        blocker: Box::new(blocker),
        single_partition: if reads_union_branches(below_exchange) {
            SinglePartitionOutcome::StillRejected
        } else {
            single_stage
        },
    };
    let Some(partial) = below_exchange.downcast_ref::<AggregateExec>() else {
        let blocker = PlanBlocker::DistributedAggregate(format!(
            "the final stage reads {} instead of a partial stage",
            below_exchange.name()
        ));
        return Err(rejection(candidate, blocker, SinglePartitionOutcome::Lowers));
    };
    if let Some(blocker) = distributed_aggregate_pair_defect(aggregate, partial) {
        return Err(rejection(candidate, blocker, single_stage_outcome(partial)));
    }

    Ok(DistributedAggregatePeel {
        aggregate: partial,
        aggregate_allows_partial: true,
        exchange: aggregate.input().downcast_ref::<AggregateExec>().is_none().then(|| aggregate.input().name()),
        aggregate_output_schema: aggregate.schema(),
    })
}

/// The plan below the exchange that feeds a final aggregate. A merge of sorted
/// partitions is an exchange here: DataFusion places it below a final
/// aggregate that streams over sorted groups, and the merge only interleaves
/// the rows of its inputs. With a fetch it would drop rows, so it stays.
fn peel_aggregate_exchange(plan: &dyn ExecutionPlan) -> &dyn ExecutionPlan {
    match plan.downcast_ref::<SortPreservingMergeExec>() {
        Some(merge) if merge.fetch().is_none() => merge.input().as_ref(),
        _ => peel_repartition(plan),
    }
}

/// What one partition does to a pair whose partial stage is `partial`: it
/// lowers when DataFusion's single-stage aggregate over the same input passes
/// native validation. An aggregate expression that no native kernel serves is
/// rejected there as well.
///
/// DataFusion keeps both stages of a grouping set aggregate at any partition
/// count, because the final stage groups on the grouping id the partial stage
/// adds and `CombinePartialFinalAggregate` merges only stages with equal
/// groups.
fn single_stage_outcome(partial: &AggregateExec) -> SinglePartitionOutcome {
    let single_stage_lowers = !partial.group_expr().has_grouping_set()
        && AggregateExec::try_new(
            AggregateMode::Single,
            partial.group_expr().clone(),
            partial.aggr_expr().to_vec(),
            partial.filter_expr().to_vec(),
            Arc::clone(partial.input()),
            partial.input_schema(),
        )
        .is_ok_and(|single| validate_aggregate_for_native_lowering(&single, false).is_ok());
    if single_stage_lowers { SinglePartitionOutcome::Lowers } else { SinglePartitionOutcome::StillRejected }
}

/// What keeps a final aggregate and the partial aggregate below it from
/// lowering as one native aggregate, or `None` when the pair is safe.
///
/// Safety is structural: group columns must be the partial output columns,
/// aggregate functions, distinctness, and state fields must match, and no
/// order-by semantics may be hidden in the pair. Per-aggregate filters are
/// carried by the partial aggregate, where input rows are still available.
fn distributed_aggregate_pair_defect(
    final_aggregate: &AggregateExec,
    partial_aggregate: &AggregateExec,
) -> Option<PlanBlocker> {
    let stages = |detail: &str| Some(PlanBlocker::DistributedAggregate(detail.to_string()));
    if !matches!(partial_aggregate.mode(), AggregateMode::Partial) {
        return stages("the lower stage is not a partial aggregate");
    }
    if !distributed_group_expr_layout_is_native_safe(final_aggregate, partial_aggregate)
        || final_aggregate.group_expr().expr().len() != partial_group_output_count(partial_aggregate)
    {
        return stages("the stages differ in group key layout");
    }
    if final_aggregate.aggr_expr().len() != partial_aggregate.aggr_expr().len() {
        return stages("the stages differ in the number of aggregates");
    }
    if !distributed_aggregate_filters_are_native_safe(final_aggregate, partial_aggregate) {
        let filtered = final_aggregate
            .aggr_expr()
            .iter()
            .zip(final_aggregate.filter_expr().iter().zip(partial_aggregate.filter_expr()))
            .find_map(|(expr, (final_filter, partial_filter))| {
                distributed_filters_conflict(final_filter, partial_filter).then_some(expr)
            });
        return Some(match filtered.or_else(|| final_aggregate.aggr_expr().first()) {
            Some(expr) => {
                PlanBlocker::Aggregate { function: expr.fun().name().to_string(), defect: AggregateDefect::Filter }
            },
            None => PlanBlocker::DistributedAggregate("the stages differ in aggregate filters".to_string()),
        });
    }
    if !distributed_group_exprs_are_native_safe(final_aggregate, partial_aggregate) {
        return stages("the stages differ in group key layout");
    }
    for (final_expr, partial_expr) in final_aggregate.aggr_expr().iter().zip(partial_aggregate.aggr_expr()) {
        let kind = native_aggregate_function_kind(final_expr.fun());
        let aggregate = |defect| PlanBlocker::Aggregate { function: final_expr.fun().name().to_string(), defect };
        let kinds_match = matches!(
            (kind, native_aggregate_function_kind(partial_expr.fun())),
            (Some(final_kind), Some(partial_kind)) if final_kind == partial_kind
        );
        if !kinds_match || !distributed_aggregate_function_is_native_safe(kind) {
            return Some(aggregate(AggregateDefect::UnsupportedFunction));
        }
        if !distributed_final_output_is_native_safe(final_aggregate, final_expr.name(), kind) {
            return stages("the final output type has no native form");
        }
        if final_expr.is_distinct() != partial_expr.is_distinct()
            || (final_expr.is_distinct() && !matches!(kind, Some(NativeAggregateFunction::Count)))
        {
            return Some(aggregate(AggregateDefect::Distinct));
        }
        if !final_expr.order_bys().is_empty() || !partial_expr.order_bys().is_empty() {
            return Some(aggregate(AggregateDefect::OrderBy));
        }
    }
    if !distributed_aggregate_state_exprs_are_native_safe(final_aggregate, partial_aggregate) {
        return stages("the stages differ in aggregate state fields");
    }
    None
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
        && final_aggregate
            .filter_expr()
            .iter()
            .zip(partial_aggregate.filter_expr())
            .all(|(final_filter, partial_filter)| !distributed_filters_conflict(final_filter, partial_filter))
}

fn distributed_filters_conflict(
    final_filter: &Option<Arc<dyn PhysicalExpr>>,
    partial_filter: &Option<Arc<dyn PhysicalExpr>>,
) -> bool {
    match (final_filter, partial_filter) {
        (Some(final_filter), Some(partial_filter)) => !final_filter.eq(partial_filter),
        (None, _) => false,
        (Some(_), None) => true,
    }
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

fn distributed_aggregate_function_is_native_safe(kind: Option<NativeAggregateFunction>) -> bool {
    // AVG is allowed only because the state schema checks below require
    // DataFusion's explicit sum/count state layout and the final output check
    // requires a floating result. It is not an average-of-averages shortcut.
    matches!(
        kind,
        Some(
            NativeAggregateFunction::Sum
                | NativeAggregateFunction::Min
                | NativeAggregateFunction::Max
                | NativeAggregateFunction::Count
                | NativeAggregateFunction::Avg
        )
    )
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
    state_fields: &[FieldRef],
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
    kind: Option<NativeAggregateFunction>,
) -> bool {
    // AVG finalization consumes numeric state and produces a floating value;
    // other distributed functions use the normal native type whitelist.
    let schema = final_aggregate.schema();
    let Ok(field) = schema.field_with_name(aggregate_name) else {
        return false;
    };
    if matches!(kind, Some(NativeAggregateFunction::Avg)) {
        matches!(field.data_type(), DataType::Float32 | DataType::Float64 | DataType::Decimal128(_, _))
    } else {
        is_supported_arrow_type(field.data_type())
    }
}
