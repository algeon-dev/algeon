use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    datasource::source::DataSourceExec,
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use super::validate::is_supported_native_type;
use crate::{policy::FallbackReason, report::GpuCandidate};

const LOCAL_LIMIT_SHAPE: &str = "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))";
const LOCAL_LIMIT_NO_FILTER_SHAPE: &str = "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(LocalParquetScan))))";
const LOCAL_LIMIT_NO_PRE_PROJECTION_SHAPE: &str =
    "LocalLimitExec(SortExec(AggregateExec(FilterExec(LocalParquetScan))))";
const LOCAL_LIMIT_MINIMAL_SHAPE: &str = "LocalLimitExec(SortExec(AggregateExec(LocalParquetScan)))";
const LOCAL_LIMIT_POST_PRE_FILTER_SHAPE: &str =
    "LocalLimitExec(SortExec(ProjectionExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan))))))";
const LOCAL_LIMIT_POST_PRE_NO_FILTER_SHAPE: &str =
    "LocalLimitExec(SortExec(ProjectionExec(AggregateExec(ProjectionExec(LocalParquetScan)))))";
const LOCAL_LIMIT_POST_FILTER_SHAPE: &str =
    "LocalLimitExec(SortExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))))";
const LOCAL_LIMIT_POST_NO_FILTER_SHAPE: &str =
    "LocalLimitExec(SortExec(ProjectionExec(AggregateExec(LocalParquetScan))))";
const SORT_FETCH_SHAPE: &str = "SortExec(fetch)(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan))))";
const SORT_FETCH_NO_FILTER_SHAPE: &str = "SortExec(fetch)(AggregateExec(ProjectionExec(LocalParquetScan)))";
const SORT_FETCH_NO_PRE_PROJECTION_SHAPE: &str = "SortExec(fetch)(AggregateExec(FilterExec(LocalParquetScan)))";
const SORT_FETCH_MINIMAL_SHAPE: &str = "SortExec(fetch)(AggregateExec(LocalParquetScan))";
const SQL_SORT_FETCH_POST_PRE_FILTER_SHAPE: &str =
    "SortExec(fetch)(ProjectionExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))";
const SQL_SORT_FETCH_POST_PRE_NO_FILTER_SHAPE: &str =
    "SortExec(fetch)(ProjectionExec(AggregateExec(ProjectionExec(LocalParquetScan))))";
const SQL_SORT_FETCH_SHAPE: &str = "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))";
const SQL_SORT_FETCH_NO_FILTER_SHAPE: &str = "SortExec(fetch)(ProjectionExec(AggregateExec(LocalParquetScan)))";
const SPM_SORT_FETCH_POST_PRE_FILTER_SHAPE: &str = "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan))))))";
const SPM_SORT_FETCH_POST_PRE_NO_FILTER_SHAPE: &str =
    "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(AggregateExec(ProjectionExec(LocalParquetScan)))))";
const SPM_SORT_FETCH_POST_FILTER_SHAPE: &str =
    "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))))";
const SPM_SORT_FETCH_DISTRIBUTED_POST_FILTER_SHAPE: &str = "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec(RepartitionExec(PartialAggregateExec(FilterExec(RepartitionExec(LocalParquetScan)))))))";
const SPM_SORT_FETCH_POST_NO_FILTER_SHAPE: &str =
    "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(AggregateExec(LocalParquetScan))))";
const SPM_SORT_FETCH_PRE_FILTER_SHAPE: &str =
    "SortPreservingMergeExec(fetch)(SortExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))";
const SPM_SORT_FETCH_PRE_NO_FILTER_SHAPE: &str =
    "SortPreservingMergeExec(fetch)(SortExec(AggregateExec(ProjectionExec(LocalParquetScan))))";
const SPM_SORT_FETCH_NO_PRE_FILTER_SHAPE: &str =
    "SortPreservingMergeExec(fetch)(SortExec(AggregateExec(FilterExec(LocalParquetScan))))";
const SPM_SORT_FETCH_MINIMAL_SHAPE: &str = "SortPreservingMergeExec(fetch)(SortExec(AggregateExec(LocalParquetScan)))";
const GLOBAL_AGG_POST_SHAPE: &str = "ProjectionExec(AggregateExec(LocalParquetScan))";
const GLOBAL_AGG_MINIMAL_SHAPE: &str = "AggregateExec(LocalParquetScan)";
const GLOBAL_AGG_POST_PRE_FILTER_SHAPE: &str =
    "ProjectionExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan))))";
const GLOBAL_AGG_POST_PRE_NO_FILTER_SHAPE: &str = "ProjectionExec(AggregateExec(ProjectionExec(LocalParquetScan)))";
const GLOBAL_AGG_POST_FILTER_SHAPE: &str = "ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))";
const GLOBAL_AGG_PRE_FILTER_SHAPE: &str = "AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))";
const GLOBAL_AGG_PRE_NO_FILTER_SHAPE: &str = "AggregateExec(ProjectionExec(LocalParquetScan))";
const GLOBAL_AGG_FILTER_SHAPE: &str = "AggregateExec(FilterExec(LocalParquetScan))";
const GLOBAL_AGG_DISTRIBUTED_POST_SHAPE: &str =
    "ProjectionExec(FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(LocalParquetScan))))";
const GLOBAL_AGG_DISTRIBUTED_MINIMAL_SHAPE: &str =
    "FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(LocalParquetScan)))";
const GLOBAL_AGG_DISTRIBUTED_POST_PRE_FILTER_SHAPE: &str = "ProjectionExec(FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(ProjectionExec(FilterExec(LocalParquetScan))))))";
const GLOBAL_AGG_DISTRIBUTED_POST_PRE_NO_FILTER_SHAPE: &str = "ProjectionExec(FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(ProjectionExec(LocalParquetScan)))))";
const GLOBAL_AGG_DISTRIBUTED_POST_FILTER_SHAPE: &str =
    "ProjectionExec(FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(FilterExec(LocalParquetScan)))))";
const GLOBAL_AGG_DISTRIBUTED_PRE_FILTER_SHAPE: &str =
    "FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))";
const GLOBAL_AGG_DISTRIBUTED_PRE_NO_FILTER_SHAPE: &str =
    "FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(ProjectionExec(LocalParquetScan))))";
const GLOBAL_AGG_DISTRIBUTED_FILTER_SHAPE: &str =
    "FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(FilterExec(LocalParquetScan))))";
const SORT_PRESERVING_MERGE_SHAPE: &str = "SortPreservingMergeExec";
const SORT_PRESERVING_MERGE_FETCH_SHAPE: &str = "SortPreservingMergeExec(fetch)(SortExec)";
const UNSUPPORTED_NATIVE_PLAN_SHAPE: &str = "UnsupportedNativePhysicalPlan";

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativePlan<'a> {
    pub(super) candidate: GpuCandidate,
    pub(super) source_exec: &'a DataSourceExec,
    pub(super) filter: Option<&'a FilterExec>,
    pub(super) pre_aggregate_projection: Option<&'a ProjectionExec>,
    pub(super) aggregate: &'a AggregateExec,
    pub(super) aggregate_allows_partial: bool,
    pub(super) aggregate_output_schema: SchemaRef,
    pub(super) post_aggregate_projection: Option<&'a ProjectionExec>,
    pub(super) sort: Option<&'a SortExec>,
    pub(super) limit_fetch: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NativePlanRejection {
    pub(super) candidate: GpuCandidate,
    pub(super) reason: FallbackReason,
}

#[derive(Debug, Clone, Copy)]
enum NativeStackRoot {
    LocalLimit,
    SortFetch,
    SortPreservingMergeFetch,
}

pub(super) fn normalize_native_plan(plan: &dyn ExecutionPlan) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    if let Some(merge) = plan.as_any().downcast_ref::<SortPreservingMergeExec>() {
        let Some(fetch) = merge.fetch() else {
            return Err(NativePlanRejection {
                candidate: GpuCandidate::with_operator_count(
                    SORT_PRESERVING_MERGE_SHAPE,
                    plan.schema().fields().len(),
                    1,
                ),
                reason: FallbackReason::NativeSortPreservingMergeUnsupported,
            });
        };
        let candidate =
            GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_FETCH_SHAPE, plan.schema().fields().len(), 2);
        let sort = merge
            .input()
            .as_any()
            .downcast_ref::<SortExec>()
            .ok_or(NativePlanRejection { candidate, reason: FallbackReason::NativeSortPreservingMergeUnsupported })?;
        if !sort_orderings_match(merge.expr(), sort.expr()) {
            return Err(NativePlanRejection {
                candidate,
                reason: FallbackReason::NativeSortPreservingMergeUnsupported,
            });
        }
        if sort.fetch().is_some_and(|sort_fetch| sort_fetch < fetch) {
            return Err(NativePlanRejection {
                candidate,
                reason: FallbackReason::NativeSortPreservingMergeUnsupported,
            });
        }
        return normalize_sort_stack(
            NativeStackRoot::SortPreservingMergeFetch,
            plan.schema().fields().len(),
            sort,
            Some(fetch),
        );
    }

    if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
        let candidate = GpuCandidate::with_operator_count(LOCAL_LIMIT_SHAPE, plan.schema().fields().len(), 5);
        let sort = required_child::<SortExec>(limit.input().as_ref(), candidate)?;
        let fetch = sort.fetch().map_or(limit.fetch(), |sort_fetch| sort_fetch.min(limit.fetch()));
        return normalize_sort_stack(NativeStackRoot::LocalLimit, plan.schema().fields().len(), sort, Some(fetch));
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        let candidate = GpuCandidate::with_operator_count(SORT_FETCH_SHAPE, plan.schema().fields().len(), 5);
        let Some(fetch) = sort.fetch() else {
            return Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator });
        };
        return normalize_sort_stack(NativeStackRoot::SortFetch, plan.schema().fields().len(), sort, Some(fetch));
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        let candidate = GpuCandidate::with_operator_count(GLOBAL_AGG_POST_SHAPE, plan.schema().fields().len(), 3);
        let aggregate = required_child::<AggregateExec>(projection.input().as_ref(), candidate)?;
        return normalize_global_aggregate_stack(plan.schema().fields().len(), Some(projection), aggregate);
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        return normalize_global_aggregate_stack(plan.schema().fields().len(), None, aggregate);
    }

    Err(NativePlanRejection {
        candidate: GpuCandidate::with_operator_count(UNSUPPORTED_NATIVE_PLAN_SHAPE, plan.schema().fields().len(), 1),
        reason: FallbackReason::UnsupportedOperator,
    })
}

fn normalize_sort_stack<'a>(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &'a SortExec,
    limit_fetch: Option<usize>,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let initial_candidate = native_stack_candidate(root, output_schema_fields, false, true, true, false);
    if sort.preserve_partitioning() && !matches!(root, NativeStackRoot::SortPreservingMergeFetch) {
        return Err(NativePlanRejection {
            candidate: initial_candidate,
            reason: FallbackReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let Some(limit_fetch) = limit_fetch else {
        return Err(NativePlanRejection { candidate: initial_candidate, reason: FallbackReason::UnsupportedOperator });
    };
    let (post_aggregate_projection, aggregate) = peel_post_aggregate_projection(sort, initial_candidate)?;
    let (aggregate, aggregate_allows_partial, has_distributed_aggregate, aggregate_output_schema) =
        peel_distributed_aggregate_pair_if_present(aggregate, initial_candidate)?;
    let (pre_aggregate_projection, filter_or_source) = peel_pre_aggregate_projection(aggregate);
    let (filter, source_input) = peel_filter(peel_repartition(filter_or_source));
    let candidate = native_stack_candidate(
        root,
        output_schema_fields,
        post_aggregate_projection.is_some(),
        pre_aggregate_projection.is_some(),
        filter.is_some(),
        has_distributed_aggregate,
    );
    let source_exec = required_child::<DataSourceExec>(peel_repartition(source_input), candidate)?;

    Ok(NormalizedNativePlan {
        candidate,
        source_exec,
        filter,
        pre_aggregate_projection,
        aggregate,
        aggregate_allows_partial,
        aggregate_output_schema,
        post_aggregate_projection,
        sort: Some(sort),
        limit_fetch: Some(limit_fetch),
    })
}

fn normalize_global_aggregate_stack<'a>(
    output_schema_fields: usize,
    post_aggregate_projection: Option<&'a ProjectionExec>,
    aggregate: &'a AggregateExec,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let candidate = GpuCandidate::with_operator_count(
        if post_aggregate_projection.is_some() { GLOBAL_AGG_POST_SHAPE } else { GLOBAL_AGG_MINIMAL_SHAPE },
        output_schema_fields,
        2 + usize::from(post_aggregate_projection.is_some()),
    );
    let (aggregate, aggregate_allows_partial, has_distributed_aggregate, aggregate_output_schema) =
        peel_distributed_aggregate_pair_if_present(aggregate, candidate)?;
    let (pre_aggregate_projection, filter_or_source) = peel_pre_aggregate_projection(aggregate);
    let (filter, source_input) = peel_filter(peel_repartition(filter_or_source));
    let candidate = global_aggregate_candidate(
        output_schema_fields,
        post_aggregate_projection.is_some(),
        pre_aggregate_projection.is_some(),
        filter.is_some(),
        has_distributed_aggregate,
    );
    let source_exec = required_child::<DataSourceExec>(peel_repartition(source_input), candidate)?;

    Ok(NormalizedNativePlan {
        candidate,
        source_exec,
        filter,
        pre_aggregate_projection,
        aggregate,
        aggregate_allows_partial,
        aggregate_output_schema,
        post_aggregate_projection,
        sort: None,
        limit_fetch: None,
    })
}

fn global_aggregate_candidate(
    output_schema_fields: usize,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    let shape = match (has_distributed_aggregate, has_post_projection, has_pre_projection, has_filter) {
        (false, true, true, true) => GLOBAL_AGG_POST_PRE_FILTER_SHAPE,
        (false, true, true, false) => GLOBAL_AGG_POST_PRE_NO_FILTER_SHAPE,
        (false, true, false, true) => GLOBAL_AGG_POST_FILTER_SHAPE,
        (false, true, false, false) => GLOBAL_AGG_POST_SHAPE,
        (false, false, true, true) => GLOBAL_AGG_PRE_FILTER_SHAPE,
        (false, false, true, false) => GLOBAL_AGG_PRE_NO_FILTER_SHAPE,
        (false, false, false, true) => GLOBAL_AGG_FILTER_SHAPE,
        (false, false, false, false) => GLOBAL_AGG_MINIMAL_SHAPE,
        (true, true, true, true) => GLOBAL_AGG_DISTRIBUTED_POST_PRE_FILTER_SHAPE,
        (true, true, true, false) => GLOBAL_AGG_DISTRIBUTED_POST_PRE_NO_FILTER_SHAPE,
        (true, true, false, true) => GLOBAL_AGG_DISTRIBUTED_POST_FILTER_SHAPE,
        (true, true, false, false) => GLOBAL_AGG_DISTRIBUTED_POST_SHAPE,
        (true, false, true, true) => GLOBAL_AGG_DISTRIBUTED_PRE_FILTER_SHAPE,
        (true, false, true, false) => GLOBAL_AGG_DISTRIBUTED_PRE_NO_FILTER_SHAPE,
        (true, false, false, true) => GLOBAL_AGG_DISTRIBUTED_FILTER_SHAPE,
        (true, false, false, false) => GLOBAL_AGG_DISTRIBUTED_MINIMAL_SHAPE,
    };
    let operator_count = 2
        + usize::from(has_post_projection)
        + usize::from(has_pre_projection)
        + usize::from(has_filter)
        + if has_distributed_aggregate { 2 } else { 0 };
    GpuCandidate::with_operator_count(shape, output_schema_fields, operator_count)
}

fn peel_post_aggregate_projection(
    sort: &SortExec,
    candidate: GpuCandidate,
) -> Result<(Option<&ProjectionExec>, &AggregateExec), NativePlanRejection> {
    if let Some(projection) = sort.input().as_any().downcast_ref::<ProjectionExec>() {
        return Ok((Some(projection), required_child::<AggregateExec>(projection.input().as_ref(), candidate)?));
    }
    Ok((None, required_child::<AggregateExec>(sort.input().as_ref(), candidate)?))
}

fn peel_pre_aggregate_projection(aggregate: &AggregateExec) -> (Option<&ProjectionExec>, &dyn ExecutionPlan) {
    if let Some(projection) = aggregate.input().as_any().downcast_ref::<ProjectionExec>() {
        (Some(projection), projection.input().as_ref())
    } else {
        (None, aggregate.input().as_ref())
    }
}

fn peel_distributed_aggregate_pair_if_present<'a>(
    aggregate: &'a AggregateExec,
    candidate: GpuCandidate,
) -> Result<(&'a AggregateExec, bool, bool, SchemaRef), NativePlanRejection> {
    if !matches!(aggregate.mode(), AggregateMode::Final | AggregateMode::FinalPartitioned) {
        return Ok((aggregate, false, false, aggregate.schema()));
    }

    let partial = peel_repartition(aggregate.input().as_ref())
        .as_any()
        .downcast_ref::<AggregateExec>()
        .ok_or(NativePlanRejection { candidate, reason: FallbackReason::NativeDistributedAggregateUnsupported })?;
    if !distributed_aggregate_pair_is_safe(aggregate, partial) {
        return Err(NativePlanRejection { candidate, reason: FallbackReason::NativeDistributedAggregateUnsupported });
    }

    Ok((partial, true, true, aggregate.schema()))
}

fn distributed_aggregate_pair_is_safe(final_aggregate: &AggregateExec, partial_aggregate: &AggregateExec) -> bool {
    matches!(partial_aggregate.mode(), AggregateMode::Partial)
        && final_aggregate.group_expr().expr().len() == partial_aggregate.group_expr().expr().len()
        && final_aggregate.aggr_expr().len() == partial_aggregate.aggr_expr().len()
        && final_aggregate.filter_expr().iter().all(Option::is_none)
        && partial_aggregate.filter_expr().iter().all(Option::is_none)
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

fn distributed_group_exprs_are_native_safe(final_aggregate: &AggregateExec, partial_aggregate: &AggregateExec) -> bool {
    final_aggregate.group_expr().expr().iter().zip(partial_aggregate.group_expr().expr()).all(
        |((final_expr, _final_alias), (_partial_expr, partial_alias))| {
            final_expr
                .as_any()
                .downcast_ref::<Column>()
                .is_some_and(|final_column| final_column.name() == partial_alias)
        },
    )
}

fn distributed_aggregate_function_is_native_safe(name: &str) -> bool {
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
    let schema = final_aggregate.schema();
    let Ok(field) = schema.field_with_name(aggregate_name) else {
        return false;
    };
    if function_name.eq_ignore_ascii_case("avg") {
        matches!(field.data_type(), DataType::Float32 | DataType::Float64)
    } else {
        is_supported_native_type(field.data_type())
    }
}

fn peel_repartition(plan: &dyn ExecutionPlan) -> &dyn ExecutionPlan {
    if let Some(repartition) = plan.as_any().downcast_ref::<RepartitionExec>() {
        repartition.input().as_ref()
    } else if let Some(coalesce) = plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
        coalesce.input().as_ref()
    } else {
        plan
    }
}

fn peel_filter(plan: &dyn ExecutionPlan) -> (Option<&FilterExec>, &dyn ExecutionPlan) {
    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        (Some(filter), filter.input().as_ref())
    } else {
        (None, plan)
    }
}

fn native_stack_candidate(
    root: NativeStackRoot,
    output_schema_fields: usize,
    has_post_aggregate_projection: bool,
    has_pre_aggregate_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    if has_distributed_aggregate {
        let shape = match (root, has_post_aggregate_projection, has_pre_aggregate_projection, has_filter) {
            (NativeStackRoot::SortPreservingMergeFetch, true, false, true) => {
                SPM_SORT_FETCH_DISTRIBUTED_POST_FILTER_SHAPE
            },
            _ => SORT_PRESERVING_MERGE_FETCH_SHAPE,
        };
        return GpuCandidate::with_operator_count(shape, output_schema_fields, 9);
    }

    let shape = match (root, has_post_aggregate_projection, has_pre_aggregate_projection, has_filter) {
        (NativeStackRoot::LocalLimit, false, true, true) => LOCAL_LIMIT_SHAPE,
        (NativeStackRoot::LocalLimit, false, true, false) => LOCAL_LIMIT_NO_FILTER_SHAPE,
        (NativeStackRoot::LocalLimit, false, false, true) => LOCAL_LIMIT_NO_PRE_PROJECTION_SHAPE,
        (NativeStackRoot::LocalLimit, false, false, false) => LOCAL_LIMIT_MINIMAL_SHAPE,
        (NativeStackRoot::LocalLimit, true, true, true) => LOCAL_LIMIT_POST_PRE_FILTER_SHAPE,
        (NativeStackRoot::LocalLimit, true, true, false) => LOCAL_LIMIT_POST_PRE_NO_FILTER_SHAPE,
        (NativeStackRoot::LocalLimit, true, false, true) => LOCAL_LIMIT_POST_FILTER_SHAPE,
        (NativeStackRoot::LocalLimit, true, false, false) => LOCAL_LIMIT_POST_NO_FILTER_SHAPE,
        (NativeStackRoot::SortFetch, false, true, true) => SORT_FETCH_SHAPE,
        (NativeStackRoot::SortFetch, false, true, false) => SORT_FETCH_NO_FILTER_SHAPE,
        (NativeStackRoot::SortFetch, false, false, true) => SORT_FETCH_NO_PRE_PROJECTION_SHAPE,
        (NativeStackRoot::SortFetch, false, false, false) => SORT_FETCH_MINIMAL_SHAPE,
        (NativeStackRoot::SortFetch, true, true, true) => SQL_SORT_FETCH_POST_PRE_FILTER_SHAPE,
        (NativeStackRoot::SortFetch, true, true, false) => SQL_SORT_FETCH_POST_PRE_NO_FILTER_SHAPE,
        (NativeStackRoot::SortFetch, true, false, true) => SQL_SORT_FETCH_SHAPE,
        (NativeStackRoot::SortFetch, true, false, false) => SQL_SORT_FETCH_NO_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, false, true, true) => SPM_SORT_FETCH_PRE_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, false, true, false) => SPM_SORT_FETCH_PRE_NO_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, false, false, true) => SPM_SORT_FETCH_NO_PRE_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, false, false, false) => SPM_SORT_FETCH_MINIMAL_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, true, true, true) => SPM_SORT_FETCH_POST_PRE_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, true, true, false) => SPM_SORT_FETCH_POST_PRE_NO_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, true, false, true) => SPM_SORT_FETCH_POST_FILTER_SHAPE,
        (NativeStackRoot::SortPreservingMergeFetch, true, false, false) => SPM_SORT_FETCH_POST_NO_FILTER_SHAPE,
    };
    let operator_count = 3
        + usize::from(has_post_aggregate_projection)
        + usize::from(has_pre_aggregate_projection)
        + usize::from(has_filter);
    GpuCandidate::with_operator_count(
        shape,
        output_schema_fields,
        operator_count + usize::from(matches!(root, NativeStackRoot::SortPreservingMergeFetch)),
    )
}

fn required_child<T: 'static>(plan: &dyn ExecutionPlan, candidate: GpuCandidate) -> Result<&T, NativePlanRejection> {
    plan.as_any()
        .downcast_ref::<T>()
        .ok_or(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
}

fn sort_orderings_match(left: &LexOrdering, right: &LexOrdering) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right.iter()).all(|(left, right)| sort_expr_key(left) == sort_expr_key(right))
}

fn sort_expr_key(sort_expr: &PhysicalSortExpr) -> Option<(&str, bool, bool)> {
    let column = sort_expr.expr.as_any().downcast_ref::<Column>()?;
    Some((column.name(), sort_expr.options.descending, sort_expr.options.nulls_first))
}
