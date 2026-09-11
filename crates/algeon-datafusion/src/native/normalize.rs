use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::{
    datasource::{memory::MemorySourceConfig, source::DataSourceExec},
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties,
        aggregates::AggregateExec,
        coalesce_partitions::CoalescePartitionsExec,
        coop::CooperativeExec,
        empty::EmptyExec,
        filter::FilterExec,
        joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
        limit::{GlobalLimitExec, LocalLimitExec},
        placeholder_row::PlaceholderRowExec,
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        union::UnionExec,
        windows::{BoundedWindowAggExec, WindowAggExec},
    },
};
use query_engine::expr::LimitSpec as NativeLimitSpec;

use crate::{
    capability::CapabilityReason,
    native_exec::{gpu_resident_producer_input_name, gpu_resident_producer_unique_output_key},
    report::GpuCandidate,
    table_format::AlgeonIcebergScanExec,
};

mod distributed_aggregate;
mod shape;
mod types;

use distributed_aggregate::peel_distributed_aggregate_pair_if_present;
use shape::{
    SORT_PRESERVING_MERGE_FETCH_INPUT_SHAPE, SORT_PRESERVING_MERGE_FETCH_SHAPE, SORT_PRESERVING_MERGE_SHAPE,
    SORT_PRESERVING_MERGE_SORT_SHAPE, UNSUPPORTED_NATIVE_PLAN_SHAPE, global_aggregate_candidate,
    global_aggregate_candidate_for_input, input_root_candidate, native_stack_candidate,
    native_stack_candidate_for_input,
};
pub(in crate::native) use types::{
    NativeAggregateShapeKind, NativePlanRejection, NativeStackRoot, NormalizedNativeAggregate,
    NormalizedNativeAggregateRoot, NormalizedNativeCrossJoin, NormalizedNativeFilter,
    NormalizedNativeGpuResidentSource, NormalizedNativeHashJoin, NormalizedNativeInput, NormalizedNativeInputRoot,
    NormalizedNativeLimit, NormalizedNativeLiteralSource, NormalizedNativeNestedLoopJoin, NormalizedNativePlan,
    NormalizedNativeProjection, NormalizedNativeRoot, NormalizedNativeScan, NormalizedNativeSort,
    NormalizedNativeSortExprs, NormalizedNativeSource, NormalizedNativeUnion, NormalizedNativeWindow,
    NormalizedNativeWindowExec, PostAggregateProjectionOrder,
};

pub(super) fn normalize_native_plan(plan: &dyn ExecutionPlan) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    let plan = peel_repartition(plan);

    // Match root shapes from most specific to most general. The ordering keeps
    // diagnostic candidates stable: SPM(fetch) is not reported as a plain sort,
    // projection roots first try aggregate stacks, and bare joins are input
    // roots rather than unsupported unary stacks.
    if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
        return normalize_sort_preserving_merge_root(merge, None);
    }

    if let Some(limit) = plan.downcast_ref::<LocalLimitExec>() {
        let candidate =
            native_stack_candidate(NativeStackRoot::LocalLimit, plan.schema().fields().len(), false, true, true, false);
        let sort = required_child::<SortExec>(limit.input().as_ref(), candidate)?;
        let fetch = sort.fetch().map_or(limit.fetch(), |sort_fetch| sort_fetch.min(limit.fetch()));
        return normalize_sort_stack(
            NativeStackRoot::LocalLimit,
            plan.schema().fields().len(),
            sort,
            Some(NativeLimitSpec::fetch(fetch)),
            None,
        );
    }

    if let Some(limit) = plan.downcast_ref::<GlobalLimitExec>() {
        let limit_spec = global_limit_spec(limit, GpuCandidate::new("GlobalLimitExec", plan.schema().fields().len()))?;
        return normalize_global_limit_root(plan.schema().fields().len(), limit.input().as_ref(), limit_spec);
    }

    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        let root = if sort.fetch().is_some() { NativeStackRoot::SortFetch } else { NativeStackRoot::Sort };
        return normalize_sort_stack(
            root,
            plan.schema().fields().len(),
            sort,
            sort.fetch().map(NativeLimitSpec::fetch),
            None,
        );
    }

    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        if let Some(merge) = projection.input().downcast_ref::<SortPreservingMergeExec>() {
            return normalize_sort_preserving_merge_root(merge, Some(projection));
        }

        if let Some(limit) = projection.input().downcast_ref::<GlobalLimitExec>() {
            let limit_spec = global_limit_spec(
                limit,
                GpuCandidate::with_operator_count(
                    wrap_shape("ProjectionExec", "GlobalLimitExec".to_string()),
                    plan.schema().fields().len(),
                    2,
                ),
            )?;
            let candidate = global_aggregate_candidate(plan.schema().fields().len(), true, false, false, false);
            match required_child::<AggregateExec>(limit.input().as_ref(), candidate.clone()) {
                Ok(aggregate) => {
                    let normalized =
                        normalize_global_aggregate_stack(plan.schema().fields().len(), Some(projection), aggregate)?;
                    return Ok(with_global_limit(normalized, limit_spec));
                },
                Err(rejection) if rejection.reason == CapabilityReason::UnsupportedOperator => {
                    let normalized = normalize_input_root(
                        plan.schema().fields().len(),
                        None,
                        None,
                        None,
                        Some(projection),
                        limit.input().as_ref(),
                        candidate,
                        PostAggregateProjectionOrder::BelowSort,
                    )?;
                    return Ok(with_global_limit(normalized, limit_spec));
                },
                Err(rejection) => return Err(rejection),
            }
        }

        if let Some(sort) = projection.input().downcast_ref::<SortExec>() {
            let root = if sort.fetch().is_some() { NativeStackRoot::SortFetch } else { NativeStackRoot::Sort };
            return normalize_sort_stack(
                root,
                plan.schema().fields().len(),
                sort,
                sort.fetch().map(NativeLimitSpec::fetch),
                Some(projection),
            );
        }

        let candidate = global_aggregate_candidate(plan.schema().fields().len(), true, false, false, false);
        match required_child::<AggregateExec>(projection.input().as_ref(), candidate.clone()) {
            Ok(aggregate) => {
                return normalize_global_aggregate_stack(plan.schema().fields().len(), Some(projection), aggregate);
            },
            Err(rejection) if rejection.reason == CapabilityReason::UnsupportedOperator => {
                return normalize_input_root(
                    plan.schema().fields().len(),
                    None,
                    None,
                    None,
                    Some(projection),
                    projection.input().as_ref(),
                    candidate,
                    PostAggregateProjectionOrder::BelowSort,
                );
            },
            Err(rejection) => return Err(rejection),
        }
    }

    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return normalize_global_aggregate_stack(plan.schema().fields().len(), None, aggregate);
    }

    if plan.downcast_ref::<WindowAggExec>().is_some() || plan.downcast_ref::<BoundedWindowAggExec>().is_some() {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(
            plan.schema().fields().len(),
            None,
            None,
            None,
            None,
            plan,
            candidate,
            PostAggregateProjectionOrder::BelowSort,
        );
    }

    if plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<DataSourceExec>().is_some()
        || plan.downcast_ref::<PlaceholderRowExec>().is_some()
        || plan.downcast_ref::<EmptyExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.downcast_ref::<CooperativeExec>().is_some()
        || gpu_resident_producer_input_name(plan).is_some()
        || plan.downcast_ref::<AlgeonIcebergScanExec>().is_some()
    {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(
            plan.schema().fields().len(),
            None,
            None,
            None,
            None,
            plan,
            candidate,
            PostAggregateProjectionOrder::BelowSort,
        );
    }

    if plan.downcast_ref::<HashJoinExec>().is_some()
        || plan.downcast_ref::<CrossJoinExec>().is_some()
        || plan.downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.downcast_ref::<UnionExec>().is_some()
    {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(
            plan.schema().fields().len(),
            None,
            None,
            None,
            None,
            plan,
            candidate,
            PostAggregateProjectionOrder::BelowSort,
        );
    }

    Err(NativePlanRejection {
        candidate: GpuCandidate::with_operator_count(UNSUPPORTED_NATIVE_PLAN_SHAPE, plan.schema().fields().len(), 1),
        reason: CapabilityReason::UnsupportedOperator,
    })
}

pub(super) fn make_declared_ordered_hash_join_fetch_explicit<'a>(
    normalized: &mut NormalizedNativePlan<'a>,
    ordering: Option<&'a LexOrdering>,
    output_schema: SchemaRef,
) {
    let Some(ordering) = ordering else {
        return;
    };
    if normalized.sort.is_some() || normalized.limit.is_some() {
        return;
    }
    let NormalizedNativeRoot::Input(root) = &mut normalized.root else {
        return;
    };
    let NormalizedNativeInput::HashJoin(join) = &mut root.input else {
        return;
    };
    let Some(fetch) = join.fetch.take() else {
        return;
    };

    normalized.sort =
        Some(NormalizedNativeSortExprs::from_ordering(ordering, Arc::clone(&output_schema), output_schema));
    normalized.limit = Some(fetch);
    normalized.candidate =
        normalized.candidate.clone().with_outer_operator(super::DECLARED_OUTPUT_ORDERING_SORT_OPERATOR);
}

fn normalize_sort_stack<'a>(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &'a SortExec,
    limit: Option<NativeLimitSpec>,
    outer_projection: Option<&'a ProjectionExec>,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let initial_candidate =
        native_stack_candidate(root, output_schema_fields, outer_projection.is_some(), true, true, false);
    if !partition_preserving_sort_is_supported(sort, native_stack_root_is_sort_preserving_merge(root)) {
        return Err(NativePlanRejection {
            candidate: initial_candidate,
            reason: CapabilityReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let (post_aggregate_projection, aggregate, effective_sort) =
        match peel_post_aggregate_projection(sort, initial_candidate.clone(), outer_projection) {
            Ok(stack) => stack,
            Err(rejection) if rejection.reason == CapabilityReason::UnsupportedOperator => {
                return normalize_sort_input_root(
                    root,
                    output_schema_fields,
                    sort,
                    limit,
                    initial_candidate,
                    outer_projection,
                );
            },
            Err(rejection) => return Err(rejection),
        };
    let aggregate_peel = peel_distributed_aggregate_pair_if_present(aggregate, initial_candidate.clone())?;
    let (pre_aggregate_projection, native_input) =
        peel_pre_aggregate_projection_and_input(aggregate_peel.aggregate, initial_candidate.clone())?;
    let candidate = native_stack_candidate_for_input(
        root,
        output_schema_fields,
        &native_input,
        post_aggregate_projection.is_some(),
        pre_aggregate_projection.is_some(),
        aggregate_peel.has_distributed_aggregate,
    );

    let effective_limit = limit.or_else(|| effective_sort.fetch().map(NativeLimitSpec::fetch));

    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Aggregate(NormalizedNativeAggregateRoot {
            input: native_input,
            pre_aggregate_projection,
            aggregate: aggregate_peel.aggregate,
            statistics_output: aggregate,
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            post_aggregate_projection,
            projection_order: if outer_projection.is_some() {
                PostAggregateProjectionOrder::AboveSort
            } else {
                PostAggregateProjectionOrder::BelowSort
            },
        }),
        sort: Some(NormalizedNativeSortExprs::from_sort_exec(effective_sort)),
        limit: effective_limit,
    })
}

fn normalize_sort_preserving_merge_root<'a>(
    merge: &'a SortPreservingMergeExec,
    outer_projection: Option<&'a ProjectionExec>,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let output_schema_fields = merge.schema().fields().len();
    let root_without_child_sort = if merge.fetch().is_some() {
        NativeStackRoot::SortPreservingMergeFetch
    } else {
        NativeStackRoot::SortPreservingMerge
    };
    let root_with_child_sort = if merge.fetch().is_some() {
        NativeStackRoot::SortPreservingMergeFetchSort
    } else {
        NativeStackRoot::SortPreservingMergeSort
    };
    let candidate = sort_preserving_merge_candidate(merge);
    if !sort_ordering_is_column_keyed(merge.expr()) {
        return Err(NativePlanRejection { candidate, reason: CapabilityReason::NativeSortPreservingMergeUnsupported });
    }
    if !plan_partitions_match_ordering(merge.input().as_ref(), merge.expr()) {
        return Err(NativePlanRejection { candidate, reason: CapabilityReason::NativeSortPreservingMergeUnsupported });
    }

    let (inner_projection, after_proj) = peel_root_projection(merge.input().as_ref());
    if let Some(sort) = after_proj.downcast_ref::<SortExec>() {
        if merge.fetch().is_none() && sort.fetch().is_some() {
            return Err(NativePlanRejection {
                candidate,
                reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
            });
        }
        if let Some(fetch) = merge.fetch()
            && sort.fetch().is_some_and(|sort_fetch| sort_fetch < fetch)
        {
            return Err(NativePlanRejection {
                candidate,
                reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
            });
        }
        if outer_projection.is_none() && (sort_orderings_match(merge.expr(), sort.expr()) || inner_projection.is_some())
        {
            return normalize_sort_stack(
                root_with_child_sort,
                output_schema_fields,
                sort,
                merge.fetch().map(NativeLimitSpec::fetch),
                outer_projection.or(inner_projection),
            );
        }
    }

    let input_plan = merge.input().as_ref();
    let sort = NormalizedNativeSortExprs::from_ordering(merge.expr(), input_plan.schema(), merge.schema());
    if let Some(outer_projection) = outer_projection {
        return normalize_input_root(
            outer_projection.schema().fields().len(),
            Some(root_without_child_sort),
            Some(sort),
            merge.fetch().map(NativeLimitSpec::fetch),
            Some(outer_projection),
            input_plan,
            candidate,
            PostAggregateProjectionOrder::AboveSort,
        );
    }
    let (projection, input) = peel_root_projection(input_plan);
    normalize_input_root(
        output_schema_fields,
        Some(root_without_child_sort),
        Some(sort),
        merge.fetch().map(NativeLimitSpec::fetch),
        projection,
        input,
        candidate,
        PostAggregateProjectionOrder::BelowSort,
    )
}

fn sort_preserving_merge_candidate(merge: &SortPreservingMergeExec) -> GpuCandidate {
    let has_child_sort = merge.input().downcast_ref::<SortExec>().is_some()
        || peel_root_projection(merge.input().as_ref()).1.downcast_ref::<SortExec>().is_some();
    if merge.fetch().is_some() {
        if has_child_sort {
            GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_FETCH_SHAPE, merge.schema().fields().len(), 2)
        } else {
            GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_FETCH_INPUT_SHAPE, merge.schema().fields().len(), 1)
        }
    } else if has_child_sort {
        GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_SORT_SHAPE, merge.schema().fields().len(), 2)
    } else {
        GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_SHAPE, merge.schema().fields().len(), 1)
    }
}

fn plan_partitions_match_ordering(plan: &dyn ExecutionPlan, ordering: &LexOrdering) -> bool {
    plan.equivalence_properties().ordering_satisfy(ordering.iter().cloned()).unwrap_or(false)
        || plan.downcast_ref::<UnionExec>().is_some_and(|union| {
            union.inputs().iter().all(|input| plan_partitions_match_ordering(input.as_ref(), ordering))
        })
}

fn normalize_global_aggregate_stack<'a>(
    output_schema_fields: usize,
    post_aggregate_projection: Option<&'a ProjectionExec>,
    aggregate: &'a AggregateExec,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let candidate =
        global_aggregate_candidate(output_schema_fields, post_aggregate_projection.is_some(), false, false, false);
    let aggregate_peel = peel_distributed_aggregate_pair_if_present(aggregate, candidate.clone())?;
    let (pre_aggregate_projection, native_input) =
        peel_pre_aggregate_projection_and_input(aggregate_peel.aggregate, candidate.clone())?;
    let candidate = global_aggregate_candidate_for_input(
        output_schema_fields,
        &native_input,
        post_aggregate_projection.is_some(),
        pre_aggregate_projection.is_some(),
        aggregate_peel.has_distributed_aggregate,
    );

    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Aggregate(NormalizedNativeAggregateRoot {
            input: native_input,
            pre_aggregate_projection,
            aggregate: aggregate_peel.aggregate,
            statistics_output: aggregate,
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            post_aggregate_projection,
            projection_order: PostAggregateProjectionOrder::BelowSort,
        }),
        sort: None,
        limit: None,
    })
}

fn normalize_global_limit_root(
    output_schema_fields: usize,
    input: &dyn ExecutionPlan,
    limit: NativeLimitSpec,
) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    if let Some(projection) = input.downcast_ref::<ProjectionExec>() {
        let candidate = global_aggregate_candidate(output_schema_fields, true, false, false, false);
        match required_child::<AggregateExec>(projection.input().as_ref(), candidate.clone()) {
            Ok(aggregate) => {
                let normalized = normalize_global_aggregate_stack(output_schema_fields, Some(projection), aggregate)?;
                return Ok(with_global_limit(normalized, limit));
            },
            Err(rejection) if rejection.reason == CapabilityReason::UnsupportedOperator => {
                let normalized = normalize_input_root(
                    output_schema_fields,
                    None,
                    None,
                    None,
                    Some(projection),
                    projection.input().as_ref(),
                    candidate,
                    PostAggregateProjectionOrder::BelowSort,
                )?;
                return Ok(with_global_limit(normalized, limit));
            },
            Err(rejection) => return Err(rejection),
        }
    }

    if let Some(aggregate) = input.downcast_ref::<AggregateExec>() {
        let normalized = normalize_global_aggregate_stack(output_schema_fields, None, aggregate)?;
        return Ok(with_global_limit(normalized, limit));
    }

    let candidate = input_root_candidate(None, output_schema_fields, input.name().to_string(), false, 1);
    let normalized = normalize_input_root(
        output_schema_fields,
        None,
        None,
        None,
        None,
        input,
        candidate,
        PostAggregateProjectionOrder::BelowSort,
    )?;
    Ok(with_global_limit(normalized, limit))
}

fn with_global_limit(mut normalized: NormalizedNativePlan<'_>, limit: NativeLimitSpec) -> NormalizedNativePlan<'_> {
    normalized.limit = Some(normalized.limit.map_or(limit, |existing| compose_limits(existing, limit)));
    normalized.candidate = GpuCandidate::with_operator_count(
        wrap_shape("GlobalLimitExec", normalized.candidate.shape),
        normalized.candidate.output_schema_fields,
        normalized.candidate.operator_count + 1,
    );
    normalized
}

fn compose_limits(inner: NativeLimitSpec, outer: NativeLimitSpec) -> NativeLimitSpec {
    let skip = inner.skip.saturating_add(outer.skip);
    let fetch = match (inner.fetch, outer.fetch) {
        (Some(inner_fetch), Some(outer_fetch)) => Some(inner_fetch.saturating_sub(outer.skip).min(outer_fetch)),
        (Some(inner_fetch), None) => Some(inner_fetch.saturating_sub(outer.skip)),
        (None, Some(outer_fetch)) => Some(outer_fetch),
        (None, None) => None,
    };
    NativeLimitSpec::new(skip, fetch)
}

fn global_limit_spec(
    limit: &GlobalLimitExec,
    _candidate: GpuCandidate,
) -> Result<NativeLimitSpec, NativePlanRejection> {
    Ok(NativeLimitSpec::new(limit.skip(), limit.fetch()))
}

fn normalize_sort_input_root<'a>(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &'a SortExec,
    limit: Option<NativeLimitSpec>,
    candidate: GpuCandidate,
    outer_projection: Option<&'a ProjectionExec>,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    if !partition_preserving_sort_is_supported(sort, native_stack_root_is_sort_preserving_merge(root)) {
        return Err(NativePlanRejection {
            candidate,
            reason: CapabilityReason::NativePartitionPreservingSortUnsupported,
        });
    }
    if let Some(outer) = outer_projection {
        return normalize_input_root(
            output_schema_fields,
            Some(root),
            Some(NormalizedNativeSortExprs::from_sort_exec(sort)),
            limit.or_else(|| sort.fetch().map(NativeLimitSpec::fetch)),
            Some(outer),
            sort.input().as_ref(),
            candidate,
            PostAggregateProjectionOrder::AboveSort,
        );
    }
    let (projection, input) = peel_root_projection(sort.input().as_ref());
    normalize_input_root(
        output_schema_fields,
        Some(root),
        Some(NormalizedNativeSortExprs::from_sort_exec(sort)),
        limit.or_else(|| sort.fetch().map(NativeLimitSpec::fetch)),
        projection,
        input,
        candidate,
        PostAggregateProjectionOrder::BelowSort,
    )
}

#[expect(clippy::too_many_arguments, reason = "inputs are distinct domain values that share no owner")]
fn normalize_input_root<'a>(
    output_schema_fields: usize,
    stack_root: Option<NativeStackRoot>,
    sort: Option<NormalizedNativeSortExprs<'a>>,
    limit: Option<NativeLimitSpec>,
    projection: Option<&'a ProjectionExec>,
    input_plan: &'a dyn ExecutionPlan,
    candidate: GpuCandidate,
    projection_order: PostAggregateProjectionOrder,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let input = normalize_native_input(input_plan, candidate.clone(), sort.is_some())?;
    let candidate = input_root_candidate(
        stack_root,
        output_schema_fields,
        input.shape(),
        projection.is_some(),
        input.operator_count(),
    );
    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Input(NormalizedNativeInputRoot { input, projection, projection_order }),
        sort,
        limit,
    })
}

fn peel_post_aggregate_projection<'a>(
    sort: &'a SortExec,
    candidate: GpuCandidate,
    outer_projection: Option<&'a ProjectionExec>,
) -> Result<(Option<&'a ProjectionExec>, &'a AggregateExec, &'a SortExec), NativePlanRejection> {
    if let Some(outer) = outer_projection {
        if let Some(aggregate) = sort.input().downcast_ref::<AggregateExec>() {
            return Ok((Some(outer), aggregate, sort));
        }
        return Err(NativePlanRejection { candidate, reason: CapabilityReason::UnsupportedOperator });
    }

    if let Some(projection) = sort.input().downcast_ref::<ProjectionExec>() {
        if let Some(aggregate) = projection.input().downcast_ref::<AggregateExec>() {
            return Ok((Some(projection), aggregate, sort));
        }
        return Err(NativePlanRejection { candidate, reason: CapabilityReason::UnsupportedOperator });
    }

    Ok((None, required_child::<AggregateExec>(sort.input().as_ref(), candidate)?, sort))
}

fn peel_root_projection(plan: &dyn ExecutionPlan) -> (Option<&ProjectionExec>, &dyn ExecutionPlan) {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        (Some(projection), projection.input().as_ref())
    } else {
        (None, plan)
    }
}

fn peel_pre_aggregate_projection(aggregate: &AggregateExec) -> (Option<&ProjectionExec>, &dyn ExecutionPlan) {
    if let Some(projection) = aggregate.input().downcast_ref::<ProjectionExec>() {
        (Some(projection), projection.input().as_ref())
    } else {
        (None, aggregate.input().as_ref())
    }
}

fn peel_pre_aggregate_projection_and_input(
    aggregate: &AggregateExec,
    candidate: GpuCandidate,
) -> Result<(Option<&ProjectionExec>, NormalizedNativeInput<'_>), NativePlanRejection> {
    let (projection, input) = peel_pre_aggregate_projection(aggregate);
    Ok((projection, normalize_native_input(peel_repartition(input), candidate, false)?))
}

fn native_stack_root_is_sort_preserving_merge(root: NativeStackRoot) -> bool {
    matches!(
        root,
        NativeStackRoot::SortPreservingMerge
            | NativeStackRoot::SortPreservingMergeFetch
            | NativeStackRoot::SortPreservingMergeSort
            | NativeStackRoot::SortPreservingMergeFetchSort
    )
}

fn partition_preserving_sort_is_supported(sort: &SortExec, allow_multi_partition: bool) -> bool {
    !sort.preserve_partitioning()
        || allow_multi_partition
        || sort.input().properties().output_partitioning().partition_count() == 1
}

fn normalize_native_input(
    plan: &dyn ExecutionPlan,
    candidate: GpuCandidate,
    allow_ordered_multi_partition_memory_source: bool,
) -> Result<NormalizedNativeInput<'_>, NativePlanRejection> {
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
        && let Some(fetch) = coalesce.fetch()
    {
        let input = normalize_native_input(coalesce.input().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::Limit(NormalizedNativeLimit {
            spec: NativeLimitSpec::fetch(fetch),
            input: Box::new(input),
        }));
    }

    let plan = peel_repartition(plan);
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        let input = peel_repartition(filter.input().as_ref());
        let child_allows_ordered_multi_partition_memory_source =
            filter.fetch().is_none() && allow_ordered_multi_partition_memory_source;
        let normalized = if let Some(source_exec) = input.downcast_ref::<DataSourceExec>()
            && !data_source_is_memory(source_exec)
        {
            NormalizedNativeInput::Source(NormalizedNativeSource {
                scan: NormalizedNativeScan::LocalParquet(source_exec),
                filter: Some(filter),
            })
        } else if let Some(scan_exec) = input.downcast_ref::<AlgeonIcebergScanExec>() {
            if let Some(reason) = scan_exec.native_outcome().rejection_reason() {
                return Err(NativePlanRejection { candidate, reason });
            }
            NormalizedNativeInput::Source(NormalizedNativeSource {
                scan: NormalizedNativeScan::TableFormat(scan_exec),
                filter: Some(filter),
            })
        } else {
            let input = normalize_native_input(input, candidate, child_allows_ordered_multi_partition_memory_source)?;
            NormalizedNativeInput::Filter(NormalizedNativeFilter { filter, input: Box::new(input) })
        };
        if let Some(fetch) = filter.fetch() {
            return Ok(NormalizedNativeInput::Limit(NormalizedNativeLimit {
                spec: NativeLimitSpec::fetch(fetch),
                input: Box::new(normalized),
            }));
        }
        return Ok(normalized);
    }
    if let Some(source_exec) = plan.downcast_ref::<DataSourceExec>() {
        if data_source_is_memory(source_exec) {
            if !allow_ordered_multi_partition_memory_source
                && ordered_multi_partition_memory_source_is_unsupported(source_exec)
            {
                return Err(NativePlanRejection {
                    candidate,
                    reason: CapabilityReason::NativeOrderedOutputUnsupported,
                });
            }
            return Ok(NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::Memory(source_exec)));
        }
        return Ok(NormalizedNativeInput::Source(NormalizedNativeSource {
            scan: NormalizedNativeScan::LocalParquet(source_exec),
            filter: None,
        }));
    }
    if let Some(placeholder) = plan.downcast_ref::<PlaceholderRowExec>() {
        return Ok(NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::PlaceholderRow(placeholder)));
    }
    if let Some(empty) = plan.downcast_ref::<EmptyExec>() {
        return Ok(NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::Empty(empty)));
    }
    if let Some(scan_exec) = plan.downcast_ref::<AlgeonIcebergScanExec>() {
        if let Some(reason) = scan_exec.native_outcome().rejection_reason() {
            return Err(NativePlanRejection { candidate, reason });
        }
        return Ok(NormalizedNativeInput::Source(NormalizedNativeSource {
            scan: NormalizedNativeScan::TableFormat(scan_exec),
            filter: None,
        }));
    }
    if let Some(input_name) = gpu_resident_producer_input_name(plan) {
        return Ok(NormalizedNativeInput::GpuResidentSource(NormalizedNativeGpuResidentSource {
            source_exec: plan,
            input_name,
            row_count_upper_bound: crate::native_exec::gpu_resident_producer_hard_output_row_upper_bound(plan),
            unique_output_key: gpu_resident_producer_unique_output_key(plan),
        }));
    }
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let input = normalize_native_input(
            projection.input().as_ref(),
            candidate,
            allow_ordered_multi_partition_memory_source,
        )?;
        return Ok(NormalizedNativeInput::Projection(NormalizedNativeProjection {
            projection,
            input: Box::new(input),
        }));
    }
    if let Some(limit) = plan.downcast_ref::<GlobalLimitExec>() {
        let spec = global_limit_spec(limit, candidate.clone())?;
        let input = normalize_native_input(limit.input().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::Limit(NormalizedNativeLimit { spec, input: Box::new(input) }));
    }
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        let aggregate_peel = peel_distributed_aggregate_pair_if_present(aggregate, candidate.clone())?;
        let input = normalize_native_input(aggregate_peel.aggregate.input().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::Aggregate(NormalizedNativeAggregate {
            aggregate: aggregate_peel.aggregate,
            statistics_output: aggregate,
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            input: Box::new(input),
        }));
    }
    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        if !partition_preserving_sort_is_supported(sort, allow_ordered_multi_partition_memory_source) {
            return Err(NativePlanRejection {
                candidate,
                reason: CapabilityReason::NativePartitionPreservingSortUnsupported,
            });
        }
        let input = normalize_native_input(sort.input().as_ref(), candidate, true)?;
        return Ok(NormalizedNativeInput::Sort(NormalizedNativeSort {
            sort: NormalizedNativeSortExprs::from_sort_exec(sort),
            fetch: sort.fetch().map(NativeLimitSpec::fetch),
            input: Box::new(input),
        }));
    }
    if let Some(window) = plan.downcast_ref::<WindowAggExec>() {
        let input = normalize_native_input(window.input().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::Window(NormalizedNativeWindow {
            window: NormalizedNativeWindowExec::Unbounded(window),
            input: Box::new(input),
        }));
    }
    if let Some(window) = plan.downcast_ref::<BoundedWindowAggExec>() {
        let input = normalize_native_input(window.input().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::Window(NormalizedNativeWindow {
            window: NormalizedNativeWindowExec::Bounded(window),
            input: Box::new(input),
        }));
    }
    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        let left = normalize_native_input(
            join.left().as_ref(),
            candidate.clone(),
            allow_ordered_multi_partition_memory_source,
        )?;
        let right =
            normalize_native_input(join.right().as_ref(), candidate, allow_ordered_multi_partition_memory_source)?;
        return Ok(NormalizedNativeInput::HashJoin(NormalizedNativeHashJoin {
            join,
            fetch: join.fetch().map(NativeLimitSpec::fetch),
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(join) = plan.downcast_ref::<CrossJoinExec>() {
        let left = normalize_native_input(
            join.left().as_ref(),
            candidate.clone(),
            allow_ordered_multi_partition_memory_source,
        )?;
        let right =
            normalize_native_input(join.right().as_ref(), candidate, allow_ordered_multi_partition_memory_source)?;
        return Ok(NormalizedNativeInput::CrossJoin(NormalizedNativeCrossJoin {
            join,
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(join) = plan.downcast_ref::<NestedLoopJoinExec>() {
        let left = normalize_native_input(
            join.left().as_ref(),
            candidate.clone(),
            allow_ordered_multi_partition_memory_source,
        )?;
        let right =
            normalize_native_input(join.right().as_ref(), candidate, allow_ordered_multi_partition_memory_source)?;
        return Ok(NormalizedNativeInput::NestedLoopJoin(NormalizedNativeNestedLoopJoin {
            join,
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(union) = plan.downcast_ref::<UnionExec>() {
        let inputs = union
            .inputs()
            .iter()
            .map(|input| {
                normalize_native_input(input.as_ref(), candidate.clone(), allow_ordered_multi_partition_memory_source)
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(NormalizedNativeInput::Union(NormalizedNativeUnion { union, inputs }));
    }
    Err(NativePlanRejection { candidate, reason: CapabilityReason::UnsupportedOperator })
}

fn peel_repartition(plan: &dyn ExecutionPlan) -> &dyn ExecutionPlan {
    // Native shapes treat repartition and coalesce-partitions as exchange
    // boundaries that can be peeled while preserving the child semantics.
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
        repartition.input().as_ref()
    } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
        if coalesce.fetch().is_some() { plan } else { coalesce.input().as_ref() }
    } else if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        cooperative.input().as_ref()
    } else {
        plan
    }
}

fn data_source_is_memory(source_exec: &DataSourceExec) -> bool {
    source_exec.data_source().downcast_ref::<MemorySourceConfig>().is_some()
}

fn ordered_multi_partition_memory_source_is_unsupported(source_exec: &DataSourceExec) -> bool {
    source_exec
        .data_source()
        .downcast_ref::<MemorySourceConfig>()
        .is_some_and(|memory| source_exec.properties().output_ordering().is_some() && memory.partitions().len() > 1)
}

fn wrap_shape(operator: &str, input: String) -> String {
    format!("{operator}({input})")
}

fn required_child<T: ExecutionPlan>(
    plan: &dyn ExecutionPlan,
    candidate: GpuCandidate,
) -> Result<&T, NativePlanRejection> {
    plan.downcast_ref::<T>().ok_or(NativePlanRejection { candidate, reason: CapabilityReason::UnsupportedOperator })
}

fn sort_orderings_match(left: &LexOrdering, right: &LexOrdering) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right.iter()).all(|(left, right)| sort_expr_key(left) == sort_expr_key(right))
}

fn sort_ordering_is_column_keyed(ordering: &LexOrdering) -> bool {
    !ordering.is_empty() && ordering.iter().all(|sort_expr| sort_expr.expr.downcast_ref::<Column>().is_some())
}

fn sort_expr_key(sort_expr: &PhysicalSortExpr) -> Option<(usize, &str, bool, bool)> {
    let column = sort_expr.expr.downcast_ref::<Column>()?;
    Some((column.index(), column.name(), sort_expr.options.descending, sort_expr.options.nulls_first))
}
