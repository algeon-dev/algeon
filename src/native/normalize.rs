use datafusion::{
    datasource::source::DataSourceExec,
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use crate::{policy::FallbackReason, report::GpuCandidate};

mod distributed_aggregate;
mod shape;
mod types;

use distributed_aggregate::peel_distributed_aggregate_pair_if_present;
use shape::*;
pub(in crate::native) use types::*;

pub(super) fn normalize_native_plan(plan: &dyn ExecutionPlan) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    // Match root shapes from most specific to most general. The ordering keeps
    // diagnostic candidates stable: SPM(fetch) is not reported as a plain sort,
    // projection roots first try aggregate stacks, and bare joins are input
    // roots rather than unsupported unary stacks.
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
        let sort = merge.input().as_any().downcast_ref::<SortExec>().ok_or(NativePlanRejection {
            candidate: candidate.clone(),
            reason: FallbackReason::NativeSortPreservingMergeUnsupported,
        })?;
        if !sort_orderings_match(merge.expr(), sort.expr()) {
            return Err(NativePlanRejection {
                candidate: candidate.clone(),
                reason: FallbackReason::NativeSortPreservingMergeUnsupported,
            });
        }
        if sort.fetch().is_some_and(|sort_fetch| sort_fetch < fetch) {
            return Err(NativePlanRejection {
                candidate: candidate.clone(),
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
        let candidate =
            native_stack_candidate(NativeStackRoot::LocalLimit, plan.schema().fields().len(), false, true, true, false);
        let sort = required_child::<SortExec>(limit.input().as_ref(), candidate)?;
        let fetch = sort.fetch().map_or(limit.fetch(), |sort_fetch| sort_fetch.min(limit.fetch()));
        return normalize_sort_stack(NativeStackRoot::LocalLimit, plan.schema().fields().len(), sort, Some(fetch));
    }

    if let Some(limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
        let limit_fetch =
            global_limit_fetch(limit, GpuCandidate::new("GlobalLimitExec", plan.schema().fields().len()))?;
        return normalize_global_limit_root(plan.schema().fields().len(), limit.input().as_ref(), limit_fetch);
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        let root = if sort.fetch().is_some() { NativeStackRoot::SortFetch } else { NativeStackRoot::Sort };
        return normalize_sort_stack(root, plan.schema().fields().len(), sort, sort.fetch());
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        if let Some(limit) = projection.input().as_any().downcast_ref::<GlobalLimitExec>() {
            let limit_fetch = global_limit_fetch(
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
                    return Ok(with_global_limit_fetch(normalized, limit_fetch));
                },
                Err(rejection) if rejection.reason == FallbackReason::UnsupportedOperator => {
                    let normalized = normalize_input_root(
                        plan.schema().fields().len(),
                        None,
                        None,
                        None,
                        Some(projection),
                        limit.input().as_ref(),
                        candidate,
                    )?;
                    return Ok(with_global_limit_fetch(normalized, limit_fetch));
                },
                Err(rejection) => return Err(rejection),
            }
        }

        let candidate = global_aggregate_candidate(plan.schema().fields().len(), true, false, false, false);
        match required_child::<AggregateExec>(projection.input().as_ref(), candidate.clone()) {
            Ok(aggregate) => {
                return normalize_global_aggregate_stack(plan.schema().fields().len(), Some(projection), aggregate);
            },
            Err(rejection) if rejection.reason == FallbackReason::UnsupportedOperator => {
                return normalize_input_root(
                    plan.schema().fields().len(),
                    None,
                    None,
                    None,
                    Some(projection),
                    projection.input().as_ref(),
                    candidate,
                );
            },
            Err(rejection) => return Err(rejection),
        }
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        return normalize_global_aggregate_stack(plan.schema().fields().len(), None, aggregate);
    }

    if plan.as_any().downcast_ref::<HashJoinExec>().is_some()
        || plan.as_any().downcast_ref::<CrossJoinExec>().is_some()
        || plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some()
    {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(plan.schema().fields().len(), None, None, None, None, plan, candidate);
    }

    Err(NativePlanRejection {
        candidate: GpuCandidate::with_operator_count(UNSUPPORTED_NATIVE_PLAN_SHAPE, plan.schema().fields().len(), 1),
        reason: FallbackReason::UnsupportedOperator,
    })
}

fn normalize_sort_stack(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &SortExec,
    limit_fetch: Option<usize>,
) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    let initial_candidate = native_stack_candidate(root, output_schema_fields, false, true, true, false);
    if sort.preserve_partitioning() && !matches!(root, NativeStackRoot::SortPreservingMergeFetch) {
        return Err(NativePlanRejection {
            candidate: initial_candidate,
            reason: FallbackReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let (post_aggregate_projection, aggregate) = match peel_post_aggregate_projection(sort, initial_candidate.clone()) {
        Ok(stack) => stack,
        Err(rejection) if rejection.reason == FallbackReason::UnsupportedOperator => {
            return normalize_sort_input_root(root, output_schema_fields, sort, limit_fetch, initial_candidate);
        },
        Err(rejection) => return Err(rejection),
    };
    let (aggregate, aggregate_allows_partial, has_distributed_aggregate, aggregate_output_schema) =
        peel_distributed_aggregate_pair_if_present(aggregate, initial_candidate.clone())?;
    let (pre_aggregate_projection, native_input) =
        peel_pre_aggregate_projection_and_input(aggregate, initial_candidate.clone())?;
    let candidate = native_stack_candidate_for_input(
        root,
        output_schema_fields,
        &native_input,
        post_aggregate_projection.is_some(),
        pre_aggregate_projection.is_some(),
        has_distributed_aggregate,
    );

    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Aggregate(NormalizedNativeAggregateRoot {
            input: native_input,
            pre_aggregate_projection,
            aggregate,
            aggregate_allows_partial,
            aggregate_output_schema,
            post_aggregate_projection,
        }),
        sort: Some(sort),
        limit_fetch,
    })
}

fn normalize_global_aggregate_stack<'a>(
    output_schema_fields: usize,
    post_aggregate_projection: Option<&'a ProjectionExec>,
    aggregate: &'a AggregateExec,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let candidate =
        global_aggregate_candidate(output_schema_fields, post_aggregate_projection.is_some(), false, false, false);
    let (aggregate, aggregate_allows_partial, has_distributed_aggregate, aggregate_output_schema) =
        peel_distributed_aggregate_pair_if_present(aggregate, candidate.clone())?;
    let (pre_aggregate_projection, native_input) =
        peel_pre_aggregate_projection_and_input(aggregate, candidate.clone())?;
    let candidate = global_aggregate_candidate_for_input(
        output_schema_fields,
        &native_input,
        post_aggregate_projection.is_some(),
        pre_aggregate_projection.is_some(),
        has_distributed_aggregate,
    );

    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Aggregate(NormalizedNativeAggregateRoot {
            input: native_input,
            pre_aggregate_projection,
            aggregate,
            aggregate_allows_partial,
            aggregate_output_schema,
            post_aggregate_projection,
        }),
        sort: None,
        limit_fetch: None,
    })
}

fn normalize_global_limit_root<'a>(
    output_schema_fields: usize,
    input: &'a dyn ExecutionPlan,
    limit_fetch: usize,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    if let Some(projection) = input.as_any().downcast_ref::<ProjectionExec>() {
        let candidate = global_aggregate_candidate(output_schema_fields, true, false, false, false);
        match required_child::<AggregateExec>(projection.input().as_ref(), candidate.clone()) {
            Ok(aggregate) => {
                let normalized = normalize_global_aggregate_stack(output_schema_fields, Some(projection), aggregate)?;
                return Ok(with_global_limit_fetch(normalized, limit_fetch));
            },
            Err(rejection) if rejection.reason == FallbackReason::UnsupportedOperator => {
                let normalized = normalize_input_root(
                    output_schema_fields,
                    None,
                    None,
                    None,
                    Some(projection),
                    projection.input().as_ref(),
                    candidate,
                )?;
                return Ok(with_global_limit_fetch(normalized, limit_fetch));
            },
            Err(rejection) => return Err(rejection),
        }
    }

    if let Some(aggregate) = input.as_any().downcast_ref::<AggregateExec>() {
        let normalized = normalize_global_aggregate_stack(output_schema_fields, None, aggregate)?;
        return Ok(with_global_limit_fetch(normalized, limit_fetch));
    }

    let candidate = input_root_candidate(None, output_schema_fields, input.name().to_string(), false, 1);
    let normalized = normalize_input_root(output_schema_fields, None, None, None, None, input, candidate)?;
    Ok(with_global_limit_fetch(normalized, limit_fetch))
}

fn with_global_limit_fetch(mut normalized: NormalizedNativePlan<'_>, fetch: usize) -> NormalizedNativePlan<'_> {
    normalized.limit_fetch = Some(normalized.limit_fetch.map_or(fetch, |existing| existing.min(fetch)));
    normalized.candidate = GpuCandidate::with_operator_count(
        wrap_shape("GlobalLimitExec", normalized.candidate.shape),
        normalized.candidate.output_schema_fields,
        normalized.candidate.operator_count + 1,
    );
    normalized
}

fn global_limit_fetch(limit: &GlobalLimitExec, candidate: GpuCandidate) -> Result<usize, NativePlanRejection> {
    if limit.skip() != 0 {
        return Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator });
    }
    limit.fetch().ok_or(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
}

fn normalize_sort_input_root(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &SortExec,
    limit_fetch: Option<usize>,
    candidate: GpuCandidate,
) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    if sort.preserve_partitioning() && !matches!(root, NativeStackRoot::SortPreservingMergeFetch) {
        return Err(NativePlanRejection {
            candidate,
            reason: FallbackReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let (projection, input) = peel_root_projection(sort.input().as_ref());
    normalize_input_root(output_schema_fields, Some(root), Some(sort), limit_fetch, projection, input, candidate)
}

fn normalize_input_root<'a>(
    output_schema_fields: usize,
    stack_root: Option<NativeStackRoot>,
    sort: Option<&'a SortExec>,
    limit_fetch: Option<usize>,
    projection: Option<&'a ProjectionExec>,
    input_plan: &'a dyn ExecutionPlan,
    candidate: GpuCandidate,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let input = normalize_native_input(input_plan, candidate.clone())?;
    if matches!(input, NormalizedNativeInput::Source(_)) && projection.is_none() {
        return Err(NativePlanRejection {
            candidate: GpuCandidate::with_operator_count(UNSUPPORTED_NATIVE_PLAN_SHAPE, output_schema_fields, 1),
            reason: FallbackReason::UnsupportedOperator,
        });
    }
    let candidate = input_root_candidate(
        stack_root,
        output_schema_fields,
        input.shape(),
        projection.is_some(),
        input.operator_count(),
    );
    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Input(NormalizedNativeInputRoot { input, projection }),
        sort,
        limit_fetch,
    })
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

fn peel_root_projection(plan: &dyn ExecutionPlan) -> (Option<&ProjectionExec>, &dyn ExecutionPlan) {
    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        (Some(projection), projection.input().as_ref())
    } else {
        (None, plan)
    }
}

fn peel_pre_aggregate_projection(aggregate: &AggregateExec) -> (Option<&ProjectionExec>, &dyn ExecutionPlan) {
    if let Some(projection) = aggregate.input().as_any().downcast_ref::<ProjectionExec>() {
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
    Ok((projection, normalize_native_input(peel_repartition(input), candidate)?))
}

fn normalize_native_input<'a>(
    plan: &'a dyn ExecutionPlan,
    candidate: GpuCandidate,
) -> Result<NormalizedNativeInput<'a>, NativePlanRejection> {
    let plan = peel_repartition(plan);
    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        let input = peel_repartition(filter.input().as_ref());
        if let Some(source_exec) = input.as_any().downcast_ref::<DataSourceExec>() {
            return Ok(NormalizedNativeInput::Source(NormalizedNativeSource { source_exec, filter: Some(filter) }));
        }
        let input = normalize_native_input(input, candidate)?;
        return Ok(NormalizedNativeInput::Filter(NormalizedNativeFilter { filter, input: Box::new(input) }));
    }
    if let Some(source_exec) = plan.as_any().downcast_ref::<DataSourceExec>() {
        return Ok(NormalizedNativeInput::Source(NormalizedNativeSource { source_exec, filter: None }));
    }
    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        let input = normalize_native_input(projection.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Projection(NormalizedNativeProjection {
            projection,
            input: Box::new(input),
        }));
    }
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        let (aggregate, aggregate_allows_partial, _has_distributed_aggregate, aggregate_output_schema) =
            peel_distributed_aggregate_pair_if_present(aggregate, candidate.clone())?;
        let input = normalize_native_input(aggregate.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Aggregate(NormalizedNativeAggregate {
            aggregate,
            aggregate_allows_partial,
            aggregate_output_schema,
            input: Box::new(input),
        }));
    }
    if let Some(join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        let left = normalize_native_input(join.left().as_ref(), candidate.clone())?;
        let right = normalize_native_input(join.right().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::HashJoin(NormalizedNativeHashJoin {
            join,
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(join) = plan.as_any().downcast_ref::<CrossJoinExec>() {
        let left = normalize_native_input(join.left().as_ref(), candidate.clone())?;
        let right = normalize_native_input(join.right().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::CrossJoin(NormalizedNativeCrossJoin {
            join,
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
        let left = normalize_native_input(join.left().as_ref(), candidate.clone())?;
        let right = normalize_native_input(join.right().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::NestedLoopJoin(NormalizedNativeNestedLoopJoin {
            join,
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
}

fn peel_repartition(plan: &dyn ExecutionPlan) -> &dyn ExecutionPlan {
    // Native shapes treat repartition and coalesce-partitions as exchange
    // boundaries that can be peeled while preserving the child semantics.
    if let Some(repartition) = plan.as_any().downcast_ref::<RepartitionExec>() {
        repartition.input().as_ref()
    } else if let Some(coalesce) = plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
        coalesce.input().as_ref()
    } else {
        plan
    }
}

fn wrap_shape(operator: &str, input: String) -> String {
    format!("{operator}({input})")
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
