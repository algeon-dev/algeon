use cudf_query_engine::LimitSpec as NativeLimitSpec;
use datafusion::{
    datasource::source::DataSourceExec,
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        coalesce_partitions::CoalescePartitionsExec,
        coop::CooperativeExec,
        filter::FilterExec,
        joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        union::UnionExec,
        windows::{BoundedWindowAggExec, WindowAggExec},
    },
};

use crate::{
    native_exec::gpu_resident_producer_input_name, policy::FallbackReason, report::GpuCandidate,
    table_format::CudfTableFormatScanExec,
};

mod distributed_aggregate;
mod shape;
mod types;

use distributed_aggregate::peel_distributed_aggregate_pair_if_present;
use shape::*;
pub(in crate::native) use types::*;

pub(super) fn normalize_native_plan(plan: &dyn ExecutionPlan) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    let plan = peel_repartition(plan);

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
            Some(NativeLimitSpec::fetch(fetch)),
        );
    }

    if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
        let candidate =
            native_stack_candidate(NativeStackRoot::LocalLimit, plan.schema().fields().len(), false, true, true, false);
        let sort = required_child::<SortExec>(limit.input().as_ref(), candidate)?;
        let fetch = sort.fetch().map_or(limit.fetch(), |sort_fetch| sort_fetch.min(limit.fetch()));
        return normalize_sort_stack(
            NativeStackRoot::LocalLimit,
            plan.schema().fields().len(),
            sort,
            Some(NativeLimitSpec::fetch(fetch)),
        );
    }

    if let Some(limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
        let limit_spec = global_limit_spec(limit, GpuCandidate::new("GlobalLimitExec", plan.schema().fields().len()))?;
        return normalize_global_limit_root(plan.schema().fields().len(), limit.input().as_ref(), limit_spec);
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        let root = if sort.fetch().is_some() { NativeStackRoot::SortFetch } else { NativeStackRoot::Sort };
        return normalize_sort_stack(
            root,
            plan.schema().fields().len(),
            sort,
            sort.fetch().map(NativeLimitSpec::fetch),
        );
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        if let Some(limit) = projection.input().as_any().downcast_ref::<GlobalLimitExec>() {
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
                    return Ok(with_global_limit(normalized, limit_spec));
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
                let sort_fetch = projection
                    .input()
                    .as_any()
                    .downcast_ref::<SortExec>()
                    .and_then(SortExec::fetch)
                    .map(NativeLimitSpec::fetch);
                return normalize_input_root(
                    plan.schema().fields().len(),
                    None,
                    None,
                    sort_fetch,
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

    if plan.as_any().downcast_ref::<WindowAggExec>().is_some()
        || plan.as_any().downcast_ref::<BoundedWindowAggExec>().is_some()
    {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(plan.schema().fields().len(), None, None, None, None, plan, candidate);
    }

    if plan.as_any().downcast_ref::<FilterExec>().is_some()
        || plan.as_any().downcast_ref::<DataSourceExec>().is_some()
        || plan.as_any().downcast_ref::<CoalescePartitionsExec>().is_some()
        || gpu_resident_producer_input_name(plan).is_some()
        || plan.as_any().downcast_ref::<CudfTableFormatScanExec>().is_some()
    {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(plan.schema().fields().len(), None, None, None, None, plan, candidate);
    }

    if plan.as_any().downcast_ref::<HashJoinExec>().is_some()
        || plan.as_any().downcast_ref::<CrossJoinExec>().is_some()
        || plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.as_any().downcast_ref::<UnionExec>().is_some()
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
    limit: Option<NativeLimitSpec>,
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
            return normalize_sort_input_root(root, output_schema_fields, sort, limit, initial_candidate);
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

    Ok(NormalizedNativePlan {
        candidate,
        root: NormalizedNativeRoot::Aggregate(NormalizedNativeAggregateRoot {
            input: native_input,
            pre_aggregate_projection,
            aggregate: aggregate_peel.aggregate,
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            post_aggregate_projection,
        }),
        sort: Some(sort),
        limit,
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
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            post_aggregate_projection,
        }),
        sort: None,
        limit: None,
    })
}

fn normalize_global_limit_root<'a>(
    output_schema_fields: usize,
    input: &'a dyn ExecutionPlan,
    limit: NativeLimitSpec,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    if let Some(projection) = input.as_any().downcast_ref::<ProjectionExec>() {
        let candidate = global_aggregate_candidate(output_schema_fields, true, false, false, false);
        match required_child::<AggregateExec>(projection.input().as_ref(), candidate.clone()) {
            Ok(aggregate) => {
                let normalized = normalize_global_aggregate_stack(output_schema_fields, Some(projection), aggregate)?;
                return Ok(with_global_limit(normalized, limit));
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
                return Ok(with_global_limit(normalized, limit));
            },
            Err(rejection) => return Err(rejection),
        }
    }

    if let Some(aggregate) = input.as_any().downcast_ref::<AggregateExec>() {
        let normalized = normalize_global_aggregate_stack(output_schema_fields, None, aggregate)?;
        return Ok(with_global_limit(normalized, limit));
    }

    let candidate = input_root_candidate(None, output_schema_fields, input.name().to_string(), false, 1);
    let normalized = normalize_input_root(output_schema_fields, None, None, None, None, input, candidate)?;
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
    NativeLimitSpec::new(inner.skip.saturating_add(outer.skip), inner.fetch.saturating_sub(outer.skip).min(outer.fetch))
}

fn global_limit_spec(limit: &GlobalLimitExec, candidate: GpuCandidate) -> Result<NativeLimitSpec, NativePlanRejection> {
    let fetch = limit.fetch().ok_or(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })?;
    Ok(NativeLimitSpec::new(limit.skip(), fetch))
}

fn normalize_sort_input_root(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &SortExec,
    limit: Option<NativeLimitSpec>,
    candidate: GpuCandidate,
) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    if sort.preserve_partitioning() && !matches!(root, NativeStackRoot::SortPreservingMergeFetch) {
        return Err(NativePlanRejection {
            candidate,
            reason: FallbackReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let (projection, input) = peel_root_projection(sort.input().as_ref());
    normalize_input_root(output_schema_fields, Some(root), Some(sort), limit, projection, input, candidate)
}

fn normalize_input_root<'a>(
    output_schema_fields: usize,
    stack_root: Option<NativeStackRoot>,
    sort: Option<&'a SortExec>,
    limit: Option<NativeLimitSpec>,
    projection: Option<&'a ProjectionExec>,
    input_plan: &'a dyn ExecutionPlan,
    candidate: GpuCandidate,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    let input = normalize_native_input(input_plan, candidate.clone())?;
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
        limit,
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
    if let Some(coalesce) = plan.as_any().downcast_ref::<CoalescePartitionsExec>()
        && let Some(fetch) = coalesce.fetch()
    {
        let input = normalize_native_input(coalesce.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Limit(NormalizedNativeLimit {
            spec: NativeLimitSpec::fetch(fetch),
            input: Box::new(input),
        }));
    }

    let plan = peel_repartition(plan);
    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
        let input = peel_repartition(filter.input().as_ref());
        let normalized = if let Some(source_exec) = input.as_any().downcast_ref::<DataSourceExec>() {
            NormalizedNativeInput::Source(NormalizedNativeSource {
                scan: NormalizedNativeScan::LocalParquet(source_exec),
                filter: Some(filter),
            })
        } else if let Some(scan_exec) = input.as_any().downcast_ref::<CudfTableFormatScanExec>() {
            NormalizedNativeInput::Source(NormalizedNativeSource {
                scan: NormalizedNativeScan::TableFormat(scan_exec),
                filter: Some(filter),
            })
        } else {
            let input = normalize_native_input(input, candidate)?;
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
    if let Some(source_exec) = plan.as_any().downcast_ref::<DataSourceExec>() {
        return Ok(NormalizedNativeInput::Source(NormalizedNativeSource {
            scan: NormalizedNativeScan::LocalParquet(source_exec),
            filter: None,
        }));
    }
    if let Some(scan_exec) = plan.as_any().downcast_ref::<CudfTableFormatScanExec>() {
        return Ok(NormalizedNativeInput::Source(NormalizedNativeSource {
            scan: NormalizedNativeScan::TableFormat(scan_exec),
            filter: None,
        }));
    }
    if let Some(input_name) = gpu_resident_producer_input_name(plan) {
        return Ok(NormalizedNativeInput::GpuResidentSource(NormalizedNativeGpuResidentSource {
            source_exec: plan,
            input_name,
        }));
    }
    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        let input = normalize_native_input(projection.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Projection(NormalizedNativeProjection {
            projection,
            input: Box::new(input),
        }));
    }
    if let Some(limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
        let spec = global_limit_spec(limit, candidate.clone())?;
        let input = normalize_native_input(limit.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Limit(NormalizedNativeLimit { spec, input: Box::new(input) }));
    }
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        let aggregate_peel = peel_distributed_aggregate_pair_if_present(aggregate, candidate.clone())?;
        let input = normalize_native_input(aggregate_peel.aggregate.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Aggregate(NormalizedNativeAggregate {
            aggregate: aggregate_peel.aggregate,
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            input: Box::new(input),
        }));
    }
    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        if sort.preserve_partitioning() {
            return Err(NativePlanRejection {
                candidate,
                reason: FallbackReason::NativePartitionPreservingSortUnsupported,
            });
        }
        let input = normalize_native_input(sort.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Sort(NormalizedNativeSort { sort, input: Box::new(input) }));
    }
    if let Some(window) = plan.as_any().downcast_ref::<WindowAggExec>() {
        let input = normalize_native_input(window.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Window(NormalizedNativeWindow {
            window: NormalizedNativeWindowExec::Unbounded(window),
            input: Box::new(input),
        }));
    }
    if let Some(window) = plan.as_any().downcast_ref::<BoundedWindowAggExec>() {
        let input = normalize_native_input(window.input().as_ref(), candidate)?;
        return Ok(NormalizedNativeInput::Window(NormalizedNativeWindow {
            window: NormalizedNativeWindowExec::Bounded(window),
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
    if let Some(union) = plan.as_any().downcast_ref::<UnionExec>() {
        let inputs = union
            .inputs()
            .iter()
            .map(|input| normalize_native_input(input.as_ref(), candidate.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(NormalizedNativeInput::Union(NormalizedNativeUnion { union, inputs }));
    }
    Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
}

fn peel_repartition(plan: &dyn ExecutionPlan) -> &dyn ExecutionPlan {
    // Native shapes treat repartition and coalesce-partitions as exchange
    // boundaries that can be peeled while preserving the child semantics.
    if let Some(repartition) = plan.as_any().downcast_ref::<RepartitionExec>() {
        repartition.input().as_ref()
    } else if let Some(coalesce) = plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
        if coalesce.fetch().is_some() { plan } else { coalesce.input().as_ref() }
    } else if let Some(cooperative) = plan.as_any().downcast_ref::<CooperativeExec>() {
        cooperative.input().as_ref()
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
