use datafusion::{
    datasource::{memory::MemorySourceConfig, source::DataSourceExec},
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
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
use nexus_query_engine::LimitSpec as NativeLimitSpec;

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
    if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
        return normalize_sort_preserving_merge_root(merge);
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
        );
    }

    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
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
                let sort_fetch =
                    projection.input().downcast_ref::<SortExec>().and_then(SortExec::fetch).map(NativeLimitSpec::fetch);
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

    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return normalize_global_aggregate_stack(plan.schema().fields().len(), None, aggregate);
    }

    if plan.downcast_ref::<WindowAggExec>().is_some() || plan.downcast_ref::<BoundedWindowAggExec>().is_some() {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(plan.schema().fields().len(), None, None, None, None, plan, candidate);
    }

    if plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<DataSourceExec>().is_some()
        || plan.downcast_ref::<PlaceholderRowExec>().is_some()
        || plan.downcast_ref::<EmptyExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.downcast_ref::<CooperativeExec>().is_some()
        || gpu_resident_producer_input_name(plan).is_some()
        || plan.downcast_ref::<CudfTableFormatScanExec>().is_some()
    {
        let candidate = input_root_candidate(None, plan.schema().fields().len(), plan.name().to_string(), false, 1);
        return normalize_input_root(plan.schema().fields().len(), None, None, None, None, plan, candidate);
    }

    if plan.downcast_ref::<HashJoinExec>().is_some()
        || plan.downcast_ref::<CrossJoinExec>().is_some()
        || plan.downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.downcast_ref::<UnionExec>().is_some()
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
    if !partition_preserving_sort_is_supported(sort, native_stack_root_is_sort_preserving_merge(root)) {
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
        sort: Some(NormalizedNativeSortExprs::from_sort_exec(sort)),
        limit,
    })
}

fn normalize_sort_preserving_merge_root(
    merge: &SortPreservingMergeExec,
) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
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
        return Err(NativePlanRejection { candidate, reason: FallbackReason::NativeSortPreservingMergeUnsupported });
    }
    if !plan_output_ordering_matches(merge.input().as_ref(), merge.expr()) {
        return Err(NativePlanRejection { candidate, reason: FallbackReason::NativeSortPreservingMergeUnsupported });
    }

    if let Some(sort) = merge.input().downcast_ref::<SortExec>() {
        if merge.fetch().is_none() && sort.fetch().is_some() {
            return Err(NativePlanRejection {
                candidate,
                reason: FallbackReason::NativeSortPreservingMergeUnsupported,
            });
        }
        if let Some(fetch) = merge.fetch()
            && sort.fetch().is_some_and(|sort_fetch| sort_fetch < fetch)
        {
            return Err(NativePlanRejection {
                candidate,
                reason: FallbackReason::NativeSortPreservingMergeUnsupported,
            });
        }
        if sort_orderings_match(merge.expr(), sort.expr()) {
            return normalize_sort_stack(
                root_with_child_sort,
                output_schema_fields,
                sort,
                merge.fetch().map(NativeLimitSpec::fetch),
            );
        }
    }

    let input_plan = merge.input().as_ref();
    let sort = NormalizedNativeSortExprs::from_ordering(merge.expr(), input_plan.schema(), merge.schema());
    let (projection, input) = peel_root_projection(input_plan);
    normalize_input_root(
        output_schema_fields,
        Some(root_without_child_sort),
        Some(sort),
        merge.fetch().map(NativeLimitSpec::fetch),
        projection,
        input,
        candidate,
    )
}

fn sort_preserving_merge_candidate(merge: &SortPreservingMergeExec) -> GpuCandidate {
    if merge.fetch().is_some() {
        if merge.input().downcast_ref::<SortExec>().is_some() {
            GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_FETCH_SHAPE, merge.schema().fields().len(), 2)
        } else {
            GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_FETCH_INPUT_SHAPE, merge.schema().fields().len(), 1)
        }
    } else if merge.input().downcast_ref::<SortExec>().is_some() {
        GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_SORT_SHAPE, merge.schema().fields().len(), 2)
    } else {
        GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_SHAPE, merge.schema().fields().len(), 1)
    }
}

fn plan_output_ordering_matches(plan: &dyn ExecutionPlan, ordering: &LexOrdering) -> bool {
    plan.properties().output_ordering().is_some_and(|output_ordering| sort_orderings_match(output_ordering, ordering))
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

    if let Some(aggregate) = input.downcast_ref::<AggregateExec>() {
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

fn normalize_sort_input_root(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &SortExec,
    limit: Option<NativeLimitSpec>,
    candidate: GpuCandidate,
) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    if !partition_preserving_sort_is_supported(sort, native_stack_root_is_sort_preserving_merge(root)) {
        return Err(NativePlanRejection {
            candidate,
            reason: FallbackReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let (projection, input) = peel_root_projection(sort.input().as_ref());
    normalize_input_root(
        output_schema_fields,
        Some(root),
        Some(NormalizedNativeSortExprs::from_sort_exec(sort)),
        limit,
        projection,
        input,
        candidate,
    )
}

fn normalize_input_root<'a>(
    output_schema_fields: usize,
    stack_root: Option<NativeStackRoot>,
    sort: Option<NormalizedNativeSortExprs<'a>>,
    limit: Option<NativeLimitSpec>,
    projection: Option<&'a ProjectionExec>,
    input_plan: &'a dyn ExecutionPlan,
    candidate: GpuCandidate,
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
        root: NormalizedNativeRoot::Input(NormalizedNativeInputRoot { input, projection }),
        sort,
        limit,
    })
}

fn peel_post_aggregate_projection(
    sort: &SortExec,
    candidate: GpuCandidate,
) -> Result<(Option<&ProjectionExec>, &AggregateExec), NativePlanRejection> {
    if let Some(projection) = sort.input().downcast_ref::<ProjectionExec>() {
        return Ok((Some(projection), required_child::<AggregateExec>(projection.input().as_ref(), candidate)?));
    }
    Ok((None, required_child::<AggregateExec>(sort.input().as_ref(), candidate)?))
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
        } else if let Some(scan_exec) = input.downcast_ref::<CudfTableFormatScanExec>() {
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
                return Err(NativePlanRejection { candidate, reason: FallbackReason::NativeOrderedOutputUnsupported });
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
    if let Some(scan_exec) = plan.downcast_ref::<CudfTableFormatScanExec>() {
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
            aggregate_allows_partial: aggregate_peel.aggregate_allows_partial,
            aggregate_output_schema: aggregate_peel.aggregate_output_schema,
            input: Box::new(input),
        }));
    }
    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        if !partition_preserving_sort_is_supported(sort, false) {
            return Err(NativePlanRejection {
                candidate,
                reason: FallbackReason::NativePartitionPreservingSortUnsupported,
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
        let left = normalize_native_input(join.left().as_ref(), candidate.clone(), false)?;
        let right = normalize_native_input(join.right().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::HashJoin(NormalizedNativeHashJoin {
            join,
            fetch: join.fetch().map(NativeLimitSpec::fetch),
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(join) = plan.downcast_ref::<CrossJoinExec>() {
        let left = normalize_native_input(join.left().as_ref(), candidate.clone(), false)?;
        let right = normalize_native_input(join.right().as_ref(), candidate, false)?;
        return Ok(NormalizedNativeInput::CrossJoin(NormalizedNativeCrossJoin {
            join,
            left: Box::new(left),
            right: Box::new(right),
        }));
    }
    if let Some(join) = plan.downcast_ref::<NestedLoopJoinExec>() {
        let left = normalize_native_input(join.left().as_ref(), candidate.clone(), false)?;
        let right = normalize_native_input(join.right().as_ref(), candidate, false)?;
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
            .map(|input| normalize_native_input(input.as_ref(), candidate.clone(), false))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(NormalizedNativeInput::Union(NormalizedNativeUnion { union, inputs }));
    }
    Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
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
    plan.downcast_ref::<T>().ok_or(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
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
