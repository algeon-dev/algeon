//! Normalization of a DataFusion physical subtree into one native input tree.
//!
//! Normalization removes the operators that only move rows (repartition,
//! coalesce, cooperative yield), reconstructs a partial/final aggregate pair as
//! one aggregate under the final schema, and turns a sort-preserving merge into
//! one sort over the concatenated partitions. The result keeps a reference to
//! each DataFusion operator it represents, so a rejection can name the right
//! one. A rejection carries the candidate shape, the blocking operator, and
//! what one partition would change.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::{
    datasource::{memory::MemorySourceConfig, source::DataSourceExec},
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, Partitioning,
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
use query_planning::expr::LimitSpec as NativeLimitSpec;

use crate::{
    capability::CapabilityReason,
    native::validate::validate_sort_ordering,
    native_exec::{
        gpu_resident_producer_hard_output_row_upper_bound, gpu_resident_producer_input_name,
        gpu_resident_producer_unique_output_key,
    },
    report::{GpuCandidate, PlanBlocker, SinglePartitionOutcome, UNSUPPORTED_NATIVE_PLAN_OPERATOR},
};

mod distributed_aggregate;
mod types;

use distributed_aggregate::peel_distributed_aggregate_pair_if_present;
pub(in crate::native) use types::{
    NativePlanRejection, NormalizedNativeAggregate, NormalizedNativeCrossJoin, NormalizedNativeFilter,
    NormalizedNativeGpuResidentSource, NormalizedNativeHashJoin, NormalizedNativeInput, NormalizedNativeLimit,
    NormalizedNativeLiteralSource, NormalizedNativeNestedLoopJoin, NormalizedNativePlan, NormalizedNativeProjection,
    NormalizedNativeSort, NormalizedNativeSortExprs, NormalizedNativeSortOrigin, NormalizedNativeSource,
    NormalizedNativeUnion, NormalizedNativeWindow, NormalizedNativeWindowExec,
};

/// Normalizes the physical subtree rooted at `plan` into one native input
/// tree. A rejection carries the candidate shape and the operator that blocked
/// it. When the root itself has no native counterpart, the shape is the
/// unsupported plan shape and the blocker keeps the root operator's name.
pub(super) fn normalize_native_plan(plan: &dyn ExecutionPlan) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    let output_schema_fields = plan.schema().fields().len();
    let plan = peel_repartition(plan);
    let root_shape = if root_operator_is_normalized(plan) { plan.name() } else { UNSUPPORTED_NATIVE_PLAN_OPERATOR };
    let candidate = GpuCandidate::new(root_shape, output_schema_fields);
    let input = normalize_native_input(plan, candidate, false)?;
    Ok(NormalizedNativePlan::new(input, output_schema_fields))
}

fn root_operator_is_normalized(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<SortPreservingMergeExec>().is_some()
        || plan.downcast_ref::<LocalLimitExec>().is_some()
        || plan.downcast_ref::<GlobalLimitExec>().is_some()
        || plan.downcast_ref::<SortExec>().is_some()
        || plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<AggregateExec>().is_some()
        || plan.downcast_ref::<WindowAggExec>().is_some()
        || plan.downcast_ref::<BoundedWindowAggExec>().is_some()
        || plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<DataSourceExec>().is_some()
        || plan.downcast_ref::<PlaceholderRowExec>().is_some()
        || plan.downcast_ref::<EmptyExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.downcast_ref::<HashJoinExec>().is_some()
        || plan.downcast_ref::<CrossJoinExec>().is_some()
        || plan.downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.downcast_ref::<UnionExec>().is_some()
        || gpu_resident_producer_input_name(plan).is_some()
}

/// A hash join at the root that stops after `fetch` rows emits them in probe
/// order, which DataFusion reports as the plan's ordering. The native join
/// does not keep probe order, so the fetch becomes a sort by that ordering
/// followed by the fetch.
pub(super) fn make_declared_ordered_hash_join_fetch_explicit<'a>(
    normalized: NormalizedNativePlan<'a>,
    ordering: Option<&'a LexOrdering>,
    output_schema: SchemaRef,
) -> NormalizedNativePlan<'a> {
    let output_schema_fields = normalized.candidate.output_schema_fields;
    match (ordering, normalized.input) {
        (Some(ordering), NormalizedNativeInput::HashJoin(mut join)) if join.fetch.is_some() => {
            let fetch = join.fetch.take();
            NormalizedNativePlan::new(
                NormalizedNativeInput::Sort(NormalizedNativeSort {
                    origin: NormalizedNativeSortOrigin::DeclaredOutputOrdering,
                    sort: NormalizedNativeSortExprs::from_ordering(ordering, Arc::clone(&output_schema), output_schema),
                    fetch,
                    input: Box::new(NormalizedNativeInput::HashJoin(join)),
                }),
                output_schema_fields,
            )
        },
        (_, input) => NormalizedNativePlan { candidate: normalized.candidate, input },
    }
}

/// Normalizes a merge of sorted partitions. One native sort over the
/// concatenated partitions reproduces the merge. When the partitions come from
/// a per-partition sort, the merge only adds its fetch, which must not ask for
/// more rows than each partition kept.
fn normalize_sort_preserving_merge<'a>(
    merge: &'a SortPreservingMergeExec,
    candidate: GpuCandidate,
) -> Result<NormalizedNativeInput<'a>, NativePlanRejection> {
    let unsupported = || NativePlanRejection {
        candidate: candidate.clone(),
        reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
        blocker: Box::new(PlanBlocker::Operator(merge.name().to_string())),
        single_partition: merge_single_partition_outcome(merge),
    };
    if !sort_ordering_is_column_keyed(merge.expr())
        || !plan_partitions_match_ordering(merge.input().as_ref(), merge.expr())
    {
        return Err(unsupported());
    }
    let merge_fetch = merge.fetch().map(NativeLimitSpec::fetch);
    let (projection, below_projection) = peel_root_projection(merge.input().as_ref());
    if let Some(sort) = below_projection.downcast_ref::<SortExec>() {
        match (merge.fetch(), sort.fetch()) {
            (None, Some(_)) => return Err(unsupported()),
            (Some(merge_fetch), Some(sort_fetch)) if sort_fetch < merge_fetch => return Err(unsupported()),
            _ => {},
        }
        if projection.is_some() || sort_orderings_match(merge.expr(), sort.expr()) {
            let sorted = normalize_native_input(merge.input().as_ref(), candidate, true)?;
            return Ok(with_merge_fetch(sorted, merge_fetch));
        }
    }
    let input = normalize_native_input(merge.input().as_ref(), candidate, true)?;
    Ok(NormalizedNativeInput::Sort(NormalizedNativeSort {
        origin: NormalizedNativeSortOrigin::SortPreservingMerge,
        sort: NormalizedNativeSortExprs::from_ordering(merge.expr(), merge.input().schema(), merge.schema()),
        fetch: merge_fetch,
        input: Box::new(input),
    }))
}

/// What one partition does to a merge that normalization rejected. The merge
/// stays when the partitions it reads are union branches. Otherwise DataFusion
/// plans one sort over one partition, which lowers when native sort validation
/// accepts the merge's keys.
fn merge_single_partition_outcome(merge: &SortPreservingMergeExec) -> SinglePartitionOutcome {
    if reads_union_branches(merge.input().as_ref())
        || validate_sort_ordering(merge.expr(), &merge.input().schema()).is_err()
    {
        SinglePartitionOutcome::StillRejected
    } else {
        SinglePartitionOutcome::Lowers
    }
}

/// The merge over its child sort becomes that sort. The child sort keeps at
/// least as many rows per partition as the merge returns, so the merge fetch
/// replaces the sort's own. A projection between them keeps the fetch above
/// it.
fn with_merge_fetch(sorted: NormalizedNativeInput<'_>, fetch: Option<NativeLimitSpec>) -> NormalizedNativeInput<'_> {
    match sorted {
        NormalizedNativeInput::Sort(mut sort) => {
            sort.origin = NormalizedNativeSortOrigin::SortPreservingMergeOverSort { sort_fetch: sort.fetch.is_some() };
            sort.fetch = fetch;
            NormalizedNativeInput::Sort(sort)
        },
        mut projected => {
            if let NormalizedNativeInput::Projection(projection) = &mut projected
                && let NormalizedNativeInput::Sort(sort) = projection.input.as_mut()
            {
                sort.fetch = None;
            }
            match fetch {
                Some(fetch) => with_limit("SortPreservingMergeExec(fetch)", projected, fetch, true),
                None => projected,
            }
        },
    }
}

/// `spec` applied to `input`: folded into a sort's fetch or an inner limit,
/// otherwise a new limit node named after `operator`. `input_ordered` says
/// whether DataFusion reports an ordering for the limit's input, whose first
/// rows the limit then keeps.
fn with_limit<'a>(
    operator: &'static str,
    input: NormalizedNativeInput<'a>,
    spec: NativeLimitSpec,
    input_ordered: bool,
) -> NormalizedNativeInput<'a> {
    match input {
        NormalizedNativeInput::Sort(mut sort) => {
            sort.fetch = Some(sort.fetch.map_or(spec, |inner| compose_limits(inner, spec)));
            NormalizedNativeInput::Sort(sort)
        },
        NormalizedNativeInput::Limit(mut limit) => {
            limit.spec = compose_limits(limit.spec, spec);
            limit.input_ordered |= input_ordered;
            NormalizedNativeInput::Limit(limit)
        },
        input => NormalizedNativeInput::Limit(NormalizedNativeLimit {
            operator,
            spec,
            input_ordered,
            input: Box::new(input),
        }),
    }
}

fn plan_partitions_match_ordering(plan: &dyn ExecutionPlan, ordering: &LexOrdering) -> bool {
    plan.equivalence_properties().ordering_satisfy(ordering.iter().cloned()).unwrap_or(false)
        || plan.downcast_ref::<UnionExec>().is_some_and(|union| {
            union.inputs().iter().all(|input| plan_partitions_match_ordering(input.as_ref(), ordering))
        })
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

fn peel_root_projection(plan: &dyn ExecutionPlan) -> (Option<&ProjectionExec>, &dyn ExecutionPlan) {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        (Some(projection), projection.input().as_ref())
    } else {
        (None, plan)
    }
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
    // A repartition above a fetch-limited coalesce redistributes the limited
    // rows without changing them, so the limit is found after peeling it.
    let plan = peel_repartition(plan);
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
        && let Some(fetch) = coalesce.fetch()
    {
        let input = normalize_native_input(coalesce.input().as_ref(), candidate, false)?;
        // Coalescing partitions keeps no order, so any rows satisfy the fetch.
        return Ok(with_limit("GlobalLimitExec", input, NativeLimitSpec::fetch(fetch), false));
    }
    if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
        return normalize_sort_preserving_merge(merge, candidate);
    }
    if let Some(limit) = plan.downcast_ref::<LocalLimitExec>() {
        // A local limit keeps `fetch` rows of every partition; over one
        // partition that is the global fetch the native plan applies.
        if limit.input().output_partitioning().partition_count() != 1 {
            return Err(NativePlanRejection {
                candidate,
                reason: CapabilityReason::NativeLocalLimitPartitionUnsupported,
                blocker: Box::new(PlanBlocker::Operator(limit.name().to_string())),
                single_partition: SinglePartitionOutcome::Lowers,
            });
        }
        let input = normalize_native_input(limit.input().as_ref(), candidate, false)?;
        return Ok(with_limit(
            "LocalLimitExec",
            input,
            NativeLimitSpec::fetch(limit.fetch()),
            limit.input().output_ordering().is_some(),
        ));
    }

    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        let input = peel_repartition(filter.input().as_ref());
        let child_allows_ordered_multi_partition_memory_source =
            filter.fetch().is_none() && allow_ordered_multi_partition_memory_source;
        let normalized = if let Some(source_exec) = input.downcast_ref::<DataSourceExec>()
            && !data_source_is_memory(source_exec)
        {
            NormalizedNativeInput::Source(NormalizedNativeSource { scan: source_exec, filter: Some(filter) })
        } else {
            let input = normalize_native_input(input, candidate, child_allows_ordered_multi_partition_memory_source)?;
            NormalizedNativeInput::Filter(NormalizedNativeFilter { filter, input: Box::new(input) })
        };
        if let Some(fetch) = filter.fetch() {
            return Ok(with_limit(
                "GlobalLimitExec",
                normalized,
                NativeLimitSpec::fetch(fetch),
                filter.properties().output_ordering().is_some(),
            ));
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
                    blocker: Box::new(PlanBlocker::Operator(source_exec.name().to_string())),
                    single_partition: SinglePartitionOutcome::Lowers,
                });
            }
            return Ok(NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::Memory(source_exec)));
        }
        return Ok(NormalizedNativeInput::Source(NormalizedNativeSource { scan: source_exec, filter: None }));
    }
    if let Some(placeholder) = plan.downcast_ref::<PlaceholderRowExec>() {
        return Ok(NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::PlaceholderRow(placeholder)));
    }
    if let Some(empty) = plan.downcast_ref::<EmptyExec>() {
        return Ok(NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::Empty(empty)));
    }
    if let Some(input_name) = gpu_resident_producer_input_name(plan) {
        return Ok(NormalizedNativeInput::GpuResidentSource(NormalizedNativeGpuResidentSource {
            source_exec: plan,
            input_name,
            row_count_upper_bound: gpu_resident_producer_hard_output_row_upper_bound(plan),
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
        let input = normalize_native_input(limit.input().as_ref(), candidate, false)?;
        return Ok(with_limit(
            "GlobalLimitExec",
            input,
            NativeLimitSpec::new(limit.skip(), limit.fetch()),
            limit.input().output_ordering().is_some(),
        ));
    }
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        let aggregate_peel = peel_distributed_aggregate_pair_if_present(aggregate, candidate.clone())?;
        let (pre_aggregate_projection, aggregate_input) =
            peel_root_projection(aggregate_peel.aggregate.input().as_ref());
        let input = normalize_native_input(peel_repartition(aggregate_input), candidate, false)?;
        return Ok(NormalizedNativeInput::Aggregate(NormalizedNativeAggregate {
            aggregate: aggregate_peel.aggregate,
            pre_aggregate_projection,
            distributed_exchange: aggregate_peel.exchange,
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
                blocker: Box::new(PlanBlocker::Operator(sort.name().to_string())),
                single_partition: SinglePartitionOutcome::Lowers,
            });
        }
        let input = normalize_native_input(sort.input().as_ref(), candidate, true)?;
        return Ok(NormalizedNativeInput::Sort(NormalizedNativeSort {
            origin: NormalizedNativeSortOrigin::Sort,
            sort: NormalizedNativeSortExprs::from_sort_exec(sort),
            fetch: sort.fetch().map(NativeLimitSpec::fetch),
            input: Box::new(input),
        }));
    }
    if let Some(window) = plan.downcast_ref::<WindowAggExec>() {
        let input = normalize_native_input(window.input().as_ref(), candidate, true)?;
        return Ok(NormalizedNativeInput::Window(NormalizedNativeWindow {
            window: NormalizedNativeWindowExec::Unbounded(window),
            input: Box::new(input),
        }));
    }
    if let Some(window) = plan.downcast_ref::<BoundedWindowAggExec>() {
        let input = normalize_native_input(window.input().as_ref(), candidate, true)?;
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
    Err(NativePlanRejection {
        candidate,
        reason: CapabilityReason::UnsupportedOperator,
        blocker: Box::new(PlanBlocker::Operator(plan.name().to_string())),
        single_partition: SinglePartitionOutcome::Lowers,
    })
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

/// Whether `stage` delivers the branches of a `UnionExec` as separate
/// partitions to the operator above it. Each branch stays a partition at any
/// `datafusion.execution.target_partitions`. An aggregate over a union keeps
/// its partial and final stages at one partition, and a sort over a union keeps
/// the merge of its sorted partitions.
///
/// The search follows the inputs of `stage` down to a union. A merge of
/// partitions or a hash repartition on the way ends that path: the partitions
/// below no longer reach the operator above. DataFusion also merges the
/// collected side of a join, so a union there ends at the merge and only a
/// union on the probe side counts.
fn reads_union_branches(stage: &dyn ExecutionPlan) -> bool {
    stage.is::<UnionExec>()
        || stage.children().into_iter().any(|input| {
            let ends_partitions = input.is::<CoalescePartitionsExec>()
                || input.is::<SortPreservingMergeExec>()
                || input
                    .downcast_ref::<RepartitionExec>()
                    .is_some_and(|repartition| matches!(repartition.partitioning(), Partitioning::Hash(..)));
            !ends_partitions && reads_union_branches(input.as_ref())
        })
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
