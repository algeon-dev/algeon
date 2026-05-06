use arrow::datatypes::{DataType, SchemaRef};
use datafusion::{
    datasource::source::DataSourceExec,
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        joins::{HashJoinExec, NestedLoopJoinExec},
        limit::LocalLimitExec,
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use super::validate::is_supported_native_type;
use crate::{policy::FallbackReason, report::GpuCandidate};

const SORT_PRESERVING_MERGE_SHAPE: &str = "SortPreservingMergeExec";
const SORT_PRESERVING_MERGE_FETCH_SHAPE: &str = "SortPreservingMergeExec(fetch)(SortExec)";
const UNSUPPORTED_NATIVE_PLAN_SHAPE: &str = "UnsupportedNativePhysicalPlan";

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativePlan<'a> {
    pub(super) candidate: GpuCandidate,
    pub(super) root: NormalizedNativeRoot<'a>,
    pub(super) sort: Option<&'a SortExec>,
    pub(super) limit_fetch: Option<usize>,
}

impl<'a> NormalizedNativePlan<'a> {
    pub(super) fn single_source_input(&self) -> Option<&NormalizedNativeSource<'a>> {
        let root = self.aggregate_root()?;
        match &root.input {
            NormalizedNativeInput::Source(source) => Some(source),
            NormalizedNativeInput::Filter(_)
            | NormalizedNativeInput::Projection(_)
            | NormalizedNativeInput::Aggregate(_)
            | NormalizedNativeInput::HashJoin(_)
            | NormalizedNativeInput::NestedLoopJoin(_) => None,
        }
    }

    pub(super) fn root_input(&self) -> &NormalizedNativeInput<'a> {
        match &self.root {
            NormalizedNativeRoot::Aggregate(root) => &root.input,
            NormalizedNativeRoot::Input(root) => &root.input,
        }
    }

    pub(super) fn aggregate_root(&self) -> Option<&NormalizedNativeAggregateRoot<'a>> {
        match &self.root {
            NormalizedNativeRoot::Aggregate(root) => Some(root),
            NormalizedNativeRoot::Input(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum NormalizedNativeRoot<'a> {
    Aggregate(NormalizedNativeAggregateRoot<'a>),
    Input(NormalizedNativeInputRoot<'a>),
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeAggregateRoot<'a> {
    pub(super) input: NormalizedNativeInput<'a>,
    pub(super) pre_aggregate_projection: Option<&'a ProjectionExec>,
    pub(super) aggregate: &'a AggregateExec,
    pub(super) aggregate_allows_partial: bool,
    pub(super) aggregate_output_schema: SchemaRef,
    pub(super) post_aggregate_projection: Option<&'a ProjectionExec>,
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeInputRoot<'a> {
    pub(super) input: NormalizedNativeInput<'a>,
    pub(super) projection: Option<&'a ProjectionExec>,
}

#[derive(Debug, Clone)]
pub(super) enum NormalizedNativeInput<'a> {
    Source(NormalizedNativeSource<'a>),
    Filter(NormalizedNativeFilter<'a>),
    Projection(NormalizedNativeProjection<'a>),
    Aggregate(NormalizedNativeAggregate<'a>),
    HashJoin(NormalizedNativeHashJoin<'a>),
    NestedLoopJoin(NormalizedNativeNestedLoopJoin<'a>),
}

impl NormalizedNativeInput<'_> {
    fn shape(&self) -> String {
        match self {
            Self::Source(source) => source.shape(),
            Self::Filter(filter) => filter.shape(),
            Self::Projection(projection) => projection.shape(),
            Self::Aggregate(aggregate) => aggregate.shape(),
            Self::HashJoin(join) => join.shape(),
            Self::NestedLoopJoin(join) => join.shape(),
        }
    }

    fn operator_count(&self) -> usize {
        match self {
            Self::Source(source) => source.operator_count(),
            Self::Filter(filter) => filter.operator_count(),
            Self::Projection(projection) => projection.operator_count(),
            Self::Aggregate(aggregate) => aggregate.operator_count(),
            Self::HashJoin(join) => join.operator_count(),
            Self::NestedLoopJoin(join) => join.operator_count(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeSource<'a> {
    pub(super) source_exec: &'a DataSourceExec,
    pub(super) filter: Option<&'a FilterExec>,
}

impl NormalizedNativeSource<'_> {
    fn shape(&self) -> String {
        if self.filter.is_some() {
            wrap_shape("FilterExec", "LocalParquetScan".to_string())
        } else {
            "LocalParquetScan".to_string()
        }
    }

    fn operator_count(&self) -> usize {
        1 + usize::from(self.filter.is_some())
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeFilter<'a> {
    pub(super) filter: &'a FilterExec,
    pub(super) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeFilter<'_> {
    fn shape(&self) -> String {
        wrap_shape("FilterExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeProjection<'a> {
    pub(super) projection: &'a ProjectionExec,
    pub(super) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeProjection<'_> {
    fn shape(&self) -> String {
        wrap_shape("ProjectionExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeAggregate<'a> {
    pub(super) aggregate: &'a AggregateExec,
    pub(super) aggregate_allows_partial: bool,
    pub(super) aggregate_output_schema: SchemaRef,
    pub(super) input: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeAggregate<'_> {
    fn shape(&self) -> String {
        wrap_shape("AggregateExec", self.input.shape())
    }

    fn operator_count(&self) -> usize {
        1 + self.input.operator_count()
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeHashJoin<'a> {
    pub(super) join: &'a HashJoinExec,
    pub(super) left: Box<NormalizedNativeInput<'a>>,
    pub(super) right: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeHashJoin<'_> {
    fn shape(&self) -> String {
        wrap_shape("HashJoinExec", format!("{},{}", self.left.shape(), self.right.shape()))
    }

    fn operator_count(&self) -> usize {
        1 + self.left.operator_count() + self.right.operator_count()
    }
}

#[derive(Debug, Clone)]
pub(super) struct NormalizedNativeNestedLoopJoin<'a> {
    pub(super) join: &'a NestedLoopJoinExec,
    pub(super) left: Box<NormalizedNativeInput<'a>>,
    pub(super) right: Box<NormalizedNativeInput<'a>>,
}

impl NormalizedNativeNestedLoopJoin<'_> {
    fn shape(&self) -> String {
        wrap_shape("NestedLoopJoinExec", format!("{},{}", self.left.shape(), self.right.shape()))
    }

    fn operator_count(&self) -> usize {
        1 + self.left.operator_count() + self.right.operator_count()
    }
}

#[derive(Debug, Clone)]
pub(super) struct NativePlanRejection {
    pub(super) candidate: GpuCandidate,
    pub(super) reason: FallbackReason,
}

#[derive(Debug, Clone, Copy)]
enum NativeStackRoot {
    LocalLimit,
    Sort,
    SortFetch,
    SortPreservingMergeFetch,
}

#[derive(Debug, Clone, Copy)]
enum NativeAggregateShapeKind {
    Global,
    SortStack(NativeStackRoot),
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

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        let root = if sort.fetch().is_some() { NativeStackRoot::SortFetch } else { NativeStackRoot::Sort };
        return normalize_sort_stack(root, plan.schema().fields().len(), sort, sort.fetch());
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
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
) -> Result<NormalizedNativePlan, NativePlanRejection> {
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

fn normalize_sort_input_root(
    root: NativeStackRoot,
    output_schema_fields: usize,
    sort: &SortExec,
    limit_fetch: Option<usize>,
    candidate: GpuCandidate,
) -> Result<NormalizedNativePlan, NativePlanRejection> {
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

fn global_aggregate_candidate(
    output_schema_fields: usize,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    let shape = native_aggregate_shape(
        NativeAggregateShapeKind::Global,
        has_post_projection,
        has_pre_projection,
        has_filter,
        has_distributed_aggregate,
    );
    let operator_count = 2
        + usize::from(has_post_projection)
        + usize::from(has_pre_projection)
        + usize::from(has_filter)
        + if has_distributed_aggregate { 2 } else { 0 };
    GpuCandidate::with_operator_count(shape, output_schema_fields, operator_count)
}

fn global_aggregate_candidate_for_input(
    output_schema_fields: usize,
    input: &NormalizedNativeInput<'_>,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    if let NormalizedNativeInput::Source(source) = input {
        return global_aggregate_candidate(
            output_schema_fields,
            has_post_projection,
            has_pre_projection,
            source.filter.is_some(),
            has_distributed_aggregate,
        );
    }

    let shape = native_aggregate_shape_with_input(
        NativeAggregateShapeKind::Global,
        input.shape(),
        has_post_projection,
        has_pre_projection,
        has_distributed_aggregate,
    );
    let operator_count = input.operator_count()
        + 1
        + usize::from(has_post_projection)
        + usize::from(has_pre_projection)
        + if has_distributed_aggregate { 2 } else { 0 };
    GpuCandidate::with_operator_count(shape, output_schema_fields, operator_count)
}

fn native_stack_candidate_for_input(
    root: NativeStackRoot,
    output_schema_fields: usize,
    input: &NormalizedNativeInput<'_>,
    has_post_aggregate_projection: bool,
    has_pre_aggregate_projection: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    if let NormalizedNativeInput::Source(source) = input {
        return native_stack_candidate(
            root,
            output_schema_fields,
            has_post_aggregate_projection,
            has_pre_aggregate_projection,
            source.filter.is_some(),
            has_distributed_aggregate,
        );
    }

    let aggregate = native_aggregate_shape_with_input(
        NativeAggregateShapeKind::SortStack(root),
        input.shape(),
        has_post_aggregate_projection,
        has_pre_aggregate_projection,
        has_distributed_aggregate,
    );
    let shape = match root {
        NativeStackRoot::LocalLimit => wrap_shape("LocalLimitExec", wrap_shape("SortExec", aggregate)),
        NativeStackRoot::Sort => wrap_shape("SortExec", aggregate),
        NativeStackRoot::SortFetch => wrap_shape("SortExec(fetch)", aggregate),
        NativeStackRoot::SortPreservingMergeFetch => {
            wrap_shape("SortPreservingMergeExec(fetch)", wrap_shape("SortExec", aggregate))
        },
    };
    let operator_count = input.operator_count()
        + 2
        + usize::from(has_post_aggregate_projection)
        + usize::from(has_pre_aggregate_projection)
        + usize::from(matches!(root, NativeStackRoot::LocalLimit | NativeStackRoot::SortPreservingMergeFetch))
        + if has_distributed_aggregate { 2 } else { 0 };
    GpuCandidate::with_operator_count(shape, output_schema_fields, operator_count)
}

fn input_root_candidate(
    root: Option<NativeStackRoot>,
    output_schema_fields: usize,
    mut input: String,
    has_projection: bool,
    input_operator_count: usize,
) -> GpuCandidate {
    if has_projection {
        input = wrap_shape("ProjectionExec", input);
    }
    let shape = match root {
        Some(NativeStackRoot::LocalLimit) => wrap_shape("LocalLimitExec", wrap_shape("SortExec", input)),
        Some(NativeStackRoot::Sort) => wrap_shape("SortExec", input),
        Some(NativeStackRoot::SortFetch) => wrap_shape("SortExec(fetch)", input),
        Some(NativeStackRoot::SortPreservingMergeFetch) => {
            wrap_shape("SortPreservingMergeExec(fetch)", wrap_shape("SortExec", input))
        },
        None => input,
    };
    let operator_count = input_operator_count
        + usize::from(has_projection)
        + usize::from(root.is_some())
        + usize::from(matches!(root, Some(NativeStackRoot::LocalLimit | NativeStackRoot::SortPreservingMergeFetch)));
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
) -> Result<(Option<&ProjectionExec>, NormalizedNativeInput), NativePlanRejection> {
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

fn peel_distributed_aggregate_pair_if_present(
    aggregate: &AggregateExec,
    candidate: GpuCandidate,
) -> Result<(&AggregateExec, bool, bool, SchemaRef), NativePlanRejection> {
    if !matches!(aggregate.mode(), AggregateMode::Final | AggregateMode::FinalPartitioned) {
        return Ok((aggregate, false, false, aggregate.schema()));
    }

    let partial = peel_repartition(aggregate.input().as_ref()).as_any().downcast_ref::<AggregateExec>().ok_or(
        NativePlanRejection {
            candidate: candidate.clone(),
            reason: FallbackReason::NativeDistributedAggregateUnsupported,
        },
    )?;
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

fn native_stack_candidate(
    root: NativeStackRoot,
    output_schema_fields: usize,
    has_post_aggregate_projection: bool,
    has_pre_aggregate_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    let shape = native_sort_stack_shape(
        root,
        has_post_aggregate_projection,
        has_pre_aggregate_projection,
        has_filter,
        has_distributed_aggregate,
    );
    let operator_count = 3
        + usize::from(has_post_aggregate_projection)
        + usize::from(has_pre_aggregate_projection)
        + usize::from(has_filter);
    GpuCandidate::with_operator_count(
        shape,
        output_schema_fields,
        if has_distributed_aggregate {
            9
        } else {
            operator_count + usize::from(matches!(root, NativeStackRoot::SortPreservingMergeFetch))
        },
    )
}

fn native_sort_stack_shape(
    root: NativeStackRoot,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> String {
    let aggregate = native_aggregate_shape(
        NativeAggregateShapeKind::SortStack(root),
        has_post_projection,
        has_pre_projection,
        has_filter,
        has_distributed_aggregate,
    );
    match root {
        NativeStackRoot::LocalLimit => wrap_shape("LocalLimitExec", wrap_shape("SortExec", aggregate)),
        NativeStackRoot::Sort => wrap_shape("SortExec", aggregate),
        NativeStackRoot::SortFetch => wrap_shape("SortExec(fetch)", aggregate),
        NativeStackRoot::SortPreservingMergeFetch => {
            wrap_shape("SortPreservingMergeExec(fetch)", wrap_shape("SortExec", aggregate))
        },
    }
}

fn native_aggregate_shape(
    kind: NativeAggregateShapeKind,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> String {
    let mut source = "LocalParquetScan".to_string();
    if matches!(kind, NativeAggregateShapeKind::SortStack(NativeStackRoot::SortPreservingMergeFetch))
        && has_distributed_aggregate
    {
        source = wrap_shape("RepartitionExec", source);
    }
    if has_filter {
        source = wrap_shape("FilterExec", source);
    }
    if has_pre_projection {
        source = wrap_shape("ProjectionExec", source);
    }

    let mut aggregate = if has_distributed_aggregate {
        match kind {
            NativeAggregateShapeKind::SortStack(NativeStackRoot::SortPreservingMergeFetch) => wrap_shape(
                "FinalAggregateExec",
                wrap_shape("RepartitionExec", wrap_shape("PartialAggregateExec", source)),
            ),
            NativeAggregateShapeKind::Global | NativeAggregateShapeKind::SortStack(_) => wrap_shape(
                "FinalAggregateExec",
                wrap_shape("CoalescePartitionsExec", wrap_shape("PartialAggregateExec", source)),
            ),
        }
    } else {
        wrap_shape("AggregateExec", source)
    };

    if has_post_projection {
        aggregate = wrap_shape("ProjectionExec", aggregate);
    }
    aggregate
}

fn native_aggregate_shape_with_input(
    kind: NativeAggregateShapeKind,
    mut input: String,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_distributed_aggregate: bool,
) -> String {
    if matches!(kind, NativeAggregateShapeKind::SortStack(NativeStackRoot::SortPreservingMergeFetch))
        && has_distributed_aggregate
    {
        input = wrap_shape("RepartitionExec", input);
    }
    if has_pre_projection {
        input = wrap_shape("ProjectionExec", input);
    }

    let mut aggregate = if has_distributed_aggregate {
        match kind {
            NativeAggregateShapeKind::SortStack(NativeStackRoot::SortPreservingMergeFetch) => wrap_shape(
                "FinalAggregateExec",
                wrap_shape("RepartitionExec", wrap_shape("PartialAggregateExec", input)),
            ),
            NativeAggregateShapeKind::Global | NativeAggregateShapeKind::SortStack(_) => wrap_shape(
                "FinalAggregateExec",
                wrap_shape("CoalescePartitionsExec", wrap_shape("PartialAggregateExec", input)),
            ),
        }
    } else {
        wrap_shape("AggregateExec", input)
    };

    if has_post_projection {
        aggregate = wrap_shape("ProjectionExec", aggregate);
    }
    aggregate
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
