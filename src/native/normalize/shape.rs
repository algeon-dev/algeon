use super::*;

// Shape strings are diagnostics/report contract values. The grammar is a
// compact wrapper form: `Operator`, `Operator(child)`, and for binary inputs
// `Operator(left,right)`. Suffixes such as `(fetch)` are part of the operator
// token, not parsed metadata, so do not rename them casually.
pub(super) const SORT_PRESERVING_MERGE_SHAPE: &str = "SortPreservingMergeExec";
pub(super) const SORT_PRESERVING_MERGE_FETCH_SHAPE: &str = "SortPreservingMergeExec(fetch)(SortExec)";
pub(super) const UNSUPPORTED_NATIVE_PLAN_SHAPE: &str = "UnsupportedNativePhysicalPlan";

pub(super) fn global_aggregate_candidate(
    output_schema_fields: usize,
    has_post_projection: bool,
    has_pre_projection: bool,
    has_filter: bool,
    has_distributed_aggregate: bool,
) -> GpuCandidate {
    // Candidate helpers are the single source of truth for report shape strings
    // and operator counts. Keep these in lockstep with the normalized type
    // family and the explicit shape grammar above.
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

pub(super) fn global_aggregate_candidate_for_input(
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

pub(super) fn native_stack_candidate_for_input(
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

pub(super) fn input_root_candidate(
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

pub(super) fn native_stack_candidate(
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
