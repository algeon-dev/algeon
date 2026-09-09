use std::collections::BTreeSet;

use super::{
    BoundedJoinLeafSourceStreamDynamicFilterStage, DynamicFilter, DynamicFilterConsumerLowering,
    DynamicFilterExecutionOptions, DynamicFilterId, DynamicFilterRoute, DynamicFilterSet, DynamicFilterSourceSnapshot,
    FilterNode, GpuDataFrame, LocalParquetSource, NodeId, QueryMetrics, Result, SourceChunk, SourceChunkBatch,
    SourceFooterFacts, dynamic_filter_predicate_for_schema, dynamic_filter_reader_predicate_should_merge,
};

pub(super) fn merged_reader_predicate_route_ids(
    source: &LocalParquetSource,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
    options: DynamicFilterExecutionOptions,
    footers: SourceFooterFacts<'_>,
) -> Result<BTreeSet<DynamicFilterId>> {
    let mut ids = BTreeSet::new();
    for route in routes {
        if deferred_reader_predicate_routes.contains(&route.id) {
            continue;
        }
        if !matches!(
            route.consumer_lowering,
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate
        ) {
            continue;
        }
        let Some(filter) = filter_set.produced(route.id) else {
            continue;
        };
        let Some(column) = route.consumer_columns.first() else {
            continue;
        };
        if dynamic_filter_reader_predicate_should_merge(
            source,
            column,
            filter,
            options,
            crate::rapids_backend::rapids_analysis_context(),
            footers,
        )? {
            ids.insert(route.id);
        }
    }
    Ok(ids)
}

pub(super) fn source_reader_predicate_dynamic_filter_stages(
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    merged_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
) -> Vec<BoundedJoinLeafSourceStreamDynamicFilterStage> {
    routes
        .iter()
        .filter_map(|route| {
            if !merged_reader_predicate_routes.contains(&route.id) {
                return None;
            }
            let filter = filter_set.produced(route.id)?;
            if matches!(filter, DynamicFilter::Empty | DynamicFilter::HashSetContainsI64 { .. }) {
                return None;
            }
            Some(BoundedJoinLeafSourceStreamDynamicFilterStage { route: (*route).clone(), filter: filter.clone() })
        })
        .collect()
}

pub(super) fn deferred_large_small_in_list_reader_predicate_routes(
    source: &LocalParquetSource,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    options: DynamicFilterExecutionOptions,
) -> BTreeSet<DynamicFilterId> {
    let mut large_small_in_list_routes = BTreeSet::new();
    let mut retained_reader_predicate_count = usize::from(source.reader_predicate.is_some());
    for route in routes {
        if !matches!(
            route.consumer_lowering,
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate
        ) {
            continue;
        }
        let Some(filter) = filter_set.produced(route.id) else {
            continue;
        };
        if large_small_in_list_reader_predicate_should_defer(filter, options) {
            large_small_in_list_routes.insert(route.id);
        } else if !matches!(filter, DynamicFilter::Empty | DynamicFilter::HashSetContainsI64 { .. }) {
            retained_reader_predicate_count = retained_reader_predicate_count.saturating_add(1);
        }
    }
    if retained_reader_predicate_count == 0 {
        return BTreeSet::new();
    }
    large_small_in_list_routes
}

pub(super) fn large_small_in_list_reader_predicate_should_defer(
    filter: &DynamicFilter,
    options: DynamicFilterExecutionOptions,
) -> bool {
    matches!(
        filter,
        DynamicFilter::SmallInListI64 {
            values,
            ..
        } if values.len() > options.max_deferred_small_in_list_reader_values
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn source_post_decode_dynamic_filter_stages(
    source: &LocalParquetSource,
    source_node_id: NodeId,
    routes: &[&DynamicFilterRoute],
    filter_set: &DynamicFilterSet,
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
    deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
    options: DynamicFilterExecutionOptions,
) -> Vec<(DynamicFilterRoute, DynamicFilter, FilterNode)> {
    routes
        .iter()
        .filter_map(|route| {
            let filter = filter_set.produced(route.id)?;
            if !source_dynamic_filter_requires_post_decode(
                source,
                route,
                filter,
                before,
                after,
                deferred_reader_predicate_routes,
                options,
            ) {
                return None;
            }
            let column = route.consumer_columns.first()?;
            let predicate = dynamic_filter_predicate_for_schema(source.schema.as_ref(), column, filter)?;
            Some(((*route).clone(), filter.clone(), FilterNode { input: source_node_id, predicate }))
        })
        .collect()
}

pub(super) fn source_dynamic_filter_requires_post_decode(
    source: &LocalParquetSource,
    route: &DynamicFilterRoute,
    filter: &DynamicFilter,
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
    deferred_reader_predicate_routes: &BTreeSet<DynamicFilterId>,
    options: DynamicFilterExecutionOptions,
) -> bool {
    !matches!(filter, DynamicFilter::Empty)
        && (deferred_reader_predicate_routes.contains(&route.id)
            || unpruned_reader_hash_set_should_run_post_decode(source, route, filter, before, after, options)
            || matches!(
                route.consumer_lowering,
                DynamicFilterConsumerLowering::PostDecodeOnly
                    | DynamicFilterConsumerLowering::SkippedUnsupportedStaticPredicateMix
            ))
}

pub(super) fn unpruned_reader_hash_set_should_run_post_decode(
    source: &LocalParquetSource,
    route: &DynamicFilterRoute,
    filter: &DynamicFilter,
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
    options: DynamicFilterExecutionOptions,
) -> bool {
    if route.consumer_lowering != DynamicFilterConsumerLowering::ReaderPredicate {
        return false;
    }
    if !dynamic_filter_source_pruning_unchanged(before, after) {
        return false;
    }
    let DynamicFilter::HashSetContainsI64 { values, .. } = filter else {
        return false;
    };
    if values.len() > options.max_unpruned_hash_set_post_decode_values {
        return false;
    }
    selected_source_known_rows(source).is_some_and(|rows| rows >= options.min_unpruned_hash_set_post_decode_source_rows)
}

pub(super) fn dynamic_filter_source_pruning_unchanged(
    before: DynamicFilterSourceSnapshot,
    after: DynamicFilterSourceSnapshot,
) -> bool {
    before.row_groups == after.row_groups && before.source_bytes == after.source_bytes
}

pub(super) fn selected_source_known_rows(source: &LocalParquetSource) -> Option<u64> {
    source.files.iter().try_fold(0_u64, |total, file| {
        let row_group_rows = file.row_group_rows.as_ref()?;
        let selected_rows = match &file.row_groups {
            Some(row_groups) => row_groups.iter().try_fold(0_u64, |sum, row_group| {
                row_group_rows.get(*row_group).and_then(|rows| sum.checked_add(*rows))
            })?,
            None => row_group_rows.iter().try_fold(0_u64, |sum, rows| sum.checked_add(*rows))?,
        };
        total.checked_add(selected_rows)
    })
}

pub(super) fn record_dynamic_filter_reader_predicate_rows(
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    chunk: &SourceChunk,
    frame: &GpuDataFrame,
    metrics: &mut QueryMetrics,
) {
    if reader_dynamic_filters.is_empty() {
        return;
    }
    let rows_before = chunk.known_row_count;
    let rows_after = Some(frame.num_rows() as u64);
    metrics.record_dynamic_filter_reader_predicate_source_reduction(rows_before, rows_after);
    for dynamic_filter in reader_dynamic_filters {
        metrics.record_dynamic_filter_source_reader_predicate_rows(
            &dynamic_filter.route,
            &dynamic_filter.filter,
            chunk.file_index,
            chunk.chunk_index,
            rows_before,
            rows_after,
        );
    }
}

pub(super) fn record_dynamic_filter_reader_predicate_batch_rows(
    reader_dynamic_filters: &[BoundedJoinLeafSourceStreamDynamicFilterStage],
    batch: &SourceChunkBatch,
    frame: &GpuDataFrame,
    metrics: &mut QueryMetrics,
) {
    if reader_dynamic_filters.is_empty() {
        return;
    }
    let rows_before =
        batch.chunks.iter().try_fold(0_u64, |total, chunk| Some(total.saturating_add(chunk.known_row_count?)));
    let rows_after = Some(frame.num_rows() as u64);
    metrics.record_dynamic_filter_reader_predicate_source_reduction(rows_before, rows_after);
    for dynamic_filter in reader_dynamic_filters {
        metrics.record_dynamic_filter_source_reader_predicate_batch_rows(
            &dynamic_filter.route,
            &dynamic_filter.filter,
            rows_before,
            rows_after,
        );
    }
}
