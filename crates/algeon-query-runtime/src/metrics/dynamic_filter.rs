use query_planning::dynamic_filter::{DynamicFilter, DynamicFilterConsumerReductionStage, DynamicFilterRoute};

use super::{QueryId, QueryMetrics, optional_i64, optional_str, optional_u64};

/// Source-read footprint before or after applying a dynamic filter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DynamicFilterSourceSnapshot {
    /// Planned row groups visible to the source consumer.
    pub row_groups: u64,
    /// Estimated source-read bytes, when known.
    pub source_bytes: Option<u64>,
}

/// Runtime diagnostic row for one dynamic-filter route event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicFilterRuntimeMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub(crate) query_id: Option<QueryId>,
    /// Plan-local dynamic-filter route id.
    pub dynamic_filter_id: String,
    /// Event kind: `produced`, `skipped`, `applied`, `reader_predicate_rows`,
    /// `applied_post_decode`, or `post_decode_rows`.
    pub event: String,
    /// Producer kind label.
    pub producer_kind: String,
    /// Producer plan node id.
    pub producer_node_id: String,
    /// Consumer kind label.
    pub consumer_kind: String,
    /// Consumer plan node id.
    pub consumer_node_id: String,
    /// Consumer lowering label selected during route discovery.
    pub consumer_lowering: String,
    /// Stage where the consumer can first reduce work.
    pub consumer_reduction_stage: String,
    /// Payload kind, or `none` for skipped events.
    pub filter_kind: String,
    /// Exact payload value count, when available.
    pub payload_value_count: Option<u64>,
    /// Inclusive integer-like minimum, when the payload carries one.
    pub payload_min_i64: Option<i64>,
    /// Inclusive integer-like maximum, when the payload carries one.
    pub payload_max_i64: Option<i64>,
    /// Producer output rows observed while creating the payload, when known.
    pub producer_rows: Option<u64>,
    /// Stable reason code for skipped events.
    pub skipped_reason: Option<String>,
    /// Source row groups before application.
    pub selected_row_groups_before: Option<u64>,
    /// Source row groups after application.
    pub selected_row_groups_after: Option<u64>,
    /// Source-read bytes before application.
    pub selected_source_bytes_before: Option<u64>,
    /// Source-read bytes after application.
    pub selected_source_bytes_after: Option<u64>,
    /// Source file index for source-reader row evidence.
    pub source_file_index: Option<u64>,
    /// Source chunk index for source-reader row evidence.
    pub source_chunk_index: Option<u64>,
    /// Selected source rows before the reader predicate was applied.
    pub reader_predicate_rows_before: Option<u64>,
    /// Materialized source rows after the reader predicate was applied.
    pub reader_predicate_rows_after: Option<u64>,
    /// Source rows removed by the reader predicate.
    pub reader_predicate_rows_filtered: u64,
    /// Materialized rows before a post-decode dynamic-filter stage.
    pub post_decode_rows_before: Option<u64>,
    /// Materialized rows after a post-decode dynamic-filter stage.
    pub post_decode_rows_after: Option<u64>,
    /// Materialized rows removed by a post-decode dynamic-filter stage.
    pub post_decode_rows_filtered: u64,
    /// Row groups removed by this application.
    pub row_groups_pruned: u64,
    /// Source-read bytes removed by this application.
    pub source_bytes_pruned: u64,
}

impl DynamicFilterRuntimeMetrics {
    fn produced(route: &DynamicFilterRoute, filter: &DynamicFilter, producer_rows: u64) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        Self {
            query_id: None,
            dynamic_filter_id: route.id.to_string(),
            event: "produced".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: filter.kind().to_string(),
            payload_value_count: filter.value_count().and_then(|value| u64::try_from(value).ok()),
            payload_min_i64,
            payload_max_i64,
            producer_rows: Some(producer_rows),
            skipped_reason: None,
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    fn skipped(route: &DynamicFilterRoute, reason: impl Into<String>) -> Self {
        Self {
            query_id: None,
            dynamic_filter_id: route.id.to_string(),
            event: "skipped".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: "none".to_string(),
            payload_value_count: None,
            payload_min_i64: None,
            payload_max_i64: None,
            producer_rows: None,
            skipped_reason: Some(reason.into()),
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    fn applied(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        before: DynamicFilterSourceSnapshot,
        after: DynamicFilterSourceSnapshot,
        row_groups_pruned: u64,
        source_bytes_pruned: u64,
    ) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        Self {
            query_id: None,
            dynamic_filter_id: route.id.to_string(),
            event: "applied".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: filter.kind().to_string(),
            payload_value_count: filter.value_count().and_then(|value| u64::try_from(value).ok()),
            payload_min_i64,
            payload_max_i64,
            producer_rows: None,
            skipped_reason: None,
            selected_row_groups_before: Some(before.row_groups),
            selected_row_groups_after: Some(after.row_groups),
            selected_source_bytes_before: before.source_bytes,
            selected_source_bytes_after: after.source_bytes,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned,
            source_bytes_pruned,
        }
    }

    fn source_reader_predicate_rows(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        source_file_index: Option<usize>,
        source_chunk_index: Option<usize>,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        let rows_filtered = rows_before.zip(rows_after).map_or(0, |(before, after)| before.saturating_sub(after));
        Self {
            query_id: None,
            dynamic_filter_id: route.id.to_string(),
            event: "reader_predicate_rows".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.as_str().to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: filter.kind().as_str().to_string(),
            payload_value_count: filter.value_count().map(|value| value as u64),
            payload_min_i64,
            payload_max_i64,
            producer_rows: None,
            skipped_reason: None,
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: source_file_index.map(|index| index as u64),
            source_chunk_index: source_chunk_index.map(|index| index as u64),
            reader_predicate_rows_before: rows_before,
            reader_predicate_rows_after: rows_after,
            reader_predicate_rows_filtered: rows_filtered,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    fn applied_post_decode(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event(route, filter, "applied_post_decode", rows_before, rows_after)
    }

    fn source_post_decode_stage(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event_with_stage(
            route,
            filter,
            "applied_post_decode",
            DynamicFilterConsumerReductionStage::SourcePostDecode,
            rows_before,
            rows_after,
        )
    }

    fn post_decode_rows(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event(route, filter, "post_decode_rows", rows_before, rows_after)
    }

    fn post_decode_event(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        event: &'static str,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event_with_stage(
            route,
            filter,
            event,
            route.consumer_reduction_stage(),
            rows_before,
            rows_after,
        )
    }

    fn post_decode_event_with_stage(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        event: &'static str,
        consumer_reduction_stage: DynamicFilterConsumerReductionStage,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        let rows_filtered = rows_before.zip(rows_after).map_or(0, |(before, after)| before.saturating_sub(after));
        Self {
            query_id: None,
            dynamic_filter_id: route.id.to_string(),
            event: event.to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: consumer_reduction_stage.to_string(),
            filter_kind: filter.kind().to_string(),
            payload_value_count: filter.value_count().and_then(|value| u64::try_from(value).ok()),
            payload_min_i64,
            payload_max_i64,
            producer_rows: None,
            skipped_reason: None,
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: rows_before,
            post_decode_rows_after: rows_after,
            post_decode_rows_filtered: rows_filtered,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    /// Serializes this row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "dynamic_filter_id={},event={},producer_kind={},producer_node_id={},consumer_kind={},consumer_node_id={},consumer_lowering={},consumer_reduction_stage={},filter_kind={},payload_value_count={},payload_min_i64={},payload_max_i64={},producer_rows={},skipped_reason={},selected_row_groups_before={},selected_row_groups_after={},selected_source_bytes_before={},selected_source_bytes_after={},source_file_index={},source_chunk_index={},reader_predicate_rows_before={},reader_predicate_rows_after={},reader_predicate_rows_filtered={},post_decode_rows_before={},post_decode_rows_after={},post_decode_rows_filtered={},row_groups_pruned={},source_bytes_pruned={}",
            self.dynamic_filter_id,
            self.event,
            self.producer_kind,
            self.producer_node_id,
            self.consumer_kind,
            self.consumer_node_id,
            self.consumer_lowering,
            self.consumer_reduction_stage,
            self.filter_kind,
            optional_u64(self.payload_value_count),
            optional_i64(self.payload_min_i64),
            optional_i64(self.payload_max_i64),
            optional_u64(self.producer_rows),
            optional_str(self.skipped_reason.as_deref()),
            optional_u64(self.selected_row_groups_before),
            optional_u64(self.selected_row_groups_after),
            optional_u64(self.selected_source_bytes_before),
            optional_u64(self.selected_source_bytes_after),
            optional_u64(self.source_file_index),
            optional_u64(self.source_chunk_index),
            optional_u64(self.reader_predicate_rows_before),
            optional_u64(self.reader_predicate_rows_after),
            self.reader_predicate_rows_filtered,
            optional_u64(self.post_decode_rows_before),
            optional_u64(self.post_decode_rows_after),
            self.post_decode_rows_filtered,
            self.row_groups_pruned,
            self.source_bytes_pruned,
        )
    }
}

impl QueryMetrics {
    fn record_dynamic_filter_runtime_row(&mut self, mut row: DynamicFilterRuntimeMetrics) {
        if row.query_id.is_none() {
            row.query_id = self.inherited_query_id();
        }
        self.dynamic_filter_runtime_rows.push(row);
    }

    /// Records a dynamic-filter payload produced at runtime.
    pub fn record_dynamic_filter_produced(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        producer_rows: u64,
    ) {
        self.dynamic_filter_produced_count = self.dynamic_filter_produced_count.saturating_add(1);
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::produced(route, filter, producer_rows));
    }

    /// Records a dynamic-filter producer that completed without a usable payload.
    pub fn record_dynamic_filter_skipped(&mut self, route: &DynamicFilterRoute, reason: impl Into<String>) {
        self.dynamic_filter_skipped_count = self.dynamic_filter_skipped_count.saturating_add(1);
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::skipped(route, reason));
    }

    /// Records a dynamic-filter payload applied to a local-Parquet source consumer.
    pub fn record_dynamic_filter_applied(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        before: DynamicFilterSourceSnapshot,
        after: DynamicFilterSourceSnapshot,
    ) {
        let row_groups_pruned = before.row_groups.saturating_sub(after.row_groups);
        let source_bytes_pruned = before.source_bytes.unwrap_or(0).saturating_sub(after.source_bytes.unwrap_or(0));
        self.dynamic_filter_applied_count = self.dynamic_filter_applied_count.saturating_add(1);
        self.dynamic_filter_row_groups_pruned = self.dynamic_filter_row_groups_pruned.saturating_add(row_groups_pruned);
        self.dynamic_filter_source_bytes_pruned =
            self.dynamic_filter_source_bytes_pruned.saturating_add(source_bytes_pruned);
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::applied(
            route,
            filter,
            before,
            after,
            row_groups_pruned,
            source_bytes_pruned,
        ));
    }

    /// Records a dynamic-filter payload applied to a non-source consumer.
    pub fn record_dynamic_filter_applied_post_decode(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.dynamic_filter_applied_count = self.dynamic_filter_applied_count.saturating_add(1);
        self.record_dynamic_filter_post_decode_reduction(route.consumer_reduction_stage(), rows_before, rows_after);
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::applied_post_decode(
            route,
            filter,
            rows_before,
            rows_after,
        ));
    }

    /// Records an additional post-decode source stage for a route that was
    /// already counted as applied to its source consumer.
    pub fn record_dynamic_filter_source_post_decode_stage(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.record_dynamic_filter_post_decode_reduction(
            DynamicFilterConsumerReductionStage::SourcePostDecode,
            rows_before,
            rows_after,
        );
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::source_post_decode_stage(
            route,
            filter,
            rows_before,
            rows_after,
        ));
    }

    /// Records post-decode row evidence for a route already counted as applied.
    ///
    /// Lazy stream consumers install a route once, but the row reduction is
    /// only known as chunks are consumed.
    pub fn record_dynamic_filter_post_decode_rows(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.record_dynamic_filter_post_decode_reduction(route.consumer_reduction_stage(), rows_before, rows_after);
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::post_decode_rows(
            route,
            filter,
            rows_before,
            rows_after,
        ));
    }

    /// Records source-reader row evidence for a dynamic-filter reader predicate.
    pub fn record_dynamic_filter_source_reader_predicate_rows(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        source_file_index: usize,
        source_chunk_index: usize,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::source_reader_predicate_rows(
            route,
            filter,
            Some(source_file_index),
            Some(source_chunk_index),
            rows_before,
            rows_after,
        ));
    }

    /// Records source-reader row evidence for a batched dynamic-filter reader predicate.
    pub fn record_dynamic_filter_source_reader_predicate_batch_rows(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.record_dynamic_filter_runtime_row(DynamicFilterRuntimeMetrics::source_reader_predicate_rows(
            route,
            filter,
            None,
            None,
            rows_before,
            rows_after,
        ));
    }

    /// Records de-duplicated source-reader row reduction for a chunk.
    pub fn record_dynamic_filter_reader_predicate_source_reduction(
        &mut self,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        if let Some(rows_before) = rows_before {
            self.dynamic_filter_reader_predicate_rows_before =
                self.dynamic_filter_reader_predicate_rows_before.saturating_add(rows_before);
        }
        if let Some(rows_after) = rows_after {
            self.dynamic_filter_reader_predicate_rows_after =
                self.dynamic_filter_reader_predicate_rows_after.saturating_add(rows_after);
        }
        if let Some(rows_filtered) = rows_before.zip(rows_after).map(|(before, after)| before.saturating_sub(after)) {
            self.dynamic_filter_reader_predicate_rows_filtered =
                self.dynamic_filter_reader_predicate_rows_filtered.saturating_add(rows_filtered);
        }
    }

    fn record_dynamic_filter_post_decode_reduction(
        &mut self,
        stage: DynamicFilterConsumerReductionStage,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        if let Some(rows_before) = rows_before {
            self.dynamic_filter_post_decode_rows_before =
                self.dynamic_filter_post_decode_rows_before.saturating_add(rows_before);
        }
        if let Some(rows_after) = rows_after {
            self.dynamic_filter_post_decode_rows_after =
                self.dynamic_filter_post_decode_rows_after.saturating_add(rows_after);
        }
        if let Some(rows_filtered) = rows_before.zip(rows_after).map(|(before, after)| before.saturating_sub(after)) {
            self.dynamic_filter_post_decode_rows_filtered =
                self.dynamic_filter_post_decode_rows_filtered.saturating_add(rows_filtered);
            match stage {
                DynamicFilterConsumerReductionStage::SourcePostDecode => {
                    self.dynamic_filter_source_post_decode_rows_filtered =
                        self.dynamic_filter_source_post_decode_rows_filtered.saturating_add(rows_filtered);
                },
                DynamicFilterConsumerReductionStage::FilterPostDecode => {
                    self.dynamic_filter_filter_post_decode_rows_filtered =
                        self.dynamic_filter_filter_post_decode_rows_filtered.saturating_add(rows_filtered);
                },
                DynamicFilterConsumerReductionStage::JoinProbePostDecode => {
                    self.dynamic_filter_join_probe_post_decode_rows_filtered =
                        self.dynamic_filter_join_probe_post_decode_rows_filtered.saturating_add(rows_filtered);
                },
                DynamicFilterConsumerReductionStage::SourceReader => {},
            }
        }
    }
}
