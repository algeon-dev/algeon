use std::time::Duration;

use query_planning::correlation::QueryId;

use super::{MetricsProjection, QueryMetrics, optional_u64};
use crate::observability::QueryMetricsCheckpoint;

/// Runtime evidence for one local-Parquet source read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceReadRuntimeMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Native source node id.
    pub source_node_id: String,
    /// Source read shape, for example `chunk` or `chunk_batch`.
    pub read_kind: String,
    /// Parquet read strategy used by the chunk or batch.
    pub read_strategy: String,
    /// Lower-layer Parquet read evidence reported by cudf.
    pub parquet_read_evidence: SourceReadParquetEvidenceMetrics,
    /// Batch index within the source stream, or the chunk index for singleton reads.
    pub batch_index: u64,
    /// Number of source chunks represented by this read.
    pub chunk_count: u64,
    /// First source chunk index represented by this read.
    pub first_chunk_index: u64,
    /// Last source chunk index represented by this read.
    pub last_chunk_index: u64,
    /// First source file index represented by this read.
    pub first_file_index: u64,
    /// Last source file index represented by this read.
    pub last_file_index: u64,
    /// Number of distinct files represented by this read.
    pub file_count: u64,
    /// Selected row groups represented by this read.
    pub row_group_count: u64,
    /// Number of physical read columns per represented file.
    pub read_columns_per_file: u64,
    /// Sum of physical read-column selections across represented files.
    pub read_columns_total: u64,
    /// Sum of known file bytes for represented files, when available.
    pub known_file_bytes: u64,
    /// Best available source-read byte estimate for represented chunks.
    pub estimated_source_read_bytes: u64,
    /// Sum of known selected row-group bytes, when available.
    pub known_row_group_bytes: u64,
    /// Sum of known projected compressed bytes, when available.
    pub known_projected_compressed_bytes: u64,
    /// Sum of known projected uncompressed bytes, when available.
    pub known_projected_uncompressed_bytes: u64,
    /// Sum of known pure-filter compressed bytes, when available.
    pub known_pure_filter_compressed_bytes: u64,
    /// Sum of known pure-filter uncompressed bytes, when available.
    pub known_pure_filter_uncompressed_bytes: u64,
    /// Number of output rows materialized by the read after reader predicates.
    pub output_rows: u64,
    /// Whether this read carried a source reader predicate.
    pub reader_predicate: bool,
    /// Number of dynamic-filter reader predicate routes attached to this read.
    pub reader_dynamic_filter_count: u64,
    /// Stable diagnostic reason for the source chunk batching decision.
    pub source_batching_reason: String,
    /// Source chunk planning mode requested for this source read.
    pub source_chunk_planning_mode: String,
    /// Stable diagnostic reason for the source chunk row-group and byte target.
    pub source_chunk_planning_reason: String,
    /// Elapsed time spent in the source read.
    pub elapsed: Duration,
}

impl SourceReadRuntimeMetrics {
    /// Serializes this row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "source_node_id={},read_kind={},read_strategy={},batch_index={},chunk_count={},first_chunk_index={},last_chunk_index={},first_file_index={},last_file_index={},file_count={},row_group_count={},read_columns_per_file={},read_columns_total={},known_file_bytes={},estimated_source_read_bytes={},known_row_group_bytes={},known_projected_compressed_bytes={},known_projected_uncompressed_bytes={},known_pure_filter_compressed_bytes={},known_pure_filter_uncompressed_bytes={},output_rows={},reader_predicate={},reader_dynamic_filter_count={},source_batching_reason={},source_chunk_planning_mode={},source_chunk_planning_reason={},elapsed_ns={},parquet_read_input_kind={},remote_read_mode={},parquet_read_backend={},remote_backend_build_status={},remote_read_bytes={},remote_read_range_count={},device_read_support={},parquet_pruning_input_row_groups={},parquet_row_groups_after_stats_filter={},parquet_row_groups_after_bloom_filter={},parquet_bloom_pruning_status={}",
            self.source_node_id,
            self.read_kind,
            self.read_strategy,
            self.batch_index,
            self.chunk_count,
            self.first_chunk_index,
            self.last_chunk_index,
            self.first_file_index,
            self.last_file_index,
            self.file_count,
            self.row_group_count,
            self.read_columns_per_file,
            self.read_columns_total,
            self.known_file_bytes,
            self.estimated_source_read_bytes,
            self.known_row_group_bytes,
            self.known_projected_compressed_bytes,
            self.known_projected_uncompressed_bytes,
            self.known_pure_filter_compressed_bytes,
            self.known_pure_filter_uncompressed_bytes,
            self.output_rows,
            self.reader_predicate,
            self.reader_dynamic_filter_count,
            self.source_batching_reason,
            self.source_chunk_planning_mode,
            self.source_chunk_planning_reason,
            self.elapsed.as_nanos().min(u128::from(u64::MAX)),
            self.parquet_read_evidence.input_kind,
            self.parquet_read_evidence.remote_read_mode,
            self.parquet_read_evidence.backend,
            self.parquet_read_evidence.backend_build_status,
            optional_u64(self.parquet_read_evidence.remote_bytes),
            optional_u64(self.parquet_read_evidence.remote_range_count),
            self.parquet_read_evidence.device_read_support,
            optional_u64(self.parquet_read_evidence.pruning_input_row_groups),
            optional_u64(self.parquet_read_evidence.row_groups_after_stats_filter),
            optional_u64(self.parquet_read_evidence.row_groups_after_bloom_filter),
            self.parquet_read_evidence.bloom_pruning_status,
        )
    }
}

/// Lower-layer Parquet read evidence attached to a source-read metric row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceReadParquetEvidenceMetrics {
    /// Input kind passed to the Parquet reader.
    pub input_kind: String,
    /// Requested remote read mode, or `none` for non-remote reads.
    pub remote_read_mode: String,
    /// Parquet reader backend evidence.
    pub backend: String,
    /// Build-time remote backend status reported by cudf/libcudf.
    pub backend_build_status: String,
    /// Remote bytes read, when reported by the lower layer.
    pub remote_bytes: Option<u64>,
    /// Remote range count, when reported by the lower layer.
    pub remote_range_count: Option<u64>,
    /// Direct device-read support evidence reported by the lower layer.
    pub device_read_support: String,
    /// Row groups participating in this read after any caller row-group selection.
    pub pruning_input_row_groups: Option<u64>,
    /// Row groups surviving libcudf statistics pruning, when a predicate was present.
    pub row_groups_after_stats_filter: Option<u64>,
    /// Row groups surviving libcudf Bloom-filter pruning, when a predicate was present.
    pub row_groups_after_bloom_filter: Option<u64>,
    /// Stable status explaining Bloom-pruning evidence or known ineligibility.
    pub bloom_pruning_status: String,
}

impl Default for SourceReadParquetEvidenceMetrics {
    fn default() -> Self {
        Self {
            input_kind: "none".to_string(),
            remote_read_mode: "none".to_string(),
            backend: "none".to_string(),
            backend_build_status: "none".to_string(),
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "none".to_string(),
            pruning_input_row_groups: None,
            row_groups_after_stats_filter: None,
            row_groups_after_bloom_filter: None,
            bloom_pruning_status: "not_reported".to_string(),
        }
    }
}

impl QueryMetrics {
    /// Projects a plan execution's metrics from its owner checkpoint.
    ///
    /// Every fact-driven field is replaced by the checkpoint's absolute value,
    /// so applying the same checkpoint twice leaves the value unchanged. Only
    /// the identity fields this value was created with are kept;
    /// the observation owner is the only writer of everything else.
    pub fn apply_owner_checkpoint(&mut self, checkpoint: &QueryMetricsCheckpoint) {
        let query_id = self.query_id.clone();
        let trace_id = self.trace_id.clone();
        let plan_task_mapping_datafusion_shape = self.plan_task_mapping_datafusion_shape.clone();
        *self = checkpoint.metrics.clone();
        self.query_id = query_id;
        self.trace_id = trace_id;
        self.plan_task_mapping_datafusion_shape = plan_task_mapping_datafusion_shape;
        self.apply_source_read_checkpoint(checkpoint);
        self.stamp_existing_rows_with_query_id();
    }

    /// Projects a plan execution's source-read metrics from its owner checkpoint.
    ///
    /// Every source-read field is replaced by the checkpoint's absolute value,
    /// so applying the same checkpoint twice leaves the value unchanged. The
    /// observation owner is the only writer of this family; nothing else may
    /// set these fields, or this projection would overwrite it.
    pub(crate) fn apply_source_read_checkpoint(&mut self, checkpoint: &QueryMetricsCheckpoint) {
        let totals = &checkpoint.summary.source_read;
        self.input_batches = totals.input_batches;
        self.input_rows = totals.input_rows;
        self.parquet_files = totals.parquet_files;
        self.parquet_bytes_known = totals.parquet_bytes_known;
        self.row_groups_selected = totals.row_groups_selected;
        self.read_columns = totals.read_columns;
        self.source_chunks_planned = totals.source_chunks_planned;
        self.source_chunks_attempted = totals.source_chunks_attempted;
        self.source_chunks_committed = totals.source_chunks_committed;
        self.source_chunks_retried = totals.source_chunks_retried;
        self.row_groups_planned = totals.row_groups_planned;
        self.row_groups_committed = totals.row_groups_committed;
        self.source_read_prefetches_started = totals.source_read_prefetches_started;
        self.source_read_prefetches_completed = totals.source_read_prefetches_completed;
        self.source_read_prefetches_skipped = totals.source_read_prefetches_skipped;
        self.source_read_prefetches_abandoned = totals.source_read_prefetches_abandoned;
        self.source_metadata_load_time = totals.source_metadata_load_time;
        self.source_read_queue_wait_time = totals.source_read_queue_wait_time;
        self.parquet_read_time = totals.parquet_read_time;
        self.source_read_time = totals.source_read_time;

        let query_id = self.inherited_query_id();
        self.source_read_runtime_rows = checkpoint
            .source_read_rows
            .iter()
            .cloned()
            .map(|mut row| {
                if row.query_id.is_none() {
                    row.query_id.clone_from(&query_id);
                }
                row
            })
            .collect();
        self.projection = Some(MetricsProjection {
            revision: Some(checkpoint.revision),
            sealed: checkpoint.sealed,
            completeness: checkpoint.completeness,
        });
    }

    /// Records one local-Parquet source read or source-batch read.
    pub fn record_source_read_runtime(&mut self, mut metrics: SourceReadRuntimeMetrics) {
        if metrics.query_id.is_none() {
            metrics.query_id = self.inherited_query_id();
        }
        self.source_read_runtime_rows.push(metrics);
    }

    /// Records one input batch and its row count.
    pub fn record_input_batch(&mut self, rows: u64) {
        self.input_batches += 1;
        self.input_rows += rows;
    }

    /// Records one Parquet source file observation.
    pub fn record_parquet_source(&mut self, known_bytes: Option<u64>, selected_row_groups: u64, read_columns: u64) {
        self.parquet_files += 1;
        self.parquet_bytes_known += known_bytes.unwrap_or(0);
        self.row_groups_selected += selected_row_groups;
        self.read_columns += read_columns;
    }

    /// Records that a source chunk was planned.
    pub fn record_source_chunk_planned(&mut self, row_groups: u64) {
        self.source_chunks_planned = self.source_chunks_planned.saturating_add(1);
        self.row_groups_planned = self.row_groups_planned.saturating_add(row_groups);
    }

    /// Records one source chunk attempt.
    pub fn record_source_chunk_attempt(&mut self) {
        self.source_chunks_attempted = self.source_chunks_attempted.saturating_add(1);
    }

    /// Records one committed source chunk and row-group count.
    pub fn record_source_chunk_commit(&mut self, row_groups: u64) {
        self.record_source_chunk_commit_count(1, row_groups);
    }

    /// Records committed source chunks and row-group count.
    pub fn record_source_chunk_commit_count(&mut self, chunk_count: u64, row_groups: u64) {
        self.source_chunks_committed = self.source_chunks_committed.saturating_add(chunk_count);
        self.row_groups_committed = self.row_groups_committed.saturating_add(row_groups);
    }

    /// Records one source chunk retry.
    pub fn record_source_chunk_retry(&mut self) {
        self.source_chunks_retried = self.source_chunks_retried.saturating_add(1);
    }

    /// Records a started source-read prefetch.
    pub fn record_source_read_prefetch_started(&mut self) {
        self.source_read_prefetches_started = self.source_read_prefetches_started.saturating_add(1);
    }

    /// Records a completed source-read prefetch.
    pub fn record_source_read_prefetch_completed(&mut self) {
        self.source_read_prefetches_completed = self.source_read_prefetches_completed.saturating_add(1);
    }

    /// Records a skipped source-read prefetch.
    pub fn record_source_read_prefetch_skipped(&mut self) {
        self.source_read_prefetches_skipped = self.source_read_prefetches_skipped.saturating_add(1);
    }

    /// Records an abandoned source-read prefetch.
    pub fn record_source_read_prefetch_abandoned(&mut self) {
        self.source_read_prefetches_abandoned = self.source_read_prefetches_abandoned.saturating_add(1);
    }
}

#[cfg(test)]
mod projection_tests {
    use std::time::Duration;

    use super::{QueryMetrics, SourceReadRuntimeMetrics};
    use crate::observability::{
        MetricsCaptureMode, MetricsCompleteness, QueryMetricsCheckpoint, QueryMetricsSummary, SourceReadTotals,
    };

    fn checkpoint(revision: u64, input_rows: u64, capture_mode: MetricsCaptureMode) -> QueryMetricsCheckpoint {
        QueryMetricsCheckpoint {
            revision,
            sealed: false,
            summary: QueryMetricsSummary {
                source_read: SourceReadTotals {
                    input_rows,
                    source_read_queue_wait_time: Duration::from_millis(input_rows),
                    ..SourceReadTotals::default()
                },
            },
            source_read_rows: vec![SourceReadRuntimeMetrics { batch_index: revision, ..Default::default() }],
            metrics: QueryMetrics::default(),
            completeness: MetricsCompleteness {
                capture_mode,
                detail_rows_dropped: 0,
                truncated: false,
                totals_saturated: false,
            },
        }
    }

    #[test]
    fn projecting_a_checkpoint_again_does_not_add_it_twice() {
        let checkpoint = checkpoint(4, 10, MetricsCaptureMode::Bounded);
        let mut metrics = QueryMetrics::new("projection-query");

        metrics.apply_source_read_checkpoint(&checkpoint);
        let once = metrics.clone();
        metrics.apply_source_read_checkpoint(&checkpoint);

        assert_eq!(metrics, once, "a repeated projection must be idempotent, not additive");
        assert_eq!(metrics.input_rows, 10);
        assert_eq!(metrics.source_read_runtime_rows.len(), 1);
        assert_eq!(metrics.projection.and_then(|projection| projection.revision), Some(4));
    }

    #[test]
    fn a_later_checkpoint_replaces_an_earlier_one() {
        let mut metrics = QueryMetrics::new("projection-query");

        metrics.apply_source_read_checkpoint(&checkpoint(1, 10, MetricsCaptureMode::Bounded));
        metrics.apply_source_read_checkpoint(&checkpoint(2, 25, MetricsCaptureMode::Bounded));

        assert_eq!(metrics.input_rows, 25, "a preview then a final projection must not sum");
        assert_eq!(metrics.source_read_queue_wait_time, Duration::from_millis(25));
        assert_eq!(metrics.source_read_runtime_rows[0].batch_index, 2);
    }

    #[test]
    fn projected_rows_inherit_the_query_id() {
        let mut metrics = QueryMetrics::new("projection-query");

        metrics.apply_source_read_checkpoint(&checkpoint(1, 1, MetricsCaptureMode::Bounded));

        assert_eq!(
            metrics.source_read_runtime_rows[0].query_id.as_ref().map(|id| id.as_str()),
            Some("projection-query")
        );
    }

    #[test]
    fn merging_projections_never_claims_more_than_either_side() {
        let mut bounded = QueryMetrics::new("merged-query");
        bounded.apply_source_read_checkpoint(&checkpoint(3, 1, MetricsCaptureMode::Bounded));
        let mut disabled = QueryMetrics::new("merged-query");
        disabled.apply_source_read_checkpoint(&checkpoint(7, 1, MetricsCaptureMode::Disabled));

        bounded.merge_from(disabled);

        let projection = bounded.projection.expect("both sides were projected");
        assert_eq!(projection.revision, None, "a merge of two checkpoints is no single revision");
        assert_eq!(projection.completeness.capture_mode, MetricsCaptureMode::Disabled);
        assert!(!projection.completeness.detail_is_complete());
    }

    #[test]
    fn json_reports_whether_its_rows_are_the_whole_detail() {
        let unprojected = QueryMetrics::new("json-query");
        let mut truncated = QueryMetrics::new("json-query");
        let mut lossy = checkpoint(9, 1, MetricsCaptureMode::Bounded);
        lossy.completeness.detail_rows_dropped = 3;
        lossy.completeness.truncated = true;
        truncated.apply_source_read_checkpoint(&lossy);
        let mut not_captured = QueryMetrics::new("json-query");
        not_captured.apply_source_read_checkpoint(&checkpoint(2, 1, MetricsCaptureMode::Disabled));

        assert_eq!(unprojected.to_json()["completeness"]["status"], "unknown");
        let completeness = &truncated.to_json()["completeness"];
        assert_eq!(completeness["status"], "incomplete");
        assert_eq!(completeness["revision"], 9);
        assert_eq!(completeness["detail_rows_dropped"], 3);
        assert_eq!(not_captured.to_json()["completeness"]["status"], "not_captured");
    }

    #[test]
    fn an_unprojected_side_does_not_erase_a_projection() {
        let mut accumulated = QueryMetrics::new("merged-query");
        let mut stage = QueryMetrics::new("merged-query");
        stage.apply_source_read_checkpoint(&checkpoint(5, 1, MetricsCaptureMode::Bounded));

        accumulated.merge_from(stage);

        assert_eq!(accumulated.projection.and_then(|projection| projection.revision), Some(5));
    }
}
