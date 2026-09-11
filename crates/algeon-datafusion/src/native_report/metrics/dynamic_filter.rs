use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{
    labels_by_name, optional_i64, optional_i64_label, optional_str, optional_string_label, optional_usize,
    optional_usize_label, string_label, usize_label,
};

const NATIVE_DYNAMIC_FILTER_RUNTIME_ROWS_METRIC_NAME: &str = "native_dynamic_filter_runtime_rows";

/// Stable query-level classification of measured dynamic-filter reduction.
///
/// The value describes observed runtime reduction, not whether a filter route
/// was produced or reached a consumer. The string values are public TSV
/// sentinels and should remain stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDynamicFilterMeasuredReductionStatus {
    /// No dynamic-filter runtime evidence was observed.
    Unknown,
    /// Dynamic filters removed source row groups or source-read bytes.
    MeasuredSourcePruned,
    /// Dynamic reader predicates reduced materialized source rows.
    MeasuredReaderRowsReduced,
    /// Post-decode dynamic filters reduced rows after source decode.
    MeasuredPostDecodeReduced,
    /// Dynamic-filter routes ran or were considered but no measured reduction was observed.
    MeasuredNoReduction,
}

impl NativeDynamicFilterMeasuredReductionStatus {
    /// Stable TSV label for this status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::MeasuredSourcePruned => "measured_source_pruned",
            Self::MeasuredReaderRowsReduced => "measured_reader_rows_reduced",
            Self::MeasuredPostDecodeReduced => "measured_post_decode_reduced",
            Self::MeasuredNoReduction => "measured_no_reduction",
        }
    }
}

/// Stable query-level classification of which work dynamic filters avoided.
///
/// This is stricter than [`NativeDynamicFilterMeasuredReductionStatus`]:
/// it preserves where the strongest observed reduction occurred so admission
/// and gate code can avoid treating route lifecycle events as wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDynamicFilterWorkAvoidanceStatus {
    /// No dynamic-filter runtime evidence was observed.
    Unknown,
    /// Source row groups or source bytes were pruned before read/decode work.
    SourceReadPruned,
    /// Source-reader predicates reduced materialized source rows.
    SourceReaderRowsReduced,
    /// Source-adjacent post-decode filters reduced rows before downstream operators.
    SourcePostDecodeRowsReduced,
    /// Filter-operator post-decode filters reduced rows before downstream operators.
    FilterPostDecodeRowsReduced,
    /// Hash-join-probe post-decode filters reduced rows at the probe boundary.
    JoinProbePostDecodeRowsReduced,
    /// More than one post-decode consumer stage reduced rows.
    MixedPostDecodeRowsReduced,
    /// Dynamic-filter routes ran or were considered but no measured reduction was observed.
    MeasuredNoReduction,
}

impl NativeDynamicFilterWorkAvoidanceStatus {
    /// Stable TSV label for this status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::SourceReadPruned => "source_read_pruned",
            Self::SourceReaderRowsReduced => "source_reader_rows_reduced",
            Self::SourcePostDecodeRowsReduced => "source_post_decode_rows_reduced",
            Self::FilterPostDecodeRowsReduced => "filter_post_decode_rows_reduced",
            Self::JoinProbePostDecodeRowsReduced => "join_probe_post_decode_rows_reduced",
            Self::MixedPostDecodeRowsReduced => "mixed_post_decode_rows_reduced",
            Self::MeasuredNoReduction => "measured_no_reduction",
        }
    }
}

/// Stable query-level classification of where post-decode row reduction occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDynamicFilterPostDecodeReductionStage {
    /// No post-decode row reduction was measured.
    None,
    /// Post-decode reduction happened at a local source after decode.
    SourcePostDecode,
    /// Post-decode reduction happened at a filter operator before downstream work.
    FilterPostDecode,
    /// Post-decode reduction happened at a hash-join probe boundary.
    JoinProbePostDecode,
    /// More than one post-decode consumer stage reduced rows.
    MixedPostDecode,
    /// Top-level counters show reduction but nested rows lack usable stage evidence.
    Unknown,
}

impl NativeDynamicFilterPostDecodeReductionStage {
    /// Stable TSV label for this stage.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SourcePostDecode => "source_post_decode",
            Self::FilterPostDecode => "filter_post_decode",
            Self::JoinProbePostDecode => "join_probe_post_decode",
            Self::MixedPostDecode => "mixed_post_decode",
            Self::Unknown => "unknown",
        }
    }
}

/// Stable nested metric row for one dynamic-filter runtime event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeDynamicFilterRuntimeMetricRow {
    /// Plan-local dynamic-filter route id.
    pub dynamic_filter_id: String,
    /// Event kind, such as `produced`, `skipped`, `applied`,
    /// `reader_predicate_rows`, or `applied_post_decode`.
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
    pub payload_value_count: Option<usize>,
    /// Inclusive integer-like minimum, when present.
    pub payload_min_i64: Option<i64>,
    /// Inclusive integer-like maximum, when present.
    pub payload_max_i64: Option<i64>,
    /// Producer rows observed while creating the payload, when known.
    pub producer_rows: Option<usize>,
    /// Stable reason for skipped events.
    pub skipped_reason: Option<String>,
    /// Source row groups before application.
    pub selected_row_groups_before: Option<usize>,
    /// Source row groups after application.
    pub selected_row_groups_after: Option<usize>,
    /// Source-read bytes before application.
    pub selected_source_bytes_before: Option<usize>,
    /// Source-read bytes after application.
    pub selected_source_bytes_after: Option<usize>,
    /// Source file index for source-reader row evidence.
    pub source_file_index: Option<usize>,
    /// Source chunk index for source-reader row evidence.
    pub source_chunk_index: Option<usize>,
    /// Selected source rows before the reader predicate was applied.
    pub reader_predicate_rows_before: Option<usize>,
    /// Materialized source rows after the reader predicate was applied.
    pub reader_predicate_rows_after: Option<usize>,
    /// Source rows removed by the reader predicate.
    pub reader_predicate_rows_filtered: usize,
    /// Materialized rows before a post-decode dynamic-filter stage.
    pub post_decode_rows_before: Option<usize>,
    /// Materialized rows after a post-decode dynamic-filter stage.
    pub post_decode_rows_after: Option<usize>,
    /// Materialized rows removed by a post-decode dynamic-filter stage.
    pub post_decode_rows_filtered: usize,
    /// Row groups removed by this application.
    pub row_groups_pruned: usize,
    /// Source-read bytes removed by this application.
    pub source_bytes_pruned: usize,
}

impl NativeDynamicFilterRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
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
            optional_usize(self.payload_value_count),
            optional_i64(self.payload_min_i64),
            optional_i64(self.payload_max_i64),
            optional_usize(self.producer_rows),
            optional_str(self.skipped_reason.as_deref()),
            optional_usize(self.selected_row_groups_before),
            optional_usize(self.selected_row_groups_after),
            optional_usize(self.selected_source_bytes_before),
            optional_usize(self.selected_source_bytes_after),
            optional_usize(self.source_file_index),
            optional_usize(self.source_chunk_index),
            optional_usize(self.reader_predicate_rows_before),
            optional_usize(self.reader_predicate_rows_after),
            self.reader_predicate_rows_filtered,
            optional_usize(self.post_decode_rows_before),
            optional_usize(self.post_decode_rows_after),
            self.post_decode_rows_filtered,
            self.row_groups_pruned,
            self.source_bytes_pruned,
        )
    }
}

pub(super) fn dynamic_filter_runtime_rows(metrics: &MetricsSet) -> Vec<NativeDynamicFilterRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_DYNAMIC_FILTER_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "dynamic_filter_id"))
        .map(|metric| labels_to_dynamic_filter_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows
}

fn labels_to_dynamic_filter_runtime_row(labels: BTreeMap<&str, &str>) -> NativeDynamicFilterRuntimeMetricRow {
    NativeDynamicFilterRuntimeMetricRow {
        dynamic_filter_id: string_label(&labels, "dynamic_filter_id"),
        event: string_label(&labels, "event"),
        producer_kind: string_label(&labels, "producer_kind"),
        producer_node_id: string_label(&labels, "producer_node_id"),
        consumer_kind: string_label(&labels, "consumer_kind"),
        consumer_node_id: string_label(&labels, "consumer_node_id"),
        consumer_lowering: string_label(&labels, "consumer_lowering"),
        consumer_reduction_stage: string_label(&labels, "consumer_reduction_stage"),
        filter_kind: string_label(&labels, "filter_kind"),
        payload_value_count: optional_usize_label(&labels, "payload_value_count"),
        payload_min_i64: optional_i64_label(&labels, "payload_min_i64"),
        payload_max_i64: optional_i64_label(&labels, "payload_max_i64"),
        producer_rows: optional_usize_label(&labels, "producer_rows"),
        skipped_reason: optional_string_label(&labels, "skipped_reason"),
        selected_row_groups_before: optional_usize_label(&labels, "selected_row_groups_before"),
        selected_row_groups_after: optional_usize_label(&labels, "selected_row_groups_after"),
        selected_source_bytes_before: optional_usize_label(&labels, "selected_source_bytes_before"),
        selected_source_bytes_after: optional_usize_label(&labels, "selected_source_bytes_after"),
        source_file_index: optional_usize_label(&labels, "source_file_index"),
        source_chunk_index: optional_usize_label(&labels, "source_chunk_index"),
        reader_predicate_rows_before: optional_usize_label(&labels, "reader_predicate_rows_before"),
        reader_predicate_rows_after: optional_usize_label(&labels, "reader_predicate_rows_after"),
        reader_predicate_rows_filtered: usize_label(&labels, "reader_predicate_rows_filtered"),
        post_decode_rows_before: optional_usize_label(&labels, "post_decode_rows_before"),
        post_decode_rows_after: optional_usize_label(&labels, "post_decode_rows_after"),
        post_decode_rows_filtered: usize_label(&labels, "post_decode_rows_filtered"),
        row_groups_pruned: usize_label(&labels, "row_groups_pruned"),
        source_bytes_pruned: usize_label(&labels, "source_bytes_pruned"),
    }
}
