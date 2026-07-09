use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{
    bool_label, labels_by_name, optional_str, optional_string_label, optional_usize, optional_usize_label,
    string_label, usize_label,
};

const NATIVE_GRAPH_RUNTIME_ROWS_METRIC_NAME: &str = "native_graph_runtime_rows";
const NATIVE_GRAPH_RESERVATION_ROWS_METRIC_NAME: &str = "native_graph_reservation_rows";

/// Stable nested metric row for one graph memory-tier admission.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeGraphReservationMetricRow {
    /// Native graph algorithm node id.
    pub node_id: String,
    /// Stable graph algorithm label.
    pub algorithm: String,
    /// Graph memory class label.
    pub class: String,
    /// Admission outcome label.
    pub outcome: String,
    /// Raw edge rows before explicit edge normalization, if known.
    pub raw_input_edges: Option<usize>,
    /// Effective graph-stage edge rows after null filtering and normalization.
    pub input_edges: usize,
    /// Conservative vertex upper bound used by the model.
    pub vertex_upper_bound: usize,
    /// Vertex count signal actually used by the model.
    pub vertex_estimate: usize,
    /// Provenance for the vertex estimate.
    pub vertex_estimate_source: String,
    /// Vertex id width in bytes.
    pub vertex_width_bytes: usize,
    /// Consumed edge-weight width in bytes, if present.
    pub weight_width_bytes: Option<usize>,
    /// Whether a stable edge id column was bound.
    pub has_edge_id: bool,
    /// Whether the graph was treated as directed.
    pub directed: bool,
    /// Whether cuGraph renumbering was enabled.
    pub renumber: bool,
    /// Graph construction policy label.
    pub construction_policy: String,
    /// Whether an explicit edge-normalize operator fed this graph stage.
    pub edge_normalize_inserted: bool,
    /// Topology family used by calibrated graph memory pricing.
    pub topology_family: String,
    /// Provenance for non-default graph calibration inputs.
    pub calibration_source: String,
    /// Seed count for seed-dependent algorithms, when applicable.
    pub seed_count: Option<usize>,
    /// Seed-selection mode for seed-dependent algorithms, or `none`.
    pub seed_mode: String,
    /// Whether the estimate is flagged as data-dependent.
    pub data_dependent: bool,
    /// Raw graph model ask before bounded reservation floors.
    pub model_requested_bytes: usize,
    /// Requested graph-tier bytes.
    pub requested_bytes: usize,
    /// Bounded device budget bytes, when bounded.
    pub budget_bytes: Option<usize>,
    /// Model ask divided by measured graph-stage peak, when observed.
    pub graph_model_error_ratio: Option<String>,
    /// Stable denial reason or `none`.
    pub reason: String,
}

impl NativeGraphReservationMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},algorithm={},class={},outcome={},raw_input_edges={},input_edges={},vertex_upper_bound={},vertex_estimate={},vertex_estimate_source={},vertex_width_bytes={},weight_width_bytes={},has_edge_id={},directed={},renumber={},construction_policy={},edge_normalize_inserted={},topology_family={},calibration_source={},seed_count={},seed_mode={},data_dependent={},model_requested_bytes={},requested_bytes={},budget_bytes={},graph_model_error_ratio={},reason={}",
            self.node_id,
            self.algorithm,
            self.class,
            self.outcome,
            optional_usize(self.raw_input_edges),
            self.input_edges,
            self.vertex_upper_bound,
            self.vertex_estimate,
            self.vertex_estimate_source,
            self.vertex_width_bytes,
            optional_usize(self.weight_width_bytes),
            self.has_edge_id,
            self.directed,
            self.renumber,
            self.construction_policy,
            self.edge_normalize_inserted,
            self.topology_family,
            self.calibration_source,
            optional_usize(self.seed_count),
            self.seed_mode,
            self.data_dependent,
            self.model_requested_bytes,
            self.requested_bytes,
            optional_usize(self.budget_bytes),
            optional_str(self.graph_model_error_ratio.as_deref()),
            self.reason,
        )
    }
}

/// Stable nested metric row for one graph algorithm execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeGraphRuntimeMetricRow {
    /// Native graph algorithm node id.
    pub node_id: String,
    /// Stable graph algorithm label.
    pub algorithm: String,
    /// Runtime outcome label.
    pub runtime_outcome: String,
    /// Input boundary label.
    pub input_boundary: String,
    /// Output boundary label.
    pub output_boundary: String,
    /// Graph construction policy label.
    pub construction_policy: String,
    /// Source vertex column name.
    pub source_column: String,
    /// Destination vertex column name.
    pub destination_column: String,
    /// Optional edge-weight column name.
    pub weight_column: Option<String>,
    /// Whether the graph was treated as directed.
    pub directed: bool,
    /// Whether cuGraph renumbering was enabled.
    pub renumber: bool,
    /// Stable serialized algorithm parameters.
    pub algorithm_parameters: String,
    /// Stable serialized algorithm metadata.
    pub algorithm_metadata: String,
    /// Input row count.
    pub input_rows: usize,
    /// Input column count.
    pub input_columns: usize,
    /// Output row count.
    pub output_rows: usize,
    /// Output column count.
    pub output_columns: usize,
    /// Output device bytes, when known.
    pub output_device_bytes: Option<usize>,
    /// Graph-stage allocator high-water bytes, when observed.
    pub graph_peak_bytes: Option<usize>,
    /// Runtime elapsed nanoseconds.
    pub elapsed_ns: usize,
    /// GPU-resident input handoffs for this algorithm.
    pub gpu_resident_input_handoffs: usize,
    /// Host-to-device uploads for this algorithm.
    pub host_to_device_uploads: usize,
    /// Device-to-host materializations for this algorithm.
    pub device_to_host_materializations: usize,
}

impl NativeGraphRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},algorithm={},runtime_outcome={},input_boundary={},output_boundary={},construction_policy={},source_column={},destination_column={},weight_column={},directed={},renumber={},algorithm_parameters={},algorithm_metadata={},input_rows={},input_columns={},output_rows={},output_columns={},output_device_bytes={},graph_peak_bytes={},elapsed_ns={},gpu_resident_input_handoffs={},host_to_device_uploads={},device_to_host_materializations={}",
            self.node_id,
            self.algorithm,
            self.runtime_outcome,
            self.input_boundary,
            self.output_boundary,
            self.construction_policy,
            self.source_column,
            self.destination_column,
            optional_str(self.weight_column.as_deref()),
            self.directed,
            self.renumber,
            self.algorithm_parameters,
            self.algorithm_metadata,
            self.input_rows,
            self.input_columns,
            self.output_rows,
            self.output_columns,
            optional_usize(self.output_device_bytes),
            optional_usize(self.graph_peak_bytes),
            self.elapsed_ns,
            self.gpu_resident_input_handoffs,
            self.host_to_device_uploads,
            self.device_to_host_materializations,
        )
    }
}

pub(super) fn graph_runtime_rows(metrics: &MetricsSet) -> Vec<NativeGraphRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_GRAPH_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "algorithm"))
        .map(|metric| labels_to_graph_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.node_id
            .cmp(&right.node_id)
            .then_with(|| left.algorithm.cmp(&right.algorithm))
            .then_with(|| left.source_column.cmp(&right.source_column))
            .then_with(|| left.destination_column.cmp(&right.destination_column))
    });
    rows
}

pub(super) fn graph_reservation_rows(metrics: &MetricsSet) -> Vec<NativeGraphReservationMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_GRAPH_RESERVATION_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "algorithm"))
        .map(|metric| labels_to_graph_reservation_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.node_id
            .cmp(&right.node_id)
            .then_with(|| left.algorithm.cmp(&right.algorithm))
            .then_with(|| left.to_stable_kv_row().cmp(&right.to_stable_kv_row()))
    });
    rows
}

fn labels_to_graph_runtime_row(labels: BTreeMap<&str, &str>) -> NativeGraphRuntimeMetricRow {
    NativeGraphRuntimeMetricRow {
        node_id: string_label(&labels, "node_id"),
        algorithm: string_label(&labels, "algorithm"),
        runtime_outcome: string_label(&labels, "runtime_outcome"),
        input_boundary: string_label(&labels, "input_boundary"),
        output_boundary: string_label(&labels, "output_boundary"),
        construction_policy: string_label(&labels, "construction_policy"),
        source_column: string_label(&labels, "source_column"),
        destination_column: string_label(&labels, "destination_column"),
        weight_column: optional_string_label(&labels, "weight_column"),
        directed: bool_label(&labels, "directed"),
        renumber: bool_label(&labels, "renumber"),
        algorithm_parameters: string_label(&labels, "algorithm_parameters"),
        algorithm_metadata: string_label(&labels, "algorithm_metadata"),
        input_rows: usize_label(&labels, "input_rows"),
        input_columns: usize_label(&labels, "input_columns"),
        output_rows: usize_label(&labels, "output_rows"),
        output_columns: usize_label(&labels, "output_columns"),
        output_device_bytes: optional_usize_label(&labels, "output_device_bytes"),
        graph_peak_bytes: optional_usize_label(&labels, "graph_peak_bytes"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        gpu_resident_input_handoffs: usize_label(&labels, "gpu_resident_input_handoffs"),
        host_to_device_uploads: usize_label(&labels, "host_to_device_uploads"),
        device_to_host_materializations: usize_label(&labels, "device_to_host_materializations"),
    }
}

fn labels_to_graph_reservation_row(labels: BTreeMap<&str, &str>) -> NativeGraphReservationMetricRow {
    NativeGraphReservationMetricRow {
        node_id: string_label(&labels, "node_id"),
        algorithm: string_label(&labels, "algorithm"),
        class: string_label(&labels, "class"),
        outcome: string_label(&labels, "outcome"),
        raw_input_edges: optional_usize_label(&labels, "raw_input_edges"),
        input_edges: usize_label(&labels, "input_edges"),
        vertex_upper_bound: usize_label(&labels, "vertex_upper_bound"),
        vertex_estimate: usize_label(&labels, "vertex_estimate"),
        vertex_estimate_source: string_label(&labels, "vertex_estimate_source"),
        vertex_width_bytes: usize_label(&labels, "vertex_width_bytes"),
        weight_width_bytes: optional_usize_label(&labels, "weight_width_bytes"),
        has_edge_id: bool_label(&labels, "has_edge_id"),
        directed: bool_label(&labels, "directed"),
        renumber: bool_label(&labels, "renumber"),
        construction_policy: string_label(&labels, "construction_policy"),
        edge_normalize_inserted: bool_label(&labels, "edge_normalize_inserted"),
        topology_family: string_label(&labels, "topology_family"),
        calibration_source: string_label(&labels, "calibration_source"),
        seed_count: optional_usize_label(&labels, "seed_count"),
        seed_mode: string_label(&labels, "seed_mode"),
        data_dependent: bool_label(&labels, "data_dependent"),
        model_requested_bytes: usize_label(&labels, "model_requested_bytes"),
        requested_bytes: usize_label(&labels, "requested_bytes"),
        budget_bytes: optional_usize_label(&labels, "budget_bytes"),
        graph_model_error_ratio: optional_string_label(&labels, "graph_model_error_ratio"),
        reason: string_label(&labels, "reason"),
    }
}
