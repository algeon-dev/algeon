use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::labels::{labels_by_name, optional_usize, optional_usize_label, string_label, usize_label};

const NATIVE_TASK_OPERATOR_ROWS_METRIC_NAME: &str = "native_task_operator_rows";
const NATIVE_TASK_RUNTIME_ROWS_METRIC_NAME: &str = "native_task_runtime_rows";

/// Aggregated task metrics by native operator kind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTaskOperatorMetricRow {
    /// Operator-kind label shared by aggregated tasks.
    pub operator_kind: String,
    /// Number of tasks observed for this operator kind.
    pub task_count: usize,
    /// Cumulative elapsed nanoseconds across tasks.
    pub elapsed_ns: usize,
    /// Cumulative reservation bytes requested.
    pub reservation_requested_bytes: usize,
    /// Cumulative reservation bytes granted.
    pub reservation_granted_bytes: usize,
    /// Cumulative reservation bytes released.
    pub reservation_released_bytes: usize,
    /// Peak live reserved bytes for this operator kind.
    pub reservation_live_peak_bytes: usize,
    /// Largest lower-level observed allocator peak for this operator kind.
    pub max_observed_peak_bytes: Option<usize>,
    /// Largest lower-level retained byte count for this operator kind.
    pub max_retained_bytes: Option<usize>,
}

impl NativeTaskOperatorMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "operator_kind={},task_count={},elapsed_ns={},reservation_requested_bytes={},reservation_granted_bytes={},reservation_released_bytes={},reservation_live_peak_bytes={},max_observed_peak_bytes={},max_retained_bytes={}",
            self.operator_kind,
            self.task_count,
            self.elapsed_ns,
            self.reservation_requested_bytes,
            self.reservation_granted_bytes,
            self.reservation_released_bytes,
            self.reservation_live_peak_bytes,
            optional_usize(self.max_observed_peak_bytes),
            optional_usize(self.max_retained_bytes)
        )
    }
}

/// Runtime metric row for one compiled pipeline task.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTaskRuntimeMetricRow {
    /// Stable compiled task id.
    pub compiled_task_id: String,
    /// Operator-kind label for the task.
    pub operator_kind: String,
    /// Cumulative elapsed task time in nanoseconds.
    pub elapsed_ns: usize,
    /// Cumulative reservation bytes requested by the task.
    pub reservation_requested_bytes: usize,
    /// Cumulative reservation bytes granted to the task.
    pub reservation_granted_bytes: usize,
    /// Cumulative reservation bytes released by the task.
    pub reservation_released_bytes: usize,
    /// Peak live reserved bytes for the task.
    pub reservation_live_peak_bytes: usize,
    /// Largest lower-level observed allocator peak for the task.
    pub max_observed_peak_bytes: Option<usize>,
    /// Largest lower-level retained byte count for the task.
    pub max_retained_bytes: Option<usize>,
}

impl NativeTaskRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "compiled_task_id={},operator_kind={},elapsed_ns={},reservation_requested_bytes={},reservation_granted_bytes={},reservation_released_bytes={},reservation_live_peak_bytes={},max_observed_peak_bytes={},max_retained_bytes={}",
            self.compiled_task_id,
            self.operator_kind,
            self.elapsed_ns,
            self.reservation_requested_bytes,
            self.reservation_granted_bytes,
            self.reservation_released_bytes,
            self.reservation_live_peak_bytes,
            optional_usize(self.max_observed_peak_bytes),
            optional_usize(self.max_retained_bytes)
        )
    }
}

pub(super) fn task_operator_rows(metrics: &MetricsSet) -> Vec<NativeTaskOperatorMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_TASK_OPERATOR_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "operator_kind"))
        .map(|metric| labels_to_task_operator_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.operator_kind.cmp(&right.operator_kind));
    rows
}

pub(super) fn task_runtime_rows(metrics: &MetricsSet) -> Vec<NativeTaskRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_TASK_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "compiled_task_id"))
        .map(|metric| labels_to_task_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.compiled_task_id.cmp(&right.compiled_task_id));
    rows
}

fn labels_to_task_operator_row(labels: BTreeMap<&str, &str>) -> NativeTaskOperatorMetricRow {
    NativeTaskOperatorMetricRow {
        operator_kind: string_label(&labels, "operator_kind"),
        task_count: usize_label(&labels, "task_count"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        reservation_requested_bytes: usize_label(&labels, "reservation_requested_bytes"),
        reservation_granted_bytes: usize_label(&labels, "reservation_granted_bytes"),
        reservation_released_bytes: usize_label(&labels, "reservation_released_bytes"),
        reservation_live_peak_bytes: usize_label(&labels, "reservation_live_peak_bytes"),
        max_observed_peak_bytes: optional_usize_label(&labels, "max_observed_peak_bytes"),
        max_retained_bytes: optional_usize_label(&labels, "max_retained_bytes"),
    }
}

fn labels_to_task_runtime_row(labels: BTreeMap<&str, &str>) -> NativeTaskRuntimeMetricRow {
    NativeTaskRuntimeMetricRow {
        compiled_task_id: string_label(&labels, "compiled_task_id"),
        operator_kind: string_label(&labels, "operator_kind"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        reservation_requested_bytes: usize_label(&labels, "reservation_requested_bytes"),
        reservation_granted_bytes: usize_label(&labels, "reservation_granted_bytes"),
        reservation_released_bytes: usize_label(&labels, "reservation_released_bytes"),
        reservation_live_peak_bytes: usize_label(&labels, "reservation_live_peak_bytes"),
        max_observed_peak_bytes: optional_usize_label(&labels, "max_observed_peak_bytes"),
        max_retained_bytes: optional_usize_label(&labels, "max_retained_bytes"),
    }
}
