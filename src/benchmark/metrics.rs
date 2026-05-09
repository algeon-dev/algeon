use std::time::Duration;

use datafusion::physical_plan::metrics::MetricsSet;

use super::timing::duration_from_nanos;

pub(super) const PHASE_C_MEMORY_CONTROL_UNBOUNDED: &str = "unbounded";
pub(super) const PHASE_C_RUNTIME_GATE_NOT_EVALUATED: &str = "not_evaluated";
pub(super) const PHASE_C_MEMORY_CONTROL_BOUNDED: &str = "bounded";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargerIslandBenchmarkMetrics {
    pub phase_c_memory_control_policy: &'static str,
    pub phase_c_runtime_gate: &'static str,
    pub input_batches: usize,
    pub input_rows: usize,
    pub resident_dataframe_peak_bytes: Option<usize>,
    pub allocator_high_water_bytes: Option<usize>,
    pub reservation_requested_bytes: usize,
    pub reservation_granted_bytes: usize,
    pub reservation_denied_count: usize,
    pub source_chunks_planned: usize,
    pub source_chunks_attempted: usize,
    pub source_chunks_committed: usize,
    pub source_chunks_retried: usize,
    pub row_groups_planned: usize,
    pub row_groups_committed: usize,
    pub partial_aggregate_count: usize,
    pub partial_aggregate_rows: usize,
    pub final_merge_attempts: usize,
    pub final_merge_levels: usize,
    pub retry_count: usize,
    pub retry_resource_exhausted_count: usize,
    pub retry_reservation_failure_count: usize,
    pub peak_device_memory_bytes: Option<usize>,
    pub host_to_device_uploads: usize,
    pub host_to_device_uploaded_rows: usize,
    pub host_to_device_uploaded_bytes: usize,
    pub device_to_host_materializations: usize,
    pub device_to_host_materialized_rows: usize,
    pub device_to_host_materialized_bytes: usize,
    pub parquet_files: usize,
    pub parquet_bytes_known: usize,
    pub row_groups_selected: usize,
    pub read_columns: usize,
    pub parquet_read_time: Duration,
    pub record_batch_upload_time: Duration,
    pub cudf_operator_time: Duration,
    pub record_batch_materialization_time: Duration,
    pub elapsed_compute: Option<Duration>,
    pub output_rows: Option<usize>,
}

impl LargerIslandBenchmarkMetrics {
    pub fn from_metrics(metrics: &MetricsSet) -> Self {
        Self {
            phase_c_memory_control_policy: PHASE_C_MEMORY_CONTROL_UNBOUNDED,
            phase_c_runtime_gate: PHASE_C_RUNTIME_GATE_NOT_EVALUATED,
            input_batches: metric_count(metrics, "input_batches"),
            input_rows: metric_count(metrics, "input_rows"),
            resident_dataframe_peak_bytes: None,
            allocator_high_water_bytes: None,
            reservation_requested_bytes: 0,
            reservation_granted_bytes: 0,
            reservation_denied_count: 0,
            source_chunks_planned: 0,
            source_chunks_attempted: 0,
            source_chunks_committed: 0,
            source_chunks_retried: 0,
            row_groups_planned: 0,
            row_groups_committed: 0,
            partial_aggregate_count: 0,
            partial_aggregate_rows: 0,
            final_merge_attempts: 0,
            final_merge_levels: 0,
            retry_count: 0,
            retry_resource_exhausted_count: 0,
            retry_reservation_failure_count: 0,
            peak_device_memory_bytes: None,
            host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
            host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
            host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
            device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
            device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
            device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
            parquet_files: metric_count(metrics, "parquet_files"),
            parquet_bytes_known: metric_count(metrics, "parquet_bytes_known"),
            row_groups_selected: metric_count(metrics, "row_groups_selected"),
            read_columns: metric_count(metrics, "read_columns"),
            parquet_read_time: metric_duration(metrics, "parquet_read_time"),
            record_batch_upload_time: metric_duration(metrics, "record_batch_upload_time"),
            cudf_operator_time: metric_duration(metrics, "cudf_operator_time"),
            record_batch_materialization_time: metric_duration(metrics, "record_batch_materialization_time"),
            elapsed_compute: metrics.elapsed_compute().map(duration_from_nanos),
            output_rows: metrics.output_rows(),
        }
    }

    pub fn assess_phase_c_runtime_gate(&self, output_equality_verified: bool) -> PhaseCRuntimeGateStatus {
        if self.phase_c_memory_control_policy == PHASE_C_MEMORY_CONTROL_UNBOUNDED {
            return PhaseCRuntimeGateStatus::NotEvaluated;
        }
        if self.phase_c_memory_control_policy != PHASE_C_MEMORY_CONTROL_BOUNDED {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if !output_equality_verified {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.host_to_device_uploads != 0
            || self.host_to_device_uploaded_rows != 0
            || self.host_to_device_uploaded_bytes != 0
        {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.device_to_host_materializations != 1 {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.source_chunks_planned == 0
            || self.source_chunks_committed == 0
            || self.source_chunks_committed != self.source_chunks_planned
            || self.row_groups_planned == 0
            || self.row_groups_committed != self.row_groups_planned
        {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.resident_dataframe_peak_bytes.is_none() {
            return PhaseCRuntimeGateStatus::Fail;
        }

        PhaseCRuntimeGateStatus::Pass
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseCRuntimeGateStatus {
    Pass,
    Fail,
    NotEvaluated,
}

impl PhaseCRuntimeGateStatus {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::NotEvaluated => PHASE_C_RUNTIME_GATE_NOT_EVALUATED,
        }
    }
}

fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

fn metric_duration(metrics: &MetricsSet, name: &str) -> Duration {
    duration_from_nanos(metric_count(metrics, name))
}
