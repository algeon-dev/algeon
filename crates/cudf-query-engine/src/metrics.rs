use std::time::Duration;

use crate::runtime::MemoryControlPolicy;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    pub boundary: BoundaryMetrics,
    pub operators: Vec<OperatorMetrics>,
    pub phase_c_memory_control_policy: MemoryControlPolicy,
    pub input_batches: u64,
    pub input_rows: u64,
    /// Backward-compatible alias for `resident_dataframe_peak_bytes`.
    ///
    /// This is not an allocator high-water mark.
    pub peak_device_memory_bytes: Option<u64>,
    pub resident_dataframe_peak_bytes: Option<u64>,
    pub allocator_high_water_bytes: Option<u64>,
    pub reservation_requested_bytes: u64,
    pub reservation_granted_bytes: u64,
    pub reservation_denied_count: u64,
    pub source_chunks_planned: u64,
    pub source_chunks_attempted: u64,
    pub source_chunks_committed: u64,
    pub source_chunks_retried: u64,
    pub row_groups_planned: u64,
    pub row_groups_committed: u64,
    pub partial_aggregate_count: u64,
    pub partial_aggregate_rows: u64,
    pub final_merge_attempts: u64,
    pub final_merge_levels: u64,
    pub retry_count: u64,
    pub retry_resource_exhausted_count: u64,
    pub retry_reservation_failure_count: u64,
    pub parquet_files: u64,
    pub parquet_bytes_known: u64,
    pub row_groups_selected: u64,
    pub read_columns: u64,
    pub parquet_read_time: Duration,
    pub source_read_time: Duration,
    pub operator_time: Duration,
    pub chunk_operator_time: Duration,
    pub final_merge_time: Duration,
    pub final_sort_time: Duration,
    pub materialization_time: Duration,
}

impl QueryMetrics {
    pub fn record_operator_metrics(&mut self, metrics: OperatorMetrics) {
        self.operators.push(metrics);
    }

    pub fn record_input_batch(&mut self, rows: u64) {
        self.input_batches += 1;
        self.input_rows += rows;
    }

    pub fn record_parquet_source(&mut self, known_bytes: Option<u64>, selected_row_groups: u64, read_columns: u64) {
        self.parquet_files += 1;
        self.parquet_bytes_known += known_bytes.unwrap_or(0);
        self.row_groups_selected += selected_row_groups;
        self.read_columns += read_columns;
    }

    pub fn record_device_memory_observation(&mut self, bytes: u64) {
        self.peak_device_memory_bytes = Some(self.peak_device_memory_bytes.map_or(bytes, |peak| peak.max(bytes)));
        self.resident_dataframe_peak_bytes =
            Some(self.resident_dataframe_peak_bytes.map_or(bytes, |peak| peak.max(bytes)));
    }

    pub fn record_reservation_granted(&mut self, requested_bytes: u64, granted_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(granted_bytes);
    }

    pub fn record_reservation_denied(&mut self, requested_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(1);
    }

    pub fn record_source_chunk_planned(&mut self, row_groups: u64) {
        self.source_chunks_planned = self.source_chunks_planned.saturating_add(1);
        self.row_groups_planned = self.row_groups_planned.saturating_add(row_groups);
    }

    pub fn record_source_chunk_attempt(&mut self) {
        self.source_chunks_attempted = self.source_chunks_attempted.saturating_add(1);
    }

    pub fn record_source_chunk_commit(&mut self, row_groups: u64) {
        self.source_chunks_committed = self.source_chunks_committed.saturating_add(1);
        self.row_groups_committed = self.row_groups_committed.saturating_add(row_groups);
    }

    pub fn record_source_chunk_retry(&mut self) {
        self.source_chunks_retried = self.source_chunks_retried.saturating_add(1);
    }

    pub fn record_retry_resource_exhausted(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_resource_exhausted_count = self.retry_resource_exhausted_count.saturating_add(1);
    }

    pub fn record_retry_reservation_failure(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_reservation_failure_count = self.retry_reservation_failure_count.saturating_add(1);
    }

    pub fn record_partial_aggregate(&mut self, rows: u64) {
        self.partial_aggregate_count = self.partial_aggregate_count.saturating_add(1);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(rows);
    }

    pub fn record_final_merge_attempt(&mut self) {
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(1);
    }

    pub fn record_final_merge_level(&mut self) {
        self.final_merge_levels = self.final_merge_levels.saturating_add(1);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperatorMetrics {
    pub node_id: String,
    pub operator_kind: String,
    pub input_rows: u64,
    pub input_bytes: u64,
    pub output_rows: u64,
    pub output_bytes: u64,
    pub join: Option<JoinOperatorMetrics>,
}

impl OperatorMetrics {
    pub fn join(node_id: impl ToString, join: JoinOperatorMetrics) -> Self {
        Self { node_id: node_id.to_string(), operator_kind: "join".to_string(), join: Some(join), ..Default::default() }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinOperatorMetrics {
    pub join_type: String,
    pub key_count: usize,
    pub residual_predicate: bool,
    pub null_equality: String,
    pub build_side: String,
    pub probe_side: String,
    pub build_rows: u64,
    pub build_bytes: u64,
    pub probe_rows: u64,
    pub probe_bytes: u64,
    pub hash_table_bytes: u64,
    pub output_rows: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundaryMetrics {
    pub host_to_device_uploads: u64,
    pub host_to_device_uploaded_rows: u64,
    pub host_to_device_uploaded_bytes: u64,
    pub device_to_host_materializations: u64,
    pub device_to_host_materialized_rows: u64,
    pub device_to_host_materialized_bytes: u64,
    pub device_to_device_copies: u64,
    pub device_to_device_copied_bytes: u64,
}

impl BoundaryMetrics {
    pub fn record_host_to_device_upload(&mut self, rows: u64, bytes: u64) {
        self.host_to_device_uploads += 1;
        self.host_to_device_uploaded_rows += rows;
        self.host_to_device_uploaded_bytes += bytes;
    }

    pub fn record_device_to_host_materialization(&mut self, rows: u64, bytes: u64) {
        self.device_to_host_materializations += 1;
        self.device_to_host_materialized_rows += rows;
        self.device_to_host_materialized_bytes += bytes;
    }

    pub fn record_device_to_device_copy(&mut self, bytes: u64) {
        self.device_to_device_copies += 1;
        self.device_to_device_copied_bytes += bytes;
    }
}
