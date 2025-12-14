use std::time::Duration;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    pub boundary: BoundaryMetrics,
    pub input_batches: u64,
    pub input_rows: u64,
    pub parquet_files: u64,
    pub parquet_bytes_known: u64,
    pub row_groups_selected: u64,
    pub read_columns: u64,
    pub parquet_read_time: Duration,
    pub operator_time: Duration,
    pub materialization_time: Duration,
}

impl QueryMetrics {
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
