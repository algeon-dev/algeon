use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, Time,
};

#[derive(Debug, Clone)]
pub(super) struct CudfGpuIslandMetrics {
    pub(super) baseline: BaselineMetrics,
    pub(super) input_batches: Count,
    pub(super) input_rows: Count,
    pub(super) host_to_device_uploads: Count,
    pub(super) host_to_device_uploaded_rows: Count,
    pub(super) host_to_device_uploaded_bytes: Count,
    pub(super) device_to_host_materializations: Count,
    pub(super) device_to_host_materialized_rows: Count,
    pub(super) device_to_host_materialized_bytes: Count,
    pub(super) parquet_files: Count,
    pub(super) parquet_bytes_known: Count,
    pub(super) row_groups_selected: Count,
    pub(super) read_columns: Count,
    pub(super) parquet_read_time: Time,
    pub(super) record_batch_upload_time: Time,
    pub(super) cudf_operator_time: Time,
    pub(super) record_batch_materialization_time: Time,
}

impl CudfGpuIslandMetrics {
    pub(super) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            input_batches: summary_counter(metrics, "input_batches", partition),
            input_rows: summary_counter(metrics, "input_rows", partition),
            host_to_device_uploads: summary_counter(metrics, "host_to_device_uploads", partition),
            host_to_device_uploaded_rows: summary_counter(metrics, "host_to_device_uploaded_rows", partition),
            host_to_device_uploaded_bytes: summary_counter(metrics, "host_to_device_uploaded_bytes", partition),
            device_to_host_materializations: summary_counter(metrics, "device_to_host_materializations", partition),
            device_to_host_materialized_rows: summary_counter(metrics, "device_to_host_materialized_rows", partition),
            device_to_host_materialized_bytes: summary_counter(metrics, "device_to_host_materialized_bytes", partition),
            parquet_files: summary_counter(metrics, "parquet_files", partition),
            parquet_bytes_known: summary_counter(metrics, "parquet_bytes_known", partition),
            row_groups_selected: summary_counter(metrics, "row_groups_selected", partition),
            read_columns: summary_counter(metrics, "read_columns", partition),
            parquet_read_time: summary_timer(metrics, "parquet_read_time", partition),
            record_batch_upload_time: summary_timer(metrics, "record_batch_upload_time", partition),
            cudf_operator_time: summary_timer(metrics, "cudf_operator_time", partition),
            record_batch_materialization_time: summary_timer(metrics, "record_batch_materialization_time", partition),
        }
    }
}

fn summary_counter(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Count {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).counter(name, partition)
}

fn summary_timer(metrics: &ExecutionPlanMetricsSet, name: &'static str, partition: usize) -> Time {
    MetricBuilder::new(metrics).with_type(MetricType::SUMMARY).subset_time(name, partition)
}
