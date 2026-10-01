use serde_json::{Value, json};

use super::QueryMetrics;

/// Physical unit or logical value class for a diagnostic metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricUnit {
    /// Unitless count.
    Count,
    /// Row count.
    Rows,
    /// Bytes resident on or allocated from device memory.
    DeviceBytes,
    /// Bytes resident in host memory.
    HostBytes,
    /// Parquet compressed bytes.
    ParquetBytes,
    /// Time in nanoseconds.
    Nanos,
    /// Time in milliseconds.
    Millis,
    /// Boolean flag.
    Boolean,
    /// Stable string label.
    Label,
}

impl MetricUnit {
    /// Stable descriptor label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Rows => "rows",
            Self::DeviceBytes => "device_bytes",
            Self::HostBytes => "host_bytes",
            Self::ParquetBytes => "parquet_bytes",
            Self::Nanos => "nanos",
            Self::Millis => "millis",
            Self::Boolean => "boolean",
            Self::Label => "label",
        }
    }
}

/// How values for a metric are aggregated over a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricAggregation {
    /// Values are summed over the query lifetime.
    Sum,
    /// Value is the maximum observed point-in-time value.
    Peak,
    /// Value is the final query-level label or mode.
    Last,
    /// Value is the number of nested evidence rows.
    RowCount,
}

impl MetricAggregation {
    /// Stable descriptor label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Peak => "peak",
            Self::Last => "last",
            Self::RowCount => "row_count",
        }
    }
}

/// Directional health hint for metrics that have an obvious desired movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricDirection {
    /// Lower values are usually healthier.
    LowerIsBetter,
    /// Higher values are usually healthier.
    HigherIsBetter,
    /// No universal health direction.
    Informational,
}

impl MetricDirection {
    /// Stable descriptor label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LowerIsBetter => "lower_is_better",
            Self::HigherIsBetter => "higher_is_better",
            Self::Informational => "informational",
        }
    }
}

/// Machine-readable descriptor for one query metric or exported evidence row set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetricDescriptor {
    /// Stable metric name.
    pub name: &'static str,
    /// Metric unit.
    pub unit: MetricUnit,
    /// Query-level aggregation semantics.
    pub aggregation: MetricAggregation,
    /// One-line human-readable semantics.
    pub semantics: &'static str,
    /// Optional direction/health hint.
    pub direction: Option<MetricDirection>,
    /// Optional healthy-value hint.
    pub healthy: Option<&'static str>,
}

impl MetricDescriptor {
    /// Serializes the descriptor as a named JSON object.
    pub fn to_json(self) -> Value {
        json!({
            "name": self.name,
            "unit": self.unit.as_str(),
            "aggregation": self.aggregation.as_str(),
            "semantics": self.semantics,
            "direction": self.direction.map(MetricDirection::as_str),
            "healthy": self.healthy,
        })
    }
}

const fn metric(
    name: &'static str,
    unit: MetricUnit,
    aggregation: MetricAggregation,
    semantics: &'static str,
    direction: Option<MetricDirection>,
    healthy: Option<&'static str>,
) -> MetricDescriptor {
    MetricDescriptor { name, unit, aggregation, semantics, direction, healthy }
}

/// QueryMetrics scalar and nested-row descriptors.
pub(crate) const QUERY_METRIC_DESCRIPTORS: &[MetricDescriptor] = &[
    metric(
        "input_batches",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Input Arrow batches observed by native execution.",
        None,
        None,
    ),
    metric(
        "input_rows",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Input rows observed by native execution.",
        None,
        None,
    ),
    metric(
        "gpu_resident_input_handoffs",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "GPU-resident child outputs handed directly to native execution.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "host_to_device_uploads",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Host-to-device upload operations.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "host_to_device_uploaded_rows",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows uploaded from host to device.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "host_to_device_uploaded_bytes",
        MetricUnit::HostBytes,
        MetricAggregation::Sum,
        "Bytes uploaded from host to device.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "device_to_host_materializations",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Device-to-host materialization operations.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "device_to_host_materialized_rows",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows materialized from device to host.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "device_to_host_materialized_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Bytes materialized from device to host.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "device_to_device_copies",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Device-to-device copy operations.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "device_to_device_copied_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Bytes copied between device frames.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "max_task_observed_peak_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Peak,
        "Largest lower-level cuDF peak bytes reported by a task-like operation.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "max_task_observed_peak_bytes_observed",
        MetricUnit::Boolean,
        MetricAggregation::Last,
        "Whether max_task_observed_peak_bytes was reported by a lower layer.",
        None,
        None,
    ),
    metric(
        "max_task_retained_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Peak,
        "Largest retained byte count reported by a task-like lower-level cuDF operation.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "max_task_retained_bytes_observed",
        MetricUnit::Boolean,
        MetricAggregation::Last,
        "Whether max_task_retained_bytes was reported by a lower layer.",
        None,
        None,
    ),
    metric(
        "native_graph_claim_calibration_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested measured graph-execution peak rows.",
        None,
        None,
    ),
    metric(
        "native_graph_runtime_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested graph algorithm runtime evidence rows.",
        None,
        None,
    ),
    metric(
        "native_task_operator_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested per-operator task aggregate rows.",
        None,
        None,
    ),
    metric(
        "native_task_runtime_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested per-compiled-task runtime rows.",
        None,
        None,
    ),
    metric(
        "native_plan_task_mapping_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested native plan-node to compiled-task mapping rows.",
        None,
        None,
    ),
    metric(
        "native_dynamic_filter_runtime_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested dynamic-filter runtime evidence rows.",
        None,
        None,
    ),
    metric(
        "native_source_read_runtime_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested source-read runtime evidence rows.",
        None,
        None,
    ),
    metric(
        "peak_device_memory_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Peak,
        "Backward-compatible alias for resident_dataframe_peak_bytes.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "resident_dataframe_peak_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Peak,
        "Peak bytes retained by resident cudf DataFrames.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "resident_dataframe_peak_bytes_observed",
        MetricUnit::Boolean,
        MetricAggregation::Last,
        "Whether resident_dataframe_peak_bytes was observed.",
        None,
        None,
    ),
    metric(
        "allocator_high_water_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Peak,
        "cuDF allocator high-water bytes when reported by lower layers.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "allocator_high_water_bytes_observed",
        MetricUnit::Boolean,
        MetricAggregation::Last,
        "Whether allocator_high_water_bytes was observed.",
        None,
        None,
    ),
    metric(
        "source_chunks_planned",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source chunks planned by native source execution.",
        None,
        None,
    ),
    metric(
        "source_chunks_attempted",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source chunk attempts including retries.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "source_chunks_committed",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source chunks committed successfully.",
        None,
        None,
    ),
    metric(
        "source_chunks_retried",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source chunks retried.",
        Some(MetricDirection::LowerIsBetter),
        Some("0"),
    ),
    metric(
        "source_read_prefetches_started",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source-read prefetches started.",
        None,
        None,
    ),
    metric(
        "source_read_prefetches_completed",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source-read prefetches completed.",
        None,
        None,
    ),
    metric(
        "source_read_prefetches_skipped",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source-read prefetches skipped.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "source_read_prefetches_abandoned",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Source-read prefetches abandoned.",
        Some(MetricDirection::LowerIsBetter),
        Some("0"),
    ),
    metric(
        "row_groups_planned",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Row groups planned across source chunks.",
        None,
        None,
    ),
    metric(
        "row_groups_committed",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Row groups committed across source chunks.",
        None,
        None,
    ),
    metric(
        "partial_aggregate_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Partial aggregate tasks executed.",
        None,
        None,
    ),
    metric(
        "partial_aggregate_input_rows",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows consumed by partial aggregates.",
        None,
        None,
    ),
    metric(
        "partial_aggregate_rows",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows produced by partial aggregates.",
        None,
        None,
    ),
    metric(
        "partial_aggregate_retained_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Peak,
        "Peak bytes retained by partial aggregate outputs.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "final_merge_attempts",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Final aggregate merge attempts.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "final_merge_levels",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Final aggregate merge levels.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "final_merge_max_fan_in",
        MetricUnit::Count,
        MetricAggregation::Peak,
        "Largest final merge fan-in.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "compiled_pipeline_attempts",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Compiled-pipeline execution attempts admitted before runtime.",
        None,
        None,
    ),
    metric(
        "compiled_pipeline_tasks_executed",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Compiled-pipeline tasks executed by the scheduler.",
        None,
        None,
    ),
    metric(
        "compiled_pipeline_host_arrow_sinks",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Host Arrow sink boundaries materialized by compiled pipelines.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "parquet_files",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Parquet files observed by source reads.",
        None,
        None,
    ),
    metric(
        "parquet_bytes_known",
        MetricUnit::ParquetBytes,
        MetricAggregation::Sum,
        "Known Parquet file bytes.",
        None,
        None,
    ),
    metric(
        "row_groups_selected",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Row groups selected by source reads.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "read_columns",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Physical read columns selected by source reads.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "retry_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Native execution retries.",
        Some(MetricDirection::LowerIsBetter),
        Some("0"),
    ),
    metric(
        "retry_resource_exhausted_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Retries caused by lower-level resource exhaustion.",
        Some(MetricDirection::LowerIsBetter),
        Some("0"),
    ),
    metric(
        "dynamic_filter_produced_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Dynamic-filter payloads produced.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_skipped_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Dynamic-filter routes skipped without a usable payload.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_applied_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Dynamic-filter payloads applied to runtime consumers.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_row_groups_pruned",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Row groups pruned by dynamic filters.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_source_bytes_pruned",
        MetricUnit::ParquetBytes,
        MetricAggregation::Sum,
        "Estimated source bytes pruned by dynamic filters.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_reader_predicate_rows_before",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Source rows selected before dynamic reader predicates.",
        None,
        None,
    ),
    metric(
        "dynamic_filter_reader_predicate_rows_after",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Source rows materialized after dynamic reader predicates.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_reader_predicate_rows_filtered",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Source rows filtered by dynamic reader predicates.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_post_decode_rows_before",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows before post-decode dynamic filters.",
        None,
        None,
    ),
    metric(
        "dynamic_filter_post_decode_rows_after",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows after post-decode dynamic filters.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_post_decode_rows_filtered",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows filtered by post-decode dynamic filters.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_source_post_decode_rows_filtered",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows filtered by source-adjacent post-decode dynamic filters.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_filter_post_decode_rows_filtered",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows filtered by filter-operator post-decode dynamic filters.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "dynamic_filter_join_probe_post_decode_rows_filtered",
        MetricUnit::Rows,
        MetricAggregation::Sum,
        "Rows filtered by hash-join-probe post-decode dynamic filters.",
        Some(MetricDirection::HigherIsBetter),
        None,
    ),
    metric(
        "hash_join_build_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Reusable cuDF hash-join artifacts built.",
        None,
        None,
    ),
    metric(
        "filtered_join_build_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "Reusable cuDF filtered-join artifacts built.",
        None,
        None,
    ),
    metric(
        "one_shot_join_count",
        MetricUnit::Count,
        MetricAggregation::Sum,
        "One-shot keyed equality joins executed.",
        None,
        None,
    ),
    metric(
        "hash_join_hash_table_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Device bytes retained by reusable hash-join build artifacts.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "filtered_join_hash_table_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Device bytes retained by reusable filtered-join build artifacts.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "filtered_join_probe_state_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Device bytes retained by filtered-join probe states.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "mixed_join_size_data_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Device bytes retained by mixed join size-data states.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "one_shot_join_index_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Device bytes retained by one-shot join index maps.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "one_shot_join_materialization_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Reservation bytes requested for one-shot join output materialization.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "one_shot_join_output_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Observed device bytes of one-shot join output frames.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "join_gather_map_bytes",
        MetricUnit::DeviceBytes,
        MetricAggregation::Sum,
        "Device bytes retained by join gather maps.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "native_aggregate_strategy_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested aggregate strategy decision rows.",
        None,
        None,
    ),
    metric(
        "native_join_strategy_rows",
        MetricUnit::Count,
        MetricAggregation::RowCount,
        "Nested join strategy decision rows.",
        None,
        None,
    ),
    metric(
        "source_metadata_load_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent loading source metadata.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "source_read_queue_wait_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent waiting in the source-read queue.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "parquet_read_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent inside Parquet read calls.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "source_read_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent in source-read orchestration.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "operator_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent in non-source operators.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "chunk_operator_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent in per-chunk operators.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "final_merge_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent in final merge operators.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "final_sort_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent in final sort operators.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
    metric(
        "materialization_time",
        MetricUnit::Nanos,
        MetricAggregation::Sum,
        "Time spent materializing final outputs.",
        Some(MetricDirection::LowerIsBetter),
        None,
    ),
];

impl QueryMetrics {
    /// Returns the query-metric descriptor catalog.
    pub fn descriptors() -> &'static [MetricDescriptor] {
        QUERY_METRIC_DESCRIPTORS
    }
}
