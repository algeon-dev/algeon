//! Native query-metric recording into DataFusion wrapper metrics.

use std::{
    collections::BTreeMap,
    fmt::{self, Write},
    ops::Deref,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use datafusion::physical_plan::metrics::{BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType};
use nexus_query_engine::{
    error::QueryError,
    exec::output::{QueryExecutionOutcome, QueryOutput},
    metrics::{JoinOperatorMetrics, QueryMetrics},
    observability::QueryAttemptReport,
    runtime::{ExecutionBoundaryPolicy, LakehouseCacheFillEventSnapshot},
};

use super::{
    SERVER_CAPTURE_LOSS_METRIC_NAME, capture::NativeExecutionMetricsCapture, contract::metric_name,
    registry::NativeMetricRegistry,
};
use crate::native_exec::admission::QueryAdmissionFacts;

mod mapping;

#[cfg(test)]
pub(super) use mapping::{NATIVE_NEXUS_RECORDING_DESCRIPTORS, NativeMetricRecordingKind};

/// A root-native-wrapper target that can receive one attempt projection after
/// its owner has sealed the immutable report. It does not retain the report,
/// own attempt lifecycle state, or expose unbounded labels.
#[derive(Debug, Clone)]
pub(crate) struct SealedAttemptMetricsTarget {
    metrics: ExecutionPlanMetricsSet,
}

impl SealedAttemptMetricsTarget {
    pub(crate) fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self { metrics: metrics.clone() }
    }

    /// Emit the single fixed-cardinality DataFusion row for this attempt.
    ///
    /// The caller elects exactly one root wrapper before execution. All labels
    /// are derived from the sealed projection; in particular, the attempt id and device
    /// ordinal deliberately never become metric labels.
    pub(crate) fn record(self, report: &QueryAttemptReport) {
        let labels = report.sealed_projection().labels();
        MetricBuilder::new(&self.metrics)
            .with_type(MetricType::Summary)
            .with_new_label("disposition", labels.disposition)
            .with_new_label("outcome", labels.outcome)
            .with_new_label("cleanup_disposition", labels.cleanup_disposition)
            .with_new_label("cleanup_failure_count_complete", labels.cleanup_failure_count_complete)
            .with_new_label("capacity_disposition", labels.capacity_disposition)
            .with_new_label("health_transition", labels.health_transition)
            .with_new_label("retry_recovery", labels.retry_recovery)
            .with_new_label("retry_prerequisite", labels.retry_prerequisite)
            .with_new_label("observation_truncated", labels.observation_truncated)
            .with_new_label("observation_degraded", labels.observation_degraded)
            .with_new_label("native_fault_evidence", labels.native_fault_evidence)
            .with_new_label("resource_evidence", labels.resource_evidence)
            .global_counter(metric_name::attempt_report_rows())
            .add(1);
    }
}

/// Private translation layer from native QueryMetrics into DataFusion metrics.
#[derive(Debug, Clone)]
pub(crate) struct NexusNativeMetrics {
    metrics_set: ExecutionPlanMetricsSet,
    capture: Option<NativeExecutionMetricsCapture>,
    plan_task_mapping_datafusion_shape: Option<String>,
    pub(crate) baseline: BaselineMetrics,
    registry: NativeMetricRegistry,
    row_capture_budget: Arc<Mutex<MetricRowCaptureBudget>>,
    row_capture_loss: Count,
    row_capture_loss_reported: Arc<AtomicBool>,
}

#[derive(Debug)]
struct MetricRowCaptureBudget {
    remaining_records: u64,
    remaining_bytes: u64,
    truncated: bool,
    counter_overflow: bool,
}

impl MetricRowCaptureBudget {
    fn new(capture: Option<&NativeExecutionMetricsCapture>) -> Self {
        let limits = capture.map_or_else(
            || nexus_query_engine::observability::RuntimeObservationConfig::native_execution().limits(),
            NativeExecutionMetricsCapture::metric_capture_limits,
        );
        Self {
            remaining_records: limits.max_retained_records,
            remaining_bytes: limits.max_retained_bytes,
            truncated: false,
            counter_overflow: false,
        }
    }

    fn admit(&mut self, row: &impl fmt::Debug) -> bool {
        if self.remaining_records == 0 {
            self.truncated = true;
            return false;
        }
        let mut probe = MetricRowSizeProbe { remaining: self.remaining_bytes, exceeded: false };
        let _ = write!(probe, "{row:?}");
        if probe.exceeded {
            self.truncated = true;
            return false;
        }
        let used = self.remaining_bytes - probe.remaining;
        self.remaining_records = self.remaining_records.checked_sub(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            0
        });
        self.remaining_bytes = self.remaining_bytes.checked_sub(used).unwrap_or_else(|| {
            self.counter_overflow = true;
            0
        });
        true
    }
}

struct MetricRowSizeProbe {
    remaining: u64,
    exceeded: bool,
}

impl fmt::Write for MetricRowSizeProbe {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let bytes = u64::try_from(value.len()).unwrap_or(u64::MAX);
        if bytes > self.remaining {
            self.exceeded = true;
            self.remaining = 0;
            return Err(fmt::Error);
        }
        self.remaining -= bytes;
        Ok(())
    }
}

// Recorder methods use descriptor-generated field identifiers directly while the
// registry retains ownership of all registered DataFusion metric handles.
impl Deref for NexusNativeMetrics {
    type Target = NativeMetricRegistry;

    fn deref(&self) -> &Self::Target {
        &self.registry
    }
}

impl NexusNativeMetrics {
    #[cfg(test)]
    pub(crate) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self::new_with_capture_and_shape(metrics, partition, None, None)
    }

    pub(crate) fn new_with_capture(
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
        capture: Option<NativeExecutionMetricsCapture>,
    ) -> Self {
        Self::new_with_capture_and_shape(metrics, partition, capture, None)
    }

    pub(crate) fn new_with_capture_and_shape(
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
        capture: Option<NativeExecutionMetricsCapture>,
        plan_task_mapping_datafusion_shape: Option<&str>,
    ) -> Self {
        let row_capture_budget = Arc::new(Mutex::new(MetricRowCaptureBudget::new(capture.as_ref())));
        let row_capture_loss = MetricBuilder::new(metrics)
            .with_type(MetricType::Summary)
            .counter(SERVER_CAPTURE_LOSS_METRIC_NAME, partition);
        Self {
            metrics_set: metrics.clone(),
            capture,
            plan_task_mapping_datafusion_shape: plan_task_mapping_datafusion_shape.map(ToString::to_string),
            baseline: BaselineMetrics::new(metrics, partition),
            registry: NativeMetricRegistry::register(metrics, partition),
            row_capture_budget,
            row_capture_loss,
            row_capture_loss_reported: Arc::new(AtomicBool::new(false)),
        }
    }

    fn admit_metric_row(&self, row: &impl fmt::Debug) -> bool {
        let admitted = self.row_capture_budget.lock().unwrap_or_else(std::sync::PoisonError::into_inner).admit(row);
        if !admitted && !self.row_capture_loss_reported.swap(true, Ordering::Relaxed) {
            self.row_capture_loss.add(1);
        }
        admitted
    }

    #[cfg(test)]
    pub(super) fn row_capture_state(&self) -> (u64, u64, bool, bool) {
        let budget = self.row_capture_budget.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (budget.remaining_records, budget.remaining_bytes, budget.truncated, budget.counter_overflow)
    }

    pub(crate) fn record_admission_facts(&self, facts: QueryAdmissionFacts) {
        self.record_descriptor_admission_facts(facts);
    }

    pub(crate) fn record_query_metrics(&self, metrics: &QueryMetrics) {
        self.record_descriptor_query_metrics(metrics);
    }

    /// Records every evidence surface before returning the native result to the
    /// adapter's `QueryError -> DataFusionError` conversion boundary.
    pub(crate) fn finalize_query_outcome(
        &self,
        outcome: QueryExecutionOutcome,
    ) -> std::result::Result<QueryOutput, QueryError> {
        let (result, metrics, _observation) = outcome.into_parts();
        self.record_query_metrics(&metrics);
        result
    }

    /// Finalize a native outcome whose logical host sink is materialized later
    /// by the adapter's bounded GPU-to-host cursor.
    pub(crate) fn finalize_query_outcome_for_deferred_host(
        &self,
        outcome: QueryExecutionOutcome,
    ) -> std::result::Result<QueryOutput, QueryError> {
        let (result, mut metrics, _observation) = outcome.into_parts();
        if metrics.compiled_pipeline_attempts > 0 {
            metrics.record_compiled_pipeline_tasks_executed(1);
            metrics.record_compiled_pipeline_host_arrow_sink();
        }
        self.record_query_metrics(&metrics);
        result
    }

    pub(crate) fn record_deferred_host_materialization(&self, rows: usize, bytes: usize) {
        self.device_to_host_materializations.add(1);
        self.device_to_host_materialized_rows.add(rows);
        self.device_to_host_materialized_bytes.add(bytes);
    }

    pub(crate) fn record_lakehouse_cache_fill_events(&self, events: &[LakehouseCacheFillEventSnapshot]) {
        self.record_descriptor_cache_events(events);
    }

    fn record_lakehouse_cache_fill_event_rows(&self, events: &[LakehouseCacheFillEventSnapshot]) {
        for event in events {
            if !self.admit_metric_row(event) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("outcome", event.outcome.to_string())
                .with_new_label("bytes", event.bytes.map_or_else(|| "none".to_string(), |value| value.to_string()))
                .with_new_label("elapsed_ns", event.elapsed.as_nanos().min(u128::from(u64::MAX)).to_string())
                .with_new_label("cache_path_hash", event.cache_path_hash.clone())
                .with_new_label("stable_id_hash", event.stable_id_hash.clone())
                .with_new_label(
                    "file_index",
                    event.file_index.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "chunk_index",
                    event.chunk_index.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("error_code", event.error_code.clone().unwrap_or_else(|| "none".to_string()))
                .with_new_label("error_kind", event.error_kind.clone().unwrap_or_else(|| "none".to_string()))
                .with_new_label("error_status", event.error_status.clone().unwrap_or_else(|| "none".to_string()))
                .with_new_label("error_operation", event.error_operation.clone().unwrap_or_else(|| "none".to_string()))
                .global_counter(metric_name::native_lakehouse_cache_fill_event_snapshot_rows())
                .add(1);
        }
    }

    fn record_graph_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.graph_algorithm_runtime_rows {
            if !self.admit_metric_row(row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("node_id", row.node_id.clone())
                .with_new_label("algorithm", row.algorithm.clone())
                .with_new_label("runtime_outcome", row.runtime_outcome.clone())
                .with_new_label("input_boundary", row.input_boundary.clone())
                .with_new_label("output_boundary", row.output_boundary.clone())
                .with_new_label("construction_policy", row.construction_policy.clone())
                .with_new_label("source_column", row.source_column.clone())
                .with_new_label("destination_column", row.destination_column.clone())
                .with_new_label(
                    "weight_column",
                    row.weight_column.as_deref().map_or_else(|| "none".to_string(), ToOwned::to_owned),
                )
                .with_new_label("directed", row.directed.to_string())
                .with_new_label("renumber", row.renumber.to_string())
                .with_new_label("algorithm_parameters", row.algorithm_parameters.clone())
                .with_new_label("algorithm_metadata", row.algorithm_metadata.clone())
                .with_new_label("input_rows", row.input_rows.to_string())
                .with_new_label("input_columns", row.input_columns.to_string())
                .with_new_label("output_rows", row.output_rows.to_string())
                .with_new_label("output_columns", row.output_columns.to_string())
                .with_new_label(
                    "output_device_bytes",
                    row.output_device_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_peak_bytes",
                    row.graph_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("elapsed_ns", row.elapsed_ns.to_string())
                .with_new_label("gpu_resident_input_handoffs", row.gpu_resident_input_handoffs.to_string())
                .with_new_label("host_to_device_uploads", row.host_to_device_uploads.to_string())
                .with_new_label("device_to_host_materializations", row.device_to_host_materializations.to_string())
                .global_counter(metric_name::native_graph_runtime_rows())
                .add(1);
        }
    }

    fn record_graph_reservation_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.graph_reservation_rows {
            if !self.admit_metric_row(row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("node_id", row.node_id.clone())
                .with_new_label("algorithm", row.algorithm.as_str().to_string())
                .with_new_label("class", row.class.as_str().to_string())
                .with_new_label("outcome", row.outcome.as_str().to_string())
                .with_new_label(
                    "raw_input_edges",
                    row.raw_input_edges.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("input_edges", row.input_edges.to_string())
                .with_new_label("vertex_upper_bound", row.vertex_upper_bound.to_string())
                .with_new_label("vertex_estimate", row.vertex_estimate.to_string())
                .with_new_label("vertex_estimate_source", row.vertex_estimate_source.as_str().to_string())
                .with_new_label("vertex_width_bytes", row.vertex_width_bytes.to_string())
                .with_new_label(
                    "weight_width_bytes",
                    row.weight_width_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "pair_rows",
                    row.pair_rows.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "pair_vertex_width_bytes",
                    row.pair_vertex_width_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "pair_cardinality_source",
                    row.pair_cardinality_source
                        .map_or_else(|| "none".to_string(), |source| source.as_str().to_string()),
                )
                .with_new_label(
                    "pair_cardinality_admission_mode",
                    row.pair_cardinality_admission_mode
                        .map_or_else(|| "none".to_string(), |mode| mode.as_str().to_string()),
                )
                .with_new_label("has_edge_id", row.has_edge_id.to_string())
                .with_new_label("directed", row.directed.to_string())
                .with_new_label("renumber", row.renumber.to_string())
                .with_new_label("construction_policy", row.construction_policy.as_str().to_string())
                .with_new_label("edge_normalize_inserted", row.edge_normalize_inserted.to_string())
                .with_new_label("calibration_source", row.calibration_source.as_str().to_string())
                .with_new_label(
                    "seed_count",
                    row.seed_count.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "seed_mode",
                    row.seed_mode.map_or_else(|| "none".to_string(), |mode| mode.as_str().to_string()),
                )
                .with_new_label("data_dependent", row.data_dependent.to_string())
                .with_new_label("model_requested_bytes", row.model_requested_bytes.to_string())
                .with_new_label("requested_bytes", row.requested_bytes.to_string())
                .with_new_label(
                    "budget_bytes",
                    row.budget_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_model_error_ratio",
                    row.graph_model_error_ratio.map_or_else(|| "none".to_string(), |ratio| ratio.as_decimal_string()),
                )
                .with_new_label(
                    "measured_peak_bytes",
                    row.measured_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_statistics_exact_vertices",
                    row.statistics
                        .map_or_else(|| "none".to_string(), |statistics| statistics.exact_vertices.to_string()),
                )
                .with_new_label(
                    "graph_statistics_degree_max",
                    row.statistics
                        .and_then(|statistics| statistics.degree_max)
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_statistics_degree_p99",
                    row.statistics
                        .and_then(|statistics| statistics.degree_p99)
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_statistics_sum_min_degree_oriented",
                    row.statistics
                        .and_then(|statistics| statistics.sum_min_degree_oriented)
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_statistics_oriented_edge_count",
                    row.statistics
                        .and_then(|statistics| statistics.oriented_edge_count)
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_statistics_peak_chunk_min_degree_sum",
                    row.statistics
                        .and_then(|statistics| statistics.peak_chunk_min_degree_sum)
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "graph_statistics_sum_out_degree",
                    row.statistics
                        .and_then(|statistics| statistics.sum_out_degree)
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("reason", row.reason.unwrap_or("none").to_string())
                .global_counter(metric_name::native_graph_reservation_rows())
                .add(1);
        }
    }

    fn record_task_operator_rows(&self, metrics: &QueryMetrics) {
        for row in metrics.task_operator_metrics.values() {
            if !self.admit_metric_row(row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("operator_kind", row.operator_kind.clone())
                .with_new_label("task_count", row.task_count.to_string())
                .with_new_label("elapsed_ns", row.elapsed_time.as_nanos().min(u128::from(u64::MAX)).to_string())
                .with_new_label("reservation_requested_bytes", row.reservation_requested_bytes.to_string())
                .with_new_label("reservation_granted_bytes", row.reservation_granted_bytes.to_string())
                .with_new_label("reservation_released_bytes", row.reservation_released_bytes.to_string())
                .with_new_label("reservation_live_peak_bytes", row.reservation_live_peak_bytes.to_string())
                .with_new_label(
                    "max_observed_peak_bytes",
                    row.max_observed_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "max_retained_bytes",
                    row.max_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .global_counter(metric_name::native_task_operator_rows())
                .add(1);
        }
    }

    fn record_task_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in metrics.task_runtime_metrics.values() {
            if !self.admit_metric_row(row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("compiled_task_id", row.compiled_task_id.clone())
                .with_new_label("operator_kind", row.operator_kind.clone())
                .with_new_label("elapsed_ns", row.elapsed_time.as_nanos().min(u128::from(u64::MAX)).to_string())
                .with_new_label("reservation_requested_bytes", row.reservation_requested_bytes.to_string())
                .with_new_label("reservation_granted_bytes", row.reservation_granted_bytes.to_string())
                .with_new_label("reservation_released_bytes", row.reservation_released_bytes.to_string())
                .with_new_label("reservation_live_peak_bytes", row.reservation_live_peak_bytes.to_string())
                .with_new_label(
                    "max_observed_peak_bytes",
                    row.max_observed_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "max_retained_bytes",
                    row.max_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .global_counter(metric_name::native_task_runtime_rows())
                .add(1);
        }
    }

    fn record_plan_task_mapping_rows(&self, metrics: &QueryMetrics) {
        for mut row in metrics.plan_task_mappings() {
            if let Some(shape) = &self.plan_task_mapping_datafusion_shape {
                row.datafusion_shape = Some(shape.clone());
            }
            if !self.admit_metric_row(&row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("query_id", row.query_id.as_str().to_string())
                .with_new_label(
                    "plan_node_id",
                    row.plan_node_id.as_ref().map_or_else(|| "none".to_string(), |value| value.as_str().to_string()),
                )
                .with_new_label(
                    "compiled_task_id",
                    row.compiled_task_id
                        .as_ref()
                        .map_or_else(|| "none".to_string(), |value| value.as_str().to_string()),
                )
                .with_new_label("operator_kind", row.operator_kind.unwrap_or_else(|| "none".to_string()))
                .with_new_label("datafusion_shape", row.datafusion_shape.unwrap_or_else(|| "none".to_string()))
                .global_counter(metric_name::native_plan_task_mapping_rows())
                .add(1);
        }
    }

    fn record_dynamic_filter_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.dynamic_filter_runtime_rows {
            if !self.admit_metric_row(row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("dynamic_filter_id", row.dynamic_filter_id.clone())
                .with_new_label("event", row.event.clone())
                .with_new_label("producer_kind", row.producer_kind.clone())
                .with_new_label("producer_node_id", row.producer_node_id.clone())
                .with_new_label("consumer_kind", row.consumer_kind.clone())
                .with_new_label("consumer_node_id", row.consumer_node_id.clone())
                .with_new_label("consumer_lowering", row.consumer_lowering.clone())
                .with_new_label("consumer_reduction_stage", row.consumer_reduction_stage.clone())
                .with_new_label("filter_kind", row.filter_kind.clone())
                .with_new_label(
                    "payload_value_count",
                    row.payload_value_count.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "payload_min_i64",
                    row.payload_min_i64.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "payload_max_i64",
                    row.payload_max_i64.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "producer_rows",
                    row.producer_rows.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "skipped_reason",
                    row.skipped_reason.as_deref().map_or_else(|| "none".to_string(), ToOwned::to_owned),
                )
                .with_new_label(
                    "selected_row_groups_before",
                    row.selected_row_groups_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "selected_row_groups_after",
                    row.selected_row_groups_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "selected_source_bytes_before",
                    row.selected_source_bytes_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "selected_source_bytes_after",
                    row.selected_source_bytes_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "source_file_index",
                    row.source_file_index.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "source_chunk_index",
                    row.source_chunk_index.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "reader_predicate_rows_before",
                    row.reader_predicate_rows_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "reader_predicate_rows_after",
                    row.reader_predicate_rows_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("reader_predicate_rows_filtered", row.reader_predicate_rows_filtered.to_string())
                .with_new_label(
                    "post_decode_rows_before",
                    row.post_decode_rows_before.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "post_decode_rows_after",
                    row.post_decode_rows_after.map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("post_decode_rows_filtered", row.post_decode_rows_filtered.to_string())
                .with_new_label("row_groups_pruned", row.row_groups_pruned.to_string())
                .with_new_label("source_bytes_pruned", row.source_bytes_pruned.to_string())
                .global_counter(metric_name::native_dynamic_filter_runtime_rows())
                .add(1);
        }
    }

    fn record_source_read_runtime_rows(&self, metrics: &QueryMetrics) {
        for row in &metrics.source_read_runtime_rows {
            if !self.admit_metric_row(row) {
                continue;
            }
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("source_node_id", row.source_node_id.clone())
                .with_new_label("read_kind", row.read_kind.clone())
                .with_new_label("read_strategy", row.read_strategy.clone())
                .with_new_label("parquet_read_input_kind", row.parquet_read_evidence.input_kind.clone())
                .with_new_label("remote_read_mode", row.parquet_read_evidence.remote_read_mode.clone())
                .with_new_label("parquet_read_backend", row.parquet_read_evidence.backend.clone())
                .with_new_label("remote_backend_build_status", row.parquet_read_evidence.backend_build_status.clone())
                .with_new_label(
                    "remote_read_bytes",
                    row.parquet_read_evidence
                        .remote_bytes
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "remote_read_range_count",
                    row.parquet_read_evidence
                        .remote_range_count
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("device_read_support", row.parquet_read_evidence.device_read_support.clone())
                .with_new_label(
                    "parquet_pruning_input_row_groups",
                    row.parquet_read_evidence
                        .pruning_input_row_groups
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "parquet_row_groups_after_stats_filter",
                    row.parquet_read_evidence
                        .row_groups_after_stats_filter
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label(
                    "parquet_row_groups_after_bloom_filter",
                    row.parquet_read_evidence
                        .row_groups_after_bloom_filter
                        .map_or_else(|| "none".to_string(), |value| value.to_string()),
                )
                .with_new_label("parquet_bloom_pruning_status", row.parquet_read_evidence.bloom_pruning_status.clone())
                .with_new_label("batch_index", row.batch_index.to_string())
                .with_new_label("chunk_count", row.chunk_count.to_string())
                .with_new_label("first_chunk_index", row.first_chunk_index.to_string())
                .with_new_label("last_chunk_index", row.last_chunk_index.to_string())
                .with_new_label("first_file_index", row.first_file_index.to_string())
                .with_new_label("last_file_index", row.last_file_index.to_string())
                .with_new_label("file_count", row.file_count.to_string())
                .with_new_label("row_group_count", row.row_group_count.to_string())
                .with_new_label("read_columns_per_file", row.read_columns_per_file.to_string())
                .with_new_label("read_columns_total", row.read_columns_total.to_string())
                .with_new_label("known_file_bytes", row.known_file_bytes.to_string())
                .with_new_label("estimated_source_read_bytes", row.estimated_source_read_bytes.to_string())
                .with_new_label("known_row_group_bytes", row.known_row_group_bytes.to_string())
                .with_new_label("known_projected_compressed_bytes", row.known_projected_compressed_bytes.to_string())
                .with_new_label(
                    "known_projected_uncompressed_bytes",
                    row.known_projected_uncompressed_bytes.to_string(),
                )
                .with_new_label(
                    "known_pure_filter_compressed_bytes",
                    row.known_pure_filter_compressed_bytes.to_string(),
                )
                .with_new_label(
                    "known_pure_filter_uncompressed_bytes",
                    row.known_pure_filter_uncompressed_bytes.to_string(),
                )
                .with_new_label("output_rows", row.output_rows.to_string())
                .with_new_label("reader_predicate", row.reader_predicate.to_string())
                .with_new_label("reader_dynamic_filter_count", row.reader_dynamic_filter_count.to_string())
                .with_new_label("source_batching_reason", row.source_batching_reason.clone())
                .with_new_label("source_chunk_planning_mode", row.source_chunk_planning_mode.clone())
                .with_new_label("source_chunk_planning_reason", row.source_chunk_planning_reason.clone())
                .with_new_label("elapsed_ns", row.elapsed.as_nanos().min(u128::from(u64::MAX)).to_string())
                .global_counter(metric_name::native_source_read_runtime_rows())
                .add(1);
        }
    }

    fn record_join_strategy_rows(&self, metrics: &QueryMetrics) {
        let mut decision_facts_by_node: BTreeMap<&str, BTreeMap<String, String>> = BTreeMap::new();
        for decision in &metrics.join_strategy_decisions {
            if !self.admit_metric_row(decision) {
                continue;
            }
            decision_facts_by_node
                .entry(decision.node_id.as_str())
                .or_default()
                .extend(decision.facts.iter().map(|(key, value)| (key.clone(), value.clone())));
        }
        for operator in metrics.operators.iter().filter(|operator| operator.operator_kind == "join") {
            if !self.admit_metric_row(operator) {
                continue;
            }
            let Some(join) = operator.join.as_ref() else {
                continue;
            };
            let facts = decision_facts_by_node.get(operator.node_id.as_str());
            let join_strategy = facts
                .and_then(|facts| facts.get("join_strategy").map(String::as_str))
                .unwrap_or_else(|| join_strategy_label(join));
            let runtime = join_runtime_counters(join);
            let builder = MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("node_id", operator.node_id.clone())
                .with_new_label("join_strategy", join_strategy.to_string())
                .with_new_label("execution_strategy", join.execution_strategy.clone())
                .with_new_label("join_type", join.join_type.clone())
                .with_new_label("build_side", join.build_side.clone())
                .with_new_label("key_count", join.key_count.to_string())
                .with_new_label("estimated_build_rows", optional_fact(facts, "estimated_build_rows"))
                .with_new_label(
                    "observed_build_rows",
                    facts
                        .and_then(|facts| facts.get("observed_build_rows").cloned())
                        .unwrap_or_else(|| join.build_rows.to_string()),
                )
                .with_new_label(
                    "build_side_bytes",
                    facts
                        .and_then(|facts| facts.get("build_side_bytes").cloned())
                        .unwrap_or_else(|| join.build_bytes.to_string()),
                )
                .with_new_label("probe_reuse_count", optional_fact(facts, "probe_reuse_count"))
                .with_new_label("probe_side_bytes", optional_fact(facts, "probe_side_bytes"))
                .with_new_label("estimated_output_rows", optional_fact(facts, "estimated_output_rows"))
                .with_new_label("estimated_output_bytes", optional_fact(facts, "estimated_output_bytes"))
                .with_new_label("build_key_unique", optional_fact(facts, "build_key_unique"))
                .with_new_label(
                    "build_key_uniqueness_probe_status",
                    optional_fact(facts, "build_key_uniqueness_probe_status"),
                )
                .with_new_label(
                    "build_key_uniqueness_probe_error_code",
                    optional_fact(facts, "build_key_uniqueness_probe_error_code"),
                )
                .with_new_label("partition_count", optional_fact(facts, "partition_count"))
                .with_new_label("partition_target_bytes", optional_fact(facts, "partition_target_bytes"))
                .with_new_label("build_partition_batch_count", optional_fact(facts, "build_partition_batch_count"))
                .with_new_label("probe_partition_batch_count", optional_fact(facts, "probe_partition_batch_count"))
                .with_new_label("build_partitioned_bytes", optional_fact(facts, "build_partitioned_bytes"))
                .with_new_label("probe_partitioned_bytes", optional_fact(facts, "probe_partitioned_bytes"))
                .with_new_label(
                    "reusable_hash_join_rejected_reason",
                    optional_fact(facts, "reusable_hash_join_rejected_reason"),
                )
                .with_new_label(
                    "partitioned_hash_join_rejected_reason",
                    optional_fact(facts, "partitioned_hash_join_rejected_reason"),
                )
                .with_new_label("one_shot_equi_rejected_reason", optional_fact(facts, "one_shot_equi_rejected_reason"))
                .with_new_label("partitioned_hash_join_count", runtime.partitioned_hash_join_count.to_string())
                .with_new_label("hash_join_build_count", runtime.hash_join_build_count.to_string())
                .with_new_label("hash_join_hash_table_bytes", runtime.hash_join_hash_table_bytes.to_string())
                .with_new_label("filtered_join_build_count", runtime.filtered_join_build_count.to_string())
                .with_new_label("filtered_join_hash_table_bytes", runtime.filtered_join_hash_table_bytes.to_string())
                .with_new_label("filtered_join_probe_state_bytes", runtime.filtered_join_probe_state_bytes.to_string())
                .with_new_label("mixed_join_size_data_bytes", runtime.mixed_join_size_data_bytes.to_string())
                .with_new_label("one_shot_join_count", runtime.one_shot_join_count.to_string())
                .with_new_label("one_shot_join_index_bytes", runtime.one_shot_join_index_bytes.to_string())
                .with_new_label(
                    "one_shot_join_materialization_bytes",
                    runtime.one_shot_join_materialization_bytes.to_string(),
                )
                .with_new_label("one_shot_join_output_bytes", runtime.one_shot_join_output_bytes.to_string())
                .with_new_label("join_gather_map_bytes", join.gather_map_bytes.to_string())
                .with_new_label("join_output_reservation_bytes", join.output_reservation_bytes.to_string())
                .with_new_label("join_output_rows", join.output_rows.to_string())
                .with_new_label("join_output_bytes", join.output_bytes.to_string());
            builder.global_counter(metric_name::native_join_strategy_rows()).add(1);
        }
    }

    fn record_aggregate_strategy_rows(&self, metrics: &QueryMetrics) {
        for decision in &metrics.aggregate_strategy_decisions {
            if !self.admit_metric_row(decision) {
                continue;
            }
            let facts = &decision.facts;
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("node_id", decision.node_id.clone())
                .with_new_label("selected_strategy", aggregate_fact(facts, "selected_strategy"))
                .with_new_label("execution_stage", aggregate_fact(facts, "execution_stage"))
                .with_new_label("memory_mode", aggregate_fact(facts, "memory_mode"))
                .with_new_label("group_key_count", aggregate_fact(facts, "group_key_count"))
                .with_new_label("group_key_exprs", aggregate_fact(facts, "group_key_exprs"))
                .with_new_label("group_key_dtypes", aggregate_fact(facts, "group_key_dtypes"))
                .with_new_label("aggregate_functions", aggregate_fact(facts, "aggregate_functions"))
                .with_new_label(
                    "floating_statistical_aggregate",
                    aggregate_fact(facts, "floating_statistical_aggregate"),
                )
                .with_new_label(
                    "statistical_aggregate_execution_mode",
                    aggregate_fact(facts, "statistical_aggregate_execution_mode"),
                )
                .with_new_label(
                    "statistical_aggregate_tolerance_contract",
                    aggregate_fact(facts, "statistical_aggregate_tolerance_contract"),
                )
                .with_new_label(
                    "statistical_aggregate_functions",
                    aggregate_fact(facts, "statistical_aggregate_functions"),
                )
                .with_new_label("statistical_aggregate_ddof", aggregate_fact(facts, "statistical_aggregate_ddof"))
                .with_new_label("stddev_samp_execution_mode", aggregate_fact(facts, "stddev_samp_execution_mode"))
                .with_new_label("stddev_samp_ddof", aggregate_fact(facts, "stddev_samp_ddof"))
                .with_new_label("aggregate_filter_count", aggregate_fact(facts, "aggregate_filter_count"))
                .with_new_label(
                    "filtered_input_materialization_strategy",
                    aggregate_fact(facts, "filtered_input_materialization_strategy"),
                )
                .with_new_label(
                    "filtered_input_materialization_estimated_bytes",
                    aggregate_fact(facts, "filtered_input_materialization_estimated_bytes"),
                )
                .with_new_label(
                    "filtered_input_materialization_budget_bytes",
                    aggregate_fact(facts, "filtered_input_materialization_budget_bytes"),
                )
                .with_new_label(
                    "filtered_input_materialization_threshold_bytes",
                    aggregate_fact(facts, "filtered_input_materialization_threshold_bytes"),
                )
                .with_new_label(
                    "filtered_input_lower_level_unresolved_contracts",
                    aggregate_fact(facts, "filtered_input_lower_level_unresolved_contracts"),
                )
                .with_new_label(
                    "filtered_global_rows_per_chunk",
                    aggregate_fact(facts, "filtered_global_rows_per_chunk"),
                )
                .with_new_label(
                    "filtered_global_inputs_per_segment",
                    aggregate_fact(facts, "filtered_global_inputs_per_segment"),
                )
                .with_new_label(
                    "filtered_global_segment_target_temporary_bytes",
                    aggregate_fact(facts, "filtered_global_segment_target_temporary_bytes"),
                )
                .with_new_label(
                    "filtered_global_actual_chunk_count",
                    aggregate_fact(facts, "filtered_global_actual_chunk_count"),
                )
                .with_new_label(
                    "filtered_global_actual_segment_count",
                    aggregate_fact(facts, "filtered_global_actual_segment_count"),
                )
                .with_new_label(
                    "filtered_global_max_segment_rows",
                    aggregate_fact(facts, "filtered_global_max_segment_rows"),
                )
                .with_new_label(
                    "filtered_global_observed_temporary_peak_bytes",
                    aggregate_fact(facts, "filtered_global_observed_temporary_peak_bytes"),
                )
                .with_new_label(
                    "filtered_global_observed_retained_peak_bytes",
                    aggregate_fact(facts, "filtered_global_observed_retained_peak_bytes"),
                )
                .with_new_label(
                    "filtered_global_state_rows_merged",
                    aggregate_fact(facts, "filtered_global_state_rows_merged"),
                )
                .with_new_label(
                    "filtered_grouped_rows_per_chunk",
                    aggregate_fact(facts, "filtered_grouped_rows_per_chunk"),
                )
                .with_new_label(
                    "filtered_grouped_inputs_per_segment",
                    aggregate_fact(facts, "filtered_grouped_inputs_per_segment"),
                )
                .with_new_label(
                    "filtered_grouped_segment_target_temporary_bytes",
                    aggregate_fact(facts, "filtered_grouped_segment_target_temporary_bytes"),
                )
                .with_new_label(
                    "filtered_grouped_actual_chunk_count",
                    aggregate_fact(facts, "filtered_grouped_actual_chunk_count"),
                )
                .with_new_label(
                    "filtered_grouped_actual_segment_count",
                    aggregate_fact(facts, "filtered_grouped_actual_segment_count"),
                )
                .with_new_label(
                    "filtered_grouped_max_segment_rows",
                    aggregate_fact(facts, "filtered_grouped_max_segment_rows"),
                )
                .with_new_label(
                    "filtered_grouped_observed_temporary_peak_bytes",
                    aggregate_fact(facts, "filtered_grouped_observed_temporary_peak_bytes"),
                )
                .with_new_label(
                    "filtered_grouped_observed_retained_peak_bytes",
                    aggregate_fact(facts, "filtered_grouped_observed_retained_peak_bytes"),
                )
                .with_new_label(
                    "filtered_grouped_state_rows_merged",
                    aggregate_fact(facts, "filtered_grouped_state_rows_merged"),
                )
                .with_new_label(
                    "filtered_grouped_high_cardinality_admission",
                    aggregate_fact(facts, "filtered_grouped_high_cardinality_admission"),
                )
                .with_new_label("distinct_aggregate_count", aggregate_fact(facts, "distinct_aggregate_count"))
                .with_new_label("input_rows", aggregate_fact(facts, "input_rows"))
                .with_new_label("output_group_count", aggregate_fact(facts, "output_group_count"))
                .with_new_label("reduction_ratio_ppm", aggregate_fact(facts, "reduction_ratio_ppm"))
                .with_new_label("chunk_row_count", aggregate_fact(facts, "chunk_row_count"))
                .with_new_label("chunk_count", aggregate_fact(facts, "chunk_count"))
                .with_new_label("partial_aggregate_count", aggregate_fact(facts, "partial_aggregate_count"))
                .with_new_label("source_adjacent_partial", aggregate_fact(facts, "source_adjacent_partial"))
                .with_new_label("source_chunk_row_group_count", aggregate_fact(facts, "source_chunk_row_group_count"))
                .with_new_label(
                    "source_chunk_reads_all_row_groups",
                    aggregate_fact(facts, "source_chunk_reads_all_row_groups"),
                )
                .with_new_label("source_chunk_known_row_count", aggregate_fact(facts, "source_chunk_known_row_count"))
                .with_new_label(
                    "source_chunk_estimated_source_read_bytes",
                    aggregate_fact(facts, "source_chunk_estimated_source_read_bytes"),
                )
                .with_new_label(
                    "source_chunk_size_estimate_based",
                    aggregate_fact(facts, "source_chunk_size_estimate_based"),
                )
                .with_new_label(
                    "source_adjacent_partition_strategy",
                    aggregate_fact(facts, "source_adjacent_partition_strategy"),
                )
                .with_new_label(
                    "source_adjacent_partition_count",
                    aggregate_fact(facts, "source_adjacent_partition_count"),
                )
                .with_new_label(
                    "source_adjacent_non_empty_partition_count",
                    aggregate_fact(facts, "source_adjacent_non_empty_partition_count"),
                )
                .with_new_label(
                    "source_adjacent_partition_key_count",
                    aggregate_fact(facts, "source_adjacent_partition_key_count"),
                )
                .with_new_label(
                    "source_adjacent_max_partition_fan_in",
                    aggregate_fact(facts, "source_adjacent_max_partition_fan_in"),
                )
                .with_new_label(
                    "partitioned_merge_partition_count",
                    aggregate_fact(facts, "partitioned_merge_partition_count"),
                )
                .with_new_label(
                    "partitioned_merge_non_empty_partition_count",
                    aggregate_fact(facts, "partitioned_merge_non_empty_partition_count"),
                )
                .with_new_label(
                    "partitioned_merge_max_partition_fan_in",
                    aggregate_fact(facts, "partitioned_merge_max_partition_fan_in"),
                )
                .with_new_label("selection_elapsed_ns", aggregate_fact(facts, "selection_elapsed_ns"))
                .with_new_label("encode_elapsed_ns", aggregate_fact(facts, "encode_elapsed_ns"))
                .with_new_label("groupby_elapsed_ns", aggregate_fact(facts, "groupby_elapsed_ns"))
                .with_new_label("decode_elapsed_ns", aggregate_fact(facts, "decode_elapsed_ns"))
                .with_new_label("strategy_operator_elapsed_ns", aggregate_fact(facts, "strategy_operator_elapsed_ns"))
                .with_new_label("aggregate_structural_key", aggregate_fact(facts, "aggregate_structural_key"))
                .with_new_label(
                    "prior_cardinality_evidence_scope",
                    aggregate_fact(facts, "prior_cardinality_evidence_scope"),
                )
                .with_new_label(
                    "aggregate_strategy_options_version",
                    aggregate_fact(facts, "aggregate_strategy_options_version"),
                )
                .with_new_label(
                    "aggregate_encoded_key_partial_low_cardinality_threshold_ppm",
                    aggregate_fact(facts, "aggregate_encoded_key_partial_low_cardinality_threshold_ppm"),
                )
                .with_new_label(
                    "aggregate_encoded_key_partial_min_input_rows",
                    aggregate_fact(facts, "aggregate_encoded_key_partial_min_input_rows"),
                )
                .with_new_label(
                    "aggregate_exact_grouped_key_cardinality_probe_max_key_columns",
                    aggregate_fact(facts, "aggregate_exact_grouped_key_cardinality_probe_max_key_columns"),
                )
                .with_new_label(
                    "aggregate_exact_single_string_avg_cardinality_probe_min_input_rows",
                    aggregate_fact(facts, "aggregate_exact_single_string_avg_cardinality_probe_min_input_rows"),
                )
                .with_new_label(
                    "aggregate_exact_single_key_cardinality_probe_min_input_rows",
                    aggregate_fact(facts, "aggregate_exact_single_key_cardinality_probe_min_input_rows"),
                )
                .with_new_label(
                    "aggregate_exact_grouped_key_cardinality_probe_min_input_rows",
                    aggregate_fact(facts, "aggregate_exact_grouped_key_cardinality_probe_min_input_rows"),
                )
                .with_new_label(
                    "aggregate_exact_single_string_key_prepared_max_groups",
                    aggregate_fact(facts, "aggregate_exact_single_string_key_prepared_max_groups"),
                )
                .with_new_label(
                    "aggregate_exact_three_key_cardinality_probe_max_input_rows",
                    aggregate_fact(facts, "aggregate_exact_three_key_cardinality_probe_max_input_rows"),
                )
                .with_new_label(
                    "aggregate_exact_five_key_cardinality_probe_max_input_rows",
                    aggregate_fact(facts, "aggregate_exact_five_key_cardinality_probe_max_input_rows"),
                )
                .with_new_label(
                    "aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm",
                    aggregate_fact(facts, "aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm"),
                )
                .with_new_label(
                    "aggregate_source_adjacent_prepared_sort_groupby_max_row_groups",
                    aggregate_fact(facts, "aggregate_source_adjacent_prepared_sort_groupby_max_row_groups"),
                )
                .with_new_label(
                    "aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes",
                    aggregate_fact(facts, "aggregate_source_adjacent_prepared_sort_groupby_max_source_read_bytes"),
                )
                .with_new_label(
                    "aggregate_source_adjacent_prepared_sort_groupby_max_fan_in",
                    aggregate_fact(facts, "aggregate_source_adjacent_prepared_sort_groupby_max_fan_in"),
                )
                .with_new_label(
                    "encoded_key_partial_cost_model",
                    aggregate_fact(facts, "encoded_key_partial_cost_model"),
                )
                .with_new_label(
                    "encoded_key_partial_cost_model_status",
                    aggregate_fact(facts, "encoded_key_partial_cost_model_status"),
                )
                .with_new_label(
                    "encoded_key_partial_selection_reason",
                    aggregate_fact(facts, "encoded_key_partial_selection_reason"),
                )
                .with_new_label(
                    "encoded_key_partial_reduction_ratio_threshold_ppm",
                    aggregate_fact(facts, "encoded_key_partial_reduction_ratio_threshold_ppm"),
                )
                .with_new_label(
                    "encoded_key_partial_prior_reduction_ratio_ppm",
                    aggregate_fact(facts, "encoded_key_partial_prior_reduction_ratio_ppm"),
                )
                .with_new_label(
                    "encoded_key_partial_prior_strategy",
                    aggregate_fact(facts, "encoded_key_partial_prior_strategy"),
                )
                .with_new_label(
                    "exact_cardinality_probe_status",
                    aggregate_fact(facts, "exact_cardinality_probe_status"),
                )
                .with_new_label("exact_cardinality_probe_key", aggregate_fact(facts, "exact_cardinality_probe_key"))
                .with_new_label(
                    "exact_cardinality_probe_input_rows",
                    aggregate_fact(facts, "exact_cardinality_probe_input_rows"),
                )
                .with_new_label(
                    "exact_cardinality_probe_non_null_distinct_count",
                    aggregate_fact(facts, "exact_cardinality_probe_non_null_distinct_count"),
                )
                .with_new_label(
                    "exact_cardinality_probe_null_count",
                    aggregate_fact(facts, "exact_cardinality_probe_null_count"),
                )
                .with_new_label(
                    "exact_cardinality_probe_num_groups",
                    aggregate_fact(facts, "exact_cardinality_probe_num_groups"),
                )
                .with_new_label(
                    "exact_cardinality_probe_reduction_ratio_ppm",
                    aggregate_fact(facts, "exact_cardinality_probe_reduction_ratio_ppm"),
                )
                .with_new_label(
                    "exact_cardinality_probe_elapsed_ns",
                    aggregate_fact(facts, "exact_cardinality_probe_elapsed_ns"),
                )
                .with_new_label(
                    "prepared_sort_groupby_rejected_reason",
                    aggregate_fact(facts, "prepared_sort_groupby_rejected_reason"),
                )
                .with_new_label(
                    "prepared_sort_groupby_contract_status",
                    aggregate_fact(facts, "prepared_sort_groupby_contract_status"),
                )
                .with_new_label(
                    "prepared_sort_groupby_required_contract",
                    aggregate_fact(facts, "prepared_sort_groupby_required_contract"),
                )
                .with_new_label(
                    "prepared_sort_groupby_cost_model",
                    aggregate_fact(facts, "prepared_sort_groupby_cost_model"),
                )
                .with_new_label(
                    "prepared_sort_groupby_cost_model_status",
                    aggregate_fact(facts, "prepared_sort_groupby_cost_model_status"),
                )
                .with_new_label(
                    "prepared_sort_groupby_selection_reason",
                    aggregate_fact(facts, "prepared_sort_groupby_selection_reason"),
                )
                .with_new_label(
                    "prepared_sort_groupby_num_groups",
                    aggregate_fact(facts, "prepared_sort_groupby_num_groups"),
                )
                .with_new_label(
                    "prepared_sort_groupby_grouped_key_output_bytes",
                    aggregate_fact(facts, "prepared_sort_groupby_grouped_key_output_bytes"),
                )
                .with_new_label(
                    "prepared_sort_groupby_max_group_size",
                    aggregate_fact(facts, "prepared_sort_groupby_max_group_size"),
                )
                .with_new_label("rejected_candidate_strategies", aggregate_fact(facts, "rejected_candidate_strategies"))
                .global_counter(metric_name::native_aggregate_strategy_rows())
                .add(1);
            self.native_aggregate_strategy_rows.add(1);
        }
    }

    fn record_boundary_policy(&self, policy: ExecutionBoundaryPolicy) {
        // These counters record which boundary policy was selected for this native query execution,
        // not rows or bytes crossing that boundary.
        if matches!(policy.source_read(), nexus_query_engine::runtime::SourceReadBoundary::DirectDeviceLocalParquet) {
            self.source_read_boundary_direct_device.add(1);
        }
        if matches!(
            policy.build_materialization(),
            nexus_query_engine::runtime::BuildMaterializationBoundary::QueryScopedGpuHashJoin
        ) {
            self.build_materialization_boundary_query_scoped.add(1);
        }
        if matches!(policy.probe_execution(), nexus_query_engine::runtime::ProbeExecutionBoundary::ReusableGpuHashJoin)
        {
            self.probe_execution_boundary_reusable.add(1);
        }
        if matches!(
            policy.result_materialization(),
            nexus_query_engine::runtime::ResultMaterializationBoundary::ExplicitSink
        ) {
            self.result_materialization_boundary_explicit.add(1);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct JoinRuntimeCounters {
    partitioned_hash_join_count: u64,
    hash_join_build_count: u64,
    hash_join_hash_table_bytes: u64,
    filtered_join_build_count: u64,
    filtered_join_hash_table_bytes: u64,
    filtered_join_probe_state_bytes: u64,
    mixed_join_size_data_bytes: u64,
    one_shot_join_count: u64,
    one_shot_join_index_bytes: u64,
    one_shot_join_materialization_bytes: u64,
    one_shot_join_output_bytes: u64,
}

fn optional_fact(facts: Option<&BTreeMap<String, String>>, key: &str) -> String {
    facts.and_then(|facts| facts.get(key).cloned()).unwrap_or_else(|| "none".to_string())
}

fn aggregate_fact(facts: &BTreeMap<String, String>, key: &str) -> String {
    facts.get(key).cloned().unwrap_or_else(|| "none".to_string())
}

fn join_strategy_label(join: &JoinOperatorMetrics) -> &'static str {
    match join.execution_strategy.as_str() {
        "bounded_one_shot_equi_join" => "one_shot_equi",
        "bounded_partitioned_hash_equi_join" => "partitioned_hash_equi",
        "bounded_partitioned_mixed_keyed_residual_join" => "partitioned_mixed_keyed_residual",
        "bounded_reusable_hash_join" | "hash_join" => "reusable_hash_equi",
        "bounded_reusable_filtered_join" => "filtered_join",
        "bounded_mixed_keyed_residual_join" | "mixed_join" => "mixed_keyed_residual",
        "bounded_keyless_conditional_join" | "conditional_join" => "keyless_conditional",
        "bounded_projected_cross_join" | "cross_join" => "projected_cross",
        _ => "unknown",
    }
}

fn join_runtime_counters(join: &JoinOperatorMetrics) -> JoinRuntimeCounters {
    match join_strategy_label(join) {
        "reusable_hash_equi" => JoinRuntimeCounters {
            hash_join_build_count: u64::from(join.hash_table_bytes > 0),
            hash_join_hash_table_bytes: join.hash_table_bytes,
            ..Default::default()
        },
        "filtered_join" => JoinRuntimeCounters {
            filtered_join_build_count: u64::from(join.hash_table_bytes > 0),
            filtered_join_hash_table_bytes: join.hash_table_bytes,
            filtered_join_probe_state_bytes: join.filtered_probe_state_bytes,
            ..Default::default()
        },
        "one_shot_equi" => JoinRuntimeCounters {
            one_shot_join_count: 1,
            one_shot_join_index_bytes: join.gather_map_bytes,
            one_shot_join_materialization_bytes: join.output_reservation_bytes,
            one_shot_join_output_bytes: join.output_bytes,
            ..Default::default()
        },
        "partitioned_hash_equi" => JoinRuntimeCounters { partitioned_hash_join_count: 1, ..Default::default() },
        "partitioned_mixed_keyed_residual" => JoinRuntimeCounters {
            partitioned_hash_join_count: 1,
            mixed_join_size_data_bytes: join.mixed_size_data_bytes,
            ..Default::default()
        },
        "mixed_keyed_residual" => {
            JoinRuntimeCounters { mixed_join_size_data_bytes: join.mixed_size_data_bytes, ..Default::default() }
        },
        _ => JoinRuntimeCounters::default(),
    }
}

fn count(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn join_gather_map_bytes(metrics: &QueryMetrics) -> u64 {
    metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).map(|join| join.gather_map_bytes).sum()
}

fn add_optional_count(metric: &Count, observed: Option<&Count>, value: Option<u64>) {
    if let Some(value) = value {
        // `*_observed` distinguishes a real `Some(0)` sample from no observation at all.
        if let Some(observed) = observed {
            observed.add(1);
        }
        metric.add(count(value));
    }
}

#[cfg(test)]
mod tests {
    use nexus_query_engine::observability::{
        AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptHealthTransition, AttemptId,
        AttemptRetryAdvice, RuntimeTerminalOutcome, SEALED_ATTEMPT_PROJECTION_SCHEMA_V1, SealedAttemptProjection,
    };

    #[test]
    fn sealed_attempt_metric_labels_derive_from_the_sealed_projection() {
        let projection = SealedAttemptProjection {
            schema: SEALED_ATTEMPT_PROJECTION_SCHEMA_V1,
            attempt_id: AttemptId::mint(),
            disposition: AttemptDisposition::Native,
            outcome: RuntimeTerminalOutcome::Panicked,
            cleanup_disposition: AttemptCleanupDisposition::CompletedWithFailures,
            cleanup_failure_count: 7,
            cleanup_failure_count_complete: false,
            capacity_disposition: AttemptCapacityDisposition::RetainedQuarantine,
            health_transition: Some(AttemptHealthTransition::BackendUnhealthy),
            retry_advice: AttemptRetryAdvice::indeterminate(),
            observation_truncated: true,
            observation_degraded: false,
            native_fault_evidence_present: true,
            resource_evidence_present: false,
        };

        assert_eq!(
            projection.labels(),
            nexus_query_engine::observability::SealedAttemptLabels {
                disposition: "native",
                outcome: "panicked",
                cleanup_disposition: "completed_with_failures",
                cleanup_failure_count_complete: "false",
                capacity_disposition: "retained_quarantine",
                health_transition: "backend_unhealthy",
                retry_recovery: "indeterminate",
                retry_prerequisite: "evidence_unavailable",
                observation_truncated: "true",
                observation_degraded: "false",
                native_fault_evidence: "present",
                resource_evidence: "absent",
            }
        );
    }
}
