//! Native query-metric recording into DataFusion wrapper metrics.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::{self, Write},
    ops::Deref,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    physical_plan::metrics::{BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet},
};
use query_engine::exec::output::{PlanMetrics, QueryExecutionOutcome, QueryOutput};
use query_planning::error::QueryError;
use query_runtime::{
    metrics::{JoinOperatorMetrics, QueryMetrics},
    observability::{
        AttemptId, LakehouseCacheFillEventSnapshot, QueryAttemptReport, RuntimeObservationConfig,
        graph_observation::{GraphClaimExecutionState, GraphMeasuredPeakRow},
    },
};

use super::{
    SERVER_CAPTURE_LOSS_METRIC_NAME, capture::NativeExecutionMetricsCapture, contract::metric_name,
    registry::NativeMetricRegistry,
};
use crate::{
    native_exec::{admission::QueryAdmissionFacts, config::AlgeonNativeExecutionContext},
    native_report::NativeDataFusionExecutionMetrics,
};

mod mapping;

#[cfg(test)]
pub(super) use mapping::NATIVE_ALGEON_RECORDING_DESCRIPTORS;

/// A root-native-wrapper target that can receive one attempt projection after
/// its owner has sealed the immutable report. It does not retain the report,
/// own attempt lifecycle state, or expose unbounded labels.
#[derive(Debug, Clone)]
pub(crate) struct SealedAttemptMetricsTarget {
    metrics: ExecutionPlanMetricsSet,
    graph_claim_calibration_captures: Vec<GraphClaimCalibrationCapture>,
    /// Every native wrapper's live metric set projected at seal, starting with
    /// the elected root's own set. Only the root receives the attempt row.
    execution_metrics: Vec<ExecutionPlanMetricsSet>,
}

impl SealedAttemptMetricsTarget {
    pub(crate) fn new(
        metrics: &ExecutionPlanMetricsSet,
        graph_claim_calibration_capture: GraphClaimCalibrationCapture,
    ) -> Self {
        Self {
            metrics: metrics.clone(),
            graph_claim_calibration_captures: vec![graph_claim_calibration_capture],
            execution_metrics: vec![metrics.clone()],
        }
    }

    /// Adds the metric sets of the other native wrappers that execute under
    /// this attempt, such as producers and relation functions below a composed
    /// root.
    pub(crate) fn with_execution_metrics(mut self, metrics: impl IntoIterator<Item = ExecutionPlanMetricsSet>) -> Self {
        self.execution_metrics.extend(metrics);
        self
    }

    pub(crate) fn with_graph_claim_calibration_captures(
        mut self,
        captures: impl IntoIterator<Item = GraphClaimCalibrationCapture>,
    ) -> Self {
        self.graph_claim_calibration_captures.extend(captures);
        self
    }

    pub(crate) fn graph_claim_calibration_captures(&self) -> impl Iterator<Item = GraphClaimCalibrationCapture> + '_ {
        self.graph_claim_calibration_captures.iter().cloned()
    }

    /// Emit the single fixed-cardinality DataFusion row for this attempt and
    /// project the attempt's native execution metrics.
    ///
    /// The caller elects exactly one root wrapper before execution. All labels
    /// are derived from the sealed projection; in particular, the attempt id and device
    /// ordinal deliberately never become metric labels.
    ///
    /// Each wrapper's metrics project separately and merge with
    /// [`NativeDataFusionExecutionMetrics::merge_from`]: cumulative counters
    /// add and peaks keep the maximum, the rule the engine also applies when
    /// it merges `QueryMetrics`. Admission sequence, wait, device ordinal, and
    /// grant come from the sealed report, which records the attempt's one
    /// admission even when several wrappers observed it.
    pub(crate) fn record(
        self,
        report: &QueryAttemptReport,
    ) -> (SealedAttemptGraphClaimCalibrationSnapshot, NativeDataFusionExecutionMetrics) {
        let execution = sealed_execution_metrics(&self.execution_metrics, report);
        let graph_claim_calibration = SealedAttemptGraphClaimCalibrationSnapshot::from_captures(
            report.attempt_id,
            self.graph_claim_calibration_captures,
        );
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
        (graph_claim_calibration, execution)
    }
}

fn sealed_execution_metrics(
    sets: &[ExecutionPlanMetricsSet],
    report: &QueryAttemptReport,
) -> NativeDataFusionExecutionMetrics {
    let mut seen = HashSet::new();
    let mut execution: Option<NativeDataFusionExecutionMetrics> = None;
    for set in sets {
        // Execution clones and delegating wrappers expose the same metric
        // handles; each handle counts once.
        let mut wrapper = MetricsSet::new();
        for metric in set.clone_inner().iter() {
            if seen.insert(Arc::as_ptr(metric)) {
                wrapper.push(Arc::clone(metric));
            }
        }
        if wrapper.iter().next().is_none() {
            continue;
        }
        let projected = NativeDataFusionExecutionMetrics::from_metrics(&wrapper);
        match execution.as_mut() {
            Some(execution) => execution.merge_from(projected),
            None => execution = Some(projected),
        }
    }
    let mut execution = execution.unwrap_or_default();
    let admission = report.admission.as_ref();
    execution.admission_sequence = admission.map(|admission| count(admission.admission_sequence));
    execution.admission_wait_time_ns = admission.map(|admission| count(admission.admission_wait_ns));
    execution.device_ordinal = report.device_ordinal;
    execution.query_grant_bytes = admission.map(|admission| count(admission.grant_bytes));
    execution
}

/// Maximum graph calibration rows retained by one public sealed-attempt
/// snapshot. The engine can retain richer diagnostic metrics internally, but
/// the adapter projection stays bounded for runner handoff.
pub const SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS: usize = 64;

/// Immutable, typed graph claim-versus-peak rows projected by one sealed
/// attempt.
///
/// Rows originate directly from engine [`QueryMetrics`]. The adapter neither
/// parses DataFusion metric labels nor uses this snapshot for admission,
/// retries, or device accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedAttemptGraphClaimCalibrationSnapshot {
    attempt_id: AttemptId,
    rows: Box<[GraphMeasuredPeakRow]>,
    truncated: bool,
    execution: Option<GraphClaimCalibrationExecution>,
    execution_complete: bool,
}

/// Typed graph-execution classification sealed with an attempt's calibration
/// rows. A physical classification requires the engine to record that the
/// graph executor started; peak qualification remains fail-closed downstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphClaimCalibrationExecution {
    Physical,
    NonPhysical(GraphClaimCalibrationNonPhysicalReason),
}

/// Closed vocabulary for graph attempts that intentionally did not execute a
/// graph algorithm locally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphClaimCalibrationNonPhysicalReason {
    /// This exact BFS request reused a completed leader result. The leader's
    /// engine-minted attempt id is retained when the request was attempt-bound
    /// so qualification can require that leader's physical peak evidence.
    BfsSingleflightFollower { leader_attempt_id: Option<AttemptId> },
}

impl SealedAttemptGraphClaimCalibrationSnapshot {
    fn from_captures(attempt_id: AttemptId, captures: impl IntoIterator<Item = GraphClaimCalibrationCapture>) -> Self {
        let mut rows = Vec::new();
        let mut truncated = false;
        let mut execution = None;
        let mut execution_complete = true;
        for capture in captures {
            let captured = capture.seal();
            truncated |= captured.truncated;
            if let Some(captured_execution) = captured.execution {
                // A composed root contributes an otherwise empty capture while
                // its graph producer owns the classification. An absent
                // classification is neutral here; only a reported, conflicting
                // classification can make the merged projection incomplete.
                execution_complete &= captured.execution_complete;
                match execution.as_ref() {
                    // The sealed target may receive the same capture through a
                    // composed root and its graph producer. Classification is
                    // state, not an event, so identical observations coalesce.
                    Some(existing) if existing == &captured_execution => {},
                    Some(_) => {
                        execution = Some(captured_execution);
                        execution_complete = false;
                    },
                    None => execution = Some(captured_execution),
                }
            }
            let remaining = SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS.saturating_sub(rows.len());
            if captured.rows.len() > remaining {
                truncated = true;
            }
            rows.extend(captured.rows.into_iter().take(remaining));
        }
        Self { attempt_id, rows: rows.into_boxed_slice(), truncated, execution, execution_complete }
    }

    /// Attempt that owns this immutable diagnostic projection.
    pub const fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    /// Typed engine rows in execution observation order.
    pub fn rows(&self) -> &[GraphMeasuredPeakRow] {
        &self.rows
    }

    /// Whether the adapter omitted rows beyond its fixed public bound.
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Explicit BFS follower classification or the engine-row-derived graph
    /// executor state. `None` remains fail-closed when no exact boundary was
    /// observed.
    pub fn execution(&self) -> Option<&GraphClaimCalibrationExecution> {
        self.execution.as_ref()
    }

    /// Whether the execution owner recorded exactly one non-conflicting
    /// classification before the attempt sealed.
    pub const fn execution_complete(&self) -> bool {
        self.execution_complete && self.execution.is_some()
    }
}

/// Per-wrapper typed calibration collector. It is shared by execution clones,
/// then frozen exactly once by the elected sealed-attempt target.
#[derive(Debug, Clone, Default)]
pub(crate) struct GraphClaimCalibrationCapture {
    state: Arc<Mutex<GraphClaimCalibrationCaptureState>>,
}

#[derive(Debug, Default)]
struct GraphClaimCalibrationCaptureState {
    rows: Vec<GraphMeasuredPeakRow>,
    truncated: bool,
    execution: Option<GraphClaimCalibrationExecution>,
    execution_complete: bool,
    sealed: bool,
}

struct CapturedGraphMeasuredPeakRows {
    rows: Vec<GraphMeasuredPeakRow>,
    truncated: bool,
    execution: Option<GraphClaimCalibrationExecution>,
    execution_complete: bool,
}

impl GraphClaimCalibrationCapture {
    /// Records a completed exact-request BFS follower. The follower did not
    /// execute a graph algorithm; it must point at its physical leader rather
    /// than borrow or synthesize that leader's peak.
    #[cfg(feature = "cugraph")]
    pub(crate) fn record_bfs_singleflight_follower(&self, leader_attempt_id: Option<AttemptId>) {
        self.record_execution(GraphClaimCalibrationExecution::NonPhysical(
            GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower { leader_attempt_id },
        ));
    }

    #[cfg(feature = "cugraph")]
    fn record_execution(&self, execution: GraphClaimCalibrationExecution) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.sealed {
            return;
        }
        match state.execution.as_ref() {
            // A wrapper can reach the same physical boundary through nested
            // execution helpers. Repeating the identical fact does not make
            // the sealed ownership classification ambiguous.
            Some(existing) if existing == &execution => {},
            Some(_) => {
                state.execution = Some(execution);
                state.execution_complete = false;
            },
            None => {
                state.execution = Some(execution);
                state.execution_complete = true;
            },
        }
    }

    pub(crate) fn record(&self, rows: &[GraphMeasuredPeakRow]) {
        if rows.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.sealed {
            return;
        }
        let remaining = SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS.saturating_sub(state.rows.len());
        if rows.len() > remaining {
            state.truncated = true;
        }
        state.rows.extend(rows.iter().take(remaining).cloned());
    }

    fn seal(&self) -> CapturedGraphMeasuredPeakRows {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.sealed = true;
        let (row_execution, row_execution_complete) = graph_claim_execution_from_rows(&state.rows);
        let (execution, execution_complete) = match state.execution.clone() {
            // BFS followers have an engine-minted request identity rather than
            // a local graph row. Keep that explicit route distinct from the
            // row-derived graph-executor lifecycle.
            Some(execution) => (Some(execution), state.execution_complete),
            None => (row_execution, row_execution_complete),
        };
        CapturedGraphMeasuredPeakRows {
            rows: state.rows.clone(),
            truncated: state.truncated,
            execution,
            execution_complete,
        }
    }
}

/// Derives an attempt classification only from terminal states authored by the
/// engine's graph-claim row. Pending or inconsistent rows never become a
/// physical graph execution at the adapter boundary.
fn graph_claim_execution_from_rows(rows: &[GraphMeasuredPeakRow]) -> (Option<GraphClaimCalibrationExecution>, bool) {
    let mut execution = None;
    let mut complete = !rows.is_empty();
    for row in rows {
        let row_execution = match row.execution_state {
            GraphClaimExecutionState::Pending => {
                complete = false;
                continue;
            },
            GraphClaimExecutionState::AlgorithmStarted => GraphClaimCalibrationExecution::Physical,
        };
        match execution.as_ref() {
            Some(existing) if existing == &row_execution => {},
            Some(_) => {
                execution = Some(row_execution);
                complete = false;
            },
            None => execution = Some(row_execution),
        }
    }
    let complete = complete && execution.is_some();
    (execution, complete)
}

/// Private translation layer from native QueryMetrics into DataFusion metrics.
#[derive(Debug, Clone)]
pub(crate) struct AlgeonNativeMetrics {
    metrics_set: ExecutionPlanMetricsSet,
    plan_task_mapping_datafusion_shape: Option<String>,
    graph_claim_calibration_capture: Option<GraphClaimCalibrationCapture>,
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
            || RuntimeObservationConfig::native_execution().limits(),
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
impl Deref for AlgeonNativeMetrics {
    type Target = NativeMetricRegistry;

    fn deref(&self) -> &Self::Target {
        &self.registry
    }
}

impl AlgeonNativeMetrics {
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
            plan_task_mapping_datafusion_shape: plan_task_mapping_datafusion_shape.map(ToString::to_string),
            graph_claim_calibration_capture: None,
            baseline: BaselineMetrics::new(metrics, partition),
            registry: NativeMetricRegistry::register(metrics, partition),
            row_capture_budget,
            row_capture_loss,
            row_capture_loss_reported: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Attach the elected wrapper's typed graph calibration collector.
    ///
    /// This is intentionally separate from DataFusion row metrics: it copies
    /// engine rows before label rendering and has no admission authority.
    pub(crate) fn with_graph_claim_calibration_capture(mut self, capture: GraphClaimCalibrationCapture) -> Self {
        self.graph_claim_calibration_capture = Some(capture);
        self
    }

    fn admit_metric_row(&self, row: &impl fmt::Debug) -> bool {
        let admitted = self.row_capture_budget.lock().unwrap_or_else(std::sync::PoisonError::into_inner).admit(row);
        if !admitted {
            self.report_row_capture_loss();
        }
        admitted
    }

    /// Admits a detail row the metrics owner already selected under its
    /// retention limits. A projected value is recorded as-is; selecting its
    /// rows a second time would lose rows the owner kept. Only a value built
    /// outside an owner still goes through this recorder's own budget.
    fn admit_owner_selected_row(&self, metrics: &QueryMetrics, row: &impl fmt::Debug) -> bool {
        metrics.projection.is_some() || self.admit_metric_row(row)
    }

    fn report_row_capture_loss(&self) {
        if !self.row_capture_loss_reported.swap(true, Ordering::Relaxed) {
            self.row_capture_loss.add(1);
        }
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
        if metrics.projection.as_ref().is_some_and(|projection| projection.completeness.truncated) {
            self.report_row_capture_loss();
        }
        if let Some(capture) = &self.graph_claim_calibration_capture {
            capture.record(metrics.graph_claim_calibration_rows());
        }
        self.record_descriptor_query_metrics(metrics);
    }

    /// Records every evidence surface before returning the native result to the
    /// adapter's `QueryError -> DataFusionError` conversion boundary.
    pub(crate) fn finalize_query_outcome(
        &self,
        outcome: QueryExecutionOutcome,
    ) -> std::result::Result<QueryOutput, QueryError> {
        let (result, plan, _observation) = outcome.into_parts();
        self.record_query_metrics(&plan.freeze());
        result
    }

    /// Finalizes a native outcome whose logical host sink the adapter's
    /// bounded GPU-to-host cursor runs later. A failed plan records its
    /// metrics now; a successful one hands them, with `context_lease`, to the
    /// cursor. Either way the metrics freeze before the lease is released.
    pub(crate) fn finalize_query_outcome_for_deferred_host(
        &self,
        outcome: QueryExecutionOutcome,
        context_lease: AlgeonNativeExecutionContext,
    ) -> std::result::Result<(QueryOutput, DeferredHostMetrics), QueryError> {
        let (result, plan, _observation) = outcome.into_parts();
        match result {
            Ok(output) => Ok((output, DeferredHostMetrics::new(self.clone(), plan, context_lease))),
            Err(error) => {
                self.record_query_metrics(&plan.freeze());
                Err(error)
            },
        }
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
            if !self.admit_owner_selected_row(metrics, row) {
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
            if !self.admit_owner_selected_row(metrics, row) {
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
            if !self.admit_owner_selected_row(metrics, decision) {
                continue;
            }
            decision_facts_by_node
                .entry(decision.node_id.as_str())
                .or_default()
                .extend(decision.facts.iter().map(|(key, value)| (key.clone(), value.clone())));
        }
        for operator in metrics.operators.iter().filter(|operator| operator.operator_kind == "join") {
            if !self.admit_owner_selected_row(metrics, operator) {
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
                .with_new_label(
                    "join_output_reservation_bytes",
                    join.output_reservation_bytes.map_or_else(|| "none".to_string(), |bytes| bytes.to_string()),
                )
                .with_new_label("join_output_rows", join.output_rows.to_string())
                .with_new_label("join_output_bytes", join.output_bytes.to_string());
            builder.global_counter(metric_name::native_join_strategy_rows()).add(1);
        }
    }

    fn record_aggregate_strategy_rows(&self, metrics: &QueryMetrics) {
        for decision in &metrics.aggregate_strategy_decisions {
            if !self.admit_owner_selected_row(metrics, decision) {
                continue;
            }
            let facts = &decision.facts;
            MetricBuilder::new(&self.metrics_set)
                .with_type(MetricType::Summary)
                .with_new_label("node_id", decision.node_id.clone())
                .with_new_label("selected_strategy", aggregate_fact(facts, "selected_strategy"))
                .with_new_label("execution_stage", aggregate_fact(facts, "execution_stage"))
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

/// The metrics of a plan whose host sink runs in the adapter's cursor.
///
/// The cursor records the sink's work into the plan through its recorder.
/// The plan's metrics freeze and reach DataFusion once, when this value
/// drops: at the end of the output, on a cursor failure, or when the consumer
/// stops early.
#[derive(Debug)]
pub(crate) struct DeferredHostMetrics {
    metrics: AlgeonNativeMetrics,
    plan: PlanMetrics,
    compiled_sink_pending: bool,
    // The admitted context of the cursor's steps. A field drops after
    // `Drop::drop` freezes the metrics, so the attempt cannot start its
    // release before they are recorded.
    _context_lease: AlgeonNativeExecutionContext,
}

impl DeferredHostMetrics {
    fn new(metrics: AlgeonNativeMetrics, plan: PlanMetrics, context_lease: AlgeonNativeExecutionContext) -> Self {
        let compiled_sink_pending = plan.snapshot().compiled_pipeline_attempts > 0;
        Self { metrics, plan, compiled_sink_pending, _context_lease: context_lease }
    }

    pub(crate) const fn baseline(&self) -> &BaselineMetrics {
        &self.metrics.baseline
    }

    /// Records the host sink a compiled pipeline deferred to the adapter, once,
    /// when the cursor has run it: after its first batch, or at the end of an
    /// output that had none.
    pub(crate) fn record_sink_executed(&mut self) {
        if std::mem::take(&mut self.compiled_sink_pending) {
            let recorder = self.plan.recorder();
            recorder.record_compiled_pipeline_tasks_executed(1);
            recorder.record_compiled_pipeline_host_arrow_sink();
        }
    }

    pub(crate) fn record_materialization(&self, batch: &RecordBatch) {
        self.plan.recorder().record_device_to_host_materialization(
            u64::try_from(batch.num_rows()).unwrap_or(u64::MAX),
            u64::try_from(get_record_batch_memory_size(batch)).unwrap_or(u64::MAX),
        );
    }
}

impl Drop for DeferredHostMetrics {
    fn drop(&mut self) {
        self.metrics.record_query_metrics(&self.plan.freeze());
    }
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
            // Counters sum the bounds that exist; an unbounded output adds none.
            one_shot_join_materialization_bytes: join.output_reservation_bytes.unwrap_or_default(),
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
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use query_planning::plan::GraphAlgorithm;
    use query_runtime::{
        metrics::QueryMetrics,
        observability::{
            AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptHealthTransition,
            AttemptId, AttemptRetryAdvice, RuntimeTerminalOutcome, SEALED_ATTEMPT_PROJECTION_SCHEMA,
            SealedAttemptLabels, SealedAttemptProjection,
            graph_observation::{GraphClaimExecutionState, GraphMeasuredPeakProvenance, GraphMeasuredPeakRow},
        },
    };

    use super::*;

    fn calibration_row(index: usize, algorithm: GraphAlgorithm) -> GraphMeasuredPeakRow {
        let mut row = GraphMeasuredPeakRow::pending(None, format!("n{index}"), index, algorithm);
        row.record_execution_state_for_test(GraphClaimExecutionState::AlgorithmStarted);
        row.record_measured_peak(Some(512 + index as u64), GraphMeasuredPeakProvenance::CacheScopeHighWater);
        row
    }

    #[test]
    fn sealed_graph_calibration_snapshot_keeps_typed_rows_and_fixed_bound() {
        let metrics_set = ExecutionPlanMetricsSet::new();
        let capture = GraphClaimCalibrationCapture::default();
        let metrics = AlgeonNativeMetrics::new(&metrics_set, 0).with_graph_claim_calibration_capture(capture.clone());
        let mut query_metrics = QueryMetrics::default();
        for index in 0..=SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS {
            query_metrics.graph_claim_calibration_rows.push(calibration_row(index, GraphAlgorithm::PageRank));
        }

        metrics.record_query_metrics(&query_metrics);
        let attempt_id = AttemptId::mint();
        let snapshot = SealedAttemptGraphClaimCalibrationSnapshot::from_captures(attempt_id, [capture]);

        assert_eq!(snapshot.attempt_id(), attempt_id);
        assert!(snapshot.execution_complete());
        assert!(matches!(snapshot.execution(), Some(GraphClaimCalibrationExecution::Physical)));
        assert_eq!(snapshot.rows().len(), SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS);
        assert!(snapshot.truncated());
        assert_eq!(snapshot.rows()[0].node_index, 0);
        assert_eq!(snapshot.rows()[0].algorithm, GraphAlgorithm::PageRank);
        assert_eq!(snapshot.rows()[0].measured_peak_provenance, GraphMeasuredPeakProvenance::CacheScopeHighWater);
        assert_eq!(
            snapshot.rows()[SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS - 1].node_index,
            SEALED_ATTEMPT_GRAPH_CLAIM_CALIBRATION_MAX_ROWS - 1
        );

        let mut late_metrics = QueryMetrics::default();
        late_metrics.graph_claim_calibration_rows.push(calibration_row(999, GraphAlgorithm::Bfs));
        metrics.record_query_metrics(&late_metrics);
        assert!(snapshot.rows().iter().all(|row| row.node_index != 999));
    }

    #[test]
    #[cfg(feature = "cugraph")]
    fn sealed_graph_calibration_snapshot_retains_a_nonphysical_follower_leader_link() {
        let capture = GraphClaimCalibrationCapture::default();
        let leader_attempt_id = AttemptId::mint();
        capture.record_bfs_singleflight_follower(Some(leader_attempt_id));

        let snapshot = SealedAttemptGraphClaimCalibrationSnapshot::from_captures(AttemptId::mint(), [capture]);

        assert!(snapshot.execution_complete());
        assert!(snapshot.rows().is_empty());
        assert_eq!(
            snapshot.execution(),
            Some(&GraphClaimCalibrationExecution::NonPhysical(
                GraphClaimCalibrationNonPhysicalReason::BfsSingleflightFollower {
                    leader_attempt_id: Some(leader_attempt_id),
                },
            ))
        );
    }

    #[test]
    fn sealed_graph_calibration_snapshot_uses_the_physical_producer_classification_from_a_composed_root() {
        let root_capture = GraphClaimCalibrationCapture::default();
        let producer_capture = GraphClaimCalibrationCapture::default();
        producer_capture.record(&[calibration_row(1, GraphAlgorithm::PageRank)]);

        let snapshot = SealedAttemptGraphClaimCalibrationSnapshot::from_captures(
            AttemptId::mint(),
            [root_capture, producer_capture],
        );

        assert!(snapshot.execution_complete());
        assert!(matches!(snapshot.execution(), Some(GraphClaimCalibrationExecution::Physical)));
    }

    #[test]
    fn sealed_attempt_metric_labels_derive_from_the_sealed_projection() {
        let projection = SealedAttemptProjection {
            schema: SEALED_ATTEMPT_PROJECTION_SCHEMA,
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
            SealedAttemptLabels {
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
