use std::collections::BTreeMap;

use super::{BudgetedExecutionEvent, QueryId, QueryMetrics};
use crate::{graph_memory::GraphReservationReport, harness::max_option};

/// cuGraph algorithm runtime metric row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphAlgorithmRuntimeMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
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
    pub input_rows: u64,
    /// Input column count.
    pub input_columns: u64,
    /// Output row count.
    pub output_rows: u64,
    /// Output column count.
    pub output_columns: u64,
    /// Output device bytes when known.
    pub output_device_bytes: Option<u64>,
    /// Graph-stage CUDA async-pool high-water bytes above pre-stage current use, when observed.
    ///
    /// If physical pool usage is unavailable, execution falls back to the
    /// graph-boundary budget-tracking high-water. Successful graph rows that do
    /// no graph allocation report zero instead of null.
    pub graph_peak_bytes: Option<u64>,
    /// Runtime elapsed nanoseconds.
    pub elapsed_ns: u64,
    /// GPU-resident input handoffs performed for this algorithm.
    pub gpu_resident_input_handoffs: u64,
    /// Host-to-device uploads performed for this algorithm.
    pub host_to_device_uploads: u64,
    /// Device-to-host materializations performed for this algorithm.
    pub device_to_host_materializations: u64,
}

impl QueryMetrics {
    /// Appends one graph algorithm runtime row.
    pub fn record_graph_algorithm_runtime(&mut self, mut metrics: GraphAlgorithmRuntimeMetrics) {
        if metrics.query_id.is_none() {
            metrics.query_id = self.inherited_query_id();
        }
        self.graph_algorithm_runtime_rows.push(metrics);
    }

    /// Appends one graph memory-tier reservation row.
    pub fn record_graph_reservation(&mut self, mut report: GraphReservationReport) {
        if report.query_id.is_none() {
            report.query_id = self.inherited_query_id();
        }
        self.graph_reservation_rows.push(report);
    }

    /// Records model error evidence on the matching graph reservation row.
    pub fn record_graph_model_error_ratio(&mut self, node_id: &str, algorithm: &str, measured_peak_bytes: Option<u64>) {
        let Some(measured_peak_bytes) = measured_peak_bytes else {
            return;
        };
        for row in self.graph_reservation_rows.iter_mut().rev() {
            if row.node_id == node_id && row.algorithm.as_str() == algorithm {
                row.record_model_error_ratio(measured_peak_bytes);
                return;
            }
        }
    }

    /// Records a graph execution bounded by a budget-tracking RMM resource.
    pub fn record_graph_budget_tracking(
        &mut self,
        operation: impl Into<String>,
        budget_bytes: u64,
        allocator_high_water_bytes: Option<u64>,
        status: impl Into<String>,
        context_facts: BTreeMap<String, String>,
    ) {
        self.record_graph_budget_tracking_with_boundary(
            operation,
            crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
            Some("cugraph_workspace_preflight".to_string()),
            budget_bytes,
            allocator_high_water_bytes,
            status,
            context_facts,
        );
    }

    /// Records a graph-related budget-tracking RMM resource with an explicit boundary.
    pub fn record_graph_budget_tracking_with_boundary(
        &mut self,
        operation: impl Into<String>,
        boundary: impl Into<String>,
        unresolved_contract: Option<String>,
        budget_bytes: u64,
        allocator_high_water_bytes: Option<u64>,
        status: impl Into<String>,
        context_facts: BTreeMap<String, String>,
    ) {
        self.max_task_observed_peak_bytes = max_option(self.max_task_observed_peak_bytes, allocator_high_water_bytes);
        if let Some(compiled_task_id) = self.current_compiled_task_id.clone() {
            self.record_task_budget_observation(&compiled_task_id, allocator_high_water_bytes, None);
        }
        self.budgeted_execution_events.push(BudgetedExecutionEvent {
            query_id: self.inherited_query_id(),
            operation: operation.into(),
            boundary: boundary.into(),
            budget_bytes,
            enforcement: "budget_tracking_memory_resource".to_string(),
            status: status.into(),
            allocator_high_water_bytes,
            retained_bytes: None,
            output_bytes: None,
            output_rows: None,
            unresolved_contract,
            context_facts,
        });
    }
}
