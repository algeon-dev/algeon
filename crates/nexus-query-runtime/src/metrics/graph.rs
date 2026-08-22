use std::collections::BTreeMap;

use super::{BudgetedExecutionEvent, QueryId, QueryMetrics};
use crate::{
    graph_memory::{
        GraphClaimCalibrationRow, GraphClaimExecutionState, GraphMeasuredPeakProvenance, GraphReservationReport,
    },
    harness::max_option,
    plan::{GraphAlgorithm, NodeId},
};

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
    /// Graph-stage reservation high-water bytes, or CUDA async-pool high-water bytes when reservation evidence is absent.
    /// Successful graph rows that do no graph allocation report zero instead of null.
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

    /// Appends the pending model-claim calibration row for one graph executor
    /// invocation.
    ///
    /// The caller supplies the plan [`NodeId`] that produced `report`, so the
    /// completion sample can correlate by typed node identity rather than
    /// reparsing the rendered `node_id` string. This diagnostic projection is
    /// intentionally independent of whether the graph tier actually reserved
    /// bytes: a non-reserved graph algorithm still needs graph-model calibration
    /// row.
    pub fn record_graph_claim_calibration(&mut self, mut report: GraphReservationReport, node_id: NodeId) {
        if report.query_id.is_none() {
            report.query_id = self.inherited_query_id();
        }
        self.graph_claim_calibration_rows.push(GraphClaimCalibrationRow::from_reservation_report(&report, node_id));
    }

    /// Completes the most recent pending calibration row for the exact graph
    /// node/algorithm invocation.
    ///
    /// The node index and enum are carried directly from the executor. No
    /// peak provenance is inferred from a rendered error, a label, or a
    /// missing numeric value.
    pub fn record_graph_claim_calibration_measurement(
        &mut self,
        node_id: NodeId,
        algorithm: GraphAlgorithm,
        measured_peak_bytes: Option<u64>,
        measured_peak_provenance: GraphMeasuredPeakProvenance,
    ) {
        let Some(row) = self
            .graph_claim_calibration_rows
            .iter_mut()
            .rev()
            .find(|row| row.node_index == node_id.index() && row.algorithm == algorithm)
        else {
            return;
        };
        row.record_measured_peak(measured_peak_bytes, measured_peak_provenance);
    }

    /// Records the exact transition into the native graph algorithm executor.
    /// A claim that remains pending never becomes physical by implication.
    pub fn record_graph_claim_calibration_algorithm_started(&mut self, node_id: NodeId, algorithm: GraphAlgorithm) {
        self.record_graph_claim_calibration_execution_state(
            node_id,
            algorithm,
            GraphClaimExecutionState::AlgorithmStarted,
        );
    }

    /// Records a graph-memory model denial that occurred after its claim was
    /// emitted but before the native graph algorithm executor was entered.
    pub fn record_graph_claim_calibration_graph_memory_admission_denied(
        &mut self,
        node_id: NodeId,
        algorithm: GraphAlgorithm,
    ) {
        self.record_graph_claim_calibration_execution_state(
            node_id,
            algorithm,
            GraphClaimExecutionState::GraphMemoryAdmissionDenied,
        );
    }

    fn record_graph_claim_calibration_execution_state(
        &mut self,
        node_id: NodeId,
        algorithm: GraphAlgorithm,
        execution_state: GraphClaimExecutionState,
    ) {
        let Some(row) = self
            .graph_claim_calibration_rows
            .iter_mut()
            .rev()
            .find(|row| row.node_index == node_id.index() && row.algorithm == algorithm)
        else {
            return;
        };
        row.record_execution_state(execution_state);
    }

    /// Returns the fixed-shape graph claim-versus-peak rows collected for
    /// this execution. The returned projection is read-only and cannot alter
    /// admission state or memory accounting.
    pub fn graph_claim_calibration_rows(&self) -> &[GraphClaimCalibrationRow] {
        &self.graph_claim_calibration_rows
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
        self.record_graph_budget_tracking_with_boundary_and_enforcement(
            operation,
            boundary,
            unresolved_contract,
            budget_bytes,
            "budget_tracking_memory_resource",
            allocator_high_water_bytes,
            status,
            context_facts,
        );
    }

    /// Records a graph-related budget event with caller-supplied enforcement evidence.
    pub fn record_graph_budget_tracking_with_boundary_and_enforcement(
        &mut self,
        operation: impl Into<String>,
        boundary: impl Into<String>,
        unresolved_contract: Option<String>,
        budget_bytes: u64,
        enforcement: impl Into<String>,
        allocator_high_water_bytes: Option<u64>,
        status: impl Into<String>,
        context_facts: BTreeMap<String, String>,
    ) {
        self.max_task_observed_peak_bytes = max_option(self.max_task_observed_peak_bytes, allocator_high_water_bytes);
        self.budgeted_execution_events.push(BudgetedExecutionEvent {
            query_id: self.inherited_query_id(),
            operation: operation.into(),
            boundary: boundary.into(),
            budget_bytes,
            enforcement: enforcement.into(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_memory::{GraphClaimExecutionState, GraphMeasuredPeakProvenance, GraphModelErrorRatio},
        metrics::QueryId,
    };

    fn calibration_row(
        query_id: &str,
        node_index: usize,
        algorithm: GraphAlgorithm,
        model_requested_bytes: u64,
    ) -> GraphClaimCalibrationRow {
        GraphClaimCalibrationRow {
            query_id: Some(QueryId::new(query_id)),
            node_id: format!("n{node_index}"),
            node_index,
            algorithm,
            model_requested_bytes,
            requested_bytes: model_requested_bytes.saturating_add(64),
            budget_bytes: Some(16 * 1024),
            execution_state: GraphClaimExecutionState::Pending,
            measured_peak_bytes: None,
            measured_peak_provenance: GraphMeasuredPeakProvenance::Unavailable,
            graph_model_error_ratio: None,
        }
    }

    #[test]
    fn graph_claim_measurement_correlates_by_typed_node_and_algorithm() {
        let node = NodeId::from_index_for_internal_use(7);
        let mut metrics = QueryMetrics::new("query-a");
        metrics.graph_claim_calibration_rows.push(calibration_row("query-a", node.index(), GraphAlgorithm::Bfs, 1_000));
        metrics.graph_claim_calibration_rows.push(calibration_row(
            "query-b",
            node.index(),
            GraphAlgorithm::PageRank,
            2_000,
        ));

        metrics.record_graph_claim_calibration_measurement(
            node,
            GraphAlgorithm::PageRank,
            Some(1_600),
            GraphMeasuredPeakProvenance::PoolDelta,
        );

        let bfs = &metrics.graph_claim_calibration_rows[0];
        assert_eq!(bfs.query_id.as_ref().map(QueryId::as_str), Some("query-a"));
        assert_eq!(bfs.measured_peak_bytes, None, "same node but different algorithm must not correlate");
        assert_eq!(bfs.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);

        let pagerank = &metrics.graph_claim_calibration_rows[1];
        assert_eq!(pagerank.query_id.as_ref().map(QueryId::as_str), Some("query-b"));
        assert_eq!(pagerank.measured_peak_bytes, Some(1_600));
        assert_eq!(pagerank.measured_peak_provenance, GraphMeasuredPeakProvenance::PoolDelta);
        assert_eq!(pagerank.graph_model_error_ratio, GraphModelErrorRatio::from_requested_and_measured(2_000, 1_600));
    }

    #[test]
    fn graph_claim_execution_state_records_only_the_first_exact_boundary() {
        let node = NodeId::from_index_for_internal_use(7);
        let mut metrics = QueryMetrics::new("query-a");
        metrics.graph_claim_calibration_rows.push(calibration_row("query-a", node.index(), GraphAlgorithm::Bfs, 1_000));

        metrics.record_graph_claim_calibration_graph_memory_admission_denied(node, GraphAlgorithm::Bfs);
        metrics.record_graph_claim_calibration_algorithm_started(node, GraphAlgorithm::Bfs);

        assert_eq!(
            metrics.graph_claim_calibration_rows[0].execution_state,
            GraphClaimExecutionState::GraphMemoryAdmissionDenied,
            "a graph-memory denial must not be rewritten as physical execution"
        );
    }
}
