use query_planning::{
    correlation::QueryId,
    plan::{GraphAlgorithm, NodeId},
};

use super::QueryMetrics;
use crate::observability::graph_observation::{
    GraphClaimExecutionState, GraphMeasuredPeakProvenance, GraphMeasuredPeakRow,
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

    /// Opens a pending measured-peak observation row for one graph executor
    /// invocation.
    ///
    /// The caller supplies the plan [`NodeId`] that owns this graph node, so
    /// the completion sample can correlate by typed node identity rather than
    /// reparsing the rendered `node_id` string.
    pub fn record_graph_claim_calibration(&mut self, node_id: NodeId, algorithm: GraphAlgorithm) {
        self.graph_claim_calibration_rows.push(GraphMeasuredPeakRow::pending(
            self.inherited_query_id(),
            node_id.to_string(),
            node_id.index(),
            algorithm,
        ));
    }

    /// Completes the most recent pending measured-peak row for the exact
    /// graph node/algorithm invocation.
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
        // The query peak is a total; it must not depend on finding the row.
        self.max_task_observed_peak_bytes = self.max_task_observed_peak_bytes.max(measured_peak_bytes);
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
        let Some(row) = self
            .graph_claim_calibration_rows
            .iter_mut()
            .rev()
            .find(|row| row.node_index == node_id.index() && row.algorithm == algorithm)
        else {
            return;
        };
        row.record_execution_state(GraphClaimExecutionState::AlgorithmStarted);
    }

    /// Returns the fixed-shape graph measured-peak rows collected for this
    /// execution. The returned projection is read-only and cannot alter
    /// admission state or memory accounting.
    pub fn graph_claim_calibration_rows(&self) -> &[GraphMeasuredPeakRow] {
        &self.graph_claim_calibration_rows
    }
}

#[cfg(test)]
mod tests {
    use query_planning::correlation::QueryId;

    use super::*;
    use crate::observability::graph_observation::{GraphClaimExecutionState, GraphMeasuredPeakProvenance};

    #[test]
    fn graph_claim_measurement_correlates_by_typed_node_and_algorithm() {
        let node = NodeId::from_index_for_test(7);
        let mut metrics = QueryMetrics::new("query-a");
        metrics.graph_claim_calibration_rows.push(GraphMeasuredPeakRow::pending(
            Some(QueryId::new("query-a")),
            "n7",
            node.index(),
            GraphAlgorithm::Bfs,
        ));
        metrics.graph_claim_calibration_rows.push(GraphMeasuredPeakRow::pending(
            Some(QueryId::new("query-b")),
            "n7",
            node.index(),
            GraphAlgorithm::PageRank,
        ));

        metrics.record_graph_claim_calibration_measurement(
            node,
            GraphAlgorithm::PageRank,
            Some(1_600),
            GraphMeasuredPeakProvenance::CacheScopeHighWater,
        );

        let bfs = &metrics.graph_claim_calibration_rows[0];
        assert_eq!(bfs.query_id.as_ref().map(QueryId::as_str), Some("query-a"));
        assert_eq!(bfs.measured_peak_bytes, None, "same node but different algorithm must not correlate");
        assert_eq!(bfs.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);

        let pagerank = &metrics.graph_claim_calibration_rows[1];
        assert_eq!(pagerank.query_id.as_ref().map(QueryId::as_str), Some("query-b"));
        assert_eq!(pagerank.measured_peak_bytes, Some(1_600));
        assert_eq!(pagerank.measured_peak_provenance, GraphMeasuredPeakProvenance::CacheScopeHighWater);
    }

    #[test]
    fn graph_claim_execution_state_records_only_the_first_exact_boundary() {
        let node = NodeId::from_index_for_test(7);
        let mut metrics = QueryMetrics::new("query-a");
        metrics.graph_claim_calibration_rows.push(GraphMeasuredPeakRow::pending(
            Some(QueryId::new("query-a")),
            "n7",
            node.index(),
            GraphAlgorithm::Bfs,
        ));

        metrics.record_graph_claim_calibration_algorithm_started(node, GraphAlgorithm::Bfs);
        metrics.record_graph_claim_calibration_algorithm_started(node, GraphAlgorithm::Bfs);

        assert_eq!(
            metrics.graph_claim_calibration_rows[0].execution_state,
            GraphClaimExecutionState::AlgorithmStarted,
            "an exact boundary must not be rewritten by a later call"
        );
    }
}
