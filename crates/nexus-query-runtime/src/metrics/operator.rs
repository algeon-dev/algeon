use super::{JoinOperatorMetrics, QueryId, QueryMetrics};

/// Generic operator metric row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperatorMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Native node id.
    pub node_id: String,
    /// Operator-kind label.
    pub operator_kind: String,
    /// Input rows consumed by the operator.
    pub input_rows: u64,
    /// Input bytes consumed by the operator.
    pub input_bytes: u64,
    /// Output rows produced by the operator.
    pub output_rows: u64,
    /// Output bytes produced by the operator.
    pub output_bytes: u64,
    /// Join-specific metrics when this is a join row.
    pub join: Option<JoinOperatorMetrics>,
}

impl OperatorMetrics {
    /// Creates a join operator metric row.
    pub fn join(node_id: impl ToString, join: JoinOperatorMetrics) -> Self {
        Self { node_id: node_id.to_string(), operator_kind: "join".to_string(), join: Some(join), ..Default::default() }
    }
}

impl QueryMetrics {
    /// Appends one operator metric row.
    pub fn record_operator_metrics(&mut self, mut metrics: OperatorMetrics) {
        if metrics.query_id.is_none() {
            metrics.query_id = self.inherited_query_id();
        }
        if let Some(join) = &mut metrics.join
            && join.query_id.is_none()
        {
            join.query_id = metrics.query_id.clone();
        }
        self.operators.push(metrics);
    }
}
