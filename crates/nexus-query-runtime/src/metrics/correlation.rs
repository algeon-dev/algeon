use nexus_query_planning::correlation::{CompiledTaskId, PlanNodeId, QueryId};

/// One joinable diagnostic mapping between native plan structure and compiled tasks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanTaskMapping {
    /// Query id shared by the metrics and report rows.
    pub query_id: QueryId,
    /// Native plan node id, when known.
    pub plan_node_id: Option<PlanNodeId>,
    /// Compiled task id, when known.
    pub compiled_task_id: Option<CompiledTaskId>,
    /// Native operator/task kind, when known.
    pub operator_kind: Option<String>,
    /// DataFusion shape label, when an adapter layer supplies it.
    pub datafusion_shape: Option<String>,
}

pub(super) fn inherited_query_id(query_id: &QueryId) -> Option<QueryId> {
    query_id.is_assigned().then(|| query_id.clone())
}
