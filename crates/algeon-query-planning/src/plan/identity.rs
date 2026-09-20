use super::{PlanNode, QueryPlan};
use crate::source::SourceNode;

fn canonical_execution_scope(plan: &QueryPlan) -> QueryPlan {
    let mut plan = plan.clone_with_deferred_host_materialization_sink();
    plan.advisory_outputs.clear();
    plan
}

#[doc(hidden)]
pub fn canonical_execution_scope_fingerprint(plan: &QueryPlan) -> String {
    format!("{:#?}", canonical_execution_scope(plan))
}

/// Stable identity for a graph fragment whose missing source cardinality facts
/// may be completed after Arrow staging.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct DeferredGraphExecutionScopeIdentity {
    execution_scope_fingerprint: String,
    source_fact_slots: Vec<DeferredGraphSourceFactSlots>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeferredGraphSourceFactSlots {
    node_index: usize,
    row_count_upper_bound: bool,
    graph_vertex_upper_bound: bool,
}

impl DeferredGraphExecutionScopeIdentity {
    pub(crate) fn for_plan(plan: &QueryPlan) -> Self {
        let canonical = canonical_execution_scope(plan);
        let source_fact_slots = canonical
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(node_index, node)| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) => {
                        let slots = DeferredGraphSourceFactSlots {
                            node_index,
                            row_count_upper_bound: source.row_count_upper_bound.is_none(),
                            graph_vertex_upper_bound: source.graph_vertex_upper_bound.is_none(),
                        };
                        (slots.row_count_upper_bound || slots.graph_vertex_upper_bound).then_some(slots)
                    },
                    _ => None,
                },
                _ => None,
            })
            .collect();
        Self { execution_scope_fingerprint: format!("{canonical:#?}"), source_fact_slots }
    }

    pub(crate) fn matches(&self, plan: &QueryPlan) -> bool {
        let canonical = canonical_execution_scope(plan);
        let mut nodes = canonical.nodes().to_vec();
        for slots in &self.source_fact_slots {
            let Some(node) = nodes.get_mut(slots.node_index) else {
                return false;
            };
            let PlanNode::Source(source) = &mut node.kind else {
                return false;
            };
            let SourceNode::GpuResident(source) = source.as_mut() else {
                return false;
            };
            if slots.row_count_upper_bound {
                source.row_count_upper_bound = None;
            }
            if slots.graph_vertex_upper_bound {
                source.graph_vertex_upper_bound = None;
            }
        }
        let normalized = QueryPlan::from_rewritten_parts(nodes, canonical.root_id(), canonical.sink().clone())
            .with_rewrite_state_from(&canonical);
        self.execution_scope_fingerprint == format!("{normalized:#?}")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{plan::QueryPlanBuilder, source::GpuResidentSource};

    fn resident_plan(name: &str, row_bound: Option<u64>, vertex_bound: Option<u64>) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
        let mut source = GpuResidentSource::new(name, schema);
        if let Some(row_bound) = row_bound {
            source = source.with_row_count_upper_bound(row_bound);
        }
        if let Some(vertex_bound) = vertex_bound {
            source = source.with_graph_vertex_upper_bound(vertex_bound);
        }
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::GpuResident(source));
        builder.finish_gpu_resident(root)
    }

    #[test]
    fn deferred_graph_identity_only_fills_facts_missing_from_the_admitted_plan() {
        let admitted = resident_plan("edges", None, Some(9));
        let identity = DeferredGraphExecutionScopeIdentity::for_plan(&admitted);

        assert!(identity.matches(&resident_plan("edges", Some(12), Some(9))));
        assert!(!identity.matches(&resident_plan("edges", Some(12), Some(10))));
        assert!(!identity.matches(&resident_plan("other_edges", Some(12), Some(9))));
    }

    #[test]
    fn deferred_graph_identity_does_not_weaken_known_source_facts() {
        let admitted = resident_plan("edges", Some(12), Some(9));
        let identity = DeferredGraphExecutionScopeIdentity::for_plan(&admitted);

        assert!(identity.matches(&admitted));
        assert!(!identity.matches(&resident_plan("edges", Some(13), Some(9))));
    }
}
