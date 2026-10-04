use std::collections::BTreeMap;

use query_planning::correlation::QueryId;

use super::QueryMetrics;

/// Aggregate-strategy decision facts captured for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AggregateStrategyDecisionMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub(crate) query_id: Option<QueryId>,
    /// Native aggregate node id, or a stage-local fallback label when unavailable.
    pub node_id: String,
    /// Stable decision fact keys and values.
    pub facts: BTreeMap<String, String>,
}

impl QueryMetrics {
    /// Records aggregate-strategy diagnostics for a node or execution stage.
    pub fn record_aggregate_strategy_decision(
        &mut self,
        node_id: impl ToString,
        facts: &BTreeMap<&'static str, String>,
    ) {
        self.aggregate_strategy_decisions.push(AggregateStrategyDecisionMetrics {
            query_id: self.inherited_query_id(),
            node_id: node_id.to_string(),
            facts: facts.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect(),
        });
    }

    /// Records one partial aggregate task summary.
    pub fn record_partial_aggregate(&mut self, input_rows: u64, output_rows: u64, retained_bytes: u64) {
        self.partial_aggregate_count = self.partial_aggregate_count.saturating_add(1);
        self.partial_aggregate_input_rows = self.partial_aggregate_input_rows.saturating_add(input_rows);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(output_rows);
        self.partial_aggregate_retained_bytes = self.partial_aggregate_retained_bytes.max(retained_bytes);
    }

    /// Records one final merge attempt and fan-in.
    pub fn record_final_merge_attempt(&mut self, merge_fan_in: usize) {
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(1);
        self.final_merge_max_fan_in = self.final_merge_max_fan_in.max(merge_fan_in as u64);
    }

    /// Records one final merge level.
    pub fn record_final_merge_level(&mut self) {
        self.final_merge_levels = self.final_merge_levels.saturating_add(1);
    }
}
