use std::collections::BTreeMap;

use super::{QueryId, QueryMetrics};

/// Join-strategy decision facts captured for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinStrategyDecisionMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Native join node id.
    pub node_id: String,
    /// Stable decision fact keys and values.
    pub facts: BTreeMap<String, String>,
}

/// Join-specific operator metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinOperatorMetrics {
    /// Query-wide diagnostic correlation id, when assigned.
    pub query_id: Option<QueryId>,
    /// Join type label.
    pub join_type: String,
    /// Execution strategy label.
    pub execution_strategy: String,
    /// Number of equality key pairs.
    pub key_count: usize,
    /// Whether a residual predicate was present.
    pub residual_predicate: bool,
    /// Null-equality policy label.
    pub null_equality: String,
    /// Planner orientation used to execute or describe this join.
    ///
    /// For reusable hash joins this is the side that owns the hash table. For
    /// keyless conditional and cross joins no hash table is built, so this is an
    /// orientation label for comparable metrics rather than a physical build
    /// artifact.
    pub build_side: String,
    /// The side opposite `build_side` in the planner orientation.
    pub probe_side: String,
    /// Build-side input row count.
    pub build_rows: u64,
    /// Build-side input bytes.
    pub build_bytes: u64,
    /// Probe-side input row count.
    pub probe_rows: u64,
    /// Probe-side input bytes.
    pub probe_bytes: u64,
    /// Device bytes retained by the reusable cuDF join build artifact.
    ///
    /// This is a direct cudf observation. For hash joins it is the retained
    /// hash table; for filtered joins it is the retained filtered-join build
    /// artifact. It remains `0` for join paths without a reusable build artifact.
    pub hash_table_bytes: u64,
    /// Device bytes retained by a filtered-join probe state.
    pub filtered_probe_state_bytes: u64,
    /// Device bytes retained by a mixed join retained size-data state.
    pub mixed_size_data_bytes: u64,
    /// Device bytes retained by join gather maps produced for this join.
    ///
    /// This is a direct cudf observation over the returned device index
    /// columns, not a row-count estimate.
    pub gather_map_bytes: u64,
    /// Bytes reserved for join output materialization.
    pub output_reservation_bytes: u64,
    /// Join output row count.
    pub output_rows: u64,
    /// Join output bytes.
    pub output_bytes: u64,
}

impl QueryMetrics {
    /// Records join-strategy diagnostics for a node.
    pub fn record_join_strategy_decision(&mut self, node_id: impl ToString, facts: &BTreeMap<&'static str, String>) {
        self.join_strategy_decisions.push(JoinStrategyDecisionMetrics {
            query_id: self.inherited_query_id(),
            node_id: node_id.to_string(),
            facts: facts.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect(),
        });
    }

    /// Records one query-scoped hash-join cache hit.
    pub fn record_hash_join_cache_hit(&mut self) {
        self.hash_join_cache_hits = self.hash_join_cache_hits.saturating_add(1);
    }

    /// Records one query-scoped hash-join cache miss.
    pub fn record_hash_join_cache_miss(&mut self) {
        self.hash_join_cache_misses = self.hash_join_cache_misses.saturating_add(1);
    }

    /// Records one reusable hash-join build.
    pub fn record_hash_join_build(&mut self) {
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(1);
    }

    /// Records one reusable filtered-join build.
    pub fn record_filtered_join_build(&mut self) {
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(1);
    }

    /// Records a one-shot join execution.
    pub fn record_one_shot_join(&mut self) {
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(1);
    }

    /// Adds retained hash-join hash table bytes.
    pub fn record_hash_join_hash_table_bytes(&mut self, bytes: u64) {
        self.hash_join_hash_table_bytes = self.hash_join_hash_table_bytes.saturating_add(bytes);
    }

    /// Adds retained filtered-join hash table bytes.
    pub fn record_filtered_join_hash_table_bytes(&mut self, bytes: u64) {
        self.filtered_join_hash_table_bytes = self.filtered_join_hash_table_bytes.saturating_add(bytes);
    }

    /// Adds retained filtered-join probe-state bytes.
    pub fn record_filtered_join_probe_state_bytes(&mut self, bytes: u64) {
        self.filtered_join_probe_state_bytes = self.filtered_join_probe_state_bytes.saturating_add(bytes);
    }

    /// Adds retained mixed-join size-data bytes.
    pub fn record_mixed_join_size_data_bytes(&mut self, bytes: u64) {
        self.mixed_join_size_data_bytes = self.mixed_join_size_data_bytes.saturating_add(bytes);
    }

    /// Adds one-shot join index-map bytes.
    pub fn record_one_shot_join_index_bytes(&mut self, bytes: u64) {
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(bytes);
    }

    /// Adds one-shot join output materialization reservation bytes.
    pub fn record_one_shot_join_materialization_bytes(&mut self, bytes: u64) {
        self.one_shot_join_materialization_bytes = self.one_shot_join_materialization_bytes.saturating_add(bytes);
    }

    /// Adds one-shot join output bytes.
    pub fn record_one_shot_join_output_bytes(&mut self, bytes: u64) {
        self.one_shot_join_output_bytes = self.one_shot_join_output_bytes.saturating_add(bytes);
    }
}
