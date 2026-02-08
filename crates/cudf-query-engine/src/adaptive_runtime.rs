//! Adaptive-only runtime evidence stores.
//!
//! These structures are descriptive inputs for adaptive admission. They do not
//! participate in strict bounded contract validation and are intentionally kept
//! outside the mode-agnostic pipeline graph.

use std::{collections::BTreeMap, sync::RwLock};

use crate::{
    metrics::QueryMetrics,
    plan::{PlanNode, QueryPlan, ResultSink},
    source::SourceNode,
};

/// Structural key used to look up prior adaptive memory observations.
///
/// The key is derived from native IR shape and boundary metadata only. It
/// deliberately avoids SQL text, table names, file paths, literal values, and
/// benchmark query ids so history can inform adaptive evidence without becoming
/// query-specific routing.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdaptiveMemoryHistoryKey(String);

impl AdaptiveMemoryHistoryKey {
    pub fn for_query_plan(plan: &QueryPlan) -> Self {
        let mut parts = vec![
            format!("node_count={}", plan.node_count()),
            format!("root={}", plan.root_id().index()),
            format!("sink={}", sink_signature(plan.sink())),
            format!("output_fields={}", plan.output_schema().fields().len()),
        ];
        parts.extend(plan.nodes().iter().map(|node| {
            format!(
                "node{}:{}:inputs={}:fields={}",
                node.id.index(),
                node_signature(&node.kind),
                node.input_ids().len(),
                node.output_schema().fields().len()
            )
        }));
        Self(parts.join("|"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One query observation that can be folded into adaptive memory history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveMemoryObservation {
    pub resident_dataframe_peak_bytes: Option<u64>,
    pub reservation_granted_bytes: u64,
    pub source_chunks_committed: u64,
}

impl AdaptiveMemoryObservation {
    pub fn from_query_metrics(metrics: &QueryMetrics) -> Self {
        Self {
            resident_dataframe_peak_bytes: metrics.resident_dataframe_peak_bytes,
            reservation_granted_bytes: metrics.reservation_granted_bytes,
            source_chunks_committed: metrics.source_chunks_committed,
        }
    }
}

/// Aggregated adaptive memory evidence for one structural query key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveMemoryHistorySummary {
    pub observation_count: u64,
    pub max_resident_dataframe_peak_bytes: Option<u64>,
    pub max_reservation_granted_bytes: u64,
    pub max_source_chunks_committed: u64,
}

impl AdaptiveMemoryHistorySummary {
    fn record(&mut self, observation: &AdaptiveMemoryObservation) {
        self.observation_count = self.observation_count.saturating_add(1);
        self.max_resident_dataframe_peak_bytes =
            match (self.max_resident_dataframe_peak_bytes, observation.resident_dataframe_peak_bytes) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            };
        self.max_reservation_granted_bytes =
            self.max_reservation_granted_bytes.max(observation.reservation_granted_bytes);
        self.max_source_chunks_committed = self.max_source_chunks_committed.max(observation.source_chunks_committed);
    }
}

/// In-memory adaptive history store.
#[derive(Debug, Default)]
pub struct AdaptiveMemoryHistory {
    summaries: RwLock<BTreeMap<AdaptiveMemoryHistoryKey, AdaptiveMemoryHistorySummary>>,
}

impl AdaptiveMemoryHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_observation(&self, key: AdaptiveMemoryHistoryKey, observation: AdaptiveMemoryObservation) {
        let mut summaries = self.summaries.write().expect("adaptive memory history lock poisoned");
        summaries.entry(key).or_default().record(&observation);
    }

    pub fn lookup(&self, key: &AdaptiveMemoryHistoryKey) -> Option<AdaptiveMemoryHistorySummary> {
        self.summaries.read().expect("adaptive memory history lock poisoned").get(key).cloned()
    }

    pub fn lookup_query_plan(&self, plan: &QueryPlan) -> Option<AdaptiveMemoryHistorySummary> {
        self.lookup(&AdaptiveMemoryHistoryKey::for_query_plan(plan))
    }
}

fn sink_signature(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
    }
}

fn node_signature(node: &PlanNode) -> String {
    match node {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => format!(
                "source:local_parquet:format={}:read_columns={}:projected_columns={}:zero_column_count={}",
                match source.diagnostics.format {
                    crate::source::SourceFormat::Parquet => "parquet",
                },
                source.diagnostics.read_column_count,
                source.diagnostics.projected_output_column_count,
                source.diagnostics.zero_column_count_disposition.as_str()
            ),
            SourceNode::GpuResident(_) => "source:gpu_resident".to_string(),
        },
        PlanNode::Filter(_) => "filter".to_string(),
        PlanNode::Projection(node) => format!("projection:outputs={}", node.outputs.len()),
        PlanNode::Aggregate(node) => {
            format!("aggregate:groups={}:aggregates={}", node.spec.group_exprs.len(), node.spec.aggr_exprs.len())
        },
        PlanNode::Sort(node) => format!("sort:exprs={}", node.spec.expressions.len()),
        PlanNode::Limit(node) => format!("limit:skip={}:fetch={}", node.spec.skip > 0, node.spec.fetch > 0),
        PlanNode::Join(node) => format!(
            "join:type={}:keys={}:residual={}:null_equality={}:build_side={}:outputs={}",
            node.spec.join_type,
            node.spec.keys.len(),
            node.spec.residual_predicate.is_some(),
            node.spec.null_equality,
            node.spec.build_side,
            node.spec.output_projection.len()
        ),
    }
}
