use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[cfg(feature = "test-hooks")]
use crate::{error::Result, observability::RuntimeExecutionObservationScope, runtime::ExecutionContext};
#[cfg(any(test, feature = "test-hooks"))]
use crate::{
    exec::output::PlanMetrics,
    metrics::{QueryId, QueryMetrics},
};
use crate::{
    observability::{MetricsRecorder, PlanExecutionId, RuntimeObservationHandle},
    plan::NodeId,
};

/// Where one plan execution reports metrics, plus the evidence the engine
/// reuses across that execution's occurrences.
///
/// Metrics go to the plan execution's observation owner, which may retain only
/// part of the detail. Evidence that steers execution is kept here, typed, so
/// no decision depends on what the owner retained.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    recorder: MetricsRecorder,
    evidence: Arc<Mutex<ExecutionEvidence>>,
}

#[derive(Debug, Default)]
struct ExecutionEvidence {
    partial_aggregates: PartialAggregateEvidence,
    // Null edge rows dropped per filtered node, summed over its occurrences.
    null_edge_rows_dropped: HashMap<String, u64>,
}

impl ExecutionMetrics {
    /// Binds the metrics of one plan execution; see [`MetricsRecorder::bind`].
    pub(crate) fn bind(
        observation: &RuntimeObservationHandle,
        plan_execution_id: PlanExecutionId,
        query_id: &str,
        trace_id: Option<&str>,
    ) -> Self {
        Self::new(MetricsRecorder::bind(observation, plan_execution_id, query_id, trace_id))
    }

    fn new(recorder: MetricsRecorder) -> Self {
        Self { recorder, evidence: Arc::default() }
    }

    /// Where this plan execution's metrics are reduced.
    pub(crate) const fn recorder(&self) -> &MetricsRecorder {
        &self.recorder
    }

    /// A plan execution with a private metrics owner, for callers that drive
    /// engine stages without a query.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn detached(label: &str) -> Self {
        Self::bind(&RuntimeObservationHandle::disabled(), PlanExecutionId::new(label), label, None)
    }

    /// `context` scoped to this plan execution, so work that reports through
    /// the context reaches the same owner. For callers that drive engine
    /// stages without [`crate::exec::execute_query_plan`].
    #[cfg(feature = "test-hooks")]
    pub fn scope_context(&self, context: &ExecutionContext) -> Result<ExecutionContext> {
        context.with_runtime_execution_scope(RuntimeExecutionObservationScope::for_metrics(
            context.runtime_observation().clone(),
            self.recorder.clone(),
        ))
    }

    /// The owner's current metrics as a detached value.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn snapshot(&self) -> QueryMetrics {
        PlanMetrics::new(self.recorder.clone(), QueryId::default(), None).snapshot()
    }

    /// Records one aggregate strategy decision and keeps the cardinality
    /// evidence a completed partial aggregate carries.
    pub(crate) fn record_aggregate_strategy_decision(
        &self,
        node_id: impl ToString,
        facts: &BTreeMap<&'static str, String>,
    ) {
        let node_id = node_id.to_string();
        if let Some((structural_key, evidence)) = PartialAggregateCardinality::from_facts(facts) {
            self.evidence().partial_aggregates.insert(&node_id, structural_key, evidence);
        }
        self.recorder.record_aggregate_strategy_decision(node_id, facts);
    }

    /// The latest completed partial aggregate evidence for a structural key:
    /// the same node's first, then any node's.
    pub(in crate::exec) fn prior_partial_aggregate_cardinality(
        &self,
        node_id: Option<&str>,
        structural_key: &str,
    ) -> Option<(PartialAggregateCardinality, PriorEvidenceScope)> {
        self.evidence().partial_aggregates.get(node_id, structural_key)
    }

    /// Records edge rows a node's null filter dropped.
    pub(in crate::exec) fn record_null_edge_rows_dropped(&self, node_id: NodeId, rows: u64) {
        let mut evidence = self.evidence();
        let total = evidence.null_edge_rows_dropped.entry(node_id.to_string()).or_default();
        *total = total.saturating_add(rows);
    }

    /// Edge rows the null filters of `node_id` dropped so far.
    #[cfg(feature = "cugraph")]
    pub(in crate::exec) fn null_edge_rows_dropped(&self, node_id: NodeId) -> u64 {
        self.evidence().null_edge_rows_dropped.get(&node_id.to_string()).copied().unwrap_or(0)
    }

    fn evidence(&self) -> MutexGuard<'_, ExecutionEvidence> {
        self.evidence.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Deref for ExecutionMetrics {
    type Target = MetricsRecorder;

    fn deref(&self) -> &MetricsRecorder {
        &self.recorder
    }
}

/// Where prior evidence came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum PriorEvidenceScope {
    ExactNode,
    StructuralReused,
}

/// Cardinality a completed partial aggregate observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::exec) struct PartialAggregateCardinality {
    pub(in crate::exec) input_rows: u64,
    pub(in crate::exec) output_group_count: u64,
    pub(in crate::exec) reduction_ratio_ppm: u64,
    pub(in crate::exec) selected_strategy: String,
}

impl PartialAggregateCardinality {
    fn from_facts<'a>(facts: &'a BTreeMap<&'static str, String>) -> Option<(&'a str, Self)> {
        if facts.get("execution_stage").map(String::as_str) != Some("partial") {
            return None;
        }
        let structural_key = facts.get("aggregate_structural_key")?;
        let input_rows = facts.get("input_rows")?.parse::<u64>().ok()?;
        if input_rows == 0 {
            return None;
        }
        Some((
            structural_key,
            Self {
                input_rows,
                output_group_count: facts.get("output_group_count")?.parse().ok()?,
                reduction_ratio_ppm: facts.get("reduction_ratio_ppm")?.parse().ok()?,
                selected_strategy: facts.get("selected_strategy")?.clone(),
            },
        ))
    }
}

/// Latest valid evidence per (node, structural key) and per structural key;
/// a later insert replaces the earlier entry for the same key.
#[derive(Debug, Default)]
struct PartialAggregateEvidence {
    by_node: HashMap<(String, String), PartialAggregateCardinality>,
    by_structure: HashMap<String, PartialAggregateCardinality>,
}

impl PartialAggregateEvidence {
    fn insert(&mut self, node_id: &str, structural_key: &str, evidence: PartialAggregateCardinality) {
        self.by_node.insert((node_id.to_string(), structural_key.to_string()), evidence.clone());
        self.by_structure.insert(structural_key.to_string(), evidence);
    }

    fn get(
        &self,
        node_id: Option<&str>,
        structural_key: &str,
    ) -> Option<(PartialAggregateCardinality, PriorEvidenceScope)> {
        if let Some(node_id) = node_id
            && let Some(evidence) = self.by_node.get(&(node_id.to_string(), structural_key.to_string()))
        {
            return Some((evidence.clone(), PriorEvidenceScope::ExactNode));
        }
        self.by_structure.get(structural_key).map(|evidence| (evidence.clone(), PriorEvidenceScope::StructuralReused))
    }
}
