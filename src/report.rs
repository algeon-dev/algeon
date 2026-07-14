//! Planning report shape: per-candidate `Rewritten` / `Rejected` outcomes.
//!
//! Tests and downstream report rows assert on these structured outcomes
//! rather than parsing printed physical plans.

use nexus_query_engine::{
    metrics::{PlanNodeId, QueryId, TraceId},
    triage::{TriageEvidenceRef, TriageSeverity, TriageSignal, TriageSummary, TriageVerdict},
};
use serde_json::{Value, json};

use crate::policy::FallbackReason;

/// Physical-plan candidate considered for native GPU execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCandidate {
    /// Stable shape label used by planning and fallback reports.
    pub shape: String,
    /// Native plan node that anchors this candidate, when lowering reached native IR.
    pub plan_node_id: Option<PlanNodeId>,
    /// Number of fields in the candidate output schema.
    pub output_schema_fields: usize,
    /// Number of physical operators covered by the candidate.
    pub operator_count: usize,
}

impl GpuCandidate {
    /// Creates a single-operator candidate summary.
    pub fn new(shape: impl Into<String>, output_schema_fields: usize) -> Self {
        Self { shape: shape.into(), plan_node_id: None, output_schema_fields, operator_count: 1 }
    }

    /// Creates a candidate summary with an explicit covered-operator count.
    pub fn with_operator_count(shape: impl Into<String>, output_schema_fields: usize, operator_count: usize) -> Self {
        Self { shape: shape.into(), plan_node_id: None, output_schema_fields, operator_count }
    }

    /// Returns this candidate anchored to a native plan node.
    #[must_use]
    pub fn with_plan_node_id(mut self, plan_node_id: impl Into<PlanNodeId>) -> Self {
        self.plan_node_id = Some(plan_node_id.into());
        self
    }

    pub(crate) fn with_outer_operator(mut self, operator: &str) -> Self {
        self.shape = format!("{operator}({})", self.shape);
        self.operator_count += 1;
        self.plan_node_id = None;
        self
    }
}

/// Planning outcome for one native GPU candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateOutcome {
    /// Candidate was lowered and rewritten to native execution.
    Rewritten { candidate: GpuCandidate },
    /// Candidate stayed on CPU or became an error, depending on fallback policy.
    Rejected {
        candidate: GpuCandidate,
        /// Stable reason code for the rejection.
        reason: FallbackReason,
    },
}

/// Ordered planning outcomes emitted by one native optimizer pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PlanningReport {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    outcomes: Vec<CandidateOutcome>,
}

/// Headline rejection caused by table layout metadata rather than unsupported
/// query semantics.
#[derive(Debug, Clone, Copy)]
pub struct DataLayoutFixableRejection<'a> {
    pub outcome_index: usize,
    pub candidate: &'a GpuCandidate,
    pub reason: FallbackReason,
    pub property: &'static str,
}

impl PlanningReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_outcomes(outcomes: impl IntoIterator<Item = CandidateOutcome>) -> Self {
        Self { outcomes: outcomes.into_iter().collect(), ..Self::default() }
    }

    /// Returns the query-wide diagnostic correlation id, when assigned.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Assigns a query id to this planning report.
    pub fn set_query_id(&mut self, query_id: impl Into<QueryId>) {
        self.query_id = Some(query_id.into());
    }

    /// Returns this planning report with a query id assigned.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.set_query_id(query_id);
        self
    }

    /// Assigns a parent trace id to this planning report.
    pub fn set_trace_id(&mut self, trace_id: impl Into<TraceId>) {
        self.trace_id = Some(trace_id.into());
    }

    /// Returns this planning report with a parent trace id assigned.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        self.set_trace_id(trace_id);
        self
    }

    /// Copies diagnostic correlation ids from another planning report.
    #[must_use]
    pub fn with_correlation_from(mut self, other: &PlanningReport) -> Self {
        self.query_id.clone_from(&other.query_id);
        self.trace_id.clone_from(&other.trace_id);
        self
    }

    /// Anchors rewritten candidates to the native root node they produced.
    pub(crate) fn anchor_rewritten_candidates_to_plan_node(&mut self, plan_node_id: impl Into<PlanNodeId>) {
        let plan_node_id = plan_node_id.into();
        for outcome in &mut self.outcomes {
            if let CandidateOutcome::Rewritten { candidate } = outcome {
                candidate.plan_node_id = Some(plan_node_id.clone());
            }
        }
    }

    pub(crate) fn wrap_rewritten_candidates_with_outer_operator(&mut self, operator: &str) {
        for outcome in &mut self.outcomes {
            if let CandidateOutcome::Rewritten { candidate } = outcome {
                *candidate = candidate.clone().with_outer_operator(operator);
            }
        }
    }

    pub fn push_rewritten(&mut self, candidate: GpuCandidate) {
        self.outcomes.push(CandidateOutcome::Rewritten { candidate });
    }

    pub fn push_rejected(&mut self, candidate: GpuCandidate, reason: FallbackReason) {
        self.outcomes.push(CandidateOutcome::Rejected { candidate, reason });
    }

    /// Returns outcomes in discovery order.
    pub fn outcomes(&self) -> &[CandidateOutcome] {
        &self.outcomes
    }

    pub fn rewritten_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::Rewritten { .. })).count()
    }

    pub fn single_rewritten_candidate(&self) -> Option<&GpuCandidate> {
        let mut rewritten = self.outcomes.iter().filter_map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => Some(candidate),
            CandidateOutcome::Rejected { .. } => None,
        });
        let candidate = rewritten.next()?;
        rewritten.next().is_none().then_some(candidate)
    }

    pub fn rejected_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::Rejected { .. })).count()
    }

    /// Returns the first data-layout-fixable rejection in discovery order.
    pub fn data_layout_fixable_rejection(&self) -> Option<DataLayoutFixableRejection<'_>> {
        self.outcomes.iter().enumerate().find_map(|(outcome_index, outcome)| {
            let CandidateOutcome::Rejected { candidate, reason } = outcome else {
                return None;
            };
            reason.data_layout_fixable_property().map(|property| DataLayoutFixableRejection {
                outcome_index,
                candidate,
                reason: *reason,
                property,
            })
        })
    }

    /// Serializes this planning report as structured candidate-outcome JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": "datafusion_nexus.planning_report.v1",
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "trace_id": self.trace_id.as_ref().map(TraceId::as_str),
            "triage": self.triage_summary().to_json(),
            "headline_facts": self.headline_facts_to_json(),
            "rewritten_count": self.rewritten_count(),
            "rejected_count": self.rejected_count(),
            "outcomes": self.outcomes.iter().map(CandidateOutcome::to_json).collect::<Vec<_>>(),
        })
    }

    fn headline_facts_to_json(&self) -> Value {
        match self.data_layout_fixable_rejection() {
            Some(rejection) => json!({
                "data_layout_fixable_rejection": true,
                "data_layout_fixable_reason_code": rejection.reason.code(),
                "data_layout_fixable_property": rejection.property,
                "data_layout_fixable_candidate_shape": rejection.candidate.shape.as_str(),
                "data_layout_fixable_outcome_index": rejection.outcome_index,
            }),
            None => json!({
                "data_layout_fixable_rejection": false,
            }),
        }
    }

    /// Builds a compact triage summary for native planning outcomes.
    pub fn triage_summary(&self) -> TriageSummary {
        let signals = self
            .outcomes
            .iter()
            .enumerate()
            .filter_map(|(index, outcome)| match outcome {
                CandidateOutcome::Rewritten { .. } => None,
                CandidateOutcome::Rejected { candidate, reason } => {
                    let mut evidence = TriageEvidenceRef::new("planning_report.outcomes").with_row_index(index);
                    if let Some(query_id) = self.query_id.clone() {
                        evidence = evidence.with_query_id(query_id);
                    }
                    if let Some(plan_node_id) = candidate.plan_node_id.clone() {
                        evidence = evidence.with_plan_node_id(plan_node_id);
                    }
                    Some(
                        TriageSignal::new(
                            TriageSeverity::Warning,
                            reason.triage_code(),
                            format!("native candidate {} was rejected: {}", candidate.shape, reason.code()),
                        )
                        .with_evidence(evidence),
                    )
                },
            })
            .collect::<Vec<_>>();
        let verdict = if signals.is_empty() {
            TriageVerdict::Healthy
        } else if self.rewritten_count() == 0 {
            TriageVerdict::FellBack
        } else {
            TriageVerdict::Degraded
        };
        TriageSummary::new(verdict, signals)
    }
}

impl CandidateOutcome {
    /// Serializes this candidate outcome as named JSON.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Rewritten { candidate } => json!({
                "outcome": "rewritten",
                "candidate": candidate.to_json(),
                "reason_code": null,
            }),
            Self::Rejected { candidate, reason } => json!({
                "outcome": "rejected",
                "candidate": candidate.to_json(),
                "reason_code": reason.code(),
            }),
        }
    }
}

impl GpuCandidate {
    /// Serializes this candidate summary as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "shape": self.shape,
            "plan_node_id": self.plan_node_id.as_ref().map(PlanNodeId::as_str),
            "output_schema_fields": self.output_schema_fields,
            "operator_count": self.operator_count,
        })
    }
}
