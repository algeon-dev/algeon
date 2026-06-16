//! Planning report shape: per-candidate `Rewritten` / `Rejected` outcomes.
//!
//! Tests and downstream report rows assert on these structured outcomes
//! rather than parsing printed physical plans.

use crate::policy::FallbackReason;

/// Physical-plan candidate considered for native GPU execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCandidate {
    /// Stable shape label used by planning and fallback reports.
    pub shape: String,
    /// Number of fields in the candidate output schema.
    pub output_schema_fields: usize,
    /// Number of physical operators covered by the candidate.
    pub operator_count: usize,
}

impl GpuCandidate {
    /// Creates a single-operator candidate summary.
    pub fn new(shape: impl Into<String>, output_schema_fields: usize) -> Self {
        Self { shape: shape.into(), output_schema_fields, operator_count: 1 }
    }

    /// Creates a candidate summary with an explicit covered-operator count.
    pub fn with_operator_count(shape: impl Into<String>, output_schema_fields: usize, operator_count: usize) -> Self {
        Self { shape: shape.into(), output_schema_fields, operator_count }
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
    outcomes: Vec<CandidateOutcome>,
}

impl PlanningReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_outcomes(outcomes: impl IntoIterator<Item = CandidateOutcome>) -> Self {
        Self { outcomes: outcomes.into_iter().collect() }
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
}
