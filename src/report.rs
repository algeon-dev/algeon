use crate::policy::FallbackReason;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCandidate {
    pub shape: &'static str,
    pub output_schema_fields: usize,
    pub operator_count: usize,
}

impl GpuCandidate {
    pub const fn new(shape: &'static str, output_schema_fields: usize) -> Self {
        Self { shape, output_schema_fields, operator_count: 1 }
    }

    pub const fn with_operator_count(shape: &'static str, output_schema_fields: usize, operator_count: usize) -> Self {
        Self { shape, output_schema_fields, operator_count }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateOutcome {
    Rewritten { candidate: GpuCandidate },
    Rejected { candidate: GpuCandidate, reason: FallbackReason },
}

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

    pub fn outcomes(&self) -> &[CandidateOutcome] {
        &self.outcomes
    }

    pub fn rewritten_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::Rewritten { .. })).count()
    }

    pub fn rejected_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::Rejected { .. })).count()
    }
}
