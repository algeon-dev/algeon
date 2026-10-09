//! Planning report shape: capability, cost-selection, and final-plan evidence.
//!
//! Each report has one reader. [`CandidateOutcome`] gives the optimizer one
//! row per candidate, the final-plan coverage in [`crate::gpu_coverage`] gives
//! the developer the completed plan, and the terminal attempt report gives the
//! caller how execution ended. Tests and downstream report rows assert on these
//! structured outcomes; they do not parse printed physical plans.
//!
//! [`CandidateOutcome`] is per-candidate evidence: one report holds one row
//! per optimizer candidate, and several decision kinds can coexist. Candidate
//! rows never describe final-plan composition; see
//! [`crate::gpu_coverage::QueryGpuPlanningOutcome`] for the completed plan.
//! [`CandidateOutcomeKind`] owns the stable `selected`, `not_supported`, and
//! `not_selected_by_cost` projection labels shared by
//! [`crate::gpu_coverage`] and [`crate::native_report`].

pub use query_planning::{
    capability::Rejection,
    correlation::{PlanNodeId, QueryId, TraceId},
    plan::NativePlanCostReason,
    triage::{TriageCode, TriageSummary, TriageVerdict},
};
use query_planning::{
    plan::NativePlanCostFacts,
    triage::{TriageEvidenceRef, TriageSeverity, TriageSignal},
};
use serde_json::{Value, json};

mod blocking;

pub use blocking::Blocking;
pub(crate) use blocking::{AggregateDefect, PlanBlocker, SinglePartitionOutcome, WindowDefect};

use crate::{capability::CapabilityReason, native::ExpressionRejection};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnsupportedColumnDetail {
    pub(crate) name: String,
    pub(crate) data_type: String,
    pub(crate) role: &'static str,
}

impl UnsupportedColumnDetail {
    pub(crate) fn new(name: impl Into<String>, data_type: impl Into<String>, role: &'static str) -> Self {
        Self { name: name.into(), data_type: data_type.into(), role }
    }
}

/// Evidence a `NotSupported` outcome carries about its cause.
///
/// The public [`CapabilityReason`] stays payload-free; this detail names the
/// unsupported column, the expression-contract rejection, the plan element
/// (operator, aggregate or window function, or declared ordering), or the
/// admission rejection (with the validator's code, operation, and facts) that
/// caused it. [`Self::blocking`] is the one description of it. The detail
/// also says whether planning for one partition would lower the candidate,
/// which the remedy of a reason that prescribes one partition reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NotSupportedDetail {
    pub(crate) column: Option<UnsupportedColumnDetail>,
    pub(crate) expression: Option<Box<ExpressionRejection>>,
    pub(crate) blocker: Option<Box<PlanBlocker>>,
    pub(crate) admission: Option<Rejection>,
    pub(crate) single_partition: SinglePartitionOutcome,
}

impl NotSupportedDetail {
    /// The admission rejection this candidate missed, when admission decided it.
    pub fn admission_rejection(&self) -> Option<&Rejection> {
        self.admission.as_ref()
    }
}

/// Operator token of a candidate whose root operator has no native
/// counterpart.
pub(crate) const UNSUPPORTED_NATIVE_PLAN_OPERATOR: &str = "UnsupportedNativePhysicalPlan";

/// Physical operators a candidate covers, as a tree of operator tokens.
///
/// A token names one physical operator from a stable diagnostic vocabulary;
/// variant labels are part of the token, as in `SortExec(fetch)`. The display
/// grammar is a leaf `Op`, a unary `Op(child)`, or `Op(left,right)`. The
/// broader families are named by the
/// [`NATIVE_CANDIDATE_SHAPE_FAMILY_*`](crate::native_report::NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN)
/// constants and computed from this structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateShape {
    operator: String,
    inputs: Vec<CandidateShape>,
    relation_input: bool,
}

impl CandidateShape {
    /// One operator without covered inputs.
    pub fn leaf(operator: impl Into<String>) -> Self {
        Self { operator: operator.into(), inputs: Vec::new(), relation_input: false }
    }

    /// One operator over the shapes of its covered inputs.
    pub fn new(operator: impl Into<String>, inputs: Vec<CandidateShape>) -> Self {
        Self { operator: operator.into(), inputs, relation_input: false }
    }

    /// `shape` labelled as the input relation `label` of a GPU relation
    /// function, such as `cugraph_pagerank.edges`. The label is not an
    /// operator, so [`Self::without_relation_input_label`] sees through it.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn relation_input(label: &str, shape: CandidateShape) -> Self {
        Self { operator: label.to_string(), inputs: vec![shape], relation_input: true }
    }

    /// The shape under a relation-input label, or this shape without one.
    pub(crate) fn without_relation_input_label(&self) -> &CandidateShape {
        match self.inputs.as_slice() {
            [input] if self.relation_input => input,
            _ => self,
        }
    }

    /// The operator token at the root of the shape.
    pub fn operator(&self) -> &str {
        &self.operator
    }

    /// The shapes of the root operator's covered inputs.
    pub fn inputs(&self) -> &[CandidateShape] {
        &self.inputs
    }

    /// Number of operators the shape covers.
    pub fn operator_count(&self) -> usize {
        usize::from(!self.relation_input) + self.inputs.iter().map(Self::operator_count).sum::<usize>()
    }

    /// Whether any covered operator has this token.
    pub fn contains_operator(&self, operator: &str) -> bool {
        self.operator == operator || self.inputs.iter().any(|input| input.contains_operator(operator))
    }
}

impl std::fmt::Display for CandidateShape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.operator)?;
        if let Some((first, rest)) = self.inputs.split_first() {
            write!(formatter, "({first}")?;
            for input in rest {
                write!(formatter, ",{input}")?;
            }
            formatter.write_str(")")?;
        }
        Ok(())
    }
}

impl From<&str> for CandidateShape {
    fn from(operator: &str) -> Self {
        Self::leaf(operator)
    }
}

impl From<String> for CandidateShape {
    fn from(operator: String) -> Self {
        Self::leaf(operator)
    }
}

/// Physical-plan candidate considered for native GPU execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCandidate {
    /// Physical operators the candidate covers.
    /// [`NativeDataFusionPlanningRow::native_candidate_shape`](crate::native_report::NativeDataFusionPlanningRow::native_candidate_shape)
    /// is the report projection of this value.
    pub shape: CandidateShape,
    /// Native plan node that anchors this candidate, when lowering reached native IR.
    pub plan_node_id: Option<PlanNodeId>,
    /// Number of fields in the candidate output schema.
    pub output_schema_fields: usize,
}

impl GpuCandidate {
    /// Creates a candidate summary for the operators `shape` covers.
    pub fn new(shape: impl Into<CandidateShape>, output_schema_fields: usize) -> Self {
        Self { shape: shape.into(), plan_node_id: None, output_schema_fields }
    }

    /// Number of physical operators covered by the candidate.
    pub fn operator_count(&self) -> usize {
        self.shape.operator_count()
    }

    /// Returns this candidate anchored to a native plan node.
    #[must_use]
    pub fn with_plan_node_id(mut self, plan_node_id: impl Into<PlanNodeId>) -> Self {
        self.plan_node_id = Some(plan_node_id.into());
        self
    }

    /// This candidate labelled as an input relation of a GPU relation
    /// function. The candidate keeps its native plan node.
    #[must_use]
    #[cfg(feature = "cugraph")]
    pub(crate) fn with_relation_input_label(mut self, label: &str) -> Self {
        self.shape = CandidateShape::relation_input(label, self.shape);
        self
    }

    pub(crate) fn with_outer_operator(mut self, operator: &str) -> Self {
        self.shape = CandidateShape::new(operator, vec![self.shape]);
        self.plan_node_id = None;
        self
    }
}

/// Canonical classification for a native GPU candidate decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateOutcomeKind {
    /// Candidate was selected for native execution.
    Selected,
    /// Candidate cannot execute under the native capability contract.
    NotSupported,
    /// Candidate is native-capable but was not selected by the cost model.
    NotSelectedByCost,
}

impl CandidateOutcomeKind {
    /// Stable TSV, JSON, and Arrow label for this candidate outcome kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::NotSupported => "not_supported",
            Self::NotSelectedByCost => "not_selected_by_cost",
        }
    }
}

impl std::fmt::Display for CandidateOutcomeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Planning outcome for one native GPU candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateOutcome {
    /// Candidate was selected for native execution.
    Selected {
        /// Candidate selected for native execution.
        candidate: GpuCandidate,
    },
    /// Candidate cannot execute under the native capability contract.
    NotSupported {
        /// Candidate whose selection decision this outcome records.
        candidate: GpuCandidate,
        /// Stable capability reason code.
        capability_reason: CapabilityReason,
        /// Evidence naming the cause.
        detail: NotSupportedDetail,
    },
    /// Candidate is native-capable but was not selected by the cost model.
    NotSelectedByCost {
        /// Candidate whose selection decision this outcome records.
        candidate: GpuCandidate,
        /// Stable cost-selection reason code.
        cost_reason: NativePlanCostReason,
        /// Facts evaluated by the cost model.
        evidence: NativeCandidateCostEvidence,
    },
}

impl CandidateOutcome {
    /// Returns the canonical outcome kind for this candidate decision.
    pub const fn kind(&self) -> CandidateOutcomeKind {
        match self {
            Self::Selected { .. } => CandidateOutcomeKind::Selected,
            Self::NotSupported { .. } => CandidateOutcomeKind::NotSupported,
            Self::NotSelectedByCost { .. } => CandidateOutcomeKind::NotSelectedByCost,
        }
    }
}

/// Stable root-owned cost evidence for a native candidate selection decision.
///
/// The native engine owns the cost decision. This DTO preserves the
/// decision facts at the adapter's public report boundary without making the
/// report wire format depend on an engine-internal struct layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCandidateCostEvidence {
    /// Number of nodes in the native candidate IR.
    pub node_count: usize,
    /// Number of source nodes in the candidate.
    pub source_count: usize,
    /// Number of candidate sources backed by local Parquet.
    pub local_parquet_source_count: usize,
    /// Whether all local-Parquet sources provide row-group selection metadata.
    pub selected_row_groups_known: bool,
    /// Total selected Parquet row groups, or `None` when selection metadata is unavailable.
    pub selected_row_groups: Option<usize>,
    /// Estimated source rows, or `None` when any local-Parquet source row count is unavailable.
    pub estimated_source_rows: Option<u64>,
    /// Estimated projected source bytes, or `None` when any local-Parquet source byte estimate is unavailable.
    pub estimated_source_bytes: Option<u64>,
    /// Whether all local-Parquet sources provide selected row-group byte evidence.
    pub selected_source_bytes_known: bool,
    /// Number of joins in the candidate.
    pub join_count: usize,
    /// Number of joins with a residual predicate beyond their equality keys.
    pub residual_join_count: usize,
    /// Number of aggregate nodes in the candidate.
    pub aggregate_count: usize,
    /// Number of aggregate nodes with grouping keys.
    pub grouped_aggregate_count: usize,
    /// Number of sort nodes in the candidate.
    pub sort_count: usize,
    /// Number of limit nodes in the candidate.
    pub limit_count: usize,
    /// Smallest bounded sort fetch in rows, or `None` when no top-k bound exists.
    pub min_topk_fetch: Option<usize>,
}

impl NativeCandidateCostEvidence {
    pub(crate) fn from_cost_facts(facts: NativePlanCostFacts) -> Self {
        Self {
            node_count: facts.node_count,
            source_count: facts.source_count,
            local_parquet_source_count: facts.local_parquet_source_count,
            selected_row_groups_known: facts.selected_row_groups_known,
            selected_row_groups: facts.selected_row_groups,
            estimated_source_rows: facts.estimated_source_rows,
            estimated_source_bytes: facts.estimated_source_bytes,
            selected_source_bytes_known: facts.selected_source_bytes_known,
            join_count: facts.join_count,
            residual_join_count: facts.residual_join_count,
            aggregate_count: facts.aggregate_count,
            grouped_aggregate_count: facts.grouped_aggregate_count,
            sort_count: facts.sort_count,
            limit_count: facts.limit_count,
            min_topk_fetch: facts.min_topk_fetch,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "node_count": self.node_count,
            "source_count": self.source_count,
            "local_parquet_source_count": self.local_parquet_source_count,
            "selected_row_groups_known": self.selected_row_groups_known,
            "selected_row_groups": self.selected_row_groups,
            "estimated_source_rows": self.estimated_source_rows,
            "estimated_source_bytes": self.estimated_source_bytes,
            "selected_source_bytes_known": self.selected_source_bytes_known,
            "join_count": self.join_count,
            "residual_join_count": self.residual_join_count,
            "aggregate_count": self.aggregate_count,
            "grouped_aggregate_count": self.grouped_aggregate_count,
            "sort_count": self.sort_count,
            "limit_count": self.limit_count,
            "min_topk_fetch": self.min_topk_fetch,
        })
    }
}

/// Ordered planning outcomes emitted by one native optimizer pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PlanningReport {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    outcomes: Vec<CandidateOutcome>,
}

/// Headline capability miss caused by table layout metadata. Unsupported query
/// semantics are not the cause.
#[derive(Debug, Clone, Copy)]
pub struct DataLayoutFixableNotSupported<'a> {
    /// Zero-based position of this miss in the report's discovery-ordered outcomes.
    pub outcome_index: usize,
    /// Borrowed candidate that the native optimizer declined.
    pub candidate: &'a GpuCandidate,
    /// Stable reason identifying the unsupported layout contract.
    pub capability_reason: CapabilityReason,
    /// Layout property a table owner can change to remove this capability miss.
    pub property: &'static str,
}

impl PlanningReport {
    /// Creates an empty report with no correlation identifiers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Collects candidate decisions in iterator order with no correlation identifiers.
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
    pub(crate) fn anchor_selected_candidates_to_plan_node(&mut self, plan_node_id: impl Into<PlanNodeId>) {
        let plan_node_id = plan_node_id.into();
        for outcome in &mut self.outcomes {
            match outcome {
                CandidateOutcome::Selected { candidate } => {
                    candidate.plan_node_id = Some(plan_node_id.clone());
                },
                CandidateOutcome::NotSupported { .. } | CandidateOutcome::NotSelectedByCost { .. } => {},
            }
        }
    }

    pub(crate) fn wrap_selected_candidates_with_outer_operator(&mut self, operator: &str) {
        for outcome in &mut self.outcomes {
            match outcome {
                CandidateOutcome::Selected { candidate } => {
                    *candidate = candidate.clone().with_outer_operator(operator);
                },
                CandidateOutcome::NotSupported { .. } | CandidateOutcome::NotSelectedByCost { .. } => {},
            }
        }
    }

    /// Labels every candidate as an input relation of a GPU relation function.
    #[cfg(feature = "cugraph")]
    pub(crate) fn label_relation_input(&mut self, label: &str) {
        for outcome in &mut self.outcomes {
            let (CandidateOutcome::Selected { candidate }
            | CandidateOutcome::NotSupported { candidate, .. }
            | CandidateOutcome::NotSelectedByCost { candidate, .. }) = outcome;
            *candidate = candidate.clone().with_relation_input_label(label);
        }
    }

    /// Appends another report's outcomes after this report's outcomes.
    pub(crate) fn append(&mut self, other: PlanningReport) {
        self.outcomes.extend(other.outcomes);
    }

    /// Appends a candidate selected for native replacement.
    pub fn push_selected(&mut self, candidate: GpuCandidate) {
        self.outcomes.push(CandidateOutcome::Selected { candidate });
    }

    /// Appends a capability decline with no additional blocking evidence.
    ///
    /// A decline preserves the candidate's DataFusion execution path; it is not an execution error.
    pub fn push_not_supported(&mut self, candidate: GpuCandidate, capability_reason: CapabilityReason) {
        self.push_not_supported_with_detail(candidate, capability_reason, NotSupportedDetail::default());
    }

    pub(crate) fn push_not_supported_with_detail(
        &mut self,
        candidate: GpuCandidate,
        capability_reason: CapabilityReason,
        detail: NotSupportedDetail,
    ) {
        self.outcomes.push(CandidateOutcome::NotSupported { candidate, capability_reason, detail });
    }

    /// Appends a native-capable candidate retained on DataFusion with its cost reason and evidence.
    pub fn push_not_selected_by_cost(
        &mut self,
        candidate: GpuCandidate,
        cost_reason: NativePlanCostReason,
        evidence: NativeCandidateCostEvidence,
    ) {
        self.outcomes.push(CandidateOutcome::NotSelectedByCost { candidate, cost_reason, evidence });
    }

    /// Returns outcomes in discovery order.
    pub fn outcomes(&self) -> &[CandidateOutcome] {
        &self.outcomes
    }

    /// Counts candidates selected for native replacement.
    pub fn selected_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::Selected { .. })).count()
    }

    /// Returns the selected candidate only when the report contains exactly one.
    ///
    /// Returns `None` for zero or several selected candidates.
    pub fn single_selected_candidate(&self) -> Option<&GpuCandidate> {
        let mut rewritten = self.outcomes.iter().filter_map(|outcome| match outcome {
            CandidateOutcome::Selected { candidate } => Some(candidate),
            CandidateOutcome::NotSupported { .. } | CandidateOutcome::NotSelectedByCost { .. } => None,
        });
        let candidate = rewritten.next()?;
        rewritten.next().is_none().then_some(candidate)
    }

    /// Counts capability declines and cost declines together.
    pub fn not_selected_count(&self) -> usize {
        self.not_supported_count() + self.not_selected_by_cost_count()
    }

    /// Counts candidates whose native capability contract was not satisfied.
    pub fn not_supported_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::NotSupported { .. })).count()
    }

    /// Counts native-capable candidates retained on DataFusion by cost selection.
    pub fn not_selected_by_cost_count(&self) -> usize {
        self.outcomes.iter().filter(|outcome| matches!(outcome, CandidateOutcome::NotSelectedByCost { .. })).count()
    }

    /// Returns the first data-layout-fixable capability miss in discovery order.
    pub fn data_layout_fixable_not_supported(&self) -> Option<DataLayoutFixableNotSupported<'_>> {
        self.outcomes.iter().enumerate().find_map(|(outcome_index, outcome)| {
            let CandidateOutcome::NotSupported { candidate, capability_reason, .. } = outcome else {
                return None;
            };
            capability_reason.facts().layout_property.map(|property| DataLayoutFixableNotSupported {
                outcome_index,
                candidate,
                capability_reason: *capability_reason,
                property,
            })
        })
    }

    /// Serializes this planning report as structured candidate-outcome JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": "algeon_datafusion.planning_report",
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "trace_id": self.trace_id.as_ref().map(TraceId::as_str),
            "triage": self.triage_summary().to_json(),
            "headline_facts": self.headline_facts_to_json(),
            "selected_count": self.selected_count(),
            "not_supported_count": self.not_supported_count(),
            "not_selected_by_cost_count": self.not_selected_by_cost_count(),
            "not_selected_count": self.not_selected_count(),
            "outcomes": self.outcomes.iter().map(CandidateOutcome::to_json).collect::<Vec<_>>(),
        })
    }

    fn headline_facts_to_json(&self) -> Value {
        match self.data_layout_fixable_not_supported() {
            Some(not_supported) => json!({
                "data_layout_fixable_not_supported": true,
                "data_layout_fixable_reason_code": not_supported.capability_reason.code(),
                "data_layout_fixable_property": not_supported.property,
                "data_layout_fixable_candidate_shape": not_supported.candidate.shape.to_string(),
                "data_layout_fixable_outcome_index": not_supported.outcome_index,
            }),
            None => json!({
                "data_layout_fixable_not_supported": false,
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
                CandidateOutcome::Selected { .. } => None,
                CandidateOutcome::NotSupported { candidate, capability_reason, .. } => {
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
                            capability_reason.facts().triage,
                            format!(
                                "native candidate {} is not supported: {}",
                                candidate.shape,
                                capability_reason.code()
                            ),
                        )
                        .with_evidence(evidence),
                    )
                },
                CandidateOutcome::NotSelectedByCost { candidate, cost_reason, .. } => {
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
                            TriageCode::NotSelectedByCost,
                            format!(
                                "native-capable candidate {} was not selected by cost: {}",
                                candidate.shape,
                                cost_reason.code()
                            ),
                        )
                        .with_evidence(evidence),
                    )
                },
            })
            .collect::<Vec<_>>();
        let verdict = if signals.is_empty() {
            TriageVerdict::Healthy
        } else if self.selected_count() == 0 {
            TriageVerdict::NotSelected
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
            Self::Selected { candidate } => json!({
                "outcome": self.kind().as_str(),
                "candidate": candidate.to_json(),
                "capability_reason_code": null,
                "cost_reason_code": null,
                "cost_evidence": null,
            }),
            Self::NotSupported { candidate, capability_reason, .. } => json!({
                "outcome": self.kind().as_str(),
                "candidate": candidate.to_json(),
                "capability_reason_code": capability_reason.code(),
                "cost_reason_code": null,
                "cost_evidence": null,
            }),
            Self::NotSelectedByCost { candidate, cost_reason, evidence } => json!({
                "outcome": self.kind().as_str(),
                "candidate": candidate.to_json(),
                "capability_reason_code": null,
                "cost_reason_code": cost_reason.code(),
                "cost_evidence": evidence.to_json(),
            }),
        }
    }
}

impl GpuCandidate {
    /// Serializes this candidate summary as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "shape": self.shape.to_string(),
            "plan_node_id": self.plan_node_id.as_ref().map(PlanNodeId::as_str),
            "output_schema_fields": self.output_schema_fields,
            "operator_count": self.operator_count(),
        })
    }
}
