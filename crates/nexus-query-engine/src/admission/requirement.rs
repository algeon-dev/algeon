//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

use crate::{
    capability::ReservationRequirement,
    error::{QueryError, QueryErrorCode, Result},
    plan::QueryPlan,
    runtime::ReservationKind,
};

/// Trustworthiness of the descriptive working-set hint.
///
/// Ordering is `Unknown < Modeled < Measured`; aggregation retains the weakest
/// evidence so a multi-fragment invocation never overstates its estimate.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EstimateQuality {
    #[default]
    Unknown,
    Modeled,
    Measured,
}

/// Resolved, planner-owned capability evidence for one bounded execution attempt.
///
/// Callers can forward this value into admission, but cannot author a floor,
/// hint, or device-capability set.  The installed device profile owns baseline
/// grant sizing; planner evidence may only record a proven lower-layer raise
/// and a descriptive model result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptCapabilityEvidence {
    /// Deployment-wide floor supplied by the installed device profile.  This
    /// remains crate-visible only for synthetic controller tests.
    pub(crate) configured_floor_bytes: u64,
    /// Additional lower-layer workspace contracts proven before admission.
    pub(crate) capability_floor_raise_bytes: u64,
    /// Descriptive target only; it never proves fit or admission.
    pub(crate) soft_working_set_hint_bytes: Option<u64>,
    /// Evidence quality for [`Self::soft_working_set_hint_bytes`].
    pub(crate) estimate_quality: EstimateQuality,
    /// Capability contracts required by the admitted fragments.
    pub(crate) required_capabilities: Vec<ReservationRequirement>,
    /// Exact graph plans for which the engine may replace its generic
    /// lower-level graph-workspace blocker with the cuGraph contract.
    ///
    /// This remains part of the engine-owned evidence bundle from admission
    /// through execution. A capability label alone is deliberately not enough
    /// to make that substitution: it would let a combined proof authorize an
    /// unrelated graph plan.
    pub(crate) attested_cugraph_graph_execution_scopes: Vec<AttestedCugraphGraphExecutionScope>,
}

/// Collision-free structural identity for one graph plan whose frontend owns
/// construction and workspace lifetimes.
///
/// The identity retains the full derived structural form instead of a
/// fixed-width hash. These scopes are short-lived admission evidence, not
/// externally stored keys, so a hash collision cannot become an authorization
/// path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttestedCugraphGraphExecutionScope {
    structural_fingerprint: String,
}

impl AttestedCugraphGraphExecutionScope {
    fn for_plan(plan: &QueryPlan) -> Self {
        Self { structural_fingerprint: format!("{plan:#?}") }
    }

    fn matches(&self, plan: &QueryPlan) -> bool {
        self.structural_fingerprint == Self::for_plan(plan).structural_fingerprint
    }
}

/// Opaque capability proof issued from a native plan or a canonical graph
/// execution contract.
///
/// Public callers can forward a proof into admission, but cannot author a
/// capability set, a floor, or a working-set target.  In particular, there is
/// deliberately no empty-proof constructor: the absence of a native fragment
/// remains absence of proof rather than authorization for an exact profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPlanCapabilityProof {
    requirement: AttemptCapabilityEvidence,
}

impl QueryPlanCapabilityProof {
    /// Derives proof from one lowered native query plan.
    ///
    /// `None` means the plan has no modeled capability contracts and must not
    /// be used to manufacture an explicit empty admission proof.
    pub fn for_query_plan(plan: &QueryPlan) -> Option<Self> {
        let requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        (!requirement.required_capabilities.is_empty()).then_some(Self { requirement })
    }

    /// Derives a plan-scoped proof for the exact internal plan issued by the
    /// cuGraph frontend.
    ///
    /// The generic graph node analysis deliberately keeps its lower-level
    /// bounded-execution blocker: the engine alone cannot establish that an
    /// embedding graph runtime owns all graph construction and workspace
    /// lifetimes.  cuGraph establishes that contract at its own boundary, so
    /// its engine-issued proof replaces only that blocker with the canonical
    /// cuGraph execution capability. The replacement is bound to a private,
    /// collision-free structural scope for this exact plan; combining proofs
    /// preserves every scope, and execution rechecks it against the plan
    /// about to run. All other plan requirements, including edge
    /// normalization and result-sink materialization, remain exact.
    pub fn for_cugraph_graph_execution_plan(plan: &QueryPlan) -> Option<Self> {
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        let graph_requirement = generic_graph_execution_blocker();
        if !requirement.required_capabilities.contains(&graph_requirement) {
            return (!requirement.required_capabilities.is_empty()).then_some(Self { requirement });
        }
        requirement.required_capabilities.retain(|candidate| candidate != &graph_requirement);
        requirement.required_capabilities.push(cugraph_graph_execution_capability());
        requirement.attested_cugraph_graph_execution_scopes.push(AttestedCugraphGraphExecutionScope::for_plan(plan));
        Some(Self { requirement })
    }

    /// Combines proofs already issued for native fragments in one invocation.
    ///
    /// Requiring a first proof prevents public callers from using aggregation
    /// as an empty-proof constructor.
    pub fn combine(first: Self, remaining: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut requirements = vec![first.requirement];
        requirements.extend(remaining.into_iter().map(|proof| proof.requirement));
        AttemptCapabilityEvidence::aggregate(requirements).map(|requirement| Self { requirement })
    }

    /// Returns the device-capability contracts captured by the engine-issued
    /// proof. This is diagnostic-only and cannot be used to construct a new
    /// proof.
    pub fn required_capabilities(&self) -> &[ReservationRequirement] {
        &self.requirement.required_capabilities
    }

    /// Resolves the proof's immutable floor against an installed profile.
    /// This is read-only inspection and never selects a grant target.
    pub fn required_floor_bytes_for(&self, configured_floor_bytes: u64) -> Result<u64> {
        self.requirement.required_floor_bytes_for(configured_floor_bytes)
    }

    pub(crate) fn into_requirement(self) -> AttemptCapabilityEvidence {
        self.requirement
    }

    pub(crate) fn as_requirement(&self) -> &AttemptCapabilityEvidence {
        &self.requirement
    }
}

impl AttemptCapabilityEvidence {
    /// Returns evidence derived from the engine's capability analysis for one
    /// lowered native plan.  The plan, rather than an embedding caller,
    /// determines the capability set.
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        Self::from_parts(0, 0, None, EstimateQuality::Unknown, plan.capability_analysis().requirements)
    }

    /// Returns the descriptive working-set target, if a planner produced one.
    pub(crate) fn soft_working_set_hint_bytes(&self) -> Option<u64> {
        self.soft_working_set_hint_bytes
    }

    /// Returns the provenance quality of the descriptive working-set target.
    pub(crate) fn estimate_quality(&self) -> EstimateQuality {
        self.estimate_quality
    }

    /// Returns the capability contracts captured by the planner.
    pub(crate) fn required_capabilities(&self) -> &[ReservationRequirement] {
        &self.required_capabilities
    }

    /// Resolves this evidence against an already-installed device-profile
    /// floor.  This is read-only inspection; it never selects a grant.
    pub(crate) fn required_floor_bytes_for(&self, configured_floor_bytes: u64) -> Result<u64> {
        configured_floor_bytes.checked_add(self.capability_floor_raise_bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.admission.requirement",
                "attempt capability floor overflows u64",
            )
            .with_fact("reason", "attempt_requirement_floor_overflow")
            .with_fact("configured_floor_bytes", configured_floor_bytes)
            .with_fact("capability_floor_raise_bytes", self.capability_floor_raise_bytes)
        })
    }

    /// Combines evidence from all fragments in one invocation.
    ///
    /// The installed device profile supplies the baseline floor exactly once;
    /// only independently live lower-layer raises are summed here.
    pub(crate) fn aggregate<I: IntoIterator<Item = Self>>(fragments: I) -> Result<Self> {
        Self::aggregate_with_configured_floor(0, fragments)
    }

    pub(crate) fn empty() -> Self {
        Self::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new())
    }

    pub(crate) fn from_parts(
        configured_floor_bytes: u64,
        capability_floor_raise_bytes: u64,
        soft_working_set_hint_bytes: Option<u64>,
        estimate_quality: EstimateQuality,
        required_capabilities: Vec<ReservationRequirement>,
    ) -> Self {
        Self {
            configured_floor_bytes,
            capability_floor_raise_bytes,
            soft_working_set_hint_bytes,
            estimate_quality,
            required_capabilities,
            attested_cugraph_graph_execution_scopes: Vec::new(),
        }
    }

    /// Stable, human-readable identities for structured device-capability
    /// rejection facts.
    pub(crate) fn capability_labels(required_capabilities: &[ReservationRequirement]) -> String {
        required_capabilities
            .iter()
            .map(|capability| format!("{}:{}", capability.kind.as_str(), capability.required_contract))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The non-negotiable attempt floor, rejecting integer overflow.
    pub(crate) fn required_floor_bytes(&self) -> Result<u64> {
        self.required_floor_bytes_for(self.configured_floor_bytes)
    }

    /// Verifies that this admitted requirement covers the plan about to run.
    ///
    /// A ticket may aggregate proofs for several native fragments, so the
    /// admitted set may be a strict superset.  Missing contracts are unsafe:
    /// the execution plan must be rejected before it allocates or dispatches
    /// GPU work.
    pub(crate) fn validate_covers_plan(&self, plan: &QueryPlan) -> Result<()> {
        let required_capabilities = plan.capability_analysis().requirements;
        let cugraph_graph_execution_is_proven =
            self.required_capabilities.contains(&cugraph_graph_execution_capability())
                && self.attested_cugraph_graph_execution_scopes.iter().any(|scope| scope.matches(plan));
        let missing_capabilities = required_capabilities
            .iter()
            .filter(|required| {
                !self.required_capabilities.contains(required)
                    && !(cugraph_graph_execution_is_proven && is_generic_graph_execution_blocker(required))
            })
            .cloned()
            .collect::<Vec<_>>();
        if missing_capabilities.is_empty() {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.requirement",
            "admitted capability proof does not cover the native plan about to execute",
        )
        .with_fact("reason", "admitted_capability_evidence_mismatch")
        .with_fact("required_capabilities", Self::capability_labels(&required_capabilities))
        .with_fact("admitted_capabilities", Self::capability_labels(&self.required_capabilities))
        .with_fact("missing_capabilities", Self::capability_labels(&missing_capabilities)))
    }

    /// Aggregates all native fragments in one invocation.
    ///
    /// Capability workspace raises are summed deliberately: independently
    /// scheduled fragments can be concurrently live under one attempt grant.
    /// The configured deployment floor is supplied once and is never summed.
    pub(crate) fn aggregate_with_configured_floor<I: IntoIterator<Item = Self>>(
        configured_floor_bytes: u64,
        fragments: I,
    ) -> Result<Self> {
        let mut capability_floor_raise_bytes = 0_u64;
        let mut soft_working_set_hint_bytes: Option<u64> = None;
        let mut estimate_quality = EstimateQuality::Measured;
        let mut required_capabilities = Vec::new();
        let mut attested_cugraph_graph_execution_scopes = Vec::new();
        for fragment in fragments {
            capability_floor_raise_bytes =
                capability_floor_raise_bytes.checked_add(fragment.capability_floor_raise_bytes).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.admission.requirement",
                        "aggregated capability floor raise overflows u64",
                    )
                    .with_fact("reason", "attempt_requirement_capability_raise_overflow")
                })?;
            soft_working_set_hint_bytes = match (soft_working_set_hint_bytes, fragment.soft_working_set_hint_bytes) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (left @ Some(_), None) => left,
                (None, right) => right,
            };
            estimate_quality = estimate_quality.min(fragment.estimate_quality);
            for capability in fragment.required_capabilities {
                if !required_capabilities.contains(&capability) {
                    required_capabilities.push(capability);
                }
            }
            for scope in fragment.attested_cugraph_graph_execution_scopes {
                if !attested_cugraph_graph_execution_scopes.contains(&scope) {
                    attested_cugraph_graph_execution_scopes.push(scope);
                }
            }
        }
        let mut aggregate = Self::from_parts(
            configured_floor_bytes,
            capability_floor_raise_bytes,
            soft_working_set_hint_bytes,
            estimate_quality,
            required_capabilities,
        );
        aggregate.attested_cugraph_graph_execution_scopes = attested_cugraph_graph_execution_scopes;
        Ok(aggregate)
    }
}

fn generic_graph_execution_blocker() -> ReservationRequirement {
    ReservationRequirement::blocked_by_lower_level(
        ReservationKind::OperatorOutput,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
        crate::capability::ContractQuality::ExactPreflight,
    )
}

fn cugraph_graph_execution_capability() -> ReservationRequirement {
    ReservationRequirement::modeled(
        ReservationKind::GraphAlgorithm,
        "cugraph_graph_execution",
        "cugraph_graph_execution_contract",
        crate::capability::ContractQuality::CapabilityProof,
    )
}

fn is_generic_graph_execution_blocker(requirement: &ReservationRequirement) -> bool {
    requirement == &generic_graph_execution_blocker()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        capability::{ContractQuality, ReservationRequirement},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphEdgeColumnBindings, QueryPlan, QueryPlanBuilder,
        },
        runtime::ReservationKind,
        source::{GpuResidentSource, SourceNode},
    };

    fn capability() -> ReservationRequirement {
        ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "test",
            "test_contract",
            ContractQuality::CapabilityProof,
        )
    }

    fn graph_bfs_plan() -> QueryPlan {
        graph_bfs_plan_with_source_vertex(0)
    }

    fn graph_bfs_plan_with_source_vertex(source_vertex: i64) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", schema)));
        let graph = builder.add_graph_algorithm(
            edges,
            GraphAlgorithmSpec::with_stable_schema(
                GraphAlgorithm::Bfs,
                GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
                GraphConstructionOptions::default(),
                GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(source_vertex, None))),
            ),
        );
        builder.finish_with_host_sink(graph)
    }

    fn ordinary_native_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        builder.finish_with_host_sink(source)
    }

    #[test]
    fn required_floor_uses_checked_arithmetic() {
        let requirement =
            AttemptCapabilityEvidence::from_parts(u64::MAX, 1, None, EstimateQuality::Unknown, Vec::new());
        assert_eq!(
            requirement.required_floor_bytes().unwrap_err().fact_value("reason"),
            Some("attempt_requirement_floor_overflow")
        );
    }

    #[test]
    fn legacy_band_is_an_interim_floor_and_hint() {
        let requirement = AttemptCapabilityEvidence::from_parts(10, 0, Some(20), EstimateQuality::Modeled, Vec::new());
        assert_eq!(requirement.required_floor_bytes().unwrap(), 10);
        assert_eq!(requirement.soft_working_set_hint_bytes, Some(20));
        assert_eq!(requirement.estimate_quality, EstimateQuality::Modeled);
    }

    #[test]
    fn aggregate_sums_only_raises_and_unions_capabilities() {
        let capability = capability();
        let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
            10,
            [
                AttemptCapabilityEvidence::from_parts(
                    99,
                    2,
                    Some(12),
                    EstimateQuality::Measured,
                    vec![capability.clone()],
                ),
                AttemptCapabilityEvidence::from_parts(88, 3, Some(20), EstimateQuality::Unknown, vec![capability]),
            ],
        )
        .unwrap();
        assert_eq!(result.configured_floor_bytes, 10);
        assert_eq!(result.capability_floor_raise_bytes, 5);
        assert_eq!(result.soft_working_set_hint_bytes, Some(20));
        assert_eq!(result.estimate_quality, EstimateQuality::Unknown);
        assert_eq!(result.required_capabilities.len(), 1);
    }

    #[test]
    fn aggregate_rejects_raise_overflow() {
        let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
            0,
            [
                AttemptCapabilityEvidence::empty(),
                AttemptCapabilityEvidence::from_parts(0, u64::MAX, None, EstimateQuality::Unknown, Vec::new()),
                AttemptCapabilityEvidence::from_parts(0, 1, None, EstimateQuality::Unknown, Vec::new()),
            ],
        );
        assert_eq!(result.unwrap_err().fact_value("reason"), Some("attempt_requirement_capability_raise_overflow"));
    }

    #[test]
    fn cugraph_graph_scope_rejects_an_unattested_native_plan_with_stable_facts() {
        let attested_plan = graph_bfs_plan_with_source_vertex(1);
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&attested_plan)
            .expect("attested graph execution has capability requirements");
        let error = proof
            .as_requirement()
            .validate_covers_plan(&graph_bfs_plan())
            .expect_err("a plan-scoped cuGraph proof must not authorize a different native graph plan");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
        assert_eq!(
            error.fact_value("missing_capabilities"),
            Some("operator_output:unsupported_graph_algorithm_reservation_contract")
        );
    }

    #[test]
    fn exact_cugraph_plan_proof_replaces_only_the_engine_graph_blocker() {
        let plan = graph_bfs_plan();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph execution has capability requirements");

        assert!(proof.required_capabilities().iter().any(|requirement| {
            requirement.kind == ReservationKind::GraphAlgorithm
                && requirement.required_contract == "cugraph_graph_execution_contract"
        }));
        assert!(!proof.required_capabilities().iter().any(is_generic_graph_execution_blocker));
        assert_eq!(proof.as_requirement().attested_cugraph_graph_execution_scopes.len(), 1);
        proof
            .as_requirement()
            .validate_covers_plan(&plan)
            .expect("the issued cuGraph proof must cover its exact internal plan");
    }

    #[test]
    fn combining_cugraph_proofs_retains_each_attested_graph_scope() {
        let native_plan = ordinary_native_plan();
        let first_plan = graph_bfs_plan_with_source_vertex(0);
        let second_plan = graph_bfs_plan_with_source_vertex(1);
        let unscoped_plan = graph_bfs_plan_with_source_vertex(2);
        let proof = QueryPlanCapabilityProof::combine(
            QueryPlanCapabilityProof::for_query_plan(&native_plan)
                .expect("native plan has an explicit sink capability requirement"),
            [
                QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&first_plan)
                    .expect("first graph plan has capability requirements"),
                QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&second_plan)
                    .expect("second graph plan has capability requirements"),
            ],
        )
        .expect("native and plan-scoped graph proofs must combine");

        let requirement = proof.as_requirement();
        assert_eq!(requirement.attested_cugraph_graph_execution_scopes.len(), 2);
        requirement.validate_covers_plan(&native_plan).expect("combined proof must retain the native plan capability");
        requirement.validate_covers_plan(&first_plan).expect("combined proof must retain the first graph scope");
        requirement.validate_covers_plan(&second_plan).expect("combined proof must retain the second graph scope");
        let error = requirement
            .validate_covers_plan(&unscoped_plan)
            .expect_err("combined proof must not widen its graph scope");
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
        assert_eq!(
            error.fact_value("missing_capabilities"),
            Some("operator_output:unsupported_graph_algorithm_reservation_contract")
        );
    }
}
