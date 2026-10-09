//! Capability and plan-scope evidence for one admitted execution attempt.
//!
//! This evidence answers two questions and no others: which device
//! capabilities the plan requires, and whether the plan about to execute is
//! the plan that was admitted. The first decides placement once, when the
//! attempt claims its grant; the answer to the second grows as fragments
//! planned after the claim join that grant. Memory sizing is not represented here. The
//! query domain's allocator is the single arbiter of a query's allocation
//! limit, so a planning-time byte model would be a second, weaker source of
//! truth that can only reject work the allocator would have admitted.

use crate::{
    admission::scope::{AttemptExecutionManifest, AttemptExecutionScope, AttemptProducerOccurrence},
    error::{QueryError, QueryErrorCode, Result},
    plan::{DeferredGraphExecutionScopeIdentity, PlanFingerprint, QueryPlan, SchemaFingerprint},
};

/// Immutable device execution capability required by an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionCapability {
    name: &'static str,
}

impl ExecutionCapability {
    /// Creates a capability with a stable diagnostic identity.
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// Returns the stable capability identity.
    pub(crate) const fn as_str(self) -> &'static str {
        self.name
    }
}

/// Device capability required for cuGraph algorithm execution.
pub const CUGRAPH_GRAPH_EXECUTION_CAPABILITY: ExecutionCapability = ExecutionCapability::new("cugraph_graph_execution");

/// A cuGraph graph-execution plan scope admitted for one attempt.
///
/// The scope is retained. A capability label alone would let evidence issued
/// for one graph plan authorize an unrelated one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttestedCugraphGraphExecutionScope {
    pub(crate) execution_scope_fingerprint: PlanFingerprint,
    pub(crate) deferred_execution_scope_identity: Option<DeferredGraphExecutionScopeIdentity>,
}

impl AttestedCugraphGraphExecutionScope {
    pub(crate) fn for_plan(plan: &QueryPlan) -> Self {
        Self { execution_scope_fingerprint: plan.execution_fingerprint(), deferred_execution_scope_identity: None }
    }

    pub(crate) fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == plan.execution_fingerprint()
            || self.deferred_execution_scope_identity.as_ref().is_some_and(|scope| scope.matches(plan))
    }
}

/// Device capabilities and plan scopes admitted for one attempt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AttemptCapabilityEvidence {
    /// Device execution capabilities required by the admitted fragments.
    #[doc(hidden)]
    pub required_capabilities: Vec<ExecutionCapability>,
    /// Graph plans admitted for cuGraph execution.
    pub(crate) attested_cugraph_graph_execution_scopes: Vec<AttestedCugraphGraphExecutionScope>,
    /// Plan-scope and output-schema identity for each admitted fragment.
    pub(crate) execution_scopes: Vec<AttemptExecutionScope>,
    /// GPU UDTF occurrences and CPU dependency boundaries for this DataFusion invocation.
    pub(crate) execution_manifest: Option<AttemptExecutionManifest>,
}

impl AttemptCapabilityEvidence {
    /// Evidence for an attempt with no capability or scope requirement.
    #[doc(hidden)]
    pub fn empty() -> Self {
        Self::default()
    }

    #[doc(hidden)]
    pub fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            required_capabilities: Vec::new(),
            attested_cugraph_graph_execution_scopes: Vec::new(),
            execution_scopes: vec![AttemptExecutionScope::for_query_plan(plan)],
            execution_manifest: None,
        }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn from_parts(
        required_capabilities: Vec<ExecutionCapability>,
        execution_scopes: Vec<AttemptExecutionScope>,
    ) -> Self {
        Self {
            required_capabilities,
            attested_cugraph_graph_execution_scopes: Vec::new(),
            execution_scopes,
            execution_manifest: None,
        }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn required_capabilities(&self) -> &[ExecutionCapability] {
        &self.required_capabilities
    }

    #[doc(hidden)]
    pub fn execution_scopes(&self) -> &[AttemptExecutionScope] {
        &self.execution_scopes
    }

    #[doc(hidden)]
    pub fn execution_manifest(&self) -> Option<&AttemptExecutionManifest> {
        self.execution_manifest.as_ref()
    }

    #[doc(hidden)]
    pub fn deferred_graph_identity_for_scope(
        &self,
        execution_scope_fingerprint: PlanFingerprint,
    ) -> Option<DeferredGraphExecutionScopeIdentity> {
        self.attested_cugraph_graph_execution_scopes
            .iter()
            .find(|scope| scope.execution_scope_fingerprint == execution_scope_fingerprint)
            .and_then(|scope| scope.deferred_execution_scope_identity.clone())
    }

    /// Removes invocation-owned occurrence ordering when deriving evidence for
    /// one nested fragment. The outer invocation retains the sole manifest.
    #[doc(hidden)]
    pub fn without_execution_manifest(mut self) -> Self {
        self.execution_manifest = None;
        self
    }

    #[doc(hidden)]
    pub fn with_execution_manifest(mut self, manifest: AttemptExecutionManifest) -> Result<Self> {
        if self.execution_manifest.is_some() {
            return Err(duplicate_execution_manifest_error());
        }
        self.execution_manifest = Some(manifest);
        Ok(self)
    }

    #[doc(hidden)]
    pub fn attest_cugraph_graph_execution_scope(&mut self, plan: &QueryPlan) {
        if !self.required_capabilities.contains(&CUGRAPH_GRAPH_EXECUTION_CAPABILITY) {
            self.required_capabilities.push(CUGRAPH_GRAPH_EXECUTION_CAPABILITY);
        }
        let scope = AttestedCugraphGraphExecutionScope::for_plan(plan);
        if !self.attested_cugraph_graph_execution_scopes.contains(&scope) {
            self.attested_cugraph_graph_execution_scopes.push(scope);
        }
    }

    /// Attests a graph scope whose currently missing source cardinality facts
    /// may be filled from this occurrence's staged Arrow inputs.
    #[doc(hidden)]
    pub fn attest_deferred_cugraph_graph_execution_scope(&mut self, plan: &QueryPlan) {
        self.attest_cugraph_graph_execution_scope(plan);
        let exact = plan.execution_fingerprint();
        if let Some(scope) = self
            .attested_cugraph_graph_execution_scopes
            .iter_mut()
            .find(|scope| scope.execution_scope_fingerprint == exact)
        {
            scope.deferred_execution_scope_identity = Some(DeferredGraphExecutionScopeIdentity::for_plan(plan));
        }
    }

    /// Binds a producer occurrence to the most recently added scope, which is
    /// the consumer scope under assembly: a consumer's producers are pushed
    /// after its scope and before any further scope.
    #[doc(hidden)]
    pub fn push_producer_occurrence(&mut self, occurrence: AttemptProducerOccurrence) -> Result<()> {
        let Some(scope) = self.execution_scopes.last_mut() else {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.admission.evidence",
                "a producer occurrence has no admitted consumer scope to bind to",
            )
            .with_fact("reason", "missing_consumer_scope"));
        };
        scope.push_producer_occurrence(occurrence);
        Ok(())
    }

    /// Whether every capability, scope, and manifest `fragment` carries is
    /// already part of this evidence, so admitting this evidence admitted it.
    #[doc(hidden)]
    pub fn covers(&self, fragment: &Self) -> bool {
        fragment.required_capabilities.iter().all(|capability| self.required_capabilities.contains(capability))
            && fragment
                .attested_cugraph_graph_execution_scopes
                .iter()
                .all(|scope| self.attested_cugraph_graph_execution_scopes.contains(scope))
            && fragment.execution_scopes.iter().all(|scope| self.execution_scopes.contains(scope))
            && fragment
                .execution_manifest
                .as_ref()
                .is_none_or(|manifest| self.execution_manifest.as_ref() == Some(manifest))
    }

    /// Merges the evidence of independently lowered fragments of one
    /// invocation. Capabilities are unioned because every fragment executes
    /// under the same attempt.
    #[doc(hidden)]
    pub fn aggregate<I: IntoIterator<Item = Self>>(evidence: I) -> Result<Self> {
        let mut aggregate = Self::empty();
        for one in evidence {
            if one.execution_manifest.is_some() && aggregate.execution_manifest.is_some() {
                return Err(duplicate_execution_manifest_error());
            }
            aggregate.union(&one);
        }
        Ok(aggregate)
    }

    /// Adds a fragment planned after the attempt's claim. Placement was fixed
    /// by the claim and is not revisited here: the caller first checks the
    /// fragment's capabilities against the granted device. The fragment's
    /// capabilities and plan identity then join this evidence. The attempt
    /// keeps one execution manifest, so a fragment's manifest must be the one
    /// already bound, if any.
    #[doc(hidden)]
    pub fn join(&mut self, fragment: &Self) -> Result<()> {
        if let (Some(bound), Some(manifest)) = (&self.execution_manifest, &fragment.execution_manifest)
            && bound != manifest
        {
            return Err(duplicate_execution_manifest_error());
        }
        self.union(fragment);
        Ok(())
    }

    /// Set union of every list; the first manifest is kept. `join` and
    /// `aggregate` reject a conflicting second manifest before calling this.
    fn union(&mut self, other: &Self) {
        for capability in &other.required_capabilities {
            if !self.required_capabilities.contains(capability) {
                self.required_capabilities.push(*capability);
            }
        }
        for scope in &other.attested_cugraph_graph_execution_scopes {
            if !self.attested_cugraph_graph_execution_scopes.contains(scope) {
                self.attested_cugraph_graph_execution_scopes.push(scope.clone());
            }
        }
        for scope in &other.execution_scopes {
            if !self.execution_scopes.contains(scope) {
                self.execution_scopes.push(scope.clone());
            }
        }
        if self.execution_manifest.is_none() {
            self.execution_manifest.clone_from(&other.execution_manifest);
        }
    }

    #[doc(hidden)]
    pub fn producer_occurrence_scope_count_for_plan(&self, plan: &QueryPlan) -> usize {
        let execution_scope = plan.execution_fingerprint();
        self.execution_scopes.iter().map(|scope| scope.producer_occurrence_scope_count(execution_scope)).sum()
    }

    /// Validates the ordered producer occurrences of one composed native plan.
    ///
    /// Checks go from coarse to fine so the error fact names the first
    /// mismatch: a scope for the plan and schema, then the occurrence count,
    /// then the occurrence order.
    #[doc(hidden)]
    pub fn validate_composed_occurrences(&self, plan: &QueryPlan, occurrence_ids: &[String]) -> Result<()> {
        let expected_schema = SchemaFingerprint::of(plan.caller_visible_output_schema());
        let matching = self
            .execution_scopes
            .iter()
            .filter(|shape| shape.matches(plan) && shape.output_schema_fingerprint() == expected_schema)
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.admission.evidence",
                "the admitted capability evidence has no occurrence manifest for the native plan",
            )
            .with_fact("reason", "admitted_capability_scope_mismatch")
            .with_fact("matching_shape_count", 0));
        }
        let scheduled = matching
            .iter()
            .map(|shape| {
                shape.producer_occurrences().iter().map(|occurrence| occurrence.occurrence_id()).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if scheduled.iter().any(|candidate| candidate.as_slice() == occurrence_ids) {
            return Ok(());
        }
        let Some(same_len) = scheduled.iter().find(|candidate| candidate.len() == occurrence_ids.len()) else {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.admission.evidence",
                "the runtime composed input count differs from the admitted capability manifest",
            )
            .with_fact("schedule_reason", "runtime_occurrence_count_mismatch")
            .with_fact("scheduled_occurrence_count", scheduled.first().map_or(0, Vec::len))
            .with_fact("runtime_occurrence_count", occurrence_ids.len()));
        };
        for (scheduled, runtime) in same_len.iter().zip(occurrence_ids) {
            if *scheduled != runtime {
                return Err(QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "query_engine.admission.evidence",
                    "the runtime composed input order differs from the admitted capability manifest",
                )
                .with_fact("schedule_reason", "runtime_occurrence_order_mismatch")
                .with_fact("scheduled_occurrence_id", *scheduled)
                .with_fact("runtime_occurrence_id", runtime));
            }
        }
        Ok(())
    }

    /// Rejects execution of a plan the admitted evidence does not cover.
    ///
    /// Device capability and plan scope must both cover graph execution. A
    /// plan's scope is covered by any one of an admitted fragment scope, a
    /// producer occurrence bound to a consumer scope, or a manifest
    /// occurrence, since each records an admission of that plan.
    #[doc(hidden)]
    pub fn validate_covers_plan(&self, plan: &QueryPlan) -> Result<()> {
        if plan.contains_graph_algorithm()
            && (!self.required_capabilities.contains(&CUGRAPH_GRAPH_EXECUTION_CAPABILITY)
                || !self.attested_cugraph_graph_execution_scopes.iter().any(|scope| scope.matches(plan)))
        {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.admission.evidence",
                "admitted capability evidence does not authorize cuGraph execution for the native plan",
            )
            .with_fact("reason", "admitted_capability_evidence_mismatch")
            .with_fact("required_capability", CUGRAPH_GRAPH_EXECUTION_CAPABILITY.as_str())
            .with_fact("admitted_capabilities", capability_labels(&self.required_capabilities)));
        }
        let execution_scope = plan.execution_fingerprint();
        let scope_matches = self
            .execution_scopes
            .iter()
            .filter(|scope| scope.matches_execution_scope_fingerprint(execution_scope))
            .count();
        let occurrence_matches = self
            .execution_scopes
            .iter()
            .map(|scope| scope.producer_occurrence_scope_count(execution_scope))
            .sum::<usize>();
        let manifest_matches = self.execution_manifest.as_ref().and_then(|manifest| manifest.occurrence_for_plan(plan));
        if scope_matches > 0 || occurrence_matches > 0 || manifest_matches.is_some() {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::InternalInvariantViolation,
            "query_engine.admission.evidence",
            "admitted capability evidence has no scope for the native plan about to execute",
        )
        .with_fact("reason", "admitted_capability_scope_mismatch")
        .with_fact("plan_scope", execution_scope))
    }
}

fn duplicate_execution_manifest_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::InternalInvariantViolation,
        "query_engine.admission.evidence",
        "an admitted attempt has more than one execution manifest",
    )
    .with_fact("reason", "duplicate_execution_manifest")
}

pub fn capability_labels(capabilities: &[ExecutionCapability]) -> String {
    capabilities.iter().map(|capability| capability.as_str()).collect::<Vec<_>>().join(",")
}
