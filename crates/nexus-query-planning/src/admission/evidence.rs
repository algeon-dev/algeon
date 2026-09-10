//! Capability and plan-scope evidence for one admitted execution attempt.
//!
//! This evidence answers two questions and no others: which device
//! capabilities the plan requires, and whether the plan about to execute is
//! the plan that was admitted. Memory sizing is not represented here. The
//! query domain's allocator is the single arbiter of a query's allocation
//! limit, so a planning-time byte model would be a second, weaker source of
//! truth that can only reject work the allocator would have admitted.

use crate::{
    admission::scope::{AttemptExecutionScope, AttemptProducerOccurrence},
    error::{QueryError, QueryErrorCode, Result},
    plan::{QueryPlan, canonical_execution_scope_fingerprint},
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
    pub const fn as_str(self) -> &'static str {
        self.name
    }
}

/// Device capability required for cuGraph algorithm execution.
pub const CUGRAPH_GRAPH_EXECUTION_CAPABILITY: ExecutionCapability = ExecutionCapability::new("cugraph_graph_execution");

/// A cuGraph graph-execution plan scope admitted for one attempt.
///
/// The scope is retained rather than a capability label alone: a label would
/// let evidence issued for one graph plan authorize an unrelated one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttestedCugraphGraphExecutionScope {
    #[doc(hidden)]
    pub execution_scope_fingerprint: String,
}

impl AttestedCugraphGraphExecutionScope {
    #[doc(hidden)]
    pub fn for_plan(plan: &QueryPlan) -> Self {
        Self { execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan) }
    }

    #[doc(hidden)]
    pub fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }
}

/// Device capabilities and plan scopes admitted for one attempt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AttemptCapabilityEvidence {
    /// Device execution capabilities required by the admitted fragments.
    #[doc(hidden)]
    pub required_capabilities: Vec<ExecutionCapability>,
    /// Graph plans admitted for cuGraph execution.
    #[doc(hidden)]
    pub attested_cugraph_graph_execution_scopes: Vec<AttestedCugraphGraphExecutionScope>,
    /// Plan-scope and output-schema identity for each admitted fragment.
    #[doc(hidden)]
    pub execution_scopes: Vec<AttemptExecutionScope>,
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
        }
    }

    #[doc(hidden)]
    pub fn from_parts(
        required_capabilities: Vec<ExecutionCapability>,
        execution_scopes: Vec<AttemptExecutionScope>,
    ) -> Self {
        Self { required_capabilities, attested_cugraph_graph_execution_scopes: Vec::new(), execution_scopes }
    }

    #[doc(hidden)]
    pub fn required_capabilities(&self) -> &[ExecutionCapability] {
        &self.required_capabilities
    }

    #[doc(hidden)]
    pub fn execution_scopes(&self) -> &[AttemptExecutionScope] {
        &self.execution_scopes
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

    #[doc(hidden)]
    pub fn push_execution_scope(&mut self, scope: AttemptExecutionScope) {
        self.execution_scopes.push(scope);
    }

    #[doc(hidden)]
    pub fn push_producer_occurrence(&mut self, occurrence: AttemptProducerOccurrence) -> Result<()> {
        let Some(scope) = self.execution_scopes.last_mut() else {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.evidence",
                "a producer occurrence has no admitted consumer scope to bind to",
            )
            .with_fact("reason", "missing_consumer_scope"));
        };
        scope.push_producer_occurrence(occurrence);
        Ok(())
    }

    /// Merges the evidence of independently lowered fragments of one
    /// invocation. Capabilities are unioned because every fragment executes
    /// under the same attempt.
    #[doc(hidden)]
    pub fn aggregate<I: IntoIterator<Item = Self>>(evidence: I) -> Self {
        let mut aggregate = Self::empty();
        for one in evidence {
            for capability in one.required_capabilities {
                if !aggregate.required_capabilities.contains(&capability) {
                    aggregate.required_capabilities.push(capability);
                }
            }
            for scope in one.attested_cugraph_graph_execution_scopes {
                if !aggregate.attested_cugraph_graph_execution_scopes.contains(&scope) {
                    aggregate.attested_cugraph_graph_execution_scopes.push(scope);
                }
            }
            aggregate.execution_scopes.extend(one.execution_scopes);
        }
        aggregate
    }

    #[doc(hidden)]
    pub fn producer_occurrence_scope_count_for_plan(&self, plan: &QueryPlan) -> usize {
        let execution_scope = canonical_execution_scope_fingerprint(plan);
        self.execution_scopes.iter().map(|scope| scope.producer_occurrence_scope_count(&execution_scope)).sum()
    }

    /// Rejects execution of a plan the admitted evidence does not cover.
    ///
    /// Device capability and plan scope must both cover graph execution.
    #[doc(hidden)]
    pub fn validate_covers_plan(&self, plan: &QueryPlan) -> Result<()> {
        if plan.contains_graph_algorithm()
            && (!self.required_capabilities.contains(&CUGRAPH_GRAPH_EXECUTION_CAPABILITY)
                || !self.attested_cugraph_graph_execution_scopes.iter().any(|scope| scope.matches(plan)))
        {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.evidence",
                "admitted capability evidence does not authorize cuGraph execution for the native plan",
            )
            .with_fact("reason", "admitted_capability_evidence_mismatch")
            .with_fact("required_capability", CUGRAPH_GRAPH_EXECUTION_CAPABILITY.as_str())
            .with_fact("admitted_capabilities", capability_labels(&self.required_capabilities)));
        }
        let execution_scope = canonical_execution_scope_fingerprint(plan);
        let scope_matches = self
            .execution_scopes
            .iter()
            .filter(|scope| scope.matches_execution_scope_fingerprint(&execution_scope))
            .count();
        let occurrence_matches = self
            .execution_scopes
            .iter()
            .map(|scope| scope.producer_occurrence_scope_count(&execution_scope))
            .sum::<usize>();
        if scope_matches > 0 || occurrence_matches > 0 {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.evidence",
            "admitted capability evidence has no scope for the native plan about to execute",
        )
        .with_fact("reason", "admitted_capability_scope_mismatch")
        .with_fact("plan_scope", execution_scope))
    }
}

fn capability_labels(capabilities: &[ExecutionCapability]) -> String {
    capabilities.iter().map(|capability| capability.as_str()).collect::<Vec<_>>().join(",")
}
