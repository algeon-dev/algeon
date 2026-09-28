//! Admitted-plan capability validation for execution contexts.

use std::sync::{Arc, Mutex, PoisonError};

use query_planning::{
    admission::{AttemptCapabilityEvidence, AttemptProducerOccurrence},
    error::{QueryError, QueryErrorCode, Result},
    plan::{QueryPlan, SchemaFingerprint},
};

use super::{
    super::{
        MemoryGrant,
        resident::{GpuResidentInputMap, GpuResultHandle},
    },
    ActivatedDeferredResidentInput, ActivatedDeferredResidentPhase, DeferredResidentFacts, ExecutionContext,
};
use crate::observability::AttemptComposedScheduleProjection;

/// The plan identities an admitted attempt may execute: the evidence its
/// claim submitted, then each fragment that joined its grant after the claim.
/// Identities are only added; the claim's placement is never revisited. A
/// placement-only claim starts empty, and its plans fail closed until a
/// fragment joins.
#[derive(Debug, Default)]
pub(in crate::runtime) struct AdmittedPlanIdentities {
    evidence: Mutex<Option<Arc<AttemptCapabilityEvidence>>>,
}

impl AdmittedPlanIdentities {
    pub(in crate::runtime) fn new(evidence: Option<AttemptCapabilityEvidence>) -> Self {
        Self { evidence: Mutex::new(evidence.map(Arc::new)) }
    }

    /// The identities admitted so far.
    pub(in crate::runtime) fn current(&self) -> Option<Arc<AttemptCapabilityEvidence>> {
        self.evidence.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Adds a fragment whose capabilities the granted device already
    /// provides.
    pub(in crate::runtime) fn join(&self, fragment: &AttemptCapabilityEvidence) -> Result<()> {
        let mut evidence = self.evidence.lock().unwrap_or_else(PoisonError::into_inner);
        let mut joined = evidence.as_deref().cloned().unwrap_or_default();
        joined.join(fragment)?;
        *evidence = Some(Arc::new(joined));
        Ok(())
    }
}

impl ExecutionContext {
    /// The admitted evidence and immutable grant a plan validates against, or
    /// `None` for a standalone context. The grant, allocation domain, and
    /// attempt identity come from one admission binding, so only the
    /// evidence can be missing: a placement-only admission carries none
    /// until a fragment joins it.
    fn validated_admitted_identity(
        &self,
        plan: &QueryPlan,
    ) -> Result<Option<(Arc<AttemptCapabilityEvidence>, MemoryGrant)>> {
        let Some(attempt) = self.admitted() else {
            return Ok(None);
        };
        let evidence = attempt.plan_identities.current().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.context",
                "a query-service context requires its admitted capability evidence",
            )
            .with_fact("reason", "admitted_capability_evidence_missing")
            .with_fact("plan_scope", plan.execution_fingerprint())
        })?;
        let immutable_grant = self.admitted_memory_grant(attempt, "query_engine.runtime.context.memory_grant")?;
        Ok(Some((evidence, immutable_grant)))
    }

    fn validate_admitted_capability_evidence(
        &self,
        evidence: &AttemptCapabilityEvidence,
        plan: &QueryPlan,
    ) -> Result<()> {
        evidence.validate_covers_plan(plan)?;
        let execution_scope = plan.execution_fingerprint();
        let expected_schema = SchemaFingerprint::of(plan.caller_visible_output_schema());
        let scope_matches = evidence.execution_scopes().iter().filter(|shape| shape.matches(plan)).collect::<Vec<_>>();
        let matching_shapes = scope_matches
            .iter()
            .copied()
            .filter(|shape| shape.output_schema_fingerprint() == expected_schema)
            .collect::<Vec<_>>();
        if matching_shapes.is_empty() {
            if evidence.producer_occurrence_scope_count_for_plan(plan) > 0 {
                return Ok(());
            }
            if evidence.execution_manifest().and_then(|manifest| manifest.occurrence_for_plan(plan)).is_some() {
                return Ok(());
            }
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.context",
                "the admitted capability evidence output schema differs from the native plan",
            )
            .with_fact("reason", "admitted_capability_schema_mismatch")
            .with_fact("plan_scope", execution_scope)
            .with_fact("matching_shape_count", scope_matches.len()));
        }
        for shape in matching_shapes {
            shape.validate_manifest_core()?;
        }
        Ok(())
    }

    fn validate_admitted_occurrences(
        &self,
        evidence: &AttemptCapabilityEvidence,
        plan: &QueryPlan,
        occurrence_ids: &[String],
    ) -> Result<()> {
        evidence.validate_composed_occurrences(plan, occurrence_ids)
    }

    fn validate_admitted_evidence(
        &self,
        plan: &QueryPlan,
        occurrence_ids: Option<&[String]>,
    ) -> Result<Option<AttemptComposedScheduleProjection>> {
        let Some((submitted_evidence, immutable_grant)) = self.validated_admitted_identity(plan)? else {
            return Ok(None);
        };
        // An activated deferred-resident phase for this exact plan scope
        // replaces the submitted evidence with the scope actually proven at
        // activation (`with_admitted_gpu_inputs`); every other plan validates
        // against the evidence submitted at admission.
        let evidence = match &self.plan.activated_deferred_resident_phase {
            Some(activated) if activated.consumer_execution_scope_fingerprint == plan.execution_fingerprint() => {
                if activated.grant_identity != immutable_grant.identity() {
                    return Err(QueryError::new(
                        QueryErrorCode::InternalInvariantViolation,
                        "query_engine.runtime.context",
                        "the activated deferred phase differs from the admitted plan or immutable grant",
                    )
                    .with_fact("reason", "deferred_resident_phase_identity_mismatch"));
                }
                &activated.capability_evidence
            },
            _ => submitted_evidence.as_ref(),
        };
        self.validate_admitted_capability_evidence(evidence, plan)?;
        if let Some(occurrence_ids) = occurrence_ids {
            self.validate_admitted_occurrences(evidence, plan, occurrence_ids)?;
        }
        Ok(None)
    }

    /// Universal pre-GPU gate for service-created execution contexts. Direct
    /// contexts remain unrestricted and return success without a evidence.
    pub fn validate_admitted_plan_capabilities(&self, plan: &QueryPlan) -> Result<()> {
        let Some((evidence, _)) = self.validated_admitted_identity(plan)? else {
            return Ok(());
        };
        self.validate_admitted_capability_evidence(&evidence, plan)
    }

    /// Activates the deferred-resident phase for `plan` from its exact
    /// materialized resident inputs.
    ///
    /// Expected producer, occurrence, and schema identities come from the
    /// admitted compiled binding. Runtime frame identities are recorded only
    /// after those independent facts have matched.
    pub fn with_admitted_gpu_inputs(&self, plan: &QueryPlan, gpu_inputs: &GpuResidentInputMap) -> Result<Self> {
        let Some((submitted_evidence, immutable_grant)) = self.validated_admitted_identity(plan)? else {
            return Ok(self.clone());
        };
        if gpu_inputs.is_empty() {
            return Ok(self.clone());
        }
        // A phase activated for this same scope under a different grant is a
        // real admission violation. Otherwise the phase is re-derived from the
        // frames actually presented: one execution context legitimately runs a
        // plan more than once, and a later round may carry a different frame.
        if let Some(activated) = &self.plan.activated_deferred_resident_phase
            && activated.consumer_execution_scope_fingerprint == plan.execution_fingerprint()
            && activated.grant_identity != immutable_grant.identity()
        {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.context",
                "the activated deferred phase differs from the admitted immutable grant",
            )
            .with_fact("reason", "deferred_resident_phase_identity_mismatch"));
        }
        submitted_evidence.validate_covers_plan(plan)?;
        let expected_scopes = submitted_evidence
            .execution_scopes()
            .iter()
            .filter(|scope| scope.matches(plan) && !scope.producer_occurrences().is_empty())
            .collect::<Vec<_>>();
        if expected_scopes.is_empty() {
            return Ok(self.clone());
        }
        for scope in &expected_scopes {
            scope.validate_manifest_core()?;
        }
        let expected_scope = expected_scopes
            .iter()
            .copied()
            .find(|scope| {
                scope.producer_occurrences().iter().all(|expected| {
                    gpu_inputs.get(expected.occurrence_id()).is_some_and(|input| {
                        DeferredResidentFacts::read(expected.occurrence_id(), input)
                            .is_ok_and(|facts| facts_match_occurrence(&facts, input, expected))
                    })
                })
            })
            .unwrap_or(expected_scopes[0]);
        let mut capability_evidence = submitted_evidence.as_ref().clone();
        capability_evidence.attest_cugraph_graph_execution_scope(plan);
        let inputs = expected_scope
            .producer_occurrences()
            .iter()
            .map(|expected| {
                let input_name = expected.occurrence_id();
                let input = gpu_inputs.get(input_name).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InternalInvariantViolation,
                        "query_engine.runtime.context",
                        "the admitted deferred resident occurrence is missing",
                    )
                    .with_fact("reason", "deferred_resident_input_missing")
                    .with_fact("input_name", input_name)
                })?;
                let facts = DeferredResidentFacts::read(input_name, input)?;
                if !facts_match_occurrence(&facts, input, expected) {
                    return Err(QueryError::new(
                        QueryErrorCode::InternalInvariantViolation,
                        "query_engine.runtime.context",
                        "a deferred resident input differs from its admitted producer binding",
                    )
                    .with_fact("reason", "deferred_resident_producer_binding_mismatch")
                    .with_fact("input_name", input_name)
                    .with_fact("expected_producer_scope", expected.execution_scope_fingerprint())
                    .with_fact("actual_producer_scope", facts.producer_execution_scope_fingerprint())
                    .with_fact("expected_schema", expected.output_schema_fingerprint())
                    .with_fact("actual_schema", SchemaFingerprint::of(facts.schema())));
                }
                ActivatedDeferredResidentInput::bind(input_name.to_owned(), facts, input, &immutable_grant)
            })
            .collect::<Result<Vec<_>>>()?;
        let activated = ActivatedDeferredResidentPhase {
            consumer_execution_scope_fingerprint: plan.execution_fingerprint(),
            inputs,
            capability_evidence,
            grant_identity: immutable_grant.identity(),
        };
        activated.validate_inputs(gpu_inputs, &immutable_grant)?;
        Ok(self.with_plan(|plan| plan.activated_deferred_resident_phase = Some(Arc::new(activated))))
    }

    #[doc(hidden)]
    pub fn validate_admitted_plan_capabilities_with_gpu_inputs(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> Result<()> {
        self.validate_admitted_plan_capabilities(plan)?;
        let Some(activated) = self
            .plan
            .activated_deferred_resident_phase
            .as_ref()
            .filter(|activated| activated.consumer_execution_scope_fingerprint == plan.execution_fingerprint())
        else {
            return Ok(());
        };
        let immutable_grant = self.memory_grant()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.context",
                "the activated deferred phase lost its immutable grant",
            )
            .with_fact("reason", "deferred_resident_grant_missing")
        })?;
        activated.validate_inputs(gpu_inputs, &immutable_grant)
    }

    /// Validates one composed attempt against its admitted capability and
    /// resident-input identity contracts. Physical allocation remains owned
    /// by the immutable grant and live resident handles.
    pub fn validate_admitted_composed_attempt(
        &self,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
    ) -> Result<Option<AttemptComposedScheduleProjection>> {
        self.validate_admitted_evidence(consumer_plan, Some(occurrence_ids))
    }
}

/// Whether an input's facts are those of the producer occurrence the attempt
/// admitted: written by the admitted producer scope, or by a plan equivalent
/// to it, with the admitted output schema.
fn facts_match_occurrence(
    facts: &DeferredResidentFacts,
    input: &GpuResultHandle,
    expected: &AttemptProducerOccurrence,
) -> bool {
    (facts.producer_execution_scope_fingerprint() == expected.execution_scope_fingerprint()
        || input.producer_execution_plan().is_some_and(|plan| expected.matches_execution_plan(plan)))
        && SchemaFingerprint::of(facts.schema()) == expected.output_schema_fingerprint()
}
