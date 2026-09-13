//! Admitted-plan capability validation for execution contexts.

use std::sync::Arc;

use query_planning::{
    admission::{AttemptCapabilityEvidence, canonical_execution_scope_fingerprint, log_safe_fingerprint},
    error::{QueryError, QueryErrorCode, Result},
    plan::QueryPlan,
};

use super::{
    super::{MemoryGrant, resident::GpuResidentInputMap},
    ActivatedDeferredResidentInput, ActivatedDeferredResidentPhase, ExecutionContext,
};
use crate::observability::AttemptComposedScheduleProjection;
#[cfg(test)]
use crate::runtime::attempt_domain::AttemptAllocationDomain;

impl ExecutionContext {
    fn validated_admitted_identity(
        &self,
        plan: &QueryPlan,
    ) -> Result<Option<(&AttemptCapabilityEvidence, MemoryGrant)>> {
        if !self.query_service_admitted {
            return Ok(None);
        }
        let evidence = self.admitted_capability_evidence.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "a query-service context requires its admitted capability evidence",
            )
            .with_fact("reason", "admitted_capability_evidence_missing")
            .with_fact("plan_scope", canonical_execution_scope_fingerprint(plan))
        })?;
        let runtime_device_ordinal = self.runtime()?.device().ordinal();
        let immutable_grant = self.memory_grant()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "an admitted context requires its immutable grant",
            )
            .with_fact("reason", "admitted_grant_missing")
        })?;
        if immutable_grant.device_ordinal() != runtime_device_ordinal {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the immutable grant device differs from the runtime device",
            )
            .with_fact("reason", "admitted_grant_device_mismatch")
            .with_fact("grant_device_ordinal", immutable_grant.device_ordinal())
            .with_fact("runtime_device_ordinal", runtime_device_ordinal));
        }
        if self.attempt_id != Some(immutable_grant.identity().attempt_id) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the context attempt identity differs from the immutable grant",
            )
            .with_fact("reason", "admitted_grant_attempt_mismatch")
            .with_fact(
                "context_attempt_id",
                self.attempt_id.map(|attempt_id| attempt_id.to_string()).unwrap_or_default(),
            )
            .with_fact("grant_attempt_id", immutable_grant.identity().attempt_id));
        }
        let domain = self.attempt_allocation_domain.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "an admitted context requires its allocation domain",
            )
            .with_fact("reason", "admitted_allocation_domain_missing")
        })?;
        if domain.device_ordinal() != runtime_device_ordinal || domain.grant_identity() != immutable_grant.identity() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "query_engine.runtime.context",
                "the allocation domain identity differs from the immutable grant",
            )
            .with_fact("reason", "admitted_allocation_domain_identity_mismatch")
            .with_fact("domain_device_ordinal", domain.device_ordinal())
            .with_fact("grant_device_ordinal", immutable_grant.device_ordinal())
            .with_fact("domain_generation", domain.grant_identity().generation)
            .with_fact("grant_generation", immutable_grant.identity().generation));
        }
        Ok(Some((evidence, immutable_grant)))
    }

    fn validate_admitted_capability_evidence(
        &self,
        evidence: &AttemptCapabilityEvidence,
        plan: &QueryPlan,
    ) -> Result<()> {
        evidence.validate_covers_plan(plan)?;
        let execution_scope = canonical_execution_scope_fingerprint(plan);
        let caller_visible_plan = plan.clone_with_caller_visible_sink_contract();
        let expected_schema = log_safe_fingerprint(&format!("{:?}", caller_visible_plan.output_schema()));
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
                QueryErrorCode::UnsupportedPlanShape,
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
        let evidence = match &self.activated_deferred_resident_phase {
            Some(activated)
                if activated.consumer_execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan) =>
            {
                if activated.grant_identity != immutable_grant.identity() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "the activated deferred phase differs from the admitted plan or immutable grant",
                    )
                    .with_fact("reason", "deferred_resident_phase_identity_mismatch"));
                }
                &activated.capability_evidence
            },
            _ => submitted_evidence,
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
        self.validate_admitted_capability_evidence(evidence, plan)
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
        if let Some(activated) = &self.activated_deferred_resident_phase
            && activated.consumer_execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
            && activated.grant_identity != immutable_grant.identity()
        {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
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
                    let Some(input) = gpu_inputs.get(expected.occurrence_id()) else {
                        return false;
                    };
                    let Some(actual_scope) = input.producer_execution_scope() else {
                        return false;
                    };
                    let actual_schema = log_safe_fingerprint(&format!("{:?}", input.schema()));
                    (actual_scope == expected.execution_scope_fingerprint()
                        || input.producer_execution_plan().is_some_and(|plan| expected.matches_execution_plan(plan)))
                        && actual_schema == expected.output_schema_fingerprint()
                })
            })
            .unwrap_or(expected_scopes[0]);
        let mut capability_evidence = submitted_evidence.clone();
        capability_evidence.attest_cugraph_graph_execution_scope(plan);
        let inputs = expected_scope
            .producer_occurrences()
            .iter()
            .map(|expected| {
                let input_name = expected.occurrence_id();
                let input = gpu_inputs.get(input_name).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "the admitted deferred resident occurrence is missing",
                    )
                    .with_fact("reason", "deferred_resident_input_missing")
                    .with_fact("input_name", input_name)
                })?;
                let actual_scope = input.producer_execution_scope().ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "an admitted deferred resident occurrence has no producer scope",
                    )
                    .with_fact("reason", "deferred_resident_input_scope_missing")
                    .with_fact("input_name", input_name)
                })?;
                let actual_schema = log_safe_fingerprint(&format!("{:?}", input.schema()));
                if actual_scope != expected.execution_scope_fingerprint()
                    && !input.producer_execution_plan().is_some_and(|plan| expected.matches_execution_plan(plan))
                    || actual_schema != expected.output_schema_fingerprint()
                {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "a deferred resident input differs from its admitted producer binding",
                    )
                    .with_fact("reason", "deferred_resident_producer_binding_mismatch")
                    .with_fact("input_name", input_name)
                    .with_fact("expected_producer_scope", expected.execution_scope_fingerprint())
                    .with_fact("actual_producer_scope", actual_scope)
                    .with_fact("expected_schema", expected.output_schema_fingerprint())
                    .with_fact("actual_schema", actual_schema));
                }
                let physical_bytes = input.device_allocated_size_bytes()?;
                if physical_bytes > 0 && input.retained_owner_identity_for(&immutable_grant).is_none() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedPlanShape,
                        "query_engine.runtime.context",
                        "a deferred resident input is not retained by the admitted grant",
                    )
                    .with_fact("reason", "deferred_resident_grant_owner_mismatch")
                    .with_fact("input_name", input_name)
                    .with_fact("input_device_bytes", physical_bytes));
                }
                let input_row_count = u64::try_from(input.frame().num_rows()).map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::SizeOverflow,
                        "query_engine.runtime.context",
                        "the resident input row count exceeds the admission integer range",
                    )
                    .with_fact("reason", "deferred_resident_input_row_count_overflow")
                    .with_fact("input_name", input_name)
                })?;
                Ok(ActivatedDeferredResidentInput {
                    producer_execution_scope_fingerprint: actual_scope.to_owned(),
                    input_name: input_name.to_owned(),
                    input_schema: Arc::clone(input.schema()),
                    input_memory_identity: input.frame().memory_identity(),
                    input_row_count,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let activated = ActivatedDeferredResidentPhase {
            consumer_execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            inputs,
            capability_evidence,
            grant_identity: immutable_grant.identity(),
        };
        activated.validate_inputs(gpu_inputs, &immutable_grant)?;
        let mut context = self.clone();
        context.activated_deferred_resident_phase = Some(Arc::new(activated));
        Ok(context)
    }

    #[doc(hidden)]
    pub fn validate_admitted_plan_capabilities_with_gpu_inputs(
        &self,
        plan: &QueryPlan,
        gpu_inputs: &GpuResidentInputMap,
    ) -> Result<()> {
        self.validate_admitted_plan_capabilities(plan)?;
        let Some(activated) = self.activated_deferred_resident_phase.as_ref().filter(|activated| {
            activated.consumer_execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
        }) else {
            return Ok(());
        };
        let immutable_grant = self.memory_grant()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
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

    #[cfg(test)]
    #[doc(hidden)]
    pub fn replace_admitted_capability_evidence_for_test(&mut self, evidence: Option<Arc<AttemptCapabilityEvidence>>) {
        self.admitted_capability_evidence = evidence;
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn admitted_capability_evidence_for_test(&self) -> Option<&AttemptCapabilityEvidence> {
        self.admitted_capability_evidence.as_deref()
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn replace_attempt_allocation_domain_for_test(&mut self, domain: Option<AttemptAllocationDomain>) {
        self.attempt_allocation_domain = domain;
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn attempt_allocation_domain_for_test(&self) -> Option<AttemptAllocationDomain> {
        self.attempt_allocation_domain.clone()
    }

    // Plain `#[cfg(test)]` would keep this invisible to downstream test
    // targets, which need it to cover the adapter fail-closed admission guard.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn replace_query_service_admitted_for_test(&mut self, admitted: bool) {
        self.query_service_admitted = admitted;
    }
}
