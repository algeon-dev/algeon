//! Analysis-owned bounded-pipeline admission.
//!
//! Admission composes capability analysis with compiled-pipeline structural
//! support. It does not execute GPU work.

mod bounded;
mod diagnostics;
mod envelope;
mod join_strategy;
mod requirement;
mod shared;
mod string_payload;
mod support;

pub(crate) use bounded::bounded_cugraph_pipeline_admission;
pub use bounded::{
    BoundedPipelineAdmission, bounded_pipeline_admission, bounded_pipeline_admission_with_statistical_aggregate_mode,
    bounded_pipeline_rejection_projection, bounded_pipeline_rejection_projection_with_statistical_aggregate_mode,
};
pub use diagnostics::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts};
pub(crate) use diagnostics::{
    bounded_join_unsupported_shape_facts_with_projection, insert_bounded_join_rejection_projection_error_facts,
};
pub(crate) use envelope::{
    AttemptReservationEnvelope, AttemptReservationPolicy, canonical_execution_scope_fingerprint,
};
pub(crate) use join_strategy::{CompiledJoinExecutionStrategy, CompiledJoinStrategySelection};
pub(crate) use requirement::{AttemptBoundDerivation, AttemptCapabilityEvidence as AttemptRequirement};
#[cfg(any(test, feature = "test-hooks"))]
pub use requirement::{AttemptCapabilityEvidence, EstimateQuality};
pub use requirement::{ComposedGraphProducerCandidate, ComposedProducerBindingDiagnostics, QueryPlanCapabilityProof};
pub(crate) use string_payload::plan_node_is_unique_on;

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::gpu::{GrantIdentity, MemoryGrant},
};

/// Immutable execution identity produced only after a ledger grant commits.
///
/// The pre-admission requirement and reservation envelope remain separate
/// planning artifacts. Once a grant exists, execution carries this one
/// post-grant proof instead of independently validating those artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdmittedAttemptProof {
    reservation_envelope: AttemptReservationEnvelope,
    submitted_capability_evidence: AttemptRequirement,
    selected_device: usize,
    grant_identity: GrantIdentity,
}

impl AdmittedAttemptProof {
    /// Builds the post-grant proof at the admission publication seam.
    ///
    /// Placement-only requests intentionally have no execution proof. A
    /// partially populated pair is an admission invariant violation and must
    /// roll back the provisional envelope rather than publish an incomplete
    /// context.
    pub(crate) fn from_post_grant(
        reservation_envelope: Option<AttemptReservationEnvelope>,
        submitted_capability_evidence: Option<AttemptRequirement>,
        physical_grant: &MemoryGrant,
    ) -> Result<Option<Self>> {
        let selected_device = physical_grant.device_ordinal();
        let grant_identity = physical_grant.identity();
        match (reservation_envelope, submitted_capability_evidence) {
            (Some(reservation_envelope), Some(submitted_capability_evidence)) => {
                Ok(Some(Self { reservation_envelope, submitted_capability_evidence, selected_device, grant_identity }))
            },
            (None, None) => Ok(None),
            (Some(_), None) | (None, Some(_)) => Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.admission.admitted_attempt_proof",
                "post-grant admission proof is missing one of its immutable planning artifacts",
            )
            .with_fact("reason", "admitted_attempt_proof_incomplete")
            .with_fact("selected_device", selected_device)
            .with_fact("grant_generation", grant_identity.generation)),
        }
    }

    pub(crate) fn reservation_envelope(&self) -> &AttemptReservationEnvelope {
        &self.reservation_envelope
    }

    pub(crate) fn submitted_capability_evidence(&self) -> &AttemptRequirement {
        &self.submitted_capability_evidence
    }

    pub(crate) const fn selected_device(&self) -> usize {
        self.selected_device
    }

    pub(crate) const fn grant_identity(&self) -> GrantIdentity {
        self.grant_identity
    }

    #[cfg(test)]
    pub(crate) fn with_identity_for_test(&self, selected_device: usize, grant_identity: GrantIdentity) -> Self {
        Self {
            reservation_envelope: self.reservation_envelope.clone(),
            submitted_capability_evidence: self.submitted_capability_evidence.clone(),
            selected_device,
            grant_identity,
        }
    }
}
#[doc(hidden)]
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
