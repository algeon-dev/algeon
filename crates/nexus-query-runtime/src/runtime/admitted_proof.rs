use crate::{
    admission::{AttemptRequirement, AttemptReservationEnvelope},
    error::{QueryError, QueryErrorCode, Result},
    runtime::gpu::{GrantIdentity, MemoryGrant},
};

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AdmittedAttemptProof {
    reservation_envelope: AttemptReservationEnvelope,
    submitted_capability_evidence: AttemptRequirement,
    selected_device: usize,
    grant_identity: GrantIdentity,
}

impl AdmittedAttemptProof {
    #[doc(hidden)]
    pub fn from_post_grant(
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
                "nexus_query_engine.runtime.admitted_attempt_proof",
                "post-grant admission proof is missing one of its immutable planning artifacts",
            )
            .with_fact("reason", "admitted_attempt_proof_incomplete")
            .with_fact("selected_device", selected_device)
            .with_fact("grant_generation", grant_identity.generation)),
        }
    }

    #[doc(hidden)]
    pub fn reservation_envelope(&self) -> &AttemptReservationEnvelope {
        &self.reservation_envelope
    }

    #[doc(hidden)]
    pub fn submitted_capability_evidence(&self) -> &AttemptRequirement {
        &self.submitted_capability_evidence
    }

    #[doc(hidden)]
    pub const fn selected_device(&self) -> usize {
        self.selected_device
    }

    #[doc(hidden)]
    pub const fn grant_identity(&self) -> GrantIdentity {
        self.grant_identity
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn with_identity_for_test(&self, selected_device: usize, grant_identity: GrantIdentity) -> Self {
        Self {
            reservation_envelope: self.reservation_envelope.clone(),
            submitted_capability_evidence: self.submitted_capability_evidence.clone(),
            selected_device,
            grant_identity,
        }
    }
}
