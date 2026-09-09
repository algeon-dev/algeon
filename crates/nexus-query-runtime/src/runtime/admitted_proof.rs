use nexus_query_planning::admission::AttemptRequirement;

use crate::runtime::gpu::{GrantIdentity, MemoryGrant};

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AdmittedAttemptProof {
    submitted_capability_evidence: AttemptRequirement,
    selected_device: usize,
    grant_identity: GrantIdentity,
}

impl AdmittedAttemptProof {
    #[doc(hidden)]
    pub fn from_post_grant(
        submitted_capability_evidence: Option<AttemptRequirement>,
        physical_grant: &MemoryGrant,
    ) -> Option<Self> {
        let selected_device = physical_grant.device_ordinal();
        let grant_identity = physical_grant.identity();
        submitted_capability_evidence.map(|submitted_capability_evidence| Self {
            submitted_capability_evidence,
            selected_device,
            grant_identity,
        })
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
            submitted_capability_evidence: self.submitted_capability_evidence.clone(),
            selected_device,
            grant_identity,
        }
    }
}
