//! Immutable hard allocator domain for one admitted attempt.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::{MemoryGrant, gpu::GrantIdentity, memory_domain::create_grant_memory_domain};
use crate::error::{QueryError, QueryErrorCode, Result};

/// The authoritative query-memory domain of one immutable ledger grant.
#[derive(Debug, Clone)]
pub struct AttemptAllocationDomain {
    domain: Arc<cudf::QueryMemoryDomain>,
    grant_identity: GrantIdentity,
    next_member_id: Arc<AtomicU64>,
    member_limit_bytes: u64,
}

impl AttemptAllocationDomain {
    /// Creates the domain with `grant == ceiling == immutable ledger grant`.
    ///
    /// Construction is restricted to the runtime's admission envelope.  The
    /// envelope is the sole owner of this hard resource; contexts receive
    /// clones of that owned value and member leases, never a rediscovered
    /// domain keyed by grant identity.
    pub(super) fn create(runtime: &cudf::Runtime, grant: &MemoryGrant) -> Result<Self> {
        if runtime.device().ordinal() != grant.device_ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.attempt_allocation_domain.create",
                "attempt domain runtime device does not match immutable grant device",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("grant_device_ordinal", grant.device_ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        let domain = create_grant_memory_domain(runtime, grant)?;
        let usage = domain.usage().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.attempt_allocation_domain.create",
                "failed to verify the created attempt allocation domain",
                error,
            )
            .with_fact("grant_id", grant.grant_id())
        })?;
        if usage.domain_id != grant.identity().generation
            || usage.allocation_limit_bytes != grant.budget_bytes()
            || usage.ceiling_bytes != grant.budget_bytes()
        {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                "nexus_query_engine.runtime.attempt_allocation_domain.create",
                "attempt allocation domain does not match immutable admission grant",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("domain_id", usage.domain_id)
            .with_fact("grant_bytes", grant.budget_bytes())
            .with_fact("allocation_limit_bytes", usage.allocation_limit_bytes)
            .with_fact("ceiling_bytes", usage.ceiling_bytes));
        }
        Ok(Self {
            domain,
            grant_identity: grant.identity(),
            next_member_id: Arc::new(AtomicU64::new(1)),
            member_limit_bytes: grant.budget_bytes(),
        })
    }

    /// Binds `stream` as a new member of this attempt's domain.
    ///
    /// Callers own the lease lifetime: one member exists per execution
    /// stream for the attempt and the shared lease is cloned into per-stage
    /// budget tokens, which never attach a second member to the stream. The
    /// member detaches only when every clone drops.
    pub fn create_member(&self, stream: &cudf::runtime::CudaStream) -> Result<cudf::StreamReservationLease> {
        if stream.device().ordinal() != self.device_ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.attempt_allocation_domain.member",
                "attempt domain member stream must live on the attempt device",
            )
            .with_fact("domain_id", self.domain_id())
            .with_fact("domain_device_ordinal", self.device_ordinal())
            .with_fact("stream_device_ordinal", stream.device().ordinal()));
        }
        let member_id = self.next_member_id.fetch_add(1, Ordering::Relaxed);
        let limit = usize::try_from(self.member_limit_bytes).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.attempt_allocation_domain.member",
                "attempt member limit exceeds platform usize range",
            )
            .with_fact("domain_id", self.domain_id())
            .with_fact("member_limit_bytes", self.member_limit_bytes)
        })?;
        self.domain.reserve_stream(stream, member_id, limit).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.attempt_allocation_domain.member",
                "failed to bind stream as an attempt domain member",
                error,
            )
            .with_fact("domain_id", self.domain_id())
            .with_fact("member_id", member_id)
        })
    }

    pub fn as_cudf(&self) -> &Arc<cudf::QueryMemoryDomain> {
        &self.domain
    }

    pub const fn domain_id(&self) -> u64 {
        self.grant_identity.generation
    }

    pub const fn grant_identity(&self) -> GrantIdentity {
        self.grant_identity
    }

    /// Immutable allocation cap copied from the admitted memory grant.
    pub const fn cap_bytes(&self) -> u64 {
        self.member_limit_bytes
    }

    pub fn device_ordinal(&self) -> usize {
        self.domain.device().ordinal()
    }

    pub fn usage(&self) -> Result<cudf::QueryMemoryDomainUsage> {
        self.domain.usage().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.attempt_allocation_domain.usage",
                "failed to sample attempt allocation domain usage",
                error,
            )
            .with_fact("domain_id", self.domain_id())
        })
    }

    pub(crate) fn pool_usage(&self) -> Result<cudf::DeviceMemoryPoolUsage> {
        self.domain.pool_usage().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.attempt_allocation_domain.pool_usage",
                "failed to sample attempt allocation domain pool usage",
                error,
            )
            .with_fact("domain_id", self.domain_id())
        })
    }
}
