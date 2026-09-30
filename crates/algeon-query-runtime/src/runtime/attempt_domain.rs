//! Immutable hard allocator domain for one admitted attempt.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use cudf::{
    AttemptStreamProviderRegistration,
    runtime::{CudaStream, DeviceEntryGuard},
};
use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{MemoryGrant, gpu::GrantIdentity, memory_domain::create_grant_memory_domain};
use crate::native_error::cudf_query_error;

/// The authoritative query-memory domain of one immutable ledger grant.
#[derive(Debug, Clone)]
pub struct AttemptAllocationDomain {
    domain: Arc<cudf::QueryMemoryDomain>,
    grant_identity: GrantIdentity,
    next_member_id: Arc<AtomicU64>,
    member_limit_bytes: u64,
}

/// Native fork-stream ownership held for the full admitted-attempt lifetime.
///
/// Field order is load-bearing: unregister the native route before detaching
/// its domain members or releasing the CUDA stream owners.
#[derive(Debug)]
#[doc(hidden)]
pub struct AttemptStreamProviderRoute {
    registration: cudf::AttemptStreamProviderRegistration,
    _member_leases: Vec<cudf::StreamReservationLease>,
    _fork_streams: Vec<CudaStream>,
}

impl AttemptStreamProviderRoute {
    /// Maximum execution width this attempt's auxiliary fork pool can back:
    /// the route's `S_d`, excluding the parent's reserved fork.
    ///
    /// The pool is fixed at admission and holds exactly as many auxiliary
    /// forks as the attempt has auxiliary permits, and every alias -- lane,
    /// source worker, or auxiliary fragment context alike -- holds one of them
    /// alone for its lifetime. Claims therefore never exceed permit holders,
    /// which never exceed `lane_capacity()`, so a caller holding a fresh
    /// permit always finds a free fork and no two callers are ever served the
    /// same stream. A run of `k` lanes leaves `lane_capacity() - k` forks for
    /// the other permit holders of the same attempt.
    pub fn lane_capacity(&self) -> usize {
        self.registration.auxiliary_count()
    }

    /// Process-unique identity of the fork reserved for this route's parent
    /// stream. No alias can ever be served it, so it is disjoint from every
    /// fork a work lease holds.
    #[doc(hidden)]
    pub fn reserved_root_fork_identity(&self) -> usize {
        self._fork_streams.last().expect("a registered route owns at least a reserved root fork").identity()
    }

    /// Process-unique identity of auxiliary fork `index` of this route.
    #[doc(hidden)]
    pub fn auxiliary_fork_identity(&self, index: usize) -> Option<usize> {
        (index < self.lane_capacity()).then(|| self._fork_streams[index].identity())
    }

    /// Attaches `stream` as a fork parent alias of this route, holding one
    /// auxiliary fork alone.
    ///
    /// The route claims an auxiliary fork no live alias holds and confines
    /// forks issued from `stream`, and from the fork it is served, to it. The
    /// claim lives with the alias, so it is released when the alias detaches.
    /// It fails typed when every auxiliary fork is claimed, which a caller
    /// holding a fresh auxiliary permit never sees.
    pub(super) fn attach(&self, stream: &CudaStream) -> Result<cudf::AttemptStreamProviderAlias> {
        self.registration.attach(stream).map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_stream_provider.attach",
                "failed to attach an attempt member stream to the fork-stream provider",
                error,
            )
        })
    }

    pub(super) fn synchronize_auxiliary_for_teardown(&self) -> Result<()> {
        self.registration.synchronize_auxiliary_for_teardown().map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_stream_provider.teardown",
                "failed to drain an attempt auxiliary stream",
                error,
            )
        })
    }
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
                "query_engine.runtime.attempt_allocation_domain.create",
                "attempt domain runtime device does not match immutable grant device",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("grant_device_ordinal", grant.device_ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        let domain = create_grant_memory_domain(runtime, grant)?;
        let usage = domain.usage().map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_allocation_domain.create",
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
                "query_engine.runtime.attempt_allocation_domain.create",
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
    pub(crate) fn create_member(&self, stream: &CudaStream) -> Result<cudf::StreamReservationLease> {
        if stream.device().ordinal() != self.device_ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.runtime.attempt_allocation_domain.member",
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
                "query_engine.runtime.attempt_allocation_domain.member",
                "attempt member limit exceeds platform usize range",
            )
            .with_fact("domain_id", self.domain_id())
            .with_fact("member_limit_bytes", self.member_limit_bytes)
        })?;
        self.domain.reserve_stream(stream, member_id, limit).map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_allocation_domain.member",
                "failed to bind stream as an attempt domain member",
                error,
            )
            .with_fact("domain_id", self.domain_id())
            .with_fact("member_id", member_id)
        })
    }

    /// Create and register the bounded native fork stream set used by
    /// libcudf's `fork_streams` provider.
    ///
    /// `auxiliary_capacity` forks back the attempt's auxiliary permit holders,
    /// one each; one further fork is registered last and reserved for the root
    /// parent stream, so the root never shares a fork with a lane, a source
    /// worker, or an auxiliary fragment context.
    pub(super) fn create_stream_provider_route(
        &self,
        parent: &CudaStream,
        auxiliary_capacity: usize,
    ) -> Result<AttemptStreamProviderRoute> {
        let operation = "query_engine.runtime.attempt_stream_provider.create";
        if parent.device().ordinal() != self.device_ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "attempt stream-provider parent must live on the attempt device",
            )
            .with_fact("domain_device_ordinal", self.device_ordinal())
            .with_fact("stream_device_ordinal", parent.device().ordinal()));
        }
        // Auxiliary forks plus the parent's reserved fork, which the native
        // route takes as the last registered entry.
        let capacity = auxiliary_capacity.max(1) + 1;
        let _entry = DeviceEntryGuard::enter(parent.device(), operation).map_err(|error| {
            cudf_query_error(operation, "failed to enter the attempt stream-provider device", error)
        })?;
        let mut fork_streams = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            fork_streams.push(
                CudaStream::new().map_err(|error| {
                    cudf_query_error(operation, "failed to create an attempt auxiliary stream", error)
                })?,
            );
        }
        let mut member_leases = Vec::with_capacity(fork_streams.len());
        for stream in &fork_streams {
            member_leases.push(self.create_member(stream)?);
        }
        let registration = AttemptStreamProviderRegistration::register(parent, &fork_streams)
            .map_err(|error| cudf_query_error(operation, "failed to register attempt fork streams", error))?;
        Ok(AttemptStreamProviderRoute { registration, _member_leases: member_leases, _fork_streams: fork_streams })
    }

    pub(crate) fn as_cudf(&self) -> &Arc<cudf::QueryMemoryDomain> {
        &self.domain
    }

    pub(crate) const fn domain_id(&self) -> u64 {
        self.grant_identity.generation
    }

    pub const fn grant_identity(&self) -> GrantIdentity {
        self.grant_identity
    }

    /// Immutable allocation cap copied from the admitted memory grant.
    pub(crate) const fn cap_bytes(&self) -> u64 {
        self.member_limit_bytes
    }

    pub fn device_ordinal(&self) -> usize {
        self.domain.device().ordinal()
    }

    pub(crate) fn usage(&self) -> Result<cudf::QueryMemoryDomainUsage> {
        self.domain.usage().map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_allocation_domain.usage",
                "failed to sample attempt allocation domain usage",
                error,
            )
            .with_fact("domain_id", self.domain_id())
        })
    }

    /// Permanently closes the producer domain to allocation after all attempt
    /// streams have drained and returns its measured resident charge.
    pub(crate) fn freeze(&self) -> Result<cudf::QueryMemoryDomainUsage> {
        self.domain.freeze().map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_allocation_domain.freeze",
                "failed to freeze the attempt allocation domain",
                error,
            )
            .with_fact("domain_id", self.domain_id())
        })
    }

    /// Snapshots the allocations this domain's registry still owns.
    pub(crate) fn live_allocations(&self) -> Result<Vec<cudf::QueryMemoryDomainAllocation>> {
        self.domain.live_allocations().map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_allocation_domain.live_allocations",
                "failed to read the attempt allocation domain registry",
                error,
            )
            .with_fact("domain_id", self.domain_id())
        })
    }

    pub(crate) fn pool_usage(&self) -> Result<cudf::DeviceMemoryPoolUsage> {
        self.domain.pool_usage().map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.attempt_allocation_domain.pool_usage",
                "failed to sample attempt allocation domain pool usage",
                error,
            )
            .with_fact("domain_id", self.domain_id())
        })
    }
}
