//! Grant-backed allocator domains.
//!
//! Attempt grants are immutable, so allocator domains are created once with
//! `grant_bytes == ceiling_bytes == grant` and never resized.  Construction is
//! intentionally one-shot: the admission envelope owns the hard domain and
//! passes that value to every derived context.  In particular, this module
//! does not keep process-global weak registries or identity-based creation
//! locks to rediscover a resource after ownership has been established.

use std::sync::Arc;

use nexus_query_planning::error::{QueryError, QueryErrorCode, Result};

use super::MemoryGrant;

/// Creates the authoritative hard allocation domain for one admitted attempt.
///
/// The domain grant equals its ceiling equals the immutable admission grant;
/// Nexus never calls `resize_grant` on this domain.  The caller must retain
/// the returned value and propagate it through the admission envelope rather
/// than attempting another lookup by grant identity.
pub(super) fn create_grant_memory_domain(
    runtime: &cudf::Runtime,
    grant: &MemoryGrant,
) -> Result<Arc<cudf::QueryMemoryDomain>> {
    let grant_bytes = grant_budget_usize(grant, "nexus_query_engine.runtime.grant_memory_domain")?;
    let manager = runtime.device_memory_manager().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_memory_domain",
            "failed to resolve device memory manager for grant query memory domain",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
    })?;
    let domain = manager
        .query_memory_domain(grant.grant_id(), cudf::QueryMemoryDomainOptions::new(grant_bytes))
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.grant_memory_domain",
                "failed to create grant query memory domain",
                error,
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("admission_request_id", grant.admission_request_id())
            .with_fact("budget_bytes", grant.budget_bytes())
        })?;
    Ok(Arc::new(domain))
}

fn grant_budget_usize(grant: &MemoryGrant, operation: &'static str) -> Result<usize> {
    usize::try_from(grant.budget_bytes()).map_err(|_error| {
        QueryError::new(QueryErrorCode::InvalidPlan, operation, "memory grant budget exceeds platform usize range")
            .with_fact("grant_id", grant.grant_id())
            .with_fact("admission_request_id", grant.admission_request_id())
            .with_fact("budget_bytes", grant.budget_bytes())
    })
}
