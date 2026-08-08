//! Device-scoped allocator domains for GPU-resident cache subdomains.
//!
//! A cache domain owns exactly one private CUDA async allocator plus one
//! hard-cap accounting adaptor. Cache implementations obtain independent
//! runtime leases from that domain, so source and graph caches can use their
//! own streams while their retained allocations share one physical allocation
//! limit and one live byte counter. Per-operation scopes add child accounting
//! adaptors and their own stream-to-resource routes without retaining streams
//! after their scopes have been destroyed.

use std::{
    fmt,
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{CudaStream, DeviceId, Runtime, device_guard::DeviceGuard};
use crate::core::{
    error::{Error, ErrorCode, Result},
    memory::{
        AllocatorEventRead, AllocatorEventRingBounds, DeviceMemoryResource, DeviceMemoryResourceUsage,
    },
};

static NEXT_CACHE_ALLOCATION_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CACHE_ALLOCATION_SCOPE_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity for one device-scoped cache allocation domain.
///
/// The id is process-local and distinct from a CUDA device ordinal. It lets
/// independently implemented cache subdomains prove that they lease the same
/// physical allocation domain without comparing raw allocator handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheAllocationDomainId(NonZeroU64);

impl CacheAllocationDomainId {
    fn next() -> Result<Self> {
        let value = NEXT_CACHE_ALLOCATION_DOMAIN_ID.fetch_add(1, Ordering::Relaxed);
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "runtime::cache_allocation::CacheAllocationDomainId::next",
                "cache allocation domain identity space was exhausted",
            )
            .with_detail("domain_id", value.to_string())
        })
    }

    /// Return the stable non-zero numeric identity.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for CacheAllocationDomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable identity for one cache allocation scope instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheAllocationScopeId(NonZeroU64);

impl CacheAllocationScopeId {
    fn next() -> Result<Self> {
        let value = NEXT_CACHE_ALLOCATION_SCOPE_ID.fetch_add(1, Ordering::Relaxed);
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "runtime::cache_allocation::CacheAllocationScopeId::next",
                "cache allocation scope identity space was exhausted",
            )
            .with_detail("scope_id", value.to_string())
        })
    }

    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for CacheAllocationScopeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Complete identity for one cache allocation scope instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheAllocationScopeIdentity {
    pub device: DeviceId,
    pub domain_id: CacheAllocationDomainId,
    pub scope_id: CacheAllocationScopeId,
    pub generation: u64,
}

#[derive(Debug)]
struct CacheAllocationDomainInner {
    id: CacheAllocationDomainId,
    device: DeviceId,
    capacity_bytes: u64,
    capacity_bytes_usize: usize,
    memory_resource: DeviceMemoryResource,
}

/// One private, device-scoped allocation domain for GPU-resident caches.
///
/// The configured capacity is a hard allocation ceiling, not an eager pool
/// reservation. The domain creates no physical allocation until a lease uses
/// it. Every lease uses the same tracked memory resource, so
/// [`Self::usage`] is the authoritative aggregate for all source and graph
/// cache allocations made through those leases.
///
/// This type deliberately does not accept a caller-supplied reservation id:
/// cache subdomains share the domain's allocator rather than minting private
/// stream-reservation ranges.
#[derive(Clone, Debug)]
pub struct CacheAllocationDomain {
    inner: Arc<CacheAllocationDomainInner>,
}

impl CacheAllocationDomain {
    /// Create a cache allocation domain on `device` with a hard live-byte cap.
    ///
    /// A zero cap means resident caching is disabled and is rejected before
    /// any CUDA resource is created. This preserves a disabled-by-default
    /// cache profile without creating an otherwise unused allocator domain.
    pub fn new(device: DeviceId, capacity_bytes: u64) -> Result<Self> {
        let capacity_bytes_usize = usize::try_from(capacity_bytes).map_err(|_| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "runtime::cache_allocation::CacheAllocationDomain::new",
                "cache allocation-domain capacity does not fit the platform allocation size",
            )
            .with_detail("capacity_bytes", capacity_bytes.to_string())
        })?;
        if capacity_bytes_usize == 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "runtime::cache_allocation::CacheAllocationDomain::new",
                "a zero cache capacity cannot create a cache allocation domain",
            )
            .with_detail("capacity_bytes", capacity_bytes.to_string())
            .with_detail("minimum_capacity_bytes", "1"));
        }

        // The async resource owns a dedicated physical CUDA pool. The tracked
        // adaptor retains the native upstream lifetime, and its zero release
        // threshold keeps bytes from becoming a permanent reservation once a
        // cache entry has been destroyed.
        let async_pool = DeviceMemoryResource::cuda_async_on(device, None, Some(0))?;
        let memory_resource = async_pool.budget_tracking_limit(capacity_bytes_usize)?;

        Ok(Self {
            inner: Arc::new(CacheAllocationDomainInner {
                id: CacheAllocationDomainId::next()?,
                device,
                capacity_bytes,
                capacity_bytes_usize,
                memory_resource,
            }),
        })
    }

    /// Stable identity shared by every lease of this allocation domain.
    pub fn id(&self) -> CacheAllocationDomainId {
        self.inner.id
    }

    /// CUDA device on which every lease and allocation in this domain resides.
    pub fn device(&self) -> DeviceId {
        self.inner.device
    }

    /// Immutable hard live-byte allocation cap for this domain.
    pub fn capacity_bytes(&self) -> u64 {
        self.inner.capacity_bytes
    }

    /// Return the shared memory resource used by every cache lease.
    ///
    /// Prefer [`Self::lease_runtime`] for cache work. This accessor exists for
    /// native consumers that must receive an explicit memory-resource binding
    /// rather than a `cudf` runtime.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        &self.inner.memory_resource
    }

    /// Create a runtime lease with an independent CUDA stream in this domain.
    ///
    /// The runtime retains the domain's memory resource, and the returned
    /// lease retains the domain identity for cache diagnostics. Different
    /// leases may submit on different streams, but their retained allocations
    /// are charged to the same physical domain and hard cap.
    pub fn lease_runtime(&self) -> Result<CacheAllocationLease> {
        let runtime = Runtime::builder()
            .with_device(self.device())
            .with_memory_resource(self.inner.memory_resource.clone())
            .build()?;
        Ok(CacheAllocationLease { domain: self.clone(), runtime })
    }

    /// Create one independently tracked allocation scope within this domain.
    ///
    /// The scope wraps this domain's already hard-capped memory resource in a
    /// child budget-tracking resource. Its [`CacheAllocationScope::usage`]
    /// therefore reports only allocations routed through that scope's runtime,
    /// while the parent domain remains the one shared physical hard cap for
    /// every scope and lease. A scope is not a reservation or a partition of
    /// the cache capacity: concurrent scopes may each attempt to use the full
    /// capacity, but their aggregate cannot exceed this domain's cap.
    ///
    /// Each scope owns a distinct CUDA stream so native current-resource
    /// routing remains unique for each retained graph. The child resource's
    /// private retirement registry owns that stream only until the scope's
    /// runtime and graph outputs have been torn down, so insertion churn cannot
    /// grow the domain's retirement-stream registry or fence unrelated scopes.
    pub fn allocation_scope(&self) -> Result<CacheAllocationScope> {
        let scope_stream = {
            let _guard = DeviceGuard::new(self.device())?;
            CudaStream::new()?
        };
        let memory_resource = self
            .inner
            .memory_resource
            .budget_tracking_limit_for_scope(self.inner.capacity_bytes_usize, &scope_stream)?;
        let runtime = Runtime::builder()
            .with_device(self.device())
            .with_stream(scope_stream)
            .with_memory_resource(memory_resource.clone())
            .build()?;
        let scope_id = CacheAllocationScopeId::next()?;
        let generation = scope_id.get();
        Ok(CacheAllocationScope { runtime, memory_resource, domain: self.clone(), scope_id, generation })
    }

    /// Snapshot aggregate live-allocation accounting for every lease.
    pub fn usage(&self) -> Result<DeviceMemoryResourceUsage> {
        self.inner.memory_resource.budget_tracking_usage()
    }
}

/// A cache subdomain's runtime lease from [`CacheAllocationDomain`].
///
/// Holding the lease keeps the shared allocation domain alive through cache
/// entry teardown. It has no independent byte accounting or allocator: use
/// [`Self::domain`] to inspect the one shared domain.
#[derive(Debug)]
pub struct CacheAllocationLease {
    domain: CacheAllocationDomain,
    runtime: Runtime,
}

impl CacheAllocationLease {
    /// Shared cache allocation domain retained by this lease.
    pub fn domain(&self) -> &CacheAllocationDomain {
        &self.domain
    }

    /// Stable identity of the shared cache allocation domain.
    pub fn domain_id(&self) -> CacheAllocationDomainId {
        self.domain.id()
    }

    /// Runtime whose output allocations are charged to the shared domain.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }
}

/// One independently tracked cache-allocation operation within a shared domain.
///
/// A scope owns a child budget-tracking memory resource and a runtime bound to
/// it on one unique CUDA stream. The child resource's private retirement
/// registry fences and releases that stream during scope teardown without
/// touching sibling cache streams. Retain the scope for as long as outputs
/// allocated by its runtime can remain live: this preserves both the exact
/// operation accounting and the parent domain's device/provenance lifetime.
///
/// Fields are ordered so the child resource is destroyed before the retained
/// parent domain. The native child also retains its upstream lifetime, but the
/// explicit Rust ownership makes the scope's provenance and teardown ordering
/// clear at this API boundary.
#[derive(Debug)]
pub struct CacheAllocationScope {
    runtime: Runtime,
    memory_resource: DeviceMemoryResource,
    domain: CacheAllocationDomain,
    scope_id: CacheAllocationScopeId,
    generation: u64,
}

impl CacheAllocationScope {
    /// Complete identity for this scope instance.
    pub fn identity(&self) -> CacheAllocationScopeIdentity {
        CacheAllocationScopeIdentity {
            device: self.domain.device(),
            domain_id: self.domain.id(),
            scope_id: self.scope_id,
            generation: self.generation,
        }
    }

    /// Allocator event ring bounds for the scope-local ring (stub).
    pub fn allocator_event_ring_bounds(&self) -> Result<AllocatorEventRingBounds> {
        Ok(AllocatorEventRingBounds { capacity: 0, first_available_sequence: None, next_sequence: 0 })
    }

    /// Read one sequence from the scope-local ring (stub: always Pending).
    pub fn read_allocator_event(&self, _sequence: u64) -> Result<AllocatorEventRead> {
        Ok(AllocatorEventRead::Pending)
    }

    /// Synchronize the scope's stream (used as scope fence).
    pub fn synchronize(&self) -> Result<()> {
        self.runtime.synchronize()
    }

    /// Shared physical cache allocation domain retained by this scope.
    pub fn domain(&self) -> &CacheAllocationDomain {
        &self.domain
    }

    /// Stable identity of the shared cache allocation domain.
    pub fn domain_id(&self) -> CacheAllocationDomainId {
        self.domain.id()
    }

    /// CUDA device on which this scope's runtime and allocations reside.
    pub fn device(&self) -> DeviceId {
        self.domain.device()
    }

    /// Child budget-tracking resource used by this scope's runtime.
    ///
    /// Prefer [`Self::runtime`] for cache work. This accessor is for native
    /// consumers that require an explicit resource binding while retaining the
    /// scope's accounting and parent-domain provenance.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        &self.memory_resource
    }

    /// Runtime whose output allocations are charged to this scope.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Snapshot allocations routed through this scope only.
    ///
    /// The returned limit equals the parent domain capacity, but this counter
    /// never includes allocations from sibling scopes or ordinary runtime
    /// leases. Use [`CacheAllocationDomain::usage`] for the aggregate domain
    /// view.
    pub fn usage(&self) -> Result<DeviceMemoryResourceUsage> {
        self.memory_resource.budget_tracking_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_retirement_registration_is_private_and_released_with_the_scope() {
        let domain = CacheAllocationDomain::new(DeviceId::current().unwrap(), 8 * 1024 * 1024).unwrap();
        assert_eq!(domain.memory_resource().__debug_retirement_stream_count(), 0);

        for _ in 0..3 {
            let scope = domain.allocation_scope().unwrap();
            assert_eq!(scope.memory_resource().__debug_retirement_stream_count(), 1);
            assert_eq!(
                domain.memory_resource().__debug_retirement_stream_count(),
                0,
                "a scope stream must not become a permanent domain retirement registration"
            );

            let allocation = scope.runtime().device_buffer(1024).unwrap();
            scope.runtime().synchronize().unwrap();
            drop(allocation);
            drop(scope);
            assert_eq!(domain.memory_resource().__debug_retirement_stream_count(), 0);
        }
    }
}
