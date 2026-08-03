//! Device-scoped allocator domains for GPU-resident cache subdomains.
//!
//! A cache domain owns exactly one private CUDA async allocator plus one
//! accounting adaptor. Cache implementations obtain independent runtime leases
//! from that domain, so source and graph caches can use their own streams while
//! their retained allocations share one physical allocation limit and one live
//! byte counter.

use std::{
    fmt,
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{DeviceId, Runtime};
use crate::core::{
    error::{Error, ErrorCode, Result},
    memory::{DeviceMemoryResource, DeviceMemoryResourceUsage},
};

static NEXT_CACHE_ALLOCATION_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);

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

#[derive(Debug)]
struct CacheAllocationDomainInner {
    id: CacheAllocationDomainId,
    device: DeviceId,
    capacity_bytes: u64,
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
