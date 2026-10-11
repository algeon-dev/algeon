//! Usage-counter snapshots for resources and query domains.

use crate::DeviceId;

/// Usage counters reported by a production budget-tracking memory resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryResourceUsage {
    pub allocation_limit_bytes: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub total_bytes: u64,
    pub current_aligned_bytes: u64,
    pub peak_aligned_bytes: u64,
    pub total_allocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub failed_allocation_count: u64,
    pub largest_failed_allocation_bytes: u64,
}

/// Physical CUDA async-pool counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryPoolUsage {
    pub reserved_current_bytes: u64,
    pub reserved_high_bytes: u64,
    pub used_current_bytes: u64,
    pub used_high_bytes: u64,
}

impl From<cudf_sys::cudf_memory_resource_usage_t> for DeviceMemoryResourceUsage {
    fn from(value: cudf_sys::cudf_memory_resource_usage_t) -> Self {
        Self {
            allocation_limit_bytes: value.allocation_limit_bytes,
            current_bytes: value.current_bytes,
            peak_bytes: value.peak_bytes,
            total_bytes: value.total_bytes,
            current_aligned_bytes: value.current_aligned_bytes,
            peak_aligned_bytes: value.peak_aligned_bytes,
            total_allocations: value.total_allocations,
            current_allocations: value.current_allocations,
            peak_allocations: value.peak_allocations,
            failed_allocation_count: value.failed_allocation_count,
            largest_failed_allocation_bytes: value.largest_failed_allocation_bytes,
        }
    }
}

impl From<cudf_sys::cudf_memory_pool_usage_t> for DeviceMemoryPoolUsage {
    fn from(value: cudf_sys::cudf_memory_pool_usage_t) -> Self {
        Self {
            reserved_current_bytes: value.reserved_current_bytes,
            reserved_high_bytes: value.reserved_high_bytes,
            used_current_bytes: value.used_current_bytes,
            used_high_bytes: value.used_high_bytes,
        }
    }
}

/// Usage counters for a query-domain stream member.
///
/// These counters are recorded by the device-level reservation-aware memory
/// manager. `current_*` counters include retained allocations that were
/// created while the reservation was attached, even after the reservation has
/// been detached from its stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservationUsage {
    /// Native identity of this query-domain stream member.
    pub reservation_id: u64,
    /// Whether the member remains attached to its stream; detached allocations can remain retained.
    pub attached: bool,
    /// Allocations made through a query-domain member's direct resource.
    pub direct_total_allocations: u64,
    /// Allocations routed from the device-current resource into a query-domain
    /// member.
    pub routed_total_allocations: u64,
    /// Enforced live allocation limit in bytes, charged using aligned allocation sizes.
    pub allocation_limit_bytes: u64,
    /// Requested bytes in allocations still retained at this snapshot.
    pub current_bytes: u64,
    /// Largest observed sum of requested bytes in simultaneously retained allocations.
    pub peak_bytes: u64,
    /// Cumulative requested bytes across successful allocations, including allocations later released.
    pub total_bytes: u64,
    /// Live allocation charge in bytes after alignment.
    pub current_aligned_bytes: u64,
    /// Largest observed live allocation charge in bytes after alignment.
    pub peak_aligned_bytes: u64,
    /// Cumulative count of successful allocations, including allocations later released.
    pub total_allocations: u64,
    /// Number of allocations still retained at this snapshot.
    pub current_allocations: u64,
    /// Largest observed number of simultaneously retained allocations.
    pub peak_allocations: u64,
    /// Number of allocation requests rejected by a limit or the upstream allocator.
    pub failed_allocation_count: u64,
    /// Largest failed allocation request in bytes; zero when no allocation failure was recorded.
    pub largest_failed_allocation_bytes: u64,
}

/// Stable native identity for one query-memory-domain instance.
///
/// `generation` is distinct from the caller-supplied domain id and changes for
/// every native domain creation. Consumers must retain both values when
/// correlating asynchronous allocator events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryMemoryDomainIdentity {
    pub device: DeviceId,
    pub domain_id: u64,
    pub generation: u64,
}

/// Usage counters for an opt-in query memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryMemoryDomainUsage {
    /// Caller-supplied query-memory-domain identity.
    pub domain_id: u64,
    /// Whether the domain remains active for allocation.
    pub active: bool,
    /// Number of stream members currently attached to the domain.
    pub active_members: u64,
    /// Immutable logical ceiling declared when this domain was created, in bytes.
    pub ceiling_bytes: u64,
    /// Enforced live allocation limit in bytes, charged using aligned allocation sizes.
    pub allocation_limit_bytes: u64,
    /// Requested bytes in allocations still retained at this snapshot.
    pub current_bytes: u64,
    /// Largest observed sum of requested bytes in simultaneously retained allocations.
    pub peak_bytes: u64,
    /// Cumulative requested bytes across successful allocations, including allocations later released.
    pub total_bytes: u64,
    /// Live allocation charge in bytes after alignment.
    pub current_aligned_bytes: u64,
    /// Largest observed live allocation charge in bytes after alignment.
    pub peak_aligned_bytes: u64,
    /// Cumulative count of successful allocations, including allocations later released.
    pub total_allocations: u64,
    /// Number of allocations still retained at this snapshot.
    pub current_allocations: u64,
    /// Largest observed number of simultaneously retained allocations.
    pub peak_allocations: u64,
    /// Number of allocation requests rejected by a limit or the upstream allocator.
    pub failed_allocation_count: u64,
    /// Largest failed allocation request in bytes; zero when no allocation failure was recorded.
    pub largest_failed_allocation_bytes: u64,
}

impl From<cudf_sys::cudf_memory_resource_usage_t> for ReservationUsage {
    fn from(value: cudf_sys::cudf_memory_resource_usage_t) -> Self {
        Self {
            reservation_id: value.reserved[0],
            attached: value.reserved[1] != 0,
            direct_total_allocations: value.reserved[2],
            routed_total_allocations: value.reserved[3],
            allocation_limit_bytes: value.allocation_limit_bytes,
            current_bytes: value.current_bytes,
            peak_bytes: value.peak_bytes,
            total_bytes: value.total_bytes,
            current_aligned_bytes: value.current_aligned_bytes,
            peak_aligned_bytes: value.peak_aligned_bytes,
            total_allocations: value.total_allocations,
            current_allocations: value.current_allocations,
            peak_allocations: value.peak_allocations,
            failed_allocation_count: value.failed_allocation_count,
            largest_failed_allocation_bytes: value.largest_failed_allocation_bytes,
        }
    }
}

impl From<cudf_sys::cudf_memory_resource_usage_t> for QueryMemoryDomainUsage {
    fn from(value: cudf_sys::cudf_memory_resource_usage_t) -> Self {
        Self {
            domain_id: value.reserved[0],
            active: value.reserved[1] != 0,
            active_members: value.reserved[2],
            ceiling_bytes: value.reserved[3],
            allocation_limit_bytes: value.allocation_limit_bytes,
            current_bytes: value.current_bytes,
            peak_bytes: value.peak_bytes,
            total_bytes: value.total_bytes,
            current_aligned_bytes: value.current_aligned_bytes,
            peak_aligned_bytes: value.peak_aligned_bytes,
            total_allocations: value.total_allocations,
            current_allocations: value.current_allocations,
            peak_allocations: value.peak_allocations,
            failed_allocation_count: value.failed_allocation_count,
            largest_failed_allocation_bytes: value.largest_failed_allocation_bytes,
        }
    }
}

impl From<ReservationUsage> for DeviceMemoryResourceUsage {
    fn from(value: ReservationUsage) -> Self {
        Self {
            allocation_limit_bytes: value.allocation_limit_bytes,
            current_bytes: value.current_bytes,
            peak_bytes: value.peak_bytes,
            total_bytes: value.total_bytes,
            current_aligned_bytes: value.current_aligned_bytes,
            peak_aligned_bytes: value.peak_aligned_bytes,
            total_allocations: value.total_allocations,
            current_allocations: value.current_allocations,
            peak_allocations: value.peak_allocations,
            failed_allocation_count: value.failed_allocation_count,
            largest_failed_allocation_bytes: value.largest_failed_allocation_bytes,
        }
    }
}

/// One allocation a query memory domain's registry still owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryMemoryDomainAllocation {
    /// Base device pointer the allocator registered.
    pub device_pointer: usize,
    /// Requested bytes for this allocation.
    pub bytes: u64,
    /// Charged bytes after alignment.
    pub aligned_bytes: u64,
    /// Alignment requested for this allocation.
    pub alignment: u64,
}

impl From<cudf_sys::cudf_query_memory_domain_allocation_t> for QueryMemoryDomainAllocation {
    fn from(value: cudf_sys::cudf_query_memory_domain_allocation_t) -> Self {
        Self {
            device_pointer: value.device_pointer as usize,
            bytes: value.bytes as u64,
            aligned_bytes: value.aligned_bytes as u64,
            alignment: value.alignment as u64,
        }
    }
}
