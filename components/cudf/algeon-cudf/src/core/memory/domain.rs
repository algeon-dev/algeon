//! Stream reservations, reservation arenas, and query memory domains.

use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{
    manager::{CurrentResourceRouteLease, DeviceMemoryManager},
    resource::{DeviceMemoryResource, DeviceMemoryResourceHandle, record_native_teardown_failure},
    usage::{
        DeviceMemoryPoolUsage, QueryMemoryDomainAllocation, QueryMemoryDomainIdentity, QueryMemoryDomainUsage,
        ReservationArenaUsage, ReservationUsage,
    },
};
use crate::{
    DeviceId,
    core::error::{self, Result},
    runtime::{self, device_guard::DeviceGuard, stream::CudaStream},
};

/// A live reservation attached to a CUDA stream.
///
/// Dropping this value detaches the reservation from the stream. Allocations
/// made while it was attached remain accounted to the reservation identity
/// until those retained objects are destroyed.
#[derive(Debug, Clone)]
pub struct StreamReservationLease {
    // Field order is load-bearing: Rust drops fields in declaration order,
    // and the reservation must detach while the stream (whose raw value keys
    // the native attachment map) and the manager are still alive. If the
    // stream dropped first, a recycled raw stream value could inherit this
    // dying reservation's accounting.
    //
    // The handle is shared: clones of this lease keep the member attached,
    // and the member detaches only when the final clone drops. This lets an
    // attempt hold one member per stream for its whole lifetime while
    // handing shared views to per-stage budget tokens.
    pub(super) reservation: Arc<ReservationHandle>,
    pub(super) resource: Option<DeviceMemoryResource>,
    pub(super) _query_domain: Option<QueryMemoryDomain>,
    pub(super) arena: Option<ReservationArena>,
    pub(super) stream: CudaStream,
    pub(super) manager: DeviceMemoryManager,
}

/// Query-scoped parent reservation shared by stream-bound member leases.
///
/// Dropping the final handle prevents new members from attaching. Existing
/// allocations made by member leases remain accounted natively until those GPU
/// allocations are released.
#[derive(Clone, Debug)]
pub struct ReservationArena {
    pub(super) handle: Arc<ReservationArenaHandle>,
    pub(super) manager: DeviceMemoryManager,
}

/// Query memory domain sharing its device manager's CUDA async pool.
#[derive(Clone, Debug)]
pub struct QueryMemoryDomain {
    pub(super) handle: Arc<QueryMemoryDomainHandle>,
    pub(super) manager: DeviceMemoryManager,
}

#[derive(Debug)]
pub(super) struct ReservationHandle {
    pub(super) raw: Option<NonNull<cudf_sys::cudf_reservation_handle_state>>,
    pub(super) device: DeviceId,
}

#[derive(Debug)]
pub(super) struct ReservationArenaHandle {
    pub(super) raw: Option<NonNull<cudf_sys::cudf_reservation_arena_handle_state>>,
    pub(super) device: DeviceId,
}

#[derive(Debug)]
pub(super) struct QueryMemoryDomainHandle {
    pub(super) raw: Option<NonNull<cudf_sys::cudf_query_memory_domain_handle_state>>,
    pub(super) device: DeviceId,
}

// SAFETY: native detach/usage paths are synchronized by the manager.
unsafe impl Send for ReservationHandle {}
unsafe impl Sync for ReservationHandle {}

// SAFETY: native arena usage/resize/destroy paths are synchronized by the manager.
unsafe impl Send for ReservationArenaHandle {}
unsafe impl Sync for ReservationArenaHandle {}

// SAFETY: native query-domain usage/resize/destroy paths are synchronized by the manager.
unsafe impl Send for QueryMemoryDomainHandle {}
unsafe impl Sync for QueryMemoryDomainHandle {}

impl ReservationArena {
    /// Manager that owns this arena.
    pub fn manager(&self) -> &DeviceMemoryManager {
        &self.manager
    }

    /// CUDA device owned by this arena's manager.
    pub fn device(&self) -> DeviceId {
        self.manager.device()
    }

    /// Attach a stream-bound member reservation to this arena.
    pub fn reserve_stream(
        &self,
        stream: &CudaStream,
        reservation_id: u64,
        limit_bytes: usize,
    ) -> Result<StreamReservationLease> {
        self.manager.reserve_stream_with_arena(stream, reservation_id, limit_bytes, Some(self))
    }

    /// Snapshot usage counters for this arena.
    pub fn usage(&self) -> Result<ReservationArenaUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_reservation_arena_usage(self.handle.raw(), &mut usage) },
            "memory::reservation_arena::usage",
        )?;
        Ok(usage.into())
    }

    /// Resize this arena's hard live-byte limit.
    ///
    /// Shrinking below current live usage is allowed. Future member allocations
    /// fail until enough memory is released or the arena is grown again.
    pub fn resize(&self, limit_bytes: usize) -> Result<()> {
        let _guard = DeviceGuard::new(self.device())?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_reservation_arena_resize(self.handle.raw(), limit_bytes) },
            "memory::reservation_arena::resize",
            [("device", self.device().ordinal().to_string()), ("limit_bytes", limit_bytes.to_string())],
        )
    }
}

impl QueryMemoryDomain {
    /// Manager that owns this query memory domain.
    pub fn manager(&self) -> &DeviceMemoryManager {
        &self.manager
    }

    /// CUDA device owned by this query memory domain.
    pub fn device(&self) -> DeviceId {
        self.manager.device()
    }

    /// Returns the native domain instance identity used by allocator events.
    ///
    /// The domain id alone is not sufficient for correlation because callers
    /// may create a later domain with the same logical id after teardown.
    pub fn identity(&self) -> Result<QueryMemoryDomainIdentity> {
        let _guard = DeviceGuard::new(self.device())?;
        // Native initializes every output field, including `struct_size`.
        let mut identity = unsafe { std::mem::zeroed::<cudf_sys::cudf_query_memory_domain_identity_t>() };
        error::check_status_in(
            unsafe { cudf_sys::cudf_query_memory_domain_identity(self.handle.raw(), &mut identity) },
            "memory::query_memory_domain::identity",
        )?;
        if identity.struct_size != core::mem::size_of::<cudf_sys::cudf_query_memory_domain_identity_t>() {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::query_memory_domain::identity",
                "native query memory domain identity returned an incompatible struct size",
            )
            .with_detail("struct_size", identity.struct_size.to_string()));
        }
        if identity.device_ordinal < 0 || identity.domain_id == 0 || identity.generation == 0 {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::query_memory_domain::identity",
                "native query memory domain returned an incomplete identity",
            )
            .with_detail("device_ordinal", identity.device_ordinal.to_string())
            .with_detail("domain_id", identity.domain_id.to_string())
            .with_detail("generation", identity.generation.to_string()));
        }
        let device = DeviceId::from_validated_ordinal_raw(identity.device_ordinal);
        if device != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::query_memory_domain::identity",
                "native query memory domain identity does not match its owning device",
            )
            .with_detail("domain_device", device.ordinal().to_string())
            .with_detail("owner_device", self.device().ordinal().to_string()));
        }
        Ok(QueryMemoryDomainIdentity { device, domain_id: identity.domain_id, generation: identity.generation })
    }

    /// Attach a stream to this query memory domain.
    pub fn reserve_stream(
        &self,
        stream: &CudaStream,
        reservation_id: u64,
        limit_bytes: usize,
    ) -> Result<StreamReservationLease> {
        let _guard = DeviceGuard::new(self.device())?;
        if stream.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::query_memory_domain::reserve_stream",
                "query memory domain requires stream and domain to be on the same CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(stream.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "stream_and_query_memory_domain")
            .with_detail("reason", "cross_device_query_memory_domain"));
        }

        let mut raw_reservation = ptr::null_mut();
        let mut raw_resource = ptr::null_mut();
        let status = unsafe {
            cudf_sys::cudf_query_memory_domain_attach_stream(
                self.handle.raw(),
                stream.as_raw(),
                reservation_id,
                limit_bytes,
                &mut raw_reservation,
                &mut raw_resource,
            )
        };
        // Adopt both native outputs before validating either one. If native
        // code ever returns a partial pair, the successfully-created handle is
        // still released on every error path.
        let reservation =
            NonNull::new(raw_reservation).map(|raw| ReservationHandle { raw: Some(raw), device: self.device() });
        let resource = NonNull::new(raw_resource).map(|raw| DeviceMemoryResource {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: Some(raw),
                device: self.device(),
                retirement_streams: None,
                _failure_callbacks: Vec::new(),
            }),
        });
        error::check_status_in_with_details(
            status,
            "memory::query_memory_domain::reserve_stream",
            [
                ("device", self.device().ordinal().to_string()),
                ("reservation_id", reservation_id.to_string()),
                ("limit_bytes", limit_bytes.to_string()),
            ],
        )?;
        let reservation = reservation.ok_or_else(|| {
            error::require_non_null(
                raw_reservation,
                "memory::query_memory_domain::reserve_stream",
                "stream reservation",
            )
            .unwrap_err()
        })?;
        let resource = resource.ok_or_else(|| {
            error::require_non_null(
                raw_resource,
                "memory::query_memory_domain::reserve_stream",
                "device memory resource",
            )
            .unwrap_err()
        })?;
        Ok(StreamReservationLease {
            manager: self.manager.clone(),
            stream: stream.clone(),
            arena: None,
            _query_domain: Some(self.clone()),
            resource: Some(resource),
            reservation: Arc::new(reservation),
        })
    }

    /// Snapshot usage counters for this query memory domain.
    pub fn usage(&self) -> Result<QueryMemoryDomainUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_query_memory_domain_usage(self.handle.raw(), &mut usage) },
            "memory::query_memory_domain::usage",
        )?;
        Ok(usage.into())
    }

    /// Read physical CUDA async-pool counters shared by this device manager.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_pool_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_query_memory_domain_pool_usage(self.handle.raw(), &mut usage) },
            "memory::query_memory_domain::pool_usage",
        )?;
        Ok(usage.into())
    }

    /// Resize this domain's live-byte grant.
    pub fn resize_grant(&self, grant_bytes: usize) -> Result<()> {
        let _guard = DeviceGuard::new(self.device())?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_query_memory_domain_resize_grant(self.handle.raw(), grant_bytes) },
            "memory::query_memory_domain::resize_grant",
            [("device", self.device().ordinal().to_string()), ("grant_bytes", grant_bytes.to_string())],
        )
    }

    /// Permanently closes this domain to new allocations and returns its
    /// measured live usage. Deallocation remains valid for resident owners.
    pub fn freeze(&self) -> Result<QueryMemoryDomainUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        error::check_status_in(
            unsafe { cudf_sys::cudf_query_memory_domain_freeze(self.handle.raw()) },
            "memory::query_memory_domain::freeze",
        )?;
        self.usage()
    }

    /// Snapshots every allocation this domain's registry still owns.
    ///
    /// The records are the allocator's own base pointers, so a caller can
    /// decide whether each surviving allocation belongs to a published output
    /// owner. Counter totals cannot make that distinction.
    pub fn live_allocations(&self) -> Result<Vec<QueryMemoryDomainAllocation>> {
        let _guard = DeviceGuard::new(self.device())?;
        // The registry can change between the count and the copy only while
        // the domain still accepts work; a frozen domain's live set is
        // stable, so one retry after a growth is enough for callers that
        // enumerate before freezing.
        let mut capacity = self.live_allocation_count()?;
        loop {
            let mut records = vec![
                cudf_sys::cudf_query_memory_domain_allocation_t {
                    struct_size: std::mem::size_of::<cudf_sys::cudf_query_memory_domain_allocation_t>(),
                    device_pointer: std::ptr::null_mut(),
                    bytes: 0,
                    aligned_bytes: 0,
                    alignment: 0,
                    reserved: [0; 2],
                };
                capacity
            ];
            let mut count = 0usize;
            error::check_status_in(
                unsafe {
                    cudf_sys::cudf_query_memory_domain_live_allocations(
                        self.handle.raw(),
                        records.as_mut_ptr(),
                        capacity,
                        &mut count,
                    )
                },
                "memory::query_memory_domain::live_allocations",
            )?;
            if count <= capacity {
                records.truncate(count);
                return Ok(records.into_iter().map(QueryMemoryDomainAllocation::from).collect());
            }
            capacity = count;
        }
    }

    fn live_allocation_count(&self) -> Result<usize> {
        let mut count = 0usize;
        error::check_status_in(
            unsafe {
                cudf_sys::cudf_query_memory_domain_live_allocations(
                    self.handle.raw(),
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            },
            "memory::query_memory_domain::live_allocation_count",
        )?;
        Ok(count)
    }
}

impl StreamReservationLease {
    /// Manager that owns the reservation-aware device resource.
    pub fn manager(&self) -> &DeviceMemoryManager {
        &self.manager
    }

    /// CUDA stream this reservation is attached to.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Device-level memory resource to place in explicit execution options.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.resource.as_ref().unwrap_or_else(|| self.manager.resource())
    }

    /// Parent reservation arena for this lease, when it is an arena member.
    pub fn reservation_arena(&self) -> Option<&ReservationArena> {
        self.arena.as_ref()
    }

    /// Snapshot usage counters for this reservation.
    pub fn usage(&self) -> Result<ReservationUsage> {
        let _guard = DeviceGuard::new(self.manager.device())?;
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_reservation_usage(self.reservation.raw(), &mut usage) },
            "memory::stream_reservation_lease::usage",
        )?;
        Ok(usage.into())
    }

    /// Resize this reservation's hard live-byte limit.
    ///
    /// Shrinking below current live usage is allowed. Future allocations fail
    /// until enough memory is released or the reservation is grown again.
    /// Query-memory-domain members enforce this independently of the parent
    /// domain grant.
    pub fn resize(&self, limit_bytes: usize) -> Result<()> {
        let _guard = DeviceGuard::new(self.manager.device())?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_reservation_resize(self.reservation.raw(), limit_bytes) },
            "memory::stream_reservation_lease::resize",
            [("device", self.manager.device().ordinal().to_string()), ("limit_bytes", limit_bytes.to_string())],
        )
    }

    /// Borrow this lease as per-operation execution overrides.
    pub fn execution_overrides(&self) -> crate::ExecutionOverrides<'_> {
        crate::ExecutionOverrides::new().with_stream(&self.stream).with_memory_resource(self.memory_resource())
    }
}

impl Drop for ReservationArenaHandle {
    fn drop(&mut self) {
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        runtime::drop_device_resource(&mut raw, self.device, "ReservationArenaHandle::drop", |owned| {
            let status = unsafe { cudf_sys::cudf_reservation_arena_destroy(owned) };
            if status != cudf_sys::CUDF_SUCCESS {
                record_native_teardown_failure(self.device, "ReservationArenaHandle::drop", status);
            }
        });
    }
}

impl Drop for QueryMemoryDomainHandle {
    fn drop(&mut self) {
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        runtime::drop_device_resource(&mut raw, self.device, "QueryMemoryDomainHandle::drop", |owned| {
            let status = unsafe { cudf_sys::cudf_query_memory_domain_destroy(owned) };
            if status != cudf_sys::CUDF_SUCCESS {
                record_native_teardown_failure(self.device, "QueryMemoryDomainHandle::drop", status);
            }
        });
    }
}

impl Drop for CurrentResourceRouteLease {
    fn drop(&mut self) {
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        runtime::drop_device_resource(&mut raw, self.device, "CurrentResourceRouteLease::drop", |owned| {
            let status = unsafe { cudf_sys::cudf_current_resource_route_destroy(owned) };
            if status != cudf_sys::CUDF_SUCCESS {
                record_native_teardown_failure(self.device, "CurrentResourceRouteLease::drop", status);
            }
        });
    }
}

impl Drop for ReservationHandle {
    fn drop(&mut self) {
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        runtime::drop_device_resource(&mut raw, self.device, "ReservationHandle::drop", |owned| {
            let status = unsafe { cudf_sys::cudf_reservation_detach(owned) };
            if status != cudf_sys::CUDF_SUCCESS {
                record_native_teardown_failure(self.device, "ReservationHandle::drop", status);
            }
        });
    }
}
