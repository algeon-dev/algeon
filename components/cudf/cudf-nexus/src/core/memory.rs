//! Device memory resource management (RMM).
//!
//! A device memory resource is an RMM allocator handle.
//!
//! In `cudf`, these handles participate in execution in two distinct ways:
//!
//! - [`crate::Runtime`] carries one default memory-resource handle as part of
//!   its retained-provenance bundle. By default this is the process-local
//!   [`DeviceMemoryManager`] resource for the runtime's CUDA device.
//! - [`crate::ExecutionOverrides`] can borrow `&DeviceMemoryResource` to
//!   override allocator selection for one supported operation without mutating
//!   the runtime's default.
//!
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit
//! memory-resource handles, and `cudf` routes supported operations and
//! constructors through the resolved allocator for the active runtime/override
//! context.
//!
//! Query-style bounded execution should use the device-level manager plus
//! stream-bound reservations: allocate a [`crate::Runtime`], create an
//! [`crate::ExecutionBudgetToken`] or [`StreamReservationLease`], and execute
//! with the lease's stream and manager resource. Query engines that need one
//! hard cap across several streams can create a [`ReservationArena`] and attach
//! stream leases as members under it. This keeps one long-lived allocator
//! adaptor per device while charging allocations to the active stream-bound
//! reservation and, when present, its parent arena.
//!
//! Explicit CUDA and pool resources remain available for advanced library use,
//! diagnostics, tests, and non-query workflows. Use
//! [`crate::RuntimeBuilder::with_pool_memory_resource`] when you specifically
//! want one fresh pool owned by a runtime, or [`DeviceMemoryResource::pool_on`]
//! / [`DeviceMemoryResource::cuda_on`] when you need an explicit handle first
//! so you can install it into one or more runtimes, or borrow it for a one-off
//! override. In all cases, the runtime, stream, and memory resource must be on
//! the same CUDA device.
//!
//! # Examples
//!
//! ```no_run
//! use cudf::{
//!     DeviceId, ExecutionOverrides, Runtime, SortOptions,
//!     core::memory::DeviceMemoryResource,
//! };
//!
//! let device = DeviceId::new(0).unwrap();
//!
//! // Default query-backend path: use the device-level manager resource.
//! let runtime = Runtime::builder().with_device(device).build().unwrap();
//! let manager = runtime.device_memory_manager().unwrap();
//! let lease = runtime.stream_reservation(7, 64 * 1024 * 1024).unwrap();
//!
//! assert_eq!(runtime.memory_resource().device(), device);
//! assert_eq!(manager.resource().device(), device);
//! assert_eq!(lease.memory_resource().device(), device);
//!
//! // Advanced: create a runtime-owned pool when you specifically need
//! // allocator isolation instead of stream-bound query reservations.
//! let runtime = Runtime::builder()
//!     .with_device(device)
//!     .with_pool_memory_resource(256 * 1024 * 1024, None)
//!     .build()
//!     .unwrap();
//!
//! // Advanced: create an allocator handle immediately on an explicit device.
//! let shared_pool = DeviceMemoryResource::pool_on(
//!     device,
//!     512 * 1024 * 1024, // 512 MiB initial pool
//!     None,              // None = unbounded maximum
//! )
//! .unwrap();
//!
//! // Install that handle as another runtime's owned default allocator.
//! let runtime_with_explicit_mr = Runtime::builder()
//!     .with_device(device)
//!     .with_memory_resource(shared_pool.clone())
//!     .build()
//!     .unwrap();
//!
//! // Or borrow it for one supported operation without changing the runtime's
//! // default allocator.
//! let series = runtime.series(&[3i32, 1, 2]).unwrap();
//! let ctx = ExecutionOverrides::new().with_memory_resource(&shared_pool);
//! let sorted = series.sort_values_in(SortOptions::new(), &ctx).unwrap();
//!
//! assert_eq!(runtime.memory_resource().device(), device);
//! assert_eq!(runtime_with_explicit_mr.memory_resource().device(), device);
//! assert_eq!(sorted.len(), 3);
//! ```

use std::{
    collections::{HashMap, hash_map::Entry},
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::{self, NonNull},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use crate::{
    DeviceId,
    core::error::{self, Result},
    runtime::{device_guard::DeviceGuard, stream::CudaStream},
};

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

/// Source category for a device-memory allocation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryFailureSource {
    /// A physical upstream allocation failed.
    Physical,
    /// A stream reservation or reservation arena limit rejected the allocation.
    ReservationLimit,
    /// A cache-scope or arena allocation limit rejected the allocation.
    AllocationLimit,
    /// A newer native library reported a source this cudf version does not know.
    Unknown(u32),
}

impl DeviceMemoryFailureSource {
    fn from_sys(value: u32) -> Self {
        match value {
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL => Self::Physical,
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT => Self::ReservationLimit,
            // Newer kernels report AllocationLimit as 2; preserve as distinct variant so
            // downstream `AllocationLimit` matching compiles even when native is older.
            2 => Self::AllocationLimit,
            other => Self::Unknown(other),
        }
    }
}

/// Budget domain that rejected an allocation.
///
/// This identifies the budget that rejected the allocation, not every budget
/// that was attached to the stream. For example, a reservation attached to an
/// arena reports [`Self::StreamReservation`] when its own member limit rejects
/// first, and [`Self::ReservationArena`] when the arena cap rejects first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryFailureAllocationDomain {
    /// A stream-bound reservation rejected the allocation.
    StreamReservation { reservation_id: u64 },
    /// A query-scoped reservation arena rejected the allocation.
    ReservationArena { arena_id: u64, reservation_id: u64 },
    /// A query memory domain rejected the allocation.
    QueryMemoryDomain { domain_id: u64, member_id: u64 },
    /// A cache allocation scope rejected the allocation.
    CacheAllocationScope { cache_domain_id: u64, scope_id: u64 },
    /// A newer native library reported a domain kind this cudf version does not know.
    Unknown { kind: u64, primary_id: u64, secondary_id: u64 },
}

impl DeviceMemoryFailureAllocationDomain {
    fn from_sys(kind: u64, primary_id: u64, secondary_id: u64) -> Option<Self> {
        match kind {
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_DOMAIN_NONE as u64 => None,
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_DOMAIN_STREAM_RESERVATION as u64 => {
                Some(Self::StreamReservation { reservation_id: primary_id })
            },
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_DOMAIN_RESERVATION_ARENA as u64 => {
                Some(Self::ReservationArena { arena_id: primary_id, reservation_id: secondary_id })
            },
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN as u64 => {
                Some(Self::QueryMemoryDomain { domain_id: primary_id, member_id: secondary_id })
            },
            // Native newer kind 4 is cache allocation scope; keep compiled even on older sys.
            4 => Some(Self::CacheAllocationScope { cache_domain_id: primary_id, scope_id: secondary_id }),
            other => Some(Self::Unknown { kind: other, primary_id, secondary_id }),
        }
    }
}

#[cfg(test)]
mod tests;

/// Allocation-failure payload passed to observation callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryFailureInfo {
    pub device: DeviceId,
    pub source: DeviceMemoryFailureSource,
    pub allocation_domain: Option<DeviceMemoryFailureAllocationDomain>,
    pub requested_bytes: u64,
    pub callback_attempt: u32,
    pub trim_retry_count: u32,
    pub pool_usage: Option<DeviceMemoryPoolUsage>,
}

impl DeviceMemoryFailureInfo {
    fn from_sys(value: cudf_sys::cudf_memory_failure_info_t) -> Self {
        let pool_usage =
            (value.flags & cudf_sys::CUDF_MEMORY_FAILURE_INFO_HAS_POOL_USAGE != 0).then_some(DeviceMemoryPoolUsage {
                reserved_current_bytes: value.pool_reserved_current_bytes,
                reserved_high_bytes: value.pool_reserved_high_bytes,
                used_current_bytes: value.pool_used_current_bytes,
                used_high_bytes: value.pool_used_high_bytes,
            });
        let device = if value.device_ordinal >= 0 {
            DeviceId::from_validated_ordinal_raw(value.device_ordinal)
        } else {
            DeviceId::DEFAULT
        };
        let allocation_domain = (value.flags & cudf_sys::CUDF_MEMORY_FAILURE_INFO_HAS_ALLOCATION_DOMAIN != 0)
            .then(|| {
                DeviceMemoryFailureAllocationDomain::from_sys(value.reserved[0], value.reserved[1], value.reserved[2])
            })
            .flatten();
        Self {
            device,
            source: DeviceMemoryFailureSource::from_sys(value.failure_source),
            allocation_domain,
            requested_bytes: value.requested_bytes,
            callback_attempt: value.callback_attempt,
            trim_retry_count: value.trim_retry_count,
            pool_usage,
        }
    }
}

/// Completeness facts for one allocator-event record.
///
/// A cleared flag means the native callback could not observe that field; it
/// does not encode a zero-valued observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorEventCompleteness {
    pub timestamp: bool,
    pub allocation_domain: bool,
    pub pool_usage: bool,
    pub action: bool,
}

impl AllocatorEventCompleteness {
    fn from_sys(flags: u32) -> Self {
        Self {
            timestamp: flags & cudf_sys::CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_TIMESTAMP != 0,
            allocation_domain: flags & cudf_sys::CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_DOMAIN_IDENTITY != 0,
            pool_usage: flags & cudf_sys::CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_POOL_USAGE != 0,
            action: flags & cudf_sys::CUDF_ALLOCATOR_EVENT_COMPLETENESS_HAS_ACTION != 0,
        }
    }
}

/// Domain identity carried by an allocator-event record.
///
/// A generation is minted by native cuDF for every query-domain instance, so
/// a reused logical domain id cannot be correlated as an older attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorEventDomainIdentity {
    pub domain: DeviceMemoryFailureAllocationDomain,
    pub generation: u64,
}

/// Terminal action recorded by the allocator failure boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorEventAction {
    Decline,
    Retry,
    TrimAndRetry,
    CallbackPanicked,
    Unknown(i32),
}

impl AllocatorEventAction {
    fn from_sys(value: i32) -> Self {
        match value {
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_ACTION_DECLINE as i32 => Self::Decline,
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_ACTION_RETRY as i32 => Self::Retry,
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_ACTION_TRIM_AND_RETRY as i32 => Self::TrimAndRetry,
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_ACTION_CALLBACK_PANICKED => Self::CallbackPanicked,
            other => Self::Unknown(other),
        }
    }
}

/// One bounded allocator event copied from the per-device fixed ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorEvent {
    pub sequence: u64,
    pub monotonic_time_ns: Option<u64>,
    pub device: Option<DeviceId>,
    pub allocation_domain: Option<AllocatorEventDomainIdentity>,
    pub requested_bytes: u64,
    pub source: DeviceMemoryFailureSource,
    pub action: Option<AllocatorEventAction>,
    pub callback_attempt: u32,
    pub trim_retry_count: u32,
    pub pool_usage: Option<DeviceMemoryPoolUsage>,
    pub completeness: AllocatorEventCompleteness,
}

impl AllocatorEvent {
    fn from_sys(value: cudf_sys::cudf_allocator_event_t) -> Self {
        let completeness = AllocatorEventCompleteness::from_sys(value.completeness);
        let device = (value.device_ordinal >= 0).then(|| DeviceId::from_validated_ordinal_raw(value.device_ordinal));
        let allocation_domain = completeness
            .allocation_domain
            .then(|| {
                DeviceMemoryFailureAllocationDomain::from_sys(
                    value.domain_kind as u64,
                    value.domain_id,
                    value.domain_member_id,
                )
                .map(|domain| AllocatorEventDomainIdentity { domain, generation: value.domain_generation })
            })
            .flatten();
        let pool_usage = completeness.pool_usage.then_some(DeviceMemoryPoolUsage {
            reserved_current_bytes: value.pool_reserved_current_bytes,
            reserved_high_bytes: value.pool_reserved_high_bytes,
            used_current_bytes: value.pool_used_current_bytes,
            used_high_bytes: value.pool_used_high_bytes,
        });
        Self {
            sequence: value.sequence,
            monotonic_time_ns: completeness.timestamp.then_some(value.monotonic_time_ns),
            device,
            allocation_domain,
            requested_bytes: value.requested_bytes,
            source: DeviceMemoryFailureSource::from_sys(value.failure_source),
            action: completeness.action.then(|| AllocatorEventAction::from_sys(value.action)),
            callback_attempt: value.callback_attempt,
            trim_retry_count: value.trim_retry_count,
            pool_usage,
            completeness,
        }
    }
}

/// Snapshot bounds for the fixed per-device allocator-event ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorEventRingBounds {
    pub capacity: u64,
    pub first_available_sequence: Option<u64>,
    pub next_sequence: u64,
}

impl From<cudf_sys::cudf_allocator_event_ring_bounds_t> for AllocatorEventRingBounds {
    fn from(value: cudf_sys::cudf_allocator_event_ring_bounds_t) -> Self {
        Self {
            capacity: value.capacity,
            first_available_sequence: (value.first_available_sequence != 0).then_some(value.first_available_sequence),
            next_sequence: value.next_sequence,
        }
    }
}

/// Result of reading one exact sequence from the fixed allocator-event ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorEventRead {
    /// The exact sequence was copied from one stable ring record.
    Available(AllocatorEvent),
    /// The sequence currently has no stable record. It must not be attributed,
    /// even if a later read happens to observe a record.
    Pending,
    /// The fixed ring no longer retains this sequence.
    Overwritten,
    /// No producer has reserved this sequence.
    NotYetRecorded,
}

/// Whether cudf has already installed a process-local device memory manager
/// for a CUDA device, and whether that manager was created with a failure
/// callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryManagerFailureCallbackState {
    pub exists: bool,
    pub failure_callback_installed: bool,
    pub current_resource_installed: bool,
    pub failure_callback_panicked: bool,
}

/// Stable identity of the process-lifetime callback and memory partition that
/// owns a device memory manager installation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceMemoryManagerInstallationIdentity {
    pub owner_fingerprint: [u8; 16],
    pub callback_version: u32,
    pub partition_fingerprint: [u8; 16],
}

impl DeviceMemoryManagerInstallationIdentity {
    pub const fn new(owner_fingerprint: [u8; 16], callback_version: u32, partition_fingerprint: [u8; 16]) -> Self {
        Self { owner_fingerprint, callback_version, partition_fingerprint }
    }
}

fn installation_identity_to_sys(
    identity: DeviceMemoryManagerInstallationIdentity,
) -> cudf_sys::cudf_memory_manager_installation_identity_t {
    cudf_sys::cudf_memory_manager_installation_identity_t {
        owner_fingerprint: identity.owner_fingerprint,
        callback_version: identity.callback_version,
        reserved_padding: 0,
        partition_fingerprint: identity.partition_fingerprint,
    }
}

fn installation_identity_from_sys(
    identity: cudf_sys::cudf_memory_manager_installation_identity_t,
) -> DeviceMemoryManagerInstallationIdentity {
    DeviceMemoryManagerInstallationIdentity::new(
        identity.owner_fingerprint,
        identity.callback_version,
        identity.partition_fingerprint,
    )
}

/// Authoritative state of the process-lifetime manager/callback installation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceMemoryManagerInstallationStatus {
    Uninstalled,
    Incomplete,
    Owned,
    Unidentified,
    CallbackPanicked,
}

/// Whether a later owner may install or reuse the observed manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceMemoryManagerInstallationRetry {
    Installable,
    Retryable,
    Reusable,
    PermanentlyUnusable,
}

/// Closed native reason for an installation disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceMemoryManagerInstallationCause {
    None,
    CurrentResourceInstallFailed,
    CurrentResourceReplaced,
    FailureCallbackPanicked,
    FailureCallbackMissing,
    IdentityMissing,
    OptionsMismatch,
    IdentityMismatch,
    CallbackBindingMismatch,
}

/// Atomic inspection of a device's native manager and Rust callback owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryManagerInstallationState {
    pub device_ordinal: usize,
    pub status: DeviceMemoryManagerInstallationStatus,
    pub retry: DeviceMemoryManagerInstallationRetry,
    pub cause: DeviceMemoryManagerInstallationCause,
    pub effective_options: Option<DeviceMemoryManagerOptions>,
    pub identity: Option<DeviceMemoryManagerInstallationIdentity>,
    pub installation_generation: Option<u64>,
    pub failure_callback_installed: bool,
    pub current_resource_installed: bool,
    pub failure_callback_panicked: bool,
}

/// Whether an identity-aware install created or reused the process manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryManagerInstallationOutcomeKind {
    Installed,
    Reused,
}

/// Result of one successful identity-aware installation transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryManagerInstallationOutcome {
    pub kind: DeviceMemoryManagerInstallationOutcomeKind,
    pub state: DeviceMemoryManagerInstallationState,
}

/// Failure from an identity-aware install together with the exact native
/// record returned by that same locked transaction, when one was available.
#[derive(Debug)]
pub struct DeviceMemoryManagerInstallationError {
    error: error::Error,
    state: Option<Box<DeviceMemoryManagerInstallationState>>,
}

impl DeviceMemoryManagerInstallationError {
    /// The ordinary cudf error for callers that do not need recovery state.
    pub fn error(&self) -> &error::Error {
        &self.error
    }

    /// The authoritative record returned by the rejected native transaction.
    ///
    /// This is `None` only when validation or Rust callback-registry access
    /// failed before the native install/inspect transaction ran, or when the
    /// native record itself violated its ABI contract.
    pub fn installation_state(&self) -> Option<DeviceMemoryManagerInstallationState> {
        self.state.as_deref().copied()
    }

    /// Discard the optional installation state and recover the ordinary error.
    pub fn into_error(self) -> error::Error {
        self.error
    }
}

impl std::fmt::Display for DeviceMemoryManagerInstallationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for DeviceMemoryManagerInstallationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

fn installation_state_from_native_record(
    record: cudf_sys::cudf_memory_manager_installation_record_t,
) -> Result<DeviceMemoryManagerInstallationState> {
    if record.struct_size != core::mem::size_of::<cudf_sys::cudf_memory_manager_installation_record_t>() {
        return Err(error::Error::new_with(
            error::ErrorCode::ForeignInvariantViolation,
            "memory::device_memory_manager::installation_state_for_device",
            "native installation record returned an incompatible struct size",
        )
        .with_detail("struct_size", record.struct_size.to_string()));
    }
    let status = match record.status {
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_UNINSTALLED => DeviceMemoryManagerInstallationStatus::Uninstalled,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_INCOMPLETE => DeviceMemoryManagerInstallationStatus::Incomplete,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_OWNED => DeviceMemoryManagerInstallationStatus::Owned,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_UNIDENTIFIED => DeviceMemoryManagerInstallationStatus::Unidentified,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CALLBACK_PANICKED => {
            DeviceMemoryManagerInstallationStatus::CallbackPanicked
        },
        value => {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::device_memory_manager::installation_state_for_device",
                "native installation record returned an unknown status",
            )
            .with_detail("status", value.to_string()));
        },
    };
    let retry = match record.retry {
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_INSTALLABLE => DeviceMemoryManagerInstallationRetry::Installable,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_RETRYABLE => DeviceMemoryManagerInstallationRetry::Retryable,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_REUSABLE => DeviceMemoryManagerInstallationRetry::Reusable,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_PERMANENTLY_UNUSABLE => {
            DeviceMemoryManagerInstallationRetry::PermanentlyUnusable
        },
        value => {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::device_memory_manager::installation_state_for_device",
                "native installation record returned an unknown retry classification",
            )
            .with_detail("retry", value.to_string()));
        },
    };
    let cause = match record.cause {
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_NONE => DeviceMemoryManagerInstallationCause::None,
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_INSTALL_FAILED => {
            DeviceMemoryManagerInstallationCause::CurrentResourceInstallFailed
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CURRENT_RESOURCE_REPLACED => {
            DeviceMemoryManagerInstallationCause::CurrentResourceReplaced
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_FAILURE_CALLBACK_PANICKED => {
            DeviceMemoryManagerInstallationCause::FailureCallbackPanicked
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_FAILURE_CALLBACK_MISSING => {
            DeviceMemoryManagerInstallationCause::FailureCallbackMissing
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_IDENTITY_MISSING => {
            DeviceMemoryManagerInstallationCause::IdentityMissing
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_OPTIONS_MISMATCH => {
            DeviceMemoryManagerInstallationCause::OptionsMismatch
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_IDENTITY_MISMATCH => {
            DeviceMemoryManagerInstallationCause::IdentityMismatch
        },
        cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_CAUSE_CALLBACK_BINDING_MISMATCH => {
            DeviceMemoryManagerInstallationCause::CallbackBindingMismatch
        },
        value => {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::device_memory_manager::installation_state_for_device",
                "native installation record returned an unknown cause",
            )
            .with_detail("cause", value.to_string()));
        },
    };
    let exists = record.flags & cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_EXISTS != 0;
    let has_identity = record.flags & cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_HAS_IDENTITY != 0;
    let device_ordinal = usize::try_from(record.device_ordinal).map_err(|_| {
        error::Error::new_with(
            error::ErrorCode::ForeignInvariantViolation,
            "memory::device_memory_manager::installation_state_for_device",
            "native installation record returned a negative device ordinal",
        )
    })?;
    Ok(DeviceMemoryManagerInstallationState {
        device_ordinal,
        status,
        retry,
        cause,
        effective_options: exists
            .then(|| DeviceMemoryManagerOptions::from_sys(record.effective_options))
            .transpose()?,
        identity: has_identity.then(|| installation_identity_from_sys(record.identity)),
        installation_generation: (record.installation_generation != 0).then_some(record.installation_generation),
        failure_callback_installed: record.flags
            & cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_FAILURE_CALLBACK_INSTALLED
            != 0,
        current_resource_installed: record.flags
            & cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CURRENT_RESOURCE_INSTALLED
            != 0,
        failure_callback_panicked: record.flags & cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CALLBACK_PANICKED
            != 0,
    })
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

/// Physical base used by the process-local device memory manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryManagerBase {
    /// Use CUDA's stream-ordered async memory pool.
    CudaAsync { initial_pool_size: Option<usize>, release_threshold: Option<usize> },
    /// Use RMM's pool resource around the raw CUDA resource.
    Pool { initial_pool_size: usize, maximum_pool_size: Option<usize> },
}

impl Default for DeviceMemoryManagerBase {
    fn default() -> Self {
        Self::CudaAsync { initial_pool_size: None, release_threshold: None }
    }
}

/// Options applied when a device memory manager is first created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceMemoryManagerOptions {
    pub base: DeviceMemoryManagerBase,
    /// Immutable hard limit enforced by the process manager. `None` leaves the
    /// selected physical base unpartitioned.
    pub allocation_limit_bytes: Option<usize>,
}

impl DeviceMemoryManagerOptions {
    fn normalized(self) -> Self {
        let base = match self.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size, release_threshold } => {
                DeviceMemoryManagerBase::CudaAsync {
                    initial_pool_size: initial_pool_size.filter(|size| *size != 0),
                    release_threshold: release_threshold.filter(|threshold| *threshold != usize::MAX),
                }
            },
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size } => DeviceMemoryManagerBase::Pool {
                initial_pool_size,
                maximum_pool_size: maximum_pool_size.filter(|size| *size != 0),
            },
        };
        Self { base, allocation_limit_bytes: self.allocation_limit_bytes }
    }

    /// Validate this configuration without consulting CUDA or process-global
    /// allocator state.
    pub fn validate(&self) -> Result<()> {
        if self.allocation_limit_bytes == Some(0) {
            return Err(invalid_memory_manager_options(
                "allocation_limit_bytes",
                "allocation limit must be greater than zero when configured",
            ));
        }
        let normalized = (*self).normalized();
        let Some(limit) = normalized.allocation_limit_bytes else {
            if let DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size: Some(maximum) } =
                normalized.base
                && maximum < initial_pool_size
            {
                return Err(invalid_memory_manager_options(
                    "maximum_pool_size",
                    "maximum pool size must not be below the initial pool size",
                ));
            }
            return Ok(());
        };
        match normalized.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size: Some(initial), .. } if initial > limit => Err(
                invalid_memory_manager_options("initial_pool_size", "initial pool size exceeds the allocation limit"),
            ),
            DeviceMemoryManagerBase::Pool { initial_pool_size, .. } if initial_pool_size > limit => Err(
                invalid_memory_manager_options("initial_pool_size", "initial pool size exceeds the allocation limit"),
            ),
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size: Some(maximum) }
                if maximum < initial_pool_size =>
            {
                Err(invalid_memory_manager_options(
                    "maximum_pool_size",
                    "maximum pool size must not be below the initial pool size",
                ))
            },
            DeviceMemoryManagerBase::Pool { maximum_pool_size: Some(maximum), .. } if maximum > limit => Err(
                invalid_memory_manager_options("maximum_pool_size", "maximum pool size exceeds the allocation limit"),
            ),
            _ => Ok(()),
        }
    }

    fn to_sys(self) -> Result<cudf_sys::cudf_memory_manager_options_t> {
        self.validate()?;
        let mut out = cudf_sys::cudf_memory_manager_options_t {
            allocation_limit_bytes: self.allocation_limit_bytes.unwrap_or(0) as u64,
            ..Default::default()
        };
        match self.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size, release_threshold } => {
                out.base_kind = cudf_sys::CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC;
                out.initial_pool_size_bytes = initial_pool_size.unwrap_or(0) as u64;
                out.release_threshold_bytes = release_threshold.unwrap_or(usize::MAX) as u64;
            },
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size } => {
                out.base_kind = cudf_sys::CUDF_MEMORY_MANAGER_BASE_POOL;
                out.initial_pool_size_bytes = initial_pool_size as u64;
                out.maximum_pool_size_bytes = maximum_pool_size.unwrap_or(0) as u64;
            },
        }
        Ok(out)
    }

    fn from_sys(value: cudf_sys::cudf_memory_manager_options_t) -> Result<Self> {
        let initial_pool_size = memory_manager_option_bytes(value.initial_pool_size_bytes, "initial_pool_size_bytes")?;
        let base = match value.base_kind {
            cudf_sys::CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC => {
                let release_threshold = if value.release_threshold_bytes == u64::MAX {
                    None
                } else {
                    Some(memory_manager_option_bytes(value.release_threshold_bytes, "release_threshold_bytes")?)
                };
                DeviceMemoryManagerBase::CudaAsync {
                    initial_pool_size: (initial_pool_size != 0).then_some(initial_pool_size),
                    release_threshold,
                }
            },
            cudf_sys::CUDF_MEMORY_MANAGER_BASE_POOL => {
                let maximum_pool_size = if value.maximum_pool_size_bytes == 0 {
                    None
                } else {
                    Some(memory_manager_option_bytes(value.maximum_pool_size_bytes, "maximum_pool_size_bytes")?)
                };
                DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size }
            },
            other => {
                return Err(error::Error::new_with(
                    error::ErrorCode::ForeignInvariantViolation,
                    "memory::device_memory_manager::options_for_device",
                    "FFI returned an unknown memory-manager base kind",
                )
                .with_detail("base_kind", other.to_string()));
            },
        };
        let allocation_limit_bytes = if value.allocation_limit_bytes == 0 {
            None
        } else {
            Some(memory_manager_option_bytes(value.allocation_limit_bytes, "allocation_limit_bytes")?)
        };
        let options = Self { base, allocation_limit_bytes };
        options.validate()?;
        Ok(options)
    }
}

fn invalid_memory_manager_options(field: &'static str, message: &'static str) -> error::Error {
    error::Error::new_with(error::ErrorCode::ValueOutOfRange, "memory::device_memory_manager::options", message)
        .with_detail("field", field)
        .with_detail("reason", "invalid_device_memory_manager_options")
}

fn memory_manager_option_bytes(value: u64, field: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        error::Error::new_with(
            error::ErrorCode::ValueOutOfRange,
            "memory::device_memory_manager::options_for_device",
            "installed device memory manager option does not fit usize",
        )
        .with_detail("field", field)
        .with_detail("value", value.to_string())
    })
}

fn ensure_memory_manager_options_match(
    existing: DeviceMemoryManagerOptions,
    requested: DeviceMemoryManagerOptions,
) -> Result<()> {
    if existing == requested.normalized() {
        return Ok(());
    }
    Err(error::Error::new_with(
        error::ErrorCode::RuntimeBindingMismatch,
        "memory::device_memory_manager::for_device",
        "device memory manager already exists with different physical-base options",
    )
    .with_detail("existing_options", format!("{existing:?}"))
    .with_detail("requested_options", format!("{requested:?}"))
    .with_detail("reason", "device_memory_manager_options_conflict"))
}

/// Usage counters for a stream-bound reservation.
///
/// These counters are recorded by the device-level reservation-aware memory
/// manager. `current_*` counters include retained allocations that were
/// created while the reservation was attached, even after the reservation has
/// been detached from its stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservationUsage {
    pub reservation_id: u64,
    pub attached: bool,
    /// Allocations made through a query-domain member's direct resource.
    /// Zero for legacy and arena reservations.
    pub direct_total_allocations: u64,
    /// Allocations routed from the device-current resource into a query-domain
    /// member. Zero for legacy and arena reservations.
    pub routed_total_allocations: u64,
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

/// Usage counters for a query-scoped reservation arena.
///
/// The counters include allocations retained by member reservations even after
/// those members have detached from their streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservationArenaUsage {
    pub arena_id: u64,
    pub active: bool,
    pub active_members: u64,
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

/// Options for an opt-in query memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryMemoryDomainOptions {
    pub grant_bytes: usize,
    pub ceiling_bytes: usize,
}

impl QueryMemoryDomainOptions {
    /// Create a domain whose initial grant and immutable logical ceiling are equal.
    pub fn new(grant_bytes: usize) -> Self {
        Self { grant_bytes, ceiling_bytes: grant_bytes }
    }

    /// Set the immutable logical ceiling for grant resizing.
    pub fn with_ceiling_bytes(mut self, ceiling_bytes: usize) -> Self {
        self.ceiling_bytes = ceiling_bytes;
        self
    }
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
    pub domain_id: u64,
    pub active: bool,
    pub active_members: u64,
    pub ceiling_bytes: u64,
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

impl From<cudf_sys::cudf_memory_resource_usage_t> for ReservationArenaUsage {
    fn from(value: cudf_sys::cudf_memory_resource_usage_t) -> Self {
        Self {
            arena_id: value.reserved[0],
            active: value.reserved[1] != 0,
            active_members: value.reserved[2],
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

/// An owned device memory resource.
///
/// Wraps an `rmm::mr::device_memory_resource` via the C shim.
///
/// Each resource records the CUDA device it was created on. This provenance is
/// used by [`crate::Runtime`] builder validation to reject cross-device
/// handle combinations.
///
/// `DeviceMemoryResource` does not execute work by itself. Install it into
/// [`crate::Runtime::builder`] when you want it to become a runtime's owned
/// default allocator, or borrow it with
/// [`crate::ExecutionOverrides::with_memory_resource`] when you want a
/// same-device override for one supported operation.
#[derive(Clone)]
pub struct DeviceMemoryResource {
    handle: Arc<DeviceMemoryResourceHandle>,
}

struct DeviceMemoryResourceHandle {
    raw: Option<NonNull<cudf_sys::cudf_memory_resource_handle_state>>,
    device: DeviceId,
    // Pool-backed resources release physical allocations during native
    // destruction. Keep every runtime stream that selected such a resource so
    // its outstanding work can retire before that release begins.
    retirement_streams: Option<Arc<RetirementStreams>>,
    // Keeps callback states alive until after `raw` and any native upstream
    // lifetimes retained by derived resources are destroyed in Drop.
    _failure_callbacks: Vec<Arc<FailureCallbackState>>,
}

/// Streams that have submitted work through one destructible memory-resource
/// lineage.
///
/// The set lives with the resource handle rather than an execution domain:
/// domains can borrow another stream or share a pool across runtimes, while
/// only the final native resource destruction needs a retirement fence. Each
/// entry retains its stream owner so field-drop order cannot destroy the stream
/// before the resource has been fenced.
#[derive(Default)]
struct RetirementStreams {
    streams: Mutex<Vec<CudaStream>>,
}

impl RetirementStreams {
    /// Create a dedicated registry for one short-lived resource lineage. It
    /// owns `stream` until that resource has fenced and released it during
    /// teardown, and is intentionally not shared with the upstream resource.
    fn for_scoped_resource(stream: &CudaStream) -> Arc<Self> {
        let streams = Arc::new(Self::default());
        streams.register(stream);
        streams
    }

    fn register(&self, stream: &CudaStream) {
        let mut streams = match self.streams.lock() {
            Ok(streams) => streams,
            Err(poisoned) => poisoned.into_inner(),
        };
        if streams.iter().any(|registered| registered.identity() == stream.identity()) {
            return;
        }
        streams.push(stream.clone());
    }

    fn synchronize_for_teardown(&self) -> Result<()> {
        let streams = match self.streams.lock() {
            Ok(streams) => streams.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        for stream in streams {
            stream.synchronize_for_teardown("DeviceMemoryResourceHandle::drop")?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn registered_stream_count(&self) -> usize {
        match self.streams.lock() {
            Ok(streams) => streams.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

// SAFETY: RMM device_memory_resource implementations are thread-safe — allocate()
// and deallocate() use internal synchronization. The handle is an opaque pointer
// to an RMM allocator that supports concurrent access from multiple threads.
unsafe impl Send for DeviceMemoryResourceHandle {}
unsafe impl Sync for DeviceMemoryResourceHandle {}

type FailureCallback = dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static;

struct CallbackInstallationFailure {
    error: error::Error,
    state: Option<Box<DeviceMemoryManagerInstallationState>>,
}

impl CallbackInstallationFailure {
    fn with_state(error: error::Error, state: Option<DeviceMemoryManagerInstallationState>) -> Self {
        Self { error, state: state.map(Box::new) }
    }
}

impl From<error::Error> for CallbackInstallationFailure {
    fn from(error: error::Error) -> Self {
        Self { error, state: None }
    }
}

impl From<CallbackInstallationFailure> for DeviceMemoryManagerInstallationError {
    fn from(failure: CallbackInstallationFailure) -> Self {
        Self { error: failure.error, state: failure.state }
    }
}

struct FailureCallbackState {
    callback: Arc<FailureCallback>,
    panicked: AtomicBool,
    installation_generation: AtomicU64,
}

impl FailureCallbackState {
    fn new(callback: Arc<FailureCallback>) -> Self {
        Self { callback, panicked: AtomicBool::new(false), installation_generation: AtomicU64::new(0) }
    }

    fn installation_generation(&self) -> Option<u64> {
        let generation = self.installation_generation.load(Ordering::Acquire);
        (generation != 0).then_some(generation)
    }

    fn bind_installation_generation(&self, generation: u64) -> Result<()> {
        if generation == 0 {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::device_memory_manager::install_owned",
                "native installation record returned a zero callback generation",
            ));
        }
        match self.installation_generation.compare_exchange(0, generation, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(()),
            Err(existing) if existing == generation => Ok(()),
            Err(existing) => Err(error::Error::new_with(
                error::ErrorCode::InvariantViolation,
                "memory::device_memory_manager::install_owned",
                "callback state was rebound to a different installation generation",
            )
            .with_detail("existing_generation", existing.to_string())
            .with_detail("observed_generation", generation.to_string())),
        }
    }
}

unsafe extern "C" fn failure_callback_trampoline(
    info: *const cudf_sys::cudf_memory_failure_info_t,
    callback_arg: *mut c_void,
) -> i32 {
    if callback_arg.is_null() {
        return 0;
    }
    let state = unsafe { &*(callback_arg as *const FailureCallbackState) };
    if state.panicked.load(Ordering::Acquire) {
        return cudf_sys::CUDF_MEMORY_FAILURE_ACTION_CALLBACK_PANICKED;
    }
    if info.is_null() {
        return 0;
    }
    let info = DeviceMemoryFailureInfo::from_sys(unsafe { *info });
    match catch_unwind(AssertUnwindSafe(|| (state.callback)(info))) {
        Ok(()) => cudf_sys::CUDF_MEMORY_FAILURE_ACTION_DECLINE as i32,
        Err(_) => {
            // This path runs under allocator failure handling. Publish the
            // fail-closed state with one atomic store only; formatting or
            // logging here could allocate or take a blocking stdio lock.
            state.panicked.store(true, Ordering::Release);
            cudf_sys::CUDF_MEMORY_FAILURE_ACTION_CALLBACK_PANICKED
        },
    }
}

fn wrap_simple_failure_callback(callback: Arc<dyn Fn(usize) + Send + Sync + 'static>) -> Arc<FailureCallback> {
    Arc::new(move |info| callback(info.requested_bytes as usize))
}

fn device_manager_failure_callbacks() -> &'static Mutex<HashMap<i32, Arc<FailureCallbackState>>> {
    static CALLBACKS: OnceLock<Mutex<HashMap<i32, Arc<FailureCallbackState>>>> = OnceLock::new();
    CALLBACKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Process-local reservation-aware memory manager for one CUDA device.
///
/// The native layer creates at most one manager per device. The manager owns a
/// shared physical pool and one reservation-aware adaptor installed as that
/// device's current RMM resource. Query reservations attach to streams; normal
/// execution options keep using the same device-level adaptor as their memory
/// resource.
#[derive(Clone, Debug)]
pub struct DeviceMemoryManager {
    handle: Arc<DeviceMemoryManagerHandle>,
    resource: DeviceMemoryResource,
}

#[derive(Debug)]
struct DeviceMemoryManagerHandle {
    raw: NonNull<cudf_sys::cudf_memory_manager_handle_state>,
    device: DeviceId,
}

// SAFETY: the native manager is process-local, internally synchronized, and
// intentionally shared across threads for one CUDA device.
unsafe impl Send for DeviceMemoryManagerHandle {}
unsafe impl Sync for DeviceMemoryManagerHandle {}

/// Lease that routes legacy global-RMM allocations on one stream to an
/// explicit runtime memory resource.
///
/// The device manager's permanent current RMM resource is never replaced.
/// Instead, its reservation-aware adaptor dispatches allocations made through
/// that current resource by stream and retains the target resource until each
/// routed allocation is deallocated. This is for native libraries whose API
/// cannot accept an explicit allocator even when their surrounding runtime
/// has one.
#[must_use = "dropping the lease removes the stream route for future current-RMM allocations"]
#[derive(Debug)]
pub struct CurrentResourceRouteLease {
    raw: Option<NonNull<cudf_sys::cudf_current_resource_route_handle_state>>,
    device: DeviceId,
    // Keep the target owner, stream identity, and manager handle live while
    // native code can still allocate through the registered route.
    _target: DeviceMemoryResource,
    _stream: CudaStream,
    _manager: DeviceMemoryManager,
}

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
    reservation: Arc<ReservationHandle>,
    resource: Option<DeviceMemoryResource>,
    _query_domain: Option<QueryMemoryDomain>,
    arena: Option<ReservationArena>,
    stream: CudaStream,
    manager: DeviceMemoryManager,
}

/// Query-scoped parent reservation shared by stream-bound member leases.
///
/// Dropping the final handle prevents new members from attaching. Existing
/// allocations made by member leases remain accounted natively until those GPU
/// allocations are released.
#[derive(Clone, Debug)]
pub struct ReservationArena {
    handle: Arc<ReservationArenaHandle>,
    manager: DeviceMemoryManager,
}

/// Query memory domain sharing its device manager's CUDA async pool.
#[derive(Clone, Debug)]
pub struct QueryMemoryDomain {
    handle: Arc<QueryMemoryDomainHandle>,
    manager: DeviceMemoryManager,
}

#[derive(Debug)]
struct ReservationHandle {
    raw: Option<NonNull<cudf_sys::cudf_reservation_handle_state>>,
    device: DeviceId,
}

#[derive(Debug)]
struct ReservationArenaHandle {
    raw: Option<NonNull<cudf_sys::cudf_reservation_arena_handle_state>>,
    device: DeviceId,
}

#[derive(Debug)]
struct QueryMemoryDomainHandle {
    raw: Option<NonNull<cudf_sys::cudf_query_memory_domain_handle_state>>,
    device: DeviceId,
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

impl DeviceMemoryManager {
    /// Explicitly install the memory domain for `device` (idempotent).
    ///
    /// cudf owns the memory management of every CUDA device it uses: the
    /// first time a device's manager is created, its reservation-aware
    /// adaptor is installed as RMM's current resource for that device — and
    /// it stays installed for the process lifetime. By default that happens
    /// implicitly on the first `Runtime` build for the device; services
    /// should instead call this once per device at startup so the
    /// installation point is explicit, failures surface before the first
    /// query, and other in-process rmm consumers see the final resource from
    /// the start.
    ///
    /// ```no_run
    /// use cudf::{DeviceId, DeviceMemoryManager, Runtime};
    ///
    /// // Service startup: claim every device the service will use.
    /// for ordinal in 0..DeviceId::count().unwrap() {
    ///     DeviceMemoryManager::install_for_device(
    ///         DeviceId::new(ordinal).unwrap(),
    ///     )
    ///     .unwrap();
    /// }
    ///
    /// // Each pipeline then gets its own Runtime (own stream) plus a
    /// // stream reservation for budgeting; they share the device domain.
    /// let rt = Runtime::builder().build().unwrap();
    /// let _lease = rt.stream_reservation(1, 1 << 30).unwrap();
    /// ```
    pub fn install_for_device(device: DeviceId) -> Result<Self> {
        Self::for_device_with_options(device, DeviceMemoryManagerOptions::default())
    }

    /// Explicitly install the memory domain for `device` with physical-base
    /// options.
    ///
    /// If a manager already exists, its effective options must match exactly;
    /// the process-global allocator cannot be replaced after first use.
    pub fn install_for_device_with_options(device: DeviceId, options: DeviceMemoryManagerOptions) -> Result<Self> {
        Self::for_device_with_options(device, options)
    }

    /// Explicitly install the memory domain for `device` with a process-lifetime
    /// allocation-failure callback.
    ///
    /// The first successful call for a device creates the device manager,
    /// installs the reservation-aware allocator as the device's current RMM
    /// resource, then wraps that allocator in RMM's failure-callback adaptor.
    /// The callback observes reservation-limit failures and upstream
    /// out-of-memory failures, with [`DeviceMemoryFailureInfo::source`]
    /// identifying the failing allocator boundary. It runs inside the
    /// allocating thread and must be re-entrancy-safe: do not allocate through
    /// the same resource, block on application locks, or unwind.
    ///
    /// Later calls for a manager that was already installed with a failure
    /// callback return the existing manager; the original callback remains in
    /// force. Later calls after a plain [`Self::install_for_device`] or
    /// [`Self::for_device`] fail because the process-global current resource
    /// cannot be swapped safely after first use.
    pub fn install_for_device_with_failure_callback(
        device: DeviceId,
        callback: Arc<dyn Fn(usize) + Send + Sync + 'static>,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(
            device,
            wrap_simple_failure_callback(callback),
            DeviceMemoryManagerOptions::default(),
            None,
        )
        .map(|(manager, _, _, _)| manager)
        .map_err(|failure| failure.error)
    }

    /// Explicitly install the memory domain for `device` with a
    /// observation-only process-lifetime allocation-failure callback.
    pub fn install_for_device_with_failure_callback_details(
        device: DeviceId,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(
            device,
            callback,
            DeviceMemoryManagerOptions::default(),
            None,
        )
        .map(|(manager, _, _, _)| manager)
        .map_err(|failure| failure.error)
    }

    /// Explicitly install the memory domain for `device` with physical-base
    /// options and a process-lifetime allocation-failure callback.
    pub fn install_for_device_with_failure_callback_and_options(
        device: DeviceId,
        callback: Arc<dyn Fn(usize) + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(
            device,
            wrap_simple_failure_callback(callback),
            options,
            None,
        )
        .map(|(manager, _, _, _)| manager)
        .map_err(|failure| failure.error)
    }

    /// Explicitly install the memory domain for `device` with physical-base
    /// options and an observation-only process-lifetime allocation-failure
    /// callback.
    pub fn install_for_device_with_failure_callback_details_and_options(
        device: DeviceId,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(device, callback, options, None)
            .map(|(manager, _, _, _)| manager)
            .map_err(|failure| failure.error)
    }

    /// Atomically installs or reuses the process-lifetime manager for an
    /// identified owner, callback contract, and enforced memory partition.
    ///
    /// Reuse requires exact identity and effective options. An existing plain,
    /// legacy-unidentified, foreign, or panicked callback is rejected rather
    /// than rebound to the new caller.
    pub fn install_for_device_with_owned_failure_callback_details_and_options(
        device: DeviceId,
        identity: DeviceMemoryManagerInstallationIdentity,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> Result<(Self, DeviceMemoryManagerInstallationOutcome)> {
        Self::try_install_for_device_with_owned_failure_callback_details_and_options(
            device, identity, callback, options,
        )
        .map_err(DeviceMemoryManagerInstallationError::into_error)
    }

    /// Identity-aware install with the exact native record preserved on
    /// incomplete, incompatible, or permanently unusable outcomes.
    pub fn try_install_for_device_with_owned_failure_callback_details_and_options(
        device: DeviceId,
        identity: DeviceMemoryManagerInstallationIdentity,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> std::result::Result<(Self, DeviceMemoryManagerInstallationOutcome), DeviceMemoryManagerInstallationError> {
        let (manager, installed, callback_state, state) =
            Self::for_device_with_failure_callback_details_and_options(device, callback, options, Some(identity))?;
        let state = state.ok_or_else(|| DeviceMemoryManagerInstallationError {
            error: error::Error::new_with(
                error::ErrorCode::InvariantViolation,
                "memory::device_memory_manager::install_owned",
                "identity-aware native installation did not return its authoritative record",
            ),
            state: None,
        })?;
        if state.status != DeviceMemoryManagerInstallationStatus::Owned
            || state.retry != DeviceMemoryManagerInstallationRetry::Reusable
            || state.cause != DeviceMemoryManagerInstallationCause::None
            || state.identity != Some(identity)
            || state.installation_generation != callback_state.installation_generation()
            || !state.failure_callback_installed
            || !state.current_resource_installed
        {
            return Err(DeviceMemoryManagerInstallationError {
                error: error::Error::new_with(
                    error::ErrorCode::InvariantViolation,
                    "memory::device_memory_manager::install_owned",
                    "installed device memory manager did not retain the requested ownership identity",
                )
                .with_detail("requested_identity", format!("{identity:?}"))
                .with_detail("observed_state", format!("{state:?}"))
                .with_detail("reason", "device_memory_manager_installation_identity_lost"),
                state: Some(Box::new(state)),
            });
        }
        let kind = if installed {
            DeviceMemoryManagerInstallationOutcomeKind::Installed
        } else {
            DeviceMemoryManagerInstallationOutcomeKind::Reused
        };
        Ok((manager, DeviceMemoryManagerInstallationOutcome { kind, state }))
    }

    /// Returns whether `device` already has a process-local memory manager and
    /// whether that manager was created with a failure callback.
    pub fn failure_callback_state_for_device(device: DeviceId) -> Result<DeviceMemoryManagerFailureCallbackState> {
        Self::failure_callback_state_for_device_raw(device)
    }

    /// Inspects native manager state and the original Rust callback identity
    /// while serializing against identity-aware installation.
    pub fn installation_state_for_device(device: DeviceId) -> Result<DeviceMemoryManagerInstallationState> {
        let record = Self::native_installation_record_for_device(device)?;
        installation_state_from_native_record(record)
    }

    /// Returns the effective physical-base options for an existing process-local manager.
    pub fn options_for_device(device: DeviceId) -> Result<Option<DeviceMemoryManagerOptions>> {
        let _guard = DeviceGuard::new(device)?;
        let mut exists = 0;
        let mut options = cudf_sys::cudf_memory_manager_options_t::default();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_options_for_device(device.ordinal_raw(), &mut exists, &mut options)
            },
            "memory::device_memory_manager::options_for_device",
            [("device", device.ordinal().to_string())],
        )?;
        (exists != 0).then(|| DeviceMemoryManagerOptions::from_sys(options)).transpose()
    }

    /// Resolve the process-local manager for `device`, creating it if needed.
    ///
    /// On first creation the manager installs its reservation-aware adaptor as
    /// the current RMM resource for `device`. Later calls return aliases to
    /// the same manager and resource.
    pub fn for_device(device: DeviceId) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let mut manager = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_for_device(device.ordinal_raw(), &mut manager) },
            "memory::device_memory_manager::for_device",
            [("device", device.ordinal().to_string())],
        )?;
        let manager = error::require_non_null(manager, "memory::device_memory_manager::for_device", "memory manager")?;

        let mut resource = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_manager_resource(manager.as_ptr(), &mut resource) },
            "memory::device_memory_manager::resource",
        )?;
        let resource =
            DeviceMemoryResource::from_created_handle(resource, device, "memory::device_memory_manager::resource")?;

        Ok(Self { handle: Arc::new(DeviceMemoryManagerHandle { raw: manager, device }), resource })
    }

    /// Resolve the process-local manager for `device`, creating it with
    /// `options` if needed. An existing manager with different effective
    /// options returns [`crate::ErrorCode::RuntimeBindingMismatch`].
    pub fn for_device_with_options(device: DeviceId, options: DeviceMemoryManagerOptions) -> Result<Self> {
        options.validate()?;
        if let Some(existing) = Self::options_for_device(device)? {
            ensure_memory_manager_options_match(existing, options)?;
        }
        let _guard = DeviceGuard::new(device)?;
        let options = options.to_sys()?;
        let mut manager = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_for_device_with_options(device.ordinal_raw(), &options, &mut manager)
            },
            "memory::device_memory_manager::for_device",
            [("device", device.ordinal().to_string())],
        )?;
        let manager = error::require_non_null(manager, "memory::device_memory_manager::for_device", "memory manager")?;

        let mut resource = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_manager_resource(manager.as_ptr(), &mut resource) },
            "memory::device_memory_manager::resource",
        )?;
        let resource =
            DeviceMemoryResource::from_created_handle(resource, device, "memory::device_memory_manager::resource")?;

        Ok(Self { handle: Arc::new(DeviceMemoryManagerHandle { raw: manager, device }), resource })
    }

    fn for_device_with_failure_callback_details_and_options(
        device: DeviceId,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
        installation_identity: Option<DeviceMemoryManagerInstallationIdentity>,
    ) -> std::result::Result<
        (Self, bool, Arc<FailureCallbackState>, Option<DeviceMemoryManagerInstallationState>),
        CallbackInstallationFailure,
    > {
        options.validate()?;
        let callbacks = device_manager_failure_callbacks();
        let mut callbacks = callbacks.lock().map_err(|_| {
            error::Error::new_with(
                error::ErrorCode::InvariantViolation,
                "memory::device_memory_manager::for_device_with_failure_callback",
                "device memory manager failure callback registry mutex was poisoned",
            )
        })?;
        let native_state_before = if installation_identity.is_none() {
            if let Some(existing) = Self::options_for_device(device)? {
                ensure_memory_manager_options_match(existing, options)?;
            }
            Some(Self::failure_callback_state_for_device_raw(device)?)
        } else {
            None
        };
        let _guard = DeviceGuard::new(device)?;
        let device_ordinal = device.ordinal_raw();
        let options = options.to_sys()?;
        if native_state_before.is_some_and(|state| state.exists && state.failure_callback_installed)
            && !callbacks.contains_key(&device_ordinal)
        {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::for_device_with_failure_callback",
                "device memory manager already exists with a failure callback not owned by this cudf callback registry",
            )
            .into());
        }
        let mut inserted_callback_state = false;
        let callback_state = match callbacks.entry(device_ordinal) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                inserted_callback_state = true;
                Arc::clone(entry.insert(Arc::new(FailureCallbackState::new(callback))))
            },
        };

        let callback_arg = Arc::as_ptr(&callback_state) as *mut c_void;
        let mut manager = ptr::null_mut();
        let mut installation_record = cudf_sys::cudf_memory_manager_installation_record_t::default();
        let status = if let Some(identity) = installation_identity {
            let identity = installation_identity_to_sys(identity);
            unsafe {
                cudf_sys::cudf_memory_manager_install_owned_with_failure_callback_v2_and_options(
                    device.ordinal_raw(),
                    Some(failure_callback_trampoline),
                    callback_arg,
                    &options,
                    &identity,
                    &mut manager,
                    &mut installation_record,
                )
            }
        } else {
            unsafe {
                cudf_sys::cudf_memory_manager_for_device_with_failure_callback_v2_and_options(
                    device.ordinal_raw(),
                    Some(failure_callback_trampoline),
                    callback_arg,
                    &options,
                    &mut manager,
                )
            }
        };
        let created_by_call = installation_identity.is_some()
            && installation_record.flags & cudf_sys::CUDF_MEMORY_MANAGER_INSTALLATION_FLAG_CREATED_BY_CALL != 0;
        if installation_identity.is_some() && (!inserted_callback_state || created_by_call) {
            callback_state.bind_installation_generation(installation_record.installation_generation)?;
        }
        if installation_identity.is_some()
            && inserted_callback_state
            && !created_by_call
            && callbacks.get(&device_ordinal).is_some_and(|state| Arc::ptr_eq(state, &callback_state))
        {
            callbacks.remove(&device_ordinal);
        }
        let installation_state = installation_identity
            .is_some()
            .then(|| installation_state_from_native_record(installation_record))
            .transpose()?;
        if status != cudf_sys::CUDF_SUCCESS {
            let cause = installation_state.map(|state| state.cause);
            drop(callbacks);
            let native_error = error::check_status_in_with_details(
                status,
                "memory::device_memory_manager::for_device_with_failure_callback",
                [("device", device.ordinal().to_string())],
            )
            .expect_err("non-success native status must produce an error");
            let installation_error = match cause {
                Some(
                    DeviceMemoryManagerInstallationCause::CurrentResourceReplaced
                    | DeviceMemoryManagerInstallationCause::FailureCallbackPanicked
                    | DeviceMemoryManagerInstallationCause::FailureCallbackMissing
                    | DeviceMemoryManagerInstallationCause::IdentityMissing
                    | DeviceMemoryManagerInstallationCause::OptionsMismatch
                    | DeviceMemoryManagerInstallationCause::IdentityMismatch
                    | DeviceMemoryManagerInstallationCause::CallbackBindingMismatch,
                ) => error::Error::new_with(
                    error::ErrorCode::RuntimeBindingMismatch,
                    "memory::device_memory_manager::install_owned",
                    "device memory manager installation is incompatible or permanently unusable",
                )
                .with_detail("native_error", native_error.to_string()),
                _ => native_error,
            };
            let error = match cause {
                Some(cause) => installation_error.with_detail("installation_cause", format!("{cause:?}")),
                None => installation_error,
            };
            return Err(CallbackInstallationFailure::with_state(error, installation_state));
        }
        drop(callbacks);
        let manager = error::require_non_null(
            manager,
            "memory::device_memory_manager::for_device_with_failure_callback",
            "memory manager",
        )
        .map_err(|error| CallbackInstallationFailure::with_state(error, installation_state))?;

        let mut resource = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_manager_resource(manager.as_ptr(), &mut resource) },
            "memory::device_memory_manager::resource",
        )
        .map_err(|error| CallbackInstallationFailure::with_state(error, installation_state))?;
        let resource =
            DeviceMemoryResource::from_created_handle(resource, device, "memory::device_memory_manager::resource")
                .map_err(|error| CallbackInstallationFailure::with_state(error, installation_state))?;

        let installed = installation_identity
            .map_or_else(|| native_state_before.is_none_or(|state| !state.exists), |_| created_by_call);
        Ok((
            Self { handle: Arc::new(DeviceMemoryManagerHandle { raw: manager, device }), resource },
            installed,
            callback_state,
            installation_state,
        ))
    }

    fn failure_callback_state_for_device_raw(device: DeviceId) -> Result<DeviceMemoryManagerFailureCallbackState> {
        let state = Self::installation_state_for_device(device)?;
        Ok(DeviceMemoryManagerFailureCallbackState {
            exists: state.status != DeviceMemoryManagerInstallationStatus::Uninstalled,
            failure_callback_installed: state.failure_callback_installed,
            current_resource_installed: state.current_resource_installed,
            failure_callback_panicked: state.failure_callback_panicked,
        })
    }

    fn native_installation_record_for_device(
        device: DeviceId,
    ) -> Result<cudf_sys::cudf_memory_manager_installation_record_t> {
        let _guard = DeviceGuard::new(device)?;
        let mut record = cudf_sys::cudf_memory_manager_installation_record_t::default();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_installation_record(device.ordinal_raw(), &mut record) },
            "memory::device_memory_manager::installation_state_for_device",
            [("device", device.ordinal().to_string())],
        )?;
        Ok(record)
    }

    /// CUDA device owned by this manager.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Device-level reservation-aware adaptor for explicit execution options.
    pub fn resource(&self) -> &DeviceMemoryResource {
        &self.resource
    }

    /// Route legacy current-RMM allocations made on `stream` to `resource`.
    ///
    /// This preserves the manager's permanent current RMM resource and is
    /// intended for a native API that accepts the stream explicitly but still
    /// performs some retained allocations through RMM's current resource.
    /// The returned lease must outlive that native owner. Allocations made
    /// before the lease drops retain their target resource until deallocation.
    ///
    /// Passing this manager's own resource explicitly authorizes that stream
    /// to use the manager's physical backend for the lease lifetime.
    pub fn route_current_resource(
        &self,
        stream: &CudaStream,
        resource: &DeviceMemoryResource,
    ) -> Result<CurrentResourceRouteLease> {
        let _guard = DeviceGuard::new(self.device())?;
        if stream.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::route_current_resource",
                "current-resource routing requires stream and memory manager to be on the same CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(stream.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "stream_and_memory_manager")
            .with_detail("reason", "cross_device_current_resource_route"));
        }
        if resource.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::route_current_resource",
                "current-resource routing requires target resource and memory manager to be on the same CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(resource.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "target_resource_and_memory_manager")
            .with_detail("reason", "cross_device_current_resource_route"));
        }

        let mut raw = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_route_current_resource(
                    self.handle.raw.as_ptr(),
                    stream.as_raw(),
                    resource.as_raw(),
                    &mut raw,
                )
            },
            "memory::device_memory_manager::route_current_resource",
            [
                ("device", self.device().ordinal().to_string()),
                ("stream_device", stream.device().ordinal().to_string()),
                ("resource_device", resource.device().ordinal().to_string()),
            ],
        )?;

        Ok(CurrentResourceRouteLease {
            raw: NonNull::new(raw),
            device: self.device(),
            _target: resource.clone(),
            _stream: stream.clone(),
            _manager: self.clone(),
        })
    }

    /// Read physical CUDA async-pool counters for this manager.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_pool_usage_t::default();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_pool_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::device_memory_manager::pool_usage",
            [("device", self.device().ordinal().to_string())],
        )?;
        Ok(usage.into())
    }

    /// Explicitly trim idle bytes from this manager's CUDA async pool.
    pub fn trim_to(&self, keep_bytes: u64) -> Result<()> {
        let _guard = DeviceGuard::new(self.device())?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_trim_to(self.handle.raw.as_ptr(), keep_bytes) },
            "memory::device_memory_manager::trim_to",
            [("device", self.device().ordinal().to_string()), ("keep_bytes", keep_bytes.to_string())],
        )
    }

    /// Returns the current bounds of this device's fixed allocator-event ring.
    ///
    /// Event producers are allocation-failure callbacks; this consumer API can
    /// take normal runtime locks and does not participate in allocator
    /// re-entrancy.
    pub fn allocator_event_ring_bounds(&self) -> Result<AllocatorEventRingBounds> {
        let _guard = DeviceGuard::new(self.device())?;
        // Native initializes every output field, including `struct_size`.
        let mut bounds = unsafe { std::mem::zeroed::<cudf_sys::cudf_allocator_event_ring_bounds_t>() };
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_allocator_event_ring_bounds(self.handle.raw.as_ptr(), &mut bounds) },
            "memory::device_memory_manager::allocator_event_ring_bounds",
            [("device", self.device().ordinal().to_string())],
        )?;
        if bounds.struct_size != core::mem::size_of::<cudf_sys::cudf_allocator_event_ring_bounds_t>() {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::device_memory_manager::allocator_event_ring_bounds",
                "native allocator-event ring bounds returned an incompatible struct size",
            )
            .with_detail("struct_size", bounds.struct_size.to_string()));
        }
        Ok(bounds.into())
    }

    /// Reads one exact allocator-event sequence without treating absence as a
    /// zero-valued event. Callers must handle pending and overwritten results
    /// explicitly before attributing an event.
    pub fn read_allocator_event(&self, sequence: u64) -> Result<AllocatorEventRead> {
        let _guard = DeviceGuard::new(self.device())?;
        // Native zeroes unavailable reads and fully initializes available ones.
        let mut event = unsafe { std::mem::zeroed::<cudf_sys::cudf_allocator_event_t>() };
        let mut status = cudf_sys::CUDF_ALLOCATOR_EVENT_READ_NOT_YET_RECORDED;
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_read_allocator_event(
                    self.handle.raw.as_ptr(),
                    sequence,
                    &mut event,
                    &mut status,
                )
            },
            "memory::device_memory_manager::read_allocator_event",
            [("device", self.device().ordinal().to_string()), ("sequence", sequence.to_string())],
        )?;
        match status {
            value if value == cudf_sys::CUDF_ALLOCATOR_EVENT_READ_AVAILABLE => {
                if event.struct_size != core::mem::size_of::<cudf_sys::cudf_allocator_event_t>() {
                    return Err(error::Error::new_with(
                        error::ErrorCode::ForeignInvariantViolation,
                        "memory::device_memory_manager::read_allocator_event",
                        "native allocator event returned an incompatible struct size",
                    )
                    .with_detail("struct_size", event.struct_size.to_string()));
                }
                if event.sequence != sequence {
                    return Err(error::Error::new_with(
                        error::ErrorCode::RuntimeBindingMismatch,
                        "memory::device_memory_manager::read_allocator_event",
                        "allocator event availability returned a different sequence",
                    )
                    .with_detail("requested_sequence", sequence.to_string())
                    .with_detail("observed_sequence", event.sequence.to_string()));
                }
                Ok(AllocatorEventRead::Available(AllocatorEvent::from_sys(event)))
            },
            value if value == cudf_sys::CUDF_ALLOCATOR_EVENT_READ_PENDING => Ok(AllocatorEventRead::Pending),
            value if value == cudf_sys::CUDF_ALLOCATOR_EVENT_READ_OVERWRITTEN => Ok(AllocatorEventRead::Overwritten),
            value if value == cudf_sys::CUDF_ALLOCATOR_EVENT_READ_NOT_YET_RECORDED => {
                Ok(AllocatorEventRead::NotYetRecorded)
            },
            other => Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::read_allocator_event",
                "allocator event reader returned an unknown status",
            )
            .with_detail("status", other.to_string())),
        }
    }

    /// Attach a hard live-byte reservation to `stream`.
    pub fn reserve_stream(
        &self,
        stream: &CudaStream,
        reservation_id: u64,
        limit_bytes: usize,
    ) -> Result<StreamReservationLease> {
        self.reserve_stream_with_arena(stream, reservation_id, limit_bytes, None)
    }

    /// Create a query-scoped reservation arena owned by this manager.
    pub fn reservation_arena(&self, arena_id: u64, limit_bytes: usize) -> Result<ReservationArena> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut arena = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_create_reservation_arena(
                    self.handle.raw.as_ptr(),
                    arena_id,
                    limit_bytes,
                    &mut arena,
                )
            },
            "memory::device_memory_manager::reservation_arena",
            [
                ("device", self.device().ordinal().to_string()),
                ("arena_id", arena_id.to_string()),
                ("limit_bytes", limit_bytes.to_string()),
            ],
        )?;
        let arena = ReservationArenaHandle {
            raw: Some(error::require_non_null(
                arena,
                "memory::device_memory_manager::reservation_arena",
                "reservation arena",
            )?),
            device: self.device(),
        };
        Ok(ReservationArena { handle: Arc::new(arena), manager: self.clone() })
    }

    /// Create a query memory domain backed by this manager's shared CUDA async pool.
    ///
    /// When this manager owns a static allocation partition, live allocations
    /// through the shared pool consume that same partition limit.
    pub fn query_memory_domain(&self, domain_id: u64, options: QueryMemoryDomainOptions) -> Result<QueryMemoryDomain> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut domain = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_create_query_memory_domain(
                    self.handle.raw.as_ptr(),
                    domain_id,
                    options.grant_bytes,
                    options.ceiling_bytes,
                    &mut domain,
                )
            },
            "memory::device_memory_manager::query_memory_domain",
            [
                ("device", self.device().ordinal().to_string()),
                ("domain_id", domain_id.to_string()),
                ("grant_bytes", options.grant_bytes.to_string()),
                ("ceiling_bytes", options.ceiling_bytes.to_string()),
            ],
        )?;
        let domain = QueryMemoryDomainHandle {
            raw: Some(error::require_non_null(
                domain,
                "memory::device_memory_manager::query_memory_domain",
                "query memory domain",
            )?),
            device: self.device(),
        };
        Ok(QueryMemoryDomain { handle: Arc::new(domain), manager: self.clone() })
    }

    fn reserve_stream_with_arena(
        &self,
        stream: &CudaStream,
        reservation_id: u64,
        limit_bytes: usize,
        arena: Option<&ReservationArena>,
    ) -> Result<StreamReservationLease> {
        let _guard = DeviceGuard::new(self.device())?;
        if stream.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::reserve_stream",
                "stream reservation requires stream and memory manager to be on the same CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(stream.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "stream_and_memory_manager")
            .with_detail("reason", "cross_device_stream_reservation"));
        }
        if let Some(arena) = arena
            && arena.manager.device() != self.device()
        {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::reserve_stream",
                "reservation arena requires arena and memory manager to be on the same CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(arena.manager.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "arena_and_memory_manager")
            .with_detail("reason", "cross_device_reservation_arena"));
        }

        let mut reservation = ptr::null_mut();
        let status = match arena {
            Some(arena) => unsafe {
                cudf_sys::cudf_memory_manager_attach_reservation_to_arena(
                    self.handle.raw.as_ptr(),
                    stream.as_raw(),
                    arena.handle.raw(),
                    reservation_id,
                    limit_bytes,
                    &mut reservation,
                )
            },
            None => unsafe {
                cudf_sys::cudf_memory_manager_attach_reservation(
                    self.handle.raw.as_ptr(),
                    stream.as_raw(),
                    reservation_id,
                    limit_bytes,
                    &mut reservation,
                )
            },
        };
        error::check_status_in_with_details(
            status,
            "memory::device_memory_manager::reserve_stream",
            [
                ("device", self.device().ordinal().to_string()),
                ("reservation_id", reservation_id.to_string()),
                ("limit_bytes", limit_bytes.to_string()),
            ],
        )?;
        let reservation = ReservationHandle {
            raw: Some(error::require_non_null(
                reservation,
                "memory::device_memory_manager::reserve_stream",
                "stream reservation",
            )?),
            device: self.device(),
        };
        Ok(StreamReservationLease {
            manager: self.clone(),
            stream: stream.clone(),
            arena: arena.cloned(),
            _query_domain: None,
            resource: None,
            reservation: Arc::new(reservation),
        })
    }
}

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
        crate::runtime::drop_device_resource(&mut raw, self.device, "ReservationArenaHandle::drop", |owned| {
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
        crate::runtime::drop_device_resource(&mut raw, self.device, "QueryMemoryDomainHandle::drop", |owned| {
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
        crate::runtime::drop_device_resource(&mut raw, self.device, "CurrentResourceRouteLease::drop", |owned| {
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
        crate::runtime::drop_device_resource(&mut raw, self.device, "ReservationHandle::drop", |owned| {
            let status = unsafe { cudf_sys::cudf_reservation_detach(owned) };
            if status != cudf_sys::CUDF_SUCCESS {
                record_native_teardown_failure(self.device, "ReservationHandle::drop", status);
            }
        });
    }
}

impl DeviceMemoryResource {
    fn raw_maximum_pool_size(maximum_pool_size: Option<usize>) -> usize {
        maximum_pool_size.unwrap_or(0)
    }

    fn from_created_handle(
        handle: cudf_sys::cudf_memory_resource_handle,
        device: DeviceId,
        constructor: &'static str,
    ) -> Result<Self> {
        Self::from_created_handle_with_callbacks(handle, device, constructor, None, Vec::new())
    }

    /// Adopt an owned native memory-resource handle for unstable test support.
    ///
    /// # Safety
    /// `handle` must be a non-null, uniquely owned memory-resource handle
    /// created for `device`. It must not be borrowed elsewhere and must be
    /// destroyed with `cudf_memory_resource_destroy` exactly once.
    pub(crate) unsafe fn adopt_owned_for_testing(
        handle: cudf_sys::cudf_memory_resource_handle,
        device: DeviceId,
    ) -> Result<Self> {
        Self::from_created_handle(handle, device, "testing::core::adopt_owned_memory_resource_for_test")
    }

    fn from_created_handle_with_callbacks(
        handle: cudf_sys::cudf_memory_resource_handle,
        device: DeviceId,
        constructor: &'static str,
        retirement_streams: Option<Arc<RetirementStreams>>,
        failure_callbacks: Vec<Arc<FailureCallbackState>>,
    ) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: Some(error::require_non_null(handle, constructor, "device memory resource")?),
                device,
                retirement_streams,
                _failure_callbacks: failure_callbacks,
            }),
        })
    }

    /// Create a CUDA (default) memory resource handle on the calling thread's
    /// current device.
    pub fn cuda() -> Result<Self> {
        Self::cuda_on(DeviceId::current()?)
    }

    /// Create a CUDA (default) memory resource handle on an explicit device.
    ///
    /// The resource is created immediately on `device`, and the calling
    /// thread's previous CUDA current device is restored before this function
    /// returns.
    pub fn cuda_on(device: DeviceId) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut handle) },
            "memory::device_memory_resource::cuda_on",
        )?;
        Self::from_created_handle(handle, device, "memory::device_memory_resource::cuda_on")
    }

    /// Create a pool memory resource handle on the calling thread's current
    /// device.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn pool(initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Result<Self> {
        Self::pool_on(DeviceId::current()?, initial_pool_size, maximum_pool_size)
    }

    /// Create a pool memory resource handle on an explicit device.
    ///
    /// The resource is created immediately on `device`, and the calling
    /// thread's previous CUDA current device is restored before this function
    /// returns.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn pool_on(device: DeviceId, initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let raw_maximum_pool_size = Self::raw_maximum_pool_size(maximum_pool_size);
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_pool(initial_pool_size, raw_maximum_pool_size, &mut handle)
            },
            "memory::device_memory_resource::pool_on",
            [
                ("initial_pool_size", initial_pool_size.to_string()),
                ("maximum_pool_size", maximum_pool_size.map_or_else(|| "None".to_string(), |size| size.to_string())),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            device,
            "memory::device_memory_resource::pool_on",
            Some(Arc::new(RetirementStreams::default())),
            Vec::new(),
        )
    }

    /// Create a cudaMallocAsync-backed memory resource handle on the calling
    /// thread's current device.
    ///
    /// `release_threshold = None` keeps the CUDA default used by this crate:
    /// retain idle memory across synchronizations until an explicit
    /// [`Self::trim_to`] call.
    pub fn cuda_async(initial_pool_size: Option<usize>, release_threshold: Option<usize>) -> Result<Self> {
        Self::cuda_async_on(DeviceId::current()?, initial_pool_size, release_threshold)
    }

    /// Create a cudaMallocAsync-backed memory resource handle on an explicit
    /// device.
    pub fn cuda_async_on(
        device: DeviceId,
        initial_pool_size: Option<usize>,
        release_threshold: Option<usize>,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_cuda_async(
                    initial_pool_size.unwrap_or(0),
                    release_threshold.unwrap_or(usize::MAX) as u64,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::cuda_async_on",
            [
                ("initial_pool_size", initial_pool_size.map_or_else(|| "None".to_string(), |size| size.to_string())),
                ("release_threshold", release_threshold.map_or_else(|| "None".to_string(), |size| size.to_string())),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            device,
            "memory::device_memory_resource::cuda_async_on",
            Some(Arc::new(RetirementStreams::default())),
            Vec::new(),
        )
    }

    /// Create a hard-limiting memory resource around this resource.
    ///
    /// The returned resource uses `self` as its upstream allocator and rejects
    /// allocations once the adaptor's live allocated bytes would exceed
    /// `allocation_limit_bytes`. Existing upstream allocations are not counted.
    ///
    /// This is an execution boundary, not a preflight sizing contract: callers
    /// still need operation-specific sizing APIs for retained outputs they
    /// intend to reserve before execution.
    pub fn limiting(&self, allocation_limit_bytes: usize) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_limiting(self.handle.raw(), allocation_limit_bytes, &mut handle)
            },
            "memory::device_memory_resource::limiting",
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            "memory::device_memory_resource::limiting",
            self.handle.retirement_streams.clone(),
            self.handle._failure_callbacks.clone(),
        )
    }

    /// Create a failure-callback memory resource around this resource.
    ///
    /// The callback observes each failed allocation before the resource
    /// preserves the original allocation failure.
    ///
    /// The callback runs inside RMM's allocation path on the allocating thread,
    /// so it must be re-entrancy-safe and must not allocate through the same
    /// memory resource. Panics are caught at the FFI boundary, so Rust
    /// unwinding never crosses into C++ or changes the allocation failure.
    pub fn with_failure_callback(&self, callback: Arc<dyn Fn(usize) + Send + Sync + 'static>) -> Result<Self> {
        self.with_failure_callback_details(wrap_simple_failure_callback(callback))
    }

    /// Create an observation-only failure-callback memory resource around this
    /// resource.
    ///
    /// The callback receives device, request, retry-attempt, and async-pool
    /// usage facts without changing allocation control flow.
    pub fn with_failure_callback_details(
        &self,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let callback_state = Arc::new(FailureCallbackState::new(callback));
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        // The callback state is stored in the returned handle so `callback_arg`
        // remains valid until the native resource is destroyed.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_failure_callback_v2(
                    self.handle.raw(),
                    Some(failure_callback_trampoline),
                    Arc::as_ptr(&callback_state) as *mut c_void,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::with_failure_callback",
            [("upstream_device", self.device().ordinal().to_string())],
        )?;
        let mut failure_callbacks = self.handle._failure_callbacks.clone();
        failure_callbacks.push(callback_state);
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            "memory::device_memory_resource::with_failure_callback",
            self.handle.retirement_streams.clone(),
            failure_callbacks,
        )
    }

    /// Create a hard-limiting memory resource that also reports current and
    /// peak allocation counters.
    ///
    /// The returned resource uses `self` as its upstream allocator, rejects
    /// allocations once aligned live bytes through the adaptor would exceed
    /// `allocation_limit_bytes`, and exposes requested/aligned live-byte
    /// counters through [`DeviceMemoryResource::budget_tracking_usage`].
    pub fn budget_tracking_limit(&self, allocation_limit_bytes: usize) -> Result<Self> {
        self.create_budget_tracking_limit(
            allocation_limit_bytes,
            self.handle.retirement_streams.clone(),
            "memory::device_memory_resource::budget_tracking_limit",
        )
    }

    /// Create one child budget tracker with a private retirement registry for
    /// `stream`.
    ///
    /// Unlike [`Self::budget_tracking_limit`], this child does not inherit the
    /// upstream allocator's retirement-stream registry. Its final handle drop
    /// fences and releases only `stream` before the child adaptor is destroyed.
    /// This is for retained per-operation allocation scopes whose runtime owns
    /// exactly one stream; callers must keep the returned resource alive until
    /// that runtime and every native owner using it have been destroyed.
    pub(crate) fn budget_tracking_limit_for_scope(
        &self,
        allocation_limit_bytes: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        if stream.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_resource::budget_tracking_limit_for_scope",
                "scoped budget tracker requires its stream and upstream resource to share one CUDA device",
            )
            .with_fact(error::FACT_LEFT_DEVICE, error::DeviceIdFact(stream.device().ordinal_raw()))
            .with_fact(error::FACT_RIGHT_DEVICE, error::DeviceIdFact(self.device().ordinal_raw()))
            .with_detail("resource", "stream_and_upstream_memory_resource")
            .with_detail("reason", "cross_device_scoped_budget_tracker"));
        }
        self.create_budget_tracking_limit(
            allocation_limit_bytes,
            Some(RetirementStreams::for_scoped_resource(stream)),
            "memory::device_memory_resource::budget_tracking_limit_for_scope",
        )
    }

    fn create_budget_tracking_limit(
        &self,
        allocation_limit_bytes: usize,
        retirement_streams: Option<Arc<RetirementStreams>>,
        operation: &'static str,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_budget_tracking(
                    self.handle.raw(),
                    allocation_limit_bytes,
                    &mut handle,
                )
            },
            operation,
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            operation,
            retirement_streams,
            self.handle._failure_callbacks.clone(),
        )
    }

    #[cfg(test)]
    pub(crate) fn __debug_retirement_stream_count(&self) -> usize {
        self.handle.retirement_streams.as_ref().map_or(0, |streams| streams.registered_stream_count())
    }

    /// Read usage counters from a budget-tracking memory resource.
    ///
    /// The returned counters are diagnostic snapshots loaded independently;
    /// they are not a multi-field atomic live-polling snapshot. Operation
    /// reports should query usage after the budgeted call returns.
    ///
    /// Returns an error if this resource was not created by
    /// [`DeviceMemoryResource::budget_tracking_limit`].
    pub fn budget_tracking_usage(&self) -> Result<DeviceMemoryResourceUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_usage(self.handle.raw(), &mut usage) },
            "memory::device_memory_resource::budget_tracking_usage",
        )?;
        Ok(usage.into())
    }

    /// Read physical CUDA async-pool counters for this resource.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut usage = cudf_sys::cudf_memory_pool_usage_t::default();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_resource_pool_usage(self.handle.raw(), &mut usage) },
            "memory::device_memory_resource::pool_usage",
            [("device", self.device().ordinal().to_string())],
        )?;
        Ok(usage.into())
    }

    /// Explicitly trim idle bytes from this resource's CUDA async pool.
    pub fn trim_to(&self, keep_bytes: u64) -> Result<()> {
        let _guard = DeviceGuard::new(self.device())?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_resource_trim_to(self.handle.raw(), keep_bytes) },
            "memory::device_memory_resource::trim_to",
            [("device", self.device().ordinal().to_string()), ("keep_bytes", keep_bytes.to_string())],
        )
    }

    /// The CUDA device this memory resource was created on.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Register a same-device stream whose work must retire before this
    /// pool-backed resource can release its physical backing allocation.
    pub(crate) fn register_stream_for_retirement(&self, stream: &CudaStream) {
        debug_assert_eq!(stream.device(), self.device());
        if let Some(retirement_streams) = &self.handle.retirement_streams {
            retirement_streams.register(stream);
        }
    }

    /// Borrow this owner as a low-level `cudf-sys` memory-resource handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `DeviceMemoryResource` remains responsible for
    /// destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.handle.raw()) }
    }

    pub(crate) fn as_raw(&self) -> cudf_sys::cudf_memory_resource_handle {
        self.handle.raw()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_raw(self.handle.raw(), &mut out) },
            "memory::device_memory_resource::interop_raw_handle",
        )?;
        Ok(out)
    }
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        self.interop_raw_handle().map(|raw| raw as usize).unwrap_or(0)
    }
}

impl Drop for DeviceMemoryResourceHandle {
    fn drop(&mut self) {
        let retirement_error = self
            .raw
            .is_some()
            .then_some(self.retirement_streams.as_ref())
            .flatten()
            .and_then(|retirement_streams| retirement_streams.synchronize_for_teardown().err());
        if let Some(error) = retirement_error {
            // A pool may release backing allocations in native destruction.
            // Without a successful fence, leaking is safer than racing queued
            // device work with that release. Leak the registered stream owners
            // with it: letting their native stream handles drop could otherwise
            // retire queued work after the allocator has been quarantined.
            let _ = self.raw.take();
            std::mem::forget(std::mem::take(&mut self.retirement_streams));
            if !self._failure_callbacks.is_empty() {
                std::mem::forget(std::mem::take(&mut self._failure_callbacks));
            }
            crate::runtime::quarantine_device_resource(self.device, "DeviceMemoryResourceHandle::drop", &error);
            return;
        }
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        let mut status = None;
        crate::runtime::drop_device_resource(&mut raw, self.device, "DeviceMemoryResourceHandle::drop", |owned| {
            status = Some(unsafe { cudf_sys::cudf_memory_resource_destroy(owned, self.device.ordinal_raw()) });
        });
        if status != Some(cudf_sys::CUDF_SUCCESS) {
            if !self._failure_callbacks.is_empty() {
                // Native destroy can be rejected when the resource is installed
                // as current. In that case the native adaptor may still call
                // `callback_arg`, so leak only the callback states instead of
                // leaving dangling Rust pointers behind.
                std::mem::forget(std::mem::take(&mut self._failure_callbacks));
            }
            if let Some(status) = status {
                record_native_teardown_failure(self.device, "DeviceMemoryResourceHandle::drop", status);
            }
        }
    }
}

impl DeviceMemoryResourceHandle {
    fn raw(&self) -> cudf_sys::cudf_memory_resource_handle {
        self.raw.expect("live memory resource retains its native handle").as_ptr()
    }
}

impl ReservationHandle {
    fn raw(&self) -> cudf_sys::cudf_reservation_handle {
        self.raw.expect("live reservation retains its native handle").as_ptr()
    }
}

impl ReservationArenaHandle {
    fn raw(&self) -> cudf_sys::cudf_reservation_arena_handle {
        self.raw.expect("live reservation arena retains its native handle").as_ptr()
    }
}

impl QueryMemoryDomainHandle {
    fn raw(&self) -> cudf_sys::cudf_query_memory_domain_handle {
        self.raw.expect("live query memory domain retains its native handle").as_ptr()
    }
}

fn record_native_teardown_failure(device: DeviceId, operation: &'static str, status: cudf_sys::cudf_status_t) {
    if status == cudf_sys::CUDF_ERROR_FATAL_CUDA {
        crate::runtime::report_device_fault(
            device,
            crate::runtime::DeviceFaultKind::FatalDeviceOperation,
            operation,
            Some(crate::runtime::DeviceFaultNativeStatus::Cudf(status)),
        );
    }
    crate::runtime::device_health::record_teardown_failure(
        device,
        operation,
        &format_args!("native teardown failed with status {status}"),
    );
}

impl std::fmt::Debug for DeviceMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryResource")
            .field("device", &self.device())
            .field("handle", &self.__debug_raw_handle_value())
            .finish()
    }
}
