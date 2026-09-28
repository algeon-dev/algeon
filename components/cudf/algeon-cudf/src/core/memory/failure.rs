//! Allocation-failure payloads, allocator-event records, and failure callbacks.

use std::{
    collections::HashMap,
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use super::usage::DeviceMemoryPoolUsage;
use crate::{
    DeviceId,
    core::error::{self, Result},
};

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

/// Allocation owner whose cap rejected an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryFailureAllocationDomain {
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
            value if value == cudf_sys::CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN as u64 => {
                Some(Self::QueryMemoryDomain { domain_id: primary_id, member_id: secondary_id })
            },
            // Native newer kind 4 is cache allocation scope; keep compiled even on older sys.
            4 => Some(Self::CacheAllocationScope { cache_domain_id: primary_id, scope_id: secondary_id }),
            other => Some(Self::Unknown { kind: other, primary_id, secondary_id }),
        }
    }
}

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
    pub(super) fn from_sys(value: cudf_sys::cudf_memory_failure_info_t) -> Self {
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
    pub(super) fn from_sys(value: cudf_sys::cudf_allocator_event_t) -> Self {
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

pub(super) type FailureCallback = dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static;

pub(super) struct FailureCallbackState {
    callback: Arc<FailureCallback>,
    panicked: AtomicBool,
    installation_generation: AtomicU64,
}

impl FailureCallbackState {
    pub(super) fn new(callback: Arc<FailureCallback>) -> Self {
        Self { callback, panicked: AtomicBool::new(false), installation_generation: AtomicU64::new(0) }
    }

    pub(super) fn installation_generation(&self) -> Option<u64> {
        let generation = self.installation_generation.load(Ordering::Acquire);
        (generation != 0).then_some(generation)
    }

    pub(super) fn bind_installation_generation(&self, generation: u64) -> Result<()> {
        if generation == 0 {
            return Err(error::Error::new_with(
                error::ErrorCode::ForeignInvariantViolation,
                "memory::device_memory_manager::install",
                "native installation record returned a zero callback generation",
            ));
        }
        match self.installation_generation.compare_exchange(0, generation, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(()),
            Err(existing) if existing == generation => Ok(()),
            Err(existing) => Err(error::Error::new_with(
                error::ErrorCode::InvariantViolation,
                "memory::device_memory_manager::install",
                "callback state was rebound to a different installation generation",
            )
            .with_detail("existing_generation", existing.to_string())
            .with_detail("observed_generation", generation.to_string())),
        }
    }
}

pub(super) unsafe extern "C" fn failure_callback_trampoline(
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

pub(super) fn wrap_simple_failure_callback(
    callback: Arc<dyn Fn(usize) + Send + Sync + 'static>,
) -> Arc<FailureCallback> {
    Arc::new(move |info| callback(info.requested_bytes as usize))
}

pub(super) fn device_manager_failure_callbacks() -> &'static Mutex<HashMap<i32, Arc<FailureCallbackState>>> {
    static CALLBACKS: OnceLock<Mutex<HashMap<i32, Arc<FailureCallbackState>>>> = OnceLock::new();
    CALLBACKS.get_or_init(|| Mutex::new(HashMap::new()))
}
