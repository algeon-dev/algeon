//! Device memory resource management (RMM).
//!
//! A device memory resource is an RMM allocator handle.
//!
//! In `cudf-rs`, these handles participate in execution in two distinct ways:
//!
//! - [`crate::Runtime`] carries one default memory-resource handle as part of
//!   its retained-provenance bundle. By default this is the process-local
//!   [`DeviceMemoryManager`] resource for the runtime's CUDA device.
//! - [`crate::ExecutionOverrides`] can borrow `&DeviceMemoryResource` to
//!   override allocator selection for one supported operation without mutating
//!   the runtime's default.
//!
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit
//! memory-resource handles, and `cudf-rs` routes supported operations and
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
        atomic::{AtomicBool, Ordering},
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

/// Action returned by a reclaim-capable device-memory failure callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceMemoryFailureAction {
    /// Preserve the original allocation failure.
    Decline,
    /// Retry the failed allocation without reclaiming through cudf-rs.
    Retry,
    /// Trim idle async-pool pages and retry if trimming succeeds.
    TrimAndRetry,
}

impl DeviceMemoryFailureAction {
    fn to_sys(self) -> cudf_sys::cudf_memory_failure_action_t {
        match self {
            Self::Decline => cudf_sys::cudf_memory_failure_action_t::CUDF_MEMORY_FAILURE_ACTION_DECLINE,
            Self::Retry => cudf_sys::cudf_memory_failure_action_t::CUDF_MEMORY_FAILURE_ACTION_RETRY,
            Self::TrimAndRetry => cudf_sys::cudf_memory_failure_action_t::CUDF_MEMORY_FAILURE_ACTION_TRIM_AND_RETRY,
        }
    }
}

/// Source category for a device-memory allocation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryFailureSource {
    /// A physical upstream allocation failed.
    Physical,
    /// A stream reservation or reservation arena limit rejected the allocation.
    ReservationLimit,
    /// A newer native library reported a source this cudf-rs version does not know.
    Unknown(u32),
}

impl DeviceMemoryFailureSource {
    fn from_sys(value: u32) -> Self {
        match value {
            value if value == cudf_sys::cudf_memory_failure_source_t::CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL as u32 => {
                Self::Physical
            },
            value
                if value
                    == cudf_sys::cudf_memory_failure_source_t::CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT as u32 =>
            {
                Self::ReservationLimit
            },
            other => Self::Unknown(other),
        }
    }
}

/// Allocation-failure payload passed to reclaim-capable failure callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryFailureInfo {
    pub device: DeviceId,
    pub source: DeviceMemoryFailureSource,
    pub requested_bytes: u64,
    pub callback_attempt: u32,
    pub trim_retry_count: u32,
    pub pool_usage: Option<DeviceMemoryPoolUsage>,
}

impl DeviceMemoryFailureInfo {
    const TRIM_RETRY_IDLE_TO_REQUEST_RATIO: u64 = 10;

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
        Self {
            device,
            source: DeviceMemoryFailureSource::from_sys(value.failure_source),
            requested_bytes: value.requested_bytes,
            callback_attempt: value.callback_attempt,
            trim_retry_count: value.trim_retry_count,
            pool_usage,
        }
    }

    /// Returns whether trimming idle async-pool pages is likely to help the
    /// failed allocation, using cudf-rs' default trim-retry policy.
    pub fn should_trim_and_retry(&self) -> bool {
        if self.source != DeviceMemoryFailureSource::Physical {
            return false;
        }
        let Some(pool_usage) = self.pool_usage else {
            return false;
        };
        if self.requested_bytes == 0 || self.trim_retry_count > 0 {
            return false;
        }
        if pool_usage.reserved_current_bytes <= pool_usage.used_current_bytes {
            return false;
        }
        let idle_bytes = pool_usage.reserved_current_bytes - pool_usage.used_current_bytes;
        idle_bytes / self.requested_bytes >= Self::TRIM_RETRY_IDLE_TO_REQUEST_RATIO
    }
}

/// Whether cudf-rs has already installed a process-local device memory manager
/// for a CUDA device, and whether that manager was created with a failure
/// callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryManagerFailureCallbackState {
    pub exists: bool,
    pub failure_callback_installed: bool,
    pub failure_callback_panicked: bool,
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
}

impl DeviceMemoryManagerOptions {
    fn to_sys(self) -> cudf_sys::cudf_memory_manager_options_t {
        let mut out = cudf_sys::cudf_memory_manager_options_t::default();
        match self.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size, release_threshold } => {
                out.base_kind = cudf_sys::cudf_memory_manager_base_kind_t::CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC;
                out.initial_pool_size_bytes = initial_pool_size.unwrap_or(0) as u64;
                out.release_threshold_bytes = release_threshold.unwrap_or(usize::MAX) as u64;
            },
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size } => {
                out.base_kind = cudf_sys::cudf_memory_manager_base_kind_t::CUDF_MEMORY_MANAGER_BASE_POOL;
                out.initial_pool_size_bytes = initial_pool_size as u64;
                out.maximum_pool_size_bytes = maximum_pool_size.unwrap_or(0) as u64;
            },
        }
        out
    }
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
    pub release_threshold_bytes: Option<usize>,
}

impl QueryMemoryDomainOptions {
    /// Create a domain whose initial grant and immutable pool ceiling are equal.
    pub fn new(grant_bytes: usize) -> Self {
        Self { grant_bytes, ceiling_bytes: grant_bytes, release_threshold_bytes: None }
    }

    /// Set the immutable driver pool ceiling.
    pub fn with_ceiling_bytes(mut self, ceiling_bytes: usize) -> Self {
        self.ceiling_bytes = ceiling_bytes;
        self
    }

    /// Set the CUDA pool release threshold.
    pub fn with_release_threshold_bytes(mut self, release_threshold_bytes: usize) -> Self {
        self.release_threshold_bytes = Some(release_threshold_bytes);
        self
    }

    fn release_threshold_for_sys(self) -> u64 {
        self.release_threshold_bytes.unwrap_or(self.ceiling_bytes) as u64
    }
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
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
    // Keeps callback states alive until after `raw` and any native upstream
    // lifetimes retained by derived resources are destroyed in Drop.
    _failure_callbacks: Vec<Arc<FailureCallbackState>>,
}

// SAFETY: RMM device_memory_resource implementations are thread-safe — allocate()
// and deallocate() use internal synchronization. The handle is an opaque pointer
// to an RMM allocator that supports concurrent access from multiple threads.
unsafe impl Send for DeviceMemoryResourceHandle {}
unsafe impl Sync for DeviceMemoryResourceHandle {}

type FailureCallback = dyn Fn(DeviceMemoryFailureInfo) -> DeviceMemoryFailureAction + Send + Sync + 'static;

struct FailureCallbackState {
    callback: Arc<FailureCallback>,
    panicked: AtomicBool,
}

impl FailureCallbackState {
    fn new(callback: Arc<FailureCallback>) -> Self {
        Self { callback, panicked: AtomicBool::new(false) }
    }

    fn has_panicked(&self) -> bool {
        self.panicked.load(Ordering::Acquire)
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
        return 0;
    }
    if info.is_null() {
        return 0;
    }
    let info = DeviceMemoryFailureInfo::from_sys(unsafe { *info });
    match catch_unwind(AssertUnwindSafe(|| (state.callback)(info))) {
        Ok(action) => action.to_sys() as i32,
        Err(_) => {
            if !state.panicked.swap(true, Ordering::AcqRel) {
                eprintln!(
                    "[cudf] warning: device memory failure callback panicked while handling allocation failure for {} bytes; disabling callback",
                    info.requested_bytes
                );
            }
            0
        },
    }
}

fn wrap_simple_failure_callback(callback: Arc<dyn Fn(usize) -> bool + Send + Sync + 'static>) -> Arc<FailureCallback> {
    Arc::new(move |info| {
        if info.should_trim_and_retry() {
            return DeviceMemoryFailureAction::TrimAndRetry;
        }
        if callback(info.requested_bytes as usize) {
            DeviceMemoryFailureAction::Retry
        } else {
            DeviceMemoryFailureAction::Decline
        }
    })
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
    raw: NonNull<std::ffi::c_void>,
    device: DeviceId,
}

// SAFETY: the native manager is process-local, internally synchronized, and
// intentionally shared across threads for one CUDA device.
unsafe impl Send for DeviceMemoryManagerHandle {}
unsafe impl Sync for DeviceMemoryManagerHandle {}

/// A live reservation attached to a CUDA stream.
///
/// Dropping this value detaches the reservation from the stream. Allocations
/// made while it was attached remain accounted to the reservation identity
/// until those retained objects are destroyed.
#[derive(Debug)]
pub struct StreamReservationLease {
    // Field order is load-bearing: Rust drops fields in declaration order,
    // and the reservation must detach while the stream (whose raw value keys
    // the native attachment map) and the manager are still alive. If the
    // stream dropped first, a recycled raw stream value could inherit this
    // dying reservation's accounting.
    reservation: ReservationHandle,
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

/// Opt-in query memory domain backed by a private CUDA async pool.
#[derive(Clone, Debug)]
pub struct QueryMemoryDomain {
    handle: Arc<QueryMemoryDomainHandle>,
    manager: DeviceMemoryManager,
}

#[derive(Debug)]
struct ReservationHandle {
    raw: NonNull<std::ffi::c_void>,
}

#[derive(Debug)]
struct ReservationArenaHandle {
    raw: NonNull<std::ffi::c_void>,
}

#[derive(Debug)]
struct QueryMemoryDomainHandle {
    raw: NonNull<std::ffi::c_void>,
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
    /// cudf-rs owns the memory management of every CUDA device it uses: the
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
        Self::for_device(device)
    }

    /// Explicitly install the memory domain for `device` with physical-base
    /// options.
    ///
    /// Options only affect first creation for a device. If a manager already
    /// exists, the existing process-local manager is returned.
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
    /// distinguishing which retry actions can help. It runs inside the
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
        callback: Arc<dyn Fn(usize) -> bool + Send + Sync + 'static>,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(
            device,
            wrap_simple_failure_callback(callback),
            DeviceMemoryManagerOptions::default(),
        )
    }

    /// Explicitly install the memory domain for `device` with a
    /// reclaim-capable process-lifetime allocation-failure callback.
    pub fn install_for_device_with_failure_callback_details(
        device: DeviceId,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) -> DeviceMemoryFailureAction + Send + Sync + 'static>,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(
            device,
            callback,
            DeviceMemoryManagerOptions::default(),
        )
    }

    /// Explicitly install the memory domain for `device` with physical-base
    /// options and a process-lifetime allocation-failure callback.
    pub fn install_for_device_with_failure_callback_and_options(
        device: DeviceId,
        callback: Arc<dyn Fn(usize) -> bool + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(
            device,
            wrap_simple_failure_callback(callback),
            options,
        )
    }

    /// Explicitly install the memory domain for `device` with physical-base
    /// options and a reclaim-capable process-lifetime allocation-failure
    /// callback.
    pub fn install_for_device_with_failure_callback_details_and_options(
        device: DeviceId,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) -> DeviceMemoryFailureAction + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> Result<Self> {
        Self::for_device_with_failure_callback_details_and_options(device, callback, options)
    }

    /// Returns whether `device` already has a process-local memory manager and
    /// whether that manager was created with a failure callback.
    pub fn failure_callback_state_for_device(device: DeviceId) -> Result<DeviceMemoryManagerFailureCallbackState> {
        Self::failure_callback_state_for_device_raw(device)
    }

    /// Resolve the process-local manager for `device`, creating it if needed.
    ///
    /// On first creation the manager installs its reservation-aware adaptor as
    /// the current RMM resource for `device`. Later calls return aliases to
    /// the same manager and resource.
    pub fn for_device(device: DeviceId) -> Result<Self> {
        Self::for_device_with_options(device, DeviceMemoryManagerOptions::default())
    }

    /// Resolve the process-local manager for `device`, creating it with
    /// `options` if needed.
    pub fn for_device_with_options(device: DeviceId, options: DeviceMemoryManagerOptions) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let options = options.to_sys();
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
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) -> DeviceMemoryFailureAction + Send + Sync + 'static>,
        options: DeviceMemoryManagerOptions,
    ) -> Result<Self> {
        let _guard = DeviceGuard::new(device)?;
        let device_ordinal = device.ordinal_raw();
        let options = options.to_sys();
        let native_state_before = Self::failure_callback_state_for_device_raw(device)?;
        let callbacks = device_manager_failure_callbacks();
        let mut callbacks = callbacks.lock().map_err(|_| {
            error::Error::new_with(
                error::ErrorCode::InvariantViolation,
                "memory::device_memory_manager::for_device_with_failure_callback",
                "device memory manager failure callback registry mutex was poisoned",
            )
        })?;
        if native_state_before.exists
            && native_state_before.failure_callback_installed
            && !callbacks.contains_key(&device_ordinal)
        {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::for_device_with_failure_callback",
                "device memory manager already exists with a failure callback not owned by this cudf-rs callback registry",
            ));
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
        let status = unsafe {
            cudf_sys::cudf_memory_manager_for_device_with_failure_callback_v2_and_options(
                device.ordinal_raw(),
                Some(failure_callback_trampoline),
                callback_arg,
                &options,
                &mut manager,
            )
        };
        if status != cudf_sys::CUDF_SUCCESS {
            let native_state = Self::native_failure_callback_state_for_device_raw(device);
            if inserted_callback_state
                && callbacks.get(&device_ordinal).is_some_and(|state| Arc::ptr_eq(state, &callback_state))
                && native_state.as_ref().is_ok_and(|state| !state.exists || !state.failure_callback_installed)
            {
                callbacks.remove(&device_ordinal);
            }
            error::check_status_in_with_details(
                status,
                "memory::device_memory_manager::for_device_with_failure_callback",
                [("device", device.ordinal().to_string())],
            )?;
        }
        drop(callbacks);
        let manager = error::require_non_null(
            manager,
            "memory::device_memory_manager::for_device_with_failure_callback",
            "memory manager",
        )?;

        let mut resource = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_manager_resource(manager.as_ptr(), &mut resource) },
            "memory::device_memory_manager::resource",
        )?;
        let resource =
            DeviceMemoryResource::from_created_handle(resource, device, "memory::device_memory_manager::resource")?;

        Ok(Self { handle: Arc::new(DeviceMemoryManagerHandle { raw: manager, device }), resource })
    }

    fn failure_callback_state_for_device_raw(device: DeviceId) -> Result<DeviceMemoryManagerFailureCallbackState> {
        let mut state = Self::native_failure_callback_state_for_device_raw(device)?;
        if state.failure_callback_installed {
            let callbacks = device_manager_failure_callbacks().lock().map_err(|_| {
                error::Error::new_with(
                    error::ErrorCode::InvariantViolation,
                    "memory::device_memory_manager::failure_callback_state_for_device",
                    "device memory manager failure callback registry mutex was poisoned",
                )
            })?;
            state.failure_callback_panicked =
                callbacks.get(&device.ordinal_raw()).is_some_and(|state| state.has_panicked());
        }
        Ok(state)
    }

    fn native_failure_callback_state_for_device_raw(
        device: DeviceId,
    ) -> Result<DeviceMemoryManagerFailureCallbackState> {
        let _guard = DeviceGuard::new(device)?;
        let mut exists = 0;
        let mut failure_callback_installed = 0;
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_failure_callback_state(
                    device.ordinal_raw(),
                    &mut exists,
                    &mut failure_callback_installed,
                )
            },
            "memory::device_memory_manager::failure_callback_state_for_device",
            [("device", device.ordinal().to_string())],
        )?;
        Ok(DeviceMemoryManagerFailureCallbackState {
            exists: exists != 0,
            failure_callback_installed: failure_callback_installed != 0,
            failure_callback_panicked: false,
        })
    }

    /// CUDA device owned by this manager.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Device-level reservation-aware adaptor for explicit execution options.
    pub fn resource(&self) -> &DeviceMemoryResource {
        &self.resource
    }

    /// Read physical CUDA async-pool counters for this manager.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
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
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_manager_trim_to(self.handle.raw.as_ptr(), keep_bytes) },
            "memory::device_memory_manager::trim_to",
            [("device", self.device().ordinal().to_string()), ("keep_bytes", keep_bytes.to_string())],
        )
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
            raw: error::require_non_null(
                arena,
                "memory::device_memory_manager::reservation_arena",
                "reservation arena",
            )?,
        };
        Ok(ReservationArena { handle: Arc::new(arena), manager: self.clone() })
    }

    /// Create an opt-in query memory domain backed by a private CUDA async pool.
    pub fn query_memory_domain(&self, domain_id: u64, options: QueryMemoryDomainOptions) -> Result<QueryMemoryDomain> {
        let mut domain = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_manager_create_query_memory_domain(
                    self.handle.raw.as_ptr(),
                    domain_id,
                    options.grant_bytes,
                    options.ceiling_bytes,
                    options.release_threshold_for_sys(),
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
            raw: error::require_non_null(
                domain,
                "memory::device_memory_manager::query_memory_domain",
                "query memory domain",
            )?,
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
        if let Some(arena) = arena {
            if arena.manager.device() != self.device() {
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
        }

        let mut reservation = ptr::null_mut();
        let status = match arena {
            Some(arena) => unsafe {
                cudf_sys::cudf_memory_manager_attach_reservation_to_arena(
                    self.handle.raw.as_ptr(),
                    stream.as_raw(),
                    arena.handle.raw.as_ptr(),
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
            raw: error::require_non_null(
                reservation,
                "memory::device_memory_manager::reserve_stream",
                "stream reservation",
            )?,
        };
        Ok(StreamReservationLease {
            manager: self.clone(),
            stream: stream.clone(),
            arena: arena.cloned(),
            _query_domain: None,
            reservation,
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
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_reservation_arena_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::reservation_arena::usage",
        )?;
        Ok(usage.into())
    }

    /// Resize this arena's hard live-byte limit.
    ///
    /// Shrinking below current live usage is allowed. Future member allocations
    /// fail until enough memory is released or the arena is grown again.
    pub fn resize(&self, limit_bytes: usize) -> Result<()> {
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_reservation_arena_resize(self.handle.raw.as_ptr(), limit_bytes) },
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

    /// Attach a stream to this query memory domain.
    pub fn reserve_stream(&self, stream: &CudaStream, reservation_id: u64) -> Result<StreamReservationLease> {
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

        let mut reservation = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_query_memory_domain_attach_stream(
                    self.handle.raw.as_ptr(),
                    stream.as_raw(),
                    reservation_id,
                    &mut reservation,
                )
            },
            "memory::query_memory_domain::reserve_stream",
            [("device", self.device().ordinal().to_string()), ("reservation_id", reservation_id.to_string())],
        )?;
        let reservation = ReservationHandle {
            raw: error::require_non_null(
                reservation,
                "memory::query_memory_domain::reserve_stream",
                "stream reservation",
            )?,
        };
        Ok(StreamReservationLease {
            manager: self.manager.clone(),
            stream: stream.clone(),
            arena: None,
            _query_domain: Some(self.clone()),
            reservation,
        })
    }

    /// Snapshot usage counters for this query memory domain.
    pub fn usage(&self) -> Result<QueryMemoryDomainUsage> {
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_query_memory_domain_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::query_memory_domain::usage",
        )?;
        Ok(usage.into())
    }

    /// Read physical CUDA async-pool counters for this query memory domain.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
        let mut usage = cudf_sys::cudf_memory_pool_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_query_memory_domain_pool_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::query_memory_domain::pool_usage",
        )?;
        Ok(usage.into())
    }

    /// Resize this domain's live-byte grant.
    pub fn resize_grant(&self, grant_bytes: usize) -> Result<()> {
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_query_memory_domain_resize_grant(self.handle.raw.as_ptr(), grant_bytes) },
            "memory::query_memory_domain::resize_grant",
            [("device", self.device().ordinal().to_string()), ("grant_bytes", grant_bytes.to_string())],
        )
    }

    /// Explicitly trim idle bytes from this domain's CUDA async pool.
    pub fn trim_to(&self, keep_bytes: u64) -> Result<()> {
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_query_memory_domain_trim(self.handle.raw.as_ptr(), keep_bytes) },
            "memory::query_memory_domain::trim_to",
            [("device", self.device().ordinal().to_string()), ("keep_bytes", keep_bytes.to_string())],
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
        self.manager.resource()
    }

    /// Parent reservation arena for this lease, when it is an arena member.
    pub fn reservation_arena(&self) -> Option<&ReservationArena> {
        self.arena.as_ref()
    }

    /// Snapshot usage counters for this reservation.
    pub fn usage(&self) -> Result<ReservationUsage> {
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_reservation_usage(self.reservation.raw.as_ptr(), &mut usage) },
            "memory::stream_reservation_lease::usage",
        )?;
        Ok(usage.into())
    }

    /// Resize this reservation's hard live-byte limit.
    ///
    /// Shrinking below current live usage is allowed. Future allocations fail
    /// until enough memory is released or the reservation is grown again.
    /// Query memory domain member leases cannot be resized through this
    /// method; resize the domain with [`QueryMemoryDomain::resize_grant`].
    pub fn resize(&self, limit_bytes: usize) -> Result<()> {
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_reservation_resize(self.reservation.raw.as_ptr(), limit_bytes) },
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
        let status = unsafe { cudf_sys::cudf_reservation_arena_destroy(self.raw.as_ptr()) };
        if status != cudf_sys::CUDF_SUCCESS {
            eprintln!("[cudf::ReservationArena] warning: failed to destroy reservation arena");
        }
    }
}

impl Drop for QueryMemoryDomainHandle {
    fn drop(&mut self) {
        let status = unsafe { cudf_sys::cudf_query_memory_domain_destroy(self.raw.as_ptr()) };
        if status != cudf_sys::CUDF_SUCCESS {
            eprintln!("[cudf::QueryMemoryDomain] warning: failed to destroy query memory domain");
        }
    }
}

impl Drop for ReservationHandle {
    fn drop(&mut self) {
        let status = unsafe { cudf_sys::cudf_reservation_detach(self.raw.as_ptr()) };
        if status != cudf_sys::CUDF_SUCCESS {
            eprintln!("[cudf::StreamReservationLease] warning: failed to detach stream reservation");
        }
    }
}

impl DeviceMemoryResource {
    fn raw_maximum_pool_size(maximum_pool_size: Option<usize>) -> usize {
        maximum_pool_size.unwrap_or(0)
    }

    fn from_created_handle(handle: *mut std::ffi::c_void, device: DeviceId, constructor: &'static str) -> Result<Self> {
        Self::from_created_handle_with_callbacks(handle, device, constructor, Vec::new())
    }

    fn from_created_handle_with_callbacks(
        handle: *mut std::ffi::c_void,
        device: DeviceId,
        constructor: &'static str,
        failure_callbacks: Vec<Arc<FailureCallbackState>>,
    ) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: error::require_non_null(handle, constructor, "device memory resource")?,
                device,
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
        Self::from_created_handle(handle, device, "memory::device_memory_resource::pool_on")
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
        Self::from_created_handle(handle, device, "memory::device_memory_resource::cuda_async_on")
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
                cudf_sys::cudf_memory_resource_create_limiting(
                    self.handle.raw.as_ptr(),
                    allocation_limit_bytes,
                    &mut handle,
                )
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
            self.handle._failure_callbacks.clone(),
        )
    }

    /// Create a failure-callback memory resource around this resource.
    ///
    /// The returned resource first applies cudf-rs' default trim-retry policy
    /// when the failure payload shows enough idle async-pool memory. If that
    /// policy does not request a retry, it calls `callback(bytes)`. Returning
    /// `true` retries the failed allocation; returning `false` preserves the
    /// original allocation failure.
    ///
    /// The callback runs inside RMM's allocation path on the allocating thread,
    /// so it must be re-entrancy-safe and must not allocate through the same
    /// memory resource. Panics are caught at the FFI boundary and treated as
    /// `false`, so Rust unwinding never crosses into C++.
    pub fn with_failure_callback(&self, callback: Arc<dyn Fn(usize) -> bool + Send + Sync + 'static>) -> Result<Self> {
        self.with_failure_callback_details(wrap_simple_failure_callback(callback))
    }

    /// Create a reclaim-capable failure-callback memory resource around this
    /// resource.
    ///
    /// The callback receives device, request, retry-attempt, and async-pool
    /// usage facts and returns a [`DeviceMemoryFailureAction`].
    pub fn with_failure_callback_details(
        &self,
        callback: Arc<dyn Fn(DeviceMemoryFailureInfo) -> DeviceMemoryFailureAction + Send + Sync + 'static>,
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
                    self.handle.raw.as_ptr(),
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
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        // SAFETY: handle is initialized to null and set by the C shim on success.
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_memory_resource_create_budget_tracking(
                    self.handle.raw.as_ptr(),
                    allocation_limit_bytes,
                    &mut handle,
                )
            },
            "memory::device_memory_resource::budget_tracking_limit",
            [
                ("allocation_limit_bytes", allocation_limit_bytes.to_string()),
                ("upstream_device", self.device().ordinal().to_string()),
            ],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            "memory::device_memory_resource::budget_tracking_limit",
            self.handle._failure_callbacks.clone(),
        )
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
        let mut usage = cudf_sys::cudf_memory_resource_usage_t::default();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::device_memory_resource::budget_tracking_usage",
        )?;
        Ok(usage.into())
    }

    /// Read physical CUDA async-pool counters for this resource.
    pub fn pool_usage(&self) -> Result<DeviceMemoryPoolUsage> {
        let mut usage = cudf_sys::cudf_memory_pool_usage_t::default();
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_resource_pool_usage(self.handle.raw.as_ptr(), &mut usage) },
            "memory::device_memory_resource::pool_usage",
            [("device", self.device().ordinal().to_string())],
        )?;
        Ok(usage.into())
    }

    /// Explicitly trim idle bytes from this resource's CUDA async pool.
    pub fn trim_to(&self, keep_bytes: u64) -> Result<()> {
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_memory_resource_trim_to(self.handle.raw.as_ptr(), keep_bytes) },
            "memory::device_memory_resource::trim_to",
            [("device", self.device().ordinal().to_string()), ("keep_bytes", keep_bytes.to_string())],
        )
    }

    /// The CUDA device this memory resource was created on.
    pub fn device(&self) -> DeviceId {
        self.handle.device
    }

    /// Borrow this owner as a low-level `cudf-sys` memory-resource handle.
    ///
    /// The returned wrapper does not transfer ownership. Its lifetime is tied
    /// to `&self`, and only `DeviceMemoryResource` remains responsible for
    /// destruction.
    pub fn as_borrowed_sys(&self) -> cudf_sys::BorrowedMemoryResource<'_> {
        unsafe { cudf_sys::BorrowedMemoryResource::from_raw(self.handle.raw.as_ptr()) }
    }

    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.handle.raw.as_ptr()
    }

    pub(crate) fn interop_raw_handle(&self) -> Result<*mut std::ffi::c_void> {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_memory_resource_raw(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::interop_raw_handle",
        )?;
        Ok(out)
    }
    pub(crate) fn __debug_raw_handle_value(&self) -> usize {
        let mut out = std::ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_raw(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::__debug_raw_handle_value",
        )
        .map(|_| out as usize)
        .unwrap_or(0)
    }
    pub(crate) fn __test_stats() -> Result<Self> {
        let device = DeviceId::current()?;
        let mut handle = ptr::null_mut();
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_create_stats(&mut handle) },
            "memory::device_memory_resource::__test_stats",
        )?;
        Ok(Self {
            handle: Arc::new(DeviceMemoryResourceHandle {
                raw: error::require_non_null(
                    handle,
                    "memory::device_memory_resource::__test_stats",
                    "stats memory resource",
                )?,
                device,
                _failure_callbacks: Vec::new(),
            }),
        })
    }
    pub(crate) fn __test_fail_first_allocation(&self) -> Result<Self> {
        let _guard = DeviceGuard::new(self.device())?;
        let mut handle = ptr::null_mut();
        error::check_status_in_with_details(
            unsafe {
                cudf_sys::cudf_test_memory_resource_create_fail_first_allocation(self.handle.raw.as_ptr(), &mut handle)
            },
            "memory::device_memory_resource::__test_fail_first_allocation",
            [("upstream_device", self.device().ordinal().to_string())],
        )?;
        Self::from_created_handle_with_callbacks(
            handle,
            self.device(),
            "memory::device_memory_resource::__test_fail_first_allocation",
            self.handle._failure_callbacks.clone(),
        )
    }
    pub(crate) fn __test_allocation_count(&self) -> Result<i64> {
        let mut out = 0_i64;
        error::check_status_in(
            unsafe { cudf_sys::cudf_test_memory_resource_allocation_count(self.handle.raw.as_ptr(), &mut out) },
            "memory::device_memory_resource::__test_allocation_count",
        )?;
        Ok(out)
    }
}

impl Drop for DeviceMemoryResourceHandle {
    fn drop(&mut self) {
        let status = unsafe { cudf_sys::cudf_memory_resource_destroy(self.raw.as_ptr(), self.device.ordinal_raw()) };
        if status != cudf_sys::CUDF_SUCCESS {
            if !self._failure_callbacks.is_empty() {
                // Native destroy can be rejected when the resource is installed
                // as current. In that case the native adaptor may still call
                // `callback_arg`, so leak only the callback states instead of
                // leaving dangling Rust pointers behind.
                std::mem::forget(std::mem::take(&mut self._failure_callbacks));
            }
            eprintln!(
                "[cudf::DeviceMemoryResourceHandle] warning: failed to destroy memory resource on device {}",
                self.device.ordinal_raw()
            );
        }
    }
}

impl std::fmt::Debug for DeviceMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryResource")
            .field("device", &self.device())
            .field("handle", &self.__debug_raw_handle_value())
            .finish()
    }
}
