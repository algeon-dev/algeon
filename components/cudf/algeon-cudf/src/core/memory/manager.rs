//! Process-local device memory manager and current-resource routing.

use std::{
    collections::hash_map::Entry,
    ffi::c_void,
    ptr::{self, NonNull},
    sync::Arc,
};

use super::{
    domain::{
        QueryMemoryDomain, QueryMemoryDomainHandle, ReservationArena, ReservationArenaHandle, ReservationHandle,
        StreamReservationLease,
    },
    failure::{
        AllocatorEvent, AllocatorEventRead, AllocatorEventRingBounds, CallbackInstallationFailure,
        DeviceMemoryFailureInfo, FailureCallbackState, device_manager_failure_callbacks, failure_callback_trampoline,
        wrap_simple_failure_callback,
    },
    installation::{
        DeviceMemoryManagerFailureCallbackState, DeviceMemoryManagerInstallationCause,
        DeviceMemoryManagerInstallationError, DeviceMemoryManagerInstallationIdentity,
        DeviceMemoryManagerInstallationOutcome, DeviceMemoryManagerInstallationOutcomeKind,
        DeviceMemoryManagerInstallationRetry, DeviceMemoryManagerInstallationState,
        DeviceMemoryManagerInstallationStatus, installation_identity_to_sys, installation_state_from_native_record,
    },
    options::{DeviceMemoryManagerOptions, QueryMemoryDomainOptions, ensure_memory_manager_options_match},
    resource::{DeviceMemoryResource, record_native_teardown_failure},
    usage::DeviceMemoryPoolUsage,
};
use crate::{
    DeviceId,
    core::error::{self, Result},
    runtime::{self, device_guard::DeviceGuard, stream::CudaStream},
};

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
    pub(super) raw: Option<NonNull<cudf_sys::cudf_current_resource_route_handle_state>>,
    pub(super) device: DeviceId,
    // Keep the target owner, stream identity, and manager handle live while
    // native code can still allocate through the registered route.
    _target: DeviceMemoryResource,
    _stream: CudaStream,
    _manager: DeviceMemoryManager,
}

/// Lease routing current-resource allocations on a caller-owned raw CUDA stream.
///
/// The stream is borrowed, not retained. The caller must keep it alive until
/// this lease is dropped. Allocations made through the route retain their target
/// resource until deallocation.
#[must_use = "dropping the lease removes the raw stream route for future current-RMM allocations"]
#[derive(Debug)]
pub struct RawCurrentResourceRouteLease {
    pub(super) raw: Option<NonNull<cudf_sys::cudf_current_resource_route_handle_state>>,
    pub(super) device: DeviceId,
    _target: DeviceMemoryResource,
    _manager: DeviceMemoryManager,
}

impl Drop for RawCurrentResourceRouteLease {
    fn drop(&mut self) {
        let mut raw = self.raw.take().map_or(ptr::null_mut(), NonNull::as_ptr);
        runtime::drop_device_resource(&mut raw, self.device, "RawCurrentResourceRouteLease::drop", |owned| {
            let status = unsafe { cudf_sys::cudf_current_resource_route_destroy(owned) };
            if status != cudf_sys::CUDF_SUCCESS {
                record_native_teardown_failure(self.device, "RawCurrentResourceRouteLease::drop", status);
            }
        });
    }
}

impl DeviceMemoryManager {
    pub(super) fn raw_handle(&self) -> cudf_sys::cudf_memory_manager_handle {
        self.handle.raw.as_ptr()
    }

    /// Explicitly install the memory domain for `device` (idempotent).
    ///
    /// cudf owns the memory management of every CUDA device it uses: the
    /// first time a device's manager is created, its reservation-aware
    /// adaptor is installed as RMM's current resource for that device; and
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
    ///
    /// This is the installation entry point, not a handle lookup. Code that
    /// only needs to observe an already-installed manager (for example pool
    /// telemetry sampling) must use [`Self::for_device`]: calling an
    /// `install_*` variant there fails whenever the installed options differ,
    /// and swallowing that failure (for example with `.ok()`) silently
    /// reports every pool counter as zero instead of surfacing the misuse.
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
    ///
    /// This is the read/alias entry point for already-installed devices.
    /// Telemetry samplers and other observers must use this rather than any
    /// `install_*` variant, which rejects an existing manager whose effective
    /// options differ.
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

    /// Route legacy current-RMM allocations made on a borrowed raw CUDA stream to `resource`.
    ///
    /// The caller must keep `raw_stream` valid and alive until the returned
    /// lease is dropped. Native validation rejects null, default, invalid, or
    /// cross-device streams. This API is for native owners that expose a raw
    /// stream but cannot pass an explicit allocator.
    ///
    /// # Safety
    ///
    /// `raw_stream` must be a live CUDA stream owned by the caller, and must
    /// remain alive until the returned lease is dropped.
    pub unsafe fn route_current_resource_raw(
        &self,
        raw_stream: *mut c_void,
        resource: &DeviceMemoryResource,
    ) -> Result<RawCurrentResourceRouteLease> {
        let _guard = DeviceGuard::new(self.device())?;
        if resource.device() != self.device() {
            return Err(error::Error::new_with(
                error::ErrorCode::RuntimeBindingMismatch,
                "memory::device_memory_manager::route_current_resource_raw",
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
                cudf_sys::cudf_memory_manager_route_current_resource_raw(
                    self.handle.raw.as_ptr(),
                    raw_stream,
                    resource.as_raw(),
                    &mut raw,
                )
            },
            "memory::device_memory_manager::route_current_resource_raw",
            [("device", self.device().ordinal().to_string())],
        )?;

        Ok(RawCurrentResourceRouteLease {
            raw: NonNull::new(raw),
            device: self.device(),
            _target: resource.clone(),
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

    pub(super) fn reserve_stream_with_arena(
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
