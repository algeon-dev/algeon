//! Installation identity, native state records, and install outcomes.

use std::sync::Arc;

use super::{failure::DeviceMemoryFailureInfo, options::DeviceMemoryManagerOptions};
use crate::core::error::{self, Result};

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

/// Process-lifetime allocator configuration for one device.
///
/// `owner` requires `on_failure`. Identified reuse requires the same owner
/// and physical-base options; the first callback remains installed. Without
/// an owner, callback installation retains the original callback on reuse.
#[derive(Clone, Default)]
pub struct InstallSpec {
    pub options: DeviceMemoryManagerOptions,
    pub owner: Option<DeviceMemoryManagerInstallationIdentity>,
    /// Runs on the allocating thread. It must not re-enter the allocator,
    /// block on application locks, or unwind across the native boundary.
    pub on_failure: Option<Arc<dyn Fn(DeviceMemoryFailureInfo) + Send + Sync + 'static>>,
}

pub(super) fn installation_identity_to_sys(
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

/// Whether an explicit install created or reused the process manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryManagerInstallationOutcomeKind {
    Installed,
    Reused,
}

/// Result of one successful explicit installation transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryManagerInstallationOutcome {
    pub kind: DeviceMemoryManagerInstallationOutcomeKind,
    pub state: DeviceMemoryManagerInstallationState,
}

/// Failure from an explicit install together with the exact native
/// record returned by that same locked transaction, when one was available.
#[derive(Debug)]
pub struct DeviceMemoryManagerInstallationError {
    pub(super) error: error::Error,
    pub(super) state: Option<Box<DeviceMemoryManagerInstallationState>>,
}

impl From<error::Error> for DeviceMemoryManagerInstallationError {
    fn from(error: error::Error) -> Self {
        Self { error, state: None }
    }
}

impl DeviceMemoryManagerInstallationError {
    pub(super) fn with_state(error: error::Error, state: Option<DeviceMemoryManagerInstallationState>) -> Self {
        Self { error, state: state.map(Box::new) }
    }

    /// The ordinary cudf error for callers that do not need recovery state.
    pub fn error(&self) -> &error::Error {
        &self.error
    }

    /// The authoritative record returned by the rejected native transaction.
    ///
    /// This is `None` when Rust callback-ownership checks or native validation
    /// reject before the install transaction reaches its registry, or when
    /// the native record itself violates its ABI contract. A preliminary
    /// diagnostic inspection does not supply transactional installation state.
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

pub(super) fn installation_state_from_native_record(
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
