//! Deployment IO capability probes for readiness and diagnostics.
//!
//! These types are readiness/reporting contracts. They do not select an
//! execution path by themselves.

use std::path::{Path, PathBuf};

/// Default Linux procfs path created by the `nvidia-fs` kernel module.
pub const NVIDIA_FS_PROC_PATH: &str = "/proc/driver/nvidia-fs";
/// Alternate sysfs module path that also indicates `nvidia-fs` is loaded.
pub const NVIDIA_FS_SYSFS_PATH: &str = "/sys/module/nvidia_fs";
/// Environment variable used by KvikIO to force compatibility/POSIX mode.
pub const KVIKIO_COMPAT_MODE_ENV: &str = "KVIKIO_COMPAT_MODE";

/// Deployment-level IO capability summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentIoCapability {
    /// Whether this process should treat GDS as available for eligible local sources.
    pub gds_available: bool,
    /// Stable reason for the deployment-level GDS availability decision.
    pub gds_status: GdsAvailabilityStatus,
    /// `nvidia-fs` module presence probe result.
    pub nvidia_fs_status: NvidiaFsProbeStatus,
    /// Parsed `KVIKIO_COMPAT_MODE` environment state.
    pub kvikio_compat_mode: KvikioCompatModeStatus,
    /// cuFile/libcufile driver-open probe result.
    pub gds_driver_probe_status: GdsDriverProbeStatus,
}

impl DeploymentIoCapability {
    /// Probes the current host and returns a non-failing diagnostic summary.
    ///
    /// Skips the cuFile driver probe when `nvidia-fs` is absent or KvikIO
    /// compatibility is forced. Per-query paths should use
    /// [`DeploymentIoCapability::cached_current_host`].
    pub fn probe_current_host() -> Self {
        let nvidia_fs_status = probe_nvidia_fs_status();
        let kvikio_compat_mode =
            KvikioCompatModeStatus::from_env_value(std::env::var(KVIKIO_COMPAT_MODE_ENV).ok().as_deref());
        configure_kvikio_compatibility(&nvidia_fs_status, &kvikio_compat_mode);
        let gds_driver_probe_status =
            probe_cudf_gds_driver_if_eligible(&nvidia_fs_status, &kvikio_compat_mode, probe_cudf_gds_driver);
        Self::from_probe_inputs(nvidia_fs_status, kvikio_compat_mode, gds_driver_probe_status)
    }

    /// Returns a process-wide cached host capability summary, probing once.
    ///
    /// Treats deployment IO state as process-static and avoids repeating the
    /// expensive cuFile driver probe on each execution-context build.
    pub fn cached_current_host() -> Self {
        static CACHED: std::sync::OnceLock<DeploymentIoCapability> = std::sync::OnceLock::new();
        CACHED.get_or_init(Self::probe_current_host).clone()
    }

    /// Builds a capability summary from already-collected pure probe inputs.
    pub fn from_probe_inputs(
        nvidia_fs_status: NvidiaFsProbeStatus,
        kvikio_compat_mode: KvikioCompatModeStatus,
        gds_driver_probe_status: GdsDriverProbeStatus,
    ) -> Self {
        let gds_status = gds_availability_status(&nvidia_fs_status, &kvikio_compat_mode, &gds_driver_probe_status);
        Self {
            gds_available: gds_status == GdsAvailabilityStatus::Available,
            gds_status,
            nvidia_fs_status,
            kvikio_compat_mode,
            gds_driver_probe_status,
        }
    }

    /// Stable report facts for runtime/service readiness diagnostics.
    pub fn diagnostic_facts(&self) -> Vec<(&'static str, String)> {
        let mut facts = vec![
            ("gds_available", self.gds_available.to_string()),
            ("gds_status", self.gds_status.as_str().to_string()),
            ("nvidia_fs_status", self.nvidia_fs_status.as_str().to_string()),
            ("kvikio_compat_mode", self.kvikio_compat_mode.as_str().to_string()),
            ("gds_driver_probe_status", self.gds_driver_probe_status.as_str().to_string()),
        ];
        if let Some(path) = self.nvidia_fs_status.path() {
            facts.push(("nvidia_fs_path", path.to_string_lossy().into_owned()));
        }
        if let Some(value) = self.kvikio_compat_mode.raw_value() {
            facts.push(("kvikio_compat_mode_raw", value.to_string()));
        }
        if let GdsDriverProbeStatus::Unavailable { code, operation, message } = &self.gds_driver_probe_status {
            facts.push(("gds_driver_probe_code", code.clone()));
            facts.push(("gds_driver_probe_operation", operation.clone()));
            facts.push(("gds_driver_probe_message", message.clone()));
        }
        facts
    }
}

impl Default for DeploymentIoCapability {
    fn default() -> Self {
        Self::probe_current_host()
    }
}

/// Stable deployment-level GDS availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdsAvailabilityStatus {
    /// All deployment-level probes succeeded.
    Available,
    /// GDS is unavailable because `nvidia-fs` was not detected.
    DriverMissing,
    /// GDS is unavailable because KvikIO compatibility mode is forced on.
    CompatModeForced,
    /// GDS is unavailable because the cuFile/libcufile probe boundary is not implemented.
    ProbeUnimplemented,
    /// GDS is unavailable because the cuFile/libcufile driver probe failed.
    DriverUnavailable,
}

impl GdsAvailabilityStatus {
    /// Stable report code for this status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::DriverMissing => "driver_missing",
            Self::CompatModeForced => "compat_mode_forced",
            Self::ProbeUnimplemented => "probe_unimplemented",
            Self::DriverUnavailable => "driver_unavailable",
        }
    }
}

/// Result of checking whether `nvidia-fs` is loaded.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NvidiaFsProbeStatus {
    /// A known `nvidia-fs` procfs/sysfs path exists.
    Present {
        /// Path that proved `nvidia-fs` presence.
        path: PathBuf,
    },
    /// No known `nvidia-fs` procfs/sysfs path exists.
    DriverMissing,
}

impl NvidiaFsProbeStatus {
    /// Stable report code for this status.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Present { .. } => "present",
            Self::DriverMissing => "driver_missing",
        }
    }

    /// Path that proved module presence, when available.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Present { path } => Some(path.as_path()),
            Self::DriverMissing => None,
        }
    }
}

/// Parsed `KVIKIO_COMPAT_MODE` environment state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KvikioCompatModeStatus {
    /// The environment variable is unset or empty.
    Unset,
    /// The environment variable explicitly forces compatibility/POSIX mode on.
    ForcedOn {
        /// Raw environment variable value.
        raw_value: String,
    },
    /// The environment variable explicitly forces compatibility/POSIX mode off.
    ForcedOff {
        /// Raw environment variable value.
        raw_value: String,
    },
    /// The environment variable is set to a value this crate does not interpret.
    Unrecognized {
        /// Raw environment variable value.
        raw_value: String,
    },
}

impl KvikioCompatModeStatus {
    /// Parses a raw `KVIKIO_COMPAT_MODE` value without reading process env.
    pub fn from_env_value(value: Option<&str>) -> Self {
        let Some(raw_value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
            return Self::Unset;
        };
        match raw_value.to_ascii_uppercase().as_str() {
            "ON" | "1" | "TRUE" | "YES" => Self::ForcedOn { raw_value: raw_value.to_string() },
            "OFF" | "0" | "FALSE" | "NO" => Self::ForcedOff { raw_value: raw_value.to_string() },
            _ => Self::Unrecognized { raw_value: raw_value.to_string() },
        }
    }

    /// Stable report code for this status.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Unset => "unset",
            Self::ForcedOn { .. } => "forced_on",
            Self::ForcedOff { .. } => "forced_off",
            Self::Unrecognized { .. } => "unrecognized",
        }
    }

    /// Whether compatibility/POSIX mode is forced on.
    pub const fn is_forced_on(&self) -> bool {
        matches!(self, Self::ForcedOn { .. })
    }

    /// Raw environment variable value, when one was provided.
    pub fn raw_value(&self) -> Option<&str> {
        match self {
            Self::Unset => None,
            Self::ForcedOn { raw_value } | Self::ForcedOff { raw_value } | Self::Unrecognized { raw_value } => {
                Some(raw_value.as_str())
            },
        }
    }
}

/// Result of probing the cuFile/libcufile driver boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GdsDriverProbeStatus {
    /// The cuFile driver-open probe succeeded.
    Available,
    /// No clean cudf/libcudf probe boundary exists in this build.
    ProbeUnimplemented,
    /// The probe was skipped because `nvidia-fs` is absent.
    SkippedDriverMissing,
    /// The probe was skipped because compatibility mode is forced on.
    SkippedCompatModeForced,
    /// The cuFile driver-open probe failed.
    Unavailable {
        /// Stable cudf error code.
        code: String,
        /// cudf operation where the failure surfaced.
        operation: String,
        /// One-line cudf error message.
        message: String,
    },
}

impl GdsDriverProbeStatus {
    /// Stable report code for this status.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::ProbeUnimplemented => "probe_unimplemented",
            Self::SkippedDriverMissing => "skipped_driver_missing",
            Self::SkippedCompatModeForced => "skipped_compat_mode_forced",
            Self::Unavailable { .. } => "unavailable",
        }
    }
}

fn probe_nvidia_fs_status() -> NvidiaFsProbeStatus {
    probe_nvidia_fs_status_from_paths([NVIDIA_FS_PROC_PATH, NVIDIA_FS_SYSFS_PATH].iter().map(Path::new))
}

fn probe_nvidia_fs_status_from_paths<'a>(paths: impl IntoIterator<Item = &'a Path>) -> NvidiaFsProbeStatus {
    paths
        .into_iter()
        .find(|path| path.exists())
        .map(|path| NvidiaFsProbeStatus::Present { path: path.to_path_buf() })
        .unwrap_or(NvidiaFsProbeStatus::DriverMissing)
}

fn probe_cudf_gds_driver() -> GdsDriverProbeStatus {
    match cudf::io::gds_driver_available() {
        Ok(()) => GdsDriverProbeStatus::Available,
        Err(error) => GdsDriverProbeStatus::Unavailable {
            code: error.code().to_string(),
            operation: error.operation().to_string(),
            message: error.message().to_string(),
        },
    }
}

fn probe_cudf_gds_driver_if_eligible(
    nvidia_fs_status: &NvidiaFsProbeStatus,
    kvikio_compat_mode: &KvikioCompatModeStatus,
    probe: impl FnOnce() -> GdsDriverProbeStatus,
) -> GdsDriverProbeStatus {
    if matches!(nvidia_fs_status, NvidiaFsProbeStatus::DriverMissing) {
        return GdsDriverProbeStatus::SkippedDriverMissing;
    }
    if kvikio_compat_mode.is_forced_on() {
        return GdsDriverProbeStatus::SkippedCompatModeForced;
    }
    probe()
}

fn configure_kvikio_compatibility(nvidia_fs_status: &NvidiaFsProbeStatus, kvikio_compat_mode: &KvikioCompatModeStatus) {
    if !matches!(nvidia_fs_status, NvidiaFsProbeStatus::DriverMissing) && !kvikio_compat_mode.is_forced_on() {
        return;
    }
    if let Err(error) = cudf::io::set_kvikio_compat_mode(cudf::io::KvikioCompatMode::On) {
        eprintln!("[nexus-query-engine] warning: failed to configure KvikIO compatibility mode: {error}");
    }
}

fn gds_availability_status(
    nvidia_fs_status: &NvidiaFsProbeStatus,
    kvikio_compat_mode: &KvikioCompatModeStatus,
    gds_driver_probe_status: &GdsDriverProbeStatus,
) -> GdsAvailabilityStatus {
    if matches!(nvidia_fs_status, NvidiaFsProbeStatus::DriverMissing) {
        return GdsAvailabilityStatus::DriverMissing;
    }
    if kvikio_compat_mode.is_forced_on() {
        return GdsAvailabilityStatus::CompatModeForced;
    }
    match gds_driver_probe_status {
        GdsDriverProbeStatus::Available => GdsAvailabilityStatus::Available,
        GdsDriverProbeStatus::ProbeUnimplemented => GdsAvailabilityStatus::ProbeUnimplemented,
        GdsDriverProbeStatus::SkippedDriverMissing => GdsAvailabilityStatus::DriverMissing,
        GdsDriverProbeStatus::SkippedCompatModeForced => GdsAvailabilityStatus::CompatModeForced,
        GdsDriverProbeStatus::Unavailable { .. } => GdsAvailabilityStatus::DriverUnavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kvikio_compat_mode_detection_distinguishes_forced_on_off_and_unset() {
        assert_eq!(KvikioCompatModeStatus::from_env_value(None), KvikioCompatModeStatus::Unset);
        assert_eq!(KvikioCompatModeStatus::from_env_value(Some(" ")), KvikioCompatModeStatus::Unset);
        assert!(KvikioCompatModeStatus::from_env_value(Some("ON")).is_forced_on());
        assert!(KvikioCompatModeStatus::from_env_value(Some("true")).is_forced_on());
        assert_eq!(
            KvikioCompatModeStatus::from_env_value(Some("off")).as_str(),
            KvikioCompatModeStatus::ForcedOff { raw_value: "off".to_string() }.as_str()
        );
        assert_eq!(KvikioCompatModeStatus::from_env_value(Some("AUTO")).as_str(), "unrecognized");
    }

    #[test]
    fn deployment_probe_inputs_distinguish_missing_driver_forced_compat_and_unimplemented_probe() {
        let driver_missing = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::Available,
        );
        assert!(!driver_missing.gds_available);
        assert_eq!(driver_missing.gds_status.as_str(), "driver_missing");

        let compat_forced = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::Present { path: PathBuf::from(NVIDIA_FS_PROC_PATH) },
            KvikioCompatModeStatus::ForcedOn { raw_value: "ON".to_string() },
            GdsDriverProbeStatus::Available,
        );
        assert!(!compat_forced.gds_available);
        assert_eq!(compat_forced.gds_status.as_str(), "compat_mode_forced");

        let probe_unimplemented = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::Present { path: PathBuf::from(NVIDIA_FS_PROC_PATH) },
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        assert!(!probe_unimplemented.gds_available);
        assert_eq!(probe_unimplemented.gds_status.as_str(), "probe_unimplemented");
    }

    #[test]
    fn diagnostic_facts_include_stable_gds_available_and_failure_status() {
        let capability = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::ForcedOn { raw_value: "ON".to_string() },
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let facts = capability.diagnostic_facts();
        assert!(facts.contains(&("gds_available", "false".to_string())));
        assert!(facts.contains(&("gds_status", "driver_missing".to_string())));
        assert!(facts.contains(&("kvikio_compat_mode", "forced_on".to_string())));
        assert!(facts.contains(&("gds_driver_probe_status", "probe_unimplemented".to_string())));
    }

    #[test]
    fn nvidia_fs_probe_accepts_proc_or_sysfs_presence() {
        let missing = probe_nvidia_fs_status_from_paths([Path::new("/definitely/missing/nvidia-fs")]);
        assert_eq!(missing, NvidiaFsProbeStatus::DriverMissing);
        let present = probe_nvidia_fs_status_from_paths([Path::new(".")]);
        assert!(matches!(present, NvidiaFsProbeStatus::Present { .. }));
    }

    #[test]
    fn gds_driver_probe_is_skipped_before_cufile_when_ineligible() {
        let mut called = false;
        let status = probe_cudf_gds_driver_if_eligible(
            &NvidiaFsProbeStatus::DriverMissing,
            &KvikioCompatModeStatus::Unset,
            || {
                called = true;
                GdsDriverProbeStatus::Available
            },
        );
        assert_eq!(status, GdsDriverProbeStatus::SkippedDriverMissing);
        assert!(!called);

        let status = probe_cudf_gds_driver_if_eligible(
            &NvidiaFsProbeStatus::Present { path: PathBuf::from(NVIDIA_FS_PROC_PATH) },
            &KvikioCompatModeStatus::ForcedOn { raw_value: "ON".to_string() },
            || {
                called = true;
                GdsDriverProbeStatus::Available
            },
        );
        assert_eq!(status, GdsDriverProbeStatus::SkippedCompatModeForced);
        assert!(!called);
    }
}
