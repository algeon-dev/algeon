use super::{
    env::{DurationZeroPolicy, EnvValues},
    file::ServerAdmissionFileConfig,
    native::{duration_millis_from_value, optional_duration_millis_value, optional_native_u64, optional_native_usize},
    *,
};
use crate::backend::{GpuDeviceMemoryLimit, NexusGpuDeviceProfile, NexusGpuQueuePolicy};

pub const NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV: &str = "NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS";
pub const NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV: &str = "NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS";
pub const NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV: &str = "NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER";
pub const NEXUS_SERVER_GPU_DEVICES_ENV: &str = "NEXUS_SERVER_GPU_DEVICES";
pub(super) fn server_queue_policy(
    values: &mut EnvValues<'_>,
    file: &ServerAdmissionFileConfig,
) -> Result<NexusGpuQueuePolicy> {
    let queued = optional_native_usize(
        values,
        NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV,
        "admission.max_queued_attempts",
        file.max_queued_attempts,
    )?;
    let wait = optional_duration_millis_value(
        values,
        NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV,
        "admission.max_queue_wait_ms",
        file.max_queue_wait_ms,
    )?;
    let overtakes = optional_native_u64(
        values,
        NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV,
        "admission.max_overtakes_per_waiter",
        file.max_overtakes_per_waiter,
    )?;
    for value in [queued.as_ref().map(|v| (v.value, v.field))].into_iter().flatten() {
        if value.0 == 0 {
            return Err(ServerError::invalid_config(value.1, "must be greater than zero"));
        }
    }
    if let Some(value) = overtakes.as_ref()
        && value.value > u64::from(u32::MAX)
    {
        return Err(ServerError::invalid_config(value.field, "must fit in u32"));
    }
    let mut policy = NexusGpuQueuePolicy::default();
    if let Some(value) = queued {
        policy = policy.with_max_queued_attempts(value.value);
    }
    if let Some(value) = wait {
        policy = policy
            .with_max_queue_wait(duration_millis_from_value(value, DurationZeroPolicy::Reject)?.expect("rejects zero"));
    }
    if let Some(value) = overtakes {
        policy = policy.with_max_overtakes_per_waiter(value.value as u32);
    }
    Ok(policy)
}

/// Resolve the server's complete ordinal-keyed device envelope set.
///
/// `NEXUS_SERVER_GPU_DEVICES` remains a deployment selector, but it no
/// longer broadcasts one scalar resource policy. Every selected device,
/// including a single selected device, must name one TOML profile.
pub(super) fn server_device_profiles(
    file: &ServerAdmissionFileConfig,
    device_ordinals: &[usize],
) -> Result<Vec<NexusGpuDeviceProfile>> {
    if file.device_profiles.is_empty() {
        return Err(ServerError::invalid_config(
            "admission.device_profiles",
            "one explicit ordinal-keyed profile is required for every configured GPU device",
        )
        .with_fact("reason", "missing_device_profiles")
        .with_fact("device_ordinals", format_device_ordinals(device_ordinals)));
    }

    let selected = device_ordinals.iter().copied().collect::<std::collections::BTreeSet<_>>();
    let mut profiles = std::collections::BTreeMap::new();
    for raw in &file.device_profiles {
        let profile = profile_from_file(raw)?;
        let ordinal = profile.device_ordinal();
        if profiles.insert(ordinal, profile).is_some() {
            return Err(ServerError::invalid_config(
                "admission.device_profiles",
                format!("contains duplicate profile for device ordinal {ordinal}"),
            )
            .with_fact("reason", "duplicate_device")
            .with_fact("device_ordinal", ordinal));
        }
    }
    let configured = profiles.keys().copied().collect::<std::collections::BTreeSet<_>>();
    let missing = selected.difference(&configured).copied().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(ServerError::invalid_config(
            "admission.device_profiles",
            "must configure every device selected by NEXUS_SERVER_GPU_DEVICES",
        )
        .with_fact("reason", "missing_devices")
        .with_fact("device_ordinals", format_device_ordinals(&missing)));
    }
    let unconfigured = configured.difference(&selected).copied().collect::<Vec<_>>();
    if !unconfigured.is_empty() {
        return Err(ServerError::invalid_config(
            "admission.device_profiles",
            "must not configure a device outside NEXUS_SERVER_GPU_DEVICES",
        )
        .with_fact("reason", "unconfigured_devices")
        .with_fact("device_ordinals", format_device_ordinals(&unconfigured)));
    }
    Ok(device_ordinals
        .iter()
        .map(|ordinal| profiles.remove(ordinal).expect("validated selected profile exists"))
        .collect())
}

fn profile_from_file(raw: &super::file::ServerAdmissionDeviceProfileFileConfig) -> Result<NexusGpuDeviceProfile> {
    for (field, value) in [
        ("admission.device_profiles.memory_limit_bytes", raw.memory_limit_bytes),
        ("admission.device_profiles.attempt_floor_bytes", raw.attempt_floor_bytes),
    ] {
        if value == Some(0) {
            return Err(ServerError::invalid_config(field, "must be greater than zero"));
        }
    }
    for (field, value) in [
        ("admission.device_profiles.max_active_attempts", raw.max_active_attempts),
        ("admission.device_profiles.source_work_per_attempt", raw.source_work_per_attempt),
        ("admission.device_profiles.stream_pool_size", raw.stream_pool_size),
        ("admission.device_profiles.worker_count", raw.worker_count),
        ("admission.device_profiles.lane_count", raw.lane_count),
    ] {
        if value == Some(0) {
            return Err(ServerError::invalid_config(field, "must be greater than zero"));
        }
    }
    let memory_limit =
        raw.memory_limit_bytes.map(|limit_bytes| GpuDeviceMemoryLimit::Explicit { limit_bytes }).unwrap_or_default();
    let mut profile = NexusGpuDeviceProfile::new(raw.device_ordinal)
        .with_memory_limit(memory_limit)
        .with_backend_reserve_bytes(raw.backend_reserve_bytes.unwrap_or(0))
        .with_attempt_floor_bytes(raw.attempt_floor_bytes)
        .with_source_work_per_attempt(raw.source_work_per_attempt.unwrap_or(1))
        .with_max_active_attempts(raw.max_active_attempts.unwrap_or(1))
        .with_telemetry(raw.install_telemetry.unwrap_or(true))
        .with_diagnostics(raw.install_diagnostics.unwrap_or(true));
    match (raw.stream_pool_size, raw.worker_count, raw.lane_count) {
        (None, None, None) => {},
        (Some(stream_pool_size), Some(worker_count), Some(lane_count)) => {
            profile = profile.with_installation_bounds(stream_pool_size, worker_count, lane_count);
        },
        _ => {
            return Err(ServerError::invalid_config(
                "admission.device_profiles",
                "stream_pool_size, worker_count, and lane_count must be configured together",
            ));
        },
    }
    Ok(profile)
}

fn format_device_ordinals(device_ordinals: &[usize]) -> String {
    device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

pub(super) fn server_gpu_devices(values: &mut EnvValues<'_>) -> Result<Vec<usize>> {
    let Some(raw) = values.string(NEXUS_SERVER_GPU_DEVICES_ENV) else {
        return Ok(vec![0]);
    };
    let mut devices = Vec::new();
    for part in raw.split(',') {
        let ordinal = part.trim().parse::<usize>().map_err(|error| {
            ServerError::invalid_config(NEXUS_SERVER_GPU_DEVICES_ENV, format!("must contain integer ordinals: {error}"))
        })?;
        if devices.contains(&ordinal) {
            return Err(ServerError::invalid_config(NEXUS_SERVER_GPU_DEVICES_ENV, "must not contain duplicates"));
        }
        devices.push(ordinal);
    }
    Ok(devices)
}
