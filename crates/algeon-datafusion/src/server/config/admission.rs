use super::{
    env::EnvValues,
    file::ServerAdmissionFileConfig,
    native::{duration_millis_from_value, optional_duration_millis_value, optional_native_usize},
    *,
};
use crate::server::backend::{
    AlgeonGpuDeviceInstallationBounds, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
};

pub const ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV: &str = "ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS";
pub const ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV: &str = "ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS";
pub const ALGEON_SERVER_GPU_DEVICES_ENV: &str = "ALGEON_SERVER_GPU_DEVICES";
pub(super) fn server_queue_policy(
    values: &mut EnvValues<'_>,
    file: &ServerAdmissionFileConfig,
) -> Result<AlgeonGpuQueuePolicy> {
    let queued = optional_native_usize(
        values,
        ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV,
        "admission.max_queued_attempts",
        file.max_queued_attempts,
    )?;
    let wait = optional_duration_millis_value(
        values,
        ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV,
        "admission.max_queue_wait_ms",
        file.max_queue_wait_ms,
    )?;
    for value in [queued.as_ref().map(|v| (v.value, v.field))].into_iter().flatten() {
        if value.0 == 0 {
            return Err(ServerError::invalid_config(value.1, "must be greater than zero"));
        }
    }
    let mut policy = AlgeonGpuQueuePolicy::default();
    if let Some(value) = queued {
        policy = policy.with_max_queued_attempts(value.value);
    }
    if let Some(value) = wait {
        policy = policy.with_max_queue_wait(duration_millis_from_value(value)?);
    }
    Ok(policy)
}

/// Resolve the server's complete ordinal-keyed device envelope set.
///
/// `ALGEON_SERVER_GPU_DEVICES` remains a deployment selector, but it no
/// longer broadcasts one scalar resource policy. Every selected device,
/// including a single selected device, must name one TOML profile.
pub(super) fn server_device_profiles(
    file: &ServerAdmissionFileConfig,
    device_ordinals: &[usize],
) -> Result<Vec<AlgeonGpuDeviceProfile>> {
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
            "must configure every device selected by ALGEON_SERVER_GPU_DEVICES",
        )
        .with_fact("reason", "missing_devices")
        .with_fact("device_ordinals", format_device_ordinals(&missing)));
    }
    let unconfigured = configured.difference(&selected).copied().collect::<Vec<_>>();
    if !unconfigured.is_empty() {
        return Err(ServerError::invalid_config(
            "admission.device_profiles",
            "must not configure a device outside ALGEON_SERVER_GPU_DEVICES",
        )
        .with_fact("reason", "unconfigured_devices")
        .with_fact("device_ordinals", format_device_ordinals(&unconfigured)));
    }
    Ok(device_ordinals
        .iter()
        .map(|ordinal| profiles.remove(ordinal).expect("validated selected profile exists"))
        .collect())
}

fn profile_from_file(raw: &super::file::ServerAdmissionDeviceProfileFileConfig) -> Result<AlgeonGpuDeviceProfile> {
    for (field, value) in [
        ("admission.device_profiles.memory_limit_bytes", raw.memory_limit_bytes),
        ("admission.device_profiles.default_query_limit_bytes", raw.default_query_limit_bytes),
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
    // Automatic capacity bakes in driver headroom; an explicit limit does not,
    // so an explicit limit must declare its reserve (0 is a deliberate choice).
    if raw.memory_limit_bytes.is_some() && raw.backend_reserve_bytes.is_none() {
        return Err(ServerError::invalid_config(
            "admission.device_profiles.backend_reserve_bytes",
            "is required when memory_limit_bytes is explicit; declare 0 to run with no backend headroom",
        ));
    }
    // Start from the root projection of engine defaults and apply only values
    // explicitly selected by the server file. This must not restate device
    // controller defaults such as the automatic query limit, two active attempts, or
    // one source-work slot.
    let mut profile = AlgeonGpuDeviceProfile::new(raw.device_ordinal);
    if let Some(limit_bytes) = raw.memory_limit_bytes {
        profile = profile.with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes });
    }
    if let Some(backend_reserve_bytes) = raw.backend_reserve_bytes {
        profile = profile.with_backend_reserve_bytes(backend_reserve_bytes);
    }
    if let Some(cache_cap_bytes) = raw.cache_cap_bytes {
        profile = profile.with_cache_cap_bytes(cache_cap_bytes);
    }
    if let Some(default_query_limit_bytes) = raw.default_query_limit_bytes {
        profile = profile.with_default_query_limit_bytes(default_query_limit_bytes);
    }
    if let Some(source_work_per_attempt) = raw.source_work_per_attempt {
        profile = profile.with_source_work_per_attempt(source_work_per_attempt);
    }
    if let Some(max_active_attempts) = raw.max_active_attempts {
        profile = profile.with_max_active_attempts(max_active_attempts);
    }
    if let Some(install_telemetry) = raw.install_telemetry {
        profile = profile.with_telemetry(install_telemetry);
    }
    match (raw.stream_pool_size, raw.worker_count, raw.lane_count) {
        (None, None, None) => {},
        (Some(stream_pool_size), Some(worker_count), Some(lane_count)) => {
            profile = profile.with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(
                stream_pool_size,
                worker_count,
                lane_count,
            ));
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
    let Some(raw) = values.string(ALGEON_SERVER_GPU_DEVICES_ENV) else {
        return Ok(vec![0]);
    };
    let mut devices = Vec::new();
    for part in raw.split(',') {
        let ordinal = part.trim().parse::<usize>().map_err(|error| {
            ServerError::invalid_config(
                ALGEON_SERVER_GPU_DEVICES_ENV,
                format!("must contain integer ordinals: {error}"),
            )
        })?;
        if devices.contains(&ordinal) {
            return Err(ServerError::invalid_config(ALGEON_SERVER_GPU_DEVICES_ENV, "must not contain duplicates"));
        }
        devices.push(ordinal);
    }
    Ok(devices)
}
