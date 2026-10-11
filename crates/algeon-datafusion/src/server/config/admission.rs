use super::{
    file::{ServerAdmissionDeviceProfileFileConfig, ServerAdmissionFileConfig},
    *,
};
use crate::server::backend::{AlgeonGpuDeviceInstallationBounds, GpuDeviceMemoryLimit};

pub(super) fn server_queue_policy(
    file: &ServerAdmissionFileConfig,
    sources: &ConfigSources,
) -> Result<AlgeonGpuQueuePolicy> {
    let mut policy = AlgeonGpuQueuePolicy::default();
    if let Some(queued) = file.max_queued_attempts {
        if queued == 0 {
            return Err(sources.invalid("admission.max_queued_attempts", "must be greater than zero"));
        }
        policy = policy.with_max_queued_attempts(queued);
    }
    if let Some(wait_ms) = file.max_queue_wait_ms {
        if wait_ms == 0 {
            return Err(sources.invalid("admission.max_queue_wait_ms", "must be greater than zero"));
        }
        policy = policy.with_max_queue_wait(Duration::from_millis(wait_ms));
    }
    Ok(policy)
}

/// One profile per `[[admission.device_profiles]]` entry, in file order. The
/// profile ordinals are the selected GPUs. Without any entry the server
/// selects GPU 0 with an all-default profile and reports `true`; a default
/// profile carries no scalar setting, so no value is applied to several GPUs.
pub(super) fn server_device_profiles(file: &ServerAdmissionFileConfig) -> Result<(Vec<AlgeonGpuDeviceProfile>, bool)> {
    if file.device_profiles.is_empty() {
        return Ok((vec![AlgeonGpuDeviceProfile::new(0)], true));
    }
    let profiles = file.device_profiles.iter().map(profile_from_file).collect::<Result<Vec<_>>>()?;
    Ok((profiles, false))
}

fn profile_from_file(raw: &ServerAdmissionDeviceProfileFileConfig) -> Result<AlgeonGpuDeviceProfile> {
    for (field, value) in [
        ("admission.device_profiles.memory_limit_bytes", raw.memory_limit_bytes),
        ("admission.device_profiles.default_query_limit_bytes", raw.default_query_limit_bytes),
    ] {
        if value == Some(0) {
            return Err(profile_error(raw, field, "must be greater than zero"));
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
            return Err(profile_error(raw, field, "must be greater than zero"));
        }
    }
    // Automatic capacity includes the driver reserve; an explicit limit does not,
    // so an explicit limit must declare its reserve (0 is a deliberate choice).
    if raw.memory_limit_bytes.is_some() && raw.backend_reserve_bytes.is_none() {
        return Err(profile_error(
            raw,
            "admission.device_profiles.backend_reserve_bytes",
            "is required when memory_limit_bytes is explicit; declare 0 to run with no backend headroom",
        ));
    }
    // Start from the engine defaults and apply only the values the file sets,
    // so the automatic query limit, active attempts and source-work slots keep
    // one definition in the engine.
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
            return Err(profile_error(
                raw,
                "admission.device_profiles",
                "sets only some of stream_pool_size, worker_count and lane_count; set all three or none",
            ));
        },
    }
    Ok(profile)
}

fn profile_error(raw: &ServerAdmissionDeviceProfileFileConfig, field: &'static str, reason: &str) -> ServerError {
    ServerError::invalid_config(field, format!("{reason} (entry for GPU {})", raw.device_ordinal))
        .with_fact("device_ordinal", raw.device_ordinal)
}
