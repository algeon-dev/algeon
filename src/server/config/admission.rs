use std::num::NonZeroUsize;

use super::{
    env::{DurationZeroPolicy, EnvValues, NativeConfigValue},
    file::{ServerAdmissionFileConfig, ServerHostExecutionQosFileConfig, ServerMemoryShareFileValue},
    native::{
        duration_millis_from_value, optional_duration_millis, optional_duration_millis_value, optional_native_string,
        optional_native_usize, parse_memory_share, parse_query_placement_policy, parse_slot_guarantee,
        placement_policy_field_for_pinned_source,
    },
    *,
};

/// Maximum number of GPU-native queries admitted at once. Defaults come from
/// `NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE`; set a larger value only when the
/// GPU has enough memory for concurrent work.
pub const NEXUS_SERVER_MAX_IN_FLIGHT_ENV: &str = "NEXUS_SERVER_MAX_IN_FLIGHT";
/// Preferred maximum number of GPU-native queries admitted at once. Defaults
/// come from `NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE`;
/// `NEXUS_SERVER_MAX_IN_FLIGHT` remains accepted as a compatibility alias.
pub const NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV: &str = "NEXUS_SERVER_MAX_CONCURRENT_QUERIES";
/// Admission default profile. `balanced` preserves conservative legacy
/// defaults; `small_query` defaults admission to eight optimistic queries per
/// configured GPU and a 1 GiB minimum grant.
pub const NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV: &str = "NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE";
/// Minimum per-query admission grant. Accepts a byte integer or fractional
/// capacity such as `0.125`. Defaults come from
/// `NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE`.
pub const NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV: &str = "NEXUS_SERVER_ADMISSION_GRANT_MIN";
/// Preferred per-query admission grant. Accepts a byte integer or fractional
/// capacity such as `0.5`. Defaults are derived by backend admission policy.
pub const NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV: &str = "NEXUS_SERVER_ADMISSION_GRANT_WANT";
/// Empty-slot grant policy: `guaranteed_minimum` or `optimistic`.
pub const NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV: &str = "NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE";
/// Query placement policy across configured GPU devices: `least_loaded`,
/// `best_fit`, or `pinned`.
pub const NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV: &str = "NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY";
/// CUDA device ordinal used when admission placement policy is `pinned`.
pub const NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV: &str = "NEXUS_SERVER_ADMISSION_PINNED_DEVICE";
/// Optional maximum number of GPU-native queries admitted on one CUDA device.
pub const NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV: &str = "NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE";
/// Maximum time a GPU-native query waits in query-service admission before a
/// structured overload error. Defaults to `60000`.
pub const NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV: &str = "NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS";
/// Maximum concurrent host-executing statements. Defaults to `2`; set `0` to
/// disable host execution QoS.
pub const NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS_ENV: &str =
    "NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS";
/// Maximum time a statement waits for host execution QoS. Defaults to
/// `30000`; set `0` to wait indefinitely.
pub const NEXUS_SERVER_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS_ENV: &str = "NEXUS_SERVER_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS";
/// Fraction of a GPU's total memory eligible for server native-query budgets.
/// The fixed native headroom remains a floor, so `1.0` preserves the historical
/// auto-sized `total - headroom` behavior.
pub const NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV: &str = "NEXUS_SERVER_DEVICE_BUDGET_FRACTION";
/// Comma-separated CUDA device ordinals available to the Flight SQL server.
/// Independent queries are placed across this set by the native query service.
pub const NEXUS_SERVER_GPU_DEVICES_ENV: &str = "NEXUS_SERVER_GPU_DEVICES";
const DEFAULT_MAX_IN_FLIGHT: usize = 2;
const SMALL_QUERY_PROFILE_MAX_QUERIES_PER_DEVICE: usize = 8;
const SMALL_QUERY_PROFILE_GRANT_MIN_BYTES: u64 = 1 << 30;
const DEFAULT_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS: usize = 2;
const DEFAULT_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy)]
pub(super) struct ServerAdmissionPlacementConfig {
    pub(super) max_queries_per_device: Option<NativeConfigValue<usize>>,
    pub(super) placement_policy: QueryPlacementPolicy,
    pub(super) placement_policy_field: Option<&'static str>,
    pub(super) pinned_device_ordinal_field: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub(super) struct ServerAdmissionPolicyConfig {
    pub(super) policy: NexusAdmissionPolicy,
    pub(super) grant_min_field: Option<&'static str>,
    pub(super) grant_want_field: Option<&'static str>,
    pub(super) slot_guarantee_field: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ServerAdmissionWorkloadProfile {
    Balanced,
    SmallQuery,
}

impl ServerAdmissionWorkloadProfile {
    fn parse(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "balanced" => Ok(Self::Balanced),
            "small_query" | "small-query" => Ok(Self::SmallQuery),
            other => Err(format!("must be `balanced` or `small_query`, got `{other}`")),
        }
    }

    const fn default_max_concurrent_queries(self, placement_eligible_device_count: usize) -> usize {
        match self {
            Self::Balanced => DEFAULT_MAX_IN_FLIGHT,
            Self::SmallQuery => SMALL_QUERY_PROFILE_MAX_QUERIES_PER_DEVICE * placement_eligible_device_count,
        }
    }

    const fn default_grant_min(self) -> Option<MemoryShare> {
        match self {
            Self::Balanced => None,
            Self::SmallQuery => Some(MemoryShare::Bytes(SMALL_QUERY_PROFILE_GRANT_MIN_BYTES)),
        }
    }

    const fn default_slot_guarantee(self) -> SlotGuarantee {
        match self {
            Self::Balanced => SlotGuarantee::GuaranteedMinimum,
            Self::SmallQuery => SlotGuarantee::Optimistic,
        }
    }
}
pub(super) fn server_admission_workload_profile(
    values: &mut EnvValues<'_>,
    file: &ServerAdmissionFileConfig,
) -> Result<NativeConfigValue<ServerAdmissionWorkloadProfile>> {
    let selected = optional_native_string(
        values,
        NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV,
        "admission.workload_profile",
        file.workload_profile.as_deref(),
    );
    match selected {
        Some(raw) => ServerAdmissionWorkloadProfile::parse(&raw.value)
            .map(|profile| NativeConfigValue::new(profile, raw.field))
            .map_err(|reason| ServerError::invalid_config(raw.field, reason)),
        None => Ok(NativeConfigValue::new(
            ServerAdmissionWorkloadProfile::Balanced,
            NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV,
        )),
    }
}

pub(super) fn server_max_concurrent_queries(
    values: &mut EnvValues<'_>,
    file: &ServerAdmissionFileConfig,
    workload_profile: ServerAdmissionWorkloadProfile,
    placement_eligible_device_count: usize,
) -> Result<NativeConfigValue<usize>> {
    let preferred = values
        .optional_usize(NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV)?
        .map(|value| NativeConfigValue::new(value, NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV));
    let legacy = values
        .optional_usize(NEXUS_SERVER_MAX_IN_FLIGHT_ENV)?
        .map(|value| NativeConfigValue::new(value, NEXUS_SERVER_MAX_IN_FLIGHT_ENV));
    let selected = match (preferred, legacy) {
        (Some(preferred), Some(legacy)) if preferred.value != legacy.value => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV,
                format!(
                    "conflicts with {NEXUS_SERVER_MAX_IN_FLIGHT_ENV}; set only one admission limit or use the same value"
                ),
            ));
        },
        (Some(preferred), _) => preferred,
        (None, Some(legacy)) => legacy,
        (None, None) => file
            .max_concurrent_queries
            .map(|value| NativeConfigValue::new(value, "admission.max_concurrent_queries"))
            .unwrap_or(NativeConfigValue::new(
                workload_profile.default_max_concurrent_queries(placement_eligible_device_count),
                NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV,
            )),
    };
    if selected.value == 0 {
        return Err(ServerError::invalid_config(selected.field, "must be greater than zero"));
    }
    Ok(selected)
}

pub(super) fn server_admission_placement_config(
    values: &mut EnvValues<'_>,
    file: &ServerAdmissionFileConfig,
    gpu_devices: &[usize],
) -> Result<ServerAdmissionPlacementConfig> {
    let max_queries_per_device = optional_native_usize(
        values,
        NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV,
        "admission.max_queries_per_device",
        file.max_queries_per_device,
    )?;
    if let Some(value) = max_queries_per_device
        && value.value == 0
    {
        return Err(ServerError::invalid_config(value.field, "must be greater than zero"));
    }
    let pinned_device_ordinal = optional_native_usize(
        values,
        NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV,
        "admission.pinned_device_ordinal",
        file.pinned_device_ordinal,
    )?;
    let placement_policy = optional_native_string(
        values,
        NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV,
        "admission.placement_policy",
        file.placement_policy.as_deref(),
    );
    let placement_policy_env_override =
        placement_policy.as_ref().is_some_and(|raw| raw.field == NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV);
    let pinned_device_ordinal_from_file =
        pinned_device_ordinal.as_ref().is_some_and(|value| value.field == "admission.pinned_device_ordinal");
    let placement_policy_field = placement_policy.as_ref().map(|raw| raw.field);
    let placement_policy = placement_policy
        .map(|raw| parse_query_placement_policy(raw.field, &raw.value, pinned_device_ordinal))
        .transpose()?
        .unwrap_or_default();
    if let QueryPlacementPolicy::Pinned { device_ordinal } = placement_policy
        && !gpu_devices.contains(&device_ordinal)
    {
        return Err(ServerError::invalid_config(
            pinned_device_ordinal.map_or(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, |value| value.field),
            format!("pinned device must be one of {NEXUS_SERVER_GPU_DEVICES_ENV}"),
        ));
    }
    if let Some(pinned_device_ordinal) = pinned_device_ordinal
        && !matches!(placement_policy, QueryPlacementPolicy::Pinned { .. })
        && !(placement_policy_env_override && pinned_device_ordinal_from_file)
    {
        let conflicting_field = placement_policy_field
            .unwrap_or_else(|| placement_policy_field_for_pinned_source(pinned_device_ordinal.field));
        return Err(ServerError::invalid_config(
            pinned_device_ordinal.field,
            format!("requires {conflicting_field}=pinned"),
        )
        .with_fact("reason", "pinned_device_requires_pinned_placement")
        .with_fact("conflicting_config_field", conflicting_field));
    }
    let pinned_device_ordinal_field = pinned_device_ordinal
        .filter(|_| !(placement_policy_env_override && pinned_device_ordinal_from_file))
        .map(|value| value.field);
    Ok(ServerAdmissionPlacementConfig {
        max_queries_per_device,
        placement_policy,
        placement_policy_field,
        pinned_device_ordinal_field,
    })
}

pub(super) fn server_admission_policy(
    values: &mut EnvValues<'_>,
    file: &ServerAdmissionFileConfig,
    max_in_flight: usize,
    workload_profile: NativeConfigValue<ServerAdmissionWorkloadProfile>,
    placement: ServerAdmissionPlacementConfig,
) -> Result<ServerAdmissionPolicyConfig> {
    let min = optional_admission_memory_share(
        values,
        NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV,
        "admission.grant_min",
        file.grant_min.as_ref(),
    )?;
    let want = optional_admission_memory_share(
        values,
        NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV,
        "admission.grant_want",
        file.grant_want.as_ref(),
    )?;
    let grant_min_field = min
        .as_ref()
        .map(|value| value.field)
        .or_else(|| workload_profile.value.default_grant_min().map(|_| workload_profile.field));
    let grant_want_field = want.as_ref().map(|value| value.field);
    let profile_default_min = workload_profile.value.default_grant_min();
    if let (Some(min), Some(want)) = (min.map(|value| value.value).or(profile_default_min), &want) {
        validate_comparable_memory_share_order(min, want.value, want.field)?;
    }
    let slot_guarantee = optional_native_string(
        values,
        NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV,
        "admission.slot_guarantee",
        file.slot_guarantee.as_deref(),
    );
    let slot_guarantee_field = slot_guarantee.as_ref().map(|raw| raw.field);
    let slot_guarantee = slot_guarantee
        .map(|raw| parse_slot_guarantee(raw.field, &raw.value))
        .transpose()?
        .unwrap_or(workload_profile.value.default_slot_guarantee());
    let wait_timeout = optional_duration_millis(
        values,
        NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV,
        "admission.wait_timeout_ms",
        file.wait_timeout_ms,
        DurationZeroPolicy::Reject,
    )?
    .unwrap_or_else(|| NexusAdmissionPolicy::default().wait_timeout());
    let mut policy = NexusAdmissionPolicy::default()
        .with_max_concurrent_queries(max_in_flight)
        .with_max_queries_per_device(placement.max_queries_per_device.map(|value| value.value))
        .with_placement_policy(placement.placement_policy)
        .with_slot_guarantee(slot_guarantee)
        .with_wait_timeout(wait_timeout);
    if let Some(min) = min {
        policy = policy.with_grant_min(min.value);
    } else if let Some(min) = profile_default_min {
        policy = policy.with_grant_min(min);
    }
    if let Some(want) = want {
        policy = policy.with_grant_want(want.value);
    }
    Ok(ServerAdmissionPolicyConfig { policy, grant_min_field, grant_want_field, slot_guarantee_field })
}

pub(super) fn server_placement_eligible_device_count(
    device_count: usize,
    placement_policy: QueryPlacementPolicy,
) -> usize {
    match placement_policy {
        QueryPlacementPolicy::LeastLoaded | QueryPlacementPolicy::BestFit => device_count.max(1),
        QueryPlacementPolicy::Pinned { .. } => 1,
        _ => device_count.max(1),
    }
}

pub(super) fn server_host_execution_qos_max_concurrent_statements(
    values: &mut EnvValues<'_>,
    file: &ServerHostExecutionQosFileConfig,
) -> Result<Option<NonZeroUsize>> {
    let selected = match values.optional_usize(NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS_ENV)? {
        Some(value) => NativeConfigValue::new(value, NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS_ENV),
        None => file
            .max_concurrent_statements
            .map(|value| NativeConfigValue::new(value, "host_execution_qos.max_concurrent_statements"))
            .unwrap_or(NativeConfigValue::new(
                DEFAULT_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS,
                NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS_ENV,
            )),
    };
    Ok(NonZeroUsize::new(selected.value))
}

pub(super) fn server_host_execution_qos_wait_timeout(
    values: &mut EnvValues<'_>,
    file: &ServerHostExecutionQosFileConfig,
) -> Result<Option<Duration>> {
    let selected = optional_duration_millis_value(
        values,
        NEXUS_SERVER_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS_ENV,
        "host_execution_qos.wait_timeout_ms",
        file.wait_timeout_ms,
    )?
    .unwrap_or(NativeConfigValue::new(
        DEFAULT_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS,
        NEXUS_SERVER_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS_ENV,
    ));
    duration_millis_from_value(selected, DurationZeroPolicy::Indefinite)
}

pub(super) fn optional_admission_memory_share(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<&ServerMemoryShareFileValue>,
) -> Result<Option<NativeConfigValue<MemoryShare>>> {
    optional_native_string(values, env_name, file_field, file_value.map(ServerMemoryShareFileValue::as_raw).as_deref())
        .map(|raw| parse_memory_share(raw.field, &raw.value).map(|share| NativeConfigValue::new(share, raw.field)))
        .transpose()
}

pub(super) fn validate_comparable_memory_share_order(
    min: MemoryShare,
    want: MemoryShare,
    want_field: &'static str,
) -> Result<()> {
    match (min, want) {
        (MemoryShare::Bytes(min), MemoryShare::Bytes(want)) if want < min => {
            Err(ServerError::invalid_config(want_field, "must be at least admission grant_min"))
        },
        (MemoryShare::Fraction(min), MemoryShare::Fraction(want)) if want < min => {
            Err(ServerError::invalid_config(want_field, "must be at least admission grant_min"))
        },
        _ => Ok(()),
    }
}

pub(super) fn validate_admission_grant_min_capacity(
    admission: &ServerAdmissionPolicyConfig,
    optimizer: &NexusNativeOptimizerConfig,
) -> Result<()> {
    let MemoryControlPolicy::Bounded { device_budget_bytes: device_capacity_bytes, .. } =
        optimizer.native_execution_config().memory_control_policy()
    else {
        return Ok(());
    };
    let MemoryShare::Bytes(grant_min_bytes) = admission.policy.grant().min else {
        return Ok(());
    };
    if grant_min_bytes <= device_capacity_bytes {
        return Ok(());
    }
    let field = admission.grant_min_field.unwrap_or(NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV);
    Err(ServerError::invalid_config(field, "minimum admission grant exceeds per-device capacity")
        .with_fact("reason", "admission_grant_min_above_device_capacity")
        .with_fact("grant_min_bytes", grant_min_bytes)
        .with_fact("device_capacity_bytes", device_capacity_bytes))
}

pub(super) fn first_non_neutral_unbounded_admission_field(
    workload_profile: NativeConfigValue<ServerAdmissionWorkloadProfile>,
    admission: &ServerAdmissionPolicyConfig,
    placement: ServerAdmissionPlacementConfig,
) -> Option<&'static str> {
    if workload_profile.value != ServerAdmissionWorkloadProfile::Balanced {
        return Some(workload_profile.field);
    }
    if let Some(field) = admission.grant_min_field {
        return Some(field);
    }
    if let Some(field) = admission.grant_want_field {
        return Some(field);
    }
    if let Some(field) = admission.slot_guarantee_field {
        return Some(field);
    }
    if placement.placement_policy != QueryPlacementPolicy::LeastLoaded {
        return placement.placement_policy_field;
    }
    if let Some(field) = placement.pinned_device_ordinal_field {
        return Some(field);
    }
    placement.max_queries_per_device.map(|value| value.field)
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
