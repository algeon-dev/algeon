use std::time::Duration;

use query_engine::observability::RuntimeObservationConfig;

use super::{EnvValues, Result, ServerError, checked_config_duration, file::ServerObservabilityFileConfig};
use crate::server::coordinator::logical_capture_reservation_bytes;

pub const ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV: &str =
    "ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS";
pub const ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES";
pub const ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV: &str =
    "ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES";
pub const ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV: &str =
    "ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT";
pub const ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV: &str =
    "ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES";
pub const ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS";
pub const ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS";

const DEFAULT_GLOBAL_SEALED_MAX_COUNT: usize = 1_024;
const DEFAULT_GLOBAL_SEALED_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const DEFAULT_SEALED_TTL_SECS: u64 = 60 * 60;
const DEFAULT_SHUTDOWN_GRACE_SECS: u64 = 30;

/// Validated server policy for native runtime artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerObservabilityConfig {
    pub per_query_max_records: u64,
    pub per_query_max_bytes: u64,
    /// Logical capture bytes all live queries may reserve together. `None`
    /// covers every attempt the admission bounds can hold at once plus 1 GiB
    /// for queries outside admission (planning, finalizing, or never
    /// admitted); an explicit budget below the admission requirement is
    /// rejected.
    pub active_logical_capture_max_bytes: Option<u64>,
    pub global_sealed_max_count: usize,
    pub global_sealed_max_bytes: u64,
    pub sealed_ttl: Duration,
    pub shutdown_grace: Duration,
}

impl Default for ServerObservabilityConfig {
    fn default() -> Self {
        Self {
            per_query_max_records: RuntimeObservationConfig::NATIVE_EXECUTION_MAX_RECORDS,
            per_query_max_bytes: RuntimeObservationConfig::NATIVE_EXECUTION_MAX_BYTES,
            active_logical_capture_max_bytes: None,
            global_sealed_max_count: DEFAULT_GLOBAL_SEALED_MAX_COUNT,
            global_sealed_max_bytes: DEFAULT_GLOBAL_SEALED_MAX_BYTES,
            sealed_ttl: Duration::from_secs(DEFAULT_SEALED_TTL_SECS),
            shutdown_grace: Duration::from_secs(DEFAULT_SHUTDOWN_GRACE_SECS),
        }
    }
}

impl ServerObservabilityConfig {
    /// Logical capture budget for queries outside admission when
    /// `active_logical_capture_max_bytes` is unset: a query keeps its
    /// reservation while it plans and finalizes, around its admission slot.
    pub(crate) const DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES: u64 = 1024 * 1024 * 1024;

    pub(crate) const fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        RuntimeObservationConfig::bounded(self.per_query_max_records, self.per_query_max_bytes)
    }

    pub(crate) fn per_query_artifact_max_bytes(&self) -> u64 {
        self.per_query_max_bytes.saturating_mul(2).min(self.global_sealed_max_bytes)
    }

    /// Logical capture budget the query coordinator enforces when admission
    /// can hold `admission_live_attempts` attempts at once: enough to register
    /// all of them beside queries outside admission, so queries reach the
    /// admission queue bound before the coordinator refuses them as overloaded.
    pub(crate) fn active_logical_capture_budget_bytes(&self, admission_live_attempts: u64) -> Result<u64> {
        let overflow = || {
            ServerError::invalid_config(
                ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV,
                "overflows the logical capture budget of the admission bounds",
            )
            .with_fact("admission_live_attempts", admission_live_attempts)
        };
        let required_bytes = logical_capture_reservation_bytes(
            self.runtime_observation_config().limits().max_retained_bytes,
            self.per_query_max_bytes,
            self.per_query_artifact_max_bytes(),
        )
        .and_then(|bytes| bytes.checked_mul(admission_live_attempts))
        .ok_or_else(overflow)?;
        match self.active_logical_capture_max_bytes {
            None => required_bytes.checked_add(Self::DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES).ok_or_else(overflow),
            Some(configured_bytes) if configured_bytes >= required_bytes => Ok(configured_bytes),
            Some(configured_bytes) => Err(ServerError::invalid_config(
                ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
                "must cover the logical capture of every attempt admission can hold at once",
            )
            .with_fact("reason", "below_admission_live_attempts")
            .with_fact("configured_bytes", configured_bytes)
            .with_fact("required_bytes", required_bytes)
            .with_fact("admission_live_attempts", admission_live_attempts)),
        }
    }
}

pub(super) fn server_observability_config(
    values: &mut EnvValues<'_>,
    overlay: &ServerObservabilityFileConfig,
) -> Result<ServerObservabilityConfig> {
    let defaults = ServerObservabilityConfig::default();
    let per_query_max_records = select_u64(
        values,
        ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV,
        overlay.per_query_max_records,
        defaults.per_query_max_records,
    )?;
    let per_query_max_bytes = select_u64(
        values,
        ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV,
        overlay.per_query_max_bytes,
        defaults.per_query_max_bytes,
    )?;
    let active_logical_capture_max_bytes = optional_positive_u64(
        values,
        ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
        overlay.active_logical_capture_max_bytes,
    )?;
    let global_sealed_max_count = select_usize(
        values,
        ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV,
        overlay.global_sealed_max_count,
        defaults.global_sealed_max_count,
    )?;
    let global_sealed_max_bytes = select_u64(
        values,
        ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV,
        overlay.global_sealed_max_bytes,
        defaults.global_sealed_max_bytes,
    )?;
    let sealed_ttl_secs = select_u64(
        values,
        ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV,
        overlay.sealed_ttl_secs,
        defaults.sealed_ttl.as_secs(),
    )?;
    let shutdown_grace_secs = select_u64(
        values,
        ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
        overlay.shutdown_grace_secs,
        defaults.shutdown_grace.as_secs(),
    )?;

    Ok(ServerObservabilityConfig {
        per_query_max_records,
        per_query_max_bytes,
        active_logical_capture_max_bytes,
        global_sealed_max_count,
        global_sealed_max_bytes,
        sealed_ttl: checked_config_duration(ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, sealed_ttl_secs)?,
        shutdown_grace: checked_config_duration(
            ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
            shutdown_grace_secs,
        )?,
    })
}

fn select_u64(values: &mut EnvValues<'_>, field: &'static str, file_value: Option<u64>, default: u64) -> Result<u64> {
    Ok(optional_positive_u64(values, field, file_value)?.unwrap_or(default))
}

fn optional_positive_u64(
    values: &mut EnvValues<'_>,
    field: &'static str,
    file_value: Option<u64>,
) -> Result<Option<u64>> {
    let value = values.optional_u64(field)?.or(file_value);
    if value == Some(0) {
        return Err(ServerError::invalid_config(field, "must be greater than zero"));
    }
    Ok(value)
}

fn select_usize(
    values: &mut EnvValues<'_>,
    field: &'static str,
    file_value: Option<usize>,
    default: usize,
) -> Result<usize> {
    let value = values.optional_usize(field)?.or(file_value).unwrap_or(default);
    if value == 0 {
        return Err(ServerError::invalid_config(field, "must be greater than zero"));
    }
    Ok(value)
}
