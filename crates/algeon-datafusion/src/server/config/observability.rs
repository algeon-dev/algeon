use std::time::Duration;

use query_runtime::observability::RuntimeObservationConfig;

use super::{ConfigSources, Result, ServerError, checked_config_duration, file::ServerObservabilityFileConfig};
use crate::server::coordinator::logical_capture_reservation_bytes;

const DEFAULT_GLOBAL_SEALED_MAX_COUNT: usize = 1_024;
const DEFAULT_GLOBAL_SEALED_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const DEFAULT_SEALED_TTL_SECS: u64 = 60 * 60;
const DEFAULT_SHUTDOWN_GRACE_SECS: u64 = 30;

/// Validated server policy for native runtime artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerObservabilityConfig {
    /// Maximum runtime observation records retained for one query.
    pub per_query_max_records: u64,
    /// Maximum runtime observation payload bytes retained for one query.
    pub per_query_max_bytes: u64,
    /// Logical capture bytes all live queries may reserve together. `None`
    /// covers every attempt the admission bounds can hold at once plus 1 GiB
    /// for queries outside admission (planning, finalizing, or never
    /// admitted). A query the budget cannot hold runs without capture, and
    /// its stats report `active_capture_budget`.
    pub active_logical_capture_max_bytes: Option<u64>,
    /// Maximum number of completed query artifacts retained across the server.
    pub global_sealed_max_count: usize,
    /// Maximum logical bytes retained by completed query artifacts across the server.
    pub global_sealed_max_bytes: u64,
    /// Maximum retention duration for a sealed query artifact.
    pub sealed_ttl: Duration,
    /// Time allowed for runtime-artifact finalization during server shutdown.
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

    /// Logical capture bytes of `admission_live_attempts` live queries, the
    /// attempts admission can hold at once, saturating at `u64::MAX`.
    pub(crate) fn admission_logical_capture_bytes(&self, admission_live_attempts: u64) -> u64 {
        logical_capture_reservation_bytes(
            self.runtime_observation_config().limits().max_retained_bytes,
            self.per_query_max_bytes,
            self.per_query_artifact_max_bytes(),
        )
        .map_or(u64::MAX, |bytes| bytes.saturating_mul(admission_live_attempts))
    }

    /// Logical capture budget the query coordinator enforces: the configured
    /// value, or, when unset, the capture of every attempt admission can hold
    /// at once plus 1 GiB for queries planning or finalizing beside them.
    pub(crate) fn active_logical_capture_budget_bytes(&self, admission_live_attempts: u64) -> Result<u64> {
        match self.active_logical_capture_max_bytes {
            Some(configured_bytes) => Ok(configured_bytes),
            None => self
                .admission_logical_capture_bytes(admission_live_attempts)
                .checked_add(Self::DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES)
                .ok_or_else(|| {
                    ServerError::invalid_config(
                        "observability.per_query_max_bytes",
                        "overflows the logical capture budget of the admission bounds",
                    )
                    .with_fact("admission_live_attempts", admission_live_attempts)
                }),
        }
    }
}

pub(super) fn server_observability_config(
    file: &ServerObservabilityFileConfig,
    sources: &ConfigSources,
) -> Result<ServerObservabilityConfig> {
    let defaults = ServerObservabilityConfig::default();
    let positive = |path: &'static str, value: Option<u64>, default: u64| -> Result<u64> {
        match value.unwrap_or(default) {
            0 => Err(sources.invalid(path, "must be greater than zero")),
            value => Ok(value),
        }
    };
    let active_logical_capture_max_bytes = match file.active_logical_capture_max_bytes {
        Some(0) => {
            return Err(sources.invalid("observability.active_logical_capture_max_bytes", "must be greater than zero"));
        },
        value => value,
    };
    let global_sealed_max_count = match file.global_sealed_max_count.unwrap_or(defaults.global_sealed_max_count) {
        0 => return Err(sources.invalid("observability.global_sealed_max_count", "must be greater than zero")),
        value => value,
    };
    let sealed_ttl_secs =
        positive("observability.sealed_ttl_secs", file.sealed_ttl_secs, defaults.sealed_ttl.as_secs())?;
    let shutdown_grace_secs =
        positive("observability.shutdown_grace_secs", file.shutdown_grace_secs, defaults.shutdown_grace.as_secs())?;
    Ok(ServerObservabilityConfig {
        per_query_max_records: positive(
            "observability.per_query_max_records",
            file.per_query_max_records,
            defaults.per_query_max_records,
        )?,
        per_query_max_bytes: positive(
            "observability.per_query_max_bytes",
            file.per_query_max_bytes,
            defaults.per_query_max_bytes,
        )?,
        active_logical_capture_max_bytes,
        global_sealed_max_count,
        global_sealed_max_bytes: positive(
            "observability.global_sealed_max_bytes",
            file.global_sealed_max_bytes,
            defaults.global_sealed_max_bytes,
        )?,
        sealed_ttl: checked_config_duration(
            sources,
            "observability.sealed_ttl_secs",
            Duration::from_secs(sealed_ttl_secs),
        )?,
        shutdown_grace: checked_config_duration(
            sources,
            "observability.shutdown_grace_secs",
            Duration::from_secs(shutdown_grace_secs),
        )?,
    })
}
