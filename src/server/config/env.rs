use super::*;

/// Socket address the Flight SQL server listens on. Defaults to `0.0.0.0:50051`.
pub const NEXUS_SERVER_BIND_ENV: &str = "NEXUS_SERVER_BIND";
/// Tracing log filter for the server process. Defaults to `info`.
pub const NEXUS_SERVER_LOG_ENV: &str = "NEXUS_SERVER_LOG";
/// Seconds an unconsumed Flight SQL statement ticket may wait for `DoGet`
/// before it expires. Defaults to `300`.
pub const NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV: &str = "NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS";
/// DataFusion physical-plan profile for Flight SQL sessions. Use `native` for
/// GPU-native plan shapes or `cpu_fallback` to keep DataFusion's CPU-oriented
/// execution settings; the default is `native` when Iceberg is configured and
/// `cpu_fallback` otherwise.
pub const NEXUS_SERVER_SESSION_PROFILE_ENV: &str = "NEXUS_SERVER_SESSION_PROFILE";
/// Path to a TOML server config file. The file supplies durable server config
/// such as `[auth]` and `[native]`; scalar env vars remain bootstrap overrides.
pub const NEXUS_SERVER_CONFIG_FILE_ENV: &str = "NEXUS_SERVER_CONFIG_FILE";
pub(super) const DEFAULT_BIND_ADDR: &str = "0.0.0.0:50051";
pub(super) const DEFAULT_LOG_FILTER: &str = "info";
pub(super) const DEFAULT_STATEMENT_TICKET_TTL_SECS: u64 = 300;

#[derive(Debug, Clone, Copy)]
pub(super) struct NativeConfigValue<T> {
    pub(super) value: T,
    pub(super) field: &'static str,
}

impl<T> NativeConfigValue<T> {
    pub(super) const fn new(value: T, field: &'static str) -> Self {
        Self { value, field }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DurationZeroPolicy {
    Reject,
    Indefinite,
}
/// DataFusion physical-plan profile applied to the Flight SQL session.
///
/// Decisions tracked here only affect DataFusion's logical/physical plan shape
/// for queries served through Flight SQL. They are orthogonal to the native
/// nexus-query-engine memory budget, which is controlled by the server native
/// config (`[native]` TOML or `NEXUS_SERVER_NATIVE_*` environment variables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerSessionProfile {
    /// Native-friendly profile that matches the workload harness defaults:
    /// `target_partitions=1`, repartitioning disabled for file scans /
    /// aggregations / joins / sorts, and all dynamic-filter-pushdown variants
    /// disabled. Required for GPU-only table-format leaves.
    Native,
    /// Preserve DataFusion's CPU-style profile derived from
    /// `DATAFUSION_EXECUTION_*` environment variables (single underscores,
    /// matching DataFusion's `SessionConfig::from_env()` mapping).
    ///
    /// For Iceberg, this requires the `iceberg-cpu` feature; otherwise the
    /// catalog still exposes GPU-only leaves and planning can fail. Non-Iceberg
    /// CPU sessions do not need that feature.
    CpuFallback,
}

impl ServerSessionProfile {
    pub fn parse(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "native" => Ok(Self::Native),
            "cpu_fallback" | "cpu-fallback" => Ok(Self::CpuFallback),
            other => Err(format!("must be `native` or `cpu_fallback`, got `{other}`")),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::CpuFallback => "cpu_fallback",
        }
    }
}

impl std::fmt::Display for ServerSessionProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
pub(super) struct EnvValues<'a> {
    get: &'a mut dyn FnMut(&str) -> Option<String>,
    cache: BTreeMap<&'static str, Option<String>>,
}

impl<'a> EnvValues<'a> {
    pub(super) fn new(get: &'a mut dyn FnMut(&str) -> Option<String>) -> Self {
        Self { get, cache: BTreeMap::new() }
    }

    pub(super) fn any_present<const N: usize>(&mut self, names: [&'static str; N]) -> bool {
        names.into_iter().any(|name| self.string(name).is_some())
    }

    #[cfg(feature = "iceberg")]
    pub(super) fn required_string(&mut self, name: &'static str) -> Result<String> {
        self.string(name).ok_or_else(|| ServerError::invalid_config(name, "is required"))
    }

    pub(super) fn string(&mut self, name: &'static str) -> Option<String> {
        if let Some(cached) = self.cache.get(name) {
            return cached.clone();
        }
        let value = (self.get)(name).and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        self.cache.insert(name, value.clone());
        value
    }

    #[cfg(feature = "iceberg")]
    pub(super) fn string_uncached(&mut self, name: &str) -> Option<String> {
        (self.get)(name).and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
        })
    }

    pub(super) fn socket_addr(&mut self, name: &'static str, default: &str) -> Result<SocketAddr> {
        let raw = self.string(name).unwrap_or_else(|| default.to_string());
        raw.parse().map_err(|error| ServerError::invalid_config_with_source(name, "must be a socket address", error))
    }

    pub(super) fn positive_u64(&mut self, name: &'static str, default: u64) -> Result<u64> {
        let value = self.optional_u64(name)?.unwrap_or(default);
        if value == 0 {
            return Err(ServerError::invalid_config(name, "must be greater than zero"));
        }
        Ok(value)
    }

    pub(super) fn optional_usize(&mut self, name: &'static str) -> Result<Option<usize>> {
        self.string(name)
            .map(|raw| {
                raw.parse::<usize>()
                    .map_err(|error| ServerError::invalid_config_with_source(name, "must be an integer", error))
            })
            .transpose()
    }

    pub(super) fn optional_u64(&mut self, name: &'static str) -> Result<Option<u64>> {
        self.string(name)
            .map(|raw| {
                raw.parse::<u64>()
                    .map_err(|error| ServerError::invalid_config_with_source(name, "must be an integer", error))
            })
            .transpose()
    }

    pub(super) fn optional_f64(&mut self, name: &'static str) -> Result<Option<f64>> {
        self.string(name)
            .map(|raw| {
                let value = raw.parse::<f64>().map_err(|error| {
                    ServerError::invalid_config_with_source(name, "must be a decimal number", error)
                })?;
                if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
                    return Err(ServerError::invalid_config(name, "must be greater than 0 and at most 1"));
                }
                Ok(value)
            })
            .transpose()
    }

    pub(super) fn optional_session_profile(&mut self, name: &'static str) -> Result<Option<ServerSessionProfile>> {
        self.string(name)
            .map(|raw| ServerSessionProfile::parse(&raw).map_err(|reason| ServerError::invalid_config(name, reason)))
            .transpose()
    }

    pub(super) fn optional_bool(&mut self, name: &'static str) -> Result<Option<bool>> {
        self.string(name)
            .map(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                other => {
                    Err(ServerError::invalid_config(name, format!("must be a boolean (`true`/`false`), got `{other}`")))
                },
            })
            .transpose()
    }
}
