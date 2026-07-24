use super::*;

/// Socket address the Flight SQL server listens on. Defaults to `0.0.0.0:50051`.
pub const NEXUS_SERVER_BIND_ENV: &str = "NEXUS_SERVER_BIND";
/// Tracing log filter for the server process. Defaults to `info`.
pub const NEXUS_SERVER_LOG_ENV: &str = "NEXUS_SERVER_LOG";
/// Seconds an unconsumed Flight SQL statement ticket may wait for `DoGet`
/// before it expires. Defaults to `300`.
pub const NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV: &str = "NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS";
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
