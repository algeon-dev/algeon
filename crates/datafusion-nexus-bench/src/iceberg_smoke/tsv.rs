use super::*;

pub(crate) fn optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

pub(crate) fn parse_optional_bool(value: Option<String>, name: &'static str) -> SmokeResult<bool> {
    match value.map(|value| value.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("false" | "0" | "off" | "no") => Ok(false),
        Some("true" | "1" | "on" | "yes") => Ok(true),
        Some(other) => {
            Err(smoke_error(format!("unsupported {name}={other}; expected true/false, 1/0, on/off, or yes/no")))
        },
    }
}

pub(crate) fn parse_optional_timeout_secs(value: Option<String>, name: &'static str) -> SmokeResult<Option<u64>> {
    match value {
        Some(value) if !value.trim().is_empty() => {
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                smoke_error(format!("invalid {name}={}: expected unsigned integer seconds: {error}", value.trim()))
            })?;
            if parsed == 0 {
                return Err(smoke_error(format!("{name} must be greater than zero when set")));
            }
            Ok(Some(parsed))
        },
        _ => Ok(None),
    }
}

pub(crate) fn optional_env_u64_get(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
) -> SmokeResult<Option<u64>> {
    parse_optional_positive_u64(get(name), name)
}

pub(crate) fn optional_env_usize_get(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
) -> SmokeResult<Option<usize>> {
    match parse_optional_positive_u64(get(name), name)? {
        Some(value) => usize::try_from(value)
            .map(Some)
            .map_err(|_| smoke_error(format!("{name} is too large for this platform's usize"))),
        None => Ok(None),
    }
}

pub(crate) fn parse_optional_positive_u64(value: Option<String>, name: &'static str) -> SmokeResult<Option<u64>> {
    match value {
        Some(value) if !value.trim().is_empty() => {
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                smoke_error(format!("invalid {name}={}: expected unsigned integer: {error}", value.trim()))
            })?;
            if parsed == 0 {
                return Err(smoke_error(format!("{name} must be greater than zero when set")));
            }
            Ok(Some(parsed))
        },
        _ => Ok(None),
    }
}

pub(crate) fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

pub(crate) fn optional_u128(value: Option<u128>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

pub(crate) fn optional_duration_ms_text(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("{:.3}", value as f64 / 1_000_000.0))
}
pub(crate) fn sanitize_nested(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ").replace('|', "/")
}

pub(crate) fn list_or_none(values: &[String]) -> String {
    if values.is_empty() { "none".to_string() } else { values.join(",") }
}

pub(crate) fn empty_as_none(value: &str) -> String {
    if value.is_empty() { "none".to_string() } else { value.to_string() }
}

pub fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(crate) fn smoke_error(message: impl Into<String>) -> Box<dyn StdError + Send + Sync> {
    Box::new(SmokeError(message.into()))
}

#[derive(Debug)]
struct SmokeError(String);

impl fmt::Display for SmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl StdError for SmokeError {}

pub(crate) trait FallbackReasonExt {
    fn from_code(code: &str) -> Option<FallbackReason>;
}

impl FallbackReasonExt for FallbackReason {
    fn from_code(code: &str) -> Option<FallbackReason> {
        [
            FallbackReason::TableFormatUnsupported,
            FallbackReason::TableFormatPlanningFailed,
            FallbackReason::IcebergNonParquetDataFile,
            FallbackReason::IcebergDeleteFilesUnsupported,
            FallbackReason::IcebergFieldIdMappingUnsupported,
            FallbackReason::IcebergNameMappingUnsupported,
            FallbackReason::IcebergNestedProjectionUnsupported,
            FallbackReason::IcebergPartitionTransformUnsupported,
            FallbackReason::IcebergResidualPredicateUnsupported,
            FallbackReason::IcebergRowSelectionUnsupported,
            FallbackReason::IcebergObjectStoreReadUnsupported,
            FallbackReason::IcebergSnapshotUnsupported,
            FallbackReason::TableFormatGpuOnlyLeafNotRewritten,
        ]
        .into_iter()
        .find(|reason| reason.code() == code)
    }
}
