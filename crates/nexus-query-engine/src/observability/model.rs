//! Typed runtime observation identities and records.

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{
    Deserialize, Serialize,
    de::{self, SeqAccess, Visitor},
};

use crate::metrics::QueryId;

/// JSONL schema emitted by sealed runtime observation snapshots.
pub const RUNTIME_OBSERVATION_JSONL_SCHEMA: &str = "nexus_query_engine.runtime_observation.v2";
/// Schema label for the fixed execution-attempt report envelope.
pub const QUERY_ATTEMPT_REPORT_SCHEMA_V1: &str = "nexus_query_engine.query_attempt_report.v1";
/// Schema label for the bounded public projection of a sealed attempt report.
///
/// The projection deliberately omits correlations, diagnostics, allocator
/// events, and other payloads that are not suitable for fixed-cardinality
/// consumers such as DataFusion metrics and retained Flight control metadata.
pub const SEALED_ATTEMPT_PROJECTION_SCHEMA_V1: &str = "nexus_query_engine.sealed_attempt_projection.v1";
/// Maximum UTF-8 byte length for caller-supplied attempt query correlation.
pub const MAX_ATTEMPT_QUERY_ID_BYTES: usize = 256;
/// Maximum UTF-8 byte length for caller-supplied opaque trace context.
pub const MAX_ATTEMPT_TRACE_CONTEXT_BYTES: usize = 1024;

static NEXT_ATTEMPT_ID: AtomicU64 = AtomicU64::new(1);

fn attempt_id_seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        // TODO: better implementation
        nanos ^ ((std::process::id() as u64) << 32) ^ 0x9e37_79b9_7f4a_7c15
    })
}

/// Opaque engine-minted identity for one physical execution attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttemptId([u8; 16]);

impl AttemptId {
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Mints a fresh process-unique attempt identity.
    ///
    /// Callers never supply an [`AttemptId`] when beginning an attempt; the
    /// backend mints one per physical execution invocation.
    pub fn mint() -> Self {
        let counter = NEXT_ATTEMPT_ID.fetch_add(1, Ordering::Relaxed).to_be_bytes();
        let seed = attempt_id_seed().to_be_bytes();
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&counter);
        bytes[8..].copy_from_slice(&seed);
        Self(bytes)
    }
}

impl fmt::Display for AttemptId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "attempt-")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for AttemptId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AttemptId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let value = String::deserialize(deserializer)?;
        let hex = value.strip_prefix("attempt-").ok_or_else(|| serde::de::Error::custom("attempt id prefix"))?;
        if hex.len() != 32 {
            return Err(serde::de::Error::custom("attempt id length"));
        }
        let mut bytes = [0; 16];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).map_err(serde::de::Error::custom)?;
        }
        Ok(Self(bytes))
    }
}

macro_rules! attempt_correlation {
    ($name:ident, $limit:ident, $field:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);
        impl $name {
            pub fn new(value: impl Into<String>) -> crate::error::Result<Self> {
                let value = value.into();
                if value.trim().is_empty() || value.len() > $limit {
                    return Err(crate::error::QueryError::new(
                        crate::error::QueryErrorCode::InvalidAttemptCorrelation,
                        "nexus_query_engine.attempt.correlation",
                        "attempt correlation must be non-blank and within its byte limit",
                    )
                    .with_fact("field", $field)
                    .with_fact("max_bytes", $limit)
                    .with_fact("reason", if value.trim().is_empty() { "blank" } else { "too_long" }));
                }
                Ok(Self(value))
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where D: serde::Deserializer<'de> {
                // Validation is part of the value contract on every
                // construction path, including report decoding.
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}
attempt_correlation!(AttemptQueryId, MAX_ATTEMPT_QUERY_ID_BYTES, "query_id");
attempt_correlation!(AttemptTraceContext, MAX_ATTEMPT_TRACE_CONTEXT_BYTES, "trace_context");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    Unplanned,
    CpuFallback,
    PartialNative,
    Native,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptCleanupDisposition {
    NotRequired,
    Completed,
    CompletedWithFailures,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptCapacityDisposition {
    NotAdmitted,
    Released,
    RetainedQuarantine,
}

/// The device-health result decided by an admitted attempt's terminal release
/// commit.
///
/// This is deliberately distinct from [`AttemptCapacityDisposition`]: both a
/// device-local fault and an unknown-isolation backend fault retain capacity,
/// but they have materially different health consequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptHealthTransition {
    HealthyReuse,
    DeviceQuarantined,
    BackendUnhealthy,
}

/// Maximum retained UTF-8 bytes in one stable failure-text projection.
///
/// The source error remains untouched on its returned path. This is only the
/// bounded, redacted summary that may enter an immutable attempt report.
pub const MAX_ATTEMPT_FAILURE_TEXT_BYTES: usize = 256;
const MAX_ATTEMPT_FAILURE_TEXT_INSPECTION_BYTES: usize = 1024;
const MAX_ATTEMPT_FAILURE_FACTS: usize = 4;

/// Owning layer that classified a stable attempt failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureOrigin {
    Native,
    Datafusion,
    AttemptControl,
}

/// Closed error-code projection suitable for an attempt report.
///
/// This deliberately has no string-carrying catch-all variant. New native
/// error codes must extend this closed report contract explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureCode {
    Cancelled,
    UnsupportedPlanShape,
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    InvalidPlan,
    InvalidAttemptCorrelation,
    GraphInputInvalid,
    CudfInvalidInput,
    CudfUnsupported,
    CudfResourceExhausted,
    SizeOverflow,
    HostMaterializationLimitExceeded,
    CudfTemporaryFailure,
    CudfPersistentFailure,
    CudfDependencyFailure,
    SourceAccessDenied,
    SourceUnauthenticated,
    SourceObjectNotFound,
    SourceTemporaryFailure,
    CudfInvariantViolation,
    MemoryReservationFailed,
    ServiceOverloaded,
    Internal,
    BackendClosing,
    DatafusionError,
    Dropped,
    Panicked,
    StreamSynchronizationFailed,
    CleanupCallbackAbandoned,
}

impl AttemptFailureCode {
    pub const fn from_query_error_code(code: crate::error::QueryErrorCode) -> Self {
        match code {
            crate::error::QueryErrorCode::Cancelled => Self::Cancelled,
            crate::error::QueryErrorCode::UnsupportedPlanShape => Self::UnsupportedPlanShape,
            crate::error::QueryErrorCode::UnsupportedOperator => Self::UnsupportedOperator,
            crate::error::QueryErrorCode::UnsupportedExpression => Self::UnsupportedExpression,
            crate::error::QueryErrorCode::UnsupportedType => Self::UnsupportedType,
            crate::error::QueryErrorCode::InvalidPlan => Self::InvalidPlan,
            crate::error::QueryErrorCode::InvalidAttemptCorrelation => Self::InvalidAttemptCorrelation,
            crate::error::QueryErrorCode::GraphInputInvalid => Self::GraphInputInvalid,
            crate::error::QueryErrorCode::CudfInvalidInput => Self::CudfInvalidInput,
            crate::error::QueryErrorCode::CudfUnsupported => Self::CudfUnsupported,
            crate::error::QueryErrorCode::CudfResourceExhausted => Self::CudfResourceExhausted,
            crate::error::QueryErrorCode::SizeOverflow => Self::SizeOverflow,
            crate::error::QueryErrorCode::HostMaterializationLimitExceeded => Self::HostMaterializationLimitExceeded,
            crate::error::QueryErrorCode::CudfTemporaryFailure => Self::CudfTemporaryFailure,
            crate::error::QueryErrorCode::CudfPersistentFailure => Self::CudfPersistentFailure,
            crate::error::QueryErrorCode::CudfDependencyFailure => Self::CudfDependencyFailure,
            crate::error::QueryErrorCode::SourceAccessDenied => Self::SourceAccessDenied,
            crate::error::QueryErrorCode::SourceUnauthenticated => Self::SourceUnauthenticated,
            crate::error::QueryErrorCode::SourceObjectNotFound => Self::SourceObjectNotFound,
            crate::error::QueryErrorCode::SourceTemporaryFailure => Self::SourceTemporaryFailure,
            crate::error::QueryErrorCode::CudfInvariantViolation => Self::CudfInvariantViolation,
            crate::error::QueryErrorCode::MemoryReservationFailed => Self::MemoryReservationFailed,
            crate::error::QueryErrorCode::ServiceOverloaded => Self::ServiceOverloaded,
            crate::error::QueryErrorCode::Internal => Self::Internal,
        }
    }

    /// Converts a legacy stable label at an adapter boundary without retaining
    /// it as an open report value. Unknown labels fail closed to `internal`.
    pub fn from_label(label: &str) -> Self {
        match label {
            "cancelled" => Self::Cancelled,
            "unsupported_plan_shape" => Self::UnsupportedPlanShape,
            "unsupported_operator" => Self::UnsupportedOperator,
            "unsupported_expression" => Self::UnsupportedExpression,
            "unsupported_type" => Self::UnsupportedType,
            "invalid_plan" => Self::InvalidPlan,
            "invalid_attempt_correlation" => Self::InvalidAttemptCorrelation,
            "graph_input_invalid" => Self::GraphInputInvalid,
            "cudf_invalid_input" => Self::CudfInvalidInput,
            "cudf_unsupported" => Self::CudfUnsupported,
            "cudf_resource_exhausted" => Self::CudfResourceExhausted,
            "size_overflow" => Self::SizeOverflow,
            "host_materialization_limit_exceeded" => Self::HostMaterializationLimitExceeded,
            "cudf_temporary_failure" => Self::CudfTemporaryFailure,
            "cudf_persistent_failure" => Self::CudfPersistentFailure,
            "cudf_dependency_failure" => Self::CudfDependencyFailure,
            "source_access_denied" => Self::SourceAccessDenied,
            "source_unauthenticated" => Self::SourceUnauthenticated,
            "source_object_not_found" => Self::SourceObjectNotFound,
            "source_temporary_failure" => Self::SourceTemporaryFailure,
            "cudf_invariant_violation" => Self::CudfInvariantViolation,
            "memory_reservation_failed" => Self::MemoryReservationFailed,
            "service_overloaded" => Self::ServiceOverloaded,
            "backend_closing" => Self::BackendClosing,
            "datafusion_error" => Self::DatafusionError,
            "dropped" => Self::Dropped,
            "panicked" => Self::Panicked,
            "stream_synchronization_failed" => Self::StreamSynchronizationFailed,
            "cleanup_callback_abandoned" => Self::CleanupCallbackAbandoned,
            _ => Self::Internal,
        }
    }
}

/// Closed stability/recovery status for a report-visible failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureStatus {
    NotApplicable,
    Permanent,
    Temporary,
    Persistent,
}

impl AttemptFailureStatus {
    pub const fn from_query_error_status(status: crate::error::QueryErrorStatus) -> Self {
        match status {
            crate::error::QueryErrorStatus::Permanent => Self::Permanent,
            crate::error::QueryErrorStatus::Temporary => Self::Temporary,
            crate::error::QueryErrorStatus::Persistent => Self::Persistent,
        }
    }

    pub fn from_label(label: &str) -> Self {
        match label {
            "temporary" => Self::Temporary,
            "persistent" => Self::Persistent,
            "not_applicable" => Self::NotApplicable,
            _ => Self::Permanent,
        }
    }
}

/// Coarse owning operation for a stable failure projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureOperation {
    Planning,
    Admission,
    Execution,
    Cleanup,
    Datafusion,
    AttemptControl,
}

impl AttemptFailureOperation {
    fn from_native_operation(operation: &str) -> Self {
        if operation.contains("query_service") || operation.contains("admission") {
            Self::Admission
        } else if operation.contains("cleanup") {
            Self::Cleanup
        } else if operation.contains("plan") || operation.contains("validate") {
            Self::Planning
        } else {
            Self::Execution
        }
    }
}

/// Stable allowlisted fact keys for bounded failure summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureFactKey {
    Reason,
}

/// Stable allowlisted fact values for bounded failure summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureFactValue {
    BackendClosing,
    DeadlineExpired,
    CallerCancelled,
    AdmissionQueueFull,
    QueueWaitElapsed,
    MemoryGrantWaitTimeout,
    ProtectedWaiterNoRunnableDevice,
    RequiredFloorExceedsDeviceCapacity,
    BackendUnhealthy,
}

/// One closed, bounded fact retained beside an attempt failure summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptFailureFact {
    key: AttemptFailureFactKey,
    value: AttemptFailureFactValue,
}

impl AttemptFailureFact {
    pub const fn key(&self) -> AttemptFailureFactKey {
        self.key
    }

    pub const fn value(&self) -> AttemptFailureFactValue {
        self.value
    }
}

/// Bounded, UTF-8-safe and redacted diagnostic text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttemptFailureText {
    value: String,
    truncated: bool,
    redacted: bool,
    invalid_utf8: bool,
}

#[derive(Deserialize)]
struct AttemptFailureTextWire {
    #[serde(deserialize_with = "deserialize_bounded_attempt_failure_text_value")]
    value: String,
    truncated: bool,
    redacted: bool,
    invalid_utf8: bool,
}

fn deserialize_bounded_attempt_failure_text_value<'de, D>(deserializer: D) -> Result<String, D::Error>
where D: serde::Deserializer<'de> {
    struct BoundedTextVisitor;

    impl<'de> Visitor<'de> for BoundedTextVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a UTF-8 string no longer than {MAX_ATTEMPT_FAILURE_TEXT_BYTES} bytes")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where E: de::Error {
            if value.len() > MAX_ATTEMPT_FAILURE_TEXT_BYTES {
                return Err(E::custom("attempt failure text exceeds its byte bound"));
            }
            Ok(value.to_owned())
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where E: de::Error {
            if value.len() > MAX_ATTEMPT_FAILURE_TEXT_BYTES {
                return Err(E::custom("attempt failure text exceeds its byte bound"));
            }
            Ok(value)
        }
    }

    deserializer.deserialize_string(BoundedTextVisitor)
}

impl<'de> Deserialize<'de> for AttemptFailureText {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let wire = AttemptFailureTextWire::deserialize(deserializer)?;
        if wire.value.len() > MAX_ATTEMPT_FAILURE_TEXT_BYTES {
            return Err(serde::de::Error::custom("attempt failure text exceeds its byte bound"));
        }
        if wire.redacted {
            if wire.value != "<redacted>" {
                return Err(serde::de::Error::custom("redacted attempt failure text must not retain source text"));
            }
        } else if attempt_failure_text_requires_redaction(&wire.value.to_ascii_lowercase()) {
            return Err(serde::de::Error::custom("attempt failure text requires redaction"));
        }
        Ok(Self {
            value: wire.value,
            truncated: wire.truncated,
            redacted: wire.redacted,
            invalid_utf8: wire.invalid_utf8,
        })
    }
}

impl AttemptFailureText {
    /// Retains only a deterministic safe projection of arbitrary error bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let inspected_len = bytes.len().min(MAX_ATTEMPT_FAILURE_TEXT_INSPECTION_BYTES);
        let inspected = &bytes[..inspected_len];
        let invalid_utf8 = std::str::from_utf8(inspected).is_err();
        let mut value = String::from_utf8_lossy(inspected).split_whitespace().collect::<Vec<_>>().join(" ");
        let mut redacted = false;
        let lowercase = value.to_ascii_lowercase();
        if attempt_failure_text_requires_redaction(&lowercase) {
            value = "<redacted>".to_string();
            redacted = true;
        }
        let (value, truncated_by_bound) = truncate_utf8(&value, MAX_ATTEMPT_FAILURE_TEXT_BYTES);
        Self { value, truncated: bytes.len() > inspected_len || truncated_by_bound, redacted, invalid_utf8 }
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    pub const fn redacted(&self) -> bool {
        self.redacted
    }

    pub const fn invalid_utf8(&self) -> bool {
        self.invalid_utf8
    }
}

fn attempt_failure_text_requires_redaction(lowercase: &str) -> bool {
    lowercase.contains("://")
        || lowercase.contains("password")
        || lowercase.contains("passwd")
        || lowercase.contains("credential")
        || lowercase.contains("authorization")
        || lowercase.contains("token")
        || lowercase.contains("secret")
        || lowercase.contains("api_key")
        || lowercase.contains("access_key")
        || lowercase.contains("bearer ")
        || looks_like_sql_statement(lowercase)
}

fn looks_like_sql_statement(value: &str) -> bool {
    let value = value.trim_start();
    ["select", "insert", "update", "delete", "merge", "with", "create", "alter", "drop"].into_iter().any(|keyword| {
        value.strip_prefix(keyword).is_some_and(|suffix| {
            suffix.is_empty()
                || suffix
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_whitespace() || matches!(character, '*' | '('))
        })
    })
}

fn truncate_utf8(value: &str, max_bytes: usize) -> (String, bool) {
    if value.len() <= max_bytes {
        return (value.to_string(), false);
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (value[..end].to_string(), true)
}

/// Bounded retained classification for a primary terminal failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttemptFailureSummary {
    origin: AttemptFailureOrigin,
    code: AttemptFailureCode,
    status: AttemptFailureStatus,
    operation: AttemptFailureOperation,
    text: Option<AttemptFailureText>,
    facts: Vec<AttemptFailureFact>,
}

#[derive(Deserialize)]
struct AttemptFailureSummaryWire {
    origin: AttemptFailureOrigin,
    code: AttemptFailureCode,
    status: AttemptFailureStatus,
    operation: AttemptFailureOperation,
    #[serde(default)]
    text: Option<AttemptFailureText>,
    #[serde(default, deserialize_with = "deserialize_bounded_attempt_failure_facts")]
    facts: Vec<AttemptFailureFact>,
}

fn deserialize_bounded_attempt_failure_facts<'de, D>(deserializer: D) -> Result<Vec<AttemptFailureFact>, D::Error>
where D: serde::Deserializer<'de> {
    struct BoundedFactsVisitor;

    impl<'de> Visitor<'de> for BoundedFactsVisitor {
        type Value = Vec<AttemptFailureFact>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "at most {MAX_ATTEMPT_FAILURE_FACTS} attempt failure facts")
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where A: SeqAccess<'de> {
            let mut facts = Vec::with_capacity(MAX_ATTEMPT_FAILURE_FACTS);
            while let Some(fact) = sequence.next_element()? {
                if facts.len() == MAX_ATTEMPT_FAILURE_FACTS {
                    return Err(de::Error::custom("attempt failure facts exceed their fixed bound"));
                }
                facts.push(fact);
            }
            Ok(facts)
        }
    }

    deserializer.deserialize_seq(BoundedFactsVisitor)
}

impl<'de> Deserialize<'de> for AttemptFailureSummary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let wire = AttemptFailureSummaryWire::deserialize(deserializer)?;
        Self::from_parts(wire.origin, wire.code, wire.status, wire.operation, wire.text, wire.facts)
            .map_err(serde::de::Error::custom)
    }
}

impl AttemptFailureSummary {
    fn from_parts(
        origin: AttemptFailureOrigin,
        code: AttemptFailureCode,
        status: AttemptFailureStatus,
        operation: AttemptFailureOperation,
        text: Option<AttemptFailureText>,
        facts: Vec<AttemptFailureFact>,
    ) -> Result<Self, &'static str> {
        if facts.len() > MAX_ATTEMPT_FAILURE_FACTS {
            return Err("attempt failure facts exceed their fixed bound");
        }
        if facts.windows(2).any(|pair| pair[0].key >= pair[1].key) {
            return Err("attempt failure facts must use unique deterministic key order");
        }
        Ok(Self { origin, code, status, operation, text, facts })
    }

    /// Captures a native error by reference without touching its source chain.
    ///
    /// The native message is deliberately not copied into the stable report:
    /// it can contain arbitrary application metadata even when it does not
    /// match a known URI, credential, or SQL pattern. The returned
    /// [`crate::error::QueryError`] keeps that complete source text on its
    /// ordinary error path.
    pub fn from_native_error(error: &crate::error::QueryError) -> Self {
        let facts = match error.fact_value("reason") {
            Some("backend_closing") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::BackendClosing,
            }],
            Some("deadline_expired") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::DeadlineExpired,
            }],
            Some("caller_cancelled") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::CallerCancelled,
            }],
            Some("admission_queue_full") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::AdmissionQueueFull,
            }],
            Some("queue_wait_elapsed") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::QueueWaitElapsed,
            }],
            Some("memory_grant_wait_timeout") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::MemoryGrantWaitTimeout,
            }],
            Some("protected_waiter_no_runnable_device") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::ProtectedWaiterNoRunnableDevice,
            }],
            Some("required_floor_exceeds_device_capacity") | Some("impossible_requirement") => {
                vec![AttemptFailureFact {
                    key: AttemptFailureFactKey::Reason,
                    value: AttemptFailureFactValue::RequiredFloorExceedsDeviceCapacity,
                }]
            },
            Some("backend_unhealthy") => vec![AttemptFailureFact {
                key: AttemptFailureFactKey::Reason,
                value: AttemptFailureFactValue::BackendUnhealthy,
            }],
            _ => Vec::new(),
        };
        Self {
            origin: AttemptFailureOrigin::Native,
            code: AttemptFailureCode::from_query_error_code(error.code()),
            status: AttemptFailureStatus::from_query_error_status(error.status()),
            operation: AttemptFailureOperation::from_native_operation(error.operation()),
            text: None,
            facts,
        }
    }

    /// Closed root-adapter fallback. It intentionally retains no DataFusion
    /// display text, source-chain text, SQL, URI, or credential material.
    pub const fn datafusion_fallback() -> Self {
        Self {
            origin: AttemptFailureOrigin::Datafusion,
            code: AttemptFailureCode::DatafusionError,
            status: AttemptFailureStatus::Permanent,
            operation: AttemptFailureOperation::Datafusion,
            text: None,
            facts: Vec::new(),
        }
    }

    pub const fn backend_closing() -> Self {
        Self {
            origin: AttemptFailureOrigin::AttemptControl,
            code: AttemptFailureCode::BackendClosing,
            status: AttemptFailureStatus::Permanent,
            operation: AttemptFailureOperation::Admission,
            text: None,
            facts: Vec::new(),
        }
    }

    const fn control(code: AttemptFailureCode) -> Self {
        Self {
            origin: AttemptFailureOrigin::AttemptControl,
            code,
            status: AttemptFailureStatus::NotApplicable,
            operation: AttemptFailureOperation::AttemptControl,
            text: None,
            facts: Vec::new(),
        }
    }

    pub const fn origin(&self) -> AttemptFailureOrigin {
        self.origin
    }

    pub const fn code(&self) -> AttemptFailureCode {
        self.code
    }

    pub const fn status(&self) -> AttemptFailureStatus {
        self.status
    }

    pub const fn operation(&self) -> AttemptFailureOperation {
        self.operation
    }

    pub fn text(&self) -> Option<&AttemptFailureText> {
        self.text.as_ref()
    }

    pub fn facts(&self) -> &[AttemptFailureFact] {
        &self.facts
    }
}

/// Closed lifecycle phase at which a secondary cleanup failure occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupFailurePhase {
    SchedulingCycle,
    RunnableTaskSelection,
    TaskPreparation,
    TaskExecution,
    TaskPanic,
    TaskCompletion,
    CompletedResourceRelease,
    FinalValidation,
    QueryServiceStreamSynchronization,
    QueryServiceCallbackAbandoned,
    QueryServiceCleanup,
}

/// First bounded cleanup failure retained by an attempt lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CleanupFailureSummary {
    phase: CleanupFailurePhase,
    failure: AttemptFailureSummary,
    /// `false` means another cleanup failure followed this first retained
    /// summary. The exact count remains in the enclosing attempt report.
    #[serde(default = "default_cleanup_failure_summary_complete")]
    complete: bool,
}

const fn default_cleanup_failure_summary_complete() -> bool {
    true
}

impl CleanupFailureSummary {
    pub fn from_native_error(phase: CleanupFailurePhase, error: &crate::error::QueryError) -> Self {
        Self { phase, failure: AttemptFailureSummary::from_native_error(error), complete: true }
    }

    pub const fn stream_synchronization_failed() -> Self {
        Self {
            phase: CleanupFailurePhase::QueryServiceStreamSynchronization,
            failure: AttemptFailureSummary::control(AttemptFailureCode::StreamSynchronizationFailed),
            complete: true,
        }
    }

    pub const fn callback_abandoned() -> Self {
        Self {
            phase: CleanupFailurePhase::QueryServiceCallbackAbandoned,
            failure: AttemptFailureSummary::control(AttemptFailureCode::CleanupCallbackAbandoned),
            complete: true,
        }
    }

    pub const fn phase(&self) -> CleanupFailurePhase {
        self.phase
    }

    pub const fn failure(&self) -> &AttemptFailureSummary {
        &self.failure
    }

    pub const fn complete(&self) -> bool {
        self.complete
    }

    pub(crate) fn mark_incomplete(&mut self) {
        self.complete = false;
    }
}

/// Exact final counters observed for one attempt allocation domain.
///
/// A report carries this only when its final sample completed successfully.
/// `None` in [`QueryAttemptReport::final_domain_usage`] means that no
/// trustworthy final sample is available; it never represents a synthetic
/// all-zero domain. This includes ledger-only attempts, failed final samples,
/// and unknown-isolation cleanup paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptFinalDomainUsage {
    pub domain_id: u64,
    pub active: bool,
    pub active_members: u64,
    pub ceiling_bytes: u64,
    pub effective_driver_ceiling_bytes: u64,
    pub driver_ceiling_verified: bool,
    pub allocation_limit_bytes: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub total_bytes: u64,
    pub current_aligned_bytes: u64,
    pub peak_aligned_bytes: u64,
    pub total_allocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub failed_allocation_count: u64,
    pub largest_failed_allocation_bytes: u64,
}

impl From<cudf::QueryMemoryDomainUsage> for AttemptFinalDomainUsage {
    fn from(usage: cudf::QueryMemoryDomainUsage) -> Self {
        Self {
            domain_id: usage.domain_id,
            active: usage.active,
            active_members: usage.active_members,
            ceiling_bytes: usage.ceiling_bytes,
            effective_driver_ceiling_bytes: usage.effective_driver_ceiling_bytes,
            driver_ceiling_verified: usage.driver_ceiling_verified,
            allocation_limit_bytes: usage.allocation_limit_bytes,
            current_bytes: usage.current_bytes,
            peak_bytes: usage.peak_bytes,
            total_bytes: usage.total_bytes,
            current_aligned_bytes: usage.current_aligned_bytes,
            peak_aligned_bytes: usage.peak_aligned_bytes,
            total_allocations: usage.total_allocations,
            current_allocations: usage.current_allocations,
            peak_allocations: usage.peak_allocations,
            failed_allocation_count: usage.failed_allocation_count,
            largest_failed_allocation_bytes: usage.largest_failed_allocation_bytes,
        }
    }
}

/// Isolation decision committed with one device-backed attempt's terminal
/// native-health observation.
///
/// `DeviceLocal` means the release path proved that the relevant cleanup and
/// health facts belong to the attempt's immutable device. `Unknown` means the
/// cleanup path could not establish that boundary, so the engine closes the
/// entire backend rather than guessing a device owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFaultIsolation {
    DeviceLocal,
    Unknown,
}

/// Cleanup state retained with terminal native-health evidence.
///
/// This is an engine-owned projection of the release worker's result. It does
/// not infer a CUDA status from a later runtime query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptTeardownHealth {
    Completed,
    StreamSynchronizationFailed,
    CallbackAbandoned,
}

/// Health state read from cuDF's process-lifetime device-health authority at
/// the terminal release commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDeviceHealthStatus {
    Healthy,
    Unhealthy,
}

/// Closed projection of the device-fault kind supplied by cuDF and cuGraph.
///
/// The source enum is non-exhaustive. A newly introduced sibling kind remains
/// observable as `Unknown` without making a report falsely claim an existing
/// classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptNativeFaultKind {
    CurrentDeviceQueryFailed,
    OwnerDeviceEntryFailed,
    AmbientDeviceRestoreFailed,
    FatalDeviceOperation,
    TeardownQuarantined,
    Unknown,
}

/// Exact status family and value retained by the native runtime.
///
/// cuDF and CUDA status values intentionally remain distinct. The engine
/// receives this value from the sibling-owned device-health record; it never
/// parses display text or samples CUDA after the original native failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptNativeStatus {
    Cudf(u32),
    Cuda(i32),
}

/// First structured native fault retained by cuDF's device-health authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptNativeFault {
    pub kind: AttemptNativeFaultKind,
    #[serde(default)]
    pub status: Option<AttemptNativeStatus>,
}

/// Terminal health counters from the authoritative cuDF device record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptDeviceHealth {
    pub status: AttemptDeviceHealthStatus,
    pub fault_count: u64,
    pub rejected_operation_count: u64,
    pub quarantined_teardown_count: u64,
}

/// Bounded cuDF/cuGraph native-fault evidence assembled by the release commit.
///
/// This snapshot is deliberately terminal-only. P3.4 will add same-sequence
/// allocator and ledger evidence; this type does not imply that a native fault
/// was caused by a particular allocation or reconstruct a failure instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptNativeFaultEvidence {
    pub isolation: AttemptFaultIsolation,
    pub teardown_health: AttemptTeardownHealth,
    /// No device health sample is retained when cleanup lost isolation before
    /// the engine could establish a truthful device-local observation.
    #[serde(default)]
    pub device_health: Option<AttemptDeviceHealth>,
    #[serde(default)]
    pub first_fault: Option<AttemptNativeFault>,
}

/// How one bounded resource-evidence field was obtained.
///
/// The release commit does not infer unavailable values, and it never upgrades
/// a device-level observation into an attempt fact merely because it was read
/// near an attempt's terminal boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptEvidenceProvenance {
    /// The value is immutable for, or was joined exactly to, this attempt.
    AttemptExact,
    /// The value is a bounded snapshot of the selected CUDA device only.
    DeviceSnapshot,
    /// The value was sampled after the attempt reached its terminal boundary;
    /// it is not reconstructed as the instant of the original failure.
    SampledAfterFailure,
    /// No truthful value was available at the terminal release commit.
    Unavailable,
}

/// One bounded resource-evidence value together with its provenance.
///
/// `value` is absent exactly for [`AttemptEvidenceProvenance::Unavailable`]
/// in engine-produced reports. Consumers must retain the provenance rather
/// than interpreting an absent value as an ordinal zero or an exact sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptEvidenceField<T> {
    pub provenance: AttemptEvidenceProvenance,
    pub value: Option<T>,
}

impl<T> AttemptEvidenceField<T> {
    pub const fn attempt_exact(value: T) -> Self {
        Self { provenance: AttemptEvidenceProvenance::AttemptExact, value: Some(value) }
    }

    pub const fn device_snapshot(value: T) -> Self {
        Self { provenance: AttemptEvidenceProvenance::DeviceSnapshot, value: Some(value) }
    }

    pub const fn sampled_after_failure(value: T) -> Self {
        Self { provenance: AttemptEvidenceProvenance::SampledAfterFailure, value: Some(value) }
    }

    pub const fn unavailable() -> Self {
        Self { provenance: AttemptEvidenceProvenance::Unavailable, value: None }
    }
}

impl<T> Default for AttemptEvidenceField<T> {
    fn default() -> Self {
        Self::unavailable()
    }
}

/// Exact, separately timed usage sample from an admitted allocation domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptTimedDomainUsage {
    /// Query-relative monotonic time recorded from the attempt clock while
    /// sampling the usage. This is intentionally separate from the outcome
    /// terminal timestamp and from the post-cleanup sample time.
    pub sampled_ns: u64,
    pub usage: AttemptFinalDomainUsage,
}

/// Closed source category copied from one cuDF allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptAllocatorEventSource {
    Physical,
    ReservationLimit,
    Unknown,
}

/// Closed allocator callback action copied from one allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptAllocatorEventAction {
    Decline,
    Retry,
    TrimAndRetry,
    CallbackPanicked,
    Unknown,
}

/// Bounded async-pool counters copied from one allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptAllocatorPoolUsage {
    pub reserved_current_bytes: u64,
    pub reserved_high_bytes: u64,
    pub used_current_bytes: u64,
    pub used_high_bytes: u64,
}

/// One allocator-ring record retained in an attempt resource-evidence block.
///
/// The enclosing evidence field is `attempt_exact` only after the engine's
/// bounded domain-generation join proves ownership. Optional callback fields
/// retain their own provenance so an unavailable callback timestamp, action,
/// or pool sample is never replaced with a zero value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptAllocatorEvent {
    pub sequence: u64,
    pub monotonic_time_ns: AttemptEvidenceField<u64>,
    pub requested_bytes: u64,
    pub source: AttemptAllocatorEventSource,
    pub action: AttemptEvidenceField<AttemptAllocatorEventAction>,
    pub callback_attempt: u32,
    pub trim_retry_count: u32,
    pub pool_usage: AttemptEvidenceField<AttemptAllocatorPoolUsage>,
}

/// A device-level allocator sample which was deliberately not attributed to
/// the attempt. It is diagnostic context only and does not establish cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptUnattributedAllocatorEvent {
    pub device_ordinal: usize,
    /// Query-relative monotonic time at which the release commit sampled this
    /// retained ring record.
    pub sampled_ns: u64,
    pub event: AttemptAllocatorEvent,
}

/// One bounded, truthful resource-evidence block committed with an admitted
/// attempt's post-cleanup terminal disposition.
///
/// Every field carries an explicit provenance. The ledger values are obtained
/// from one lock-held sequence; the profile and grant/domain caps are
/// immutable attempt facts; and both usage samples retain their own times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptResourceEvidence {
    pub ledger_sequence: AttemptEvidenceField<u64>,
    pub ledger_sample_ns: AttemptEvidenceField<u64>,
    pub configured_limit_bytes: AttemptEvidenceField<u64>,
    pub headroom_bytes: AttemptEvidenceField<u64>,
    pub backend_reserve_bytes: AttemptEvidenceField<u64>,
    pub managed_capacity_bytes: AttemptEvidenceField<u64>,
    pub live_grants_bytes: AttemptEvidenceField<u64>,
    pub live_reserved_bytes: AttemptEvidenceField<u64>,
    pub protected_slot: AttemptEvidenceField<bool>,
    pub protected_reserve_bytes: AttemptEvidenceField<u64>,
    pub cache_evictable_resident_bytes: AttemptEvidenceField<u64>,
    pub cache_pinned_in_use_bytes: AttemptEvidenceField<u64>,
    pub active_attempt_count: AttemptEvidenceField<u64>,
    pub grant_id: AttemptEvidenceField<u64>,
    pub grant_cap_bytes: AttemptEvidenceField<u64>,
    pub domain_id: AttemptEvidenceField<u64>,
    pub domain_cap_bytes: AttemptEvidenceField<u64>,
    pub outcome_terminal_usage: AttemptEvidenceField<AttemptTimedDomainUsage>,
    pub post_cleanup_usage: AttemptEvidenceField<AttemptTimedDomainUsage>,
    pub allocator_event: AttemptEvidenceField<AttemptAllocatorEvent>,
    pub unattributed_device_allocator_event: AttemptEvidenceField<AttemptUnattributedAllocatorEvent>,
}

impl AttemptResourceEvidence {
    /// Starts a block with every non-derivable value explicitly unavailable.
    pub const fn unavailable() -> Self {
        Self {
            ledger_sequence: AttemptEvidenceField::unavailable(),
            ledger_sample_ns: AttemptEvidenceField::unavailable(),
            configured_limit_bytes: AttemptEvidenceField::unavailable(),
            headroom_bytes: AttemptEvidenceField::unavailable(),
            backend_reserve_bytes: AttemptEvidenceField::unavailable(),
            managed_capacity_bytes: AttemptEvidenceField::unavailable(),
            live_grants_bytes: AttemptEvidenceField::unavailable(),
            live_reserved_bytes: AttemptEvidenceField::unavailable(),
            protected_slot: AttemptEvidenceField::unavailable(),
            protected_reserve_bytes: AttemptEvidenceField::unavailable(),
            cache_evictable_resident_bytes: AttemptEvidenceField::unavailable(),
            cache_pinned_in_use_bytes: AttemptEvidenceField::unavailable(),
            active_attempt_count: AttemptEvidenceField::unavailable(),
            grant_id: AttemptEvidenceField::unavailable(),
            grant_cap_bytes: AttemptEvidenceField::unavailable(),
            domain_id: AttemptEvidenceField::unavailable(),
            domain_cap_bytes: AttemptEvidenceField::unavailable(),
            outcome_terminal_usage: AttemptEvidenceField::unavailable(),
            post_cleanup_usage: AttemptEvidenceField::unavailable(),
            allocator_event: AttemptEvidenceField::unavailable(),
            unattributed_device_allocator_event: AttemptEvidenceField::unavailable(),
        }
    }
}

/// The immutable cleanup and capacity facts committed after an admitted
/// attempt's device cleanup and before its observation report is sealed.
///
/// This stays crate-internal because the query service is the sole authority
/// that observes the cleanup result and returns (or retains) capacity. The
/// attempt owner projects this value into [`QueryAttemptReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttemptTerminalCommit {
    capacity_disposition: AttemptCapacityDisposition,
    final_domain_usage: Option<AttemptFinalDomainUsage>,
    health_transition: AttemptHealthTransition,
    native_fault_evidence: Option<AttemptNativeFaultEvidence>,
    resource_evidence: Option<AttemptResourceEvidence>,
    alternate_device_eligibility: AttemptAlternateDeviceEligibility,
}

/// Result of the post-quarantine check over the immutable compatible-device
/// set captured at admission. This is internal finalizer input rather than a
/// second scheduler decision: it only gates retry advice in the sealed report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttemptAlternateDeviceEligibility {
    NotApplicable,
    Eligible,
    NoEligibleAlternate,
    Unavailable,
}

impl AttemptTerminalCommit {
    pub(crate) const fn new(
        capacity_disposition: AttemptCapacityDisposition,
        final_domain_usage: Option<AttemptFinalDomainUsage>,
        health_transition: AttemptHealthTransition,
    ) -> Self {
        Self {
            capacity_disposition,
            final_domain_usage,
            health_transition,
            native_fault_evidence: None,
            resource_evidence: None,
            alternate_device_eligibility: AttemptAlternateDeviceEligibility::NotApplicable,
        }
    }

    /// Adds the sole terminal native-health observation from the owning
    /// release commit. The finalizer remains the only report producer.
    pub(crate) const fn with_native_fault_evidence(mut self, evidence: AttemptNativeFaultEvidence) -> Self {
        self.native_fault_evidence = Some(evidence);
        self
    }

    /// Adds the sole bounded resource-evidence block from the owning release
    /// commit. The finalizer remains the only report producer.
    pub(crate) const fn with_resource_evidence(mut self, evidence: AttemptResourceEvidence) -> Self {
        self.resource_evidence = Some(evidence);
        self
    }

    /// Adds the one post-quarantine eligibility result taken from the frozen
    /// compatible-device set. A new admission still owns its own placement;
    /// this value never reserves or selects an alternate device.
    pub(crate) const fn with_alternate_device_eligibility(
        mut self,
        eligibility: AttemptAlternateDeviceEligibility,
    ) -> Self {
        self.alternate_device_eligibility = eligibility;
        self
    }

    pub(crate) const fn capacity_disposition(self) -> AttemptCapacityDisposition {
        self.capacity_disposition
    }

    pub(crate) const fn final_domain_usage(self) -> Option<AttemptFinalDomainUsage> {
        self.final_domain_usage
    }

    pub(crate) const fn health_transition(self) -> AttemptHealthTransition {
        self.health_transition
    }

    pub(crate) const fn native_fault_evidence(self) -> Option<AttemptNativeFaultEvidence> {
        self.native_fault_evidence
    }

    pub(crate) const fn resource_evidence(self) -> Option<AttemptResourceEvidence> {
        self.resource_evidence
    }

    pub(crate) const fn alternate_device_eligibility(self) -> AttemptAlternateDeviceEligibility {
        self.alternate_device_eligibility
    }

    pub(crate) const fn has_consistent_capacity_health_transition(self) -> bool {
        matches!(
            (self.capacity_disposition, self.health_transition),
            (AttemptCapacityDisposition::Released, AttemptHealthTransition::HealthyReuse)
                | (AttemptCapacityDisposition::RetainedQuarantine, AttemptHealthTransition::DeviceQuarantined)
                | (AttemptCapacityDisposition::RetainedQuarantine, AttemptHealthTransition::BackendUnhealthy)
        )
    }

    pub(crate) const fn is_admitted_cleanup_commit(self) -> bool {
        !matches!(self.capacity_disposition, AttemptCapacityDisposition::NotAdmitted)
            && self.has_consistent_capacity_health_transition()
    }
}

/// Closed recovery action advised by one sealed attempt report.
///
/// This is advice only. The engine never retries an attempt from this value;
/// callers must satisfy the paired [`AttemptRetryPrerequisite`] before they
/// choose to submit a new attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptRetryRecovery {
    /// The attempt succeeded and requires no further recovery action.
    NoRetryNeeded,
    /// The attempt was cancelled, dropped, or panicked; do not retry it from
    /// this report.
    DoNotRetry,
    /// Retry may be considered only after bounded capacity evidence supports
    /// waiting for the service to drain.
    Backoff,
    /// The attempt cannot succeed until its query, capability selection, or
    /// configured resource limits change.
    ChangeConfiguration,
    /// Retry may be considered only on another eligible device after the
    /// failed device has been retained in quarantine.
    RetryAnotherEligibleDevice,
    /// The backend must recover before a new attempt can be considered.
    RecoverBackend,
    /// The bounded terminal report cannot truthfully recommend recovery.
    Indeterminate,
}

impl AttemptRetryRecovery {
    /// Stable closed label for a public projection of sealed retry advice.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoRetryNeeded => "no_retry_needed",
            Self::DoNotRetry => "do_not_retry",
            Self::Backoff => "backoff",
            Self::ChangeConfiguration => "change_configuration",
            Self::RetryAnotherEligibleDevice => "retry_another_eligible_device",
            Self::RecoverBackend => "recover_backend",
            Self::Indeterminate => "indeterminate",
        }
    }
}

/// Closed condition that must hold before acting on an
/// [`AttemptRetryRecovery`] recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptRetryPrerequisite {
    None,
    CapacityAvailable,
    ConfigurationChanged,
    QuarantinedDeviceAndEligibleAlternate,
    BackendRecovered,
    EvidenceUnavailable,
}

impl AttemptRetryPrerequisite {
    /// Stable closed label for a public projection of sealed retry advice.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CapacityAvailable => "capacity_available",
            Self::ConfigurationChanged => "configuration_changed",
            Self::QuarantinedDeviceAndEligibleAlternate => "quarantined_device_and_eligible_alternate",
            Self::BackendRecovered => "backend_recovered",
            Self::EvidenceUnavailable => "evidence_unavailable",
        }
    }
}

/// One bounded retry/recovery recommendation sealed with an attempt report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AttemptRetryAdvice {
    pub recovery: AttemptRetryRecovery,
    pub prerequisite: AttemptRetryPrerequisite,
}

#[derive(Deserialize)]
struct AttemptRetryAdviceWire {
    recovery: AttemptRetryRecovery,
    prerequisite: AttemptRetryPrerequisite,
}

impl<'de> Deserialize<'de> for AttemptRetryAdvice {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let wire = AttemptRetryAdviceWire::deserialize(deserializer)?;
        Self::from_parts(wire.recovery, wire.prerequisite).map_err(de::Error::custom)
    }
}

impl AttemptRetryAdvice {
    fn from_parts(
        recovery: AttemptRetryRecovery,
        prerequisite: AttemptRetryPrerequisite,
    ) -> std::result::Result<Self, &'static str> {
        let valid = matches!(
            (recovery, prerequisite),
            (AttemptRetryRecovery::NoRetryNeeded, AttemptRetryPrerequisite::None)
                | (AttemptRetryRecovery::DoNotRetry, AttemptRetryPrerequisite::None)
                | (AttemptRetryRecovery::Backoff, AttemptRetryPrerequisite::CapacityAvailable)
                | (AttemptRetryRecovery::ChangeConfiguration, AttemptRetryPrerequisite::ConfigurationChanged)
                | (
                    AttemptRetryRecovery::RetryAnotherEligibleDevice,
                    AttemptRetryPrerequisite::QuarantinedDeviceAndEligibleAlternate
                )
                | (AttemptRetryRecovery::RecoverBackend, AttemptRetryPrerequisite::BackendRecovered)
                | (AttemptRetryRecovery::Indeterminate, AttemptRetryPrerequisite::EvidenceUnavailable)
        );
        valid.then_some(Self { recovery, prerequisite }).ok_or("retry recovery and prerequisite are inconsistent")
    }

    pub const fn no_retry_needed() -> Self {
        Self { recovery: AttemptRetryRecovery::NoRetryNeeded, prerequisite: AttemptRetryPrerequisite::None }
    }

    pub const fn do_not_retry() -> Self {
        Self { recovery: AttemptRetryRecovery::DoNotRetry, prerequisite: AttemptRetryPrerequisite::None }
    }

    pub const fn indeterminate() -> Self {
        Self {
            recovery: AttemptRetryRecovery::Indeterminate,
            prerequisite: AttemptRetryPrerequisite::EvidenceUnavailable,
        }
    }

    fn backoff() -> Self {
        Self { recovery: AttemptRetryRecovery::Backoff, prerequisite: AttemptRetryPrerequisite::CapacityAvailable }
    }

    fn change_configuration() -> Self {
        Self {
            recovery: AttemptRetryRecovery::ChangeConfiguration,
            prerequisite: AttemptRetryPrerequisite::ConfigurationChanged,
        }
    }

    fn retry_another_eligible_device() -> Self {
        Self {
            recovery: AttemptRetryRecovery::RetryAnotherEligibleDevice,
            prerequisite: AttemptRetryPrerequisite::QuarantinedDeviceAndEligibleAlternate,
        }
    }

    fn recover_backend() -> Self {
        Self {
            recovery: AttemptRetryRecovery::RecoverBackend,
            prerequisite: AttemptRetryPrerequisite::BackendRecovered,
        }
    }

    /// Derives the only retry advice retained for an immutable attempt
    /// report. This consumes only closed report facts and terminal evidence;
    /// it deliberately never turns a coarse temporary error status or an
    /// arbitrary error message into a retry recommendation.
    pub(crate) fn from_sealed_attempt(
        outcome: &RuntimeTerminalOutcome,
        primary_failure: Option<&AttemptFailureSummary>,
        capacity_disposition: AttemptCapacityDisposition,
        health_transition: Option<AttemptHealthTransition>,
        native_fault_evidence: Option<AttemptNativeFaultEvidence>,
        resource_evidence: Option<AttemptResourceEvidence>,
        alternate_device_eligibility: AttemptAlternateDeviceEligibility,
    ) -> Self {
        if requires_backend_recovery(primary_failure, health_transition, native_fault_evidence) {
            return Self::recover_backend();
        }

        match outcome {
            RuntimeTerminalOutcome::Succeeded => Self::no_retry_needed(),
            RuntimeTerminalOutcome::Cancelled | RuntimeTerminalOutcome::Dropped | RuntimeTerminalOutcome::Panicked => {
                Self::do_not_retry()
            },
            RuntimeTerminalOutcome::Failed { .. } => {
                if has_proven_device_local_fault(capacity_disposition, health_transition, native_fault_evidence) {
                    return match alternate_device_eligibility {
                        AttemptAlternateDeviceEligibility::Eligible => Self::retry_another_eligible_device(),
                        AttemptAlternateDeviceEligibility::NoEligibleAlternate => Self::do_not_retry(),
                        AttemptAlternateDeviceEligibility::NotApplicable
                        | AttemptAlternateDeviceEligibility::Unavailable => Self::indeterminate(),
                    };
                }

                let Some(failure) = primary_failure else {
                    return Self::indeterminate();
                };
                if is_configuration_failure(failure) {
                    return Self::change_configuration();
                }
                if has_reason(failure, AttemptFailureFactValue::RequiredFloorExceedsDeviceCapacity) {
                    return Self::change_configuration();
                }
                if is_queue_overload(failure) {
                    return Self::backoff();
                }
                if matches!(
                    failure.code(),
                    AttemptFailureCode::CudfResourceExhausted | AttemptFailureCode::MemoryReservationFailed
                ) {
                    return match exact_allocator_event_source(resource_evidence) {
                        Some(AttemptAllocatorEventSource::ReservationLimit) => Self::change_configuration(),
                        // An exact physical allocator failure establishes
                        // neither that capacity will drain nor that the same
                        // demand can fit on a subsequent attempt. P3.5 has a
                        // concrete backoff rule only for queue overload.
                        Some(AttemptAllocatorEventSource::Physical) => Self::indeterminate(),
                        Some(AttemptAllocatorEventSource::Unknown) | None => Self::indeterminate(),
                    };
                }
                Self::indeterminate()
            },
        }
    }
}

fn has_reason(failure: &AttemptFailureSummary, expected: AttemptFailureFactValue) -> bool {
    failure.facts().iter().any(|fact| fact.key() == AttemptFailureFactKey::Reason && fact.value() == expected)
}

fn is_configuration_failure(failure: &AttemptFailureSummary) -> bool {
    matches!(
        failure.code(),
        AttemptFailureCode::UnsupportedPlanShape
            | AttemptFailureCode::UnsupportedOperator
            | AttemptFailureCode::UnsupportedExpression
            | AttemptFailureCode::UnsupportedType
            | AttemptFailureCode::CudfUnsupported
    )
}

fn is_queue_overload(failure: &AttemptFailureSummary) -> bool {
    failure.code() == AttemptFailureCode::ServiceOverloaded
        && failure.facts().iter().any(|fact| {
            fact.key() == AttemptFailureFactKey::Reason
                && matches!(
                    fact.value(),
                    AttemptFailureFactValue::AdmissionQueueFull
                        | AttemptFailureFactValue::QueueWaitElapsed
                        | AttemptFailureFactValue::MemoryGrantWaitTimeout
                        | AttemptFailureFactValue::ProtectedWaiterNoRunnableDevice
                )
        })
}

fn requires_backend_recovery(
    primary_failure: Option<&AttemptFailureSummary>,
    health_transition: Option<AttemptHealthTransition>,
    native_fault_evidence: Option<AttemptNativeFaultEvidence>,
) -> bool {
    matches!(health_transition, Some(AttemptHealthTransition::BackendUnhealthy))
        || native_fault_evidence.is_some_and(|evidence| evidence.isolation == AttemptFaultIsolation::Unknown)
        || primary_failure.is_some_and(|failure| has_reason(failure, AttemptFailureFactValue::BackendUnhealthy))
}

fn has_proven_device_local_fault(
    capacity_disposition: AttemptCapacityDisposition,
    health_transition: Option<AttemptHealthTransition>,
    native_fault_evidence: Option<AttemptNativeFaultEvidence>,
) -> bool {
    matches!(capacity_disposition, AttemptCapacityDisposition::RetainedQuarantine)
        && matches!(health_transition, Some(AttemptHealthTransition::DeviceQuarantined))
        && native_fault_evidence.as_ref().is_some_and(has_proven_device_local_fault_evidence)
}

/// The native evidence forms that establish a device-local fault rather than
/// a conservative quarantine. The release finalizer and sealed report share
/// this predicate so alternate eligibility cannot be calculated for weaker
/// evidence than the report will accept.
pub(crate) fn has_proven_device_local_fault_evidence(evidence: &AttemptNativeFaultEvidence) -> bool {
    if evidence.isolation != AttemptFaultIsolation::DeviceLocal {
        return false;
    }
    matches!(evidence.teardown_health, AttemptTeardownHealth::StreamSynchronizationFailed)
        || (evidence.device_health.is_some_and(|health| health.status == AttemptDeviceHealthStatus::Unhealthy)
            && evidence.first_fault.is_some())
}

fn exact_allocator_event_source(
    resource_evidence: Option<AttemptResourceEvidence>,
) -> Option<AttemptAllocatorEventSource> {
    let allocator_event = resource_evidence?.allocator_event;
    (allocator_event.provenance == AttemptEvidenceProvenance::AttemptExact)
        .then_some(allocator_event.value?)
        .map(|event| event.source)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptAdmissionFact {
    pub admission_sequence: u64,
    pub admission_wait_ns: u64,
    pub device_ordinal: usize,
    pub grant_bytes: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptTimestamps {
    pub submitted_ns: u64,
    pub planning_started_ns: Option<u64>,
    pub planning_finished_ns: Option<u64>,
    pub admission_requested_ns: Option<u64>,
    pub admitted_ns: Option<u64>,
    pub physical_execution_started_ns: Option<u64>,
    pub gpu_execution_started_ns: Option<u64>,
    pub outcome_terminal_ns: u64,
    pub cleanup_complete_ns: Option<u64>,
    pub report_ready_ns: Option<u64>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryAttemptReport {
    pub schema: String,
    pub attempt_id: AttemptId,
    pub query_id: Option<AttemptQueryId>,
    pub trace_context: Option<AttemptTraceContext>,
    pub disposition: AttemptDisposition,
    pub outcome: RuntimeTerminalOutcome,
    /// Bounded primary terminal-failure facts retained by the attempt
    /// lifecycle. Successful attempts have no primary summary.
    #[serde(default)]
    pub primary_failure: Option<AttemptFailureSummary>,
    /// The sole evidence-gated recovery recommendation derived as this
    /// report is sealed. Legacy reports without this field fail closed to
    /// `indeterminate`; it is never inferred from an error status at read
    /// time.
    #[serde(default = "AttemptRetryAdvice::indeterminate")]
    pub retry_advice: AttemptRetryAdvice,
    pub device_ordinal: Option<usize>,
    pub admission: Option<AttemptAdmissionFact>,
    pub timestamps: AttemptTimestamps,
    pub cleanup_disposition: AttemptCleanupDisposition,
    /// Exact when `cleanup_failure_count_complete` is true. The lifecycle
    /// never publishes a saturated arithmetic result as exact.
    pub cleanup_failure_count: u64,
    #[serde(default = "default_cleanup_failure_count_complete")]
    pub cleanup_failure_count_complete: bool,
    /// First bounded cleanup failure. Further failures affect only the exact
    /// count; timeline retention is intentionally not authoritative.
    #[serde(default)]
    pub first_cleanup_failure: Option<CleanupFailureSummary>,
    pub capacity_disposition: AttemptCapacityDisposition,
    /// Exact final domain usage after admitted cleanup, when sampling
    /// succeeded. `None` is an unavailable sample, never a zero sample.
    #[serde(default)]
    pub final_domain_usage: Option<AttemptFinalDomainUsage>,
    /// Health result of the admitted release commit. `None` means no
    /// trustworthy admitted terminal release-commit fact was recorded (for
    /// example, an unadmitted or legacy attempt).
    #[serde(default)]
    pub health_transition: Option<AttemptHealthTransition>,
    /// Bounded cuDF/cuGraph native status, isolation, and teardown facts from
    /// the same terminal release commit. `None` means the attempt did not own
    /// a device-backed domain, not a fabricated healthy device sample.
    #[serde(default)]
    pub native_fault_evidence: Option<AttemptNativeFaultEvidence>,
    /// Bounded ledger, immutable-cap, domain-usage, and allocator evidence
    /// from the sole terminal release commit. `None` means the attempt did
    /// not own a device-backed admission domain.
    #[serde(default)]
    pub resource_evidence: Option<AttemptResourceEvidence>,
    pub observation_lost_records: u64,
    pub observation_retained_records: u64,
    pub observation_retained_bytes: u64,
    pub observation_truncated: bool,
    pub observation_degraded: bool,
    pub unix_epoch_ns: Option<u64>,
}

const fn default_cleanup_failure_count_complete() -> bool {
    true
}

impl QueryAttemptReport {
    /// Projects the sealed observation footer's completeness facts into the report.
    pub(crate) fn set_observation_completeness(&mut self, footer: &RuntimeObservationFooter) {
        self.observation_lost_records = footer.dropped_by_class.iter().map(|loss| loss.count).sum();
        self.observation_retained_records = footer.retained_records;
        self.observation_retained_bytes = footer.retained_bytes;
        self.observation_truncated = footer.truncated;
        self.observation_degraded = footer.degraded;
    }

    /// Returns the compact, fixed-shape projection used by post-seal public
    /// consumers. The complete report remains the canonical retained record;
    /// this helper is only a read-only projection and never reconstructs facts
    /// from an error or delivery result.
    pub fn sealed_projection(&self) -> SealedAttemptProjection {
        SealedAttemptProjection::from(self)
    }
}

/// Bounded projection of facts from one already-sealed attempt report.
///
/// Unlike [`QueryAttemptReport`], this type is intentionally suitable for
/// fixed-cardinality metrics and small transport metadata. It contains no
/// caller correlations, diagnostic text, failure facts, allocator event data,
/// or query payload. The attempt id is retained solely for delivery surfaces
/// that already identify a single attempt; metric consumers must not use it as
/// a label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SealedAttemptProjection {
    pub schema: &'static str,
    pub attempt_id: AttemptId,
    pub disposition: AttemptDisposition,
    pub outcome: RuntimeTerminalOutcome,
    pub cleanup_disposition: AttemptCleanupDisposition,
    pub cleanup_failure_count: u64,
    pub cleanup_failure_count_complete: bool,
    pub capacity_disposition: AttemptCapacityDisposition,
    pub health_transition: Option<AttemptHealthTransition>,
    pub retry_advice: AttemptRetryAdvice,
    pub observation_truncated: bool,
    pub observation_degraded: bool,
    pub native_fault_evidence_present: bool,
    pub resource_evidence_present: bool,
}

impl From<&QueryAttemptReport> for SealedAttemptProjection {
    fn from(report: &QueryAttemptReport) -> Self {
        Self {
            schema: SEALED_ATTEMPT_PROJECTION_SCHEMA_V1,
            attempt_id: report.attempt_id,
            disposition: report.disposition,
            outcome: report.outcome.clone(),
            cleanup_disposition: report.cleanup_disposition,
            cleanup_failure_count: report.cleanup_failure_count,
            cleanup_failure_count_complete: report.cleanup_failure_count_complete,
            capacity_disposition: report.capacity_disposition,
            health_transition: report.health_transition,
            retry_advice: report.retry_advice,
            observation_truncated: report.observation_truncated,
            observation_degraded: report.observation_degraded,
            native_fault_evidence_present: report.native_fault_evidence.is_some(),
            resource_evidence_present: report.resource_evidence.is_some(),
        }
    }
}
impl QueryAttemptReport {
    pub fn total_ns(&self) -> Option<u64> {
        self.timestamps.report_ready_ns.map(|ready| ready.saturating_sub(self.timestamps.submitted_ns))
    }
    pub fn planning_ns(&self) -> Option<u64> {
        self.timestamps
            .planning_started_ns
            .zip(self.timestamps.planning_finished_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    pub fn queue_wait_ns(&self) -> Option<u64> {
        self.timestamps
            .admission_requested_ns
            .zip(self.timestamps.admitted_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    pub fn physical_execution_ns(&self) -> Option<u64> {
        self.timestamps
            .physical_execution_started_ns
            .map(|start| self.timestamps.outcome_terminal_ns.saturating_sub(start))
    }
    pub fn cleanup_ns(&self) -> Option<u64> {
        self.timestamps.cleanup_complete_ns.map(|end| end.saturating_sub(self.timestamps.outcome_terminal_ns))
    }
}

macro_rules! string_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Creates an identity from its stable, scope-local label.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Returns the stable label.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

string_id!(PlanExecutionId, "Identity for one invocation of a native plan.");
string_id!(RuntimeTraceId, "Optional parent trace identity for one observation session.");
string_id!(RuntimePlanNodeId, "Plan-node identity scoped to one plan execution.");
string_id!(RuntimeTaskId, "Compiled-task identity scoped to one plan execution.");
string_id!(RuntimeReservationId, "Reservation identity scoped to one plan execution.");
string_id!(RuntimeResourceId, "Resident-resource identity scoped to one plan execution.");
string_id!(RuntimeJobId, "Executor-job identity scoped to one query.");
string_id!(RuntimeWorkerId, "Worker identity scoped to one query.");
string_id!(RuntimePipelineEdgeId, "Compiled-pipeline edge identity scoped to one plan execution.");
string_id!(RuntimeBarrierId, "Compiled pipeline-barrier identity scoped to one plan execution.");
string_id!(RuntimeSinkId, "Compiled pipeline-sink identity scoped to one plan execution.");
string_id!(RuntimeReservationGateId, "Reservation-gate identity scoped to one plan execution.");

/// Service-local identity for one admission attempt.
///
/// This value is never a query identity. It is meaningful only for joining
/// admission lifecycle events inside one process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AdmissionRequestId(u64);

impl AdmissionRequestId {
    /// Creates a service-local admission request identity.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the service-local numeric value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AdmissionRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Opaque query-scoped identity for one source location.
///
/// Construction is restricted to the observation session's keyed digest path;
/// callers can inspect or deserialize an exported identity but cannot turn raw
/// source text into an apparently valid identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeSourceLocationId(String);

impl RuntimeSourceLocationId {
    pub(crate) fn from_keyed_digest(value: String) -> Self {
        Self(value)
    }

    /// Returns the opaque query-scoped label.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RuntimeSourceLocationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Immutable canonical query identity for one observation session.
///
/// This delegates to [`QueryId`] so manifests and execution contexts cannot
/// retain different representations of blank or whitespace-only identities.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct RuntimeQueryId(String);

impl RuntimeQueryId {
    /// Creates an identity using the native query-id canonicalization policy.
    pub fn new(value: impl Into<String>) -> Self {
        Self(QueryId::new(value).to_string())
    }

    /// Returns the canonical query identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<QueryId> for RuntimeQueryId {
    fn from(value: QueryId) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for RuntimeQueryId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RuntimeQueryId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for RuntimeQueryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RuntimeQueryId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        String::deserialize(deserializer).map(Self::new)
    }
}

/// Total recorder order assigned to every attempted record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationSequence(u64);

impl ObservationSequence {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the one-based sequence value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Span identity scoped to one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeSpanId(u64);

impl RuntimeSpanId {
    /// Creates a span identity from a query-local value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the query-local value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Attempt number scoped to one plan execution or task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeAttemptId(u32);

impl RuntimeAttemptId {
    /// Creates an attempt identity.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the attempt number.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Identity for one attempt of a compiled task.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskAttemptId {
    pub task_id: RuntimeTaskId,
    pub attempt: RuntimeAttemptId,
}

impl TaskAttemptId {
    /// Creates an attempt identity beneath one compiled task.
    pub const fn new(task_id: RuntimeTaskId, attempt: RuntimeAttemptId) -> Self {
        Self { task_id, attempt }
    }
}

/// Recorder limits copied into the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationLimits {
    /// Maximum retained topology rows and timeline events.
    pub max_retained_records: u64,
    /// Maximum deterministic logical bytes charged to retained records.
    pub max_retained_bytes: u64,
}

/// Clock domain for host runtime observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeClockDomain {
    /// Query-relative time sampled from a monotonic host clock.
    HostMonotonic,
}

/// First JSONL record describing immutable query identity and capture limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationManifest {
    /// Versioned JSONL schema label.
    pub schema: String,
    /// Query-wide diagnostic identity.
    pub query_id: RuntimeQueryId,
    /// Optional parent trace identity.
    pub trace_id: Option<RuntimeTraceId>,
    /// Optional wall-clock anchor captured once when the session is created.
    pub unix_epoch_ns: Option<u64>,
    /// Clock domain used by timeline offsets.
    pub clock_domain: RuntimeClockDomain,
    /// Deterministic per-query retention limits.
    pub limits: RuntimeObservationLimits,
}

/// One native plan node in the static topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanNodeRow {
    pub plan_execution_id: PlanExecutionId,
    pub node_id: RuntimePlanNodeId,
    pub operator_kind: String,
}

/// One directed edge in the static native plan topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanEdgeRow {
    pub plan_execution_id: PlanExecutionId,
    pub from_node_id: RuntimePlanNodeId,
    pub to_node_id: RuntimePlanNodeId,
    pub input_ordinal: u32,
}

/// One compiled task definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskDefinitionRow {
    pub plan_execution_id: PlanExecutionId,
    pub task_id: RuntimeTaskId,
    pub plan_node_id: Option<RuntimePlanNodeId>,
    pub task_kind: String,
}

/// One compiled dependency edge in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompiledEdgeRow {
    pub plan_execution_id: PlanExecutionId,
    pub edge_id: RuntimePipelineEdgeId,
    pub from_task_id: RuntimeTaskId,
    pub to_task_id: RuntimeTaskId,
    pub edge_kind: String,
    pub resource_ids: Vec<RuntimeResourceId>,
    pub gate_ids: Vec<RuntimeReservationGateId>,
}

/// One compiled barrier definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BarrierDefinitionRow {
    pub plan_execution_id: PlanExecutionId,
    pub barrier_id: RuntimeBarrierId,
    pub barrier_kind: String,
    pub upstream_task_ids: Vec<RuntimeTaskId>,
    pub downstream_task_ids: Vec<RuntimeTaskId>,
    pub gate_ids: Vec<RuntimeReservationGateId>,
}

/// The compiled result-sink definition in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SinkDefinitionRow {
    pub plan_execution_id: PlanExecutionId,
    pub sink_id: RuntimeSinkId,
    pub sink_kind: String,
    pub input_task_id: RuntimeTaskId,
    pub gate_ids: Vec<RuntimeReservationGateId>,
}

/// One reservation gate and its owner in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReservationGateRow {
    pub plan_execution_id: PlanExecutionId,
    pub gate_id: RuntimeReservationGateId,
    pub owner_kind: String,
    pub owner_id: String,
    pub reservation_kind: String,
    pub requirement_scope: String,
    pub memory_boundary: String,
    pub required_contract: String,
    pub contract_status: String,
    pub contract_quality: String,
}

/// One GPU executor job definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JobDefinitionRow {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub job_id: RuntimeJobId,
    pub job_kind: String,
    pub device_ordinal: u32,
}

/// One host worker definition used by GPU or source execution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerDefinitionRow {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub worker_id: RuntimeWorkerId,
    pub worker_kind: String,
    pub device_ordinal: Option<u32>,
}

/// Resource domain carried by a static resource definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeResourceDomain {
    Host,
    Device { device_ordinal: u32 },
    Storage,
}

/// One resident resource in the static runtime topology.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RuntimeResourceRow {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub resource_id: RuntimeResourceId,
    pub resource_kind: String,
    pub domain: RuntimeResourceDomain,
    pub producer_task_id: Option<RuntimeTaskId>,
    pub consumer_task_ids: Vec<RuntimeTaskId>,
    pub lifetime: String,
}

/// Static topology accepted by the bounded recorder.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuntimeTopologyRecord {
    PlanNode(PlanNodeRow),
    PlanEdge(PlanEdgeRow),
    TaskDefinition(TaskDefinitionRow),
    CompiledEdge(CompiledEdgeRow),
    BarrierDefinition(BarrierDefinitionRow),
    SinkDefinition(SinkDefinitionRow),
    ReservationGate(ReservationGateRow),
    JobDefinition(JobDefinitionRow),
    WorkerDefinition(WorkerDefinitionRow),
    Resource(RuntimeResourceRow),
}

/// Query-level observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryBoundaryKind {
    Started,
    CleanupStarted,
    CleanupFinished,
    Succeeded,
    Failed,
    Cancelled,
    Dropped,
    Panicked,
}

/// Plan-execution observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanExecutionBoundaryKind {
    Started,
    Finished,
    Failed,
    Cancelled,
}

/// Exactly-once lifecycle state for one compiled task attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TaskLifecycleState {
    Ready,
    Started,
    Finished,
    Failed { error_code: String },
    Cancelled,
    Dropped,
    Panicked,
}

/// Reservation operation recorded at the ownership boundary that applied it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReservationAction {
    Reserve,
    Grow,
    Shrink,
    Transfer,
    Alias,
    Relabel,
    SourceCachePromote,
    SourceCacheEvict,
    Release,
    Deny,
    Rollback,
}

/// Terminal or nonterminal ownership disposition for a reservation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReservationDisposition {
    Applied,
    Granted,
    Denied,
    ExplicitRelease,
    DropRelease,
    UnwindRelease,
    CleanupRelease,
    RollbackRelease,
    AliasDeferred,
    CancelledRelease,
}

/// Query-service admission lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AdmissionLifecycleState {
    Enqueued,
    Admitted,
    Rejected { reason: String },
    TimedOut,
    CancellationRequested,
    Cancelled,
    Released,
    Dropped,
}

/// GPU execution-pool job lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GpuJobLifecycleState {
    Enqueued,
    Started,
    Finished,
    Failed { error_code: String },
    CancellationRequested,
    Cancelled,
    Dropped,
    Panicked,
}

/// How a service stream lease was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamLeaseAcquisition {
    Reused,
    Created,
}

/// Service stream-lease lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StreamLeaseLifecycleState {
    Enqueued,
    Acquired {
        acquisition: StreamLeaseAcquisition,
        host_wait_ns: u64,
    },
    CancellationRequested,
    Cancelled,
    Failed {
        error_code: String,
    },
    /// Query ownership transferred synchronously to pool-managed cleanup.
    Released,
}

/// Explicit correlation available when a service-stream lease is acquired.
///
/// Callers compose this value at execution boundaries; stream acquisition does
/// not consult thread-local or task-local ambient state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStreamLeaseScope {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub task_id: Option<RuntimeTaskId>,
    pub source_worker_id: Option<RuntimeWorkerId>,
    pub gpu_job_id: Option<RuntimeJobId>,
    pub gpu_worker_id: Option<RuntimeWorkerId>,
}

impl RuntimeStreamLeaseScope {
    pub fn for_plan(plan_execution_id: PlanExecutionId) -> Self {
        Self { plan_execution_id: Some(plan_execution_id), ..Self::default() }
    }

    pub fn with_source_worker(mut self, task_id: RuntimeTaskId, worker_id: RuntimeWorkerId) -> Self {
        self.task_id = Some(task_id);
        self.source_worker_id = Some(worker_id);
        self
    }

    pub fn with_gpu_worker(mut self, job_id: RuntimeJobId, worker_id: RuntimeWorkerId) -> Self {
        self.gpu_job_id = Some(job_id);
        self.gpu_worker_id = Some(worker_id);
        self
    }
}

/// Source-read scheduler lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SourceReadLifecycleState {
    Enqueued,
    WorkerStarted { queue_wait_ns: u64 },
    ReadFinished { read_duration_ns: u64 },
    ReadFailed { error_code: String, read_duration_ns: u64 },
    ConsumerJoinStarted,
    ConsumerJoinFinished { join_duration_ns: u64 },
    Cancelled,
    Panicked,
}

/// Scope of one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMeasurementScope {
    Query,
    Task,
    Process,
    DeviceGlobal,
}

/// Provenance of one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMeasurementSource {
    ReservationLedger,
    FrameResidency,
    AllocatorHighWater,
    DeviceSampler,
    StreamPool,
}

/// Evidence quality of one resource measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMeasurementQuality {
    Exact,
    Observed,
    Estimated,
}

macro_rules! runtime_observation_body {
    ($(#[$meta:meta])* $name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $name {
            $(pub $field: $ty),*
        }
    };
}

runtime_observation_body!(
    /// Task-lifecycle payload body retained out of line.
    RuntimeTaskLifecycleObservation {
        attempt_id: TaskAttemptId,
        task_kind: String,
        lifecycle: TaskLifecycleState,
    }
);

runtime_observation_body!(
    /// Admission-lifecycle payload body retained out of line.
    RuntimeAdmissionObservation {
        admission_request_id: AdmissionRequestId,
        lifecycle: AdmissionLifecycleState,
        queue_depth: usize,
        device_ordinal: Option<u32>,
        grant_bytes: Option<u64>,
    }
);

runtime_observation_body!(
    /// GPU-job payload body retained out of line.
    RuntimeGpuJobObservation {
        job_id: RuntimeJobId,
        job_kind: String,
        device_ordinal: u32,
        worker_id: Option<RuntimeWorkerId>,
        lifecycle: GpuJobLifecycleState,
    }
);

runtime_observation_body!(
    /// Stream-lease payload body retained out of line.
    RuntimeStreamLeaseObservation {
        lease_id: RuntimeResourceId,
        device_ordinal: u32,
        scope: RuntimeStreamLeaseScope,
        lifecycle: StreamLeaseLifecycleState,
    }
);

runtime_observation_body!(
    /// Source-read payload body retained out of line.
    RuntimeSourceReadObservation {
        task_id: RuntimeTaskId,
        worker_id: RuntimeWorkerId,
        chunk_index: usize,
        file_index: usize,
        lifecycle: SourceReadLifecycleState,
    }
);

runtime_observation_body!(
    /// Resource-measurement payload body retained out of line.
    RuntimeResourceMeasurementObservation {
        resource_id: RuntimeResourceId,
        scope: RuntimeMeasurementScope,
        source: RuntimeMeasurementSource,
        quality: RuntimeMeasurementQuality,
        measurement_kind: String,
        current_value: u64,
        delta: i128,
        limit: Option<u64>,
        device_ordinal: Option<u32>,
        grant_bytes: Option<u64>,
    }
);

/// Context-rich reservation payload body retained out of line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReservationObservation {
    pub reservation_id: RuntimeReservationId,
    pub lineage_id: RuntimeReservationId,
    pub task_attempt_id: Option<TaskAttemptId>,
    pub task_kind: Option<String>,
    pub action: RuntimeReservationAction,
    pub denied_action: Option<RuntimeReservationAction>,
    pub disposition: RuntimeReservationDisposition,
    pub alias: bool,
    pub reservation_kind: String,
    pub memory_space: String,
    pub memory_boundary: String,
    pub requested_bytes: u64,
    pub granted_bytes: u64,
    pub charged_delta_bytes: i128,
    pub current_reserved_bytes: u64,
    pub budget_bytes: u64,
    pub node_id: Option<String>,
    pub stage: Option<String>,
    pub chunk_index: Option<usize>,
    pub file_index: Option<usize>,
    pub source_location_id: Option<RuntimeSourceLocationId>,
    pub row_groups: Option<Vec<usize>>,
    pub attempt: Option<usize>,
    pub max_attempts: Option<usize>,
    pub resident_resource_id: Option<String>,
    pub source_read_contract_quality: Option<String>,
    pub source_read_planning_bytes: Option<u64>,
    pub source_read_estimated_bytes: Option<u64>,
}

runtime_observation_body!(
    /// Secondary-cleanup-failure payload body retained out of line.
    RuntimeCleanupFailureObservation {
        summary: CleanupFailureSummary,
    }
);

/// Operator-substage category for blocking native primitives.
///
/// Pair events by [`RuntimeTimelineEvent::span_id`]. Kinds must stay short and
/// free of SQL, paths, or unbounded diagnostic text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOperatorStageKind {
    RegexReplacePreflight,
    RegexReplaceExecute,
    ExactCardinalityProbe,
    PreparedGroupbyPrepare,
    PreparedGroupbyAggregate,
    LibcudfGroupby,
    HashPartition,
    PartialAggregate,
    FinalMerge,
    PartitionLocalTopK,
    FinalSort,
    VectorPack,
    VectorNative,
    VectorAssemble,
}

impl RuntimeOperatorStageKind {
    /// Stable snake_case label used in facts and JSON.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RegexReplacePreflight => "regex_replace_preflight",
            Self::RegexReplaceExecute => "regex_replace_execute",
            Self::ExactCardinalityProbe => "exact_cardinality_probe",
            Self::PreparedGroupbyPrepare => "prepared_groupby_prepare",
            Self::PreparedGroupbyAggregate => "prepared_groupby_aggregate",
            Self::LibcudfGroupby => "libcudf_groupby",
            Self::HashPartition => "hash_partition",
            Self::PartialAggregate => "partial_aggregate",
            Self::FinalMerge => "final_merge",
            Self::PartitionLocalTopK => "partition_local_top_k",
            Self::FinalSort => "final_sort",
            Self::VectorPack => "vector_pack",
            Self::VectorNative => "vector_native",
            Self::VectorAssemble => "vector_assemble",
        }
    }
}

impl fmt::Display for RuntimeOperatorStageKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Lifecycle state for one operator-substage span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RuntimeOperatorStageState {
    Started,
    Finished,
    Failed { error_code: String },
    Cancelled,
}

/// Operator-substage payload body retained out of line.
///
/// Safe numerical facts only: no SQL, raw paths, or unbounded strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeOperatorStageObservation {
    pub stage_kind: RuntimeOperatorStageKind,
    pub lifecycle: RuntimeOperatorStageState,
    pub node_id: Option<String>,
    pub task_id: Option<RuntimeTaskId>,
    pub chunk_index: Option<usize>,
    pub partition_index: Option<usize>,
    pub attempt: Option<u32>,
    pub strategy_label: Option<String>,
    pub input_rows: Option<u64>,
    pub output_rows: Option<u64>,
    pub input_bytes: Option<u64>,
    pub output_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_relation_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_dimension: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix_layout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_binding_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_ordinal: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_version_major: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_version_minor: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_version_patch: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounded_control_scalar_events: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounded_control_scalar_bytes: Option<u64>,
}

impl RuntimeOperatorStageObservation {
    /// Creates a stage observation with only the required identity fields set.
    pub fn new(stage_kind: RuntimeOperatorStageKind, lifecycle: RuntimeOperatorStageState) -> Self {
        Self {
            stage_kind,
            lifecycle,
            node_id: None,
            task_id: None,
            chunk_index: None,
            partition_index: None,
            attempt: None,
            strategy_label: None,
            input_rows: None,
            output_rows: None,
            input_bytes: None,
            output_bytes: None,
            reserved_bytes: None,
            vector_algorithm: None,
            vector_relation_role: None,
            vector_dimension: None,
            matrix_layout: None,
            execution_binding_id: None,
            device_ordinal: None,
            native_version_major: None,
            native_version_minor: None,
            native_version_patch: None,
            bounded_control_scalar_events: None,
            bounded_control_scalar_bytes: None,
        }
    }

    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    pub fn with_task_id(mut self, task_id: RuntimeTaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }

    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    pub fn with_partition_index(mut self, partition_index: usize) -> Self {
        self.partition_index = Some(partition_index);
        self
    }

    pub fn with_attempt(mut self, attempt: u32) -> Self {
        self.attempt = Some(attempt);
        self
    }

    pub fn with_strategy_label(mut self, strategy_label: impl Into<String>) -> Self {
        self.strategy_label = Some(strategy_label.into());
        self
    }

    pub fn with_input_rows(mut self, input_rows: u64) -> Self {
        self.input_rows = Some(input_rows);
        self
    }

    pub fn with_output_rows(mut self, output_rows: u64) -> Self {
        self.output_rows = Some(output_rows);
        self
    }

    pub fn with_input_bytes(mut self, input_bytes: u64) -> Self {
        self.input_bytes = Some(input_bytes);
        self
    }

    pub fn with_output_bytes(mut self, output_bytes: u64) -> Self {
        self.output_bytes = Some(output_bytes);
        self
    }

    pub fn with_reserved_bytes(mut self, reserved_bytes: u64) -> Self {
        self.reserved_bytes = Some(reserved_bytes);
        self
    }

    pub fn with_vector_algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.vector_algorithm = Some(algorithm.into());
        self
    }

    pub fn with_vector_relation_role(mut self, role: impl Into<String>) -> Self {
        self.vector_relation_role = Some(role.into());
        self
    }

    pub fn with_vector_dimension(mut self, dimension: usize) -> Self {
        self.vector_dimension = Some(dimension);
        self
    }

    pub fn with_matrix_layout(mut self, layout: impl Into<String>) -> Self {
        self.matrix_layout = Some(layout.into());
        self
    }

    pub fn with_execution_binding_id(mut self, binding_id: u64) -> Self {
        self.execution_binding_id = Some(binding_id);
        self
    }

    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    pub fn with_native_version(mut self, major: u16, minor: u16, patch: u16) -> Self {
        self.native_version_major = Some(major);
        self.native_version_minor = Some(minor);
        self.native_version_patch = Some(patch);
        self
    }

    pub fn with_bounded_control_scalars(mut self, events: u64, bytes: u64) -> Self {
        self.bounded_control_scalar_events = Some(events);
        self.bounded_control_scalar_bytes = Some(bytes);
        self
    }

    pub(crate) fn logical_retained_bytes(&self) -> u64 {
        string_bytes(self.stage_kind.as_str())
            + match &self.lifecycle {
                RuntimeOperatorStageState::Failed { error_code } => string_bytes(error_code),
                _ => 1,
            }
            + self.node_id.as_ref().map_or(0, |value| string_bytes(value))
            + self.task_id.as_ref().map_or(0, |value| string_bytes(value.as_str()))
            + self.strategy_label.as_ref().map_or(0, |value| string_bytes(value))
            + 48
    }
}

/// Typed dynamic event payload. Later phases extend this enum at authoritative boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeObservationPayload {
    QueryBoundary {
        boundary: QueryBoundaryKind,
    },
    PlanExecutionBoundary {
        boundary: PlanExecutionBoundaryKind,
    },
    TaskLifecycle(Box<RuntimeTaskLifecycleObservation>),
    Admission(Box<RuntimeAdmissionObservation>),
    GpuJob(Box<RuntimeGpuJobObservation>),
    StreamLease(Box<RuntimeStreamLeaseObservation>),
    SourceRead(Box<RuntimeSourceReadObservation>),
    ResourceMeasurement(Box<RuntimeResourceMeasurementObservation>),
    Reservation(Box<RuntimeReservationObservation>),
    /// Secondary cleanup failure retained without replacing a primary query error.
    CleanupFailure(Box<RuntimeCleanupFailureObservation>),
    /// Operator-substage lifecycle for blocking native primitives.
    OperatorStage(Box<RuntimeOperatorStageObservation>),
}

/// One dynamic event in recorder total order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimelineEvent {
    pub sequence: ObservationSequence,
    pub monotonic_ns: u64,
    pub plan_execution_id: Option<PlanExecutionId>,
    pub span_id: Option<RuntimeSpanId>,
    pub parent_span_id: Option<RuntimeSpanId>,
    pub payload: RuntimeObservationPayload,
}

/// Query terminal outcome stored outside the bounded record collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeTerminalOutcome {
    Succeeded,
    /// A terminal error classified from the terminal structured error itself.
    Failed {
        error_code: AttemptFailureCode,
        error_status: AttemptFailureStatus,
    },
    Cancelled,
    Dropped,
    Panicked,
}

impl RuntimeTerminalOutcome {
    pub const fn failed(summary: &AttemptFailureSummary) -> Self {
        Self::Failed { error_code: summary.code(), error_status: summary.status() }
    }

    /// Reduces a native error to its closed terminal projection. Attempt
    /// owners must still call the borrowed summary recorder first when they
    /// publish an attempt report.
    pub fn from_query_error(error: &crate::error::QueryError) -> Self {
        if error.code() == crate::error::QueryErrorCode::Cancelled {
            Self::Cancelled
        } else {
            Self::failed(&AttemptFailureSummary::from_native_error(error))
        }
    }

    pub(crate) fn fallback_failure_summary(&self) -> Option<AttemptFailureSummary> {
        match self {
            Self::Succeeded => None,
            Self::Failed { error_code, error_status } => Some(AttemptFailureSummary {
                origin: AttemptFailureOrigin::Native,
                code: *error_code,
                status: *error_status,
                operation: AttemptFailureOperation::Execution,
                text: None,
                facts: Vec::new(),
            }),
            Self::Cancelled => Some(AttemptFailureSummary::control(AttemptFailureCode::Cancelled)),
            Self::Dropped => Some(AttemptFailureSummary::control(AttemptFailureCode::Dropped)),
            Self::Panicked => Some(AttemptFailureSummary::control(AttemptFailureCode::Panicked)),
        }
    }
}

/// Retention class used by explicit loss accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObservationRecordClass {
    PlanNode,
    PlanEdge,
    TaskDefinition,
    CompiledEdge,
    BarrierDefinition,
    SinkDefinition,
    ReservationGate,
    JobDefinition,
    WorkerDefinition,
    Resource,
    TimelineEvent,
}

/// Reason an attempted record was not retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeObservationLossReason {
    RecordLimit,
    ByteLimit,
    Oversized,
    SourceLocationKeyUnavailable,
}

/// Count of dropped records in one class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationClassLoss {
    pub class: RuntimeObservationRecordClass,
    pub count: u64,
}

/// Count of dropped records for one reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationReasonLoss {
    pub reason: RuntimeObservationLossReason,
    pub count: u64,
}

fn serialize_optional_attempt_report<S>(
    report: &Option<std::sync::Arc<QueryAttemptReport>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match report {
        Some(report) => serializer.serialize_some(report.as_ref()),
        None => serializer.serialize_none(),
    }
}

fn deserialize_optional_attempt_report<'de, D>(
    deserializer: D,
) -> Result<Option<std::sync::Arc<QueryAttemptReport>>, D::Error>
where D: serde::Deserializer<'de> {
    Option::<QueryAttemptReport>::deserialize(deserializer).map(|report| report.map(std::sync::Arc::new))
}

/// Final loss, completeness, and terminal metadata.
///
/// An attempt-owned snapshot may also carry the exact immutable terminal
/// report. The `Arc` is shared with attempt completion; it does not create a
/// second report or event store. Keeping it in the exported footer lets the
/// retained JSONL carry the same sealed facts as the other projections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeObservationFooter {
    pub complete: bool,
    pub terminal: Option<RuntimeTerminalOutcome>,
    pub retained_records: u64,
    pub retained_bytes: u64,
    pub first_dropped_sequence: Option<ObservationSequence>,
    pub last_attempted_sequence: Option<ObservationSequence>,
    pub dropped_by_class: Vec<RuntimeObservationClassLoss>,
    pub dropped_by_reason: Vec<RuntimeObservationReasonLoss>,
    pub truncated: bool,
    pub degraded: bool,
    pub poison_recoveries: u64,
    /// The exact sealed attempt report, when this observation belongs to an
    /// execution attempt. This is optional so non-attempt observations retain
    /// their existing JSONL shape.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_attempt_report",
        deserialize_with = "deserialize_optional_attempt_report"
    )]
    pub attempt_report: Option<std::sync::Arc<QueryAttemptReport>>,
    /// At least one recorder sequence or accounting counter saturated.
    ///
    /// Saturated values remain monotonic, while this marker prevents exported
    /// observations from presenting them as exact.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub counter_overflow: bool,
}

/// One independently parseable JSONL record.
// Timeline records retain the stable public enum shape even though reservation
// payloads are deliberately context-rich.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "record_type", content = "data", rename_all = "snake_case")]
pub enum RuntimeObservationRecord {
    Manifest(RuntimeObservationManifest),
    PlanNode { sequence: ObservationSequence, row: PlanNodeRow },
    PlanEdge { sequence: ObservationSequence, row: PlanEdgeRow },
    TaskDefinition { sequence: ObservationSequence, row: TaskDefinitionRow },
    CompiledEdge { sequence: ObservationSequence, row: CompiledEdgeRow },
    BarrierDefinition { sequence: ObservationSequence, row: BarrierDefinitionRow },
    SinkDefinition { sequence: ObservationSequence, row: SinkDefinitionRow },
    ReservationGate { sequence: ObservationSequence, row: ReservationGateRow },
    JobDefinition { sequence: ObservationSequence, row: JobDefinitionRow },
    WorkerDefinition { sequence: ObservationSequence, row: WorkerDefinitionRow },
    Resource { sequence: ObservationSequence, row: RuntimeResourceRow },
    TimelineEvent(RuntimeTimelineEvent),
    Footer(RuntimeObservationFooter),
}

pub(crate) trait LogicalRetainedBytes {
    fn logical_retained_bytes(&self) -> u64;
}

fn string_bytes(value: &str) -> u64 {
    value.len() as u64
}

impl RuntimeTopologyRecord {
    pub(crate) const fn class(&self) -> RuntimeObservationRecordClass {
        match self {
            Self::PlanNode(_) => RuntimeObservationRecordClass::PlanNode,
            Self::PlanEdge(_) => RuntimeObservationRecordClass::PlanEdge,
            Self::TaskDefinition(_) => RuntimeObservationRecordClass::TaskDefinition,
            Self::CompiledEdge(_) => RuntimeObservationRecordClass::CompiledEdge,
            Self::BarrierDefinition(_) => RuntimeObservationRecordClass::BarrierDefinition,
            Self::SinkDefinition(_) => RuntimeObservationRecordClass::SinkDefinition,
            Self::ReservationGate(_) => RuntimeObservationRecordClass::ReservationGate,
            Self::JobDefinition(_) => RuntimeObservationRecordClass::JobDefinition,
            Self::WorkerDefinition(_) => RuntimeObservationRecordClass::WorkerDefinition,
            Self::Resource(_) => RuntimeObservationRecordClass::Resource,
        }
    }
}

impl LogicalRetainedBytes for RuntimeTopologyRecord {
    fn logical_retained_bytes(&self) -> u64 {
        const ENVELOPE: u64 = 24;
        ENVELOPE
            + match self {
                Self::PlanNode(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.node_id.as_str())
                        + string_bytes(&row.operator_kind)
                },
                Self::PlanEdge(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.from_node_id.as_str())
                        + string_bytes(row.to_node_id.as_str())
                        + 4
                },
                Self::TaskDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.task_id.as_str())
                        + row.plan_node_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(&row.task_kind)
                        + 1
                },
                Self::CompiledEdge(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.edge_id.as_str())
                        + string_bytes(row.from_task_id.as_str())
                        + string_bytes(row.to_task_id.as_str())
                        + string_bytes(&row.edge_kind)
                        + row.resource_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.gate_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::BarrierDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.barrier_id.as_str())
                        + string_bytes(&row.barrier_kind)
                        + row.upstream_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.downstream_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + row.gate_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::SinkDefinition(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.sink_id.as_str())
                        + string_bytes(&row.sink_kind)
                        + string_bytes(row.input_task_id.as_str())
                        + row.gate_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                },
                Self::ReservationGate(row) => {
                    string_bytes(row.plan_execution_id.as_str())
                        + string_bytes(row.gate_id.as_str())
                        + string_bytes(&row.owner_kind)
                        + string_bytes(&row.owner_id)
                        + string_bytes(&row.reservation_kind)
                        + string_bytes(&row.requirement_scope)
                        + string_bytes(&row.memory_boundary)
                        + string_bytes(&row.required_contract)
                        + string_bytes(&row.contract_status)
                        + string_bytes(&row.contract_quality)
                },
                Self::JobDefinition(row) => {
                    row.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(row.job_id.as_str())
                        + string_bytes(&row.job_kind)
                        + 4
                },
                Self::WorkerDefinition(row) => {
                    row.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(row.worker_id.as_str())
                        + string_bytes(&row.worker_kind)
                        + 5
                },
                Self::Resource(row) => {
                    row.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + string_bytes(row.resource_id.as_str())
                        + string_bytes(&row.resource_kind)
                        + row.producer_task_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                        + row.consumer_task_ids.iter().map(|id| string_bytes(id.as_str())).sum::<u64>()
                        + string_bytes(&row.lifetime)
                        + match row.domain {
                            RuntimeResourceDomain::Host | RuntimeResourceDomain::Storage => 1,
                            RuntimeResourceDomain::Device { .. } => 5,
                        }
                },
            }
    }
}

impl LogicalRetainedBytes for RuntimeTimelineEvent {
    fn logical_retained_bytes(&self) -> u64 {
        const ENVELOPE: u64 = 42;
        let plan_execution_bytes = self.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()));
        let payload_bytes = match &self.payload {
            RuntimeObservationPayload::QueryBoundary { .. } => 2,
            RuntimeObservationPayload::PlanExecutionBoundary { .. } => 2,
            RuntimeObservationPayload::TaskLifecycle(observation) => {
                string_bytes(observation.attempt_id.task_id.as_str())
                    + 4
                    + string_bytes(&observation.task_kind)
                    + match &observation.lifecycle {
                        TaskLifecycleState::Failed { error_code } => string_bytes(error_code),
                        _ => 1,
                    }
            },
            RuntimeObservationPayload::Admission(observation) => {
                (match &observation.lifecycle {
                    AdmissionLifecycleState::Rejected { reason } => string_bytes(reason),
                    _ => 1,
                }) + 29
            },
            RuntimeObservationPayload::GpuJob(observation) => {
                string_bytes(observation.job_id.as_str())
                    + string_bytes(&observation.job_kind)
                    + observation.worker_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + match &observation.lifecycle {
                        GpuJobLifecycleState::Failed { error_code } => string_bytes(error_code),
                        _ => 1,
                    }
                    + 5
            },
            RuntimeObservationPayload::StreamLease(observation) => {
                string_bytes(observation.lease_id.as_str())
                    + observation.scope.plan_execution_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.task_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.source_worker_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.gpu_job_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + observation.scope.gpu_worker_id.as_ref().map_or(0, |id| string_bytes(id.as_str()))
                    + match &observation.lifecycle {
                        StreamLeaseLifecycleState::Failed { error_code } => string_bytes(error_code),
                        StreamLeaseLifecycleState::Acquired { .. } => 9,
                        _ => 1,
                    }
                    + 4
            },
            RuntimeObservationPayload::SourceRead(observation) => {
                string_bytes(observation.task_id.as_str())
                    + string_bytes(observation.worker_id.as_str())
                    + match &observation.lifecycle {
                        SourceReadLifecycleState::ReadFailed { error_code, .. } => string_bytes(error_code) + 8,
                        SourceReadLifecycleState::WorkerStarted { .. }
                        | SourceReadLifecycleState::ReadFinished { .. }
                        | SourceReadLifecycleState::ConsumerJoinFinished { .. } => 8,
                        _ => 1,
                    }
                    + 16
            },
            RuntimeObservationPayload::ResourceMeasurement(observation) => {
                string_bytes(observation.resource_id.as_str()) + string_bytes(&observation.measurement_kind) + 47
            },
            RuntimeObservationPayload::Reservation(observation) => {
                let RuntimeReservationObservation {
                    reservation_id,
                    lineage_id,
                    task_attempt_id,
                    task_kind,
                    reservation_kind,
                    memory_space,
                    memory_boundary,
                    node_id,
                    stage,
                    source_location_id,
                    row_groups,
                    resident_resource_id,
                    source_read_contract_quality,
                    ..
                } = observation.as_ref();
                string_bytes(reservation_id.as_str())
                    + string_bytes(lineage_id.as_str())
                    + task_attempt_id.as_ref().map_or(0, |attempt| string_bytes(attempt.task_id.as_str()) + 4)
                    + task_kind.as_ref().map_or(0, |value| string_bytes(value))
                    + string_bytes(reservation_kind)
                    + string_bytes(memory_space)
                    + string_bytes(memory_boundary)
                    + node_id.as_ref().map_or(0, |value| string_bytes(value))
                    + stage.as_ref().map_or(0, |value| string_bytes(value))
                    + source_location_id.as_ref().map_or(0, |value| string_bytes(value.as_str()))
                    + row_groups.as_ref().map_or(0, |values| values.len() as u64 * 8)
                    + resident_resource_id.as_ref().map_or(0, |value| string_bytes(value))
                    + source_read_contract_quality.as_ref().map_or(0, |value| string_bytes(value))
                    + 66
            },
            RuntimeObservationPayload::CleanupFailure(observation) => {
                let summary = &observation.summary;
                let failure = summary.failure();
                let text_bytes = failure.text().map_or(0, |text| string_bytes(text.value()));
                64 + text_bytes + u64::try_from(failure.facts().len()).unwrap_or(u64::MAX) * 16
            },
            RuntimeObservationPayload::OperatorStage(observation) => observation.logical_retained_bytes(),
        };
        ENVELOPE + plan_execution_bytes + payload_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure(code: AttemptFailureCode, facts: Vec<AttemptFailureFact>) -> AttemptFailureSummary {
        AttemptFailureSummary {
            origin: AttemptFailureOrigin::Native,
            code,
            status: AttemptFailureStatus::Temporary,
            operation: AttemptFailureOperation::Execution,
            text: None,
            facts,
        }
    }

    fn reason(value: AttemptFailureFactValue) -> AttemptFailureFact {
        AttemptFailureFact { key: AttemptFailureFactKey::Reason, value }
    }

    fn exact_allocator_evidence(source: AttemptAllocatorEventSource) -> AttemptResourceEvidence {
        let mut evidence = AttemptResourceEvidence::unavailable();
        evidence.allocator_event = AttemptEvidenceField::attempt_exact(AttemptAllocatorEvent {
            sequence: 1,
            monotonic_time_ns: AttemptEvidenceField::unavailable(),
            requested_bytes: 1024,
            source,
            action: AttemptEvidenceField::unavailable(),
            callback_attempt: 0,
            trim_retry_count: 0,
            pool_usage: AttemptEvidenceField::unavailable(),
        });
        evidence
    }

    fn device_local_fault_evidence() -> AttemptNativeFaultEvidence {
        AttemptNativeFaultEvidence {
            isolation: AttemptFaultIsolation::DeviceLocal,
            teardown_health: AttemptTeardownHealth::Completed,
            device_health: Some(AttemptDeviceHealth {
                status: AttemptDeviceHealthStatus::Unhealthy,
                fault_count: 1,
                rejected_operation_count: 1,
                quarantined_teardown_count: 0,
            }),
            first_fault: Some(AttemptNativeFault {
                kind: AttemptNativeFaultKind::FatalDeviceOperation,
                status: Some(AttemptNativeStatus::Cuda(700)),
            }),
        }
    }

    fn advice_for(
        outcome: &RuntimeTerminalOutcome,
        primary_failure: Option<&AttemptFailureSummary>,
        capacity_disposition: AttemptCapacityDisposition,
        health_transition: Option<AttemptHealthTransition>,
        native_fault_evidence: Option<AttemptNativeFaultEvidence>,
        resource_evidence: Option<AttemptResourceEvidence>,
        alternate_device_eligibility: AttemptAlternateDeviceEligibility,
    ) -> AttemptRetryAdvice {
        AttemptRetryAdvice::from_sealed_attempt(
            outcome,
            primary_failure,
            capacity_disposition,
            health_transition,
            native_fault_evidence,
            resource_evidence,
            alternate_device_eligibility,
        )
    }

    #[test]
    fn attempt_failure_text_is_utf8_safe_and_bounded() {
        let multibyte = "é".repeat(MAX_ATTEMPT_FAILURE_TEXT_BYTES);
        let text = AttemptFailureText::from_bytes(multibyte.as_bytes());
        assert!(text.value().len() <= MAX_ATTEMPT_FAILURE_TEXT_BYTES);
        assert!(text.truncated());
        assert!(text.value().is_char_boundary(text.value().len()));

        let invalid = AttemptFailureText::from_bytes(&[b'o', b'k', 0xff, b'!']);
        assert!(invalid.invalid_utf8());
        assert!(invalid.value().contains('\u{fffd}'));
        assert!(invalid.value().len() <= MAX_ATTEMPT_FAILURE_TEXT_BYTES);
    }

    #[test]
    fn attempt_failure_text_redacts_uris_credentials_and_sql_shapes() {
        for sensitive in [
            "reader failed at s3://user:password@example.invalid/bucket/object",
            "authorization: bearer secret-value",
            "SELECT* FROM private_table",
        ] {
            let text = AttemptFailureText::from_bytes(sensitive.as_bytes());
            assert_eq!(text.value(), "<redacted>");
            assert!(text.redacted());
        }
    }

    #[test]
    fn failure_summary_deserialization_enforces_all_bounds_before_publication() {
        let oversized_text = serde_json::json!({
            "value": "x".repeat(MAX_ATTEMPT_FAILURE_TEXT_BYTES + 1),
            "truncated": false,
            "redacted": false,
            "invalid_utf8": false,
        });
        assert!(serde_json::from_value::<AttemptFailureText>(oversized_text).is_err());

        let fact = serde_json::json!({ "key": "reason", "value": "backend_closing" });
        let too_many_facts = serde_json::json!({
            "origin": "native",
            "code": "internal",
            "status": "permanent",
            "operation": "execution",
            "facts": vec![fact; MAX_ATTEMPT_FAILURE_FACTS + 1],
        });
        assert!(serde_json::from_value::<AttemptFailureSummary>(too_many_facts).is_err());

        let raw_uri = serde_json::json!({
            "value": "failed at s3://private-bucket/object",
            "truncated": false,
            "redacted": false,
            "invalid_utf8": false,
        });
        assert!(serde_json::from_value::<AttemptFailureText>(raw_uri).is_err());

        let falsely_redacted = serde_json::json!({
            "value": "authorization: bearer secret-value",
            "truncated": false,
            "redacted": true,
            "invalid_utf8": false,
        });
        assert!(serde_json::from_value::<AttemptFailureText>(falsely_redacted).is_err());
    }

    #[test]
    fn resource_evidence_preserves_explicit_unavailable_provenance_through_serde() {
        let mut evidence = AttemptResourceEvidence::unavailable();
        evidence.grant_cap_bytes = AttemptEvidenceField::attempt_exact(4096);
        evidence.ledger_sequence = AttemptEvidenceField::device_snapshot(17);
        evidence.outcome_terminal_usage = AttemptEvidenceField::sampled_after_failure(AttemptTimedDomainUsage {
            sampled_ns: 23,
            usage: AttemptFinalDomainUsage {
                domain_id: 7,
                active: true,
                active_members: 1,
                ceiling_bytes: 4096,
                effective_driver_ceiling_bytes: 4096,
                driver_ceiling_verified: true,
                allocation_limit_bytes: 4096,
                current_bytes: 32,
                peak_bytes: 32,
                total_bytes: 32,
                current_aligned_bytes: 32,
                peak_aligned_bytes: 32,
                total_allocations: 1,
                current_allocations: 1,
                peak_allocations: 1,
                failed_allocation_count: 0,
                largest_failed_allocation_bytes: 0,
            },
        });

        let encoded = serde_json::to_value(evidence).expect("resource evidence serializes");
        assert_eq!(encoded["grant_cap_bytes"]["provenance"], "attempt_exact");
        assert_eq!(encoded["ledger_sequence"]["provenance"], "device_snapshot");
        assert_eq!(encoded["outcome_terminal_usage"]["provenance"], "sampled_after_failure");
        assert_eq!(encoded["allocator_event"]["provenance"], "unavailable");
        assert!(encoded["allocator_event"]["value"].is_null());
        assert_eq!(
            serde_json::from_value::<AttemptResourceEvidence>(encoded).expect("resource evidence deserializes"),
            evidence
        );
    }

    #[test]
    fn sealed_retry_advice_is_exhaustive_and_evidence_gated() {
        let no_commit =
            (AttemptCapacityDisposition::NotAdmitted, None, None, None, AttemptAlternateDeviceEligibility::Unavailable);
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::Succeeded,
                None,
                no_commit.0,
                no_commit.1,
                no_commit.2,
                no_commit.3,
                no_commit.4,
            ),
            AttemptRetryAdvice::no_retry_needed()
        );
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::Cancelled,
                None,
                no_commit.0,
                no_commit.1,
                no_commit.2,
                no_commit.3,
                no_commit.4,
            ),
            AttemptRetryAdvice::do_not_retry()
        );

        let unsupported = failure(AttemptFailureCode::UnsupportedOperator, Vec::new());
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&unsupported),
                Some(&unsupported),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                no_commit.3,
                no_commit.4,
            ),
            AttemptRetryAdvice::change_configuration()
        );

        let queue =
            failure(AttemptFailureCode::ServiceOverloaded, vec![reason(AttemptFailureFactValue::AdmissionQueueFull)]);
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&queue),
                Some(&queue),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                no_commit.3,
                no_commit.4,
            ),
            AttemptRetryAdvice::backoff()
        );
        let impossible_floor = failure(
            AttemptFailureCode::ServiceOverloaded,
            vec![reason(AttemptFailureFactValue::RequiredFloorExceedsDeviceCapacity)],
        );
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&impossible_floor),
                Some(&impossible_floor),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                no_commit.3,
                no_commit.4,
            ),
            AttemptRetryAdvice::change_configuration()
        );

        let resource_exhausted = failure(AttemptFailureCode::CudfResourceExhausted, Vec::new());
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&resource_exhausted),
                Some(&resource_exhausted),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                Some(exact_allocator_evidence(AttemptAllocatorEventSource::ReservationLimit)),
                no_commit.4,
            ),
            AttemptRetryAdvice::change_configuration()
        );
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&resource_exhausted),
                Some(&resource_exhausted),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                Some(exact_allocator_evidence(AttemptAllocatorEventSource::Physical)),
                no_commit.4,
            ),
            AttemptRetryAdvice::indeterminate(),
            "a physical allocator failure alone does not prove capacity will become available"
        );
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&resource_exhausted),
                Some(&resource_exhausted),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                None,
                no_commit.4,
            ),
            AttemptRetryAdvice::indeterminate(),
            "OOM status without exact allocator evidence cannot recommend a retry"
        );

        let temporary = failure(AttemptFailureCode::CudfTemporaryFailure, Vec::new());
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&temporary),
                Some(&temporary),
                no_commit.0,
                no_commit.1,
                no_commit.2,
                no_commit.3,
                no_commit.4,
            ),
            AttemptRetryAdvice::indeterminate(),
            "a temporary code alone is not retry evidence"
        );

        let unknown_isolation = AttemptNativeFaultEvidence {
            isolation: AttemptFaultIsolation::Unknown,
            teardown_health: AttemptTeardownHealth::CallbackAbandoned,
            device_health: None,
            first_fault: None,
        };
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::Succeeded,
                None,
                AttemptCapacityDisposition::RetainedQuarantine,
                Some(AttemptHealthTransition::BackendUnhealthy),
                Some(unknown_isolation),
                None,
                AttemptAlternateDeviceEligibility::NotApplicable,
            ),
            AttemptRetryAdvice::recover_backend(),
            "unknown isolation is a backend recovery requirement even when cleanup did not replace the outcome"
        );

        let local_fault = device_local_fault_evidence();
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&temporary),
                Some(&temporary),
                AttemptCapacityDisposition::RetainedQuarantine,
                Some(AttemptHealthTransition::DeviceQuarantined),
                Some(local_fault),
                None,
                AttemptAlternateDeviceEligibility::Eligible,
            ),
            AttemptRetryAdvice::retry_another_eligible_device()
        );
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&temporary),
                Some(&temporary),
                AttemptCapacityDisposition::RetainedQuarantine,
                Some(AttemptHealthTransition::DeviceQuarantined),
                Some(local_fault),
                None,
                AttemptAlternateDeviceEligibility::NoEligibleAlternate,
            ),
            AttemptRetryAdvice::do_not_retry()
        );
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&temporary),
                Some(&temporary),
                AttemptCapacityDisposition::RetainedQuarantine,
                Some(AttemptHealthTransition::DeviceQuarantined),
                Some(local_fault),
                None,
                AttemptAlternateDeviceEligibility::Unavailable,
            ),
            AttemptRetryAdvice::indeterminate()
        );

        let stream_synchronization_fault = AttemptNativeFaultEvidence {
            isolation: AttemptFaultIsolation::DeviceLocal,
            teardown_health: AttemptTeardownHealth::StreamSynchronizationFailed,
            device_health: Some(AttemptDeviceHealth {
                status: AttemptDeviceHealthStatus::Healthy,
                fault_count: 0,
                rejected_operation_count: 0,
                quarantined_teardown_count: 1,
            }),
            first_fault: None,
        };
        assert_eq!(
            advice_for(
                &RuntimeTerminalOutcome::failed(&temporary),
                Some(&temporary),
                AttemptCapacityDisposition::RetainedQuarantine,
                Some(AttemptHealthTransition::DeviceQuarantined),
                Some(stream_synchronization_fault),
                None,
                AttemptAlternateDeviceEligibility::Eligible,
            ),
            AttemptRetryAdvice::retry_another_eligible_device(),
            "a stream-synchronization failure is itself proven device-local cleanup evidence"
        );
    }

    #[test]
    fn retry_advice_deserialization_rejects_inconsistent_pairs() {
        let error = serde_json::from_str::<AttemptRetryAdvice>(
            r#"{"recovery":"retry_another_eligible_device","prerequisite":"none"}"#,
        )
        .expect_err("retry advice must reject an impossible recovery/prerequisite pair");

        assert!(error.to_string().contains("retry recovery and prerequisite are inconsistent"));
    }
}
