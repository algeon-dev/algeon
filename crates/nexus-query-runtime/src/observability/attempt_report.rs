//! Typed sealed execution-attempt reports and terminal facts.

use std::fmt;

use nexus_query_planning::{
    error::AttemptFailureCode,
    schedule_projection::{AttemptComposedScheduleProjection, AttemptId},
};
use serde::{
    Deserialize, Serialize,
    de::{self, SeqAccess, Visitor},
};

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

macro_rules! attempt_correlation {
    ($name:ident, $limit:ident, $field:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);
        impl $name {
            pub fn new(value: impl Into<String>) -> nexus_query_planning::error::Result<Self> {
                let value = value.into();
                if value.trim().is_empty() || value.len() > $limit {
                    return Err(nexus_query_planning::error::QueryError::new(
                        nexus_query_planning::error::QueryErrorCode::InvalidAttemptCorrelation,
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
    FrozenResident,
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
    pub const fn from_query_error_status(status: nexus_query_planning::error::QueryErrorStatus) -> Self {
        match status {
            nexus_query_planning::error::QueryErrorStatus::Permanent => Self::Permanent,
            nexus_query_planning::error::QueryErrorStatus::Temporary => Self::Temporary,
            nexus_query_planning::error::QueryErrorStatus::Persistent => Self::Persistent,
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
    AllocationSource,
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
    /// The requested query cap does not fit any eligible device once resident
    /// ownership is deducted. Retrying the same request cannot succeed.
    QueryCapExceedsResidentCapacity,
    BackendUnhealthy,
    CugraphRmmOutOfMemory,
    CugraphRmmBadAlloc,
    CugraphStdBadAlloc,
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

    impl Visitor<'_> for BoundedTextVisitor {
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
    /// [`nexus_query_planning::error::QueryError`] keeps that complete source text on its
    /// ordinary error path.
    pub fn from_native_error(error: &nexus_query_planning::error::QueryError) -> Self {
        let mut facts = Vec::with_capacity(2);
        let reason = match error.fact_value("reason") {
            Some("backend_closing") => Some(AttemptFailureFactValue::BackendClosing),
            Some("deadline_expired") => Some(AttemptFailureFactValue::DeadlineExpired),
            Some("caller_cancelled") => Some(AttemptFailureFactValue::CallerCancelled),
            Some("admission_queue_full") => Some(AttemptFailureFactValue::AdmissionQueueFull),
            Some("queue_wait_elapsed") => Some(AttemptFailureFactValue::QueueWaitElapsed),
            Some("memory_grant_wait_timeout") => Some(AttemptFailureFactValue::MemoryGrantWaitTimeout),
            Some("protected_waiter_no_runnable_device") => {
                Some(AttemptFailureFactValue::ProtectedWaiterNoRunnableDevice)
            },
            Some("required_floor_exceeds_device_capacity") | Some("impossible_requirement") => {
                Some(AttemptFailureFactValue::RequiredFloorExceedsDeviceCapacity)
            },
            Some("query_cap_exceeds_resident_capacity") => {
                Some(AttemptFailureFactValue::QueryCapExceedsResidentCapacity)
            },
            Some("backend_unhealthy") => Some(AttemptFailureFactValue::BackendUnhealthy),
            _ => None,
        };
        if let Some(value) = reason {
            facts.push(AttemptFailureFact { key: AttemptFailureFactKey::Reason, value });
        }
        let allocation_source = match error.cugraph_allocation_source() {
            Some(nexus_query_planning::error::CugraphAllocationSource::RmmOutOfMemory) => {
                Some(AttemptFailureFactValue::CugraphRmmOutOfMemory)
            },
            Some(nexus_query_planning::error::CugraphAllocationSource::RmmBadAlloc) => {
                Some(AttemptFailureFactValue::CugraphRmmBadAlloc)
            },
            Some(nexus_query_planning::error::CugraphAllocationSource::StdBadAlloc) => {
                Some(AttemptFailureFactValue::CugraphStdBadAlloc)
            },
            Some(nexus_query_planning::error::CugraphAllocationSource::Unknown) | None => None,
        };
        if let Some(value) = allocation_source {
            facts.push(AttemptFailureFact { key: AttemptFailureFactKey::AllocationSource, value });
        }
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
    /// summary. The exact count remains in the enclosing attempt report; the
    /// field is required so decoding never fabricates exactness.
    complete: bool,
}

impl CleanupFailureSummary {
    pub fn from_native_error(phase: CleanupFailurePhase, error: &nexus_query_planning::error::QueryError) -> Self {
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

    #[doc(hidden)]
    pub fn mark_incomplete(&mut self) {
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
/// This snapshot is deliberately terminal-only. Same-sequence allocator and
/// ledger evidence is carried separately; this type does not imply that a
/// native fault was caused by a particular allocation or reconstruct a failure
/// instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptNativeFaultEvidence {
    pub isolation: AttemptFaultIsolation,
    pub teardown_health: AttemptTeardownHealth,
    /// No device health sample is retained when cleanup lost isolation before
    /// the engine could establish a truthful device-local observation.
    pub device_health: Option<AttemptDeviceHealth>,
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
    /// Kept separate from logical usage so physical pool pages never widen the grant.
    pub pool_usage: Option<AttemptAllocatorPoolUsage>,
}

/// Closed source category copied from one cuDF allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptAllocatorEventSource {
    Physical,
    ReservationLimit,
    AllocationLimit,
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
    pub resident_owner_bytes: AttemptEvidenceField<u64>,
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
            resident_owner_bytes: AttemptEvidenceField::unavailable(),
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
#[doc(hidden)]
pub struct AttemptTerminalCommit {
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
#[doc(hidden)]
pub enum AttemptAlternateDeviceEligibility {
    NotApplicable,
    Eligible,
    NoEligibleAlternate,
    Unavailable,
}

impl AttemptTerminalCommit {
    #[doc(hidden)]
    pub const fn new(
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
    #[doc(hidden)]
    pub const fn with_native_fault_evidence(mut self, evidence: AttemptNativeFaultEvidence) -> Self {
        self.native_fault_evidence = Some(evidence);
        self
    }

    /// Adds the sole bounded resource-evidence block from the owning release
    /// commit. The finalizer remains the only report producer.
    #[doc(hidden)]
    pub const fn with_resource_evidence(mut self, evidence: AttemptResourceEvidence) -> Self {
        self.resource_evidence = Some(evidence);
        self
    }

    /// Adds the one post-quarantine eligibility result taken from the frozen
    /// compatible-device set. A new admission still owns its own placement;
    /// this value never reserves or selects an alternate device.
    #[doc(hidden)]
    pub const fn with_alternate_device_eligibility(mut self, eligibility: AttemptAlternateDeviceEligibility) -> Self {
        self.alternate_device_eligibility = eligibility;
        self
    }

    #[doc(hidden)]
    pub const fn capacity_disposition(self) -> AttemptCapacityDisposition {
        self.capacity_disposition
    }

    #[doc(hidden)]
    pub const fn final_domain_usage(self) -> Option<AttemptFinalDomainUsage> {
        self.final_domain_usage
    }

    #[doc(hidden)]
    pub const fn health_transition(self) -> AttemptHealthTransition {
        self.health_transition
    }

    #[doc(hidden)]
    pub const fn native_fault_evidence(self) -> Option<AttemptNativeFaultEvidence> {
        self.native_fault_evidence
    }

    #[doc(hidden)]
    pub const fn resource_evidence(self) -> Option<AttemptResourceEvidence> {
        self.resource_evidence
    }

    #[doc(hidden)]
    pub const fn alternate_device_eligibility(self) -> AttemptAlternateDeviceEligibility {
        self.alternate_device_eligibility
    }

    #[doc(hidden)]
    pub const fn has_consistent_capacity_health_transition(self) -> bool {
        matches!(
            (self.capacity_disposition, self.health_transition),
            (AttemptCapacityDisposition::Released, AttemptHealthTransition::HealthyReuse)
                | (AttemptCapacityDisposition::FrozenResident, AttemptHealthTransition::HealthyReuse)
                | (AttemptCapacityDisposition::RetainedQuarantine, AttemptHealthTransition::DeviceQuarantined)
                | (AttemptCapacityDisposition::RetainedQuarantine, AttemptHealthTransition::BackendUnhealthy)
        )
    }

    #[doc(hidden)]
    pub const fn is_admitted_cleanup_commit(self) -> bool {
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
    #[doc(hidden)]
    pub fn from_sealed_attempt(
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
                if has_reason(failure, AttemptFailureFactValue::RequiredFloorExceedsDeviceCapacity)
                    || has_reason(failure, AttemptFailureFactValue::QueryCapExceedsResidentCapacity)
                {
                    // An impossible cap is not queue pressure: backing off
                    // cannot change the outcome, so the caller must change the
                    // cap, the device, or release resident owners.
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
                        Some(
                            AttemptAllocatorEventSource::ReservationLimit
                            | AttemptAllocatorEventSource::AllocationLimit,
                        ) => Self::change_configuration(),
                        // An exact physical allocator failure establishes
                        // neither that capacity will drain nor that the same
                        // demand can fit on a subsequent attempt. Backoff is
                        // defined only for queue overload.
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
            | AttemptFailureCode::QueryMemoryLimitExceeded
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
#[doc(hidden)]
pub fn has_proven_device_local_fault_evidence(evidence: &AttemptNativeFaultEvidence) -> bool {
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

/// Final finite-overtake state captured when a queued admission receives its
/// immutable grant.
///
/// This is intentionally a sealed controller fact rather than a runner-side
/// estimate. `submission_sequence` is FIFO identity, independent of the
/// later admission sequence. `protection_device_ordinal` records the
/// ledger-local protection that was held at placement, if finite overtakes
/// had exhausted the waiter's allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptAdmissionWaitFact {
    pub submission_sequence: u64,
    pub overtake_limit: u32,
    pub overtakes_consumed: u32,
    pub overtakes_remaining: u32,
    pub protection_device_ordinal: Option<usize>,
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
    pub primary_failure: Option<AttemptFailureSummary>,
    /// The sole evidence-gated recovery recommendation derived as this
    /// report is sealed. It is required and never inferred from an error
    /// status at read time.
    pub retry_advice: AttemptRetryAdvice,
    pub device_ordinal: Option<usize>,
    pub admission: Option<AttemptAdmissionFact>,
    /// Final queue/overtake facts for an admitted request. `None` means the
    /// execution path did not retain a queue-backed admission handle; it is
    /// never reconstructed from snapshots after the report is sealed.
    pub admission_wait: Option<AttemptAdmissionWaitFact>,
    /// Immutable engine-owned composed schedule, when this admitted attempt
    /// used the composed graph execution contract. Ordinary attempts leave it
    /// absent; it is never reconstructed from runtime observations.
    #[serde(default)]
    pub composed_schedule: Option<AttemptComposedScheduleProjection>,
    pub timestamps: AttemptTimestamps,
    pub cleanup_disposition: AttemptCleanupDisposition,
    /// Exact when `cleanup_failure_count_complete` is true. The lifecycle
    /// never publishes a saturated arithmetic result as exact.
    pub cleanup_failure_count: u64,
    /// Required so decoding never fabricates an exact cleanup-failure count.
    pub cleanup_failure_count_complete: bool,
    /// First bounded cleanup failure. Further failures affect only the exact
    /// count; timeline retention is intentionally not authoritative.
    pub first_cleanup_failure: Option<CleanupFailureSummary>,
    pub capacity_disposition: AttemptCapacityDisposition,
    /// Exact final domain usage after admitted cleanup, when sampling
    /// succeeded. `None` is an unavailable sample, never a zero sample.
    pub final_domain_usage: Option<AttemptFinalDomainUsage>,
    /// Health result of the admitted release commit. `None` means no
    /// trustworthy admitted terminal release-commit fact was recorded (for
    /// example, an unadmitted attempt).
    pub health_transition: Option<AttemptHealthTransition>,
    /// Bounded cuDF/cuGraph native status, isolation, and teardown facts from
    /// the same terminal release commit. `None` means the attempt did not own
    /// a device-backed domain, not a fabricated healthy device sample.
    pub native_fault_evidence: Option<AttemptNativeFaultEvidence>,
    /// Bounded ledger, immutable-cap, domain-usage, and allocator evidence
    /// from the sole terminal release commit. `None` means the attempt did
    /// not own a device-backed admission domain.
    pub resource_evidence: Option<AttemptResourceEvidence>,
    pub observation_lost_records: u64,
    pub observation_retained_records: u64,
    pub observation_retained_bytes: u64,
    pub observation_truncated: bool,
    pub observation_degraded: bool,
    pub unix_epoch_ns: Option<u64>,
}

impl QueryAttemptReport {
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

/// Stable fixed-cardinality labels derived from one sealed attempt projection.
///
/// Every consumer that renders attempt facts with labels (DataFusion metrics,
/// Flight query stats, and future bounded exporters) must use this projection
/// rather than duplicating enum-to-string mappings at its own boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedAttemptLabels {
    pub disposition: &'static str,
    pub outcome: &'static str,
    pub cleanup_disposition: &'static str,
    pub cleanup_failure_count_complete: &'static str,
    pub capacity_disposition: &'static str,
    pub health_transition: &'static str,
    pub retry_recovery: &'static str,
    pub retry_prerequisite: &'static str,
    pub observation_truncated: &'static str,
    pub observation_degraded: &'static str,
    pub native_fault_evidence: &'static str,
    pub resource_evidence: &'static str,
}

impl SealedAttemptProjection {
    /// Returns the sole stable label mapping for bounded post-seal consumers.
    pub const fn labels(&self) -> SealedAttemptLabels {
        SealedAttemptLabels {
            disposition: match self.disposition {
                AttemptDisposition::Unplanned => "unplanned",
                AttemptDisposition::CpuFallback => "cpu_fallback",
                AttemptDisposition::PartialNative => "partial_native",
                AttemptDisposition::Native => "native",
            },
            outcome: self.outcome.stable_label(),
            cleanup_disposition: match self.cleanup_disposition {
                AttemptCleanupDisposition::NotRequired => "not_required",
                AttemptCleanupDisposition::Completed => "completed",
                AttemptCleanupDisposition::CompletedWithFailures => "completed_with_failures",
            },
            cleanup_failure_count_complete: if self.cleanup_failure_count_complete { "true" } else { "false" },
            capacity_disposition: match self.capacity_disposition {
                AttemptCapacityDisposition::NotAdmitted => "not_admitted",
                AttemptCapacityDisposition::Released => "released",
                AttemptCapacityDisposition::FrozenResident => "frozen_resident",
                AttemptCapacityDisposition::RetainedQuarantine => "retained_quarantine",
            },
            health_transition: match self.health_transition {
                None => "none",
                Some(AttemptHealthTransition::HealthyReuse) => "healthy_reuse",
                Some(AttemptHealthTransition::DeviceQuarantined) => "device_quarantined",
                Some(AttemptHealthTransition::BackendUnhealthy) => "backend_unhealthy",
            },
            retry_recovery: self.retry_advice.recovery.as_str(),
            retry_prerequisite: self.retry_advice.prerequisite.as_str(),
            observation_truncated: if self.observation_truncated { "true" } else { "false" },
            observation_degraded: if self.observation_degraded { "true" } else { "false" },
            native_fault_evidence: if self.native_fault_evidence_present { "present" } else { "absent" },
            resource_evidence: if self.resource_evidence_present { "present" } else { "absent" },
        }
    }
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
    pub fn post_planning_pre_admission_ns(&self) -> Option<u64> {
        self.timestamps
            .planning_finished_ns
            .zip(self.timestamps.admission_requested_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    pub fn admitted_to_physical_execution_ns(&self) -> Option<u64> {
        self.timestamps
            .admitted_ns
            .zip(self.timestamps.physical_execution_started_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    pub fn physical_to_gpu_dispatch_ns(&self) -> Option<u64> {
        self.timestamps
            .physical_execution_started_ns
            .zip(self.timestamps.gpu_execution_started_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    pub fn admitted_to_gpu_dispatch_ns(&self) -> Option<u64> {
        self.timestamps
            .admitted_ns
            .zip(self.timestamps.gpu_execution_started_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    pub fn submitted_to_gpu_dispatch_ns(&self) -> Option<u64> {
        self.timestamps.gpu_execution_started_ns.map(|end| end.saturating_sub(self.timestamps.submitted_ns))
    }
    pub fn gpu_dispatch_to_terminal_ns(&self) -> Option<u64> {
        self.timestamps.gpu_execution_started_ns.map(|start| self.timestamps.outcome_terminal_ns.saturating_sub(start))
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
    /// Stable closed label for bounded post-seal projections.
    pub const fn stable_label(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
            Self::Panicked => "panicked",
        }
    }

    pub const fn failed(summary: &AttemptFailureSummary) -> Self {
        Self::Failed { error_code: summary.code(), error_status: summary.status() }
    }

    /// Reduces a native error to its closed terminal projection. Attempt
    /// owners must still call the borrowed summary recorder first when they
    /// publish an attempt report.
    pub fn from_query_error(error: &nexus_query_planning::error::QueryError) -> Self {
        if error.code() == nexus_query_planning::error::QueryErrorCode::Cancelled {
            Self::Cancelled
        } else {
            Self::failed(&AttemptFailureSummary::from_native_error(error))
        }
    }

    /// Returns the bounded primary-failure projection that must accompany a
    /// terminal native error. Ordinary caller cancellation intentionally
    /// remains a text-free control outcome, while an expired deadline retains
    /// its stable reason beside the same `Cancelled` terminal outcome.
    #[doc(hidden)]
    pub fn primary_failure_from_query_error(
        error: &nexus_query_planning::error::QueryError,
    ) -> Option<AttemptFailureSummary> {
        match error.code() {
            nexus_query_planning::error::QueryErrorCode::Cancelled
                if error.fact_value("reason") != Some("deadline_expired") =>
            {
                None
            },
            _ => Some(AttemptFailureSummary::from_native_error(error)),
        }
    }

    #[doc(hidden)]
    pub const fn from_primary_failure(summary: &AttemptFailureSummary) -> Self {
        match summary.code() {
            AttemptFailureCode::Cancelled => Self::Cancelled,
            _ => Self::failed(summary),
        }
    }

    #[doc(hidden)]
    pub fn fallback_failure_summary(&self) -> Option<AttemptFailureSummary> {
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
#[cfg(test)]
mod tests;
