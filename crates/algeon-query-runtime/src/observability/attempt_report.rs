//! Typed sealed execution-attempt reports and terminal facts.

use std::fmt;

use query_planning::error::{
    AllocationSource, AttemptFailureCode, QueryError, QueryErrorCode, QueryErrorStatus, Result,
};
use serde::{
    Deserialize, Serialize,
    de::{self, SeqAccess, Visitor},
};

use crate::observability::{AttemptComposedScheduleProjection, AttemptId};

/// Schema label of [`QueryAttemptReport`]. Decoding rejects every other label.
pub const QUERY_ATTEMPT_REPORT_SCHEMA: &str = "query_engine.query_attempt_report";
/// Schema label for the bounded public projection of a sealed attempt report.
///
/// The projection deliberately omits correlations, diagnostics, allocator
/// events, and other payloads that are not suitable for fixed-cardinality
/// consumers such as DataFusion metrics and retained Flight control metadata.
pub const SEALED_ATTEMPT_PROJECTION_SCHEMA: &str = "query_engine.sealed_attempt_projection";
/// Maximum UTF-8 byte length for caller-supplied attempt query correlation.
pub const MAX_ATTEMPT_QUERY_ID_BYTES: usize = 256;
/// Maximum UTF-8 byte length for caller-supplied opaque trace context.
pub const MAX_ATTEMPT_TRACE_CONTEXT_BYTES: usize = 1024;

macro_rules! attempt_correlation {
    ($name:ident, $limit:ident, $field:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);
        impl $name {
            /// Preserves the supplied UTF-8 correlation after validation.
            ///
            /// Returns `InvalidAttemptCorrelation` for blank text or text
            /// exceeding this correlation type's byte limit. Whitespace is
            /// used to detect blank values and is otherwise preserved.
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                if value.trim().is_empty() || value.len() > $limit {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidAttemptCorrelation,
                        "query_engine.attempt.correlation",
                        "attempt correlation must be non-blank and within its byte limit",
                    )
                    .with_fact("field", $field)
                    .with_fact("max_bytes", $limit)
                    .with_fact("reason", if value.trim().is_empty() { "blank" } else { "too_long" }));
                }
                Ok(Self(value))
            }
            /// Returns the original validated correlation text.
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
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where D: serde::Deserializer<'de> {
                // Validation is part of the value contract on every
                // construction path, including report decoding.
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(de::Error::custom)
            }
        }
    };
}
attempt_correlation!(
    AttemptQueryId,
    MAX_ATTEMPT_QUERY_ID_BYTES,
    "query_id",
    "Opaque caller query correlation, limited to 256 UTF-8 bytes and required to be non-blank."
);
attempt_correlation!(
    AttemptTraceContext,
    MAX_ATTEMPT_TRACE_CONTEXT_BYTES,
    "trace_context",
    "Opaque caller trace correlation, limited to 1024 UTF-8 bytes and required to be non-blank."
);

/// Execution coverage assigned after planning, independently of terminal outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    /// The attempt did not obtain a classified physical plan.
    Unplanned,
    /// The physical plan executes through DataFusion without native operators.
    CpuFallback,
    /// The physical plan contains native and DataFusion execution.
    PartialNative,
    /// The physical plan has native execution coverage.
    Native,
}

/// Cleanup result fixed before the terminal report is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptCleanupDisposition {
    /// No admitted GPU cleanup was required.
    NotRequired,
    /// Required cleanup finished without recorded failures.
    Completed,
    /// Cleanup finished with at least one recorded failure.
    CompletedWithFailures,
}

/// Ownership of admitted capacity after the terminal release commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptCapacityDisposition {
    /// The attempt acquired no GPU admission grant.
    NotAdmitted,
    /// Capacity returned to the device ledger after cleanup.
    Released,
    /// Capacity remains owned by a resident result after attempt cleanup.
    FrozenResident,
    /// Capacity remains reserved because cleanup or device health prevents reuse.
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
    /// Cleanup and health evidence permit capacity reuse on the selected device.
    HealthyReuse,
    /// A fault isolated to the selected device closed its admission ledger.
    DeviceQuarantined,
    /// Fault isolation was unavailable and admission closed across the backend.
    BackendUnhealthy,
}

/// Maximum retained UTF-8 bytes in one stable failure-text projection.
///
/// The source error remains untouched on its returned path. This is only the
/// bounded, redacted summary that may enter an immutable attempt report.
pub(crate) const MAX_ATTEMPT_FAILURE_TEXT_BYTES: usize = 256;
#[cfg(test)]
const MAX_ATTEMPT_FAILURE_TEXT_INSPECTION_BYTES: usize = 1024;
const MAX_ATTEMPT_FAILURE_FACTS: usize = 4;

/// Owning layer that classified a stable attempt failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureOrigin {
    /// Classified by the native planning, runtime or execution layer.
    Native,
    /// Classified by a DataFusion boundary.
    Datafusion,
    /// Classified by attempt lifecycle control, such as backend closure.
    AttemptControl,
}

/// Closed stability/recovery status for a report-visible failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureStatus {
    /// The owning failure taxonomy provides no native recovery status.
    NotApplicable,
    /// The condition requires changed input, configuration or code.
    Permanent,
    /// The condition may clear; retry still requires sealed evidence.
    Temporary,
    /// A dependency condition is known or observed to persist.
    Persistent,
}

impl AttemptFailureStatus {
    pub(crate) const fn from_query_error_status(status: QueryErrorStatus) -> Self {
        match status {
            QueryErrorStatus::Permanent => Self::Permanent,
            QueryErrorStatus::Temporary => Self::Temporary,
            QueryErrorStatus::Persistent => Self::Persistent,
        }
    }
}

/// Coarse owning operation for a stable failure projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureOperation {
    /// Logical or physical plan construction.
    Planning,
    /// Capacity placement, queuing or grant acquisition.
    Admission,
    /// Execution after planning or admission.
    Execution,
    /// Release of resources after a terminal outcome.
    Cleanup,
    /// A DataFusion-owned failure without a more specific native operation.
    Datafusion,
    /// Attempt lifecycle control outside a native operation.
    AttemptControl,
}

/// Stable allowlisted fact keys for bounded failure summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureFactKey {
    /// Closed classification of the condition that ended the attempt.
    Reason,
    /// Native allocator family that reported allocation failure.
    AllocationSource,
}

/// Stable allowlisted fact values for bounded failure summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureFactValue {
    /// Admission or attempt creation encountered a closing backend.
    BackendClosing,
    /// The invocation's absolute monotonic deadline elapsed.
    DeadlineExpired,
    /// The caller requested cooperative cancellation.
    CallerCancelled,
    /// The shared admission queue had reached its waiter bound.
    AdmissionQueueFull,
    /// A queued request exceeded the configured maximum queue wait.
    QueueWaitElapsed,
    /// The admission wait ended before a memory grant was available.
    MemoryGrantWaitTimeout,
    /// The required plan memory exceeds the eligible device capacity.
    RequiredFloorExceedsDeviceCapacity,
    /// The requested query cap does not fit any eligible device once resident
    /// ownership is deducted. Retrying the same request cannot succeed.
    QueryCapExceedsResidentCapacity,
    /// Device or backend health prevents new admission.
    BackendUnhealthy,
    /// cuGraph reported an RMM out-of-memory failure.
    CugraphRmmOutOfMemory,
    /// cuGraph reported an RMM allocation exception.
    CugraphRmmBadAlloc,
    /// cuGraph reported a standard allocation exception.
    CugraphStdBadAlloc,
}

/// One closed, bounded fact retained beside an attempt failure summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptFailureFact {
    key: AttemptFailureFactKey,
    value: AttemptFailureFactValue,
}

impl AttemptFailureFact {
    /// Returns the allowlisted classification represented by this fact.
    pub const fn key(&self) -> AttemptFailureFactKey {
        self.key
    }

    /// Returns the closed value retained by the failure projection.
    pub const fn value(&self) -> AttemptFailureFactValue {
        self.value
    }
}

/// Bounded, UTF-8-safe and redacted diagnostic text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct AttemptFailureText {
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

fn deserialize_bounded_attempt_failure_text_value<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where D: serde::Deserializer<'de> {
    struct BoundedTextVisitor;

    impl Visitor<'_> for BoundedTextVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a UTF-8 string no longer than {MAX_ATTEMPT_FAILURE_TEXT_BYTES} bytes")
        }

        fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
        where E: de::Error {
            if value.len() > MAX_ATTEMPT_FAILURE_TEXT_BYTES {
                return Err(E::custom("attempt failure text exceeds its byte bound"));
            }
            Ok(value.to_owned())
        }

        fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
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
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let wire = AttemptFailureTextWire::deserialize(deserializer)?;
        if wire.value.len() > MAX_ATTEMPT_FAILURE_TEXT_BYTES {
            return Err(de::Error::custom("attempt failure text exceeds its byte bound"));
        }
        if wire.redacted {
            if wire.value != "<redacted>" {
                return Err(de::Error::custom("redacted attempt failure text must not retain source text"));
            }
        } else if attempt_failure_text_requires_redaction(&wire.value.to_ascii_lowercase()) {
            return Err(de::Error::custom("attempt failure text requires redaction"));
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
    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
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

    #[cfg(test)]
    pub(crate) const fn truncated(&self) -> bool {
        self.truncated
    }

    #[cfg(test)]
    pub(crate) const fn redacted(&self) -> bool {
        self.redacted
    }

    #[cfg(test)]
    pub(crate) const fn invalid_utf8(&self) -> bool {
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

#[cfg(test)]
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

fn deserialize_bounded_attempt_failure_facts<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<AttemptFailureFact>, D::Error>
where D: serde::Deserializer<'de> {
    struct BoundedFactsVisitor;

    impl<'de> Visitor<'de> for BoundedFactsVisitor {
        type Value = Vec<AttemptFailureFact>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "at most {MAX_ATTEMPT_FAILURE_FACTS} attempt failure facts")
        }

        fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
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
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let wire = AttemptFailureSummaryWire::deserialize(deserializer)?;
        Self::from_parts(wire.origin, wire.code, wire.status, wire.operation, wire.text, wire.facts)
            .map_err(de::Error::custom)
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
    ) -> std::result::Result<Self, &'static str> {
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
    /// [`query_planning::error::QueryError`] keeps that complete source text on its
    /// ordinary error path.
    pub fn from_native_error(error: &QueryError, operation: AttemptFailureOperation) -> Self {
        let mut facts = Vec::with_capacity(2);
        let reason = match error.fact_value("reason") {
            Some("backend_closing") => Some(AttemptFailureFactValue::BackendClosing),
            Some("deadline_expired") => Some(AttemptFailureFactValue::DeadlineExpired),
            Some("caller_cancelled") => Some(AttemptFailureFactValue::CallerCancelled),
            Some("admission_queue_full") => Some(AttemptFailureFactValue::AdmissionQueueFull),
            Some("queue_wait_elapsed") => Some(AttemptFailureFactValue::QueueWaitElapsed),
            Some("memory_grant_wait_timeout") => Some(AttemptFailureFactValue::MemoryGrantWaitTimeout),
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
        let allocation_source = match error.context().allocation_source {
            Some(AllocationSource::RmmOutOfMemory) => Some(AttemptFailureFactValue::CugraphRmmOutOfMemory),
            Some(AllocationSource::RmmBadAlloc) => Some(AttemptFailureFactValue::CugraphRmmBadAlloc),
            Some(AllocationSource::StdBadAlloc) => Some(AttemptFailureFactValue::CugraphStdBadAlloc),
            Some(AllocationSource::Unknown) | None => None,
        };
        if let Some(value) = allocation_source {
            facts.push(AttemptFailureFact { key: AttemptFailureFactKey::AllocationSource, value });
        }
        Self {
            origin: AttemptFailureOrigin::Native,
            code: AttemptFailureCode::from_query_error_code(error.code()),
            status: AttemptFailureStatus::from_query_error_status(error.status()),
            operation,
            text: None,
            facts,
        }
    }

    /// Closed root-adapter fallback. It intentionally retains no adapter-side
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

    /// Returns a text-free admission failure for a closing backend.
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

    /// Returns the layer that classified this failure.
    pub const fn origin(&self) -> AttemptFailureOrigin {
        self.origin
    }

    /// Returns the closed machine identity retained from the primary failure.
    pub const fn code(&self) -> AttemptFailureCode {
        self.code
    }

    /// Returns the source condition classification, without granting retry permission.
    pub const fn status(&self) -> AttemptFailureStatus {
        self.status
    }

    /// Returns the lifecycle operation in which the failure was recorded.
    pub const fn operation(&self) -> AttemptFailureOperation {
        self.operation
    }

    pub(crate) fn text(&self) -> Option<&AttemptFailureText> {
        self.text.as_ref()
    }

    /// Returns allowlisted facts in unique deterministic key order.
    /// Arbitrary error facts and source text are excluded.
    pub fn facts(&self) -> &[AttemptFailureFact] {
        &self.facts
    }
}

/// Closed lifecycle phase at which a secondary cleanup failure occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupFailurePhase {
    /// Scheduler cycle cleanup.
    SchedulingCycle,
    /// Cleanup after runnable-task selection.
    RunnableTaskSelection,
    /// Cleanup after task input and resource preparation.
    TaskPreparation,
    /// Cleanup after native task execution returned an error.
    TaskExecution,
    /// Cleanup after a task panic was caught.
    TaskPanic,
    /// Cleanup during task result completion.
    TaskCompletion,
    /// Release of resources retained by a completed task.
    CompletedResourceRelease,
    /// Final scheduler validation before release.
    FinalValidation,
    /// Synchronization of the admitted query stream failed.
    QueryServiceStreamSynchronization,
    /// An admitted query cleanup callback was abandoned.
    QueryServiceCallbackAbandoned,
    /// Cleanup owned by the admission service.
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
    /// Projects one native cleanup failure without copying its message or source.
    /// The summary is initially complete; subsequent cleanup failures can make
    /// the retained first summary incomplete.
    pub fn from_native_error(phase: CleanupFailurePhase, error: &QueryError) -> Self {
        Self {
            phase,
            failure: AttemptFailureSummary::from_native_error(error, AttemptFailureOperation::Cleanup),
            complete: true,
        }
    }

    pub(crate) const fn stream_synchronization_failed() -> Self {
        Self {
            phase: CleanupFailurePhase::QueryServiceStreamSynchronization,
            failure: AttemptFailureSummary::control(AttemptFailureCode::StreamSynchronizationFailed),
            complete: true,
        }
    }

    pub(crate) const fn callback_abandoned() -> Self {
        Self {
            phase: CleanupFailurePhase::QueryServiceCallbackAbandoned,
            failure: AttemptFailureSummary::control(AttemptFailureCode::CleanupCallbackAbandoned),
            complete: true,
        }
    }

    /// Returns the cleanup boundary that recorded the failure.
    pub const fn phase(&self) -> CleanupFailurePhase {
        self.phase
    }

    /// Returns the bounded failure identity and allowlisted facts.
    pub const fn failure(&self) -> &AttemptFailureSummary {
        &self.failure
    }

    /// Returns whether this first summary accounts for all recorded cleanup failures.
    /// The enclosing report carries the count and its separate completeness flag.
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
/// all-zero domain. This includes releases without an allocation domain,
/// failed final samples, and unknown-isolation cleanup paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptFinalDomainUsage {
    /// Identity of the admitted allocation domain.
    pub domain_id: u64,
    pub(crate) active: bool,
    pub(crate) active_members: u64,
    pub(crate) ceiling_bytes: u64,
    pub(crate) allocation_limit_bytes: u64,
    /// Logical live allocation bytes at the final sample.
    pub current_bytes: u64,
    /// Highest logical live allocation bytes observed over the domain lifetime.
    pub peak_bytes: u64,
    pub(crate) total_bytes: u64,
    pub(crate) current_aligned_bytes: u64,
    /// Highest allocator-aligned live allocation bytes over the domain lifetime.
    pub peak_aligned_bytes: u64,
    pub(crate) total_allocations: u64,
    pub(crate) current_allocations: u64,
    pub(crate) peak_allocations: u64,
    pub(crate) failed_allocation_count: u64,
    pub(crate) largest_failed_allocation_bytes: u64,
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
/// entire backend without guessing a device owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttemptFaultIsolation {
    DeviceLocal,
    Unknown,
}

/// Cleanup state retained with terminal native-health evidence.
///
/// This is an engine-owned projection of the release worker's result. It does
/// not infer a CUDA status from a later runtime query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttemptTeardownHealth {
    Completed,
    StreamSynchronizationFailed,
    CallbackAbandoned,
}

/// Health state read from cuDF's process-lifetime device-health authority at
/// the terminal release commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttemptDeviceHealthStatus {
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
pub(crate) enum AttemptNativeFaultKind {
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
pub(crate) enum AttemptNativeStatus {
    Cudf(u32),
    Cuda(i32),
}

/// First structured native fault retained by cuDF's device-health authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AttemptNativeFault {
    pub(crate) kind: AttemptNativeFaultKind,
    pub(crate) status: Option<AttemptNativeStatus>,
}

/// Terminal health counters from the authoritative cuDF device record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AttemptDeviceHealth {
    pub(crate) status: AttemptDeviceHealthStatus,
    pub(crate) fault_count: u64,
    pub(crate) rejected_operation_count: u64,
    pub(crate) quarantined_teardown_count: u64,
}

/// Bounded cuDF/cuGraph native-fault evidence assembled by the release commit.
///
/// This snapshot is deliberately terminal-only. Same-sequence allocator and
/// ledger evidence is carried separately; this type does not imply that a
/// native fault was caused by a particular allocation or reconstruct a failure
/// instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptNativeFaultEvidence {
    pub(crate) isolation: AttemptFaultIsolation,
    pub(crate) teardown_health: AttemptTeardownHealth,
    /// No device health sample is retained when cleanup lost isolation before
    /// the engine could establish a truthful device-local observation.
    pub(crate) device_health: Option<AttemptDeviceHealth>,
    pub(crate) first_fault: Option<AttemptNativeFault>,
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
    /// The value was sampled after the attempt reached its terminal boundary,
    /// whatever its outcome; it is not reconstructed as the instant of an
    /// earlier failure. Outcome lives in the report's terminal fields.
    SampledAtTerminalBoundary,
    /// No truthful value was available at the terminal release commit.
    Unavailable,
}

/// One bounded resource-evidence value together with its provenance.
///
/// In engine-produced reports, `value` is absent if and only if the provenance
/// is [`AttemptEvidenceProvenance::Unavailable`]. Consumers must retain the
/// provenance. An absent value is neither an ordinal zero nor an exact sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptEvidenceField<T> {
    /// Scope and sampling classification that consumers must retain with the value.
    pub provenance: AttemptEvidenceProvenance,
    /// Observed value, or `None` when provenance is `Unavailable`.
    pub value: Option<T>,
}

impl<T> AttemptEvidenceField<T> {
    pub(crate) const fn attempt_exact(value: T) -> Self {
        Self { provenance: AttemptEvidenceProvenance::AttemptExact, value: Some(value) }
    }

    pub(crate) const fn device_snapshot(value: T) -> Self {
        Self { provenance: AttemptEvidenceProvenance::DeviceSnapshot, value: Some(value) }
    }

    pub(crate) const fn sampled_at_terminal_boundary(value: T) -> Self {
        Self { provenance: AttemptEvidenceProvenance::SampledAtTerminalBoundary, value: Some(value) }
    }

    pub(crate) const fn unavailable() -> Self {
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
    pub(crate) sampled_ns: u64,
    pub(crate) usage: AttemptFinalDomainUsage,
    /// Kept separate from logical usage so physical pool pages never widen the grant.
    pub pool_usage: Option<AttemptAllocatorPoolUsage>,
}

/// Closed source category copied from one cuDF allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptAllocatorEventSource {
    /// The underlying allocator could not satisfy the request.
    Physical,
    /// The request exceeded the allocation domain reservation ceiling.
    ReservationLimit,
    /// The request exceeded a logical allocation limit.
    AllocationLimit,
    /// The native source category is unavailable or unrecognized.
    Unknown,
}

/// Closed allocator callback action copied from one allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttemptAllocatorEventAction {
    Decline,
    Retry,
    TrimAndRetry,
    CallbackPanicked,
    Unknown,
}

/// Bounded async-pool counters copied from one allocator-ring record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptAllocatorPoolUsage {
    pub(crate) reserved_current_bytes: u64,
    /// Highest bytes reserved by the CUDA async allocation pool.
    pub reserved_high_bytes: u64,
    pub(crate) used_current_bytes: u64,
    /// Highest bytes used within the CUDA async allocation pool.
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
    /// Monotonic identity in the device allocator event ring.
    pub sequence: u64,
    pub(crate) monotonic_time_ns: AttemptEvidenceField<u64>,
    /// Size in bytes of the allocation request associated with this event.
    pub requested_bytes: u64,
    /// Closed allocator failure source retained from the native record.
    pub source: AttemptAllocatorEventSource,
    pub(crate) action: AttemptEvidenceField<AttemptAllocatorEventAction>,
    pub(crate) callback_attempt: u32,
    pub(crate) trim_retry_count: u32,
    pub(crate) pool_usage: AttemptEvidenceField<AttemptAllocatorPoolUsage>,
}

/// A device-level allocator sample which was deliberately not attributed to
/// the attempt. It is diagnostic context only and does not establish cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptUnattributedAllocatorEvent {
    pub(crate) device_ordinal: usize,
    /// Query-relative monotonic time at which the release commit sampled this
    /// retained ring record.
    pub(crate) sampled_ns: u64,
    pub(crate) event: AttemptAllocatorEvent,
}

/// One bounded, truthful resource-evidence block committed with an admitted
/// attempt's post-cleanup terminal disposition.
///
/// Every field carries an explicit provenance. The ledger values are obtained
/// from one lock-held sequence; the profile and grant/domain caps are
/// immutable attempt facts; and both usage samples retain their own times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptResourceEvidence {
    /// Device ledger transition sequence from the terminal release snapshot.
    pub ledger_sequence: AttemptEvidenceField<u64>,
    pub(crate) ledger_sample_ns: AttemptEvidenceField<u64>,
    pub(crate) configured_limit_bytes: AttemptEvidenceField<u64>,
    pub(crate) headroom_bytes: AttemptEvidenceField<u64>,
    pub(crate) backend_reserve_bytes: AttemptEvidenceField<u64>,
    pub(crate) managed_capacity_bytes: AttemptEvidenceField<u64>,
    pub(crate) live_grants_bytes: AttemptEvidenceField<u64>,
    pub(crate) resident_owner_bytes: AttemptEvidenceField<u64>,
    /// Cached resident bytes eligible for eviction in the device snapshot.
    pub cache_evictable_resident_bytes: AttemptEvidenceField<u64>,
    /// Cached resident bytes pinned by active use in the device snapshot.
    pub cache_pinned_in_use_bytes: AttemptEvidenceField<u64>,
    pub(crate) active_attempt_count: AttemptEvidenceField<u64>,
    /// Immutable identity of this attempt's admission grant.
    pub grant_id: AttemptEvidenceField<u64>,
    pub(crate) grant_cap_bytes: AttemptEvidenceField<u64>,
    /// Immutable identity of this attempt's allocation domain.
    pub domain_id: AttemptEvidenceField<u64>,
    pub(crate) domain_cap_bytes: AttemptEvidenceField<u64>,
    /// Domain sample taken at the outcome terminal boundary, before cleanup.
    pub outcome_terminal_usage: AttemptEvidenceField<AttemptTimedDomainUsage>,
    pub(crate) post_cleanup_usage: AttemptEvidenceField<AttemptTimedDomainUsage>,
    /// Allocator event attributed to this attempt by a bounded domain-generation join.
    pub allocator_event: AttemptEvidenceField<AttemptAllocatorEvent>,
    /// Device allocator context whose ownership could not be attributed to the attempt.
    pub unattributed_device_allocator_event: AttemptEvidenceField<AttemptUnattributedAllocatorEvent>,
}

impl AttemptResourceEvidence {
    /// Starts a block with every non-derivable value explicitly unavailable.
    pub(crate) const fn unavailable() -> Self {
        Self {
            ledger_sequence: AttemptEvidenceField::unavailable(),
            ledger_sample_ns: AttemptEvidenceField::unavailable(),
            configured_limit_bytes: AttemptEvidenceField::unavailable(),
            headroom_bytes: AttemptEvidenceField::unavailable(),
            backend_reserve_bytes: AttemptEvidenceField::unavailable(),
            managed_capacity_bytes: AttemptEvidenceField::unavailable(),
            live_grants_bytes: AttemptEvidenceField::unavailable(),
            resident_owner_bytes: AttemptEvidenceField::unavailable(),
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
/// set captured at admission. This is internal finalizer input that only gates
/// retry advice in the sealed report; it adds no second scheduler decision.
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
                | (AttemptCapacityDisposition::FrozenResident, AttemptHealthTransition::HealthyReuse)
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
    /// No prerequisite accompanies success or a do-not-retry recommendation.
    None,
    /// Capacity evidence permits a new admission after waiting.
    CapacityAvailable,
    /// The query, capability selection or configured limit has changed.
    ConfigurationChanged,
    /// The failed device remains quarantined and another device is eligible.
    QuarantinedDeviceAndEligibleAlternate,
    /// Backend recovery completed before a new attempt.
    BackendRecovered,
    /// The sealed evidence cannot establish a recovery prerequisite.
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
    /// Recovery action supported by the terminal evidence.
    pub recovery: AttemptRetryRecovery,
    /// Condition required before submitting a new attempt for that action.
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

    /// Returns the success recommendation with no recovery prerequisite.
    pub const fn no_retry_needed() -> Self {
        Self { recovery: AttemptRetryRecovery::NoRetryNeeded, prerequisite: AttemptRetryPrerequisite::None }
    }

    /// Returns the terminal control recommendation with no recovery prerequisite.
    pub const fn do_not_retry() -> Self {
        Self { recovery: AttemptRetryRecovery::DoNotRetry, prerequisite: AttemptRetryPrerequisite::None }
    }

    /// Returns an indeterminate recommendation because recovery evidence is unavailable.
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
                if matches!(failure.code(), AttemptFailureCode::Query(QueryErrorCode::MemoryReservationFailed)) {
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
        AttemptFailureCode::Query(QueryErrorCode::UnsupportedPipelineContract)
            | AttemptFailureCode::Query(QueryErrorCode::UnsupportedOperator)
            | AttemptFailureCode::Query(QueryErrorCode::UnsupportedExpression)
            | AttemptFailureCode::Query(QueryErrorCode::UnsupportedType)
            | AttemptFailureCode::Query(QueryErrorCode::CudfUnsupported)
            | AttemptFailureCode::Query(QueryErrorCode::QueryMemoryLimitExceeded)
    )
}

fn is_queue_overload(failure: &AttemptFailureSummary) -> bool {
    failure.code() == AttemptFailureCode::Query(QueryErrorCode::ServiceOverloaded)
        && failure.facts().iter().any(|fact| {
            fact.key() == AttemptFailureFactKey::Reason
                && matches!(
                    fact.value(),
                    AttemptFailureFactValue::AdmissionQueueFull
                        | AttemptFailureFactValue::QueueWaitElapsed
                        | AttemptFailureFactValue::MemoryGrantWaitTimeout
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

/// The native evidence forms that establish a device-local fault, as distinct
/// from a conservative quarantine. The release finalizer and sealed report share
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
/// Immutable device grant and wait duration committed by admission.
pub struct AttemptAdmissionFact {
    /// Grant publication sequence, independent of FIFO submission identity.
    pub admission_sequence: u64,
    /// Elapsed nanoseconds from admission submission to the grant.
    pub admission_wait_ns: u64,
    /// CUDA device selected for the immutable grant.
    pub device_ordinal: usize,
    /// Immutable admitted query capacity in bytes.
    pub grant_bytes: u64,
}

/// Final queue state captured when a queued admission receives its immutable
/// grant.
///
/// This is a sealed controller fact and is not a runner-side estimate.
/// `submission_sequence` is FIFO identity, independent of the later
/// admission sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptAdmissionWaitFact {
    /// FIFO admission queue identity fixed at submission.
    pub submission_sequence: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Attempt lifecycle times in nanoseconds from one monotonic attempt clock.
/// Optional boundaries are absent when they were not recorded. These values
/// are elapsed times, without a wall-clock epoch.
pub struct AttemptTimestamps {
    /// Time at attempt creation.
    pub submitted_ns: u64,
    /// Time planning began, or `None` when planning never started.
    pub planning_started_ns: Option<u64>,
    /// Time planning finished, or `None` when no finish was recorded.
    pub planning_finished_ns: Option<u64>,
    /// Sampled when the attempt claims its one admission, the first step of
    /// the query-service admission request.
    pub admission_requested_ns: Option<u64>,
    /// Time the immutable grant was published, or `None` when unadmitted.
    pub admitted_ns: Option<u64>,
    /// Time physical execution began, or `None` when it never began.
    pub physical_execution_started_ns: Option<u64>,
    /// Time first GPU dispatch began, or `None` when there was no dispatch.
    pub gpu_execution_started_ns: Option<u64>,
    /// Time the first terminal execution outcome was committed.
    pub outcome_terminal_ns: u64,
    /// Time cleanup completed, or `None` when that boundary was not recorded.
    pub cleanup_complete_ns: Option<u64>,
    /// Time the terminal report was fixed for publication, or `None` when absent.
    pub report_ready_ns: Option<u64>,
}
fn deserialize_query_attempt_report_schema<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where D: serde::Deserializer<'de> {
    let schema = String::deserialize(deserializer)?;
    if schema != QUERY_ATTEMPT_REPORT_SCHEMA {
        return Err(de::Error::custom(format!(
            "unsupported attempt report schema {schema:?}; expected {QUERY_ATTEMPT_REPORT_SCHEMA:?}"
        )));
    }
    Ok(schema)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Immutable terminal attempt record published after cleanup and sealing.
///
/// The report preserves the execution outcome separately from cleanup, health
/// and capacity disposition. Retry advice is derived from those committed
/// facts; delivery failures do not change the report. Duration helpers
/// return `None` for missing boundaries and saturate reversed intervals at zero.
pub struct QueryAttemptReport {
    #[serde(deserialize_with = "deserialize_query_attempt_report_schema")]
    /// Schema identity accepted by report decoding.
    pub schema: String,
    /// Runtime identity shared by the attempt and its observations.
    pub attempt_id: AttemptId,
    /// Validated caller query correlation, or `None` when none was supplied.
    pub query_id: Option<AttemptQueryId>,
    /// Validated opaque caller trace correlation, or `None` when none was supplied.
    pub trace_context: Option<AttemptTraceContext>,
    /// Planning coverage of this attempt, independent of execution success.
    pub disposition: AttemptDisposition,
    /// First terminal execution outcome committed by the attempt owner.
    pub outcome: RuntimeTerminalOutcome,
    /// Bounded primary terminal-failure facts retained by the attempt
    /// lifecycle. Successful attempts have no primary summary.
    pub primary_failure: Option<AttemptFailureSummary>,
    /// The sole evidence-gated recovery recommendation derived as this
    /// report is sealed. It is required and never inferred from an error
    /// status at read time.
    pub retry_advice: AttemptRetryAdvice,
    /// Selected CUDA device, or `None` when no device-backed admission occurred.
    pub device_ordinal: Option<usize>,
    /// Immutable grant facts, or `None` when the attempt was unadmitted.
    pub admission: Option<AttemptAdmissionFact>,
    /// Final queue facts for an admitted request. `None` means the
    /// execution path did not retain a queue-backed admission handle; it is
    /// never reconstructed from snapshots after the report is sealed.
    pub admission_wait: Option<AttemptAdmissionWaitFact>,
    /// Immutable engine-owned composed schedule, when this admitted attempt
    /// used the composed graph execution contract. Ordinary attempts leave it
    /// absent; it is never reconstructed from runtime observations.
    #[serde(default)]
    pub composed_schedule: Option<AttemptComposedScheduleProjection>,
    /// Monotonic lifecycle boundary times for duration calculations.
    pub timestamps: AttemptTimestamps,
    /// Cleanup completion classification committed before publication.
    pub cleanup_disposition: AttemptCleanupDisposition,
    /// Exact when `cleanup_failure_count_complete` is true. The lifecycle
    /// never publishes a saturated arithmetic result as exact.
    pub cleanup_failure_count: u64,
    /// Required so decoding never fabricates an exact cleanup-failure count.
    pub cleanup_failure_count_complete: bool,
    /// First bounded cleanup failure. Further failures affect only the exact
    /// count; timeline retention is intentionally not authoritative.
    pub first_cleanup_failure: Option<CleanupFailureSummary>,
    /// Disposition of the immutable grant after terminal cleanup.
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
    /// a device-backed domain. No native device status applies to such an attempt.
    pub native_fault_evidence: Option<AttemptNativeFaultEvidence>,
    /// Bounded ledger, immutable-cap, domain-usage, and allocator evidence
    /// from the sole terminal release commit. `None` means the attempt did
    /// not own a device-backed admission domain.
    pub resource_evidence: Option<AttemptResourceEvidence>,
    /// Observation records lost because retention or recording failed.
    pub observation_lost_records: u64,
    /// Records retained in the sealed bounded observation.
    pub observation_retained_records: u64,
    /// Deterministic logical bytes retained in the sealed observation.
    pub observation_retained_bytes: u64,
    /// Whether retention limits prevented complete observation capture.
    pub observation_truncated: bool,
    /// Whether recording degraded for reasons other than configured retention limits.
    pub observation_degraded: bool,
    /// Optional wall-clock nanosecond anchor associated with the observation clock.
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
    /// Schema label for this bounded projection.
    pub schema: &'static str,
    /// Attempt identity for individual delivery; excluded from metric labels.
    pub attempt_id: AttemptId,
    /// Planning coverage copied from the report.
    pub disposition: AttemptDisposition,
    /// Terminal execution outcome copied from the report.
    pub outcome: RuntimeTerminalOutcome,
    /// Cleanup result copied from the report.
    pub cleanup_disposition: AttemptCleanupDisposition,
    /// Recorded cleanup failure count; consult its completeness flag.
    pub cleanup_failure_count: u64,
    /// Whether the cleanup failure count is exact.
    pub cleanup_failure_count_complete: bool,
    /// Grant ownership after cleanup.
    pub capacity_disposition: AttemptCapacityDisposition,
    /// Committed admitted-device health result, or `None` when unavailable.
    pub health_transition: Option<AttemptHealthTransition>,
    /// Recovery recommendation derived at report seal.
    pub retry_advice: AttemptRetryAdvice,
    /// Whether configured retention bounds truncated observation capture.
    pub observation_truncated: bool,
    /// Whether recording encountered degradation beyond retention limits.
    pub observation_degraded: bool,
    /// Whether the full report contains native-fault evidence.
    pub native_fault_evidence_present: bool,
    /// Whether the full report contains terminal resource evidence.
    pub resource_evidence_present: bool,
}

/// Stable fixed-cardinality labels derived from one sealed attempt projection.
///
/// Every consumer that renders attempt facts with labels (DataFusion metrics,
/// Flight query stats, and future bounded exporters) must use this projection
/// and must not duplicate enum-to-string mappings at its own boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedAttemptLabels {
    /// Stable planning coverage label.
    pub disposition: &'static str,
    /// Stable terminal outcome label.
    pub outcome: &'static str,
    /// Stable cleanup result label.
    pub cleanup_disposition: &'static str,
    /// `true` or `false` for cleanup count exactness.
    pub cleanup_failure_count_complete: &'static str,
    /// Stable post-cleanup grant ownership label.
    pub capacity_disposition: &'static str,
    /// Stable health label, including `none` for unavailable evidence.
    pub health_transition: &'static str,
    /// Stable advised recovery label.
    pub retry_recovery: &'static str,
    /// Stable recovery prerequisite label.
    pub retry_prerequisite: &'static str,
    /// `true` or `false` for retention truncation.
    pub observation_truncated: &'static str,
    /// `true` or `false` for recording degradation.
    pub observation_degraded: &'static str,
    /// `present` or `absent` for native-fault evidence.
    pub native_fault_evidence: &'static str,
    /// `present` or `absent` for resource evidence.
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
            schema: SEALED_ATTEMPT_PROJECTION_SCHEMA,
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
    /// Returns submission-to-report-ready duration in nanoseconds.
    /// Returns `None` when report readiness was not recorded.
    pub fn total_ns(&self) -> Option<u64> {
        self.timestamps.report_ready_ns.map(|ready| ready.saturating_sub(self.timestamps.submitted_ns))
    }
    /// Returns planning-start-to-planning-finish duration in nanoseconds.
    /// Returns `None` when either boundary is absent.
    pub fn planning_ns(&self) -> Option<u64> {
        self.timestamps
            .planning_started_ns
            .zip(self.timestamps.planning_finished_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    /// Returns admission-request-to-grant duration in nanoseconds.
    /// Returns `None` when either boundary is absent.
    pub fn queue_wait_ns(&self) -> Option<u64> {
        self.timestamps
            .admission_requested_ns
            .zip(self.timestamps.admitted_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    /// Returns planning-finish-to-admission-request duration in nanoseconds.
    /// Returns `None` when either boundary is absent.
    pub fn post_planning_pre_admission_ns(&self) -> Option<u64> {
        self.timestamps
            .planning_finished_ns
            .zip(self.timestamps.admission_requested_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    /// Returns grant-to-physical-execution-start duration in nanoseconds.
    /// Returns `None` when either boundary is absent.
    pub fn admitted_to_physical_execution_ns(&self) -> Option<u64> {
        self.timestamps
            .admitted_ns
            .zip(self.timestamps.physical_execution_started_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    /// Returns physical-execution-start-to-GPU-dispatch duration in nanoseconds.
    /// Returns `None` when either boundary is absent.
    pub fn physical_to_gpu_dispatch_ns(&self) -> Option<u64> {
        self.timestamps
            .physical_execution_started_ns
            .zip(self.timestamps.gpu_execution_started_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    /// Returns grant-to-first-GPU-dispatch duration in nanoseconds.
    /// Returns `None` when either boundary is absent.
    pub fn admitted_to_gpu_dispatch_ns(&self) -> Option<u64> {
        self.timestamps
            .admitted_ns
            .zip(self.timestamps.gpu_execution_started_ns)
            .map(|(start, end)| end.saturating_sub(start))
    }
    /// Returns submission-to-first-GPU-dispatch duration in nanoseconds.
    /// Returns `None` when GPU dispatch was not recorded.
    pub fn submitted_to_gpu_dispatch_ns(&self) -> Option<u64> {
        self.timestamps.gpu_execution_started_ns.map(|end| end.saturating_sub(self.timestamps.submitted_ns))
    }
    /// Returns first-GPU-dispatch-to-terminal-outcome duration in nanoseconds.
    /// Returns `None` when GPU dispatch was not recorded.
    pub fn gpu_dispatch_to_terminal_ns(&self) -> Option<u64> {
        self.timestamps.gpu_execution_started_ns.map(|start| self.timestamps.outcome_terminal_ns.saturating_sub(start))
    }
    /// Returns physical-execution-start-to-terminal-outcome duration in nanoseconds.
    /// Returns `None` when physical execution did not start.
    pub fn physical_execution_ns(&self) -> Option<u64> {
        self.timestamps
            .physical_execution_started_ns
            .map(|start| self.timestamps.outcome_terminal_ns.saturating_sub(start))
    }
    /// Returns terminal-outcome-to-cleanup-complete duration in nanoseconds.
    /// Returns `None` when cleanup completion was not recorded.
    pub fn cleanup_ns(&self) -> Option<u64> {
        self.timestamps.cleanup_complete_ns.map(|end| end.saturating_sub(self.timestamps.outcome_terminal_ns))
    }
}
/// Query terminal outcome stored outside the bounded record collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeTerminalOutcome {
    /// Execution reached successful completion.
    Succeeded,
    /// A terminal error classified from the terminal structured error itself.
    Failed {
        /// Stable machine identity of the terminal error.
        error_code: AttemptFailureCode,
        /// Source condition classification; recovery requires the sealed retry advice.
        error_status: AttemptFailureStatus,
    },
    /// Caller cancellation or deadline expiry ended execution.
    Cancelled,
    /// The attempt or result stream was dropped before another terminal outcome.
    Dropped,
    /// An unwind ended the attempt.
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

    /// Copies the stable code and status from a primary failure summary.
    pub const fn failed(summary: &AttemptFailureSummary) -> Self {
        Self::Failed { error_code: summary.code(), error_status: summary.status() }
    }

    /// Reduces a native error to its closed terminal projection. Attempt
    /// owners must still call the borrowed summary recorder first when they
    /// publish an attempt report.
    pub fn from_query_error(error: &QueryError) -> Self {
        if error.code() == QueryErrorCode::Cancelled {
            Self::Cancelled
        } else {
            Self::Failed {
                error_code: AttemptFailureCode::from_query_error_code(error.code()),
                error_status: AttemptFailureStatus::from_query_error_status(error.status()),
            }
        }
    }

    /// Returns the bounded primary-failure projection that must accompany a
    /// terminal native error. Ordinary caller cancellation intentionally
    /// remains a text-free control outcome, while an expired deadline retains
    /// its stable reason beside the same `Cancelled` terminal outcome.
    pub(crate) fn primary_failure_from_query_error(
        error: &QueryError,
        operation: AttemptFailureOperation,
    ) -> Option<AttemptFailureSummary> {
        match error.code() {
            QueryErrorCode::Cancelled if error.fact_value("reason") != Some("deadline_expired") => None,
            _ => Some(AttemptFailureSummary::from_native_error(error, operation)),
        }
    }

    pub(crate) const fn from_primary_failure(summary: &AttemptFailureSummary) -> Self {
        match summary.code() {
            AttemptFailureCode::Query(QueryErrorCode::Cancelled) => Self::Cancelled,
            _ => Self::failed(summary),
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
            Self::Cancelled => {
                Some(AttemptFailureSummary::control(AttemptFailureCode::Query(QueryErrorCode::Cancelled)))
            },
            Self::Dropped => Some(AttemptFailureSummary::control(AttemptFailureCode::Dropped)),
            Self::Panicked => Some(AttemptFailureSummary::control(AttemptFailureCode::Panicked)),
        }
    }
}
#[cfg(test)]
mod tests;
