//! One error taxonomy for the DataFusion adapter boundary.
//!
//! Adapter errors classify failures from DataFusion planning/execution,
//! native execution, cuDF calls made directly by the adapter, and native
//! engine errors wrapped at the adapter boundary. The stable
//! machine-readable [`ErrorCode`] is the contract; the human message is
//! diagnostic only. Native errors keep their original classification in
//! `native_*` facts while this layer maps them into adapter-level
//! [`ErrorCode`] values suitable for DataFusion.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
};

use datafusion::common::DataFusionError;

mod code;

use code::code_for_native_error as adapter_code_for_native_error;
pub use code::{ErrorCode, ErrorKind, ErrorStatus, kind_for_code, status_for_code};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

const NATIVE_FACT_COLLISION_KEYS: &str = "native_fact_collision_keys";
const FACT_COLLISION_KEYS: &str = "fact_collision_keys";
const EMPTY_FACT_KEY: &str = "<empty>";
const FALLBACK_OPERATION: &str = "datafusion_nexus.error.unspecified_operation";
const FALLBACK_MESSAGE: &str = "datafusion-nexus error without a message";

/// Adapter result type used before conversion to `DataFusionError`.
pub type Result<T> = std::result::Result<T, Error>;

const MAX_SOURCE_CHAIN_DEPTH: usize = 32;
const MAX_LOG_CAUSE_BYTES: usize = 512;
const MAX_LOG_SOURCE_PATH_BYTES: usize = 1_024;
const MAX_NEXUS_ERROR_TRAVERSAL_DEPTH: usize = 64;
const MAX_NEXUS_ERROR_TRAVERSAL_NODES: usize = 256;
const MAX_LOG_SAFE_FACT_COUNT: usize = 16;
const MAX_LOG_SAFE_FACT_VALUE_BYTES: usize = 256;
const MAX_LOG_SAFE_FACTS_RENDERED_BYTES: usize = 2_048;
pub(crate) const ERROR_FACTS_TRUNCATED_KEY: &str = "error_facts_truncated";

/// DataFusion error inspection helpers provided by datafusion-nexus.
pub trait DataFusionErrorExt {
    /// Finds the first Nexus [`Error`] in this DataFusion error tree.
    ///
    /// Traversal is deterministic depth-first pre-order. Generic
    /// [`std::error::Error::source`] links are followed, while every
    /// [`DataFusionError::Collection`] branch is visited in collection order.
    /// The first match in that order is returned. Traversal is bounded
    /// defensively by both depth and total node count.
    fn find_nexus_error(&self) -> Option<&Error>;
}

impl DataFusionErrorExt for DataFusionError {
    fn find_nexus_error(&self) -> Option<&Error> {
        let mut visited_nodes = 0;
        find_nexus_error_in(self, 0, &mut visited_nodes)
    }
}

fn find_nexus_error_in<'a>(
    error: &'a (dyn StdError + 'static),
    depth: usize,
    visited_nodes: &mut usize,
) -> Option<&'a Error> {
    if *visited_nodes >= MAX_NEXUS_ERROR_TRAVERSAL_NODES {
        return None;
    }
    *visited_nodes += 1;

    if let Some(error) = error.downcast_ref::<Error>() {
        return Some(error);
    }
    if depth >= MAX_NEXUS_ERROR_TRAVERSAL_DEPTH {
        return None;
    }

    if let Some(DataFusionError::Collection(errors)) = error.downcast_ref::<DataFusionError>() {
        return errors.iter().find_map(|error| find_nexus_error_in(error, depth + 1, visited_nodes));
    }

    error.source().and_then(|source| find_nexus_error_in(source, depth + 1, visited_nodes))
}

/// Machine-readable facts attached to an [`Error`].
///
/// Fact keys are static strings so they can be treated as a documented contract
/// by tests, reports, and callers. Values are stringified with [`ToString`].
/// Iteration is sorted by key because facts are stored in a [`BTreeMap`].
/// Raw lookup and iteration preserve every value exactly. [`fmt::Debug`]
/// exposes only facts explicitly approved through [`Error::with_log_safe_fact`],
/// using the same sanitized and bounded projection as server observability.
///
/// Empty keys are ignored and duplicate inserts preserve the first value and
/// disclosure choice. Rejected keys are reported deterministically under
/// `fact_collision_keys`. The adapter reserves `cudf_*` for cuDF
/// classification and `native_*` for wrapped `nexus-query-engine` errors;
/// colliding native facts are reported under `native_fact_collision_keys`.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, FactValue>,
    collision_keys: BTreeSet<&'static str>,
}

#[derive(Clone, PartialEq, Eq)]
struct FactValue {
    value: String,
    log_safe: bool,
}

/// Sanitized, bounded facts approved for logs and retained diagnostics.
pub struct LogSafeFactProjection {
    entries: Vec<(&'static str, String)>,
    truncated: bool,
}

impl Facts {
    /// Creates an empty fact set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the string value for `key` when present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|fact| fact.value.as_str())
    }

    /// Iterates facts in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.values.iter().map(|(key, fact)| (*key, fact.value.as_str()))
    }

    fn insert(&mut self, key: &'static str, value: String) {
        self.insert_with_log_safety(key, value, false);
    }

    pub fn insert_log_safe(&mut self, key: &'static str, value: String) {
        // The truncation marker is owned by the projection. Preserve a caller's
        // raw typed value, but never let it opt itself into disclosure or spoof
        // the synthetic retained marker.
        self.insert_with_log_safety(key, value, key != ERROR_FACTS_TRUNCATED_KEY);
    }

    fn insert_with_log_safety(&mut self, key: &'static str, value: String, log_safe: bool) {
        if key.is_empty() {
            self.record_collision(EMPTY_FACT_KEY);
            return;
        }
        if key == FACT_COLLISION_KEYS || self.values.contains_key(key) {
            self.record_collision(key);
            return;
        }
        self.values.insert(key, FactValue { value, log_safe });
    }

    fn record_collision(&mut self, key: &'static str) {
        self.collision_keys.insert(key);
        let value = self.collision_keys.iter().copied().collect::<Vec<_>>().join(",");
        self.values.insert(FACT_COLLISION_KEYS, FactValue { value, log_safe: true });
    }

    pub fn log_safe_projection(&self) -> LogSafeFactProjection {
        let safe_facts = self
            .values
            .iter()
            .filter(|(key, fact)| **key != ERROR_FACTS_TRUNCATED_KEY && fact.log_safe)
            .map(|(key, fact)| (*key, fact.value.as_str()))
            .collect::<Vec<_>>();
        let mut truncated = safe_facts.len() > MAX_LOG_SAFE_FACT_COUNT;
        let mut entries = Vec::with_capacity(safe_facts.len().min(MAX_LOG_SAFE_FACT_COUNT));
        let mut rendered_bytes = 0usize;

        for (key, raw_value) in safe_facts.into_iter().take(MAX_LOG_SAFE_FACT_COUNT) {
            let sanitized = sanitize_log_text(raw_value);
            let (mut value, value_truncated) = truncate_with_marker(&sanitized, MAX_LOG_SAFE_FACT_VALUE_BYTES);
            truncated |= value_truncated;

            let separator_bytes = usize::from(!entries.is_empty()) * 2;
            let fixed_bytes = separator_bytes.saturating_add(key.len()).saturating_add(1);
            let Some(value_budget) = MAX_LOG_SAFE_FACTS_RENDERED_BYTES
                .checked_sub(rendered_bytes)
                .and_then(|remaining| remaining.checked_sub(fixed_bytes))
            else {
                truncated = true;
                break;
            };
            let (total_capped_value, total_truncated) = truncate_with_marker(&value, value_budget);
            value = total_capped_value;
            truncated |= total_truncated;
            rendered_bytes = rendered_bytes.saturating_add(fixed_bytes).saturating_add(value.len());
            entries.push((key, value));
            if total_truncated {
                break;
            }
        }

        LogSafeFactProjection { entries, truncated }
    }
}

impl fmt::Debug for Facts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let projection = self.log_safe_projection();
        f.debug_struct("Facts")
            .field("values", &projection.entries)
            .field(ERROR_FACTS_TRUNCATED_KEY, &projection.truncated)
            .finish()
    }
}

impl LogSafeFactProjection {
    pub fn retained_entries(&self) -> Vec<(String, String)> {
        let mut entries =
            self.entries.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect::<Vec<_>>();
        if self.truncated {
            entries.push((ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string()));
            entries.sort_by(|left, right| left.0.cmp(&right.0));
        }
        entries
    }

    pub fn rendered(&self) -> Option<String> {
        (!self.entries.is_empty())
            .then(|| self.entries.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join(", "))
    }

    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

fn sanitize_log_text(value: &str) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.split_whitespace().any(is_credential_token) {
        return "<redacted-credential>".to_string();
    }
    normalized
        .split_whitespace()
        .map(|token| if token.contains("://") { "<redacted-uri>" } else { token })
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_credential_token(token: &str) -> bool {
    let token = token
        .trim_matches(|ch: char| matches!(ch, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '"' | '\''))
        .to_ascii_lowercase();
    let key = token.split_once(['=', ':']).map_or(token.as_str(), |(key, _)| key);
    let normalized_key = key.chars().filter(|ch| !matches!(ch, '-' | '_' | '.')).collect::<String>();
    matches!(
        normalized_key.as_str(),
        "authorization"
            | "proxyauthorization"
            | "cookie"
            | "setcookie"
            | "password"
            | "passwd"
            | "apikey"
            | "accesskey"
            | "secret"
            | "token"
            | "clientsecret"
            | "refreshtoken"
            | "awssecretaccesskey"
            | "secretaccesskey"
    )
}

fn truncate_with_marker(value: &str, max_bytes: usize) -> (String, bool) {
    if value.len() <= max_bytes {
        return (value.to_string(), false);
    }
    const MARKER: &str = "...";
    if max_bytes <= MARKER.len() {
        return (MARKER[..max_bytes].to_string(), true);
    }
    let mut end = max_bytes - MARKER.len();
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (format!("{}{MARKER}", &value[..end]), true)
}

/// Structured adapter error with stable classification and facts.
///
/// [`fmt::Display`] renders only the human-facing [`Self::message`]. Consumers
/// that need machine-readable diagnostics should use the typed accessors or
/// [`fmt::Debug`] rather than parsing display text.
pub struct Error {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    source: Option<BoxedSourceError>,
}

/// Human diagnostics derived from an error source chain.
///
/// Retained query diagnostics should keep machine-readable fields (`code`,
/// `kind`, `status`, `operation`, facts) separate from human root-cause text.
/// This formatter walks `std::error::Error::source()` and renders project
/// errors by their human message, not by their structured `Display`
/// representation. Server-side logs should use [`ErrorLogDiagnostics`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorSourceDiagnostics {
    chain: Option<String>,
    root_cause: Option<String>,
}

/// Bounded source diagnostics for privileged server-side logs.
///
/// This projection keeps only the deepest cause. Consumers that log the
/// boundary error's message separately may omit an identical cause. Full
/// source chains remain available through [`ErrorSourceDiagnostics`] for
/// retained diagnostics.
/// Local source paths are preserved for incident response; URI-shaped paths
/// and credential-bearing messages remain redacted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorLogDiagnostics {
    cause: Option<String>,
    source_path: Option<String>,
    truncated: bool,
}

impl ErrorLogDiagnostics {
    /// Builds log diagnostics from `error` and all of its sources.
    pub fn from_error(error: &(dyn StdError + 'static)) -> Self {
        let mut current = Some(error);
        let mut cause = None;
        let mut diagnostics = Self { cause: None, source_path: None, truncated: false };
        for _ in 0..MAX_SOURCE_CHAIN_DEPTH {
            let Some(error) = current else {
                break;
            };
            if diagnostics.source_path.is_none() {
                diagnostics.set_source_path(diagnostic_source_path(error));
            }
            let message = log_diagnostic_error_message(error);
            if !message.is_empty() {
                cause = Some(message);
            }
            current = error.source();
        }
        let depth_truncated = current.is_some();
        let (cause, message_truncated) = match cause {
            Some(cause) => {
                let (cause, truncated) = truncate_with_marker(&cause, MAX_LOG_CAUSE_BYTES);
                (Some(cause), truncated)
            },
            None => (None, false),
        };
        diagnostics.cause = cause;
        diagnostics.truncated |= depth_truncated || message_truncated;
        diagnostics
    }

    /// Deepest bounded and redacted cause message.
    pub fn cause(&self) -> Option<&str> {
        self.cause.as_deref()
    }

    /// Input path associated with the failure, when present.
    pub fn source_path(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// Whether traversal or any projected diagnostic exceeded its log budget.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    fn set_source_path(&mut self, path: Option<&str>) {
        let Some(path) = path.map(str::trim).filter(|path| !path.is_empty()) else {
            return;
        };
        let path = if path.contains("://") { "<redacted-uri>" } else { path };
        let (path, truncated) = truncate_with_marker(path, MAX_LOG_SOURCE_PATH_BYTES);
        self.source_path = Some(path);
        self.truncated |= truncated;
    }
}

impl ErrorSourceDiagnostics {
    /// Builds diagnostics from `error` and all of its sources.
    pub fn from_error(error: &(dyn StdError + 'static)) -> Self {
        Self::from_iter(Some(error))
    }

    /// Builds diagnostics from an optional source, excluding the wrapper that
    /// already has explicit machine fields at the current boundary.
    pub fn from_source(source: Option<&(dyn StdError + 'static)>) -> Self {
        Self::from_iter(source)
    }

    /// Full source chain text, with adjacent duplicate messages removed.
    pub fn chain(&self) -> Option<&str> {
        self.chain.as_deref()
    }

    /// Deepest source message.
    pub fn root_cause(&self) -> Option<&str> {
        self.root_cause.as_deref()
    }

    /// Deepest source message when it adds information beyond the full chain.
    pub fn root_cause_when_distinct(&self) -> Option<&str> {
        match (self.chain(), self.root_cause()) {
            (Some(chain), Some(root_cause)) if chain != root_cause => Some(root_cause),
            _ => None,
        }
    }

    fn from_iter(mut current: Option<&(dyn StdError + 'static)>) -> Self {
        let mut messages = Vec::new();
        for _ in 0..MAX_SOURCE_CHAIN_DEPTH {
            let Some(error) = current else {
                break;
            };
            let message = diagnostic_error_message(error);
            if !message.is_empty() && messages.last() != Some(&message) {
                messages.push(message);
            }
            current = error.source();
        }
        let root_cause = messages.last().cloned();
        let chain = (!messages.is_empty()).then(|| messages.join(": "));
        Self { chain, root_cause }
    }
}

impl Error {
    /// Creates an adapter error with a stable operation id and message.
    ///
    /// `operation` should be a stable dotted identifier such as
    /// `datafusion_nexus.native_exec.execute`. `message` is human-facing
    /// diagnostic text. Add machine-readable context with [`Self::with_fact`].
    pub fn new(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        let operation = if operation.is_empty() { FALLBACK_OPERATION } else { operation };
        let message = if message.is_empty() { FALLBACK_MESSAGE.to_string() } else { message };
        Self { code, operation, message, facts: Facts::new(), source: None }
    }

    /// Adds one machine-readable fact.
    ///
    /// `key` must be a non-empty static string and should be treated as part of
    /// the diagnostic contract once exposed. `value` is stringified with
    /// [`ToString`]. Empty keys are ignored. Duplicate keys preserve the first
    /// value and are recorded under `fact_collision_keys`. Facts added here
    /// remain raw and are hidden from logs, retained diagnostics, and
    /// [`Facts`]'s [`fmt::Debug`] output unless the caller instead makes the explicit
    /// [`Self::with_log_safe_fact`] disclosure choice.
    #[must_use]
    pub fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.facts.insert(key, value.to_string());
        self
    }

    /// Adds a fact that may be projected into logs and retained diagnostics.
    ///
    /// This is an explicit disclosure contract, not a convenience alias for
    /// [`Self::with_fact`]. Use it only for bounded operational classification
    /// values that contain no credentials, identifiers, paths, query literals,
    /// backend details, or other sensitive/user-provided data. The projection
    /// still redacts URI and credential-like tokens and enforces size caps as
    /// defense in depth, but sanitization does not make an unsafe value safe to
    /// opt in. Ordinary and externally supplied facts must use
    /// [`Self::with_fact`] and remain hidden by default. The reserved
    /// `error_facts_truncated` key remains available through raw typed lookup,
    /// but cannot be opted into the projection; only the projection itself may
    /// emit that retained marker.
    /// Empty and duplicate keys follow [`Self::with_fact`]'s first-write
    /// collision policy.
    #[must_use]
    pub fn with_log_safe_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.facts.insert_log_safe(key, value.to_string());
        self
    }

    pub fn with_preserved_facts(mut self, facts: Facts) -> Self {
        self.facts = facts;
        self
    }

    /// Attaches the source error when no source is already present.
    ///
    /// A second source is discarded so error decoration cannot replace the
    /// original causal chain.
    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        if self.source.is_none() {
            self.source = Some(Box::new(source));
        }
        self
    }

    /// Attaches an already boxed source error when no source is already present.
    ///
    /// A second source is discarded so error decoration cannot replace the
    /// original causal chain.
    #[must_use]
    pub fn with_boxed_source(mut self, source: Box<dyn StdError + Send + Sync + 'static>) -> Self {
        if self.source.is_none() {
            self.source = Some(source);
        }
        self
    }

    /// Wraps a `cudf::Error` at the adapter boundary.
    ///
    /// `operation` should be the stable dotted id for the adapter operation that
    /// called cuDF, and `message` should describe the failure for humans. cuDF
    /// classification and a URI-redacted message are preserved as `cudf_*`
    /// facts.
    pub fn from_cudf(operation: &'static str, message: impl Into<String>, source: cudf::Error) -> Self {
        let code = code_for_cudf_error(&source);
        let mut error = Self::new(code, operation, message)
            .with_log_safe_fact("cudf_code", source.code())
            .with_log_safe_fact("cudf_kind", source.kind())
            .with_log_safe_fact("cudf_status", source.status())
            .with_log_safe_fact("cudf_operation", source.operation());
        if let Some(detail) = source.ffi_detail() {
            error = error
                .with_log_safe_fact("failure_source", detail.failure_source)
                .with_log_safe_fact("requested_allocation_bytes", detail.requested_bytes)
                .with_log_safe_fact("allocation_alignment", detail.alignment)
                .with_log_safe_fact("cuda_status", detail.cuda_error);
        }
        match cudf_message_fact(&source) {
            Some(cudf_message) => error.with_fact("cudf_message", cudf_message),
            None => error,
        }
        .with_source(source)
    }

    /// Wraps a native-engine error at the DataFusion adapter boundary.
    ///
    /// The adapter maps the native code into the closest [`ErrorCode`] and
    /// preserves the native classification under reserved facts: `native_code`,
    /// `native_kind`, `native_status`, `native_operation`, and the native
    /// diagnostic text under `native_message`. Native facts are copied
    /// unprefixed when they do not collide with adapter facts. If a native fact
    /// would overwrite an adapter/native classification fact, the adapter
    /// preserves the existing fact, skips the colliding native fact, and records
    /// the skipped keys in `native_fact_collision_keys`.
    ///
    /// `operation` should be the stable dotted id for the adapter operation, and
    /// `message` should be human-facing adapter context.
    pub(crate) fn from_native(
        operation: &'static str,
        message: impl Into<String>,
        source: nexus_query_engine::error::QueryError,
    ) -> Self {
        let code = code_for_native_error(&source);
        let mut error = Self::new(code, operation, message)
            .with_log_safe_fact("native_code", source.code())
            .with_log_safe_fact("native_kind", source.kind())
            .with_log_safe_fact("native_status", source.status())
            .with_log_safe_fact("native_operation", source.operation())
            .with_fact("native_message", source.message());
        let mut collision_keys = Vec::new();
        for (key, value) in source.facts().iter() {
            if key == NATIVE_FACT_COLLISION_KEYS || error.facts.get(key).is_some() {
                collision_keys.push(key);
                continue;
            }
            if native_fact_is_log_safe(key) {
                error.facts.insert_log_safe(key, value.to_string());
            } else {
                error.facts.insert(key, value.to_string());
            }
        }
        if !collision_keys.is_empty() {
            error.facts.insert_log_safe(NATIVE_FACT_COLLISION_KEYS, collision_keys.join(","));
        }
        error.with_source(source)
    }

    pub(crate) fn from_native_with_blocking_diagnostics(
        operation: &'static str,
        message: impl Into<String>,
        source: nexus_query_engine::error::QueryError,
    ) -> Self {
        const CLIENT_FACTS: &[&str] = &[
            "stage",
            "phase",
            "schedule_reason",
            "sort_rejection_reason",
            "reason",
            "occurrence_id",
            "node_id",
            "input_node_id",
            "column",
            "left_rows",
            "right_rows",
            "output_rows",
            "input_row_multiplicity",
            "producer_row_count_upper_bound",
            "consumer_row_count_upper_bound",
            "max_compatible_managed_capacity_bytes",
            "minimum_compatible_required_floor_bytes",
            "minimum_compatible_dominant_owner",
            "minimum_compatible_dominant_owner_role",
            "minimum_compatible_dominant_owner_bytes",
            "minimum_compatible_dominant_owner_occurrence",
        ];
        let mut message = message.into();
        message.push_str(": ");
        message.push_str(source.message());
        let facts = CLIENT_FACTS
            .iter()
            .filter_map(|key| source.fact_value(key).map(|value| format!("{key}={value}")))
            .collect::<Vec<_>>();
        if !facts.is_empty() {
            message.push_str(" [");
            message.push_str(&facts.join(", "));
            message.push(']');
        }
        Self::from_native(operation, message, source)
    }

    #[cfg(feature = "server-internal")]
    #[doc(hidden)]
    pub fn from_native_for_internal_consumer(
        operation: &'static str,
        message: impl Into<String>,
        source: nexus_query_engine::error::QueryError,
    ) -> Self {
        Self::from_native(operation, message, source)
    }

    /// Returns the stable machine-readable adapter error code.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Returns the coarse category derived from [`Self::code`].
    pub fn kind(&self) -> ErrorKind {
        self.code.kind()
    }

    /// Returns the retry/remediation status derived from [`Self::code`].
    pub fn status(&self) -> ErrorStatus {
        self.code.status()
    }

    /// Returns the stable dotted operation id that produced the error.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the human-facing diagnostic message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the sorted machine-readable fact set.
    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    /// Returns one fact value by key.
    pub fn fact_value(&self, key: &str) -> Option<&str> {
        self.facts.get(key)
    }

    /// Returns the underlying source error when one was attached.
    pub fn source_error(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }
}

fn native_fact_is_log_safe(key: &str) -> bool {
    matches!(
        key,
        "max_compatible_managed_capacity_bytes"
            | "minimum_compatible_required_floor_bytes"
            | "minimum_compatible_configured_floor_bytes"
            | "minimum_compatible_required_phase_floor_bytes"
            | "minimum_compatible_capability_floor_raise_bytes"
            | "minimum_compatible_owner_term_count"
            | "minimum_compatible_owner_term_bytes"
            | "minimum_compatible_dominant_owner_bytes"
            | "minimum_compatible_dominant_owner"
            | "minimum_compatible_dominant_owner_role"
    )
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let source = self.source.as_ref().map(|_| "<source omitted>");
        f.debug_struct("Error")
            .field("code", &self.code)
            .field("kind", &self.kind())
            .field("status", &self.status())
            .field("operation", &self.operation)
            .field("message", &self.message)
            .field("facts", &self.facts)
            .field("source", &source)
            .finish()
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

// Adapter errors cross the DataFusion boundary as `External` so callers can
// downcast the boxed source back to `datafusion_nexus::Error` and inspect code,
// kind, status, operation, facts, and source.
impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        Self::External(Box::new(error))
    }
}

/// Returns the trimmed foreign cuDF message, redacting URI-like tokens.
fn cudf_message_fact(source: &cudf::Error) -> Option<String> {
    let message = source.message().trim();
    if message.is_empty() {
        return None;
    }
    Some(redact_uri_tokens(message))
}

/// Redacts whitespace-delimited `scheme://...` tokens and normalizes spaces.
fn redact_uri_tokens(message: &str) -> String {
    message
        .split_whitespace()
        .map(|token| if token.contains("://") { "<redacted-uri>" } else { token })
        .collect::<Vec<_>>()
        .join(" ")
}

// cuDF classification preserves permanent size/host-limit codes first, then
// generic resource exhaustion, retry status (`Temporary`/`Persistent`), and
// permanent failures by cuDF kind. This keeps
// cuDF's coarse status contract intact. Attempt retry advice is derived only
// from a sealed report; Persistent still records a condition needing
// escalation or intervention. Notable permanent mappings:
// `NotFound` is treated as invalid input, `ExecutionContext` is treated as an
// invariant violation, and unknown kinds fall back to dependency failure.
fn code_for_cudf_error(error: &cudf::Error) -> ErrorCode {
    if error.budget_exhaustion().is_some() {
        return ErrorCode::QueryMemoryLimitExceeded;
    }
    if let Some(detail) = error.ffi_detail() {
        match detail.failure_source {
            cudf::CudfErrorFailureSource::ReservationLimit => return ErrorCode::QueryMemoryLimitExceeded,
            cudf::CudfErrorFailureSource::Physical => return ErrorCode::DeviceOutOfMemory,
            cudf::CudfErrorFailureSource::Unknown => {},
            _ => {},
        }
    }
    match cudf::classify_cudf_error(error) {
        cudf::CudfErrorClassification::SizeOverflow => ErrorCode::SizeOverflow,
        cudf::CudfErrorClassification::HostMaterializationLimitExceeded => ErrorCode::HostMaterializationLimitExceeded,
        cudf::CudfErrorClassification::ResourceExhausted => ErrorCode::DeviceAllocationFailed,
        cudf::CudfErrorClassification::TemporaryFailure => ErrorCode::CudfTemporaryFailure,
        cudf::CudfErrorClassification::PersistentFailure => ErrorCode::DeviceFault,
        cudf::CudfErrorClassification::InvalidInput => ErrorCode::CudfInvalidInput,
        cudf::CudfErrorClassification::Unsupported => ErrorCode::CudfUnsupported,
        cudf::CudfErrorClassification::DependencyFailure => ErrorCode::CudfDependencyFailure,
        cudf::CudfErrorClassification::InvariantViolation => ErrorCode::InternalInvariantViolation,
    }
}

// Native-to-adapter mapping is intentionally lossy at the top-level code:
// native `UnsupportedPlanShape` becomes adapter `UnsupportedOperator` because
// the adapter has no shape-specific code. The exact native code remains
// available in the `native_code` fact attached by `Error::from_native`.
fn code_for_native_error(error: &nexus_query_engine::error::QueryError) -> ErrorCode {
    adapter_code_for_native_error(error.code())
}

fn diagnostic_error_message(error: &(dyn StdError + 'static)) -> String {
    if let Some(error) = error.downcast_ref::<Error>() {
        return normalize_diagnostic_message(error.message());
    }
    if let Some(error) = error.downcast_ref::<nexus_query_engine::error::QueryError>() {
        return normalize_diagnostic_message(error.message());
    }
    if let Some(error) = error.downcast_ref::<DataFusionError>() {
        return datafusion_error_human_message(error);
    }
    normalize_diagnostic_message(&error.to_string())
}

fn log_diagnostic_error_message(error: &(dyn StdError + 'static)) -> String {
    let message = if let Some(error) = error.downcast_ref::<cudf::Error>() {
        normalize_diagnostic_message(error.message())
    } else {
        diagnostic_error_message(error)
    };
    sanitize_log_text(&message)
}

fn diagnostic_source_path<'a>(error: &'a (dyn StdError + 'static)) -> Option<&'a str> {
    if let Some(error) = error.downcast_ref::<Error>() {
        return error.fact_value("path");
    }
    error.downcast_ref::<nexus_query_engine::error::QueryError>().and_then(|error| error.fact_value("path"))
}

/// Returns human-facing text for the current DataFusion error layer.
///
/// DataFusion renders SQL parser errors with `Debug` in
/// [`DataFusionError::message`]. This extractor uses the parser's `Display`
/// representation so Rust enum syntax does not leak into diagnostics.
pub fn datafusion_error_human_message(error: &DataFusionError) -> String {
    match error {
        DataFusionError::Context(message, _) => normalize_diagnostic_message(message),
        DataFusionError::Diagnostic(diagnostic, _) => normalize_diagnostic_message(&diagnostic.message),
        DataFusionError::External(source) => diagnostic_error_message(source.as_ref()),
        DataFusionError::SQL(parser, _) => normalize_diagnostic_message(&parser.to_string()),
        DataFusionError::Collection(errors) => match errors.first() {
            Some(first) => {
                let first = datafusion_error_human_message(first);
                normalize_diagnostic_message(&format!("DataFusion error collection ({} errors): {first}", errors.len()))
            },
            None => "DataFusion error collection (0 errors)".to_string(),
        },
        DataFusionError::Shared(source) => datafusion_error_human_message(source.as_ref()),
        _ => normalize_diagnostic_message(&error.message()),
    }
}

fn normalize_diagnostic_message(message: &str) -> String {
    message.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
#[path = "error_contract_tests.rs"]
mod error_contract_tests;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::error::ArrowError;
    use datafusion::common::Diagnostic;

    use super::*;

    #[derive(Debug)]
    struct TestSourceError {
        message: &'static str,
        source: Option<Box<dyn StdError + Send + Sync>>,
    }

    impl TestSourceError {
        fn leaf(message: &'static str) -> Self {
            Self { message, source: None }
        }

        fn with_source(message: &'static str, source: impl StdError + Send + Sync + 'static) -> Self {
            Self { message, source: Some(Box::new(source)) }
        }
    }

    impl fmt::Display for TestSourceError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.message)
        }
    }

    impl StdError for TestSourceError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
        }
    }

    fn nexus_error(operation: &'static str) -> Error {
        Error::new(ErrorCode::Internal, operation, "test Nexus error")
    }

    #[test]
    fn construction_and_fact_collisions_do_not_unwind_and_preserve_first_values() {
        let error = std::panic::catch_unwind(|| {
            Error::new(ErrorCode::Internal, "", "")
                .with_fact("", "ignored-empty-key")
                .with_fact("unsafe_first", "first-unsafe")
                .with_log_safe_fact("unsafe_first", "second-safe")
                .with_log_safe_fact("safe_first", "first-safe")
                .with_fact("safe_first", "second-unsafe")
        })
        .expect("constructing an error must not unwind");

        assert_eq!(error.operation(), FALLBACK_OPERATION);
        assert_eq!(error.message(), FALLBACK_MESSAGE);
        assert_eq!(error.fact_value(""), None);
        assert_eq!(error.fact_value("unsafe_first"), Some("first-unsafe"));
        assert_eq!(error.fact_value("safe_first"), Some("first-safe"));
        assert_eq!(error.fact_value(FACT_COLLISION_KEYS), Some("<empty>,safe_first,unsafe_first"));

        let projection = error.facts().log_safe_projection();
        assert_eq!(
            projection.entries,
            vec![
                (FACT_COLLISION_KEYS, "<empty>,safe_first,unsafe_first".to_string()),
                ("safe_first", "first-safe".to_string()),
            ]
        );
    }

    #[test]
    fn repeated_source_attachment_does_not_unwind_or_replace_the_first_source() {
        let first_typed = std::panic::catch_unwind(|| {
            Error::new(ErrorCode::Internal, "datafusion_nexus.test.source", "source test")
                .with_source(TestSourceError::leaf("first typed source"))
                .with_boxed_source(Box::new(TestSourceError::leaf("second boxed source")))
        })
        .expect("attaching a second boxed source must not unwind");
        assert_eq!(first_typed.source_error().map(ToString::to_string).as_deref(), Some("first typed source"));

        let first_boxed = std::panic::catch_unwind(|| {
            Error::new(ErrorCode::Internal, "datafusion_nexus.test.source", "source test")
                .with_boxed_source(Box::new(TestSourceError::leaf("first boxed source")))
                .with_source(TestSourceError::leaf("second typed source"))
        })
        .expect("attaching a second typed source must not unwind");
        assert_eq!(first_boxed.source_error().map(ToString::to_string).as_deref(), Some("first boxed source"));
    }

    #[test]
    fn log_safe_fact_projection_sanitizes_and_sorts_without_changing_raw_facts() {
        let error = Error::new(ErrorCode::Internal, "datafusion_nexus.test.facts", "fact projection test")
            .with_log_safe_fact("z_newline", "first\nsecond\tthird")
            .with_log_safe_fact("a_uri", "read s3://user:secret@bucket/path?token=value now")
            .with_log_safe_fact("b_credential", "Authorization=Bearer-secret")
            .with_log_safe_fact("c_client_secret", "CLIENT-SECRET=client-secret-value")
            .with_log_safe_fact("d_refresh_token", "Refresh.Token:refresh-token-value")
            .with_log_safe_fact("e_aws_secret", "AWS_SECRET_ACCESS_KEY=aws-secret-value")
            .with_fact("literal", "customer-secret")
            .with_fact("path", "/private/customer.parquet")
            .with_fact("config_value", "super-secret-config")
            .with_fact("native_message", "native secret")
            .with_fact("cudf_message", "cudf secret");

        assert_eq!(error.fact_value("literal"), Some("customer-secret"));
        assert_eq!(error.fact_value("path"), Some("/private/customer.parquet"));
        let projection = error.facts().log_safe_projection();
        assert_eq!(
            projection.entries,
            vec![
                ("a_uri", "read <redacted-uri> now".to_string()),
                ("b_credential", "<redacted-credential>".to_string()),
                ("c_client_secret", "<redacted-credential>".to_string()),
                ("d_refresh_token", "<redacted-credential>".to_string()),
                ("e_aws_secret", "<redacted-credential>".to_string()),
                ("z_newline", "first second third".to_string()),
            ]
        );
        assert!(!projection.truncated());
        let rendered = projection.rendered().expect("safe facts must render");
        assert_eq!(
            rendered,
            "a_uri=read <redacted-uri> now, b_credential=<redacted-credential>, c_client_secret=<redacted-credential>, d_refresh_token=<redacted-credential>, e_aws_secret=<redacted-credential>, z_newline=first second third"
        );
        for hidden in
            ["customer-secret", "/private/customer.parquet", "super-secret-config", "native secret", "cudf secret"]
        {
            assert!(!rendered.contains(hidden), "safe projection leaked {hidden}");
        }
    }

    #[test]
    fn log_safe_fact_projection_caps_values_count_and_total_size() {
        let long_value = "v".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES + 20);
        let value_capped = Error::new(ErrorCode::Internal, "datafusion_nexus.test.value_cap", "value cap")
            .with_log_safe_fact("long", &long_value);
        let projection = value_capped.facts().log_safe_projection();
        assert!(projection.truncated());
        assert_eq!(projection.entries.len(), 1);
        assert_eq!(projection.entries[0].1.len(), MAX_LOG_SAFE_FACT_VALUE_BYTES);
        assert!(projection.entries[0].1.ends_with("..."));

        let keys = [
            "fact_00", "fact_01", "fact_02", "fact_03", "fact_04", "fact_05", "fact_06", "fact_07", "fact_08",
            "fact_09", "fact_10", "fact_11", "fact_12", "fact_13", "fact_14", "fact_15", "fact_16",
        ];
        let mut count_capped = Error::new(ErrorCode::Internal, "datafusion_nexus.test.count_cap", "count cap");
        for key in keys {
            count_capped = count_capped.with_log_safe_fact(key, "safe");
        }
        let projection = count_capped.facts().log_safe_projection();
        assert_eq!(projection.entries.len(), MAX_LOG_SAFE_FACT_COUNT);
        assert!(projection.truncated());
        assert_eq!(projection.retained_entries()[0], (ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string()));

        let mut total_capped = Error::new(ErrorCode::Internal, "datafusion_nexus.test.total_cap", "total cap");
        for key in keys.into_iter().take(10) {
            total_capped = total_capped.with_log_safe_fact(key, "x".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES));
        }
        let projection = total_capped.facts().log_safe_projection();
        let rendered = projection.rendered().expect("partially projected facts must render");
        assert!(projection.truncated());
        assert!(projection.entries.len() < 10);
        assert!(rendered.len() <= MAX_LOG_SAFE_FACTS_RENDERED_BYTES);
    }

    #[test]
    fn reserved_truncation_marker_cannot_spoof_retained_projection() {
        let not_truncated = Error::new(ErrorCode::Internal, "datafusion_nexus.test.marker", "marker")
            .with_log_safe_fact(ERROR_FACTS_TRUNCATED_KEY, "true")
            .with_log_safe_fact("safe", "classification");
        assert_eq!(not_truncated.fact_value(ERROR_FACTS_TRUNCATED_KEY), Some("true"));
        let projection = not_truncated.facts().log_safe_projection();
        assert_eq!(projection.entries, vec![("safe", "classification".to_string())]);
        assert!(!projection.truncated());
        assert!(projection.retained_entries().iter().all(|(key, _)| key != ERROR_FACTS_TRUNCATED_KEY));

        let actually_truncated = Error::new(ErrorCode::Internal, "datafusion_nexus.test.marker", "marker")
            .with_log_safe_fact(ERROR_FACTS_TRUNCATED_KEY, "false")
            .with_log_safe_fact("long", "x".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES + 1));
        assert_eq!(actually_truncated.fact_value(ERROR_FACTS_TRUNCATED_KEY), Some("false"));
        let projection = actually_truncated.facts().log_safe_projection();
        assert!(projection.truncated());
        assert!(projection.entries.iter().all(|(key, _)| *key != ERROR_FACTS_TRUNCATED_KEY));
        let retained = projection.retained_entries();
        assert_eq!(
            retained.iter().filter(|(key, _)| key == ERROR_FACTS_TRUNCATED_KEY).collect::<Vec<_>>(),
            vec![&(ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string())]
        );
    }

    #[test]
    fn facts_debug_uses_only_the_sanitized_capped_projection() {
        let error = Error::new(ErrorCode::Internal, "datafusion_nexus.test.debug", "debug projection")
            .with_fact("literal", "never-log-this-literal")
            .with_log_safe_fact("safe", "line one\nhttps://user:password@example.test/private line two")
            .with_log_safe_fact("long", "z".repeat(MAX_LOG_SAFE_FACT_VALUE_BYTES + 1));

        let debug = format!("{:?}", error.facts());
        assert!(!debug.contains("never-log-this-literal"), "{debug}");
        assert!(!debug.contains("user:password"), "{debug}");
        assert!(!debug.contains('\n'), "{debug}");
        assert!(debug.contains("<redacted-uri>"), "{debug}");
        assert!(debug.contains("error_facts_truncated: true"), "{debug}");
    }

    #[test]
    fn foreign_wrappers_only_expose_audited_classification_facts() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::InvalidPlan,
            "nexus_query_engine.test.safe_projection",
            "native-message-must-stay-hidden",
        )
        .with_fact("literal", "native-user-literal")
        .with_fact("path", "/private/native.parquet");
        let native_adapter =
            Error::from_native("datafusion_nexus.test.safe_projection", "wrapped native error", native);
        let projection = native_adapter.facts().log_safe_projection();
        assert_eq!(
            projection.entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec!["native_code", "native_kind", "native_operation", "native_status"]
        );
        let rendered = projection.rendered().unwrap();
        for hidden in ["native-message-must-stay-hidden", "native-user-literal", "/private/native.parquet"] {
            assert!(!rendered.contains(hidden), "{rendered}");
        }

        let cudf = cudf::io::parquet::RemoteParquetSource::s3("gs://user:password@bucket/private.parquet")
            .expect_err("non-s3 scheme must fail");
        let cudf_adapter = Error::from_cudf("datafusion_nexus.test.safe_projection", "wrapped cudf error", cudf);
        let projection = cudf_adapter.facts().log_safe_projection();
        assert_eq!(
            projection.entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec!["cudf_code", "cudf_kind", "cudf_operation", "cudf_status"]
        );
        assert!(!projection.rendered().unwrap().contains("password"));
    }

    #[test]
    fn impossible_admission_facts_are_bounded_for_api_observability() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.admission",
            "no compatible device can satisfy the attempt required floor",
        )
        .with_fact("max_compatible_managed_capacity_bytes", 101_492_260_864_u64)
        .with_fact("minimum_compatible_required_floor_bytes", 4_918_766_690_098_841_543_u64)
        .with_fact("minimum_compatible_required_phase_floor_bytes", 4_918_766_690_098_841_543_u64)
        .with_fact("minimum_compatible_dominant_owner_bytes", 1_034_659_300_617_477_892_u64)
        .with_fact("minimum_compatible_dominant_owner", "consumer")
        .with_fact("minimum_compatible_dominant_owner_role", "consumer_workspace")
        .with_fact("minimum_compatible_dominant_owner_occurrence", "composed:private-table")
        .with_fact("path", "/private/customer/edges.parquet");
        let adapter = Error::from_native(
            "datafusion_nexus.native_exec.context",
            "failed to acquire backend query-service nexus native execution context",
            native,
        );
        let projection = adapter.facts().log_safe_projection();
        let rendered = projection.rendered().expect("admission diagnostics must reach API logs");

        for expected in [
            "max_compatible_managed_capacity_bytes=101492260864",
            "minimum_compatible_required_floor_bytes=4918766690098841543",
            "minimum_compatible_required_phase_floor_bytes=4918766690098841543",
            "minimum_compatible_dominant_owner_bytes=1034659300617477892",
            "minimum_compatible_dominant_owner=consumer",
            "minimum_compatible_dominant_owner_role=consumer_workspace",
        ] {
            assert!(rendered.contains(expected), "{rendered}");
        }
        assert!(!rendered.contains("private-table"), "{rendered}");
        assert!(!rendered.contains("/private/customer"), "{rendered}");
    }

    #[test]
    fn blocking_native_diagnostics_reach_display_without_unrelated_facts() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.composed_consumer",
            "a composed graph attempt lacks a complete ordered memory schedule",
        )
        .with_fact("schedule_reason", "consumer_local_source_rows_unavailable")
        .with_fact("stage", "prepared_composed_admission")
        .with_fact("node_id", 17)
        .with_fact("path", "/private/customer.parquet")
        .with_fact("literal", "secret");
        let adapter = Error::from_native_with_blocking_diagnostics(
            "datafusion_nexus.native_composed.admission_proof",
            "failed to derive the complete engine-owned composed admission bound",
            native,
        );
        let display = adapter.to_string();

        assert!(display.contains("a composed graph attempt lacks a complete ordered memory schedule"));
        assert!(display.contains("schedule_reason=consumer_local_source_rows_unavailable"));
        assert!(display.contains("stage=prepared_composed_admission"));
        assert!(display.contains("node_id=17"));
        assert!(!display.contains("/private/customer.parquet"));
        assert!(!display.contains("secret"));
        assert_eq!(adapter.fact_value("path"), Some("/private/customer.parquet"));
        assert!(DataFusionError::from(adapter).to_string().contains("stage=prepared_composed_admission"));

        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.admission",
            "the complete requirement exceeds every compatible device capacity",
        )
        .with_fact("reason", "impossible_requirement")
        .with_fact("stage", "ticket_admission")
        .with_fact("minimum_compatible_required_floor_bytes", 200)
        .with_fact("max_compatible_managed_capacity_bytes", 100)
        .with_fact("minimum_compatible_dominant_owner", "consumer")
        .with_fact("minimum_compatible_dominant_owner_role", "consumer_output")
        .with_fact("minimum_compatible_dominant_owner_bytes", 150);
        let adapter = Error::from_native_with_blocking_diagnostics(
            "datafusion_nexus.native_exec.context",
            "failed to acquire backend query-service nexus native execution context",
            native,
        );
        assert_eq!(adapter.fact_value("native_code"), Some("service_overloaded"));
        let display = DataFusionError::from(adapter).to_string();
        for fact in [
            "reason=impossible_requirement",
            "minimum_compatible_required_floor_bytes=200",
            "max_compatible_managed_capacity_bytes=100",
            "minimum_compatible_dominant_owner=consumer",
            "minimum_compatible_dominant_owner_role=consumer_output",
            "minimum_compatible_dominant_owner_bytes=150",
        ] {
            assert!(display.contains(fact), "{display}");
        }
    }

    #[test]
    fn error_debug_omits_native_and_cudf_source_payloads_but_keeps_typed_access() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::InvalidPlan,
            "nexus_query_engine.test.debug_source",
            "native-source-message-secret",
        )
        .with_fact("literal", "native-source-fact-secret");
        let native_adapter = Error::from_native("datafusion_nexus.test.debug_source", "native wrapper", native);
        let native_source = native_adapter
            .source_error()
            .and_then(|source| source.downcast_ref::<nexus_query_engine::error::QueryError>())
            .expect("native source must remain typed");
        assert_eq!(native_source.fact_value("literal"), Some("native-source-fact-secret"));
        let native_debug = format!("{native_adapter:?}");
        assert!(native_debug.contains("<source omitted>"), "{native_debug}");
        assert!(!native_debug.contains("native-source-message-secret"), "{native_debug}");
        assert!(!native_debug.contains("native-source-fact-secret"), "{native_debug}");

        let cudf = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::ForeignInvalidArgument,
            "cudf.test.debug_source",
            "cudf-source-detail-secret client_secret=cudf-secret-value",
        );
        let cudf_adapter = Error::from_cudf("datafusion_nexus.test.debug_source", "cudf wrapper", cudf);
        let cudf_source = cudf_adapter
            .source_error()
            .and_then(|source| source.downcast_ref::<cudf::Error>())
            .expect("cuDF source must remain typed");
        assert_eq!(cudf_source.message(), "cudf-source-detail-secret client_secret=cudf-secret-value");
        let cudf_debug = format!("{cudf_adapter:?}");
        assert!(cudf_debug.contains("<source omitted>"), "{cudf_debug}");
        assert!(!cudf_debug.contains("cudf-source-detail-secret"), "{cudf_debug}");
        assert!(!cudf_debug.contains("cudf-secret-value"), "{cudf_debug}");
    }

    #[test]
    fn find_nexus_error_finds_direct_external_error() {
        let error = DataFusionError::External(Box::new(nexus_error("datafusion_nexus.test.direct")));

        assert_eq!(error.find_nexus_error().map(Error::operation), Some("datafusion_nexus.test.direct"));
    }

    #[test]
    fn native_error_status_projection_agrees_with_the_engine_for_every_code() {
        for native_code in nexus_query_engine::error::QueryErrorCode::ALL {
            let native = nexus_query_engine::error::QueryError::new(
                *native_code,
                "nexus_query_engine.test.status_projection",
                "native status projection test",
            );
            let adapter =
                Error::from_native("datafusion_nexus.test.status_projection", "adapter status projection test", native);

            assert_eq!(
                adapter.status().as_str(),
                native_code.status().as_str(),
                "{native_code} must retain its native retry/remediation status"
            );
        }
    }

    #[test]
    fn find_nexus_error_traverses_context() {
        let error = DataFusionError::Context(
            "context".to_string(),
            Box::new(DataFusionError::External(Box::new(nexus_error("datafusion_nexus.test.context")))),
        );

        assert_eq!(error.find_nexus_error().map(Error::operation), Some("datafusion_nexus.test.context"));
    }

    #[test]
    fn find_nexus_error_traverses_diagnostic() {
        let error = DataFusionError::Diagnostic(
            Box::new(Diagnostic::new_error("diagnostic", None)),
            Box::new(DataFusionError::External(Box::new(nexus_error("datafusion_nexus.test.diagnostic")))),
        );

        assert_eq!(error.find_nexus_error().map(Error::operation), Some("datafusion_nexus.test.diagnostic"));
    }

    #[test]
    fn find_nexus_error_traverses_shared() {
        let error = DataFusionError::Shared(Arc::new(DataFusionError::External(Box::new(nexus_error(
            "datafusion_nexus.test.shared",
        )))));

        assert_eq!(error.find_nexus_error().map(Error::operation), Some("datafusion_nexus.test.shared"));
    }

    #[test]
    fn find_nexus_error_traverses_nested_collection_in_deterministic_order() {
        let error = DataFusionError::Collection(vec![
            DataFusionError::Plan("no Nexus error".to_string()),
            DataFusionError::Collection(vec![
                DataFusionError::Execution("still no Nexus error".to_string()),
                DataFusionError::External(Box::new(nexus_error("datafusion_nexus.test.first"))),
            ]),
            DataFusionError::External(Box::new(nexus_error("datafusion_nexus.test.second"))),
        ]);

        assert_eq!(error.find_nexus_error().map(Error::operation), Some("datafusion_nexus.test.first"));
    }

    #[test]
    fn find_nexus_error_traverses_collection_inside_arrow_external_wrapper() {
        let collection = DataFusionError::Collection(vec![
            DataFusionError::Plan("no Nexus error".to_string()),
            DataFusionError::External(Box::new(nexus_error("datafusion_nexus.test.arrow_collection"))),
        ]);
        let arrow = ArrowError::from_external_error(Box::new(collection));
        let error = DataFusionError::ArrowError(Box::new(arrow), None);

        assert_eq!(error.find_nexus_error().map(Error::operation), Some("datafusion_nexus.test.arrow_collection"));
    }

    #[test]
    fn find_nexus_error_returns_none_without_match() {
        let error = DataFusionError::Collection(vec![
            DataFusionError::Plan("planning failed".to_string()),
            DataFusionError::Context(
                "context".to_string(),
                Box::new(DataFusionError::Execution("execution failed".to_string())),
            ),
        ]);

        assert!(error.find_nexus_error().is_none());
    }

    #[test]
    fn source_diagnostics_walk_sources_and_deduplicate_adjacent_messages() {
        let error = TestSourceError::with_source(
            "dependency wrapper",
            TestSourceError::with_source("dependency wrapper", TestSourceError::leaf("socket refused")),
        );

        let diagnostics = ErrorSourceDiagnostics::from_error(&error);

        assert_eq!(diagnostics.chain(), Some("dependency wrapper: socket refused"));
        assert_eq!(diagnostics.root_cause(), Some("socket refused"));
        assert_eq!(diagnostics.root_cause_when_distinct(), Some("socket refused"));
    }

    #[test]
    fn source_diagnostics_render_project_errors_by_message_not_structured_display() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.test",
            "native root cause",
        )
        .with_fact("native_fact", "value");
        let adapter = Error::from_native("datafusion_nexus.test", "adapter wrapper", native);
        let error = DataFusionError::External(Box::new(adapter));

        let diagnostics = ErrorSourceDiagnostics::from_error(&error);
        let chain = diagnostics.chain().expect("source chain should be present");

        assert_eq!(chain, "adapter wrapper: native root cause");
        assert_eq!(diagnostics.root_cause(), Some("native root cause"));
        assert!(!chain.contains("code="), "{chain}");
        assert!(!chain.contains("facts="), "{chain}");
    }

    #[test]
    fn log_diagnostics_keep_one_bounded_redacted_cudf_cause() {
        let cudf = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::ForeignInvalidArgument,
            "cudf.test.log_diagnostics",
            "CUDF failure at: /home/customer/cudf/reader.cu:518: Parquet header parsing failed with code(s) 0x400",
        );
        let native = nexus_query_engine::error::QueryError::from_cudf(
            "nexus_query_engine.test.log_diagnostics",
            "failed to read bounded native local parquet source chunk with cudf",
            cudf,
        )
        .with_fact("path", "/srv/nexus/dataset/part-00000.parquet");
        let adapter = Error::from_native(
            "datafusion_nexus.test.log_diagnostics",
            "native cuGraph edge-source execution failed",
            native,
        );

        let diagnostics = ErrorLogDiagnostics::from_error(&adapter);

        assert_eq!(
            diagnostics.cause(),
            Some(
                "CUDF failure at: /home/customer/cudf/reader.cu:518: Parquet header parsing failed with code(s) 0x400"
            )
        );
        assert_eq!(diagnostics.source_path(), Some("/srv/nexus/dataset/part-00000.parquet"));
        assert!(!diagnostics.truncated());
    }

    #[test]
    fn log_diagnostics_redact_credentials_and_bound_foreign_messages() {
        let credentials = TestSourceError::leaf("request failed Authorization=Bearer-secret");
        assert_eq!(ErrorLogDiagnostics::from_error(&credentials).cause(), Some("<redacted-credential>"));

        let uri = Error::new(ErrorCode::Internal, "datafusion_nexus.test.log_uri", "read failed")
            .with_fact("path", "s3://user:secret@bucket/private.parquet");
        assert_eq!(ErrorLogDiagnostics::from_error(&uri).source_path(), Some("<redacted-uri>"));

        let long =
            TestSourceError { message: Box::leak("x".repeat(MAX_LOG_CAUSE_BYTES + 1).into_boxed_str()), source: None };
        let diagnostics = ErrorLogDiagnostics::from_error(&long);
        assert_eq!(diagnostics.cause().expect("bounded cause").len(), MAX_LOG_CAUSE_BYTES);
        assert!(diagnostics.cause().expect("bounded cause").ends_with("..."));
        assert!(diagnostics.truncated());
    }
}
