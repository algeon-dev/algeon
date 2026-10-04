//! One error taxonomy for the DataFusion adapter boundary.
//!
//! Adapter errors classify failures from DataFusion planning/execution,
//! native execution, cuDF calls made directly by the adapter, and native
//! engine errors wrapped at the adapter boundary. The stable
//! machine-readable [`ErrorCode`] is the contract; the human message is
//! diagnostic only. Native errors keep their original classification in
//! `native_*` facts while this layer maps them into adapter-level
//! [`ErrorCode`] values suitable for DataFusion.

#[cfg(feature = "server")]
use std::sync::Arc;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
};

use datafusion::common::DataFusionError;
#[cfg(feature = "server")]
use datafusion::{
    common::SchemaError,
    logical_expr::{DdlStatement, LogicalPlan},
    object_store::Error as ObjectStoreError,
};
#[cfg(feature = "server")]
use query_engine::object_store_error::query_error_code_for_object_store_error;
use query_planning::error::QueryError;
use query_runtime::native_error::{cudf_message_fact, query_error_code_for_cudf_error};

mod code;

use code::code_for_native_error;
pub use code::{ErrorCode, ErrorKind, ErrorStatus};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

const NATIVE_FACT_COLLISION_KEYS: &str = "native_fact_collision_keys";
const FACT_COLLISION_KEYS: &str = "fact_collision_keys";
const EMPTY_FACT_KEY: &str = "<empty>";
const FALLBACK_OPERATION: &str = "algeon_datafusion.error.unspecified_operation";
const FALLBACK_MESSAGE: &str = "Algeon DataFusion adapter error without a message";

/// Adapter result type used before conversion to `DataFusionError`.
pub type Result<T> = std::result::Result<T, Error>;

const MAX_SOURCE_CHAIN_DEPTH: usize = 32;
const MAX_SOURCE_CHAIN_MESSAGE_BYTES: usize = 512;
const MAX_SOURCE_CHAIN_BYTES: usize = 2_048;
const MAX_LOG_CAUSE_BYTES: usize = 512;
const MAX_LOG_SOURCE_PATH_BYTES: usize = 1_024;
const MAX_ALGEON_ERROR_TRAVERSAL_DEPTH: usize = 64;
const MAX_ALGEON_ERROR_TRAVERSAL_NODES: usize = 256;
const MAX_LOG_SAFE_FACT_COUNT: usize = 16;
const MAX_LOG_SAFE_FACT_VALUE_BYTES: usize = 256;
const MAX_LOG_SAFE_FACTS_RENDERED_BYTES: usize = 2_048;
pub(crate) const ERROR_FACTS_TRUNCATED_KEY: &str = "error_facts_truncated";

/// Caller schema detail an error names.
///
/// The values name the query caller's own columns and Arrow types (nested
/// types print field names), so they never enter operational logs,
/// [`fmt::Debug`], or [`Error::message`]. The Flight status text, the Flight
/// metadata allowlist, and query-stats capture read them from here.
#[derive(Clone, PartialEq, Eq)]
pub enum ClientDetail {
    /// A column whose Arrow type the configured GPU execution requirement
    /// cannot execute.
    UnsupportedColumn { name: String, data_type: String, role: &'static str },
    /// A relation field whose Arrow type a GPU function rejects.
    RelationField { field: String, data_type: String },
    /// A field a rejection names: absent from its relation, nullable where
    /// the function requires non-null values, or named by native validation.
    Field { name: String },
    /// A column native validation names.
    Column { name: String },
    /// An input ORDER BY key that is not an output column of its input.
    OrderingKey { name: String },
}

impl ClientDetail {
    /// The detail as wire facts, under the keys query stats and Flight
    /// metadata carry.
    pub fn entries(&self) -> Vec<(&'static str, &str)> {
        match self {
            Self::UnsupportedColumn { name, data_type, role } => vec![
                ("unsupported_column", name.as_str()),
                ("unsupported_column_type", data_type.as_str()),
                ("unsupported_column_role", role),
            ],
            Self::RelationField { field, data_type } => {
                vec![("field", field.as_str()), ("actual_type", data_type.as_str())]
            },
            Self::Field { name } => vec![("field", name.as_str())],
            Self::Column { name } => vec![("column", name.as_str())],
            Self::OrderingKey { name } => vec![("key", name.as_str())],
        }
    }

    /// The wire fact `key` of this detail.
    pub fn value(&self, key: &str) -> Option<&str> {
        self.entries().into_iter().find_map(|(entry, value)| (entry == key).then_some(value))
    }
}

impl fmt::Debug for ClientDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedColumn { role, .. } => {
                f.debug_struct("UnsupportedColumn").field("role", role).finish_non_exhaustive()
            },
            Self::RelationField { .. } => f.debug_struct("RelationField").finish_non_exhaustive(),
            Self::Field { .. } => f.debug_struct("Field").finish_non_exhaustive(),
            Self::Column { .. } => f.debug_struct("Column").finish_non_exhaustive(),
            Self::OrderingKey { .. } => f.debug_struct("OrderingKey").finish_non_exhaustive(),
        }
    }
}

/// DataFusion error inspection helpers provided by algeon-datafusion.
pub trait DataFusionErrorExt {
    /// Finds the first Algeon [`Error`] in this DataFusion error tree.
    ///
    /// Traversal is deterministic depth-first pre-order. Generic
    /// [`std::error::Error::source`] links are followed, while every
    /// [`DataFusionError::Collection`] branch is visited in collection order.
    /// The first match in that order is returned. Traversal is bounded
    /// defensively by both depth and total node count. A `None` result only
    /// means no Algeon error was found within that budget; it does not prove
    /// that branches beyond the budget contain no Algeon identity.
    fn find_algeon_error(&self) -> Option<&Error>;
}

impl DataFusionErrorExt for DataFusionError {
    fn find_algeon_error(&self) -> Option<&Error> {
        let mut visited_nodes = 0;
        find_algeon_error_in(self, 0, &mut visited_nodes)
    }
}

fn find_algeon_error_in<'a>(
    error: &'a (dyn StdError + 'static),
    depth: usize,
    visited_nodes: &mut usize,
) -> Option<&'a Error> {
    if *visited_nodes >= MAX_ALGEON_ERROR_TRAVERSAL_NODES {
        return None;
    }
    *visited_nodes += 1;

    if let Some(error) = error.downcast_ref::<Error>() {
        return Some(error);
    }
    if depth >= MAX_ALGEON_ERROR_TRAVERSAL_DEPTH {
        return None;
    }

    if let Some(DataFusionError::Collection(errors)) = error.downcast_ref::<DataFusionError>() {
        return errors.iter().find_map(|error| find_algeon_error_in(error, depth + 1, visited_nodes));
    }

    error.source().and_then(|source| find_algeon_error_in(source, depth + 1, visited_nodes))
}

/// The statement whose DataFusion error [`classify`] maps.
#[cfg(feature = "server")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailedStatement {
    /// `DROP TABLE` or `DROP VIEW`. Neither reads table data, and DataFusion
    /// 55 reports every failure of both, including a catalog that cannot
    /// deregister the object, as its missing-object `Execution` error.
    DropTableOrView,
    /// Any other statement, or a failure before a statement was planned.
    Other,
}

#[cfg(feature = "server")]
impl FailedStatement {
    pub(crate) fn of(plan: &LogicalPlan) -> Self {
        match plan {
            LogicalPlan::Ddl(DdlStatement::DropTable(_) | DdlStatement::DropView(_)) => Self::DropTableOrView,
            _ => Self::Other,
        }
    }
}

/// Classifies a DataFusion error into the adapter code the Flight server reports.
///
/// An Algeon error anywhere in the tree decides first. Otherwise a typed
/// object-store error in the source chain maps to its source code, and the
/// deepest DataFusion error reached decides the rest, together with the
/// statement that failed. The walk visits at most `MAX_SOURCE_CHAIN_DEPTH + 1`
/// layers and classifies the deepest DataFusion error seen, so an arbitrarily
/// deep wrapper tree still yields a defined code.
#[cfg(feature = "server")]
pub(crate) fn classify(error: &DataFusionError, statement: FailedStatement) -> ErrorCode {
    if let Some(adapter) = error.find_algeon_error() {
        return adapter.code();
    }
    let mut deepest = error;
    let mut object_store = None;
    let mut layer: Option<&(dyn StdError + 'static)> = Some(error);
    for _ in 0..=MAX_SOURCE_CHAIN_DEPTH {
        let Some(current) = layer else { break };
        if let Some(datafusion) = current.downcast_ref::<DataFusionError>() {
            deepest = datafusion;
        } else if let Some(shared) = current.downcast_ref::<Arc<DataFusionError>>() {
            deepest = shared.as_ref();
        } else if object_store.is_none() {
            object_store = current.downcast_ref::<ObjectStoreError>();
        }
        layer = current.source();
    }
    if let Some(object_store) = object_store {
        return code_for_native_error(query_error_code_for_object_store_error(object_store));
    }
    match deepest {
        DataFusionError::SQL(_, _) => ErrorCode::SqlParse,
        DataFusionError::Plan(_) | DataFusionError::Configuration(_) => ErrorCode::DataFusionPlanning,
        DataFusionError::SchemaError(schema, _) => match schema.as_ref() {
            SchemaError::FieldNotFound { .. } => ErrorCode::FieldNotFound,
            SchemaError::AmbiguousReference { .. }
            | SchemaError::DuplicateQualifiedField { .. }
            | SchemaError::DuplicateUnqualifiedField { .. } => ErrorCode::DataFusionPlanning,
        },
        DataFusionError::NotImplemented(_) => ErrorCode::UnsupportedOperator,
        DataFusionError::ResourcesExhausted(_) => ErrorCode::HostMemoryExhausted,
        DataFusionError::ExecutionJoin(join) if join.is_cancelled() => ErrorCode::Cancelled,
        DataFusionError::Execution(_) if statement == FailedStatement::DropTableOrView => ErrorCode::ObjectNotFound,
        // Other execution errors may include reader internals, so they stay
        // behind the generic `Internal` client surface.
        _ => ErrorCode::Internal,
    }
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
/// classification and `native_*` for wrapped `query-engine` errors;
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

    /// Retained entries for query-caller-visible capture (query stats).
    ///
    /// Combines the log-safe operational projection with the error's
    /// [`ClientDetail`]. Detail values are raw caller schema identifiers: they
    /// are sanitized and bounded exactly like log-safe values, but their
    /// presence here is a disclosure to the query caller, not a log-safety
    /// claim. A single `error_facts_truncated` marker covers truncation in
    /// either projection. Operational logs and [`fmt::Debug`] keep using
    /// [`Self::log_safe_projection`] without the detail.
    pub fn retained_capture_entries(&self, client_detail: Option<&ClientDetail>) -> Vec<(String, String)> {
        let safe = self.log_safe_projection();
        let mut truncated = safe.truncated;
        let mut entries =
            safe.entries.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect::<Vec<_>>();
        for (key, raw) in client_detail.map(ClientDetail::entries).unwrap_or_default() {
            let (value, value_truncated) = truncate_with_marker(&sanitize_log_text(raw), MAX_LOG_SAFE_FACT_VALUE_BYTES);
            truncated |= value_truncated;
            entries.push((key.to_string(), value));
        }
        if truncated {
            entries.push((ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string()));
        }
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        entries
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
/// [`fmt::Display`] renders only the human-facing [`Self::message`].
/// Machine policy must use the typed accessors (`code`, `kind`, `status`,
/// `operation`, fact lookups, `find_algeon_error`), never parsed text.
/// [`fmt::Debug`] is a human diagnostic: it carries only the bounded
/// log-safe fact projection and omits the source chain, so it is not a
/// machine-readable contract either.
pub struct Error {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    client_detail: Option<Box<ClientDetail>>,
    source: Option<BoxedSourceError>,
}

/// Human diagnostics derived from an error source chain.
///
/// Retained query diagnostics should keep machine-readable fields (`code`,
/// `kind`, `status`, `operation`, facts) separate from human root-cause text.
/// This formatter walks `std::error::Error::source()` and renders project
/// errors by their human message, not by their structured `Display`
/// representation. Each layer's text is capped at
/// `MAX_SOURCE_CHAIN_MESSAGE_BYTES` and the joined chain at
/// `MAX_SOURCE_CHAIN_BYTES`, both with `...` markers; the in-process error is
/// never rewritten. Server-side logs should use [`ErrorLogDiagnostics`].
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
    ///
    /// Each layer's text is truncated to the log cause budget before it is
    /// retained, so one extreme foreign `Display` cannot force every
    /// projection to render it in full. A foreign implementation can still
    /// spend memory producing its own message; that cost cannot be bounded
    /// from the outside.
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
            let (message, message_truncated) =
                truncate_with_marker(&log_diagnostic_error_message(error), MAX_LOG_CAUSE_BYTES);
            diagnostics.truncated |= message_truncated;
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
        let mut chain_bytes = 0usize;
        let mut chain_dropped = false;
        for _ in 0..MAX_SOURCE_CHAIN_DEPTH {
            let Some(error) = current else {
                break;
            };
            // Truncate each layer's own text before retaining it: an extreme
            // but supported foreign `Display` must not make every projection
            // unbounded. Markers keep the cut observable; the in-process error
            // itself is never rewritten.
            let (message, _) = truncate_with_marker(&diagnostic_error_message(error), MAX_SOURCE_CHAIN_MESSAGE_BYTES);
            if !message.is_empty() && messages.last() != Some(&message) {
                chain_bytes = chain_bytes.saturating_add(message.len());
                if chain_bytes > MAX_SOURCE_CHAIN_BYTES {
                    chain_dropped = true;
                    break;
                }
                messages.push(message);
            }
            current = error.source();
        }
        let root_cause = messages.last().cloned();
        let chain = (!messages.is_empty()).then(|| {
            let joined = messages.join(": ");
            let (mut capped, truncated) = truncate_with_marker(&joined, MAX_SOURCE_CHAIN_BYTES);
            if chain_dropped && !truncated {
                capped = truncate_with_marker(&format!("{capped} ..."), MAX_SOURCE_CHAIN_BYTES).0;
            }
            capped
        });
        Self { chain, root_cause }
    }
}

impl Error {
    /// Creates an adapter error with a stable operation id and message.
    ///
    /// `operation` should be a stable dotted identifier such as
    /// `algeon_datafusion.native_exec.execute`. `message` is human-facing
    /// diagnostic text. Add machine-readable context with [`Self::with_fact`].
    pub fn new(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        let operation = if operation.is_empty() { FALLBACK_OPERATION } else { operation };
        let message = if message.is_empty() { FALLBACK_MESSAGE.to_string() } else { message };
        Self { code, operation, message, facts: Facts::new(), client_detail: None, source: None }
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
    /// [`Self::with_fact`] and remain hidden by default.
    ///
    /// Caller schema details (for example strict-rejection column names) are
    /// a [`ClientDetail`], never log-safe facts. The reserved
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

    /// Attaches the caller schema detail this error names, when none is
    /// attached yet.
    #[must_use]
    pub fn with_client_detail(mut self, detail: ClientDetail) -> Self {
        self.client_detail.get_or_insert_with(|| Box::new(detail));
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
        // Server I/O lifecycle wraps cuDF errors without a native QueryError, so
        // the adapter derives the code from the runtime's single cuDF
        // classification rather than a second table.
        let code = code_for_native_error(query_error_code_for_cudf_error(&source));
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
    pub(crate) fn from_native(operation: &'static str, message: impl Into<String>, source: QueryError) -> Self {
        let code = code_for_native_error(source.code());
        let mut error = Self::new(code, operation, message)
            .with_log_safe_fact("native_code", source.code())
            .with_log_safe_fact("native_kind", source.kind())
            .with_log_safe_fact("native_status", source.status())
            .with_log_safe_fact("native_operation", source.operation())
            .with_fact("native_message", source.message());
        let mut collision_keys = Vec::new();
        for (key, value) in source.wire_facts() {
            if key == NATIVE_FACT_COLLISION_KEYS || error.facts.get(key).is_some() {
                collision_keys.push(key);
                continue;
            }
            error.facts.insert(key, value.into_owned());
        }
        if !collision_keys.is_empty() {
            error.facts.insert_log_safe(NATIVE_FACT_COLLISION_KEYS, collision_keys.join(","));
        }
        error.with_source(source)
    }

    pub(crate) fn from_native_with_blocking_diagnostics(
        operation: &'static str,
        message: impl Into<String>,
        source: QueryError,
    ) -> Self {
        // `field` and `column` name the caller's schema, so they travel as
        // client detail rather than in the logged message.
        const CLIENT_FACTS: &[&str] = &[
            "device_ordinal",
            "stage",
            "phase",
            "allocation_limit_bytes",
            "physical_total_bytes",
            "configured_limit_bytes",
            "owned_capacity_bytes",
            "usable_capacity_bytes",
            "managed_capacity_bytes",
            "backend_reserve_bytes",
            "limit_bytes",
            "fallback_bytes",
            "schedule_reason",
            "sort_rejection_reason",
            "reason",
            "validation_reason",
            "relation_role",
            "training_rows",
            "non_finite_count",
            "remedy",
            "occurrence_id",
            "node_id",
            "input_node_id",
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
        let wire = source.wire_facts().collect::<BTreeMap<_, _>>();
        let facts = CLIENT_FACTS
            .iter()
            .filter_map(|key| wire.get(key).map(|value| format!("{key}={value}")))
            .collect::<Vec<_>>();
        if !facts.is_empty() {
            message.push_str(" [");
            message.push_str(&facts.join(", "));
            message.push(']');
        }
        let detail = match (wire.get("column"), wire.get("field")) {
            (Some(column), _) => Some(ClientDetail::Column { name: column.to_string() }),
            (None, Some(field)) => Some(ClientDetail::Field { name: field.to_string() }),
            (None, None) => None,
        };
        let error = Self::from_native(operation, message, source);
        match detail {
            Some(detail) => error.with_client_detail(detail),
            None => error,
        }
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
    /// Caller schema detail this error names.
    pub fn client_detail(&self) -> Option<&ClientDetail> {
        self.client_detail.as_deref()
    }

    pub fn source_error(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }
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
// downcast the boxed source back to `algeon_datafusion::Error` and inspect code,
// kind, status, operation, facts, and source.
impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        Self::External(Box::new(error))
    }
}

fn diagnostic_error_message(error: &(dyn StdError + 'static)) -> String {
    if let Some(error) = error.downcast_ref::<Error>() {
        return normalize_diagnostic_message(error.message());
    }
    if let Some(error) = error.downcast_ref::<QueryError>() {
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
    error.downcast_ref::<QueryError>().and_then(|error| error.fact_value("path"))
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
    use cudf::{ErrorCode as CudfErrorCode, io::parquet::RemoteParquetSource, testing::error::new_with_for_test};
    use datafusion::common::Diagnostic;
    use query_planning::error::QueryErrorCode;
    use query_runtime::native_error::cudf_query_error;

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

    fn algeon_error(operation: &'static str) -> Error {
        Error::new(ErrorCode::Internal, operation, "test Algeon error")
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
            Error::new(ErrorCode::Internal, "algeon_datafusion.test.source", "source test")
                .with_source(TestSourceError::leaf("first typed source"))
                .with_boxed_source(Box::new(TestSourceError::leaf("second boxed source")))
        })
        .expect("attaching a second boxed source must not unwind");
        assert_eq!(first_typed.source_error().map(ToString::to_string).as_deref(), Some("first typed source"));

        let first_boxed = std::panic::catch_unwind(|| {
            Error::new(ErrorCode::Internal, "algeon_datafusion.test.source", "source test")
                .with_boxed_source(Box::new(TestSourceError::leaf("first boxed source")))
                .with_source(TestSourceError::leaf("second typed source"))
        })
        .expect("attaching a second typed source must not unwind");
        assert_eq!(first_boxed.source_error().map(ToString::to_string).as_deref(), Some("first boxed source"));
    }

    #[test]
    fn log_safe_fact_projection_sanitizes_and_sorts_without_changing_raw_facts() {
        let error = Error::new(ErrorCode::Internal, "algeon_datafusion.test.facts", "fact projection test")
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
        let value_capped = Error::new(ErrorCode::Internal, "algeon_datafusion.test.value_cap", "value cap")
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
        let mut count_capped = Error::new(ErrorCode::Internal, "algeon_datafusion.test.count_cap", "count cap");
        for key in keys {
            count_capped = count_capped.with_log_safe_fact(key, "safe");
        }
        let projection = count_capped.facts().log_safe_projection();
        assert_eq!(projection.entries.len(), MAX_LOG_SAFE_FACT_COUNT);
        assert!(projection.truncated());
        assert_eq!(projection.retained_entries()[0], (ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string()));

        let mut total_capped = Error::new(ErrorCode::Internal, "algeon_datafusion.test.total_cap", "total cap");
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
        let not_truncated = Error::new(ErrorCode::Internal, "algeon_datafusion.test.marker", "marker")
            .with_log_safe_fact(ERROR_FACTS_TRUNCATED_KEY, "true")
            .with_log_safe_fact("safe", "classification");
        assert_eq!(not_truncated.fact_value(ERROR_FACTS_TRUNCATED_KEY), Some("true"));
        let projection = not_truncated.facts().log_safe_projection();
        assert_eq!(projection.entries, vec![("safe", "classification".to_string())]);
        assert!(!projection.truncated());
        assert!(projection.retained_entries().iter().all(|(key, _)| key != ERROR_FACTS_TRUNCATED_KEY));

        let actually_truncated = Error::new(ErrorCode::Internal, "algeon_datafusion.test.marker", "marker")
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
        let error = Error::new(ErrorCode::Internal, "algeon_datafusion.test.debug", "debug projection")
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
        let native = QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.test.safe_projection",
            "native-message-must-stay-hidden",
        )
        .with_fact("literal", "native-user-literal")
        .with_fact("path", "/private/native.parquet");
        let native_adapter =
            Error::from_native("algeon_datafusion.test.safe_projection", "wrapped native error", native);
        let projection = native_adapter.facts().log_safe_projection();
        assert_eq!(
            projection.entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec!["native_code", "native_kind", "native_operation", "native_status"]
        );
        let rendered = projection.rendered().unwrap();
        for hidden in ["native-message-must-stay-hidden", "native-user-literal", "/private/native.parquet"] {
            assert!(!rendered.contains(hidden), "{rendered}");
        }

        let cudf =
            RemoteParquetSource::s3("gs://user:password@bucket/private.parquet").expect_err("non-s3 scheme must fail");
        let cudf_adapter = Error::from_cudf("algeon_datafusion.test.safe_projection", "wrapped cudf error", cudf);
        let projection = cudf_adapter.facts().log_safe_projection();
        assert_eq!(
            projection.entries.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec!["cudf_code", "cudf_kind", "cudf_operation", "cudf_status"]
        );
        assert!(!projection.rendered().unwrap().contains("password"));
    }

    #[test]
    fn blocking_native_diagnostics_reach_display_without_unrelated_facts() {
        let native = QueryError::new(
            QueryErrorCode::UnsupportedPipelineContract,
            "query_engine.admission.composed_consumer",
            "a composed graph attempt lacks a complete ordered memory schedule",
        )
        .with_fact("schedule_reason", "consumer_local_source_rows_unavailable")
        .with_fact("stage", "prepared_composed_admission")
        .with_fact("node_id", 17)
        .with_fact("path", "/private/customer.parquet")
        .with_fact("literal", "secret");
        let adapter = Error::from_native_with_blocking_diagnostics(
            "algeon_datafusion.native_composed.admission_evidence",
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

        let native = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.query_service.admission",
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
            "algeon_datafusion.native_exec.context",
            "failed to acquire backend query-service algeon native execution context",
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

        let native = QueryError::new(
            QueryErrorCode::CudfInvalidInput,
            "query_engine.vector_algorithm.pack",
            "failed to pack the prepared vector feature table",
        )
        .with_fact("relation_role", "dataset")
        .with_fact("training_rows", 0)
        .with_fact("validation_reason", "non_finite_vector_value")
        .with_fact("non_finite_count", 5)
        .with_fact("remedy", "replace or reject null, NaN, and infinite values before GPU consumer execution")
        .with_fact("cuvs_message", "/private/customer.parquet");
        let adapter = Error::from_native_with_blocking_diagnostics(
            "algeon_datafusion.cuvs_sql.execute",
            "native cuVS vector query execution failed",
            native,
        );
        assert_eq!(adapter.kind(), ErrorKind::InvalidInput);
        let display = adapter.to_string();
        for fact in [
            "relation_role=dataset",
            "training_rows=0",
            "validation_reason=non_finite_vector_value",
            "non_finite_count=5",
            "remedy=replace or reject null, NaN, and infinite values before GPU consumer execution",
        ] {
            assert!(display.contains(fact), "{display}");
        }
        assert!(!display.contains("/private/customer.parquet"), "{display}");
    }

    #[test]
    fn error_debug_omits_native_and_cudf_source_payloads_but_keeps_typed_access() {
        let native = QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.test.debug_source",
            "native-source-message-secret",
        )
        .with_fact("literal", "native-source-fact-secret");
        let native_adapter = Error::from_native("algeon_datafusion.test.debug_source", "native wrapper", native);
        let native_source = native_adapter
            .source_error()
            .and_then(|source| source.downcast_ref::<QueryError>())
            .expect("native source must remain typed");
        assert_eq!(native_source.fact_value("literal"), Some("native-source-fact-secret"));
        let native_debug = format!("{native_adapter:?}");
        assert!(native_debug.contains("<source omitted>"), "{native_debug}");
        assert!(!native_debug.contains("native-source-message-secret"), "{native_debug}");
        assert!(!native_debug.contains("native-source-fact-secret"), "{native_debug}");

        let cudf = new_with_for_test(
            CudfErrorCode::ForeignInvalidArgument,
            "cudf.test.debug_source",
            "cudf-source-detail-secret client_secret=cudf-secret-value",
        );
        let cudf_adapter = Error::from_cudf("algeon_datafusion.test.debug_source", "cudf wrapper", cudf);
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
    fn find_algeon_error_finds_direct_external_error() {
        let error = DataFusionError::External(Box::new(algeon_error("algeon_datafusion.test.direct")));

        assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.direct"));
    }

    #[test]
    fn native_error_status_projection_agrees_with_the_engine_for_every_code() {
        for native_code in QueryErrorCode::ALL {
            let native =
                QueryError::new(*native_code, "query_engine.test.status_projection", "native status projection test");
            let adapter = Error::from_native(
                "algeon_datafusion.test.status_projection",
                "adapter status projection test",
                native,
            );

            assert_eq!(
                adapter.status().as_str(),
                native_code.status().as_str(),
                "{native_code} must retain its native retry/remediation status"
            );
        }
    }

    #[test]
    fn find_algeon_error_traverses_context() {
        let error = DataFusionError::Context(
            "context".to_string(),
            Box::new(DataFusionError::External(Box::new(algeon_error("algeon_datafusion.test.context")))),
        );

        assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.context"));
    }

    #[test]
    fn find_algeon_error_traverses_diagnostic() {
        let error = DataFusionError::Diagnostic(
            Box::new(Diagnostic::new_error("diagnostic", None)),
            Box::new(DataFusionError::External(Box::new(algeon_error("algeon_datafusion.test.diagnostic")))),
        );

        assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.diagnostic"));
    }

    #[test]
    fn find_algeon_error_traverses_shared() {
        let error = DataFusionError::Shared(Arc::new(DataFusionError::External(Box::new(algeon_error(
            "algeon_datafusion.test.shared",
        )))));

        assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.shared"));
    }

    #[test]
    fn find_algeon_error_traverses_nested_collection_in_deterministic_order() {
        let error = DataFusionError::Collection(vec![
            DataFusionError::Plan("no Algeon error".to_string()),
            DataFusionError::Collection(vec![
                DataFusionError::Execution("still no Algeon error".to_string()),
                DataFusionError::External(Box::new(algeon_error("algeon_datafusion.test.first"))),
            ]),
            DataFusionError::External(Box::new(algeon_error("algeon_datafusion.test.second"))),
        ]);

        assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.first"));
    }

    #[test]
    fn find_algeon_error_traverses_collection_inside_arrow_external_wrapper() {
        let collection = DataFusionError::Collection(vec![
            DataFusionError::Plan("no Algeon error".to_string()),
            DataFusionError::External(Box::new(algeon_error("algeon_datafusion.test.arrow_collection"))),
        ]);
        let arrow = ArrowError::from_external_error(Box::new(collection));
        let error = DataFusionError::ArrowError(Box::new(arrow), None);

        assert_eq!(error.find_algeon_error().map(Error::operation), Some("algeon_datafusion.test.arrow_collection"));
    }

    #[test]
    fn find_algeon_error_returns_none_without_match() {
        let error = DataFusionError::Collection(vec![
            DataFusionError::Plan("planning failed".to_string()),
            DataFusionError::Context(
                "context".to_string(),
                Box::new(DataFusionError::Execution("execution failed".to_string())),
            ),
        ]);

        assert!(error.find_algeon_error().is_none());
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
    fn source_diagnostics_bound_extreme_chain_bytes_without_rewriting_the_error() {
        let mut error = DataFusionError::Execution("y".repeat(4_096));
        for depth in 0..8 {
            error = DataFusionError::Context(format!("wrapper-{depth}"), Box::new(error));
        }
        let error_display = error.to_string();
        assert!(error_display.len() > 4_096);

        let diagnostics = ErrorSourceDiagnostics::from_error(&error);

        let chain = diagnostics.chain().expect("source chain should be present");
        assert!(chain.len() <= 2_048, "chain must be bounded: {}", chain.len());
        assert!(chain.contains("..."), "the cut must stay observable: {chain}");
        let root_cause = diagnostics.root_cause().expect("root cause should be present");
        assert!(root_cause.len() <= 512, "root cause must be bounded: {}", root_cause.len());
        assert_eq!(error.to_string(), error_display, "projection must not rewrite the in-process error");
    }

    #[test]
    fn log_diagnostics_mark_truncation_from_any_layer_not_just_the_cause() {
        let error = Error::new(ErrorCode::Internal, "algeon_datafusion.test.log_layer_truncation", "outer wrapper")
            .with_source(DataFusionError::Context(
                "middle ".to_string() + &"m".repeat(4_000),
                Box::new(DataFusionError::Execution("leaf cause".to_string())),
            ));

        let diagnostics = ErrorLogDiagnostics::from_error(&error);

        assert_eq!(diagnostics.cause(), Some("leaf cause"));
        assert!(diagnostics.truncated(), "an intermediate cut must stay observable");
    }

    #[test]
    fn retained_capture_keeps_caller_schema_detail_outside_log_projection() {
        let error = Error::new(
            ErrorCode::FinalPlanRequirementUnsatisfied,
            "algeon_datafusion.test.strict_capture",
            "strict rejection probe",
        )
        .with_client_detail(ClientDetail::UnsupportedColumn {
            name: "countDetail".to_string(),
            data_type: "List(Int64)".to_string(),
            role: "projected",
        })
        .with_log_safe_fact("reason", "unsupported_type");

        let logs = error.facts().log_safe_projection().rendered().unwrap_or_default();
        assert!(!logs.contains("countDetail"), "operational logs must not carry caller schema detail: {logs}");

        let retained = error.facts().retained_capture_entries(error.client_detail());
        assert!(
            retained.contains(&("unsupported_column".to_string(), "countDetail".to_string())),
            "query-caller capture must retain the strict-rejection detail: {retained:?}"
        );
        assert!(retained.contains(&("unsupported_column_type".to_string(), "List(Int64)".to_string())), "{retained:?}");
        assert!(retained.contains(&("reason".to_string(), "unsupported_type".to_string())), "{retained:?}");
    }

    #[test]
    fn source_diagnostics_render_project_errors_by_message_not_structured_display() {
        let native = QueryError::new(QueryErrorCode::CudfDependencyFailure, "query_engine.test", "native root cause")
            .with_fact("native_fact", "value");
        let adapter = Error::from_native("algeon_datafusion.test", "adapter wrapper", native);
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
        let cudf = new_with_for_test(
            CudfErrorCode::ForeignInvalidArgument,
            "cudf.test.log_diagnostics",
            "CUDF failure at: /home/customer/cudf/reader.cu:518: Parquet header parsing failed with code(s) 0x400",
        );
        let native = cudf_query_error(
            "query_engine.test.log_diagnostics",
            "failed to read bounded native local parquet source chunk with cudf",
            cudf,
        )
        .with_fact("path", "/srv/algeon/dataset/part-00000.parquet");
        let adapter = Error::from_native(
            "algeon_datafusion.test.log_diagnostics",
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
        assert_eq!(diagnostics.source_path(), Some("/srv/algeon/dataset/part-00000.parquet"));
        assert!(!diagnostics.truncated());
    }

    #[test]
    fn log_diagnostics_redact_credentials_and_bound_foreign_messages() {
        let credentials = TestSourceError::leaf("request failed Authorization=Bearer-secret");
        assert_eq!(ErrorLogDiagnostics::from_error(&credentials).cause(), Some("<redacted-credential>"));

        let uri = Error::new(ErrorCode::Internal, "algeon_datafusion.test.log_uri", "read failed")
            .with_fact("path", "s3://user:secret@bucket/private.parquet");
        assert_eq!(ErrorLogDiagnostics::from_error(&uri).source_path(), Some("<redacted-uri>"));

        let long =
            TestSourceError { message: Box::leak("x".repeat(MAX_LOG_CAUSE_BYTES + 1).into_boxed_str()), source: None };
        let diagnostics = ErrorLogDiagnostics::from_error(&long);
        assert_eq!(diagnostics.cause().expect("bounded cause").len(), MAX_LOG_CAUSE_BYTES);
        assert!(diagnostics.cause().expect("bounded cause").ends_with("..."));
        assert!(diagnostics.truncated());
    }

    fn cudf_allocation_error_for_test(
        code: CudfErrorCode,
        failure_source: Option<cudf::CudfErrorFailureSource>,
        with_budget_scope: bool,
    ) -> cudf::Error {
        use cudf::testing::error;
        let mut error = error::new_with_for_test(code, "cudf.test.allocation_attribution", "synthetic cudf failure");
        if let Some(failure_source) = failure_source {
            error = error::with_ffi_detail_for_test(
                error,
                cudf::CudfErrorDetail {
                    status: 0,
                    failure_source,
                    domain_id: 0,
                    member_id: 0,
                    generation: 0,
                    requested_bytes: 4096,
                    alignment: 256,
                    cuda_error: -1,
                    message: "synthetic allocation failure".to_string(),
                },
            );
        }
        if with_budget_scope {
            error = error::with_budget_exhaustion_for_test(
                error,
                error::budget_exhaustion_detail_for_test(
                    1024,
                    1024,
                    1024,
                    512,
                    512,
                    4096,
                    1,
                    "test_boundary",
                    "cudf.test.allocation_attribution",
                ),
            );
        }
        error
    }

    #[test]
    fn cudf_physical_failure_with_budget_scope_is_device_out_of_memory() {
        let error = Error::from_cudf(
            "algeon_datafusion.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(
                CudfErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Physical),
                true,
            ),
        );

        assert_eq!(error.code(), ErrorCode::DeviceOutOfMemory);
    }

    #[test]
    fn cudf_reservation_limit_is_query_memory_limit_exceeded() {
        let error = Error::from_cudf(
            "algeon_datafusion.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(
                CudfErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::ReservationLimit),
                true,
            ),
        );

        assert_eq!(error.code(), ErrorCode::QueryMemoryLimitExceeded);
    }

    #[test]
    fn cudf_unknown_allocation_with_scope_usage_is_not_a_cap_rejection() {
        let error = Error::from_cudf(
            "algeon_datafusion.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(
                CudfErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Unknown),
                true,
            ),
        );

        assert_eq!(error.code(), ErrorCode::DeviceAllocationFailed);
    }

    #[test]
    fn cudf_dependency_failure_through_a_budgeted_scope_is_not_a_cap_rejection() {
        let error = Error::from_cudf(
            "algeon_datafusion.test.allocation_attribution",
            "wrapped cudf failure",
            cudf_allocation_error_for_test(CudfErrorCode::IoFailure, None, true),
        );

        assert_eq!(error.code(), ErrorCode::CudfDependencyFailure);
    }

    #[test]
    fn adapter_and_native_boundaries_agree_on_cudf_allocation_attribution() {
        let cases = [
            (
                CudfErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Physical),
                true,
                ErrorCode::DeviceOutOfMemory,
                QueryErrorCode::DeviceOutOfMemory,
            ),
            (
                CudfErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::ReservationLimit),
                true,
                ErrorCode::QueryMemoryLimitExceeded,
                QueryErrorCode::QueryMemoryLimitExceeded,
            ),
            (
                CudfErrorCode::OutOfMemory,
                Some(cudf::CudfErrorFailureSource::Unknown),
                true,
                ErrorCode::DeviceAllocationFailed,
                QueryErrorCode::DeviceAllocationFailed,
            ),
            (
                CudfErrorCode::IoFailure,
                None,
                true,
                ErrorCode::CudfDependencyFailure,
                QueryErrorCode::CudfDependencyFailure,
            ),
        ];

        for (index, (code, failure_source, with_budget_scope, expected_adapter, expected_native)) in
            cases.into_iter().enumerate()
        {
            let adapter = Error::from_cudf(
                "algeon_datafusion.test.allocation_attribution",
                "wrapped cudf failure",
                cudf_allocation_error_for_test(code, failure_source, with_budget_scope),
            );
            assert_eq!(adapter.code(), expected_adapter, "adapter mismatch for case {index}");
            let native = cudf_query_error(
                "query_engine.test.allocation_attribution",
                "wrapped cudf failure",
                cudf_allocation_error_for_test(code, failure_source, with_budget_scope),
            );
            assert_eq!(native.code(), expected_native, "native mismatch for case {index}");
        }
    }
}
