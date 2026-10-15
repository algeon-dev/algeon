//! One error taxonomy for the DataFusion adapter boundary.
//!
//! Adapter errors classify failures from DataFusion planning/execution,
//! native execution, cuDF calls made directly by the adapter, and native
//! engine errors wrapped at the adapter boundary. The stable
//! machine-readable [`ErrorCode`] is the contract; the human message is
//! diagnostic only. Native errors keep their original classification in
//! `native_*` facts while this layer maps them into adapter-level
//! [`ErrorCode`] values suitable for DataFusion.
//!
//! The boundary rules this module enforces are owned by
//! `website/docs/design/errors.mdx`:
//!
//! - An [`Error`] crosses into DataFusion once, as `DataFusionError::External`,
//!   through `From`. There is no conversion back; `find_algeon_error` walks
//!   whatever DataFusion wrapped around it.
//! - A native `QueryError` is converted when it reaches the adapter, through
//!   `from_native`. `kind` and `status` are derived from `code`, never chosen
//!   per site.
//! - A candidate decline is a report outcome, never an error from here.
//! - Facts are raw unless a producer opts them into the log-safe projection.
//!   Caller schema names travel as [`ClientDetail`], outside `message()`,
//!   logs and `Debug`, because the Flight server logs the message verbatim.
//! - Ordinary DataFusion errors keep their identity; `classify` assigns them a
//!   server code without making them Algeon errors.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
    sync::Arc,
};

use datafusion::{
    common::{DataFusionError, SchemaError},
    logical_expr::{DdlStatement, LogicalPlan},
    object_store::Error as ObjectStoreError,
};
use query_engine::object_store_error::query_error_code_for_object_store_error;
use query_planning::error::QueryError;
use query_runtime::native_error::{cudf_message_fact, query_error_code_for_cudf_error};

pub(crate) mod client;
mod code;

use code::code_for_native_error;
pub use code::{ErrorCode, ErrorKind, ErrorStatus};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

const NATIVE_FACT_COLLISION_KEYS: &str = "native_fact_collision_keys";
/// Byte counts of a native memory failure. Numeric values contain no caller
/// text and are included in retained query diagnostics.
const NATIVE_MEMORY_FACT_KEYS: [&str; 4] =
    ["requested_allocation_bytes", "budget_bytes", "budget_peak_bytes", "budget_current_bytes"];
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
/// Because these values contain caller column identifiers and schema types
/// (nested types print field names), they are excluded from operational logs,
/// [`fmt::Debug`], and [`Error::message`]. The Flight status text, the Flight
/// metadata allowlist, and query-stats capture read them from here.
#[derive(Clone, PartialEq, Eq)]
pub enum ClientDetail {
    /// A column whose Arrow type the configured GPU execution requirement
    /// cannot execute.
    UnsupportedColumn {
        /// Caller column identifier rejected by the execution requirement.
        name: String,
        /// Display form of the column's Arrow type, including nested field names.
        data_type: String,
        /// Semantic input role used in client-visible diagnostic facts.
        role: &'static str,
    },
    /// A relation field whose Arrow type a GPU function rejects.
    RelationField {
        /// Caller relation-field identifier rejected by a GPU function.
        field: String,
        /// Display form of the field's Arrow type.
        data_type: String,
    },
    /// A field a rejection names: absent from its relation or named by
    /// native validation.
    Field {
        /// Caller field identifier named by the rejection.
        name: String,
    },
    /// A column native validation or execution names.
    Column {
        /// Caller column identifier named by native validation or execution.
        name: String,
    },
    /// An input ORDER BY key that is not an output column of its input.
    OrderingKey {
        /// ORDER BY expression that could not bind to the input's output columns.
        name: String,
    },
    /// The plan element that keeps the configured GPU execution requirement
    /// from holding, as one line that can quote the caller's SQL and
    /// identifiers.
    BlockingSubject {
        /// Client-visible plan-element description, which may include SQL literals and identifiers.
        text: String,
    },
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
            Self::BlockingSubject { text } => vec![("blocking_subject", text.as_str())],
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
            Self::BlockingSubject { .. } => f.debug_struct("BlockingSubject").finish_non_exhaustive(),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailedStatement {
    /// `DROP TABLE` or `DROP VIEW`. Neither reads table data, and DataFusion
    /// 55 reports every failure of both, including a catalog that cannot
    /// deregister the object, as its missing-object `Execution` error.
    DropTableOrView,
    /// Any other statement, or a failure before a statement was planned.
    Other,
}

impl FailedStatement {
    #[cfg_attr(
        not(feature = "server"),
        expect(dead_code, reason = "only the Flight statement boundary names a DDL statement")
    )]
    pub(crate) fn of(plan: &LogicalPlan) -> Self {
        match plan {
            LogicalPlan::Ddl(DdlStatement::DropTable(_) | DdlStatement::DropView(_)) => Self::DropTableOrView,
            _ => Self::Other,
        }
    }
}

/// Classifies a DataFusion error into the adapter code the Flight server reports.
///
/// An Algeon error anywhere in the tree takes precedence. Otherwise a typed
/// object-store error in the source chain maps to its source code, and the
/// deepest DataFusion error reached decides classification alongside the failing
/// statement. Traversal visits at most `MAX_SOURCE_CHAIN_DEPTH + 1` layers,
/// ensuring deep wrapper trees yield a defined code.
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
/// Raw lookup and iteration preserve every value. [`fmt::Debug`]
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

    /// Inserts a fact approved by the caller for log disclosure.
    ///
    /// Empty or duplicate keys follow [`Facts`] collision rules. The reserved
    /// truncation marker is retained as raw data and cannot mark itself log-safe.
    pub fn insert_log_safe(&mut self, key: &'static str, value: String) {
        // The truncation marker is owned by the projection. A caller value under
        // this key is stored raw and never marked log-safe. Such a value cannot
        // spoof the synthetic retained marker.
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

    /// Returns only disclosure-approved facts after sanitizing and applying count and byte bounds.
    ///
    /// The projection records truncation without modifying the stored raw facts.
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
    /// are sanitized and bounded the same way as log-safe values. Their presence
    /// here discloses them to the query caller and does not claim log safety.
    /// A single `error_facts_truncated` marker covers truncation in either
    /// projection. Operational logs and [`fmt::Debug`] keep using
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
    /// Copies sanitized entries for retained diagnostics, adding
    /// `error_facts_truncated=true` when any bound shortened the projection.
    pub fn retained_entries(&self) -> Vec<(String, String)> {
        let mut entries =
            self.entries.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect::<Vec<_>>();
        if self.truncated {
            entries.push((ERROR_FACTS_TRUNCATED_KEY.to_string(), "true".to_string()));
            entries.sort_by(|left, right| left.0.cmp(&right.0));
        }
        entries
    }

    /// Formats the sanitized entries as comma-separated `key=value` pairs.
    ///
    /// Returns `None` when no approved entries remain.
    pub fn rendered(&self) -> Option<String> {
        (!self.entries.is_empty())
            .then(|| self.entries.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join(", "))
    }

    /// Returns whether sanitization bounds omitted or shortened any approved fact.
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
/// [`fmt::Debug`] renders diagnostic text containing only log-safe facts and
/// omits the underlying causal chain.
pub struct Error {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    /// Boxed so an `Error` without detail stays small; the detail is present
    /// on strict rejections and schema validation failures.
    client_detail: Option<Box<ClientDetail>>,
    /// Set at most once; later `with_source` calls keep the first cause.
    source: Option<BoxedSourceError>,
}

/// Human diagnostics derived from an error source chain.
///
/// Retained query diagnostics should keep machine-readable fields (`code`,
/// `kind`, `status`, `operation`, facts) separate from human root-cause text.
/// This formatter walks `std::error::Error::source()` and renders each project
/// error by its human message. Structured `Display` output is not used. Each
/// layer's text is capped at `MAX_SOURCE_CHAIN_MESSAGE_BYTES` and the joined
/// chain at `MAX_SOURCE_CHAIN_BYTES`, both with `...` markers; the in-process
/// error is never rewritten. Server-side logs should use [`ErrorLogDiagnostics`].
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
    /// Each layer's text is truncated to the log cause budget before retention,
    /// bounding rendered message sizes across arbitrary foreign `Display`
    /// implementations. A foreign implementation can still spend memory
    /// producing its own message; that cost cannot be bounded externally.
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
    /// Opts a fact into log and diagnostic projection under the disclosure policy.
    /// This method differs from [`Self::with_fact`] by permitting external visibility,
    /// and callers must restrict usage to bounded operational classifications that omit
    /// credentials, identifiers, paths, query literals, and sensitive payloads.
    /// Projections apply defensive redaction for URI and credential-like tokens
    /// alongside size limits, but sanitization does not replace careful classification.
    /// External or arbitrary facts must use [`Self::with_fact`] to remain hidden by default.
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

    /// Replaces this error's fact set with an existing set, preserving its disclosure flags and collision evidence.
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
        // Server I/O lifecycle wraps cuDF errors without a native QueryError. The
        // adapter derives the code from the runtime's single cuDF classification,
        // so only one mapping table exists.
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
            let value = value.into_owned();
            if NATIVE_MEMORY_FACT_KEYS.contains(&key) && value.parse::<u64>().is_ok() {
                error.facts.insert_log_safe(key, value);
            } else {
                error.facts.insert(key, value);
            }
        }
        if !collision_keys.is_empty() {
            error.facts.insert_log_safe(NATIVE_FACT_COLLISION_KEYS, collision_keys.join(","));
        }
        error.with_source(source)
    }

    /// [`Self::from_native`] whose adapter message also carries the native
    /// message and the allowlisted facts below.
    ///
    /// Used at boundaries whose consumers print only the outermost message
    /// (backend construction, device probing, composed admission), preserving
    /// the cause and actionable capacity numbers. Schema identifiers such as
    /// column and field names are isolated into [`ClientDetail`].
    pub(crate) fn from_native_with_blocking_diagnostics(
        operation: &'static str,
        message: impl Into<String>,
        source: QueryError,
    ) -> Self {
        // `field` and `column` name the caller's schema, so they travel as
        // client detail and stay out of the logged message.
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
            "null_count",
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

    /// Caller schema detail this error names.
    pub fn client_detail(&self) -> Option<&ClientDetail> {
        self.client_detail.as_deref()
    }

    /// Returns the underlying source error when one was attached.
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
mod tests;
