//! Error taxonomy for the DataFusion adapter boundary.
//!
//! Adapter errors classify failures from DataFusion planning/execution,
//! native execution, cuDF calls made directly by the adapter, and native
//! `nexus-query-engine` errors wrapped by [`Error::from_native`]. Native errors
//! keep their original classification in `native_*` facts while this layer maps
//! them into adapter-level [`ErrorCode`] values suitable for DataFusion.

use std::{collections::BTreeMap, error::Error as StdError, fmt};

use datafusion::common::DataFusionError;

mod code;

pub use code::{ErrorCode, ErrorKind, ErrorStatus, kind_for_code, status_for_code};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

const NATIVE_FACT_COLLISION_KEYS: &str = "native_fact_collision_keys";

/// Adapter result type used before conversion to `DataFusionError`.
pub type Result<T> = std::result::Result<T, Error>;

const MAX_SOURCE_CHAIN_DEPTH: usize = 32;

/// Machine-readable facts attached to an [`Error`].
///
/// Fact keys are static strings so they can be treated as a documented contract
/// by tests, reports, and callers. Values are stringified with [`ToString`].
/// Iteration is sorted by key because facts are stored in a [`BTreeMap`].
///
/// Empty keys and duplicate inserts panic. The adapter reserves `cudf_*` for
/// cuDF classification and `native_*` for wrapped `nexus-query-engine` errors;
/// colliding native facts are reported under `native_fact_collision_keys`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, String>,
}

impl Facts {
    /// Creates an empty fact set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the string value for `key` when present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    /// Iterates facts in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.values.iter().map(|(key, value)| (*key, value.as_str()))
    }

    fn insert(&mut self, key: &'static str, value: String) {
        assert!(!key.is_empty(), "datafusion-nexus error fact key must not be empty");
        let previous = self.values.insert(key, value);
        assert!(previous.is_none(), "datafusion-nexus error fact key inserted twice: {key}");
    }
}

/// Structured adapter error with stable classification and facts.
pub struct Error {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    source: Option<BoxedSourceError>,
}

/// Human diagnostics derived from an error source chain.
///
/// Boundary logs and retained query diagnostics should keep machine-readable
/// fields (`code`, `kind`, `status`, `operation`, facts) separate from human
/// root-cause text. This formatter walks `std::error::Error::source()` and
/// renders project errors by their human message, not by their structured
/// `Display` representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorSourceDiagnostics {
    chain: Option<String>,
    root_cause: Option<String>,
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
        assert!(!operation.is_empty(), "datafusion-nexus error operation must not be empty");
        assert!(!message.is_empty(), "datafusion-nexus error message must not be empty");
        Self { code, operation, message, facts: Facts::new(), source: None }
    }

    /// Adds one machine-readable fact.
    ///
    /// `key` must be a non-empty static string and should be treated as part of
    /// the diagnostic contract once exposed. `value` is stringified with
    /// [`ToString`]. Duplicate keys panic.
    #[must_use]
    pub fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.facts.insert(key, value.to_string());
        self
    }

    /// Attaches the source error, replacing any source already present.
    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        self.source = Some(Box::new(source));
        self
    }

    /// Attaches an already boxed source error, replacing any source already present.
    #[must_use]
    pub fn with_boxed_source(mut self, source: Box<dyn StdError + Send + Sync + 'static>) -> Self {
        self.source = Some(source);
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
        let error = Self::new(code, operation, message)
            .with_fact("cudf_code", source.code())
            .with_fact("cudf_kind", source.kind())
            .with_fact("cudf_status", source.status())
            .with_fact("cudf_operation", source.operation());
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
    pub fn from_native(
        operation: &'static str,
        message: impl Into<String>,
        source: nexus_query_engine::error::QueryError,
    ) -> Self {
        let code = code_for_native_error(&source);
        let mut error = Self::new(code, operation, message)
            .with_fact("native_code", source.code())
            .with_fact("native_kind", source.kind())
            .with_fact("native_status", source.status())
            .with_fact("native_operation", source.operation())
            .with_fact("native_message", source.message());
        let mut collision_keys = Vec::new();
        for (key, value) in source.facts().iter() {
            if key == NATIVE_FACT_COLLISION_KEYS || error.facts.get(key).is_some() {
                collision_keys.push(key);
                continue;
            }
            error.facts.insert(key, value.to_string());
        }
        if !collision_keys.is_empty() {
            error.facts.insert(NATIVE_FACT_COLLISION_KEYS, collision_keys.join(","));
        }
        error.with_source(source)
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "code={} kind={} status={} operation={} message={}",
            self.code(),
            self.kind(),
            self.status(),
            self.operation(),
            quote(&self.message),
        )?;

        if self.facts.values.is_empty() {
            f.write_str(" facts={}")?;
        } else {
            f.write_str(" facts={")?;
            for (index, (key, value)) in self.facts.iter().enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{key}={}", quote(value))?;
            }
            f.write_str("}")?;
        }

        if let Some(source) = &self.source {
            write!(f, " source={}", quote(&source.to_string()))?;
        }

        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("code", &self.code)
            .field("kind", &self.kind())
            .field("status", &self.status())
            .field("operation", &self.operation)
            .field("message", &self.message)
            .field("facts", &self.facts)
            .field("source", &self.source.as_ref().map(|source| source.to_string()))
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

// cuDF classification priority is resource exhaustion first, then retry status
// (`Temporary`/`Persistent`), then permanent failures by cuDF kind. Notable
// permanent mappings: `NotFound` is treated as invalid input,
// `ExecutionContext` is treated as an invariant violation, and unknown kinds
// fall back to dependency failure.
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

fn code_for_cudf_error(error: &cudf::Error) -> ErrorCode {
    if error.kind() == cudf::ErrorKind::ResourceExhausted {
        return ErrorCode::CudfResourceExhausted;
    }

    match error.status() {
        cudf::ErrorStatus::Temporary => return ErrorCode::CudfTemporaryFailure,
        cudf::ErrorStatus::Persistent => return ErrorCode::CudfPersistentFailure,
        cudf::ErrorStatus::Permanent => {},
    }

    match error.kind() {
        cudf::ErrorKind::InvalidInput | cudf::ErrorKind::NotFound => ErrorCode::CudfInvalidInput,
        cudf::ErrorKind::Unsupported => ErrorCode::CudfUnsupported,
        cudf::ErrorKind::DependencyFailure => ErrorCode::CudfDependencyFailure,
        cudf::ErrorKind::InvariantViolation | cudf::ErrorKind::ExecutionContext => ErrorCode::CudfInvariantViolation,
        _ => ErrorCode::CudfDependencyFailure,
    }
}

// Native-to-adapter mapping is intentionally lossy at the top-level code:
// native `UnsupportedPlanShape` becomes adapter `UnsupportedOperator` because
// the adapter has no shape-specific code. The exact native code remains
// available in the `native_code` fact attached by `Error::from_native`.
fn code_for_native_error(error: &nexus_query_engine::error::QueryError) -> ErrorCode {
    match error.code() {
        nexus_query_engine::error::QueryErrorCode::Cancelled => ErrorCode::Cancelled,
        nexus_query_engine::error::QueryErrorCode::UnsupportedPlanShape => ErrorCode::UnsupportedOperator,
        nexus_query_engine::error::QueryErrorCode::UnsupportedOperator => ErrorCode::UnsupportedOperator,
        nexus_query_engine::error::QueryErrorCode::UnsupportedExpression => ErrorCode::UnsupportedExpression,
        nexus_query_engine::error::QueryErrorCode::UnsupportedType => ErrorCode::UnsupportedType,
        nexus_query_engine::error::QueryErrorCode::InvalidPlan => ErrorCode::InvalidExecutionPlan,
        nexus_query_engine::error::QueryErrorCode::CudfInvalidInput => ErrorCode::CudfInvalidInput,
        nexus_query_engine::error::QueryErrorCode::CudfUnsupported => ErrorCode::CudfUnsupported,
        nexus_query_engine::error::QueryErrorCode::CudfResourceExhausted => ErrorCode::CudfResourceExhausted,
        nexus_query_engine::error::QueryErrorCode::CudfTemporaryFailure => ErrorCode::CudfTemporaryFailure,
        nexus_query_engine::error::QueryErrorCode::CudfPersistentFailure => ErrorCode::CudfPersistentFailure,
        nexus_query_engine::error::QueryErrorCode::CudfDependencyFailure => ErrorCode::CudfDependencyFailure,
        nexus_query_engine::error::QueryErrorCode::SourceAccessDenied => ErrorCode::SourceAccessDenied,
        nexus_query_engine::error::QueryErrorCode::SourceUnauthenticated => ErrorCode::SourceUnauthenticated,
        nexus_query_engine::error::QueryErrorCode::SourceObjectNotFound => ErrorCode::SourceObjectNotFound,
        nexus_query_engine::error::QueryErrorCode::SourceTemporaryFailure => ErrorCode::SourceTemporaryFailure,
        nexus_query_engine::error::QueryErrorCode::CudfInvariantViolation => ErrorCode::CudfInvariantViolation,
        nexus_query_engine::error::QueryErrorCode::MemoryReservationFailed => ErrorCode::MemoryReservationFailed,
        nexus_query_engine::error::QueryErrorCode::ServiceOverloaded => ErrorCode::ServiceOverloaded,
        nexus_query_engine::error::QueryErrorCode::Internal => ErrorCode::Internal,
        _ => ErrorCode::Internal,
    }
}

fn quote(value: &str) -> String {
    if value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/' | '(' | ')')) {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}

fn diagnostic_error_message(error: &(dyn StdError + 'static)) -> String {
    if let Some(error) = error.downcast_ref::<Error>() {
        return normalize_diagnostic_message(error.message());
    }
    if let Some(error) = error.downcast_ref::<nexus_query_engine::error::QueryError>() {
        return normalize_diagnostic_message(error.message());
    }
    if let Some(error) = error.downcast_ref::<DataFusionError>() {
        return datafusion_current_error_message(error);
    }
    normalize_diagnostic_message(&error.to_string())
}

fn datafusion_current_error_message(error: &DataFusionError) -> String {
    match error {
        DataFusionError::Context(message, _) => normalize_diagnostic_message(message),
        DataFusionError::Diagnostic(diagnostic, _) => normalize_diagnostic_message(&diagnostic.message),
        DataFusionError::External(source) => diagnostic_error_message(source.as_ref()),
        DataFusionError::Collection(errors) => match errors.first() {
            Some(first) => {
                let first = datafusion_current_error_message(first);
                normalize_diagnostic_message(&format!("DataFusion error collection ({} errors): {first}", errors.len()))
            },
            None => "DataFusion error collection (0 errors)".to_string(),
        },
        DataFusionError::Shared(source) => datafusion_current_error_message(source.as_ref()),
        _ => normalize_diagnostic_message(&error.message()),
    }
}

fn normalize_diagnostic_message(message: &str) -> String {
    message.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
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
}
