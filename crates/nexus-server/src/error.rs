use std::{error::Error as StdError, fmt, sync::Arc};

use datafusion::common::{DataFusionError, SchemaError};
use tonic::{Status, metadata::MetadataValue};

use crate::{
    adapter_error::{
        DataFusionErrorExt, Error, ErrorCode, ErrorKind, ErrorLogDiagnostics, ErrorSourceDiagnostics, ErrorStatus,
        Facts, datafusion_error_human_message,
    },
    observability::CapturedError,
};

/// Flight SQL response metadata key containing the stable adapter error code.
pub const ERROR_CODE_METADATA_KEY: &str = "x-datafusion-nexus-error-code";
/// Flight SQL response metadata key containing the stable adapter error kind.
pub const ERROR_KIND_METADATA_KEY: &str = "x-datafusion-nexus-error-kind";
/// Flight SQL response metadata key containing the retry/status classification.
pub const ERROR_STATUS_METADATA_KEY: &str = "x-datafusion-nexus-error-status";
/// Flight SQL response metadata key containing the stable adapter operation id.
pub const ERROR_OPERATION_METADATA_KEY: &str = "x-datafusion-nexus-error-operation";
/// Flight SQL response metadata key containing the first unsupported column name, when known.
pub const UNSUPPORTED_COLUMN_METADATA_KEY: &str = "x-datafusion-nexus-unsupported-column";
/// Flight SQL response metadata key containing the first unsupported column Arrow type, when known.
pub const UNSUPPORTED_COLUMN_TYPE_METADATA_KEY: &str = "x-datafusion-nexus-unsupported-column-type";
/// Flight SQL response metadata key containing the first unsupported column role, when known.
pub const UNSUPPORTED_COLUMN_ROLE_METADATA_KEY: &str = "x-datafusion-nexus-unsupported-column-role";
/// Flight SQL response metadata key containing the graph predicate role, when known.
pub const PREDICATE_ROLE_METADATA_KEY: &str = "x-datafusion-nexus-predicate-role";
/// Flight SQL response metadata key containing the graph validation reason, when known.
pub const REASON_METADATA_KEY: &str = "x-datafusion-nexus-reason";
/// Flight SQL response metadata key containing the required graph option, when known.
pub const REQUIRED_OPTION_METADATA_KEY: &str = "x-datafusion-nexus-required-option";
/// Flight SQL response metadata key containing the native diagnostic message, when known.
pub const NATIVE_MESSAGE_METADATA_KEY: &str = "x-datafusion-nexus-native-message";
const DATAFUSION_ERROR_FALLBACK_OPERATION: &str = "datafusion_nexus.server.datafusion_error";

pub type Result<T> = std::result::Result<T, ServerError>;

#[derive(Debug)]
pub struct ServerError {
    inner: Error,
}

impl ServerError {
    pub fn invalid_config(field: &'static str, reason: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.server.config",
            format!("invalid server configuration: {field} {}", reason.into()),
        )
        .with_fact("config_field", field)
    }

    pub fn invalid_config_with_source(
        field: &'static str,
        reason: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::invalid_config(field, reason).with_source(source)
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.server.config",
            format!("invalid server configuration: {}", message.into()),
        )
    }

    pub fn configuration_with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.server.config",
            format!("invalid server configuration: {}", message.into()),
        )
        .with_source(source)
    }

    pub fn startup(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, "datafusion_nexus.server.startup", message)
    }

    pub(crate) fn shutdown(reason: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, "datafusion_nexus.server.shutdown", message).with_fact("reason", reason)
    }

    pub(crate) fn shutdown_with_source(
        reason: &'static str,
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::shutdown(reason, message).with_source(source)
    }

    pub(crate) fn shutdown_from_shared_adapter_error(
        reason: &'static str,
        message: impl Into<String>,
        source: Arc<Error>,
    ) -> Self {
        Self {
            inner: Error::new(source.code(), "datafusion_nexus.server.shutdown", message)
                .with_fact("reason", reason)
                .with_source(source),
        }
    }

    pub(crate) fn builder_validation(reason: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidRequest, "datafusion_nexus.server.builder", message).with_fact("reason", reason)
    }

    pub(crate) fn builder_conflict(
        reason: &'static str,
        field: &'static str,
        config_value: impl ToString,
        backend_value: impl ToString,
    ) -> Self {
        Self::builder_validation(
            reason,
            format!(
                "server config field `{field}` conflicts with injected backend: config={} backend={}",
                config_value.to_string(),
                backend_value.to_string()
            ),
        )
        .with_fact("config_field", field)
        .with_fact("config_value", config_value)
        .with_fact("backend_value", backend_value)
    }

    pub fn session(message: impl Into<String>, source: impl StdError + Send + Sync + 'static) -> Self {
        let message = message.into();
        let source: Box<dyn StdError + Send + Sync> = Box::new(source);
        let source = match source.downcast::<Error>() {
            Ok(adapter_error) => return Self { inner: adapter_error_with_message(message, *adapter_error) },
            Err(source) => source,
        };
        let source = match source.downcast::<nexus_query_engine::error::QueryError>() {
            Ok(native_error) => {
                return Self {
                    inner: Error::from_native_for_internal_consumer(
                        "datafusion_nexus.server.session",
                        message,
                        *native_error,
                    ),
                };
            },
            Err(source) => source,
        };
        let source = match source.downcast::<DataFusionError>() {
            Ok(datafusion_error) => {
                return Self { inner: server_error_from_datafusion_error(message, *datafusion_error) };
            },
            Err(source) => source,
        };
        #[cfg(feature = "iceberg")]
        let source = match source.downcast::<nexus_iceberg::IcebergError>() {
            Ok(iceberg_error) => {
                return Self {
                    inner: crate::table_format::iceberg::root_error_from_iceberg(
                        *iceberg_error,
                        "datafusion_nexus.server.session",
                        message,
                    ),
                };
            },
            Err(source) => source,
        };
        Self {
            inner: Error::new(ErrorCode::Internal, "datafusion_nexus.server.session", message)
                .with_source(ServerSource(source)),
        }
    }

    pub fn transport(source: tonic::transport::Error) -> Self {
        Self::new(ErrorCode::ServerTransportFailure, "datafusion_nexus.server.transport", "Flight SQL transport failed")
            .with_source(source)
    }

    pub(crate) fn transport_at_addr(addr: std::net::SocketAddr, source: tonic::transport::Error) -> Self {
        Self::new(
            ErrorCode::ServerTransportFailure,
            "datafusion_nexus.server.transport",
            format!("Flight SQL transport failed while serving {addr}"),
        )
        .with_fact("bind_addr", addr)
        .with_source(source)
    }

    fn new(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        Self { inner: Error::new(code, operation, message) }
    }

    #[must_use]
    pub(crate) fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.inner = self.inner.with_fact(key, value);
        self
    }

    #[must_use]
    fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        self.inner = self.inner.with_source(source);
        self
    }

    pub fn code(&self) -> ErrorCode {
        self.inner.code()
    }

    pub fn kind(&self) -> ErrorKind {
        self.inner.kind()
    }

    pub fn status(&self) -> ErrorStatus {
        self.inner.status()
    }

    pub fn operation(&self) -> &'static str {
        self.inner.operation()
    }

    pub fn message(&self) -> &str {
        self.inner.message()
    }

    pub fn facts(&self) -> &Facts {
        self.inner.facts()
    }

    pub fn fact_value(&self, key: &str) -> Option<&str> {
        self.inner.fact_value(key)
    }

    pub fn as_adapter_error(&self) -> &Error {
        &self.inner
    }
}

fn adapter_error_with_message(message: String, source: Error) -> Error {
    let facts = source.facts().clone();
    Error::new(source.code(), source.operation(), message).with_preserved_facts(facts).with_source(source)
}

fn server_error_from_datafusion_error(message: String, source: DataFusionError) -> Error {
    if let Some((code, operation, facts)) = source
        .find_nexus_error()
        .map(|adapter_source| (adapter_source.code(), adapter_source.operation(), adapter_source.facts().clone()))
    {
        return Error::new(code, operation, message).with_preserved_facts(facts).with_source(source);
    }
    let code = classify_datafusion_error(&source);
    let datafusion_error = datafusion_error_variant(&source);
    Error::new(code, "datafusion_nexus.server.session", message)
        .with_log_safe_fact("datafusion_error", datafusion_error)
        .with_source(source)
}

#[derive(Debug)]
struct ServerSource(Box<dyn StdError + Send + Sync>);

impl fmt::Display for ServerSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl StdError for ServerSource {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.0.as_ref())
    }
}

#[derive(Debug)]
struct StatusSourceDiagnostics {
    source_chain: Option<String>,
    root_cause: Option<String>,
}

impl fmt::Display for StatusSourceDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.source_chain.as_deref().or(self.root_cause.as_deref()) {
            Some(message) => f.write_str(message),
            None => f.write_str("no source diagnostics"),
        }
    }
}

impl StdError for StatusSourceDiagnostics {}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "code={} kind={} status={} operation={} message={}",
            self.code(),
            self.kind(),
            self.status(),
            self.operation(),
            quote(self.message()),
        )?;
        let projection = self.facts().log_safe_projection();
        write!(f, " facts={{{}}}", projection.rendered().unwrap_or_default())?;
        if projection.truncated() {
            f.write_str(" error_facts_truncated=true")?;
        }
        Ok(())
    }
}

impl StdError for ServerError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.inner.source_error()
    }
}

fn quote(value: &str) -> String {
    if value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/' | '(' | ')')) {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}

pub fn datafusion_error_to_status(error: DataFusionError) -> Status {
    datafusion_error_to_status_for_operation(error, DATAFUSION_ERROR_FALLBACK_OPERATION)
}

pub fn datafusion_error_to_status_for_operation(error: DataFusionError, operation: &'static str) -> Status {
    if let Some(adapter_error) = error.find_nexus_error() {
        return adapter_error_to_status(adapter_error);
    }
    adapter_error_to_status(&adapter_error_from_datafusion_error(error, operation))
}

pub(super) fn datafusion_error_to_captured_error(error: &DataFusionError, operation: &'static str) -> CapturedError {
    if let Some(adapter_error) = error.find_nexus_error() {
        return CapturedError::from_adapter_error(adapter_error);
    }
    let code = classify_datafusion_error(error);
    let message = client_datafusion_error_message(error);
    let datafusion_error = datafusion_error_variant(error);
    let mut facts = Facts::new();
    facts.insert_log_safe("datafusion_error", datafusion_error.to_string());
    CapturedError::from_datafusion_error(error, code, operation, message, facts)
}

pub(crate) fn adapter_error_to_status(error: &Error) -> Status {
    let message = client_status_message(error);
    let diagnostics = ErrorLogDiagnostics::from_error(error);
    let log_messages = adapter_status_log_messages(error.message(), message.as_str(), &diagnostics);
    let log_facts = adapter_status_log_facts(error);
    macro_rules! log_status_conversion {
        ($level:expr) => {
            tracing::event!(
                $level,
                error_code = error.code().as_str(),
                error_kind = error.kind().as_str(),
                error_status = error.status().as_str(),
                error_operation = error.operation(),
                error_message = error.message(),
                error_cause = log_messages.cause,
                error_source_path = diagnostics.source_path(),
                error_diagnostics_truncated = log_messages.diagnostics_truncated,
                client_message = log_messages.client_message,
                error_facts = log_facts.rendered.as_deref(),
                error_facts_truncated = log_facts.truncated,
                "converted adapter error to Flight SQL status"
            )
        };
    }
    match adapter_status_log_level(error.kind()) {
        AdapterStatusLogLevel::Info => log_status_conversion!(tracing::Level::INFO),
        AdapterStatusLogLevel::Warn => log_status_conversion!(tracing::Level::WARN),
        AdapterStatusLogLevel::Error => log_status_conversion!(tracing::Level::ERROR),
    }
    let status = match error.kind() {
        ErrorKind::Cancelled => Status::cancelled(message),
        ErrorKind::NotFound => Status::not_found(message),
        ErrorKind::PermissionDenied => Status::permission_denied(message),
        ErrorKind::Unauthenticated => Status::unauthenticated(message),
        ErrorKind::InvalidInput | ErrorKind::Planning => Status::invalid_argument(message),
        ErrorKind::Unsupported => Status::unimplemented(message),
        ErrorKind::ResourceExhausted => Status::resource_exhausted(message),
        ErrorKind::Execution | ErrorKind::DependencyFailure | ErrorKind::InvariantViolation | ErrorKind::Internal => {
            Status::internal(message)
        },
        _ => Status::internal(message),
    };
    status_with_error_metadata(status, error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdapterStatusLogLevel {
    Info,
    Warn,
    Error,
}

fn adapter_status_log_level(kind: ErrorKind) -> AdapterStatusLogLevel {
    match kind {
        ErrorKind::Cancelled
        | ErrorKind::NotFound
        | ErrorKind::PermissionDenied
        | ErrorKind::Unauthenticated
        | ErrorKind::Planning
        | ErrorKind::InvalidInput
        | ErrorKind::Unsupported => AdapterStatusLogLevel::Info,
        ErrorKind::ResourceExhausted | ErrorKind::DependencyFailure => AdapterStatusLogLevel::Warn,
        ErrorKind::Execution | ErrorKind::InvariantViolation | ErrorKind::Internal => AdapterStatusLogLevel::Error,
        _ => AdapterStatusLogLevel::Error,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct AdapterStatusLogMessages<'a> {
    cause: Option<&'a str>,
    diagnostics_truncated: Option<bool>,
    client_message: Option<&'a str>,
}

#[derive(Debug, PartialEq, Eq)]
struct AdapterStatusLogFacts {
    rendered: Option<String>,
    truncated: Option<bool>,
}

fn adapter_status_log_facts(error: &Error) -> AdapterStatusLogFacts {
    let projection = error.facts().log_safe_projection();
    AdapterStatusLogFacts { rendered: projection.rendered(), truncated: projection.truncated().then_some(true) }
}

fn adapter_status_log_messages<'a>(
    error_message: &'a str,
    client_message: &'a str,
    diagnostics: &'a ErrorLogDiagnostics,
) -> AdapterStatusLogMessages<'a> {
    AdapterStatusLogMessages {
        cause: diagnostics.cause().filter(|cause| *cause != error_message),
        diagnostics_truncated: diagnostics.truncated().then_some(true),
        client_message: (client_message != error_message).then_some(client_message),
    }
}

pub(super) fn status_source_diagnostics(status: &Status) -> (Option<String>, Option<String>) {
    let Some(source) = StdError::source(status) else {
        return (None, None);
    };
    if let Some(diagnostics) = source.downcast_ref::<StatusSourceDiagnostics>() {
        return (diagnostics.source_chain.clone(), diagnostics.root_cause.clone());
    }
    let diagnostics = ErrorSourceDiagnostics::from_error(source);
    (diagnostics.chain().map(str::to_string), diagnostics.root_cause().map(str::to_string))
}

fn client_status_message(error: &Error) -> String {
    match error.code() {
        ErrorCode::SqlParse | ErrorCode::FieldNotFound | ErrorCode::ObjectNotFound | ErrorCode::DataFusionPlanning => {
            error.message().to_string()
        },
        ErrorCode::SourceAccessDenied => "source object storage access denied while processing the request".to_string(),
        ErrorCode::SourceUnauthenticated => {
            "source object storage credentials are missing or invalid while processing the request".to_string()
        },
        ErrorCode::SourceObjectNotFound => {
            "source object storage object was not found while processing the request".to_string()
        },
        ErrorCode::SourceTemporaryFailure => {
            "temporary source object storage failure while processing the request; retry later".to_string()
        },
        ErrorCode::ServiceOverloaded => "service overloaded; retry later".to_string(),
        _ if error.kind() == ErrorKind::ResourceExhausted => {
            format!("resource exhausted: {}. Retry later or reduce query resource usage.", error.message())
        },
        _ if matches!(
            error.kind(),
            ErrorKind::Execution | ErrorKind::DependencyFailure | ErrorKind::InvariantViolation | ErrorKind::Internal
        ) =>
        {
            "internal server error while processing the request".to_string()
        },
        _ => error.message().to_string(),
    }
}

fn status_with_error_metadata(mut status: Status, error: &Error) -> Status {
    insert_metadata(status.metadata_mut(), ERROR_CODE_METADATA_KEY, error.code().as_str());
    insert_metadata(status.metadata_mut(), ERROR_KIND_METADATA_KEY, error.kind().as_str());
    insert_metadata(status.metadata_mut(), ERROR_STATUS_METADATA_KEY, error.status().as_str());
    insert_metadata(status.metadata_mut(), ERROR_OPERATION_METADATA_KEY, error.operation());
    let diagnostics = ErrorSourceDiagnostics::from_source(error.source_error());
    if diagnostics.chain().is_some() || diagnostics.root_cause().is_some() {
        status.set_source(Arc::new(StatusSourceDiagnostics {
            source_chain: diagnostics.chain().map(str::to_string),
            root_cause: diagnostics.root_cause().map(str::to_string),
        }));
    }
    insert_fact_metadata(status.metadata_mut(), UNSUPPORTED_COLUMN_METADATA_KEY, error, "unsupported_column");
    insert_fact_metadata(status.metadata_mut(), UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, error, "unsupported_column_type");
    insert_fact_metadata(status.metadata_mut(), UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, error, "unsupported_column_role");
    insert_fact_metadata(status.metadata_mut(), PREDICATE_ROLE_METADATA_KEY, error, "predicate_role");
    insert_fact_metadata(status.metadata_mut(), REASON_METADATA_KEY, error, "reason");
    insert_fact_metadata(status.metadata_mut(), REQUIRED_OPTION_METADATA_KEY, error, "required_option");
    insert_fact_metadata(status.metadata_mut(), NATIVE_MESSAGE_METADATA_KEY, error, "native_message");
    status
}

fn insert_metadata(metadata: &mut tonic::metadata::MetadataMap, key: &'static str, value: &'static str) {
    if let Ok(value) = MetadataValue::try_from(value) {
        metadata.insert(key, value);
    }
}

fn insert_fact_metadata(
    metadata: &mut tonic::metadata::MetadataMap,
    metadata_key: &'static str,
    error: &Error,
    fact_key: &str,
) {
    if let Some(value) = error.fact_value(fact_key)
        && let Ok(value) = MetadataValue::try_from(value)
    {
        metadata.insert(metadata_key, value);
    }
}

fn adapter_error_from_datafusion_error(error: DataFusionError, operation: &'static str) -> Error {
    let code = classify_datafusion_error(&error);
    let message = client_datafusion_error_message(&error);
    let datafusion_error = datafusion_error_variant(&error);
    Error::new(code, operation, message).with_log_safe_fact("datafusion_error", datafusion_error).with_source(error)
}

fn client_datafusion_error_message(error: &DataFusionError) -> String {
    let effective = effective_datafusion_error(error);
    let human_message = datafusion_error_human_message(effective);
    match effective {
        DataFusionError::SQL(_, _) => format!("invalid SQL statement: {human_message}"),
        _ => human_message,
    }
}

fn effective_datafusion_error(error: &DataFusionError) -> &DataFusionError {
    match error {
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            effective_datafusion_error(source)
        },
        DataFusionError::Collection(errors) => errors.first().map(effective_datafusion_error).unwrap_or(error),
        DataFusionError::Shared(source) => effective_datafusion_error(source),
        _ => error,
    }
}

fn classify_datafusion_error(error: &DataFusionError) -> ErrorCode {
    match error {
        DataFusionError::SQL(_, _) => ErrorCode::SqlParse,
        DataFusionError::Plan(_) => ErrorCode::DataFusionPlanning,
        DataFusionError::SchemaError(schema, _) => match schema.as_ref() {
            SchemaError::FieldNotFound { .. } => ErrorCode::FieldNotFound,
            SchemaError::AmbiguousReference { .. }
            | SchemaError::DuplicateQualifiedField { .. }
            | SchemaError::DuplicateUnqualifiedField { .. } => ErrorCode::DataFusionPlanning,
        },
        DataFusionError::NotImplemented(_) => ErrorCode::UnsupportedOperator,
        DataFusionError::Configuration(_) => ErrorCode::DataFusionPlanning,
        // Non-not-found execution errors may include reader internals, so keep
        // them behind the generic `Internal` client surface.
        DataFusionError::Execution(_) => ErrorCode::Internal,
        DataFusionError::ResourcesExhausted(_) => ErrorCode::MemoryReservationFailed,
        DataFusionError::Context(_, source) => classify_datafusion_error(source),
        // DataFusion diagnostics add spans, notes, and hints around several
        // planning failures. Their wrapped `Plan` variant has no structured
        // relation-miss discriminator, so it cannot truthfully be projected
        // as `ObjectNotFound` without parsing display text. Preserve the
        // generic planning classification instead.
        DataFusionError::Diagnostic(_, source) => classify_datafusion_error(source),
        DataFusionError::Collection(errors) => {
            errors.first().map(classify_datafusion_error).unwrap_or(ErrorCode::Internal)
        },
        DataFusionError::Shared(source) => classify_datafusion_error(source),
        DataFusionError::Internal(_)
        | DataFusionError::ArrowError(_, _)
        | DataFusionError::IoError(_)
        | DataFusionError::External(_)
        | DataFusionError::ExecutionJoin(_)
        | DataFusionError::Ffi(_) => ErrorCode::Internal,
        _ => ErrorCode::Internal,
    }
}

fn datafusion_error_variant(error: &DataFusionError) -> &'static str {
    match error {
        DataFusionError::ArrowError(_, _) => "arrow",
        DataFusionError::IoError(_) => "io",
        DataFusionError::SQL(_, _) => "sql",
        DataFusionError::NotImplemented(_) => "not_implemented",
        DataFusionError::Internal(_) => "internal",
        DataFusionError::Plan(_) => "plan",
        DataFusionError::Configuration(_) => "configuration",
        DataFusionError::SchemaError(_, _) => "schema",
        DataFusionError::Execution(_) => "execution",
        DataFusionError::ExecutionJoin(_) => "execution_join",
        DataFusionError::ResourcesExhausted(_) => "resources_exhausted",
        DataFusionError::External(_) => "external",
        DataFusionError::Context(_, _) => "context",
        DataFusionError::Diagnostic(_, _) => "diagnostic",
        DataFusionError::Collection(_) => "collection",
        DataFusionError::Shared(_) => "shared",
        DataFusionError::Ffi(_) => "ffi",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, fmt};

    use datafusion::{
        common::{Column, Diagnostic, SchemaError, TableReference},
        sql::sqlparser::parser::ParserError,
    };
    use tonic::Code;

    use super::*;

    const TEST_OPERATION: &str = "datafusion_nexus.server.flight.test.create_logical_plan";

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
    fn sql_parser_message_uses_display_context_and_deduplicated_source_diagnostics() {
        let parser = ParserError::ParserError("Expected: an SQL statement, found: quit at Line: 1, Column: 1".into());
        let parser_display = parser.to_string();
        let datafusion_error = DataFusionError::SQL(Box::new(parser), None);

        let adapter = adapter_error_from_datafusion_error(datafusion_error, TEST_OPERATION);

        assert_eq!(adapter.message(), format!("invalid SQL statement: {parser_display}"));
        assert!(!adapter.message().contains("ParserError("), "{}", adapter.message());
        assert!(matches!(
            adapter.source_error().and_then(|source| source.downcast_ref::<DataFusionError>()),
            Some(DataFusionError::SQL(_, _))
        ));

        let diagnostics = ErrorSourceDiagnostics::from_source(adapter.source_error());
        assert_eq!(diagnostics.chain(), Some(parser_display.as_str()));
        assert_eq!(diagnostics.root_cause(), Some(parser_display.as_str()));
        assert_eq!(diagnostics.root_cause_when_distinct(), None);
        assert_eq!(diagnostics.chain().unwrap().matches(&parser_display).count(), 1);
        assert!(!diagnostics.chain().unwrap().contains("ParserError("));

        let log_diagnostics = ErrorLogDiagnostics::from_error(&adapter);
        let log_messages = adapter_status_log_messages(adapter.message(), adapter.message(), &log_diagnostics);
        assert_eq!(log_messages.cause, Some(parser_display.as_str()));
        assert_eq!(log_messages.diagnostics_truncated, None);
        assert_eq!(log_messages.client_message, None);

        let status = adapter_error_to_status(&adapter);
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.message(), format!("invalid SQL statement: {parser_display}"));
        assert!(!status.message().contains("ParserError("));
        assert_error_metadata(&status, "sql_parse", "invalid_input", "permanent");
        assert_eq!(status_source_diagnostics(&status), (Some(parser_display.clone()), Some(parser_display.clone())));
    }

    #[test]
    fn boundary_log_omits_source_and_client_messages_equal_to_error_message() {
        let error = Error::new(ErrorCode::InvalidRequest, TEST_OPERATION, "same human message")
            .with_source(TestSourceError::leaf("same human message"));
        let diagnostics = ErrorLogDiagnostics::from_error(&error);

        assert_eq!(
            adapter_status_log_messages(error.message(), error.message(), &diagnostics),
            AdapterStatusLogMessages { cause: None, diagnostics_truncated: None, client_message: None }
        );
    }

    #[test]
    fn boundary_log_uses_safe_fact_projection_and_truncation_marker() {
        let error = Error::new(ErrorCode::Internal, TEST_OPERATION, "fact projection")
            .with_log_safe_fact("classification", "native\nhttps://user:secret@example.test/private")
            .with_log_safe_fact("long", "x".repeat(300))
            .with_fact("literal", "customer-secret")
            .with_fact("path", "/private/customer.parquet");

        let log_facts = adapter_status_log_facts(&error);
        let rendered = log_facts.rendered.expect("safe facts should be logged");
        assert!(rendered.contains("classification=native <redacted-uri>"), "{rendered}");
        assert!(rendered.contains("long="), "{rendered}");
        assert!(!rendered.contains("customer-secret"), "{rendered}");
        assert!(!rendered.contains("/private/customer.parquet"), "{rendered}");
        assert_eq!(log_facts.truncated, Some(true));
    }

    #[test]
    fn retained_adapter_facts_use_the_same_safe_projection() {
        let error = Error::new(ErrorCode::Internal, TEST_OPERATION, "retained fact projection")
            .with_log_safe_fact("classification", "runtime\nerror")
            .with_log_safe_fact("long", "x".repeat(300))
            .with_fact("literal", "customer-secret")
            .with_fact("value", "private-value")
            .with_fact("path", "/private/customer.parquet")
            .with_fact("config_value", "private-config")
            .with_fact("native_message", "private-native-message")
            .with_fact("cudf_message", "private-cudf-message");

        let captured = CapturedError::from_adapter_error(&error);

        assert!(captured.facts.contains(&("classification".to_string(), "runtime error".to_string())));
        assert!(captured.facts.contains(&("error_facts_truncated".to_string(), "true".to_string())));
        for hidden_key in ["literal", "value", "path", "config_value", "native_message", "cudf_message"] {
            assert!(captured.facts.iter().all(|(key, _)| key != hidden_key), "retained unsafe fact {hidden_key}");
        }
    }

    #[test]
    fn server_error_display_uses_the_same_safe_fact_projection() {
        let error = ServerError {
            inner: Error::new(ErrorCode::Internal, TEST_OPERATION, "server display projection")
                .with_log_safe_fact("classification", "runtime\nhttps://user:secret@example.test/private")
                .with_log_safe_fact("long", "x".repeat(300))
                .with_fact("literal", "customer-secret")
                .with_fact("config_value", "private-config"),
        };

        let rendered = error.to_string();
        assert!(rendered.contains("classification=runtime <redacted-uri>"), "{rendered}");
        assert!(rendered.contains("error_facts_truncated=true"), "{rendered}");
        assert!(!rendered.contains("customer-secret"), "{rendered}");
        assert!(!rendered.contains("private-config"), "{rendered}");
        assert!(!rendered.contains("user:secret"), "{rendered}");
    }

    #[test]
    fn credential_key_variants_are_redacted_across_all_projection_sinks() {
        let make_error = || {
            Error::new(ErrorCode::Internal, TEST_OPERATION, "credential projection")
                .with_log_safe_fact("client", "CLIENT-SECRET=client-value-secret")
                .with_log_safe_fact("refresh", "Refresh.Token:refresh-value-secret")
                .with_log_safe_fact("aws", "AWS_SECRET_ACCESS_KEY=aws-value-secret")
        };

        let facts_debug = format!("{:?}", make_error().facts());
        let boundary = adapter_status_log_facts(&make_error()).rendered.expect("safe facts should render");
        let retained = format!("{:?}", CapturedError::from_adapter_error(&make_error()).facts);
        let display = ServerError { inner: make_error() }.to_string();
        for sink in [&facts_debug, &boundary, &retained, &display] {
            assert_eq!(sink.matches("<redacted-credential>").count(), 3, "{sink}");
            for secret in ["client-value-secret", "refresh-value-secret", "aws-value-secret"] {
                assert!(!sink.contains(secret), "{sink}");
            }
        }
    }

    #[test]
    fn server_error_debug_omits_native_and_cudf_source_payloads_but_keeps_typed_access() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::InvalidPlan,
            "nexus_query_engine.test.server_debug_source",
            "server-native-source-message-secret",
        )
        .with_fact("literal", "server-native-source-fact-secret");
        let native_error = ServerError {
            inner: Error::from_native_for_internal_consumer(
                "datafusion_nexus.test.server_debug_source",
                "native server wrapper",
                native,
            ),
        };
        let native_source = native_error
            .source()
            .and_then(|source| source.downcast_ref::<nexus_query_engine::error::QueryError>())
            .expect("native source must remain typed");
        assert_eq!(native_source.fact_value("literal"), Some("server-native-source-fact-secret"));
        let native_debug = format!("{native_error:?}");
        assert!(native_debug.contains("<source omitted>"), "{native_debug}");
        assert!(!native_debug.contains("server-native-source-message-secret"), "{native_debug}");
        assert!(!native_debug.contains("server-native-source-fact-secret"), "{native_debug}");

        let cudf = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::ForeignInvalidArgument,
            "cudf.test.server_debug_source",
            "server-cudf-source-detail-secret refresh_token=cudf-refresh-secret",
        );
        let cudf_error = ServerError {
            inner: Error::from_cudf("datafusion_nexus.test.server_debug_source", "cudf server wrapper", cudf),
        };
        let cudf_source = cudf_error
            .source()
            .and_then(|source| source.downcast_ref::<cudf::Error>())
            .expect("cuDF source must remain typed");
        assert_eq!(cudf_source.message(), "server-cudf-source-detail-secret refresh_token=cudf-refresh-secret");
        let cudf_debug = format!("{cudf_error:?}");
        assert!(cudf_debug.contains("<source omitted>"), "{cudf_debug}");
        assert!(!cudf_debug.contains("server-cudf-source-detail-secret"), "{cudf_debug}");
        assert!(!cudf_debug.contains("cudf-refresh-secret"), "{cudf_debug}");
    }

    #[test]
    fn log_safe_facts_do_not_widen_flight_metadata() {
        let status = adapter_error_to_status(
            &Error::new(ErrorCode::InvalidRequest, TEST_OPERATION, "invalid request")
                .with_log_safe_fact("log_classification", "request_shape")
                .with_fact("unsupported_column", "countDetail"),
        );

        assert_metadata_value(&status, UNSUPPORTED_COLUMN_METADATA_KEY, "countDetail");
        assert!(status.metadata().get("x-datafusion-nexus-log-classification").is_none());
        assert!(status.metadata().get("log_classification").is_none());
    }

    #[test]
    fn boundary_log_severity_is_derived_from_error_kind() {
        let cases = [
            (ErrorKind::Cancelled, AdapterStatusLogLevel::Info),
            (ErrorKind::NotFound, AdapterStatusLogLevel::Info),
            (ErrorKind::PermissionDenied, AdapterStatusLogLevel::Info),
            (ErrorKind::Unauthenticated, AdapterStatusLogLevel::Info),
            (ErrorKind::Planning, AdapterStatusLogLevel::Info),
            (ErrorKind::InvalidInput, AdapterStatusLogLevel::Info),
            (ErrorKind::Unsupported, AdapterStatusLogLevel::Info),
            (ErrorKind::ResourceExhausted, AdapterStatusLogLevel::Warn),
            (ErrorKind::DependencyFailure, AdapterStatusLogLevel::Warn),
            (ErrorKind::Execution, AdapterStatusLogLevel::Error),
            (ErrorKind::InvariantViolation, AdapterStatusLogLevel::Error),
            (ErrorKind::Internal, AdapterStatusLogLevel::Error),
        ];

        for (kind, expected) in cases {
            assert_eq!(adapter_status_log_level(kind), expected, "unexpected boundary log level for {kind}");
        }
    }

    #[test]
    fn adapter_kind_to_flight_code_contract_matches_consumer_documentation() {
        let cases = [
            (ErrorCode::Cancelled, Code::Cancelled),
            (ErrorCode::ObjectNotFound, Code::NotFound),
            (ErrorCode::SourceAccessDenied, Code::PermissionDenied),
            (ErrorCode::SourceUnauthenticated, Code::Unauthenticated),
            (ErrorCode::InvalidRequest, Code::InvalidArgument),
            (ErrorCode::DataFusionPlanning, Code::InvalidArgument),
            (ErrorCode::UnsupportedOperator, Code::Unimplemented),
            (ErrorCode::ServiceOverloaded, Code::ResourceExhausted),
            (ErrorCode::SizeOverflow, Code::ResourceExhausted),
            (ErrorCode::HostMaterializationLimitExceeded, Code::ResourceExhausted),
            (ErrorCode::HostStagingLimitExceeded, Code::ResourceExhausted),
            (ErrorCode::CudfDependencyFailure, Code::Internal),
            (ErrorCode::CudfInvariantViolation, Code::Internal),
            (ErrorCode::Internal, Code::Internal),
        ];

        for (error_code, expected_flight_code) in cases {
            let error = Error::new(error_code, TEST_OPERATION, "Flight mapping contract");
            assert_eq!(adapter_error_to_status(&error).code(), expected_flight_code, "{error_code}");
        }
    }

    #[test]
    fn field_not_found_becomes_invalid_input_with_client_message() {
        let error = DataFusionError::SchemaError(
            Box::new(SchemaError::FieldNotFound {
                field: Box::new(Column::from_name("missing")),
                valid_fields: vec![Column::from_name("present")],
            }),
            Box::new(None),
        );

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION);

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "field_not_found", "invalid_input", "permanent");
        assert!(status.message().contains("No field named missing"));
        assert!(status.message().contains("Valid fields are present"));
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn ambiguous_schema_reference_remains_planning() {
        let error = DataFusionError::SchemaError(
            Box::new(SchemaError::AmbiguousReference { field: Box::new(Column::from_name("id")) }),
            Box::new(None),
        );

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION);

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn datafusion_plan_classification_is_variant_based_not_message_based() {
        for message in [
            "optimizer rule was not found in registry",
            "table 'does_not_exist' not found",
            "table function 'cugraph_bfs' not found",
        ] {
            let status =
                datafusion_error_to_status_for_operation(DataFusionError::Plan(message.to_string()), TEST_OPERATION);

            assert_eq!(status.code(), Code::InvalidArgument, "{message}");
            assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
            assert_eq!(status.message(), message);
            assert!(!status.message().contains("NEXUS_SERVER_CUGRAPH_ENABLED"));
        }
    }

    #[test]
    fn diagnostic_wrapped_plan_errors_remain_generic_planning_without_message_parsing() {
        for message in [
            "table p39_missing_table was not found",
            "function p39_missing_function does not exist",
            "column p39_value must appear in GROUP BY",
        ] {
            let error = DataFusionError::Diagnostic(
                Box::new(Diagnostic::new_error("synthetic planning diagnostic", None)),
                Box::new(DataFusionError::Plan(message.to_string())),
            );
            let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION);

            assert_eq!(status.code(), Code::InvalidArgument, "{message}");
            assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
        }
    }

    #[tokio::test]
    async fn session_context_missing_table_remains_generic_planning_without_a_typed_relation_error() {
        let context = datafusion::prelude::SessionContext::new();
        let error = context
            .sql("SELECT * FROM p39_missing_table")
            .await
            .expect_err("an unregistered relation must fail planning");

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION);

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
    }

    #[test]
    fn datafusion_execution_classification_is_variant_based_not_message_based() {
        for message in [
            "Table 'does_not_exist' doesn't exist.",
            "Parquet error reading /var/lib/nexus/secret/customers.parquet: invalid magic bytes",
        ] {
            let status = datafusion_error_to_status_for_operation(
                DataFusionError::Execution(message.to_string()),
                TEST_OPERATION,
            );

            assert_eq!(status.code(), Code::Internal, "{message}");
            assert_error_metadata(&status, "internal", "internal", "permanent");
            assert_eq!(status.message(), "internal server error while processing the request");
            assert!(!status.message().contains("/var/lib/nexus"));
        }
    }

    #[test]
    fn generic_execution_error_is_genericized_and_does_not_leak_internal_detail() {
        // Execution detail can include local paths or object-store URLs; do not
        // echo it to clients.
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Execution(
                "Parquet error reading /var/lib/nexus/secret/customers.parquet: invalid magic bytes".to_string(),
            ),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal server error while processing the request");
        assert!(!status.message().contains("/var/lib/nexus"));
        assert!(!status.message().contains("customers.parquet"));
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn duplicate_schema_reference_remains_planning() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::SchemaError(
                Box::new(SchemaError::DuplicateQualifiedField {
                    qualifier: Box::new(TableReference::bare("t")),
                    name: "id".to_string(),
                }),
                Box::new(None),
            ),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
    }

    #[test]
    fn wrapped_datafusion_error_uses_inner_client_message() {
        let error = DataFusionError::Context(
            "server wrapper context".to_string(),
            Box::new(DataFusionError::SchemaError(
                Box::new(SchemaError::FieldNotFound {
                    field: Box::new(Column::from_name("missing")),
                    valid_fields: vec![Column::from_name("present")],
                }),
                Box::new(None),
            )),
        );

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION);

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "field_not_found", "invalid_input", "permanent");
        assert!(status.message().contains("No field named missing"));
        assert!(!status.message().contains("server wrapper context"));
        assert!(!status.message().contains("caused by"));
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_service_overloaded_projects_resource_exhausted_status() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::ServiceOverloaded,
                "datafusion_nexus.native.query_service.admit",
                "native query service queue is full",
            )
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_error_metadata(&status, "service_overloaded", "resource_exhausted", "temporary");
        assert_eq!(status.message(), "service overloaded; retry later");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn permanent_host_materialization_limit_projects_flight_resource_exhausted_without_retryability() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::HostMaterializationLimitExceeded,
                "datafusion_nexus.native_exec.host_batch_cursor",
                "one Arrow row exceeds the configured host materialization ceiling",
            )
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_error_metadata(&status, "host_materialization_limit_exceeded", "resource_exhausted", "permanent");
        assert!(status.message().contains("resource exhausted"));
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_server_transport_failure_projects_internal_status() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::ServerTransportFailure,
                "datafusion_nexus.server.transport",
                "Flight SQL transport failed",
            )
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_error_metadata(&status, "server_transport_failure", "dependency_failure", "permanent");
        assert_eq!(status.message(), "internal server error while processing the request");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn captured_adapter_error_keeps_structured_fields_separate_from_source_chain() {
        let error =
            Error::new(ErrorCode::CudfDependencyFailure, "datafusion_nexus.server.test", "adapter wrapper failed")
                .with_source(TestSourceError::with_source(
                    "dependency wrapper",
                    TestSourceError::leaf("root credential expired"),
                ));

        let captured = CapturedError::from_adapter_error(&error);

        assert_eq!(captured.message, "adapter wrapper failed");
        assert_eq!(captured.diagnostic, "dependency wrapper: root credential expired");
        assert_eq!(captured.source_chain.as_deref(), Some("dependency wrapper: root credential expired"));
        assert_eq!(captured.root_cause.as_deref(), Some("root credential expired"));
        assert!(!captured.diagnostic.contains("code="), "{}", captured.diagnostic);
        assert!(!captured.diagnostic.contains("facts="), "{}", captured.diagnostic);
    }

    #[test]
    fn borrowed_datafusion_error_capture_walks_source_chain_without_owned_adapter_wrapper() {
        let error = DataFusionError::Context(
            "server wrapper".to_string(),
            Box::new(DataFusionError::External(Box::new(TestSourceError::with_source(
                "dependency wrapper",
                TestSourceError::leaf("root credential expired"),
            )))),
        );

        let captured = datafusion_error_to_captured_error(&error, TEST_OPERATION);

        assert_eq!(captured.operation, TEST_OPERATION);
        assert_eq!(
            captured.source_chain.as_deref(),
            Some("server wrapper: dependency wrapper: root credential expired")
        );
        assert_eq!(captured.root_cause.as_deref(), Some("root credential expired"));
        assert!(captured.facts.iter().any(|(key, _)| key == "datafusion_error"));
        assert_eq!(captured.facts, vec![("datafusion_error".to_string(), "context".to_string())]);
        assert!(!captured.diagnostic.contains("code="), "{}", captured.diagnostic);
        assert!(!captured.diagnostic.contains("facts="), "{}", captured.diagnostic);
    }

    #[cfg(feature = "iceberg")]
    #[test]
    fn server_error_session_preserves_foreign_iceberg_classification() {
        use nexus_iceberg::{IcebergError, IcebergErrorCode, IcebergErrorStatus, IcebergOperation};

        let auth = ServerError::session(
            "catalog authentication failed",
            IcebergError::new(
                IcebergErrorCode::Authentication,
                IcebergErrorStatus::Permanent,
                IcebergOperation::CatalogConnect,
                "authentication failed",
            ),
        );
        let access = ServerError::session(
            "catalog access failed",
            IcebergError::new(
                IcebergErrorCode::Access,
                IcebergErrorStatus::Permanent,
                IcebergOperation::CatalogListTables,
                "access denied",
            ),
        );
        let unknown = ServerError::session("foreign source failed", std::io::Error::other("unknown"));

        assert_eq!(auth.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(access.code(), ErrorCode::SourceAccessDenied);
        assert_eq!(unknown.code(), ErrorCode::Internal);
    }

    #[test]
    fn server_error_session_preserves_native_query_error_classification() {
        let native = nexus_query_engine::error::QueryError::new(
            nexus_query_engine::error::QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.admit",
            "native query service queue is full",
        )
        .with_fact("reason", "queue_full");

        let error = ServerError::session("query service admission failed", native);

        assert_eq!(error.code(), ErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("native_code"), Some("service_overloaded"));
        assert_eq!(error.fact_value("native_operation"), Some("nexus_query_engine.query_service.admit"));
        assert_eq!(error.fact_value("reason"), Some("queue_full"));

        let status = adapter_error_to_status(error.as_adapter_error());
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_error_metadata(&status, "service_overloaded", "resource_exhausted", "temporary");
    }

    #[test]
    fn adapter_unsupported_column_facts_are_client_visible() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::FinalPlanRequirementUnsatisfied,
                "datafusion_nexus.planner.final_plan_requirement",
                "GPU execution does not support projected column `countDetail` with Arrow type `List(Float64)`",
            )
            .with_fact("reason", "unsupported_type")
            .with_fact("unsupported_column", "countDetail")
            .with_fact("unsupported_column_type", "List(Float64)")
            .with_fact("unsupported_column_role", "projected")
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "final_plan_requirement_unsatisfied", "planning", "permanent");
        assert_eq!(
            status.message(),
            "GPU execution does not support projected column `countDetail` with Arrow type `List(Float64)`"
        );
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_METADATA_KEY, "countDetail");
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, "List(Float64)");
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, "projected");
        assert_metadata_value(&status, REASON_METADATA_KEY, "unsupported_type");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_graph_predicate_reason_facts_are_client_visible() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.execute",
                "native cuGraph SQL query execution failed",
            )
            .with_fact("reason", "partial_source_vertices_absent")
            .with_fact("predicate_role", "source_vertices")
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "invalid_execution_plan", "planning", "permanent");
        assert_metadata_value(&status, REASON_METADATA_KEY, "partial_source_vertices_absent");
        assert_metadata_value(&status, PREDICATE_ROLE_METADATA_KEY, "source_vertices");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_graph_unsupported_validation_facts_are_client_visible() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::UnsupportedType,
                "datafusion_nexus.cugraph_sql.execute_gpu",
                "native cuGraph SQL GPU-resident query execution failed",
            )
            .with_fact("native_message", "BFS predicate id column dtype must match the graph vertex dtype")
            .with_fact("reason", "predicate_dtype_mismatch")
            .with_fact("predicate_role", "target_vertices")
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::Unimplemented);
        assert_error_metadata(&status, "unsupported_type", "unsupported", "permanent");
        assert_metadata_value(
            &status,
            NATIVE_MESSAGE_METADATA_KEY,
            "BFS predicate id column dtype must match the graph vertex dtype",
        );
        assert_metadata_value(&status, REASON_METADATA_KEY, "predicate_dtype_mismatch");
        assert_metadata_value(&status, PREDICATE_ROLE_METADATA_KEY, "target_vertices");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_graph_planning_rejection_surfaces_native_message_and_facts() {
        // Mirrors the `Error` shape produced by `Error::from_native` for a
        // cuGraph betweenness vertex-threshold rejection: a generic adapter
        // message overriding the specific native message, with the native
        // message and structured facts preserved as facts.
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.execute_gpu",
                "native cuGraph SQL GPU-resident query execution failed",
            )
            .with_fact(
                "native_message",
                "exact betweenness requires explicit seeds or k above the configured vertex threshold",
            )
            .with_fact("reason", "exact_betweenness_vertex_threshold_exceeded")
            .with_fact("required_option", "seeds_or_k")
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "invalid_execution_plan", "planning", "permanent");
        assert_metadata_value(
            &status,
            NATIVE_MESSAGE_METADATA_KEY,
            "exact betweenness requires explicit seeds or k above the configured vertex threshold",
        );
        assert_metadata_value(&status, REASON_METADATA_KEY, "exact_betweenness_vertex_threshold_exceeded");
        assert_metadata_value(&status, REQUIRED_OPTION_METADATA_KEY, "seeds_or_k");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_graph_execution_failure_stays_generic() {
        // An internal/execution-kind graph failure must not surface the native
        // message: it is collapsed to the generic internal message and the
        // graph-context enrichment is gated to client-actionable graph kinds.
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::Internal,
                "datafusion_nexus.cugraph_sql.execute_gpu",
                "native cuGraph SQL GPU-resident query execution failed",
            )
            .with_fact("native_message", "internal cuGraph scheduler state corruption")
            .with_fact("reason", "internal_scheduler_state")
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal server error while processing the request");
        assert!(!status.message().contains("internal cuGraph scheduler state"));
        assert!(!status.message().contains("reason="));
    }

    #[test]
    fn adapter_non_graph_reason_fact_stays_generic_for_internal_errors() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(ErrorCode::Internal, "datafusion_nexus.native.execution", "native execution failed")
                .with_fact("reason", "internal_scheduler_state")
                .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal server error while processing the request");
        assert!(!status.message().contains("reason="));
    }

    fn assert_error_metadata(status: &Status, code: &str, kind: &str, error_status: &str) {
        assert_metadata_value(status, ERROR_CODE_METADATA_KEY, code);
        assert_metadata_value(status, ERROR_KIND_METADATA_KEY, kind);
        assert_metadata_value(status, ERROR_STATUS_METADATA_KEY, error_status);
    }

    fn assert_metadata_value(status: &Status, key: &'static str, expected: &str) {
        let actual = status.metadata().get(key).unwrap_or_else(|| panic!("{key} metadata must be present"));
        assert_eq!(actual.to_str().unwrap_or_else(|_| panic!("{key} metadata must be ASCII")), expected);
    }

    fn assert_no_diagnostic_fields(message: &str) {
        for needle in ["code=", "kind=", "status=", "operation=", "facts=", "source="] {
            assert!(!message.contains(needle), "client message leaked diagnostic field {needle}: {message}");
        }
    }
}
