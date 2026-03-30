use std::{error::Error as StdError, fmt};

use datafusion::common::{DataFusionError, SchemaError};
use tonic::{Status, metadata::MetadataValue};

use super::{datafusion_error::find_adapter_error, observability::CapturedError};
use crate::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts};

/// Flight SQL response metadata key containing the stable adapter error code.
pub const ERROR_CODE_METADATA_KEY: &str = "x-datafusion-nexus-error-code";
/// Flight SQL response metadata key containing the stable adapter error kind.
pub const ERROR_KIND_METADATA_KEY: &str = "x-datafusion-nexus-error-kind";
/// Flight SQL response metadata key containing the retry/status classification.
pub const ERROR_STATUS_METADATA_KEY: &str = "x-datafusion-nexus-error-status";
/// Flight SQL response metadata key containing the first unsupported column name, when known.
pub const UNSUPPORTED_COLUMN_METADATA_KEY: &str = "x-datafusion-nexus-unsupported-column";
/// Flight SQL response metadata key containing the first unsupported column Arrow type, when known.
pub const UNSUPPORTED_COLUMN_TYPE_METADATA_KEY: &str = "x-datafusion-nexus-unsupported-column-type";
/// Flight SQL response metadata key containing the first unsupported column role, when known.
pub const UNSUPPORTED_COLUMN_ROLE_METADATA_KEY: &str = "x-datafusion-nexus-unsupported-column-role";
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

    pub fn session(message: impl Into<String>, source: impl StdError + Send + Sync + 'static) -> Self {
        let message = message.into();
        let source: Box<dyn StdError + Send + Sync> = Box::new(source);
        let source = match source.downcast::<Error>() {
            Ok(adapter_error) => return Self { inner: adapter_error_with_message(message, *adapter_error) },
            Err(source) => source,
        };
        let source = match source.downcast::<nexus_query_engine::QueryError>() {
            Ok(native_error) => {
                return Self { inner: Error::from_native("datafusion_nexus.server.session", message, *native_error) };
            },
            Err(source) => source,
        };
        let source = match source.downcast::<DataFusionError>() {
            Ok(datafusion_error) => {
                return Self { inner: server_error_from_datafusion_error(message, *datafusion_error) };
            },
            Err(source) => source,
        };
        Self {
            inner: Error::new(ErrorCode::CudfDependencyFailure, "datafusion_nexus.server.session", message)
                .with_source(ServerSource(source)),
        }
    }

    pub fn transport(source: tonic::transport::Error) -> Self {
        Self::new(ErrorCode::CudfDependencyFailure, "datafusion_nexus.server.transport", "Flight SQL transport failed")
            .with_source(source)
    }

    fn new(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        Self { inner: Error::new(code, operation, message) }
    }

    #[must_use]
    fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
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
    let mut error = Error::new(source.code(), source.operation(), message);
    for (key, value) in source.facts().iter() {
        error = error.with_fact(key, value);
    }
    error.with_source(source)
}

fn server_error_from_datafusion_error(message: String, source: DataFusionError) -> Error {
    if let Some((code, operation, facts)) = find_adapter_error(&source).map(|adapter_source| {
        (
            adapter_source.code(),
            adapter_source.operation(),
            adapter_source.facts().iter().map(|(key, value)| (key, value.to_string())).collect::<Vec<_>>(),
        )
    }) {
        let mut error = Error::new(code, operation, message);
        for (key, value) in facts {
            error = error.with_fact(key, value);
        }
        return error.with_source(source);
    }
    let code = classify_datafusion_error(&source);
    let datafusion_error = datafusion_error_variant(&source);
    Error::new(code, "datafusion_nexus.server.session", message)
        .with_fact("datafusion_error", datafusion_error)
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
        if self.facts().iter().next().is_none() {
            f.write_str(" facts={}")?;
        } else {
            f.write_str(" facts={")?;
            for (index, (key, value)) in self.facts().iter().enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{key}={}", quote(value))?;
            }
            f.write_str("}")?;
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
    if let Some(adapter_error) = find_adapter_error(&error) {
        return adapter_error_to_status(adapter_error);
    }
    adapter_error_to_status(&adapter_error_from_datafusion_error(error, operation))
}

pub(super) fn datafusion_error_to_captured_error(error: &DataFusionError, operation: &'static str) -> CapturedError {
    if let Some(adapter_error) = find_adapter_error(error) {
        return CapturedError::from_adapter_error(adapter_error);
    }
    let code = classify_datafusion_error(error);
    let message = client_datafusion_error_message(error);
    let datafusion_error = datafusion_error_variant(error);
    let adapter_error = Error::new(code, operation, message).with_fact("datafusion_error", datafusion_error);
    CapturedError::from_adapter_error(&adapter_error)
}

pub(crate) fn adapter_error_to_status(error: &Error) -> Status {
    let message = client_status_message(error);
    tracing::warn!(
        error_code = error.code().as_str(),
        error_kind = error.kind().as_str(),
        error_status = error.status().as_str(),
        error_operation = error.operation(),
        client_message = message.as_str(),
        diagnostic = diagnostic_status_message(error).as_str(),
        "converted adapter error to Flight SQL status"
    );
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
    };
    status_with_error_metadata(status, error)
}

fn client_status_message(error: &Error) -> String {
    let message = match error.code() {
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
    };
    let message = append_unsupported_column_context(error, message);
    append_graph_predicate_context(error, message)
}

fn append_unsupported_column_context(error: &Error, message: String) -> String {
    let Some(column) = error.fact_value("unsupported_column") else {
        return message;
    };
    let column_type = error.fact_value("unsupported_column_type").unwrap_or("unknown");
    let role = error.fact_value("unsupported_column_role").unwrap_or("unknown");
    if message.contains("unsupported_column=") {
        return message;
    }
    format!(
        "{message}; unsupported_column={column}; unsupported_column_type={column_type}; unsupported_column_role={role}"
    )
}

fn append_graph_predicate_context(error: &Error, message: String) -> String {
    let Some(role) = error.fact_value("predicate_role") else {
        return message;
    };
    let graph_context = error.operation().contains("cugraph")
        || error.fact_value("native_operation").is_some_and(|operation| operation.contains("graph_algorithm"));
    if !graph_context {
        return message;
    }
    let mut context = Vec::new();
    if let Some(reason) = error.fact_value("reason")
        && !message.contains("reason=")
    {
        context.push(format!("reason={reason}"));
    }
    if !message.contains("predicate_role=") {
        context.push(format!("predicate_role={role}"));
    }
    if context.is_empty() { message } else { format!("{message}; {}", context.join("; ")) }
}

fn diagnostic_status_message(error: &Error) -> String {
    error.to_string()
}

fn status_with_error_metadata(mut status: Status, error: &Error) -> Status {
    insert_metadata(status.metadata_mut(), ERROR_CODE_METADATA_KEY, error.code().as_str());
    insert_metadata(status.metadata_mut(), ERROR_KIND_METADATA_KEY, error.kind().as_str());
    insert_metadata(status.metadata_mut(), ERROR_STATUS_METADATA_KEY, error.status().as_str());
    insert_fact_metadata(status.metadata_mut(), UNSUPPORTED_COLUMN_METADATA_KEY, error, "unsupported_column");
    insert_fact_metadata(status.metadata_mut(), UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, error, "unsupported_column_type");
    insert_fact_metadata(status.metadata_mut(), UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, error, "unsupported_column_role");
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
    Error::new(code, operation, message).with_fact("datafusion_error", datafusion_error).with_source(error)
}

fn client_datafusion_error_message(error: &DataFusionError) -> String {
    let message = effective_datafusion_error(error).message().into_owned();
    maybe_append_cugraph_feature_hint(message)
}

/// When a missing `cugraph_*` table function shows up in a DataFusion
/// not-found message, the bare "table function '...' not found" text hides the
/// real cause. The hint differs by build:
///
/// - Built *without* the `cugraph` feature: the cuGraph SQL surface is compiled
///   out, so the function was never registered. Point operators at rebuilding
///   with the feature.
/// - Built *with* the `cugraph` feature: the code is present but registration is
///   gated on `NEXUS_SERVER_CUGRAPH_ENABLED=true` (and the optional
///   `NEXUS_SERVER_CUGRAPH_ALGORITHMS` allowlist). A not-found here
///   means the runtime toggle is off or the function was excluded, not a build
///   problem — point operators at the env vars instead.
#[cfg(not(feature = "cugraph"))]
fn maybe_append_cugraph_feature_hint(message: String) -> String {
    if is_cugraph_table_function_not_found(&message) {
        format!(
            "{message}; this build was compiled without the `cugraph` feature, so cuGraph SQL \
             functions are unavailable - rebuild with --features server-cugraph (or \
             --features server,cugraph) to enable them"
        )
    } else {
        message
    }
}

#[cfg(feature = "cugraph")]
fn maybe_append_cugraph_feature_hint(message: String) -> String {
    if is_cugraph_table_function_not_found(&message) {
        format!(
            "{message}; this build includes the `cugraph` feature but cuGraph SQL functions are \
             not registered - set NEXUS_SERVER_CUGRAPH_ENABLED=true (and ensure the \
             function is not excluded by NEXUS_SERVER_CUGRAPH_ALGORITHMS) to enable them"
        )
    } else {
        message
    }
}

/// True when `message` is a DataFusion "table function not found" error whose
/// missing function name starts with `cugraph_`. Only the quoted/suffixed
/// function name is inspected (not the whole message) so a future "did you
/// mean" suggestion that merely mentions a `cugraph_*` name elsewhere does not
/// trigger a false positive.
fn is_cugraph_table_function_not_found(message: &str) -> bool {
    missing_table_function_name(message).is_some_and(|name| name.to_ascii_lowercase().starts_with("cugraph_"))
}

/// Extracts the missing table-function name from a DataFusion not-found
/// message, handling both the quoted form (`table function 'name' not found`)
/// and the suffixed form (`table function not found: name`). Returns `None` for
/// any other message shape.
fn missing_table_function_name(message: &str) -> Option<&str> {
    let message = message.trim();
    if let Some(rest) = message.strip_prefix("table function '") {
        return rest.split('\'').next().filter(|name| !name.is_empty());
    }
    if let Some(rest) = message.strip_prefix("table function not found:") {
        let name = rest.trim();
        return (!name.is_empty()).then_some(name);
    }
    None
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
        DataFusionError::Plan(message) if datafusion_not_found_message(message) => ErrorCode::ObjectNotFound,
        DataFusionError::Plan(_) => ErrorCode::DataFusionPlanning,
        DataFusionError::SchemaError(schema, _) => match schema.as_ref() {
            SchemaError::FieldNotFound { .. } => ErrorCode::FieldNotFound,
            SchemaError::AmbiguousReference { .. }
            | SchemaError::DuplicateQualifiedField { .. }
            | SchemaError::DuplicateUnqualifiedField { .. } => ErrorCode::DataFusionPlanning,
        },
        DataFusionError::NotImplemented(_) => ErrorCode::UnsupportedOperator,
        DataFusionError::Configuration(_) => ErrorCode::DataFusionPlanning,
        DataFusionError::Execution(message) if datafusion_not_found_message(message) => ErrorCode::ObjectNotFound,
        DataFusionError::Execution(_) => ErrorCode::DataFusionPlanning,
        DataFusionError::ResourcesExhausted(_) => ErrorCode::MemoryReservationFailed,
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            classify_datafusion_error(source)
        },
        DataFusionError::Collection(errors) => {
            errors.first().map(classify_datafusion_error).unwrap_or(ErrorCode::Internal)
        },
        DataFusionError::Shared(source) => classify_datafusion_error(source),
        DataFusionError::Internal(_)
        | DataFusionError::ArrowError(_, _)
        | DataFusionError::IoError(_)
        | DataFusionError::External(_)
        | DataFusionError::Substrait(_)
        | DataFusionError::ExecutionJoin(_)
        | DataFusionError::Ffi(_) => ErrorCode::Internal,
        _ => ErrorCode::Internal,
    }
}

fn datafusion_not_found_message(message: &str) -> bool {
    let message = message.trim().to_ascii_lowercase();
    (message.starts_with("table '") && message.contains("' not found"))
        || (message.starts_with("table '")
            && (message.contains("' doesn't exist") || message.contains("' does not exist")))
        || (message.starts_with("view '")
            && (message.contains("' doesn't exist") || message.contains("' does not exist")))
        || message.starts_with("table not found:")
        || (message.starts_with("table function '") && message.contains("' not found"))
        || message.starts_with("table function not found:")
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
        DataFusionError::Substrait(_) => "substrait",
        DataFusionError::Diagnostic(_, _) => "diagnostic",
        DataFusionError::Collection(_) => "collection",
        DataFusionError::Shared(_) => "shared",
        DataFusionError::Ffi(_) => "ffi",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use datafusion::common::{Column, SchemaError, TableReference};
    use tonic::Code;

    use super::*;

    const TEST_OPERATION: &str = "datafusion_nexus.server.flight.test.create_logical_plan";

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
    fn known_missing_table_plan_message_is_object_not_found() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("table 'does_not_exist' not found".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::NotFound);
        assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
        assert_eq!(status.message(), "table 'does_not_exist' not found");
    }

    #[test]
    fn missing_drop_target_execution_message_is_object_not_found() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Execution("Table 'does_not_exist' doesn't exist.".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::NotFound);
        assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
        assert_eq!(status.message(), "Table 'does_not_exist' doesn't exist.");
    }

    #[test]
    fn generic_not_found_plan_message_is_not_object_not_found() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("optimizer rule was not found in registry".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
        assert_eq!(status.message(), "optimizer rule was not found in registry");
    }

    #[test]
    fn table_function_not_found_plan_message_is_object_not_found() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("table function 'read_json' not found".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::NotFound);
        assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
    }

    #[cfg(not(feature = "cugraph"))]
    #[test]
    fn cugraph_table_function_not_found_appends_build_hint_without_feature() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("table function 'cugraph_list_algorithms' not found".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::NotFound);
        assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
        assert!(status.message().contains("cugraph_list_algorithms"));
        assert!(status.message().contains("without the `cugraph` feature"));
        assert!(status.message().contains("server-cugraph"));
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cugraph_table_function_not_found_appends_runtime_hint_with_feature() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("table function 'cugraph_list_algorithms' not found".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::NotFound);
        assert_error_metadata(&status, "object_not_found", "not_found", "permanent");
        assert!(status.message().contains("cugraph_list_algorithms"));
        assert!(status.message().contains("NEXUS_SERVER_CUGRAPH_ENABLED=true"));
        // The runtime build must not tell operators to rebuild.
        assert!(!status.message().contains("without the `cugraph` feature"));
    }

    #[test]
    fn non_cugraph_table_function_not_found_has_no_hint() {
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("table function 'read_json' not found".to_string()),
            TEST_OPERATION,
        );

        assert_eq!(status.message(), "table function 'read_json' not found");
    }

    #[test]
    fn cugraph_mention_outside_function_name_is_not_a_false_positive() {
        // A non-cugraph function whose suggestion text merely mentions a
        // cugraph_* name must not trigger the cuGraph hint on either build.
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Plan("table function 'read_json' not found. Did you mean 'cugraph_bfs'?".to_string()),
            TEST_OPERATION,
        );

        assert!(!status.message().contains("CUGRAPH_ENABLED"));
        assert!(!status.message().contains("server-cugraph"));
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
    fn server_error_session_preserves_native_query_error_classification() {
        let native = nexus_query_engine::QueryError::new(
            nexus_query_engine::QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.admit",
            "native query service queue is full",
        )
        .with_fact("tenant_id", "default");

        let error = ServerError::session("query service admission failed", native);

        assert_eq!(error.code(), ErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("native_code"), Some("service_overloaded"));
        assert_eq!(error.fact_value("native_operation"), Some("nexus_query_engine.query_service.admit"));
        assert_eq!(error.fact_value("tenant_id"), Some("default"));

        let status = adapter_error_to_status(error.as_adapter_error());
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_error_metadata(&status, "service_overloaded", "resource_exhausted", "temporary");
    }

    #[test]
    fn adapter_unsupported_column_facts_are_client_visible() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::NativeCandidateRejected,
                "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
                "table-format scan leaf was not rewritten to native; first rejection=unsupported_type; no CPU Arrow reader fallback",
            )
            .with_fact("unsupported_column", "countDetail")
            .with_fact(
                "unsupported_column_type",
                "Map(\"entries\": non-null Struct(\"key\": non-null Utf8, \"value\": Int32), unsorted)",
            )
            .with_fact("unsupported_column_role", "projected")
            .into(),
            TEST_OPERATION,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "native_candidate_rejected", "planning", "permanent");
        assert!(status.message().contains("first rejection=unsupported_type"));
        assert!(status.message().contains("unsupported_column=countDetail"));
        assert!(status.message().contains(
            "unsupported_column_type=Map(\"entries\": non-null Struct(\"key\": non-null Utf8, \"value\": Int32), unsorted)"
        ));
        assert!(status.message().contains("unsupported_column_role=projected"));
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_METADATA_KEY, "countDetail");
        assert_metadata_value(
            &status,
            UNSUPPORTED_COLUMN_TYPE_METADATA_KEY,
            "Map(\"entries\": non-null Struct(\"key\": non-null Utf8, \"value\": Int32), unsorted)",
        );
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, "projected");
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
        assert!(status.message().contains("reason=partial_source_vertices_absent"));
        assert!(status.message().contains("predicate_role=source_vertices"));
        assert_no_diagnostic_fields(status.message());
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
