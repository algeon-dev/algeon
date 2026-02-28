use std::{error::Error as StdError, fmt};

use datafusion::common::{DataFusionError, SchemaError};
use tonic::{Status, metadata::MetadataValue};

use crate::{Error, ErrorCode, ErrorKind};

pub const ERROR_CODE_METADATA_KEY: &str = "x-datafusion-nexus-error-code";
pub const ERROR_KIND_METADATA_KEY: &str = "x-datafusion-nexus-error-kind";
pub const ERROR_STATUS_METADATA_KEY: &str = "x-datafusion-nexus-error-status";
const DATAFUSION_ERROR_FALLBACK_OPERATION: &str = "datafusion_nexus.server.datafusion_error";

pub type Result<T> = std::result::Result<T, ServerError>;

#[derive(Debug)]
pub struct ServerError {
    message: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl ServerError {
    pub fn invalid_config(field: &'static str, reason: impl Into<String>) -> Self {
        Self { message: format!("invalid server configuration: {field} {}", reason.into()), source: None }
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self { message: format!("invalid server configuration: {}", message.into()), source: None }
    }

    pub fn configuration_with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self { message: format!("invalid server configuration: {}", message.into()), source: Some(Box::new(source)) }
    }

    pub fn session(message: impl Into<String>, source: impl StdError + Send + Sync + 'static) -> Self {
        Self { message: message.into(), source: Some(Box::new(source)) }
    }

    pub fn transport(source: tonic::transport::Error) -> Self {
        Self { message: "Flight SQL transport failed".to_string(), source: Some(Box::new(source)) }
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl StdError for ServerError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
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

fn adapter_error_to_status(error: &Error) -> Status {
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
    match error.code() {
        ErrorCode::SqlParse | ErrorCode::FieldNotFound | ErrorCode::ObjectNotFound | ErrorCode::DataFusionPlanning => {
            error.message().to_string()
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

fn diagnostic_status_message(error: &Error) -> String {
    error.to_string()
}

fn status_with_error_metadata(mut status: Status, error: &Error) -> Status {
    insert_metadata(status.metadata_mut(), ERROR_CODE_METADATA_KEY, error.code().as_str());
    insert_metadata(status.metadata_mut(), ERROR_KIND_METADATA_KEY, error.kind().as_str());
    insert_metadata(status.metadata_mut(), ERROR_STATUS_METADATA_KEY, error.status().as_str());
    status
}

fn insert_metadata(metadata: &mut tonic::metadata::MetadataMap, key: &'static str, value: &'static str) {
    if let Ok(value) = MetadataValue::try_from(value) {
        metadata.insert(key, value);
    }
}

fn adapter_error_from_datafusion_error(error: DataFusionError, operation: &'static str) -> Error {
    let code = classify_datafusion_error(&error);
    let message = client_datafusion_error_message(&error);
    let datafusion_error = datafusion_error_variant(&error);
    Error::new(code, operation, message).with_fact("datafusion_error", datafusion_error).with_source(error)
}

fn find_adapter_error(error: &DataFusionError) -> Option<&Error> {
    match error {
        DataFusionError::External(source) => source.downcast_ref::<Error>(),
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => find_adapter_error(source),
        DataFusionError::Collection(errors) => errors.iter().find_map(find_adapter_error),
        DataFusionError::Shared(source) => find_adapter_error(source),
        _ => None,
    }
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
///   gated on `DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED=true` (and the optional
///   `DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS` allowlist). A not-found here
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
             not registered - set DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED=true (and ensure the \
             function is not excluded by DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS) to enable them"
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
        DataFusionError::Configuration(_) | DataFusionError::Execution(_) => ErrorCode::DataFusionPlanning,
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
        assert!(status.message().contains("DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED=true"));
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

    fn assert_error_metadata(status: &Status, code: &str, kind: &str, error_status: &str) {
        let actual_code = status.metadata().get(ERROR_CODE_METADATA_KEY).expect("error code metadata must be present");
        assert_eq!(actual_code.to_str().expect("error code metadata must be ASCII"), code);
        let actual_kind = status.metadata().get(ERROR_KIND_METADATA_KEY).expect("error kind metadata must be present");
        assert_eq!(actual_kind.to_str().expect("error kind metadata must be ASCII"), kind);
        let actual_status =
            status.metadata().get(ERROR_STATUS_METADATA_KEY).expect("error status metadata must be present");
        assert_eq!(actual_status.to_str().expect("error status metadata must be ASCII"), error_status);
    }

    fn assert_no_diagnostic_fields(message: &str) {
        for needle in ["code=", "kind=", "status=", "operation=", "facts=", "source="] {
            assert!(!message.contains(needle), "client message leaked diagnostic field {needle}: {message}");
        }
    }
}
