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
    effective_datafusion_error(error).message().into_owned()
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
