use std::{error::Error as StdError, fmt};

use datafusion::common::{DataFusionError, SchemaError};
use tonic::{Status, metadata::MetadataValue};

use crate::{Error, ErrorCode, ErrorKind};

pub const ERROR_CODE_METADATA_KEY: &str = "x-datafusion-cudf-error-code";
pub const ERROR_KIND_METADATA_KEY: &str = "x-datafusion-cudf-error-kind";
pub const ERROR_STATUS_METADATA_KEY: &str = "x-datafusion-cudf-error-status";

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
    if let Some(adapter_error) = find_adapter_error(&error) {
        return adapter_error_to_status(adapter_error);
    }
    adapter_error_to_status(&adapter_error_from_datafusion_error(error))
}

fn adapter_error_to_status(error: &Error) -> Status {
    let message = error.to_string();
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

fn adapter_error_from_datafusion_error(error: DataFusionError) -> Error {
    let code = classify_datafusion_error(&error);
    let message = error.message().into_owned();
    let datafusion_error = datafusion_error_variant(&error);
    Error::new(code, "datafusion_cudf.server.datafusion_error", message)
        .with_fact("datafusion_error", datafusion_error)
        .with_source(error)
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

fn classify_datafusion_error(error: &DataFusionError) -> ErrorCode {
    match error {
        DataFusionError::SQL(_, _) => ErrorCode::SqlParse,
        DataFusionError::Plan(message) if datafusion_not_found_message(message) => ErrorCode::ObjectNotFound,
        DataFusionError::Plan(_) => ErrorCode::DataFusionPlanning,
        DataFusionError::SchemaError(schema, _) => match schema.as_ref() {
            SchemaError::FieldNotFound { .. } => ErrorCode::ObjectNotFound,
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
    let message = message.to_ascii_lowercase();
    message.contains(" not found") || message.contains("not found:")
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
