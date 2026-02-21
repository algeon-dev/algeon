use std::{error::Error as StdError, fmt};

use datafusion::common::DataFusionError;
use tonic::Status;

use crate::{Error, ErrorKind};

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
    if let DataFusionError::External(source) = &error {
        if let Some(adapter_error) = source.downcast_ref::<Error>() {
            return adapter_error_to_status(adapter_error);
        }
    }
    Status::internal(error.to_string())
}

fn adapter_error_to_status(error: &Error) -> Status {
    let message = error.to_string();
    match error.kind() {
        ErrorKind::InvalidInput | ErrorKind::Planning => Status::invalid_argument(message),
        ErrorKind::Unsupported => Status::unimplemented(message),
        ErrorKind::ResourceExhausted => Status::resource_exhausted(message),
        ErrorKind::Execution | ErrorKind::DependencyFailure | ErrorKind::InvariantViolation | ErrorKind::Internal => {
            Status::internal(message)
        },
    }
}
