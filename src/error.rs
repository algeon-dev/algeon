use std::{collections::BTreeMap, error::Error as StdError, fmt};

use datafusion::common::DataFusionError;

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    GpuCandidateRejected,
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    InvalidExecutionPlan,
    InvalidChildCount,
    CudfInvalidInput,
    CudfUnsupported,
    CudfResourceExhausted,
    CudfTemporaryFailure,
    CudfPersistentFailure,
    CudfDependencyFailure,
    CudfInvariantViolation,
    MemoryReservationFailed,
    Internal,
}

impl ErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuCandidateRejected => "gpu_candidate_rejected",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidExecutionPlan => "invalid_execution_plan",
            Self::InvalidChildCount => "invalid_child_count",
            Self::CudfInvalidInput => "cudf_invalid_input",
            Self::CudfUnsupported => "cudf_unsupported",
            Self::CudfResourceExhausted => "cudf_resource_exhausted",
            Self::CudfTemporaryFailure => "cudf_temporary_failure",
            Self::CudfPersistentFailure => "cudf_persistent_failure",
            Self::CudfDependencyFailure => "cudf_dependency_failure",
            Self::CudfInvariantViolation => "cudf_invariant_violation",
            Self::MemoryReservationFailed => "memory_reservation_failed",
            Self::Internal => "internal",
        }
    }

    pub const fn kind(self) -> ErrorKind {
        kind_for_code(self)
    }

    pub const fn status(self) -> ErrorStatus {
        status_for_code(self)
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    Planning,
    InvalidInput,
    Unsupported,
    Execution,
    ResourceExhausted,
    DependencyFailure,
    InvariantViolation,
    Internal,
}

impl ErrorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::InvalidInput => "invalid_input",
            Self::Unsupported => "unsupported",
            Self::Execution => "execution",
            Self::ResourceExhausted => "resource_exhausted",
            Self::DependencyFailure => "dependency_failure",
            Self::InvariantViolation => "invariant_violation",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    Permanent,
    Temporary,
    Persistent,
}

impl ErrorStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permanent => "permanent",
            Self::Temporary => "temporary",
            Self::Persistent => "persistent",
        }
    }
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const fn kind_for_code(code: ErrorCode) -> ErrorKind {
    match code {
        ErrorCode::GpuCandidateRejected | ErrorCode::InvalidExecutionPlan => ErrorKind::Planning,
        ErrorCode::UnsupportedOperator | ErrorCode::UnsupportedExpression | ErrorCode::UnsupportedType => {
            ErrorKind::Unsupported
        },
        ErrorCode::InvalidChildCount => ErrorKind::Internal,
        ErrorCode::CudfInvalidInput => ErrorKind::InvalidInput,
        ErrorCode::CudfUnsupported => ErrorKind::Unsupported,
        ErrorCode::CudfResourceExhausted | ErrorCode::MemoryReservationFailed => ErrorKind::ResourceExhausted,
        ErrorCode::CudfTemporaryFailure | ErrorCode::CudfPersistentFailure | ErrorCode::CudfDependencyFailure => {
            ErrorKind::DependencyFailure
        },
        ErrorCode::CudfInvariantViolation => ErrorKind::InvariantViolation,
        ErrorCode::Internal => ErrorKind::Internal,
    }
}

pub const fn status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::CudfResourceExhausted | ErrorCode::CudfTemporaryFailure | ErrorCode::MemoryReservationFailed => {
            ErrorStatus::Temporary
        },
        ErrorCode::CudfPersistentFailure => ErrorStatus::Persistent,
        ErrorCode::GpuCandidateRejected
        | ErrorCode::UnsupportedOperator
        | ErrorCode::UnsupportedExpression
        | ErrorCode::UnsupportedType
        | ErrorCode::InvalidExecutionPlan
        | ErrorCode::InvalidChildCount
        | ErrorCode::CudfInvalidInput
        | ErrorCode::CudfUnsupported
        | ErrorCode::CudfDependencyFailure
        | ErrorCode::CudfInvariantViolation
        | ErrorCode::Internal => ErrorStatus::Permanent,
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, String>,
}

impl Facts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.values.iter().map(|(key, value)| (*key, value.as_str()))
    }

    fn insert(&mut self, key: &'static str, value: String) {
        assert!(!key.is_empty(), "datafusion-cudf error fact key must not be empty");
        let previous = self.values.insert(key, value);
        assert!(previous.is_none(), "datafusion-cudf error fact key inserted twice: {key}");
    }
}

pub struct Error {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    source: Option<BoxedSourceError>,
}

impl Error {
    pub fn new(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "datafusion-cudf error operation must not be empty");
        assert!(!message.is_empty(), "datafusion-cudf error message must not be empty");
        Self { code, operation, message, facts: Facts::new(), source: None }
    }

    #[must_use]
    pub fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.facts.insert(key, value.to_string());
        self
    }

    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        self.source = Some(Box::new(source));
        self
    }

    pub fn from_cudf(operation: &'static str, message: impl Into<String>, source: cudf::Error) -> Self {
        let code = code_for_cudf_error(&source);
        Self::new(code, operation, message)
            .with_fact("cudf_code", source.code())
            .with_fact("cudf_kind", source.kind())
            .with_fact("cudf_status", source.status())
            .with_fact("cudf_operation", source.operation())
            .with_source(source)
    }

    pub fn from_native(
        operation: &'static str,
        message: impl Into<String>,
        source: cudf_query_engine::QueryError,
    ) -> Self {
        let code = code_for_native_error(&source);
        let mut error = Self::new(code, operation, message)
            .with_fact("native_code", source.code())
            .with_fact("native_kind", source.kind())
            .with_fact("native_status", source.status())
            .with_fact("native_operation", source.operation());
        for (key, value) in source.facts().iter() {
            error.facts.insert(key, value.to_string());
        }
        error.with_source(source)
    }

    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn kind(&self) -> ErrorKind {
        self.code.kind()
    }

    pub fn status(&self) -> ErrorStatus {
        self.code.status()
    }

    pub fn operation(&self) -> &'static str {
        self.operation
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    pub fn fact_value(&self, key: &str) -> Option<&str> {
        self.facts.get(key)
    }

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

impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        Self::External(Box::new(error))
    }
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

fn code_for_native_error(error: &cudf_query_engine::QueryError) -> ErrorCode {
    match error.code() {
        cudf_query_engine::QueryErrorCode::UnsupportedPlanShape => ErrorCode::UnsupportedOperator,
        cudf_query_engine::QueryErrorCode::UnsupportedOperator => ErrorCode::UnsupportedOperator,
        cudf_query_engine::QueryErrorCode::UnsupportedExpression => ErrorCode::UnsupportedExpression,
        cudf_query_engine::QueryErrorCode::UnsupportedType => ErrorCode::UnsupportedType,
        cudf_query_engine::QueryErrorCode::InvalidPlan => ErrorCode::InvalidExecutionPlan,
        cudf_query_engine::QueryErrorCode::CudfInvalidInput => ErrorCode::CudfInvalidInput,
        cudf_query_engine::QueryErrorCode::CudfUnsupported => ErrorCode::CudfUnsupported,
        cudf_query_engine::QueryErrorCode::CudfResourceExhausted => ErrorCode::CudfResourceExhausted,
        cudf_query_engine::QueryErrorCode::CudfTemporaryFailure => ErrorCode::CudfTemporaryFailure,
        cudf_query_engine::QueryErrorCode::CudfPersistentFailure => ErrorCode::CudfPersistentFailure,
        cudf_query_engine::QueryErrorCode::CudfDependencyFailure => ErrorCode::CudfDependencyFailure,
        cudf_query_engine::QueryErrorCode::CudfInvariantViolation => ErrorCode::CudfInvariantViolation,
        cudf_query_engine::QueryErrorCode::MemoryReservationFailed => ErrorCode::MemoryReservationFailed,
        cudf_query_engine::QueryErrorCode::Internal => ErrorCode::Internal,
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
