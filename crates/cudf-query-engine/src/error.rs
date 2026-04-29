use std::{collections::BTreeMap, error::Error as StdError, fmt};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

pub type Result<T> = std::result::Result<T, QueryError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueryErrorCode {
    UnsupportedPlanShape,
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    InvalidPlan,
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

impl QueryErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedPlanShape => "unsupported_plan_shape",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidPlan => "invalid_plan",
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

    pub const fn kind(self) -> QueryErrorKind {
        kind_for_code(self)
    }

    pub const fn status(self) -> QueryErrorStatus {
        status_for_code(self)
    }
}

impl fmt::Display for QueryErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueryErrorKind {
    Planning,
    InvalidInput,
    Unsupported,
    ResourceExhausted,
    DependencyFailure,
    InvariantViolation,
    Internal,
}

impl QueryErrorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::InvalidInput => "invalid_input",
            Self::Unsupported => "unsupported",
            Self::ResourceExhausted => "resource_exhausted",
            Self::DependencyFailure => "dependency_failure",
            Self::InvariantViolation => "invariant_violation",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for QueryErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryErrorStatus {
    Permanent,
    Temporary,
    Persistent,
}

impl QueryErrorStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permanent => "permanent",
            Self::Temporary => "temporary",
            Self::Persistent => "persistent",
        }
    }
}

impl fmt::Display for QueryErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const fn kind_for_code(code: QueryErrorCode) -> QueryErrorKind {
    match code {
        QueryErrorCode::UnsupportedPlanShape
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType => QueryErrorKind::Unsupported,
        QueryErrorCode::InvalidPlan => QueryErrorKind::Planning,
        QueryErrorCode::CudfInvalidInput => QueryErrorKind::InvalidInput,
        QueryErrorCode::CudfUnsupported => QueryErrorKind::Unsupported,
        QueryErrorCode::CudfResourceExhausted | QueryErrorCode::MemoryReservationFailed => {
            QueryErrorKind::ResourceExhausted
        },
        QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::CudfPersistentFailure
        | QueryErrorCode::CudfDependencyFailure => QueryErrorKind::DependencyFailure,
        QueryErrorCode::CudfInvariantViolation => QueryErrorKind::InvariantViolation,
        QueryErrorCode::Internal => QueryErrorKind::Internal,
    }
}

pub const fn status_for_code(code: QueryErrorCode) -> QueryErrorStatus {
    match code {
        QueryErrorCode::CudfResourceExhausted
        | QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::MemoryReservationFailed => QueryErrorStatus::Temporary,
        QueryErrorCode::CudfPersistentFailure => QueryErrorStatus::Persistent,
        QueryErrorCode::UnsupportedPlanShape
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType
        | QueryErrorCode::InvalidPlan
        | QueryErrorCode::CudfInvalidInput
        | QueryErrorCode::CudfUnsupported
        | QueryErrorCode::CudfDependencyFailure
        | QueryErrorCode::CudfInvariantViolation
        | QueryErrorCode::Internal => QueryErrorStatus::Permanent,
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
        assert!(!key.is_empty(), "cudf-query-engine error fact key must not be empty");
        let previous = self.values.insert(key, value);
        assert!(previous.is_none(), "cudf-query-engine error fact key inserted twice: {key}");
    }
}

pub struct QueryError {
    code: QueryErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    source: Option<BoxedSourceError>,
}

impl QueryError {
    pub fn new(code: QueryErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "cudf-query-engine error operation must not be empty");
        assert!(!message.is_empty(), "cudf-query-engine error message must not be empty");
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

    pub fn code(&self) -> QueryErrorCode {
        self.code
    }

    pub fn kind(&self) -> QueryErrorKind {
        self.code.kind()
    }

    pub fn status(&self) -> QueryErrorStatus {
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

impl fmt::Display for QueryError {
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

impl fmt::Debug for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryError")
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

impl StdError for QueryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

fn code_for_cudf_error(error: &cudf::Error) -> QueryErrorCode {
    if error.kind() == cudf::ErrorKind::ResourceExhausted {
        return QueryErrorCode::CudfResourceExhausted;
    }

    match error.status() {
        cudf::ErrorStatus::Temporary => return QueryErrorCode::CudfTemporaryFailure,
        cudf::ErrorStatus::Persistent => return QueryErrorCode::CudfPersistentFailure,
        cudf::ErrorStatus::Permanent => {},
    }

    match error.kind() {
        cudf::ErrorKind::InvalidInput | cudf::ErrorKind::NotFound => QueryErrorCode::CudfInvalidInput,
        cudf::ErrorKind::Unsupported => QueryErrorCode::CudfUnsupported,
        cudf::ErrorKind::DependencyFailure => QueryErrorCode::CudfDependencyFailure,
        cudf::ErrorKind::InvariantViolation | cudf::ErrorKind::ExecutionContext => {
            QueryErrorCode::CudfInvariantViolation
        },
        _ => QueryErrorCode::CudfDependencyFailure,
    }
}

fn quote(value: &str) -> String {
    if value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/' | '(' | ')')) {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}
