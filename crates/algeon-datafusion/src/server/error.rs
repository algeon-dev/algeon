use std::{error::Error as StdError, fmt, net::SocketAddr, sync::Arc};

use datafusion::common::DataFusionError;
use query_engine::error::QueryError;
#[cfg(test)]
use query_engine::error::QueryErrorCode;
use tonic::{
    Status,
    metadata::{MetadataMap, MetadataValue},
    transport::Error as TonicTransportError,
};

use crate::server::{
    adapter_error::{
        DataFusionErrorExt, Error, ErrorCode, ErrorKind, ErrorLogDiagnostics, ErrorSourceDiagnostics, ErrorStatus,
        Facts, FailedStatement, classify, datafusion_error_human_message,
    },
    observability::CapturedError,
};

/// Flight SQL response metadata keys
pub const ERROR_CODE_METADATA_KEY: &str = "x-algeon-error-code";
pub const ERROR_KIND_METADATA_KEY: &str = "x-algeon-error-kind";
pub const ERROR_STATUS_METADATA_KEY: &str = "x-algeon-error-status";
pub const ERROR_OPERATION_METADATA_KEY: &str = "x-algeon-error-operation";
pub const UNSUPPORTED_COLUMN_METADATA_KEY: &str = "x-algeon-unsupported-column";
pub const UNSUPPORTED_COLUMN_TYPE_METADATA_KEY: &str = "x-algeon-unsupported-column-type";
pub const UNSUPPORTED_COLUMN_ROLE_METADATA_KEY: &str = "x-algeon-unsupported-column-role";
pub const PREDICATE_ROLE_METADATA_KEY: &str = "x-algeon-predicate-role";
pub const REASON_METADATA_KEY: &str = "x-algeon-reason";
pub const REQUIRED_OPTION_METADATA_KEY: &str = "x-algeon-required-option";
pub const NATIVE_MESSAGE_METADATA_KEY: &str = "x-algeon-native-message";
pub const ALGORITHM_METADATA_KEY: &str = "x-algeon-algorithm";
pub const ITERATIONS_METADATA_KEY: &str = "x-algeon-iterations";
pub const MAX_ITERATIONS_METADATA_KEY: &str = "x-algeon-max-iterations";
pub const EPSILON_METADATA_KEY: &str = "x-algeon-epsilon";
/// Lists client-detail fact keys withheld from Flight metadata because their
/// values are not transmittable ASCII (comma-joined fact keys).
pub const OMITTED_FACTS_METADATA_KEY: &str = "x-algeon-omitted-facts";

/// Maximum bytes for one client-detail metadata value. Longer ASCII values
/// are truncated with a `...` marker; the full diagnosis stays available in
/// the status message and query stats. The fact-metadata key set is fixed, so
/// this per-value bound is also a fixed total bound by construction.
const MAX_CLIENT_FACT_METADATA_VALUE_BYTES: usize = 256;

/// Maximum bytes for a client status message derived from foreign
/// DataFusion text. Bounds what one extreme upstream `Display` can emit onto
/// the wire; the in-process error is never rewritten.
const MAX_CLIENT_DATAFUSION_MESSAGE_BYTES: usize = 2_048;

/// Maximum depth for server-side DataFusion message traversal. Mirrors the
/// adapter chain budget used by `classify`: extraction always terminates,
/// returning the deepest node reached when the budget is exhausted.
const MAX_SERVER_DATAFUSION_TRAVERSAL_DEPTH: usize = 32;

pub type Result<T> = std::result::Result<T, ServerError>;

#[derive(Debug)]
pub struct ServerError {
    inner: Error,
}

impl ServerError {
    pub fn invalid_config(field: &'static str, reason: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::InvalidRequest,
            "algeon_datafusion.server.config",
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
            "algeon_datafusion.server.config",
            format!("invalid server configuration: {}", message.into()),
        )
    }

    pub fn configuration_with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::new(
            ErrorCode::InvalidRequest,
            "algeon_datafusion.server.config",
            format!("invalid server configuration: {}", message.into()),
        )
        .with_source(source)
    }

    pub fn startup(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, "algeon_datafusion.server.startup", message)
    }

    pub(crate) fn shutdown(reason: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, "algeon_datafusion.server.shutdown", message).with_fact("reason", reason)
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
            inner: Error::new(source.code(), "algeon_datafusion.server.shutdown", message)
                .with_fact("reason", reason)
                .with_source(source),
        }
    }

    pub(crate) fn builder_validation(reason: &'static str, message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidRequest, "algeon_datafusion.server.builder", message).with_fact("reason", reason)
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
        let source = match source.downcast::<QueryError>() {
            Ok(native_error) => {
                return Self { inner: Error::from_native("algeon_datafusion.server.session", message, *native_error) };
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
            inner: Error::new(ErrorCode::Internal, "algeon_datafusion.server.session", message)
                .with_source(ServerSource(source)),
        }
    }

    pub fn transport(source: TonicTransportError) -> Self {
        Self::new(
            ErrorCode::ServerTransportFailure,
            "algeon_datafusion.server.transport",
            "Flight SQL transport failed",
        )
        .with_source(source)
    }

    pub(crate) fn transport_at_addr(addr: SocketAddr, source: TonicTransportError) -> Self {
        Self::new(
            ErrorCode::ServerTransportFailure,
            "algeon_datafusion.server.transport",
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
        .find_algeon_error()
        .map(|adapter_source| (adapter_source.code(), adapter_source.operation(), adapter_source.facts().clone()))
    {
        return Error::new(code, operation, message).with_preserved_facts(facts).with_source(source);
    }
    let code = classify(&source, FailedStatement::Other);
    let datafusion_error = datafusion_error_variant(&source);
    Error::new(code, "algeon_datafusion.server.session", message)
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

pub fn datafusion_error_to_status_for_operation(
    error: DataFusionError,
    operation: &'static str,
    statement: FailedStatement,
) -> Status {
    if let Some(adapter_error) = error.find_algeon_error() {
        return adapter_error_to_status(adapter_error);
    }
    adapter_error_to_status(&adapter_error_from_datafusion_error(error, operation, statement))
}

pub(super) fn datafusion_error_to_captured_error(
    error: &DataFusionError,
    operation: &'static str,
    statement: FailedStatement,
) -> CapturedError {
    if let Some(adapter_error) = error.find_algeon_error() {
        return CapturedError::from_adapter_error(adapter_error);
    }
    let code = classify(error, statement);
    let message = client_datafusion_error_message(error);
    let datafusion_error = datafusion_error_variant(error);
    let mut facts = Facts::new();
    facts.insert_log_safe("datafusion_error", datafusion_error.to_string());
    CapturedError::from_datafusion_error(error, code, operation, message, facts)
}

pub(crate) fn adapter_error_to_status(error: &Error) -> Status {
    // Caller schema identifiers reach the wire status only: the logged client
    // message is the text without them, like `error_message`.
    let log_message = client_status_message(error);
    let message = client_schema_detail_message(error).unwrap_or_else(|| log_message.clone());
    let diagnostics = ErrorLogDiagnostics::from_error(error);
    let log_messages = adapter_status_log_messages(error.message(), log_message.as_str(), &diagnostics);
    // The logged `error_message` is owner-authored adapter text: foreign
    // DataFusion text enters it only through the bounded
    // `client_datafusion_error_message` constructor, and foreign source chains
    // only through the bounded `cause`/`source_path` diagnostics above.
    // Arbitrary foreign `Display` implementations can still spend memory
    // producing their own message; that cost cannot be bounded from outside.
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
    let status = match error.code() {
        ErrorCode::AlgorithmDidNotConverge => Status::failed_precondition(message),
        _ => match error.kind() {
            ErrorKind::Cancelled => Status::cancelled(message),
            ErrorKind::NotFound => Status::not_found(message),
            ErrorKind::PermissionDenied => Status::permission_denied(message),
            ErrorKind::Unauthenticated => Status::unauthenticated(message),
            ErrorKind::InvalidInput | ErrorKind::Planning => Status::invalid_argument(message),
            ErrorKind::Unsupported => Status::unimplemented(message),
            ErrorKind::ResourceExhausted => Status::resource_exhausted(message),
            ErrorKind::Execution
            | ErrorKind::DependencyFailure
            | ErrorKind::InvariantViolation
            | ErrorKind::Internal => Status::internal(message),
        },
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
        ErrorCode::AlgorithmDidNotConverge => format!(
            "{} did not converge after {} iterations (max_iterations={}, epsilon={}); change the convergence parameters and retry",
            error.fact_value("algorithm").unwrap_or("native algorithm"),
            error.fact_value("iterations").unwrap_or("unknown"),
            error.fact_value("max_iterations").unwrap_or("unknown"),
            error.fact_value("epsilon").unwrap_or("unknown"),
        ),
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
        // Resource messages never promise that an unchanged retry will
        // succeed: only confirmed admission overload (above) describes queue
        // pressure. Cap, shape, and physical failures name the change the
        // caller must make; execution failures point at the sealed attempt
        // report instead of synthesizing retry permission from the
        // `Temporary` status or the gRPC `RESOURCE_EXHAUSTED` code.
        ErrorCode::QueryMemoryLimitExceeded => format!(
            "query memory cap exceeded: {}. Raise the query memory cap or reduce query resource usage; retrying the same request unchanged cannot succeed",
            error.message()
        ),
        ErrorCode::QueryBudgetUnavailable => format!(
            "query budget cannot fit on any eligible device: {}. Request a smaller query cap or wait for device capacity to free; do not retry the same request unchanged",
            error.message()
        ),
        ErrorCode::SizeOverflow => format!(
            "result size computation overflowed the platform range: {}. Change the request shape to produce a smaller result; retrying the same request unchanged cannot succeed",
            error.message()
        ),
        ErrorCode::HostMaterializationLimitExceeded => format!(
            "one result row exceeds the configured host materialization limit: {}. Narrow the request or raise the limit; retrying the same request unchanged cannot succeed",
            error.message()
        ),
        ErrorCode::HostStagingLimitExceeded => format!(
            "host staging exceeded its configured byte ceiling: {}. Narrow the request or raise the ceiling; retrying the same request unchanged cannot succeed",
            error.message()
        ),
        ErrorCode::FlightFrameTooLarge => format!(
            "one row exceeds the Flight message limit: {}. Project a smaller value; retrying the same request unchanged cannot succeed",
            error.message()
        ),
        ErrorCode::HostMemoryExhausted => format!(
            "host memory pool exhausted while processing the request: {}. Reduce query resource usage; retrying the same request unchanged may fail again",
            error.message()
        ),
        ErrorCode::DeviceOutOfMemory
        | ErrorCode::DeviceAllocationFailed
        | ErrorCode::MemoryReservationFailed
        | ErrorCode::CudfResourceExhausted => format!(
            "device memory exhausted while processing the request: {}. Reduce query resource usage and consult the sealed attempt report for retry guidance; retrying the same request unchanged may fail again",
            error.message()
        ),
        _ if error.kind() == ErrorKind::ResourceExhausted => format!(
            "resource exhausted while processing the request: {}. Change the request to use fewer resources; do not retry the same request unchanged",
            error.message()
        ),
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

/// Wire status text for rejections that name the caller's own schema
/// identifiers.
///
/// Owners keep column and field names and their Arrow types out of
/// `Error::message()`, which the boundary log records verbatim. The values
/// travel as ordinary facts and are composed here, so they reach the query
/// caller and never the log. Returns `None` when the error does not carry the
/// complete fact set of one of these rejections.
fn client_schema_detail_message(error: &Error) -> Option<String> {
    match error.code() {
        ErrorCode::FinalPlanRequirementUnsatisfied => Some(format!(
            "GPU execution does not support {} column `{}` with Arrow type `{}`",
            error.fact_value("unsupported_column_role")?,
            error.fact_value("unsupported_column")?,
            error.fact_value("unsupported_column_type")?,
        )),
        ErrorCode::UnsupportedType => Some(format!(
            "{} (field '{}' has Arrow type `{}`)",
            error.message(),
            error.fact_value("field")?,
            error.fact_value("actual_type")?,
        )),
        _ => None,
    }
}

/// Whether raw native diagnostic text may cross the Flight boundary as the
/// `native_message` trailer.
///
/// `native_message` remains trusted in-process diagnosis, but its key being
/// allowlisted must not by itself publish raw text: internal, execution,
/// dependency, invariant, and resource failures stay generic on the wire.
/// Only client-actionable validation kinds project it alongside their
/// structured facts.
fn native_message_is_client_visible(error: &Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::Cancelled
            | ErrorKind::NotFound
            | ErrorKind::PermissionDenied
            | ErrorKind::Unauthenticated
            | ErrorKind::Planning
            | ErrorKind::InvalidInput
            | ErrorKind::Unsupported
    )
}

/// Client wire contract for Flight error metadata.
///
/// Stable identity (`error-code`, `error-kind`, `error-status`,
/// `error-operation`) is always present and never truncated. Client-detail
/// facts use an explicit per-key allowlist below: values must be transmittable
/// ASCII metadata and are capped at
/// [`MAX_CLIENT_FACT_METADATA_VALUE_BYTES`] with a `...` marker. Values that
/// cannot cross the wire (for example non-ASCII schema details) are omitted
/// and listed under [`OMITTED_FACTS_METADATA_KEY`]; the full diagnosis stays
/// available in the status message and query stats. The key set is fixed, so
/// the per-value bound is also a fixed total bound by construction.
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
    let mut omitted: Vec<&'static str> = Vec::new();
    insert_fact_metadata(
        status.metadata_mut(),
        UNSUPPORTED_COLUMN_METADATA_KEY,
        error,
        "unsupported_column",
        &mut omitted,
    );
    insert_fact_metadata(
        status.metadata_mut(),
        UNSUPPORTED_COLUMN_TYPE_METADATA_KEY,
        error,
        "unsupported_column_type",
        &mut omitted,
    );
    insert_fact_metadata(
        status.metadata_mut(),
        UNSUPPORTED_COLUMN_ROLE_METADATA_KEY,
        error,
        "unsupported_column_role",
        &mut omitted,
    );
    insert_fact_metadata(status.metadata_mut(), PREDICATE_ROLE_METADATA_KEY, error, "predicate_role", &mut omitted);
    insert_fact_metadata(status.metadata_mut(), REASON_METADATA_KEY, error, "reason", &mut omitted);
    insert_fact_metadata(status.metadata_mut(), REQUIRED_OPTION_METADATA_KEY, error, "required_option", &mut omitted);
    if native_message_is_client_visible(error) {
        insert_fact_metadata(status.metadata_mut(), NATIVE_MESSAGE_METADATA_KEY, error, "native_message", &mut omitted);
    }
    insert_fact_metadata(status.metadata_mut(), ALGORITHM_METADATA_KEY, error, "algorithm", &mut omitted);
    insert_fact_metadata(status.metadata_mut(), ITERATIONS_METADATA_KEY, error, "iterations", &mut omitted);
    insert_fact_metadata(status.metadata_mut(), MAX_ITERATIONS_METADATA_KEY, error, "max_iterations", &mut omitted);
    insert_fact_metadata(status.metadata_mut(), EPSILON_METADATA_KEY, error, "epsilon", &mut omitted);
    if !omitted.is_empty()
        && let Ok(marker) = MetadataValue::try_from(omitted.join(","))
    {
        status.metadata_mut().insert(OMITTED_FACTS_METADATA_KEY, marker);
    }
    status
}

fn insert_metadata(metadata: &mut MetadataMap, key: &'static str, value: &'static str) {
    if let Ok(value) = MetadataValue::try_from(value) {
        metadata.insert(key, value);
    }
}

fn insert_fact_metadata(
    metadata: &mut MetadataMap,
    metadata_key: &'static str,
    error: &Error,
    fact_key: &'static str,
    omitted: &mut Vec<&'static str>,
) {
    let Some(value) = error.fact_value(fact_key) else {
        return;
    };
    if !value.is_ascii() {
        omitted.push(fact_key);
        return;
    }
    let (value, _) = truncate_metadata_value(value, MAX_CLIENT_FACT_METADATA_VALUE_BYTES);
    if let Ok(value) = MetadataValue::try_from(value.as_str()) {
        metadata.insert(metadata_key, value);
    } else {
        omitted.push(fact_key);
    }
}

fn truncate_metadata_value(value: &str, max_bytes: usize) -> (String, bool) {
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

fn adapter_error_from_datafusion_error(
    error: DataFusionError,
    operation: &'static str,
    statement: FailedStatement,
) -> Error {
    let code = classify(&error, statement);
    let message = client_datafusion_error_message(&error);
    let datafusion_error = datafusion_error_variant(&error);
    Error::new(code, operation, message).with_log_safe_fact("datafusion_error", datafusion_error).with_source(error)
}

fn client_datafusion_error_message(error: &DataFusionError) -> String {
    let effective = effective_datafusion_error(error);
    let human_message = datafusion_error_human_message(effective);
    let message = match effective {
        DataFusionError::SQL(_, _) => format!("invalid SQL statement: {human_message}"),
        _ => human_message,
    };
    // Foreign upstream text enters the client message here; cap what one
    // extreme `Display` can emit onto the wire. The in-process error keeps its
    // full text for trusted diagnosis.
    truncate_metadata_value(message.as_str(), MAX_CLIENT_DATAFUSION_MESSAGE_BYTES).0
}

/// Resolves the effective DataFusion error following the same wrapper
/// precedence as [`classify`]: `Context`/`Diagnostic` unwrap, `Collection`
/// follows the first branch, `Shared` unwraps.
///
/// Traversal is iterative with an explicit depth budget so an extreme but
/// supported wrapper tree always terminates with a defined error. When the
/// budget is exhausted the deepest node reached is returned; callers must not
/// treat the result as proof that deeper layers hold no other variant.
fn effective_datafusion_error(error: &DataFusionError) -> &DataFusionError {
    let mut current = error;
    for _ in 0..MAX_SERVER_DATAFUSION_TRAVERSAL_DEPTH {
        match current {
            DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => current = source.as_ref(),
            DataFusionError::Collection(errors) => match errors.first() {
                Some(first) => current = first,
                None => break,
            },
            DataFusionError::Shared(source) => current = source.as_ref(),
            _ => break,
        }
    }
    current
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
    use std::{
        error::Error as StdError,
        fmt,
        io::Write,
        sync::{Arc, Mutex},
    };

    use arrow::{datatypes::Schema, error::ArrowError};
    use cudf::testing::error::new_with_for_test;
    use datafusion::{
        common::{Column, Diagnostic, SchemaError, TableReference},
        object_store::Error as ObjectStoreError,
        physical_plan::{ExecutionPlan, empty::EmptyExec},
        prelude::SessionContext,
        sql::sqlparser::parser::ParserError,
    };
    use tonic::Code;
    use tracing::subscriber::with_default;

    use super::*;
    use crate::{
        capability::CapabilityReason,
        gpu_functions::dense_relation_errors::DenseRelationErrorContext,
        planner::{FinalPlanDisposition, native},
        report::{GpuCandidate, NotSupportedDetail, PlanningReport, UnsupportedColumnDetail},
    };

    const TEST_OPERATION: &str = "algeon_datafusion.server.flight.test.create_logical_plan";

    #[derive(Clone, Default)]
    struct LogBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log buffer lock").extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn with_captured_logs<T>(run: impl FnOnce() -> T) -> (T, String) {
        let buffer = LogBuffer::default();
        let writer = buffer.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_target(false)
            .with_writer(move || writer.clone())
            .finish();
        let value = with_default(subscriber, run);
        let output = String::from_utf8(buffer.0.lock().expect("log buffer lock").clone()).expect("UTF-8 logs");
        (value, output)
    }

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

        let adapter = adapter_error_from_datafusion_error(datafusion_error, TEST_OPERATION, FailedStatement::Other);

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
        assert_eq!(status_source_diagnostics(&status), (Some(parser_display.clone()), Some(parser_display)));
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
        let native = QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.test.server_debug_source",
            "server-native-source-message-secret",
        )
        .with_fact("literal", "server-native-source-fact-secret");
        let native_error = ServerError {
            inner: Error::from_native("algeon_datafusion.test.server_debug_source", "native server wrapper", native),
        };
        let native_source = native_error
            .source()
            .and_then(|source| source.downcast_ref::<QueryError>())
            .expect("native source must remain typed");
        assert_eq!(native_source.fact_value("literal"), Some("server-native-source-fact-secret"));
        let native_debug = format!("{native_error:?}");
        assert!(native_debug.contains("<source omitted>"), "{native_debug}");
        assert!(!native_debug.contains("server-native-source-message-secret"), "{native_debug}");
        assert!(!native_debug.contains("server-native-source-fact-secret"), "{native_debug}");

        let cudf = new_with_for_test(
            cudf::ErrorCode::ForeignInvalidArgument,
            "cudf.test.server_debug_source",
            "server-cudf-source-detail-secret refresh_token=cudf-refresh-secret",
        );
        let cudf_error = ServerError {
            inner: Error::from_cudf("algeon_datafusion.test.server_debug_source", "cudf server wrapper", cudf),
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
        assert!(status.metadata().get("x-algeon-log-classification").is_none());
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
        // This match is an independent statement of the mapping contract, not
        // a call into `adapter_error_to_status`: it must keep failing the same
        // way production code would if the two ever diverge. Because the
        // server now shares the adapter crate, this match is exhaustive and a
        // new kind fails compilation until its Flight mapping is declared.
        fn expected_flight_code(code: ErrorCode) -> Code {
            if code == ErrorCode::AlgorithmDidNotConverge {
                return Code::FailedPrecondition;
            }
            match code.kind() {
                ErrorKind::Cancelled => Code::Cancelled,
                ErrorKind::NotFound => Code::NotFound,
                ErrorKind::PermissionDenied => Code::PermissionDenied,
                ErrorKind::Unauthenticated => Code::Unauthenticated,
                ErrorKind::InvalidInput | ErrorKind::Planning => Code::InvalidArgument,
                ErrorKind::Unsupported => Code::Unimplemented,
                ErrorKind::ResourceExhausted => Code::ResourceExhausted,
                ErrorKind::Execution
                | ErrorKind::DependencyFailure
                | ErrorKind::InvariantViolation
                | ErrorKind::Internal => Code::Internal,
            }
        }

        for &error_code in ErrorCode::ALL {
            let error = Error::new(error_code, TEST_OPERATION, "Flight mapping contract");
            let status = adapter_error_to_status(&error);

            assert_eq!(
                status.code(),
                expected_flight_code(error_code),
                "unexpected Flight status code for {error_code}"
            );

            let metadata = status.metadata();
            let get = |key: &'static str| {
                metadata
                    .get(key)
                    .unwrap_or_else(|| panic!("{error_code}: {key} trailer metadata must be present"))
                    .to_str()
                    .unwrap_or_else(|_| panic!("{error_code}: {key} trailer metadata must be ASCII"))
            };
            assert_eq!(
                get(ERROR_CODE_METADATA_KEY),
                error_code.as_str(),
                "unexpected {ERROR_CODE_METADATA_KEY} for {error_code}"
            );
            assert_eq!(
                get(ERROR_KIND_METADATA_KEY),
                error_code.kind().as_str(),
                "unexpected {ERROR_KIND_METADATA_KEY} for {error_code}"
            );
            assert_eq!(
                get(ERROR_STATUS_METADATA_KEY),
                error_code.status().as_str(),
                "unexpected {ERROR_STATUS_METADATA_KEY} for {error_code}"
            );
            assert!(
                !get(ERROR_OPERATION_METADATA_KEY).is_empty(),
                "{error_code}: {ERROR_OPERATION_METADATA_KEY} trailer metadata must be non-empty"
            );
        }
    }

    #[test]
    fn algorithm_nonconvergence_is_actionable_and_projects_allowlisted_metadata() {
        let error = Error::new(
            ErrorCode::AlgorithmDidNotConverge,
            "algeon_datafusion.cugraph.eigenvector_centrality",
            "native graph algorithm failed",
        )
        .with_fact("algorithm", "eigenvector_centrality")
        .with_fact("iterations", "200")
        .with_fact("max_iterations", "200")
        .with_fact("epsilon", "0.00000001")
        .with_fact("native_secret", "must-not-cross-flight-boundary");

        let status = adapter_error_to_status(&error);

        assert_eq!(status.code(), Code::FailedPrecondition);
        assert_eq!(
            status.message(),
            "eigenvector_centrality did not converge after 200 iterations (max_iterations=200, epsilon=0.00000001); change the convergence parameters and retry"
        );
        assert_metadata_value(&status, ALGORITHM_METADATA_KEY, "eigenvector_centrality");
        assert_metadata_value(&status, ITERATIONS_METADATA_KEY, "200");
        assert_metadata_value(&status, MAX_ITERATIONS_METADATA_KEY, "200");
        assert_metadata_value(&status, EPSILON_METADATA_KEY, "0.00000001");
        assert!(status.metadata().get("x-algeon-native-secret").is_none());
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

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other);

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

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other);

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
            let status = datafusion_error_to_status_for_operation(
                DataFusionError::Plan(message.to_string()),
                TEST_OPERATION,
                FailedStatement::Other,
            );

            assert_eq!(status.code(), Code::InvalidArgument, "{message}");
            assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
            assert_eq!(status.message(), message);
            assert!(!status.message().contains("ALGEON_SERVER_CUGRAPH_ENABLED"));
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
            let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other);

            assert_eq!(status.code(), Code::InvalidArgument, "{message}");
            assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
        }
    }

    #[tokio::test]
    async fn session_context_missing_table_remains_generic_planning_without_a_typed_relation_error() {
        let context = SessionContext::new();
        let error = context
            .sql("SELECT * FROM p39_missing_table")
            .await
            .expect_err("an unregistered relation must fail planning");

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other);

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "datafusion_planning", "planning", "permanent");
    }

    #[test]
    fn datafusion_execution_classification_is_variant_based_not_message_based() {
        for message in [
            "Table 'does_not_exist' doesn't exist.",
            "Parquet error reading /var/lib/algeon/secret/customers.parquet: invalid magic bytes",
        ] {
            let status = datafusion_error_to_status_for_operation(
                DataFusionError::Execution(message.to_string()),
                TEST_OPERATION,
                FailedStatement::Other,
            );

            assert_eq!(status.code(), Code::Internal, "{message}");
            assert_error_metadata(&status, "internal", "internal", "permanent");
            assert_eq!(status.message(), "internal server error while processing the request");
            assert!(!status.message().contains("/var/lib/algeon"));
        }
    }

    #[test]
    fn generic_execution_error_is_genericized_and_does_not_leak_internal_detail() {
        // Execution detail can include local paths or object-store URLs; do not
        // echo it to clients.
        let status = datafusion_error_to_status_for_operation(
            DataFusionError::Execution(
                "Parquet error reading /var/lib/algeon/secret/customers.parquet: invalid magic bytes".to_string(),
            ),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal server error while processing the request");
        assert!(!status.message().contains("/var/lib/algeon"));
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
            FailedStatement::Other,
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

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other);

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
                "algeon_datafusion.native.query_service.admit",
                "native query service queue is full",
            )
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
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
                "algeon_datafusion.native_exec.host_batch_cursor",
                "one Arrow row exceeds the configured host materialization ceiling",
            )
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_error_metadata(&status, "host_materialization_limit_exceeded", "resource_exhausted", "permanent");
        assert!(status.message().contains("host materialization limit"), "{}", status.message());
        assert!(
            !status.message().to_ascii_lowercase().contains("retry later"),
            "permanent limit must not suggest an unchanged retry: {}",
            status.message()
        );
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn resource_failures_name_a_change_and_never_promise_an_unchanged_retry() {
        let cases = [
            (ErrorCode::QueryMemoryLimitExceeded, "query memory cap exceeded"),
            (ErrorCode::QueryBudgetUnavailable, "query budget cannot fit"),
            (ErrorCode::SizeOverflow, "overflowed the platform range"),
            (ErrorCode::HostStagingLimitExceeded, "host staging exceeded"),
            (ErrorCode::FlightFrameTooLarge, "Flight message limit"),
            (ErrorCode::DeviceOutOfMemory, "sealed attempt report"),
            (ErrorCode::DeviceAllocationFailed, "sealed attempt report"),
            (ErrorCode::MemoryReservationFailed, "sealed attempt report"),
            (ErrorCode::CudfResourceExhausted, "sealed attempt report"),
        ];

        for (code, expected) in cases {
            let status = datafusion_error_to_status_for_operation(
                Error::new(code, TEST_OPERATION, "synthetic resource failure").into(),
                TEST_OPERATION,
                FailedStatement::Other,
            );

            assert_eq!(status.code(), Code::ResourceExhausted, "unexpected Flight code for {code}");
            assert!(status.message().contains(expected), "{code} must name its remediation, got: {}", status.message());
            assert!(
                !status.message().to_ascii_lowercase().contains("retry later"),
                "{code} must not suggest an unchanged retry, got: {}",
                status.message()
            );
            assert_no_diagnostic_fields(status.message());
        }
    }

    #[test]
    fn queue_overload_remains_the_only_resource_message_that_suggests_a_retry() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::ServiceOverloaded,
                "algeon_datafusion.native.query_service.admit",
                "native query service queue is full",
            )
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_eq!(status.message(), "service overloaded; retry later");
    }

    #[test]
    fn adapter_server_transport_failure_projects_internal_status() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::ServerTransportFailure,
                "algeon_datafusion.server.transport",
                "Flight SQL transport failed",
            )
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_error_metadata(&status, "server_transport_failure", "dependency_failure", "permanent");
        assert_eq!(status.message(), "internal server error while processing the request");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn captured_adapter_error_keeps_structured_fields_separate_from_source_chain() {
        let error =
            Error::new(ErrorCode::CudfDependencyFailure, "algeon_datafusion.server.test", "adapter wrapper failed")
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

        let captured = datafusion_error_to_captured_error(&error, TEST_OPERATION, FailedStatement::Other);

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

    #[test]
    fn server_error_session_preserves_native_query_error_classification() {
        let native = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.query_service.admit",
            "native query service queue is full",
        )
        .with_fact("reason", "queue_full");

        let error = ServerError::session("query service admission failed", native);

        assert_eq!(error.code(), ErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("native_code"), Some("service_overloaded"));
        assert_eq!(error.fact_value("native_operation"), Some("query_engine.query_service.admit"));
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
                "algeon_datafusion.planner.final_plan_requirement",
                "GPU execution does not support a projected column with an unsupported Arrow type",
            )
            .with_fact("reason", "unsupported_type")
            .with_fact("unsupported_column", "countDetail")
            .with_fact("unsupported_column_type", "List(Decimal128(10, 2))")
            .with_fact("unsupported_column_role", "projected")
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(status.code(), Code::InvalidArgument);
        assert_error_metadata(&status, "final_plan_requirement_unsatisfied", "planning", "permanent");
        assert_eq!(
            status.message(),
            "GPU execution does not support projected column `countDetail` with Arrow type `List(Decimal128(10, 2))`"
        );
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_METADATA_KEY, "countDetail");
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, "List(Decimal128(10, 2))");
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY, "projected");
        assert_metadata_value(&status, REASON_METADATA_KEY, "unsupported_type");
        assert_no_diagnostic_fields(status.message());
    }

    #[test]
    fn adapter_graph_predicate_reason_facts_are_client_visible() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.execute",
                "native cuGraph SQL query execution failed",
            )
            .with_fact("reason", "partial_source_vertices_absent")
            .with_fact("predicate_role", "source_vertices")
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
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
                "algeon_datafusion.cugraph_sql.execute_gpu",
                "native cuGraph SQL GPU-resident query execution failed",
            )
            .with_fact("native_message", "BFS predicate id column dtype must match the graph vertex dtype")
            .with_fact("reason", "predicate_dtype_mismatch")
            .with_fact("predicate_role", "target_vertices")
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
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
                "algeon_datafusion.cugraph_sql.execute_gpu",
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
            FailedStatement::Other,
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

    fn strict_rejection_error(column: &str, column_type: &str) -> Error {
        Error::new(
            ErrorCode::FinalPlanRequirementUnsatisfied,
            "algeon_datafusion.planner.final_plan_requirement",
            "GPU execution does not support a projected column with an unsupported Arrow type",
        )
        .with_log_safe_fact("reason", "unsupported_type")
        .with_fact("unsupported_column", column)
        .with_fact("unsupported_column_type", column_type)
        .with_fact("unsupported_column_role", "projected")
    }

    #[test]
    fn strict_unsupported_column_projections_distinguish_message_metadata_logs_and_stats() {
        // ASCII short detail: every client projection carries it, logs do not.
        let ascii = strict_rejection_error("countDetail", "List(Decimal128(10, 2))");
        let status = adapter_error_to_status(&ascii);
        assert!(status.message().contains("countDetail"), "{}", status.message());
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_METADATA_KEY, "countDetail");
        assert!(status.metadata().get(OMITTED_FACTS_METADATA_KEY).is_none());
        let logs = adapter_status_log_facts(&ascii).rendered.as_deref().unwrap_or_default().to_string();
        assert!(!logs.contains("countDetail"), "{logs}");
        let captured = CapturedError::from_adapter_error(&ascii);
        assert!(captured.facts.contains(&("unsupported_column".to_string(), "countDetail".to_string())));

        // Non-ASCII detail: the message and query-stats capture keep it, the
        // ASCII-only metadata omits it and says so.
        let unicode = strict_rejection_error("訂單明細", "List(Int64)");
        let status = adapter_error_to_status(&unicode);
        assert!(status.message().contains("訂單明細"), "{}", status.message());
        assert!(status.metadata().get(UNSUPPORTED_COLUMN_METADATA_KEY).is_none());
        assert_metadata_value(&status, OMITTED_FACTS_METADATA_KEY, "unsupported_column");
        let logs = adapter_status_log_facts(&unicode).rendered.as_deref().unwrap_or_default().to_string();
        assert!(!logs.contains("訂單明細"), "{logs}");
        let captured = CapturedError::from_adapter_error(&unicode);
        assert!(captured.facts.contains(&("unsupported_column".to_string(), "訂單明細".to_string())));

        // Over-long ASCII detail: the message keeps it whole, metadata and
        // capture truncate it with a marker.
        let long_column = "x".repeat(300);
        let long = strict_rejection_error(&long_column, "List(Int64)");
        let status = adapter_error_to_status(&long);
        assert!(status.message().contains(&long_column), "{}", status.message().len());
        let wire = status
            .metadata()
            .get(UNSUPPORTED_COLUMN_METADATA_KEY)
            .expect("long ASCII detail stays on the wire truncated")
            .to_str()
            .expect("wire values stay ASCII")
            .to_string();
        assert_eq!(wire.len(), MAX_CLIENT_FACT_METADATA_VALUE_BYTES, "{wire}");
        assert!(wire.ends_with("..."), "{wire}");
        let captured = CapturedError::from_adapter_error(&long);
        let retained = captured
            .facts
            .iter()
            .find(|(key, _)| key == "unsupported_column")
            .expect("capture keeps the truncated detail")
            .1
            .clone();
        assert!(retained.len() <= MAX_CLIENT_FACT_METADATA_VALUE_BYTES, "{retained}");
        assert!(retained.ends_with("..."), "{retained}");
        assert!(retained.starts_with(&"x".repeat(200)), "{retained}");
        assert!(captured.facts.contains(&("error_facts_truncated".to_string(), "true".to_string())));

        // URI-like detail: metadata carries the ASCII wire value raw while
        // retained capture sanitizes it like every other retained fact.
        let uri = strict_rejection_error("s3://bucket/secret/key", "Utf8");
        let status = adapter_error_to_status(&uri);
        assert_metadata_value(&status, UNSUPPORTED_COLUMN_METADATA_KEY, "s3://bucket/secret/key");
        let captured = CapturedError::from_adapter_error(&uri);
        assert!(captured.facts.contains(&("unsupported_column".to_string(), "<redacted-uri>".to_string())));
    }

    #[test]
    fn deep_wrapper_classification_terminates_at_traversal_budget() {
        let mut error =
            DataFusionError::SQL(Box::new(ParserError::ParserError("Expected: an SQL statement".into())), None);
        for depth in 0..40 {
            error = DataFusionError::Context(format!("wrapper layer {depth}"), Box::new(error));
        }

        assert_eq!(classify(&error, FailedStatement::Other), ErrorCode::Internal);

        let mut shallow =
            DataFusionError::SQL(Box::new(ParserError::ParserError("Expected: an SQL statement".into())), None);
        for depth in 0..3 {
            shallow = DataFusionError::Context(format!("wrapper layer {depth}"), Box::new(shallow));
        }
        assert_eq!(classify(&shallow, FailedStatement::Other), ErrorCode::SqlParse);
    }

    #[test]
    fn classify_maps_object_store_not_found_to_source_object_not_found() {
        let not_found =
            || ObjectStoreError::NotFound { path: "bucket/missing.parquet".to_string(), source: "absent".into() };
        let direct = DataFusionError::ObjectStore(Box::new(not_found()));
        let wrapped = DataFusionError::Context(
            "scan".to_string(),
            Box::new(DataFusionError::Shared(Arc::new(DataFusionError::External(Box::new(not_found()))))),
        );
        let through_arrow =
            DataFusionError::ArrowError(Box::new(ArrowError::ExternalError(Box::new(not_found()))), None);

        for error in [direct, wrapped, through_arrow] {
            assert_eq!(classify(&error, FailedStatement::Other), ErrorCode::SourceObjectNotFound, "{error:?}");
        }

        let status = datafusion_error_to_status_for_operation(
            DataFusionError::ObjectStore(Box::new(not_found())),
            TEST_OPERATION,
            FailedStatement::Other,
        );
        assert_eq!(status.code(), Code::NotFound);
        assert_error_metadata(&status, "source_object_not_found", "not_found", "permanent");

        let denied = ObjectStoreError::PermissionDenied { path: "bucket/key".to_string(), source: "denied".into() };
        assert_eq!(
            classify(&DataFusionError::ObjectStore(Box::new(denied)), FailedStatement::Other),
            ErrorCode::SourceAccessDenied
        );
        let unauthenticated =
            ObjectStoreError::Unauthenticated { path: "bucket/key".to_string(), source: "no credentials".into() };
        assert_eq!(
            classify(&DataFusionError::ObjectStore(Box::new(unauthenticated)), FailedStatement::Other),
            ErrorCode::SourceUnauthenticated
        );
    }

    #[test]
    fn classify_maps_resources_exhausted_to_host_memory() {
        let exhausted = || DataFusionError::ResourcesExhausted("Failed to allocate 1024 bytes".to_string());
        let wrapped = DataFusionError::Context("collect".to_string(), Box::new(exhausted()));
        let through_arrow =
            DataFusionError::ArrowError(Box::new(ArrowError::ExternalError(Box::new(exhausted()))), None);

        for error in [exhausted(), wrapped, through_arrow] {
            assert_eq!(classify(&error, FailedStatement::Other), ErrorCode::HostMemoryExhausted, "{error:?}");
        }

        let status = datafusion_error_to_status_for_operation(exhausted(), TEST_OPERATION, FailedStatement::Other);
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_error_metadata(&status, "host_memory_exhausted", "resource_exhausted", "temporary");
        assert!(status.message().starts_with("host memory pool exhausted"), "{}", status.message());
        assert!(!status.message().contains("device memory"), "{}", status.message());
    }

    #[tokio::test]
    async fn classify_maps_cancelled_execution_join_to_cancelled() {
        let task = tokio::spawn(std::future::pending::<()>());
        task.abort();
        let cancelled = DataFusionError::ExecutionJoin(Box::new(task.await.unwrap_err()));

        assert_eq!(classify(&cancelled, FailedStatement::Other), ErrorCode::Cancelled);
    }

    #[test]
    fn native_required_error_message_omits_column_identifiers() {
        const COLUMN: &str = "sentinel_customer_column";
        const COLUMN_TYPE: &str = "List(Int64)";
        let mut report = PlanningReport::new();
        report.push_not_supported_with_detail(
            GpuCandidate::new("ParquetExec", 1),
            CapabilityReason::UnsupportedType,
            NotSupportedDetail {
                column: Some(UnsupportedColumnDetail::new(COLUMN, COLUMN_TYPE, "projected")),
                expression: None,
            },
        );
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let error =
            native::final_plan_requirement_error(plan.as_ref(), FinalPlanDisposition::DataFusion, Some(&report));

        assert!(!error.message().contains(COLUMN), "{}", error.message());
        assert_eq!(error.fact_value("unsupported_column"), Some(COLUMN));
        assert!(!error.facts().log_safe_projection().rendered().unwrap_or_default().contains(COLUMN));

        let (status, logs) = with_captured_logs(|| adapter_error_to_status(&error));

        assert!(logs.contains("converted adapter error to Flight SQL status"), "{logs}");
        assert!(!logs.contains(COLUMN), "{logs}");
        assert!(!logs.contains(COLUMN_TYPE), "{logs}");
        assert_eq!(
            status.message(),
            format!("GPU execution does not support projected column `{COLUMN}` with Arrow type `{COLUMN_TYPE}`")
        );
    }

    #[test]
    fn dense_relation_unsupported_type_keeps_field_identifiers_out_of_message_and_logs() {
        const FIELD: &str = "sentinel_feature_column";
        const ACTUAL_TYPE: &str = "List(Field { name: \"sentinel_item\" })";
        let context = DenseRelationErrorContext {
            family: "cuML",
            operation_fact: "cuml_operation",
            validate_operation: "algeon_datafusion.cuml_sql.validate",
            validate_schema_operation: "algeon_datafusion.cuml_sql.validate_schema",
            execute_operation: "algeon_datafusion.cuml_sql.execute",
            required_feature: "cuml",
            feature_disabled_message: "cuML algorithm execution requires the cuml Cargo feature",
        };
        let error = context.unsupported_type("cuml_tsvd", "input", FIELD, ACTUAL_TYPE, "Float32");
        let adapter = error.find_algeon_error().expect("typed adapter error");

        assert_eq!(adapter.code(), ErrorCode::UnsupportedType);
        assert!(!adapter.message().contains(FIELD), "{}", adapter.message());
        assert!(adapter.message().contains("Float32"), "{}", adapter.message());
        assert_eq!(adapter.fact_value("field"), Some(FIELD));
        assert_eq!(adapter.fact_value("actual_type"), Some(ACTUAL_TYPE));

        let (status, logs) = with_captured_logs(|| {
            datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other)
        });

        assert!(logs.contains("converted adapter error to Flight SQL status"), "{logs}");
        assert!(!logs.contains(FIELD), "{logs}");
        assert!(!logs.contains("sentinel_item"), "{logs}");
        assert_eq!(status.code(), Code::Unimplemented);
        assert!(status.message().contains(FIELD), "{}", status.message());
        assert!(status.message().contains(ACTUAL_TYPE), "{}", status.message());
    }

    #[test]
    fn extreme_foreign_message_is_bounded_on_the_wire_but_kept_in_process() {
        let error = DataFusionError::SQL(Box::new(ParserError::ParserError("y".repeat(100_000))), None);
        assert!(error.to_string().len() > 100_000);

        let captured = datafusion_error_to_captured_error(&error, TEST_OPERATION, FailedStatement::Other);
        assert!(captured.message.len() <= MAX_CLIENT_DATAFUSION_MESSAGE_BYTES, "{}", captured.message.len());
        assert!(captured.message.ends_with("..."), "{}", &captured.message[captured.message.len() - 10..]);

        let status = datafusion_error_to_status_for_operation(error, TEST_OPERATION, FailedStatement::Other);
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().len() <= MAX_CLIENT_DATAFUSION_MESSAGE_BYTES, "{}", status.message().len());
        assert!(status.message().ends_with("..."));
    }

    #[test]
    fn adapter_graph_execution_failure_stays_generic() {
        // An internal/execution-kind graph failure must not surface the native
        // message anywhere on the client projection: the status message is
        // generic and the `native_message` trailer is withheld for
        // non-client-actionable kinds. Raw native text stays trusted
        // in-process diagnosis only.
        let status = datafusion_error_to_status_for_operation(
            Error::new(
                ErrorCode::Internal,
                "algeon_datafusion.cugraph_sql.execute_gpu",
                "native cuGraph SQL GPU-resident query execution failed",
            )
            .with_fact("native_message", "internal cuGraph scheduler state corruption")
            .with_fact("reason", "internal_scheduler_state")
            .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal server error while processing the request");
        assert!(!status.message().contains("internal cuGraph scheduler state"));
        assert!(!status.message().contains("reason="));
        assert!(status.metadata().get(NATIVE_MESSAGE_METADATA_KEY).is_none());
    }

    #[test]
    fn internal_native_message_is_withheld_from_flight_metadata_but_actionable_kinds_keep_it() {
        let internal = datafusion_error_to_status_for_operation(
            Error::new(ErrorCode::Internal, TEST_OPERATION, "native execution failed")
                .with_fact("native_message", "internal scheduler state corruption marker")
                .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_eq!(internal.message(), "internal server error while processing the request");
        assert!(internal.metadata().get(NATIVE_MESSAGE_METADATA_KEY).is_none());

        let actionable = datafusion_error_to_status_for_operation(
            Error::new(ErrorCode::UnsupportedType, TEST_OPERATION, "native validation failed")
                .with_fact("native_message", "BFS predicate id column dtype must match the graph vertex dtype")
                .into(),
            TEST_OPERATION,
            FailedStatement::Other,
        );

        assert_metadata_value(
            &actionable,
            NATIVE_MESSAGE_METADATA_KEY,
            "BFS predicate id column dtype must match the graph vertex dtype",
        );
    }

    #[test]
    fn adapter_non_graph_reason_fact_stays_generic_for_internal_errors() {
        let status = datafusion_error_to_status_for_operation(
            Error::new(ErrorCode::Internal, "algeon_datafusion.native.execution", "native execution failed")
                .with_fact("reason", "internal_scheduler_state")
                .into(),
            TEST_OPERATION,
            FailedStatement::Other,
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
