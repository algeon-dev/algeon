//! Client-visible projection of adapter errors.
//!
//! The Flight SQL status boundary and the rejected `EXPLAIN GPU` summary show
//! a query caller the same code and text for one failed statement, so both read
//! them from here. The status message of a blocking plan element adds a pointer
//! to `EXPLAIN GPU`, which the summary omits. Text of the failure kinds whose
//! raw message stays on the server is replaced by a static message, and caller
//! schema identifiers join only the wire text, never the logged one.

use datafusion::common::DataFusionError;
use query_planning::error::bytes_with_gib;

use super::{
    ClientDetail, DataFusionErrorExt, Error, ErrorCode, ErrorKind, FailedStatement, classify,
    datafusion_error_human_message, truncate_with_marker,
};

/// Operation of the temporary error that projects a foreign DataFusion error without an Algeon identity.
const FOREIGN_FAILURE_OPERATION: &str = "algeon_datafusion.error.client_failure";

/// Status message of the failure kinds whose raw text stays server-side. The
/// query id travels in the `x-algeon-query-id` metadata of the same status.
pub(crate) const MASKED_INTERNAL_ERROR_MESSAGE: &str = "internal server error while processing the request; the full diagnostic is available through the GetQueryStats action for the query id in the x-algeon-query-id metadata";

/// Maximum bytes for a client status message derived from foreign
/// DataFusion text. Bounds what one extreme upstream `Display` can emit onto
/// the wire; the in-process error is never rewritten.
pub(crate) const MAX_CLIENT_DATAFUSION_MESSAGE_BYTES: usize = 2_048;

/// Maximum depth for DataFusion message traversal. Mirrors the
/// adapter chain budget used by `classify`: extraction always terminates,
/// returning the deepest node reached when the budget is exhausted.
const MAX_SERVER_DATAFUSION_TRAVERSAL_DEPTH: usize = 32;

/// Closing clause of the status message that names a blocking plan element. It
/// sends a caller who only has the status to the report, so the report omits it.
const BLOCKING_SUBJECT_POINTER: &str = "EXPLAIN GPU lists every blocking candidate";

/// The texts a query caller sees for one failure.
pub(crate) struct ClientMessage {
    /// The text without caller schema identifiers, which the boundary log may record.
    #[cfg_attr(not(feature = "server"), expect(dead_code, reason = "only the Flight boundary logs the message"))]
    pub(crate) logged: String,
    /// The failure text a caller receives: `logged` plus the caller schema
    /// detail the error names, without the closing pointer to `EXPLAIN GPU`.
    pub(crate) text: String,
    #[cfg_attr(not(feature = "server"), expect(dead_code, reason = "only the Flight boundary appends the pointer"))]
    pointer: Option<&'static str>,
}

impl ClientMessage {
    /// The message of the Flight status: `text` and, for a blocking plan
    /// element, the pointer to `EXPLAIN GPU`.
    #[cfg_attr(not(feature = "server"), expect(dead_code, reason = "only the Flight boundary builds a status"))]
    pub(crate) fn status_message(&self) -> String {
        match self.pointer {
            Some(pointer) => format!("{}; {pointer}", self.text),
            None => self.text.clone(),
        }
    }
}

/// The texts the status boundary shows a query caller for `error`.
pub(crate) fn client_message(error: &Error) -> ClientMessage {
    let logged = client_status_message(error);
    let detail = client_actionable_kind(error).then(|| error.client_detail()).flatten();
    let text = detail.map_or_else(|| logged.clone(), |detail| client_schema_detail_message(detail, &logged));
    let pointer = matches!(detail, Some(ClientDetail::BlockingSubject { .. })).then_some(BLOCKING_SUBJECT_POINTER);
    ClientMessage { logged, text, pointer }
}

/// The code and the text a query caller sees when `error` fails a statement.
/// The text is the Flight status message without its pointer to `EXPLAIN GPU`.
pub(crate) fn client_failure(error: &DataFusionError, statement: FailedStatement) -> (ErrorCode, String) {
    if let Some(adapter) = error.find_algeon_error() {
        return (adapter.code(), client_message(adapter).text);
    }
    let code = classify(error, statement);
    let foreign = Error::new(code, FOREIGN_FAILURE_OPERATION, client_datafusion_error_message(error));
    (code, client_message(&foreign).text)
}

fn client_status_message(error: &Error) -> String {
    match error.code() {
        // The native layer that knows the algorithm names the parameters to change in `remedy`.
        ErrorCode::AlgorithmDidNotConverge => format!(
            "{} did not converge after {} iterations (max_iterations={}, epsilon={}); {} and retry",
            error.fact_value("algorithm").unwrap_or("native algorithm"),
            error.fact_value("iterations").unwrap_or("unknown"),
            error.fact_value("max_iterations").unwrap_or("unknown"),
            error.fact_value("epsilon").unwrap_or("unknown"),
            error.fact_value("remedy").unwrap_or("change the convergence parameters"),
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
            "query memory cap exceeded{}. Reduce what the failing operation holds (filter or project earlier, split the query) or raise the query memory cap; resubmitting the request unchanged fails again",
            memory_failure_detail(error).unwrap_or_else(|| format!(": {}", error.message()))
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
        ErrorCode::DeviceOutOfMemory | ErrorCode::DeviceAllocationFailed => format!(
            "device memory exhausted{}. Reduce what the failing operation holds (filter or project earlier, split the query) and consult the sealed attempt report for retry guidance; resubmitting the request unchanged may fail again",
            memory_failure_detail(error)
                .unwrap_or_else(|| format!(" while processing the request: {}", error.message()))
        ),
        ErrorCode::MemoryReservationFailed => format!(
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
            MASKED_INTERNAL_ERROR_MESSAGE.to_string()
        },
        _ => error.message().to_string(),
    }
}

/// Names the native operation and the allocator facts of a memory failure:
/// the requested bytes, the allocation peak, and the query grant. The peak
/// stands in for the bytes held when the allocation failed. A current-usage
/// sample is taken after the failing operation released its working set, so
/// it would not add up with the request against the grant. The native layers
/// copy these numbers and operation identifiers onto the error; no foreign
/// text enters the message. Returns `None` when the error carries none of
/// them.
fn memory_failure_detail(error: &Error) -> Option<String> {
    let bytes = |key: &str| error.fact_value(key).and_then(|value| value.parse::<u64>().ok());
    let operation = error.fact_value("native_operation").or_else(|| error.fact_value("budget_operation"));
    let clauses = [
        bytes("requested_allocation_bytes").map(|value| format!("it requested {}", bytes_with_gib(value))),
        bytes("budget_peak_bytes").map(|value| format!("the allocation peak was {}", bytes_with_gib(value))),
        bytes("budget_bytes").map(|value| format!("the query grant is {}", bytes_with_gib(value))),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let location = operation.map(|operation| format!(" in {operation}"));
    if clauses.is_empty() && location.is_none() {
        return None;
    }
    let figures = match clauses.as_slice() {
        [] => String::new(),
        [only] => format!(": {only}"),
        [first, second] => format!(": {first} and {second}"),
        [init @ .., last] => format!(": {}, and {last}", init.join(", ")),
    };
    Some(format!("{}{figures}", location.unwrap_or_default()))
}

/// Client text for rejections that name the caller's own schema identifiers.
///
/// Owners keep column and field names and their Arrow types out of
/// `Error::message()`, which the boundary log records verbatim, and attach
/// them as a [`ClientDetail`]. The text is composed here, so the identifiers
/// reach the query caller and never the log. Kinds whose client text is
/// generic keep it generic.
fn client_schema_detail_message(detail: &ClientDetail, client_message: &str) -> String {
    match detail {
        ClientDetail::UnsupportedColumn { name, data_type, role } => {
            format!("GPU execution does not support {role} column `{name}` with Arrow type `{data_type}`")
        },
        ClientDetail::RelationField { field, data_type } => {
            format!("{client_message} (field '{field}' has Arrow type `{data_type}`)")
        },
        ClientDetail::Field { name } => format!("{client_message} (field '{name}')"),
        ClientDetail::Column { name } => format!("{client_message} (column '{name}')"),
        ClientDetail::OrderingKey { name } => format!("{client_message} (ORDER BY key '{name}')"),
        ClientDetail::BlockingSubject { text } => format!("GPU execution does not support {text}"),
    }
}

/// Whether the error's kind is a client-actionable validation kind, whose
/// caller detail and raw native diagnostic text (the `native_message`
/// trailer) may cross the Flight boundary.
///
/// `native_message` remains trusted in-process diagnosis, but its key being
/// allowlisted must not by itself publish raw text: internal, execution,
/// dependency, invariant, and resource failures stay generic on the wire.
pub(crate) fn client_actionable_kind(error: &Error) -> bool {
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

pub(crate) fn client_datafusion_error_message(error: &DataFusionError) -> String {
    let effective = effective_datafusion_error(error);
    let human_message = datafusion_error_human_message(effective);
    let message = match effective {
        DataFusionError::SQL(_, _) => format!("invalid SQL statement: {human_message}"),
        _ => human_message,
    };
    // Foreign upstream text enters the client message here; cap what one
    // extreme `Display` can emit onto the wire. The in-process error keeps its
    // full text for trusted diagnosis.
    truncate_with_marker(message.as_str(), MAX_CLIENT_DATAFUSION_MESSAGE_BYTES).0
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
