//! The truthful error contract for `cudf-rs`.
//!
//! `Error` stores one authoritative failure identity: [`ErrorCode`]. Both
//! [`ErrorKind`] and [`ErrorStatus`] are pure projections of `code` — they are
//! never stored independently, and there are no override APIs.
//!
//! The public surface is read-only: construction and mutation helpers are
//! crate-internal (`pub(crate)`).

use std::{borrow::Cow, error::Error as StdError, ffi::CStr, fmt, ptr::NonNull};

use rapids_interop::{InteropError, InteropErrorKind};

pub mod code;
pub mod facts;
pub(crate) mod testing_hooks;

pub use code::{ErrorCode, ErrorKind, ErrorStatus, kind_for_code, status_for_code};
pub(crate) use facts::{
    DeviceIdFact, FACT_ACTUAL_COUNT, FACT_ACTUAL_ROWS, FACT_COLUMN_INDEX, FACT_COLUMN_NAME, FACT_DISTINCT_KEYS_COUNT,
    FACT_EXPECTED_COUNT, FACT_EXPECTED_ROWS, FACT_KEYS_COUNT, FACT_LEFT_DEVICE, FACT_LEFT_TYPE_ID, FACT_RIGHT_DEVICE,
    FACT_RIGHT_TYPE_ID, FACT_TABLE_COLUMN_COUNT, FACT_TABLE_ROW_COUNT, FACT_TYPE_ID, FactKey, IntoFactValue,
};
pub use facts::{FactKeyName, FactValue, Facts};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

/// A cudf-rs error. See the module-level docs for the contract.
pub struct Error {
    inner: Box<ErrorInner>,
}

struct ErrorInner {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    details: Vec<(&'static str, String)>,
    budget_exhaustion: Option<BudgetExhaustionDetail>,
    call_chain: Vec<&'static str>,
    source: Option<BoxedSourceError>,
}

/// Typed budget-exhaustion counters attached by budget-token execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetExhaustionDetail {
    hard_budget_bytes: usize,
    peak_bytes: usize,
    peak_aligned_bytes: usize,
    current_bytes: usize,
    largest_failed_allocation_bytes: usize,
    failed_allocation_count: usize,
    boundary: Cow<'static, str>,
    operation: &'static str,
}

impl BudgetExhaustionDetail {
    pub(crate) fn new(
        hard_budget_bytes: usize,
        peak_bytes: usize,
        peak_aligned_bytes: usize,
        current_bytes: usize,
        largest_failed_allocation_bytes: usize,
        failed_allocation_count: usize,
        boundary: impl Into<Cow<'static, str>>,
        operation: &'static str,
    ) -> Self {
        Self {
            hard_budget_bytes,
            peak_bytes,
            peak_aligned_bytes,
            current_bytes,
            largest_failed_allocation_bytes,
            failed_allocation_count,
            boundary: boundary.into(),
            operation,
        }
    }

    pub fn hard_budget_bytes(&self) -> usize {
        self.hard_budget_bytes
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    pub fn peak_aligned_bytes(&self) -> usize {
        self.peak_aligned_bytes
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    pub fn largest_failed_allocation_bytes(&self) -> usize {
        self.largest_failed_allocation_bytes
    }

    pub fn failed_allocation_count(&self) -> usize {
        self.failed_allocation_count
    }

    pub fn boundary(&self) -> &str {
        &self.boundary
    }

    pub fn operation(&self) -> &'static str {
        self.operation
    }
}

impl Error {
    // -- Construction (crate-internal) --------------------------------------

    /// Create a new error with the full contract: a leaf `code`, a cudf-rs
    /// boundary `operation`, and a primary `message`.
    ///
    /// `operation` must be a non-empty `&'static str`. `message` must be
    /// non-empty. These are programmer errors inside cudf-rs, so they panic
    /// rather than return `Result`.
    pub(crate) fn new_with(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "cudf-rs internal error: operation label must not be empty");
        assert!(!message.is_empty(), "cudf-rs internal error: message must not be empty");
        Self {
            inner: Box::new(ErrorInner {
                code,
                operation,
                message,
                facts: Facts::new(),
                details: Vec::new(),
                budget_exhaustion: None,
                call_chain: Vec::new(),
                source: None,
            }),
        }
    }

    /// Internal: insert a typed fact, panicking on duplicate or wrong pairing.
    #[must_use]
    pub(crate) fn with_fact<T>(mut self, key: FactKey<T>, value: T) -> Self
    where T: IntoFactValue {
        self.inner.facts.insert(key.name(), value.into_fact_value());
        self
    }

    /// Internal: append the canonical column-name fact.
    #[must_use]
    pub(crate) fn with_column(self, name: impl Into<Cow<'static, str>>) -> Self {
        self.with_fact(FACT_COLUMN_NAME, name.into())
    }

    /// Internal: append a diagnostic detail. Repeated keys are allowed.
    ///
    /// Panics if `key` is a canonical fact snake_case name — such keys must be
    /// written via [`with_fact`](Self::with_fact), never through the detail
    /// path. The check is structural so overlap-rule violations cannot be
    /// introduced by reviewer inattention.
    #[must_use]
    pub(crate) fn with_detail(mut self, key: &'static str, value: impl Into<String>) -> Self {
        assert_canonical_key_not_used_as_detail(key);
        self.inner.details.push((key, value.into()));
        self
    }

    /// Internal: append a typed fact only when that canonical fact is absent.
    #[must_use]
    pub(crate) fn with_fact_if_absent<T>(self, key: FactKey<T>, value: T) -> Self
    where T: IntoFactValue {
        if self.fact_value(key.name()).is_some() { self } else { self.with_fact(key, value) }
    }

    /// Internal: append a diagnostic detail only when that detail key is
    /// absent.
    #[must_use]
    pub(crate) fn with_detail_if_absent(self, key: &'static str, value: impl Into<String>) -> Self {
        if self.detail_value(key).is_some() { self } else { self.with_detail(key, value) }
    }

    /// Internal: append several diagnostic details.
    ///
    /// Panics if any `key` is a canonical fact snake_case name. See
    /// [`with_detail`](Self::with_detail).
    #[must_use]
    pub(crate) fn with_details<I>(mut self, details: I) -> Self
    where I: IntoIterator<Item = (&'static str, String)> {
        let details = details.into_iter();
        self.inner.details.reserve(details.size_hint().0);
        for (key, value) in details {
            assert_canonical_key_not_used_as_detail(key);
            self.inner.details.push((key, value));
        }
        self
    }

    /// Internal: attach typed budget-exhaustion counters if absent.
    #[must_use]
    pub(crate) fn with_budget_exhaustion_if_absent(mut self, detail: BudgetExhaustionDetail) -> Self {
        if self.inner.budget_exhaustion.is_none() {
            self.inner.budget_exhaustion = Some(detail);
        }
        self
    }

    /// Internal: attach a source error.
    #[must_use]
    pub(crate) fn with_source(mut self, src: impl Into<BoxedSourceError>) -> Self {
        self.inner.source = Some(src.into());
        self
    }

    /// Internal: re-anchor this error at a new boundary.
    ///
    /// The previous primary operation is appended to `call_chain` as-is with
    /// no deduplication. All other fields are preserved.
    #[must_use]
    pub(crate) fn reanchor(mut self, operation: &'static str) -> Self {
        assert!(!operation.is_empty(), "cudf-rs internal error: operation label must not be empty");
        self.inner.call_chain.push(self.inner.operation);
        self.inner.operation = operation;
        self
    }

    /// Internal: re-anchor this error only when the requested operation
    /// differs from the current anchor.
    #[must_use]
    pub(crate) fn reanchor_if_needed(self, operation: &'static str) -> Self {
        if self.operation() == operation { self } else { self.reanchor(operation) }
    }

    // -- Public read accessors ----------------------------------------------

    /// The stable machine-readable failure identity.
    pub fn code(&self) -> ErrorCode {
        self.inner.code
    }

    /// The coarse failure category (pure projection of `code`).
    pub fn kind(&self) -> ErrorKind {
        kind_for_code(self.inner.code)
    }

    /// The retry semantic (pure projection of `code`).
    pub fn status(&self) -> ErrorStatus {
        status_for_code(self.inner.code)
    }

    /// The cudf-rs boundary at which this error was surfaced.
    pub fn operation(&self) -> &'static str {
        self.inner.operation
    }

    /// The one-line primary failure description.
    pub fn message(&self) -> &str {
        &self.inner.message
    }

    /// Canonical typed facts.
    pub fn facts(&self) -> &Facts {
        &self.inner.facts
    }

    /// The value of a canonical fact, if present.
    pub fn fact_value(&self, key: FactKeyName) -> Option<&FactValue> {
        self.inner.facts.get(key)
    }

    /// Open-ended diagnostic annotations (may contain repeated keys).
    pub fn details(&self) -> &[(&'static str, String)] {
        &self.inner.details
    }

    /// The first detail value recorded for `key`, if any.
    pub fn detail_value(&self, key: &str) -> Option<&str> {
        self.inner.details.iter().find_map(|(k, v)| (*k == key).then_some(v.as_str()))
    }

    /// All detail values recorded for `key`, in insertion order.
    pub fn detail_values<'a>(&'a self, key: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.inner.details.iter().filter_map(move |(k, v)| (*k == key).then_some(v.as_str()))
    }

    /// Typed budget-exhaustion counters attached by a hard budget token.
    pub fn budget_exhaustion(&self) -> Option<&BudgetExhaustionDetail> {
        self.inner.budget_exhaustion.as_ref()
    }

    /// Ordered trail of prior primary operations this error has been
    /// re-anchored from. No deduplication.
    pub fn call_chain(&self) -> &[&'static str] {
        &self.inner.call_chain
    }

    /// Underlying error cause, if any.
    pub fn source_error(&self) -> Option<&(dyn StdError + 'static)> {
        self.inner.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }

    /// Stable diagnostic string for logs and query-engine adapters.
    ///
    /// This formats the existing error contract without introducing parallel
    /// stored kind/status fields or string-only categories. Callers that need
    /// policy decisions should keep using [`code`](Self::code),
    /// [`kind`](Self::kind), [`status`](Self::status), and
    /// [`facts`](Self::facts) directly.
    pub fn diagnostic_string(&self) -> String {
        let mut out = format!(
            "code={} kind={} status={} operation={} message={}",
            self.code(),
            self.kind(),
            self.status(),
            self.operation(),
            diagnostic_quote(self.message()),
        );

        out.push_str(" facts={");
        for (index, (key, value)) in self.facts().iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            out.push_str(key.as_snake_case());
            out.push('=');
            out.push_str(&value.to_string());
        }
        out.push('}');

        out.push_str(" details=[");
        for (index, (key, value)) in self.details().iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            out.push_str(key);
            out.push('=');
            out.push_str(&diagnostic_quote(value));
        }
        out.push(']');

        out.push_str(" call_chain=[");
        for (index, operation) in self.call_chain().iter().enumerate() {
            if index > 0 {
                out.push_str(" -> ");
            }
            out.push_str(operation);
        }
        out.push(']');

        out.push_str(" source=");
        match self.source_error() {
            Some(source) => out.push_str(&diagnostic_quote(&source.to_string())),
            None => out.push_str("none"),
        }

        out
    }

    // -- Status/identity helpers (exact aliases only) -----------------------

    /// Exact alias: `status() == Temporary`.
    pub fn is_retryable(&self) -> bool {
        self.status() == ErrorStatus::Temporary
    }

    /// Exact alias: `status() == Temporary`.
    pub fn is_temporary(&self) -> bool {
        self.status() == ErrorStatus::Temporary
    }

    /// Exact alias: `status() == Permanent`.
    pub fn is_permanent(&self) -> bool {
        self.status() == ErrorStatus::Permanent
    }

    /// Exact alias: `status() == Persistent`.
    pub fn is_persistent(&self) -> bool {
        self.status() == ErrorStatus::Persistent
    }

    /// Exact alias: `code() == ErrorCode::OutOfMemory`.
    ///
    /// This is the single domestic OOM leaf. Foreign resource exhaustion is a
    /// different leaf and does not alias here.
    pub fn is_oom(&self) -> bool {
        self.inner.code == ErrorCode::OutOfMemory
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format: "{kind} ({status}) at {operation}: {message}[; k: v; ...][; source: {src}]"
        write!(f, "{} ({}) at {}: {}", self.kind(), self.status(), self.inner.operation, self.inner.message)?;
        for (k, v) in &self.inner.details {
            write!(f, "; {}: {}", k, v)?;
        }
        if let Some(src) = &self.inner.source {
            write!(f, "; source: {}", src)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "code: {}", self.inner.code)?;
        writeln!(f, "kind: {} (derived)", self.kind())?;
        writeln!(f, "status: {}", self.status())?;
        writeln!(f, "operation: {}", self.inner.operation)?;
        writeln!(f, "call_chain: {:?}", self.inner.call_chain)?;
        writeln!(f, "facts: {}", self.inner.facts)?;
        for (k, v) in &self.inner.details {
            writeln!(f, "{}: {}", k, v)?;
        }
        writeln!(f, "message: {}", self.inner.message)?;
        writeln!(f, "source: {:?}", self.inner.source)?;
        Ok(())
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

// --------------------------------------------------------------------------
// Internal helpers
// --------------------------------------------------------------------------

/// Reject canonical fact snake_case names on the detail write path.
///
/// Canonical snake_case keys are reserved for `facts` and must never appear
/// in `details`. This is enforced structurally at the writer so the boundary
/// cannot be violated by reviewer inattention. A violation is a crate-internal
/// bug and panics in every build profile.
fn assert_canonical_key_not_used_as_detail(key: &'static str) {
    if FactKeyName::from_snake_case(key).is_some() {
        panic!(
            "cudf-rs internal error: canonical fact key `{key}` written through the detail path; \
             use `with_fact` instead",
        );
    }
}

fn diagnostic_quote(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('"');
    for ch in input.chars() {
        out.extend(ch.escape_default());
    }
    out.push('"');
    out
}

/// Build an `InvariantViolation` error.
pub(crate) fn invariant_violation(operation: &'static str, message: impl Into<String>) -> Error {
    Error::new_with(ErrorCode::InvariantViolation, operation, message)
}

/// Build an error describing an interior null byte.
///
/// Emits the leaf code [`ErrorCode::NullUnexpected`]. See that variant's
/// doc comment for the full contract: `NullUnexpected` covers both
/// SQL-level null values (mask bit null where not allowed) and interior
/// null bytes in string inputs (a `\0` byte in the middle of a string
/// that cannot survive a C null-terminated contract). Downstreams that
/// key on this code handle both by treating the input as unclean.
pub(crate) fn interior_null_byte_error(
    operation: &'static str,
    resource: &'static str,
    message: &'static str,
) -> Error {
    Error::new_with(ErrorCode::NullUnexpected, operation, message)
        .with_detail("resource", resource)
        .with_detail("reason", "interior_null_byte")
}

/// Build a `ColumnLengthMismatch` error for a table-like row-count contract.
pub(crate) fn row_count_mismatch_error(
    operation: &'static str,
    column_index: usize,
    actual_rows: usize,
    expected_rows: usize,
) -> Error {
    Error::new_with(
        ErrorCode::ColumnLengthMismatch,
        operation,
        format!("column {column_index} has {actual_rows} rows but column 0 has {expected_rows} rows"),
    )
    .with_fact(FACT_COLUMN_INDEX, column_index)
    .with_fact(FACT_ACTUAL_ROWS, actual_rows)
    .with_fact(FACT_EXPECTED_ROWS, expected_rows)
}

/// Build an `UnsupportedOperation` error for a value that exceeded a
/// hard numeric bound.
#[allow(clippy::too_many_arguments)]
pub(crate) fn limit_validation_error(
    operation: &'static str,
    message: impl Into<String>,
    resource: &'static str,
    value_key: &'static str,
    limit: &'static str,
    value: usize,
    remedy: impl Into<String>,
) -> Error {
    let reason = match limit {
        "i32::MAX" => "exceeds_i32_max",
        "i64::MAX" => "exceeds_i64_max",
        _ => "exceeds_supported_range",
    };

    let err = Error::new_with(ErrorCode::UnsupportedOperation, operation, message);
    let err = match FactKeyName::from_snake_case(value_key) {
        Some(FactKeyName::ColumnIndex) => err.with_fact(FACT_COLUMN_INDEX, value),
        Some(FactKeyName::ExpectedRows) => err.with_fact(FACT_EXPECTED_ROWS, value),
        Some(FactKeyName::ActualRows) => err.with_fact(FACT_ACTUAL_ROWS, value),
        Some(FactKeyName::ExpectedCount) => err.with_fact(FACT_EXPECTED_COUNT, value),
        Some(FactKeyName::ActualCount) => err.with_fact(FACT_ACTUAL_COUNT, value),
        Some(FactKeyName::TableRowCount) => err.with_fact(FACT_TABLE_ROW_COUNT, value),
        Some(FactKeyName::TableColumnCount) => err.with_fact(FACT_TABLE_COLUMN_COUNT, value),
        Some(FactKeyName::KeysCount) => err.with_fact(FACT_KEYS_COUNT, value),
        Some(FactKeyName::DistinctKeysCount) => err.with_fact(FACT_DISTINCT_KEYS_COUNT, value),
        _ => err.with_detail(value_key, value.to_string()),
    };

    err.with_detail("resource", resource).with_detail("reason", reason).with_detail("remedy", remedy.into())
}

/// Build a `ForeignInvariantViolation` with the canonical
/// `resource/reason/remedy` contract details.
pub(crate) fn foreign_invariant_violation(
    operation: &'static str,
    message: impl Into<String>,
    resource: &'static str,
    reason: &'static str,
    remedy: impl Into<String>,
) -> Error {
    Error::new_with(ErrorCode::ForeignInvariantViolation, operation, message)
        .with_detail("resource", resource)
        .with_detail("reason", reason)
        .with_detail("remedy", remedy.into())
}

/// Fill in missing foreign-contract details on invariant-shaped errors
/// without overwriting existing payload.
#[must_use]
pub(crate) fn annotate_foreign_invariant_contract_if_absent(
    mut err: Error,
    resource: &'static str,
    reason: &'static str,
    remedy: impl Into<String>,
) -> Error {
    if matches!(err.code(), ErrorCode::ForeignInvariantViolation | ErrorCode::InvariantViolation) {
        let remedy = remedy.into();
        err = err.with_detail_if_absent("resource", resource);
        err = err.with_detail_if_absent("reason", reason);
        err = err.with_detail_if_absent("remedy", remedy);
    }
    err
}

/// Build an `UnsupportedOperation` error for a range-bound overflow.
pub(crate) fn range_limit_error(
    operation: &'static str,
    field: &'static str,
    limit: &'static str,
    value: usize,
) -> Error {
    limit_validation_error(
        operation,
        format!("{field} exceeds {limit}"),
        field,
        field,
        limit,
        value,
        format!("reduce {field} to fit within {limit}"),
    )
}

/// Build an `UnsupportedOperation` error for a row-range bound overflow.
pub(crate) fn row_range_limit_error(operation: &'static str, bound: &'static str, value: usize) -> Error {
    let (message, key, reason) = match bound {
        "start" => ("row range start exceeds i32::MAX", "start_row", "start_exceeds_i32_max"),
        "end" => ("row range end exceeds i32::MAX", "end_row", "end_exceeds_i32_max"),
        _ => ("row range bound exceeds i32::MAX", "row_bound", "bound_exceeds_i32_max"),
    };

    Error::new_with(ErrorCode::UnsupportedOperation, operation, message)
        .with_detail(key, value.to_string())
        .with_detail("resource", "row_range")
        .with_detail("reason", reason)
        .with_detail("remedy", "reduce row range bounds to fit within i32::MAX")
}

/// Build an `UnsupportedOperation` error for a row-range end that overflows
/// `usize`.
pub(crate) fn row_range_overflow_error(operation: &'static str, start: usize, len: usize) -> Error {
    let end_row = (start as u128) + (len as u128);

    Error::new_with(ErrorCode::UnsupportedOperation, operation, "row range end overflowed usize")
        .with_detail("start_row", start.to_string())
        .with_detail("end_row", end_row.to_string())
        .with_detail("resource", "row_range")
        .with_detail("reason", "end_overflowed_usize")
        .with_detail("remedy", "reduce row range bounds to fit within usize")
}

/// Ensure a raw FFI pointer is non-null. On null, returns an
/// `InvariantViolation` error stating the FFI contract was violated.
pub(crate) fn require_non_null<T>(ptr: *mut T, operation: &'static str, target: &'static str) -> Result<NonNull<T>> {
    NonNull::new(ptr).ok_or_else(|| {
        Error::new_with(
            ErrorCode::InvariantViolation,
            operation,
            format!("FFI returned null {target} handle on success"),
        )
        .with_detail("target", target)
    })
}

/// Convert a `usize` to an `i32`, producing a range-limit error on overflow.
pub(crate) fn usize_to_i32(value: usize, operation: &'static str, field: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| range_limit_error(operation, field, "i32::MAX", value))
}

/// Convert a `usize` to an `i64`, producing a range-limit error on overflow.
pub(crate) fn usize_to_i64(value: usize, operation: &'static str, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| range_limit_error(operation, field, "i64::MAX", value))
}

// --------------------------------------------------------------------------
// FFI status translation
// --------------------------------------------------------------------------

fn leaf_code_for_status(status: cudf_sys::cudf_status_t, message: &str) -> ErrorCode {
    if status == cudf_sys::CUDF_ERROR_INVALID_OPERATION
        && message.contains("device memory manager already exists with different physical-base options")
    {
        return ErrorCode::RuntimeBindingMismatch;
    }
    match status {
        cudf_sys::CUDF_ERROR_INVALID_ARGUMENT | cudf_sys::CUDF_ERROR_INVALID_OPERATION | cudf_sys::CUDF_ERROR_TYPE => {
            ErrorCode::ForeignInvalidArgument
        },
        cudf_sys::CUDF_ERROR_CUDA => ErrorCode::CudaRuntime,
        cudf_sys::CUDF_ERROR_FATAL_CUDA => ErrorCode::FatalCuda,
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => ErrorCode::OutOfMemory,
        cudf_sys::CUDF_ERROR_UNSUPPORTED => ErrorCode::UnsupportedOperation,
        cudf_sys::CUDF_ERROR_IO if is_transient_cuda_io_message(message) => ErrorCode::TransientIoFailure,
        cudf_sys::CUDF_ERROR_IO => ErrorCode::IoFailure,
        cudf_sys::CUDF_ERROR_NULL => ErrorCode::NullUnexpected,
        cudf_sys::CUDF_ERROR_INDEX => ErrorCode::IndexOutOfBounds,
        _ => ErrorCode::ForeignInvariantViolation,
    }
}

fn is_transient_cuda_io_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    let invalid_value = normalized.contains("cudaerrorinvalidvalue") && normalized.contains("invalid argument");
    let copy_context = normalized.contains("cuda_memcpy.cu")
        || normalized.contains("cudamemsetasync")
        || normalized.contains("device_buffer.cpp")
        || normalized.contains("h2d ctor")
        || normalized.contains("d->h")
        || normalized.contains("thrust");

    invalid_value && copy_context
}

fn error_for_ffi_status(status: cudf_sys::cudf_status_t, operation: &'static str, message: String) -> Error {
    let code = leaf_code_for_status(status, &message);
    let error = Error::new_with(code, operation, message);
    if code == ErrorCode::TransientIoFailure {
        error.with_detail("transient_promotion", "cuda_invalid_value_copy")
    } else {
        error
    }
}

fn read_ffi_last_error_message() -> String {
    // SAFETY: cudf_last_error_message returns either null or a pointer to a
    // thread-local static C string that remains valid until the next FFI
    // call on this thread. We immediately copy it into an owned String below.
    unsafe {
        let ptr = cudf_sys::cudf_last_error_message();
        if ptr.is_null() {
            String::from("(no error message)")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Convert a C shim status into a `Result<()>` anchored at the caller's
/// operation label.
pub(crate) fn check_status_in(status: cudf_sys::cudf_status_t, operation: &'static str) -> Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }
    let message = read_ffi_last_error_message();
    Err(error_for_ffi_status(status, operation, message))
}

/// Convert a C shim status, attaching diagnostic details on failure.
///
/// Attach non-canonical diagnostic details supplied by FFI boundaries.
///
/// Canonical facts should use typed fact helpers; this helper is reserved for
/// annotations that are useful for diagnosis but are not part of the stable
/// machine-readable error identity.
pub(crate) fn check_status_in_with<I>(
    status: cudf_sys::cudf_status_t,
    operation: &'static str,
    details: I,
) -> Result<()>
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    check_status_in(status, operation).map_err(|err| err.with_details(details))
}

/// Alias for `check_status_in_with` under the new canonical name. Kept for
/// call sites updated to the new vocabulary.
pub(crate) fn check_status_in_with_details<I>(
    status: cudf_sys::cudf_status_t,
    operation: &'static str,
    details: I,
) -> Result<()>
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    check_status_in_with(status, operation, details)
}

// --------------------------------------------------------------------------
// Foreign boundary translation
// --------------------------------------------------------------------------

fn leaf_code_for_interop_kind(kind: InteropErrorKind) -> ErrorCode {
    match kind {
        InteropErrorKind::NotFound => ErrorCode::ForeignInvalidArgument,
        InteropErrorKind::UnsupportedType => ErrorCode::UnsupportedDtype,
        InteropErrorKind::NullMaskUnsupported => ErrorCode::ForeignInvalidArgument,
        InteropErrorKind::InvalidView => ErrorCode::ForeignInvalidArgument,
        InteropErrorKind::ExecutionBindingUnavailable => ErrorCode::ExecutionBindingUnavailable,
        InteropErrorKind::ReadinessUnavailable => ErrorCode::ReadinessUnavailable,
        InteropErrorKind::UnsupportedLayout => ErrorCode::UnsupportedLayout,
        InteropErrorKind::PlacementUnsupported | InteropErrorKind::ProfileUnsupported => {
            ErrorCode::UnsupportedOperation
        },
        InteropErrorKind::MixedDeviceUnsupported => ErrorCode::CrossDeviceInputs,
        InteropErrorKind::ExecutionBindingMismatch | InteropErrorKind::ReadinessMismatch => {
            ErrorCode::RuntimeBindingMismatch
        },
        InteropErrorKind::LeaseAlreadyActive => ErrorCode::LeaseAlreadyActive,
    }
}

/// Translate an [`InteropError`] into a cudf-rs [`Error`] at the
/// caller-supplied cudf-rs boundary `operation`.
///
/// The translated error always uses the caller-supplied cudf-rs operation
/// label as its anchor. Foreign metadata is preserved as follows:
///
/// - `operation` is the caller-supplied cudf-rs constant, never the foreign
///   layer's operation label.
/// - The foreign layer's operation label goes into `details["foreign_operation"]`.
/// - Foreign `called` breadcrumbs go into `details["foreign_called"]` (may
///   repeat).
/// - The inner `InteropError` is preserved as the standard source chain, so
///   any cudf-rs `Error` nested inside remains reachable via
///   [`find_cudf_error_in_chain`].
pub(crate) fn from_interop_in(error: InteropError, operation: &'static str) -> Error {
    let code = leaf_code_for_interop_kind(error.kind());
    let message = error.message().to_string();

    let mut out = Error::new_with(code, operation, message);

    for (key, value) in error.context() {
        match *key {
            "operation" => {
                out = out.with_detail("foreign_operation", value.clone());
            },
            "called" => {
                out = out.with_detail("foreign_called", value.clone());
            },
            other => {
                out = attach_foreign_context_entry(out, other, value);
            },
        }
    }

    out.with_source(error)
}

/// Attach a foreign-context entry to `err`.
///
/// Canonical snake_case fact keys like `left_device`, `right_device`, and
/// `type_id` are not allowed in `details`. Foreign context comes in as weakly
/// typed `(key, String)` pairs, so this helper:
///
/// - Promotes recognised device-role breadcrumbs to [`DeviceIdFact`] facts
///   when the value parses as an `i32`, falling back to a non-canonical
///   `foreign_<role>_device_raw` detail otherwise.
/// - Rewrites any other key that collides with a canonical snake_case fact
///   name onto a `foreign_<name>` detail, preserving the diagnostic without
///   violating the overlap rule.
/// - Keeps every remaining breadcrumb on the detail path unchanged.
fn attach_foreign_context_entry(err: Error, key: &'static str, value: &str) -> Error {
    match foreign_device_role(key) {
        Some(ForeignDeviceRole::Left) => {
            attach_foreign_device_fact(err, FACT_LEFT_DEVICE, "foreign_left_device_raw", value)
        },
        Some(ForeignDeviceRole::Right) => {
            attach_foreign_device_fact(err, FACT_RIGHT_DEVICE, "foreign_right_device_raw", value)
        },
        None => match foreign_context_detail_key(key) {
            Some(rewritten) => err.with_detail(rewritten, value.to_string()),
            None => err.with_detail(key, value.to_string()),
        },
    }
}

fn attach_foreign_device_fact(
    err: Error,
    fact_key: FactKey<DeviceIdFact>,
    fallback_detail_key: &'static str,
    value: &str,
) -> Error {
    match value.parse::<i32>() {
        Ok(device_id) if err.fact_value(fact_key.name()).is_none() => err.with_fact(fact_key, DeviceIdFact(device_id)),
        _ => err.with_detail(fallback_detail_key, value.to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForeignDeviceRole {
    Left,
    Right,
}

/// Collapse the nine raw foreign device-role keys into canonical left/right
/// roles.
fn foreign_device_role(key: &str) -> Option<ForeignDeviceRole> {
    match key {
        "producer_device" | "expected_device" | "descriptor_device" | "release_device" => Some(ForeignDeviceRole::Left),
        "runtime_device" | "actual_device" | "event_device" | "stream_device" | "consumer_device" => {
            Some(ForeignDeviceRole::Right)
        },
        _ => None,
    }
}

/// Rewrite any foreign context key whose string form collides with a
/// canonical fact snake_case name. Returning `None` means the key is safe to
/// keep as-is on the detail path.
fn foreign_context_detail_key(key: &'static str) -> Option<&'static str> {
    FactKeyName::from_snake_case(key)?;
    Some(match key {
        "column_name" => "foreign_column_name",
        "column_index" => "foreign_column_index",
        "expected_rows" => "foreign_expected_rows",
        "actual_rows" => "foreign_actual_rows",
        "expected_count" => "foreign_expected_count",
        "actual_count" => "foreign_actual_count",
        "left_type_id" => "foreign_left_type_id",
        "right_type_id" => "foreign_right_type_id",
        "left_device" => "foreign_left_device_raw",
        "right_device" => "foreign_right_device_raw",
        "table_row_count" => "foreign_table_row_count",
        "table_column_count" => "foreign_table_column_count",
        "keys_count" => "foreign_keys_count",
        "distinct_keys_count" => "foreign_distinct_keys_count",
        "type_id" => "foreign_type_id",
        // Unreachable unless a new canonical key is added. Use a safe
        // fallback rather than panicking inside a translator.
        _ => "foreign_canonical_context",
    })
}

/// Walk the `source()` chain looking for a nested cudf-rs [`Error`].
pub fn find_cudf_error_in_chain<'a>(error: &'a (dyn StdError + 'static)) -> Option<&'a Error> {
    let mut current = Some(error);
    while let Some(err) = current {
        if let Some(cudf_error) = err.downcast_ref::<Error>() {
            return Some(cudf_error);
        }
        current = err.source();
    }
    None
}
