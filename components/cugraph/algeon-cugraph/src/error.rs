//! Errors carry a [`ErrorKind`] variant, an [`ErrorStatus`] recording the
//! observed source condition, an operation name describing what was being
//! attempted, an optional human-readable message, optional key-value context
//! pairs, and an optional source error. This makes it easy to diagnose
//! failures without losing the underlying cause. The status is the
//! observation and does not grant permission to retry. Callers decide recovery
//! only after cleanup and device-health checks.

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error as StdError,
    fmt,
};

#[cfg(any(test, feature = "test-hooks"))]
use cudf::testing::device_safety;
use cudf::{
    DeviceId,
    runtime::{DeviceFaultKind, DeviceFaultNativeStatus, report_device_fault},
};
use cugraph_sys::{
    cugraph_allocation_source_t, cugraph_error_alignment, cugraph_error_allocation_source, cugraph_error_code_t,
    cugraph_error_cuda_error, cugraph_error_domain_id, cugraph_error_failure_source, cugraph_error_failure_source_t,
    cugraph_error_free, cugraph_error_generation, cugraph_error_member_id, cugraph_error_message,
    cugraph_error_requested_bytes, cugraph_error_t,
};

const CUGRAPH_RS_CUDA_FAILURE_NONE: i32 = 0;
const CUGRAPH_RS_CUDA_FAILURE_RECOVERABLE: i32 = 1;
const CUGRAPH_RS_CUDA_FAILURE_FATAL: i32 = 2;

unsafe extern "C" {
    fn cugraph_rs_classify_cuda_failure_after_error(cuda_status: *mut i32) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeCudaFailure {
    None,
    Recoverable(i32),
    Fatal(i32),
}

/// Categorisation of cuGraph and library-level errors.
///
/// Each variant describes **what the caller should do**. It does not name the
/// dependency that failed. Variants that originate in the raw C API map to the
/// corresponding `CUGRAPH_*` error codes; library-only variants cover Rust-side
/// interop, lifecycle, and I/O boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An unexpected or unknown error occurred (`CUGRAPH_UNKNOWN_ERROR`).
    Unexpected,
    /// The resource handle is invalid (`CUGRAPH_INVALID_HANDLE`).
    InvalidHandle,
    /// GPU memory allocation failed (`CUGRAPH_ALLOC_ERROR`).
    OutOfMemory,
    /// An input argument was invalid (`CUGRAPH_INVALID_INPUT`).
    InvalidInput,
    /// An external I/O operation failed.
    IoError,
    /// A library-level precondition, lifecycle contract, or state transition failed.
    InvalidState,
    /// The requested operation is not implemented (`CUGRAPH_NOT_IMPLEMENTED`).
    NotImplemented,
    /// The combination of types is not supported (`CUGRAPH_UNSUPPORTED_TYPE_COMBINATION`).
    UnsupportedType,
    /// An iterative algorithm reached its configured iteration bound.
    NonConvergence,
}

impl ErrorKind {
    /// Map a C error code to an [`ErrorKind`].
    pub fn from_error_code(code: cugraph_error_code_t) -> Self {
        match code {
            cugraph_error_code_t::CUGRAPH_INVALID_HANDLE => ErrorKind::InvalidHandle,
            cugraph_error_code_t::CUGRAPH_ALLOC_ERROR => ErrorKind::OutOfMemory,
            cugraph_error_code_t::CUGRAPH_INVALID_INPUT => ErrorKind::InvalidInput,
            cugraph_error_code_t::CUGRAPH_NOT_IMPLEMENTED => ErrorKind::NotImplemented,
            cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION => ErrorKind::UnsupportedType,
            _ => ErrorKind::Unexpected,
        }
    }

    /// Map from a raw C error-code integer value.
    pub fn from_error_code_raw(raw: u32) -> Self {
        Self::from_error_code(cugraph_error_code_t(raw))
    }

    /// Whether a backtrace should be captured for this error kind.
    ///
    /// Only truly unexpected errors benefit from backtraces; expected
    /// conditions like `InvalidInput` or `NotImplemented` have sufficient
    /// context from the error kind and message alone.
    fn capture_backtrace(&self) -> bool {
        matches!(self, ErrorKind::Unexpected)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Unexpected => write!(f, "Unexpected"),
            ErrorKind::InvalidHandle => write!(f, "InvalidHandle"),
            ErrorKind::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::IoError => write!(f, "IoError"),
            ErrorKind::InvalidState => write!(f, "InvalidState"),
            ErrorKind::NotImplemented => write!(f, "NotImplemented"),
            ErrorKind::UnsupportedType => write!(f, "UnsupportedType"),
            ErrorKind::NonConvergence => write!(f, "NonConvergence"),
        }
    }
}

/// Which allocation-failure exception type produced a cuGraph allocation error.
///
/// These variants record the C++ exception type that cuGraph caught, nothing
/// more. Note that `rmm::out_of_memory` derives from `rmm::bad_alloc`, which in
/// turn derives from `std::bad_alloc`, so the most-derived matching variant is
/// reported. `Unknown` is fail-closed: it covers non-allocation errors, missing
/// error objects, and unrecognized native values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AllocationSource {
    /// The C error object did not explicitly report a known allocation path.
    #[default]
    Unknown,
    /// `rmm::out_of_memory`.
    RmmOutOfMemory,
    /// `rmm::bad_alloc`.
    RmmBadAlloc,
    /// `std::bad_alloc`.
    StdBadAlloc,
}

impl AllocationSource {
    fn from_native(value: cugraph_allocation_source_t) -> Self {
        match value {
            cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY => Self::RmmOutOfMemory,
            cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_RMM_BAD_ALLOC => Self::RmmBadAlloc,
            cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC => Self::StdBadAlloc,
            _ => Self::Unknown,
        }
    }

    unsafe fn from_error(error: *const cugraph_error_t) -> Self {
        // SAFETY: the caller guarantees that `error` is either null or a live
        // cuGraph error object for the duration of this call.
        let value = unsafe { cugraph_error_allocation_source(error) };
        Self::from_native(value)
    }

    /// Returns the stable fact value for this allocation source.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::RmmOutOfMemory => "rmm_out_of_memory",
            Self::RmmBadAlloc => "rmm_bad_alloc",
            Self::StdBadAlloc => "std_bad_alloc",
        }
    }
}

impl fmt::Display for AllocationSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which owner or device condition caused an allocation failure.
///
/// This is separate from [`AllocationSource`], which records only the C++
/// exception type caught at the native boundary. `Unknown` is fail-closed and
/// must not be inferred from an error message or from `ErrorStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AllocationFailureSource {
    /// The native error did not carry an attributable failure source.
    #[default]
    Unknown,
    /// The query or owner reservation limit rejected the allocation.
    ReservationLimit,
    /// The device allocator reported physical memory pressure.
    Physical,
}

impl AllocationFailureSource {
    fn from_native(value: cugraph_error_failure_source_t) -> Self {
        match value {
            cugraph_error_failure_source_t::CUGRAPH_ERROR_FAILURE_SOURCE_PHYSICAL => Self::Physical,
            cugraph_error_failure_source_t::CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT => Self::ReservationLimit,
            _ => Self::Unknown,
        }
    }

    unsafe fn from_error(error: *const cugraph_error_t) -> Self {
        // SAFETY: the caller guarantees that `error` is either null or a live
        // cuGraph error object for the duration of this call.
        let value = unsafe { cugraph_error_failure_source(error) };
        Self::from_native(value)
    }

    /// Returns the stable fact value for this failure source.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::ReservationLimit => "reservation_limit",
            Self::Physical => "physical",
        }
    }
}

impl fmt::Display for AllocationFailureSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Source condition recorded on an error.
///
/// The status describes the condition observed at the source; it is not
/// permission to retry. A caller must still complete its own cleanup and
/// device-health checks before deciding anything:
///
/// - [`Permanent`](ErrorStatus::Permanent): will never succeed without
///   external changes (e.g. invalid input, unsupported operation). Do not
///   retry the same request.
/// - [`Temporary`](ErrorStatus::Temporary): transient condition that may
///   clear (e.g. GPU memory pressure, recoverable CUDA failure). Retry only
///   after cleanup and a health check.
/// - [`Persistent`](ErrorStatus::Persistent): the source reports a
///   continuing condition (e.g. a fatal CUDA probe result, recorded without
///   any retry having happened). Stop automatic retry, finish cleanup, and
///   escalate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// The error is permanent; retrying will not help without external
    /// intervention.
    Permanent,
    /// The source condition is transient and may clear. Retry remains the
    /// caller's decision after cleanup and device-health checks.
    Temporary,
    /// The source reports a continuing condition. It may be set directly by
    /// a fatal probe result without any retry having happened; stop
    /// automatic retry, finish cleanup, and escalate.
    Persistent,
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorStatus::Permanent => write!(f, "permanent"),
            ErrorStatus::Temporary => write!(f, "temporary"),
            ErrorStatus::Persistent => write!(f, "persistent"),
        }
    }
}

/// A cuGraph error with structured context.
///
/// Carries the kind of error, retry status, the operation that was being
/// performed, an optional human-readable message, optional key-value context
/// pairs for debugging, and an optional source error.
///
/// # Display formats
///
/// - **`Display`** (single-line): for structured logs and machine parsing.
///   Format: `{kind} ({status}) at {operation}[ => {message}][, context: { ... }][, source: ...]`
/// - **`Debug`** (multi-line): for human debugging with context breakdown.
/// - **`Debug` alternate** (`{:#?}`): standard Rust struct-style inspection.
#[derive(Debug, Clone, Copy)]
struct AllocationFacts {
    source: AllocationSource,
    failure_source: AllocationFailureSource,
    domain_id: u64,
    member_id: u64,
    generation: u64,
    requested_bytes: u64,
    alignment: u64,
    cuda_error: i32,
}

impl Default for AllocationFacts {
    fn default() -> Self {
        Self {
            source: AllocationSource::Unknown,
            failure_source: AllocationFailureSource::Unknown,
            domain_id: 0,
            member_id: 0,
            generation: 0,
            requested_bytes: 0,
            alignment: 0,
            cuda_error: -1,
        }
    }
}

pub struct Error {
    kind: ErrorKind,
    status: ErrorStatus,
    allocation: Option<Box<AllocationFacts>>,
    operation: &'static str,
    message: String,
    context: Vec<(&'static str, String)>,
    source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    backtrace: Option<Box<Backtrace>>,
}

impl Error {
    /// Create a new error for the given kind and operation.
    ///
    /// The error defaults to [`ErrorStatus::Permanent`]. A backtrace is
    /// captured only for [`ErrorKind::Unexpected`] errors (when the
    /// `RUST_BACKTRACE` environment variable is set).
    pub fn new(kind: ErrorKind, operation: &'static str) -> Self {
        Self {
            kind,
            status: ErrorStatus::Permanent,
            allocation: None,
            operation,
            message: String::new(),
            context: Vec::new(),
            source: None,
            backtrace: kind
                .capture_backtrace()
                .then(Backtrace::capture)
                .filter(|bt| bt.status() == BacktraceStatus::Captured)
                .map(Box::new),
        }
    }

    fn allocation_facts(&self) -> AllocationFacts {
        self.allocation.as_deref().copied().unwrap_or_default()
    }

    /// Set the human-readable error message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Set the operation, stacking the previous one as context.
    ///
    /// If an operation is already set, it is preserved as
    /// `("called", old_operation)` context, creating a logical call chain
    /// for debugging.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        if !self.operation.is_empty() {
            self.context.push(("called", self.operation.to_string()));
        }
        self.operation = operation;
        self
    }

    /// Attach a key-value context pair for debugging.
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Attach an underlying source error.
    ///
    /// Accepts any source error that is safe to store and report from this
    /// library's public error type.
    pub fn set_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        debug_assert!(self.source.is_none(), "the source error has been set");
        self.source = Some(Box::new(source));
        self
    }

    /// Record a transient source condition.
    ///
    /// This marks the observation only. Retry remains the caller's decision
    /// after its cleanup and device-health checks.
    pub fn set_temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    /// Record a permanent source condition: retrying the same request will
    /// not help without external changes.
    pub fn set_permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    /// Record a continuing source condition.
    ///
    /// A fatal probe result can set this status directly, without any retry
    /// having happened. Callers must stop automatic retry, finish cleanup,
    /// and escalate.
    pub fn set_persistent(mut self) -> Self {
        self.status = ErrorStatus::Persistent;
        self
    }

    /// The kind of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The retry status of this error.
    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    /// Which allocation-failure exception type produced this error.
    ///
    /// This is `Unknown` unless the C error object explicitly classified an
    /// allocation failure. It reports only the caught exception type and is
    /// never inferred from message text or error status.
    pub fn allocation_source(&self) -> AllocationSource {
        self.allocation_facts().source
    }

    /// Which owner or device condition caused this allocation failure.
    pub fn allocation_failure_source(&self) -> AllocationFailureSource {
        self.allocation_facts().failure_source
    }

    /// Attributes this error to an allocation failure source.
    ///
    /// Native code attributes allocation failures from the C error object.
    /// This constructor lets consumers exercise the attribution contract
    /// without a device.
    #[cfg(any(test, feature = "test-hooks"))]
    #[must_use]
    pub fn with_allocation_failure_source(mut self, failure_source: AllocationFailureSource) -> Self {
        let mut facts = self.allocation_facts();
        facts.failure_source = failure_source;
        self.allocation = Some(Box::new(facts));
        self
    }

    /// Stable domain identity supplied by the native allocation owner.
    pub fn allocation_domain_id(&self) -> u64 {
        self.allocation_facts().domain_id
    }

    /// Stable member identity supplied by the native allocation owner.
    pub fn allocation_member_id(&self) -> u64 {
        self.allocation_facts().member_id
    }

    /// Native generation for the allocation owner identity.
    pub fn allocation_generation(&self) -> u64 {
        self.allocation_facts().generation
    }

    /// Bytes requested by the failed allocation, when known.
    pub fn allocation_requested_bytes(&self) -> u64 {
        self.allocation_facts().requested_bytes
    }

    /// Alignment requested by the failed allocation, when known.
    pub fn allocation_alignment(&self) -> u64 {
        self.allocation_facts().alignment
    }

    /// CUDA status observed at the allocation boundary, or `-1` when absent.
    pub fn allocation_cuda_error(&self) -> i32 {
        self.allocation_facts().cuda_error
    }

    /// Whether the source reported a transient condition.
    ///
    /// This is the observation and does not permit a retry. The caller must
    /// still complete cleanup and device-health checks first.
    pub fn is_temporary(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }

    /// Whether the source reported a permanent condition.
    pub fn is_permanent(&self) -> bool {
        self.status == ErrorStatus::Permanent
    }

    /// Whether the source reported a continuing condition.
    ///
    /// This status can be set directly by a fatal probe result without any
    /// retry having happened.
    pub fn is_persistent(&self) -> bool {
        self.status == ErrorStatus::Persistent
    }

    /// The operation that was being performed when the error occurred.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// The human-readable error message, if any.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns one structured context value by key.
    pub fn context_value(&self, key: &str) -> Option<&str> {
        self.context.iter().rev().find(|(candidate, _)| *candidate == key).map(|(_, value)| value.as_str())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            // {:#?}; standard Rust struct-style inspection.
            let allocation = self.allocation_facts();
            let mut d = f.debug_struct("Error");
            d.field("kind", &self.kind);
            d.field("status", &self.status);
            d.field("allocation_source", &allocation.source);
            d.field("allocation_failure_source", &allocation.failure_source);
            d.field("allocation_domain_id", &allocation.domain_id);
            d.field("allocation_member_id", &allocation.member_id);
            d.field("allocation_generation", &allocation.generation);
            d.field("allocation_requested_bytes", &allocation.requested_bytes);
            d.field("allocation_alignment", &allocation.alignment);
            d.field("allocation_cuda_error", &allocation.cuda_error);
            d.field("operation", &self.operation);
            if !self.message.is_empty() {
                d.field("message", &self.message);
            }
            if !self.context.is_empty() {
                d.field("context", &self.context);
            }
            if let Some(ref src) = self.source {
                d.field("source", src);
            }
            d.finish()
        } else {
            // {:?}; multi-line format for human debugging.
            write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;
            if !self.message.is_empty() {
                write!(f, " => {}", self.message)?;
            }
            if !self.context.is_empty() {
                write!(f, "\n\nContext:")?;
                for (key, value) in &self.context {
                    write!(f, "\n   {}: {}", key, value)?;
                }
            }
            if let Some(ref src) = self.source {
                write!(f, "\n\nSource: {:#}", src)?;
            }
            if let Some(ref bt) = self.backtrace {
                write!(f, "\n\nBacktrace:\n{}", bt)?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;
        if !self.message.is_empty() {
            write!(f, " => {}", self.message)?;
        }
        if !self.context.is_empty() {
            write!(f, ", context: {{")?;
            for (i, (key, value)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, " {}: {}", key, value)?;
            }
            write!(f, " }}")?;
        }
        if let Some(ref src) = self.source {
            write!(f, ", source: {}", src)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn StdError + 'static))
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn unexpected_null_pointer(operation: &'static str, pointer_name: &'static str) -> Error {
    Error::new(ErrorKind::Unexpected, operation)
        .with_message(format!("C API returned null pointer for {pointer_name}"))
        .with_context("pointer", pointer_name)
}

pub(crate) fn require_non_null<T>(ptr: *mut T, operation: &'static str, pointer_name: &'static str) -> Result<*mut T> {
    if ptr.is_null() { Err(unexpected_null_pointer(operation, pointer_name)) } else { Ok(ptr) }
}

/// Check a GPU-capable C API return code and convert to a Rust `Result`.
///
/// On success (`CUGRAPH_SUCCESS`), returns `Ok(())`. Otherwise extracts the
/// error message from the C error object, frees it, and returns an appropriate
/// [`Error`].
///
/// `OutOfMemory` errors are automatically marked as
/// [`ErrorStatus::Temporary`] since GPU memory pressure is often transient.
///
/// # Safety
///
/// `error_ptr` must be null or a valid `cugraph_error_t` populated by the C API
/// call. The caller must still hold owner-device entry for `device` so the
/// native typed CUDA failure probe inspects the same immutable device binding.
pub(crate) unsafe fn check_error(
    code: cugraph_error_code_t,
    error_ptr: *mut cugraph_error_t,
    device: i32,
    operation: &'static str,
) -> Result<()> {
    if code == cugraph_error_code_t::CUGRAPH_SUCCESS {
        return Ok(());
    }

    // Copy every field while the C error object is still live. The CUDA
    // classification probe below is another FFI call and must not be allowed
    // to race a native payload that is only valid until the next call.
    // SAFETY: the native call leaves either a null pointer or a live error
    // object until this function frees it after copying the payload.
    let payload = unsafe { copy_native_error_payload(error_ptr) };
    // SAFETY: owner-device entry remains live and the native probe accepts no
    // borrowed resource other than its valid stack out-pointer.
    let cuda_failure = unsafe { native_cuda_failure_after_error() };
    // SAFETY: this forwards the caller's error-pointer contract and consumes
    // the pointer exactly once on the failure path.
    unsafe { check_error_with_payload(code, error_ptr, payload, Some((device, cuda_failure)), operation) }
}

/// Check a host-only cuGraph C API call that cannot execute CUDA work.
///
/// # Safety
///
/// `error_ptr` must be null or a valid `cugraph_error_t` populated by the C API
/// call and consumed exactly once by this function on failure.
pub(crate) unsafe fn check_host_error(
    code: cugraph_error_code_t,
    error_ptr: *mut cugraph_error_t,
    operation: &'static str,
) -> Result<()> {
    if code == cugraph_error_code_t::CUGRAPH_SUCCESS {
        return Ok(());
    }
    // SAFETY: the native call leaves either a null pointer or a live error
    // object until this function frees it after copying the payload.
    let payload = unsafe { copy_native_error_payload(error_ptr) };
    // SAFETY: this forwards the caller's error-pointer contract and consumes
    // the pointer exactly once on the failure path.
    unsafe { check_error_with_payload(code, error_ptr, payload, None, operation) }
}

unsafe fn native_cuda_failure_after_error() -> NativeCudaFailure {
    let mut cuda_status = 0;
    // SAFETY: `cuda_status` is a valid writable out-pointer and the shim does
    // not retain it after returning.
    let classification = unsafe { cugraph_rs_classify_cuda_failure_after_error(&mut cuda_status) };
    match classification {
        CUGRAPH_RS_CUDA_FAILURE_NONE => NativeCudaFailure::None,
        CUGRAPH_RS_CUDA_FAILURE_FATAL => NativeCudaFailure::Fatal(cuda_status),
        CUGRAPH_RS_CUDA_FAILURE_RECOVERABLE => NativeCudaFailure::Recoverable(cuda_status),
        _ => NativeCudaFailure::Recoverable(cuda_status),
    }
}

struct NativeErrorPayload {
    allocation_source: AllocationSource,
    allocation_failure_source: AllocationFailureSource,
    domain_id: u64,
    member_id: u64,
    generation: u64,
    requested_bytes: u64,
    alignment: u64,
    cuda_error: i32,
    message: String,
}

impl NativeErrorPayload {
    fn unknown() -> Self {
        Self {
            allocation_source: AllocationSource::Unknown,
            allocation_failure_source: AllocationFailureSource::Unknown,
            domain_id: 0,
            member_id: 0,
            generation: 0,
            requested_bytes: 0,
            alignment: 0,
            cuda_error: -1,
            message: String::new(),
        }
    }
}

unsafe fn copy_native_error_payload(error: *const cugraph_error_t) -> NativeErrorPayload {
    if error.is_null() {
        return NativeErrorPayload::unknown();
    }

    // SAFETY: the caller guarantees that `error` is live until this complete
    // owned copy finishes.
    let message = unsafe { cugraph_error_message(error) };
    let message = if message.is_null() {
        String::new()
    } else {
        // SAFETY: cuGraph returns a NUL-terminated string owned by `error`.
        unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy().into_owned()
    };

    // SAFETY: `error` is non-null and live for these accessors. Each call
    // reads the native object without retaining the pointer.
    let (
        allocation_source,
        allocation_failure_source,
        domain_id,
        member_id,
        generation,
        requested_bytes,
        alignment,
        cuda_error,
    ) = unsafe {
        (
            AllocationSource::from_error(error),
            AllocationFailureSource::from_error(error),
            cugraph_error_domain_id(error),
            cugraph_error_member_id(error),
            cugraph_error_generation(error),
            cugraph_error_requested_bytes(error),
            cugraph_error_alignment(error),
            cugraph_error_cuda_error(error),
        )
    };

    NativeErrorPayload {
        allocation_source,
        allocation_failure_source,
        domain_id,
        member_id,
        generation,
        requested_bytes,
        alignment,
        cuda_error,
        message,
    }
}

unsafe fn check_error_with_payload(
    code: cugraph_error_code_t,
    error_ptr: *mut cugraph_error_t,
    payload: NativeErrorPayload,
    device_failure: Option<(i32, NativeCudaFailure)>,
    operation: &'static str,
) -> Result<()> {
    let kind = ErrorKind::from_error_code(code);
    let mut err = Error::new(kind, operation);
    err.allocation = Some(Box::new(AllocationFacts {
        source: payload.allocation_source,
        failure_source: payload.allocation_failure_source,
        domain_id: payload.domain_id,
        member_id: payload.member_id,
        generation: payload.generation,
        requested_bytes: payload.requested_bytes,
        alignment: payload.alignment,
        cuda_error: payload.cuda_error,
    }));

    // OutOfMemory is typically transient; GPU memory may free up.
    if kind == ErrorKind::OutOfMemory {
        err = err.set_temporary();
    }

    if let Some((device, cuda_failure)) = device_failure {
        match cuda_failure {
            NativeCudaFailure::None => {},
            NativeCudaFailure::Recoverable(cuda_status) => {
                err = err
                    .with_context("cuda_failure_classification", "recoverable")
                    .with_context("cuda_status_code", cuda_status.to_string());
            },
            NativeCudaFailure::Fatal(cuda_status) => {
                let physical = usize::try_from(device).ok().and_then(|ordinal| DeviceId::new(ordinal).ok());
                #[cfg(any(test, feature = "test-hooks"))]
                let physical = physical.or_else(|| device_safety::injected_device_id(device));
                if let Some(device_id) = physical {
                    report_device_fault(
                        device_id,
                        DeviceFaultKind::FatalDeviceOperation,
                        operation,
                        Some(DeviceFaultNativeStatus::Cuda(cuda_status)),
                    );
                }
                err = err
                    .set_persistent()
                    .with_context("cuda_failure_classification", "fatal")
                    .with_context("cuda_status_code", cuda_status.to_string())
                    .with_context("device", device.to_string());
            },
        }
    }

    if !payload.message.is_empty() {
        err = err.with_message(payload.message);
    }

    if !error_ptr.is_null() {
        // SAFETY: `error_ptr` is a live cuGraph error object and this function
        // consumes it exactly once on the failure path. Upstream owns only a
        // host `std::string`; no CUDA resource is touched by this destructor.
        unsafe { cugraph_error_free(error_ptr) };
    }

    Err(err)
}

#[cfg(test)]
mod tests {
    use cudf::runtime::DeviceHealthStatus;

    use super::*;

    #[test]
    fn allocation_source_maps_native_values_and_fails_closed() {
        assert_eq!(
            AllocationSource::from_native(cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_UNKNOWN),
            AllocationSource::Unknown
        );
        assert_eq!(
            AllocationSource::from_native(cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY),
            AllocationSource::RmmOutOfMemory
        );
        assert_eq!(
            AllocationSource::from_native(cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_RMM_BAD_ALLOC),
            AllocationSource::RmmBadAlloc
        );
        assert_eq!(
            AllocationSource::from_native(cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC),
            AllocationSource::StdBadAlloc
        );
        assert_eq!(AllocationSource::from_native(cugraph_allocation_source_t(999)), AllocationSource::Unknown);
    }

    #[test]
    fn allocation_failure_source_maps_native_values_and_fails_closed() {
        assert_eq!(
            AllocationFailureSource::from_native(cugraph_error_failure_source_t::CUGRAPH_ERROR_FAILURE_SOURCE_PHYSICAL),
            AllocationFailureSource::Physical
        );
        assert_eq!(
            AllocationFailureSource::from_native(
                cugraph_error_failure_source_t::CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT
            ),
            AllocationFailureSource::ReservationLimit
        );
        assert_eq!(
            AllocationFailureSource::from_native(cugraph_error_failure_source_t::CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN),
            AllocationFailureSource::Unknown
        );
        assert_eq!(
            AllocationFailureSource::from_native(cugraph_error_failure_source_t(999)),
            AllocationFailureSource::Unknown
        );
    }

    #[test]
    fn test_error_kind_from_error_code() {
        assert_eq!(ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_INVALID_HANDLE), ErrorKind::InvalidHandle);
        assert_eq!(ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR), ErrorKind::OutOfMemory);
        assert_eq!(ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_INVALID_INPUT), ErrorKind::InvalidInput);
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_NOT_IMPLEMENTED),
            ErrorKind::NotImplemented
        );
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION),
            ErrorKind::UnsupportedType
        );
    }

    #[test]
    fn test_error_display_with_message_and_context() {
        let err = Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
            .with_message("src and dst must have the same length")
            .with_context("src_len", "10")
            .with_context("dst_len", "5");

        let display = err.to_string();
        assert_eq!(
            display,
            "InvalidInput (permanent) at GraphBuilder::build \
             => src and dst must have the same length, \
             context: { src_len: 10, dst_len: 5 }"
        );
    }

    #[test]
    fn test_error_source_chain() {
        let source = std::io::Error::other("GPU fault");
        let err = Error::new(ErrorKind::Unexpected, "ResourceHandle::new").set_source(source);

        let std_source = std::error::Error::source(&err);
        assert!(std_source.is_some());
        assert_eq!(std_source.unwrap().to_string(), "GPU fault");
    }

    #[test]
    fn native_payload_preserves_allocation_facts_and_error_identity() {
        let payload = NativeErrorPayload {
            allocation_source: AllocationSource::RmmOutOfMemory,
            allocation_failure_source: AllocationFailureSource::ReservationLimit,
            domain_id: 11,
            member_id: 17,
            generation: 23,
            requested_bytes: 4096,
            alignment: 256,
            cuda_error: 2,
            message: "reservation limit".to_string(),
        };

        // SAFETY: a null error pointer is the supported missing-native-error
        // case, so the conversion has nothing to free.
        let err = unsafe {
            check_error_with_payload(
                cugraph_error_code_t::CUGRAPH_ALLOC_ERROR,
                std::ptr::null_mut(),
                payload,
                None,
                "allocation_op",
            )
        }
        .expect_err("allocation payload must produce an error");

        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert_eq!(err.status(), ErrorStatus::Temporary);
        assert_eq!(err.allocation_source(), AllocationSource::RmmOutOfMemory);
        assert_eq!(err.allocation_failure_source(), AllocationFailureSource::ReservationLimit);
        assert_eq!(err.allocation_domain_id(), 11);
        assert_eq!(err.allocation_member_id(), 17);
        assert_eq!(err.allocation_generation(), 23);
        assert_eq!(err.allocation_requested_bytes(), 4096);
        assert_eq!(err.allocation_alignment(), 256);
        assert_eq!(err.allocation_cuda_error(), 2);
        assert_eq!(err.message(), "reservation limit");
    }

    #[test]
    fn test_with_operation_stacking() {
        let err = Error::new(ErrorKind::InvalidInput, "inner_op").with_message("bad data").with_operation("outer_op");

        assert_eq!(err.operation(), "outer_op");
        let display = err.to_string();
        assert!(display.contains("at outer_op"));
        assert!(display.contains("called: inner_op"));
    }

    #[test]
    fn test_check_error_success() {
        // SAFETY: Passing a success code with a null error pointer matches the C API contract;
        // `check_error` will not dereference the pointer on success.
        unsafe {
            let result = check_host_error(cugraph_error_code_t::CUGRAPH_SUCCESS, std::ptr::null_mut(), "test");
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_require_non_null_rejects_null_pointers() {
        let err = require_non_null::<u8>(std::ptr::null_mut(), "test_op", "result_ptr").expect_err("null must fail");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_op");
        assert!(err.message().contains("result_ptr"));
    }

    #[test]
    fn test_check_error_failure_null_ptr() {
        // SAFETY: `check_error` accepts a null error pointer for failure paths and degrades to a
        // synthesized Rust error without dereferencing the pointer.
        unsafe {
            let result = check_host_error(cugraph_error_code_t::CUGRAPH_INVALID_INPUT, std::ptr::null_mut(), "test_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidInput);
            assert_eq!(err.operation(), "test_op");
            assert!(err.is_permanent());
        }
    }

    #[test]
    fn test_check_error_failure_unknown_code_maps_to_unexpected() {
        // SAFETY: The test intentionally passes an unknown status code and a null error pointer to
        // verify `check_error`'s fallback mapping logic.
        unsafe {
            let result = check_host_error(cugraph_error_code_t(777), std::ptr::null_mut(), "test_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::Unexpected);
            assert_eq!(err.operation(), "test_op");
        }
    }

    #[test]
    fn test_check_error_out_of_memory_is_temporary() {
        // SAFETY: The allocation failure code is exercised with a null error pointer to verify
        // Rust-side status mapping; `check_error` handles this without dereferencing the pointer.
        unsafe {
            let result = check_host_error(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR, std::ptr::null_mut(), "alloc_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::OutOfMemory);
            assert!(err.is_temporary());
            assert_eq!(err.allocation_source(), AllocationSource::Unknown);
        }
    }

    #[test]
    fn fatal_native_cuda_classification_quarantines_the_explicit_owner_device() {
        let scope = device_safety::DeviceSafetyTestScope::install([]);
        let device = device_safety::injected_device_id(7).expect("injected owner device");

        // SAFETY: the null error pointer is permitted on failure and the typed
        // CUDA classification is injected directly into the Rust translation
        // layer, without crossing the production CUDA probe.
        let error = unsafe {
            check_error_with_payload(
                cugraph_error_code_t::CUGRAPH_UNKNOWN_ERROR,
                std::ptr::null_mut(),
                NativeErrorPayload::unknown(),
                Some((7, NativeCudaFailure::Fatal(700))),
                "test_fatal_cugraph_call",
            )
        }
        .expect_err("fatal CUDA classification must fail");

        assert!(error.is_persistent());
        assert!(error.to_string().contains("cuda_failure_classification: fatal"));
        let snapshot = scope.health_snapshot(device);
        assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.fault_count(), 1);
        let first = snapshot.first_fault().expect("fatal fault");
        assert_eq!(first.kind(), DeviceFaultKind::FatalDeviceOperation);
        assert_eq!(first.operation(), "test_fatal_cugraph_call");
        assert_eq!(first.native_status(), Some(DeviceFaultNativeStatus::Cuda(700)));
    }
}
