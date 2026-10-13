//! Low-level I/O entry points and format-specific modules.
//!
//! Most users should start with the high-level [`crate::DataFrame`] helpers
//! such as `DataFrame::read_parquet` or `DataFrame::write_parquet`. `cudf::io` is the
//! lower-level table-oriented layer that those conveniences build on.
//!
//! # Module Shape
//!
//! - [`parquet`] exposes file-format readers and writers over low-level owned
//!   [`Table`] values.
//! - Host-copy Arrow `RecordBatch` interop lives under
//!   [`crate::interop::arrow_host`], outside `cudf::io`, because it is an
//!   external data interchange boundary. That module does not read or write
//!   file formats.
//!
//! # Naming and Ownership
//!
//! Low-level file readers return [`TableReadOutput`], which pairs the
//! [`Table`] with optional schema names, because `Table` itself is unnamed.
//! When a format can report schema names, they are returned separately via
//! `column_names` so callers can decide whether to rebuild a named
//! [`crate::DataFrame`] or keep working at the low-level table boundary.
//!
//! Writers borrow owned [`Table`] inputs and do not rename or otherwise mutate
//! the source table. Any output names supplied
//! through write options only affect the serialized file format.
//!
//! # Runtime Placement
//!
//! Parquet option types may carry an explicit [`crate::Runtime`].
//! For reads, that runtime determines the device, stream, and memory resource
//! retained by the returned low-level table. For writes, it controls the
//! device-pinned stream and memory resource used for the I/O operation itself
//! while the input table remains a borrow.

use std::{
    error::Error as StdError,
    ffi::{CStr, CString, c_char},
    fs,
    path::Path,
    sync::Arc,
};

use crate::{
    core::{
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
        table::{Table, TableView},
    },
    runtime::{Runtime, execution_resolver::ResolvedExecution},
};

pub mod parquet;
pub mod source;

/// KvikIO compatibility mode, the `KVIKIO_COMPAT_MODE` setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvikioCompatMode {
    /// cuFile where the file system and driver support it, POSIX otherwise.
    Auto,
    /// POSIX reads through a pinned host bounce buffer.
    On,
    /// cuFile only; a read fails where cuFile is unavailable.
    Off,
}

impl KvikioCompatMode {
    const fn as_ffi(self) -> i32 {
        match self {
            Self::Auto => cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_AUTO as i32,
            Self::On => cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_ON as i32,
            Self::Off => cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_OFF as i32,
        }
    }

    fn from_ffi(value: i32) -> Option<Self> {
        match value {
            value if value == cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_AUTO as i32 => Some(Self::Auto),
            value if value == cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_ON as i32 => Some(Self::On),
            value if value == cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_OFF as i32 => Some(Self::Off),
            _ => None,
        }
    }

    /// KvikIO's lowercase spelling of the mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

/// KvikIO's process-global settings: one thread pool and one task size shared
/// by local compatibility-mode reads and remote range reads, across every
/// device in the process.
///
/// Design: `website/docs/design/data-path.mdx#local-nvme-and-compatibility-mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvikioProcessConfig {
    /// `KVIKIO_COMPAT_MODE`.
    pub compat_mode: KvikioCompatMode,
    /// `KVIKIO_NTHREADS`: workers in the process-wide pool.
    pub thread_count: u32,
    /// `KVIKIO_TASK_SIZE`: bytes per pool task; one read splits into tasks of
    /// at most this size.
    pub task_size_bytes: u64,
}

impl KvikioProcessConfig {
    /// Algeon's values for the fields the KvikIO environment leaves unset.
    ///
    /// 64 threads keep a cold local NVMe read near the device's throughput,
    /// where libcudf's own four-thread setup reached about a quarter of it,
    /// and match the concurrency remote range reads need. 4 MiB tasks are
    /// KvikIO's default; a larger task splits one read into fewer parallel
    /// pieces.
    pub const DEFAULT: Self =
        Self { compat_mode: KvikioCompatMode::On, thread_count: 64, task_size_bytes: 4 * 1024 * 1024 };

    fn from_ffi(raw: &cudf_sys::cudf_io_kvikio_process_config_t) -> Option<Self> {
        Some(Self {
            compat_mode: KvikioCompatMode::from_ffi(raw.compat_mode)?,
            thread_count: raw.thread_count,
            task_size_bytes: raw.task_size_bytes,
        })
    }

    const fn as_ffi(&self) -> cudf_sys::cudf_io_kvikio_process_config_t {
        cudf_sys::cudf_io_kvikio_process_config_t {
            compat_mode: self.compat_mode.as_ffi(),
            thread_count: self.thread_count,
            task_size_bytes: self.task_size_bytes,
        }
    }
}

const fn zeroed_kvikio_process_config() -> cudf_sys::cudf_io_kvikio_process_config_t {
    cudf_sys::cudf_io_kvikio_process_config_t { compat_mode: 0, thread_count: 0, task_size_bytes: 0 }
}

/// Apply `defaults` to every KvikIO process setting the environment leaves
/// unset, and return the effective configuration.
///
/// Call this once per process before any I/O worker starts. libcudf's own
/// once-only KvikIO setup runs first, with `device_ordinal` current, so a
/// later local file open cannot reset the pool; the ordinal must be a device
/// the caller owns because that setup initializes the CUDA runtime on it. A
/// field named by `KVIKIO_COMPAT_MODE`, `KVIKIO_NTHREADS` or
/// `KVIKIO_TASK_SIZE` keeps the environment's value. Active I/O returns
/// [`ErrorCode::IoBusy`] without changing anything, and a cuDF build without
/// KvikIO support returns [`ErrorCode::UnsupportedOperation`]. Repeated calls
/// are idempotent.
pub fn configure_kvikio_process(device_ordinal: usize, defaults: &KvikioProcessConfig) -> Result<KvikioProcessConfig> {
    const OPERATION: &str = "io::configure_kvikio_process";
    let device_ordinal = i32::try_from(device_ordinal).map_err(|_| {
        Error::new_with(ErrorCode::ForeignInvalidArgument, OPERATION, "device ordinal exceeds the CUDA device range")
            .with_detail("device_ordinal", device_ordinal.to_string())
    })?;
    let raw_defaults = defaults.as_ffi();
    let mut raw_effective = zeroed_kvikio_process_config();
    let mut active_io = 0u64;
    error::check_status_in_with_details(
        // SAFETY: every pointer refers to a live, aligned stack value for the
        // duration of the call.
        unsafe {
            cudf_sys::cudf_io_kvikio_configure_process(
                device_ordinal,
                &raw_defaults,
                &mut raw_effective,
                &mut active_io,
            )
        },
        OPERATION,
        [("resource", "kvikio_defaults".to_string())],
    )?;
    kvikio_configure_result(&raw_effective, active_io)
}

fn kvikio_configure_result(
    raw_effective: &cudf_sys::cudf_io_kvikio_process_config_t,
    active_io: u64,
) -> Result<KvikioProcessConfig> {
    const OPERATION: &str = "io::configure_kvikio_process";
    if active_io != 0 {
        return Err(kvikio_busy_error(
            OPERATION,
            "cannot configure KvikIO while I/O is active",
            "kvikio_defaults",
            active_io,
            "stop and join all I/O workers before configuring KvikIO",
        ));
    }
    KvikioProcessConfig::from_ffi(raw_effective).ok_or_else(|| {
        Error::new_with(
            ErrorCode::ForeignInvariantViolation,
            OPERATION,
            "KvikIO reported an unknown compatibility mode",
        )
        .with_detail("compat_mode", raw_effective.compat_mode.to_string())
    })
}

/// Return KvikIO's effective process settings, or `None` for a cuDF build
/// without KvikIO support.
pub fn kvikio_process_config() -> Option<KvikioProcessConfig> {
    let mut raw = zeroed_kvikio_process_config();
    // SAFETY: the out-pointer refers to a live aligned stack value.
    let status = unsafe { cudf_sys::cudf_io_kvikio_process_config(&mut raw) };
    (status == cudf_sys::CUDF_SUCCESS).then(|| KvikioProcessConfig::from_ffi(&raw)).flatten()
}

fn kvikio_busy_error(
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    active_io: u64,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::IoBusy, operation, message)
        .with_detail("resource", resource)
        .with_detail("reason", "active_io")
        .with_detail("active_io", active_io.to_string())
        .with_detail("remedy", remedy)
}

/// Drain KvikIO's retained host bounce-buffer pools.
///
/// This operation is idempotent. It must run after I/O workers stop and before
/// CUDA teardown. Active I/O returns [`ErrorCode::IoBusy`] without clearing
/// any buffer.
pub fn drain_kvikio_bounce_buffer_pools() -> Result<u64> {
    const OPERATION: &str = "io::drain_kvikio_bounce_buffer_pools";
    let mut cleared_bytes = 0u64;
    let mut active_io = 0u64;
    error::check_status_in_with_details(
        // SAFETY: both out-pointers refer to live aligned u64 stack values.
        unsafe { cudf_sys::cudf_io_kvikio_drain_bounce_buffer_pools(&mut cleared_bytes, &mut active_io) },
        OPERATION,
        [("resource", "kvikio_bounce_buffer_pools".to_string())],
    )?;
    kvikio_drain_result(cleared_bytes, active_io)
}

fn kvikio_drain_result(cleared_bytes: u64, active_io: u64) -> Result<u64> {
    if active_io == 0 {
        return Ok(cleared_bytes);
    }
    Err(kvikio_busy_error(
        "io::drain_kvikio_bounce_buffer_pools",
        "cannot drain KvikIO bounce buffers while I/O is active",
        "kvikio_bounce_buffer_pools",
        active_io,
        "stop and join all I/O workers before retrying bounce-buffer drain",
    ))
}

/// Probe whether the cuFile driver can be opened by this process.
///
/// This is a deployment capability check for GPUDirect Storage. It only
/// verifies that `libcufile` can be loaded and `cuFileDriverOpen()` succeeds;
/// callers still need their own source-location eligibility checks before
/// choosing a GDS read path.
pub fn gds_driver_available() -> Result<()> {
    const OPERATION: &str = "io::gds_driver_available";
    error::check_status_in_with_details(
        // SAFETY: The shim takes no borrowed pointers and performs a
        // process-local cuFile driver open/close probe.
        unsafe { cudf_sys::cudf_gds_driver_available() },
        OPERATION,
        [("resource", "cufile_driver".to_string())],
    )
}

/// Shared output envelope for low-level table readers.
pub struct TableReadOutput {
    pub table: Table,
    pub column_names: Option<Vec<String>>,
}

fn resolve_write_execution(
    table: &TableView<'_>,
    runtime: Option<&Runtime>,
    operation: &'static str,
) -> Result<ResolvedExecution> {
    let write_domain = runtime
        .map(|runtime| Arc::clone(runtime.execution_domain()))
        .unwrap_or_else(|| Arc::clone(table.execution_domain()));
    ResolvedExecution::with_output_domain(write_domain, [table.execution_domain()], operation)
}

/// Build a domestic IO error carrying the offending path and the underlying
/// source error.
///
/// Local filesystem and parser IO failures are classified as
/// `ErrorCode::IoFailure` regardless of whether the underlying `std::io`
/// error kind was `NotFound` or another flavor. The caller performs local
/// file I/O, so the failure surface is I/O-shaped, and the error kind is
/// `DependencyFailure` even for a missing file. The exact underlying cause
/// stays reachable through the standard source chain.
fn map_path_io_error(
    source: impl StdError + Send + Sync + 'static,
    path: &Path,
    operation: &'static str,
    message: &'static str,
) -> Error {
    Error::new_with(ErrorCode::IoFailure, operation, message)
        .with_detail("path", path.display().to_string())
        .with_source(source)
}

fn validate_input_path(path: &Path, operation: &'static str) -> Result<()> {
    if path.to_string_lossy().contains("://") {
        return Ok(());
    }

    let metadata = fs::metadata(path)
        .map_err(|source| map_path_io_error(source, path, operation, "failed to access input path"))?;

    if metadata.is_file() {
        return Ok(());
    }

    if metadata.is_dir() {
        return Err(Error::new_with(ErrorCode::PathIsDirectory, operation, "input path is a directory")
            .with_detail("path", path.display().to_string())
            .with_detail("resource", "input_path")
            .with_detail("reason", "path_is_directory")
            .with_detail("remedy", "provide a regular file path instead of a directory"));
    }

    Err(Error::new_with(ErrorCode::IoFailure, operation, "input path is not a regular file")
        .with_detail("path", path.display().to_string())
        .with_detail("resource", "input_path")
        .with_detail("reason", "path_not_regular_file")
        .with_detail("remedy", "provide a regular file path"))
}

/// Convert a filesystem path to a `CString` for FFI, with operation context.
fn path_to_cstring(path: &Path, operation: &'static str) -> Result<CString> {
    let path_str = path.to_str().ok_or_else(|| {
        // Local-IO precondition failure: the filesystem path cannot be
        // presented to libcudf's C API. The boundary is this crate's local
        // file I/O contract, so the failure is a domestic `IoFailure`.
        Error::new_with(ErrorCode::IoFailure, operation, "path is not valid UTF-8")
            .with_detail("resource", "path")
            .with_detail("reason", "invalid_utf8")
    })?;
    CString::new(path_str).map_err(|_| {
        error::interior_null_byte_error(operation, "path", "path contains interior null byte")
            .with_detail("path", path.to_string_lossy().into_owned())
    })
}

unsafe fn collect_column_names(
    names_ptr: *mut *mut c_char,
    num_names: i32,
    operation: &'static str,
) -> Result<Vec<String>> {
    if num_names < 0 {
        return Err(error::foreign_invariant_violation(
            operation,
            "ffi returned negative column name count",
            "column_name_count",
            "negative_name_count",
            "ensure the foreign reader reports a non-negative column name count",
        )
        .with_detail("column_name_count", num_names.to_string()));
    }

    if num_names > 0 && names_ptr.is_null() {
        return Err(error::foreign_invariant_violation(
            operation,
            "ffi returned null column name array with positive count",
            "column_names",
            "null_name_array",
            "ensure the foreign reader returns a name array whenever column_name_count is positive",
        )
        .with_detail("column_name_count", num_names.to_string()));
    }

    let mut names = Vec::with_capacity(num_names as usize);
    for i in 0..num_names {
        // SAFETY: the caller guarantees `names_ptr` addresses at least
        // `num_names` entries; the positive-count null case was rejected.
        let name_ptr = unsafe { *names_ptr.add(i as usize) };
        if name_ptr.is_null() {
            return Err(error::foreign_invariant_violation(
                operation,
                "ffi returned null column name entry",
                "column_name",
                "null_name_entry",
                "ensure the foreign reader returns one non-null name entry per reported column",
            )
            .with_detail("column_name_count", num_names.to_string())
            .with_fact(FACT_COLUMN_INDEX, i as usize));
        }

        // SAFETY: the FFI contract requires each non-null entry to reference
        // a NUL-terminated string for the duration of this conversion.
        let cstr = unsafe { CStr::from_ptr(name_ptr) };
        let name = cstr.to_str().map_err(|_| {
            error::foreign_invariant_violation(
                operation,
                "column name from FFI is not valid UTF-8",
                "column_name",
                "invalid_utf8",
                "ensure the foreign reader returns valid UTF-8 column names",
            )
            .with_detail("column_name_count", num_names.to_string())
            .with_fact(FACT_COLUMN_INDEX, i as usize)
        })?;
        names.push(name.to_owned());
    }

    Ok(names)
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, fs, os::raw::c_char, ptr};

    use crate::core::error::{ErrorCode, ErrorKind, ErrorStatus, FactKeyName, FactValue};

    #[test]
    fn kvikio_bounce_buffer_drain_busy_error_is_flat_and_retryable() {
        let error = super::kvikio_drain_result(0, 2).unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoBusy);
        assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
        assert_eq!(error.status(), ErrorStatus::Temporary);
        assert_eq!(error.detail_value("active_io"), Some("2"));
        assert_eq!(error.detail_value("reason"), Some("active_io"));
    }

    #[test]
    fn kvikio_bounce_buffer_drain_idle_result_is_idempotent() {
        assert_eq!(super::kvikio_drain_result(16 * 1024 * 1024, 0).unwrap(), 16 * 1024 * 1024);
        assert_eq!(super::kvikio_drain_result(0, 0).unwrap(), 0);
    }

    #[test]
    fn test_collect_column_names_rejects_negative_count() {
        let err = unsafe { super::collect_column_names(ptr::null_mut(), -1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvariantViolation);
        assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
        assert_eq!(err.operation(), "test");
        assert!(err.details().iter().any(|(k, v)| *k == "column_name_count" && v == "-1"));
        assert!(!err.details().iter().any(|(k, _)| *k == "num_names"));
        assert!(err.details().iter().any(|(k, v)| *k == "resource" && v == "column_name_count"));
        assert!(err.details().iter().any(|(k, v)| *k == "reason" && v == "negative_name_count"));
        assert!(
            err.details()
                .iter()
                .any(|(k, v)| *k == "remedy"
                    && v == "ensure the foreign reader reports a non-negative column name count")
        );
    }

    #[test]
    fn test_collect_column_names_rejects_null_names_ptr_with_positive_count() {
        let err = unsafe { super::collect_column_names(ptr::null_mut(), 1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvariantViolation);
        assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
        assert_eq!(err.operation(), "test");
        assert!(err.details().iter().any(|(k, v)| *k == "column_name_count" && v == "1"));
        assert!(err.details().iter().any(|(k, v)| *k == "resource" && v == "column_names"));
        assert!(err.details().iter().any(|(k, v)| *k == "reason" && v == "null_name_array"));
        assert!(err.details().iter().any(|(k, v)| *k == "remedy"
            && v == "ensure the foreign reader returns a name array whenever column_name_count is positive"));
        assert!(!err.details().iter().any(|(k, _)| *k == "names_ptr_null"));
    }

    #[test]
    fn test_collect_column_names_rejects_null_entry_pointer() {
        let mut ptrs: Vec<*mut c_char> = vec![ptr::null_mut()];
        let err = unsafe { super::collect_column_names(ptrs.as_mut_ptr(), 1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvariantViolation);
        assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
        assert_eq!(err.operation(), "test");
        assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(0)));
        assert!(!err.details().iter().any(|(k, _)| *k == "column_index"));
        assert!(err.details().iter().any(|(k, v)| *k == "resource" && v == "column_name"));
        assert!(err.details().iter().any(|(k, v)| *k == "reason" && v == "null_name_entry"));
        assert!(err.details().iter().any(|(k, v)| *k == "remedy"
            && v == "ensure the foreign reader returns one non-null name entry per reported column"));
        assert!(!err.details().iter().any(|(k, _)| *k == "name_index"));
    }

    #[test]
    fn test_collect_column_names_success() {
        let mut raw = [CString::new("alpha").unwrap(), CString::new("beta").unwrap()];
        let mut ptrs: Vec<*mut c_char> = raw.iter_mut().map(|s| s.as_ptr() as *mut c_char).collect();
        let names = unsafe { super::collect_column_names(ptrs.as_mut_ptr(), ptrs.len() as i32, "test") }.unwrap();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_validate_input_path_reports_io_failure() {
        let path = std::env::temp_dir().join("cudf_rs_missing_input_path_test_12345.csv");
        let err = super::validate_input_path(&path, "test").unwrap_err();

        // Local filesystem IO failures classify as the domestic `IoFailure`
        // leaf. The `DependencyFailure` kind projection is the coarse
        // downstream policy bucket that covers local IO, CUDA runtime, and
        // fatal CUDA.
        assert_eq!(err.code(), ErrorCode::IoFailure);
        assert_eq!(err.kind(), ErrorKind::DependencyFailure);
        assert_eq!(err.operation(), "test");
        assert!(err.details().iter().any(|(k, v)| *k == "path" && v == &path.display().to_string()));
    }

    #[test]
    fn test_validate_input_path_rejects_directory() {
        let path = std::env::temp_dir().join(format!("cudf_rs_directory_input_path_test_{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();

        let result = super::validate_input_path(&path, "test");
        let _ = fs::remove_dir(&path);
        let err = result.unwrap_err();
        let path_display = path.display().to_string();

        assert_eq!(err.code(), ErrorCode::PathIsDirectory);
        assert_eq!(err.code().to_string(), "PathIsDirectory");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.status(), ErrorStatus::Permanent);
        assert_eq!(err.operation(), "test");
        assert_eq!(err.detail_value("path"), Some(path_display.as_str()));
        assert_eq!(err.detail_value("resource"), Some("input_path"));
        assert_eq!(err.detail_value("reason"), Some("path_is_directory"));
        assert_eq!(err.detail_value("remedy"), Some("provide a regular file path instead of a directory"));
    }
}
