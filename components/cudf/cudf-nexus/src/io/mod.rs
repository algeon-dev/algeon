//! Low-level I/O entry points and format-specific modules.
//!
//! Most users should start with the high-level [`crate::DataFrame`] helpers
//! such as `DataFrame::read_csv`, `DataFrame::read_json`,
//! `DataFrame::read_parquet`, `DataFrame::write_csv`,
//! `DataFrame::write_json`, or `DataFrame::write_parquet`. `cudf::io` is the
//! lower-level table-oriented layer that those conveniences build on.
//!
//! # Module Shape
//!
//! - [`csv`], [`json`], and [`parquet`] expose file-format readers and writers
//!   over low-level owned [`Table`] values.
//! - Host-copy Arrow `RecordBatch` interop lives under
//!   [`crate::interop::arrow_host`], not under `cudf::io`, because it is an
//!   external data interchange boundary rather than a file-format reader or
//!   writer.
//!
//! # Naming and Ownership
//!
//! Low-level file readers return [`TableReadOutput`] instead of a bare
//! [`Table`] because `Table` itself is unnamed.
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
//! CSV, JSON, and Parquet option types may carry an explicit [`crate::Runtime`].
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

pub mod csv;
pub mod json;
pub mod parquet;
pub mod source;

/// Process-default KvikIO compatibility mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvikioCompatMode {
    Auto,
    On,
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
}

/// Configure KvikIO compatibility without mutating process environment.
///
/// Call this before starting I/O workers. Existing KvikIO file handles retain
/// the mode selected when they were created.
pub fn set_kvikio_compat_mode(mode: KvikioCompatMode) -> Result<()> {
    const OPERATION: &str = "io::set_kvikio_compat_mode";
    error::check_status_in_with_details(
        // SAFETY: the mode is one of the native enum's declared values.
        unsafe { cudf_sys::cudf_io_kvikio_set_compat_mode(mode.as_ffi()) },
        OPERATION,
        [("resource", "kvikio_defaults".to_string())],
    )
}

/// Return KvikIO's configured process default, when this build supports it.
pub fn kvikio_compat_mode() -> Option<KvikioCompatMode> {
    // SAFETY: this getter takes no pointers and returns a small integer tag.
    match unsafe { cudf_sys::cudf_io_kvikio_compat_mode() } {
        value if value == cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_AUTO as i32 => Some(KvikioCompatMode::Auto),
        value if value == cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_ON as i32 => Some(KvikioCompatMode::On),
        value if value == cudf_sys::CUDF_IO_KVIKIO_COMPAT_MODE_OFF as i32 => Some(KvikioCompatMode::Off),
        _ => None,
    }
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
    Err(Error::new_with(
        ErrorCode::IoBusy,
        "io::drain_kvikio_bounce_buffer_pools",
        "cannot drain KvikIO bounce buffers while I/O is active",
    )
    .with_detail("resource", "kvikio_bounce_buffer_pools")
    .with_detail("reason", "active_io")
    .with_detail("active_io", active_io.to_string())
    .with_detail("remedy", "stop and join all I/O workers before retrying bounce-buffer drain"))
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
/// error kind was `NotFound` or another flavor: the caller is performing
/// local file I/O, and the failure surface is I/O-shaped, not a
/// missing-named-resource contract failure. The exact underlying cause
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
        // presented to libcudf's C API. Classified as a domestic
        // `IoFailure` rather than a foreign or invariant bucket because
        // the boundary is our own local file I/O contract.
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
    fn kvikio_native_lifecycle_gate_rejects_drain_while_io_is_active() {
        struct ActiveIoGuard(bool);

        impl Drop for ActiveIoGuard {
            fn drop(&mut self) {
                if !self.0 {
                    return;
                }
                // SAFETY: this guard owns the matching test-only active-I/O
                // increment. Cleanup is best-effort while unwinding; the
                // normal path below asserts the status explicitly.
                let _ = unsafe { cudf_sys::test_support::cudf_test_io_kvikio_active_io_end() };
            }
        }

        // SAFETY: the test hook takes no pointers and the guard balances the
        // process-global lifecycle count even if the assertion below panics.
        assert_eq!(unsafe { cudf_sys::test_support::cudf_test_io_kvikio_active_io_begin() }, cudf_sys::CUDF_SUCCESS);
        let mut active_io = ActiveIoGuard(true);
        if super::kvikio_compat_mode().is_some() {
            let error = super::drain_kvikio_bounce_buffer_pools().unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoBusy);
            assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
            assert_eq!(error.status(), ErrorStatus::Temporary);
            assert_eq!(error.detail_value("reason"), Some("active_io"));
        }
        // SAFETY: this is the matching decrement for the begin call above.
        let status = unsafe { cudf_sys::test_support::cudf_test_io_kvikio_active_io_end() };
        active_io.0 = false;
        assert_eq!(status, cudf_sys::CUDF_SUCCESS);
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
