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
};

use crate::core::{
    error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
    table::Table,
};

pub mod csv;
pub mod json;
pub mod parquet;
pub mod source;

/// Shared output envelope for low-level table readers.
pub struct TableReadOutput {
    pub table: Table,
    pub column_names: Option<Vec<String>>,
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
        let name_ptr = *names_ptr.add(i as usize);
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

        let cstr = CStr::from_ptr(name_ptr);
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
