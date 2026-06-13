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
//! - [`arrow`] exposes Arrow conversion helpers for [`crate::DataFrame`] using
//!   the Rust `arrow` crate's [`arrow::record_batch::RecordBatch`]. That path
//!   is host-copy interop, not zero-copy external GPU interop. If you need
//!   Arrow C Device ABI handoff for existing device buffers, use
//!   [`crate::interop::arrow_device`] instead.
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
    fs, io as stdio,
    path::Path,
};

use crate::core::{
    error::{self, Error, ErrorCode, ErrorKind, Result},
    table::Table,
};

pub mod arrow;
pub mod csv;
pub mod json;
pub mod parquet;

/// Shared output envelope for low-level table readers.
pub struct TableReadOutput {
    pub table: Table,
    pub column_names: Option<Vec<String>>,
}

fn classify_io_error_kind(kind: stdio::ErrorKind) -> ErrorKind {
    match kind {
        stdio::ErrorKind::NotFound => ErrorKind::NotFound,
        _ => ErrorKind::DependencyFailure,
    }
}

fn map_path_io_error(
    source: impl StdError + Send + Sync + 'static,
    kind: ErrorKind,
    path: &Path,
    operation: &'static str,
    message: &'static str,
) -> Error {
    let err = Error::new(kind, message)
        .with_operation(operation)
        .with_context("path", path.display().to_string())
        .set_source(source);
    if kind == ErrorKind::NotFound { err.with_code(ErrorCode::MissingResource) } else { err }
}

fn map_std_io_error(source: stdio::Error, path: &Path, operation: &'static str, message: &'static str) -> Error {
    let kind = classify_io_error_kind(source.kind());
    map_path_io_error(source, kind, path, operation, message)
}

fn map_csv_error(source: ::csv::Error, path: &Path, operation: &'static str, message: &'static str) -> Error {
    let kind = match source.kind() {
        ::csv::ErrorKind::Io(io_error) => classify_io_error_kind(io_error.kind()),
        _ => ErrorKind::DependencyFailure,
    };
    map_path_io_error(source, kind, path, operation, message)
}

fn validate_input_path(path: &Path, operation: &'static str) -> Result<()> {
    fs::metadata(path)
        .map(|_| ())
        .map_err(|source| map_std_io_error(source, path, operation, "failed to access input path"))
}

fn interior_null_byte_error(operation: &'static str, resource: &'static str, message: &'static str) -> Error {
    error::interior_null_byte_error(operation, resource, message)
}

/// Convert a filesystem path to a `CString` for FFI, with operation context.
fn path_to_cstring(path: &Path, operation: &'static str) -> Result<CString> {
    let path_str = path.to_str().ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, "path is not valid UTF-8")
            .with_code(ErrorCode::InvalidInput)
            .with_operation(operation)
            .with_context("resource", "path")
            .with_context("reason", "invalid_utf8")
    })?;
    CString::new(path_str).map_err(|_| {
        interior_null_byte_error(operation, "path", "path contains interior null byte")
            .with_context("path", path.to_string_lossy().into_owned())
    })
}

unsafe fn collect_column_names(
    names_ptr: *mut *mut c_char,
    num_names: i32,
    operation: &'static str,
) -> Result<Vec<String>> {
    if num_names < 0 {
        return Err(Error::new(ErrorKind::InvariantViolation, "ffi returned negative column name count")
            .with_code(ErrorCode::InvariantViolation)
            .with_operation(operation)
            .with_context("column_name_count", num_names.to_string())
            .with_context("resource", "column_name_count")
            .with_context("reason", "negative_name_count")
            .with_context("remedy", "ensure the foreign reader reports a non-negative column name count"));
    }

    if num_names > 0 && names_ptr.is_null() {
        return Err(Error::new(
            ErrorKind::InvariantViolation,
            "ffi returned null column name array with positive count",
        )
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("column_name_count", num_names.to_string())
        .with_context("resource", "column_names")
        .with_context("reason", "null_name_array")
        .with_context(
            "remedy",
            "ensure the foreign reader returns a name array whenever column_name_count is positive",
        ));
    }

    let mut names = Vec::with_capacity(num_names as usize);
    for i in 0..num_names {
        let name_ptr = *names_ptr.add(i as usize);
        if name_ptr.is_null() {
            return Err(Error::new(ErrorKind::InvariantViolation, "ffi returned null column name entry")
                .with_code(ErrorCode::InvariantViolation)
                .with_operation(operation)
                .with_context("column_name_count", num_names.to_string())
                .with_context("column_index", i.to_string())
                .with_context("resource", "column_name")
                .with_context("reason", "null_name_entry")
                .with_context(
                    "remedy",
                    "ensure the foreign reader returns one non-null name entry per reported column",
                ));
        }

        let cstr = CStr::from_ptr(name_ptr);
        let name = cstr.to_str().map_err(|_| {
            Error::new(ErrorKind::InvariantViolation, "column name from FFI is not valid UTF-8")
                .with_code(ErrorCode::InvariantViolation)
                .with_operation(operation)
                .with_context("column_name_count", num_names.to_string())
                .with_context("column_index", i.to_string())
                .with_context("resource", "column_name")
                .with_context("reason", "invalid_utf8")
                .with_context("remedy", "ensure the foreign reader returns valid UTF-8 column names")
        })?;
        names.push(name.to_owned());
    }

    Ok(names)
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, os::raw::c_char, ptr};

    use crate::core::error::{ErrorCode, ErrorKind};

    #[test]
    fn test_collect_column_names_rejects_negative_count() {
        let err = unsafe { super::collect_column_names(ptr::null_mut(), -1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvariantViolation);
        assert_eq!(err.code(), ErrorCode::InvariantViolation);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "column_name_count" && v == "-1"));
        assert!(!err.context().iter().any(|(k, _)| *k == "num_names"));
        assert!(err.context().iter().any(|(k, v)| *k == "resource" && v == "column_name_count"));
        assert!(err.context().iter().any(|(k, v)| *k == "reason" && v == "negative_name_count"));
        assert!(
            err.context()
                .iter()
                .any(|(k, v)| *k == "remedy"
                    && v == "ensure the foreign reader reports a non-negative column name count")
        );
    }

    #[test]
    fn test_collect_column_names_rejects_null_names_ptr_with_positive_count() {
        let err = unsafe { super::collect_column_names(ptr::null_mut(), 1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvariantViolation);
        assert_eq!(err.code(), ErrorCode::InvariantViolation);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "column_name_count" && v == "1"));
        assert!(err.context().iter().any(|(k, v)| *k == "resource" && v == "column_names"));
        assert!(err.context().iter().any(|(k, v)| *k == "reason" && v == "null_name_array"));
        assert!(err.context().iter().any(|(k, v)| *k == "remedy"
            && v == "ensure the foreign reader returns a name array whenever column_name_count is positive"));
        assert!(!err.context().iter().any(|(k, _)| *k == "names_ptr_null"));
    }

    #[test]
    fn test_collect_column_names_rejects_null_entry_pointer() {
        let mut ptrs: Vec<*mut c_char> = vec![ptr::null_mut()];
        let err = unsafe { super::collect_column_names(ptrs.as_mut_ptr(), 1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvariantViolation);
        assert_eq!(err.code(), ErrorCode::InvariantViolation);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "column_index" && v == "0"));
        assert!(err.context().iter().any(|(k, v)| *k == "resource" && v == "column_name"));
        assert!(err.context().iter().any(|(k, v)| *k == "reason" && v == "null_name_entry"));
        assert!(err.context().iter().any(|(k, v)| *k == "remedy"
            && v == "ensure the foreign reader returns one non-null name entry per reported column"));
        assert!(!err.context().iter().any(|(k, _)| *k == "name_index"));
    }

    #[test]
    fn test_collect_column_names_success() {
        let mut raw = [CString::new("alpha").unwrap(), CString::new("beta").unwrap()];
        let mut ptrs: Vec<*mut c_char> = raw.iter_mut().map(|s| s.as_ptr() as *mut c_char).collect();
        let names = unsafe { super::collect_column_names(ptrs.as_mut_ptr(), ptrs.len() as i32, "test") }.unwrap();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_validate_input_path_reports_not_found() {
        let path = std::env::temp_dir().join("cudf_rs_missing_input_path_test_12345.csv");
        let err = super::validate_input_path(&path, "test").unwrap_err();

        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "path" && v == &path.display().to_string()));
    }
}
