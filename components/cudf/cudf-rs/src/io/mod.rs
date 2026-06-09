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
//!   over low-level [`Table`](crate::core::table::Table) and
//!   [`TableView`](crate::core::table::TableView) values.
//! - [`arrow`] exposes Arrow conversion helpers for [`crate::DataFrame`].
//!   That path is host-copy interop, not zero-copy external GPU interop.
//!
//! # Naming and Ownership
//!
//! Low-level file readers return [`IoTableResult`] instead of a bare
//! [`Table`](crate::core::table::Table) because `Table` itself is unnamed.
//! When a format can report schema names, they are returned separately via
//! `column_names` so callers can decide whether to rebuild a named
//! [`crate::DataFrame`] or keep working at the low-level table boundary.
//!
//! Writers borrow [`TableView`](crate::core::table::TableView) inputs and do
//! not rename or otherwise mutate the source table. Any output names supplied
//! through write options only affect the serialized file format.
//!
//! # Runtime Placement
//!
//! CSV, JSON, and Parquet option types may carry an explicit [`crate::Runtime`].
//! For reads, that runtime determines the execution provenance retained by the
//! returned low-level table. For writes, it controls the explicit execution
//! context used for the I/O operation itself while the input table remains a
//! borrow.

use std::{
    error::Error as StdError,
    ffi::{CStr, CString, c_char},
    fs, io as stdio,
    path::Path,
};

use crate::core::{
    error::{Error, ErrorKind, Result},
    table::Table,
};

pub mod arrow;
pub mod csv;
pub mod json;
pub mod parquet;

/// Shared result envelope for low-level table readers.
pub struct IoTableResult {
    pub table: Table,
    pub column_names: Option<Vec<String>>,
}

fn classify_io_error_kind(kind: stdio::ErrorKind) -> ErrorKind {
    match kind {
        stdio::ErrorKind::NotFound => ErrorKind::NotFound,
        _ => ErrorKind::IoError,
    }
}

fn map_path_io_error(
    source: impl StdError + Send + Sync + 'static,
    kind: ErrorKind,
    path: &Path,
    operation: &'static str,
    message: &'static str,
) -> Error {
    Error::new(kind, message)
        .with_operation(operation)
        .with_context("path", path.display().to_string())
        .set_source(source)
}

fn map_std_io_error(source: stdio::Error, path: &Path, operation: &'static str, message: &'static str) -> Error {
    let kind = classify_io_error_kind(source.kind());
    map_path_io_error(source, kind, path, operation, message)
}

fn map_csv_error(source: ::csv::Error, path: &Path, operation: &'static str, message: &'static str) -> Error {
    let kind = match source.kind() {
        ::csv::ErrorKind::Io(io_error) => classify_io_error_kind(io_error.kind()),
        _ => ErrorKind::IoError,
    };
    map_path_io_error(source, kind, path, operation, message)
}

fn validate_input_path(path: &Path, operation: &'static str) -> Result<()> {
    fs::metadata(path)
        .map(|_| ())
        .map_err(|source| map_std_io_error(source, path, operation, "failed to access input path"))
}

/// Convert a filesystem path to a `CString` for FFI, with operation context.
fn path_to_cstring(path: &Path, operation: &'static str) -> Result<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::new(ErrorKind::InvalidArgument, "path is not valid UTF-8").with_operation(operation))?;
    CString::new(path_str).map_err(|_| {
        Error::new(ErrorKind::InvalidArgument, "path contains interior null byte").with_operation(operation)
    })
}

unsafe fn collect_column_names(
    names_ptr: *mut *mut c_char,
    num_names: i32,
    operation: &'static str,
) -> Result<Vec<String>> {
    if num_names < 0 {
        return Err(Error::new(ErrorKind::InvalidOperation, "ffi returned negative column name count")
            .with_operation(operation)
            .with_context("num_names", num_names.to_string()));
    }

    if num_names > 0 && names_ptr.is_null() {
        return Err(Error::new(ErrorKind::InvalidOperation, "ffi returned null column name array with positive count")
            .with_operation(operation)
            .with_context("num_names", num_names.to_string())
            .with_context("names_ptr_null", "true"));
    }

    let mut names = Vec::with_capacity(num_names as usize);
    for i in 0..num_names {
        let name_ptr = *names_ptr.add(i as usize);
        if name_ptr.is_null() {
            return Err(Error::new(ErrorKind::InvalidOperation, "ffi returned null column name entry")
                .with_operation(operation)
                .with_context("num_names", num_names.to_string())
                .with_context("name_index", i.to_string()));
        }

        let cstr = CStr::from_ptr(name_ptr);
        let name = cstr.to_str().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "column name from FFI is not valid UTF-8")
                .with_operation(operation)
                .with_context("name_index", i.to_string())
        })?;
        names.push(name.to_owned());
    }

    Ok(names)
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, os::raw::c_char, ptr};

    use crate::core::error::ErrorKind;

    #[test]
    fn test_collect_column_names_rejects_negative_count() {
        let err = unsafe { super::collect_column_names(ptr::null_mut(), -1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "num_names" && v == "-1"));
    }

    #[test]
    fn test_collect_column_names_rejects_null_names_ptr_with_positive_count() {
        let err = unsafe { super::collect_column_names(ptr::null_mut(), 1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "names_ptr_null" && v == "true"));
    }

    #[test]
    fn test_collect_column_names_rejects_null_entry_pointer() {
        let mut ptrs: Vec<*mut c_char> = vec![ptr::null_mut()];
        let err = unsafe { super::collect_column_names(ptrs.as_mut_ptr(), 1, "test") }.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "test");
        assert!(err.context().iter().any(|(k, v)| *k == "name_index" && v == "0"));
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
