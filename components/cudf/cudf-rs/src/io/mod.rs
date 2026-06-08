//! I/O operations for reading and writing tables to various file formats.

use std::{
    ffi::{CStr, CString, c_char},
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
        names.push(cstr.to_string_lossy().into_owned());
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
}
