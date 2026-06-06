//! I/O operations for reading and writing tables to various file formats.

use std::{ffi::CString, path::Path};

use crate::core::error::{Error, ErrorKind, Result};

pub mod arrow;
pub mod csv;
pub mod parquet;

/// Convert a filesystem path to a `CString` for FFI, with operation context.
fn path_to_cstring(path: &Path, operation: &'static str) -> Result<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "path is not valid UTF-8").with_operation(operation))?;
    CString::new(path_str).map_err(|_| {
        Error::new(ErrorKind::InvalidOperation, "path contains interior null byte").with_operation(operation)
    })
}
