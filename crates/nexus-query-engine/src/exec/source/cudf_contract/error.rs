use std::path::Path;

use super::super::*;

pub(in crate::exec::source) fn path_fact(path: &Path) -> String {
    crate::source::redacted_path_diagnostic(path)
}

pub(in crate::exec::source) fn source_has_object_store_location(source: &LocalParquetSource) -> bool {
    source.files.iter().any(|file| !file.location.is_local_path())
}

pub(in crate::exec::source) fn cudf_error_for_source(
    source: &LocalParquetSource,
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
) -> QueryError {
    if source_has_object_store_location(source) {
        QueryError::from_cudf_without_source(operation, message, &error)
    } else {
        QueryError::from_cudf(operation, message, error)
    }
}

pub(in crate::exec::source) fn cudf_error_for_location(
    location: &crate::source::ParquetSourceLocation,
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
) -> QueryError {
    if location.is_local_path() {
        QueryError::from_cudf(operation, message, error)
    } else {
        QueryError::from_cudf_without_source(operation, message, &error)
    }
}

pub(in crate::exec::source) fn cudf_error_for_chunk(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    chunk: &SourceChunk,
) -> QueryError {
    cudf_error_for_location(&chunk.location, operation, message, error)
}

pub(in crate::exec::source) fn cudf_error_for_file(
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
    file: &LocalParquetFile,
) -> QueryError {
    cudf_error_for_location(&file.location, operation, message, error)
}
