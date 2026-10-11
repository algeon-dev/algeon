use std::path::Path;

use query_planning::source::{ParquetSourceLocation, redacted_path_diagnostic};

use crate::exec::source::{
    LocalParquetFile, LocalParquetSource, QueryError, SourceChunk, cudf_query_error, cudf_query_error_without_source,
};

pub(in crate::exec::source) fn path_fact(path: &Path) -> String {
    redacted_path_diagnostic(path)
}

pub(in crate::exec::source) fn source_has_object_store_location(source: &LocalParquetSource) -> bool {
    source.files.iter().any(|file| !file.location.is_local_path())
}

/// Wraps a cuDF error raised while converting or reading `source`.
///
/// cuDF error text for an object-store location can contain the object URI, including query
/// parameters such as presigned signatures. If any file of the source is not a local path, the
/// wrapper keeps the cuDF classification and a redacted `cudf_message` fact and drops the source
/// chain, so `Display` and `Debug` of the `QueryError` do not print the URI. Local paths keep the
/// chain. `cudf_error_for_location`, `cudf_error_for_chunk` and `cudf_error_for_file` apply the
/// same rule to a single location.
pub(in crate::exec::source) fn cudf_error_for_source(
    source: &LocalParquetSource,
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
) -> QueryError {
    if source_has_object_store_location(source) {
        cudf_query_error_without_source(operation, message, &error)
    } else {
        cudf_query_error(operation, message, error)
    }
}

pub(in crate::exec::source) fn cudf_error_for_location(
    location: &ParquetSourceLocation,
    operation: &'static str,
    message: &'static str,
    error: cudf::Error,
) -> QueryError {
    if location.is_local_path() {
        cudf_query_error(operation, message, error)
    } else {
        cudf_query_error_without_source(operation, message, &error)
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
