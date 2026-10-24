//! Errors for Iceberg scans and catalog calls.
//!
//! A snapshot the scan cannot read correctly fails with
//! [`ErrorCode::UnsupportedTableFeature`] and a log-safe `reason` fact.
//! Catalog and metadata-storage failures map onto the adapter's source codes.

use std::error::Error as StdError;

use algeon_datafusion::error::{Error, ErrorCode};
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind, TableIdent};

pub(crate) const SCAN_OPERATION: &str = "algeon_iceberg.scan";

/// Maximum source-chain layers searched for typed causes and AWS markers.
const MAX_ERROR_CHAIN_DEPTH: usize = 32;

/// A table-format feature the scan does not read. The string form is the
/// `reason` fact of the [`ErrorCode::UnsupportedTableFeature`] error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnsupportedReason {
    DeleteFiles,
    FieldIdMismatch,
    MissingFieldIds,
    PartitionColumnAbsent,
    RequiredColumnAbsent,
    EncryptedFile,
    NonParquetFile,
    MultipleBuckets,
    UnsupportedLocationScheme,
    StoragePropertiesMismatch,
    ColumnTypeMismatch,
}

impl UnsupportedReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DeleteFiles => "iceberg_delete_files",
            Self::FieldIdMismatch => "iceberg_field_id_mismatch",
            Self::MissingFieldIds => "iceberg_missing_field_ids",
            Self::PartitionColumnAbsent => "iceberg_partition_column_absent",
            Self::RequiredColumnAbsent => "iceberg_required_column_absent",
            Self::EncryptedFile => "iceberg_encrypted_file",
            Self::NonParquetFile => "iceberg_non_parquet_file",
            Self::MultipleBuckets => "iceberg_multiple_buckets",
            Self::UnsupportedLocationScheme => "iceberg_unsupported_location_scheme",
            Self::StoragePropertiesMismatch => "iceberg_storage_properties_mismatch",
            Self::ColumnTypeMismatch => "iceberg_column_type_mismatch",
        }
    }
}

/// The rejection of a snapshot that uses `reason`. The table and snapshot
/// id are raw facts because they name caller data.
pub(crate) fn unsupported(
    reason: UnsupportedReason,
    table: &TableIdent,
    snapshot_id: Option<i64>,
    message: impl Into<String>,
) -> Error {
    let error = Error::new(ErrorCode::UnsupportedTableFeature, SCAN_OPERATION, message)
        .with_log_safe_fact("reason", reason.as_str())
        .with_fact("table", table);
    match snapshot_id {
        Some(snapshot_id) => error.with_fact("snapshot_id", snapshot_id),
        None => error,
    }
}

/// An adapter error for a failed catalog or Iceberg metadata call.
pub(crate) fn iceberg_error(operation: &'static str, message: impl Into<String>, source: IcebergError) -> Error {
    let code = iceberg_error_code(&source);
    let mut error = Error::new(code, operation, message)
        .with_fact("iceberg_error_kind", source.kind().into_static())
        .with_fact("iceberg_retryable", source.retryable());
    if let Some(marker) = aws_marker(&chain_text(&source)) {
        error = error.with_fact("aws_error_detail", marker.detail);
    }
    if let Some(opendal) = find_source::<opendal::Error>(&source) {
        error = error.with_fact("opendal_error_kind", opendal.kind().into_static());
    }
    error.with_source(source)
}

/// The adapter code of an Iceberg error. The typed kind decides first;
/// credential and access markers in the rendered chain decide next, because
/// the Glue catalog flattens AWS SDK errors into text and the REST catalog
/// renders the HTTP status into the error context; a typed OpenDAL cause
/// decides last.
pub(crate) fn iceberg_error_code(source: &IcebergError) -> ErrorCode {
    if matches!(source.kind(), IcebergErrorKind::TableNotFound | IcebergErrorKind::NamespaceNotFound) {
        return ErrorCode::ObjectNotFound;
    }
    if let Some(marker) = aws_marker(&chain_text(source)) {
        return marker.code;
    }
    if let Some(opendal) = find_source::<opendal::Error>(source) {
        return opendal_error_code(opendal);
    }
    if source.retryable() { ErrorCode::SourceTemporaryFailure } else { ErrorCode::SourceDependencyFailure }
}

fn opendal_error_code(source: &opendal::Error) -> ErrorCode {
    match source.kind() {
        opendal::ErrorKind::NotFound => ErrorCode::SourceObjectNotFound,
        opendal::ErrorKind::PermissionDenied => ErrorCode::SourceAccessDenied,
        opendal::ErrorKind::RateLimited => ErrorCode::SourceTemporaryFailure,
        _ if source.is_temporary() => ErrorCode::SourceTemporaryFailure,
        _ => ErrorCode::SourceDependencyFailure,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AwsMarker {
    code: ErrorCode,
    detail: &'static str,
}

/// Credential and access markers rendered by the AWS SDK, S3 and HTTP
/// clients, in the order the first match wins.
const AWS_MARKERS: &[(&[&str], ErrorCode, &str)] = &[
    (&["expiredtokenexception", "expiredtoken"], ErrorCode::SourceUnauthenticated, "expired_token"),
    (&["invalidaccesskeyid"], ErrorCode::SourceUnauthenticated, "invalid_access_key"),
    (&["unrecognizedclientexception"], ErrorCode::SourceUnauthenticated, "invalid_client"),
    (&["signaturedoesnotmatch"], ErrorCode::SourceAccessDenied, "signature_mismatch"),
    (&["accessdeniedexception", "accessdenied"], ErrorCode::SourceAccessDenied, "access_denied"),
    (&["entitynotfoundexception"], ErrorCode::ObjectNotFound, "not_found"),
    (
        &["401 unauthorized", "status code: 401", "status: 401", "http 401"],
        ErrorCode::SourceUnauthenticated,
        "http_401",
    ),
    (&["403 forbidden", "status code: 403", "status: 403", "http 403"], ErrorCode::SourceAccessDenied, "http_403"),
    (
        &["no credentials", "missing credentials", "could not load credentials"],
        ErrorCode::SourceUnauthenticated,
        "missing_credentials",
    ),
];

fn aws_marker(text: &str) -> Option<AwsMarker> {
    let lower = text.to_ascii_lowercase();
    AWS_MARKERS
        .iter()
        .find(|(needles, _, _)| needles.iter().any(|needle| lower.contains(needle)))
        .map(|&(_, code, detail)| AwsMarker { code, detail })
}

fn chain_text(source: &(dyn StdError + 'static)) -> String {
    let mut text = source.to_string();
    let mut current = source.source();
    for _ in 0..MAX_ERROR_CHAIN_DEPTH {
        let Some(layer) = current else { break };
        text.push_str(": ");
        text.push_str(&layer.to_string());
        current = layer.source();
    }
    text
}

fn find_source<'a, E: StdError + 'static>(source: &'a (dyn StdError + 'static)) -> Option<&'a E> {
    let mut current = Some(source);
    for _ in 0..MAX_ERROR_CHAIN_DEPTH {
        let layer = current?;
        if let Some(found) = layer.downcast_ref::<E>() {
            return Some(found);
        }
        current = layer.source();
    }
    None
}

#[cfg(test)]
mod tests;
