use algeon_datafusion::error::ErrorCode;
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind};

use super::iceberg_error_code;

#[test]
fn missing_tables_and_namespaces_are_object_not_found() {
    for kind in [IcebergErrorKind::TableNotFound, IcebergErrorKind::NamespaceNotFound] {
        assert_eq!(iceberg_error_code(&IcebergError::new(kind, "missing")), ErrorCode::ObjectNotFound);
    }
}

#[test]
fn rest_status_and_aws_markers_classify_credentials_and_access() {
    let rest = |status: &str| {
        IcebergError::new(IcebergErrorKind::Unexpected, "Received response with unexpected status code")
            .with_context("status", status)
    };
    assert_eq!(iceberg_error_code(&rest("401 Unauthorized")), ErrorCode::SourceUnauthenticated);
    assert_eq!(iceberg_error_code(&rest("403 Forbidden")), ErrorCode::SourceAccessDenied);
    let glue = IcebergError::new(IcebergErrorKind::Unexpected, "aws sdk error: ServiceError { ExpiredTokenException }");
    assert_eq!(iceberg_error_code(&glue), ErrorCode::SourceUnauthenticated);
}

#[test]
fn typed_storage_causes_and_retryability_decide_the_rest() {
    let missing_object = IcebergError::new(IcebergErrorKind::Unexpected, "read failed")
        .with_source(opendal::Error::new(opendal::ErrorKind::NotFound, "no such key"));
    assert_eq!(iceberg_error_code(&missing_object), ErrorCode::SourceObjectNotFound);
    let throttled = IcebergError::new(IcebergErrorKind::Unexpected, "read failed")
        .with_source(opendal::Error::new(opendal::ErrorKind::RateLimited, "slow down"));
    assert_eq!(iceberg_error_code(&throttled), ErrorCode::SourceTemporaryFailure);
    let retryable = IcebergError::new(IcebergErrorKind::Unexpected, "connection reset").with_retryable(true);
    assert_eq!(iceberg_error_code(&retryable), ErrorCode::SourceTemporaryFailure);
    let invalid = IcebergError::new(IcebergErrorKind::DataInvalid, "bad manifest");
    assert_eq!(iceberg_error_code(&invalid), ErrorCode::SourceDependencyFailure);
}
