use object_store::path::Path as ObjectStorePath;
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    source::{ParquetSourceLocation, ResolvedParquetSource, redacted_object_store_path_diagnostic},
};

use super::cache::stable_log_hash;

/// An S3 object named by a source URI. The region comes from the process environment and not
/// from the URI.
pub(in crate::exec::source) struct CacheMaintenanceLocation {
    pub(in crate::exec::source) bucket: String,
    pub(in crate::exec::source) region: String,
    pub(in crate::exec::source) key: ObjectStorePath,
    pub(in crate::exec::source) storage_profile_id: Option<String>,
}

/// Resolves the object a cache maintenance fetch reads. Only `s3://` object-store locations are
/// supported; a local path fails with `InvalidPlan` and another scheme with `UnsupportedOperator`.
pub(super) fn cache_maintenance_location(
    location: &ParquetSourceLocation,
    resolved_source: &ResolvedParquetSource,
) -> Result<CacheMaintenanceLocation> {
    let ParquetSourceLocation::ObjectStoreUri { scheme, uri } = location else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.lakehouse_cache.maintenance.location",
            "lakehouse cache maintenance requires an object-store URI source location",
        )
        .with_fact("read_capability", location.read_capability()));
    };
    if scheme != "s3" {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.lakehouse_cache.maintenance.location",
            "object-store lakehouse cache maintenance currently supports only S3 URIs",
        )
        .with_fact("scheme", scheme));
    }
    parse_s3_source_uri(uri, resolved_source.storage_profile_id().map(str::to_owned))
}

/// Splits `s3://bucket/key` into bucket and key. The region is `AWS_REGION`, else
/// `AWS_DEFAULT_REGION`, else `us-east-1`; an object in another region needs one of the variables
/// set. The result also keys default-chain credential resolution for remote-direct reads. Error
/// facts name the failed rule and do not echo the URI.
pub(in crate::exec::source) fn parse_s3_source_uri(
    uri: &str,
    storage_profile_id: Option<String>,
) -> Result<CacheMaintenanceLocation> {
    let rest = uri.strip_prefix("s3://").ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.lakehouse_cache.maintenance.s3_uri",
            "S3 source URI must start with s3://",
        )
        .with_fact("reason", "missing_s3_scheme")
    })?;
    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.lakehouse_cache.maintenance.s3_uri",
            "S3 source URI must include a bucket and key",
        )
        .with_fact("reason", "missing_key")
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.lakehouse_cache.maintenance.s3_uri",
            "S3 source URI bucket and key must be non-empty",
        )
        .with_fact("reason", "empty_bucket_or_key"));
    }
    let region = std::env::var("AWS_REGION")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "us-east-1".to_string());
    Ok(CacheMaintenanceLocation {
        bucket: bucket.to_string(),
        region,
        key: ObjectStorePath::from(key),
        storage_profile_id,
    })
}

/// Stable hash of the bucket name for error facts, so the name itself is not logged.
pub(super) fn object_store_bucket_fact(bucket: &str) -> String {
    format!("object_store_bucket:fnv64:{}", stable_log_hash(bucket))
}

pub(super) fn object_store_key_fact(location: &ObjectStorePath) -> String {
    redacted_object_store_path_diagnostic(location.as_ref())
}
