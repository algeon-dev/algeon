use crate::{
    error::{QueryError, QueryErrorCode, Result},
    source::{ParquetSourceLocation, ResolvedParquetSource},
};

pub(super) struct SourceFetchLocation {
    pub(super) bucket: String,
    pub(super) region: String,
    pub(super) key: ::object_store::path::Path,
    pub(super) storage_profile_id: Option<String>,
}

pub(super) fn source_fetch_location(
    location: &ParquetSourceLocation,
    resolved_source: &ResolvedParquetSource,
) -> Result<SourceFetchLocation> {
    let ParquetSourceLocation::ObjectStoreUri { scheme, uri } = location else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.location",
            "SourceFetch requires an object-store URI source location",
        )
        .with_fact("read_capability", location.read_capability()));
    };
    if scheme != "s3" {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.source_fetch.location",
            "object-store source fetch currently supports only S3 URIs",
        )
        .with_fact("scheme", scheme));
    }
    parse_s3_source_uri(uri, resolved_source.storage_profile_id.clone())
}

pub(super) fn parse_s3_source_uri(uri: &str, storage_profile_id: Option<String>) -> Result<SourceFetchLocation> {
    let rest = uri.strip_prefix("s3://").ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.s3_uri",
            "S3 source URI must start with s3://",
        )
        .with_fact("reason", "missing_s3_scheme")
    })?;
    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.s3_uri",
            "S3 source URI must include a bucket and key",
        )
        .with_fact("reason", "missing_key")
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.s3_uri",
            "S3 source URI bucket and key must be non-empty",
        )
        .with_fact("reason", "empty_bucket_or_key"));
    }
    let region = std::env::var("AWS_REGION")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "us-east-1".to_string());
    Ok(SourceFetchLocation {
        bucket: bucket.to_string(),
        region,
        key: ::object_store::path::Path::from(key),
        storage_profile_id,
    })
}

pub(super) fn object_store_bucket_fact(bucket: &str) -> String {
    format!("object_store_bucket:fnv64:{}", stable_log_hash(bucket))
}

pub(super) fn object_store_key_fact(location: &::object_store::path::Path) -> String {
    crate::source::redacted_object_store_path_diagnostic(location.as_ref())
}

fn stable_log_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}
