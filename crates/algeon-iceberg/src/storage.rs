//! Iceberg FileIO properties derived from the server's `[s3]` storage.
//!
//! Iceberg metadata (metadata.json, manifest lists, manifests) is read
//! through the catalog's FileIO, and data files through the DataFusion object
//! store and the native reader, which both use the `[s3]` profile. The FileIO
//! is configured from the same profile, and a loaded table whose FileIO
//! names another endpoint, region or addressing style is rejected, so the
//! two read paths address the same storage.

use std::collections::HashMap;

use algeon_datafusion::native_exec::S3SourceStorageProfile;
use iceberg::io::{
    CLIENT_REGION, S3_ACCESS_KEY_ID, S3_ALLOW_ANONYMOUS, S3_ENDPOINT, S3_PATH_STYLE_ACCESS, S3_REGION,
    S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
};
use query_runtime::runtime::S3SourceStorageCredential;

/// FileIO properties that make Iceberg read metadata with `storage`.
pub(crate) fn file_io_properties(storage: &S3SourceStorageProfile) -> HashMap<String, String> {
    let mut properties = HashMap::new();
    if let Some(endpoint) = &storage.endpoint {
        properties.insert(S3_ENDPOINT.to_string(), endpoint.clone());
    }
    if let Some(region) = &storage.region {
        properties.insert(S3_REGION.to_string(), region.clone());
    }
    if let Some(virtual_hosted) = storage.virtual_hosted_style_request {
        properties.insert(S3_PATH_STYLE_ACCESS.to_string(), (!virtual_hosted).to_string());
    }
    match &storage.credential {
        S3SourceStorageCredential::DefaultChain => {},
        S3SourceStorageCredential::Unsigned => {
            properties.insert(S3_ALLOW_ANONYMOUS.to_string(), "true".to_string());
        },
        S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            properties.insert(S3_ACCESS_KEY_ID.to_string(), access_key_id.clone());
            properties.insert(S3_SECRET_ACCESS_KEY.to_string(), secret_access_key.clone());
            if let Some(session_token) = session_token {
                properties.insert(S3_SESSION_TOKEN.to_string(), session_token.clone());
            }
        },
    }
    properties
}

/// The first FileIO property of a loaded table that addresses storage other
/// than `storage`, or `None` when they agree.
///
/// The endpoint must match exactly, since a different endpoint is a
/// different store. Region and addressing style are compared when both
/// sides set them; an unset side takes the client default.
pub(crate) fn mismatched_property(
    properties: &HashMap<String, String>,
    storage: &S3SourceStorageProfile,
) -> Option<&'static str> {
    let normalized = |endpoint: &str| endpoint.trim().trim_end_matches('/').to_string();
    let file_io_endpoint = properties.get(S3_ENDPOINT).map(|endpoint| normalized(endpoint));
    if file_io_endpoint != storage.endpoint.as_deref().map(normalized) {
        return Some(S3_ENDPOINT);
    }
    let region_key = if properties.contains_key(CLIENT_REGION) { CLIENT_REGION } else { S3_REGION };
    if let (Some(file_io_region), Some(region)) = (properties.get(region_key), &storage.region)
        && file_io_region.trim() != region.trim()
    {
        return Some(region_key);
    }
    let file_io_path_style =
        properties.get(S3_PATH_STYLE_ACCESS).map(|value| value.trim().eq_ignore_ascii_case("true"));
    if let (Some(file_io_path_style), Some(virtual_hosted)) = (file_io_path_style, storage.virtual_hosted_style_request)
        && file_io_path_style == virtual_hosted
    {
        return Some(S3_PATH_STYLE_ACCESS);
    }
    None
}

#[cfg(test)]
mod tests;
