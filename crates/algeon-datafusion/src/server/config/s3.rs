use super::{env::EnvValues, file::ServerS3FileConfig, *};
use crate::server::native_exec::S3SourceStorageProfile;

/// S3-compatible endpoint, such as `http://localhost:9000`, for every
/// `s3://` source.
pub const ALGEON_SERVER_S3_ENDPOINT_ENV: &str = "ALGEON_SERVER_S3_ENDPOINT";
/// Region for every `s3://` source.
pub const ALGEON_SERVER_S3_REGION_ENV: &str = "ALGEON_SERVER_S3_REGION";
/// Path-style addressing (`true`) instead of virtual-hosted buckets.
pub const ALGEON_SERVER_S3_PATH_STYLE_ENV: &str = "ALGEON_SERVER_S3_PATH_STYLE";
/// Unsigned requests (`true`) instead of the default AWS credential chain.
pub const ALGEON_SERVER_S3_UNSIGNED_ENV: &str = "ALGEON_SERVER_S3_UNSIGNED";

/// The S3 storage `[s3]` and the `ALGEON_SERVER_S3_*` variables describe, or
/// `None` when neither is present. Credentials come from the default AWS
/// chain unless `unsigned` is set; each variable overrides its TOML field.
pub(super) fn s3_storage(
    values: &mut EnvValues<'_>,
    file: Option<&ServerS3FileConfig>,
) -> Result<Option<S3SourceStorageProfile>> {
    let endpoint = values.string(ALGEON_SERVER_S3_ENDPOINT_ENV);
    let region = values.string(ALGEON_SERVER_S3_REGION_ENV);
    let path_style = values.optional_bool(ALGEON_SERVER_S3_PATH_STYLE_ENV)?;
    let unsigned = values.optional_bool(ALGEON_SERVER_S3_UNSIGNED_ENV)?;
    if file.is_none() && endpoint.is_none() && region.is_none() && path_style.is_none() && unsigned.is_none() {
        return Ok(None);
    }
    let file = file.cloned().unwrap_or_default();
    let mut storage = S3SourceStorageProfile::default_chain();
    if let Some(endpoint) = endpoint.or(file.endpoint) {
        storage = storage.with_endpoint(endpoint);
    }
    if let Some(region) = region.or(file.region) {
        storage = storage.with_region(region);
    }
    if let Some(path_style) = path_style.or(file.path_style) {
        storage = storage.with_path_style_access(path_style);
    }
    if unsigned.or(file.unsigned).unwrap_or(false) {
        storage = storage.with_unsigned_requests();
    }
    Ok(Some(storage))
}
