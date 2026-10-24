use super::{ConfigSources, Result, S3SourceStorageProfile, file::ServerS3FileConfig};

/// The storage `[s3]` describes, or `None` without the table. Credentials
/// come from the default AWS chain unless `unsigned` is set.
///
/// The native Parquet reader addresses a custom endpoint path-style only
/// and rejects virtual-hosted addressing there at query time, so loading
/// rejects that combination first.
pub(super) fn s3_storage(
    file: Option<&ServerS3FileConfig>,
    sources: &ConfigSources,
) -> Result<Option<S3SourceStorageProfile>> {
    let Some(file) = file else {
        return Ok(None);
    };
    if file.endpoint.is_some() && file.path_style == Some(false) {
        return Err(sources
            .invalid(
                "s3.path_style",
                "is `false`, and virtual-hosted addressing is unsupported with a custom `s3.endpoint`; set `path_style = true` or remove `endpoint`",
            )
            .with_fact("reason", "s3_custom_endpoint_requires_path_style"));
    }
    let mut storage = S3SourceStorageProfile::default_chain();
    if let Some(endpoint) = &file.endpoint {
        storage = storage.with_endpoint(endpoint);
    }
    if let Some(region) = &file.region {
        storage = storage.with_region(region);
    }
    if let Some(path_style) = file.path_style {
        storage = storage.with_path_style_access(path_style);
    }
    if file.unsigned == Some(true) {
        storage = storage.with_unsigned_requests();
    }
    Ok(Some(storage))
}
