use std::sync::Arc;

use cudf::io::{
    parquet::{
        RemoteCredentialMode, RemoteParquetCapability, RemoteParquetCapabilityDisposition, RemoteParquetProfile,
        RemoteReadMode, remote_parquet_capability,
    },
    source::LocalParquetFile as CudfLocalParquetFile,
};
use object_store::aws::AwsCredential;
use query_planning::source::{CredentialMode, ParquetReadStrategy, ResolvedParquetInput, ResolvedParquetSource};
use query_runtime::runtime::{S3ObjectStoreKey, S3SourceStorageCredential, SourceStorageProfile};

use crate::exec::source::{cudf_query_error_without_source, *};

pub(in crate::exec::source) fn attach_cudf_remote_parquet_contract(
    context: &ExecutionContext,
    cudf_file: CudfLocalParquetFile,
    resolved_source: &ResolvedParquetSource,
    operation: &'static str,
) -> Result<CudfLocalParquetFile> {
    if cudf_file.remote_source().is_none() {
        return Ok(cudf_file);
    }
    let profile = cudf_remote_parquet_profile(context, resolved_source)?;
    let read_mode = RemoteReadMode::RequireKvikio;
    let cudf_file = cudf_file.with_remote_parquet_contract(profile, read_mode).map_err(|error| {
        cudf_query_error_without_source(
            operation,
            "failed to attach typed remote Parquet contract to cudf source file",
            &error,
        )
        .with_fact("resource", "remote_parquet_source")
        .with_fact("uri_scheme", resolved_remote_uri_scheme(resolved_source))
        .with_fact("read_strategy", resolved_source.read_strategy().as_str())
        .with_fact("remote_read_mode", read_mode.as_str())
        .with_fact("credential_mode", resolved_source.credential_mode().as_str())
        .with_fact("fallback_allowed", false)
    })?;
    let remote_source = cudf_file.remote_source().expect("remote contract attachment preserves an object-store source");
    let capability = remote_parquet_capability(remote_source);
    validate_remote_kvikio_capability(resolved_source, &capability, operation)?;
    Ok(cudf_file)
}

pub(in crate::exec::source) fn validate_remote_kvikio_capability(
    resolved_source: &ResolvedParquetSource,
    capability: &RemoteParquetCapability,
    operation: &'static str,
) -> Result<()> {
    if capability.disposition != RemoteParquetCapabilityDisposition::KvikioRemoteDatasource {
        return Err(QueryError::new(
            QueryErrorCode::CudfUnsupported,
            operation,
            "explicit KvikIO remote Parquet datasource is unavailable",
        )
        .with_fact("resource", "remote_parquet_source")
        .with_fact("reason", "remote_kvikio_backend_unavailable")
        .with_fact("uri_scheme", resolved_remote_uri_scheme(resolved_source))
        .with_fact("read_strategy", resolved_source.read_strategy().as_str())
        .with_fact("remote_read_mode", RemoteReadMode::RequireKvikio.as_str())
        .with_fact("credential_mode", resolved_source.credential_mode().as_str())
        .with_fact("remote_capability_disposition", remote_parquet_capability_disposition_label(capability.disposition))
        .with_fact("remote_backend_build_status", capability.backend_build_status.as_str())
        .with_fact("fallback_allowed", false));
    }
    Ok(())
}

pub(in crate::exec::source) fn resolved_remote_uri_scheme(resolved_source: &ResolvedParquetSource) -> &'static str {
    let ResolvedParquetInput::RemoteKvikio { uri, .. } = &resolved_source.input else {
        return "unknown";
    };
    if uri.starts_with("s3://") { "s3" } else { "unknown" }
}

pub(in crate::exec::source) fn resolved_remote_kvikio_access<'a>(
    resolved_source: &'a ResolvedParquetSource,
    operation: &'static str,
) -> Result<(&'a str, CredentialMode, Option<&'a str>)> {
    let ResolvedParquetInput::RemoteKvikio { uri, credential_mode, storage_profile_id } = &resolved_source.input else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "remote Parquet contract requires a tagged remote KvikIO input",
        )
        .with_fact("resource", "remote_parquet_source")
        .with_fact("reason", "remote_kvikio_input_required")
        .with_fact("read_strategy", resolved_source.read_strategy().as_str())
        .with_fact("fallback_allowed", false));
    };
    Ok((uri, *credential_mode, storage_profile_id.as_deref()))
}

pub(in crate::exec::source) fn remote_parquet_capability_disposition_label(
    disposition: RemoteParquetCapabilityDisposition,
) -> &'static str {
    match disposition {
        RemoteParquetCapabilityDisposition::PathReaderUnverified => "path_reader_unverified",
        RemoteParquetCapabilityDisposition::KvikioRemoteDatasource => "kvikio_remote_datasource",
        RemoteParquetCapabilityDisposition::BackendBuildDisabled => "backend_build_disabled",
        RemoteParquetCapabilityDisposition::CredentialModeUnsupported => "credential_mode_unsupported",
        RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable => "direct_evidence_unavailable",
        RemoteParquetCapabilityDisposition::KvikioEvidenceUnavailable => "kvikio_evidence_unavailable",
    }
}

pub(in crate::exec::source) fn cudf_remote_parquet_profile(
    context: &ExecutionContext,
    resolved_source: &ResolvedParquetSource,
) -> Result<RemoteParquetProfile> {
    let (_, credential_mode, profile_id) =
        resolved_remote_kvikio_access(resolved_source, "query_engine.remote_parquet_profile.resolve")?;
    let Some(profile_id) = profile_id else {
        return Ok(cudf_remote_parquet_profile_from_storage(None, None, None));
    };
    let storage_profile = context.source_storage_profile(profile_id).map_err(|error| {
        error
            .with_fact_if_absent("resource", "remote_parquet_source")
            .with_fact_if_absent("reason", "remote_storage_profile_unavailable")
            .with_fact_if_absent("uri_scheme", resolved_remote_uri_scheme(resolved_source))
            .with_fact_if_absent("read_strategy", resolved_source.read_strategy().as_str())
            .with_fact_if_absent("remote_read_mode", RemoteReadMode::RequireKvikio.as_str())
            .with_fact_if_absent("credential_mode", credential_mode.as_str())
            .with_fact_if_absent("fallback_allowed", false)
    })?;
    let default_chain_credential =
        resolve_remote_direct_default_chain_credential(context, resolved_source, storage_profile)?;
    Ok(cudf_remote_parquet_profile_from_storage(
        Some(profile_id),
        Some(storage_profile),
        default_chain_credential.as_deref(),
    ))
}

/// Resolves default-chain AWS credentials for remote-direct reads.
///
/// KvikIO requires static credentials; the object-store provider owns the
/// shared credential identity and its cache/refresh behavior.
pub(in crate::exec::source) fn resolve_remote_direct_default_chain_credential(
    context: &ExecutionContext,
    resolved_source: &ResolvedParquetSource,
    storage_profile: &SourceStorageProfile,
) -> Result<Option<Arc<AwsCredential>>> {
    const OPERATION: &str = "query_engine.remote_parquet_profile.default_chain_credentials";
    let (uri, credential_mode, storage_profile_id) = resolved_remote_kvikio_access(resolved_source, OPERATION)?;
    if resolved_source.read_strategy() != ParquetReadStrategy::RemoteDirect
        || !matches!(storage_profile.s3_profile().credential, S3SourceStorageCredential::DefaultChain)
    {
        return Ok(None);
    }
    let attach_resolution_facts = |error: QueryError| {
        error
            .with_fact_if_absent("resource", "remote_parquet_source")
            .with_fact_if_absent("uri_scheme", resolved_remote_uri_scheme(resolved_source))
            .with_fact_if_absent("read_strategy", resolved_source.read_strategy().as_str())
            .with_fact_if_absent("remote_read_mode", RemoteReadMode::RequireKvikio.as_str())
            .with_fact_if_absent("credential_mode", credential_mode.as_str())
            .with_fact_if_absent("fallback_allowed", false)
    };
    let location = parse_s3_source_uri(uri, storage_profile_id.map(str::to_owned)).map_err(attach_resolution_facts)?;
    context
        .resolve_s3_default_chain_credential(
            S3ObjectStoreKey {
                bucket: location.bucket,
                region: location.region,
                storage_profile_id: location.storage_profile_id,
            },
            OPERATION,
        )
        .map(Some)
        .map_err(attach_resolution_facts)
}

pub(in crate::exec::source) fn cudf_remote_parquet_profile_from_storage(
    profile_id: Option<&str>,
    storage_profile: Option<&SourceStorageProfile>,
    default_chain_credential: Option<&AwsCredential>,
) -> RemoteParquetProfile {
    let mut profile = RemoteParquetProfile::shared_process_env();
    if let Some(profile_id) = profile_id {
        profile = profile.with_profile_id(profile_id);
    }
    let Some(storage_profile) = storage_profile else {
        return profile;
    };
    if storage_profile.requires_scoped_remote_direct_profile() {
        profile = profile.with_credential_mode(RemoteCredentialMode::ScopedProfile);
    }
    let s3_profile = storage_profile.s3_profile();
    if let Some(region) = s3_profile.region.as_deref().filter(|region| !region.is_empty()) {
        profile = profile.with_region(region);
    }
    if let Some(endpoint) = s3_profile.endpoint.as_deref().filter(|endpoint| !endpoint.is_empty()) {
        profile = profile.with_endpoint(endpoint);
    }
    if let Some(virtual_hosted_style_request) = s3_profile.virtual_hosted_style_request {
        profile = profile.with_path_style_access(!virtual_hosted_style_request);
    }
    match &s3_profile.credential {
        S3SourceStorageCredential::DefaultChain => {
            if let Some(credential) = default_chain_credential {
                profile = profile.with_static_credentials(
                    credential.key_id.clone(),
                    credential.secret_key.clone(),
                    credential.token.clone(),
                );
            }
        },
        S3SourceStorageCredential::Unsigned => {
            profile = profile.with_unsigned_requests();
        },
        S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            profile = profile.with_static_credentials(
                access_key_id.clone(),
                secret_access_key.clone(),
                session_token.clone(),
            );
        },
    }
    profile
}
