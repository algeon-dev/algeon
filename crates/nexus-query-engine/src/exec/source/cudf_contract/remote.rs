use super::super::*;

/// Explicit datasource-level probe facts for a resolved remote Parquet source.
///
/// These facts come from opening libcudf's path datasource. They are not
/// per-read Parquet evidence and must not be used to satisfy
/// the fail-closed KvikIO read contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRemoteParquetDatasourceProbe {
    /// Whether the opened datasource reports device-read support.
    pub supports_device_read: bool,
    /// Whether the opened datasource prefers device reads for the requested size.
    pub device_read_preferred: bool,
    /// Datasource size reported by libcudf, when available.
    pub size_bytes: Option<u64>,
    /// cuDF/KvikIO remote backend build status.
    pub backend_build_status: String,
    /// Conservative typed-source capability disposition at the cudf-rs boundary.
    pub source_capability_disposition: String,
    /// URI scheme only, never the full URI.
    pub uri_scheme: String,
    /// Selected Nexus read strategy.
    pub read_strategy: String,
    /// Requested remote read mode.
    pub remote_read_mode: String,
    /// Credential mode selected by the resolved source.
    pub credential_mode: String,
    /// Whether this probe supplies per-read backend/byte evidence.
    pub per_read_evidence_available: bool,
}

/// Probe libcudf datasource capability for an explicitly resolved remote
/// path-based Parquet source.
///
/// This API may contact object-store metadata services while opening the
/// datasource. It is intentionally not used by default source planning, and it
/// rejects `RequireDirect` because datasource capability is weaker than the
/// per-read backend/range evidence required by that mode.
pub fn probe_resolved_remote_parquet_datasource(
    context: &ExecutionContext,
    resolved_source: &crate::source::ResolvedParquetSource,
    preferred_read_size_bytes: Option<u64>,
) -> Result<ResolvedRemoteParquetDatasourceProbe> {
    const OPERATION: &str = "nexus_query_engine.remote_parquet_datasource.probe";
    validate_remote_parquet_datasource_probe_request(resolved_source, OPERATION)?;
    let remote_source = cudf_remote_parquet_source_for_resolved_source(context, resolved_source, OPERATION)?;
    let source_capability = cudf::io::parquet::remote_parquet_capability(&remote_source);
    let _io_lease = context.acquire_io_lease(OPERATION)?;
    let probe = cudf::io::parquet::probe_remote_parquet_datasource(&remote_source, preferred_read_size_bytes).map_err(
        |error| {
            QueryError::from_cudf_without_source(
                OPERATION,
                "failed to probe resolved remote Parquet datasource capability",
                &error,
            )
            .with_fact("resource", "remote_parquet_datasource")
            .with_fact("read_strategy", resolved_source.read_strategy().as_str())
            .with_fact("remote_read_mode", "require_kvikio")
            .with_fact("credential_mode", resolved_source.credential_mode().as_str())
            .with_fact("uri_scheme", source_capability.uri_scheme.as_str())
        },
    )?;
    Ok(ResolvedRemoteParquetDatasourceProbe {
        supports_device_read: probe.supports_device_read,
        device_read_preferred: probe.device_read_preferred,
        size_bytes: probe.size_bytes,
        backend_build_status: probe.backend_build_status.as_str().to_string(),
        source_capability_disposition: remote_parquet_capability_disposition_label(source_capability.disposition)
            .to_string(),
        uri_scheme: source_capability.uri_scheme,
        read_strategy: resolved_source.read_strategy().as_str().to_string(),
        remote_read_mode: "require_kvikio".to_string(),
        credential_mode: resolved_source.credential_mode().as_str().to_string(),
        per_read_evidence_available: false,
    })
}

pub(in crate::exec::source) fn attach_cudf_remote_parquet_contract(
    context: &ExecutionContext,
    cudf_file: cudf::io::source::LocalParquetFile,
    resolved_source: &crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<cudf::io::source::LocalParquetFile> {
    if cudf_file.remote_source().is_none() {
        return Ok(cudf_file);
    }
    let profile = cudf_remote_parquet_profile(context, resolved_source)?;
    let read_mode = cudf::io::parquet::RemoteReadMode::RequireKvikio;
    let cudf_file = cudf_file.with_remote_parquet_contract(profile, read_mode).map_err(|error| {
        QueryError::from_cudf_without_source(
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
    let capability = cudf::io::parquet::remote_parquet_capability(remote_source);
    validate_remote_kvikio_capability(resolved_source, &capability, operation)?;
    Ok(cudf_file)
}

pub(in crate::exec::source) fn validate_remote_kvikio_capability(
    resolved_source: &crate::source::ResolvedParquetSource,
    capability: &cudf::io::parquet::RemoteParquetCapability,
    operation: &'static str,
) -> Result<()> {
    if capability.disposition != cudf::io::parquet::RemoteParquetCapabilityDisposition::KvikioRemoteDatasource {
        return Err(QueryError::new(
            QueryErrorCode::CudfUnsupported,
            operation,
            "explicit KvikIO remote Parquet datasource is unavailable",
        )
        .with_fact("resource", "remote_parquet_source")
        .with_fact("reason", "remote_kvikio_backend_unavailable")
        .with_fact("uri_scheme", resolved_remote_uri_scheme(resolved_source))
        .with_fact("read_strategy", resolved_source.read_strategy().as_str())
        .with_fact("remote_read_mode", cudf::io::parquet::RemoteReadMode::RequireKvikio.as_str())
        .with_fact("credential_mode", resolved_source.credential_mode().as_str())
        .with_fact("remote_capability_disposition", remote_parquet_capability_disposition_label(capability.disposition))
        .with_fact("remote_backend_build_status", capability.backend_build_status.as_str())
        .with_fact("fallback_allowed", false));
    }
    Ok(())
}

pub(in crate::exec::source) fn validate_remote_parquet_datasource_probe_request(
    resolved_source: &crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<()> {
    if !resolved_source.is_remote_kvikio() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "remote Parquet datasource probes require a resolved remote path-based source",
        )
        .with_fact("resource", "remote_parquet_datasource")
        .with_fact("reason", "remote_datasource_probe_requires_path_remote_strategy")
        .with_fact("read_strategy", resolved_source.read_strategy().as_str())
        .with_fact("remote_read_mode", "require_kvikio")
        .with_fact("credential_mode", resolved_source.credential_mode().as_str())
        .with_fact("remote_backend_build_status", cudf::io::parquet::remote_parquet_backend_build_status().as_str()));
    }
    Ok(())
}

pub(in crate::exec::source) fn cudf_remote_parquet_source_for_resolved_source(
    context: &ExecutionContext,
    resolved_source: &crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<cudf::io::parquet::RemoteParquetSource> {
    let (uri, _, _) = resolved_remote_kvikio_access(resolved_source, operation)?;
    let profile = cudf_remote_parquet_profile(context, resolved_source)?;
    cudf::io::parquet::RemoteParquetSource::s3(uri)
        .map(|source| source.with_profile(profile).with_read_mode(cudf::io::parquet::RemoteReadMode::RequireKvikio))
        .map_err(|error| {
            QueryError::from_cudf_without_source(
                operation,
                "failed to build typed remote Parquet datasource probe source",
                &error,
            )
            .with_fact("resource", "remote_parquet_datasource")
            .with_fact("uri_scheme", resolved_remote_uri_scheme(resolved_source))
            .with_fact("read_strategy", resolved_source.read_strategy().as_str())
            .with_fact("remote_read_mode", "require_kvikio")
            .with_fact("credential_mode", resolved_source.credential_mode().as_str())
        })
}

pub(in crate::exec::source) fn resolved_remote_uri_scheme(
    resolved_source: &crate::source::ResolvedParquetSource,
) -> &'static str {
    let crate::source::ResolvedParquetInput::RemoteKvikio { uri, .. } = &resolved_source.input else {
        return "unknown";
    };
    if uri.starts_with("s3://") { "s3" } else { "unknown" }
}

pub(in crate::exec::source) fn resolved_remote_kvikio_access<'a>(
    resolved_source: &'a crate::source::ResolvedParquetSource,
    operation: &'static str,
) -> Result<(&'a str, crate::source::CredentialMode, Option<&'a str>)> {
    let crate::source::ResolvedParquetInput::RemoteKvikio { uri, credential_mode, storage_profile_id } =
        &resolved_source.input
    else {
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
    disposition: cudf::io::parquet::RemoteParquetCapabilityDisposition,
) -> &'static str {
    match disposition {
        cudf::io::parquet::RemoteParquetCapabilityDisposition::PathReaderUnverified => "path_reader_unverified",
        cudf::io::parquet::RemoteParquetCapabilityDisposition::KvikioRemoteDatasource => "kvikio_remote_datasource",
        cudf::io::parquet::RemoteParquetCapabilityDisposition::BackendBuildDisabled => "backend_build_disabled",
        cudf::io::parquet::RemoteParquetCapabilityDisposition::CredentialModeUnsupported => {
            "credential_mode_unsupported"
        },
        cudf::io::parquet::RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable => {
            "direct_evidence_unavailable"
        },
        cudf::io::parquet::RemoteParquetCapabilityDisposition::KvikioEvidenceUnavailable => {
            "kvikio_evidence_unavailable"
        },
    }
}

pub(in crate::exec::source) fn cudf_remote_parquet_profile(
    context: &ExecutionContext,
    resolved_source: &crate::source::ResolvedParquetSource,
) -> Result<cudf::io::parquet::RemoteParquetProfile> {
    let (_, credential_mode, profile_id) =
        resolved_remote_kvikio_access(resolved_source, "nexus_query_engine.remote_parquet_profile.resolve")?;
    let Some(profile_id) = profile_id else {
        return Ok(cudf_remote_parquet_profile_from_storage(None, None, None));
    };
    let storage_profile = context.source_storage_profile(profile_id).map_err(|error| {
        error
            .with_fact_if_absent("resource", "remote_parquet_source")
            .with_fact_if_absent("reason", "remote_storage_profile_unavailable")
            .with_fact_if_absent("uri_scheme", resolved_remote_uri_scheme(resolved_source))
            .with_fact_if_absent("read_strategy", resolved_source.read_strategy().as_str())
            .with_fact_if_absent("remote_read_mode", cudf::io::parquet::RemoteReadMode::RequireKvikio.as_str())
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
    resolved_source: &crate::source::ResolvedParquetSource,
    storage_profile: &crate::runtime::SourceStorageProfile,
) -> Result<Option<std::sync::Arc<::object_store::aws::AwsCredential>>> {
    const OPERATION: &str = "nexus_query_engine.remote_parquet_profile.default_chain_credentials";
    let (uri, credential_mode, storage_profile_id) = resolved_remote_kvikio_access(resolved_source, OPERATION)?;
    if resolved_source.read_strategy() != crate::source::ParquetReadStrategy::RemoteDirect
        || !matches!(storage_profile.s3_profile().credential, crate::runtime::S3SourceStorageCredential::DefaultChain)
    {
        return Ok(None);
    }
    let attach_resolution_facts = |error: QueryError| {
        error
            .with_fact_if_absent("resource", "remote_parquet_source")
            .with_fact_if_absent("uri_scheme", resolved_remote_uri_scheme(resolved_source))
            .with_fact_if_absent("read_strategy", resolved_source.read_strategy().as_str())
            .with_fact_if_absent("remote_read_mode", cudf::io::parquet::RemoteReadMode::RequireKvikio.as_str())
            .with_fact_if_absent("credential_mode", credential_mode.as_str())
            .with_fact_if_absent("fallback_allowed", false)
    };
    let location = parse_s3_source_uri(uri, storage_profile_id.map(str::to_owned)).map_err(attach_resolution_facts)?;
    context
        .resolve_s3_default_chain_credential(
            crate::runtime::S3ObjectStoreKey {
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
    storage_profile: Option<&crate::runtime::SourceStorageProfile>,
    default_chain_credential: Option<&::object_store::aws::AwsCredential>,
) -> cudf::io::parquet::RemoteParquetProfile {
    let mut profile = cudf::io::parquet::RemoteParquetProfile::shared_process_env();
    if let Some(profile_id) = profile_id {
        profile = profile.with_profile_id(profile_id);
    }
    let Some(storage_profile) = storage_profile else {
        return profile;
    };
    if storage_profile.requires_scoped_remote_direct_profile() {
        profile = profile.with_credential_mode(cudf::io::parquet::RemoteCredentialMode::ScopedProfile);
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
        crate::runtime::S3SourceStorageCredential::DefaultChain => {
            if let Some(credential) = default_chain_credential {
                profile = profile.with_static_credentials(
                    credential.key_id.clone(),
                    credential.secret_key.clone(),
                    credential.token.clone(),
                );
            }
        },
        crate::runtime::S3SourceStorageCredential::Unsigned => {
            profile = profile.with_unsigned_requests();
        },
        crate::runtime::S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            profile = profile.with_static_credentials(
                access_key_id.clone(),
                secret_access_key.clone(),
                session_token.clone(),
            );
        },
    }
    profile
}
