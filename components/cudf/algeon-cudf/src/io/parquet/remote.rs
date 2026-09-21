//! Typed remote Parquet sources, capability probes, and profile FFI arguments.

use std::{ffi::CString, fmt, ptr};

use crate::core::error::{self, Error, ErrorCode, Result};

/// Credential mode used by a typed remote Parquet source profile.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteCredentialMode {
    /// libcudf/KvikIO resolves credentials from process-global configuration.
    #[default]
    SharedProcessEnv,
    /// Placeholder for future scoped credential passing across the FFI boundary.
    ScopedProfile,
}

impl RemoteCredentialMode {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SharedProcessEnv => "shared_process_env",
            Self::ScopedProfile => "scoped_profile",
        }
    }
}

/// Non-secret remote Parquet source profile.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct RemoteParquetProfile {
    profile_id: Option<String>,
    region: Option<String>,
    endpoint: Option<String>,
    path_style_access: Option<bool>,
    credential_mode: RemoteCredentialMode,
    credential: RemoteParquetCredential,
}

#[derive(Clone, Default, PartialEq, Eq, Hash)]
enum RemoteParquetCredential {
    #[default]
    DefaultChain,
    Unsigned,
    Static {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
}

impl RemoteParquetProfile {
    /// Empty profile that relies on libcudf/KvikIO process-global settings.
    pub fn shared_process_env() -> Self {
        Self::default()
    }

    /// Attach a non-secret profile identifier for diagnostics.
    pub fn with_profile_id(mut self, profile_id: impl Into<String>) -> Self {
        self.profile_id = Some(profile_id.into());
        self
    }

    /// Attach a non-secret region hint.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Attach a non-secret endpoint hint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Attach path-style access intent.
    pub fn with_path_style_access(mut self, path_style_access: bool) -> Self {
        self.path_style_access = Some(path_style_access);
        self
    }

    /// Select how credentials are resolved.
    pub fn with_credential_mode(mut self, credential_mode: RemoteCredentialMode) -> Self {
        self.credential_mode = credential_mode;
        self
    }

    /// Use unsigned S3 requests for this remote profile.
    pub fn with_unsigned_requests(mut self) -> Self {
        self.credential = RemoteParquetCredential::Unsigned;
        self.credential_mode = RemoteCredentialMode::ScopedProfile;
        self
    }

    /// Use scoped static S3 credentials for this remote profile.
    pub fn with_static_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.credential = RemoteParquetCredential::Static {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
        };
        self.credential_mode = RemoteCredentialMode::ScopedProfile;
        self
    }

    pub fn profile_id(&self) -> Option<&str> {
        self.profile_id.as_deref()
    }

    pub fn region(&self) -> Option<&str> {
        self.region.as_deref()
    }

    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    pub fn path_style_access(&self) -> Option<bool> {
        self.path_style_access
    }

    pub fn credential_mode(&self) -> RemoteCredentialMode {
        self.credential_mode
    }

    fn uses_unsigned_requests(&self) -> bool {
        matches!(self.credential, RemoteParquetCredential::Unsigned)
    }

    fn is_path_reader_compatible(&self) -> bool {
        self.region.is_none()
            && self.endpoint.is_none()
            && self.path_style_access.is_none()
            && self.credential_mode == RemoteCredentialMode::SharedProcessEnv
            && matches!(self.credential, RemoteParquetCredential::DefaultChain)
    }
}

impl fmt::Debug for RemoteParquetProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteParquetProfile")
            .field("profile_id", &self.profile_id)
            .field("region", &self.region)
            .field("endpoint", &self.endpoint.as_ref().map(|_| "<configured>"))
            .field("path_style_access", &self.path_style_access)
            .field("credential_mode", &self.credential_mode)
            .field("credential", &self.credential)
            .finish()
    }
}

impl fmt::Debug for RemoteParquetCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DefaultChain => f.write_str("DefaultChain"),
            Self::Unsigned => f.write_str("Unsigned"),
            Self::Static { session_token, .. } => f
                .debug_struct("Static")
                .field("access_key_id", &"<redacted>")
                .field("secret_access_key", &"<redacted>")
                .field("session_token", &session_token.as_ref().map(|_| "<redacted>"))
                .finish(),
        }
    }
}

/// Remote read fallback policy requested by the caller.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteReadMode {
    /// Fail unless cudf can prove direct/range-aware remote IO.
    RequireDirect,
    /// Fail unless cudf uses its explicit KvikIO remote datasource.
    ///
    /// Unlike [`Self::RequireDirect`], this accepts both native device reads
    /// and KvikIO compatibility IO. The caller still never supplies remote
    /// object bytes or falls back to libcudf's unverified path reader.
    RequireKvikio,
    /// Prefer direct/range-aware remote IO but allow lower layers to choose.
    #[default]
    PreferDirect,
    /// Explicitly allow lower-layer host fallback when libcudf chooses it.
    AllowHostFallback,
}

impl RemoteReadMode {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequireDirect => "require_direct",
            Self::RequireKvikio => "require_kvikio",
            Self::PreferDirect => "prefer_direct",
            Self::AllowHostFallback => "allow_host_fallback",
        }
    }

    pub(super) const fn requires_explicit_kvikio_datasource(self) -> bool {
        matches!(self, Self::RequireDirect | Self::RequireKvikio)
    }
}

/// Build-time remote Parquet backend capability reported by cudf-sys/libcudf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteParquetBackendBuildStatus {
    /// The lower layer did not report remote IO build status.
    Unknown,
    /// libcudf was compiled without `CUDF_KVIKIO_REMOTE_IO`.
    KvikioRemoteIoDisabled,
    /// libcudf was compiled with `CUDF_KVIKIO_REMOTE_IO`.
    KvikioRemoteIoEnabled,
}

impl RemoteParquetBackendBuildStatus {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::KvikioRemoteIoDisabled => "kvikio_remote_io_disabled",
            Self::KvikioRemoteIoEnabled => "kvikio_remote_io_enabled",
        }
    }
}

/// Probe remote Parquet backend capability available without starting a read.
pub fn remote_parquet_backend_build_status() -> RemoteParquetBackendBuildStatus {
    match unsafe { cudf_sys::cudf_io_remote_parquet_kvikio_remote_io_enabled() } {
        0 => RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled,
        1 => RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled,
        _ => RemoteParquetBackendBuildStatus::Unknown,
    }
}

/// Typed remote Parquet source.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RemoteParquetSource {
    uri: String,
    profile: RemoteParquetProfile,
    read_mode: RemoteReadMode,
    known_size_bytes: Option<u64>,
}

impl RemoteParquetSource {
    /// Build an S3 remote Parquet source.
    pub fn s3(uri: impl Into<String>) -> Result<Self> {
        let uri = uri.into();
        validate_remote_parquet_uri(&uri)?;
        Ok(Self {
            uri,
            profile: RemoteParquetProfile::default(),
            read_mode: RemoteReadMode::PreferDirect,
            known_size_bytes: None,
        })
    }

    /// Attach a non-secret remote profile.
    pub fn with_profile(mut self, profile: RemoteParquetProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Select remote read fallback policy.
    pub fn with_read_mode(mut self, read_mode: RemoteReadMode) -> Self {
        self.read_mode = read_mode;
        self
    }

    /// Attach the known full object size in bytes as an object-store metadata
    /// probe avoidance hint.
    pub fn with_known_size_bytes(mut self, known_size_bytes: u64) -> Self {
        self.known_size_bytes = Some(known_size_bytes);
        self
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn profile(&self) -> &RemoteParquetProfile {
        &self.profile
    }

    pub fn read_mode(&self) -> RemoteReadMode {
        self.read_mode
    }

    pub fn known_size_bytes(&self) -> Option<u64> {
        self.known_size_bytes
    }

    pub(crate) fn reader_path(&self) -> &str {
        &self.uri
    }

    pub(crate) fn diagnostic_location(&self) -> String {
        format!("{}://<redacted>", remote_uri_scheme(&self.uri).unwrap_or("remote"))
    }

    pub(crate) fn validate_contract_for_current_reader(&self, operation: &'static str) -> Result<()> {
        if self.profile.endpoint().is_some() && self.profile.path_style_access() == Some(false) {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                operation,
                "virtual-hosted custom S3 endpoints are not supported by the scoped remote Parquet datasource",
            )
            .with_detail("resource", "remote_parquet_source")
            .with_detail("reason", "remote_direct_virtual_hosted_endpoint_unsupported")
            .with_detail("credential_mode", self.profile.credential_mode.as_str())
            .with_detail("uri_scheme", remote_uri_scheme(&self.uri).unwrap_or("unknown"))
            .with_detail(
                "remedy",
                "use path-style access for custom S3-compatible endpoints or omit the endpoint override",
            ));
        }
        Ok(())
    }

    pub(super) fn can_use_path_reader_fallback(&self) -> bool {
        !self.read_mode.requires_explicit_kvikio_datasource() && self.profile.is_path_reader_compatible()
    }

    pub(super) fn should_use_explicit_remote_datasource(&self) -> bool {
        !self.can_use_path_reader_fallback()
            || remote_parquet_backend_build_status() == RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled
    }
}

impl fmt::Debug for RemoteParquetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteParquetSource")
            .field("uri_scheme", &remote_uri_scheme(&self.uri))
            .field("uri", &"<configured>")
            .field("profile", &self.profile)
            .field("read_mode", &self.read_mode)
            .field("known_size_bytes", &self.known_size_bytes)
            .finish()
    }
}

/// Remote Parquet capability disposition available at the Rust API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteParquetCapabilityDisposition {
    /// The URI can be handed to libcudf's path reader, but the backend is not proven.
    PathReaderUnverified,
    /// cudf can route the URI through its explicit KvikIO remote datasource.
    KvikioRemoteDatasource,
    /// libcudf was built without the KvikIO remote IO backend.
    BackendBuildDisabled,
    /// The requested credential mode cannot be honored by this cudf build.
    CredentialModeUnsupported,
    /// The requested direct policy cannot be proven by this cudf build.
    DirectEvidenceUnavailable,
    /// The requested explicit KvikIO datasource cannot be proven by this cudf build.
    KvikioEvidenceUnavailable,
}

/// Capability probe result for a typed remote Parquet source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteParquetCapability {
    pub disposition: RemoteParquetCapabilityDisposition,
    pub uri_scheme: String,
    pub read_mode: RemoteReadMode,
    pub credential_mode: RemoteCredentialMode,
    pub backend_build_status: RemoteParquetBackendBuildStatus,
}

/// Datasource-level capability reported by libcudf for a path/URI.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteParquetDatasourceProbe {
    pub supports_device_read: bool,
    pub device_read_preferred: bool,
    pub size_bytes: Option<u64>,
    pub backend_build_status: RemoteParquetBackendBuildStatus,
}

/// Probe remote Parquet capability available at the cudf boundary.
///
/// This is intentionally conservative: until cudf-sys/libcudf returns backend
/// evidence, `RequireDirect` is reported unavailable and other modes are
/// reported as unverified path-reader capability.
pub fn remote_parquet_capability(source: &RemoteParquetSource) -> RemoteParquetCapability {
    remote_parquet_capability_with_status(source, remote_parquet_backend_build_status())
}

pub(super) fn remote_parquet_capability_with_status(
    source: &RemoteParquetSource,
    backend_build_status: RemoteParquetBackendBuildStatus,
) -> RemoteParquetCapability {
    RemoteParquetCapability {
        disposition: if source.profile.endpoint().is_some() && source.profile.path_style_access() == Some(false) {
            RemoteParquetCapabilityDisposition::CredentialModeUnsupported
        } else if backend_build_status == RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled {
            RemoteParquetCapabilityDisposition::KvikioRemoteDatasource
        } else if source.can_use_path_reader_fallback() {
            RemoteParquetCapabilityDisposition::PathReaderUnverified
        } else if backend_build_status == RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled {
            RemoteParquetCapabilityDisposition::BackendBuildDisabled
        } else if source.profile.credential_mode != RemoteCredentialMode::SharedProcessEnv {
            RemoteParquetCapabilityDisposition::CredentialModeUnsupported
        } else if source.read_mode == RemoteReadMode::RequireKvikio {
            RemoteParquetCapabilityDisposition::KvikioEvidenceUnavailable
        } else {
            RemoteParquetCapabilityDisposition::DirectEvidenceUnavailable
        },
        uri_scheme: remote_uri_scheme(source.uri()).unwrap_or("unknown").to_string(),
        read_mode: source.read_mode,
        credential_mode: source.profile.credential_mode,
        backend_build_status,
    }
}

/// Probe libcudf datasource capability for a typed remote Parquet source.
///
/// This opens the datasource but does not launch a Parquet decode. The lower
/// layer may still contact object-store metadata services while opening the
/// datasource, so callers should use this as an explicit capability check, not
/// as a cheap pure function.
pub fn probe_remote_parquet_datasource(
    source: &RemoteParquetSource,
    preferred_read_size_bytes: Option<u64>,
) -> Result<RemoteParquetDatasourceProbe> {
    const OPERATION: &str = "io::parquet::probe_remote_parquet_datasource";
    source.validate_contract_for_current_reader(OPERATION)?;
    let reader_path = CString::new(source.reader_path()).map_err(|_| {
        error::interior_null_byte_error(
            OPERATION,
            "remote_parquet_uri",
            "remote Parquet URI contains interior null byte",
        )
        .with_detail("resource", "remote_parquet_source")
    })?;
    let preferred_read_size = match preferred_read_size_bytes {
        Some(value) => i64::try_from(value).map_err(|_| {
            Error::new_with(ErrorCode::UnsupportedOperation, OPERATION, "preferred read size exceeds i64::MAX")
                .with_detail("resource", "preferred_read_size_bytes")
                .with_detail("reason", "preferred_read_size_exceeds_i64_max")
                .with_detail("preferred_read_size_bytes", value.to_string())
        })?,
        None => -1,
    };
    let profile_ffi = RemoteProfileFfi::new(source.profile(), OPERATION)?;
    let profile_raw = profile_ffi.as_raw(source.profile(), source.known_size_bytes());
    let mut out = cudf_sys::cudf_io_datasource_probe_result_t {
        can_open: 0,
        supports_device_read: 0,
        device_read_preferred: 0,
        size_bytes: -1,
    };
    error::check_status_in_with_details(
        // SAFETY: reader_path is a valid C string and `out` points to a valid
        // result struct for the duration of the call. `profile_raw` points to
        // C strings owned by `profile_ffi`, which also outlives the call.
        unsafe {
            cudf_sys::cudf_io_probe_remote_parquet_datasource(
                reader_path.as_ptr(),
                &profile_raw,
                preferred_read_size,
                &mut out,
            )
        },
        OPERATION,
        [
            ("resource", "remote_parquet_datasource".to_string()),
            ("uri_scheme", remote_uri_scheme(source.uri()).unwrap_or("unknown").to_string()),
            ("read_mode", source.read_mode().as_str().to_string()),
            ("credential_mode", source.profile().credential_mode().as_str().to_string()),
            ("preferred_read_size_bytes", preferred_read_size.to_string()),
        ],
    )?;
    Ok(RemoteParquetDatasourceProbe {
        supports_device_read: out.supports_device_read != 0,
        device_read_preferred: out.device_read_preferred != 0,
        size_bytes: u64::try_from(out.size_bytes).ok(),
        backend_build_status: remote_parquet_backend_build_status(),
    })
}

fn validate_remote_parquet_uri(uri: &str) -> Result<()> {
    const OPERATION: &str = "io::parquet::RemoteParquetSource::s3";
    match remote_uri_scheme(uri) {
        Some(scheme) if scheme.eq_ignore_ascii_case("s3") => Ok(()),
        Some(scheme) => {
            Err(Error::new_with(ErrorCode::UnsupportedOperation, OPERATION, "unsupported remote Parquet URI scheme")
                .with_detail("resource", "remote_parquet_uri")
                .with_detail("reason", "unsupported_uri_scheme")
                .with_detail("scheme", scheme.to_string()))
        },
        None => {
            Err(Error::new_with(ErrorCode::ForeignInvalidArgument, OPERATION, "remote Parquet URI is missing a scheme")
                .with_detail("resource", "remote_parquet_uri")
                .with_detail("reason", "missing_uri_scheme"))
        },
    }
}

fn remote_uri_scheme(uri: &str) -> Option<&str> {
    uri.split_once("://").map(|(scheme, _)| scheme).filter(|scheme| !scheme.is_empty())
}

pub(super) struct RemoteProfileFfi {
    region: Option<CString>,
    access_key_id: Option<CString>,
    secret_access_key: Option<CString>,
    session_token: Option<CString>,
    endpoint: Option<CString>,
}

impl RemoteProfileFfi {
    pub(super) fn empty() -> Self {
        Self { region: None, access_key_id: None, secret_access_key: None, session_token: None, endpoint: None }
    }

    pub(super) fn new(profile: &RemoteParquetProfile, operation: &'static str) -> Result<Self> {
        Ok(Self {
            region: optional_cstring(profile.region(), operation, "aws_region")?,
            access_key_id: match &profile.credential {
                RemoteParquetCredential::Static { access_key_id, .. } => {
                    Some(cstring_field(access_key_id, operation, "aws_access_key_id")?)
                },
                _ => None,
            },
            secret_access_key: match &profile.credential {
                RemoteParquetCredential::Static { secret_access_key, .. } => {
                    Some(cstring_field(secret_access_key, operation, "aws_secret_access_key")?)
                },
                _ => None,
            },
            session_token: match &profile.credential {
                RemoteParquetCredential::Static { session_token: Some(session_token), .. } => {
                    Some(cstring_field(session_token, operation, "aws_session_token")?)
                },
                _ => None,
            },
            endpoint: match profile.endpoint() {
                Some(endpoint) => {
                    validate_remote_endpoint_url_no_secret(endpoint, operation)?;
                    Some(cstring_field(endpoint, operation, "aws_endpoint_url")?)
                },
                None => None,
            },
        })
    }

    pub(super) fn as_raw(
        &self,
        profile: &RemoteParquetProfile,
        known_size_bytes: Option<u64>,
    ) -> cudf_sys::cudf_io_remote_parquet_profile_t {
        cudf_sys::cudf_io_remote_parquet_profile_t {
            aws_region: self.region.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            aws_access_key_id: self.access_key_id.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            aws_secret_access_key: self.secret_access_key.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            aws_session_token: self.session_token.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            aws_endpoint_url: self.endpoint.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            path_style_access: profile.path_style_access().map_or(-1, i32::from),
            unsigned_request: i32::from(profile.uses_unsigned_requests()),
            known_size_bytes: remote_known_size_bytes_raw(known_size_bytes),
        }
    }
}

pub(super) fn remote_known_size_bytes_raw(known_size_bytes: Option<u64>) -> i64 {
    known_size_bytes.and_then(|bytes| i64::try_from(bytes).ok()).unwrap_or(-1)
}

fn optional_cstring(value: Option<&str>, operation: &'static str, field: &'static str) -> Result<Option<CString>> {
    value.map(|value| cstring_field(value, operation, field)).transpose()
}

fn cstring_field(value: &str, operation: &'static str, field: &'static str) -> Result<CString> {
    CString::new(value).map_err(|_| {
        error::interior_null_byte_error(operation, field, "remote Parquet profile field contains interior null byte")
            .with_detail("resource", "remote_parquet_profile")
    })
}

fn validate_remote_endpoint_url_no_secret(endpoint: &str, operation: &'static str) -> Result<()> {
    if endpoint.contains('@') || endpoint.contains('?') || endpoint.contains('#') {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            operation,
            "remote Parquet endpoint URL must not contain userinfo, query, or fragment components",
        )
        .with_detail("resource", "remote_parquet_profile")
        .with_detail("reason", "endpoint_contains_secret_bearing_components")
        .with_detail("field", "aws_endpoint_url"));
    }
    Ok(())
}
