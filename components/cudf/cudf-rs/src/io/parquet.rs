//! Parquet file I/O.

use std::{
    ffi::CString,
    fmt,
    marker::PhantomData,
    os::raw::c_char,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use typed_builder::TypedBuilder;

use crate::{
    core::{
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
        schema,
        table::Table,
    },
    io::TableReadOutput,
    ops::ast::AstExpression,
    runtime::Runtime,
};

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
    /// Fail unless cudf-rs can prove direct/range-aware remote IO.
    RequireDirect,
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
            Self::PreferDirect => "prefer_direct",
            Self::AllowHostFallback => "allow_host_fallback",
        }
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

    fn validate_for_current_reader(&self, operation: &'static str) -> Result<()> {
        self.validate_contract_for_current_reader(operation)?;
        Ok(())
    }

    fn can_use_path_reader_fallback(&self) -> bool {
        self.read_mode != RemoteReadMode::RequireDirect && self.profile.is_path_reader_compatible()
    }

    fn should_use_explicit_remote_datasource(&self) -> bool {
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
    /// cudf-rs can route the URI through its explicit KvikIO remote datasource.
    KvikioRemoteDatasource,
    /// libcudf was built without the KvikIO remote IO backend.
    BackendBuildDisabled,
    /// The requested credential mode cannot be honored by this cudf-rs build.
    CredentialModeUnsupported,
    /// The requested direct policy cannot be proven by this cudf-rs build.
    DirectEvidenceUnavailable,
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

/// Probe remote Parquet capability available at the cudf-rs boundary.
///
/// This is intentionally conservative: until cudf-sys/libcudf returns backend
/// evidence, `RequireDirect` is reported unavailable and other modes are
/// reported as unverified path-reader capability.
pub fn remote_parquet_capability(source: &RemoteParquetSource) -> RemoteParquetCapability {
    let backend_build_status = remote_parquet_backend_build_status();
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

/// Per-read Parquet source evidence returned by cudf-rs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParquetReadEvidence {
    pub requested_remote_read_mode: Option<RemoteReadMode>,
    pub input_kind: &'static str,
    pub backend: &'static str,
    pub fallback_reason: &'static str,
    pub remote_bytes: Option<u64>,
    pub remote_range_count: Option<u64>,
    pub device_read_support: &'static str,
    pub backend_build_status: RemoteParquetBackendBuildStatus,
    pub pruning: Option<ParquetPruningEvidence>,
}

/// Row-group pruning evidence returned by libcudf for a Parquet read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParquetPruningEvidence {
    /// Row groups participating in this read after any caller row-group selection.
    pub input_row_groups: u64,
    pub row_groups_after_stats_filter: Option<u64>,
    pub row_groups_after_bloom_filter: Option<u64>,
}

impl ParquetReadEvidence {
    fn remote_path_reader(source: &RemoteParquetSource) -> Self {
        Self {
            requested_remote_read_mode: Some(source.read_mode),
            input_kind: "remote_path",
            backend: "libcudf_path_reader_unverified",
            fallback_reason: "not_reported_by_libcudf",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_reported_by_libcudf",
            backend_build_status: remote_parquet_backend_build_status(),
            pruning: None,
        }
    }

    fn local_path_reader() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "local_path",
            backend: "libcudf_path_reader",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    pub(crate) fn local_path_footer_materialized() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "local_path",
            backend: "parquet_footer_materialized",
            fallback_reason: "all_null_columns_elided",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    pub(crate) fn local_path_multi_footer_materialized() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "local_path_multi",
            backend: "parquet_footer_materialized",
            fallback_reason: "all_null_columns_elided",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    fn host_buffer() -> Self {
        Self {
            requested_remote_read_mode: None,
            input_kind: "host_buffer",
            backend: "host_buffer",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        }
    }

    fn with_pruning(mut self, pruning: Option<ParquetPruningEvidence>) -> Self {
        self.pruning = pruning;
        self
    }
}

/// Options for reading Parquet into a low-level [`Table`].
///
/// The source must be exactly one of:
///
/// - `filepath`: a path-based libcudf/KvikIO read.
/// - `filebytes`: a borrowed host-memory Parquet payload.
///
/// Host-buffer reads borrow `filebytes` only for the duration of the reader
/// call. The returned table owns device-resident decoded columns and does not
/// retain a pointer into the host buffer.
///
/// **Advanced.** Most users should prefer
/// [`DataFrame::read_parquet`](crate::DataFrame::read_parquet).
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetTableReadOptions<'a> {
    #[builder(default, setter(transform = |path: impl AsRef<Path>| Some(path.as_ref().to_path_buf())))]
    filepath: Option<PathBuf>,
    #[builder(default, setter(strip_option))]
    remote_source: Option<RemoteParquetSource>,
    #[builder(default, setter(transform = |bytes: &'a [u8]| Some(bytes)))]
    filebytes: Option<&'a [u8]>,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    columns: Option<Vec<String>>,
    #[builder(default)]
    skip_rows: usize,
    #[builder(default, setter(transform = |num_rows: usize| Some(num_rows)))]
    num_rows: Option<usize>,
    #[builder(default, setter(transform = |groups: &[i32]| if groups.is_empty() { None } else { Some(groups.to_vec()) }))]
    row_groups: Option<Vec<i32>>,
    #[builder(default, setter(strip_option, into))]
    predicate: Option<Arc<AstExpression>>,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    runtime: Runtime,
    #[builder(default, setter(skip))]
    _lifetime: PhantomData<&'a [u8]>,
}

/// Options for reading multiple Parquet paths into one low-level [`Table`].
///
/// This is the multi-source counterpart to [`ParquetTableReadOptions`] for
/// path-compatible reads. File order is preserved by libcudf. When row groups are
/// supplied, each entry in `row_groups_per_file` applies to the corresponding
/// `inputs` entry.
#[derive(Debug, Clone)]
pub(crate) struct ParquetMultiTableReadOptions {
    pub(crate) inputs: Vec<ParquetMultiTableInput>,
    pub(crate) columns: Option<Vec<String>>,
    pub(crate) row_groups_per_file: Option<Vec<Vec<i32>>>,
    pub(crate) predicate: Option<Arc<AstExpression>>,
    pub(crate) runtime: Runtime,
}

#[derive(Debug, Clone)]
pub(crate) enum ParquetMultiTableInput {
    LocalPath(PathBuf),
    RemoteSource(RemoteParquetSource),
}

impl ParquetMultiTableInput {
    fn display_path(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::RemoteSource(source) => source.diagnostic_location(),
        }
    }

    fn remote_source(&self) -> Option<&RemoteParquetSource> {
        match self {
            Self::LocalPath(_) => None,
            Self::RemoteSource(source) => Some(source),
        }
    }
}

impl ParquetMultiTableReadOptions {
    pub(crate) fn from_inputs(
        inputs: Vec<ParquetMultiTableInput>,
        columns: Option<Vec<String>>,
        row_groups_per_file: Option<Vec<Vec<i32>>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self { inputs, columns, row_groups_per_file, predicate, runtime }
    }
}

impl<'a> ParquetTableReadOptions<'a> {
    pub(crate) fn from_parts(
        filepath: PathBuf,
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self {
            filepath: Some(filepath),
            remote_source: None,
            filebytes: None,
            columns,
            skip_rows,
            num_rows,
            row_groups,
            predicate,
            runtime,
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn from_host_buffer_parts(
        filebytes: &'a [u8],
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self {
            filepath: None,
            remote_source: None,
            filebytes: Some(filebytes),
            columns,
            skip_rows,
            num_rows,
            row_groups,
            predicate,
            runtime,
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn from_remote_parts(
        remote_source: RemoteParquetSource,
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self {
            filepath: None,
            remote_source: Some(remote_source),
            filebytes: None,
            columns,
            skip_rows,
            num_rows,
            row_groups,
            predicate,
            runtime,
            _lifetime: PhantomData,
        }
    }
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

struct RemoteProfileFfi {
    region: Option<CString>,
    access_key_id: Option<CString>,
    secret_access_key: Option<CString>,
    session_token: Option<CString>,
    endpoint: Option<CString>,
}

impl RemoteProfileFfi {
    fn new(profile: &RemoteParquetProfile, operation: &'static str) -> Result<Self> {
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

    fn as_raw(
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

fn remote_known_size_bytes_raw(known_size_bytes: Option<u64>) -> i64 {
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

fn empty_remote_read_evidence_raw() -> cudf_sys::cudf_io_remote_parquet_read_evidence_t {
    cudf_sys::cudf_io_remote_parquet_read_evidence_t {
        backend: 0,
        fallback_reason: 1,
        supports_device_read: 0,
        device_read_used: 0,
        remote_bytes: -1,
        remote_range_count: -1,
        host_read_bytes: -1,
        host_read_count: -1,
        device_read_bytes: -1,
        device_read_count: -1,
    }
}

fn empty_pruning_evidence_raw() -> cudf_sys::cudf_io_parquet_pruning_evidence_t {
    cudf_sys::cudf_io_parquet_pruning_evidence_t {
        input_row_groups: -1,
        has_row_groups_after_stats_filter: 0,
        row_groups_after_stats_filter: -1,
        has_row_groups_after_bloom_filter: 0,
        row_groups_after_bloom_filter: -1,
    }
}

fn pruning_evidence_from_raw(raw: cudf_sys::cudf_io_parquet_pruning_evidence_t) -> Option<ParquetPruningEvidence> {
    let input_row_groups = u64::try_from(raw.input_row_groups).ok()?;
    Some(ParquetPruningEvidence {
        input_row_groups,
        row_groups_after_stats_filter: if raw.has_row_groups_after_stats_filter != 0 {
            u64::try_from(raw.row_groups_after_stats_filter).ok()
        } else {
            None
        },
        row_groups_after_bloom_filter: if raw.has_row_groups_after_bloom_filter != 0 {
            u64::try_from(raw.row_groups_after_bloom_filter).ok()
        } else {
            None
        },
    })
}

fn remote_read_evidence_from_raw(
    source: Option<&RemoteParquetSource>,
    input_kind: &'static str,
    raw: cudf_sys::cudf_io_remote_parquet_read_evidence_t,
) -> ParquetReadEvidence {
    let requested_remote_read_mode = source.map(RemoteParquetSource::read_mode);
    let backend = match raw.backend {
        1 => "cudf_sys_kvikio_remote_datasource",
        _ => "unavailable",
    };
    let fallback_reason = match raw.fallback_reason {
        0 => "none",
        _ => "backend_unavailable",
    };
    let device_read_support = if raw.supports_device_read == 0 {
        "unsupported"
    } else if raw.device_read_used != 0 {
        "supported_used"
    } else {
        "supported_not_used"
    };
    ParquetReadEvidence {
        requested_remote_read_mode,
        input_kind,
        backend,
        fallback_reason,
        remote_bytes: u64::try_from(raw.remote_bytes).ok(),
        remote_range_count: u64::try_from(raw.remote_range_count).ok(),
        device_read_support,
        backend_build_status: remote_parquet_backend_build_status(),
        pruning: None,
    }
}

fn validate_require_direct_evidence(
    evidence: &ParquetReadEvidence,
    require_direct: bool,
    operation: &'static str,
) -> Result<()> {
    if !require_direct {
        return Ok(());
    }
    let direct_backend = evidence.backend == "cudf_sys_kvikio_remote_datasource";
    let no_fallback = evidence.fallback_reason == "none";
    let has_remote_bytes = evidence.remote_bytes.is_some_and(|bytes| bytes > 0);
    let has_remote_ranges = evidence.remote_range_count.is_some_and(|ranges| ranges > 0);
    let used_device_read = evidence.device_read_support == "supported_used";
    if direct_backend && no_fallback && has_remote_bytes && has_remote_ranges && used_device_read {
        return Ok(());
    }
    Err(Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "RequireDirect remote Parquet read did not produce required direct remote evidence",
    )
    .with_detail("resource", "remote_parquet_read_evidence")
    .with_detail("reason", "require_direct_evidence_unavailable")
    .with_detail("remote_read_mode", RemoteReadMode::RequireDirect.as_str())
    .with_detail("input_kind", evidence.input_kind)
    .with_detail("backend", evidence.backend)
    .with_detail("fallback_reason", evidence.fallback_reason)
    .with_detail("remote_bytes_positive", has_remote_bytes.to_string())
    .with_detail("remote_range_count_positive", has_remote_ranges.to_string())
    .with_detail("device_read_support", evidence.device_read_support)
    .with_detail("device_read_used", used_device_read.to_string())
    .with_detail("remote_backend_build_status", evidence.backend_build_status.as_str()))
}

fn checked_i64_from_usize(value: usize, field: &'static str, operation: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| error::range_limit_error(operation, field, "i64::MAX", value))
}

fn row_group_combination_error(
    row_group_count: usize,
    skip_rows: usize,
    num_rows_context: String,
    operation: &'static str,
) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "row_groups cannot be combined with skip_rows or num_rows",
    )
    .with_detail("row_group_count", row_group_count.to_string())
    .with_detail("skip_rows", skip_rows.to_string())
    .with_detail("num_rows", num_rows_context)
    .with_detail("resource", "row_groups")
    .with_detail("reason", "incompatible_option_combination")
    .with_detail("remedy", "use row_groups by itself, or remove row_groups and use skip_rows/num_rows")
}

/// Internal Parquet reader output with execution evidence.
pub(crate) struct ParquetTableReadOutput {
    output: TableReadOutput,
    read_evidence: Option<ParquetReadEvidence>,
}

impl ParquetTableReadOutput {
    pub(crate) fn read_evidence(&self) -> Option<&ParquetReadEvidence> {
        self.read_evidence.as_ref()
    }

    pub(crate) fn into_table_read_output(self) -> TableReadOutput {
        self.output
    }
}

/// Read a Parquet file into a low-level table result.
///
/// Returns the table data and the column names stored in the Parquet file's
/// schema. This avoids having to hardcode column indices when working with
/// Parquet files.
pub fn read_parquet_table(options: &ParquetTableReadOptions<'_>) -> Result<TableReadOutput> {
    read_parquet_table_with_evidence(options).map(ParquetTableReadOutput::into_table_read_output)
}

pub(crate) fn read_parquet_table_with_evidence(
    options: &ParquetTableReadOptions<'_>,
) -> Result<ParquetTableReadOutput> {
    const OPERATION: &str = "io::parquet::read_parquet_table";
    let filepath = options.filepath.as_ref().map(|path| super::path_to_cstring(path, OPERATION)).transpose()?;
    if let Some(path) = options.filepath.as_ref() {
        super::validate_input_path(path, OPERATION)?;
    }
    let remote_reader_path = options
        .remote_source
        .as_ref()
        .map(|source| {
            source.validate_for_current_reader(OPERATION)?;
            CString::new(source.reader_path()).map_err(|_| {
                error::interior_null_byte_error(
                    OPERATION,
                    "remote_parquet_uri",
                    "remote Parquet URI contains interior null byte",
                )
                .with_detail("resource", "remote_parquet_source")
            })
        })
        .transpose()?;
    let selected_source_count = usize::from(options.filepath.is_some())
        + usize::from(options.remote_source.is_some())
        + usize::from(options.filebytes.is_some());
    match selected_source_count {
        0 => {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                OPERATION,
                "Parquet read options must select a source",
            )
            .with_detail("resource", "parquet_source")
            .with_detail("reason", "missing_source")
            .with_detail("remedy", "set filepath, remote_source, or filebytes"));
        },
        1 => {},
        _ => {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                OPERATION,
                "Parquet read options must select exactly one source",
            )
            .with_detail("resource", "parquet_source")
            .with_detail("reason", "ambiguous_source")
            .with_detail("remedy", "set only one of filepath, remote_source, or filebytes"));
        },
    }
    if matches!(options.filebytes, Some([])) {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            OPERATION,
            "host-buffer Parquet reads require a non-empty buffer",
        )
        .with_detail("resource", "filebytes")
        .with_detail("reason", "empty_host_buffer"));
    }
    let (path_input, mut read_evidence, source_label) = if let Some(filepath) = filepath.as_ref() {
        (Some(filepath), ParquetReadEvidence::local_path_reader(), "filepath")
    } else if let Some(remote_reader_path) = remote_reader_path.as_ref() {
        let source =
            options.remote_source.as_ref().expect("remote source validation requires source when remote path exists");
        (Some(remote_reader_path), ParquetReadEvidence::remote_path_reader(source), "remote_path")
    } else {
        (None, ParquetReadEvidence::host_buffer(), "host_buffer")
    };
    let num_rows_context = options.num_rows.map(|num_rows| num_rows.to_string()).unwrap_or_else(|| "all".to_string());

    // Validate that row_groups is not combined with skip_rows or num_rows.
    if options.row_groups.is_some() && (options.skip_rows > 0 || options.num_rows.is_some()) {
        return Err(row_group_combination_error(
            options.row_groups.as_ref().map_or(0, Vec::len),
            options.skip_rows,
            num_rows_context,
            OPERATION,
        ));
    }
    let skip_rows = checked_i64_from_usize(options.skip_rows, "skip_rows", OPERATION)?;
    let num_rows = match options.num_rows {
        Some(num_rows) => checked_i64_from_usize(num_rows, "num_rows", OPERATION)?,
        None => -1,
    };

    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.exec_options_raw();

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut c_char = ptr::null_mut();
    let mut num_names: i32 = 0;

    // Convert column names to CStrings at the call site, then build the pointer array.
    let cstring_columns: Option<Vec<CString>> = options
        .columns
        .as_ref()
        .map(|cols| {
            cols.iter()
                .enumerate()
                .map(|(index, s)| {
                    CString::new(s.as_str()).map_err(|_| {
                        error::interior_null_byte_error(
                            OPERATION,
                            "column_name",
                            "column name contains interior null byte",
                        )
                        .with_fact(FACT_COLUMN_INDEX, index)
                        .with_column(s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let col_ptrs_vec: Vec<*const c_char> =
        cstring_columns.as_ref().map(|cols| cols.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (col_ptrs, num_cols) =
        if col_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (col_ptrs_vec.as_ptr(), col_ptrs_vec.len() as i32) };

    // Build row_groups pointer; the Option<Vec> must outlive the FFI call.
    let (rg_ptr, num_rg) = match options.row_groups.as_ref() {
        Some(rg) if !rg.is_empty() => (rg.as_ptr(), rg.len() as i32),
        _ => (ptr::null(), 0),
    };
    let predicate = options.predicate.as_ref().map_or(ptr::null_mut(), |predicate| predicate.raw_handle());

    let mut remote_raw_evidence = empty_remote_read_evidence_raw();
    let mut pruning_raw_evidence = empty_pruning_evidence_raw();
    let remote_profile_ffi =
        options.remote_source.as_ref().map(|source| RemoteProfileFfi::new(source.profile(), OPERATION)).transpose()?;
    let remote_profile_raw = options
        .remote_source
        .as_ref()
        .zip(remote_profile_ffi.as_ref())
        .map(|(source, profile_ffi)| profile_ffi.as_raw(source.profile(), source.known_size_bytes()));

    let use_explicit_remote_datasource =
        options.remote_source.as_ref().is_some_and(RemoteParquetSource::should_use_explicit_remote_datasource);
    let status = if use_explicit_remote_datasource {
        let filepath = path_input.expect("remote source validation requires path input");
        let profile_raw = remote_profile_raw.as_ref().expect("remote profile exists for remote source");
        // SAFETY: filepath is a valid CString. Profile pointers refer to
        // CStrings owned by remote_profile_ffi. Other pointers follow the same
        // contracts as the path reader.
        unsafe {
            cudf_sys::cudf_io_read_remote_parquet_with_metadata_filter_ex(
                filepath.as_ptr(),
                profile_raw,
                col_ptrs,
                num_cols,
                skip_rows,
                num_rows,
                rg_ptr,
                num_rg,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut remote_raw_evidence,
                &mut pruning_raw_evidence,
                &exec_options,
            )
        }
    } else if let Some(filepath) = path_input {
        // SAFETY: filepath is a valid CString. col_ptrs is either null or a
        // valid pointer to num_cols CString pointers. rg_ptr is either null
        // or a valid pointer to num_rg i32 values. predicate is either null
        // or an owned AST handle retained by options for the duration of this call.
        unsafe {
            cudf_sys::cudf_io_read_parquet_with_metadata_filter_ex(
                filepath.as_ptr(),
                col_ptrs,
                num_cols,
                skip_rows,
                num_rows,
                rg_ptr,
                num_rg,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut pruning_raw_evidence,
                &exec_options,
            )
        }
    } else {
        let bytes = options.filebytes.expect("source validation requires filebytes when filepath is absent");
        // SAFETY: bytes is a non-empty borrowed host buffer that outlives this
        // FFI call. Other pointers follow the same contracts as the path reader.
        unsafe {
            cudf_sys::cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
                bytes.as_ptr().cast(),
                bytes.len(),
                col_ptrs,
                num_cols,
                skip_rows,
                num_rows,
                rg_ptr,
                num_rg,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut pruning_raw_evidence,
                &exec_options,
            )
        }
    };
    if use_explicit_remote_datasource {
        let source =
            options.remote_source.as_ref().expect("remote source exists when explicit remote datasource is used");
        read_evidence = remote_read_evidence_from_raw(Some(source), "remote_path", remote_raw_evidence);
    }
    read_evidence = read_evidence.with_pruning(pruning_evidence_from_raw(pruning_raw_evidence));

    error::check_status_in_with_details(
        status,
        OPERATION,
        [
            ("column_name_count", num_cols.to_string()),
            ("skip_rows", options.skip_rows.to_string()),
            ("num_rows", num_rows_context),
            ("parquet_source", source_label.to_string()),
        ],
    )?;

    // SAFETY: table_handle was returned by a successful FFI call, transferring ownership.
    // Wrap early so table is freed via RAII if collect_column_names() fails.
    let table = unsafe { Table::from_raw_with_domain(table_handle, output_domain) }?;

    // Convert the C string array into Vec<String> and free the C memory.
    // SAFETY: names_ptr and num_names were returned by a successful FFI call.
    // cudf_io_free_column_names frees the C-allocated name strings and array.
    let column_names = unsafe {
        let names = super::collect_column_names(names_ptr, num_names, OPERATION);
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };
    validate_require_direct_evidence(
        &read_evidence,
        options.remote_source.as_ref().is_some_and(|source| source.read_mode() == RemoteReadMode::RequireDirect),
        OPERATION,
    )?;

    Ok(ParquetTableReadOutput {
        output: TableReadOutput { table, column_names: Some(column_names) },
        read_evidence: Some(read_evidence),
    })
}

/// Read multiple Parquet files into a low-level table result.
///
/// This uses libcudf's multi-source Parquet reader. It does not materialize
/// partition columns or perform DataFrame-level source projection.
pub(crate) fn read_parquet_multi_table(options: &ParquetMultiTableReadOptions) -> Result<ParquetTableReadOutput> {
    const OPERATION: &str = "io::parquet::read_parquet_multi_table";
    if options.inputs.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, OPERATION, "multi-file Parquet read requires files")
            .with_detail("resource", "filepaths")
            .with_detail("reason", "empty_input"));
    }
    for (file_index, input) in options.inputs.iter().enumerate() {
        match input {
            ParquetMultiTableInput::LocalPath(path) => {
                super::validate_input_path(path, OPERATION).map_err(|error| {
                    error
                        .with_detail("file_index", file_index.to_string())
                        .with_detail("path", path.display().to_string())
                })?;
            },
            ParquetMultiTableInput::RemoteSource(source) => {
                source.validate_for_current_reader(OPERATION).map_err(|error| {
                    error
                        .with_detail("file_index", file_index.to_string())
                        .with_detail("path", input.display_path())
                        .with_detail("remote_read_mode", source.read_mode().as_str())
                })?;
            },
        }
    }
    if let Some(row_groups_per_file) = &options.row_groups_per_file {
        if row_groups_per_file.len() != options.inputs.len() {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                OPERATION,
                "row-group selection count must match file count",
            )
            .with_detail("reason", "row_group_file_count_mismatch")
            .with_detail("file_count", options.inputs.len().to_string())
            .with_detail("row_group_file_count", row_groups_per_file.len().to_string()));
        }
    }

    let file_count = i32::try_from(options.inputs.len())
        .map_err(|_| error::range_limit_error(OPERATION, "filepaths", "i32::MAX", options.inputs.len()))?;
    let filepath_cstrings = options
        .inputs
        .iter()
        .enumerate()
        .map(|(file_index, input)| match input {
            ParquetMultiTableInput::LocalPath(path) => super::path_to_cstring(path, OPERATION).map_err(|error| {
                error.with_detail("file_index", file_index.to_string()).with_detail("path", path.display().to_string())
            }),
            ParquetMultiTableInput::RemoteSource(source) => CString::new(source.reader_path()).map_err(|_| {
                error::interior_null_byte_error(
                    OPERATION,
                    "remote_parquet_uri",
                    "remote Parquet URI contains interior null byte",
                )
                .with_detail("resource", "remote_parquet_source")
                .with_detail("file_index", file_index.to_string())
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let filepath_ptrs = filepath_cstrings.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();

    let cstring_columns: Option<Vec<CString>> = options
        .columns
        .as_ref()
        .map(|cols| {
            cols.iter()
                .enumerate()
                .map(|(index, s)| {
                    CString::new(s.as_str()).map_err(|_| {
                        error::interior_null_byte_error(
                            OPERATION,
                            "column_name",
                            "column name contains interior null byte",
                        )
                        .with_fact(FACT_COLUMN_INDEX, index)
                        .with_column(s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let col_ptrs_vec: Vec<*const c_char> =
        cstring_columns.as_ref().map(|cols| cols.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (col_ptrs, num_cols) =
        if col_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (col_ptrs_vec.as_ptr(), col_ptrs_vec.len() as i32) };

    let mut row_group_offsets = Vec::new();
    let mut row_groups = Vec::new();
    if let Some(row_groups_per_file) = &options.row_groups_per_file {
        row_group_offsets.reserve(row_groups_per_file.len() + 1);
        row_group_offsets.push(0_i32);
        for groups in row_groups_per_file {
            row_groups.extend(groups.iter().copied());
            let offset = i32::try_from(row_groups.len())
                .map_err(|_| error::range_limit_error(OPERATION, "row_groups", "i32::MAX", row_groups.len()))?;
            row_group_offsets.push(offset);
        }
    }
    let (row_group_offsets_ptr, num_row_group_offsets) = if row_group_offsets.is_empty() {
        (ptr::null(), 0)
    } else {
        (row_group_offsets.as_ptr(), row_group_offsets.len() as i32)
    };
    let (row_groups_ptr, num_row_groups) =
        if row_groups.is_empty() { (ptr::null(), 0) } else { (row_groups.as_ptr(), row_groups.len() as i32) };

    let output_domain = Arc::clone(options.runtime.execution_domain());
    let exec_options = output_domain.exec_options_raw();
    let predicate = options.predicate.as_ref().map_or(ptr::null_mut(), |predicate| predicate.raw_handle());
    let explicit_remote_source_flags = options
        .inputs
        .iter()
        .map(|input| input.remote_source().is_some_and(RemoteParquetSource::should_use_explicit_remote_datasource))
        .collect::<Vec<_>>();
    let has_explicit_remote_sources = explicit_remote_source_flags.iter().any(|flag| *flag);
    let remote_profile_ffi = if has_explicit_remote_sources {
        Some(
            options
                .inputs
                .iter()
                .map(|input| match input.remote_source() {
                    Some(source) => RemoteProfileFfi::new(source.profile(), OPERATION),
                    None => Ok(RemoteProfileFfi {
                        region: None,
                        access_key_id: None,
                        secret_access_key: None,
                        session_token: None,
                        endpoint: None,
                    }),
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let remote_profile_raw = remote_profile_ffi.as_ref().map(|profiles| {
        options
            .inputs
            .iter()
            .zip(profiles)
            .map(|(input, profile_ffi)| match input.remote_source() {
                Some(source) => profile_ffi.as_raw(source.profile(), source.known_size_bytes()),
                None => cudf_sys::cudf_io_remote_parquet_profile_t {
                    aws_region: ptr::null(),
                    aws_access_key_id: ptr::null(),
                    aws_secret_access_key: ptr::null(),
                    aws_session_token: ptr::null(),
                    aws_endpoint_url: ptr::null(),
                    path_style_access: -1,
                    unsigned_request: 0,
                    known_size_bytes: -1,
                },
            })
            .collect::<Vec<_>>()
    });
    let remote_source_flags = if has_explicit_remote_sources {
        Some(explicit_remote_source_flags.iter().map(|flag| i32::from(*flag)).collect::<Vec<_>>())
    } else {
        None
    };

    let mut table_handle = ptr::null_mut();
    let mut names_ptr: *mut *mut c_char = ptr::null_mut();
    let mut num_names: i32 = 0;
    let mut remote_raw_evidence = empty_remote_read_evidence_raw();
    let mut pruning_raw_evidence = empty_pruning_evidence_raw();

    // SAFETY: filepath_ptrs and column/row-group buffers are valid for this
    // call. The predicate handle, when present, is retained by options.
    let status = if has_explicit_remote_sources {
        let remote_source_flags = remote_source_flags.as_ref().expect("remote flags exist when remote sources exist");
        let remote_profile_raw = remote_profile_raw.as_ref().expect("remote profiles exist when remote sources exist");
        unsafe {
            cudf_sys::cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex(
                filepath_ptrs.as_ptr(),
                remote_source_flags.as_ptr(),
                remote_profile_raw.as_ptr(),
                file_count,
                col_ptrs,
                num_cols,
                row_group_offsets_ptr,
                num_row_group_offsets,
                row_groups_ptr,
                num_row_groups,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut remote_raw_evidence,
                &mut pruning_raw_evidence,
                &exec_options,
            )
        }
    } else {
        unsafe {
            cudf_sys::cudf_io_read_parquet_multi_with_metadata_filter_ex(
                filepath_ptrs.as_ptr(),
                file_count,
                col_ptrs,
                num_cols,
                row_group_offsets_ptr,
                num_row_group_offsets,
                row_groups_ptr,
                num_row_groups,
                predicate,
                &mut table_handle,
                &mut names_ptr,
                &mut num_names,
                &mut pruning_raw_evidence,
                &exec_options,
            )
        }
    };

    error::check_status_in_with_details(
        status,
        OPERATION,
        [
            ("file_count", options.inputs.len().to_string()),
            ("column_name_count", num_cols.to_string()),
            ("row_group_count", row_groups.len().to_string()),
        ],
    )?;

    // SAFETY: table_handle was returned by a successful FFI call.
    let table = unsafe { Table::from_raw_with_domain(table_handle, output_domain) }?;

    // SAFETY: names_ptr and num_names were returned by a successful FFI call.
    let column_names = unsafe {
        let names = super::collect_column_names(names_ptr, num_names, OPERATION);
        cudf_sys::cudf_io_free_column_names(names_ptr, num_names);
        names?
    };

    let read_evidence = if has_explicit_remote_sources {
        let first_remote = options.inputs.iter().find_map(ParquetMultiTableInput::remote_source);
        Some(remote_read_evidence_from_raw(
            first_remote,
            if options.inputs.iter().all(|input| input.remote_source().is_some()) {
                "remote_path_multi"
            } else {
                "mixed_path_multi"
            },
            remote_raw_evidence,
        ))
    } else {
        multi_table_read_evidence(&options.inputs)
    }
    .map(|evidence| evidence.with_pruning(pruning_evidence_from_raw(pruning_raw_evidence)));
    if let Some(evidence) = read_evidence.as_ref() {
        validate_require_direct_evidence(
            evidence,
            options
                .inputs
                .iter()
                .filter_map(ParquetMultiTableInput::remote_source)
                .any(|source| source.read_mode() == RemoteReadMode::RequireDirect),
            OPERATION,
        )?;
    }
    Ok(ParquetTableReadOutput { output: TableReadOutput { table, column_names: Some(column_names) }, read_evidence })
}

fn multi_table_read_evidence(inputs: &[ParquetMultiTableInput]) -> Option<ParquetReadEvidence> {
    let remote_sources = inputs.iter().filter_map(ParquetMultiTableInput::remote_source).collect::<Vec<_>>();
    if remote_sources.is_empty() {
        return Some(ParquetReadEvidence {
            requested_remote_read_mode: None,
            input_kind: "local_path_multi",
            backend: "libcudf_path_reader",
            fallback_reason: "not_applicable",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_applicable",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        });
    }
    let requested_remote_read_mode = remote_sources
        .first()
        .map(|source| source.read_mode())
        .filter(|first_mode| remote_sources.iter().all(|source| source.read_mode() == *first_mode));
    Some(ParquetReadEvidence {
        requested_remote_read_mode,
        input_kind: if remote_sources.len() == inputs.len() { "remote_path_multi" } else { "mixed_path_multi" },
        backend: "libcudf_path_reader_unverified",
        fallback_reason: "not_reported_by_libcudf",
        remote_bytes: None,
        remote_range_count: None,
        device_read_support: "not_reported_by_libcudf",
        backend_build_status: remote_parquet_backend_build_status(),
        pruning: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_evidence() -> ParquetReadEvidence {
        ParquetReadEvidence {
            requested_remote_read_mode: Some(RemoteReadMode::RequireDirect),
            input_kind: "remote_path",
            backend: "cudf_sys_kvikio_remote_datasource",
            fallback_reason: "none",
            remote_bytes: Some(1024),
            remote_range_count: Some(3),
            device_read_support: "supported_used",
            backend_build_status: RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled,
            pruning: None,
        }
    }

    #[test]
    fn require_direct_accepts_kvikio_remote_datasource_evidence() {
        validate_require_direct_evidence(&direct_evidence(), true, "test").unwrap();
    }

    #[test]
    fn require_direct_rejects_unverified_remote_path_evidence() {
        let evidence = ParquetReadEvidence {
            requested_remote_read_mode: Some(RemoteReadMode::RequireDirect),
            input_kind: "remote_path",
            backend: "libcudf_path_reader_unverified",
            fallback_reason: "not_reported_by_libcudf",
            remote_bytes: None,
            remote_range_count: None,
            device_read_support: "not_reported_by_libcudf",
            backend_build_status: RemoteParquetBackendBuildStatus::Unknown,
            pruning: None,
        };

        let error = validate_require_direct_evidence(&evidence, true, "test").unwrap_err();

        assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(error.detail_value("reason"), Some("require_direct_evidence_unavailable"));
        assert_eq!(error.detail_value("backend"), Some("libcudf_path_reader_unverified"));
    }

    #[test]
    fn require_direct_rejects_host_only_kvikio_remote_datasource_evidence() {
        let mut evidence = direct_evidence();
        evidence.device_read_support = "supported_not_used";

        let error = validate_require_direct_evidence(&evidence, true, "test").unwrap_err();

        assert_eq!(error.code(), ErrorCode::UnsupportedOperation);
        assert_eq!(error.detail_value("reason"), Some("require_direct_evidence_unavailable"));
        assert_eq!(error.detail_value("backend"), Some("cudf_sys_kvikio_remote_datasource"));
        assert_eq!(error.detail_value("device_read_used"), Some("false"));
    }

    #[test]
    fn remote_profile_raw_carries_known_size_bytes_hint() {
        let source = RemoteParquetSource::s3("s3://bucket/table/part-0.parquet").unwrap().with_known_size_bytes(4096);
        let profile_ffi = RemoteProfileFfi::new(source.profile(), "test").unwrap();

        let raw = profile_ffi.as_raw(source.profile(), source.known_size_bytes());

        assert_eq!(source.known_size_bytes(), Some(4096));
        assert_eq!(raw.known_size_bytes, 4096);
        assert_eq!(remote_known_size_bytes_raw(None), -1);
        assert_eq!(remote_known_size_bytes_raw(Some(i64::MAX as u64 + 1)), -1);
    }
}

/// Options for writing a Parquet file.
///
/// **Advanced.** Most users should prefer [`DataFrame::write_parquet`](crate::DataFrame::write_parquet).
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetTableWriteOptions {
    #[builder(setter(transform = |path: impl AsRef<Path>| path.as_ref().to_path_buf()))]
    filepath: PathBuf,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    column_names: Option<Vec<String>>,
    #[builder(default)]
    row_group_size_rows: i32,
    #[builder(default)]
    dictionary_policy: ParquetDictionaryPolicy,
    #[builder(default, setter(prefix = "with_", transform = |runtime: &Runtime| Some(runtime.clone())))]
    runtime: Option<Runtime>,
}

impl ParquetTableWriteOptions {
    pub(crate) fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }
}

/// Parquet dictionary encoding policy for cuDF writes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ParquetDictionaryPolicy {
    /// Use libcudf's default writer policy.
    #[default]
    Default,
    /// Use libcudf's adaptive dictionary policy.
    Adaptive,
    /// Disable dictionary encoding.
    Never,
    /// Always attempt dictionary encoding.
    Always,
}

impl ParquetDictionaryPolicy {
    pub(crate) const fn ffi_value(self) -> i32 {
        match self {
            Self::Default => -1,
            Self::Adaptive => 0,
            Self::Never => 1,
            Self::Always => 2,
        }
    }
}

/// Write a table to a Parquet file.
///
/// Returns an error if `column_names` were set on the options and their count
/// does not match the number of columns in the table.
pub fn write_parquet(table: &Table, options: &ParquetTableWriteOptions) -> Result<()> {
    let operation = "io::parquet::write_parquet";
    let table_view = table.view()?;
    let execution = super::resolve_write_execution(&table_view, options.runtime(), operation)?;
    write_parquet_with_exec(&table_view, options, execution.exec_options())
}

pub(crate) fn write_parquet_with_exec(
    table: &crate::core::TableView<'_>,
    options: &ParquetTableWriteOptions,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<()> {
    let filepath = super::path_to_cstring(&options.filepath, "io::parquet::write_parquet")?;

    // Validate column name count matches the table before the FFI call.
    if let Some(ref names) = options.column_names {
        let expected = table.num_columns()?;
        if names.len() != expected {
            return Err(schema::column_name_count_mismatch_error(
                "io::parquet::write_parquet",
                format!("column name count ({}) does not match table column count ({})", names.len(), expected),
                names.len(),
                expected,
            ));
        }
    }

    // Convert column names to CStrings at the call site.
    let cstring_names: Option<Vec<CString>> = options
        .column_names
        .as_ref()
        .map(|names| {
            names
                .iter()
                .enumerate()
                .map(|(index, s)| {
                    CString::new(s.as_str()).map_err(|_| {
                        error::interior_null_byte_error(
                            "io::parquet::write_parquet",
                            "column_name",
                            "column name contains interior null byte",
                        )
                        .with_fact(FACT_COLUMN_INDEX, index)
                        .with_column(s.to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    // Build the pointer array; the Vec must outlive the FFI call.
    let name_ptrs_vec: Vec<*const c_char> =
        cstring_names.as_ref().map(|names| names.iter().map(|c| c.as_ptr()).collect()).unwrap_or_default();
    let (name_ptrs, num_cols) =
        if name_ptrs_vec.is_empty() { (ptr::null(), 0) } else { (name_ptrs_vec.as_ptr(), name_ptrs_vec.len() as i32) };

    error::check_status_in_with_details(
        // SAFETY: table handle is valid (NonNull). filepath is a valid CString.
        // name_ptrs is either null or a valid pointer to num_cols CString pointers.
        unsafe {
            cudf_sys::cudf_io_write_parquet_ex(
                table.raw_handle(),
                filepath.as_ptr(),
                name_ptrs,
                num_cols,
                options.row_group_size_rows,
                options.dictionary_policy.ffi_value(),
                exec_options,
            )
        },
        "io::parquet::write_parquet",
        [("column_name_count", num_cols.to_string())],
    )
}
