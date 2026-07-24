use std::{collections::HashMap, fmt};

/// One credential mode for an S3-compatible source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3CredentialMode {
    /// Allow the SDK/default provider chain to resolve credentials.
    DefaultChain,
    /// Use credentials explicitly supplied to this policy.
    ExplicitStatic,
    /// Make unsigned requests and do not inspect ambient credential sources.
    Unsigned,
}

/// Whether a resolved Iceberg table's FileIO proves use of its frozen S3
/// source-access policy.
///
/// This intentionally carries no property values: endpoints and static
/// credentials remain inside the policy/FileIO implementations. The root
/// uses it only to decide whether a remote S3 native path may share the CPU
/// delegate's proven source configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3FileIoConfigVerification {
    /// FileIO's complete S3 property set exactly matches the frozen policy.
    ExactPolicy,
    /// FileIO has no S3 properties. This is valid for local CPU/native paths,
    /// but cannot prove a remote S3 configuration.
    NoS3Properties,
    /// FileIO carries S3 properties that differ from the frozen policy.
    Mismatch,
}

/// Borrowed S3 access view for a CPU delegate or native-profile conversion.
///
/// Values in [`Self::credential`] are sensitive when static credentials are
/// selected. Callers must pass them directly to a storage client/profile and
/// must never put them in logs, diagnostics, or scan candidates.
#[derive(Clone, Copy)]
pub struct S3SourceAccessRef<'a> {
    pub endpoint: Option<&'a str>,
    pub region: Option<&'a str>,
    pub path_style_access: Option<bool>,
    pub credential: S3CredentialRef<'a>,
}

/// Borrowed credential material from the frozen source-access policy.
#[derive(Clone, Copy)]
pub enum S3CredentialRef<'a> {
    DefaultChain,
    Static { access_key_id: &'a str, secret_access_key: &'a str, session_token: Option<&'a str> },
    Unsigned,
}

impl S3CredentialRef<'_> {
    /// Returns the credential-selection mode without exposing material.
    pub const fn mode(self) -> S3CredentialMode {
        match self {
            Self::DefaultChain => S3CredentialMode::DefaultChain,
            Self::Static { .. } => S3CredentialMode::ExplicitStatic,
            Self::Unsigned => S3CredentialMode::Unsigned,
        }
    }
}

impl fmt::Debug for S3SourceAccessRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3SourceAccessRef")
            .field("endpoint", &self.endpoint.map(|_| "<redacted>"))
            .field("region", &self.region)
            .field("path_style_access", &self.path_style_access)
            .field("credential_mode", &self.credential.mode())
            .finish()
    }
}

impl fmt::Debug for S3CredentialRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3CredentialRef").field("mode", &self.mode()).field("material", &"<redacted>").finish()
    }
}

/// Credential-safe access policy shared by leaf scan planning and the later
/// root CPU/native delegates.
#[derive(Clone, PartialEq, Eq)]
pub enum SourceAccessPolicy {
    S3(S3SourceAccessPolicy),
}

impl SourceAccessPolicy {
    /// Creates an S3 policy that permits the normal AWS default chain.
    pub fn s3_default_chain(region: impl Into<Option<String>>) -> Self {
        Self::S3(S3SourceAccessPolicy::default_chain(region))
    }

    /// Creates a custom-endpoint S3 policy. It defaults to unsigned requests,
    /// preventing a local endpoint from silently consuming ambient AWS
    /// credentials. Call [`S3SourceAccessPolicy::with_default_chain`] only for
    /// an explicit operator opt-in.
    pub fn s3_custom_endpoint(
        endpoint: impl Into<String>,
        region: impl Into<Option<String>>,
        path_style_access: Option<bool>,
    ) -> Self {
        Self::S3(S3SourceAccessPolicy::custom_endpoint(endpoint, region, path_style_access))
    }

    /// Returns S3 policy details when this is an S3 source.
    pub const fn as_s3(&self) -> &S3SourceAccessPolicy {
        match self {
            Self::S3(policy) => policy,
        }
    }

    /// Returns the exact frozen material that both leaf FileIO and a later
    /// root native-profile conversion must consume. The view borrows this
    /// policy and therefore cannot be retained in a scan candidate.
    pub fn s3_access(&self) -> S3SourceAccessRef<'_> {
        self.as_s3().access_ref()
    }

    /// Verifies whether a resolved table's FileIO S3 properties are exactly
    /// the properties derived from this frozen policy. The caller must keep
    /// the property map private because it can contain static credentials.
    pub fn verify_s3_file_io_properties(
        &self,
        file_io_properties: &HashMap<String, String>,
    ) -> S3FileIoConfigVerification {
        match self {
            Self::S3(policy) => policy.verify_file_io_properties(file_io_properties),
        }
    }

    /// Applies this source policy to an Iceberg catalog/FileIO property map.
    /// Callers must not log the returned map because it can hold static S3
    /// credentials.
    #[cfg_attr(not(feature = "catalog"), allow(dead_code))]
    pub(crate) fn apply_catalog_properties(&self, properties: &mut HashMap<String, String>) {
        match self {
            Self::S3(policy) => policy.apply_catalog_properties(properties),
        }
    }

    /// Returns exactly the properties that construct this policy's FileIO.
    /// Catalog credentials are deliberately excluded: Glue metadata and table
    /// data can be authorized independently.
    #[cfg(feature = "catalog")]
    pub(crate) fn file_io_properties(&self) -> HashMap<String, String> {
        let mut properties = HashMap::new();
        self.apply_catalog_properties(&mut properties);
        properties
    }
}

impl fmt::Debug for SourceAccessPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S3(policy) => f.debug_tuple("S3").field(policy).finish(),
        }
    }
}

/// S3-compatible endpoint and credential-selection policy.
#[derive(Clone, PartialEq, Eq)]
pub struct S3SourceAccessPolicy {
    endpoint: Option<String>,
    region: Option<String>,
    path_style_access: Option<bool>,
    credential: S3Credential,
}

#[derive(Clone, PartialEq, Eq)]
enum S3Credential {
    DefaultChain,
    ExplicitStatic(S3StaticCredentials),
    Unsigned,
}

#[derive(Clone, PartialEq, Eq)]
struct S3StaticCredentials {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

impl S3SourceAccessPolicy {
    /// Creates a policy for AWS's normal endpoint.
    pub fn default_chain(region: impl Into<Option<String>>) -> Self {
        Self {
            endpoint: None,
            region: region.into().filter(|value| !value.trim().is_empty()),
            path_style_access: None,
            credential: S3Credential::DefaultChain,
        }
    }

    /// Creates a custom-endpoint policy with the secure default-deny setting.
    pub fn custom_endpoint(
        endpoint: impl Into<String>,
        region: impl Into<Option<String>>,
        path_style_access: Option<bool>,
    ) -> Self {
        Self {
            endpoint: Some(endpoint.into()),
            region: region.into().filter(|value| !value.trim().is_empty()),
            path_style_access,
            credential: S3Credential::Unsigned,
        }
    }

    /// Explicitly opts the source into the ambient/default credential chain.
    pub fn with_default_chain(mut self) -> Self {
        self.credential = S3Credential::DefaultChain;
        self
    }

    /// Selects static credentials. They are redacted in `Debug` and are never
    /// copied into scan candidates or error facts.
    pub fn with_static_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.credential = S3Credential::ExplicitStatic(S3StaticCredentials {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
        });
        self
    }

    /// Returns the configured endpoint. Endpoint URLs may include user info;
    /// consumers must treat the value as sensitive and never put it in facts.
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    /// Returns the configured region.
    pub fn region(&self) -> Option<&str> {
        self.region.as_deref()
    }

    /// Returns the configured path-style setting.
    pub const fn path_style_access(&self) -> Option<bool> {
        self.path_style_access
    }

    /// Returns the credential-selection mode without exposing credential
    /// material.
    pub const fn credential_mode(&self) -> S3CredentialMode {
        match self.credential {
            S3Credential::DefaultChain => S3CredentialMode::DefaultChain,
            S3Credential::ExplicitStatic(_) => S3CredentialMode::ExplicitStatic,
            S3Credential::Unsigned => S3CredentialMode::Unsigned,
        }
    }

    /// Whether this policy has the custom-endpoint default-deny behavior.
    pub fn denies_ambient_credentials(&self) -> bool {
        self.endpoint.is_some() && !matches!(self.credential, S3Credential::DefaultChain)
    }

    fn access_ref(&self) -> S3SourceAccessRef<'_> {
        let credential = match &self.credential {
            S3Credential::DefaultChain => S3CredentialRef::DefaultChain,
            S3Credential::ExplicitStatic(credentials) => S3CredentialRef::Static {
                access_key_id: &credentials.access_key_id,
                secret_access_key: &credentials.secret_access_key,
                session_token: credentials.session_token.as_deref(),
            },
            S3Credential::Unsigned => S3CredentialRef::Unsigned,
        };
        S3SourceAccessRef {
            endpoint: self.endpoint(),
            region: self.region(),
            path_style_access: self.path_style_access,
            credential,
        }
    }

    #[cfg_attr(not(feature = "catalog"), allow(dead_code))]
    fn apply_catalog_properties(&self, properties: &mut HashMap<String, String>) {
        if let Some(endpoint) = &self.endpoint {
            properties.insert("s3.endpoint".to_string(), endpoint.clone());
        }
        if let Some(region) = &self.region {
            properties.insert("s3.region".to_string(), region.clone());
        }
        if let Some(path_style_access) = self.path_style_access {
            properties.insert("s3.path-style-access".to_string(), path_style_access.to_string());
        }
        match &self.credential {
            S3Credential::DefaultChain => {},
            S3Credential::ExplicitStatic(credentials) => {
                properties.insert("s3.access-key-id".to_string(), credentials.access_key_id.clone());
                properties.insert("s3.secret-access-key".to_string(), credentials.secret_access_key.clone());
                if let Some(session_token) = &credentials.session_token {
                    properties.insert("s3.session-token".to_string(), session_token.clone());
                }
                if self.endpoint.is_some() {
                    properties.insert("s3.disable-config-load".to_string(), "true".to_string());
                    properties.insert("s3.disable-ec2-metadata".to_string(), "true".to_string());
                }
            },
            S3Credential::Unsigned => {
                properties.insert("s3.allow-anonymous".to_string(), "true".to_string());
                properties.insert("s3.disable-config-load".to_string(), "true".to_string());
                properties.insert("s3.disable-ec2-metadata".to_string(), "true".to_string());
            },
        }
    }

    fn verify_file_io_properties(&self, file_io_properties: &HashMap<String, String>) -> S3FileIoConfigVerification {
        let configured_s3_properties =
            file_io_properties.iter().filter(|(key, _)| key.starts_with("s3.")).collect::<HashMap<_, _>>();
        if configured_s3_properties.is_empty() {
            return S3FileIoConfigVerification::NoS3Properties;
        }

        let mut expected_s3_properties = HashMap::new();
        self.apply_catalog_properties(&mut expected_s3_properties);
        let exact = configured_s3_properties.len() == expected_s3_properties.len()
            && configured_s3_properties.into_iter().all(|(key, value)| expected_s3_properties.get(key) == Some(value));
        if exact { S3FileIoConfigVerification::ExactPolicy } else { S3FileIoConfigVerification::Mismatch }
    }
}

impl fmt::Debug for S3SourceAccessPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3SourceAccessPolicy")
            .field("endpoint", &self.endpoint.as_ref().map(|_| "<redacted>"))
            .field("region", &self.region)
            .field("path_style_access", &self.path_style_access)
            .field("credential_mode", &self.credential_mode())
            .field("ambient_credentials", &if self.denies_ambient_credentials() { "default_deny" } else { "allowed" })
            .finish()
    }
}

/// Glue catalog configuration. Its source policy is the single policy shared
/// by the resolved table, leaf scan planning, and future root delegates.
#[derive(Clone, PartialEq, Eq)]
pub struct GlueIcebergCatalogConfig {
    pub catalog_handle: String,
    pub warehouse: String,
    pub region: String,
    pub catalog_id: Option<String>,
    pub profile: Option<String>,
    uri: Option<String>,
    source_access: SourceAccessPolicy,
    catalog_credentials: GlueCatalogCredentials,
}

/// Frozen authentication for Glue metadata operations.
///
/// This intentionally remains separate from the S3 source policy: a catalog
/// can require AWS credentials while table data uses a different credential
/// at a custom endpoint.
#[derive(Clone, PartialEq, Eq)]
enum GlueCatalogCredentials {
    DefaultChain,
    ExplicitStatic(S3StaticCredentials),
}

impl GlueCatalogCredentials {
    fn apply_catalog_properties(&self, properties: &mut HashMap<String, String>) {
        let Self::ExplicitStatic(credentials) = self else {
            return;
        };
        // These keys are the upstream Glue catalog builder's typed property
        // contract. They deliberately differ from the `s3.*` FileIO keys:
        // Glue metadata and table-data access can use separate credentials.
        properties.insert("aws_access_key_id".to_string(), credentials.access_key_id.clone());
        properties.insert("aws_secret_access_key".to_string(), credentials.secret_access_key.clone());
        if let Some(session_token) = &credentials.session_token {
            properties.insert("aws_session_token".to_string(), session_token.clone());
        }
    }

    const fn mode(&self) -> S3CredentialMode {
        match self {
            Self::DefaultChain => S3CredentialMode::DefaultChain,
            Self::ExplicitStatic(_) => S3CredentialMode::ExplicitStatic,
        }
    }
}

impl GlueIcebergCatalogConfig {
    /// Creates Glue catalog configuration with an AWS-default source policy.
    pub fn new(catalog_handle: impl Into<String>, warehouse: impl Into<String>, region: impl Into<String>) -> Self {
        let region = region.into();
        Self {
            catalog_handle: catalog_handle.into(),
            warehouse: warehouse.into(),
            source_access: SourceAccessPolicy::s3_default_chain(Some(region.clone())),
            region,
            catalog_id: None,
            profile: None,
            uri: None,
            catalog_credentials: GlueCatalogCredentials::DefaultChain,
        }
    }

    pub fn with_catalog_id(mut self, catalog_id: impl Into<String>) -> Self {
        self.catalog_id = Some(catalog_id.into());
        self
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Sets the Glue endpoint. It is redacted from `Debug` because URLs can
    /// contain credentials.
    pub fn with_uri(mut self, uri: impl Into<String>) -> Self {
        self.uri = Some(uri.into());
        self
    }

    /// Freezes explicit credentials for Glue metadata operations. S3 FileIO
    /// continues to use [`Self::source_access_policy`].
    pub fn with_catalog_static_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.catalog_credentials = GlueCatalogCredentials::ExplicitStatic(S3StaticCredentials {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
        });
        self
    }

    /// Replaces the source-access policy used for all table data reads.
    pub fn with_source_access_policy(mut self, source_access: SourceAccessPolicy) -> Self {
        self.source_access = source_access;
        self
    }

    pub const fn source_access_policy(&self) -> &SourceAccessPolicy {
        &self.source_access
    }

    #[cfg_attr(not(feature = "catalog"), allow(dead_code))]
    pub(crate) fn catalog_properties(&self) -> HashMap<String, String> {
        let mut properties = HashMap::from([
            ("warehouse".to_string(), self.warehouse.clone()),
            ("region_name".to_string(), self.region.clone()),
        ]);
        if let Some(catalog_id) = self.catalog_id.as_ref().filter(|value| !value.trim().is_empty()) {
            properties.insert("catalog_id".to_string(), catalog_id.clone());
        }
        if let Some(profile) = self.profile.as_ref().filter(|value| !value.trim().is_empty()) {
            properties.insert("profile_name".to_string(), profile.clone());
        }
        if let Some(uri) = self.uri.as_ref().filter(|value| !value.trim().is_empty()) {
            properties.insert("uri".to_string(), uri.clone());
        }
        self.source_access.apply_catalog_properties(&mut properties);
        self.catalog_credentials.apply_catalog_properties(&mut properties);
        properties
    }
}

impl fmt::Debug for GlueIcebergCatalogConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueIcebergCatalogConfig")
            .field("catalog_handle", &self.catalog_handle)
            .field("warehouse", &self.warehouse)
            .field("region", &self.region)
            .field("catalog_id", &self.catalog_id)
            .field("profile", &self.profile)
            .field("uri", &self.uri.as_ref().map(|_| "<redacted>"))
            .field("source_access", &self.source_access)
            .field("catalog_credential_mode", &self.catalog_credentials.mode())
            .finish()
    }
}

/// REST catalog configuration. The REST bearer token and endpoint are hidden
/// from `Debug`; scan candidates receive neither.
#[derive(Clone, PartialEq, Eq)]
pub struct RestIcebergCatalogConfig {
    pub catalog_handle: String,
    uri: String,
    pub warehouse: String,
    pub prefix: Option<String>,
    bearer_token: Option<String>,
    source_access: SourceAccessPolicy,
}

impl RestIcebergCatalogConfig {
    /// Creates REST catalog configuration. A custom S3 endpoint can be added
    /// later through [`Self::with_source_access_policy`].
    pub fn new(catalog_handle: impl Into<String>, uri: impl Into<String>, warehouse: impl Into<String>) -> Self {
        Self {
            catalog_handle: catalog_handle.into(),
            uri: uri.into(),
            warehouse: warehouse.into(),
            prefix: None,
            bearer_token: None,
            source_access: SourceAccessPolicy::s3_default_chain(None),
        }
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Adds a REST bearer token. This is retained only for catalog connection,
    /// is redacted in `Debug`, and is never included in candidates or facts.
    pub fn with_bearer_token(mut self, bearer_token: impl Into<String>) -> Self {
        self.bearer_token = Some(bearer_token.into());
        self
    }

    pub fn with_source_access_policy(mut self, source_access: SourceAccessPolicy) -> Self {
        self.source_access = source_access;
        self
    }

    /// Returns the REST URI for connection code. Treat it as sensitive because
    /// a URI can embed user info.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub const fn source_access_policy(&self) -> &SourceAccessPolicy {
        &self.source_access
    }

    #[cfg_attr(not(feature = "catalog"), allow(dead_code))]
    pub(crate) fn catalog_properties(&self) -> HashMap<String, String> {
        let mut properties =
            HashMap::from([("uri".to_string(), self.uri.clone()), ("warehouse".to_string(), self.warehouse.clone())]);
        if let Some(prefix) = self.prefix.as_ref().filter(|value| !value.trim().is_empty()) {
            properties.insert("prefix".to_string(), prefix.clone());
        }
        if let Some(token) = self.bearer_token.as_ref().filter(|value| !value.trim().is_empty()) {
            properties.insert("token".to_string(), token.clone());
        }
        self.source_access.apply_catalog_properties(&mut properties);
        properties
    }
}

impl fmt::Debug for RestIcebergCatalogConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestIcebergCatalogConfig")
            .field("catalog_handle", &self.catalog_handle)
            .field("uri", &"<redacted>")
            .field("warehouse", &self.warehouse)
            .field("prefix", &self.prefix)
            .field("bearer_token", &self.bearer_token.as_ref().map(|_| "<redacted>"))
            .field("source_access", &self.source_access)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_endpoint_defaults_to_unsigned_and_redacts_debug() {
        let policy = SourceAccessPolicy::s3_custom_endpoint(
            "http://access:secret@localhost:9000",
            Some("us-east-1".to_string()),
            Some(true),
        );
        let s3 = policy.as_s3();

        assert_eq!(s3.credential_mode(), S3CredentialMode::Unsigned);
        assert!(s3.denies_ambient_credentials());
        let debug = format!("{policy:?}");
        assert!(!debug.contains("secret"));
        assert!(!debug.contains("access:secret"));
    }

    #[test]
    fn glue_keeps_catalog_and_source_credentials_separate() {
        let policy =
            SourceAccessPolicy::s3_custom_endpoint("http://localhost:9000", Some("us-east-1".to_string()), Some(true))
                .as_s3()
                .clone()
                .with_static_credentials("local-access", "local-secret", Some("session".to_string()));
        let policy = SourceAccessPolicy::S3(policy);

        let glue = GlueIcebergCatalogConfig::new("glue", "s3://warehouse", "us-east-1")
            .with_profile("glue-profile")
            .with_source_access_policy(policy.clone())
            .with_catalog_static_credentials("glue-access", "glue-secret", Some("glue-session".to_string()));
        let rest = RestIcebergCatalogConfig::new("rest", "http://catalog:8181", "s3://warehouse")
            .with_source_access_policy(policy);

        for properties in [glue.catalog_properties(), rest.catalog_properties()] {
            assert_eq!(properties.get("s3.endpoint"), Some(&"http://localhost:9000".to_string()));
            assert_eq!(properties.get("s3.access-key-id"), Some(&"local-access".to_string()));
            assert_eq!(properties.get("s3.disable-config-load"), Some(&"true".to_string()));
        }
        let glue_properties = glue.catalog_properties();
        assert_eq!(glue_properties.get("aws_access_key_id"), Some(&"glue-access".to_string()));
        assert_eq!(glue_properties.get("aws_secret_access_key"), Some(&"glue-secret".to_string()));
        assert_eq!(glue_properties.get("aws_session_token"), Some(&"glue-session".to_string()));
        assert_eq!(glue_properties.get("region_name"), Some(&"us-east-1".to_string()));
        assert_eq!(glue_properties.get("profile_name"), Some(&"glue-profile".to_string()));
        assert!(!glue_properties.contains_key("aws.access-key-id"));
        assert!(!glue_properties.contains_key("aws.secret-access-key"));
        assert!(!glue_properties.contains_key("aws.session-token"));
        assert!(!glue_properties.contains_key("aws.region"));
        assert!(!glue_properties.contains_key("aws.profile-name"));
        assert_ne!(glue_properties.get("aws_access_key_id"), glue_properties.get("s3.access-key-id"));
        let access = glue.source_access_policy().s3_access();
        assert_eq!(access.credential.mode(), S3CredentialMode::ExplicitStatic);
        assert!(format!("{access:?}").contains("<redacted>"));
    }

    #[test]
    fn unsigned_custom_endpoint_enables_anonymous_file_io_requests() {
        let policy =
            SourceAccessPolicy::s3_custom_endpoint("http://localhost:9000", Some("us-east-1".to_string()), Some(true));
        let rest = RestIcebergCatalogConfig::new("rest", "http://catalog:8181", "s3://warehouse")
            .with_source_access_policy(policy);

        let properties = rest.catalog_properties();

        assert_eq!(properties.get("s3.allow-anonymous"), Some(&"true".to_string()));
        assert_eq!(properties.get("s3.disable-config-load"), Some(&"true".to_string()));
        assert_eq!(properties.get("s3.disable-ec2-metadata"), Some(&"true".to_string()));
    }

    #[test]
    fn file_io_verification_requires_an_exact_s3_policy_match() {
        let policy =
            SourceAccessPolicy::s3_custom_endpoint("http://localhost:9000", Some("us-east-1".to_string()), Some(true));
        let rest = RestIcebergCatalogConfig::new("rest", "http://catalog:8181", "s3://warehouse")
            .with_source_access_policy(policy.clone());
        let matching = rest.catalog_properties();

        assert_eq!(policy.verify_s3_file_io_properties(&matching), S3FileIoConfigVerification::ExactPolicy);
        assert_eq!(policy.verify_s3_file_io_properties(&HashMap::new()), S3FileIoConfigVerification::NoS3Properties);

        let mut mismatched = matching;
        mismatched.insert("s3.region".to_string(), "us-west-2".to_string());
        assert_eq!(policy.verify_s3_file_io_properties(&mismatched), S3FileIoConfigVerification::Mismatch);
    }
}
