use std::{collections::BTreeSet, fmt};

use super::{
    ConfigSources, Result, S3SourceStorageProfile,
    file::{FileSecret, ServerIcebergCatalogFileConfig, ServerIcebergFileConfig, ServerIcebergRestAuthFileConfig},
};

/// DataFusion's default catalog name, which a server session always holds.
const DEFAULT_DATAFUSION_CATALOG: &str = "datafusion";

/// One `[[iceberg.catalogs]]` entry: an Iceberg catalog registered as a
/// DataFusion catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerIcebergCatalogConfig {
    /// DataFusion catalog name. Loading rejects an empty name, a name used
    /// by another entry, and `datafusion`.
    pub name: String,
    /// Catalog service and its connection settings.
    pub kind: ServerIcebergCatalogKind,
}

/// The catalog service of a [`ServerIcebergCatalogConfig`], from its `kind` key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerIcebergCatalogKind {
    /// `kind = "rest"`: an Iceberg REST catalog.
    Rest(ServerIcebergRestCatalogConfig),
    /// `kind = "glue"`: the AWS Glue Data Catalog.
    Glue(ServerIcebergGlueCatalogConfig),
}

/// Settings of an Iceberg REST catalog entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerIcebergRestCatalogConfig {
    /// Catalog base URI, such as `http://127.0.0.1:8181`.
    pub uri: String,
    /// Warehouse identifier sent to the catalog's config endpoint.
    pub warehouse: Option<String>,
    /// Path prefix inserted after the catalog URI's version segment.
    pub prefix: Option<String>,
    /// Authentication against the catalog service.
    pub auth: ServerIcebergRestAuth,
}

/// Authentication of an Iceberg REST catalog entry, from its `auth` table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ServerIcebergRestAuth {
    /// `kind = "none"`, the default: requests carry no authorization.
    #[default]
    None,
    /// `kind = "oauth2"`: client credentials exchanged for a token.
    OAuth2 {
        /// `client_id:client_secret`, or a client secret alone.
        credential: ServerIcebergSecret,
        /// Token endpoint. Unset uses the catalog's own token endpoint.
        server_uri: Option<String>,
        /// OAuth2 scope requested with the token.
        scope: Option<String>,
    },
    /// `kind = "token"`: a bearer token sent with every request.
    Token(ServerIcebergSecret),
}

/// Settings of an AWS Glue catalog entry. Credentials come from the default
/// AWS credential chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerIcebergGlueCatalogConfig {
    /// Warehouse location, such as `s3://bucket/warehouse`.
    pub warehouse: String,
    /// AWS account ID of the Glue catalog. Unset uses the caller's account.
    pub catalog_id: Option<String>,
    /// Region of the Glue service. Unset follows the AWS environment.
    pub region: Option<String>,
    /// Glue service endpoint override. Loading requires `[s3]` to name its
    /// endpoint too: iceberg-rust's Glue catalog reads table metadata through
    /// this endpoint when no S3 endpoint is set.
    pub endpoint: Option<String>,
}

/// A catalog secret from the config file. `Debug` prints `<redacted>`, and
/// the type has no `Display`; [`Self::expose`] returns the value.
#[derive(Clone, PartialEq, Eq)]
pub struct ServerIcebergSecret(String);

impl ServerIcebergSecret {
    /// Wraps `secret`.
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// The secret value, for the catalog client that sends it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ServerIcebergSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl From<FileSecret> for ServerIcebergSecret {
    fn from(secret: FileSecret) -> Self {
        Self(secret.0)
    }
}

/// Validates the `[[iceberg.catalogs]]` entries. Iceberg data files are read
/// through the `[s3]` storage, so any entry requires `[s3]`.
pub(super) fn iceberg_catalogs(
    file: ServerIcebergFileConfig,
    s3: Option<&S3SourceStorageProfile>,
    sources: &ConfigSources,
) -> Result<Vec<ServerIcebergCatalogConfig>> {
    if !file.catalogs.is_empty() && s3.is_none() {
        return Err(sources
            .invalid(
                "iceberg.catalogs",
                "needs an `[s3]` table, which the server reads Iceberg data files through; add `[s3]`",
            )
            .with_fact("reason", "iceberg_catalogs_require_s3"));
    }
    let mut names = BTreeSet::new();
    let mut catalogs = Vec::with_capacity(file.catalogs.len());
    for entry in file.catalogs {
        let catalog = catalog_config(entry);
        if catalog.name.is_empty() {
            return Err(sources.invalid("iceberg.catalogs.name", "must not be empty"));
        }
        if catalog.name == DEFAULT_DATAFUSION_CATALOG {
            return Err(sources
                .invalid(
                    "iceberg.catalogs.name",
                    format!("`{DEFAULT_DATAFUSION_CATALOG}` is the default DataFusion catalog; choose another name"),
                )
                .with_fact("reason", "iceberg_catalog_name_reserved"));
        }
        if !names.insert(catalog.name.clone()) {
            return Err(sources
                .invalid(
                    "iceberg.catalogs.name",
                    format!("`{}` names two entries; give each catalog its own name", catalog.name),
                )
                .with_fact("reason", "duplicate_iceberg_catalog"));
        }
        // iceberg-rust's Glue catalog copies its endpoint into the table's
        // FileIO `s3.endpoint` when that property is unset, which sends
        // metadata reads to the Glue host and fails the scan's storage check.
        if let ServerIcebergCatalogKind::Glue(glue) = &catalog.kind
            && glue.endpoint.is_some()
            && s3.and_then(|storage| storage.endpoint.as_deref()).is_none_or(|endpoint| endpoint.trim().is_empty())
        {
            return Err(sources
                .invalid(
                    "iceberg.catalogs.endpoint",
                    format!(
                        "is set for Glue catalog `{}` while `[s3]` names no `endpoint`; set `s3.endpoint`, for example to the regional S3 endpoint, or remove the Glue endpoint",
                        catalog.name
                    ),
                )
                .with_fact("reason", "iceberg_glue_endpoint_requires_s3_endpoint"));
        }
        catalogs.push(catalog);
    }
    Ok(catalogs)
}

fn catalog_config(entry: ServerIcebergCatalogFileConfig) -> ServerIcebergCatalogConfig {
    match entry {
        ServerIcebergCatalogFileConfig::Rest { name, uri, warehouse, prefix, auth } => {
            let auth = match auth {
                ServerIcebergRestAuthFileConfig::None {} => ServerIcebergRestAuth::None,
                ServerIcebergRestAuthFileConfig::OAuth2 { credential, server_uri, scope } => {
                    ServerIcebergRestAuth::OAuth2 { credential: credential.into(), server_uri, scope }
                },
                ServerIcebergRestAuthFileConfig::Token { token } => ServerIcebergRestAuth::Token(token.into()),
            };
            ServerIcebergCatalogConfig {
                name,
                kind: ServerIcebergCatalogKind::Rest(ServerIcebergRestCatalogConfig { uri, warehouse, prefix, auth }),
            }
        },
        ServerIcebergCatalogFileConfig::Glue { name, warehouse, catalog_id, region, endpoint } => {
            ServerIcebergCatalogConfig {
                name,
                kind: ServerIcebergCatalogKind::Glue(ServerIcebergGlueCatalogConfig {
                    warehouse,
                    catalog_id,
                    region,
                    endpoint,
                }),
            }
        },
    }
}
