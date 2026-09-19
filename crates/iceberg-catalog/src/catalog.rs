use std::{collections::HashMap, fmt, sync::Arc};

use iceberg::{
    Catalog, CatalogBuilder, NamespaceIdent, TableIdent,
    io::{Storage, StorageConfig, StorageFactory},
};
#[cfg(feature = "glue")]
use iceberg_storage_opendal::OpenDalStorageFactory;
#[cfg(feature = "glue")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CatalogErrorKind, Error, ErrorContext, GlueCatalogConfig, Operation, ResolvedTable, RestCatalogConfig, Result,
    SourceAccessPolicy, TableIdentifier,
};

/// Catalog implementation kind, independent from a DataFusion catalog name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogKind {
    Glue,
    Rest,
}

impl CatalogKind {
    const fn error_kind(self) -> CatalogErrorKind {
        match self {
            Self::Glue => CatalogErrorKind::Glue,
            Self::Rest => CatalogErrorKind::Rest,
        }
    }
}

/// Catalog namespace identity without upstream catalog types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Namespace {
    pub parts: Vec<String>,
}

/// Backend-neutral connected Iceberg catalog.
///
/// It applies one frozen source-access policy to its Iceberg catalog/FileIO
/// properties. The exact same policy is then placed in each
/// [`ResolvedTable`].
pub struct CatalogBackend {
    kind: CatalogKind,
    catalog_handle: String,
    catalog: Arc<dyn Catalog>,
    source_access: SourceAccessPolicy,
    source_access_is_enforced_by_catalog_file_io: bool,
}

impl CatalogBackend {
    fn new(
        kind: CatalogKind,
        catalog_handle: String,
        catalog: Arc<dyn Catalog>,
        source_access: SourceAccessPolicy,
        source_access_is_enforced_by_catalog_file_io: bool,
    ) -> Self {
        Self { kind, catalog_handle, catalog, source_access, source_access_is_enforced_by_catalog_file_io }
    }

    pub const fn kind(&self) -> CatalogKind {
        self.kind
    }

    /// Lists catalog namespaces without exposing upstream catalog types.
    pub async fn list_namespaces(&self) -> Result<Vec<Namespace>> {
        let namespaces = self
            .catalog
            .list_namespaces(None)
            .await
            .map_err(|source| self.normalize(Operation::CatalogListNamespaces, source))?;
        Ok(namespaces.into_iter().map(|namespace| Namespace { parts: namespace.inner() }).collect())
    }

    /// Lists table identities in a namespace.
    pub async fn list_tables(&self, namespace: Namespace) -> Result<Vec<TableIdentifier>> {
        let namespace = NamespaceIdent::from_vec(namespace.parts)
            .map_err(|source| self.normalize(Operation::CatalogListTables, source))?;
        let tables = self
            .catalog
            .list_tables(&namespace)
            .await
            .map_err(|source| self.normalize(Operation::CatalogListTables, source))?;
        Ok(tables
            .into_iter()
            .map(|table| TableIdentifier {
                namespace: table.namespace().as_ref().clone(),
                name: table.name().to_string(),
            })
            .collect())
    }

    /// Loads a table and freezes its snapshot/schema/source-access tuple.
    pub async fn load_table(&self, table: TableIdentifier) -> Result<ResolvedTable> {
        let table_display = table.display_name();
        let namespace = NamespaceIdent::from_vec(table.namespace).map_err(|source| {
            self.normalize(Operation::TableLoad, source)
                .with_context(ErrorContext { table_identifier: Some(table_display.clone()), ..self.context() })
        })?;
        let table = TableIdent::new(namespace, table.name);
        let loaded = self.catalog.load_table(&table).await.map_err(|source| {
            self.normalize(Operation::TableLoad, source)
                .with_context(ErrorContext { table_identifier: Some(table_display), ..self.context() })
        })?;
        if self.source_access_is_enforced_by_catalog_file_io {
            ResolvedTable::from_catalog_file_io(loaded, self.source_access.clone())
        } else {
            ResolvedTable::from_table(loaded, self.source_access.clone())
        }
    }

    fn context(&self) -> ErrorContext {
        ErrorContext {
            catalog_kind: Some(self.kind.error_kind()),
            catalog_handle: Some(self.catalog_handle.clone()),
            ..ErrorContext::default()
        }
    }

    fn normalize(&self, operation: Operation, source: iceberg::Error) -> Error {
        Error::from_iceberg_error(operation, source).with_context(self.context())
    }
}

impl fmt::Debug for CatalogBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogBackend")
            .field("kind", &self.kind)
            .field("catalog_handle", &self.catalog_handle)
            .field("catalog", &"<connected Iceberg catalog>")
            .field("source_access", &self.source_access)
            .field("source_access_is_enforced_by_catalog_file_io", &self.source_access_is_enforced_by_catalog_file_io)
            .finish()
    }
}

/// Connected AWS Glue catalog backend.
#[cfg(feature = "glue")]
pub struct GlueCatalogBackend(CatalogBackend);

#[cfg(feature = "glue")]
impl GlueCatalogBackend {
    /// Connects Glue with source/FileIO properties derived from the frozen
    /// source-access policy.
    pub async fn connect(config: GlueCatalogConfig) -> Result<Self> {
        use iceberg_catalog_glue::GlueCatalogBuilder;

        let catalog_handle = config.catalog_handle.clone();
        let source_access = config.source_access_policy().clone();
        let glue = GlueCatalogBuilder::default()
            .with_storage_factory(Arc::new(GlueSourceStorageFactory::new(&source_access)))
            .load(catalog_handle.clone(), config.catalog_properties())
            .await
            .map_err(|source| {
                Error::from_iceberg_error(Operation::CatalogConnect, source).with_context(ErrorContext {
                    catalog_kind: Some(CatalogErrorKind::Glue),
                    catalog_handle: Some(catalog_handle.clone()),
                    ..ErrorContext::default()
                })
            })?;
        Ok(Self(CatalogBackend::new(CatalogKind::Glue, catalog_handle, Arc::new(glue), source_access, true)))
    }

    pub const fn backend(&self) -> &CatalogBackend {
        &self.0
    }

    pub fn into_backend(self) -> CatalogBackend {
        self.0
    }
}

#[cfg(feature = "glue")]
impl fmt::Debug for GlueCatalogBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("GlueCatalogBackend").field(&self.0).finish()
    }
}

/// Connected Iceberg REST catalog backend.
#[cfg(feature = "rest")]
pub struct RestCatalogBackend(CatalogBackend);

#[cfg(feature = "rest")]
impl RestCatalogBackend {
    /// Connects a REST catalog with an S3 OpenDAL FileIO factory. The catalog
    /// and FileIO properties come from the same frozen source-access policy.
    pub async fn connect(config: RestCatalogConfig) -> Result<Self> {
        use iceberg_catalog_rest::RestCatalogBuilder;
        use iceberg_storage_opendal::OpenDalStorageFactory;

        let catalog_handle = config.catalog_handle.clone();
        let source_access = config.source_access_policy().clone();
        let storage_factory = Arc::new(OpenDalStorageFactory::S3 { customized_credential_load: None });
        let rest = RestCatalogBuilder::default()
            .with_storage_factory(storage_factory)
            .load(catalog_handle.clone(), config.catalog_properties())
            .await
            .map_err(|source| {
                Error::from_iceberg_error(Operation::CatalogConnect, source).with_context(ErrorContext {
                    catalog_kind: Some(CatalogErrorKind::Rest),
                    catalog_handle: Some(catalog_handle.clone()),
                    ..ErrorContext::default()
                })
            })?;
        Ok(Self(CatalogBackend::new(CatalogKind::Rest, catalog_handle, Arc::new(rest), source_access, false)))
    }

    pub const fn backend(&self) -> &CatalogBackend {
        &self.0
    }

    pub fn into_backend(self) -> CatalogBackend {
        self.0
    }
}

#[cfg(feature = "rest")]
impl fmt::Debug for RestCatalogBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RestCatalogBackend").field(&self.0).finish()
    }
}

/// Glue passes metadata credentials to its FileIO builder as a convenience.
/// This factory discards that mixed input and builds the actual FileIO from the
/// independently frozen source policy only.
#[cfg(feature = "glue")]
#[derive(Clone)]
struct GlueSourceStorageFactory {
    source_properties: HashMap<String, String>,
}

#[cfg(feature = "glue")]
impl GlueSourceStorageFactory {
    fn new(source_access: &SourceAccessPolicy) -> Self {
        Self { source_properties: source_access.file_io_properties() }
    }
}

#[cfg(feature = "glue")]
impl fmt::Debug for GlueSourceStorageFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueSourceStorageFactory").field("source_properties", &"<redacted>").finish()
    }
}

// `StorageFactory` uses typetag serialization, but this factory freezes source
// credentials specifically to keep them out of Glue's metadata FileIO map.
// It is process-local: serializing it would export those credentials, while a
// redacted round trip could not reconstruct a working source policy.
#[cfg(feature = "glue")]
impl Serialize for GlueSourceStorageFactory {
    fn serialize<S>(&self, _serializer: S) -> core::result::Result<S::Ok, S::Error>
    where S: Serializer {
        Err(serde::ser::Error::custom("Algeon Glue source storage factories are process-local"))
    }
}

#[cfg(feature = "glue")]
impl<'de> Deserialize<'de> for GlueSourceStorageFactory {
    fn deserialize<D>(_deserializer: D) -> core::result::Result<Self, D::Error>
    where D: Deserializer<'de> {
        Err(serde::de::Error::custom("Algeon Glue source storage factories are process-local"))
    }
}

#[cfg(feature = "glue")]
#[typetag::serde(name = "AlgeonGlueSourceStorageFactory")]
impl StorageFactory for GlueSourceStorageFactory {
    fn build(&self, _catalog_file_io_config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        let source_config = StorageConfig::from_props(self.source_properties.clone());
        OpenDalStorageFactory::S3 { customized_credential_load: None }.build(&source_config)
    }
}

#[cfg(all(test, feature = "glue"))]
mod tests {
    use super::*;

    fn assert_catalog_credentials_do_not_reach_file_io(source_access: SourceAccessPolicy) {
        let factory = GlueSourceStorageFactory::new(&source_access);
        let source_properties = &factory.source_properties;

        assert_eq!(
            source_access.verify_s3_file_io_properties(source_properties),
            crate::S3FileIoConfigVerification::ExactPolicy
        );
        assert!(source_properties.keys().all(|key| !key.starts_with("aws_")));
        assert!(
            !source_properties.contains_key("s3.session-token")
                || source_properties["s3.session-token"] == "source-session"
        );
    }

    #[test]
    fn glue_file_io_uses_only_the_frozen_source_policy() {
        assert_catalog_credentials_do_not_reach_file_io(SourceAccessPolicy::s3_custom_endpoint(
            "http://localhost:9000",
            Some("us-east-1".to_string()),
            Some(true),
        ));
        assert_catalog_credentials_do_not_reach_file_io(SourceAccessPolicy::s3_default_chain(Some(
            "us-east-1".to_string(),
        )));
        let static_source = SourceAccessPolicy::S3(
            SourceAccessPolicy::s3_custom_endpoint("http://localhost:9000", Some("us-east-1".to_string()), Some(true))
                .as_s3()
                .clone()
                .with_static_credentials("source-access", "source-secret", None),
        );
        assert_catalog_credentials_do_not_reach_file_io(static_source);
    }

    #[test]
    fn glue_file_io_factory_refuses_serialization_of_frozen_source_credentials() {
        let source_access = SourceAccessPolicy::S3(
            SourceAccessPolicy::s3_custom_endpoint("http://localhost:9000", Some("us-east-1".to_string()), Some(true))
                .as_s3()
                .clone()
                .with_static_credentials("source-access", "source-secret", Some("source-session".to_string())),
        );
        let error = serde_json::to_string(&GlueSourceStorageFactory::new(&source_access))
            .expect_err("a credential-bearing storage factory must remain process-local");

        assert!(error.to_string().contains("process-local"));
        assert!(!error.to_string().contains("source-secret"));
    }
}
