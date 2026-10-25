//! Iceberg REST and Glue catalogs as DataFusion catalogs.

use std::{collections::HashMap, fmt, sync::Arc};

use algeon_datafusion::{
    error::{ErrorCode, Result},
    native_exec::S3SourceStorageProfile,
    server::{
        ServerIcebergCatalogConfig, ServerIcebergCatalogKind, ServerIcebergGlueCatalogConfig, ServerIcebergRestAuth,
        ServerIcebergRestCatalogConfig,
    },
};
use async_trait::async_trait;
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider, TableProvider},
    common::{DataFusionError, Result as DataFusionResult},
};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_catalog_glue::{
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_URI, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};
use iceberg_catalog_rest::{
    REST_CATALOG_PROP_AUTH_TYPE, REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder,
};
use iceberg_storage_opendal::OpenDalStorageFactory;

use crate::{error::iceberg_error, storage::file_io_properties, table::IcebergTableProvider};

const CONNECT_OPERATION: &str = "algeon_iceberg.catalog.connect";
const LIST_OPERATION: &str = "algeon_iceberg.catalog.list";

/// Connects the Iceberg catalog `options` describes, an
/// `[[iceberg.catalogs]]` entry of the server configuration.
///
/// Metadata files are read with FileIO properties derived from `storage`.
/// The Glue client is built on the caller's Tokio runtime, which must be
/// current.
pub async fn connect_catalog(
    options: &ServerIcebergCatalogConfig,
    storage: Option<&S3SourceStorageProfile>,
) -> Result<Arc<dyn Catalog>> {
    let mut properties = storage.map(file_io_properties).unwrap_or_default();
    let factory = Arc::new(OpenDalStorageFactory::S3 { customized_credential_load: None });
    let connect_error = |source| {
        iceberg_error(CONNECT_OPERATION, "failed to connect the Iceberg catalog", source)
            .with_fact("catalog", &options.name)
    };
    match &options.kind {
        ServerIcebergCatalogKind::Rest(rest) => {
            properties.extend(rest_properties(rest));
            let catalog = RestCatalogBuilder::default()
                .with_storage_factory(factory)
                .load(options.name.clone(), properties)
                .await
                .map_err(connect_error)?;
            Ok(Arc::new(catalog))
        },
        ServerIcebergCatalogKind::Glue(glue) => {
            properties.extend(glue_properties(glue));
            let catalog = GlueCatalogBuilder::default()
                .with_storage_factory(factory)
                .load(options.name.clone(), properties)
                .await
                .map_err(connect_error)?;
            Ok(Arc::new(catalog))
        },
    }
}

/// REST catalog properties. Vended credentials are never requested: data
/// files are read with the server's storage, not the catalog's.
fn rest_properties(rest: &ServerIcebergRestCatalogConfig) -> HashMap<String, String> {
    let mut properties = HashMap::from([(REST_CATALOG_PROP_URI.to_string(), rest.uri.clone())]);
    let mut set = |key: &str, value: &Option<String>| {
        if let Some(value) = value {
            properties.insert(key.to_string(), value.clone());
        }
    };
    set(REST_CATALOG_PROP_WAREHOUSE, &rest.warehouse);
    set("prefix", &rest.prefix);
    match &rest.auth {
        ServerIcebergRestAuth::None => {
            properties.insert(REST_CATALOG_PROP_AUTH_TYPE.to_string(), "none".to_string());
        },
        ServerIcebergRestAuth::OAuth2 { credential, server_uri, scope } => {
            properties.insert(REST_CATALOG_PROP_AUTH_TYPE.to_string(), "oauth2".to_string());
            properties.insert("credential".to_string(), credential.expose().to_string());
            if let Some(server_uri) = server_uri {
                properties.insert("oauth2-server-uri".to_string(), server_uri.clone());
            }
            if let Some(scope) = scope {
                properties.insert("scope".to_string(), scope.clone());
            }
        },
        ServerIcebergRestAuth::Token(token) => {
            properties.insert(REST_CATALOG_PROP_AUTH_TYPE.to_string(), "oauth2".to_string());
            properties.insert("token".to_string(), token.expose().to_string());
        },
    }
    properties
}

fn glue_properties(glue: &ServerIcebergGlueCatalogConfig) -> HashMap<String, String> {
    let mut properties = HashMap::from([(GLUE_CATALOG_PROP_WAREHOUSE.to_string(), glue.warehouse.clone())]);
    for (key, value) in [
        (GLUE_CATALOG_PROP_CATALOG_ID, &glue.catalog_id),
        ("region_name", &glue.region),
        (GLUE_CATALOG_PROP_URI, &glue.endpoint),
    ] {
        if let Some(value) = value {
            properties.insert(key.to_string(), value.clone());
        }
    }
    properties
}

/// A DataFusion catalog over one Iceberg catalog, with one schema per
/// top-level namespace.
///
/// Namespaces and table names are listed once, when the provider is built;
/// nested namespaces are not listed. [`SchemaProvider::table`] loads the
/// table from the catalog on every call, so a table created later is
/// queryable by name although it is missing from the listing.
#[derive(Debug)]
pub struct IcebergCatalogProvider {
    schemas: HashMap<String, Arc<IcebergSchemaProvider>>,
}

impl IcebergCatalogProvider {
    /// Connects the catalog `options` describes and lists its namespaces
    /// and tables.
    pub async fn connect(
        options: &ServerIcebergCatalogConfig,
        storage: Option<&S3SourceStorageProfile>,
    ) -> Result<Self> {
        let catalog = connect_catalog(options, storage).await?;
        Self::try_new(catalog, storage.cloned()).await.map_err(|error| error.with_fact("catalog", &options.name))
    }

    /// Lists the namespaces and tables of a connected `catalog`. `storage`
    /// is the S3 storage the session reads data files with.
    pub async fn try_new(catalog: Arc<dyn Catalog>, storage: Option<S3SourceStorageProfile>) -> Result<Self> {
        let list_error = |source| iceberg_error(LIST_OPERATION, "failed to list the Iceberg catalog", source);
        let namespaces = catalog.list_namespaces(None).await.map_err(list_error)?;
        let mut schemas = HashMap::new();
        for namespace in namespaces {
            let [name] = namespace.as_ref().as_slice() else {
                continue;
            };
            let tables = catalog.list_tables(&namespace).await.map_err(list_error)?;
            let schema = IcebergSchemaProvider {
                catalog: Arc::clone(&catalog),
                namespace: namespace.clone(),
                table_names: tables.into_iter().map(|table| table.name().to_string()).collect(),
                storage: storage.clone(),
            };
            schemas.insert(name.clone(), Arc::new(schema));
        }
        Ok(Self { schemas })
    }
}

impl CatalogProvider for IcebergCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        let mut names = self.schemas.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).map(|schema| Arc::clone(schema) as Arc<dyn SchemaProvider>)
    }
}

/// One Iceberg namespace as a DataFusion schema.
pub struct IcebergSchemaProvider {
    catalog: Arc<dyn Catalog>,
    namespace: NamespaceIdent,
    table_names: Vec<String>,
    storage: Option<S3SourceStorageProfile>,
}

impl fmt::Debug for IcebergSchemaProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergSchemaProvider")
            .field("namespace", &self.namespace)
            .field("table_names", &self.table_names)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SchemaProvider for IcebergSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.table_names.clone()
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        let ident = TableIdent::new(self.namespace.clone(), name.to_string());
        match IcebergTableProvider::try_new(Arc::clone(&self.catalog), ident, self.storage.clone()).await {
            Ok(provider) => Ok(Some(Arc::new(provider))),
            Err(error) if error.code() == ErrorCode::ObjectNotFound => Ok(None),
            Err(error) => Err(DataFusionError::from(error)),
        }
    }

    fn table_exist(&self, name: &str) -> bool {
        self.table_names.iter().any(|table| table == name)
    }
}
