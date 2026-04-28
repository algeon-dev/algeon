fn env_with_server_toml(toml_body: &str) -> (tempfile::TempDir, BTreeMap<&'static str, String>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, toml_body).expect("write config file");
    let path_str = path.to_string_lossy().into_owned();
    (dir, BTreeMap::from([(NEXUS_SERVER_CONFIG_FILE_ENV, path_str)]))
}

#[derive(Debug)]
struct ReadOnlyCatalog {
    schema_name: String,
    schema: Arc<ReadOnlySchema>,
}

impl ReadOnlyCatalog {
    fn new(
        schema_name: impl Into<String>,
        tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>,
    ) -> Self {
        Self { schema_name: schema_name.into(), schema: Arc::new(ReadOnlySchema::new(tables)) }
    }
}

impl datafusion::catalog::CatalogProvider for ReadOnlyCatalog {
    fn schema_names(&self) -> Vec<String> {
        vec![self.schema_name.clone()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn datafusion::catalog::SchemaProvider>> {
        (name == self.schema_name).then(|| Arc::clone(&self.schema) as Arc<dyn datafusion::catalog::SchemaProvider>)
    }
}

#[derive(Debug)]
struct ReadOnlySchema {
    tables: BTreeMap<String, Arc<dyn datafusion::datasource::TableProvider>>,
}

impl ReadOnlySchema {
    fn new(tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>) -> Self {
        Self { tables: tables.into_iter().map(|(name, table)| (name.to_string(), table)).collect() }
    }
}

#[async_trait::async_trait]
impl datafusion::catalog::SchemaProvider for ReadOnlySchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(
        &self,
        name: &str,
    ) -> datafusion::common::Result<Option<Arc<dyn datafusion::datasource::TableProvider>>> {
        Ok(self.tables.get(name).cloned())
    }

    async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<datafusion::datasource::TableType>> {
        Ok(self.tables.get(name).map(|table| table.table_type()))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

fn edge_table() -> Arc<dyn datafusion::datasource::TableProvider> {
    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::datasource::MemTable;

    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![0, 1])), Arc::new(Int64Array::from(vec![1, 2]))],
    )
    .unwrap();
    Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
}

fn batch_i64_value(batch: &arrow::record_batch::RecordBatch, column: usize) -> i64 {
    batch.column(column).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0)
}

#[cfg(feature = "iceberg")]
fn expect_glue_backend(
    iceberg: &datafusion_nexus::server::IcebergServerConfig,
) -> &datafusion_nexus::server::GlueServerBackendConfig {
    match &iceberg.backend {
        datafusion_nexus::server::IcebergCatalogBackendConfig::Glue(glue) => glue,
        other => panic!("expected Glue backend, got {other:?}"),
    }
}

#[cfg(feature = "iceberg")]
fn expect_rest_backend(
    iceberg: &datafusion_nexus::server::IcebergServerConfig,
) -> &datafusion_nexus::table_format::iceberg::catalog::RestIcebergCatalogConfig {
    match &iceberg.backend {
        datafusion_nexus::server::IcebergCatalogBackendConfig::Rest(rest) => rest,
        other => panic!("expected REST backend, got {other:?}"),
    }
}
