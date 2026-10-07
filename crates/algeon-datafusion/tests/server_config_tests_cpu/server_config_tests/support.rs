use super::*;

pub(super) fn env_with_server_toml(toml_body: &str) -> (tempfile::TempDir, BTreeMap<&'static str, String>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    let toml = if toml_body.contains("[[admission.device_profiles]]") {
        toml_body.to_owned()
    } else {
        format!("{toml_body}\n\n[[admission.device_profiles]]\ndevice_ordinal = 0\n")
    };
    std::fs::write(&path, toml).expect("write config file");
    let path_str = path.to_string_lossy().into_owned();
    (dir, BTreeMap::from([(ALGEON_SERVER_CONFIG_FILE_ENV, path_str)]))
}

pub(super) fn env_with_default_device_profile() -> (tempfile::TempDir, BTreeMap<&'static str, String>) {
    env_with_server_toml("")
}

pub(super) fn server_config_with_default_device_profile(
    mut env: BTreeMap<&'static str, String>,
) -> ServerResult<ServerConfig> {
    let (_dir, profile_env) = env_with_default_device_profile();
    env.extend(profile_env);
    ServerConfig::from_env_get(|name| env.get(name).cloned())
}

#[derive(Debug)]
pub(super) struct ReadOnlyCatalog {
    pub(super) schema_name: String,
    pub(super) schema: Arc<ReadOnlySchema>,
}

impl ReadOnlyCatalog {
    pub(super) fn new(
        schema_name: impl Into<String>,
        tables: impl IntoIterator<Item = (&'static str, Arc<dyn TableProvider>)>,
    ) -> Self {
        Self { schema_name: schema_name.into(), schema: Arc::new(ReadOnlySchema::new(tables)) }
    }
}

impl CatalogProvider for ReadOnlyCatalog {
    fn schema_names(&self) -> Vec<String> {
        vec![self.schema_name.clone()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        (name == self.schema_name).then(|| Arc::clone(&self.schema) as Arc<dyn SchemaProvider>)
    }
}

#[derive(Debug)]
pub(super) struct ReadOnlySchema {
    pub(super) tables: BTreeMap<String, Arc<dyn TableProvider>>,
}

impl ReadOnlySchema {
    pub(super) fn new(tables: impl IntoIterator<Item = (&'static str, Arc<dyn TableProvider>)>) -> Self {
        Self { tables: tables.into_iter().map(|(name, table)| (name.to_string(), table)).collect() }
    }
}

#[async_trait::async_trait]
impl SchemaProvider for ReadOnlySchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.get(name).cloned())
    }

    async fn table_type(&self, name: &str) -> DataFusionResult<Option<TableType>> {
        Ok(self.tables.get(name).map(|table| table.table_type()))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

pub(super) fn edge_table() -> Arc<dyn TableProvider> {
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

pub(super) fn batch_i64_value(batch: &RecordBatch, column: usize) -> i64 {
    batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap().value(0)
}
