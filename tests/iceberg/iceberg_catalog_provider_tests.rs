//! Tests for the dynamic Iceberg catalog/schema provider (Milestone 2).
//!
//! All tests in this file are credential-free and use a `MockIcebergCatalogBackend`.

#![cfg(feature = "iceberg")]

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::array::Array;
use async_trait::async_trait;
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::table_format::iceberg::catalog::{
    CudfIcebergCatalogProvider, IcebergCatalogBackend, IcebergProviderKind, build_catalog_provider_from_backend,
    build_scoped_catalog_provider_from_backend,
};
use iceberg::{NamespaceIdent, TableIdent};

// ---------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct MockIcebergCatalogBackend {
    namespaces: Vec<NamespaceIdent>,
    tables_by_namespace: HashMap<String, Vec<String>>,
}

impl MockIcebergCatalogBackend {
    fn new(namespaces: Vec<NamespaceIdent>, tables_by_namespace: HashMap<String, Vec<String>>) -> Self {
        Self { namespaces, tables_by_namespace }
    }

    fn namespace_key(ns: &NamespaceIdent) -> String {
        ns.as_ref().join(".")
    }
}

#[async_trait]
impl IcebergCatalogBackend for MockIcebergCatalogBackend {
    async fn list_namespaces(&self) -> datafusion_nexus::error::Result<Vec<NamespaceIdent>> {
        Ok(self.namespaces.clone())
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> datafusion_nexus::error::Result<Vec<TableIdent>> {
        let key = Self::namespace_key(namespace);
        let names = self.tables_by_namespace.get(&key).cloned().unwrap_or_default();
        let idents = names.into_iter().map(|name| TableIdent::new(namespace.clone(), name)).collect();
        Ok(idents)
    }

    async fn load_table(&self, _table: &TableIdent) -> datafusion_nexus::error::Result<iceberg::table::Table> {
        Err(datafusion_nexus::error::Error::new(
            datafusion_nexus::error::ErrorCode::CudfDependencyFailure,
            "mock_backend.load_table",
            "MockIcebergCatalogBackend does not support loading tables",
        ))
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn mock_backend_with_tpch() -> MockIcebergCatalogBackend {
    MockIcebergCatalogBackend::new(
        vec![NamespaceIdent::new("tpch_sf1".to_string())],
        HashMap::from([(
            "tpch_sf1".to_string(),
            vec!["tpch_sf1_nation".to_string(), "tpch_sf1_lineitem".to_string(), "tpch_sf1_orders".to_string()],
        )]),
    )
}

// ---------------------------------------------------------------------------
// schema_names / table_names / table_exist
// ---------------------------------------------------------------------------

#[tokio::test]
async fn catalog_provider_schema_names_from_namespace_list() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    let names = provider.schema_names();
    assert_eq!(names, vec!["tpch_sf1"]);
}

#[tokio::test]
async fn catalog_provider_schema_lookup() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    assert!(provider.schema("tpch_sf1").is_some());
    assert!(provider.schema("nonexistent_db").is_none());
}

#[tokio::test]
async fn schema_provider_table_names_from_table_list() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let schema = provider.schema("tpch_sf1").unwrap();

    let mut names = schema.table_names();
    names.sort();
    assert_eq!(names, vec!["tpch_sf1_lineitem", "tpch_sf1_nation", "tpch_sf1_orders"]);
}

#[tokio::test]
async fn schema_provider_table_exist_matches_discovered_names() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let schema = provider.schema("tpch_sf1").unwrap();

    assert!(schema.table_exist("tpch_sf1_nation"));
    assert!(schema.table_exist("tpch_sf1_lineitem"));
    assert!(!schema.table_exist("nonexistent_table"));
    // Case-sensitive: must match exactly.
    assert!(!schema.table_exist("TPCH_SF1_NATION"));
}

#[tokio::test]
async fn schema_provider_table_returns_none_for_unknown_name() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let schema = provider.schema("tpch_sf1").unwrap();

    let result = schema.table("ghost_table").await.unwrap();
    assert!(result.is_none(), "unknown table should return None, not error");
}

// ---------------------------------------------------------------------------
// Empty namespace list
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_namespace_list_yields_empty_catalog() {
    let backend = Arc::new(MockIcebergCatalogBackend::new(vec![], HashMap::new()));
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    assert!(provider.schema_names().is_empty());
    assert!(provider.schema("anything").is_none());
}

// ---------------------------------------------------------------------------
// Multi-namespace catalog
// ---------------------------------------------------------------------------

#[tokio::test]
async fn catalog_provider_multi_namespace() {
    let backend = Arc::new(MockIcebergCatalogBackend::new(
        vec![NamespaceIdent::new("db_a".to_string()), NamespaceIdent::new("db_b".to_string())],
        HashMap::from([
            ("db_a".to_string(), vec!["orders".to_string()]),
            ("db_b".to_string(), vec!["products".to_string(), "inventory".to_string()]),
        ]),
    ));
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    let mut schema_names = provider.schema_names();
    schema_names.sort();
    assert_eq!(schema_names, vec!["db_a", "db_b"]);

    assert_eq!(provider.schema("db_a").unwrap().table_names(), vec!["orders"]);
    let mut db_b = provider.schema("db_b").unwrap().table_names();
    db_b.sort();
    assert_eq!(db_b, vec!["inventory", "products"]);
}

#[tokio::test]
async fn scoped_catalog_provider_exposes_only_requested_namespace() {
    let backend = Arc::new(MockIcebergCatalogBackend::new(
        vec![NamespaceIdent::new("db_a".to_string()), NamespaceIdent::new("db_b".to_string())],
        HashMap::from([
            ("db_a".to_string(), vec!["orders".to_string()]),
            ("db_b".to_string(), vec!["products".to_string(), "inventory".to_string()]),
        ]),
    ));
    let provider =
        build_scoped_catalog_provider_from_backend(backend, "glue", "db_b", IcebergProviderKind::Gpu).await.unwrap();

    assert_eq!(provider.schema_names(), vec!["db_b"]);
    assert!(provider.schema("db_a").is_none());

    let mut db_b = provider.schema("db_b").unwrap().table_names();
    db_b.sort();
    assert_eq!(db_b, vec!["inventory", "products"]);
}

#[tokio::test]
async fn scoped_catalog_provider_does_not_enumerate_global_namespace_list() {
    let backend = Arc::new(MockIcebergCatalogBackend::new(
        vec![NamespaceIdent::new("unrelated".to_string())],
        HashMap::from([("tpch_sf1".to_string(), vec!["tpch_sf1_nation".to_string()])]),
    ));
    let provider = build_scoped_catalog_provider_from_backend(backend, "glue", "tpch_sf1", IcebergProviderKind::Gpu)
        .await
        .unwrap();

    assert_eq!(provider.schema_names(), vec!["tpch_sf1"]);
    assert_eq!(provider.schema("tpch_sf1").unwrap().table_names(), vec!["tpch_sf1_nation"]);
    assert!(provider.schema("unrelated").is_none());
}

// ---------------------------------------------------------------------------
// Alias seeding
// ---------------------------------------------------------------------------

#[tokio::test]
async fn seeded_alias_resolves_through_schema_provider() {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;

    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    // Seed a MemTable under the alias "nation".
    let schema = Arc::new(Schema::new(vec![Field::new("n_nationkey", DataType::Int32, false)]));
    let mem_table = Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap());
    let schema_provider = provider.schema_concrete("tpch_sf1").unwrap();
    schema_provider.seed_table("nation".to_string(), mem_table).await;

    // The alias should be resolvable via the schema provider.
    let resolved = provider.schema("tpch_sf1").unwrap().table("nation").await.unwrap();
    assert!(resolved.is_some(), "seeded alias 'nation' should resolve");

    // The real table name should also be discoverable (backend would load it).
    assert!(provider.schema("tpch_sf1").unwrap().table_exist("tpch_sf1_nation"));
}

#[tokio::test]
async fn seeding_same_alias_twice_keeps_first() {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;

    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let schema_provider = provider.schema_concrete("tpch_sf1").unwrap();

    let make_table = |tag: &str| {
        // Embed the tag in a field name so we can distinguish the two providers.
        let schema = Arc::new(Schema::new(vec![Field::new(tag, DataType::Int32, false)]));
        Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap()) as Arc<dyn datafusion::datasource::TableProvider>
    };

    let first = make_table("first_col");
    let second = make_table("second_col");

    schema_provider.seed_table("shared_name".to_string(), Arc::clone(&first)).await;
    schema_provider.seed_table("shared_name".to_string(), Arc::clone(&second)).await;

    let resolved = schema_provider.table("shared_name").await.unwrap().unwrap();
    // First write wins; the schema should have "first_col".
    assert!(resolved.schema().field_with_name("first_col").is_ok(), "first seed should win");
}

// ---------------------------------------------------------------------------
// Information-schema SQL (requires information_schema feature)
// ---------------------------------------------------------------------------

async fn session_with_catalog(catalog_name: &str, provider: CudfIcebergCatalogProvider) -> SessionContext {
    let cfg = SessionConfig::new().with_information_schema(true);
    let ctx = SessionContext::new_with_config(cfg);
    ctx.register_catalog(catalog_name, Arc::new(provider));
    ctx
}

#[tokio::test]
async fn information_schema_schemata_lists_namespaces() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let ctx = session_with_catalog("glue", provider).await;

    let sql = "SELECT schema_name FROM information_schema.schemata WHERE catalog_name = 'glue'";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let schema_names: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b.column(0);
            let arr = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            (0..arr.len()).map(|i| arr.value(i).to_string()).collect::<Vec<_>>()
        })
        .collect();

    assert!(schema_names.contains(&"tpch_sf1".to_string()), "tpch_sf1 must appear in schemata");
}

#[tokio::test]
async fn information_schema_tables_lists_discovered_tables() {
    let backend = Arc::new(mock_backend_with_tpch());
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let ctx = session_with_catalog("glue", provider).await;

    let sql = "SELECT table_name FROM information_schema.tables \
               WHERE table_catalog = 'glue' AND table_schema = 'tpch_sf1' \
               ORDER BY table_name";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let table_names: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b.column(0);
            let arr = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            (0..arr.len()).map(|i| arr.value(i).to_string()).collect::<Vec<_>>()
        })
        .collect();

    assert!(table_names.contains(&"tpch_sf1_nation".to_string()));
    assert!(table_names.contains(&"tpch_sf1_lineitem".to_string()));
    assert!(table_names.contains(&"tpch_sf1_orders".to_string()));
}

#[tokio::test]
async fn information_schema_tables_does_not_load_iceberg_table_metadata() {
    let (counting_backend, load_count) = CountingMockBackend::new(mock_backend_with_tpch());
    let backend = Arc::new(counting_backend);
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();
    let ctx = session_with_catalog("glue", provider).await;

    let sql = "SELECT table_name FROM information_schema.tables \
               WHERE table_catalog = 'glue' AND table_schema = 'tpch_sf1'";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();

    assert_eq!(row_count, 3);
    assert_eq!(load_count.load(Ordering::SeqCst), 0, "information_schema.tables must use table_type()");
}

// ---------------------------------------------------------------------------
// Multi-level namespace: join with "."
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_level_namespace_joined_as_schema_name() {
    let ns = NamespaceIdent::from_strs(["a", "b"]).unwrap();
    let backend = Arc::new(MockIcebergCatalogBackend::new(
        vec![ns],
        HashMap::from([("a.b".to_string(), vec!["t1".to_string()])]),
    ));
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    assert_eq!(provider.schema_names(), vec!["a.b"]);
    let schema = provider.schema("a.b").unwrap();
    assert_eq!(schema.table_names(), vec!["t1"]);
}

#[tokio::test]
async fn scoped_catalog_provider_accepts_multi_level_namespace_scope() {
    let backend =
        Arc::new(MockIcebergCatalogBackend::new(vec![], HashMap::from([("a.b".to_string(), vec!["t1".to_string()])])));
    let provider =
        build_scoped_catalog_provider_from_backend(backend, "glue", "a.b", IcebergProviderKind::Gpu).await.unwrap();

    assert_eq!(provider.schema_names(), vec!["a.b"]);
    assert_eq!(provider.schema("a.b").unwrap().table_names(), vec!["t1"]);
}

// ---------------------------------------------------------------------------
// Namespace with no tables
// ---------------------------------------------------------------------------

#[tokio::test]
async fn namespace_with_no_tables_is_still_discoverable() {
    let backend = Arc::new(MockIcebergCatalogBackend::new(
        vec![NamespaceIdent::new("empty_db".to_string())],
        HashMap::from([("empty_db".to_string(), vec![])]),
    ));
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    assert_eq!(provider.schema_names(), vec!["empty_db"]);
    let schema = provider.schema("empty_db").unwrap();
    assert!(schema.table_names().is_empty());
    assert!(!schema.table_exist("anything"));
}

// ---------------------------------------------------------------------------
// Seed prevents backend load (counting mock)
// ---------------------------------------------------------------------------

/// A mock backend that counts calls to `load_table`.
#[derive(Debug)]
struct CountingMockBackend {
    inner: MockIcebergCatalogBackend,
    load_count: Arc<AtomicUsize>,
}

impl CountingMockBackend {
    fn new(inner: MockIcebergCatalogBackend) -> (Self, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        (Self { inner, load_count: Arc::clone(&counter) }, counter)
    }
}

#[async_trait]
impl IcebergCatalogBackend for CountingMockBackend {
    async fn list_namespaces(&self) -> datafusion_nexus::error::Result<Vec<NamespaceIdent>> {
        self.inner.list_namespaces().await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> datafusion_nexus::error::Result<Vec<TableIdent>> {
        self.inner.list_tables(namespace).await
    }

    async fn load_table(&self, table: &TableIdent) -> datafusion_nexus::error::Result<iceberg::table::Table> {
        self.load_count.fetch_add(1, Ordering::SeqCst);
        self.inner.load_table(table).await
    }
}

#[tokio::test]
async fn seed_prevents_backend_load_on_first_table_access() {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;

    let (counting_backend, load_count) = CountingMockBackend::new(mock_backend_with_tpch());
    let backend = Arc::new(counting_backend);
    let provider = build_catalog_provider_from_backend(backend, "glue", IcebergProviderKind::Gpu).await.unwrap();

    // Seed the table before any query touches it.
    let schema = Arc::new(Schema::new(vec![Field::new("n_nationkey", DataType::Int32, false)]));
    let mem_table = Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap());
    let schema_provider = provider.schema_concrete("tpch_sf1").unwrap();
    schema_provider.seed_table("tpch_sf1_nation".to_string(), mem_table).await;

    // Querying the seeded table should NOT call load_table.
    let result = schema_provider.table("tpch_sf1_nation").await.unwrap();
    assert!(result.is_some());
    assert_eq!(load_count.load(Ordering::SeqCst), 0, "seed should prevent a backend load");
}
