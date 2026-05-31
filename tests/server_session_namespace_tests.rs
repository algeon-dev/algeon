//! Tests for three-part Iceberg catalog namespace registration.
//!
//! These tests exercise the catalog/schema wiring and table registration logic
//! introduced in Milestone 1 of the Iceberg namespace DSL.  They use MemTable
//! providers instead of real Glue credentials so they can run in CI without AWS
//! access.

#![cfg(all(feature = "server", feature = "iceberg"))]

use std::sync::Arc;

use arrow::{
    array::{Int32Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use datafusion::{
    catalog::{MemoryCatalogProvider, MemorySchemaProvider},
    common::TableReference,
    datasource::{MemTable, TableProvider},
    prelude::{SessionConfig, SessionContext},
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn nation_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("n_nationkey", DataType::Int32, false),
        Field::new("n_name", DataType::Utf8, false),
        Field::new("n_regionkey", DataType::Int32, false),
    ]))
}

fn nation_batch() -> RecordBatch {
    let schema = nation_schema();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2])),
            Arc::new(StringArray::from(vec!["ALGERIA", "ARGENTINA", "BRAZIL"])),
            Arc::new(Int32Array::from(vec![0, 1, 1])),
        ],
    )
    .unwrap()
}

/// Registers a MemTable under the given three-part name and optionally also
/// under a bare alias — exactly mirroring what `register_iceberg_tables` does.
fn register_test_table(
    ctx: &SessionContext,
    catalog_name: &str,
    namespace: &str,
    real_table_name: &str,
    alias: Option<&str>,
) {
    let schema = nation_schema();
    let batch = nation_batch();
    let provider: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());

    // Ensure catalog exists.
    if ctx.catalog(catalog_name).is_none() {
        ctx.register_catalog(catalog_name, Arc::new(MemoryCatalogProvider::new()));
    }
    let catalog = ctx.catalog(catalog_name).unwrap();

    // Ensure schema/namespace exists.
    if catalog.schema(namespace).is_none() {
        catalog.register_schema(namespace, Arc::new(MemorySchemaProvider::new())).unwrap();
    }

    // Three-part: catalog.namespace.real_table_name
    ctx.register_table(TableReference::full(catalog_name, namespace, real_table_name), Arc::clone(&provider)).unwrap();

    // Three-part alias when it differs from the real table name.
    if matches!(alias, Some(a) if a != real_table_name) {
        ctx.register_table(TableReference::full(catalog_name, namespace, alias.unwrap()), Arc::clone(&provider))
            .unwrap();
    }

    // Bare alias for backward compatibility.
    let bare = alias.unwrap_or(real_table_name);
    ctx.register_table(bare, Arc::clone(&provider)).unwrap();
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn three_part_catalog_namespace_wiring_creates_schema_in_catalog() {
    let ctx = SessionContext::new();

    // Before registration the catalog should not exist.
    assert!(ctx.catalog("glue").is_none());

    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", None);

    let catalog = ctx.catalog("glue").expect("glue catalog must exist after registration");
    assert!(catalog.schema("demo_db").is_some(), "demo_db schema must exist");
}

#[test]
fn three_part_full_reference_resolves_after_registration() {
    let ctx = SessionContext::new();
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", None);

    // Full three-part reference must resolve.
    assert!(
        ctx.table_exist(TableReference::full("glue", "demo_db", "tpch_sf1_nation")).unwrap(),
        "three-part table reference must resolve"
    );
}

#[test]
fn alias_registration_exposes_both_real_name_and_alias() {
    let ctx = SessionContext::new();
    // alias "nation" → real table "tpch_sf1_nation"
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    assert!(
        ctx.table_exist(TableReference::full("glue", "demo_db", "tpch_sf1_nation")).unwrap(),
        "real table name must resolve under three-part reference"
    );
    assert!(
        ctx.table_exist(TableReference::full("glue", "demo_db", "nation")).unwrap(),
        "alias must resolve under three-part reference"
    );
    // Bare alias backward compat.
    assert!(ctx.table_exist(TableReference::bare("nation")).unwrap(), "bare alias must resolve");
}

#[test]
fn table_equals_alias_does_not_double_register_in_schema() {
    let ctx = SessionContext::new();
    // When alias == real table name, only one entry should appear in the schema.
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("tpch_sf1_nation"));

    let catalog = ctx.catalog("glue").unwrap();
    let schema = catalog.schema("demo_db").unwrap();
    let names = schema.table_names();
    assert_eq!(names.len(), 1, "only one entry in schema when alias equals real name: {names:?}");
    assert_eq!(names[0], "tpch_sf1_nation");
}

#[tokio::test]
async fn information_schema_schemata_lists_namespace() {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    let result = ctx
        .sql("SELECT schema_name FROM information_schema.schemata WHERE catalog_name = 'glue'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let schema_names: Vec<String> = result
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(schema_names.contains(&"demo_db".to_string()), "schemata must list demo_db: {schema_names:?}");
}

#[tokio::test]
async fn information_schema_tables_lists_real_table_name() {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_information_schema(true));
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    let result = ctx
        .sql(
            "SELECT table_name FROM information_schema.tables \
             WHERE table_catalog = 'glue' AND table_schema = 'demo_db'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let table_names: Vec<String> = result
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(
        table_names.contains(&"tpch_sf1_nation".to_string()),
        "information_schema.tables must list tpch_sf1_nation: {table_names:?}"
    );
}

#[tokio::test]
async fn select_from_three_part_name_returns_rows() {
    let ctx = SessionContext::new();
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    let batches = ctx
        .sql("SELECT n_nationkey, n_name FROM glue.demo_db.tpch_sf1_nation ORDER BY n_nationkey")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(row_count, 3, "three-part SELECT must return 3 rows");
}

#[tokio::test]
async fn select_from_three_part_alias_returns_rows() {
    let ctx = SessionContext::new();
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    let batches = ctx.sql("SELECT n_nationkey FROM glue.demo_db.nation").await.unwrap().collect().await.unwrap();

    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(row_count, 3, "three-part alias SELECT must return 3 rows");
}

#[tokio::test]
async fn select_from_bare_alias_still_works_after_three_part_registration() {
    let ctx = SessionContext::new();
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    let batches = ctx.sql("SELECT n_nationkey FROM nation").await.unwrap().collect().await.unwrap();
    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(row_count, 3, "bare alias SELECT must return 3 rows");
}

#[tokio::test]
async fn bare_alias_resolves_when_alias_equals_real_table_name() {
    let ctx = SessionContext::new();
    // When no alias is provided (alias == real_table_name), bare path must still work.
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("tpch_sf1_nation"));

    let batches = ctx.sql("SELECT n_nationkey FROM tpch_sf1_nation").await.unwrap().collect().await.unwrap();
    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(row_count, 3, "bare path should return rows when alias == real table name");
}

#[tokio::test]
async fn select_with_where_clause_via_three_part_name() {
    let ctx = SessionContext::new();
    register_test_table(&ctx, "glue", "demo_db", "tpch_sf1_nation", Some("nation"));

    let batches = ctx
        .sql("SELECT n_name FROM glue.demo_db.tpch_sf1_nation WHERE n_regionkey = 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let names: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(
        names,
        vec!["ARGENTINA", "BRAZIL"],
        "WHERE filter on three-part SELECT should return n_regionkey=1 rows"
    );
}
