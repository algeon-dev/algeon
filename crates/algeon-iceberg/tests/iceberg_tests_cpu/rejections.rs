use std::{collections::HashMap, sync::Arc};

use algeon_datafusion::{error::ErrorCode, native_exec::S3SourceStorageProfile};
use algeon_iceberg::IcebergCatalogProvider;
use datafusion::prelude::SessionContext;
use iceberg::spec::{
    DataFile, DataFileFormat, Literal, Struct, Transform, UnboundPartitionField, UnboundPartitionSpec,
};

use crate::support::{
    Failure, Warehouse, data_file, people_arrow_schema, people_batch, people_schema, plan_failure, write_parquet,
    write_people,
};

fn assert_rejected(failure: &Failure, table: &str, reason: &str) {
    assert_eq!(failure.code, ErrorCode::UnsupportedTableFeature, "{failure:?}");
    assert_eq!(failure.fact("reason"), Some(reason), "{failure:?}");
    assert_eq!(failure.fact("table"), Some(format!("ns.{table}").as_str()), "{failure:?}");
    assert!(failure.fact("snapshot_id").is_some(), "{failure:?}");
}

/// A table holding one hand-written Parquet file whose Arrow schema has
/// the field ids `ids` (none when `None`), with `rows` ids.
async fn table_with_written_file(
    warehouse: &Warehouse,
    name: &str,
    ids: Option<[i32; 3]>,
    properties: HashMap<String, String>,
) {
    let table = warehouse.create_table(name, people_schema(), None, properties).await;
    let path = warehouse.local_path(&format!("{name}.parquet"));
    let size = write_parquet(path.as_ref(), &people_batch(&people_arrow_schema(ids), &[1, 2]));
    warehouse.append(&table, vec![data_file(&format!("file://{path}"), size, 2).build().unwrap()]).await;
}

/// A table holding one manifest entry per location, none of them read.
async fn table_with_entries(warehouse: &Warehouse, name: &str, entries: Vec<DataFile>) {
    let table = warehouse.create_table(name, people_schema(), None, HashMap::new()).await;
    warehouse.append(&table, entries).await;
}

#[tokio::test]
async fn a_column_whose_field_id_belongs_to_another_name_is_rejected() {
    let warehouse = Warehouse::new().await;
    table_with_written_file(&warehouse, "renamed", Some([1, 9, 3]), HashMap::new()).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT name FROM iceberg.ns.renamed").await;
    assert_rejected(&failure, "renamed", "iceberg_field_id_mismatch");
    assert_eq!(failure.fact("column"), Some("name"));
    // Columns whose ids agree stay readable.
    assert!(ctx.sql("SELECT id FROM iceberg.ns.renamed").await.unwrap().collect().await.is_ok());
}

#[tokio::test]
async fn a_file_without_field_ids_and_no_name_mapping_is_rejected() {
    let warehouse = Warehouse::new().await;
    table_with_written_file(&warehouse, "unmapped", None, HashMap::new()).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT id FROM iceberg.ns.unmapped").await;
    assert_rejected(&failure, "unmapped", "iceberg_missing_field_ids");
}

#[tokio::test]
async fn a_name_mapping_that_resolves_a_column_to_another_name_is_rejected() {
    let warehouse = Warehouse::new().await;
    let mapping =
        r#"[{"field-id":1,"names":["id"]},{"field-id":2,"names":["full_name"]},{"field-id":3,"names":["year"]}]"#;
    let properties = HashMap::from([("schema.name-mapping.default".to_string(), mapping.to_string())]);
    table_with_written_file(&warehouse, "mismapped", None, properties).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT name FROM iceberg.ns.mismapped").await;
    assert_rejected(&failure, "mismapped", "iceberg_missing_field_ids");
}

#[tokio::test]
async fn a_required_column_missing_from_a_file_is_rejected() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("no_id", people_schema(), None, HashMap::new()).await;
    let path = warehouse.local_path("no_id.parquet");
    let batch = people_batch(&people_arrow_schema(Some([1, 2, 3])), &[1, 2]).project(&[1, 2]).unwrap();
    let size = write_parquet(path.as_ref(), &batch);
    warehouse.append(&table, vec![data_file(&format!("file://{path}"), size, 2).build().unwrap()]).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT id FROM iceberg.ns.no_id").await;
    assert_rejected(&failure, "no_id", "iceberg_required_column_absent");
    // An optional column absent from the file reads as null.
    let batch = people_batch(&people_arrow_schema(Some([1, 2, 3])), &[1, 2]).project(&[0, 1]).unwrap();
    let path = warehouse.local_path("no_year.parquet");
    let size = write_parquet(path.as_ref(), &batch);
    let table = warehouse.create_table("no_year", people_schema(), None, HashMap::new()).await;
    warehouse.append(&table, vec![data_file(&format!("file://{path}"), size, 2).build().unwrap()]).await;
    let ctx = warehouse.session().await;
    let batches = ctx.sql("SELECT year FROM iceberg.ns.no_year").await.unwrap().collect().await.unwrap();
    assert_eq!(batches.iter().map(|batch| batch.column(0).null_count()).sum::<usize>(), 2);
}

#[tokio::test]
async fn an_identity_partition_column_kept_only_in_partition_data_is_rejected() {
    let warehouse = Warehouse::new().await;
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(
            UnboundPartitionField::builder()
                .source_ids(vec![3])
                .name("year".to_string())
                .transform(Transform::Identity)
                .build()
                .unwrap(),
        )
        .unwrap()
        .build();
    let table = warehouse.create_table("by_year", people_schema(), Some(spec), HashMap::new()).await;
    let path = warehouse.local_path("by_year.parquet");
    let batch = people_batch(&people_arrow_schema(Some([1, 2, 3])), &[1, 2]).project(&[0, 1]).unwrap();
    let size = write_parquet(path.as_ref(), &batch);
    let file = data_file(&format!("file://{path}"), size, 2)
        .partition(Struct::from_iter([Some(Literal::int(2000))]))
        .partition_spec_id(table.metadata().default_partition_spec_id())
        .build()
        .unwrap();
    warehouse.append(&table, vec![file]).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT year FROM iceberg.ns.by_year").await;
    assert_rejected(&failure, "by_year", "iceberg_partition_column_absent");
}

#[tokio::test]
async fn delete_files_are_rejected() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("people", people_schema(), None, HashMap::new()).await;
    let files = write_people(&table, &[1, 2, 3]).await;
    let target = files[0].file_path().to_string();
    let table = warehouse.append(&table, files).await;
    warehouse.commit_position_delete(&table, "with_deletes", &target).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT id FROM iceberg.ns.with_deletes").await;
    assert_rejected(&failure, "with_deletes", "iceberg_delete_files");
}

#[tokio::test]
async fn a_data_file_that_is_not_parquet_is_rejected() {
    let warehouse = Warehouse::new().await;
    let path = format!("file://{}", warehouse.local_path("data.orc"));
    let entry = data_file(&path, 100, 1).file_format(DataFileFormat::Orc).build().unwrap();
    table_with_entries(&warehouse, "orc", vec![entry]).await;
    let ctx = warehouse.session().await;

    assert_rejected(&plan_failure(&ctx, "SELECT id FROM iceberg.ns.orc").await, "orc", "iceberg_non_parquet_file");
}

#[tokio::test]
async fn an_encrypted_data_file_is_rejected() {
    let warehouse = Warehouse::new().await;
    let path = format!("file://{}", warehouse.local_path("sealed.parquet"));
    let entry = data_file(&path, 100, 1).key_metadata(Some(vec![1, 2, 3])).build().unwrap();
    table_with_entries(&warehouse, "sealed", vec![entry]).await;
    let ctx = warehouse.session().await;

    assert_rejected(&plan_failure(&ctx, "SELECT id FROM iceberg.ns.sealed").await, "sealed", "iceberg_encrypted_file");
}

#[tokio::test]
async fn a_data_file_outside_s3_storage_is_rejected() {
    let warehouse = Warehouse::new().await;
    let entry = data_file("gs://bucket/data.parquet", 100, 1).build().unwrap();
    table_with_entries(&warehouse, "gcs", vec![entry]).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT id FROM iceberg.ns.gcs").await;
    assert_rejected(&failure, "gcs", "iceberg_unsupported_location_scheme");
}

#[tokio::test]
async fn data_files_in_two_buckets_are_rejected() {
    let warehouse = Warehouse::new().await;
    let entries = vec![
        data_file("s3://first/a.parquet", 100, 1).build().unwrap(),
        data_file("s3a://second/b.parquet", 100, 1).build().unwrap(),
    ];
    table_with_entries(&warehouse, "split", entries).await;
    let ctx = warehouse.session().await;

    assert_rejected(&plan_failure(&ctx, "SELECT id FROM iceberg.ns.split").await, "split", "iceberg_multiple_buckets");
}

#[tokio::test]
async fn a_table_whose_file_io_addresses_other_storage_is_rejected() {
    let catalog_endpoint = HashMap::from([("s3.endpoint".to_string(), "http://catalog-store:9000".to_string())]);
    let warehouse = Warehouse::with_properties(catalog_endpoint).await;
    let table = warehouse.create_table("people", people_schema(), None, HashMap::new()).await;
    let files = write_people(&table, &[1]).await;
    warehouse.append(&table, files).await;
    let storage = S3SourceStorageProfile::default_chain().with_endpoint("http://data-store:9000");
    let ctx = SessionContext::new();
    let provider = IcebergCatalogProvider::try_new(Arc::clone(&warehouse.catalog), Some(storage)).await.unwrap();
    ctx.register_catalog("iceberg", Arc::new(provider));

    let failure = plan_failure(&ctx, "SELECT id FROM iceberg.ns.people").await;
    assert_rejected(&failure, "people", "iceberg_storage_properties_mismatch");
    assert_eq!(failure.fact("file_io_property"), Some("s3.endpoint"));
}

#[tokio::test]
async fn a_data_file_whose_size_differs_from_the_manifest_fails_as_a_source_error() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("resized", people_schema(), None, HashMap::new()).await;
    let path = warehouse.local_path("resized.parquet");
    let size = write_parquet(path.as_ref(), &people_batch(&people_arrow_schema(Some([1, 2, 3])), &[1]));
    warehouse.append(&table, vec![data_file(&format!("file://{path}"), size + 1, 1).build().unwrap()]).await;
    let ctx = warehouse.session().await;

    let failure = plan_failure(&ctx, "SELECT id FROM iceberg.ns.resized").await;
    assert_eq!(failure.code, ErrorCode::SourceDependencyFailure, "{failure:?}");
    assert_eq!(failure.fact("manifest_size"), Some((size + 1).to_string().as_str()));
}
