use std::{collections::HashMap, sync::Arc};

use datafusion::{
    arrow::datatypes::DataType,
    catalog::TableProvider,
    common::stats::Precision,
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::object_store::ObjectStoreUrl,
    object_store::{ObjectStoreExt, PutPayload, memory::InMemory, path::Path as ObjectPath},
    physical_plan::empty::EmptyExec,
};
use iceberg::spec::DataFile;

use crate::support::{
    Warehouse, data_file, file_scan, people_arrow_schema, people_batch, people_schema, plan, sorted_ids, write_parquet,
    write_parquet_bytes, write_people,
};

#[tokio::test]
async fn each_task_is_one_whole_file_with_manifest_size_and_row_count_and_a_cached_footer() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("people", people_schema(), None, HashMap::new()).await;
    let mut files = write_people(&table, &[1, 2, 3]).await;
    files.extend(write_people(&table, &[101, 102]).await);
    let manifest =
        files.iter().map(|file| (file.file_path().to_string(), file.file_size_in_bytes())).collect::<HashMap<_, _>>();
    warehouse.append(&table, files).await;
    let ctx = warehouse.session().await;

    let plan = plan(&ctx, "SELECT id, name FROM iceberg.ns.people").await.unwrap();
    let scan = file_scan(&plan);
    assert_eq!(scan.object_store_url, ObjectStoreUrl::local_filesystem());
    let scan_files = scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    assert_eq!(scan_files.len(), 2);
    let cache = ctx.runtime_env().cache_manager.get_file_metadata_cache();
    let mut row_counts = Vec::new();
    for file in scan_files {
        assert!(file.range.is_none());
        let location = format!("file:///{}", file.object_meta.location);
        assert_eq!(Some(&file.object_meta.size), manifest.get(&location), "{location}");
        let rows = file.statistics.as_ref().unwrap().num_rows;
        row_counts.push(rows);
        assert!(cache.get(&file.object_meta.location).is_some(), "the footer of {location} must be cached");
    }
    row_counts.sort_by_key(|rows| rows.get_value().copied());
    assert_eq!(row_counts, [Precision::Exact(2), Precision::Exact(3)]);
    assert_eq!(sorted_ids(&ctx, "SELECT id FROM iceberg.ns.people").await, [1, 2, 3, 101, 102]);
}

#[tokio::test]
async fn a_filter_prunes_data_files_by_manifest_bounds() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("people", people_schema(), None, HashMap::new()).await;
    let mut files = write_people(&table, &[1, 2, 3]).await;
    files.extend(write_people(&table, &[101, 102]).await);
    warehouse.append(&table, files).await;
    let ctx = warehouse.session().await;

    let sql = "SELECT id FROM iceberg.ns.people WHERE id >= 100";
    let scan = file_scan(&plan(&ctx, sql).await.unwrap());
    assert_eq!(scan.file_groups.iter().map(|group| group.len()).sum::<usize>(), 1);
    assert_eq!(sorted_ids(&ctx, sql).await, [101, 102]);
}

#[tokio::test]
async fn a_table_without_a_snapshot_scans_no_rows() {
    let warehouse = Warehouse::new().await;
    warehouse.create_table("empty", people_schema(), None, HashMap::new()).await;
    let ctx = warehouse.session().await;

    let plan = plan(&ctx, "SELECT id, name FROM iceberg.ns.empty").await.unwrap();
    assert!(
        plan.downcast_ref::<EmptyExec>().is_some()
            || plan.children().iter().any(|child| child.downcast_ref::<EmptyExec>().is_some())
    );
    assert!(sorted_ids(&ctx, "SELECT id FROM iceberg.ns.empty").await.is_empty());
}

#[tokio::test]
async fn s3a_locations_read_through_the_s3_bucket_store() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("remote", people_schema(), None, HashMap::new()).await;
    let store = Arc::new(InMemory::new());
    let bytes = write_parquet_bytes(&people_batch(&people_arrow_schema(Some([1, 2, 3])), &[7, 8]));
    let size = bytes.len() as u64;
    store.put(&ObjectPath::from("data/remote.parquet"), PutPayload::from(bytes)).await.unwrap();
    let file: DataFile = data_file("s3a://bucket/data/remote.parquet", size, 2).build().unwrap();
    warehouse.append(&table, vec![file]).await;
    let ctx = warehouse.session().await;
    ctx.register_object_store(ObjectStoreUrl::parse("s3://bucket").unwrap().as_ref(), store);

    let plan = plan(&ctx, "SELECT id FROM iceberg.ns.remote").await.unwrap();
    let scan = file_scan(&plan);
    assert_eq!(scan.object_store_url, ObjectStoreUrl::parse("s3://bucket").unwrap());
    let location = &scan.file_groups[0].files()[0].object_meta.location;
    assert_eq!(location.as_ref(), "data/remote.parquet");
    assert!(ctx.runtime_env().cache_manager.get_file_metadata_cache().get(location).is_some());
    assert_eq!(sorted_ids(&ctx, "SELECT id FROM iceberg.ns.remote").await, [7, 8]);
}

#[tokio::test]
async fn files_without_field_ids_bind_through_the_table_name_mapping() {
    let warehouse = Warehouse::new().await;
    let mapping = r#"[{"field-id":1,"names":["id"]},{"field-id":2,"names":["name"]},{"field-id":3,"names":["year"]}]"#;
    let properties = HashMap::from([("schema.name-mapping.default".to_string(), mapping.to_string())]);
    let table = warehouse.create_table("added", people_schema(), None, properties).await;
    let path = warehouse.local_path("added.parquet");
    let size = write_parquet(path.as_ref(), &people_batch(&people_arrow_schema(None), &[4, 5, 6]));
    warehouse.append(&table, vec![data_file(&format!("file://{path}"), size, 3).build().unwrap()]).await;
    let ctx = warehouse.session().await;

    assert_eq!(sorted_ids(&ctx, "SELECT id FROM iceberg.ns.added WHERE name <> 'person-5'").await, [4, 6]);
}

/// String columns use the type a DataFusion listing table infers for the
/// same files under the session's default `schema_force_view_types`.
#[tokio::test]
async fn string_columns_have_the_listing_table_type() {
    let warehouse = Warehouse::new().await;
    let table = warehouse.create_table("people", people_schema(), None, HashMap::new()).await;
    let files = write_people(&table, &[1, 2]).await;
    let location = files[0].file_path().to_string();
    warehouse.append(&table, files).await;
    let ctx = warehouse.session().await;

    let url = ListingTableUrl::parse(&location).unwrap();
    let options = ListingOptions::new(Arc::new(ParquetFormat::default()));
    let config = ListingTableConfig::new(url).with_listing_options(options).infer_schema(&ctx.state()).await.unwrap();
    let listing = ListingTable::try_new(config).unwrap();
    let listing_type = listing.schema().field_with_name("name").unwrap().data_type().clone();

    let iceberg = ctx.table("iceberg.ns.people").await.unwrap();
    let iceberg_type = iceberg.schema().field_with_name(None, "name").unwrap().data_type().clone();
    assert_eq!(listing_type, DataType::Utf8View);
    assert_eq!(iceberg_type, listing_type);
}
