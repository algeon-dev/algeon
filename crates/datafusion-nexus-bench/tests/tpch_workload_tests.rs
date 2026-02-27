use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus_bench::{
    TPCH_QUERY_COUNT, TPCH_TABLES, describe_tpch_dataset_with_content_fingerprint, tiny_tpch_dataset_metadata,
    tpch_fallback_pressure_workload_items, tpch_query_ids, tpch_report_session_config, write_tiny_tpch_parquet_dataset,
};

#[tokio::test]
async fn tpch_workload_builder_generates_stable_query_items() {
    let dir = unique_temp_dir("datafusion-nexus-tpch-workload-builder");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    for table in TPCH_TABLES {
        assert!(dir.join(format!("{table}.parquet")).exists());
    }
    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    assert_eq!(metadata.scale_factor, "tiny");
    assert_eq!(metadata.provenance, "generated_tiny");
    assert_eq!(metadata.generation_command, "datafusion-nexus::write_tiny_tpch_parquet_dataset");
    assert_eq!(metadata.table_count, TPCH_TABLES.len());
    assert_eq!(metadata.file_count, TPCH_TABLES.len());
    assert_eq!(metadata.total_rows, 26);
    assert_eq!(
        metadata.table_row_counts(),
        "customer=3,lineitem=3,nation=5,orders=3,part=3,partsupp=3,region=3,supplier=3"
    );
    assert_eq!(
        metadata.table_file_counts(),
        "customer=1,lineitem=1,nation=1,orders=1,part=1,partsupp=1,region=1,supplier=1"
    );
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert_eq!(metadata.content_fingerprint, None);

    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let actual_ids = items.iter().map(|item| item.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(TPCH_QUERY_COUNT, 22);
    assert_eq!(items.len(), TPCH_QUERY_COUNT);
    assert_eq!(actual_ids, expected_ids);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_metadata_accepts_directory_layouts_and_pins_fingerprint_semantics() {
    let source = unique_temp_dir("datafusion-nexus-tpch-directory-layout-source");
    let layout_a = unique_temp_dir("datafusion-nexus-tpch-directory-layout-a");
    let layout_b = unique_temp_dir("datafusion-nexus-tpch-directory-layout-b");
    write_tiny_tpch_parquet_dataset(&source).unwrap();
    copy_tiny_tpch_as_directory_layout(&source, &layout_a, &["part-000.parquet"]);
    copy_tiny_tpch_as_directory_layout(&source, &layout_b, &["renamed-000.parquet"]);

    let metadata =
        describe_tpch_dataset_with_content_fingerprint(&layout_a, "tiny", "directory_layout", "copy_tiny", true)
            .unwrap();
    assert_eq!(metadata.table_count, TPCH_TABLES.len());
    assert_eq!(metadata.file_count, TPCH_TABLES.len());
    assert_eq!(metadata.total_rows, 26);
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert!(metadata.content_fingerprint.as_deref().unwrap().starts_with("content-fnv1a64:"));

    let lineitem = metadata.tables.iter().find(|table| table.table == "lineitem").unwrap();
    assert_eq!(lineitem.path, layout_a.join("lineitem"));
    assert_eq!(lineitem.file_count, 1);
    assert_eq!(lineitem.files[0].relative_path, "lineitem/part-000.parquet");
    assert!(lineitem.schema.contains("l_orderkey"));

    let same_content_different_paths =
        describe_tpch_dataset_with_content_fingerprint(&layout_b, "tiny", "directory_layout", "copy_tiny", true)
            .unwrap();
    assert_ne!(
        metadata.metadata_fingerprint, same_content_different_paths.metadata_fingerprint,
        "metadata fingerprint must include table-relative file paths"
    );
    assert_ne!(
        metadata.content_fingerprint, same_content_different_paths.content_fingerprint,
        "content fingerprint includes relative paths as part of dataset identity"
    );

    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(layout_a).unwrap();
    fs::remove_dir_all(layout_b).unwrap();
}

#[tokio::test]
async fn tpch_workload_builder_registers_multi_file_directory_layout() {
    let source = unique_temp_dir("datafusion-nexus-tpch-multi-file-source");
    let layout = unique_temp_dir("datafusion-nexus-tpch-multi-file-layout");
    write_tiny_tpch_parquet_dataset(&source).unwrap();
    copy_tiny_tpch_as_directory_layout(&source, &layout, &["part-000.parquet", "part-001.parquet"]);

    let metadata =
        describe_tpch_dataset_with_content_fingerprint(&layout, "tiny", "directory_layout", "copy_tiny", false)
            .unwrap();
    assert_eq!(metadata.file_count, TPCH_TABLES.len() * 2);
    assert_eq!(metadata.total_rows, 52);
    assert_eq!(
        metadata.table_file_counts(),
        "customer=2,lineitem=2,nation=2,orders=2,part=2,partsupp=2,region=2,supplier=2"
    );

    let items = tpch_fallback_pressure_workload_items(&layout, tpch_report_session_config()).await.unwrap();
    assert_eq!(items.len(), TPCH_QUERY_COUNT);

    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(layout).unwrap();
}

// SF100 emits one Parquet file per table for everything except lineitem, which
// trips the 20 GiB max-file-size rule and is sharded into siblings named
// `lineitem_0000.parquet`, `lineitem_0001.parquet`, ... at the dataset root
// (no `lineitem/` subdirectory). Pin that layout end-to-end: metadata discovery,
// fingerprinting, and workload registration must all treat the shards as one
// logical `lineitem` table.
#[tokio::test]
async fn tpch_workload_builder_accepts_sf100_sibling_split_layout() {
    let source = unique_temp_dir("datafusion-nexus-tpch-sibling-split-source");
    let layout = unique_temp_dir("datafusion-nexus-tpch-sibling-split-layout");
    write_tiny_tpch_parquet_dataset(&source).unwrap();
    copy_tiny_tpch_as_sibling_split_layout(&source, &layout, &["lineitem"], 2);

    assert!(layout.join("customer.parquet").is_file());
    assert!(layout.join("lineitem_0000.parquet").is_file());
    assert!(layout.join("lineitem_0001.parquet").is_file());
    assert!(!layout.join("lineitem.parquet").exists());
    assert!(!layout.join("lineitem").exists());

    let metadata =
        describe_tpch_dataset_with_content_fingerprint(&layout, "tiny", "sibling_split_layout", "copy_tiny", true)
            .unwrap();
    assert_eq!(metadata.table_count, TPCH_TABLES.len());
    assert_eq!(metadata.file_count, TPCH_TABLES.len() + 1);
    assert_eq!(
        metadata.table_file_counts(),
        "customer=1,lineitem=2,nation=1,orders=1,part=1,partsupp=1,region=1,supplier=1"
    );
    assert_eq!(
        metadata.table_row_counts(),
        "customer=3,lineitem=6,nation=5,orders=3,part=3,partsupp=3,region=3,supplier=3"
    );
    let lineitem = metadata.tables.iter().find(|table| table.table == "lineitem").unwrap();
    assert_eq!(lineitem.file_count, 2);
    let relative_paths: Vec<_> = lineitem.files.iter().map(|file| file.relative_path.as_str()).collect();
    assert_eq!(relative_paths, vec!["lineitem_0000.parquet", "lineitem_0001.parquet"]);
    assert!(metadata.content_fingerprint.as_deref().unwrap().starts_with("content-fnv1a64:"));

    let items = tpch_fallback_pressure_workload_items(&layout, tpch_report_session_config()).await.unwrap();
    assert_eq!(items.len(), TPCH_QUERY_COUNT);

    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(layout).unwrap();
}

fn copy_tiny_tpch_as_directory_layout(source: &Path, destination: &Path, file_names: &[&str]) {
    for table in TPCH_TABLES {
        let table_dir = destination.join(table);
        fs::create_dir_all(&table_dir).unwrap();
        let source_file = source.join(format!("{table}.parquet"));
        for file_name in file_names {
            fs::copy(&source_file, table_dir.join(file_name)).unwrap();
        }
    }
}

fn copy_tiny_tpch_as_sibling_split_layout(
    source: &Path,
    destination: &Path,
    split_tables: &[&str],
    split_count: usize,
) {
    assert!(split_count >= 1, "split_count must be >= 1");
    fs::create_dir_all(destination).unwrap();
    for table in TPCH_TABLES {
        let source_file = source.join(format!("{table}.parquet"));
        if split_tables.contains(table) {
            for shard in 0..split_count {
                let shard_name = format!("{table}_{shard:04}.parquet");
                fs::copy(&source_file, destination.join(shard_name)).unwrap();
            }
        } else {
            fs::copy(&source_file, destination.join(format!("{table}.parquet"))).unwrap();
        }
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
