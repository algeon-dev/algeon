use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::{
    TPCDS_QUERIES, TPCDS_QUERY_COUNT, TPCDS_TABLES, describe_tpcds_dataset, tiny_tpcds_dataset_metadata,
    tpcds_native_optimizer_config, tpcds_native_report_session_config, tpcds_query_ids, tpcds_report_session_config,
    write_tiny_tpcds_parquet_dataset,
};

#[test]
fn tpcds_query_count_matches_official_99_queries() {
    assert_eq!(TPCDS_QUERY_COUNT, 99);
    assert_eq!(TPCDS_QUERIES.len(), 99);
}

#[test]
fn tpcds_query_ids_are_q01_through_q99_in_order() {
    let expected: Vec<String> = (1..=99).map(|n| format!("q{n:02}")).collect();
    let actual: Vec<&'static str> = tpcds_query_ids().collect();
    assert_eq!(actual, expected.iter().map(String::as_str).collect::<Vec<_>>());
    let from_table: Vec<&'static str> = TPCDS_QUERIES.iter().map(|query| query.id).collect();
    assert_eq!(from_table, expected.iter().map(String::as_str).collect::<Vec<_>>());
}

#[test]
fn tpcds_query_sql_strings_are_non_empty_and_have_no_trailing_semicolon() {
    for query in TPCDS_QUERIES {
        let sql = query.sql;
        assert!(!sql.trim().is_empty(), "{} sql is empty", query.id);
        let trimmed = sql.trim_end();
        assert!(!trimmed.ends_with(';'), "{} sql has a trailing semicolon", query.id);
    }
}

#[test]
fn tpcds_tables_list_has_24_official_tpcds_tables() {
    assert_eq!(TPCDS_TABLES.len(), 24);
    for table in ["store_sales", "catalog_sales", "web_sales", "date_dim", "customer", "item", "web_site"] {
        assert!(TPCDS_TABLES.contains(&table), "TPCDS_TABLES missing {table}");
    }
}

#[test]
fn tpcds_native_report_session_config_avoids_datafusion_statistics_estimates() {
    let report_config = tpcds_report_session_config();
    let native_config = tpcds_native_report_session_config();

    assert!(
        report_config.options().execution.collect_statistics,
        "shared TPC-DS report config remains CPU-compatible; native contexts apply their own statistics safety"
    );
    assert!(
        !native_config.options().execution.collect_statistics,
        "native TPC-DS report planning reads source metadata directly and must not depend on DataFusion join statistics"
    );
    assert!(!native_config.options().optimizer.repartition_joins);
    assert_eq!(native_config.options().execution.target_partitions, 1);
}

#[test]
fn tpcds_native_optimizer_config_disables_cpu_preference_for_gpu_only_sql_surface() {
    let config = tpcds_native_optimizer_config();

    assert!(
        !config.native_plan_preference_options().enabled,
        "TPC-DS native surfaces must not emit cpu_preferred_not_run while SQL may contain GPU-only cugraph_* calls"
    );
}

#[test]
fn write_tiny_tpcds_parquet_dataset_emits_24_parquet_files() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-tiny-fixture");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            name.strip_suffix(".parquet").map(|stem| stem.to_string())
        })
        .collect();
    files.sort();

    let mut expected = TPCDS_TABLES.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    expected.sort();
    assert_eq!(files, expected);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiny_tpcds_dataset_metadata_describes_24_tables() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-tiny-describe");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpcds_dataset_metadata(&dir).unwrap();
    assert_eq!(metadata.table_count, 24);
    assert_eq!(metadata.file_count, 24);
    assert_eq!(metadata.scale_factor, "tiny");
    assert_eq!(metadata.provenance, "generated_tiny");
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert!(metadata.content_fingerprint.is_none());

    let table_names: Vec<&str> = metadata.tables.iter().map(|table| table.table.as_str()).collect();
    for expected in TPCDS_TABLES {
        assert!(table_names.contains(expected), "metadata.tables missing {expected}");
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiny_tpcds_dataset_metadata_projects_into_workload_dataset_metadata() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-tiny-workload-projection");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let metadata = describe_tpcds_dataset(&dir, "tiny", "generated_tiny", "test_projection").unwrap();
    let workload = metadata.to_workload_dataset_metadata();
    assert_eq!(workload.workload, "tpcds");
    assert_eq!(workload.scale_factor.as_deref(), Some("tiny"));
    assert_eq!(workload.provenance, "generated_tiny");
    assert_eq!(workload.generation_command, "test_projection");
    assert_eq!(workload.table_count, 24);
    assert_eq!(workload.file_count, 24);

    fs::remove_dir_all(dir).unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
