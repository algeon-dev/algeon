use super::*;

#[test]
fn default_config_selects_all_tpch_queries() {
    let config = NativeTpchBenchmarkConfig::default();
    assert_eq!(config.query_ids.len(), TPCH_QUERIES.len());
    assert_eq!(config.selected_queries().len(), TPCH_QUERIES.len());
    assert_eq!(config.iterations, 1);
    assert_eq!(config.dataset_mode.as_str(), "sf1_external");
    assert_eq!(NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf10).as_str(), "sf10_external");
    assert_eq!(config.native_memory_policy.as_str(), "bounded");
    assert_eq!(config.native_source_cache_policy.as_str(), "disabled");
    config.validate().unwrap();
}

#[test]
fn sf1_external_dataset_mode_rejects_tiny_or_unknown_metadata() {
    dataset::validate_sf1_external_metadata(&benchmark_metadata()).unwrap();
    dataset::validate_sf10_external_metadata(&benchmark_sf10_metadata()).unwrap();

    let tiny = TpchDatasetMetadata {
        scale_factor: "tiny".to_string(),
        provenance: "generated_tiny".to_string(),
        ..benchmark_metadata()
    };
    let error =
        dataset::validate_sf1_external_metadata(&tiny).expect_err("SF1 benchmark mode must not accept tiny metadata");
    assert!(error.to_string().contains("requires CUDF_TPCH_SCALE_FACTOR=sf1"));

    let unknown = TpchDatasetMetadata {
        scale_factor: "unknown".to_string(),
        provenance: "external_parquet_root".to_string(),
        ..benchmark_metadata()
    };
    let error = dataset::validate_sf1_external_metadata(&unknown)
        .expect_err("SF1 benchmark mode must not accept an unknown scale factor");
    assert!(error.to_string().contains("observed unknown"));

    let without_content_fingerprint = TpchDatasetMetadata { content_fingerprint: None, ..benchmark_metadata() };
    let error = dataset::validate_sf1_external_metadata(&without_content_fingerprint)
        .expect_err("SF1 benchmark mode must preserve a content fingerprint");
    assert!(error.to_string().contains("requires a content fingerprint"));

    let unknown_generation =
        TpchDatasetMetadata { generation_command: "unknown_external".to_string(), ..benchmark_metadata() };
    let error = dataset::validate_sf1_external_metadata(&unknown_generation)
        .expect_err("SF1 benchmark mode must preserve reproducible generation metadata");
    assert!(error.to_string().contains("requires a reproducible CUDF_TPCH_GENERATION_COMMAND"));

    let mut mislabeled_tiny = benchmark_metadata();
    mislabeled_tiny.tables.iter_mut().find(|table| table.table == "lineitem").unwrap().row_count = 3;
    let error = dataset::validate_sf1_external_metadata(&mislabeled_tiny)
        .expect_err("SF1 benchmark mode must reject mislabeled tiny data");
    assert!(error.to_string().contains("requires SF1 row count lineitem=6001215"));

    let error = dataset::validate_sf1_external_metadata(&benchmark_sf10_metadata())
        .expect_err("SF1 benchmark mode must reject SF10 metadata");
    assert!(error.to_string().contains("requires CUDF_TPCH_SCALE_FACTOR=sf1"));

    let error = dataset::validate_sf10_external_metadata(&benchmark_metadata())
        .expect_err("SF10 benchmark mode must reject SF1 metadata");
    assert!(error.to_string().contains("requires CUDF_TPCH_SCALE_FACTOR=sf10"));
}

#[test]
fn sf1_external_dataset_mode_rejects_inconsistent_row_group_metadata_lengths() {
    let mut rows_len_mismatch = benchmark_metadata();
    let table = rows_len_mismatch.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.row_group_count = 2;
    table.row_group_rows = vec![3_000_000];
    table.row_group_bytes = vec![1, 1];
    let error = dataset::validate_sf1_external_metadata(&rows_len_mismatch)
        .expect_err("SF1 benchmark metadata must reject row group row length mismatches");
    assert!(error.to_string().contains("inconsistent row_group_rows length for table lineitem"));

    let mut bytes_len_mismatch = benchmark_metadata();
    let table = bytes_len_mismatch.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.row_group_count = 2;
    table.row_group_rows = vec![3_000_000, 3_001_215];
    table.row_group_bytes = vec![1];
    let error = dataset::validate_sf1_external_metadata(&bytes_len_mismatch)
        .expect_err("SF1 benchmark metadata must reject row group byte length mismatches");
    assert!(error.to_string().contains("inconsistent row_group_bytes length for table lineitem"));

    let mut zero_row_group_bytes = benchmark_metadata();
    zero_row_group_bytes.tables.iter_mut().find(|table| table.table == "lineitem").unwrap().row_group_bytes = vec![0];
    let error = dataset::validate_sf1_external_metadata(&zero_row_group_bytes)
        .expect_err("SF1 benchmark metadata must reject zero-byte row groups");
    assert!(error.to_string().contains("zero-byte row groups for table lineitem"));

    let mut oversized_row_group_bytes = benchmark_metadata();
    let table = oversized_row_group_bytes.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.byte_size = 1;
    table.row_group_bytes = vec![2];
    let error = dataset::validate_sf1_external_metadata(&oversized_row_group_bytes)
        .expect_err("SF1 benchmark metadata must reject row-group byte totals larger than table bytes");
    assert!(error.to_string().contains("row-group bytes greater than table bytes for table lineitem"));
}

#[test]
fn query_id_filter_preserves_configured_order_and_rejects_duplicates() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q03".to_string(), "q01".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };

    let selected = config.selected_queries().into_iter().map(|query| query.id).collect::<Vec<_>>();

    assert_eq!(selected, vec!["q03", "q01"]);

    let duplicate = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string(), "q01".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };
    let error = duplicate.validate().expect_err("duplicate query ids must not silently collapse rows");
    assert!(error.to_string().contains("duplicate TPC-H query id q01"));
}

#[test]
fn direct_config_rejects_empty_query_list() {
    let empty = NativeTpchBenchmarkConfig { query_ids: Vec::new(), ..NativeTpchBenchmarkConfig::default() };
    let error = empty.validate().expect_err("empty direct query lists must not produce empty benchmark reports");
    assert!(error.to_string().contains("must name at least one query id"));
}

#[tokio::test]
async fn benchmark_runner_validates_direct_config_before_selecting_queries() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q99".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let dir = unique_temp_dir("datafusion-cudf-native-tpch-invalid-config");
    let error = run_native_tpch_benchmark(&config, &dir).await.expect_err("invalid direct configs must not panic");

    assert!(error.to_string().contains("unknown TPC-H query id q99"));
    fs::remove_dir_all(dir).unwrap();
}
