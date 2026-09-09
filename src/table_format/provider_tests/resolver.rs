#[test]
fn table_format_parquet_resolver_reports_local_remote_and_cache_strategies() {
    let config = TableFormatParquetReadStrategyConfig::default();
    let local = resolve_table_format_parquet_source(
        "file:///tmp/table/data.parquet",
        &TableFormatReadCapability::local_file_system(),
        &config,
    )
    .unwrap();
    assert_eq!(local.reader_location(), "/tmp/table/data.parquet");
    assert_eq!(local.read_strategy(), ParquetReadStrategy::LocalKvikio);
    assert!(matches!(&local.input, ResolvedParquetInput::LocalKvikio { .. }));
    assert_eq!(local.credential_mode(), CredentialMode::None);

    let remote = resolve_table_format_parquet_source(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &config,
    )
    .unwrap();
    assert_eq!(remote.read_strategy(), ParquetReadStrategy::RemoteDirect);
    assert!(matches!(&remote.input, ResolvedParquetInput::RemoteKvikio { .. }));
    assert_eq!(remote.credential_mode(), CredentialMode::SharedProcessEnv);

    let cache_dir = tempfile::tempdir().unwrap();
    let cache_miss_config = TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let cache_miss = resolve_table_format_parquet_source(
        "s3://bucket/table/missing.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_miss_config,
    )
    .unwrap();
    assert_eq!(cache_miss.read_strategy(), ParquetReadStrategy::RemoteDirect);
    assert_eq!(cache_miss.cache_status, LakehouseCacheStatus::Miss);

    let cache_config = TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let cache_metadata =
        TableFormatParquetSourceMetadata::default().with_size_bytes(4096).with_content_identity("etag:abc123");
    let cache_path_probe = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &TableFormatParquetReadStrategyConfig {
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
            ..cache_config.clone()
        },
        &cache_metadata,
    )
    .unwrap();
    let cache_file = PathBuf::from(cache_path_probe.reader_location());
    std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
    std::fs::write(&cache_file, vec![0u8; 4096]).unwrap();
    let cached = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &cache_metadata,
    )
    .unwrap();
    assert_eq!(cached.read_strategy(), ParquetReadStrategy::LocalCacheKvikio);
    assert!(matches!(&cached.input, ResolvedParquetInput::LocalKvikio { .. }));
    assert_eq!(cached.cache_status, LakehouseCacheStatus::Hit);
    assert_eq!(cached.reader_location(), cache_file.display().to_string());
}

#[test]
fn table_format_parquet_resolver_selects_fill_before_read_for_local_cache_miss_policy() {
    let cache_dir = tempfile::tempdir().unwrap();
    let cache_config = TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        lakehouse_cache_max_bytes: Some(16 * 1024 * 1024),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };

    let resolved = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &TableFormatParquetSourceMetadata::default()
            .with_size_bytes(4096)
            .with_content_identity("etag:abc123"),
    )
    .unwrap();

    assert_eq!(resolved.read_strategy(), ParquetReadStrategy::LocalCacheKvikio);
    assert!(matches!(&resolved.input, ResolvedParquetInput::LocalKvikio { .. }));
    assert_eq!(resolved.cache_status, LakehouseCacheStatus::FillBeforeRead);
    assert_eq!(resolved.original_location, "s3://bucket/table/data.parquet");
    assert_eq!(resolved.lakehouse_cache_root.as_deref(), Some(cache_dir.path().to_str().unwrap()));
    assert_eq!(resolved.lakehouse_cache_max_bytes, Some(16 * 1024 * 1024));
    assert!(resolved.reader_location().contains("/v1/s3/bucket/etag_abc123/"));
    assert!(resolved.reader_location().contains("data.parquet__h"));
}

#[test]
fn table_format_parquet_resolver_scopes_local_cache_by_storage_domain_and_content_identity() {
    let cache_dir = tempfile::tempdir().unwrap();
    let cache_config = TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(12))
        .with_storage_domain("rustfs:bucket-a")
        .with_content_identity("etag:abc123");
    let resolved = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &metadata,
    )
    .unwrap();
    let cache_path = PathBuf::from(resolved.reader_location());
    assert!(resolved.reader_location().contains("/v1/s3/rustfs_bucket-a/etag_abc123/"));

    std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
    std::fs::write(&cache_path, vec![0u8; 4096]).unwrap();
    let cached = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &metadata,
    )
    .unwrap();
    assert_eq!(cached.cache_status, LakehouseCacheStatus::Hit);
    assert_eq!(cached.reader_location(), resolved.reader_location());
    assert_eq!(cached.storage_domain.as_deref(), Some("rustfs:bucket-a"));
    assert_eq!(cached.content_identity.as_deref(), Some("etag:abc123"));

    let changed_metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(8192), Some(12))
        .with_storage_domain("rustfs:bucket-a")
        .with_content_identity("etag:def456");
    let changed = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &changed_metadata,
    )
    .unwrap();
    assert_eq!(changed.cache_status, LakehouseCacheStatus::FillBeforeRead);
    assert_ne!(changed.reader_location(), cached.reader_location());
}

#[test]
fn table_format_parquet_resolver_scopes_local_cache_by_catalog_endpoint_without_endpoint_secrets() {
    let cache_dir = tempfile::tempdir().unwrap();
    let cache_config = TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root: Some(cache_dir.path().to_path_buf()),
        lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
        storage_endpoint: Some("https://user:secret@rustfs.local:9000/minio?X-Amz-Signature=secret#frag".to_string()),
        storage_profile_id: Some("rest-s3".to_string()),
        credential_mode: CredentialMode::SharedProcessEnv,
        ..Default::default()
    };
    let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(12))
        .with_content_identity("etag:abc123");
    let resolved = resolve_table_format_parquet_source_with_metadata(
        "s3://bucket-a/table/data.parquet",
        &TableFormatReadCapability::object_store("s3"),
        &cache_config,
        &metadata,
    )
    .unwrap();

    assert_eq!(resolved.storage_domain.as_deref(), Some("bucket-a@https://rustfs.local:9000/minio"));
    assert_eq!(resolved.storage_profile_id(), Some("rest-s3"));
    assert!(resolved.reader_location().contains("rustfs.local_9000"));
    assert!(!resolved.reader_location().contains("user"));
    assert!(!resolved.reader_location().contains("secret"));
    assert!(!resolved.reader_location().contains("X-Amz-Signature"));
}
