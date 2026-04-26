use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::{
    decimal::validate_supplied_reader_decimal_types_match_footer,
    metadata::{SourceMetadataCacheKey, local_parquet_footer_metadata_path, sidecar_content_identity},
    *,
};
use crate::{
    expr::{BinaryOp, Expr, LiteralValue},
    plan::NodeId,
    runtime::{MemoryControlPolicy, SourceChunkPlanningMode},
    source::{
        CredentialMode, LakehouseCacheReadMode, LakehouseCacheStatus, LocalParquetFile, LocalParquetSource,
        ParquetPositionDeleteFile, ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation,
        ResolvedParquetSource,
    },
};

fn batchable_chunk(index: usize, file_index: usize, bytes: u64) -> SourceChunk {
    SourceChunk {
        source_node_id: NodeId::from_index_for_internal_use(1),
        chunk_index: index,
        file_index,
        location: ParquetSourceLocation::LocalPath(PathBuf::from(format!("part-{file_index}.parquet"))),
        path: PathBuf::from(format!("part-{file_index}.parquet")),
        resolved_source: ResolvedParquetSource::local_kvikio(format!("part-{file_index}.parquet")),
        stable_id: format!("file-{file_index}"),
        row_group_indexes: vec![0],
        reads_all_row_groups: false,
        known_file_bytes: Some(bytes),
        known_row_group_bytes: Some(bytes),
        known_projected_compressed_bytes: Some(bytes),
        known_projected_uncompressed_bytes: Some(bytes),
        known_pure_filter_compressed_bytes: Some(0),
        known_pure_filter_uncompressed_bytes: Some(0),
        estimated_chunk_bytes: Some(bytes),
        estimated_source_read_bytes: Some(bytes),
        exact_source_read_preflight_bytes: Some(bytes),
        source_file_row_groups: None,
        source_file_row_group_bytes: Some(vec![bytes]),
        source_file_projected_compressed_bytes: Some(vec![bytes]),
        source_file_projected_uncompressed_bytes: Some(vec![bytes]),
        source_file_pure_filter_compressed_bytes: Some(vec![0]),
        source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
        source_file_row_group_rows: Some(vec![1]),
        reader_decimal_types: Vec::new(),
        reader_decimal_types_footer_verified: false,
        read_columns: vec!["c0".to_string()],
        partition_values: vec![],
        zero_column_count: false,
        known_row_count: Some(1),
        scan_limit_rows: None,
        size_estimate_based: false,
        source_batching_reason: None,
        source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
        source_chunk_planning_reason: "test_policy".to_string(),
    }
}

fn remote_direct_require_batchable_chunk(index: usize, file_index: usize, bytes: u64) -> SourceChunk {
    let mut chunk = batchable_chunk(index, file_index, bytes);
    let location = format!("s3://bucket/part-{file_index}.parquet");
    chunk.resolved_source = ResolvedParquetSource::remote_kvikio(location.clone(), CredentialMode::SharedProcessEnv);
    chunk.location = ParquetSourceLocation::object_store_uri(&location).expect("valid s3 object URI");
    chunk.path = PathBuf::from(location);
    chunk
}

fn bounded_policy(
    device_budget_bytes: u64,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
) -> MemoryControlPolicy {
    MemoryControlPolicy::Bounded {
        device_budget_bytes,
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    }
}

fn local_source_file() -> LocalParquetFile {
    LocalParquetFile::new("/tmp/source.parquet")
        .with_row_groups((0..2048).collect())
        .with_row_group_bytes(vec![1024; 2048])
        .with_projected_row_group_bytes(vec![512; 2048], vec![1024; 2048])
        .with_pure_filter_row_group_bytes(vec![0; 2048], vec![0; 2048])
        .with_row_group_rows(vec![1; 2048])
}

fn source_from_file(file: LocalParquetFile) -> LocalParquetSource {
    let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
    LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string()])
}

fn local_source() -> LocalParquetSource {
    source_from_file(local_source_file())
}

fn position_delete_file() -> ParquetPositionDeleteFile {
    ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-0",
        ResolvedParquetSource::local_kvikio("/tmp/delete.parquet"),
    )
    .unwrap()
}

fn missing_metadata_source() -> LocalParquetSource {
    source_from_file(LocalParquetFile::new("/tmp/missing-metadata.parquet").with_row_groups((0..2048).collect()))
}

fn write_i64_parquet(path: &PathBuf, values: &[i64], max_row_group_rows: usize) {
    let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(max_row_group_rows)).build();
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn conservative_source_chunk_planning_preserves_configured_row_group_cap() {
    let source = local_source();
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::Conservative,
    );

    assert_eq!(policy.max_row_groups_per_chunk, 1);
    assert_eq!(policy.max_source_chunk_bytes, Some(64 * 1024 * 1024));
}

#[test]
fn local_throughput_source_chunk_planning_raises_row_group_cap_for_large_local_kvikio_source() {
    let source = local_source();
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(policy.max_source_chunk_bytes, Some(64 * 1024 * 1024));
}

#[test]
fn local_throughput_source_chunk_planning_uses_configured_expert_options() {
    let source = local_source();
    let options = SourceChunkPlanningOptions::default()
        .with_local_parquet_throughput_min_device_budget_bytes(8 * 1024 * 1024 * 1024)
        .with_local_parquet_throughput_max_row_groups_per_chunk(16);
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
        &source,
        bounded_policy(8 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
        options,
    );

    assert_eq!(policy.max_row_groups_per_chunk, 16);
    assert_eq!(policy.source_chunk_planning_reason(), "local_parquet_throughput_admitted");
}

#[test]
fn source_chunk_planning_options_reject_zero_thresholds() {
    let error = SourceChunkPlanningOptions::default()
        .with_local_parquet_throughput_max_row_groups_per_chunk(0)
        .validate()
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));

    let error = SourceChunkPlanningOptions::default()
        .with_local_parquet_throughput_min_device_budget_bytes(0)
        .validate()
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_min_device_budget_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn local_throughput_source_chunk_planning_accepts_local_cache_reader_path() {
    let path = std::env::temp_dir().join(format!(
        "nexus-query-engine-local-cache-throughput-{}-{}.parquet",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::write(&path, b"placeholder").unwrap();
    let resolved_source = ResolvedParquetSource::local_cache_kvikio(
        "s3://bucket/table/part-0.parquet",
        path.display().to_string(),
        CredentialMode::SharedProcessEnv,
        LakehouseCacheStatus::Hit,
    );
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
        .unwrap()
        .with_row_groups((0..2048).collect())
        .with_row_group_bytes(vec![1024; 2048])
        .with_projected_row_group_bytes(vec![512; 2048], vec![1024; 2048])
        .with_pure_filter_row_group_bytes(vec![0; 2048], vec![0; 2048])
        .with_row_group_rows(vec![1; 2048]);
    let source = source_from_file(file);
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(policy.source_chunk_planning_reason(), "local_parquet_throughput_admitted");
    fs::remove_file(path).unwrap();
}

#[test]
fn remote_direct_metadata_path_respects_cache_read_bypass() {
    let path = std::env::temp_dir().join(format!(
        "nexus-query-engine-remote-direct-bypass-{}-{}.parquet",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::write(&path, b"placeholder").unwrap();
    let resolved_source =
        ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
            .with_lakehouse_cache_path(path.display().to_string())
            .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Bypassed)
            .with_content_identity("content-a");
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source).unwrap().with_stable_id("stable-a");

    assert!(local_parquet_footer_metadata_path(&file).is_none());
    assert_eq!(sidecar_content_identity(&file), "content-a");

    fs::remove_file(path).unwrap();
}

#[test]
fn source_metadata_cache_key_distinguishes_object_store_profile_and_content_identity() {
    let base_resolved =
        ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv);
    let base = LocalParquetFile::from_resolved_parquet_source(base_resolved)
        .expect("valid remote direct source")
        .with_stable_id("part-0")
        .with_file_size_bytes(4096);
    let other_profile = base
        .clone()
        .with_resolved_parquet_source(base.resolved_source.clone().with_storage_profile_id("profile-b"))
        .expect("valid profiled remote direct source");
    let other_content = base
        .clone()
        .with_resolved_parquet_source(base.resolved_source.clone().with_content_identity("content-b"))
        .expect("valid content-scoped remote direct source");
    let other_location = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
        "s3://bucket/table/part-1.parquet",
        CredentialMode::SharedProcessEnv,
    ))
    .expect("valid remote direct source")
    .with_stable_id("part-0")
    .with_file_size_bytes(4096);

    assert_ne!(SourceMetadataCacheKey::from_file(&base), SourceMetadataCacheKey::from_file(&other_profile));
    assert_ne!(SourceMetadataCacheKey::from_file(&base), SourceMetadataCacheKey::from_file(&other_content));
    assert_ne!(SourceMetadataCacheKey::from_file(&base), SourceMetadataCacheKey::from_file(&other_location));
}

#[test]
fn remote_direct_decimal_reader_predicate_requires_footer_type_metadata() {
    let schema = Arc::new(Schema::new(vec![Field::new("discount", DataType::Decimal128(15, 2), true)]));
    let resolved =
        ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv);
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("part-0")
        .with_file_size_bytes(4096);
    let predicate = Expr::binary(
        Expr::column("discount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
        .with_column_file_indices(vec![0], vec![0], vec![])
        .with_reader_predicate(predicate);

    let error = plan_local_parquet_source_chunks(
        NodeId::from_index_for_internal_use(7),
        &source,
        SourceChunkPlanningPolicy::new(None, 1),
    )
    .expect_err("remote decimal reader predicates need proven footer dtype metadata");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_unavailable"));
    assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
    assert_eq!(error.fact_value("remote_read_mode"), Some("require_kvikio"));
    assert_eq!(error.fact_value("missing_reader_decimal_columns"), Some("discount"));
    assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
    assert!(!error.to_string().contains("s3://bucket"));
}

#[test]
fn remote_direct_decimal_reader_predicate_rejects_supplied_footer_type_mismatch() {
    let schema = Arc::new(Schema::new(vec![Field::new("discount", DataType::Decimal128(15, 2), true)]));
    let resolved =
        ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv);
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("part-0")
        .with_reader_decimal_types(vec![ParquetReaderDecimalType::new(
            "discount",
            ParquetReaderDecimalTypeId::Decimal32,
            -2,
        )]);
    let predicate = Expr::binary(
        Expr::column("discount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
        .with_column_file_indices(vec![0], vec![0], vec![])
        .with_reader_predicate(predicate);
    let footer_reader_decimal_types =
        vec![ParquetReaderDecimalType::new("discount", ParquetReaderDecimalTypeId::Decimal64, -2)];

    let error = validate_supplied_reader_decimal_types_match_footer(
        &source,
        &source.files[0],
        &source.files[0].reader_decimal_types,
        &footer_reader_decimal_types,
    )
    .expect_err("pre-attached reader decimal types must match footer metadata");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("reader_decimal_type_metadata_mismatch"));
    assert_eq!(error.fact_value("read_strategy"), Some("remote_direct"));
    assert_eq!(error.fact_value("remote_read_mode"), Some("require_kvikio"));
    assert!(
        error
            .fact_value("missing_reader_decimal_columns")
            .is_some_and(|value| { value.contains("discount:decimal32:-2!=footer:decimal64:-2") })
    );
    assert!(!error.to_string().contains("s3://bucket"));
}

#[test]
fn local_throughput_source_chunk_planning_keeps_small_budget_conservative() {
    let source = local_source();
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES - 1, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.max_row_groups_per_chunk, 1);
}

#[test]
fn local_throughput_source_chunk_planning_requires_represented_source_read_bytes() {
    let source = missing_metadata_source();
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.max_row_groups_per_chunk, 1);

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();
    assert_eq!(plan.chunk_count(), 2048);
    assert!(plan.chunks.iter().all(|chunk| chunk.row_group_count() == 1));
}

#[test]
fn local_throughput_source_chunk_planning_rejects_object_store_sources() {
    let object_store = source_from_file(
        LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet").expect("valid s3 object URI"),
    );
    let policy = bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1);

    assert_eq!(
        SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &object_store,
            policy,
            SourceChunkPlanningMode::LocalParquetThroughput,
        )
        .max_row_groups_per_chunk,
        1
    );
}

#[test]
fn object_store_source_without_row_group_metadata_plans_read_all_row_groups_chunk() {
    let source = source_from_file(
        LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet").expect("valid s3 object URI"),
    );
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::Conservative,
    );

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert!(plan.chunks[0].reads_all_row_groups);
    assert!(plan.chunks[0].row_group_indexes.is_empty());
    assert_eq!(plan.chunks[0].source_file_row_group_rows, None);
}

#[test]
fn object_store_zero_column_count_uses_synthetic_row_count_metadata() {
    let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
        .expect("valid s3 object URI")
        .with_file_size_bytes(1024)
        .with_row_group_rows(vec![123]);
    let source = LocalParquetSource::zero_column_count(vec![file]);
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::Conservative,
    );

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert!(!plan.chunks[0].reads_all_row_groups);
    assert_eq!(plan.chunks[0].known_row_count, Some(123));
    assert_eq!(plan.chunks[0].source_file_row_group_rows.as_deref(), Some(&[123][..]));
}

#[test]
fn empty_zero_column_count_source_plans_one_zero_weight_identity() {
    let file = LocalParquetFile::object_store_uri("s3://bucket/table/empty.parquet")
        .expect("valid s3 object URI")
        .with_file_size_bytes(1024)
        .with_row_groups(Vec::new())
        .with_row_group_rows(Vec::new());
    let source = LocalParquetSource::zero_column_count(vec![file]);
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::Conservative,
    );

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    let identity = &plan.chunks[0];
    assert!(identity.zero_column_count);
    assert!(!identity.reads_all_row_groups);
    assert!(identity.row_group_indexes.is_empty());
    assert_eq!(identity.known_row_count, Some(0));
    assert_eq!(identity.estimated_source_read_bytes, Some(0));
    assert!(identity.exact_source_read_preflight_bytes.is_some());
}

#[test]
fn chunk_planning_prefers_footer_row_counts_over_whole_file_synthetic_count() {
    let path = std::env::temp_dir().join(format!(
        "nexus-query-engine-zero-column-count-footer-{}-{}.parquet",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    write_i64_parquet(&path, &[1, 2, 3, 4, 5], 2);
    let file_size = fs::metadata(&path).unwrap().len();
    let file = LocalParquetFile::new(path.clone()).with_file_size_bytes(file_size).with_row_group_rows(vec![5]);
    let source = LocalParquetSource::zero_column_count(vec![file]);
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::Conservative,
    );

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_internal_use(3), &source, policy).unwrap();

    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
    assert_eq!(plan.chunks[2].row_group_indexes, vec![2]);
    assert_eq!(
        plan.chunks.iter().map(|chunk| chunk.known_row_count).collect::<Vec<_>>(),
        vec![Some(2), Some(2), Some(1)]
    );
    assert!(plan.chunks.iter().all(|chunk| chunk.source_file_row_group_rows.as_deref() == Some(&[2, 2, 1][..])));
    fs::remove_file(path).unwrap();
}

#[test]
fn source_chunk_planning_splits_position_delete_files_into_row_position_chunks() {
    let file = LocalParquetFile::new("/tmp/position-delete-source.parquet")
        .with_row_group_bytes(vec![100, 200, 300])
        .with_projected_row_group_bytes(vec![50, 100, 150], vec![100, 200, 300])
        .with_pure_filter_row_group_bytes(vec![0, 0, 0], vec![0, 0, 0])
        .with_row_group_rows(vec![10, 20, 30])
        .with_position_delete_files(vec![position_delete_file()]);
    let source = source_from_file(file);

    let plan = plan_local_parquet_source_chunks(
        NodeId::from_index_for_internal_use(3),
        &source,
        SourceChunkPlanningPolicy::for_position_delete_scan_limit(),
    )
    .unwrap();

    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
    assert_eq!(
        source_chunk_row_position_range(&plan.chunks[1]).unwrap(),
        Some(SourceChunkRowPositionRange { start: 10, end: 30 })
    );
}

#[test]
fn position_delete_source_scan_limit_does_not_raw_trim_chunk_plan() {
    let mut source = source_from_file(
        LocalParquetFile::new("/tmp/position-delete-source.parquet")
            .with_row_group_bytes(vec![100, 200, 300])
            .with_projected_row_group_bytes(vec![50, 100, 150], vec![100, 200, 300])
            .with_pure_filter_row_group_bytes(vec![0, 0, 0], vec![0, 0, 0])
            .with_row_group_rows(vec![10, 20, 30])
            .with_position_delete_files(vec![position_delete_file()]),
    );
    source.scan_limit = Some(1);

    let plan = plan_local_parquet_source_chunks(
        NodeId::from_index_for_internal_use(3),
        &source,
        SourceChunkPlanningPolicy::for_position_delete_scan_limit(),
    )
    .unwrap();

    assert_eq!(plan.chunk_count(), 3);
    assert!(plan.chunks.iter().all(|chunk| chunk.scan_limit_rows.is_none()));
}

#[test]
fn local_throughput_source_chunk_planning_rejects_partitioned_and_scan_limit_sources() {
    let mut partitioned_file = local_source_file();
    partitioned_file.partition_values = vec![LiteralValue::Int64(1)];
    let partitioned = source_from_file(partitioned_file);
    let mut scan_limited = local_source();
    scan_limited.scan_limit = Some(10);
    let policy = bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1);

    assert_eq!(
        SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &partitioned,
            policy,
            SourceChunkPlanningMode::LocalParquetThroughput,
        )
        .max_row_groups_per_chunk,
        1
    );
    assert_eq!(
        SourceChunkPlanningPolicy::from_memory_policy_with_mode(
            &scan_limited,
            policy,
            SourceChunkPlanningMode::LocalParquetThroughput,
        )
        .max_row_groups_per_chunk,
        1
    );
}

#[test]
fn source_chunk_batches_group_adjacent_compatible_chunks_within_byte_target() {
    let chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 1, 20), batchable_chunk(2, 2, 30)];

    let batches = plan_source_chunk_batches(&chunks, Some(35));

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(batches[0].estimated_source_read_bytes, Some(30));
    assert_eq!(batches[1].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn source_chunk_batches_group_adjacent_row_group_chunks_from_same_file() {
    let mut chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 0, 20), batchable_chunk(2, 0, 30)];
    chunks[0].row_group_indexes = vec![0];
    chunks[1].row_group_indexes = vec![1];
    chunks[2].row_group_indexes = vec![2];

    let batches = plan_source_chunk_batches(&chunks, Some(35));

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(batches[0].estimated_source_read_bytes, Some(30));
    assert_eq!(batches[1].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn source_chunk_batches_keep_ineligible_chunks_singleton() {
    let mut read_all = batchable_chunk(1, 1, 20);
    read_all.reads_all_row_groups = true;
    read_all.row_group_indexes.clear();
    let remote_direct = remote_direct_require_batchable_chunk(2, 2, 40);

    let batches = plan_source_chunk_batches(
        &[batchable_chunk(0, 0, 10), read_all, remote_direct, batchable_chunk(3, 3, 50)],
        None,
    );

    assert_eq!(batches.len(), 4);
    assert!(batches.iter().all(SourceChunkBatch::is_singleton));
    assert_eq!(batches[2].source_batching_reason, "incompatible_read_strategy");
}

#[test]
fn source_chunk_batches_group_adjacent_remote_direct_chunks() {
    let chunks = vec![
        remote_direct_require_batchable_chunk(0, 0, 10),
        remote_direct_require_batchable_chunk(1, 1, 20),
        remote_direct_require_batchable_chunk(2, 2, 30),
    ];

    let batches = plan_source_chunk_batches(&chunks, Some(35));

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(batches[0].estimated_source_read_bytes, Some(30));
    assert_eq!(batches[1].chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn annotate_source_chunk_batching_reasons_applies_source_level_gate() {
    let mut chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 1, 20)];

    annotate_source_chunk_batching_reasons(&mut chunks, Some("source_batching_disabled"), Some(35));

    assert_eq!(
        chunks.iter().map(|chunk| chunk.source_batching_reason.as_deref()).collect::<Vec<_>>(),
        vec![Some("source_batching_disabled"), Some("source_batching_disabled")]
    );
}

#[test]
fn annotate_source_chunk_batching_reasons_uses_planned_batch_reasons() {
    let mut chunks = vec![batchable_chunk(0, 0, 10), batchable_chunk(1, 1, 20), batchable_chunk(2, 2, 30)];

    annotate_source_chunk_batching_reasons(&mut chunks, None, Some(35));

    assert_eq!(
        chunks.iter().map(|chunk| chunk.source_batching_reason.as_deref()).collect::<Vec<_>>(),
        vec![Some("batched"), Some("batched"), Some("trailing_singleton")]
    );
}

#[test]
fn source_chunk_batch_split_for_retry_bisects_and_recomputes_estimates() {
    let batch = SourceChunkBatch {
        batch_index: 0,
        chunks: vec![
            batchable_chunk(0, 0, 10),
            batchable_chunk(1, 1, 20),
            batchable_chunk(2, 2, 30),
            batchable_chunk(3, 3, 40),
            batchable_chunk(4, 4, 50),
        ],
        estimated_source_read_bytes: Some(150),
        source_batching_reason: "batched".to_string(),
    };

    let (left, right) = batch.split_for_retry().expect("multi-chunk batch should split");

    assert_eq!(left.batch_index, 0);
    assert_eq!(left.chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(left.estimated_source_read_bytes, Some(30));
    assert_eq!(right.batch_index, 2);
    assert_eq!(right.chunks.iter().map(|chunk| chunk.chunk_index).collect::<Vec<_>>(), vec![2, 3, 4]);
    assert_eq!(right.estimated_source_read_bytes, Some(120));
}

#[test]
fn source_chunk_batch_split_for_retry_rejects_singletons() {
    let batch = SourceChunkBatch {
        batch_index: 0,
        chunks: vec![batchable_chunk(0, 0, 10)],
        estimated_source_read_bytes: Some(10),
        source_batching_reason: "trailing_singleton".to_string(),
    };

    assert!(batch.split_for_retry().is_none());
}

#[test]
fn source_chunk_detects_whole_known_file_row_group_coverage() {
    let mut chunk = batchable_chunk(0, 0, 10);
    chunk.source_file_row_groups = Some(vec![0, 1, 2]);
    chunk.row_group_indexes = vec![2, 0, 1];

    assert!(chunk.selects_all_known_file_row_groups());

    chunk.row_group_indexes = vec![0, 2];
    assert!(!chunk.selects_all_known_file_row_groups());

    chunk.row_group_indexes = vec![0, 1, 2];
    chunk.source_file_row_groups = None;
    chunk.source_file_row_group_rows = Some(vec![10, 10, 10]);
    chunk.source_file_row_group_bytes = Some(vec![1, 1, 1]);
    chunk.source_file_projected_compressed_bytes = Some(vec![1, 1, 1]);
    chunk.source_file_projected_uncompressed_bytes = Some(vec![1, 1, 1]);
    chunk.source_file_pure_filter_compressed_bytes = Some(vec![0, 0, 0]);
    chunk.source_file_pure_filter_uncompressed_bytes = Some(vec![0, 0, 0]);
    assert!(chunk.selects_all_known_file_row_groups());

    chunk.row_group_indexes = vec![0, 2];
    assert!(!chunk.selects_all_known_file_row_groups());
}
