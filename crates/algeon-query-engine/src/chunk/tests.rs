use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int32Array, Int64Array, ListArray, StringArray},
    datatypes::{DataType, Field, Int32Type, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_planning::{
    backend::BackendCapabilities,
    chunk::{
        LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK, SourceChunkPlanningOptions, SourceChunkPlanningPolicy,
        SourceTaskMemoryModel, decimal::validate_supplied_reader_decimal_types_match_footer,
    },
    error::QueryErrorCode,
    execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode},
    expr::{BinaryOp, Expr, LiteralValue},
    plan::NodeId,
    source::{
        CredentialMode, LakehouseCacheReadMode, LakehouseCacheStatus, LocalParquetFile, LocalParquetSource,
        ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, ResolvedParquetSource,
    },
};

use super::*;
use crate::{
    rapids_backend::rapids_backend,
    source_chunk::metadata::{SourceMetadataCacheKey, local_parquet_footer_metadata_path, sidecar_content_identity},
};

fn batchable_chunk(index: usize, file_index: usize, bytes: u64) -> SourceChunk {
    SourceChunk {
        source_node_id: NodeId::from_index_for_test(1),
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
        estimated_decoded_bytes: None,
        estimated_column_elements: Some(1),
        max_column_elements: u64::MAX,
        exact_source_read_preflight_bytes: Some(bytes),
        source_file_row_groups: None,
        source_file_row_group_bytes: Some(vec![bytes]),
        source_file_projected_compressed_bytes: Some(vec![bytes]),
        source_file_projected_uncompressed_bytes: Some(vec![bytes]),
        source_file_pure_filter_compressed_bytes: Some(vec![0]),
        source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
        source_file_row_group_decoded_bytes: None,
        source_file_row_group_column_elements: None,
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
    MemoryControlPolicy {
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
fn source_chunks_bound_decoded_dictionary_strings_by_the_byte_target() {
    let path = std::env::temp_dir().join(format!(
        "query-engine-decoded-chunk-target-{}-{}.parquet",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let rows = 8 * 4096;
    let labels = ['a', 'b', 'c'].map(|fill| fill.to_string().repeat(200));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(StringArray::from_iter_values((0..rows).map(|row| labels[row % 3].as_str()))),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_column_dictionary_enabled("key".into(), false)
        .set_max_row_group_row_count(Some(4096))
        .build();
    let mut writer = ArrowWriter::try_new(fs::File::create(&path).unwrap(), Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path.clone())],
        schema,
        vec!["key".to_string(), "label".to_string()],
    );
    // Each row group decodes to 4096 * (8 key + 8 offset + 200 label) bytes,
    // so two fit the target; its dictionary-encoded footer bytes all fit.
    let target = 2 * 1024 * 1024;

    let plan = plan_local_parquet_source_chunks(
        NodeId::from_index_for_test(3),
        &source,
        SourceChunkPlanningPolicy::new(Some(target), LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK),
    )
    .unwrap();

    assert!(plan.chunks.iter().map(|chunk| chunk.known_projected_uncompressed_bytes.unwrap()).sum::<u64>() < target);
    assert_eq!(plan.chunks.iter().map(|chunk| chunk.row_group_indexes.len()).collect::<Vec<_>>(), vec![2, 2, 2, 2]);
    for chunk in &plan.chunks {
        assert_eq!(chunk.estimated_decoded_bytes, Some(2 * 4096 * (8 + 8 + 200)));
    }
    fs::remove_file(path).unwrap();
}

const VECTOR_ROW_GROUP_ROWS: u64 = 200_000;
const VECTOR_WIDTH: i32 = 2048;

/// A 2048-wide float vector source like an embedding table: each 200,000-row
/// row group decodes to 409,600,000 list values, within the backend's
/// per-column element limit, and five row groups together are too.
fn vector_source(row_group_rows: Vec<u64>) -> LocalParquetSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "fp",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), VECTOR_WIDTH),
            true,
        ),
    ]));
    let file = LocalParquetFile::new("/tmp/missing-vector-source.parquet")
        .with_row_groups((0..row_group_rows.len()).collect())
        .with_row_group_bytes(vec![1024; row_group_rows.len()])
        .with_row_group_rows(row_group_rows);
    LocalParquetSource::from_files(vec![file], schema, vec!["id".to_string(), "fp".to_string()])
}

fn assert_within_element_limit(chunks: &[SourceChunk]) {
    let limit = rapids_backend().max_column_elements();
    for chunk in chunks {
        assert_eq!(chunk.max_column_elements, limit);
        assert!(chunk.estimated_column_elements.is_some_and(|elements| elements <= limit), "{chunk:?}");
    }
}

#[test]
fn vector_source_chunks_and_batches_stay_within_the_backend_column_element_limit() {
    let source = vector_source(vec![VECTOR_ROW_GROUP_ROWS; 15]);
    let row_group_elements = VECTOR_ROW_GROUP_ROWS * VECTOR_WIDTH as u64;

    let widened =
        plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, SourceChunkPlanningPolicy::default())
            .unwrap();
    assert_eq!(widened.chunks.iter().map(SourceChunk::row_group_count).collect::<Vec<_>>(), vec![5, 5, 5]);
    assert_within_element_limit(&widened.chunks);
    assert!(widened.chunks.iter().all(|chunk| chunk.estimated_column_elements == Some(5 * row_group_elements)));

    let single = plan_local_parquet_source_chunks(
        NodeId::from_index_for_test(3),
        &source,
        SourceChunkPlanningPolicy::new(None, 1),
    )
    .unwrap();
    assert_eq!(single.chunk_count(), 15);
    let batches = plan_source_chunk_batches(&single.chunks, None);
    assert_eq!(batches.iter().map(SourceChunkBatch::chunk_count).collect::<Vec<_>>(), vec![5, 5, 5]);
    for batch in &batches {
        let batch_elements = batch.chunks.iter().filter_map(|chunk| chunk.estimated_column_elements).sum::<u64>();
        assert!(batch_elements <= rapids_backend().max_column_elements(), "{batch:?}");
    }
    let (left, right) = batches[0].clone().split_for_retry().expect("a five-chunk batch splits");
    assert_within_element_limit(&left.chunks);
    assert_within_element_limit(&right.chunks);
}

#[test]
fn vector_row_group_above_the_column_element_limit_fails_with_the_row_group_size_that_fits() {
    let source = vector_source(vec![VECTOR_ROW_GROUP_ROWS, 2_897_819]);

    let error =
        plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, SourceChunkPlanningPolicy::default())
            .expect_err("one row group of 2,897,819 vectors cannot decode into one backend column");

    assert_eq!(error.code(), QueryErrorCode::SizeOverflow);
    let fact = |key| error.wire_fact(key).map(|value| value.into_owned());
    assert_eq!(fact("column").as_deref(), Some("fp"));
    assert_eq!(fact("row_group").as_deref(), Some("1"));
    assert_eq!(fact("row_group_rows").as_deref(), Some("2897819"));
    assert_eq!(fact("element_count"), Some((2_897_819 * VECTOR_WIDTH as u64).to_string()));
    assert_eq!(fact("element_limit"), Some(rapids_backend().max_column_elements().to_string()));
    let max_row_group_rows = rapids_backend().max_column_elements() / VECTOR_WIDTH as u64;
    assert_eq!(fact("max_row_group_rows"), Some(max_row_group_rows.to_string()));
    assert!(fact("remedy").is_some_and(|remedy| remedy.contains(&format!("at most {max_row_group_rows} rows"))));
}

#[test]
fn variable_list_row_groups_without_footer_value_counts_are_read_alone() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        true,
    )]));
    let file = LocalParquetFile::new("/tmp/missing-list-source.parquet")
        .with_row_groups(vec![0, 1, 2])
        .with_row_group_bytes(vec![1024; 3])
        .with_row_group_rows(vec![10; 3]);
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["values".to_string()]);

    let plan =
        plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, SourceChunkPlanningPolicy::default())
            .unwrap();

    assert_eq!(plan.chunks.iter().map(SourceChunk::row_group_count).collect::<Vec<_>>(), vec![1, 1, 1]);
    assert!(plan.chunks.iter().all(|chunk| chunk.estimated_column_elements.is_none()));
    assert!(plan_source_chunk_batches(&plan.chunks, None).iter().all(SourceChunkBatch::is_singleton));
}

#[test]
fn variable_list_elements_come_from_footer_leaf_value_counts() {
    let path = std::env::temp_dir().join(format!(
        "query-engine-list-elements-{}-{}.parquet",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    ));
    let values = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2), Some(3)]),
        Some(vec![]),
        Some(vec![Some(4), Some(5), Some(6), Some(7), Some(8)]),
        None,
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("values", values.data_type().clone(), true),
    ]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(values)])
            .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(fs::File::create(&path).unwrap(), Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path.clone())],
        schema,
        vec!["key".to_string(), "values".to_string()],
    );

    let plan = plan_local_parquet_source_chunks(
        NodeId::from_index_for_test(3),
        &source,
        SourceChunkPlanningPolicy::new(None, 1),
    )
    .unwrap();

    // Each row group counts one leaf entry per value plus one per empty or
    // null list: 3 + 1 and 5 + 1.
    assert_eq!(
        plan.chunks.iter().map(|chunk| chunk.estimated_column_elements).collect::<Vec<_>>(),
        vec![Some(4), Some(6)]
    );
    assert_eq!(plan.chunks[0].source_file_row_group_column_elements, Some(vec![4, 6]));
    fs::remove_file(path).unwrap();
}

#[test]
fn adaptive_source_targets_follow_immutable_grant_and_explicit_targets() {
    let model = SourceTaskMemoryModel::aggregate_pipeline(true);
    for grant in [256 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024] {
        let adaptive = bounded_policy(grant, MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES, 1);
        let target = SourceChunkPlanningPolicy::for_source_task(adaptive, model);
        assert_eq!(target.max_source_chunk_bytes, Some(grant / model.live_slot_count()));
        assert_eq!(target.max_row_groups_per_chunk, 1);
        let source_target = SourceChunkPlanningPolicy::from(adaptive).max_source_chunk_bytes.unwrap();
        assert!(source_target > 0 && source_target < grant);
        let explicit = SourceChunkPlanningPolicy::for_source_task(bounded_policy(grant, 16 * 1024 * 1024, 2), model);
        assert_eq!(explicit.max_source_chunk_bytes, Some(16 * 1024 * 1024));
        assert_eq!(explicit.max_row_groups_per_chunk, 2);
    }
}

#[test]
fn local_throughput_source_chunk_planning_raises_row_group_cap_for_large_local_kvikio_source() {
    let source = local_source();
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(policy.max_source_chunk_bytes, Some(64 * 1024 * 1024));
}

#[test]
fn local_throughput_source_chunk_planning_uses_configured_expert_options() {
    let source = local_source();
    let options = SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(16);
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

    assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
    assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn local_throughput_source_chunk_planning_accepts_local_cache_reader_path() {
    let path = std::env::temp_dir().join(format!(
        "query-engine-local-cache-throughput-{}-{}.parquet",
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
        bounded_policy(64 * 1024 * 1024 * 1024, 64 * 1024 * 1024, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(policy.source_chunk_planning_reason(), "local_parquet_throughput_admitted");
    fs::remove_file(path).unwrap();
}

#[test]
fn remote_direct_metadata_path_respects_cache_read_bypass() {
    let path = std::env::temp_dir().join(format!(
        "query-engine-remote-direct-bypass-{}-{}.parquet",
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
        NodeId::from_index_for_test(7),
        &source,
        SourceChunkPlanningPolicy::new(None, 1),
    )
    .expect_err("remote decimal reader predicates need proven footer dtype metadata");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
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

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
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
fn local_throughput_source_chunk_planning_widens_chunks_under_a_small_grant() {
    // The source-byte target scales chunks with the grant, so a 24 GiB grant
    // still plans throughput chunks.
    let source = local_source();
    let policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode(
        &source,
        bounded_policy(24 * 1024 * 1024 * 1024, MemoryControlPolicy::DEFAULT_MAX_SOURCE_CHUNK_BYTES, 1),
        SourceChunkPlanningMode::LocalParquetThroughput,
    );

    assert_eq!(policy.source_chunk_planning_reason(), "local_parquet_throughput_admitted");
    assert_eq!(policy.max_row_groups_per_chunk, LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
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

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, policy).unwrap();
    assert_eq!(plan.chunk_count(), 2048);
    assert!(plan.chunks.iter().all(|chunk| chunk.row_group_count() == 1));
}

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

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, policy).unwrap();

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

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, policy).unwrap();

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

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, policy).unwrap();

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
        "query-engine-zero-column-count-footer-{}-{}.parquet",
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

    let plan = plan_local_parquet_source_chunks(NodeId::from_index_for_test(3), &source, policy).unwrap();

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

fn local_throughput_source_chunk_planning_rejects_partitioned_and_scan_limit_sources_case() {
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
fn local_throughput_source_chunk_planning_rejects_partitioned_and_scan_limit_sources() {
    local_throughput_source_chunk_planning_rejects_partitioned_and_scan_limit_sources_case();
    local_throughput_source_chunk_planning_rejects_object_store_sources();
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
