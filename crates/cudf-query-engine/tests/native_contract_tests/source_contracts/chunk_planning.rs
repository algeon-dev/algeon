use super::*;

#[test]
fn source_chunk_planning_preserves_selected_row_groups_and_file_order() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet").with_file_size_bytes(300).with_row_groups(vec![0, 2, 4]),
            LocalParquetFile::new("/tmp/part-001.parquet").with_file_size_bytes(200).with_row_groups(vec![1, 3]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 2)).unwrap();

    assert_eq!(plan.source_node_id, source_id);
    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.row_group_count(), 5);
    assert!(!plan.row_group_metadata_available);
    assert!(plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 2);
    assert_eq!(plan.chunks[0].chunk_index, 0);
    assert_eq!(plan.chunks[0].file_index, 0);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 2]);
    assert_eq!(plan.chunks[0].known_file_bytes, Some(300));
    assert_eq!(plan.chunks[0].known_row_group_bytes, None);
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(200));
    assert_eq!(plan.chunks[0].read_columns, vec!["key".to_string()]);
    assert!(!plan.chunks[0].reads_all_row_groups);
    assert_eq!(plan.chunks[1].file_index, 0);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![4]);
    assert_eq!(plan.chunks[1].estimated_chunk_bytes, Some(100));
    assert_eq!(plan.chunks[2].file_index, 1);
    assert_eq!(plan.chunks[2].row_group_indexes, vec![1, 3]);
    assert_eq!(plan.chunks[2].estimated_chunk_bytes, Some(200));
}

#[test]
fn source_chunk_planning_uses_contract_row_group_bytes_without_file_metadata() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_file_size_bytes(600)
                .with_row_groups(vec![0, 2])
                .with_row_group_bytes(vec![100, 200, 300]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 2);
    assert!(plan.row_group_metadata_available);
    assert!(!plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 0);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].known_row_group_bytes, Some(100));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[1].known_row_group_bytes, Some(300));
}

#[test]
fn source_chunk_planning_uses_projected_uncompressed_bytes_before_read_bytes() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_file_size_bytes(4_000)
                .with_row_groups(vec![0, 1, 2, 3])
                .with_row_group_bytes(vec![1_000, 1_000, 1_000, 1_000])
                .with_projected_row_group_bytes(vec![50, 50, 50, 50], vec![100, 100, 100, 100])
                .with_pure_filter_row_group_bytes(vec![100, 100, 100, 100], vec![400, 400, 400, 400])
                .with_row_group_rows(vec![10, 10, 10, 10]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[0].known_row_group_bytes, Some(2_000));
    assert_eq!(plan.chunks[0].known_projected_compressed_bytes, Some(100));
    assert_eq!(plan.chunks[0].known_projected_uncompressed_bytes, Some(200));
    assert_eq!(plan.chunks[0].known_pure_filter_compressed_bytes, Some(200));
    assert_eq!(plan.chunks[0].known_pure_filter_uncompressed_bytes, Some(800));
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(200));
    assert_eq!(plan.chunks[0].estimated_source_read_bytes, Some(1_000));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2, 3]);
    assert_eq!(plan.chunks[1].estimated_chunk_bytes, Some(200));
}

#[test]
fn source_chunk_planning_applies_scan_limit_to_planned_chunks() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_file_size_bytes(4_000)
                .with_row_groups(vec![0, 1, 2, 3])
                .with_row_group_bytes(vec![1_000, 1_000, 1_000, 1_000])
                .with_projected_row_group_bytes(vec![50, 50, 50, 50], vec![100, 100, 100, 100])
                .with_row_group_rows(vec![10, 10, 10, 10]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    )
    .with_scan_limit(Some(25));
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[0].known_row_count, Some(20));
    assert_eq!(plan.chunks[0].scan_limit_rows, None);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2, 3]);
    assert_eq!(plan.chunks[1].known_row_count, Some(5));
    assert_eq!(plan.chunks[1].scan_limit_rows, Some(5));
}

#[test]
fn source_chunk_planning_keeps_row_group_limit_as_ceiling_for_projected_bytes() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_row_groups(vec![0, 1, 2])
                .with_row_group_bytes(vec![1_000, 1_000, 1_000])
                .with_projected_row_group_bytes(vec![1, 1, 1], vec![1, 1, 1])
                .with_row_group_rows(vec![10, 10, 10]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 2)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
}

#[test]
fn source_chunk_planning_allows_single_oversized_projected_row_group() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_row_groups(vec![0, 1])
                .with_row_group_bytes(vec![2_000, 100])
                .with_projected_row_group_bytes(vec![500, 10], vec![500, 10])
                .with_row_group_rows(vec![10, 10]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(100), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(500));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
}

#[test]
fn source_chunk_planning_carries_zero_column_count_row_counts() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/part-000.parquet")
            .with_file_size_bytes(600)
            .with_row_groups(vec![0, 2])
            .with_row_group_bytes(vec![100, 200, 300])
            .with_row_group_rows(vec![5, 7, 11]),
    ]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 2);
    assert!(plan.chunks.iter().all(|chunk| chunk.zero_column_count));
    assert!(plan.chunks.iter().all(|chunk| chunk.read_columns.is_empty()));
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].known_row_count, Some(5));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[1].known_row_count, Some(11));
}

#[test]
fn source_chunk_planning_packs_zero_column_count_by_ceiling_when_projected_bytes_are_zero() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/part-000.parquet")
            .with_file_size_bytes(600)
            .with_row_groups(vec![0, 1, 2])
            .with_row_group_bytes(vec![0, 0, 0])
            .with_projected_row_group_bytes(vec![0, 0, 0], vec![0, 0, 0])
            .with_row_group_rows(vec![5, 7, 11]),
    ]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1), 2)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[0].known_row_count, Some(12));
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(0));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[1].known_row_count, Some(11));
}

#[test]
fn source_chunk_planning_expands_all_row_groups_from_parquet_metadata() {
    let dir = unique_temp_dir("cudf-query-engine-source-chunks");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 2)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 3);
    assert!(plan.row_group_metadata_available);
    assert!(!plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 0);
    assert_eq!(plan.metadata_load_count, 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[0].stable_id, path.display().to_string());
    assert_eq!(plan.chunks[0].source_file_row_group_rows, Some(vec![2, 2, 2]));
    assert!(plan.chunks.iter().all(|chunk| chunk.known_row_group_bytes.unwrap() > 0));
    assert!(plan.chunks.iter().all(|chunk| chunk.source_file_row_group_bytes.as_ref().unwrap().len() == 3));
    assert!(plan.chunks.iter().all(|chunk| !chunk.reads_all_row_groups));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_reuses_footer_metadata_cache() {
    let dir = unique_temp_dir("cudf-query-engine-source-chunk-cache");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(vec![LocalParquetFile::new(&path)], schema, vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let mut metadata_cache = SourceMetadataCache::default();

    let first = plan_local_parquet_source_chunks_with_metadata_cache(
        source_id,
        &source,
        SourceChunkPlanningPolicy::new(None, 2),
        &mut metadata_cache,
    )
    .unwrap();
    let second = plan_local_parquet_source_chunks_with_metadata_cache(
        source_id,
        &source,
        SourceChunkPlanningPolicy::new(None, 2),
        &mut metadata_cache,
    )
    .unwrap();

    assert_eq!(metadata_cache.cached_footer_count(), 1);
    assert_eq!(first.metadata_load_count, 1);
    assert_eq!(second.metadata_load_count, 0);
    assert_eq!(second.metadata_load_time, std::time::Duration::ZERO);
    assert_eq!(first.chunks, second.chunks);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_uses_single_all_row_groups_chunk_when_metadata_is_unavailable() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/metadata-unavailable.parquet").with_file_size_bytes(4096)],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 2)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.row_group_count(), 0);
    assert!(!plan.row_group_metadata_available);
    assert!(plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 1);
    assert!(plan.chunks[0].reads_all_row_groups);
    assert!(plan.chunks[0].row_group_indexes.is_empty());
    assert_eq!(plan.chunks[0].known_file_bytes, Some(4096));
    assert_eq!(plan.chunks[0].known_row_group_bytes, None);
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(4096));
}

#[test]
fn source_chunk_planning_honors_max_source_chunk_bytes_when_row_group_sizes_are_known() {
    let dir = unique_temp_dir("cudf-query-engine-byte-source-chunks");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(vec![LocalParquetFile::new(&path)], schema, vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1), 128)).unwrap();

    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
    assert_eq!(plan.chunks[2].row_group_indexes, vec![2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_rejects_selected_row_groups_outside_metadata() {
    let dir = unique_temp_dir("cudf-query-engine-source-chunk-rejects");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 4, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_groups(vec![0, 2])],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let error = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::default()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_group_position"), Some("1"));
    assert_eq!(error.fact_value("row_group"), Some("2"));
    assert_eq!(error.fact_value("max_row_group"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

fn write_row_grouped_parquet(path: &Path, row_count: usize, rows_per_group: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let values = (0..row_count).map(|value| value as i32).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
