use std::{
    fs::File,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use super::*;

#[test]
fn source_task_memory_model_derives_chunk_target_from_live_budget_slots() {
    let memory_policy = MemoryControlPolicy::Bounded {
        device_budget_bytes: 2_000,
        max_source_chunk_bytes: 1_600,
        max_row_groups_per_chunk: 512,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    };

    let aggregate_model = SourceTaskMemoryModel::aggregate_pipeline(true);
    assert_eq!(aggregate_model.source_read_slots(), 1);
    assert_eq!(aggregate_model.source_adjacent_operator_slots(), 3);
    assert_eq!(aggregate_model.downstream_operator_slots(), 1);
    assert_eq!(aggregate_model.retained_output_slots(), 1);
    assert_eq!(aggregate_model.source_read_preflight_slack_slots(), 1);
    assert_eq!(aggregate_model.live_slot_count(), 7);
    let aggregate = SourceChunkPlanningPolicy::for_source_task(memory_policy, aggregate_model);
    assert_eq!(aggregate.max_source_chunk_bytes, Some(285));
    assert_eq!(aggregate.max_row_groups_per_chunk, 512);

    let join_model = SourceTaskMemoryModel::join_leaf_stream(true);
    assert_eq!(join_model.source_read_slots(), 1);
    assert_eq!(join_model.source_adjacent_operator_slots(), 1);
    assert_eq!(join_model.downstream_operator_slots(), 1);
    assert_eq!(join_model.retained_output_slots(), 2);
    assert_eq!(join_model.source_read_preflight_slack_slots(), 1);
    assert_eq!(join_model.live_slot_count(), 6);
    let join = SourceChunkPlanningPolicy::for_source_task(memory_policy, join_model);
    assert_eq!(join.max_source_chunk_bytes, Some(333));
    assert_eq!(join.max_row_groups_per_chunk, 512);
}

#[test]
fn source_task_memory_model_preserves_configured_cap_when_budget_slots_allow_it() {
    let memory_policy = MemoryControlPolicy::Bounded {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1_600,
        max_row_groups_per_chunk: 512,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    };

    let policy =
        SourceChunkPlanningPolicy::for_source_task(memory_policy, SourceTaskMemoryModel::aggregate_pipeline(false));
    assert_eq!(policy.max_source_chunk_bytes, Some(1_600));
    assert_eq!(policy.max_row_groups_per_chunk, 512);
}

#[test]
fn source_task_memory_model_preserves_configured_cap_when_retry_admission_owns_an_oversized_source_read() {
    let memory_policy = MemoryControlPolicy::Bounded {
        device_budget_bytes: 1_000,
        max_source_chunk_bytes: 1_600,
        max_row_groups_per_chunk: 512,
        max_retry_attempts: 2,
        min_retry_chunk_row_groups: 1,
    };

    let policy =
        SourceChunkPlanningPolicy::for_source_task(memory_policy, SourceTaskMemoryModel::aggregate_pipeline(true));
    assert_eq!(policy.max_source_chunk_bytes, Some(1_600));
    assert_eq!(policy.max_row_groups_per_chunk, 512);
}

#[test]
fn source_task_memory_model_leaves_unbounded_planning_unchanged() {
    let policy = SourceChunkPlanningPolicy::for_source_task(
        MemoryControlPolicy::Unbounded,
        SourceTaskMemoryModel::join_leaf_stream(true),
    );

    assert_eq!(policy, SourceChunkPlanningPolicy::default());
}

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
fn source_chunk_planning_uses_proportional_file_bytes_when_row_group_bytes_are_unknown() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_file_size_bytes(1_200)
                .with_row_groups(vec![0, 1, 2, 3]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(500), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 4);
    assert_eq!(plan.row_group_count(), 4);
    assert!(!plan.row_group_metadata_available);
    assert!(plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 1);
    for (index, chunk) in plan.chunks.iter().enumerate() {
        assert_eq!(chunk.row_group_indexes, vec![index]);
        assert_eq!(chunk.known_row_group_bytes, None);
        assert_eq!(chunk.estimated_chunk_bytes, Some(300));
    }
}

#[test]
fn source_chunk_planning_uses_source_read_uncompressed_bytes_for_chunk_ceiling() {
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

    assert_eq!(plan.chunk_count(), 4);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].known_row_group_bytes, Some(1_000));
    assert_eq!(plan.chunks[0].known_projected_compressed_bytes, Some(50));
    assert_eq!(plan.chunks[0].known_projected_uncompressed_bytes, Some(100));
    assert_eq!(plan.chunks[0].known_pure_filter_compressed_bytes, Some(100));
    assert_eq!(plan.chunks[0].known_pure_filter_uncompressed_bytes, Some(400));
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(100));
    assert_eq!(plan.chunks[0].estimated_source_read_bytes, Some(500));
    let minimum_read_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
    assert_eq!(plan.chunks[0].exact_source_read_preflight_bytes, Some(minimum_read_budget));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
    assert_eq!(plan.chunks[1].estimated_chunk_bytes, Some(100));
}

#[test]
fn source_chunk_planning_does_not_double_count_overlapping_projected_filter_columns() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_row_groups(vec![0, 1])
                .with_row_group_bytes(vec![1_000, 1_000])
                .with_projected_row_group_bytes(vec![50, 50], vec![100, 100])
                .with_pure_filter_row_group_bytes(vec![50, 50], vec![100, 100])
                .with_row_group_rows(vec![10, 10]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.chunks[0].known_projected_uncompressed_bytes, Some(200));
    assert_eq!(plan.chunks[0].known_pure_filter_uncompressed_bytes, Some(200));
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(200));
    assert_eq!(plan.chunks[0].estimated_source_read_bytes, Some(200));
    let minimum_read_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
    assert_eq!(plan.chunks[0].exact_source_read_preflight_bytes, Some(minimum_read_budget));
}

#[test]
fn source_chunk_planning_uses_exact_required_budget_when_it_exceeds_minimum_source_read_budget() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int64, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![1_000])
                .with_projected_row_group_bytes(vec![100], vec![100])
                .with_row_group_rows(vec![20_000]),
        ],
        Arc::clone(&schema),
        vec!["payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.chunks[0].estimated_source_read_bytes, Some(100));
    assert_eq!(plan.chunks[0].exact_source_read_preflight_bytes, Some(160_000));
}

#[test]
fn source_chunk_planning_leaves_variable_width_source_reads_on_fallback_contract() {
    let schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![1_000])
                .with_projected_row_group_bytes(vec![100], vec![400])
                .with_row_group_rows(vec![10]),
        ],
        Arc::clone(&schema),
        vec!["payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.chunks[0].estimated_source_read_bytes, Some(400));
    assert_eq!(plan.chunks[0].exact_source_read_preflight_bytes, None);
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
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[1].known_row_count, Some(5));
    assert_eq!(plan.chunks[1].known_row_group_bytes, Some(1_000));
    assert_eq!(plan.chunks[1].known_projected_compressed_bytes, Some(50));
    assert_eq!(plan.chunks[1].known_projected_uncompressed_bytes, Some(100));
    assert_eq!(plan.chunks[1].scan_limit_rows, Some(5));
}

#[test]
fn source_chunk_planning_does_not_apply_scan_limit_prefix_before_equality_deletes() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
        "delete-000",
        ResolvedParquetSource::local_kvikio("/tmp/delete-000.parquet"),
        vec![1],
        vec!["key".to_string()],
    )
    .unwrap();
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_file_size_bytes(4_000)
                .with_row_groups(vec![0, 1, 2, 3])
                .with_row_group_bytes(vec![1_000, 1_000, 1_000, 1_000])
                .with_projected_row_group_bytes(vec![50, 50, 50, 50], vec![100, 100, 100, 100])
                .with_row_group_rows(vec![10, 10, 10, 10])
                .with_equality_delete_files(vec![delete_file]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    )
    .with_scan_limit(Some(5));
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[0].known_row_count, Some(20));
    assert_eq!(plan.chunks[0].scan_limit_rows, None);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2, 3]);
    assert_eq!(plan.chunks[1].known_row_count, Some(20));
    assert_eq!(plan.chunks[1].scan_limit_rows, None);
}

#[test]
fn source_chunk_planning_trims_scan_limit_before_large_chunk_packing() {
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
    .with_scan_limit(Some(5));
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(10_000), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.row_group_count(), 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].known_row_count, Some(5));
    assert_eq!(plan.chunks[0].known_row_group_bytes, Some(1_000));
    assert_eq!(plan.chunks[0].known_projected_uncompressed_bytes, Some(100));
    assert_eq!(plan.chunks[0].scan_limit_rows, Some(5));
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
    let dir = unique_temp_dir("nexus-query-engine-source-chunks");
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
fn source_chunk_planning_merges_supplied_row_group_bytes_with_footer_metadata() {
    let dir = unique_temp_dir("nexus-query-engine-mixed-source-footer-chunks");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let supplied_row_group_bytes = vec![111, 222, 333];
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(supplied_row_group_bytes.clone()),
        ],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 2)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 3);
    assert!(plan.row_group_metadata_available);
    assert_eq!(plan.metadata_unavailable_files, 0);
    assert_eq!(plan.metadata_load_count, 1);
    assert_eq!(plan.chunks[0].known_row_group_bytes, Some(333));
    assert_eq!(plan.chunks[1].known_row_group_bytes, Some(333));
    assert_eq!(plan.chunks[0].source_file_row_group_bytes, Some(supplied_row_group_bytes.clone()));
    assert_eq!(plan.chunks[1].source_file_row_group_bytes, Some(supplied_row_group_bytes));
    assert_eq!(plan.chunks[0].source_file_row_group_rows, Some(vec![2, 2, 2]));
    let projected_uncompressed = plan.chunks[0]
        .source_file_projected_uncompressed_bytes
        .as_ref()
        .expect("footer metadata should fill projected uncompressed bytes");
    assert_eq!(projected_uncompressed.len(), 3);
    assert!(projected_uncompressed.iter().all(|bytes| *bytes > 0));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_loads_warm_local_cache_footer_metadata() {
    let dir = unique_temp_dir("nexus-query-engine-local-cache-source-chunks");
    let path = dir.join("cache-entry.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let resolved_source = ResolvedParquetSource::local_cache_kvikio(
        "s3://bucket/table/part-0.parquet",
        path.display().to_string(),
        CredentialMode::SharedProcessEnv,
        LakehouseCacheStatus::Hit,
    );
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
        .unwrap()
        .with_file_size_bytes(fs::metadata(&path).unwrap().len());
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 2)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 3);
    assert!(plan.row_group_metadata_available);
    assert_eq!(plan.metadata_unavailable_files, 0);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[0].source_file_row_group_rows, Some(vec![2, 2, 2]));
    assert!(plan.chunks.iter().all(|chunk| !chunk.reads_all_row_groups));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_keeps_local_cache_miss_conservative() {
    let dir = unique_temp_dir("nexus-query-engine-local-cache-miss-source-chunks");
    let path = dir.join("missing-cache-entry.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let resolved_source = ResolvedParquetSource::local_cache_kvikio(
        "s3://bucket/table/part-0.parquet",
        path.display().to_string(),
        CredentialMode::SharedProcessEnv,
        LakehouseCacheStatus::FillBeforeRead,
    );
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source).unwrap().with_file_size_bytes(4096);
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 2)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.row_group_count(), 0);
    assert!(!plan.row_group_metadata_available);
    assert_eq!(plan.metadata_unavailable_files, 1);
    assert!(plan.chunks[0].reads_all_row_groups);
    assert!(plan.chunks[0].row_group_indexes.is_empty());
    assert_eq!(plan.chunks[0].source_file_row_group_rows, None);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dynamic_filter_small_in_list_prunes_source_row_groups_and_builds_reader_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-dynamic-filter-small-in-list");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_group_rows(vec![2, 2, 2])],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::ReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::SmallInListI64 { values: vec![2, 3], min: 2, max: 3 },
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();
    let plan =
        plan_local_parquet_source_chunks(source_id, &rewritten, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.row_group_count(), 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![1]);
    assert!(matches!(rewritten.reader_predicate, Some(Expr::InList { .. })));
    let stats = rewritten.diagnostics.selected_row_group_stats.as_ref().unwrap();
    assert_eq!(stats.selected_row_groups, 1);
    assert_eq!(stats.total_row_groups, 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dynamic_filter_row_group_pruning_uses_leaf_stats_for_root_column() {
    let dir = unique_temp_dir("nexus-query-engine-dynamic-filter-nested-stats");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nested.parquet");
    let schema = write_nested_key_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_group_rows(vec![1, 1])],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::ReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::SmallInListI64 { values: vec![5], min: 5, max: 5 },
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();
    let plan =
        plan_local_parquet_source_chunks(source_id, &rewritten, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.row_group_count(), 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![1]);
    let stats = rewritten.diagnostics.selected_row_group_stats.as_ref().unwrap();
    assert_eq!(stats.selected_row_groups, 1);
    assert_eq!(stats.total_row_groups, 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dynamic_filter_position_delete_source_prunes_row_groups_without_reader_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-dynamic-filter-position-delete");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 8, 2);
    let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-0",
        ResolvedParquetSource::local_kvikio(dir.join("delete.parquet").display().to_string()),
    )
    .unwrap();
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_row_group_rows(vec![2, 2, 2, 2])
                .with_position_delete_files(vec![delete_file]),
        ],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::ReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::SmallInListI64 { values: vec![0, 5], min: 0, max: 5 },
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();
    let plan =
        plan_local_parquet_source_chunks(source_id, &rewritten, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.row_group_count(), 3);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1, 2]);
    assert!(rewritten.reader_predicate.is_none());
    let stats = rewritten.diagnostics.selected_row_group_stats.as_ref().unwrap();
    assert_eq!(stats.selected_row_groups, 3);
    assert_eq!(stats.total_row_groups, 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dynamic_filter_empty_payload_records_explicit_empty_row_group_selection() {
    let dir = unique_temp_dir("nexus-query-engine-dynamic-filter-empty");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_group_rows(vec![2, 2, 2])],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::ReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::Empty,
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();
    let plan =
        plan_local_parquet_source_chunks(source_id, &rewritten, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.chunk_count(), 0);
    assert_eq!(plan.row_group_count(), 0);
    assert_eq!(rewritten.files[0].row_groups.as_deref(), Some([].as_slice()));
    assert_eq!(rewritten.scan_limit, Some(0));
    let stats = rewritten.diagnostics.selected_row_group_stats.as_ref().unwrap();
    assert_eq!(stats.selected_row_groups, 0);
    assert_eq!(stats.total_row_groups, 3);
    assert_eq!(stats.selected_rows, Some(0));
    assert_eq!(stats.total_rows, Some(6));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_preserves_explicit_empty_row_group_selection() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/empty-row-group-selection.parquet")
                .with_row_groups(Vec::new())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.chunk_count(), 0);
    assert_eq!(plan.row_group_count(), 0);
    let stats = source.diagnostics.selected_row_group_stats.as_ref().unwrap();
    assert_eq!(stats.selected_row_groups, 0);
    assert_eq!(stats.total_row_groups, 3);
    assert_eq!(stats.selected_rows, Some(0));
    assert_eq!(stats.total_rows, Some(6));
}

#[test]
fn dynamic_filter_range_merges_with_static_reader_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-dynamic-filter-merged-range");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let static_predicate = Expr::binary(Expr::column("key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(1)));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_group_rows(vec![2, 2, 2])],
        schema,
        vec!["key".to_string()],
    )
    .with_reader_predicate(static_predicate);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::MergedReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::RangeI64 { min: 2, max: 3 },
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();
    let plan =
        plan_local_parquet_source_chunks(source_id, &rewritten, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.row_group_count(), 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![1]);
    let predicate = rewritten.reader_predicate.as_ref().unwrap().to_string();
    assert!(predicate.contains("AND"));
    assert!(predicate.contains(">= 1"));
    assert!(predicate.contains(">= 2"));
    assert!(predicate.contains("<= 3"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dynamic_filter_range_does_not_create_unsupported_merged_reader_predicate() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let static_predicate = Expr::binary(
        Expr::column("amount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 100, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::object_store_uri("s3://test-bucket/input.parquet").unwrap()],
        schema,
        vec!["key".to_string(), "amount".to_string()],
    )
    .with_reader_predicate(static_predicate);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::MergedReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::RangeI64 { min: 2, max: 3 },
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();

    let predicate = rewritten.reader_predicate.as_ref().expect("static reader predicate should remain");
    assert!(
        rewritten
            .reader_predicate_is_supported(predicate, nexus_query_engine::rapids_backend::rapids_analysis_context())
    );
}

#[test]
fn dynamic_filter_hash_set_prunes_row_groups_without_reader_predicate_lowering() {
    let dir = unique_temp_dir("nexus-query-engine-dynamic-filter-hash-set");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_group_rows(vec![2, 2, 2])],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let route = dynamic_filter_source_route(
        source_id,
        nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering::ReaderPredicate,
    );
    let mut filters = nexus_query_engine::dynamic_filter::DynamicFilterSet::new();
    filters.insert(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilter::HashSetContainsI64 { values: vec![2, 3], min: 2, max: 3 },
    );

    let rewritten = nexus_query_engine::dynamic_filter::apply_dynamic_filters_to_local_parquet_source(
        &source,
        &[&route],
        &filters,
        nexus_query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap();
    let plan =
        plan_local_parquet_source_chunks(source_id, &rewritten, SourceChunkPlanningPolicy::new(None, 8)).unwrap();

    assert_eq!(plan.row_group_count(), 1);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![1]);
    assert_eq!(rewritten.reader_predicate, None);
    let stats = rewritten.diagnostics.selected_row_group_stats.as_ref().unwrap();
    assert_eq!(stats.selected_row_groups, 1);
    assert_eq!(stats.total_row_groups, 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_reuses_footer_metadata_cache() {
    let dir = unique_temp_dir("nexus-query-engine-source-chunk-cache");
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
    let dir = unique_temp_dir("nexus-query-engine-byte-source-chunks");
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
    let dir = unique_temp_dir("nexus-query-engine-source-chunk-rejects");
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

fn write_nested_key_parquet(path: &Path) -> Arc<Schema> {
    let payload_a = Arc::new(Field::new("a", DataType::Int32, false));
    let payload_b = Arc::new(Field::new("b", DataType::Int32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Struct(Fields::from(vec![payload_a.clone(), payload_b.clone()])), false),
        Field::new("key", DataType::Int32, false),
    ]));
    let payload_a_values = Arc::new(Int32Array::from(vec![0, 0])) as ArrayRef;
    let payload_b_values = Arc::new(Int32Array::from(vec![5, 100])) as ArrayRef;
    let payload =
        Arc::new(StructArray::from(vec![(payload_a, payload_a_values), (payload_b, payload_b_values)])) as ArrayRef;
    let key = Arc::new(Int32Array::from(vec![100, 5])) as ArrayRef;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![payload, key]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn dynamic_filter_source_route(
    source_id: nexus_query_engine::plan::NodeId,
    lowering: nexus_query_engine::dynamic_filter::DynamicFilterConsumerLowering,
) -> nexus_query_engine::dynamic_filter::DynamicFilterRoute {
    nexus_query_engine::dynamic_filter::DynamicFilterRoute::new(
        nexus_query_engine::dynamic_filter::DynamicFilterId::new(0),
        nexus_query_engine::dynamic_filter::DynamicFilterProducer::HashJoinBuild {
            join_node_id: source_id,
            build_side: JoinSide::Right,
            build_input: source_id,
        },
        nexus_query_engine::dynamic_filter::DynamicFilterConsumer::LocalParquetSource { source_node_id: source_id },
        vec!["key".to_string()],
        vec!["key".to_string()],
        vec![
            nexus_query_engine::dynamic_filter::DynamicFilterKind::Range,
            nexus_query_engine::dynamic_filter::DynamicFilterKind::SmallInList,
        ],
        lowering,
    )
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
