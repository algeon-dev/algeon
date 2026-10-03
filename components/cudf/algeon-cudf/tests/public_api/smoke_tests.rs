#[path = "../core/debug_tests.rs"]
mod core_debug_tests;
#[path = "../frame/dataframe/debug_tests.rs"]
mod dataframe_debug_tests;
#[path = "../frame/dataframe/display_tests.rs"]
mod dataframe_display_tests;
#[path = "../frame/contracts/high_level_reexport_tests.rs"]
mod high_level_reexport_tests;
#[path = "interop_canonical_contract_imports.rs"]
mod interop_canonical_contract_imports;
#[path = "../runtime/debug_tests.rs"]
mod runtime_debug_tests;
#[path = "../frame/series/display_tests.rs"]
mod series_display_tests;

#[test]
fn query_memory_manager_surface_is_publicly_importable() {
    use cudf::{
        AllocatorEvent, AllocatorEventAction, AllocatorEventCompleteness, AllocatorEventDomainIdentity,
        AllocatorEventRead, AllocatorEventRingBounds, DeviceMemoryFailureAllocationDomain, DeviceMemoryFailureInfo,
        DeviceMemoryFailureSource, DeviceMemoryManager, DeviceMemoryManagerFailureCallbackState, DeviceMemoryResource,
        ExecutionBudgetEnforcement, InstallSpec, QueryMemoryDomainIdentity, ReservationUsage, Runtime,
        StreamReservationLease,
    };

    let _ = cudf::ExecutionBudgetToken::from_shared_lease;
    let _ = DeviceMemoryResource::with_failure_callback;
    let _ = DeviceMemoryManager::allocator_event_ring_bounds;
    let _ = DeviceMemoryManager::read_allocator_event;
    let _ = DeviceMemoryResource::with_failure_callback_details;
    let _ = DeviceMemoryManager::install;
    let _: Option<InstallSpec> = None;
    let _ = DeviceMemoryManager::failure_callback_state_for_device;
    let _ = DeviceMemoryManager::options_for_device;
    let _: Option<DeviceMemoryFailureInfo> = None;
    let _ = DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id: 1, member_id: 1 };
    let _ = DeviceMemoryFailureSource::Physical;
    let _: Option<DeviceMemoryManagerFailureCallbackState> = None;
    let _: Option<AllocatorEvent> = None;
    let _: Option<AllocatorEventAction> = None;
    let _: Option<AllocatorEventCompleteness> = None;
    let _: Option<AllocatorEventDomainIdentity> = None;
    let _: Option<AllocatorEventRead> = None;
    let _: Option<AllocatorEventRingBounds> = None;
    let _: Option<QueryMemoryDomainIdentity> = None;
    let _: Option<DeviceMemoryManager> = None;
    let _: Option<StreamReservationLease> = None;
    let _: Option<ReservationUsage> = None;
    let _ = (Runtime::device_memory_manager, cudf::RuntimeBuilder::with_device_memory_manager);
    let _ = ExecutionBudgetEnforcement::QueryDomainMember;
}

#[test]
fn join_indices_and_hash_join_slice6_surface_is_publicly_importable() {
    use cudf::{
        DataFrame, DataFrameHashJoin, DataFrameHashJoinOptions, DataFrameProjectedCrossJoinPlan,
        ops::join::NullEquality,
    };

    let _options = DataFrameHashJoinOptions::new().null_equality(NullEquality::Equal);
    let _: Option<DataFrameHashJoin> = None;
    let _: Option<DataFrameProjectedCrossJoinPlan> = None;
    let _ = cudf::FrameOps::hash_join_pre_build_reservation_size_bytes::<[&str; 1], &str>;
    let _ = cudf::FrameOps::materialize_join_indices_reservation_size_bytes::<[&str; 1], &str>;
    let _ = DataFrame::projected_cross_join_plan;
    let _ = (
        DataFrameProjectedCrossJoinPlan::output_rows,
        DataFrameProjectedCrossJoinPlan::output_column_names,
        DataFrameProjectedCrossJoinPlan::reservation_size_bytes,
        DataFrameProjectedCrossJoinPlan::materialize,
    );
}

#[test]
fn slice8_ast_conditional_and_mixed_join_surface_is_publicly_importable() {
    use cudf::{
        DataFrameConditionalFullJoinSizeData, DataFrameMixedJoinSizeData, DataFrameMixedSingleSideJoinSizeData,
        ops::ast::{AstExpression, AstOperator},
    };

    let _: Option<AstExpression> = None;
    let _operator = AstOperator::Less;
    let _ = (AstExpression::cast_to_float64, AstExpression::cast_to_int64, AstExpression::cast_to_uint64);
    let _ = (
        cudf::FrameOps::conditional_inner_join_size,
        cudf::FrameOps::conditional_inner_join_indices_with_output_size,
        cudf::FrameOps::conditional_left_join_size,
        cudf::FrameOps::conditional_left_join_indices_with_output_size,
        cudf::FrameOps::conditional_left_semi_join_size,
        cudf::FrameOps::conditional_left_semi_join_indices_with_output_size,
        cudf::FrameOps::conditional_left_anti_join_size,
        cudf::FrameOps::conditional_left_anti_join_indices_with_output_size,
        cudf::FrameOps::conditional_full_join_size_data,
        cudf::FrameOps::mixed_inner_join_size_data,
        cudf::FrameOps::mixed_left_join_size_data,
        cudf::FrameOps::mixed_full_join_size_data,
        cudf::FrameOps::mixed_left_semi_join_size_data,
        cudf::FrameOps::mixed_left_anti_join_size_data,
    );
    let _: Option<DataFrameConditionalFullJoinSizeData<'_>> = None;
    let _ = (
        DataFrameConditionalFullJoinSizeData::output_size,
        DataFrameConditionalFullJoinSizeData::device_allocated_size_bytes,
        DataFrameConditionalFullJoinSizeData::gather_map_reservation_size_bytes,
        DataFrameConditionalFullJoinSizeData::materialize_indices,
    );
    let _: Option<DataFrameMixedJoinSizeData<'_>> = None;
    let _: Option<DataFrameMixedSingleSideJoinSizeData<'_>> = None;
    let _ = (
        DataFrameMixedJoinSizeData::output_size,
        DataFrameMixedJoinSizeData::device_allocated_size_bytes,
        DataFrameMixedJoinSizeData::gather_map_reservation_size_bytes,
        DataFrameMixedJoinSizeData::materialize_indices,
    );
    let _ = (
        DataFrameMixedSingleSideJoinSizeData::output_size,
        DataFrameMixedSingleSideJoinSizeData::device_allocated_size_bytes,
        DataFrameMixedSingleSideJoinSizeData::gather_map_reservation_size_bytes,
        DataFrameMixedSingleSideJoinSizeData::materialize_indices,
    );
}

#[test]
fn slice9_multi_file_parquet_surface_is_publicly_importable() {
    use std::path::PathBuf;

    use cudf::{DataFrame, Runtime, frame::dataframe::MultiParquetReadOptions};

    fn assert_reader(_: fn(MultiParquetReadOptions, cudf::Exec<'_>) -> cudf::Result<DataFrame>) {}

    let rt = Runtime::builder().build().unwrap();
    let paths = [PathBuf::from("part-0.parquet"), PathBuf::from("part-1.parquet")];
    let _options = MultiParquetReadOptions::builder().filepaths(&paths).columns(&["id"]).with_runtime(&rt).build();
    let no_files: [PathBuf; 0] = [];
    let empty = MultiParquetReadOptions::builder().filepaths(&no_files).with_runtime(&rt).build();
    assert!(DataFrame::read_parquet_many(empty, cudf::Exec::RETAINED).is_err());

    assert_reader(DataFrame::read_parquet_many);
}

#[test]
fn parquet_source_planning_surface_is_publicly_importable_under_io_source() {
    use cudf::{
        DataType, ParquetSourceChunkReadOutput, SourceRead, TypeId,
        io::source::{
            ByteRangeDisposition, LocalParquetFile, LocalParquetSource, LocalParquetSourceKind, ParquetSourceLocation,
            PartitionColumn, PartitionColumnDisposition, PartitionValue, PredicateDisposition, RejectedSourceFeature,
            SchemaMappingDisposition, SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, SourceColumn,
            SourceColumnRef, SourceColumnRole, SourceOutputColumn, SourceReadMode, SourceRequiredColumn, SourceSchema,
            ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition, plan_local_parquet_source_chunks,
            source_row_count_to_i64, source_row_group_index_to_i32,
        },
    };

    fn assert_source(_: &LocalParquetSource) {}
    fn assert_location(_: &ParquetSourceLocation) {}
    fn assert_plan(_: &SourceChunkPlan) {}
    fn assert_chunk(_: &SourceChunk) {}
    fn assert_source_reader(
        _: fn(SourceRead<'_>, &cudf::Runtime, cudf::Exec<'_>) -> cudf::Result<ParquetSourceChunkReadOutput>,
    ) {
    }

    let dtype = DataType::new(TypeId::Int64);
    let schema = SourceSchema::new([
        SourceColumn::new("id", dtype, false).unwrap(),
        SourceColumn::new("region", DataType::new(TypeId::String), false).unwrap(),
        SourceColumn::new("filter_only", DataType::new(TypeId::Int32), true).unwrap(),
    ])
    .unwrap();
    let source = LocalParquetSource::data(
        [LocalParquetFile::new("part-0.parquet", "part-0", [PartitionValue::String("AMER".to_string())]).unwrap()],
        schema,
        ["id", "filter_only"],
        [
            SourceOutputColumn::new("id", SourceColumnRef::ReadColumn(0)).unwrap(),
            SourceOutputColumn::new("region", SourceColumnRef::Partition(0)).unwrap(),
        ],
        [SourceRequiredColumn::new("filter_only", SourceColumnRef::ReadColumn(1), SourceColumnRole::PureFilter)
            .unwrap()],
        [PartitionColumn::new("region", DataType::new(TypeId::String), false).unwrap()],
    )
    .unwrap();
    assert_source(&source);

    let diagnostics = source.diagnostics();
    assert_eq!(source.kind(), LocalParquetSourceKind::Data);
    assert_location(source.files()[0].location());
    assert_eq!(diagnostics.predicate_disposition(), PredicateDisposition::Absent);
    assert_eq!(diagnostics.byte_range_disposition(), ByteRangeDisposition::FullFiles);
    assert_eq!(diagnostics.partition_column_disposition(), PartitionColumnDisposition::Represented);
    assert_eq!(diagnostics.zero_column_count_disposition(), ZeroColumnCountDisposition::NotRequired);
    assert_eq!(diagnostics.schema_mapping_disposition(), SchemaMappingDisposition::DirectByName);
    assert_eq!(RejectedSourceFeature::NonLocalSource.as_str(), "non_local_source");

    let count_source =
        LocalParquetSource::zero_column_count([LocalParquetFile::new("counts.parquet", "counts", []).unwrap()])
            .unwrap();
    assert_eq!(count_source.kind(), LocalParquetSourceKind::MetadataRowCount);
    assert_eq!(count_source.output_columns()[0].name(), ZERO_COLUMN_COUNT_COLUMN);
    assert_eq!(count_source.output_columns()[0].source(), SourceColumnRef::ZeroColumnCount);
    assert_eq!(count_source.diagnostics().zero_column_count_disposition(), ZeroColumnCountDisposition::Represented);

    let policy = SourceChunkPlanningPolicy::default();
    assert_eq!(policy.max_source_chunk_bytes(), None);
    assert!(policy.max_row_groups_per_chunk() > 0);
    let plan = plan_local_parquet_source_chunks(&source, policy).unwrap();
    assert_plan(&plan);
    assert_chunk(&plan.chunks()[0]);
    assert_eq!(source_row_group_index_to_i32(0).unwrap(), 0);
    assert_eq!(source_row_count_to_i64(0).unwrap(), 0);
    assert_source_reader(cudf::DataFrame::read_parquet_source);
    assert_eq!(SourceReadMode::FinalOutput, SourceReadMode::FinalOutput);
}

#[test]
fn fast_hash_surface_is_publicly_importable() {
    use cudf::hash::{
        FastHashMap, FastHashSet, FastIndexMap, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
        fast_index_map, fast_index_map_with_capacity,
    };

    let mut map: FastHashMap<String, i32> = fast_hash_map_with_capacity(4);
    map.insert("a".to_string(), 1);
    assert_eq!(map.get("a"), Some(&1));

    let mut set: FastHashSet<&str> = fast_hash_set_with_capacity(4);
    set.insert("x");
    assert!(set.contains("x"));

    let mut index_map: FastIndexMap<String, i32> = fast_index_map_with_capacity(2);
    index_map.insert("first".to_string(), 0);
    index_map.insert("second".to_string(), 1);
    let keys: Vec<&str> = index_map.keys().map(String::as_str).collect();
    assert_eq!(keys, ["first", "second"]);

    let empty: FastIndexMap<String, i32> = fast_index_map();
    assert!(empty.is_empty());
}
