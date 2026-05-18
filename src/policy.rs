#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GpuFallbackPolicy {
    GpuOnly,
    #[default]
    GpuPreferred,
    CpuOnly,
}

/// Stable reason codes explaining why a candidate physical plan stayed on CPU
/// or was rejected by the native GPU path.
///
/// Operator/expression/type reasons describe unsupported DataFusion plan
/// semantics, while `NativeSource*` reasons describe source-adapter
/// requirements that could not be mapped to a native local-Parquet source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    ExplicitCpuPolicy,
    /// DataFusion produced a sort-preserving merge shape that the native
    /// planner does not lower as an execution boundary yet.
    NativeSortPreservingMergeUnsupported,
    /// DataFusion produced distributed partial/final aggregate stages whose
    /// native aggregate state contract is not modeled for this shape.
    NativeDistributedAggregateUnsupported,
    /// DataFusion produced a partition-preserving sort shape that the native
    /// planner does not currently lower.
    NativePartitionPreservingSortUnsupported,
    NativeBoundedJoinUnsupported,
    NativeAdaptivePipelineUnsupported,
    NativeSourceNonLocalObjectStore,
    NativeSourcePartitionColumns,
    NativeSourceExprAdapter,
    /// The DataFusion source uses a Parquet file reader factory that the native
    /// local-Parquet adapter cannot reproduce.
    NativeSourceParquetFileReaderFactory,
    NativeSourceScanLimit,
    NativeSourcePreserveOrder,
    NativeSourceOutputOrdering,
    /// The source has pushed-down Parquet predicates that native row-group
    /// pruning cannot translate without changing scan semantics.
    NativeSourcePushedDownParquetPredicate,
    NativeSourceEmpty,
    NativeSourcePartitionValues,
    NativeSourceByteRange,
    NativeSourceFileOrdering,
    NativeSourceRowSelection,
    NativeSourceSchemaMapping,
    TableFormatUnsupported,
    TableFormatPlanningFailed,
    IcebergNonParquetDataFile,
    IcebergDeleteFilesUnsupported,
    IcebergFieldIdMappingUnsupported,
    IcebergNameMappingUnsupported,
    IcebergNestedProjectionUnsupported,
    IcebergPartitionTransformUnsupported,
    IcebergResidualPredicateUnsupported,
    IcebergRowSelectionUnsupported,
    IcebergObjectStoreReadUnsupported,
    IcebergSnapshotUnsupported,
}

impl FallbackReason {
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::ExplicitCpuPolicy => "explicit_cpu_policy",
            Self::NativeSortPreservingMergeUnsupported => "native_sort_preserving_merge_unsupported",
            Self::NativeDistributedAggregateUnsupported => "native_distributed_aggregate_unsupported",
            Self::NativePartitionPreservingSortUnsupported => "native_partition_preserving_sort_unsupported",
            Self::NativeBoundedJoinUnsupported => "native_bounded_join_unsupported",
            Self::NativeAdaptivePipelineUnsupported => "native_adaptive_pipeline_unsupported",
            Self::NativeSourceNonLocalObjectStore => "native_source_non_local_object_store",
            Self::NativeSourcePartitionColumns => "native_source_partition_columns",
            Self::NativeSourceExprAdapter => "native_source_expr_adapter",
            Self::NativeSourceParquetFileReaderFactory => "native_source_parquet_file_reader_factory",
            Self::NativeSourceScanLimit => "native_source_scan_limit",
            Self::NativeSourcePreserveOrder => "native_source_preserve_order",
            Self::NativeSourceOutputOrdering => "native_source_output_ordering",
            Self::NativeSourcePushedDownParquetPredicate => "native_source_pushed_down_parquet_predicate",
            Self::NativeSourceEmpty => "native_source_empty",
            Self::NativeSourcePartitionValues => "native_source_partition_values",
            Self::NativeSourceByteRange => "native_source_byte_range",
            Self::NativeSourceFileOrdering => "native_source_file_ordering",
            Self::NativeSourceRowSelection => "native_source_row_selection",
            Self::NativeSourceSchemaMapping => "native_source_schema_mapping",
            Self::TableFormatUnsupported => "table_format_unsupported",
            Self::TableFormatPlanningFailed => "table_format_planning_failed",
            Self::IcebergNonParquetDataFile => "iceberg_non_parquet_data_file",
            Self::IcebergDeleteFilesUnsupported => "iceberg_delete_files_unsupported",
            Self::IcebergFieldIdMappingUnsupported => "iceberg_field_id_mapping_unsupported",
            Self::IcebergNameMappingUnsupported => "iceberg_name_mapping_unsupported",
            Self::IcebergNestedProjectionUnsupported => "iceberg_nested_projection_unsupported",
            Self::IcebergPartitionTransformUnsupported => "iceberg_partition_transform_unsupported",
            Self::IcebergResidualPredicateUnsupported => "iceberg_residual_predicate_unsupported",
            Self::IcebergRowSelectionUnsupported => "iceberg_row_selection_unsupported",
            Self::IcebergObjectStoreReadUnsupported => "iceberg_object_store_read_unsupported",
            Self::IcebergSnapshotUnsupported => "iceberg_snapshot_unsupported",
        }
    }
}
