/// Policy controlling whether DataFusion may fall back to CPU execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GpuFallbackPolicy {
    /// Require native GPU execution; rejected candidates become errors.
    GpuOnly,
    /// Prefer native GPU execution but allow CPU DataFusion fallback.
    #[default]
    GpuPreferred,
    /// Skip native GPU execution and use CPU DataFusion.
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
    /// Physical operator is not supported by native lowering.
    UnsupportedOperator,
    /// Physical expression is not supported by native lowering.
    UnsupportedExpression,
    /// Data type or physical representation is not supported.
    UnsupportedType,
    /// Caller selected an explicit CPU-only policy.
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
    /// Strict bounded native join admission rejected the shape.
    NativeBoundedJoinUnsupported,
    /// Adaptive native pipeline admission rejected the shape.
    NativeAdaptivePipelineUnsupported,
    /// Source uses a non-local object store unsupported by native local reads.
    NativeSourceNonLocalObjectStore,
    /// Source partition columns cannot be mapped to native source columns.
    NativeSourcePartitionColumns,
    /// Source expression adapter cannot be represented in native source IR.
    NativeSourceExprAdapter,
    /// The DataFusion source uses a Parquet file reader factory that the native
    /// local-Parquet adapter cannot reproduce.
    NativeSourceParquetFileReaderFactory,
    /// Source scan limit is not supported for this native source shape.
    NativeSourceScanLimit,
    /// Source order-preservation requirements cannot be maintained natively.
    NativeSourcePreserveOrder,
    /// Source output ordering cannot be represented in native source IR.
    NativeSourceOutputOrdering,
    /// The source has pushed-down Parquet predicates that native row-group
    /// pruning cannot translate without changing scan semantics.
    NativeSourcePushedDownParquetPredicate,
    /// Source contains no files or row groups to read.
    NativeSourceEmpty,
    /// Source partition values cannot be represented by native source IR.
    NativeSourcePartitionValues,
    /// Source byte ranges are unsupported by native local reads.
    NativeSourceByteRange,
    /// Source file ordering requirements cannot be maintained natively.
    NativeSourceFileOrdering,
    /// Source row-selection metadata cannot be represented natively.
    NativeSourceRowSelection,
    /// Source schema mapping cannot be represented natively.
    NativeSourceSchemaMapping,
    /// Table-format scan requested an unsupported feature.
    TableFormatUnsupported,
    /// Table-format planning failed before a native scan could be built.
    TableFormatPlanningFailed,
    /// Iceberg data file is not Parquet.
    IcebergNonParquetDataFile,
    /// Iceberg delete files are present and unsupported.
    IcebergDeleteFilesUnsupported,
    /// Iceberg field-id mapping is required and unsupported.
    IcebergFieldIdMappingUnsupported,
    /// Iceberg name mapping is required and unsupported.
    IcebergNameMappingUnsupported,
    /// Iceberg projection includes nested fields unsupported by native scans.
    IcebergNestedProjectionUnsupported,
    /// Iceberg partition transform cannot be represented natively.
    IcebergPartitionTransformUnsupported,
    /// Iceberg residual predicate cannot be represented natively.
    IcebergResidualPredicateUnsupported,
    /// Iceberg row selection cannot be represented natively.
    IcebergRowSelectionUnsupported,
    /// Iceberg object-store read cannot be lowered to native local reads.
    IcebergObjectStoreReadUnsupported,
    /// Iceberg snapshot selection or loading is unsupported.
    IcebergSnapshotUnsupported,
}

impl FallbackReason {
    /// Stable snake-case reason code used by reports and tests.
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
