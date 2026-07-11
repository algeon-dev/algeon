//! CPU/GPU fallback policy and stable fallback-reason codes.
//!
//! `FallbackReason::code()` produces the snake-case strings that planning,
//! execution, and fallback-pressure reports emit verbatim. Changing a code is
//! a TSV schema break and requires bumping the consuming report's schema
//! version.

use nexus_query_engine::TriageCode;

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
    /// Native GPU execution is supported, but conservative plan-preference
    /// facts predict CPU will be better for a small selective top-k shape.
    NativeSmallSelectiveTopkCpuPreferred,
    /// Native GPU execution is supported, but conservative plan-preference
    /// facts predict CPU will be better for a small selective residual join.
    NativeSmallSelectiveResidualJoinCpuPreferred,
    /// Native GPU execution is supported, but conservative plan-preference
    /// facts predict CPU will be better for a small selective full-sort shape.
    NativeSmallSelectiveFullSortCpuPreferred,
    /// The replaced subtree declares an output ordering the lowered native plan
    /// cannot reproduce.
    NativeOrderedOutputUnsupported,
    /// Source uses a non-local object store unsupported by native local reads.
    NativeSourceNonLocalObjectStore,
    /// Source partition columns cannot be mapped to native source columns.
    ///
    /// Reserved: no current raising site (partition failures raise
    /// [`Self::NativeSourcePartitionValues`] or [`Self::UnsupportedType`]).
    /// Kept because reason codes are a stable report contract.
    NativeSourcePartitionColumns,
    /// Source expression adapter cannot be represented in native source IR.
    NativeSourceExprAdapter,
    /// The DataFusion source uses a Parquet file reader factory that the native
    /// local-Parquet adapter cannot reproduce.
    NativeSourceParquetFileReaderFactory,
    /// Source scan limit is not supported for this native source shape.
    ///
    /// Reserved: no current raising site (native scan limits are supported).
    /// Kept because reason codes are a stable report contract.
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
    ///
    /// Reserved: no current raising site (per-file ordering is carried since
    /// declared-order native scan support). Kept because reason codes are a
    /// stable report contract.
    NativeSourceFileOrdering,
    /// Source row-selection metadata cannot be represented natively.
    NativeSourceRowSelection,
    /// Source schema mapping cannot be represented natively.
    NativeSourceSchemaMapping,
    /// Table-format scan requested an unsupported feature.
    TableFormatUnsupported,
    /// Table-format planning failed before a native scan could be built.
    ///
    /// Reserved: no current raising site (real planning failures surface as
    /// hard errors). Kept because reason codes are a stable report contract.
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
    ///
    /// Reserved: no current raising site (time travel by snapshot id works;
    /// branch/tag refs are not modeled in scan requests). Kept because reason
    /// codes are a stable report contract.
    IcebergSnapshotUnsupported,
    /// A GPU-only table-format scan leaf survived native optimization, so the
    /// remaining plan cannot fall back to DataFusion CPU execution.
    TableFormatGpuOnlyLeafNotRewritten,
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
            Self::NativeSmallSelectiveTopkCpuPreferred => "native_small_selective_topk_cpu_preferred",
            Self::NativeSmallSelectiveResidualJoinCpuPreferred => "native_small_selective_residual_join_cpu_preferred",
            Self::NativeSmallSelectiveFullSortCpuPreferred => "native_small_selective_full_sort_cpu_preferred",
            Self::NativeOrderedOutputUnsupported => "native_ordered_output_unsupported",
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
            Self::TableFormatGpuOnlyLeafNotRewritten => "table_format_gpu_only_leaf_not_rewritten",
        }
    }

    /// Shared triage code for this planning fallback reason.
    pub const fn triage_code(self) -> TriageCode {
        match self {
            Self::UnsupportedOperator => TriageCode::UnsupportedOperator,
            Self::UnsupportedExpression => TriageCode::UnsupportedExpression,
            Self::UnsupportedType => TriageCode::UnsupportedType,
            Self::ExplicitCpuPolicy => TriageCode::ExplicitCpuPolicy,
            Self::NativeSmallSelectiveTopkCpuPreferred
            | Self::NativeSmallSelectiveResidualJoinCpuPreferred
            | Self::NativeSmallSelectiveFullSortCpuPreferred => TriageCode::CpuPreferred,
            Self::NativeBoundedJoinUnsupported => TriageCode::StrictBoundedRejected,
            Self::NativeAdaptivePipelineUnsupported => TriageCode::AdaptivePipelineRejected,
            Self::NativeSourceNonLocalObjectStore
            | Self::NativeSourcePartitionColumns
            | Self::NativeSourceExprAdapter
            | Self::NativeSourceParquetFileReaderFactory
            | Self::NativeSourceScanLimit
            | Self::NativeSourcePreserveOrder
            | Self::NativeSourceOutputOrdering
            | Self::NativeSourcePushedDownParquetPredicate
            | Self::NativeSourceEmpty
            | Self::NativeSourcePartitionValues
            | Self::NativeSourceByteRange
            | Self::NativeSourceFileOrdering
            | Self::NativeSourceRowSelection
            | Self::NativeSourceSchemaMapping
            | Self::TableFormatUnsupported
            | Self::IcebergNonParquetDataFile
            | Self::IcebergDeleteFilesUnsupported
            | Self::IcebergFieldIdMappingUnsupported
            | Self::IcebergNameMappingUnsupported
            | Self::IcebergNestedProjectionUnsupported
            | Self::IcebergPartitionTransformUnsupported
            | Self::IcebergResidualPredicateUnsupported
            | Self::IcebergRowSelectionUnsupported
            | Self::IcebergObjectStoreReadUnsupported
            | Self::IcebergSnapshotUnsupported => TriageCode::UnsupportedSource,
            Self::TableFormatPlanningFailed => TriageCode::InvalidPlan,
            Self::NativeSortPreservingMergeUnsupported
            | Self::NativeDistributedAggregateUnsupported
            | Self::NativePartitionPreservingSortUnsupported
            | Self::NativeOrderedOutputUnsupported
            | Self::TableFormatGpuOnlyLeafNotRewritten => TriageCode::UnsupportedPlanShape,
        }
    }

    /// Returns the data-layout property an operator can change to avoid this
    /// native-planning rejection, when the rejection is layout-fixable.
    pub const fn data_layout_fixable_property(self) -> Option<&'static str> {
        match self {
            Self::NativeOrderedOutputUnsupported
            | Self::NativeSourceOutputOrdering
            | Self::NativeSourcePreserveOrder
            | Self::NativeSourceFileOrdering => Some("declared_output_ordering"),
            _ => None,
        }
    }

    /// Whether this rejection is caused by table layout metadata rather than
    /// unsupported query semantics.
    pub const fn is_data_layout_fixable(self) -> bool {
        self.data_layout_fixable_property().is_some()
    }
}
