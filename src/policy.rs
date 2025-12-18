#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GpuFallbackPolicy {
    GpuOnly,
    #[default]
    GpuPreferred,
    CpuOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    InsufficientIslandSize,
    PlanPropertyMismatch,
    ExplicitCpuPolicy,
    NativeSortPreservingMergeUnsupported,
    NativePartitionPreservingSortUnsupported,
    NativeSourceNonLocalObjectStore,
    NativeSourcePartitionColumns,
    NativeSourceExprAdapter,
    NativeSourceScanLimit,
    NativeSourcePreserveOrder,
    NativeSourceOutputOrdering,
    NativeSourcePushedDownParquetPredicate,
    NativeSourceEmpty,
    NativeSourcePartitionValues,
    NativeSourceByteRange,
    NativeSourceFileOrdering,
    NativeSourceRowSelection,
}

impl FallbackReason {
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InsufficientIslandSize => "insufficient_island_size",
            Self::PlanPropertyMismatch => "plan_property_mismatch",
            Self::ExplicitCpuPolicy => "explicit_cpu_policy",
            Self::NativeSortPreservingMergeUnsupported => "native_sort_preserving_merge_unsupported",
            Self::NativePartitionPreservingSortUnsupported => "native_partition_preserving_sort_unsupported",
            Self::NativeSourceNonLocalObjectStore => "native_source_non_local_object_store",
            Self::NativeSourcePartitionColumns => "native_source_partition_columns",
            Self::NativeSourceExprAdapter => "native_source_expr_adapter",
            Self::NativeSourceScanLimit => "native_source_scan_limit",
            Self::NativeSourcePreserveOrder => "native_source_preserve_order",
            Self::NativeSourceOutputOrdering => "native_source_output_ordering",
            Self::NativeSourcePushedDownParquetPredicate => "native_source_pushed_down_parquet_predicate",
            Self::NativeSourceEmpty => "native_source_empty",
            Self::NativeSourcePartitionValues => "native_source_partition_values",
            Self::NativeSourceByteRange => "native_source_byte_range",
            Self::NativeSourceFileOrdering => "native_source_file_ordering",
            Self::NativeSourceRowSelection => "native_source_row_selection",
        }
    }
}
