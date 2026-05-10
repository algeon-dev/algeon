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
    ExplicitCpuPolicy,
    NativeSortPreservingMergeUnsupported,
    NativeDistributedAggregateUnsupported,
    NativePartitionPreservingSortUnsupported,
    NativeBoundedJoinUnsupported,
    NativeSourceNonLocalObjectStore,
    NativeSourcePartitionColumns,
    NativeSourceExprAdapter,
    NativeSourceParquetFileReaderFactory,
    NativeSourceScanLimit,
    NativeSourcePreserveOrder,
    NativeSourceOutputOrdering,
    NativeSourcePushedDownParquetPredicate,
    NativeSourceEmpty,
    NativeSourcePartitionValues,
    NativeSourceByteRange,
    NativeSourceFileOrdering,
    NativeSourceRowSelection,
    NativeSourceSchemaMapping,
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
        }
    }
}
