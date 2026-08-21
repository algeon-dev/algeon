//! CPU/GPU candidate-outcome capability reason codes.
//!
//! `CapabilityReason::code()` produces the snake-case strings that planning
//! and report projections emit verbatim. Changing a code is
//! a TSV schema break and requires bumping the consuming report's schema
//! version.

use nexus_query_engine::triage::TriageCode;

/// Stable reason codes explaining why a candidate cannot execute on the native
/// GPU path.
///
/// Operator/expression/type reasons describe unsupported DataFusion plan
/// semantics, while `NativeSource*` reasons describe source-adapter
/// requirements that could not be mapped to a native local-Parquet source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityReason {
    /// Physical operator is not supported by native lowering.
    UnsupportedOperator,
    /// Physical expression is not supported by native lowering.
    UnsupportedExpression,
    /// Data type or physical representation is not supported.
    UnsupportedType,
    /// DataFusion produced a sort-preserving merge shape that the native
    /// planner does not lower as an execution boundary yet.
    NativeSortPreservingMergeUnsupported,
    /// DataFusion produced distributed partial/final aggregate stages whose
    /// native aggregate state contract is not modeled for this shape.
    NativeDistributedAggregateUnsupported,
    /// DataFusion produced a partition-preserving sort shape that the native
    /// planner does not currently lower.
    NativePartitionPreservingSortUnsupported,
    /// Bounded native join capability admission rejected the shape.
    NativeBoundedJoinUnsupported,
    /// The replaced subtree declares an output ordering the lowered native plan
    /// cannot reproduce.
    NativeOrderedOutputUnsupported,
    /// Source uses a non-local object store unsupported by native local reads.
    NativeSourceNonLocalObjectStore,
    /// Source expression adapter cannot be represented in native source IR.
    NativeSourceExprAdapter,
    /// The DataFusion source uses a Parquet file reader factory that the native
    /// local-Parquet adapter cannot reproduce.
    NativeSourceParquetFileReaderFactory,
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
    /// Source row-selection metadata cannot be represented natively.
    NativeSourceRowSelection,
    /// Source schema mapping cannot be represented natively.
    NativeSourceSchemaMapping,
    /// Table-format scan requested an unsupported feature.
    TableFormatUnsupported,
    /// Table-format scan inspection failed or no scan plan could be built
    /// before a native scan could be lowered.
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
    /// Iceberg remote S3 FileIO cannot be proven to use the frozen
    /// CPU/native source-access policy.
    IcebergFileIoSourceAccessUnverified,
    /// Iceberg metadata contains an encrypted data or delete file.
    IcebergEncryptedFileUnsupported,
    /// An Iceberg metadata scalar cannot be materialized by the native source.
    IcebergScalarValueUnsupported,
}

impl CapabilityReason {
    /// Every stable capability reason emitted by native planning.
    ///
    /// Internal diagnostics projections use this list to assert that a new
    /// reason cannot accidentally ship without actionable guidance.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 33] = [
        Self::UnsupportedOperator,
        Self::UnsupportedExpression,
        Self::UnsupportedType,
        Self::NativeSortPreservingMergeUnsupported,
        Self::NativeDistributedAggregateUnsupported,
        Self::NativePartitionPreservingSortUnsupported,
        Self::NativeBoundedJoinUnsupported,
        Self::NativeOrderedOutputUnsupported,
        Self::NativeSourceNonLocalObjectStore,
        Self::NativeSourceExprAdapter,
        Self::NativeSourceParquetFileReaderFactory,
        Self::NativeSourcePreserveOrder,
        Self::NativeSourceOutputOrdering,
        Self::NativeSourcePushedDownParquetPredicate,
        Self::NativeSourceEmpty,
        Self::NativeSourcePartitionValues,
        Self::NativeSourceByteRange,
        Self::NativeSourceRowSelection,
        Self::NativeSourceSchemaMapping,
        Self::TableFormatUnsupported,
        Self::TableFormatPlanningFailed,
        Self::IcebergNonParquetDataFile,
        Self::IcebergDeleteFilesUnsupported,
        Self::IcebergFieldIdMappingUnsupported,
        Self::IcebergNameMappingUnsupported,
        Self::IcebergNestedProjectionUnsupported,
        Self::IcebergPartitionTransformUnsupported,
        Self::IcebergResidualPredicateUnsupported,
        Self::IcebergRowSelectionUnsupported,
        Self::IcebergObjectStoreReadUnsupported,
        Self::IcebergFileIoSourceAccessUnverified,
        Self::IcebergEncryptedFileUnsupported,
        Self::IcebergScalarValueUnsupported,
    ];

    /// Stable snake-case reason code used by reports and tests.
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::NativeSortPreservingMergeUnsupported => "native_sort_preserving_merge_unsupported",
            Self::NativeDistributedAggregateUnsupported => "native_distributed_aggregate_unsupported",
            Self::NativePartitionPreservingSortUnsupported => "native_partition_preserving_sort_unsupported",
            Self::NativeBoundedJoinUnsupported => "native_bounded_join_unsupported",
            Self::NativeOrderedOutputUnsupported => "native_ordered_output_unsupported",
            Self::NativeSourceNonLocalObjectStore => "native_source_non_local_object_store",
            Self::NativeSourceExprAdapter => "native_source_expr_adapter",
            Self::NativeSourceParquetFileReaderFactory => "native_source_parquet_file_reader_factory",
            Self::NativeSourcePreserveOrder => "native_source_preserve_order",
            Self::NativeSourceOutputOrdering => "native_source_output_ordering",
            Self::NativeSourcePushedDownParquetPredicate => "native_source_pushed_down_parquet_predicate",
            Self::NativeSourceEmpty => "native_source_empty",
            Self::NativeSourcePartitionValues => "native_source_partition_values",
            Self::NativeSourceByteRange => "native_source_byte_range",
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
            Self::IcebergFileIoSourceAccessUnverified => "iceberg_file_io_source_access_unverified",
            Self::IcebergEncryptedFileUnsupported => "iceberg_encrypted_file_unsupported",
            Self::IcebergScalarValueUnsupported => "iceberg_scalar_value_unsupported",
        }
    }

    /// Shared triage code for this capability miss.
    pub const fn triage_code(self) -> TriageCode {
        match self {
            Self::UnsupportedOperator => TriageCode::UnsupportedOperator,
            Self::UnsupportedExpression => TriageCode::UnsupportedExpression,
            Self::UnsupportedType => TriageCode::UnsupportedType,
            Self::NativeBoundedJoinUnsupported => TriageCode::CapabilityEnvelopeRejected,
            Self::NativeSourceNonLocalObjectStore
            | Self::NativeSourceExprAdapter
            | Self::NativeSourceParquetFileReaderFactory
            | Self::NativeSourcePreserveOrder
            | Self::NativeSourceOutputOrdering
            | Self::NativeSourcePushedDownParquetPredicate
            | Self::NativeSourceEmpty
            | Self::NativeSourcePartitionValues
            | Self::NativeSourceByteRange
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
            | Self::IcebergFileIoSourceAccessUnverified
            | Self::IcebergEncryptedFileUnsupported
            | Self::IcebergScalarValueUnsupported => TriageCode::UnsupportedSource,
            Self::TableFormatPlanningFailed => TriageCode::InvalidPlan,
            Self::NativeSortPreservingMergeUnsupported
            | Self::NativeDistributedAggregateUnsupported
            | Self::NativePartitionPreservingSortUnsupported
            | Self::NativeOrderedOutputUnsupported => TriageCode::UnsupportedPlanShape,
        }
    }

    /// Returns the data-layout property an operator can change to avoid this
    /// native-planning rejection, when the rejection is layout-fixable.
    pub const fn data_layout_fixable_property(self) -> Option<&'static str> {
        match self {
            Self::NativeOrderedOutputUnsupported
            | Self::NativeSourceOutputOrdering
            | Self::NativeSourcePreserveOrder => Some("declared_output_ordering"),
            _ => None,
        }
    }

    /// Whether this rejection is caused by table layout metadata rather than
    /// unsupported query semantics.
    pub const fn is_data_layout_fixable(self) -> bool {
        self.data_layout_fixable_property().is_some()
    }
}
