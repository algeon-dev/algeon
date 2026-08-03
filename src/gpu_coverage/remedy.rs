//! Stable actionable guidance for native candidate-selection outcomes.

use nexus_query_engine::plan::NativePlanCostReason;

use super::model::{QueryGpuRemedy, QueryGpuRemedyKind};
use crate::capability::CapabilityReason;

pub(super) const fn capability_remedy(reason: CapabilityReason) -> QueryGpuRemedy {
    match reason {
        CapabilityReason::UnsupportedOperator => QueryGpuRemedy::new(
            "unsupported_operator_gap",
            QueryGpuRemedyKind::EngineGap,
            "The physical operator is not yet supported by native lowering; rewrite the query or accept the DataFusion path.",
        ),
        CapabilityReason::UnsupportedExpression => QueryGpuRemedy::new(
            "rewrite_expression",
            QueryGpuRemedyKind::UserAction,
            "Rewrite the expression into a supported form; use the unsupported category and candidate shape to isolate the expression family.",
        ),
        CapabilityReason::UnsupportedType => QueryGpuRemedy::new(
            "cast_or_exclude_type",
            QueryGpuRemedyKind::UserAction,
            "Cast, exclude, or materialize unsupported column types before running the GPU query.",
        ),
        CapabilityReason::NativeSortPreservingMergeUnsupported
        | CapabilityReason::NativeDistributedAggregateUnsupported
        | CapabilityReason::NativePartitionPreservingSortUnsupported
        | CapabilityReason::NativeOrderedOutputUnsupported => QueryGpuRemedy::new(
            "use_native_friendly_session_profile",
            QueryGpuRemedyKind::ConfigAction,
            "Use a native-friendly session profile or rewrite distributed ordering and aggregate stages into a shape native lowering supports.",
        ),
        CapabilityReason::NativeBoundedJoinUnsupported => QueryGpuRemedy::new(
            "capability_contract_gap",
            QueryGpuRemedyKind::CudfRsGap,
            "Bounded capability admission lacks a required lower-level reservation contract for this join shape.",
        ),
        CapabilityReason::NativeSourceNonLocalObjectStore | CapabilityReason::IcebergObjectStoreReadUnsupported => {
            QueryGpuRemedy::new(
                "stage_local_parquet",
                QueryGpuRemedyKind::UserAction,
                "Stage or cache the data as local Parquet until native object-store reads are supported.",
            )
        },
        CapabilityReason::NativeSourcePushedDownParquetPredicate => QueryGpuRemedy::new(
            "disable_parquet_pushdown",
            QueryGpuRemedyKind::ConfigAction,
            "Disable the unsupported Parquet predicate pushdown for diagnosis or rewrite the query.",
        ),
        CapabilityReason::NativeSourceScanLimit => QueryGpuRemedy::new(
            "avoid_scan_limit_pushdown",
            QueryGpuRemedyKind::UserAction,
            "Avoid query shapes that push LIMIT into the scan.",
        ),
        CapabilityReason::IcebergDeleteFilesUnsupported => QueryGpuRemedy::new(
            "rewrite_table_without_delete_files",
            QueryGpuRemedyKind::UserAction,
            "Compact or rewrite the Iceberg table so the scanned snapshot has no delete files.",
        ),
        CapabilityReason::NativeSourcePartitionColumns
        | CapabilityReason::NativeSourceExprAdapter
        | CapabilityReason::NativeSourceParquetFileReaderFactory
        | CapabilityReason::NativeSourcePreserveOrder
        | CapabilityReason::NativeSourceOutputOrdering
        | CapabilityReason::NativeSourceEmpty
        | CapabilityReason::NativeSourcePartitionValues
        | CapabilityReason::NativeSourceByteRange
        | CapabilityReason::NativeSourceFileOrdering
        | CapabilityReason::NativeSourceRowSelection
        | CapabilityReason::NativeSourceSchemaMapping
        | CapabilityReason::TableFormatUnsupported
        | CapabilityReason::TableFormatPlanningFailed
        | CapabilityReason::IcebergNonParquetDataFile
        | CapabilityReason::IcebergFieldIdMappingUnsupported
        | CapabilityReason::IcebergNameMappingUnsupported
        | CapabilityReason::IcebergNestedProjectionUnsupported
        | CapabilityReason::IcebergPartitionTransformUnsupported
        | CapabilityReason::IcebergResidualPredicateUnsupported
        | CapabilityReason::IcebergRowSelectionUnsupported
        | CapabilityReason::IcebergFileIoSourceAccessUnverified
        | CapabilityReason::IcebergSnapshotUnsupported
        | CapabilityReason::IcebergEncryptedFileUnsupported
        | CapabilityReason::IcebergScalarValueUnsupported => QueryGpuRemedy::new(
            "engine_gap",
            QueryGpuRemedyKind::EngineGap,
            "This source or table-format feature is not yet representable by native lowering; rewrite or materialize a supported local Parquet snapshot, or accept the DataFusion path.",
        ),
    }
}

pub(super) const fn cost_remedy(reason: NativePlanCostReason) -> QueryGpuRemedy {
    match reason {
        NativePlanCostReason::SmallSelectiveTopK
        | NativePlanCostReason::SmallSelectiveResidualJoin
        | NativePlanCostReason::SmallSelectiveFullSort => QueryGpuRemedy::new(
            "keep_cpu_for_small_selective_work",
            QueryGpuRemedyKind::CpuPreferred,
            "The cost model expects CPU execution to be cheaper for this small selective shape; increase useful GPU work or keep the DataFusion path.",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_capability_reason_has_a_stable_nonempty_remedy() {
        for reason in CapabilityReason::ALL {
            let remedy = capability_remedy(reason);
            assert!(!remedy.code.is_empty(), "{reason:?}");
            assert!(!remedy.guidance.is_empty(), "{reason:?}");
        }
    }

    #[test]
    fn every_capability_reason_preserves_its_remedy_code_and_kind() {
        for reason in CapabilityReason::ALL {
            let (expected_code, expected_kind) = match reason {
                CapabilityReason::UnsupportedOperator => ("unsupported_operator_gap", QueryGpuRemedyKind::EngineGap),
                CapabilityReason::UnsupportedExpression => ("rewrite_expression", QueryGpuRemedyKind::UserAction),
                CapabilityReason::UnsupportedType => ("cast_or_exclude_type", QueryGpuRemedyKind::UserAction),
                CapabilityReason::NativeSortPreservingMergeUnsupported
                | CapabilityReason::NativeDistributedAggregateUnsupported
                | CapabilityReason::NativePartitionPreservingSortUnsupported
                | CapabilityReason::NativeOrderedOutputUnsupported => {
                    ("use_native_friendly_session_profile", QueryGpuRemedyKind::ConfigAction)
                },
                CapabilityReason::NativeBoundedJoinUnsupported => {
                    ("capability_contract_gap", QueryGpuRemedyKind::CudfRsGap)
                },
                CapabilityReason::NativeSourceNonLocalObjectStore
                | CapabilityReason::IcebergObjectStoreReadUnsupported => {
                    ("stage_local_parquet", QueryGpuRemedyKind::UserAction)
                },
                CapabilityReason::NativeSourcePushedDownParquetPredicate => {
                    ("disable_parquet_pushdown", QueryGpuRemedyKind::ConfigAction)
                },
                CapabilityReason::NativeSourceScanLimit => {
                    ("avoid_scan_limit_pushdown", QueryGpuRemedyKind::UserAction)
                },
                CapabilityReason::IcebergDeleteFilesUnsupported => {
                    ("rewrite_table_without_delete_files", QueryGpuRemedyKind::UserAction)
                },
                CapabilityReason::NativeSourcePartitionColumns
                | CapabilityReason::NativeSourceExprAdapter
                | CapabilityReason::NativeSourceParquetFileReaderFactory
                | CapabilityReason::NativeSourcePreserveOrder
                | CapabilityReason::NativeSourceOutputOrdering
                | CapabilityReason::NativeSourceEmpty
                | CapabilityReason::NativeSourcePartitionValues
                | CapabilityReason::NativeSourceByteRange
                | CapabilityReason::NativeSourceFileOrdering
                | CapabilityReason::NativeSourceRowSelection
                | CapabilityReason::NativeSourceSchemaMapping
                | CapabilityReason::TableFormatUnsupported
                | CapabilityReason::TableFormatPlanningFailed
                | CapabilityReason::IcebergNonParquetDataFile
                | CapabilityReason::IcebergFieldIdMappingUnsupported
                | CapabilityReason::IcebergNameMappingUnsupported
                | CapabilityReason::IcebergNestedProjectionUnsupported
                | CapabilityReason::IcebergPartitionTransformUnsupported
                | CapabilityReason::IcebergResidualPredicateUnsupported
                | CapabilityReason::IcebergRowSelectionUnsupported
                | CapabilityReason::IcebergFileIoSourceAccessUnverified
                | CapabilityReason::IcebergSnapshotUnsupported
                | CapabilityReason::IcebergEncryptedFileUnsupported
                | CapabilityReason::IcebergScalarValueUnsupported => ("engine_gap", QueryGpuRemedyKind::EngineGap),
            };
            let remedy = capability_remedy(reason);
            assert_eq!(remedy.code, expected_code, "{reason:?}");
            assert_eq!(remedy.kind, expected_kind, "{reason:?}");
        }
    }

    #[test]
    fn every_cost_reason_preserves_its_remedy_code_and_kind() {
        for reason in [
            NativePlanCostReason::SmallSelectiveTopK,
            NativePlanCostReason::SmallSelectiveResidualJoin,
            NativePlanCostReason::SmallSelectiveFullSort,
        ] {
            let remedy = cost_remedy(reason);
            assert_eq!(remedy.code, "keep_cpu_for_small_selective_work", "{reason:?}");
            assert_eq!(remedy.kind, QueryGpuRemedyKind::CpuPreferred, "{reason:?}");
        }
    }

    #[test]
    fn prescribed_remedies_remain_stable() {
        let local = capability_remedy(CapabilityReason::NativeSourceNonLocalObjectStore);
        assert_eq!(local.code, "stage_local_parquet");
        assert_eq!(local.kind, QueryGpuRemedyKind::UserAction);

        let unsupported = capability_remedy(CapabilityReason::UnsupportedExpression);
        assert_eq!(unsupported.code, "rewrite_expression");
        assert_eq!(unsupported.kind, QueryGpuRemedyKind::UserAction);

        let cost = cost_remedy(NativePlanCostReason::SmallSelectiveTopK);
        assert_eq!(cost.code, "keep_cpu_for_small_selective_work");
        assert_eq!(cost.kind, QueryGpuRemedyKind::CpuPreferred);
    }
}
