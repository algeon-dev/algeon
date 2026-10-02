//! CPU/GPU candidate-outcome capability reason codes.
//!
//! [`CapabilityReason::facts`] is the one table every projection of a reason
//! reads: the report code, triage code, coverage reason kind and remedy, the
//! default TSV category, the data-layout property, and the Flight log class.
//! A changed value changes every report that carries it, in place.

pub use query_engine::triage::TriageCode;

use crate::gpu_coverage::{QueryGpuCandidateReasonKind, QueryGpuRemedy, QueryGpuRemedyKind};

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
    /// DataFusion produced a local limit over several partitions; it keeps
    /// rows per partition, which one native output partition cannot express.
    NativeLocalLimitPartitionUnsupported,
    /// The exact statistical aggregate mode needs DataFusion to read the
    /// aggregate input as one partition; an exchange that merges partitions
    /// interleaves rows in arrival order, so DataFusion's floating-point result
    /// varies between runs.
    NativeExactStatisticalInputPartitioned,
    /// Bounded native join capability admission rejected the shape.
    NativeBoundedJoinUnsupported,
    /// No compiled pipeline contract covers a native plan without a join; the
    /// candidate's capability decision row names the missing contract.
    NativeCompiledPipelineUnsupported,
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
}

impl CapabilityReason {
    /// Every stable capability reason emitted by native planning.
    ///
    /// Internal diagnostics projections use this list to assert that a new
    /// reason cannot accidentally ship without actionable guidance.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 22] = [
        Self::UnsupportedOperator,
        Self::UnsupportedExpression,
        Self::UnsupportedType,
        Self::NativeSortPreservingMergeUnsupported,
        Self::NativeDistributedAggregateUnsupported,
        Self::NativePartitionPreservingSortUnsupported,
        Self::NativeLocalLimitPartitionUnsupported,
        Self::NativeExactStatisticalInputPartitioned,
        Self::NativeBoundedJoinUnsupported,
        Self::NativeCompiledPipelineUnsupported,
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
    ];

    /// Stable snake-case reason code used by reports and tests.
    pub const fn code(self) -> &'static str {
        self.facts().code
    }

    /// Every projection of this reason.
    pub const fn facts(self) -> ReasonFacts {
        match self {
            Self::UnsupportedOperator => ReasonFacts {
                code: "unsupported_operator",
                triage: TriageCode::UnsupportedOperator,
                kind: QueryGpuCandidateReasonKind::Operator,
                remedy: QueryGpuRemedy::new(
                    "unsupported_operator_gap",
                    QueryGpuRemedyKind::EngineGap,
                    "The physical operator is not yet supported by native lowering; rewrite the query or accept the DataFusion path.",
                ),
                tsv_category: "operator_shape",
                layout_property: None,
                flight_class: FlightSqlNativeNotSupportedClass::Other,
            },
            Self::UnsupportedExpression => ReasonFacts {
                code: "unsupported_expression",
                triage: TriageCode::UnsupportedExpression,
                kind: QueryGpuCandidateReasonKind::Expression,
                remedy: QueryGpuRemedy::new(
                    "rewrite_expression",
                    QueryGpuRemedyKind::UserAction,
                    "Rewrite the expression into a supported form; use the unsupported category and candidate shape to isolate the expression family.",
                ),
                tsv_category: "expression",
                layout_property: None,
                flight_class: FlightSqlNativeNotSupportedClass::Other,
            },
            Self::UnsupportedType => ReasonFacts {
                code: "unsupported_type",
                triage: TriageCode::UnsupportedType,
                kind: QueryGpuCandidateReasonKind::ArrowType,
                remedy: QueryGpuRemedy::new(
                    "cast_or_exclude_type",
                    QueryGpuRemedyKind::UserAction,
                    "Cast, exclude, or materialize unsupported column types before running the GPU query.",
                ),
                tsv_category: "type",
                layout_property: None,
                flight_class: FlightSqlNativeNotSupportedClass::Other,
            },
            Self::NativeSortPreservingMergeUnsupported => ReasonFacts {
                flight_class: FlightSqlNativeNotSupportedClass::NativeSortPreservingMerge,
                ..plan_shape("native_sort_preserving_merge_unsupported", "operator_sort")
            },
            Self::NativeDistributedAggregateUnsupported => {
                plan_shape("native_distributed_aggregate_unsupported", "aggregate_distributed")
            },
            Self::NativePartitionPreservingSortUnsupported => {
                plan_shape("native_partition_preserving_sort_unsupported", "operator_sort")
            },
            Self::NativeLocalLimitPartitionUnsupported => {
                plan_shape("native_local_limit_partition_unsupported", "operator_limit")
            },
            Self::NativeExactStatisticalInputPartitioned => {
                plan_shape("native_exact_statistical_input_partitioned", "aggregate_partitioned_input")
            },
            Self::NativeOrderedOutputUnsupported => ReasonFacts {
                layout_property: Some(DECLARED_OUTPUT_ORDERING),
                ..plan_shape("native_ordered_output_unsupported", "operator_ordering")
            },
            // Both reasons project one admission rejection: no compiled
            // pipeline contract covers the plan. A join only changes which
            // reason names it and the TSV category.
            Self::NativeBoundedJoinUnsupported => {
                compiled_contract_gap("native_bounded_join_unsupported", "operator_join")
            },
            Self::NativeCompiledPipelineUnsupported => {
                compiled_contract_gap("native_compiled_pipeline_unsupported", "compiled_pipeline_contract")
            },
            Self::NativeSourceNonLocalObjectStore => ReasonFacts {
                remedy: QueryGpuRemedy::new(
                    "stage_local_parquet",
                    QueryGpuRemedyKind::UserAction,
                    "Stage or cache the data as local Parquet until native object-store reads are supported.",
                ),
                ..source_gap("native_source_non_local_object_store")
            },
            Self::NativeSourcePushedDownParquetPredicate => ReasonFacts {
                remedy: QueryGpuRemedy::new(
                    "disable_parquet_pushdown",
                    QueryGpuRemedyKind::ConfigAction,
                    "Disable the unsupported Parquet predicate pushdown for diagnosis or rewrite the query.",
                ),
                ..source_gap("native_source_pushed_down_parquet_predicate")
            },
            Self::NativeSourcePreserveOrder => ReasonFacts {
                layout_property: Some(DECLARED_OUTPUT_ORDERING),
                ..source_gap("native_source_preserve_order")
            },
            Self::NativeSourceOutputOrdering => ReasonFacts {
                layout_property: Some(DECLARED_OUTPUT_ORDERING),
                ..source_gap("native_source_output_ordering")
            },
            Self::NativeSourceExprAdapter => source_gap("native_source_expr_adapter"),
            Self::NativeSourceParquetFileReaderFactory => source_gap("native_source_parquet_file_reader_factory"),
            Self::NativeSourceEmpty => source_gap("native_source_empty"),
            Self::NativeSourcePartitionValues => source_gap("native_source_partition_values"),
            Self::NativeSourceByteRange => source_gap("native_source_byte_range"),
            Self::NativeSourceRowSelection => source_gap("native_source_row_selection"),
            Self::NativeSourceSchemaMapping => source_gap("native_source_schema_mapping"),
        }
    }
}

/// The projections of one [`CapabilityReason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasonFacts {
    /// Report and log reason code.
    pub code: &'static str,
    /// Shared triage code of the capability miss.
    pub triage: TriageCode,
    /// Coverage reason kind.
    pub kind: QueryGpuCandidateReasonKind,
    /// Coverage remedy.
    pub remedy: QueryGpuRemedy,
    /// Planning TSV category when the plan names nothing more specific.
    pub tsv_category: &'static str,
    /// Data-layout property a table owner can change to avoid the miss.
    pub layout_property: Option<&'static str>,
    /// Flight SQL planning log class.
    pub flight_class: FlightSqlNativeNotSupportedClass,
}

/// Flight SQL planning log class of the first capability miss.
///
/// The named variant covers a common Flight SQL capability miss; every other
/// [`CapabilityReason`] logs as [`FlightSqlNativeNotSupportedClass::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightSqlNativeNotSupportedClass {
    /// DataFusion left a `SortPreservingMergeExec` edge case that native
    /// lowering could not safely reproduce.
    NativeSortPreservingMerge,
    /// Any capability-miss code outside the common Flight SQL classes.
    Other,
}

impl FlightSqlNativeNotSupportedClass {
    /// Stable snake-case label used by log lines and tests.
    ///
    /// The named label matches the corresponding [`CapabilityReason::code`]
    /// value so one keyword search across server logs covers both the
    /// capability-miss code and the class label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeSortPreservingMerge => CapabilityReason::NativeSortPreservingMergeUnsupported.code(),
            Self::Other => "other",
        }
    }
}

const DECLARED_OUTPUT_ORDERING: &str = "declared_output_ordering";

const fn plan_shape(code: &'static str, tsv_category: &'static str) -> ReasonFacts {
    ReasonFacts {
        code,
        triage: TriageCode::UnsupportedPlanShape,
        kind: QueryGpuCandidateReasonKind::PlanShape,
        remedy: QueryGpuRemedy::new(
            "use_native_friendly_session_profile",
            QueryGpuRemedyKind::ConfigAction,
            "Use a native-friendly session profile or rewrite distributed ordering and aggregate stages into a shape native lowering supports.",
        ),
        tsv_category,
        layout_property: None,
        flight_class: FlightSqlNativeNotSupportedClass::Other,
    }
}

const fn compiled_contract_gap(code: &'static str, tsv_category: &'static str) -> ReasonFacts {
    ReasonFacts {
        code,
        triage: TriageCode::CapabilityEnvelopeRejected,
        kind: QueryGpuCandidateReasonKind::PlanShape,
        remedy: QueryGpuRemedy::new(
            "compiled_contract_gap",
            QueryGpuRemedyKind::EngineGap,
            "No compiled pipeline contract covers this plan yet; the capability decision row names the missing contract.",
        ),
        tsv_category,
        layout_property: None,
        flight_class: FlightSqlNativeNotSupportedClass::Other,
    }
}

const fn source_gap(code: &'static str) -> ReasonFacts {
    ReasonFacts {
        code,
        triage: TriageCode::UnsupportedSource,
        kind: QueryGpuCandidateReasonKind::Source,
        remedy: QueryGpuRemedy::new(
            "engine_gap",
            QueryGpuRemedyKind::EngineGap,
            "This source feature is not yet representable by native lowering; rewrite or materialize a supported local Parquet snapshot, or accept the DataFusion path.",
        ),
        tsv_category: "source",
        layout_property: None,
        flight_class: FlightSqlNativeNotSupportedClass::Other,
    }
}
