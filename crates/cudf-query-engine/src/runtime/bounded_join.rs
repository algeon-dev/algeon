use super::{ExecutionBoundaryPolicy, ReservationKind};

mod diagnostics;

/// Status of a bounded join reservation requirement.
///
/// Status strings flow into bounded-join reservation facts. `Modeled` means the
/// engine has an explicit reservation path for that boundary; other statuses are
/// kept as reportable blockers/deferred work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundedJoinReservationStatus {
    /// The reservation is modeled and enforced by current bounded execution.
    Modeled,
    /// The reservation depends on a lower-level cudf-rs/libcudf contract that is
    /// not available yet.
    BlockedByLowerLevel,
    /// The reservation is intentionally deferred until another prerequisite is
    /// available.
    DeferredUntilPrerequisite,
}

impl BoundedJoinReservationStatus {
    /// Stable diagnostic string for this status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::BlockedByLowerLevel => "blocked_by_lower_level",
            Self::DeferredUntilPrerequisite => "deferred_until_prerequisite",
        }
    }
}

impl std::fmt::Display for BoundedJoinReservationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One row in the bounded join reservation contract.
///
/// Each requirement identifies a query-engine memory boundary plus the
/// observability or admission contract needed to bound it safely. Some modeled
/// requests are planning/admission contracts backed by metadata estimates or
/// configured thresholds rather than exact device-memory sizing. The
/// `memory_boundary`, `required_contract`, and `detail` strings are
/// report-facing diagnostics; avoid renaming them unless downstream reports and
/// tests are updated deliberately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedJoinReservationRequirement {
    /// Reservation category enforced by bounded execution.
    reservation_kind: ReservationKind,
    /// Stable memory-boundary name used in facts and reservation requests.
    memory_boundary: &'static str,
    /// Query-engine or cudf-rs contract required to observe or admit the boundary.
    required_contract: &'static str,
    /// Current modeling status for this boundary.
    status: BoundedJoinReservationStatus,
    /// Stable detail string explaining how the requirement is modeled or blocked.
    detail: &'static str,
}

impl BoundedJoinReservationRequirement {
    /// Creates a static bounded join reservation requirement.
    pub const fn new(
        reservation_kind: ReservationKind,
        memory_boundary: &'static str,
        required_contract: &'static str,
        status: BoundedJoinReservationStatus,
        detail: &'static str,
    ) -> Self {
        Self { reservation_kind, memory_boundary, required_contract, status, detail }
    }

    /// Reservation category for this requirement.
    pub const fn reservation_kind(self) -> ReservationKind {
        self.reservation_kind
    }

    /// Stable memory-boundary name for this requirement.
    pub const fn memory_boundary(self) -> &'static str {
        self.memory_boundary
    }

    /// Stable lower-level or query-engine contract string for this requirement.
    pub const fn required_contract(self) -> &'static str {
        self.required_contract
    }

    /// Current modeling status for this requirement.
    pub const fn status(self) -> BoundedJoinReservationStatus {
        self.status
    }

    /// Stable explanatory detail string.
    pub const fn detail(self) -> &'static str {
        self.detail
    }
}

/// Report-facing bounded join reservation requirement catalogue.
///
/// This list feeds contract tests, fallback reports, and unsupported-shape facts.
/// Ordering is part of the practical report contract because CSV helpers preserve
/// this order.
pub const BOUNDED_JOIN_RESERVATION_REQUIREMENTS: [BoundedJoinReservationRequirement; 11] = [
    BoundedJoinReservationRequirement::new(
        ReservationKind::SourceRead,
        "join_source_read",
        "query_engine_source_read_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_before_source_handoff",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::OperatorOutput,
        "join_operator_output",
        "query_engine_unary_output_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_bounded_resident_frame_ownership",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::BuildSide,
        "join_build_side",
        "query_engine_build_side_source_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_before_execution",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::HashTable,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_pre_build_sizing",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::ProbeSide,
        "join_probe_side",
        "query_engine_probe_side_source_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_ordered_bounded_join_execution_path",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::FilteredProbeState,
        "join_filtered_probe_state",
        "filtered_join_retained_probe_state",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_filtered_probe_state_observability",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::MixedSizeData,
        "join_mixed_size_data",
        "mixed_keyed_residual_or_conditional_full_retained_size_data",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_retained_join_size_data_observability",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::GatherMap,
        "join_gather_map",
        "join_gather_map_reservation_size_bytes",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_join_gather_map_sizing",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::JoinOutput,
        "join_output",
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::FinalSort,
        "join_pipeline_final_sort",
        "query_engine_final_sort_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_bounded_resident_frame_ownership",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::Materialization,
        "host_materialization",
        "query_engine_explicit_sink_materialization_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_explicit_sink_boundary",
    ),
];

/// Bounded join execution families emitted in reports and diagnostic facts.
///
/// `as_str()` values are contract strings consumed by tests, TSV reports, and
/// downstream diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundedJoinExecutionFamily {
    /// A single root equi join backed by a reusable hash table.
    RootReusableHashJoin,
    /// A single root semi/anti equi join backed by filtered probe state.
    RootFilteredSemiAntiJoin,
    /// A single root keyless residual-predicate join.
    KeylessConditionalJoin,
    /// A keyless conditional root join with bounded input subplans.
    KeylessConditionalJoinInputSubplanPipeline,
    /// A keyless conditional root join with aggregate-over-join subplan inputs.
    KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    /// A single root projected cross join.
    RootProjectedCrossJoin,
    /// A single root keyed join that also evaluates a residual predicate.
    RootMixedKeyedResidualJoin,
    /// A multi-join tree using reusable hash-join families.
    MultiJoinReusableHashPipeline,
    /// A multi-join tree that may include keyed residual inner/semi/anti joins.
    MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
    /// A multi-join tree filtered by a grouped aggregate over another bounded
    /// join subplan, followed by projection and final TopK.
    MultiJoinAggregateOverJoinSubplanTopKPipeline,
    /// A keyed residual root join whose inputs are a bounded join subtree and a
    /// projection-wrapped grouped local-Parquet aggregate.
    MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
    /// A left-semi keyed root hash join with a resident grouped aggregate input
    /// and post-join grouped aggregate TopK.
    HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline,
    /// A semi-filtered multi-join tree with a resident grouped aggregate
    /// subquery branch and final sort.
    MultiJoinResidentGroupedAggregateSubqueryFilterPipeline,
    /// A final sort over keyed hash joins with resident grouped/scalar revenue
    /// subplans.
    KeyedHashJoinResidentSubplanPipeline,
}

impl BoundedJoinExecutionFamily {
    /// Stable diagnostic string for this execution family.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RootReusableHashJoin => "root_reusable_hash_join",
            Self::RootFilteredSemiAntiJoin => "root_filtered_semi_anti_join",
            Self::KeylessConditionalJoin => "keyless_conditional_join",
            Self::KeylessConditionalJoinInputSubplanPipeline => "keyless_conditional_join_input_subplan_pipeline",
            Self::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline => {
                "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline"
            },
            Self::RootProjectedCrossJoin => "root_projected_cross_join",
            Self::RootMixedKeyedResidualJoin => "root_mixed_keyed_residual_join",
            Self::MultiJoinReusableHashPipeline => "multi_join_reusable_hash_pipeline",
            Self::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline => {
                "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline"
            },
            Self::MultiJoinAggregateOverJoinSubplanTopKPipeline => "multi_join_with_aggregate_over_join_subplan_topk",
            Self::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline => {
                "mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline"
            },
            Self::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline => {
                "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk"
            },
            Self::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline => {
                "multi_join_with_resident_grouped_aggregate_subquery_filter"
            },
            Self::KeyedHashJoinResidentSubplanPipeline => "keyed_hash_join_resident_subplan_pipeline",
        }
    }
}

/// Stable admission model label for the current bounded join family catalogue.
///
/// Phase A of the native pipeline-DAG migration keeps these families as a
/// strict transitional admission/dispatch surface. The value is intentionally
/// reportable so downstream diagnostics do not mistake family admission for the
/// future capability-analysis source of truth.
pub const BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL: &str = "transitional_topology_family_catalogue";

/// Stable scope label for newly added bounded join families.
///
/// New family recognizers must be topology-only: they may describe plan shape,
/// join kinds, blocking operators, residency, and ordered boundaries, but not a
/// TPC-H query id, table name, column name, literal constant, file path, or SQL
/// text.
pub const BOUNDED_JOIN_TOPOLOGY_ONLY_SCOPE: &str =
    "new_families_must_be_topology_only_no_query_id_table_name_column_name_literal";

/// Phase A contract row for a report-facing bounded join execution family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedJoinExecutionFamilyContract {
    family: BoundedJoinExecutionFamily,
    admission_model: &'static str,
    topology_contract: &'static str,
}

impl BoundedJoinExecutionFamilyContract {
    /// Creates a static execution-family contract row.
    pub const fn new(
        family: BoundedJoinExecutionFamily,
        admission_model: &'static str,
        topology_contract: &'static str,
    ) -> Self {
        Self { family, admission_model, topology_contract }
    }

    /// Execution family described by this row.
    pub const fn family(self) -> BoundedJoinExecutionFamily {
        self.family
    }

    /// Admission model for this family.
    pub const fn admission_model(self) -> &'static str {
        self.admission_model
    }

    /// Topology-only shape contract for this family.
    pub const fn topology_contract(self) -> &'static str {
        self.topology_contract
    }
}

/// Report-facing catalogue of bounded join execution families.
///
/// Ordering is preserved in CSV facts and tests.
pub const BOUNDED_JOIN_EXECUTION_FAMILIES: [BoundedJoinExecutionFamily; 14] = [
    BoundedJoinExecutionFamily::RootReusableHashJoin,
    BoundedJoinExecutionFamily::RootFilteredSemiAntiJoin,
    BoundedJoinExecutionFamily::KeylessConditionalJoin,
    BoundedJoinExecutionFamily::KeylessConditionalJoinInputSubplanPipeline,
    BoundedJoinExecutionFamily::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    BoundedJoinExecutionFamily::RootProjectedCrossJoin,
    BoundedJoinExecutionFamily::RootMixedKeyedResidualJoin,
    BoundedJoinExecutionFamily::MultiJoinReusableHashPipeline,
    BoundedJoinExecutionFamily::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
    BoundedJoinExecutionFamily::MultiJoinAggregateOverJoinSubplanTopKPipeline,
    BoundedJoinExecutionFamily::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
    BoundedJoinExecutionFamily::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline,
    BoundedJoinExecutionFamily::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline,
    BoundedJoinExecutionFamily::KeyedHashJoinResidentSubplanPipeline,
];

/// Phase A topology contracts for bounded join execution families.
///
/// Ordering mirrors `BOUNDED_JOIN_EXECUTION_FAMILIES`. These strings are
/// intentionally structural; avoid query-id, table-name, column-name, literal,
/// path, or SQL-text vocabulary when adding a family.
pub const BOUNDED_JOIN_EXECUTION_FAMILY_CONTRACTS: [BoundedJoinExecutionFamilyContract; 14] = [
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::RootReusableHashJoin,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "single_root_equi_hash_join_with_source_filter_projection_leaf_inputs",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::RootFilteredSemiAntiJoin,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "single_root_filtered_semi_anti_hash_join_with_source_filter_projection_leaf_inputs",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::KeylessConditionalJoin,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "single_root_keyless_conditional_join_with_source_filter_projection_leaf_inputs",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::KeylessConditionalJoinInputSubplanPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "keyless_conditional_root_join_with_bounded_projection_wrapped_input_subplans",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "keyless_conditional_root_join_with_aggregate_over_join_subplan_inputs_and_unary_post_root_pipeline",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::RootProjectedCrossJoin,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "single_root_projected_cross_join_with_source_filter_projection_leaf_inputs",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::RootMixedKeyedResidualJoin,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "single_root_keyed_hash_join_with_residual_predicate_and_source_filter_projection_leaf_inputs",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::MultiJoinReusableHashPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "multi_join_tree_of_reusable_hash_join_operators_with_unary_post_root_pipeline",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "multi_join_tree_with_keyed_residual_inner_semi_anti_joins_and_unary_post_root_pipeline",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::MultiJoinAggregateOverJoinSubplanTopKPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "topk_over_projection_over_root_hash_join_with_multi_join_input_and_aggregate_over_join_subplan_input",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "projected_aggregate_over_keyed_residual_root_join_with_join_subtree_input_and_resident_grouped_aggregate_input",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "topk_over_grouped_aggregate_over_hash_join_with_multi_join_input_and_resident_grouped_aggregate_input",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "final_sort_over_multi_join_tree_with_resident_grouped_aggregate_subquery_filter_branch",
    ),
    BoundedJoinExecutionFamilyContract::new(
        BoundedJoinExecutionFamily::KeyedHashJoinResidentSubplanPipeline,
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL,
        "final_sort_over_keyed_hash_joins_with_resident_grouped_subplan_and_scalar_aggregate_branch",
    ),
];

/// Public bounded join reservation contract and fact builder.
///
/// This type intentionally has no fields: it is a stable handle for the current
/// contract catalogue. Methods return report-facing strings and fact maps used
/// by contract tests and adapter diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BoundedJoinReservationContract;

impl BoundedJoinReservationContract {
    /// Returns the bounded join reservation requirements in report order.
    pub const fn requirements(self) -> &'static [BoundedJoinReservationRequirement] {
        &BOUNDED_JOIN_RESERVATION_REQUIREMENTS
    }

    /// Stable model-status fact value.
    pub const fn model_status(self) -> &'static str {
        "modeled"
    }

    /// Stable execution-status fact value for supported bounded join families.
    pub const fn execution_status(self) -> &'static str {
        "enabled"
    }

    /// Returns whether bounded join execution is enabled by this contract.
    pub const fn enabled(self) -> bool {
        true
    }

    /// Returns supported bounded join execution families in report order.
    pub const fn execution_families(self) -> &'static [BoundedJoinExecutionFamily] {
        &BOUNDED_JOIN_EXECUTION_FAMILIES
    }

    /// Returns the Phase A topology contract rows in report order.
    pub const fn execution_family_contracts(self) -> &'static [BoundedJoinExecutionFamilyContract] {
        &BOUNDED_JOIN_EXECUTION_FAMILY_CONTRACTS
    }

    /// Stable admission-model fact value.
    pub const fn admission_model(self) -> &'static str {
        BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL
    }

    /// Stable admission-scope fact value for new bounded join families.
    pub const fn admission_scope(self) -> &'static str {
        BOUNDED_JOIN_TOPOLOGY_ONLY_SCOPE
    }
}
