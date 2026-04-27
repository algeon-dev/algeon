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
    /// The reservation depends on a lower-level cudf/libcudf contract that is
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
    /// Query-engine or cudf contract required to observe or admit the boundary.
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
        "modeled_by_query_engine_ordered_probe_input",
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

/// Public bounded join reservation contract and fact builder.
///
/// This type intentionally has no fields: it is a stable handle for the current
/// reservation contract catalogue. Methods return lower-level capability facts
/// used by contract tests and adapter diagnostics. Normal strict bounded
/// reporting is gate-first and does not project execution families or paths.
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
}
