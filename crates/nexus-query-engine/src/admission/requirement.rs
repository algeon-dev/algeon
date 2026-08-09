//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

#[cfg(test)]
use std::collections::BTreeSet;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::{
    array::{Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, Schema, SchemaRef, TimeUnit},
};

use crate::{
    admission::{
        AttemptReservationEnvelope, AttemptReservationPolicy,
        envelope::{
            AttemptReservationConsumerState, AttemptReservationProducerOccurrence, AttemptReservationSchedule,
            AttemptReservationShape, canonical_execution_scope_fingerprint,
        },
        string_payload::{PlanColumnEvidence, PlanColumnEvidenceAnalysis},
    },
    capability::ReservationRequirement,
    datatypes::cudf_dtype_from_arrow_type,
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateFunction, Expr, LiteralValue, alias_wrapped_column, representation_cast_preserves_identity},
    graph_memory::{
        GraphMemoryAdmissionEvidence, GraphMemoryOwnerTermFact, GraphMemoryOwnerTermRole, GraphSourceStageLifetime,
        query_plan_hard_row_upper_bound, query_plan_node_hard_row_upper_bound,
    },
    plan::{
        AggregateNode, JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, ResultSink,
        estimated_local_parquet_source_bytes, estimated_reachable_local_parquet_source_bytes,
    },
    runtime::ReservationKind,
    source::SourceNode,
};

/// Trustworthiness of the descriptive working-set hint.
///
/// Ordering is `Unknown < Modeled < Measured`; aggregation retains the weakest
/// evidence so a multi-fragment invocation never overstates its estimate.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EstimateQuality {
    #[default]
    Unknown,
    Modeled,
    Measured,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum OwnerId {
    Attempt,
    Graph(NodeId),
    Producer(NodeId),
    Aggregate(NodeId),
    Source(NodeId),
    Consumer(NodeId),
    SideInput(NodeId),
    Sort(NodeId),
    Limit(NodeId),
    CacheLedger,
}

impl OwnerId {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Attempt => "attempt",
            Self::Graph(_) => "graph",
            Self::Producer(_) => "producer",
            Self::Aggregate(_) => "aggregate",
            Self::Source(_) => "source",
            Self::Consumer(_) => "consumer",
            Self::SideInput(_) => "side_input",
            Self::Sort(_) => "sort",
            Self::Limit(_) => "limit",
            Self::CacheLedger => "cache_ledger",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum OwnerTermRole {
    AttemptWideFixed,
    GraphPersistent,
    ConstructionTransient,
    ExecutionWorkspace,
    GraphLiveReservationFloorTopUp,
    SideInputWorkspace,
    AggregatePartialState,
    AggregateOutput,
    RetainedOutput,
    SourceStage,
    ConsumerOutput,
    ConsumerPredicateMask,
    ConsumerHashTable,
    ConsumerPairIndices,
    ConsumerWorkspace,
    ConsumerProjectionCastWorkspace,
    ConsumerProjectionExpressionWorkspace,
    SideInputFrame,
    LimitGatherMap,
    LimitNativeWorkspace,
    LimitOutputFrame,
    LimitOutputNullMask,
    SortKeyInput,
    SortKeyOutput,
    SortSequence,
    SortTemporaryWorkspace,
    SortRetainedOrder,
    SortOutputFrame,
    SortOutputNullMask,
}

impl OwnerTermRole {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::AttemptWideFixed => "attempt_wide_fixed",
            Self::GraphPersistent => "graph_persistent",
            Self::ConstructionTransient => "construction_transient",
            Self::ExecutionWorkspace => "execution_workspace",
            Self::GraphLiveReservationFloorTopUp => "graph_live_reservation_floor_top_up",
            Self::SideInputWorkspace => "side_input_workspace",
            Self::AggregatePartialState => "aggregate_partial_state",
            Self::AggregateOutput => "aggregate_output",
            Self::RetainedOutput => "retained_output",
            Self::SourceStage => "source_stage",
            Self::ConsumerOutput => "consumer_output",
            Self::ConsumerPredicateMask => "consumer_predicate_mask",
            Self::ConsumerHashTable => "consumer_hash_table",
            Self::ConsumerPairIndices => "consumer_pair_indices",
            Self::ConsumerWorkspace => "consumer_workspace",
            Self::ConsumerProjectionCastWorkspace => "consumer_projection_cast_workspace",
            Self::ConsumerProjectionExpressionWorkspace => "consumer_projection_expression_workspace",
            Self::SideInputFrame => "side_input_frame",
            Self::LimitGatherMap => "limit_gather_map",
            Self::LimitNativeWorkspace => "limit_native_workspace",
            Self::LimitOutputFrame => "limit_output_frame",
            Self::LimitOutputNullMask => "limit_output_null_mask",
            Self::SortKeyInput => "sort_key_input",
            Self::SortKeyOutput => "sort_key_output",
            Self::SortSequence => "sort_sequence",
            Self::SortTemporaryWorkspace => "sort_temporary_workspace",
            Self::SortRetainedOrder => "sort_retained_order",
            Self::SortOutputFrame => "sort_output_frame",
            Self::SortOutputNullMask => "sort_output_null_mask",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum AllocationClass {
    AttemptWideFixed,
    ActivePhase,
    AttemptRetained,
}

impl AllocationClass {
    const fn as_str(self) -> &'static str {
        match self {
            Self::AttemptWideFixed => "attempt_wide_fixed",
            Self::ActivePhase => "active_phase",
            Self::AttemptRetained => "attempt_retained",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerTermConstructionPolicy {
    Generic,
    AllowLiteralBatch,
    PriceLiteralBatch,
    AllowMixedSourceComposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum OccurrenceId {
    PlanLocal(NodeId),
    Composed(String),
}

impl OccurrenceId {
    pub(crate) fn as_str(&self) -> String {
        match self {
            Self::PlanLocal(node_id) => format!("plan_local:{node_id}"),
            Self::Composed(occurrence_id) => format!("composed:{occurrence_id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct OwnerTermId {
    owner: OwnerId,
    role: OwnerTermRole,
    occurrence: OccurrenceId,
    execution_scope_fingerprint: String,
}

impl OwnerTermId {
    pub(crate) fn role_label(&self) -> &'static str {
        self.role.as_str()
    }

    pub(crate) fn occurrence_label(&self) -> String {
        self.occurrence.as_str()
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    pub(crate) fn is_composed_occurrence(&self) -> bool {
        matches!(self.occurrence, OccurrenceId::Composed(_))
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_test(occurrence_id: &str, execution_scope_fingerprint: &str) -> Self {
        Self {
            owner: OwnerId::Attempt,
            role: OwnerTermRole::AttemptWideFixed,
            occurrence: OccurrenceId::Composed(occurrence_id.to_owned()),
            execution_scope_fingerprint: execution_scope_fingerprint.to_owned(),
        }
    }
}

/// Internal provenance for the finite bound selected for one admitted
/// attempt.  This is engine evidence, not a caller-selectable policy mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttemptBoundDerivation {
    OrderedTimeline,
    ConservativeOwnerSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SortRadixPreflight {
    key_input_bytes: u64,
    key_output_bytes: u64,
    sequence_bytes: u64,
    temporary_workspace_bytes: u64,
    retained_order_bytes: u64,
    active_phase_peak_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LimitRangeGatherPreflight {
    gather_map_bytes: u64,
    output_frame_bytes: u64,
    output_null_mask_bytes: u64,
    native_workspace_bytes: u64,
    active_phase_peak_bytes: u64,
}

impl LimitRangeGatherPreflight {
    fn checked_from_cudf(preflight: cudf::ops::copying::GatherFixedWidthDontCheckPreflight) -> Result<Self> {
        let gather_map_bytes = u64::try_from(preflight.gather_map_bytes())
            .map_err(|_| limit_owner_error("native_byte_count_exceeds_u64", "gather_map_bytes"))?;
        let output_frame_bytes = u64::try_from(preflight.output_frame_bytes())
            .map_err(|_| limit_owner_error("native_byte_count_exceeds_u64", "output_frame_bytes"))?;
        let output_null_mask_bytes = u64::try_from(preflight.output_null_mask_bytes())
            .map_err(|_| limit_owner_error("native_byte_count_exceeds_u64", "output_null_mask_bytes"))?;
        let native_workspace_bytes = u64::try_from(preflight.native_workspace_bytes())
            .map_err(|_| limit_owner_error("native_byte_count_exceeds_u64", "native_workspace_bytes"))?;
        let active_phase_peak_bytes = u64::try_from(preflight.active_phase_peak_bytes())
            .map_err(|_| limit_owner_error("native_byte_count_exceeds_u64", "active_phase_peak_bytes"))?;
        let expected_peak = gather_map_bytes
            .checked_add(output_frame_bytes)
            .and_then(|value| value.checked_add(output_null_mask_bytes))
            .and_then(|value| value.checked_add(native_workspace_bytes))
            .ok_or_else(|| limit_owner_error("component_sum_overflow", "active_phase_peak_bytes"))?;
        if expected_peak != active_phase_peak_bytes {
            return Err(limit_owner_error("active_phase_component_sum_mismatch", "active_phase_peak_bytes"));
        }
        Ok(Self {
            gather_map_bytes,
            output_frame_bytes,
            output_null_mask_bytes,
            native_workspace_bytes,
            active_phase_peak_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        gather_map_bytes: u64,
        output_frame_bytes: u64,
        output_null_mask_bytes: u64,
        native_workspace_bytes: u64,
    ) -> Self {
        Self {
            gather_map_bytes,
            output_frame_bytes,
            output_null_mask_bytes,
            native_workspace_bytes,
            active_phase_peak_bytes: gather_map_bytes
                .checked_add(output_frame_bytes)
                .and_then(|value| value.checked_add(output_null_mask_bytes))
                .and_then(|value| value.checked_add(native_workspace_bytes))
                .expect("test limit preflight components fit"),
        }
    }
}

impl SortRadixPreflight {
    fn checked_from_cudf(preflight: cudf::ops::sorting::SortedOrderRadixPreflight) -> Result<Self> {
        let key_input_bytes = u64::try_from(preflight.key_input_bytes())
            .map_err(|_| sort_owner_error("native_byte_count_exceeds_u64", "key_input_bytes"))?;
        let key_output_bytes = u64::try_from(preflight.key_output_bytes())
            .map_err(|_| sort_owner_error("native_byte_count_exceeds_u64", "key_output_bytes"))?;
        let sequence_bytes = u64::try_from(preflight.sequence_bytes())
            .map_err(|_| sort_owner_error("native_byte_count_exceeds_u64", "sequence_bytes"))?;
        let temporary_workspace_bytes = u64::try_from(preflight.temporary_workspace_bytes())
            .map_err(|_| sort_owner_error("native_byte_count_exceeds_u64", "temporary_workspace_bytes"))?;
        let retained_order_bytes = u64::try_from(preflight.retained_order_bytes())
            .map_err(|_| sort_owner_error("native_byte_count_exceeds_u64", "retained_order_bytes"))?;
        let active_phase_peak_bytes = u64::try_from(preflight.active_phase_peak_bytes())
            .map_err(|_| sort_owner_error("native_byte_count_exceeds_u64", "active_phase_peak_bytes"))?;
        let expected_peak = key_input_bytes
            .checked_add(key_output_bytes)
            .and_then(|value| value.checked_add(sequence_bytes))
            .and_then(|value| value.checked_add(temporary_workspace_bytes))
            .and_then(|value| value.checked_add(retained_order_bytes))
            .ok_or_else(|| sort_owner_error("component_sum_overflow", "active_phase_peak_bytes"))?;
        if expected_peak != active_phase_peak_bytes {
            return Err(sort_owner_error("active_phase_peak_component_sum_mismatch", "active_phase_peak_bytes"));
        }
        if sequence_bytes != retained_order_bytes {
            return Err(sort_owner_error("sequence_retained_component_mismatch", "retained_order_bytes"));
        }
        Ok(Self {
            key_input_bytes,
            key_output_bytes,
            sequence_bytes,
            temporary_workspace_bytes,
            retained_order_bytes,
            active_phase_peak_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        key_output_bytes: u64,
        sequence_bytes: u64,
        temporary_workspace_bytes: u64,
        retained_order_bytes: u64,
    ) -> Self {
        Self::new_for_test_with_key_input(
            0,
            key_output_bytes,
            sequence_bytes,
            temporary_workspace_bytes,
            retained_order_bytes,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_for_test_with_key_input(
        key_input_bytes: u64,
        key_output_bytes: u64,
        sequence_bytes: u64,
        temporary_workspace_bytes: u64,
        retained_order_bytes: u64,
    ) -> Self {
        Self {
            key_input_bytes,
            key_output_bytes,
            sequence_bytes,
            temporary_workspace_bytes,
            retained_order_bytes,
            active_phase_peak_bytes: key_input_bytes
                .checked_add(key_output_bytes)
                .and_then(|value| value.checked_add(sequence_bytes))
                .and_then(|value| value.checked_add(temporary_workspace_bytes))
                .and_then(|value| value.checked_add(retained_order_bytes))
                .expect("test preflight components fit"),
        }
    }
}

/// Device-independent metadata for the one supported sorted-order radix path.
///
/// The device-dependent CUB workspace query is intentionally deferred until a
/// candidate ordinal is selected.  No ambient device participates in this
/// descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SortMetadataDescriptor {
    pub(crate) node_id: NodeId,
    pub(crate) execution_scope_fingerprint: String,
    pub(crate) num_rows: usize,
    pub(crate) key_type: cudf::DataType,
    pub(crate) null_count: usize,
    pub(crate) stable: bool,
    pub(crate) key_order: cudf::ops::sorting::Order,
    pub(crate) output_frame_bytes: u64,
    pub(crate) output_null_mask_bytes: u64,
    pub(crate) limit: Option<LimitMetadataDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LimitMetadataDescriptor {
    pub(crate) node_id: NodeId,
    pub(crate) execution_scope_fingerprint: String,
    pub(crate) input_rows: u64,
    pub(crate) output_rows: usize,
    pub(crate) output_columns: Vec<(cudf::DataType, bool)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FiniteBytes(u64);

impl FiniteBytes {
    fn new(bytes: u64) -> Result<Self> {
        if bytes == u64::MAX {
            return Err(owner_term_error("saturated", None, None));
        }
        Ok(Self(bytes))
    }

    const fn get(self) -> u64 {
        self.0
    }
}

fn owner_term_shape_is_valid(owner: &OwnerId, role: OwnerTermRole, class: AllocationClass) -> bool {
    if role == OwnerTermRole::AttemptWideFixed {
        return owner == &OwnerId::Attempt && class == AllocationClass::AttemptWideFixed;
    }
    match owner {
        OwnerId::Attempt => false,
        OwnerId::Graph(_) => match role {
            OwnerTermRole::GraphPersistent
            | OwnerTermRole::ConstructionTransient
            | OwnerTermRole::ExecutionWorkspace
            | OwnerTermRole::GraphLiveReservationFloorTopUp
            | OwnerTermRole::SideInputWorkspace => class == AllocationClass::ActivePhase,
            OwnerTermRole::RetainedOutput => class == AllocationClass::AttemptRetained,
            _ => false,
        },
        OwnerId::Producer(_) => role == OwnerTermRole::RetainedOutput && class == AllocationClass::AttemptRetained,
        OwnerId::Aggregate(_) => {
            matches!(role, OwnerTermRole::AggregatePartialState | OwnerTermRole::AggregateOutput)
                && class == AllocationClass::ActivePhase
        },
        OwnerId::Source(_) => role == OwnerTermRole::SourceStage && class == AllocationClass::ActivePhase,
        OwnerId::Consumer(_) => {
            matches!(
                role,
                OwnerTermRole::ConsumerOutput
                    | OwnerTermRole::ConsumerPredicateMask
                    | OwnerTermRole::ConsumerHashTable
                    | OwnerTermRole::ConsumerPairIndices
                    | OwnerTermRole::ConsumerWorkspace
                    | OwnerTermRole::ConsumerProjectionCastWorkspace
                    | OwnerTermRole::ConsumerProjectionExpressionWorkspace
            ) && class == AllocationClass::ActivePhase
        },
        OwnerId::SideInput(_) => role == OwnerTermRole::SideInputFrame && class == AllocationClass::ActivePhase,
        OwnerId::Sort(_) => match role {
            OwnerTermRole::SortKeyInput
            | OwnerTermRole::SortKeyOutput
            | OwnerTermRole::SortSequence
            | OwnerTermRole::SortTemporaryWorkspace => class == AllocationClass::ActivePhase,
            OwnerTermRole::SortRetainedOrder | OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask => {
                class == AllocationClass::AttemptRetained
            },
            _ => false,
        },
        OwnerId::Limit(_) => match role {
            OwnerTermRole::LimitGatherMap | OwnerTermRole::LimitNativeWorkspace => {
                class == AllocationClass::ActivePhase
            },
            OwnerTermRole::LimitOutputFrame | OwnerTermRole::LimitOutputNullMask => {
                class == AllocationClass::AttemptRetained
            },
            _ => false,
        },
        OwnerId::CacheLedger => false,
    }
}

fn validate_retention_shape(
    role: OwnerTermRole,
    class: AllocationClass,
    retained_row_count_upper_bound: Option<u64>,
) -> Result<()> {
    if role == OwnerTermRole::RetainedOutput {
        if class != AllocationClass::AttemptRetained || retained_row_count_upper_bound.is_none() {
            return Err(owner_term_error("retained_output_shape_invalid", Some(role), Some(class)));
        }
    } else if matches!(
        role,
        OwnerTermRole::SortRetainedOrder
            | OwnerTermRole::SortOutputFrame
            | OwnerTermRole::SortOutputNullMask
            | OwnerTermRole::LimitOutputFrame
            | OwnerTermRole::LimitOutputNullMask
    ) {
        if class != AllocationClass::AttemptRetained || retained_row_count_upper_bound.is_some() {
            return Err(owner_term_error("retained_sort_output_shape_invalid", Some(role), Some(class)));
        }
    } else if class == AllocationClass::AttemptRetained || retained_row_count_upper_bound.is_some() {
        return Err(owner_term_error("non_retained_output_shape_invalid", Some(role), Some(class)));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AllocationOwnerTerm {
    id: OwnerTermId,
    class: AllocationClass,
    bytes: FiniteBytes,
    retained_row_count_upper_bound: Option<u64>,
}

impl AllocationOwnerTerm {
    fn new(
        owner: OwnerId,
        role: OwnerTermRole,
        occurrence: OccurrenceId,
        execution_scope_fingerprint: String,
        class: AllocationClass,
        bytes: u64,
        retained_row_count_upper_bound: Option<u64>,
    ) -> Result<Self> {
        if owner == OwnerId::CacheLedger {
            return Err(owner_term_error("cache_ledger_owner_forbidden", None, None));
        }
        if !owner_term_shape_is_valid(&owner, role, class) {
            return Err(owner_term_error("owner_role_class_mismatch", Some(role), Some(class))
                .with_fact("owner", owner.as_str()));
        }
        if execution_scope_fingerprint.is_empty() {
            return Err(owner_term_error("missing_scope", None, None));
        }
        if matches!(occurrence, OccurrenceId::Composed(ref value) if value.is_empty()) {
            return Err(owner_term_error("missing_occurrence", None, None));
        }
        validate_retention_shape(role, class, retained_row_count_upper_bound)?;
        Ok(Self {
            id: OwnerTermId { owner, role, occurrence, execution_scope_fingerprint },
            class,
            bytes: FiniteBytes::new(bytes)?,
            retained_row_count_upper_bound,
        })
    }

    fn rebind_composed_occurrence(&self, occurrence_id: &str) -> Result<Self> {
        let occurrence = match &self.id.occurrence {
            OccurrenceId::PlanLocal(_) => occurrence_id.to_owned(),
            OccurrenceId::Composed(inner) => {
                format!("nested:{}:{occurrence_id}{inner}", occurrence_id.len())
            },
        };
        Self::new(
            self.id.owner.clone(),
            self.id.role,
            OccurrenceId::Composed(occurrence),
            self.id.execution_scope_fingerprint.clone(),
            self.class,
            self.bytes.get(),
            self.retained_row_count_upper_bound,
        )
    }

    #[cfg(test)]
    pub(crate) fn rebind_composed_for_test(&self, occurrence_id: &str) -> Result<Self> {
        self.rebind_composed_occurrence(occurrence_id)
    }

    pub(crate) fn id(&self) -> &OwnerTermId {
        &self.id
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes.get()
    }

    pub(crate) fn owner_label(&self) -> &'static str {
        self.id.owner.as_str()
    }

    pub(crate) const fn is_retained_output(&self) -> bool {
        matches!(self.id.role, OwnerTermRole::RetainedOutput)
    }

    pub(crate) const fn is_side_input_frame(&self) -> bool {
        matches!(self.id.role, OwnerTermRole::SideInputFrame)
    }

    pub(crate) fn is_attempt_wide_fixed(&self) -> bool {
        matches!(self.id.owner, OwnerId::Attempt)
            && self.id.role == OwnerTermRole::AttemptWideFixed
            && self.class == AllocationClass::AttemptWideFixed
    }

    pub(crate) const fn retained_row_count_upper_bound(&self) -> Option<u64> {
        self.retained_row_count_upper_bound
    }

    pub(crate) const fn is_graph_phase_term(&self) -> bool {
        matches!(self.id.owner, OwnerId::Graph(_))
            && matches!(
                self.id.role,
                OwnerTermRole::GraphPersistent
                    | OwnerTermRole::ConstructionTransient
                    | OwnerTermRole::ExecutionWorkspace
                    | OwnerTermRole::GraphLiveReservationFloorTopUp
                    | OwnerTermRole::SideInputWorkspace
            )
    }

    pub(crate) const fn is_graph_persistent(&self) -> bool {
        matches!(self.id.role, OwnerTermRole::GraphPersistent)
    }

    pub(crate) const fn is_graph_phase_alternative(&self) -> bool {
        self.is_graph_persistent() || matches!(self.id.role, OwnerTermRole::GraphLiveReservationFloorTopUp)
    }

    pub(crate) const fn is_source_stage(&self) -> bool {
        matches!(self.id.role, OwnerTermRole::SourceStage)
    }

    pub(crate) const fn is_consumer_schedule_owner(&self) -> bool {
        matches!(
            self.id.owner,
            OwnerId::Graph(_)
                | OwnerId::Aggregate(_)
                | OwnerId::Source(_)
                | OwnerId::Consumer(_)
                | OwnerId::SideInput(_)
                | OwnerId::Sort(_)
                | OwnerId::Limit(_)
        )
    }

    pub(crate) fn has_consumer_schedule_identity(&self) -> bool {
        match (&self.id.owner, self.id.role, &self.id.occurrence) {
            (OwnerId::Graph(node_id), role, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
                    && matches!(
                        role,
                        OwnerTermRole::GraphPersistent
                            | OwnerTermRole::ConstructionTransient
                            | OwnerTermRole::ExecutionWorkspace
                            | OwnerTermRole::GraphLiveReservationFloorTopUp
                            | OwnerTermRole::SideInputWorkspace
                            | OwnerTermRole::RetainedOutput
                    )
            },
            (OwnerId::Source(node_id), OwnerTermRole::SourceStage, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
            },
            (OwnerId::Aggregate(node_id), role, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
                    && matches!(role, OwnerTermRole::AggregatePartialState | OwnerTermRole::AggregateOutput)
            },
            (OwnerId::Consumer(node_id), role, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
                    && matches!(
                        role,
                        OwnerTermRole::ConsumerOutput
                            | OwnerTermRole::ConsumerPredicateMask
                            | OwnerTermRole::ConsumerHashTable
                            | OwnerTermRole::ConsumerPairIndices
                            | OwnerTermRole::ConsumerWorkspace
                            | OwnerTermRole::ConsumerProjectionCastWorkspace
                            | OwnerTermRole::ConsumerProjectionExpressionWorkspace
                    )
            },
            (OwnerId::SideInput(node_id), OwnerTermRole::SideInputFrame, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
            },
            (OwnerId::Sort(node_id), role, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
                    && matches!(
                        role,
                        OwnerTermRole::SortKeyInput
                            | OwnerTermRole::SortKeyOutput
                            | OwnerTermRole::SortSequence
                            | OwnerTermRole::SortTemporaryWorkspace
                            | OwnerTermRole::SortRetainedOrder
                            | OwnerTermRole::SortOutputFrame
                            | OwnerTermRole::SortOutputNullMask
                    )
            },
            (OwnerId::Limit(node_id), role, OccurrenceId::PlanLocal(occurrence)) => {
                node_id == occurrence
                    && matches!(
                        role,
                        OwnerTermRole::LimitGatherMap
                            | OwnerTermRole::LimitNativeWorkspace
                            | OwnerTermRole::LimitOutputFrame
                            | OwnerTermRole::LimitOutputNullMask
                    )
            },
            _ => false,
        }
    }

    pub(crate) const fn is_construction_phase_term(&self) -> bool {
        matches!(
            self.id.role,
            OwnerTermRole::GraphPersistent
                | OwnerTermRole::ConstructionTransient
                | OwnerTermRole::GraphLiveReservationFloorTopUp
        )
    }

    pub(crate) const fn is_execution_phase_term(&self) -> bool {
        matches!(
            self.id.role,
            OwnerTermRole::GraphPersistent
                | OwnerTermRole::ExecutionWorkspace
                | OwnerTermRole::GraphLiveReservationFloorTopUp
                | OwnerTermRole::SideInputWorkspace
        )
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.id.execution_scope_fingerprint
    }

    pub(crate) fn provides_complete_phase_coverage(&self) -> bool {
        !matches!(self.id.role, OwnerTermRole::SourceStage)
    }
}

fn owner_term_error(reason: &'static str, role: Option<OwnerTermRole>, class: Option<AllocationClass>) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "attempt owner-term inventory is unavailable",
    )
    .with_fact("reason", format!("attempt_owner_term_{reason}"));
    let error = match role {
        Some(role) => error.with_fact("owner_term_role", role.as_str()),
        None => error,
    };
    match class {
        Some(class) => error.with_fact("allocation_class", class.as_str()),
        None => error,
    }
}

fn owner_derivation_error(reason: impl ToString) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "attempt bound derivation is unavailable",
    )
    .with_fact("reason", reason)
    .with_fact("derivation", "conservative_owner_sum")
}

fn sort_owner_error(reason: &'static str, resource: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "sorted-order owner preflight is unavailable",
    )
    .with_fact("reason", format!("sort_owner_{reason}"))
    .with_fact("resource", resource)
}

fn limit_owner_error(reason: &'static str, resource: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "bounded limit range-gather owner preflight is unavailable",
    )
    .with_fact("reason", format!("limit_owner_{reason}"))
    .with_fact("resource", resource)
}

fn graph_owner_term_from_fact(fact: &GraphMemoryOwnerTermFact) -> Result<AllocationOwnerTerm> {
    let (role, class) = match fact.role() {
        GraphMemoryOwnerTermRole::GraphPersistent => (OwnerTermRole::GraphPersistent, AllocationClass::ActivePhase),
        GraphMemoryOwnerTermRole::ConstructionTransient => {
            (OwnerTermRole::ConstructionTransient, AllocationClass::ActivePhase)
        },
        GraphMemoryOwnerTermRole::ExecutionWorkspace => {
            (OwnerTermRole::ExecutionWorkspace, AllocationClass::ActivePhase)
        },
        GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp => {
            (OwnerTermRole::GraphLiveReservationFloorTopUp, AllocationClass::ActivePhase)
        },
        GraphMemoryOwnerTermRole::SideInputWorkspace => {
            (OwnerTermRole::SideInputWorkspace, AllocationClass::ActivePhase)
        },
        GraphMemoryOwnerTermRole::RetainedOutput => (OwnerTermRole::RetainedOutput, AllocationClass::AttemptRetained),
    };
    AllocationOwnerTerm::new(
        OwnerId::Graph(fact.graph_node_id()),
        role,
        OccurrenceId::PlanLocal(fact.graph_node_id()),
        fact.execution_scope_fingerprint().to_owned(),
        class,
        fact.bytes(),
        fact.row_count_upper_bound(),
    )
}

fn validate_owner_term(term: &AllocationOwnerTerm) -> Result<()> {
    if term.id.owner == OwnerId::CacheLedger {
        return Err(owner_term_error("cache_ledger_owner_forbidden", Some(term.id.role), Some(term.class)));
    }
    if !owner_term_shape_is_valid(&term.id.owner, term.id.role, term.class) {
        return Err(owner_term_error("owner_role_class_mismatch", Some(term.id.role), Some(term.class))
            .with_fact("owner", term.id.owner.as_str()));
    }
    validate_retention_shape(term.id.role, term.class, term.retained_row_count_upper_bound)?;
    if term.bytes.get() == u64::MAX {
        return Err(owner_term_error("saturated", Some(term.id.role), Some(term.class)));
    }
    if term.id.execution_scope_fingerprint.is_empty() {
        return Err(owner_term_error("missing_scope", Some(term.id.role), Some(term.class)));
    }
    if matches!(term.id.occurrence, OccurrenceId::Composed(ref value) if value.is_empty()) {
        return Err(owner_term_error("missing_occurrence", Some(term.id.role), Some(term.class)));
    }
    Ok(())
}

fn merge_owner_terms(
    existing: &[AllocationOwnerTerm],
    additions: impl IntoIterator<Item = AllocationOwnerTerm>,
) -> Result<Vec<AllocationOwnerTerm>> {
    let additions = additions.into_iter().collect::<Vec<_>>();
    let mut merged = Vec::with_capacity(existing.len() + additions.len());
    let mut seen = HashSet::with_capacity(existing.len() + additions.len());
    for term in existing.iter().chain(additions.iter()) {
        validate_owner_term(term)?;
        if !seen.insert(term.id.clone()) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.requirement",
                "attempt owner-term inventory contains a duplicate exact term",
            )
            .with_fact("reason", "attempt_owner_term_duplicate")
            .with_fact("owner", term.id.owner.as_str())
            .with_fact("owner_term_role", term.id.role.as_str())
            .with_fact("occurrence", term.id.occurrence.as_str())
            .with_fact("execution_scope_fingerprint", term.id.execution_scope_fingerprint.as_str()));
        }
        merged.push(term.clone());
    }
    Ok(merged)
}

fn graph_owner_terms_from_evidence(evidence: &GraphMemoryAdmissionEvidence) -> Result<Vec<AllocationOwnerTerm>> {
    const REQUIRED_GRAPH_OWNER_TERM_COUNT: usize = 4;
    const MAX_GRAPH_OWNER_TERM_COUNT: usize = REQUIRED_GRAPH_OWNER_TERM_COUNT + 2;
    evidence.graph_live_reservation_floor_top_up_bytes()?;
    let facts = evidence.owner_term_facts();
    if !(REQUIRED_GRAPH_OWNER_TERM_COUNT..=MAX_GRAPH_OWNER_TERM_COUNT).contains(&facts.len()) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.requirement",
            "graph memory evidence does not contain the complete owner-term inventory",
        )
        .with_fact("reason", "graph_memory_owner_terms_incomplete")
        .with_fact("owner_term_count", facts.len())
        .with_fact("required_owner_term_count", REQUIRED_GRAPH_OWNER_TERM_COUNT)
        .with_fact("max_owner_term_count", MAX_GRAPH_OWNER_TERM_COUNT));
    }

    let mut roles = HashSet::with_capacity(facts.len());
    let mut terms = Vec::with_capacity(facts.len());
    for fact in facts {
        if fact.execution_scope_fingerprint() != evidence.execution_scope_fingerprint() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.requirement",
                "graph memory owner-term scope does not match its evidence scope",
            )
            .with_fact("reason", "graph_memory_owner_term_scope_mismatch")
            .with_fact("owner_term_role", fact.role().as_str()));
        }
        if !roles.insert(fact.role()) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.requirement",
                "graph memory evidence repeats an owner-term role",
            )
            .with_fact("reason", "graph_memory_owner_term_duplicate_role")
            .with_fact("owner_term_role", fact.role().as_str()));
        }
        terms.push(graph_owner_term_from_fact(fact)?);
    }

    let required_roles = [
        GraphMemoryOwnerTermRole::GraphPersistent,
        GraphMemoryOwnerTermRole::ConstructionTransient,
        GraphMemoryOwnerTermRole::ExecutionWorkspace,
        GraphMemoryOwnerTermRole::RetainedOutput,
    ];
    if required_roles.iter().any(|role| !roles.contains(role)) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.requirement",
            "graph memory evidence is missing an owner-term role",
        )
        .with_fact("reason", "graph_memory_owner_term_role_missing"));
    }
    Ok(terms)
}

fn local_parquet_source_owner_terms(plan: &QueryPlan) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = canonical_execution_scope_fingerprint(plan);
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut terms = Vec::new();
    let mut has_local_source = false;
    let mut has_non_local_source = false;
    let mut has_disallowed_non_local_source = false;
    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        let PlanNode::Source(source) = &node.kind else {
            continue;
        };
        match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                has_local_source = true;
                let Some(bytes) = estimated_local_parquet_source_bytes(source) else {
                    return Err(owner_term_error(
                        "local_source_bytes_unavailable",
                        Some(OwnerTermRole::SourceStage),
                        None,
                    ));
                };
                terms.push(AllocationOwnerTerm::new(
                    OwnerId::Source(node.id),
                    OwnerTermRole::SourceStage,
                    OccurrenceId::PlanLocal(node.id),
                    scope.clone(),
                    AllocationClass::ActivePhase,
                    bytes,
                    None,
                )?);
            },
            SourceNode::GpuResident(_) => has_non_local_source = true,
            SourceNode::LiteralBatch(source) if source.row_count() > 0 && !source.schema.fields().is_empty() => {
                has_non_local_source = true;
                has_disallowed_non_local_source = true;
            },
            SourceNode::LiteralBatch(_) => {},
        }
    }
    if has_local_source && has_non_local_source && has_disallowed_non_local_source {
        return Err(owner_term_error("mixed_source_kinds", Some(OwnerTermRole::SourceStage), None));
    }
    Ok(terms)
}

fn has_mixed_local_gpu_sources(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut has_local = false;
    let mut has_gpu = false;
    let mut has_other = false;
    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        let PlanNode::Source(source) = &node.kind else {
            continue;
        };
        match source.as_ref() {
            SourceNode::LocalParquet(_) => has_local = true,
            SourceNode::GpuResident(_) => has_gpu = true,
            SourceNode::LiteralBatch(_) => has_other = true,
        }
    }
    has_local && has_gpu && !has_other
}

fn generic_gpu_resident_owner_terms(
    plan: &QueryPlan,
    include_gpu_output_owner: bool,
) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = canonical_execution_scope_fingerprint(plan);
    if !matches!(plan.sink(), ResultSink::GpuResident { .. })
        || plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::GraphAlgorithm(_)))
    {
        return Ok(Vec::new());
    }
    if !include_gpu_output_owner {
        return Ok(Vec::new());
    }
    let ResultSink::GpuResident { input } = plan.sink() else {
        unreachable!();
    };
    let local_source_terms = local_parquet_source_owner_terms(plan)?;
    let reachable_local_sources = plan
        .nodes()
        .iter()
        .filter(|node| {
            !plan.unreachable_node_ids().contains(&node.id)
                && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)))
        })
        .collect::<Vec<_>>();
    if !local_source_terms.is_empty()
        && let [source_node] = reachable_local_sources.as_slice()
        && source_node.id == *input
    {
        let rows = query_plan_hard_row_upper_bound(plan).ok_or_else(|| {
            owner_term_error("gpu_resident_row_bound_unavailable", Some(OwnerTermRole::RetainedOutput), None)
        })?;
        let source_bytes = local_source_terms.first().map(AllocationOwnerTerm::bytes).ok_or_else(|| {
            owner_term_error("gpu_resident_source_owner_missing", Some(OwnerTermRole::SourceStage), None)
        })?;
        return Ok(vec![AllocationOwnerTerm::new(
            OwnerId::Producer(*input),
            OwnerTermRole::RetainedOutput,
            OccurrenceId::PlanLocal(*input),
            scope,
            AllocationClass::AttemptRetained,
            source_bytes,
            Some(rows),
        )?]);
    }
    let reachable_gpu_sources = plan.nodes().iter().filter(|node| {
        !plan.unreachable_node_ids().contains(&node.id)
            && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_)))
    }).collect::<Vec<_>>();
    if local_source_terms.is_empty()
        && let [source_node] = reachable_gpu_sources.as_slice()
        && source_node.id == *input
    {
        // A direct GPU-resident sink forwards the upstream handle; its frame
        // remains owned by the producer or input tracker that supplied it.
        return Ok(Vec::new());
    }
    if local_source_terms.is_empty() && reachable_gpu_sources.is_empty() {
        return Ok(Vec::new());
    }
    let rows = query_plan_hard_row_upper_bound(plan).ok_or_else(|| {
        owner_term_error("gpu_resident_row_bound_unavailable", Some(OwnerTermRole::RetainedOutput), None)
    })?;
    let output_evidence = PlanColumnEvidenceAnalysis::for_plan(plan, &HashMap::new()).root(plan);
    let output_bytes = bounded_consumer_frame_bytes_with_evidence(rows, plan.output_schema(), &output_evidence)
        .map_err(|error| {
            owner_term_error("gpu_resident_output_bytes_unavailable", Some(OwnerTermRole::RetainedOutput), None)
                .with_fact("sizing_reason", error.fact_value("schedule_reason").unwrap_or("unknown"))
        })?;
    let mut terms = reachable_gpu_sources
        .iter()
        .map(|node| {
            let PlanNode::Source(source) = &node.kind else {
                unreachable!();
            };
            let SourceNode::GpuResident(source) = source.as_ref() else {
                unreachable!();
            };
            let input_rows = source.row_count_upper_bound.ok_or_else(|| {
                owner_term_error("gpu_resident_input_row_bound_unavailable", Some(OwnerTermRole::SourceStage), None)
            })?;
            let input_bytes = bounded_consumer_frame_bytes(input_rows, &source.schema).map_err(|error| {
                owner_term_error("gpu_resident_input_bytes_unavailable", Some(OwnerTermRole::SourceStage), None)
                    .with_fact("sizing_reason", error.fact_value("schedule_reason").unwrap_or("unknown"))
            })?;
            AllocationOwnerTerm::new(
                OwnerId::Source(node.id),
                OwnerTermRole::SourceStage,
                OccurrenceId::PlanLocal(node.id),
                scope.clone(),
                AllocationClass::ActivePhase,
                input_bytes,
                None,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    terms.extend(projection_cast_workspace_owner_terms(plan, &scope)?);
    terms.push(AllocationOwnerTerm::new(
        OwnerId::Producer(*input),
        OwnerTermRole::RetainedOutput,
        OccurrenceId::PlanLocal(*input),
        scope,
        AllocationClass::AttemptRetained,
        output_bytes,
        Some(rows),
    )?);
    Ok(terms)
}

fn projection_cast_workspace_owner_terms(plan: &QueryPlan, scope: &str) -> Result<Vec<AllocationOwnerTerm>> {
    let unreachable = plan.unreachable_node_ids();
    let mut terms = Vec::new();
    for node in plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)) {
        let PlanNode::Projection(projection) = &node.kind else {
            continue;
        };
        let input_schema = plan.node(projection.input).ok_or_else(|| {
            owner_term_error(
                "projection_cast_input_missing",
                Some(OwnerTermRole::ConsumerProjectionCastWorkspace),
                Some(AllocationClass::ActivePhase),
            )
        })?;
        let workspace_bytes = projection
            .outputs
            .iter()
            .map(|output| {
                let actual_type =
                    crate::exec::validate::projection_output_type_for_admission(output, input_schema.output_schema())?;
                let expected_type = output.field.data_type();
                if actual_type == *expected_type {
                    return Ok(0_u64);
                }
                let Some(actual_type) = cudf_dtype_from_arrow_type(&actual_type) else {
                    return Ok(0);
                };
                let Some(expected_type) = cudf_dtype_from_arrow_type(expected_type) else {
                    return Ok(0);
                };
                let bytes = cudf::ops::unary::decimal_rescale_temporary_workspace_reservation_bytes(
                    &actual_type,
                    &expected_type,
                )
                .unwrap_or(0);
                u64::try_from(bytes).map_err(|_| {
                    owner_term_error(
                        "projection_cast_workspace_bytes_exceed_u64",
                        Some(OwnerTermRole::ConsumerProjectionCastWorkspace),
                        Some(AllocationClass::ActivePhase),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        if workspace_bytes == 0 {
            continue;
        }
        terms.push(AllocationOwnerTerm::new(
            OwnerId::Consumer(node.id),
            OwnerTermRole::ConsumerProjectionCastWorkspace,
            OccurrenceId::PlanLocal(node.id),
            scope.to_owned(),
            AllocationClass::ActivePhase,
            workspace_bytes,
            None,
        )?);
    }
    Ok(terms)
}

fn generic_aggregate_owner_terms(plan: &QueryPlan) -> Result<Vec<AllocationOwnerTerm>> {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let reachable = plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).collect::<Vec<_>>();
    if reachable.iter().any(|node| {
        !matches!(
            node.kind,
            PlanNode::Source(_)
                | PlanNode::Filter(_)
                | PlanNode::Projection(_)
                | PlanNode::Aggregate(_)
                | PlanNode::Sort(_)
                | PlanNode::Limit(_)
        )
    }) {
        return Ok(Vec::new());
    }
    let local_sources = reachable
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let aggregates = reachable
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some((node.id, aggregate, node.output_schema())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let ([source], [(aggregate_id, aggregate, output_schema)]) = (local_sources.as_slice(), aggregates.as_slice())
    else {
        return Ok(Vec::new());
    };
    if reachable.iter().any(|node| {
        matches!(&node.kind, PlanNode::Source(source) if !matches!(source.as_ref(), SourceNode::LocalParquet(_)))
    }) || !aggregate_owner_state_is_supported(aggregate)
    {
        return Ok(Vec::new());
    }
    let selected_row_group_rows = source.exact_selected_row_group_rows();
    let input_rows = query_plan_node_hard_row_upper_bound(plan, aggregate.input);
    let (Some(selected_row_group_rows), Some(input_rows)) = (selected_row_group_rows, input_rows) else {
        return Ok(Vec::new());
    };
    let Some(source_rows) = selected_row_group_rows.iter().try_fold(0_u64, |total, rows| total.checked_add(*rows))
    else {
        return Ok(Vec::new());
    };
    if selected_row_group_rows.is_empty() || input_rows > source_rows {
        return Ok(Vec::new());
    }
    if output_schema.fields().iter().any(|field| fixed_width_consumer_output_column(field.data_type()).is_none()) {
        return Ok(Vec::new());
    }

    let output_rows = if aggregate.spec.group_exprs.is_empty() { 1 } else { input_rows };
    let partial_shapes = selected_row_group_rows.into_iter().map(|row_group_rows| {
        if aggregate.spec.group_exprs.is_empty() {
            crate::exec::PartialAggregateShape::Global
        } else {
            crate::exec::PartialAggregateShape::Grouped { input_rows: row_group_rows }
        }
    });
    aggregate_owner_terms_for_shapes(plan, *aggregate_id, aggregate, output_schema, partial_shapes, output_rows, None)
}

fn aggregate_owner_state_is_supported(aggregate: &AggregateNode) -> bool {
    aggregate.spec.grouping_sets.is_none()
        && aggregate.spec.aggr_exprs.iter().all(|expr| {
            !expr.distinct
                && expr.filter.is_none()
                && matches!(
                    expr.function,
                    AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max | AggregateFunction::Count
                )
        })
}

fn aggregate_owner_terms_for_shapes(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    aggregate: &AggregateNode,
    output_schema: &SchemaRef,
    partial_shapes: impl IntoIterator<Item = crate::exec::PartialAggregateShape>,
    output_rows: u64,
    column_evidence: Option<&PlanColumnEvidence>,
) -> Result<Vec<AllocationOwnerTerm>> {
    let mut partial_columns = output_schema
        .fields()
        .iter()
        .map(|field| consumer_output_column_with_optional_evidence(field, column_evidence))
        .collect::<Result<Vec<_>>>()?;
    partial_columns.extend(
        aggregate
            .spec
            .aggr_exprs
            .iter()
            .map(|_| fixed_width_consumer_output_column(&DataType::UInt64).expect("UInt64 has a fixed-width bound")),
    );
    let partial_bytes = partial_shapes.into_iter().try_fold(0_u64, |total, partial_shape| {
        let partial = crate::exec::partial_aggregate_upper_bound_bytes(partial_shape, &partial_columns)
            .map_err(|_| owner_term_error("aggregate_partial_state_unavailable", None, None))?;
        if partial == u64::MAX {
            return Err(owner_term_error("aggregate_partial_state_saturated", None, None));
        }
        total.checked_add(partial).ok_or_else(|| owner_term_error("aggregate_partial_state_overflow", None, None))
    })?;
    let output_copies =
        if matches!(plan.sink(), ResultSink::GpuResident { input } if *input == aggregate_id) { 1 } else { 2 };
    let output_bytes = match column_evidence {
        Some(evidence) => bounded_consumer_frame_bytes_with_evidence(output_rows, output_schema, evidence),
        None => bounded_consumer_frame_bytes(output_rows, output_schema),
    }?
    .checked_mul(output_copies)
    .ok_or_else(|| owner_term_error("aggregate_output_overflow", None, None))?;
    let scope = canonical_execution_scope_fingerprint(plan);
    Ok(vec![
        AllocationOwnerTerm::new(
            OwnerId::Aggregate(aggregate_id),
            OwnerTermRole::AggregatePartialState,
            OccurrenceId::PlanLocal(aggregate_id),
            scope.clone(),
            AllocationClass::ActivePhase,
            partial_bytes,
            None,
        )?,
        AllocationOwnerTerm::new(
            OwnerId::Aggregate(aggregate_id),
            OwnerTermRole::AggregateOutput,
            OccurrenceId::PlanLocal(aggregate_id),
            scope,
            AllocationClass::ActivePhase,
            output_bytes,
            None,
        )?,
    ])
}

fn literal_batch_owner_term_rejection_reason(
    plan: &QueryPlan,
    policy: OwnerTermConstructionPolicy,
) -> Option<&'static str> {
    if matches!(policy, OwnerTermConstructionPolicy::AllowLiteralBatch | OwnerTermConstructionPolicy::PriceLiteralBatch)
    {
        return None;
    }
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).find_map(|node| {
        let PlanNode::Source(source) = &node.kind else {
            return None;
        };
        let SourceNode::LiteralBatch(source) = source.as_ref() else {
            return None;
        };
        (source.row_count() > 0 && !source.schema.fields().is_empty())
            .then_some("attempt_owner_term_literal_source_bytes_unavailable")
    })
}

fn literal_batch_source_owner_terms(plan: &QueryPlan) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = canonical_execution_scope_fingerprint(plan);
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    plan.nodes()
        .iter()
        .filter(|node| !unreachable.contains(&node.id))
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LiteralBatch(source) if source.row_count() > 0 && !source.schema.fields().is_empty() => {
                    Some((node.id, source))
                },
                _ => None,
            },
            _ => None,
        })
        .map(|(node_id, source)| {
            let rows = u64::try_from(source.row_count()).map_err(|_| {
                owner_term_error("literal_source_rows_exceed_u64", Some(OwnerTermRole::SourceStage), None)
            })?;
            let columns = literal_batch_output_columns(source)?;
            let bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns).map_err(|_| {
                owner_term_error("literal_source_bytes_unavailable", Some(OwnerTermRole::SourceStage), None)
            })?;
            if bytes == u64::MAX {
                return Err(owner_term_error("literal_source_bytes_saturated", Some(OwnerTermRole::SourceStage), None));
            }
            AllocationOwnerTerm::new(
                OwnerId::Source(node_id),
                OwnerTermRole::SourceStage,
                OccurrenceId::PlanLocal(node_id),
                scope.clone(),
                AllocationClass::ActivePhase,
                bytes,
                None,
            )
        })
        .collect()
}

fn literal_batch_output_columns(
    source: &crate::source::LiteralBatchSource,
) -> Result<Vec<crate::exec::JoinOutputColumnUpperBoundInput>> {
    source
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if let Some(column) = fixed_width_consumer_output_column(field.data_type()) {
                return Ok(column);
            }
            let stats = source.batches.iter().try_fold((0_u64, 0_u64), |acc, batch| {
                let array = batch.column(index);
                let mut lengths =
                    (0..array.len()).filter(|row| !array.is_null(*row)).map(|row| match field.data_type() {
                        DataType::Utf8 => {
                            array.as_any().downcast_ref::<StringArray>().map(|array| array.value(row).len())
                        },
                        DataType::LargeUtf8 => {
                            array.as_any().downcast_ref::<LargeStringArray>().map(|array| array.value(row).len())
                        },
                        DataType::Utf8View => {
                            array.as_any().downcast_ref::<StringViewArray>().map(|array| array.value(row).len())
                        },
                        _ => None,
                    });
                lengths.try_fold(acc, |(max_len, total_bytes), length| {
                    let length = u64::try_from(length?).ok()?;
                    Some((max_len.max(length), total_bytes.checked_add(length)?))
                })
            });
            let Some((max_len, total_bytes)) = stats else {
                return Err(owner_term_error(
                    "literal_source_bytes_unavailable",
                    Some(OwnerTermRole::SourceStage),
                    None,
                ));
            };
            Ok(crate::exec::JoinOutputColumnUpperBoundInput {
                data_type: field.data_type().clone(),
                input_max_string_length_bytes: Some(max_len),
                input_total_string_bytes: Some(total_bytes),
            })
        })
        .collect()
}

/// Resolved, planner-owned capability evidence for one bounded execution attempt.
///
/// Callers can forward this value into admission, but cannot author a floor,
/// hint, or device-capability set.  The installed device profile owns baseline
/// grant sizing; planner evidence may only record a proven lower-layer raise
/// and a descriptive model result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptCapabilityEvidence {
    /// Deployment-wide floor supplied by the installed device profile.  This
    /// remains crate-visible only for synthetic controller tests.
    pub(crate) configured_floor_bytes: u64,
    /// Additional lower-layer workspace contracts proven before admission.
    pub(crate) capability_floor_raise_bytes: u64,
    /// Hard lower bound for the service-resolved physical phase envelope.
    pub(crate) required_phase_floor_bytes: u64,
    /// Descriptive local-Parquet stage, distinct from a whole-attempt target.
    pub(crate) source_stage_bytes: Option<u64>,
    /// Descriptive target only; it never proves fit or admission.
    pub(crate) soft_working_set_hint_bytes: Option<u64>,
    /// Evidence quality for [`Self::soft_working_set_hint_bytes`].
    pub(crate) estimate_quality: EstimateQuality,
    /// Capability contracts required by the admitted fragments.
    pub(crate) required_capabilities: Vec<ReservationRequirement>,
    /// Exact engine-issued allocation owners used by bounded derivations.
    pub(crate) owner_terms: Vec<AllocationOwnerTerm>,
    /// Provenance selected when this evidence is resolved into an immutable
    /// reservation envelope.  It is intentionally absent before selection.
    pub(crate) bound_derivation: Option<AttemptBoundDerivation>,
    /// A mixed LocalParquet/GPU consumer is incomplete until composition binds
    /// every GPU-resident occurrence to a retained producer term.
    pub(crate) requires_composed_producer_binding: bool,
    /// Exact graph plans for which the engine may replace its generic
    /// lower-level graph-workspace blocker with the cuGraph contract.
    ///
    /// This remains part of the engine-owned evidence bundle from admission
    /// through execution. A capability label alone is deliberately not enough
    /// to make that substitution: it would let a combined proof authorize an
    /// unrelated graph plan.
    pub(crate) attested_cugraph_graph_execution_scopes: Vec<AttestedCugraphGraphExecutionScope>,
    /// Private, exact plan scopes resolved only after the service fixes
    /// scheduling and chunk policy for this admission.
    pub(crate) reservation_shapes: Vec<AttemptReservationShape>,
}

/// Collision-free structural identity for one graph plan whose frontend owns
/// construction and workspace lifetimes.
///
/// The identity retains the full canonical execution form rather than a
/// fixed-width hash. These scopes are short-lived admission evidence, so a
/// graph contract cannot widen to a separately materialized plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttestedCugraphGraphExecutionScope {
    execution_scope_fingerprint: String,
}

impl AttestedCugraphGraphExecutionScope {
    fn for_plan(plan: &QueryPlan) -> Self {
        Self { execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan) }
    }

    fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    fn matches_reservation_shape(&self, shape: &AttemptReservationShape) -> bool {
        shape.matches_execution_scope_fingerprint(&self.execution_scope_fingerprint)
    }

    fn matches_graph_memory_evidence(&self, evidence: &GraphMemoryAdmissionEvidence) -> bool {
        evidence.matches_execution_scope_fingerprint(&self.execution_scope_fingerprint)
    }
}

/// Opaque capability proof issued from a native plan or a canonical graph
/// execution contract.
///
/// Public callers can forward a proof into admission, but cannot author a
/// capability set, a floor, or a working-set target.  In particular, there is
/// deliberately no empty-proof constructor: every proof comes from an exact
/// native plan and carries its private execution scope, even when that plan
/// has no resource-capability requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPlanCapabilityProof {
    requirement: AttemptCapabilityEvidence,
}

/// Adapter-observed identity and opaque admission proof for one graph producer
/// in a composed attempt.
///
/// The adapter may report an unavailable identity or proof, but it cannot
/// provide graph-memory bytes or construct a reservation shape. The engine
/// issues a composed schedule only when every observation is complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedGraphProducerCandidate {
    runtime_occurrence_id: Option<String>,
    graph_result_input_name: Option<String>,
    capability_proof: Option<QueryPlanCapabilityProof>,
}

impl ComposedGraphProducerCandidate {
    /// Creates one adapter-observed graph-producer candidate in execution
    /// order.
    pub fn new(
        runtime_occurrence_id: Option<String>,
        graph_result_input_name: Option<String>,
        capability_proof: Option<QueryPlanCapabilityProof>,
    ) -> Self {
        Self { runtime_occurrence_id, graph_result_input_name, capability_proof }
    }
}

impl QueryPlanCapabilityProof {
    /// Derives proof from one lowered native query plan.
    ///
    /// The proof always retains the plan's exact private execution scope. A
    /// capability-free native fragment therefore still cannot run under a
    /// grant issued for a different fragment, while adding no resource
    /// capability or grant-floor allowance of its own.
    pub fn for_query_plan(plan: &QueryPlan) -> Option<Self> {
        let requirement =
            AttemptCapabilityEvidence::for_query_plan_with_policy(plan, OwnerTermConstructionPolicy::Generic);
        Some(Self { requirement })
    }

    /// Validates one lowered native query plan before issuing its capability proof.
    pub fn for_query_plan_preflighted(plan: &QueryPlan) -> Result<Self> {
        crate::exec::preflight_query_plan(plan)?;
        Self::for_query_plan(plan).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.query_plan_capability_proof",
                "native query plan did not issue an exact capability proof",
            )
        })
    }

    /// Derives the consumer proof used only while composing an exact GPU
    /// island. This policy permits a reachable LocalParquet source alongside
    /// GPU-resident branches so the shared traversal can consume the
    /// canonical source-stage terms. Literal and other non-GPU source kinds
    /// remain rejected.
    pub fn for_composed_consumer_plan(plan: &QueryPlan) -> Option<Self> {
        let mut requirement = AttemptCapabilityEvidence::for_query_plan_without_gpu_output_owner(
            plan,
            OwnerTermConstructionPolicy::AllowMixedSourceComposition,
        );
        requirement.requires_composed_producer_binding = has_mixed_local_gpu_sources(plan);
        Some(Self { requirement })
    }

    /// Derives the proof for the exact native fragment that materializes the
    /// cuGraph edge input. The graph-memory model owns that final resident
    /// frame, so its scope is retained for execution equality but contributes
    /// no second whole-attempt phase.
    pub fn for_cugraph_graph_input_plan(plan: &QueryPlan) -> Option<Self> {
        let mut requirement = AttemptCapabilityEvidence::for_query_plan_without_gpu_output_owner(
            plan,
            OwnerTermConstructionPolicy::AllowLiteralBatch,
        );
        requirement.mark_cugraph_graph_input_scope();
        Some(Self { requirement })
    }

    fn for_cugraph_materialized_graph_input_producer_plan(plan: &QueryPlan) -> Result<Self> {
        let mut requirement =
            AttemptCapabilityEvidence::for_query_plan_with_policy(plan, OwnerTermConstructionPolicy::PriceLiteralBatch);
        if let ResultSink::GpuResident { input } = plan.sink()
            && plan
                .nodes()
                .iter()
                .any(|node| matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LiteralBatch(_))))
            && !requirement.owner_terms.iter().any(AllocationOwnerTerm::is_retained_output)
        {
            let rows = query_plan_hard_row_upper_bound(plan)
                .ok_or_else(|| composed_schedule_error("producer_retained_output_rows_missing", None))?;
            let output_evidence = PlanColumnEvidenceAnalysis::for_plan(plan, &HashMap::new()).root(plan);
            let output_bytes = bounded_consumer_frame_bytes_with_evidence(rows, plan.output_schema(), &output_evidence)?;
            let retained = AllocationOwnerTerm::new(
                OwnerId::Producer(*input),
                OwnerTermRole::RetainedOutput,
                OccurrenceId::PlanLocal(*input),
                canonical_execution_scope_fingerprint(plan),
                AllocationClass::AttemptRetained,
                output_bytes,
                Some(rows),
            )?;
            requirement.owner_terms.push(retained);
        }
        requirement.mark_cugraph_graph_input_scope();
        Ok(Self { requirement })
    }

    /// Issues a proof for one exact native predicate fragment staged as a
    /// cuGraph side input. The graph model prices cuGraph-owned copies and
    /// masks; this separate phase prices the native fragment's own output
    /// frame, which remains live while graph execution consumes it.
    ///
    /// The frame contract is deliberately narrow: the exact IR must carry a
    /// hard root row bound and every output field must use a supported fixed
    /// width type. Unknown cardinality, variable-width output, and arithmetic
    /// saturation fail closed before an admission ticket is issued.
    pub fn try_for_cugraph_side_input_plan(plan: &QueryPlan) -> Result<Self> {
        let frame_bytes = cugraph_side_input_frame_upper_bound_bytes(plan)?;
        let mut requirement = AttemptCapabilityEvidence::for_query_plan_without_gpu_output_owner(
            plan,
            OwnerTermConstructionPolicy::AllowLiteralBatch,
        );
        let [shape] = requirement.reservation_shapes.as_mut_slice() else {
            return Err(
                composed_schedule_error("side_input_scope_ambiguous", None).with_fact("phase", "side_input_frame")
            );
        };
        shape.set_cugraph_side_input_phase_bytes(frame_bytes)?;
        let owner_term = AllocationOwnerTerm::new(
            OwnerId::SideInput(plan.root_id()),
            OwnerTermRole::SideInputFrame,
            OccurrenceId::PlanLocal(plan.root_id()),
            canonical_execution_scope_fingerprint(plan),
            AllocationClass::ActivePhase,
            frame_bytes,
            None,
        )?;
        requirement.owner_terms = merge_owner_terms(&requirement.owner_terms, [owner_term])?;
        Ok(Self { requirement })
    }

    /// Derives proof for an exact host handoff of one frontend-attested graph
    /// result. The plan must do no native work: it can only expose the named
    /// GPU-resident graph result to its host sink. Its device frame is still
    /// owned by the graph-execution phase, while its structural scope remains
    /// available for execution-time equality validation.
    pub fn for_attested_graph_result_passthrough_plan(plan: &QueryPlan, graph_result_input_name: &str) -> Option<Self> {
        if !is_attested_graph_result_passthrough_plan(plan, graph_result_input_name) {
            return None;
        }
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        requirement.remove_generic_gpu_resident_retained_output_terms();
        requirement.mark_attested_graph_result_passthrough_scope();
        Some(Self { requirement })
    }

    /// Derives proof for the bounded native sort phase that follows one
    /// frontend-attested graph result. The graph result remains the only
    /// external GPU frame; the engine executes the sort through the same
    /// immutable bounded-attempt resource, so it cannot exceed that attempt's
    /// grant while the graph and sort phases remain sequential.
    pub fn for_attested_graph_result_bounded_consumer_plan(
        plan: &QueryPlan,
        graph_result_input_name: &str,
    ) -> Option<Self> {
        if !is_attested_graph_result_bounded_consumer_plan(plan, graph_result_input_name) {
            return None;
        }
        Some(Self::bounded_consumer_proof_or_sticky_rejection(
            plan,
            Self::try_for_attested_graph_result_bounded_consumer_plan(plan, graph_result_input_name),
        ))
    }

    /// Derives the bounded graph-result sort proof using the retained output
    /// row bound owned by one opaque graph producer proof.
    ///
    /// A GPU-resident source has no independent plan-time cardinality. The
    /// producer's graph-memory evidence is the only owner of that bound, so
    /// this seam keeps the consumer sort priced without accepting adapter
    /// estimates or mutating the lowered source node.
    pub fn for_attested_graph_result_bounded_consumer_plan_with_producer(
        plan: &QueryPlan,
        graph_result_input_name: &str,
        producer: &Self,
    ) -> Option<Self> {
        if !is_attested_graph_result_bounded_consumer_plan(plan, graph_result_input_name) {
            return None;
        }
        let result = producer.requirement.retained_output_rows_for_graph_result().and_then(|row_count| {
            Self::try_for_attested_graph_result_bounded_consumer_plan_with_row_bound(
                plan,
                graph_result_input_name,
                Some(row_count),
            )
        });
        Some(Self::bounded_consumer_proof_or_sticky_rejection(plan, result))
    }

    /// Issues the typed proof for a supported unstable one-key graph-result
    /// sort.  The legacy option-returning constructor above remains for
    /// callers that intentionally treat unsupported alternatives as a normal
    /// shape miss; admission-owned callers should use this checked seam so
    /// stable, nullable, unsupported, or unbounded sorts retain their reason.
    pub fn try_for_attested_graph_result_bounded_consumer_plan(
        plan: &QueryPlan,
        graph_result_input_name: &str,
    ) -> Result<Self> {
        Self::try_for_attested_graph_result_bounded_consumer_plan_with_row_bound(plan, graph_result_input_name, None)
    }

    fn try_for_attested_graph_result_bounded_consumer_plan_with_row_bound(
        plan: &QueryPlan,
        graph_result_input_name: &str,
        producer_row_count_upper_bound: Option<u64>,
    ) -> Result<Self> {
        let descriptor =
            sort_metadata_for_attested_graph_result(plan, graph_result_input_name, producer_row_count_upper_bound)?;
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        requirement.remove_generic_gpu_resident_retained_output_terms();
        requirement.mark_attested_graph_result_bounded_consumer_scope();
        if let [shape] = requirement.reservation_shapes.as_mut_slice() {
            shape.set_sort_metadata(descriptor)?;
        }
        Ok(Self { requirement })
    }

    fn bounded_consumer_proof_or_sticky_rejection(plan: &QueryPlan, result: Result<Self>) -> Self {
        match result {
            Ok(proof) => proof,
            Err(error) => {
                let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
                requirement.mark_attested_graph_result_bounded_consumer_scope();
                if let [shape] = requirement.reservation_shapes.as_mut_slice() {
                    shape.set_sort_rejection_reason(
                        error.fact_value("reason").unwrap_or("sort_owner_metadata_invalid").to_owned(),
                    );
                }
                Self { requirement }
            },
        }
    }

    /// Derives a plan-scoped proof for the exact internal plan issued by the
    /// cuGraph frontend.
    ///
    /// The generic graph node analysis deliberately keeps its lower-level
    /// bounded-execution blocker: the engine alone cannot establish that an
    /// embedding graph runtime owns all graph construction and workspace
    /// lifetimes.  cuGraph establishes that contract at its own boundary, so
    /// its engine-issued proof replaces only that blocker with the canonical
    /// cuGraph execution capability. The replacement is bound to a private,
    /// collision-free structural scope for this exact plan; combining proofs
    /// preserves every scope, and execution rechecks it against the plan
    /// about to run. The adapter-only host/deferred-GPU sink boundary is
    /// canonicalized; all execution-relevant graph and operator facts remain
    /// exact.
    pub fn for_cugraph_graph_execution_plan(plan: &QueryPlan) -> Option<Self> {
        Self::for_cugraph_graph_execution_plan_with_resident_inputs(plan, std::iter::empty())
            .expect("an empty resident-input binding cannot be invalid")
    }

    /// Derives the cuGraph proof while binding GPU-resident inputs to the exact
    /// native plans that produce them.
    pub fn for_cugraph_graph_execution_plan_with_resident_inputs<'a>(
        plan: &QueryPlan,
        resident_inputs: impl IntoIterator<Item = (&'a str, &'a QueryPlan)>,
    ) -> Result<Option<Self>> {
        let mut resident_outputs = HashMap::new();
        let mut resident_input_scopes = Vec::new();
        for (input_name, input_plan) in resident_inputs {
            if resident_outputs.contains_key(input_name) {
                return Err(composed_schedule_error("duplicate_resident_input_evidence", Some(input_name)));
            }
            let mut matching_sources = plan.nodes().iter().filter_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) if source.input_name == input_name => Some(source),
                    _ => None,
                },
                _ => None,
            });
            let Some(first_source) = matching_sources.next() else {
                return Err(composed_schedule_error("resident_input_evidence_source_missing", Some(input_name)));
            };
            if first_source.schema != *input_plan.output_schema()
                || matching_sources.any(|source| source.schema != first_source.schema)
            {
                return Err(composed_schedule_error("resident_input_evidence_schema_mismatch", Some(input_name)));
            }
            resident_outputs.insert(
                input_name.to_owned(),
                PlanColumnEvidenceAnalysis::for_plan(input_plan, &HashMap::new()).root(input_plan),
            );
            resident_input_scopes.push((input_name.to_owned(), canonical_execution_scope_fingerprint(input_plan)));
        }
        let mut requirement =
            AttemptCapabilityEvidence::for_query_plan_with_policy(plan, OwnerTermConstructionPolicy::PriceLiteralBatch);
        if let [shape] = requirement.reservation_shapes.as_mut_slice() {
            shape.set_output_column_evidence(PlanColumnEvidenceAnalysis::for_plan(plan, &resident_outputs).root(plan));
            shape.set_output_column_evidence_input_scopes(resident_input_scopes);
        }
        requirement.remove_generic_gpu_resident_retained_output_terms();
        let graph_requirement = generic_graph_execution_blocker();
        if !requirement.required_capabilities.contains(&graph_requirement) {
            return Ok((!requirement.required_capabilities.is_empty()).then_some(Self { requirement }));
        }
        requirement.required_capabilities.retain(|candidate| candidate != &graph_requirement);
        requirement.required_capabilities.push(cugraph_graph_execution_capability());
        let local_parquet_source_count = plan
            .nodes()
            .iter()
            .filter(|node| matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_))))
            .count();
        if local_parquet_source_count == 1 {
            requirement.mark_cugraph_graph_input_scope();
        }
        requirement.attested_cugraph_graph_execution_scopes.push(AttestedCugraphGraphExecutionScope::for_plan(plan));
        Ok(Some(Self { requirement }))
    }

    /// Combines proofs already issued for native fragments in one invocation.
    ///
    /// Requiring a first proof prevents public callers from using aggregation
    /// as an empty-proof constructor.
    pub fn combine(first: Self, remaining: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut requirements = vec![first.requirement];
        requirements.extend(remaining.into_iter().map(|proof| proof.requirement));
        AttemptCapabilityEvidence::aggregate(requirements).map(|requirement| Self { requirement })
    }

    /// Builds the ordered source-to-graph schedule when a resident edge frame
    /// passes through materializing native operators before graph execution.
    pub fn try_combine_cugraph_materializing_input_attempt<'a>(
        graph_plan: &QueryPlan,
        graph: Self,
        input_name: &str,
        producer_plan: &QueryPlan,
        additional_producer_plans: impl IntoIterator<Item = (String, &'a QueryPlan)>,
    ) -> Result<Option<Self>> {
        let graph_nodes = graph_plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::GraphAlgorithm(graph) => Some(graph),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [graph_node] = graph_nodes.as_slice() else {
            return Err(composed_schedule_error("graph_scope_missing_or_ambiguous", Some(input_name)));
        };
        let source_nodes = graph_plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) if source.input_name == input_name => Some(node.id),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        let [source_node] = source_nodes.as_slice() else {
            return Err(composed_schedule_error("resident_input_evidence_source_missing", Some(input_name)));
        };
        let direct_input = match &graph_plan
            .node(graph_node.input)
            .ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", Some(input_name)))?
            .kind
        {
            PlanNode::Source(_) => Some(graph_node.input),
            PlanNode::EdgeNormalize(normalize)
                if matches!(graph_plan.node(normalize.input).map(|node| &node.kind), Some(PlanNode::Source(_))) =>
            {
                Some(normalize.input)
            },
            _ => None,
        };
        let Some(direct_input) = direct_input else {
            let mut producers =
                vec![(input_name.to_owned(), Self::for_cugraph_materialized_graph_input_producer_plan(producer_plan)?)];
            for (name, plan) in additional_producer_plans {
                producers.push((name, Self::for_cugraph_materialized_graph_input_producer_plan(plan)?));
            }
            return Self::combine_composed_graph_attempt(graph_plan, graph, producers).map(Some);
        };
        if direct_input != *source_node {
            return Err(composed_schedule_error("graph_input_source_identity_mismatch", Some(input_name)));
        }
        Ok(None)
    }

    fn normalize_composed_candidate(candidate: ComposedGraphProducerCandidate) -> Result<(String, Self)> {
        let occurrence_id = candidate
            .runtime_occurrence_id
            .ok_or_else(|| composed_schedule_error("runtime_producer_occurrence_missing", None))?;
        if let Some(graph_result_input_name) = candidate.graph_result_input_name.as_deref()
            && occurrence_id != graph_result_input_name
        {
            return Err(composed_schedule_error("producer_occurrence_identity_mismatch", Some(&occurrence_id))
                .with_fact("graph_result_input_name", graph_result_input_name));
        }
        let proof = candidate.capability_proof.ok_or_else(|| {
            composed_schedule_error("producer_admission_proof_missing_or_ambiguous", Some(&occurrence_id))
        })?;
        Ok((occurrence_id, proof))
    }

    fn normalize_composed_producer(
        occurrence_id: String,
        mut producer: Self,
        occurrence_ids: &mut HashSet<String>,
    ) -> Result<NormalizedComposedProducer> {
        if occurrence_id.is_empty() || !occurrence_ids.insert(occurrence_id.clone()) {
            return Err(composed_schedule_error("duplicate_producer_occurrence_id", Some(&occurrence_id)));
        }
        let matching_shapes =
            if let [attested_scope] = producer.requirement.attested_cugraph_graph_execution_scopes.as_slice() {
                producer
                    .requirement
                    .reservation_shapes
                    .iter()
                    .filter(|shape| attested_scope.matches_reservation_shape(shape))
                    .collect::<Vec<_>>()
            } else {
                producer.requirement.reservation_shapes.iter().collect::<Vec<_>>()
            };
        let [producer_shape] = matching_shapes.as_slice() else {
            return Err(composed_schedule_error("producer_scope_coverage_mismatch", Some(&occurrence_id)));
        };
        let execution_scope_fingerprint = producer_shape.execution_scope_fingerprint().to_owned();
        let nested_peak_terms = producer.requirement.take_nested_schedule_peak_terms()?;
        let rebound_ids = producer.requirement.rebind_owner_terms_to_composed_occurrence(&occurrence_id)?;
        let remap = |term_id: &OwnerTermId| {
            rebound_ids
                .get(term_id)
                .cloned()
                .ok_or_else(|| composed_schedule_error("nested_owner_term_reference_missing", Some(&occurrence_id)))
        };
        let nested_peak_terms = nested_peak_terms
            .map(|(groups, dominated, retained)| {
                let groups = groups
                    .iter()
                    .map(|group| group.iter().map(&remap).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?;
                let dominated = dominated.iter().map(&remap).collect::<Result<Vec<_>>>()?;
                let retained = retained.iter().map(&remap).collect::<Result<HashSet<_>>>()?;
                Ok((groups, dominated, retained))
            })
            .transpose()?;
        let nested_retained_output_term_ids =
            nested_peak_terms.as_ref().map(|(_, _, retained)| retained.clone()).unwrap_or_default();
        let retained_output_term = producer
            .requirement
            .retained_output_term_for_composition(&nested_retained_output_term_ids)
            .map_err(|error| error.with_fact("occurrence_id", occurrence_id.as_str()))?;
        let retained_output_term_id = retained_output_term.id().clone();
        let row_count_upper_bound = retained_output_term
            .retained_row_count_upper_bound()
            .ok_or_else(|| composed_schedule_error("producer_retained_output_rows_missing", Some(&occurrence_id)))?;
        let output_column_evidence =
            producer.requirement.output_column_evidence_for_scope(&execution_scope_fingerprint);
        let (phase_term_groups, alternative_phase_term_ids, concurrent_side_input_term_ids, dominated_term_ids) =
            match nested_peak_terms {
                Some((groups, dominated, _)) => {
                    let mut counts = HashMap::<OwnerTermId, usize>::new();
                    for group in &groups {
                        let mut seen = HashSet::with_capacity(group.len());
                        for term_id in group {
                            if !seen.insert(term_id) {
                                return Err(composed_schedule_error(
                                    "duplicate_owner_term_reference",
                                    Some(&occurrence_id),
                                ));
                            }
                            *counts.entry(term_id.clone()).or_default() += 1;
                        }
                    }
                    let alternatives =
                        counts.into_iter().filter_map(|(term_id, count)| (count > 1).then_some(term_id)).collect();
                    (groups, alternatives, Vec::new(), dominated)
                },
                None => {
                    let (groups, alternatives) = producer
                        .requirement
                        .isolated_phase_term_groups_with_alternatives(Some(&execution_scope_fingerprint));
                    (groups, alternatives, producer.requirement.concurrent_side_input_term_ids(), Vec::new())
                },
            };
        let nested_schedule = !nested_retained_output_term_ids.is_empty();
        Ok(NormalizedComposedProducer {
            occurrence_id,
            execution_scope_fingerprint,
            retained_output_term_id,
            row_count_upper_bound,
            output_column_evidence,
            phase_term_groups,
            alternative_phase_term_ids,
            concurrent_side_input_term_ids,
            dominated_term_ids,
            nested_schedule,
            proof: producer,
        })
    }

    /// Combines one consumer proof with graph-producer proofs in the exact
    /// order used by the composed executor.
    ///
    /// This seam carries only opaque, engine-issued proofs and occurrence
    /// identifiers. It never accepts adapter-authored byte arithmetic: graph
    /// output evidence stays attached to each producer proof and the native
    /// consumer's allocations are derived from its exact engine IR here.
    pub fn combine_composed_graph_attempt(
        consumer_plan: &QueryPlan,
        consumer: Self,
        ordered_producers: impl IntoIterator<Item = (String, Self)>,
    ) -> Result<Self> {
        let ordered_producers = ordered_producers.into_iter().collect::<Vec<_>>();
        if ordered_producers.is_empty() {
            return Err(composed_schedule_error("missing_producer_evidence", None));
        }

        let [consumer_shape] = consumer.requirement.reservation_shapes.as_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        if !consumer_shape.matches(consumer_plan) {
            return Err(composed_schedule_error("consumer_scope_coverage_mismatch", None));
        }
        let mut occurrence_ids = std::collections::HashSet::with_capacity(ordered_producers.len());
        let mut occurrences = Vec::with_capacity(ordered_producers.len());
        let mut producer_phases = Vec::with_capacity(ordered_producers.len());
        let mut producer_outputs = Vec::with_capacity(ordered_producers.len());
        let mut producer_proofs = Vec::with_capacity(ordered_producers.len());
        let mut retained_terms_by_occurrence = HashMap::with_capacity(ordered_producers.len());
        for (occurrence_id, producer) in ordered_producers {
            let normalized = Self::normalize_composed_producer(occurrence_id, producer, &mut occurrence_ids)?;
            retained_terms_by_occurrence
                .insert(normalized.occurrence_id.clone(), normalized.retained_output_term_id.clone());
            producer_outputs.push(ComposedGraphProducerOutputEvidence {
                occurrence_id: normalized.occurrence_id.clone(),
                row_count_upper_bound: normalized.row_count_upper_bound,
                column_evidence: normalized.output_column_evidence.clone(),
            });
            occurrences.push(AttemptReservationProducerOccurrence::from_term_id(
                normalized.occurrence_id.clone(),
                normalized.execution_scope_fingerprint,
                normalized.retained_output_term_id,
            ));
            producer_phases.push(
                if normalized.nested_schedule {
                    crate::admission::envelope::AttemptReservationProducerPhase::from_nested_term_groups(
                        normalized.phase_term_groups,
                        normalized.alternative_phase_term_ids,
                        normalized.dominated_term_ids,
                    )
                } else {
                    crate::admission::envelope::AttemptReservationProducerPhase::from_term_groups_with_alternatives_and_side_inputs(
                        normalized.phase_term_groups,
                        normalized.alternative_phase_term_ids,
                        normalized.concurrent_side_input_term_ids,
                    )
                },
            );
            producer_proofs.push(normalized.proof);
        }

        let ComposedConsumerBranchBound {
            owner_terms: consumer_owner_terms,
            peak_owner_term_groups: mut consumer_peak_term_groups,
            dominated_owner_term_ids: consumer_dominated_term_ids,
            allocation_state: consumer_state,
            consumed_occurrences,
            ..
        } = composed_consumer_bound_with_peak_terms(
            consumer_plan,
            &producer_outputs,
            &consumer.requirement.owner_terms,
        )?;
        consumer_peak_term_groups.extend(
            consumer.requirement.graph_phase_term_groups(
                consumer
                    .requirement
                    .reservation_shapes
                    .iter()
                    .find(|shape| shape.matches(consumer_plan))
                    .ok_or_else(|| composed_schedule_error("missing_consumer_scope", None))?
                    .execution_scope_fingerprint(),
            ),
        );
        let mut consumer = consumer;
        consumer.requirement.requires_composed_producer_binding = false;
        consumer.requirement.remove_generic_gpu_resident_retained_output_terms();
        consumer.requirement.owner_terms = merge_owner_terms(&consumer.requirement.owner_terms, consumer_owner_terms)?;
        let attempt_wide_fixed_term_ids = producer_proofs
            .iter()
            .flat_map(|proof| proof.requirement.owner_terms.iter())
            .chain(consumer.requirement.owner_terms.iter())
            .filter(|term| term.is_attempt_wide_fixed())
            .map(|term| term.id().clone())
            .collect::<Vec<_>>();
        let schedule = AttemptReservationSchedule::from_phase_term_groups_with_dominated(
            attempt_wide_fixed_term_ids,
            producer_phases,
            consumer_state,
            consumer_peak_term_groups,
            consumer_dominated_term_ids,
        );
        let [consumer_shape] = consumer.requirement.reservation_shapes.as_mut_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        for occurrence_id in &consumed_occurrences {
            if !retained_terms_by_occurrence.contains_key(occurrence_id) {
                return Err(composed_schedule_error("consumer_retained_input_binding_missing", Some(occurrence_id)));
            }
        }
        consumer_shape.set_producer_occurrences(occurrences)?;
        consumer_shape.attach_ordered_schedule(schedule)?;

        let first = producer_proofs.remove(0);
        AttemptCapabilityEvidence::aggregate(
            std::iter::once(first.requirement)
                .chain(producer_proofs.into_iter().map(|proof| proof.requirement))
                .chain(std::iter::once(consumer.requirement)),
        )
        .map(|requirement| Self { requirement })
    }

    /// Resolves adapter-observed graph-producer identities into the complete
    /// composed schedule when the engine owns every allocation in the attempt.
    ///
    /// Every producer count, including `N = 1`, uses this generalized path.
    /// A partial schedule cannot establish the live allocation timeline, so it
    /// fails closed with the underlying structured schedule error.
    pub fn resolve_composed_graph_attempt(
        consumer_plan: &QueryPlan,
        consumer: Self,
        ordered_producers: impl IntoIterator<Item = ComposedGraphProducerCandidate>,
    ) -> Result<Self> {
        let normalized_candidates =
            ordered_producers.into_iter().map(Self::normalize_composed_candidate).collect::<Result<Vec<_>>>()?;

        if let Some(proof) = Self::reuse_complete_schedule_for_passthrough_consumer(
            consumer_plan,
            consumer.clone(),
            &normalized_candidates,
        )? {
            return Ok(proof);
        }

        let ordered_result =
            Self::combine_composed_graph_attempt(consumer_plan, consumer.clone(), normalized_candidates.clone());
        match ordered_result {
            Ok(proof) => Ok(proof),
            Err(ordered_error) => {
                let conservative_result =
                    Self::combine_conservative_composed_graph_attempt(consumer_plan, consumer, normalized_candidates);
                match conservative_result {
                    Ok(proof) => Ok(proof),
                    Err(conservative_error) => {
                        let ordered_reason = ordered_error
                            .fact_value("schedule_reason")
                            .unwrap_or("ordered_timeline_unavailable")
                            .to_owned();
                        let conservative_reason = conservative_error
                            .fact_value("schedule_reason")
                            .or_else(|| conservative_error.fact_value("reason"))
                            .unwrap_or("conservative_owner_sum_unavailable")
                            .to_owned();
                        Err(ordered_error
                            .with_fact("ordered_reason", ordered_reason)
                            .with_fact("conservative_reason", conservative_reason))
                    },
                }
            },
        }
    }

    fn reuse_complete_schedule_for_passthrough_consumer(
        consumer_plan: &QueryPlan,
        consumer: Self,
        producers: &[(String, Self)],
    ) -> Result<Option<Self>> {
        let [(occurrence_id, producer)] = producers else {
            return Ok(None);
        };
        if !producer.requirement.reservation_shapes.iter().any(|shape| shape.ordered_schedule().is_some()) {
            return Ok(None);
        }
        let unreachable = consumer_plan.unreachable_node_ids();
        let reachable = consumer_plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).collect::<Vec<_>>();
        let [node] = reachable.as_slice() else {
            return Ok(None);
        };
        let PlanNode::Source(source) = &node.kind else {
            return Ok(None);
        };
        let SourceNode::GpuResident(source) = source.as_ref() else {
            return Ok(None);
        };
        if source.input_name != *occurrence_id || consumer_plan.root_id() != node.id {
            return Ok(None);
        }
        let mut requirement =
            AttemptCapabilityEvidence::aggregate([producer.requirement.clone(), consumer.requirement])?;
        requirement.requires_composed_producer_binding = false;
        Ok(Some(Self { requirement }))
    }

    fn combine_conservative_composed_graph_attempt(
        consumer_plan: &QueryPlan,
        consumer: Self,
        ordered_producers: Vec<(String, Self)>,
    ) -> Result<Self> {
        if ordered_producers.is_empty() {
            return Err(composed_schedule_error("missing_producer_evidence", None));
        }
        let [consumer_shape] = consumer.requirement.reservation_shapes.as_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        if !consumer_shape.matches(consumer_plan) {
            return Err(composed_schedule_error("consumer_scope_coverage_mismatch", None));
        }

        let mut occurrence_ids = HashSet::with_capacity(ordered_producers.len());
        let mut producer_occurrences = Vec::with_capacity(ordered_producers.len());
        let mut producer_outputs = Vec::with_capacity(ordered_producers.len());
        let mut producer_requirements = Vec::with_capacity(ordered_producers.len());
        for (occurrence_id, producer) in ordered_producers {
            let normalized = Self::normalize_composed_producer(occurrence_id, producer, &mut occurrence_ids)?;
            producer_occurrences.push(AttemptReservationProducerOccurrence::from_term_id(
                normalized.occurrence_id.clone(),
                normalized.execution_scope_fingerprint,
                normalized.retained_output_term_id,
            ));
            producer_outputs.push(ComposedGraphProducerOutputEvidence {
                occurrence_id: normalized.occurrence_id,
                row_count_upper_bound: normalized.row_count_upper_bound,
                column_evidence: normalized.output_column_evidence,
            });
            producer_requirements.push(normalized.proof.requirement);
        }

        let mut consumer = consumer;
        consumer.requirement.requires_composed_producer_binding = false;
        consumer.requirement.remove_generic_gpu_resident_retained_output_terms();
        let ComposedConsumerBranchBound { owner_terms, consumed_occurrences, .. } =
            composed_consumer_bound_with_peak_terms(
                consumer_plan,
                &producer_outputs,
                &consumer.requirement.owner_terms,
            )?;
        consumer.requirement.owner_terms = merge_owner_terms(&consumer.requirement.owner_terms, owner_terms)?;
        let [consumer_shape] = consumer.requirement.reservation_shapes.as_mut_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        for occurrence_id in &consumed_occurrences {
            if !producer_occurrences.iter().any(|occurrence| occurrence.occurrence_id() == occurrence_id) {
                return Err(composed_schedule_error("consumer_retained_input_binding_missing", Some(occurrence_id)));
            }
        }
        consumer_shape.set_producer_occurrences(producer_occurrences)?;
        producer_requirements.push(consumer.requirement);
        AttemptCapabilityEvidence::aggregate(producer_requirements).map(|requirement| Self { requirement })
    }

    /// Refines this engine-issued proof with an opaque cold graph-model hint.
    ///
    /// The model-owned graph evidence classifies whether the source frame and
    /// graph input share one owner or are concurrently live through explicit
    /// normalization. It becomes a complete phase only when it can be bound
    /// to one exact cuGraph execution scope; callers cannot supply raw
    /// graph-model bytes or a lifecycle choice.
    pub fn with_graph_memory_admission_evidence(mut self, evidence: &GraphMemoryAdmissionEvidence) -> Result<Self> {
        self.requirement.extend_with_graph_memory_admission_evidence(evidence)?;
        Ok(self)
    }

    /// Returns the device-capability contracts captured by the engine-issued
    /// proof. This is diagnostic-only and cannot be used to construct a new
    /// proof.
    pub fn required_capabilities(&self) -> &[ReservationRequirement] {
        &self.requirement.required_capabilities
    }

    /// Resolves the proof's immutable floor against an installed profile.
    /// This is read-only inspection and never selects a grant target.
    pub fn required_floor_bytes_for(&self, configured_floor_bytes: u64) -> Result<u64> {
        self.requirement.required_floor_bytes_for(configured_floor_bytes)
    }

    /// Returns the proof's descriptive working-set target, if planning
    /// supplied one. This is diagnostic-only; admission remains ledger-owned.
    pub fn soft_working_set_hint_bytes(&self) -> Option<u64> {
        self.requirement.soft_working_set_hint_bytes()
    }

    #[cfg(feature = "test-hooks")]
    #[doc(hidden)]
    pub fn ordered_schedule_phase_group_counts_for_test(&self) -> Option<(usize, usize)> {
        self.requirement
            .reservation_shapes
            .iter()
            .find_map(AttemptReservationShape::ordered_schedule)
            .map(AttemptReservationSchedule::phase_group_counts_for_test)
    }

    /// Resolves all device-dependent owner terms against one explicit
    /// candidate device ordinal.
    pub fn requirement_for_device(&self, device_ordinal: usize) -> Result<Self> {
        self.requirement.resolve_for_device(device_ordinal).map(|requirement| Self { requirement })
    }

    #[cfg(test)]
    pub(crate) fn requirement_for_device_with_preflight<F>(&self, device_ordinal: usize, preflight: F) -> Result<Self>
    where F: Fn(&SortMetadataDescriptor, usize) -> Result<SortRadixPreflight> {
        self.requirement
            .resolve_for_device_with_preflight_and_limit(device_ordinal, preflight, |_, _| {
                Err(limit_owner_error("test_preflight_missing", "limit"))
            })
            .map(|requirement| Self { requirement })
    }

    #[cfg(test)]
    pub(crate) fn requirement_for_device_with_preflight_and_limit<F, G>(
        &self,
        device_ordinal: usize,
        sort_preflight: F,
        limit_preflight: G,
    ) -> Result<Self>
    where
        F: Fn(&SortMetadataDescriptor, usize) -> Result<SortRadixPreflight>,
        G: Fn(&LimitMetadataDescriptor, usize) -> Result<LimitRangeGatherPreflight>,
    {
        self.requirement
            .resolve_for_device_with_preflight_and_limit(device_ordinal, sort_preflight, limit_preflight)
            .map(|requirement| Self { requirement })
    }

    /// Resolves the proof's private plan shapes against the policy installed
    /// by the query service that is about to admit it.
    pub(crate) fn resolve_reservation_envelope(
        &self,
        policy: AttemptReservationPolicy,
    ) -> Result<AttemptReservationEnvelope> {
        if self.requirement.requires_composed_producer_binding {
            return Err(composed_schedule_error("consumer_producer_binding_required", None));
        }
        AttemptReservationEnvelope::resolve_with_owner_terms(
            &self.requirement.reservation_shapes,
            &self.requirement.owner_terms,
            policy,
        )
    }

    #[cfg(test)]
    pub(crate) fn into_requirement(self) -> AttemptCapabilityEvidence {
        self.requirement
    }

    pub(crate) fn as_requirement(&self) -> &AttemptCapabilityEvidence {
        &self.requirement
    }
}

impl AttemptCapabilityEvidence {
    fn output_column_evidence_for_scope(&self, execution_scope_fingerprint: &str) -> PlanColumnEvidence {
        let matching = self
            .reservation_shapes
            .iter()
            .filter(|shape| shape.execution_scope_fingerprint() == execution_scope_fingerprint)
            .collect::<Vec<_>>();
        let [shape] = matching.as_slice() else {
            return PlanColumnEvidence::default();
        };
        if shape.output_column_evidence_input_scopes().iter().any(|(_, input_scope)| {
            !self.reservation_shapes.iter().any(|candidate| candidate.execution_scope_fingerprint() == input_scope)
        }) {
            return PlanColumnEvidence::default();
        }
        shape.output_column_evidence().clone()
    }

    fn remove_generic_gpu_resident_retained_output_terms(&mut self) {
        self.owner_terms.retain(|term| {
            !(matches!(term.id.owner, OwnerId::Producer(_)) && term.id.role == OwnerTermRole::RetainedOutput)
        });
    }

    /// Returns evidence derived from the engine's capability analysis for one
    /// lowered native plan.  The plan, rather than an embedding caller,
    /// determines the capability set.
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        Self::for_query_plan_with_policy(plan, OwnerTermConstructionPolicy::Generic)
    }

    fn for_query_plan_with_policy(plan: &QueryPlan, policy: OwnerTermConstructionPolicy) -> Self {
        Self::for_query_plan_with_policy_and_gpu_output(plan, policy, true)
    }

    fn for_query_plan_without_gpu_output_owner(plan: &QueryPlan, policy: OwnerTermConstructionPolicy) -> Self {
        Self::for_query_plan_with_policy_and_gpu_output(plan, policy, false)
    }

    fn for_query_plan_with_policy_and_gpu_output(
        plan: &QueryPlan,
        policy: OwnerTermConstructionPolicy,
        include_gpu_output_owner: bool,
    ) -> Self {
        let source_stage_bytes = estimated_reachable_local_parquet_source_bytes(plan);
        let source_owner_result = local_parquet_source_owner_terms(plan);
        let literal_owner_result = if policy == OwnerTermConstructionPolicy::PriceLiteralBatch {
            literal_batch_source_owner_terms(plan)
        } else {
            Ok(Vec::new())
        };
        let gpu_owner_result = generic_gpu_resident_owner_terms(plan, include_gpu_output_owner);
        let aggregate_owner_result = generic_aggregate_owner_terms(plan);
        let mut source_owner_terms = Vec::new();
        if let Ok(terms) = &source_owner_result {
            source_owner_terms.extend(terms.iter().cloned());
        }
        if let Ok(terms) = &literal_owner_result {
            source_owner_terms.extend(terms.iter().cloned());
        }
        if let Ok(gpu_owner_terms) = &gpu_owner_result {
            if gpu_owner_terms.iter().any(|term| {
                matches!(term.id.owner, OwnerId::Producer(producer_id) if source_owner_terms.iter().any(|source| {
                    matches!(source.id.owner, OwnerId::Source(source_id) if source_id == producer_id)
                }))
            }) {
                // A source-only GPU sink transfers the source frame into the
                // producer handle; retain one owner, never both lifetimes.
                source_owner_terms.clear();
            }
            source_owner_terms.extend(gpu_owner_terms.iter().cloned());
        }
        if let Ok(aggregate_owner_terms) = &aggregate_owner_result {
            source_owner_terms.extend(aggregate_owner_terms.iter().cloned());
        }
        let literal_rejection_reason = literal_batch_owner_term_rejection_reason(plan, policy);
        let mut evidence = Self::from_parts(
            0,
            0,
            None,
            source_stage_bytes.map_or(EstimateQuality::Unknown, |_| EstimateQuality::Modeled),
            plan.capability_analysis().requirements,
        );
        evidence.source_stage_bytes = source_stage_bytes;
        evidence.owner_terms = source_owner_terms;
        let mut shape = AttemptReservationShape::for_query_plan(plan);
        shape.set_output_column_evidence(PlanColumnEvidenceAnalysis::for_plan(plan, &HashMap::new()).root(plan));
        if let Some(reason) = literal_rejection_reason {
            shape.set_owner_term_rejection_reason(reason);
        } else if let Err(error) = source_owner_result {
            shape.set_owner_term_rejection_reason(
                error.fact_value("reason").unwrap_or("attempt_owner_term_unavailable"),
            );
        } else if let Err(error) = literal_owner_result {
            shape.set_owner_term_rejection_reason(
                error.fact_value("reason").unwrap_or("attempt_owner_term_unavailable"),
            );
        } else if let Err(error) = gpu_owner_result {
            shape.set_owner_term_rejection_reason(
                error.fact_value("reason").unwrap_or("attempt_owner_term_unavailable"),
            );
        } else if let Err(error) = aggregate_owner_result {
            shape.set_owner_term_rejection_reason(
                error.fact_value("reason").unwrap_or("attempt_owner_term_unavailable"),
            );
        }
        evidence.reservation_shapes.push(shape);
        evidence
    }

    fn mark_cugraph_graph_input_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_cugraph_graph_input_scope();
        }
    }

    fn mark_attested_graph_result_passthrough_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_attested_graph_result_passthrough_scope();
        }
    }

    fn mark_attested_graph_result_bounded_consumer_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_attested_graph_result_bounded_consumer_scope();
        }
    }

    /// Returns the descriptive working-set target, if a planner produced one.
    pub(crate) fn soft_working_set_hint_bytes(&self) -> Option<u64> {
        self.soft_working_set_hint_bytes
    }

    #[cfg(test)]
    pub(crate) fn bound_derivation(&self) -> Option<AttemptBoundDerivation> {
        self.bound_derivation
    }

    /// Computes the checked fallback bound from the one engine-issued owner
    /// inventory.  The inventory is an exact multiset: IDs are never silently
    /// deduplicated and every finite term is charged once.
    #[cfg(test)]
    pub(crate) fn conservative_owner_sum(&self) -> Result<u64> {
        checked_owner_sum(&self.owner_terms, &self.reservation_shapes)
    }

    /// Returns the provenance quality of the descriptive working-set target.
    #[cfg(test)]
    pub(crate) fn estimate_quality(&self) -> EstimateQuality {
        self.estimate_quality
    }

    #[cfg(test)]
    pub(crate) fn side_input_frame_owner_term_bytes(&self) -> Vec<u64> {
        self.owner_terms
            .iter()
            .filter(|term| term.id.role == OwnerTermRole::SideInputFrame)
            .map(|term| term.bytes.get())
            .collect()
    }

    /// Resolves this evidence against an already-installed device-profile
    /// floor.  This is read-only inspection; it never selects a grant.
    pub(crate) fn required_floor_bytes_for(&self, configured_floor_bytes: u64) -> Result<u64> {
        configured_floor_bytes
            .max(self.configured_floor_bytes)
            .max(self.required_phase_floor_bytes)
            .checked_add(self.capability_floor_raise_bytes)
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.admission.requirement",
                    "attempt capability floor overflows u64",
                )
                .with_fact("reason", "attempt_requirement_floor_overflow")
                .with_fact("configured_floor_bytes", configured_floor_bytes)
                .with_fact("required_phase_floor_bytes", self.required_phase_floor_bytes)
                .with_fact("capability_floor_raise_bytes", self.capability_floor_raise_bytes)
            })
    }

    /// Combines evidence from all fragments in one invocation.
    ///
    /// The installed device profile supplies the baseline floor exactly once;
    /// only independently live lower-layer raises are summed here.
    pub(crate) fn aggregate<I: IntoIterator<Item = Self>>(fragments: I) -> Result<Self> {
        Self::aggregate_with_configured_floor(0, fragments)
    }

    pub(crate) fn empty() -> Self {
        Self::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new())
    }

    pub fn from_parts(
        configured_floor_bytes: u64,
        capability_floor_raise_bytes: u64,
        soft_working_set_hint_bytes: Option<u64>,
        estimate_quality: EstimateQuality,
        required_capabilities: Vec<ReservationRequirement>,
    ) -> Self {
        Self {
            configured_floor_bytes,
            capability_floor_raise_bytes,
            required_phase_floor_bytes: 0,
            source_stage_bytes: None,
            soft_working_set_hint_bytes,
            estimate_quality,
            required_capabilities,
            owner_terms: Vec::new(),
            bound_derivation: None,
            requires_composed_producer_binding: false,
            attested_cugraph_graph_execution_scopes: Vec::new(),
            reservation_shapes: Vec::new(),
        }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn with_test_placement_overlay(mut self, test_requirement: AttemptCapabilityEvidence) -> Result<Self> {
        let placement_only = test_requirement.capability_floor_raise_bytes == 0
            && test_requirement.required_phase_floor_bytes == 0
            && test_requirement.source_stage_bytes.is_none()
            && test_requirement.required_capabilities.is_empty()
            && test_requirement.owner_terms.is_empty()
            && test_requirement.bound_derivation.is_none()
            && test_requirement.reservation_shapes.is_empty()
            && !test_requirement.requires_composed_producer_binding
            && test_requirement.attested_cugraph_graph_execution_scopes.is_empty();
        if !placement_only {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.admission.requirement",
                "test placement overlay contains non-placement capability evidence",
            )
            .with_fact("reason", "test_placement_overlay_not_placement_only"));
        }
        let test_target_bytes = test_requirement.soft_working_set_hint_bytes.unwrap_or(0);
        self.configured_floor_bytes =
            self.configured_floor_bytes.max(test_requirement.configured_floor_bytes).max(test_target_bytes);
        self.soft_working_set_hint_bytes =
            match (self.soft_working_set_hint_bytes, test_requirement.soft_working_set_hint_bytes) {
                (Some(base), Some(test)) => Some(base.max(test)),
                (Some(base), None) => Some(base),
                (None, Some(test)) => Some(test),
                (None, None) => None,
            };
        Ok(self)
    }

    /// Applies a service-resolved envelope immediately before placement. A
    /// graph envelope with an incomplete scope is rejected before this point;
    /// non-graph incomplete shapes retain the ledger's conservative candidate
    /// ceiling rather than a partial phase estimate.
    pub(crate) fn with_reservation_envelope(mut self, envelope: &AttemptReservationEnvelope) -> Self {
        self.required_phase_floor_bytes = self.required_phase_floor_bytes.max(envelope.required_phase_floor_bytes());
        self.soft_working_set_hint_bytes = envelope.complete_phase_peak_bytes();
        self.bound_derivation = envelope.bound_derivation();
        self.estimate_quality =
            envelope.complete_phase_peak_bytes().map_or(EstimateQuality::Unknown, |_| EstimateQuality::Modeled);
        self
    }

    /// Combines an engine-issued graph-native cold peak with the matching
    /// native source stage using the graph model's exact source/graph lifetime
    /// classification. Raw graph input shares its prepared frame with the
    /// graph peak; explicit normalization retains the source while producing
    /// a separate graph input and therefore composes additively.
    fn extend_with_graph_memory_admission_evidence(&mut self, evidence: &GraphMemoryAdmissionEvidence) -> Result<()> {
        let graph_shape_index = self.graph_memory_evidence_scope_index(evidence)?;
        let graph_owner_terms = graph_owner_terms_from_evidence(evidence)?;
        let whole_attempt_peak_bytes = match self.source_stage_bytes {
            Some(source_stage_bytes) => evidence.bounded_live_graph_peak_with_source_stage_bytes(source_stage_bytes)?,
            None => evidence.bounded_live_graph_peak_bytes()?,
        };
        let source_stage_bytes = self.source_stage_bytes;

        let mut owner_terms = self.owner_terms.clone();
        self.remove_shared_graph_input_source_stage_owners(graph_shape_index, evidence, &mut owner_terms)?;
        owner_terms = merge_owner_terms(&owner_terms, graph_owner_terms)?;

        // Validate every graph-owned scope before changing the hint or any
        // reservation shape so a rejected evidence token leaves the proof
        // untouched.
        self.reservation_shapes[graph_shape_index].set_graph_memory_phase(evidence, source_stage_bytes);

        self.account_graph_owned_runtime_scopes(graph_shape_index);

        self.owner_terms = owner_terms;
        self.soft_working_set_hint_bytes = Some(whole_attempt_peak_bytes);
        self.estimate_quality = EstimateQuality::Modeled;
        Ok(())
    }

    fn remove_shared_graph_input_source_stage_owners(
        &self,
        graph_shape_index: usize,
        evidence: &GraphMemoryAdmissionEvidence,
        owner_terms: &mut Vec<AllocationOwnerTerm>,
    ) -> Result<()> {
        if evidence.source_stage_lifetime() != GraphSourceStageLifetime::SharedPreparedEdgeFrame {
            return Ok(());
        }
        let graph_shape = &self.reservation_shapes[graph_shape_index];
        if !graph_shape.is_cugraph_graph_input_scope() || !graph_shape.has_modeled_source_stage() {
            return Ok(());
        }
        let matching = owner_terms
            .iter()
            .filter(|term| {
                term.id().role == OwnerTermRole::SourceStage
                    && term.execution_scope_fingerprint() == evidence.execution_scope_fingerprint()
            })
            .map(|term| term.id().clone())
            .collect::<Vec<_>>();
        if matching.len() > 1 {
            return Err(owner_term_error("graph_input_source_stage_ambiguous", Some(OwnerTermRole::SourceStage), None)
                .with_fact("execution_scope_fingerprint", evidence.execution_scope_fingerprint())
                .with_fact("matching_owner_count", matching.len()));
        }
        if let Some(owner_id) = matching.first() {
            owner_terms.retain(|term| term.id() != owner_id);
        }
        Ok(())
    }

    fn graph_memory_evidence_scope_index(&self, evidence: &GraphMemoryAdmissionEvidence) -> Result<usize> {
        let [scope] = self.attested_cugraph_graph_execution_scopes.as_slice() else {
            let reason = if self.attested_cugraph_graph_execution_scopes.is_empty() {
                "graph_memory_evidence_scope_missing"
            } else {
                "graph_memory_evidence_scope_ambiguous"
            };
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.requirement",
                "graph memory admission evidence requires exactly one attested graph execution scope",
            )
            .with_fact("reason", reason)
            .with_fact("attested_scope_count", self.attested_cugraph_graph_execution_scopes.len()));
        };
        if !scope.matches_graph_memory_evidence(evidence) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.requirement",
                "graph memory admission evidence scope does not match the attested graph execution plan",
            )
            .with_fact("reason", "graph_memory_evidence_scope_mismatch"));
        }
        let matching_shapes = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| scope.matches_reservation_shape(shape).then_some(index))
            .collect::<Vec<_>>();
        let [index] = matching_shapes.as_slice() else {
            let reason = if matching_shapes.is_empty() {
                "graph_memory_evidence_scope_missing"
            } else {
                "graph_memory_evidence_scope_ambiguous"
            };
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.requirement",
                "graph memory admission evidence has no unique reservation shape",
            )
            .with_fact("reason", reason)
            .with_fact("matching_shape_count", matching_shapes.len()));
        };
        Ok(*index)
    }

    fn retained_output_rows_for_graph_result(&self) -> Result<u64> {
        let [scope] = self.attested_cugraph_graph_execution_scopes.as_slice() else {
            let reason = if self.attested_cugraph_graph_execution_scopes.is_empty() {
                "producer_graph_scope_missing"
            } else {
                "producer_graph_scope_ambiguous"
            };
            return Err(sort_owner_error(reason, "graph_result_output"));
        };
        let matching_shapes =
            self.reservation_shapes.iter().filter(|shape| scope.matches_reservation_shape(shape)).collect::<Vec<_>>();
        let [shape] = matching_shapes.as_slice() else {
            let reason = if matching_shapes.is_empty() {
                "producer_graph_scope_missing"
            } else {
                "producer_graph_scope_ambiguous"
            };
            return Err(sort_owner_error(reason, "graph_result_output"));
        };
        let evidence = shape
            .graph_memory_admission_evidence()
            .ok_or_else(|| sort_owner_error("producer_graph_memory_evidence_missing", "graph_result_output"))?;
        if !scope.matches_graph_memory_evidence(evidence) {
            return Err(sort_owner_error("producer_graph_scope_mismatch", "graph_result_output"));
        }
        Ok(evidence.retained_output_rows_upper_bound())
    }

    fn retained_output_term_for_composition(
        &self,
        nested_retained_output_term_ids: &HashSet<OwnerTermId>,
    ) -> Result<&AllocationOwnerTerm> {
        let mut retained = self
            .owner_terms
            .iter()
            .filter(|term| term.is_retained_output() && !nested_retained_output_term_ids.contains(term.id()));
        let term = retained.next().ok_or_else(|| {
            composed_schedule_error("producer_retained_output_term_missing", None).with_fact("phase", "retained_output")
        })?;
        if retained.next().is_some() {
            return Err(composed_schedule_error("producer_retained_output_term_ambiguous", None)
                .with_fact("phase", "retained_output"));
        }
        if term.retained_row_count_upper_bound().is_none() {
            return Err(composed_schedule_error("producer_retained_output_rows_missing", None)
                .with_fact("phase", "retained_output"));
        }
        Ok(term)
    }

    #[cfg(test)]
    fn isolated_phase_term_groups(&self) -> Vec<Vec<OwnerTermId>> {
        self.isolated_phase_term_groups_with_alternatives(None).0
    }

    fn concurrent_side_input_term_ids(&self) -> Vec<OwnerTermId> {
        self.owner_terms.iter().filter(|term| term.is_side_input_frame()).map(|term| term.id().clone()).collect()
    }

    fn graph_phase_term_groups(&self, execution_scope_fingerprint: &str) -> Vec<Vec<OwnerTermId>> {
        let mut construction = Vec::new();
        let mut execution = Vec::new();
        for term in
            self.owner_terms.iter().filter(|term| term.execution_scope_fingerprint() == execution_scope_fingerprint)
        {
            if term.is_construction_phase_term() {
                construction.push(term.id().clone());
            }
            if term.is_execution_phase_term() || term.is_retained_output() {
                execution.push(term.id().clone());
            }
        }
        [construction, execution].into_iter().filter(|group| !group.is_empty()).collect()
    }

    fn isolated_phase_term_groups_with_alternatives(
        &self,
        execution_scope_fingerprint: Option<&str>,
    ) -> (Vec<Vec<OwnerTermId>>, Vec<OwnerTermId>) {
        let mut construction = Vec::new();
        let mut execution = Vec::new();
        let mut other_active = Vec::new();
        let mut alternative_term_ids = Vec::new();
        let separate_graph_source_scopes = self
            .reservation_shapes
            .iter()
            .filter_map(AttemptReservationShape::graph_memory_admission_evidence)
            .filter(|evidence| evidence.source_stage_lifetime() == GraphSourceStageLifetime::SeparateNormalizedInput)
            .map(|evidence| evidence.execution_scope_fingerprint())
            .collect::<HashSet<_>>();
        for term in &self.owner_terms {
            if execution_scope_fingerprint.is_some_and(|scope| term.execution_scope_fingerprint() != scope) {
                continue;
            }
            if term.is_attempt_wide_fixed() {
                continue;
            }
            if term.is_retained_output() {
                execution.push(term.id().clone());
                continue;
            }
            if term.is_construction_phase_term() {
                construction.push(term.id().clone());
            }
            if term.is_execution_phase_term() {
                execution.push(term.id().clone());
            }
            let separate_graph_source = term.id().role == OwnerTermRole::SourceStage
                && separate_graph_source_scopes.contains(term.execution_scope_fingerprint());
            if separate_graph_source {
                construction.push(term.id().clone());
                execution.push(term.id().clone());
            } else if !term.is_graph_phase_term() {
                other_active.push(term.id().clone());
            }
        }
        let mut groups = Vec::new();
        if !construction.is_empty() {
            groups.push(construction);
        }
        if !execution.is_empty() {
            groups.push(execution);
        }
        if !other_active.is_empty() {
            groups.push(other_active);
        }
        for term_id in groups.first().into_iter().flat_map(|group| group.iter()) {
            if groups.get(1).is_some_and(|group| group.contains(term_id)) {
                let Some(term) = self.owner_terms.iter().find(|term| term.id() == term_id) else {
                    continue;
                };
                let separate_source =
                    term.is_source_stage() && separate_graph_source_scopes.contains(term.execution_scope_fingerprint());
                if term.is_graph_phase_alternative() || separate_source {
                    alternative_term_ids.push(term_id.clone());
                }
            }
        }
        (groups, alternative_term_ids)
    }

    /// Retains exact scopes for graph-owned frames while treating each one as
    /// part of the already-modeled graph phase. This must run again after
    /// composition, because a prepared graph ticket is refined before an
    /// outer native host-passthrough adds its exact result scope.
    fn account_graph_owned_runtime_scopes(&mut self, graph_shape_index: usize) {
        // The graph phase consumes each exact local-Parquet stage carried by
        // this invocation. Keep those scopes for execution-time equality
        // validation, but do not add their bytes a second time after the
        // graph model already classified their lifetime.
        for (index, shape) in self.reservation_shapes.iter_mut().enumerate() {
            if index != graph_shape_index && shape.is_cugraph_graph_input_scope() && shape.has_modeled_source_stage() {
                shape.account_for_by_graph_phase();
            }
        }

        // The adapter identifies exactly one native fragment as the producer
        // of the graph's prepared edge frame. Its output is the graph model's
        // input owner, including literal/DataFusion sources that do not have a
        // local-Parquet stage. Retain the structural scope for execution-time
        // validation while folding that known lifetime into the graph phase.
        let graph_input_scope_indices = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| {
                (index != graph_shape_index
                    && shape.is_cugraph_graph_input_scope()
                    && (!shape.has_local_parquet_source() || shape.has_modeled_source_stage()))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if let [graph_input_scope_index] = graph_input_scope_indices.as_slice() {
            self.reservation_shapes[*graph_input_scope_index].account_for_by_graph_phase();
        }

        // A composed native root may merely expose the graph executor's
        // already-live result frame to its host sink. Only the graph frontend
        // can attest that exact source identity; keep the scope for runtime
        // equality while charging no second phase for the same frame.
        let graph_result_passthrough_scope_indices = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.is_attested_graph_result_passthrough_scope().then_some(index))
            .collect::<Vec<_>>();
        if let [graph_result_passthrough_scope_index] = graph_result_passthrough_scope_indices.as_slice() {
            self.reservation_shapes[*graph_result_passthrough_scope_index].account_for_by_graph_phase();
        }
    }

    /// A graph model token has no caller-provided plan identity. It can only
    /// refine a proof that attests exactly one graph execution scope, and it
    /// must attach to the corresponding reservation shape rather than a
    /// generic fragment in a composed proof.
    fn exact_attested_graph_reservation_shape_index(&self) -> Option<usize> {
        let [scope] = self.attested_cugraph_graph_execution_scopes.as_slice() else {
            return None;
        };
        let mut matching_shapes = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| scope.matches_reservation_shape(shape).then_some(index));
        let index = matching_shapes.next()?;
        matching_shapes.next().is_none().then_some(index)
    }

    fn take_nested_schedule_peak_terms(
        &mut self,
    ) -> Result<Option<crate::admission::envelope::NestedProducerPeakTerms>> {
        let scheduled = self
            .reservation_shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.ordered_schedule().is_some().then_some(index))
            .collect::<Vec<_>>();
        if scheduled.len() > 1 {
            return Err(composed_schedule_error("duplicate_schedule_evidence", None));
        }
        let Some(index) = scheduled.first().copied() else {
            return Ok(None);
        };
        let shape = &self.reservation_shapes[index];
        let schedule = shape.ordered_schedule().expect("scheduled index retains its schedule");
        schedule.validate_term_references(shape, &self.owner_terms, &self.reservation_shapes)?;
        let peak_terms = schedule.nested_producer_peak_terms(shape.producer_occurrences())?;
        self.reservation_shapes[index]
            .take_ordered_schedule_for_nesting()
            .expect("scheduled index retains its schedule");
        Ok(Some(peak_terms))
    }

    fn rebind_owner_terms_to_composed_occurrence(
        &mut self,
        occurrence_id: &str,
    ) -> Result<HashMap<OwnerTermId, OwnerTermId>> {
        if occurrence_id.is_empty() {
            return Err(owner_term_error("missing_occurrence", None, None));
        }
        let rebound = self
            .owner_terms
            .iter()
            .map(|term| {
                let rebound = term.rebind_composed_occurrence(occurrence_id)?;
                Ok((term.id().clone(), rebound))
            })
            .collect::<Result<Vec<_>>>()?;
        let identities = rebound.iter().map(|(before, after)| (before.clone(), after.id().clone())).collect();
        self.owner_terms = merge_owner_terms(&[], rebound.into_iter().map(|(_, term)| term))?;
        Ok(identities)
    }

    /// Stable, human-readable identities for structured device-capability
    /// rejection facts.
    pub(crate) fn capability_labels(required_capabilities: &[ReservationRequirement]) -> String {
        required_capabilities
            .iter()
            .map(|capability| format!("{}:{}", capability.kind.as_str(), capability.required_contract))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The non-negotiable attempt floor, rejecting integer overflow.
    pub(crate) fn required_floor_bytes(&self) -> Result<u64> {
        self.required_floor_bytes_for(self.configured_floor_bytes)
    }

    fn resolve_for_device(&self, device_ordinal: usize) -> Result<Self> {
        if !self.reservation_shapes.iter().any(|shape| shape.sort_metadata().is_some()) {
            return Ok(self.clone());
        }
        let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.admission.requirement.sort_preflight_device",
                "candidate device is unavailable for sorted-order owner preflight",
                error,
            )
            .with_fact("reason", "sort_owner_candidate_device_unavailable")
        })?;
        self.resolve_for_device_with_preflight_and_limit(
            device_ordinal,
            |descriptor, _| {
                let preflight = cudf::DataFramePreparedSort::sorted_order_radix_preflight_for_metadata(
                    descriptor.num_rows,
                    descriptor.key_type,
                    descriptor.null_count,
                    descriptor.stable,
                    descriptor.key_order,
                    device,
                )
                .map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.admission.requirement.sort_preflight",
                        "native sorted-order owner preflight failed",
                        error,
                    )
                    .with_fact("reason", "sort_owner_native_preflight_failed")
                })?;
                SortRadixPreflight::checked_from_cudf(preflight)
            },
            |descriptor, _| {
                let preflight = cudf::ops::copying::gather_fixed_width_dont_check_preflight_for_metadata(
                    &descriptor.output_columns,
                    descriptor.output_rows,
                    device,
                )
                .map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.admission.requirement.limit_preflight",
                        "native bounded limit range-gather owner preflight failed",
                        error,
                    )
                    .with_fact("reason", "limit_owner_native_preflight_failed")
                })?;
                LimitRangeGatherPreflight::checked_from_cudf(preflight)
            },
        )
    }

    fn resolve_for_device_with_preflight_and_limit<F, G>(
        &self,
        device_ordinal: usize,
        sort_preflight: F,
        limit_preflight: G,
    ) -> Result<Self>
    where
        F: Fn(&SortMetadataDescriptor, usize) -> Result<SortRadixPreflight>,
        G: Fn(&LimitMetadataDescriptor, usize) -> Result<LimitRangeGatherPreflight>,
    {
        let mut resolved = self.clone();
        let descriptors =
            self.reservation_shapes.iter().filter_map(AttemptReservationShape::sort_metadata).collect::<Vec<_>>();
        for descriptor in &descriptors {
            validate_sort_metadata_descriptor(descriptor)?;
            let components = sort_preflight(descriptor, device_ordinal)?;
            let terms = sort_owner_terms(descriptor, components)?;
            resolved.owner_terms = merge_owner_terms(&resolved.owner_terms, terms)?;
            resolved.set_sort_phase_peak(descriptor, components.active_phase_peak_bytes)?;
            if let Some(limit) = descriptor.limit.as_ref() {
                let components = limit_preflight(limit, device_ordinal)?;
                let terms = limit_owner_terms(limit, components)?;
                resolved.owner_terms = merge_owner_terms(&resolved.owner_terms, terms)?;
                resolved.set_limit_phase_peak(limit, components.active_phase_peak_bytes)?;
            }
        }
        Ok(resolved)
    }

    fn set_sort_phase_peak(&mut self, descriptor: &SortMetadataDescriptor, peak_bytes: u64) -> Result<()> {
        let matching = self
            .reservation_shapes
            .iter_mut()
            .filter(|shape| shape.matches_execution_scope_fingerprint(&descriptor.execution_scope_fingerprint));
        let mut matching = matching.peekable();
        let Some(shape) = matching.next() else {
            return Err(sort_owner_error("reservation_scope_missing", "sort_scope"));
        };
        if matching.peek().is_some() {
            return Err(sort_owner_error("reservation_scope_ambiguous", "sort_scope"));
        }
        shape.set_sort_phase_peak_bytes(peak_bytes)
    }

    fn set_limit_phase_peak(&mut self, descriptor: &LimitMetadataDescriptor, peak_bytes: u64) -> Result<()> {
        let matching = self
            .reservation_shapes
            .iter_mut()
            .filter(|shape| shape.matches_execution_scope_fingerprint(&descriptor.execution_scope_fingerprint));
        let mut matching = matching.peekable();
        let Some(shape) = matching.next() else {
            return Err(limit_owner_error("reservation_scope_missing", "limit_scope"));
        };
        if matching.peek().is_some() {
            return Err(limit_owner_error("reservation_scope_ambiguous", "limit_scope"));
        }
        shape.set_limit_phase_peak_bytes(peak_bytes)
    }

    /// Verifies that this admitted requirement covers the plan about to run.
    ///
    /// A ticket may aggregate proofs for several native fragments, so the
    /// admitted set may be a strict superset.  Missing contracts are unsafe:
    /// the execution plan must be rejected before it allocates or dispatches
    /// GPU work.
    pub(crate) fn validate_covers_plan(&self, plan: &QueryPlan) -> Result<()> {
        let required_capabilities = plan.capability_analysis().requirements;
        let cugraph_graph_execution_is_proven =
            self.required_capabilities.contains(&cugraph_graph_execution_capability())
                && self.attested_cugraph_graph_execution_scopes.iter().any(|scope| scope.matches(plan));
        let missing_capabilities = required_capabilities
            .iter()
            .filter(|required| {
                !self.required_capabilities.contains(required)
                    && !(cugraph_graph_execution_is_proven && is_generic_graph_execution_blocker(required))
            })
            .cloned()
            .collect::<Vec<_>>();
        if missing_capabilities.is_empty() {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.requirement",
            "admitted capability proof does not cover the native plan about to execute",
        )
        .with_fact("reason", "admitted_capability_evidence_mismatch")
        .with_fact("required_capabilities", Self::capability_labels(&required_capabilities))
        .with_fact("admitted_capabilities", Self::capability_labels(&self.required_capabilities))
        .with_fact("missing_capabilities", Self::capability_labels(&missing_capabilities)))
    }

    /// Aggregates all native fragments in one invocation.
    ///
    /// Capability workspace raises are summed deliberately: independently
    /// scheduled fragments can be concurrently live under one attempt grant.
    /// The configured deployment floor is supplied once and is never summed.
    pub(crate) fn aggregate_with_configured_floor<I: IntoIterator<Item = Self>>(
        configured_floor_bytes: u64,
        fragments: I,
    ) -> Result<Self> {
        let mut capability_floor_raise_bytes = 0_u64;
        let mut required_phase_floor_bytes = 0_u64;
        let mut source_stage_bytes = Some(0_u64);
        let mut source_stage_was_modeled = false;
        let mut graph_source_stage_bytes = None;
        let mut graph_source_stage_unmodeled = false;
        let mut soft_working_set_hint_bytes = Some(0_u64);
        let mut estimate_quality = EstimateQuality::Measured;
        let mut required_capabilities = Vec::new();
        let mut owner_terms = Vec::new();
        let mut attested_cugraph_graph_execution_scopes = Vec::new();
        let mut reservation_shapes = Vec::new();
        let mut cugraph_side_input_scopes = HashSet::new();
        let mut requires_composed_producer_binding = false;
        for fragment in fragments {
            requires_composed_producer_binding |= fragment.requires_composed_producer_binding;
            for shape in &fragment.reservation_shapes {
                if shape.is_cugraph_side_input_scope()
                    && !cugraph_side_input_scopes.insert(shape.execution_scope_fingerprint().to_owned())
                {
                    return Err(composed_schedule_error("duplicate_predicate_identity", None)
                        .with_fact("phase", "side_input_frame"));
                }
            }
            owner_terms = merge_owner_terms(&owner_terms, fragment.owner_terms.clone())?;
            capability_floor_raise_bytes =
                capability_floor_raise_bytes.checked_add(fragment.capability_floor_raise_bytes).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.admission.requirement",
                        "aggregated capability floor raise overflows u64",
                    )
                    .with_fact("reason", "attempt_requirement_capability_raise_overflow")
                })?;
            required_phase_floor_bytes = required_phase_floor_bytes.max(fragment.required_phase_floor_bytes);
            let fragment_has_reachable_local_source =
                fragment.reservation_shapes.iter().any(AttemptReservationShape::has_local_parquet_source);
            source_stage_was_modeled |= fragment.source_stage_bytes.is_some();
            source_stage_bytes = match (source_stage_bytes, fragment.source_stage_bytes) {
                (None, _) => None,
                (Some(_), None) if fragment_has_reachable_local_source => None,
                (Some(total), Some(bytes)) => {
                    Some(total.checked_add(bytes).ok_or_else(|| source_stage_overflow_error("source_stage_bytes"))?)
                },
                (Some(total), None) => Some(total),
            };
            let fragment_has_graph_input_scope =
                fragment.reservation_shapes.iter().any(AttemptReservationShape::is_cugraph_graph_input_scope);
            if fragment_has_graph_input_scope {
                let fragment_has_unmodeled_graph_source = fragment.reservation_shapes.iter().any(|shape| {
                    shape.is_cugraph_graph_input_scope()
                        && shape.has_local_parquet_source()
                        && !shape.has_modeled_source_stage()
                });
                if fragment_has_unmodeled_graph_source {
                    graph_source_stage_unmodeled = true;
                    graph_source_stage_bytes = None;
                } else if !graph_source_stage_unmodeled {
                    graph_source_stage_bytes = match (graph_source_stage_bytes, fragment.source_stage_bytes) {
                        (None, None) => None,
                        (None, Some(bytes)) | (Some(bytes), None) => Some(bytes),
                        (Some(total), Some(bytes)) => Some(
                            total
                                .checked_add(bytes)
                                .ok_or_else(|| source_stage_overflow_error("source_stage_bytes"))?,
                        ),
                    };
                }
            }
            soft_working_set_hint_bytes = match (soft_working_set_hint_bytes, fragment.soft_working_set_hint_bytes) {
                (Some(left), Some(right)) => Some(left.max(right)),
                _ => None,
            };
            estimate_quality = estimate_quality.min(fragment.estimate_quality);
            for capability in fragment.required_capabilities {
                if !required_capabilities.contains(&capability) {
                    required_capabilities.push(capability);
                }
            }
            for scope in fragment.attested_cugraph_graph_execution_scopes {
                if !attested_cugraph_graph_execution_scopes.contains(&scope) {
                    attested_cugraph_graph_execution_scopes.push(scope);
                }
            }
            for shape in fragment.reservation_shapes {
                // A reservation shape is an allocation occurrence.  Structural
                // equality is evidence identity, not runtime multiplicity;
                // equal graph producers must remain separate retained outputs.
                reservation_shapes.push(shape);
            }
        }
        let mut aggregate = Self::from_parts(
            configured_floor_bytes,
            capability_floor_raise_bytes,
            soft_working_set_hint_bytes,
            estimate_quality,
            required_capabilities,
        );
        aggregate.required_phase_floor_bytes = required_phase_floor_bytes;
        aggregate.source_stage_bytes = if attested_cugraph_graph_execution_scopes.is_empty() {
            source_stage_bytes.filter(|_| source_stage_was_modeled)
        } else if graph_source_stage_unmodeled {
            None
        } else {
            graph_source_stage_bytes
        };
        aggregate.owner_terms = owner_terms;
        aggregate.requires_composed_producer_binding = requires_composed_producer_binding;
        aggregate.attested_cugraph_graph_execution_scopes = attested_cugraph_graph_execution_scopes;
        aggregate.reservation_shapes = reservation_shapes;
        if let Some(graph_shape_index) = aggregate.exact_attested_graph_reservation_shape_index()
            && aggregate.reservation_shapes[graph_shape_index].has_graph_memory_phase()
        {
            aggregate.account_graph_owned_runtime_scopes(graph_shape_index);
        }
        // Provenance is selected only after the service resolves the final
        // immutable envelope. Aggregation must not badge a derivation that has
        // not yet computed the final peak.
        aggregate.bound_derivation = None;
        Ok(aggregate)
    }
}

pub(crate) fn checked_owner_sum(
    owner_terms: &[AllocationOwnerTerm],
    scopes: &[AttemptReservationShape],
) -> Result<u64> {
    for scope in scopes {
        if let Some(reason) = scope.owner_term_rejection_reason() {
            return Err(owner_derivation_error(reason));
        }
        if let Some(schedule) = scope.ordered_schedule() {
            schedule
                .validate_term_references_for_conservative_sum(scope, owner_terms, scopes)
                .map_err(|_| owner_derivation_error("attempt_owner_term_missing"))?;
        }
        scope.validate_manifest_core().map_err(|_| owner_derivation_error("attempt_manifest_core_invalid"))?;
        for occurrence in scope.producer_occurrences() {
            let term_id = occurrence.retained_output_term_id();
            let term = owner_terms.iter().find(|term| term.id() == term_id).ok_or_else(|| {
                owner_derivation_error("attempt_retained_input_term_missing")
                    .with_fact("occurrence_id", occurrence.occurrence_id())
            })?;
            if occurrence.validate_retained_output_term(term).is_err() {
                return Err(owner_derivation_error("attempt_retained_input_binding_invalid")
                    .with_fact("occurrence_id", occurrence.occurrence_id()));
            }
        }
    }
    let mut seen = HashSet::with_capacity(owner_terms.len());
    owner_terms.iter().try_fold(0_u64, |total, term| {
        validate_owner_term(term)?;
        if !seen.insert(term.id.clone()) {
            return Err(owner_derivation_error("attempt_owner_term_duplicate"));
        }
        total.checked_add(term.bytes.get()).ok_or_else(|| {
            owner_derivation_error("attempt_owner_term_sum_overflow")
                .with_fact("owner_term_role", term.id.role.as_str())
                .with_fact("occurrence", term.id.occurrence.as_str())
        })
    })
}

pub(crate) fn owner_inventory_covers_scopes(
    owner_terms: &[AllocationOwnerTerm],
    scopes: &[AttemptReservationShape],
    owner_sum: u64,
    required_phase_floor_bytes: u64,
    ordered_schedule_present: bool,
) -> bool {
    if !ordered_schedule_present && owner_sum < required_phase_floor_bytes {
        return false;
    }
    if owner_terms.is_empty()
        || owner_terms.iter().any(|term| {
            !scopes.iter().any(|scope| scope.execution_scope_fingerprint() == term.execution_scope_fingerprint())
        })
        || scopes.iter().any(|scope| {
            !scope.is_accounted_by_graph_phase()
                && !owner_terms
                    .iter()
                    .any(|term| term.execution_scope_fingerprint() == scope.execution_scope_fingerprint())
        })
        || owner_terms.iter().all(|term| !term.provides_complete_phase_coverage())
    {
        return false;
    }
    checked_owner_sum(owner_terms, scopes).is_ok()
}

fn source_stage_overflow_error(term: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.admission.requirement",
        "local-Parquet source stage bytes overflow u64",
    )
    .with_fact("reason", "attempt_requirement_source_stage_overflow")
    .with_fact("term", term)
    .with_fact("operation", "addition")
}

#[derive(Debug, Clone)]
struct ComposedGraphProducerOutputEvidence {
    occurrence_id: String,
    row_count_upper_bound: u64,
    column_evidence: PlanColumnEvidence,
}

#[derive(Debug, Clone)]
struct NormalizedComposedProducer {
    occurrence_id: String,
    execution_scope_fingerprint: String,
    retained_output_term_id: OwnerTermId,
    row_count_upper_bound: u64,
    output_column_evidence: PlanColumnEvidence,
    phase_term_groups: Vec<Vec<OwnerTermId>>,
    alternative_phase_term_ids: Vec<OwnerTermId>,
    concurrent_side_input_term_ids: Vec<OwnerTermId>,
    dominated_term_ids: Vec<OwnerTermId>,
    nested_schedule: bool,
    proof: QueryPlanCapabilityProof,
}

#[derive(Debug, Clone)]
struct ComposedConsumerBranchBound {
    rows: u64,
    schema: SchemaRef,
    /// Bytes allocated by this consumer branch that remain live at its root.
    /// Producer result handles are intentionally excluded: the schedule fold
    /// retains them outside the consumer phase.
    final_owned_bytes: u64,
    /// Maximum additional bytes owned while this branch is materialized.
    additional_peak_bytes: u64,
    owner_terms: Vec<AllocationOwnerTerm>,
    /// Alternative consumer lifetime peaks; each group is summed and the
    /// schedule takes the maximum across groups.
    peak_owner_term_groups: Vec<Vec<OwnerTermId>>,
    /// Flattened view retained for branch-local dominance decisions.
    peak_owner_term_ids: Vec<OwnerTermId>,
    dominated_owner_term_ids: Vec<OwnerTermId>,
    final_owned_term_ids: Vec<OwnerTermId>,
    allocation_state: AttemptReservationConsumerState,
    consumed_occurrences: Vec<String>,
}

#[cfg(test)]
fn composed_consumer_additional_peak_bytes_for_plan(
    consumer_plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
) -> Result<(u64, Vec<AllocationOwnerTerm>)> {
    let bound = composed_consumer_bound_with_peak_terms(consumer_plan, producer_outputs, &[])?;
    Ok((bound.additional_peak_bytes, bound.owner_terms))
}

fn composed_consumer_bound_with_peak_terms(
    consumer_plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
    owner_terms: &[AllocationOwnerTerm],
) -> Result<ComposedConsumerBranchBound> {
    if !consumer_plan.unreachable_node_ids().is_empty() {
        return Err(composed_schedule_error("consumer_contains_unreachable_nodes", None));
    }
    let mut outputs = HashMap::with_capacity(producer_outputs.len());
    for output in producer_outputs {
        if output.occurrence_id.is_empty() || outputs.insert(output.occurrence_id.as_str(), output).is_some() {
            return Err(composed_schedule_error("duplicate_producer_occurrence_id", Some(&output.occurrence_id)));
        }
    }
    let resident_outputs = producer_outputs
        .iter()
        .map(|output| (output.occurrence_id.clone(), output.column_evidence.clone()))
        .collect::<HashMap<_, _>>();
    let column_evidence = PlanColumnEvidenceAnalysis::for_plan(consumer_plan, &resident_outputs);
    let mut used_occurrences = HashSet::with_capacity(outputs.len());
    let mut memo = HashMap::new();
    let root = consumer_plan.root_id();
    let mut root_bound = composed_consumer_branch_bound(
        consumer_plan,
        root,
        &outputs,
        &mut used_occurrences,
        owner_terms,
        &column_evidence,
        &mut memo,
    )?;
    if consumer_plan.contains_graph_algorithm() {
        let mut inventory = owner_terms.to_vec();
        inventory.extend(root_bound.owner_terms.iter().cloned());
        root_bound.peak_owner_term_groups = memoized_graph_consumer_peak_groups(consumer_plan, &memo, &inventory)?;
        root_bound.peak_owner_term_ids = flatten_consumer_peak_groups(&root_bound.peak_owner_term_groups);
        root_bound.additional_peak_bytes = consumer_peak_groups_bytes(&root_bound.peak_owner_term_groups, &inventory)?;
    }
    for group in &mut root_bound.peak_owner_term_groups {
        let mut seen = HashSet::with_capacity(group.len());
        group.retain(|term_id| seen.insert(term_id.clone()));
    }
    let mut seen_owner_terms = HashSet::with_capacity(root_bound.owner_terms.len());
    root_bound.owner_terms.retain(|term| seen_owner_terms.insert(term.id.clone()));
    root_bound.owner_terms.retain(|term| !owner_terms.contains(term));
    match root_bound.allocation_state {
        AttemptReservationConsumerState::Passthrough => {
            if root_bound.additional_peak_bytes != 0
                || !root_bound.peak_owner_term_groups.is_empty()
                || !root_bound.peak_owner_term_ids.is_empty()
                || !root_bound.dominated_owner_term_ids.is_empty()
                || !root_bound.final_owned_term_ids.is_empty()
                || !root_bound.owner_terms.is_empty()
            {
                return Err(composed_schedule_error("consumer_passthrough_owner_terms_present", None));
            }
        },
        AttemptReservationConsumerState::Bounded => {
            if root_bound.peak_owner_term_groups.is_empty() || root_bound.peak_owner_term_ids.is_empty() {
                return Err(composed_schedule_error("consumer_bounded_owner_terms_missing", None));
            }
        },
    }
    if let Some(unconsumed) = producer_outputs.iter().find(|output| !used_occurrences.contains(&output.occurrence_id)) {
        return Err(composed_schedule_error(
            "consumer_producer_occurrence_unconsumed",
            Some(&unconsumed.occurrence_id),
        ));
    }
    root_bound.consumed_occurrences = producer_outputs.iter().map(|output| output.occurrence_id.clone()).collect();
    Ok(root_bound)
}

fn memoized_graph_consumer_peak_groups(
    plan: &QueryPlan,
    memo: &HashMap<NodeId, ComposedConsumerBranchBound>,
    owner_terms: &[AllocationOwnerTerm],
) -> Result<Vec<Vec<OwnerTermId>>> {
    fn visit(plan: &QueryPlan, node_id: NodeId, seen: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) -> Result<()> {
        if !seen.insert(node_id) {
            return Ok(());
        }
        let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
        for input in node.input_ids() {
            visit(plan, *input, seen, order)?;
        }
        order.push(node_id);
        Ok(())
    }

    let mut order = Vec::new();
    visit(plan, plan.root_id(), &mut HashSet::new(), &mut order)?;
    let scope = canonical_execution_scope_fingerprint(plan);
    let mut live = Vec::<OwnerTermId>::new();
    let mut live_set = HashSet::<OwnerTermId>::new();
    let mut groups = Vec::new();
    for node_id in order {
        let Some(bound) = memo.get(&node_id) else {
            continue;
        };
        let node_terms = owner_terms
            .iter()
            .filter(|term| {
                term.execution_scope_fingerprint() == scope
                    && match term.id.owner {
                        OwnerId::Graph(id)
                        | OwnerId::Aggregate(id)
                        | OwnerId::Source(id)
                        | OwnerId::Consumer(id)
                        | OwnerId::Sort(id)
                        | OwnerId::Limit(id) => id == node_id,
                        _ => false,
                    }
            })
            .collect::<Vec<_>>();
        if matches!(plan.node(node_id).map(|node| &node.kind), Some(PlanNode::GraphAlgorithm(_))) {
            let graph_input = match &plan
                .node(node_id)
                .ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?
                .kind
            {
                PlanNode::GraphAlgorithm(graph) => graph.input,
                _ => unreachable!(),
            };
            let credited = memo
                .get(&graph_input)
                .map(|input| input.final_owned_term_ids.iter().collect::<HashSet<_>>())
                .unwrap_or_default();
            let graph_base = live.iter().filter(|term| !credited.contains(term)).cloned().collect::<Vec<_>>();
            for phase_terms in [
                node_terms
                    .iter()
                    .filter(|term| term.is_construction_phase_term())
                    .map(|term| term.id.clone())
                    .collect::<Vec<_>>(),
                node_terms
                    .iter()
                    .filter(|term| term.is_execution_phase_term())
                    .map(|term| term.id.clone())
                    .collect::<Vec<_>>(),
            ] {
                if phase_terms.is_empty() {
                    return Err(composed_schedule_error("consumer_graph_phase_terms_unavailable", None));
                }
                let mut group = graph_base.clone();
                group.extend(phase_terms);
                groups.push(group);
            }
        } else if !node_terms.is_empty() {
            let mut group = live.clone();
            group.extend(node_terms.iter().map(|term| term.id.clone()));
            groups.push(group);
        }
        for term_id in &bound.final_owned_term_ids {
            if live_set.insert(term_id.clone()) {
                live.push(term_id.clone());
            }
        }
    }
    if groups.is_empty() {
        return Err(composed_schedule_error("consumer_bounded_owner_terms_missing", None));
    }
    Ok(groups)
}

fn composed_consumer_branch_bound(
    plan: &QueryPlan,
    node_id: NodeId,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    if let Some(cached) = memo.get(&node_id) {
        return Ok(ComposedConsumerBranchBound {
            rows: cached.rows,
            schema: Arc::clone(&cached.schema),
            final_owned_bytes: cached.final_owned_bytes,
            additional_peak_bytes: 0,
            owner_terms: Vec::new(),
            peak_owner_term_groups: Vec::new(),
            peak_owner_term_ids: Vec::new(),
            dominated_owner_term_ids: Vec::new(),
            final_owned_term_ids: cached.final_owned_term_ids.clone(),
            allocation_state: cached.allocation_state,
            consumed_occurrences: Vec::new(),
        });
    }
    let bound = composed_consumer_branch_bound_uncached(
        plan,
        node_id,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    memo.insert(node_id, bound.clone());
    Ok(bound)
}

fn composed_consumer_branch_bound_uncached(
    plan: &QueryPlan,
    node_id: NodeId,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    match &node.kind {
        PlanNode::Source(source) => {
            match source.as_ref() {
                SourceNode::GpuResident(source) => {
                    let Some(output) = outputs.get(source.input_name.as_str()) else {
                        return Err(composed_schedule_error(
                            "consumer_source_occurrence_missing",
                            Some(&source.input_name),
                        ));
                    };
                    if !used_occurrences.insert(source.input_name.clone()) {
                        return Err(composed_schedule_error(
                            "consumer_source_occurrence_reused",
                            Some(&source.input_name),
                        ));
                    }
                    if let Some(consumer_row_count_upper_bound) = source.row_count_upper_bound
                        && consumer_row_count_upper_bound != output.row_count_upper_bound
                    {
                        return Err(composed_schedule_error(
                            "consumer_source_row_bound_mismatch",
                            Some(&source.input_name),
                        )
                        .with_fact("producer_row_count_upper_bound", output.row_count_upper_bound)
                        .with_fact("consumer_row_count_upper_bound", consumer_row_count_upper_bound));
                    }
                    Ok(ComposedConsumerBranchBound {
                        rows: output.row_count_upper_bound,
                        schema: source.schema.clone(),
                        final_owned_bytes: 0,
                        additional_peak_bytes: 0,
                        owner_terms: Vec::new(),
                        peak_owner_term_groups: Vec::new(),
                        peak_owner_term_ids: Vec::new(),
                        dominated_owner_term_ids: Vec::new(),
                        final_owned_term_ids: Vec::new(),
                        allocation_state: AttemptReservationConsumerState::Passthrough,
                        consumed_occurrences: Vec::new(),
                    })
                },
                SourceNode::LocalParquet(source) => {
                    let source_term = owner_terms.iter().find(|term| {
                        matches!(term.id.owner, OwnerId::Source(id) if id == node_id)
                            && term.id.role == OwnerTermRole::SourceStage
                            && term.id.occurrence == OccurrenceId::PlanLocal(node_id)
                            && term.execution_scope_fingerprint() == canonical_execution_scope_fingerprint(plan)
                    });
                    let Some(source_term) = source_term else {
                        return Err(composed_schedule_error("consumer_local_source_owner_missing", None)
                            .with_fact("node_id", node_id));
                    };
                    let rows = source
                        .exact_selected_row_count()
                        .map(|rows| rows.min(source.scan_limit.unwrap_or(usize::MAX) as u64))
                        .map(|rows| rows.min(source.max_rows.unwrap_or(usize::MAX) as u64))
                        .ok_or_else(|| {
                            composed_schedule_error("consumer_local_source_rows_unavailable", None)
                                .with_fact("node_id", node_id)
                        })?;
                    // The source-stage owner is issued by the ordinary plan
                    // proof. Reuse its identity and bytes instead of minting a
                    // second consumer term for the same decoded frame.
                    Ok(ComposedConsumerBranchBound {
                        rows,
                        schema: source.output_schema().clone(),
                        final_owned_bytes: source_term.bytes.get(),
                        additional_peak_bytes: source_term.bytes.get(),
                        owner_terms: Vec::new(),
                        peak_owner_term_groups: vec![vec![source_term.id.clone()]],
                        peak_owner_term_ids: vec![source_term.id.clone()],
                        dominated_owner_term_ids: Vec::new(),
                        final_owned_term_ids: vec![source_term.id.clone()],
                        allocation_state: AttemptReservationConsumerState::Bounded,
                        consumed_occurrences: Vec::new(),
                    })
                },
                SourceNode::LiteralBatch(source) => {
                    let rows = u64::try_from(source.row_count())
                        .map_err(|_| composed_schedule_error("consumer_literal_source_rows_exceed_u64", None))?;
                    let columns = literal_batch_output_columns(source)?;
                    let bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns)
                        .map_err(|_| composed_schedule_error("consumer_literal_source_bytes_unavailable", None))?;
                    let source_term = AllocationOwnerTerm::new(
                        OwnerId::Source(node_id),
                        OwnerTermRole::SourceStage,
                        OccurrenceId::PlanLocal(node_id),
                        canonical_execution_scope_fingerprint(plan),
                        AllocationClass::ActivePhase,
                        bytes,
                        None,
                    )?;
                    let source_term_id = source_term.id.clone();
                    Ok(ComposedConsumerBranchBound {
                        rows,
                        schema: source.schema.clone(),
                        final_owned_bytes: bytes,
                        additional_peak_bytes: bytes,
                        owner_terms: vec![source_term],
                        peak_owner_term_groups: vec![vec![source_term_id.clone()]],
                        peak_owner_term_ids: vec![source_term_id.clone()],
                        dominated_owner_term_ids: Vec::new(),
                        final_owned_term_ids: vec![source_term_id],
                        allocation_state: AttemptReservationConsumerState::Bounded,
                        consumed_occurrences: Vec::new(),
                    })
                },
            }
        },
        PlanNode::Filter(filter) => {
            let input = composed_consumer_branch_bound(
                plan,
                filter.input,
                outputs,
                used_occurrences,
                owner_terms,
                column_evidence,
                memo,
            )?;
            let output_bytes = bounded_consumer_frame_bytes_with_evidence(
                input.rows,
                node.output_schema(),
                &column_evidence.node(node_id),
            )?;
            let predicate_mask_bytes = bounded_consumer_boolean_mask_bytes(input.rows)?;
            let output_term = consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerOutput,
                AllocationClass::ActivePhase,
                output_bytes,
                plan,
            )?;
            let predicate_term = consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerPredicateMask,
                AllocationClass::ActivePhase,
                predicate_mask_bytes,
                plan,
            )?;
            let output_term_id = output_term.id.clone();
            let predicate_term_id = predicate_term.id.clone();
            let peak_owner_term_groups =
                consumer_unary_peak_groups(&input, [output_term_id.clone(), predicate_term_id.clone()]);
            let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
            let mut consumer_owner_terms = input.owner_terms;
            consumer_owner_terms.push(output_term);
            consumer_owner_terms.push(predicate_term);
            let mut inventory = owner_terms.to_vec();
            inventory.extend(consumer_owner_terms.iter().cloned());
            let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: output_bytes,
                additional_peak_bytes,
                owner_terms: consumer_owner_terms,
                peak_owner_term_groups,
                peak_owner_term_ids,
                dominated_owner_term_ids: input.dominated_owner_term_ids,
                final_owned_term_ids: vec![output_term_id],
                allocation_state: AttemptReservationConsumerState::Bounded,
                consumed_occurrences: Vec::new(),
            })
        },
        PlanNode::Projection(projection) => {
            let input = composed_consumer_branch_bound(
                plan,
                projection.input,
                outputs,
                used_occurrences,
                owner_terms,
                column_evidence,
                memo,
            )?;
            let mut materializes_output = false;
            for output in &projection.outputs {
                let (column, representation_cast) = if let Some(column) = alias_wrapped_column(&output.expr) {
                    (column, false)
                } else if let Some((column, target_type)) = representation_cast_projection(&output.expr) {
                    if *target_type != *output.field.data_type() {
                        return Err(composed_schedule_error(
                            "consumer_projection_expression_workspace_unmodeled",
                            None,
                        )
                        .with_fact("expression", output.expr.clone()));
                    }
                    materializes_output = true;
                    (column, true)
                } else if projection.outputs.iter().any(|output| alias_wrapped_literal(&output.expr).is_some()) {
                    return composed_consumer_literal_projection_bound(
                        plan,
                        node_id,
                        node,
                        projection,
                        input,
                        owner_terms,
                    );
                } else {
                    return composed_consumer_expression_projection_bound(
                        plan,
                        node_id,
                        node,
                        projection,
                        input,
                        owner_terms,
                        column_evidence,
                    );
                };
                let source = input.schema.field_with_name(column.name()).map_err(|_| {
                    composed_schedule_error("consumer_projection_source_column_missing", None)
                        .with_fact("column", column.name())
                })?;
                if representation_cast {
                    if !representation_cast_preserves_identity(source.data_type(), output.field.data_type()) {
                        return Err(composed_schedule_error(
                            "consumer_projection_expression_workspace_unmodeled",
                            None,
                        )
                        .with_fact("expression", output.expr.clone()));
                    }
                } else if source.data_type() != output.field.data_type() {
                    return Err(composed_schedule_error("consumer_projection_output_type_mismatch", None)
                        .with_fact("column", column.name())
                        .with_fact("input_type", source.data_type())
                        .with_fact("output_type", output.field.data_type()));
                }
            }
            if materializes_output {
                let output_bytes = bounded_consumer_frame_bytes_with_evidence(
                    input.rows,
                    node.output_schema(),
                    &column_evidence.node(node_id),
                )?;
                let output_term = consumer_owner_term(
                    node_id,
                    OwnerTermRole::ConsumerOutput,
                    AllocationClass::ActivePhase,
                    output_bytes,
                    plan,
                )?;
                let output_term_id = output_term.id.clone();
                let peak_owner_term_groups = consumer_unary_peak_groups(&input, [output_term_id.clone()]);
                let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
                let mut consumer_owner_terms = input.owner_terms;
                consumer_owner_terms.push(output_term);
                let mut inventory = owner_terms.to_vec();
                inventory.extend(consumer_owner_terms.iter().cloned());
                let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
                return Ok(ComposedConsumerBranchBound {
                    rows: input.rows,
                    schema: node.output_schema().clone(),
                    final_owned_bytes: output_bytes,
                    additional_peak_bytes,
                    owner_terms: consumer_owner_terms,
                    peak_owner_term_groups,
                    peak_owner_term_ids,
                    dominated_owner_term_ids: input.dominated_owner_term_ids,
                    final_owned_term_ids: vec![output_term_id],
                    allocation_state: AttemptReservationConsumerState::Bounded,
                    consumed_occurrences: Vec::new(),
                });
            }
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: input.final_owned_bytes,
                additional_peak_bytes: input.additional_peak_bytes,
                owner_terms: input.owner_terms,
                peak_owner_term_groups: input.peak_owner_term_groups,
                peak_owner_term_ids: input.peak_owner_term_ids,
                dominated_owner_term_ids: input.dominated_owner_term_ids,
                final_owned_term_ids: input.final_owned_term_ids,
                allocation_state: input.allocation_state,
                consumed_occurrences: Vec::new(),
            })
        },
        PlanNode::Window(window) => composed_consumer_global_row_number_peak_bytes(
            plan,
            node_id,
            window,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::Aggregate(aggregate) => composed_consumer_aggregate_peak_bytes(
            plan,
            node_id,
            aggregate,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::Join(join) => composed_consumer_join_peak_bytes(
            plan,
            node_id,
            join,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::Union(union) => composed_consumer_union_peak_bytes(
            plan,
            node_id,
            union,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::Sort(sort) => composed_consumer_sort_peak_bytes(
            plan,
            node_id,
            sort,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::Limit(limit) => composed_consumer_limit_peak_bytes(
            plan,
            node_id,
            limit,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::EdgeNormalize(normalize) => composed_consumer_edge_normalize_peak_bytes(
            plan,
            node_id,
            normalize,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        PlanNode::GraphAlgorithm(graph) => composed_consumer_graph_peak_bytes(
            plan,
            node_id,
            graph,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        ),
        _ => Err(composed_schedule_error("consumer_branch_topology_unmodeled", None)),
    }
}

fn composed_consumer_edge_normalize_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    normalize: &crate::plan::EdgeNormalizeNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(
        plan,
        normalize.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let output_rows = if normalize.spec.add_reverse {
        input
            .rows
            .checked_mul(2)
            .ok_or_else(|| composed_schedule_error("consumer_edge_normalize_rows_overflow", None))?
    } else {
        input.rows
    };
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(output_rows, node.output_schema(), &column_evidence.node(node_id))?;
    let workspace_bytes = output_bytes
        .checked_mul(2)
        .ok_or_else(|| composed_schedule_error("consumer_edge_normalize_workspace_overflow", None))?;
    let output_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, output_bytes, plan)?;
    let workspace_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerWorkspace,
        AllocationClass::ActivePhase,
        workspace_bytes,
        plan,
    )?;
    let output_term_id = output_term.id.clone();
    let workspace_term_id = workspace_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [output_term_id.clone(), workspace_term_id]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.extend([output_term, workspace_term]);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_consumer_global_row_number_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    window: &crate::plan::WindowNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    if !window.spec.partition_by.is_empty()
        || window.spec.functions.len() != 1
        || !matches!(
            window.spec.functions.first(),
            Some(crate::plan::WindowFunction::RowNumber { output_type: DataType::Int64, .. })
        )
    {
        return Err(composed_schedule_error("consumer_window_topology_unmodeled", None));
    }
    let input = composed_consumer_branch_bound(
        plan,
        window.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(input.rows, node.output_schema(), &column_evidence.node(node_id))?;
    let output_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, output_bytes, plan)?;
    let output_term_id = output_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [output_term_id.clone()]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.push(output_term);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_consumer_graph_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(
        plan,
        graph.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let scope = canonical_execution_scope_fingerprint(plan);
    let graph_terms = owner_terms
        .iter()
        .filter(|term| term.id.owner == OwnerId::Graph(node_id) && term.execution_scope_fingerprint() == scope)
        .collect::<Vec<_>>();
    let retained =
        graph_terms.iter().filter(|term| term.id.role == OwnerTermRole::RetainedOutput).copied().collect::<Vec<_>>();
    let [retained] = retained.as_slice() else {
        return Err(composed_schedule_error("consumer_graph_owner_terms_unavailable", None));
    };
    let construction = graph_terms
        .iter()
        .filter(|term| term.is_construction_phase_term())
        .map(|term| term.id.clone())
        .collect::<Vec<_>>();
    let execution = graph_terms
        .iter()
        .filter(|term| term.is_execution_phase_term())
        .map(|term| term.id.clone())
        .collect::<Vec<_>>();
    if construction.is_empty() || execution.is_empty() {
        return Err(composed_schedule_error("consumer_graph_phase_terms_unavailable", None));
    }
    let mut peak_owner_term_groups = input.peak_owner_term_groups.clone();
    for graph_group in [construction, execution] {
        let mut group = input.final_owned_term_ids.clone();
        group.extend(graph_group);
        peak_owner_term_groups.push(group);
    }
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(input.owner_terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    let rows = retained
        .retained_row_count_upper_bound()
        .ok_or_else(|| composed_schedule_error("consumer_graph_output_rows_unavailable", None))?;
    Ok(ComposedConsumerBranchBound {
        rows,
        schema: plan
            .node(node_id)
            .ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?
            .output_schema()
            .clone(),
        final_owned_bytes: retained.bytes(),
        additional_peak_bytes,
        owner_terms: input.owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![retained.id.clone()],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn alias_wrapped_literal(expr: &Expr) -> Option<&LiteralValue> {
    match expr {
        Expr::Literal(literal) => Some(literal),
        Expr::Alias { expr, .. } => alias_wrapped_literal(expr),
        _ => None,
    }
}

fn composed_consumer_literal_projection_bound(
    plan: &QueryPlan,
    node_id: NodeId,
    node: &crate::plan::QueryNode,
    projection: &crate::plan::ProjectionNode,
    input: ComposedConsumerBranchBound,
    owner_terms: &[AllocationOwnerTerm],
) -> Result<ComposedConsumerBranchBound> {
    let mut selected_columns = HashSet::with_capacity(projection.outputs.len());
    let mut retains_input = false;
    let mut literal_fields = Vec::new();
    for output in &projection.outputs {
        if let Some(column) = alias_wrapped_column(&output.expr) {
            if !selected_columns.insert(column.name()) {
                return Err(composed_schedule_error("consumer_projection_source_column_reused", None)
                    .with_fact("column", column.name()));
            }
            let source = input.schema.field_with_name(column.name()).map_err(|_| {
                composed_schedule_error("consumer_projection_source_column_missing", None)
                    .with_fact("column", column.name())
            })?;
            if source.data_type() != output.field.data_type() {
                return Err(composed_schedule_error("consumer_projection_output_type_mismatch", None)
                    .with_fact("column", column.name())
                    .with_fact("input_type", source.data_type())
                    .with_fact("output_type", output.field.data_type()));
            }
            retains_input = true;
            continue;
        }
        let Some(literal) = alias_wrapped_literal(&output.expr) else {
            return Err(composed_schedule_error("consumer_projection_expression_workspace_unmodeled", None)
                .with_fact("expression", output.expr.clone()));
        };
        if literal.data_type() != *output.field.data_type()
            || matches!(literal, LiteralValue::Null(_)) && !output.field.is_nullable()
            || fixed_width_consumer_output_column(output.field.data_type()).is_none()
        {
            return Err(composed_schedule_error("consumer_projection_literal_output_unmodeled", None)
                .with_fact("expression", output.expr.clone())
                .with_fact("output_type", output.field.data_type()));
        }
        literal_fields.push(output.field.clone());
    }
    if literal_fields.is_empty() {
        return Err(composed_schedule_error("consumer_projection_expression_workspace_unmodeled", None));
    }
    let literal_schema = Arc::new(Schema::new(literal_fields));
    let literal_bytes =
        bounded_consumer_frame_bytes_with_evidence(input.rows, &literal_schema, &PlanColumnEvidence::default())?;
    let literal_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, literal_bytes, plan)?;
    let literal_term_id = literal_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [literal_term_id.clone()]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let final_owned_bytes = if retains_input {
        checked_consumer_add(
            input.final_owned_bytes,
            literal_bytes,
            "consumer_projection_literal_final_owned_overflow",
        )?
    } else {
        literal_bytes
    };
    let final_owned_term_ids = if retains_input {
        input.final_owned_term_ids.iter().cloned().chain(std::iter::once(literal_term_id.clone())).collect()
    } else {
        vec![literal_term_id.clone()]
    };
    let mut consumer_owner_terms = input.owner_terms;
    consumer_owner_terms.push(literal_term);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(consumer_owner_terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: node.output_schema().clone(),
        final_owned_bytes,
        additional_peak_bytes,
        owner_terms: consumer_owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids,
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_consumer_expression_projection_bound(
    plan: &QueryPlan,
    node_id: NodeId,
    node: &crate::plan::QueryNode,
    projection: &crate::plan::ProjectionNode,
    input: ComposedConsumerBranchBound,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
) -> Result<ComposedConsumerBranchBound> {
    let rows = usize::try_from(input.rows)
        .map_err(|_| composed_schedule_error("consumer_projection_rows_exceed_usize", None))?;
    let input_column_evidence = column_evidence.node(projection.input);
    let input_columns = input
        .schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
                composed_schedule_error("consumer_projection_input_type_unmodeled", None)
                    .with_fact("column", field.name())
                    .with_fact("data_type", field.data_type())
            })?;
            let field_schema = Arc::new(Schema::new(vec![field.as_ref().clone()]));
            let mut field_evidence = PlanColumnEvidence::default();
            if let Some(bound) = input_column_evidence.string_payload_bound(field.name()) {
                field_evidence.string_payload_bounds.insert(field.name().clone(), bound.clone());
            }
            let retained_bytes =
                bounded_consumer_frame_bytes_with_evidence(input.rows, &field_schema, &field_evidence)?;
            let retained_bytes = usize::try_from(retained_bytes)
                .map_err(|_| composed_schedule_error("consumer_projection_input_bytes_exceed_usize", None))?;
            Ok(cudf::ops::expr::DataFrameExpressionInputColumn::new(
                field.name(),
                dtype,
                field.is_nullable(),
                retained_bytes,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = projection
        .outputs
        .iter()
        .map(|output| {
            crate::exec::lower_projection_output_expr_for_admission(output)
                .map(|expr| (output.field.name().to_string(), expr))
        })
        .collect::<Result<Vec<_>>>()?;
    let preflight =
        cudf::ops::expr::dataframe_expression_allocation_preflight_for_metadata(rows, &input_columns, &outputs, false)
            .map_err(|error| {
                composed_schedule_error("consumer_projection_expression_owner_preflight_unavailable", None)
                    .with_fact("cudf_error", error.to_string())
                    .with_fact("required_contract", cudf::ops::expr::DATAFRAME_EXPRESSION_ALLOCATION_PREFLIGHT_CONTRACT)
            })?;
    let workspace_bytes = preflight.temporary_workspace_peak_bytes();
    let workspace_bytes = u64::try_from(workspace_bytes)
        .map_err(|_| composed_schedule_error("consumer_projection_expression_workspace_exceeds_u64", None))?;
    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(input.rows, node.output_schema(), &column_evidence.node(node_id))?;
    let output_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, output_bytes, plan)?;
    let output_term_id = output_term.id.clone();
    let workspace_term = (workspace_bytes > 0)
        .then(|| {
            consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerProjectionExpressionWorkspace,
                AllocationClass::ActivePhase,
                workspace_bytes,
                plan,
            )
        })
        .transpose()?;
    let current_terms = std::iter::once(output_term_id.clone())
        .chain(workspace_term.iter().map(|term| term.id.clone()))
        .collect::<Vec<_>>();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, current_terms);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut consumer_owner_terms = input.owner_terms;
    consumer_owner_terms.push(output_term);
    if let Some(workspace_term) = workspace_term {
        consumer_owner_terms.push(workspace_term);
    }
    let mut inventory = owner_terms.to_vec();
    inventory.extend(consumer_owner_terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: consumer_owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn consumer_peak_group_bytes(group: &[OwnerTermId], owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
    let mut seen = HashSet::with_capacity(group.len());
    group.iter().try_fold(0_u64, |total, term_id| {
        if !seen.insert(term_id) {
            return Ok(total);
        }
        let term = owner_terms.iter().find(|term| term.id() == term_id).ok_or_else(|| {
            composed_schedule_error("consumer_peak_owner_term_missing", None)
                .with_fact("owner_term_role", term_id.role_label())
                .with_fact("owner_term_occurrence", term_id.occurrence_label())
        })?;
        total.checked_add(term.bytes()).ok_or_else(|| composed_schedule_error("consumer_peak_owner_sum_overflow", None))
    })
}

fn consumer_peak_groups_bytes(groups: &[Vec<OwnerTermId>], owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
    groups.iter().try_fold(0_u64, |peak, group| Ok(peak.max(consumer_peak_group_bytes(group, owner_terms)?)))
}

fn flatten_consumer_peak_groups(groups: &[Vec<OwnerTermId>]) -> Vec<OwnerTermId> {
    let mut seen = HashSet::new();
    groups.iter().flatten().filter(|term_id| seen.insert((*term_id).clone())).cloned().collect()
}

fn consumer_unary_peak_groups(
    input: &ComposedConsumerBranchBound,
    current_term_ids: impl IntoIterator<Item = OwnerTermId>,
) -> Vec<Vec<OwnerTermId>> {
    let mut groups = input.peak_owner_term_groups.clone();
    let mut operator_group = input.final_owned_term_ids.clone();
    operator_group.extend(current_term_ids);
    groups.push(operator_group);
    groups
}

fn representation_cast_projection(expr: &Expr) -> Option<(&crate::expr::ColumnRef, &DataType)> {
    match expr {
        Expr::Alias { expr, .. } => representation_cast_projection(expr),
        Expr::Cast { expr, target_type } => alias_wrapped_column(expr).map(|column| (column, target_type)),
        _ => None,
    }
}

fn composed_consumer_union_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    union: &crate::plan::UnionNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let mut total_rows = 0_u64;
    let mut branch_terms = Vec::new();
    let mut peak_owner_term_groups = Vec::new();
    let mut dominated_owner_term_ids = Vec::new();
    let mut retained_prior_term_ids = Vec::new();
    for input in &union.inputs {
        let branch = composed_consumer_branch_bound(
            plan,
            *input,
            outputs,
            used_occurrences,
            owner_terms,
            column_evidence,
            memo,
        )?;
        total_rows = checked_consumer_add(total_rows, branch.rows, "consumer_union_rows_overflow")?;
        for group in &branch.peak_owner_term_groups {
            let mut peak_group = retained_prior_term_ids.clone();
            peak_group.extend(group.iter().cloned());
            peak_owner_term_groups.push(peak_group);
        }
        retained_prior_term_ids.extend(branch.final_owned_term_ids.iter().cloned());
        dominated_owner_term_ids.extend(branch.dominated_owner_term_ids);
        branch_terms.extend(branch.owner_terms);
    }
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let union_output =
        bounded_consumer_frame_bytes_with_evidence(total_rows, node.output_schema(), &column_evidence.node(node_id))?;
    let union_output_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, union_output, plan)?;
    branch_terms.push(union_output_term);
    let union_output_term_id = branch_terms.last().expect("union output term inserted").id.clone();
    let mut final_peak_group = retained_prior_term_ids;
    final_peak_group.push(union_output_term_id.clone());
    peak_owner_term_groups.push(final_peak_group);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let final_owned_term_ids = vec![union_output_term_id];
    let mut inventory = owner_terms.to_vec();
    inventory.extend(branch_terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: total_rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: union_output,
        additional_peak_bytes,
        owner_terms: branch_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids,
        final_owned_term_ids,
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_consumer_aggregate_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    aggregate: &AggregateNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    if !aggregate_owner_state_is_supported(aggregate) {
        return Err(composed_schedule_error("consumer_aggregate_state_unmodeled", None));
    }

    let input = composed_consumer_branch_bound(
        plan,
        aggregate.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_schema = node.output_schema();
    let output_rows = if aggregate.spec.group_exprs.is_empty() { 1 } else { input.rows };
    let partial_shape = if aggregate.spec.group_exprs.is_empty() {
        crate::exec::PartialAggregateShape::Global
    } else {
        crate::exec::PartialAggregateShape::Grouped { input_rows: input.rows }
    };
    let node_evidence = column_evidence.node(node_id);
    let mut aggregate_terms = aggregate_owner_terms_for_shapes(
        plan,
        node_id,
        aggregate,
        output_schema,
        [partial_shape],
        output_rows,
        Some(&node_evidence),
    )
    .map_err(|error| {
        composed_schedule_error("consumer_aggregate_owner_terms_unavailable", None)
            .with_fact("aggregate_owner_reason", error.fact_value("reason").unwrap_or("unknown"))
    })?;
    let output_term = aggregate_terms.pop().expect("aggregate owner helper emits its output term");
    let partial_term = aggregate_terms.pop().expect("aggregate owner helper emits its partial-state term");
    let output_bytes = output_term.bytes();
    let partial_term_id = partial_term.id.clone();
    let output_term_id = output_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [partial_term_id.clone(), output_term_id.clone()]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.extend([partial_term, output_term]);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: output_schema.clone(),
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn is_attested_sort_scope(plan: &QueryPlan) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(gpu) => is_attested_graph_result_bounded_consumer_plan(plan, &gpu.input_name),
            _ => false,
        },
        _ => false,
    })
}

fn composed_consumer_sort_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    sort: &crate::plan::SortNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(
        plan,
        sort.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_schema = node.output_schema().clone();
    if is_attested_sort_scope(plan) {
        let scope = canonical_execution_scope_fingerprint(plan);
        let sort_terms = [
            OwnerTermRole::SortKeyInput,
            OwnerTermRole::SortKeyOutput,
            OwnerTermRole::SortSequence,
            OwnerTermRole::SortTemporaryWorkspace,
            OwnerTermRole::SortRetainedOrder,
            OwnerTermRole::SortOutputFrame,
            OwnerTermRole::SortOutputNullMask,
        ]
        .into_iter()
        .map(|role| OwnerTermId {
            owner: OwnerId::Sort(node_id),
            role,
            occurrence: OccurrenceId::PlanLocal(node_id),
            execution_scope_fingerprint: scope.clone(),
        })
        .collect::<Vec<_>>();
        let final_owned_term_ids = sort_terms
            .iter()
            .filter(|term_id| {
                matches!(term_id.role, OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask)
            })
            .cloned()
            .collect();
        let peak_owner_term_groups = consumer_unary_peak_groups(&input, sort_terms.clone());
        let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
        return Ok(ComposedConsumerBranchBound {
            rows: input.rows,
            schema: output_schema,
            final_owned_bytes: 0,
            additional_peak_bytes: 0,
            owner_terms: input.owner_terms,
            peak_owner_term_groups,
            peak_owner_term_ids,
            dominated_owner_term_ids: input.dominated_owner_term_ids,
            final_owned_term_ids,
            allocation_state: AttemptReservationConsumerState::Bounded,
            consumed_occurrences: Vec::new(),
        });
    }

    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(input.rows, &output_schema, &column_evidence.node(node_id))?;
    let workspace_bytes = output_bytes.saturating_mul(2);
    let output_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, output_bytes, plan)?;
    let workspace_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerWorkspace,
        AllocationClass::ActivePhase,
        workspace_bytes,
        plan,
    )?;
    let output_term_id = output_term.id.clone();
    let workspace_term_id = workspace_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [output_term_id.clone(), workspace_term_id]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.extend([output_term, workspace_term]);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: output_schema,
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_consumer_limit_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    limit: &crate::plan::LimitNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(
        plan,
        limit.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_schema = node.output_schema().clone();
    let fetch_usize =
        limit.spec.fetch.ok_or_else(|| composed_schedule_error("consumer_limit_fetch_unbounded", None))?;
    let skip =
        u64::try_from(limit.spec.skip).map_err(|_| composed_schedule_error("consumer_limit_skip_exceeds_u64", None))?;
    let fetch =
        u64::try_from(fetch_usize).map_err(|_| composed_schedule_error("consumer_limit_fetch_exceeds_u64", None))?;
    limit
        .spec
        .skip
        .checked_add(fetch_usize)
        .ok_or_else(|| composed_schedule_error("consumer_limit_skip_fetch_overflow", None))?;
    let output_rows = input.rows.saturating_sub(skip).min(fetch);

    if is_attested_sort_scope(plan) {
        let scope = canonical_execution_scope_fingerprint(plan);
        let limit_terms = [
            OwnerTermRole::LimitGatherMap,
            OwnerTermRole::LimitNativeWorkspace,
            OwnerTermRole::LimitOutputFrame,
            OwnerTermRole::LimitOutputNullMask,
        ]
        .into_iter()
        .map(|role| OwnerTermId {
            owner: OwnerId::Limit(node_id),
            role,
            occurrence: OccurrenceId::PlanLocal(node_id),
            execution_scope_fingerprint: scope.clone(),
        })
        .collect::<Vec<_>>();
        let final_owned_term_ids = limit_terms
            .iter()
            .filter(|term_id| {
                matches!(term_id.role, OwnerTermRole::LimitOutputFrame | OwnerTermRole::LimitOutputNullMask)
            })
            .cloned()
            .collect::<Vec<_>>();
        let peak_owner_term_groups = consumer_unary_peak_groups(&input, limit_terms.clone());
        let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
        return Ok(ComposedConsumerBranchBound {
            rows: output_rows,
            schema: output_schema,
            final_owned_bytes: 0,
            additional_peak_bytes: 0,
            owner_terms: input.owner_terms,
            peak_owner_term_groups,
            peak_owner_term_ids,
            dominated_owner_term_ids: input.dominated_owner_term_ids,
            final_owned_term_ids,
            allocation_state: AttemptReservationConsumerState::Bounded,
            consumed_occurrences: Vec::new(),
        });
    }

    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(output_rows, &output_schema, &column_evidence.node(node_id))?;
    let limit_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, output_bytes, plan)?;
    let limit_term_id = limit_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [limit_term_id.clone()]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.push(limit_term);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: output_schema,
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![limit_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_consumer_join_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    join: &JoinNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left)
        || join.spec.residual_predicate.is_some()
        || join.spec.keys.is_empty()
    {
        return Err(composed_schedule_error("consumer_join_topology_unmodeled", None));
    }
    let left = composed_consumer_branch_bound(
        plan,
        join.inputs[0],
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let right = composed_consumer_branch_bound(
        plan,
        join.inputs[1],
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let (build, probe, build_keys, build_node_id, probe_node_id) = match join.spec.build_side {
        JoinSide::Left => (
            &left,
            &right,
            join.spec.keys.iter().map(|key| &key.left).collect::<Vec<_>>(),
            join.inputs[0],
            join.inputs[1],
        ),
        JoinSide::Right => (
            &right,
            &left,
            join.spec.keys.iter().map(|key| &key.right).collect::<Vec<_>>(),
            join.inputs[1],
            join.inputs[0],
        ),
    };
    let build_column_evidence = column_evidence.node(build_node_id);
    let probe_column_evidence = column_evidence.node(probe_node_id);
    let key_metadata = build_keys
        .iter()
        .map(|key| hash_join_key_metadata(&build.schema, &build_column_evidence, key.name()))
        .collect::<Result<Vec<_>>>()?;
    for key in &join.spec.keys {
        let probe_key = match join.spec.build_side {
            JoinSide::Left => &key.right,
            JoinSide::Right => &key.left,
        };
        hash_join_key_metadata(&probe.schema, &probe_column_evidence, probe_key.name())?;
    }
    let output_rows = match join.spec.join_type {
        JoinType::Inner if join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) => left.rows,
        JoinType::Inner if join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Right) => right.rows,
        JoinType::Inner => left
            .rows
            .checked_mul(right.rows)
            .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?,
        JoinType::Left if join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) => left.rows,
        JoinType::Left => left
            .rows
            .checked_mul(right.rows.max(1))
            .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?,
        _ => return Err(composed_schedule_error("consumer_join_topology_unmodeled", None)),
    };
    let output_rows_usize = usize::try_from(output_rows)
        .map_err(|_| composed_schedule_error("consumer_join_output_rows_exceed_usize", None))?;
    let preprocessing_mode = cudf::ops::join::HashJoinPreprocessingMode::for_keys(&key_metadata);
    let hash_preflight = cudf::ops::join::hash_join_allocation_preflight_for_metadata(
        &key_metadata,
        build.rows,
        cudf::DataFrameHashJoinOptions::new().load_factor_value(),
        preprocessing_mode,
    )
    .map_err(|error| {
        composed_schedule_error("consumer_join_hash_preflight_unavailable", None)
            .with_fact("cudf_error", error.to_string())
            .with_fact("required_contract", cudf::ops::join::HASH_JOIN_ALLOCATION_PREFLIGHT_CONTRACT)
    })?;
    let hash_table_bytes = u64::try_from(hash_preflight.active_phase_peak_bytes())
        .map_err(|_| composed_schedule_error("consumer_join_hash_preflight_exceeds_u64", None))?;
    let pair_indices_bytes = cudf::DataFrameHashJoinPairIndices::reservation_size_bytes_for_len(output_rows_usize)
        .map_err(|_| {
            composed_join_preflight_error(
                "consumer_join_pair_indices_preflight_unavailable",
                node_id,
                join,
                left.rows,
                right.rows,
                build.rows,
                probe.rows,
                output_rows,
            )
        })?;
    let pair_indices_bytes = u64::try_from(pair_indices_bytes)
        .map_err(|_| composed_schedule_error("consumer_join_pair_indices_exceed_u64", None))?;
    let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(output_rows, node.output_schema(), &column_evidence.node(node_id))?;
    let hash_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerHashTable,
        AllocationClass::ActivePhase,
        hash_table_bytes,
        plan,
    )?;
    let pair_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerPairIndices,
        AllocationClass::ActivePhase,
        pair_indices_bytes,
        plan,
    )?;
    let output_term =
        consumer_owner_term(node_id, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, output_bytes, plan)?;
    let current_terms = vec![hash_term, pair_term, output_term];
    let final_owned_term_id = current_terms.last().expect("join output term inserted").id.clone();
    let mut peak_owner_term_groups = left.peak_owner_term_groups.clone();
    for group in &right.peak_owner_term_groups {
        let mut peak_group = left.final_owned_term_ids.clone();
        peak_group.extend(group.iter().cloned());
        peak_owner_term_groups.push(peak_group);
    }
    let mut join_peak_group = left.final_owned_term_ids.clone();
    join_peak_group.extend(right.final_owned_term_ids.iter().cloned());
    join_peak_group.extend(current_terms.iter().map(|term| term.id.clone()));
    peak_owner_term_groups.push(join_peak_group);
    let mut consumer_owner_terms = left.owner_terms;
    consumer_owner_terms.extend(right.owner_terms);
    consumer_owner_terms.extend(current_terms);
    let mut inventory = owner_terms.to_vec();
    inventory.extend(consumer_owner_terms.iter().cloned());
    let additional_peak_bytes = consumer_peak_groups_bytes(&peak_owner_term_groups, &inventory)?;
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut dominated_owner_term_ids = left.dominated_owner_term_ids;
    dominated_owner_term_ids.extend(right.dominated_owner_term_ids);
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        additional_peak_bytes,
        owner_terms: consumer_owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids,
        final_owned_term_ids: vec![final_owned_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_join_preflight_error(
    reason: &'static str,
    node_id: NodeId,
    join: &JoinNode,
    left_rows: u64,
    right_rows: u64,
    build_rows: u64,
    probe_rows: u64,
    output_rows: u64,
) -> QueryError {
    let join_keys = join
        .spec
        .keys
        .iter()
        .map(|key| format!("{}={}", key.left.name(), key.right.name()))
        .collect::<Vec<_>>()
        .join(",");
    composed_schedule_error(reason, None)
        .with_fact("node_id", node_id)
        .with_fact("join_type", format!("{:?}", join.spec.join_type))
        .with_fact("left_rows", left_rows)
        .with_fact("right_rows", right_rows)
        .with_fact("output_rows", output_rows)
        .with_fact("build_rows", build_rows)
        .with_fact("probe_rows", probe_rows)
        .with_fact("join_keys", join_keys)
}

fn consumer_owner_term(
    node_id: NodeId,
    role: OwnerTermRole,
    class: AllocationClass,
    bytes: u64,
    plan: &QueryPlan,
) -> Result<AllocationOwnerTerm> {
    AllocationOwnerTerm::new(
        OwnerId::Consumer(node_id),
        role,
        OccurrenceId::PlanLocal(node_id),
        canonical_execution_scope_fingerprint(plan),
        class,
        bytes,
        None,
    )
}

fn hash_join_key_metadata(
    schema: &SchemaRef,
    evidence: &PlanColumnEvidence,
    field_name: &str,
) -> Result<cudf::ops::join::HashJoinKeyMetadata> {
    let field = schema
        .field_with_name(field_name)
        .map_err(|_| composed_schedule_error("consumer_join_key_missing", None).with_fact("column", field_name))?;
    if let Some(type_id) = arrow_fixed_width_cudf_type_id(field.data_type()) {
        return Ok(cudf::ops::join::HashJoinKeyMetadata::fixed_width(type_id, field.is_nullable()));
    }
    if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
        let bound =
            evidence.string_payload_bound(field_name).filter(|bound| !bound.lineages.is_empty()).ok_or_else(|| {
                composed_schedule_error("consumer_join_variable_width_key", None).with_fact("column", field_name)
            })?;
        let payload_upper_bound_bytes = usize::try_from(bound.payload_upper_bound_bytes).map_err(|_| {
            composed_schedule_error("consumer_join_string_payload_exceeds_usize", None).with_fact("column", field_name)
        })?;
        return Ok(cudf::ops::join::HashJoinKeyMetadata::string(
            payload_upper_bound_bytes,
            cudf::ops::join::HashJoinStringOffsetWidth::I32,
            field.is_nullable(),
        ));
    }
    Err(composed_schedule_error("consumer_join_variable_width_key", None).with_fact("column", field_name))
}

fn arrow_fixed_width_cudf_type_id(data_type: &DataType) -> Option<cudf::TypeId> {
    Some(match data_type {
        DataType::Boolean => cudf::TypeId::Bool8,
        DataType::Int8 => cudf::TypeId::Int8,
        DataType::Int16 => cudf::TypeId::Int16,
        DataType::Int32 => cudf::TypeId::Int32,
        DataType::Int64 => cudf::TypeId::Int64,
        DataType::UInt8 => cudf::TypeId::UInt8,
        DataType::UInt16 => cudf::TypeId::UInt16,
        DataType::UInt32 => cudf::TypeId::UInt32,
        DataType::UInt64 => cudf::TypeId::UInt64,
        DataType::Float32 => cudf::TypeId::Float32,
        DataType::Float64 => cudf::TypeId::Float64,
        DataType::Date32 => cudf::TypeId::TimestampDays,
        DataType::Date64 => cudf::TypeId::TimestampMilliseconds,
        DataType::Time32(TimeUnit::Second) => cudf::TypeId::DurationSeconds,
        DataType::Time32(TimeUnit::Millisecond) => cudf::TypeId::DurationMilliseconds,
        DataType::Time64(TimeUnit::Microsecond) => cudf::TypeId::DurationMicroseconds,
        DataType::Time64(TimeUnit::Nanosecond) => cudf::TypeId::DurationNanoseconds,
        DataType::Timestamp(TimeUnit::Second, None) => cudf::TypeId::TimestampSeconds,
        DataType::Timestamp(TimeUnit::Millisecond, None) => cudf::TypeId::TimestampMilliseconds,
        DataType::Timestamp(TimeUnit::Microsecond, None) => cudf::TypeId::TimestampMicroseconds,
        DataType::Timestamp(TimeUnit::Nanosecond, None) => cudf::TypeId::TimestampNanoseconds,
        DataType::Duration(TimeUnit::Second) => cudf::TypeId::DurationSeconds,
        DataType::Duration(TimeUnit::Millisecond) => cudf::TypeId::DurationMilliseconds,
        DataType::Duration(TimeUnit::Microsecond) => cudf::TypeId::DurationMicroseconds,
        DataType::Duration(TimeUnit::Nanosecond) => cudf::TypeId::DurationNanoseconds,
        DataType::Decimal128(_, _) => cudf::TypeId::Decimal128,
        _ => return None,
    })
}

fn bounded_consumer_boolean_mask_bytes(rows: u64) -> Result<u64> {
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "predicate",
        DataType::Boolean,
        false,
    )]));
    bounded_consumer_frame_bytes(rows, &schema)
}

fn bounded_consumer_frame_bytes(rows: u64, schema: &SchemaRef) -> Result<u64> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| fixed_width_consumer_output_column(field.data_type()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| composed_schedule_error("consumer_variable_width_frame_unmodeled", None))?;
    let bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns)
        .map_err(|_| composed_schedule_error("consumer_frame_upper_bound_unavailable", None))?;
    // The legacy join-size helper intentionally saturates because it also
    // serves best-effort diagnostics. Admission cannot use a saturated value
    // as a finite allocation contract, so fail closed instead.
    if bytes == u64::MAX {
        return Err(composed_schedule_error("consumer_frame_bytes_overflow", None));
    }
    Ok(bytes)
}

fn bounded_consumer_frame_bytes_with_evidence(
    rows: u64,
    schema: &SchemaRef,
    evidence: &PlanColumnEvidence,
) -> Result<u64> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| consumer_output_column_with_evidence(field, evidence))
        .collect::<Result<Vec<_>>>()?;
    let bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns)
        .map_err(|_| composed_schedule_error("consumer_frame_upper_bound_unavailable", None))?;
    if bytes == u64::MAX {
        return Err(composed_schedule_error("consumer_frame_bytes_overflow", None));
    }
    Ok(bytes)
}

fn consumer_output_column_with_optional_evidence(
    field: &arrow::datatypes::Field,
    evidence: Option<&PlanColumnEvidence>,
) -> Result<crate::exec::JoinOutputColumnUpperBoundInput> {
    match evidence {
        Some(evidence) => consumer_output_column_with_evidence(field, evidence),
        None => fixed_width_consumer_output_column(field.data_type())
            .ok_or_else(|| owner_term_error("aggregate_output_type_unmodeled", None, None)),
    }
}

fn consumer_output_column_with_evidence(
    field: &arrow::datatypes::Field,
    evidence: &PlanColumnEvidence,
) -> Result<crate::exec::JoinOutputColumnUpperBoundInput> {
    if let Some(column) = fixed_width_consumer_output_column(field.data_type()) {
        return Ok(column);
    }
    if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
        let bound = evidence.string_payload_bound(field.name()).ok_or_else(|| {
            composed_schedule_error("consumer_string_payload_evidence_missing", None).with_fact("column", field.name())
        })?;
        if bound.lineages.is_empty() {
            return Err(composed_schedule_error("consumer_string_payload_lineage_missing", None)
                .with_fact("column", field.name()));
        }
        return Ok(crate::exec::JoinOutputColumnUpperBoundInput {
            data_type: field.data_type().clone(),
            input_max_string_length_bytes: Some(bound.payload_upper_bound_bytes),
            input_total_string_bytes: None,
        });
    }
    Err(composed_schedule_error("consumer_variable_width_frame_unmodeled", None)
        .with_fact("column", field.name())
        .with_fact("data_type", field.data_type()))
}

fn cugraph_side_input_frame_upper_bound_bytes(plan: &QueryPlan) -> Result<u64> {
    let rows = query_plan_hard_row_upper_bound(plan).ok_or_else(|| {
        composed_schedule_error("side_input_row_bound_unavailable", None)
            .with_fact("phase", "side_input_frame")
            .with_fact("row_bound_source", "native_query_plan")
    })?;
    if plan.output_schema().fields().is_empty() {
        return Err(composed_schedule_error("side_input_schema_empty", None).with_fact("phase", "side_input_frame"));
    }
    let columns = cugraph_side_input_columns(plan)?;
    let bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns).map_err(|error| {
        composed_schedule_error("side_input_frame_bound_unavailable", None)
            .with_fact("phase", "side_input_frame")
            .with_fact("sizing_error", error.code())
    })?;
    if bytes == u64::MAX {
        return Err(composed_schedule_error("side_input_frame_bytes_overflow", None)
            .with_fact("phase", "side_input_frame")
            .with_fact("row_bound", rows));
    }
    Ok(bytes)
}

fn cugraph_side_input_columns(plan: &QueryPlan) -> Result<Vec<crate::exec::JoinOutputColumnUpperBoundInput>> {
    let literal = match &plan.root_node().kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LiteralBatch(source) => Some(source),
            _ => None,
        },
        _ => None,
    };
    plan.output_schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if let Some(column) = fixed_width_consumer_output_column(field.data_type()) {
                return Ok(column);
            }
            let Some(literal) = literal else {
                return Err(composed_schedule_error("side_input_schema_unmodeled", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
            };
            let stats = literal.batches.iter().try_fold((0_u64, 0_u64), |acc, batch| {
                let array = batch.column(index);
                let mut lengths =
                    (0..array.len()).filter(|row| !array.is_null(*row)).map(|row| match field.data_type() {
                        DataType::Utf8 => {
                            array.as_any().downcast_ref::<StringArray>().map(|array| array.value(row).len())
                        },
                        DataType::LargeUtf8 => {
                            array.as_any().downcast_ref::<LargeStringArray>().map(|array| array.value(row).len())
                        },
                        DataType::Utf8View => {
                            array.as_any().downcast_ref::<StringViewArray>().map(|array| array.value(row).len())
                        },
                        _ => None,
                    });
                lengths.try_fold(acc, |(max_len, total_bytes), length| {
                    let length = u64::try_from(length?).ok()?;
                    Some((max_len.max(length), total_bytes.checked_add(length)?))
                })
            });
            let Some((max_len, total_bytes)) = stats else {
                return Err(composed_schedule_error("side_input_literal_string_stats_unavailable", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
            };
            Ok(crate::exec::JoinOutputColumnUpperBoundInput {
                data_type: field.data_type().clone(),
                input_max_string_length_bytes: Some(max_len),
                input_total_string_bytes: Some(total_bytes),
            })
        })
        .collect()
}

fn fixed_width_consumer_output_column(data_type: &DataType) -> Option<crate::exec::JoinOutputColumnUpperBoundInput> {
    match data_type {
        DataType::Boolean
        | DataType::Int8
        | DataType::UInt8
        | DataType::Int16
        | DataType::UInt16
        | DataType::Int32
        | DataType::UInt32
        | DataType::Int64
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, None)
        | DataType::Duration(_)
        | DataType::Decimal128(_, _) => Some(crate::exec::JoinOutputColumnUpperBoundInput {
            data_type: data_type.clone(),
            input_max_string_length_bytes: None,
            input_total_string_bytes: None,
        }),
        _ => None,
    }
}

fn checked_consumer_add(left: u64, right: u64, reason: &'static str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| composed_schedule_error(reason, None))
}

fn generic_graph_execution_blocker() -> ReservationRequirement {
    ReservationRequirement::blocked_by_lower_level(
        ReservationKind::OperatorOutput,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
        crate::capability::ContractQuality::ExactPreflight,
    )
}

fn composed_schedule_error(schedule_reason: &'static str, occurrence_id: Option<&str>) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "a composed graph attempt lacks a complete ordered memory schedule",
    )
    .with_fact("reason", "graph_attempt_complete_phase_unmodeled")
    .with_fact("schedule_reason", schedule_reason);
    if let Some(occurrence_id) = occurrence_id {
        error = error.with_fact("occurrence_id", occurrence_id);
    }
    error
}

fn sort_owner_terms(
    descriptor: &SortMetadataDescriptor,
    components: SortRadixPreflight,
) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = descriptor.execution_scope_fingerprint.clone();
    let owner = OwnerId::Sort(descriptor.node_id);
    [
        (OwnerTermRole::SortKeyInput, AllocationClass::ActivePhase, components.key_input_bytes),
        (OwnerTermRole::SortKeyOutput, AllocationClass::ActivePhase, components.key_output_bytes),
        (OwnerTermRole::SortSequence, AllocationClass::ActivePhase, components.sequence_bytes),
        (OwnerTermRole::SortTemporaryWorkspace, AllocationClass::ActivePhase, components.temporary_workspace_bytes),
        (OwnerTermRole::SortRetainedOrder, AllocationClass::AttemptRetained, components.retained_order_bytes),
        (OwnerTermRole::SortOutputFrame, AllocationClass::AttemptRetained, descriptor.output_frame_bytes),
        (OwnerTermRole::SortOutputNullMask, AllocationClass::AttemptRetained, descriptor.output_null_mask_bytes),
    ]
    .into_iter()
    .map(|(role, class, bytes)| {
        AllocationOwnerTerm::new(
            owner.clone(),
            role,
            OccurrenceId::PlanLocal(descriptor.node_id),
            scope.clone(),
            class,
            bytes,
            None,
        )
    })
    .collect()
}

fn limit_owner_terms(
    descriptor: &LimitMetadataDescriptor,
    components: LimitRangeGatherPreflight,
) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = descriptor.execution_scope_fingerprint.clone();
    let owner = OwnerId::Limit(descriptor.node_id);
    [
        (OwnerTermRole::LimitGatherMap, AllocationClass::ActivePhase, components.gather_map_bytes),
        (OwnerTermRole::LimitNativeWorkspace, AllocationClass::ActivePhase, components.native_workspace_bytes),
        (OwnerTermRole::LimitOutputFrame, AllocationClass::AttemptRetained, components.output_frame_bytes),
        (OwnerTermRole::LimitOutputNullMask, AllocationClass::AttemptRetained, components.output_null_mask_bytes),
    ]
    .into_iter()
    .map(|(role, class, bytes)| {
        AllocationOwnerTerm::new(
            owner.clone(),
            role,
            OccurrenceId::PlanLocal(descriptor.node_id),
            scope.clone(),
            class,
            bytes,
            None,
        )
    })
    .collect()
}

fn validate_sort_metadata_descriptor(descriptor: &SortMetadataDescriptor) -> Result<()> {
    if descriptor.execution_scope_fingerprint.is_empty() {
        return Err(sort_owner_error("missing_scope", "sort_scope"));
    }
    if descriptor.stable {
        return Err(sort_owner_error("stable_sort_unsupported", "sort_dispatch"));
    }
    if descriptor.null_count != 0 {
        return Err(sort_owner_error("nullable_key_unsupported", "sort_key"));
    }
    let key_type = descriptor.key_type.id();
    if !(key_type.is_signed_int()
        || key_type.is_unsigned_int()
        || key_type.is_timestamp()
        || key_type.is_duration()
        || key_type.is_float())
    {
        return Err(sort_owner_error("key_type_not_radix_dispatchable", "sort_key"));
    }
    if descriptor.num_rows > i32::MAX as usize {
        return Err(sort_owner_error("row_count_exceeds_cudf_size_type", "sort_input"));
    }
    FiniteBytes::new(descriptor.output_frame_bytes)
        .map_err(|_| sort_owner_error("sort_output_bytes_saturated", "sort_output"))?;
    FiniteBytes::new(descriptor.output_null_mask_bytes)
        .map_err(|_| sort_owner_error("sort_output_null_mask_saturated", "sort_output_null_mask"))?;
    if let Some(limit) = &descriptor.limit {
        if limit.execution_scope_fingerprint != descriptor.execution_scope_fingerprint {
            return Err(limit_owner_error("scope_mismatch", "limit_scope"));
        }
        if limit.input_rows != descriptor.num_rows as u64 || limit.output_rows > descriptor.num_rows {
            return Err(limit_owner_error("row_bound_mismatch", "limit_rows"));
        }
        if limit.output_columns.is_empty() {
            return Err(limit_owner_error("output_schema_empty", "limit_output"));
        }
    }
    Ok(())
}

fn sort_metadata_for_attested_graph_result(
    plan: &QueryPlan,
    graph_result_input_name: &str,
    producer_row_count_upper_bound: Option<u64>,
) -> Result<SortMetadataDescriptor> {
    if !is_attested_graph_result_bounded_consumer_plan(plan, graph_result_input_name) {
        return Err(sort_owner_error("consumer_topology_unmodeled", "sort_plan"));
    }
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut source = None;
    let mut sort = None;
    let mut limit = None;
    let mut filter_count = 0_usize;
    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        match &node.kind {
            PlanNode::Source(source_node) if matches!(source_node.as_ref(), SourceNode::GpuResident(source) if source.input_name == graph_result_input_name) => {
                if source.replace((node.id, source_node.as_ref())).is_some() {
                    return Err(sort_owner_error("duplicate_graph_result_source", "sort_input"));
                }
            },
            PlanNode::Sort(sort_node) => {
                if sort.replace((node.id, sort_node)).is_some() {
                    return Err(sort_owner_error("multiple_sort_nodes", "sort_plan"));
                }
            },
            PlanNode::Limit(limit_node) => {
                if limit.replace((node.id, limit_node)).is_some() {
                    return Err(sort_owner_error("multiple_limit_nodes", "limit_plan"));
                }
            },
            PlanNode::Filter(_) => {
                filter_count = filter_count
                    .checked_add(1)
                    .ok_or_else(|| sort_owner_error("filter_node_count_overflow", "sort_plan"))?;
                if filter_count > 1 {
                    return Err(sort_owner_error("multiple_filter_nodes", "sort_plan"));
                }
            },
            _ => return Err(sort_owner_error("sort_plan_contains_unmodeled_node", "sort_plan")),
        }
    }
    let Some((source_node_id, source_node)) = source else {
        return Err(sort_owner_error("graph_result_source_missing", "sort_input"));
    };
    let SourceNode::GpuResident(source_node) = source_node else {
        return Err(sort_owner_error("graph_result_source_not_gpu_resident", "sort_input"));
    };
    let Some((sort_node_id, sort_node)) = sort else {
        return Err(sort_owner_error("sort_node_missing", "sort_plan"));
    };
    if sort_node.input != source_node_id
        && !plan
            .node(sort_node.input)
            .is_some_and(|node| matches!(&node.kind, PlanNode::Filter(filter) if filter.input == source_node_id))
    {
        return Err(sort_owner_error("sort_input_topology_unmodeled", "sort_plan"));
    }
    let [sort_expr] = sort_node.spec.expressions.as_slice() else {
        return Err(sort_owner_error("sort_requires_exactly_one_key", "sort_key"));
    };
    let Some(key_column) = alias_wrapped_column(&sort_expr.expr) else {
        return Err(sort_owner_error("sort_key_expression_unmodeled", "sort_key"));
    };
    let field = source_node
        .schema
        .field_with_name(key_column.name())
        .map_err(|_| sort_owner_error("sort_key_column_missing", "sort_key").with_fact("column", key_column.name()))?;
    if field.is_nullable() {
        return Err(sort_owner_error("nullable_key_unsupported", "sort_key"));
    }
    let key_type = cudf_dtype_from_arrow_type(field.data_type())
        .ok_or_else(|| sort_owner_error("key_type_unavailable", "sort_key"))?;
    let key_type_id = key_type.id();
    if !(key_type_id.is_signed_int()
        || key_type_id.is_unsigned_int()
        || key_type_id.is_timestamp()
        || key_type_id.is_duration()
        || key_type_id.is_float())
    {
        return Err(
            sort_owner_error("key_type_not_radix_dispatchable", "sort_key").with_fact("key_type", key_type_id.name())
        );
    }
    if let (Some(consumer_row_count_upper_bound), Some(producer_row_count_upper_bound)) =
        (source_node.row_count_upper_bound, producer_row_count_upper_bound)
        && consumer_row_count_upper_bound != producer_row_count_upper_bound
    {
        return Err(sort_owner_error("graph_result_row_bound_mismatch", "sort_input")
            .with_fact("producer_row_count_upper_bound", producer_row_count_upper_bound)
            .with_fact("consumer_row_count_upper_bound", consumer_row_count_upper_bound));
    }
    let row_count = producer_row_count_upper_bound
        .or(source_node.row_count_upper_bound)
        .ok_or_else(|| sort_owner_error("row_bound_missing", "sort_input"))?;
    let num_rows = usize::try_from(row_count).map_err(|_| sort_owner_error("row_bound_exceeds_usize", "sort_input"))?;
    let sort_output_schema =
        plan.node(sort_node_id).ok_or_else(|| sort_owner_error("sort_node_missing", "sort_plan"))?.output_schema();
    let (output_frame_bytes, output_null_mask_bytes) =
        bounded_consumer_frame_components(row_count, sort_output_schema)?;
    let limit = limit
        .map(|(limit_node_id, limit_node)| {
            if limit_node.input != sort_node_id || plan.root_id() != limit_node_id || sort_node.input != source_node_id
            {
                return Err(sort_owner_error("limit_input_topology_unmodeled", "limit_plan"));
            }
            let fetch = limit_node.spec.fetch.ok_or_else(|| limit_owner_error("fetch_unbounded", "limit_rows"))?;
            let _ = limit_node
                .spec
                .skip
                .checked_add(fetch)
                .ok_or_else(|| limit_owner_error("skip_fetch_overflow", "limit_rows"))?;
            let skip =
                u64::try_from(limit_node.spec.skip).map_err(|_| limit_owner_error("skip_exceeds_u64", "limit_rows"))?;
            let fetch = u64::try_from(fetch).map_err(|_| limit_owner_error("fetch_exceeds_u64", "limit_rows"))?;
            let output_rows_u64 = row_count.saturating_sub(skip).min(fetch);
            let output_rows = usize::try_from(output_rows_u64)
                .map_err(|_| limit_owner_error("output_rows_exceed_usize", "limit_rows"))?;
            let output_columns = plan
                .root_node()
                .output_schema()
                .fields()
                .iter()
                .map(|field| {
                    let data_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
                        limit_owner_error("output_dtype_unmodeled", "limit_output")
                            .with_fact("column", field.name())
                            .with_fact("data_type", format!("{:?}", field.data_type()))
                    })?;
                    Ok((data_type, field.is_nullable()))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(LimitMetadataDescriptor {
                node_id: limit_node_id,
                execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
                input_rows: row_count,
                output_rows,
                output_columns,
            })
        })
        .transpose()?;
    Ok(SortMetadataDescriptor {
        node_id: sort_node_id,
        execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
        num_rows,
        key_type,
        null_count: 0,
        stable: false,
        key_order: if sort_expr.descending {
            cudf::ops::sorting::Order::Descending
        } else {
            cudf::ops::sorting::Order::Ascending
        },
        output_frame_bytes,
        output_null_mask_bytes,
        limit,
    })
}

fn bounded_consumer_frame_components(rows: u64, schema: &SchemaRef) -> Result<(u64, u64)> {
    let mut frame = 0_u64;
    let mut null_mask = 0_u64;
    for field in schema.fields() {
        let width = fixed_width_byte_size(field.data_type())
            .ok_or_else(|| sort_owner_error("sort_output_dtype_unmodeled", "sort_output"))?;
        let data =
            rows.checked_mul(width).ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?;
        frame = frame
            .checked_add(data)
            .and_then(|value| value.checked_add(crate::exec::PER_COLUMN_ALIGNMENT_PADDING_BYTES))
            .ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?;
        let mask = if rows == 0 {
            0
        } else {
            rows.div_ceil(512)
                .checked_mul(64)
                .ok_or_else(|| sort_owner_error("sort_output_null_mask_overflow", "sort_output_null_mask"))?
        };
        null_mask = null_mask
            .checked_add(mask)
            .ok_or_else(|| sort_owner_error("sort_output_null_mask_overflow", "sort_output_null_mask"))?;
    }
    FiniteBytes::new(frame).map_err(|_| sort_owner_error("sort_output_bytes_saturated", "sort_output"))?;
    FiniteBytes::new(null_mask)
        .map_err(|_| sort_owner_error("sort_output_null_mask_saturated", "sort_output_null_mask"))?;
    Ok((frame, null_mask))
}

fn fixed_width_byte_size(data_type: &DataType) -> Option<u64> {
    match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => Some(1),
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 | DataType::Time32(_) => Some(4),
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_) => Some(8),
        DataType::Decimal128(_, _) => Some(16),
        _ => None,
    }
}

fn cugraph_graph_execution_capability() -> ReservationRequirement {
    ReservationRequirement::modeled(
        ReservationKind::GraphAlgorithm,
        "cugraph_graph_execution",
        "cugraph_graph_execution_contract",
        crate::capability::ContractQuality::CapabilityProof,
    )
}

fn is_generic_graph_execution_blocker(requirement: &ReservationRequirement) -> bool {
    requirement == &generic_graph_execution_blocker()
}

fn is_attested_graph_result_passthrough_plan(plan: &QueryPlan, graph_result_input_name: &str) -> bool {
    let [node] = plan.nodes() else {
        return false;
    };
    if !plan.unreachable_node_ids().is_empty() || plan.root_id() != node.id {
        return false;
    }
    matches!(
        &node.kind,
        PlanNode::Source(source)
            if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name == graph_result_input_name)
    )
}

fn is_attested_graph_result_bounded_consumer_plan(plan: &QueryPlan, graph_result_input_name: &str) -> bool {
    let mut graph_result_source_count = 0_usize;
    let mut graph_result_source_id = None;
    let mut sort_count = 0_usize;
    let mut sort_id = None;
    let mut filter_count = 0_usize;
    let mut limit_count = 0_usize;
    let mut limit_id = None;
    let mut limit_input = None;
    let mut sort_input = None;
    for node in plan.nodes() {
        if plan.unreachable_node_ids().contains(&node.id) {
            continue;
        }
        match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name == graph_result_input_name) =>
            {
                graph_result_source_count += 1;
                graph_result_source_id = Some(node.id);
            },
            PlanNode::Sort(sort) => {
                sort_count += 1;
                sort_id = Some(node.id);
                sort_input = Some(sort.input);
            },
            PlanNode::Filter(_) => {
                filter_count += 1;
            },
            PlanNode::Limit(limit) => {
                limit_count += 1;
                limit_id = Some(node.id);
                limit_input = Some(limit.input);
            },
            _ => return false,
        }
    }
    if graph_result_source_count != 1 || sort_count != 1 {
        return false;
    }
    match (limit_count, limit_id, limit_input, sort_id, graph_result_source_id, sort_input) {
        (1, Some(limit_id), Some(limit_input), Some(sort_id), Some(source_id), Some(sort_input)) => {
            filter_count == 0 && plan.root_id() == limit_id && limit_input == sort_id && sort_input == source_id
        },
        (0, None, None, Some(sort_id), Some(source_id), Some(sort_input)) => {
            if plan.root_id() != sort_id || filter_count > 1 {
                return false;
            }
            if sort_input == source_id {
                return true;
            }
            filter_count == 1
                && plan
                    .node(sort_input)
                    .is_some_and(|node| matches!(&node.kind, PlanNode::Filter(filter) if filter.input == source_id))
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests;
