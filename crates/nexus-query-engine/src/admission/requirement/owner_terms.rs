use super::*;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EstimateQuality {
    #[default]
    Unknown,
    Modeled,
    Measured,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum OwnerId {
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
    pub(crate) fn as_str(&self) -> &'static str {
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
pub(crate) enum OwnerTermRole {
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
pub(crate) enum AllocationClass {
    AttemptWideFixed,
    ActivePhase,
    AttemptRetained,
}

impl AllocationClass {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::AttemptWideFixed => "attempt_wide_fixed",
            Self::ActivePhase => "active_phase",
            Self::AttemptRetained => "attempt_retained",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OwnerTermConstructionPolicy {
    Generic,
    AllowLiteralBatch,
    PriceLiteralBatch,
    AllowMixedSourceComposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum OccurrenceId {
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
    pub(crate) owner: OwnerId,
    pub(crate) role: OwnerTermRole,
    pub(crate) occurrence: OccurrenceId,
    pub(crate) execution_scope_fingerprint: String,
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
    pub(crate) key_input_bytes: u64,
    pub(crate) key_output_bytes: u64,
    pub(crate) sequence_bytes: u64,
    pub(crate) temporary_workspace_bytes: u64,
    pub(crate) retained_order_bytes: u64,
    pub(crate) active_phase_peak_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LimitRangeGatherPreflight {
    pub(crate) gather_map_bytes: u64,
    pub(crate) output_frame_bytes: u64,
    pub(crate) output_null_mask_bytes: u64,
    pub(crate) native_workspace_bytes: u64,
    pub(crate) active_phase_peak_bytes: u64,
}

impl LimitRangeGatherPreflight {
    pub(crate) fn checked_from_cudf(preflight: cudf::ops::copying::GatherFixedWidthDontCheckPreflight) -> Result<Self> {
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
    pub(crate) fn checked_from_cudf(preflight: cudf::ops::sorting::SortedOrderRadixPreflight) -> Result<Self> {
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
pub(crate) struct FiniteBytes(pub(crate) u64);

impl FiniteBytes {
    pub(crate) fn new(bytes: u64) -> Result<Self> {
        if bytes == u64::MAX {
            return Err(owner_term_error("saturated", None, None));
        }
        Ok(Self(bytes))
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

pub(crate) fn owner_term_shape_is_valid(owner: &OwnerId, role: OwnerTermRole, class: AllocationClass) -> bool {
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

pub(crate) fn validate_retention_shape(
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
    pub(crate) id: OwnerTermId,
    pub(crate) class: AllocationClass,
    pub(crate) bytes: FiniteBytes,
    pub(crate) retained_row_count_upper_bound: Option<u64>,
}

impl AllocationOwnerTerm {
    pub(crate) fn new(
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

    pub(crate) fn rebind_composed_occurrence(&self, occurrence_id: &str) -> Result<Self> {
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

pub(crate) fn owner_term_error(
    reason: &'static str,
    role: Option<OwnerTermRole>,
    class: Option<AllocationClass>,
) -> QueryError {
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

pub(crate) fn owner_derivation_error(reason: impl ToString) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "attempt bound derivation is unavailable",
    )
    .with_fact("reason", reason)
    .with_fact("derivation", "conservative_owner_sum")
}

pub(crate) fn sort_owner_error(reason: &'static str, resource: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "sorted-order owner preflight is unavailable",
    )
    .with_fact("reason", format!("sort_owner_{reason}"))
    .with_fact("resource", resource)
}

pub(crate) fn limit_owner_error(reason: &'static str, resource: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.requirement",
        "bounded limit range-gather owner preflight is unavailable",
    )
    .with_fact("reason", format!("limit_owner_{reason}"))
    .with_fact("resource", resource)
}

pub(crate) fn graph_owner_term_from_fact(fact: &GraphMemoryOwnerTermFact) -> Result<AllocationOwnerTerm> {
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

pub(crate) fn validate_owner_term(term: &AllocationOwnerTerm) -> Result<()> {
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

pub(crate) fn merge_owner_terms(
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

pub(crate) fn graph_owner_terms_from_evidence(
    evidence: &GraphMemoryAdmissionEvidence,
) -> Result<Vec<AllocationOwnerTerm>> {
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

pub(crate) fn local_parquet_source_owner_terms(plan: &QueryPlan) -> Result<Vec<AllocationOwnerTerm>> {
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

pub(crate) fn has_mixed_local_gpu_sources(plan: &QueryPlan) -> bool {
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

pub(crate) fn generic_gpu_resident_owner_terms(
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

pub(crate) fn projection_cast_workspace_owner_terms(plan: &QueryPlan, scope: &str) -> Result<Vec<AllocationOwnerTerm>> {
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

pub(crate) fn generic_aggregate_owner_terms(plan: &QueryPlan) -> Result<Vec<AllocationOwnerTerm>> {
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

pub(crate) fn aggregate_owner_state_is_supported(aggregate: &AggregateNode) -> bool {
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

pub(crate) fn aggregate_owner_terms_for_shapes(
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

pub(crate) fn literal_batch_owner_term_rejection_reason(
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

pub(crate) fn literal_batch_source_owner_terms(plan: &QueryPlan) -> Result<Vec<AllocationOwnerTerm>> {
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
            let mut bytes = crate::exec::join_output_upper_bound_bytes(rows, &columns).map_err(|_| {
                owner_term_error("literal_source_bytes_unavailable", Some(OwnerTermRole::SourceStage), None)
            })?;
            for (index, field) in source.schema.fields().iter().enumerate() {
                if matches!(field.data_type(), DataType::List(_) | DataType::LargeList(_)) {
                    let arrays = source.batches.iter().map(|batch| Arc::clone(batch.column(index))).collect::<Vec<_>>();
                    let list_bytes = cudf::interop::arrow_host::estimate_list_import_device_memory_bytes(&arrays)
                        .map_err(|_| {
                            owner_term_error("literal_source_bytes_unavailable", Some(OwnerTermRole::SourceStage), None)
                        })?;
                    bytes = bytes
                        .checked_add(u64::try_from(list_bytes).map_err(|_| {
                            owner_term_error("literal_source_bytes_saturated", Some(OwnerTermRole::SourceStage), None)
                        })?)
                        .ok_or_else(|| {
                            owner_term_error("literal_source_bytes_saturated", Some(OwnerTermRole::SourceStage), None)
                        })?;
                }
            }
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

pub(crate) fn literal_batch_output_columns(
    source: &crate::source::LiteralBatchSource,
) -> Result<Vec<crate::exec::JoinOutputColumnUpperBoundInput>> {
    source
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if matches!(field.data_type(), DataType::List(_) | DataType::LargeList(_)) {
                return Ok(None);
            }
            if let Some(column) = fixed_width_consumer_output_column(field.data_type()) {
                return Ok(Some(column));
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
            Ok(Some(crate::exec::JoinOutputColumnUpperBoundInput {
                data_type: field.data_type().clone(),
                input_max_string_length_bytes: Some(max_len),
                input_total_string_bytes: Some(total_bytes),
            }))
        })
        .filter_map(|column| column.transpose())
        .collect()
}
