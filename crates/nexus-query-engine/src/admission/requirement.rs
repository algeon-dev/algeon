//! Device-independent requirements for one bounded execution attempt.
//!
//! Estimates in this module are intentionally soft: they are useful to choose
//! a target grant, but never participate in the required floor or capability
//! admission decision.

use std::collections::{BTreeSet, HashMap, HashSet};

use arrow::datatypes::{DataType, SchemaRef, TimeUnit};

use crate::{
    admission::{
        AttemptReservationEnvelope, AttemptReservationPolicy,
        envelope::{
            AttemptReservationProducerOccurrence, AttemptReservationSchedule, AttemptReservationShape,
            canonical_execution_scope_fingerprint,
        },
    },
    capability::ReservationRequirement,
    datatypes::cudf_dtype_from_arrow_type,
    error::{QueryError, QueryErrorCode, Result},
    expr::alias_wrapped_column,
    graph_memory::{
        GraphMemoryAdmissionEvidence, GraphMemoryOwnerTermFact, GraphMemoryOwnerTermRole,
        query_plan_hard_row_upper_bound,
    },
    plan::{
        JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, ResultSink, estimated_local_parquet_source_bytes,
        estimated_reachable_local_parquet_source_bytes,
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
    Source(NodeId),
    Consumer(NodeId),
    SideInput(NodeId),
    Sort(NodeId),
    CacheLedger,
}

impl OwnerId {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Attempt => "attempt",
            Self::Graph(_) => "graph",
            Self::Producer(_) => "producer",
            Self::Source(_) => "source",
            Self::Consumer(_) => "consumer",
            Self::SideInput(_) => "side_input",
            Self::Sort(_) => "sort",
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
    SideInputWorkspace,
    RetainedOutput,
    SourceStage,
    ConsumerOutput,
    ConsumerPredicateMask,
    ConsumerBuildInput,
    ConsumerProbeInput,
    ConsumerHashTable,
    ConsumerPairIndices,
    SideInputFrame,
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
            Self::SideInputWorkspace => "side_input_workspace",
            Self::RetainedOutput => "retained_output",
            Self::SourceStage => "source_stage",
            Self::ConsumerOutput => "consumer_output",
            Self::ConsumerPredicateMask => "consumer_predicate_mask",
            Self::ConsumerBuildInput => "consumer_build_input",
            Self::ConsumerProbeInput => "consumer_probe_input",
            Self::ConsumerHashTable => "consumer_hash_table",
            Self::ConsumerPairIndices => "consumer_pair_indices",
            Self::SideInputFrame => "side_input_frame",
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
    key_output_bytes: u64,
    sequence_bytes: u64,
    temporary_workspace_bytes: u64,
    retained_order_bytes: u64,
    active_phase_peak_bytes: u64,
}

impl SortRadixPreflight {
    fn checked_from_cudf(preflight: cudf::ops::sorting::SortedOrderRadixPreflight) -> Result<Self> {
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
        let expected_peak = key_output_bytes
            .checked_add(sequence_bytes)
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
        Self {
            key_output_bytes,
            sequence_bytes,
            temporary_workspace_bytes,
            retained_order_bytes,
            active_phase_peak_bytes: key_output_bytes
                .checked_add(sequence_bytes)
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
            | OwnerTermRole::SideInputWorkspace => class == AllocationClass::ActivePhase,
            OwnerTermRole::RetainedOutput => class == AllocationClass::AttemptRetained,
            _ => false,
        },
        OwnerId::Producer(_) => role == OwnerTermRole::RetainedOutput && class == AllocationClass::AttemptRetained,
        OwnerId::Source(_) => role == OwnerTermRole::SourceStage && class == AllocationClass::ActivePhase,
        OwnerId::Consumer(_) => {
            matches!(
                role,
                OwnerTermRole::ConsumerOutput
                    | OwnerTermRole::ConsumerPredicateMask
                    | OwnerTermRole::ConsumerBuildInput
                    | OwnerTermRole::ConsumerProbeInput
                    | OwnerTermRole::ConsumerHashTable
                    | OwnerTermRole::ConsumerPairIndices
            ) && class == AllocationClass::ActivePhase
        },
        OwnerId::SideInput(_) => role == OwnerTermRole::SideInputFrame && class == AllocationClass::ActivePhase,
        OwnerId::Sort(_) => match role {
            OwnerTermRole::SortKeyOutput | OwnerTermRole::SortSequence | OwnerTermRole::SortTemporaryWorkspace => {
                class == AllocationClass::ActivePhase
            },
            OwnerTermRole::SortRetainedOrder | OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask => {
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
        OwnerTermRole::SortRetainedOrder | OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask
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
        let OccurrenceId::PlanLocal(_) = self.id.occurrence else {
            return Err(owner_term_error("occurrence_already_composed", Some(self.id.role), Some(self.class)));
        };
        Self::new(
            self.id.owner.clone(),
            self.id.role,
            OccurrenceId::Composed(occurrence_id.to_owned()),
            self.id.execution_scope_fingerprint.clone(),
            self.class,
            self.bytes.get(),
            self.retained_row_count_upper_bound,
        )
    }

    pub(crate) fn id(&self) -> &OwnerTermId {
        &self.id
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes.get()
    }

    pub(crate) const fn is_retained_output(&self) -> bool {
        matches!(self.id.role, OwnerTermRole::RetainedOutput)
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
                    | OwnerTermRole::SideInputWorkspace
            )
    }

    pub(crate) const fn is_construction_phase_term(&self) -> bool {
        matches!(self.id.role, OwnerTermRole::GraphPersistent | OwnerTermRole::ConstructionTransient)
    }

    pub(crate) const fn is_execution_phase_term(&self) -> bool {
        matches!(
            self.id.role,
            OwnerTermRole::GraphPersistent | OwnerTermRole::ExecutionWorkspace | OwnerTermRole::SideInputWorkspace
        )
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.id.execution_scope_fingerprint
    }

    pub(crate) fn provides_complete_phase_coverage(&self) -> bool {
        !matches!(self.id.role, OwnerTermRole::SourceStage)
    }

    #[cfg(not(test))]
    pub(crate) fn is_consumer_scope(&self, scope: &str) -> bool {
        matches!(self.id.owner, OwnerId::Consumer(_)) && self.id.execution_scope_fingerprint == scope
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

fn graph_owner_term_from_fact(fact: &GraphMemoryOwnerTermFact) -> Result<AllocationOwnerTerm> {
    let (role, class) = match fact.role() {
        GraphMemoryOwnerTermRole::GraphPersistent => (OwnerTermRole::GraphPersistent, AllocationClass::ActivePhase),
        GraphMemoryOwnerTermRole::ConstructionTransient => {
            (OwnerTermRole::ConstructionTransient, AllocationClass::ActivePhase)
        },
        GraphMemoryOwnerTermRole::ExecutionWorkspace => {
            (OwnerTermRole::ExecutionWorkspace, AllocationClass::ActivePhase)
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
    const MAX_GRAPH_OWNER_TERM_COUNT: usize = REQUIRED_GRAPH_OWNER_TERM_COUNT + 1;
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

fn local_parquet_source_owner_terms(
    plan: &QueryPlan,
    allow_mixed_source_composition: bool,
) -> Result<Vec<AllocationOwnerTerm>> {
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
        let SourceNode::LocalParquet(source) = source.as_ref() else {
            has_non_local_source = true;
            has_disallowed_non_local_source |= !matches!(source.as_ref(), SourceNode::GpuResident(_));
            continue;
        };
        has_local_source = true;
        let Some(bytes) = estimated_local_parquet_source_bytes(source) else {
            return Err(owner_term_error("local_source_bytes_unavailable", Some(OwnerTermRole::SourceStage), None));
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
    }
    if has_local_source && has_non_local_source && (!allow_mixed_source_composition || has_disallowed_non_local_source)
    {
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
    allow_mixed_source_composition: bool,
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
    let local_source_terms = local_parquet_source_owner_terms(plan, allow_mixed_source_composition)?;
    if !local_source_terms.is_empty() {
        let reachable_sources = plan
            .nodes()
            .iter()
            .filter(|node| {
                !plan.unreachable_node_ids().contains(&node.id)
                    && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)))
            })
            .collect::<Vec<_>>();
        let [source_node] = reachable_sources.as_slice() else {
            return Err(owner_term_error(
                "gpu_resident_output_owner_alias_unmodeled",
                Some(OwnerTermRole::RetainedOutput),
                None,
            ));
        };
        if source_node.id != *input {
            return Err(owner_term_error(
                "gpu_resident_output_owner_alias_unmodeled",
                Some(OwnerTermRole::RetainedOutput),
                None,
            ));
        }
        let rows = query_plan_hard_row_upper_bound(plan).ok_or_else(|| {
            owner_term_error("gpu_resident_row_bound_unavailable", Some(OwnerTermRole::RetainedOutput), None)
        })?;
        let output_bytes = bounded_consumer_frame_bytes(rows, source_node.output_schema()).map_err(|error| {
            owner_term_error("gpu_resident_output_bytes_unavailable", Some(OwnerTermRole::RetainedOutput), None)
                .with_fact("sizing_reason", error.fact_value("schedule_reason").unwrap_or("unknown"))
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
            source_bytes.max(output_bytes),
            Some(rows),
        )?]);
    }
    if !plan.nodes().iter().any(|node| {
        !plan.unreachable_node_ids().contains(&node.id)
            && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_)))
    }) {
        return Ok(Vec::new());
    }
    // A GPU-resident source already names an upstream retained handle.  The
    // generic plan proof cannot establish whether this sink materializes a
    // distinct output or merely forwards that handle, so minting another
    // producer owner would either double-charge a passthrough or leave a
    // materialized output unbound to its source.  Composed admission supplies
    // the retained-input binding when it owns that contract; reject this
    // standalone shape until then.
    Err(owner_term_error("gpu_resident_output_owner_alias_unmodeled", Some(OwnerTermRole::RetainedOutput), None))
}

fn literal_batch_owner_term_rejection_reason(
    plan: &QueryPlan,
    policy: OwnerTermConstructionPolicy,
) -> Option<&'static str> {
    if policy == OwnerTermConstructionPolicy::AllowLiteralBatch {
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

    /// Derives the consumer proof used only while composing an exact GPU
    /// island. This policy permits a reachable LocalParquet source alongside
    /// GPU-resident branches so the shared traversal can consume the
    /// canonical source-stage terms. Literal and other non-GPU source kinds
    /// remain rejected.
    pub fn for_composed_consumer_plan(plan: &QueryPlan) -> Option<Self> {
        let mut requirement = AttemptCapabilityEvidence::for_query_plan_with_policy(
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
        let mut requirement = AttemptCapabilityEvidence::for_query_plan(plan);
        requirement.remove_generic_gpu_resident_retained_output_terms();
        let graph_requirement = generic_graph_execution_blocker();
        if !requirement.required_capabilities.contains(&graph_requirement) {
            return (!requirement.required_capabilities.is_empty()).then_some(Self { requirement });
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
        Some(Self { requirement })
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
        let consumer_scope = consumer_shape.execution_scope_fingerprint().to_owned();

        let mut occurrence_ids = std::collections::HashSet::with_capacity(ordered_producers.len());
        let mut occurrences = Vec::with_capacity(ordered_producers.len());
        let mut producer_outputs = Vec::with_capacity(ordered_producers.len());
        let mut producer_proofs = Vec::with_capacity(ordered_producers.len());
        let mut retained_terms_by_occurrence = HashMap::with_capacity(ordered_producers.len());
        for (occurrence_id, mut producer) in ordered_producers {
            if occurrence_id.is_empty() {
                return Err(composed_schedule_error("missing_producer_occurrence_id", None));
            }
            if !occurrence_ids.insert(occurrence_id.clone()) {
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
            let producer_scope = producer_shape.execution_scope_fingerprint().to_owned();
            producer.requirement.rebind_owner_terms_to_composed_occurrence(&occurrence_id)?;
            let retained_output_term = producer
                .requirement
                .retained_output_term_for_composition()
                .map_err(|error| error.with_fact("occurrence_id", occurrence_id.as_str()))?;
            let retained_output_term_id = retained_output_term.id().clone();
            let row_count_upper_bound = retained_output_term.retained_row_count_upper_bound().ok_or_else(|| {
                composed_schedule_error("producer_retained_output_rows_missing", Some(&occurrence_id))
            })?;
            let producer_term_groups = producer.requirement.isolated_phase_term_groups();
            retained_terms_by_occurrence.insert(occurrence_id.clone(), retained_output_term_id.clone());
            producer_outputs.push(ComposedGraphProducerOutputEvidence {
                occurrence_id: occurrence_id.clone(),
                row_count_upper_bound,
            });
            occurrences.push(AttemptReservationProducerOccurrence::from_term_groups(
                occurrence_id,
                producer_scope,
                producer_term_groups,
                Some(retained_output_term_id),
            ));
            producer_proofs.push(producer);
        }

        let ComposedConsumerBranchBound {
            owner_terms: consumer_owner_terms,
            peak_owner_term_ids: consumer_peak_term_ids,
            consumed_occurrences,
            ..
        } = composed_consumer_bound_with_peak_terms(
            consumer_plan,
            &producer_outputs,
            &consumer.requirement.owner_terms,
        )?;
        let mut consumer = consumer;
        consumer.requirement.requires_composed_producer_binding = false;
        consumer.requirement.remove_generic_gpu_resident_retained_output_terms();
        consumer.requirement.owner_terms = merge_owner_terms(&consumer.requirement.owner_terms, consumer_owner_terms)?;
        let schedule = AttemptReservationSchedule::from_term_ids(
            // Current bounded composed attempts have no independent
            // attempt-wide device allocation.  Keep this explicit so a future
            // attempt-wide owner must be represented in the schedule rather
            // than silently folded into a producer or consumer phase.
            Some(0),
            None,
            occurrences,
            Some(consumer_scope),
            consumer_peak_term_ids,
        );
        let [consumer_shape] = consumer.requirement.reservation_shapes.as_mut_slice() else {
            return Err(composed_schedule_error("missing_consumer_scope", None));
        };
        let bindings = consumed_occurrences
            .iter()
            .map(|occurrence_id| {
                retained_terms_by_occurrence
                    .get(occurrence_id)
                    .cloned()
                    .map(|term_id| (occurrence_id.clone(), term_id))
                    .ok_or_else(|| {
                        composed_schedule_error("consumer_retained_input_binding_missing", Some(occurrence_id))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        consumer_shape.set_retained_input_bindings(bindings)?;
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
        let ordered_producers = ordered_producers.into_iter().collect::<Vec<_>>();

        let ordered_result = ordered_producers
            .iter()
            .map(|producer| {
                let occurrence_id = producer
                    .runtime_occurrence_id
                    .clone()
                    .ok_or_else(|| composed_schedule_error("runtime_producer_occurrence_missing", None))?;
                if producer
                    .graph_result_input_name
                    .as_deref()
                    .is_some_and(|graph_result_input_name| occurrence_id != graph_result_input_name)
                {
                    return Err(composed_schedule_error("producer_occurrence_identity_mismatch", Some(&occurrence_id))
                        .with_fact(
                            "graph_result_input_name",
                            producer.graph_result_input_name.clone().unwrap_or_default(),
                        ));
                }
                let proof = producer.capability_proof.clone().ok_or_else(|| {
                    composed_schedule_error("producer_admission_proof_missing_or_ambiguous", Some(&occurrence_id))
                })?;
                Ok((occurrence_id, proof))
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|proofs| Self::combine_composed_graph_attempt(consumer_plan, consumer.clone(), proofs));
        match ordered_result {
            Ok(proof) => Ok(proof),
            Err(ordered_error) => {
                let conservative_result = ordered_producers
                    .into_iter()
                    .map(|producer| {
                        let occurrence_id = producer
                            .runtime_occurrence_id
                            .ok_or_else(|| composed_schedule_error("runtime_producer_occurrence_missing", None))?;
                        let proof = producer.capability_proof.ok_or_else(|| {
                            composed_schedule_error(
                                "producer_admission_proof_missing_or_ambiguous",
                                Some(&occurrence_id),
                            )
                        })?;
                        Ok((occurrence_id, proof))
                    })
                    .collect::<Result<Vec<_>>>()
                    .and_then(|proofs| {
                        Self::combine_conservative_composed_graph_attempt(consumer_plan, consumer, proofs)
                    })
                    .map(|mut proof| {
                        for shape in &mut proof.requirement.reservation_shapes {
                            shape.detach_ordered_schedule();
                        }
                        proof
                    });
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
        let mut producer_outputs = Vec::with_capacity(ordered_producers.len());
        let mut producer_requirements = Vec::with_capacity(ordered_producers.len());
        let mut retained_terms_by_occurrence = HashMap::with_capacity(ordered_producers.len());
        for (occurrence_id, mut producer) in ordered_producers {
            if occurrence_id.is_empty() || !occurrence_ids.insert(occurrence_id.clone()) {
                return Err(composed_schedule_error("duplicate_producer_occurrence_id", Some(&occurrence_id)));
            }
            producer.requirement.rebind_owner_terms_to_composed_occurrence(&occurrence_id)?;
            let retained_output_term = producer
                .requirement
                .retained_output_term_for_composition()
                .map_err(|error| error.with_fact("occurrence_id", occurrence_id.as_str()))?;
            let row_count_upper_bound = retained_output_term.retained_row_count_upper_bound().ok_or_else(|| {
                composed_schedule_error("producer_retained_output_rows_missing", Some(&occurrence_id))
            })?;
            let retained_output_term_id = retained_output_term.id().clone();
            retained_terms_by_occurrence.insert(occurrence_id.clone(), retained_output_term_id);
            producer_outputs.push(ComposedGraphProducerOutputEvidence {
                occurrence_id: occurrence_id.clone(),
                row_count_upper_bound,
            });
            producer_requirements.push(producer.requirement);
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
        let bindings = consumed_occurrences
            .iter()
            .map(|occurrence_id| {
                retained_terms_by_occurrence
                    .get(occurrence_id)
                    .cloned()
                    .map(|term_id| (occurrence_id.clone(), term_id))
                    .ok_or_else(|| {
                        composed_schedule_error("consumer_retained_input_binding_missing", Some(occurrence_id))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        consumer_shape.set_retained_input_bindings(bindings)?;
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

    /// Resolves all device-dependent owner terms against one explicit
    /// candidate device ordinal.
    pub fn requirement_for_device(&self, device_ordinal: usize) -> Result<Self> {
        self.requirement.resolve_for_device(device_ordinal).map(|requirement| Self { requirement })
    }

    #[cfg(test)]
    pub(crate) fn requirement_for_device_with_preflight<F>(&self, device_ordinal: usize, preflight: F) -> Result<Self>
    where F: Fn(&SortMetadataDescriptor, usize) -> Result<SortRadixPreflight> {
        self.requirement
            .resolve_for_device_with_preflight(device_ordinal, preflight)
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
        let allow_mixed_source_composition = policy == OwnerTermConstructionPolicy::AllowMixedSourceComposition;
        let source_owner_result = local_parquet_source_owner_terms(plan, allow_mixed_source_composition);
        let gpu_owner_result =
            generic_gpu_resident_owner_terms(plan, include_gpu_output_owner, allow_mixed_source_composition);
        let mut source_owner_terms = Vec::new();
        if let Ok(terms) = &source_owner_result {
            source_owner_terms.extend(terms.iter().cloned());
        }
        if let Ok(gpu_owner_terms) = &gpu_owner_result {
            if !gpu_owner_terms.is_empty() && !source_owner_terms.is_empty() {
                // A source-only GPU sink transfers the source frame into the
                // producer handle; retain one owner, never both lifetimes.
                source_owner_terms.clear();
            }
            source_owner_terms.extend(gpu_owner_terms.iter().cloned());
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
        if let Some(reason) = literal_rejection_reason {
            shape.set_owner_term_rejection_reason(reason);
        } else if let Err(error) = source_owner_result {
            shape.set_owner_term_rejection_reason(
                error.fact_value("reason").unwrap_or("attempt_owner_term_unavailable"),
            );
        } else if let Err(error) = gpu_owner_result {
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
        let owner_terms = merge_owner_terms(&self.owner_terms, graph_owner_terms)?;
        let graph_native_peak_bytes = evidence.cold_model_requested_bytes();
        let whole_attempt_peak_bytes = match self.source_stage_bytes {
            Some(source_stage_bytes) => evidence.cold_model_with_source_stage_bytes(source_stage_bytes)?,
            None => graph_native_peak_bytes,
        };
        let source_stage_bytes = self.source_stage_bytes;

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

    fn retained_output_term_for_composition(&self) -> Result<&AllocationOwnerTerm> {
        let mut retained = self.owner_terms.iter().filter(|term| term.is_retained_output());
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

    fn isolated_phase_term_groups(&self) -> Vec<Vec<OwnerTermId>> {
        let mut construction = Vec::new();
        let mut execution = Vec::new();
        let mut other_active = Vec::new();
        for term in &self.owner_terms {
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
            if !term.is_graph_phase_term() {
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
        groups
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

    fn rebind_owner_terms_to_composed_occurrence(&mut self, occurrence_id: &str) -> Result<()> {
        if occurrence_id.is_empty() {
            return Err(owner_term_error("missing_occurrence", None, None));
        }
        let rebound = self
            .owner_terms
            .iter()
            .map(|term| term.rebind_composed_occurrence(occurrence_id))
            .collect::<Result<Vec<_>>>()?;
        self.owner_terms = merge_owner_terms(&[], rebound)?;
        Ok(())
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
        self.resolve_for_device_with_preflight(device_ordinal, |descriptor, _| {
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
        })
    }

    fn resolve_for_device_with_preflight<F>(&self, device_ordinal: usize, preflight: F) -> Result<Self>
    where F: Fn(&SortMetadataDescriptor, usize) -> Result<SortRadixPreflight> {
        let mut resolved = self.clone();
        let descriptors =
            self.reservation_shapes.iter().filter_map(AttemptReservationShape::sort_metadata).collect::<Vec<_>>();
        for descriptor in &descriptors {
            validate_sort_metadata_descriptor(descriptor)?;
            let components = preflight(descriptor, device_ordinal)?;
            let terms = sort_owner_terms(descriptor, components)?;
            resolved.owner_terms = merge_owner_terms(&resolved.owner_terms, terms)?;
            resolved.set_sort_phase_peak(descriptor, components.active_phase_peak_bytes)?;
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
                .validate_term_references(owner_terms)
                .map_err(|_| owner_derivation_error("attempt_owner_term_missing"))?;
        }
        for (occurrence_id, term_id) in scope.retained_input_bindings() {
            let term = owner_terms.iter().find(|term| term.id() == term_id).ok_or_else(|| {
                owner_derivation_error("attempt_retained_input_term_missing")
                    .with_fact("occurrence_id", occurrence_id.as_str())
            })?;
            if !term.is_retained_output() || term.id().occurrence_label() != format!("composed:{occurrence_id}") {
                return Err(owner_derivation_error("attempt_retained_input_binding_invalid")
                    .with_fact("occurrence_id", occurrence_id.as_str()));
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
    peak_owner_term_ids: Vec<OwnerTermId>,
    final_owned_term_ids: Vec<OwnerTermId>,
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
    let mut used_occurrences = HashSet::with_capacity(outputs.len());
    let root = consumer_plan.root_id();
    let mut root_bound = match &consumer_plan.root_node().kind {
        PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) => {
            composed_consumer_branch_bound(consumer_plan, root, &outputs, &mut used_occurrences, owner_terms)?
        },
        PlanNode::Union(union) => {
            let mut total_rows = 0_u64;
            let mut branch_peak_bytes = 0_u64;
            let mut branch_terms = Vec::new();
            let mut peak_owner_term_ids = Vec::new();
            for input in &union.inputs {
                let branch = composed_consumer_branch_bound(
                    consumer_plan,
                    *input,
                    &outputs,
                    &mut used_occurrences,
                    owner_terms,
                )?;
                total_rows = checked_consumer_add(total_rows, branch.rows, "consumer_union_rows_overflow")?;
                branch_peak_bytes = checked_consumer_add(
                    branch_peak_bytes,
                    branch.additional_peak_bytes,
                    "consumer_union_branch_peak_overflow",
                )?;
                peak_owner_term_ids.extend(branch.peak_owner_term_ids);
                branch_terms.extend(branch.owner_terms);
            }
            let union_output = bounded_consumer_frame_bytes(total_rows, consumer_plan.root_node().output_schema())?;
            let union_output_term = consumer_owner_term(
                root,
                OwnerTermRole::ConsumerOutput,
                AllocationClass::ActivePhase,
                union_output,
                consumer_plan,
            )?;
            branch_terms.push(union_output_term);
            let union_output_term_id = branch_terms.last().expect("union output term inserted").id.clone();
            // Union branches are materialized sequentially and their retained
            // outputs coexist until the union output is formed.
            // `branch_peak_bytes` already captures that additive lifetime;
            // retain the corresponding owner identities below.
            let mut final_owned_term_ids = Vec::new();
            peak_owner_term_ids.push(union_output_term_id.clone());
            final_owned_term_ids.push(union_output_term_id);
            ComposedConsumerBranchBound {
                rows: total_rows,
                schema: consumer_plan.root_node().output_schema().clone(),
                final_owned_bytes: union_output,
                additional_peak_bytes: checked_consumer_add(
                    branch_peak_bytes,
                    union_output,
                    "consumer_union_output_overflow",
                )?,
                owner_terms: branch_terms,
                peak_owner_term_ids,
                final_owned_term_ids,
                consumed_occurrences: Vec::new(),
            }
        },
        PlanNode::Join(join) => {
            composed_consumer_join_peak_bytes(consumer_plan, root, join, &outputs, &mut used_occurrences, owner_terms)?
        },
        PlanNode::Sort(sort) => {
            composed_consumer_sort_peak_bytes(consumer_plan, root, sort, &outputs, &mut used_occurrences, owner_terms)?
        },
        _ => return Err(composed_schedule_error("consumer_topology_unmodeled", None)),
    };
    if root_bound.owner_terms.is_empty() {
        let output_term =
            consumer_owner_term(root, OwnerTermRole::ConsumerOutput, AllocationClass::ActivePhase, 0, consumer_plan)?;
        let output_term_id = output_term.id.clone();
        root_bound.owner_terms.push(output_term);
        root_bound.peak_owner_term_ids.push(output_term_id.clone());
        root_bound.final_owned_term_ids.push(output_term_id);
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

fn composed_consumer_branch_bound(
    plan: &QueryPlan,
    node_id: NodeId,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
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
                    if source.row_count_upper_bound.is_some_and(|rows| rows != output.row_count_upper_bound) {
                        return Err(composed_schedule_error(
                            "consumer_source_row_bound_mismatch",
                            Some(&source.input_name),
                        )
                        .with_fact("producer_row_count_upper_bound", output.row_count_upper_bound)
                        .with_fact(
                            "consumer_row_count_upper_bound",
                            source.row_count_upper_bound.unwrap_or_default(),
                        ));
                    }
                    // The source handle is retained by the producer schedule, not
                    // allocated by the consumer. Its exact typed bytes are therefore
                    // intentionally excluded from this branch's additional phase.
                    Ok(ComposedConsumerBranchBound {
                        rows: output.row_count_upper_bound,
                        schema: source.schema.clone(),
                        final_owned_bytes: 0,
                        additional_peak_bytes: 0,
                        owner_terms: Vec::new(),
                        peak_owner_term_ids: Vec::new(),
                        final_owned_term_ids: Vec::new(),
                        consumed_occurrences: Vec::new(),
                    })
                },
                SourceNode::LocalParquet(source) => {
                    let source_term = owner_terms
                        .iter()
                        .filter(|term| {
                            matches!(term.id.owner, OwnerId::Source(id) if id == node_id)
                                && term.id.role == OwnerTermRole::SourceStage
                                && term.id.occurrence == OccurrenceId::PlanLocal(node_id)
                                && term.execution_scope_fingerprint() == canonical_execution_scope_fingerprint(plan)
                        })
                        .collect::<Vec<_>>();
                    let [source_term] = source_term.as_slice() else {
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
                        final_owned_bytes: source_term.bytes(),
                        additional_peak_bytes: source_term.bytes(),
                        owner_terms: Vec::new(),
                        peak_owner_term_ids: vec![source_term.id().clone()],
                        final_owned_term_ids: vec![source_term.id().clone()],
                        consumed_occurrences: Vec::new(),
                    })
                },
                SourceNode::LiteralBatch(_) => Err(composed_schedule_error("consumer_non_graph_source", None)),
            }
        },
        PlanNode::Filter(filter) => {
            let input = composed_consumer_branch_bound(plan, filter.input, outputs, used_occurrences, owner_terms)?;
            let output_bytes = bounded_consumer_frame_bytes(input.rows, node.output_schema())?;
            // cuDF filter materializes the selected frame while a Boolean
            // predicate mask is live. Size the mask through the same typed
            // frame-bound primitive rather than treating it as host metadata.
            let predicate_mask_bytes = bounded_consumer_boolean_mask_bytes(input.rows)?;
            let stage_bytes = checked_consumer_add(
                checked_consumer_add(input.final_owned_bytes, output_bytes, "consumer_filter_output_overflow")?,
                predicate_mask_bytes,
                "consumer_filter_mask_overflow",
            )?;
            let mut owner_terms = input.owner_terms;
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
            let mut peak_owner_term_ids = input.peak_owner_term_ids.clone();
            let stage_peak = checked_consumer_add(
                checked_consumer_add(input.final_owned_bytes, output_bytes, "consumer_filter_output_overflow")?,
                predicate_mask_bytes,
                "consumer_filter_mask_overflow",
            )?;
            if stage_peak >= input.additional_peak_bytes {
                peak_owner_term_ids = input.final_owned_term_ids.clone();
                peak_owner_term_ids.push(output_term_id.clone());
                peak_owner_term_ids.push(predicate_term_id);
            }
            owner_terms.push(output_term);
            owner_terms.push(predicate_term);
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: output_bytes,
                additional_peak_bytes: input.additional_peak_bytes.max(stage_bytes),
                owner_terms,
                peak_owner_term_ids,
                final_owned_term_ids: vec![output_term_id],
                consumed_occurrences: Vec::new(),
            })
        },
        PlanNode::Projection(projection) => {
            let input = composed_consumer_branch_bound(plan, projection.input, outputs, used_occurrences, owner_terms)?;
            let mut selected_columns = HashSet::with_capacity(projection.outputs.len());
            for output in &projection.outputs {
                let Some(column) = alias_wrapped_column(&output.expr) else {
                    return Err(composed_schedule_error("consumer_projection_expression_workspace_unmodeled", None));
                };
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
            }
            // Column selection and rename retain their input columns. The
            // executor takes the same zero-copy path, so the projection owns
            // neither a fresh frame nor a duplicate of branch-owned data.
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: input.final_owned_bytes,
                additional_peak_bytes: input.additional_peak_bytes,
                owner_terms: input.owner_terms,
                peak_owner_term_ids: input.peak_owner_term_ids,
                final_owned_term_ids: input.final_owned_term_ids,
                consumed_occurrences: Vec::new(),
            })
        },
        _ => Err(composed_schedule_error("consumer_branch_topology_unmodeled", None)),
    }
}

fn composed_consumer_sort_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    sort: &crate::plan::SortNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
) -> Result<ComposedConsumerBranchBound> {
    let graph_result_input_names = plan
        .nodes()
        .iter()
        .filter(|node| !plan.unreachable_node_ids().contains(&node.id))
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let [graph_result_input_name] = graph_result_input_names.as_slice() else {
        return Err(composed_schedule_error("consumer_topology_unmodeled", None));
    };
    if !is_attested_graph_result_bounded_consumer_plan(plan, graph_result_input_name) {
        return Err(composed_schedule_error("consumer_topology_unmodeled", None));
    }
    let input = composed_consumer_branch_bound(plan, sort.input, outputs, used_occurrences, owner_terms)?;
    let scope = canonical_execution_scope_fingerprint(plan);
    let sort_terms = [
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
            matches!(
                term_id.role,
                OwnerTermRole::SortRetainedOrder | OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask
            )
        })
        .cloned()
        .collect();
    let output_schema = plan
        .node(node_id)
        .ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?
        .output_schema()
        .clone();
    let mut peak_owner_term_ids = input.peak_owner_term_ids;
    peak_owner_term_ids.extend(sort_terms);
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: output_schema,
        final_owned_bytes: 0,
        additional_peak_bytes: 0,
        owner_terms: input.owner_terms,
        peak_owner_term_ids,
        final_owned_term_ids,
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
) -> Result<ComposedConsumerBranchBound> {
    if join.spec.join_type != JoinType::Inner || join.spec.residual_predicate.is_some() || join.spec.keys.is_empty() {
        return Err(composed_schedule_error("consumer_join_topology_unmodeled", None));
    }
    let left = composed_consumer_branch_bound(plan, join.inputs[0], outputs, used_occurrences, owner_terms)?;
    let right = composed_consumer_branch_bound(plan, join.inputs[1], outputs, used_occurrences, owner_terms)?;
    let (build, probe, build_keys) = match join.spec.build_side {
        JoinSide::Left => (&left, &right, join.spec.keys.iter().map(|key| &key.left).collect::<Vec<_>>()),
        JoinSide::Right => (&right, &left, join.spec.keys.iter().map(|key| &key.right).collect::<Vec<_>>()),
    };
    let key_type_ids = build_keys
        .iter()
        .map(|key| non_nullable_fixed_width_cudf_type_id(&build.schema, key.name()))
        .collect::<Result<Vec<_>>>()?;
    // Probe keys must retain the same fixed-width non-null contract. This is
    // stricter than the metadata preflight's build requirement and prevents a
    // consumer proof from accepting a join whose runtime key conversion has no
    // typed allocation contract.
    for key in &join.spec.keys {
        let probe_key = match join.spec.build_side {
            JoinSide::Left => &key.right,
            JoinSide::Right => &key.left,
        };
        non_nullable_fixed_width_cudf_type_id(&probe.schema, probe_key.name())?;
    }
    let output_rows = left
        .rows
        .checked_mul(right.rows)
        .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?;
    let output_rows_usize = usize::try_from(output_rows)
        .map_err(|_| composed_schedule_error("consumer_join_output_rows_exceed_usize", None))?;
    let hash_table_bytes =
        cudf::ops::join::hash_join_pre_build_reservation_size_bytes_from_non_nullable_fixed_width_metadata(
            &key_type_ids,
            build.rows,
            cudf::DataFrameHashJoinOptions::new().load_factor_value(),
        )
        .map_err(|error| {
            composed_schedule_error("consumer_join_hash_preflight_unavailable", None)
                .with_fact("cudf_error", error.to_string())
        })?;
    let hash_table_bytes = u64::try_from(hash_table_bytes)
        .map_err(|_| composed_schedule_error("consumer_join_hash_preflight_exceeds_u64", None))?;
    let pair_indices_bytes = cudf::DataFrameHashJoinPairIndices::reservation_size_bytes_for_len(output_rows_usize)
        .map_err(|_| composed_schedule_error("consumer_join_pair_indices_preflight_unavailable", None))?;
    let pair_indices_bytes = u64::try_from(pair_indices_bytes)
        .map_err(|_| composed_schedule_error("consumer_join_pair_indices_exceed_u64", None))?;
    let output_bytes = bounded_consumer_frame_bytes(output_rows, plan.root_node().output_schema())?;
    let build_selected_bytes =
        bounded_selected_join_input_bytes(&build.schema, build.rows, join, join.spec.build_side)?;
    let probe_selected_bytes =
        bounded_selected_join_input_bytes(&probe.schema, probe.rows, join, opposite_join_side(join.spec.build_side))?;
    let peak = [
        left.additional_peak_bytes,
        right.additional_peak_bytes,
        build_selected_bytes,
        probe_selected_bytes,
        hash_table_bytes,
        pair_indices_bytes,
        output_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| checked_consumer_add(total, bytes, "consumer_join_peak_overflow"))?;
    let build_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerBuildInput,
        AllocationClass::ActivePhase,
        build_selected_bytes,
        plan,
    )?;
    let probe_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerProbeInput,
        AllocationClass::ActivePhase,
        probe_selected_bytes,
        plan,
    )?;
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
    let mut peak_owner_term_ids = left.peak_owner_term_ids;
    peak_owner_term_ids.extend(right.peak_owner_term_ids);
    peak_owner_term_ids.extend([
        build_term.id.clone(),
        probe_term.id.clone(),
        hash_term.id.clone(),
        pair_term.id.clone(),
        output_term.id.clone(),
    ]);
    let mut owner_terms = left.owner_terms;
    owner_terms.extend(right.owner_terms);
    owner_terms.extend([build_term, probe_term, hash_term, pair_term, output_term]);
    let final_owned_term_id = owner_terms.last().expect("join output term inserted").id.clone();
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: plan.root_node().output_schema().clone(),
        final_owned_bytes: output_bytes,
        additional_peak_bytes: peak,
        owner_terms,
        peak_owner_term_ids,
        final_owned_term_ids: vec![final_owned_term_id],
        consumed_occurrences: Vec::new(),
    })
}

fn bounded_selected_join_input_bytes(schema: &SchemaRef, rows: u64, join: &JoinNode, side: JoinSide) -> Result<u64> {
    let mut names = BTreeSet::new();
    for key in &join.spec.keys {
        let column = match side {
            JoinSide::Left => &key.left,
            JoinSide::Right => &key.right,
        };
        names.insert(column.name());
    }
    for column in join.spec.output_projection.iter().filter(|column| column.side == side) {
        names.insert(column.column.name());
    }
    if names.is_empty() {
        return Err(composed_schedule_error("consumer_join_selected_input_empty", None));
    }
    let fields = names
        .into_iter()
        .map(|name| {
            schema.field_with_name(name).map(|field| field.as_ref().clone()).map_err(|_| {
                composed_schedule_error("consumer_join_selected_column_missing", None).with_fact("column", name)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    bounded_consumer_frame_bytes(rows, &std::sync::Arc::new(arrow::datatypes::Schema::new(fields)))
}

fn opposite_join_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
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

fn non_nullable_fixed_width_cudf_type_id(schema: &SchemaRef, field_name: &str) -> Result<cudf::TypeId> {
    let field = schema
        .field_with_name(field_name)
        .map_err(|_| composed_schedule_error("consumer_join_key_missing", None).with_fact("column", field_name))?;
    if field.is_nullable() {
        return Err(composed_schedule_error("consumer_join_nullable_key", None).with_fact("column", field_name));
    }
    arrow_fixed_width_cudf_type_id(field.data_type()).ok_or_else(|| {
        composed_schedule_error("consumer_join_variable_width_key", None).with_fact("column", field_name)
    })
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

fn cugraph_side_input_frame_upper_bound_bytes(plan: &QueryPlan) -> Result<u64> {
    let rows = query_plan_hard_row_upper_bound(plan).ok_or_else(|| {
        composed_schedule_error("side_input_row_bound_unavailable", None)
            .with_fact("phase", "side_input_frame")
            .with_fact("row_bound_source", "native_query_plan")
    })?;
    if plan.output_schema().fields().is_empty() {
        return Err(composed_schedule_error("side_input_schema_empty", None).with_fact("phase", "side_input_frame"));
    }
    let columns = plan
        .output_schema()
        .fields()
        .iter()
        .map(|field| {
            fixed_width_consumer_output_column(field.data_type()).ok_or_else(|| {
                composed_schedule_error("side_input_schema_unmodeled", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
            })
        })
        .collect::<Result<Vec<_>>>()?;
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

fn fixed_width_consumer_output_column(data_type: &DataType) -> Option<crate::exec::JoinOutputColumnUpperBoundInput> {
    matches!(
        data_type,
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
            | DataType::Decimal128(_, _)
    )
    .then(|| crate::exec::JoinOutputColumnUpperBoundInput {
        data_type: data_type.clone(),
        input_max_string_length_bytes: None,
        input_total_string_bytes: None,
    })
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
    if !(key_type.is_signed_int() || key_type.is_unsigned_int() || key_type.is_timestamp() || key_type.is_duration()) {
        return Err(sort_owner_error("key_type_not_integral_or_chrono", "sort_key"));
    }
    if descriptor.num_rows > i32::MAX as usize {
        return Err(sort_owner_error("row_count_exceeds_cudf_size_type", "sort_input"));
    }
    FiniteBytes::new(descriptor.output_frame_bytes)
        .map_err(|_| sort_owner_error("sort_output_bytes_saturated", "sort_output"))?;
    FiniteBytes::new(descriptor.output_null_mask_bytes)
        .map_err(|_| sort_owner_error("sort_output_null_mask_saturated", "sort_output_null_mask"))?;
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
    let Some((_, source_node)) = source else {
        return Err(sort_owner_error("graph_result_source_missing", "sort_input"));
    };
    let SourceNode::GpuResident(source_node) = source_node else {
        return Err(sort_owner_error("graph_result_source_not_gpu_resident", "sort_input"));
    };
    let Some((sort_node_id, sort_node)) = sort else {
        return Err(sort_owner_error("sort_node_missing", "sort_plan"));
    };
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
        || key_type_id.is_duration())
    {
        return Err(
            sort_owner_error("key_type_not_integral_or_chrono", "sort_key").with_fact("key_type", key_type_id.name())
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
        .or_else(|| query_plan_hard_row_upper_bound(plan))
        .ok_or_else(|| sort_owner_error("row_bound_missing", "sort_input"))?;
    let num_rows = usize::try_from(row_count).map_err(|_| sort_owner_error("row_bound_exceeds_usize", "sort_input"))?;
    let (output_frame_bytes, output_null_mask_bytes) =
        bounded_consumer_frame_components(row_count, plan.root_node().output_schema())?;
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
    let mut sort_count = 0_usize;
    let mut filter_count = 0_usize;
    for node in plan.nodes() {
        if plan.unreachable_node_ids().contains(&node.id) {
            continue;
        }
        match &node.kind {
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name == graph_result_input_name) =>
            {
                graph_result_source_count += 1;
            },
            PlanNode::Sort(_) => {
                sort_count += 1;
            },
            PlanNode::Filter(_) => {
                filter_count += 1;
            },
            _ => return false,
        }
    }
    graph_result_source_count == 1 && sort_count == 1 && filter_count <= 1
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        capability::{ContractQuality, ReservationRequirement},
        chunk::SourceChunkPlanningOptions,
        expr::{Expr, SortExpr, SortSpec},
        graph_memory::{GraphMemoryCalibrationOverride, graph_memory_bounded_admission_evidence_for_plan},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphConstructionPolicy, GraphEdgeColumnBindings, JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
            ProjectionOutput, QueryPlan, QueryPlanBuilder,
        },
        runtime::{ReservationKind, SourceChunkPlanningMode, SourceSchedulerPolicy},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    fn capability() -> ReservationRequirement {
        ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "test",
            "test_contract",
            ContractQuality::CapabilityProof,
        )
    }

    fn graph_bfs_plan() -> QueryPlan {
        graph_bfs_plan_with_source_vertex(0)
    }

    fn graph_bfs_plan_with_source_vertex(source_vertex: i64) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("edges", schema).with_row_count_upper_bound(20),
        ));
        let graph = builder.add_graph_algorithm(
            edges,
            GraphAlgorithmSpec::with_stable_schema(
                GraphAlgorithm::Bfs,
                GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
                GraphConstructionOptions::default(),
                GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(source_vertex, None))),
            ),
        );
        builder.finish_with_host_sink(graph)
    }

    fn ordinary_native_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        builder.finish_with_host_sink(source)
    }

    #[test]
    fn non_sort_requirement_does_not_validate_candidate_device() {
        let plan = ordinary_native_plan();
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("ordinary plan must issue a proof");
        let resolved = proof
            .requirement_for_device(usize::MAX)
            .expect("non-sort requirements do not need a CUDA device for resolution");

        assert_eq!(resolved.as_requirement(), proof.as_requirement());
        assert!(resolved.as_requirement().reservation_shapes.iter().all(|shape| shape.sort_metadata().is_none()));
    }

    fn graph_bfs_proof_with_typed_output(source_vertex: i64) -> QueryPlanCapabilityProof {
        let plan = graph_bfs_plan_with_source_vertex(source_vertex);
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("test graph plan must produce typed admission evidence");
        QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("test graph plan must issue a capability proof")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence")
    }

    fn relational_gpu_producer_proof() -> QueryPlanCapabilityProof {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/requirement-relational-producer.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_rows(vec![8])
                    .with_row_group_bytes(vec![128])
                    .with_projected_row_group_bytes(vec![64], vec![128]),
            ],
            schema,
            vec!["value".to_owned()],
        )));
        let plan = builder.finish_gpu_resident(source);
        QueryPlanCapabilityProof::for_query_plan(&plan).expect("bounded relational GPU producer must issue a proof")
    }

    fn graph_output_union_consumer(input_names: &[&str]) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let inputs = input_names
            .iter()
            .map(|name| builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(*name, Arc::clone(&schema)))))
            .collect();
        let root = builder.add_union(inputs, schema);
        builder.finish_with_host_sink(root)
    }

    fn mixed_local_graph_union_consumer(source: LocalParquetSource) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let local = builder.add_source(SourceNode::LocalParquet(source));
        let graph = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("graph", Arc::clone(&schema))));
        let root = builder.add_union(vec![local, graph], schema);
        builder.finish_with_host_sink(root)
    }

    fn mixed_local_graph_union_source(row_count: Option<u64>, bytes: Option<u64>) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut file = LocalParquetFile::new("/tmp/requirement-mixed-composition.parquet");
        if let Some(row_count) = row_count {
            file = file.with_row_group_rows(vec![row_count]);
        }
        if let Some(bytes) = bytes {
            file = file.with_projected_row_group_bytes(vec![bytes / 2], vec![bytes]);
        }
        LocalParquetSource::from_files(vec![file], schema, vec!["value".to_owned()])
    }

    fn graph_output_projection_consumer(expr: Expr, output_type: DataType) -> QueryPlan {
        let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("only", input_schema)));
        let projection = builder
            .add_projection(source, vec![ProjectionOutput::new(expr, Field::new("projected", output_type, false))]);
        builder.finish_with_host_sink(projection)
    }

    fn graph_output_filter_consumer(input_name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&schema))));
        let filter = builder.add_filter(source, Expr::column("value"), schema);
        builder.finish_with_host_sink(filter)
    }

    fn graph_output_inner_join_consumer(
        key_type: DataType,
        key_nullable: bool,
        include_unary_branch: bool,
    ) -> QueryPlan {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("vertex", key_type.clone(), key_nullable),
            Field::new("value", DataType::Float64, false),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("left_vertex", key_type.clone(), key_nullable),
            Field::new("left_value", DataType::Float64, false),
            Field::new("right_value", DataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let left =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&input_schema))));
        let projected_left = if include_unary_branch {
            let filtered_left = builder.add_filter(left, Expr::column("vertex"), Arc::clone(&input_schema));
            builder.add_projection(
                filtered_left,
                vec![
                    ProjectionOutput::new(Expr::column("vertex"), input_schema.field(0).as_ref().clone()),
                    ProjectionOutput::new(Expr::column("value"), input_schema.field(1).as_ref().clone()),
                ],
            )
        } else {
            left
        };
        let right =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&input_schema))));
        let spec = JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                crate::expr::ColumnRef::input(0, "vertex", "vertex"),
                crate::expr::ColumnRef::input(1, "vertex", "vertex"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "vertex", "vertex"), "left_vertex"),
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "value", "value"), "left_value"),
            JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "value", "value"), "right_value"),
        ]);
        let join = builder.add_join(projected_left, right, spec, output_schema);
        builder.finish_with_host_sink(join)
    }

    fn reservation_policy() -> AttemptReservationPolicy {
        AttemptReservationPolicy::new(
            64 * 1024,
            1,
            3,
            1,
            SourceSchedulerPolicy::new(2),
            SourceChunkPlanningMode::default(),
            SourceChunkPlanningOptions::default(),
        )
    }

    #[test]
    fn known_local_parquet_sources_keep_a_separate_modeled_source_stage() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/requirement-known.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![100])
                    .with_projected_row_group_bytes(vec![40], vec![96]),
            ],
            schema,
            vec!["value".to_owned()],
        )));
        let plan = builder.finish_with_host_sink(source);

        let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
        assert_eq!(requirement.source_stage_bytes, Some(96));
        assert_eq!(requirement.soft_working_set_hint_bytes(), None);
        assert_eq!(requirement.estimate_quality(), EstimateQuality::Modeled);
    }

    #[test]
    fn unknown_local_parquet_sources_reject_before_envelope_resolution() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/requirement-unknown.parquet",
            schema,
            vec!["value".to_owned()],
        )));
        let plan = builder.finish_with_host_sink(source);

        let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
        assert_eq!(requirement.source_stage_bytes, None);
        assert_eq!(requirement.soft_working_set_hint_bytes(), None);
        assert_eq!(requirement.estimate_quality(), EstimateQuality::Unknown);
        let proof = QueryPlanCapabilityProof { requirement };
        let error = proof
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("unknown local source must fail before placement");
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_local_source_bytes_unavailable"));
    }

    #[test]
    fn mixed_local_and_non_local_sources_reject_before_envelope_resolution() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let local = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/requirement-mixed.parquet").with_file_size_bytes(96)],
            Arc::clone(&schema),
            vec!["value".to_owned()],
        )));
        let gpu = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("gpu", Arc::clone(&schema))));
        let root = builder.add_union(vec![local, gpu], schema);
        let plan = builder.finish_with_host_sink(root);

        let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
        let error = QueryPlanCapabilityProof { requirement }
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("mixed sources must fail before placement");
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_mixed_source_kinds"));
    }

    #[test]
    fn composed_union_reuses_exact_local_source_stage_alongside_gpu_producer() {
        let consumer_plan = mixed_local_graph_union_consumer(mixed_local_graph_union_source(Some(3), Some(64)));
        let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("composition policy issues the mixed local/GPU consumer proof");
        let direct_error = consumer
            .clone()
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("a mixed consumer cannot resolve before a retained producer is bound");
        assert_eq!(direct_error.fact_value("schedule_reason"), Some("consumer_producer_binding_required"));
        let aggregated = AttemptCapabilityEvidence::aggregate([consumer.clone().into_requirement()]).unwrap();
        let aggregate_error = QueryPlanCapabilityProof { requirement: aggregated }
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("generic aggregation must preserve the incomplete composition marker");
        assert_eq!(aggregate_error.fact_value("schedule_reason"), Some("consumer_producer_binding_required"));
        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            consumer,
            [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
        )
        .expect("the exact local branch and one GPU producer must compose");

        let source_terms = combined
            .as_requirement()
            .owner_terms
            .iter()
            .filter(|term| term.id.role == OwnerTermRole::SourceStage)
            .collect::<Vec<_>>();
        assert_eq!(source_terms.len(), 1, "composition must reuse one canonical SourceStage owner");
        assert_eq!(source_terms[0].bytes(), 64);
        let consumer_shape = combined
            .as_requirement()
            .reservation_shapes
            .iter()
            .find(|shape| shape.matches(&consumer_plan))
            .expect("consumer scope is retained");
        assert_eq!(consumer_shape.retained_input_bindings().len(), 1);
        assert_eq!(consumer_shape.retained_input_bindings()[0].0, "graph");
        let envelope = combined
            .resolve_reservation_envelope(reservation_policy())
            .expect("mixed local/GPU composition has one finite owner inventory");
        assert!(envelope.complete_phase_peak_bytes().is_some());
    }

    #[test]
    fn composed_union_rejects_unknown_local_rows_or_bytes_before_placement() {
        let unknown_rows = mixed_local_graph_union_consumer(mixed_local_graph_union_source(None, Some(64)));
        let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&unknown_rows).unwrap();
        let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &unknown_rows,
            consumer,
            [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
        )
        .expect_err("missing local footer rows cannot be turned into a consumer bound");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_local_source_rows_unavailable"));

        let unknown_bytes = mixed_local_graph_union_consumer(mixed_local_graph_union_source(Some(3), None));
        let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&unknown_bytes).unwrap();
        let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &unknown_bytes,
            consumer,
            [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
        )
        .expect_err("missing local source bytes cannot be turned into a consumer bound");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_local_source_owner_missing"));
    }

    #[test]
    fn composed_consumer_rejects_literal_source_mixed_with_gpu_producer() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let local = builder.add_source(SourceNode::LocalParquet(mixed_local_graph_union_source(Some(3), Some(64))));
        let literal_batch = arrow::record_batch::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(arrow::array::Int64Array::from(vec![1]))],
        )
        .unwrap();
        let literal = builder.add_source(SourceNode::LiteralBatch(crate::source::LiteralBatchSource::new(
            Arc::clone(&schema),
            vec![literal_batch],
        )));
        let root = builder.add_union(vec![local, literal], schema);
        let plan = builder.finish_with_host_sink(root);
        let proof = QueryPlanCapabilityProof::for_composed_consumer_plan(&plan).unwrap();
        let error = proof.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_literal_source_bytes_unavailable"));
    }

    #[test]
    fn saturated_local_source_rejects_before_owner_term_insertion() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/requirement-saturated.parquet").with_file_size_bytes(u64::MAX)],
            schema,
            vec!["value".to_owned()],
        )));
        let plan = builder.finish_with_host_sink(source);

        let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
        assert!(requirement.owner_terms.is_empty());
        let error = QueryPlanCapabilityProof { requirement }
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("saturated local source must fail before placement");
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_saturated"));
    }

    #[test]
    fn plan_issued_source_terms_reject_aggregate_source_stage_overflow() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let plan_for_bytes = |path: &str, bytes| {
            let mut builder = QueryPlanBuilder::new();
            let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
                vec![LocalParquetFile::new(path).with_file_size_bytes(bytes)],
                Arc::clone(&schema),
                vec!["value".to_owned()],
            )));
            builder.finish_with_host_sink(source)
        };
        let first = QueryPlanCapabilityProof::for_query_plan(&plan_for_bytes(
            "/tmp/requirement-overflow-first.parquet",
            u64::MAX - 1,
        ))
        .expect("finite source must issue a proof");
        let second =
            QueryPlanCapabilityProof::for_query_plan(&plan_for_bytes("/tmp/requirement-overflow-second.parquet", 2))
                .expect("finite source must issue a proof");
        let error = QueryPlanCapabilityProof::combine(first, [second])
            .expect_err("aggregate source-stage overflow must reject");
        assert_eq!(error.fact_value("reason"), Some("attempt_requirement_source_stage_overflow"));
    }

    #[test]
    fn zero_column_count_source_resolves_a_parquet_read_floor_without_a_partial_target() {
        let source = LocalParquetSource::zero_column_count(vec![
            LocalParquetFile::new("/tmp/requirement-count.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![4_096])
                .with_projected_row_group_bytes(vec![0], vec![0]),
        ]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let plan = builder.finish_with_host_sink(source);

        let minimum_read_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("sink plan must issue a proof");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        let requirement = proof.into_requirement().with_reservation_envelope(&envelope);

        assert_eq!(requirement.source_stage_bytes, Some(0));
        assert_eq!(envelope.required_phase_floor_bytes(), minimum_read_budget);
        assert_eq!(envelope.complete_phase_peak_bytes(), None);
        assert_eq!(requirement.soft_working_set_hint_bytes(), None);
        assert_eq!(requirement.required_floor_bytes_for(1).unwrap(), minimum_read_budget);
    }

    #[test]
    fn graph_evidence_does_not_double_count_its_prepared_local_source_frame() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/retained-graph-source.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_rows(vec![10])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&schema),
            vec!["src".to_owned(), "dst".to_owned()],
        )));
        let graph = builder.add_graph_algorithm(source, spec.clone());
        let plan = builder.finish_with_host_sink(graph);
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue an engine-owned capability proof");

        assert_eq!(proof.soft_working_set_hint_bytes(), None);
        assert_eq!(proof.as_requirement().source_stage_bytes, Some(96));
        assert_eq!(
            proof.with_graph_memory_admission_evidence(&evidence).unwrap().soft_working_set_hint_bytes(),
            Some(96_u64.max(evidence.cold_model_requested_bytes())),
            "the graph cold peak already owns the prepared edge frame"
        );
    }

    #[test]
    fn graph_evidence_adds_a_source_stage_that_normalizes_into_a_separate_graph_input() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        )
        .with_edge_normalize_inserted(true);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/normalized-graph-source.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_rows(vec![10])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&schema),
            vec!["src".to_owned(), "dst".to_owned()],
        )));
        let graph = builder.add_graph_algorithm(source, spec.clone());
        let plan = builder.finish_with_host_sink(graph);
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue an engine-owned capability proof");

        assert_eq!(proof.soft_working_set_hint_bytes(), None);
        assert_eq!(proof.as_requirement().source_stage_bytes, Some(96));
        assert_eq!(
            proof.with_graph_memory_admission_evidence(&evidence).unwrap().soft_working_set_hint_bytes(),
            Some(96_u64.saturating_add(evidence.cold_model_requested_bytes())),
            "the explicit normalize source and derived graph input are separate live owners"
        );
    }

    #[test]
    fn graph_evidence_without_a_source_stage_remains_the_native_peak() {
        let plan = graph_bfs_plan();
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .unwrap();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue an engine-owned capability proof");

        assert_eq!(proof.soft_working_set_hint_bytes(), None);
        assert_eq!(
            proof.with_graph_memory_admission_evidence(&evidence).unwrap().soft_working_set_hint_bytes(),
            Some(evidence.cold_model_requested_bytes())
        );
    }

    #[test]
    fn graph_memory_evidence_attachment_rejects_missing_ambiguous_and_mismatched_scopes_atomically() {
        let plan = graph_bfs_plan();
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
                .unwrap();

        let missing = QueryPlanCapabilityProof::for_query_plan(&plan).expect("native plan must issue a proof");
        let missing_before = missing.clone();
        let error = missing
            .clone()
            .with_graph_memory_admission_evidence(&evidence)
            .expect_err("generic proofs do not attest a graph execution scope");
        assert_eq!(error.fact_value("reason"), Some("graph_memory_evidence_scope_missing"));
        assert_eq!(missing, missing_before);

        let second_plan = graph_bfs_plan_with_source_vertex(1);
        let ambiguous = QueryPlanCapabilityProof::combine(
            QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap(),
            [QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&second_plan).unwrap()],
        )
        .unwrap();
        let ambiguous_before = ambiguous.clone();
        let error = ambiguous
            .clone()
            .with_graph_memory_admission_evidence(&evidence)
            .expect_err("multiple attested graph scopes are ambiguous");
        assert_eq!(error.fact_value("reason"), Some("graph_memory_evidence_scope_ambiguous"));
        assert_eq!(ambiguous, ambiguous_before);

        let mismatched_evidence = graph_memory_bounded_admission_evidence_for_plan(
            &second_plan,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .unwrap();
        let mismatched = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap();
        let mismatched_before = mismatched.clone();
        let error = mismatched
            .clone()
            .with_graph_memory_admission_evidence(&mismatched_evidence)
            .expect_err("evidence must match the attested graph plan");
        assert_eq!(error.fact_value("reason"), Some("graph_memory_evidence_scope_mismatch"));
        assert_eq!(mismatched, mismatched_before);
    }

    #[test]
    fn required_floor_uses_checked_arithmetic() {
        let requirement =
            AttemptCapabilityEvidence::from_parts(u64::MAX, 1, None, EstimateQuality::Unknown, Vec::new());
        assert_eq!(
            requirement.required_floor_bytes().unwrap_err().fact_value("reason"),
            Some("attempt_requirement_floor_overflow")
        );
    }

    #[test]
    fn legacy_band_is_an_interim_floor_and_hint() {
        let requirement = AttemptCapabilityEvidence::from_parts(10, 0, Some(20), EstimateQuality::Modeled, Vec::new());
        assert_eq!(requirement.required_floor_bytes().unwrap(), 10);
        assert_eq!(requirement.soft_working_set_hint_bytes, Some(20));
        assert_eq!(requirement.estimate_quality, EstimateQuality::Modeled);
    }

    #[test]
    fn aggregate_sums_only_raises_and_unions_capabilities() {
        let capability = capability();
        let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
            10,
            [
                AttemptCapabilityEvidence::from_parts(
                    99,
                    2,
                    Some(12),
                    EstimateQuality::Measured,
                    vec![capability.clone()],
                ),
                AttemptCapabilityEvidence::from_parts(88, 3, Some(20), EstimateQuality::Unknown, vec![capability]),
            ],
        )
        .unwrap();
        assert_eq!(result.configured_floor_bytes, 10);
        assert_eq!(result.capability_floor_raise_bytes, 5);
        assert_eq!(result.soft_working_set_hint_bytes, Some(20));
        assert_eq!(result.estimate_quality, EstimateQuality::Unknown);
        assert_eq!(result.required_capabilities.len(), 1);
    }

    #[test]
    fn aggregate_clears_the_target_when_any_fragment_is_unmodeled() {
        let result = AttemptCapabilityEvidence::aggregate([
            AttemptCapabilityEvidence::from_parts(0, 0, Some(12), EstimateQuality::Modeled, Vec::new()),
            AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new()),
        ])
        .unwrap();

        assert_eq!(result.soft_working_set_hint_bytes(), None);
    }

    #[test]
    fn aggregate_preserves_absent_source_stage_for_a_gpu_only_proof() {
        let proof =
            QueryPlanCapabilityProof::for_query_plan(&ordinary_native_plan()).expect("native plan issues a proof");
        let aggregate = QueryPlanCapabilityProof::combine(proof.clone(), std::iter::empty())
            .expect("one native proof still passes through the aggregate seam");

        assert_eq!(aggregate.as_requirement().source_stage_bytes, None);
        assert_eq!(aggregate, proof, "aggregation must not manufacture a zero local-Parquet stage");
    }

    #[test]
    fn generic_gpu_sink_with_a_gpu_source_rejects_an_unbound_output_owner() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let plan = builder.finish_gpu_resident(source);
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
        let error = proof
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("a GPU-source sink cannot mint an unbound generic retained output");
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_gpu_resident_output_owner_alias_unmodeled"));
    }

    #[test]
    fn cugraph_side_input_disables_generic_gpu_output_ownership() {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("predicate", Arc::clone(&schema)).with_row_count_upper_bound(2),
        ));
        let plan = builder.finish_gpu_resident(source);

        let proof = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&plan)
            .expect("a bounded GPU-resident side input must issue a proof");
        assert!(
            proof.as_requirement().owner_terms.iter().all(|term| !matches!(term.id().owner, OwnerId::Producer(_))),
            "the side-input frame must be the sole owner for this native fragment"
        );
        assert_eq!(proof.as_requirement().side_input_frame_owner_term_bytes().len(), 1);
    }

    #[test]
    fn aggregate_rejects_raise_overflow() {
        let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
            0,
            [
                AttemptCapabilityEvidence::empty(),
                AttemptCapabilityEvidence::from_parts(0, u64::MAX, None, EstimateQuality::Unknown, Vec::new()),
                AttemptCapabilityEvidence::from_parts(0, 1, None, EstimateQuality::Unknown, Vec::new()),
            ],
        );
        assert_eq!(result.unwrap_err().fact_value("reason"), Some("attempt_requirement_capability_raise_overflow"));
    }

    fn finite_test_owner_terms(plan: &QueryPlan, first: u64, second: u64) -> Vec<AllocationOwnerTerm> {
        let scope = canonical_execution_scope_fingerprint(plan);
        vec![
            AllocationOwnerTerm::new(
                OwnerId::Source(plan.root_id()),
                OwnerTermRole::SourceStage,
                OccurrenceId::PlanLocal(plan.root_id()),
                scope.clone(),
                AllocationClass::ActivePhase,
                first,
                None,
            )
            .unwrap(),
            AllocationOwnerTerm::new(
                OwnerId::Consumer(plan.root_id()),
                OwnerTermRole::ConsumerOutput,
                OccurrenceId::PlanLocal(plan.root_id()),
                scope,
                AllocationClass::ActivePhase,
                second,
                None,
            )
            .unwrap(),
        ]
    }

    #[test]
    fn conservative_owner_sum_charges_each_shared_inventory_term_once() {
        let plan = ordinary_native_plan();
        let mut evidence = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
        evidence.owner_terms = finite_test_owner_terms(&plan, 11, 7);
        evidence.reservation_shapes.push(AttemptReservationShape::for_query_plan(&plan));
        assert_eq!(evidence.conservative_owner_sum().unwrap(), 18);
        let aggregate = AttemptCapabilityEvidence::aggregate([evidence]).unwrap();
        assert_eq!(aggregate.bound_derivation(), None, "provenance is selected when the envelope resolves");
    }

    #[test]
    fn conservative_derivation_accepts_a_complete_zero_byte_inventory_but_not_missing_coverage() {
        let plan = ordinary_native_plan();
        let mut complete = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
        complete.owner_terms = finite_test_owner_terms(&plan, 0, 0);
        complete.reservation_shapes.push(AttemptReservationShape::for_query_plan(&plan));
        let complete = AttemptCapabilityEvidence::aggregate([complete]).unwrap();
        assert_eq!(complete.bound_derivation(), None, "provenance is selected when the envelope resolves");

        let mut missing = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
        missing.owner_terms = vec![finite_test_owner_terms(&plan, 0, 0).remove(0)];
        missing.reservation_shapes.push(AttemptReservationShape::for_query_plan(&plan));
        let missing = AttemptCapabilityEvidence::aggregate([missing]).unwrap();
        assert_eq!(missing.bound_derivation(), None);
    }

    #[test]
    fn proof_envelope_pins_one_producer_peak_to_fixed_plus_output_plus_consumer() {
        let consumer_plan = graph_output_filter_consumer("producer");
        let producer = graph_bfs_proof_with_typed_output(0);
        let combined = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("producer".to_owned()),
                Some("wrong-producer-name".to_owned()),
                Some(producer),
            )],
        )
        .expect("the conservative production seam must retain a complete owner inventory");

        assert_eq!(combined.as_requirement().bound_derivation(), None);
        let retained_terms =
            combined.as_requirement().owner_terms.iter().filter(|term| term.is_retained_output()).collect::<Vec<_>>();
        assert_eq!(retained_terms.len(), 1, "one producer occurrence owns one retained output");
        let consumer_shapes = combined
            .as_requirement()
            .reservation_shapes
            .iter()
            .filter(|shape| shape.matches(&consumer_plan))
            .collect::<Vec<_>>();
        let [consumer_shape] = consumer_shapes.as_slice() else {
            panic!("consumer scope must be unique in the production composition");
        };
        assert_eq!(consumer_shape.retained_input_bindings().len(), 1);
        assert_eq!(consumer_shape.retained_input_bindings()[0].1, retained_terms[0].id().clone());
        assert!(
            combined
                .as_requirement()
                .owner_terms
                .iter()
                .all(|term| term.id().role != OwnerTermRole::ConsumerBuildInput),
            "the consumer source handle is retained by the producer, not allocated as a second input frame"
        );

        let expected_owner_sum = combined
            .as_requirement()
            .owner_terms
            .iter()
            .map(AllocationOwnerTerm::bytes)
            .try_fold(0_u64, u64::checked_add)
            .expect("test owner inventory fits");
        let envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_owner_sum));
    }

    #[test]
    fn conservative_owner_sum_rejects_duplicate_saturated_and_overflow_terms() {
        let plan = ordinary_native_plan();
        let terms = finite_test_owner_terms(&plan, 11, 7);
        let mut duplicate = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
        duplicate.owner_terms = vec![terms[0].clone(), terms[0].clone()];
        assert_eq!(
            duplicate.conservative_owner_sum().unwrap_err().fact_value("reason"),
            Some("attempt_owner_term_duplicate")
        );

        let mut saturated = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
        let mut saturated_term = terms[0].clone();
        saturated_term.bytes = FiniteBytes(u64::MAX);
        saturated.owner_terms = vec![saturated_term];
        assert_eq!(
            saturated.conservative_owner_sum().unwrap_err().fact_value("reason"),
            Some("attempt_owner_term_saturated")
        );

        let mut overflowing = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
        overflowing.owner_terms = finite_test_owner_terms(&plan, u64::MAX - 1, 2);
        assert_eq!(
            overflowing.conservative_owner_sum().unwrap_err().fact_value("reason"),
            Some("attempt_owner_term_sum_overflow")
        );
    }

    #[test]
    fn cugraph_graph_scope_rejects_an_unattested_native_plan_with_stable_facts() {
        let attested_plan = graph_bfs_plan_with_source_vertex(1);
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&attested_plan)
            .expect("attested graph execution has capability requirements");
        let error = proof
            .as_requirement()
            .validate_covers_plan(&graph_bfs_plan())
            .expect_err("a plan-scoped cuGraph proof must not authorize a different native graph plan");

        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
        assert_eq!(
            error.fact_value("missing_capabilities"),
            Some("operator_output:unsupported_graph_algorithm_reservation_contract")
        );
    }

    #[test]
    fn exact_cugraph_plan_proof_replaces_only_the_engine_graph_blocker() {
        let plan = graph_bfs_plan();
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph execution has capability requirements");

        assert!(proof.required_capabilities().iter().any(|requirement| {
            requirement.kind == ReservationKind::GraphAlgorithm
                && requirement.required_contract == "cugraph_graph_execution_contract"
        }));
        assert!(!proof.required_capabilities().iter().any(is_generic_graph_execution_blocker));
        assert_eq!(proof.as_requirement().attested_cugraph_graph_execution_scopes.len(), 1);
        proof
            .as_requirement()
            .validate_covers_plan(&plan)
            .expect("the issued cuGraph proof must cover its exact internal plan");
    }

    #[test]
    fn combining_cugraph_proofs_retains_each_attested_graph_scope() {
        let native_plan = ordinary_native_plan();
        let first_plan = graph_bfs_plan_with_source_vertex(0);
        let second_plan = graph_bfs_plan_with_source_vertex(1);
        let unscoped_plan = graph_bfs_plan_with_source_vertex(2);
        let proof = QueryPlanCapabilityProof::combine(
            QueryPlanCapabilityProof::for_query_plan(&native_plan)
                .expect("native plan has an explicit sink capability requirement"),
            [
                QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&first_plan)
                    .expect("first graph plan has capability requirements"),
                QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&second_plan)
                    .expect("second graph plan has capability requirements"),
            ],
        )
        .expect("native and plan-scoped graph proofs must combine");

        let requirement = proof.as_requirement();
        assert_eq!(requirement.attested_cugraph_graph_execution_scopes.len(), 2);
        requirement.validate_covers_plan(&native_plan).expect("combined proof must retain the native plan capability");
        requirement.validate_covers_plan(&first_plan).expect("combined proof must retain the first graph scope");
        requirement.validate_covers_plan(&second_plan).expect("combined proof must retain the second graph scope");
        let error = requirement
            .validate_covers_plan(&unscoped_plan)
            .expect_err("combined proof must not widen its graph scope");
        assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
        assert_eq!(
            error.fact_value("missing_capabilities"),
            Some("operator_output:unsupported_graph_algorithm_reservation_contract")
        );
    }

    #[test]
    fn graph_owner_terms_reject_duplicate_standalone_proofs() {
        let proof = graph_bfs_proof_with_typed_output(0);
        let graph_persistent = proof
            .as_requirement()
            .owner_terms
            .iter()
            .find(|term| term.id.role == OwnerTermRole::GraphPersistent)
            .expect("graph proof carries a persistent graph owner");
        assert_eq!(graph_persistent.class, AllocationClass::ActivePhase);
        let duplicate =
            AttemptCapabilityEvidence::aggregate([proof.as_requirement().clone(), proof.as_requirement().clone()])
                .expect_err("standalone graph terms with one plan-local occurrence cannot be deduplicated");
        assert_eq!(duplicate.fact_value("reason"), Some("attempt_owner_term_duplicate"));
        assert_eq!(duplicate.fact_value("owner_term_role"), Some("graph_persistent"));
        assert!(duplicate.fact_value("occurrence").is_some_and(|value| value.starts_with("plan_local:n")));
    }

    #[test]
    fn source_graph_and_consumer_terms_are_scoped_and_typed_in_one_inventory() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut source_builder = QueryPlanBuilder::new();
        let source_id = source_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/inventory-source.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            Arc::clone(&schema),
            vec!["value".to_owned()],
        )));
        let source_plan = source_builder.finish_with_host_sink(source_id);
        let source_proof = QueryPlanCapabilityProof::for_query_plan(&source_plan).unwrap();
        let source_requirement = source_proof.as_requirement();
        let source_term = source_requirement
            .owner_terms
            .iter()
            .find(|term| matches!(term.id.owner, OwnerId::Source(id) if id == source_id))
            .expect("known local source has one NodeId-keyed owner term");
        assert_eq!(source_term.id.role, OwnerTermRole::SourceStage);
        assert_eq!(source_term.class, AllocationClass::ActivePhase);
        assert_eq!(source_term.id.execution_scope_fingerprint, canonical_execution_scope_fingerprint(&source_plan));

        let graph_plan = graph_bfs_plan();
        let graph_evidence =
            graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
                .unwrap();
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .unwrap()
            .with_graph_memory_admission_evidence(&graph_evidence)
            .unwrap();
        assert!(graph_proof.as_requirement().owner_terms.iter().any(|term| {
            matches!(term.id.owner, OwnerId::Graph(_))
                && term.id.execution_scope_fingerprint == canonical_execution_scope_fingerprint(&graph_plan)
        }));

        let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, true);
        let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
                ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
            ],
        )
        .unwrap();
        assert!(composed.as_requirement().owner_terms.iter().any(|term| {
            matches!(term.id.owner, OwnerId::Consumer(_))
                && term.id.execution_scope_fingerprint == canonical_execution_scope_fingerprint(&consumer_plan)
        }));
    }

    #[test]
    fn composed_schedule_keeps_zero_fixed_projection_without_a_fake_owner_term() {
        let consumer_plan = graph_output_union_consumer(&["fixed-left", "fixed-right"]);
        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ("fixed-left".to_owned(), graph_bfs_proof_with_typed_output(0)),
                ("fixed-right".to_owned(), graph_bfs_proof_with_typed_output(1)),
            ],
        )
        .unwrap();
        assert!(combined.as_requirement().owner_terms.iter().all(|term| term.bytes.get() != 0));
    }

    #[test]
    fn attempt_owner_term_rejects_mismatched_owner_role_and_class() {
        let plan = graph_output_union_consumer(&["fixed-shape"]);
        AllocationOwnerTerm::new(
            OwnerId::Attempt,
            OwnerTermRole::AttemptWideFixed,
            OccurrenceId::PlanLocal(plan.root_id()),
            canonical_execution_scope_fingerprint(&plan),
            AllocationClass::AttemptWideFixed,
            1,
            None,
        )
        .expect("attempt owner terms accept the dedicated fixed role and class");
        let error = AllocationOwnerTerm::new(
            OwnerId::Attempt,
            OwnerTermRole::GraphPersistent,
            OccurrenceId::PlanLocal(plan.root_id()),
            canonical_execution_scope_fingerprint(&plan),
            AllocationClass::ActivePhase,
            1,
            None,
        )
        .expect_err("attempt owner terms must use the dedicated fixed role and class");
        assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_owner_role_class_mismatch"));
    }

    #[test]
    fn graph_evidence_ignores_an_unrelated_relational_source_stage() {
        let graph_plan = graph_bfs_plan();
        let evidence =
            graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
                .expect("graph plan must produce bounded evidence");
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue a capability proof")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("graph evidence must attach to its exact plan");

        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut source_builder = QueryPlanBuilder::new();
        let source = source_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/unrelated-relational-source.parquet")
                    .with_row_groups(vec![0])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            schema,
            vec!["value".to_owned()],
        )));
        let source_plan = source_builder.finish_with_host_sink(source);
        let source_proof = QueryPlanCapabilityProof::for_query_plan(&source_plan)
            .expect("relational source must issue its own capability proof");

        let combined = QueryPlanCapabilityProof::combine(graph_proof, [source_proof])
            .expect("graph and unrelated relational proofs compose");
        let graph_floor = combined
            .as_requirement()
            .reservation_shapes
            .iter()
            .filter_map(|shape| shape.graph_memory_admission_evidence())
            .map(GraphMemoryAdmissionEvidence::cold_model_requested_bytes)
            .next()
            .expect("combined proof retains graph scope");
        let requirement = combined.as_requirement();
        assert_eq!(requirement.source_stage_bytes, None, "an unrelated relational source must not become graph input");
        assert!(requirement.reservation_shapes.iter().any(|shape| {
            shape.has_local_parquet_source()
                && shape.has_modeled_source_stage()
                && !shape.is_cugraph_graph_input_scope()
                && !shape.is_accounted_by_graph_phase()
        }));
        assert!(requirement.reservation_shapes.iter().any(|shape| {
            shape.graph_memory_admission_evidence().is_some()
                && shape.graph_memory_admission_evidence().unwrap().cold_model_requested_bytes() == graph_floor
        }));
    }

    #[test]
    fn equal_composed_graph_producers_retain_distinct_owner_term_occurrences() {
        let consumer_plan = graph_output_union_consumer(&["same-0", "same-1"]);
        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ("same-0".to_owned(), graph_bfs_proof_with_typed_output(0)),
                ("same-1".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ],
        )
        .expect("runtime occurrence rebinding keeps equal graph producers distinct");
        let requirement = combined.as_requirement();
        let expected_owner_term_count = if cfg!(feature = "cugraph") { 11 } else { 9 };
        assert_eq!(requirement.owner_terms.len(), expected_owner_term_count);
        let retained_occurrences = requirement
            .owner_terms
            .iter()
            .filter(|term| term.id.role == OwnerTermRole::RetainedOutput)
            .map(|term| term.id.occurrence.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(retained_occurrences, HashSet::from(["composed:same-0".to_owned(), "composed:same-1".to_owned()]));
        assert_eq!(
            requirement.owner_terms.iter().filter(|term| term.id.role == OwnerTermRole::RetainedOutput).count(),
            2,
            "each producer retained term is represented exactly once"
        );
        assert!(requirement.conservative_owner_sum().is_ok());
    }

    #[test]
    fn composed_consumer_rejects_an_extra_unconsumed_producer_occurrence() {
        let consumer_plan = graph_output_union_consumer(&["only"]);
        let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ("only".to_owned(), graph_bfs_proof_with_typed_output(0)),
                ("extra".to_owned(), graph_bfs_proof_with_typed_output(1)),
            ],
        )
        .expect_err("an extra producer occurrence must not be silently dropped");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_producer_occurrence_unconsumed"));
        assert_eq!(error.fact_value("occurrence_id"), Some("extra"));
    }

    #[test]
    fn composed_graph_schedule_seam_derives_complete_typed_union_evidence() {
        let consumer_plan =
            graph_output_union_consumer(&["zeta-occurrence", "alpha-occurrence", "mu-occurrence", "beta-occurrence"]);
        let first = graph_bfs_proof_with_typed_output(0);
        let second = graph_bfs_proof_with_typed_output(1);
        let third = graph_bfs_proof_with_typed_output(2);
        let fourth = graph_bfs_proof_with_typed_output(3);
        let consumer =
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer proof must be engine-issued");

        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            consumer,
            [
                ("zeta-occurrence".to_owned(), first),
                ("alpha-occurrence".to_owned(), second),
                ("mu-occurrence".to_owned(), third),
                ("beta-occurrence".to_owned(), fourth),
            ],
        )
        .expect("typed producer and engine consumer evidence must derive one complete schedule");
        assert_eq!(combined.as_requirement().reservation_shapes.len(), 5);
        let consumer_shape = combined
            .as_requirement()
            .reservation_shapes
            .iter()
            .find(|shape| shape.matches(&consumer_plan))
            .expect("the composed consumer scope must be present");
        let binding_occurrences = consumer_shape
            .retained_input_bindings()
            .iter()
            .map(|(occurrence_id, _)| occurrence_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            binding_occurrences,
            vec!["zeta-occurrence", "alpha-occurrence", "mu-occurrence", "beta-occurrence"]
        );
        let envelope = combined
            .resolve_reservation_envelope(reservation_policy())
            .expect("complete typed graph and union evidence must resolve before placement");
        assert!(envelope.complete_phase_peak_bytes().is_some());
    }

    #[test]
    fn composed_graph_schedule_prices_the_canonical_join_and_materialized_unary_branch() {
        let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, true);
        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
                ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
            ],
        )
        .expect("fixed-width non-null graph join must have complete engine-owned evidence");
        let envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(envelope.complete_phase_peak_bytes().is_some());
        assert!(envelope.required_phase_floor_bytes() > 0);
    }

    #[test]
    fn composed_resolver_keeps_graph_derivation_when_a_relational_gpu_producer_is_added() {
        let graph_only_consumer = graph_output_union_consumer(&["graph-left", "graph-right"]);
        let graph_only = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &graph_only_consumer,
            QueryPlanCapabilityProof::for_query_plan(&graph_only_consumer).unwrap(),
            [
                ComposedGraphProducerCandidate::new(
                    Some("graph-left".to_owned()),
                    Some("graph-left".to_owned()),
                    Some(graph_bfs_proof_with_typed_output(0)),
                ),
                ComposedGraphProducerCandidate::new(
                    Some("graph-right".to_owned()),
                    Some("graph-right".to_owned()),
                    Some(graph_bfs_proof_with_typed_output(1)),
                ),
            ],
        )
        .expect("two graph producers must resolve through the production seam");
        let graph_only_envelope = graph_only
            .resolve_reservation_envelope(reservation_policy())
            .expect("two graph producers must resolve a finite envelope");
        let graph_only_derivation =
            graph_only_envelope.bound_derivation().expect("two graph producers must select a finite bound derivation");

        let mixed_consumer = graph_output_union_consumer(&["graph-left", "graph-right", "relational"]);
        let relational = relational_gpu_producer_proof();
        let mixed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &mixed_consumer,
            QueryPlanCapabilityProof::for_query_plan(&mixed_consumer).unwrap(),
            [
                ComposedGraphProducerCandidate::new(
                    Some("graph-left".to_owned()),
                    Some("graph-left".to_owned()),
                    Some(graph_bfs_proof_with_typed_output(0)),
                ),
                ComposedGraphProducerCandidate::new(
                    Some("graph-right".to_owned()),
                    Some("graph-right".to_owned()),
                    Some(graph_bfs_proof_with_typed_output(1)),
                ),
                ComposedGraphProducerCandidate::new(Some("relational".to_owned()), None, Some(relational)),
            ],
        )
        .expect("adding a complete relational GPU producer must not change admissibility");
        let mixed_requirement = mixed.as_requirement();
        let relational_terms = mixed_requirement
            .owner_terms
            .iter()
            .filter(|term| term.id().occurrence_label() == "composed:relational")
            .collect::<Vec<_>>();
        assert!(!relational_terms.is_empty(), "the relational producer must retain engine-owned terms");
        assert!(relational_terms.iter().all(|term| term.bytes() > 0));
        let mixed_envelope = mixed
            .resolve_reservation_envelope(reservation_policy())
            .expect("the mixed producer set must resolve a finite envelope");
        assert_eq!(mixed_envelope.bound_derivation(), Some(graph_only_derivation));
        assert!(mixed_envelope.complete_phase_peak_bytes().is_some());
    }

    #[test]
    fn composed_graph_schedule_routes_n1_and_equal_n3_occurrences_through_one_path() {
        let n1_consumer = graph_output_union_consumer(&["only"]);
        let n1 = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &n1_consumer,
            QueryPlanCapabilityProof::for_query_plan(&n1_consumer).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("only".to_owned()),
                Some("only".to_owned()),
                Some(graph_bfs_proof_with_typed_output(0)),
            )],
        )
        .unwrap();
        assert!(n1.resolve_reservation_envelope(reservation_policy()).unwrap().complete_phase_peak_bytes().is_some());

        let n3_consumer = graph_output_union_consumer(&["same-0", "same-1", "same-2"]);
        let equal = graph_bfs_proof_with_typed_output(0);
        let n3 = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &n3_consumer,
            QueryPlanCapabilityProof::for_query_plan(&n3_consumer).unwrap(),
            [("same-0".to_owned(), equal.clone()), ("same-1".to_owned(), equal.clone()), ("same-2".to_owned(), equal)],
        )
        .unwrap();
        let envelope = n3.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(envelope.complete_phase_peak_bytes().is_some());
        assert_eq!(n3.as_requirement().reservation_shapes.len(), 4, "equal producer scopes remain three occurrences");
    }

    #[test]
    fn incomplete_ordered_metadata_falls_back_to_rebound_conservative_owner_terms() {
        let consumer_plan = graph_output_union_consumer(&["only"]);
        let proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("only".to_owned()),
                None,
                Some(graph_bfs_proof_with_typed_output(0)),
            )],
        )
        .expect("complete typed proof can conservatively compose without ordered metadata");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
        assert!(envelope.complete_phase_peak_bytes().is_some());
    }

    #[test]
    fn conservative_resolver_rejects_a_non_graph_source_proof_without_retained_output_term() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let source_bytes = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
        let mut producer_builder = QueryPlanBuilder::new();
        let producer_source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/conservative-producer.parquet").with_file_size_bytes(source_bytes)],
            Arc::clone(&schema),
            vec!["value".to_owned()],
        )));
        let producer_plan = producer_builder.finish_with_host_sink(producer_source);
        let mut consumer_builder = QueryPlanBuilder::new();
        let consumer_source = consumer_builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("only", Arc::clone(&schema)).with_row_count_upper_bound(1),
        ));
        let consumer_plan = consumer_builder.finish_with_host_sink(consumer_source);

        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("only".to_owned()),
                None,
                Some(QueryPlanCapabilityProof::for_query_plan(&producer_plan).unwrap()),
            )],
        )
        .expect_err("a non-graph producer without an AttemptRetained output must reject before placement");
        assert_eq!(error.fact_value("conservative_reason"), Some("producer_retained_output_term_missing"));
    }

    #[test]
    fn incomplete_single_graph_candidate_fails_closed_with_structured_schedule_facts() {
        let consumer_plan = graph_output_union_consumer(&["only"]);
        let incomplete_graph_proof =
            QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_bfs_plan_with_source_vertex(0))
                .expect("the graph plan issues its scoped capability proof before evidence is hydrated");

        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(
                Some("only".to_owned()),
                Some("only".to_owned()),
                Some(incomplete_graph_proof),
            )],
        )
        .expect_err("an incomplete N=1 candidate must not retain an unscheduled admission proof");

        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("producer_retained_output_term_missing"));
        assert_eq!(error.fact_value("occurrence_id"), Some("only"));
    }

    #[test]
    fn missing_single_graph_proof_fails_closed_with_structured_schedule_facts() {
        let consumer_plan = graph_output_union_consumer(&["only"]);
        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [ComposedGraphProducerCandidate::new(Some("only".to_owned()), Some("only".to_owned()), None)],
        )
        .expect_err("missing N=1 proof evidence must fail before admission");

        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("producer_admission_proof_missing_or_ambiguous"));
        assert_eq!(error.fact_value("occurrence_id"), Some("only"));
    }

    #[test]
    fn incomplete_multi_graph_candidate_remains_fail_closed() {
        let consumer_plan = graph_output_union_consumer(&["first", "second"]);
        let complete_graph_proof = graph_bfs_proof_with_typed_output(0);
        let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [
                ComposedGraphProducerCandidate::new(
                    Some("first".to_owned()),
                    Some("first".to_owned()),
                    Some(complete_graph_proof),
                ),
                ComposedGraphProducerCandidate::new(Some("second".to_owned()), Some("second".to_owned()), None),
            ],
        )
        .expect_err("two retained graph outputs require complete evidence");

        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("producer_admission_proof_missing_or_ambiguous"));
    }

    #[test]
    fn composed_consumer_alias_column_projection_reuses_its_input_allocation() {
        let consumer_plan = graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Int64);
        let output =
            ComposedGraphProducerOutputEvidence { occurrence_id: "only".to_owned(), row_count_upper_bound: 128 };
        let (bytes, owner_terms) = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &[output])
            .expect("an alias-only projection must have a bounded zero-copy contract");
        assert_eq!(bytes, 0, "an alias-only projection is a zero-copy select/rename");
        assert_eq!(owner_terms.iter().map(AllocationOwnerTerm::bytes).collect::<Vec<_>>(), vec![0]);

        let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
            [("only".to_owned(), graph_bfs_proof_with_typed_output(0))],
        )
        .expect("an exact-type alias projection must issue complete composed evidence");
        assert!(
            combined.resolve_reservation_envelope(reservation_policy()).unwrap().complete_phase_peak_bytes().is_some()
        );
    }

    #[test]
    fn composed_consumer_rejects_projected_column_type_changes_and_computation() {
        let output =
            [ComposedGraphProducerOutputEvidence { occurrence_id: "only".to_owned(), row_count_upper_bound: 128 }];
        let type_changed =
            graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Float64);
        let error = composed_consumer_additional_peak_bytes_for_plan(&type_changed, &output)
            .expect_err("a type-changing projection would allocate a new output frame");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_projection_output_type_mismatch"));

        for expression in
            [Expr::cast(Expr::column("value"), DataType::Float64), Expr::literal(crate::expr::LiteralValue::Int64(1))]
        {
            let consumer_plan = graph_output_projection_consumer(expression, DataType::Int64);
            let error = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &output)
                .expect_err("casts and computed expressions lack a zero-copy allocation contract");
            assert_eq!(error.fact_value("schedule_reason"), Some("consumer_projection_expression_workspace_unmodeled"));
        }
    }

    #[test]
    fn composed_graph_schedule_rejects_nullable_and_variable_width_join_keys() {
        for (data_type, nullable, expected_reason) in [
            (DataType::Int64, true, "consumer_join_nullable_key"),
            (DataType::Utf8, false, "consumer_join_variable_width_key"),
        ] {
            let consumer_plan = graph_output_inner_join_consumer(data_type, nullable, false);
            let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
                &consumer_plan,
                QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
                [
                    ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
                    ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
                ],
            )
            .expect_err("unsupported key evidence must reject before placement");
            assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
            assert_eq!(error.fact_value("schedule_reason"), Some(expected_reason));
        }
    }

    #[test]
    fn composed_consumer_rejects_join_output_row_overflow() {
        let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, false);
        let producer_outputs = [
            ComposedGraphProducerOutputEvidence { occurrence_id: "left".to_owned(), row_count_upper_bound: u64::MAX },
            ComposedGraphProducerOutputEvidence { occurrence_id: "right".to_owned(), row_count_upper_bound: 2 },
        ];
        let error = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &producer_outputs)
            .expect_err("overflowing inner-join output rows cannot become a finite admission contract");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_join_output_rows_overflow"));
    }

    #[test]
    fn composed_consumer_uses_the_linked_cudf_metadata_hash_preflight() {
        assert!(
            cudf::ops::join::hash_join_pre_build_reservation_size_bytes_from_non_nullable_fixed_width_metadata(
                &[cudf::TypeId::Int64],
                10,
                0.5,
            )
            .expect("the linked cuDF metadata preflight must accept a fixed-width non-null key")
                > 0
        );
    }

    fn graph_result_sort_plan(
        data_type: DataType,
        nullable: bool,
        row_bound: Option<u64>,
        key_count: usize,
    ) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, nullable)]));
        let mut builder = QueryPlanBuilder::new();
        let mut source = GpuResidentSource::new("graph_result", Arc::clone(&schema));
        if let Some(row_bound) = row_bound {
            source = source.with_row_count_upper_bound(row_bound);
        }
        let source = builder.add_source(SourceNode::GpuResident(source));
        let expressions = (0..key_count).map(|_| SortExpr::new(Expr::column("value"), false, false)).collect();
        let sort = builder.add_sort(source, SortSpec::new(expressions), schema);
        builder.finish_with_host_sink(sort)
    }

    fn graph_result_filter_sort_plan(row_bound: Option<u64>) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let mut builder = QueryPlanBuilder::new();
        let mut source = GpuResidentSource::new("graph_result", Arc::clone(&schema));
        if let Some(row_bound) = row_bound {
            source = source.with_row_count_upper_bound(row_bound);
        }
        let source = builder.add_source(SourceNode::GpuResident(source));
        let filter = builder.add_filter(source, Expr::column("value"), Arc::clone(&schema));
        let sort =
            builder.add_sort(filter, SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]), schema);
        builder.finish_with_host_sink(sort)
    }

    #[test]
    fn supported_graph_result_sort_resolves_exact_owner_components_per_device() {
        let plan = graph_result_sort_plan(DataType::Int32, false, Some(128), 1);
        let proof =
            QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result")
                .expect("supported sort metadata must issue a typed proof");
        let resolved = proof
            .requirement_for_device_with_preflight(7, |descriptor, ordinal| {
                assert_eq!(ordinal, 7);
                assert_eq!(descriptor.num_rows, 128);
                Ok(SortRadixPreflight::new_for_test(128, 512, 1_024, 512))
            })
            .expect("fake owner preflight resolves without CUDA");
        let terms = resolved
            .as_requirement()
            .owner_terms
            .iter()
            .filter(|term| matches!(term.id.owner, OwnerId::Sort(_)))
            .collect::<Vec<_>>();
        assert_eq!(terms.len(), 6);
        assert!(terms.iter().all(|term| term.bytes.get() > 0));
        assert!(
            terms
                .iter()
                .all(|term| term.id.execution_scope_fingerprint == canonical_execution_scope_fingerprint(&plan))
        );
    }

    #[test]
    fn composed_graph_result_sort_resolves_through_the_production_schedule_seam() {
        let consumer_plan = graph_result_sort_plan(DataType::Int32, false, None, 1);
        let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &consumer_plan,
            "graph_result",
            &graph_bfs_proof_with_typed_output(0),
        )
        .expect("supported sort metadata must issue a typed consumer proof");
        let composed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            consumer,
            [ComposedGraphProducerCandidate::new(
                Some("graph_result".to_owned()),
                Some("graph_result".to_owned()),
                Some(graph_bfs_proof_with_typed_output(0)),
            )],
        )
        .expect("one graph producer and its supported sort consumer must compose");
        let resolved = composed
            .requirement_for_device_with_preflight(0, |descriptor, _| {
                assert!(descriptor.num_rows > 0);
                Ok(SortRadixPreflight::new_for_test(128, 256, 512, 256))
            })
            .expect("the candidate device preflight must hydrate sort owner terms");
        let sort_owner_terms = resolved
            .as_requirement()
            .owner_terms
            .iter()
            .filter(|term| matches!(term.id.owner, OwnerId::Sort(_)))
            .count();
        assert_eq!(sort_owner_terms, 6);
        let envelope = resolved
            .resolve_reservation_envelope(reservation_policy())
            .expect("the composed sort proof must resolve a finite ordered envelope");
        assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
        assert!(envelope.complete_phase_peak_bytes().is_some());
    }

    #[test]
    fn composed_filter_then_sort_references_filter_and_sort_owner_terms() {
        let consumer_plan = graph_result_filter_sort_plan(None);
        let producer = graph_bfs_proof_with_typed_output(0);
        let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &consumer_plan,
            "graph_result",
            &producer,
        )
        .expect("Filter then one-key integral Sort has a bounded consumer proof");
        let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            consumer,
            [("graph_result".to_owned(), producer)],
        )
        .expect("Filter then Sort must consume one producer exactly once");
        let resolved = composed
            .requirement_for_device_with_preflight(0, |descriptor, _| {
                assert!(descriptor.num_rows > 0);
                Ok(SortRadixPreflight::new_for_test(64, 128, 256, 128))
            })
            .expect("the existing cuDF sort owner preflight resolves the consumer terms");
        assert!(resolved.as_requirement().owner_terms.iter().any(|term| {
            matches!(term.id.owner, OwnerId::Consumer(_)) && term.id.role == OwnerTermRole::ConsumerPredicateMask
        }));
        assert!(resolved.as_requirement().owner_terms.iter().any(|term| {
            matches!(term.id.owner, OwnerId::Sort(_)) && term.id.role == OwnerTermRole::SortTemporaryWorkspace
        }));
        let envelope = resolved
            .resolve_reservation_envelope(reservation_policy())
            .expect("Filter and Sort owner terms share one ordered consumer inventory");
        assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    }

    #[test]
    fn graph_result_sort_uses_the_producer_owned_row_bound_when_consumer_source_is_unbounded() {
        let plan = graph_result_sort_plan(DataType::Int32, false, None, 1);
        let producer_plan = graph_bfs_plan();
        let producer_evidence = graph_memory_bounded_admission_evidence_for_plan(
            &producer_plan,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .unwrap();
        let expected_rows = usize::try_from(producer_evidence.retained_output_rows_upper_bound()).unwrap();
        let producer = graph_bfs_proof_with_typed_output(0);
        let proof = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &plan,
            "graph_result",
            &producer,
        )
        .expect("producer-owned output rows must complete the bounded sort proof");
        let resolved = proof
            .requirement_for_device_with_preflight(3, |descriptor, ordinal| {
                assert_eq!(ordinal, 3);
                assert_eq!(descriptor.num_rows, expected_rows);
                Ok(SortRadixPreflight::new_for_test(64, 128, 256, 128))
            })
            .expect("producer-owned row bound must reach candidate preflight");
        assert!(resolved.as_requirement().owner_terms.iter().any(|term| {
            matches!(term.id.owner, OwnerId::Sort(_)) && term.id.role == OwnerTermRole::SortOutputFrame
        }));
    }

    #[test]
    fn graph_result_sort_producer_row_bound_failures_remain_sticky() {
        let plan = graph_result_sort_plan(DataType::Int32, false, None, 1);
        let ordinary = QueryPlanCapabilityProof::for_query_plan(&ordinary_native_plan()).unwrap();
        let missing = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &plan,
            "graph_result",
            &ordinary,
        )
        .unwrap();
        let missing_error = missing.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(missing_error.fact_value("sort_rejection_reason"), Some("sort_owner_producer_graph_scope_missing"));

        let ambiguous_producer = QueryPlanCapabilityProof::combine(
            graph_bfs_proof_with_typed_output(0),
            [graph_bfs_proof_with_typed_output(1)],
        )
        .unwrap();
        let ambiguous = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &plan,
            "graph_result",
            &ambiguous_producer,
        )
        .unwrap();
        let ambiguous_error = ambiguous.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(
            ambiguous_error.fact_value("sort_rejection_reason"),
            Some("sort_owner_producer_graph_scope_ambiguous")
        );

        let mismatched_plan = graph_result_sort_plan(DataType::Int32, false, Some(2), 1);
        let mismatched = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &mismatched_plan,
            "graph_result",
            &graph_bfs_proof_with_typed_output(0),
        )
        .unwrap();
        let mismatch_error = mismatched.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(
            mismatch_error.fact_value("sort_rejection_reason"),
            Some("sort_owner_graph_result_row_bound_mismatch")
        );
    }

    #[test]
    fn composed_resolver_preserves_sticky_sort_rejection_until_envelope_resolution() {
        let graph_producer = graph_bfs_proof_with_typed_output(0);
        let cases = [
            (
                graph_result_sort_plan(DataType::Float32, false, None, 1),
                graph_producer.clone(),
                "sort_owner_key_type_not_integral_or_chrono",
            ),
            (
                graph_result_sort_plan(DataType::Int32, false, None, 1),
                QueryPlanCapabilityProof::for_query_plan(&ordinary_native_plan()).unwrap(),
                "sort_owner_producer_graph_scope_missing",
            ),
        ];
        for (consumer_plan, producer_for_sort, expected_sort_reason) in cases {
            let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
                &consumer_plan,
                "graph_result",
                &producer_for_sort,
            )
            .expect("the attested sort shape must retain a typed rejection");
            let composed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
                &consumer_plan,
                consumer,
                [ComposedGraphProducerCandidate::new(
                    Some("graph_result".to_owned()),
                    Some("graph_result".to_owned()),
                    Some(graph_producer.clone()),
                )],
            )
            .expect("sticky sort rejection remains inside the production proof until envelope resolution");
            let error = composed
                .resolve_reservation_envelope(reservation_policy())
                .expect_err("unsupported sort must fail before an admission envelope is issued");
            assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
            assert_eq!(error.fact_value("schedule_reason"), Some("sort_metadata_rejected"));
            assert_eq!(error.fact_value("sort_rejection_reason"), Some(expected_sort_reason));
        }
    }

    #[test]
    fn unsupported_graph_result_sort_shapes_reject_before_preflight() {
        for (plan, reason) in [
            (
                graph_result_sort_plan(DataType::Float32, false, Some(8), 1),
                "sort_owner_key_type_not_integral_or_chrono",
            ),
            (graph_result_sort_plan(DataType::Utf8, false, Some(8), 1), "sort_owner_key_type_not_integral_or_chrono"),
            (graph_result_sort_plan(DataType::Int32, true, Some(8), 1), "sort_owner_nullable_key_unsupported"),
            (graph_result_sort_plan(DataType::Int32, false, Some(8), 2), "sort_owner_sort_requires_exactly_one_key"),
            (graph_result_sort_plan(DataType::Int32, false, None, 1), "sort_owner_row_bound_missing"),
        ] {
            let error =
                QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result")
                    .expect_err("unsupported sort shape must reject before native preflight");
            assert_eq!(error.fact_value("reason"), Some(reason));
        }
    }

    #[test]
    fn stable_sort_metadata_rejects_before_the_fake_native_preflight() {
        let plan = graph_result_sort_plan(DataType::Int32, false, Some(8), 1);
        let mut proof =
            QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result")
                .unwrap();
        proof.requirement.reservation_shapes.first_mut().unwrap().set_sort_stable_for_test(true);
        let error = proof
            .requirement_for_device_with_preflight(0, |_, _| {
                panic!("unsupported stable sort must not call native preflight")
            })
            .expect_err("stable sort metadata must reject before preflight");
        assert_eq!(error.fact_value("reason"), Some("sort_owner_stable_sort_unsupported"));
    }
}
