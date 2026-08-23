use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use super::{
    AggregateFunction, AggregateNode, AllocationClass, AllocationOwnerTerm, AttemptReservationConsumerState, DataType,
    Expr, FiniteBytes, JoinNode, JoinSide, JoinType, LimitMetadataDescriptor, LimitRangeGatherPreflight, LiteralValue,
    NodeId, OccurrenceId, OwnerId, OwnerTermId, OwnerTermRole, PlanColumnEvidence, PlanColumnEvidenceAnalysis,
    PlanNode, QueryError, QueryErrorCode, QueryPlan, QueryPlanCapabilityProof, ReservationKind, ReservationRequirement,
    Result, ResultSink, Schema, SchemaRef, SortKeyMetadataDescriptor, SortMetadataDescriptor, SortPreflight,
    SourceNode, TimeUnit, aggregate_owner_state_is_supported, aggregate_owner_terms_for_shapes, alias_wrapped_column,
    canonical_execution_scope_fingerprint, cudf_dtype_from_arrow_type, graph_output_is_final, limit_owner_error,
    owner_term_error, query_plan_hard_row_upper_bound, query_plan_node_hard_row_upper_bound,
    representation_cast_preserves_identity, sort_owner_error,
};

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct ComposedGraphProducerOutputEvidence {
    #[doc(hidden)]
    pub occurrence_id: String,
    #[doc(hidden)]
    pub row_count_upper_bound: u64,
    #[doc(hidden)]
    pub column_evidence: PlanColumnEvidence,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct NormalizedComposedProducer {
    #[doc(hidden)]
    pub occurrence_id: String,
    #[doc(hidden)]
    pub execution_scope_fingerprint: String,
    #[doc(hidden)]
    pub retained_output_term_id: OwnerTermId,
    #[doc(hidden)]
    pub row_count_upper_bound: u64,
    pub(super) output_column_evidence: PlanColumnEvidence,
    #[doc(hidden)]
    pub phase_term_groups: Vec<Vec<OwnerTermId>>,
    #[doc(hidden)]
    pub alternative_phase_term_ids: Vec<OwnerTermId>,
    #[doc(hidden)]
    pub concurrent_side_input_term_ids: Vec<OwnerTermId>,
    #[doc(hidden)]
    pub dominated_term_ids: Vec<OwnerTermId>,
    #[doc(hidden)]
    pub nested_schedule: bool,
    #[doc(hidden)]
    pub proof: QueryPlanCapabilityProof,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct ComposedConsumerBranchBound {
    #[doc(hidden)]
    pub rows: u64,
    #[doc(hidden)]
    pub schema: SchemaRef,
    /// Bytes allocated by this consumer branch that remain live at its root.
    /// Producer result handles are intentionally excluded: the schedule fold
    /// retains them outside the consumer phase.
    #[doc(hidden)]
    pub final_owned_bytes: u64,
    #[doc(hidden)]
    pub owner_terms: Vec<AllocationOwnerTerm>,
    /// Alternative consumer lifetime peaks; each group is summed and the
    /// schedule takes the maximum across groups.
    #[doc(hidden)]
    pub peak_owner_term_groups: Vec<Vec<OwnerTermId>>,
    /// Flattened view retained for branch-local dominance decisions.
    #[doc(hidden)]
    pub peak_owner_term_ids: Vec<OwnerTermId>,
    #[doc(hidden)]
    pub dominated_owner_term_ids: Vec<OwnerTermId>,
    #[doc(hidden)]
    pub final_owned_term_ids: Vec<OwnerTermId>,
    #[doc(hidden)]
    pub allocation_state: AttemptReservationConsumerState,
    #[doc(hidden)]
    pub consumed_occurrences: Vec<String>,
}

/// One composed-consumer bound analysis traversal: the plan under analysis,
/// its producer-output evidence, the plan-wide owner-term inventory, and the
/// memo and consumed-occurrence state the traversal maintains.
///
/// Node-specific data — the recursion target [`NodeId`], the operator node,
/// and operator inputs — stays explicit at each call site. The analysis owns
/// exactly the values every composed-consumer bound helper shares.
#[doc(hidden)]
pub struct ComposedConsumerAnalysis<'a> {
    pub(super) plan: &'a QueryPlan,
    pub(super) outputs: HashMap<&'a str, &'a ComposedGraphProducerOutputEvidence>,
    pub(super) owner_terms: std::borrow::Cow<'a, [AllocationOwnerTerm]>,
    pub(super) column_evidence: &'a PlanColumnEvidenceAnalysis,
    pub(super) used_occurrences: HashSet<String>,
    pub(super) memo: HashMap<NodeId, ComposedConsumerBranchBound>,
}

impl<'a> ComposedConsumerAnalysis<'a> {
    /// Builds the analysis over non-duplicate producer occurrences.
    pub(super) fn new(
        plan: &'a QueryPlan,
        producer_outputs: &'a [ComposedGraphProducerOutputEvidence],
        owner_terms: &'a [AllocationOwnerTerm],
        column_evidence: &'a PlanColumnEvidenceAnalysis,
    ) -> Result<Self> {
        let mut outputs = HashMap::with_capacity(producer_outputs.len());
        for output in producer_outputs {
            if output.occurrence_id.is_empty() || outputs.insert(output.occurrence_id.as_str(), output).is_some() {
                return Err(composed_schedule_error("duplicate_producer_occurrence_id", Some(&output.occurrence_id)));
            }
        }
        let used_occurrences_capacity = outputs.len();
        Ok(Self {
            plan,
            outputs,
            owner_terms: std::borrow::Cow::Borrowed(owner_terms),
            column_evidence,
            used_occurrences: HashSet::with_capacity(used_occurrences_capacity),
            memo: HashMap::new(),
        })
    }

    /// Root entry: memoized bound for the plan root node.
    pub(super) fn root_bound(&mut self) -> Result<ComposedConsumerBranchBound> {
        self.branch_bound(self.plan.root_id())
    }

    /// Memoized bound for one node in the analysis plan.
    pub(super) fn branch_bound(&mut self, node_id: NodeId) -> Result<ComposedConsumerBranchBound> {
        composed_consumer_branch_bound(self, node_id)
    }

    /// Graph-algorithm peak groups over the traversal memo.
    pub(super) fn graph_peak_groups(&mut self) -> Result<Vec<Vec<OwnerTermId>>> {
        let mut inventory = self.owner_terms.clone().into_owned();
        let root_bound = self
            .memo
            .get(&self.plan.root_id())
            .cloned()
            .ok_or_else(|| composed_schedule_error("consumer_root_bound_missing", None))?;
        inventory.extend(root_bound.owner_terms.iter().cloned());
        memoized_graph_consumer_peak_groups(self.plan, &self.memo, &inventory)
    }

    /// Runs `visit` with a temporary owner-term inventory, restoring the
    /// plan-wide inventory afterwards. Graph side inputs extend the traversal
    /// with the input branch's derived terms before binding their own.
    pub(super) fn with_owner_terms_override<R>(
        &mut self,
        override_terms: Vec<AllocationOwnerTerm>,
        visit: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let plan_terms = std::mem::replace(&mut self.owner_terms, std::borrow::Cow::Owned(override_terms));
        let result = visit(self);
        self.owner_terms = plan_terms;
        result
    }

    /// The producer occurrences this analysis consumed so far.
    pub(super) fn used_occurrences(&self) -> &HashSet<String> {
        &self.used_occurrences
    }
}

#[cfg(test)]
#[doc(hidden)]
pub fn composed_consumer_additional_peak_bytes_for_plan(
    consumer_plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
) -> Result<(u64, Vec<AllocationOwnerTerm>)> {
    let bound = composed_consumer_bound_with_peak_terms(consumer_plan, producer_outputs, &[])?;
    let additional_peak_bytes = consumer_peak_groups_bytes(&bound.peak_owner_term_groups, &bound.owner_terms)?;
    Ok((additional_peak_bytes, bound.owner_terms))
}

#[doc(hidden)]
pub fn composed_consumer_bound_with_peak_terms(
    consumer_plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
    owner_terms: &[AllocationOwnerTerm],
) -> Result<ComposedConsumerBranchBound> {
    if !consumer_plan.unreachable_node_ids().is_empty() {
        return Err(composed_schedule_error("consumer_contains_unreachable_nodes", None));
    }
    let resident_outputs = producer_outputs
        .iter()
        .map(|output| (output.occurrence_id.clone(), output.column_evidence.clone()))
        .collect::<HashMap<_, _>>();
    let resident_row_upper_bounds = producer_outputs
        .iter()
        .map(|output| (output.occurrence_id.clone(), output.row_count_upper_bound))
        .collect::<HashMap<_, _>>();
    let column_evidence = PlanColumnEvidenceAnalysis::for_plan_with_resident_rows(
        consumer_plan,
        &resident_outputs,
        &resident_row_upper_bounds,
    )?;
    let mut analysis = ComposedConsumerAnalysis::new(consumer_plan, producer_outputs, owner_terms, &column_evidence)?;
    let mut root_bound = analysis.root_bound()?;
    if consumer_plan.contains_graph_algorithm() {
        root_bound.peak_owner_term_groups = analysis.graph_peak_groups()?;
        root_bound.peak_owner_term_ids = flatten_consumer_peak_groups(&root_bound.peak_owner_term_groups);
    }
    for group in &mut root_bound.peak_owner_term_groups {
        let mut seen = HashSet::with_capacity(group.len());
        group.retain(|term_id| seen.insert(term_id.clone()));
    }
    let mut seen_owner_terms = HashSet::with_capacity(root_bound.owner_terms.len());
    root_bound.owner_terms.retain(|term| seen_owner_terms.insert(term.id.clone()));
    root_bound.owner_terms.retain(|term| !owner_terms.contains(term));
    match root_bound.allocation_state {
        AttemptReservationConsumerState::DeferredUntilResidentEvidence => {
            return Err(composed_schedule_error("consumer_deferred_state_cannot_be_derived_statically", None));
        },
        AttemptReservationConsumerState::Passthrough => {
            if !root_bound.peak_owner_term_groups.is_empty()
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
    if let Some(unconsumed) =
        producer_outputs.iter().find(|output| !analysis.used_occurrences().contains(&output.occurrence_id))
    {
        return Err(composed_schedule_error(
            "consumer_producer_occurrence_unconsumed",
            Some(&unconsumed.occurrence_id),
        ));
    }
    transfer_gpu_sink_consumer_output(consumer_plan, &mut root_bound)?;
    root_bound.consumed_occurrences = producer_outputs.iter().map(|output| output.occurrence_id.clone()).collect();
    Ok(root_bound)
}

fn transfer_gpu_sink_consumer_output(plan: &QueryPlan, bound: &mut ComposedConsumerBranchBound) -> Result<()> {
    let ResultSink::GpuResident { input } = plan.sink() else {
        return Ok(());
    };
    let transferable = bound.final_owned_term_ids.iter().all(|term_id| {
        matches!(
            term_id.role,
            OwnerTermRole::ConsumerOutput
                | OwnerTermRole::AggregateOutput
                | OwnerTermRole::SortOutputFrame
                | OwnerTermRole::SortOutputNullMask
                | OwnerTermRole::LimitOutputFrame
                | OwnerTermRole::LimitOutputNullMask
        )
    });
    if bound.final_owned_term_ids.is_empty() || !transferable {
        return Ok(());
    }
    let transferred_ids = bound.final_owned_term_ids.iter().cloned().collect::<HashSet<_>>();
    let retained = AllocationOwnerTerm::new(
        OwnerId::Producer(*input),
        OwnerTermRole::RetainedOutput,
        OccurrenceId::PlanLocal(*input),
        canonical_execution_scope_fingerprint(plan),
        AllocationClass::AttemptRetained,
        bound.final_owned_bytes,
        Some(bound.rows),
    )?;
    let new_id = retained.id().clone();
    for term_id in bound
        .peak_owner_term_groups
        .iter_mut()
        .flatten()
        .chain(bound.peak_owner_term_ids.iter_mut())
        .chain(bound.dominated_owner_term_ids.iter_mut())
        .chain(bound.final_owned_term_ids.iter_mut())
    {
        if transferred_ids.contains(term_id) {
            *term_id = new_id.clone();
        }
    }
    for group in &mut bound.peak_owner_term_groups {
        let mut seen = HashSet::with_capacity(group.len());
        group.retain(|term_id| seen.insert(term_id.clone()));
    }
    bound.peak_owner_term_ids.sort();
    bound.peak_owner_term_ids.dedup();
    bound.dominated_owner_term_ids.sort();
    bound.dominated_owner_term_ids.dedup();
    bound.final_owned_term_ids.sort();
    bound.final_owned_term_ids.dedup();
    bound.owner_terms.retain(|term| !transferred_ids.contains(term.id()));
    bound.owner_terms.push(retained);
    Ok(())
}

#[doc(hidden)]
pub fn memoized_graph_consumer_peak_groups(
    plan: &QueryPlan,
    memo: &HashMap<NodeId, ComposedConsumerBranchBound>,
    owner_terms: &[AllocationOwnerTerm],
) -> Result<Vec<Vec<OwnerTermId>>> {
    fn visit(
        plan: &QueryPlan,
        node_id: NodeId,
        visiting: &mut HashSet<NodeId>,
        seen: &mut HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<()> {
        if seen.contains(&node_id) {
            return Ok(());
        }
        if !visiting.insert(node_id) {
            return Err(composed_schedule_error("consumer_plan_cycle", None).with_fact("node_id", node_id));
        }
        let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
        for input in node.input_ids() {
            visit(plan, *input, visiting, seen, order)?;
        }
        visiting.remove(&node_id);
        seen.insert(node_id);
        order.push(node_id);
        Ok(())
    }

    let mut order = Vec::new();
    visit(plan, plan.root_id(), &mut HashSet::new(), &mut HashSet::new(), &mut order)?;
    let mut remaining_consumers = HashMap::<NodeId, usize>::new();
    for node_id in &order {
        let node = plan.node(*node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
        for input in node.input_ids() {
            *remaining_consumers.entry(*input).or_default() += 1;
        }
    }
    let scope = canonical_execution_scope_fingerprint(plan);
    let mut live = Vec::<OwnerTermId>::new();
    let mut live_references = HashMap::<OwnerTermId, usize>::new();
    let mut groups = Vec::new();
    for node_id in order {
        let Some(bound) = memo.get(&node_id) else {
            return Err(composed_schedule_error("consumer_bound_missing", None).with_fact("node_id", node_id));
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
        let mut node_term_ids = node_terms.iter().map(|term| term.id.clone()).collect::<Vec<_>>();
        node_term_ids.extend(
            bound
                .peak_owner_term_groups
                .iter()
                .flatten()
                .filter(|term_id| {
                    term_id.execution_scope_fingerprint() == scope
                        && match term_id.owner {
                            OwnerId::Graph(id)
                            | OwnerId::Aggregate(id)
                            | OwnerId::Source(id)
                            | OwnerId::Consumer(id)
                            | OwnerId::Sort(id)
                            | OwnerId::Limit(id) => id == node_id,
                            _ => false,
                        }
                })
                .cloned(),
        );
        node_term_ids.sort();
        node_term_ids.dedup();
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
                    .filter(|term| term.is_execution_phase_term() || term.is_retained_output())
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
        } else if !node_term_ids.is_empty() {
            let mut group = live.clone();
            group.extend(node_term_ids);
            groups.push(group);
        }
        let node = plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
        for input in node.input_ids() {
            let Some(remaining) = remaining_consumers.get_mut(input) else {
                return Err(composed_schedule_error("consumer_count_missing", None)
                    .with_fact("node_id", node_id)
                    .with_fact("input_node_id", input));
            };
            *remaining = remaining.checked_sub(1).ok_or_else(|| {
                composed_schedule_error("consumer_count_underflow", None).with_fact("node_id", node_id)
            })?;
            if *remaining == 0
                && let Some(input_bound) = memo.get(input)
            {
                for term_id in &input_bound.final_owned_term_ids {
                    let Some(references) = live_references.get_mut(term_id) else {
                        return Err(composed_schedule_error("consumer_live_term_missing", None)
                            .with_fact("node_id", node_id)
                            .with_fact("input_node_id", input)
                            .with_fact("owner_role", term_id.role_label()));
                    };
                    *references = references.checked_sub(1).ok_or_else(|| {
                        composed_schedule_error("consumer_live_term_reference_underflow", None)
                            .with_fact("node_id", node_id)
                            .with_fact("input_node_id", input)
                    })?;
                    if *references == 0 {
                        live_references.remove(term_id);
                        live.retain(|live_term| live_term != term_id);
                    }
                }
            }
        }
        for term_id in &bound.final_owned_term_ids {
            let references = live_references.entry(term_id.clone()).or_default();
            *references = references.checked_add(1).ok_or_else(|| {
                composed_schedule_error("consumer_live_term_reference_overflow", None).with_fact("node_id", node_id)
            })?;
            if *references == 1 {
                live.push(term_id.clone());
            }
        }
    }
    if groups.is_empty() {
        return Err(composed_schedule_error("consumer_bounded_owner_terms_missing", None));
    }
    Ok(groups)
}

#[doc(hidden)]
pub fn composed_consumer_branch_bound(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
) -> Result<ComposedConsumerBranchBound> {
    if let Some(cached) = analysis.memo.get(&node_id) {
        return Ok(ComposedConsumerBranchBound {
            rows: cached.rows,
            schema: Arc::clone(&cached.schema),
            final_owned_bytes: cached.final_owned_bytes,
            owner_terms: cached.owner_terms.clone(),
            peak_owner_term_groups: Vec::new(),
            peak_owner_term_ids: Vec::new(),
            dominated_owner_term_ids: Vec::new(),
            final_owned_term_ids: cached.final_owned_term_ids.clone(),
            allocation_state: cached.allocation_state,
            consumed_occurrences: Vec::new(),
        });
    }
    let bound = composed_consumer_branch_bound_uncached(analysis, node_id)?;
    analysis.memo.insert(node_id, bound.clone());
    Ok(bound)
}

#[doc(hidden)]
pub fn composed_consumer_branch_bound_uncached(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
) -> Result<ComposedConsumerBranchBound> {
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    match &node.kind {
        PlanNode::Source(source) => {
            match source.as_ref() {
                SourceNode::GpuResident(source) => {
                    let Some(output) = analysis.outputs.get(source.input_name.as_str()) else {
                        return Err(composed_schedule_error(
                            "consumer_source_occurrence_missing",
                            Some(&source.input_name),
                        ));
                    };
                    analysis.used_occurrences.insert(source.input_name.clone());
                    let rows = source
                        .row_count_upper_bound
                        .map_or(output.row_count_upper_bound, |rows| rows.min(output.row_count_upper_bound));
                    Ok(ComposedConsumerBranchBound {
                        rows,
                        schema: source.schema.clone(),
                        final_owned_bytes: 0,
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
                    let source_term = analysis.owner_terms.iter().find(|term| {
                        matches!(term.id.owner, OwnerId::Source(id) if id == node_id)
                            && term.id.role == OwnerTermRole::SourceStage
                            && term.id.occurrence == OccurrenceId::PlanLocal(node_id)
                            && term.execution_scope_fingerprint()
                                == canonical_execution_scope_fingerprint(analysis.plan)
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
                    let source_term = analysis.owner_terms.iter().find(|term| {
                        matches!(term.id.owner, OwnerId::Source(id) if id == node_id)
                            && term.id.role == OwnerTermRole::SourceStage
                            && term.id.occurrence == OccurrenceId::PlanLocal(node_id)
                            && term.execution_scope_fingerprint()
                                == canonical_execution_scope_fingerprint(analysis.plan)
                    });
                    let Some(source_term) = source_term else {
                        return Err(composed_schedule_error("consumer_literal_source_owner_missing", None)
                            .with_fact("node_id", node_id));
                    };
                    let bytes = source_term.bytes.get();
                    let source_term_id = source_term.id.clone();
                    Ok(ComposedConsumerBranchBound {
                        rows,
                        schema: source.schema.clone(),
                        final_owned_bytes: bytes,
                        owner_terms: Vec::new(),
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
            let input = composed_consumer_branch_bound(analysis, filter.input)?;
            let output_bytes = bounded_consumer_frame_bytes_with_evidence(
                input.rows,
                node.output_schema(),
                &analysis.column_evidence.node(node_id),
            )?;
            let predicate_mask_bytes = bounded_consumer_boolean_mask_bytes(input.rows)?;
            let output_term = consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerOutput,
                AllocationClass::ActivePhase,
                output_bytes,
                analysis.plan,
            )?;
            let predicate_term = consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerPredicateMask,
                AllocationClass::ActivePhase,
                predicate_mask_bytes,
                analysis.plan,
            )?;
            let output_term_id = output_term.id.clone();
            let predicate_term_id = predicate_term.id.clone();
            let peak_owner_term_groups =
                consumer_unary_peak_groups(&input, [output_term_id.clone(), predicate_term_id.clone()]);
            let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
            let mut consumer_owner_terms = input.owner_terms;
            consumer_owner_terms.push(output_term);
            consumer_owner_terms.push(predicate_term);
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: output_bytes,
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
            let input = composed_consumer_branch_bound(analysis, projection.input)?;
            for output in &projection.outputs {
                let (column, representation_cast) = if let Some(column) = alias_wrapped_column(&output.expr) {
                    (column, false)
                } else if let Some((column, target_type)) =
                    consumer_representation_cast_column(&output.expr, &input.schema)
                {
                    if !representation_cast_preserves_identity(target_type, output.field.data_type()) {
                        return Err(composed_schedule_error(
                            "consumer_projection_expression_workspace_unmodeled",
                            None,
                        )
                        .with_fact("expression", output.expr.clone())
                        .with_fact("cast_target_type", format!("{target_type:?}"))
                        .with_fact("output_type", format!("{:?}", output.field.data_type())));
                    }
                    (column, true)
                } else if crate::expr::representation_cast_column(&output.expr).is_some() {
                    return Err(composed_schedule_error("consumer_projection_expression_workspace_unmodeled", None)
                        .with_fact("expression", output.expr.clone()));
                } else if projection.outputs.iter().any(|output| alias_wrapped_literal(&output.expr).is_some()) {
                    return composed_consumer_literal_projection_bound(analysis, node_id, node, projection, input);
                } else {
                    return composed_consumer_expression_projection_bound(analysis, node_id, node, projection, input);
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
                        .with_fact("expression", output.expr.clone())
                        .with_fact("input_type", format!("{:?}", source.data_type()))
                        .with_fact("output_type", format!("{:?}", output.field.data_type())));
                    }
                } else if source.data_type() != output.field.data_type() {
                    return Err(composed_schedule_error("consumer_projection_output_type_mismatch", None)
                        .with_fact("column", column.name())
                        .with_fact("input_type", source.data_type())
                        .with_fact("output_type", output.field.data_type()));
                }
            }
            Ok(ComposedConsumerBranchBound {
                rows: input.rows,
                schema: node.output_schema().clone(),
                final_owned_bytes: input.final_owned_bytes,
                owner_terms: input.owner_terms,
                peak_owner_term_groups: input.peak_owner_term_groups,
                peak_owner_term_ids: input.peak_owner_term_ids,
                dominated_owner_term_ids: input.dominated_owner_term_ids,
                final_owned_term_ids: input.final_owned_term_ids,
                allocation_state: input.allocation_state,
                consumed_occurrences: Vec::new(),
            })
        },
        PlanNode::Window(window) => composed_consumer_global_row_number_peak_bytes(analysis, node_id, window),
        PlanNode::Aggregate(aggregate) => composed_consumer_aggregate_peak_bytes(analysis, node_id, aggregate),
        PlanNode::Join(join) => composed_consumer_join_peak_bytes(analysis, node_id, join),
        PlanNode::Union(union) => composed_consumer_union_peak_bytes(analysis, node_id, union),
        PlanNode::Sort(sort) => composed_consumer_sort_peak_bytes(analysis, node_id, sort),
        PlanNode::Limit(limit) => composed_consumer_limit_peak_bytes(analysis, node_id, limit),
        PlanNode::EdgeNormalize(normalize) => composed_consumer_edge_normalize_peak_bytes(analysis, node_id, normalize),
        PlanNode::GraphAlgorithm(graph) => composed_consumer_graph_peak_bytes(analysis, node_id, graph),
        _ => Err(composed_schedule_error("consumer_branch_topology_unmodeled", None)),
    }
}

#[doc(hidden)]
pub fn composed_consumer_edge_normalize_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    normalize: &crate::plan::EdgeNormalizeNode,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(analysis, normalize.input)?;
    let output_rows = if normalize.spec.add_reverse {
        input
            .rows
            .checked_mul(2)
            .ok_or_else(|| composed_schedule_error("consumer_edge_normalize_rows_overflow", None))?
    } else {
        input.rows
    };
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_bytes = bounded_consumer_frame_bytes_with_evidence(
        output_rows,
        node.output_schema(),
        &analysis.column_evidence.node(node_id),
    )?;
    let workspace_bytes = output_bytes
        .checked_mul(2)
        .ok_or_else(|| composed_schedule_error("consumer_edge_normalize_workspace_overflow", None))?;
    let output_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        output_bytes,
        analysis.plan,
    )?;
    let workspace_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerWorkspace,
        AllocationClass::ActivePhase,
        workspace_bytes,
        analysis.plan,
    )?;
    let output_term_id = output_term.id.clone();
    let workspace_term_id = workspace_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [output_term_id.clone(), workspace_term_id]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.extend([output_term, workspace_term]);
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn composed_consumer_global_row_number_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    window: &crate::plan::WindowNode,
) -> Result<ComposedConsumerBranchBound> {
    let Some(crate::plan::WindowFunction::RowNumber { alias, output_type }) = window.spec.functions.first() else {
        return Err(composed_schedule_error("consumer_window_topology_unmodeled", None));
    };
    if !window.spec.partition_by.is_empty()
        || window.spec.functions.len() != 1
        || !matches!(output_type, DataType::Int64 | DataType::UInt64)
    {
        return Err(composed_schedule_error("consumer_window_topology_unmodeled", None));
    }
    let input = composed_consumer_branch_bound(analysis, window.input)?;
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_field = node.output_schema().field_with_name(alias).map_err(|_| {
        composed_schedule_error("consumer_window_output_field_missing", None).with_fact("column", alias)
    })?;
    let output_schema = Arc::new(Schema::new(vec![output_field.clone()]));
    let output_bytes =
        bounded_consumer_frame_bytes_with_evidence(input.rows, &output_schema, &PlanColumnEvidence::default())?;
    let output_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        output_bytes,
        analysis.plan,
    )?;
    let output_term_id = output_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [output_term_id.clone()]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let final_owned_bytes =
        checked_consumer_add(input.final_owned_bytes, output_bytes, "consumer_window_final_owned_overflow")?;
    let final_owned_term_ids =
        input.final_owned_term_ids.iter().cloned().chain(std::iter::once(output_term_id.clone())).collect();
    let mut terms = input.owner_terms;
    terms.push(output_term);
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: node.output_schema().clone(),
        final_owned_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids,
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn composed_consumer_graph_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
) -> Result<ComposedConsumerBranchBound> {
    let mut input = analysis.branch_bound(graph.input)?;
    for side_input in graph.inputs.iter().skip(1) {
        let mut input_inventory = analysis.owner_terms.clone().into_owned();
        input_inventory.extend(input.owner_terms.iter().cloned());
        let side =
            analysis.with_owner_terms_override(input_inventory, |analysis| analysis.branch_bound(*side_input))?;
        input.owner_terms.extend(side.owner_terms);
        input.peak_owner_term_groups.extend(side.peak_owner_term_groups);
        input.final_owned_term_ids.extend(side.final_owned_term_ids);
        input.dominated_owner_term_ids.extend(side.dominated_owner_term_ids);
    }
    let scope = canonical_execution_scope_fingerprint(analysis.plan);
    let graph_terms = analysis
        .owner_terms
        .iter()
        .filter(|term| term.id.owner == OwnerId::Graph(node_id) && term.execution_scope_fingerprint() == scope)
        .collect::<Vec<_>>();
    let outputs = graph_terms
        .iter()
        .filter(|term| matches!(term.id.role, OwnerTermRole::GraphOutput | OwnerTermRole::RetainedOutput))
        .copied()
        .collect::<Vec<_>>();
    let [output] = outputs.as_slice() else {
        return Err(composed_schedule_error("consumer_graph_owner_terms_unavailable", None));
    };
    let construction = graph_terms
        .iter()
        .filter(|term| term.is_construction_phase_term())
        .map(|term| term.id.clone())
        .collect::<Vec<_>>();
    let execution = graph_terms
        .iter()
        .filter(|term| term.is_execution_phase_term() || term.is_retained_output())
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
    let rows = output
        .retained_row_count_upper_bound()
        .or_else(|| query_plan_node_hard_row_upper_bound(analysis.plan, node_id))
        .ok_or_else(|| composed_schedule_error("consumer_graph_output_rows_unavailable", None))?;
    Ok(ComposedConsumerBranchBound {
        rows,
        schema: analysis
            .plan
            .node(node_id)
            .ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?
            .output_schema()
            .clone(),
        final_owned_bytes: output.bytes(),
        owner_terms: input.owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output.id.clone()],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn alias_wrapped_literal(expr: &Expr) -> Option<&LiteralValue> {
    match expr {
        Expr::Literal(literal) => Some(literal),
        Expr::Alias { expr, .. } => alias_wrapped_literal(expr),
        Expr::Cast { expr, target_type } => {
            let literal = alias_wrapped_literal(expr)?;
            representation_cast_preserves_identity(&literal.data_type(), target_type).then_some(literal)
        },
        _ => None,
    }
}

fn consumer_representation_cast_column<'a>(
    expr: &'a Expr,
    input_schema: &'a SchemaRef,
) -> Option<(&'a crate::expr::ColumnRef, &'a DataType)> {
    match expr {
        Expr::Alias { expr, .. } => consumer_representation_cast_column(expr, input_schema),
        Expr::Cast { expr, target_type } => {
            let (column, source_type) = match expr.as_ref() {
                Expr::Column(column) => (column, input_schema.field_with_name(column.name()).ok()?.data_type()),
                Expr::Alias { .. } | Expr::Cast { .. } => consumer_representation_cast_column(expr, input_schema)?,
                _ => return None,
            };
            representation_cast_preserves_identity(source_type, target_type).then_some((column, target_type))
        },
        _ => None,
    }
}

#[doc(hidden)]
pub fn composed_consumer_literal_projection_bound(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    node: &crate::plan::QueryNode,
    projection: &crate::plan::ProjectionNode,
    input: ComposedConsumerBranchBound,
) -> Result<ComposedConsumerBranchBound> {
    let mut retains_input = false;
    let mut literal_fields = Vec::new();
    for output in &projection.outputs {
        let identity_column = alias_wrapped_column(&output.expr).map(|column| (column, false)).or_else(|| {
            consumer_representation_cast_column(&output.expr, &input.schema).map(|(column, _)| (column, true))
        });
        if let Some((column, representation_cast)) = identity_column {
            let source = input.schema.field_with_name(column.name()).map_err(|_| {
                composed_schedule_error("consumer_projection_source_column_missing", None)
                    .with_fact("column", column.name())
            })?;
            let type_matches = if representation_cast {
                representation_cast_preserves_identity(source.data_type(), output.field.data_type())
            } else {
                source.data_type() == output.field.data_type()
            };
            if !type_matches {
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
        let string_output =
            matches!(output.field.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View);
        if !representation_cast_preserves_identity(&literal.data_type(), output.field.data_type())
            || matches!(literal, LiteralValue::Null(_)) && !output.field.is_nullable()
            || fixed_width_consumer_output_column(output.field.data_type()).is_none() && !string_output
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
    let mut literal_evidence = PlanColumnEvidenceAnalysis::for_plan(analysis.plan, &HashMap::new())?.node(node_id);
    for field in literal_schema.fields() {
        let Some(bound) = literal_evidence.string_payload_bounds.get_mut(field.name()) else {
            continue;
        };
        if bound.total_payload_upper_bound_bytes.is_none()
            && let Some(max_length) = bound.max_single_value_length_bytes
        {
            bound.total_payload_upper_bound_bytes = Some(
                input
                    .rows
                    .checked_mul(max_length)
                    .ok_or_else(|| composed_schedule_error("consumer_projection_literal_payload_overflow", None))?,
            );
        }
    }
    let literal_bytes = bounded_consumer_frame_bytes_with_evidence(input.rows, &literal_schema, &literal_evidence)?;
    let literal_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        literal_bytes,
        analysis.plan,
    )?;
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
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: node.output_schema().clone(),
        final_owned_bytes,
        owner_terms: consumer_owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids,
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn composed_consumer_expression_projection_bound(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    node: &crate::plan::QueryNode,
    projection: &crate::plan::ProjectionNode,
    input: ComposedConsumerBranchBound,
) -> Result<ComposedConsumerBranchBound> {
    let rows = usize::try_from(input.rows)
        .map_err(|_| composed_schedule_error("consumer_projection_rows_exceed_usize", None))?;
    let input_column_evidence = analysis.column_evidence.node(projection.input);
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
            crate::prepared::lower_projection_output_expr(output).map(|expr| (output.field.name().to_string(), expr))
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
    let output_bytes = bounded_consumer_frame_bytes_with_evidence(
        input.rows,
        node.output_schema(),
        &analysis.column_evidence.node(node_id),
    )?;
    let output_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        output_bytes,
        analysis.plan,
    )?;
    let output_term_id = output_term.id.clone();
    let workspace_term = (workspace_bytes > 0)
        .then(|| {
            consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerProjectionExpressionWorkspace,
                AllocationClass::ActivePhase,
                workspace_bytes,
                analysis.plan,
            )
            .map(|term| {
                term.with_component_preflight("expr::dataframe_expression_allocation_preflight", workspace_bytes)
            })
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
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        owner_terms: consumer_owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[cfg(test)]
#[doc(hidden)]
pub fn consumer_peak_group_bytes(group: &[OwnerTermId], owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
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

#[cfg(test)]
#[doc(hidden)]
pub fn consumer_peak_groups_bytes(groups: &[Vec<OwnerTermId>], owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
    groups.iter().try_fold(0_u64, |peak, group| Ok(peak.max(consumer_peak_group_bytes(group, owner_terms)?)))
}

#[doc(hidden)]
pub fn flatten_consumer_peak_groups(groups: &[Vec<OwnerTermId>]) -> Vec<OwnerTermId> {
    let mut seen = HashSet::new();
    groups.iter().flatten().filter(|term_id| seen.insert((*term_id).clone())).cloned().collect()
}

#[doc(hidden)]
pub fn consumer_unary_peak_groups(
    input: &ComposedConsumerBranchBound,
    current_term_ids: impl IntoIterator<Item = OwnerTermId>,
) -> Vec<Vec<OwnerTermId>> {
    let mut groups = input.peak_owner_term_groups.clone();
    let mut operator_group = input.final_owned_term_ids.clone();
    operator_group.extend(current_term_ids);
    groups.push(operator_group);
    groups
}

#[doc(hidden)]
pub fn composed_consumer_union_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    union: &crate::plan::UnionNode,
) -> Result<ComposedConsumerBranchBound> {
    let mut total_rows = 0_u64;
    let mut branch_terms = Vec::new();
    let mut peak_owner_term_groups = Vec::new();
    let mut dominated_owner_term_ids = Vec::new();
    let mut retained_prior_term_ids = Vec::new();
    for input in &union.inputs {
        let branch = composed_consumer_branch_bound(analysis, *input)?;
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
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let union_output = bounded_consumer_frame_bytes_with_evidence(
        total_rows,
        node.output_schema(),
        &analysis.column_evidence.node(node_id),
    )?;
    let union_output_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        union_output,
        analysis.plan,
    )?;
    branch_terms.push(union_output_term);
    let union_output_term_id = branch_terms.last().expect("union output term inserted").id.clone();
    let mut final_peak_group = retained_prior_term_ids;
    final_peak_group.push(union_output_term_id.clone());
    peak_owner_term_groups.push(final_peak_group);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let final_owned_term_ids = vec![union_output_term_id];
    Ok(ComposedConsumerBranchBound {
        rows: total_rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: union_output,
        owner_terms: branch_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids,
        final_owned_term_ids,
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn composed_consumer_aggregate_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    aggregate: &AggregateNode,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(analysis, aggregate.input)?;
    let distinct_workspace = composed_global_nunique_workspace_bytes(aggregate, &input.schema, input.rows)?;
    if distinct_workspace.is_none() && !aggregate_owner_state_is_supported(aggregate) {
        return Err(composed_schedule_error("consumer_aggregate_state_unmodeled", None));
    }
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_schema = node.output_schema();
    let output_rows = if aggregate.spec.group_exprs.is_empty() { 1 } else { input.rows };
    let partial_shape = if aggregate.spec.group_exprs.is_empty() {
        crate::materialization_sizing::PartialAggregateShape::Global
    } else {
        crate::materialization_sizing::PartialAggregateShape::Grouped { input_rows: input.rows }
    };
    let node_evidence = analysis.column_evidence.node(node_id);
    let mut aggregate_terms = aggregate_owner_terms_for_shapes(
        analysis.plan,
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
    let mut peak_owner_term_groups =
        consumer_unary_peak_groups(&input, [partial_term_id.clone(), output_term_id.clone()]);
    let mut terms = input.owner_terms;
    terms.extend([partial_term, output_term]);
    if let Some((distinct_workspace_bytes, operation)) = distinct_workspace {
        let workspace = AllocationOwnerTerm::new(
            OwnerId::Aggregate(node_id),
            OwnerTermRole::AggregateDistinctWorkspace,
            OccurrenceId::PlanLocal(node_id),
            canonical_execution_scope_fingerprint(analysis.plan),
            AllocationClass::ActivePhase,
            distinct_workspace_bytes,
            None,
        )?
        .with_component_preflight(operation, distinct_workspace_bytes);
        let mut workspace_peak = input.final_owned_term_ids.clone();
        workspace_peak.push(workspace.id.clone());
        peak_owner_term_groups.insert(peak_owner_term_groups.len() - 1, workspace_peak);
        terms.push(workspace);
    }
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: output_schema.clone(),
        final_owned_bytes: output_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![output_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

fn composed_global_nunique_workspace_bytes(
    aggregate: &AggregateNode,
    input_schema: &SchemaRef,
    input_rows: u64,
) -> Result<Option<(u64, &'static str)>> {
    if aggregate.spec.grouping_sets.is_some() || !aggregate.spec.group_exprs.is_empty() {
        return Ok(None);
    }
    let mut distincts = aggregate.spec.aggr_exprs.iter().filter(|expr| {
        expr.function == AggregateFunction::NUnique || expr.distinct && expr.function == AggregateFunction::Count
    });
    let Some(distinct) = distincts.next() else {
        return Ok(None);
    };
    if distincts.next().is_some() {
        return Err(composed_schedule_error("consumer_aggregate_state_unmodeled", None)
            .with_fact("aggregate_contract", "global_single_fixed_width_nunique"));
    }
    for expr in &aggregate.spec.aggr_exprs {
        if expr.filter.is_some() {
            return Err(composed_schedule_error("consumer_aggregate_state_unmodeled", None)
                .with_fact("aggregate_contract", "global_single_fixed_width_nunique"));
        }
        if expr.function == AggregateFunction::NUnique || expr.distinct && expr.function == AggregateFunction::Count {
            continue;
        } else if expr.function != AggregateFunction::Count || expr.distinct {
            return Err(composed_schedule_error("consumer_aggregate_state_unmodeled", None)
                .with_fact("aggregate_contract", "global_single_fixed_width_nunique"));
        }
    }
    let Expr::Column(column) = &distinct.expr else {
        return Err(composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
            .with_fact("aggregate_owner_reason", "nunique_argument_not_column"));
    };
    let field = input_schema.field_with_name(column.name()).map_err(|_| {
        composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
            .with_fact("aggregate_owner_reason", "nunique_input_column_missing")
            .with_fact("column", column.name())
    })?;
    if field.is_nullable() {
        return Err(composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
            .with_fact("aggregate_owner_reason", "nunique_nullable_input_unmodeled")
            .with_fact("column", field.name())
            .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    let type_id = arrow_fixed_width_cudf_type_id(field.data_type()).ok_or_else(|| {
        composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
            .with_fact("aggregate_owner_reason", "nunique_variable_width_input_unmodeled")
            .with_fact("column", field.name())
            .with_fact("data_type", format!("{:?}", field.data_type()))
    })?;
    let rows = usize::try_from(input_rows).map_err(|_| {
        composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
            .with_fact("aggregate_owner_reason", "nunique_input_rows_exceed_usize")
            .with_fact("input_rows", input_rows)
    })?;
    let preflight =
        cudf::ScalarReductionDistinctWorkspacePreflight::for_metadata(rows, cudf::DataType::new(type_id), 0).map_err(
            |error| {
                composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
                    .with_fact(
                        "aggregate_owner_reason",
                        error.detail_value("reason").unwrap_or("libcudf_nunique_workspace_preflight_failed"),
                    )
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                    .with_fact("input_rows", input_rows)
            },
        )?;
    u64::try_from(preflight.active_workspace_peak_bytes())
        .map(|bytes| Some((bytes, "series::scalar_reduction_distinct_workspace_preflight")))
        .map_err(|_| {
            composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
                .with_fact("aggregate_owner_reason", "nunique_workspace_exceeds_u64")
                .with_fact("input_rows", input_rows)
        })
}

#[doc(hidden)]
pub fn has_reachable_sort(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids();
    plan.nodes().iter().any(|node| !unreachable.contains(&node.id) && matches!(node.kind, PlanNode::Sort(_)))
}

#[doc(hidden)]
pub fn composed_consumer_sort_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    sort: &crate::plan::SortNode,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(analysis, sort.input)?;
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_schema = node.output_schema().clone();
    let scope = canonical_execution_scope_fingerprint(analysis.plan);
    let sort_terms = [
        OwnerTermRole::SortKeyInput,
        OwnerTermRole::SortKeyOutput,
        OwnerTermRole::SortSequence,
        OwnerTermRole::SortComparatorState,
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
        .filter(|term_id| matches!(term_id.role, OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask))
        .cloned()
        .collect();
    let (output_frame_bytes, output_null_mask_bytes) =
        bounded_consumer_frame_components(input.rows, &output_schema, Some(&analysis.column_evidence.node(node_id)))?;
    let final_owned_bytes = output_frame_bytes
        .checked_add(output_null_mask_bytes)
        .ok_or_else(|| composed_schedule_error("consumer_sort_output_bytes_overflow", None))?;
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, sort_terms.clone());
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    Ok(ComposedConsumerBranchBound {
        rows: input.rows,
        schema: output_schema,
        final_owned_bytes,
        owner_terms: input.owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids,
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn composed_consumer_limit_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    limit: &crate::plan::LimitNode,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(analysis, limit.input)?;
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
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

    if matches!(analysis.plan.node(limit.input).map(|node| &node.kind), Some(PlanNode::Sort(_))) {
        let scope = canonical_execution_scope_fingerprint(analysis.plan);
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
        let (output_frame_bytes, output_null_mask_bytes) = bounded_consumer_frame_components(
            output_rows,
            &output_schema,
            Some(&analysis.column_evidence.node(node_id)),
        )?;
        let final_owned_bytes = output_frame_bytes
            .checked_add(output_null_mask_bytes)
            .ok_or_else(|| composed_schedule_error("consumer_limit_output_bytes_overflow", None))?;
        let peak_owner_term_groups = consumer_unary_peak_groups(&input, limit_terms.clone());
        let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
        return Ok(ComposedConsumerBranchBound {
            rows: output_rows,
            schema: output_schema,
            final_owned_bytes,
            owner_terms: input.owner_terms,
            peak_owner_term_groups,
            peak_owner_term_ids,
            dominated_owner_term_ids: input.dominated_owner_term_ids,
            final_owned_term_ids,
            allocation_state: AttemptReservationConsumerState::Bounded,
            consumed_occurrences: Vec::new(),
        });
    }

    let output_bytes = bounded_consumer_frame_bytes_with_evidence(
        output_rows,
        &output_schema,
        &analysis.column_evidence.node(node_id),
    )?;
    let limit_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        output_bytes,
        analysis.plan,
    )?;
    let limit_term_id = limit_term.id.clone();
    let peak_owner_term_groups = consumer_unary_peak_groups(&input, [limit_term_id.clone()]);
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut terms = input.owner_terms;
    terms.push(limit_term);
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: output_schema,
        final_owned_bytes: output_bytes,
        owner_terms: terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids: input.dominated_owner_term_ids,
        final_owned_term_ids: vec![limit_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn composed_consumer_join_peak_bytes(
    analysis: &mut ComposedConsumerAnalysis<'_>,
    node_id: NodeId,
    join: &JoinNode,
) -> Result<ComposedConsumerBranchBound> {
    if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti)
        || join.spec.residual_predicate.is_some()
        || join.spec.keys.is_empty()
        || (matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
            && join.spec.build_side != JoinSide::Right)
    {
        return Err(composed_schedule_error("consumer_join_topology_unmodeled", None)
            .with_fact("node_id", node_id)
            .with_fact("join_type", format!("{:?}", join.spec.join_type))
            .with_fact("build_side", format!("{:?}", join.spec.build_side))
            .with_fact("join_key_count", join.spec.keys.len())
            .with_fact("residual_predicate", join.spec.residual_predicate.is_some()));
    }
    let left = composed_consumer_branch_bound(analysis, join.inputs[0])?;
    let right = composed_consumer_branch_bound(analysis, join.inputs[1])?;
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
    let build_column_evidence = analysis.column_evidence.node(build_node_id);
    let probe_column_evidence = analysis.column_evidence.node(probe_node_id);
    let left_column_evidence = analysis.column_evidence.node(join.inputs[0]);
    let right_column_evidence = analysis.column_evidence.node(join.inputs[1]);
    let left_join_keys = join.spec.keys.iter().map(|key| key.left.name().to_owned()).collect::<BTreeSet<_>>();
    let right_join_keys = join.spec.keys.iter().map(|key| key.right.name().to_owned()).collect::<BTreeSet<_>>();
    let left_is_unique = left_column_evidence.is_unique_on(&left_join_keys);
    let right_is_unique = right_column_evidence.is_unique_on(&right_join_keys);
    let key_metadata = build_keys
        .iter()
        .map(|key| hash_join_key_metadata(&build.schema, &build_column_evidence, build.rows, key.name()))
        .collect::<Result<Vec<_>>>()
        .map_err(|error| {
            error
                .with_fact("node_id", node_id)
                .with_fact("owner_kind", "consumer")
                .with_fact("owner_node_id", node_id)
                .with_fact(
                    "owner_execution_scope_fingerprint",
                    crate::admission::log_safe_fingerprint(&canonical_execution_scope_fingerprint(analysis.plan)),
                )
                .with_fact("left_rows", left.rows)
                .with_fact("right_rows", right.rows)
        })?;
    for key in &join.spec.keys {
        let probe_key = match join.spec.build_side {
            JoinSide::Left => &key.right,
            JoinSide::Right => &key.left,
        };
        hash_join_key_metadata(&probe.schema, &probe_column_evidence, probe.rows, probe_key.name()).map_err(
            |error| {
                error
                    .with_fact("node_id", node_id)
                    .with_fact("owner_kind", "consumer")
                    .with_fact("owner_node_id", node_id)
                    .with_fact(
                        "owner_execution_scope_fingerprint",
                        crate::admission::log_safe_fingerprint(&canonical_execution_scope_fingerprint(analysis.plan)),
                    )
                    .with_fact("left_rows", left.rows)
                    .with_fact("right_rows", right.rows)
            },
        )?;
    }
    let output_rows = match join.spec.join_type {
        JoinType::Inner if join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) => left.rows,
        JoinType::Inner if join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Right) => right.rows,
        JoinType::Inner if left_is_unique && right_is_unique => left.rows.min(right.rows),
        JoinType::Inner if left_is_unique => right.rows,
        JoinType::Inner if right_is_unique => left.rows,
        JoinType::Inner => left
            .rows
            .checked_mul(right.rows)
            .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?,
        JoinType::Left if join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) => left.rows,
        JoinType::Left if left_is_unique => left
            .rows
            .checked_add(right.rows)
            .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?,
        JoinType::Left => left
            .rows
            .checked_mul(right.rows.max(1))
            .ok_or_else(|| composed_schedule_error("consumer_join_output_rows_overflow", None))?,
        JoinType::LeftSemi | JoinType::LeftAnti => left.rows,
        _ => return Err(composed_schedule_error("consumer_join_topology_unmodeled", None)),
    };
    let output_rows_usize = usize::try_from(output_rows)
        .map_err(|_| composed_schedule_error("consumer_join_output_rows_exceed_usize", None))?;
    let (
        hash_table_bytes,
        pair_indices_bytes,
        filtered_workspace_bytes,
        filtered_probe_state_bytes,
        hash_component_operation,
        pair_component_operation,
    ) = if matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti) {
        let null_equality = match join.spec.null_equality {
            crate::plan::JoinNullEquality::NullsDoNotMatch => cudf::NullEquality::Unequal,
            crate::plan::JoinNullEquality::NullsMatch => cudf::NullEquality::Equal,
        };
        let options = cudf::ops::join::FilteredJoinOptions::new().null_equality(null_equality);
        let build_preflight =
            cudf::ops::join::filtered_join_allocation_preflight_for_metadata(&key_metadata, build.rows, options)
                .map_err(|error| {
                    composed_schedule_error("consumer_join_filtered_build_preflight_unavailable", None)
                        .with_fact("cudf_error", error.to_string())
                        .with_fact("required_contract", cudf::ops::join::FILTERED_JOIN_ALLOCATION_PREFLIGHT_CONTRACT)
                })?;
        let probe_preflight = cudf::ops::join::filtered_join_probe_allocation_preflight(probe.rows, output_rows)
            .map_err(|error| {
                composed_schedule_error("consumer_join_filtered_probe_preflight_unavailable", None)
                    .with_fact("cudf_error", error.to_string())
                    .with_fact("required_contract", cudf::ops::join::FILTERED_JOIN_ALLOCATION_PREFLIGHT_CONTRACT)
            })?;
        let build_frame_bytes =
            bounded_consumer_frame_bytes_with_evidence(build.rows, &build.schema, &build_column_evidence)?;
        (
            u64::try_from(build_preflight.active_build_phase_peak_bytes())
                .map_err(|_| composed_schedule_error("consumer_join_filtered_build_preflight_exceeds_u64", None))?,
            u64::try_from(probe_preflight.gather_map_bytes())
                .map_err(|_| composed_schedule_error("consumer_join_pair_indices_exceed_u64", None))?,
            build_frame_bytes,
            u64::try_from(probe_preflight.retained_probe_state_bytes())
                .map_err(|_| composed_schedule_error("consumer_join_filtered_probe_state_exceeds_u64", None))?,
            "ops::join::filtered_join_allocation_preflight_for_metadata",
            Some("ops::join::filtered_join_allocation_preflight_for_metadata"),
        )
    } else {
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
        let pair_indices_bytes =
            cudf::DataFrameHashJoinPairIndices::reservation_upper_bound_bytes_for_row_bound(output_rows_usize);
        (
            u64::try_from(hash_preflight.active_phase_peak_bytes())
                .map_err(|_| composed_schedule_error("consumer_join_hash_preflight_exceeds_u64", None))?,
            u64::try_from(pair_indices_bytes)
                .map_err(|_| composed_schedule_error("consumer_join_pair_indices_exceed_u64", None))?,
            0,
            0,
            "ops::join::hash_join_allocation_preflight_for_metadata",
            None,
        )
    };
    let node =
        analysis.plan.node(node_id).ok_or_else(|| composed_schedule_error("consumer_missing_plan_node", None))?;
    let output_bytes = bounded_consumer_frame_bytes_with_evidence(
        output_rows,
        node.output_schema(),
        &analysis.column_evidence.node(node_id),
    )
    .map_err(|error| {
        error
            .with_fact("node_id", node_id)
            .with_fact("left_rows", left.rows)
            .with_fact("right_rows", right.rows)
            .with_fact("output_rows", output_rows)
    })?;
    let hash_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerHashTable,
        AllocationClass::ActivePhase,
        hash_table_bytes,
        analysis.plan,
    )?
    .with_component_preflight(hash_component_operation, hash_table_bytes);
    let mut pair_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerPairIndices,
        AllocationClass::ActivePhase,
        pair_indices_bytes,
        analysis.plan,
    )?;
    if let Some(operation) = pair_component_operation {
        pair_term = pair_term.with_component_owner_operation(operation);
    }
    let output_term = consumer_owner_term(
        node_id,
        OwnerTermRole::ConsumerOutput,
        AllocationClass::ActivePhase,
        output_bytes,
        analysis.plan,
    )?;
    let mut current_terms = Vec::new();
    if filtered_workspace_bytes > 0 {
        current_terms.push(consumer_owner_term(
            node_id,
            OwnerTermRole::ConsumerWorkspace,
            AllocationClass::ActivePhase,
            filtered_workspace_bytes,
            analysis.plan,
        )?);
    }
    current_terms.push(hash_term);
    if filtered_probe_state_bytes > 0 {
        current_terms.push(
            consumer_owner_term(
                node_id,
                OwnerTermRole::ConsumerPredicateMask,
                AllocationClass::ActivePhase,
                filtered_probe_state_bytes,
                analysis.plan,
            )?
            .with_component_owner_operation("ops::join::filtered_join_allocation_preflight_for_metadata"),
        );
    }
    current_terms.extend([pair_term, output_term]);
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
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
    let mut dominated_owner_term_ids = left.dominated_owner_term_ids;
    dominated_owner_term_ids.extend(right.dominated_owner_term_ids);
    Ok(ComposedConsumerBranchBound {
        rows: output_rows,
        schema: node.output_schema().clone(),
        final_owned_bytes: output_bytes,
        owner_terms: consumer_owner_terms,
        peak_owner_term_groups,
        peak_owner_term_ids,
        dominated_owner_term_ids,
        final_owned_term_ids: vec![final_owned_term_id],
        allocation_state: AttemptReservationConsumerState::Bounded,
        consumed_occurrences: Vec::new(),
    })
}

#[doc(hidden)]
pub fn consumer_owner_term(
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

#[doc(hidden)]
pub fn hash_join_key_metadata(
    schema: &SchemaRef,
    evidence: &PlanColumnEvidence,
    rows: u64,
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
        let payload_upper_bound_bytes =
            match (bound.max_single_value_length_bytes, bound.total_payload_upper_bound_bytes) {
                (Some(max_single), Some(total)) => rows.saturating_mul(max_single).min(total),
                (Some(max_single), None) => rows.checked_mul(max_single).ok_or_else(|| {
                    composed_schedule_error("consumer_join_string_payload_overflow", None)
                        .with_fact("column", field_name)
                })?,
                (None, Some(total)) => total,
                (None, None) => {
                    return Err(composed_schedule_error("consumer_join_string_payload_bound_missing", None)
                        .with_fact("column", field_name));
                },
            };
        let payload_upper_bound_bytes = usize::try_from(payload_upper_bound_bytes).map_err(|_| {
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

#[doc(hidden)]
pub fn arrow_fixed_width_cudf_type_id(data_type: &DataType) -> Option<cudf::TypeId> {
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
        DataType::Decimal32(_, _) | DataType::Decimal64(_, _) => {
            crate::datatypes::cudf_dtype_from_arrow_type(data_type)?.id()
        },
        DataType::Decimal128(_, _) => cudf::TypeId::Decimal128,
        _ => return None,
    })
}

#[doc(hidden)]
pub fn bounded_consumer_boolean_mask_bytes(rows: u64) -> Result<u64> {
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "predicate",
        DataType::Boolean,
        false,
    )]));
    bounded_consumer_frame_bytes(rows, &schema)
}

#[doc(hidden)]
pub fn bounded_consumer_frame_bytes(rows: u64, schema: &SchemaRef) -> Result<u64> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| fixed_width_consumer_output_column(field.data_type()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| composed_schedule_error("consumer_variable_width_frame_unmodeled", None))?;
    let bytes = crate::materialization_sizing::join_output_upper_bound_bytes(rows, &columns).map_err(|error| {
        let reason = if error.code() == QueryErrorCode::SizeOverflow {
            "consumer_frame_bytes_overflow"
        } else {
            "consumer_frame_upper_bound_unavailable"
        };
        composed_schedule_error(reason, None)
    })?;
    Ok(bytes)
}

#[doc(hidden)]
pub fn bounded_consumer_frame_bytes_with_evidence(
    rows: u64,
    schema: &SchemaRef,
    evidence: &PlanColumnEvidence,
) -> Result<u64> {
    bounded_frame_bytes_with_evidence(rows, schema, evidence, false)
}

#[doc(hidden)]
pub fn bounded_retained_output_frame_bytes_with_evidence(
    rows: u64,
    schema: &SchemaRef,
    evidence: &PlanColumnEvidence,
) -> Result<u64> {
    bounded_frame_bytes_with_evidence(rows, schema, evidence, false)
}

fn bounded_frame_bytes_with_evidence(
    rows: u64,
    schema: &SchemaRef,
    evidence: &PlanColumnEvidence,
    require_max_string_length: bool,
) -> Result<u64> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| output_column_with_evidence(field, evidence, require_max_string_length))
        .collect::<Result<Vec<_>>>()?;
    let bytes = crate::materialization_sizing::join_output_upper_bound_bytes(rows, &columns).map_err(|error| {
        let reason = if error.code() == QueryErrorCode::SizeOverflow {
            "consumer_frame_bytes_overflow"
        } else {
            "consumer_frame_upper_bound_unavailable"
        };
        composed_schedule_error(reason, None)
    })?;
    Ok(bytes)
}

#[doc(hidden)]
pub fn consumer_output_column_with_optional_evidence(
    field: &arrow::datatypes::Field,
    evidence: Option<&PlanColumnEvidence>,
) -> Result<crate::materialization_sizing::JoinOutputColumnUpperBoundInput> {
    match evidence {
        Some(evidence) => consumer_output_column_with_evidence(field, evidence),
        None => fixed_width_consumer_output_column(field.data_type())
            .ok_or_else(|| owner_term_error("aggregate_output_type_unmodeled", None, None)),
    }
}

#[doc(hidden)]
pub fn consumer_output_column_with_evidence(
    field: &arrow::datatypes::Field,
    evidence: &PlanColumnEvidence,
) -> Result<crate::materialization_sizing::JoinOutputColumnUpperBoundInput> {
    output_column_with_evidence(field, evidence, false)
}

fn output_column_with_evidence(
    field: &arrow::datatypes::Field,
    evidence: &PlanColumnEvidence,
    require_max_string_length: bool,
) -> Result<crate::materialization_sizing::JoinOutputColumnUpperBoundInput> {
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
        let max_single_value_length_bytes = bound.max_single_value_length_bytes;
        if require_max_string_length && max_single_value_length_bytes.is_none() {
            return Err(composed_schedule_error("consumer_string_max_length_evidence_missing", None)
                .with_fact("column", field.name())
                .with_fact(
                    "total_payload_upper_bound_bytes",
                    bound
                        .total_payload_upper_bound_bytes
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                )
                .with_fact(
                    "distinct_payload_upper_bound_bytes",
                    bound
                        .distinct_payload_upper_bound_bytes
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                )
                .with_fact(
                    "distinct_value_count_upper_bound",
                    bound
                        .distinct_value_count_upper_bound
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                )
                .with_fact("string_payload_lineages", format!("{:?}", bound.lineages)));
        }
        return Ok(crate::materialization_sizing::JoinOutputColumnUpperBoundInput {
            data_type: field.data_type().clone(),
            input_max_string_length_bytes: max_single_value_length_bytes,
            input_total_string_bytes: bound.total_payload_upper_bound_bytes,
        });
    }
    Err(composed_schedule_error("consumer_variable_width_frame_unmodeled", None)
        .with_fact("column", field.name())
        .with_fact("data_type", field.data_type()))
}

#[doc(hidden)]
pub fn resident_side_input_frame_upper_bound_bytes(plan: &QueryPlan) -> Result<u64> {
    let rows = query_plan_hard_row_upper_bound(plan).ok_or_else(|| {
        composed_schedule_error("side_input_row_bound_unavailable", None)
            .with_fact("phase", "side_input_frame")
            .with_fact("row_bound_source", "native_query_plan")
    })?;
    if plan.output_schema().fields().is_empty() {
        return Err(composed_schedule_error("side_input_schema_empty", None).with_fact("phase", "side_input_frame"));
    }
    let columns = resident_side_input_columns(plan)?;
    let bytes = crate::materialization_sizing::join_output_upper_bound_bytes(rows, &columns).map_err(|error| {
        let reason = if error.code() == QueryErrorCode::SizeOverflow {
            "side_input_frame_bytes_overflow"
        } else {
            "side_input_frame_bound_unavailable"
        };
        composed_schedule_error(reason, None)
            .with_fact("phase", "side_input_frame")
            .with_fact("sizing_error", error.code())
    })?;
    Ok(bytes)
}

#[doc(hidden)]
pub fn resident_side_input_columns(
    plan: &QueryPlan,
) -> Result<Vec<crate::materialization_sizing::JoinOutputColumnUpperBoundInput>> {
    let evidence = PlanColumnEvidenceAnalysis::for_plan(plan, &HashMap::new())?.root(plan);
    plan.output_schema()
        .fields()
        .iter()
        .map(|field| {
            if let Some(column) = fixed_width_consumer_output_column(field.data_type()) {
                return Ok(column);
            }
            if !matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
                return Err(composed_schedule_error("side_input_schema_unmodeled", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
            }
            let Some(bound) = evidence.string_payload_bound(field.name()) else {
                return Err(composed_schedule_error("side_input_schema_unmodeled", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
            };
            let (Some(max_len), Some(total_bytes)) =
                (bound.max_single_value_length_bytes, bound.total_payload_upper_bound_bytes)
            else {
                return Err(composed_schedule_error("side_input_string_payload_incomplete", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
            };
            if bound.lineages.is_empty() {
                return Err(composed_schedule_error("side_input_string_payload_lineage_missing", None)
                    .with_fact("phase", "side_input_frame")
                    .with_fact("column", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
            }
            Ok(crate::materialization_sizing::JoinOutputColumnUpperBoundInput {
                data_type: field.data_type().clone(),
                input_max_string_length_bytes: Some(max_len),
                input_total_string_bytes: Some(total_bytes),
            })
        })
        .collect()
}

#[doc(hidden)]
pub fn fixed_width_consumer_output_column(
    data_type: &DataType,
) -> Option<crate::materialization_sizing::JoinOutputColumnUpperBoundInput> {
    arrow_fixed_width_cudf_type_id(data_type).map(|_| crate::materialization_sizing::JoinOutputColumnUpperBoundInput {
        data_type: data_type.clone(),
        input_max_string_length_bytes: None,
        input_total_string_bytes: None,
    })
}

#[doc(hidden)]
pub fn checked_consumer_add(left: u64, right: u64, reason: &'static str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| composed_schedule_error(reason, None))
}

#[doc(hidden)]
pub fn generic_graph_execution_blocker() -> ReservationRequirement {
    ReservationRequirement::blocked_by_lower_level(
        ReservationKind::OperatorOutput,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
        crate::capability::ContractQuality::ExactPreflight,
    )
}

#[doc(hidden)]
pub fn composed_schedule_error(schedule_reason: &'static str, occurrence_id: Option<&str>) -> QueryError {
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

#[doc(hidden)]
pub fn sort_owner_terms(
    descriptor: &SortMetadataDescriptor,
    components: SortPreflight,
) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = descriptor.execution_scope_fingerprint.clone();
    let owner = OwnerId::Sort(descriptor.node_id);
    let component_operation = components.component_owner_operation;
    let component_peak = components.active_phase_peak_bytes;
    [
        (OwnerTermRole::SortKeyInput, AllocationClass::ActivePhase, components.key_input_bytes),
        (OwnerTermRole::SortKeyOutput, AllocationClass::ActivePhase, components.key_output_bytes),
        (OwnerTermRole::SortSequence, AllocationClass::ActivePhase, components.sequence_bytes),
        (OwnerTermRole::SortComparatorState, AllocationClass::ActivePhase, components.comparator_state_bytes),
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
        .map(|term| term.with_component_preflight(component_operation, component_peak))
    })
    .collect()
}

#[doc(hidden)]
pub fn limit_owner_terms(
    descriptor: &LimitMetadataDescriptor,
    components: LimitRangeGatherPreflight,
) -> Result<Vec<AllocationOwnerTerm>> {
    let scope = descriptor.execution_scope_fingerprint.clone();
    let owner = OwnerId::Limit(descriptor.node_id);
    let component_operation = components.component_owner_operation;
    let component_peak = components.active_phase_peak_bytes;
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
        .map(|term| term.with_component_preflight(component_operation, component_peak))
    })
    .collect()
}

#[doc(hidden)]
pub fn validate_sort_metadata_descriptor(descriptor: &SortMetadataDescriptor) -> Result<()> {
    if descriptor.execution_scope_fingerprint.is_empty() {
        return Err(sort_owner_error("missing_scope", "sort_scope"));
    }
    if descriptor.stable {
        return Err(sort_owner_error("stable_sort_unsupported", "sort_dispatch"));
    }
    validate_sort_key_metadata(&descriptor.keys, descriptor.num_rows)?;
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

fn validate_sort_key_metadata(keys: &[SortKeyMetadataDescriptor], num_rows: usize) -> Result<()> {
    if keys.is_empty() {
        return Err(sort_owner_error("sort_key_missing", "sort_key"));
    }
    for key in keys {
        if key.null_count > num_rows {
            return Err(sort_owner_error("null_count_exceeds_row_count", "sort_key"));
        }
        let key_type = key.key_type.id();
        let radix = key_type.is_signed_int()
            || key_type.is_unsigned_int()
            || key_type.is_timestamp()
            || key_type.is_duration()
            || key_type.is_float();
        let lexicographic = radix || key_type.is_boolean() || key_type.is_decimal() || key_type == cudf::TypeId::String;
        if !radix && !lexicographic {
            return Err(sort_owner_error("key_type_not_lexicographic_dispatchable", "sort_key"));
        }
    }
    Ok(())
}

#[doc(hidden)]
pub fn sort_metadata_for_composed_consumer(
    plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
) -> Result<Vec<SortMetadataDescriptor>> {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let sorts = plan
        .nodes()
        .iter()
        .filter(|node| !unreachable.contains(&node.id))
        .filter_map(|node| match &node.kind {
            PlanNode::Sort(sort) => Some((node.id, sort)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if sorts.is_empty() {
        return Err(sort_owner_error("sort_node_missing", "sort_plan"));
    }
    let resident_rows = producer_outputs
        .iter()
        .map(|output| (output.occurrence_id.clone(), output.row_count_upper_bound))
        .collect::<HashMap<_, _>>();
    let resident_evidence = producer_outputs
        .iter()
        .map(|output| (output.occurrence_id.clone(), output.column_evidence.clone()))
        .collect::<HashMap<_, _>>();
    let column_evidence =
        PlanColumnEvidenceAnalysis::for_plan_with_resident_rows(plan, &resident_evidence, &resident_rows)?;
    let scope = canonical_execution_scope_fingerprint(plan);
    sorts
        .into_iter()
        .map(|(sort_node_id, sort_node)| {
            if sort_node.spec.expressions.is_empty() {
                return Err(sort_owner_error("sort_key_missing", "sort_key"));
            }
            let input_schema = plan
                .node(sort_node.input)
                .ok_or_else(|| sort_owner_error("sort_input_missing", "sort_input"))?
                .output_schema();
            let row_count = crate::graph_memory::query_plan_node_hard_row_upper_bound_with_resident_rows(
                plan,
                sort_node.input,
                &resident_rows,
            )
            .ok_or_else(|| sort_owner_error("sort_input_row_bound_unavailable", "sort_input"))?;
            let num_rows =
                usize::try_from(row_count).map_err(|_| sort_owner_error("row_bound_exceeds_usize", "sort_input"))?;
            let keys = sort_node
                .spec
                .expressions
                .iter()
                .map(|sort_expr| {
                    let key_column = alias_wrapped_column(&sort_expr.expr)
                        .ok_or_else(|| sort_owner_error("sort_key_expression_unmodeled", "sort_key"))?;
                    let field = input_schema.field_with_name(key_column.name()).map_err(|_| {
                        sort_owner_error("sort_key_column_missing", "sort_key").with_fact("column", key_column.name())
                    })?;
                    let key_type = cudf_dtype_from_arrow_type(field.data_type())
                        .ok_or_else(|| sort_owner_error("key_type_unavailable", "sort_key"))?;
                    Ok(SortKeyMetadataDescriptor {
                        key_type,
                        null_count: if field.is_nullable() { num_rows } else { 0 },
                        key_order: if sort_expr.descending {
                            cudf::ops::sorting::Order::Descending
                        } else {
                            cudf::ops::sorting::Order::Ascending
                        },
                        null_precedence: if sort_expr.nulls_first {
                            cudf::ops::sorting::NullOrder::Before
                        } else {
                            cudf::ops::sorting::NullOrder::After
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            validate_sort_key_metadata(&keys, num_rows)?;
            let sort_output_schema = plan
                .node(sort_node_id)
                .ok_or_else(|| sort_owner_error("sort_node_missing", "sort_plan"))?
                .output_schema();
            let (output_frame_bytes, output_null_mask_bytes) = bounded_consumer_frame_components(
                row_count,
                sort_output_schema,
                Some(&column_evidence.node(sort_node_id)),
            )?;
            let mut limits = plan.nodes().iter().filter_map(|node| match &node.kind {
                PlanNode::Limit(limit) if limit.input == sort_node_id && !unreachable.contains(&node.id) => {
                    Some((node.id, limit))
                },
                _ => None,
            });
            let limit = limits.next();
            if limits.next().is_some() {
                return Err(sort_owner_error("multiple_limit_consumers", "limit_plan"));
            }
            let limit = limit
                .map(|(limit_node_id, limit_node)| {
                    let fetch =
                        limit_node.spec.fetch.ok_or_else(|| limit_owner_error("fetch_unbounded", "limit_rows"))?;
                    let _ = limit_node
                        .spec
                        .skip
                        .checked_add(fetch)
                        .ok_or_else(|| limit_owner_error("skip_fetch_overflow", "limit_rows"))?;
                    let skip = u64::try_from(limit_node.spec.skip)
                        .map_err(|_| limit_owner_error("skip_exceeds_u64", "limit_rows"))?;
                    let fetch =
                        u64::try_from(fetch).map_err(|_| limit_owner_error("fetch_exceeds_u64", "limit_rows"))?;
                    let output_rows_u64 = row_count.saturating_sub(skip).min(fetch);
                    let output_rows = usize::try_from(output_rows_u64)
                        .map_err(|_| limit_owner_error("output_rows_exceed_usize", "limit_rows"))?;
                    let output_schema = plan
                        .node(limit_node_id)
                        .ok_or_else(|| limit_owner_error("limit_node_missing", "limit_output"))?
                        .output_schema();
                    let output_columns = output_schema
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
                    let (output_frame_bytes, output_null_mask_bytes) = bounded_consumer_frame_components(
                        output_rows_u64,
                        output_schema,
                        Some(&column_evidence.node(limit_node_id)),
                    )?;
                    Ok(LimitMetadataDescriptor {
                        node_id: limit_node_id,
                        execution_scope_fingerprint: scope.clone(),
                        input_rows: row_count,
                        output_rows,
                        output_columns,
                        output_frame_bytes,
                        output_null_mask_bytes,
                    })
                })
                .transpose()?;
            let retained_node_id = limit.as_ref().map_or(sort_node_id, |limit| limit.node_id);
            Ok(SortMetadataDescriptor {
                node_id: sort_node_id,
                execution_scope_fingerprint: scope.clone(),
                num_rows,
                keys,
                stable: false,
                output_frame_bytes,
                output_null_mask_bytes,
                limit,
                retained_output: graph_output_is_final(plan, retained_node_id),
            })
        })
        .collect()
}

#[doc(hidden)]
pub fn bounded_consumer_frame_components(
    rows: u64,
    schema: &SchemaRef,
    evidence: Option<&PlanColumnEvidence>,
) -> Result<(u64, u64)> {
    let mut frame = 0_u64;
    let mut null_mask = 0_u64;
    for field in schema.fields() {
        let empty_evidence = PlanColumnEvidence::default();
        let column =
            consumer_output_column_with_evidence(field, evidence.unwrap_or(&empty_evidence)).map_err(|error| {
                let reason = match error.fact_value("schedule_reason") {
                    Some("consumer_string_payload_evidence_missing") => "sort_output_string_payload_evidence_missing",
                    Some("consumer_string_payload_lineage_missing") => "sort_output_string_payload_lineage_missing",
                    _ => "sort_output_dtype_unmodeled",
                };
                sort_owner_error(reason, "sort_output").with_fact("column", field.name())
            })?;
        let data = if matches!(column.data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) {
            let payload = match (column.input_max_string_length_bytes, column.input_total_string_bytes) {
                (Some(max), Some(total)) => rows
                    .checked_mul(max)
                    .ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?
                    .min(total),
                (Some(max), None) => rows
                    .checked_mul(max)
                    .ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?,
                (None, Some(total)) => total,
                (None, None) => {
                    return Err(sort_owner_error("sort_output_string_payload_evidence_missing", "sort_output")
                        .with_fact("column", field.name()));
                },
            };
            let offsets = rows
                .checked_add(1)
                .and_then(|value| value.checked_mul(crate::materialization_sizing::STRING_OFFSET_BYTES_CONSERVATIVE))
                .ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?;
            payload.checked_add(offsets).ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?
        } else {
            let width = fixed_width_byte_size(&column.data_type)
                .ok_or_else(|| sort_owner_error("sort_output_dtype_unmodeled", "sort_output"))?;
            rows.checked_mul(width).ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?
        };
        frame = frame.checked_add(data).ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?;
        let mask = if rows == 0 || !field.is_nullable() {
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

#[doc(hidden)]
pub fn fixed_width_byte_size(data_type: &DataType) -> Option<u64> {
    arrow_fixed_width_cudf_type_id(data_type)?;
    crate::datatypes::arrow_fixed_width_byte_size(data_type)
}

#[doc(hidden)]
pub fn cugraph_graph_execution_capability() -> ReservationRequirement {
    ReservationRequirement::modeled(
        ReservationKind::GraphAlgorithm,
        "cugraph_graph_execution",
        "cugraph_graph_execution_contract",
        crate::capability::ContractQuality::CapabilityProof,
    )
}

#[doc(hidden)]
pub fn is_generic_graph_execution_blocker(requirement: &ReservationRequirement) -> bool {
    requirement == &generic_graph_execution_blocker()
}
