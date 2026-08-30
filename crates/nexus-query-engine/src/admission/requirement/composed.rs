use super::*;

#[derive(Debug, Clone)]
pub(crate) struct ComposedGraphProducerOutputEvidence {
    pub(crate) occurrence_id: String,
    pub(crate) row_count_upper_bound: u64,
    pub(crate) column_evidence: PlanColumnEvidence,
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedComposedProducer {
    pub(crate) occurrence_id: String,
    pub(crate) execution_scope_fingerprint: String,
    pub(crate) retained_output_term_id: OwnerTermId,
    pub(crate) row_count_upper_bound: u64,
    pub(super) output_column_evidence: PlanColumnEvidence,
    pub(crate) phase_term_groups: Vec<Vec<OwnerTermId>>,
    pub(crate) alternative_phase_term_ids: Vec<OwnerTermId>,
    pub(crate) concurrent_side_input_term_ids: Vec<OwnerTermId>,
    pub(crate) dominated_term_ids: Vec<OwnerTermId>,
    pub(crate) nested_schedule: bool,
    pub(crate) proof: QueryPlanCapabilityProof,
}

#[derive(Debug, Clone)]
pub(crate) struct ComposedConsumerBranchBound {
    pub(crate) rows: u64,
    pub(crate) schema: SchemaRef,
    /// Bytes allocated by this consumer branch that remain live at its root.
    /// Producer result handles are intentionally excluded: the schedule fold
    /// retains them outside the consumer phase.
    pub(crate) final_owned_bytes: u64,
    /// Maximum additional bytes owned while this branch is materialized.
    pub(crate) additional_peak_bytes: u64,
    pub(crate) owner_terms: Vec<AllocationOwnerTerm>,
    /// Alternative consumer lifetime peaks; each group is summed and the
    /// schedule takes the maximum across groups.
    pub(crate) peak_owner_term_groups: Vec<Vec<OwnerTermId>>,
    /// Flattened view retained for branch-local dominance decisions.
    pub(crate) peak_owner_term_ids: Vec<OwnerTermId>,
    pub(crate) dominated_owner_term_ids: Vec<OwnerTermId>,
    pub(crate) final_owned_term_ids: Vec<OwnerTermId>,
    pub(crate) allocation_state: AttemptReservationConsumerState,
    pub(crate) consumed_occurrences: Vec<String>,
}

#[cfg(test)]
pub(crate) fn composed_consumer_additional_peak_bytes_for_plan(
    consumer_plan: &QueryPlan,
    producer_outputs: &[ComposedGraphProducerOutputEvidence],
) -> Result<(u64, Vec<AllocationOwnerTerm>)> {
    let bound = composed_consumer_bound_with_peak_terms(consumer_plan, producer_outputs, &[])?;
    Ok((bound.additional_peak_bytes, bound.owner_terms))
}

pub(crate) fn composed_consumer_bound_with_peak_terms(
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
    let resident_row_upper_bounds = producer_outputs
        .iter()
        .map(|output| (output.occurrence_id.clone(), output.row_count_upper_bound))
        .collect::<HashMap<_, _>>();
    let column_evidence = PlanColumnEvidenceAnalysis::for_plan_with_resident_rows(
        consumer_plan,
        &resident_outputs,
        &resident_row_upper_bounds,
    )?;
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

pub(crate) fn memoized_graph_consumer_peak_groups(
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
        } else if !node_terms.is_empty() {
            let mut group = live.clone();
            group.extend(node_terms.iter().map(|term| term.id.clone()));
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

pub(crate) fn composed_consumer_branch_bound(
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

pub(crate) fn composed_consumer_branch_bound_uncached(
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

pub(crate) fn composed_consumer_edge_normalize_peak_bytes(
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

pub(crate) fn composed_consumer_global_row_number_peak_bytes(
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

pub(crate) fn composed_consumer_graph_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let mut input = composed_consumer_branch_bound(
        plan,
        graph.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    for side_input in graph.inputs.iter().skip(1) {
        let mut input_inventory = owner_terms.to_vec();
        input_inventory.extend(input.owner_terms.iter().cloned());
        let side = composed_consumer_branch_bound(
            plan,
            *side_input,
            outputs,
            used_occurrences,
            &input_inventory,
            column_evidence,
            memo,
        )?;
        input.owner_terms.extend(side.owner_terms);
        input.peak_owner_term_groups.extend(side.peak_owner_term_groups);
        input.final_owned_term_ids.extend(side.final_owned_term_ids);
        input.dominated_owner_term_ids.extend(side.dominated_owner_term_ids);
    }
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

pub(crate) fn alias_wrapped_literal(expr: &Expr) -> Option<&LiteralValue> {
    match expr {
        Expr::Literal(literal) => Some(literal),
        Expr::Alias { expr, .. } => alias_wrapped_literal(expr),
        _ => None,
    }
}

pub(crate) fn composed_consumer_literal_projection_bound(
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

pub(crate) fn composed_consumer_expression_projection_bound(
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

pub(crate) fn consumer_peak_group_bytes(group: &[OwnerTermId], owner_terms: &[AllocationOwnerTerm]) -> Result<u64> {
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

pub(crate) fn consumer_peak_groups_bytes(
    groups: &[Vec<OwnerTermId>],
    owner_terms: &[AllocationOwnerTerm],
) -> Result<u64> {
    groups.iter().try_fold(0_u64, |peak, group| Ok(peak.max(consumer_peak_group_bytes(group, owner_terms)?)))
}

pub(crate) fn flatten_consumer_peak_groups(groups: &[Vec<OwnerTermId>]) -> Vec<OwnerTermId> {
    let mut seen = HashSet::new();
    groups.iter().flatten().filter(|term_id| seen.insert((*term_id).clone())).cloned().collect()
}

pub(crate) fn consumer_unary_peak_groups(
    input: &ComposedConsumerBranchBound,
    current_term_ids: impl IntoIterator<Item = OwnerTermId>,
) -> Vec<Vec<OwnerTermId>> {
    let mut groups = input.peak_owner_term_groups.clone();
    let mut operator_group = input.final_owned_term_ids.clone();
    operator_group.extend(current_term_ids);
    groups.push(operator_group);
    groups
}

pub(crate) fn representation_cast_projection(expr: &Expr) -> Option<(&crate::expr::ColumnRef, &DataType)> {
    match expr {
        Expr::Alias { expr, .. } => representation_cast_projection(expr),
        Expr::Cast { expr, target_type } => alias_wrapped_column(expr).map(|column| (column, target_type)),
        _ => None,
    }
}

pub(crate) fn composed_consumer_union_peak_bytes(
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

pub(crate) fn composed_consumer_aggregate_peak_bytes(
    plan: &QueryPlan,
    node_id: NodeId,
    aggregate: &AggregateNode,
    outputs: &HashMap<&str, &ComposedGraphProducerOutputEvidence>,
    used_occurrences: &mut HashSet<String>,
    owner_terms: &[AllocationOwnerTerm],
    column_evidence: &PlanColumnEvidenceAnalysis,
    memo: &mut HashMap<NodeId, ComposedConsumerBranchBound>,
) -> Result<ComposedConsumerBranchBound> {
    let input = composed_consumer_branch_bound(
        plan,
        aggregate.input,
        outputs,
        used_occurrences,
        owner_terms,
        column_evidence,
        memo,
    )?;
    let distinct_workspace_bytes = composed_global_nunique_workspace_bytes(aggregate, &input.schema, input.rows)?;
    if distinct_workspace_bytes.is_none() && !aggregate_owner_state_is_supported(aggregate) {
        return Err(composed_schedule_error("consumer_aggregate_state_unmodeled", None));
    }
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
    let mut peak_owner_term_groups =
        consumer_unary_peak_groups(&input, [partial_term_id.clone(), output_term_id.clone()]);
    let mut terms = input.owner_terms;
    terms.extend([partial_term, output_term]);
    if let Some(distinct_workspace_bytes) = distinct_workspace_bytes {
        let workspace = AllocationOwnerTerm::new(
            OwnerId::Aggregate(node_id),
            OwnerTermRole::AggregateDistinctWorkspace,
            OccurrenceId::PlanLocal(node_id),
            canonical_execution_scope_fingerprint(plan),
            AllocationClass::ActivePhase,
            distinct_workspace_bytes,
            None,
        )?;
        let mut workspace_peak = input.final_owned_term_ids.clone();
        workspace_peak.push(workspace.id.clone());
        peak_owner_term_groups.insert(peak_owner_term_groups.len() - 1, workspace_peak);
        terms.push(workspace);
    }
    let peak_owner_term_ids = flatten_consumer_peak_groups(&peak_owner_term_groups);
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

fn composed_global_nunique_workspace_bytes(
    aggregate: &AggregateNode,
    input_schema: &SchemaRef,
    input_rows: u64,
) -> Result<Option<u64>> {
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
    u64::try_from(preflight.active_workspace_peak_bytes()).map(Some).map_err(|_| {
        composed_schedule_error("consumer_aggregate_nunique_owner_preflight_unavailable", None)
            .with_fact("aggregate_owner_reason", "nunique_workspace_exceeds_u64")
            .with_fact("input_rows", input_rows)
    })
}

pub(crate) fn is_attested_sort_scope(plan: &QueryPlan) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(gpu) => is_attested_graph_result_bounded_consumer_plan(plan, &gpu.input_name),
            _ => false,
        },
        _ => false,
    })
}

pub(crate) fn composed_consumer_sort_peak_bytes(
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

pub(crate) fn composed_consumer_limit_peak_bytes(
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

pub(crate) fn composed_consumer_join_peak_bytes(
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
        .map(|key| hash_join_key_metadata(&build.schema, &build_column_evidence, build.rows, key.name()))
        .collect::<Result<Vec<_>>>()
        .map_err(|error| {
            error.with_fact("node_id", node_id).with_fact("left_rows", left.rows).with_fact("right_rows", right.rows)
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
                    .with_fact("left_rows", left.rows)
                    .with_fact("right_rows", right.rows)
            },
        )?;
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
        bounded_consumer_frame_bytes_with_evidence(output_rows, node.output_schema(), &column_evidence.node(node_id))
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

pub(crate) fn composed_join_preflight_error(
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

pub(crate) fn consumer_owner_term(
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

pub(crate) fn hash_join_key_metadata(
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

pub(crate) fn arrow_fixed_width_cudf_type_id(data_type: &DataType) -> Option<cudf::TypeId> {
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

pub(crate) fn bounded_consumer_boolean_mask_bytes(rows: u64) -> Result<u64> {
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "predicate",
        DataType::Boolean,
        false,
    )]));
    bounded_consumer_frame_bytes(rows, &schema)
}

pub(crate) fn bounded_consumer_frame_bytes(rows: u64, schema: &SchemaRef) -> Result<u64> {
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

pub(crate) fn bounded_consumer_frame_bytes_with_evidence(
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

pub(crate) fn consumer_output_column_with_optional_evidence(
    field: &arrow::datatypes::Field,
    evidence: Option<&PlanColumnEvidence>,
) -> Result<crate::exec::JoinOutputColumnUpperBoundInput> {
    match evidence {
        Some(evidence) => consumer_output_column_with_evidence(field, evidence),
        None => fixed_width_consumer_output_column(field.data_type())
            .ok_or_else(|| owner_term_error("aggregate_output_type_unmodeled", None, None)),
    }
}

pub(crate) fn consumer_output_column_with_evidence(
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
            input_max_string_length_bytes: bound.max_single_value_length_bytes,
            input_total_string_bytes: bound.total_payload_upper_bound_bytes,
        });
    }
    Err(composed_schedule_error("consumer_variable_width_frame_unmodeled", None)
        .with_fact("column", field.name())
        .with_fact("data_type", field.data_type()))
}

pub(crate) fn cugraph_side_input_frame_upper_bound_bytes(plan: &QueryPlan) -> Result<u64> {
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

pub(crate) fn cugraph_side_input_columns(
    plan: &QueryPlan,
) -> Result<Vec<crate::exec::JoinOutputColumnUpperBoundInput>> {
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
            Ok(crate::exec::JoinOutputColumnUpperBoundInput {
                data_type: field.data_type().clone(),
                input_max_string_length_bytes: Some(max_len),
                input_total_string_bytes: Some(total_bytes),
            })
        })
        .collect()
}

pub(crate) fn fixed_width_consumer_output_column(
    data_type: &DataType,
) -> Option<crate::exec::JoinOutputColumnUpperBoundInput> {
    arrow_fixed_width_cudf_type_id(data_type).map(|_| crate::exec::JoinOutputColumnUpperBoundInput {
        data_type: data_type.clone(),
        input_max_string_length_bytes: None,
        input_total_string_bytes: None,
    })
}

pub(crate) fn checked_consumer_add(left: u64, right: u64, reason: &'static str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| composed_schedule_error(reason, None))
}

pub(crate) fn generic_graph_execution_blocker() -> ReservationRequirement {
    ReservationRequirement::blocked_by_lower_level(
        ReservationKind::OperatorOutput,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
        crate::capability::ContractQuality::ExactPreflight,
    )
}

pub(crate) fn composed_schedule_error(schedule_reason: &'static str, occurrence_id: Option<&str>) -> QueryError {
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

pub(crate) fn sort_owner_terms(
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

pub(crate) fn limit_owner_terms(
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

pub(crate) fn validate_sort_metadata_descriptor(descriptor: &SortMetadataDescriptor) -> Result<()> {
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

pub(crate) fn sort_metadata_for_attested_graph_result(
    plan: &QueryPlan,
    graph_result_input_name: &str,
    producer_row_count_upper_bound: Option<u64>,
    producer_column_evidence: Option<&PlanColumnEvidence>,
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
        bounded_consumer_frame_components(row_count, sort_output_schema, producer_column_evidence)?;
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

pub(crate) fn bounded_consumer_frame_components(
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
                .and_then(|value| value.checked_mul(crate::exec::STRING_OFFSET_BYTES_CONSERVATIVE))
                .ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?;
            payload.checked_add(offsets).ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?
        } else {
            let width = fixed_width_byte_size(&column.data_type)
                .ok_or_else(|| sort_owner_error("sort_output_dtype_unmodeled", "sort_output"))?;
            rows.checked_mul(width).ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?
        };
        frame = frame
            .checked_add(data)
            .and_then(|value| value.checked_add(crate::exec::PER_COLUMN_ALIGNMENT_PADDING_BYTES))
            .ok_or_else(|| sort_owner_error("sort_output_bytes_overflow", "sort_output"))?;
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

pub(crate) fn fixed_width_byte_size(data_type: &DataType) -> Option<u64> {
    arrow_fixed_width_cudf_type_id(data_type)?;
    crate::datatypes::arrow_fixed_width_byte_size(data_type)
}

pub(crate) fn cugraph_graph_execution_capability() -> ReservationRequirement {
    ReservationRequirement::modeled(
        ReservationKind::GraphAlgorithm,
        "cugraph_graph_execution",
        "cugraph_graph_execution_contract",
        crate::capability::ContractQuality::CapabilityProof,
    )
}

pub(crate) fn is_generic_graph_execution_blocker(requirement: &ReservationRequirement) -> bool {
    requirement == &generic_graph_execution_blocker()
}

pub(crate) fn is_attested_graph_result_passthrough_plan(plan: &QueryPlan, graph_result_input_name: &str) -> bool {
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

pub(crate) fn is_attested_graph_result_bounded_consumer_plan(plan: &QueryPlan, graph_result_input_name: &str) -> bool {
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
