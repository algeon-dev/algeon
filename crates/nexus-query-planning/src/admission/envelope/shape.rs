/// Service-owned policy inputs that affect the bounded attempt's physical
/// liveness. `device_budget_bytes` is deliberately absent: a grant is the
/// result of admission, not an input to its proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_chunk_planning_options: SourceChunkPlanningOptions,
}

impl AttemptReservationPolicy {
    #[doc(hidden)]
pub const fn new(
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> Self {
        Self {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        }
    }

    fn matches_execution(
        self,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> bool {
        let MemoryControlPolicy::Bounded {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            ..
        } = memory_policy
        else {
            return false;
        };
        self.max_source_chunk_bytes == max_source_chunk_bytes
            && self.max_row_groups_per_chunk == max_row_groups_per_chunk
            && self.max_retry_attempts == max_retry_attempts
            && self.min_retry_chunk_row_groups == min_retry_chunk_row_groups
            && self.source_scheduler_policy == source_scheduler_policy
            && self.source_chunk_planning_mode == source_chunk_planning_mode
            && self.source_chunk_planning_options == source_chunk_planning_options
    }

    fn source_read_phase_slots(self, source_chunk_count: u64) -> u64 {
        // `OrderedSourceReadQueue` retains at most the current decoded source
        // plus one optional prefetch. Its public scheduler bound can be larger
        // for future scheduling work, but it cannot create a third concurrent
        // source read in the current bounded executor.
        if source_chunk_count > 1 && self.source_scheduler_policy.enables_prefetch() { 2 } else { 1 }
    }
}

/// Immutable plan facts carried privately by a capability proof until the
/// query service can resolve them against its installed policy.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationShape {
    execution_scope_fingerprint: String,
    output_schema_fingerprint: String,
    output_column_evidence: crate::plan::column_evidence::PlanColumnEvidence,
    output_column_evidence_input_scopes: Vec<(String, String)>,
    has_local_parquet_source: bool,
    complete_literal_source: bool,
    complete_zero_byte_literal_source: bool,
    source_stage_bytes: Option<u64>,
    complete_zero_column_global_aggregate: Option<ZeroColumnGlobalAggregateShape>,
    complete_graph_memory_phase: Option<GraphMemoryAdmissionEvidence>,
    scope_role: AttemptReservationScopeRole,
    sort_metadata: Vec<crate::admission::requirement::SortMetadataDescriptor>,
    sort_rejection_reason: Option<String>,
    owner_term_rejection_reason: Option<String>,
    producer_occurrences: Vec<AttemptReservationProducerOccurrence>,
    nested_consumer_occurrence_ids: Option<Vec<String>>,
    deferred_owner_term_occurrence: Option<String>,
    ordered_schedule: Option<AttemptReservationSchedule>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum AttemptReservationScopeRole {
    #[default]
    Ordinary,
    ResidentProducerInput,
    ResidentSideInput,
    AccountedByProducer,
    AccountedResidentProducerInput,
}

impl AttemptReservationScopeRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::ResidentProducerInput => "resident_producer_input",
            Self::ResidentSideInput => "resident_side_input",
            Self::AccountedByProducer => "accounted_by_producer",
            Self::AccountedResidentProducerInput => "accounted_resident_producer_input",
        }
    }
}

impl AttemptReservationShape {
    #[doc(hidden)]
pub fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            output_schema_fingerprint: crate::admission::log_safe_fingerprint(&format!("{:?}", plan.output_schema())),
            output_column_evidence: crate::plan::column_evidence::PlanColumnEvidence::default(),
            output_column_evidence_input_scopes: Vec::new(),
            has_local_parquet_source: plan_has_reachable_local_parquet_source(plan),
            complete_literal_source: is_complete_literal_source_plan(plan),
            complete_zero_byte_literal_source: is_complete_zero_byte_literal_source_plan(plan),
            source_stage_bytes: estimated_reachable_local_parquet_source_bytes(plan),
            complete_zero_column_global_aggregate: zero_column_global_aggregate_shape(plan),
            complete_graph_memory_phase: None,
            scope_role: AttemptReservationScopeRole::Ordinary,
            sort_metadata: Vec::new(),
            sort_rejection_reason: None,
            owner_term_rejection_reason: None,
            producer_occurrences: Vec::new(),
            nested_consumer_occurrence_ids: None,
            deferred_owner_term_occurrence: None,
            ordered_schedule: None,
        }
    }

    #[doc(hidden)]
pub fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    #[doc(hidden)]
pub fn output_schema_fingerprint(&self) -> &str {
        &self.output_schema_fingerprint
    }

    #[doc(hidden)]
pub fn scope_role_label(&self) -> &'static str {
        self.scope_role.as_str()
    }

    pub(super) fn output_column_evidence(&self) -> &crate::plan::column_evidence::PlanColumnEvidence {
        &self.output_column_evidence
    }

    pub(super) fn set_output_column_evidence(
        &mut self,
        evidence: crate::plan::column_evidence::PlanColumnEvidence,
    ) {
        self.output_column_evidence = evidence;
    }

    pub(super) fn set_output_column_evidence_input_scopes(&mut self, scopes: Vec<(String, String)>) {
        self.output_column_evidence_input_scopes = scopes;
    }

    pub(super) fn output_column_evidence_input_scopes(&self) -> &[(String, String)] {
        &self.output_column_evidence_input_scopes
    }

    #[doc(hidden)]
pub fn bind_output_column_evidence_to_producers(
        &mut self,
        producer_scopes: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        for (input_name, input_scope) in &mut self.output_column_evidence_input_scopes {
            *input_scope = producer_scopes.get(input_name).cloned().ok_or_else(|| {
                graph_schedule_error("output_evidence_producer_binding_missing", Some(input_name), Some("manifest"))
            })?;
        }
        Ok(())
    }

    #[doc(hidden)]
pub fn attach_ordered_schedule(&mut self, schedule: AttemptReservationSchedule) -> Result<()> {
        if self.ordered_schedule.is_some() {
            return Err(graph_schedule_error("duplicate_schedule_evidence", None, Some("schedule")));
        }
        self.validate_manifest_core()?;
        if schedule.producer_phase_count_internal() != self.producer_occurrences.len() {
            return Err(graph_schedule_error("producer_phase_row_count_mismatch", None, Some("schedule"))
                .with_fact("producer_phase_row_count", schedule.producer_phase_count_internal())
                .with_fact("producer_occurrence_count", self.producer_occurrences.len()));
        }
        self.ordered_schedule = Some(schedule);
        Ok(())
    }

    #[doc(hidden)]
pub fn ordered_schedule(&self) -> Option<&AttemptReservationSchedule> {
        self.ordered_schedule.as_ref()
    }

    #[doc(hidden)]
pub fn resolve_deferred_consumer_schedule(
        &mut self,
        consumer_state: AttemptReservationConsumerState,
        consumer_peak_term_groups: Vec<Vec<OwnerTermId>>,
        consumer_dominated_term_ids: Vec<OwnerTermId>,
    ) -> Result<()> {
        self.ordered_schedule
            .as_mut()
            .ok_or_else(|| graph_schedule_error("missing_ordered_schedule", None, Some("consumer_peak")))?
            .resolve_deferred_consumer(
                consumer_state,
                consumer_peak_term_groups,
                consumer_dominated_term_ids,
            )
    }

    #[doc(hidden)]
pub fn take_ordered_schedule_for_nesting(
        &mut self,
    ) -> Option<(AttemptReservationSchedule, Vec<AttemptReservationProducerOccurrence>)> {
        let schedule = self.ordered_schedule.take()?;
        self.nested_consumer_occurrence_ids = Some(
            self.producer_occurrences
                .iter()
                .map(|occurrence| occurrence.occurrence_id().to_owned())
                .collect(),
        );
        Some((schedule, std::mem::take(&mut self.producer_occurrences)))
    }

    #[doc(hidden)]
pub fn nested_consumer_occurrence_ids(&self) -> Option<&[String]> {
        self.nested_consumer_occurrence_ids.as_deref()
    }

    #[doc(hidden)]
pub fn rebind_deferred_owner_terms(&mut self, occurrence_id: &str) {
        if self.sort_metadata.is_empty() {
            return;
        }
        self.deferred_owner_term_occurrence = Some(match self.deferred_owner_term_occurrence.take() {
            Some(inner) => format!("nested:{}:{occurrence_id}{inner}", occurrence_id.len()),
            None => occurrence_id.to_owned(),
        });
    }

    #[doc(hidden)]
pub fn deferred_owner_term_occurrence(&self) -> Option<&str> {
        self.deferred_owner_term_occurrence.as_deref()
    }

    #[cfg(test)]
    #[doc(hidden)]
pub fn declares_deferred_retained_output_term(&self, term_id: &OwnerTermId) -> bool {
        let Some(occurrence_id) = self.deferred_owner_term_occurrence.as_ref() else {
            return false;
        };
        if term_id.occurrence != crate::admission::requirement::OccurrenceId::Composed(occurrence_id.clone()) {
            return false;
        }
        self.sort_metadata.iter().filter(|metadata| metadata.retained_output).any(|metadata| {
            if let Some(limit) = &metadata.limit
                && term_id.owner == OwnerId::Limit(limit.node_id)
                && term_id.role == OwnerTermRole::LimitOutputFrame
                && term_id.execution_scope_fingerprint == limit.execution_scope_fingerprint
            {
                return true;
            }
            term_id.owner == OwnerId::Sort(metadata.node_id)
                && term_id.role == OwnerTermRole::SortOutputFrame
                && term_id.execution_scope_fingerprint == metadata.execution_scope_fingerprint
        })
    }

    #[doc(hidden)]
pub fn owner_term_rejection_reason(&self) -> Option<&str> {
        self.owner_term_rejection_reason.as_deref()
    }

    #[doc(hidden)]
pub fn set_producer_occurrences(
        &mut self,
        occurrences: impl IntoIterator<Item = AttemptReservationProducerOccurrence>,
    ) -> Result<()> {
        if !self.producer_occurrences.is_empty() {
            return Err(graph_schedule_error("duplicate_producer_occurrence_rows", None, Some("manifest")));
        }
        self.producer_occurrences = occurrences.into_iter().collect();
        self.validate_manifest_core()?;
        Ok(())
    }

    #[doc(hidden)]
    pub fn producer_occurrences(&self) -> &[AttemptReservationProducerOccurrence] {
        &self.producer_occurrences
    }

    pub(crate) fn producer_occurrence_scope_count(&self, execution_scope_fingerprint: &str) -> usize {
        self.producer_occurrences
            .iter()
            .filter(|occurrence| occurrence.execution_scope_fingerprint == execution_scope_fingerprint)
            .count()
    }

    #[doc(hidden)]
pub fn validate_manifest_core(&self) -> Result<()> {
        let mut occurrence_ids = HashSet::with_capacity(self.producer_occurrences.len());
        for occurrence in &self.producer_occurrences {
            if occurrence.occurrence_id.is_empty() {
                return Err(graph_schedule_error("missing_producer_occurrence_id", None, Some("manifest")));
            }
            if !occurrence_ids.insert(&occurrence.occurrence_id) {
                return Err(graph_schedule_error(
                    "duplicate_producer_occurrence_id",
                    Some(&occurrence.occurrence_id),
                    Some("manifest"),
                ));
            }
            if occurrence.execution_scope_fingerprint.is_empty() {
                return Err(graph_schedule_error(
                    "missing_producer_occurrence_scope",
                    Some(&occurrence.occurrence_id),
                    Some("manifest"),
                ));
            }
        }
        if self.execution_scope_fingerprint.is_empty() {
            return Err(graph_schedule_error("missing_consumer_scope", None, Some("manifest")));
        }
        Ok(())
    }

    fn manifest_scope_counts(&self) -> std::collections::HashMap<&str, usize> {
        let mut counts = std::collections::HashMap::with_capacity(self.producer_occurrences.len() + 1);
        *counts.entry(self.execution_scope_fingerprint.as_str()).or_default() += 1;
        for occurrence in &self.producer_occurrences {
            *counts.entry(occurrence.execution_scope_fingerprint.as_str()).or_default() += 1;
        }
        counts
    }

    #[doc(hidden)]
pub fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    #[doc(hidden)]
pub fn matches_execution_scope_fingerprint(&self, execution_scope_fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == execution_scope_fingerprint
    }

    #[doc(hidden)]
pub fn set_graph_memory_phase(
        &mut self,
        evidence: &GraphMemoryAdmissionEvidence,
        source_stage_bytes: Option<u64>,
    ) {
        self.complete_graph_memory_phase = Some(evidence.clone());
        // A composed graph proof may obtain its prepared edge frame from a
        // separately lowered native source. The aggregate proof owns that
        // exact stage byte count, so replace this graph scope's local view
        // instead of adding it a second time.
        if let Some(source_stage_bytes) = source_stage_bytes {
            self.source_stage_bytes = Some(source_stage_bytes);
        }
    }

    #[doc(hidden)]
pub fn account_for_by_producer_phase(&mut self) {
        self.scope_role = match self.scope_role {
            AttemptReservationScopeRole::ResidentProducerInput => {
                AttemptReservationScopeRole::AccountedResidentProducerInput
            },
            _ => AttemptReservationScopeRole::AccountedByProducer,
        };
    }

    #[doc(hidden)]
pub fn account_for_by_passthrough_producer(&mut self, occurrence_id: String) {
        self.account_for_by_producer_phase();
        self.nested_consumer_occurrence_ids = Some(vec![occurrence_id]);
    }

    #[doc(hidden)]
pub fn is_accounted_by_producer_phase(&self) -> bool {
        matches!(
            self.scope_role,
            AttemptReservationScopeRole::AccountedByProducer
                | AttemptReservationScopeRole::AccountedResidentProducerInput
        ) || self.is_passthrough_scope()
    }

    #[doc(hidden)]
pub fn mark_resident_producer_input_scope(&mut self) {
        self.scope_role = match self.scope_role {
            AttemptReservationScopeRole::AccountedByProducer => {
                AttemptReservationScopeRole::AccountedResidentProducerInput
            },
            _ => AttemptReservationScopeRole::ResidentProducerInput,
        };
    }

    #[doc(hidden)]
pub fn mark_resident_side_input_scope(&mut self) -> Result<()> {
        if self.scope_role == AttemptReservationScopeRole::ResidentSideInput {
            return Err(graph_schedule_error("duplicate_side_input_frame_evidence", None, Some("side_input_frame")));
        }
        self.scope_role = AttemptReservationScopeRole::ResidentSideInput;
        Ok(())
    }

    #[doc(hidden)]
pub fn set_sort_metadata(
        &mut self,
        metadata: crate::admission::requirement::SortMetadataDescriptor,
    ) -> Result<()> {
        if self.sort_metadata.iter().any(|existing| existing.node_id == metadata.node_id) {
            return Err(graph_schedule_error("duplicate_sort_metadata", None, Some("sort")));
        }
        self.sort_metadata.push(metadata);
        Ok(())
    }

    #[doc(hidden)]
pub fn set_sort_rejection_reason(&mut self, reason: impl Into<String>) {
        self.sort_rejection_reason = Some(reason.into());
    }

    #[doc(hidden)]
pub fn set_owner_term_rejection_reason(&mut self, reason: impl Into<String>) {
        self.owner_term_rejection_reason = Some(reason.into());
    }

    #[doc(hidden)]
pub fn clear_resolved_gpu_output_owner_rejection(&mut self) {
        if matches!(
            self.owner_term_rejection_reason.as_deref(),
            Some(
                "gpu_resident_output_bytes_unavailable"
                    | "gpu_resident_row_bound_unavailable"
                    | "attempt_owner_term_gpu_resident_output_bytes_unavailable"
                    | "attempt_owner_term_gpu_resident_row_bound_unavailable"
            )
        ) {
            self.owner_term_rejection_reason = None;
        }
    }

    #[doc(hidden)]
pub fn sort_metadata(&self) -> &[crate::admission::requirement::SortMetadataDescriptor] {
        &self.sort_metadata
    }

    #[cfg(test)]
    #[doc(hidden)]
pub fn set_sort_stable_for_test(&mut self, stable: bool) {
        for metadata in &mut self.sort_metadata {
            metadata.stable = stable;
        }
    }

    #[doc(hidden)]
pub fn has_modeled_source_stage(&self) -> bool {
        self.source_stage_bytes.is_some()
    }

    #[doc(hidden)]
pub fn has_local_parquet_source(&self) -> bool {
        self.has_local_parquet_source
    }

    #[doc(hidden)]
pub fn is_complete_literal_source(&self) -> bool {
        self.complete_literal_source
    }

    #[doc(hidden)]
pub fn is_resident_producer_input_scope(&self) -> bool {
        matches!(
            self.scope_role,
            AttemptReservationScopeRole::ResidentProducerInput
                | AttemptReservationScopeRole::AccountedResidentProducerInput
        )
    }

    #[doc(hidden)]
pub fn is_resident_side_input_scope(&self) -> bool {
        self.scope_role == AttemptReservationScopeRole::ResidentSideInput
    }

    #[doc(hidden)]
pub fn is_passthrough_scope(&self) -> bool {
        self.ordered_schedule
            .as_ref()
            .is_some_and(|schedule| schedule.consumer_state() == AttemptReservationConsumerState::Passthrough)
    }

    #[doc(hidden)]
pub fn has_graph_memory_phase(&self) -> bool {
        self.complete_graph_memory_phase.is_some()
    }

    #[doc(hidden)]
pub fn graph_memory_admission_evidence(&self) -> Option<&GraphMemoryAdmissionEvidence> {
        self.complete_graph_memory_phase.as_ref()
    }

    fn resolve(
        &self,
        policy: AttemptReservationPolicy,
        owner_terms: Option<&[AllocationOwnerTerm]>,
    ) -> Result<ResolvedReservationShape> {
        if let Some(reason) = &self.owner_term_rejection_reason {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "attempt owner-term inventory is unavailable",
            )
            .with_fact("reason", reason.as_str()));
        }
        if self.scope_role == AttemptReservationScopeRole::AccountedByProducer
            || self.scope_role == AttemptReservationScopeRole::AccountedResidentProducerInput
            || self.is_passthrough_scope()
        {
            return Ok(ResolvedReservationShape { required_phase_floor_bytes: 0, complete_phase_peak_bytes: Some(0) });
        }
        if self.complete_literal_source && self.owner_term_rejection_reason.is_none() {
            if !self.complete_zero_byte_literal_source {
                return Ok(ResolvedReservationShape { required_phase_floor_bytes: 0, complete_phase_peak_bytes: None });
            }
            return Ok(ResolvedReservationShape { required_phase_floor_bytes: 0, complete_phase_peak_bytes: Some(0) });
        }
        if let Some(graph_phase) = &self.complete_graph_memory_phase {
            let graph_peak = match self.source_stage_bytes {
                Some(source_stage_bytes) => {
                    graph_phase.bounded_live_graph_peak_with_source_stage_bytes(source_stage_bytes)?
                },
                None => graph_phase.bounded_live_graph_peak_bytes()?,
            };
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: graph_peak,
                complete_phase_peak_bytes: Some(graph_peak),
            });
        }

        if !self.sort_metadata.is_empty() || self.sort_rejection_reason.is_some() {
            if let Some(reason) = &self.sort_rejection_reason {
                return Err(graph_schedule_error("sort_metadata_rejected", None, Some("sort"))
                    .with_fact("sort_rejection_reason", reason.as_str()));
            }
            let owner_terms = owner_terms
                .ok_or_else(|| graph_schedule_error("sort_preflight_unresolved", None, Some("sort")))?;
            let sort_roles = [
                OwnerTermRole::SortKeyInput,
                OwnerTermRole::SortKeyOutput,
                OwnerTermRole::SortSequence,
                OwnerTermRole::SortComparatorState,
                OwnerTermRole::SortTemporaryWorkspace,
                OwnerTermRole::SortRetainedOrder,
                OwnerTermRole::SortOutputFrame,
                OwnerTermRole::SortOutputNullMask,
            ];
            for metadata in &self.sort_metadata {
                let final_output_node_id = metadata.limit.as_ref().map_or(metadata.node_id, |limit| limit.node_id);
                let output_transferred = metadata.retained_output
                    && owner_terms.iter().any(|term| {
                        term.id().owner == OwnerId::Producer(final_output_node_id)
                            && term.id().role == OwnerTermRole::RetainedOutput
                            && term.execution_scope_fingerprint() == metadata.execution_scope_fingerprint
                    });
                let sort_output_transferred = output_transferred && metadata.limit.is_none();
                if sort_roles
                    .iter()
                    .filter(|role| {
                        !(sort_output_transferred
                            && matches!(role, OwnerTermRole::SortOutputFrame | OwnerTermRole::SortOutputNullMask))
                    })
                    .any(|role| {
                    !owner_terms.iter().any(|term| {
                        term.id().owner == OwnerId::Sort(metadata.node_id)
                            && term.id().role == *role
                            && term.execution_scope_fingerprint() == metadata.execution_scope_fingerprint
                    })
                    })
                {
                    return Err(graph_schedule_error("sort_preflight_unresolved", None, Some("sort")));
                }
                let Some(limit) = &metadata.limit else {
                    continue;
                };
                let limit_roles = [
                    OwnerTermRole::LimitGatherMap,
                    OwnerTermRole::LimitNativeWorkspace,
                    OwnerTermRole::LimitOutputFrame,
                    OwnerTermRole::LimitOutputNullMask,
                ];
                if limit_roles
                    .iter()
                    .filter(|role| {
                        !(output_transferred
                            && matches!(role, OwnerTermRole::LimitOutputFrame | OwnerTermRole::LimitOutputNullMask))
                    })
                    .any(|role| {
                        !owner_terms.iter().any(|term| {
                            term.id().owner == OwnerId::Limit(limit.node_id)
                                && term.id().role == *role
                                && term.execution_scope_fingerprint() == limit.execution_scope_fingerprint
                        })
                    })
                {
                    return Err(graph_schedule_error("limit_preflight_unresolved", None, Some("limit")));
                }
            }
            return Ok(ResolvedReservationShape { required_phase_floor_bytes: 0, complete_phase_peak_bytes: Some(0) });
        }

        if self.is_resident_producer_input_scope()
            && let Some(owner_terms) = owner_terms
        {
            let source_terms = owner_terms
                .iter()
                .filter(|term| term.execution_scope_fingerprint() == self.execution_scope_fingerprint)
                .collect::<Vec<_>>();
            if !source_terms.is_empty() && source_terms.iter().all(|term| term.is_source_stage()) {
                let source_peak = source_terms.iter().try_fold(0_u64, |total, term| {
                    total.checked_add(term.bytes()).ok_or_else(|| envelope_overflow_error("resident_producer_source"))
                })?;
                return Ok(ResolvedReservationShape {
                    required_phase_floor_bytes: source_peak,
                    complete_phase_peak_bytes: Some(source_peak),
                });
            }
        }

        let minimum_source_read_bytes =
            self.has_local_parquet_source.then(minimum_parquet_source_read_budget_bytes).unwrap_or_default();
        let Some(shape) = &self.complete_zero_column_global_aggregate else {
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: minimum_source_read_bytes,
                complete_phase_peak_bytes: None,
            });
        };

        let source_chunk_count = shape.source_chunk_count(policy.max_row_groups_per_chunk)?;
        let source_phase_bytes = minimum_source_read_bytes
            .checked_mul(policy.source_read_phase_slots(source_chunk_count))
            .ok_or_else(|| envelope_overflow_error("source_read_phase"))?;
        let retained_partial_bytes = shape
            .partial_state_upper_bound_bytes
            .checked_mul(source_chunk_count)
            .ok_or_else(|| envelope_overflow_error("retained_partial_state"))?;
        // The scalar aggregate result and its immediate projection can overlap
        // with the retained partial list during finalization. They are bounded
        // independently by the same closed-form device-output contract.
        let final_scalar_phase_bytes =
            shape.maximum_scalar_frame_bytes.checked_mul(2).ok_or_else(|| envelope_overflow_error("final_scalar"))?;
        let peak = source_phase_bytes
            .checked_add(retained_partial_bytes)
            .and_then(|value| value.checked_add(final_scalar_phase_bytes))
            .ok_or_else(|| envelope_overflow_error("zero_column_global_aggregate"))?;
        Ok(ResolvedReservationShape { required_phase_floor_bytes: peak, complete_phase_peak_bytes: Some(peak) })
    }
}

fn is_complete_literal_source_plan(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids();
    let reachable = plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).collect::<Vec<_>>();
    let [node] = reachable.as_slice() else {
        return false;
    };
    matches!(
        &node.kind,
        PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LiteralBatch(_))
    ) && plan.root_id() == node.id
}

fn is_complete_zero_byte_literal_source_plan(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids();
    let reachable = plan.nodes().iter().filter(|node| !unreachable.contains(&node.id)).collect::<Vec<_>>();
    let [node] = reachable.as_slice() else {
        return false;
    };
    matches!(
        &node.kind,
        PlanNode::Source(source)
            if matches!(source.as_ref(), SourceNode::LiteralBatch(source) if source.row_count() == 0 || source.schema.fields().is_empty())
    ) && plan.root_id() == node.id
}

#[derive(Debug, Clone, Copy)]
struct ResolvedReservationShape {
    required_phase_floor_bytes: u64,
    complete_phase_peak_bytes: Option<u64>,
}

/// Exact plan facts for the specialized zero-column aggregate executor.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ZeroColumnGlobalAggregateShape {
    selected_row_groups_per_file: Vec<usize>,
    partial_state_upper_bound_bytes: u64,
    maximum_scalar_frame_bytes: u64,
}

impl ZeroColumnGlobalAggregateShape {
    fn source_chunk_count(&self, max_row_groups_per_chunk: usize) -> Result<u64> {
        let max_row_groups_per_chunk = u64::try_from(max_row_groups_per_chunk.max(1))
            .map_err(|_| envelope_overflow_error("max_row_groups_per_chunk"))?;
        let mut chunks = 0_u64;
        for row_groups in &self.selected_row_groups_per_file {
            let row_groups = u64::try_from(*row_groups).map_err(|_| envelope_overflow_error("selected_row_groups"))?;
            chunks = chunks
                .checked_add(row_groups.div_ceil(max_row_groups_per_chunk))
                .ok_or_else(|| envelope_overflow_error("source_chunk_count"))?;
        }
        Ok(chunks.max(1))
    }
}

fn zero_column_global_aggregate_shape(plan: &QueryPlan) -> Option<ZeroColumnGlobalAggregateShape> {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut row_groups_per_file = Vec::new();
    let mut partial_state_upper_bound_bytes = 0_u64;
    let mut maximum_scalar_frame_bytes = 0_u64;
    let mut has_aggregate = false;

    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        maximum_scalar_frame_bytes =
            maximum_scalar_frame_bytes.max(scalar_frame_upper_bound_bytes(node.output_schema())?);
        match &node.kind {
            PlanNode::Source(source) => {
                let SourceNode::LocalParquet(source) = source.as_ref() else {
                    return None;
                };
                if !source.is_zero_column_count_source() || source.files.is_empty() {
                    return None;
                }
                for file in &source.files {
                    row_groups_per_file.push(file.row_groups.as_ref()?.len());
                }
            },
            PlanNode::Aggregate(aggregate) => {
                if !aggregate.spec.group_exprs.is_empty() {
                    return None;
                }
                partial_state_upper_bound_bytes = partial_state_upper_bound_bytes.max(
                    global_partial_state_upper_bound_bytes(node.output_schema(), aggregate.spec.aggr_exprs.len())?,
                );
                has_aggregate = true;
            },
            // This is the exact scalar path selected for COUNT(*) and related
            // zero-column global aggregates. Any widening, join, sort, graph,
            // or source-adjacent operator leaves the complete peak unknown so
            // admission deliberately uses the candidate grant ceiling.
            PlanNode::Projection(_) | PlanNode::Limit(_) => {},
            _ => return None,
        }
    }

    (has_aggregate && !row_groups_per_file.is_empty()).then_some(ZeroColumnGlobalAggregateShape {
        selected_row_groups_per_file: row_groups_per_file,
        partial_state_upper_bound_bytes,
        maximum_scalar_frame_bytes,
    })
}

fn global_partial_state_upper_bound_bytes(schema: &arrow::datatypes::SchemaRef, aggregate_count: usize) -> Option<u64> {
    let mut columns = fixed_width_columns(schema)?;
    // SUM/MIN/MAX partials retain an extra validity count and several aggregate
    // families retain more than their final projection. One fixed-width state
    // per aggregate is a conservative owner-side floor for the scalar
    // zero-column path; unsupported variable-width state declines the model.
    columns.extend((0..aggregate_count).map(|_| fixed_width_column(DataType::UInt64)));
    partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &columns).ok()
}

fn scalar_frame_upper_bound_bytes(schema: &arrow::datatypes::SchemaRef) -> Option<u64> {
    crate::materialization_sizing::join_output_upper_bound_bytes(1, &fixed_width_columns(schema)?).ok()
}

fn fixed_width_columns(schema: &arrow::datatypes::SchemaRef) -> Option<Vec<JoinOutputColumnUpperBoundInput>> {
    schema.fields().iter().map(|field| fixed_width_column_checked(field.data_type())).collect()
}

fn fixed_width_column_checked(data_type: &DataType) -> Option<JoinOutputColumnUpperBoundInput> {
    crate::admission::requirement::arrow_fixed_width_cudf_type_id(data_type)
        .map(|_| fixed_width_column(data_type.clone()))
}

fn fixed_width_column(data_type: DataType) -> JoinOutputColumnUpperBoundInput {
    JoinOutputColumnUpperBoundInput { data_type, input_max_string_length_bytes: None, input_total_string_bytes: None }
}

fn plan_has_reachable_local_parquet_source(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    plan.nodes().iter().any(|node| {
        !unreachable.contains(&node.id)
            && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(source) if !source.files.is_empty()))
    })
}

fn minimum_parquet_source_read_budget_bytes() -> u64 {
    u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap_or(u64::MAX)
}
