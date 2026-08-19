/// Service-owned policy inputs that affect the bounded attempt's physical
/// liveness. `device_budget_bytes` is deliberately absent: a grant is the
/// result of admission, not an input to its proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttemptReservationPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_chunk_planning_options: SourceChunkPlanningOptions,
}

impl AttemptReservationPolicy {
    pub(crate) const fn new(
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
pub(crate) struct AttemptReservationShape {
    execution_scope_fingerprint: String,
    output_column_evidence: crate::admission::string_payload::PlanColumnEvidence,
    output_column_evidence_input_scopes: Vec<(String, String)>,
    has_local_parquet_source: bool,
    source_stage_bytes: Option<u64>,
    complete_zero_column_global_aggregate: Option<ZeroColumnGlobalAggregateShape>,
    complete_graph_memory_phase: Option<GraphMemoryAdmissionEvidence>,
    complete_cugraph_side_input_phase_bytes: Option<u64>,
    accounted_by_graph_phase: bool,
    is_cugraph_graph_input_scope: bool,
    is_cugraph_side_input_scope: bool,
    is_attested_graph_result_passthrough_scope: bool,
    is_deferred_graph_result_passthrough_scope: bool,
    is_attested_graph_result_bounded_consumer_scope: bool,
    sort_phase_peak_bytes: Option<u64>,
    limit_phase_peak_bytes: Option<u64>,
    sort_metadata: Option<crate::admission::requirement::SortMetadataDescriptor>,
    sort_rejection_reason: Option<String>,
    owner_term_rejection_reason: Option<String>,
    producer_occurrences: Vec<AttemptReservationProducerOccurrence>,
    ordered_schedule: Option<AttemptReservationSchedule>,
}

impl AttemptReservationShape {
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            output_column_evidence: crate::admission::string_payload::PlanColumnEvidence::default(),
            output_column_evidence_input_scopes: Vec::new(),
            has_local_parquet_source: plan_has_reachable_local_parquet_source(plan),
            source_stage_bytes: estimated_reachable_local_parquet_source_bytes(plan),
            complete_zero_column_global_aggregate: zero_column_global_aggregate_shape(plan),
            complete_graph_memory_phase: None,
            complete_cugraph_side_input_phase_bytes: None,
            accounted_by_graph_phase: false,
            is_cugraph_graph_input_scope: false,
            is_cugraph_side_input_scope: false,
            is_attested_graph_result_passthrough_scope: false,
            is_deferred_graph_result_passthrough_scope: false,
            is_attested_graph_result_bounded_consumer_scope: false,
            sort_phase_peak_bytes: None,
            limit_phase_peak_bytes: None,
            sort_metadata: None,
            sort_rejection_reason: None,
            owner_term_rejection_reason: None,
            producer_occurrences: Vec::new(),
            ordered_schedule: None,
        }
    }

    pub(crate) fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    pub(super) fn output_column_evidence(&self) -> &crate::admission::string_payload::PlanColumnEvidence {
        &self.output_column_evidence
    }

    pub(super) fn set_output_column_evidence(
        &mut self,
        evidence: crate::admission::string_payload::PlanColumnEvidence,
    ) {
        self.output_column_evidence = evidence;
    }

    pub(super) fn set_output_column_evidence_input_scopes(&mut self, scopes: Vec<(String, String)>) {
        self.output_column_evidence_input_scopes = scopes;
    }

    pub(super) fn output_column_evidence_input_scopes(&self) -> &[(String, String)] {
        &self.output_column_evidence_input_scopes
    }

    pub(crate) fn attach_ordered_schedule(&mut self, schedule: AttemptReservationSchedule) -> Result<()> {
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

    pub(crate) fn ordered_schedule(&self) -> Option<&AttemptReservationSchedule> {
        self.ordered_schedule.as_ref()
    }

    pub(crate) fn take_ordered_schedule_for_nesting(
        &mut self,
    ) -> Option<(AttemptReservationSchedule, Vec<AttemptReservationProducerOccurrence>)> {
        let schedule = self.ordered_schedule.take()?;
        Some((schedule, std::mem::take(&mut self.producer_occurrences)))
    }

    pub(crate) fn owner_term_rejection_reason(&self) -> Option<&str> {
        self.owner_term_rejection_reason.as_deref()
    }

    pub(crate) fn set_producer_occurrences(
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

    pub(crate) fn producer_occurrences(&self) -> &[AttemptReservationProducerOccurrence] {
        &self.producer_occurrences
    }

    pub(crate) fn validate_manifest_core(&self) -> Result<()> {
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

    pub(crate) fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    pub(crate) fn matches_execution_scope_fingerprint(&self, execution_scope_fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == execution_scope_fingerprint
    }

    pub(crate) fn set_graph_memory_phase(
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

    pub(crate) fn account_for_by_graph_phase(&mut self) {
        self.accounted_by_graph_phase = true;
    }

    pub(crate) fn is_accounted_by_graph_phase(&self) -> bool {
        self.accounted_by_graph_phase
    }

    pub(crate) fn mark_cugraph_graph_input_scope(&mut self) {
        self.is_cugraph_graph_input_scope = true;
    }

    pub(crate) fn set_cugraph_side_input_phase_bytes(&mut self, frame_bytes: u64) -> Result<()> {
        if self.complete_cugraph_side_input_phase_bytes.replace(frame_bytes).is_some() {
            return Err(graph_schedule_error("duplicate_side_input_frame_evidence", None, Some("side_input_frame")));
        }
        self.is_cugraph_side_input_scope = true;
        Ok(())
    }

    pub(crate) fn mark_attested_graph_result_passthrough_scope(&mut self) {
        self.is_attested_graph_result_passthrough_scope = true;
    }

    pub(crate) fn mark_deferred_graph_result_passthrough_scope(&mut self) {
        self.is_deferred_graph_result_passthrough_scope = true;
        self.account_for_by_graph_phase();
    }

    pub(crate) fn mark_attested_graph_result_bounded_consumer_scope(&mut self) {
        self.is_attested_graph_result_bounded_consumer_scope = true;
    }

    pub(crate) fn set_sort_phase_peak_bytes(&mut self, peak_bytes: u64) -> Result<()> {
        if self.sort_phase_peak_bytes.replace(peak_bytes).is_some() {
            return Err(graph_schedule_error("duplicate_sort_phase_evidence", None, Some("sort")));
        }
        Ok(())
    }

    pub(crate) fn set_limit_phase_peak_bytes(&mut self, peak_bytes: u64) -> Result<()> {
        if self.limit_phase_peak_bytes.replace(peak_bytes).is_some() {
            return Err(graph_schedule_error("duplicate_limit_phase_evidence", None, Some("limit")));
        }
        Ok(())
    }

    pub(crate) fn set_sort_metadata(
        &mut self,
        metadata: crate::admission::requirement::SortMetadataDescriptor,
    ) -> Result<()> {
        if self.sort_metadata.replace(metadata).is_some() {
            return Err(graph_schedule_error("duplicate_sort_metadata", None, Some("sort")));
        }
        Ok(())
    }

    pub(crate) fn set_sort_rejection_reason(&mut self, reason: impl Into<String>) {
        self.sort_rejection_reason = Some(reason.into());
    }

    pub(crate) fn set_owner_term_rejection_reason(&mut self, reason: impl Into<String>) {
        self.owner_term_rejection_reason = Some(reason.into());
    }

    pub(crate) fn sort_metadata(&self) -> Option<&crate::admission::requirement::SortMetadataDescriptor> {
        self.sort_metadata.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn set_sort_stable_for_test(&mut self, stable: bool) {
        if let Some(metadata) = self.sort_metadata.as_mut() {
            metadata.stable = stable;
        }
    }

    pub(crate) fn has_modeled_source_stage(&self) -> bool {
        self.source_stage_bytes.is_some()
    }

    pub(crate) fn has_local_parquet_source(&self) -> bool {
        self.has_local_parquet_source
    }

    pub(crate) fn is_cugraph_graph_input_scope(&self) -> bool {
        self.is_cugraph_graph_input_scope
    }

    pub(crate) fn is_cugraph_side_input_scope(&self) -> bool {
        self.is_cugraph_side_input_scope
    }

    pub(crate) fn is_attested_graph_result_passthrough_scope(&self) -> bool {
        self.is_attested_graph_result_passthrough_scope
    }

    pub(crate) fn is_deferred_graph_result_passthrough_scope(&self) -> bool {
        self.is_deferred_graph_result_passthrough_scope
    }

    pub(crate) fn has_graph_memory_phase(&self) -> bool {
        self.complete_graph_memory_phase.is_some()
    }

    pub(crate) fn graph_memory_admission_evidence(&self) -> Option<&GraphMemoryAdmissionEvidence> {
        self.complete_graph_memory_phase.as_ref()
    }

    fn resolve(&self, policy: AttemptReservationPolicy) -> Result<ResolvedReservationShape> {
        if let Some(reason) = &self.owner_term_rejection_reason {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "attempt owner-term inventory is unavailable",
            )
            .with_fact("reason", reason.as_str()));
        }
        if self.accounted_by_graph_phase {
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

        if let Some(frame_bytes) = self.complete_cugraph_side_input_phase_bytes {
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: frame_bytes,
                complete_phase_peak_bytes: Some(frame_bytes),
            });
        }

        if self.is_attested_graph_result_bounded_consumer_scope {
            if let Some(reason) = &self.sort_rejection_reason {
                return Err(graph_schedule_error("sort_metadata_rejected", None, Some("sort"))
                    .with_fact("sort_rejection_reason", reason.as_str()));
            }
            let limit_expected = self.sort_metadata.as_ref().is_some_and(|metadata| metadata.limit.is_some());
            match (limit_expected, self.limit_phase_peak_bytes.is_some()) {
                (true, false) => {
                    return Err(graph_schedule_error("limit_preflight_unresolved", None, Some("limit")));
                },
                (false, true) => {
                    return Err(graph_schedule_error("unexpected_limit_phase_evidence", None, Some("limit")));
                },
                _ => {},
            }
            let sort_peak = self.sort_phase_peak_bytes;
            let limit_peak = self.limit_phase_peak_bytes;
            let peak = sort_peak
                .into_iter()
                .chain(limit_peak)
                .max()
                .ok_or_else(|| graph_schedule_error("sort_preflight_unresolved", None, Some("sort")))?;
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: peak,
                complete_phase_peak_bytes: Some(peak),
            });
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
    crate::exec::join_output_upper_bound_bytes(1, &fixed_width_columns(schema)?).ok()
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

/// The DataFusion adapter always executes host-facing native plans through a
/// GPU-resident physical sink, retaining the host contract only for the final
/// cursor. Admission must therefore scope the physical plan rather than the
/// wrapper-only host boundary. GPU-resident plans remain unchanged.
fn canonical_execution_scope(plan: &QueryPlan) -> QueryPlan {
    plan.clone_with_deferred_host_materialization_sink()
}

/// Collision-free structural identity for the exact plan shape that enters the
/// engine. Formatting follows the full derived IR value rather than a hash;
/// the scope is short-lived private admission evidence, never an external key.
pub(crate) fn canonical_execution_scope_fingerprint(plan: &QueryPlan) -> String {
    format!("{:#?}", canonical_execution_scope(plan))
}
