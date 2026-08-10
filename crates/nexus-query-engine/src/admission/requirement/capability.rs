use super::*;

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
    pub(crate) execution_scope_fingerprint: String,
}

impl AttestedCugraphGraphExecutionScope {
    pub(crate) fn for_plan(plan: &QueryPlan) -> Self {
        Self { execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan) }
    }

    pub(crate) fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    pub(crate) fn matches_reservation_shape(&self, shape: &AttemptReservationShape) -> bool {
        shape.matches_execution_scope_fingerprint(&self.execution_scope_fingerprint)
    }

    pub(crate) fn matches_graph_memory_evidence(&self, evidence: &GraphMemoryAdmissionEvidence) -> bool {
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
    pub(crate) requirement: AttemptCapabilityEvidence,
}

/// Adapter-observed identity and opaque admission proof for one graph producer
/// in a composed attempt.
///
/// The adapter may report an unavailable identity or proof, but it cannot
/// provide graph-memory bytes or construct a reservation shape. The engine
/// issues a composed schedule only when every observation is complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedGraphProducerCandidate {
    pub(crate) runtime_occurrence_id: Option<String>,
    pub(crate) graph_result_input_name: Option<String>,
    pub(crate) capability_proof: Option<QueryPlanCapabilityProof>,
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

    pub(crate) fn for_cugraph_materialized_graph_input_producer_plan(plan: &QueryPlan) -> Result<Self> {
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

    pub(crate) fn try_for_attested_graph_result_bounded_consumer_plan_with_row_bound(
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

    pub(crate) fn bounded_consumer_proof_or_sticky_rejection(plan: &QueryPlan, result: Result<Self>) -> Self {
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

    pub(crate) fn normalize_composed_candidate(candidate: ComposedGraphProducerCandidate) -> Result<(String, Self)> {
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

    pub(crate) fn normalize_composed_producer(
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

    pub(crate) fn reuse_complete_schedule_for_passthrough_consumer(
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

    pub(crate) fn combine_conservative_composed_graph_attempt(
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
    pub(super) fn output_column_evidence_for_scope(&self, execution_scope_fingerprint: &str) -> PlanColumnEvidence {
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

    pub(crate) fn remove_generic_gpu_resident_retained_output_terms(&mut self) {
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

    pub(crate) fn for_query_plan_with_policy(plan: &QueryPlan, policy: OwnerTermConstructionPolicy) -> Self {
        Self::for_query_plan_with_policy_and_gpu_output(plan, policy, true)
    }

    pub(crate) fn for_query_plan_without_gpu_output_owner(
        plan: &QueryPlan,
        policy: OwnerTermConstructionPolicy,
    ) -> Self {
        Self::for_query_plan_with_policy_and_gpu_output(plan, policy, false)
    }

    pub(crate) fn for_query_plan_with_policy_and_gpu_output(
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

    pub(crate) fn mark_cugraph_graph_input_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_cugraph_graph_input_scope();
        }
    }

    pub(crate) fn mark_attested_graph_result_passthrough_scope(&mut self) {
        if let [shape] = self.reservation_shapes.as_mut_slice() {
            shape.mark_attested_graph_result_passthrough_scope();
        }
    }

    pub(crate) fn mark_attested_graph_result_bounded_consumer_scope(&mut self) {
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
    pub(crate) fn extend_with_graph_memory_admission_evidence(
        &mut self,
        evidence: &GraphMemoryAdmissionEvidence,
    ) -> Result<()> {
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

    pub(crate) fn remove_shared_graph_input_source_stage_owners(
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

    pub(crate) fn graph_memory_evidence_scope_index(&self, evidence: &GraphMemoryAdmissionEvidence) -> Result<usize> {
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

    pub(crate) fn retained_output_rows_for_graph_result(&self) -> Result<u64> {
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

    pub(crate) fn retained_output_term_for_composition(
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
    pub(crate) fn isolated_phase_term_groups(&self) -> Vec<Vec<OwnerTermId>> {
        self.isolated_phase_term_groups_with_alternatives(None).0
    }

    pub(crate) fn concurrent_side_input_term_ids(&self) -> Vec<OwnerTermId> {
        self.owner_terms.iter().filter(|term| term.is_side_input_frame()).map(|term| term.id().clone()).collect()
    }

    pub(crate) fn graph_phase_term_groups(&self, execution_scope_fingerprint: &str) -> Vec<Vec<OwnerTermId>> {
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

    pub(crate) fn isolated_phase_term_groups_with_alternatives(
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
    pub(crate) fn account_graph_owned_runtime_scopes(&mut self, graph_shape_index: usize) {
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
    pub(crate) fn exact_attested_graph_reservation_shape_index(&self) -> Option<usize> {
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

    pub(crate) fn take_nested_schedule_peak_terms(
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

    pub(crate) fn rebind_owner_terms_to_composed_occurrence(
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

    pub(crate) fn resolve_for_device(&self, device_ordinal: usize) -> Result<Self> {
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

    pub(crate) fn resolve_for_device_with_preflight_and_limit<F, G>(
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

    pub(crate) fn set_sort_phase_peak(&mut self, descriptor: &SortMetadataDescriptor, peak_bytes: u64) -> Result<()> {
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

    pub(crate) fn set_limit_phase_peak(&mut self, descriptor: &LimitMetadataDescriptor, peak_bytes: u64) -> Result<()> {
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

pub(crate) fn source_stage_overflow_error(term: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.admission.requirement",
        "local-Parquet source stage bytes overflow u64",
    )
    .with_fact("reason", "attempt_requirement_source_stage_overflow")
    .with_fact("term", term)
    .with_fact("operation", "addition")
}
