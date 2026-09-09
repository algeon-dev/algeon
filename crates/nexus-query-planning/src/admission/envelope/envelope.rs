/// Resolved service-admission envelope retained by the admitted context.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptReservationEnvelope {
    policy: AttemptReservationPolicy,
    scopes: Vec<AttemptReservationShape>,
    required_phase_floor_bytes: u64,
    complete_phase_peak_bytes: Option<u64>,
    ordered_schedule: Option<AttemptReservationSchedule>,
    bound_derivation: Option<AttemptBoundDerivation>,
    owner_terms: Vec<AllocationOwnerTerm>,
}

fn validate_nested_runtime_occurrences(scheduled: &[String], runtime: &[String]) -> Result<()> {
    if scheduled.len() != runtime.len() {
        return Err(graph_schedule_error("runtime_occurrence_count_mismatch", None, Some("runtime"))
            .with_fact("scheduled_occurrence_count", scheduled.len())
            .with_fact("runtime_occurrence_count", runtime.len()));
    }
    for (scheduled, runtime) in scheduled.iter().zip(runtime) {
        if scheduled != runtime {
            return Err(graph_schedule_error("runtime_occurrence_order_mismatch", Some(runtime), Some("runtime"))
                .with_fact("scheduled_occurrence_id", scheduled));
        }
    }
    Ok(())
}

impl AttemptReservationEnvelope {
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
pub fn resolve(
        reservation_shapes: &[AttemptReservationShape],
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        Self::resolve_internal(reservation_shapes, None, policy)
    }

    #[doc(hidden)]
pub fn resolve_with_owner_terms(
        reservation_shapes: &[AttemptReservationShape],
        owner_terms: &[AllocationOwnerTerm],
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        Self::resolve_internal(reservation_shapes, Some(owner_terms), policy)
    }

    fn resolve_internal(
        reservation_shapes: &[AttemptReservationShape],
        owner_terms: Option<&[AllocationOwnerTerm]>,
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        // Reservation shapes are allocation occurrences, not structural set
        // members.  Equal graph plans may retain two distinct result handles
        // and must remain two entries in the resolved envelope.
        let scopes = reservation_shapes.to_vec();
        let mut ordered_schedule =
            match scopes.iter().filter_map(AttemptReservationShape::ordered_schedule).collect::<Vec<_>>().as_slice() {
                [] => None,
                [schedule] => Some((*schedule).clone()),
                _ => return Err(graph_schedule_error("duplicate_schedule_evidence", None, Some("schedule"))),
            };

        let mut required_phase_floor_bytes = 0_u64;
        let mut complete_phase_peak_bytes = Some(0_u64);
        let has_graph_memory_phase = scopes.iter().any(AttemptReservationShape::has_graph_memory_phase);
        let mut scheduled_scope_counts = ordered_schedule
            .as_ref()
            .map(|schedule| {
                let manifest = admitted_manifest_shape(&scopes, Some(schedule))?;
                let mut counts = manifest.manifest_scope_counts();
                for scope in schedule.concurrent_side_input_scope_fingerprints() {
                    *counts.entry(scope).or_default() += 1;
                }
                for scope in schedule.referenced_scope_fingerprints() {
                    counts.entry(scope).or_insert(1);
                }
                for (scope, referenced) in schedule.referenced_composed_occurrence_scope_counts() {
                    let count = counts.entry(scope).or_default();
                    *count = (*count).max(referenced);
                }
                Ok(counts)
            })
            .transpose()?;
        for scope in &scopes {
            let resolved = scope.resolve(policy, owner_terms)?;
            let is_scheduled_scope = scheduled_scope_counts.as_mut().is_some_and(|counts| {
                let Some(remaining) = counts.get_mut(scope.execution_scope_fingerprint()) else {
                    return false;
                };
                if *remaining == 0 {
                    return false;
                }
                *remaining -= 1;
                true
            });
            if is_scheduled_scope {
                // The ordered fold is the only complete peak for its covered
                // producer/consumer occurrences.  Adding their isolated
                // peaks here would reintroduce the obsolete blind sum.
                continue;
            }
            required_phase_floor_bytes =
                required_phase_floor_bytes.checked_add(resolved.required_phase_floor_bytes).ok_or_else(|| {
                    ordered_schedule.as_ref().map_or_else(
                        || envelope_overflow_error("combined_required_phase"),
                        |_| {
                            graph_schedule_error("combined_required_phase_overflow", None, Some("required_phase_floor"))
                        },
                    )
                })?;
            complete_phase_peak_bytes = match (complete_phase_peak_bytes, resolved.complete_phase_peak_bytes) {
                (Some(current), Some(peak)) => match current.checked_add(peak) {
                    Some(value) => Some(value),
                    None if ordered_schedule.is_some() => {
                        return Err(graph_schedule_error(
                            "auxiliary_complete_phase_overflow",
                            None,
                            Some("auxiliary_complete_phase"),
                        ));
                    },
                    None => None,
                },
                _ => None,
            };
        }

        let mut selected_derivation = None;
        if let Some(schedule) = &ordered_schedule {
            let ordered_result = match owner_terms {
                Some(owner_terms) => schedule.checked_fold_with_inventory(&scopes, owner_terms),
                None => schedule.checked_fold_with_scopes(&scopes),
            };
            match ordered_result {
                Ok(complete_peak) => {
                    let Some(auxiliary_peak) = complete_phase_peak_bytes else {
                        return Err(graph_schedule_error("auxiliary_scope_unmodeled", None, Some("auxiliary_scope")));
                    };
                    required_phase_floor_bytes = required_phase_floor_bytes.max(complete_peak);
                    complete_phase_peak_bytes = Some(auxiliary_peak.max(complete_peak));
                    selected_derivation = Some(AttemptBoundDerivation::OrderedTimeline);
                },
                Err(ordered_error)
                    if owner_terms.is_some()
                        && ordered_error.fact_value("ordered_inventory_incomplete") == Some("true") =>
                {
                    let conservative = checked_owner_sum(owner_terms.expect("owner terms present"), &scopes);
                    match conservative {
                        Ok(owner_sum)
                            if owner_inventory_covers_scopes(
                                owner_terms.expect("owner terms present"),
                                &scopes,
                                owner_sum,
                                required_phase_floor_bytes,
                                false,
                            ) =>
                        {
                            ordered_schedule = None;
                            required_phase_floor_bytes = owner_sum.max(required_phase_floor_bytes);
                            complete_phase_peak_bytes = Some(owner_sum);
                            selected_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
                        },
                        Ok(_) => {
                            return Err(derivation_unavailable_error(
                                ordered_error,
                                owner_derivation_incomplete_error(),
                            ));
                        },
                        Err(conservative_error) => {
                            return Err(derivation_unavailable_error(ordered_error, conservative_error));
                        },
                    }
                },
                Err(ordered_error) if owner_terms.is_some() => {
                    let conservative_error = checked_owner_sum(owner_terms.expect("owner terms present"), &scopes)
                        .err()
                        .unwrap_or_else(owner_derivation_incomplete_error);
                    return Err(derivation_unavailable_error(ordered_error, conservative_error));
                },
                Err(error) => return Err(error),
            }
        } else if let Some(owner_terms) = owner_terms {
            match checked_owner_sum(owner_terms, &scopes) {
                Ok(owner_sum)
                    if owner_inventory_covers_scopes(
                        owner_terms,
                        &scopes,
                        owner_sum,
                        required_phase_floor_bytes,
                        false,
                    ) =>
                {
                    required_phase_floor_bytes = required_phase_floor_bytes.max(owner_sum);
                    complete_phase_peak_bytes = Some(owner_sum);
                    selected_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
                },
                Ok(_) => {},
                Err(error) => return Err(error),
            }
        }
        if selected_derivation.is_none() && complete_phase_peak_bytes.is_some() {
            selected_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
        }

        if ordered_schedule.is_none() && has_graph_memory_phase && complete_phase_peak_bytes.is_none() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "a graph attempt contains a runtime scope without a complete bounded phase target",
            )
            .with_fact("reason", GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
        }
        Ok(Self {
            policy,
            scopes,
            required_phase_floor_bytes,
            complete_phase_peak_bytes,
            ordered_schedule,
            bound_derivation: selected_derivation,
            owner_terms: owner_terms.map_or_else(Vec::new, ToOwned::to_owned),
        })
    }

    #[doc(hidden)]
pub fn required_phase_floor_bytes(&self) -> u64 {
        self.required_phase_floor_bytes
    }

    #[doc(hidden)]
pub fn complete_phase_peak_bytes(&self) -> Option<u64> {
        self.complete_phase_peak_bytes
    }

    #[doc(hidden)]
pub fn with_retained_graph_input_owner(
        mut self,
        evidence: &GraphMemoryAdmissionEvidence,
        exact_owner_bytes: u64,
    ) -> Result<Self> {
        let excess_bytes = exact_owner_bytes.saturating_sub(evidence.retained_normalization_source_model_bytes());
        if excess_bytes == 0 {
            return Ok(self);
        }
        if self.bound_derivation == Some(AttemptBoundDerivation::OrderedTimeline) {
            return Err(graph_schedule_error(
                "retained_graph_input_owner_requires_unscheduled_phase",
                None,
                Some("resident_owner"),
            ));
        }
        let complete_peak = self
            .complete_phase_peak_bytes
            .ok_or_else(|| graph_schedule_error("resident_owner_phase_peak_missing", None, Some("resident_owner")))?
            .checked_add(excess_bytes)
            .ok_or_else(|| graph_schedule_error("resident_owner_phase_peak_overflow", None, Some("resident_owner")))?;
        self.complete_phase_peak_bytes = Some(complete_peak);
        self.required_phase_floor_bytes = self.required_phase_floor_bytes.max(complete_peak);
        Ok(self)
    }

    #[doc(hidden)]
pub fn bound_derivation(&self) -> Option<AttemptBoundDerivation> {
        self.bound_derivation
    }

    fn required_grant_bound_bytes(&self) -> Result<u64> {
        match self.bound_derivation {
            Some(AttemptBoundDerivation::OrderedTimeline) => self
                .ordered_schedule
                .as_ref()
                .ok_or_else(|| graph_schedule_error("ordered_schedule_missing", None, Some("bound")))?
                .checked_fold_with_inventory(&self.scopes, &self.owner_terms),
            Some(AttemptBoundDerivation::ConservativeOwnerSum) => self
                .complete_phase_peak_bytes
                .ok_or_else(|| graph_schedule_error("conservative_complete_phase_peak_missing", None, Some("bound"))),
            None => Ok(self.complete_phase_peak_bytes.unwrap_or(self.required_phase_floor_bytes)),
        }
    }

    #[cfg(test)]
    #[doc(hidden)]
pub fn ordered_schedule(&self) -> Option<&AttemptReservationSchedule> {
        self.ordered_schedule.as_ref()
    }

    #[doc(hidden)]
    pub fn validate_composed_execution(
        &self,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
        immutable_grant_bytes: u64,
    ) -> Result<()> {
        if self.ordered_schedule.is_none() {
            let nested_matches = self
                .scopes
                .iter()
                .filter(|scope| scope.nested_consumer_occurrence_ids().is_some() && scope.matches(consumer_plan))
                .collect::<Vec<_>>();
            if let [nested] = nested_matches.as_slice() {
                validate_nested_runtime_occurrences(
                    nested
                        .nested_consumer_occurrence_ids()
                        .expect("nested match retains its occurrence identities"),
                    occurrence_ids,
                )?;
                let required_peak = self.required_grant_bound_bytes()?;
                if immutable_grant_bytes < required_peak {
                    return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                        .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                        .with_fact("complete_phase_peak_bytes", required_peak));
                }
                return Ok(());
            }
            let passthroughs = self
                .scopes
                .iter()
                .filter(|scope| scope.is_accounted_by_producer_phase() && scope.matches(consumer_plan))
                .count();
            if passthroughs == 1 {
                let manifest = admitted_manifest_shape(&self.scopes, None)?;
                let [occurrence] = manifest.producer_occurrences() else {
                    return Err(graph_schedule_error(
                        "deferred_passthrough_occurrence_missing_or_ambiguous",
                        None,
                        Some("runtime"),
                    ));
                };
                let expected_occurrence = occurrence.occurrence_id();
                let [runtime_occurrence] = occurrence_ids else {
                    return Err(graph_schedule_error(
                        "runtime_occurrence_count_mismatch",
                        None,
                        Some("runtime"),
                    )
                    .with_fact("scheduled_occurrence_count", 1)
                    .with_fact("runtime_occurrence_count", occurrence_ids.len()));
                };
                if runtime_occurrence != expected_occurrence {
                    return Err(graph_schedule_error(
                        "runtime_occurrence_order_mismatch",
                        Some(runtime_occurrence),
                        Some("runtime"),
                    )
                    .with_fact("scheduled_occurrence_id", expected_occurrence));
                }
                let required_peak = self.required_grant_bound_bytes()?;
                if immutable_grant_bytes < required_peak {
                    return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                        .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                        .with_fact("complete_phase_peak_bytes", required_peak));
                }
                return Ok(());
            }
        }
        let manifest = admitted_manifest_shape(&self.scopes, self.ordered_schedule.as_ref())?;
        manifest.validate_manifest_core()?;
        if let Some(schedule) = &self.ordered_schedule {
            let is_manifest_consumer = manifest.matches(consumer_plan);
            let passthrough_matches = self
                .scopes
                .iter()
                .filter(|scope| scope.is_accounted_by_producer_phase() && scope.matches(consumer_plan))
                .count();
            let nested_matches = self
                .scopes
                .iter()
                .filter(|scope| scope.nested_consumer_occurrence_ids().is_some() && scope.matches(consumer_plan))
                .collect::<Vec<_>>();
            if is_manifest_consumer {
                schedule.validate_runtime(manifest, consumer_plan, occurrence_ids)?;
            } else if let [nested] = nested_matches.as_slice() {
                validate_nested_runtime_occurrences(
                    nested
                        .nested_consumer_occurrence_ids()
                        .expect("nested match retains its occurrence identities"),
                    occurrence_ids,
                )?;
            } else if passthrough_matches != 1 {
                return Err(graph_schedule_error("runtime_consumer_scope_mismatch", None, Some("runtime")));
            }
        } else {
            if manifest.execution_scope_fingerprint() != canonical_execution_scope_fingerprint(consumer_plan) {
                return Err(graph_schedule_error("runtime_consumer_scope_mismatch", None, Some("runtime")));
            }
            if manifest.producer_occurrences().len() != occurrence_ids.len() {
                return Err(graph_schedule_error("runtime_occurrence_count_mismatch", None, Some("runtime"))
                    .with_fact("scheduled_occurrence_count", manifest.producer_occurrences().len())
                    .with_fact("runtime_occurrence_count", occurrence_ids.len()));
            }
            for (scheduled, runtime) in manifest.producer_occurrences().iter().zip(occurrence_ids) {
                if scheduled.occurrence_id() != runtime {
                    return Err(graph_schedule_error(
                        "runtime_occurrence_order_mismatch",
                        Some(runtime),
                        Some("runtime"),
                    )
                    .with_fact("scheduled_occurrence_id", scheduled.occurrence_id()));
                }
            }
        }
        let required_peak = self.required_grant_bound_bytes()?;
        if immutable_grant_bytes < required_peak {
            return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                .with_fact("complete_phase_peak_bytes", required_peak));
        }
        Ok(())
    }

    /// Validates the admitted ordered schedule before optionally publishing
    /// its identity-bound projection. Ordinary contexts have no attempt
    /// identity and therefore validate without publishing an artifact.
    #[doc(hidden)]
pub fn validate_and_project_composed_execution(
        &self,
        consumer_plan: &QueryPlan,
        occurrence_ids: &[String],
        immutable_grant_bytes: u64,
        projection_identity: Option<(crate::schedule_projection::AttemptId, usize, u64)>,
    ) -> Result<Option<AttemptComposedScheduleProjection>> {
        self.validate_composed_execution(consumer_plan, occurrence_ids, immutable_grant_bytes)?;
        let Some((attempt_id, device_ordinal, grant_id)) = projection_identity else {
            return Ok(None);
        };
        self.composed_schedule_projection(attempt_id, device_ordinal, grant_id, immutable_grant_bytes).map(Some)
    }

    /// Publishes the immutable schedule priced by this envelope, bound to the
    /// grant and device selected by the admission transaction.
    #[doc(hidden)]
pub fn composed_schedule_projection(
        &self,
        attempt_id: crate::schedule_projection::AttemptId,
        device_ordinal: usize,
        grant_id: u64,
        grant_bytes: u64,
    ) -> Result<AttemptComposedScheduleProjection> {
        let Some(schedule) = &self.ordered_schedule else {
            return Err(graph_schedule_error("missing_admitted_schedule", None, Some("projection")));
        };
        schedule.projection(&self.scopes, &self.owner_terms, attempt_id, device_ordinal, grant_id, grant_bytes)
    }

    #[doc(hidden)]
pub fn validate_execution(
        &self,
        plan: &QueryPlan,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> Result<()> {
        if !self.policy.matches_execution(
            memory_policy,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        ) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "admitted reservation envelope does not match the execution policy",
            )
            .with_fact("reason", "admitted_reservation_policy_mismatch"));
        }
        if self.scopes.iter().any(|scope| scope.matches(plan)) {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.reservation_envelope",
            "admitted reservation envelope does not cover the native plan about to execute",
        )
        .with_fact("reason", "admitted_reservation_envelope_mismatch"))
    }

    #[doc(hidden)]
pub fn validate_execution_with_grant(
        &self,
        plan: &QueryPlan,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
        immutable_grant_bytes: u64,
    ) -> Result<()> {
        self.validate_execution(
            plan,
            memory_policy,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        )?;
        let required_bound = self.required_grant_bound_bytes()?;
        if immutable_grant_bytes < required_bound {
            return Err(graph_schedule_error("immutable_grant_below_schedule_peak", None, Some("runtime"))
                .with_fact("immutable_grant_bytes", immutable_grant_bytes)
                .with_fact("complete_phase_peak_bytes", required_bound));
        }
        Ok(())
    }
}

fn envelope_overflow_error(phase: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.admission.reservation_envelope",
        "attempt reservation envelope overflows u64",
    )
    .with_fact("reason", "attempt_reservation_envelope_overflow")
    .with_fact("phase", phase)
}

fn graph_schedule_error(reason: &'static str, occurrence_id: Option<&str>, phase: Option<&'static str>) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.reservation_envelope",
        "a composed graph attempt lacks a complete ordered memory schedule",
    )
    .with_fact("reason", GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON)
    .with_fact("schedule_reason", reason);
    if let Some(occurrence_id) = occurrence_id {
        error = error.with_fact("occurrence_id", occurrence_id);
    }
    if let Some(phase) = phase {
        error = error.with_fact("phase", phase);
    }
    error
}

fn derivation_unavailable_error(ordered: QueryError, conservative: QueryError) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.reservation_envelope",
        "no complete bounded attempt derivation is available",
    )
    .with_fact("reason", "attempt_bound_derivation_unavailable")
    .with_fact("ordered_reason", ordered.fact_value("schedule_reason").unwrap_or("ordered_timeline_unavailable"))
    .with_fact("conservative_reason", conservative.fact_value("reason").unwrap_or("conservative_owner_sum_unavailable"))
}

fn owner_derivation_incomplete_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.reservation_envelope",
        "attempt owner-term inventory does not cover every incomplete scope",
    )
    .with_fact("reason", "attempt_owner_term_coverage_incomplete")
    .with_fact("derivation", "conservative_owner_sum")
}
