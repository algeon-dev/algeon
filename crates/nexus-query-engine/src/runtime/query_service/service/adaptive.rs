//! Adaptive grant registration, growth, and shrink under pressure.

use super::*;

impl QueryService {
    pub(in crate::runtime::query_service) fn adaptive_shrink_deficit_bytes(
        available_bytes: u64,
        min_bytes: u64,
        reserved_for_remaining_slots: u64,
        capacity_bytes: u64,
    ) -> Option<u64> {
        if reserved_for_remaining_slots == u64::MAX {
            return None;
        }
        let required_floor = min_bytes.checked_add(reserved_for_remaining_slots)?;
        if required_floor > capacity_bytes || available_bytes >= required_floor {
            return None;
        }
        Some(required_floor - available_bytes)
    }

    pub(in crate::runtime::query_service) fn register_active_adaptive_grant(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
        device_ordinal: usize,
        grant_policy: TenantGrantPolicy,
        memory_grant: &MemoryGrant,
    ) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        if grant_policy.want_bytes <= grant_policy.min_bytes {
            return;
        }
        state.active_adaptive_grants.insert(
            admission_request_id,
            ActiveAdaptiveGrant {
                device_ordinal,
                min_bytes: memory_grant.min_bytes(),
                target_bytes: grant_policy.want_bytes.max(memory_grant.min_bytes()),
                memory_grant: memory_grant.clone(),
            },
        );
    }

    pub(in crate::runtime::query_service::service) fn adaptive_growth_available_bytes(
        &self,
        state: &AdmissionState,
        device_ordinal: usize,
    ) -> Result<u64> {
        let Some(resources) = &self.inner.resources else {
            let available = self.inner.fallback_memory_governor.available_bytes()?;
            return Ok(available.saturating_sub(self.reserved_min_bytes_for_empty_slots(state.active_queries)));
        };
        let context = resources.devices.context_by_ordinal(device_ordinal)?;
        let available = context.available_bytes()?;
        let reserved =
            self.reserved_min_bytes_for_selected_device_empty_slots(resources, context.device(), state.active_queries)?;
        Ok(available.saturating_sub(reserved))
    }

    pub(in crate::runtime::query_service) fn grow_active_adaptive_grants_locked(&self, state: &mut AdmissionState) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        let device_ordinals =
            state.active_adaptive_grants.values().map(|active| active.device_ordinal).collect::<BTreeSet<_>>();
        for device_ordinal in device_ordinals {
            let available_bytes = match self.adaptive_growth_available_bytes(state, device_ordinal) {
                Ok(bytes) => bytes,
                Err(error) => {
                    eprintln!("[nexus-query-engine] warning: failed to compute adaptive grant growth: {error}");
                    continue;
                },
            };
            if available_bytes == 0 {
                continue;
            }
            let mut candidates = Vec::new();
            let admission_request_ids = state
                .active_adaptive_grants
                .iter()
                .filter_map(|(admission_request_id, active)| {
                    (active.device_ordinal == device_ordinal).then_some(*admission_request_id)
                })
                .collect::<Vec<_>>();
            for admission_request_id in admission_request_ids {
                let Some(active) = state.active_adaptive_grants.get(&admission_request_id) else {
                    continue;
                };
                let current_bytes = active.memory_grant.budget_bytes();
                let target_bytes = active.target_bytes.max(active.min_bytes);
                if current_bytes >= target_bytes {
                    continue;
                }
                candidates.push(AdaptiveGrowthCandidate {
                    admission_request_id,
                    missing_bytes: target_bytes.saturating_sub(current_bytes),
                    weight: active.memory_grant.qos().budget_weight(),
                });
            }
            for (admission_request_id, allocated_bytes) in
                weighted_adaptive_growth_allocations(&candidates, available_bytes)
            {
                let Some(active) = state.active_adaptive_grants.get(&admission_request_id).cloned() else {
                    continue;
                };
                let current_bytes = active.memory_grant.budget_bytes();
                let target_bytes = active.target_bytes.max(active.min_bytes);
                let grow_bytes = target_bytes.saturating_sub(current_bytes).min(allocated_bytes);
                if grow_bytes == 0 {
                    continue;
                }
                match active.memory_grant.try_grow(grow_bytes) {
                    Ok(true) => continue,
                    Ok(false) => continue,
                    Err(error) => {
                        eprintln!("[nexus-query-engine] warning: failed to grow adaptive grant: {error}");
                        continue;
                    },
                }
            }
        }
    }

    pub(in crate::runtime::query_service) fn request_adaptive_grant_shrink_for_blocked_request_locked(
        &self,
        state: &AdmissionState,
        request: &AdmissionRequest,
    ) -> Result<()> {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return Ok(());
        }
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return Ok(());
        }

        let min_bytes = request.grant_policy.min_bytes;
        let Some(resources) = &self.inner.resources else {
            if !self.fallback_active_query_cap_allows(state.active_queries) {
                return Ok(());
            }
            let available_bytes = self.inner.fallback_memory_governor.available_bytes()?;
            if available_bytes < min_bytes {
                self.request_active_adaptive_grant_shrink_on_device_locked(
                    state,
                    self.inner.config.device_ordinal,
                    min_bytes - available_bytes,
                )?;
            }
            return Ok(());
        };

        let placement_request = PlacementRequest::with_grant_range(min_bytes, request.grant_policy.want_bytes)
            .with_tenant_id(request.tenant_id.as_str())
            .with_placement_hint(request.placement_hint.clone());
        for device in self.placement_candidates(resources, &placement_request)? {
            let context = resources.devices.context(device)?;
            if context.capacity_bytes() < min_bytes {
                continue;
            }
            if !self.device_active_query_cap_allows(context) {
                continue;
            }
            let available_bytes = context.available_bytes()?;
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, device, state.active_queries)?;
            if self.grant_budget_bytes(available_bytes, request.grant_policy, reserved_for_remaining_slots).is_some() {
                return Ok(());
            }
            if let Some(deficit_bytes) = Self::adaptive_shrink_deficit_bytes(
                available_bytes,
                min_bytes,
                reserved_for_remaining_slots,
                context.capacity_bytes(),
            ) {
                self.request_active_adaptive_grant_shrink_on_device_locked(state, device.ordinal(), deficit_bytes)?;
            }
        }
        Ok(())
    }

    pub(in crate::runtime::query_service::service) fn request_active_adaptive_grant_shrink_on_device_locked(
        &self,
        state: &AdmissionState,
        device_ordinal: usize,
        deficit_bytes: u64,
    ) -> Result<()> {
        if deficit_bytes == 0 {
            return Ok(());
        }
        let mut admission_request_ids = state
            .active_adaptive_grants
            .iter()
            .filter_map(|(admission_request_id, active)| {
                (active.device_ordinal == device_ordinal).then_some(*admission_request_id)
            })
            .collect::<Vec<_>>();
        admission_request_ids.sort_unstable();

        let mut remaining_deficit = deficit_bytes;
        for admission_request_id in admission_request_ids {
            let Some(active) = state.active_adaptive_grants.get(&admission_request_id) else {
                continue;
            };
            let current_bytes = active.memory_grant.budget_bytes();
            if current_bytes <= active.min_bytes {
                continue;
            }
            let shrinkable_bytes = current_bytes - active.min_bytes;
            let shrink_bytes = shrinkable_bytes.min(remaining_deficit);
            active.memory_grant.request_shrink_to(current_bytes - shrink_bytes)?;
            remaining_deficit = remaining_deficit.saturating_sub(shrink_bytes);
            if remaining_deficit == 0 {
                break;
            }
        }
        Ok(())
    }

    pub(in crate::runtime::query_service) fn refresh_active_adaptive_shrink_requests_locked(
        &self,
        state: &AdmissionState,
    ) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        for active in state.active_adaptive_grants.values() {
            if let Err(error) = active.memory_grant.clear_requested_shrink() {
                eprintln!("[nexus-query-engine] warning: failed to rescind adaptive grant shrink request: {error}");
            }
        }
        for request in &state.queued {
            if let Err(error) = self.request_adaptive_grant_shrink_for_blocked_request_locked(state, request) {
                eprintln!("[nexus-query-engine] warning: failed to refresh adaptive grant shrink request: {error}");
            }
        }
    }
}
