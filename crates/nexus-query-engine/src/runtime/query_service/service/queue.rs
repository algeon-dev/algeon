//! Admission queue selection, sweep, lock helpers, and rotation.

use super::*;

impl QueryService {
    pub(in crate::runtime::query_service) fn remove_queued_request_without_admission_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = remove_queued_request(state, admission_request_id);
        if removed.is_some() {
            self.refresh_active_adaptive_shrink_requests_locked(state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(state);
            }
        }
        removed
    }

    pub(in crate::runtime::query_service) fn abandon_queued_request_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = self.remove_queued_request_without_admission_locked(state, admission_request_id);
        if removed.is_some() {
            self.inner.notify_admission_available();
        }
        removed
    }

    pub(in crate::runtime::query_service) fn lock_admission(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        self.inner.admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission",
                "query service admission mutex was poisoned",
            )
        })
    }

    pub(in crate::runtime::query_service) fn admission_is_closed(&self, state: &AdmissionState) -> bool {
        !self.inner.admission_open.load(Ordering::Acquire) || state.closed
    }

    pub(in crate::runtime::query_service::service) fn lock_admission_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        match crate::runtime::deadline::mutex_lock_until(&self.inner.admission, deadline) {
            Ok(state) => Ok(state),
            Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => self.lock_admission(),
            Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                Err(query_service_shutdown_timeout(operation, None, "admission_mutex"))
            },
        }
    }

    pub(in crate::runtime::query_service) fn request_is_admissible(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Result<bool> {
        self.sweep_expired_queued_requests_locked(state);
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return Ok(false);
        }
        let Some(position) = self.select_next_request_position(state)? else {
            return Ok(false);
        };
        Ok(state.queued[position].admission_request_id == admission_request_id)
    }

    pub(in crate::runtime::query_service::service) fn sweep_expired_queued_requests_locked(
        &self,
        state: &mut AdmissionState,
    ) {
        let now = Instant::now();
        let mut expired = Vec::new();
        let mut retained = VecDeque::with_capacity(state.queued.len());
        while let Some(request) = state.queued.pop_front() {
            if request.deadline <= now {
                expired.push(request);
            } else {
                retained.push_back(request);
            }
        }
        state.queued = retained;
        if !expired.is_empty() {
            self.refresh_active_adaptive_shrink_requests_locked(state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(state);
            }
            self.inner.notify_admission_available();
            for request in expired {
                record_admission_lifecycle(
                    &request.observation,
                    request.admission_request_id,
                    &request.tenant_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
        }
    }

    pub(in crate::runtime::query_service::service) fn select_next_request_position(
        &self,
        state: &AdmissionState,
    ) -> Result<Option<usize>> {
        if state.queued.is_empty() {
            return Ok(None);
        }

        let mut first_candidate = None;
        for priority in Priority::DESCENDING {
            for offset in 0..self.inner.tenant_order.len() {
                let tenant_index = (state.next_tenant_index + offset) % self.inner.tenant_order.len();
                let tenant_id = &self.inner.tenant_order[tenant_index];
                for position in state.queued.iter().enumerate().filter_map(|(position, request)| {
                    (&request.tenant_id == tenant_id && request.qos.priority() == priority).then_some(position)
                }) {
                    first_candidate.get_or_insert(position);
                    if self.request_can_receive_minimum_grant(state, &state.queued[position])? {
                        return Ok(Some(position));
                    }
                }
            }
        }

        Ok(first_candidate.or(Some(0)))
    }

    pub(in crate::runtime::query_service::service) fn request_can_receive_minimum_grant(
        &self,
        state: &AdmissionState,
        request: &AdmissionRequest,
    ) -> Result<bool> {
        let Some(resources) = &self.inner.resources else {
            if !self.fallback_active_query_cap_allows(state.active_queries) {
                return Ok(false);
            }
            let available_bytes = self.inner.fallback_memory_governor.available_bytes()?;
            return Ok(self
                .grant_budget_bytes(
                    available_bytes,
                    request.grant_policy,
                    self.reserved_min_bytes_for_remaining_slots(state.active_queries),
                )
                .is_some());
        };
        let placement_request =
            PlacementRequest::with_grant_range(request.grant_policy.min_bytes, request.grant_policy.want_bytes)
                .with_tenant_id(request.tenant_id.as_str());
        for device in self.placement_candidates(resources, &placement_request)? {
            let context = resources.devices.context(device)?;
            if !self.device_active_query_cap_allows(context) {
                continue;
            }
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, context.device(), state.active_queries)?;
            if context.capacity_bytes() >= request.grant_policy.min_bytes
                && self
                    .grant_budget_bytes(context.available_bytes()?, request.grant_policy, reserved_for_remaining_slots)
                    .is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(in crate::runtime::query_service) fn memory_grant_wait_timeout_error(
        &self,
        tenant_id: &TenantId,
        grant_policy: TenantGrantPolicy,
        queued_queries: usize,
    ) -> QueryError {
        let mut error = service_overloaded_error(
            "memory_grant_wait_timeout",
            tenant_id,
            queued_queries,
            self.inner.config.max_admission_queue_len,
        )
        .with_fact("requested_grant_min_bytes", grant_policy.min_bytes)
        .with_fact("requested_grant_bytes", grant_policy.want_bytes)
        .with_fact("effective_min_device_budget_bytes", grant_policy.min_bytes)
        .with_fact("effective_want_device_budget_bytes", grant_policy.want_bytes)
        .with_fact("slot_guarantee", self.inner.config.slot_guarantee.as_str())
        .with_fact("max_concurrent_queries", self.inner.config.max_concurrent_queries)
        .with_fact("max_tenant_min_device_budget_bytes", self.inner.max_tenant_min_bytes);
        if let Ok(ceiling) = self.guaranteed_min_idle_grant_ceiling_bytes() {
            error = error.with_fact("guaranteed_min_idle_grant_ceiling_bytes", ceiling);
        }
        error
    }

    pub(in crate::runtime::query_service) fn rotate_after_admission(
        &self,
        state: &mut AdmissionState,
        tenant_id: &TenantId,
    ) {
        if let Some(index) = self.inner.tenant_order.iter().position(|candidate| candidate == tenant_id) {
            state.next_tenant_index = (index + 1) % self.inner.tenant_order.len();
        }
    }
}

pub(in crate::runtime::query_service::service) fn query_service_shutdown_timeout(
    operation: &'static str,
    state: Option<&AdmissionState>,
    deadline_stage: &'static str,
) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query service shutdown cleanup",
    )
    .with_fact("reason", "query_service_shutdown_timeout")
    .with_fact("deadline_stage", deadline_stage);
    match state {
        Some(state) => error
            .with_fact("active_queries", state.active_queries)
            .with_fact("cleanup_in_progress", state.cleanup_in_progress)
            .with_fact("queued_queries", state.queued.len()),
        None => error,
    }
}
