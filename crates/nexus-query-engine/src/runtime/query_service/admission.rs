use super::*;

/// Cancel-safe queued admission request.
///
/// Dropping this value before admission removes the queued request from the
/// service queue. This lets async servers wait without parking a blocking
/// thread inside [`QueryService::admit_query`].
#[derive(Debug)]
pub struct QueryAdmissionWaiter {
    pub(super) service: QueryService,
    pub(super) tenant_id: TenantId,
    pub(super) request_id: u64,
    pub(super) requested_at: Instant,
    pub(super) deadline: Instant,
    pub(super) grant_policy: TenantGrantPolicy,
    pub(super) queued: bool,
}

#[derive(Debug, Default)]
pub(super) struct AdmissionState {
    pub(super) active_queries: usize,
    pub(super) queued: VecDeque<AdmissionRequest>,
    pub(super) active_adaptive_grants: HashMap<u64, ActiveAdaptiveGrant>,
    pub(super) next_tenant_index: usize,
    pub(super) closed: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ActiveAdaptiveGrant {
    pub(super) device_ordinal: usize,
    pub(super) min_bytes: u64,
    pub(super) target_bytes: u64,
    pub(super) memory_grant: MemoryGrant,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AdaptiveGrowthCandidate {
    pub(super) query_id: u64,
    pub(super) missing_bytes: u64,
    pub(super) weight: u32,
}

#[derive(Debug, Clone)]
pub(super) struct AdmissionRequest {
    pub(super) request_id: u64,
    pub(super) tenant_id: TenantId,
    pub(super) grant_policy: TenantGrantPolicy,
    pub(super) qos: QueryQos,
    pub(super) placement_hint: QueryPlacementHint,
    pub(super) deadline: Instant,
}

#[derive(Debug)]
pub(super) struct PlacedQuery {
    pub(super) device_ordinal: usize,
    pub(super) memory_grant: MemoryGrant,
    pub(super) device_slot: Option<DeviceActiveQuerySlot>,
}

pub(super) fn weighted_adaptive_growth_allocations(
    candidates: &[AdaptiveGrowthCandidate],
    available_bytes: u64,
) -> Vec<(u64, u64)> {
    if available_bytes == 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut remaining = candidates
        .iter()
        .filter_map(|candidate| {
            (candidate.missing_bytes > 0).then_some(AdaptiveGrowthCandidate {
                query_id: candidate.query_id,
                missing_bytes: candidate.missing_bytes,
                weight: candidate.weight.max(QueryQos::DEFAULT_BUDGET_WEIGHT),
            })
        })
        .collect::<Vec<_>>();
    remaining.sort_by_key(|candidate| candidate.query_id);

    let mut available = available_bytes;
    let mut allocations = BTreeMap::<u64, u64>::new();
    while available > 0 && !remaining.is_empty() {
        let total_weight = remaining.iter().map(|candidate| u128::from(candidate.weight)).sum::<u128>().max(1);
        let mut round_allocations = vec![0u64; remaining.len()];
        let mut remainders = Vec::with_capacity(remaining.len());
        let mut round_total = 0u64;
        for (index, candidate) in remaining.iter().enumerate() {
            let weighted_available = u128::from(available).saturating_mul(u128::from(candidate.weight));
            let base_share = (weighted_available / total_weight).min(u128::from(candidate.missing_bytes)) as u64;
            round_allocations[index] = base_share;
            round_total = round_total.saturating_add(base_share);
            if base_share < candidate.missing_bytes {
                remainders.push((index, weighted_available % total_weight, candidate.weight, candidate.query_id));
            }
        }

        remainders.sort_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| right.2.cmp(&left.2)).then_with(|| left.3.cmp(&right.3))
        });
        if round_total == 0 {
            let Some((index, ..)) = remainders.first().copied() else {
                break;
            };
            round_allocations[index] = 1;
            round_total = 1;
        } else if round_total < available {
            let mut extra = available - round_total;
            for (index, ..) in remainders {
                if extra == 0 {
                    break;
                }
                if round_allocations[index] < remaining[index].missing_bytes {
                    round_allocations[index] = round_allocations[index].saturating_add(1);
                    round_total = round_total.saturating_add(1);
                    extra -= 1;
                }
            }
        }

        if round_total == 0 {
            break;
        }
        for (index, bytes) in round_allocations.into_iter().enumerate() {
            if bytes == 0 {
                continue;
            }
            let candidate = &mut remaining[index];
            let bytes = bytes.min(candidate.missing_bytes);
            let allocation = allocations.entry(candidate.query_id).or_default();
            *allocation = allocation.saturating_add(bytes);
            candidate.missing_bytes = candidate.missing_bytes.saturating_sub(bytes);
        }
        available = available.saturating_sub(round_total);
        remaining.retain(|candidate| candidate.missing_bytes > 0);
    }

    allocations.into_iter().collect()
}

impl QueryAdmissionWaiter {
    /// Attempts to complete this queued admission without blocking.
    ///
    /// Returns `Ok(None)` while the request is still queued behind active work
    /// or tenant-fairness rotation. Queue timeout and service close remove the
    /// request and return a structured error.
    pub fn try_admit(&mut self) -> Result<Option<QueryHandle>> {
        if !self.queued {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "query admission waiter has already completed",
            )
            .with_fact("tenant_id", self.tenant_id.as_str())
            .with_fact("query_id", self.request_id)
            .with_fact("reason", "admission_waiter_completed"));
        }

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.service.lock_admission()?;
        if state.closed {
            self.service.remove_queued_request_without_admission_locked(&mut state, self.request_id);
            self.queued = false;
            self.service.inner.notify_admission_available();
            return Err(service_closed_error(&self.tenant_id));
        }
        let request_is_admissible = match self.service.request_is_admissible(&mut state, self.request_id) {
            Ok(admissible) => admissible,
            Err(error) => {
                self.service.abandon_queued_request_locked(&mut state, self.request_id);
                self.queued = false;
                return Err(error);
            },
        };
        if request_is_admissible
            && let Some(position) = state.queued.iter().position(|queued| queued.request_id == self.request_id)
        {
            let placed_query = match self.service.try_place_query(
                &self.tenant_id,
                self.request_id,
                self.grant_policy,
                state.queued[position].qos,
                &state.queued[position].placement_hint,
                Some(self.deadline),
                state.active_queries,
                &mut slot_notifications,
            ) {
                Ok(placed_query) => placed_query,
                Err(error) => {
                    self.service.abandon_queued_request_locked(&mut state, self.request_id);
                    self.queued = false;
                    return Err(error);
                },
            };
            let Some(placed_query) = placed_query else {
                if let Err(error) = self
                    .service
                    .request_adaptive_grant_shrink_for_blocked_request_locked(&state, &state.queued[position])
                {
                    self.service.abandon_queued_request_locked(&mut state, self.request_id);
                    self.queued = false;
                    return Err(error);
                }
                if Instant::now() >= self.deadline {
                    self.service.abandon_queued_request_locked(&mut state, self.request_id);
                    self.queued = false;
                    return Err(self.service.memory_grant_wait_timeout_error(
                        &self.tenant_id,
                        self.grant_policy,
                        state.queued.len(),
                    ));
                }
                return Ok(None);
            };
            let admitted = state.queued.remove(position).expect("admissible request exists");
            self.service.attach_admission_release_wakeup(&placed_query.memory_grant)?;
            self.service.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant)?;
            state.active_queries = state.active_queries.saturating_add(1);
            self.service.register_active_adaptive_grant(
                &mut state,
                self.request_id,
                placed_query.device_ordinal,
                self.grant_policy,
                &placed_query.memory_grant,
            );
            self.service.rotate_after_admission(&mut state, &admitted.tenant_id);
            self.queued = false;
            self.service.refresh_active_adaptive_shrink_requests_locked(&state);
            if state.queued.is_empty() {
                self.service.grow_active_adaptive_grants_locked(&mut state);
            }
            self.service.inner.notify_admission_available();
            let admission_sequence = self.service.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
            let service = Arc::clone(&self.service.inner);
            let device_budget_bytes = placed_query.memory_grant.budget_bytes();
            let device_ordinal = placed_query.device_ordinal;
            return Ok(Some(QueryHandle {
                lifetime: QueryHandleLifetime::new(
                    Arc::clone(&service),
                    self.tenant_id.clone(),
                    self.request_id,
                    placed_query.memory_grant,
                    placed_query.device_slot,
                ),
                service,
                tenant_id: self.tenant_id.clone(),
                query_id: self.request_id,
                admission_sequence,
                admission_wait: self.requested_at.elapsed(),
                device_budget_bytes,
                device_ordinal,
            }));
        }

        if Instant::now() >= self.deadline {
            self.service.abandon_queued_request_locked(&mut state, self.request_id);
            self.queued = false;
            return Err(service_overloaded_error(
                "admission_wait_timeout",
                &self.tenant_id,
                state.queued.len(),
                self.service.inner.config.max_admission_queue_len,
            ));
        }

        Ok(None)
    }

    /// Remaining time before this queued admission reaches the service timeout.
    pub fn remaining_wait(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

impl Drop for QueryAdmissionWaiter {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        match self.service.inner.admission.lock() {
            Ok(mut state) => {
                self.service.remove_queued_request_without_admission_locked(&mut state, self.request_id);
                self.service.inner.notify_admission_available();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to remove queued query service admission request");
            },
        }
    }
}

pub(super) fn remove_queued_request(state: &mut AdmissionState, request_id: u64) -> Option<AdmissionRequest> {
    if let Some(position) = state.queued.iter().position(|request| request.request_id == request_id) {
        state.queued.remove(position)
    } else {
        None
    }
}

pub(super) fn service_overloaded_error(
    reason: &'static str,
    tenant_id: &TenantId,
    queued_queries: usize,
    max_admission_queue_len: usize,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.query_service.admission",
        "query service admission is overloaded",
    )
    .with_fact("reason", reason)
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact("queued_queries", queued_queries)
    .with_fact("max_admission_queue_len", max_admission_queue_len)
}

pub(super) fn service_closed_error(tenant_id: &TenantId) -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "nexus_query_engine.query_service.admission",
        "query service admission is closed",
    )
    .with_fact("reason", "admission_closed")
    .with_fact("tenant_id", tenant_id.as_str())
}
