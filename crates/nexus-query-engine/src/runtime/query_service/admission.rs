use super::*;

/// Cancel-safe queued admission request.
///
/// Dropping this value before admission removes the queued request from the
/// service queue. This lets async servers wait without parking a blocking
/// thread inside [`QueryService::admit_query`].
#[derive(Debug)]
pub(crate) struct QueryAdmissionWaiter {
    pub(super) service: QueryService,
    pub(super) tenant_id: TenantId,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) requested_at: Instant,
    pub(super) deadline: Instant,
    pub(super) grant_policy: TenantGrantPolicy,
    pub(super) queued: bool,
}

#[derive(Debug, Default)]
pub(super) struct AdmissionState {
    pub(super) active_queries: usize,
    /// Query releases whose GPU-owned resources are still being destroyed.
    ///
    /// Grant callbacks must not wake admission while this is non-zero: the
    /// released capacity is not reusable until cleanup drops the device slot.
    pub(super) cleanup_in_progress: usize,
    pub(super) queued: VecDeque<AdmissionRequest>,
    pub(super) active_adaptive_grants: HashMap<AdmissionRequestId, ActiveAdaptiveGrant>,
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
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) missing_bytes: u64,
    pub(super) weight: u32,
}

#[derive(Debug)]
pub(super) struct AdmissionRequest {
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) observation: RuntimeObservationHandle,
    pub(super) completion: QueryCompletion,
    pub(super) tenant_id: TenantId,
    pub(super) grant_policy: TenantGrantPolicy,
    #[cfg(feature = "cugraph")]
    pub(super) warm_graph_grant_policy: Option<TenantGrantPolicy>,
    pub(super) qos: QueryQos,
    pub(super) placement_hint: QueryPlacementHint,
    pub(super) deadline: Instant,
}

/// The one ownership boundary for terminal admission completion.
///
/// This payload starts at backend admission, remains with a queued request,
/// and, once admitted, stays with the query handle until every GPU-bound
/// resource has been torn down.  It deliberately owns the opaque activity
/// lease so activity callbacks cannot run while admission or resource locks
/// are held.
#[derive(Debug)]
pub(crate) struct QueryCompletion {
    observation_owner: Option<RuntimeObservationOwner>,
    observation_permit: Option<ObservationPermit>,
    observation_activity: Option<QueryActivityLease>,
    completion_dispatcher: Option<ObservationCompletionDispatcher>,
    gpu_resource_permit: Option<GpuResourcePermit>,
}

impl QueryCompletion {
    pub(super) fn new(
        observation_owner: Option<RuntimeObservationOwner>,
        observation_permit: Option<ObservationPermit>,
        observation_activity: Option<QueryActivityLease>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
        gpu_resource_permit: Option<GpuResourcePermit>,
    ) -> Self {
        Self { observation_owner, observation_permit, observation_activity, completion_dispatcher, gpu_resource_permit }
    }

    /// Transfers the GPU-generation permit into the handle's internal cleanup
    /// owner. The remaining payload can only run after that owner has dropped
    /// the permit and all CUDA-bound state.
    pub(super) fn into_admitted(mut self) -> (Self, Option<GpuResourcePermit>) {
        let gpu_resource_permit = self.gpu_resource_permit.take();
        (self, gpu_resource_permit)
    }

    pub(super) fn observation_owner(&self) -> Option<RuntimeObservationOwner> {
        self.observation_owner.clone()
    }

    pub(super) fn begin_query_cleanup(&self) {
        if let Some(owner) = &self.observation_owner {
            owner.begin_query_cleanup();
        }
    }

    pub(super) fn finish_error(self, error: &QueryError) {
        self.finish(RuntimeTerminalOutcome::Failed {
            error_code: error.code().to_string(),
            error_status: error.status().as_str().to_string(),
        });
    }

    pub(super) fn finish_timeout(self, error: &QueryError) {
        self.finish_error(error);
    }

    pub(super) fn finish_cancelled(self) {
        self.finish(RuntimeTerminalOutcome::Cancelled);
    }

    pub(super) fn finish_after_gpu_cleanup(self, fallback: RuntimeTerminalOutcome) {
        self.dispatch_after_gpu_cleanup(fallback);
    }

    fn finish(mut self, fallback: RuntimeTerminalOutcome) {
        // A rejected or cancelled request has no handle cleanup owner. Release
        // its backend generation before observer delivery so enqueue fallback
        // cannot synchronously reenter backend close while retaining GPU state.
        drop(self.gpu_resource_permit.take());
        self.dispatch_after_gpu_cleanup(fallback);
    }

    fn dispatch_after_gpu_cleanup(mut self, fallback: RuntimeTerminalOutcome) {
        debug_assert!(self.gpu_resource_permit.is_none(), "GPU permit must leave before observer completion");
        if let Some(dispatcher) = self.completion_dispatcher.take() {
            let activity_tail_dispatcher = dispatcher.clone();
            dispatcher.dispatch(move || {
                let activity = self.finish_observation(fallback);
                if let Some(activity) = activity {
                    activity_tail_dispatcher.dispatch_activity_tail(move || Self::finish_activity(Some(activity)));
                }
            });
        } else {
            self.finish_inline(fallback);
        }
    }

    fn finish_inline(mut self, fallback: RuntimeTerminalOutcome) {
        // Recorder sealing may invoke observer callbacks. It must happen after
        // every GPU reference has been released and outside admission/resource
        // locks. The recorder itself preserves a prior structured execution
        // outcome over this fallback.
        let activity = self.finish_observation(fallback);
        Self::finish_activity(activity);
    }

    /// Seals and delivers the observation while its permit still fences that
    /// work, then relinquishes the permit before returning the opaque activity
    /// lease. The caller decides the activity lane so observer delivery cannot
    /// be starved by a reentrant activity close.
    fn finish_observation(&mut self, fallback: RuntimeTerminalOutcome) -> Option<QueryActivityLease> {
        let owner = self.observation_owner.take();
        let permit = self.observation_permit.take();
        let activity = self.observation_activity.take();
        if let Some(owner) = owner {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = owner.finish_query_cleanup(fallback);
            }));
        }
        // The permit fences observer delivery but cannot include an opaque
        // activity destructor: that destructor may synchronously reenter
        // backend close and wait for unrelated observation permits.
        drop(permit);
        activity
    }

    fn finish_activity(activity: Option<QueryActivityLease>) {
        // Keep activity last even if a recorder observer panicked. A panicking
        // activity destructor is isolated here so Drop never double-panics.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(activity)));
    }
}

impl Drop for QueryCompletion {
    fn drop(&mut self) {
        if self.observation_owner.is_none()
            && self.observation_permit.is_none()
            && self.observation_activity.is_none()
            && self.gpu_resource_permit.is_none()
        {
            return;
        }

        // Recover every terminal path that unwinds before it can explicitly
        // finish. Do not let a cleanup panic turn an already-unwinding path
        // into an abort; recorder sealing receives an explicit Panicked
        // fallback in that case.
        let fallback =
            if std::thread::panicking() { RuntimeTerminalOutcome::Panicked } else { RuntimeTerminalOutcome::Dropped };
        let recovered = Self {
            observation_owner: self.observation_owner.take(),
            observation_permit: self.observation_permit.take(),
            observation_activity: self.observation_activity.take(),
            completion_dispatcher: self.completion_dispatcher.take(),
            gpu_resource_permit: self.gpu_resource_permit.take(),
        };
        // `finish` consumes the payload. Keep a local mutable binding only so
        // its fields are emptied before this Drop implementation returns.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recovered.finish(fallback)));
    }
}

#[derive(Debug)]
pub(super) struct PlacedQuery {
    pub(super) device_ordinal: usize,
    pub(super) memory_grant: MemoryGrant,
    pub(super) device_slot: Option<DeviceActiveQuerySlot>,
    pub(super) grant_policy: TenantGrantPolicy,
    #[cfg(feature = "cugraph")]
    pub(super) warm_graph_reservation: Option<WarmGraphReservation>,
}

pub(super) fn weighted_adaptive_growth_allocations(
    candidates: &[AdaptiveGrowthCandidate],
    available_bytes: u64,
) -> Vec<(AdmissionRequestId, u64)> {
    if available_bytes == 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut remaining = candidates
        .iter()
        .filter_map(|candidate| {
            (candidate.missing_bytes > 0).then_some(AdaptiveGrowthCandidate {
                admission_request_id: candidate.admission_request_id,
                missing_bytes: candidate.missing_bytes,
                weight: candidate.weight.max(QueryQos::DEFAULT_BUDGET_WEIGHT),
            })
        })
        .collect::<Vec<_>>();
    remaining.sort_by_key(|candidate| candidate.admission_request_id);

    let mut available = available_bytes;
    let mut allocations = BTreeMap::<AdmissionRequestId, u64>::new();
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
                remainders.push((
                    index,
                    weighted_available % total_weight,
                    candidate.weight,
                    candidate.admission_request_id,
                ));
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
            let allocation = allocations.entry(candidate.admission_request_id).or_default();
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
            .with_fact("admission_request_id", self.admission_request_id)
            .with_fact("reason", "admission_waiter_completed"));
        }

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = match self.service.inner.admission.lock() {
            Ok(state) => state,
            Err(error) => {
                self.queued = false;
                return Err(self
                    .service
                    .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
            },
        };
        let expired = self.service.sweep_expired_queued_requests_locked(&mut state);
        if !expired.is_empty() {
            drop(state);
            finish_expired_queued_requests(expired);
            state = match self.service.inner.admission.lock() {
                Ok(state) => state,
                Err(error) => {
                    self.queued = false;
                    return Err(self
                        .service
                        .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id));
                },
            };
        }
        if self.service.admission_is_closed(&state) {
            let removed =
                self.service.remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
            self.queued = false;
            self.service.inner.notify_admission_available();
            if removed.is_some() {
                record_admission_lifecycle(
                    &self.observation,
                    self.admission_request_id,
                    &self.tenant_id,
                    AdmissionLifecycleState::Rejected { reason: "admission_closed".to_string() },
                    state.queued.len(),
                    None,
                    None,
                );
            }
            let error = service_closed_error(&self.tenant_id);
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }
        let request_is_admissible = match self.service.request_is_admissible(&mut state, self.admission_request_id) {
            Ok(admissible) => admissible,
            Err(error) => {
                let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                self.queued = false;
                if removed.is_some() {
                    record_admission_lifecycle(
                        &self.observation,
                        self.admission_request_id,
                        &self.tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            },
        };
        if request_is_admissible
            && let Some(position) =
                state.queued.iter().position(|queued| queued.admission_request_id == self.admission_request_id)
        {
            let placed_query = match self.service.try_place_query(
                &self.tenant_id,
                self.admission_request_id,
                self.grant_policy,
                #[cfg(feature = "cugraph")]
                state.queued[position].warm_graph_grant_policy,
                state.queued[position].qos,
                &state.queued[position].placement_hint,
                Some(self.deadline),
                state.active_queries,
                &mut slot_notifications,
            ) {
                Ok(placed_query) => placed_query,
                Err(error) => {
                    let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            &self.tenant_id,
                            admission_rejection(&error),
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                },
            };
            let Some(placed_query) = placed_query else {
                if let Err(error) = self
                    .service
                    .request_adaptive_grant_shrink_for_blocked_request_locked(&state, &state.queued[position])
                {
                    let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            &self.tenant_id,
                            admission_rejection(&error),
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                }
                if Instant::now() >= self.deadline {
                    let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
                    self.queued = false;
                    if removed.is_some() {
                        record_admission_lifecycle(
                            &self.observation,
                            self.admission_request_id,
                            &self.tenant_id,
                            AdmissionLifecycleState::TimedOut,
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    let error = self.service.memory_grant_wait_timeout_error(
                        &self.tenant_id,
                        self.grant_policy,
                        state.queued.len(),
                    );
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                }
                return Ok(None);
            };
            let admitted = state.queued.remove(position).expect("admissible request exists");
            if let Err(error) = self.service.attach_admission_release_wakeup(&placed_query.memory_grant) {
                record_admission_lifecycle(
                    &admitted.observation,
                    self.admission_request_id,
                    &self.tenant_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.queued = false;
                drop(state);
                drop(placed_query);
                finish_removed_request_error(Some(admitted), &error);
                return Err(error);
            }
            if let Err(error) = self.service.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant) {
                record_admission_lifecycle(
                    &admitted.observation,
                    self.admission_request_id,
                    &self.tenant_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.queued = false;
                drop(state);
                drop(placed_query);
                finish_removed_request_error(Some(admitted), &error);
                return Err(error);
            }
            if self.service.admission_is_closed(&state) {
                let error = service_closed_error(&self.tenant_id);
                record_admission_lifecycle(
                    &admitted.observation,
                    self.admission_request_id,
                    &self.tenant_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.queued = false;
                self.service.inner.notify_admission_available();
                drop(state);
                drop(placed_query);
                finish_removed_request_error(Some(admitted), &error);
                return Err(error);
            }
            state.active_queries = state.active_queries.saturating_add(1);
            self.service.register_active_adaptive_grant(
                &mut state,
                self.admission_request_id,
                placed_query.device_ordinal,
                placed_query.grant_policy,
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
            record_admission_lifecycle(
                &self.observation,
                self.admission_request_id,
                &self.tenant_id,
                AdmissionLifecycleState::Admitted,
                state.queued.len(),
                Some(device_ordinal),
                Some(device_budget_bytes),
            );
            let (completion, gpu_resource_permit) = admitted.completion.into_admitted();
            return Ok(Some(QueryHandle {
                lifetime: QueryHandleLifetime::new(
                    Arc::clone(&service),
                    self.tenant_id.clone(),
                    self.admission_request_id,
                    admitted.observation,
                    completion,
                    gpu_resource_permit,
                    placed_query.memory_grant,
                    placed_query.device_slot,
                    #[cfg(feature = "cugraph")]
                    placed_query.warm_graph_reservation,
                ),
                config: service.config.clone(),
                tenant_id: self.tenant_id.clone(),
                admission_request_id: self.admission_request_id,
                admission_sequence,
                admission_wait: self.requested_at.elapsed(),
                device_budget_bytes,
                device_ordinal,
            }));
        }

        if Instant::now() >= self.deadline {
            let removed = self.service.abandon_queued_request_locked(&mut state, self.admission_request_id);
            self.queued = false;
            if removed.is_some() {
                record_admission_lifecycle(
                    &self.observation,
                    self.admission_request_id,
                    &self.tenant_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
            let error = service_overloaded_error(
                "admission_wait_timeout",
                &self.tenant_id,
                state.queued.len(),
                self.service.inner.config.max_admission_queue_len,
            );
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
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
        let removed = match self.service.inner.admission.lock() {
            Ok(mut state) => {
                let removed =
                    self.service.remove_queued_request_without_admission_locked(&mut state, self.admission_request_id);
                self.service.inner.notify_admission_available();
                if removed.is_some() {
                    record_admission_lifecycle(
                        &self.observation,
                        self.admission_request_id,
                        &self.tenant_id,
                        AdmissionLifecycleState::CancellationRequested,
                        state.queued.len(),
                        None,
                        None,
                    );
                    record_admission_lifecycle(
                        &self.observation,
                        self.admission_request_id,
                        &self.tenant_id,
                        AdmissionLifecycleState::Cancelled,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                removed
            },
            Err(error) => {
                self.queued = false;
                let error = self
                    .service
                    .recover_queued_request_after_admission_poison(error.into_inner(), self.admission_request_id);
                eprintln!(
                    "[nexus-query-engine] warning: recovered queued query service admission request after {error}"
                );
                return;
            },
        };
        finish_removed_request_cancelled(removed);
    }
}

pub(super) fn remove_queued_request(
    state: &mut AdmissionState,
    admission_request_id: AdmissionRequestId,
) -> Option<AdmissionRequest> {
    if let Some(position) = state.queued.iter().position(|request| request.admission_request_id == admission_request_id)
    {
        state.queued.remove(position)
    } else {
        None
    }
}

pub(super) fn finish_removed_request_error(request: Option<AdmissionRequest>, error: &QueryError) {
    if let Some(request) = request {
        request.completion.finish_error(error);
    }
}

pub(super) fn finish_removed_request_cancelled(request: Option<AdmissionRequest>) {
    if let Some(request) = request {
        request.completion.finish_cancelled();
    }
}

pub(super) fn finish_expired_queued_requests(expired: Vec<AdmissionRequest>) {
    for request in expired {
        let error = service_overloaded_error("admission_wait_timeout", &request.tenant_id, 0, 0);
        request.completion.finish_timeout(&error);
    }
}

pub(super) fn finish_closed_queued_requests(closed: Vec<AdmissionRequest>) {
    for request in closed {
        let error = service_closed_error(&request.tenant_id);
        record_admission_lifecycle(
            &request.observation,
            request.admission_request_id,
            &request.tenant_id,
            AdmissionLifecycleState::Rejected { reason: "admission_closed".to_string() },
            0,
            None,
            None,
        );
        finish_removed_request_error(Some(request), &error);
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
