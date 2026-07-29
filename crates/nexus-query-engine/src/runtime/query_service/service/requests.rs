//! Blocking/async admission entrypoints and service close paths.

use super::*;

impl QueryService {
    pub fn admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<QueryHandle> {
        self.admit_query_with_qos(tenant_id, QueryQos::default())
    }

    /// Admits a query with caller-supplied QoS metadata.
    ///
    /// Priority affects queued admission order only; it does not preempt active
    /// CUDA work or change memory-governor accounting.
    pub fn admit_query_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<QueryHandle> {
        self.admit_query_with_qos_and_requested_grant_bytes(tenant_id, qos, None)
    }

    /// Admits a query with caller-supplied QoS metadata and an optional explicit
    /// preferred grant.
    ///
    /// `requested_grant_bytes` overrides only the preferred grant for this
    /// request. It must be inside the tenant's configured `[min, want]` band and
    /// no larger than the service device capacity; invalid asks are rejected
    /// instead of silently clamped.
    pub fn admit_query_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<QueryHandle> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.admit_query_with_qos_and_grant_request(tenant_id, qos, grant_request)
    }

    /// Admits a query with caller-supplied QoS metadata and optional explicit
    /// minimum/preferred grant overrides.
    ///
    /// `requested_grant_min_bytes` raises the per-request floor inside the
    /// tenant's configured `[min, want]` band. Admission either grants at least
    /// that floor or queues the request; it never admits the request below the
    /// floor. `requested_grant_bytes` overrides only the preferred grant. Invalid
    /// request bands are rejected instead of silently clamped.
    pub fn admit_query_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryHandle> {
        self.admit_query_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
    }

    /// Admits a query with consolidated request-scoped admission metadata.
    pub fn admit_query_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<QueryHandle> {
        self.admit_query_with_request_and_permits(tenant_id.into(), request, None, None, None)
    }

    pub(crate) fn admit_query_with_backend_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.admit_query_with_request_and_permits(
            tenant_id,
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
    }

    fn admit_query_with_request_and_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryHandle> {
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let grant_policy = match self.tenant_grant_policy_for_request(&tenant_id, grant_request) {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_policy = match warm_graph_grant_request
            .map(|request| self.tenant_grant_policy_for_request(&tenant_id, request))
            .transpose()
        {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let request = AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            completion,
            tenant_id: tenant_id.clone(),
            grant_policy,
            #[cfg(feature = "cugraph")]
            warm_graph_grant_policy,
            qos,
            placement_hint,
            deadline,
        };

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        let expired = self.sweep_expired_queued_requests_locked(&mut state);
        if !expired.is_empty() {
            drop(state);
            finish_expired_queued_requests(expired);
            state = self.lock_admission()?;
        }
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            finish_removed_request_error(Some(request), &error);
            return Err(error);
        }
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            let error = service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            finish_removed_request_error(Some(request), &error);
            return Err(error);
        }
        state.queued.push_back(request);
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            &tenant_id,
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );

        loop {
            let expired = self.sweep_expired_queued_requests_locked(&mut state);
            if !expired.is_empty() {
                drop(state);
                finish_expired_queued_requests(expired);
                state = match self.inner.admission.lock() {
                    Ok(state) => state,
                    Err(error) => {
                        return Err(self
                            .recover_queued_request_after_admission_poison(error.into_inner(), admission_request_id));
                    },
                };
                continue;
            }
            if self.admission_is_closed(&state) {
                let removed = self.remove_queued_request_without_admission_locked(&mut state, admission_request_id);
                self.inner.notify_admission_available();
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        &tenant_id,
                        AdmissionLifecycleState::Rejected { reason: "admission_closed".to_string() },
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_closed_error(&tenant_id);
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
            let request_is_admissible = match self.request_is_admissible(&mut state, admission_request_id) {
                Ok(admissible) => admissible,
                Err(error) => {
                    let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                    if removed.is_some() {
                        record_admission_lifecycle(
                            observation_owner.as_ref(),
                            admission_request_id,
                            &tenant_id,
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
                    state.queued.iter().position(|queued| queued.admission_request_id == admission_request_id)
            {
                let placed_query = match self.try_place_query(
                    &tenant_id,
                    admission_request_id,
                    grant_policy,
                    #[cfg(feature = "cugraph")]
                    state.queued[position].warm_graph_grant_policy,
                    qos,
                    &state.queued[position].placement_hint,
                    Some(deadline),
                    state.active_queries,
                    &mut slot_notifications,
                ) {
                    Ok(placed_query) => placed_query,
                    Err(error) => {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
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
                    if let Err(error) =
                        self.request_adaptive_grant_shrink_for_blocked_request_locked(&state, &state.queued[position])
                    {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
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
                    if slot_notifications.has_pending() {
                        drop(state);
                        slot_notifications.notify();
                        state = match self.inner.admission.lock() {
                            Ok(state) => state,
                            Err(error) => {
                                return Err(self.recover_queued_request_after_admission_poison(
                                    error.into_inner(),
                                    admission_request_id,
                                ));
                            },
                        };
                        if self.admission_is_closed(&state) {
                            continue;
                        }
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
                                AdmissionLifecycleState::TimedOut,
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        let error = self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len());
                        drop(state);
                        finish_removed_request_error(removed, &error);
                        return Err(error);
                    }
                    let wait = deadline.saturating_duration_since(now);
                    let (next_state, timeout) = match self.inner.admission_available.wait_timeout(state, wait) {
                        Ok(waited) => waited,
                        Err(error) => {
                            return Err(self.recover_queued_request_after_admission_poison(
                                error.into_inner().0,
                                admission_request_id,
                            ));
                        },
                    };
                    state = next_state;
                    if timeout.timed_out() {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
                                AdmissionLifecycleState::TimedOut,
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        let error = self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len());
                        drop(state);
                        finish_removed_request_error(removed, &error);
                        return Err(error);
                    }
                    continue;
                };
                let admitted = state.queued.remove(position).expect("admissible request exists");
                if let Err(error) = self.attach_admission_release_wakeup(&placed_query.memory_grant) {
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
                        &tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                    drop(state);
                    drop(placed_query);
                    finish_removed_request_error(Some(admitted), &error);
                    return Err(error);
                }
                if let Err(error) = self.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant) {
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
                        &tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                    drop(state);
                    drop(placed_query);
                    finish_removed_request_error(Some(admitted), &error);
                    return Err(error);
                }
                if self.admission_is_closed(&state) {
                    let error = service_closed_error(&tenant_id);
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
                        &tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                    self.inner.notify_admission_available();
                    drop(state);
                    drop(placed_query);
                    finish_removed_request_error(Some(admitted), &error);
                    return Err(error);
                }
                state.active_queries = state.active_queries.saturating_add(1);
                self.register_active_adaptive_grant(
                    &mut state,
                    admission_request_id,
                    placed_query.device_ordinal,
                    placed_query.grant_policy,
                    &placed_query.memory_grant,
                );
                self.rotate_after_admission(&mut state, &admitted.tenant_id);
                self.refresh_active_adaptive_shrink_requests_locked(&state);
                if state.queued.is_empty() {
                    self.grow_active_adaptive_grants_locked(&mut state);
                }
                self.inner.notify_admission_available();
                let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
                let service = Arc::clone(&self.inner);
                let device_budget_bytes = placed_query.memory_grant.budget_bytes();
                let device_ordinal = placed_query.device_ordinal;
                record_admission_lifecycle(
                    &admitted.observation,
                    admission_request_id,
                    &tenant_id,
                    AdmissionLifecycleState::Admitted,
                    state.queued.len(),
                    Some(device_ordinal),
                    Some(device_budget_bytes),
                );
                let (completion, gpu_resource_permit) = admitted.completion.into_admitted();
                return Ok(QueryHandle {
                    lifetime: QueryHandleLifetime::new(
                        Arc::clone(&service),
                        tenant_id.clone(),
                        admission_request_id,
                        admitted.observation,
                        completion,
                        gpu_resource_permit,
                        placed_query.memory_grant,
                        placed_query.device_slot,
                        #[cfg(feature = "cugraph")]
                        placed_query.warm_graph_reservation,
                    ),
                    config: service.config.clone(),
                    tenant_id,
                    admission_request_id,
                    admission_sequence,
                    admission_wait: requested_at.elapsed(),
                    device_budget_bytes,
                    device_ordinal,
                });
            }

            let now = Instant::now();
            if now >= deadline {
                let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        &tenant_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                );
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
            let wait = deadline.saturating_duration_since(now);
            let (next_state, timeout) = match self.inner.admission_available.wait_timeout(state, wait) {
                Ok(waited) => waited,
                Err(error) => {
                    return Err(
                        self.recover_queued_request_after_admission_poison(error.into_inner().0, admission_request_id)
                    );
                },
            };
            state = next_state;
            if timeout.timed_out() {
                let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        &tenant_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                );
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
        }
    }

    /// Enqueues a query admission request without blocking the caller.
    ///
    /// Call [`QueryAdmissionWaiter::try_admit`] until it returns a handle. If
    /// the waiter is dropped first, the queued request is removed.
    pub fn enqueue_admission(&self, tenant_id: impl Into<TenantId>) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_qos(tenant_id, QueryQos::default())
    }

    /// Admits a query asynchronously without parking a blocking thread.
    ///
    /// This is the async counterpart to [`Self::admit_query`]. It uses the
    /// same queue, tenant fairness, timeout, and overload error taxonomy, but
    /// waits on an event-driven wakeup instead of polling
    /// [`QueryAdmissionWaiter::try_admit`].
    pub async fn admit_async(&self, tenant_id: impl Into<TenantId>) -> Result<QueryHandle> {
        self.admit_async_with_qos(tenant_id, QueryQos::default()).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata.
    pub async fn admit_async_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<QueryHandle> {
        self.admit_async_with_qos_and_requested_grant_bytes(tenant_id, qos, None).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata and an
    /// optional explicit preferred grant.
    pub async fn admit_async_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<QueryHandle> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.admit_async_with_qos_and_grant_request(tenant_id, qos, grant_request).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata and
    /// optional explicit minimum/preferred grant overrides.
    pub async fn admit_async_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryHandle> {
        self.admit_async_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
        .await
    }

    /// Admits a query asynchronously with consolidated request-scoped admission metadata.
    pub async fn admit_async_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(tenant_id.into(), request, None, None, None).await
    }

    pub(crate) async fn admit_async_with_backend_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(
            tenant_id,
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
        .await
    }

    async fn admit_async_with_request_and_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryHandle> {
        let mut waiter = self.enqueue_admission_with_request_and_permits(
            tenant_id,
            request,
            observation_permit,
            gpu_resource_permit,
            completion_dispatcher,
        )?;
        loop {
            let notified = self.inner.admission_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(handle) = waiter.try_admit()? {
                return Ok(handle);
            }
            let remaining_wait = waiter.remaining_wait();
            if remaining_wait.is_zero() {
                continue;
            }
            let _ = time::timeout(remaining_wait, &mut notified).await;
        }
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata.
    pub fn enqueue_admission_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_qos_and_requested_grant_bytes(tenant_id, qos, None)
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata and
    /// an optional explicit preferred grant.
    pub fn enqueue_admission_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<QueryAdmissionWaiter> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.enqueue_admission_with_qos_and_grant_request(tenant_id, qos, grant_request)
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata and
    /// optional explicit minimum/preferred grant overrides.
    pub fn enqueue_admission_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
    }

    /// Enqueues a query admission request with consolidated request-scoped metadata.
    pub fn enqueue_admission_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request_and_permits(tenant_id.into(), request, None, None, None)
    }

    fn enqueue_admission_with_request_and_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryAdmissionWaiter> {
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let grant_policy = match self.tenant_grant_policy_for_request(&tenant_id, grant_request) {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_policy = match warm_graph_grant_request
            .map(|request| self.tenant_grant_policy_for_request(&tenant_id, request))
            .transpose()
        {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let mut state = self.lock_admission()?;
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        let expired = self.sweep_expired_queued_requests_locked(&mut state);
        if !expired.is_empty() {
            drop(state);
            finish_expired_queued_requests(expired);
            state = self.lock_admission()?;
        }
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            let error = service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        state.queued.push_back(AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            completion,
            tenant_id: tenant_id.clone(),
            grant_policy,
            #[cfg(feature = "cugraph")]
            warm_graph_grant_policy,
            qos,
            placement_hint,
            deadline,
        });
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            &tenant_id,
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );
        self.inner.notify_admission_available();
        Ok(QueryAdmissionWaiter {
            service: self.clone(),
            tenant_id,
            admission_request_id,
            observation: observation_owner.handle,
            requested_at,
            deadline,
            grant_policy,
            queued: true,
        })
    }

    /// Attempts to admit a query without queueing or waiting.
    ///
    /// Returns `Ok(None)` when all admission slots are active or when queued
    /// requests already exist and this caller must not bypass them.
    pub fn try_admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_qos(tenant_id, QueryQos::default())
    }

    /// Attempts to admit a query with caller-supplied QoS metadata without
    /// queueing or waiting.
    pub fn try_admit_query_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_qos_and_requested_grant_bytes(tenant_id, qos, None)
    }

    /// Attempts to admit a query with caller-supplied QoS metadata and an
    /// optional explicit preferred grant without queueing or waiting.
    pub fn try_admit_query_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<Option<QueryHandle>> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.try_admit_query_with_qos_and_grant_request(tenant_id, qos, grant_request)
    }

    /// Attempts to admit a query with caller-supplied QoS metadata and optional
    /// explicit minimum/preferred grant overrides.
    pub fn try_admit_query_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
    }

    /// Attempts to admit a query with consolidated request-scoped metadata.
    pub fn try_admit_query_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request_and_permits(tenant_id.into(), request, None, None, None)
    }

    pub(crate) fn try_admit_query_with_backend_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request_and_permits(
            tenant_id,
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
    }

    fn try_admit_query_with_request_and_permits(
        &self,
        tenant_id: TenantId,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<Option<QueryHandle>> {
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let grant_policy = match self.tenant_grant_policy_for_request(&tenant_id, grant_request) {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_policy = match warm_graph_grant_request
            .map(|request| self.tenant_grant_policy_for_request(&tenant_id, request))
            .transpose()
        {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        let expired = self.sweep_expired_queued_requests_locked(&mut state);
        if !expired.is_empty() {
            drop(state);
            finish_expired_queued_requests(expired);
            state = self.lock_admission()?;
        }
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        if state.active_queries >= self.inner.config.max_concurrent_queries || !state.queued.is_empty() {
            drop(state);
            return Ok(None);
        }
        let placed_query = match self.try_place_query(
            &tenant_id,
            admission_request_id,
            grant_policy,
            #[cfg(feature = "cugraph")]
            warm_graph_grant_policy,
            qos,
            &placement_hint,
            None,
            state.active_queries,
            &mut slot_notifications,
        ) {
            Ok(placed_query) => placed_query,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                drop(state);
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let Some(placed_query) = placed_query else {
            drop(state);
            return Ok(None);
        };
        if let Err(error) = self.attach_admission_release_wakeup(&placed_query.memory_grant) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            drop(placed_query);
            completion.finish_error(&error);
            return Err(error);
        }
        if let Err(error) = self.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            drop(placed_query);
            completion.finish_error(&error);
            return Err(error);
        }
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            drop(placed_query);
            completion.finish_error(&error);
            return Err(error);
        }
        state.active_queries = state.active_queries.saturating_add(1);
        self.register_active_adaptive_grant(
            &mut state,
            admission_request_id,
            placed_query.device_ordinal,
            placed_query.grant_policy,
            &placed_query.memory_grant,
        );
        self.rotate_after_admission(&mut state, &tenant_id);
        let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let service = Arc::clone(&self.inner);
        let device_budget_bytes = placed_query.memory_grant.budget_bytes();
        let device_ordinal = placed_query.device_ordinal;
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            &tenant_id,
            AdmissionLifecycleState::Admitted,
            state.queued.len(),
            Some(device_ordinal),
            Some(device_budget_bytes),
        );
        let (completion, gpu_resource_permit) = completion.into_admitted();
        Ok(Some(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&service),
                tenant_id.clone(),
                admission_request_id,
                observation_owner.handle,
                completion,
                gpu_resource_permit,
                placed_query.memory_grant,
                placed_query.device_slot,
                #[cfg(feature = "cugraph")]
                placed_query.warm_graph_reservation,
            ),
            config: service.config.clone(),
            tenant_id,
            admission_request_id,
            admission_sequence,
            admission_wait: Duration::ZERO,
            device_budget_bytes,
            device_ordinal,
        }))
    }

    /// Closes admission and wakes every queued waiter.
    ///
    /// The atomic gate closes before the best-effort mutex update, so this
    /// signal never waits behind an admission transaction. Commit paths
    /// recheck the gate while holding the admission mutex. Active query
    /// handles keep their slots until dropped or finished.
    pub fn close(&self) -> Result<()> {
        self.inner.admission_open.store(false, Ordering::Release);
        let (close_result, closed_requests) = match self.inner.admission.try_lock() {
            Ok(mut state) => {
                state.closed = true;
                let closed_requests = std::mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();
                if !closed_requests.is_empty() {
                    self.refresh_active_adaptive_shrink_requests_locked(&state);
                    if state.queued.is_empty() {
                        self.grow_active_adaptive_grants_locked(&mut state);
                    }
                }
                (Ok(()), closed_requests)
            },
            Err(std::sync::TryLockError::Poisoned(error)) => {
                let mut state = error.into_inner();
                state.closed = true;
                let closed_requests = std::mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();
                if !closed_requests.is_empty() {
                    self.refresh_active_adaptive_shrink_requests_locked(&state);
                    if state.queued.is_empty() {
                        self.grow_active_adaptive_grants_locked(&mut state);
                    }
                }
                (
                    Err(QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.query_service.admission",
                        "query service admission mutex was poisoned",
                    )
                    .with_fact("reason", "admission_mutex_poisoned")),
                    closed_requests,
                )
            },
            Err(std::sync::TryLockError::WouldBlock) => (Ok(()), Vec::new()),
        };
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result =
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools);
        self.inner.notify_admission_available();
        finish_closed_queued_requests(closed_requests);
        close_result.and(stream_pool_close_result)
    }

    /// Closes admission and waits for every admitted handle and queued waiter
    /// to leave the service.
    ///
    /// This is the synchronous teardown boundary used when an async server
    /// future is cancelled: returning guarantees no query cleanup still owns
    /// service runtime resources, so the owning resource aggregate can be
    /// destroyed. The process-wide backend owner remains attached to that
    /// aggregate until its resources are dropped.
    pub fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        self.close_and_wait_for_quiescence_until(deadline)
    }

    /// Closes admission and waits for complete cleanup under one caller-owned
    /// absolute deadline.
    ///
    /// Time spent acquiring the admission lock or closing the service consumes
    /// this same deadline instead of starting a second grace period.
    pub fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.close_drain_step_until(deadline).into_result()
    }

    pub(crate) fn close_drain_step_until(&self, deadline: Instant) -> QueryServiceDrainOutcome {
        let operation = "nexus_query_engine.query_service.close_and_wait_for_quiescence";
        self.inner.admission_open.store(false, Ordering::Release);
        self.inner.service_epoch.invalidate();
        let mut phases = Vec::with_capacity(9);
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolClose,
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools),
        ));
        self.inner.notify_admission_available();
        // Local-Parquet cache checkouts may outlive their query handle. Drain
        // cache ownership before waiting for those query-scoped overlays so a
        // pinned escaped output cannot form a shutdown ordering cycle.
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::LocalSourceCache,
            self.inner
                .resources
                .as_ref()
                .map_or(Ok(()), |resources| resources.close_and_drain_local_parquet_source_caches_until(deadline)),
        ));
        let admission_quiescence = self.wait_for_admission_quiescence_until(deadline, operation);
        let admission_is_quiescent = admission_quiescence.is_quiescent();
        phases.push(admission_quiescence);
        if !admission_is_quiescent {
            // Query-handle release schedules ordered CUDA destruction on the
            // device execution pool. A bounded wait may expire while an
            // outward context still owns the final query anchor, so closing
            // that pool here would make the later release impossible to run.
            // Report every dependent phase as pending and let the next drain
            // pass enter them after admission cleanup has finished.
            #[cfg(feature = "cugraph")]
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::ResidentGraphCache,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::ExecutionPoolQuiescence,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::StreamPoolQuiescence,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::ResourceEpoch,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            phases.push(dependent_drain_phase_pending(
                QueryServiceDrainPhase::FinalCompletion,
                QueryServiceDrainPhase::AdmissionQuiescence,
                operation,
            ));
            return QueryServiceDrainOutcome::new(phases);
        }
        #[cfg(feature = "cugraph")]
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::ResidentGraphCache,
            self.inner
                .resources
                .as_ref()
                .map_or(Ok(()), |resources| resources.close_and_drain_resident_graph_caches_until(deadline)),
        ));
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::ExecutionPoolQuiescence,
            self.inner
                .resources
                .as_ref()
                .map_or(Ok(()), |resources| resources.close_and_drain_gpu_execution_pools_until(deadline)),
        ));
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolQuiescence,
            self.inner
                .resources
                .as_ref()
                .map_or(Ok(()), |resources| resources.close_and_drain_stream_pools_until(deadline)),
        ));
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::ResourceEpoch,
            self.inner.service_epoch.wait_for_idle_until(deadline, operation),
        ));
        let blocked_by_phase = phases.iter().find(|outcome| !outcome.is_quiescent()).map(|outcome| outcome.phase());
        phases.push(match blocked_by_phase {
            Some(blocked_by_phase) => {
                dependent_drain_phase_pending(QueryServiceDrainPhase::FinalCompletion, blocked_by_phase, operation)
            },
            None if crate::runtime::deadline::deadline_expired(deadline) => {
                QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::FinalCompletion,
                    error: query_service_shutdown_timeout(operation, None, "final_completion"),
                }
            },
            None => QueryServiceDrainPhaseOutcome::Quiescent { phase: QueryServiceDrainPhase::FinalCompletion },
        });
        QueryServiceDrainOutcome::new(phases)
    }

    fn wait_for_admission_quiescence_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> QueryServiceDrainPhaseOutcome {
        let (mut state, admission_was_poisoned) =
            match crate::runtime::deadline::mutex_lock_until(&self.inner.admission, deadline) {
                Ok(state) => (state, false),
                Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => {
                    let state = match self.inner.admission.try_lock() {
                        Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
                        Ok(_) | Err(std::sync::TryLockError::WouldBlock) => {
                            return QueryServiceDrainPhaseOutcome::PermanentFailure {
                                phase: QueryServiceDrainPhase::AdmissionQuiescence,
                                error: QueryError::new(
                                    QueryErrorCode::Internal,
                                    operation,
                                    "query service admission mutex poison state changed during shutdown recovery",
                                )
                                .with_fact("reason", "admission_mutex_poison_recovery_raced"),
                            };
                        },
                    };
                    (state, true)
                },
                Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                    return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                        phase: QueryServiceDrainPhase::AdmissionQuiescence,
                        error: query_service_shutdown_timeout(operation, None, "admission_mutex"),
                    };
                },
            };
        state.closed = true;
        let closed_requests = std::mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();
        if !closed_requests.is_empty() {
            self.refresh_active_adaptive_shrink_requests_locked(&state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(&mut state);
            }
            drop(state);
            finish_closed_queued_requests(closed_requests);
            state = match self.lock_admission_until(deadline, operation) {
                Ok(state) => state,
                Err(error) => {
                    return QueryServiceDrainPhaseOutcome::from_result(
                        QueryServiceDrainPhase::AdmissionQuiescence,
                        Err(error),
                    );
                },
            };
        }
        if admission_was_poisoned {
            drop(state);
            return QueryServiceDrainPhaseOutcome::PermanentFailure {
                phase: QueryServiceDrainPhase::AdmissionQuiescence,
                error: QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission mutex was poisoned during shutdown",
                )
                .with_fact("reason", "admission_mutex_poisoned"),
            };
        }
        let mut waited_for_quiescence = false;
        loop {
            if state.active_queries == 0 && state.cleanup_in_progress == 0 && state.queued.is_empty() {
                if waited_for_quiescence && crate::runtime::deadline::deadline_expired(deadline) {
                    return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                        phase: QueryServiceDrainPhase::AdmissionQuiescence,
                        error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                    };
                }
                return QueryServiceDrainPhaseOutcome::Quiescent { phase: QueryServiceDrainPhase::AdmissionQuiescence };
            }
            if crate::runtime::deadline::deadline_expired(deadline) {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                };
            }
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                };
            };
            waited_for_quiescence = true;
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_add(1, Ordering::AcqRel);
            let waited = self.inner.admission_available.wait_timeout(state, remaining);
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_sub(1, Ordering::AcqRel);
            let (next, timeout) = match waited {
                Ok(waited) => waited,
                Err(_error) => {
                    return QueryServiceDrainPhaseOutcome::PermanentFailure {
                        phase: QueryServiceDrainPhase::AdmissionQuiescence,
                        error: QueryError::new(
                            QueryErrorCode::Internal,
                            operation,
                            "query service admission condition variable was poisoned during shutdown",
                        ),
                    };
                },
            };
            state = next;
            if timeout.timed_out() {
                return QueryServiceDrainPhaseOutcome::DeadlineElapsed {
                    phase: QueryServiceDrainPhase::AdmissionQuiescence,
                    error: query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"),
                };
            }
        }
    }
}

fn dependent_drain_phase_pending(
    phase: QueryServiceDrainPhase,
    blocked_by_phase: QueryServiceDrainPhase,
    operation: &'static str,
) -> QueryServiceDrainPhaseOutcome {
    QueryServiceDrainPhaseOutcome::DeadlineElapsed {
        phase,
        error: QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            operation,
            "query service cleanup phase is waiting for an earlier drain phase",
        )
        .with_fact("reason", "query_service_cleanup_dependency_pending")
        .with_fact("deadline_stage", phase.as_str())
        .with_fact("blocked_by_phase", blocked_by_phase.as_str())
        .with_cleanup_deadline_elapsed(),
    }
}
