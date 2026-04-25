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
        let tenant_id = tenant_id.into();
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        observation_owner.reject_activity_error(admission_request_id, &tenant_id)?;
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
                return Err(error);
            },
        };
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let request = AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            observation_owner: observation_owner.owner.clone(),
            observation_activity: observation_owner.activity.take(),
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
            return Err(error);
        }
        self.sweep_expired_queued_requests_locked(&mut state);
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
                return Err(service_closed_error(&tenant_id));
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
                        return Err(error);
                    }
                    if slot_notifications.has_pending() {
                        drop(state);
                        slot_notifications.notify();
                        state = self.lock_admission()?;
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
                        return Err(self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len()));
                    }
                    let wait = deadline.saturating_duration_since(now);
                    let (next_state, timeout) =
                        self.inner.admission_available.wait_timeout(state, wait).map_err(|_error| {
                            QueryError::new(
                                QueryErrorCode::Internal,
                                "nexus_query_engine.query_service.admission",
                                "query service admission mutex was poisoned while waiting",
                            )
                        })?;
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
                        return Err(self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len()));
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
                return Ok(QueryHandle {
                    lifetime: QueryHandleLifetime::new(
                        Arc::clone(&service),
                        tenant_id.clone(),
                        admission_request_id,
                        admitted.observation,
                        admitted.observation_owner,
                        admitted.observation_activity,
                        placed_query.memory_grant,
                        placed_query.device_slot,
                        #[cfg(feature = "cugraph")]
                        placed_query.warm_graph_reservation,
                    ),
                    service,
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
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
            }
            let wait = deadline.saturating_duration_since(now);
            let (next_state, timeout) = self.inner.admission_available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission",
                    "query service admission mutex was poisoned while waiting",
                )
            })?;
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
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
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
        let tenant_id = tenant_id.into();
        let mut waiter = self.enqueue_admission_with_request(tenant_id, request)?;
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
        let tenant_id = tenant_id.into();
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        observation_owner.reject_activity_error(admission_request_id, &tenant_id)?;
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
            return Err(error);
        }
        self.sweep_expired_queued_requests_locked(&mut state);
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
            return Err(error);
        }
        state.queued.push_back(AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            observation_owner: observation_owner.owner.clone(),
            observation_activity: observation_owner.activity.take(),
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
        let tenant_id = tenant_id.into();
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        observation_owner.reject_activity_error(admission_request_id, &tenant_id)?;
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
            return Err(error);
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.active_queries >= self.inner.config.max_concurrent_queries || !state.queued.is_empty() {
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
                return Err(error);
            },
        };
        let Some(placed_query) = placed_query else {
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
        Ok(Some(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&service),
                tenant_id.clone(),
                admission_request_id,
                observation_owner.handle,
                observation_owner.owner,
                observation_owner.activity,
                placed_query.memory_grant,
                placed_query.device_slot,
                #[cfg(feature = "cugraph")]
                placed_query.warm_graph_reservation,
            ),
            service,
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
        let close_result = match self.inner.admission.try_lock() {
            Ok(mut state) => {
                state.closed = true;
                Ok(())
            },
            Err(std::sync::TryLockError::Poisoned(_)) => Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission",
                "query service admission mutex was poisoned",
            )),
            Err(std::sync::TryLockError::WouldBlock) => Ok(()),
        };
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result =
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools);
        self.inner.notify_admission_available();
        close_result.and(stream_pool_close_result)
    }

    /// Closes admission and waits for every admitted handle and queued waiter
    /// to leave the service.
    ///
    /// This is the synchronous teardown boundary used when an async server
    /// future is cancelled: returning guarantees no query cleanup still owns
    /// service runtime resources or the process-wide governor claim.
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
        let operation = "nexus_query_engine.query_service.close_and_wait_for_quiescence";
        self.inner.admission_open.store(false, Ordering::Release);
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result =
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools);
        self.inner.notify_admission_available();
        // Local-Parquet cache checkouts may outlive their query handle. Drain
        // cache ownership before waiting for those query-scoped overlays so a
        // pinned escaped output cannot form a shutdown ordering cycle.
        let local_cache_result = self
            .inner
            .resources
            .as_ref()
            .map_or(Ok(()), |resources| resources.close_and_drain_local_parquet_source_caches_until(deadline));
        let mut state = self.lock_admission_until(deadline, operation)?;
        state.closed = true;
        let mut waited_for_quiescence = false;
        loop {
            if state.active_queries == 0 && state.cleanup_in_progress == 0 && state.queued.is_empty() {
                if waited_for_quiescence && crate::runtime::deadline::deadline_expired(deadline) {
                    return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
                }
                break;
            }
            if crate::runtime::deadline::deadline_expired(deadline) {
                return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
            }
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
            };
            waited_for_quiescence = true;
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_add(1, Ordering::AcqRel);
            let waited = self.inner.admission_available.wait_timeout(state, remaining);
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_sub(1, Ordering::AcqRel);
            let (next, timeout) = waited.map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission condition variable was poisoned during shutdown",
                )
            })?;
            state = next;
            if timeout.timed_out() {
                return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
            }
        }
        drop(state);
        #[cfg(feature = "cugraph")]
        let resident_cache_result = self
            .inner
            .resources
            .as_ref()
            .map_or(Ok(()), |resources| resources.close_and_drain_resident_graph_caches_until(deadline));
        let stream_pool_drain_result = self
            .inner
            .resources
            .as_ref()
            .map_or(Ok(()), |resources| resources.close_and_drain_stream_pools_until(deadline));
        let epoch_result = self.inner.service_epoch.wait_for_idle_until(deadline, operation);
        stream_pool_close_result?;
        local_cache_result?;
        #[cfg(feature = "cugraph")]
        resident_cache_result?;
        stream_pool_drain_result?;
        epoch_result?;
        if crate::runtime::deadline::deadline_expired(deadline) {
            return Err(query_service_shutdown_timeout(operation, None, "final_completion"));
        }
        Ok(())
    }
}
