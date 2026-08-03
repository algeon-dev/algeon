//! Blocking/async admission entrypoints and service close paths.

use super::*;

// `ExecutionCancellation` is intentionally a lightweight shared flag. The
// blocking queue has no caller-owned wake handle, so bounded waits observe it
// at this small fixed interval instead of retaining a stale provisional grant
// until the full queue deadline.
const ADMISSION_CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);

impl QueryService {
    pub fn admit_query(&self) -> Result<QueryHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::new())
    }

    /// Admits a query with consolidated request-scoped admission metadata.
    pub fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        self.admit_query_with_request_and_permits(request, None, None, None)
    }

    pub(crate) fn admit_query_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.admit_query_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
    }

    fn admit_query_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryHandle> {
        let queue_deadline = request.queue_deadline;
        let execution_cancellation = request.execution_cancellation.clone();
        let attempt_requirement = request.resolved_attempt_requirement();
        let mut observation_owner = request.into_runtime_observation();
        let attempt_id = observation_owner.attempt_id;
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_owner.attempt_finalizer_slot.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_error();
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                0,
                None,
                None,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                0,
                None,
                None,
            );
            completion.finish_cancelled();
            return Err(error);
        }
        let requirement = match self.resolve_attempt_requirement_for_submission(attempt_requirement) {
            Ok(requirement) => requirement,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
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
        let configured_deadline = requested_at.checked_add(self.inner.queue_bounds.max_wait).unwrap_or(requested_at);
        let deadline = queue_deadline.map_or(configured_deadline, |deadline| deadline.min(configured_deadline));
        let required_capabilities = super::placement::required_capabilities_for_placement(&requirement);
        let compatible_devices = self.compatible_devices_for_capabilities(&required_capabilities)?;
        if let Err(error) =
            self.reject_impossible_requirement(&requirement, &required_capabilities, &compatible_devices)
        {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let request = AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            attempt_id,
            completion,
            requirement: requirement.clone(),
            submission_seq: self.inner.next_submission_seq.fetch_add(1, Ordering::Relaxed),
            compatible_devices,
            overtakes_remaining: self.inner.queue_bounds.max_overtakes_per_waiter,
            protected_device: None,
            deadline,
            execution_cancellation,
            terminal: QueuedAdmissionTerminal::default(),
        };
        let terminal = request.terminal.clone();

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        let sweep = self.sweep_expired_queued_requests_locked(&mut state);
        if !sweep.is_empty() {
            drop(state);
            sweep.finish();
            state = self.lock_admission()?;
        }
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            finish_removed_request_error(Some(request), &error);
            return Err(error);
        }
        if state.queued.len() >= self.inner.queue_bounds.max_waiters {
            let error = service_overloaded_error(
                "admission_queue_full",
                state.queued.len(),
                self.inner.queue_bounds.max_waiters,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
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
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );

        loop {
            if let Some(error) = terminal.take() {
                return Err(error);
            }
            let sweep = self.sweep_expired_queued_requests_locked(&mut state);
            if !sweep.is_empty() {
                drop(state);
                sweep.finish();
                state = match self.inner.admission.lock() {
                    Ok(state) => state,
                    Err(error) => {
                        return Err(self
                            .recover_queued_request_after_admission_poison(error.into_inner(), admission_request_id));
                    },
                };
                continue;
            }
            if let Some(error) = self.admission_terminal_rejection(&state) {
                let removed = self.remove_queued_request_without_admission_locked(&mut state, admission_request_id);
                self.inner.notify_admission_available();
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
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
            let selected = match self.select_next_runnable_locked(&mut state) {
                Ok(selected) => match selected
                    .filter(|(position, _)| state.queued[*position].admission_request_id == admission_request_id)
                {
                    Some(selected) => Ok(Some(selected)),
                    None => self.fallback_driver_selection_locked(&state, admission_request_id),
                },
                Err(error) => Err(error),
            };
            let selected = match selected {
                Ok(selected) => selected,
                Err(error) => {
                    let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                    if removed.is_some() {
                        record_admission_lifecycle(
                            observation_owner.as_ref(),
                            admission_request_id,
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
            if let Some((position, device_ordinal)) = selected {
                let placed_query = match self.try_place_on_device(
                    device_ordinal,
                    admission_request_id,
                    &state.queued[position].requirement,
                    &mut slot_notifications,
                ) {
                    Ok(placed_query) => placed_query,
                    Err(error) => {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
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
                                AdmissionLifecycleState::TimedOut,
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        let error = self.memory_grant_wait_timeout_error(&requirement, state.queued.len());
                        drop(state);
                        finish_removed_request_error(removed, &error);
                        return Err(error);
                    }
                    let wait = deadline.saturating_duration_since(now).min(ADMISSION_CANCELLATION_POLL_INTERVAL);
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
                    if timeout.timed_out() && Instant::now() >= deadline {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                AdmissionLifecycleState::TimedOut,
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        let error = self.memory_grant_wait_timeout_error(&requirement, state.queued.len());
                        drop(state);
                        finish_removed_request_error(removed, &error);
                        return Err(error);
                    }
                    continue;
                };
                let admitted = state.queued.remove(position).expect("admissible request exists");
                let admitted_submission_seq = admitted.submission_seq;
                if let Some(error) = self.admission_terminal_rejection(&state) {
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
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
                let lifecycle_ticket = state.lifecycle.begin_provisional();
                self.inner.notify_admission_available();
                drop(state);
                return self.commit_placed_admission(
                    placed_query,
                    admitted.requirement.clone(),
                    admitted.compatible_devices.clone(),
                    admitted.observation,
                    admitted.attempt_id,
                    admitted.completion,
                    admission_request_id,
                    requested_at.elapsed(),
                    deadline,
                    admitted.execution_cancellation,
                    Some(admitted_submission_seq),
                    lifecycle_ticket,
                );
            }

            let now = Instant::now();
            if now >= deadline {
                let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_overloaded_error(
                    "queue_wait_elapsed",
                    state.queued.len(),
                    self.inner.queue_bounds.max_waiters,
                );
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
            let wait = deadline.saturating_duration_since(now).min(ADMISSION_CANCELLATION_POLL_INTERVAL);
            let (next_state, timeout) = match self.inner.admission_available.wait_timeout(state, wait) {
                Ok(waited) => waited,
                Err(error) => {
                    return Err(
                        self.recover_queued_request_after_admission_poison(error.into_inner().0, admission_request_id)
                    );
                },
            };
            state = next_state;
            if timeout.timed_out() && Instant::now() >= deadline {
                let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_overloaded_error(
                    "queue_wait_elapsed",
                    state.queued.len(),
                    self.inner.queue_bounds.max_waiters,
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
    pub fn enqueue_admission(&self) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request(QueryAdmissionRequest::new())
    }

    /// Admits a query asynchronously without parking a blocking thread.
    ///
    /// This is the async counterpart to [`Self::admit_query`]. It uses the
    /// same queue, timeout, and overload error taxonomy, but waits on an
    /// event-driven wakeup instead of polling
    /// [`QueryAdmissionWaiter::try_admit`].
    pub async fn admit_async(&self) -> Result<QueryHandle> {
        self.admit_async_with_request(QueryAdmissionRequest::new()).await
    }

    /// Admits a query asynchronously with consolidated request-scoped admission metadata.
    pub async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(request, None, None, None).await
    }

    pub(crate) async fn admit_async_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
        .await
    }

    async fn admit_async_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryHandle> {
        let mut waiter = self.enqueue_admission_with_request_and_permits(
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
            let _ = time::timeout(remaining_wait.min(ADMISSION_CANCELLATION_POLL_INTERVAL), &mut notified).await;
        }
    }

    /// Enqueues a query admission request with consolidated request-scoped metadata.
    pub fn enqueue_admission_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request_and_permits(request, None, None, None)
    }

    fn enqueue_admission_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryAdmissionWaiter> {
        let queue_deadline = request.queue_deadline;
        let execution_cancellation = request.execution_cancellation.clone();
        let attempt_requirement = request.resolved_attempt_requirement();
        let mut observation_owner = request.into_runtime_observation();
        let attempt_id = observation_owner.attempt_id;
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_owner.attempt_finalizer_slot.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_error();
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                0,
                None,
                None,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                0,
                None,
                None,
            );
            completion.finish_cancelled();
            return Err(error);
        }
        let requirement = match self.resolve_attempt_requirement_for_submission(attempt_requirement) {
            Ok(requirement) => requirement,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
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
        let configured_deadline = requested_at.checked_add(self.inner.queue_bounds.max_wait).unwrap_or(requested_at);
        let deadline = queue_deadline.map_or(configured_deadline, |deadline| deadline.min(configured_deadline));
        let required_capabilities = super::placement::required_capabilities_for_placement(&requirement);
        let compatible_devices = self.compatible_devices_for_capabilities(&required_capabilities)?;
        if let Err(error) =
            self.reject_impossible_requirement(&requirement, &required_capabilities, &compatible_devices)
        {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let mut state = self.lock_admission()?;
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        let sweep = self.sweep_expired_queued_requests_locked(&mut state);
        if !sweep.is_empty() {
            drop(state);
            sweep.finish();
            state = self.lock_admission()?;
        }
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        if state.queued.len() >= self.inner.queue_bounds.max_waiters {
            let error = service_overloaded_error(
                "admission_queue_full",
                state.queued.len(),
                self.inner.queue_bounds.max_waiters,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
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
            attempt_id,
            completion,
            requirement: requirement.clone(),
            submission_seq: self.inner.next_submission_seq.fetch_add(1, Ordering::Relaxed),
            compatible_devices,
            overtakes_remaining: self.inner.queue_bounds.max_overtakes_per_waiter,
            protected_device: None,
            deadline,
            execution_cancellation: execution_cancellation.clone(),
            terminal: QueuedAdmissionTerminal::default(),
        });
        let terminal = state.queued.back().expect("queued admission request exists").terminal.clone();
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );
        self.inner.notify_admission_available();
        Ok(QueryAdmissionWaiter {
            service: self.clone(),
            admission_request_id,
            observation: observation_owner.handle,
            requested_at,
            deadline,
            requirement: requirement.clone(),
            execution_cancellation,
            terminal,
            queued: true,
        })
    }

    /// Shared admission-commit tail (P2.6). The caller has selected and
    /// placed the request, removed it from the queue, and provisionally
    /// counted it as admitting; the admission mutex is not held. Builds the
    /// admission envelope — lane, root stream, allocation domain, root
    /// member, worker scope — while the mutex is released, then publishes
    /// admission. Any failure restores the provisional count exactly, releases
    /// every envelope component in reverse order outside the mutex, and
    /// finishes the request with the primary error.
    pub(in crate::runtime::query_service) fn commit_placed_admission(
        &self,
        placed_query: PlacedQuery,
        admitted_capability_evidence: AttemptRequirement,
        admitted_compatible_devices: Vec<usize>,
        observation: RuntimeObservationHandle,
        attempt_id: Option<crate::observability::AttemptId>,
        completion: QueryCompletion,
        admission_request_id: AdmissionRequestId,
        admission_wait: Duration,
        deadline: Instant,
        execution_cancellation: ExecutionCancellation,
        admitted_submission_seq: Option<u64>,
        lifecycle_ticket: super::super::admission::AdmissionTicket,
    ) -> Result<QueryHandle> {
        let probe =
            self.inner.admission_transaction_probe.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let envelope = super::transaction::AdmissionEnvelope::acquire(
            &self.inner,
            placed_query,
            lifecycle_ticket,
            &execution_cancellation,
            deadline,
            &observation,
            probe.as_deref(),
        );
        let mut state = self.inner.admission.lock().unwrap_or_else(|error| {
            eprintln!("[nexus-query-engine] warning: recovering poisoned admission state during admission commit");
            error.into_inner()
        });
        let mut envelope = match envelope {
            Ok(envelope) => envelope,
            Err(primary) => {
                if primary.code() == QueryErrorCode::Cancelled {
                    record_admission_lifecycle(
                        &observation,
                        admission_request_id,
                        AdmissionLifecycleState::CancellationRequested,
                        state.queued.len(),
                        None,
                        None,
                    );
                    record_admission_lifecycle(
                        &observation,
                        admission_request_id,
                        AdmissionLifecycleState::Cancelled,
                        state.queued.len(),
                        None,
                        None,
                    );
                } else if primary.fact_value("reason") == Some("queue_wait_elapsed") {
                    record_admission_lifecycle(
                        &observation,
                        admission_request_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                } else {
                    record_admission_lifecycle(
                        &observation,
                        admission_request_id,
                        admission_rejection(&primary),
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                self.inner.notify_admission_available();
                drop(state);
                if primary.code() == QueryErrorCode::Cancelled {
                    completion.finish_cancelled();
                } else if primary.fact_value("reason") == Some("queue_wait_elapsed") {
                    completion.finish_timeout(&primary);
                } else {
                    completion.finish_error(&primary);
                }
                return Err(primary);
            },
        };
        if let Some(attempt_id) = attempt_id {
            envelope.register_allocator_domain_attempt(attempt_id);
        }
        if let Err(error) =
            execution_cancellation.check("nexus_query_engine.query_service.admission_transaction.publish")
        {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                state.queued.len(),
                None,
                None,
            );
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            super::transaction::pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_cancelled();
            return Err(error);
        }
        if let Some(error) = self.admission_terminal_rejection(&state) {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            // Reverse-order component release happens outside the admission
            // mutex: the grant drop invokes the ledger release hook.
            super::transaction::pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        if let Some(error) = super::transaction::publication_deadline_error(deadline, probe.as_deref()) {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::TimedOut,
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            super::transaction::pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_timeout(&error);
            return Err(error);
        }
        if let Some(admitted_submission_seq) = admitted_submission_seq
            && let Err(error) =
                self.record_overtake_locked(&mut state, admitted_submission_seq, envelope.device_ordinal(), deadline)
        {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                if error.fact_value("reason") == Some("queue_wait_elapsed") {
                    AdmissionLifecycleState::TimedOut
                } else {
                    admission_rejection(&error)
                },
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            // A failed overtake commit is still a failed admission publish;
            // release the complete provisional envelope outside the mutex.
            super::transaction::pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            if error.fact_value("reason") == Some("queue_wait_elapsed") {
                completion.finish_timeout(&error);
            } else {
                completion.finish_error(&error);
            }
            return Err(error);
        }
        if let Err(error) = envelope.activate_lifecycle_ticket() {
            envelope.preaccount_rollback_cleanup(&mut state);
            self.inner.notify_admission_available();
            drop(state);
            super::transaction::pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        let completion = match completion.into_admitted(&mut envelope) {
            Ok(completion) => completion,
            Err(error) => {
                // A duplicate binding cannot manufacture a second post-cleanup
                // finalizer. The provisional envelope must roll back and the
                // fleet closes before its grant can wake another admission.
                envelope.preaccount_rollback_cleanup(&mut state);
                record_admission_lifecycle(
                    &observation,
                    admission_request_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.inner.notify_admission_available();
                drop(state);
                self.inner.close_unknown_isolation_admission("attempt finalizer unavailable during admission publish");
                super::transaction::pause_after_rollback_accounting(probe.as_deref());
                drop(envelope);
                self.inner.notify_admission_available();
                return Err(error);
            },
        };
        let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let device_ordinal = envelope.device_ordinal();
        let device_budget_bytes = envelope.memory_grant().budget_bytes();
        record_admission_lifecycle(
            &observation,
            admission_request_id,
            AdmissionLifecycleState::Admitted,
            state.queued.len(),
            Some(device_ordinal),
            Some(device_budget_bytes),
        );
        drop(state);
        Ok(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&self.inner),
                admission_request_id,
                observation,
                completion,
                envelope,
                admitted_capability_evidence.clone(),
                admitted_compatible_devices,
            ),
            config: self.inner.config.clone(),
            queue_bounds: self.inner.queue_bounds.clone(),
            admitted_capability_evidence,
            admission_request_id,
            admission_sequence,
            admission_wait,
            device_budget_bytes,
            device_ordinal,
        })
    }

    /// Attempts to admit a query without queueing or waiting.
    ///
    /// Returns `Ok(None)` when all admission slots are active or when queued
    /// requests already exist and this caller must not bypass them.
    pub fn try_admit_query(&self) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request(QueryAdmissionRequest::new())
    }

    /// Attempts to admit a query with consolidated request-scoped metadata.
    pub fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request_and_permits(request, None, None, None)
    }

    pub(crate) fn try_admit_query_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
    }

    fn try_admit_query_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<Option<QueryHandle>> {
        let queue_deadline = request.queue_deadline;
        let execution_cancellation = request.execution_cancellation.clone();
        let attempt_requirement = request.resolved_attempt_requirement();
        let mut observation_owner = request.into_runtime_observation();
        let attempt_id = observation_owner.attempt_id;
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_owner.attempt_finalizer_slot.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_error();
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                0,
                None,
                None,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                0,
                None,
                None,
            );
            completion.finish_cancelled();
            return Err(error);
        }
        let requirement = match self.resolve_attempt_requirement_for_submission(attempt_requirement) {
            Ok(requirement) => requirement,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
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
        let required_capabilities = super::placement::required_capabilities_for_placement(&requirement);
        let compatible_devices = self.compatible_devices_for_capabilities(&required_capabilities)?;
        if let Err(error) =
            self.reject_impossible_requirement(&requirement, &required_capabilities, &compatible_devices)
        {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let mut state = self.lock_admission()?;
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        let sweep = self.sweep_expired_queued_requests_locked(&mut state);
        if !sweep.is_empty() {
            drop(state);
            sweep.finish();
            state = self.lock_admission()?;
        }
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        if !state.queued.is_empty() {
            drop(state);
            return Ok(None);
        }
        let placed_query = match self.try_place_query(admission_request_id, &requirement, &mut slot_notifications) {
            Ok(placed_query) => placed_query,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
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
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
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
        let lifecycle_ticket = state.lifecycle.begin_provisional();
        self.inner.notify_admission_available();
        drop(state);
        // The try path never waits for queue capacity, but the commit
        // transaction may briefly wait for a root stream handoff from a
        // releasing attempt; that wait observes the attempt-wide deadline.
        let configured_deadline =
            Instant::now().checked_add(self.inner.queue_bounds.max_wait).unwrap_or_else(Instant::now);
        let deadline = queue_deadline.map_or(configured_deadline, |deadline| deadline.min(configured_deadline));
        self.commit_placed_admission(
            placed_query,
            requirement,
            compatible_devices,
            observation_owner.handle,
            attempt_id,
            completion,
            admission_request_id,
            Duration::ZERO,
            deadline,
            execution_cancellation,
            None,
            lifecycle_ticket,
        )
        .map(Some)
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
                for request in &closed_requests {
                    if let Err(error) = self.end_request_protection(request) {
                        eprintln!("[nexus-query-engine] warning: failed to end queue protection on close: {error}");
                    }
                }

                (Ok(()), closed_requests)
            },
            Err(std::sync::TryLockError::Poisoned(error)) => {
                let mut state = error.into_inner();
                state.closed = true;
                let closed_requests = std::mem::take(&mut state.queued).into_iter().collect::<Vec<_>>();
                for request in &closed_requests {
                    if let Err(error) = self.end_request_protection(request) {
                        eprintln!("[nexus-query-engine] warning: failed to end queue protection on close: {error}");
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
        let mut phases = Vec::with_capacity(8);
        phases.push(QueryServiceDrainPhaseOutcome::from_result(
            QueryServiceDrainPhase::StreamPoolClose,
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools),
        ));
        self.inner.notify_admission_available();
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
        for request in &closed_requests {
            if let Err(error) = self.end_request_protection(request) {
                eprintln!("[nexus-query-engine] warning: failed to end queue protection on close: {error}");
            }
        }
        if !closed_requests.is_empty() {
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
            if state.lifecycle.is_quiescent() && state.queued.is_empty() {
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

impl QueryService {
    /// Largest device managed capacity represented by this query service.
    pub fn max_device_capacity_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.inner.fallback_device_ledger.as_ref().map_or(0, |ledger| ledger.managed_capacity_bytes()),
            |resources| resources.devices.contexts().map(|context| context.capacity_bytes()).max().unwrap_or(0),
        )
    }

    /// Largest physical device-total byte count represented by this query service.
    ///
    /// Admission-only services have no CUDA inventory, so they fall back to the
    /// configured synthetic-ledger capacity used by deterministic CPU-only tests.
    pub fn max_device_total_memory_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.max_device_capacity_bytes(),
            |resources| resources.devices.contexts().map(|context| context.facts().total_bytes).max().unwrap_or(0),
        )
    }

    /// One atomic snapshot per device ledger, for placement, reports, and
    /// metrics. Admission-only services expose their synthetic ledger.
    pub(in crate::runtime::query_service) fn device_ledger_snapshots(
        &self,
    ) -> Result<Vec<crate::runtime::gpu::LedgerSnapshot>> {
        let Some(resources) = &self.inner.resources else {
            return self
                .inner
                .fallback_device_ledger
                .as_ref()
                .map(|ledger| ledger.snapshot())
                .transpose()
                .map(|snapshot| snapshot.into_iter().collect());
        };
        resources.devices.contexts().map(|context| context.ledger().snapshot()).collect()
    }
}
