//! Shared admission-submission preamble and placed-admission commit.

#[cfg(any(test, feature = "test-hooks"))]
use std::sync::Mutex;
use std::{
    collections::BTreeMap,
    sync::{Arc, MutexGuard, atomic::Ordering},
    time::Duration,
};

use query_planning::{
    admission::AttemptCapabilityEvidence,
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::{
        admission::{AdmissionState, AdmissionTicket, PlacedAdmission, PlacedQuery, QueryCompletion},
        admission_rejection,
        handle::{QueryHandle, QueryHandleLifetime},
        record_admission_lifecycle,
    },
    QueryService, QueryServiceInner,
};
#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::{deadline::block_until, gpu_execution_pool::GpuExecutionPool};
use crate::{
    attempt::AttemptLifecycleError,
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptId,
        RuntimeObservationHandle,
    },
    runtime::{
        ExecutionCancellation, GpuExecutionJobKind,
        gpu_execution_pool::GpuExecutionJobClass,
        query_service::{
            controller::admission_mutex_poisoned_error,
            service::transaction::{
                AdmissionEnvelope, AdmissionTransactionProbe, check_publication_cancellation,
                pause_after_rollback_accounting,
            },
        },
    },
};

mod enqueue;

/// What an admission envelope is built from.
struct EnvelopeRequest {
    placed_query: PlacedQuery,
    lifecycle_ticket: AdmissionTicket,
    execution_cancellation: ExecutionCancellation,
    observation: RuntimeObservationHandle,
}

impl EnvelopeRequest {
    /// Rolls the placed grant back without building an envelope and returns
    /// the error that stopped it.
    fn abandon(self, service: &Arc<QueryServiceInner>, error: QueryError) -> QueryError {
        AdmissionEnvelope::abandon(service, self.placed_query, self.lifecycle_ticket);
        error
    }

    fn acquire(
        self,
        service: &Arc<QueryServiceInner>,
        probe: Option<&AdmissionTransactionProbe>,
    ) -> Result<AdmissionEnvelope> {
        AdmissionEnvelope::acquire(
            service,
            self.placed_query,
            self.lifecycle_ticket,
            &self.execution_cancellation,
            &self.observation,
            probe,
        )
    }
}

struct AdmissionPublication {
    observation: RuntimeObservationHandle,
    attempt_id: AttemptId,
    completion: QueryCompletion,
    admitted_device_requirements: BTreeMap<usize, AttemptCapabilityEvidence>,
    submitted_plan_capability_evidence: Option<AttemptCapabilityEvidence>,
    admitted_compatible_devices: Vec<usize>,
    admission_request_id: AdmissionRequestId,
    admission_wait: Duration,
    execution_cancellation: ExecutionCancellation,
    admission_wait_fact: Option<AttemptAdmissionWaitFact>,
}

impl QueryService {
    /// Shared admission-commit tail. The caller has selected and placed the
    /// request, removed it from the queue, and provisionally counted it as
    /// admitting; the admission mutex is not held. Builds the admission
    /// envelope (lane, root stream, allocation domain, root member, worker
    /// scope) while the mutex is released, then publishes admission. Any
    /// failure restores the provisional count, releases every envelope
    /// component in reverse order outside the mutex, and finishes the request
    /// with the primary error.
    ///
    /// The envelope's root stream, allocation domain, and root member are
    /// built in a context-setup job on the placed device's GPU pool, so the
    /// caller's Tokio worker neither waits on the stream pool nor creates
    /// CUDA state. Production passes no probe.
    pub(in crate::runtime::query_service) async fn commit_placed_admission_async(
        &self,
        commit: PlacedAdmission,
        probe: Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryHandle> {
        let (request, publication) = self.begin_placed_commit(commit);
        let envelope = self.acquire_envelope_on_gpu_pool(request, probe.clone()).await;
        self.finish_placed_commit(envelope, publication, &probe)
    }

    /// [`Self::commit_placed_admission_async`] driven to completion on the
    /// calling thread. The envelope still builds on the placed device's GPU
    /// pool; entering that pool's runtime lets the commit's cancellation
    /// timers register while this thread blocks.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(in crate::runtime::query_service) fn commit_placed_admission(
        &self,
        commit: PlacedAdmission,
        probe: Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryHandle> {
        // A missing pool fails the commit itself, which abandons the placement.
        let pool = self.inner.resources.gpu_execution_pool_for_device(commit.placed_query.device_ordinal).ok();
        let _runtime = pool.as_ref().map(GpuExecutionPool::enter_runtime);
        block_until(self.commit_placed_admission_async(commit, probe), None)
            .expect("a commit wait without a deadline yields its result")
    }

    async fn acquire_envelope_on_gpu_pool(
        &self,
        request: EnvelopeRequest,
        probe: Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<AdmissionEnvelope> {
        let pool = match self.inner.resources.gpu_execution_pool_for_device(request.placed_query.device_ordinal) {
            Ok(pool) => pool,
            Err(error) => return Err(request.abandon(&self.inner, error)),
        };
        let kind = GpuExecutionJobKind::ContextSetup;
        let class_permit = match pool
            .wait_for_class_permit(kind, GpuExecutionJobClass::Lifecycle, &request.execution_cancellation)
            .await
        {
            Ok(class_permit) => class_permit,
            Err(error) => return Err(request.abandon(&self.inner, error)),
        };
        let service = Arc::clone(&self.inner);
        let observation = request.observation.clone();
        pool.run_observed_scoped_with_class_permit(kind, class_permit, observation, move |_scope| async move {
            request.acquire(&service, probe.as_deref())
        })
        .await
        .and_then(|envelope| envelope)
    }

    /// Splits a placed commit into its envelope inputs and its publication.
    fn begin_placed_commit(&self, commit: PlacedAdmission) -> (EnvelopeRequest, AdmissionPublication) {
        let PlacedAdmission {
            placed_query,
            admitted_device_requirements,
            submitted_plan_capability_evidence,
            admitted_compatible_devices,
            observation,
            attempt_id,
            completion,
            admission_request_id,
            admission_wait,
            execution_cancellation,
            admission_wait_fact,
            lifecycle_ticket,
        } = commit;
        let request = EnvelopeRequest {
            placed_query,
            lifecycle_ticket,
            execution_cancellation: execution_cancellation.clone(),
            observation: observation.clone(),
        };
        let publication = AdmissionPublication {
            observation,
            attempt_id,
            completion,
            admitted_device_requirements,
            submitted_plan_capability_evidence,
            admitted_compatible_devices,
            admission_request_id,
            admission_wait,
            execution_cancellation,
            admission_wait_fact,
        };
        (request, publication)
    }

    /// Publishes a built envelope, or finishes the request with the error
    /// that stopped the build.
    fn finish_placed_commit(
        &self,
        envelope: Result<AdmissionEnvelope>,
        publication: AdmissionPublication,
        probe: &Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryHandle> {
        // Registration reads the domain's allocator-event watermark through
        // cuDF and takes the correlation index lock, so it runs before the
        // admission mutex. A rejected envelope discards it when dropped.
        let envelope = envelope.map(|mut envelope| {
            envelope.register_allocator_domain_attempt(publication.attempt_id);
            envelope
        });
        // A poisoned admission state refuses publication: the envelope rolls
        // back through the same branch as a closed service.
        let (state, admission_poisoned) = match self.inner.admission.state.lock() {
            Ok(state) => (state, false),
            Err(error) => (error.into_inner(), true),
        };
        let envelope = match envelope {
            Ok(envelope) => envelope,
            Err(primary) => {
                return Err(self.finish_rejected_envelope(
                    state,
                    primary,
                    &publication.observation,
                    publication.admission_request_id,
                    publication.completion,
                ));
            },
        };
        self.publish_admission_envelope(state, admission_poisoned, envelope, publication, probe)
    }

    fn finish_rejected_envelope(
        &self,
        state: MutexGuard<'_, AdmissionState>,
        primary: QueryError,
        observation: &RuntimeObservationHandle,
        admission_request_id: AdmissionRequestId,
        completion: QueryCompletion,
    ) -> QueryError {
        if primary.code() == QueryErrorCode::Cancelled {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                state.queued.len(),
                None,
                None,
            );
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                state.queued.len(),
                None,
                None,
            );
        } else if primary.fact_value("reason") == Some("queue_wait_elapsed") {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::TimedOut,
                state.queued.len(),
                None,
                None,
            );
        } else {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                admission_rejection(&primary),
                state.queued.len(),
                None,
                None,
            );
        }
        self.inner.admission.notify_admission_available();
        // Completion observers may take the admission mutex; release it first.
        drop(state);
        if primary.fact_value("reason") == Some("queue_wait_elapsed") {
            completion.finish_timeout(&primary);
        } else {
            completion.finish_error(&primary);
        }
        primary
    }

    fn publish_admission_envelope(
        &self,
        mut state: MutexGuard<'_, AdmissionState>,
        admission_poisoned: bool,
        mut envelope: AdmissionEnvelope,
        publication: AdmissionPublication,
        probe: &Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryHandle> {
        let AdmissionPublication {
            observation,
            attempt_id,
            completion,
            admitted_device_requirements,
            submitted_plan_capability_evidence,
            admitted_compatible_devices,
            admission_request_id,
            admission_wait,
            execution_cancellation,
            admission_wait_fact,
        } = publication;
        let admitted_capability_evidence = submitted_plan_capability_evidence;
        if let Err(error) = check_publication_cancellation(&execution_cancellation, probe.as_deref()) {
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
            self.inner.admission.notify_admission_available();
            // The grant release observer takes the admission mutex; release the guard first.
            drop(state);
            pause_after_rollback_accounting(probe.as_deref());
            // Release or park the envelope's capacity before the notify and the completion below.
            drop(envelope);
            self.inner.admission.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        let rejection = if admission_poisoned {
            Some(admission_mutex_poisoned_error())
        } else {
            self.inner.admission.admission_terminal_rejection(&state)
        };
        if let Some(error) = rejection {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            self.inner.admission.notify_admission_available();
            // The grant release observer takes the admission mutex; release the guard first.
            drop(state);
            // Reverse-order component release happens outside the admission
            // mutex because the grant drop invokes the ledger release hook.
            pause_after_rollback_accounting(probe.as_deref());
            // Release or park the envelope's capacity before the notify and the completion below.
            drop(envelope);
            self.inner.admission.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        if let Err(error) = envelope.activate_lifecycle_ticket() {
            envelope.preaccount_rollback_cleanup(&mut state);
            self.inner.admission.notify_admission_available();
            // The grant release observer takes the admission mutex; release the guard first.
            drop(state);
            pause_after_rollback_accounting(probe.as_deref());
            // Release or park the envelope's capacity before the notify and the completion below.
            drop(envelope);
            self.inner.admission.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        let admission_sequence = self.inner.admission.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let device_ordinal = envelope.device_ordinal();
        let device_budget_bytes = envelope.memory_grant().budget_bytes();
        let admitted_fact = AttemptAdmissionFact {
            admission_sequence,
            admission_wait_ns: u64::try_from(admission_wait.as_nanos()).unwrap_or(u64::MAX),
            device_ordinal,
            grant_bytes: device_budget_bytes,
        };
        if let Err(lifecycle_error) = completion.record_attempt_admitted(admitted_fact, admission_wait_fact) {
            // The attempt owner committed a terminal outcome while this
            // admission was in flight. The grant would never execute, so the
            // envelope rolls back and the claim finishes the attempt as not
            // admitted, keeping the owner's outcome.
            let error = attempt_terminal_before_admission_error(&lifecycle_error);
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            self.inner.admission.notify_admission_available();
            // The grant release observer takes the admission mutex; release the guard first.
            drop(state);
            pause_after_rollback_accounting(probe.as_deref());
            // Release or park the envelope's capacity before the notify and the completion below.
            drop(envelope);
            self.inner.admission.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        let completion = completion.into_admitted(&mut envelope);
        record_admission_lifecycle(
            &observation,
            admission_request_id,
            AdmissionLifecycleState::Admitted,
            state.queued.len(),
            Some(device_ordinal),
            Some(device_budget_bytes),
        );
        // Release admission lock before constructing and returning the handle.
        drop(state);
        Ok(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&self.inner),
                admission_request_id,
                observation,
                attempt_id,
                completion,
                envelope,
                admitted_capability_evidence,
                admitted_device_requirements,
                admitted_compatible_devices,
            ),
            config: self.inner.config.clone(),
            queue_bounds: self.inner.admission.queue_bounds.clone(),
            admission_request_id,
            admission_sequence,
            admission_wait,
            admission_wait_fact,
            device_budget_bytes,
            device_ordinal,
            #[cfg(any(test, feature = "test-hooks"))]
            test_attempt: Mutex::new(None),
        })
    }
}

fn attempt_terminal_before_admission_error(error: &AttemptLifecycleError) -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "query_engine.query_service.admission_transaction.publish",
        "the attempt ended before its admission was published",
    )
    .with_fact("reason", "attempt_terminal_before_admission")
    .with_source(error.clone())
}
