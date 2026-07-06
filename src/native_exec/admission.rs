//! Plan-scoped query-service admission tickets for backend-managed execution.

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, Weak},
    time::Duration,
};

use datafusion::execution::context::TaskContext;
use nexus_query_engine::{ExecutionCancellation, QueryHandle, QueryQos, QueryService, TenantId, TenantQueryContext};
use tokio::sync::Notify;

/// Backend-owned service admission source copied into optimizer configs.
#[derive(Debug, Clone)]
pub(crate) struct QueryAdmissionSource {
    service: QueryService,
    tenant_id: TenantId,
    qos: QueryQos,
}

impl QueryAdmissionSource {
    pub(crate) fn new(service: QueryService, tenant_id: TenantId) -> Self {
        Self { service, tenant_id, qos: QueryQos::default() }
    }
}

/// Lazy admission ticket shared by every native node emitted in one optimize pass.
#[derive(Debug)]
pub(crate) struct QueryAdmissionTicket {
    #[cfg(test)]
    id: u64,
    source: QueryAdmissionSource,
    state: Mutex<TicketState>,
    notify: Arc<Notify>,
}

#[derive(Debug, Default)]
struct TicketState {
    rounds: HashMap<usize, TicketRound>,
    admitting: HashSet<usize>,
    next_epoch: u64,
}

#[derive(Debug)]
struct TicketRound {
    handle: Option<Arc<QueryHandle>>,
    active_contexts: usize,
    task_context: Weak<TaskContext>,
    cleanup_started: bool,
    epoch: u64,
}

/// Context acquired through a [`QueryAdmissionTicket`].
#[derive(Debug)]
pub(crate) struct TicketTenantQueryContext {
    tenant_context: TenantQueryContext,
    _guard: QueryAdmissionTicketContextGuard,
}

impl TicketTenantQueryContext {
    pub(crate) fn context(&self) -> &TenantQueryContext {
        &self.tenant_context
    }
}

#[derive(Debug)]
pub(crate) struct QueryAdmissionTicketContextGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    epoch: u64,
    released: bool,
}

struct TicketAdmitResetGuard {
    ticket: Arc<QueryAdmissionTicket>,
    task_key: usize,
    disarmed: bool,
}

impl QueryAdmissionTicket {
    pub(crate) fn new(source: QueryAdmissionSource) -> Arc<Self> {
        #[cfg(test)]
        static NEXT_TICKET_ID: AtomicU64 = AtomicU64::new(1);
        Arc::new(Self {
            #[cfg(test)]
            id: NEXT_TICKET_ID.fetch_add(1, Ordering::Relaxed),
            source,
            state: Mutex::new(TicketState::default()),
            notify: Arc::new(Notify::new()),
        })
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) async fn acquire_context(
        self: &Arc<Self>,
        task_context: Arc<TaskContext>,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::Result<TicketTenantQueryContext> {
        let task_key = Self::task_key(&task_context);
        loop {
            let notify = Arc::clone(&self.notify);
            let notified = notify.notified_owned();
            tokio::pin!(notified);
            let should_admit = {
                let mut state = self.lock_state()?;
                Self::remove_stale_rounds_locked(&mut state);
                if let Some(round) = state.rounds.get_mut(&task_key)
                    && let Some(handle) = round.handle.as_ref().cloned()
                {
                    let epoch = round.epoch;
                    round.active_contexts = round.active_contexts.saturating_add(1);
                    return self.acquire_from_handle(handle, task_key, epoch, cancellation);
                }
                notified.as_mut().enable();
                if state.admitting.contains(&task_key) {
                    false
                } else {
                    state.admitting.insert(task_key);
                    true
                }
            };
            if !should_admit {
                notified.await;
                continue;
            }

            let mut reset_guard = TicketAdmitResetGuard { ticket: Arc::clone(self), task_key, disarmed: false };
            let handle =
                match self.source.service.admit_async_with_qos(self.source.tenant_id.clone(), self.source.qos).await {
                    Ok(handle) => Arc::new(handle),
                    Err(error) => {
                        reset_guard.reset();
                        return Err(error);
                    },
                };
            let mut state = self.lock_state()?;
            state.admitting.remove(&task_key);
            state.next_epoch = state.next_epoch.saturating_add(1);
            let epoch = state.next_epoch;
            state.rounds.insert(
                task_key,
                TicketRound {
                    handle: Some(Arc::clone(&handle)),
                    active_contexts: 1,
                    task_context: Arc::downgrade(&task_context),
                    cleanup_started: false,
                    epoch,
                },
            );
            reset_guard.disarmed = true;
            self.notify.notify_waiters();
            drop(state);
            return self.acquire_from_handle(handle, task_key, epoch, cancellation);
        }
    }

    fn acquire_from_handle(
        self: &Arc<Self>,
        handle: Arc<QueryHandle>,
        task_key: usize,
        epoch: u64,
        cancellation: ExecutionCancellation,
    ) -> nexus_query_engine::Result<TicketTenantQueryContext> {
        match handle.acquire_context_with_cancellation(cancellation) {
            Ok(tenant_context) => Ok(TicketTenantQueryContext {
                tenant_context,
                _guard: QueryAdmissionTicketContextGuard { ticket: Arc::clone(self), task_key, epoch, released: false },
            }),
            Err(error) => {
                self.release_epoch_context(task_key, epoch, true);
                Err(error)
            },
        }
    }

    fn release_epoch_context(self: &Arc<Self>, task_key: usize, epoch: u64, force: bool) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to release poisoned query admission ticket state");
            return;
        };
        let Some(round) = state.rounds.get_mut(&task_key) else {
            return;
        };
        if round.epoch != epoch || round.active_contexts == 0 {
            return;
        }
        round.active_contexts -= 1;
        if round.active_contexts == 0 {
            if force || round.task_context.upgrade().is_none() {
                state.rounds.remove(&task_key);
            } else if !round.cleanup_started {
                round.cleanup_started = true;
                Self::spawn_task_context_cleanup(Arc::clone(self), task_key, epoch, round.task_context.clone());
            }
            self.notify.notify_waiters();
        }
    }

    fn reset_admitting(&self, task_key: usize) {
        let Ok(mut state) = self.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to reset poisoned query admission ticket state");
            return;
        };
        state.admitting.remove(&task_key);
        self.notify.notify_waiters();
    }

    fn lock_state(&self) -> nexus_query_engine::Result<std::sync::MutexGuard<'_, TicketState>> {
        self.state.lock().map_err(|_error| {
            nexus_query_engine::QueryError::new(
                nexus_query_engine::QueryErrorCode::Internal,
                "datafusion_nexus.native_exec.admission_ticket",
                "query admission ticket state mutex was poisoned",
            )
        })
    }

    fn task_key(task_context: &Arc<TaskContext>) -> usize {
        Arc::as_ptr(task_context) as usize
    }

    fn remove_stale_rounds_locked(state: &mut TicketState) {
        state.rounds.retain(|_task_key, round| round.active_contexts > 0 || round.task_context.upgrade().is_some());
    }

    fn spawn_task_context_cleanup(
        ticket: Arc<QueryAdmissionTicket>,
        task_key: usize,
        epoch: u64,
        task_context: Weak<TaskContext>,
    ) {
        tokio::spawn(async move {
            while task_context.strong_count() > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let Ok(mut state) = ticket.state.lock() else {
                eprintln!("[datafusion-nexus] warning: failed to clean poisoned query admission ticket state");
                return;
            };
            let should_remove =
                state.rounds.get(&task_key).is_some_and(|round| round.epoch == epoch && round.active_contexts == 0);
            if should_remove {
                state.rounds.remove(&task_key);
                ticket.notify.notify_waiters();
            }
        });
    }
}

impl Drop for QueryAdmissionTicketContextGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        self.ticket.release_epoch_context(self.task_key, self.epoch, false);
    }
}

impl TicketAdmitResetGuard {
    fn reset(&mut self) {
        if !self.disarmed {
            self.disarmed = true;
            self.ticket.reset_admitting(self.task_key);
        }
    }
}

impl Drop for TicketAdmitResetGuard {
    fn drop(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use datafusion::execution::context::TaskContext;
    use nexus_query_engine::{
        DeploymentIoCapability, ExecutionCancellation, GdsDriverProbeStatus, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryServiceConfig, QueryServiceTenantConfig,
    };

    use super::*;

    #[tokio::test]
    async fn failed_context_acquire_releases_ticket_epoch_for_reexecution() {
        let tenant_id = TenantId::new("tenant");
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let service = QueryService::new_admission_only(
            QueryServiceConfig {
                max_concurrent_queries: 1,
                device_capacity_bytes: Some(1024),
                max_admission_queue_len: 1,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::new(tenant_id.clone(), 1024)],
            deployment,
        )
        .unwrap();
        let ticket = QueryAdmissionTicket::new(QueryAdmissionSource::new(service.clone(), tenant_id));
        let task_context = Arc::new(TaskContext::default());

        let first = ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()).await;
        assert!(first.is_err(), "admission-only services cannot acquire runtime contexts");
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);

        let second = tokio::time::timeout(
            Duration::from_millis(100),
            ticket.acquire_context(Arc::clone(&task_context), ExecutionCancellation::new()),
        )
        .await
        .expect("second execution round should not wait on the failed first round");
        assert!(second.is_err(), "admission-only services cannot acquire runtime contexts");
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    }
}
