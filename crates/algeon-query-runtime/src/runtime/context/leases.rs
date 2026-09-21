//! Attempt work leases and stream-lease derivation for execution contexts.

use std::sync::Arc;

use cudf::Runtime as CudfRuntime;
use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    super::{AttemptWorkLease, StreamLease},
    AttemptAuxiliaryPermit, ExecutionContext, StreamBinding, validate_service_stream_pool_device,
};
use crate::observability::{RuntimeTaskId, RuntimeWorkerId};

impl ExecutionContext {
    /// Acquires one attempt work lease: an attempt auxiliary permit, a
    /// service stream lease, and the derived context that owns them.
    ///
    /// This is the single bundle behind every piece of auxiliary attempt work
    /// -- execution lanes and scheduled source reads alike. The derived
    /// context's alias holds one auxiliary fork of the attempt pool alone for
    /// the lease's lifetime, so its native forks never reach a stream another
    /// live holder is launching on. The permit it takes is what bounds those
    /// claims to the pool's width.
    #[doc(hidden)]
    pub fn acquire_attempt_work_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
    ) -> Result<AttemptWorkLease> {
        // Admission reserves this bounded auxiliary share per attempt. Hold
        // the local permit for the exact lifetime of the derived stream lease,
        // so another attempt's reservation remains physically reachable.
        let attempt_auxiliary_permit = self
            .admitted()
            .map(|attempt| {
                attempt.auxiliary_slots.acquire_with_cancellation(
                    self.cancellation(),
                    "query_engine.runtime.attempt_work_lease.attempt_auxiliary_acquire",
                )
            })
            .transpose()?;
        self.derive_attempt_work_lease(source_worker, attempt_auxiliary_permit)
    }

    /// Acquires one attempt work lease only if this attempt's auxiliary
    /// envelope has a free slot right now.
    ///
    /// Callers that already hold an attempt auxiliary permit -- the root
    /// execution job dispatching lanes among them -- must never block on a
    /// second one, because the holder they would wait for may be waiting on
    /// them. Returns `Ok(None)` when the envelope is fully claimed, so the
    /// caller narrows its own width instead of deadlocking.
    #[doc(hidden)]
    pub fn try_acquire_attempt_work_lease(&self) -> Result<Option<AttemptWorkLease>> {
        let attempt_auxiliary_permit = match self.admitted() {
            Some(attempt) => {
                match attempt
                    .auxiliary_slots
                    .try_acquire("query_engine.runtime.attempt_work_lease.attempt_auxiliary_try_acquire")?
                {
                    Some(permit) => Some(permit),
                    None => return Ok(None),
                }
            },
            None => None,
        };
        self.derive_attempt_work_lease(None, attempt_auxiliary_permit).map(Some)
    }

    fn derive_attempt_work_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
        attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
    ) -> Result<AttemptWorkLease> {
        let mut stream_scope = self.plan.runtime_stream_lease_scope.clone();
        if let Some((task_id, worker_id)) = source_worker {
            stream_scope = stream_scope.with_source_worker(task_id, worker_id);
        }
        let stream_lease = self.service_stream_pool().acquire_observed_with_cancellation(
            self.cancellation(),
            "query_engine.runtime.attempt_work_lease.acquire",
            self.plan.runtime_observation.clone(),
            stream_scope,
        )?;
        let context = self.with_stream_lease(&stream_lease)?;
        Ok(AttemptWorkLease::new(context, stream_lease, attempt_auxiliary_permit))
    }

    /// Whether this context runs on a stream leased for attempt work rather
    /// than on the attempt's own root stream.
    ///
    /// Work issued from such a context already holds an attempt auxiliary
    /// permit, so it must run inline instead of leasing a second one.
    #[doc(hidden)]
    pub fn is_attempt_work_lease_context(&self) -> bool {
        matches!(*self.stream, StreamBinding::AttemptWork { .. })
    }

    /// Which auxiliary fork of the attempt pool this context's alias holds
    /// alone. A context with no alias -- the attempt root, or a direct context
    /// -- reports `None`; the root runs on the route's reserved fork instead.
    #[doc(hidden)]
    pub fn attempt_claimed_fork_index(&self) -> Option<usize> {
        match &*self.stream {
            StreamBinding::AttemptWork { alias, .. } => Some(alias.claimed_fork_index()),
            StreamBinding::Standalone { .. } | StreamBinding::AttemptRoot { .. } => None,
        }
    }

    /// Builds a short-lived execution context from a borrowed stream lease.
    ///
    /// The returned context shares this context's query policies, caches, and
    /// cancellation flag, but executes GPU work on the leased runtime. The
    /// caller must keep `stream_lease` alive until all work submitted through
    /// the returned context has completed.
    pub fn with_stream_lease(&self, stream_lease: &StreamLease) -> Result<Self> {
        self.check_service_epoch("query_engine.runtime.context.with_stream_lease")?;
        validate_service_stream_pool_device(stream_lease.pool(), self.device())?;
        let runtime = Arc::new(stream_lease.runtime()?);
        Ok(self.with_stream(self.work_stream(runtime, None)?))
    }

    /// Builds an auxiliary fragment context that owns `stream_lease` and the
    /// attempt `permit` paired with it.
    ///
    /// Both live in the context's stream binding, so cloning an outward
    /// context cannot retain the stream while releasing its attempt envelope
    /// share early.
    pub(crate) fn with_owned_auxiliary_stream(
        &self,
        stream_lease: StreamLease,
        permit: AttemptAuxiliaryPermit,
    ) -> Result<Self> {
        let operation = "query_engine.runtime.context.with_owned_auxiliary_stream";
        self.check_service_epoch(operation)?;
        validate_service_stream_pool_device(stream_lease.pool(), self.device())?;
        if self.admitted().is_none() {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "an auxiliary fragment stream belongs to an admitted attempt",
            )
            .with_fact("reason", "auxiliary_stream_without_admission"));
        }
        let runtime = Arc::new(stream_lease.runtime()?);
        Ok(self.with_stream(self.work_stream(runtime, Some((stream_lease, permit)))?))
    }

    /// Binds `runtime`'s stream as work of this context's query. For an
    /// attempt: a member of its allocation domain, then an alias on the route
    /// of its root stream. A standalone context's work stream is its runtime.
    fn work_stream(
        &self,
        runtime: Arc<CudfRuntime>,
        owned: Option<(StreamLease, AttemptAuxiliaryPermit)>,
    ) -> Result<StreamBinding> {
        let (Some(attempt), Some(route)) = (self.admitted(), self.stream.route()) else {
            return Ok(StreamBinding::Standalone { runtime });
        };
        let member = attempt.allocation_domain.create_member(runtime.stream())?;
        let alias = Arc::new(route.attach(runtime.stream())?);
        Ok(StreamBinding::AttemptWork { runtime, alias, member, route: Arc::clone(route), owned })
    }
}
