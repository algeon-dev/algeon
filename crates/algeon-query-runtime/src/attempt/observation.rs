//! The observation session an attempt owns.

#[cfg(test)]
use std::sync::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[cfg(test)]
use crate::observability::RuntimeObservationDeliveryGate;
use crate::observability::{
    AttemptReportSealProvider, QueryBoundaryKind, QueryCleanupSealDelivery, RuntimeObservationClock,
    RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationPayload, RuntimeObservationSession,
    RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTraceId, SealedRuntimeObservationSnapshot,
};

/// An attempt's observation session.
///
/// The attempt's finalizer orders its one seal, so the session needs no
/// terminal fence of its own: a seal that loses to an earlier one returns the
/// earlier snapshot and delivers nothing.
#[derive(Debug)]
pub(crate) struct AttemptObservation {
    session: RuntimeObservationSession,
    cleanup_started: AtomicBool,
    #[cfg(test)]
    delivery_gate: Mutex<Option<Arc<RuntimeObservationDeliveryGate>>>,
}

impl AttemptObservation {
    pub(super) fn new(
        query_id: RuntimeQueryId,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
    ) -> Self {
        let session = RuntimeObservationSession::with_clock(query_id, trace_id, config, clock);
        session.handle().record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started });
        Self {
            session,
            cleanup_started: AtomicBool::new(false),
            #[cfg(test)]
            delivery_gate: Mutex::new(None),
        }
    }

    /// Recording access without seal authority.
    pub(crate) fn handle(&self) -> RuntimeObservationHandle {
        self.session.handle()
    }

    /// Records that the attempt's cleanup began, once.
    pub(crate) fn begin_cleanup(&self) {
        if !self.cleanup_started.swap(true, Ordering::AcqRel) {
            self.handle()
                .record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupStarted });
        }
    }

    /// Records the end of cleanup and the terminal boundary, then seals the
    /// session with `outcome` and the report `report` builds. Observer
    /// delivery is deferred to the returned value.
    pub(crate) fn seal(
        &self,
        outcome: RuntimeTerminalOutcome,
        report: &AttemptReportSealProvider,
    ) -> QueryCleanupSealDelivery {
        self.begin_cleanup();
        let handle = self.handle();
        handle.record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished });
        let boundary = terminal_boundary(&outcome);
        handle.record(|| RuntimeObservationPayload::QueryBoundary { boundary });
        let delivery = self.session.seal_with_report(outcome, report);
        #[cfg(test)]
        let delivery = delivery
            .with_delivery_gate(self.delivery_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take());
        delivery
    }

    /// Registers a notification for the session's seal; see
    /// [`RuntimeObservationSession::register_seal_observer`].
    pub(crate) fn register_seal_observer<F>(&self, observer: F)
    where F: Fn(SealedRuntimeObservationSnapshot) + Send + Sync + 'static {
        self.session.register_seal_observer(observer);
    }

    #[cfg(test)]
    pub(crate) fn sealed_snapshot(&self) -> Option<SealedRuntimeObservationSnapshot> {
        self.session.sealed_snapshot()
    }

    /// Holds the seal's observer delivery until the returned gate opens.
    #[cfg(test)]
    pub(crate) fn arm_delivery_gate(&self) -> Arc<RuntimeObservationDeliveryGate> {
        let gate = Arc::new(RuntimeObservationDeliveryGate::new());
        let mut slot = self.delivery_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(slot.is_none(), "an attempt observation can arm only one delivery gate");
        *slot = Some(Arc::clone(&gate));
        gate
    }
}

fn terminal_boundary(outcome: &RuntimeTerminalOutcome) -> QueryBoundaryKind {
    match outcome {
        RuntimeTerminalOutcome::Succeeded => QueryBoundaryKind::Succeeded,
        RuntimeTerminalOutcome::Failed { .. } => QueryBoundaryKind::Failed,
        RuntimeTerminalOutcome::Cancelled => QueryBoundaryKind::Cancelled,
        RuntimeTerminalOutcome::Dropped => QueryBoundaryKind::Dropped,
        RuntimeTerminalOutcome::Panicked => QueryBoundaryKind::Panicked,
    }
}
