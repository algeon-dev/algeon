use super::*;

#[derive(Debug, Clone)]
pub(super) struct ReservationDiagnosticContext {
    pub(super) task_attempt_id: Option<TaskAttemptId>,
    pub(super) task_kind: Option<String>,
    pub(super) span_id: Option<RuntimeSpanId>,
    pub(super) resident_resource_id: Option<String>,
    pub(super) release_disposition: RuntimeReservationDisposition,
}

impl Default for ReservationDiagnosticContext {
    fn default() -> Self {
        Self {
            task_attempt_id: None,
            task_kind: None,
            span_id: None,
            resident_resource_id: None,
            release_disposition: RuntimeReservationDisposition::ExplicitRelease,
        }
    }
}

/// Restores the previous typed reservation context on every return path and unwind.
#[must_use]
pub(in crate::exec) struct ReservationDiagnosticScope {
    pub(super) context: Rc<RefCell<ReservationDiagnosticContext>>,
    pub(super) previous: Option<ReservationDiagnosticContext>,
}

impl Drop for ReservationDiagnosticScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            *self.context.borrow_mut() = previous;
        }
    }
}
