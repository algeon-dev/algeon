use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    CompletionAuthority, CompletionBinding, IncompleteDropBehavior, LeaseActivation, LeaseBackend,
    LeaseCompletionResult, LeaseExportContext, PreparedColumnExport, PreparedTableExport, ReadStartAuthority, Result,
};

#[derive(Debug, Clone, Default)]
pub(crate) struct LeaseRootState {
    active_leases: Arc<AtomicUsize>,
}

impl LeaseRootState {
    pub(crate) fn try_acquire(&self, export_kind: &'static str) -> Result<ActiveLeaseRegistration> {
        let previous = self.active_leases.fetch_add(1, Ordering::AcqRel);
        if previous == usize::MAX {
            // Undo the wrapping increment.
            self.active_leases.fetch_sub(1, Ordering::AcqRel);
            return Err(crate::InteropError::new(
                crate::InteropErrorKind::ProfileUnsupported,
                "active lease counter overflow",
            )
            .with_context("export_kind", export_kind));
        }
        Ok(ActiveLeaseRegistration { active_leases: Arc::clone(&self.active_leases) })
    }
}

#[derive(Debug)]
pub(crate) struct ActiveLeaseRegistration {
    active_leases: Arc<AtomicUsize>,
}

impl Drop for ActiveLeaseRegistration {
    fn drop(&mut self) {
        let previous = self.active_leases.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "active lease registration underflow");
    }
}

/// Consumer-activated read lease over a prepared column export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion follows the backend-defined incomplete-drop behavior"]
pub struct ActiveColumnReadLease<'export, 'producer, 'consumer> {
    pub(crate) export: &'export PreparedColumnExport<'producer>,
    pub(crate) read_start: ReadStartAuthority<'consumer>,
    pub(crate) completion: CompletionAuthority<'consumer>,
    pub(crate) completion_binding: CompletionBinding,
    pub(crate) export_context: LeaseExportContext,
    pub(crate) lease_backend: LeaseBackend,
    pub(crate) _activation: LeaseActivation,
    pub(crate) _registration: ActiveLeaseRegistration,
    pub(crate) completed: bool,
}

impl<'export, 'producer, 'consumer> ActiveColumnReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedColumnExport<'producer> {
        self.export
    }

    pub fn read_start(&self) -> ReadStartAuthority<'consumer> {
        self.read_start
    }

    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.completion
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    pub fn complete(mut self) -> Result<LeaseCompletionResult> {
        let result = self.lease_backend.complete(&self.export_context, self.completion_binding)?;
        self.completed = true;
        Ok(result)
    }
}

impl Drop for ActiveColumnReadLease<'_, '_, '_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.lease_backend.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback) {
            let _ = self.lease_backend.drop_incomplete(&self.export_context, self.completion_binding);
        }
    }
}

/// Consumer-activated read lease over a prepared table export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion follows the backend-defined incomplete-drop behavior"]
pub struct ActiveTableReadLease<'export, 'producer, 'consumer> {
    pub(crate) export: &'export PreparedTableExport<'producer>,
    pub(crate) read_start: ReadStartAuthority<'consumer>,
    pub(crate) completion: CompletionAuthority<'consumer>,
    pub(crate) completion_binding: CompletionBinding,
    pub(crate) export_context: LeaseExportContext,
    pub(crate) lease_backend: LeaseBackend,
    pub(crate) _activation: LeaseActivation,
    pub(crate) _registration: ActiveLeaseRegistration,
    pub(crate) completed: bool,
}

impl<'export, 'producer, 'consumer> ActiveTableReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedTableExport<'producer> {
        self.export
    }

    pub fn read_start(&self) -> ReadStartAuthority<'consumer> {
        self.read_start
    }

    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.completion
    }

    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.lease_backend.incomplete_drop_behavior()
    }

    pub fn complete(mut self) -> Result<LeaseCompletionResult> {
        let result = self.lease_backend.complete(&self.export_context, self.completion_binding)?;
        self.completed = true;
        Ok(result)
    }
}

impl Drop for ActiveTableReadLease<'_, '_, '_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.lease_backend.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback) {
            let _ = self.lease_backend.drop_incomplete(&self.export_context, self.completion_binding);
        }
    }
}
