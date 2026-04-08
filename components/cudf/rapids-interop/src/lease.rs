use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::{
    CompletionAuthority, CompletionBinding, IncompleteDropBehavior, InteropError, InteropErrorKind, LeaseActivation,
    LeaseBackend, LeaseCompletionResult, LeaseExportContext, PreparedColumnExport, PreparedTableExport,
    ReadStartAuthority, Result,
};

#[derive(Debug, Clone, Default)]
pub(crate) struct LeaseRootState {
    active: Arc<AtomicBool>,
}

impl LeaseRootState {
    pub(crate) fn try_acquire(&self, export_kind: &'static str) -> Result<ActiveLeaseRegistration> {
        match self.active.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(ActiveLeaseRegistration { active: Arc::clone(&self.active) }),
            Err(_) => Err(InteropError::new(
                InteropErrorKind::LeaseAlreadyActive,
                "prepared export root does not support overlapping active read leases",
            )
            .with_context("export_kind", export_kind)),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ActiveLeaseRegistration {
    active: Arc<AtomicBool>,
}

impl Drop for ActiveLeaseRegistration {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
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
