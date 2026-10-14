use std::{any::Any, fmt, mem, process, sync::Arc};

use crate::{
    CompletionAuthority, CompletionWitness, LeaseActivation, LeaseBackend, PreparedColumnExport, PreparedTableExport,
    ProducerReadiness, ReadStartAuthority, Result,
};

/// Backend state of a lease over a device export. Host leases have none.
#[derive(Debug)]
pub(crate) struct DeviceLease<'producer> {
    backend: LeaseBackend,
    readiness: ProducerReadiness<'producer>,
    _activation: LeaseActivation,
}

impl<'producer> DeviceLease<'producer> {
    pub(crate) fn new(
        backend: LeaseBackend,
        readiness: ProducerReadiness<'producer>,
        activation: LeaseActivation,
    ) -> Self {
        Self { backend, readiness, _activation: activation }
    }
}

/// Consumer authorities and completion state shared by column and table
/// leases. `Drop` runs the backend fallback for a device lease that was not
/// completed.
#[derive(Debug)]
pub(crate) struct LeaseCore<'producer, 'consumer> {
    read_start: ReadStartAuthority<'consumer>,
    completion: CompletionAuthority<'consumer>,
    device: Option<DeviceLease<'producer>>,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    completed: bool,
}

impl<'producer, 'consumer> LeaseCore<'producer, 'consumer> {
    pub(crate) fn new(
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
        device: Option<DeviceLease<'producer>>,
        retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Self {
        Self { read_start, completion, device, retained_owner, completed: false }
    }

    fn complete(mut self) -> Result<LeaseCompletion> {
        // Host leases read host memory before returning, so they complete
        // without backend work.
        let witness = match &self.device {
            Some(device) => device.backend.complete(device.readiness, self.completion)?,
            None => None,
        };
        let completion = LeaseCompletion::new(witness, self.retained_owner.as_ref())?;
        self.completed = true;
        Ok(completion)
    }
}

impl Drop for LeaseCore<'_, '_> {
    fn drop(&mut self) {
        if !self.completed
            && let Some(device) = &self.device
            && device.backend.drop_incomplete(device.readiness, self.completion).is_err()
        {
            let Some(owner) = self.retained_owner.take() else {
                // Returning or unwinding would end the borrow while native
                // work may still read storage we cannot retain.
                process::abort();
            };
            mem::forget(owner);
            // Backend activation resources may still be referenced by CUDA.
            mem::forget(self.device.take());
        }
    }
}

/// Completion of one read lease.
///
/// When the backend returns a witness and the export has a retained owner,
/// this value holds both until the consumer's reads of producer storage are
/// known to be finished. Without a retained owner, the witness is synchronized
/// before the lease's `complete` returns, so producer storage never depends on
/// a pending witness that nothing keeps alive.
///
/// Dropping a pending completion waits on its witness. If that wait fails, the
/// retained owner is leaked: queued consumer work may still read producer
/// storage. The backend observes and quarantines the failed final wait.
#[must_use = "dropping a pending completion blocks until consumer reads of producer storage finish"]
pub struct LeaseCompletion {
    witness: Option<CompletionWitness>,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
}

impl LeaseCompletion {
    fn new(witness: Option<CompletionWitness>, retained_owner: Option<&Arc<dyn Any + Send + Sync>>) -> Result<Self> {
        let Some(witness) = witness else {
            return Ok(Self::finished());
        };
        match retained_owner {
            Some(owner) => Ok(Self { witness: Some(witness), retained_owner: Some(Arc::clone(owner)) }),
            None => {
                witness.synchronize()?;
                Ok(Self::finished())
            },
        }
    }

    const fn finished() -> Self {
        Self { witness: None, retained_owner: None }
    }

    /// Queries, without waiting, whether the consumer's reads of producer
    /// storage have finished.
    pub fn is_ready(&self) -> Result<bool> {
        self.witness.as_ref().map_or(Ok(true), CompletionWitness::is_ready)
    }

    /// Waits until the consumer's reads of producer storage have finished,
    /// then releases the retained owner. After `Err` the witness and owner
    /// stay held, so a later call or `Drop` waits again.
    pub fn synchronize(&mut self) -> Result<()> {
        if let Some(witness) = &self.witness {
            witness.synchronize()?;
        }
        self.witness = None;
        self.retained_owner = None;
        Ok(())
    }

    /// Releases the witness and retained owner without waiting.
    ///
    /// # Safety
    ///
    /// Every release of producer storage must already be ordered after the
    /// consumer's reads on the device, for example by a producer stream wait
    /// on the consumer's completion event. Merely holding a witness does not
    /// establish that order.
    ///
    /// ```compile_fail,E0133
    /// use rapids_interop::LeaseCompletion;
    /// fn skip_wait(completion: LeaseCompletion) {
    ///     completion.release_stream_ordered();
    /// }
    /// ```
    pub unsafe fn release_stream_ordered(mut self) {
        self.witness = None;
        self.retained_owner = None;
    }
}

impl Drop for LeaseCompletion {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            if let Some(witness) = &self.witness {
                witness.quarantine(&error);
            }
            if let Some(owner) = self.retained_owner.take() {
                mem::forget(owner);
            }
            mem::forget(self.witness.take());
        }
    }
}

impl fmt::Debug for LeaseCompletion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeaseCompletion")
            .field("witness", &self.witness)
            .field("retains_owner", &self.retained_owner.is_some())
            .finish()
    }
}

/// Consumer-activated read lease over a prepared column export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion runs the backend's drop fallback"]
pub struct ActiveColumnReadLease<'export, 'producer, 'consumer> {
    pub(crate) export: &'export PreparedColumnExport<'producer>,
    pub(crate) core: LeaseCore<'producer, 'consumer>,
}

impl<'export, 'producer, 'consumer> ActiveColumnReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedColumnExport<'producer> {
        self.export
    }

    pub fn read_start(&self) -> ReadStartAuthority<'consumer> {
        self.core.read_start
    }

    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.core.completion
    }

    /// Completes the lease. See [`LeaseCompletion`] for when the returned
    /// value keeps the export's retained owner alive.
    pub fn complete(self) -> Result<LeaseCompletion> {
        self.core.complete()
    }
}

/// Consumer-activated read lease over a prepared table export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion runs the backend's drop fallback"]
pub struct ActiveTableReadLease<'export, 'producer, 'consumer> {
    pub(crate) export: &'export PreparedTableExport<'producer>,
    pub(crate) core: LeaseCore<'producer, 'consumer>,
}

impl<'export, 'producer, 'consumer> ActiveTableReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedTableExport<'producer> {
        self.export
    }

    pub fn read_start(&self) -> ReadStartAuthority<'consumer> {
        self.core.read_start
    }

    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.core.completion
    }

    /// Completes the lease. See [`LeaseCompletion`] for when the returned
    /// value keeps the export's retained owner alive.
    pub fn complete(self) -> Result<LeaseCompletion> {
        self.core.complete()
    }
}
