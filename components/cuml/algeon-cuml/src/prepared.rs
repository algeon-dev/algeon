use std::{any::Any, marker::PhantomData, mem, sync::Arc};

use cudf::{
    interop::contract::{
        ActiveColumnReadLease, CompletionAuthority, CompletionWitness, IncompleteDropBehavior, LeaseCompletionResult,
        PreparedColumnExport, PreparedExportProfile, ReadStartAuthority,
    },
    runtime::ExecutionBindingId,
};

use crate::{
    FailurePhase,
    error::{ErrorCode, Result, error, from_interop},
    resources::CumlResources,
};

#[must_use = "prepared reads must be completed after native work is enqueued"]
pub struct PreparedMatrixLease<'export, 'producer, 'consumer> {
    lease: Option<ActiveColumnReadLease<'export, 'producer, 'consumer>>,
    export: &'export PreparedColumnExport<'producer>,
    binding: ExecutionBindingId,
    _consumer: PhantomData<&'consumer CumlResources>,
}

impl PreparedMatrixLease<'_, '_, '_> {
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.binding
    }
    pub fn complete(mut self) -> Result<PreparedReadCompletion> {
        let retained = self.export.clone_retained_owner();
        let lease = self.lease.take().expect("active prepared lease");
        let result = lease.complete().map_err(|e| from_interop("PreparedMatrixLease::complete", e))?;
        PreparedReadCompletion::new(result, retained, "PreparedMatrixLease::complete")
    }
}

pub struct PreparedReadCompletion {
    retained: Option<Arc<dyn Any + Send + Sync>>,
    witness: Option<CompletionWitness>,
}

impl PreparedReadCompletion {
    fn new(
        result: LeaseCompletionResult,
        retained: Option<Arc<dyn Any + Send + Sync>>,
        op: &'static str,
    ) -> Result<Self> {
        match result {
            LeaseCompletionResult::FallbackSynchronized => Ok(Self { retained: None, witness: None }),
            LeaseCompletionResult::Completed(w) => {
                if let Some(retained) = retained {
                    Ok(Self { retained: Some(retained), witness: Some(w) })
                } else {
                    // Without a retained owner, completion must precede return to the producer.
                    w.synchronize().map_err(|e| from_interop(op, e))?;
                    Ok(Self { retained: None, witness: None })
                }
            },
        }
    }
    pub fn witness(&self) -> Option<&CompletionWitness> {
        self.witness.as_ref()
    }
    pub fn retains_producer_owner(&self) -> bool {
        self.retained.is_some()
    }
    pub fn synchronize(mut self) -> Result<()> {
        if let Some(w) = self.witness.take() {
            w.synchronize().map_err(|e| from_interop("PreparedReadCompletion::synchronize", e))?;
        }
        self.retained.take();
        Ok(())
    }
}

impl Drop for PreparedReadCompletion {
    fn drop(&mut self) {
        if let Some(w) = self.witness.take()
            && w.synchronize().is_err()
            && let Some(owner) = self.retained.take()
        {
            // A failed fence leaves native readers possible; leaking the owner
            // is safer than freeing device storage still in use.
            mem::forget(owner);
        }
    }
}

impl CumlResources {
    pub fn activate_prepared_matrix<'e, 'p, 'c>(
        &'c self,
        export: &'e PreparedColumnExport<'p>,
    ) -> Result<PreparedMatrixLease<'e, 'p, 'c>> {
        let op = "CumlResources::activate_prepared_matrix";
        self.validate_exact_binding(op)?;
        if export.profile() != PreparedExportProfile::Device
            || export.incomplete_drop_behavior() != IncompleteDropBehavior::GuaranteedFallback
        {
            return Err(error(
                ErrorCode::UnsupportedLayout,
                FailurePhase::Preflight,
                op,
                "prepared export must be device-resident with guaranteed fallback",
            ));
        }
        let binding = self.runtime().execution_binding().map_err(|e| from_interop(op, e))?;
        let stream = binding.cuda_stream();
        let lease = export
            .try_activate_read_lease(
                ReadStartAuthority::ConsumerStream { stream },
                CompletionAuthority::SameDeviceStream { stream },
            )
            .map_err(|e| from_interop(op, e))?;
        Ok(PreparedMatrixLease { lease: Some(lease), export, binding: self.binding_id(), _consumer: PhantomData })
    }
}
