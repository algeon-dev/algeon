use std::marker::PhantomData;

use cudf::{
    interop::contract::{
        ActiveColumnReadLease, CompletionAuthority, LeaseCompletion, PreparedColumnExport, PreparedExportProfile,
        ReadStartAuthority,
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
    binding: ExecutionBindingId,
    _consumer: PhantomData<&'consumer CumlResources>,
}

impl PreparedMatrixLease<'_, '_, '_> {
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.binding
    }
    pub fn complete(mut self) -> Result<LeaseCompletion> {
        let lease = self.lease.take().expect("active prepared lease");
        lease.complete().map_err(|e| from_interop("PreparedMatrixLease::complete", e))
    }
}

impl CumlResources {
    pub fn activate_prepared_matrix<'e, 'p, 'c>(
        &'c self,
        export: &'e PreparedColumnExport<'p>,
    ) -> Result<PreparedMatrixLease<'e, 'p, 'c>> {
        let op = "CumlResources::activate_prepared_matrix";
        self.validate_exact_binding(op)?;
        if export.profile() != PreparedExportProfile::Device {
            return Err(error(
                ErrorCode::UnsupportedLayout,
                FailurePhase::Preflight,
                op,
                "prepared export must be device-resident",
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
        Ok(PreparedMatrixLease { lease: Some(lease), binding: self.binding_id(), _consumer: PhantomData })
    }
}
