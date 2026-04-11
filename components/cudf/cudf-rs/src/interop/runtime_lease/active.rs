use std::fmt;

use rapids_interop::{
    CompletionAuthority, IncompleteDropBehavior, InteropError, PreparedColumnExport, PreparedTableExport,
    ProducerReadiness, ReadStartAuthority,
};

use super::{
    ACTIVATE_RUNTIME_COLUMN_LEASE_OP, ACTIVATE_RUNTIME_TABLE_LEASE_OP, COMPLETE_RUNTIME_COLUMN_LEASE_OP,
    COMPLETE_RUNTIME_TABLE_LEASE_OP, DROP_RUNTIME_COLUMN_LEASE_OP, DROP_RUNTIME_TABLE_LEASE_OP,
    authorities::{
        ProducerReadinessBinding, RuntimeCompletionAuthority, RuntimeLeaseCompletionResult, RuntimeReadStartAuthority,
        map_runtime_completion_interop_error, producer_readiness_binding, runtime_completion_as_shared,
        runtime_completion_result_from_shared, runtime_read_start_as_shared,
    },
    sync::synchronize_completion_fallback,
};
use crate::core::error::Result;

trait RuntimeSharedLease {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior;
    fn complete(self) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult>;
}

impl RuntimeSharedLease for super::super::ActiveColumnReadLease<'_, '_, '_> {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        super::super::ActiveColumnReadLease::incomplete_drop_behavior(self)
    }

    fn complete(self) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult> {
        super::super::ActiveColumnReadLease::complete(self)
    }
}

impl RuntimeSharedLease for super::super::ActiveTableReadLease<'_, '_, '_> {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        super::super::ActiveTableReadLease::incomplete_drop_behavior(self)
    }

    fn complete(self) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult> {
        super::super::ActiveTableReadLease::complete(self)
    }
}

struct RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease: RuntimeSharedLease> {
    export: &'export Export,
    read_start: RuntimeReadStartAuthority<'consumer>,
    completion: RuntimeCompletionAuthority<'consumer>,
    producer_readiness: ProducerReadinessBinding,
    incomplete_drop_behavior: IncompleteDropBehavior,
    shared_lease: Option<SharedLease>,
    complete_operation: &'static str,
    drop_operation: &'static str,
    completed: bool,
}

impl<'export, 'consumer, Export, SharedLease: RuntimeSharedLease>
    RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease>
{
    fn new(
        export: &'export Export,
        read_start: RuntimeReadStartAuthority<'consumer>,
        completion: RuntimeCompletionAuthority<'consumer>,
        producer_readiness: ProducerReadinessBinding,
        shared_lease: SharedLease,
        complete_operation: &'static str,
        drop_operation: &'static str,
    ) -> Self {
        let incomplete_drop_behavior = shared_lease.incomplete_drop_behavior();
        Self {
            export,
            read_start,
            completion,
            producer_readiness,
            incomplete_drop_behavior,
            shared_lease: Some(shared_lease),
            complete_operation,
            drop_operation,
            completed: false,
        }
    }

    fn export(&self) -> &'export Export {
        self.export
    }

    fn read_start(&self) -> &RuntimeReadStartAuthority<'consumer> {
        &self.read_start
    }

    fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
        &self.completion
    }

    fn producer_device(&self) -> crate::runtime::DeviceId {
        self.producer_readiness.device()
    }

    fn complete(mut self) -> Result<RuntimeLeaseCompletionResult> {
        let shared_lease = self.shared_lease.take().expect("runtime lease missing shared lease state");
        let shared_result = shared_lease
            .complete()
            .map_err(|error| map_runtime_completion_interop_error(self.complete_operation, error))?;
        let result = runtime_completion_result_from_shared(shared_result, self.complete_operation)?;
        self.completed = true;
        Ok(result)
    }
}

impl<Export, SharedLease: RuntimeSharedLease> Drop for RuntimeActiveLeaseState<'_, '_, Export, SharedLease> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.incomplete_drop_behavior, IncompleteDropBehavior::GuaranteedFallback) {
            let _ = synchronize_completion_fallback(self.completion, self.drop_operation);
        }
    }
}

struct RuntimeActiveReadLease<'export, 'consumer, Export, SharedLease>
where SharedLease: RuntimeSharedLease
{
    inner: RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease>,
}

impl<'export, 'consumer, Export, SharedLease> RuntimeActiveReadLease<'export, 'consumer, Export, SharedLease>
where SharedLease: RuntimeSharedLease
{
    fn new(inner: RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease>) -> Self {
        Self { inner }
    }

    fn export(&self) -> &'export Export {
        self.inner.export()
    }

    fn read_start(&self) -> &RuntimeReadStartAuthority<'consumer> {
        self.inner.read_start()
    }

    fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
        self.inner.completion()
    }

    fn complete(self) -> Result<RuntimeLeaseCompletionResult> {
        self.inner.complete()
    }

    fn fmt_debug(&self, name: &'static str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(name)
            .field("read_start", &self.read_start())
            .field("completion", &self.completion())
            .field("producer_device", &self.inner.producer_device())
            .finish()
    }
}

fn activate_runtime_read_lease<'export, 'consumer, Export, SharedLease>(
    export: &'export Export,
    read_start: RuntimeReadStartAuthority<'consumer>,
    completion: RuntimeCompletionAuthority<'consumer>,
    readiness: &ProducerReadiness<'_>,
    activate_operation: &'static str,
    complete_operation: &'static str,
    drop_operation: &'static str,
    activate_shared: impl FnOnce(
        ReadStartAuthority<'consumer>,
        CompletionAuthority<'consumer>,
    ) -> std::result::Result<SharedLease, InteropError>,
) -> std::result::Result<RuntimeActiveReadLease<'export, 'consumer, Export, SharedLease>, InteropError>
where
    SharedLease: RuntimeSharedLease,
{
    let shared_read_start = runtime_read_start_as_shared(read_start, activate_operation)?;
    let shared_completion = runtime_completion_as_shared(completion, activate_operation)?;
    let shared_lease = activate_shared(shared_read_start, shared_completion)?;
    let producer_readiness = producer_readiness_binding(readiness, activate_operation)?;
    Ok(RuntimeActiveReadLease::new(RuntimeActiveLeaseState::new(
        export,
        read_start,
        completion,
        producer_readiness,
        shared_lease,
        complete_operation,
        drop_operation,
    )))
}

macro_rules! impl_runtime_active_read_lease_surface {
    ($lease:ident, $export_ty:ty, $debug_name:literal) => {
        impl<'export, 'producer, 'consumer> $lease<'export, 'producer, 'consumer> {
            pub fn export(&self) -> &'export $export_ty {
                self.inner.export()
            }

            pub fn read_start(&self) -> &RuntimeReadStartAuthority<'consumer> {
                self.inner.read_start()
            }

            pub fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
                self.inner.completion()
            }

            pub fn complete(self) -> Result<RuntimeLeaseCompletionResult> {
                self.inner.complete()
            }
        }

        impl fmt::Debug for $lease<'_, '_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.inner.fmt_debug($debug_name, f)
            }
        }
    };
}

macro_rules! impl_try_activate_runtime_read_lease {
    ($trait_name:ident, $export_ty:ty, $lease:ident, $activate_op:expr, $complete_op:expr, $drop_op:expr) => {
        impl<'producer> $trait_name for $export_ty {
            fn try_activate_runtime_read_lease<'consumer>(
                &self,
                read_start: RuntimeReadStartAuthority<'consumer>,
                completion: RuntimeCompletionAuthority<'consumer>,
            ) -> std::result::Result<$lease<'_, '_, 'consumer>, InteropError> {
                let inner = activate_runtime_read_lease(
                    self,
                    read_start,
                    completion,
                    self.readiness(),
                    $activate_op,
                    $complete_op,
                    $drop_op,
                    |shared_read_start, shared_completion| {
                        self.try_activate_read_lease(shared_read_start, shared_completion)
                    },
                )?;
                Ok($lease { inner })
            }
        }
    };
}

/// CUDA-specific compatibility wrapper over a shared active column read lease.
///
/// Activation still goes through [`PreparedColumnExport::try_activate_read_lease`]
/// under the hood, so export-root overlapping-lease coordination, shared
/// liveness retention, and backend-declared incomplete-drop behavior all stay
/// aligned with the shared contract. This wrapper only changes the completion
/// surface exposed to CUDA-runtime-oriented callers.
#[must_use = "lease must be completed explicitly via .complete(); dropping without completion falls back to conservative synchronization"]
pub struct RuntimeActiveColumnReadLease<'export, 'producer, 'consumer> {
    inner: RuntimeActiveReadLease<
        'export,
        'consumer,
        PreparedColumnExport<'producer>,
        super::super::ActiveColumnReadLease<'export, 'producer, 'consumer>,
    >,
}
impl_runtime_active_read_lease_surface!(
    RuntimeActiveColumnReadLease,
    PreparedColumnExport<'producer>,
    "RuntimeActiveColumnReadLease"
);

/// CUDA-specific compatibility wrapper over a shared active table read lease.
///
/// Activation still goes through [`PreparedTableExport::try_activate_read_lease`]
/// under the hood, so export-root overlapping-lease coordination, shared
/// liveness retention, and backend-declared incomplete-drop behavior all stay
/// aligned with the shared contract. This wrapper only changes the completion
/// surface exposed to CUDA-runtime-oriented callers.
#[must_use = "lease must be completed explicitly via .complete(); dropping without completion falls back to conservative synchronization"]
pub struct RuntimeActiveTableReadLease<'export, 'producer, 'consumer> {
    inner: RuntimeActiveReadLease<
        'export,
        'consumer,
        PreparedTableExport<'producer>,
        super::super::ActiveTableReadLease<'export, 'producer, 'consumer>,
    >,
}
impl_runtime_active_read_lease_surface!(
    RuntimeActiveTableReadLease,
    PreparedTableExport<'producer>,
    "RuntimeActiveTableReadLease"
);

/// CUDA-specific compatibility activation over a prepared column export.
///
/// Prefer [`PreparedColumnExport::try_activate_read_lease`] for new shared
/// interop APIs. Use this trait when you specifically need the
/// `cudf-rs`-owned runtime witness surface.
pub trait TryActivateRuntimeColumnReadLease {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        read_start: RuntimeReadStartAuthority<'consumer>,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveColumnReadLease<'_, '_, 'consumer>, InteropError>;
}

impl_try_activate_runtime_read_lease!(
    TryActivateRuntimeColumnReadLease,
    PreparedColumnExport<'producer>,
    RuntimeActiveColumnReadLease,
    ACTIVATE_RUNTIME_COLUMN_LEASE_OP,
    COMPLETE_RUNTIME_COLUMN_LEASE_OP,
    DROP_RUNTIME_COLUMN_LEASE_OP
);

/// CUDA-specific compatibility activation over a prepared table export.
///
/// Prefer [`PreparedTableExport::try_activate_read_lease`] for new shared
/// interop APIs. Use this trait when you specifically need the
/// `cudf-rs`-owned runtime witness surface.
pub trait TryActivateRuntimeTableReadLease {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        read_start: RuntimeReadStartAuthority<'consumer>,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveTableReadLease<'_, '_, 'consumer>, InteropError>;
}

impl_try_activate_runtime_read_lease!(
    TryActivateRuntimeTableReadLease,
    PreparedTableExport<'producer>,
    RuntimeActiveTableReadLease,
    ACTIVATE_RUNTIME_TABLE_LEASE_OP,
    COMPLETE_RUNTIME_TABLE_LEASE_OP,
    DROP_RUNTIME_TABLE_LEASE_OP
);
