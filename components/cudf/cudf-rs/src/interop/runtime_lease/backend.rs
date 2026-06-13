use std::sync::Arc;

use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, CompletionBinding, IncompleteDropBehavior, InteropError,
    InteropErrorKind, LeaseActivation, LeaseBackend, LeaseCompletionResult, LeaseExportContext,
};

use super::{
    SHARED_ACTIVATE_LEASE_OP, SHARED_COMPLETE_LEASE_OP, SHARED_DROP_LEASE_OP,
    authorities::producer_readiness_binding_from_export,
    sync::{
        complete_cross_device_event_shared, complete_same_device_stream_shared,
        order_consumer_stream_after_producer_readiness_raw, synchronize_cross_device_event_raw,
        synchronize_producer_readiness, synchronize_same_device_stream_raw,
    },
};
use crate::runtime::{BoundaryObservation, BoundaryObserverRef, DeviceId, record_boundary_observation_for_observer};

struct SharedCudaLeaseActivation {
    _barrier: Option<crate::runtime::CudaEvent>,
}

impl ActiveLeaseActivation for SharedCudaLeaseActivation {}

#[derive(Clone)]
struct SharedCudaLeaseBackend {
    incomplete_drop_behavior: IncompleteDropBehavior,
    observation: Option<BoundaryObservation>,
    observer: Option<BoundaryObserverRef>,
}

impl ActiveLeaseBackend for SharedCudaLeaseBackend {
    fn activate(
        &self,
        export: &LeaseExportContext,
        read_start: rapids_interop::ReadStartBinding,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseActivation> {
        let producer_readiness = producer_readiness_binding_from_export(export, SHARED_ACTIVATE_LEASE_OP)?;
        let _ = completion;

        match read_start {
            rapids_interop::ReadStartBinding::SynchronousByReturn => {
                synchronize_producer_readiness(producer_readiness, SHARED_ACTIVATE_LEASE_OP)?;
                if let Some(observation) = self.observation {
                    record_boundary_observation_for_observer(self.observer.as_ref(), observation);
                }
                Ok(LeaseActivation::noop())
            },
            rapids_interop::ReadStartBinding::ConsumerStream { raw, device } => {
                let consumer_device = DeviceId::new(device as usize).map_err(|error| {
                    super::super::map_cudf_error(
                        InteropErrorKind::ReadinessUnavailable,
                        SHARED_ACTIVATE_LEASE_OP,
                        error,
                    )
                })?;
                let barrier = order_consumer_stream_after_producer_readiness_raw(
                    producer_readiness,
                    raw,
                    consumer_device,
                    SHARED_ACTIVATE_LEASE_OP,
                )?;

                if let Some(observation) = self.observation {
                    record_boundary_observation_for_observer(self.observer.as_ref(), observation);
                }

                Ok(match barrier {
                    Some(barrier) => {
                        LeaseActivation::with_state(Arc::new(SharedCudaLeaseActivation { _barrier: Some(barrier) }))
                    },
                    None => LeaseActivation::noop(),
                })
            },
        }
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.incomplete_drop_behavior
    }

    fn complete(
        &self,
        export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseCompletionResult> {
        match completion {
            CompletionBinding::SynchronousByReturn => Ok(LeaseCompletionResult::FallbackSynchronized),
            CompletionBinding::SameDeviceStream { raw, device } => {
                let producer_device = export.producer_device().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::ReadinessUnavailable,
                        "same-device stream completion requires a concrete producer CUDA device",
                    )
                    .with_context("operation", SHARED_COMPLETE_LEASE_OP)
                    .with_context("completion", "same_device_stream")
                })?;
                let producer_device = DeviceId::new(producer_device as usize).map_err(|error| {
                    super::super::map_cudf_error(
                        InteropErrorKind::ReadinessUnavailable,
                        SHARED_COMPLETE_LEASE_OP,
                        error,
                    )
                })?;
                let consumer_device = DeviceId::new(device as usize).map_err(|error| {
                    super::super::map_cudf_error(
                        InteropErrorKind::ReadinessUnavailable,
                        SHARED_COMPLETE_LEASE_OP,
                        error,
                    )
                })?;
                complete_same_device_stream_shared(producer_device, raw, consumer_device, SHARED_COMPLETE_LEASE_OP)
                    .map(LeaseCompletionResult::Completed)
            },
            CompletionBinding::CrossDeviceEvent { raw, device } => {
                let producer_readiness = producer_readiness_binding_from_export(export, SHARED_COMPLETE_LEASE_OP)?;
                let consumer_device = DeviceId::new(device as usize).map_err(|error| {
                    super::super::map_cudf_error(
                        InteropErrorKind::ReadinessUnavailable,
                        SHARED_COMPLETE_LEASE_OP,
                        error,
                    )
                })?;
                complete_cross_device_event_shared(producer_readiness, raw, consumer_device, SHARED_COMPLETE_LEASE_OP)
                    .map(LeaseCompletionResult::Completed)
            },
        }
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<()> {
        match completion {
            CompletionBinding::SynchronousByReturn => Ok(()),
            CompletionBinding::SameDeviceStream { raw, device } => {
                let device = DeviceId::new(device as usize).map_err(|error| {
                    super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })?;
                synchronize_same_device_stream_raw(raw, device, SHARED_DROP_LEASE_OP).map_err(|error| {
                    super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })
            },
            CompletionBinding::CrossDeviceEvent { raw, device } => {
                let device = DeviceId::new(device as usize).map_err(|error| {
                    super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })?;
                synchronize_cross_device_event_raw(raw, device, SHARED_DROP_LEASE_OP).map_err(|error| {
                    super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })
            },
        }
    }
}

fn shared_cuda_lease_backend_with_behavior(
    incomplete_drop_behavior: IncompleteDropBehavior,
    observation: Option<BoundaryObservation>,
    observer: Option<BoundaryObserverRef>,
) -> LeaseBackend {
    LeaseBackend::new(SharedCudaLeaseBackend { incomplete_drop_behavior, observation, observer })
}

pub(crate) fn shared_cuda_lease_backend_with_observation(
    observation: BoundaryObservation,
    observer: Option<BoundaryObserverRef>,
) -> LeaseBackend {
    shared_cuda_lease_backend_with_behavior(IncompleteDropBehavior::GuaranteedFallback, Some(observation), observer)
}

/// CUDA-backed prepared-lease backend that requires explicit completion.
///
/// This is an opt-in backend for callers that want the real CUDA runtime
/// activation and witness path but do not want abandoned leases to
/// conservatively synchronize during `drop`.
///
/// Callers must ensure that every activated lease is completed explicitly via
/// `.complete()` before the producer's underlying buffers are invalidated.
/// Dropping an incomplete lease with this backend does not synchronize, which
/// can cause GPU use-after-free if the consumer is still accessing producer
/// memory.
pub fn cuda_explicit_completion_required_lease_backend() -> LeaseBackend {
    shared_cuda_lease_backend_with_behavior(IncompleteDropBehavior::ExplicitCompletionRequired, None, None)
}
