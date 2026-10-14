use std::sync::Arc;

use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, CompletionAuthority, CompletionWitness, InteropErrorKind,
    LeaseActivation, LeaseBackend, ProducerReadiness, ReadStartAuthority,
};

use super::{
    SHARED_ACTIVATE_LEASE_OP, SHARED_COMPLETE_LEASE_OP, SHARED_DROP_LEASE_OP,
    authorities::producer_readiness_binding,
    sync::{
        complete_same_device_stream_shared, order_consumer_stream_after_producer_readiness_raw,
        synchronize_producer_readiness, synchronize_same_device_stream_raw,
    },
};
use crate::{
    interop::map_cudf_error,
    runtime::{self, CudaEvent, DeviceId},
};

struct SharedCudaLeaseActivation {
    _barrier: Option<CudaEvent>,
}

impl ActiveLeaseActivation for SharedCudaLeaseActivation {}

#[derive(Clone)]
struct SharedCudaLeaseBackend;

// SAFETY: activation installs readiness waits on the consumer stream. Complete
// records a fence after its reads; fallback synchronizes that stream directly.
unsafe impl ActiveLeaseBackend for SharedCudaLeaseBackend {
    fn activate(
        &self,
        readiness: ProducerReadiness<'_>,
        read_start: ReadStartAuthority<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<LeaseActivation> {
        let producer_readiness = producer_readiness_binding(readiness, SHARED_ACTIVATE_LEASE_OP)?;

        match read_start {
            ReadStartAuthority::SynchronousByReturn => {
                synchronize_producer_readiness(producer_readiness, SHARED_ACTIVATE_LEASE_OP)?;
                Ok(LeaseActivation::noop())
            },
            ReadStartAuthority::ConsumerStream { stream } => {
                let consumer_device = DeviceId::new(stream.device() as usize).map_err(|error| {
                    map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_ACTIVATE_LEASE_OP, error)
                })?;
                let barrier = order_consumer_stream_after_producer_readiness_raw(
                    producer_readiness,
                    stream.as_raw(),
                    consumer_device,
                    SHARED_ACTIVATE_LEASE_OP,
                )?;

                Ok(match barrier {
                    Some(barrier) => {
                        LeaseActivation::with_state(Arc::new(SharedCudaLeaseActivation { _barrier: Some(barrier) }))
                    },
                    None => LeaseActivation::noop(),
                })
            },
        }
    }

    fn complete(
        &self,
        readiness: ProducerReadiness<'_>,
        completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<Option<CompletionWitness>> {
        match completion {
            CompletionAuthority::SynchronousByReturn => Ok(None),
            CompletionAuthority::SameDeviceStream { stream } => {
                let producer_device = DeviceId::new(readiness.device() as usize).map_err(|error| {
                    map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETE_LEASE_OP, error)
                })?;
                let consumer_device = DeviceId::new(stream.device() as usize).map_err(|error| {
                    map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETE_LEASE_OP, error)
                })?;
                complete_same_device_stream_shared(
                    producer_device,
                    stream.as_raw(),
                    consumer_device,
                    SHARED_COMPLETE_LEASE_OP,
                )
                .map(Some)
            },
        }
    }

    fn drop_incomplete(
        &self,
        _readiness: ProducerReadiness<'_>,
        completion: CompletionAuthority<'_>,
    ) -> rapids_interop::Result<()> {
        match completion {
            CompletionAuthority::SynchronousByReturn => Ok(()),
            CompletionAuthority::SameDeviceStream { stream } => {
                let device = DeviceId::new(stream.device() as usize).map_err(|error| {
                    map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })?;
                synchronize_same_device_stream_raw(stream.as_raw(), device, SHARED_DROP_LEASE_OP).map_err(|error| {
                    runtime::quarantine_device_resource(device, SHARED_DROP_LEASE_OP, &error);
                    map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })
            },
        }
    }
}

pub(crate) fn shared_cuda_lease_backend() -> LeaseBackend {
    LeaseBackend::new(SharedCudaLeaseBackend)
}
