//! Shared cudf CUDA device-entry and fail-closed teardown adapters.
//!
//! cuGraph owners do not maintain independent device health or thread poison:
//! every normal entry and destructor uses cudf's process-lifetime authority.

use std::{fmt, ptr};

use cudf::{
    DeviceId,
    runtime::{DeviceEntryGuard, drop_device_resource as cudf_drop_device_resource, quarantine_device_resource},
};

use crate::error::{Error, ErrorKind, Result};

const CUGRAPH_RS_DESTROY_SUCCESS: i32 = 0;
const CUGRAPH_RS_DESTROY_FATAL_CUDA: i32 = 2;

unsafe extern "C" {
    fn cugraph_rs_destroy_device_resource(resource: *mut std::ffi::c_void, kind: i32, cuda_status: *mut i32) -> i32;
    #[cfg(test)]
    fn cugraph_rs_test_fail_next_device_resource_destroy(status: i32, cuda_status: i32);
}

/// Typed local-shim discriminator for every device-bound cuGraph owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(crate) enum DeviceResourceKind {
    SimilarityResult = 1,
    VertexPairs = 2,
    RngState = 3,
    CoreResult = 4,
    KCoreResult = 5,
    DegreesResult = 6,
    InducedSubgraphResult = 7,
    LayoutResult = 8,
    HierarchicalClusteringResult = 9,
    TriangleCountResult = 10,
    RandomWalkResult = 11,
    SampleResult = 12,
    ClusteringResult = 13,
    Edgelist = 14,
    Graph = 15,
    BfsPredicateResult = 16,
    PathsResult = 17,
    ExtractPathsResult = 18,
    RuntimeResourceHandle = 19,
    ResourceHandle = 20,
    CentralityResult = 21,
    EdgeCentralityResult = 22,
    HitsResult = 23,
    LabelingResult = 24,
    DeviceArray = 26,
}

struct NativeDestroyFailure {
    status: i32,
    cuda_status: i32,
}

impl fmt::Display for NativeDestroyFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "native cuGraph teardown failed with status {} and CUDA status {}",
            self.status, self.cuda_status
        )
    }
}

fn map_cudf_device_error(error: cudf::Error, operation: &'static str) -> Error {
    Error::new(ErrorKind::InvalidState, operation)
        .with_message("cudf device-entry safety policy rejected cuGraph work")
        .with_context("cudf_code", error.code().to_string())
        .with_context("cudf_operation", error.operation().to_string())
        .with_context("cudf_message", error.message().to_string())
        .set_source(error)
        .set_persistent()
}

fn validated_device(device: i32, operation: &'static str) -> Result<DeviceId> {
    usize::try_from(device)
        .ok()
        .and_then(|ordinal| DeviceId::new(ordinal).ok())
        .or_else(|| cudf::testing::device_safety::injected_device_id(device))
        .ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message("invalid CUDA owner-device ordinal")
                .with_context("device", device.to_string())
        })
}

/// Convert a raw cudf status code into the crate's structured error type.
pub(crate) fn check_cudf_status(status: cudf_sys::cudf_status_t, operation: &'static str) -> Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }

    let error = match status {
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => Error::new(ErrorKind::OutOfMemory, operation)
            .with_message(format!("cudf helper failed with status {status}"))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {status}"))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_FATAL_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {status}"))
            .set_persistent(),
        _ => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {status}")),
    };
    Err(error)
}

pub(crate) fn check_cudf_status_on_device(
    status: cudf_sys::cudf_status_t,
    device: i32,
    operation: &'static str,
) -> Result<()> {
    if status == cudf_sys::CUDF_ERROR_FATAL_CUDA
        && let Ok(device) = validated_device(device, operation)
    {
        cudf::runtime::report_device_fault(
            device,
            cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
            operation,
            Some(status as i32),
        );
    }
    check_cudf_status(status, operation)
}

/// cuGraph-local wrapper around cudf's host-thread device-entry proof.
#[derive(Debug)]
pub(crate) struct DeviceGuard {
    _inner: DeviceEntryGuard,
}

impl DeviceGuard {
    /// Temporarily enter an immutable owner's CUDA device on this host thread.
    /// This never migrates a query or a device allocation.
    pub(crate) fn enter_owner_device(target: i32, operation: &'static str) -> Result<Self> {
        let device = validated_device(target, operation)?;
        let inner =
            DeviceEntryGuard::enter(device, operation).map_err(|error| map_cudf_device_error(error, operation))?;
        Ok(Self { _inner: inner })
    }
}

/// Consume one cuGraph pointer and use cudf's owner-device teardown policy.
pub(crate) fn drop_device_resource<T>(
    raw: &mut *mut T,
    device: i32,
    operation: &'static str,
    kind: DeviceResourceKind,
) {
    let Ok(device) = validated_device(device, operation) else {
        // The ordinal was captured from a previously validated owner. If it is
        // no longer valid, fail closed without crossing FFI.
        *raw = ptr::null_mut();
        return;
    };
    cudf_drop_device_resource(raw, device, operation, |owned| {
        let mut cuda_status = 0;
        // SAFETY: cudf's teardown authority consumed the unique pointer and
        // proved owner-device entry. The kind matches the pointer's native type,
        // and `cuda_status` is a valid stack out-pointer.
        let status = unsafe {
            cugraph_rs_destroy_device_resource(owned.cast::<std::ffi::c_void>(), kind as i32, &mut cuda_status)
        };
        if status == CUGRAPH_RS_DESTROY_SUCCESS {
            return;
        }
        if status == CUGRAPH_RS_DESTROY_FATAL_CUDA {
            cudf::runtime::report_device_fault(
                device,
                cudf::runtime::DeviceFaultKind::FatalDeviceOperation,
                operation,
                Some(cuda_status),
            );
        }
        quarantine_device_resource(device, operation, &NativeDestroyFailure { status, cuda_status });
    });
}

pub(crate) fn quarantine_device_drop(operation: &'static str, device: i32, reason: &dyn fmt::Display) {
    if let Ok(device) = validated_device(device, operation) {
        quarantine_device_resource(device, operation, reason);
    }
}

#[cfg(test)]
pub(crate) use cudf::testing::device_safety::{
    DeviceEntryTestCall as TestCudaCall, DeviceSafetyTestScope as TestCudaHooks,
};

#[cfg(test)]
pub(crate) fn fail_next_fatal_device_resource_destroy(cuda_status: i32) {
    // SAFETY: the native test hook stores two scalar values in thread-local
    // state and retains no Rust pointers.
    unsafe { cugraph_rs_test_fail_next_device_resource_destroy(CUGRAPH_RS_DESTROY_FATAL_CUDA, cuda_status) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_cudf_status_maps_out_of_memory_to_temporary_oom() {
        let err = check_cudf_status(cudf_sys::CUDF_ERROR_OUT_OF_MEMORY, "test_op").expect_err("expected error");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.is_temporary());
    }

    #[test]
    fn check_cudf_status_maps_fatal_cuda_to_persistent_unexpected() {
        let err = check_cudf_status(cudf_sys::CUDF_ERROR_FATAL_CUDA, "test_op").expect_err("expected error");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_persistent());
    }
}
