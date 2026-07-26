//! Unstable injected CUDA device-entry and health support for contract tests.

use std::{cell::RefCell, collections::VecDeque, marker::PhantomData, sync::Arc};

use crate::runtime::{
    DeviceHealthSnapshot, DeviceId,
    device_health::{DeviceHealthRegistry, new_test_registry},
};

/// One injected host-thread CUDA device API result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceEntryTestCall {
    GetDevice { status: cudf_sys::cudf_status_t, device: i32 },
    SetDevice { expected_target: i32, status: cudf_sys::cudf_status_t },
}

struct InjectedDeviceSafety {
    registry: Arc<DeviceHealthRegistry>,
    calls: VecDeque<DeviceEntryTestCall>,
}

thread_local! {
    static INJECTED: RefCell<Option<InjectedDeviceSafety>> = const { RefCell::new(None) };
}

/// Same-thread scope with an isolated health registry and injected get/set
/// results. Dropping the scope removes both, so fault tests do not poison the
/// process-global registry used by unrelated tests.
pub struct DeviceSafetyTestScope {
    registry: Arc<DeviceHealthRegistry>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl DeviceSafetyTestScope {
    /// Install one isolated same-thread test registry and call sequence.
    pub fn install(calls: impl IntoIterator<Item = DeviceEntryTestCall>) -> Self {
        crate::runtime::device_guard::reset_thread_authority_for_testing();
        let registry = new_test_registry();
        INJECTED.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "device safety test scope is already installed on this thread");
            *slot = Some(InjectedDeviceSafety { registry: Arc::clone(&registry), calls: calls.into_iter().collect() });
        });
        Self { registry, _not_send_sync: PhantomData }
    }

    /// Read health from this scope's isolated registry.
    pub fn health_snapshot(&self, device: DeviceId) -> DeviceHealthSnapshot {
        self.registry.state(device).snapshot(device)
    }

    /// Number of injected get/set results not yet consumed.
    pub fn remaining_call_count(&self) -> usize {
        INJECTED.with(|slot| slot.borrow().as_ref().map_or(0, |injected| injected.calls.len()))
    }

    /// Number of device-owned pointers quarantined without owner provenance.
    pub fn unknown_device_quarantined_count(&self) -> u64 {
        self.registry.unknown_device_quarantined_count()
    }
}

impl Drop for DeviceSafetyTestScope {
    fn drop(&mut self) {
        INJECTED.with(|slot| {
            slot.borrow_mut().take();
        });
        crate::runtime::device_guard::reset_thread_authority_for_testing();
    }
}

pub(crate) fn injected_health_registry() -> Option<Arc<DeviceHealthRegistry>> {
    INJECTED.with(|slot| slot.borrow().as_ref().map(|injected| Arc::clone(&injected.registry)))
}

/// Build an ordinal for an installed injected test scope without consulting
/// physical device inventory. Returns `None` outside such a scope.
pub fn injected_device_id(ordinal: i32) -> Option<DeviceId> {
    if ordinal < 0 || injected_health_registry().is_none() {
        return None;
    }
    Some(DeviceId::from_validated_ordinal_raw(ordinal))
}

pub(crate) fn cuda_get_device(current: &mut i32) -> cudf_sys::cudf_status_t {
    let injected = INJECTED.with(|slot| {
        let mut slot = slot.borrow_mut();
        let injected = slot.as_mut()?;
        match injected.calls.pop_front()? {
            DeviceEntryTestCall::GetDevice { status, device } => {
                if status == cudf_sys::CUDF_SUCCESS {
                    *current = device;
                }
                Some(status)
            },
            unexpected => panic!("expected CUDA get-device call, got {unexpected:?}"),
        }
    });
    injected.unwrap_or_else(|| {
        // SAFETY: `current` is a valid out pointer for the duration of this call.
        unsafe { cudf_sys::cudf_cuda_get_device(current) }
    })
}

pub(crate) fn cuda_set_device(target: i32) -> cudf_sys::cudf_status_t {
    let injected = INJECTED.with(|slot| {
        let mut slot = slot.borrow_mut();
        let injected = slot.as_mut()?;
        match injected.calls.pop_front()? {
            DeviceEntryTestCall::SetDevice { expected_target, status } => {
                assert_eq!(target, expected_target, "unexpected CUDA device-entry target");
                Some(status)
            },
            unexpected => panic!("expected CUDA set-device call, got {unexpected:?}"),
        }
    });
    injected.unwrap_or_else(|| {
        // SAFETY: the caller validated the target device ordinal.
        unsafe { cudf_sys::cudf_cuda_set_device(target) }
    })
}
