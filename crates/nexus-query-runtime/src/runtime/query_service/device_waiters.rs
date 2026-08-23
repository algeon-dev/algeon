//! Process-wide device-slot waiter registration and wakeups.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use nexus_query_planning::error::{QueryError, QueryErrorCode, Result};

use super::service::QueryServiceInner;

pub(super) fn register_shared_device_admission_waiters(inner: &Arc<QueryServiceInner>) -> Result<()> {
    let Some(resources) = &inner.resources else {
        return Ok(());
    };
    let mut waiters = shared_device_admission_waiters().lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.query_service.admission_waiters",
            "shared device admission waiter registry mutex was poisoned",
        )
    })?;
    for ordinal in resources.devices.device_ordinals() {
        let waiters_for_device = waiters.entry(*ordinal).or_default();
        prune_dead_shared_device_admission_waiters(waiters_for_device);
        waiters_for_device.push(Arc::downgrade(inner));
    }
    Ok(())
}

pub(super) fn shared_device_admission_waiters() -> &'static Mutex<HashMap<usize, Vec<Weak<QueryServiceInner>>>> {
    static WAITERS: OnceLock<Mutex<HashMap<usize, Vec<Weak<QueryServiceInner>>>>> = OnceLock::new();
    WAITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn prune_dead_shared_device_admission_waiters(waiters_for_device: &mut Vec<Weak<QueryServiceInner>>) {
    waiters_for_device.retain(|weak| weak.strong_count() > 0);
}

pub(super) fn notify_shared_device_admission_available(device_ordinal: usize) {
    let services = match shared_device_admission_waiters().lock() {
        Ok(mut waiters) => {
            let Some(waiters_for_device) = waiters.get_mut(&device_ordinal) else {
                return;
            };
            let mut services = Vec::with_capacity(waiters_for_device.len());
            waiters_for_device.retain(|weak| {
                if let Some(service) = weak.upgrade() {
                    services.push(service);
                    true
                } else {
                    false
                }
            });
            services
        },
        Err(_error) => {
            eprintln!(
                "[nexus-query-engine] warning: failed to lock shared device admission waiter registry for wakeup"
            );
            return;
        },
    };
    for service in services {
        let Ok(_guard) = service.admission.lock() else {
            eprintln!("[nexus-query-engine] warning: failed to lock query service admission for shared device wakeup");
            continue;
        };
        service.notify_admission_available();
    }
}

#[derive(Debug, Default)]
pub(super) struct PendingDeviceSlotNotifications {
    device_ordinals: Vec<usize>,
}

impl PendingDeviceSlotNotifications {
    /// Records a device-health transition while the caller still owns an
    /// admission mutex. Notification is deferred until that mutex is dropped
    /// because shared waiters also lock their service admission state.
    pub(super) fn record_device_health_transition(&mut self, device_ordinal: usize) {
        self.device_ordinals.push(device_ordinal);
    }

    pub(super) fn has_pending(&self) -> bool {
        !self.device_ordinals.is_empty()
    }

    pub(super) fn notify(&mut self) {
        self.device_ordinals.sort_unstable();
        self.device_ordinals.dedup();
        for device_ordinal in self.device_ordinals.drain(..) {
            notify_shared_device_admission_available(device_ordinal);
        }
    }
}

impl Drop for PendingDeviceSlotNotifications {
    fn drop(&mut self) {
        self.notify();
    }
}
