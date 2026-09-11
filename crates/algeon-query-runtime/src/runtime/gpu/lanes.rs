//! Per-device attempt lane identity.
//!
//! The bounded controller admits at most `A_d` concurrent attempts per
//! device. Each admitted attempt holds exactly one lane index for its whole
//! lifetime. The lane index is identity evidence: it binds the attempt's
//! ledger slot, root stream, allocation domain, and worker scope to one
//! observable position in the device's attempt array.
//!
//! Lane acquisition is non-blocking by construction: the ledger admits an
//! attempt only while `active_attempts < A_d`, so a free lane always exists
//! for a freshly granted attempt. A missing lane therefore indicates a
//! lifecycle bug and surfaces as a structured internal error, never as a
//! wait.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

/// Per-device array of `A_d` attempt lanes.
#[derive(Debug)]
#[doc(hidden)]
pub struct DeviceLanePermits {
    device_ordinal: usize,
    lanes: Box<[AtomicBool]>,
}

/// One held attempt lane. Dropping the permit returns the lane to the array.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct DeviceLanePermit {
    lanes: Arc<DeviceLanePermits>,
    index: usize,
}

/// Identity evidence binding an admitted attempt to its device worker pool
/// and lane. Non-blocking by design: every field is already known once the
/// admission transaction holds the ledger grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct DeviceWorkerScope {
    device_ordinal: usize,
    pool_id: u64,
    lane_index: usize,
}

impl DeviceLanePermits {
    #[doc(hidden)]
    pub fn new(device_ordinal: usize, lane_count: usize) -> Self {
        Self { device_ordinal, lanes: (0..lane_count.max(1)).map(|_| AtomicBool::new(false)).collect() }
    }

    /// Takes the first free lane. Returns a structured error when every lane
    /// is held; the ledger's `active_attempts <= A_d` invariant makes that
    /// unreachable for attempts admitted through the controller.
    #[doc(hidden)]
    pub fn acquire(self: &Arc<Self>) -> Result<DeviceLanePermit> {
        for (index, held) in self.lanes.iter().enumerate() {
            if held.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return Ok(DeviceLanePermit { lanes: Arc::clone(self), index });
            }
        }
        Err(QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.gpu.device_lanes",
            "device attempt lanes exhausted while the ledger admitted the attempt",
        )
        .with_fact("device_ordinal", self.device_ordinal)
        .with_fact("lane_count", self.lanes.len())
        .with_fact("reason", "lane_exhausted"))
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn held_count(&self) -> usize {
        self.lanes.iter().filter(|held| held.load(Ordering::Acquire)).count()
    }

    fn release(&self, index: usize) {
        if let Some(held) = self.lanes.get(index) {
            held.store(false, Ordering::Release);
        }
    }
}

impl DeviceLanePermit {
    #[doc(hidden)]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl Drop for DeviceLanePermit {
    fn drop(&mut self) {
        self.lanes.release(self.index);
    }
}

impl DeviceWorkerScope {
    #[doc(hidden)]
    pub fn new(device_ordinal: usize, pool_id: u64, lane_index: usize) -> Self {
        Self { device_ordinal, pool_id, lane_index }
    }

    #[doc(hidden)]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    #[doc(hidden)]
    pub fn pool_id(&self) -> u64 {
        self.pool_id
    }

    #[doc(hidden)]
    pub fn lane_index(&self) -> usize {
        self.lane_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_are_handled_out_first_free_and_returned_on_drop() {
        let lanes = Arc::new(DeviceLanePermits::new(0, 2));
        let first = lanes.acquire().unwrap();
        let second = lanes.acquire().unwrap();
        assert_eq!(first.index(), 0);
        assert_eq!(second.index(), 1);
        assert!(lanes.acquire().is_err(), "third acquire exceeds the lane array");
        drop(first);
        let third = lanes.acquire().unwrap();
        assert_eq!(third.index(), 0, "released lane is reused");
    }

    #[test]
    fn exhaustion_is_a_structured_internal_error() {
        let lanes = Arc::new(DeviceLanePermits::new(3, 1));
        let _held = lanes.acquire().unwrap();
        let error = lanes.acquire().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.fact_value("reason"), Some("lane_exhausted"));
        assert_eq!(error.fact_value("device_ordinal"), Some("3"));
    }

    #[test]
    fn worker_scope_carries_identity() {
        let scope = DeviceWorkerScope::new(2, 41, 1);
        assert_eq!(scope.device_ordinal(), 2);
        assert_eq!(scope.pool_id(), 41);
        assert_eq!(scope.lane_index(), 1);
    }
}
