//! GPU-resident cache ownership and lifecycle contracts.
//!
//! Cache implementations own their keys and GPU objects. This module owns the
//! one state machine that makes their byte residency visible to the per-device
//! ledger.

pub(crate) mod factory;
pub(crate) mod local_parquet;
pub(crate) mod residency;
#[cfg(feature = "cugraph")]
pub(crate) mod resident_graph;

pub(crate) use factory::DeviceCacheAllocationFactory;
#[allow(unused_imports)]
pub(crate) use residency::{
    CacheResidency, CacheResidencyChargeTransaction, CacheResidencyDestruction, CacheResidencyPin, CacheResidencyState,
};
