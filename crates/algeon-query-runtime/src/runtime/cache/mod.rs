//! GPU-resident cache ownership and lifecycle contracts.
//!
//! Cache implementations own their keys and GPU objects. This module owns the
//! one state machine that makes their byte residency visible to the per-device
//! ledger.

use serde::Serialize;

pub(crate) mod factory;
#[doc(hidden)]
pub mod local_parquet;
pub(crate) mod residency;
#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub mod resident_graph;

pub(crate) use factory::DeviceCacheAllocationFactory;
pub(crate) use residency::{
    CacheResidency, CacheResidencyChargeTransaction, CacheResidencyDestruction, CacheResidencyPin, CacheResidencyState,
};

/// Bounded, cache-owner event observation for one source-cache operation.
///
/// The source cache emits these from the same state machine that owns load
/// claims and resident copies.  They are deliberately diagnostic facts, not a
/// second residency ledger: byte authority remains the paired device ledger.
/// Both caller correlation and canonical source-key identity are one-way
/// fingerprints so this public observation cannot disclose query identifiers
/// or source locations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CacheSourceEventSnapshot {
    /// Per-domain monotonically increasing event sequence.
    pub(crate) sequence: u64,
    pub(crate) device_ordinal: usize,
    pub(crate) allocation_domain_id: u64,
    /// Stable one-way fingerprint of the attempt's caller correlation, when
    /// the execution context carried one.
    pub(crate) correlation_fingerprint: Option<String>,
    /// Stable one-way fingerprint of the exact canonical source-cache key.
    pub(crate) cache_key_fingerprint: String,
    /// Cache-state-machine action such as `load_leader`, `load_follower`,
    /// `load_hit`, or `promotion_inserted`.
    pub(crate) action: &'static str,
    /// Bytes retained by an inserted source-cache promotion, if applicable.
    pub(crate) retained_device_bytes: u64,
}

/// Produces the credential-free correlation token used by cache-owner event
/// observations.  The same public function lets callers match an event to a
/// correlation they supplied without exposing the original string.
pub fn cache_observation_fingerprint(value: &str) -> String {
    format!("b3:{}", blake3::hash(value.as_bytes()).to_hex())
}

/// Public, point-in-time observation of one device's resident-cache domain.
///
/// The physical fields are sampled directly from cuDF's one shared allocation
/// domain.  The activity fields are cache-owned event counters, not a second
/// byte ledger: resident-byte authority remains the device ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCacheReport {
    pub device_ordinal: usize,
    pub allocation_domain_id: u64,
    pub cache_cap_bytes: u64,
    /// The canonical ledger sequence sampled for this physical-domain report.
    pub ledger_sequence: u64,
    pub ledger_evictable_resident_bytes: u64,
    pub ledger_pinned_in_use_bytes: u64,
    pub(crate) allocation_limit_bytes: u64,
    pub physical_current_bytes: u64,
    pub(crate) physical_current_aligned_bytes: u64,
    pub physical_peak_bytes: u64,
    pub(crate) physical_total_bytes: u64,
    pub(crate) physical_total_allocations: u64,
    pub(crate) source: CacheSourceActivity,
    pub graph: CacheGraphActivity,
}

/// Cumulative source-cache activity and point-in-time key counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CacheSourceActivity {
    pub(crate) entry_count: usize,
    pub(crate) promotions_in_progress: usize,
    pub(crate) lookup_hits: u64,
    pub(crate) lookup_misses: u64,
    pub(crate) promotions_inserted: u64,
    pub(crate) promotions_coalesced: u64,
    pub(crate) promotions_capacity_skipped: u64,
    pub(crate) promotions_allocation_failed: u64,
    pub(crate) promoted_device_bytes: u64,
    pub(crate) eviction_batches: u64,
    pub(crate) evicted_device_bytes: u64,
}

/// Cumulative graph-cache activity and point-in-time key counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CacheGraphActivity {
    pub available_entries: usize,
    pub checked_out_entries: usize,
    pub preparation_pins: usize,
    pub lookup_hits: u64,
    pub(crate) lookup_misses: u64,
    pub insertions: u64,
    pub(crate) duplicate_inserts: u64,
}
