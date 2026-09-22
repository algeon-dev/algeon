use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

use tokio::task::JoinHandle;

use super::*;

/// Runner-local facts captured at the close boundary. The evidence
/// reducer combines these with sealed attempt outcomes; nothing here infers
/// queue membership from runner ordering.
pub(super) struct CloseUnderLoadBarrier {
    pub(super) close_requested_offset_ns: u64,
    pub(super) close_initiated: bool,
    pub(super) admitted_at_close: usize,
    pub(super) queued_at_close: usize,
    pub(super) in_flight_at_close: usize,
    pub(super) in_flight_sequences: BTreeSet<usize>,
    pub(super) submitted_attempts: u64,
}

/// The per-attempt deadline/cancellation storm knobs, resolved from the arrival plan once.
#[derive(Clone, Copy, Default)]
pub(super) struct StormKnobs {
    pub(super) attempt_deadline_ms: Option<u64>,
    pub(super) cancel_every_n: Option<usize>,
    pub(super) cancel_after_ms: Option<u64>,
}

impl StormKnobs {
    pub(super) fn of(plan: &ArrivalPlan) -> Self {
        Self {
            attempt_deadline_ms: plan.attempt_deadline_ms,
            cancel_every_n: plan.cancel_every_n,
            cancel_after_ms: plan.cancel_after_ms,
        }
    }
}

/// Runner-local facts sampled around a synchronous controlled quarantine.
/// `QuarantineFallbackEvidence` turns these plus sealed reports into the
/// versioned artifact projection after all tasks have drained.
pub(super) struct QuarantineBarrier {
    pub(super) target_ordinal: usize,
    pub(super) pre_ack_witness_offset_ns: u64,
    pub(super) pre_ack_ledger_sequence: u64,
    pub(super) pre_ack_live_grants: usize,
    pub(super) pre_ack_active_query_jobs: usize,
    pub(super) acknowledgement_offset_ns: u64,
    pub(super) device_health_generation: u64,
    pub(super) newly_quarantined: bool,
    pub(super) target_ledger_closed: bool,
    pub(super) post_ack_sequences: BTreeSet<usize>,
    pub(super) pre_existing_target_work_released: bool,
    pub(super) pre_ack_cache: Option<serde_json::Value>,
    pub(super) post_ack_cache: Option<serde_json::Value>,
    pub(super) final_cache: Option<serde_json::Value>,
}

/// Runner-local lease facts around the one physical-pressure probe. The
/// reducer combines these bounded CUDA snapshots with sealed attempt reports;
/// it never treats the sidecar itself as a synthetic allocator error.
pub(super) struct PhysicalAllocatorPressureBarrier {
    pub(super) pressure: PhysicalPressureLeaseFacts,
    pub(super) release: PhysicalPressureLeaseFacts,
    pub(super) pressure_established_before_q01: bool,
    pub(super) pressure_released_before_recovery: bool,
}

pub(super) struct PendingCancellationBarrier {
    pub(super) attempt_sequence: usize,
    pub(super) requested_after: Duration,
    pub(super) requested: Arc<AtomicBool>,
    pub(super) task: Option<JoinHandle<()>>,
}
