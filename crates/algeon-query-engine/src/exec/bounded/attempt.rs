use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use query_runtime::observability::SourceReadFact;

use super::{BoundedPartitionTargetPolicy, ExecutionContext, ExecutionMetrics};

/// One admitted bounded execution attempt.
///
/// The immutable cap is a batching input. Allocation enforcement and resident
/// ownership belong to the query memory domain and grant.
///
/// This is a borrowed view, cheap to build: the coordinator holds one over the
/// root context, and every lane job builds its own over the lane's derived
/// context so operators run on the lane's stream while reporting into the
/// same metrics owner.
pub(in crate::exec) struct BoundedExecutionAttempt<'a> {
    context: &'a ExecutionContext,
    /// Cap applied to every budget token created under this attempt. Tokens
    /// label stages for diagnostics; the allocator member enforces the grant.
    device_budget_bytes: u64,
    metrics: &'a ExecutionMetrics,
    /// Present on the coordinator's attempt only. Lane jobs probe tables the
    /// coordinator retained and never retain their own.
    retained_join_tables: Option<Arc<JoinRetentionLedger>>,
    /// Probe slices that partitioned hash joins defer for partitions without a
    /// kept table. Present on the coordinator's attempt only, like the tables.
    deferred_join_probes: Option<Arc<JoinRetentionLedger>>,
}

impl<'a> BoundedExecutionAttempt<'a> {
    pub(in crate::exec::bounded) fn new(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a ExecutionMetrics,
    ) -> Self {
        let policy = BoundedPartitionTargetPolicy::default();
        Self {
            context,
            device_budget_bytes,
            metrics,
            retained_join_tables: Some(JoinRetentionLedger::new(
                policy.partitioned_join_retained_hash_table_bytes(device_budget_bytes),
            )),
            deferred_join_probes: Some(JoinRetentionLedger::new(
                policy.partitioned_join_target_bytes(device_budget_bytes),
            )),
        }
    }

    /// The attempt of one lane job. It reports into the coordinator's metrics
    /// and cannot retain partitioned join hash tables or deferred probe slices.
    pub(in crate::exec::bounded) fn for_lane(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a ExecutionMetrics,
    ) -> Self {
        Self { context, device_budget_bytes, metrics, retained_join_tables: None, deferred_join_probes: None }
    }

    pub(in crate::exec) const fn context(&self) -> &'a ExecutionContext {
        self.context
    }

    /// Where this attempt reports its metrics. The attempt holds no metrics of
    /// its own; the plan execution's observation owner is the only store.
    pub(in crate::exec) const fn metrics(&self) -> &'a ExecutionMetrics {
        self.metrics
    }

    /// Folds one source-read fact into this attempt's plan execution.
    pub(in crate::exec) fn record_source_read(&self, fact: SourceReadFact) {
        self.metrics.record_source_read(fact);
    }

    pub(in crate::exec) const fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// Charges `bytes` of a partition hash table that a partitioned hash join
    /// keeps across probe pieces. Returns `None` when the charge would take the
    /// attempt's retained total above its cap, and always on a lane attempt.
    /// The cap only limits what is retained; the query domain's allocator
    /// still decides whether the table can be allocated.
    pub(in crate::exec::bounded) fn charge_retained_join_table(&self, bytes: u64) -> Option<RetainedJoinTableCharge> {
        let ledger = self.retained_join_tables.as_ref()?;
        ledger.try_grow(bytes).then(|| RetainedJoinTableCharge { ledger: Arc::clone(ledger), bytes })
    }

    /// An empty charge for the probe slices that one partitioned hash join
    /// defers, against the attempt's deferred total. `None` on a lane attempt.
    pub(in crate::exec::bounded) fn deferred_join_probe_charge(&self) -> Option<DeferredJoinProbeCharge> {
        let ledger = self.deferred_join_probes.as_ref()?;
        Some(DeferredJoinProbeCharge { ledger: Arc::clone(ledger), bytes: 0 })
    }
}

/// Bytes that the attempt's partitioned hash joins hold across probe pieces
/// in one category, against that category's cap.
struct JoinRetentionLedger {
    cap_bytes: u64,
    held_bytes: AtomicU64,
}

impl JoinRetentionLedger {
    fn new(cap_bytes: u64) -> Arc<Self> {
        Arc::new(Self { cap_bytes, held_bytes: AtomicU64::new(0) })
    }

    fn try_grow(&self, bytes: u64) -> bool {
        self.held_bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                held.checked_add(bytes).filter(|total| *total <= self.cap_bytes)
            })
            .is_ok()
    }

    fn shrink(&self, bytes: u64) {
        self.held_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// One retained table's share of the attempt's retained total, returned to
/// the ledger when the charge drops with the table.
pub(in crate::exec::bounded) struct RetainedJoinTableCharge {
    ledger: Arc<JoinRetentionLedger>,
    bytes: u64,
}

impl Drop for RetainedJoinTableCharge {
    fn drop(&mut self) {
        self.ledger.shrink(self.bytes);
    }
}

/// The probe slices one partitioned hash join defers, as a share of the
/// attempt's deferred total that
/// `BoundedPartitionTargetPolicy::partitioned_join_target_bytes` caps. It
/// resizes as slices are deferred and joined, and returns to the ledger when
/// it drops.
pub(in crate::exec::bounded) struct DeferredJoinProbeCharge {
    ledger: Arc<JoinRetentionLedger>,
    bytes: u64,
}

impl DeferredJoinProbeCharge {
    /// The largest size this charge can take now: the cap less what the
    /// attempt's other joins defer.
    pub(in crate::exec::bounded) fn limit_bytes(&self) -> u64 {
        let others = self.ledger.held_bytes.load(Ordering::Relaxed).saturating_sub(self.bytes);
        self.ledger.cap_bytes.saturating_sub(others)
    }

    /// Resizes the charge to `bytes`. Shrinking always succeeds; growing
    /// fails, leaving the charge as it was, when it would pass the cap.
    pub(in crate::exec::bounded) fn try_resize(&mut self, bytes: u64) -> bool {
        if bytes <= self.bytes {
            self.ledger.shrink(self.bytes - bytes);
        } else if !self.ledger.try_grow(bytes - self.bytes) {
            return false;
        }
        self.bytes = bytes;
        true
    }
}

impl Drop for DeferredJoinProbeCharge {
    fn drop(&mut self) {
        self.ledger.shrink(self.bytes);
    }
}
