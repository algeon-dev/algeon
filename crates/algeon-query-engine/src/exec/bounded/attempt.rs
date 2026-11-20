use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use query_runtime::observability::SourceReadFact;

use super::{BoundedPartitionTargetPolicy, ExecutionContext, ExecutionMetrics, MemoryControlPolicy};

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
    /// State the attempt keeps resident while other input streams past it: the
    /// hash tables partitioned hash joins keep across probe pieces and the
    /// input slices and partial states grouped aggregates hold before merging
    /// them. Its cap is the retained-state share of [`MemoryControlPolicy`].
    /// Present on the coordinator's attempt only. Lane jobs probe tables the
    /// coordinator retained and hand their aggregate slices to the coordinator.
    retained_state: Option<Arc<RetentionLedger>>,
    /// Probe slices that partitioned hash joins defer for partitions without a
    /// kept table. Present on the coordinator's attempt only, like the tables.
    deferred_join_probes: Option<Arc<RetentionLedger>>,
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
            retained_state: Some(RetentionLedger::new(MemoryControlPolicy::retained_state_share_bytes(
                device_budget_bytes,
            ))),
            deferred_join_probes: Some(RetentionLedger::new(policy.partitioned_join_target_bytes(device_budget_bytes))),
        }
    }

    /// The attempt of one lane job. It reports into the coordinator's metrics
    /// and cannot retain partitioned join hash tables, deferred probe slices or
    /// aggregate holds.
    pub(in crate::exec::bounded) fn for_lane(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a ExecutionMetrics,
    ) -> Self {
        Self { context, device_budget_bytes, metrics, retained_state: None, deferred_join_probes: None }
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
    /// attempt's retained-state total above its cap, and always on a lane
    /// attempt. The cap only limits what is retained; the query domain's
    /// allocator still decides whether the table can be allocated.
    pub(in crate::exec::bounded) fn charge_retained_join_table(&self, bytes: u64) -> Option<RetainedJoinTableCharge> {
        let ledger = self.retained_state.as_ref()?;
        ledger.try_grow(bytes).then(|| RetainedJoinTableCharge { ledger: Arc::clone(ledger), bytes })
    }

    /// An empty charge for the slices that one grouped aggregate holds instead
    /// of merging them, against the attempt's retained-state total. `None` on a
    /// lane attempt.
    pub(in crate::exec::bounded) fn retained_state_charge(&self) -> Option<RetentionCharge> {
        let ledger = self.retained_state.as_ref()?;
        Some(RetentionCharge { ledger: Arc::clone(ledger), bytes: 0 })
    }

    /// An empty charge for the probe slices that one partitioned hash join
    /// defers, against the attempt's deferred total. `None` on a lane attempt.
    pub(in crate::exec::bounded) fn deferred_join_probe_charge(&self) -> Option<RetentionCharge> {
        let ledger = self.deferred_join_probes.as_ref()?;
        Some(RetentionCharge { ledger: Arc::clone(ledger), bytes: 0 })
    }
}

/// Bytes that the attempt holds across stream pieces in one category, against
/// that category's cap.
struct RetentionLedger {
    cap_bytes: u64,
    held_bytes: AtomicU64,
}

impl RetentionLedger {
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
    ledger: Arc<RetentionLedger>,
    bytes: u64,
}

impl Drop for RetainedJoinTableCharge {
    fn drop(&mut self) {
        self.ledger.shrink(self.bytes);
    }
}

/// One holder's share of a ledger whose size follows what it holds: the
/// probe slices one partitioned hash join defers, against the deferred total
/// that `BoundedPartitionTargetPolicy::partitioned_join_target_bytes` caps, or
/// the slices one grouped aggregate holds, against the retained-state total.
/// It resizes as slices are held and released, and returns to the ledger when
/// it drops.
pub(in crate::exec::bounded) struct RetentionCharge {
    ledger: Arc<RetentionLedger>,
    bytes: u64,
}

impl RetentionCharge {
    /// The largest size this charge can take now: the cap less what the
    /// attempt's other holders charge.
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

impl Drop for RetentionCharge {
    fn drop(&mut self) {
        self.ledger.shrink(self.bytes);
    }
}
