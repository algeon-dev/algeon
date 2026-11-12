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
    retained_join_tables: Option<Arc<RetainedJoinTableLedger>>,
}

impl<'a> BoundedExecutionAttempt<'a> {
    pub(in crate::exec::bounded) fn new(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a ExecutionMetrics,
    ) -> Self {
        let retained_cap_bytes =
            BoundedPartitionTargetPolicy::default().partitioned_join_retained_hash_table_bytes(device_budget_bytes);
        Self {
            context,
            device_budget_bytes,
            metrics,
            retained_join_tables: Some(Arc::new(RetainedJoinTableLedger {
                cap_bytes: retained_cap_bytes,
                retained_bytes: AtomicU64::new(0),
            })),
        }
    }

    /// The attempt of one lane job. It reports into the coordinator's metrics
    /// and cannot retain partitioned join hash tables.
    pub(in crate::exec::bounded) fn for_lane(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a ExecutionMetrics,
    ) -> Self {
        Self { context, device_budget_bytes, metrics, retained_join_tables: None }
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
        ledger
            .retained_bytes
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |retained| {
                retained.checked_add(bytes).filter(|total| *total <= ledger.cap_bytes)
            })
            .ok()?;
        Some(RetainedJoinTableCharge { ledger: Arc::clone(ledger), bytes })
    }
}

/// Bytes of partition hash tables retained by the attempt's partitioned hash
/// joins, against the share of the attempt's cap that
/// `BoundedPartitionTargetPolicy::partitioned_join_retained_hash_table_bytes`
/// sets.
struct RetainedJoinTableLedger {
    cap_bytes: u64,
    retained_bytes: AtomicU64,
}

/// One retained table's share of the attempt's retained total, returned to
/// the ledger when the charge drops with the table.
pub(in crate::exec::bounded) struct RetainedJoinTableCharge {
    ledger: Arc<RetainedJoinTableLedger>,
    bytes: u64,
}

impl Drop for RetainedJoinTableCharge {
    fn drop(&mut self) {
        self.ledger.retained_bytes.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}
