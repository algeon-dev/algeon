//! Accounted lifecycle for one GPU-resident cache entry.
//!
//! A cache implementation owns its concrete GPU object, but not an independent
//! byte counter. The only legal lifecycle is:
//!
//! ```text
//! AttemptOwned -> Evictable <-> Pinned -> Destroyed
//! ```
//!
//! The initial cache charge is held by a linear transaction. Dropping that
//! transaction rolls it back; destruction retains its charge until the caller
//! has completed GPU-safe native destruction and explicitly commits the final
//! release.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::gpu::DeviceLedger,
};

const OPERATION: &str = "nexus_query_engine.runtime.cache.residency";

/// Logical ownership state of one cache entry's resident GPU bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum CacheResidencyState {
    /// Bytes remain owned by their producing attempt while retention is being
    /// prepared. The matching charge transaction may be rolled back.
    #[cfg(test)]
    AttemptOwned,
    /// The cache owns the bytes and may evict the entry.
    Evictable,
    /// One or more cache clients use the entry, so eviction is forbidden.
    Pinned,
    /// The cache entry is no longer reusable. Its ledger charge has either
    /// been released after destruction or was deliberately retained while a
    /// quarantined device remains closed.
    Destroyed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheResidencyChargeLocation {
    Evictable,
    Pinned,
    Released,
}

#[derive(Debug)]
struct CacheResidencyInnerState {
    state: CacheResidencyState,
    charge_location: CacheResidencyChargeLocation,
    pin_count: usize,
    eviction_in_progress: bool,
}

struct CacheResidencyInner {
    ledger: DeviceLedger,
    bytes: u64,
    state: Mutex<CacheResidencyInnerState>,
}

impl CacheResidencyInner {
    fn lock(&self, action: &'static str) -> Result<MutexGuard<'_, CacheResidencyInnerState>> {
        self.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "resident-cache state mutex was poisoned")
                .with_fact("action", action)
                .with_fact("reason", "cache_residency_mutex_poisoned")
        })
    }
}

/// Linear pre-retention cache charge.
///
/// It represents the `AttemptOwned` stage and is intentionally non-cloneable.
/// Its `Drop` path returns the exact pre-reserved evictable charge once unless
/// it has been consumed by [`Self::commit_evictable`].
#[must_use = "dropping a cache charge transaction rolls its reservation back; commit it only after retention succeeds"]
#[doc(hidden)]
pub struct CacheResidencyChargeTransaction {
    ledger: DeviceLedger,
    bytes: u64,
    charged: bool,
}

impl CacheResidencyChargeTransaction {
    /// State represented before the concrete cache entry accepts this charge.
    #[cfg(test)]
    #[doc(hidden)]
    pub const fn state(&self) -> CacheResidencyState {
        CacheResidencyState::AttemptOwned
    }

    /// Exact bytes reserved for the prospective resident entry.
    #[cfg(all(test, feature = "cugraph"))]
    #[doc(hidden)]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Reconciles a conservative pre-reservation down to the exact retained
    /// allocation before the entry becomes evictable.
    ///
    /// Growing after a physical copy would introduce an undercharged window,
    /// so callers must reserve a conservative upper bound. If the observed
    /// allocation is larger, destroy it while this transaction remains live
    /// and then roll the transaction back.
    #[doc(hidden)]
    pub fn shrink_to(&mut self, retained_bytes: u64) -> Result<bool> {
        if retained_bytes > self.bytes {
            return Ok(false);
        }
        let released_bytes = self.bytes - retained_bytes;
        self.ledger.release_cache_residency_precharge_excess(released_bytes)?;
        self.bytes = retained_bytes;
        Ok(true)
    }

    /// Consumes this linear transaction after the cache has safely retained
    /// the concrete GPU object.
    #[doc(hidden)]
    pub fn commit_evictable(mut self) -> CacheResidency {
        self.charged = false;
        CacheResidency {
            inner: Arc::new(CacheResidencyInner {
                ledger: self.ledger.clone(),
                bytes: self.bytes,
                state: Mutex::new(CacheResidencyInnerState {
                    state: CacheResidencyState::Evictable,
                    charge_location: CacheResidencyChargeLocation::Evictable,
                    pin_count: 0,
                    eviction_in_progress: false,
                }),
            }),
        }
    }

    /// Explicitly rolls this prospective charge back before cache retention.
    #[doc(hidden)]
    pub fn rollback(mut self) -> Result<()> {
        self.rollback_inner()
    }

    /// Seals an unsafe prospective-cache teardown without returning its
    /// precharge.  The caller must close admission before returning: concrete
    /// allocations may still survive native destruction, so automatic rollback
    /// would create an unowned physical-allocation window.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn retain_charge_after_unsafe_teardown(mut self) {
        self.charged = false;
    }

    /// Commits a rollback only after the caller has proved that every
    /// allocation routed through the prospective cache scope was destroyed.
    /// A ledger failure keeps its existing charge but suppresses `Drop` retry:
    /// callers close admission rather than risk releasing later without a new
    /// teardown proof.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn release_after_proven_empty(mut self) -> Result<()> {
        if !self.charged {
            return Ok(());
        }
        let result = self.ledger.release_cache_residency_charge_after_destroy(self.bytes);
        self.charged = false;
        result
    }

    fn rollback_inner(&mut self) -> Result<()> {
        if !self.charged {
            return Ok(());
        }
        self.ledger.release_cache_residency_charge_after_destroy(self.bytes)?;
        self.charged = false;
        Ok(())
    }
}

impl Drop for CacheResidencyChargeTransaction {
    fn drop(&mut self) {
        if let Err(error) = self.rollback_inner() {
            // A failed rollback must retain the ledger charge rather than
            // risk under-accounting a potentially live allocation. The normal
            // path is exact and infallible unless the ledger itself is broken.
            eprintln!("[nexus-query-engine] warning: failed to roll back resident-cache charge transaction: {error}");
        }
    }
}

/// One cache entry's shared, non-cloneable residency authority.
///
/// Cache entries keep this object by value (usually inside their entry `Arc`).
/// Checkout callers receive [`CacheResidencyPin`] leases; they must never mint
/// another residency authority or byte charge for the same GPU allocation.
#[doc(hidden)]
pub struct CacheResidency {
    inner: Arc<CacheResidencyInner>,
}

impl CacheResidency {
    /// Pre-reserves an evictable charge before a cache copies or adopts bytes
    /// from an attempt domain. A failed reservation returns `Ok(None)` and
    /// leaves the ledger untouched.
    #[doc(hidden)]
    pub fn begin_attempt_owned(ledger: DeviceLedger, bytes: u64) -> Result<Option<CacheResidencyChargeTransaction>> {
        if !ledger.try_reserve_evictable_bytes(bytes)? {
            return Ok(None);
        }
        Ok(Some(CacheResidencyChargeTransaction { ledger, bytes, charged: true }))
    }

    /// Current lifecycle state.
    #[doc(hidden)]
    pub fn state(&self) -> Result<CacheResidencyState> {
        Ok(self.inner.lock("state")?.state)
    }

    /// Exact bytes owned by this lifecycle authority.
    #[doc(hidden)]
    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Tries to pin this entry for a cache hit or preparation operation.
    ///
    /// A final pin is refused after eviction begins or after destruction. The
    /// first pin moves the whole charge into the ledger's pinned field; later
    /// pins share that one charge until the final lease drops.
    #[doc(hidden)]
    pub fn try_pin(&self) -> Result<Option<CacheResidencyPin>> {
        let mut state = self.inner.lock("try_pin")?;
        match state.state {
            CacheResidencyState::Evictable if !state.eviction_in_progress => {
                self.inner.ledger.pin_cache_residency_charge(self.inner.bytes)?;
                state.state = CacheResidencyState::Pinned;
                state.charge_location = CacheResidencyChargeLocation::Pinned;
                state.pin_count = 1;
                Ok(Some(CacheResidencyPin { inner: Some(Arc::clone(&self.inner)) }))
            },
            CacheResidencyState::Pinned => {
                state.pin_count = state.pin_count.checked_add(1).ok_or_else(|| {
                    QueryError::new(QueryErrorCode::Internal, OPERATION, "resident-cache pin count overflowed")
                        .with_fact("reason", "cache_residency_pin_count_overflow")
                })?;
                Ok(Some(CacheResidencyPin { inner: Some(Arc::clone(&self.inner)) }))
            },
            #[cfg(test)]
            CacheResidencyState::AttemptOwned => Ok(None),
            CacheResidencyState::Destroyed | CacheResidencyState::Evictable => Ok(None),
        }
    }

    /// Claims this evictable entry for destruction.
    ///
    /// Pinned entries are deliberately refused without changing their charge.
    /// The returned token keeps the evictable charge in the ledger until the
    /// caller has destroyed its concrete GPU object.
    #[doc(hidden)]
    pub fn begin_eviction(&self) -> Result<Option<CacheResidencyDestruction>> {
        let mut state = self.inner.lock("begin_eviction")?;
        if state.state != CacheResidencyState::Evictable || state.eviction_in_progress {
            return Ok(None);
        }
        debug_assert_eq!(state.charge_location, CacheResidencyChargeLocation::Evictable);
        state.eviction_in_progress = true;
        Ok(Some(CacheResidencyDestruction { inner: Some(Arc::clone(&self.inner)) }))
    }
}

/// A conservative pinned charge for transient work that must execute in the
/// shared cache allocation domain.
///
/// A parked cuGraph graph is bound to that domain's runtime.  Its algorithm
/// outputs and workspaces can therefore exist in the same domain while the
/// graph is checked out, even though they do not become reusable cache
/// entries.  This guard reserves that bounded in-use interval in the one
/// ledger before the operation starts, then releases it only after the caller
/// has destroyed every transient cache-domain object.  Dropping it without
/// [`Self::release_after_destroy`] deliberately strands the charge rather than
/// allowing an uncharged physical allocation.
#[cfg(feature = "cugraph")]
#[must_use = "transient cache-domain work must release its ledger charge only after physical destruction"]
#[doc(hidden)]
pub struct CacheExecutionReservation {
    residency: CacheResidency,
    pin: Option<CacheResidencyPin>,
}

#[cfg(feature = "cugraph")]
impl CacheExecutionReservation {
    /// Conservatively reserves `bytes` before transient cache-domain work can
    /// allocate. `None` means the one aggregate cache cap cannot prove the
    /// operation safe.
    #[doc(hidden)]
    pub fn begin(ledger: DeviceLedger, bytes: u64) -> Result<Option<Self>> {
        let Some(transaction) = CacheResidency::begin_attempt_owned(ledger, bytes)? else {
            return Ok(None);
        };
        let residency = transaction.commit_evictable();
        let pin = residency.try_pin()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "fresh transient cache-domain reservation could not become pinned",
            )
            .with_fact("reason", "cache_execution_reservation_pin_refused")
        })?;
        Ok(Some(Self { residency, pin: Some(pin) }))
    }

    /// Releases this charge after all transient objects allocated in the
    /// shared cache domain have been destroyed.
    #[doc(hidden)]
    pub fn release_after_destroy(mut self) -> Result<()> {
        // The transient CUDA objects are already gone, so expose the charge
        // as evictable before committing its exact ledger release.
        drop(self.pin.take());
        let destruction = self.residency.begin_eviction()?.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "transient cache-domain reservation was not evictable after destruction",
            )
            .with_fact("reason", "cache_execution_reservation_not_evictable")
        })?;
        destruction.commit_after_destroy()
    }
}

/// Non-cloneable in-use lease for one resident cache entry.
#[must_use = "dropping a cache pin returns the entry to evictable residency after the final lease"]
#[doc(hidden)]
pub struct CacheResidencyPin {
    inner: Option<Arc<CacheResidencyInner>>,
}

impl Drop for CacheResidencyPin {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        if let Err(error) = release_pin(&inner) {
            // Failing closed leaves the pin and its ledger charge intact. A
            // poisoned state cannot make an entry evictable or free capacity.
            eprintln!("[nexus-query-engine] warning: failed to release resident-cache pin: {error}");
        }
    }
}

fn release_pin(inner: &CacheResidencyInner) -> Result<()> {
    let mut state = inner.lock("release_pin")?;
    if state.state != CacheResidencyState::Pinned || state.pin_count == 0 {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            OPERATION,
            "resident-cache pin lease does not match pinned entry state",
        )
        .with_fact("reason", "cache_residency_pin_state_mismatch"));
    }
    if state.pin_count > 1 {
        state.pin_count -= 1;
        return Ok(());
    }
    inner.ledger.unpin_cache_residency_charge(inner.bytes)?;
    state.pin_count = 0;
    state.state = CacheResidencyState::Evictable;
    state.charge_location = CacheResidencyChargeLocation::Evictable;
    Ok(())
}

/// Linear destruction authority for an evictable resident entry.
///
/// Cache code must remove the map entry and destroy the concrete GPU object
/// before consuming this token with [`Self::commit_after_destroy`].
#[must_use = "a destruction token retains the ledger charge until destruction is committed or explicitly aborted"]
#[doc(hidden)]
pub struct CacheResidencyDestruction {
    inner: Option<Arc<CacheResidencyInner>>,
}

impl CacheResidencyDestruction {
    /// Releases the exact ledger charge after GPU-safe concrete destruction.
    #[doc(hidden)]
    pub fn commit_after_destroy(mut self) -> Result<()> {
        let Some(inner) = self.inner.take() else {
            return Ok(());
        };
        let mut state = inner.lock("commit_after_destroy")?;
        if state.state != CacheResidencyState::Evictable || !state.eviction_in_progress {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident-cache destruction commit does not match an evictable entry",
            )
            .with_fact("reason", "cache_residency_destroy_state_mismatch"));
        }
        inner.ledger.release_cache_residency_charge_after_destroy(inner.bytes)?;
        state.state = CacheResidencyState::Destroyed;
        state.charge_location = CacheResidencyChargeLocation::Released;
        state.eviction_in_progress = false;
        Ok(())
    }

    /// Seals a failed or unsafe destruction attempt without releasing bytes.
    ///
    /// This is the quarantine path: callers must close the owning device
    /// ledger before returning, because the charge remains while native memory
    /// may still be resident and the cache object is no longer reusable.
    #[doc(hidden)]
    pub fn retain_charge_after_failed_destruction(mut self) -> Result<()> {
        let Some(inner) = self.inner.take() else {
            return Ok(());
        };
        let mut state = inner.lock("retain_charge_after_failed_destruction")?;
        if state.state != CacheResidencyState::Evictable || !state.eviction_in_progress {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident-cache failed destruction does not match an evictable entry",
            )
            .with_fact("reason", "cache_residency_failed_destroy_state_mismatch"));
        }
        state.state = CacheResidencyState::Destroyed;
        state.eviction_in_progress = false;
        Ok(())
    }
}

impl Drop for CacheResidencyDestruction {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        if let Err(error) = abort_eviction(&inner) {
            // Leaving the charge in place is the only safe fallback here.
            eprintln!("[nexus-query-engine] warning: failed to abort resident-cache destruction: {error}");
        }
    }
}

fn abort_eviction(inner: &CacheResidencyInner) -> Result<()> {
    let mut state = inner.lock("abort_eviction")?;
    if state.state != CacheResidencyState::Evictable || !state.eviction_in_progress {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            OPERATION,
            "resident-cache eviction abort does not match an evictable entry",
        )
        .with_fact("reason", "cache_residency_abort_state_mismatch"));
    }
    state.eviction_in_progress = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Barrier, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        thread,
    };

    use super::*;
    use crate::runtime::gpu::{DeviceCapabilities, DeviceResourceProfile};

    fn ledger(cache_cap_bytes: u64) -> DeviceLedger {
        let profile = DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
            0,
            10_000,
            0,
            0,
            100,
            1,
            1,
            2,
            2,
            DeviceCapabilities::default(),
            cache_cap_bytes,
        )
        .expect("cache test profile must resolve");
        DeviceLedger::new(profile)
    }

    #[test]
    fn pin_refuses_eviction_and_destruction_releases_only_after_commit() {
        let ledger = ledger(500);
        let transaction = CacheResidency::begin_attempt_owned(ledger.clone(), 200).unwrap().expect("charge fits");
        assert_eq!(transaction.state(), CacheResidencyState::AttemptOwned);
        assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 200);

        let residency = transaction.commit_evictable();
        assert_eq!(residency.state().unwrap(), CacheResidencyState::Evictable);
        let first_pin = residency.try_pin().unwrap().expect("evictable entry pins");
        let second_pin = residency.try_pin().unwrap().expect("pinned entry accepts another lease");
        let pinned = ledger.snapshot().unwrap();
        assert_eq!(pinned.cache_evictable_resident_bytes, 0);
        assert_eq!(pinned.cache_pinned_in_use_bytes, 200);
        assert!(residency.begin_eviction().unwrap().is_none(), "pinned entry must not evict");
        assert_eq!(ledger.snapshot().unwrap(), pinned, "refusal leaves its charge unchanged");

        drop(first_pin);
        assert_eq!(residency.state().unwrap(), CacheResidencyState::Pinned);
        drop(second_pin);
        assert_eq!(residency.state().unwrap(), CacheResidencyState::Evictable);
        let evictable = ledger.snapshot().unwrap();
        assert_eq!(evictable.cache_evictable_resident_bytes, 200);
        assert_eq!(evictable.cache_pinned_in_use_bytes, 0);

        let destruction = residency.begin_eviction().unwrap().expect("un-pinned entry evicts");
        assert_eq!(residency.state().unwrap(), CacheResidencyState::Evictable);
        assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 200);
        destruction.commit_after_destroy().unwrap();
        assert_eq!(residency.state().unwrap(), CacheResidencyState::Destroyed);
        let destroyed = ledger.snapshot().unwrap();
        assert_eq!(destroyed.cache_evictable_resident_bytes, 0);
        assert_eq!(destroyed.cache_pinned_in_use_bytes, 0);
    }

    #[test]
    fn dropped_charge_transaction_rolls_back_exactly_once() {
        let ledger = ledger(500);
        let before = ledger.snapshot().unwrap().sequence;
        let transaction = CacheResidency::begin_attempt_owned(ledger.clone(), 150).unwrap().expect("charge fits");
        assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 150);
        drop(transaction);
        let after = ledger.snapshot().unwrap();
        assert_eq!(after.cache_evictable_resident_bytes, 0);
        assert_eq!(after.cache_pinned_in_use_bytes, 0);
        assert_eq!(after.sequence, before + 2, "one reserve and one rollback transition only");
    }

    #[test]
    fn quarantined_resident_entry_keeps_its_charge_and_closes_admission() {
        let ledger = ledger(500);
        let transaction = CacheResidency::begin_attempt_owned(ledger.clone(), 200).unwrap().expect("charge fits");
        let residency = transaction.commit_evictable();

        let destruction = residency.begin_eviction().unwrap().expect("evictable entry begins destruction");
        destruction.retain_charge_after_failed_destruction().unwrap();
        ledger.close_admission("resident-cache teardown quarantined").unwrap();

        assert_eq!(residency.state().unwrap(), CacheResidencyState::Destroyed);
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.cache_evictable_resident_bytes, 200, "unsafe destruction must retain the sole charge");
        assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
        assert!(matches!(snapshot.health, crate::runtime::gpu::DeviceAdmissionHealth::Closed { .. }));
        assert!(
            ledger
                .try_admit(
                    crate::observability::AdmissionRequestId::new(1),
                    &crate::admission::AttemptRequirement::empty(),
                )
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn concurrent_insert_and_evict_never_undercounts_true_residency() {
        const WORKERS: usize = 4;
        const ROUNDS: usize = 80;
        const BYTES: u64 = 37;

        let ledger = Arc::new(ledger(2_000));
        let physically_resident_bytes = Arc::new(Mutex::new(0u64));
        let running = Arc::new(AtomicBool::new(true));
        let start = Arc::new(Barrier::new(WORKERS + 2));

        let observer_ledger = Arc::clone(&ledger);
        let observer_physical = Arc::clone(&physically_resident_bytes);
        let observer_running = Arc::clone(&running);
        let observer_start = Arc::clone(&start);
        let observer = thread::spawn(move || {
            observer_start.wait();
            while observer_running.load(Ordering::Acquire) {
                // Hold the physical model lock through the snapshot. Insert
                // charges before adding physical bytes; eviction removes the
                // physical bytes before its ledger-release commit under this
                // same lock, so every sample is a coherent observation point.
                let physical = observer_physical.lock().unwrap();
                let true_residency = *physical;
                let snapshot = observer_ledger.snapshot().unwrap();
                drop(physical);
                let ledger_residency = snapshot.cache_evictable_resident_bytes + snapshot.cache_pinned_in_use_bytes;
                assert!(
                    ledger_residency >= true_residency,
                    "ledger undercounted physical residency: ledger={ledger_residency}, physical={true_residency}"
                );
                thread::yield_now();
            }
        });

        let workers = (0..WORKERS)
            .map(|worker| {
                let ledger = Arc::clone(&ledger);
                let physical = Arc::clone(&physically_resident_bytes);
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    for round in 0..ROUNDS {
                        let transaction = loop {
                            if let Some(transaction) =
                                CacheResidency::begin_attempt_owned((*ledger).clone(), BYTES).unwrap()
                            {
                                break transaction;
                            }
                            thread::yield_now();
                        };
                        let residency = transaction.commit_evictable();
                        {
                            let mut bytes = physical.lock().unwrap();
                            *bytes += BYTES;
                        }
                        if (worker + round) % 2 == 0 {
                            let pin = residency.try_pin().unwrap().expect("fresh entry pins");
                            assert!(residency.begin_eviction().unwrap().is_none());
                            drop(pin);
                        }
                        let destruction = residency.begin_eviction().unwrap().expect("entry becomes evictable");
                        let mut bytes = physical.lock().unwrap();
                        *bytes -= BYTES;
                        destruction.commit_after_destroy().unwrap();
                    }
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        for worker in workers {
            worker.join().expect("residency worker must not panic");
        }
        running.store(false, Ordering::Release);
        observer.join().expect("residency observer must not panic");
        assert_eq!(*physically_resident_bytes.lock().unwrap(), 0);
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
        assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn transient_cache_execution_reservation_is_pinned_until_physical_cleanup() {
        let ledger = ledger(500);
        let reservation = CacheExecutionReservation::begin(ledger.clone(), 200)
            .unwrap()
            .expect("the transient cache reservation fits");

        let during_execution = ledger.snapshot().unwrap();
        assert_eq!(during_execution.cache_evictable_resident_bytes, 0);
        assert_eq!(during_execution.cache_pinned_in_use_bytes, 200);

        reservation.release_after_destroy().unwrap();

        let after_cleanup = ledger.snapshot().unwrap();
        assert_eq!(after_cleanup.cache_evictable_resident_bytes, 0);
        assert_eq!(after_cleanup.cache_pinned_in_use_bytes, 0);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn dropped_transient_cache_execution_reservation_retains_charge_fail_closed() {
        let ledger = ledger(500);
        let reservation = CacheExecutionReservation::begin(ledger.clone(), 200)
            .unwrap()
            .expect("the transient cache reservation fits");

        drop(reservation);

        let stranded = ledger.snapshot().unwrap();
        assert_eq!(stranded.cache_evictable_resident_bytes, 200);
        assert_eq!(stranded.cache_pinned_in_use_bytes, 0);
    }
}
