//! Backend observation and GPU-resource gates.
//!
//! Both permit counts share one atomic close bit.  Observation permits retain
//! only this lifecycle state; the backend core composes a GPU count permit with
//! the detachable resource generation it protects.

use std::{
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use crate::error::{QueryError, QueryErrorCode, Result};

const COUNT_BITS: u32 = (usize::BITS - 1) / 2;
const OBSERVATION_COUNT_MASK: usize = (1usize << COUNT_BITS) - 1;
const GPU_COUNT_SHIFT: u32 = COUNT_BITS;
const GPU_COUNT_ONE: usize = 1usize << GPU_COUNT_SHIFT;
const GPU_COUNT_MASK: usize = OBSERVATION_COUNT_MASK << GPU_COUNT_SHIFT;
const CLOSING_BIT: usize = 1usize << (usize::BITS - 1);
const PUBLIC_HANDLES_FIRED_BIT: usize = CLOSING_BIT;
const PUBLIC_HANDLES_COUNT_MASK: usize = CLOSING_BIT - 1;

const OBSERVATION_OPERATION: &str = "nexus_query_engine.backend_lifecycle.begin_observation";
const OBSERVATION_WAIT_OPERATION: &str = "nexus_query_engine.backend_lifecycle.wait_for_observation_quiescence";
const GPU_OPERATION: &str = "nexus_query_engine.backend_lifecycle.begin_gpu_resource";
const GPU_WAIT_OPERATION: &str = "nexus_query_engine.backend_lifecycle.wait_for_gpu_quiescence";

/// Backend lifecycle state.
///
/// The state word's closed bit and both permit counts form one linearization domain:
/// an acquisition that wins the compare-and-exchange is counted before close,
/// while one that observes the closed bit is rejected.  The mutex is only a
/// condition-variable coordination point; it is not part of the acquisition
/// gate and therefore [`BackendLifecycle::close`] never waits for it.
struct BackendLifecycleInner {
    state: AtomicUsize,
    wait: Mutex<()>,
    changed: Condvar,
}

impl fmt::Debug for BackendLifecycleInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendLifecycleInner")
            .field("state", &self.state.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

/// Shared backend lifecycle gate.
///
/// Cloning this value shares one open/closing gate.  It does not represent a
/// backend resource owner and does not contain or retain any GPU resource.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct BackendLifecycle {
    inner: Arc<BackendLifecycleInner>,
}

/// A counted, observation-only permission created before backend close.
///
/// A permit may outlive GPU-resource quiescence because it retains only the
/// observation lifecycle.  Dropping it is the observation-quiescence release
/// point.
#[must_use = "an observation permit must be held until observation work finishes"]
#[derive(Debug)]
#[doc(hidden)]
pub struct ObservationPermit {
    lifecycle: BackendLifecycle,
}

/// Counted permission to enter the current GPU resource generation.
///
/// This value deliberately contains no resource reference. `backend_core`
/// combines it with an `Arc<ResourceGeneration>` while holding the detachable
/// resource-core lock, so close cannot separate a winning permit from the
/// generation it protects.
#[must_use = "a GPU lifecycle permit must be held until GPU resource work finishes"]
#[derive(Debug)]
#[doc(hidden)]
pub struct GpuLifecyclePermit {
    lifecycle: BackendLifecycle,
}

/// Snapshot of the observation lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct BackendLifecycleSnapshot {
    /// Whether close has won the lifecycle gate.
    #[doc(hidden)]
    pub closing: bool,
    /// Number of permits acquired before close and not yet dropped.
    #[doc(hidden)]
    pub active_observation_permits: usize,
    /// Number of GPU-resource permits acquired before close and not yet dropped.
    #[doc(hidden)]
    pub active_gpu_permits: usize,
}

impl BackendLifecycleSnapshot {
    /// Returns whether no observation work remains.
    #[cfg(test)]
    #[doc(hidden)]
    pub const fn is_observation_quiescent(self) -> bool {
        self.active_observation_permits == 0
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub const fn is_gpu_quiescent(self) -> bool {
        self.active_gpu_permits == 0
    }
}

impl BackendLifecycle {
    /// Creates an open observation lifecycle.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(BackendLifecycleInner {
                state: AtomicUsize::new(0),
                wait: Mutex::new(()),
                changed: Condvar::new(),
            }),
        }
    }

    /// Attempts to begin one observation before close wins the gate.
    ///
    /// The close bit and count are changed in one atomic state machine, so a
    /// losing acquisition creates no permit and cannot leave a half-created
    /// observation.  The returned permit has no resource-core reference.
    #[doc(hidden)]
    pub fn try_begin_observation(&self) -> Result<ObservationPermit> {
        let mut state = self.inner.state.load(Ordering::Acquire);
        loop {
            if state & CLOSING_BIT != 0 {
                return Err(backend_closing_error());
            }

            let active = state & OBSERVATION_COUNT_MASK;
            if active == OBSERVATION_COUNT_MASK {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    OBSERVATION_OPERATION,
                    "backend observation permit count overflow",
                )
                .with_fact("reason", "observation_permit_count_overflow"));
            }

            match self.inner.state.compare_exchange_weak(state, state + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Ok(ObservationPermit { lifecycle: self.clone() });
                },
                Err(observed) => state = observed,
            }
        }
    }

    /// Attempts to enter the current GPU resource generation before close.
    #[doc(hidden)]
    pub fn try_begin_gpu_resource(&self) -> Result<GpuLifecyclePermit> {
        let mut state = self.inner.state.load(Ordering::Acquire);
        loop {
            if state & CLOSING_BIT != 0 {
                return Err(backend_closing_error_for(GPU_OPERATION, "GPU resource request"));
            }
            let active = (state & GPU_COUNT_MASK) >> GPU_COUNT_SHIFT;
            if active == OBSERVATION_COUNT_MASK {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    GPU_OPERATION,
                    "backend GPU resource permit count overflow",
                )
                .with_fact("reason", "gpu_resource_permit_count_overflow"));
            }
            match self.inner.state.compare_exchange_weak(
                state,
                state + GPU_COUNT_ONE,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(GpuLifecyclePermit { lifecycle: self.clone() }),
                Err(observed) => state = observed,
            }
        }
    }

    /// Atomically transitions this lifecycle from open to closing.
    ///
    /// Returns `true` only for the caller that wins the transition.  Repeated
    /// and concurrent calls are safe and return `false` after the first close.
    /// The signal itself is non-blocking, including while the wait mutex is
    /// contended.
    #[doc(hidden)]
    pub fn close(&self) -> bool {
        let previous = self.inner.state.fetch_or(CLOSING_BIT, Ordering::AcqRel);
        self.inner.changed.notify_all();
        previous & CLOSING_BIT == 0
    }

    /// Returns whether this lifecycle has entered closing state.
    #[doc(hidden)]
    pub fn is_closing(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) & CLOSING_BIT != 0
    }

    /// Returns the currently counted observation permits.
    #[doc(hidden)]
    pub fn active_observation_permits(&self) -> usize {
        self.inner.state.load(Ordering::Acquire) & OBSERVATION_COUNT_MASK
    }

    #[doc(hidden)]
    pub fn active_gpu_permits(&self) -> usize {
        (self.inner.state.load(Ordering::Acquire) & GPU_COUNT_MASK) >> GPU_COUNT_SHIFT
    }

    /// Takes a point-in-time observation state snapshot.
    #[doc(hidden)]
    pub fn snapshot(&self) -> BackendLifecycleSnapshot {
        let state = self.inner.state.load(Ordering::Acquire);
        BackendLifecycleSnapshot {
            closing: state & CLOSING_BIT != 0,
            active_observation_permits: state & OBSERVATION_COUNT_MASK,
            active_gpu_permits: (state & GPU_COUNT_MASK) >> GPU_COUNT_SHIFT,
        }
    }

    /// Closes this lifecycle and waits for observation permits under an
    /// absolute monotonic deadline.
    ///
    /// An observation that reaches zero only after the deadline is not
    /// accepted as successful quiescence.  The initial uncontended state may
    /// still complete immediately at an already-expired deadline; no waiting
    /// has then occurred.  The same deadline covers wait-mutex contention and
    /// condition-variable waiting.
    #[doc(hidden)]
    pub fn wait_for_observation_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.close();

        self.wait_for_count_until(deadline, CountKind::Observation)
    }

    /// Closes the lifecycle and waits for every GPU-resource permit.
    #[doc(hidden)]
    pub fn wait_for_gpu_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.close();

        self.wait_for_count_until(deadline, CountKind::Gpu)
    }

    fn wait_for_count_until(&self, deadline: Instant, kind: CountKind) -> Result<()> {
        let mut wait = match super::deadline::mutex_lock_until(&self.inner.wait, deadline) {
            Ok(wait) => wait,
            Err(super::deadline::DeadlineMutexLockError::Poisoned) => {
                return Err(wait_poisoned_error(kind));
            },
            Err(super::deadline::DeadlineMutexLockError::TimedOut) => {
                return Err(wait_timeout_error(kind, None));
            },
        };

        let mut waited_for_quiescence = false;
        loop {
            let active = kind.active(self);
            if active == 0 {
                if waited_for_quiescence && super::deadline::deadline_expired(deadline) {
                    return Err(wait_timeout_error(kind, Some(0)));
                }
                return Ok(());
            }

            if super::deadline::deadline_expired(deadline) {
                return Err(wait_timeout_error(kind, Some(active)));
            }
            let Some(remaining) = super::deadline::deadline_remaining(deadline) else {
                return Err(wait_timeout_error(kind, Some(active)));
            };

            waited_for_quiescence = true;
            let (next, _timeout) =
                self.inner.changed.wait_timeout(wait, remaining).map_err(|_error| wait_poisoned_error(kind))?;
            wait = next;
        }
    }

    fn release_observation(&self) {
        // Holding the coordination mutex while decrementing closes the gap in
        // which a waiter could observe an active permit, then miss the release
        // notification before entering `wait_timeout`.
        let _wait = self.inner.wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = self.inner.state.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous & OBSERVATION_COUNT_MASK > 0, "backend observation permit count underflow");
        let next = previous - 1;
        if next & CLOSING_BIT != 0 && next & OBSERVATION_COUNT_MASK == 0 {
            self.inner.changed.notify_all();
        }
    }

    fn release_gpu(&self) {
        let _wait = self.inner.wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = self.inner.state.fetch_sub(GPU_COUNT_ONE, Ordering::AcqRel);
        debug_assert!(previous & GPU_COUNT_MASK != 0, "backend GPU permit count underflow");
        let next = previous - GPU_COUNT_ONE;
        if next & CLOSING_BIT != 0 && next & GPU_COUNT_MASK == 0 {
            self.inner.changed.notify_all();
        }
    }
}

impl Drop for ObservationPermit {
    fn drop(&mut self) {
        self.lifecycle.release_observation();
    }
}

impl Drop for GpuLifecyclePermit {
    fn drop(&mut self) {
        self.lifecycle.release_gpu();
    }
}

#[derive(Clone, Copy)]
enum CountKind {
    Observation,
    Gpu,
}

impl CountKind {
    fn active(self, lifecycle: &BackendLifecycle) -> usize {
        match self {
            Self::Observation => lifecycle.active_observation_permits(),
            Self::Gpu => lifecycle.active_gpu_permits(),
        }
    }

    fn operation(self) -> &'static str {
        match self {
            Self::Observation => OBSERVATION_WAIT_OPERATION,
            Self::Gpu => GPU_WAIT_OPERATION,
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Observation => "observation_timeout",
            Self::Gpu => "gpu_resource_timeout",
        }
    }

    fn fact(self) -> &'static str {
        match self {
            Self::Observation => "active_observation_permits",
            Self::Gpu => "active_gpu_permits",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Observation => "observation",
            Self::Gpu => "GPU resource",
        }
    }
}

fn backend_closing_error() -> QueryError {
    backend_closing_error_for(OBSERVATION_OPERATION, "observation")
}

fn backend_closing_error_for(operation: &'static str, label: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, format!("backend is closing and no new {label} can begin"))
        .with_fact("reason", "backend_closing")
}

fn wait_timeout_error(kind: CountKind, active: Option<usize>) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        kind.operation(),
        format!("timed out waiting for backend {} quiescence", kind.label()),
    )
    .with_fact("reason", kind.reason())
    .with_cleanup_deadline_elapsed();
    match active {
        Some(active) => error.with_fact(kind.fact(), active),
        None => error,
    }
}

fn wait_poisoned_error(kind: CountKind) -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        kind.operation(),
        format!("backend {} lifecycle mutex was poisoned during shutdown", kind.label()),
    )
    .with_fact(
        "reason",
        match kind {
            CountKind::Observation => "observation_wait_poisoned",
            CountKind::Gpu => "gpu_resource_wait_poisoned",
        },
    )
}

/// Internal tracker for public backend-handle identities.
///
/// This counter is deliberately separate from every `Arc` count.  The last
/// lease atomically transitions the tracker to a fired state before taking and
/// invoking the callback, preventing resurrection and guaranteeing one
/// callback even when drops race on different threads.
#[doc(hidden)]
pub struct PublicHandleTracker {
    state: AtomicUsize,
    on_last_handle: Mutex<Option<Box<dyn FnOnce() + Send + 'static>>>,
}

impl fmt::Debug for PublicHandleTracker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicHandleTracker")
            .field("active_handles", &self.active_handles())
            .field("callback_fired", &self.callback_fired())
            .finish_non_exhaustive()
    }
}

/// One counted public backend handle identity.
///
/// Cloning this lease increments the tracker's explicit counter.  Dropping
/// the last lease invokes the tracker's callback exactly once; this operation
/// never consults `Arc::strong_count`.
#[doc(hidden)]
pub struct PublicHandleLease {
    tracker: Arc<PublicHandleTracker>,
}

impl fmt::Debug for PublicHandleLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("PublicHandleLease").field("active_handles", &self.tracker.active_handles()).finish()
    }
}

impl PublicHandleTracker {
    /// Creates an empty tracker.  Call [`Self::acquire`] to mint the first
    /// public-handle lease.
    #[doc(hidden)]
    pub fn new(on_last_handle: impl FnOnce() + Send + 'static) -> Arc<Self> {
        Arc::new(Self { state: AtomicUsize::new(0), on_last_handle: Mutex::new(Some(Box::new(on_last_handle))) })
    }

    /// Mints one lease, or returns `None` after the last lease already fired
    /// the callback.
    #[doc(hidden)]
    pub fn acquire(self: &Arc<Self>) -> Option<PublicHandleLease> {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if state & PUBLIC_HANDLES_FIRED_BIT != 0 {
                return None;
            }
            let active = state & PUBLIC_HANDLES_COUNT_MASK;
            if active == PUBLIC_HANDLES_COUNT_MASK {
                // `Clone` cannot report a structured error.  A process with
                // this many simultaneously live public handles has violated
                // the explicit counter's representable domain.
                panic!("public backend handle lease count overflow");
            }
            match self.state.compare_exchange_weak(state, state + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(PublicHandleLease { tracker: Arc::clone(self) }),
                Err(observed) => state = observed,
            }
        }
    }

    /// Returns the number of currently minted public-handle leases.
    #[doc(hidden)]
    pub fn active_handles(&self) -> usize {
        self.state.load(Ordering::Acquire) & PUBLIC_HANDLES_COUNT_MASK
    }

    /// Returns whether the last-handle callback has fired.
    #[doc(hidden)]
    pub fn callback_fired(&self) -> bool {
        self.state.load(Ordering::Acquire) & PUBLIC_HANDLES_FIRED_BIT != 0
    }

    fn release(&self) {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            debug_assert!(state & PUBLIC_HANDLES_FIRED_BIT == 0, "public handle released after callback fired");
            let active = state & PUBLIC_HANDLES_COUNT_MASK;
            debug_assert!(active > 0, "public handle lease count underflow");
            let next = if active == 1 { PUBLIC_HANDLES_FIRED_BIT } else { state - 1 };
            match self.state.compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) if active == 1 => {
                    let callback = self.on_last_handle.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
                    if let Some(callback) = callback {
                        callback();
                    }
                    return;
                },
                Ok(_) => return,
                Err(observed) => state = observed,
            }
        }
    }
}

impl Clone for PublicHandleLease {
    fn clone(&self) -> Self {
        self.tracker.acquire().expect("a live public-handle lease must be cloneable before its drop")
    }
}

impl Drop for PublicHandleLease {
    fn drop(&mut self) {
        self.tracker.release();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    use super::*;

    fn expired_deadline() -> Instant {
        Instant::now().checked_sub(Duration::from_secs(1)).expect("one second before now is representable")
    }

    fn assert_backend_closing(error: QueryError, operation: &'static str) {
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("backend_closing"));
        assert_eq!(error.operation(), operation);
    }

    #[test]
    fn acquisition_and_close_are_one_linearization_domain() {
        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().expect("open lifecycle permits observation");
        assert_eq!(lifecycle.active_observation_permits(), 1);
        assert!(lifecycle.close());
        assert!(!lifecycle.close(), "concurrent/idempotent close winner is unique");
        let error = lifecycle.try_begin_observation().expect_err("close rejects a new observation");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("backend_closing"));
        assert_eq!(error.operation(), OBSERVATION_OPERATION);
        assert!(lifecycle.snapshot().closing);
        drop(permit);
        assert_eq!(
            lifecycle.snapshot(),
            BackendLifecycleSnapshot { closing: true, active_observation_permits: 0, active_gpu_permits: 0 }
        );
    }

    #[test]
    fn observation_gpu_acquisitions_and_close_race_in_one_atomic_domain() {
        const OBSERVATION_ACQUIRERS: usize = 16;
        const GPU_ACQUIRERS: usize = 16;

        let lifecycle = BackendLifecycle::new();
        let start = Arc::new(Barrier::new(OBSERVATION_ACQUIRERS + GPU_ACQUIRERS + 2));

        let observation_threads = (0..OBSERVATION_ACQUIRERS)
            .map(|_| {
                let lifecycle = lifecycle.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    lifecycle.try_begin_observation()
                })
            })
            .collect::<Vec<_>>();
        let gpu_threads = (0..GPU_ACQUIRERS)
            .map(|_| {
                let lifecycle = lifecycle.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    lifecycle.try_begin_gpu_resource()
                })
            })
            .collect::<Vec<_>>();
        let closing = lifecycle.clone();
        let close_start = Arc::clone(&start);
        let close_thread = thread::spawn(move || {
            close_start.wait();
            closing.close()
        });

        start.wait();
        assert!(close_thread.join().unwrap(), "the only close caller wins the close bit");

        let mut observation_permits = Vec::new();
        for thread in observation_threads {
            match thread.join().unwrap() {
                Ok(permit) => observation_permits.push(permit),
                Err(error) => assert_backend_closing(error, OBSERVATION_OPERATION),
            }
        }
        let mut gpu_permits = Vec::new();
        for thread in gpu_threads {
            match thread.join().unwrap() {
                Ok(permit) => gpu_permits.push(permit),
                Err(error) => assert_backend_closing(error, GPU_OPERATION),
            }
        }

        assert_eq!(
            lifecycle.snapshot(),
            BackendLifecycleSnapshot {
                closing: true,
                active_observation_permits: observation_permits.len(),
                active_gpu_permits: gpu_permits.len(),
            },
            "every acquisition linearized before close is counted and every later acquisition is rejected"
        );
        assert_backend_closing(lifecycle.try_begin_observation().unwrap_err(), OBSERVATION_OPERATION);
        assert_backend_closing(lifecycle.try_begin_gpu_resource().unwrap_err(), GPU_OPERATION);

        drop(observation_permits);
        drop(gpu_permits);
        assert_eq!(
            lifecycle.snapshot(),
            BackendLifecycleSnapshot { closing: true, active_observation_permits: 0, active_gpu_permits: 0 }
        );
    }

    #[test]
    fn concurrent_close_has_exactly_one_winner_and_is_idempotent() {
        const CLOSE_CALLERS: usize = 32;

        let lifecycle = BackendLifecycle::new();
        let start = Arc::new(Barrier::new(CLOSE_CALLERS + 1));
        let threads = (0..CLOSE_CALLERS)
            .map(|_| {
                let lifecycle = lifecycle.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    lifecycle.close()
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        let winners = threads.into_iter().map(|thread| usize::from(thread.join().unwrap())).sum::<usize>();
        assert_eq!(winners, 1);
        assert!(!lifecycle.close(), "close remains idempotent after all concurrent callers finish");
        assert!(lifecycle.is_closing());
        assert_backend_closing(lifecycle.try_begin_observation().unwrap_err(), OBSERVATION_OPERATION);
        assert_backend_closing(lifecycle.try_begin_gpu_resource().unwrap_err(), GPU_OPERATION);
    }

    #[test]
    fn close_does_not_wait_for_contended_wait_mutex() {
        let lifecycle = BackendLifecycle::new();
        let wait_guard = lifecycle.inner.wait.lock().unwrap();
        let closing = lifecycle.clone();
        let (closed_tx, closed_rx) = std::sync::mpsc::channel();
        let thread = thread::spawn(move || {
            closed_tx.send(closing.close()).unwrap();
        });
        assert!(closed_rx.recv().unwrap());
        drop(wait_guard);
        thread.join().unwrap();
        assert!(lifecycle.is_closing());
        assert!(lifecycle.try_begin_observation().is_err());
    }

    #[test]
    fn zero_deadline_does_not_wait_for_a_contended_coordination_mutex() {
        let lifecycle = BackendLifecycle::new();
        let wait_guard = lifecycle.inner.wait.lock().unwrap();
        let error = lifecycle
            .wait_for_observation_quiescence_until(Instant::now())
            .expect_err("an already-expired deadline cannot wait for the held coordination mutex");
        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("reason"), Some("observation_timeout"));
        assert_eq!(error.fact_value("active_observation_permits"), None);
        assert!(lifecycle.is_closing());
        drop(wait_guard);
        lifecycle
            .wait_for_observation_quiescence_until(expired_deadline())
            .expect("an uncontended, already-quiescent lifecycle completes best-effort close after the timeout");
    }

    #[test]
    fn expired_deadlines_report_each_active_scope_and_allow_late_release() {
        let lifecycle = BackendLifecycle::new();
        let observation = lifecycle.try_begin_observation().unwrap();
        let gpu = lifecycle.try_begin_gpu_resource().unwrap();
        let deadline = expired_deadline();

        let observation_error = lifecycle
            .wait_for_observation_quiescence_until(deadline)
            .expect_err("the active observation outlives the deadline");
        assert_eq!(observation_error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(observation_error.operation(), OBSERVATION_WAIT_OPERATION);
        assert_eq!(observation_error.fact_value("reason"), Some("observation_timeout"));
        assert_eq!(observation_error.fact_value("active_observation_permits"), Some("1"));

        let gpu_error =
            lifecycle.wait_for_gpu_quiescence_until(deadline).expect_err("the active GPU permit outlives the deadline");
        assert_eq!(gpu_error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(gpu_error.operation(), GPU_WAIT_OPERATION);
        assert_eq!(gpu_error.fact_value("reason"), Some("gpu_resource_timeout"));
        assert_eq!(gpu_error.fact_value("active_gpu_permits"), Some("1"));

        drop(observation);
        assert!(lifecycle.snapshot().is_observation_quiescent());
        assert!(!lifecycle.snapshot().is_gpu_quiescent());
        lifecycle
            .wait_for_observation_quiescence_until(deadline)
            .expect("a late observation release is visible to a best-effort retry");

        drop(gpu);
        assert!(lifecycle.snapshot().is_gpu_quiescent());
        lifecycle
            .wait_for_gpu_quiescence_until(deadline)
            .expect("a late GPU release is visible to a best-effort retry");
    }

    #[test]
    fn quiescence_waits_are_scope_specific_while_close_admission_is_shared() {
        let lifecycle = BackendLifecycle::new();
        let observation = lifecycle.try_begin_observation().unwrap();
        let gpu = lifecycle.try_begin_gpu_resource().unwrap();
        drop(gpu);

        lifecycle
            .wait_for_gpu_quiescence_until(expired_deadline())
            .expect("an observation-only permit cannot delay GPU-resource quiescence");
        assert_eq!(lifecycle.active_observation_permits(), 1);
        assert_eq!(lifecycle.active_gpu_permits(), 0);
        assert_backend_closing(lifecycle.try_begin_observation().unwrap_err(), OBSERVATION_OPERATION);
        assert_backend_closing(lifecycle.try_begin_gpu_resource().unwrap_err(), GPU_OPERATION);

        let error = lifecycle
            .wait_for_observation_quiescence_until(expired_deadline())
            .expect_err("the remaining observation permit is counted only by the observation scope");
        assert_eq!(error.fact_value("active_observation_permits"), Some("1"));
        drop(observation);
        lifecycle
            .wait_for_observation_quiescence_until(expired_deadline())
            .expect("the observation scope becomes quiescent after its own permit is released");
    }

    #[test]
    fn observation_permit_does_not_keep_an_external_resource_alive() {
        struct ResourceDropProbe(Arc<AtomicUsize>);

        impl Drop for ResourceDropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }

        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let resource = Arc::new(ResourceDropProbe(Arc::clone(&drops)));
        let weak_resource = Arc::downgrade(&resource);

        drop(resource);
        assert!(weak_resource.upgrade().is_none(), "an observation-only permit must not retain a resource Arc");
        assert_eq!(drops.load(Ordering::Acquire), 1);
        assert_eq!(lifecycle.active_observation_permits(), 1, "the observation remains live independently");

        drop(permit);
        assert_eq!(lifecycle.active_observation_permits(), 0);
        assert!(!lifecycle.is_closing());
    }

    #[test]
    fn public_handle_tracker_invokes_callback_once_on_last_drop() {
        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_for_callback = Arc::clone(&callback_count);
        let tracker = PublicHandleTracker::new(move || {
            callback_count_for_callback.fetch_add(1, Ordering::AcqRel);
        });
        let first = tracker.acquire().expect("first public handle");
        let second = first.clone();
        let third = second.clone();
        assert_eq!(tracker.active_handles(), 3);
        drop(first);
        drop(second);
        assert_eq!(callback_count.load(Ordering::Acquire), 0);
        drop(third);
        assert_eq!(callback_count.load(Ordering::Acquire), 1);
        assert!(tracker.callback_fired());
        assert_eq!(tracker.active_handles(), 0);
        assert!(tracker.acquire().is_none(), "a fired tracker cannot mint a replacement handle");
    }

    #[test]
    fn public_handle_last_drop_callback_is_exactly_once_under_concurrent_drops() {
        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_for_callback = Arc::clone(&callback_count);
        let (callback_started_tx, callback_started_rx) = mpsc::channel();
        let (allow_callback_finish_tx, allow_callback_finish_rx) = mpsc::channel();
        let tracker = PublicHandleTracker::new(move || {
            callback_count_for_callback.fetch_add(1, Ordering::AcqRel);
            callback_started_tx.send(()).unwrap();
            allow_callback_finish_rx.recv().unwrap();
        });
        let root = tracker.acquire().unwrap();
        let mut leases = Vec::new();
        for _ in 0..31 {
            leases.push(root.clone());
        }
        leases.push(root);
        let barrier = Arc::new(Barrier::new(leases.len() + 1));
        let threads = leases
            .into_iter()
            .map(|lease| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    drop(lease);
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        callback_started_rx.recv().expect("one concurrent drop must claim and invoke the last-handle callback");
        assert_eq!(callback_count.load(Ordering::Acquire), 1);
        assert_eq!(tracker.active_handles(), 0);
        assert!(tracker.callback_fired());
        assert!(tracker.acquire().is_none(), "the fired state prevents resurrection while the callback is running");

        allow_callback_finish_tx.send(()).unwrap();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(callback_count.load(Ordering::Acquire), 1);
        assert!(tracker.callback_fired());
    }

    #[test]
    fn explicit_close_and_last_public_handle_drop_elect_one_signal_winner() {
        let lifecycle = BackendLifecycle::new();
        let winners = Arc::new(AtomicUsize::new(0));
        let callback_lifecycle = lifecycle.clone();
        let callback_winners = Arc::clone(&winners);
        let tracker = PublicHandleTracker::new(move || {
            if callback_lifecycle.close() {
                callback_winners.fetch_add(1, Ordering::AcqRel);
            }
        });
        let last_handle = tracker.acquire().expect("initial public handle");
        let barrier = Arc::new(Barrier::new(3));

        let drop_barrier = Arc::clone(&barrier);
        let drop_thread = thread::spawn(move || {
            drop_barrier.wait();
            drop(last_handle);
        });
        let close_barrier = Arc::clone(&barrier);
        let close_lifecycle = lifecycle.clone();
        let close_winners = Arc::clone(&winners);
        let close_thread = thread::spawn(move || {
            close_barrier.wait();
            if close_lifecycle.close() {
                close_winners.fetch_add(1, Ordering::AcqRel);
            }
        });

        barrier.wait();
        drop_thread.join().unwrap();
        close_thread.join().unwrap();
        assert_eq!(winners.load(Ordering::Acquire), 1);
        assert!(tracker.callback_fired());
        assert!(lifecycle.is_closing());
    }
}
