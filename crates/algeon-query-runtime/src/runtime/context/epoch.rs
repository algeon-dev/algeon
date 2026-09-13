//! Revocable service and query resource epochs.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

#[cfg(test)]
#[path = "query_service_epoch_tests.rs"]
mod query_service_epoch_tests;

type QueryServiceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;
type QueryResourceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;

/// Revocable lifetime shared by every resource derived from one query service.
#[derive(Clone)]
#[doc(hidden)]
pub struct QueryServiceEpoch {
    inner: Arc<QueryServiceEpochInner>,
}

struct QueryServiceEpochInner {
    open: AtomicBool,
    active_users: AtomicUsize,
    idle_state: Mutex<QueryServiceEpochIdleState>,
    idle_changed: Condvar,
}

struct QueryServiceEpochIdleState {
    on_idle: Option<QueryServiceEpochIdleCallback>,
    complete: bool,
}

impl std::fmt::Debug for QueryServiceEpoch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QueryServiceEpoch")
            .field("open", &self.inner.open.load(Ordering::Acquire))
            .field("active_users", &self.inner.active_users.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl QueryServiceEpoch {
    #[doc(hidden)]
    pub fn new(on_idle: Option<QueryServiceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryServiceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                idle_state: Mutex::new(QueryServiceEpochIdleState { on_idle, complete: false }),
                idle_changed: Condvar::new(),
            }),
        }
    }

    #[doc(hidden)]
    pub fn retain_existing_user(&self) -> QueryServiceEpochUser {
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        QueryServiceEpochUser { epoch: self.clone() }
    }

    fn acquire(&self, operation: &'static str) -> Result<QueryServiceEpochUser> {
        if !self.inner.open.load(Ordering::Acquire) {
            return Err(query_service_epoch_closed(operation));
        }
        let user = self.retain_existing_user();
        if !self.inner.open.load(Ordering::Acquire) {
            drop(user);
            return Err(query_service_epoch_closed(operation));
        }
        Ok(user)
    }

    #[doc(hidden)]
    pub fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_service_epoch_closed(operation)) }
    }

    #[doc(hidden)]
    pub fn invalidate(&self) {
        self.inner.open.store(false, Ordering::Release);
        self.inner.idle_changed.notify_all();
    }

    fn release_user(&self) {
        let previous = self.inner.active_users.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "query service epoch user count underflow");
        if previous == 1 {
            self.release_if_idle();
        }
    }

    fn release_if_idle(&self) {
        if self.inner.open.load(Ordering::Acquire) || self.inner.active_users.load(Ordering::Acquire) != 0 {
            return;
        }
        let mut state = self.inner.idle_state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.complete_idle_locked(&mut state);
    }

    fn complete_idle_locked(&self, state: &mut QueryServiceEpochIdleState) {
        if state.complete
            || self.inner.open.load(Ordering::Acquire)
            || self.inner.active_users.load(Ordering::Acquire) != 0
        {
            return;
        }
        let callback = state.on_idle.take();
        if let Some(callback) = callback {
            callback();
        }
        state.complete = true;
        self.inner.idle_changed.notify_all();
    }

    #[doc(hidden)]
    pub fn wait_for_idle_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let mut state = match crate::runtime::deadline::mutex_lock_until(&self.inner.idle_state, deadline) {
            Ok(state) => state,
            Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service resource epoch idle mutex was poisoned during shutdown",
                ));
            },
            Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            },
        };
        let mut waited_for_idle = false;
        let mut completed_here = false;
        loop {
            if !state.complete
                && !self.inner.open.load(Ordering::Acquire)
                && self.inner.active_users.load(Ordering::Acquire) == 0
            {
                self.complete_idle_locked(&mut state);
                completed_here = true;
            }
            if state.complete {
                if (waited_for_idle || completed_here) && crate::runtime::deadline::deadline_expired(deadline) {
                    return Err(query_service_epoch_idle_timeout(
                        operation,
                        self.inner.active_users.load(Ordering::Acquire),
                    ));
                }
                return Ok(());
            }
            if crate::runtime::deadline::deadline_expired(deadline) {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            }
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            };
            waited_for_idle = true;
            let (next, _timeout) = self.inner.idle_changed.wait_timeout(state, remaining).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service resource epoch condition variable was poisoned during shutdown",
                )
            })?;
            state = next;
        }
    }
}

fn query_service_epoch_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "query service execution epoch is closed")
        .with_fact("reason", "query_service_epoch_closed")
}

fn query_service_epoch_idle_timeout(operation: &'static str, active_users: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query service resource epoch to become idle",
    )
    .with_fact("reason", "query_service_resource_epoch_timeout")
    .with_fact("active_resource_users", active_users)
    .with_cleanup_deadline_elapsed()
}

#[derive(Debug)]
#[doc(hidden)]
pub struct QueryServiceEpochUser {
    epoch: QueryServiceEpoch,
}

/// Revocable lifetime shared by resources derived from one admitted query.
///
/// The epoch owns only revocation and drain accounting. Each counted anchor
/// owns the query lifetime separately, so the epoch cannot form a cycle with
/// the query's uncounted root-context template.
#[derive(Clone)]
#[doc(hidden)]
pub struct QueryResourceEpoch {
    inner: Arc<QueryResourceEpochInner>,
}

struct QueryResourceEpochInner {
    open: AtomicBool,
    active_users: AtomicUsize,
    on_idle: Mutex<Option<QueryResourceEpochIdleCallback>>,
}

impl std::fmt::Debug for QueryResourceEpoch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QueryResourceEpoch")
            .field("open", &self.inner.open.load(Ordering::Acquire))
            .field("active_users", &self.inner.active_users.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl QueryResourceEpoch {
    #[doc(hidden)]
    pub fn new(on_idle: Option<QueryResourceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryResourceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                on_idle: Mutex::new(on_idle),
            }),
        }
    }

    #[doc(hidden)]
    pub fn acquire(&self, owner: Arc<dyn Send + Sync>, operation: &'static str) -> Result<Arc<QueryResourceAnchor>> {
        self.check_open(operation)?;
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        let anchor = Arc::new(QueryResourceAnchor { epoch: self.clone(), _owner: owner });
        if let Err(error) = self.check_open(operation) {
            drop(anchor);
            return Err(error);
        }
        Ok(anchor)
    }

    #[doc(hidden)]
    pub fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_resource_epoch_closed(operation)) }
    }

    #[doc(hidden)]
    pub fn invalidate(&self) {
        self.inner.open.store(false, Ordering::Release);
        self.notify_if_idle();
    }

    #[doc(hidden)]
    pub fn is_idle(&self) -> bool {
        self.inner.active_users.load(Ordering::Acquire) == 0
    }

    fn release_user(&self) {
        let previous = self.inner.active_users.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "query resource epoch user count underflow");
        if previous == 1 {
            self.notify_if_idle();
        }
    }

    fn notify_if_idle(&self) {
        if self.inner.open.load(Ordering::Acquire) || !self.is_idle() {
            return;
        }
        let callback = match self.inner.on_idle.lock() {
            Ok(mut callback) => callback.take(),
            Err(error) => error.into_inner().take(),
        };
        if let Some(callback) = callback {
            callback();
        }
    }
}

/// One counted outward ownership root for an admitted query.
///
/// Clones of the surrounding context/resource share this anchor. The owner is
/// deliberately held by the anchor rather than by [`QueryResourceEpoch`].
#[doc(hidden)]
pub struct QueryResourceAnchor {
    epoch: QueryResourceEpoch,
    _owner: Arc<dyn Send + Sync>,
}

impl QueryResourceAnchor {
    fn check_open(&self, operation: &'static str) -> Result<()> {
        self.epoch.check_open(operation)
    }

    pub(super) fn belongs_to(&self, epoch: &QueryResourceEpoch) -> bool {
        Arc::ptr_eq(&self.epoch.inner, &epoch.inner)
    }
}

impl std::fmt::Debug for QueryResourceAnchor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("QueryResourceAnchor").field("epoch", &self.epoch).finish_non_exhaustive()
    }
}

impl Drop for QueryResourceAnchor {
    fn drop(&mut self) {
        // Drop runs while `_owner` is still live, so the idle callback can
        // always upgrade its weak query-lifetime reference.
        self.epoch.release_user();
    }
}

fn query_resource_epoch_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "query resource epoch is closed")
        .with_fact("reason", "query_handle_released")
}

/// Shared counted owner attached to cloneable service-derived resources.
#[derive(Debug)]
#[doc(hidden)]
pub struct QueryServiceResourceEpoch {
    epoch: QueryServiceEpoch,
    _user: QueryServiceEpochUser,
    query_anchor: Option<Arc<QueryResourceAnchor>>,
}

impl QueryServiceResourceEpoch {
    #[doc(hidden)]
    pub fn acquire(
        epoch: &QueryServiceEpoch,
        query_anchor: Option<&Arc<QueryResourceAnchor>>,
        operation: &'static str,
    ) -> Result<Arc<Self>> {
        let user = epoch.acquire(operation)?;
        let query_anchor = query_anchor.cloned();
        if let Some(anchor) = &query_anchor
            && let Err(error) = anchor.check_open(operation)
        {
            drop(user);
            return Err(error);
        }
        Ok(Arc::new(Self { epoch: epoch.clone(), _user: user, query_anchor }))
    }

    #[doc(hidden)]
    pub fn check_open(&self, operation: &'static str) -> Result<()> {
        self.epoch.check_open(operation)?;
        if let Some(anchor) = &self.query_anchor {
            anchor.check_open(operation)?;
        }
        Ok(())
    }

    /// Whether this lease also retains its query anchor.
    #[doc(hidden)]
    pub fn has_query_anchor(&self) -> bool {
        self.query_anchor.is_some()
    }

    /// Shares only the backend service generation with a resident owner.
    ///
    /// Frozen residents outlive their attempt, so they must not retain the
    /// query anchor: keeping the query epoch alive would stall report
    /// publication and the frozen-resident handoff. The returned lease keeps
    /// the device generation alive until the retained GPU bytes release.
    #[doc(hidden)]
    pub fn service_generation_only(&self, operation: &'static str) -> Result<Arc<Self>> {
        self.epoch.check_open(operation)?;
        let user = self.epoch.retain_existing_user();
        // Re-check after retaining: a close that began between the two checks
        // must not hand back a lease that would keep the generation alive
        // past its own teardown. Failing here drops `user` and releases it.
        self.epoch.check_open(operation)?;
        Ok(Arc::new(Self { epoch: self.epoch.clone(), _user: user, query_anchor: None }))
    }
}

impl Drop for QueryServiceEpochUser {
    fn drop(&mut self) {
        self.epoch.release_user();
    }
}
