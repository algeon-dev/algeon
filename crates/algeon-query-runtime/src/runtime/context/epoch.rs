//! Revocable service and query resource epochs.
//!
//! A `QueryServiceEpoch` spans a single query service. `QueryService::close` and
//! cleanup procedures invalidate the epoch, awaiting user completion. A
//! `QueryResourceEpoch` spans an admitted query handle; handle release
//! invalidates the epoch and dispatches resource cleanup via an idle callback.
//! Epochs close irreversibly and do not reopen.
//!
//! Neither epoch holds a lock protecting `open` state and active user counts.
//! Acquisition increments the count and re-verifies `open`; users created within
//! a closing window are dropped immediately. Consequently, an observer noting a
//! zero count following a close store cannot be preceded by surviving users.
//! Idle completion marks the transition to zero active users following closure,
//! triggering the registered one-shot callback and waking `wait_for_idle_until`.

use std::{
    fmt, sync,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use crate::runtime::deadline::{DeadlineMutexLockError, deadline_expired, deadline_remaining, mutex_lock_until};

#[cfg(test)]
mod tests;

type QueryServiceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;
type QueryResourceEpochIdleCallback = Arc<dyn Fn() + Send + Sync>;

/// Revocable lifetime shared by every resource derived from one query service.
#[derive(Clone)]
pub(crate) struct QueryServiceEpoch {
    inner: Arc<QueryServiceEpochInner>,
}

struct QueryServiceEpochInner {
    /// Cleared once by `invalidate`, never set again.
    open: AtomicBool,
    /// Live `QueryServiceEpochUser`s, including the `_user` inside every
    /// `QueryServiceResourceEpoch`.
    active_users: AtomicUsize,
    /// Guards the one-shot callback and the completion flag that
    /// `wait_for_idle_until` sleeps on.
    idle_state: Mutex<QueryServiceEpochIdleState>,
    idle_changed: Condvar,
}

struct QueryServiceEpochIdleState {
    on_idle: Option<QueryServiceEpochIdleCallback>,
    /// Set once, under `idle_state`, after the callback has run.
    complete: bool,
}

impl fmt::Debug for QueryServiceEpoch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueryServiceEpoch")
            .field("open", &self.inner.open.load(Ordering::Acquire))
            .field("active_users", &self.inner.active_users.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl QueryServiceEpoch {
    pub(crate) fn new(on_idle: Option<QueryServiceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryServiceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                idle_state: Mutex::new(QueryServiceEpochIdleState { on_idle, complete: false }),
                idle_changed: Condvar::new(),
            }),
        }
    }

    /// Adds a user without checking `open`. For callers whose own membership
    /// already proves the epoch admitted them, such as a query handle
    /// retaining the service for its lifetime, and for
    /// `QueryServiceResourceEpoch::service_generation_only`, which re-checks
    /// around the call.
    pub(crate) fn retain_existing_user(&self) -> QueryServiceEpochUser {
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        QueryServiceEpochUser { epoch: self.clone() }
    }

    /// Takes a counted user if the epoch is open. The re-check after the
    /// increment is what makes the count a reliable drain census; see the
    /// module docs.
    fn acquire(&self, operation: &'static str) -> Result<QueryServiceEpochUser> {
        if !self.inner.open.load(Ordering::Acquire) {
            return Err(query_service_epoch_closed(operation));
        }
        let user = self.retain_existing_user();
        if !self.inner.open.load(Ordering::Acquire) {
            return Err(query_service_epoch_closed(operation));
        }
        Ok(user)
    }

    pub(crate) fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_service_epoch_closed(operation)) }
    }

    /// Closes the epoch and wakes idle waiters. Idempotent. Idle completion
    /// itself runs on the last user release or inside `wait_for_idle_until`,
    /// so an epoch closed with no users and no waiter runs its callback only
    /// when a waiter arrives.
    pub(crate) fn invalidate(&self) {
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

    /// Runs the idle callback and records completion, once, when the epoch is
    /// closed with no users. The callback runs under `idle_state`, which
    /// serializes it against a `wait_for_idle_until` caller reaching the same
    /// conclusion at the same time.
    fn release_if_idle(&self) {
        if self.inner.open.load(Ordering::Acquire) || self.inner.active_users.load(Ordering::Acquire) != 0 {
            return;
        }
        let mut state = self.inner.idle_state.lock().unwrap_or_else(sync::PoisonError::into_inner);
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

    /// Blocks until the epoch is closed with no users, or `deadline` passes.
    ///
    /// The waiter completes idle itself when it finds the condition already
    /// met, so the callback is not lost when the last release happened before
    /// any waiter existed. A completion first observed after the deadline, by
    /// a caller that had to wait or that completed it here, is still reported
    /// as the timeout error, which carries the live user count and the
    /// cleanup-deadline marker. A caller that finds completion already
    /// recorded returns `Ok` whatever the deadline.
    pub(crate) fn wait_for_idle_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        let mut state = match mutex_lock_until(&self.inner.idle_state, deadline) {
            Ok(state) => state,
            Err(DeadlineMutexLockError::Poisoned) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service resource epoch idle mutex was poisoned during shutdown",
                ));
            },
            Err(DeadlineMutexLockError::TimedOut) => {
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
                if (waited_for_idle || completed_here) && deadline_expired(deadline) {
                    return Err(query_service_epoch_idle_timeout(
                        operation,
                        self.inner.active_users.load(Ordering::Acquire),
                    ));
                }
                return Ok(());
            }
            if deadline_expired(deadline) {
                return Err(query_service_epoch_idle_timeout(
                    operation,
                    self.inner.active_users.load(Ordering::Acquire),
                ));
            }
            let Some(remaining) = deadline_remaining(deadline) else {
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

/// Counted membership in a service epoch. Dropping the last user after the
/// close completes idle.
#[derive(Debug)]
pub(crate) struct QueryServiceEpochUser {
    epoch: QueryServiceEpoch,
}

/// Revocable lifetime shared by resources derived from one admitted query.
///
/// The epoch owns only revocation and drain accounting. Each counted anchor
/// owns the query lifetime separately, so the epoch cannot form a cycle with
/// the query's uncounted root-context template.
#[derive(Clone)]
pub(crate) struct QueryResourceEpoch {
    inner: Arc<QueryResourceEpochInner>,
}

struct QueryResourceEpochInner {
    /// Cleared once by `invalidate`.
    open: AtomicBool,
    /// Live `QueryResourceAnchor`s.
    active_users: AtomicUsize,
    /// Taken once, by whichever of `invalidate` and the last anchor drop finds
    /// the epoch closed and idle.
    on_idle: Mutex<Option<QueryResourceEpochIdleCallback>>,
}

impl fmt::Debug for QueryResourceEpoch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QueryResourceEpoch")
            .field("open", &self.inner.open.load(Ordering::Acquire))
            .field("active_users", &self.inner.active_users.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl QueryResourceEpoch {
    pub(crate) fn new(on_idle: Option<QueryResourceEpochIdleCallback>) -> Self {
        Self {
            inner: Arc::new(QueryResourceEpochInner {
                open: AtomicBool::new(true),
                active_users: AtomicUsize::new(0),
                on_idle: Mutex::new(on_idle),
            }),
        }
    }

    /// Mints a counted anchor that keeps `owner` alive. The anchor holds the
    /// owner and the epoch holds none, so an owner may hold the epoch without
    /// a reference cycle; the query handle lifetime passes itself here. Fails
    /// with `Cancelled` and reason `query_handle_released` once closed.
    pub(crate) fn acquire(
        &self,
        owner: Arc<dyn Send + Sync>,
        operation: &'static str,
    ) -> Result<Arc<QueryResourceAnchor>> {
        self.check_open(operation)?;
        self.inner.active_users.fetch_add(1, Ordering::AcqRel);
        let anchor = Arc::new(QueryResourceAnchor { epoch: self.clone(), _owner: owner });
        self.check_open(operation)?;
        Ok(anchor)
    }

    pub(crate) fn check_open(&self, operation: &'static str) -> Result<()> {
        if self.inner.open.load(Ordering::Acquire) { Ok(()) } else { Err(query_resource_epoch_closed(operation)) }
    }

    /// Closes the epoch and, when no anchor is live, runs the idle callback
    /// at once. Otherwise the last anchor drop runs it.
    pub(crate) fn invalidate(&self) {
        self.inner.open.store(false, Ordering::Release);
        self.notify_if_idle();
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.inner.active_users.load(Ordering::Acquire) == 0
    }

    fn release_user(&self) {
        let previous = self.inner.active_users.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "query resource epoch user count underflow");
        if previous == 1 {
            self.notify_if_idle();
        }
    }

    /// Runs the idle callback once the epoch is closed with no live anchor.
    /// The callback runs after the `on_idle` lock is released because it
    /// releases the query handle's envelope, which drops further resources.
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
/// Clones of the surrounding context/resource share this anchor. The anchor
/// holds the owner, and [`QueryResourceEpoch`] holds none.
pub(crate) struct QueryResourceAnchor {
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

impl fmt::Debug for QueryResourceAnchor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
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
///
/// `_user` keeps the service epoch from reaching idle while the resource
/// lives; `query_anchor` does the same for the query handle when the resource
/// was derived from an outward context. `check_open` consults both, so a pool
/// clone, grant clone, or stream lease bound to this epoch fails after either
/// close.
#[derive(Debug)]
pub(crate) struct QueryServiceResourceEpoch {
    epoch: QueryServiceEpoch,
    _user: QueryServiceEpochUser,
    query_anchor: Option<Arc<QueryResourceAnchor>>,
}

impl QueryServiceResourceEpoch {
    /// Takes a service user and, when an anchor is given, verifies the query
    /// epoch is still open. Fails when either lifetime is closed; the user is
    /// released on the anchor check failing, so a failed acquisition leaves
    /// the drain census unchanged.
    pub(crate) fn acquire(
        epoch: &QueryServiceEpoch,
        query_anchor: Option<&Arc<QueryResourceAnchor>>,
        operation: &'static str,
    ) -> Result<Arc<Self>> {
        let user = epoch.acquire(operation)?;
        let query_anchor = query_anchor.cloned();
        if let Some(anchor) = &query_anchor
            && let Err(error) = anchor.check_open(operation)
        {
            return Err(error);
        }
        Ok(Arc::new(Self { epoch: epoch.clone(), _user: user, query_anchor }))
    }

    pub(crate) fn check_open(&self, operation: &'static str) -> Result<()> {
        self.epoch.check_open(operation)?;
        if let Some(anchor) = &self.query_anchor {
            anchor.check_open(operation)?;
        }
        Ok(())
    }

    /// Whether this lease also retains its query anchor.
    #[cfg(test)]
    pub(crate) fn has_query_anchor(&self) -> bool {
        self.query_anchor.is_some()
    }

    /// Shares only the backend service generation with a resident owner.
    ///
    /// Frozen residents outlive their attempt, so they must not retain the
    /// query anchor: keeping the query epoch alive would stall report
    /// publication and the frozen-resident handoff. The returned lease keeps
    /// the device generation alive until the retained GPU bytes release.
    pub(crate) fn service_generation_only(&self, operation: &'static str) -> Result<Arc<Self>> {
        self.epoch.check_open(operation)?;
        let user = self.epoch.retain_existing_user();
        // Re-check after retaining: a close that began between the two checks
        // must not return a lease that would keep the generation alive
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
