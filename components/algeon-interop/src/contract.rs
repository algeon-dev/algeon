use core::fmt;
use std::sync::Arc;

use crate::{CompletionAuthority, ProducerReadiness, ReadStartAuthority, types::Result};

/// Backend operations supported by a producer-owned completion witness.
///
/// A witness records the device-queue point after which the consumer's reads
/// of producer storage are complete. Backends are `Send + Sync` and must
/// support shared calls from any thread, so each call binds the device context
/// its native handle requires instead of relying on the caller's current
/// device. This does not make an active lease or a consumer wrapper that holds
/// a witness usable across threads.
pub trait CompletionWitnessBackend: Send + Sync {
    /// Queries, without waiting, whether the recorded completion point has
    /// been reached.
    ///
    /// `Ok(false)` means consumer work may still be pending. `Err` means the
    /// query itself failed and must not be treated as ready.
    fn is_ready(&self) -> Result<bool>;

    /// Blocks the calling thread until the recorded completion point has been
    /// reached.
    ///
    /// `Ok(())` establishes completion. `Err` means completion was not
    /// established, so it does not authorize releasing producer storage.
    fn synchronize(&self) -> Result<()>;
}

/// Producer-owned completion witness returned by a completed lease.
#[derive(Clone)]
pub struct CompletionWitness {
    backend: Arc<dyn CompletionWitnessBackend>,
}

impl CompletionWitness {
    pub fn new(backend: Arc<dyn CompletionWitnessBackend>) -> Self {
        Self { backend }
    }

    pub fn is_ready(&self) -> Result<bool> {
        self.backend.is_ready()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }
}

impl fmt::Debug for CompletionWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionWitness").finish_non_exhaustive()
    }
}

/// Backend-owned activation state retained for the lifetime of one active
/// lease.
///
/// The lease never inspects this state. It keeps the state alive until the
/// lease itself is dropped: after a successful `complete` returns, or after
/// the incomplete-drop fallback of a lease that was not completed. A backend
/// uses it for state that must outlive activation, such as a readiness barrier
/// that orders consumer reads after the producer.
pub trait ActiveLeaseActivation: Send + Sync {}

/// Backend activation result retained by an active lease.
#[derive(Clone, Default)]
pub struct LeaseActivation {
    state: Option<Arc<dyn ActiveLeaseActivation>>,
}

impl LeaseActivation {
    pub fn noop() -> Self {
        Self { state: None }
    }

    pub fn with_state(state: Arc<dyn ActiveLeaseActivation>) -> Self {
        Self { state: Some(state) }
    }
}

impl fmt::Debug for LeaseActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeaseActivation").field("has_backend_state", &self.state.is_some()).finish()
    }
}

/// Backend implementation for shared active read leases.
///
/// One backend is shared by every lease of a prepared export and its clones,
/// and it is called from whichever thread activates, completes, or drops a
/// lease. Every call receives the export's producer readiness and the lease's
/// consumer authorities, which stay borrowed for the lease's lifetime. The
/// producer CUDA device is `readiness.device()`.
///
/// # Lease lifecycle
///
/// 1. [`activate`](Self::activate) runs while an export creates a lease. The
///    lease exists only if it returns `Ok`. After `Err` no lease exists, so
///    neither [`complete`](Self::complete) nor
///    [`drop_incomplete`](Self::drop_incomplete) runs for that attempt; the
///    backend must clean up any partial activation before returning.
/// 2. [`complete`](Self::complete) runs when the consumer completes the lease.
///    The lease is marked completed only when it returns `Ok`.
/// 3. A lease dropped without being marked completed, including one whose
///    `complete` returned `Err`, calls
///    [`drop_incomplete`](Self::drop_incomplete) from `Drop`. The backend must
///    therefore handle a fallback after `complete` has done part of its work
///    and failed.
pub trait ActiveLeaseBackend: Send + Sync {
    /// Starts consumer access for one lease.
    ///
    /// `read_start` orders consumer reads after `readiness`, and `completion`
    /// is the authority later passed to `complete` or `drop_incomplete` for the
    /// same lease. The returned [`LeaseActivation`] is retained until the lease
    /// is dropped. The default performs no backend work and retains no state.
    fn activate(
        &self,
        _readiness: ProducerReadiness<'_>,
        _read_start: ReadStartAuthority<'_>,
        _completion: CompletionAuthority<'_>,
    ) -> Result<LeaseActivation> {
        Ok(LeaseActivation::noop())
    }

    /// Establishes completion for one lease.
    ///
    /// `Some(witness)` means a completion point was recorded, not that
    /// consumer device work has finished; the lease holds the export's retained
    /// owner until the witness is synchronized. `None` means completion was
    /// established before returning. `Err` leaves the lease incomplete.
    fn complete(
        &self,
        readiness: ProducerReadiness<'_>,
        completion: CompletionAuthority<'_>,
    ) -> Result<Option<CompletionWitness>>;

    /// Drop-time fallback for a lease that was not completed.
    ///
    /// Called from the lease's `Drop`, which may run during unwinding; it must
    /// not panic. The lease discards the returned `Result`. Completion is
    /// established only when this returns `Ok`: after `Err`, consumer reads may
    /// still be pending even though the lease, and with it the borrow of the
    /// prepared export, has ended.
    fn drop_incomplete(&self, readiness: ProducerReadiness<'_>, completion: CompletionAuthority<'_>) -> Result<()>;
}

/// Shared backend descriptor attached to prepared exports.
#[derive(Clone)]
pub struct LeaseBackend {
    inner: Arc<dyn ActiveLeaseBackend>,
}

impl LeaseBackend {
    pub fn new<B>(backend: B) -> Self
    where B: ActiveLeaseBackend + 'static {
        Self { inner: Arc::new(backend) }
    }

    pub(crate) fn activate(
        &self,
        readiness: ProducerReadiness<'_>,
        read_start: ReadStartAuthority<'_>,
        completion: CompletionAuthority<'_>,
    ) -> Result<LeaseActivation> {
        self.inner.activate(readiness, read_start, completion)
    }

    pub(crate) fn complete(
        &self,
        readiness: ProducerReadiness<'_>,
        completion: CompletionAuthority<'_>,
    ) -> Result<Option<CompletionWitness>> {
        self.inner.complete(readiness, completion)
    }

    pub(crate) fn drop_incomplete(
        &self,
        readiness: ProducerReadiness<'_>,
        completion: CompletionAuthority<'_>,
    ) -> Result<()> {
        self.inner.drop_incomplete(readiness, completion)
    }
}

impl fmt::Debug for LeaseBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeaseBackend").finish_non_exhaustive()
    }
}

/// Placement of a prepared export's bytes, derived from its
/// [`PreparedAccess`](crate::PreparedAccess).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedExportProfile {
    Device,
    Host,
}
