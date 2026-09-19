//! Bounded, post-callback attribution for cuDF allocator-ring records.
//!
//! The allocator callback cannot inspect engine state.  Instead, the engine
//! records the exact domain identity that an admitted envelope owns and joins
//! ring records only after the callback has completed. This module deliberately
//! retains no domain handles; terminal finalization may request one bounded,
//! exactly joined record without creating a second event store.

#[cfg(feature = "cugraph")]
use std::sync::Arc;
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    sync::Mutex,
};

use crate::observability::AttemptId;

const DEFAULT_MAX_LIVE_ENTRIES: usize = 1024;
const DEFAULT_MAX_RETIRED_ENTRIES: usize = 1024;

/// Attempt-owned carrier for optional resident-cache allocator correlation.
///
/// The carrier keeps cugraph-specific allocation evidence out of the general
/// query-service and execution-context data flow. Direct contexts and builds
/// without cugraph carry an inert value, while admitted cugraph attempts share
/// one collector across every context clone until terminal report commit.
#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct AttemptCacheScopeAllocatorCorrelation {
    #[cfg(feature = "cugraph")]
    collector: Option<Arc<AttemptCacheScopeCorrelationCollector>>,
}

impl AttemptCacheScopeAllocatorCorrelation {
    #[doc(hidden)]
    pub fn for_admitted_attempt(attempt_id: Option<AttemptId>) -> Self {
        #[cfg(feature = "cugraph")]
        {
            Self { collector: attempt_id.is_some().then(AttemptCacheScopeCorrelationCollector::new) }
        }
        #[cfg(not(feature = "cugraph"))]
        {
            let _ = attempt_id;
            Self {}
        }
    }

    #[doc(hidden)]
    pub fn seal_and_take_exact_event(&self) -> Option<cudf::AllocatorEvent> {
        #[cfg(feature = "cugraph")]
        {
            self.collector.as_ref().and_then(|collector| collector.seal_and_take_exact_event())
        }
        #[cfg(not(feature = "cugraph"))]
        {
            None
        }
    }

    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn begin(&self, scope: Arc<cudf::CacheAllocationScope>) -> Option<CacheScopeCorrelationGuard> {
        self.collector.as_ref().map(|collector| collector.begin(scope))
    }
}

/// Exact identity of one cache scope's private allocator-event ring.
///
/// Cache scopes remain alive across multiple graph checkouts, unlike admitted
/// query domains. A correlation therefore owns one explicit sequence window
/// per checkout rather than retiring the scope identity after its first use.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CacheScopeAllocatorIdentity {
    device_ordinal: usize,
    cache_domain_id: u64,
    scope_id: u64,
    generation: u64,
}

#[cfg(feature = "cugraph")]
impl CacheScopeAllocatorIdentity {
    fn from_cudf(identity: cudf::CacheAllocationScopeIdentity) -> Self {
        Self {
            device_ordinal: identity.device.ordinal(),
            cache_domain_id: identity.domain_id.get(),
            scope_id: identity.scope_id.get(),
            generation: identity.generation,
        }
    }

    fn matches(self, event: cudf::AllocatorEvent) -> bool {
        event.completeness.allocation_domain
            && event.device.is_some_and(|device| device.ordinal() == self.device_ordinal)
            && matches!(
                event.allocation_domain,
                Some(cudf::AllocatorEventDomainIdentity {
                    domain:
                        cudf::DeviceMemoryFailureAllocationDomain::CacheAllocationScope {
                            cache_domain_id,
                            scope_id,
                        },
                    generation,
                }) if cache_domain_id == self.cache_domain_id
                    && scope_id == self.scope_id
                    && generation == self.generation
            )
    }
}

/// Attempt-local collector for exact cache-scope allocator failures.
///
/// A collector exists only on an admitted attempt context. Its guards can open
/// one window while a resident graph checkout remains exclusive, then retain a
/// single exact event for final report projection. Events from different scope
/// rings have no common sequence order, so competing exact candidates are
/// deliberately unavailable instead of arbitrarily choosing one.
#[cfg(feature = "cugraph")]
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct AttemptCacheScopeCorrelationCollector {
    state: Mutex<AttemptCacheScopeCorrelationState>,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Default)]
struct AttemptCacheScopeCorrelationState {
    active: BTreeSet<CacheScopeAllocatorIdentity>,
    exact: Option<(CacheScopeAllocatorIdentity, cudf::AllocatorEvent)>,
    ambiguous_exact_event: bool,
    /// Once a window cannot prove its complete private-ring interval, this
    /// attempt must not substitute a cache event for its admitted-domain
    /// evidence. The collector is deliberately fail-closed across its
    /// lifetime: the unknown window could contain the terminal failure.
    discarded: bool,
    /// The terminal report commit closes admission to new windows before it
    /// consumes evidence. Context clones can retain the collector, so this is
    /// an explicit state transition rather than an ownership assumption.
    sealed: bool,
}

/// Outcome of reading one fully bounded private allocator-event interval.
///
/// A window with no failures is different from an invalid window: a later
/// sequential checkout of the same scope may still supply exact evidence only
/// in the former case. Every unstable, overwritten, malformed, or incomplete
/// read makes the whole attempt's cache fallback unavailable.
#[cfg(feature = "cugraph")]
#[derive(Debug, PartialEq, Eq)]
enum CacheScopeWindowEvidence {
    NoEvent,
    Exact(cudf::AllocatorEvent),
    Discard,
}

#[cfg(feature = "cugraph")]
impl AttemptCacheScopeCorrelationCollector {
    #[doc(hidden)]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Opens a scope-private interval after a resident checkout has made the
    /// scope exclusive. If native bounds cannot be read or the same scope is
    /// already active, the returned guard records no evidence and never
    /// changes execution behavior. An unavailable lower bound is still a
    /// terminal evidence failure for this attempt: a later cache event cannot
    /// safely prove it was the terminal allocation failure.
    #[doc(hidden)]
    pub fn begin(self: &Arc<Self>, scope: Arc<cudf::CacheAllocationScope>) -> CacheScopeCorrelationGuard {
        let identity = CacheScopeAllocatorIdentity::from_cudf(scope.identity());
        let Some(observed_after_sequence) =
            scope.allocator_event_ring_bounds().ok().and_then(|bounds| bounds.next_sequence.checked_sub(1))
        else {
            self.discard();
            return CacheScopeCorrelationGuard::inactive(Arc::clone(self));
        };
        if !self.register_active_window(identity) {
            return CacheScopeCorrelationGuard::inactive(Arc::clone(self));
        }
        CacheScopeCorrelationGuard {
            collector: Arc::clone(self),
            scope: Some(scope),
            identity: Some(identity),
            observed_after_sequence,
        }
    }

    /// Seals the attempt-owned collector before final report commit and
    /// transfers its one unambiguous exact cache event. Active, dropped, or
    /// double-sealed windows are unavailable: the report must never race a
    /// later context clone or infer a terminal event around an unclosed
    /// interval.
    #[doc(hidden)]
    pub fn seal_and_take_exact_event(&self) -> Option<cudf::AllocatorEvent> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.sealed {
            state.discarded = true;
            state.exact = None;
            return None;
        }
        state.sealed = true;
        if !state.active.is_empty() {
            state.active.clear();
            state.discarded = true;
            state.exact = None;
        }
        (!state.discarded && !state.ambiguous_exact_event).then(|| state.exact.take().map(|(_, event)| event)).flatten()
    }

    fn finish_window(&self, identity: CacheScopeAllocatorIdentity, evidence: CacheScopeWindowEvidence) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.sealed || !state.active.remove(&identity) {
            state.discarded = true;
            state.exact = None;
            return;
        }
        let event = match evidence {
            CacheScopeWindowEvidence::NoEvent => return,
            CacheScopeWindowEvidence::Exact(event) => event,
            CacheScopeWindowEvidence::Discard => {
                state.discarded = true;
                state.exact = None;
                return;
            },
        };
        if state.discarded {
            return;
        }
        if state.ambiguous_exact_event {
            return;
        }
        match state.exact {
            None => state.exact = Some((identity, event)),
            // Sequence ordering is meaningful within one scope ring, so a
            // later checkout of that same graph can replace an earlier event.
            Some((existing_identity, existing)) if existing_identity == identity => {
                if event.sequence > existing.sequence {
                    state.exact = Some((identity, event));
                }
            },
            // Separate scope rings have independent sequences; the finalizer
            // cannot know which exact event caused a later terminal outcome.
            Some(_) => {
                state.exact = None;
                state.ambiguous_exact_event = true;
            },
        }
    }

    /// A resident checkout owns the only live execution path for its scope.
    /// A duplicate registration means a caller could no longer prove which
    /// checkout produced a record, so it must remain unavailable.
    fn register_active_window(&self, identity: CacheScopeAllocatorIdentity) -> bool {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.sealed || state.discarded || !state.active.insert(identity) {
            state.discarded = true;
            state.exact = None;
            return false;
        }
        true
    }

    fn discard_window(&self, identity: CacheScopeAllocatorIdentity) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.active.remove(&identity) {
            state.discarded = true;
            state.exact = None;
        }
    }

    fn discard(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.discarded = true;
        state.exact = None;
    }
}

/// One exclusive resident-cache checkout correlation window.
#[cfg(feature = "cugraph")]
#[must_use = "dropping an unclosed scope window deliberately discards its allocator evidence"]
#[derive(Debug)]
#[doc(hidden)]
pub struct CacheScopeCorrelationGuard {
    collector: Arc<AttemptCacheScopeCorrelationCollector>,
    scope: Option<Arc<cudf::CacheAllocationScope>>,
    identity: Option<CacheScopeAllocatorIdentity>,
    observed_after_sequence: u64,
}

#[cfg(feature = "cugraph")]
impl CacheScopeCorrelationGuard {
    fn inactive(collector: Arc<AttemptCacheScopeCorrelationCollector>) -> Self {
        Self { collector, scope: None, identity: None, observed_after_sequence: 0 }
    }

    /// Closes the interval only after cache-runtime temporaries have been
    /// destroyed and before the checkout returns its scope to the cache.
    #[doc(hidden)]
    pub fn finish(mut self) {
        let Some(scope) = self.scope.take() else {
            return;
        };
        let identity = self.identity.take().expect("active cache correlation guard has an identity");
        let evidence = cache_scope_window_evidence(&scope, identity, self.observed_after_sequence);
        self.collector.finish_window(identity, evidence);
    }
}

#[cfg(feature = "cugraph")]
impl Drop for CacheScopeCorrelationGuard {
    fn drop(&mut self) {
        if let Some(identity) = self.identity.take() {
            self.collector.discard_window(identity);
        }
    }
}

#[cfg(feature = "cugraph")]
fn cache_scope_window_evidence(
    scope: &cudf::CacheAllocationScope,
    identity: CacheScopeAllocatorIdentity,
    observed_after_sequence: u64,
) -> CacheScopeWindowEvidence {
    let Ok(bounds) = scope.allocator_event_ring_bounds() else {
        return CacheScopeWindowEvidence::Discard;
    };
    let Some(observed_through_sequence) = bounds.next_sequence.checked_sub(1) else {
        return CacheScopeWindowEvidence::Discard;
    };
    if observed_through_sequence <= observed_after_sequence {
        return CacheScopeWindowEvidence::NoEvent;
    }
    let Some(first_sequence) = observed_after_sequence.checked_add(1) else {
        return CacheScopeWindowEvidence::Discard;
    };
    let Some(window_length) = observed_through_sequence.checked_sub(observed_after_sequence) else {
        return CacheScopeWindowEvidence::Discard;
    };
    let Some(first_available_sequence) = bounds.first_available_sequence else {
        return CacheScopeWindowEvidence::Discard;
    };
    if bounds.capacity == 0
        || window_length > bounds.capacity
        || first_available_sequence == 0
        || first_available_sequence > first_sequence
    {
        return CacheScopeWindowEvidence::Discard;
    }
    newest_exact_cache_scope_event_in_window(
        identity,
        observed_after_sequence,
        observed_through_sequence,
        (first_sequence..=observed_through_sequence).map(|sequence| (sequence, scope.read_allocator_event(sequence))),
    )
}

/// Retains only stable records whose sequence is inside one already-bounded
/// checkout interval. The iterator form isolates the no-guess boundary rules
/// from the native reader so they have CPU-only coverage.
#[cfg(feature = "cugraph")]
fn newest_exact_cache_scope_event_in_window<I, E>(
    identity: CacheScopeAllocatorIdentity,
    observed_after_sequence: u64,
    observed_through_sequence: u64,
    reads: I,
) -> CacheScopeWindowEvidence
where
    I: IntoIterator<Item = (u64, std::result::Result<cudf::AllocatorEventRead, E>)>,
{
    let Some(first_sequence) = observed_after_sequence.checked_add(1) else {
        return CacheScopeWindowEvidence::Discard;
    };
    if observed_through_sequence < first_sequence {
        return CacheScopeWindowEvidence::NoEvent;
    }
    let mut expected_sequence = first_sequence;
    let mut newest = None;
    for (sequence, read) in reads {
        if sequence != expected_sequence || sequence > observed_through_sequence {
            return CacheScopeWindowEvidence::Discard;
        }
        let Ok(cudf::AllocatorEventRead::Available(event)) = read else {
            return CacheScopeWindowEvidence::Discard;
        };
        // A malformed reader response cannot cross the caller's exact
        // sequence boundary merely because its identity happens to match.
        if event.sequence != sequence || !identity.matches(event) {
            return CacheScopeWindowEvidence::Discard;
        }
        newest = Some(event);
        expected_sequence = match expected_sequence.checked_add(1) {
            Some(next) => next,
            None if sequence == observed_through_sequence => break,
            None => return CacheScopeWindowEvidence::Discard,
        };
    }
    if expected_sequence != observed_through_sequence.saturating_add(1) {
        return CacheScopeWindowEvidence::Discard;
    }
    newest.map_or(CacheScopeWindowEvidence::NoEvent, CacheScopeWindowEvidence::Exact)
}

/// The complete allocator identity supplied by cuDF for one domain instance.
///
/// A domain id is not sufficient because allocator-domain ids may be reused.
/// The generation is therefore part of every lookup key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc(hidden)]
pub struct AllocatorDomainIdentity {
    device_ordinal: usize,
    domain_id: u64,
    generation: u64,
}

impl AllocatorDomainIdentity {
    #[doc(hidden)]
    pub const fn new(device_ordinal: usize, domain_id: u64, generation: u64) -> Self {
        Self { device_ordinal, domain_id, generation }
    }

    /// Adapts the complete identity that cuDF returns for one native domain
    /// instance. Callers must not substitute a grant id for its generation.
    #[doc(hidden)]
    pub fn from_cudf(identity: cudf::QueryMemoryDomainIdentity) -> Self {
        Self::new(identity.device.ordinal(), identity.domain_id, identity.generation)
    }
}

/// The availability result of one ring position after it is read by the engine.
///
/// `Missing` and `Overwritten` are terminal attribution states.  They must not
/// fall back to a later device sample or to another attempt-domain identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorEventAvailability {
    Available,
    Missing,
    Overwritten,
}

/// The minimum allocator-ring data needed by the correlation layer.
///
/// The adapter populates this exact input from the record's sequence, read
/// outcome, and domain identity. The remaining record is retained only after
/// this identity-only join proves it belongs to the attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct AllocatorEventCorrelationInput {
    sequence: u64,
    availability: AllocatorEventAvailability,
    domain: Option<AllocatorDomainIdentity>,
}

impl AllocatorEventCorrelationInput {
    #[cfg(test)]
    #[doc(hidden)]
    pub const fn available(sequence: u64, domain: AllocatorDomainIdentity) -> Self {
        Self { sequence, availability: AllocatorEventAvailability::Available, domain: Some(domain) }
    }

    #[doc(hidden)]
    pub const fn missing(sequence: u64) -> Self {
        Self { sequence, availability: AllocatorEventAvailability::Missing, domain: None }
    }

    #[doc(hidden)]
    pub const fn overwritten(sequence: u64) -> Self {
        Self { sequence, availability: AllocatorEventAvailability::Overwritten, domain: None }
    }

    #[doc(hidden)]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    #[doc(hidden)]
    pub const fn availability(self) -> AllocatorEventAvailability {
        self.availability
    }

    #[doc(hidden)]
    pub const fn domain(self) -> Option<AllocatorDomainIdentity> {
        self.domain
    }

    /// Adapts one exact cuDF ring read without fabricating an identity for
    /// incomplete, pending, or non-query-domain records.
    #[doc(hidden)]
    pub fn from_cudf_read(sequence: u64, read: cudf::AllocatorEventRead) -> Self {
        match read {
            cudf::AllocatorEventRead::Available(event) => {
                let domain = if event.completeness.allocation_domain {
                    match (event.device, event.allocation_domain) {
                        (
                            Some(device),
                            Some(cudf::AllocatorEventDomainIdentity {
                                domain: cudf::DeviceMemoryFailureAllocationDomain::QueryMemoryDomain { domain_id, .. },
                                generation,
                            }),
                        ) => Some(AllocatorDomainIdentity::new(device.ordinal(), domain_id, generation)),
                        _ => None,
                    }
                } else {
                    None
                };
                Self { sequence: event.sequence, availability: AllocatorEventAvailability::Available, domain }
            },
            cudf::AllocatorEventRead::Overwritten => Self::overwritten(sequence),
            // A pending producer may publish later, but it is not an event
            // available for this correlation pass. `NotYetRecorded` likewise
            // carries no event payload or identity to attribute.
            cudf::AllocatorEventRead::Pending | cudf::AllocatorEventRead::NotYetRecorded => Self::missing(sequence),
        }
    }
}

/// Whether a precisely correlated record belongs to an active or already
/// released attempt-domain interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorEventAttemptState {
    Live,
    Retired,
}

/// Why an allocator record cannot truthfully name an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorEventUnattributedReason {
    Missing,
    Overwritten,
    MissingDomainIdentity,
    MissingDomainAttempt,
    Stale,
    Ambiguous,
}

/// Result of joining one allocator-ring read to the bounded domain index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorEventCorrelation {
    Exact { attempt_id: AttemptId, state: AllocatorEventAttemptState },
    Unattributed(AllocatorEventUnattributedReason),
}

/// Outcome of registering a newly admitted attempt domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorDomainRegistration {
    Registered,
    AlreadyRegistered,
    Ambiguous,
    CapacityExhausted,
}

/// Outcome of retiring one admitted attempt domain after cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorDomainRetirement {
    Retired,
    AmbiguousStillLive,
    AmbiguousRetired,
    Missing,
    AttemptMismatch,
    InvalidSequenceBoundary,
}

/// Outcome of dropping a live mapping without a trustworthy terminal watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AllocatorDomainDiscard {
    Discarded,
    AmbiguousStillLive,
    Missing,
    AttemptMismatch,
}

/// Identity retained by an admitted envelope until cleanup establishes the
/// final ring watermark. It contains no cuDF handle, grant, or mutable attempt
/// state, so retaining it cannot extend an allocation domain's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct AllocatorDomainAttempt {
    domain: AllocatorDomainIdentity,
    attempt_id: AttemptId,
}

/// Bounded allocator-ring evidence observed while one completed domain moves
/// to its retired correlation interval. A device record that cannot be joined
/// exactly remains explicitly unattributed; it is never a substitute attempt
/// event.
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct AllocatorEventEvidenceRead {
    #[doc(hidden)]
    pub exact: Option<cudf::AllocatorEvent>,
    #[doc(hidden)]
    pub unattributed: Option<cudf::AllocatorEvent>,
}

/// Registers one just-created native domain against an engine attempt.
///
/// Failure to observe the diagnostic identity or watermark leaves the event
/// unattributed; it must never change an already-approved admission. A full
/// bounded index likewise declines the optional registration rather than
/// evicting a live attempt mapping.
#[doc(hidden)]
pub fn register_admitted_domain(
    index: &Mutex<AllocatorEventCorrelationIndex>,
    domain: &cudf::QueryMemoryDomain,
    attempt_id: AttemptId,
) -> Option<AllocatorDomainAttempt> {
    let domain_identity = AllocatorDomainIdentity::from_cudf(domain.identity().ok()?);
    let observed_after_sequence = domain.manager().allocator_event_ring_bounds().ok()?.next_sequence.saturating_sub(1);
    let binding = AllocatorDomainAttempt { domain: domain_identity, attempt_id };
    let registration = index.lock().unwrap_or_else(std::sync::PoisonError::into_inner).register_live(
        binding.domain,
        binding.attempt_id,
        observed_after_sequence,
    );
    match registration {
        AllocatorDomainRegistration::CapacityExhausted => None,
        AllocatorDomainRegistration::Registered
        | AllocatorDomainRegistration::AlreadyRegistered
        | AllocatorDomainRegistration::Ambiguous => Some(binding),
    }
}

/// Retires one attempt mapping after GPU cleanup and scans the bounded
/// retained ring interval for one exactly correlated event.
///
/// The final watermark is observed before the correlation lock is acquired;
/// every cuDF FFI read likewise happens without that mutex. Once retirement
/// succeeds, each ring read is joined through the retired interval. Missing,
/// pending, overwritten, stale, and ambiguous records remain unavailable for
/// this attempt. The newest available non-exact record is retained only as a
/// device-level, explicitly unattributed sample.
#[doc(hidden)]
pub fn retire_admitted_domain(
    index: &Mutex<AllocatorEventCorrelationIndex>,
    binding: AllocatorDomainAttempt,
    domain: &cudf::QueryMemoryDomain,
) -> AllocatorEventEvidenceRead {
    let manager = domain.manager();
    let Ok(bounds) = manager.allocator_event_ring_bounds() else {
        let _ = index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .discard_live(binding.domain, binding.attempt_id);
        return AllocatorEventEvidenceRead::default();
    };
    let observed_through_sequence = bounds.next_sequence.saturating_sub(1);
    let retired = {
        let mut index = index.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        match index.retire(binding.domain, binding.attempt_id, observed_through_sequence) {
            AllocatorDomainRetirement::Retired => true,
            AllocatorDomainRetirement::InvalidSequenceBoundary
            | AllocatorDomainRetirement::AmbiguousStillLive
            | AllocatorDomainRetirement::AmbiguousRetired
            | AllocatorDomainRetirement::Missing
            | AllocatorDomainRetirement::AttemptMismatch => {
                let _ = index.discard_live(binding.domain, binding.attempt_id);
                false
            },
        }
    };

    let Some(first_available_sequence) = bounds.first_available_sequence else {
        return AllocatorEventEvidenceRead::default();
    };
    let mut evidence = AllocatorEventEvidenceRead::default();
    for sequence in first_available_sequence..=observed_through_sequence {
        // Never hold the engine correlation mutex across a cuDF ring read.
        let Ok(read) = manager.read_allocator_event(sequence) else {
            continue;
        };
        let cudf::AllocatorEventRead::Available(event) = read else {
            continue;
        };
        let correlation = if retired {
            index
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .correlate(AllocatorEventCorrelationInput::from_cudf_read(sequence, read))
        } else {
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::MissingDomainAttempt)
        };
        match correlation {
            AllocatorEventCorrelation::Exact { attempt_id, .. } if attempt_id == binding.attempt_id => {
                // Walking in sequence order means the retained record is the
                // newest exact callback in this completed attempt interval.
                evidence.exact = Some(event);
            },
            AllocatorEventCorrelation::Exact { .. } | AllocatorEventCorrelation::Unattributed(_) => {
                // This sample belongs to the same device ring but not
                // truthfully to this attempt. Keep at most one bounded record
                // without asserting it caused the terminal outcome.
                evidence.unattributed = Some(event);
            },
        }
    }
    evidence
}

/// Removes a live mapping on an unadmitted rollback or a cleanup path whose
/// completion cannot be proven. This deliberately creates no retired window.
#[doc(hidden)]
pub fn discard_admitted_domain(index: &Mutex<AllocatorEventCorrelationIndex>, binding: AllocatorDomainAttempt) {
    let _ = index
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .discard_live(binding.domain, binding.attempt_id);
}

/// A bounded index of active and recently retired allocator domains.
///
/// `observed_after_sequence` is an exclusive boundary captured after the
/// admission-domain identity is installed.  `observed_through_sequence` is an
/// inclusive boundary captured after that domain's cleanup.  Those two
/// boundaries prevent old ring slots and later domain-id reuse from gaining an
/// attribution merely because their three-part identity happens to match.
#[derive(Debug)]
#[doc(hidden)]
pub struct AllocatorEventCorrelationIndex {
    max_live_entries: NonZeroUsize,
    max_retired_entries: NonZeroUsize,
    live: BTreeMap<AllocatorDomainIdentity, LiveBinding>,
    retired: BTreeMap<AllocatorDomainIdentity, RetiredBinding>,
    next_retirement_order: u64,
}

#[derive(Debug)]
enum LiveBinding {
    Exact {
        attempt_id: AttemptId,
        observed_after_sequence: u64,
    },
    /// A duplicate identity is unsafe even when both domains are still live.
    /// Keep the set only to release the live slot when its actual owners end;
    /// correlation is permanently unattributed for this identity.
    Ambiguous {
        attempts: BTreeSet<AttemptId>,
    },
}

#[derive(Debug)]
enum RetiredBinding {
    Exact {
        attempt_id: AttemptId,
        observed_after_sequence: u64,
        observed_through_sequence: u64,
        retirement_order: u64,
    },
    /// Once an identity has collided, retaining its old event window would be
    /// a guess.  It remains explicitly unattributed until bounded eviction.
    Ambiguous {
        retirement_order: u64,
    },
}

impl AllocatorEventCorrelationIndex {
    #[doc(hidden)]
    pub fn new(max_live_entries: NonZeroUsize, max_retired_entries: NonZeroUsize) -> Self {
        Self {
            max_live_entries,
            max_retired_entries,
            live: BTreeMap::new(),
            retired: BTreeMap::new(),
            next_retirement_order: 0,
        }
    }

    /// Registers an admitted attempt-domain identity.
    ///
    /// The caller must use a cuDF ring watermark taken after the identity was
    /// installed.  Events at or before that watermark are deliberately stale;
    /// the callback was never allowed to look up the attempt while they ran.
    #[must_use]
    #[doc(hidden)]
    pub fn register_live(
        &mut self,
        domain: AllocatorDomainIdentity,
        attempt_id: AttemptId,
        observed_after_sequence: u64,
    ) -> AllocatorDomainRegistration {
        if let Some(binding) = self.live.get_mut(&domain) {
            return match binding {
                LiveBinding::Exact { attempt_id: existing, .. } if *existing == attempt_id => {
                    AllocatorDomainRegistration::AlreadyRegistered
                },
                LiveBinding::Exact { attempt_id: existing, .. } => {
                    let attempts = BTreeSet::from([*existing, attempt_id]);
                    *binding = LiveBinding::Ambiguous { attempts };
                    AllocatorDomainRegistration::Ambiguous
                },
                LiveBinding::Ambiguous { attempts } => {
                    attempts.insert(attempt_id);
                    AllocatorDomainRegistration::Ambiguous
                },
            };
        }

        // A new live registration with an identity retained from a former
        // domain is itself unsafe. Remove the old exact retired window before
        // deciding capacity so no later event can cross the reuse boundary.
        let collided_with_retired = self.retired.remove(&domain).is_some();
        if self.live.len() == self.max_live_entries.get() {
            return AllocatorDomainRegistration::CapacityExhausted;
        }
        if collided_with_retired {
            self.live.insert(domain, LiveBinding::Ambiguous { attempts: BTreeSet::from([attempt_id]) });
            return AllocatorDomainRegistration::Ambiguous;
        }

        self.live.insert(domain, LiveBinding::Exact { attempt_id, observed_after_sequence });
        AllocatorDomainRegistration::Registered
    }

    /// Transitions a completed attempt domain into the bounded retired index.
    ///
    /// `observed_through_sequence` must be a cuDF ring watermark captured only
    /// after cleanup for this domain has completed.  It is inclusive so an
    /// event read after the attempt is gone can still correlate exactly to the
    /// retired interval.
    #[must_use]
    #[doc(hidden)]
    pub fn retire(
        &mut self,
        domain: AllocatorDomainIdentity,
        attempt_id: AttemptId,
        observed_through_sequence: u64,
    ) -> AllocatorDomainRetirement {
        let Some(binding) = self.live.remove(&domain) else {
            return AllocatorDomainRetirement::Missing;
        };
        match binding {
            LiveBinding::Exact { attempt_id: registered_attempt_id, observed_after_sequence } => {
                if registered_attempt_id != attempt_id {
                    self.live.insert(
                        domain,
                        LiveBinding::Exact { attempt_id: registered_attempt_id, observed_after_sequence },
                    );
                    return AllocatorDomainRetirement::AttemptMismatch;
                }
                if observed_through_sequence < observed_after_sequence {
                    self.live.insert(
                        domain,
                        LiveBinding::Exact { attempt_id: registered_attempt_id, observed_after_sequence },
                    );
                    return AllocatorDomainRetirement::InvalidSequenceBoundary;
                }
                let retirement_order = self.next_retirement_order();
                self.insert_retired(
                    domain,
                    RetiredBinding::Exact {
                        attempt_id,
                        observed_after_sequence,
                        observed_through_sequence,
                        retirement_order,
                    },
                );
                AllocatorDomainRetirement::Retired
            },
            LiveBinding::Ambiguous { mut attempts } => {
                if !attempts.remove(&attempt_id) {
                    self.live.insert(domain, LiveBinding::Ambiguous { attempts });
                    return AllocatorDomainRetirement::AttemptMismatch;
                }
                if !attempts.is_empty() {
                    self.live.insert(domain, LiveBinding::Ambiguous { attempts });
                    return AllocatorDomainRetirement::AmbiguousStillLive;
                }
                let retirement_order = self.next_retirement_order();
                self.insert_retired(domain, RetiredBinding::Ambiguous { retirement_order });
                AllocatorDomainRetirement::AmbiguousRetired
            },
        }
    }

    /// Correlates a ring read only when the identity and sequence lie inside
    /// one non-ambiguous admitted-domain lifetime.
    #[must_use]
    #[doc(hidden)]
    pub fn correlate(&self, event: AllocatorEventCorrelationInput) -> AllocatorEventCorrelation {
        match event.availability() {
            AllocatorEventAvailability::Missing => {
                return AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Missing);
            },
            AllocatorEventAvailability::Overwritten => {
                return AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Overwritten);
            },
            AllocatorEventAvailability::Available => {},
        }
        let Some(domain) = event.domain() else {
            return AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::MissingDomainIdentity);
        };
        if let Some(binding) = self.live.get(&domain) {
            return match binding {
                LiveBinding::Exact { attempt_id, observed_after_sequence }
                    if event.sequence() > *observed_after_sequence =>
                {
                    AllocatorEventCorrelation::Exact {
                        attempt_id: *attempt_id,
                        state: AllocatorEventAttemptState::Live,
                    }
                },
                LiveBinding::Exact { .. } => {
                    AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Stale)
                },
                LiveBinding::Ambiguous { .. } => {
                    AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Ambiguous)
                },
            };
        }
        if let Some(binding) = self.retired.get(&domain) {
            return match binding {
                RetiredBinding::Exact { attempt_id, observed_after_sequence, observed_through_sequence, .. }
                    if event.sequence() > *observed_after_sequence
                        && event.sequence() <= *observed_through_sequence =>
                {
                    AllocatorEventCorrelation::Exact {
                        attempt_id: *attempt_id,
                        state: AllocatorEventAttemptState::Retired,
                    }
                },
                RetiredBinding::Exact { .. } => {
                    AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Stale)
                },
                RetiredBinding::Ambiguous { .. } => {
                    AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Ambiguous)
                },
            };
        }
        AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::MissingDomainAttempt)
    }

    fn next_retirement_order(&mut self) -> u64 {
        let order = self.next_retirement_order;
        self.next_retirement_order = self.next_retirement_order.wrapping_add(1);
        order
    }

    /// Drops a live binding when cleanup cannot obtain a final ring watermark.
    ///
    /// It deliberately creates no retired entry: an unknown upper boundary
    /// must not become a guess about which later event belonged to this
    /// attempt. A mismatched attempt remains untouched.
    #[must_use]
    #[doc(hidden)]
    pub fn discard_live(&mut self, domain: AllocatorDomainIdentity, attempt_id: AttemptId) -> AllocatorDomainDiscard {
        let Some(binding) = self.live.remove(&domain) else {
            return AllocatorDomainDiscard::Missing;
        };
        match binding {
            LiveBinding::Exact { attempt_id: registered_attempt_id, observed_after_sequence }
                if registered_attempt_id != attempt_id =>
            {
                self.live
                    .insert(domain, LiveBinding::Exact { attempt_id: registered_attempt_id, observed_after_sequence });
                AllocatorDomainDiscard::AttemptMismatch
            },
            LiveBinding::Exact { .. } => AllocatorDomainDiscard::Discarded,
            LiveBinding::Ambiguous { mut attempts } => {
                if !attempts.remove(&attempt_id) {
                    self.live.insert(domain, LiveBinding::Ambiguous { attempts });
                    return AllocatorDomainDiscard::AttemptMismatch;
                }
                if attempts.is_empty() {
                    AllocatorDomainDiscard::Discarded
                } else {
                    self.live.insert(domain, LiveBinding::Ambiguous { attempts });
                    AllocatorDomainDiscard::AmbiguousStillLive
                }
            },
        }
    }

    fn insert_retired(&mut self, domain: AllocatorDomainIdentity, binding: RetiredBinding) {
        debug_assert!(!self.retired.contains_key(&domain), "a domain can have only one retired correlation entry");
        if self.retired.len() == self.max_retired_entries.get() {
            let oldest =
                self.retired
                    .iter()
                    .min_by_key(|(_, binding)| match binding {
                        RetiredBinding::Exact { retirement_order, .. }
                        | RetiredBinding::Ambiguous { retirement_order } => *retirement_order,
                    })
                    .map(|(identity, _)| *identity)
                    .expect("a full retired index has an oldest entry");
            self.retired.remove(&oldest);
        }
        self.retired.insert(domain, binding);
    }
}

impl Default for AllocatorEventCorrelationIndex {
    fn default() -> Self {
        Self::new(
            NonZeroUsize::new(DEFAULT_MAX_LIVE_ENTRIES).expect("default live correlation capacity is non-zero"),
            NonZeroUsize::new(DEFAULT_MAX_RETIRED_ENTRIES).expect("default retired correlation capacity is non-zero"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    fn index(live: usize, retired: usize) -> AllocatorEventCorrelationIndex {
        AllocatorEventCorrelationIndex::new(
            NonZeroUsize::new(live).expect("test live capacity is non-zero"),
            NonZeroUsize::new(retired).expect("test retired capacity is non-zero"),
        )
    }

    const DOMAIN: AllocatorDomainIdentity = AllocatorDomainIdentity::new(3, 91, 7);

    #[cfg(feature = "cugraph")]
    fn cache_scope_identity(scope_id: u64, generation: u64) -> CacheScopeAllocatorIdentity {
        CacheScopeAllocatorIdentity {
            device_ordinal: cudf::DeviceId::DEFAULT.ordinal(),
            cache_domain_id: 41,
            scope_id,
            generation,
        }
    }

    #[cfg(feature = "cugraph")]
    fn cache_scope_event(sequence: u64, identity: CacheScopeAllocatorIdentity) -> cudf::AllocatorEvent {
        cudf::AllocatorEvent {
            sequence,
            monotonic_time_ns: None,
            device: Some(cudf::DeviceId::DEFAULT),
            allocation_domain: Some(cudf::AllocatorEventDomainIdentity {
                domain: cudf::DeviceMemoryFailureAllocationDomain::CacheAllocationScope {
                    cache_domain_id: identity.cache_domain_id,
                    scope_id: identity.scope_id,
                },
                generation: identity.generation,
            }),
            requested_bytes: 4096,
            source: cudf::DeviceMemoryFailureSource::AllocationLimit,
            action: Some(cudf::AllocatorEventAction::Decline),
            callback_attempt: 0,
            trim_retry_count: 0,
            pool_usage: None,
            completeness: cudf::AllocatorEventCompleteness {
                timestamp: false,
                allocation_domain: true,
                pool_usage: false,
                action: true,
            },
        }
    }

    #[cfg(feature = "cugraph")]
    fn cache_scope_read(read: cudf::AllocatorEventRead) -> std::result::Result<cudf::AllocatorEventRead, ()> {
        Ok(read)
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_scope_window_requires_every_sequence_to_be_stable_and_exact() {
        let identity = cache_scope_identity(7, 11);
        let first = cache_scope_event(11, identity);
        let exact = cache_scope_event(12, identity);
        let last = cache_scope_event(13, identity);

        assert_eq!(
            newest_exact_cache_scope_event_in_window(
                identity,
                10,
                13,
                [
                    (11, cache_scope_read(cudf::AllocatorEventRead::Available(first))),
                    (12, cache_scope_read(cudf::AllocatorEventRead::Available(exact))),
                    (13, cache_scope_read(cudf::AllocatorEventRead::Available(last))),
                ],
            ),
            CacheScopeWindowEvidence::Exact(last)
        );

        for unstable in [
            cudf::AllocatorEventRead::Pending,
            cudf::AllocatorEventRead::Overwritten,
            cudf::AllocatorEventRead::NotYetRecorded,
        ] {
            assert_eq!(
                newest_exact_cache_scope_event_in_window(identity, 10, 11, [(11, cache_scope_read(unstable))]),
                CacheScopeWindowEvidence::Discard
            );
        }
        assert_eq!(
            newest_exact_cache_scope_event_in_window(identity, 10, 11, [(11, Err::<cudf::AllocatorEventRead, _>(()))],),
            CacheScopeWindowEvidence::Discard
        );

        let wrong_identity = cache_scope_identity(7, 12);
        assert_eq!(
            newest_exact_cache_scope_event_in_window(
                wrong_identity,
                10,
                11,
                [(11, cache_scope_read(cudf::AllocatorEventRead::Available(cache_scope_event(11, identity))))],
            ),
            CacheScopeWindowEvidence::Discard
        );
        let mut incomplete = cache_scope_event(11, identity);
        incomplete.completeness.allocation_domain = false;
        assert_eq!(
            newest_exact_cache_scope_event_in_window(
                identity,
                10,
                11,
                [(11, cache_scope_read(cudf::AllocatorEventRead::Available(incomplete)))],
            ),
            CacheScopeWindowEvidence::Discard
        );
        assert_eq!(
            newest_exact_cache_scope_event_in_window(
                identity,
                10,
                11,
                [(12, cache_scope_read(cudf::AllocatorEventRead::Available(cache_scope_event(12, identity))))],
            ),
            CacheScopeWindowEvidence::Discard
        );
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_scope_window_never_crosses_its_bounds() {
        let identity = cache_scope_identity(7, 11);
        let stale = cache_scope_event(10, identity);
        let exact = cache_scope_event(12, identity);
        let later = cache_scope_event(14, identity);

        let event = newest_exact_cache_scope_event_in_window(
            identity,
            10,
            13,
            [
                (10, cache_scope_read(cudf::AllocatorEventRead::Available(stale))),
                (11, cache_scope_read(cudf::AllocatorEventRead::Available(cache_scope_event(11, identity)))),
                (12, cache_scope_read(cudf::AllocatorEventRead::Available(exact))),
                (14, cache_scope_read(cudf::AllocatorEventRead::Available(later))),
            ],
        );
        assert_eq!(event, CacheScopeWindowEvidence::Discard);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_scope_sequential_windows_do_not_cross() {
        let collector = AttemptCacheScopeCorrelationCollector::default();
        let identity = cache_scope_identity(7, 11);
        let first = cache_scope_event(4, identity);
        let second = cache_scope_event(9, identity);

        assert!(collector.register_active_window(identity));
        collector.finish_window(identity, CacheScopeWindowEvidence::Exact(first));
        assert!(collector.register_active_window(identity));
        collector.finish_window(identity, CacheScopeWindowEvidence::Exact(second));

        assert_eq!(collector.seal_and_take_exact_event(), Some(second));
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_scope_concurrent_registration_is_discarded() {
        let collector = AttemptCacheScopeCorrelationCollector::default();
        let identity = cache_scope_identity(7, 11);

        assert!(collector.register_active_window(identity));
        assert!(!collector.register_active_window(identity));
        collector.discard_window(identity);

        assert_eq!(collector.seal_and_take_exact_event(), None);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_scope_events_from_independent_rings_remain_ambiguous() {
        let collector = AttemptCacheScopeCorrelationCollector::default();
        let first_identity = cache_scope_identity(7, 11);
        let second_identity = cache_scope_identity(8, 13);

        assert!(collector.register_active_window(first_identity));
        collector.finish_window(first_identity, CacheScopeWindowEvidence::Exact(cache_scope_event(2, first_identity)));
        assert!(collector.register_active_window(second_identity));
        collector
            .finish_window(second_identity, CacheScopeWindowEvidence::Exact(cache_scope_event(2, second_identity)));

        assert_eq!(collector.seal_and_take_exact_event(), None);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_scope_collector_discards_active_and_double_closed_windows_at_seal() {
        let identity = cache_scope_identity(7, 11);
        let collector = AttemptCacheScopeCorrelationCollector::default();

        assert!(collector.register_active_window(identity));
        assert_eq!(collector.seal_and_take_exact_event(), None);

        let collector = AttemptCacheScopeCorrelationCollector::default();
        assert!(collector.register_active_window(identity));
        collector.finish_window(identity, CacheScopeWindowEvidence::Exact(cache_scope_event(2, identity)));
        collector.finish_window(identity, CacheScopeWindowEvidence::NoEvent);
        assert_eq!(collector.seal_and_take_exact_event(), None);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn recreated_cache_scope_cannot_receive_old_scope_event() {
        let old_identity = cache_scope_identity(7, 11);
        let recreated_identity = cache_scope_identity(7, 12);
        let old_event = cache_scope_event(12, old_identity);

        assert_eq!(
            newest_exact_cache_scope_event_in_window(
                recreated_identity,
                10,
                12,
                [
                    (
                        11,
                        cache_scope_read(cudf::AllocatorEventRead::Available(cache_scope_event(
                            11,
                            recreated_identity
                        )))
                    ),
                    (12, cache_scope_read(cudf::AllocatorEventRead::Available(old_event))),
                ],
            ),
            CacheScopeWindowEvidence::Discard
        );
    }

    #[test]
    fn cudf_adapters_preserve_only_the_complete_query_domain_identity() {
        let cudf_identity = cudf::QueryMemoryDomainIdentity {
            device: cudf::DeviceId::DEFAULT,
            domain_id: DOMAIN.domain_id,
            generation: DOMAIN.generation,
        };
        assert_eq!(AllocatorDomainIdentity::from_cudf(cudf_identity), AllocatorDomainIdentity::new(0, 91, 7));

        let event = cudf::AllocatorEvent {
            sequence: 17,
            monotonic_time_ns: None,
            device: Some(cudf::DeviceId::DEFAULT),
            allocation_domain: Some(cudf::AllocatorEventDomainIdentity {
                domain: cudf::DeviceMemoryFailureAllocationDomain::QueryMemoryDomain {
                    domain_id: DOMAIN.domain_id,
                    member_id: 3,
                },
                generation: DOMAIN.generation,
            }),
            requested_bytes: 4096,
            source: cudf::DeviceMemoryFailureSource::ReservationLimit,
            action: None,
            callback_attempt: 0,
            trim_retry_count: 0,
            pool_usage: None,
            completeness: cudf::AllocatorEventCompleteness {
                timestamp: false,
                allocation_domain: true,
                pool_usage: false,
                action: false,
            },
        };
        assert_eq!(
            AllocatorEventCorrelationInput::from_cudf_read(17, cudf::AllocatorEventRead::Available(event)),
            AllocatorEventCorrelationInput::available(17, AllocatorDomainIdentity::new(0, 91, 7))
        );

        let incomplete_event = cudf::AllocatorEvent {
            completeness: cudf::AllocatorEventCompleteness { allocation_domain: false, ..event.completeness },
            ..event
        };
        assert_eq!(
            AllocatorEventCorrelationInput::from_cudf_read(17, cudf::AllocatorEventRead::Available(incomplete_event)),
            AllocatorEventCorrelationInput {
                sequence: 17,
                availability: AllocatorEventAvailability::Available,
                domain: None,
            }
        );
    }

    #[test]
    fn exact_domain_identity_correlates_live_then_retired() {
        let mut index = index(2, 2);
        let attempt = AttemptId::mint();

        assert_eq!(index.register_live(DOMAIN, attempt, 10), AllocatorDomainRegistration::Registered);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(11, DOMAIN)),
            AllocatorEventCorrelation::Exact { attempt_id: attempt, state: AllocatorEventAttemptState::Live }
        );

        assert_eq!(index.retire(DOMAIN, attempt, 15), AllocatorDomainRetirement::Retired);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(12, DOMAIN)),
            AllocatorEventCorrelation::Exact { attempt_id: attempt, state: AllocatorEventAttemptState::Retired }
        );
    }

    #[test]
    fn duplicate_domain_identity_is_explicitly_ambiguous() {
        let mut index = index(2, 2);
        let first = AttemptId::mint();
        let second = AttemptId::mint();

        assert_eq!(index.register_live(DOMAIN, first, 10), AllocatorDomainRegistration::Registered);
        assert_eq!(index.register_live(DOMAIN, second, 10), AllocatorDomainRegistration::Ambiguous);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(11, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Ambiguous)
        );
    }

    #[test]
    fn retired_identity_reuse_invalidates_the_former_exact_window() {
        let mut index = index(1, 1);
        let first = AttemptId::mint();
        let second = AttemptId::mint();

        assert_eq!(index.register_live(DOMAIN, first, 10), AllocatorDomainRegistration::Registered);
        assert_eq!(index.retire(DOMAIN, first, 15), AllocatorDomainRetirement::Retired);
        assert_eq!(index.register_live(DOMAIN, second, 20), AllocatorDomainRegistration::Ambiguous);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(12, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Ambiguous)
        );
    }

    #[test]
    fn domain_lifetime_boundaries_reject_stale_records() {
        let mut index = index(1, 1);
        let attempt = AttemptId::mint();

        assert_eq!(index.register_live(DOMAIN, attempt, 10), AllocatorDomainRegistration::Registered);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(10, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Stale)
        );
        assert_eq!(index.retire(DOMAIN, attempt, 15), AllocatorDomainRetirement::Retired);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(16, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Stale)
        );
    }

    #[test]
    fn overwritten_and_missing_ring_reads_stay_unattributed() {
        let index = index(1, 1);

        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::overwritten(40)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Overwritten)
        );
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::missing(41)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Missing)
        );
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(42, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::MissingDomainAttempt)
        );
    }

    #[test]
    fn retired_capacity_eviction_leaves_old_records_unattributed() {
        let mut index = index(2, 1);
        let first_domain = DOMAIN;
        let second_domain = AllocatorDomainIdentity::new(3, 92, 8);
        let first = AttemptId::mint();
        let second = AttemptId::mint();

        assert_eq!(index.register_live(first_domain, first, 1), AllocatorDomainRegistration::Registered);
        assert_eq!(index.retire(first_domain, first, 2), AllocatorDomainRetirement::Retired);
        assert_eq!(index.register_live(second_domain, second, 3), AllocatorDomainRegistration::Registered);
        assert_eq!(index.retire(second_domain, second, 4), AllocatorDomainRetirement::Retired);

        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(2, first_domain)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::MissingDomainAttempt)
        );
    }

    #[test]
    fn missing_terminal_watermark_discards_the_live_mapping() {
        let mut index = index(1, 1);
        let attempt = AttemptId::mint();

        assert_eq!(index.register_live(DOMAIN, attempt, 10), AllocatorDomainRegistration::Registered);
        assert_eq!(index.discard_live(DOMAIN, attempt), AllocatorDomainDiscard::Discarded);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(11, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::MissingDomainAttempt)
        );
    }

    #[test]
    fn discarding_an_ambiguous_domain_requires_its_owner_attempt() {
        let mut index = index(2, 1);
        let first = AttemptId::mint();
        let second = AttemptId::mint();
        let unknown = AttemptId::mint();

        assert_eq!(index.register_live(DOMAIN, first, 10), AllocatorDomainRegistration::Registered);
        assert_eq!(index.register_live(DOMAIN, second, 10), AllocatorDomainRegistration::Ambiguous);
        assert_eq!(index.discard_live(DOMAIN, unknown), AllocatorDomainDiscard::AttemptMismatch);
        assert_eq!(
            index.correlate(AllocatorEventCorrelationInput::available(11, DOMAIN)),
            AllocatorEventCorrelation::Unattributed(AllocatorEventUnattributedReason::Ambiguous)
        );
    }
}
