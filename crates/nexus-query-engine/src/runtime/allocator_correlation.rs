//! Bounded, post-callback attribution for cuDF allocator-ring records.
//!
//! The allocator callback cannot inspect engine state.  Instead, the engine
//! records the exact domain identity that an admitted envelope owns and joins
//! ring records only after the callback has completed. This module deliberately
//! retains no domain handles; terminal finalization may request one bounded,
//! exactly joined record without creating a second event store.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    sync::Mutex,
};

use crate::observability::AttemptId;

const DEFAULT_MAX_LIVE_ENTRIES: usize = 1024;
const DEFAULT_MAX_RETIRED_ENTRIES: usize = 1024;

/// The complete allocator identity supplied by cuDF for one domain instance.
///
/// A domain id is not sufficient because allocator-domain ids may be reused.
/// The generation is therefore part of every lookup key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AllocatorDomainIdentity {
    device_ordinal: usize,
    domain_id: u64,
    generation: u64,
}

impl AllocatorDomainIdentity {
    pub(crate) const fn new(device_ordinal: usize, domain_id: u64, generation: u64) -> Self {
        Self { device_ordinal, domain_id, generation }
    }

    /// Adapts the complete identity that cuDF returns for one native domain
    /// instance. Callers must not substitute a grant id for its generation.
    pub(crate) fn from_cudf(identity: cudf::QueryMemoryDomainIdentity) -> Self {
        Self::new(identity.device.ordinal(), identity.domain_id, identity.generation)
    }
}

/// The availability result of one ring position after it is read by the engine.
///
/// `Missing` and `Overwritten` are terminal attribution states.  They must not
/// fall back to a later device sample or to another attempt-domain identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatorEventAvailability {
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
pub(crate) struct AllocatorEventCorrelationInput {
    sequence: u64,
    availability: AllocatorEventAvailability,
    domain: Option<AllocatorDomainIdentity>,
}

impl AllocatorEventCorrelationInput {
    pub(crate) const fn available(sequence: u64, domain: AllocatorDomainIdentity) -> Self {
        Self { sequence, availability: AllocatorEventAvailability::Available, domain: Some(domain) }
    }

    pub(crate) const fn missing(sequence: u64) -> Self {
        Self { sequence, availability: AllocatorEventAvailability::Missing, domain: None }
    }

    pub(crate) const fn overwritten(sequence: u64) -> Self {
        Self { sequence, availability: AllocatorEventAvailability::Overwritten, domain: None }
    }

    pub(crate) const fn sequence(self) -> u64 {
        self.sequence
    }

    pub(crate) const fn availability(self) -> AllocatorEventAvailability {
        self.availability
    }

    pub(crate) const fn domain(self) -> Option<AllocatorDomainIdentity> {
        self.domain
    }

    /// Adapts one exact cuDF ring read without fabricating an identity for
    /// incomplete, pending, or non-query-domain records.
    pub(crate) fn from_cudf_read(sequence: u64, read: cudf::AllocatorEventRead) -> Self {
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
pub(crate) enum AllocatorEventAttemptState {
    Live,
    Retired,
}

/// Why an allocator record cannot truthfully name an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatorEventUnattributedReason {
    Missing,
    Overwritten,
    MissingDomainIdentity,
    MissingDomainAttempt,
    Stale,
    Ambiguous,
}

/// Result of joining one allocator-ring read to the bounded domain index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatorEventCorrelation {
    Exact { attempt_id: AttemptId, state: AllocatorEventAttemptState },
    Unattributed(AllocatorEventUnattributedReason),
}

/// Outcome of registering a newly admitted attempt domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatorDomainRegistration {
    Registered,
    AlreadyRegistered,
    Ambiguous,
    CapacityExhausted,
}

/// Outcome of retiring one admitted attempt domain after cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatorDomainRetirement {
    Retired,
    AmbiguousStillLive,
    AmbiguousRetired,
    Missing,
    AttemptMismatch,
    InvalidSequenceBoundary,
}

/// Outcome of dropping a live mapping without a trustworthy terminal watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocatorDomainDiscard {
    Discarded,
    AmbiguousStillLive,
    Missing,
    AttemptMismatch,
}

/// Identity retained by an admitted envelope until cleanup establishes the
/// final ring watermark. It contains no cuDF handle, grant, or mutable attempt
/// state, so retaining it cannot extend an allocation domain's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocatorDomainAttempt {
    domain: AllocatorDomainIdentity,
    attempt_id: AttemptId,
}

/// Bounded allocator-ring evidence observed while one completed domain moves
/// to its retired correlation interval. A device record that cannot be joined
/// exactly remains explicitly unattributed; it is never a substitute attempt
/// event.
#[derive(Debug, Default)]
pub(crate) struct AllocatorEventEvidenceRead {
    pub(crate) exact: Option<cudf::AllocatorEvent>,
    pub(crate) unattributed: Option<cudf::AllocatorEvent>,
}

/// Registers one just-created native domain against an engine attempt.
///
/// Failure to observe the diagnostic identity or watermark leaves the event
/// unattributed; it must never change an already-approved admission. A full
/// bounded index likewise declines the optional registration rather than
/// evicting a live attempt mapping.
pub(crate) fn register_admitted_domain(
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
pub(crate) fn retire_admitted_domain(
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
pub(crate) fn discard_admitted_domain(index: &Mutex<AllocatorEventCorrelationIndex>, binding: AllocatorDomainAttempt) {
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
pub(crate) struct AllocatorEventCorrelationIndex {
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
    pub(crate) fn new(max_live_entries: NonZeroUsize, max_retired_entries: NonZeroUsize) -> Self {
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
    pub(crate) fn register_live(
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
    pub(crate) fn retire(
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
    pub(crate) fn correlate(&self, event: AllocatorEventCorrelationInput) -> AllocatorEventCorrelation {
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
    pub(crate) fn discard_live(
        &mut self,
        domain: AllocatorDomainIdentity,
        attempt_id: AttemptId,
    ) -> AllocatorDomainDiscard {
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
