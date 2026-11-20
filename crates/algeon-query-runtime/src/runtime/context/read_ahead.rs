//! Source reads an attempt started ahead of their consumer.
//!
//! The attempt keeps a weak reference to each one so that a query-cap retry
//! anywhere in the attempt can release them first: a retry then runs with the
//! memory it would have had with no read ahead. Design owner:
//! `website/docs/design/data-path.mdx`.

use std::{
    fmt,
    sync::{Mutex, PoisonError, Weak},
};

/// One optional read an attempt started ahead of its consumer.
pub trait ReadAheadRelease: Send + Sync {
    /// Joins and discards the read if it is still outstanding, and keeps its
    /// owner from starting another. Repeated calls do nothing.
    fn release(&self);
}

/// The attempt's registered reads. Entries are weak, so an owner that drops its
/// read ahead leaves only a dead entry behind, pruned on the next registration.
#[derive(Default)]
pub(in crate::runtime) struct ReadAheadRegistry {
    entries: Mutex<Vec<Weak<dyn ReadAheadRelease>>>,
}

impl fmt::Debug for ReadAheadRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner).len();
        formatter.debug_struct("ReadAheadRegistry").field("entries", &entries).finish()
    }
}

impl ReadAheadRegistry {
    pub(super) fn register(&self, entry: Weak<dyn ReadAheadRelease>) {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.retain(|entry| entry.strong_count() > 0);
        entries.push(entry);
    }

    /// Releases every live entry. The entries are taken out of the lock first,
    /// because a release joins a worker and records observations.
    pub(super) fn release_all(&self) {
        let entries = std::mem::take(&mut *self.entries.lock().unwrap_or_else(PoisonError::into_inner));
        for entry in entries {
            if let Some(entry) = entry.upgrade() {
                entry.release();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    struct CountingRelease(AtomicUsize);

    impl ReadAheadRelease for CountingRelease {
        fn release(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn release_reaches_live_entries_once_and_skips_dropped_ones() {
        let registry = ReadAheadRegistry::default();
        let live = Arc::new(CountingRelease(AtomicUsize::new(0)));
        let dropped = Arc::new(CountingRelease(AtomicUsize::new(0)));
        registry.register(Arc::downgrade(&live) as Weak<dyn ReadAheadRelease>);
        registry.register(Arc::downgrade(&dropped) as Weak<dyn ReadAheadRelease>);
        let dropped_weak = Arc::downgrade(&dropped);
        drop(dropped);

        registry.release_all();
        registry.release_all();

        assert_eq!(live.0.load(Ordering::Relaxed), 1, "a released entry leaves the registry");
        assert!(dropped_weak.upgrade().is_none());
    }
}
