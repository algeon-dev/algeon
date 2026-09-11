//! Process-wide ownership of the live Algeon GPU backend.

use std::{
    collections::BTreeSet,
    marker::PhantomData,
    sync::{Arc, Mutex, OnceLock},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

const CLAIM_OPERATION: &str = "query_engine.gpu.owner.claim";

/// Unforgeable proof that the current construction transaction owns the live
/// backend lease.
#[derive(Debug)]
#[doc(hidden)]
pub struct DeviceOwnerToken<'lease> {
    generation: u64,
    _lease: PhantomData<&'lease LiveBackendLease>,
}

impl DeviceOwnerToken<'_> {
    #[doc(hidden)]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// RAII ownership of the one live Algeon GPU backend allowed in this process.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct LiveBackendLease {
    inner: Arc<LiveBackendLeaseInner>,
}

#[derive(Debug)]
struct LiveBackendLeaseInner {
    registry: Arc<LiveBackendRegistryInner>,
    generation: u64,
    #[cfg(test)]
    device_ordinals: Arc<[usize]>,
}

impl LiveBackendLease {
    #[doc(hidden)]
    pub fn token(&self) -> DeviceOwnerToken<'_> {
        DeviceOwnerToken { generation: self.inner.generation, _lease: PhantomData }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn generation(&self) -> u64 {
        self.inner.generation
    }

    #[cfg(test)]
    #[doc(hidden)]
    pub fn device_ordinals(&self) -> &[usize] {
        &self.inner.device_ordinals
    }
}

impl Drop for LiveBackendLeaseInner {
    fn drop(&mut self) {
        let Ok(mut state) = self.registry.state.lock() else {
            // A poisoned registry remains fail-closed rather than permitting a
            // second independent resource ledger.
            return;
        };
        if state.active.as_ref().is_some_and(|owner| owner.generation == self.generation) {
            state.active = None;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveBackendOwner {
    generation: u64,
    device_ordinals: Arc<[usize]>,
}

#[derive(Debug, Default)]
struct LiveBackendRegistryState {
    next_generation: u64,
    active: Option<ActiveBackendOwner>,
}

#[derive(Debug, Default)]
struct LiveBackendRegistryInner {
    state: Mutex<LiveBackendRegistryState>,
}

#[derive(Debug, Clone, Default)]
struct LiveBackendRegistry {
    inner: Arc<LiveBackendRegistryInner>,
}

impl LiveBackendRegistry {
    fn claim(&self, requested_ordinals: &[usize]) -> Result<LiveBackendLease> {
        let requested_ordinals = normalize_device_ordinals(requested_ordinals)?;
        let mut state = self.inner.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                CLAIM_OPERATION,
                "live GPU backend ownership registry mutex was poisoned",
            )
            .with_fact("reason", "owner_registry_poisoned")
        })?;

        if let Some(existing) = &state.active {
            let overlap = exact_overlap(&requested_ordinals, &existing.device_ordinals);
            return Err(QueryError::new(
                QueryErrorCode::ServiceOverloaded,
                CLAIM_OPERATION,
                "an Algeon GPU backend is already live in this process",
            )
            .with_fact("reason", "backend_already_live")
            .with_fact("owner_generation", existing.generation)
            .with_fact("requested_device_ordinals", format_ordinals(&requested_ordinals))
            .with_fact("existing_device_ordinals", format_ordinals(&existing.device_ordinals))
            .with_fact("overlap_device_ordinals", format_ordinals(&overlap)));
        }

        state.next_generation = state.next_generation.checked_add(1).ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, CLAIM_OPERATION, "live GPU backend owner generation overflowed")
                .with_fact("reason", "owner_generation_overflow")
        })?;
        let generation = state.next_generation;
        let device_ordinals: Arc<[usize]> = requested_ordinals.into();
        state.active = Some(ActiveBackendOwner { generation, device_ordinals: Arc::clone(&device_ordinals) });

        Ok(LiveBackendLease {
            inner: Arc::new(LiveBackendLeaseInner {
                registry: Arc::clone(&self.inner),
                generation,
                #[cfg(test)]
                device_ordinals,
            }),
        })
    }
}

/// Claims the one live backend slot after configuration has been validated.
#[doc(hidden)]
pub fn claim_live_backend(device_ordinals: &[usize]) -> Result<LiveBackendLease> {
    static REGISTRY: OnceLock<LiveBackendRegistry> = OnceLock::new();
    REGISTRY.get_or_init(LiveBackendRegistry::default).claim(device_ordinals)
}

#[doc(hidden)]
pub fn normalize_device_ordinals(device_ordinals: &[usize]) -> Result<Vec<usize>> {
    if device_ordinals.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            CLAIM_OPERATION,
            "GPU backend requires at least one explicit CUDA device ordinal",
        )
        .with_fact("field", "device_ordinals")
        .with_fact("reason", "empty"));
    }

    let mut normalized = device_ordinals.to_vec();
    normalized.sort_unstable();
    if let Some(duplicate) = normalized.windows(2).find_map(|pair| (pair[0] == pair[1]).then_some(pair[0])) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            CLAIM_OPERATION,
            "GPU backend device ordinals must be unique",
        )
        .with_fact("field", "device_ordinals")
        .with_fact("reason", "duplicate")
        .with_fact("device_ordinal", duplicate));
    }
    Ok(normalized)
}

fn exact_overlap(left: &[usize], right: &[usize]) -> Vec<usize> {
    let right = right.iter().copied().collect::<BTreeSet<_>>();
    left.iter().copied().filter(|ordinal| right.contains(ordinal)).collect()
}

fn format_ordinals(ordinals: &[usize]) -> String {
    ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn live_backend_claims_are_atomic_exclusive_and_generation_ordered() {
        let registry = LiveBackendRegistry::default();

        assert_eq!(registry.claim(&[]).unwrap_err().fact_value("reason"), Some("empty"));
        assert_eq!(registry.claim(&[3, 3]).unwrap_err().fact_value("reason"), Some("duplicate"));

        let first = registry.claim(&[2, 0]).unwrap();
        assert_eq!(first.device_ordinals(), &[0, 2]);
        assert_eq!(first.token().generation(), first.generation());

        let partial = registry.claim(&[1, 2]).unwrap_err();
        assert_eq!(partial.fact_value("reason"), Some("backend_already_live"));
        assert_eq!(partial.fact_value("requested_device_ordinals"), Some("1,2"));
        assert_eq!(partial.fact_value("existing_device_ordinals"), Some("0,2"));
        assert_eq!(partial.fact_value("overlap_device_ordinals"), Some("2"));

        let disjoint = registry.claim(&[7]).unwrap_err();
        assert_eq!(disjoint.fact_value("reason"), Some("backend_already_live"));
        assert_eq!(disjoint.fact_value("overlap_device_ordinals"), Some(""));

        let first_generation = first.generation();
        let retained = first.clone();
        drop(first);
        assert_eq!(registry.claim(&[9]).unwrap_err().fact_value("reason"), Some("backend_already_live"));
        drop(retained);
        let second = registry.claim(&[9]).unwrap();
        assert!(second.generation() > first_generation);
        drop(second);

        let contenders = 8;
        let start = Arc::new(Barrier::new(contenders));
        let hold = Arc::new(Barrier::new(contenders));
        let mut threads = Vec::new();
        for ordinal in 0..contenders {
            let registry = registry.clone();
            let start = Arc::clone(&start);
            let hold = Arc::clone(&hold);
            threads.push(std::thread::spawn(move || {
                start.wait();
                let claim = registry.claim(&[ordinal]);
                hold.wait();
                claim
            }));
        }
        let results = threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(
            results
                .iter()
                .filter_map(|result| result.as_ref().err())
                .all(|error| error.fact_value("reason") == Some("backend_already_live"))
        );
    }
}
