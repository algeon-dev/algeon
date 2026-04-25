//! Grant-backed allocator domains and budget-resize observer registries.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use super::{MemoryGrant, QueryMemoryDomainPolicy, gpu};
use crate::error::{QueryError, QueryErrorCode, Result};

pub(super) fn create_grant_reservation_arena(
    runtime: &cudf::Runtime,
    memory_grant: Option<&MemoryGrant>,
) -> Result<Option<Arc<cudf::ReservationArena>>> {
    let Some(grant) = memory_grant else {
        return Ok(None);
    };
    let grant_key = grant.identity_key();
    let mut registry = grant_reservation_arena_registry().lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.runtime.grant_reservation_arena",
            "grant reservation arena registry mutex was poisoned",
        )
    })?;
    if let Some(arena) = registry.get(&grant_key).and_then(Weak::upgrade) {
        if arena.device() != runtime.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.grant_reservation_arena",
                "memory grant reservation arena device does not match execution runtime device",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("admission_request_id", grant.admission_request_id())
            .with_fact("arena_device_ordinal", arena.device().ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        return Ok(Some(arena));
    }
    registry.remove(&grant_key);

    let limit_bytes = usize::try_from(grant.budget_bytes()).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.runtime.grant_reservation_arena",
            "memory grant budget exceeds platform usize range",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
        .with_fact("budget_bytes", grant.budget_bytes())
    })?;
    let manager = runtime.device_memory_manager().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_reservation_arena",
            "failed to resolve device memory manager for grant-backed reservation arena",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
    })?;
    let arena = manager.reservation_arena(grant.grant_id(), limit_bytes).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_reservation_arena",
            "failed to create grant-backed reservation arena",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
        .with_fact("budget_bytes", grant.budget_bytes())
    })?;
    let arena = Arc::new(arena);
    let observer_arena = Arc::clone(&arena);
    let resize_observer = sequenced_budget_resize_observer("grant reservation arena", move |budget_bytes| {
        let limit_bytes = usize::try_from(budget_bytes)
            .map_err(|_error| "memory grant budget exceeded platform usize range".to_owned())?;
        observer_arena.resize(limit_bytes).map_err(|error| error.to_string())
    });
    let initial_budget = grant.notify_on_budget_change(Arc::clone(&resize_observer))?;
    registry.insert(grant_key, Arc::downgrade(&arena));
    resize_observer(initial_budget);
    Ok(Some(arena))
}

pub(super) fn create_grant_memory_domain(
    runtime: &cudf::Runtime,
    memory_grant: Option<&MemoryGrant>,
    policy: Option<QueryMemoryDomainPolicy>,
    device_capacity_bytes: u64,
) -> Result<Option<Arc<cudf::QueryMemoryDomain>>> {
    let (Some(grant), Some(policy)) = (memory_grant, policy) else {
        return Ok(None);
    };
    let grant_key = grant.identity_key();
    let creation_lock = grant_memory_domain_creation_lock(grant_key)?;
    let _creation_guard = creation_lock.lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.runtime.grant_memory_domain",
            "grant query memory domain creation mutex was poisoned",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
    })?;
    let existing_domain = {
        let mut registry = grant_memory_domain_registry().lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.grant_memory_domain",
                "grant query memory domain registry mutex was poisoned",
            )
        })?;
        registry.retain(|key, domain| *key == grant_key || domain.strong_count() != 0);
        let domain = registry.get(&grant_key).and_then(Weak::upgrade);
        if domain.is_none() {
            registry.remove(&grant_key);
        }
        domain
    };
    if let Some(domain) = existing_domain {
        if domain.device() != runtime.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.grant_memory_domain",
                "memory grant query domain device does not match execution runtime device",
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("admission_request_id", grant.admission_request_id())
            .with_fact("domain_device_ordinal", domain.device().ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        return Ok(Some(domain));
    }

    let (grant_bytes_u64, ceiling_bytes_u64) = grant.install_hard_cap_bytes_for_current_budget(|grant_bytes| {
        policy.ceiling_bytes(grant_bytes, device_capacity_bytes)
    })?;

    let grant_bytes = usize::try_from(grant_bytes_u64).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.runtime.grant_memory_domain",
            "memory grant budget exceeds platform usize range",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
        .with_fact("budget_bytes", grant_bytes_u64)
    })?;
    let ceiling_bytes = usize::try_from(ceiling_bytes_u64).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.runtime.grant_memory_domain",
            "memory grant query domain ceiling exceeds platform usize range",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
        .with_fact("ceiling_bytes", ceiling_bytes_u64)
    })?;
    let manager = runtime.device_memory_manager().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.runtime.grant_memory_domain",
            "failed to resolve device memory manager for grant query memory domain",
            error,
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("admission_request_id", grant.admission_request_id())
    })?;
    let domain = manager
        .query_memory_domain(
            grant.grant_id(),
            cudf::QueryMemoryDomainOptions::new(grant_bytes).with_ceiling_bytes(ceiling_bytes),
        )
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.grant_memory_domain",
                "failed to create grant query memory domain",
                error,
            )
            .with_fact("grant_id", grant.grant_id())
            .with_fact("admission_request_id", grant.admission_request_id())
            .with_fact("budget_bytes", grant_bytes_u64)
            .with_fact("ceiling_bytes", ceiling_bytes_u64)
        })?;
    let domain = Arc::new(domain);
    let observer_domain = Arc::clone(&domain);
    let resize_observer = sequenced_budget_resize_observer("grant query memory domain", move |budget_bytes| {
        let grant_bytes = usize::try_from(budget_bytes)
            .map_err(|_error| "memory grant budget exceeded platform usize range".to_owned())?;
        observer_domain.resize_grant(grant_bytes).map_err(|error| error.to_string())
    });
    let initial_budget = grant.notify_on_budget_change(Arc::clone(&resize_observer))?;
    resize_observer(initial_budget);
    grant_memory_domain_registry()
        .lock()
        .map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.grant_memory_domain",
                "grant query memory domain registry mutex was poisoned",
            )
        })?
        .insert(grant_key, Arc::downgrade(&domain));
    Ok(Some(domain))
}

fn sequenced_budget_resize_observer(
    label: &'static str,
    apply: impl Fn(u64) -> std::result::Result<(), String> + Send + Sync + 'static,
) -> gpu::BudgetChangeObserver {
    let observer_sequence = Arc::new(Mutex::new(0_u64));
    Arc::new(move |change| {
        let mut applied_sequence = match observer_sequence.lock() {
            Ok(applied_sequence) => applied_sequence,
            Err(_error) => {
                eprintln!(
                    "[nexus-query-engine] warning: failed to apply memory grant {label} resize after sequence mutex poison"
                );
                return;
            },
        };
        if change.sequence < *applied_sequence {
            return;
        }
        if let Err(error) = apply(change.budget_bytes) {
            eprintln!("[nexus-query-engine] warning: failed to resize memory grant {label}: {error}");
        }
        *applied_sequence = change.sequence;
    })
}

fn grant_reservation_arena_registry() -> &'static Mutex<HashMap<usize, Weak<cudf::ReservationArena>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<cudf::ReservationArena>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn grant_memory_domain_registry() -> &'static Mutex<HashMap<usize, Weak<cudf::QueryMemoryDomain>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<cudf::QueryMemoryDomain>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Debug)]
struct GrantMemoryDomainCreationLock {
    grant_key: usize,
    lock: Arc<Mutex<()>>,
}

impl GrantMemoryDomainCreationLock {
    fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, ()>> {
        self.lock.lock()
    }
}

impl Drop for GrantMemoryDomainCreationLock {
    fn drop(&mut self) {
        let Ok(mut locks) = grant_memory_domain_creation_locks().lock() else {
            eprintln!("[nexus-query-engine] warning: failed to clean poisoned grant-domain creation-lock registry");
            return;
        };
        let Some(registered) = locks.get(&self.grant_key).and_then(Weak::upgrade) else {
            locks.remove(&self.grant_key);
            return;
        };
        if Arc::ptr_eq(&registered, &self.lock) && Arc::strong_count(&self.lock) == 2 {
            locks.remove(&self.grant_key);
        }
    }
}

pub(super) fn grant_memory_domain_creation_locks() -> &'static Mutex<HashMap<usize, Weak<Mutex<()>>>> {
    static CREATION_LOCKS: OnceLock<Mutex<HashMap<usize, Weak<Mutex<()>>>>> = OnceLock::new();
    CREATION_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn grant_memory_domain_creation_lock(grant_key: usize) -> Result<GrantMemoryDomainCreationLock> {
    let mut locks = grant_memory_domain_creation_locks().lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.runtime.grant_memory_domain",
            "grant query memory domain creation-lock registry mutex was poisoned",
        )
    })?;
    if let Some(lock) = locks.get(&grant_key).and_then(Weak::upgrade) {
        return Ok(GrantMemoryDomainCreationLock { grant_key, lock });
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(grant_key, Arc::downgrade(&lock));
    Ok(GrantMemoryDomainCreationLock { grant_key, lock })
}
