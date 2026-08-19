//! Exact-binding extension lifecycles for cross-crate native owners.
//!
//! One stable slot is retained for each extension [`TypeId`] on an execution
//! binding. The slot does not retain the extension value: counted leases own
//! one generation, while the slot keeps only a weak generation witness. This
//! lets an extension retain a [`Runtime`](super::Runtime) without forming a
//! cycle through the runtime's [`ExecutionDomain`].

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{HashMap, HashSet},
    error::Error as StdError,
    fmt,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread::ThreadId,
};

use super::execution_domain::ExecutionDomain;
use crate::core::error::{Error, ErrorCode};

/// Maximum number of distinct extension types retained by one exact binding.
///
/// Slots are deliberately stable across value generations, so a hard bound is
/// required to prevent unbounded `TypeId` churn in long-lived services.
pub const MAX_BINDING_EXTENSION_SLOTS_PER_BINDING: usize = 64;

static NEXT_SLOT_ID: AtomicU64 = AtomicU64::new(1);

type ErasedBindingExtensionSlot = Arc<dyn Any + Send + Sync>;
type BindingExtensionSlotMap = HashMap<TypeId, ErasedBindingExtensionSlot>;

thread_local! {
    static ENTERED_BINDING_EXTENSION_SLOTS: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());
    static BINDING_EXTENSION_USER_PHASES: RefCell<Vec<BindingExtensionUserPhase>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
enum BindingExtensionUserPhase {
    Factory,
    Destructor,
}

impl BindingExtensionUserPhase {
    fn wait_reason(self) -> &'static str {
        match self {
            Self::Factory => "binding_extension_factory_cross_type_wait_forbidden",
            Self::Destructor => "binding_extension_destructor_cross_type_wait_forbidden",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Factory => "factory",
            Self::Destructor => "destructor",
        }
    }

    fn current() -> Option<Self> {
        BINDING_EXTENSION_USER_PHASES.with(|phases| phases.borrow().last().copied())
    }
}

struct BindingExtensionUserPhaseGuard;

impl BindingExtensionUserPhaseGuard {
    fn enter(phase: BindingExtensionUserPhase) -> Self {
        BINDING_EXTENSION_USER_PHASES.with(|phases| phases.borrow_mut().push(phase));
        Self
    }
}

impl Drop for BindingExtensionUserPhaseGuard {
    fn drop(&mut self) {
        BINDING_EXTENSION_USER_PHASES.with(|phases| {
            let removed = phases.borrow_mut().pop();
            debug_assert!(removed.is_some(), "binding extension user phase stack must remain balanced");
        });
    }
}

/// Failure from binding-extension lookup or initialization.
///
/// Factory failures remain typed so callers can preserve their own structured
/// errors. Lifecycle failures are `cudf` errors with stable error codes and
/// diagnostic reasons.
#[derive(Debug)]
pub enum BindingExtensionInitError<E> {
    /// The stable slot rejected the operation or was unable to proceed safely.
    Lifecycle(Error),
    /// The caller-supplied factory failed.
    Factory(E),
}

impl<E: fmt::Display> fmt::Display for BindingExtensionInitError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lifecycle(error) => write!(f, "binding extension lifecycle failed: {error}"),
            Self::Factory(error) => write!(f, "binding extension factory failed: {error}"),
        }
    }
}

impl<E> StdError for BindingExtensionInitError<E>
where E: StdError + 'static
{
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Lifecycle(error) => Some(error),
            Self::Factory(error) => Some(error),
        }
    }
}

/// Stable, type-indexed lifecycle slots scoped to one exact stream/MR binding.
pub(crate) struct BindingExtensions {
    slots: Mutex<BindingExtensionSlotMap>,
    slot_limit: usize,
}

impl Default for BindingExtensions {
    fn default() -> Self {
        Self { slots: Mutex::new(HashMap::new()), slot_limit: MAX_BINDING_EXTENSION_SLOTS_PER_BINDING }
    }
}

impl BindingExtensions {
    fn lifecycle_error(operation: &'static str, message: &'static str, reason: &'static str) -> Error {
        Error::new_with(ErrorCode::InvariantViolation, operation, message).with_detail("reason", reason)
    }

    fn lock_slots(&self, operation: &'static str) -> Result<MutexGuard<'_, BindingExtensionSlotMap>, Error> {
        self.slots.lock().map_err(|_| {
            Self::lifecycle_error(
                operation,
                "binding extension type map is poisoned",
                "binding_extension_type_map_poisoned",
            )
        })
    }

    fn slot<T>(&self, operation: &'static str) -> Result<Arc<BindingExtensionSlot<T>>, Error>
    where T: Any + Send + Sync {
        let type_id = TypeId::of::<T>();
        if let Some(existing) = self.lock_slots(operation)?.get(&type_id).cloned() {
            return existing.downcast::<BindingExtensionSlot<T>>().map_err(|_| {
                Self::lifecycle_error(
                    operation,
                    "binding extension slot type did not match its TypeId",
                    "binding_extension_slot_type_mismatch",
                )
            });
        }

        // Candidate construction and a losing candidate's destruction stay
        // outside the type-map lock. The candidate contains no user value.
        let candidate = Arc::new(BindingExtensionSlot::<T>::new(operation)?);
        let mut losing_candidate = None;
        let winner = {
            let mut slots = self.lock_slots(operation)?;
            if let Some(existing) = slots.get(&type_id).cloned() {
                losing_candidate = Some(candidate);
                existing.downcast::<BindingExtensionSlot<T>>().map_err(|_| {
                    Self::lifecycle_error(
                        operation,
                        "binding extension slot type did not match its TypeId",
                        "binding_extension_slot_type_mismatch",
                    )
                })?
            } else {
                if slots.len() >= self.slot_limit {
                    return Err(Self::lifecycle_error(
                        operation,
                        "binding extension slot cardinality limit was reached",
                        "binding_extension_slot_limit_reached",
                    )
                    .with_detail("slot_limit", self.slot_limit.to_string())
                    .with_detail("slot_count", slots.len().to_string()));
                }
                slots.insert(type_id, Arc::clone(&candidate) as Arc<dyn Any + Send + Sync>);
                candidate
            }
        };
        drop(losing_candidate);
        Ok(winner)
    }

    pub(crate) fn get_or_try_init<T, E>(
        &self,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
        factory: impl FnOnce() -> Result<T, E>,
    ) -> Result<BindingExtensionLease<T>, BindingExtensionInitError<E>>
    where
        T: Any + Send + Sync,
    {
        let slot = self.slot(operation).map_err(BindingExtensionInitError::Lifecycle)?;
        slot.acquire_or_try_init(domain, operation, factory)
    }

    #[cfg(test)]
    fn with_slot_limit(slot_limit: usize) -> Self {
        Self { slots: Mutex::new(HashMap::new()), slot_limit }
    }
}

enum Lifecycle<T> {
    Vacant,
    Initializing { owner: ThreadId },
    Live { generation_id: u64, generation: Weak<Generation<T>> },
    Destroying { generation_id: u64, owner: ThreadId },
    Poisoned { reason: &'static str },
}

struct Generation<T> {
    id: u64,
    external_leases: AtomicUsize,
    value: T,
}

struct BindingExtensionSlot<T> {
    id: u64,
    next_generation_id: AtomicU64,
    lifecycle: Mutex<Lifecycle<T>>,
    lifecycle_changed: Condvar,
    execution: Mutex<()>,
}

// MSRV 1.94 predates stable Atomic::try_update (1.95); fetch_update is the MSRV-compatible spelling.
#[allow(deprecated)]
impl<T> BindingExtensionSlot<T>
where T: Any + Send + Sync
{
    fn new(operation: &'static str) -> Result<Self, Error> {
        let id = NEXT_SLOT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| current.checked_add(1))
            .map_err(|_| {
                BindingExtensions::lifecycle_error(
                    operation,
                    "binding extension slot identity space was exhausted",
                    "binding_extension_slot_identity_exhausted",
                )
            })?;
        Ok(Self {
            id,
            next_generation_id: AtomicU64::new(1),
            lifecycle: Mutex::new(Lifecycle::Vacant),
            lifecycle_changed: Condvar::new(),
            execution: Mutex::new(()),
        })
    }

    fn lock_lifecycle(&self, operation: &'static str) -> Result<MutexGuard<'_, Lifecycle<T>>, Error> {
        self.lifecycle.lock().map_err(|_| {
            BindingExtensions::lifecycle_error(
                operation,
                "binding extension lifecycle lock is poisoned",
                "binding_extension_lifecycle_lock_poisoned",
            )
        })
    }

    fn next_generation_id(&self, operation: &'static str) -> Result<u64, Error> {
        self.next_generation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| current.checked_add(1))
            .map_err(|_| {
                BindingExtensions::lifecycle_error(
                    operation,
                    "binding extension generation identity space was exhausted",
                    "binding_extension_generation_identity_exhausted",
                )
            })
    }

    fn wait_for_lifecycle_change<'a>(
        &self,
        lifecycle: MutexGuard<'a, Lifecycle<T>>,
        operation: &'static str,
    ) -> Result<MutexGuard<'a, Lifecycle<T>>, Error> {
        self.lifecycle_changed.wait(lifecycle).map_err(|_| {
            BindingExtensions::lifecycle_error(
                operation,
                "binding extension lifecycle lock was poisoned while waiting",
                "binding_extension_lifecycle_wait_poisoned",
            )
        })
    }

    fn cross_type_wait_error(
        operation: &'static str,
        phase: BindingExtensionUserPhase,
        waiting_on: &'static str,
    ) -> Error {
        BindingExtensions::lifecycle_error(
            operation,
            "binding extension user code cannot wait on another extension lifecycle",
            phase.wait_reason(),
        )
        .with_detail("user_phase", phase.label())
        .with_detail("waiting_on", waiting_on)
    }

    fn acquire_or_try_init<E>(
        self: Arc<Self>,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
        factory: impl FnOnce() -> Result<T, E>,
    ) -> Result<BindingExtensionLease<T>, BindingExtensionInitError<E>> {
        let current_thread = std::thread::current().id();
        let mut factory = Some(factory);

        loop {
            let mut lifecycle = self.lock_lifecycle(operation).map_err(BindingExtensionInitError::Lifecycle)?;
            match &*lifecycle {
                Lifecycle::Vacant => {
                    *lifecycle = Lifecycle::Initializing { owner: current_thread };
                    drop(lifecycle);

                    let mut attempt = InitializationAttempt::new(Arc::clone(&self), current_thread);
                    let factory_result = {
                        let _phase = BindingExtensionUserPhaseGuard::enter(BindingExtensionUserPhase::Factory);
                        factory.take().expect("factory is consumed only after claiming initialization")()
                    };
                    let value = match factory_result {
                        Ok(value) => value,
                        Err(error) => {
                            attempt.factory_failed();
                            return Err(BindingExtensionInitError::Factory(error));
                        },
                    };
                    let generation_id = match self.next_generation_id(operation) {
                        Ok(generation_id) => generation_id,
                        Err(error) => {
                            attempt.poison("binding_extension_generation_identity_exhausted");
                            drop(value);
                            return Err(BindingExtensionInitError::Lifecycle(error));
                        },
                    };
                    let generation =
                        Arc::new(Generation { id: generation_id, external_leases: AtomicUsize::new(1), value });

                    let mut lifecycle = self.lock_lifecycle(operation).map_err(BindingExtensionInitError::Lifecycle)?;
                    match &*lifecycle {
                        Lifecycle::Initializing { owner } if *owner == current_thread => {},
                        _ => {
                            drop(lifecycle);
                            attempt.poison("binding_extension_initialization_state_changed");
                            drop(generation);
                            return Err(BindingExtensionInitError::Lifecycle(BindingExtensions::lifecycle_error(
                                operation,
                                "binding extension initialization state changed unexpectedly",
                                "binding_extension_initialization_state_changed",
                            )));
                        },
                    }
                    *lifecycle = Lifecycle::Live { generation_id, generation: Arc::downgrade(&generation) };
                    attempt.complete();
                    drop(lifecycle);
                    self.lifecycle_changed.notify_all();
                    return Ok(BindingExtensionLease { generation: Some(generation), slot: self, domain });
                },
                Lifecycle::Initializing { owner } => {
                    if *owner == current_thread {
                        return Err(BindingExtensionInitError::Lifecycle(BindingExtensions::lifecycle_error(
                            operation,
                            "binding extension initialization is already active on this thread",
                            "nested_binding_extension_initialization",
                        )));
                    }
                    if let Some(phase) = BindingExtensionUserPhase::current() {
                        return Err(BindingExtensionInitError::Lifecycle(Self::cross_type_wait_error(
                            operation,
                            phase,
                            "initializing",
                        )));
                    }
                    drop(
                        self.wait_for_lifecycle_change(lifecycle, operation)
                            .map_err(BindingExtensionInitError::Lifecycle)?,
                    );
                },
                Lifecycle::Live { generation_id, generation } => {
                    let Some(generation) = generation.upgrade() else {
                        *lifecycle = Lifecycle::Poisoned { reason: "binding_extension_live_generation_missing" };
                        drop(lifecycle);
                        self.lifecycle_changed.notify_all();
                        return Err(BindingExtensionInitError::Lifecycle(BindingExtensions::lifecycle_error(
                            operation,
                            "binding extension live generation was no longer retained",
                            "binding_extension_live_generation_missing",
                        )));
                    };
                    if generation.id != *generation_id {
                        *lifecycle = Lifecycle::Poisoned { reason: "binding_extension_generation_identity_mismatch" };
                        drop(lifecycle);
                        self.lifecycle_changed.notify_all();
                        return Err(BindingExtensionInitError::Lifecycle(BindingExtensions::lifecycle_error(
                            operation,
                            "binding extension generation identity did not match its slot",
                            "binding_extension_generation_identity_mismatch",
                        )));
                    }
                    generation.external_leases.fetch_add(1, Ordering::Relaxed);
                    drop(lifecycle);
                    return Ok(BindingExtensionLease { generation: Some(generation), slot: self, domain });
                },
                Lifecycle::Destroying { owner, .. } => {
                    if *owner == current_thread {
                        return Err(BindingExtensionInitError::Lifecycle(BindingExtensions::lifecycle_error(
                            operation,
                            "binding extension destruction is already active on this thread",
                            "nested_binding_extension_destruction",
                        )));
                    }
                    if let Some(phase) = BindingExtensionUserPhase::current() {
                        return Err(BindingExtensionInitError::Lifecycle(Self::cross_type_wait_error(
                            operation,
                            phase,
                            "destroying",
                        )));
                    }
                    drop(
                        self.wait_for_lifecycle_change(lifecycle, operation)
                            .map_err(BindingExtensionInitError::Lifecycle)?,
                    );
                },
                Lifecycle::Poisoned { reason } => {
                    return Err(BindingExtensionInitError::Lifecycle(BindingExtensions::lifecycle_error(
                        operation,
                        "binding extension lifecycle is poisoned",
                        reason,
                    )));
                },
            }
        }
    }

    fn poison_after_unwind(&self, owner: ThreadId, reason: &'static str) {
        let mut lifecycle = self.lifecycle.lock().expect("binding extension lifecycle mutex poisoned");
        if matches!(&*lifecycle, Lifecycle::Initializing { owner: active } if *active == owner) {
            *lifecycle = Lifecycle::Poisoned { reason };
        }
        drop(lifecycle);
        self.lifecycle_changed.notify_all();
    }

    fn release(self: &Arc<Self>, generation: Arc<Generation<T>>) {
        let current_thread = std::thread::current().id();
        let mut lifecycle = self.lifecycle.lock().expect("binding extension lifecycle mutex poisoned");
        let prior_leases =
            match generation
                .external_leases
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| current.checked_sub(1))
            {
                Ok(prior_leases) => prior_leases,
                Err(_) => {
                    *lifecycle = Lifecycle::Poisoned { reason: "binding_extension_lease_count_underflow" };
                    drop(lifecycle);
                    self.lifecycle_changed.notify_all();
                    drop(generation);
                    return;
                },
            };
        if prior_leases > 1 {
            drop(lifecycle);
            drop(generation);
            return;
        }

        let state_matches = matches!(
            &*lifecycle,
            Lifecycle::Live { generation_id, .. } if *generation_id == generation.id
        );
        if state_matches {
            *lifecycle = Lifecycle::Destroying { generation_id: generation.id, owner: current_thread };
        } else {
            *lifecycle = Lifecycle::Poisoned { reason: "binding_extension_release_state_mismatch" };
        }
        drop(lifecycle);

        let mut completion = DestructionCompletion::new(Arc::clone(self), generation.id, current_thread);
        // The user value and its native resources are destroyed outside both
        // the type-map and lifecycle locks while the slot remains Destroying.
        let destructor_phase = BindingExtensionUserPhaseGuard::enter(BindingExtensionUserPhase::Destructor);
        drop(generation);
        drop(destructor_phase);
        completion.complete();
    }

    fn complete_destruction(&self, generation_id: u64, owner: ThreadId, unwinding: bool) {
        let mut lifecycle = self.lifecycle.lock().expect("binding extension lifecycle mutex poisoned");
        let state_matches = matches!(
            &*lifecycle,
            Lifecycle::Destroying { generation_id: active_generation, owner: active_owner }
                if *active_generation == generation_id && *active_owner == owner
        );
        if state_matches && !unwinding {
            *lifecycle = Lifecycle::Vacant;
        } else {
            *lifecycle = Lifecycle::Poisoned {
                reason: if unwinding {
                    "binding_extension_destructor_panicked"
                } else {
                    "binding_extension_destruction_state_changed"
                },
            };
        }
        drop(lifecycle);
        self.lifecycle_changed.notify_all();
    }
}

struct InitializationAttempt<T>
where T: Any + Send + Sync
{
    slot: Arc<BindingExtensionSlot<T>>,
    owner: ThreadId,
    armed: bool,
}

impl<T> InitializationAttempt<T>
where T: Any + Send + Sync
{
    fn new(slot: Arc<BindingExtensionSlot<T>>, owner: ThreadId) -> Self {
        Self { slot, owner, armed: true }
    }

    fn factory_failed(&mut self) {
        let mut lifecycle = self.slot.lifecycle.lock().expect("binding extension lifecycle mutex poisoned");
        if matches!(&*lifecycle, Lifecycle::Initializing { owner } if *owner == self.owner) {
            *lifecycle = Lifecycle::Vacant;
        } else {
            *lifecycle = Lifecycle::Poisoned { reason: "binding_extension_factory_error_state_changed" };
        }
        self.armed = false;
        drop(lifecycle);
        self.slot.lifecycle_changed.notify_all();
    }

    fn poison(&mut self, reason: &'static str) {
        self.slot.poison_after_unwind(self.owner, reason);
        self.armed = false;
    }

    fn complete(&mut self) {
        self.armed = false;
    }
}

impl<T> Drop for InitializationAttempt<T>
where T: Any + Send + Sync
{
    fn drop(&mut self) {
        if self.armed {
            self.slot.poison_after_unwind(self.owner, "binding_extension_factory_panicked");
        }
    }
}

struct DestructionCompletion<T>
where T: Any + Send + Sync
{
    slot: Arc<BindingExtensionSlot<T>>,
    generation_id: u64,
    owner: ThreadId,
    armed: bool,
}

impl<T> DestructionCompletion<T>
where T: Any + Send + Sync
{
    fn new(slot: Arc<BindingExtensionSlot<T>>, generation_id: u64, owner: ThreadId) -> Self {
        Self { slot, generation_id, owner, armed: true }
    }

    fn complete(&mut self) {
        self.armed = false;
        self.slot.complete_destruction(self.generation_id, self.owner, false);
    }
}

impl<T> Drop for DestructionCompletion<T>
where T: Any + Send + Sync
{
    fn drop(&mut self) {
        if self.armed {
            self.slot.complete_destruction(self.generation_id, self.owner, true);
        }
    }
}

/// Counted lease on one binding-extension generation.
///
/// The value is intentionally opaque. It can only be borrowed through
/// [`BindingExtensionLease::enter`], which holds the slot's stable execution
/// gate. The lease has no `Deref` or ownership-extraction API.
pub struct BindingExtensionLease<T>
where T: Any + Send + Sync
{
    generation: Option<Arc<Generation<T>>>,
    slot: Arc<BindingExtensionSlot<T>>,
    // Kept until after `generation` is explicitly dropped by `Drop`, ensuring
    // extension/native destruction precedes stream and MR destruction.
    domain: Arc<ExecutionDomain>,
}

impl<T> Clone for BindingExtensionLease<T>
where T: Any + Send + Sync
{
    fn clone(&self) -> Self {
        let generation = self.generation.as_ref().expect("a live lease always owns its generation");
        generation.external_leases.fetch_add(1, Ordering::Relaxed);
        Self {
            generation: Some(Arc::clone(generation)),
            slot: Arc::clone(&self.slot),
            domain: Arc::clone(&self.domain),
        }
    }
}

impl<T> BindingExtensionLease<T>
where T: Any + Send + Sync
{
    /// Enter the stable execution gate and borrow this generation's value.
    pub fn enter(&self, operation: &'static str) -> Result<BindingExtensionGuard<'_, T>, Error> {
        let nested = ENTERED_BINDING_EXTENSION_SLOTS.with(|slots| slots.borrow().contains(&self.slot.id));
        if nested {
            return Err(BindingExtensions::lifecycle_error(
                operation,
                "binding extension execution is already entered on this thread",
                "nested_binding_extension_execution",
            ));
        }

        let execution = self.slot.execution.lock().map_err(|_| {
            BindingExtensions::lifecycle_error(
                operation,
                "binding extension execution gate is poisoned",
                "binding_extension_execution_gate_poisoned",
            )
        })?;
        ENTERED_BINDING_EXTENSION_SLOTS.with(|slots| {
            let inserted = slots.borrow_mut().insert(self.slot.id);
            debug_assert!(inserted, "nested execution was checked before locking the slot");
        });
        Ok(BindingExtensionGuard {
            lease: self,
            _execution: execution,
            _entry: BindingExtensionEntry { slot_id: self.slot.id },
        })
    }

    /// Run value-opaque native teardown under the stable execution gate.
    ///
    /// Unlike [`Self::enter`], this recovery-only path acquires a poisoned
    /// mutex through `PoisonError::into_inner` so native ownership can still
    /// be consumed and nulled exactly once. It never exposes `T`, never clears
    /// the mutex poison flag, and never makes normal value access usable again.
    /// If this thread already holds the same gate, teardown runs within that
    /// existing entered scope.
    pub fn with_teardown_gate<R>(&self, teardown: impl FnOnce() -> R) -> R {
        if self.is_entered_on_current_thread() {
            return teardown();
        }

        let execution = self.slot.execution.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        ENTERED_BINDING_EXTENSION_SLOTS.with(|slots| {
            let inserted = slots.borrow_mut().insert(self.slot.id);
            debug_assert!(inserted, "teardown entry was checked before locking the slot");
        });
        let _scope = BindingExtensionTeardownScope {
            _execution: execution,
            _entry: BindingExtensionEntry { slot_id: self.slot.id },
        };
        teardown()
    }

    /// Whether two leases refer to the same stable slot and live generation.
    pub fn same_generation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.slot, &other.slot)
            && self.generation.as_ref().map(|generation| generation.id)
                == other.generation.as_ref().map(|generation| generation.id)
    }

    /// Whether the stable execution gate is already entered on this thread.
    pub fn is_entered_on_current_thread(&self) -> bool {
        ENTERED_BINDING_EXTENSION_SLOTS.with(|slots| slots.borrow().contains(&self.slot.id))
    }
}

impl<T> Drop for BindingExtensionLease<T>
where T: Any + Send + Sync
{
    fn drop(&mut self) {
        if let Some(generation) = self.generation.take() {
            self.slot.release(generation);
        }
    }
}

struct BindingExtensionEntry {
    slot_id: u64,
}

struct BindingExtensionTeardownScope<'a> {
    // Field order is intentional: release the execution mutex before clearing
    // the thread-local reentrancy witness.
    _execution: MutexGuard<'a, ()>,
    _entry: BindingExtensionEntry,
}

impl Drop for BindingExtensionEntry {
    fn drop(&mut self) {
        ENTERED_BINDING_EXTENSION_SLOTS.with(|slots| {
            slots.borrow_mut().remove(&self.slot_id);
        });
    }
}

/// Scoped binding-extension value borrow under the stable execution gate.
pub struct BindingExtensionGuard<'a, T>
where T: Any + Send + Sync
{
    lease: &'a BindingExtensionLease<T>,
    // Field order is intentional: release the execution mutex before clearing
    // the thread-local reentrancy witness.
    _execution: MutexGuard<'a, ()>,
    _entry: BindingExtensionEntry,
}

impl<T> BindingExtensionGuard<'_, T>
where T: Any + Send + Sync
{
    /// Borrow the extension value for the lifetime of this entered scope.
    pub fn value(&self) -> &T {
        &self.lease.generation.as_ref().expect("an entered lease always owns its generation").value
    }
}

impl<T> fmt::Debug for BindingExtensionLease<T>
where T: Any + Send + Sync
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindingExtensionLease")
            .field("generation", &self.generation.as_ref().map(|generation| generation.id))
            .finish_non_exhaustive()
    }
}

impl<T> fmt::Debug for BindingExtensionGuard<'_, T>
where T: Any + Send + Sync
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindingExtensionGuard").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_map_reuses_type_ids_and_enforces_cardinality_limit() {
        let extensions = BindingExtensions::with_slot_limit(2);
        let first = extensions.slot::<u8>("slot_limit_first").expect("first slot");
        let first_again = extensions.slot::<u8>("slot_limit_first_again").expect("same slot");
        assert!(Arc::ptr_eq(&first, &first_again));
        extensions.slot::<u16>("slot_limit_second").expect("second slot");

        let error = match extensions.slot::<u32>("slot_limit_third") {
            Err(error) => error,
            Ok(_) => panic!("third type must exceed limit"),
        };
        assert_eq!(error.code(), ErrorCode::InvariantViolation);
        assert!(error.to_string().contains("binding_extension_slot_limit_reached"));
    }
}
