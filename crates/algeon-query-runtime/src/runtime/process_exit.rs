//! Process-exit coordination for engine-owned workers that can touch CUDA.
//!
//! Rust detaches a thread when its [`std::thread::JoinHandle`] is dropped. A
//! detached CUDA-capable worker can otherwise race libcudart's process
//! finalization after the last ordinary Rust owner has gone away. Each such
//! worker registers a small, non-GPU shutdown owner here. The process-exit
//! hook asks every registered owner to drain and join before control returns
//! to the C runtime's later teardown handlers.

use std::{
    collections::BTreeMap,
    ffi::c_int,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, OnceLock},
};

/// Orders shutdown around dependency direction: the backend coordinator first
/// stops new cleanup work, source readers then drain before execution-pool
/// jobs waiting on them, and stream-cleanup workers stop last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[doc(hidden)]
pub enum ProcessExitPhase {
    BackendCoordinator,
    /// Joins detached source-read workers while their leased CUDA streams are
    /// still valid. This must precede execution-pool jobs that can be waiting
    /// for a scheduled source read, as well as stream-pool shutdown.
    SourceReadWorker,
    ExecutionPool,
    StreamPool,
    ObservationCompletion,
}

/// A non-GPU owner able to synchronously stop one engine-owned worker.
///
/// Implementations must not return while their worker can still execute CUDA
/// teardown. Process exit has no safe fallback that leaves such a worker
/// detached behind libcudart destruction.
#[doc(hidden)]
pub trait ProcessExitShutdown: Send + Sync {
    fn phase(&self) -> ProcessExitPhase;

    fn shutdown_for_process_exit(&self);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct ProcessExitRegistration {
    id: Option<u64>,
}

impl ProcessExitRegistration {
    /// Removes a worker that has already stopped. This is deliberately
    /// explicit rather than a `Drop` side effect: the registration must stay
    /// live while the worker owns the last ordinary runtime reference.
    #[doc(hidden)]
    pub fn unregister(self) {
        if let Some(id) = self.id {
            global_registry().unregister(id);
        }
    }
}

#[derive(Default)]
struct ProcessExitRegistry {
    state: Mutex<ProcessExitRegistryState>,
}

#[derive(Default)]
struct ProcessExitRegistryState {
    next_id: u64,
    registrations: BTreeMap<u64, Arc<dyn ProcessExitShutdown>>,
    shutdown_started: bool,
}

impl ProcessExitRegistry {
    fn register(&self, shutdown: Arc<dyn ProcessExitShutdown>) -> ProcessExitRegistration {
        let id = {
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.shutdown_started {
                None
            } else {
                let id = state.next_id;
                state.next_id = state.next_id.checked_add(1).expect("process-exit worker registration id overflow");
                state.registrations.insert(id, Arc::clone(&shutdown));
                Some(id)
            }
        };

        if id.is_none() {
            // `shutdown` has already taken its one snapshot. A CUDA-capable
            // owner created concurrently with process exit must not be left
            // outside that snapshot, so stop it synchronously instead.
            shutdown_owner(&shutdown);
        }

        ProcessExitRegistration { id }
    }

    fn unregister(&self, id: u64) {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).registrations.remove(&id);
    }

    fn shutdown(&self) {
        let mut registrations = {
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.shutdown_started {
                return;
            }
            state.shutdown_started = true;
            std::mem::take(&mut state.registrations).into_values().collect::<Vec<_>>()
        };
        registrations.sort_by_key(|shutdown| shutdown.phase());
        for shutdown in registrations {
            shutdown_owner(&shutdown);
        }
    }

    #[cfg(test)]
    fn registration_count(&self) -> usize {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).registrations.len()
    }
}

fn shutdown_owner(shutdown: &Arc<dyn ProcessExitShutdown>) {
    // An exit hook must never unwind across the C ABI boundary. One failed
    // owner cannot justify leaving the remaining CUDA workers running, so
    // continue with every registered participant.
    let _ = catch_unwind(AssertUnwindSafe(|| shutdown.shutdown_for_process_exit()));
}

static GLOBAL_PROCESS_EXIT_REGISTRY: OnceLock<ProcessExitRegistry> = OnceLock::new();

fn global_registry() -> &'static ProcessExitRegistry {
    GLOBAL_PROCESS_EXIT_REGISTRY.get_or_init(|| {
        // libcudart is linked through the native cuDF runtime before any
        // engine-owned CUDA worker is created. Registering this hook lazily
        // therefore gives it LIFO priority over libcudart's process teardown.
        // A failure to register means we cannot uphold the worker-lifetime
        // contract, so fail during construction instead of silently detaching.
        // SAFETY: `process_exit_shutdown` has the required C ABI, does not
        // unwind, and the process-lifetime static registry outlives it.
        let status = unsafe { atexit(process_exit_shutdown) };
        assert_eq!(status, 0, "failed to register query-engine process-exit shutdown hook");
        ProcessExitRegistry::default()
    })
}

/// Registers one live CUDA-capable worker owner after entering the CUDA runtime.
///
/// The CUDA runtime installs its process-finalization handler as part of its
/// first successful runtime API call. `atexit` handlers run in reverse
/// registration order, so this call must happen before installing the engine
/// handler below: it establishes that the engine joins its workers before
/// libcudart begins teardown. A worker constructed without an available CUDA
/// runtime has no CUDA finalizer to race and therefore needs no registration.
#[doc(hidden)]
pub fn register_cuda_process_exit_shutdown(shutdown: Arc<dyn ProcessExitShutdown>) -> Option<ProcessExitRegistration> {
    // `DeviceId::current` enters the CUDA Runtime API through the canonical
    // cuDF boundary. Do not initialize the registry until that call succeeds:
    // registering first would invert `atexit` LIFO ordering if CUDA were
    // initialized later by the worker.
    cudf::DeviceId::current().ok()?;
    Some(global_registry().register(shutdown))
}

/// Runs before libcudart's later C-runtime finalizer and synchronously drains
/// the registered engine workers. No panic may cross this callback.
extern "C" fn process_exit_shutdown() {
    if let Some(registry) = GLOBAL_PROCESS_EXIT_REGISTRY.get() {
        let _ = catch_unwind(AssertUnwindSafe(|| registry.shutdown()));
    }
}

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> c_int;
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Debug)]
    struct ShutdownProbe {
        phase: ProcessExitPhase,
        label: &'static str,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ProcessExitShutdown for ShutdownProbe {
        fn phase(&self) -> ProcessExitPhase {
            self.phase
        }

        fn shutdown_for_process_exit(&self) {
            self.events.lock().unwrap().push(self.label);
        }
    }

    #[test]
    fn registry_orders_dependencies_and_drops_finished_workers() {
        let registry = ProcessExitRegistry::default();
        let events = Arc::new(Mutex::new(Vec::new()));
        let stream = registry.register(Arc::new(ShutdownProbe {
            phase: ProcessExitPhase::StreamPool,
            label: "stream",
            events: Arc::clone(&events),
        }));
        registry.register(Arc::new(ShutdownProbe {
            phase: ProcessExitPhase::BackendCoordinator,
            label: "backend",
            events: Arc::clone(&events),
        }));
        registry.register(Arc::new(ShutdownProbe {
            phase: ProcessExitPhase::ExecutionPool,
            label: "execution",
            events: Arc::clone(&events),
        }));
        registry.register(Arc::new(ShutdownProbe {
            phase: ProcessExitPhase::SourceReadWorker,
            label: "source-read",
            events: Arc::clone(&events),
        }));

        registry.unregister(stream.id.expect("local registration has an id"));
        assert_eq!(registry.registration_count(), 3);
        registry.shutdown();
        assert_eq!(*events.lock().unwrap(), ["backend", "source-read", "execution"]);
    }

    #[test]
    fn late_registration_shuts_down_inline_after_the_registry_snapshot() {
        let registry = ProcessExitRegistry::default();
        let events = Arc::new(Mutex::new(Vec::new()));
        registry.register(Arc::new(ShutdownProbe {
            phase: ProcessExitPhase::BackendCoordinator,
            label: "snapshot",
            events: Arc::clone(&events),
        }));

        registry.shutdown();
        registry.shutdown();
        let late = registry.register(Arc::new(ShutdownProbe {
            phase: ProcessExitPhase::ExecutionPool,
            label: "late",
            events: Arc::clone(&events),
        }));

        assert_eq!(*events.lock().unwrap(), ["snapshot", "late"]);
        assert_eq!(registry.registration_count(), 0);
        late.unregister();
    }
}
