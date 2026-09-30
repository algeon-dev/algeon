//! Detached observation-completion delivery for a GPU backend.
//!
//! This module owns the host-only workers that finish observation delivery
//! after GPU state has been released. It deliberately has no dependency on a
//! backend resource generation, so callers may retain a dispatcher after the
//! GPU resource owner has drained.

use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::process_exit::{
    ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
};

const BUILD_OPERATION: &str = "query_engine.gpu_backend_core.build";

type ObservationCompletionCallback = Box<dyn FnOnce() + Send + 'static>;

/// A completion callback that recovers itself if the worker drops it before
/// explicitly running it.
///
/// A receiver drops its queued messages when the worker exits unexpectedly.
/// Permit release and observer sealing are terminal work, so silently dropping
/// either would strand backend observation quiescence.
struct ObservationCompletionTask {
    callback: Option<ObservationCompletionCallback>,
}

impl ObservationCompletionTask {
    fn new<F>(callback: F) -> Self
    where F: FnOnce() + Send + 'static {
        Self { callback: Some(Box::new(callback)) }
    }

    fn run(mut self) {
        if let Some(callback) = self.callback.take() {
            execute_observation_completion(callback);
        }
    }
}

impl Drop for ObservationCompletionTask {
    fn drop(&mut self) {
        if let Some(callback) = self.callback.take() {
            execute_observation_completion(callback);
        }
    }
}

enum ObservationCompletionMessage {
    Completion(ObservationCompletionTask),
    #[cfg(test)]
    InjectWorkerFaultForTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObservationCompletionWorkerLane {
    Observation,
    ActivityTail,
}

impl ObservationCompletionWorkerLane {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "observation",
            Self::ActivityTail => "activity_tail",
        }
    }

    const fn thread_name(self) -> &'static str {
        match self {
            Self::Observation => "algeon-observation-completion-observation",
            Self::ActivityTail => "algeon-observation-completion-activity-tail",
        }
    }

    const fn slot(self) -> usize {
        match self {
            Self::Observation => 0,
            Self::ActivityTail => 1,
        }
    }
}

const OBSERVATION_COMPLETION_WORKER_RUNNING: u8 = 0;
const OBSERVATION_COMPLETION_WORKER_STOPPED: u8 = 1;
const OBSERVATION_COMPLETION_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Supervisor-visible state for the detached completion worker.
///
/// A caught supervisor fault is retained as a diagnostic without destroying
/// the only pre-created delivery thread. The worker continues consuming its
/// receiver, which keeps completion off callers such as the stream-cleanup
/// supervisor. Receiver teardown still recovers queued RAII tasks.
#[derive(Debug)]
struct ObservationCompletionExecutorState {
    worker_terminal: AtomicU8,
    worker_fault_count: AtomicU64,
    shutdown_requested: AtomicBool,
}

impl ObservationCompletionExecutorState {
    fn new() -> Self {
        Self {
            worker_terminal: AtomicU8::new(OBSERVATION_COMPLETION_WORKER_RUNNING),
            worker_fault_count: AtomicU64::new(0),
            shutdown_requested: AtomicBool::new(false),
        }
    }

    fn accepts_dispatch(&self) -> bool {
        !self.shutdown_requested.load(Ordering::Acquire)
            && self.worker_terminal.load(Ordering::Acquire) == OBSERVATION_COMPLETION_WORKER_RUNNING
    }

    fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
    }

    fn mark_stopped(&self) {
        self.worker_terminal.store(OBSERVATION_COMPLETION_WORKER_STOPPED, Ordering::Release);
    }

    fn record_fault(&self) {
        self.worker_fault_count.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn worker_fault_count(&self) -> u64 {
        self.worker_fault_count.load(Ordering::Acquire)
    }
}

/// Retains the completion-worker join handles until their final owner either
/// drains naturally or the process-exit registry stops them. Completion work
/// is host-side, but activity-tail destructors can reenter backend lifecycle
/// code, so their workers must not outlive CUDA's process finalizer either.
#[derive(Debug)]
struct ObservationCompletionSupervisor {
    observation_state: Arc<ObservationCompletionExecutorState>,
    activity_tail_state: Arc<ObservationCompletionExecutorState>,
    join: Mutex<[Option<thread::JoinHandle<()>>; 2]>,
    active_workers: AtomicUsize,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

impl ObservationCompletionSupervisor {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            observation_state: Arc::new(ObservationCompletionExecutorState::new()),
            activity_tail_state: Arc::new(ObservationCompletionExecutorState::new()),
            join: Mutex::new([None, None]),
            active_workers: AtomicUsize::new(0),
            process_exit_registration: Mutex::new(None),
        })
    }

    fn state(&self, lane: ObservationCompletionWorkerLane) -> Arc<ObservationCompletionExecutorState> {
        match lane {
            ObservationCompletionWorkerLane::Observation => Arc::clone(&self.observation_state),
            ObservationCompletionWorkerLane::ActivityTail => Arc::clone(&self.activity_tail_state),
        }
    }

    fn spawn_worker(
        self: &Arc<Self>,
        lane: ObservationCompletionWorkerLane,
        state: Arc<ObservationCompletionExecutorState>,
    ) -> Result<mpsc::Sender<ObservationCompletionMessage>> {
        let (sender, receiver) = mpsc::channel::<ObservationCompletionMessage>();
        self.active_workers.fetch_add(1, Ordering::AcqRel);
        let worker_supervisor = Arc::clone(self);
        let failed_supervisor = Arc::clone(self);
        let name = lane.thread_name();
        let worker = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                run_observation_completion_worker(receiver, state, name);
                worker_supervisor.worker_stopped();
            })
            .map_err(|error| {
                failed_supervisor.active_workers.fetch_sub(1, Ordering::AcqRel);
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to create backend observation completion worker",
                )
                .with_fact("reason", "observation_completion_worker_spawn_failed")
                .with_fact("worker_lane", lane.as_str())
                .with_fact("io_error", error.to_string())
            })?;
        self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[lane.slot()] = Some(worker);
        Ok(sender)
    }

    fn activate_process_exit(self: &Arc<Self>) {
        self.activate_process_exit_with(register_cuda_process_exit_shutdown);
    }

    /// Registers outside the slot mutex. Process-exit registration may invoke
    /// the supplied shutdown owner synchronously when global exit has already
    /// begun; that owner joins workers and re-enters `unregister_process_exit`.
    /// Keeping the mutex out of that callback path prevents a self-deadlock.
    fn activate_process_exit_with(
        self: &Arc<Self>,
        register: impl FnOnce(Arc<dyn ProcessExitShutdown>) -> Option<ProcessExitRegistration>,
    ) {
        {
            let registration = self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if registration.is_some() || self.active_workers.load(Ordering::Acquire) == 0 {
                return;
            }
        }

        let shutdown: Arc<dyn ProcessExitShutdown> = Arc::clone(self) as _;
        let registration = register(shutdown);
        let registration_to_unregister = {
            let mut slot = self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.active_workers.load(Ordering::Acquire) == 0 || slot.is_some() {
                registration
            } else {
                *slot = registration;
                None
            }
        };
        if let Some(registration) = registration_to_unregister {
            registration.unregister();
        }
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    fn worker_stopped(&self) {
        if self.active_workers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.unregister_process_exit();
        }
    }

    fn request_shutdown(&self) {
        self.observation_state.request_shutdown();
        self.activity_tail_state.request_shutdown();
    }

    fn join_for_process_exit(&self) {
        self.request_shutdown();
        let mut join = self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for slot in join.iter_mut() {
            let Some(handle) = slot.take() else {
                continue;
            };
            if handle.thread().id() == thread::current().id() {
                *slot = Some(handle);
                continue;
            }
            let _ = handle.join();
        }
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for ObservationCompletionSupervisor {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::ObservationCompletion
    }

    fn shutdown_for_process_exit(&self) {
        self.join_for_process_exit();
    }
}

/// Cloneable post-GPU cleanup dispatch capability.
///
/// The observation lane seals and delivers snapshots before releasing an
/// observation permit. The activity-tail lane drops opaque query activity only
/// after that fence is gone. Each lane has a pre-created worker and neither
/// holds a GPU resource.
#[derive(Clone)]
pub(crate) struct ObservationCompletionDispatcher {
    observation_sender: mpsc::Sender<ObservationCompletionMessage>,
    observation_state: Arc<ObservationCompletionExecutorState>,
    activity_tail_sender: mpsc::Sender<ObservationCompletionMessage>,
    activity_tail_state: Arc<ObservationCompletionExecutorState>,
}

impl fmt::Debug for ObservationCompletionDispatcher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ObservationCompletionDispatcher").finish_non_exhaustive()
    }
}

impl ObservationCompletionDispatcher {
    /// Runs one finalization payload after its caller has released GPU state.
    ///
    /// A stopped or disconnected worker cannot consume the payload, so the
    /// dispatcher synchronously executes it. A recoverable worker fault does
    /// not take this path: the pre-created worker records it and continues.
    pub(crate) fn dispatch<F>(&self, completion: F)
    where F: FnOnce() + Send + 'static {
        self.dispatch_to(&self.observation_sender, &self.observation_state, completion);
    }

    /// Runs opaque activity destruction after the observation lane released
    /// its permit. This distinct lane prevents a blocking activity callback
    /// from starving later observation completion required by backend close.
    pub(crate) fn dispatch_activity_tail<F>(&self, completion: F)
    where F: FnOnce() + Send + 'static {
        self.dispatch_to(&self.activity_tail_sender, &self.activity_tail_state, completion);
    }

    fn dispatch_to<F>(
        &self,
        sender: &mpsc::Sender<ObservationCompletionMessage>,
        state: &ObservationCompletionExecutorState,
        completion: F,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let completion = ObservationCompletionTask::new(completion);
        if !state.accepts_dispatch() {
            completion.run();
            return;
        }
        if let Err(mpsc::SendError(ObservationCompletionMessage::Completion(completion))) =
            sender.send(ObservationCompletionMessage::Completion(completion))
        {
            completion.run();
        }
    }

    #[cfg(test)]
    fn disconnected_for_test() -> Self {
        let (observation_sender, observation_receiver) = mpsc::channel();
        let (activity_tail_sender, activity_tail_receiver) = mpsc::channel();
        drop(observation_receiver);
        drop(activity_tail_receiver);
        Self {
            observation_sender,
            observation_state: Arc::new(ObservationCompletionExecutorState::new()),
            activity_tail_sender,
            activity_tail_state: Arc::new(ObservationCompletionExecutorState::new()),
        }
    }
}

/// Pre-created, backend-owned worker for detached observation completion.
///
/// The executor owns no backend resources. Its dispatcher may be retained by a
/// query after the resource generation is gone, so an enqueue failure must
/// execute the payload inline rather than drop its permit. Both receivers drain
/// accepted payloads after the executor owner drops; dispatcher clones retain
/// senders until every future completion has either been enqueued or
/// synchronously finalized.
#[derive(Debug)]
pub(crate) struct ObservationCompletionExecutor {
    dispatcher: ObservationCompletionDispatcher,
    supervisor: Arc<ObservationCompletionSupervisor>,
}

impl ObservationCompletionExecutor {
    pub(crate) fn spawn() -> Result<Self> {
        Self::spawn_with_lane_spawner(|lane, state, supervisor| supervisor.spawn_worker(lane, state))
    }

    fn spawn_with_lane_spawner(
        spawn_lane: impl Fn(
            ObservationCompletionWorkerLane,
            Arc<ObservationCompletionExecutorState>,
            Arc<ObservationCompletionSupervisor>,
        ) -> Result<mpsc::Sender<ObservationCompletionMessage>>,
    ) -> Result<Self> {
        let supervisor = ObservationCompletionSupervisor::new();
        let observation_state = supervisor.state(ObservationCompletionWorkerLane::Observation);
        let observation_sender = spawn_lane(
            ObservationCompletionWorkerLane::Observation,
            Arc::clone(&observation_state),
            Arc::clone(&supervisor),
        )?;
        let activity_tail_state = supervisor.state(ObservationCompletionWorkerLane::ActivityTail);
        let activity_tail_sender = match spawn_lane(
            ObservationCompletionWorkerLane::ActivityTail,
            Arc::clone(&activity_tail_state),
            Arc::clone(&supervisor),
        ) {
            Ok(sender) => sender,
            Err(error) => {
                // Explicitly disconnect the already-started observation
                // lane before returning an unpublished construction error.
                drop(observation_sender);
                supervisor.join_for_process_exit();
                return Err(error);
            },
        };
        Ok(Self {
            dispatcher: ObservationCompletionDispatcher {
                observation_sender,
                observation_state,
                activity_tail_sender,
                activity_tail_state,
            },
            supervisor,
        })
    }

    /// Registers both lanes only after backend service construction has
    /// entered CUDA. This keeps the engine's `atexit` callback ahead of
    /// libcudart while preserving the executor's detached delivery contract.
    pub(crate) fn activate_process_exit(&self) {
        self.supervisor.activate_process_exit();
    }

    #[cfg(test)]
    pub(crate) fn spawn_with_lane_failure_for_test(failing_lane: ObservationCompletionWorkerLane) -> Result<Self> {
        Self::spawn_with_lane_spawner(|lane, state, supervisor| {
            if lane == failing_lane {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "injected backend observation completion worker spawn failure",
                )
                .with_fact("reason", "observation_completion_worker_spawn_failed")
                .with_fact("worker_lane", lane.as_str()));
            }
            supervisor.spawn_worker(lane, state)
        })
    }

    pub(crate) fn dispatcher(&self) -> ObservationCompletionDispatcher {
        self.dispatcher.clone()
    }

    #[cfg(test)]
    fn inject_observation_worker_fault_for_test(&self) {
        self.dispatcher
            .observation_sender
            .send(ObservationCompletionMessage::InjectWorkerFaultForTest)
            .expect("test completion worker is connected");
    }

    #[cfg(test)]
    fn observation_worker_fault_count(&self) -> u64 {
        self.dispatcher.observation_state.worker_fault_count()
    }
}

fn run_observation_completion_worker(
    receiver: mpsc::Receiver<ObservationCompletionMessage>,
    state: Arc<ObservationCompletionExecutorState>,
    name: &'static str,
) {
    loop {
        if state.shutdown_requested.load(Ordering::Acquire) {
            state.mark_stopped();
            return;
        }
        let received =
            catch_unwind(AssertUnwindSafe(|| match receiver.recv_timeout(OBSERVATION_COMPLETION_POLL_INTERVAL) {
                Ok(message) => {
                    match message {
                        ObservationCompletionMessage::Completion(completion) => completion.run(),
                        #[cfg(test)]
                        ObservationCompletionMessage::InjectWorkerFaultForTest => {
                            panic!("injected observation completion worker fault");
                        },
                    }
                    true
                },
                Err(mpsc::RecvTimeoutError::Timeout) => true,
                Err(mpsc::RecvTimeoutError::Disconnected) => false,
            }));
        match received {
            Ok(true) => {},
            Ok(false) => {
                state.mark_stopped();
                return;
            },
            Err(_) => {
                state.record_fault();
                eprintln!(
                    "[query-engine] warning: backend observation completion worker {name} recovered from a fault"
                );
            },
        }
    }
}

fn execute_observation_completion(completion: ObservationCompletionCallback) {
    if catch_unwind(AssertUnwindSafe(completion)).is_err() {
        eprintln!("[query-engine] warning: backend observation completion callback panicked");
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Condvar, Mutex, mpsc},
        thread,
        time::Duration,
    };

    use super::{ObservationCompletionDispatcher, ObservationCompletionExecutor, ObservationCompletionSupervisor};

    #[derive(Debug, Default)]
    struct BlockingGate {
        released: Mutex<bool>,
        changed: Condvar,
    }

    impl BlockingGate {
        fn wait(&self) {
            let mut released = self.released.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*released {
                released = self.changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }

        fn release(&self) {
            *self.released.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            self.changed.notify_all();
        }
    }

    #[test]
    fn executor_runs_detached_callbacks_and_isolates_panics() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(2);

        dispatcher.dispatch({
            let sender = sender.clone();
            move || sender.send(thread::current().id()).expect("test receives completion worker id")
        });
        assert_ne!(
            receiver.recv_timeout(Duration::from_secs(2)).expect("completion worker ran callback"),
            caller,
            "pre-created completion worker must detach finalization from the caller"
        );

        dispatcher.dispatch(|| panic!("injected observation completion panic"));
        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("worker survives callback panic"));
        assert_ne!(
            receiver.recv_timeout(Duration::from_secs(2)).expect("completion worker survived callback panic"),
            caller
        );
    }

    #[test]
    fn observation_and_activity_tail_callbacks_use_distinct_precreated_workers() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let (sender, receiver) = mpsc::sync_channel(2);

        dispatcher.dispatch({
            let sender = sender.clone();
            move || sender.send(thread::current().id()).expect("test observes observation worker")
        });
        dispatcher.dispatch_activity_tail(move || {
            sender.send(thread::current().id()).expect("test observes activity-tail worker")
        });

        let observation_worker = receiver.recv_timeout(Duration::from_secs(2)).expect("observation callback ran");
        let activity_tail_worker = receiver.recv_timeout(Duration::from_secs(2)).expect("activity-tail callback ran");
        assert_ne!(
            observation_worker, activity_tail_worker,
            "activity drops must not occupy observation delivery capacity"
        );
    }

    #[test]
    fn process_exit_joins_workers_and_closes_dispatch() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();

        executor.supervisor.join_for_process_exit();

        assert_eq!(executor.supervisor.active_workers.load(std::sync::atomic::Ordering::Acquire), 0);
        assert!(executor.supervisor.join.lock().unwrap().iter().all(Option::is_none));
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(1);
        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("fallback dispatch runs"));
        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).expect("fallback completion runs"), caller);
    }

    #[test]
    fn late_process_exit_registration_runs_callback_without_holding_registration_slot_mutex() {
        let supervisor = ObservationCompletionSupervisor::new();
        // The injected registration path models a global process-exit race:
        // registry registration synchronously invokes the owner, which in
        // turn calls `unregister_process_exit`.
        supervisor.active_workers.store(1, std::sync::atomic::Ordering::Release);
        let callback_supervisor = Arc::clone(&supervisor);
        let callback_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_ran_slot = Arc::clone(&callback_ran);

        supervisor.activate_process_exit_with(move |shutdown| {
            assert!(
                callback_supervisor.process_exit_registration.try_lock().is_ok(),
                "registration slot mutex must be unlocked while registry registration calls back"
            );
            shutdown.shutdown_for_process_exit();
            callback_ran_slot.store(true, std::sync::atomic::Ordering::Release);
            None
        });

        assert!(callback_ran.load(std::sync::atomic::Ordering::Acquire));
        supervisor.active_workers.store(0, std::sync::atomic::Ordering::Release);
    }

    #[test]
    fn dispatcher_synchronously_finalizes_when_enqueue_fails() {
        let dispatcher = ObservationCompletionDispatcher::disconnected_for_test();
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(1);

        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("fallback runs completion"));

        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).expect("fallback completed"), caller);
    }

    #[test]
    fn recoverable_worker_fault_keeps_later_delivery_off_caller() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(2);

        dispatcher.dispatch({
            let sender = sender.clone();
            move || sender.send(thread::current().id()).expect("test observes executor worker thread")
        });
        let worker_thread =
            receiver.recv_timeout(Duration::from_secs(2)).expect("executor worker ran initial callback");

        executor.inject_observation_worker_fault_for_test();
        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("recovered queued payload completed"));

        let delivery_thread =
            receiver.recv_timeout(Duration::from_secs(2)).expect("worker resumes queued completion after a fault");
        assert_eq!(delivery_thread, worker_thread, "recovered delivery must stay on the pre-created executor thread");
        assert_ne!(delivery_thread, caller, "recovered delivery must not run on the caller or stream supervisor");
        assert_eq!(executor.observation_worker_fault_count(), 1, "worker supervisor records the recovered fault");
    }

    #[test]
    fn worker_drains_accepted_payloads_after_executor_and_dispatchers_drop() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let gate = Arc::new(BlockingGate::default());
        let worker_gate = Arc::clone(&gate);
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (completed_sender, completed_receiver) = mpsc::sync_channel(1);

        dispatcher.dispatch(move || {
            entered_sender.send(()).expect("test observes first completion callback");
            worker_gate.wait();
        });
        entered_receiver.recv_timeout(Duration::from_secs(2)).expect("completion worker began first callback");
        dispatcher.dispatch(move || completed_sender.send(()).expect("test observes drained completion callback"));

        drop(dispatcher);
        drop(executor);
        gate.release();

        completed_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker drained accepted completion after owner drop");
    }
}
