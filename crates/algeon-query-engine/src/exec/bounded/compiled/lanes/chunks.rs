use std::{
    collections::BTreeMap,
    sync::{Arc, mpsc},
};

use arrow::datatypes::SchemaRef;
use cudf::{Error as CudfError, runtime::event::CudaEvent};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{AggregateNode, NodeId},
};
use query_runtime::{
    native_error::cudf_query_error,
    observability::{CleanupFailurePhase, RuntimeExecutionObservationScope},
    runtime::{ExecutionContext, GpuExecutionJobKind, PoolBlockingWork},
};

use super::{LaneDispatchSupport, LaneWorkLease, OPERATION};
use crate::exec::{
    ExecutionMetrics,
    bounded::{
        BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, CommittedPartial,
        compiled::scheduler::record_cleanup_failure, execute_compiled_source_aggregate_chunk,
    },
};

/// Owns a share of the compiled run's lanes: expanded work never acquires
/// nested permits or starts a source-read worker from an operator worker.
///
/// The dispatch owns its lane leases (cloned `Arc`s into the same attempt
/// auxiliary permits and fork subsets the run holds), so it can be moved
/// into executors and shared by reference without borrowing the run's
/// lease table. Coordinator-owned join probe work and final-sort top-k
/// candidates reuse the run's dispatcher through the same
/// [`execute_work`](Self::execute_work) window, so a run never acquires a
/// second lane set. Every owned work item carries a stable ordinal, a
/// readiness event, and exactly-once ordered commit.
#[derive(Clone)]
pub(in crate::exec::bounded) struct AggregateLaneDispatch {
    context: Arc<ExecutionContext>,
    metrics: Arc<ExecutionMetrics>,
    #[cfg(test)]
    completion_gate: Option<super::AggregateChunkCompletionGate>,
    lanes: Vec<LaneWorkLease>,
    observation_scope: Option<RuntimeExecutionObservationScope>,
}

#[derive(Default)]
struct ChunkFlights<O> {
    observation_scope: Option<RuntimeExecutionObservationScope>,
    jobs: BTreeMap<usize, ChunkFlight<O>>,
}

/// One dispatched lane job: its lane and fork facts travel with the handle so the
/// coordinator attributes the head ordinal's readiness without borrowing lane leases.
struct ChunkFlight<O> {
    lane: usize,
    fork: Option<usize>,
    handle: PoolBlockingWork<Result<LaneWorkOutput<O>>>,
}

/// One lane worker's private output with its device-side readiness.
///
/// On success the worker records `readiness` on its lane stream after the last work enqueue
/// and returns without host-synchronizing, so the pool slot recycles while the lane's work
/// drains. The coordinator synchronizes that event before committing the head ordinal.
/// `readiness` is `None` when the event could not be recorded; the worker then host-fenced
/// its stream before returning, so the output is already verified.
struct LaneWorkOutput<O> {
    output: O,
    readiness: Option<CudaEvent>,
}

/// A completed lane output waiting for its stable ordinal commit.
struct ReadyLaneOutput<O> {
    lane: usize,
    fork: Option<usize>,
    output: LaneWorkOutput<O>,
}

impl<O> Drop for ChunkFlights<O> {
    fn drop(&mut self) {
        // A panic or early error must drain before the enclosing run returns
        // its lane leases. Joined outputs are completion-observed before release so a late
        // lane failure is still attributed instead of surfacing at an unrelated later fence;
        // uncommitted outputs are released after each join.
        while let Some((ordinal, flight)) = self.jobs.pop_first() {
            let result = flight.handle.join().and_then(|result| result);
            match result {
                Ok(completed) => {
                    if let Some(readiness) = completed.readiness
                        && let Err(sync_error) = readiness.synchronize()
                    {
                        record_cleanup_failure(
                            self.observation_scope.as_ref(),
                            CleanupFailurePhase::TaskExecution,
                            &chunk_readiness_error(ordinal, flight.lane, flight.fork, sync_error),
                        );
                    }
                },
                Err(error) => {
                    record_cleanup_failure(self.observation_scope.as_ref(), CleanupFailurePhase::TaskExecution, &error);
                },
            }
        }
    }
}

/// Attributes one chunk's readiness observation with its stable dispatch facts.
fn chunk_readiness_error(ordinal: usize, lane: usize, fork: Option<usize>, sync_error: CudfError) -> QueryError {
    let mut error = cudf_query_error(
        "query_engine.compiled.chunk_readiness_sync",
        "aggregate chunk readiness failed before coordinator commit",
        sync_error,
    )
    .with_fact("chunk_ordinal", ordinal)
    .with_fact("execution_lane", lane);
    if let Some(fork) = fork {
        error = error.with_fact("attempt_fork_index", fork);
    }
    error
}

/// Observes every completed-but-uncommitted output during an abort without committing any.
///
/// Cancellation and commit failures drop the reorder buffer; each entry's readiness is still
/// synchronized first so late lane failures are attributed to their ordinal instead of
/// leaking to an unrelated later fence. Outputs release with their entries.
fn drain_ready_on_abort<O>(
    ready: BTreeMap<usize, ReadyLaneOutput<O>>,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) {
    for (ordinal, entry) in ready {
        if let Some(readiness) = entry.output.readiness
            && let Err(sync_error) = readiness.synchronize()
        {
            record_cleanup_failure(
                observation_scope,
                CleanupFailurePhase::TaskExecution,
                &chunk_readiness_error(ordinal, entry.lane, entry.fork, sync_error),
            );
        }
    }
}

impl AggregateLaneDispatch {
    pub(in crate::exec::bounded) fn width(&self) -> usize {
        self.lanes.len().max(1)
    }

    /// Reports whether this run acquired worker lanes. Empty dispatches run
    /// every chunk inline on the coordinator, so the sink output already
    /// lives on the root context and needs no root adoption.
    pub(in crate::exec::bounded) fn has_lanes(&self) -> bool {
        !self.lanes.is_empty()
    }

    pub(super) fn new(
        support: &LaneDispatchSupport,
        lanes: &[LaneWorkLease],
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Self {
        Self {
            context: Arc::clone(&support.context),
            metrics: Arc::clone(&support.metrics),
            lanes: lanes.to_vec(),
            observation_scope: observation_scope.cloned(),
            #[cfg(test)]
            completion_gate: support.aggregate_chunk_completion_gate.clone(),
        }
    }

    /// Runs legacy aggregate subplans through the same work protocol. A
    /// worker already holding a lane executes inline instead of nesting an
    /// attempt permit; a coordinator takes only currently available leases.
    pub(in crate::exec::bounded) fn with_available_lanes<R>(
        execution: &mut BoundedExecutionAttempt<'_>,
        run: impl FnOnce(&AggregateLaneDispatch, &mut BoundedExecutionAttempt<'_>) -> Result<R>,
    ) -> Result<R> {
        let context = Arc::new(execution.context().clone());
        let requested = if context.is_attempt_work_lease_context() {
            0
        } else if let Some(route) = context.attempt_stream_provider_route() {
            route.lane_capacity()
        } else if context.source_scheduler_policy().enables_prefetch() {
            context.source_scheduler_policy().max_source_reads_in_flight()
        } else {
            0
        };
        let mut lanes = if requested > 1 { LaneWorkLease::acquire_for_run(&context, requested)? } else { Vec::new() };
        if lanes.len() < 2 {
            lanes.clear();
        }
        let observation_scope = context.runtime_execution_scope().cloned();
        let dispatcher = AggregateLaneDispatch {
            context,
            metrics: Arc::new(execution.metrics().clone()),
            lanes,
            observation_scope,
            #[cfg(test)]
            completion_gate: None,
        };
        run(&dispatcher, execution)
    }

    #[expect(clippy::too_many_arguments, reason = "chunk execution carries its aggregate occurrence and schemas")]
    pub(in crate::exec::bounded) fn execute_chunks(
        &self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        aggregate: &AggregateNode,
        input_schema: &SchemaRef,
        output_schema: &SchemaRef,
        stream: BoundedJoinLeafLocalParquetSourceStream,
        mut consume: impl FnMut(&mut BoundedExecutionAttempt<'_>, Vec<CommittedPartial>) -> Result<()>,
    ) -> Result<()> {
        let stream = Arc::new(stream);
        let aggregate = aggregate.clone();
        let input_schema = Arc::clone(input_schema);
        let output_schema = Arc::clone(output_schema);
        let mut chunks = stream.chunks().to_vec().into_iter();
        self.execute_work(
            execution,
            &mut chunks,
            |chunks, _| Ok(chunks.next()),
            move |execution, chunk| {
                execute_compiled_source_aggregate_chunk(
                    execution,
                    node_id,
                    &aggregate,
                    &input_schema,
                    &output_schema,
                    &stream,
                    chunk,
                )
            },
            |_, execution, output| {
                consume(execution, output.partials)?;
                execution.metrics().record_source_chunk_commit(output.source_row_groups);
                Ok(())
            },
        )
    }

    /// Drives source-owned work through this compiled run's lane window.
    /// Selection and commit share coordinator state; workers receive only
    /// owned work and return private outputs. Ready outputs consume capacity
    /// until their stable ordinal commits, bounding the reorder buffer.
    pub(in crate::exec::bounded) fn execute_work<S, W: Send + 'static, O: Send + 'static>(
        &self,
        execution: &mut BoundedExecutionAttempt<'_>,
        state: &mut S,
        mut next: impl FnMut(&mut S, &mut BoundedExecutionAttempt<'_>) -> Result<Option<W>>,
        work: impl Fn(&mut BoundedExecutionAttempt<'_>, W) -> Result<O> + Send + Sync + 'static,
        mut commit: impl FnMut(&mut S, &mut BoundedExecutionAttempt<'_>, O) -> Result<()>,
    ) -> Result<()> {
        if self.lanes.is_empty() {
            let mut ordinal = 0;
            loop {
                execution.context().check_cancellation(OPERATION)?;
                let Some(input) = next(state, execution)? else {
                    return Ok(());
                };
                let result = work(execution, input);
                let output = self.finish_work(execution, result)?;
                #[cfg(test)]
                if let Some(gate) = &self.completion_gate {
                    gate(ordinal, 0, execution.context().attempt_claimed_fork_index());
                }
                execution.context().check_cancellation(OPERATION)?;
                commit(state, execution, output)?;
                ordinal += 1;
                let _ = ordinal;
            }
        }

        let pool = self.context.gpu_executor()?;
        let work = Arc::new(work);
        let (completed, completions) = mpsc::channel();
        let mut flights = ChunkFlights { observation_scope: self.observation_scope.clone(), jobs: BTreeMap::new() };
        let mut ready: BTreeMap<usize, ReadyLaneOutput<O>> = BTreeMap::new();
        let mut free_lanes = (0..self.lanes.len()).collect::<Vec<_>>();
        let mut dispatched = 0;
        let mut committed = 0;
        let mut exhausted = false;
        loop {
            if let Err(error) = execution.context().check_cancellation(OPERATION) {
                drain_ready_on_abort(ready, self.observation_scope.as_ref());
                return Err(error);
            }
            while !exhausted && flights.jobs.len() + ready.len() < self.lanes.len() {
                // Before a pull failure aborts the run, every completed,
                // uncommitted output has its readiness observed so that a
                // late lane failure is attributed to its own ordinal and not
                // to an unrelated later fence. In-flight jobs drain through
                // their join below.
                let input = match next(state, execution) {
                    Ok(input) => input,
                    Err(error) => {
                        drain_ready_on_abort(ready, self.observation_scope.as_ref());
                        return Err(error);
                    },
                };
                let Some(input) = input else {
                    exhausted = true;
                    break;
                };
                let lane = free_lanes.pop().expect("an available window slot owns a free lane");
                let ordinal = dispatched;
                let lease = Arc::clone(&self.lanes[lane].lease);
                let fork = lease.context().attempt_claimed_fork_index();
                let work = Arc::clone(&work);
                let metrics = Arc::clone(&self.metrics);
                let observation_scope = self.observation_scope.clone();
                let notification = completed.clone();
                #[cfg(test)]
                let gate = self.completion_gate.clone();
                // A saturated pool submission fails without dispatching: the
                // ready outputs still need their readiness observed before
                // the abort for the same attribution reason as above.
                let handle = match pool.submit_blocking_with_completion(
                    GpuExecutionJobKind::AttemptWork,
                    self.context.runtime_observation().clone(),
                    move || {
                        let context = lease.context();
                        let mut execution =
                            BoundedExecutionAttempt::new(context, context.device_budget_bytes(), &metrics);
                        let result = work(&mut execution, input);
                        let output = finish_lane_work(&execution, result, observation_scope.as_ref())?;
                        #[cfg(test)]
                        if let Some(gate) = gate {
                            gate(ordinal, lane, context.attempt_claimed_fork_index());
                        }
                        Ok(output)
                    },
                    move || {
                        let _ = notification.send(ordinal);
                    },
                ) {
                    Ok(handle) => handle,
                    Err(error) => {
                        drain_ready_on_abort(ready, self.observation_scope.as_ref());
                        return Err(error);
                    },
                };
                flights.jobs.insert(ordinal, ChunkFlight { lane, fork, handle });
                dispatched += 1;
            }
            if flights.jobs.is_empty() {
                debug_assert!(ready.is_empty());
                debug_assert!(exhausted);
                return Ok(());
            }
            // Runtime guarantees one notification for every accepted handle,
            // including work discarded before starting during pool shutdown.
            let ordinal = match completions.recv() {
                Ok(ordinal) => ordinal,
                Err(_) => {
                    drain_ready_on_abort(ready, self.observation_scope.as_ref());
                    return Err(QueryError::new(
                        QueryErrorCode::Cancelled,
                        OPERATION,
                        "aggregate chunk completion channel closed",
                    ));
                },
            };
            let flight = flights.jobs.remove(&ordinal).expect("completion belongs to in-flight chunk");
            free_lanes.push(flight.lane);
            let completed = match flight.handle.join() {
                Ok(Ok(completed)) => completed,
                Ok(Err(error)) | Err(error) => {
                    drain_ready_on_abort(ready, self.observation_scope.as_ref());
                    return Err(error);
                },
            };
            ready.insert(ordinal, ReadyLaneOutput { lane: flight.lane, fork: flight.fork, output: completed });
            if let Err(error) = execution.context().check_cancellation(OPERATION) {
                drain_ready_on_abort(ready, self.observation_scope.as_ref());
                return Err(error);
            }
            while let Some(ready_output) = ready.remove(&committed) {
                // Synchronizing the head ordinal's readiness is the coordinator's completion
                // observation for that lane's work: a late lane failure is reported here with
                // its ordinal facts, and the run aborts without committing this or any later
                // ordinal. Source retry cannot replay input after any of its partials entered
                // coordinator state.
                if let Some(readiness) = ready_output.output.readiness
                    && let Err(sync_error) = readiness.synchronize()
                {
                    let error = chunk_readiness_error(committed, ready_output.lane, ready_output.fork, sync_error);
                    drain_ready_on_abort(ready, self.observation_scope.as_ref());
                    return Err(error);
                }
                if let Err(error) = commit(state, execution, ready_output.output.output) {
                    drain_ready_on_abort(ready, self.observation_scope.as_ref());
                    return Err(error);
                }
                committed += 1;
            }
        }
    }

    fn finish_work<O>(&self, execution: &BoundedExecutionAttempt<'_>, result: Result<O>) -> Result<O> {
        finish_chunk_work(execution, result, self.observation_scope.as_ref())
    }
}

/// Completes one lane worker's chunk.
///
/// On success the worker records a readiness event on its lane stream after the last work
/// enqueue and returns without host-synchronizing; the coordinator synchronizes the head
/// ordinal before commit, and cross-stream consumers order device-side through the retained
/// provenance fences. When the event cannot be recorded, the worker host-fences its stream
/// and returns `readiness: None`, so success is never reported for unverified work. A work
/// failure also host-fences the stream: the original error is returned, and a fault observed
/// by that fence is recorded as a cleanup failure.
fn finish_lane_work<O>(
    execution: &BoundedExecutionAttempt<'_>,
    result: Result<O>,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<LaneWorkOutput<O>> {
    match result {
        Ok(output) => {
            let readiness = match execution.context().runtime().and_then(|runtime| {
                runtime.record_completion_event().map_err(|error| {
                    cudf_query_error(
                        "query_engine.compiled.chunk_readiness_record",
                        "failed to record aggregate chunk readiness on the lane stream",
                        error,
                    )
                })
            }) {
                Ok(event) => Some(event),
                Err(record_error) => {
                    // Event creation or recording can report a prior asynchronous error, so the
                    // record failure is reported before the host fence verifies the output. A
                    // failed fence fails the chunk with the fence error.
                    eprintln!(
                        "[query_engine::finish_lane_work] warning: failed to record chunk readiness on the lane stream, falling back to host synchronization: {record_error}"
                    );
                    execution.context().runtime().and_then(|runtime| {
                        runtime.synchronize().map_err(|error| {
                            cudf_query_error(
                                "query_engine.compiled.aggregate_chunk_sync",
                                "failed to synchronize aggregate chunk work",
                                error,
                            )
                        })
                    })?;
                    None
                },
            };
            Ok(LaneWorkOutput { output, readiness })
        },
        Err(error) => {
            if let Err(sync_error) = execution.context().runtime().and_then(|runtime| {
                runtime.synchronize().map_err(|error| {
                    cudf_query_error(
                        "query_engine.compiled.aggregate_chunk_sync",
                        "failed to synchronize failed aggregate chunk work",
                        error,
                    )
                })
            }) {
                record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskExecution, &sync_error);
            }
            Err(error)
        },
    }
}

fn finish_chunk_work<O>(
    execution: &BoundedExecutionAttempt<'_>,
    result: Result<O>,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<O> {
    let synchronized = execution.context().runtime().and_then(|runtime| {
        runtime.synchronize().map_err(|error| {
            cudf_query_error(
                "query_engine.compiled.aggregate_chunk_sync",
                "failed to synchronize aggregate chunk work",
                error,
            )
        })
    });
    match result {
        Ok(output) => {
            synchronized?;
            Ok(output)
        },
        Err(error) => {
            if let Err(cleanup) = synchronized {
                record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskExecution, &cleanup);
            }
            Err(error)
        },
    }
}
