use std::{
    collections::BTreeMap,
    sync::{Arc, mpsc},
};

use arrow::datatypes::SchemaRef;

use super::{LaneDispatchSupport, LaneWorkLease, OPERATION};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    exec::bounded::{
        BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, CommittedPartial,
        execute_compiled_source_aggregate_chunk,
    },
    native_error::cudf_query_error,
    plan::{AggregateNode, NodeId},
    runtime::{GpuExecutionJobKind, PoolBlockingWork},
};

/// Borrows the compiled run's lanes: expanded work never acquires nested
/// permits or starts a source-read worker from an operator worker.
pub(in crate::exec::bounded) struct AggregateLaneDispatch<'a> {
    context: Arc<crate::runtime::ExecutionContext>,
    metrics: Arc<crate::exec::ExecutionMetrics>,
    #[cfg(any(test, feature = "test-hooks"))]
    completion_gate: Option<super::AggregateChunkCompletionGate>,
    lanes: &'a [LaneWorkLease],
    observation_scope: Option<&'a crate::observability::RuntimeExecutionObservationScope>,
}

#[derive(Default)]
struct ChunkFlights<'a, O> {
    observation_scope: Option<&'a crate::observability::RuntimeExecutionObservationScope>,
    jobs: BTreeMap<usize, (usize, PoolBlockingWork<Result<O>>)>,
}

impl<O> Drop for ChunkFlights<'_, O> {
    fn drop(&mut self) {
        // A panic or early error must drain before the enclosing run returns
        // its lane leases. Uncommitted outputs are released after each join.
        while let Some((_, (_, handle))) = self.jobs.pop_first() {
            let result = handle.join().and_then(|result| result);
            if let Err(error) = result {
                super::super::scheduler::record_cleanup_failure(
                    self.observation_scope,
                    crate::observability::CleanupFailurePhase::TaskExecution,
                    &error,
                );
            }
        }
    }
}

impl<'a> AggregateLaneDispatch<'a> {
    pub(in crate::exec::bounded) fn width(&self) -> usize {
        self.lanes.len().max(1)
    }

    pub(super) fn new(
        support: &'a LaneDispatchSupport,
        lanes: &'a [LaneWorkLease],
        observation_scope: Option<&'a crate::observability::RuntimeExecutionObservationScope>,
    ) -> Self {
        Self {
            context: Arc::clone(&support.context),
            metrics: Arc::clone(&support.metrics),
            lanes,
            observation_scope,
            #[cfg(any(test, feature = "test-hooks"))]
            completion_gate: support.aggregate_chunk_completion_gate.clone(),
        }
    }

    /// Runs legacy aggregate subplans through the same work protocol. A
    /// worker already holding a lane executes inline instead of nesting an
    /// attempt permit; a coordinator takes only currently available leases.
    pub(in crate::exec::bounded) fn with_available_lanes<R>(
        execution: &mut BoundedExecutionAttempt<'_>,
        run: impl FnOnce(&AggregateLaneDispatch<'_>, &mut BoundedExecutionAttempt<'_>) -> Result<R>,
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
            lanes: &lanes,
            observation_scope: observation_scope.as_ref(),
            #[cfg(any(test, feature = "test-hooks"))]
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
                #[cfg(any(test, feature = "test-hooks"))]
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
        let mut flights = ChunkFlights { observation_scope: self.observation_scope, jobs: BTreeMap::new() };
        let mut ready = BTreeMap::new();
        let mut free_lanes = (0..self.lanes.len()).collect::<Vec<_>>();
        let mut dispatched = 0;
        let mut committed = 0;
        let mut exhausted = false;
        loop {
            execution.context().check_cancellation(OPERATION)?;
            while !exhausted && flights.jobs.len() + ready.len() < self.lanes.len() {
                let Some(input) = next(state, execution)? else {
                    exhausted = true;
                    break;
                };
                let lane = free_lanes.pop().expect("an available window slot owns a free lane");
                let ordinal = dispatched;
                let lease = Arc::clone(&self.lanes[lane].lease);
                let work = Arc::clone(&work);
                let metrics = Arc::clone(&self.metrics);
                let observation_scope = self.observation_scope.cloned();
                let notification = completed.clone();
                #[cfg(any(test, feature = "test-hooks"))]
                let gate = self.completion_gate.clone();
                let handle = pool.submit_blocking_with_completion(
                    GpuExecutionJobKind::AttemptWork,
                    self.context.runtime_observation().clone(),
                    move || {
                        let context = lease.context();
                        let budget = context.bounded_device_budget_bytes().expect("compiled work is bounded");
                        let mut execution = BoundedExecutionAttempt::new(context, budget, &metrics)?;
                        let result = work(&mut execution, input);
                        let output = finish_chunk_work(&execution, result, observation_scope.as_ref())?;
                        #[cfg(any(test, feature = "test-hooks"))]
                        if let Some(gate) = gate {
                            gate(ordinal, lane, context.attempt_claimed_fork_index());
                        }
                        Ok(output)
                    },
                    move || {
                        let _ = notification.send(ordinal);
                    },
                )?;
                flights.jobs.insert(ordinal, (lane, handle));
                dispatched += 1;
            }
            if flights.jobs.is_empty() {
                debug_assert!(ready.is_empty());
                debug_assert!(exhausted);
                return Ok(());
            }
            // Runtime guarantees one notification for every accepted handle,
            // including work discarded before starting during pool shutdown.
            let ordinal = completions.recv().map_err(|_| {
                QueryError::new(QueryErrorCode::Cancelled, OPERATION, "aggregate chunk completion channel closed")
            })?;
            let (lane, handle) = flights.jobs.remove(&ordinal).expect("completion belongs to in-flight chunk");
            free_lanes.push(lane);
            ready.insert(ordinal, handle.join()??);
            execution.context().check_cancellation(OPERATION)?;
            while let Some(output) = ready.remove(&committed) {
                // A failed commit aborts the run. Source retry cannot replay
                // input after any of its partials entered coordinator state.
                commit(state, execution, output)?;
                committed += 1;
            }
        }
    }

    fn finish_work<O>(&self, execution: &BoundedExecutionAttempt<'_>, result: Result<O>) -> Result<O> {
        finish_chunk_work(execution, result, self.observation_scope)
    }
}

fn finish_chunk_work<O>(
    execution: &BoundedExecutionAttempt<'_>,
    result: Result<O>,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
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
                super::super::scheduler::record_cleanup_failure(
                    observation_scope,
                    crate::observability::CleanupFailurePhase::TaskExecution,
                    &cleanup,
                );
            }
            Err(error)
        },
    }
}
