#[cfg(feature = "cugraph")]
use cudf::ErrorKind;
#[cfg(feature = "cugraph")]
use cugraph::ResourceHandle;
#[cfg(feature = "cugraph")]
use cugraph::cudf_interop::TableGraphOptions;
#[cfg(feature = "cugraph")]
use cugraph::cudf_interop::graph_from_prepared_table_export;
#[cfg(feature = "cugraph")]
use query_planning::plan::ResidentGraphKey;
#[cfg(feature = "cugraph")]
use query_runtime::runtime::cache::resident_graph::GpuResidentGraphCache;
#[cfg(feature = "cugraph")]
use query_runtime::runtime::cache::resident_graph::GpuResidentGraphCacheInsert;
#[cfg(feature = "cugraph")]
use query_runtime::runtime::cache::resident_graph::GpuResidentGraphCheckout;
#[cfg(feature = "cugraph")]
use query_runtime::runtime::cache::resident_graph::GpuResidentGraphTransposition;

use super::{
    DataFrame, ExecutionContext, GraphAlgorithmExecutionResult, QueryError, QueryErrorCode, Result, cudf_query_error,
    graph_build_range, map_cugraph_error, run_with_uncached_cugraph_graph,
};

#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph<T, F>(
    context: &ExecutionContext,
    edges: DataFrame,
    options: TableGraphOptions,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: cugraph::Transposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    run_with_uncached_cugraph_graph(context, edges, options, operation, f)
}

/// Executes an eligible graph through the admitted attempt domain.
///
/// A parked cuGraph graph keeps its storage in the cache runtime that owns
/// its native allocations, but a hit resumes it under the attempt runtime:
/// the attempt stream waits on the park event, and algorithm workspace,
/// auxiliary streams, and outputs allocate through the attempt binding, so
/// they charge to the borrower's grant; never to the cache share. Only the
/// final dataframe is retained into the attempt domain, and the graph is
/// re-parked before the checkout returns to the cache.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_accounted_resident_cugraph_graph<T, F>(
    context: &ExecutionContext,
    edges: DataFrame,
    options: TableGraphOptions,
    resident_key: Option<ResidentGraphKey>,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let Some(resident_key) = resident_key else {
        return run_with_uncached_cugraph_graph(context, edges, options, operation, f);
    };
    let Some(cache) = context.resident_graph_cache() else {
        return run_with_uncached_cugraph_graph(context, edges, options, operation, f);
    };
    let attempt_runtime = context.runtime()?;
    if cache.device() != attempt_runtime.device() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.resident_cache",
            "resident graph cache device does not match the admitted attempt runtime",
        )
        .with_fact("reason", "resident_graph_cache_attempt_device_mismatch")
        .with_fact("cache_device_ordinal", cache.device().ordinal())
        .with_fact("attempt_device_ordinal", attempt_runtime.device().ordinal()));
    }
    context.check_cancellation(operation)?;

    if let Some(checkout) = cache.checkout::<T>(&resident_key, None)? {
        // A cache hit needs no input edges during algorithm workspace allocation.
        drop(edges);
        return run_checked_out_resident_graph(context, &attempt_runtime, cache, checkout, operation, f);
    }

    // Reservation is derived from the same snapshot projected by sealed
    // reports, admission TSV, and backend metrics. It is deliberately
    // conservative: the transaction shrinks to the
    // post-park graph-scope usage before it can become an evictable entry.
    let reservation_bytes = cache.available_reservation_bytes()?;
    if reservation_bytes == 0 {
        return run_with_uncached_cugraph_graph(context, edges, options, operation, f);
    }
    let Some(mut transaction) = cache.begin_insert(reservation_bytes)? else {
        return run_with_uncached_cugraph_graph(context, edges, options, operation, f);
    };
    // Cache insertion uses a child allocation scope, so its allocator failures
    // are not present in the admitted attempt domain's event ring.
    let mut cache_scope_correlation = context.begin_cache_scope_allocator_correlation(transaction.allocation_scope());
    let cache_runtime = transaction.runtime();
    let cached_edges = match edges.copy_into_runtime(cache_runtime.as_ref()) {
        Ok(edges) => edges,
        Err(error) if error.kind() == ErrorKind::ResourceExhausted => {
            // The runtime can retain a current-resource route, so release it before scope teardown proof.
            drop(cache_runtime);
            if let Some(correlation) = cache_scope_correlation.take() {
                correlation.finish();
            }
            transaction.abort_after_candidate_destroyed()?;
            return run_with_uncached_cugraph_graph(context, edges, options, operation, f);
        },
        Err(error) => {
            let primary = cudf_query_error(
                "query_engine.graph_algorithm.resident_cache.copy_edges",
                "failed to copy graph edges into the resident-cache allocation domain",
                error,
            );
            // The runtime can retain a current-resource route, so release it before scope teardown proof.
            drop(cache_runtime);
            // Keep an unsuccessful cache-scope interval fail-closed.
            transaction.abort_after_candidate_destroyed()?;
            return Err(primary);
        },
    };
    // The successful cache copy replaces the attempt input before graph construction and algorithm workspace.
    drop(edges);
    let handle = match ResourceHandle::for_runtime(std::sync::Arc::clone(&cache_runtime)) {
        Ok(handle) => handle,
        Err(error) => {
            let primary = map_cugraph_error(
                "query_engine.graph_algorithm.resident_cache.resource_handle",
                "failed to bind cuGraph to the resident-cache runtime",
            )(error);
            // The edge frame and runtime can own scope allocations/routes until their destructors run.
            drop(cached_edges);
            drop(cache_runtime);
            // Keep an unsuccessful cache-scope interval fail-closed.
            transaction.abort_after_candidate_destroyed()?;
            return Err(primary);
        },
    };
    let entered = match handle.enter(operation) {
        Ok(entered) => entered,
        Err(error) => {
            let primary = map_cugraph_error(
                "query_engine.graph_algorithm.resident_cache.enter",
                "failed to enter the resident-cache cuGraph runtime",
            )(error);
            // `ResourceHandle::enter` may retain a stream route through its
            // result temporary. Ordinary reverse drop releases every local
            // owner before the transaction's fenced fail-closed destructor
            // decides whether its prospective precharge can return.
            drop(cached_edges);
            drop(cache_runtime);
            // Keep an unsuccessful cache-scope interval fail-closed.
            transaction.abort_after_candidate_destroyed()?;
            return Err(primary);
        },
    };
    let build_result = (|| {
        let mut graph = {
            let construction_edges = cached_edges;
            let export = construction_edges.try_prepare_device_table_export().map_err(|error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "query_engine.graph_algorithm.resident_cache.prepare_table_export",
                    "failed to prepare resident-cache graph edge-list export",
                )
                .with_source(error)
            })?;
            let _graph_build_range = graph_build_range(operation);
            let mut graph =
                graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(map_cugraph_error(
                    "query_engine.graph_algorithm.resident_cache.build_graph",
                    "failed to build cuGraph graph in the resident-cache allocation domain",
                ))?;
            graph.synchronize_input_reads().map_err(map_cugraph_error(
                "query_engine.graph_algorithm.resident_cache.complete_input_reads",
                "failed to complete resident-cache graph construction input reads",
            ))?;
            graph
        };
        let outcome = f(cache_runtime.as_ref(), &entered, &mut graph)
            .and_then(|outcome| copy_graph_result_into_attempt(outcome, &attempt_runtime));
        let parked = graph.park(&entered).map_err(map_cugraph_error(
            "query_engine.graph_algorithm.resident_cache.park_graph",
            "failed to park a resident-cache cuGraph graph",
        ))?;
        Ok::<_, QueryError>((outcome, parked))
    })();
    // The handle can retain scope allocations until after native graph construction unwinds.
    drop(entered);
    drop(handle);

    let (outcome, parked) = match build_result {
        Ok(result) => result,
        Err(primary) => {
            // The runtime can retain a current-resource route, so release it before scope teardown proof.
            drop(cache_runtime);
            if let Some(correlation) = cache_scope_correlation.take() {
                correlation.finish();
            }
            transaction.abort_after_candidate_destroyed()?;
            return Err(primary);
        },
    };
    if let Err(primary) = outcome {
        // Destroy the parked native graph before its scope may return the ledger precharge.
        drop(parked);
        // The runtime can retain a current-resource route, so release it before scope teardown proof.
        drop(cache_runtime);
        if let Some(correlation) = cache_scope_correlation.take() {
            correlation.finish();
        }
        transaction.abort_after_candidate_destroyed()?;
        return Err(primary);
    }
    let shrunk = match transaction.shrink_to_scope_usage() {
        Ok(shrunk) => shrunk,
        Err(error) => {
            // Destroy the parked graph and runtime before fail-closed candidate teardown.
            drop(parked);
            // The temporary runtime can retain the scope's current-resource route during teardown.
            drop(cache_runtime);
            let _ = transaction.abort_after_candidate_destroyed();
            return Err(error);
        },
    };
    let cache_scope_peak_bytes = match transaction.scope_peak_bytes() {
        Ok(peak) => peak,
        Err(error) => {
            // Destroy the parked graph before its allocation scope may be retired.
            drop(parked);
            // The temporary runtime can retain the scope's current-resource route during teardown.
            drop(cache_runtime);
            let _ = transaction.abort_after_candidate_destroyed();
            return Err(error);
        },
    };
    if !shrunk {
        // The graph exceeded its conservative reservation and must die before scope teardown.
        drop(parked);
        // The runtime can retain a current-resource route, so release it before scope teardown proof.
        drop(cache_runtime);
        transaction.abort_after_candidate_destroyed()?;
        return outcome.map(|mut outcome| {
            outcome.cache_scope_peak_bytes = Some(cache_scope_peak_bytes);
            outcome
        });
    }
    // The entry retains its own scope runtime; this temporary clone must not delay commit/abort ownership.
    drop(cache_runtime);
    // A successful reconciliation closes a no-event window without poisoning
    // the attempt's collector before the parked scope becomes reusable.
    if let Some(correlation) = cache_scope_correlation.take() {
        correlation.finish();
    }
    match transaction.insert_if_absent(resident_key, parked)? {
        GpuResidentGraphCacheInsert::Inserted
        | GpuResidentGraphCacheInsert::AlreadyPresent
        | GpuResidentGraphCacheInsert::Closed => outcome.map(|mut outcome| {
            outcome.cache_scope_peak_bytes = Some(cache_scope_peak_bytes);
            outcome
        }),
    }
}

#[cfg(feature = "cugraph")]
fn run_checked_out_resident_graph<T, F>(
    context: &ExecutionContext,
    attempt_runtime: &cudf::Runtime,
    cache: GpuResidentGraphCache,
    mut checkout: GpuResidentGraphCheckout<T>,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    context.check_cancellation(operation)?;
    // A resident checkout is exclusive, so its scope ring can be bounded to
    // this attempt until the graph is returned to the cache.
    let mut cache_scope_correlation = context.begin_cache_scope_allocator_correlation(checkout.allocation_scope());
    // Scope this work so every attempt-runtime owner (including the runtime
    // clone on early failures) has dropped before the scope fence below. A
    // failed re-park consumes and drops the graph, but cannot return a
    // checkout; its correlation guard must therefore discard rather than
    // claim terminal allocator evidence from a drop path.
    //
    // Execution enters on the attempt binding, so workspace and outputs
    // charge to the borrower's grant: an undersized grant fails closed in the
    // attempt domain instead of borrowing cache capacity.
    let execution = {
        let attempt_handle_runtime = std::sync::Arc::new(attempt_runtime.clone());
        (|| {
            let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&attempt_handle_runtime)).map_err(
                map_cugraph_error(
                    "query_engine.graph_algorithm.resident_cache.resource_handle",
                    "failed to bind cuGraph to the admitted attempt runtime",
                ),
            )?;
            let entered = handle.enter(operation).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.resident_cache.enter",
                "failed to enter the admitted attempt cuGraph runtime",
            ))?;
            let parked = checkout.take_parked();
            let mut graph = parked.resume_under(&entered).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.resident_cache.resume_graph",
                "failed to resume a resident-cache cuGraph graph under the attempt runtime",
            ))?;
            let outcome = f(attempt_runtime, &entered, &mut graph)
                .and_then(|outcome| copy_graph_result_into_attempt(outcome, attempt_runtime));
            match graph.park_under(&entered).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.resident_cache.repark_graph",
                "failed to re-park a resident-cache cuGraph graph",
            )) {
                Ok(reparked) => Ok((outcome, reparked)),
                Err(cache_error) => outcome.and(Err(cache_error)),
            }
        })()
    };

    let (outcome, reparked) = match execution {
        Ok(execution) => execution,
        Err(error) => {
            // The failed path has no parked graph to return, so it must never
            // close an exact cache-scope window. It still fences the private
            // stream before failing closed.
            if let Err(scope_error) = checkout.synchronize_scope() {
                let cache_error = cache.close().err();
                let close_error = cache.close_admission_for_scope_retirement_fence().err();
                return Err(cache_error.or(close_error).unwrap_or(scope_error));
            }
            return Err(error);
        },
    };

    if let Err(scope_error) = checkout.synchronize_scope() {
        // A failed fence cannot establish an exact upper bound. Leave the
        // guard unclosed so its drop discards evidence, retain execution
        // capacity, and fail admission closed. Closing the graph cache before
        // consuming a re-parked graph prevents an already-admitted attempt
        // from reusing an unretired scope.
        let cache_quarantine_error = quarantine_untrusted_resident_graph(&cache, checkout, reparked).err();
        let error = match cache.close_admission_for_scope_retirement_fence() {
            Ok(()) => scope_error,
            Err(close_error) => close_error,
        };
        return Err(cache_quarantine_error.unwrap_or(error));
    }

    let cache_scope_peak_bytes = match checkout.scope_peak_bytes() {
        Ok(peak) => peak,
        Err(scope_error) => {
            let cache_quarantine_error = quarantine_untrusted_resident_graph(&cache, checkout, reparked).err();
            let error = match cache.close_admission_for_scope_invariant() {
                Ok(()) => scope_error,
                Err(close_error) => close_error,
            };
            return Err(cache_quarantine_error.unwrap_or(error));
        },
    };

    match finish_after_successful_scope_fence(
        Ok(()),
        checkout.verify_reparked_scope_for_return(),
        &mut cache_scope_correlation,
        |correlation| correlation.finish(),
    ) {
        Ok(()) => {
            // A successful fence plus retained-scope health proof establishes
            // the private-ring upper bound after every cache temporary has
            // retired. Close before the checked-out graph is made available.
        },
        Err(scope_error) => {
            // An unhealthy or baseline-mismatched re-park cannot return to
            // the cache. Leave the guard unclosed so a failed proof never
            // becomes exact allocator evidence.
            let cache_quarantine_error = quarantine_untrusted_resident_graph(&cache, checkout, reparked).err();
            let error = match cache.close_admission_for_scope_invariant() {
                Ok(()) => scope_error,
                Err(close_error) => close_error,
            };
            return Err(cache_quarantine_error.unwrap_or(error));
        },
    }

    let return_result = checkout.return_parked(reparked);
    // The closure has dropped the attempt-runtime handle, entered context,
    // graph result, and every attempt-domain temporary allocation; the
    // checkout scope fence above already proved the cache storage charge.
    match outcome {
        Err(primary) => {
            let _ = return_result;
            // The scope fence, health proof, and re-park all completed
            // before this primary graph error is returned.
            Err(primary.with_graph_cache_scope_peak_bytes(cache_scope_peak_bytes))
        },
        Ok(outcome) => return_result.map(|()| {
            let mut outcome = outcome;
            outcome.cache_scope_peak_bytes = Some(cache_scope_peak_bytes);
            outcome
        }),
    }
}

/// Runs one cache-scope finalizer only after the stream fence and re-park
/// proof establish a completed, healthy execution interval. The caller keeps
/// an unrun finalizer on either failure so its destructor conservatively
/// discards evidence.
#[cfg(feature = "cugraph")]
fn finish_after_successful_scope_fence<T>(
    scope_fence: Result<()>,
    repark_health: Result<()>,
    finalizer: &mut Option<T>,
    finish: impl FnOnce(T),
) -> Result<()> {
    scope_fence?;
    repark_health?;
    if let Some(finalizer) = finalizer.take() {
        finish(finalizer);
    }
    Ok(())
}

/// Makes a checked-out graph permanently unavailable before destroying it.
/// The cache must close before consuming `parked`: returning an unsafe scope
/// while open would make it available to an attempt admitted before the
/// ledger's fail-closed transition.
#[cfg(feature = "cugraph")]
fn quarantine_untrusted_resident_graph<T>(
    cache: &GpuResidentGraphCache,
    checkout: GpuResidentGraphCheckout<T>,
    parked: cugraph::ParkedGraph<T>,
) -> Result<()>
where
    T: GpuResidentGraphTransposition,
{
    cache.close()?;
    checkout.return_parked(parked)
}

#[cfg(feature = "cugraph")]
fn copy_graph_result_into_attempt(
    mut outcome: GraphAlgorithmExecutionResult,
    attempt_runtime: &cudf::Runtime,
) -> Result<GraphAlgorithmExecutionResult> {
    outcome.frame = outcome.frame.copy_into_runtime(attempt_runtime).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.resident_cache.copy_result",
            "failed to copy resident-cache graph output into the attempt allocation domain",
            error,
        )
    })?;
    Ok(outcome)
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use super::*;

    #[test]
    fn scope_correlation_finalizer_runs_after_a_successful_fence() {
        let mut finalizer = Some(());
        let mut finished = 0;

        finish_after_successful_scope_fence(Ok(()), Ok(()), &mut finalizer, |_| finished += 1)
            .expect("a successful scope fence must close its correlation window");

        assert_eq!(finished, 1);
        assert!(finalizer.is_none());
    }

    #[test]
    fn scope_correlation_finalizer_is_not_run_after_a_failed_fence() {
        let mut finalizer = Some(());
        let mut finished = 0;
        let scope_error = QueryError::new(
            QueryErrorCode::CudfPersistentFailure,
            "query_engine.graph_algorithm.resident_cache.test",
            "scope fence failed",
        );

        assert!(
            finish_after_successful_scope_fence(Err(scope_error), Ok(()), &mut finalizer, |_| finished += 1).is_err()
        );

        assert_eq!(finished, 0);
        assert!(finalizer.is_some());
    }

    #[test]
    fn scope_correlation_finalizer_is_not_run_before_repark_health_is_proven() {
        let mut finalizer = Some(());
        let mut finished = 0;
        let health_error = QueryError::new(
            QueryErrorCode::CudfPersistentFailure,
            "query_engine.graph_algorithm.resident_cache.test",
            "re-park health proof failed",
        );

        assert!(
            finish_after_successful_scope_fence(Ok(()), Err(health_error), &mut finalizer, |_| finished += 1).is_err()
        );

        assert_eq!(finished, 0);
        assert!(finalizer.is_some());
    }
}
