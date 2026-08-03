use super::*;

#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
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
    run_with_uncached_cugraph_graph(context, graph_memory_limit_bytes, edges, options, operation, f)
}

/// Executes an eligible graph through the per-device resident-cache domain.
///
/// A parked cuGraph graph is bound to the cache runtime that owns its native
/// allocations, so a hit must execute on that runtime and copy only its final
/// dataframe into the admitted attempt domain. The cache runtime's temporary
/// results are dropped before its checkout pin is released; consequently the
/// ledger transaction continues to describe retained graph bytes only.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_accounted_resident_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    resident_key: Option<crate::plan::ResidentGraphKey>,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: crate::runtime::cache::resident_graph::GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let Some(resident_key) = resident_key else {
        return run_with_uncached_cugraph_graph(context, graph_memory_limit_bytes, edges, options, operation, f);
    };
    let Some(cache) = context.resident_graph_cache() else {
        return run_with_uncached_cugraph_graph(context, graph_memory_limit_bytes, edges, options, operation, f);
    };
    let attempt_runtime = context.runtime()?;
    if cache.runtime().device() != attempt_runtime.device() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.resident_cache",
            "resident graph cache device does not match the admitted attempt runtime",
        )
        .with_fact("reason", "resident_graph_cache_attempt_device_mismatch")
        .with_fact("cache_device_ordinal", cache.runtime().device().ordinal())
        .with_fact("attempt_device_ordinal", attempt_runtime.device().ordinal()));
    }
    context.check_cancellation(operation)?;

    if let Some(checkout) = cache.checkout::<T>(&resident_key, None)? {
        return run_checked_out_resident_graph(context, &attempt_runtime, cache, checkout, operation, f);
    }

    // Reservation is derived from the same snapshot that projects P3/P4/TSV
    // state. It is deliberately conservative: the transaction shrinks to the
    // post-park physical delta before it can become an evictable entry.
    let reservation_bytes = cache.available_reservation_bytes()?;
    if reservation_bytes == 0 {
        return run_with_uncached_cugraph_graph(context, graph_memory_limit_bytes, edges, options, operation, f);
    }
    let Some(mut transaction) = cache.begin_insert(reservation_bytes)? else {
        return run_with_uncached_cugraph_graph(context, graph_memory_limit_bytes, edges, options, operation, f);
    };
    let before = cache.allocation_usage()?;
    let cache_runtime = std::sync::Arc::clone(transaction.runtime());
    let cached_edges = match edges.copy_into_runtime(cache_runtime.as_ref()) {
        Ok(edges) => edges,
        Err(error) if error.kind() == cudf::ErrorKind::ResourceExhausted => {
            drop(transaction);
            return run_with_uncached_cugraph_graph(context, graph_memory_limit_bytes, edges, options, operation, f);
        },
        Err(error) => {
            return Err(QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.resident_cache.copy_edges",
                "failed to copy graph edges into the resident-cache allocation domain",
                error,
            ));
        },
    };
    let handle =
        cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&cache_runtime)).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resident_cache.resource_handle",
            "failed to bind cuGraph to the resident-cache runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resident_cache.enter",
        "failed to enter the resident-cache cuGraph runtime",
    ))?;
    let build_result = (|| {
        let export = cached_edges.try_prepare_device_table_export().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.graph_algorithm.resident_cache.prepare_table_export",
                "failed to prepare resident-cache graph edge-list export",
            )
            .with_source(error)
        })?;
        let _graph_build_range = graph_build_range(operation);
        let mut graph = cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options)
            .map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.resident_cache.build_graph",
                "failed to build cuGraph graph in the resident-cache allocation domain",
            ))?;
        let outcome = f(cache_runtime.as_ref(), &entered, &mut graph)
            .and_then(|outcome| copy_graph_result_into_attempt(outcome, &attempt_runtime));
        let parked = graph.park(&entered).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resident_cache.park_graph",
            "failed to park a resident-cache cuGraph graph",
        ))?;
        Ok::<_, QueryError>((outcome, parked))
    })();
    drop(entered);
    drop(handle);
    drop(cached_edges);

    let (outcome, parked) = build_result?;
    if let Err(primary) = outcome {
        drop(parked);
        return Err(primary);
    }
    if !transaction.shrink_to_domain_delta(before)? {
        drop(parked);
        return outcome;
    }
    match transaction.insert_if_absent(resident_key, parked)? {
        crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::Inserted
        | crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::AlreadyPresent
        | crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::Closed
        | crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::CapacitySkipped => outcome,
    }
}

#[cfg(feature = "cugraph")]
fn run_checked_out_resident_graph<T, F>(
    context: &ExecutionContext,
    attempt_runtime: &cudf::Runtime,
    cache: crate::runtime::cache::resident_graph::GpuResidentGraphCache,
    mut checkout: crate::runtime::cache::resident_graph::GpuResidentGraphCheckout<T>,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: crate::runtime::cache::resident_graph::GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    context.check_cancellation(operation)?;
    let cache_runtime = std::sync::Arc::clone(cache.runtime());
    let handle =
        cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&cache_runtime)).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resident_cache.resource_handle",
            "failed to bind cuGraph to the resident-cache runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resident_cache.enter",
        "failed to enter the resident-cache cuGraph runtime",
    ))?;
    let parked = checkout.take_parked();
    let mut graph = parked.resume(&entered).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resident_cache.resume_graph",
        "failed to resume a resident-cache cuGraph graph",
    ))?;
    let outcome = f(cache_runtime.as_ref(), &entered, &mut graph)
        .and_then(|outcome| copy_graph_result_into_attempt(outcome, attempt_runtime));
    let reparked = graph.park(&entered).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resident_cache.repark_graph",
        "failed to re-park a resident-cache cuGraph graph",
    ));
    drop(entered);
    drop(handle);
    match reparked {
        Ok(reparked) => {
            let return_result = checkout.return_parked(reparked);
            match outcome {
                Err(primary) => {
                    let _ = return_result;
                    Err(primary)
                },
                Ok(outcome) => return_result.map(|()| outcome),
            }
        },
        Err(cache_error) => outcome.and(Err(cache_error)),
    }
}

#[cfg(feature = "cugraph")]
fn copy_graph_result_into_attempt(
    mut outcome: GraphAlgorithmExecutionResult,
    attempt_runtime: &cudf::Runtime,
) -> Result<GraphAlgorithmExecutionResult> {
    outcome.frame = outcome.frame.copy_into_runtime(attempt_runtime).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.resident_cache.copy_result",
            "failed to copy resident-cache graph output into the attempt allocation domain",
            error,
        )
    })?;
    Ok(outcome)
}
