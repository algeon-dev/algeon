use super::*;
use crate::runtime::cache::resident_graph::{
    GpuResidentGraphBuildJoin, GpuResidentGraphBuildWait, GpuResidentGraphTransposition,
};

/// Prepares edges, creates a runtime-bound cuGraph handle, enters it, builds
/// the graph, and runs `f` with the cudf runtime, entered context, and graph.
///
/// Centralizes the shared graph-algorithm execution shape and keeps
/// `context.runtime()` as the ingress/execution/egress domain.
///
/// If a future path routes graph results into a different `cudf::Runtime`
/// stream, insert a device-side `cudaEventRecord` + `cudaStreamWaitEvent`
/// dependency before copying — not a host synchronize.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    spec: &crate::plan::GraphAlgorithmSpec,
    resident_graph_key: Option<crate::plan::ResidentGraphKey>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
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
    run_with_cugraph_graph_mode(
        context,
        graph_memory_limit_bytes,
        spec,
        resident_graph_key,
        edges,
        options,
        operation,
        None,
        f,
    )
}

#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph_cache_hit<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    spec: &crate::plan::GraphAlgorithmSpec,
    resident_graph_key: Option<crate::plan::ResidentGraphKey>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    preparation_pin_id: u64,
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
    run_with_cugraph_graph_mode(
        context,
        graph_memory_limit_bytes,
        spec,
        resident_graph_key,
        edges,
        options,
        operation,
        Some(preparation_pin_id),
        f,
    )
}

#[cfg(feature = "cugraph")]
fn run_with_cugraph_graph_mode<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    spec: &crate::plan::GraphAlgorithmSpec,
    resident_graph_key: Option<crate::plan::ResidentGraphKey>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    preparation_pin_id: Option<u64>,
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
    let resident_graph_cache = context.resident_graph_cache();
    if let (Some(cache), Some(resident_graph_key)) = (resident_graph_cache.as_ref(), resident_graph_key) {
        return run_with_cached_cugraph_graph::<T, F>(
            context,
            graph_memory_limit_bytes,
            resident_graph_key,
            edges,
            options,
            operation,
            f,
            std::sync::Arc::clone(cache),
            preparation_pin_id,
        );
    }
    if preparation_pin_id.is_some() {
        return Err(required_resident_graph_cache_hit_error(operation, "cache_or_resident_key_missing"));
    }
    let result =
        run_with_uncached_cugraph_graph::<T, F>(context, graph_memory_limit_bytes, edges, options, operation, f);
    if resident_graph_cache.is_some() {
        let cache_metadata = graph_cache_bypass_metadata_for_missing_key(spec);
        result.map(|result| with_graph_cache_metadata(result, cache_metadata))
    } else {
        result
    }
}

#[cfg(feature = "cugraph")]
fn graph_cache_bypass_metadata_for_missing_key(spec: &crate::plan::GraphAlgorithmSpec) -> &'static str {
    match (&spec.algorithm, &spec.params) {
        (crate::plan::GraphAlgorithm::Bfs, crate::plan::GraphAlgorithmParams::Bfs(params))
            if params.predicates.include_edges.is_some() =>
        {
            "bypass_predicate_include_edges"
        },
        // BFS with stable edge ids normally dispatches through
        // `run_with_cugraph_graph_edge_ids`; this mirrors that label if the
        // routing ever changes.
        (crate::plan::GraphAlgorithm::Bfs, crate::plan::GraphAlgorithmParams::Bfs(_))
            if spec.edge_columns.edge_id.is_some() =>
        {
            "bypass_edge_id_graph"
        },
        (crate::plan::GraphAlgorithm::Bfs, crate::plan::GraphAlgorithmParams::Bfs(_)) => {
            "bypass_no_residency_descriptor"
        },
        (crate::plan::GraphAlgorithm::PageRank, crate::plan::GraphAlgorithmParams::PageRank(_))
            if spec.edge_columns.edge_id.is_some() =>
        {
            "bypass_edge_id_binding"
        },
        (crate::plan::GraphAlgorithm::PageRank, crate::plan::GraphAlgorithmParams::PageRank(_)) => {
            "bypass_no_residency_descriptor"
        },
        _ => "bypass_unkeyed_algorithm",
    }
}

#[cfg(feature = "cugraph")]
fn run_with_cached_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    resident_graph_key: crate::plan::ResidentGraphKey,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    f: F,
    cache: std::sync::Arc<crate::runtime::cache::resident_graph::GpuResidentGraphCache>,
    preparation_pin_id: Option<u64>,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    const BUILD_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    let runtime = cache.companion_runtime_for_context(context, operation)?;
    let budget = graph_execution_budget_with_runtime(
        context,
        graph_memory_limit_bytes,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        operation,
        std::sync::Arc::clone(&runtime),
    )?;
    let handle = cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the resident graph cache runtime",
    ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph resident graph cache execution context",
    ))?;

    let mut f = Some(f);
    let build_wait_deadline = Instant::now().checked_add(BUILD_WAIT_TIMEOUT).unwrap_or_else(Instant::now);
    let build_leader = loop {
        if let Some(mut checkout) = cache.checkout::<T>(&resident_graph_key, context, preparation_pin_id)? {
            let retained_device_bytes = checkout.retained_device_bytes();
            let parked = checkout.take_parked();
            match parked.resume(&entered) {
                Ok(mut graph) => {
                    let result = f.take().expect("graph algorithm closure is consumed once")(
                        runtime.as_ref(),
                        &entered,
                        &mut graph,
                    );
                    return match result {
                        Ok(result) => match graph.park(&entered) {
                            Ok(parked) => {
                                let cache_metadata = match checkout.return_parked(parked) {
                                    Ok(()) => format!("hit;retained_device_bytes={retained_device_bytes}"),
                                    Err(_error) => "hit_return_failed".to_string(),
                                };
                                attach_graph_budget_observation(
                                    Ok(with_graph_cache_metadata(result, &cache_metadata)),
                                    budget.token.as_ref(),
                                    operation,
                                )
                            },
                            Err(_error) => attach_graph_budget_observation(
                                Ok(with_graph_cache_metadata(result, "hit_repark_failed")),
                                budget.token.as_ref(),
                                operation,
                            ),
                        },
                        Err(error) => Err(attach_graph_budget_error_facts(error, budget.token.as_ref(), operation)),
                    };
                },
                Err(_error) => {
                    // A warm-sized grant is never allowed to fall through to
                    // graph construction after losing its reserved entry.
                    if preparation_pin_id.is_some() {
                        return Err(required_resident_graph_cache_hit_error(operation, "parked_graph_resume_failed"));
                    }
                },
            }
        }

        if preparation_pin_id.is_some() {
            return Err(required_resident_graph_cache_hit_error(operation, "exact_runtime_entry_missing"));
        }

        match cache.join_build(&resident_graph_key)? {
            GpuResidentGraphBuildJoin::Leader(leader) => break Some(leader),
            GpuResidentGraphBuildJoin::Follower(follower) => {
                if follower.wait(context.cancellation(), build_wait_deadline, operation)?
                    == GpuResidentGraphBuildWait::DeadlineExpired
                {
                    // Cold-priced queries retain enough budget to build
                    // privately when coordination outlives the bounded wait.
                    // Warm-priced queries return above before joining a build.
                    break None;
                }
            },
        }
    };
    let private_after_wait_timeout = build_leader.is_none();

    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let build_current_bytes_before = graph_budget_current_bytes(budget.token.as_ref());
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(|error| {
            attach_graph_budget_error_facts(
                map_cugraph_error(
                    "nexus_query_engine.graph_algorithm.build_graph",
                    "failed to build cuGraph graph from edge-list table",
                )(error),
                budget.token.as_ref(),
                operation,
            )
        })?;
    let result = f.take().expect("graph algorithm closure is consumed once")(runtime.as_ref(), &entered, &mut graph);
    match result {
        Ok(result) => {
            let retained_device_bytes = graph_retained_device_bytes_upper_bound(
                budget.token.as_ref(),
                build_current_bytes_before,
                &result.frame,
            );
            match graph.park(&entered) {
                Ok(parked) => {
                    let insert = cache.insert_if_absent(resident_graph_key, context, parked, retained_device_bytes)?;
                    let metadata = match insert {
                        crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::Inserted => {
                            if private_after_wait_timeout {
                                format!(
                                    "miss_wait_timeout_private_inserted;retained_device_bytes={retained_device_bytes}"
                                )
                            } else {
                                format!("miss_inserted;retained_device_bytes={retained_device_bytes}")
                            }
                        },
                        crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::AlreadyPresent
                            if private_after_wait_timeout =>
                        {
                            "miss_wait_timeout_private_already_present".to_string()
                        },
                        crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::AlreadyPresent => {
                            "miss_already_present".to_string()
                        },
                        crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::CapacitySkipped => {
                            "miss_capacity_skipped".to_string()
                        },
                        crate::runtime::cache::resident_graph::GpuResidentGraphCacheInsert::CopyLimitSkipped => {
                            "miss_copy_limit_skipped".to_string()
                        },
                    };
                    attach_graph_budget_observation(
                        Ok(with_graph_cache_metadata(result, &metadata)),
                        budget.token.as_ref(),
                        operation,
                    )
                },
                Err(_error) => attach_graph_budget_observation(
                    Ok(with_graph_cache_metadata(result, "miss_park_failed")),
                    budget.token.as_ref(),
                    operation,
                ),
            }
        },
        Err(error) => Err(attach_graph_budget_error_facts(error, budget.token.as_ref(), operation)),
    }
}

#[cfg(feature = "cugraph")]
fn required_resident_graph_cache_hit_error(operation: &'static str, reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.graph_algorithm.required_resident_graph_cache_hit",
        "warm-priced resident graph cache entry is no longer available",
    )
    .with_fact("operation", operation)
    .with_fact("reason", reason)
    .with_fact("retry_action", "readmit_query")
}

#[cfg(all(test, feature = "cugraph"))]
mod resident_graph_cache_error_tests {
    use super::*;

    #[test]
    fn warm_reservation_loss_is_structured_and_retryable() {
        let error = required_resident_graph_cache_hit_error(
            "nexus_query_engine.graph_algorithm.test",
            "exact_runtime_entry_missing",
        );

        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Temporary);
        assert_eq!(error.fact_value("reason"), Some("exact_runtime_entry_missing"));
        assert_eq!(error.fact_value("retry_action"), Some("readmit_query"));
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn with_graph_cache_metadata(
    mut result: GraphAlgorithmExecutionResult,
    cache_metadata: &str,
) -> GraphAlgorithmExecutionResult {
    if result.algorithm_metadata.is_empty() {
        result.algorithm_metadata = format!("graph_cache={cache_metadata}");
    } else {
        result.algorithm_metadata.push_str(",graph_cache=");
        result.algorithm_metadata.push_str(cache_metadata);
    }
    result
}

#[cfg(feature = "cugraph")]
pub(super) fn resident_graph_key_for_input<T>(
    input: &GpuDataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Option<crate::plan::ResidentGraphKey>
where
    T: GpuResidentGraphTransposition,
{
    input.graph_residency().and_then(|descriptor| {
        let key = crate::plan::planned_resident_graph_key(descriptor, spec)?;
        (key.orientation == T::orientation()).then_some(key)
    })
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn resident_graph_cache_preparation_pin(
    context: &ExecutionContext,
    input: &GpuDataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<Option<crate::runtime::cache::resident_graph::GpuResidentGraphPreparationPin>> {
    let Some(cache) = context.resident_graph_cache() else {
        context.reject_unmatched_reserved_resident_graph_price("resident_graph_cache_missing")?;
        return Ok(None);
    };
    let Some(descriptor) = input.graph_residency() else {
        context.reject_unmatched_reserved_resident_graph_price("graph_residency_descriptor_missing")?;
        return Ok(None);
    };
    let Some(key) = crate::plan::planned_resident_graph_key(descriptor, spec) else {
        context.reject_unmatched_reserved_resident_graph_price("resident_graph_key_unavailable")?;
        return Ok(None);
    };
    if let Some(pin) = context.take_reserved_resident_graph_pin(&key)? {
        return Ok(Some(pin));
    }
    cache.try_pin_available_key_for_context(&key, context)
}
