use super::*;

#[cfg(feature = "cugraph")]
thread_local! {
    // Graph statistics probing is synchronous today. Keeping this test hook
    // thread-local prevents one integration test from consuming another test's
    // forced optional-resource exhaustion.
    static FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) struct RuntimeGraphStatisticsProbe {
    pub(in crate::exec) vertex_estimate: crate::graph_memory::GraphVertexEstimate,
    pub(in crate::exec) statistics: crate::graph_memory::GraphStatistics,
}

#[cfg(all(any(test, feature = "test-hooks"), feature = "cugraph"))]
pub fn force_next_graph_statistics_optional_resource_exhausted_for_test() {
    FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED.with(|force| force.set(true));
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn runtime_graph_statistics_probe_with_budget(
    context: &ExecutionContext,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    budget_limit_bytes: u64,
    metrics: &mut QueryMetrics,
) -> Result<RuntimeGraphStatisticsProbe> {
    let operation = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";
    let budget = graph_execution_budget(
        context,
        Some(budget_limit_bytes),
        crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
        operation,
    )?;
    let probe = runtime_graph_statistics_probe_in_runtime(budget.runtime.as_ref(), frame, spec);
    match probe {
        Ok(probe) => {
            if let Some(observation) = graph_budget_observation(&budget.token, operation, "reservation_enforced") {
                observation.record_with_boundary(
                    metrics,
                    crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
                    Some("graph_vertex_estimate_probe"),
                );
            }
            Ok(probe)
        },
        Err(error) => {
            let error = attach_graph_budget_error_facts(error, &budget.token, operation);
            record_graph_budget_observation_from_error_with_boundary(
                metrics,
                &error,
                crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
                Some("graph_vertex_estimate_probe"),
            );
            Err(error)
        },
    }
}

#[cfg(feature = "cugraph")]
fn runtime_graph_statistics_probe_in_runtime(
    runtime: &cudf::Runtime,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<RuntimeGraphStatisticsProbe> {
    let operation = GRAPH_STATISTICS_OPERATION;
    let source = frame
        .series(&spec.edge_columns.source)
        .map_err(|error| QueryError::from_cudf(operation, "failed to read graph source column", error))?;
    let destination = frame
        .series(&spec.edge_columns.destination)
        .map_err(|error| QueryError::from_cudf(operation, "failed to read graph destination column", error))?;
    let source_dtype = source
        .dtype()
        .map_err(|error| QueryError::from_cudf(operation, "failed to inspect graph source dtype", error))?;
    let destination_dtype = destination
        .dtype()
        .map_err(|error| QueryError::from_cudf(operation, "failed to inspect graph destination dtype", error))?;
    if source_dtype.id() != destination_dtype.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            operation,
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_dtype:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_dtype:?}")));
    }
    let vertex_width_bytes = dtype_width_bytes(source_dtype)?;
    let device_total_memory_bytes = runtime.device().available_memory().ok().map(|memory| memory.total_bytes);
    let device_multiprocessor_count = runtime.device().multiprocessor_count().ok().filter(|count| *count > 0);
    let source = source.copy_into_runtime(runtime).map_err(|error| {
        QueryError::from_cudf(operation, "failed to place graph source column in budgeted runtime", error)
    })?;
    let destination = destination.copy_into_runtime(runtime).map_err(|error| {
        QueryError::from_cudf(operation, "failed to place graph destination column in budgeted runtime", error)
    })?;
    let ctx = graph_statistics_execution_overrides(runtime);
    let count = cudf::Series::concat_in(&[&source, &destination], &ctx)
        .and_then(|series| series.nunique_in(&ctx))
        .map_err(|error| QueryError::from_cudf(operation, "failed to count distinct graph endpoints", error))?;
    let count = u64::try_from(count).map_err(|error| {
        QueryError::new(QueryErrorCode::InvalidPlan, operation, "runtime graph vertex estimate does not fit u64")
            .with_source(error)
    })?;
    let vertex_estimate = crate::graph_memory::GraphVertexEstimate::runtime_exact_distinct(count);
    let statistics = crate::graph_memory::GraphStatistics::runtime_exact_vertices(count);

    let statistics = match graph_statistics_optional_algorithm_evidence(
        &ctx,
        source,
        destination,
        spec,
        vertex_width_bytes,
        device_total_memory_bytes,
        device_multiprocessor_count,
        statistics,
    ) {
        Ok(statistics) => statistics,
        Err(error) if error.code() == QueryErrorCode::CudfResourceExhausted => statistics,
        Err(error) => return Err(error),
    };

    Ok(RuntimeGraphStatisticsProbe { vertex_estimate, statistics })
}

#[cfg(feature = "cugraph")]
#[expect(clippy::too_many_arguments, reason = "graph algorithm admission inputs are distinct proof values")]
fn graph_statistics_optional_algorithm_evidence(
    ctx: &ExecutionOverrides<'_>,
    source: cudf::Series,
    destination: cudf::Series,
    spec: &crate::plan::GraphAlgorithmSpec,
    vertex_width_bytes: u64,
    device_total_memory_bytes: Option<u64>,
    device_multiprocessor_count: Option<u32>,
    mut statistics: crate::graph_memory::GraphStatistics,
) -> Result<crate::graph_memory::GraphStatistics> {
    let needs_sum_out_degree = graph_statistics_needs_sum_out_degree(spec.algorithm);
    let needs_triangle_degree = graph_statistics_needs_triangle_degree(spec.algorithm);
    if !needs_sum_out_degree && !needs_triangle_degree {
        return Ok(statistics);
    }
    if FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED.with(|force| force.replace(false)) {
        return Err(QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            GRAPH_STATISTICS_OPERATION,
            "forced graph optional statistics resource exhaustion for test",
        ));
    }

    let edge_frame =
        DataFrame::from_series([(GRAPH_STATS_SRC, source), (GRAPH_STATS_DST, destination)]).map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to build graph statistics edge frame", error)
        })?;
    if edge_frame.num_rows() == 0 {
        if needs_sum_out_degree {
            statistics = statistics.with_sum_out_degree(0);
        }
        if needs_triangle_degree {
            statistics = statistics.with_degree_distribution(0, 0).with_triangle_min_degree(0, 0, 0);
        }
        return Ok(statistics);
    }

    if needs_sum_out_degree {
        let out_degrees = graph_statistics_out_degrees(&edge_frame, ctx)?;
        let sum_out_degree = graph_statistics_betweenness_resident_out_degree_sum(
            &out_degrees,
            spec,
            statistics.exact_vertices,
            vertex_width_bytes,
            device_total_memory_bytes,
            ctx,
        )?;
        statistics = statistics.with_sum_out_degree(sum_out_degree);
    }
    if needs_triangle_degree {
        let triangle_edges = graph_statistics_triangle_undirected_edges(&edge_frame, ctx)?;
        drop(edge_frame);
        if triangle_edges.num_rows() == 0 {
            statistics = statistics.with_degree_distribution(0, 0).with_triangle_min_degree(0, 0, 0);
            return Ok(statistics);
        }
        let undirected_degrees = graph_statistics_undirected_degrees(&triangle_edges, ctx)?;
        let degree = undirected_degrees.series(GRAPH_STATS_OUT_DEGREE).map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph undirected-degree column", error)
        })?;
        let degree_max = graph_statistics_series_max_u64(&degree, ctx, GRAPH_STATS_OUT_DEGREE)?.unwrap_or(0);
        let degree_p99 = graph_statistics_series_p99_u64(&degree, ctx, GRAPH_STATS_OUT_DEGREE)?;
        let (sum_min_degree_oriented, oriented_edge_count, peak_chunk_min_degree_sum) =
            graph_statistics_triangle_min_degree(
                &triangle_edges,
                &undirected_degrees,
                ctx,
                device_multiprocessor_count,
            )?;
        statistics = statistics.with_degree_distribution(degree_max, degree_p99).with_triangle_min_degree(
            sum_min_degree_oriented,
            oriented_edge_count,
            peak_chunk_min_degree_sum,
        );
    }

    Ok(statistics)
}

#[cfg(feature = "cugraph")]
const GRAPH_STATISTICS_OPERATION: &str = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_SRC: &str = "__graph_stats_src";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_DST: &str = "__graph_stats_dst";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_CANONICAL_SRC: &str = "__graph_stats_canonical_src";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_CANONICAL_DST: &str = "__graph_stats_canonical_dst";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_VERTEX: &str = "__graph_stats_vertex";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ENDPOINT: &str = "__graph_stats_endpoint";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_OUT_DEGREE: &str = "__graph_stats_out_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_SRC_DEGREE: &str = "__graph_stats_src_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_DST_DEGREE: &str = "__graph_stats_dst_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_SRC: &str = "__graph_stats_oriented_src";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_DST: &str = "__graph_stats_oriented_dst";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_DEGREE: &str = "__graph_stats_oriented_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_SRC_DEGREE: &str = "__graph_stats_oriented_src_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_DST_DEGREE: &str = "__graph_stats_oriented_dst_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_EDGE_MIN_DEGREE: &str = "__graph_stats_edge_min_degree";
#[cfg(feature = "cugraph")]
fn graph_statistics_execution_overrides(runtime: &cudf::Runtime) -> ExecutionOverrides<'_> {
    ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource())
}

#[cfg(feature = "cugraph")]
const fn graph_statistics_needs_sum_out_degree(algorithm: crate::plan::GraphAlgorithm) -> bool {
    matches!(
        algorithm,
        crate::plan::GraphAlgorithm::BetweennessCentrality | crate::plan::GraphAlgorithm::EdgeBetweennessCentrality
    )
}

#[cfg(feature = "cugraph")]
const fn graph_statistics_needs_triangle_degree(algorithm: crate::plan::GraphAlgorithm) -> bool {
    matches!(algorithm, crate::plan::GraphAlgorithm::TriangleCountAll)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_out_degrees(edge_frame: &DataFrame, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
    let out_degrees = edge_frame
        .groupby([GRAPH_STATS_SRC])
        .with_execution(ctx)
        .agg(&[CudfAggSpec::new(GRAPH_STATS_DST, GroupByAgg::Count { include_nulls: false })
            .with_alias(GRAPH_STATS_OUT_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute graph out-degrees", error)
        })?;
    graph_statistics_cast_column_to_u64(&out_degrees, GRAPH_STATS_OUT_DEGREE, ctx)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_betweenness_resident_out_degree_sum(
    out_degrees: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    exact_vertices: u64,
    vertex_width_bytes: u64,
    device_total_memory_bytes: Option<u64>,
    ctx: &ExecutionOverrides<'_>,
) -> Result<u64> {
    let resident_sources = graph_statistics_betweenness_resident_sources(
        spec,
        exact_vertices,
        vertex_width_bytes,
        device_total_memory_bytes,
    )?;
    if resident_sources == 0 || out_degrees.num_rows() == 0 {
        return Ok(0);
    }
    let out_degree = out_degrees.series(GRAPH_STATS_OUT_DEGREE).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph out-degree column", error)
    })?;
    if resident_sources >= out_degrees.num_rows() as u64 {
        return graph_statistics_series_sum_u64(&out_degree, ctx, GRAPH_STATS_OUT_DEGREE);
    }
    let take = usize::try_from(resident_sources).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "betweenness resident source count does not fit usize",
        )
        .with_fact("resident_sources", resident_sources)
        .with_source(error)
    })?;
    let top_out_degrees = out_degrees
        .with_execution(ctx)
        .sort_values([GRAPH_STATS_OUT_DEGREE], CudfSortOptions::new().with_descending(vec![true]))
        .and_then(|frame| frame.with_execution(ctx).head(take))
        .map_err(|error| {
            QueryError::from_cudf(
                GRAPH_STATISTICS_OPERATION,
                "failed to compute top betweenness source out-degrees",
                error,
            )
        })?;
    let top_out_degree = top_out_degrees.series(GRAPH_STATS_OUT_DEGREE).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read top graph out-degree column", error)
    })?;
    graph_statistics_series_sum_u64(&top_out_degree, ctx, GRAPH_STATS_OUT_DEGREE)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_betweenness_resident_sources(
    spec: &crate::plan::GraphAlgorithmSpec,
    exact_vertices: u64,
    vertex_width_bytes: u64,
    device_total_memory_bytes: Option<u64>,
) -> Result<u64> {
    let requested_sources = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => params
            .seeds
            .as_ref()
            .map_or_else(|| params.k.unwrap_or(exact_vertices), |seeds| seeds.row_count_upper_bound()),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(_) => return Ok(1),
        _ => return Ok(0),
    };
    let mut resident_sources = requested_sources.clamp(1, u16::MAX as u64);
    let Some(total_memory_bytes) = device_total_memory_bytes else {
        return Ok(resident_sources);
    };
    if resident_sources <= 1 || exact_vertices == 0 {
        return Ok(resident_sources);
    }
    let bfs_result_width = vertex_width_bytes.checked_mul(2).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "runtime graph statistics pricing is not finite",
        )
        .with_fact("reason", "graph_statistics_pricing_overflow")
        .with_fact("term", "betweenness_bfs_result_width")
        .with_fact("operation", "multiplication")
    })?;
    let bfs_result_bytes_per_source = exact_vertices.checked_mul(bfs_result_width).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "runtime graph statistics pricing is not finite",
        )
        .with_fact("reason", "graph_statistics_pricing_overflow")
        .with_fact("term", "betweenness_bfs_result_bytes")
        .with_fact("operation", "multiplication")
    })?;
    if bfs_result_bytes_per_source == 0 {
        return Ok(resident_sources);
    }
    resident_sources = resident_sources.min(total_memory_bytes / 4 / bfs_result_bytes_per_source).max(1);
    Ok(resident_sources)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_triangle_undirected_edges(
    edge_frame: &DataFrame,
    ctx: &ExecutionOverrides<'_>,
) -> Result<DataFrame> {
    let non_loop_edges =
        edge_frame.with_execution(ctx).filter_expr(col(GRAPH_STATS_SRC).ne(col(GRAPH_STATS_DST))).map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to remove graph self-loops", error)
        })?;
    let canonical = non_loop_edges
        .with_execution(ctx)
        .with_columns([
            when(col(GRAPH_STATS_SRC).le(col(GRAPH_STATS_DST)))
                .then(col(GRAPH_STATS_SRC))
                .otherwise(col(GRAPH_STATS_DST))
                .alias(GRAPH_STATS_CANONICAL_SRC),
            when(col(GRAPH_STATS_SRC).le(col(GRAPH_STATS_DST)))
                .then(col(GRAPH_STATS_DST))
                .otherwise(col(GRAPH_STATS_SRC))
                .alias(GRAPH_STATS_CANONICAL_DST),
        ])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to canonicalize undirected graph edges", error)
        })?;
    let canonical = canonical
        .select([GRAPH_STATS_CANONICAL_SRC, GRAPH_STATS_CANONICAL_DST])
        .and_then(|frame| {
            frame.rename_columns([
                (GRAPH_STATS_CANONICAL_SRC, GRAPH_STATS_SRC),
                (GRAPH_STATS_CANONICAL_DST, GRAPH_STATS_DST),
            ])
        })
        .map_err(|error| {
            QueryError::from_cudf(
                GRAPH_STATISTICS_OPERATION,
                "failed to project canonical undirected graph edges",
                error,
            )
        })?;
    canonical
        .with_execution(ctx)
        .drop_duplicates(DropDuplicatesOptions::builder().subset(&[GRAPH_STATS_SRC, GRAPH_STATS_DST]).build())
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to deduplicate undirected graph edges", error)
        })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_undirected_degrees(edge_frame: &DataFrame, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
    let source = edge_frame.series(GRAPH_STATS_SRC).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph source column", error)
    })?;
    let destination = edge_frame.series(GRAPH_STATS_DST).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph destination column", error)
    })?;
    let endpoints = cudf::Series::concat_in(&[&source, &destination], ctx).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to concat graph endpoints", error)
    })?;
    let endpoint_values = endpoints.clone();
    let endpoint_frame =
        DataFrame::from_series([(GRAPH_STATS_VERTEX, endpoints), (GRAPH_STATS_ENDPOINT, endpoint_values)]).map_err(
            |error| {
                QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to build graph endpoint degree frame", error)
            },
        )?;
    let degrees = endpoint_frame
        .groupby([GRAPH_STATS_VERTEX])
        .with_execution(ctx)
        .agg(&[CudfAggSpec::new(GRAPH_STATS_ENDPOINT, GroupByAgg::Count { include_nulls: false })
            .with_alias(GRAPH_STATS_OUT_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute graph undirected degrees", error)
        })?;
    graph_statistics_cast_column_to_u64(&degrees, GRAPH_STATS_OUT_DEGREE, ctx)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_triangle_min_degree(
    edge_frame: &DataFrame,
    vertex_degrees: &DataFrame,
    ctx: &ExecutionOverrides<'_>,
    device_multiprocessor_count: Option<u32>,
) -> Result<(u64, u64, u64)> {
    let source_degrees = vertex_degrees
        .rename_columns([(GRAPH_STATS_VERTEX, GRAPH_STATS_SRC), (GRAPH_STATS_OUT_DEGREE, GRAPH_STATS_SRC_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to prepare source graph degree frame", error)
        })?;
    let destination_degrees = vertex_degrees
        .rename_columns([(GRAPH_STATS_VERTEX, GRAPH_STATS_DST), (GRAPH_STATS_OUT_DEGREE, GRAPH_STATS_DST_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to prepare destination graph degree frame", error)
        })?;
    let edge_degrees = edge_frame
        .with_execution(ctx)
        .merge(&source_degrees, cudf::MergeOptions::new().how_left().on([GRAPH_STATS_SRC]))
        .and_then(|frame| {
            frame
                .with_execution(ctx)
                .merge(&destination_degrees, cudf::MergeOptions::new().how_left().on([GRAPH_STATS_DST]))
        })
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to join graph endpoint degrees", error)
        })?;
    let edge_degrees = edge_degrees
        .with_execution(ctx)
        .with_columns([
            when(col(GRAPH_STATS_SRC_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_SRC_DEGREE))
                .alias(GRAPH_STATS_SRC_DEGREE),
            when(col(GRAPH_STATS_DST_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_DST_DEGREE))
                .alias(GRAPH_STATS_DST_DEGREE),
        ])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to normalize graph endpoint degrees", error)
        })?;
    let source_first = col(GRAPH_STATS_SRC_DEGREE).lt(col(GRAPH_STATS_DST_DEGREE))
        | (col(GRAPH_STATS_SRC_DEGREE).eq(col(GRAPH_STATS_DST_DEGREE)) & col(GRAPH_STATS_SRC).lt(col(GRAPH_STATS_DST)));
    let oriented_edges = edge_degrees
        .with_execution(ctx)
        .with_columns([
            when(source_first.clone())
                .then(col(GRAPH_STATS_SRC))
                .otherwise(col(GRAPH_STATS_DST))
                .alias(GRAPH_STATS_ORIENTED_SRC),
            when(source_first.clone())
                .then(col(GRAPH_STATS_DST))
                .otherwise(col(GRAPH_STATS_SRC))
                .alias(GRAPH_STATS_ORIENTED_DST),
        ])
        .map_err(|error| QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to orient graph edges", error))?;
    let oriented_edges = oriented_edges
        .select([GRAPH_STATS_ORIENTED_SRC, GRAPH_STATS_ORIENTED_DST])
        .and_then(|frame| {
            frame.with_execution(ctx).drop_duplicates(
                DropDuplicatesOptions::builder().subset(&[GRAPH_STATS_ORIENTED_SRC, GRAPH_STATS_ORIENTED_DST]).build(),
            )
        })
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to deduplicate oriented graph edges", error)
        })?;
    let oriented_edge_count = oriented_edges.num_rows() as u64;
    let oriented_out_degrees = oriented_edges
        .groupby([GRAPH_STATS_ORIENTED_SRC])
        .with_execution(ctx)
        .agg(&[CudfAggSpec::new(GRAPH_STATS_ORIENTED_DST, GroupByAgg::Count { include_nulls: false })
            .with_alias(GRAPH_STATS_ORIENTED_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute oriented graph degrees", error)
        })?;
    let oriented_out_degrees =
        graph_statistics_cast_column_to_u64(&oriented_out_degrees, GRAPH_STATS_ORIENTED_DEGREE, ctx)?;
    let oriented_source_degrees = oriented_out_degrees
        .rename_column(GRAPH_STATS_ORIENTED_DEGREE, GRAPH_STATS_ORIENTED_SRC_DEGREE)
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to prepare oriented source degree frame", error)
        })?;
    let oriented_destination_degrees = oriented_out_degrees
        .rename_columns([
            (GRAPH_STATS_ORIENTED_SRC, GRAPH_STATS_ORIENTED_DST),
            (GRAPH_STATS_ORIENTED_DEGREE, GRAPH_STATS_ORIENTED_DST_DEGREE),
        ])
        .map_err(|error| {
            QueryError::from_cudf(
                GRAPH_STATISTICS_OPERATION,
                "failed to prepare oriented destination degree frame",
                error,
            )
        })?;
    let oriented_edge_degrees = oriented_edges
        .with_execution(ctx)
        .merge(&oriented_source_degrees, cudf::MergeOptions::new().how_left().on([GRAPH_STATS_ORIENTED_SRC]))
        .and_then(|frame| {
            frame.with_execution(ctx).merge(
                &oriented_destination_degrees,
                cudf::MergeOptions::new().how_left().on([GRAPH_STATS_ORIENTED_DST]),
            )
        })
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to join oriented graph degrees", error)
        })?;
    let oriented_edge_degrees = oriented_edge_degrees
        .with_execution(ctx)
        .with_columns([
            when(col(GRAPH_STATS_ORIENTED_SRC_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_ORIENTED_SRC_DEGREE))
                .alias(GRAPH_STATS_ORIENTED_SRC_DEGREE),
            when(col(GRAPH_STATS_ORIENTED_DST_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_ORIENTED_DST_DEGREE))
                .alias(GRAPH_STATS_ORIENTED_DST_DEGREE),
        ])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to normalize oriented graph degrees", error)
        })?;
    let edge_terms = oriented_edge_degrees
        .with_execution(ctx)
        .with_columns([when(col(GRAPH_STATS_ORIENTED_SRC_DEGREE).le(col(GRAPH_STATS_ORIENTED_DST_DEGREE)))
            .then(col(GRAPH_STATS_ORIENTED_SRC_DEGREE))
            .otherwise(col(GRAPH_STATS_ORIENTED_DST_DEGREE))
            .alias(GRAPH_STATS_EDGE_MIN_DEGREE)])
        .and_then(|frame| frame.select([GRAPH_STATS_ORIENTED_DST, GRAPH_STATS_EDGE_MIN_DEGREE]))
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute graph edge min-degree terms", error)
        })?;
    let edge_min_degree = edge_terms.series(GRAPH_STATS_EDGE_MIN_DEGREE).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph edge min-degree column", error)
    })?;
    let sum_min_degree_oriented = graph_statistics_series_sum_u64(&edge_min_degree, ctx, GRAPH_STATS_EDGE_MIN_DEGREE)?;
    let peak_chunk_min_degree_sum = crate::graph_memory::triangle_count_peak_chunk_min_degree_pricing_term(
        sum_min_degree_oriented,
        oriented_edge_count,
        device_multiprocessor_count,
    )?;
    Ok((sum_min_degree_oriented, oriented_edge_count, peak_chunk_min_degree_sum))
}

#[cfg(feature = "cugraph")]
fn graph_statistics_cast_column_to_u64(
    frame: &DataFrame,
    column: &'static str,
    ctx: &ExecutionOverrides<'_>,
) -> Result<DataFrame> {
    frame
        .with_execution(ctx)
        .with_columns([col(column).cast(cudf::DataType::new(cudf::TypeId::UInt64)).alias(column)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to cast graph statistics column to uint64", error)
                .with_fact("column", column)
        })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_nonnegative_i64(value: i64, column: &'static str) -> Result<u64> {
    u64::try_from(value).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "graph statistics value does not fit u64",
        )
        .with_fact("column", column)
        .with_fact("value", value)
        .with_source(error)
    })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_sum_u64(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<u64> {
    let value = series.sum_value_in::<i64>(ctx).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to reduce graph statistics sum", error)
            .with_fact("column", column)
    })?;
    graph_statistics_nonnegative_i64(value, column)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_max_u64(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<Option<u64>> {
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to inspect graph statistics column dtype", error)
            .with_fact("column", column)
    })?;
    let value = match dtype.id() {
        cudf::TypeId::UInt64 => graph_statistics_series_max_value::<u64>(series, ctx, column)?,
        cudf::TypeId::UInt32 => graph_statistics_series_max_value::<u32>(series, ctx, column)?.map(u64::from),
        cudf::TypeId::UInt16 => graph_statistics_series_max_value::<u16>(series, ctx, column)?.map(u64::from),
        cudf::TypeId::UInt8 => graph_statistics_series_max_value::<u8>(series, ctx, column)?.map(u64::from),
        cudf::TypeId::Int64 => graph_statistics_series_max_value::<i64>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(value, column))
            .transpose()?,
        cudf::TypeId::Int32 => graph_statistics_series_max_value::<i32>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(i64::from(value), column))
            .transpose()?,
        cudf::TypeId::Int16 => graph_statistics_series_max_value::<i16>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(i64::from(value), column))
            .transpose()?,
        cudf::TypeId::Int8 => graph_statistics_series_max_value::<i8>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(i64::from(value), column))
            .transpose()?,
        dtype => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                GRAPH_STATISTICS_OPERATION,
                "graph statistics max requires an integer column",
            )
            .with_fact("column", column)
            .with_fact("cudf_dtype", dtype.name()));
        },
    };
    Ok(value)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_max_value<T: cudf::CudfType>(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<Option<T>> {
    series.max_value_in::<T>(ctx).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to reduce graph statistics max", error)
            .with_fact("column", column)
    })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_p99_u64(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<u64> {
    if series.is_empty() {
        return Ok(0);
    }
    let rank = series.len().saturating_mul(99).div_ceil(100).saturating_sub(1);
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to inspect graph statistics column dtype", error)
            .with_fact("column", column)
    })?;
    let value = match dtype.id() {
        cudf::TypeId::UInt64 => graph_statistics_sorted_rank_value::<u64>(series, ctx, rank, column),
        cudf::TypeId::UInt32 => graph_statistics_sorted_rank_value::<u32>(series, ctx, rank, column).map(u64::from),
        cudf::TypeId::UInt16 => graph_statistics_sorted_rank_value::<u16>(series, ctx, rank, column).map(u64::from),
        cudf::TypeId::UInt8 => graph_statistics_sorted_rank_value::<u8>(series, ctx, rank, column).map(u64::from),
        cudf::TypeId::Int64 => graph_statistics_sorted_rank_value::<i64>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(value, column)),
        cudf::TypeId::Int32 => graph_statistics_sorted_rank_value::<i32>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(i64::from(value), column)),
        cudf::TypeId::Int16 => graph_statistics_sorted_rank_value::<i16>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(i64::from(value), column)),
        cudf::TypeId::Int8 => graph_statistics_sorted_rank_value::<i8>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(i64::from(value), column)),
        dtype => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            GRAPH_STATISTICS_OPERATION,
            "graph degree p99 requires an integer column",
        )
        .with_fact("column", column)
        .with_fact("cudf_dtype", dtype.name())),
    }?;
    Ok(value)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_sorted_rank_value<T: cudf::CudfType>(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    rank: usize,
    column: &'static str,
) -> Result<T> {
    let value = series
        .sort_values_in(CudfSortOptions::new(), ctx)
        .and_then(|series| series.slice_in(rank, 1, ctx))
        .and_then(|series| series.to_vec_strict::<T>())
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph degree p99 statistic", error)
                .with_fact("column", column)
                .with_fact("rank", rank)
        })?;
    value.into_iter().next().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "graph degree p99 statistic slice unexpectedly returned no rows",
        )
        .with_fact("column", column)
        .with_fact("rank", rank)
    })
}
