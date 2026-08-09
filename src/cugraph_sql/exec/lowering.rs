use super::{node::projected_schema, *};

pub(super) fn preflight_cugraph_query_plan(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
) -> Result<()> {
    let predicate_row_upper_bounds = vec![None; predicate_schemas.len()];
    let plan = cugraph_execution_plan(
        spec,
        edge_schema,
        predicate_schemas,
        &predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )?;
    nexus_query_engine::exec::preflight_query_plan(&plan).map_err(|error| {
        if error.kind() == nexus_query_engine::error::QueryErrorKind::Unsupported {
            return DataFusionError::from(
                Error::new(
                    ErrorCode::RequiredGpuOperationUnsupported,
                    "datafusion_nexus.cugraph_sql.preflight",
                    "required cuGraph GPU operation is unsupported by the native engine",
                )
                .with_fact("native_code", error.code())
                .with_fact("native_operation", error.operation())
                .with_source(error),
            );
        }
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.preflight",
            "native cuGraph query-plan preflight failed",
            error,
        ))
    })
}

pub(super) fn execute_cugraph_query_to_gpu_with_options(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
    singleflight_leader: Option<&mut GraphBfsLeader>,
) -> Result<GpuResultHandle> {
    let graph_context = graph_algorithm_execution_context(context)?;
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )?;
    if let Some(leader) = singleflight_leader {
        leader.record_traversal_submission();
    }
    let outcome = execute_query_plan_with_gpu_inputs_observed(
        &graph_context,
        &plan.plan,
        &plan.inputs,
        metrics.runtime_observation_config(),
    );
    let output = finalize_graph_query_outcome(
        metrics,
        outcome,
        "datafusion_nexus.cugraph_sql.execute_gpu",
        "native cuGraph SQL GPU-resident query execution failed",
    )?;
    let QueryOutput::GpuResident(handle) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.execute_gpu",
            "native cuGraph SQL query did not produce GPU-resident output",
        )
        .with_fact("result_sink", "host_arrow")
        .into());
    };
    Ok(handle)
}

pub(super) fn graph_algorithm_execution_context(context: &ExecutionContext) -> Result<ExecutionContext> {
    // Graph algorithms remain outside strict bounded admission, but bounded
    // service contexts keep their memory policy so graph workspace-heavy
    // algorithms can price/admit their extra workspace before entering cuGraph.
    Ok(context.clone())
}

pub(super) struct CugraphQueryPlan {
    plan: nexus_query_engine::plan::QueryPlan,
    inputs: GpuResidentInputMap,
}

const CUGRAPH_EDGE_INPUT_NAME: &str = "__datafusion_nexus_cugraph_edges";

fn cugraph_side_input_name(index: usize) -> String {
    format!("__datafusion_nexus_cugraph_side_input_{index}")
}

pub(super) fn cugraph_query_plan(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
) -> Result<CugraphQueryPlan> {
    let handle = GpuResultHandle::new(context, edge_frame, Arc::clone(&edge_schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.gpu_input",
            "failed to bind cuGraph edge source as a GPU-resident native input",
            error,
        ))
    })?;
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(CUGRAPH_EDGE_INPUT_NAME.to_string(), handle);
    let predicate_schemas = predicate_frames.iter().map(|frame| Arc::clone(&frame.schema)).collect::<Vec<_>>();
    bind_predicate_gpu_inputs(context, &mut inputs, predicate_frames)?;
    let plan = cugraph_execution_plan_from_prepared_schemas(
        spec,
        edge_schema,
        predicate_schemas,
        predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )?;
    Ok(CugraphQueryPlan { plan, inputs })
}

/// Builds the graph fragment after applying the same schema-only preparation
/// and output deferral that runtime GPU input preparation performs.  Preflight
/// and admission attestation both use this path so a plan scope represents the
/// physical graph fragment, not just its capability set.
fn cugraph_execution_plan(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
) -> Result<nexus_query_engine::plan::QueryPlan> {
    let prepared = prepare_bfs_logical_input_schemas(spec, edge_schema, predicate_schemas)?;
    // Logical-id decoding and path reconstruction are host-side finalization
    // steps. The physical graph fragment retains its raw traversal output
    // until those steps complete, so a user projection/limit is not part of
    // its attested execution shape.
    let defer_final_projection = prepared.logical_mapping || is_bfs_path_output(spec);
    cugraph_execution_plan_from_prepared_schemas(
        &prepared.execution_spec,
        prepared.edge_schema,
        prepared.predicate_schemas,
        predicate_row_upper_bounds,
        if defer_final_projection { None } else { projection },
        if defer_final_projection { None } else { limit },
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )
}

/// Builds the physical graph fragment from already-prepared execution schemas.
/// Runtime uses this after GPU input preparation; plan-time preflight and
/// attestation reach it through [`cugraph_execution_plan`].
fn cugraph_execution_plan_from_prepared_schemas(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
) -> Result<nexus_query_engine::plan::QueryPlan> {
    let predicate_row_upper_bounds =
        prepared_predicate_row_upper_bounds(spec, &predicate_schemas, predicate_row_upper_bounds)?;
    cugraph_schema_query_plan(
        spec,
        edge_schema,
        predicate_schemas,
        &predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )
}

fn prepared_predicate_row_upper_bounds(
    spec: &CugraphAlgorithmSpec,
    predicate_schemas: &[SchemaRef],
    predicate_row_upper_bounds: &[Option<u64>],
) -> Result<Vec<Option<u64>>> {
    let mut bounds = predicate_row_upper_bounds.to_vec();
    if let CugraphAlgorithmParams::Bfs(params) = &spec.params
        && params.source.is_utf8()
    {
        let source_rows = params.source.literal_count().and_then(|rows| u64::try_from(rows).ok()).ok_or_else(|| {
            plan_error("cuGraph logical BFS source row bound does not fit the native cardinality contract")
        })?;
        match predicate_schemas.len().cmp(&bounds.len()) {
            std::cmp::Ordering::Greater if predicate_schemas.len() == bounds.len() + 1 => {
                bounds.insert(0, Some(source_rows));
            },
            std::cmp::Ordering::Equal => {
                if bounds.first().copied() != Some(Some(source_rows)) {
                    return Err(plan_error(
                        "cuGraph logical BFS source row bound does not match its exact scalar or literal source cardinality",
                    ));
                }
            },
            _ => {
                return Err(plan_error(
                    "cuGraph prepared predicate schemas and row-cardinality evidence must have matching lengths",
                ));
            },
        }
    }
    if predicate_schemas.len() != bounds.len() {
        return Err(plan_error(
            "cuGraph prepared predicate schemas and row-cardinality evidence must have matching lengths",
        ));
    }
    Ok(bounds)
}

pub(super) fn cugraph_schema_query_plan(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
) -> Result<nexus_query_engine::plan::QueryPlan> {
    if predicate_schemas.len() != predicate_row_upper_bounds.len() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.plan",
            "cuGraph predicate schemas and row-cardinality evidence must have matching lengths",
        )
        .with_fact("predicate_schema_count", predicate_schemas.len())
        .with_fact("predicate_row_upper_bound_count", predicate_row_upper_bounds.len())
        .into());
    }
    let mut builder = QueryPlanBuilder::new();
    let edge_source = GpuResidentSource::new(CUGRAPH_EDGE_INPUT_NAME, Arc::clone(&edge_schema));
    let edge_source =
        edge_row_upper_bound.map_or(edge_source.clone(), |rows| edge_source.with_row_count_upper_bound(rows));
    let edge_source = edge_vertex_upper_bound
        .map_or(edge_source.clone(), |vertices| edge_source.with_graph_vertex_upper_bound(vertices));
    let input = builder.add_source(SourceNode::GpuResident(edge_source));
    let predicate_nodes = predicate_schemas
        .into_iter()
        .enumerate()
        .map(|(index, schema)| {
            let row_count_upper_bound = predicate_row_upper_bounds.get(index).copied().flatten();
            let source = GpuResidentSource::new(cugraph_side_input_name(index), schema);
            let source = match row_count_upper_bound {
                Some(rows) => source.with_row_count_upper_bound(rows),
                None => source,
            };
            builder.add_source(SourceNode::GpuResident(source))
        })
        .collect::<Vec<_>>();
    let graph_spec = native_graph_spec(spec, edge_normalize_inserted, &predicate_nodes);
    let mut root_schema = Arc::clone(&graph_spec.output_schema);
    let mut root = builder.try_add_graph_algorithm(input, graph_spec).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.plan",
            "failed to build native cuGraph query plan",
            error,
        ))
    })?;
    if let Some(outputs) = bfs_normalized_output_projection(spec, &edge_schema)? {
        root = builder.add_projection(root, outputs);
        root_schema = Arc::clone(&spec.output_schema);
    }
    if let Some(projection) = projection {
        root_schema = projected_schema(&root_schema, Some(projection))?;
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    spec.output_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(nexus_query_engine::expr::Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root = builder.add_projection(root, outputs);
    }
    if let Some(limit) = limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    Ok(builder.finish_gpu_resident(root))
}

pub(super) fn cugraph_execution_capability_proof_with_edge_row_upper_bound(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    edge_row_upper_bound: Option<u64>,
    edge_vertex_upper_bound: Option<u64>,
) -> Result<(nexus_query_engine::plan::QueryPlan, nexus_query_engine::admission::QueryPlanCapabilityProof)> {
    let plan = cugraph_execution_plan(
        spec,
        edge_schema,
        predicate_schemas,
        predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )?;
    let graph_proof = nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .ok_or_else(|| {
            DataFusionError::from(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.plan_capability_proof",
                "native cuGraph query plan did not expose capability evidence",
            ))
        })?;
    Ok((plan, graph_proof))
}

pub(super) fn bind_predicate_gpu_inputs(
    context: &ExecutionContext,
    inputs: &mut GpuResidentInputMap,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<()> {
    for (index, CugraphGpuFrame { schema, frame }) in predicate_frames.into_iter().enumerate() {
        let input_name = cugraph_side_input_name(index);
        let handle = GpuResultHandle::new(context, frame, Arc::clone(&schema)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.gpu_input",
                "failed to bind cuGraph side-input source as a GPU-resident native input",
                error,
            ))
        })?;
        inputs.insert(input_name.clone(), handle);
    }
    Ok(())
}

pub(super) fn native_graph_spec(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
    predicate_nodes: &[nexus_query_engine::plan::NodeId],
) -> GraphAlgorithmSpec {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
    }
    let algorithm = native_algorithm(spec.algorithm);
    let params = native_algorithm_params(&spec.params, predicate_nodes);
    let output_schema = algorithm.definition().result_schema_for_params(&params);
    GraphAlgorithmSpec::new(
        algorithm,
        edge_columns,
        GraphConstructionOptions {
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            policy: if edge_normalize_inserted {
                GraphConstructionPolicy::PythonCugraph
            } else {
                match spec.graph_options.construction_policy {
                    CugraphGraphConstructionPolicy::PythonCugraph | CugraphGraphConstructionPolicy::RawLibcugraph => {
                        GraphConstructionPolicy::RawLibcugraph
                    },
                }
            },
        },
        params,
        output_schema,
    )
}

pub(super) fn native_graph_spec_for_placement_hint(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
) -> GraphAlgorithmSpec {
    let mut builder = QueryPlanBuilder::new();
    let predicate_nodes = predicate_sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
                format!("__datafusion_nexus_cugraph_placement_hint_{index}"),
                source.schema(),
            )))
        })
        .collect::<Vec<_>>();
    native_graph_spec(spec, edge_normalize_inserted, &predicate_nodes)
}

pub(super) fn bfs_normalized_output_projection(
    spec: &CugraphAlgorithmSpec,
    edge_schema: &SchemaRef,
) -> Result<Option<Vec<ProjectionOutput>>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Normalized {
        return Ok(None);
    }

    let unreachable_sentinel = bfs_unreachable_distance_sentinel(spec, edge_schema)?;
    let distance_is_unreachable = NativeExpr::binary(
        NativeExpr::column("distance"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
    );
    let predecessor_is_absent = NativeExpr::binary(
        NativeExpr::column("predecessor"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(-1)),
    );

    let fields = spec.output_schema.fields();
    let mut outputs = vec![
        ProjectionOutput::new(NativeExpr::column("vertex"), fields[0].as_ref().clone()),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(
                    distance_is_unreachable.clone(),
                    NativeExpr::literal(LiteralValue::Null(DataType::Int64)),
                )],
                NativeExpr::column("distance"),
            ),
            fields[1].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(predecessor_is_absent, NativeExpr::literal(LiteralValue::Null(DataType::Int64)))],
                NativeExpr::column("predecessor"),
            ),
            fields[2].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::binary(
                NativeExpr::column("distance"),
                BinaryOp::NotEq,
                NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
            ),
            fields[3].as_ref().clone(),
        ),
    ];
    if params.output.return_target_info {
        outputs.push(ProjectionOutput::new(NativeExpr::column("target_found"), fields[4].as_ref().clone()));
        outputs.push(ProjectionOutput::new(NativeExpr::column("target_distance"), fields[5].as_ref().clone()));
    }
    Ok(Some(outputs))
}

pub(super) fn bfs_unreachable_distance_sentinel(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<i64> {
    let source_field = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!(
            "cuGraph BFS output normalization could not find source column '{}' in edge source schema",
            spec.src_col
        ))
    })?;
    match source_field.data_type() {
        DataType::Int32 => Ok(i64::from(i32::MAX)),
        DataType::Int64 => Ok(i64::MAX),
        other => Err(plan_error(format!(
            "cuGraph BFS output normalization requires Int32 or Int64 source vertices, found {other:?}"
        ))),
    }
}

pub(super) fn native_algorithm(algorithm: CugraphAlgorithm) -> GraphAlgorithm {
    algorithm.native_algorithm()
}

pub(super) fn native_algorithm_params(
    params: &CugraphAlgorithmParams,
    predicate_nodes: &[nexus_query_engine::plan::NodeId],
) -> GraphAlgorithmParams {
    match params {
        CugraphAlgorithmParams::PageRank(params) => GraphAlgorithmParams::PageRank(GraphPageRankParams {
            alpha: params.alpha,
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
        }),
        CugraphAlgorithmParams::PersonalizedPageRank(params) => {
            let personalization_node = predicate_nodes.first().copied().expect("personalization node missing");
            GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
                pagerank: GraphPageRankParams {
                    alpha: params.pagerank.alpha,
                    epsilon: params.pagerank.epsilon,
                    max_iterations: params.pagerank.max_iterations,
                },
                personalization: NativeGraphPageRankPersonalizationInput::new(
                    personalization_node,
                    params.personalization.vertex_col.clone(),
                    params.personalization.value_col.clone(),
                ),
            })
        },
        CugraphAlgorithmParams::Bfs(params) => {
            GraphAlgorithmParams::Bfs(Box::new(native_bfs_params(params, predicate_nodes)))
        },
        CugraphAlgorithmParams::Louvain(params) => GraphAlgorithmParams::Louvain(GraphLouvainParams {
            max_level: params.max_level,
            threshold: params.threshold,
            resolution: params.resolution,
        }),
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            GraphAlgorithmParams::EigenvectorCentrality(GraphEigenvectorCentralityParams {
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            })
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            GraphAlgorithmParams::KatzCentrality(GraphKatzCentralityParams {
                alpha: params.alpha,
                beta: params.beta,
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            })
        },
        CugraphAlgorithmParams::Hits(params) => GraphAlgorithmParams::Hits(GraphHitsParams {
            epsilon: params.epsilon,
            max_iterations: params.max_iterations,
            normalize: params.normalize,
        }),
        CugraphAlgorithmParams::BetweennessCentrality(params) => {
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                seeds: params.seeds.as_ref().map(|seeds| GraphBetweennessSeeds::I64(seeds.vertices.clone())),
                k: params.k,
                normalized: params.normalized,
                include_endpoints: params.include_endpoints,
                exact_vertex_threshold: params.exact_vertex_threshold,
            })
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
                seeds: params.seeds.as_ref().map(|seeds| GraphBetweennessSeeds::I64(seeds.vertices.clone())),
                k: params.k,
                normalized: params.normalized,
                exact_vertex_threshold: params.exact_vertex_threshold,
            })
        },
        CugraphAlgorithmParams::Sssp(params) => {
            GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: params.source_vertex, cutoff: params.cutoff })
        },
        CugraphAlgorithmParams::DegreesAll(_) => GraphAlgorithmParams::DegreesAll(GraphDegreesAllParams),
        CugraphAlgorithmParams::InDegreesAll(_) => GraphAlgorithmParams::InDegreesAll(GraphInDegreesAllParams),
        CugraphAlgorithmParams::OutDegreesAll(_) => GraphAlgorithmParams::OutDegreesAll(GraphOutDegreesAllParams),
        CugraphAlgorithmParams::WeaklyConnectedComponents(_) => {
            GraphAlgorithmParams::WeaklyConnectedComponents(GraphWccParams)
        },
        CugraphAlgorithmParams::StronglyConnectedComponents(_) => {
            GraphAlgorithmParams::StronglyConnectedComponents(GraphSccParams)
        },
        CugraphAlgorithmParams::TriangleCountAll(_) => {
            GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams)
        },
        CugraphAlgorithmParams::CoreNumber(params) => GraphAlgorithmParams::CoreNumber(GraphCoreNumberParams {
            degree_type: sql_degree_type_to_native(params.degree_type),
        }),
        CugraphAlgorithmParams::KCore(params) => GraphAlgorithmParams::KCore(GraphKCoreParams {
            k: params.k,
            degree_type: sql_degree_type_to_native(params.degree_type),
        }),
        CugraphAlgorithmParams::MinimumSpanningTree(_) => GraphAlgorithmParams::MinimumSpanningTree(GraphMstParams),
        CugraphAlgorithmParams::Leiden(params) => GraphAlgorithmParams::Leiden(GraphLeidenParams {
            max_level: params.max_level,
            resolution: params.resolution,
            theta: params.theta,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::Ecg(params) => GraphAlgorithmParams::Ecg(GraphEcgParams {
            min_weight: params.min_weight,
            ensemble_size: params.ensemble_size,
            max_level: params.max_level,
            threshold: params.threshold,
            resolution: params.resolution,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
            GraphAlgorithmParams::SpectralModularityMaximization(GraphSpectralClusteringParams {
                n_clusters: params.n_clusters,
                n_eigenvectors: params.n_eigenvectors,
                evs_tolerance: params.evs_tolerance,
                evs_max_iterations: params.evs_max_iterations,
                k_means_tolerance: params.k_means_tolerance,
                k_means_max_iterations: params.k_means_max_iterations,
                seed: params.seed,
            })
        },
        CugraphAlgorithmParams::ForceAtlas2(params) => GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params {
            max_iter: params.max_iter,
            outbound_attraction_distribution: params.outbound_attraction_distribution,
            lin_log_mode: params.lin_log_mode,
            prevent_overlapping: params.prevent_overlapping,
            overlap_scaling_ratio: params.overlap_scaling_ratio,
            edge_weight_influence: params.edge_weight_influence,
            jitter_tolerance: params.jitter_tolerance,
            barnes_hut_optimize: params.barnes_hut_optimize,
            barnes_hut_theta: params.barnes_hut_theta,
            scaling_ratio: params.scaling_ratio,
            strong_gravity_mode: params.strong_gravity_mode,
            gravity: params.gravity,
            verbose: params.verbose,
            seed: params.seed,
        }),
        CugraphAlgorithmParams::Similarity(params) => {
            let vertex_pairs_node = predicate_nodes.first().copied().expect("similarity vertex-pairs node missing");
            GraphAlgorithmParams::Similarity(nexus_query_engine::plan::GraphSimilarityParams::new(
                nexus_query_engine::plan::GraphVertexPairInput::new(
                    vertex_pairs_node,
                    params.vertex_pairs.first_vertex_col.clone(),
                    params.vertex_pairs.second_vertex_col.clone(),
                ),
            ))
        },
    }
}

pub(super) fn native_bfs_params(
    params: &super::CugraphBfsParams,
    predicate_nodes: &[nexus_query_engine::plan::NodeId],
) -> GraphBfsParams {
    let mut native = GraphBfsParams::new(params.source.as_i64().unwrap_or(0), params.depth_limit);
    let mut next = predicate_nodes.iter().copied();
    if params.source.requires_source_vertices_input() {
        native.source_vertices = Some(NativeGraphPredicateInput::new(
            next.next().expect("BFS source-vertices node missing"),
            params.source.source_vertices_column(),
        ));
    }
    native.predicates = NativeGraphBfsPredicateInputs {
        include_vertices: params.predicates.include_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        exclude_vertices: params.predicates.exclude_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        target_vertices: params.predicates.target_vertices.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        include_edge_ids: params.predicates.include_edge_ids.as_ref().map(|input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), input.column.clone())
        }),
        include_edges: params.predicates.include_edges.as_ref().map(|input| {
            NativeGraphEdgePredicateInput::new(
                next.next().expect("predicate node missing"),
                input.source_col.clone(),
                input.destination_col.clone(),
                input.edge_id_col.clone(),
            )
        }),
    };
    native.return_target_info = params.output.return_target_info;
    if params.output.mode == CugraphBfsOutputMode::Path {
        native.endpoint_policy = GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
    }
    native
}

pub(super) fn sql_degree_type_to_native(degree_type: CugraphKCoreDegreeType) -> GraphKCoreDegreeType {
    match degree_type {
        CugraphKCoreDegreeType::In => GraphKCoreDegreeType::In,
        CugraphKCoreDegreeType::Out => GraphKCoreDegreeType::Out,
        CugraphKCoreDegreeType::InOut => GraphKCoreDegreeType::InOut,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::cugraph_sql::{
        CugraphBfsOutputMode, CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsSource,
        CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphTableReference, result_schema_for_params,
    };

    fn bfs_spec(source: CugraphBfsSource, output_mode: CugraphBfsOutputMode) -> CugraphAlgorithmSpec {
        let params = CugraphAlgorithmParams::Bfs(Box::new(CugraphBfsParams {
            source,
            depth_limit: Some(3),
            predicates: Default::default(),
            output: CugraphBfsOutputOptions { mode: output_mode, return_target_info: false },
        }));
        CugraphAlgorithmSpec {
            algorithm: CugraphAlgorithm::Bfs,
            table_name: CugraphTableReference::parse("edges").expect("static table identifier is valid"),
            src_col: "src".to_string(),
            dst_col: "dst".to_string(),
            weight_col: None,
            edge_id_col: None,
            graph_options: CugraphGraphOptions {
                directed: true,
                renumber: true,
                construction_policy: CugraphGraphConstructionPolicy::RawLibcugraph,
            },
            output_schema: result_schema_for_params(CugraphAlgorithm::Bfs, &params),
            params,
        }
    }

    fn edge_schema(vertex_type: DataType) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("src", vertex_type.clone(), false),
            Field::new("dst", vertex_type, false),
        ]))
    }

    fn assert_admission_proof_matches_runtime_prepared_plan(
        spec: &CugraphAlgorithmSpec,
        edge_schema: SchemaRef,
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) {
        let predicate_row_upper_bounds = Vec::new();
        let (_, admission_proof) = cugraph_execution_capability_proof_with_edge_row_upper_bound(
            spec,
            Arc::clone(&edge_schema),
            Vec::new(),
            &predicate_row_upper_bounds,
            projection,
            limit,
            false,
            None,
            None,
        )
        .expect("cuGraph plan must expose a capability proof");

        let prepared =
            prepare_bfs_logical_input_schemas(spec, edge_schema, Vec::new()).expect("BFS input schemas must prepare");
        let defer_final_projection = prepared.logical_mapping || is_bfs_path_output(spec);
        let runtime_plan = cugraph_execution_plan_from_prepared_schemas(
            &prepared.execution_spec,
            prepared.edge_schema,
            prepared.predicate_schemas,
            &predicate_row_upper_bounds,
            if defer_final_projection { None } else { projection },
            if defer_final_projection { None } else { limit },
            false,
            None,
            None,
        )
        .expect("prepared runtime graph plan must build");
        let graph_runtime_proof =
            nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&runtime_plan)
                .expect("prepared runtime graph plan must expose a capability proof");
        let runtime_proof = graph_runtime_proof;

        assert_eq!(
            admission_proof, runtime_proof,
            "admission must attest the exact prepared graph and logical-output fragments runtime executes"
        );
    }

    #[test]
    fn admission_proof_matches_runtime_bfs_logical_and_path_prepared_shapes() {
        let projection = vec![0];
        assert_admission_proof_matches_runtime_prepared_plan(
            &bfs_spec(CugraphBfsSource::scalar_utf8("alice"), CugraphBfsOutputMode::Normalized),
            edge_schema(DataType::Utf8),
            Some(&projection),
            Some(2),
        );
        assert_admission_proof_matches_runtime_prepared_plan(
            &bfs_spec(CugraphBfsSource::scalar_i64(0), CugraphBfsOutputMode::Path),
            edge_schema(DataType::Int64),
            Some(&projection),
            Some(2),
        );
    }

    #[test]
    fn graph_scope_preserves_source_vertex_bound_across_admission_and_runtime_plans() {
        let spec = bfs_spec(CugraphBfsSource::scalar_i64(0), CugraphBfsOutputMode::Raw);
        let edge_schema = edge_schema(DataType::Int64);
        let predicate_row_upper_bounds = Vec::new();
        let (admission_plan, admission_proof) = cugraph_execution_capability_proof_with_edge_row_upper_bound(
            &spec,
            Arc::clone(&edge_schema),
            Vec::new(),
            &predicate_row_upper_bounds,
            None,
            None,
            false,
            Some(5),
            Some(11),
        )
        .expect("admission plan must build");
        let prepared =
            prepare_bfs_logical_input_schemas(&spec, edge_schema, Vec::new()).expect("BFS input schemas must prepare");
        let runtime_plan = cugraph_execution_plan_from_prepared_schemas(
            &prepared.execution_spec,
            prepared.edge_schema,
            prepared.predicate_schemas,
            &predicate_row_upper_bounds,
            None,
            None,
            false,
            Some(5),
            Some(11),
        )
        .expect("runtime plan must build with the same source bounds");

        let source_metadata = |plan: &nexus_query_engine::plan::QueryPlan| {
            plan.nodes().iter().find_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    nexus_query_engine::source::SourceNode::GpuResident(source)
                        if source.input_name == CUGRAPH_EDGE_INPUT_NAME =>
                    {
                        Some((source.row_count_upper_bound, source.graph_vertex_upper_bound))
                    },
                    _ => None,
                },
                _ => None,
            })
        };
        assert_eq!(source_metadata(&admission_plan), Some((Some(5), Some(11))));
        assert_eq!(source_metadata(&runtime_plan), Some((Some(5), Some(11))));
        assert_eq!(
            admission_proof,
            nexus_query_engine::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&runtime_plan)
                .expect("runtime plan must issue the same graph proof"),
            "runtime actual rows must not replace the admitted source-bound scope metadata"
        );
    }
}
