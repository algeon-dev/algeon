use super::{node::projected_schema, *};

pub(super) fn execute_cugraph_query_to_host(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    singleflight_leader: Option<&mut GraphBfsLeader>,
) -> Result<RecordBatch> {
    let graph_context = graph_algorithm_execution_context(context)?;
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        projection,
        limit,
        CugraphQuerySink::HostArrow,
        edge_normalize_inserted,
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
        "datafusion_nexus.cugraph_sql.execute",
        "native cuGraph SQL query execution failed",
    )?;
    let QueryOutput::HostArrow(batch) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.execute",
            "native cuGraph SQL query did not produce host Arrow output",
        )
        .into());
    };
    Ok(batch)
}

pub(super) fn execute_cugraph_query_to_gpu_with_options(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    singleflight_leader: Option<&mut GraphBfsLeader>,
) -> Result<GpuResultHandle> {
    let graph_context = graph_algorithm_execution_context(context)?;
    let plan = cugraph_query_plan(
        &graph_context,
        spec,
        edge_schema,
        edge_frame,
        predicate_frames,
        projection,
        limit,
        CugraphQuerySink::GpuResident,
        edge_normalize_inserted,
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

#[derive(Debug, Clone, Copy)]
pub(super) enum CugraphQuerySink {
    HostArrow,
    GpuResident,
}

pub(super) struct CugraphQueryPlan {
    plan: nexus_query_engine::plan::QueryPlan,
    inputs: GpuResidentInputMap,
}

pub(super) fn cugraph_query_plan(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    sink: CugraphQuerySink,
    edge_normalize_inserted: bool,
) -> Result<CugraphQueryPlan> {
    let input_name = "__datafusion_nexus_cugraph_edges";
    let handle = GpuResultHandle::new(context, edge_frame, Arc::clone(&edge_schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.gpu_input",
            "failed to bind cuGraph edge source as a GPU-resident native input",
            error,
        ))
    })?;
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);

    let mut builder = QueryPlanBuilder::new();
    let input =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&edge_schema))));
    let predicate_nodes = bind_predicate_gpu_inputs(context, &mut builder, &mut inputs, predicate_frames)?;
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
    let plan = match sink {
        CugraphQuerySink::HostArrow => builder.finish_with_host_sink_schema(root, root_schema),
        CugraphQuerySink::GpuResident => builder.finish_gpu_resident(root),
    };
    Ok(CugraphQueryPlan { plan, inputs })
}

pub(super) fn bind_predicate_gpu_inputs(
    context: &ExecutionContext,
    builder: &mut QueryPlanBuilder,
    inputs: &mut GpuResidentInputMap,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<Vec<nexus_query_engine::plan::NodeId>> {
    let mut nodes = Vec::with_capacity(predicate_frames.len());
    for (index, CugraphGpuFrame { schema, frame }) in predicate_frames.into_iter().enumerate() {
        let input_name = format!("__datafusion_nexus_cugraph_side_input_{index}");
        let handle = GpuResultHandle::new(context, frame, Arc::clone(&schema)).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.gpu_input",
                "failed to bind cuGraph side-input source as a GPU-resident native input",
                error,
            ))
        })?;
        inputs.insert(input_name.clone(), handle);
        nodes.push(builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema))));
    }
    Ok(nodes)
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

pub(super) fn native_graph_spec_for_memory_estimate(
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
) -> Option<GraphAlgorithmSpec> {
    let mut edge_columns =
        GraphEdgeColumnBindings::new(spec.src_col.clone(), spec.dst_col.clone(), spec.weight_col.clone());
    if let Some(edge_id) = &spec.edge_id_col {
        edge_columns = edge_columns.with_edge_id(edge_id.clone());
    }
    let algorithm = native_algorithm(spec.algorithm);
    let params = native_algorithm_params_for_memory_estimate(&spec.params)?;
    let output_schema = algorithm.definition().result_schema_for_params(&params);
    Some(
        GraphAlgorithmSpec::new(
            algorithm,
            edge_columns,
            GraphConstructionOptions {
                directed: spec.graph_options.directed,
                renumber: spec.graph_options.renumber,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            params,
            output_schema,
        )
        .with_edge_normalize_inserted(edge_normalize_inserted),
    )
}

pub(super) fn native_algorithm_params_for_memory_estimate(
    params: &CugraphAlgorithmParams,
) -> Option<GraphAlgorithmParams> {
    match params {
        // These variants require side-input node ids in the native IR. Fall back
        // to tenant-default admission instead of minting placeholder ids.
        CugraphAlgorithmParams::PersonalizedPageRank(_) => None,
        CugraphAlgorithmParams::Bfs(_) => None,
        _ => Some(native_algorithm_params(params, &[])),
    }
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
