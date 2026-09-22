use std::sync::Arc;

use query_engine::{
    admission::{AdmittedPlan, AttemptCapabilityEvidence},
    error::QueryErrorKind,
    exec::{admit_query_plan, preflight_query_plan},
    expr::Expr,
    plan::{
        GraphBfsPathOutput, GraphSimilarityParams, GraphVertexPairInput, NodeId, QueryPlan,
        graph_result_schema_for_vertex_type,
    },
};

use super::{
    BinaryOp, CUGRAPH_EDGE_DESTINATION_COLUMN, CUGRAPH_EDGE_ID_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN,
    CUGRAPH_EDGE_WEIGHT_COLUMN, CUGRAPH_MASS_COLUMN, CUGRAPH_MOBILITY_COLUMN, CUGRAPH_RADIUS_COLUMN,
    CUGRAPH_VALUE_COLUMN, CUGRAPH_VERTEX_COLUMN, CUGRAPH_X_COLUMN, CUGRAPH_Y_COLUMN, CaseWhen, CugraphAlgorithm,
    CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBetweennessSeeds, CugraphBfsOutputMode, CugraphGpuFrame,
    CugraphGraphConstructionPolicy, CugraphKCoreDegreeType, CugraphResultPlanOptions, CugraphVertexSelector,
    DataFusionError, DataType, Error, ErrorCode, ExecutionContext, ExecutionPlan, GpuDataFrame, GpuResidentInputMap,
    GpuResidentSource, GpuResultHandle, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec,
    GraphBetweennessCentralityParams, GraphBfsEndpointPolicy, GraphBfsLeader, GraphBfsParams, GraphConstructionOptions,
    GraphConstructionPolicy, GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams,
    GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings, GraphEigenvectorCentralityParams,
    GraphExecutionMetrics, GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2Params,
    GraphForceAtlas2VertexAttributesInput, GraphHitsParams, GraphInDegreesAllParams, GraphKCoreDegreeType,
    GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams, GraphMstParams,
    GraphOutDegreesAllParams, GraphPageRankParams, GraphPersonalizedPageRankParams, GraphSccParams,
    GraphSpectralClusteringParams, GraphSsspParams, GraphSsspSource, GraphTriangleCountAllParams, GraphWccParams,
    LimitSpec, LiteralValue, NativeExpr, NativeGraphBetweennessSeeds, NativeGraphBfsPredicateInputs,
    NativeGraphEdgePredicateInput, NativeGraphPageRankPersonalizationInput, NativeGraphPredicateInput,
    ProjectionOutput, QueryExecutionRequest, QueryOutput, QueryPlanBuilder, Result, SchemaRef, SortExpr, SortSpec,
    SourceNode, add_logical_mapping_plan, decode_logical_graph_result, execute_query_plan,
    finalize_graph_query_outcome, graph_uses_logical_mapping, node::projected_schema, plan,
};

/// The graph-fragment definition shared by preflight, plan construction, and
/// capability evidence: algorithm spec, edge and predicate schemas with row
/// cardinality evidence, result options, and the shape bounds the planner
/// inserted. Every cugraph plan consumer binds the same definition; GPU
/// frames, resident-input provenance, and metrics stay separate concerns.
#[derive(Clone)]
pub(super) struct CugraphPlanDefinition {
    pub(super) spec: CugraphAlgorithmSpec,
    pub(super) edge_schema: SchemaRef,
    pub(super) predicate_schemas: Vec<SchemaRef>,
    pub(super) predicate_row_upper_bounds: Vec<Option<u64>>,
    pub(super) result_options: CugraphResultPlanOptions,
    pub(super) edge_normalize_inserted: bool,
    pub(super) edge_row_upper_bound: Option<u64>,
    pub(super) edge_vertex_upper_bound: Option<u64>,
}

pub(super) fn preflight_cugraph_query_plan(definition: &CugraphPlanDefinition) -> Result<()> {
    let definition = CugraphPlanDefinition {
        predicate_row_upper_bounds: vec![None; definition.predicate_schemas.len()],
        ..definition.clone()
    };
    let plan = cugraph_schema_query_plan(&definition)?;
    preflight_query_plan(&plan).map_err(|error| {
        if error.kind() == QueryErrorKind::Unsupported {
            return DataFusionError::from(
                Error::new(
                    ErrorCode::RequiredGpuOperationUnsupported,
                    "algeon_datafusion.cugraph_sql.preflight",
                    "required cuGraph GPU operation is unsupported by the native engine",
                )
                .with_fact("native_code", error.code())
                .with_fact("native_operation", error.operation())
                .with_source(error),
            );
        }
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.preflight",
            "native cuGraph query-plan preflight failed",
            error,
        ))
    })
}

pub(super) fn execute_cugraph_query_to_gpu_with_options(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    definition: &CugraphPlanDefinition,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
    singleflight_leader: Option<&mut GraphBfsLeader>,
) -> Result<GpuResultHandle> {
    let graph_context = graph_algorithm_execution_context(context)?;
    let plan = cugraph_query_plan(&graph_context, definition, edge_frame, predicate_frames)?;
    plan.inputs.get(CUGRAPH_EDGE_INPUT_NAME).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.cugraph_sql.execute_gpu",
            "the lowered cuGraph plan is missing its GPU-resident edge input",
        )
        .with_fact("reason", "cugraph_edge_input_missing")
    })?;
    if let Some(leader) = singleflight_leader {
        leader.record_traversal_submission();
    }
    let outcome = execute_query_plan(
        &graph_context,
        QueryExecutionRequest::admitted(&plan.admitted).with_gpu_inputs(&plan.inputs),
    );
    let output = finalize_graph_query_outcome(
        metrics,
        outcome,
        "algeon_datafusion.cugraph_sql.execute_gpu",
        "native cuGraph SQL GPU-resident query execution failed",
    )?;
    let QueryOutput::GpuResident(handle) = output else {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.cugraph_sql.execute_gpu",
            "native cuGraph SQL query did not produce GPU-resident output",
        )
        .with_fact("result_sink", "host_arrow")
        .into());
    };
    Ok(handle)
}

pub(super) fn graph_algorithm_execution_context(context: &ExecutionContext) -> Result<ExecutionContext> {
    Ok(context.clone())
}

pub(super) struct CugraphQueryPlan {
    admitted: Arc<AdmittedPlan>,
    inputs: GpuResidentInputMap,
}

pub(super) const CUGRAPH_EDGE_INPUT_NAME: &str = "__algeon_datafusion_cugraph_edges";

pub(super) fn cugraph_side_input_name(index: usize) -> String {
    format!("__algeon_datafusion_cugraph_side_input_{index}")
}

pub(super) fn cugraph_query_plan(
    context: &ExecutionContext,
    definition: &CugraphPlanDefinition,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<CugraphQueryPlan> {
    let handle = GpuResultHandle::new(context, edge_frame, Arc::clone(&definition.edge_schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.gpu_input",
            "failed to bind cuGraph edge source as a GPU-resident native input",
            error,
        ))
    })?;
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(CUGRAPH_EDGE_INPUT_NAME.to_string(), handle);
    bind_predicate_gpu_inputs(context, &mut inputs, predicate_frames)?;
    let plan = cugraph_schema_query_plan(definition)?;
    // The graph plan is built here, after its inputs are bound, so it is
    // admitted here, once; a refusal is a typed error, never a CPU rerun.
    let admitted = admit_query_plan(&plan, context.statistical_aggregate_execution_mode(), context.planning_policy())
        .map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.execute_gpu",
            "native cuGraph SQL GPU-resident query execution failed",
            error,
        ))
    })?;
    Ok(CugraphQueryPlan { admitted, inputs })
}

pub(super) fn cugraph_schema_query_plan(definition: &CugraphPlanDefinition) -> Result<QueryPlan> {
    let CugraphPlanDefinition {
        spec,
        edge_schema,
        predicate_schemas,
        predicate_row_upper_bounds,
        result_options,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    } = definition.clone();

    if predicate_schemas.len() != predicate_row_upper_bounds.len() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.cugraph_sql.plan",
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
        .iter()
        .enumerate()
        .map(|(index, schema)| {
            let row_count_upper_bound = predicate_row_upper_bounds.get(index).copied().flatten();
            let source = GpuResidentSource::new(cugraph_side_input_name(index), Arc::clone(schema));
            let source = match row_count_upper_bound {
                Some(rows) => source.with_row_count_upper_bound(rows),
                None => source,
            };
            builder.add_source(SourceNode::GpuResident(source))
        })
        .collect::<Vec<_>>();
    let logical_mapping = graph_uses_logical_mapping(&spec, &edge_schema)?;
    spec.validate_logical_edge_id_capability(logical_mapping)?;
    let (execution_spec, graph_input, graph_edge_schema, graph_predicates, codebook) = if logical_mapping {
        let mapping = add_logical_mapping_plan(
            &mut builder,
            &spec,
            input,
            Arc::clone(&edge_schema),
            predicate_nodes,
            predicate_schemas,
        )?;
        (
            mapping.execution_spec,
            mapping.encoded_edge,
            mapping.encoded_edge_schema,
            mapping.encoded_predicates,
            Some(mapping.codebook),
        )
    } else {
        (spec.clone(), input, Arc::clone(&edge_schema), predicate_nodes, None)
    };
    let bfs_path_output_rows =
        bfs_path_output_row_bound(&execution_spec, edge_vertex_upper_bound, edge_row_upper_bound)?;
    let native_vertex_type = graph_edge_schema.field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN)?.data_type();
    let graph_spec = native_graph_spec(
        &execution_spec,
        edge_normalize_inserted,
        &graph_predicates,
        bfs_path_output_rows,
        native_vertex_type,
    );
    let mut root_schema = Arc::clone(&graph_spec.output_schema);
    let mut root = builder.try_add_graph_algorithm(graph_input, graph_spec).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.plan",
            "failed to build native cuGraph query plan",
            error,
        ))
    })?;
    if let Some(outputs) = bfs_normalized_output_projection(&execution_spec, &graph_edge_schema)? {
        root = builder.add_projection(root, outputs);
        root_schema = Arc::clone(&execution_spec.output_schema);
    }
    if logical_mapping && let Some(sort) = &result_options.subsumed_top_k_sort {
        root = builder.add_sort(root, sort.clone(), Arc::clone(&root_schema));
    }
    if logical_mapping && let Some(limit) = result_options.limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    if let Some(codebook) = codebook {
        (root, root_schema) = decode_logical_graph_result(&mut builder, &spec, root, root_schema, codebook)?;
        if spec.is_bfs_path_output() {
            root = builder.add_sort(
                root,
                SortSpec::new(vec![SortExpr::new(NativeExpr::column("path_index"), false, false)]),
                Arc::clone(&root_schema),
            );
        }
    }
    if logical_mapping && let Some(sort) = &result_options.subsumed_top_k_sort {
        root = builder.add_sort(root, sort.clone(), Arc::clone(&root_schema));
    }
    if let Some(projection) = &result_options.projection {
        root_schema = projected_schema(&root_schema, Some(projection))?;
        let outputs = projection
            .iter()
            .map(|index| {
                let field = spec
                    .output_schema
                    .fields()
                    .get(*index)
                    .map(|field| field.as_ref().clone())
                    .ok_or_else(|| plan(format!("cuGraph output projection index {index} is out of range")))?;
                Ok(ProjectionOutput::new(Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root = builder.add_projection(root, outputs);
    }
    if !logical_mapping && let Some(sort) = &result_options.subsumed_top_k_sort {
        root = builder.add_sort(root, sort.clone(), Arc::clone(&root_schema));
    }
    if !logical_mapping && let Some(limit) = result_options.limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    Ok(builder.finish_gpu_resident(root))
}

pub(super) fn cugraph_execution_capability_evidence(
    definition: &CugraphPlanDefinition,
) -> Result<(QueryPlan, AttemptCapabilityEvidence)> {
    let plan = cugraph_schema_query_plan(definition)?;
    let mut graph_evidence = AttemptCapabilityEvidence::for_query_plan(&plan);
    graph_evidence.attest_cugraph_graph_execution_scope(&plan);
    Ok((plan, graph_evidence))
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
                "algeon_datafusion.cugraph_sql.gpu_input",
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
    predicate_nodes: &[NodeId],
    bfs_path_output_rows: Option<u64>,
    vertex_type: &DataType,
) -> GraphAlgorithmSpec {
    let mut edge_columns = GraphEdgeColumnBindings::new(
        CUGRAPH_EDGE_SOURCE_COLUMN.to_string(),
        CUGRAPH_EDGE_DESTINATION_COLUMN.to_string(),
        spec.has_weight.then(|| CUGRAPH_EDGE_WEIGHT_COLUMN.to_string()),
    );
    if spec.has_edge_id {
        edge_columns = edge_columns.with_edge_id(CUGRAPH_EDGE_ID_COLUMN.to_string());
    }
    let algorithm = native_algorithm(spec.algorithm);
    let params = native_algorithm_params(&spec.params, predicate_nodes, bfs_path_output_rows);
    let output_schema = graph_result_schema_for_vertex_type(algorithm, &params, vertex_type);
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
    edge_schema: &SchemaRef,
) -> Option<GraphAlgorithmSpec> {
    let mut builder = QueryPlanBuilder::new();
    let predicate_nodes = predicate_sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
                format!("__algeon_datafusion_cugraph_placement_hint_{index}"),
                source.schema(),
            )))
        })
        .collect::<Vec<_>>();
    let vertex_type = edge_schema.field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN).ok()?.data_type();
    matches!(vertex_type, DataType::Int32 | DataType::Int64)
        .then(|| native_graph_spec(spec, edge_normalize_inserted, &predicate_nodes, None, vertex_type))
}

fn bfs_path_output_row_bound(
    spec: &CugraphAlgorithmSpec,
    edge_vertex_upper_bound: Option<u64>,
    edge_row_upper_bound: Option<u64>,
) -> Result<Option<u64>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Ok(None);
    }
    if let Some(depth) = params.depth_limit {
        let depth = u64::try_from(depth).map_err(|_| plan("BFS path depth_limit must be non-negative"))?;
        return depth.checked_add(1).map(Some).ok_or_else(|| plan("BFS path output row bound overflowed u64"));
    }
    edge_vertex_upper_bound
        .or_else(|| edge_row_upper_bound.and_then(|rows| rows.checked_mul(2)))
        .map(|rows| rows.max(1))
        .map(Some)
        .ok_or_else(|| plan("BFS path output requires a depth or graph vertex row bound"))
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
    let predecessor_type = spec.output_schema.field_with_name("predecessor")?.data_type().clone();
    let predecessor_absent = match &predecessor_type {
        DataType::Int32 => LiteralValue::Int32(-1),
        DataType::Int64 => LiteralValue::Int64(-1),
        ref other => return Err(plan(format!("cuGraph BFS predecessor has unsupported type {other:?}"))),
    };
    let distance_is_unreachable = NativeExpr::binary(
        NativeExpr::column("distance"),
        BinaryOp::Eq,
        NativeExpr::literal(LiteralValue::Int64(unreachable_sentinel)),
    );
    let predecessor_is_absent =
        NativeExpr::binary(NativeExpr::column("predecessor"), BinaryOp::Eq, NativeExpr::literal(predecessor_absent));

    let fields = spec.output_schema.fields();
    let mut outputs = vec![
        ProjectionOutput::new(NativeExpr::column("vertex"), fields[0].as_ref().clone()),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(distance_is_unreachable, NativeExpr::literal(LiteralValue::Null(DataType::Int64)))],
                NativeExpr::column("distance"),
            ),
            fields[1].as_ref().clone(),
        ),
        ProjectionOutput::new(
            NativeExpr::case(
                vec![CaseWhen::new(predecessor_is_absent, NativeExpr::literal(LiteralValue::Null(predecessor_type)))],
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

pub(super) fn bfs_unreachable_distance_sentinel(_spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<i64> {
    let source_field = edge_schema.field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN).map_err(|_| {
        plan("cuGraph BFS output normalization could not find canonical source column 'src' in edge schema")
    })?;
    match source_field.data_type() {
        DataType::Int32 => Ok(i64::from(i32::MAX)),
        DataType::Int64 => Ok(i64::MAX),
        other => Err(plan(format!(
            "cuGraph BFS output normalization requires Int32 or Int64 source vertices, found {other:?}"
        ))),
    }
}

pub(super) fn native_algorithm(algorithm: CugraphAlgorithm) -> GraphAlgorithm {
    algorithm.native_algorithm()
}

pub(super) fn native_algorithm_params(
    params: &CugraphAlgorithmParams,
    predicate_nodes: &[NodeId],
    bfs_path_output_rows: Option<u64>,
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
                    CUGRAPH_VERTEX_COLUMN,
                    CUGRAPH_VALUE_COLUMN,
                ),
            })
        },
        CugraphAlgorithmParams::Bfs(params) => {
            GraphAlgorithmParams::Bfs(Box::new(native_bfs_params(params, predicate_nodes, bfs_path_output_rows)))
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
                seeds: params.seeds.as_ref().map(|seeds| native_betweenness_seeds(seeds, predicate_nodes)),
                k: params.k,
                normalized: params.normalized,
                include_endpoints: params.include_endpoints,
                exact_vertex_threshold: params.exact_vertex_threshold,
            })
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
                seeds: params.seeds.as_ref().map(|seeds| native_betweenness_seeds(seeds, predicate_nodes)),
                k: params.k,
                normalized: params.normalized,
                exact_vertex_threshold: params.exact_vertex_threshold,
            })
        },
        CugraphAlgorithmParams::Sssp(params) => {
            let source_vertex = match &params.source_vertex {
                CugraphVertexSelector::Int64(value) => GraphSsspSource::I64(*value),
                CugraphVertexSelector::Utf8(_) => GraphSsspSource::Input(NativeGraphPredicateInput::new(
                    predicate_nodes.first().copied().expect("SSSP source node missing"),
                    "__source_vertex",
                )),
            };
            GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex, cutoff: params.cutoff })
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
        CugraphAlgorithmParams::ForceAtlas2(params) => {
            let mut next = predicate_nodes.iter().copied();
            let vertex_attributes = params.vertex_attributes.as_ref().map(|input| {
                GraphForceAtlas2VertexAttributesInput::new(
                    next.next().expect("ForceAtlas2 vertex-attributes node missing"),
                    CUGRAPH_VERTEX_COLUMN,
                    input.has_radius.then_some(CUGRAPH_RADIUS_COLUMN),
                    input.has_mobility.then_some(CUGRAPH_MOBILITY_COLUMN),
                    input.has_mass.then_some(CUGRAPH_MASS_COLUMN),
                )
            });
            let initial_positions = params.initial_positions.as_ref().map(|_input| {
                GraphForceAtlas2InitialPositionsInput::new(
                    next.next().expect("ForceAtlas2 initial-positions node missing"),
                    CUGRAPH_VERTEX_COLUMN,
                    CUGRAPH_X_COLUMN,
                    CUGRAPH_Y_COLUMN,
                )
            });
            GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params {
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
                vertex_attributes,
                initial_positions,
            })
        },
        CugraphAlgorithmParams::Similarity(_params) => {
            let vertex_pairs_node = predicate_nodes.first().copied().expect("similarity vertex-pairs node missing");
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
                vertex_pairs_node,
                "first",
                "second",
            )))
        },
    }
}

fn native_betweenness_seeds(
    seeds: &CugraphBetweennessSeeds,
    predicate_nodes: &[NodeId],
) -> NativeGraphBetweennessSeeds {
    match seeds.vertices.first() {
        Some(CugraphVertexSelector::Int64(_)) => NativeGraphBetweennessSeeds::I64(
            seeds
                .vertices
                .iter()
                .map(|vertex| match vertex {
                    CugraphVertexSelector::Int64(value) => *value,
                    CugraphVertexSelector::Utf8(_) => unreachable!("parser rejects mixed betweenness seed domains"),
                })
                .collect(),
        ),
        Some(CugraphVertexSelector::Utf8(_)) => NativeGraphBetweennessSeeds::Input {
            input: NativeGraphPredicateInput::new(
                predicate_nodes.first().copied().expect("betweenness seed node missing"),
                "__seed_vertex",
            ),
            row_count_upper_bound: seeds.vertices.len() as u64,
        },
        None => unreachable!("parser rejects empty betweenness seed lists"),
    }
}

pub(super) fn native_bfs_params(
    params: &super::CugraphBfsParams,
    predicate_nodes: &[NodeId],
    path_output_rows: Option<u64>,
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
        include_vertices: params.predicates.include_vertices.as_ref().map(|_input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), CUGRAPH_VERTEX_COLUMN)
        }),
        exclude_vertices: params.predicates.exclude_vertices.as_ref().map(|_input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), CUGRAPH_VERTEX_COLUMN)
        }),
        target_vertices: params.predicates.target_vertices.as_ref().map(|_input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), CUGRAPH_VERTEX_COLUMN)
        }),
        include_edge_ids: params.predicates.include_edge_ids.as_ref().map(|_input| {
            NativeGraphPredicateInput::new(next.next().expect("predicate node missing"), CUGRAPH_EDGE_ID_COLUMN)
        }),
        include_edges: params.predicates.include_edges.as_ref().map(|input| {
            NativeGraphEdgePredicateInput::new(
                next.next().expect("predicate node missing"),
                CUGRAPH_EDGE_SOURCE_COLUMN,
                CUGRAPH_EDGE_DESTINATION_COLUMN,
                input.has_edge_id.then_some(CUGRAPH_EDGE_ID_COLUMN),
            )
        }),
    };
    native.return_target_info = params.output.return_target_info;
    if params.output.mode == CugraphBfsOutputMode::Path {
        native.endpoint_policy = GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
        native.path_output = Some(GraphBfsPathOutput::new(
            path_output_rows.expect("bounded BFS path plan must carry an output row bound"),
        ));
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
        CugraphGraphConstructionPolicy, CugraphGraphOptions, exec::PlanNode, result_schema_for_params,
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
            has_weight: false,
            has_edge_id: false,
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

    fn assert_admission_evidence_matches_runtime_prepared_plan(
        spec: &CugraphAlgorithmSpec,
        edge_schema: SchemaRef,
        projection: Option<&Vec<usize>>,
        limit: Option<usize>,
    ) {
        let predicate_row_upper_bounds = Vec::new();
        let result_options = CugraphResultPlanOptions::new(projection.cloned(), limit);
        let definition = CugraphPlanDefinition {
            spec: spec.clone(),
            edge_schema: Arc::clone(&edge_schema),
            predicate_schemas: Vec::new(),
            predicate_row_upper_bounds,
            result_options,
            edge_normalize_inserted: false,
            edge_row_upper_bound: None,
            edge_vertex_upper_bound: None,
        };
        let (_, admission_evidence) =
            cugraph_execution_capability_evidence(&definition).expect("cuGraph plan must expose a capability evidence");

        let runtime_plan = cugraph_schema_query_plan(&definition).expect("prepared runtime graph plan must build");
        let mut runtime_evidence = AttemptCapabilityEvidence::for_query_plan(&runtime_plan);
        runtime_evidence.attest_cugraph_graph_execution_scope(&runtime_plan);

        assert_eq!(
            admission_evidence, runtime_evidence,
            "admission must attest the exact prepared graph and logical-output fragments runtime executes"
        );
    }

    #[test]
    fn admission_evidence_matches_runtime_bfs_logical_and_path_prepared_shapes() {
        let projection = vec![0];
        assert_admission_evidence_matches_runtime_prepared_plan(
            &bfs_spec(CugraphBfsSource::scalar_utf8("alice"), CugraphBfsOutputMode::Normalized),
            edge_schema(DataType::Utf8),
            Some(&projection),
            Some(2),
        );
        assert_admission_evidence_matches_runtime_prepared_plan(
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
        let definition = CugraphPlanDefinition {
            spec,
            edge_schema,
            predicate_schemas: Vec::new(),
            predicate_row_upper_bounds: Vec::new(),
            result_options: CugraphResultPlanOptions::default(),
            edge_normalize_inserted: false,
            edge_row_upper_bound: Some(5),
            edge_vertex_upper_bound: Some(11),
        };
        let (admission_plan, admission_evidence) =
            cugraph_execution_capability_evidence(&definition).expect("admission plan must build");
        let runtime_plan =
            cugraph_schema_query_plan(&definition).expect("runtime plan must build with the same source bounds");

        let source_metadata = |plan: &QueryPlan| {
            plan.nodes().iter().find_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) if source.input_name == CUGRAPH_EDGE_INPUT_NAME => {
                        Some((source.row_count_upper_bound, source.graph_vertex_upper_bound))
                    },
                    _ => None,
                },
                _ => None,
            })
        };
        assert_eq!(source_metadata(&admission_plan), Some((Some(5), Some(11))));
        assert_eq!(source_metadata(&runtime_plan), Some((Some(5), Some(11))));
        let mut expected_runtime_evidence = AttemptCapabilityEvidence::for_query_plan(&runtime_plan);
        expected_runtime_evidence.attest_cugraph_graph_execution_scope(&runtime_plan);
        assert_eq!(
            admission_evidence, expected_runtime_evidence,
            "runtime actual rows must not replace the admitted source-bound scope metadata"
        );
    }
}
