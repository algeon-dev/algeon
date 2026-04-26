//! SQL-facing cuGraph table-function contracts.

mod exec;
mod metadata;
#[doc(hidden)]
pub mod parity;
mod parser;
mod table_function;
mod types;

use datafusion::{
    common::{DataFusionError, Result, ScalarValue, TableReference},
    logical_expr::Expr,
};
pub use exec::CugraphAlgoExec;
pub use parser::parse_cugraph_algorithm_spec;
pub use table_function::{CugraphAlgoTableProvider, cugraph_table_functions, register_cugraph_udtfs};
pub use types::{
    CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
    CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_ECG_FUNCTION,
    CUGRAPH_EDGE_BETWEENNESS_CENTRALITY_FUNCTION, CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION,
    CUGRAPH_FORCE_ATLAS2_FUNCTION, CUGRAPH_HITS_FUNCTION, CUGRAPH_IN_DEGREES_ALL_FUNCTION, CUGRAPH_K_CORE_FUNCTION,
    CUGRAPH_KATZ_CENTRALITY_FUNCTION, CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION,
    CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION, CUGRAPH_OPERATION_REGISTRY,
    CUGRAPH_OUT_DEGREES_ALL_FUNCTION, CUGRAPH_PAGERANK_FUNCTION, CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION,
    CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION,
    CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    CUGRAPH_VALIDATE_CALL_FUNCTION, CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION, CugraphAlgorithm,
    CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBetweennessCentralityParams, CugraphBetweennessSeeds,
    CugraphBfsOutputMode, CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsPredicateInputs, CugraphBfsSource,
    CugraphBfsSourceVertex, CugraphCoreNumberParams, CugraphDegreesAllParams, CugraphEcgParams,
    CugraphEdgeBetweennessCentralityParams, CugraphEigenvectorCentralityParams, CugraphForceAtlas2Params,
    CugraphGraphConstructionPolicy, CugraphGraphEdgePredicateTable, CugraphGraphOptions, CugraphGraphPredicateTable,
    CugraphHitsParams, CugraphInDegreesAllParams, CugraphKCoreDegreeType, CugraphKCoreParams,
    CugraphKatzCentralityParams, CugraphLeidenParams, CugraphLouvainParams, CugraphMstParams,
    CugraphOperationDefinition, CugraphOutDegreesAllParams, CugraphPageRankParams, CugraphPageRankPersonalizationTable,
    CugraphPersonalizedPageRankParams, CugraphSccParams, CugraphSpectralClusteringParams, CugraphSqlConfig,
    CugraphSsspParams, CugraphTableReference, CugraphTriangleCountAllParams, CugraphWccParams,
    GpuIslandHostBoundaryConfig, bfs_result_schema, result_schema_for_params, stable_result_schema,
};
pub(crate) use types::{
    CugraphVertexIdDomain, bfs_result_schema_for_vertex_domain, validate_vertex_id_domain_support_for_call,
    vertex_id_domain_for_call,
};

use crate::error::{Error, ErrorCode};

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}

fn validate_bfs_logical_edge_id_capability(spec: &CugraphAlgorithmSpec, logical_mapping: bool) -> Result<()> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(());
    };
    if !logical_mapping || (spec.edge_id_col.is_none() && params.predicates.include_edge_ids.is_none()) {
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::RequiredGpuOperationUnsupported,
        "datafusion_nexus.cugraph_sql.logical_mapping.preflight",
        "cuGraph BFS logical vertex-id mapping does not support edge-id inputs",
    )
    .with_fact("edge_id_col", spec.edge_id_col.is_some())
    .with_fact("include_edge_ids", params.predicates.include_edge_ids.is_some())
    .into())
}

fn datafusion_table_reference(table: &CugraphTableReference) -> TableReference {
    match table.parts() {
        [table] => TableReference::bare(table.as_str()),
        [schema, table] => TableReference::partial(schema.as_str(), table.as_str()),
        [catalog, schema, table] => TableReference::full(catalog.as_str(), schema.as_str(), table.as_str()),
        _ => unreachable!("CugraphTableReference::parse validates one to three identifier parts"),
    }
}

fn literal_string_arg(args: &[Expr], index: usize, name: &str) -> Result<String> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(value.clone()),
            _ => Err(plan_error(format!("{name} must be a non-null string literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}
