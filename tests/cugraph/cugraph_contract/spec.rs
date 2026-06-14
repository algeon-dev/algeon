//! Registry-driven CPU-only contract rows for cuGraph algorithms.
//!
//! Each row asserts one stable contract surface — stable output schema,
//! capability admission, strict-bounded rejection — for every algorithm
//! currently in the cuGraph operation registry. New algorithms must
//! extend the registry; the rows here iterate it automatically and do
//! not need per-algorithm test files.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion_nexus::{CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm, stable_result_schema};
use nexus_query_engine::{
    GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY, GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT,
    GRAPH_OPERATION_REGISTRY, GpuResidentSource, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec,
    GraphBfsParams, GraphConstructionOptions, GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams,
    GraphEdgeColumnBindings, GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams,
    GraphInDegreesAllParams, GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams,
    GraphMstParams, GraphOutDegreesAllParams, GraphPageRankParams, GraphSccParams, GraphSpectralClusteringParams,
    GraphSsspParams, GraphTriangleCountAllParams, GraphWccParams, PlanNode, QueryPlan, QueryPlanBuilder, SourceNode,
    StrictBoundedContractReport, StrictBoundedExecutionMode,
    capability::{CapabilityDecision, ContractQuality, ContractStatus, NodeOperator},
    stable_graph_result_schema,
};

/// Build a default params variant for one registered algorithm.
pub fn default_params(algorithm: GraphAlgorithm) -> GraphAlgorithmParams {
    match algorithm {
        GraphAlgorithm::PageRank => GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        GraphAlgorithm::Bfs => GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        GraphAlgorithm::Louvain => GraphAlgorithmParams::Louvain(GraphLouvainParams::default()),
        GraphAlgorithm::EigenvectorCentrality => {
            GraphAlgorithmParams::EigenvectorCentrality(GraphEigenvectorCentralityParams::default())
        },
        GraphAlgorithm::KatzCentrality => GraphAlgorithmParams::KatzCentrality(GraphKatzCentralityParams::default()),
        GraphAlgorithm::Hits => GraphAlgorithmParams::Hits(GraphHitsParams::default()),
        GraphAlgorithm::Sssp => GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: 0, cutoff: None }),
        GraphAlgorithm::DegreesAll => GraphAlgorithmParams::DegreesAll(GraphDegreesAllParams),
        GraphAlgorithm::InDegreesAll => GraphAlgorithmParams::InDegreesAll(GraphInDegreesAllParams),
        GraphAlgorithm::OutDegreesAll => GraphAlgorithmParams::OutDegreesAll(GraphOutDegreesAllParams),
        GraphAlgorithm::WeaklyConnectedComponents => GraphAlgorithmParams::WeaklyConnectedComponents(GraphWccParams),
        GraphAlgorithm::StronglyConnectedComponents => {
            GraphAlgorithmParams::StronglyConnectedComponents(GraphSccParams)
        },
        GraphAlgorithm::TriangleCountAll => GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
        GraphAlgorithm::CoreNumber => GraphAlgorithmParams::CoreNumber(GraphCoreNumberParams::default()),
        GraphAlgorithm::KCore => GraphAlgorithmParams::KCore(GraphKCoreParams::default()),
        GraphAlgorithm::MinimumSpanningTree => GraphAlgorithmParams::MinimumSpanningTree(GraphMstParams),
        GraphAlgorithm::PageRankStrict => GraphAlgorithmParams::PageRankStrict(GraphPageRankParams::default()),
        GraphAlgorithm::Leiden => GraphAlgorithmParams::Leiden(GraphLeidenParams::default()),
        GraphAlgorithm::Ecg => GraphAlgorithmParams::Ecg(GraphEcgParams::default()),
        GraphAlgorithm::SpectralModularityMaximization => {
            GraphAlgorithmParams::SpectralModularityMaximization(GraphSpectralClusteringParams::default())
        },
        GraphAlgorithm::BalancedCutClustering => {
            GraphAlgorithmParams::BalancedCutClustering(GraphSpectralClusteringParams::default())
        },
        GraphAlgorithm::ForceAtlas2 => GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
    }
}

fn edge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]))
}

/// Build the canonical CPU-only QueryPlan for the contract checks.
pub fn graph_plan(algorithm: GraphAlgorithm) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", edge_schema())));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        default_params(algorithm),
    );
    let graph = builder.add_graph_algorithm(input, spec);
    builder.finish_with_host_sink(graph)
}

#[test]
fn registry_native_and_sql_inventories_have_matching_lengths() {
    assert_eq!(
        GRAPH_OPERATION_REGISTRY.len(),
        CUGRAPH_OPERATION_REGISTRY.len(),
        "native graph operation registry and the SQL-side mirror must have equal length",
    );
}

#[test]
fn registry_algorithm_sql_output_schemas_agree_with_native_engine() {
    for entry in CUGRAPH_OPERATION_REGISTRY {
        let sql_schema = stable_result_schema(entry.algorithm);
        let native_schema = stable_graph_result_schema(entry.native_algorithm);
        assert_eq!(
            sql_schema.as_ref(),
            native_schema.as_ref(),
            "SQL-facing output schema for {} must mirror the native engine schema",
            entry.sql_function_name,
        );
    }
}

#[test]
fn registry_algorithms_admit_under_capability_with_strict_bounded_gate() {
    for entry in GRAPH_OPERATION_REGISTRY {
        let plan = graph_plan(entry.algorithm);
        let analysis = plan.capability_analysis();
        let node = analysis
            .nodes
            .iter()
            .find(|node| node.node_id == plan.root_id())
            .unwrap_or_else(|| panic!("expected graph algorithm capability row for {}", entry.name));
        assert_eq!(
            node.operator,
            NodeOperator::GraphAlgorithm { algorithm: entry.algorithm },
            "capability operator should match the registered algorithm for {}",
            entry.name,
        );
        assert_eq!(
            node.executable,
            CapabilityDecision::Satisfied,
            "capability admission must be Satisfied for {}",
            entry.name,
        );
        assert_eq!(
            node.local_requirements.len(),
            1,
            "graph algorithms expose exactly one strict-bounded reservation requirement; {} has {}",
            entry.name,
            node.local_requirements.len(),
        );
        let requirement = &node.local_requirements[0];
        assert_eq!(requirement.memory_boundary, GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY);
        assert_eq!(requirement.required_contract, GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT);
        assert_eq!(requirement.status, ContractStatus::BlockedByLowerLevel);
        assert_eq!(requirement.quality, ContractQuality::ExactPreflight);

        let report = StrictBoundedContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("strict bounded report build failed for {}: {error}", entry.name));
        assert_eq!(
            report.execution_mode(),
            StrictBoundedExecutionMode::UnsupportedStrictBounded,
            "strict bounded execution must stay rejected until lower-level memory contracts land for {}",
            entry.name,
        );
        assert!(
            report.gate_rows().iter().any(|row| {
                row.memory_boundary == GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY
                    && row.required_contract == GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT
                    && row.contract_status == ContractStatus::BlockedByLowerLevel
            }),
            "strict bounded report must carry the graph algorithm gate row for {}",
            entry.name,
        );
    }
}

#[test]
fn registry_algorithms_preserve_edge_normalize_provenance_under_default_policy() {
    for entry in GRAPH_OPERATION_REGISTRY {
        let plan = graph_plan(entry.algorithm);
        assert!(
            plan.contains_edge_normalize(),
            "default Python-cuGraph policy must keep EdgeNormalize for {}",
            entry.name,
        );
        let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
            panic!("expected GraphAlgorithm root for {}", entry.name);
        };
        assert_eq!(
            graph.spec.algorithm, entry.algorithm,
            "root spec must agree with registry entry for {}",
            entry.name,
        );
    }
}

#[test]
fn registry_pins_current_cugraph_sql_inventory() {
    // Pin the current SQL-side inventory in registry order so an unintentional
    // change shows up as a focused test failure here, separate from the cuGraph
    // SQL plumbing tests. Extend this list when a new algorithm lands; the
    // matching native-side pin lives in
    // `crates/nexus-query-engine/tests/native_contract_tests/graph_algorithm.rs`.
    let registered: Vec<_> = CUGRAPH_OPERATION_REGISTRY.iter().map(|e| e.algorithm).collect();
    assert_eq!(
        registered,
        vec![
            CugraphAlgorithm::PageRank,
            CugraphAlgorithm::Bfs,
            CugraphAlgorithm::Louvain,
            CugraphAlgorithm::EigenvectorCentrality,
            CugraphAlgorithm::KatzCentrality,
            CugraphAlgorithm::Hits,
            CugraphAlgorithm::Sssp,
            CugraphAlgorithm::DegreesAll,
            CugraphAlgorithm::InDegreesAll,
            CugraphAlgorithm::OutDegreesAll,
            CugraphAlgorithm::WeaklyConnectedComponents,
            CugraphAlgorithm::StronglyConnectedComponents,
            CugraphAlgorithm::TriangleCountAll,
            CugraphAlgorithm::CoreNumber,
            CugraphAlgorithm::KCore,
            CugraphAlgorithm::MinimumSpanningTree,
            CugraphAlgorithm::PageRankStrict,
            CugraphAlgorithm::Leiden,
            CugraphAlgorithm::Ecg,
            CugraphAlgorithm::SpectralModularityMaximization,
            CugraphAlgorithm::BalancedCutClustering,
            CugraphAlgorithm::ForceAtlas2,
        ],
        "cuGraph SQL operation registry order is part of the public surface",
    );
}
