//! Registry-driven CPU-only contract rows for cuGraph algorithms.
//!
//! Each row asserts one stable contract surface — stable output schema,
//! capability admission, bounded rejection — for every algorithm
//! currently in the cuGraph operation registry. New algorithms must
//! extend the registry; the rows here iterate it automatically and do
//! not need per-algorithm test files.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion_nexus::cugraph_sql::{CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm};
use nexus_query_engine::{
    admission::{AttemptCapabilityEvidence, CUGRAPH_GRAPH_EXECUTION_CAPABILITY},
    capability::{CapabilityDecision, NodeOperator},
    plan::{
        GRAPH_OPERATION_REGISTRY, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec,
        GraphBetweennessCentralityParams, GraphBfsParams, GraphConstructionOptions, GraphCoreNumberParams,
        GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings,
        GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams, GraphInDegreesAllParams,
        GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams, GraphMstParams,
        GraphOutDegreesAllParams, GraphPageRankParams, GraphPageRankPersonalizationInput,
        GraphPersonalizedPageRankParams, GraphSccParams, GraphSimilarityParams, GraphSpectralClusteringParams,
        GraphSsspParams, GraphTriangleCountAllParams, GraphVertexIdTypeSupport, GraphVertexPairInput, GraphWccParams,
        PlanNode, QueryPlan, QueryPlanBuilder, graph_result_schema_for_vertex_type,
    },
    source::{GpuResidentSource, SourceNode},
};

/// Build a default params variant for one registered algorithm.
pub fn default_params(algorithm: GraphAlgorithm) -> GraphAlgorithmParams {
    match algorithm {
        GraphAlgorithm::PageRank => GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        GraphAlgorithm::PersonalizedPageRank => {
            GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams::default())
        },
        GraphAlgorithm::Bfs => GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        GraphAlgorithm::Louvain => GraphAlgorithmParams::Louvain(GraphLouvainParams::default()),
        GraphAlgorithm::EigenvectorCentrality => {
            GraphAlgorithmParams::EigenvectorCentrality(GraphEigenvectorCentralityParams::default())
        },
        GraphAlgorithm::KatzCentrality => GraphAlgorithmParams::KatzCentrality(GraphKatzCentralityParams::default()),
        GraphAlgorithm::Hits => GraphAlgorithmParams::Hits(GraphHitsParams::default()),
        GraphAlgorithm::BetweennessCentrality => {
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams::default())
        },
        GraphAlgorithm::EdgeBetweennessCentrality => {
            GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams::default())
        },
        GraphAlgorithm::Sssp => GraphAlgorithmParams::Sssp(GraphSsspParams {
            source_vertex: nexus_query_engine::plan::GraphSsspSource::I64(0),
            cutoff: None,
        }),
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
        GraphAlgorithm::Leiden => GraphAlgorithmParams::Leiden(GraphLeidenParams::default()),
        GraphAlgorithm::Ecg => GraphAlgorithmParams::Ecg(GraphEcgParams::default()),
        GraphAlgorithm::SpectralModularityMaximization => {
            GraphAlgorithmParams::SpectralModularityMaximization(GraphSpectralClusteringParams::default())
        },
        GraphAlgorithm::ForceAtlas2 => GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::default())
        },
    }
}

fn edge_schema(algorithm: GraphAlgorithm) -> Arc<Schema> {
    let vertex_id_type = match algorithm.vertex_id_type_support() {
        GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType => DataType::Int32,
        GraphVertexIdTypeSupport::Int32OrInt64 => DataType::Int64,
    };
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_id_type.clone(), false),
        Field::new("dst", vertex_id_type, false),
        Field::new("weight", DataType::Float64, false),
    ]))
}

fn personalization_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

fn similarity_pair_schema(algorithm: GraphAlgorithm) -> Arc<Schema> {
    let vertex_id_type = match algorithm.vertex_id_type_support() {
        GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType => DataType::Int32,
        GraphVertexIdTypeSupport::Int32OrInt64 => DataType::Int64,
    };
    Arc::new(Schema::new(vec![
        Field::new("first", vertex_id_type.clone(), false),
        Field::new("second", vertex_id_type, false),
    ]))
}

/// Build the canonical CPU-only QueryPlan for the contract checks.
pub fn graph_plan(algorithm: GraphAlgorithm) -> QueryPlan {
    let edge_schema = edge_schema(algorithm);
    let vertex_type = edge_schema.field_with_name("src").expect("contract edge schema has src").data_type().clone();
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", edge_schema)));
    let personalization =
        if algorithm == GraphAlgorithm::PersonalizedPageRank {
            Some(builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
                "personalization",
                personalization_schema(),
            ))))
        } else {
            None
        };
    let vertex_pairs = algorithm.is_similarity().then(|| {
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "vertex_pairs",
            similarity_pair_schema(algorithm),
        )))
    });
    let params = if let Some(personalization) = personalization {
        GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
            pagerank: GraphPageRankParams::default(),
            personalization: GraphPageRankPersonalizationInput::new(personalization, "vertex", "value"),
        })
    } else if let Some(vertex_pairs) = vertex_pairs {
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
            vertex_pairs,
            "first",
            "second",
        )))
    } else {
        default_params(algorithm)
    };
    let output_schema = graph_result_schema_for_vertex_type(algorithm, &params, &vertex_type);
    let spec = GraphAlgorithmSpec::new(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        algorithm.definition().default_graph_options(&GraphConstructionOptions::default()),
        params,
        output_schema,
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
fn registry_algorithms_admit_under_execution_capability_without_a_memory_veto() {
    for entry in GRAPH_OPERATION_REGISTRY {
        let plan = graph_plan(entry.algorithm);
        let analysis = plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context());
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
        let mut evidence = AttemptCapabilityEvidence::for_query_plan(&plan);
        evidence.attest_cugraph_graph_execution_scope(&plan);
        assert!(evidence.required_capabilities().contains(&CUGRAPH_GRAPH_EXECUTION_CAPABILITY));
        evidence
            .validate_covers_plan(&plan)
            .unwrap_or_else(|error| panic!("graph capability evidence failed for {}: {error}", entry.name));
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
            CugraphAlgorithm::BetweennessCentrality,
            CugraphAlgorithm::EdgeBetweennessCentrality,
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
            CugraphAlgorithm::PersonalizedPageRank,
            CugraphAlgorithm::Leiden,
            CugraphAlgorithm::Ecg,
            CugraphAlgorithm::SpectralModularityMaximization,
            CugraphAlgorithm::ForceAtlas2,
            CugraphAlgorithm::Jaccard,
            CugraphAlgorithm::Sorensen,
            CugraphAlgorithm::Overlap,
            CugraphAlgorithm::Cosine,
        ],
        "cuGraph SQL operation registry order is part of the public surface",
    );
}
