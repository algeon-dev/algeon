use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use nexus_query_engine::{
    capability::{CapabilityDecision, ContractQuality, ContractStatus, NodeOperator},
    contract_report::CapabilityContractReport,
    error::QueryErrorCode,
    expr::{BinaryOp, Expr, LiteralValue},
    graph_memory::{
        GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, GraphCalibrationSource, GraphMemoryCalibrationOverride,
        GraphMemoryClass, GraphMemoryModel, GraphMemoryModelInput, GraphModelErrorRatio,
        GraphPairCardinalityAdmissionMode, GraphPairCardinalitySource, GraphReservationOutcome, GraphReservationReport,
        GraphSeedMode, GraphStatistics, GraphVertexEstimate, GraphVertexEstimateSource, graph_memory_class,
        graph_memory_estimates_for_plan, graph_memory_reservation_required, graph_runtime_statistics_probe_bytes,
        graph_vertex_estimate_probe_bytes, triangle_count_peak_chunk_min_degree_pricing_term,
    },
    mode::CapabilityEnvelopeStatus,
    plan::{
        GRAPH_ALGORITHM_CAPABILITY_LOWER_LEVEL_CONTRACTS, GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
        GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT, GRAPH_OPERATION_REGISTRY, GraphAlgorithm, GraphAlgorithmParams,
        GraphAlgorithmSpec, GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphBfsEndpointPolicy,
        GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
        GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings, GraphEdgePredicateInput, GraphLayoutRequirement,
        GraphLouvainParams, GraphOutputSchemaKind, GraphPageRankParams, GraphPredicateInput, GraphResidencyDescriptor,
        GraphResidencyOrientation, GraphResidencyPlanAnalysis, GraphSimilarityParams,
        GraphSourceFingerprintBypassReason, GraphSourceVertexRequirement, GraphTriangleCountAllParams,
        GraphVertexIdTypeSupport, GraphVertexPairInput, GraphWeightPolicy, PlanNode, ProjectionOutput, QueryPlan,
        QueryPlanBuilder, edge_plan_has_simple_directed_edge_topology, graph_residency_descriptor_for_edge_plan,
        planned_resident_graph_key, stable_graph_result_schema,
    },
    source::{
        CredentialMode, GpuResidentSource, LocalParquetFile, LocalParquetSource, ParquetFieldIdReadBinding,
        ParquetPositionDeleteFile, ResolvedParquetSource, SourceNode,
    },
};
#[cfg(not(feature = "cugraph"))]
use nexus_query_engine::{
    exec::execute_query_plan_with_gpu_inputs,
    runtime::{ExecutionContext, MemoryControlPolicy, resident::GpuResidentInputMap},
};

include!("graph_algorithm/support.rs");
include!("graph_algorithm/residency.rs");
include!("graph_algorithm/memory_model.rs");
include!("graph_algorithm/betweenness.rs");
include!("graph_algorithm/registry.rs");
include!("graph_algorithm/capability.rs");
