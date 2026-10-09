use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use query_engine::rapids_backend::rapids_analysis_context;
use query_planning::{
    capability::CapabilityDecision,
    error::QueryErrorCode,
    expr::{BinaryOp, Expr, LiteralValue},
    plan::{
        GRAPH_OPERATION_REGISTRY, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsEndpointPolicy,
        GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
        GraphEdgeColumnBindings, GraphEdgePredicateInput, GraphPageRankParams, GraphPredicateInput,
        GraphResidencyDescriptor, GraphResidencyOrientation, GraphResidencyPlanAnalysis, GraphSimilarityParams,
        GraphSourceFingerprintBypassReason, GraphVertexPairInput, GraphWeightPolicy, PlanNode, ProjectionOutput,
        QueryPlan, QueryPlanBuilder, edge_plan_has_simple_directed_edge_topology,
        graph_residency_descriptor_for_edge_plan, graph_result_schema_for_vertex_type, planned_resident_graph_key,
        stable_graph_result_schema,
    },
    policy::PlanningPolicy,
    source::{
        CredentialMode, GpuResidentSource, GraphSimpleDirectedEdgeTopology, LocalParquetFile, LocalParquetSource,
        ResolvedParquetSource, SourceNode,
    },
};
#[cfg(feature = "cugraph")]
use support::CUGRAPH_REGISTRY_NAMES;
use support::{
    edge_schema, edge_schema_no_weight, edge_schema_with_edge_id, graph_edge_source_plan,
    graph_edge_source_plan_from_source, graph_keyable_descriptor, graph_local_parquet_file, graph_local_parquet_source,
    graph_local_parquet_source_with_file, graph_object_store_parquet_file, graph_plan, graph_residency_bfs_spec,
};

mod registry;
mod residency;
mod support;
