use std::sync::atomic::AtomicU64;

use arrow::{
    compute::SortOptions,
    datatypes::{DataType, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::frame::dataframe::HostBatchCursorConfig;
use datafusion::{
    common::{
        DataFusionError, Result as DataFusionResult, stats::Precision, tree_node::TreeNodeRecursion,
        utils::memory::get_record_batch_memory_size,
    },
    datasource::{memory::MemorySourceConfig, physical_plan::FileScanConfig, source::DataSourceExec},
    execution::{
        context::TaskContext,
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
        ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs, StatisticsContext,
        aggregates::{AggregateExec, AggregateMode},
        coop::CooperativeExec,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::sort::SortExec,
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
    },
};
use futures::{Stream, TryStreamExt, future::BoxFuture};
use query_engine::exec::{
    QueryExecutionRequest, execute_query_plan,
    output::{PlanMetrics, QueryExecutionOutcome, QueryOutput},
};
use query_planning::{
    error::QueryError,
    expr::{
        AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr as NativeExpr, LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams,
        GraphBetweennessSeeds as NativeGraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
        GraphBfsPredicateInputs as NativeGraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
        GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams,
        GraphEdgeColumnBindings, GraphEdgePredicateInput as NativeGraphEdgePredicateInput,
        GraphEigenvectorCentralityParams, GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2Params,
        GraphForceAtlas2VertexAttributesInput, GraphHitsParams, GraphInDegreesAllParams, GraphKCoreDegreeType,
        GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams, GraphMstParams,
        GraphOutDegreesAllParams, GraphPageRankParams,
        GraphPageRankPersonalizationInput as NativeGraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
        GraphPredicateInput as NativeGraphPredicateInput, GraphResidencyDescriptor, GraphResidencyPlanAnalysis,
        GraphSccParams, GraphSourceFingerprintBypassReason, GraphSpectralClusteringParams, GraphSsspParams,
        GraphSsspSource, GraphTriangleCountAllParams, GraphWccParams, JoinKey, JoinOutputColumn, JoinSide, JoinSpec,
        JoinType, PlanNode, ProjectionOutput, QueryPlanBuilder, WindowFunction, WindowSpec,
        graph_residency_descriptor_for_edge_plan, planned_resident_graph_key,
    },
    source::{GpuResidentSource, LiteralBatchSource, SourceNode},
};
use query_runtime::{
    metrics::GraphAlgorithmRuntimeMetrics,
    runtime::{
        ExecutionContext,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
};

use self::singleflight::{
    GraphBfsFollowerOutcome, GraphBfsFollowerRoute, GraphBfsLeader, GraphBfsPublishedResult, GraphBfsSingleflightJoin,
    GraphBfsSingleflightKey,
};
use super::{
    CUGRAPH_EDGE_DESTINATION_COLUMN, CUGRAPH_EDGE_ID_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN, CUGRAPH_EDGE_WEIGHT_COLUMN,
    CUGRAPH_MASS_COLUMN, CUGRAPH_MOBILITY_COLUMN, CUGRAPH_RADIUS_COLUMN, CUGRAPH_VALUE_COLUMN, CUGRAPH_VERTEX_COLUMN,
    CUGRAPH_X_COLUMN, CUGRAPH_Y_COLUMN, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
    CugraphBetweennessSeeds, CugraphBfsOutputMode, CugraphBfsParams, CugraphBfsSourceVertex,
    CugraphGraphConstructionPolicy, CugraphKCoreDegreeType, CugraphLogicalOutputMultiplicity, CugraphVertexSelector,
    GpuIslandHostBoundaryConfig, PredicateSourceColumnRole, plan, result_schema_for_params, vertex_id_domain_for_call,
};
use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::{
        AlgeonNativeExec, AlgeonNativeExecutionConfig, AlgeonNativeMetrics, GpuResidentProducer,
        NativeExecutionMetricsCapture, admission::task_context_with_query_admission_ticket,
        execute_gpu_resident_producer_with_context, gpu_resident_producer,
        gpu_resident_producer_hard_output_row_upper_bound, graph::GraphExecutionPlan,
        prepare_gpu_resident_producer_for_execution, run_gpu_execution_task,
    },
};

static NEXT_CUGRAPH_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default)]
struct CugraphResultPlanOptions {
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    subsumed_top_k_sort: Option<SortSpec>,
}

impl CugraphResultPlanOptions {
    fn new(projection: Option<Vec<usize>>, limit: Option<usize>) -> Self {
        Self { projection, limit, subsumed_top_k_sort: None }
    }
}

fn counted_cudf_runtime(
    context: &ExecutionContext,
    operation: &'static str,
    message: &'static str,
) -> Result<cudf::Runtime> {
    context.runtime().map_err(|error| Error::from_native(operation, message, error))
}

mod admission;
mod edge_analysis;
mod input;
mod logical_mapping;
mod lowering;
mod node;
mod residency;
pub(crate) mod singleflight;
#[cfg(test)]
pub(crate) mod tests;

pub(crate) use admission::native_graph_input_execution_plan;
use admission::{
    cugraph_execution_config_after_staging, edge_source_graph_vertex_estimate, edge_source_output_rows,
    graph_input_native_producer, lowered_native_edge_plan, plan_contains_algeon_native,
    resident_side_input_output_rows, task_context_with_cugraph_execution_ticket,
    validate_cugraph_graph_owner_preflight, validate_standalone_bounded_cugraph_sources,
};
use edge_analysis::{CugraphEdgeCanonicalFacts, cugraph_edge_canonical_facts, cugraph_edge_normalize_inserted};
use input::{
    CugraphGpuFrame, CugraphInputRowBounds, EagerCugraphHostOutput, StagedCugraphInputs, UploadedCugraphInputs,
    check_host_staging_cancellation, collect_cugraph_inputs, materialize_gpu_output_eagerly,
    prepare_cugraph_source_metadata, record_batches_to_gpu,
};
use logical_mapping::{
    add_logical_mapping_plan, decode_logical_graph_result, graph_uses_logical_mapping, is_logical_vertex_type,
};
use lowering::{
    CugraphPlanDefinition, cugraph_execution_capability_evidence, cugraph_schema_query_plan,
    execute_cugraph_query_to_gpu_with_options, native_algorithm, native_graph_spec_for_placement_hint,
    preflight_cugraph_query_plan,
};
pub use node::CugraphAlgoExec;
use residency::{attach_graph_residency, cugraph_graph_residency_descriptor};

/// The metrics of one cuGraph wrapper execution: the plans its stages ran,
/// plus a plan execution of its own for the host boundary work and BFS
/// follower rows that belong to no engine plan. They reach DataFusion once.
pub(super) struct GraphExecutionMetrics<'a> {
    sink: &'a AlgeonNativeMetrics,
    stages: Vec<PlanMetrics>,
    boundary: PlanMetrics,
    published: bool,
}

impl<'a> GraphExecutionMetrics<'a> {
    pub(super) fn new(sink: &'a AlgeonNativeMetrics, context: &ExecutionContext) -> Self {
        Self { sink, stages: Vec::new(), boundary: PlanMetrics::bind(context, "cugraph-boundary"), published: false }
    }

    fn merge_outcome(&mut self, outcome: QueryExecutionOutcome) -> std::result::Result<QueryOutput, QueryError> {
        let (result, plan, _observation) = outcome.into_parts();
        self.stages.push(plan);
        if result.is_err() {
            self.publish();
        }
        result
    }

    pub(super) fn finish<T>(&mut self, result: DataFusionResult<T>) -> DataFusionResult<T> {
        self.publish();
        result
    }

    fn publish(&mut self) {
        if self.published {
            return;
        }
        let mut snapshots = self.stages.iter().chain([&self.boundary]).map(PlanMetrics::freeze);
        let mut metrics = snapshots.next().unwrap_or_default();
        for stage in snapshots {
            metrics.merge_from(stage);
        }
        self.sink.record_query_metrics(&metrics);
        self.published = true;
    }

    fn record_device_to_host_materialization(&self, batch: &RecordBatch) {
        self.boundary.recorder().record_device_to_host_materialization(
            u64::try_from(batch.num_rows()).unwrap_or(u64::MAX),
            u64::try_from(get_record_batch_memory_size(batch)).unwrap_or(u64::MAX),
        );
    }

    pub(super) fn record_graph_algorithm_runtime(&self, row: GraphAlgorithmRuntimeMetrics) {
        self.boundary.recorder().record_graph_algorithm_runtime(row);
    }
}

pub(super) fn finalize_graph_query_outcome(
    metrics: &mut GraphExecutionMetrics<'_>,
    outcome: QueryExecutionOutcome,
    operation: &'static str,
    message: &'static str,
) -> Result<QueryOutput> {
    metrics
        .merge_outcome(outcome)
        .map_err(|error| Error::from_native_with_blocking_diagnostics(operation, message, error))
}
