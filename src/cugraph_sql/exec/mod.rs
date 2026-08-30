use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Instant,
};

use arrow::{
    compute::SortOptions,
    datatypes::{DataType, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::frame::dataframe::HostBatchCursorConfig;
use datafusion::{
    common::{
        DataFusionError, Result, stats::Precision, tree_node::TreeNodeRecursion,
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
use nexus_query_engine::{
    exec::{
        execute_query_plan_observed, execute_query_plan_with_gpu_inputs_observed,
        output::{QueryExecutionOutcome, QueryOutput},
    },
    expr::{
        AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr as NativeExpr, LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    graph_memory::{GraphMemoryCalibrationOverride, GraphVertexEstimate, query_plan_graph_vertex_upper_bound},
    metrics::{GraphAlgorithmRuntimeMetrics, QueryMetrics},
    observability::RuntimeObservationConfig,
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams,
        GraphBetweennessSeeds as NativeGraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
        GraphBfsPredicateInputs as NativeGraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
        GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams,
        GraphEdgeColumnBindings, GraphEdgePredicateInput as NativeGraphEdgePredicateInput,
        GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams, GraphInDegreesAllParams,
        GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams,
        GraphMstParams, GraphOutDegreesAllParams, GraphPageRankParams,
        GraphPageRankPersonalizationInput as NativeGraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
        GraphPredicateInput as NativeGraphPredicateInput, GraphResidencyDescriptor, GraphResidencyPlanAnalysis,
        GraphSccParams, GraphSourceFingerprintBypassReason, GraphSpectralClusteringParams, GraphSsspParams,
        GraphSsspSource, GraphTriangleCountAllParams, GraphWccParams, JoinKey, JoinOutputColumn, JoinSide, JoinSpec,
        JoinType, PlanNode, ProjectionOutput, QueryPlanBuilder, WindowFunction, WindowSpec,
        graph_residency_descriptor_for_edge_plan, planned_resident_graph_key,
    },
    runtime::{
        ExecutionContext,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{GpuResidentSource, LiteralBatchSource, SourceNode},
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBetweennessSeeds, CugraphBfsOutputMode,
    CugraphBfsParams, CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphKCoreDegreeType,
    CugraphLogicalOutputMultiplicity, CugraphVertexSelector, GpuIslandHostBoundaryConfig, PredicateSourceColumnRole,
    plan_error, result_schema_for_params, vertex_id_domain_for_call,
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        NativeExecutionMetricsCapture, NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig,
        NexusNativeGpuProducerExec, NexusNativeMetrics,
        admission::task_context_with_query_admission_ticket,
        graph::{
            GraphAdmissionModelAsk, GraphBfsFollowerOutcome, GraphBfsFollowerRoute, GraphBfsLeader,
            GraphBfsPublishedResult, GraphBfsSingleflightJoin, GraphBfsSingleflightKey, GraphExecutionPlan,
            register_graph_execution_plan_discovery,
        },
        producer::{
            GpuResidentProducer, gpu_resident_producer as native_gpu_resident_producer,
            gpu_resident_producer_input_name, register_gpu_resident_producer_discovery,
        },
        run_gpu_execution_task,
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
    context.runtime().map_err(|error| DataFusionError::from(Error::from_native(operation, message, error)))
}

mod admission;
mod edge_analysis;
mod input;
mod logical_mapping;
mod lowering;
mod node;
mod path;
mod residency;
#[cfg(test)]
mod tests;

use admission::*;
use edge_analysis::*;
use input::*;
use logical_mapping::*;
use lowering::*;
pub use node::CugraphAlgoExec;
use path::*;
use residency::*;

pub(super) struct GraphExecutionMetrics<'a> {
    sink: &'a NexusNativeMetrics,
    runtime_observation_config: RuntimeObservationConfig,
    accumulated: QueryMetrics,
    stage_count: usize,
    published: bool,
}

impl<'a> GraphExecutionMetrics<'a> {
    pub(super) fn new(sink: &'a NexusNativeMetrics, runtime_observation_config: RuntimeObservationConfig) -> Self {
        Self {
            sink,
            runtime_observation_config,
            accumulated: QueryMetrics::default(),
            stage_count: 0,
            published: false,
        }
    }

    fn runtime_observation_config(&self) -> RuntimeObservationConfig {
        self.runtime_observation_config
    }

    fn merge_stage_evidence(
        &mut self,
        metrics: QueryMetrics,
        _observation: Option<nexus_query_engine::observability::SealedRuntimeObservationSnapshot>,
    ) {
        if self.stage_count == 0 {
            self.accumulated = metrics;
        } else {
            self.accumulated.merge_from(metrics);
        }
        self.stage_count = self.stage_count.saturating_add(1);
    }

    fn merge_outcome(
        &mut self,
        outcome: QueryExecutionOutcome,
    ) -> std::result::Result<QueryOutput, nexus_query_engine::error::QueryError> {
        let (result, metrics, observation) = outcome.into_parts();
        self.merge_stage_evidence(metrics, observation);
        if result.is_err() {
            self.publish();
        }
        result
    }

    pub(super) fn finish<T>(&mut self, result: Result<T>) -> Result<T> {
        self.publish();
        result
    }

    fn publish(&mut self) {
        if self.published {
            return;
        }
        self.sink.record_query_metrics(&self.accumulated);
        self.published = true;
    }

    fn record_device_to_host_materialization(&mut self, batch: &RecordBatch) {
        self.accumulated.boundary.record_device_to_host_materialization(
            u64::try_from(batch.num_rows()).unwrap_or(u64::MAX),
            u64::try_from(get_record_batch_memory_size(batch)).unwrap_or(u64::MAX),
        );
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
        .map_err(|error| DataFusionError::from(Error::from_native_with_blocking_diagnostics(operation, message, error)))
}
