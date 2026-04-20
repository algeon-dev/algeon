use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use arrow::{
    array::{Array, ArrayRef, Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use cudf::{DataFrame, Series, frame::factorization::Codebook};
use datafusion::{
    common::{DataFusionError, Result, stats::Precision},
    datasource::{memory::MemorySourceConfig, physical_plan::FileScanConfig, source::DataSourceExec},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, expressions::Column},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        aggregates::{AggregateExec, AggregateMode},
        collect,
        execution_plan::{Boundedness, EmissionType},
        metrics::{ExecutionPlanMetricsSet, MetricsSet, RecordOutput},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::sort::SortExec,
        stream::RecordBatchStreamAdapter,
        union::UnionExec,
    },
};
use futures::{TryStreamExt, future::BoxFuture};
use nexus_query_engine::{
    exec::{execute_query_plan, execute_query_plan_with_gpu_inputs, output::QueryOutput},
    expr::{BinaryOp, CaseWhen, Expr as NativeExpr, LimitSpec, LiteralValue},
    graph_memory::{graph_edge_list_output_rows_for_plan, graph_memory_estimate_for_edge_list},
    metrics::{GraphAlgorithmRuntimeMetrics, QueryMetrics},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams,
        GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
        GraphBfsPredicateInputs as NativeGraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
        GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams,
        GraphEdgeColumnBindings, GraphEdgePredicateInput as NativeGraphEdgePredicateInput,
        GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams, GraphInDegreesAllParams,
        GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLeidenParams, GraphLouvainParams,
        GraphMstParams, GraphOutDegreesAllParams, GraphPageRankParams,
        GraphPageRankPersonalizationInput as NativeGraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
        GraphPredicateInput as NativeGraphPredicateInput, GraphResidencyDescriptor, GraphResidencyPlanAnalysis,
        GraphSccParams, GraphSourceFingerprintBypassReason, GraphSpectralClusteringParams, GraphSsspParams,
        GraphTriangleCountAllParams, GraphWccParams, PlanNode, ProjectionOutput, QueryPlanBuilder,
        graph_residency_descriptor_for_edge_plan, planned_resident_graph_key,
    },
    runtime::{
        ExecutionContext, MemoryControlPolicy, QueryPlacementHint,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{GpuResidentSource, SourceNode},
};

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsOutputMode, CugraphBfsParams,
    CugraphBfsSource, CugraphBfsSourceVertex, CugraphGraphConstructionPolicy, CugraphKCoreDegreeType, plan_error,
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
        NativeCudfMetrics,
        admission::task_context_with_query_admission_ticket,
        graph::{
            GraphAdmissionModelAsk, GraphBfsFollowerOutcome, GraphBfsLeader, GraphBfsSingleflightJoin,
            GraphBfsSingleflightKey, GraphExecutionPlan, register_graph_execution_plan_discovery,
        },
        producer::{
            GpuResidentProducer, gpu_resident_producer as native_gpu_resident_producer,
            register_gpu_resident_producer_discovery,
        },
        query_admission_ticket_from_task_context, run_gpu_execution_task,
    },
};

static NEXT_CUGRAPH_GPU_RESULT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

mod admission;
mod bfs_mapping;
mod edge_analysis;
mod input;
mod lowering;
mod node;
mod path;
mod residency;
#[cfg(test)]
mod tests;

use admission::*;
use bfs_mapping::*;
use edge_analysis::*;
use input::*;
use lowering::*;
pub use node::CugraphAlgoExec;
use path::*;
use residency::*;
