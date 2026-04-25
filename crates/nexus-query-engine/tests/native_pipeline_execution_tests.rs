//! GPU execution ownership for compiled native pipelines.
//!
//! This target is intentionally separate from `native_contract_tests`: its
//! cases create CUDA runtimes, while the remaining contract target is CPU-only.

#![allow(dead_code, unused_imports)] // Shared fixture modules intentionally expose CPU-only helpers too.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use nexus_query_engine::{
    adaptive_runtime::{
        AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryObservation, AdaptiveTaskMemoryHistoryKey,
    },
    admission::{
        CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
        strict_bounded_pipeline_admission, strict_bounded_pipeline_rejection_projection,
    },
    capability::{ContractQuality, ContractStatus, ReservationRequirementScope},
    dynamic_filter::DynamicFilterProducer,
    exec::{
        CompiledGpuTaskExecutor, CompiledPipelineScheduler, execute_query_plan, execute_query_plan_with_gpu_inputs,
        materialize_gpu_result_to_host, output::QueryOutput,
    },
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, GroupingSetsSpec,
        LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    metrics::{MemoryReservationEventAction, QueryMetrics},
    mode::NativePipelineExecutionMode,
    optimizer::{JoinBuildSideSelectionOptions, NativeOptimizerOptions, optimize_query_plan},
    pipeline::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
        ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
        compile_pipeline,
    },
    plan::{
        JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanNode, ProjectionOutput, QueryPlan,
        QueryPlanBuilder, WindowFunction, WindowSpec,
    },
    runtime::{
        ExecutionContext, MemoryControlPolicy, MemoryReservationRequest, ReservationKind, resident::GpuResidentInputMap,
    },
    source::{
        GpuResidentSource, LocalParquetFile, LocalParquetSource, ParquetEqualityDeleteFile, PartitionColumn,
        ResolvedParquetSource, SourceNode,
    },
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
fn projection_outputs(expressions: Vec<Expr>, schema: Arc<Schema>) -> Vec<ProjectionOutput> {
    assert_eq!(expressions.len(), schema.fields().len(), "projection outputs must match schema");
    expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| ProjectionOutput::new(expr, field.as_ref().clone()))
        .collect()
}

#[path = "native_contract_tests/pipeline_compiled_execution/fixtures_parquet.rs"]
mod fixtures_parquet;
#[path = "native_contract_tests/pipeline_compiled_execution/fixtures_plans.rs"]
mod fixtures_plans;
#[path = "native_contract_tests/pipeline_compiled_execution/gpu_execution.rs"]
mod gpu_execution;
#[path = "native_contract_tests/pipeline_compiled_execution/gpu_support.rs"]
mod gpu_support;
#[path = "native_contract_tests/bounded_join_admission/plans/remaining_topologies.rs"]
mod remaining_topologies;

use fixtures_parquet::*;
use fixtures_plans::*;
use gpu_support::*;
use remaining_topologies::{
    aggregate_over_join_subplan_topk_plan, resident_grouped_aggregate_input_post_join_aggregate_topk_plan,
    resident_grouped_aggregate_subquery_filter_plan,
};
