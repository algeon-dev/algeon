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
    adaptive_report::{
        ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptiveCandidateStatus, AdaptiveMemoryHistoryEvidence,
        AdaptivePipelineReport, AdaptiveStrategy,
    },
    adaptive_runtime::{
        AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryObservation, AdaptiveTaskMemoryHistoryKey,
    },
    admission::{
        CompiledBoundedPipelineSupport, adaptive_bounded_pipeline_admission,
        adaptive_bounded_pipeline_admission_with_memory_history, compiled_bounded_pipeline_support,
        compiled_bounded_query_plan_support, strict_bounded_pipeline_admission,
        strict_bounded_pipeline_admission_with_statistical_aggregate_mode,
        strict_bounded_pipeline_rejection_projection,
        strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode,
    },
    capability::{
        CapabilityDecision, CapabilityEnvelopeAdmission, ContractQuality, ContractStatus, ReservationRequirementScope,
        STRICT_UNRESOLVED_PREFLIGHT_GAPS, StrictBoundedEnvelope, SubgraphInvariantKind,
        lower_level_unresolved_preflight_gap,
    },
    contract_report::{
        StrictBoundedContractGateId, StrictBoundedContractGateRow, StrictBoundedContractOwner,
        StrictBoundedContractReport, StrictBoundedExecutionMode,
    },
    dynamic_filter::DynamicFilterProducer,
    exec::{
        CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState, CompiledSchedulingCycle,
        CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, execute_query_plan,
        execute_query_plan_with_gpu_inputs, materialize_gpu_result_to_host,
        output::{QueryOutput, QueryOutputMode},
    },
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, GroupingSetsSpec,
        LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    metrics::{MemoryReservationEventAction, QueryMetrics},
    mode::{AdaptiveExecutionMode, NativePipelineExecutionMode, StatisticalAggregateExecutionMode},
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

use crate::bounded_join_admission::{
    AggregateOverJoinSubplanInputsVariant, ResidualResidentGroupedAggregateInputVariant,
    admitted_mixed_residual_multi_join_plan, keyless_conditional_aggregate_over_join_subplan_inputs_plan,
    large_keyed_join_strategy_plan,
    plans::{
        KeyedHashJoinResidentSubplanPipelineVariant, KeylessInputSubplanVariant,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant, RootTopKJoinAggregateVariant,
        aggregate_over_join_subplan_topk_plan, keyed_hash_join_resident_subplan_pipeline_plan,
        keyless_conditional_join_input_subplan_plan, resident_grouped_aggregate_input_post_join_aggregate_topk_plan,
        resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan,
        resident_grouped_aggregate_subquery_filter_plan, root_topk_join_aggregate_plan,
        root_topk_join_aggregate_variant_plan, shared_grouped_aggregate_fanout_to_join_and_grouped_aggregate_plan,
        single_join_post_root_aggregate_sort_limit_plan,
    },
    residual_resident_grouped_aggregate_input_plan, unsupported_cross_key_join_plan,
};

#[path = "pipeline_compiled_execution/admission.rs"]
mod admission;
#[path = "pipeline_compiled_execution/execution_modes.rs"]
mod execution_modes;
#[path = "pipeline_compiled_execution/fixtures_parquet.rs"]
mod fixtures_parquet;
#[path = "pipeline_compiled_execution/fixtures_plans.rs"]
mod fixtures_plans;
#[path = "pipeline_compiled_execution/gpu_execution.rs"]
mod gpu_execution;
#[path = "pipeline_compiled_execution/scheduler.rs"]
mod scheduler;
#[path = "pipeline_compiled_execution/support.rs"]
mod support;

use fixtures_parquet::*;
use fixtures_plans::*;
use support::*;
