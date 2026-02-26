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
    ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptiveCandidateStatus, AdaptiveExecutionMode,
    AdaptiveMemoryHistory, AdaptiveMemoryHistoryEvidence, AdaptiveMemoryHistoryKey, AdaptiveMemoryObservation,
    AdaptivePipelineReport, AdaptiveStrategy, AdaptiveTaskMemoryHistoryKey, AggregateExpr, AggregateFunction,
    AggregateSpec, BinaryOp, CaseWhen, ColumnRef, CompiledPipeline, DynamicFilterProducer, ExecutionContext, Expr,
    GroupingSetsSpec, JoinBuildSideSelectionOptions, JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
    LimitSpec, LiteralValue, LocalParquetFile, LocalParquetSource, MemoryControlPolicy, MemoryReservationEventAction,
    MemoryReservationRequest, NativeOptimizerOptions, NativePipelineExecutionMode, NodeId, PipelineBarrier,
    PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId, PipelineEdgeKind, PipelineTask,
    PipelineTaskId, PipelineTaskKind, PlanNode, ProjectionOutput, QueryMetrics, QueryOutputMode, QueryPlan,
    QueryPlanBuilder, ReservationGate, ReservationGateId, ReservationGateOwner, ReservationKind, ResidentLifetime,
    ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind, SortExpr, SortSpec, SourceNode,
    StrictBoundedContractGateId, StrictBoundedContractGateRow, StrictBoundedContractOwner, WindowFunction, WindowSpec,
    adaptive_bounded_pipeline_admission, adaptive_bounded_pipeline_admission_with_memory_history,
    capability::{
        CapabilityDecision, ContractQuality, ContractStatus, ReservationRequirementScope,
        STRICT_UNRESOLVED_PREFLIGHT_GAPS, SubgraphInvariantKind, lower_level_unresolved_preflight_gap,
    },
    compile_pipeline,
    contract_report::{StrictBoundedContractReport, StrictBoundedExecutionMode},
    exec::{
        CompiledBoundedPipelineSupport, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
        CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput, compiled_bounded_pipeline_support,
        compiled_bounded_query_plan_support,
    },
    execute_query_plan, optimize_query_plan, strict_bounded_pipeline_admission,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use crate::bounded_join_admission::{
    AggregateOverJoinSubplanInputsVariant, ResidualResidentGroupedAggregateInputVariant,
    admitted_mixed_residual_multi_join_plan, keyless_conditional_aggregate_over_join_subplan_inputs_plan,
    large_keyed_join_strategy_plan,
    plans::{
        KeyedHashJoinResidentSubplanPipelineVariant, KeylessInputSubplanVariant, aggregate_over_join_subplan_topk_plan,
        keyed_hash_join_resident_subplan_pipeline_plan, keyless_conditional_join_input_subplan_plan,
        resident_grouped_aggregate_input_post_join_aggregate_topk_plan,
        resident_grouped_aggregate_subquery_filter_plan,
        shared_grouped_aggregate_fanout_to_join_and_grouped_aggregate_plan,
        single_join_post_root_aggregate_sort_limit_plan,
    },
    residual_resident_grouped_aggregate_input_plan, unsupported_cross_key_join_plan,
};

#[path = "pipeline_compiled_execution/admission.rs"]
mod admission;
#[path = "pipeline_compiled_execution/fixtures_parquet.rs"]
mod fixtures_parquet;
#[path = "pipeline_compiled_execution/fixtures_plans.rs"]
mod fixtures_plans;
#[path = "pipeline_compiled_execution/gpu_execution.rs"]
mod gpu_execution;
#[path = "pipeline_compiled_execution/phase_e_modes.rs"]
mod phase_e_modes;
#[path = "pipeline_compiled_execution/scheduler.rs"]
mod scheduler;
#[path = "pipeline_compiled_execution/support.rs"]
mod support;

use fixtures_parquet::*;
use fixtures_plans::*;
use support::*;
