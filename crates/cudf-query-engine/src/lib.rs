//! Native cudf-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-cudf`.

pub(crate) mod bounded;
pub mod chunk;
pub mod datatypes;
pub mod error;
pub mod exec;
pub mod expr;
pub mod metrics;
pub mod plan;
pub mod runtime;
pub mod source;

pub use chunk::{
    SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, SourceMetadataCache, plan_local_parquet_source_chunks,
    plan_local_parquet_source_chunks_with_metadata_cache,
};
pub use datatypes::{
    arrow_decimal_scale_to_cudf_scale, cudf_dtype_from_arrow_type, is_supported_arrow_cast,
    is_supported_arrow_implicit_cast, is_supported_arrow_type, timestamp_unit_to_cudf_resolution,
};
pub use error::{Facts, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, Result};
pub use exec::{
    ExecutionMode, GpuDataFrame, GpuHashJoinBuildCache, GpuHashJoinBuildHandle, GpuLocalParquetSourceCache,
    GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey, GpuResidentInputMap, GpuResultHandle,
    QueryExecutionResult, QueryOutput, execute_query_plan, execute_query_plan_with_gpu_inputs,
    materialize_gpu_result_to_host, read_gpu_resident_source, read_local_parquet_source, validate_local_parquet_source,
};
pub use expr::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnQualifier, ColumnRef, DatePart, Expr,
    LimitSpec, LiteralValue, SortExpr, SortSpec,
};
pub use metrics::{BoundaryMetrics, BudgetedExecutionEvent, JoinOperatorMetrics, OperatorMetrics, QueryMetrics};
pub use plan::{
    AggregateNode, DataPlacement, FilterNode, HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality,
    JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitNode, NodeId, PlanFacts, PlanNode, ProjectionNode,
    ProjectionOutput, QueryNode, QueryPlan, QueryPlanBuilder, ResultSink, SchemaBehavior, SortNode,
};
pub use runtime::{
    BOUNDED_JOIN_EXECUTION_FAMILIES, BoundedJoinExecutionFamily, BoundedJoinReservationContract,
    BoundedJoinReservationRequirement, BoundedJoinReservationStatus, BuildMaterializationBoundary, DevicePlacement,
    ExecutionBoundaryPolicy, ExecutionContext, ExecutionContextBuilder, MemoryControlPolicy, MemoryReservationRequest,
    MemoryResourcePolicy, MemorySpaceId, ProbeExecutionBoundary, ReservationKind, ResultMaterializationBoundary,
    RetryFacts, SourceReadBoundary, SourceSchedulerPolicy,
};
pub use source::{
    ByteRangeDisposition, GpuResidentSource, LocalParquetFile, LocalParquetSource, MissingColumn, PartitionColumn,
    PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature, ScanLimitDisposition,
    SchemaMappingDisposition, SourceColumnRef, SourceDiagnostics, SourceFormat, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
    ZeroColumnCountDisposition,
};
