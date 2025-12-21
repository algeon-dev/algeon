//! Native cudf-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-cudf`.

pub mod chunk;
pub mod error;
pub mod exec;
pub mod expr;
pub mod metrics;
pub mod plan;
pub mod runtime;
pub mod source;

pub use chunk::{SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, plan_local_parquet_source_chunks};
pub use error::{Facts, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, Result};
pub use exec::{
    ExecutionMode, GpuDataFrame, QueryExecutionResult, QueryOutput, execute_query_plan, read_local_parquet_source,
    validate_local_parquet_source,
};
pub use expr::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec,
};
pub use metrics::{BoundaryMetrics, QueryMetrics};
pub use plan::{
    AggregateNode, DataPlacement, FilterNode, LimitNode, NodeId, PlanFacts, PlanNode, ProjectionNode, QueryNode,
    QueryPlan, QueryPlanBuilder, ResultSink, SchemaBehavior, SortNode,
};
pub use runtime::{
    DevicePlacement, ExecutionContext, ExecutionContextBuilder, MemoryControlPolicy, MemoryReservationRequest,
    MemoryResourcePolicy, ReservationKind, RetryFacts,
};
pub use source::{
    ByteRangeDisposition, LocalParquetFile, LocalParquetSource, PartitionColumn, PartitionColumnDisposition,
    PredicateDisposition, RejectedSourceFeature, SchemaMappingDisposition, SourceColumnRef, SourceDiagnostics,
    SourceFormat, SourceNode, ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition,
};
