//! Native cudf-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-cudf`.

pub mod error;
pub mod exec;
pub mod expr;
pub mod metrics;
pub mod plan;
pub mod runtime;
pub mod source;

pub use error::{Facts, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, Result};
pub use exec::ExecutionMode;
pub use expr::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec,
};
pub use metrics::{BoundaryMetrics, QueryMetrics};
pub use plan::{
    AggregateNode, DataPlacement, FilterNode, LimitNode, NodeId, PlanFacts, PlanNode, ProjectionNode, QueryNode,
    QueryPlan, QueryPlanBuilder, ResultSink, SchemaBehavior, SortNode,
};
pub use runtime::{DevicePlacement, ExecutionContext, ExecutionContextBuilder, MemoryResourcePolicy};
pub use source::{LocalParquetSource, SourceDiagnostics, SourceFormat, SourceNode};
