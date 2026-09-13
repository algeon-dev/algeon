use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema};

use super::rewrite;
use crate::{
    error::QueryError,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, LiteralValue},
    plan::{
        AggregateNode, FilterNode, JoinOutputColumn, JoinSide, JoinType, NodeId, PlanFacts, PlanNode, ProjectionNode,
        ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior,
    },
    source::{
        LocalParquetFile, LocalParquetSource, ParquetReaderDecimalType, ParquetSourceLocation, ResolvedParquetSource,
        SourceColumnRef, SourceNode,
    },
    util::{fnv1a64, sum_optional_u64},
    validation::walk_join_residual_column_sides,
};

// Shared-subplan materialization is kept as one private module surface while
// implementation families live in focused include! fragments. This preserves
// the public optimizer exports and exact diagnostics during the split.
include!("shared_subplan_materialization/api.rs");
include!("shared_subplan_materialization/diagnostics.rs");
include!("shared_subplan_materialization/scalar_derivation.rs");
include!("shared_subplan_materialization/discovery.rs");
include!("shared_subplan_materialization/estimate.rs");
include!("shared_subplan_materialization/signature.rs");
include!("shared_subplan_materialization/rewrite.rs");
include!("shared_subplan_materialization/source_widening.rs");
include!("shared_subplan_materialization/expr_helpers.rs");
include!("shared_subplan_materialization/lineage.rs");
