use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{Field, Schema, SchemaRef};

use crate::{
    bounded::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    expr::{AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
    runtime::ExecutionBoundaryPolicy,
    source::SourceNode,
};

mod builder;
mod core;
mod display;
mod facts;
mod join;
mod nodes;
mod sink;

pub use core::{NodeId, PlanNode, QueryNode, QueryPlan};

pub use builder::QueryPlanBuilder;
pub use facts::{DataPlacement, PlanFacts, SchemaBehavior};
pub use join::{
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
};
pub use nodes::{AggregateNode, FilterNode, LimitNode, ProjectionNode, ProjectionOutput, SortNode};
pub use sink::ResultSink;
