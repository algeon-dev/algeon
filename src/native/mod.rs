//! DataFusion physical-plan lowering into native query-engine IR.
//!
//! Lowering is planning-only: it returns native query plans plus structured
//! planning/source diagnostics, but does not execute or allocate GPU memory.

pub mod lakehouse_pruning;
pub(crate) mod literal;
mod lowering;
mod normalize;
mod pruning;
mod source;
mod table_format_pruning_policy;
mod validate;

use std::sync::Arc;

use datafusion::{
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_plan::ExecutionPlan,
};
use lowering::lower_normalized_plan;
use nexus_query_engine::{
    expr::{Expr as NativeExpr, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec},
    plan::{NodeId as NativeNodeId, PlanNode as NativePlanNode, QueryPlan as NativeQueryPlan},
    source::{SourceDiagnostics as NativeSourceDiagnostics, SourceNode as NativeSourceNode},
};
use normalize::normalize_native_plan;
use source::source_diagnostics_for_normalized_plan;
use validate::{native_zero_column_count_source_is_safe, validate_sort_ordering};

use crate::report::{GpuCandidate, PlanningReport};

const DECLARED_OUTPUT_ORDERING_SORT_OPERATOR: &str = "SortExec";

/// Result of attempting to lower one DataFusion physical plan.
#[derive(Debug, Clone)]
pub struct NativeLoweringResult {
    /// Native query plans produced for rewritten candidates.
    pub plans: Vec<NativeQueryPlan>,
    /// Structured candidate outcomes used by planning/report surfaces.
    pub report: PlanningReport,
    /// Source diagnostics gathered from normalized or lowered source nodes.
    pub source_diagnostics: Vec<NativeSourceDiagnostics>,
}

impl NativeLoweringResult {
    /// Returns the number of candidates recorded in the planning report.
    pub fn candidate_count(&self) -> usize {
        self.report.outcomes().len()
    }

    /// Returns the number of candidates successfully lowered.
    pub fn rewritten_count(&self) -> usize {
        self.report.rewritten_count()
    }

    /// Returns the number of rejected candidates.
    pub fn rejected_count(&self) -> usize {
        self.report.rejected_count()
    }
}

/// Lowers a physical plan using the final native rewrite rules.
pub fn lower_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
    lower_physical_plan_to_native_with_mode(plan, NativeLoweringMode::Final)
}

pub(crate) fn lower_count_only_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
    lower_physical_plan_to_native_with_mode(plan, NativeLoweringMode::CountOnly)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeLoweringMode {
    Final,
    CountOnly,
}

fn lower_physical_plan_to_native_with_mode(
    plan: Arc<dyn ExecutionPlan>,
    mode: NativeLoweringMode,
) -> NativeLoweringResult {
    match normalize_native_plan(plan.as_ref()) {
        Ok(normalized)
            if mode == NativeLoweringMode::CountOnly
                && !normalized.aggregate_root().is_some_and(|root| {
                    normalized.single_source_input().is_some_and(|source| {
                        native_zero_column_count_source_is_safe(
                            source.filter,
                            root.pre_aggregate_projection,
                            root.aggregate,
                        )
                    })
                }) =>
        {
            let mut report = PlanningReport::new();
            report.push_rejected(normalized.candidate.clone(), crate::policy::FallbackReason::UnsupportedOperator);
            NativeLoweringResult {
                plans: Vec::new(),
                report,
                source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
            }
        },
        Ok(normalized) => lower_normalized_plan(&normalized).map_or_else(
            |reason| {
                let mut report = PlanningReport::new();
                report.push_rejected(normalized.candidate.clone(), reason);
                NativeLoweringResult {
                    plans: Vec::new(),
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
            |mut native_plan| {
                let mut candidate = normalized.candidate.clone();
                if !native_plan_order_sensitive_inputs_are_reproduced(&native_plan) {
                    let mut report = PlanningReport::new();
                    report.push_rejected(candidate, crate::policy::FallbackReason::NativeOrderedOutputUnsupported);
                    return NativeLoweringResult {
                        plans: Vec::new(),
                        report,
                        source_diagnostics: source_diagnostics_for_native_plan(&native_plan),
                    };
                }
                let mut appended_declared_output_sort = false;
                if let Some(ordering) = plan.properties().output_ordering()
                    && !native_plan_reproduces_declared_output_ordering(&native_plan)
                {
                    if !native_plan_can_append_declared_output_ordering_sort(&native_plan) {
                        let mut report = PlanningReport::new();
                        report.push_rejected(
                            candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR),
                            crate::policy::FallbackReason::NativeOrderedOutputUnsupported,
                        );
                        return NativeLoweringResult {
                            plans: Vec::new(),
                            report,
                            source_diagnostics: source_diagnostics_for_native_plan(&native_plan),
                        };
                    }
                    let sort_spec = match declared_output_ordering_sort_spec(ordering, native_plan.root_node()) {
                        Ok(sort_spec) => sort_spec,
                        Err(reason) => {
                            let mut report = PlanningReport::new();
                            report.push_rejected(
                                candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR),
                                reason,
                            );
                            return NativeLoweringResult {
                                plans: Vec::new(),
                                report,
                                source_diagnostics: source_diagnostics_for_native_plan(&native_plan),
                            };
                        },
                    };
                    native_plan = native_plan.with_appended_sort(sort_spec);
                    appended_declared_output_sort = true;
                }
                if plan.properties().output_ordering().is_some()
                    && !native_plan_reproduces_declared_output_ordering(&native_plan)
                {
                    let mut report = PlanningReport::new();
                    if appended_declared_output_sort {
                        candidate = candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR);
                    }
                    report.push_rejected(candidate, crate::policy::FallbackReason::NativeOrderedOutputUnsupported);
                    return NativeLoweringResult {
                        plans: Vec::new(),
                        report,
                        source_diagnostics: source_diagnostics_for_native_plan(&native_plan),
                    };
                }
                let source_diagnostics = source_diagnostics_for_native_plan(&native_plan);
                let mut report = PlanningReport::new();
                if appended_declared_output_sort {
                    candidate = candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR);
                }
                report.push_rewritten(candidate);
                NativeLoweringResult { plans: vec![native_plan], report, source_diagnostics }
            },
        ),
        Err(rejection) => {
            let mut report = PlanningReport::new();
            report.push_rejected(rejection.candidate, rejection.reason);
            NativeLoweringResult { plans: Vec::new(), report, source_diagnostics: Vec::new() }
        },
    }
}

pub(crate) fn reestablish_declared_output_ordering_if_needed(
    native_plan: NativeQueryPlan,
    ordering: Option<&LexOrdering>,
) -> Result<(NativeQueryPlan, bool), crate::policy::FallbackReason> {
    let Some(ordering) = ordering else {
        return Ok((native_plan, false));
    };
    if !native_plan_order_sensitive_inputs_are_reproduced(&native_plan) {
        return Err(crate::policy::FallbackReason::NativeOrderedOutputUnsupported);
    }
    if native_plan_reproduces_declared_output_ordering(&native_plan) {
        return Ok((native_plan, false));
    }
    if !native_plan_can_append_declared_output_ordering_sort(&native_plan) {
        return Err(crate::policy::FallbackReason::NativeOrderedOutputUnsupported);
    }
    let sort_spec = declared_output_ordering_sort_spec(ordering, native_plan.root_node())?;
    Ok((native_plan.with_appended_sort(sort_spec), true))
}

pub(crate) fn wrap_candidate_with_declared_output_ordering_sort(candidate: GpuCandidate) -> GpuCandidate {
    candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR)
}

fn declared_output_ordering_sort_spec(
    ordering: &LexOrdering,
    output_node: &nexus_query_engine::plan::QueryNode,
) -> Result<NativeSortSpec, crate::policy::FallbackReason> {
    if ordering.is_empty() {
        return Err(crate::policy::FallbackReason::NativeOrderedOutputUnsupported);
    }
    let output_schema = output_node.output_schema();
    validate_sort_ordering(ordering, output_schema)?;
    let expressions = ordering
        .iter()
        .map(|sort_expr| declared_output_ordering_sort_expr(sort_expr, output_schema))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

fn declared_output_ordering_sort_expr(
    sort_expr: &PhysicalSortExpr,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<NativeSortExpr, crate::policy::FallbackReason> {
    let column =
        sort_expr.expr.downcast_ref::<Column>().ok_or(crate::policy::FallbackReason::NativeOrderedOutputUnsupported)?;
    let field = output_schema
        .fields()
        .get(column.index())
        .ok_or(crate::policy::FallbackReason::NativeOrderedOutputUnsupported)?;
    Ok(NativeSortExpr::new(
        NativeExpr::column(field.name().as_str()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}

fn source_diagnostics_for_native_plan(plan: &NativeQueryPlan) -> Vec<NativeSourceDiagnostics> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            NativePlanNode::Source(source) => match source.as_ref() {
                NativeSourceNode::LocalParquet(source) => Some(source.diagnostics.clone()),
                NativeSourceNode::GpuResident(_) => None,
                NativeSourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

pub(crate) fn native_plan_reproduces_declared_output_ordering(plan: &NativeQueryPlan) -> bool {
    native_node_reproduces_declared_output_ordering(plan, plan.root_id(), &mut Vec::new())
}

fn native_plan_can_append_declared_output_ordering_sort(plan: &NativeQueryPlan) -> bool {
    native_plan_order_sensitive_inputs_are_reproduced(plan)
        && native_node_allows_declared_output_ordering_sort_append(plan, plan.root_id(), &mut Vec::new())
}

fn native_plan_order_sensitive_inputs_are_reproduced(plan: &NativeQueryPlan) -> bool {
    plan.nodes().iter().all(|node| match &node.kind {
        NativePlanNode::Window(window) if !window.spec.order_by.is_empty() => {
            native_node_reproduces_declared_output_ordering(plan, window.input, &mut Vec::new())
        },
        _ => true,
    })
}

fn native_node_allows_declared_output_ordering_sort_append(
    plan: &NativeQueryPlan,
    node_id: NativeNodeId,
    stack: &mut Vec<NativeNodeId>,
) -> bool {
    if stack.contains(&node_id) {
        return false;
    }
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    stack.push(node_id);
    let allowed = match &node.kind {
        NativePlanNode::Source(_) => true,
        NativePlanNode::Filter(filter) => {
            native_node_allows_declared_output_ordering_sort_append(plan, filter.input, stack)
        },
        NativePlanNode::Projection(projection) => {
            native_node_allows_declared_output_ordering_sort_append(plan, projection.input, stack)
        },
        NativePlanNode::Aggregate(_) => true,
        NativePlanNode::Sort(_) => true,
        NativePlanNode::Limit(_)
        | NativePlanNode::Window(_)
        | NativePlanNode::Join(_)
        | NativePlanNode::Union(_)
        | NativePlanNode::EdgeNormalize(_)
        | NativePlanNode::GraphAlgorithm(_) => false,
    };
    stack.pop();
    allowed
}

fn native_node_reproduces_declared_output_ordering(
    plan: &NativeQueryPlan,
    node_id: NativeNodeId,
    stack: &mut Vec<NativeNodeId>,
) -> bool {
    if stack.contains(&node_id) {
        return false;
    }
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    stack.push(node_id);
    let reproduced = match &node.kind {
        NativePlanNode::Source(source) => match source.as_ref() {
            NativeSourceNode::LocalParquet(_) | NativeSourceNode::LiteralBatch(_) => true,
            NativeSourceNode::GpuResident(_) => false,
        },
        NativePlanNode::Filter(filter) => native_node_reproduces_declared_output_ordering(plan, filter.input, stack),
        NativePlanNode::Projection(projection) => {
            native_node_reproduces_declared_output_ordering(plan, projection.input, stack)
        },
        NativePlanNode::Limit(limit) => native_node_reproduces_limit_output_ordering(plan, limit.input, stack),
        NativePlanNode::Sort(_) => true,
        NativePlanNode::Aggregate(_)
        | NativePlanNode::Window(_)
        | NativePlanNode::Join(_)
        | NativePlanNode::Union(_)
        | NativePlanNode::EdgeNormalize(_)
        | NativePlanNode::GraphAlgorithm(_) => false,
    };
    stack.pop();
    reproduced
}

fn native_node_reproduces_limit_output_ordering(
    plan: &NativeQueryPlan,
    node_id: NativeNodeId,
    stack: &mut Vec<NativeNodeId>,
) -> bool {
    if stack.contains(&node_id) {
        return false;
    }
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    stack.push(node_id);
    let reproduced = match &node.kind {
        NativePlanNode::Source(source) => match source.as_ref() {
            NativeSourceNode::LocalParquet(_) | NativeSourceNode::LiteralBatch(_) => true,
            NativeSourceNode::GpuResident(_) => false,
        },
        NativePlanNode::Filter(filter) => native_node_reproduces_limit_output_ordering(plan, filter.input, stack),
        NativePlanNode::Projection(projection) => {
            native_node_reproduces_limit_output_ordering(plan, projection.input, stack)
        },
        NativePlanNode::Sort(_) => true,
        NativePlanNode::Limit(_)
        | NativePlanNode::Aggregate(_)
        | NativePlanNode::Window(_)
        | NativePlanNode::Join(_)
        | NativePlanNode::Union(_)
        | NativePlanNode::EdgeNormalize(_)
        | NativePlanNode::GraphAlgorithm(_) => false,
    };
    stack.pop();
    reproduced
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        compute::SortOptions as ArrowSortOptions,
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    };
    use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
    use nexus_query_engine::{
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec},
        plan::{
            EdgeNormalizeSpec, GraphAlgorithm, GraphAlgorithmParams, GraphConstructionOptions, GraphEdgeColumnBindings,
            GraphPageRankParams, JoinSpec, JoinType, ProjectionOutput, QueryPlan, QueryPlanBuilder, WindowFunction,
            WindowSpec, graph_result_schema_for_params,
        },
        source::{GpuResidentSource, LiteralBatchSource, LocalParquetSource, SourceNode},
    };

    use super::{declared_output_ordering_sort_spec, native_plan_reproduces_declared_output_ordering};
    use crate::policy::FallbackReason;

    #[test]
    fn order_reproduction_audit_accepts_local_parquet_filter_projection() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let filter = builder.add_filter(source, Expr::literal(LiteralValue::Boolean(true)), key_schema());
        let projection = builder.add_projection(
            filter,
            vec![ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::UInt32, false))],
        );
        let plan = builder.finish_with_host_sink(projection);

        assert!(native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn order_reproduction_audit_rejects_limit_over_order_breaking_input() {
        let aggregate_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("count_key", DataType::UInt64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count_key")],
            ),
            Arc::clone(&aggregate_schema),
        );
        let limit = builder.add_limit(aggregate, LimitSpec::fetch(1), aggregate_schema);
        let plan = builder.finish_with_host_sink(limit);

        assert!(!native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn order_reproduction_audit_accepts_native_sort() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", key_schema())));
        let sort = builder.add_sort(
            source,
            SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
            key_schema(),
        );
        let plan = builder.finish_with_host_sink(sort);

        assert!(native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn order_reproduction_audit_accepts_literal_batch_source() {
        let schema = key_schema();
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        let plan = builder.finish_with_host_sink(source);

        assert!(native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn declared_output_ordering_sort_spec_reuses_native_sort_key_type_admission() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "event_ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into())),
            false,
        )]));
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            col("event_ts", &schema).unwrap(),
            ArrowSortOptions::new(false, false),
        )])
        .expect("test ordering should be non-empty");
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        let plan = builder.finish_with_host_sink(source);

        let reason = declared_output_ordering_sort_spec(&ordering, plan.root_node())
            .expect_err("timestamp-with-timezone keys must stay rejected for appended declared-output ordering sorts");

        assert_eq!(reason, FallbackReason::UnsupportedType);
    }

    #[test]
    fn order_reproduction_audit_rejects_gpu_resident_source() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", key_schema())));
        let plan = builder.finish_with_host_sink(source);

        assert!(!native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn order_reproduction_audit_rejects_aggregate_window_join_union_edge_normalize_and_graph_algorithm() {
        let cases =
            [aggregate_plan(), window_plan(), join_plan(), union_plan(), edge_normalize_plan(), graph_algorithm_plan()];

        for plan in cases {
            assert!(!native_plan_reproduces_declared_output_ordering(&plan), "{plan}");
        }
    }

    fn aggregate_plan() -> QueryPlan {
        let aggregate_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("count_key", DataType::UInt64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count_key")],
            ),
            aggregate_schema,
        );
        builder.finish_with_host_sink(aggregate)
    }

    fn window_plan() -> QueryPlan {
        let window_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("rank", DataType::UInt64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let window = builder.add_window(
            source,
            WindowSpec::new(
                Vec::new(),
                vec![SortExpr::new(Expr::column("key"), false, false)],
                vec![WindowFunction::Rank { alias: "rank".to_string() }],
            ),
            window_schema,
        );
        builder.finish_with_host_sink(window)
    }

    fn join_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let join = builder.add_join(left, right, JoinSpec::new(JoinType::Cross, Vec::new()), key_schema());
        builder.finish_with_host_sink(join)
    }

    fn union_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let union = builder.add_union(vec![left, right], key_schema());
        builder.finish_with_host_sink(union)
    }

    fn edge_normalize_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_local_parquet_source()));
        let normalize = builder.add_edge_normalize(
            source,
            EdgeNormalizeSpec::python_cugraph(
                GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
                &GraphConstructionOptions::default(),
            ),
            edge_schema(),
        );
        builder.finish_with_host_sink(normalize)
    }

    fn graph_algorithm_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_local_parquet_source()));
        let params = GraphAlgorithmParams::PageRank(GraphPageRankParams::default());
        let output_schema = graph_result_schema_for_params(&params);
        let graph = builder.add_graph_algorithm(
            source,
            nexus_query_engine::plan::GraphAlgorithmSpec::new(
                GraphAlgorithm::PageRank,
                GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
                GraphConstructionOptions {
                    policy: nexus_query_engine::plan::GraphConstructionPolicy::RawLibcugraph,
                    ..Default::default()
                },
                params,
                output_schema,
            ),
        );
        builder.finish_with_host_sink(graph)
    }

    fn local_parquet_source() -> LocalParquetSource {
        LocalParquetSource::new("ordered.parquet", key_schema(), vec!["key".to_string()])
    }

    fn edge_local_parquet_source() -> LocalParquetSource {
        LocalParquetSource::new("edges.parquet", edge_schema(), vec!["src".to_string(), "dst".to_string()])
    }

    fn key_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]))
    }

    fn edge_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]))
    }
}
