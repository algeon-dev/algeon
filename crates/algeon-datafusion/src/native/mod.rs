//! DataFusion physical-plan lowering into native query-engine IR.
//!
//! Lowering is planning-only: it returns native query plans plus structured
//! planning/source diagnostics, but does not execute or allocate GPU memory.

mod expression_contract;
pub(crate) mod lakehouse_pruning;
pub(crate) mod literal;
mod lowering;
#[cfg(test)]
mod lowering_tests;
mod normalize;
mod pruning;
mod source;
mod table_format_pruning_policy;
mod validate;

impl From<expression_contract::ExpressionRejection> for CapabilityReason {
    fn from(rejection: expression_contract::ExpressionRejection) -> Self {
        let _capability = rejection.capability();
        match rejection.kind() {
            expression_contract::ExpressionRejectionKind::UnsupportedExpression => Self::UnsupportedExpression,
            expression_contract::ExpressionRejectionKind::UnsupportedType => Self::UnsupportedType,
        }
    }
}

use std::sync::Arc;

use arrow::datatypes::{FieldRef as ArrowFieldRef, SchemaRef as ArrowSchemaRef};
use datafusion::{
    physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr, expressions::Column},
    physical_plan::ExecutionPlan,
};
use lowering::lower_normalized_plan;
use normalize::{
    NormalizedNativeInput, NormalizedNativeLiteralSource, make_declared_ordered_hash_join_fetch_explicit,
    normalize_native_plan,
};
pub(crate) use pruning::scan_predicate_exactly_matches_filter;
use query_engine::{
    datatypes::{diagnostic_arrow_type, is_supported_arrow_type},
    expr::{Expr as NativeExpr, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec},
    plan::{NodeId as NativeNodeId, PlanNode as NativePlanNode, QueryNode, QueryPlan as NativeQueryPlan},
    source::{SourceDiagnostics as NativeSourceDiagnostics, SourceNode as NativeSourceNode},
};
pub(crate) use source::native_parquet_file_reader_factory_is_supported;
use source::source_diagnostics_for_normalized_plan;
#[cfg(feature = "cugraph")]
pub(crate) use source::{
    datafusion_local_parquet_scan_has_simple_directed_edge_topology, exact_row_count_for_datafusion_local_parquet_scan,
    graph_residency_descriptor_for_datafusion_local_parquet_scan,
    graph_vertex_upper_bound_for_datafusion_local_parquet_scan,
};
use validate::{native_zero_column_count_source_is_safe, validate_sort_ordering};

use crate::{
    capability::CapabilityReason,
    report::{GpuCandidate, PlanningReport, UnsupportedColumnDetail},
};

const DECLARED_OUTPUT_ORDERING_SORT_OPERATOR: &str = "SortExec";

#[derive(Debug, Clone)]
pub(crate) struct NativeLoweringResult {
    pub(crate) plans: Vec<NativeQueryPlan>,
    pub(crate) report: PlanningReport,
    pub(crate) source_diagnostics: Vec<NativeSourceDiagnostics>,
}

impl NativeLoweringResult {
    #[cfg(test)]
    pub(crate) fn candidate_count(&self) -> usize {
        self.report.outcomes().len()
    }

    pub(crate) fn selected_count(&self) -> usize {
        self.report.selected_count()
    }

    #[cfg(test)]
    pub(crate) fn not_selected_count(&self) -> usize {
        self.report.not_selected_count()
    }
}

/// Lowers a physical plan using the final native rewrite rules.
pub(crate) fn lower_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
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
    let normalized = normalize_native_plan(plan.as_ref()).map(|mut normalized| {
        if mode == NativeLoweringMode::Final {
            make_declared_ordered_hash_join_fetch_explicit(
                &mut normalized,
                plan.properties().output_ordering(),
                plan.schema(),
            );
        }
        normalized
    });
    match normalized {
        Ok(normalized)
            if mode == NativeLoweringMode::CountOnly
                && !normalized.aggregate_root().is_some_and(|root| match &root.input {
                    NormalizedNativeInput::Source(source) => native_zero_column_count_source_is_safe(
                        source.filter,
                        root.pre_aggregate_projection,
                        root.aggregate,
                    ),
                    NormalizedNativeInput::LiteralSource(NormalizedNativeLiteralSource::Memory(_)) => {
                        native_zero_column_count_source_is_safe(None, root.pre_aggregate_projection, root.aggregate)
                    },
                    _ => false,
                }) =>
        {
            let mut report = PlanningReport::new();
            report.push_not_supported(normalized.candidate.clone(), CapabilityReason::UnsupportedOperator);
            NativeLoweringResult {
                plans: Vec::new(),
                report,
                source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
            }
        },
        Ok(normalized) => lower_normalized_plan(&normalized).map_or_else(
            |reason| {
                let mut report = PlanningReport::new();
                if reason == CapabilityReason::UnsupportedType
                    && let Some(detail) = unsupported_column_for_normalized_plan(&normalized, plan.schema())
                {
                    report.push_not_supported_column(normalized.candidate.clone(), reason, detail);
                } else {
                    report.push_not_supported(normalized.candidate.clone(), reason);
                }
                NativeLoweringResult {
                    plans: Vec::new(),
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
            |mut native_plan| {
                // The Arrow schema at the DataFusion boundary can carry
                // schema-level metadata. Preserve that contract at the host
                // sink while keeping the native IR itself DataFusion-free.
                native_plan = native_plan.with_host_sink_schema(Arc::clone(&plan.schema()));
                let mut candidate = normalized.candidate.clone();
                if !native_plan_order_sensitive_inputs_are_reproduced(&native_plan) {
                    let mut report = PlanningReport::new();
                    report.push_not_supported(candidate, CapabilityReason::NativeOrderedOutputUnsupported);
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
                        report.push_not_supported(
                            candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR),
                            CapabilityReason::NativeOrderedOutputUnsupported,
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
                            report.push_not_supported(
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
                    report.push_not_supported(candidate, CapabilityReason::NativeOrderedOutputUnsupported);
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
                report.push_selected(candidate);
                NativeLoweringResult { plans: vec![native_plan], report, source_diagnostics }
            },
        ),
        Err(rejection) => {
            let mut report = PlanningReport::new();
            if rejection.reason == CapabilityReason::UnsupportedType
                && let Some(detail) = unsupported_column_for_schema(&plan.schema(), "projected")
            {
                report.push_not_supported_column(rejection.candidate, rejection.reason, detail);
            } else {
                report.push_not_supported(rejection.candidate, rejection.reason);
            }
            NativeLoweringResult { plans: Vec::new(), report, source_diagnostics: Vec::new() }
        },
    }
}

fn unsupported_column_for_normalized_plan(
    normalized: &normalize::NormalizedNativePlan<'_>,
    output_schema: ArrowSchemaRef,
) -> Option<UnsupportedColumnDetail> {
    unsupported_column_for_schema(&output_schema, "projected")
        .or_else(|| unsupported_column_for_input(normalized.root_input()))
}

fn unsupported_column_for_input(input: &NormalizedNativeInput<'_>) -> Option<UnsupportedColumnDetail> {
    match input {
        NormalizedNativeInput::Source(source) => {
            if let Some(filter) = source.filter
                && let Some(detail) =
                    unsupported_column_for_expression(filter.predicate(), &filter.input().schema(), "filter")
            {
                return Some(detail);
            }
            if let Some(scan_plan) = source.table_format_scan_exec().and_then(|scan| scan.scan_plan())
                && let Some(column) = scan_plan.first_unsupported_column()
            {
                return Some(UnsupportedColumnDetail::new(
                    column.logical_name,
                    diagnostic_arrow_type(column.data_type),
                    column.role.code(),
                ));
            }
            let schema = match source.scan {
                normalize::NormalizedNativeScan::LocalParquet(scan) => scan.schema(),
                normalize::NormalizedNativeScan::TableFormat(scan) => scan.schema(),
            };
            unsupported_column_for_schema(&schema, "source")
        },
        NormalizedNativeInput::GpuResidentSource(source) => {
            unsupported_column_for_schema(&source.source_exec.schema(), "source")
        },
        NormalizedNativeInput::LiteralSource(source) => {
            let schema = match source {
                normalize::NormalizedNativeLiteralSource::PlaceholderRow(source) => source.schema(),
                normalize::NormalizedNativeLiteralSource::Empty(source) => source.schema(),
                normalize::NormalizedNativeLiteralSource::Memory(source) => source.schema(),
            };
            unsupported_column_for_schema(&schema, "source")
        },
        NormalizedNativeInput::Filter(filter) => {
            unsupported_column_for_expression(filter.filter.predicate(), &filter.filter.input().schema(), "filter")
                .or_else(|| unsupported_column_for_input(&filter.input))
        },
        NormalizedNativeInput::Projection(projection) => {
            unsupported_column_for_schema(&projection.projection.schema(), "projected")
                .or_else(|| unsupported_column_for_input(&projection.input))
        },
        NormalizedNativeInput::Limit(limit) => unsupported_column_for_input(&limit.input),
        NormalizedNativeInput::Aggregate(aggregate) => {
            unsupported_column_for_schema(&aggregate.aggregate_output_schema, "aggregate")
                .or_else(|| unsupported_column_for_input(&aggregate.input))
        },
        NormalizedNativeInput::Sort(sort) => sort
            .sort
            .ordering()
            .iter()
            .find_map(|expr| unsupported_column_for_expression(&expr.expr, sort.sort.input_schema(), "sort_key"))
            .or_else(|| unsupported_column_for_input(&sort.input)),
        NormalizedNativeInput::Window(window) => unsupported_column_for_schema(&window.window.schema(), "window")
            .or_else(|| unsupported_column_for_input(&window.input)),
        NormalizedNativeInput::HashJoin(join) => {
            unsupported_column_for_input(&join.left).or_else(|| unsupported_column_for_input(&join.right))
        },
        NormalizedNativeInput::CrossJoin(join) => {
            unsupported_column_for_input(&join.left).or_else(|| unsupported_column_for_input(&join.right))
        },
        NormalizedNativeInput::NestedLoopJoin(join) => {
            unsupported_column_for_input(&join.left).or_else(|| unsupported_column_for_input(&join.right))
        },
        NormalizedNativeInput::Union(union) => unsupported_column_for_schema(&union.union.schema(), "union")
            .or_else(|| union.inputs.iter().find_map(unsupported_column_for_input)),
    }
}

fn unsupported_column_for_expression(
    expression: &Arc<dyn PhysicalExpr>,
    schema: &ArrowSchemaRef,
    role: &'static str,
) -> Option<UnsupportedColumnDetail> {
    let mut required = std::collections::BTreeSet::new();
    validate::collect_required_columns(expression, schema, &mut required).ok()?;
    schema.fields().iter().find_map(|field| {
        (required.contains(field.name()) && !is_supported_arrow_type(field.data_type()))
            .then(|| unsupported_column_detail(field, role))
    })
}

fn unsupported_column_for_schema(schema: &ArrowSchemaRef, role: &'static str) -> Option<UnsupportedColumnDetail> {
    schema
        .fields()
        .iter()
        .find(|field| !is_supported_arrow_type(field.data_type()))
        .map(|field| unsupported_column_detail(field, role))
}

fn unsupported_column_detail(field: &ArrowFieldRef, role: &'static str) -> UnsupportedColumnDetail {
    UnsupportedColumnDetail::new(field.name(), diagnostic_arrow_type(field.data_type()), role)
}

pub(crate) fn reestablish_declared_output_ordering_if_needed(
    native_plan: NativeQueryPlan,
    ordering: Option<&LexOrdering>,
) -> Result<(NativeQueryPlan, bool), CapabilityReason> {
    let Some(ordering) = ordering else {
        return Ok((native_plan, false));
    };
    if !native_plan_order_sensitive_inputs_are_reproduced(&native_plan) {
        return Err(CapabilityReason::NativeOrderedOutputUnsupported);
    }
    if native_plan_reproduces_declared_output_ordering(&native_plan) {
        return Ok((native_plan, false));
    }
    if !native_plan_can_append_declared_output_ordering_sort(&native_plan) {
        return Err(CapabilityReason::NativeOrderedOutputUnsupported);
    }
    let sort_spec = declared_output_ordering_sort_spec(ordering, native_plan.root_node())?;
    Ok((native_plan.with_appended_sort(sort_spec), true))
}

pub(crate) fn wrap_candidate_with_declared_output_ordering_sort(candidate: GpuCandidate) -> GpuCandidate {
    candidate.with_outer_operator(DECLARED_OUTPUT_ORDERING_SORT_OPERATOR)
}

fn declared_output_ordering_sort_spec(
    ordering: &LexOrdering,
    output_node: &QueryNode,
) -> Result<NativeSortSpec, CapabilityReason> {
    if ordering.is_empty() {
        return Err(CapabilityReason::NativeOrderedOutputUnsupported);
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
    output_schema: &ArrowSchemaRef,
) -> Result<NativeSortExpr, CapabilityReason> {
    let column = sort_expr.expr.downcast_ref::<Column>().ok_or(CapabilityReason::NativeOrderedOutputUnsupported)?;
    let field = output_schema.fields().get(column.index()).ok_or(CapabilityReason::NativeOrderedOutputUnsupported)?;
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
        NativePlanNode::Aggregate(_) | NativePlanNode::Join(_) => true,
        NativePlanNode::Sort(_) => true,
        NativePlanNode::Limit(_)
        | NativePlanNode::Window(_)
        | NativePlanNode::Union(_)
        | NativePlanNode::EdgeNormalize(_)
        | NativePlanNode::GraphAlgorithm(_)
        | NativePlanNode::VectorAlgorithm(_) => false,
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
        NativePlanNode::Window(window) => native_node_reproduces_declared_output_ordering(plan, window.input, stack),
        NativePlanNode::Aggregate(_)
        | NativePlanNode::Join(_)
        | NativePlanNode::Union(_)
        | NativePlanNode::EdgeNormalize(_)
        | NativePlanNode::GraphAlgorithm(_)
        | NativePlanNode::VectorAlgorithm(_) => false,
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
        NativePlanNode::Window(window) => native_node_reproduces_limit_output_ordering(plan, window.input, stack),
        NativePlanNode::Limit(_)
        | NativePlanNode::Aggregate(_)
        | NativePlanNode::Join(_)
        | NativePlanNode::Union(_)
        | NativePlanNode::EdgeNormalize(_)
        | NativePlanNode::GraphAlgorithm(_)
        | NativePlanNode::VectorAlgorithm(_) => false,
    };
    stack.pop();
    reproduced
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        compute::SortOptions as ArrowSortOptions,
        datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
        record_batch::RecordBatch,
    };
    use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
    use query_engine::{
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec},
        plan::{
            EdgeNormalizeSpec, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions,
            GraphConstructionPolicy, GraphEdgeColumnBindings, GraphPageRankParams, JoinSpec, JoinType,
            ProjectionOutput, QueryPlan, QueryPlanBuilder, WindowFunction, WindowSpec, graph_result_schema_for_params,
        },
        source::{GpuResidentSource, LiteralBatchSource, LocalParquetSource, SourceNode},
    };

    use super::{
        CapabilityReason, NativePlanNode, declared_output_ordering_sort_spec,
        native_plan_reproduces_declared_output_ordering, reestablish_declared_output_ordering_if_needed,
    };
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
    fn declared_output_ordering_can_be_reestablished_after_a_join() {
        let plan = join_plan();
        let schema = plan.output_schema();
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new(col("key", schema).unwrap(), ArrowSortOptions::new(false, false))])
                .expect("test ordering should be non-empty");

        let (plan, appended) = reestablish_declared_output_ordering_if_needed(plan, Some(&ordering))
            .expect("a complete join output can be sorted before the host sink");

        assert!(appended);
        assert!(matches!(plan.root_node().kind, NativePlanNode::Sort(_)));
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
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
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

        assert_eq!(reason, CapabilityReason::UnsupportedType);
    }

    #[test]
    fn order_reproduction_audit_rejects_gpu_resident_source() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", key_schema())));
        let plan = builder.finish_with_host_sink(source);

        assert!(!native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn order_reproduction_audit_rejects_aggregate_join_union_edge_normalize_and_graph_algorithm() {
        let cases = [aggregate_plan(), join_plan(), union_plan(), edge_normalize_plan(), graph_algorithm_plan()];

        for plan in cases {
            assert!(!native_plan_reproduces_declared_output_ordering(&plan), "{plan}");
        }
    }

    #[test]
    fn order_reproduction_audit_preserves_window_input_ordering() {
        let plan = window_plan();
        assert!(native_plan_reproduces_declared_output_ordering(&plan), "{plan}");
    }

    #[test]
    fn order_reproduction_audit_preserves_window_input_ordering_through_limit() {
        let plan = window_limit_plan();
        assert!(native_plan_reproduces_declared_output_ordering(&plan), "{plan}");
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

    fn window_limit_plan() -> QueryPlan {
        let window_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("rank", DataType::UInt64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let sort = builder.add_sort(
            source,
            SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
            key_schema(),
        );
        let window = builder.add_window(
            sort,
            WindowSpec::new(
                Vec::new(),
                vec![SortExpr::new(Expr::column("key"), false, false)],
                vec![WindowFunction::Rank { alias: "rank".to_string() }],
            ),
            Arc::clone(&window_schema),
        );
        let limit = builder.add_limit(window, LimitSpec::fetch(3), window_schema);
        builder.finish_with_host_sink(limit)
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
        let output_schema = graph_result_schema_for_params(GraphAlgorithm::PageRank, &params);
        let graph = builder.add_graph_algorithm(
            source,
            GraphAlgorithmSpec::new(
                GraphAlgorithm::PageRank,
                GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
                GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
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
