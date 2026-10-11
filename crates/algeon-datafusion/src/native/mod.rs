//! DataFusion physical-plan lowering into native query-engine IR.
//!
//! Lowering is planning-only: it returns native query plans plus structured
//! planning/source diagnostics, but does not execute or allocate GPU memory.
//! Its only I/O is reading Parquet footers, shared per planning pass through
//! `FooterScope`.
//!
//! `lower_physical_plan_to_native` normalizes the DataFusion subtree, lowers it
//! for the demand of its parent, and returns a `NativeLoweringResult`. The
//! `plan` is `Some` only when the report holds a single selected candidate;
//! it then has a host sink in the DataFusion output schema, schema metadata
//! included. Ordering is not enforced here: the native optimizer may change
//! what the plan reproduces, so `reestablish_declared_output_ordering_if_needed`
//! runs after it. A candidate the IR cannot express is a reported decline; a
//! lowered plan that breaks an IR invariant is an adapter defect and an error.
//!
//! The translation contract lives in `website/docs/design/native-lowering.mdx`.

mod expression_contract;
mod footers;
pub(crate) mod lakehouse_pruning;
pub(crate) mod literal;
mod lowering;
#[cfg(test)]
mod lowering_tests;
mod normalize;
mod operator_support;
mod pruning;
mod source;
mod table_format_pruning_policy;
mod validate;

pub(crate) use expression_contract::{
    ExpressionCapability, ExpressionRejection, ExpressionRejectionKind, ExpressionRejectionSubject,
};
#[cfg(test)]
pub(crate) use footers::footer_parses_on_this_thread;
pub(crate) use footers::{FooterCache, FooterScope};

/// Rejection raised while lowering a normalized candidate.
///
/// `reason` is the public classification. `cause` names what rejected the
/// candidate when the rejecting site knows: the shared expression contract,
/// whose capability and types planning reports project the unsupported
/// category from, or the operator, function, or ordering that blocked it. It
/// is boxed so a `Result` carrying a rejection stays small.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoweringRejection {
    reason: CapabilityReason,
    cause: Option<Box<RejectionCause>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RejectionCause {
    Expression(ExpressionRejection),
    Blocker(PlanBlocker),
}

impl LoweringRejection {
    pub(crate) const fn reason(&self) -> CapabilityReason {
        self.reason
    }

    /// A rejection of `reason` caused by `blocker`.
    pub(crate) fn blocked_by(reason: CapabilityReason, blocker: PlanBlocker) -> Self {
        Self { reason, cause: Some(Box::new(RejectionCause::Blocker(blocker))) }
    }

    /// The reason and the evidence this rejection carries, with the unsupported
    /// `column` the caller found in the plan schemas.
    pub(crate) fn into_detail(self, column: Option<UnsupportedColumnDetail>) -> (CapabilityReason, NotSupportedDetail) {
        let mut detail = NotSupportedDetail { column, ..NotSupportedDetail::default() };
        match self.cause.map(|cause| *cause) {
            Some(RejectionCause::Expression(expression)) => detail.expression = Some(Box::new(expression)),
            Some(RejectionCause::Blocker(blocker)) => detail.blocker = Some(Box::new(blocker)),
            None => {},
        }
        (self.reason, detail)
    }
}

impl From<CapabilityReason> for LoweringRejection {
    fn from(reason: CapabilityReason) -> Self {
        Self { reason, cause: None }
    }
}

impl From<ExpressionRejection> for LoweringRejection {
    fn from(rejection: ExpressionRejection) -> Self {
        Self { reason: rejection.capability_reason(), cause: Some(Box::new(RejectionCause::Expression(rejection))) }
    }
}

/// Why lowering a normalized candidate produced no native plan.
#[derive(Debug, Clone)]
pub(crate) enum LoweringFailure {
    /// The candidate misses a native contract; DataFusion keeps it.
    Rejected(LoweringRejection),
    /// Lowering built a native plan that breaks the IR construction
    /// invariants. That is an adapter defect, reported as an error rather
    /// than a fallback.
    Invalid(QueryError),
}

impl LoweringFailure {
    /// Names `blocker` for a rejection that names no more specific cause.
    pub(crate) fn blocked_by_if_unnamed(self, blocker: impl FnOnce() -> PlanBlocker) -> Self {
        match self {
            Self::Rejected(rejection) if rejection.cause.is_none() => {
                Self::Rejected(LoweringRejection::blocked_by(rejection.reason, blocker()))
            },
            other => other,
        }
    }

    /// Names the operator whose lowering failed, for a rejection that names no
    /// more specific cause.
    pub(crate) fn within_operator(self, operator: impl FnOnce() -> String) -> Self {
        self.blocked_by_if_unnamed(|| PlanBlocker::Operator(operator()))
    }
}

impl From<LoweringRejection> for LoweringFailure {
    fn from(rejection: LoweringRejection) -> Self {
        Self::Rejected(rejection)
    }
}

impl From<CapabilityReason> for LoweringFailure {
    fn from(reason: CapabilityReason) -> Self {
        Self::Rejected(reason.into())
    }
}

impl From<ExpressionRejection> for LoweringFailure {
    fn from(rejection: ExpressionRejection) -> Self {
        Self::Rejected(rejection.into())
    }
}

impl From<QueryError> for LoweringFailure {
    fn from(error: QueryError) -> Self {
        Self::Invalid(error)
    }
}

impl ExpressionRejection {
    /// Public classification of this rejection, for callers that report no
    /// candidate diagnostics.
    pub(crate) const fn capability_reason(&self) -> CapabilityReason {
        match self.kind() {
            ExpressionRejectionKind::UnsupportedExpression => CapabilityReason::UnsupportedExpression,
            ExpressionRejectionKind::UnsupportedType => CapabilityReason::UnsupportedType,
        }
    }
}

use std::sync::Arc;

use arrow::datatypes::{FieldRef as ArrowFieldRef, SchemaRef as ArrowSchemaRef};
use cudf::interop::arrow_host::is_supported_arrow_type;
use datafusion::{
    physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr, expressions::Column},
    physical_expr_common::physical_expr::fmt_sql,
    physical_plan::ExecutionPlan,
};
use lowering::lower_normalized_plan;
use normalize::{
    NormalizedNativeAggregate, NormalizedNativeInput, NormalizedNativeLiteralSource,
    make_declared_ordered_hash_join_fetch_explicit, normalize_native_plan,
};
pub(crate) use operator_support::first_join_operator;
pub(crate) use pruning::scan_predicate_exactly_matches_filter;
use query_planning::{
    datatypes::diagnostic_arrow_type,
    error::QueryError,
    expr::{Expr as NativeExpr, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec},
    plan::{NodeId as NativeNodeId, PlanNode as NativePlanNode, QueryNode, QueryPlan as NativeQueryPlan},
    source::{SourceDiagnostics as NativeSourceDiagnostics, SourceNode as NativeSourceNode},
};
use source::source_diagnostics_for_normalized_plan;
pub(crate) use source::{ParquetReadStrategyConfig, S3_STORAGE_PROFILE_ID, ambient_s3_endpoint};
#[cfg(feature = "cugraph")]
pub(crate) use source::{
    datafusion_local_parquet_scan_has_simple_directed_edge_topology, exact_row_count_for_datafusion_local_parquet_scan,
    graph_residency_descriptor_for_datafusion_local_parquet_scan,
    graph_vertex_upper_bound_for_datafusion_local_parquet_scan,
};
pub(crate) use validate::floating_statistical_aggregate_input_is_partitioned;
use validate::{native_zero_column_count_source_is_safe, validate_sort_ordering};

use crate::{
    capability::CapabilityReason,
    error::{Error, Result},
    report::{GpuCandidate, NotSupportedDetail, PlanBlocker, PlanningReport, UnsupportedColumnDetail},
};

pub(in crate::native) const DECLARED_OUTPUT_ORDERING_SORT_OPERATOR: &str = "SortExec";

/// What one lowering produced: at most one selected plan, the candidate
/// outcomes, and the source diagnostics of the scans it inspected. Diagnostics
/// are present for declined candidates too, so a report can describe a scan
/// that never ran.
#[derive(Debug, Clone)]
pub(crate) struct NativeLoweringResult {
    /// The lowered plan of the one candidate, when it was selected.
    pub(crate) plan: Option<NativeQueryPlan>,
    pub(crate) report: PlanningReport,
    pub(crate) source_diagnostics: Vec<NativeSourceDiagnostics>,
}

impl NativeLoweringResult {
    #[cfg(test)]
    pub(crate) fn candidate_count(&self) -> usize {
        self.report.outcomes().len()
    }

    #[cfg(test)]
    pub(crate) fn selected_count(&self) -> usize {
        self.report.selected_count()
    }

    #[cfg(test)]
    pub(crate) fn selected_plan(&self) -> &NativeQueryPlan {
        self.plan.as_ref().expect("lowering selected a native plan")
    }

    #[cfg(test)]
    pub(crate) fn not_selected_count(&self) -> usize {
        self.report.not_selected_count()
    }
}

/// Lowers a physical plan using the final native rewrite rules.
///
/// A candidate the native IR cannot express is reported as not supported. A
/// lowered plan that breaks the IR construction invariants is an error.
pub(crate) fn lower_physical_plan_to_native(
    plan: Arc<dyn ExecutionPlan>,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<NativeLoweringResult> {
    lower_physical_plan_to_native_with_mode(plan, NativeLoweringMode::Final, source_reads)
}

pub(crate) fn lower_count_only_physical_plan_to_native(
    plan: Arc<dyn ExecutionPlan>,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<NativeLoweringResult> {
    lower_physical_plan_to_native_with_mode(plan, NativeLoweringMode::CountOnly, source_reads)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeLoweringMode {
    Final,
    CountOnly,
}

fn lower_physical_plan_to_native_with_mode(
    plan: Arc<dyn ExecutionPlan>,
    mode: NativeLoweringMode,
    source_reads: &ParquetReadStrategyConfig,
) -> Result<NativeLoweringResult> {
    let _footers = FooterScope::enter();
    let normalized = normalize_native_plan(plan.as_ref()).map(|normalized| match mode {
        NativeLoweringMode::Final => make_declared_ordered_hash_join_fetch_explicit(
            normalized,
            plan.properties().output_ordering(),
            plan.schema(),
        ),
        NativeLoweringMode::CountOnly => normalized,
    });
    match normalized {
        // The probe lowers only a root aggregate that reads no column of a
        // local Parquet or memory source. Every other candidate is left to the
        // final rewrite, reported here with the root operator as blocker.
        Ok(normalized)
            if mode == NativeLoweringMode::CountOnly
                && !root_aggregate(&normalized.input).is_some_and(|root| match root.input.as_ref() {
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
            report.push_not_supported_with_detail(
                normalized.candidate.clone(),
                CapabilityReason::UnsupportedOperator,
                NotSupportedDetail {
                    blocker: Some(Box::new(PlanBlocker::Operator(plan.name().to_string()))),
                    ..NotSupportedDetail::default()
                },
            );
            Ok(NativeLoweringResult {
                plan: None,
                report,
                source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
            })
        },
        Ok(normalized) => {
            match lower_normalized_plan(&normalized, plan.properties().output_ordering().is_some(), source_reads) {
                Err(LoweringFailure::Invalid(error)) => Err(Error::from_native(
                    "algeon_datafusion.native.lower",
                    "native lowering built a malformed native plan",
                    error,
                )),
                Err(LoweringFailure::Rejected(rejection)) => {
                    let column = (rejection.reason() == CapabilityReason::UnsupportedType)
                        .then(|| unsupported_column_for_normalized_plan(&normalized, plan.schema()))
                        .flatten();
                    let (reason, detail) = rejection.into_detail(column);
                    let mut report = PlanningReport::new();
                    report.push_not_supported_with_detail(normalized.candidate.clone(), reason, detail);
                    Ok(NativeLoweringResult {
                        plan: None,
                        report,
                        source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                    })
                },
                Ok(mut native_plan) => {
                    // The Arrow schema at the DataFusion boundary can carry
                    // schema-level metadata. Preserve that contract at the host
                    // sink while keeping the native IR itself DataFusion-free.
                    native_plan = native_plan.with_host_sink_schema(Arc::clone(&plan.schema())).map_err(|error| {
                        Error::from_native(
                            "algeon_datafusion.native.lower",
                            "native lowering built a host sink that differs from the DataFusion schema",
                            error,
                        )
                    })?;
                    // The declared output ordering is enforced after the native
                    // optimizer, which may change what the plan reproduces.
                    let source_diagnostics = source_diagnostics_for_native_plan(&native_plan);
                    let mut report = PlanningReport::new();
                    report.push_selected(normalized.candidate.clone());
                    Ok(NativeLoweringResult { plan: Some(native_plan), report, source_diagnostics })
                },
            }
        },
        Err(rejection) => {
            let column = (rejection.reason == CapabilityReason::UnsupportedType)
                .then(|| unsupported_column_for_schema(&plan.schema(), "projected"))
                .flatten();
            let mut report = PlanningReport::new();
            report.push_not_supported_with_detail(
                rejection.candidate,
                rejection.reason,
                NotSupportedDetail {
                    column,
                    blocker: Some(rejection.blocker),
                    single_partition: rejection.single_partition,
                    ..NotSupportedDetail::default()
                },
            );
            Ok(NativeLoweringResult { plan: None, report, source_diagnostics: Vec::new() })
        },
    }
}

fn unsupported_column_for_normalized_plan(
    normalized: &normalize::NormalizedNativePlan<'_>,
    output_schema: ArrowSchemaRef,
) -> Option<UnsupportedColumnDetail> {
    unsupported_column_for_schema(&output_schema, "projected")
        .or_else(|| unsupported_column_for_input(&normalized.input))
}

/// The aggregate below the root projections, sorts and limits.
fn root_aggregate<'n, 'a>(input: &'n NormalizedNativeInput<'a>) -> Option<&'n NormalizedNativeAggregate<'a>> {
    match input {
        NormalizedNativeInput::Aggregate(aggregate) => Some(aggregate),
        NormalizedNativeInput::Projection(projection) => root_aggregate(&projection.input),
        NormalizedNativeInput::Sort(sort) => root_aggregate(&sort.input),
        NormalizedNativeInput::Limit(limit) => root_aggregate(&limit.input),
        _ => None,
    }
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
            unsupported_column_for_schema(&source.scan.schema(), "source")
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

/// Enforces the orderings an optimized native plan must keep: every
/// order-sensitive operator reads ordered input, and the output reproduces
/// the plan's declared ordering, by an appended sort when needed.
pub(crate) fn reestablish_declared_output_ordering_if_needed(
    native_plan: NativeQueryPlan,
    ordering: Option<&LexOrdering>,
) -> std::result::Result<(NativeQueryPlan, bool), LoweringRejection> {
    if !native_plan_order_sensitive_inputs_are_reproduced(&native_plan) {
        return Err(unreproducible_ordering("window input ordering", None));
    }
    let Some(ordering) = ordering else {
        return Ok((native_plan, false));
    };
    if native_plan_reproduces_declared_output_ordering(&native_plan) {
        return Ok((native_plan, false));
    }
    if !native_plan_can_append_declared_output_ordering_sort(&native_plan) {
        return Err(unreproducible_ordering(DECLARED_OUTPUT_ORDERING, Some(ordering)));
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
) -> std::result::Result<NativeSortSpec, LoweringRejection> {
    if ordering.is_empty() {
        return Err(unreproducible_ordering(DECLARED_OUTPUT_ORDERING, Some(ordering)));
    }
    let output_schema = output_node.output_schema();
    validate_sort_ordering(ordering, output_schema)?;
    let expressions = ordering
        .iter()
        .map(|sort_expr| declared_output_ordering_sort_expr(sort_expr, output_schema, ordering))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

fn declared_output_ordering_sort_expr(
    sort_expr: &PhysicalSortExpr,
    output_schema: &ArrowSchemaRef,
    ordering: &LexOrdering,
) -> std::result::Result<NativeSortExpr, LoweringRejection> {
    let unreproducible = || unreproducible_ordering(DECLARED_OUTPUT_ORDERING, Some(ordering));
    let column = sort_expr.expr.downcast_ref::<Column>().ok_or_else(unreproducible)?;
    let field = output_schema.fields().get(column.index()).ok_or_else(unreproducible)?;
    Ok(NativeSortExpr::new(
        NativeExpr::column(field.name().as_str()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}

const DECLARED_OUTPUT_ORDERING: &str = "declared output ordering";

fn unreproducible_ordering(required_by: &'static str, ordering: Option<&LexOrdering>) -> LoweringRejection {
    LoweringRejection::blocked_by(
        CapabilityReason::NativeOrderedOutputUnsupported,
        PlanBlocker::Ordering { required_by, ordering: ordering.map(render_ordering) },
    )
}

/// The ordering as SQL, such as `a ASC NULLS LAST, b DESC NULLS FIRST`.
fn render_ordering(ordering: &LexOrdering) -> String {
    ordering
        .iter()
        .map(|sort_expr| {
            format!(
                "{} {} NULLS {}",
                fmt_sql(sort_expr.expr.as_ref()),
                if sort_expr.options.descending { "DESC" } else { "ASC" },
                if sort_expr.options.nulls_first { "FIRST" } else { "LAST" },
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
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
        | NativePlanNode::VectorAlgorithm(_)
        | NativePlanNode::MlAlgorithm(_) => false,
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
        | NativePlanNode::VectorAlgorithm(_)
        | NativePlanNode::MlAlgorithm(_) => false,
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
        | NativePlanNode::VectorAlgorithm(_)
        | NativePlanNode::MlAlgorithm(_) => false,
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
    use query_planning::{
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
        let filter = builder.add_filter(source, Expr::literal(LiteralValue::Boolean(true))).unwrap();
        let projection = builder
            .add_projection(
                filter,
                vec![ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::UInt32, false))],
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap();

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
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    vec![Expr::column("key")],
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count_key")],
                ),
                Arc::clone(&aggregate_schema),
            )
            .unwrap();
        let limit = builder.add_limit(aggregate, LimitSpec::fetch(1)).unwrap();
        let plan = builder.finish_with_host_sink(limit).unwrap();

        assert!(!native_plan_reproduces_declared_output_ordering(&plan));
    }

    #[test]
    fn order_reproduction_audit_accepts_native_sort() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", key_schema())));
        let sort =
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
        let plan = builder.finish_with_host_sink(sort).unwrap();

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
    fn declared_output_ordering_that_a_union_breaks_names_the_ordering() {
        let plan = union_plan();
        let schema = plan.output_schema();
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new(col("key", schema).unwrap(), ArrowSortOptions::new(false, false))])
                .expect("test ordering should be non-empty");

        let rejection = reestablish_declared_output_ordering_if_needed(plan, Some(&ordering))
            .expect_err("a union cannot take an appended sort");
        let (reason, detail) = rejection.into_detail(None);
        let blocking = detail.blocking().expect("the rejection names its ordering");

        assert_eq!(reason, CapabilityReason::NativeOrderedOutputUnsupported);
        assert_eq!(blocking.kind(), "ordering");
        assert_eq!(blocking.text(), "declared output ordering `key ASC NULLS LAST`");
    }

    #[test]
    fn order_reproduction_audit_accepts_literal_batch_source() {
        let schema = key_schema();
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        let plan = builder.finish_with_host_sink(source).unwrap();

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
        let plan = builder.finish_with_host_sink(source).unwrap();

        let rejection = declared_output_ordering_sort_spec(&ordering, plan.root_node())
            .expect_err("timestamp-with-timezone keys must stay rejected for appended declared-output ordering sorts");

        assert_eq!(rejection.reason(), CapabilityReason::UnsupportedType);
    }

    #[test]
    fn order_reproduction_audit_rejects_gpu_resident_source() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", key_schema())));
        let plan = builder.finish_with_host_sink(source).unwrap();

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
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    vec![Expr::column("key")],
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count_key")],
                ),
                aggregate_schema,
            )
            .unwrap();
        builder.finish_with_host_sink(aggregate).unwrap()
    }

    fn window_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let window = builder
            .add_window(
                source,
                WindowSpec::new(
                    Vec::new(),
                    vec![SortExpr::new(Expr::column("key"), false, false)],
                    vec![WindowFunction::rank("rank")],
                ),
            )
            .unwrap();
        builder.finish_with_host_sink(window).unwrap()
    }

    fn window_limit_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let sort =
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
        let window = builder
            .add_window(
                sort,
                WindowSpec::new(
                    Vec::new(),
                    vec![SortExpr::new(Expr::column("key"), false, false)],
                    vec![WindowFunction::rank("rank")],
                ),
            )
            .unwrap();
        let limit = builder.add_limit(window, LimitSpec::fetch(3)).unwrap();
        builder.finish_with_host_sink(limit).unwrap()
    }

    fn join_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let join = builder.add_join(left, right, JoinSpec::new(JoinType::Cross, Vec::new()), key_schema()).unwrap();
        builder.finish_with_host_sink(join).unwrap()
    }

    fn union_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
        let union = builder.add_union(vec![left, right], key_schema()).unwrap();
        builder.finish_with_host_sink(union).unwrap()
    }

    fn edge_normalize_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_local_parquet_source()));
        let normalize = builder
            .add_edge_normalize(
                source,
                EdgeNormalizeSpec::python_cugraph(
                    GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
                    &GraphConstructionOptions::default(),
                ),
            )
            .unwrap();
        builder.finish_with_host_sink(normalize).unwrap()
    }

    fn graph_algorithm_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(edge_local_parquet_source()));
        let params = GraphAlgorithmParams::PageRank(GraphPageRankParams::default());
        let output_schema = graph_result_schema_for_params(GraphAlgorithm::PageRank, &params);
        let graph = builder
            .add_graph_algorithm(
                source,
                Vec::new(),
                GraphAlgorithmSpec::new(
                    GraphAlgorithm::PageRank,
                    GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
                    GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
                    params,
                    output_schema,
                ),
            )
            .unwrap();
        builder.finish_with_host_sink(graph).unwrap()
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
