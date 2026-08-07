//! Policy-resolved reservation envelopes for one admitted attempt.
//!
//! A capability proof is issued while lowering owns only the native plan. The
//! service owns source scheduling and chunk limits, so it resolves the plan
//! facts below immediately before ledger placement. This keeps a descriptive
//! source-size estimate out of immutable-grant sizing while still making
//! policy-created concurrent phases explicit.

use std::collections::HashSet;

use arrow::datatypes::DataType;

use crate::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    exec::{JoinOutputColumnUpperBoundInput, PartialAggregateShape, partial_aggregate_upper_bound_bytes},
    graph_memory::GraphMemoryAdmissionEvidence,
    plan::{PlanNode, QueryPlan, estimated_reachable_local_parquet_source_bytes},
    runtime::{MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy},
    source::SourceNode,
};

/// Service-owned policy inputs that affect the bounded attempt's physical
/// liveness. `device_budget_bytes` is deliberately absent: a grant is the
/// result of admission, not an input to its proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttemptReservationPolicy {
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    source_chunk_planning_options: SourceChunkPlanningOptions,
}

impl AttemptReservationPolicy {
    pub(crate) const fn new(
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> Self {
        Self {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        }
    }

    fn matches_execution(
        self,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> bool {
        let MemoryControlPolicy::Bounded {
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
            ..
        } = memory_policy
        else {
            return false;
        };
        self.max_source_chunk_bytes == max_source_chunk_bytes
            && self.max_row_groups_per_chunk == max_row_groups_per_chunk
            && self.max_retry_attempts == max_retry_attempts
            && self.min_retry_chunk_row_groups == min_retry_chunk_row_groups
            && self.source_scheduler_policy == source_scheduler_policy
            && self.source_chunk_planning_mode == source_chunk_planning_mode
            && self.source_chunk_planning_options == source_chunk_planning_options
    }

    fn source_read_phase_slots(self, source_chunk_count: u64) -> u64 {
        // `OrderedSourceReadQueue` retains at most the current decoded source
        // plus one optional prefetch. Its public scheduler bound can be larger
        // for future scheduling work, but it cannot create a third concurrent
        // source read in the current bounded executor.
        if source_chunk_count > 1 && self.source_scheduler_policy.enables_prefetch() { 2 } else { 1 }
    }
}

/// Immutable plan facts carried privately by a capability proof until the
/// query service can resolve them against its installed policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationShape {
    execution_scope_fingerprint: String,
    has_local_parquet_source: bool,
    source_stage_bytes: Option<u64>,
    complete_zero_column_global_aggregate: Option<ZeroColumnGlobalAggregateShape>,
    complete_graph_memory_phase: Option<GraphMemoryAdmissionEvidence>,
    post_graph_bfs_logical_output_peak_bytes: Option<u64>,
    accounted_by_graph_phase: bool,
    is_cugraph_graph_input_scope: bool,
    is_attested_graph_result_passthrough_scope: bool,
    is_attested_graph_result_bounded_consumer_scope: bool,
    is_bfs_logical_output_scope: bool,
}

impl AttemptReservationShape {
    pub(crate) fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            has_local_parquet_source: plan_has_reachable_local_parquet_source(plan),
            source_stage_bytes: estimated_reachable_local_parquet_source_bytes(plan),
            complete_zero_column_global_aggregate: zero_column_global_aggregate_shape(plan),
            complete_graph_memory_phase: None,
            post_graph_bfs_logical_output_peak_bytes: None,
            accounted_by_graph_phase: false,
            is_cugraph_graph_input_scope: false,
            is_attested_graph_result_passthrough_scope: false,
            is_attested_graph_result_bounded_consumer_scope: false,
            is_bfs_logical_output_scope: is_bfs_logical_output_projection_plan(plan),
        }
    }

    fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    pub(crate) fn matches_execution_scope_fingerprint(&self, execution_scope_fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == execution_scope_fingerprint
    }

    pub(crate) fn set_graph_memory_phase(
        &mut self,
        evidence: &GraphMemoryAdmissionEvidence,
        source_stage_bytes: Option<u64>,
    ) {
        self.complete_graph_memory_phase = Some(evidence.clone());
        // A composed graph proof may obtain its prepared edge frame from a
        // separately lowered native source. The aggregate proof owns that
        // exact stage byte count, so replace this graph scope's local view
        // instead of adding it a second time.
        if let Some(source_stage_bytes) = source_stage_bytes {
            self.source_stage_bytes = Some(source_stage_bytes);
        }
    }

    pub(crate) fn add_source_stage_bytes(&mut self, source_stage_bytes: u64) {
        self.source_stage_bytes = Some(self.source_stage_bytes.unwrap_or_default().saturating_add(source_stage_bytes));
    }

    pub(crate) fn set_post_graph_bfs_logical_output_peak_bytes(&mut self, peak_bytes: u64) {
        self.post_graph_bfs_logical_output_peak_bytes = Some(peak_bytes);
    }

    pub(crate) fn account_for_by_graph_phase(&mut self) {
        self.accounted_by_graph_phase = true;
    }

    pub(crate) fn mark_cugraph_graph_input_scope(&mut self) {
        self.is_cugraph_graph_input_scope = true;
    }

    pub(crate) fn mark_attested_graph_result_passthrough_scope(&mut self) {
        self.is_attested_graph_result_passthrough_scope = true;
    }

    pub(crate) fn mark_attested_graph_result_bounded_consumer_scope(&mut self) {
        self.is_attested_graph_result_bounded_consumer_scope = true;
    }

    pub(crate) fn has_modeled_source_stage(&self) -> bool {
        self.source_stage_bytes.is_some()
    }

    pub(crate) fn is_bfs_logical_output_scope(&self) -> bool {
        self.is_bfs_logical_output_scope
    }

    pub(crate) fn is_cugraph_graph_input_scope(&self) -> bool {
        self.is_cugraph_graph_input_scope
    }

    pub(crate) fn is_attested_graph_result_passthrough_scope(&self) -> bool {
        self.is_attested_graph_result_passthrough_scope
    }

    pub(crate) fn is_attested_graph_result_bounded_consumer_scope(&self) -> bool {
        self.is_attested_graph_result_bounded_consumer_scope
    }

    pub(crate) fn has_graph_memory_phase(&self) -> bool {
        self.complete_graph_memory_phase.is_some()
    }

    fn resolve(&self, policy: AttemptReservationPolicy) -> Result<ResolvedReservationShape> {
        if self.accounted_by_graph_phase {
            return Ok(ResolvedReservationShape { required_phase_floor_bytes: 0, complete_phase_peak_bytes: Some(0) });
        }
        if let Some(graph_phase) = &self.complete_graph_memory_phase {
            let graph_peak =
                self.source_stage_bytes.map_or(graph_phase.cold_model_requested_bytes(), |source_stage_bytes| {
                    graph_phase.cold_model_with_source_stage_bytes(source_stage_bytes)
                });
            // Construction, native graph execution, and logical BFS
            // decode/projection are sequential phases. Their full attempt
            // target is therefore their maximum, never their sum.
            let peak = self
                .post_graph_bfs_logical_output_peak_bytes
                .map_or(graph_peak, |logical_output_peak| graph_peak.max(logical_output_peak));
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: peak,
                complete_phase_peak_bytes: Some(peak),
            });
        }

        let minimum_source_read_bytes =
            self.has_local_parquet_source.then(|| minimum_parquet_source_read_budget_bytes()).unwrap_or_default();
        let Some(shape) = &self.complete_zero_column_global_aggregate else {
            return Ok(ResolvedReservationShape {
                required_phase_floor_bytes: minimum_source_read_bytes,
                complete_phase_peak_bytes: None,
            });
        };

        let source_chunk_count = shape.source_chunk_count(policy.max_row_groups_per_chunk)?;
        let source_phase_bytes = minimum_source_read_bytes
            .checked_mul(policy.source_read_phase_slots(source_chunk_count))
            .ok_or_else(|| envelope_overflow_error("source_read_phase"))?;
        let retained_partial_bytes = shape
            .partial_state_upper_bound_bytes
            .checked_mul(source_chunk_count)
            .ok_or_else(|| envelope_overflow_error("retained_partial_state"))?;
        // The scalar aggregate result and its immediate projection can overlap
        // with the retained partial list during finalization. They are bounded
        // independently by the same closed-form device-output contract.
        let final_scalar_phase_bytes =
            shape.maximum_scalar_frame_bytes.checked_mul(2).ok_or_else(|| envelope_overflow_error("final_scalar"))?;
        let peak = source_phase_bytes
            .checked_add(retained_partial_bytes)
            .and_then(|value| value.checked_add(final_scalar_phase_bytes))
            .ok_or_else(|| envelope_overflow_error("zero_column_global_aggregate"))?;
        Ok(ResolvedReservationShape { required_phase_floor_bytes: peak, complete_phase_peak_bytes: Some(peak) })
    }
}

/// Resolved service-admission envelope retained by the admitted context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttemptReservationEnvelope {
    policy: AttemptReservationPolicy,
    scopes: Vec<AttemptReservationShape>,
    required_phase_floor_bytes: u64,
    complete_phase_peak_bytes: Option<u64>,
}

impl AttemptReservationEnvelope {
    pub(crate) fn resolve(
        reservation_shapes: &[AttemptReservationShape],
        policy: AttemptReservationPolicy,
    ) -> Result<Self> {
        let mut scopes = Vec::with_capacity(reservation_shapes.len());
        for scope in reservation_shapes {
            if !scopes.contains(scope) {
                scopes.push(scope.clone());
            }
        }

        let mut required_phase_floor_bytes = 0_u64;
        let mut complete_phase_peak_bytes = Some(0_u64);
        let has_graph_memory_phase = scopes.iter().any(AttemptReservationShape::has_graph_memory_phase);
        for scope in &scopes {
            let resolved = scope.resolve(policy)?;
            required_phase_floor_bytes = required_phase_floor_bytes
                .checked_add(resolved.required_phase_floor_bytes)
                .ok_or_else(|| envelope_overflow_error("combined_required_phase"))?;
            complete_phase_peak_bytes = match (complete_phase_peak_bytes, resolved.complete_phase_peak_bytes) {
                (Some(current), Some(peak)) => current.checked_add(peak),
                _ => None,
            };
        }

        if has_graph_memory_phase && complete_phase_peak_bytes.is_none() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "a graph attempt contains a runtime scope without a complete bounded phase target",
            )
            .with_fact("reason", "graph_attempt_complete_phase_unmodeled"));
        }

        Ok(Self { policy, scopes, required_phase_floor_bytes, complete_phase_peak_bytes })
    }

    pub(crate) fn required_phase_floor_bytes(&self) -> u64 {
        self.required_phase_floor_bytes
    }

    pub(crate) fn complete_phase_peak_bytes(&self) -> Option<u64> {
        self.complete_phase_peak_bytes
    }

    pub(crate) fn validate_execution(
        &self,
        plan: &QueryPlan,
        memory_policy: MemoryControlPolicy,
        source_scheduler_policy: SourceSchedulerPolicy,
        source_chunk_planning_mode: SourceChunkPlanningMode,
        source_chunk_planning_options: SourceChunkPlanningOptions,
    ) -> Result<()> {
        if !self.policy.matches_execution(
            memory_policy,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
        ) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.admission.reservation_envelope",
                "admitted reservation envelope does not match the execution policy",
            )
            .with_fact("reason", "admitted_reservation_policy_mismatch"));
        }
        if self.scopes.iter().any(|scope| scope.matches(plan)) {
            return Ok(());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.admission.reservation_envelope",
            "admitted reservation envelope does not cover the native plan about to execute",
        )
        .with_fact("reason", "admitted_reservation_envelope_mismatch"))
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedReservationShape {
    required_phase_floor_bytes: u64,
    complete_phase_peak_bytes: Option<u64>,
}

/// Exact plan facts for the specialized zero-column aggregate executor.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ZeroColumnGlobalAggregateShape {
    selected_row_groups_per_file: Vec<usize>,
    partial_state_upper_bound_bytes: u64,
    maximum_scalar_frame_bytes: u64,
}

impl ZeroColumnGlobalAggregateShape {
    fn source_chunk_count(&self, max_row_groups_per_chunk: usize) -> Result<u64> {
        let max_row_groups_per_chunk = u64::try_from(max_row_groups_per_chunk.max(1))
            .map_err(|_| envelope_overflow_error("max_row_groups_per_chunk"))?;
        let mut chunks = 0_u64;
        for row_groups in &self.selected_row_groups_per_file {
            let row_groups = u64::try_from(*row_groups).map_err(|_| envelope_overflow_error("selected_row_groups"))?;
            chunks = chunks
                .checked_add(row_groups.div_ceil(max_row_groups_per_chunk))
                .ok_or_else(|| envelope_overflow_error("source_chunk_count"))?;
        }
        Ok(chunks.max(1))
    }
}

fn zero_column_global_aggregate_shape(plan: &QueryPlan) -> Option<ZeroColumnGlobalAggregateShape> {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut row_groups_per_file = Vec::new();
    let mut partial_state_upper_bound_bytes = 0_u64;
    let mut maximum_scalar_frame_bytes = 0_u64;
    let mut has_aggregate = false;

    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        maximum_scalar_frame_bytes =
            maximum_scalar_frame_bytes.max(scalar_frame_upper_bound_bytes(node.output_schema())?);
        match &node.kind {
            PlanNode::Source(source) => {
                let SourceNode::LocalParquet(source) = source.as_ref() else {
                    return None;
                };
                if !source.is_zero_column_count_source() || source.files.is_empty() {
                    return None;
                }
                for file in &source.files {
                    row_groups_per_file.push(file.row_groups.as_ref()?.len());
                }
            },
            PlanNode::Aggregate(aggregate) => {
                if !aggregate.spec.group_exprs.is_empty() {
                    return None;
                }
                partial_state_upper_bound_bytes = partial_state_upper_bound_bytes.max(
                    global_partial_state_upper_bound_bytes(node.output_schema(), aggregate.spec.aggr_exprs.len())?,
                );
                has_aggregate = true;
            },
            // This is the exact scalar path selected for COUNT(*) and related
            // zero-column global aggregates. Any widening, join, sort, graph,
            // or source-adjacent operator leaves the complete peak unknown so
            // admission deliberately uses the candidate grant ceiling.
            PlanNode::Projection(_) | PlanNode::Limit(_) => {},
            _ => return None,
        }
    }

    (has_aggregate && !row_groups_per_file.is_empty()).then_some(ZeroColumnGlobalAggregateShape {
        selected_row_groups_per_file: row_groups_per_file,
        partial_state_upper_bound_bytes,
        maximum_scalar_frame_bytes,
    })
}

fn global_partial_state_upper_bound_bytes(schema: &arrow::datatypes::SchemaRef, aggregate_count: usize) -> Option<u64> {
    let mut columns = fixed_width_columns(schema)?;
    // SUM/MIN/MAX partials retain an extra validity count and several aggregate
    // families retain more than their final projection. One fixed-width state
    // per aggregate is a conservative owner-side floor for the scalar
    // zero-column path; unsupported variable-width state declines the model.
    columns.extend((0..aggregate_count).map(|_| fixed_width_column(DataType::UInt64)));
    partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &columns).ok()
}

fn scalar_frame_upper_bound_bytes(schema: &arrow::datatypes::SchemaRef) -> Option<u64> {
    crate::exec::join_output_upper_bound_bytes(1, &fixed_width_columns(schema)?).ok()
}

fn fixed_width_columns(schema: &arrow::datatypes::SchemaRef) -> Option<Vec<JoinOutputColumnUpperBoundInput>> {
    schema.fields().iter().map(|field| fixed_width_column_checked(field.data_type())).collect()
}

fn fixed_width_column_checked(data_type: &DataType) -> Option<JoinOutputColumnUpperBoundInput> {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::UInt8
            | DataType::Int16
            | DataType::UInt16
            | DataType::Int32
            | DataType::UInt32
            | DataType::Int64
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, None)
            | DataType::Duration(_)
            | DataType::Decimal128(_, _)
    )
    .then(|| fixed_width_column(data_type.clone()))
}

fn fixed_width_column(data_type: DataType) -> JoinOutputColumnUpperBoundInput {
    JoinOutputColumnUpperBoundInput { data_type, input_max_string_length_bytes: None, input_total_string_bytes: None }
}

fn plan_has_reachable_local_parquet_source(plan: &QueryPlan) -> bool {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    plan.nodes().iter().any(|node| {
        !unreachable.contains(&node.id)
            && matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(source) if !source.files.is_empty()))
    })
}

/// The adapter executes this exact GPU-resident fragment after decoding a
/// logical BFS result. The stable input name is private runtime plumbing, not
/// a user-provided identifier; requiring the otherwise narrow source /
/// projection / limit shape prevents a generic GPU-resident query from being
/// silently treated as a modeled post-graph phase.
fn is_bfs_logical_output_projection_plan(plan: &QueryPlan) -> bool {
    const LOGICAL_OUTPUT_INPUT: &str = "__datafusion_nexus_cugraph_logical_output";

    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut has_logical_output_source = false;
    let mut has_projection_or_limit = false;
    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) if source.input_name == LOGICAL_OUTPUT_INPUT => {
                    has_logical_output_source = true;
                },
                _ => return false,
            },
            PlanNode::Projection(_) | PlanNode::Limit(_) => has_projection_or_limit = true,
            _ => return false,
        }
    }
    has_logical_output_source && has_projection_or_limit
}

fn minimum_parquet_source_read_budget_bytes() -> u64 {
    u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap_or(u64::MAX)
}

/// The DataFusion adapter always executes host-facing native plans through a
/// GPU-resident physical sink, retaining the host contract only for the final
/// cursor. Admission must therefore scope the physical plan rather than the
/// wrapper-only host boundary. GPU-resident plans remain unchanged.
fn canonical_execution_scope(plan: &QueryPlan) -> QueryPlan {
    plan.clone_with_deferred_host_materialization_sink()
}

/// Collision-free structural identity for the exact plan shape that enters the
/// engine. Formatting follows the full derived IR value rather than a hash;
/// the scope is short-lived private admission evidence, never an external key.
pub(crate) fn canonical_execution_scope_fingerprint(plan: &QueryPlan) -> String {
    format!("{:#?}", canonical_execution_scope(plan))
}

fn envelope_overflow_error(phase: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.admission.reservation_envelope",
        "attempt reservation envelope overflows u64",
    )
    .with_fact("reason", "attempt_reservation_envelope_overflow")
    .with_fact("phase", phase)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        admission::QueryPlanCapabilityProof,
        chunk::SourceChunkPlanningOptions,
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, SortExpr, SortSpec},
        graph_memory::{GraphMemoryCalibrationOverride, graph_memory_admission_evidence_for_edge_list},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphEdgeColumnBindings, ProjectionOutput, QueryPlanBuilder,
        },
        source::{
            GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, SourceNode,
            ZERO_COLUMN_COUNT_COLUMN,
        },
    };

    fn reservation_policy() -> AttemptReservationPolicy {
        AttemptReservationPolicy::new(
            64 * 1024,
            1,
            3,
            1,
            SourceSchedulerPolicy::new(2),
            SourceChunkPlanningMode::default(),
            SourceChunkPlanningOptions::default(),
        )
    }

    fn zero_column_count_plan(row_group_count: usize) -> QueryPlan {
        let row_groups = (0..row_group_count).collect::<Vec<_>>();
        let source = LocalParquetSource::zero_column_count(vec![
            LocalParquetFile::new("/tmp/admission-envelope-count.parquet")
                .with_row_groups(row_groups)
                .with_row_group_bytes(vec![4_096; row_group_count])
                .with_projected_row_group_bytes(vec![0; row_group_count], vec![0; row_group_count]),
        ]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column(ZERO_COLUMN_COUNT_COLUMN), "count")],
            ),
            Arc::new(Schema::new(vec![Field::new("count", DataType::UInt64, false)])),
        );
        builder.finish_with_host_sink(aggregate)
    }

    fn unmodeled_native_plan() -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "unmodeled",
            Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)])),
        )));
        builder.finish_with_host_sink(source)
    }

    fn graph_edge_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]))
    }

    fn bfs_spec() -> GraphAlgorithmSpec {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        )
    }

    fn local_graph_source(schema: Arc<Schema>) -> LocalParquetSource {
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/admission-envelope-graph.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![32], vec![96]),
            ],
            schema,
            vec!["src".to_owned(), "dst".to_owned()],
        )
    }

    fn graph_bfs_plan_with_local_source(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_graph_source(schema)));
        let graph = builder.add_graph_algorithm(source, bfs_spec());
        builder.finish_with_host_sink(graph)
    }

    fn graph_bfs_plan_with_gpu_resident_source(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("prepared-edges", schema)));
        let graph = builder.add_graph_algorithm(source, bfs_spec());
        builder.finish_with_host_sink(graph)
    }

    fn source_stage_plan(schema: Arc<Schema>) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(local_graph_source(schema)));
        builder.finish_gpu_resident(source)
    }

    fn bfs_logical_output_projection_plan() -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("reachable", DataType::Boolean, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "__datafusion_nexus_cugraph_logical_output",
            Arc::clone(&schema),
        )));
        let projection = builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("distance"), schema.field(1).as_ref().clone())],
        );
        builder.finish_gpu_resident(projection)
    }

    fn capability_free_literal_plan(data_type: DataType) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
        builder.finish_gpu_resident(source)
    }

    fn graph_result_passthrough_plan(input_name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
        builder.finish_with_host_sink(source)
    }

    fn graph_result_sort_consumer_plan(input_name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&schema))));
        let sort =
            builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]), schema);
        builder.finish_with_host_sink(sort)
    }

    #[test]
    fn capability_free_native_fragment_keeps_an_exact_reservation_scope() {
        let plan = capability_free_literal_plan(DataType::Int64);
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan)
            .expect("an executed native fragment must issue an exact-scope proof");

        assert!(proof.required_capabilities().is_empty(), "the scope-only proof must not add a capability allowance");
        assert_eq!(proof.required_floor_bytes_for(0).unwrap(), 0, "the scope-only proof must not raise admission");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.required_phase_floor_bytes(), 0, "a literal input has no source-read reservation");
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&plan)));

        let different_plan = capability_free_literal_plan(DataType::Int32);
        assert!(
            !envelope.scopes.iter().any(|scope| scope.matches(&different_plan)),
            "a scope-only proof must not authorize a different capability-free native fragment",
        );
    }

    #[test]
    fn two_row_group_zero_column_count_envelope_covers_a_retained_partial_and_next_source_read() {
        let plan = zero_column_count_plan(2);
        let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("host-sink plan issues a proof");
        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        let minimum_read = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();

        assert_eq!(envelope.complete_phase_peak_bytes(), Some(envelope.required_phase_floor_bytes()));
        assert!(
            envelope.required_phase_floor_bytes() > minimum_read * 2,
            "the second source read must coexist with the retained first partial and scalar finalization state"
        );
    }

    #[test]
    fn combining_an_unmodeled_fragment_clears_the_complete_phase_target() {
        let known = QueryPlanCapabilityProof::for_query_plan(&zero_column_count_plan(2))
            .expect("zero-column aggregate issues a proof");
        let known_envelope = known.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert!(known_envelope.complete_phase_peak_bytes().is_some());

        let unmodeled =
            QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("host-sink plan issues a proof");
        let combined = QueryPlanCapabilityProof::combine(known, [unmodeled]).unwrap();
        let combined_envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();

        assert_eq!(combined_envelope.complete_phase_peak_bytes(), None);
        assert!(combined_envelope.required_phase_floor_bytes() >= known_envelope.required_phase_floor_bytes());
    }

    #[test]
    fn graph_phase_envelope_is_complete_and_bound_to_its_exact_scope() {
        let schema = graph_edge_schema();
        let plan = graph_bfs_plan_with_local_source(Arc::clone(&schema));
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &bfs_spec(),
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("BFS edge-list model must produce admission evidence");
        let expected_peak = evidence.cold_model_with_source_stage_bytes(96);
        let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
            .expect("graph plan must issue its cuGraph capability proof")
            .with_graph_memory_admission_evidence(&evidence);

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.required_phase_floor_bytes(), expected_peak);
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_peak));

        let unmodeled =
            QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("host-sink plan issues a proof");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).unwrap();
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_accounts_for_one_composed_native_source_stage_without_losing_its_exact_scope() {
        let schema = graph_edge_schema();
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &bfs_spec(),
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("BFS edge-list model must produce admission evidence");
        let expected_graph_peak = evidence.cold_model_with_source_stage_bytes(96);
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(
            &graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema)),
        )
        .expect("graph plan must issue its cuGraph capability proof");
        let source_proof = QueryPlanCapabilityProof::for_query_plan(&source_stage_plan(schema))
            .expect("native source plan must issue a source-read capability proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [source_proof])
            .expect("graph and exact source proofs compose")
            .with_graph_memory_admission_evidence(&evidence);

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_graph_peak));
        assert_eq!(envelope.required_phase_floor_bytes(), expected_graph_peak);
    }

    #[test]
    fn graph_phase_accounts_for_one_attested_literal_graph_input_without_widening_other_scopes() {
        let schema = graph_edge_schema();
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &bfs_spec(),
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("BFS edge-list model must produce admission evidence");
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(
            &graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema)),
        )
        .expect("graph plan must issue its cuGraph capability proof");
        let graph_input = capability_free_literal_plan(DataType::Int64);
        let graph_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&graph_input)
            .expect("the exact graph-input fragment must issue a proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [graph_input_proof])
            .expect("graph and exact graph-input proofs compose")
            .with_graph_memory_admission_evidence(&evidence);

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(evidence.cold_model_requested_bytes()));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&graph_input)));

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_accounts_for_one_attested_result_passthrough_without_widening_other_scopes() {
        let schema = graph_edge_schema();
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &bfs_spec(),
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("BFS edge-list model must produce admission evidence");
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(
            &graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema)),
        )
        .expect("graph plan must issue its cuGraph capability proof");
        let passthrough = graph_result_passthrough_plan("graph_result");
        let passthrough_proof =
            QueryPlanCapabilityProof::for_attested_graph_result_passthrough_plan(&passthrough, "graph_result")
                .expect("the exact graph-result source handoff must issue its retained-scope proof");
        let proof = QueryPlanCapabilityProof::combine(
            graph_proof.with_graph_memory_admission_evidence(&evidence),
            [passthrough_proof],
        )
        .expect("a refined graph proof and its graph-result passthrough proof compose");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(evidence.cold_model_requested_bytes()));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&passthrough)));

        let different_input = QueryPlanCapabilityProof::for_attested_graph_result_passthrough_plan(
            &passthrough,
            "different_graph_result",
        );
        assert!(different_input.is_none(), "the graph frontend must attest the exact GPU-result input name");

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_accounts_for_one_attested_bounded_sort_consumer_without_widening_other_scopes() {
        let schema = graph_edge_schema();
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &bfs_spec(),
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("BFS edge-list model must produce admission evidence");
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(
            &graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema)),
        )
        .expect("graph plan must issue its cuGraph capability proof")
        .with_graph_memory_admission_evidence(&evidence);
        let sort_consumer = graph_result_sort_consumer_plan("graph_result");
        let sort_consumer_proof =
            QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan(&sort_consumer, "graph_result")
                .expect("the exact bounded graph-result sort must issue its retained-scope proof");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [sort_consumer_proof])
            .expect("a refined graph proof and its bounded sort proof compose");

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(evidence.cold_model_requested_bytes()));
        assert!(envelope.scopes.iter().any(|scope| scope.matches(&sort_consumer)));

        let unrelated = graph_result_passthrough_plan("different_graph_result");
        assert!(
            QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan(&unrelated, "graph_result")
                .is_none(),
            "the bounded consumer must consume the exact graph-result input",
        );

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }

    #[test]
    fn graph_phase_uses_the_maximum_of_native_and_typed_logical_bfs_output_phases() {
        let schema = graph_edge_schema();
        let evidence = graph_memory_admission_evidence_for_edge_list(
            &bfs_spec(),
            &schema,
            10,
            None,
            GraphMemoryCalibrationOverride::none(),
            None,
        )
        .expect("BFS edge-list model must produce admission evidence");
        let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(
            &graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema)),
        )
        .expect("graph plan must issue its cuGraph capability proof");
        let logical_output_proof = QueryPlanCapabilityProof::for_query_plan(&bfs_logical_output_projection_plan())
            .expect("logical BFS output plan must retain its exact scope");
        let proof = QueryPlanCapabilityProof::combine(graph_proof, [logical_output_proof])
            .expect("graph and logical BFS output proofs compose")
            .with_graph_memory_admission_evidence(&evidence);

        let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
        let expected_peak = evidence
            .cold_model_requested_bytes()
            .max(evidence.post_graph_bfs_logical_output_peak_bytes().expect("BFS has a typed logical-output phase"));
        assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_peak));
        assert_eq!(envelope.required_phase_floor_bytes(), expected_peak);

        let unmodeled = QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan())
            .expect("unmodeled fragment has exact scope");
        let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
        let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    }
}
