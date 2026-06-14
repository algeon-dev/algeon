use super::*;

pub(super) fn d5_test_gate_row(
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    required_contract: &'static str,
    contract_quality: ContractQuality,
    unresolved_preflight_gap: Option<&'static str>,
) -> StrictBoundedContractGateRow {
    let contract_scope = d5_test_scope_for_reservation_kind(reservation_kind);
    StrictBoundedContractGateRow {
        execution_mode: StrictBoundedExecutionMode::StrictBounded,
        contract_scope,
        gate_id: StrictBoundedContractGateId::Compiled(ReservationGateId::new(reservation_kind as usize)),
        owner: d5_test_owner_for_scope(contract_scope),
        reservation_kind,
        memory_boundary,
        required_contract,
        contract_status: ContractStatus::Modeled,
        contract_quality,
        unresolved_preflight_gap,
    }
}

pub(super) fn d5_test_scope_for_reservation_kind(reservation_kind: ReservationKind) -> ReservationRequirementScope {
    match reservation_kind {
        ReservationKind::OperatorOutput
        | ReservationKind::HashTable
        | ReservationKind::FilteredProbeState
        | ReservationKind::MixedSizeData
        | ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge => ReservationRequirementScope::Node,
        ReservationKind::SourceRead
        | ReservationKind::BuildSide
        | ReservationKind::ProbeSide
        | ReservationKind::GatherMap
        | ReservationKind::JoinOutput
        | ReservationKind::FinalSort
        | ReservationKind::Materialization => ReservationRequirementScope::Edge,
        ReservationKind::CapabilityContract => ReservationRequirementScope::Subgraph,
    }
}

pub(super) fn d5_test_owner_for_scope(scope: ReservationRequirementScope) -> StrictBoundedContractOwner {
    match scope {
        ReservationRequirementScope::Node => {
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Task(PipelineTaskId::new(0)))
        },
        ReservationRequirementScope::Edge => {
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(PipelineEdgeId::new(0)))
        },
        ReservationRequirementScope::Subgraph => {
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Barrier(PipelineBarrierId::new(0)))
        },
    }
}

// Expected support row for the D3.1 compiled-pipeline inventory tests.
pub(super) struct D31InventoryCase {
    pub(super) label: &'static str,
    pub(super) plan: QueryPlan,
    pub(super) admits: fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
    pub(super) graph_support: CompiledBoundedPipelineSupport,
    pub(super) plan_support: CompiledBoundedPipelineSupport,
    pub(super) render_fragments: &'static [&'static str],
}

pub(super) fn d3_1_inventory_cases() -> Vec<D31InventoryCase> {
    vec![
        D31InventoryCase {
            label: "shared resident producer fan-out",
            plan: keyed_hash_join_resident_subplan_pipeline_plan(
                KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily,
            ),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_keyed_hash_join_resident_subplan_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=shared_resident_producer",
                "kind=aggregate_boundary",
                "kind=join_build_before_probe",
                "consumers=t",
                "gates=g",
            ],
        },
        D31InventoryCase {
            label: "keyless conditional join input-subplan pipeline",
            plan: keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted),
            admits:
                nexus_query_engine::capability::CapabilityAnalysis::admits_keyless_conditional_join_input_subplan_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=unary",
                "kind=join_build_before_probe",
                "kind=aggregate_boundary",
                "kind=sink_boundary",
            ],
        },
        D31InventoryCase {
            label: "keyless aggregate-over-join subplan inputs",
            plan: keyless_conditional_aggregate_over_join_subplan_inputs_plan(
                AggregateOverJoinSubplanInputsVariant::Admitted,
            ),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=join_materialization",
                "kind=aggregate_boundary",
                "kind=join_build_before_probe",
            ],
        },
        D31InventoryCase {
            label: "resident grouped aggregate input",
            plan: resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=final_sort",
                "kind=limit",
                "kind=aggregate_boundary",
                "kind=sort_boundary",
            ],
        },
        D31InventoryCase {
            label: "mixed keyed-residual grouped aggregate input",
            plan: residual_resident_grouped_aggregate_input_plan(
                ResidualResidentGroupedAggregateInputVariant::BlockedFamily,
            ),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=unary",
                "kind=join_build_input",
                "kind=join_probe_input",
                "kind=aggregate_boundary",
            ],
        },
        D31InventoryCase {
            label: "filtered left-semi join strategy",
            plan: root_filtered_left_only_join_plan(JoinType::LeftSemi),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_root_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "kind=join_build_before_probe",
                "kind=join_materialization",
            ],
        },
        D31InventoryCase {
            label: "filtered left-anti join strategy",
            plan: root_filtered_left_only_join_plan(JoinType::LeftAnti),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_root_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "kind=join_build_before_probe",
                "kind=join_materialization",
            ],
        },
        D31InventoryCase {
            label: "mixed residual multi-join strategy",
            plan: admitted_mixed_residual_multi_join_plan(),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_closed_multi_join_subgraph,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "kind=join_build_before_probe",
                "kind=join_materialization",
            ],
        },
        D31InventoryCase {
            label: "large keyed partitioned equality join strategy",
            plan: large_keyed_join_strategy_plan(JoinType::Inner),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_root_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "contract=keyed_equality_join_output_size_preflight",
                "quality=hard_budgeted_report_with_unresolved_preflight",
            ],
        },
        D31InventoryCase {
            label: "broad multi-join TopK with aggregate-over-join input",
            plan: aggregate_over_join_subplan_topk_plan(),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_multi_join_aggregate_over_join_subplan_topk_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=final_sort",
                "kind=limit",
                "kind=aggregate_boundary",
                "kind=sort_boundary",
            ],
        },
        D31InventoryCase {
            label: "rank-window branches feed sink-path union",
            plan: union_of_ranked_join_branches_plan(),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_strict_bounded_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=window",
                "kind=union",
                "kind=aggregate",
                "kind=sort_boundary",
                "kind=aggregate_boundary",
            ],
        },
        D31InventoryCase {
            label: "resident grouped aggregate subquery-filter TopK path",
            plan: resident_grouped_aggregate_subquery_filter_plan(),
            admits: nexus_query_engine::capability::CapabilityAnalysis::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=final_sort",
                "kind=join_materialization",
                "kind=aggregate_boundary",
                "kind=sort_boundary",
            ],
        },
    ]
}

pub(super) fn assert_compiled_graph_inventory(label: &str, pipeline: &CompiledPipeline) {
    assert!(!pipeline.tasks().is_empty(), "{label}: compiled graph should carry explicit tasks");
    assert!(!pipeline.edges().is_empty(), "{label}: compiled graph should carry explicit task edges");
    assert!(!pipeline.barriers().is_empty(), "{label}: compiled graph should carry explicit barriers");
    assert!(!pipeline.resources().is_empty(), "{label}: compiled graph should carry explicit resident resources");
    assert!(!pipeline.gates().is_empty(), "{label}: compiled graph should carry explicit reservation gates");
    assert_eq!(pipeline.sink().kind(), SinkBoundaryKind::HostArrowMaterialization, "{label}: expected host sink");

    let rendered = pipeline.debug_render();
    for expected_section in ["tasks:", "edges:", "barriers:", "resources:", "gates:", "sink:"] {
        assert!(rendered.contains(expected_section), "{label}: graph rendering should expose {expected_section}");
    }
}

pub(super) fn assert_no_legacy_execution_selection_terms(label: &str, rendered: &str) {
    for forbidden in D5_FORBIDDEN_FAMILY_PATH_REPORT_TERMS {
        assert!(
            !rendered.contains(forbidden),
            "{label}: compiled graph diagnostics must not contain legacy execution selection term {forbidden}"
        );
    }
}

pub(super) fn assert_d5_contract_gate_row_is_family_path_free(label: &str, row: &str) {
    for forbidden in D5_FORBIDDEN_FAMILY_PATH_REPORT_TERMS {
        assert!(
            !row.contains(forbidden),
            "{label}: D.5 strict bounded contract gate rows must not contain legacy family/path report term {forbidden}"
        );
    }
}

pub(super) fn assert_d5_contract_gate_row_excludes_adaptive_terms(label: &str, row: &str) {
    for forbidden in D5_FORBIDDEN_ADAPTIVE_REPORT_TERMS {
        assert!(
            !row.contains(forbidden),
            "{label}: D.5 strict bounded contract gate rows must not contain adaptive-only term {forbidden}"
        );
    }
}

pub(super) const D5_FORBIDDEN_FAMILY_PATH_REPORT_TERMS: &[&str] = &[
    "BoundedJoinExecutionFamily",
    "BoundedJoinExecutionPath",
    "bounded_join_execution_family",
    "bounded_join_execution_families",
    "bounded_join_execution_family_contracts",
    "bounded_join_execution_path",
    "bounded_join_blocked_join_family",
    "bounded_join_required_shape_contract",
    "bounded_join_contract_modes",
    "bounded_join_unresolved_preflight_contracts",
    "root_reusable_hash_join",
    "root_filtered_semi_anti_join",
    "keyless_conditional_join",
    "keyless_conditional_join_input_subplan_pipeline",
    "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline",
    "root_projected_cross_join",
    "root_mixed_keyed_residual_join",
    "multi_join_reusable_hash_pipeline",
    "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline",
    "multi_join_with_aggregate_over_join_subplan_topk",
    "mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline",
    "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk",
    "multi_join_with_resident_grouped_aggregate_subquery_filter",
    "keyed_hash_join_resident_subplan_pipeline",
];

pub(super) const D5_FORBIDDEN_ADAPTIVE_REPORT_TERMS: &[&str] =
    &["retry", "spill", "downgrade", "memory_history", "memory-history", "memory history"];

// Scheduler test double that records resource lifecycle and injected failures.
pub(super) struct MockExecutor {
    pub(super) executed: Vec<PipelineTaskId>,
    pub(super) released: Vec<ResidentResourceId>,
    pub(super) consumed: Vec<ResidentResourceId>,
    pub(super) task_local_live: Vec<&'static str>,
    pub(super) task_local_released: Vec<&'static str>,
    pub(super) cleanup_calls: usize,
    pub(super) fail_tasks: Vec<PipelineTaskId>,
    pub(super) omit_outputs: Vec<PipelineTaskId>,
    pub(super) duplicate_outputs: Vec<PipelineTaskId>,
    pub(super) consume_inputs: Vec<PipelineTaskId>,
    pub(super) task_local_reservations: Vec<(PipelineTaskId, &'static str)>,
    pub(super) simulate_scheduler_memory_pressure: bool,
    pub(super) simulated_live_reserved_bytes: u64,
    pub(super) simulated_remaining_bytes: Option<u64>,
    pub(super) task_admission_required_bytes: Vec<(PipelineTaskId, u64)>,
    pub(super) remaining_bytes_after_tasks: Vec<(PipelineTaskId, u64)>,
}

impl MockExecutor {
    pub(super) fn new() -> Self {
        Self {
            executed: Vec::new(),
            released: Vec::new(),
            consumed: Vec::new(),
            task_local_live: Vec::new(),
            task_local_released: Vec::new(),
            cleanup_calls: 0,
            fail_tasks: Vec::new(),
            omit_outputs: Vec::new(),
            duplicate_outputs: Vec::new(),
            consume_inputs: Vec::new(),
            task_local_reservations: Vec::new(),
            simulate_scheduler_memory_pressure: false,
            simulated_live_reserved_bytes: 0,
            simulated_remaining_bytes: None,
            task_admission_required_bytes: Vec::new(),
            remaining_bytes_after_tasks: Vec::new(),
        }
    }

    pub(super) fn fail_task_for(mut self, task: PipelineTaskId) -> Self {
        self.fail_tasks.push(task);
        self
    }

    pub(super) fn omit_output_for(mut self, task: PipelineTaskId) -> Self {
        self.omit_outputs.push(task);
        self
    }

    pub(super) fn duplicate_output_for(mut self, task: PipelineTaskId) -> Self {
        self.duplicate_outputs.push(task);
        self
    }

    pub(super) fn consume_inputs_for(mut self, task: PipelineTaskId) -> Self {
        self.consume_inputs.push(task);
        self
    }

    pub(super) fn retain_task_local_reservation_for(mut self, task: PipelineTaskId, reservation: &'static str) -> Self {
        self.task_local_reservations.push((task, reservation));
        self
    }

    pub(super) fn with_scheduler_memory_pressure(mut self) -> Self {
        self.simulate_scheduler_memory_pressure = true;
        self
    }

    pub(super) fn with_admission_remaining_bytes(mut self, remaining_bytes: u64) -> Self {
        self.simulated_remaining_bytes = Some(remaining_bytes);
        self
    }

    pub(super) fn with_task_admission_required_bytes(mut self, task: PipelineTaskId, required_bytes: u64) -> Self {
        self.task_admission_required_bytes.push((task, required_bytes));
        self
    }

    pub(super) fn with_remaining_bytes_after_task(mut self, task: PipelineTaskId, remaining_bytes: u64) -> Self {
        self.remaining_bytes_after_tasks.push((task, remaining_bytes));
        self
    }

    fn task_admission_required_bytes(&self, task: PipelineTaskId) -> u64 {
        self.task_admission_required_bytes
            .iter()
            .rev()
            .find_map(|(candidate, bytes)| (*candidate == task).then_some(*bytes))
            .unwrap_or(0)
    }
}

impl CompiledTaskExecutor for MockExecutor {
    type Resource = String;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> nexus_query_engine::Result<CompiledTaskOutput<Self::Resource>> {
        self.executed.push(task.id());
        if self.fail_tasks.contains(&task.id()) {
            return Err(nexus_query_engine::QueryError::new(
                nexus_query_engine::QueryErrorCode::Internal,
                "nexus_query_engine.tests.mock_executor",
                "mock compiled executor failure",
            )
            .with_fact("compiled_task_id", task.id()));
        }
        if self.consume_inputs.contains(&task.id()) {
            for resource in task.input_resources() {
                let consumed = state.consume_task_input_resource(task, *resource)?;
                self.consumed.push(consumed.id());
                drop(consumed.into_payload());
            }
        }
        for (_, reservation) in self.task_local_reservations.iter().filter(|(owner, _)| *owner == task.id()) {
            self.task_local_live.push(*reservation);
        }
        if self.omit_outputs.contains(&task.id()) {
            return Ok(CompiledTaskOutput::empty());
        }
        let mut outputs = task
            .output_resources()
            .iter()
            .map(|resource| (*resource, format!("{}:{resource}", task.id())))
            .collect::<Vec<_>>();
        if self.duplicate_outputs.contains(&task.id())
            && let Some((resource, payload)) = outputs.first().cloned()
        {
            outputs.push((resource, payload));
        }
        if self.simulate_scheduler_memory_pressure {
            self.simulated_live_reserved_bytes =
                self.simulated_live_reserved_bytes.saturating_add(outputs.len() as u64);
        }
        if let Some((_, remaining_bytes)) =
            self.remaining_bytes_after_tasks.iter().rev().find(|(completed_task, _)| *completed_task == task.id())
        {
            self.simulated_remaining_bytes = Some(*remaining_bytes);
        }
        Ok(CompiledTaskOutput::new(outputs))
    }

    fn release_resource(
        &mut self,
        resource: ResidentResourceId,
        _payload: Self::Resource,
    ) -> nexus_query_engine::Result<()> {
        self.released.push(resource);
        if self.simulate_scheduler_memory_pressure {
            self.simulated_live_reserved_bytes = self.simulated_live_reserved_bytes.saturating_sub(1);
        }
        Ok(())
    }

    fn cleanup_after_error(&mut self) -> nexus_query_engine::Result<()> {
        self.cleanup_calls = self.cleanup_calls.saturating_add(1);
        self.task_local_released.extend(self.task_local_live.drain(..).rev());
        Ok(())
    }

    fn live_reserved_bytes(&self) -> Option<u64> {
        self.simulate_scheduler_memory_pressure.then_some(self.simulated_live_reserved_bytes)
    }

    fn remaining_reserved_bytes(&self) -> Option<u64> {
        self.simulated_remaining_bytes
    }

    fn can_admit_task(
        &self,
        task: &PipelineTask,
        _state: &CompiledPipelineState<Self::Resource>,
    ) -> nexus_query_engine::Result<CompiledTaskAdmission> {
        let Some(remaining_bytes) = self.simulated_remaining_bytes else {
            return Ok(CompiledTaskAdmission::admitted());
        };
        let required_bytes = self.task_admission_required_bytes(task.id());
        if required_bytes <= remaining_bytes {
            return Ok(CompiledTaskAdmission::admitted_with_bytes(required_bytes, remaining_bytes));
        }
        Ok(CompiledTaskAdmission::not_ready(required_bytes, remaining_bytes, "mock_remaining_budget_too_small"))
    }
}

pub(super) fn runtime_visible_gate_for_task_kind(
    pipeline: &CompiledPipeline,
    task_kind: PipelineTaskKind,
    reservation_kind: ReservationKind,
) -> Option<&ReservationGate> {
    let task = pipeline.tasks().iter().find(|task| task.kind() == task_kind)?;
    task.gates()
        .iter()
        .chain(
            pipeline
                .edges()
                .iter()
                .filter(|edge| edge.from() == task.id() || edge.to() == task.id())
                .flat_map(|edge| edge.gates()),
        )
        .chain(
            pipeline
                .barriers()
                .iter()
                .filter(|barrier| {
                    barrier.upstream_tasks().contains(&task.id()) || barrier.downstream_tasks().contains(&task.id())
                })
                .flat_map(|barrier| barrier.gates()),
        )
        .chain((pipeline.sink().input() == task.id()).then_some(pipeline.sink().gates()).into_iter().flatten())
        .find_map(|gate_id| {
            pipeline.gates().iter().find(|gate| gate.id() == *gate_id && gate.reservation_kind() == reservation_kind)
        })
}

pub(super) fn literal_expr_for_type(data_type: &DataType) -> Expr {
    let literal = match data_type {
        DataType::Decimal128(precision, scale) => {
            LiteralValue::Decimal128 { raw: 12_345, precision: *precision, scale: *scale }
        },
        _ => LiteralValue::Int32(12_345),
    };
    Expr::literal(literal).alias("secret_literal")
}

pub(super) fn int32_values(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let array = batch.column(column).as_any().downcast_ref::<Int32Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

pub(super) fn int64_values(batch: &RecordBatch, column: usize) -> Vec<Option<i64>> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

pub(super) fn int32_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<i32>, Option<i64>)> {
    let left = int32_values(batch, left_column);
    let right = int64_values(batch, right_column);
    let mut pairs = left.into_iter().zip(right).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

pub(super) fn string_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<String>, Option<i64>)> {
    let left = string_values(batch, left_column);
    let right = int64_values(batch, right_column);
    let mut pairs = left.into_iter().zip(right).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn string_values(batch: &RecordBatch, column: usize) -> Vec<Option<String>> {
    let array = batch.column(column).as_any().downcast_ref::<StringArray>().unwrap();
    (0..array.len())
        .map(|index| if array.is_null(index) { None } else { Some(array.value(index).to_string()) })
        .collect()
}

type Int32Quad = (Option<i32>, Option<i32>, Option<i32>, Option<i32>);

type Int32Triple = (Option<i32>, Option<i32>, Option<i32>);

pub(super) fn int32_triples_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
) -> Vec<Int32Triple> {
    let first = int32_values(batch, first_column);
    let second = int32_values(batch, second_column);
    let third = int32_values(batch, third_column);
    let mut triples = first
        .into_iter()
        .zip(second)
        .zip(third)
        .map(|((first, second), third)| (first, second, third))
        .collect::<Vec<_>>();
    triples.sort();
    triples
}

pub(super) fn int32_quads_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
    fourth_column: usize,
) -> Vec<Int32Quad> {
    let first = int32_values(batch, first_column);
    let second = int32_values(batch, second_column);
    let third = int32_values(batch, third_column);
    let fourth = int32_values(batch, fourth_column);
    let mut quads = first
        .into_iter()
        .zip(second)
        .zip(third)
        .zip(fourth)
        .map(|(((first, second), third), fourth)| (first, second, third, fourth))
        .collect::<Vec<_>>();
    quads.sort();
    quads
}

pub(super) fn assert_task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    assert_eq!(task_kind_count(pipeline, kind), expected, "unexpected count for {kind:?} tasks");
}

pub(super) fn task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> usize {
    pipeline.tasks().iter().filter(|task| task.kind() == kind).count()
}

pub(super) fn barrier_kind_count(pipeline: &CompiledPipeline, kind: PipelineBarrierKind) -> usize {
    pipeline.barriers().iter().filter(|barrier| barrier.kind() == kind).count()
}

pub(super) fn sink_input_edge_count(pipeline: &CompiledPipeline) -> usize {
    pipeline.edges().iter().filter(|edge| edge.kind() == PipelineEdgeKind::SinkInput).count()
}

pub(super) fn bounded_context(runtime: Arc<cudf::Runtime>) -> ExecutionContext {
    bounded_context_with_mode(runtime, NativePipelineExecutionMode::StrictBounded)
}

pub(super) fn bounded_context_with_mode(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
) -> ExecutionContext {
    bounded_context_builder(runtime, mode).build().unwrap()
}

pub(super) fn bounded_context_with_mode_and_memory_history(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
    memory_history: Arc<AdaptiveMemoryHistory>,
) -> ExecutionContext {
    bounded_context_builder(runtime, mode).with_adaptive_memory_history(memory_history).build().unwrap()
}

pub(super) fn bounded_context_builder(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
) -> nexus_query_engine::ExecutionContextBuilder {
    ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_native_pipeline_execution_mode(mode)
}

pub(super) fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
