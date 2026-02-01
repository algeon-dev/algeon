use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnRef, CompiledPipeline, ExecutionContext, Expr,
    JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitSpec, LiteralValue, LocalParquetSource,
    MemoryControlPolicy, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
    PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ProjectionOutput, QueryMetrics, QueryPlan,
    QueryPlanBuilder, ReservationGate, ReservationGateId, ReservationGateOwner, ReservationKind, ResidentLifetime,
    ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind, SortExpr, SortSpec, SourceNode,
    capability::{ContractQuality, ContractStatus, ReservationRequirementScope},
    compile_pipeline,
    exec::{
        CompiledBoundedPipelineSupport, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
        CompiledTaskExecutor, CompiledTaskOutput, compiled_bounded_pipeline_support,
        compiled_bounded_query_plan_support,
    },
    execute_query_plan,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use crate::bounded_join_admission::plans::single_join_post_root_aggregate_sort_limit_plan;

#[test]
fn compiled_bounded_support_accepts_root_join_graph_without_legacy_routing_terms() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("root_join"));

    let source_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("root join should retain source-read gates");
    assert_eq!(source_gate.status(), ContractStatus::Modeled);
    assert_eq!(source_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
}

#[test]
fn compiled_bounded_support_rejects_non_d1_task_envelope_before_execution() {
    let plan = aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate sort limit plan should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_accepts_single_join_post_root_d2_tasks_without_legacy_routing_terms() {
    let plan = single_join_post_root_aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert!(pipeline.barriers().iter().any(|barrier| barrier.kind() == PipelineBarrierKind::AggregateBoundary));
    assert!(pipeline.barriers().iter().any(|barrier| barrier.kind() == PipelineBarrierKind::SortBoundary));
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let final_sort_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::FinalSort)
        .expect("single-join post-root plan should retain final-sort gates");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_rejects_unmodeled_reservation_gates_before_execution() {
    let pipeline = root_join_shaped_pipeline_with_blocked_gate();

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_query_plan_support_rejects_non_hash_root_join_before_execution() {
    let plan = keyless_root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph shape should still match the structural envelope"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_scheduler_runs_tasks_from_dependencies_and_releases_last_consumers() {
    let pipeline = out_of_order_join_barrier_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new();

    let state = scheduler.run(&mut executor).expect("mock pipeline should execute");

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(2),
            PipelineTaskId::new(1),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
        ]
    );
    assert_eq!(executor.executed, state.execution_order());
    assert_eq!(executor.released, vec![ResidentResourceId::new(0), ResidentResourceId::new(1)]);
    assert_eq!(state.live_resource_count(), 0);
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
    assert!(state.is_resource_released(ResidentResourceId::new(1)));
}

#[test]
fn compiled_scheduler_errors_when_task_omits_declared_output_resource() {
    let pipeline = simple_unary_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new().omit_output_for(PipelineTaskId::new(0));

    let error = scheduler.run(&mut executor).expect_err("missing output should fail");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t0"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
}

#[test]
fn compiled_scheduler_state_errors_when_input_resource_is_missing() {
    let pipeline = simple_unary_pipeline();
    let state = CompiledPipelineState::<String>::new(&pipeline).expect("state should initialize");

    let error = state.prepare_task(&pipeline.tasks()[1]).expect_err("downstream task should require produced input");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t1"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
}

#[test]
fn compiled_scheduler_errors_when_task_reports_duplicate_produced_resource() {
    let pipeline = simple_unary_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new().duplicate_output_for(PipelineTaskId::new(0));

    let error = scheduler.run(&mut executor).expect_err("duplicate output should fail");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t0"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
}

#[test]
fn compiled_scheduler_errors_on_premature_last_consumer_release() {
    let pipeline = premature_last_consumer_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new();

    let error = scheduler.run(&mut executor).expect_err("premature release should fail");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
    assert_eq!(error.facts().get("compiled_task_id"), Some("t1"));
    assert_eq!(error.facts().get("incomplete_consumer_task"), Some("t2"));
}

#[test]
fn compiled_gpu_executor_runs_source_filter_projection_prefix() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d13");
    let path = dir.join("input.parquet");
    let source_schema = write_source_unary_parquet(&path);
    let plan = source_filter_projection_gpu_resident_plan(&path, Arc::clone(&source_schema));
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("source/unary plan should compile");
    let final_resource = pipeline
        .resources()
        .iter()
        .find(|resource| matches!(resource.lifetime(), ResidentLifetime::SinkOutput { .. }))
        .expect("gpu-resident plan should keep the root resource live")
        .id();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut metrics = QueryMetrics::default();
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();

    let state = {
        let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();
        scheduler.run(&mut executor).expect("compiled source/unary prefix should execute")
    };
    let output = state.resource_payload(final_resource).expect("final resource should remain GPU-resident");

    assert_eq!(state.execution_order(), &[PipelineTaskId::new(0), PipelineTaskId::new(1), PipelineTaskId::new(2)]);
    assert_eq!(output.num_rows(), 3);
    assert_eq!(output.num_columns(), 1);
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert!(metrics.reservation_granted_bytes > 0);
}

#[test]
fn compiled_gpu_executor_runs_root_hash_join_build_probe_materialize() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d14");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let compiled_plan = root_join_data_plan(&left_path, &right_path);
    let legacy_plan = root_join_data_plan(&left_path, &right_path);
    let analysis = compiled_plan.capability_analysis();
    let pipeline = compile_pipeline(&compiled_plan, &analysis).expect("root join should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut compiled_metrics = QueryMetrics::default();
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor =
            CompiledGpuTaskExecutor::new(&context, &compiled_plan, &pipeline, &gpu_inputs, &mut compiled_metrics)
                .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled root hash join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };
    let production_result = execute_query_plan(&context, &legacy_plan).expect("bounded root join should execute");
    let production_batch = production_result.output().as_host_arrow().expect("production output should be host Arrow");

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(2),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
            PipelineTaskId::new(5),
        ]
    );
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(compiled_batch.num_rows(), production_batch.num_rows());
    assert_eq!(int32_values(&compiled_batch, 0), int32_values(production_batch, 0));
    assert_eq!(int32_values(&compiled_batch, 1), int32_values(production_batch, 1));
    assert_eq!(int32_values(&compiled_batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(&compiled_batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(compiled_metrics.hash_join_build_count, 1);
    assert!(compiled_metrics.hash_join_hash_table_bytes > 0);
}

#[test]
fn execute_query_plan_routes_root_hash_join_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-root");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("root join should execute through compiled scheduler");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_root_hash_join_with_unary_inputs_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-unary-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_with_unary_inputs_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join with unary inputs should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("root join with unary inputs should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_leaves_unsupported_compiled_graphs_on_legacy_fallback() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-fallback");
    let path = dir.join("input.parquet");
    let source_schema = write_source_unary_parquet(&path);
    let plan = aggregate_sort_limit_data_plan(&path, source_schema);
    let context = bounded_context(Arc::new(runtime));

    let result =
        execute_query_plan(&context, &plan).expect("unsupported D.1 compiled graph should use legacy fallback");
    let batch = result.output().as_host_arrow().expect("fallback output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 4);
    assert_eq!(metrics.compiled_pipeline_attempts, 0);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 0);
}

#[test]
fn execute_query_plan_routes_single_join_post_root_d2_graph_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d22-single-join");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_sort_limit_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2 graph should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(batch, 0), vec![Some(3), Some(2)]);
    assert_eq!(int64_values(batch, 1), vec![Some(0), Some(1)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn compiled_host_sink_diagnostics_are_structured_without_legacy_routing_terms() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(Arc::new(runtime));
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();
    let mut metrics = QueryMetrics::default();
    let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();

    let error = executor.take_host_output().expect_err("empty host sink should be a structured invariant error");
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t5"));
    assert_eq!(error.facts().get("sink_boundary_kind"), Some("host_arrow_materialization"));
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("bounded_join_execution_path"));
}

struct MockExecutor {
    executed: Vec<PipelineTaskId>,
    released: Vec<ResidentResourceId>,
    omit_outputs: Vec<PipelineTaskId>,
    duplicate_outputs: Vec<PipelineTaskId>,
}

impl MockExecutor {
    fn new() -> Self {
        Self { executed: Vec::new(), released: Vec::new(), omit_outputs: Vec::new(), duplicate_outputs: Vec::new() }
    }

    fn omit_output_for(mut self, task: PipelineTaskId) -> Self {
        self.omit_outputs.push(task);
        self
    }

    fn duplicate_output_for(mut self, task: PipelineTaskId) -> Self {
        self.duplicate_outputs.push(task);
        self
    }
}

impl CompiledTaskExecutor for MockExecutor {
    type Resource = String;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        _state: &CompiledPipelineState<Self::Resource>,
    ) -> cudf_query_engine::Result<CompiledTaskOutput<Self::Resource>> {
        self.executed.push(task.id());
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
        Ok(CompiledTaskOutput::new(outputs))
    }

    fn release_resource(
        &mut self,
        resource: ResidentResourceId,
        _payload: Self::Resource,
    ) -> cudf_query_engine::Result<()> {
        self.released.push(resource);
        Ok(())
    }
}

fn root_join_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn keyless_root_join_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-keyless-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-keyless-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "left_id", "left_id"),
                BinaryOp::Lt,
                Expr::input_column(1, "right_id", "right_id"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
            ])
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn root_join_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn root_join_with_unary_inputs_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_source_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_source_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(&left_source_schema),
        vec!["left_id".into(), "left_value".into()],
    )));
    let left_filter = builder.add_filter(
        left,
        Expr::binary(Expr::column("left_id"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(1))),
        Arc::clone(&left_source_schema),
    );
    let left_projected = builder.add_projection(
        left_filter,
        vec![
            ProjectionOutput::new(Expr::column("left_id"), Field::new("left_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("left_value"), Field::new("left_value", DataType::Int32, false)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(&right_source_schema),
        vec!["right_id".into(), "right_value".into()],
    )));
    let right_filter = builder.add_filter(
        right,
        Expr::binary(Expr::column("right_id"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(5))),
        Arc::clone(&right_source_schema),
    );
    let right_projected = builder.add_projection(
        right_filter,
        vec![
            ProjectionOutput::new(Expr::column("right_id"), Field::new("right_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("right_value"), Field::new("right_value", DataType::Int32, false)),
        ],
    );
    let join = builder.add_join(
        left_projected,
        right_projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn single_join_post_root_aggregate_sort_limit_data_plan(customers_path: &Path, orders_path: &Path) -> QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        customers_path,
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        orders_path,
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        join,
        vec![
            ProjectionOutput::new(Expr::column("c_custkey"), Field::new("c_custkey", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("o_orderkey"), Field::new("o_orderkey", DataType::Int32, true)),
        ],
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("c_count"), false, false),
            SortExpr::new(Expr::column("c_custkey"), true, false),
        ]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

fn simple_unary_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let unary = PipelineTaskId::new(1);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(unary, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, unary, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![unary]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![unary]),
        ],
        vec![ResidentResource::new(
            frame,
            source,
            vec![unary],
            ResidentLifetime::BetweenTasks { producer: source, last_consumer: unary },
        )],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, unary),
    )
}

fn out_of_order_join_barrier_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let probe = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let materialize = PipelineTaskId::new(3);
    let sink = PipelineTaskId::new(4);
    let source_frame = ResidentResourceId::new(0);
    let join_output = ResidentResourceId::new(1);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![source_frame]),
            PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![source_frame]),
            PipelineTask::new(build, PipelineTaskKind::JoinBuild).with_input_resources(vec![source_frame]),
            PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
            PipelineTask::new(sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, probe, PipelineEdgeKind::JoinProbeInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), source, build, PipelineEdgeKind::JoinBuildInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![join_output]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![build, probe]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![build])
                .with_downstream_tasks(vec![probe]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![probe])
                .with_downstream_tasks(vec![materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![
            ResidentResource::new(
                source_frame,
                source,
                vec![build, probe],
                ResidentLifetime::BetweenTasks { producer: source, last_consumer: probe },
            ),
            ResidentResource::new(
                join_output,
                materialize,
                vec![sink],
                ResidentLifetime::BetweenTasks { producer: materialize, last_consumer: sink },
            ),
        ],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    )
}

fn premature_last_consumer_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let early_last_consumer = PipelineTaskId::new(1);
    let later_consumer = PipelineTaskId::new(2);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(early_last_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
            PipelineTask::new(later_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, early_last_consumer, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), source, later_consumer, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![early_last_consumer, later_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![later_consumer]),
        ],
        vec![ResidentResource::new(
            frame,
            source,
            vec![early_last_consumer, later_consumer],
            ResidentLifetime::BetweenTasks { producer: source, last_consumer: early_last_consumer },
        )],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, later_consumer),
    )
}

fn aggregate_sort_limit_plan() -> QueryPlan {
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-aggregate.parquet",
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("v_count"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

fn aggregate_sort_limit_data_plan(path: &Path, source_schema: Arc<Schema>) -> QueryPlan {
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("k"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

fn root_join_shaped_pipeline_with_blocked_gate() -> CompiledPipeline {
    let left_source = PipelineTaskId::new(0);
    let right_source = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let probe = PipelineTaskId::new(3);
    let materialize = PipelineTaskId::new(4);
    let sink_task = PipelineTaskId::new(5);

    let left_frame = ResidentResourceId::new(0);
    let right_frame = ResidentResourceId::new(1);
    let join_output = ResidentResourceId::new(2);
    let blocked_gate = ReservationGateId::new(0);

    let tasks = vec![
        PipelineTask::new(left_source, PipelineTaskKind::SourceRead).with_output_resources(vec![left_frame]),
        PipelineTask::new(right_source, PipelineTaskKind::SourceRead).with_output_resources(vec![right_frame]),
        PipelineTask::new(build, PipelineTaskKind::JoinBuild)
            .with_input_resources(vec![right_frame])
            .with_gates(vec![blocked_gate]),
        PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![left_frame]),
        PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
        PipelineTask::new(sink_task, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
    ];

    let edges = vec![
        PipelineEdge::new(PipelineEdgeId::new(0), right_source, build, PipelineEdgeKind::JoinBuildInput)
            .with_resources(vec![right_frame]),
        PipelineEdge::new(PipelineEdgeId::new(1), left_source, probe, PipelineEdgeKind::JoinProbeInput)
            .with_resources(vec![left_frame]),
        PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![join_output]),
    ];

    let barriers = vec![
        PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
            .with_upstream_tasks(vec![left_source])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
            .with_upstream_tasks(vec![right_source])
            .with_downstream_tasks(vec![build]),
        PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinBuildBeforeProbe)
            .with_upstream_tasks(vec![build])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::JoinMaterialization)
            .with_upstream_tasks(vec![probe])
            .with_downstream_tasks(vec![materialize]),
        PipelineBarrier::new(PipelineBarrierId::new(4), PipelineBarrierKind::SinkBoundary)
            .with_upstream_tasks(vec![sink_task]),
    ];

    let resources = vec![
        ResidentResource::new(
            left_frame,
            left_source,
            vec![probe],
            ResidentLifetime::BetweenTasks { producer: left_source, last_consumer: probe },
        ),
        ResidentResource::new(
            right_frame,
            right_source,
            vec![build],
            ResidentLifetime::BetweenTasks { producer: right_source, last_consumer: build },
        ),
        ResidentResource::new(
            join_output,
            materialize,
            vec![sink_task],
            ResidentLifetime::BetweenTasks { producer: materialize, last_consumer: sink_task },
        ),
    ];

    let gates = vec![ReservationGate::new(
        blocked_gate,
        ReservationGateOwner::Task(build),
        ReservationKind::HashTable,
        ReservationRequirementScope::Node,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        ContractStatus::BlockedByLowerLevel,
        ContractQuality::ExactPreflight,
    )];

    CompiledPipeline::new(
        tasks,
        edges,
        barriers,
        resources,
        gates,
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink_task),
    )
}

fn source_filter_projection_gpu_resident_plan(path: &Path, source_schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        Arc::clone(&source_schema),
        vec!["k".into(), "v".into()],
    )));
    let filter = builder.add_filter(
        source,
        Expr::binary(Expr::column("k"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(1))),
        Arc::clone(&source_schema),
    );
    let projection = builder.add_projection(
        filter,
        vec![ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int32, false))],
    );
    builder.finish_gpu_resident(projection)
}

fn write_source_unary_parquet(path: &Path) -> Arc<Schema> {
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_join_parquets(left_path: &Path, right_path: &Path) {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    write_parquet(left_path, Arc::clone(&left_schema), &left_batch);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![2, 4, 5])), Arc::new(Int32Array::from(vec![200, 400, 500]))],
    )
    .unwrap();
    write_parquet(right_path, Arc::clone(&right_schema), &right_batch);
}

fn write_single_join_post_root_parquets(customers_path: &Path, orders_path: &Path) {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let customer_batch =
        RecordBatch::try_new(Arc::clone(&customer_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    write_parquet(customers_path, Arc::clone(&customer_schema), &customer_batch);

    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let orders_batch = RecordBatch::try_new(
        Arc::clone(&orders_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30])), Arc::new(Int32Array::from(vec![1, 1, 2]))],
    )
    .unwrap();
    write_parquet(orders_path, Arc::clone(&orders_schema), &orders_batch);
}

fn write_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn int32_values(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let array = batch.column(column).as_any().downcast_ref::<Int32Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int64_values(batch: &RecordBatch, column: usize) -> Vec<Option<i64>> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn assert_task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    assert_eq!(
        pipeline.tasks().iter().filter(|task| task.kind() == kind).count(),
        expected,
        "unexpected count for {kind:?} tasks"
    );
}

fn bounded_context(runtime: Arc<cudf::Runtime>) -> ExecutionContext {
    ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap()
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
