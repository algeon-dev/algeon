use arrow::array::Float64Array;

use super::*;
use crate::exec::{
    ExecutionMetrics,
    bounded::compiled::{resource::CompiledGpuResource, scheduler::CompiledTaskExecutor, state::CompiledPipelineState},
};

#[test]
fn compiled_final_aggregate_fanout_keeps_independent_consumers_and_releases_shared_state() {
    let runtime =
        shared_runtime(suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime"));
    let dir = unique_temp_dir("query-engine-final-aggregate-fanout");
    let (mut builder, aggregate, schema) = incremental_partitioned_merge_builder(&dir, 128, 131_072);
    let projected = builder
        .add_projection(
            aggregate,
            schema
                .fields()
                .iter()
                .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
                .collect(),
        )
        .unwrap();
    let filtered = builder
        .add_filter(
            projected,
            Expr::binary(Expr::column("count_value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
        )
        .unwrap();
    let left_sort =
        builder.add_sort(filtered, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
    let left = builder.add_limit(left_sort, LimitSpec::new(5, Some(10))).unwrap();
    let right_sort =
        builder.add_sort(projected, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
    let right = builder.add_limit(right_sort, LimitSpec::new(15, Some(10))).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Int64, false),
        Field::new("right_key", DataType::Int64, false),
        Field::new("left_sum", DataType::Int64, true),
        Field::new("right_sum", DataType::Int64, true),
        Field::new("left_avg", DataType::Float64, true),
        Field::new("right_avg", DataType::Float64, true),
    ]));
    let joined = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "count_value", "count_value"),
                    ColumnRef::input(1, "count_value", "count_value"),
                )],
            )
            .with_build_side(JoinSide::Right)
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "left_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "key", "key"), "right_key"),
                JoinOutputColumn::left(ColumnRef::input(0, "sum_value", "sum_value"), "left_sum"),
                JoinOutputColumn::right(ColumnRef::input(1, "sum_value", "sum_value"), "right_sum"),
                JoinOutputColumn::left(ColumnRef::input(0, "avg_value", "avg_value"), "left_avg"),
                JoinOutputColumn::right(ColumnRef::input(1, "avg_value", "avg_value"), "right_avg"),
            ]),
            output_schema,
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(joined).unwrap();
    let pipeline =
        compile_pipeline(&plan, &plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()))).unwrap();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).unwrap();
    let cap = 960 * 1024 * 1024;
    let gpu_inputs = GpuResidentInputMap::new();

    let context = capped_compiled_executor_context(Arc::clone(&runtime), &plan, cap);
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let (state, batch) = {
        let metrics_context = metrics_recorder.scope_context(&context).unwrap();
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).unwrap();
        (state, executor.take_host_output().unwrap())
    };
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(batch.num_rows(), 100);
    let left_avgs = batch.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
    let right_avgs = batch.column(5).as_any().downcast_ref::<Float64Array>().unwrap();
    let mut pairs = BTreeSet::new();
    for (row, (((left, right), left_sum), right_sum)) in int64_values(&batch, 0)
        .into_iter()
        .zip(int64_values(&batch, 1))
        .zip(int64_values(&batch, 2))
        .zip(int64_values(&batch, 3))
        .enumerate()
    {
        let (left, right) = (left.unwrap(), right.unwrap());
        assert_eq!(left_sum, Some(left % 7));
        assert_eq!(right_sum, Some(right % 7));
        assert_eq!(left_avgs.value(row), (left % 7) as f64);
        assert_eq!(right_avgs.value(row), (right % 7) as f64);
        pairs.insert((left, right));
    }
    let expected = (5..15).flat_map(|left| (15..25).map(move |right| (left, right))).collect();
    assert_eq!(pairs, expected);
    let usage = context.attempt_allocation_domain_usage().unwrap().unwrap();
    assert!(usage.peak_bytes <= cap, "{usage:?}");
    assert_eq!((usage.current_bytes, usage.current_allocations), (0, 0), "{usage:?}");
    drop(context);

    let context = capped_compiled_executor_context(Arc::clone(&runtime), &plan, cap);
    let cancellation_metrics_recorder = ExecutionMetrics::detached("cancellation_metrics");
    let (result, retained_at_cancel) = {
        let cancellation_metrics_context = cancellation_metrics_recorder.scope_context(&context).unwrap();
        let mut executor = CompiledGpuTaskExecutor::new(
            &cancellation_metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &cancellation_metrics_recorder,
        )
        .unwrap();
        let mut cancel_after_consumer =
            CancelAfterFinalAggregateConsumer { executor: &mut executor, context: &context, retained_at_cancel: None };
        let result = scheduler.run(&mut cancel_after_consumer);
        (result, cancel_after_consumer.retained_at_cancel)
    };
    let error = result.err().expect("cancellation stops before the remaining consumer");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    let retained = retained_at_cancel.expect("one consumer must finish before cancellation");
    assert!(retained >= 128 * 131_072 * 8, "the remaining consumer still owns all group keys: {retained}");
    let usage = context.attempt_allocation_domain_usage().unwrap().unwrap();
    assert_eq!((usage.current_bytes, usage.current_allocations), (0, 0), "{usage:?}");
    fs::remove_dir_all(dir).unwrap();
}

struct CancelAfterFinalAggregateConsumer<'a, 'execution> {
    executor: &'a mut CompiledGpuTaskExecutor<'execution>,
    context: &'a ExecutionContext,
    retained_at_cancel: Option<u64>,
}

impl CompiledTaskExecutor for CancelAfterFinalAggregateConsumer<'_, '_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<Vec<(ResidentResourceId, Self::Resource)>> {
        let output = self.executor.execute_task(task, state)?;
        if task.kind() == PipelineTaskKind::FinalSort && self.retained_at_cancel.is_none() {
            self.retained_at_cancel = Some(self.context.attempt_allocation_domain_usage()?.unwrap().current_bytes);
            self.context.cancellation().cancel();
        }
        Ok(output)
    }

    fn release_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        self.executor.release_resource(resource, payload)
    }

    fn cleanup_after_error(&mut self) -> Result<()> {
        self.executor.cleanup_after_error()
    }
}
