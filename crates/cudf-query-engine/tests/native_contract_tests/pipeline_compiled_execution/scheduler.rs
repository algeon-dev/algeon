use super::*;

#[test]
fn compiled_scheduler_runs_tasks_from_dependencies_and_releases_after_all_consumers() {
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
fn compiled_scheduler_prioritizes_independent_join_build_branch_before_probe_branch() {
    let pipeline = downstream_build_branch_after_completed_upstream_build_pipeline();
    let scheduler = CompiledPipelineScheduler::new_with_prioritized_join_builds(
        &pipeline,
        BTreeSet::from([PipelineTaskId::new(5)]),
    )
    .expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new();

    let state = scheduler.run(&mut executor).expect("mock pipeline should execute");

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(3),
            PipelineTaskId::new(5),
            PipelineTaskId::new(2),
            PipelineTaskId::new(4),
            PipelineTaskId::new(6),
            PipelineTaskId::new(7),
            PipelineTaskId::new(8),
            PipelineTaskId::new(9),
        ]
    );
    assert_eq!(executor.executed, state.execution_order());
    assert_eq!(
        executor.released,
        vec![
            ResidentResourceId::new(0),
            ResidentResourceId::new(2),
            ResidentResourceId::new(1),
            ResidentResourceId::new(3),
            ResidentResourceId::new(4),
        ]
    );
    assert_eq!(state.live_resource_count(), 0);
}

#[test]
fn compiled_scheduler_runs_shared_resident_fanout_and_releases_after_all_consumers() {
    let pipeline = shared_resident_fanout_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("shared fan-out pipeline should be schedulable");
    let mut executor = MockExecutor::new();

    let state = scheduler.run(&mut executor).expect("shared fan-out pipeline should execute");

    assert_eq!(state.execution_order(), &[PipelineTaskId::new(0), PipelineTaskId::new(1), PipelineTaskId::new(2)]);
    assert_eq!(executor.executed, state.execution_order());
    assert_eq!(executor.released, vec![ResidentResourceId::new(0)]);
    assert_eq!(state.live_resource_count(), 0);
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
}

#[test]
fn compiled_state_keeps_shared_resource_live_until_every_consumer_completes() {
    let pipeline = shared_resident_fanout_pipeline();
    let mut state = CompiledPipelineState::<String>::new(&pipeline).expect("state should initialize");

    let released = state
        .complete_task(&pipeline.tasks()[0], vec![(ResidentResourceId::new(0), "frame".to_string())])
        .expect("source should produce shared resource");
    assert!(released.is_empty());
    assert!(state.is_resource_live(ResidentResourceId::new(0)));

    let released =
        state.complete_task(&pipeline.tasks()[1], vec![]).expect("first consumer should not release shared resource");
    assert!(released.is_empty());
    assert!(state.is_resource_live(ResidentResourceId::new(0)));

    let released = state
        .complete_task(&pipeline.tasks()[2], vec![])
        .expect("last completed consumer should release shared resource");
    assert_eq!(released.iter().map(|resource| resource.id()).collect::<Vec<_>>(), vec![ResidentResourceId::new(0)]);
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
}

#[test]
fn compiled_state_consumes_single_consumer_resource_before_task_completion() {
    let pipeline = simple_unary_pipeline();
    let mut state = CompiledPipelineState::<String>::new(&pipeline).expect("state should initialize");

    state
        .complete_task(&pipeline.tasks()[0], vec![(ResidentResourceId::new(0), "frame".to_string())])
        .expect("source should produce single-consumer resource");
    assert!(state.is_resource_live(ResidentResourceId::new(0)));

    let consumed = state
        .consume_task_input_resource(&pipeline.tasks()[1], ResidentResourceId::new(0))
        .expect("single-consumer input can be consumed");
    assert_eq!(consumed.id(), ResidentResourceId::new(0));
    assert_eq!(consumed.into_payload(), "frame");
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
    assert_eq!(state.live_resource_count(), 0);

    let released =
        state.complete_task(&pipeline.tasks()[1], vec![]).expect("completion should not release consumed input twice");
    assert!(released.is_empty());
}

#[test]
fn compiled_state_rejects_consuming_multi_consumer_resource() {
    let pipeline = shared_resident_fanout_pipeline();
    let mut state = CompiledPipelineState::<String>::new(&pipeline).expect("state should initialize");

    state
        .complete_task(&pipeline.tasks()[0], vec![(ResidentResourceId::new(0), "frame".to_string())])
        .expect("source should produce shared resource");

    let error = state
        .consume_task_input_resource(&pipeline.tasks()[1], ResidentResourceId::new(0))
        .expect_err("multi-consumer input must not be consumed early");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t1"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
    assert_eq!(error.facts().get("consumer_count"), Some("2"));
    assert!(state.is_resource_live(ResidentResourceId::new(0)));
}

#[test]
fn compiled_scheduler_allows_executor_to_consume_single_consumer_input() {
    let pipeline = simple_unary_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new().consume_inputs_for(PipelineTaskId::new(1));

    let state = scheduler.run(&mut executor).expect("mock pipeline should execute with consumed input");

    assert_eq!(executor.consumed, vec![ResidentResourceId::new(0)]);
    assert!(executor.released.is_empty());
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
    assert_eq!(state.live_resource_count(), 0);
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
fn compiled_scheduler_releases_live_resources_when_executor_fails() {
    let pipeline = simple_unary_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new().fail_task_for(PipelineTaskId::new(1));

    let error = scheduler.run(&mut executor).expect_err("downstream executor failure should fail the pipeline");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::Internal);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t1"));
    assert_eq!(executor.executed, vec![PipelineTaskId::new(0), PipelineTaskId::new(1)]);
    assert_eq!(executor.released, vec![ResidentResourceId::new(0)]);
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
    assert_eq!(executor.released, vec![ResidentResourceId::new(0)]);
}
