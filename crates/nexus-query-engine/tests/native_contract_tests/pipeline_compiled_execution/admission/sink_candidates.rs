#[test]
fn compiled_bounded_support_rejects_multiple_explicit_sink_candidates() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d36-multiple-sink-left.parquet"),
        Path::new("/tmp/compiled-d36-multiple-sink-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join TopK plan should compile");
    let malformed = pipeline_with_extra_sink_candidate(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_rejects_post_root_data_fanout() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d36-post-root-fanout-left.parquet"),
        Path::new("/tmp/compiled-d36-post-root-fanout-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join TopK plan should compile");
    let malformed = pipeline_with_post_root_fanout(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}
