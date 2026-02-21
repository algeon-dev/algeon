use std::fs;

/// The execution report binary's env vars are part of its public shell
/// contract — users wire them into orchestration scripts. Pin the names by
/// reading the binary source so accidental renames in `src/bin/...` fail
/// here loudly rather than silently breaking external runners.
#[test]
fn native_clickbench_execution_report_env_var_names_are_stable() {
    let src = fs::read_to_string("src/bin/native_clickbench_execution_report.rs")
        .expect("binary source must be readable from the workspace root");
    for required in [
        "CUDF_CLICKBENCH_EXECUTION_QUERY_IDS",
        "CUDF_CLICKBENCH_EXECUTION_PROGRESS",
        "CUDF_NATIVE_PIPELINE_EXECUTION_MODE",
    ] {
        assert!(src.contains(required), "binary source missing env var {required}");
    }
}

/// Pin that the binary uses the workload module's filtered execution entry
/// point (not the unfiltered one), so `CUDF_CLICKBENCH_EXECUTION_QUERY_IDS`
/// remains functional. The progress event type must also stay wired in so
/// `CUDF_CLICKBENCH_EXECUTION_PROGRESS` keeps writing to stderr.
#[test]
fn native_clickbench_execution_report_wires_filtered_entrypoint() {
    let src = fs::read_to_string("src/bin/native_clickbench_execution_report.rs").unwrap();
    assert!(
        src.contains("clickbench_datafusion_execution_workload_items_filtered"),
        "binary must call the filtered execution entrypoint"
    );
    assert!(
        src.contains("ClickbenchExecutionProgressEvent"),
        "binary must wire ClickbenchExecutionProgressEvent for stderr progress"
    );
    assert!(
        src.contains("prepare_clickbench_report_dataset"),
        "binary must resolve the fixture via prepare_clickbench_report_dataset"
    );
}
