use std::fs;

/// The wall-time benchmark binary's admission env vars and TSV metadata keys
/// are part of its public shell contract. Pin them by source so renames fail
/// here rather than disappearing from orchestration scripts and artifacts.
#[test]
fn native_clickbench_benchmark_admission_contract_is_stable() {
    let src = fs::read_to_string("src/bin/native_clickbench_benchmark.rs")
        .expect("binary source must be readable from the package root");
    for required in [
        "CUDF_CLICKBENCH_BENCH_NATIVE_FALLBACK_POLICY",
        "clickbench_native_profile_settings_from_env",
        "clickbench_native_optimizer_config_with_profile_settings",
        ".with_fallback_policy(config.native_fallback_policy)",
        "# clickbench_benchmark_native_fallback_policy=",
        "# clickbench_benchmark_native_profile_name=",
        "# clickbench_benchmark_native_profile_source=",
        "# clickbench_benchmark_native_device_budget_bytes=",
        "# clickbench_benchmark_native_max_source_chunk_bytes=",
        "# clickbench_benchmark_native_max_row_groups_per_chunk=",
        "# clickbench_benchmark_native_source_chunk_planning_mode=",
        "# clickbench_benchmark_kvikio_compat_mode=",
    ] {
        assert!(src.contains(required), "benchmark binary missing contract surface {required}");
    }

    let workload_src = fs::read_to_string("src/report_workload/clickbench.rs")
        .expect("workload source must be readable from the package root");
    for required in [
        "CUDF_CLICKBENCH_NATIVE_PROFILE",
        "clickbench_gpu_throughput",
        "CUDF_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE",
    ] {
        assert!(workload_src.contains(required), "clickbench workload missing contract surface {required}");
    }
}

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
        "CUDF_CLICKBENCH_NATIVE_FALLBACK_POLICY",
        "CUDF_NATIVE_PIPELINE_EXECUTION_MODE",
    ] {
        assert!(src.contains(required), "binary source missing env var {required}");
    }
    assert!(
        src.contains(".with_fallback_policy(fallback_policy)"),
        "binary must configure the optimizer from the logged fallback-policy value"
    );
    assert!(
        src.contains("[native_clickbench_execution_report] stage=run_config"),
        "binary must emit a stable stderr run-config event"
    );
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
