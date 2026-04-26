use std::fs;

/// The wall-time benchmark's admission env vars and TSV metadata keys are
/// part of its public shell contract. Like the other wall-time benches, the
/// cargo binary is only a thin `run_main` wrapper; the contract lives in the
/// implementation module (and the shared clickbench workload helpers), same
/// as this test already pins for `report_workload/clickbench.rs`.
#[test]
fn native_clickbench_benchmark_admission_contract_is_stable() {
    let bin_src = fs::read_to_string("src/bin/native_clickbench_benchmark.rs")
        .expect("benchmark binary source must be readable from the package root");
    assert!(
        bin_src.contains("datafusion_nexus_bench::clickbench_benchmark::run_main"),
        "benchmark binary must stay wired to clickbench_benchmark::run_main",
    );

    let src = fs::read_to_string("src/clickbench_benchmark/mod.rs")
        .expect("benchmark module source must be readable from the package root");
    for required in [
        "clickbench_native_profile_settings_from_env",
        "clickbench_native_optimizer_config_with_profile_settings",
        "RelationalExecutionSelection::strict_native()",
        ".configure_native_optimizer(",
        "# clickbench_benchmark_relational_execution=",
        "# clickbench_benchmark_native_profile_name=",
        "# clickbench_benchmark_native_profile_source=",
        "# clickbench_benchmark_native_device_budget_bytes=",
        "# clickbench_benchmark_native_max_source_chunk_bytes=",
        "# clickbench_benchmark_native_max_row_groups_per_chunk=",
        "# clickbench_benchmark_native_source_chunk_planning_mode=",
        "# clickbench_benchmark_kvikio_compat_mode=",
    ] {
        assert!(src.contains(required), "benchmark module missing contract surface {required}");
    }
    assert!(!src.contains("FALLBACK_POLICY"), "benchmark must not parse a removed legacy environment value");

    let workload_src = fs::read_to_string("src/report_workload/clickbench.rs")
        .expect("workload source must be readable from the package root");
    for required in [
        "NEXUS_CLICKBENCH_NATIVE_PROFILE",
        "clickbench_gpu_throughput",
        "NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE",
    ] {
        assert!(workload_src.contains(required), "clickbench workload missing contract surface {required}");
    }
}

/// The execution report emits a normal native selection together with the
/// pipeline mode. Keep the selection explicit rather than deriving it from an
/// environment policy value.
#[test]
fn native_clickbench_execution_report_records_normal_native_selection() {
    let src = fs::read_to_string("src/bin/native_clickbench_execution_report.rs")
        .expect("binary source must be readable from the workspace root");
    for required in [
        "NEXUS_CLICKBENCH_EXECUTION_QUERY_IDS",
        "NEXUS_CLICKBENCH_EXECUTION_PROGRESS",
        "NEXUS_NATIVE_PIPELINE_EXECUTION_MODE",
        "RelationalExecutionSelection::native()",
    ] {
        assert!(src.contains(required), "binary source missing env var {required}");
    }
    assert!(src.contains(".configure_native_optimizer("), "binary must configure the normal native selection");
    assert!(!src.contains("FALLBACK_POLICY"), "binary must not expose a removed legacy environment value");
    assert!(
        src.contains("[native_clickbench_execution_report] stage=run_config"),
        "binary must emit a stable stderr run-config event"
    );
}

/// Pin that the binary uses the workload module's filtered execution entry
/// point (not the unfiltered one), so `NEXUS_CLICKBENCH_EXECUTION_QUERY_IDS`
/// remains functional. The progress event type must also stay wired in so
/// `NEXUS_CLICKBENCH_EXECUTION_PROGRESS` keeps writing to stderr.
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
