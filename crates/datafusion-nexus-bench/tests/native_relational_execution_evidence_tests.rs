use std::{fs, path::Path};

fn source(path: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).unwrap_or_else(|| panic!("source missing section start {start:?}"));
    let tail = &source[start..];
    let end = tail.find(end).unwrap_or_else(|| panic!("source missing section end {end:?}"));
    &tail[..end]
}

#[test]
fn native_execution_reports_emit_normal_native_selection_and_pipeline_mode_to_stderr() {
    for (path, event_prefix) in [
        ("src/bin/native_tpch_execution_report.rs", "[native_tpch_execution_report] stage=run_config"),
        ("src/bin/native_tpcds_execution_report.rs", "[native_tpcds_execution_report] stage=run_config"),
        ("src/bin/native_clickbench_execution_report.rs", "[native_clickbench_execution_report] stage=run_config"),
    ] {
        let src = source(path);
        assert!(src.contains(event_prefix), "{path} must emit a stable stderr run-config event");
        assert!(src.contains("relational_execution={}"), "{path} must identify the selected relational execution");
        assert!(
            src.contains("native_pipeline_execution_mode={}"),
            "{path} run-config event must identify the selected native pipeline mode"
        );
        assert!(src.contains("RelationalExecutionSelection::native()"), "{path} must select normal native execution");
        assert!(src.contains(".configure_native_optimizer("), "{path} must install the selected native rule");
        assert!(!src.contains("FALLBACK_POLICY"), "{path} must not parse a removed legacy environment value");
    }
}

#[test]
fn iceberg_native_sessions_select_strict_native_execution_and_pipeline_mode_to_stderr() {
    let src = source("src/iceberg_smoke/config.rs");
    let module_src = source("src/iceberg_smoke/mod.rs");
    let session_setup =
        section(&src, "pub(crate) async fn native_session_context", "pub fn native_session_config_for_profile");
    let evidence_setup = section(&module_src, "fn native_session_execution_config_from_env", "mod config;");
    let profile_config = section(
        &src,
        "pub fn native_execution_config_for_profile_with_env",
        "pub(crate) fn native_statistical_aggregate_execution_mode_for_profile_with_env",
    );
    assert!(session_setup.contains("native_session_execution_config_from_env(native_execution_profile)?"));
    assert!(evidence_setup.contains("[native_iceberg_smoke_report] stage=native_session_config"));
    assert!(evidence_setup.contains("relational_execution={} native_pipeline_execution_mode={}"));
    assert!(session_setup.contains("RelationalExecutionSelection::strict_native()"));
    assert!(session_setup.contains(".configure_native_optimizer("));
    assert!(!module_src.contains("FALLBACK_POLICY"));
    assert!(evidence_setup.contains("execution_config.native_pipeline_execution_mode().as_str()"));
    assert!(
        profile_config.contains("with_native_pipeline_execution_mode(NativePipelineExecutionMode::StrictBounded)"),
        "Iceberg execution profiles must keep the strict-bounded pipeline contract"
    );
}

#[test]
fn tpcds_three_way_benchmark_records_the_relational_selection_and_pipeline_mode_without_changing_the_tsv_schema() {
    let runner = source("src/tpcds_three_way_bench/runner.rs");
    let config = source("src/tpcds_three_way_bench/runner/config.rs");
    let report = source("src/tpcds_three_way_bench/runner/report.rs");

    assert!(runner.contains("relational_execution={} native_pipeline_execution_mode={}"));
    assert!(runner.contains("cfg.native_pipeline_execution_mode.as_str()"));
    assert!(config.contains("tpcds_native_analytics_optimizer_config_with_profile_settings(&native_profile)"));
    assert!(config.contains(".native_pipeline_execution_mode()"));
    assert!(config.contains("Engine::NativeGpu => RelationalExecutionSelection::strict_native()"));
    assert!(config.contains("Engine::CpuDataFusion => RelationalExecutionSelection::CpuBaseline"));
    assert!(
        config.contains(".with_native_pipeline_execution_mode(cfg.native_pipeline_execution_mode)"),
        "native execution must use the same pipeline mode recorded by the run config"
    );
    assert!(
        config.contains(".with_native_pipeline_execution_mode(native_pipeline_execution_mode)"),
        "hybrid native planning must use the same pipeline mode recorded by the run config"
    );
    assert!(report.contains("# tpcds_benchmark_native_pipeline_execution_mode={}"));
    assert!(report.contains("# tpcds_benchmark_relational_execution={}"));
    assert!(report.contains("cfg.native_pipeline_execution_mode.as_str()"));
    let tsv_header = section(&report, "pub(super) const TSV_HEADER", "const TSV_COLUMNS");
    assert!(
        !tsv_header.contains("native_pipeline_execution_mode"),
        "pipeline evidence belongs in metadata, not the stable TSV row schema"
    );
}
