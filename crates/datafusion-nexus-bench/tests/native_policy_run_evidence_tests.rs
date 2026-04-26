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
fn native_execution_reports_emit_selected_policy_and_pipeline_mode_to_stderr() {
    for (path, event_prefix, fallback_env) in [
        (
            "src/bin/native_tpch_execution_report.rs",
            "[native_tpch_execution_report] stage=run_config",
            "CUDF_TPCH_NATIVE_FALLBACK_POLICY",
        ),
        (
            "src/bin/native_tpcds_execution_report.rs",
            "[native_tpcds_execution_report] stage=run_config",
            "CUDF_TPCDS_NATIVE_FALLBACK_POLICY",
        ),
        (
            "src/bin/native_clickbench_execution_report.rs",
            "[native_clickbench_execution_report] stage=run_config",
            "CUDF_CLICKBENCH_NATIVE_FALLBACK_POLICY",
        ),
    ] {
        let src = source(path);
        assert!(src.contains(fallback_env), "{path} must preserve fallback env {fallback_env}");
        assert!(src.contains(event_prefix), "{path} must emit a stable stderr run-config event");
        assert!(src.contains("fallback_policy={}"), "{path} run-config event must identify the selected policy");
        assert!(
            src.contains("native_pipeline_execution_mode={}"),
            "{path} run-config event must identify the selected native pipeline mode"
        );
        assert_eq!(
            src.matches("let fallback_policy = native_fallback_policy_from_env();").count(),
            1,
            "{path} must parse fallback policy once into the value shared by config and evidence"
        );
        assert!(
            src.contains(".with_fallback_policy(fallback_policy)"),
            "{path} must configure the optimizer from the logged fallback-policy value"
        );
        assert!(
            src.contains("gpu_fallback_policy_label(fallback_policy)"),
            "{path} must log the configured fallback-policy value with its stable label"
        );
    }
}

#[test]
fn iceberg_native_sessions_emit_selected_policy_and_strict_pipeline_mode_to_stderr() {
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
    assert!(module_src.contains("NEXUS_ICEBERG_NATIVE_FALLBACK_POLICY"));
    assert!(session_setup.contains("native_session_execution_config_from_env(native_execution_profile)?"));
    assert!(evidence_setup.contains("[native_iceberg_smoke_report] stage=native_session_config"));
    assert!(evidence_setup.contains("fallback_policy={} native_pipeline_execution_mode={}"));
    assert_eq!(
        module_src.matches("std::env::var(NATIVE_FALLBACK_POLICY_ENV)").count(),
        1,
        "Iceberg native session setup must read the fallback-policy env exactly once"
    );
    assert!(session_setup.contains("CudfOptimizerConfig::new(fallback_policy)"));
    assert!(evidence_setup.contains("gpu_fallback_policy_label(fallback_policy)"));
    assert!(evidence_setup.contains("execution_config.native_pipeline_execution_mode().as_str()"));
    assert!(
        profile_config.contains("with_native_pipeline_execution_mode(NativePipelineExecutionMode::StrictBounded)"),
        "Iceberg execution profiles must keep the strict-bounded pipeline contract"
    );
}

#[test]
fn tpcds_three_way_benchmark_records_the_configured_pipeline_mode_without_changing_the_tsv_schema() {
    let runner = source("src/tpcds_three_way_bench/runner.rs");
    let config = source("src/tpcds_three_way_bench/runner/config.rs");
    let report = source("src/tpcds_three_way_bench/runner/report.rs");

    assert!(runner.contains("native_fallback_policy={} native_pipeline_execution_mode={}"));
    assert!(runner.contains("cfg.native_pipeline_execution_mode.as_str()"));
    assert!(config.contains("tpcds_native_analytics_optimizer_config_with_profile_settings(&native_profile)"));
    assert!(config.contains(".native_pipeline_execution_mode()"));
    assert!(
        config.contains(".with_native_pipeline_execution_mode(cfg.native_pipeline_execution_mode)"),
        "native execution must use the same pipeline mode recorded by the run config"
    );
    assert!(
        config.contains(".with_native_pipeline_execution_mode(native_pipeline_execution_mode)"),
        "hybrid native planning must use the same pipeline mode recorded by the run config"
    );
    assert!(report.contains("# tpcds_benchmark_native_pipeline_execution_mode={}"));
    assert!(report.contains("cfg.native_pipeline_execution_mode.as_str()"));
    let tsv_header = section(&report, "pub(super) const TSV_HEADER", "const TSV_COLUMNS");
    assert!(
        !tsv_header.contains("native_pipeline_execution_mode"),
        "pipeline evidence belongs in metadata, not the stable TSV row schema"
    );
}
