//! Dependency-direction guards for the refactored architecture.
//!
//! These tests describe responsibility boundaries rather than broad module
//! bans. Reverse-edge baselines are removed as their owner phase completes.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use super::public_surface::rust_identifier_count;

#[test]
fn gpu_parquet_caller_materialized_byte_calls_are_forbidden() {
    let markers = ["filebytes", concat!("from_host_", "buffer")];
    let actual = edge_occurrences(&["crates/nexus-query-engine/src", "src"], &markers);

    assert!(
        actual.is_empty(),
        "production GPU Parquet reads must use typed path/URI source builders; CPU metadata/footer object-store IO is intentionally outside this symbol policy: {actual:?}"
    );
}

#[test]
fn retired_source_report_and_runtime_symbols_are_forbidden() {
    let markers = [
        concat!("source_", "fetch_bytes"),
        concat!("source_", "fetch_count"),
        concat!("host_", "fetch_budget_wait_time"),
        concat!("source_", "fetch_concurrency_wait_time"),
        concat!("source_", "fetch_time"),
        concat!("source_", "decode_time"),
        concat!("host_", "buffer_"),
        concat!("source_decode_", "input"),
        concat!("source_fallback_", "reason"),
        concat!("parquet_read_fallback_", "reason"),
        concat!("Host", "Buffer", "Fetch", "RuntimeMetrics"),
    ];
    let actual = edge_occurrences(
        &["crates/nexus-query-engine/src", "src", "crates/nexus-bench/src", "crates/nexus-tools/src"],
        &markers,
    );
    assert!(actual.is_empty(), "retired foreground-fetch and decoder report/runtime symbols returned: {actual:?}");
}

#[test]
fn parquet_execution_shapes_use_typed_path_or_uri_source_builders() {
    let source_exec = read_repo_source("crates/nexus-query-engine/src/exec/source.rs");
    assert_eq!(
        source_exec.matches("cudf_local_parquet_source_file_selection(").count(),
        1,
        "whole-file reads must build one typed file-selection contract"
    );
    assert_eq!(
        source_exec.matches("cudf_local_parquet_source_chunk(").count(),
        1,
        "bounded single-chunk reads must build one typed chunk contract"
    );
    assert_eq!(
        source_exec.matches("cudf_local_parquet_source_chunk_batch(").count(),
        1,
        "multi-chunk batch reads must build one typed batch contract"
    );

    let delete_exec_source = read_repo_source("crates/nexus-query-engine/src/exec/source/position_delete.rs");
    let delete_exec =
        delete_exec_source.split("#[cfg(test)]").next().expect("position-delete production source precedes its tests");
    assert_eq!(
        delete_exec.matches("cudf_local_parquet_source_chunk(").count(),
        1,
        "unbounded position-delete-aware chunk reads must use the typed chunk builder"
    );
    assert_eq!(
        delete_exec.matches("cudf_local_parquet_source_file_selection(").count(),
        2,
        "position and equality delete files must each use the typed file-selection builder"
    );

    let shared_file_reader = read_repo_source("crates/nexus-query-engine/src/exec/source/read.rs");
    assert_eq!(
        shared_file_reader.matches("read_parquet_source_chunk_with_evidence(").count(),
        1,
        "whole-file and delete-file reads must share the path/URI chunk reader"
    );
}

#[test]
fn ir_models_remain_independent_of_executor_implementation() {
    assert_no_edges(
        "IR plan/expr/source must remain executor-independent",
        &[
            "crates/nexus-query-engine/src/plan.rs",
            "crates/nexus-query-engine/src/plan",
            "crates/nexus-query-engine/src/expr.rs",
            "crates/nexus-query-engine/src/expr",
            "crates/nexus-query-engine/src/source.rs",
            "crates/nexus-query-engine/src/source",
        ],
        &["identifier:exec", "identifier:runtime"],
    );
}

#[test]
fn capability_does_not_delegate_source_validation_to_executor() {
    assert_no_edges(
        "capability analysis must own validation rather than call executor validation",
        &["crates/nexus-query-engine/src/capability.rs", "crates/nexus-query-engine/src/capability"],
        &["identifier:exec"],
    );
}

#[test]
fn optimizer_does_not_ask_executor_admission_entrypoints() {
    assert_no_edges(
        "optimizer decisions must use analysis-owned admission contracts, not strict/adaptive executor entrypoints",
        &["crates/nexus-query-engine/src/optimizer.rs", "crates/nexus-query-engine/src/optimizer"],
        &["identifier:exec"],
    );
}

#[test]
fn runtime_does_not_own_executor_cache_concrete_types() {
    assert_no_production_edges(
        "runtime resource ownership must depend on runtime cache contracts rather than executor cache concrete types",
        &["crates/nexus-query-engine/src/runtime.rs", "crates/nexus-query-engine/src/runtime"],
        &["identifier:exec"],
    );
}

#[test]
fn adapter_core_does_not_depend_on_cugraph_sql() {
    assert_no_edges(
        "adapter core native_exec/planner must depend on a frontend-neutral graph SPI rather than cugraph_sql",
        &["src/native_exec.rs", "src/native_exec", "src/planner.rs", "src/planner"],
        &["identifier:cugraph_sql"],
    );
}

#[test]
fn lakehouse_footer_pruning_uses_the_native_rule_product_path() {
    let native_mod = read_repo_source("src/native/mod.rs");
    assert!(
        native_mod.contains("pub(crate) mod lakehouse_pruning;"),
        "the async resolver must stay internal to the adapter product path"
    );
    assert!(!native_mod.contains("pub mod lakehouse_pruning;"), "the retired direct adapter API must not reappear");

    let native_rule = read_repo_source("src/planner/native.rs");
    assert_eq!(
        native_rule.matches("with_lakehouse_footer_pruning(lakehouse_pruning::enabled_from_env())").count(),
        1,
        "the one native physical-optimizer path must be the sole plan-time pruning hook"
    );

    let bench_runner = read_repo_source("crates/nexus-bench/src/iceberg_smoke/runner.rs");
    assert!(!bench_runner.contains("prune_table_format_native_plan"), "bench must not retain the retired resolver API");
    assert!(
        !bench_runner.contains("datafusion_nexus::native::lakehouse_pruning"),
        "bench must execute pruning through the product optimizer path"
    );
}

#[test]
fn root_lakehouse_pruning_reaches_engine_through_one_optimizer_entry_point() {
    let source = read_repo_source("src/native/lakehouse_pruning.rs");

    assert_eq!(
        source.matches("optimize_lakehouse_footer_pruning(").count(),
        1,
        "the adapter must enter lakehouse pruning through one engine-owned sequencing entry point"
    );
    assert_no_edges(
        "the adapter must not independently discover or apply native optimizer sub-passes",
        &["src/native/lakehouse_pruning.rs"],
        &[
            "identifier:seed_dimension_derived_reader_predicates",
            "identifier:discover_dimension_derived_pruning_requests",
            "identifier:apply_local_parquet_row_group_selections",
            "identifier:apply_dimension_derived_pruning_with_stats",
            "identifier:local_parquet_source",
            "identifier:reader_projection_indices",
        ],
    );
}

#[test]
fn test_only_engine_surfaces_are_crate_private() {
    let exec = read_repo_source("crates/nexus-query-engine/src/exec/mod.rs");
    assert!(
        exec.contains("#[cfg(test)]\npub(crate) mod reservation_contract_testing"),
        "reservation contract probes must compile only inside the engine test crate"
    );
    assert!(
        exec.contains("#[cfg(test)]\npub(crate) use bounded::pipeline::{ResidentFrameStreamProducerTestHarness"),
        "bounded frame harnesses must not be exported from the production executor facade"
    );

    for path in [
        "crates/nexus-query-engine/src/exec/bounded/pipeline.rs",
        "crates/nexus-query-engine/src/exec/bounded/compiled.rs",
        "crates/nexus-query-engine/src/exec/bounded/compiled/tasks/mod.rs",
    ] {
        let source = read_repo_source(path);
        assert!(
            source.contains("#[cfg(test)]\npub(crate) use"),
            "{path} must keep bounded frame test harness re-exports crate-private"
        );
    }

    let reservation = read_repo_source("crates/nexus-query-engine/src/runtime/reservation.rs");
    assert!(
        reservation.contains("pub(crate) struct RetryFacts"),
        "bounded retry facts are executor implementation details, not a runtime facade"
    );
    let runtime = read_repo_source("crates/nexus-query-engine/src/runtime.rs");
    assert!(runtime.contains("pub(crate) mod gpu;"), "runtime::gpu must remain a private implementation module");
    assert!(
        runtime.contains("pub(crate) use reservation::RetryFacts;"),
        "bounded retry facts must not be re-exported through runtime"
    );
}

#[test]
fn attempt_facts_layering_has_no_error_metrics_runtime_observability_cycle() {
    assert_no_edges(
        "error correlation construction must use the lower correlation leaf rather than metrics",
        &["crates/nexus-query-engine/src/error.rs"],
        &["crate::metrics"],
    );
    assert_no_edges(
        "metrics projection must use lower execution contracts rather than runtime ownership",
        &["crates/nexus-query-engine/src/metrics"],
        &["crate::runtime"],
    );
    assert_no_edges(
        "observability stage helpers must not depend on the concrete runtime execution context",
        &["crates/nexus-query-engine/src/observability"],
        &["crate::runtime"],
    );

    let runtime = read_repo_source("crates/nexus-query-engine/src/runtime.rs");
    assert!(runtime.contains("pub use policy::{"), "runtime must preserve public policy re-export paths");
    assert!(runtime.contains("pub use reservation::MemoryReservationRequest;"));
    let metrics = read_repo_source("crates/nexus-query-engine/src/metrics/mod.rs");
    assert!(
        metrics.contains("pub use crate::{\n    correlation::{CompiledTaskId, PlanNodeId, QueryId, TraceId},"),
        "metrics must preserve the lower correlation-id re-exports"
    );
}

#[test]
fn accounted_service_cache_and_single_observation_store_boundaries_are_preserved() {
    let runtime = read_repo_source("crates/nexus-query-engine/src/runtime.rs");
    assert!(
        runtime.contains("pub(crate) mod cache;") && runtime.contains("pub(crate) mod source_identity;"),
        "cache ownership and source identity must remain private runtime substrate"
    );
    assert_no_edges(
        "the retired root metrics-capture observation store must not be reintroduced beside cache accounting",
        &["src", "crates/nexus-query-engine/src"],
        &["NativeMetricsCaptureSink", "runtime_observations", "record_runtime_observation"],
    );
}

#[test]
fn native_engine_manifest_has_no_datafusion_dependency() {
    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(repo_root())
        .output()
        .expect("cargo metadata must be runnable for the dependency boundary check");
    assert!(output.status.success(), "cargo metadata failed: {}", String::from_utf8_lossy(&output.stderr));

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata must return valid JSON");
    let engine = metadata["packages"]
        .as_array()
        .and_then(|packages| packages.iter().find(|package| package["name"] == "nexus-query-engine"))
        .expect("workspace metadata must include nexus-query-engine");
    let datafusion_dependencies = engine["dependencies"]
        .as_array()
        .expect("package metadata must list dependencies")
        .iter()
        .filter_map(|dependency| dependency["name"].as_str())
        .filter(|name| name.to_ascii_lowercase().contains("datafusion"))
        .collect::<Vec<_>>();

    assert!(
        datafusion_dependencies.is_empty(),
        "nexus-query-engine must remain DataFusion-free; found {datafusion_dependencies:?}"
    );
}

fn assert_no_edges(rule: &str, roots: &[&str], markers: &[&str]) {
    assert!(!markers.is_empty(), "{rule}: a zero-edge rule must name the forbidden dependency markers");
    let actual = edge_occurrences(roots, markers);
    assert!(actual.is_empty(), "{rule}: reverse dependencies are forbidden; found {actual:?}");
}

fn assert_no_production_edges(rule: &str, roots: &[&str], markers: &[&str]) {
    assert!(!markers.is_empty(), "{rule}: a zero-edge rule must name the forbidden dependency markers");
    let actual = edge_occurrences(roots, markers)
        .into_iter()
        .filter(|((path, _), _)| !is_test_only_source_path(Path::new(path)))
        .collect::<BTreeMap<_, _>>();
    assert!(actual.is_empty(), "{rule}: reverse dependencies are forbidden; found {actual:?}");
}

fn is_test_only_source_path(path: &Path) -> bool {
    path.components().any(|component| {
        let component = component.as_os_str().to_string_lossy();
        let stem = component.strip_suffix(".rs").unwrap_or(&component);
        stem == "tests" || stem.ends_with("_tests")
    })
}

fn edge_occurrences(roots: &[&str], markers: &[&str]) -> BTreeMap<(String, String), usize> {
    let mut occurrences = BTreeMap::new();
    for root in roots {
        for file in source_files(root) {
            let relative = display_repo_path(&file);
            let source =
                fs::read_to_string(&file).unwrap_or_else(|error| panic!("failed to read {}: {error}", file.display()));
            for marker in markers {
                let count = marker.strip_prefix("identifier:").map_or_else(
                    || source.matches(marker).count(),
                    |identifier| rust_identifier_count(&source, identifier),
                );
                if count > 0 {
                    occurrences.insert((relative.clone(), (*marker).to_owned()), count);
                }
            }
        }
    }
    occurrences
}

fn source_files(relative: &str) -> Vec<PathBuf> {
    let path = repo_root().join(relative);
    assert!(path.exists(), "{} must exist for dependency-direction checks", path.display());
    let mut files = Vec::new();
    if path.is_dir() {
        collect_rs_files(&path, &mut files);
    } else {
        files.push(path);
    }
    files
}

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display())) {
        let path = entry.expect("source directory entry should be readable").path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("nexus-query-engine crate should remain two levels below the workspace root")
        .to_path_buf()
}

fn display_repo_path(path: &Path) -> String {
    path.strip_prefix(repo_root()).unwrap_or(path).display().to_string()
}

fn read_repo_source(relative: &str) -> String {
    let path = repo_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}
