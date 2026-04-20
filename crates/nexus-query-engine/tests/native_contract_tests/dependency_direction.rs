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
fn ir_models_do_not_consume_executor_implementation_beyond_phase1_baseline() {
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
fn capability_does_not_delegate_source_validation_to_executor_beyond_phase1_baseline() {
    assert_no_edges(
        "capability analysis must own validation rather than call executor validation",
        &["crates/nexus-query-engine/src/capability.rs", "crates/nexus-query-engine/src/capability"],
        &["identifier:exec"],
    );
}

#[test]
fn optimizer_does_not_ask_executor_admission_entrypoints_beyond_phase1_baseline() {
    assert_no_edges(
        "optimizer decisions must use analysis-owned admission contracts, not strict/adaptive executor entrypoints",
        &["crates/nexus-query-engine/src/optimizer.rs", "crates/nexus-query-engine/src/optimizer"],
        &["identifier:exec"],
    );
}

#[test]
fn runtime_does_not_own_executor_cache_concrete_types_beyond_phase1_baseline() {
    assert_no_edges(
        "runtime resource ownership must depend on runtime cache contracts rather than executor cache concrete types",
        &["crates/nexus-query-engine/src/runtime.rs", "crates/nexus-query-engine/src/runtime"],
        &["identifier:exec"],
    );
}

#[test]
fn adapter_core_does_not_depend_on_graph_sql() {
    assert_no_edges(
        "adapter core native_exec/planner must depend on a frontend-neutral graph SPI rather than graph_sql",
        &["src/native_exec.rs", "src/native_exec", "src/planner.rs", "src/planner"],
        &["identifier:graph_sql"],
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

    let bench_runner = read_repo_source("crates/datafusion-nexus-bench/src/iceberg_smoke/runner.rs");
    assert!(!bench_runner.contains("prune_table_format_native_plan"), "bench must not retain the retired resolver API");
    assert!(
        !bench_runner.contains("datafusion_nexus::native::lakehouse_pruning"),
        "bench must execute pruning through the product optimizer path"
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
