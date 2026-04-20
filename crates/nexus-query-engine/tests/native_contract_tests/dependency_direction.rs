//! Baseline-aware dependency-direction guards for the Phase 0 architecture.
//!
//! These tests intentionally describe responsibility boundaries, not a broad
//! module-name ban. Each current reverse edge is recorded with the phase that
//! owns its removal, so the baseline preserves today's behavior while rejecting
//! any new reverse dependency.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use super::public_surface::rust_identifier_count;

#[derive(Debug, Clone, Copy)]
struct EdgeBaseline {
    source: &'static str,
    marker: &'static str,
    count: usize,
    owner_phase: &'static str,
}

#[test]
fn ir_models_do_not_consume_executor_implementation_beyond_phase1_baseline() {
    assert_edge_baseline(
        "IR plan/expr/source must remain executor-independent",
        &[
            "crates/nexus-query-engine/src/plan.rs",
            "crates/nexus-query-engine/src/plan",
            "crates/nexus-query-engine/src/expr.rs",
            "crates/nexus-query-engine/src/expr",
            "crates/nexus-query-engine/src/source.rs",
            "crates/nexus-query-engine/src/source",
        ],
        &[
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/plan/core.rs",
                marker: "identifier:exec",
                count: 2,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/plan/core.rs",
                marker: "strict_bounded_pipeline_rejection_projection",
                count: 2,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/plan.rs",
                marker: "identifier:runtime",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/plan.rs",
                marker: "ExecutionBoundaryPolicy",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/plan/core.rs",
                marker: "ExecutionBoundaryPolicy",
                count: 2,
                owner_phase: "Phase 1",
            },
        ],
    );
}

#[test]
fn capability_does_not_delegate_source_validation_to_executor_beyond_phase1_baseline() {
    assert_edge_baseline(
        "capability analysis must own validation rather than call executor validation",
        &["crates/nexus-query-engine/src/capability.rs", "crates/nexus-query-engine/src/capability"],
        &[
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/capability.rs",
                marker: "identifier:exec",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/capability.rs",
                marker: "validate_local_parquet_source",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/capability/node.rs",
                marker: "validate_local_parquet_source",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/capability/node.rs",
                marker: "identifier:exec",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/capability/node.rs",
                marker: "validate_literal_batch_source",
                count: 1,
                owner_phase: "Phase 1",
            },
        ],
    );
}

#[test]
fn optimizer_does_not_ask_executor_admission_entrypoints_beyond_phase1_baseline() {
    assert_edge_baseline(
        "optimizer decisions must use analysis-owned admission contracts, not strict/adaptive executor entrypoints",
        &["crates/nexus-query-engine/src/optimizer.rs", "crates/nexus-query-engine/src/optimizer"],
        &[
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/optimizer/semijoin_reduction.rs",
                marker: "identifier:exec",
                count: 2,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/optimizer/semijoin_reduction.rs",
                marker: "strict_bounded_pipeline_admission",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/optimizer/semijoin_reduction.rs",
                marker: "adaptive_bounded_pipeline_admission",
                count: 1,
                owner_phase: "Phase 1",
            },
        ],
    );
}

#[test]
fn runtime_does_not_own_executor_cache_concrete_types_beyond_phase1_baseline() {
    assert_edge_baseline(
        "runtime resource ownership must depend on runtime cache contracts rather than executor cache concrete types",
        &["crates/nexus-query-engine/src/runtime.rs", "crates/nexus-query-engine/src/runtime"],
        &[
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime.rs",
                marker: "identifier:exec",
                count: 18,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime.rs",
                marker: "exec::GpuHashJoinBuildCache",
                count: 5,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime.rs",
                marker: "exec::GpuLocalParquetSourceCache",
                count: 5,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime.rs",
                marker: "exec::GpuResidentGraphCache",
                count: 5,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime.rs",
                marker: "exec::GpuResidentGraphPreparationPin",
                count: 3,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime/query_service/mod.rs",
                marker: "identifier:exec",
                count: 9,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime/query_service/mod.rs",
                marker: "exec::GpuLocalParquetSourceCache",
                count: 1,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime/query_service/mod.rs",
                marker: "exec::GpuResidentGraphCache",
                count: 7,
                owner_phase: "Phase 1",
            },
            EdgeBaseline {
                source: "crates/nexus-query-engine/src/runtime/query_service/mod.rs",
                marker: "exec::GpuResidentGraphPreparationPin",
                count: 1,
                owner_phase: "Phase 1",
            },
        ],
    );
}

#[test]
fn adapter_core_does_not_depend_on_graph_sql_beyond_phase2_1_baseline() {
    assert_edge_baseline(
        "adapter core native_exec/planner must depend on a frontend-neutral graph SPI rather than graph_sql",
        &["src/native_exec.rs", "src/native_exec", "src/planner.rs", "src/planner"],
        &[
            EdgeBaseline {
                source: "src/native_exec/config.rs",
                marker: "identifier:graph_sql",
                count: 1,
                owner_phase: "Phase 2.1",
            },
            EdgeBaseline {
                source: "src/native_exec/producer.rs",
                marker: "identifier:graph_sql",
                count: 1,
                owner_phase: "Phase 2.1",
            },
            EdgeBaseline {
                source: "src/planner/native.rs",
                marker: "identifier:graph_sql",
                count: 2,
                owner_phase: "Phase 2.1",
            },
        ],
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

fn assert_edge_baseline(rule: &str, roots: &[&str], baseline: &[EdgeBaseline]) {
    assert!(
        !baseline.is_empty(),
        "{rule}: existing exceptions require an explicit owner phase rather than an unbounded allowlist"
    );
    for edge in baseline {
        assert!(
            matches!(edge.owner_phase, "Phase 1" | "Phase 2.1"),
            "{rule}: {} has an invalid owner phase {:?}",
            edge.source,
            edge.owner_phase
        );
    }

    let expected = baseline
        .iter()
        .map(|edge| ((edge.source.to_owned(), edge.marker.to_owned()), edge.count))
        .collect::<BTreeMap<_, _>>();
    let markers = baseline.iter().map(|edge| edge.marker).collect::<Vec<_>>();
    let actual = edge_occurrences(roots, &markers);

    assert_eq!(
        actual, expected,
        "{rule}: new reverse dependencies are forbidden. Existing entries are temporary baselines with an owner phase."
    );
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
