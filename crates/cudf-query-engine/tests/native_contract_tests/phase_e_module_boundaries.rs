use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const STRICT_BOUNDED_MODULES: &[&str] = &["strict"];
const ADAPTIVE_BOUNDED_MODULES: &[&str] = &["adaptive"];
const SHARED_BOUNDED_RUNTIME_MODULES: &[&str] =
    &["aggregate", "chunks", "compiled", "facts", "final_sort", "join", "reservation"];

const RETIRED_TOPOLOGY_EXECUTOR_BRIDGES: &[&str] = &[
    "crates/cudf-query-engine/src/exec/bounded/pipeline.rs",
    "crates/cudf-query-engine/src/exec/bounded/pipeline",
    "crates/cudf-query-engine/src/exec/bounded/multi_join.rs",
    "crates/cudf-query-engine/src/exec/bounded/join/filtered.rs",
    "crates/cudf-query-engine/src/exec/bounded/join/keyless.rs",
    "crates/cudf-query-engine/src/exec/bounded/join/mixed.rs",
    "crates/cudf-query-engine/src/exec/bounded/join/projected_cross.rs",
];

const RETIRED_DISPATCH_ENTRYPOINTS: &[&str] = &[
    "execute_bounded_root_join_query_plan",
    "execute_bounded_join_pipeline_query_plan",
    "execute_bounded_multi_join_pipeline_query_plan",
    "execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan",
    "execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan",
    "execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan",
    "execute_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_query_plan",
    "execute_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_query_plan",
    "execute_bounded_keyed_hash_join_resident_subplan_pipeline_query_plan",
    "execute_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_query_plan",
];

const STRICT_SOURCE_ROOTS: &[&str] = &[
    "crates/cudf-query-engine/src/contract_report.rs",
    "crates/cudf-query-engine/src/exec/bounded/strict.rs",
    "crates/cudf-query-engine/src/exec/bounded/join/compiled.rs",
    "src/planner/native/strict_bounded.rs",
];

const SHARED_MODE_AGNOSTIC_ROOTS: &[&str] =
    &["crates/cudf-query-engine/src/capability", "crates/cudf-query-engine/src/pipeline"];

const ADAPTIVE_IMPORT_NEEDLES: &[&str] = &[
    "adaptive_report",
    "adaptive_runtime",
    "AdaptiveExecutionMode",
    "AdaptiveMemoryHistory",
    "AdaptivePipeline",
    "NativeAdaptivePipeline",
    "adaptive_bounded_pipeline_admission",
    "adaptive_bounded_pipeline_admission_with_memory_history",
    "try_execute_adaptive_bounded_pipeline",
    "with_adaptive_memory_history",
    "bounded::adaptive",
    "super::adaptive",
    "crate::exec::bounded::adaptive",
];

const ADAPTIVE_RUNTIME_TERM_NEEDLES: &[(&str, &[&str])] = &[
    ("memory history", &["memory history", "memory-history", "memory_history"]),
    ("learned estimate", &["learned estimate", "learned-estimate", "learned_estimate"]),
    ("spill", &["spill", "spilled", "spill_to"]),
    ("downgrade", &["downgrade", "downgraded"]),
    ("partial admission", &["partial admission", "partial-admission", "partial_admission"]),
];

#[test]
fn phase_e_bounded_direct_modules_have_explicit_ownership() {
    let bounded_dir = repo_root().join("crates/cudf-query-engine/src/exec/bounded");
    let discovered = direct_module_names(&bounded_dir);
    let ownership = bounded_module_ownership();
    let owned = ownership.keys().cloned().collect::<BTreeSet<_>>();

    assert_eq!(
        discovered, owned,
        "every direct bounded execution module must be explicitly classified as strict, adaptive, or shared runtime"
    );
}

#[test]
fn phase_e8_topology_executor_bridges_are_retired_from_the_runtime_tree() {
    let root = repo_root();
    for relative in RETIRED_TOPOLOGY_EXECUTOR_BRIDGES {
        let path = root.join(relative);
        assert!(
            !path.exists(),
            "{} must remain retired; bounded join dispatch belongs to strict/adaptive facades over compiled pipelines",
            relative
        );
    }
}

#[test]
fn phase_e8_retired_whole_query_dispatch_entrypoints_do_not_reappear() {
    let bounded_dir = repo_root().join("crates/cudf-query-engine/src/exec/bounded");
    let mut source = String::new();
    for file in source_files(&["crates/cudf-query-engine/src/exec/bounded"]) {
        source.push_str(&read_file(&file));
        source.push('\n');
    }

    assert!(bounded_dir.exists(), "{} must exist for Phase E.8 dispatch retirement checks", bounded_dir.display());
    for entrypoint in RETIRED_DISPATCH_ENTRYPOINTS {
        assert!(
            !source.contains(entrypoint),
            "{entrypoint} must not reappear; whole-query topology dispatch is retired in favor of strict/adaptive admission over compiled pipelines"
        );
    }
}

#[test]
fn phase_e_strict_exports_use_strict_boundary_facade() {
    let exec_mod = read_repo_file("crates/cudf-query-engine/src/exec/mod.rs");

    assert!(
        exec_mod.contains("pub use bounded::strict::"),
        "strict compiled exports must pass through the strict boundary facade"
    );
    assert!(
        !exec_mod.contains("pub use bounded::compiled::"),
        "exec/mod.rs must not publicly re-export strict compiled items directly from bounded::compiled"
    );
}

#[test]
fn phase_e_strict_modules_do_not_import_adaptive_modules() {
    for file in source_files(STRICT_SOURCE_ROOTS) {
        let source = read_file(&file);
        for needle in ADAPTIVE_IMPORT_NEEDLES {
            assert!(
                !source.contains(needle),
                "{} must not import or name adaptive-only APIs: found {needle:?}",
                display_repo_path(&file)
            );
        }
    }
}

#[test]
fn phase_e_strict_and_shared_graph_modules_do_not_use_adaptive_runtime_contract_terms() {
    let mut checked_files = source_files(STRICT_SOURCE_ROOTS);
    checked_files.extend(source_files(SHARED_MODE_AGNOSTIC_ROOTS));
    checked_files.sort();
    checked_files.dedup();

    for file in checked_files {
        let source = read_file(&file);
        let lower = source.to_ascii_lowercase();

        for (label, needles) in ADAPTIVE_RUNTIME_TERM_NEEDLES {
            for needle in *needles {
                assert!(
                    !lower.contains(needle),
                    "{} must not use adaptive-only runtime contract term {label:?} via {needle:?}",
                    display_repo_path(&file)
                );
            }
        }

        for (line_number, line) in lower.lines().enumerate() {
            if line.contains("retry")
                && (line.contains("contract")
                    || line.contains("quality")
                    || line.contains("evidence")
                    || line.contains("satisf"))
            {
                panic!(
                    "{}:{} must not describe retry as strict contract satisfaction, quality, or evidence",
                    display_repo_path(&file),
                    line_number + 1
                );
            }
        }
    }
}

fn bounded_module_ownership() -> BTreeMap<String, &'static str> {
    let mut ownership = BTreeMap::new();
    for name in STRICT_BOUNDED_MODULES {
        assert_eq!(ownership.insert((*name).to_owned(), "strict"), None, "duplicate bounded ownership for {name}");
    }
    for name in ADAPTIVE_BOUNDED_MODULES {
        assert_eq!(ownership.insert((*name).to_owned(), "adaptive"), None, "duplicate bounded ownership for {name}");
    }
    for name in SHARED_BOUNDED_RUNTIME_MODULES {
        assert_eq!(
            ownership.insert((*name).to_owned(), "shared-runtime"),
            None,
            "duplicate bounded ownership for {name}"
        );
    }
    ownership
}

fn direct_module_names(dir: &Path) -> BTreeSet<String> {
    let mut modules = BTreeSet::new();
    for entry in fs::read_dir(dir).unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display())) {
        let path = entry.expect("bounded dir entry should be readable").path();
        if path.is_dir() {
            modules.insert(path_name(&path));
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            let stem = path.file_stem().and_then(|stem| stem.to_str()).expect("Rust file should have UTF-8 stem");
            if stem != "mod" {
                modules.insert(stem.to_owned());
            }
        }
    }
    modules
}

fn source_files(roots: &[&str]) -> Vec<PathBuf> {
    let root = repo_root();
    let mut files = Vec::new();
    for relative in roots {
        let path = root.join(relative);
        assert!(path.exists(), "{} must exist for Phase E boundary checks", path.display());
        if path.is_dir() {
            collect_rs_files(&path, &mut files);
        } else {
            files.push(path);
        }
    }
    files
}

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display())) {
        let path = entry.expect("source dir entry should be readable").path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn read_repo_file(relative: &str) -> String {
    read_file(&repo_root().join(relative))
}

fn read_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("cudf-query-engine crate should live two levels below repo root")
        .to_path_buf()
}

fn display_repo_path(path: &Path) -> String {
    path.strip_prefix(repo_root()).unwrap_or(path).display().to_string()
}

fn path_name(path: &Path) -> String {
    let name = path.file_name().and_then(|name| name.to_str()).expect("module path should have UTF-8 name");
    name.to_owned()
}
