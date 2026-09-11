//! Dependency boundaries that Cargo enforces for the native engine.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

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
        .and_then(|packages| packages.iter().find(|package| package["name"] == "algeon-query-engine"))
        .expect("workspace metadata must include algeon-query-engine");
    let datafusion_dependencies = engine["dependencies"]
        .as_array()
        .expect("package metadata must list dependencies")
        .iter()
        .filter_map(|dependency| dependency["name"].as_str())
        .filter(|name| name.to_ascii_lowercase().contains("datafusion"))
        .collect::<Vec<_>>();

    assert!(
        datafusion_dependencies.is_empty(),
        "query-engine must remain DataFusion-free; found {datafusion_dependencies:?}"
    );
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("query-engine crate should remain two levels below the workspace root")
        .to_path_buf()
}
