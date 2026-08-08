#[allow(dead_code)]
#[path = "../cmake.rs"]
mod cmake;

use std::path::Path;

#[test]
fn explicit_root_rejects_a_conflicting_default_package_selection() {
    let error = cmake::require_explicit_root_selection(
        Some(Path::new("/opt/cuvs-nexus")),
        Path::new("/usr/include"),
        Path::new("/usr/lib/libcuvs_c.so"),
        Path::new("/usr/lib/libcuvs.so"),
    )
    .expect_err("an explicit root must reject artifacts selected from a default prefix");

    let message = error.to_string();
    assert!(message.contains("CUVS_ROOT"));
    assert!(message.contains("outside that explicit root"));
}

#[test]
fn explicit_root_accepts_only_artifacts_below_the_selected_prefix() {
    let root = Path::new("/opt/cuvs-nexus");
    cmake::require_explicit_root_selection(
        Some(root),
        &root.join("include"),
        &root.join("lib/libcuvs_c.so"),
        &root.join("lib/libcuvs.so"),
    )
    .expect("all selected artifacts live below CUVS_ROOT");
}

#[test]
fn private_dependency_trees_are_rejected() {
    let error = cmake::reject_private_dependency_path(Path::new("/opt/cugraph/_deps/cuvs-src/include"))
        .expect_err("a private dependency tree must never supply cuVS");
    assert!(error.to_string().contains("private `_deps` tree"));
}

#[test]
fn installed_prefix_paths_are_accepted() {
    cmake::reject_private_dependency_path(Path::new("/opt/cuvs-nexus/include"))
        .expect("an installed prefix is a supported dependency source");
}
