mod build_support;

use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn emit_runtime_search_path(path: &Path) {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}

fn link_search_path(directive: &str) -> Option<&Path> {
    directive.strip_prefix("cargo:rustc-link-search=native=").map(Path::new)
}

/// Recursively emit `cargo:rerun-if-changed` for every file under `dir`.
fn rerun_if_changed_recursive(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display())) {
        let entry = entry.unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            rerun_if_changed_recursive(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    let target_os =
        env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|error| panic!("failed to read CARGO_CFG_TARGET_OS: {error}"));
    build_support::ensure_supported_target_os(&target_os).unwrap_or_else(|error| panic!("{error}"));

    let native_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("native");
    let mut config = cmake::Config::new(&native_dir);

    if let Some(cuda_static_runtime) =
        build_support::normalized_cmake_bool_from_env("CUDA_STATIC_RUNTIME").unwrap_or_else(|error| panic!("{error}"))
    {
        config.define("CUDA_STATIC_RUNTIME", cuda_static_runtime);
    }

    // Build the C++ shim library using CMake, then replay the link directives it emitted.
    let dst = config.build();
    let directives = build_support::read_cargo_link_directives(&dst).unwrap_or_else(|error| panic!("{error}"));
    for directive in &directives {
        println!("{directive}");
    }
    for path in directives.iter().filter_map(|directive| link_search_path(directive)) {
        emit_runtime_search_path(path);
        println!("cargo:metadata=runtime-lib-dir={}", path.display());
    }

    // Re-run if native sources or build inputs change.
    rerun_if_changed_recursive(&native_dir.join("src"));
    rerun_if_changed_recursive(&native_dir.join("include"));
    println!("cargo:rerun-if-changed={}", native_dir.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=native/cmake/cargo-link.txt.in");
    println!("cargo:rerun-if-env-changed=CUDF_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUDF_CPP_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_STATIC_RUNTIME");
}
