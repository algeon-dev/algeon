//! Static fingerprint-matrix contracts for cudf-sys native rebuilds.
//!
//! Live Cargo fingerprint reasons are validated by
//! `scripts/build/check_fingerprint_matrix.sh` when a configured RAPIDS install is
//! available. This suite locks the rebuild matrix that script enforces so the
//! contract cannot drift back to ambient PATH invalidation.

#[allow(dead_code)]
#[path = "../build_support/mod.rs"]
mod build_support;

use std::path::{Path, PathBuf};

const BUILD_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"));
const BUILD_SUPPORT_MOD: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_support/mod.rs"));
const BUILD_SUPPORT_CMAKE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_support/cmake.rs"));
const BUILD_SUPPORT_TOOLCHAIN: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_support/toolchain.rs"));

#[test]
fn path_is_not_a_native_fingerprint_input() {
    assert!(BUILD_RS.contains("emit_build_fingerprints"), "build.rs must delegate the canonical fingerprint contract");
    assert!(!build_support::NATIVE_BUILD_ENV_VARS.contains(&"PATH"), "PATH-only churn must not rebuild cudf-sys");
}

#[test]
fn explicit_toolchain_and_discovery_env_vars_are_tracked() {
    for env_var in [
        "CC",
        "CXX",
        "CUDACXX",
        "CUDAHOSTCXX",
        "CUDF_INSTALL_DIR",
        "CUDF_CPP_BUILD_DIR",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
        "CUDA_STATIC_RUNTIME",
        "CMAKE",
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_PREFIX_PATH",
        "CMAKE_LIBRARY_PATH",
        "CMAKE_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "CUDF_RUST_CLANG_LINT",
        build_support::toolchain::CLANG_FORMAT_EXE_ENV,
        build_support::toolchain::CLANG_TIDY_EXE_ENV,
        build_support::toolchain::CMAKE_EXE_ENV,
        "CUDF_RS_SKIP_STALENESS_CHECK",
        "CARGO_FEATURE_TEST_HOOKS",
    ] {
        assert!(
            build_support::NATIVE_BUILD_ENV_VARS.contains(&env_var),
            "expected explicit native fingerprint input `{env_var}"
        );
    }
    assert!(BUILD_SUPPORT_MOD.contains("for env_var in NATIVE_BUILD_ENV_VARS"));
}

#[test]
fn selected_runtime_libraries_remain_identity_tracked() {
    for expected in [
        "emit_runtime_dep_path_metadata(\"libcudf-runtime-lib\"",
        "\"rmm-runtime-lib\"",
        "\"librmm.so\"",
        "cargo:rerun-if-changed={}",
    ] {
        assert!(BUILD_SUPPORT_CMAKE.contains(expected), "expected runtime identity tracking snippet `{expected}`");
    }
}

#[test]
fn fingerprint_matrix_install_prefix_tracks_only_explicit_build_cache() {
    let manifest_dir = Path::new("/tmp/cudf-sys-manifest");
    let install = Path::new("/opt/cudf-install");
    let other_build = PathBuf::from("/tmp/other-cpp-build");

    assert!(
        build_support::cmake::active_cudf_cmake_cache(manifest_dir, Some(install), None).is_none(),
        "CUDF_INSTALL_DIR discovery must leave an unrelated source-tree CMakeCache untracked"
    );
    assert_eq!(
        build_support::cmake::active_cudf_cmake_cache(manifest_dir, Some(install), Some(&other_build)),
        Some(other_build.join("CMakeCache.txt")),
        "explicit build metadata must remain part of install-prefix discovery"
    );
}

#[test]
fn fingerprint_matrix_build_tree_tracks_active_cache_only() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let default_build = build_support::default_cudf_build_dir(manifest_dir).unwrap();
    let other_build = manifest_dir.join("non-default-build");

    assert!(
        build_support::cmake::active_cudf_cmake_cache(manifest_dir, None, None)
            == Some(build_support::cmake::default_cudf_cmake_cache(manifest_dir)),
        "no discovery override must track the canonical monorepo build cache"
    );
    assert!(
        build_support::cmake::active_cudf_cmake_cache(manifest_dir, None, Some(default_build.as_path()))
            == Some(default_build.join("CMakeCache.txt")),
        "explicit canonical build must stay tracked"
    );
    assert!(
        build_support::cmake::active_cudf_cmake_cache(manifest_dir, None, Some(other_build.as_path()))
            == Some(other_build.join("CMakeCache.txt")),
        "non-default explicit build must track only its own cache path"
    );
}

#[test]
fn explicit_cmake_pin_is_forwarded_to_cmake_rs() {
    assert!(
        BUILD_SUPPORT_TOOLCHAIN.contains("env::set_var(\"CMAKE\", cmake)"),
        "CUDF_RUST_CMAKE must select the executable that cmake-rs actually launches"
    );
}

#[test]
fn fingerprinted_env_list_helper_matches_build_script_contract() {
    let tracked = build_support::NATIVE_BUILD_ENV_VARS;
    assert!(tracked.contains(&"CC"));
    assert!(tracked.contains(&"CXX"));
    assert!(tracked.contains(&"CUDF_INSTALL_DIR"));
    assert!(!tracked.contains(&"PATH"));
    assert!(tracked.contains(&"CUDF_RS_SKIP_STALENESS_CHECK"));
    assert!(tracked.contains(&"CARGO_FEATURE_TEST_HOOKS"));
    assert!(BUILD_SUPPORT_MOD.contains("for env_var in NATIVE_BUILD_ENV_VARS"));
}
