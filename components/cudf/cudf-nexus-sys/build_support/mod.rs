//! Internal helpers shared by the `cudf-nexus-sys` build script and its
//! focused build-contract tests.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

pub(crate) mod cmake;
pub(crate) mod elf;
pub(crate) mod lint;
pub(crate) mod toolchain;

/// The complete explicit Cargo environment fingerprint for the native build.
///
/// `PATH` deliberately remains absent: tools are resolved once to an absolute
/// executable before CMake runs, and only explicit selection inputs invalidate
/// the native output.
pub(crate) const NATIVE_BUILD_ENV_VARS: &[&str] = &[
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
    toolchain::CLANG_FORMAT_EXE_ENV,
    toolchain::CLANG_TIDY_EXE_ENV,
    toolchain::CMAKE_EXE_ENV,
    "CUDF_RS_SKIP_STALENESS_CHECK",
    "CARGO_FEATURE_TEST_HOOKS",
];

fn default_cudf_native_dir(manifest_dir: &Path, name: &str) -> Result<PathBuf, String> {
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(|repo_root| repo_root.join("target/native").join(name))
        .ok_or_else(|| format!("Could not determine the repository root from '{}'.", manifest_dir.display()))
}

pub(crate) fn default_cudf_install_dir(manifest_dir: &Path) -> Result<PathBuf, String> {
    default_cudf_native_dir(manifest_dir, "cudf-install")
}

pub(crate) fn default_cudf_build_dir(manifest_dir: &Path) -> Result<PathBuf, String> {
    default_cudf_native_dir(manifest_dir, "cudf-build")
}

/// Recursively emit `cargo:rerun-if-changed` for every file below `dir`.
pub(crate) fn rerun_if_changed_recursive(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display())) {
        let entry = entry.unwrap_or_else(|error| panic!("failed to read entry in {}: {error}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            rerun_if_changed_recursive(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Emit the native rebuild contract after a successful CMake configure/build.
pub(crate) fn emit_build_fingerprints(manifest_dir: &Path, native_dir: &Path, clang_format_config: &Path) {
    rerun_if_changed_recursive(&native_dir.join("src"));
    rerun_if_changed_recursive(&native_dir.join("include"));
    rerun_if_changed_recursive(&manifest_dir.join("build_support"));

    for path in [
        native_dir.join("CMakeLists.txt"),
        native_dir.join("cmake/cargo-link.txt.in"),
        native_dir.join("cmake/cudf-discovery.cmake"),
        native_dir.join("cmake/cudf-rust.map"),
        native_dir.join(".clang-tidy"),
        clang_format_config.to_path_buf(),
        manifest_dir.join("bindings/wrapper.h"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let install_dir = env::var_os("CUDF_INSTALL_DIR").map(PathBuf::from);
    let build_dir = env::var_os("CUDF_CPP_BUILD_DIR").map(PathBuf::from);
    if let Some(cmake_cache) =
        cmake::active_cudf_cmake_cache(manifest_dir, install_dir.as_deref(), build_dir.as_deref())
    {
        println!("cargo:rerun-if-changed={}", cmake_cache.display());
    }

    for env_var in NATIVE_BUILD_ENV_VARS {
        println!("cargo:rerun-if-env-changed={env_var}");
    }
}
