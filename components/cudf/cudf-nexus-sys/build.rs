mod build_support;

use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    // Validate the Linux-only contract and resolve absolute compiler paths
    // before consulting the CMake cache, so cache invalidation and CMake see
    // the same toolchain identity.
    let (cc, cxx) = build_support::toolchain::prepare_cmake_toolchain();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    if env::var_os("CUDF_INSTALL_DIR").is_none() && env::var_os("CUDF_CPP_BUILD_DIR").is_none() {
        let default_install_dir =
            build_support::default_cudf_install_dir(&manifest_dir).unwrap_or_else(|error| panic!("{error}"));
        let default_build_dir =
            build_support::default_cudf_build_dir(&manifest_dir).unwrap_or_else(|error| panic!("{error}"));
        // SAFETY: build scripts are single-threaded and the native CMake
        // configuration reads this selector after it is set.
        unsafe {
            env::set_var("CUDF_INSTALL_DIR", default_install_dir);
            env::set_var("CUDF_CPP_BUILD_DIR", default_build_dir);
        }
    }
    let native_dir = build_support::lint::native_dir(&manifest_dir);
    let test_hooks = if env::var_os("CARGO_FEATURE_TEST_HOOKS").is_some() { "ON" } else { "OFF" };

    // A CMake cache may otherwise retain a different compiler, native prefix,
    // or test-hook configuration across Cargo rebuilds.
    build_support::cmake::discard_stale_cmake_build_dir(&cc, &cxx, test_hooks);
    build_support::cmake::check_libcudf_staleness(&manifest_dir);

    let mut config = cmake::Config::new(&native_dir);
    configure_common(&mut config, &manifest_dir);
    config.define("CUDF_RUST_ENABLE_TEST_HOOKS", test_hooks);

    // Build the shim, replay CMake's link manifest, and expose the resolved
    // runtime identity to downstream RAPIDS consumers.
    let install_dir = config.build();
    let runtime_shim = emit_shim_metadata(&install_dir, &manifest_dir, &native_dir);

    // The `native_abi_contract` integration test validates the production
    // (test-hooks OFF) shim of the current Cargo unit: without `test-hooks`
    // that is the primary build, with `test-hooks` a dedicated sibling
    // production prefix. Bind the contract test to this exact artifact at
    // compile time.
    let production_shim = if test_hooks == "OFF" {
        runtime_shim.clone()
    } else {
        build_production_shim(&install_dir, &native_dir, &manifest_dir, &cc, &cxx)
    };

    // Downstream identity contract: the exported root, the runtime library
    // metadata, and the active linked shim must all name the primary prefix.
    build_support::cmake::verify_primary_shim_identity(&install_dir, &runtime_shim)
        .unwrap_or_else(|error| panic!("{error}"));

    println!("cargo:cudf-rust-production-shim={}", production_shim.display());
    println!("cargo:rerun-if-changed={}", production_shim.display());
    println!("cargo:rustc-env=CUDF_RUST_PRODUCTION_SHIM={}", production_shim.display());
}

/// Build the sibling production (test-hooks OFF) shim, then restore the
/// exported package root to the primary install prefix.
///
/// `cmake::Config::build` emits `cargo:root` for the prefix it configured,
/// and Cargo keeps the last value for a repeated key. Keeping the auxiliary
/// build and the root re-assertion in one function guarantees the auxiliary
/// prefix never becomes `DEP_CUDF_RUST_ROOT` while the replayed link search,
/// rpath, and `cudf-rust-runtime-lib` metadata keep naming the primary prefix.
fn build_production_shim(install_dir: &Path, native_dir: &Path, manifest_dir: &Path, cc: &Path, cxx: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let production_out = out_dir.join("production-out");
    build_support::cmake::discard_stale_cmake_build_dir_at(&production_out, cc, cxx, "OFF");
    let mut production_config = cmake::Config::new(native_dir);
    production_config.out_dir(&production_out);
    configure_common(&mut production_config, manifest_dir);
    production_config.define("CUDF_RUST_ENABLE_TEST_HOOKS", "OFF");
    let production_shim = production_config.build().join("lib/libcudf_rust.so");
    build_support::cmake::reassert_primary_root(install_dir);
    production_shim
}

/// Shared CMake configuration for the primary and any sibling production
/// shim build: opt-in linting and the CUDA static runtime selection.
fn configure_common(config: &mut cmake::Config, manifest_dir: &Path) {
    let clang_lint = build_support::cmake::normalized_cmake_bool_from_env("CUDF_RUST_CLANG_LINT")
        .unwrap_or_else(|error| panic!("{error}"));
    config.define("CUDF_RUST_ENABLE_CLANG_TIDY", clang_lint.unwrap_or("OFF"));
    if clang_lint == Some("ON") {
        let clang_format = build_support::toolchain::resolve_tool_from_env_or_name(
            build_support::toolchain::CLANG_FORMAT_EXE_ENV,
            "clang-format",
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let clang_tidy = build_support::toolchain::resolve_tool_from_env_or_name(
            build_support::toolchain::CLANG_TIDY_EXE_ENV,
            "clang-tidy",
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let clang_format_config = build_support::lint::clang_format_config_path(manifest_dir);
        let clang_tidy_config = build_support::lint::clang_tidy_config_path(manifest_dir);
        build_support::lint::ensure_file_exists(&clang_tidy_config, "clang-tidy config");
        let lint_files = build_support::lint::collect_native_clang_lint_files(manifest_dir)
            .unwrap_or_else(|error| panic!("{error}"));
        build_support::lint::run_clang_format_check(&clang_format, &clang_format_config, &lint_files);
        config.define("CUDF_RUST_CLANG_TIDY_EXE", &clang_tidy);
        config.define("CUDF_RUST_CLANG_TIDY_CONFIG_FILE", &clang_tidy_config);
    }
    if let Some(cuda_static_runtime) = build_support::cmake::normalized_cmake_bool_from_env("CUDA_STATIC_RUNTIME")
        .unwrap_or_else(|error| panic!("{error}"))
    {
        config.define("CUDA_STATIC_RUNTIME", cuda_static_runtime);
    }
}

/// Replay the CMake link manifest for the shim install prefix and emit the
/// runtime identity and native fingerprint metadata downstream consumers
/// need. Returns the runtime shim path tracked as `cudf-rust-runtime-lib`.
fn emit_shim_metadata(install_dir: &Path, manifest_dir: &Path, native_dir: &Path) -> PathBuf {
    let directives =
        build_support::cmake::read_cargo_link_directives(install_dir).unwrap_or_else(|error| panic!("{error}"));
    build_support::cmake::replay_cargo_link_directives(&directives);
    // The freshly built libcudf_rust shim lives under the CMake install
    // prefix and must be tracked as `cudf-rust-runtime-lib` so downstream
    // `scripts/build/rapids.rs` can bake the exact rpath. This is emitted
    // here (rather than inside the discovery helper) because the shim path
    // is the build output itself, not a discovered dependency.
    let shim = install_dir.join("lib/libcudf_rust.so");
    println!("cargo:rerun-if-changed={}", shim.display());
    println!("cargo:cudf-rust-runtime-lib={}", shim.display());
    println!("cargo:metadata=cudf-rust-runtime-lib={}", shim.display());
    build_support::cmake::emit_runtime_dep_library_metadata();

    // Track all native, binding-generation, compiler, and discovery inputs
    // from the canonical fingerprint contract.
    build_support::emit_build_fingerprints(
        manifest_dir,
        native_dir,
        &build_support::lint::clang_format_config_path(manifest_dir),
    );
    shim
}
