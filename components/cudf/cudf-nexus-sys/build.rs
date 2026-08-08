mod build_support;

use std::{env, path::PathBuf};

fn main() {
    // Validate the Linux-only contract and resolve absolute compiler paths
    // before consulting the CMake cache, so cache invalidation and CMake see
    // the same toolchain identity.
    let (cc, cxx) = build_support::toolchain::prepare_cmake_toolchain();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let native_dir = build_support::lint::native_dir(&manifest_dir);
    let test_hooks = if env::var_os("CARGO_FEATURE_TEST_HOOKS").is_some() { "ON" } else { "OFF" };

    // A CMake cache may otherwise retain a different compiler, native prefix,
    // or test-hook configuration across Cargo rebuilds.
    build_support::cmake::discard_stale_cmake_build_dir(&cc, &cxx, test_hooks);
    build_support::cmake::check_libcudf_staleness(&manifest_dir);

    // Configure the native shim and opt-in linting.
    let mut config = cmake::Config::new(&native_dir);
    let clang_lint = build_support::cmake::normalized_cmake_bool_from_env("CUDF_RUST_CLANG_LINT")
        .unwrap_or_else(|error| panic!("{error}"));
    config.define("CUDF_RUST_ENABLE_CLANG_TIDY", clang_lint.unwrap_or("OFF"));
    config.define("CUDF_RUST_ENABLE_TEST_HOOKS", test_hooks);
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
        let clang_format_config = build_support::lint::clang_format_config_path(&manifest_dir);
        let clang_tidy_config = build_support::lint::clang_tidy_config_path(&manifest_dir);
        build_support::lint::ensure_file_exists(&clang_tidy_config, "clang-tidy config");
        let lint_files = build_support::lint::collect_native_clang_lint_files(&manifest_dir)
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

    // Build the shim, replay CMake's link manifest, and expose the resolved
    // runtime identity to downstream RAPIDS consumers.
    let install_dir = config.build();
    let directives =
        build_support::cmake::read_cargo_link_directives(&install_dir).unwrap_or_else(|error| panic!("{error}"));
    build_support::cmake::replay_cargo_link_directives(&directives);
    // The freshly built libcudf_rust shim lives under the CMake install
    // prefix and must be tracked as `cudf-rust-runtime-lib` so downstream
    // `scripts/build/rapids.rs` can bake the exact rpath. This is emitted
    // here (rather than inside the discovery helper) because the shim path
    // is the build output itself, not a discovered dependency.
    {
        let shim = install_dir.join("lib/libcudf_rust.so");
        if shim.is_file() {
            println!("cargo:rerun-if-changed={}", shim.display());
            println!("cargo:cudf-rust-runtime-lib={}", shim.display());
            println!("cargo:metadata=cudf-rust-runtime-lib={}", shim.display());
        } else {
            println!(
                "cargo:warning=expected libcudf_rust shim at {} but it was missing; downstream rpath will be incomplete",
                shim.display()
            );
        }
    }
    build_support::cmake::emit_runtime_dep_library_metadata();

    // Track all native, binding-generation, compiler, and discovery inputs
    // from the canonical fingerprint contract.
    build_support::emit_build_fingerprints(
        &manifest_dir,
        &native_dir,
        &build_support::lint::clang_format_config_path(&manifest_dir),
    );
}
