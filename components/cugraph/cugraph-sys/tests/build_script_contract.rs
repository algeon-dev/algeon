use std::{fs, path::PathBuf};

fn read_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("build.rs")).expect("failed to read build.rs")
}

fn read_cargo_toml() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("failed to read Cargo.toml")
}

fn read_downstream_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("../cugraph-rs/build.rs")).expect("failed to read downstream build.rs")
}

fn read_test_source(path: &str) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join(path)).unwrap_or_else(|_| panic!("failed to read {path}"))
}

#[test]
fn build_rs_uses_cargo_callbacks_for_transitive_header_tracking() {
    let build_rs = read_build_rs();
    assert!(
        build_rs.contains("CargoCallbacks"),
        "build.rs should use bindgen::CargoCallbacks so transitive header changes trigger regeneration"
    );
}

#[test]
fn build_rs_uses_newtype_enums_for_ffi_safety() {
    let build_rs = read_build_rs();

    assert!(
        build_rs.contains("newtype_enum(\"cugraph_error_code_\")"),
        "cugraph_error_code_ should be generated as a newtype enum"
    );
    assert!(build_rs.contains("newtype_enum(\"bool_\")"), "bool_ should be generated as a newtype enum");
    assert!(
        build_rs.contains("newtype_enum(\"data_type_id_\")"),
        "data_type_id_ should be generated as a newtype enum"
    );

    assert!(
        !build_rs.contains("rustified_enum(\"cugraph_error_code_\")"),
        "rustified C enums are UB-prone across FFI; use newtype enums instead"
    );
    assert!(
        !build_rs.contains("rustified_enum(\"bool_\")"),
        "rustified C enums are UB-prone across FFI; use newtype enums instead"
    );
    assert!(
        !build_rs.contains("rustified_enum(\"data_type_id_\")"),
        "rustified C enums are UB-prone across FFI; use newtype enums instead"
    );
}

#[test]
fn build_rs_tracks_layout_header_for_rerun() {
    let build_rs = read_build_rs();

    assert!(
        build_rs.contains("layout_algorithms.h"),
        "build.rs must contain a rerun-if-changed directive for layout_algorithms.h \
         because it is not #included by wrapper.h and CargoCallbacks cannot track it"
    );

    assert!(
        build_rs.contains("rerun-if-changed") && build_rs.contains("layout_algorithms.h"),
        "build.rs must emit cargo:rerun-if-changed for the layout header path"
    );
}

#[test]
fn build_rs_honors_cc_for_system_include_probing() {
    let build_rs = read_build_rs();

    assert!(
        build_rs.contains("env::var(\"CC\")"),
        "build.rs should honor CC when selecting the compiler used for system include probing"
    );
    assert!(
        build_rs.contains("cargo:rerun-if-env-changed=CC"),
        "build.rs should rerun when CC changes because include probing depends on the selected compiler"
    );
    assert!(
        !build_rs.contains("Command::new(\"gcc\")"),
        "build.rs should not hard-code gcc for system include probing"
    );
}

#[test]
fn build_rs_tracks_conda_and_prefix_env_changes_for_library_discovery() {
    let build_rs = read_build_rs();

    for env_var in ["CONDA_PREFIX", "LIBRARY_PREFIX", "PREFIX"] {
        assert!(
            build_rs.contains(&format!("cargo:rerun-if-env-changed={env_var}")),
            "build.rs should rerun when {env_var} changes because fallback library discovery depends on it"
        );
    }
}

#[test]
fn build_rs_tracks_pkg_config_env_changes_for_discovery() {
    let build_rs = read_build_rs();

    for env_var in ["PKG_CONFIG", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR", "PKG_CONFIG_SYSROOT_DIR"] {
        assert!(
            build_rs.contains(&format!("cargo:rerun-if-env-changed={env_var}")),
            "build.rs should rerun when {env_var} changes because pkg-config discovery depends on it"
        );
    }
}

#[test]
fn build_rs_emits_runtime_library_metadata_for_downstream_consumers() {
    let build_rs = read_build_rs();

    assert!(
        build_rs.contains("cargo:metadata=runtime-lib-dir="),
        "build.rs should emit the legacy single runtime library metadata key for downstream consumers"
    );
    assert!(
        build_rs.contains("cargo:metadata=runtime-lib-dirs="),
        "build.rs should emit the multi-path runtime library metadata key for downstream consumers"
    );
}

#[test]
fn downstream_build_rs_consumes_runtime_library_metadata_without_guessing_submodule_paths() {
    let build_rs = read_downstream_build_rs();

    assert!(
        build_rs.contains("DEP_CUGRAPH_RUNTIME_LIB_DIRS"),
        "cugraph-rs/build.rs should consume the multi-path runtime metadata from cugraph-sys"
    );
    assert!(
        build_rs.contains("env::split_paths"),
        "cugraph-rs/build.rs should parse DEP_CUGRAPH_RUNTIME_LIB_DIRS as an OS-native path list"
    );
    assert!(
        !build_rs.contains("../cugraph/cpp/build"),
        "cugraph-rs/build.rs should not guess workspace-local cuGraph build directories on its own"
    );
}

#[test]
fn include_sensitive_tests_reuse_build_support_include_resolution() {
    for path in ["tests/api_coverage.rs", "tests/layout_header_contract.rs"] {
        let source = read_test_source(path);
        assert!(
            source.contains("resolve_include_dir"),
            "{path} should reuse build_support::resolve_include_dir so an invalid CUGRAPH_INCLUDE_DIR fails instead of silently falling back"
        );
    }
}

#[test]
fn cargo_toml_declares_links_metadata_for_native_provider() {
    let cargo_toml = read_cargo_toml();

    assert!(
        cargo_toml.contains("links = \"cugraph\""),
        "cugraph-sys/Cargo.toml should declare links = \"cugraph\" so Cargo can coordinate the native library provider"
    );
}
