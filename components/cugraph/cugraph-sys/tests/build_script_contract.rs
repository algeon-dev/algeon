use std::{fs, path::PathBuf};

fn read_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("build.rs")).expect("failed to read build.rs")
}

fn read_cargo_toml() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("failed to read Cargo.toml")
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
