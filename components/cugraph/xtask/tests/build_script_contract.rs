#[test]
fn xtask_build_rs_embeds_runtime_paths_from_native_metadata() {
    let build_rs = std::fs::read_to_string("build.rs").expect("read xtask/build.rs");
    let cargo_toml = std::fs::read_to_string("Cargo.toml").expect("read xtask/Cargo.toml");

    assert!(cargo_toml.contains("build = \"build.rs\""), "xtask should run its build script");
    assert!(
        cargo_toml.contains("cugraph-sys") && cargo_toml.contains("cudf-sys"),
        "xtask build.rs should receive direct normal-dependency native runtime metadata from sys crates"
    );
    assert!(
        build_rs.contains("DEP_CUGRAPH_RUNTIME_LIB_DIRS"),
        "xtask build.rs should consume cuGraph runtime library metadata"
    );
    assert!(
        build_rs.contains("DEP_CUDF_RUST_RUNTIME_LIB_DIR"),
        "xtask build.rs should consume cuDF runtime library metadata"
    );
    assert!(
        build_rs.contains("cargo:rustc-link-arg-bin=xtask=-Wl,-rpath"),
        "xtask binary should embed native runtime search paths for child-process CLI tests"
    );
}
