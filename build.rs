// Bake an rpath into every binary produced by this crate so the dynamic
// loader resolves libcudf_rust.so from the matching cudf-sys CMake build dir
// instead of whatever LD_LIBRARY_PATH happens to point at. cudf-sys's
// `links = "cudf_rust"` package emits its runtime lib dir as a
// `cargo:metadata=runtime-lib-dir=<dir>` line, which Cargo forwards to
// direct dependents bundled into DEP_CUDF_RUST_METADATA="runtime-lib-dir=<dir>".
// (The bundled form is because cudf-sys's build script literally writes
// `cargo:metadata=...` rather than `cargo:runtime-lib-dir=...`, so cargo
// treats `metadata` as the key. We tolerate both spellings here.)
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_METADATA");

    let dir = std::env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR")
        .ok()
        .or_else(|| extract_runtime_lib_dir(std::env::var("DEP_CUDF_RUST_METADATA").ok().as_deref()));

    let Some(dir) = dir else {
        println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake");
        return;
    };

    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-examples=-Wl,-rpath,{dir}");
}

fn extract_runtime_lib_dir(metadata: Option<&str>) -> Option<String> {
    metadata?.strip_prefix("runtime-lib-dir=").map(str::to_owned)
}
