// Bake an rpath into every binary produced by this crate so the dynamic
// loader resolves RAPIDS shared libraries from the matching sibling checkouts
// instead of whatever LD_LIBRARY_PATH happens to point at. cudf-sys's
// `links = "cudf_rust"` package may forward its runtime lib dir either as
// DEP_CUDF_RUST_RUNTIME_LIB_DIR or bundled into DEP_CUDF_RUST_METADATA.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_METADATA");

    let cudf_dir = std::env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR")
        .ok()
        .or_else(|| extract_runtime_lib_dir(std::env::var("DEP_CUDF_RUST_METADATA").ok().as_deref()));
    if let Some(dir) = cudf_dir {
        emit_rpath(&dir);
    } else {
        println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake");
    }

    if std::env::var_os("CARGO_FEATURE_CUGRAPH").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    match std::env::var("DEP_CUGRAPH_RUNTIME_LIB_DIRS") {
        Ok(dirs) => {
            for dir in std::env::split_paths(&dirs) {
                emit_rpath(&dir.display().to_string());
            }
        },
        Err(_) => {
            println!(
                "cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_RUNTIME_LIB_DIRS; skipping rpath bake"
            );
        },
    }
}

fn emit_rpath(dir: &str) {
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-examples=-Wl,-rpath,{dir}");
}

fn extract_runtime_lib_dir(metadata: Option<&str>) -> Option<String> {
    metadata?.strip_prefix("runtime-lib-dir=").map(str::to_owned)
}
