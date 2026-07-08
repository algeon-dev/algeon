// Bake RAPIDS runtime library search paths into example binaries so they can
// start without a caller-managed LD_LIBRARY_PATH.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_METADATA");
    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_METADATA");

    let cudf_runtime_dir = std::env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR")
        .ok()
        .or_else(|| extract_metadata_value(std::env::var("DEP_CUDF_RUST_METADATA").ok().as_deref(), "runtime-lib-dir"));
    match cudf_runtime_dir {
        Some(dir) => emit_runtime_rpath(&dir),
        None => println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake"),
    }

    let cugraph_runtime_dirs = std::env::var("DEP_CUGRAPH_RUNTIME_LIB_DIRS")
        .ok()
        .or_else(|| extract_metadata_value(std::env::var("DEP_CUGRAPH_METADATA").ok().as_deref(), "runtime-lib-dirs"));
    match cugraph_runtime_dirs {
        Some(dirs) => {
            for dir in std::env::split_paths(&dirs) {
                emit_runtime_rpath(&dir.display().to_string());
            }
        },
        None => {
            println!("cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_*; skipping rpath bake");
        },
    }
}

fn emit_runtime_rpath(dir: &str) {
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
}

fn extract_metadata_value(metadata: Option<&str>, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    metadata?.split(';').find_map(|entry| entry.strip_prefix(&prefix).map(str::to_owned))
}
