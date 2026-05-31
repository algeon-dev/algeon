// Bake cuGraph runtime library paths into nexus-query-engine test binaries.
//
// The top-level adapter crate has its own build script, but this workspace
// member also owns integration tests. Those binaries link cugraph through the
// native graph executor and need the same runtime search path.
fn main() {
    if std::env::var_os("CARGO_FEATURE_CUGRAPH").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_RUNTIME_LIB_DIRS");

    match std::env::var("DEP_CUGRAPH_RUNTIME_LIB_DIRS") {
        Ok(dirs) => {
            for dir in std::env::split_paths(&dirs) {
                let dir = dir.display().to_string();
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
                println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{dir}");
            }
        },
        Err(_) => {
            println!(
                "cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_RUNTIME_LIB_DIRS; skipping rpath bake"
            );
        },
    }
}
