use std::{collections::BTreeSet, env, path::PathBuf};

fn emit_runtime_search_path(path: &str) {
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
}

fn main() {
    let mut runtime_dirs = BTreeSet::new();

    if let Ok(path) = env::var("DEP_CUGRAPH_RUNTIME_LIB_DIR") {
        runtime_dirs.insert(path);
    }
    if let Ok(path) = env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR") {
        runtime_dirs.insert(path);
    }

    // cuGraph tests in this repo link directly against the submodule build, so
    // keep the fallback deterministic even if dependency metadata is absent.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    runtime_dirs.insert(manifest_dir.join("../cugraph/cpp/build").display().to_string());

    for path in runtime_dirs {
        emit_runtime_search_path(&path);
    }
}
