use std::{env, path::PathBuf};

fn emit_rpath(path: &std::path::Path) {
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{}", path.display());
    }
}

fn default_runtime_dirs() -> Vec<PathBuf> {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let workspace_root = manifest_dir.parent().expect("safe crate is under workspace root");
    let workspace_parent = workspace_root.parent().unwrap_or(workspace_root);
    vec![
        workspace_parent.join("cuvs/cpp/build/c"),
        workspace_parent.join("cuvs/cpp/build"),
        workspace_parent.join("cudf-rs/cudf-install/lib"),
        PathBuf::from("/usr/local/cuda/lib64"),
    ]
}

fn main() {
    let dirs = env::var_os("DEP_CUVS_C_RUNTIME_LIB_DIRS")
        .map(|raw| env::split_paths(&raw).collect::<Vec<_>>())
        .unwrap_or_else(default_runtime_dirs);

    for dir in dirs {
        if dir.exists() {
            emit_rpath(&dir);
        }
    }

    println!("cargo:rerun-if-env-changed=DEP_CUVS_C_RUNTIME_LIB_DIRS");
}
