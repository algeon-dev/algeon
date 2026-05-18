use std::{collections::BTreeSet, env, path::PathBuf};

fn insert_runtime_dir(runtime_dirs: &mut BTreeSet<String>, path: PathBuf) {
    if path.is_dir() {
        runtime_dirs.insert(path.display().to_string());
    }
}

fn insert_runtime_dirs_from_env_path_list(runtime_dirs: &mut BTreeSet<String>, env_var: &str) {
    let Some(value) = env::var_os(env_var) else {
        return;
    };
    for path in env::split_paths(&value) {
        insert_runtime_dir(runtime_dirs, path);
    }
}

fn insert_runtime_dir_from_env(runtime_dirs: &mut BTreeSet<String>, env_var: &str) {
    let Some(value) = env::var_os(env_var) else {
        return;
    };
    insert_runtime_dir(runtime_dirs, PathBuf::from(value));
}

fn emit_runtime_search_path(path: &str) {
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg-bin=xtask=-Wl,-rpath,{path}");
}

fn main() {
    let mut runtime_dirs = BTreeSet::new();
    insert_runtime_dirs_from_env_path_list(&mut runtime_dirs, "DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    insert_runtime_dir_from_env(&mut runtime_dirs, "DEP_CUDF_RUST_RUNTIME_LIB_DIR");

    for env_var in ["DEP_CUGRAPH_RUNTIME_LIB_DIRS", "DEP_CUDF_RUST_RUNTIME_LIB_DIR"] {
        println!("cargo:rerun-if-env-changed={env_var}");
    }

    for path in runtime_dirs {
        emit_runtime_search_path(&path);
    }
}
