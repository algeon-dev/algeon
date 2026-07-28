use std::{collections::BTreeSet, env, path::PathBuf};

fn emit_runtime_search_path(path: &str) {
    println!("cargo:rustc-link-search=native={}", path);
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
}

fn insert_runtime_dirs_from_env_path_list(runtime_dirs: &mut BTreeSet<String>, env_var: &str) {
    let Ok(value) = env::var_os(env_var).ok_or(()) else {
        return;
    };
    for path in env::split_paths(&value) {
        insert_runtime_dir(runtime_dirs, path);
    }
}

fn insert_runtime_dirs_from_metadata_path_list(runtime_dirs: &mut BTreeSet<String>, env_var: &str, key: &str) {
    let Ok(value) = env::var(env_var) else {
        return;
    };
    let Some(paths) = value.strip_prefix(&format!("{key}=")) else {
        return;
    };
    for path in env::split_paths(paths) {
        insert_runtime_dir(runtime_dirs, path);
    }
}

fn insert_runtime_dir_from_metadata(runtime_dirs: &mut BTreeSet<String>, env_var: &str, key: &str) {
    let Ok(value) = env::var(env_var) else {
        return;
    };
    let Some(path) = value.strip_prefix(&format!("{key}=")) else {
        return;
    };
    runtime_dirs.insert(path.to_string());
}

fn insert_runtime_dir(runtime_dirs: &mut BTreeSet<String>, path: PathBuf) {
    if path.is_dir() {
        runtime_dirs.insert(path.display().to_string());
    }
}

fn extend_conda_runtime_dirs(runtime_dirs: &mut BTreeSet<String>) {
    for env_var in ["CONDA_PREFIX", "LIBRARY_PREFIX", "PREFIX"] {
        let Some(prefix) = env::var_os(env_var) else {
            continue;
        };
        let prefix = PathBuf::from(prefix);
        insert_runtime_dir(runtime_dirs, prefix.clone());
        insert_runtime_dir(runtime_dirs, prefix.join("lib"));
        insert_runtime_dir(runtime_dirs, prefix.join("Library").join("lib"));
    }
}

fn main() {
    let mut runtime_dirs = BTreeSet::new();

    insert_runtime_dirs_from_env_path_list(&mut runtime_dirs, "DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    insert_runtime_dirs_from_metadata_path_list(&mut runtime_dirs, "DEP_CUGRAPH_METADATA", "runtime-lib-dirs");
    if let Ok(path) = env::var("DEP_CUGRAPH_RUNTIME_LIB_DIR") {
        runtime_dirs.insert(path);
    }
    insert_runtime_dir_from_metadata(&mut runtime_dirs, "DEP_CUGRAPH_METADATA", "runtime-lib-dir");
    if let Ok(path) = env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR") {
        runtime_dirs.insert(path);
    }
    insert_runtime_dir_from_metadata(&mut runtime_dirs, "DEP_CUDF_RUST_METADATA", "runtime-lib-dir");

    extend_conda_runtime_dirs(&mut runtime_dirs);

    for env_var in [
        "DEP_CUGRAPH_RUNTIME_LIB_DIRS",
        "DEP_CUGRAPH_RUNTIME_LIB_DIR",
        "DEP_CUGRAPH_METADATA",
        "DEP_CUDF_RUST_RUNTIME_LIB_DIR",
        "DEP_CUDF_RUST_METADATA",
        "CONDA_PREFIX",
        "LIBRARY_PREFIX",
        "PREFIX",
    ] {
        println!("cargo:rerun-if-env-changed={env_var}");
    }

    for path in runtime_dirs {
        emit_runtime_search_path(&path);
    }
}
