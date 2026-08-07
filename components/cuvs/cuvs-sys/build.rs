use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
};

fn split_paths(env_name: &str, value: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    env::split_paths(&value)
        .map(|path| {
            if path.exists() {
                Ok(path)
            } else {
                Err(format!("{env_name} contains a path that does not exist: {}", path.display()))
            }
        })
        .collect()
}

fn workspace_parent(manifest_dir: &Path) -> Result<PathBuf, String> {
    let workspace_root = manifest_dir
        .parent()
        .ok_or_else(|| format!("cannot determine workspace root from {}", manifest_dir.display()))?;
    Ok(workspace_root.parent().unwrap_or(workspace_root).to_path_buf())
}

fn default_link_dirs(manifest_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let parent = workspace_parent(manifest_dir)?;
    let cuvs_build =
        env::var_os("CUVS_CMAKE_BUILD_DIR").map(PathBuf::from).unwrap_or_else(|| parent.join("cuvs/cpp/build"));
    Ok(vec![cuvs_build.join("c"), cuvs_build])
}

fn default_runtime_dirs(manifest_dir: &Path, link_dirs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let parent = workspace_parent(manifest_dir)?;
    let cuda_lib_dir = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("CUDA_PATH").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
        .join("lib64");
    let mut dirs = link_dirs.to_vec();
    dirs.push(parent.join("cudf-rs/cudf-install/lib"));
    dirs.push(cuda_lib_dir);
    Ok(dirs)
}

fn validate_file(path: &Path, what: &str) {
    if !path.is_file() {
        panic!("{what} was not found at '{}'", path.display());
    }
}

fn emit_path_metadata(key: &str, paths: &[PathBuf]) {
    if paths.is_empty() {
        return;
    }
    let joined = env::join_paths(paths).expect("native paths must be valid OS strings");
    println!("cargo::metadata={key}={}", joined.to_string_lossy());
}

fn emit_rpath(path: &Path) {
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));

    let explicit_link_dirs = split_paths("CUVS_NATIVE_LINK_DIRS", env::var_os("CUVS_NATIVE_LINK_DIRS"))
        .unwrap_or_else(|err| panic!("{err}"));
    let link_dirs = if explicit_link_dirs.is_empty() {
        default_link_dirs(&manifest_dir).unwrap_or_else(|err| panic!("{err}"))
    } else {
        explicit_link_dirs
    };

    let explicit_runtime_dirs = split_paths("CUVS_RUNTIME_LIB_DIRS", env::var_os("CUVS_RUNTIME_LIB_DIRS"))
        .unwrap_or_else(|err| panic!("{err}"));
    let runtime_dirs = if explicit_runtime_dirs.is_empty() {
        default_runtime_dirs(&manifest_dir, &link_dirs).unwrap_or_else(|err| panic!("{err}"))
    } else {
        explicit_runtime_dirs
    };

    let include_dirs = split_paths("CUVS_NATIVE_INCLUDE_DIRS", env::var_os("CUVS_NATIVE_INCLUDE_DIRS"))
        .unwrap_or_else(|err| panic!("{err}"));

    let cuvs_c_dir = link_dirs
        .iter()
        .find(|dir| dir.join("libcuvs_c.so").is_file())
        .unwrap_or_else(|| panic!("CUVS_NATIVE_LINK_DIRS did not contain libcuvs_c.so: {link_dirs:?}"));
    let cuvs_dir = link_dirs
        .iter()
        .find(|dir| dir.join("libcuvs.so").is_file())
        .unwrap_or_else(|| panic!("CUVS_NATIVE_LINK_DIRS did not contain libcuvs.so: {link_dirs:?}"));

    validate_file(&cuvs_c_dir.join("libcuvs_c.so"), "libcuvs_c.so");
    validate_file(&cuvs_dir.join("libcuvs.so"), "libcuvs.so");

    for dir in &link_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for dir in &runtime_dirs {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    for dir in &runtime_dirs {
        if dir.exists() {
            emit_rpath(dir);
        }
    }

    println!("cargo:rustc-link-lib=dylib=cuvs_c");
    println!("cargo:rustc-link-lib=dylib=cuvs");
    println!("cargo:rustc-link-lib=dylib=cudart");

    emit_path_metadata("native-link-dirs", &link_dirs);
    emit_path_metadata("native-include-dirs", &include_dirs);
    emit_path_metadata("runtime-lib-dirs", &runtime_dirs);

    println!("cargo:rerun-if-env-changed=CUVS_CMAKE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUVS_NATIVE_LINK_DIRS");
    println!("cargo:rerun-if-env-changed=CUVS_NATIVE_INCLUDE_DIRS");
    println!("cargo:rerun-if-env-changed=CUVS_RUNTIME_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    for dir in &link_dirs {
        println!("cargo:rerun-if-changed={}", dir.display());
    }
}
