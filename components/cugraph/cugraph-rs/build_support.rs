use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShimCompileLayout {
    pub(crate) source_file: PathBuf,
    pub(crate) local_include_dir: PathBuf,
    pub(crate) native_include_dirs: Vec<PathBuf>,
}

pub(crate) fn resolve_shim_compile_layout(
    manifest_dir: &Path,
    native_include_dirs: Vec<PathBuf>,
) -> Result<ShimCompileLayout, String> {
    if native_include_dirs.is_empty() {
        return Err("cugraph-rs/build.rs did not receive DEP_CUGRAPH_NATIVE_INCLUDE_DIRS from cugraph-sys".to_string());
    }

    Ok(ShimCompileLayout {
        source_file: manifest_dir.join("shim").join("runtime_resource_handle.cpp"),
        local_include_dir: manifest_dir.join("shim"),
        native_include_dirs: prioritize_rapids_cccl_include_dirs(native_include_dirs),
    })
}

pub(crate) fn prioritize_rapids_cccl_include_dirs(native_include_dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut cccl_dirs = Vec::new();
    let mut other_dirs = Vec::new();

    for dir in native_include_dirs {
        if is_rapids_cccl_include_dir(&dir) && !is_cuda_toolkit_include_dir(&dir) {
            cccl_dirs.push(dir);
        } else {
            other_dirs.push(dir);
        }
    }

    cccl_dirs.extend(other_dirs);
    cccl_dirs
}

fn is_rapids_cccl_include_dir(dir: &Path) -> bool {
    dir.join("cuda").join("std").join("version").is_file()
        || dir.join("cub").join("version.cuh").is_file()
        || dir.join("thrust").join("version.h").is_file()
}

fn is_cuda_toolkit_include_dir(dir: &Path) -> bool {
    let rendered = dir.to_string_lossy();
    rendered.contains("/cuda/targets/")
        || rendered.contains("/cuda-")
        || rendered.ends_with("/cuda/include")
        || rendered.contains("/cuda/include/")
}

pub(crate) fn system_include_args(dir: &Path) -> Vec<OsString> {
    vec![OsString::from("-isystem"), dir.as_os_str().to_owned()]
}
