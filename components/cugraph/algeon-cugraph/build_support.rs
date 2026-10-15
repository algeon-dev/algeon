use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
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
        return Err(
            "cugraph build script (cugraph/build.rs) did not receive DEP_CUGRAPH_NATIVE_INCLUDE_DIRS from cugraph-sys"
                .to_string(),
        );
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

pub(crate) fn cuda_toolkit_include_dirs(toolkit_root: &Path) -> Vec<PathBuf> {
    let mut include_dirs = Vec::new();
    for include_dir in [
        toolkit_root.join("include"),
        toolkit_root.join("targets").join("x86_64-linux").join("include"),
        toolkit_root.join("targets").join("aarch64-linux").join("include"),
    ] {
        if include_dir.is_dir() && !include_dirs.contains(&include_dir) {
            include_dirs.push(include_dir.clone());
        }
        let cccl = include_dir.join("cccl");
        if cccl.is_dir() && !include_dirs.contains(&cccl) {
            include_dirs.push(cccl);
        }
    }
    include_dirs
}

/// Link-time search paths for the CUDA runtime this crate requests as
/// `-lcudart`. CUDA's native toolkit layout uses `lib64`, while the per-target
/// layout exposes the same libraries under `targets/<arch>/lib`.
pub(crate) fn cuda_toolkit_lib_dirs(toolkit_root: &Path) -> Vec<PathBuf> {
    let mut lib_dirs = Vec::new();
    for lib_dir in [
        toolkit_root.join("lib64"),
        toolkit_root.join("lib"),
        toolkit_root.join("targets").join("x86_64-linux").join("lib"),
        toolkit_root.join("targets").join("aarch64-linux").join("lib"),
    ] {
        if lib_dir.is_dir() && !lib_dirs.contains(&lib_dir) {
            lib_dirs.push(lib_dir);
        }
    }
    lib_dirs
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

pub(crate) fn build_cuda_shim(
    layout: &ShimCompileLayout,
    output_dir: &Path,
    architectures: &str,
) -> Result<(), String> {
    let build_dir = output_dir.join("edge-count-build");
    let include_dirs =
        layout.native_include_dirs.iter().map(|path| path.to_string_lossy()).collect::<Vec<_>>().join(";");
    let mut configure = Command::new("cmake");
    for (selector, cache_key) in
        [("CUDACXX", "CMAKE_CUDA_COMPILER"), ("CUDAHOSTCXX", "CMAKE_CUDA_HOST_COMPILER"), ("CXX", "CMAKE_CXX_COMPILER")]
    {
        if let Some(compiler) = env::var_os(selector).filter(|value| !value.is_empty()) {
            let mut argument = OsString::from(format!("-D{cache_key}="));
            argument.push(compiler);
            configure.arg(argument);
        }
    }
    let configured = configure
        .arg("-S")
        .arg(&layout.local_include_dir)
        .arg("-B")
        .arg(&build_dir)
        .args(["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"])
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={architectures}"))
        .arg(format!("-DCUGRAPH_NATIVE_INCLUDE_DIRS={include_dirs}"))
        .arg(format!("-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}", output_dir.display()))
        .status()
        .map_err(|error| format!("failed to configure component CUDA edge-count shim: {error}"))?;
    if !configured.success() {
        return Err(format!("component CUDA edge-count shim configure failed: {configured}"));
    }
    let built = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .args(["--target", "cugraph_rs_edge_count"])
        .status()
        .map_err(|error| format!("failed to compile component CUDA edge-count shim: {error}"))?;
    if !built.success() {
        return Err(format!("component CUDA edge-count shim build failed: {built}"));
    }
    Ok(())
}
