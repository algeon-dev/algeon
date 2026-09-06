mod cmake;

use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
};

const NATIVE_COMPONENT: nexus_build_support::NativeComponent = nexus_build_support::NativeComponent {
    crate_name: "cuvs-nexus-sys",
    install_dir_name: "cuvs-install",
    required_library: "lib/libcuvs.so",
    override_env: Some("CUVS_ROOT"),
    source_required_env: None,
    source_tree_sentinel: None,
    build_script: "scripts/build/build_libcuvs.sh",
};

fn emit_path_list(key: &str, paths: &[PathBuf]) -> String {
    let encoded = env::join_paths(paths).expect("native paths are valid OS strings");
    let encoded = encoded.to_string_lossy().into_owned();
    println!("cargo:{key}={encoded}");
    println!("cargo::metadata={key}={encoded}");
    encoded
}

fn emit_value(key: &str, value: &str) {
    println!("cargo:{key}={value}");
    println!("cargo::metadata={key}={value}");
}

struct CudaToolkit {
    include_dir: PathBuf,
    lib_dir: PathBuf,
}

fn checked_cuda_toolkit(root: PathBuf, selector: &str) -> Result<CudaToolkit, String> {
    let include = root.join("include");
    if !include.join("cuda_runtime.h").is_file() {
        return Err(format!(
            "{selector} does not identify a CUDA toolkit containing include/cuda_runtime.h: {}",
            root.display()
        ));
    }
    let include_dir = fs::canonicalize(&include)
        .map_err(|error| format!("failed to canonicalize CUDA include directory {}: {error}", include.display()))?;
    let mut lib_dirs = BTreeSet::new();
    for lib_dir in [root.join("lib64"), root.join("lib")] {
        if lib_dir.join("libcudart.so").is_file() {
            lib_dirs.insert(fs::canonicalize(&lib_dir).map_err(|error| {
                format!("failed to canonicalize CUDA library directory {}: {error}", lib_dir.display())
            })?);
        }
    }
    if lib_dirs.len() != 1 {
        return Err(format!(
            "{selector} must expose exactly one canonical dynamic libcudart.so directory under lib64 or lib: {}",
            root.display()
        ));
    }
    Ok(CudaToolkit { include_dir, lib_dir: lib_dirs.into_iter().next().expect("one CUDA library directory") })
}

fn cuda_toolkit() -> Result<CudaToolkit, String> {
    for selector in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(selector) {
            return checked_cuda_toolkit(PathBuf::from(root), selector);
        }
    }

    // Tracked only here: an unconditional PATH dependency would rebuild on
    // every shell difference even when a CUDA selector decides the toolkit.
    println!("cargo::rerun-if-env-changed=PATH");
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            let nvcc = directory.join("nvcc");
            if !nvcc.is_file() {
                continue;
            }
            let nvcc = fs::canonicalize(&nvcc)
                .map_err(|error| format!("failed to canonicalize nvcc at {}: {error}", nvcc.display()))?;
            let root = nvcc
                .parent()
                .and_then(|bin| bin.parent())
                .ok_or_else(|| format!("cannot determine the CUDA toolkit root from {}", nvcc.display()))?;
            return checked_cuda_toolkit(root.to_path_buf(), "nvcc from PATH");
        }
    }

    checked_cuda_toolkit(PathBuf::from("/usr/local/cuda"), "default CUDA toolkit root")
}

fn checked_dlpack_include(include: PathBuf, selector: &str) -> Result<PathBuf, String> {
    if !include.join("dlpack/dlpack.h").is_file() {
        return Err(format!(
            "{selector} does not identify an include directory containing dlpack/dlpack.h: {}",
            include.display()
        ));
    }
    fs::canonicalize(&include)
        .map_err(|error| format!("failed to canonicalize DLPack include directory {}: {error}", include.display()))
}

fn dlpack_include_dir(include_dirs: &[PathBuf]) -> Result<PathBuf, String> {
    if let Some(include) = env::var_os("DLPACK_INCLUDE_DIR") {
        return checked_dlpack_include(PathBuf::from(include), "DLPACK_INCLUDE_DIR");
    }
    for include in include_dirs {
        if include.join("dlpack/dlpack.h").is_file() {
            return checked_dlpack_include(include.clone(), "cuVS CMake target");
        }
    }
    Err(
        "the cuVS CMake target does not expose dlpack/dlpack.h; use a complete canonical prefix built by scripts/build/build_libcuvs.sh or set DLPACK_INCLUDE_DIR"
            .to_owned(),
    )
}

fn compile_abi_probe(include_dirs: &[PathBuf]) {
    let mut build = cc::Build::new();
    build.file("native/abi_probe.c").warnings(true);
    for include_dir in include_dirs {
        build.include(include_dir);
    }
    build.compile("cuvs_nexus_abi_probe");
}

fn dedup_paths(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    paths.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

fn select_prebuilt_cuvs_root(prefix: &Path) -> PathBuf {
    let prefix = fs::canonicalize(prefix)
        .unwrap_or_else(|error| panic!("failed to canonicalize cuVS prefix {}: {error}", prefix.display()));
    // SAFETY: Cargo build scripts are single-threaded, and native discovery
    // reads CUVS_ROOT only after this assignment.
    if env::var_os("CUVS_ROOT").is_none() {
        unsafe {
            env::set_var("CUVS_ROOT", &prefix);
        }
    }
    prefix
}

fn track_cuvs_prefix(prefix: &Path) {
    for relative in [
        "lib/cmake/cuvs/cuvs-config.cmake",
        "lib64/cmake/cuvs/cuvs-config.cmake",
        "share/cuvs-nexus/source-revision",
        "share/cuvs-nexus/rmm-runtime-lib",
        "share/cuvs-nexus/native-version",
    ] {
        let path = prefix.join(relative);
        if path.exists() {
            println!("cargo::rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=cmake.rs");
    println!("cargo::rerun-if-changed=cuvs_c_wrapper.h");
    println!("cargo::rerun-if-changed=bindgen-stubs/cuda_runtime.h");
    println!("cargo::rerun-if-changed=native/abi_probe.c");
    for name in [
        "CMAKE_PREFIX_PATH",
        "CONDA_PREFIX",
        "CUVS_ROOT",
        "LIBCUVS_USE_PYTHON",
        "VIRTUAL_ENV",
        "CUVS_SOURCE_DIR",
        "DLPACK_INCLUDE_DIR",
        "CUDAToolkit_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
        "DOCS_RS",
        "NEXUS_NATIVE_DIR",
        "CUVS_RS_SKIP_STALENESS_CHECK",
        "CUVS_RS_ENFORCE_STALENESS_CHECK",
        "CUVS_ENFORCE_STALENESS_CHECK",
    ] {
        println!("cargo::rerun-if-env-changed={name}");
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let linking =
        nexus_build_support::resolve_native_linking(&NATIVE_COMPONENT, &manifest_dir, &|name| env::var_os(name));
    if matches!(linking, nexus_build_support::NativeLinking::Skip) {
        return;
    }
    let selected_prefix = match linking {
        nexus_build_support::NativeLinking::Prebuilt(prefix) => Some(select_prebuilt_cuvs_root(&prefix)),
        nexus_build_support::NativeLinking::Missing(_) if env::var_os("LIBCUVS_USE_PYTHON").is_some() => None,
        nexus_build_support::NativeLinking::Missing(missing) => panic!("{missing}"),
        nexus_build_support::NativeLinking::SourceTree => unreachable!("cuVS has no source-tree native mode"),
        nexus_build_support::NativeLinking::Skip => unreachable!(),
    };
    if let Some(prefix) = &selected_prefix {
        track_cuvs_prefix(prefix);
    }
    if let Some(git_index) = cmake::recorded_cuvs_gitlink_input() {
        println!("cargo::rerun-if-changed={}", git_index.display());
    }
    let metadata = cmake::locate_cuvs().unwrap_or_else(|error| panic!("{error:#}"));
    cmake::check_libcuvs_staleness(&manifest_dir, &metadata);
    println!("cargo::rerun-if-changed={}", metadata.libcuvs_c.display());
    println!("cargo::rerun-if-changed={}", metadata.libcuvs.display());
    let cuda = cuda_toolkit().unwrap_or_else(|error| panic!("{error}"));
    let dlpack_include = dlpack_include_dir(&metadata.include_dirs).unwrap_or_else(|error| panic!("{error}"));
    let include_dirs =
        dedup_paths(metadata.include_dirs.iter().cloned().chain([cuda.include_dir.clone(), dlpack_include]));
    let native_link_dirs = dedup_paths(metadata.native_link_dirs.iter().cloned().chain([cuda.lib_dir.clone()]));
    let runtime_lib_dirs = dedup_paths(metadata.runtime_lib_dirs.iter().cloned().chain([cuda.lib_dir]));
    compile_abi_probe(&include_dirs);

    for path in &native_link_dirs {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for path in &runtime_lib_dirs {
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }

    let include_dirs = emit_path_list("include-dirs", &include_dirs);
    let native_link_dirs = emit_path_list("native-link-dirs", &native_link_dirs);
    let runtime_lib_dirs = emit_path_list("runtime-lib-dirs", &runtime_lib_dirs);
    emit_value("libcuvs-c-runtime-lib", &metadata.libcuvs_c.display().to_string());
    emit_value("libcuvs-runtime-lib", &metadata.libcuvs.display().to_string());
    emit_value("native-version", cmake::NATIVE_CUVS_VERSION);
    emit_value("source-revision", &metadata.source_revision);

    let aggregate = format!(
        "include-dirs={include_dirs};native-link-dirs={native_link_dirs};runtime-lib-dirs={runtime_lib_dirs};libcuvs-c-runtime-lib={};libcuvs-runtime-lib={};native-version={};source-revision={}",
        metadata.libcuvs_c.display(),
        metadata.libcuvs.display(),
        cmake::NATIVE_CUVS_VERSION,
        metadata.source_revision,
    );
    println!("cargo::metadata=metadata={aggregate}");

    println!("cargo:rustc-link-lib=dylib=cuvs_c");
    println!("cargo:rustc-link-lib=dylib=cuvs");
}
