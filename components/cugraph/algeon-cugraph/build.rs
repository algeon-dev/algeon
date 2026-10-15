mod build_support;

use std::{collections::BTreeSet, env, path::PathBuf};

use build_support::{
    build_cuda_shim, cuda_toolkit_include_dirs, cuda_toolkit_lib_dirs, resolve_shim_compile_layout, system_include_args,
};

fn emit_link_search_path(path: &str) {
    println!("cargo:rustc-link-search=native={}", path);
}

fn emit_runtime_search_path(path: &str) {
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

fn insert_runtime_dir(runtime_dirs: &mut BTreeSet<String>, path: PathBuf) {
    if path.is_dir() {
        runtime_dirs.insert(path.display().to_string());
    }
}

fn insert_include_dirs_from_env_path_list(include_dirs: &mut Vec<PathBuf>, env_var: &str) {
    let Ok(value) = env::var_os(env_var).ok_or(()) else {
        return;
    };
    for path in env::split_paths(&value) {
        insert_include_dir(include_dirs, path);
    }
}

fn insert_include_dir(include_dirs: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() && !include_dirs.iter().any(|existing| existing == &path) {
        include_dirs.push(path);
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let mut runtime_dirs = BTreeSet::new();
    let mut native_include_dirs = Vec::new();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    insert_runtime_dirs_from_env_path_list(&mut runtime_dirs, "DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    if let Ok(path) = env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR") {
        runtime_dirs.insert(path);
    }
    if let Ok(root) = env::var("DEP_CUDF_RUST_ROOT") {
        let include = PathBuf::from(root).join("include");
        insert_include_dir(&mut native_include_dirs, include.join("rapids"));
        insert_include_dir(&mut native_include_dirs, include);
    }
    insert_include_dirs_from_env_path_list(&mut native_include_dirs, "DEP_CUGRAPH_NATIVE_INCLUDE_DIRS");
    let cuda_toolkit_root = env::var_os("CUDAToolkit_ROOT")
        .or_else(|| env::var_os("CUDA_HOME"))
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    for include_dir in cuda_toolkit_include_dirs(&cuda_toolkit_root) {
        insert_include_dir(&mut native_include_dirs, include_dir);
    }
    let mut native_link_dirs = Vec::new();
    insert_include_dirs_from_env_path_list(&mut native_link_dirs, "DEP_CUGRAPH_NATIVE_LINK_DIRS");
    // `-lrmm` and `-lcudart` below resolve out of the cuDF prefix and the CUDA
    // toolkit, neither of which is in the cuGraph prefix. Without these the
    // linker only sees them through an rpath, which sets the runtime loader
    // path but never the link-time search path, so linking depends on the
    // caller exporting LIBRARY_PATH. cudf-sys publishes the resolved
    // librmm.so itself, which pins the one RMM runtime the components share.
    if let Some(rmm_lib_dir) =
        env::var_os("DEP_CUDF_RUST_RMM_RUNTIME_LIB").map(PathBuf::from).and_then(|lib| lib.parent().map(PathBuf::from))
    {
        insert_include_dir(&mut native_link_dirs, rmm_lib_dir);
    }
    for lib_dir in cuda_toolkit_lib_dirs(&cuda_toolkit_root) {
        insert_include_dir(&mut native_link_dirs, lib_dir);
    }

    for env_var in [
        "DEP_CUGRAPH_RUNTIME_LIB_DIRS",
        "DEP_CUGRAPH_NATIVE_INCLUDE_DIRS",
        "DEP_CUGRAPH_NATIVE_LINK_DIRS",
        "DEP_CUGRAPH_CUDA_ARCHITECTURES",
        "CUDACXX",
        "CUDAHOSTCXX",
        "DEP_CUDF_RUST_ROOT",
        "DEP_CUDF_RUST_RUNTIME_LIB_DIR",
        "DEP_CUDF_RUST_RMM_RUNTIME_LIB",
        "CUDAToolkit_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={env_var}");
    }

    for path in [
        manifest_dir.join("shim").join("CMakeLists.txt"),
        manifest_dir.join("shim").join("edge_count.cu"),
        manifest_dir.join("shim").join("edge_count.h"),
        manifest_dir.join("shim").join("runtime_resource_handle.cpp"),
        manifest_dir.join("shim").join("runtime_resource_handle.h"),
        manifest_dir.join("shim").join("memory_resource_compat.hpp"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    for path in &native_link_dirs {
        emit_link_search_path(&path.display().to_string());
    }

    for path in runtime_dirs {
        emit_runtime_search_path(&path);
    }

    println!("cargo:rustc-link-lib=dylib=rmm");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let compile_layout =
        resolve_shim_compile_layout(&manifest_dir, native_include_dirs).unwrap_or_else(|message| panic!("{message}"));

    let output_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let architectures = env::var("DEP_CUGRAPH_CUDA_ARCHITECTURES")
        .expect("cugraph-sys must publish the resolved real CUDA architectures");
    build_cuda_shim(&compile_layout, &output_dir, &architectures).unwrap_or_else(|error| panic!("{error}"));
    emit_link_search_path(&output_dir.display().to_string());
    println!("cargo:rustc-link-lib=static=cugraph_rs_edge_count");

    let mut build = cc::Build::new();
    build.cpp(true);
    build.std("c++20");
    build.file(&compile_layout.source_file);
    build.include(&compile_layout.local_include_dir);

    for include_dir in &compile_layout.native_include_dirs {
        for arg in system_include_args(include_dir) {
            build.flag(arg);
        }
    }
    build.compile("cugraph_rs_runtime_resource_handle");
}
