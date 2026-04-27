mod build_support;

use std::{collections::BTreeSet, env, path::PathBuf};

use build_support::{CompilerFamily, external_warning_control_args, resolve_shim_compile_layout, system_include_args};

fn emit_link_search_path(path: &str) {
    println!("cargo:rustc-link-search=native={}", path);
}

fn emit_runtime_search_path(path: &str) {
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
    let mut runtime_dirs = BTreeSet::new();
    let mut native_include_dirs = Vec::new();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    insert_runtime_dirs_from_env_path_list(&mut runtime_dirs, "DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    if let Ok(path) = env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR") {
        runtime_dirs.insert(path);
    }
    insert_include_dirs_from_env_path_list(&mut native_include_dirs, "DEP_CUGRAPH_NATIVE_INCLUDE_DIRS");
    let mut native_link_dirs = Vec::new();
    insert_include_dirs_from_env_path_list(&mut native_link_dirs, "DEP_CUGRAPH_NATIVE_LINK_DIRS");

    for env_var in [
        "DEP_CUGRAPH_RUNTIME_LIB_DIRS",
        "DEP_CUGRAPH_NATIVE_INCLUDE_DIRS",
        "DEP_CUGRAPH_NATIVE_LINK_DIRS",
        "DEP_CUDF_RUST_RUNTIME_LIB_DIR",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={env_var}");
    }

    for path in [
        manifest_dir.join("shim").join("runtime_resource_handle.cpp"),
        manifest_dir.join("shim").join("runtime_resource_handle.h"),
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

    let mut build = cc::Build::new();
    build.cpp(true);
    build.std("c++20");
    build.file(&compile_layout.source_file);
    build.include(&compile_layout.local_include_dir);

    let compiler_family = match build.get_compiler().is_like_msvc() {
        true => CompilerFamily::Msvc,
        false => CompilerFamily::GnuLike,
    };

    for arg in external_warning_control_args(compiler_family) {
        build.flag(arg);
    }
    for include_dir in &compile_layout.native_include_dirs {
        for arg in system_include_args(compiler_family, include_dir) {
            build.flag(arg);
        }
    }
    build.compile("cugraph_rs_runtime_resource_handle");
}
