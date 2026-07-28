use std::{env, path::PathBuf};

#[path = "build_support.rs"]
mod build_support;

fn emit_runtime_search_path(path: &std::path::Path) {
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}

fn emit_runtime_metadata(paths: &[PathBuf]) {
    if let Some(first) = paths.first() {
        println!("cargo:runtime-lib-dir={}", first.display());
        println!("cargo:metadata=runtime-lib-dir={}", first.display());
    }
    if !paths.is_empty() {
        let joined = env::join_paths(paths).expect("runtime library search paths must be valid OS strings");
        println!("cargo:runtime-lib-dirs={}", joined.to_string_lossy());
        println!("cargo:metadata=runtime-lib-dirs={}", joined.to_string_lossy());
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let include_from_env = env::var("CUGRAPH_INCLUDE_DIR").ok();
    let submodule_include = manifest_dir.join("../cugraph/cpp/include");
    let include_dirs = build_support::resolve_include_paths(include_from_env.as_deref(), &submodule_include)
        .unwrap_or_else(|err| panic!("{err}"));

    let library_from_env = env::var("CUGRAPH_LIBRARY_DIR").ok();
    let build_dir = manifest_dir.join("../cugraph/cpp/build");
    let submodule_lib_candidates = [build_dir.join("lib"), build_dir.clone()];
    let library_dirs = build_support::resolve_library_dirs(library_from_env.as_deref(), &submodule_lib_candidates)
        .unwrap_or_else(|err| panic!("{err}"));

    for library_dir in &library_dirs {
        println!("cargo:rustc-link-search=native={}", library_dir.display());
        emit_runtime_search_path(library_dir);
    }
    println!("cargo:rustc-link-lib=dylib=cugraph");
    println!("cargo:rustc-link-lib=dylib=cugraph_c");
    emit_runtime_metadata(&library_dirs);

    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CUGRAPH_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_LIBRARY_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=LIBRARY_PREFIX");
    println!("cargo:rerun-if-env-changed=PREFIX");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

    // layout_algorithms.h is not #included by wrapper.h (type collision workaround),
    // so CargoCallbacks cannot track it. Track it explicitly so changes to the
    // upstream layout API trigger binding regeneration.
    for include_dir in &include_dirs {
        let layout_header = include_dir.join("cugraph_c/layout_algorithms.h");
        if layout_header.exists() {
            println!("cargo:rerun-if-changed={}", layout_header.display());
        }
    }

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("cugraph_.*")
        .allowlist_type("cugraph_.*")
        .allowlist_type("bool_.*")
        .allowlist_type("data_type_id_.*")
        .allowlist_var("CUGRAPH_.*")
        // Map the C enums
        .allowlist_type("cugraph_error_code_.*")
        .allowlist_type("cugraph_data_type_id_.*")
        .allowlist_type("cugraph_graph_properties_t")
        // Derive common traits
        .derive_debug(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_hash(true)
        // Use newtype enums for FFI safety. Rust enums are UB-prone if C passes unknown values.
        .newtype_enum("cugraph_error_code_")
        .newtype_enum("bool_")
        .newtype_enum("data_type_id_");

    for include_dir in &include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    // Probe the selected C compiler for its system include search path so bindgen's
    // clang invocation follows the build toolchain instead of a hard-coded host GCC.
    let system_include_dir =
        build_support::detect_system_include_dir(env::var("CC").ok().as_deref()).unwrap_or_else(|err| panic!("{err}"));
    builder = builder.clang_arg(format!("-isystem{}", system_include_dir.display()));

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    bindings.write_to_file(&bindings_path).expect("Couldn't write bindings!");

    // Expose the generated bindings path to integration tests so they can
    // validate API coverage.
    println!("cargo:rustc-env=CUGRAPH_SYS_BINDINGS_RS={}", bindings_path.display());
}
