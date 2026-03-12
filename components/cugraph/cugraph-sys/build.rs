use std::{env, path::PathBuf, process::Command};

#[path = "build_support.rs"]
mod build_support;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let include_from_env = env::var("CUGRAPH_INCLUDE_DIR").ok();
    let submodule_include = manifest_dir.join("../cugraph/cpp/include");
    let include_dir = build_support::resolve_include_dir(include_from_env.as_deref(), &submodule_include)
        .unwrap_or_else(|err| panic!("{err}"));

    let library_from_env = env::var("CUGRAPH_LIBRARY_DIR").ok();
    let build_dir = manifest_dir.join("../cugraph/cpp/build");
    let submodule_lib_candidates = [build_dir.join("lib"), build_dir.clone()];
    let library_dir = build_support::resolve_library_dir(library_from_env.as_deref(), &submodule_lib_candidates)
        .unwrap_or_else(|err| panic!("{err}"));

    // If the directory exists, check that both required shared libraries are present.
    if library_dir.exists() && !build_support::has_required_cugraph_libs(&library_dir) {
        println!(
            "cargo:warning=cuGraph library directory '{}' does not contain both required \
             libraries (libcugraph and libcugraph_c). Build cuGraph C API libraries first \
             (see README) or set CUGRAPH_LIBRARY_DIR.",
            library_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=cugraph");
    println!("cargo:rustc-link-lib=dylib=cugraph_c");

    println!("cargo:rerun-if-env-changed=CUGRAPH_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_LIBRARY_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

    // layout_algorithms.h is not #included by wrapper.h (type collision workaround),
    // so CargoCallbacks cannot track it. Track it explicitly so changes to the
    // upstream layout API trigger binding regeneration.
    let layout_header = include_dir.join("cugraph_c/layout_algorithms.h");
    if layout_header.exists() {
        println!("cargo:rerun-if-changed={}", layout_header.display());
    }

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
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

    // Detect GCC system include paths so clang can find stddef.h etc.
    if let Some(gcc_include) = detect_gcc_include_dir() {
        builder = builder.clang_arg(format!("-isystem{}", gcc_include));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    bindings.write_to_file(&bindings_path).expect("Couldn't write bindings!");

    // Expose the generated bindings path to integration tests so they can
    // validate API coverage.
    println!("cargo:rustc-env=CUGRAPH_SYS_BINDINGS_RS={}", bindings_path.display());
}

/// Ask GCC for its built-in include directory (contains stddef.h, stdarg.h, etc.)
fn detect_gcc_include_dir() -> Option<String> {
    let output = Command::new("gcc")
        .args(["-v", "-E", "-x", "c", "-"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .ok()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut in_search = false;
    for line in stderr.lines() {
        if line.contains("#include <...> search starts here:") {
            in_search = true;
            continue;
        }
        if line.contains("End of search list.") {
            break;
        }
        if in_search {
            let path = line.trim();
            if PathBuf::from(path).join("stddef.h").exists() {
                return Some(path.to_string());
            }
        }
    }
    None
}
