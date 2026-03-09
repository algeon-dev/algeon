use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Resolve include path: env var -> in-tree fallback
    let include_from_env = env::var("CUGRAPH_INCLUDE_DIR").ok();
    let include_dir = include_from_env
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../cugraph/cpp/include"));
    let include_dir = include_dir.canonicalize().unwrap_or_else(|e| {
        if include_from_env.is_some() {
            panic!(
                "CUGRAPH_INCLUDE_DIR='{}' does not exist: {e}.",
                include_dir.display()
            )
        } else {
            panic!(
                "Could not resolve cuGraph include directory '{}': {e}. \
                 Initialize the git submodule (git submodule update --init --recursive) \
                 or set CUGRAPH_INCLUDE_DIR.",
                include_dir.display()
            )
        }
    });

    // Verify the directory actually contains cuGraph headers.
    if !include_dir.join("cugraph_c").is_dir() {
        if let Some(val) = &include_from_env {
            panic!(
                "CUGRAPH_INCLUDE_DIR='{val}' does not contain cuGraph headers \
                 (expected a cugraph_c/ subdirectory inside '{}').",
                include_dir.display()
            );
        } else {
            panic!(
                "cuGraph include directory '{}' does not contain a cugraph_c/ subdirectory. \
                 The git submodule may be incomplete — try: git submodule update --init --recursive",
                include_dir.display()
            );
        }
    }

    // Resolve library path: env var -> in-tree fallback.
    // CMake may place the libraries in build/ or build/lib/ depending on
    // configuration, so we probe both locations when no env var is set.
    let library_from_env = env::var("CUGRAPH_LIBRARY_DIR").ok();
    let library_dir = if let Some(ref dir) = library_from_env {
        PathBuf::from(dir)
    } else {
        let build_dir = manifest_dir.join("../cugraph/cpp/build");
        let candidates = [build_dir.join("lib"), build_dir.clone()];
        candidates
            .iter()
            .find(|p| has_required_cugraph_libs(p))
            .cloned()
            .unwrap_or(candidates[0].clone())
    };

    // Use canonicalize when the path exists for a clean absolute path,
    // otherwise fall back to the raw path and warn. This lets bindgen
    // generate the FFI bindings even before libcugraph is compiled;
    // the linker will produce the actual error if the library is missing.
    let library_dir = match library_dir.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            println!(
                "cargo:warning=cuGraph library directory not found: {}. \
                 Build libcugraph first (see README) or set CUGRAPH_LIBRARY_DIR.",
                library_dir.display()
            );
            library_dir
        }
    };

    // If the directory exists, check that both required shared libraries are present.
    if library_dir.exists() && !has_required_cugraph_libs(&library_dir) {
        if let Some(dir) = &library_from_env {
            panic!(
                "CUGRAPH_LIBRARY_DIR='{}' does not contain both required libraries \
                 (libcugraph and libcugraph_c). Build cuGraph C API libraries first \
                 or set CUGRAPH_LIBRARY_DIR to the correct directory.",
                dir
            );
        } else {
            println!(
                "cargo:warning=cuGraph library directory '{}' does not contain both required \
                 libraries (libcugraph and libcugraph_c). Build cuGraph C API libraries first \
                 (see README) or set CUGRAPH_LIBRARY_DIR.",
                library_dir.display()
            );
        }
    }

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=cugraph");
    println!("cargo:rustc-link-lib=dylib=cugraph_c");

    println!("cargo:rerun-if-env-changed=CUGRAPH_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_LIBRARY_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

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
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");

    // Expose the generated bindings path to integration tests so they can
    // validate API coverage.
    println!(
        "cargo:rustc-env=CUGRAPH_SYS_BINDINGS_RS={}",
        bindings_path.display()
    );
}

fn has_required_cugraph_libs(dir: &Path) -> bool {
    has_library(dir, "cugraph") && has_library(dir, "cugraph_c")
}

fn has_library(dir: &Path, stem: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        return dir.join(format!("lib{stem}.dylib")).exists();
    }

    #[cfg(target_os = "windows")]
    {
        return dir.join(format!("{stem}.dll")).exists()
            || dir.join(format!("lib{stem}.dll")).exists()
            || dir.join(format!("{stem}.lib")).exists()
            || dir.join(format!("lib{stem}.lib")).exists();
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        // Linux and other Unix-like platforms.
        dir.join(format!("lib{stem}.so")).exists()
            || dir.join(format!("lib{stem}.so.0")).exists()
            || dir.join(format!("lib{stem}.so.1")).exists()
    }
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
