use std::path::{Path, PathBuf};

// Bake rpaths into crate binaries so native GPU libraries resolve from matching
// sibling build dirs, not whichever LD_LIBRARY_PATH is active.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_METADATA");
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_RMM_RUNTIME_LIB");
    println!("cargo:rerun-if-env-changed=DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB");
    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_RUNTIME_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_NATIVE_LINK_DIRS");
    println!("cargo:rerun-if-env-changed=DEP_CUGRAPH_METADATA");

    let cudf_runtime_dir = std::env::var("DEP_CUDF_RUST_RUNTIME_LIB_DIR")
        .ok()
        .or_else(|| extract_metadata_value(std::env::var("DEP_CUDF_RUST_METADATA").ok().as_deref(), "runtime-lib-dir"));

    match cudf_runtime_dir {
        Some(dir) => emit_runtime_rpath(&dir),
        None => println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake"),
    }

    if std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some() {
        let cugraph_runtime_dirs = std::env::var("DEP_CUGRAPH_RUNTIME_LIB_DIRS").ok().or_else(|| {
            extract_metadata_value(std::env::var("DEP_CUGRAPH_METADATA").ok().as_deref(), "runtime-lib-dirs")
        });

        match cugraph_runtime_dirs {
            Some(dirs) => {
                for dir in std::env::split_paths(&dirs) {
                    emit_runtime_rpath(&dir.display().to_string());
                }
            },
            None => {
                println!(
                    "cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_*; skipping rpath bake"
                );
            },
        }

        enforce_single_rapids_runtime_libraries();
    }
}

fn emit_runtime_rpath(dir: &str) {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{dir}");
}

fn extract_metadata_value(metadata: Option<&str>, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    metadata?.split(';').find_map(|entry| entry.strip_prefix(&prefix).map(str::to_owned))
}

/// RAPIDS runtime libraries with process-global state, such as RMM and
/// rapids_logger, must come from one build. The dynamic loader binds exported
/// symbols process-wide, so divergent cudf/cugraph copies can execute against
/// mismatched globals and crash at runtime.
fn enforce_single_rapids_runtime_libraries() {
    for (lib_name, cudf_dep_env, cudf_metadata_key) in [
        ("librmm.so", "DEP_CUDF_RUST_RMM_RUNTIME_LIB", "rmm-runtime-lib"),
        ("librapids_logger.so", "DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB", "rapids-logger-runtime-lib"),
    ] {
        let cudf_lib = std::env::var(cudf_dep_env)
            .ok()
            .or_else(|| {
                extract_metadata_value(std::env::var("DEP_CUDF_RUST_METADATA").ok().as_deref(), cudf_metadata_key)
            })
            .map(PathBuf::from);

        let Some(cudf_lib) = cudf_lib else {
            panic!(
                "the `cugraph` feature requires libcudf to ship a shared {lib_name}, but cudf-sys did not expose \
                 one via {cudf_dep_env}. Rebuild libcudf with CUDF_RMM_BUILD_SHARED=ON (the cudf fork default), \
                 re-run `cmake --install`, and rebuild cudf-sys so the install prefix exports a single canonical \
                 {lib_name} for every RAPIDS consumer in the process."
            );
        };
        println!("cargo:rerun-if-changed={}", cudf_lib.display());

        let cugraph_libs = cugraph_native_library_candidates(lib_name);
        if cugraph_libs.is_empty() {
            panic!(
                "the `cugraph` feature is enabled but no {lib_name} is reachable from the cugraph-sys link/runtime \
                 dirs (DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS), so the build cannot prove \
                 libcugraph links the same {lib_name} as libcudf ({}). Reconfigure the cugraph C++ build against \
                 the cudf install prefix (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>) and \
                 rebuild it.",
                cudf_lib.display()
            );
        }

        for candidate in cugraph_libs {
            println!("cargo:rerun-if-changed={}", candidate.display());
            if same_library_build(&cudf_lib, &candidate) {
                continue;
            }
            panic!(
                "two divergent {lib_name} builds are linked into this process:\n  cudf-sys:    {}\n  cugraph-sys: {}\n\
                 One process must bind every {lib_name} symbol to a single build, or execution corrupts at runtime. \
                 Reconfigure the cugraph C++ build against the cudf install prefix \
                 (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>), rebuild it, and remove stale \
                 copies from the cugraph build tree.",
                cudf_lib.display(),
                candidate.display()
            );
        }
    }
}

/// Collect cugraph-visible `lib_name` copies from native link and runtime dirs.
/// Checking both lists catches stale copies that baked rpaths could load first.
fn cugraph_native_library_candidates(lib_name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for env_name in ["DEP_CUGRAPH_NATIVE_LINK_DIRS", "DEP_CUGRAPH_RUNTIME_LIB_DIRS"] {
        let Ok(dirs) = std::env::var(env_name) else { continue };
        for dir in std::env::split_paths(&dirs) {
            let candidate = dir.join(lib_name);
            if candidate.is_file() && !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

/// Two paths name the same build if they canonicalize to one file, share an
/// inode, or are byte-identical. Other copies are treated as divergent because
/// the build cannot prove one ABI and one set of globals.
fn same_library_build(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left_canonical), Ok(right_canonical)) if left_canonical == right_canonical => return true,
        _ => {},
    }
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(left_meta), Ok(right_meta)) = (std::fs::metadata(left), std::fs::metadata(right))
            && left_meta.dev() == right_meta.dev()
            && left_meta.ino() == right_meta.ino()
        {
            return true;
        }
    }
    match (std::fs::read(left), std::fs::read(right)) {
        (Ok(left_bytes), Ok(right_bytes)) => left_bytes == right_bytes,
        _ => false,
    }
}
