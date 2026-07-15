//! Shared Cargo build-script support for RAPIDS runtime libraries.
//!
//! This file is included by workspace build scripts. It deliberately is not a
//! crate: none of the runtime packages should depend on build-time policy.

use std::path::{Path, PathBuf};

const CUDF_METADATA_ENV: &str = "DEP_CUDF_RUST_METADATA";
const CUDF_RUNTIME_LIB_DIR_ENV: &str = "DEP_CUDF_RUST_RUNTIME_LIB_DIR";
const CUDF_LIBCUDF_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB";
const CUDF_RMM_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_RMM_RUNTIME_LIB";
const CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB";
const CUDF_RAPIDS_LOGGER_LINKAGE_ENV: &str = "DEP_CUDF_RUST_RAPIDS_LOGGER_LINKAGE";

const CUGRAPH_METADATA_ENV: &str = "DEP_CUGRAPH_METADATA";
const CUGRAPH_RUNTIME_LIB_DIRS_ENV: &str = "DEP_CUGRAPH_RUNTIME_LIB_DIRS";
const CUGRAPH_NATIVE_LINK_DIRS_ENV: &str = "DEP_CUGRAPH_NATIVE_LINK_DIRS";

#[derive(Clone, Copy)]
pub struct LinkArgTargets {
    all: bool,
    bins: bool,
    tests: bool,
}

impl LinkArgTargets {
    pub const fn new(all: bool, bins: bool, tests: bool) -> Self {
        Self { all, bins, tests }
    }

    fn emit_rpath(self, dir: &Path) {
        let arg = format!("-Wl,-rpath,{}", dir.display());
        if self.all {
            println!("cargo:rustc-link-arg={arg}");
        }
        if self.bins {
            println!("cargo:rustc-link-arg-bins={arg}");
        }
        if self.tests {
            println!("cargo:rustc-link-arg-tests={arg}");
        }
    }

    fn emit_required_runtime_library(self, library: &Path) {
        // Promote libcudf_rust's native children to direct NEEDED entries. The
        // wrapper carries a $ORIGIN-only DT_RUNPATH, and a binary's DT_RUNPATH
        // is not inherited when resolving those children. Direct entries let
        // this binary's selected RAPIDS rpath resolve the authoritative files.
        let arg = format!("-Wl,--push-state,--no-as-needed,{},--pop-state", library.display());
        if self.all {
            println!("cargo:rustc-link-arg={arg}");
        }
        if self.bins {
            println!("cargo:rustc-link-arg-bins={arg}");
        }
        if self.tests {
            println!("cargo:rustc-link-arg-tests={arg}");
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Each including package selects only its own feature policy.
pub enum CugraphRuntime {
    Disabled,
    EmitRpathAndRequireCudfIdentity,
}

pub struct Config {
    /// Path as Cargo should track it, relative to the package manifest.
    pub support_source: &'static str,
    pub link_arg_targets: LinkArgTargets,
    pub cugraph_runtime: CugraphRuntime,
}

pub fn configure(config: Config) {
    println!("cargo:rerun-if-changed={}", config.support_source);
    track_env(CUDF_METADATA_ENV);
    track_env(CUDF_RUNTIME_LIB_DIR_ENV);
    track_env(CUDF_LIBCUDF_RUNTIME_LIB_ENV);
    track_env(CUDF_RMM_RUNTIME_LIB_ENV);
    track_env(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV);

    let mut runtime_dirs = Vec::new();
    match cudf_runtime_lib_dir() {
        Some(dir) => push_unique_path(&mut runtime_dirs, dir),
        None => {
            println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake")
        },
    }
    let libcudf_runtime_lib = cudf_shared_runtime_library(CUDF_LIBCUDF_RUNTIME_LIB_ENV, "libcudf-runtime-lib");
    let rmm_runtime_lib = cudf_shared_runtime_library(CUDF_RMM_RUNTIME_LIB_ENV, "rmm-runtime-lib");
    let rapids_logger_runtime_lib =
        cudf_shared_runtime_library(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV, "rapids-logger-runtime-lib");
    for runtime_lib in [libcudf_runtime_lib.as_ref(), rmm_runtime_lib.as_ref(), rapids_logger_runtime_lib.as_ref()]
        .into_iter()
        .flatten()
    {
        if let Some(parent) = runtime_lib.parent() {
            push_unique_path(&mut runtime_dirs, parent.to_path_buf());
        }
    }

    let cugraph_enabled = match config.cugraph_runtime {
        CugraphRuntime::Disabled => false,
        CugraphRuntime::EmitRpathAndRequireCudfIdentity => true,
    };
    if cugraph_enabled {
        track_cugraph_env();
        match cugraph_runtime_lib_dirs() {
            Some(dirs) => {
                for dir in dirs {
                    push_unique_path(&mut runtime_dirs, dir);
                }
            },
            None => {
                println!(
                    "cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_*; skipping rpath bake"
                )
            },
        }
    }
    emit_runtime_rpaths(&runtime_dirs, config.link_arg_targets);
    match libcudf_runtime_lib {
        Some(library) => config.link_arg_targets.emit_required_runtime_library(&library),
        None => println!(
            "cargo:warning=cudf-sys did not expose its authoritative libcudf.so; \
             transitive native resolution may require an ambient loader path"
        ),
    }
    match rmm_runtime_lib {
        Some(library) => config.link_arg_targets.emit_required_runtime_library(&library),
        None => println!(
            "cargo:warning=cudf-sys did not expose its authoritative librmm.so; \
             transitive native resolution may require an ambient loader path"
        ),
    }
    if let Some(library) = rapids_logger_runtime_lib {
        config.link_arg_targets.emit_required_runtime_library(&library);
    }

    if cugraph_enabled {
        enforce_single_rapids_runtime_libraries();
    }
}

fn emit_runtime_rpaths(runtime_dirs: &[PathBuf], targets: LinkArgTargets) {
    for dir in runtime_dirs {
        targets.emit_rpath(dir);
    }
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}

fn track_env(name: &str) {
    println!("cargo:rerun-if-env-changed={name}");
}

fn track_cugraph_env() {
    track_env(CUGRAPH_RUNTIME_LIB_DIRS_ENV);
    track_env(CUGRAPH_NATIVE_LINK_DIRS_ENV);
    track_env(CUGRAPH_METADATA_ENV);
}

fn dependency_metadata_value(direct_env: &str, metadata_env: &str, metadata_key: &str) -> Option<String> {
    select_metadata_value(
        std::env::var(direct_env).ok().as_deref(),
        std::env::var(metadata_env).ok().as_deref(),
        metadata_key,
    )
}

fn select_metadata_value(direct_value: Option<&str>, metadata: Option<&str>, key: &str) -> Option<String> {
    direct_value.map(str::to_owned).or_else(|| extract_metadata_value(metadata, key))
}

fn extract_metadata_value(metadata: Option<&str>, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    metadata?.split(';').find_map(|entry| entry.strip_prefix(&prefix).map(str::to_owned))
}

fn cudf_runtime_lib_dir() -> Option<PathBuf> {
    dependency_metadata_value(CUDF_RUNTIME_LIB_DIR_ENV, CUDF_METADATA_ENV, "runtime-lib-dir").map(PathBuf::from)
}

fn cugraph_runtime_lib_dirs() -> Option<Vec<PathBuf>> {
    dependency_path_list(CUGRAPH_RUNTIME_LIB_DIRS_ENV, CUGRAPH_METADATA_ENV, "runtime-lib-dirs")
}

fn dependency_path_list(direct_env: &str, metadata_env: &str, metadata_key: &str) -> Option<Vec<PathBuf>> {
    let encoded = dependency_metadata_value(direct_env, metadata_env, metadata_key)?;
    Some(std::env::split_paths(&encoded).collect())
}

/// RAPIDS runtime libraries with process-global state, such as RMM and
/// rapids_logger, must come from one build. The dynamic loader binds exported
/// symbols process-wide, so divergent cudf/cugraph copies can execute against
/// mismatched globals and crash at runtime.
fn enforce_single_rapids_runtime_libraries() {
    track_env(CUDF_RAPIDS_LOGGER_LINKAGE_ENV);

    enforce_required_shared_rapids_library("librmm.so", CUDF_RMM_RUNTIME_LIB_ENV, "rmm-runtime-lib");
    enforce_optional_rapids_logger_from_cudf();
}

fn enforce_optional_rapids_logger_from_cudf() {
    if let Some(cudf_logger_lib) =
        cudf_shared_runtime_library(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV, "rapids-logger-runtime-lib")
    {
        enforce_shared_rapids_library_matches_cudf("librapids_logger.so", &cudf_logger_lib);
        return;
    }

    let logger_linkage =
        dependency_metadata_value(CUDF_RAPIDS_LOGGER_LINKAGE_ENV, CUDF_METADATA_ENV, "rapids-logger-linkage")
            .unwrap_or_else(|| "unknown".to_string());
    println!(
        "cargo:warning=cudf-sys did not expose a shared librapids_logger.so (linkage={logger_linkage}); \
         skipping cross-crate rapids_logger identity enforcement"
    );
}

fn enforce_required_shared_rapids_library(lib_name: &str, cudf_dep_env: &str, cudf_metadata_key: &str) {
    let Some(cudf_lib) = cudf_shared_runtime_library(cudf_dep_env, cudf_metadata_key) else {
        panic!(
            "the `cugraph` feature requires libcudf to ship a shared {lib_name}, but cudf-sys did not expose \
             one via {cudf_dep_env}. Rebuild libcudf with CUDF_RMM_BUILD_SHARED=ON (the cudf fork default), \
             re-run `cmake --install`, and rebuild cudf-sys so the install prefix exports a single canonical \
             {lib_name} for every RAPIDS consumer in the process."
        );
    };
    enforce_shared_rapids_library_matches_cudf(lib_name, &cudf_lib);
}

fn cudf_shared_runtime_library(cudf_dep_env: &str, cudf_metadata_key: &str) -> Option<PathBuf> {
    dependency_metadata_value(cudf_dep_env, CUDF_METADATA_ENV, cudf_metadata_key).map(PathBuf::from)
}

fn enforce_shared_rapids_library_matches_cudf(lib_name: &str, cudf_lib: &Path) {
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
        if same_library_build(cudf_lib, &candidate) {
            continue;
        }
        divergent_library_panic(lib_name, cudf_lib, &candidate);
    }
}

fn divergent_library_panic(lib_name: &str, cudf_lib: &Path, cugraph_lib: &Path) -> ! {
    panic!(
        "two divergent {lib_name} builds are linked into this process:\n  cudf-sys:    {}\n  cugraph-sys: {}\n\
         One process must bind every {lib_name} symbol to a single build, or execution corrupts at runtime. \
         Reconfigure the cugraph C++ build against the cudf install prefix \
         (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>), rebuild it, and remove stale \
         copies from the cugraph build tree.",
        cudf_lib.display(),
        cugraph_lib.display()
    );
}

/// Collect cugraph-visible `lib_name` copies from native link and runtime dirs.
/// Checking both lists catches stale copies that baked rpaths could load first.
fn cugraph_native_library_candidates(lib_name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for dirs in [
        dependency_path_list(CUGRAPH_NATIVE_LINK_DIRS_ENV, CUGRAPH_METADATA_ENV, "native-link-dirs"),
        cugraph_runtime_lib_dirs(),
    ]
    .into_iter()
    .flatten()
    {
        for dir in dirs {
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
        (Ok(left_canonical), Ok(right_canonical)) if left_canonical == right_canonical => {
            return true;
        },
        _ => {},
    }
    #[cfg(unix)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_metadata_value_wins_over_aggregate_fallback() {
        assert_eq!(
            select_metadata_value(
                Some("/direct/libcudf.so"),
                Some("libcudf-runtime-lib=/metadata/libcudf.so;other=value"),
                "libcudf-runtime-lib",
            ),
            Some("/direct/libcudf.so".to_string())
        );
    }

    #[test]
    fn aggregate_metadata_value_is_used_as_fallback() {
        assert_eq!(
            select_metadata_value(
                None,
                Some(
                    "libcudf-runtime-lib=/native/libcudf.so;rmm-runtime-lib=/rmm/librmm.so;\
                     rapids-logger-runtime-lib=/logger/librapids_logger.so"
                ),
                "libcudf-runtime-lib",
            ),
            Some("/native/libcudf.so".to_string())
        );
        assert_eq!(
            select_metadata_value(
                None,
                Some("rmm-runtime-lib=/rmm/librmm.so;rapids-logger-runtime-lib=/logger/librapids_logger.so"),
                "rmm-runtime-lib",
            ),
            Some("/rmm/librmm.so".to_string())
        );
        assert_eq!(select_metadata_value(None, Some("other=value"), "libcudf-runtime-lib"), None);
    }

    #[test]
    fn library_identity_accepts_same_path_and_identical_files() {
        let test_dir = unique_test_dir("identity");
        std::fs::create_dir_all(&test_dir).unwrap();
        let first = test_dir.join("first.so");
        let identical = test_dir.join("identical.so");
        let different = test_dir.join("different.so");
        std::fs::write(&first, b"same build").unwrap();
        std::fs::write(&identical, b"same build").unwrap();
        std::fs::write(&different, b"different build").unwrap();

        assert!(same_library_build(&first, &first));
        assert!(same_library_build(&first, &identical));
        assert!(!same_library_build(&first, &different));

        std::fs::remove_dir_all(test_dir).unwrap();
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "datafusion-nexus-rapids-build-support-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }
}
