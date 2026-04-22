//! Shared Cargo build-script support for RAPIDS runtime libraries, colocated
//! with the repository's other build tooling under `scripts/build`.
//!
//! This file is included by workspace build scripts. It deliberately is not a
//! crate: none of the runtime packages should depend on build-time policy.
//!
//! Environment capture, pure metadata/candidate/identity policy, filesystem
//! observation, and Cargo stdout emission are kept as separate stages so the
//! safety policy can be tested without real native libraries.

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

const CUDF_METADATA_ENV: &str = "DEP_CUDF_RUST_METADATA";
const CUDF_RUNTIME_LIB_DIR_ENV: &str = "DEP_CUDF_RUST_RUNTIME_LIB_DIR";
const CUDF_LIBCUDF_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB";
const CUDF_RMM_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_RMM_RUNTIME_LIB";
const CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB";
const CUDF_RAPIDS_LOGGER_LINKAGE_ENV: &str = "DEP_CUDF_RUST_RAPIDS_LOGGER_LINKAGE";

const CUGRAPH_METADATA_ENV: &str = "DEP_CUGRAPH_METADATA";
const CUGRAPH_RUNTIME_LIB_DIRS_ENV: &str = "DEP_CUGRAPH_RUNTIME_LIB_DIRS";
const CUGRAPH_NATIVE_LINK_DIRS_ENV: &str = "DEP_CUGRAPH_NATIVE_LINK_DIRS";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkArgTargets {
    all: bool,
    bins: bool,
    tests: bool,
}

impl LinkArgTargets {
    pub const fn new(all: bool, bins: bool, tests: bool) -> Self {
        Self { all, bins, tests }
    }

    fn directives(self, arg: &str) -> Vec<String> {
        let mut directives = Vec::new();
        if self.all {
            directives.push(format!("cargo:rustc-link-arg={arg}"));
        }
        if self.bins {
            directives.push(format!("cargo:rustc-link-arg-bins={arg}"));
        }
        if self.tests {
            directives.push(format!("cargo:rustc-link-arg-tests={arg}"));
        }
        directives
    }

    fn rpath_directives(self, dir: &Path) -> Vec<String> {
        self.directives(&format!("-Wl,-rpath,{}", dir.display()))
    }

    fn required_runtime_library_directives(self, library: &Path) -> Vec<String> {
        // Promote libcudf_rust's native children to direct NEEDED entries. The
        // wrapper carries a $ORIGIN-only DT_RUNPATH, and a binary's DT_RUNPATH
        // is not inherited when resolving those children. Direct entries let
        // this binary's selected RAPIDS rpath resolve the authoritative files.
        self.directives(&format!("-Wl,--push-state,--no-as-needed,{},--pop-state", library.display()))
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CudfEnvironment {
    aggregate_metadata: Option<String>,
    runtime_lib_dir: Option<String>,
    libcudf_runtime_lib: Option<String>,
    rmm_runtime_lib: Option<String>,
    rapids_logger_runtime_lib: Option<String>,
    rapids_logger_linkage: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CugraphEnvironment {
    aggregate_metadata: Option<String>,
    runtime_lib_dirs: Option<String>,
    native_link_dirs: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct EnvironmentSnapshot {
    cudf: CudfEnvironment,
    cugraph: CugraphEnvironment,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CudfMetadata {
    runtime_lib_dir: Option<PathBuf>,
    libcudf_runtime_lib: Option<PathBuf>,
    rmm_runtime_lib: Option<PathBuf>,
    rapids_logger_runtime_lib: Option<PathBuf>,
    rapids_logger_linkage: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CugraphMetadata {
    runtime_lib_dirs: Option<Vec<PathBuf>>,
    native_link_dirs: Option<Vec<PathBuf>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ResolvedMetadata {
    cudf: CudfMetadata,
    cugraph: CugraphMetadata,
}

fn read_environment(cugraph_enabled: bool) -> EnvironmentSnapshot {
    let cudf = CudfEnvironment {
        aggregate_metadata: std::env::var(CUDF_METADATA_ENV).ok(),
        runtime_lib_dir: std::env::var(CUDF_RUNTIME_LIB_DIR_ENV).ok(),
        libcudf_runtime_lib: std::env::var(CUDF_LIBCUDF_RUNTIME_LIB_ENV).ok(),
        rmm_runtime_lib: std::env::var(CUDF_RMM_RUNTIME_LIB_ENV).ok(),
        rapids_logger_runtime_lib: std::env::var(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV).ok(),
        rapids_logger_linkage: cugraph_enabled.then(|| std::env::var(CUDF_RAPIDS_LOGGER_LINKAGE_ENV).ok()).flatten(),
    };
    let cugraph = if cugraph_enabled {
        CugraphEnvironment {
            aggregate_metadata: std::env::var(CUGRAPH_METADATA_ENV).ok(),
            runtime_lib_dirs: std::env::var(CUGRAPH_RUNTIME_LIB_DIRS_ENV).ok(),
            native_link_dirs: std::env::var(CUGRAPH_NATIVE_LINK_DIRS_ENV).ok(),
        }
    } else {
        CugraphEnvironment::default()
    };
    EnvironmentSnapshot { cudf, cugraph }
}

fn resolve_metadata(environment: &EnvironmentSnapshot) -> ResolvedMetadata {
    let cudf = &environment.cudf;
    let cugraph = &environment.cugraph;
    ResolvedMetadata {
        cudf: CudfMetadata {
            runtime_lib_dir: select_metadata_value(
                cudf.runtime_lib_dir.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "runtime-lib-dir",
            )
            .map(PathBuf::from),
            libcudf_runtime_lib: select_metadata_value(
                cudf.libcudf_runtime_lib.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "libcudf-runtime-lib",
            )
            .map(PathBuf::from),
            rmm_runtime_lib: select_metadata_value(
                cudf.rmm_runtime_lib.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "rmm-runtime-lib",
            )
            .map(PathBuf::from),
            rapids_logger_runtime_lib: select_metadata_value(
                cudf.rapids_logger_runtime_lib.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "rapids-logger-runtime-lib",
            )
            .map(PathBuf::from),
            rapids_logger_linkage: select_metadata_value(
                cudf.rapids_logger_linkage.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "rapids-logger-linkage",
            ),
        },
        cugraph: CugraphMetadata {
            runtime_lib_dirs: select_metadata_value(
                cugraph.runtime_lib_dirs.as_deref(),
                cugraph.aggregate_metadata.as_deref(),
                "runtime-lib-dirs",
            )
            .map(|value| parse_platform_path_list(OsStr::new(&value))),
            native_link_dirs: select_metadata_value(
                cugraph.native_link_dirs.as_deref(),
                cugraph.aggregate_metadata.as_deref(),
                "native-link-dirs",
            )
            .map(|value| parse_platform_path_list(OsStr::new(&value))),
        },
    }
}

fn select_metadata_value(direct_value: Option<&str>, metadata: Option<&str>, key: &str) -> Option<String> {
    direct_value.map(str::to_owned).or_else(|| extract_metadata_value(metadata, key))
}

fn extract_metadata_value(metadata: Option<&str>, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    metadata?.split(';').find_map(|entry| entry.strip_prefix(&prefix).map(str::to_owned))
}

fn parse_platform_path_list(encoded: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(encoded).collect()
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}

fn runtime_directories(metadata: &ResolvedMetadata, cugraph_enabled: bool) -> Vec<PathBuf> {
    let mut runtime_dirs = Vec::new();
    if let Some(dir) = &metadata.cudf.runtime_lib_dir {
        push_unique_path(&mut runtime_dirs, dir.clone());
    }
    for library in [
        metadata.cudf.libcudf_runtime_lib.as_ref(),
        metadata.cudf.rmm_runtime_lib.as_ref(),
        metadata.cudf.rapids_logger_runtime_lib.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(parent) = library.parent() {
            push_unique_path(&mut runtime_dirs, parent.to_path_buf());
        }
    }
    if cugraph_enabled {
        for dir in metadata.cugraph.runtime_lib_dirs.iter().flatten() {
            push_unique_path(&mut runtime_dirs, dir.clone());
        }
    }
    runtime_dirs
}

fn link_directives(metadata: &ResolvedMetadata, cugraph_enabled: bool, targets: LinkArgTargets) -> Vec<String> {
    let mut directives = Vec::new();
    for dir in runtime_directories(metadata, cugraph_enabled) {
        directives.extend(targets.rpath_directives(&dir));
    }
    for library in [
        metadata.cudf.libcudf_runtime_lib.as_ref(),
        metadata.cudf.rmm_runtime_lib.as_ref(),
        metadata.cudf.rapids_logger_runtime_lib.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        directives.extend(targets.required_runtime_library_directives(library));
    }
    directives
}

fn emit_cargo_directives(directives: impl IntoIterator<Item = String>) {
    for directive in directives {
        println!("{directive}");
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

pub fn configure(config: Config) {
    let cugraph_enabled = matches!(config.cugraph_runtime, CugraphRuntime::EmitRpathAndRequireCudfIdentity);
    let environment = read_environment(cugraph_enabled);
    let metadata = resolve_metadata(&environment);

    println!("cargo:rerun-if-changed={}", config.support_source);
    track_env(CUDF_METADATA_ENV);
    track_env(CUDF_RUNTIME_LIB_DIR_ENV);
    track_env(CUDF_LIBCUDF_RUNTIME_LIB_ENV);
    track_env(CUDF_RMM_RUNTIME_LIB_ENV);
    track_env(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV);

    if metadata.cudf.runtime_lib_dir.is_none() {
        println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake");
    }
    if cugraph_enabled {
        track_cugraph_env();
        if metadata.cugraph.runtime_lib_dirs.is_none() {
            println!("cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_*; skipping rpath bake");
        }
    }

    emit_cargo_directives(link_directives(&metadata, cugraph_enabled, config.link_arg_targets));
    if metadata.cudf.libcudf_runtime_lib.is_none() {
        println!(
            "cargo:warning=cudf-sys did not expose its authoritative libcudf.so; \
             transitive native resolution may require an ambient loader path"
        );
    }
    if metadata.cudf.rmm_runtime_lib.is_none() {
        println!(
            "cargo:warning=cudf-sys did not expose its authoritative librmm.so; \
             transitive native resolution may require an ambient loader path"
        );
    }

    if cugraph_enabled {
        enforce_single_rapids_runtime_libraries(&metadata);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OptionalLoggerPolicy {
    Enforce(PathBuf),
    Skip { linkage: String },
}

fn optional_logger_policy(library: Option<&Path>, linkage: Option<&str>) -> OptionalLoggerPolicy {
    match library {
        Some(library) => OptionalLoggerPolicy::Enforce(library.to_path_buf()),
        None => OptionalLoggerPolicy::Skip { linkage: linkage.unwrap_or("unknown").to_string() },
    }
}

fn missing_required_library_diagnostic(lib_name: &str, cudf_dep_env: &str) -> String {
    format!(
        "the `cugraph` feature requires libcudf to ship a shared {lib_name}, but cudf-sys did not expose \
         one via {cudf_dep_env}. Rebuild libcudf with CUDF_RMM_BUILD_SHARED=ON (the cudf fork default), \
         re-run `cmake --install`, and rebuild cudf-sys so the install prefix exports a single canonical \
         {lib_name} for every RAPIDS consumer in the process."
    )
}

fn optional_logger_warning(linkage: &str) -> String {
    format!(
        "cargo:warning=cudf-sys did not expose a shared librapids_logger.so (linkage={linkage}); \
         skipping cross-crate rapids_logger identity enforcement"
    )
}

fn missing_candidate_diagnostic(lib_name: &str, cudf_lib: &Path) -> String {
    format!(
        "the `cugraph` feature is enabled but no {lib_name} is reachable from the cugraph-sys link/runtime \
         dirs (DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS), so the build cannot prove \
         libcugraph links the same {lib_name} as libcudf ({}). Reconfigure the cugraph C++ build against \
         the cudf install prefix (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>) and \
         rebuild it.",
        cudf_lib.display()
    )
}

fn divergent_library_diagnostic(lib_name: &str, cudf_lib: &Path, cugraph_lib: &Path) -> String {
    format!(
        "two divergent {lib_name} builds are linked into this process:\n  cudf-sys:    {}\n  cugraph-sys: {}\n\
         One process must bind every {lib_name} symbol to a single build, or execution corrupts at runtime. \
         Reconfigure the cugraph C++ build against the cudf install prefix \
         (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>), rebuild it, and remove stale \
         copies from the cugraph build tree.",
        cudf_lib.display(),
        cugraph_lib.display()
    )
}

fn enforce_single_rapids_runtime_libraries(metadata: &ResolvedMetadata) {
    track_env(CUDF_RAPIDS_LOGGER_LINKAGE_ENV);

    let rmm =
        metadata.cudf.rmm_runtime_lib.as_deref().unwrap_or_else(|| {
            panic!("{}", missing_required_library_diagnostic("librmm.so", CUDF_RMM_RUNTIME_LIB_ENV))
        });
    enforce_shared_rapids_library_matches_cudf("librmm.so", rmm, &metadata.cugraph);

    match optional_logger_policy(
        metadata.cudf.rapids_logger_runtime_lib.as_deref(),
        metadata.cudf.rapids_logger_linkage.as_deref(),
    ) {
        OptionalLoggerPolicy::Enforce(library) => {
            enforce_shared_rapids_library_matches_cudf("librapids_logger.so", &library, &metadata.cugraph);
        },
        OptionalLoggerPolicy::Skip { linkage } => println!("{}", optional_logger_warning(&linkage)),
    }
}

/// Produce candidate paths without reading the filesystem. Native link dirs
/// precede runtime dirs, matching the historical build-script policy.
fn cugraph_candidate_paths(lib_name: &str, metadata: &CugraphMetadata) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for dirs in [&metadata.native_link_dirs, &metadata.runtime_lib_dirs].into_iter().flatten() {
        for dir in dirs {
            push_unique_path(&mut candidates, dir.join(lib_name));
        }
    }
    candidates
}

fn existing_candidates(candidate_paths: Vec<PathBuf>) -> Vec<PathBuf> {
    candidate_paths.into_iter().filter(|candidate| candidate.is_file()).collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LibraryIdentityEvidence {
    canonical_path_equal: bool,
    same_inode: bool,
    byte_identical: bool,
}

fn identity_policy(evidence: LibraryIdentityEvidence) -> bool {
    evidence.canonical_path_equal || evidence.same_inode || evidence.byte_identical
}

fn observe_library_identity(left: &Path, right: &Path) -> LibraryIdentityEvidence {
    let canonical_path_equal = matches!(
        (left.canonicalize(), right.canonicalize()),
        (Ok(left_canonical), Ok(right_canonical)) if left_canonical == right_canonical
    );
    if canonical_path_equal {
        return LibraryIdentityEvidence { canonical_path_equal, ..LibraryIdentityEvidence::default() };
    }

    #[cfg(unix)]
    let same_inode = {
        use std::os::unix::fs::MetadataExt;
        matches!(
            (std::fs::metadata(left), std::fs::metadata(right)),
            (Ok(left_meta), Ok(right_meta))
                if left_meta.dev() == right_meta.dev() && left_meta.ino() == right_meta.ino()
        )
    };
    #[cfg(not(unix))]
    let same_inode = false;
    if same_inode {
        return LibraryIdentityEvidence { same_inode, ..LibraryIdentityEvidence::default() };
    }

    let byte_identical = matches!(
        (std::fs::read(left), std::fs::read(right)),
        (Ok(left_bytes), Ok(right_bytes)) if left_bytes == right_bytes
    );
    LibraryIdentityEvidence { byte_identical, ..LibraryIdentityEvidence::default() }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CandidateIdentity {
    path: PathBuf,
    same_build: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CandidateResolution {
    Missing,
    Consistent { selected: PathBuf, candidates: Vec<PathBuf> },
    Divergent { selected: PathBuf, divergent: PathBuf },
}

fn resolve_candidate_identities(candidates: &[CandidateIdentity]) -> CandidateResolution {
    let Some(first) = candidates.first() else {
        return CandidateResolution::Missing;
    };
    if let Some(divergent) = candidates.iter().find(|candidate| !candidate.same_build) {
        return CandidateResolution::Divergent { selected: first.path.clone(), divergent: divergent.path.clone() };
    }
    CandidateResolution::Consistent {
        selected: first.path.clone(),
        candidates: candidates.iter().map(|candidate| candidate.path.clone()).collect(),
    }
}

fn enforce_shared_rapids_library_matches_cudf(lib_name: &str, cudf_lib: &Path, cugraph_metadata: &CugraphMetadata) {
    println!("cargo:rerun-if-changed={}", cudf_lib.display());
    let candidate_paths = existing_candidates(cugraph_candidate_paths(lib_name, cugraph_metadata));
    let candidates = candidate_paths
        .iter()
        .map(|candidate| CandidateIdentity {
            path: candidate.clone(),
            same_build: identity_policy(observe_library_identity(cudf_lib, candidate)),
        })
        .collect::<Vec<_>>();
    for candidate in &candidate_paths {
        println!("cargo:rerun-if-changed={}", candidate.display());
    }
    match resolve_candidate_identities(&candidates) {
        CandidateResolution::Missing => {
            panic!("{}", missing_candidate_diagnostic(lib_name, cudf_lib))
        },
        CandidateResolution::Consistent { .. } => {},
        CandidateResolution::Divergent { divergent, .. } => {
            panic!("{}", divergent_library_diagnostic(lib_name, cudf_lib, &divergent))
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(1);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("datafusion-nexus-rapids-build-support-{label}-{}-{sequence}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn metadata_resolution_merges_direct_and_aggregate_values_table() {
        struct Case {
            name: &'static str,
            environment: CudfEnvironment,
            expected: CudfMetadata,
        }
        let aggregate = Some(
            "runtime-lib-dir=/aggregate;libcudf-runtime-lib=/aggregate/libcudf.so;\
             rmm-runtime-lib=/aggregate/librmm.so;rapids-logger-runtime-lib=/aggregate/librapids_logger.so;\
             rapids-logger-linkage=static"
                .to_string(),
        );
        let cases = [
            Case {
                name: "aggregate fallback",
                environment: CudfEnvironment { aggregate_metadata: aggregate.clone(), ..CudfEnvironment::default() },
                expected: CudfMetadata {
                    runtime_lib_dir: Some("/aggregate".into()),
                    libcudf_runtime_lib: Some("/aggregate/libcudf.so".into()),
                    rmm_runtime_lib: Some("/aggregate/librmm.so".into()),
                    rapids_logger_runtime_lib: Some("/aggregate/librapids_logger.so".into()),
                    rapids_logger_linkage: Some("static".to_string()),
                },
            },
            Case {
                name: "direct fields override independently",
                environment: CudfEnvironment {
                    aggregate_metadata: aggregate,
                    runtime_lib_dir: Some("/direct".to_string()),
                    libcudf_runtime_lib: Some("/direct/libcudf.so".to_string()),
                    rmm_runtime_lib: None,
                    rapids_logger_runtime_lib: Some("/direct/librapids_logger.so".to_string()),
                    rapids_logger_linkage: Some("shared".to_string()),
                },
                expected: CudfMetadata {
                    runtime_lib_dir: Some("/direct".into()),
                    libcudf_runtime_lib: Some("/direct/libcudf.so".into()),
                    rmm_runtime_lib: Some("/aggregate/librmm.so".into()),
                    rapids_logger_runtime_lib: Some("/direct/librapids_logger.so".into()),
                    rapids_logger_linkage: Some("shared".to_string()),
                },
            },
            Case { name: "missing", environment: CudfEnvironment::default(), expected: CudfMetadata::default() },
        ];
        for case in cases {
            let resolved = resolve_metadata(&EnvironmentSnapshot { cudf: case.environment, ..Default::default() });
            assert_eq!(resolved.cudf, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn aggregate_metadata_uses_first_exact_key_and_not_partial_names() {
        assert_eq!(
            select_metadata_value(
                None,
                Some("not-rmm-runtime-lib=wrong;rmm-runtime-lib=first;rmm-runtime-lib=second"),
                "rmm-runtime-lib"
            ),
            Some("first".to_string())
        );
        assert_eq!(select_metadata_value(None, Some("other=value"), "rmm-runtime-lib"), None);
    }

    #[test]
    fn platform_path_lists_and_direct_cugraph_precedence_are_preserved() {
        let paths = [PathBuf::from("first path"), PathBuf::from("second")];
        let encoded = std::env::join_paths(&paths).unwrap();
        assert_eq!(parse_platform_path_list(&encoded), paths);

        let resolved = resolve_metadata(&EnvironmentSnapshot {
            cugraph: CugraphEnvironment {
                aggregate_metadata: Some(format!(
                    "runtime-lib-dirs={};native-link-dirs={}",
                    std::env::join_paths(["aggregate-runtime"]).unwrap().to_string_lossy(),
                    std::env::join_paths(["aggregate-native"]).unwrap().to_string_lossy()
                )),
                runtime_lib_dirs: Some(encoded.to_string_lossy().into_owned()),
                native_link_dirs: None,
            },
            ..Default::default()
        });
        assert_eq!(resolved.cugraph.runtime_lib_dirs, Some(paths.to_vec()));
        assert_eq!(resolved.cugraph.native_link_dirs, Some(vec!["aggregate-native".into()]));
    }

    #[test]
    fn link_directives_preserve_target_and_runtime_library_order() {
        let metadata = ResolvedMetadata {
            cudf: CudfMetadata {
                runtime_lib_dir: Some("/runtime".into()),
                libcudf_runtime_lib: Some("/cudf/libcudf.so".into()),
                rmm_runtime_lib: Some("/rmm/librmm.so".into()),
                rapids_logger_runtime_lib: None,
                rapids_logger_linkage: None,
            },
            cugraph: CugraphMetadata { runtime_lib_dirs: Some(vec!["/graph".into()]), native_link_dirs: None },
        };
        let directives = link_directives(&metadata, true, LinkArgTargets::new(true, false, true));
        assert_eq!(directives[0], "cargo:rustc-link-arg=-Wl,-rpath,/runtime");
        assert_eq!(directives[1], "cargo:rustc-link-arg-tests=-Wl,-rpath,/runtime");
        assert!(directives.iter().any(|line| line.contains("/cudf/libcudf.so")));
        assert!(directives.iter().any(|line| line.contains("/rmm/librmm.so")));
        assert!(!directives.iter().any(|line| line.starts_with("cargo:rustc-link-arg-bins=")));
    }

    #[test]
    fn missing_required_rmm_and_optional_logger_policies_are_typed_and_stable() {
        assert_eq!(
            missing_required_library_diagnostic("librmm.so", CUDF_RMM_RUNTIME_LIB_ENV),
            "the `cugraph` feature requires libcudf to ship a shared librmm.so, but cudf-sys did not expose one via DEP_CUDF_RUST_RMM_RUNTIME_LIB. Rebuild libcudf with CUDF_RMM_BUILD_SHARED=ON (the cudf fork default), re-run `cmake --install`, and rebuild cudf-sys so the install prefix exports a single canonical librmm.so for every RAPIDS consumer in the process."
        );
        assert_eq!(
            optional_logger_policy(None, Some("static")),
            OptionalLoggerPolicy::Skip { linkage: "static".to_string() }
        );
        assert_eq!(
            optional_logger_warning("static"),
            "cargo:warning=cudf-sys did not expose a shared librapids_logger.so (linkage=static); skipping cross-crate rapids_logger identity enforcement"
        );
        assert_eq!(
            optional_logger_policy(Some(Path::new("/lib/librapids_logger.so")), None),
            OptionalLoggerPolicy::Enforce("/lib/librapids_logger.so".into())
        );
    }

    #[test]
    fn candidate_paths_are_deterministic_and_exactly_deduplicated() {
        let metadata = CugraphMetadata {
            native_link_dirs: Some(vec!["/z".into(), "/a".into()]),
            runtime_lib_dirs: Some(vec!["/a".into(), "/b".into()]),
        };
        assert_eq!(
            cugraph_candidate_paths("librmm.so", &metadata),
            vec![PathBuf::from("/z/librmm.so"), PathBuf::from("/a/librmm.so"), PathBuf::from("/b/librmm.so"),]
        );
    }

    #[test]
    fn multiple_candidate_resolution_selects_first_or_reports_ambiguity_deterministically() {
        let consistent = [
            CandidateIdentity { path: "/z/librmm.so".into(), same_build: true },
            CandidateIdentity { path: "/a/librmm.so".into(), same_build: true },
        ];
        assert_eq!(
            resolve_candidate_identities(&consistent),
            CandidateResolution::Consistent {
                selected: "/z/librmm.so".into(),
                candidates: vec!["/z/librmm.so".into(), "/a/librmm.so".into()]
            }
        );
        let ambiguous = [
            CandidateIdentity { path: "/z/librmm.so".into(), same_build: true },
            CandidateIdentity { path: "/a/librmm.so".into(), same_build: false },
            CandidateIdentity { path: "/b/librmm.so".into(), same_build: false },
        ];
        assert_eq!(
            resolve_candidate_identities(&ambiguous),
            CandidateResolution::Divergent { selected: "/z/librmm.so".into(), divergent: "/a/librmm.so".into() }
        );
        assert_eq!(resolve_candidate_identities(&[]), CandidateResolution::Missing);
    }

    #[test]
    fn identity_policy_accepts_canonical_same_path_same_inode_and_identical_bytes() {
        let test_dir = TestDir::new("identity");
        let first = test_dir.path().join("first.so");
        let identical = test_dir.path().join("identical.so");
        let different = test_dir.path().join("different.so");
        std::fs::write(&first, b"same build").unwrap();
        std::fs::write(&identical, b"same build").unwrap();
        std::fs::write(&different, b"different build").unwrap();

        let same_path = observe_library_identity(&first, &first);
        assert!(same_path.canonical_path_equal);
        assert!(identity_policy(same_path));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let symlink_path = test_dir.path().join("symlink.so");
            symlink(&first, &symlink_path).unwrap();
            let same_canonical_path = observe_library_identity(&first, &symlink_path);
            assert!(same_canonical_path.canonical_path_equal);
            assert!(identity_policy(same_canonical_path));

            let hard_link = test_dir.path().join("hard-link.so");
            std::fs::hard_link(&first, &hard_link).unwrap();
            let same_inode = observe_library_identity(&first, &hard_link);
            assert!(same_inode.same_inode);
            assert!(identity_policy(same_inode));
        }

        let same_bytes = observe_library_identity(&first, &identical);
        assert!(same_bytes.byte_identical);
        assert!(identity_policy(same_bytes));
        assert!(!identity_policy(observe_library_identity(&first, &different)));
    }

    #[test]
    fn divergent_and_missing_candidate_diagnostics_are_stable() {
        assert_eq!(
            missing_candidate_diagnostic("librmm.so", Path::new("/cudf/librmm.so")),
            "the `cugraph` feature is enabled but no librmm.so is reachable from the cugraph-sys link/runtime dirs (DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS), so the build cannot prove libcugraph links the same librmm.so as libcudf (/cudf/librmm.so). Reconfigure the cugraph C++ build against the cudf install prefix (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>) and rebuild it."
        );
        assert_eq!(
            divergent_library_diagnostic("librmm.so", Path::new("/cudf/librmm.so"), Path::new("/graph/librmm.so")),
            "two divergent librmm.so builds are linked into this process:\n  cudf-sys:    /cudf/librmm.so\n  cugraph-sys: /graph/librmm.so\nOne process must bind every librmm.so symbol to a single build, or execution corrupts at runtime. Reconfigure the cugraph C++ build against the cudf install prefix (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>), rebuild it, and remove stale copies from the cugraph build tree."
        );
    }
}
