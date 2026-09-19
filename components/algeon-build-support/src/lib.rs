//! Shared Cargo build-script support for RAPIDS runtime libraries.
//!
//! Every Algeon package whose build script links RAPIDS native libraries depends
//! on this crate as a build-dependency, so the published `.crate` archives stay
//! self-contained and one implementation owns the policy.
//!
//! Environment capture, pure metadata/candidate/identity policy, filesystem
//! observation, and Cargo stdout emission are kept as separate stages so the
//! safety policy can be tested without real native libraries.
//!
//! std-only by contract: it runs before any dependency is built.

use std::{
    ffi::{OsStr, OsString},
    fmt,
    path::{Path, PathBuf},
};

const DOCS_RS_ENV: &str = "DOCS_RS";
const ALGEON_NATIVE_DIR_ENV: &str = "ALGEON_NATIVE_DIR";

/// Native artifact contract owned by one sys crate.
pub struct NativeComponent {
    pub crate_name: &'static str,
    pub install_dir_name: &'static str,
    pub required_library: &'static str,
    pub override_env: Option<&'static str>,
    pub source_required_env: Option<&'static str>,
    pub source_tree_sentinel: Option<&'static str>,
    pub build_script: &'static str,
}

/// How a sys crate must obtain its native library for the current build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeLinking {
    Skip,
    Prebuilt(PathBuf),
    SourceTree,
    Missing(MissingNative),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeSearch {
    candidate: String,
    state: String,
}

/// Actionable native-discovery failure returned to a sys crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingNative {
    crate_name: &'static str,
    required_library: &'static str,
    build_script: &'static str,
    searches: Vec<NativeSearch>,
}

impl fmt::Display for MissingNative {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let library = Path::new(self.required_library)
            .file_name()
            .unwrap_or_else(|| OsStr::new(self.required_library))
            .to_string_lossy();
        writeln!(formatter, "{}: no usable {library}.", self.crate_name)?;
        writeln!(formatter)?;
        writeln!(formatter, "Algeon links against RAPIDS native libraries you build from the repository:")?;
        writeln!(formatter)?;
        writeln!(formatter, "  1. git clone --recursive https://github.com/algeon-dev/algeon")?;
        writeln!(formatter, "  2. bash {}", self.build_script)?;
        writeln!(formatter, "  3. export ALGEON_NATIVE_DIR=<repo>/target/native")?;
        writeln!(formatter)?;
        writeln!(formatter, "Searched:")?;
        for search in &self.searches {
            writeln!(formatter, "  {:<68} ({})", search.candidate, search.state)?;
        }
        write!(formatter, "\nDocs: https://algeon.dev/docs/build/from-source")
    }
}

fn missing_search(component: &NativeComponent, searches: Vec<NativeSearch>) -> NativeLinking {
    NativeLinking::Missing(MissingNative {
        crate_name: component.crate_name,
        required_library: component.required_library,
        build_script: component.build_script,
        searches,
    })
}

/// Resolve docs.rs, explicit-prefix, repository-source, and workspace-prefix
/// builds in their declared precedence order.
pub fn resolve_native_linking(
    component: &NativeComponent,
    manifest_dir: &Path,
    env: &dyn Fn(&str) -> Option<OsString>,
) -> NativeLinking {
    if env(DOCS_RS_ENV).is_some() {
        return NativeLinking::Skip;
    }

    if component.source_required_env.is_some_and(|name| env(name).is_some())
        && component.source_tree_sentinel.is_some_and(|sentinel| manifest_dir.join(sentinel).is_file())
    {
        return NativeLinking::SourceTree;
    }

    let mut searches = Vec::new();
    if let Some(name) = component.override_env {
        match env(name) {
            Some(value) => {
                let prefix = PathBuf::from(value);
                let required = prefix.join(component.required_library);
                if required.is_file() {
                    return NativeLinking::Prebuilt(prefix);
                }
                searches.push(NativeSearch {
                    candidate: format!("${name}/{}", component.required_library),
                    state: format!("{name} set to {}, but library not found", prefix.display()),
                });
                return missing_search(component, searches);
            },
            None => searches.push(NativeSearch {
                candidate: format!("${name}/{}", component.required_library),
                state: format!("{name} unset"),
            }),
        }
    }

    match env(ALGEON_NATIVE_DIR_ENV) {
        Some(value) => {
            let native_root = PathBuf::from(value);
            let prefix = native_root.join(component.install_dir_name);
            let required = prefix.join(component.required_library);
            if required.is_file() {
                return NativeLinking::Prebuilt(prefix);
            }
            searches.push(NativeSearch {
                candidate: format!("$ALGEON_NATIVE_DIR/{}/{}", component.install_dir_name, component.required_library),
                state: format!("ALGEON_NATIVE_DIR set to {}, but library not found", native_root.display()),
            });
            return missing_search(component, searches);
        },
        None => searches.push(NativeSearch {
            candidate: format!("$ALGEON_NATIVE_DIR/{}/{}", component.install_dir_name, component.required_library),
            state: "ALGEON_NATIVE_DIR unset".to_string(),
        }),
    }

    if let Some(sentinel) = component.source_tree_sentinel {
        let sentinel_path = manifest_dir.join(sentinel);
        if sentinel_path.is_file() {
            return NativeLinking::SourceTree;
        }
        searches.push(NativeSearch {
            candidate: format!("<manifest>/{sentinel}"),
            state: "source tree not present".to_string(),
        });
    }

    let workspace_prefix = manifest_dir.join("../../..").join("target/native").join(component.install_dir_name);
    let required = workspace_prefix.join(component.required_library);
    if required.is_file() {
        return NativeLinking::Prebuilt(workspace_prefix);
    }
    searches.push(NativeSearch {
        candidate: format!(
            "<manifest>/../../../target/native/{}/{}",
            component.install_dir_name, component.required_library
        ),
        state: "not found".to_string(),
    });
    missing_search(component, searches)
}

/// Verify that an installed native prefix was produced for the consuming
/// crate version.
pub fn verify_prebuilt_crate_version(
    component: &NativeComponent,
    prefix: &Path,
    marker_relative: &str,
    expected: &str,
) -> Result<(), String> {
    let marker = prefix.join(marker_relative);
    let installed = std::fs::read_to_string(&marker).map_err(|error| {
        format!(
            "{} {expected}: the prebuilt native artifacts under {} have no readable {marker_relative}: {error}.\n\nRebuild them from the matching tag:\n  git -C <repo> checkout v{expected} && bash {}",
            component.crate_name,
            prefix.display(),
            component.build_script
        )
    })?;
    let installed = installed.trim();
    if installed != expected {
        return Err(format!(
            "{} {expected}: the prebuilt native artifacts under {} were built for crate version {installed} ({marker_relative}).\n\nRebuild them from the matching tag:\n  git -C <repo> checkout v{expected} && bash {}",
            component.crate_name,
            prefix.display(),
            component.build_script
        ));
    }
    Ok(())
}

const CUDF_METADATA_ENV: &str = "DEP_CUDF_RUST_METADATA";
const CUDF_RUNTIME_LIB_DIR_ENV: &str = "DEP_CUDF_RUST_RUNTIME_LIB_DIR";
const CUDF_RUST_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_CUDF_RUST_RUNTIME_LIB";
const CUDF_LIBCUDF_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_LIBCUDF_RUNTIME_LIB";
const CUDF_RMM_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_RMM_RUNTIME_LIB";
const CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV: &str = "DEP_CUDF_RUST_RAPIDS_LOGGER_RUNTIME_LIB";
const CUDF_RAPIDS_LOGGER_LINKAGE_ENV: &str = "DEP_CUDF_RUST_RAPIDS_LOGGER_LINKAGE";

const CUGRAPH_METADATA_ENV: &str = "DEP_CUGRAPH_METADATA";
const CUGRAPH_RUNTIME_LIB_DIRS_ENV: &str = "DEP_CUGRAPH_RUNTIME_LIB_DIRS";
const CUGRAPH_NATIVE_LINK_DIRS_ENV: &str = "DEP_CUGRAPH_NATIVE_LINK_DIRS";
const CUGRAPH_CUVS_LIBRARY_PATH_ENV: &str = "DEP_CUGRAPH_CUVS_LIBRARY_PATH";
const CUGRAPH_CUVS_RUNTIME_LIB_DIR_ENV: &str = "DEP_CUGRAPH_CUVS_RUNTIME_LIB_DIR";

const CUVS_METADATA_ENV: &str = "DEP_CUVS_C_METADATA";
const CUVS_RUNTIME_LIB_DIRS_ENV: &str = "DEP_CUVS_C_RUNTIME_LIB_DIRS";
const CUVS_NATIVE_LINK_DIRS_ENV: &str = "DEP_CUVS_C_NATIVE_LINK_DIRS";
const CUVS_LIBCUVS_RUNTIME_LIB_ENV: &str = "DEP_CUVS_C_LIBCUVS_RUNTIME_LIB";
const CUVS_LIBCUVS_C_RUNTIME_LIB_ENV: &str = "DEP_CUVS_C_LIBCUVS_C_RUNTIME_LIB";

const CUDA_HOME_ENV: &str = "CUDA_HOME";
const CUDA_PATH_ENV: &str = "CUDA_PATH";
const CUDA_TOOLKIT_ROOT_ENV: &str = "CUDAToolkit_ROOT";
const NVTX3INTEROP_LIBRARY: &str = "libnvtx3interop.so";
const CANONICAL_CUDA_ROOT: &str = "/usr/local/cuda";

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RapidsConsumers {
    cugraph: bool,
    cuvs: bool,
}

impl RapidsConsumers {
    pub const fn new(cugraph: bool, cuvs: bool) -> Self {
        Self { cugraph, cuvs }
    }

    const fn any(self) -> bool {
        self.cugraph || self.cuvs
    }
}

pub struct Config {
    pub link_arg_targets: LinkArgTargets,
    pub consumers: RapidsConsumers,
    /// Whether this package's native execution code emits NVTX annotations.
    pub nvtx3interop: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CudaEnvironment {
    cuda_home: Option<PathBuf>,
    cuda_path: Option<PathBuf>,
    cuda_toolkit_root: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CudfEnvironment {
    aggregate_metadata: Option<String>,
    runtime_lib_dir: Option<String>,
    cudf_rust_runtime_lib: Option<String>,
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
    cuvs_library_path: Option<String>,
    cuvs_runtime_lib_dir: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CuvsEnvironment {
    aggregate_metadata: Option<String>,
    runtime_lib_dirs: Option<String>,
    native_link_dirs: Option<String>,
    libcuvs_runtime_lib: Option<String>,
    libcuvs_c_runtime_lib: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct EnvironmentSnapshot {
    cudf: CudfEnvironment,
    cugraph: CugraphEnvironment,
    cuvs: CuvsEnvironment,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CudfMetadata {
    runtime_lib_dir: Option<PathBuf>,
    cudf_rust_runtime_lib: Option<PathBuf>,
    libcudf_runtime_lib: Option<PathBuf>,
    rmm_runtime_lib: Option<PathBuf>,
    rapids_logger_runtime_lib: Option<PathBuf>,
    rapids_logger_linkage: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CugraphMetadata {
    runtime_lib_dirs: Option<Vec<PathBuf>>,
    native_link_dirs: Option<Vec<PathBuf>>,
    cuvs_library_path: Option<PathBuf>,
    cuvs_runtime_lib_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CuvsMetadata {
    runtime_lib_dirs: Option<Vec<PathBuf>>,
    native_link_dirs: Option<Vec<PathBuf>>,
    libcuvs_runtime_lib: Option<PathBuf>,
    libcuvs_c_runtime_lib: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ResolvedMetadata {
    cudf: CudfMetadata,
    cugraph: CugraphMetadata,
    cuvs: CuvsMetadata,
}

fn read_environment(consumers: RapidsConsumers) -> EnvironmentSnapshot {
    let cudf = CudfEnvironment {
        aggregate_metadata: std::env::var(CUDF_METADATA_ENV).ok(),
        runtime_lib_dir: std::env::var(CUDF_RUNTIME_LIB_DIR_ENV).ok(),
        cudf_rust_runtime_lib: std::env::var(CUDF_RUST_RUNTIME_LIB_ENV).ok(),
        libcudf_runtime_lib: std::env::var(CUDF_LIBCUDF_RUNTIME_LIB_ENV).ok(),
        rmm_runtime_lib: std::env::var(CUDF_RMM_RUNTIME_LIB_ENV).ok(),
        rapids_logger_runtime_lib: std::env::var(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV).ok(),
        rapids_logger_linkage: consumers.any().then(|| std::env::var(CUDF_RAPIDS_LOGGER_LINKAGE_ENV).ok()).flatten(),
    };
    let cugraph = if consumers.cugraph {
        CugraphEnvironment {
            aggregate_metadata: std::env::var(CUGRAPH_METADATA_ENV).ok(),
            runtime_lib_dirs: std::env::var(CUGRAPH_RUNTIME_LIB_DIRS_ENV).ok(),
            native_link_dirs: std::env::var(CUGRAPH_NATIVE_LINK_DIRS_ENV).ok(),
            cuvs_library_path: std::env::var(CUGRAPH_CUVS_LIBRARY_PATH_ENV).ok(),
            cuvs_runtime_lib_dir: std::env::var(CUGRAPH_CUVS_RUNTIME_LIB_DIR_ENV).ok(),
        }
    } else {
        CugraphEnvironment::default()
    };
    let cuvs = if consumers.cuvs {
        CuvsEnvironment {
            aggregate_metadata: std::env::var(CUVS_METADATA_ENV).ok(),
            runtime_lib_dirs: std::env::var(CUVS_RUNTIME_LIB_DIRS_ENV).ok(),
            native_link_dirs: std::env::var(CUVS_NATIVE_LINK_DIRS_ENV).ok(),
            libcuvs_runtime_lib: std::env::var(CUVS_LIBCUVS_RUNTIME_LIB_ENV).ok(),
            libcuvs_c_runtime_lib: std::env::var(CUVS_LIBCUVS_C_RUNTIME_LIB_ENV).ok(),
        }
    } else {
        CuvsEnvironment::default()
    };
    EnvironmentSnapshot { cudf, cugraph, cuvs }
}

fn resolve_metadata(environment: &EnvironmentSnapshot) -> ResolvedMetadata {
    let cudf = &environment.cudf;
    let cugraph = &environment.cugraph;
    let cuvs = &environment.cuvs;
    ResolvedMetadata {
        cudf: CudfMetadata {
            runtime_lib_dir: select_metadata_value(
                cudf.runtime_lib_dir.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "runtime-lib-dir",
            )
            .map(PathBuf::from),
            cudf_rust_runtime_lib: select_metadata_value(
                cudf.cudf_rust_runtime_lib.as_deref(),
                cudf.aggregate_metadata.as_deref(),
                "cudf-rust-runtime-lib",
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
            cuvs_library_path: select_metadata_value(
                cugraph.cuvs_library_path.as_deref(),
                cugraph.aggregate_metadata.as_deref(),
                "cuvs-library-path",
            )
            .map(PathBuf::from),
            cuvs_runtime_lib_dir: select_metadata_value(
                cugraph.cuvs_runtime_lib_dir.as_deref(),
                cugraph.aggregate_metadata.as_deref(),
                "cuvs-runtime-lib-dir",
            )
            .map(PathBuf::from),
        },
        cuvs: CuvsMetadata {
            runtime_lib_dirs: select_metadata_value(
                cuvs.runtime_lib_dirs.as_deref(),
                cuvs.aggregate_metadata.as_deref(),
                "runtime-lib-dirs",
            )
            .map(|value| parse_platform_path_list(OsStr::new(&value))),
            native_link_dirs: select_metadata_value(
                cuvs.native_link_dirs.as_deref(),
                cuvs.aggregate_metadata.as_deref(),
                "native-link-dirs",
            )
            .map(|value| parse_platform_path_list(OsStr::new(&value))),
            libcuvs_runtime_lib: select_metadata_value(
                cuvs.libcuvs_runtime_lib.as_deref(),
                cuvs.aggregate_metadata.as_deref(),
                "libcuvs-runtime-lib",
            )
            .map(PathBuf::from),
            libcuvs_c_runtime_lib: select_metadata_value(
                cuvs.libcuvs_c_runtime_lib.as_deref(),
                cuvs.aggregate_metadata.as_deref(),
                "libcuvs-c-runtime-lib",
            )
            .map(PathBuf::from),
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

fn runtime_directories(metadata: &ResolvedMetadata, consumers: RapidsConsumers) -> Vec<PathBuf> {
    let mut runtime_dirs = Vec::new();
    if let Some(dir) = &metadata.cudf.runtime_lib_dir {
        push_unique_path(&mut runtime_dirs, dir.clone());
    }
    for library in [
        metadata.cudf.cudf_rust_runtime_lib.as_ref(),
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
    if consumers.cugraph {
        for dir in metadata.cugraph.runtime_lib_dirs.iter().flatten() {
            push_unique_path(&mut runtime_dirs, dir.clone());
        }
        if let Some(dir) = &metadata.cugraph.cuvs_runtime_lib_dir {
            push_unique_path(&mut runtime_dirs, dir.clone());
        }
    }
    if consumers.cuvs {
        for dir in metadata.cuvs.runtime_lib_dirs.iter().flatten() {
            push_unique_path(&mut runtime_dirs, dir.clone());
        }
        for library in [metadata.cuvs.libcuvs_runtime_lib.as_ref(), metadata.cuvs.libcuvs_c_runtime_lib.as_ref()]
            .into_iter()
            .flatten()
        {
            if let Some(parent) = library.parent() {
                push_unique_path(&mut runtime_dirs, parent.to_path_buf());
            }
        }
    }
    runtime_dirs
}

fn link_directives(metadata: &ResolvedMetadata, consumers: RapidsConsumers, targets: LinkArgTargets) -> Vec<String> {
    let mut directives = Vec::new();
    for dir in runtime_directories(metadata, consumers) {
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
    if consumers.cuvs {
        for library in [metadata.cuvs.libcuvs_runtime_lib.as_ref(), metadata.cuvs.libcuvs_c_runtime_lib.as_ref()]
            .into_iter()
            .flatten()
        {
            directives.extend(targets.required_runtime_library_directives(library));
        }
    }
    directives
}

fn read_cuda_environment() -> CudaEnvironment {
    CudaEnvironment {
        cuda_home: std::env::var_os(CUDA_HOME_ENV).map(PathBuf::from),
        cuda_path: std::env::var_os(CUDA_PATH_ENV).map(PathBuf::from),
        cuda_toolkit_root: std::env::var_os(CUDA_TOOLKIT_ROOT_ENV).map(PathBuf::from),
    }
}

/// Candidate CUDA toolkit roots in declared-precedence order.  The installed
/// `/usr/local/cuda` symlink remains a final fallback for normal CUDA hosts
/// that do not export a CMake discovery variable into Cargo's environment.
fn cuda_toolkit_roots(environment: &CudaEnvironment) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let canonical_root = PathBuf::from(CANONICAL_CUDA_ROOT);
    for root in [
        environment.cuda_toolkit_root.as_ref(),
        environment.cuda_home.as_ref(),
        environment.cuda_path.as_ref(),
        Some(&canonical_root),
    ]
    .into_iter()
    .flatten()
    {
        push_unique_path(&mut roots, root.clone());
    }
    roots
}

/// CUDA names Linux target directories by architecture and OS rather than
/// Cargo's full target triple (for example, `x86_64-linux`).
fn cuda_target_directory(target: &str) -> Option<String> {
    let architecture = target.split('-').next()?;
    target.contains("-linux").then(|| format!("{architecture}-linux"))
}

fn nvtx3interop_candidate_paths(environment: &CudaEnvironment, target: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for root in cuda_toolkit_roots(environment) {
        if let Some(cuda_target) = cuda_target_directory(target) {
            push_unique_path(
                &mut candidates,
                root.join("targets").join(cuda_target).join("lib").join(NVTX3INTEROP_LIBRARY),
            );
        }
        // CUDA's native toolkit layout uses lib64, while some installations
        // expose the same library through lib.  Keep both official layouts.
        push_unique_path(&mut candidates, root.join("lib64").join(NVTX3INTEROP_LIBRARY));
        push_unique_path(&mut candidates, root.join("lib").join(NVTX3INTEROP_LIBRARY));
    }
    candidates
}

fn nvtx3interop_library_diagnostic(candidates: &[PathBuf]) -> String {
    let candidates =
        candidates.iter().map(|candidate| format!("  {}", candidate.display())).collect::<Vec<_>>().join("\n");
    format!(
        "NVTX annotations require {NVTX3INTEROP_LIBRARY}, but it was not found. Searched:\n{candidates}\n\
         Set {CUDA_TOOLKIT_ROOT_ENV}, {CUDA_HOME_ENV}, or {CUDA_PATH_ENV} to the canonical CUDA toolkit root."
    )
}

fn resolve_nvtx3interop_library(environment: &CudaEnvironment, target: &str) -> PathBuf {
    let candidates = nvtx3interop_candidate_paths(environment, target);
    candidates
        .iter()
        .find(|candidate| candidate.is_file())
        .cloned()
        .unwrap_or_else(|| panic!("{}", nvtx3interop_library_diagnostic(&candidates)))
}

fn nvtx3interop_link_directives(library: &Path, targets: LinkArgTargets) -> Vec<String> {
    let directory = library.parent().expect("NVTX interop library always has a parent directory");
    let mut directives = vec![
        format!("cargo:rustc-link-search=native={}", directory.display()),
        "cargo:rustc-link-lib=dylib=nvtx3interop".to_string(),
    ];
    directives.extend(targets.rpath_directives(directory));
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

fn track_required_runtime_library(name: &str, library: Option<&Path>) {
    let library = library.unwrap_or_else(|| {
        panic!(
            "cudf-sys did not expose {name}; rebuild the sys crate so downstream executables can bind the exact native shim"
        )
    });
    if !library.is_file() {
        panic!("cudf-sys exposed {name}={}, but that library does not exist", library.display());
    }
    println!("cargo:rerun-if-changed={}", library.display());
}

fn track_cugraph_env() {
    track_env(CUGRAPH_RUNTIME_LIB_DIRS_ENV);
    track_env(CUGRAPH_NATIVE_LINK_DIRS_ENV);
    track_env(CUGRAPH_CUVS_LIBRARY_PATH_ENV);
    track_env(CUGRAPH_CUVS_RUNTIME_LIB_DIR_ENV);
    track_env(CUGRAPH_METADATA_ENV);
}

fn track_cuvs_env() {
    track_env(CUVS_RUNTIME_LIB_DIRS_ENV);
    track_env(CUVS_NATIVE_LINK_DIRS_ENV);
    track_env(CUVS_LIBCUVS_RUNTIME_LIB_ENV);
    track_env(CUVS_LIBCUVS_C_RUNTIME_LIB_ENV);
    track_env(CUVS_METADATA_ENV);
}

fn track_cuda_environment() {
    track_env(CUDA_TOOLKIT_ROOT_ENV);
    track_env(CUDA_HOME_ENV);
    track_env(CUDA_PATH_ENV);
}

pub fn configure(config: Config) {
    track_env(DOCS_RS_ENV);
    if std::env::var_os(DOCS_RS_ENV).is_some() {
        return;
    }

    let consumers = config.consumers;
    let environment = read_environment(consumers);
    let metadata = resolve_metadata(&environment);

    track_env(CUDF_METADATA_ENV);
    track_env(CUDF_RUNTIME_LIB_DIR_ENV);
    track_env(CUDF_RUST_RUNTIME_LIB_ENV);
    track_env(CUDF_LIBCUDF_RUNTIME_LIB_ENV);
    track_env(CUDF_RMM_RUNTIME_LIB_ENV);
    track_env(CUDF_RAPIDS_LOGGER_RUNTIME_LIB_ENV);

    if metadata.cudf.runtime_lib_dir.is_none() {
        println!("cargo:warning=cudf-sys runtime lib dir not exposed via DEP_CUDF_RUST_*; skipping rpath bake");
    }
    if consumers.cugraph {
        track_cugraph_env();
        if metadata.cugraph.runtime_lib_dirs.is_none() {
            println!("cargo:warning=cugraph-sys runtime lib dirs not exposed via DEP_CUGRAPH_*; skipping rpath bake");
        }
    }
    if consumers.cuvs {
        track_cuvs_env();
        if metadata.cuvs.runtime_lib_dirs.is_none() {
            println!(
                "cargo:warning=cuvs-sys runtime lib dirs not exposed via DEP_CUVS_C_*; using authoritative library paths for rpath bake"
            );
        }
    }

    track_required_runtime_library(CUDF_RUST_RUNTIME_LIB_ENV, metadata.cudf.cudf_rust_runtime_lib.as_deref());

    emit_cargo_directives(link_directives(&metadata, consumers, config.link_arg_targets));
    if config.nvtx3interop {
        track_cuda_environment();
        let cuda_environment = read_cuda_environment();
        let target = std::env::var("TARGET").expect("Cargo build scripts always receive TARGET");
        let nvtx3interop = resolve_nvtx3interop_library(&cuda_environment, &target);
        println!("cargo:rerun-if-changed={}", nvtx3interop.display());
        emit_cargo_directives(nvtx3interop_link_directives(&nvtx3interop, config.link_arg_targets));
    }
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

    if consumers.any() {
        enforce_single_rapids_runtime_libraries(&metadata, consumers);
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
        "an enabled RAPIDS consumer requires libcudf to ship a shared {lib_name}, but cudf-sys did not expose \
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

fn absorbed_logger_shared_candidate_diagnostic(consumer: &str, candidate: &Path) -> String {
    format!(
        "cudf-sys reports rapids_logger absorbed into libcudf/librmm (no shipped RAPIDS shared library links a \
         shared librapids_logger.so), but one is reachable from the {consumer} link/runtime dirs: {}. \
         If the consumer links it, this process carries two rapids_logger builds and logger symbols bind \
         unpredictably; if it is a stale artifact from an earlier shared build, any future relink would \
         adopt it. Remove the stale file and any stale rapids_logger CMake package (lib/cmake/rapids_logger \
         under the same prefix), then reconfigure the cugraph C++ build against the cudf install prefix \
         (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>) and rebuild it.",
        candidate.display()
    )
}

fn missing_candidate_diagnostic(consumer: &str, env_names: &str, lib_name: &str, cudf_lib: &Path) -> String {
    format!(
        "the {consumer} consumer is enabled but no {lib_name} is reachable from its link/runtime dirs \
         ({env_names}), so the build cannot prove it links the same {lib_name} as libcudf ({}). Reconfigure \
         that native consumer against the canonical cudf/RMM install prefix and rebuild it.",
        cudf_lib.display()
    )
}

fn divergent_library_diagnostic(consumer: &str, lib_name: &str, cudf_lib: &Path, consumer_lib: &Path) -> String {
    format!(
        "two divergent {lib_name} builds are linked into this process:\n  cudf-sys: {}\n  {consumer}: {}\n\
         One process must bind every {lib_name} symbol to a single build, or execution corrupts at runtime. \
         Reconfigure the native consumer against the canonical cudf/RMM install prefix, rebuild it, and remove \
         stale copies from its build tree.",
        cudf_lib.display(),
        consumer_lib.display()
    )
}

fn enforce_single_rapids_runtime_libraries(metadata: &ResolvedMetadata, consumers: RapidsConsumers) {
    track_env(CUDF_RAPIDS_LOGGER_LINKAGE_ENV);

    let rmm =
        metadata.cudf.rmm_runtime_lib.as_deref().unwrap_or_else(|| {
            panic!("{}", missing_required_library_diagnostic("librmm.so", CUDF_RMM_RUNTIME_LIB_ENV))
        });
    if consumers.cugraph {
        enforce_shared_rapids_library_matches_cudf(
            "cugraph-sys",
            "DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS",
            "librmm.so",
            rmm,
            &metadata.cugraph.native_link_dirs,
            &metadata.cugraph.runtime_lib_dirs,
            true,
        );
    }
    if consumers.cuvs {
        enforce_shared_rapids_library_matches_cudf(
            "cuvs-sys",
            "DEP_CUVS_C_NATIVE_LINK_DIRS / DEP_CUVS_C_RUNTIME_LIB_DIRS",
            "librmm.so",
            rmm,
            &metadata.cuvs.native_link_dirs,
            &metadata.cuvs.runtime_lib_dirs,
            false,
        );
    }

    match optional_logger_policy(
        metadata.cudf.rapids_logger_runtime_lib.as_deref(),
        metadata.cudf.rapids_logger_linkage.as_deref(),
    ) {
        OptionalLoggerPolicy::Enforce(library) => {
            if consumers.cugraph {
                enforce_shared_rapids_library_matches_cudf(
                    "cugraph-sys",
                    "DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS",
                    "librapids_logger.so",
                    &library,
                    &metadata.cugraph.native_link_dirs,
                    &metadata.cugraph.runtime_lib_dirs,
                    true,
                );
            }
            if consumers.cuvs {
                enforce_shared_rapids_library_matches_cudf(
                    "cuvs-sys",
                    "DEP_CUVS_C_NATIVE_LINK_DIRS / DEP_CUVS_C_RUNTIME_LIB_DIRS",
                    "librapids_logger.so",
                    &library,
                    &metadata.cuvs.native_link_dirs,
                    &metadata.cuvs.runtime_lib_dirs,
                    false,
                );
            }
        },
        // cudf-sys classified rapids_logger as absorbed by inspecting real
        // DT_NEEDED linkage. The single-RMM enforcement above already pins
        // cugraph to the same install prefix, whose static logger archive is
        // therefore the only one either side can absorb — the invariant holds
        // by construction, so this state stays quiet. What must not exist is
        // a *shared* logger reachable from the cugraph dirs: that is either a
        // second logger build in the process or a stale artifact any relink
        // would adopt, and it breaks the absorbed contract.
        OptionalLoggerPolicy::Skip { linkage } if linkage == "absorbed" => {
            if consumers.cugraph {
                enforce_absorbed_logger_has_no_shared_candidate(
                    "cugraph-sys",
                    &metadata.cugraph.native_link_dirs,
                    &metadata.cugraph.runtime_lib_dirs,
                );
            }
            if consumers.cuvs {
                enforce_absorbed_logger_has_no_shared_candidate(
                    "cuvs-sys",
                    &metadata.cuvs.native_link_dirs,
                    &metadata.cuvs.runtime_lib_dirs,
                );
            }
        },
        OptionalLoggerPolicy::Skip { linkage } => println!("{}", optional_logger_warning(&linkage)),
    }

    enforce_cuvs_runtime_contract(metadata, consumers);
}

fn enforce_absorbed_logger_has_no_shared_candidate(
    consumer: &str,
    native_link_dirs: &Option<Vec<PathBuf>>,
    runtime_lib_dirs: &Option<Vec<PathBuf>>,
) {
    let candidates =
        existing_candidates(consumer_candidate_paths("librapids_logger.so", native_link_dirs, runtime_lib_dirs));
    for candidate in &candidates {
        println!("cargo:rerun-if-changed={}", candidate.display());
    }
    if let Some(candidate) = candidates.first() {
        panic!("{}", absorbed_logger_shared_candidate_diagnostic(consumer, candidate));
    }
}

/// Produce candidate paths without reading the filesystem. Native link dirs
/// precede runtime dirs, matching the historical build-script policy.
fn consumer_candidate_paths(
    lib_name: &str,
    native_link_dirs: &Option<Vec<PathBuf>>,
    runtime_lib_dirs: &Option<Vec<PathBuf>>,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for dirs in [native_link_dirs, runtime_lib_dirs].into_iter().flatten() {
        for dir in dirs {
            push_unique_path(&mut candidates, dir.join(lib_name));
        }
    }
    candidates
}

fn existing_candidates(candidate_paths: Vec<PathBuf>) -> Vec<PathBuf> {
    candidate_paths.into_iter().filter(|candidate| candidate.is_file()).collect()
}

fn require_runtime_library<'a>(feature: &str, env_name: &str, library: Option<&'a Path>) -> &'a Path {
    let library = library.unwrap_or_else(|| {
        panic!(
            "the `{feature}` feature requires {env_name} from its sys crate's Cargo metadata; \
             rebuild the sys crate against the canonical native prefix"
        )
    });
    if !library.is_file() {
        panic!(
            "the `{feature}` feature received {env_name}={}, but that runtime library does not exist",
            library.display()
        );
    }
    println!("cargo:rerun-if-changed={}", library.display());
    library
}

fn path_has_private_deps_component(path: &Path) -> bool {
    path.components().any(|component| component.as_os_str() == OsStr::new("_deps"))
}

fn reject_private_cuvs_path(consumer: &str, path: &Path) {
    if path_has_private_deps_component(path) {
        panic!(
            "{consumer} resolved libcuvs.so from a private `_deps` build ({}); configure every consumer with \
             the canonical cuVS install prefix and remove the private artifact",
            path.display()
        );
    }
}

fn canonical_runtime_path(consumer: &str, path: &Path) -> PathBuf {
    path.canonicalize()
        .unwrap_or_else(|error| panic!("failed to canonicalize {consumer} runtime library {}: {error}", path.display()))
}

fn divergent_cuvs_identity_diagnostic(cuvs_path: &Path, cugraph_path: &Path) -> String {
    format!(
        "the Rust cuVS wrapper and cuGraph resolved different canonical libcuvs.so files:\n  \
         cuvs-sys:    {}\n  cugraph-sys: {}\nConfigure both with the same CUVS_ROOT/cuvs_ROOT install prefix; \
         one process must load exactly one cuVS build.",
        cuvs_path.display(),
        cugraph_path.display()
    )
}

fn enforce_cuvs_runtime_contract(metadata: &ResolvedMetadata, consumers: RapidsConsumers) {
    let cuvs_library = consumers.cuvs.then(|| {
        let libcuvs =
            require_runtime_library("cuvs", CUVS_LIBCUVS_RUNTIME_LIB_ENV, metadata.cuvs.libcuvs_runtime_lib.as_deref());
        require_runtime_library("cuvs", CUVS_LIBCUVS_C_RUNTIME_LIB_ENV, metadata.cuvs.libcuvs_c_runtime_lib.as_deref());
        reject_private_cuvs_path("cuvs-sys", libcuvs);
        canonical_runtime_path("cuvs-sys", libcuvs)
    });

    let cugraph_library = consumers.cugraph.then(|| {
        let libcuvs = require_runtime_library(
            "cugraph",
            CUGRAPH_CUVS_LIBRARY_PATH_ENV,
            metadata.cugraph.cuvs_library_path.as_deref(),
        );
        reject_private_cuvs_path("cugraph-sys", libcuvs);
        canonical_runtime_path("cugraph-sys", libcuvs)
    });

    if let (Some(cuvs_path), Some(cugraph_path)) = (&cuvs_library, &cugraph_library)
        && cuvs_path != cugraph_path
    {
        panic!("{}", divergent_cuvs_identity_diagnostic(cuvs_path, cugraph_path));
    }
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

fn enforce_shared_rapids_library_matches_cudf(
    consumer: &str,
    env_names: &str,
    lib_name: &str,
    cudf_lib: &Path,
    native_link_dirs: &Option<Vec<PathBuf>>,
    runtime_lib_dirs: &Option<Vec<PathBuf>>,
    candidate_required: bool,
) {
    println!("cargo:rerun-if-changed={}", cudf_lib.display());
    let candidate_paths = existing_candidates(consumer_candidate_paths(lib_name, native_link_dirs, runtime_lib_dirs));
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
        CandidateResolution::Missing if candidate_required => {
            panic!("{}", missing_candidate_diagnostic(consumer, env_names, lib_name, cudf_lib))
        },
        CandidateResolution::Missing => {},
        CandidateResolution::Consistent { .. } => {},
        CandidateResolution::Divergent { divergent, .. } => {
            panic!("{}", divergent_library_diagnostic(consumer, lib_name, cudf_lib, &divergent))
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
                .join(format!("algeon-datafusion-rapids-build-support-{label}-{}-{sequence}", std::process::id()));
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

    const TEST_COMPONENT: NativeComponent = NativeComponent {
        crate_name: "cudf-sys",
        install_dir_name: "cudf-install",
        required_library: "lib/libcudf_rust.so",
        override_env: Some("CUDF_INSTALL_DIR"),
        source_required_env: Some("CARGO_FEATURE_TEST_HOOKS"),
        source_tree_sentinel: Some("../cudf/cmake/rapids_config.cmake"),
        build_script: "scripts/build/build_libcudf.sh",
    };

    #[test]
    fn native_linking_precedence_is_docs_override_native_root_source_then_workspace() {
        let test_dir = TestDir::new("native-linking-precedence");
        let manifest = test_dir.path().join("components/cudf/algeon-cudf-sys");
        std::fs::create_dir_all(manifest.join("../cudf/cmake")).unwrap();
        std::fs::write(manifest.join("../cudf/cmake/rapids_config.cmake"), b"sentinel").unwrap();

        let docs = |name: &str| (name == DOCS_RS_ENV).then(|| OsString::from("1"));
        assert_eq!(resolve_native_linking(&TEST_COMPONENT, &manifest, &docs), NativeLinking::Skip);

        let override_prefix = test_dir.path().join("override");
        std::fs::create_dir_all(override_prefix.join("lib")).unwrap();
        std::fs::write(override_prefix.join(TEST_COMPONENT.required_library), b"shim").unwrap();
        let explicit = |name: &str| (name == "CUDF_INSTALL_DIR").then(|| override_prefix.clone().into_os_string());
        assert_eq!(
            resolve_native_linking(&TEST_COMPONENT, &manifest, &explicit),
            NativeLinking::Prebuilt(override_prefix.clone())
        );

        let source_required = |name: &str| {
            (name == "CUDF_INSTALL_DIR")
                .then(|| override_prefix.clone().into_os_string())
                .or_else(|| (name == "CARGO_FEATURE_TEST_HOOKS").then(|| OsString::from("1")))
        };
        assert_eq!(resolve_native_linking(&TEST_COMPONENT, &manifest, &source_required), NativeLinking::SourceTree);

        let native_root = test_dir.path().join("native-root");
        let native_prefix = native_root.join(TEST_COMPONENT.install_dir_name);
        std::fs::create_dir_all(native_prefix.join("lib")).unwrap();
        std::fs::write(native_prefix.join(TEST_COMPONENT.required_library), b"shim").unwrap();
        let native = |name: &str| (name == ALGEON_NATIVE_DIR_ENV).then(|| native_root.clone().into_os_string());
        assert_eq!(resolve_native_linking(&TEST_COMPONENT, &manifest, &native), NativeLinking::Prebuilt(native_prefix));

        assert_eq!(resolve_native_linking(&TEST_COMPONENT, &manifest, &|_| None), NativeLinking::SourceTree);

        std::fs::remove_file(manifest.join("../cudf/cmake/rapids_config.cmake")).unwrap();
        let workspace_prefix = manifest.join("../../../target/native/cudf-install");
        std::fs::create_dir_all(workspace_prefix.join("lib")).unwrap();
        std::fs::write(workspace_prefix.join(TEST_COMPONENT.required_library), b"shim").unwrap();
        assert_eq!(
            resolve_native_linking(&TEST_COMPONENT, &manifest, &|_| None),
            NativeLinking::Prebuilt(workspace_prefix)
        );
    }

    #[test]
    fn native_linking_explicit_missing_prefix_fails_with_actionable_diagnostic() {
        let test_dir = TestDir::new("native-linking-missing");
        let manifest = test_dir.path().join("manifest");
        std::fs::create_dir_all(&manifest).unwrap();
        let missing_prefix = test_dir.path().join("missing-prefix");
        let linking = resolve_native_linking(&TEST_COMPONENT, &manifest, &|name| {
            (name == "CUDF_INSTALL_DIR").then(|| missing_prefix.clone().into_os_string())
        });
        let NativeLinking::Missing(missing) = linking else {
            panic!("an explicitly selected incomplete prefix must be rejected")
        };
        let diagnostic = missing.to_string();
        assert!(diagnostic.contains("cudf-sys: no usable libcudf_rust.so"));
        assert!(diagnostic.contains("CUDF_INSTALL_DIR set to"));
        assert!(diagnostic.contains("bash scripts/build/build_libcudf.sh"));
        assert!(diagnostic.contains("https://algeon.dev/docs/build/from-source"));
    }

    #[test]
    fn prebuilt_crate_version_mismatch_names_both_versions_and_rebuild_command() {
        let test_dir = TestDir::new("native-version-mismatch");
        let marker = test_dir.path().join("share/cudf_rust/crate-version");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, "0.0.9\n").unwrap();

        let diagnostic =
            verify_prebuilt_crate_version(&TEST_COMPONENT, test_dir.path(), "share/cudf_rust/crate-version", "0.1.0")
                .expect_err("mismatched native artifacts must be rejected");

        assert!(diagnostic.contains("cudf-sys 0.1.0"));
        assert!(diagnostic.contains("built for crate version 0.0.9"));
        assert!(diagnostic.contains("checkout v0.1.0 && bash scripts/build/build_libcudf.sh"));
    }

    #[test]
    fn metadata_resolution_merges_direct_and_aggregate_values_table() {
        struct Case {
            name: &'static str,
            environment: CudfEnvironment,
            expected: CudfMetadata,
        }
        let aggregate = Some(
            "runtime-lib-dir=/aggregate;cudf-rust-runtime-lib=/aggregate/libcudf_rust.so;\
             libcudf-runtime-lib=/aggregate/libcudf.so;\
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
                    cudf_rust_runtime_lib: Some("/aggregate/libcudf_rust.so".into()),
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
                    cudf_rust_runtime_lib: Some("/direct/libcudf_rust.so".to_string()),
                    libcudf_runtime_lib: Some("/direct/libcudf.so".to_string()),
                    rmm_runtime_lib: None,
                    rapids_logger_runtime_lib: Some("/direct/librapids_logger.so".to_string()),
                    rapids_logger_linkage: Some("shared".to_string()),
                },
                expected: CudfMetadata {
                    runtime_lib_dir: Some("/direct".into()),
                    cudf_rust_runtime_lib: Some("/direct/libcudf_rust.so".into()),
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
                cuvs_library_path: None,
                cuvs_runtime_lib_dir: None,
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
                cudf_rust_runtime_lib: Some("/runtime/libcudf_rust.so".into()),
                libcudf_runtime_lib: Some("/cudf/libcudf.so".into()),
                rmm_runtime_lib: Some("/rmm/librmm.so".into()),
                rapids_logger_runtime_lib: None,
                rapids_logger_linkage: None,
            },
            cugraph: CugraphMetadata {
                runtime_lib_dirs: Some(vec!["/graph".into()]),
                native_link_dirs: None,
                ..Default::default()
            },
            cuvs: CuvsMetadata::default(),
        };
        let directives =
            link_directives(&metadata, RapidsConsumers::new(true, false), LinkArgTargets::new(true, false, true));
        assert_eq!(directives[0], "cargo:rustc-link-arg=-Wl,-rpath,/runtime");
        assert_eq!(directives[1], "cargo:rustc-link-arg-tests=-Wl,-rpath,/runtime");
        assert!(directives.iter().any(|line| line.contains("/cudf/libcudf.so")));
        assert!(directives.iter().any(|line| line.contains("/rmm/librmm.so")));
        assert!(!directives.iter().any(|line| line.starts_with("cargo:rustc-link-arg-bins=")));
    }

    #[test]
    fn nvtx3interop_discovery_uses_cuda_roots_and_target_layout_before_lib64() {
        let test_dir = TestDir::new("nvtx");
        let toolkit = test_dir.path().join("cuda");
        let target_directory = toolkit.join("targets/x86_64-linux/lib");
        let fallback_directory = toolkit.join("lib64");
        std::fs::create_dir_all(&target_directory).unwrap();
        std::fs::create_dir_all(&fallback_directory).unwrap();
        let selected = target_directory.join(NVTX3INTEROP_LIBRARY);
        std::fs::write(&selected, b"target nvtx").unwrap();
        std::fs::write(fallback_directory.join(NVTX3INTEROP_LIBRARY), b"fallback nvtx").unwrap();

        let environment = CudaEnvironment {
            cuda_toolkit_root: Some(toolkit.clone()),
            cuda_home: Some(toolkit.clone()),
            cuda_path: Some(toolkit.clone()),
        };
        assert_eq!(cuda_toolkit_roots(&environment), vec![toolkit.clone(), PathBuf::from(CANONICAL_CUDA_ROOT)]);
        assert_eq!(resolve_nvtx3interop_library(&environment, "x86_64-unknown-linux-gnu"), selected);
    }

    #[test]
    fn nvtx3interop_link_directives_provide_search_link_and_runtime_resolution() {
        let library = Path::new("/cuda/targets/x86_64-linux/lib/libnvtx3interop.so");
        assert_eq!(
            nvtx3interop_link_directives(library, LinkArgTargets::new(true, false, true)),
            vec![
                "cargo:rustc-link-search=native=/cuda/targets/x86_64-linux/lib",
                "cargo:rustc-link-lib=dylib=nvtx3interop",
                "cargo:rustc-link-arg=-Wl,-rpath,/cuda/targets/x86_64-linux/lib",
                "cargo:rustc-link-arg-tests=-Wl,-rpath,/cuda/targets/x86_64-linux/lib",
            ]
        );
    }

    #[test]
    fn missing_required_rmm_and_optional_logger_policies_are_typed_and_stable() {
        assert_eq!(
            missing_required_library_diagnostic("librmm.so", CUDF_RMM_RUNTIME_LIB_ENV),
            "an enabled RAPIDS consumer requires libcudf to ship a shared librmm.so, but cudf-sys did not expose one via DEP_CUDF_RUST_RMM_RUNTIME_LIB. Rebuild libcudf with CUDF_RMM_BUILD_SHARED=ON (the cudf fork default), re-run `cmake --install`, and rebuild cudf-sys so the install prefix exports a single canonical librmm.so for every RAPIDS consumer in the process."
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
            ..Default::default()
        };
        assert_eq!(
            consumer_candidate_paths("librmm.so", &metadata.native_link_dirs, &metadata.runtime_lib_dirs),
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
            missing_candidate_diagnostic(
                "cugraph-sys",
                "DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS",
                "librmm.so",
                Path::new("/cudf/librmm.so")
            ),
            "the cugraph-sys consumer is enabled but no librmm.so is reachable from its link/runtime dirs (DEP_CUGRAPH_NATIVE_LINK_DIRS / DEP_CUGRAPH_RUNTIME_LIB_DIRS), so the build cannot prove it links the same librmm.so as libcudf (/cudf/librmm.so). Reconfigure that native consumer against the canonical cudf/RMM install prefix and rebuild it."
        );
        assert_eq!(
            divergent_library_diagnostic(
                "cugraph-sys",
                "librmm.so",
                Path::new("/cudf/librmm.so"),
                Path::new("/graph/librmm.so")
            ),
            "two divergent librmm.so builds are linked into this process:\n  cudf-sys: /cudf/librmm.so\n  cugraph-sys: /graph/librmm.so\nOne process must bind every librmm.so symbol to a single build, or execution corrupts at runtime. Reconfigure the native consumer against the canonical cudf/RMM install prefix, rebuild it, and remove stale copies from its build tree."
        );
        assert_eq!(
            absorbed_logger_shared_candidate_diagnostic("cugraph-sys", Path::new("/graph/librapids_logger.so")),
            "cudf-sys reports rapids_logger absorbed into libcudf/librmm (no shipped RAPIDS shared library links a shared librapids_logger.so), but one is reachable from the cugraph-sys link/runtime dirs: /graph/librapids_logger.so. If the consumer links it, this process carries two rapids_logger builds and logger symbols bind unpredictably; if it is a stale artifact from an earlier shared build, any future relink would adopt it. Remove the stale file and any stale rapids_logger CMake package (lib/cmake/rapids_logger under the same prefix), then reconfigure the cugraph C++ build against the cudf install prefix (cmake -Drmm_ROOT=<cudf-install> -Drapids_logger_ROOT=<cudf-install>) and rebuild it."
        );
    }

    #[test]
    fn absorbed_logger_policy_rejects_only_reachable_shared_candidates() {
        let test_dir = TestDir::new("absorbed-logger");
        let graph_native = test_dir.path().join("graph native");
        std::fs::create_dir_all(&graph_native).unwrap();
        let metadata = CugraphMetadata {
            native_link_dirs: Some(vec![graph_native.clone()]),
            runtime_lib_dirs: None,
            ..Default::default()
        };

        // Nothing reachable: the absorbed contract holds and stays quiet.
        enforce_absorbed_logger_has_no_shared_candidate(
            "cugraph-sys",
            &metadata.native_link_dirs,
            &metadata.runtime_lib_dirs,
        );

        // A reachable shared logger breaks the absorbed contract.
        let candidate = graph_native.join("librapids_logger.so");
        std::fs::write(&candidate, b"stale or divergent shared logger").unwrap();
        let result = std::panic::catch_unwind(|| {
            enforce_absorbed_logger_has_no_shared_candidate(
                "cugraph-sys",
                &metadata.native_link_dirs,
                &metadata.runtime_lib_dirs,
            )
        });
        let panic = result.expect_err("a reachable shared logger must fail the absorbed policy");
        let message = panic.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(message.contains("two rapids_logger builds"), "unexpected diagnostic: {message}");
        assert!(message.contains(candidate.to_str().unwrap()), "diagnostic must name the candidate: {message}");
    }
}
