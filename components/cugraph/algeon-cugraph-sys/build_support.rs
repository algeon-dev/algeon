#![cfg_attr(test, allow(dead_code))]

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

const CUDA_ARCHITECTURE_CONTRACT: &str = "share/algeon-native/cuda-architectures.env";
const CUDF_RUST_PUBLIC_HEADERS: [&str; 2] =
    ["include/cudf_rust/error_boundary.hpp", "include/cudf_rust/memory_resource.h"];

struct WorkspaceNativeBuildLayout {
    repo_root: PathBuf,
    src_dir: PathBuf,
    build_dir: PathBuf,
    cudf_install_dir: PathBuf,
    cuvs: CanonicalCuvsContract,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaArchitectureContract {
    requested: String,
    real: BTreeSet<String>,
}

fn repository_root(manifest_dir: &Path) -> Result<PathBuf, String> {
    manifest_dir.parent().and_then(Path::parent).and_then(Path::parent).map(Path::to_path_buf).ok_or_else(|| {
        format!("Could not determine the repository root from CARGO_MANIFEST_DIR='{}'.", manifest_dir.display())
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeBuildContract {
    pub(crate) native_include_dirs: Vec<PathBuf>,
    pub(crate) native_link_dirs: Vec<PathBuf>,
    pub(crate) runtime_lib_dirs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalCuvsContract {
    pub(crate) install_dir: PathBuf,
    pub(crate) cmake_config_dir: PathBuf,
    pub(crate) library_selector_path: PathBuf,
    pub(crate) library_path: PathBuf,
    pub(crate) runtime_lib_dir: PathBuf,
    pub(crate) source_revision_inputs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedCuvsLoader {
    pub(crate) cugraph_library_selector_path: PathBuf,
    pub(crate) cugraph_library_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedLinkCommand {
    pub(crate) link_dirs: Vec<PathBuf>,
    pub(crate) runtime_dirs: Vec<PathBuf>,
}

pub(crate) fn explicit_native_contract_count(
    include_dirs_env: Option<&OsString>,
    link_dirs_env: Option<&OsString>,
    runtime_lib_dirs_env: Option<&OsString>,
) -> usize {
    [include_dirs_env, link_dirs_env, runtime_lib_dirs_env].into_iter().filter(|value| value.is_some()).count()
}

pub(crate) fn prepare_workspace_cmake_export_dir(
    manifest_dir: &Path,
    cpp_build_dir_env: Option<OsString>,
    cudf_install_dir_env: Option<OsString>,
    cuvs_install_dir_env: Option<OsString>,
    auto_build_env: Option<OsString>,
    cuda_architectures_env: Option<OsString>,
) -> Result<PathBuf, String> {
    let layout =
        workspace_native_build_layout(manifest_dir, cpp_build_dir_env, cudf_install_dir_env, cuvs_install_dir_env)?;
    let cuda_architectures = resolved_cuda_architectures(cuda_architectures_env.as_deref());
    let requested = cuda_architectures.to_string_lossy().into_owned();
    // Reject inconsistent dependencies before the expensive cuGraph build; cuGraph
    // itself is only validated once it exists.
    validate_cuda_architecture_contracts(&layout.cuda_architecture_components()[..2], &requested)?;
    let auto_build_enabled = auto_native_build_enabled(auto_build_env.as_deref())?;
    if layout.build_dir.join("cugraph-config.cmake").is_file() {
        if let Some(reason) = workspace_build_reconfigure_reason(&layout, &requested)? {
            if !auto_build_enabled {
                return Err(format!(
                    "The workspace-local cuGraph CMake build at '{}' must be reconfigured ({reason}), but automatic \
                     native builds are disabled by CUGRAPH_SYS_AUTO_BUILD. Reconfigure native cuGraph manually, \
                     unset CUGRAPH_SYS_AUTO_BUILD, or set CUGRAPH_CMAKE_BUILD_DIR / the explicit CUGRAPH_NATIVE_* \
                     contract.",
                    layout.build_dir.display()
                ));
            }
            build_workspace_cugraph_cpp(&layout, &cuda_architectures, &reason)?;
        }
        validate_workspace_runtime_packages(&layout)?;
        validate_cuda_architecture_contracts(&layout.cuda_architecture_components(), &requested)?;
        return validate_cmake_export_dir(&layout.build_dir, "workspace-local cuGraph CMake build directory");
    }

    if !auto_build_enabled {
        return Err(format!(
            "Could not resolve the cuGraph native contract from '{}', and automatic native builds are disabled by \
             CUGRAPH_SYS_AUTO_BUILD. Either build native cuGraph with `bash scripts/build/build_libcugraph.sh`, \
             set CUGRAPH_CMAKE_BUILD_DIR, or set all of CUGRAPH_NATIVE_INCLUDE_DIRS, CUGRAPH_NATIVE_LINK_DIRS, and \
             CUGRAPH_RUNTIME_LIB_DIRS.",
            layout.build_dir.display()
        ));
    }

    build_workspace_cugraph_cpp(&layout, &cuda_architectures, "CMake export is missing")?;
    validate_workspace_runtime_packages(&layout)?;
    validate_cuda_architecture_contracts(&layout.cuda_architecture_components(), &requested)?;
    validate_cmake_export_dir(&layout.build_dir, "workspace-local cuGraph CMake build directory")
}

pub(crate) fn resolve_native_build_contract(
    manifest_dir: &Path,
    include_dirs_env: Option<OsString>,
    link_dirs_env: Option<OsString>,
    runtime_lib_dirs_env: Option<OsString>,
    cmake_build_dir_env: Option<OsString>,
) -> Result<NativeBuildContract, String> {
    let explicit_contract_count = explicit_native_contract_count(
        include_dirs_env.as_ref(),
        link_dirs_env.as_ref(),
        runtime_lib_dirs_env.as_ref(),
    );

    if explicit_contract_count == 3 && cmake_build_dir_env.is_some() {
        return Err("The native contract must come from exactly one source. Do not set CUGRAPH_CMAKE_BUILD_DIR when \
             CUGRAPH_NATIVE_INCLUDE_DIRS, CUGRAPH_NATIVE_LINK_DIRS, and CUGRAPH_RUNTIME_LIB_DIRS are set."
            .to_string());
    }

    if explicit_contract_count == 3 {
        return Ok(NativeBuildContract {
            native_include_dirs: resolve_native_include_dirs(include_dirs_env)?,
            native_link_dirs: resolve_native_link_dirs(link_dirs_env)?,
            runtime_lib_dirs: resolve_runtime_lib_dirs(runtime_lib_dirs_env)?,
        });
    }

    if explicit_contract_count != 0 {
        return Err("The native contract must come from exactly one source. Set all of CUGRAPH_NATIVE_INCLUDE_DIRS, \
             CUGRAPH_NATIVE_LINK_DIRS, and CUGRAPH_RUNTIME_LIB_DIRS together, or set none of them and use \
             a formal CMake export root via CUGRAPH_CMAKE_BUILD_DIR or the workspace-local target/native/cugraph-build export."
            .to_string());
    }

    let cmake_export_dir = resolve_cmake_export_dir(manifest_dir, cmake_build_dir_env)?;
    resolve_native_build_contract_from_cmake_export(&cmake_export_dir)
}

pub(crate) fn resolve_native_include_dirs(env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let include_dirs = resolve_required_path_list("CUGRAPH_NATIVE_INCLUDE_DIRS", env_val)?;
    primary_cugraph_include_dir(&include_dirs)?;
    Ok(include_dirs)
}

pub(crate) fn resolve_native_link_dirs(env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let link_dirs = resolve_required_path_list("CUGRAPH_NATIVE_LINK_DIRS", env_val)?;
    ensure_contains_cugraph_libraries("CUGRAPH_NATIVE_LINK_DIRS", &link_dirs)?;
    Ok(link_dirs)
}

pub(crate) fn resolve_runtime_lib_dirs(env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let runtime_dirs = resolve_required_path_list("CUGRAPH_RUNTIME_LIB_DIRS", env_val)?;
    ensure_contains_cugraph_libraries("CUGRAPH_RUNTIME_LIB_DIRS", &runtime_dirs)?;
    Ok(runtime_dirs)
}

pub(crate) fn primary_cugraph_include_dir(include_dirs: &[PathBuf]) -> Result<PathBuf, String> {
    include_dirs
        .iter()
        .find(|path| path.join("cugraph_c").is_dir())
        .cloned()
        .ok_or_else(|| {
            format!(
                "CUGRAPH_NATIVE_INCLUDE_DIRS is invalid. Expected at least one directory containing cugraph_c/. Provided paths: {}.",
                display_paths(include_dirs)
            )
        })
}

pub(crate) fn resolve_cmake_export_dir(manifest_dir: &Path, env_val: Option<OsString>) -> Result<PathBuf, String> {
    if let Some(raw_path) = env_val {
        return validate_cmake_export_dir(Path::new(&raw_path), "CUGRAPH_CMAKE_BUILD_DIR");
    }

    let candidate = repository_root(manifest_dir)?.join("target/native/cugraph-build");
    if candidate.join("cugraph-config.cmake").is_file() {
        return Ok(candidate.canonicalize().unwrap_or(candidate));
    }

    Err(format!(
        "Could not resolve the cuGraph native contract. Either set all of CUGRAPH_NATIVE_INCLUDE_DIRS, \
         CUGRAPH_NATIVE_LINK_DIRS, and CUGRAPH_RUNTIME_LIB_DIRS together, or point CUGRAPH_CMAKE_BUILD_DIR \
         at a build/install directory containing cugraph-config.cmake. The workspace-local export root \
         '{}' was not usable.",
        candidate.display()
    ))
}

pub(crate) fn resolve_canonical_cuvs_contract(
    manifest_dir: &Path,
    cuvs_install_dir_env: Option<OsString>,
) -> Result<CanonicalCuvsContract, String> {
    let repo_root = repository_root(manifest_dir)?;
    let default_install_dir = repo_root.join("target/native/cuvs-install");
    let requested_install_dir = cuvs_install_dir_env.map(PathBuf::from).unwrap_or_else(|| default_install_dir.clone());
    let install_dir = requested_install_dir.canonicalize().map_err(|error| {
        format!(
            "The canonical cuVS install prefix CUVS_INSTALL_DIR='{}' does not exist or is not accessible: {error}. \
             Build/install cuVS first, or set CUVS_INSTALL_DIR to that canonical prefix.",
            requested_install_dir.display()
        )
    })?;
    if !install_dir.is_dir() {
        return Err(format!(
            "The canonical cuVS install prefix CUVS_INSTALL_DIR='{}' is not a directory.",
            install_dir.display()
        ));
    }

    let cmake_config_dir = [install_dir.join("lib/cmake/cuvs"), install_dir.join("lib64/cmake/cuvs")]
        .into_iter()
        .find(|dir| dir.join("cuvs-config.cmake").is_file())
        .ok_or_else(|| {
            format!(
                "CUVS_INSTALL_DIR='{}' does not export a cuVS CMake package. Expected \
                 lib/cmake/cuvs/cuvs-config.cmake or lib64/cmake/cuvs/cuvs-config.cmake.",
                install_dir.display()
            )
        })?;
    let cmake_config_dir = cmake_config_dir.canonicalize().map_err(|error| {
        format!("Failed to canonicalize the cuVS CMake package directory '{}': {error}", cmake_config_dir.display())
    })?;

    let library_selector_path = [install_dir.join("lib/libcuvs.so"), install_dir.join("lib64/libcuvs.so")]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "CUVS_INSTALL_DIR='{}' does not contain the required shared library lib/libcuvs.so or \
                 lib64/libcuvs.so.",
                install_dir.display()
            )
        })?;
    let runtime_lib_dir =
        library_selector_path.parent().expect("libcuvs.so must have a parent directory").canonicalize().map_err(
            |error| {
                format!("Failed to canonicalize cuVS runtime directory '{}': {error}", library_selector_path.display())
            },
        )?;
    let library_path = library_selector_path.canonicalize().map_err(|error| {
        format!("Failed to resolve canonical libcuvs identity '{}': {error}", library_selector_path.display())
    })?;
    if !library_path.starts_with(&install_dir) {
        return Err(format!(
            "CUVS_INSTALL_DIR='{}' contains libcuvs.so resolving outside the canonical prefix to '{}'.",
            install_dir.display(),
            library_path.display()
        ));
    }

    let source_revision_inputs =
        if default_install_dir.canonicalize().is_ok_and(|default_install_dir| default_install_dir == install_dir) {
            validate_default_cuvs_source_revision(&repo_root, &install_dir)?
        } else {
            Vec::new()
        };

    Ok(CanonicalCuvsContract {
        install_dir,
        cmake_config_dir,
        library_selector_path,
        library_path,
        runtime_lib_dir,
        source_revision_inputs,
    })
}

fn validate_default_cuvs_source_revision(repo_root: &Path, install_dir: &Path) -> Result<Vec<PathBuf>, String> {
    const CUVS_SUBMODULE_PATH: &str = "components/cuvs/cuvs";
    let stage = command_stdout(
        "resolve the root cuVS source revision",
        Command::new("git").arg("-C").arg(repo_root).args(["ls-files", "--stage", "--", CUVS_SUBMODULE_PATH]),
    )?;
    let fields = stage.split_ascii_whitespace().collect::<Vec<_>>();
    let ["160000", expected_revision, "0", CUVS_SUBMODULE_PATH] = fields.as_slice() else {
        return Err(format!(
            "The default cuVS prefix requires exactly one stage-0 gitlink at '{}' in '{}', but git reported '{}'.",
            CUVS_SUBMODULE_PATH,
            repo_root.display(),
            stage.trim()
        ));
    };

    let marker_path = install_dir.join("share/algeon-cuvs/source-revision");
    let installed_revision = fs::read_to_string(&marker_path).map_err(|error| {
        format!(
            "The default cuVS prefix '{}' is missing its pinned source marker '{}': {error}. Rebuild it with scripts/build/build_libcuvs.sh.",
            install_dir.display(),
            marker_path.display()
        )
    })?;
    if installed_revision.trim() != *expected_revision {
        return Err(format!(
            "The default cuVS prefix '{}' was built from revision '{}', but the root cuVS gitlink pins '{}'. Rebuild the prefix with scripts/build/build_libcuvs.sh.",
            install_dir.display(),
            installed_revision.trim(),
            expected_revision
        ));
    }

    let git_index = command_stdout(
        "resolve the root Git index",
        Command::new("git").arg("-C").arg(repo_root).args([
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index",
        ]),
    )?;
    let git_index = normalize_existing_file(Path::new(git_index.trim()))?;
    Ok(vec![marker_path, git_index])
}

pub(crate) fn tracked_cmake_export_files(cmake_build_dir: &Path) -> Vec<PathBuf> {
    vec![
        cmake_build_dir.join("cugraph-config.cmake"),
        cmake_build_dir.join("CMakeCache.txt"),
        cmake_build_dir.join("cugraph-targets.cmake"),
        cmake_build_dir.join("cugraph-dependencies.cmake"),
        cmake_build_dir.join("_deps/rmm-build/rmm-config.cmake"),
        cmake_build_dir.join("_deps/rmm-build/rmm-targets.cmake"),
        cmake_build_dir.join("_deps/rmm-build/rmm-dependencies.cmake"),
        cmake_build_dir.join("_deps/raft-build/raft-config.cmake"),
        cmake_build_dir.join("_deps/raft-build/raft-targets.cmake"),
        cmake_build_dir.join("_deps/raft-build/raft-dependencies.cmake"),
    ]
}

/// One CUDA architecture participant: the component's display name, the prefix
/// holding its `share/algeon-native/cuda-architectures.env`, the command that
/// rebuilds it, and whether that contract is required.
///
/// `required` is false for an externally selected prefix (a `CUGRAPH_CMAKE_BUILD_DIR`
/// or `CUVS_INSTALL_DIR` pointing outside `target/native/`), which algeon scripts did
/// not build and therefore cannot be expected to carry a contract. This mirrors the
/// default-prefix-only policy of `validate_default_cuvs_source_revision`.
pub(crate) struct CudaArchitectureComponent<'a> {
    pub(crate) name: &'a str,
    pub(crate) root: &'a Path,
    pub(crate) rebuild_command: &'a str,
    pub(crate) required: bool,
}

/// Validate that every component carrying a CUDA architecture contract requested
/// `requested` and embeds the same real cubin set, and return the contract paths
/// that were read.
///
/// Components are compared against the first one that has a contract, so the error
/// names the dependency the mismatch is measured from.
pub(crate) fn validate_cuda_architecture_contracts(
    components: &[CudaArchitectureComponent<'_>],
    requested: &str,
) -> Result<Vec<PathBuf>, String> {
    let mut tracked = Vec::new();
    let mut reference: Option<(&str, CudaArchitectureContract)> = None;
    for component in components {
        let path = component.root.join(CUDA_ARCHITECTURE_CONTRACT);
        if !component.required && !path.is_file() {
            continue;
        }
        let contract = read_cuda_architecture_contract(&path, component.name, component.rebuild_command)?;
        if contract.requested != requested {
            return Err(format!(
                "{} was built for CMAKE_CUDA_ARCHITECTURES='{}', but this build requested '{requested}'. Run with \
                 CMAKE_CUDA_ARCHITECTURES='{requested}': {}",
                component.name, contract.requested, component.rebuild_command
            ));
        }
        match &reference {
            Some((reference_name, reference_contract)) if reference_contract.real != contract.real => {
                return Err(format!(
                    "{reference_name} contains full-kernel real CUDA cubins '{}', but {} contains '{}'. Rebuild the \
                     native stack in dependency order with one CMAKE_CUDA_ARCHITECTURES value: {}",
                    display_architectures(&reference_contract.real),
                    component.name,
                    display_architectures(&contract.real),
                    component.rebuild_command
                ));
            },
            Some(_) => {},
            None => reference = Some((component.name, contract)),
        }
        tracked.push(path);
    }
    Ok(tracked)
}

/// True when `root` is the canonical `target/native/<name>` prefix that algeon
/// scripts and the cargo auto-build own. Compares resolved paths so an equivalent
/// spelling of the default still counts as canonical.
fn is_canonical_native_prefix(root: &Path, repo_root: &Path, name: &str) -> bool {
    let canonical = repo_root.join("target/native").join(name);
    root == canonical || paths_equivalent(root, &canonical)
}

/// The cuDF -> cuVS -> cuGraph participants in dependency order, marking each as
/// required only when it sits at its canonical `target/native/` prefix.
pub(crate) fn cuda_architecture_components<'a>(
    repo_root: &Path,
    cudf_install_dir: &'a Path,
    cuvs_install_dir: &'a Path,
    cugraph_build_dir: &'a Path,
) -> [CudaArchitectureComponent<'a>; 3] {
    [
        CudaArchitectureComponent {
            name: "cuDF",
            root: cudf_install_dir,
            rebuild_command: "bash scripts/build/build_libcudf.sh",
            required: is_canonical_native_prefix(cudf_install_dir, repo_root, "cudf-install"),
        },
        CudaArchitectureComponent {
            name: "cuVS",
            root: cuvs_install_dir,
            rebuild_command: "bash scripts/build/build_libcuvs.sh",
            required: is_canonical_native_prefix(cuvs_install_dir, repo_root, "cuvs-install"),
        },
        CudaArchitectureComponent {
            name: "cuGraph",
            root: cugraph_build_dir,
            rebuild_command: "bash scripts/build/build_libcugraph.sh",
            required: is_canonical_native_prefix(cugraph_build_dir, repo_root, "cugraph-build"),
        },
    ]
}

impl WorkspaceNativeBuildLayout {
    fn cuda_architecture_components(&self) -> [CudaArchitectureComponent<'_>; 3] {
        cuda_architecture_components(&self.repo_root, &self.cudf_install_dir, &self.cuvs.install_dir, &self.build_dir)
    }
}

fn workspace_native_build_layout(
    manifest_dir: &Path,
    cpp_build_dir_env: Option<OsString>,
    cudf_install_dir_env: Option<OsString>,
    cuvs_install_dir_env: Option<OsString>,
) -> Result<WorkspaceNativeBuildLayout, String> {
    let repo_root = repository_root(manifest_dir)?;
    let src_dir = repo_root.join("components/cugraph/cugraph/cpp");
    if !src_dir.join("CMakeLists.txt").is_file() {
        return Err(format!(
            "The cuGraph C++ source tree was not found at '{}'. Initialize/update the cugraph submodule, or set \
             CUGRAPH_CMAKE_BUILD_DIR / the explicit CUGRAPH_NATIVE_* contract to a prebuilt cuGraph.",
            src_dir.display()
        ));
    }

    let build_dir =
        cpp_build_dir_env.map(PathBuf::from).unwrap_or_else(|| repo_root.join("target/native/cugraph-build"));
    let cudf_install_dir =
        cudf_install_dir_env.map(PathBuf::from).unwrap_or_else(|| repo_root.join("target/native/cudf-install"));
    let rmm_config = cudf_install_dir.join("lib/cmake/rmm/rmm-config.cmake");
    if !rmm_config.is_file() {
        return Err(format!(
            "The canonical cudf install prefix was not usable: '{}' does not exist. Build and install libcudf first \
             or set CUDF_INSTALL_DIR to a prefix that exports lib/cmake/rmm/rmm-config.cmake.",
            rmm_config.display()
        ));
    }
    let rapids_logger_config = cudf_install_dir.join("lib/cmake/rapids_logger/rapids_logger-config.cmake");
    if !rapids_logger_config.is_file() {
        return Err(format!(
            "The canonical cudf install prefix was not usable: '{}' does not exist. Build and install libcudf first \
             or set CUDF_INSTALL_DIR to a prefix that exports lib/cmake/rapids_logger/rapids_logger-config.cmake.",
            rapids_logger_config.display()
        ));
    }
    validate_cudf_public_headers(&cudf_install_dir)?;

    let cuvs = resolve_canonical_cuvs_contract(manifest_dir, cuvs_install_dir_env)?;

    Ok(WorkspaceNativeBuildLayout { repo_root, src_dir, build_dir, cudf_install_dir, cuvs })
}

pub(crate) fn validate_cudf_public_headers(cudf_install_dir: &Path) -> Result<(), String> {
    for relative_path in CUDF_RUST_PUBLIC_HEADERS {
        let header = cudf_install_dir.join(relative_path);
        let metadata = fs::symlink_metadata(&header).map_err(|error| {
            format!(
                "The canonical cudf install prefix is missing the public header '{}': {error}. Install the cuDF \
                 shim's cudf_rust headers into the prefix; a target-local manual symlink is not a supported \
                 substitute.",
                header.display()
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err(format!(
                "The canonical cudf install header '{}' is not a regular installed file. Install the cuDF shim \
                 public header into the prefix; target-local manual symlinks are not supported.",
                header.display()
            ));
        }
    }
    Ok(())
}

fn auto_native_build_enabled(env_val: Option<&std::ffi::OsStr>) -> Result<bool, String> {
    let Some(raw) = env_val else {
        return Ok(true);
    };
    let value = raw.to_string_lossy();
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("CUGRAPH_SYS_AUTO_BUILD='{}' is invalid. Use 1/true/on or 0/false/off.", value)),
    }
}

fn build_workspace_cugraph_cpp(
    layout: &WorkspaceNativeBuildLayout,
    cuda_architectures: &std::ffi::OsStr,
    reason: &str,
) -> Result<(), String> {
    println!("cargo:warning=Configuring native cuGraph at '{}' because {reason}.", layout.build_dir.display(),);

    let rmm_root = layout.cudf_install_dir.as_os_str();
    let rapids_logger_root = layout.cudf_install_dir.as_os_str();
    let cuvs_root = layout.cuvs.install_dir.as_os_str();

    let mut configure_command = Command::new("cmake");
    configure_command
        .arg("-S")
        .arg(&layout.src_dir)
        .arg("-B")
        .arg(&layout.build_dir)
        .arg("-GNinja")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={}", cuda_architectures.to_string_lossy()))
        .arg("-DBUILD_TESTS=OFF")
        .arg("-DBUILD_CUGRAPH_MG_TESTS=OFF")
        .arg("-DCUGRAPH_USE_CUVS_STATIC=OFF")
        .args([
            "-U",
            "rmm_DIR",
            "-U",
            "CPM_PACKAGE_rmm_*",
            "-U",
            "RMM_*",
            "-U",
            "FETCHCONTENT_SOURCE_DIR_RMM",
            "-U",
            "FETCHCONTENT_UPDATES_DISCONNECTED_RMM",
            "-U",
            "rapids_logger_DIR",
            "-U",
            "CPM_PACKAGE_rapids_logger_*",
            "-U",
            "RAPIDS_LOGGER_*",
            "-U",
            "FETCHCONTENT_SOURCE_DIR_RAPIDS_LOGGER",
            "-U",
            "FETCHCONTENT_UPDATES_DISCONNECTED_RAPIDS_LOGGER",
            "-U",
            "cuvs_DIR",
            "-U",
            "CPM_PACKAGE_cuvs_*",
            "-U",
            "CPM_cuvs_SOURCE",
            "-U",
            "CPM_DOWNLOAD_cuvs",
            "-U",
            "cuvs_SOURCE_DIR",
            "-U",
            "cuvs_BINARY_DIR",
            "-U",
            "CUVS_*",
            "-U",
            "FETCHCONTENT_SOURCE_DIR_CUVS",
            "-U",
            "FETCHCONTENT_UPDATES_DISCONNECTED_CUVS",
        ])
        .arg(format!("-Drmm_ROOT={}", Path::new(rmm_root).display()))
        .arg(format!("-Drapids_logger_ROOT={}", Path::new(rapids_logger_root).display()))
        .arg(format!("-Dcuvs_ROOT={}", Path::new(cuvs_root).display()))
        .arg(format!(
            "-DCMAKE_PREFIX_PATH={};{}",
            layout.cudf_install_dir.display(),
            layout.cuvs.install_dir.display()
        ));

    run_command_capture("configure native cuGraph with CMake", &mut configure_command)?;
    validate_workspace_runtime_packages(layout)?;

    let mut build_command = Command::new("cmake");
    build_command.arg("--build").arg(&layout.build_dir);
    run_command_capture("build native cuGraph with CMake", &mut build_command)?;
    reject_stale_private_runtime_artifacts(&layout.build_dir)?;
    write_cugraph_cuda_architecture_contract(layout, &cuda_architectures.to_string_lossy())
}

fn validate_workspace_runtime_packages(layout: &WorkspaceNativeBuildLayout) -> Result<(), String> {
    validate_cmake_cache_package_dir(&layout.build_dir, "rmm_DIR", &layout.cudf_install_dir.join("lib/cmake/rmm"))?;
    validate_cmake_cache_package_dir(
        &layout.build_dir,
        "rapids_logger_DIR",
        &layout.cudf_install_dir.join("lib/cmake/rapids_logger"),
    )?;
    validate_cmake_cache_package_dir(&layout.build_dir, "cuvs_DIR", &layout.cuvs.cmake_config_dir)?;

    reject_stale_private_runtime_artifacts(&layout.build_dir)
}

pub(crate) fn resolved_cuda_architectures(env_val: Option<&std::ffi::OsStr>) -> OsString {
    match env_val.filter(|value| !value.is_empty()) {
        Some(value) if value.to_string_lossy().eq_ignore_ascii_case("native") => OsString::from("NATIVE"),
        Some(value) => OsString::from(value),
        None => OsString::from("NATIVE"),
    }
}

fn read_cuda_architecture_contract(
    path: &Path,
    component: &str,
    rebuild_command: &str,
) -> Result<CudaArchitectureContract, String> {
    let contents = fs::read_to_string(path).map_err(|error| {
        format!(
            "{component} is missing its CUDA architecture contract '{}': {error}. Rebuild it with the canonical \
             native build script: {rebuild_command}",
            path.display()
        )
    })?;
    let unique_value = |key: &str| -> Result<String, String> {
        let values = contents.lines().filter_map(|line| line.strip_prefix(&format!("{key}="))).collect::<Vec<_>>();
        match values.as_slice() {
            [value] if !value.is_empty() => Ok((*value).to_string()),
            _ => Err(format!("CUDA architecture contract '{}' has no unique {key} value.", path.display())),
        }
    };
    let requested = unique_value("requested")?;
    let real = parse_real_cuda_architectures(&unique_value("real")?)
        .map_err(|error| format!("CUDA architecture contract '{}' {error}", path.display()))?;
    Ok(CudaArchitectureContract { requested, real })
}

fn display_architectures(architectures: &BTreeSet<String>) -> String {
    architectures.iter().cloned().collect::<Vec<_>>().join(";")
}

/// True for the shared real cubin architecture token grammar: digits plus an
/// optional single lowercase suffix letter (`120a` architecture-specific, `100f`
/// family-specific). `scripts/build/cuda_architecture_contract.sh` matches the
/// same grammar via `ALGEON_CUDA_REAL_ARCHITECTURE_PATTERN`, so both writers agree
/// on which cuobjdump tokens are architectures.
fn is_real_cuda_architecture(token: &str) -> bool {
    let digits = token.trim_end_matches(|character: char| character.is_ascii_lowercase());
    let suffix = &token[digits.len()..];
    !digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit()) && suffix.len() <= 1
}

fn parse_real_cuda_architectures(value: &str) -> Result<BTreeSet<String>, String> {
    let mut architectures = BTreeSet::new();
    for token in value.split(';') {
        if !is_real_cuda_architecture(token) {
            return Err(format!("contains '{token}', which is not a real CUDA architecture such as '89' or '120a'."));
        }
        architectures.insert(token.to_string());
    }
    Ok(architectures)
}

/// Extract the real cubin architectures from `cuobjdump --list-elf` output.
///
/// Primary build targets appear throughout the library; embedded dependency stubs
/// form much smaller groups, so groups under half the largest are discarded.
pub(crate) fn parse_listed_cubin_architectures(listing: &str) -> BTreeSet<String> {
    let mut counts = BTreeMap::<&str, usize>::new();
    for line in listing.lines() {
        let Some((_, suffix)) = line.rsplit_once(".sm_") else {
            continue;
        };
        if let Some((architecture, _)) = suffix.split_once(".cubin")
            && is_real_cuda_architecture(architecture)
        {
            *counts.entry(architecture).or_default() += 1;
        }
    }
    let Some(max_count) = counts.values().copied().max() else {
        return BTreeSet::new();
    };
    counts
        .into_iter()
        .filter(|&(_, count)| count * 2 >= max_count)
        .map(|(architecture, _)| architecture.to_string())
        .collect()
}

fn inspect_real_cuda_architectures(library: &Path) -> Result<BTreeSet<String>, String> {
    let toolkit_root = env::var_os("CUDAToolkit_ROOT")
        .or_else(|| env::var_os("CUDA_HOME"))
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let toolkit_cuobjdump = toolkit_root.join("bin/cuobjdump");
    let cuobjdump = if toolkit_cuobjdump.is_file() { toolkit_cuobjdump } else { PathBuf::from("cuobjdump") };
    let output = Command::new(&cuobjdump).arg("--list-elf").arg(library).output().map_err(|error| {
        format!(
            "Failed to inspect CUDA architectures in '{}' with '{}': {error}",
            library.display(),
            cuobjdump.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "cuobjdump failed while inspecting '{}': {}",
            library.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let architectures = parse_listed_cubin_architectures(&String::from_utf8_lossy(&output.stdout));
    if architectures.is_empty() {
        return Err(format!("'{}' does not contain a detectable real CUDA cubin architecture.", library.display()));
    }
    Ok(architectures)
}

fn write_cugraph_cuda_architecture_contract(
    layout: &WorkspaceNativeBuildLayout,
    requested: &str,
) -> Result<(), String> {
    let library = layout.build_dir.join("libcugraph.so");
    let real = inspect_real_cuda_architectures(&library)?;
    let path = layout.build_dir.join(CUDA_ARCHITECTURE_CONTRACT);
    fs::create_dir_all(path.parent().expect("contract path has a parent")).map_err(|error| {
        format!("Failed to create CUDA architecture contract directory for '{}': {error}", path.display())
    })?;
    fs::write(&path, format!("requested={requested}\nreal={}\n", display_architectures(&real)))
        .map_err(|error| format!("Failed to write CUDA architecture contract '{}': {error}", path.display()))
}

fn workspace_build_reconfigure_reason(
    layout: &WorkspaceNativeBuildLayout,
    requested: &str,
) -> Result<Option<String>, String> {
    let cache_path = layout.build_dir.join("CMakeCache.txt");
    let cache = match fs::read_to_string(&cache_path) {
        Ok(cache) => cache,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(Some(format!("{} is missing", cache_path.display())));
        },
        Err(error) => {
            return Err(format!("Failed to read cuGraph native CMake cache '{}': {error}", cache_path.display()));
        },
    };

    match cmake_cache_value(&cache, "CMAKE_CUDA_ARCHITECTURES") {
        Some(cached) if cached == requested => {},
        Some(cached) => {
            return Ok(Some(format!("CMAKE_CUDA_ARCHITECTURES changed from '{cached}' to '{requested}'")));
        },
        None => {
            return Ok(Some(format!("CMAKE_CUDA_ARCHITECTURES is missing from {}", cache_path.display())));
        },
    }

    // A cuGraph contract that disagrees with its dependencies is a rebuild trigger
    // here, not a hard failure: the rebuild is what restores agreement.
    if let Err(reason) = validate_cuda_architecture_contracts(&layout.cuda_architecture_components(), requested) {
        return Ok(Some(reason));
    }

    for (cache_key, expected_dir) in [
        ("rmm_DIR", layout.cudf_install_dir.join("lib/cmake/rmm")),
        ("rapids_logger_DIR", layout.cudf_install_dir.join("lib/cmake/rapids_logger")),
        ("CUGRAPH_CUDF_RUST_INCLUDE_DIR", layout.cudf_install_dir.join("include")),
        ("cuvs_DIR", layout.cuvs.cmake_config_dir.clone()),
    ] {
        let Some(cached) = cmake_cache_value(&cache, cache_key) else {
            return Ok(Some(format!("{cache_key} is missing from {}", cache_path.display())));
        };
        if !paths_equivalent(Path::new(cached), &expected_dir) {
            return Ok(Some(format!("{cache_key} changed from '{}' to '{}'", cached, expected_dir.display())));
        }
    }

    Ok(None)
}

fn validate_cmake_cache_package_dir(build_dir: &Path, cache_key: &str, expected_dir: &Path) -> Result<(), String> {
    let cache_path = build_dir.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path)
        .map_err(|error| format!("Failed to read cuGraph native CMake cache '{}': {error}", cache_path.display()))?;
    let resolved = cmake_cache_value(&cache, cache_key).ok_or_else(|| {
        format!(
            "The cuGraph native CMake cache '{}' does not contain {cache_key}. Reconfigure native cuGraph with the \
             canonical external RAPIDS package prefixes.",
            cache_path.display()
        )
    })?;
    let resolved = Path::new(resolved);
    if paths_equivalent(resolved, expected_dir) {
        return Ok(());
    }

    Err(format!(
        "The cuGraph native build resolved {cache_key}='{}' instead of '{}'. Reconfigure native cuGraph with \
         CUDF_INSTALL_DIR and CUVS_INSTALL_DIR pointing at the canonical prefixes.",
        resolved.display(),
        expected_dir.display()
    ))
}

fn reject_stale_private_runtime_artifacts(build_dir: &Path) -> Result<(), String> {
    let stale_artifacts = stale_private_runtime_artifacts(build_dir);
    if stale_artifacts.is_empty() {
        return Ok(());
    }

    Err(format!(
        "Stale private RMM/rapids_logger/cuVS artifacts remain under '{}':\n{}\nRemove the stale build tree and rerun \
         cargo build so the loader cannot resolve a private RAPIDS runtime copy.",
        build_dir.join("_deps").display(),
        stale_artifacts.iter().map(|path| format!("  {}", path.display())).collect::<Vec<_>>().join("\n")
    ))
}

fn stale_private_runtime_artifacts(build_dir: &Path) -> Vec<PathBuf> {
    let deps_dir = build_dir.join("_deps");
    let mut artifacts = Vec::new();
    let mut stack = vec![(deps_dir, 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                if depth < 3 {
                    stack.push((path, depth + 1));
                }
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("librmm.so")
                || name.starts_with("librapids_logger.so")
                || name.starts_with("libcuvs.so")
            {
                artifacts.push(path);
            }
        }
    }
    artifacts.sort();
    artifacts
}

pub(crate) fn validate_cugraph_cuvs_loader(
    native_link_dirs: &[PathBuf],
    runtime_lib_dirs: &[PathBuf],
    cuvs: &CanonicalCuvsContract,
) -> Result<ValidatedCuvsLoader, String> {
    let cugraph_library_selector_path =
        native_link_dirs.iter().map(|dir| dir.join("libcugraph.so")).find(|path| path.is_file()).ok_or_else(|| {
            format!(
                "Could not locate libcugraph.so in the resolved native link directories: {}.",
                display_paths(native_link_dirs)
            )
        })?;
    let cugraph_library_path = cugraph_library_selector_path.canonicalize().map_err(|error| {
        format!(
            "Failed to resolve canonical libcugraph identity '{}': {error}",
            cugraph_library_selector_path.display()
        )
    })?;

    let readelf_output = command_stdout(
        "inspect libcugraph ELF dependencies",
        Command::new("readelf").arg("-d").arg(&cugraph_library_selector_path),
    )?;
    let loader_search_path =
        controlled_loader_search_path(runtime_lib_dirs, &cuvs.runtime_lib_dir, env::var_os("LD_LIBRARY_PATH"))?;
    let ldd_output = command_stdout(
        "resolve libcugraph loader closure",
        Command::new("ldd").arg(&cugraph_library_selector_path).env("LD_LIBRARY_PATH", &loader_search_path),
    )?;
    validate_cuvs_loader_outputs(&readelf_output, &ldd_output, &cuvs.library_path)?;
    let ambient_ldd_output = command_stdout(
        "inspect ambient libcugraph loader resolution",
        Command::new("ldd").arg(&cugraph_library_selector_path),
    )?;
    validate_ambient_cuvs_resolution(&ambient_ldd_output, &cuvs.library_path)?;
    Ok(ValidatedCuvsLoader { cugraph_library_selector_path, cugraph_library_path })
}

pub(crate) fn controlled_loader_search_path(
    runtime_lib_dirs: &[PathBuf],
    cuvs_runtime_lib_dir: &Path,
    ambient_loader_path: Option<OsString>,
) -> Result<OsString, String> {
    let mut paths = Vec::new();
    for path in runtime_lib_dirs {
        if !paths.iter().any(|existing| existing == path) {
            paths.push(path.clone());
        }
    }
    if !paths.iter().any(|existing| existing == cuvs_runtime_lib_dir) {
        paths.push(cuvs_runtime_lib_dir.to_path_buf());
    }
    if let Some(ambient) = ambient_loader_path {
        for path in env::split_paths(&ambient) {
            if !path.as_os_str().is_empty() && !paths.iter().any(|existing| existing == &path) {
                paths.push(path);
            }
        }
    }
    env::join_paths(&paths).map_err(|error| format!("Failed to construct controlled LD_LIBRARY_PATH: {error}"))
}

pub(crate) fn validate_ambient_cuvs_resolution(ldd_output: &str, expected_library_path: &Path) -> Result<(), String> {
    if contains_private_cuvs_path(ldd_output) {
        return Err("The ambient dynamic loader resolves libcuvs from a private _deps cuVS build.".to_string());
    }
    let resolved_cuvs = ldd_output.lines().filter_map(ldd_resolved_cuvs_path).collect::<Vec<_>>();
    if resolved_cuvs.is_empty() {
        return Ok(());
    }
    if resolved_cuvs.len() != 1 {
        return Err(format!(
            "The ambient loader must resolve at most one libcuvs.so identity, but ldd reported {}.",
            resolved_cuvs.len()
        ));
    }
    let resolved = resolved_cuvs[0].canonicalize().map_err(|error| {
        format!(
            "Failed to canonicalize the ambient libcuvs path reported by ldd '{}': {error}",
            resolved_cuvs[0].display()
        )
    })?;
    let expected = expected_library_path.canonicalize().map_err(|error| {
        format!(
            "Failed to canonicalize the expected canonical libcuvs path '{}': {error}",
            expected_library_path.display()
        )
    })?;
    if resolved != expected {
        return Err(format!(
            "The ambient dynamic loader shadows canonical libcuvs '{}' with '{}'. Remove the conflicting \
             LD_LIBRARY_PATH entry.",
            expected.display(),
            resolved.display()
        ));
    }
    Ok(())
}

pub(crate) fn validate_cuvs_loader_outputs(
    readelf_output: &str,
    ldd_output: &str,
    expected_library_path: &Path,
) -> Result<PathBuf, String> {
    let needed_cuvs = readelf_output
        .lines()
        .filter(|line| {
            line.contains("(NEEDED)") && elf_needed_name(line).is_some_and(|name| name.starts_with("libcuvs.so"))
        })
        .count();
    if needed_cuvs != 1 {
        return Err(format!(
            "libcugraph must declare exactly one libcuvs.so ELF dependency, but readelf reported {needed_cuvs}."
        ));
    }
    if contains_private_cuvs_path(readelf_output) {
        return Err("libcugraph contains an RPATH/RUNPATH into a private _deps cuVS build.".to_string());
    }

    let unresolved = ldd_output.lines().filter(|line| line.contains("=> not found")).collect::<Vec<_>>();
    if !unresolved.is_empty() {
        return Err(format!(
            "libcugraph has unresolved loader dependencies:\n{}",
            unresolved.iter().map(|line| format!("  {}", line.trim())).collect::<Vec<_>>().join("\n")
        ));
    }
    if contains_private_cuvs_path(ldd_output) {
        return Err("The dynamic loader resolved libcuvs from a private _deps cuVS build.".to_string());
    }

    let resolved_cuvs = ldd_output.lines().filter_map(ldd_resolved_cuvs_path).collect::<Vec<_>>();
    if resolved_cuvs.len() != 1 {
        return Err(format!(
            "The libcugraph loader closure must resolve exactly one libcuvs.so, but ldd reported {}.",
            resolved_cuvs.len()
        ));
    }
    let resolved = resolved_cuvs[0].canonicalize().map_err(|error| {
        format!("Failed to canonicalize the libcuvs path reported by ldd '{}': {error}", resolved_cuvs[0].display())
    })?;
    let expected = expected_library_path.canonicalize().map_err(|error| {
        format!(
            "Failed to canonicalize the expected canonical libcuvs path '{}': {error}",
            expected_library_path.display()
        )
    })?;
    if resolved != expected {
        return Err(format!(
            "The dynamic loader resolved libcuvs to '{}' instead of canonical CUVS_INSTALL_DIR library '{}'.",
            resolved.display(),
            expected.display()
        ));
    }

    Ok(resolved)
}

fn elf_needed_name(line: &str) -> Option<&str> {
    let start = line.find("Shared library: [")? + "Shared library: [".len();
    let remainder = &line[start..];
    let end = remainder.find(']')?;
    Some(&remainder[..end])
}

fn contains_private_cuvs_path(output: &str) -> bool {
    output.lines().any(|line| {
        let normalized = line.replace('\\', "/");
        normalized.match_indices("/_deps/").any(|(start, marker)| {
            let tail = &normalized[start + marker.len()..];
            let path_end = tail.find([':', ']', ' ', '\t']).unwrap_or(tail.len());
            tail[..path_end].contains("cuvs")
        })
    })
}

fn ldd_resolved_cuvs_path(line: &str) -> Option<PathBuf> {
    let trimmed = line.trim();
    let (name, resolution) = trimmed.split_once("=>")?;
    if !name.trim().starts_with("libcuvs.so") {
        return None;
    }
    let path = resolution.split_whitespace().next()?;
    if path == "not" {
        return None;
    }
    Some(PathBuf::from(path))
}

fn command_stdout(action: &str, command: &mut Command) -> Result<String, String> {
    let command_display = format_command(command);
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to {action} using `{command_display}`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Failed to {action} using `{command_display}`; status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn run_command_capture(action: &str, command: &mut Command) -> Result<(), String> {
    let command_display = format_command(command);
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to {action} using `{command_display}`: {error}"))?;

    if output.status.success() {
        return Ok(());
    }

    Err(format!(
        "Failed to {action} using `{command_display}`; status {}.\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

pub(crate) fn parse_cmake_generated_path_list(contents: &str) -> Result<Vec<PathBuf>, String> {
    let mut resolved = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        push_unique_path(&mut resolved, normalize_existing_dir(Path::new(trimmed))?);
    }

    Ok(resolved)
}

pub(crate) fn parse_cmake_link_command(command: &str) -> Result<ParsedLinkCommand, String> {
    let mut link_dirs = Vec::new();
    let mut runtime_dirs = Vec::new();

    for token in split_shell_words(command)? {
        if let Some(raw_paths) = token.strip_prefix("-Wl,-rpath,") {
            for path in env::split_paths(&OsString::from(raw_paths)) {
                push_unique_path(&mut runtime_dirs, normalize_existing_dir(&path)?);
            }
            continue;
        }

        if let Some(raw_path) = token.strip_prefix("-L") {
            push_unique_path(&mut link_dirs, normalize_existing_dir(Path::new(raw_path))?);
            continue;
        }

        let path = Path::new(&token);
        if path.is_absolute()
            && looks_like_library_path(path)
            && let Some(parent) = path.parent()
        {
            push_unique_path(&mut link_dirs, normalize_existing_dir(parent)?);
        }
    }

    Ok(ParsedLinkCommand { link_dirs, runtime_dirs })
}

pub(crate) fn has_required_cugraph_libs(dir: &Path) -> bool {
    has_library(dir, "cugraph") && has_library(dir, "cugraph_c")
}

fn resolve_native_build_contract_from_cmake_export(cmake_build_dir: &Path) -> Result<NativeBuildContract, String> {
    let probe_dir = create_probe_dir()?;
    let result = resolve_native_build_contract_from_cmake_export_inner(cmake_build_dir, &probe_dir);
    let _ = fs::remove_dir_all(&probe_dir);
    result
}

fn resolve_native_build_contract_from_cmake_export_inner(
    cmake_build_dir: &Path,
    probe_dir: &Path,
) -> Result<NativeBuildContract, String> {
    write_contract_probe_files(cmake_build_dir, probe_dir)?;
    let package_prefixes = cmake_package_prefixes_from_cache(cmake_build_dir)?;
    run_cmake_probe(probe_dir, &package_prefixes)?;

    let include_output = fs::read_to_string(probe_dir.join("build/includes.txt"))
        .map_err(|error| format!("Failed to read generated include metadata from CMake probe: {error}"))?;
    let includes = parse_cmake_generated_path_list(&include_output)?;

    let link_command = find_cmake_link_command(&probe_dir.join("build"))?;
    let parsed_link = parse_cmake_link_command(&link_command)?;

    Ok(NativeBuildContract {
        native_include_dirs: resolve_native_include_dirs(Some(
            env::join_paths(&includes).expect("CMake probe include paths must be valid OS strings"),
        ))?,
        native_link_dirs: resolve_native_link_dirs(Some(
            env::join_paths(&parsed_link.link_dirs).expect("CMake probe link paths must be valid OS strings"),
        ))?,
        runtime_lib_dirs: resolve_runtime_lib_dirs(Some(
            env::join_paths(&parsed_link.runtime_dirs).expect("CMake probe runtime paths must be valid OS strings"),
        ))?,
    })
}

fn validate_cmake_export_dir(path: &Path, env_name: &str) -> Result<PathBuf, String> {
    let canonical = path.canonicalize().map_err(|_| {
        format!(
            "{env_name}='{}' does not exist or is not accessible. Expected a directory containing cugraph-config.cmake.",
            path.display()
        )
    })?;

    if !canonical.join("cugraph-config.cmake").is_file() {
        return Err(format!(
            "{env_name}='{}' is invalid. Expected '{}/cugraph-config.cmake' to exist.",
            path.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
}

fn resolve_required_path_list(env_name: &str, env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let Some(raw_paths) = env_val else {
        return Err(format!(
            "{env_name} is required. Set it to a non-empty OS-native path list that describes the native cuGraph contract."
        ));
    };

    let mut resolved = Vec::new();
    for raw_path in env::split_paths(&raw_paths) {
        let path = raw_path.as_path();
        if path.as_os_str().is_empty() {
            continue;
        }
        push_unique_path(&mut resolved, normalize_existing_dir(path)?);
    }

    if resolved.is_empty() {
        return Err(format!(
            "{env_name} is required. Set it to a non-empty OS-native path list that describes the native cuGraph contract."
        ));
    }

    Ok(resolved)
}

fn ensure_contains_cugraph_libraries(env_name: &str, dirs: &[PathBuf]) -> Result<(), String> {
    if dirs.iter().any(|dir| has_required_cugraph_libs(dir)) {
        Ok(())
    } else {
        Err(format!(
            "{env_name} is invalid. Expected at least one directory containing both libcugraph and libcugraph_c shared libraries. Provided paths: {}.",
            display_paths(dirs)
        ))
    }
}

pub(crate) fn cmake_package_prefixes_from_cache(cmake_build_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let cache_path = cmake_build_dir.join("CMakeCache.txt");
    let contents = match fs::read_to_string(&cache_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!("Failed to read cuGraph native CMake cache '{}': {error}", cache_path.display()));
        },
    };

    let mut prefixes = Vec::new();
    let mut seen = BTreeSet::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let Some((name_and_type, value)) = line.split_once('=') else {
            continue;
        };
        let Some((name, _type_name)) = name_and_type.split_once(':') else {
            continue;
        };
        if name != "CMAKE_PREFIX_PATH" && !name.ends_with("_ROOT") {
            continue;
        }

        for raw_path in value.split(';').map(str::trim).filter(|path| !path.is_empty()) {
            if raw_path.ends_with("-NOTFOUND") {
                continue;
            }
            let path = Path::new(raw_path);
            if !path.is_dir() {
                continue;
            }
            let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if seen.insert(canonical.clone()) {
                prefixes.push(canonical);
            }
        }
    }

    Ok(prefixes)
}

fn cmake_cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    for line in cache.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let Some((name_and_type, value)) = line.split_once('=') else {
            continue;
        };
        let Some((name, _type_name)) = name_and_type.split_once(':') else {
            continue;
        };
        if name == key {
            return Some(value.trim());
        }
    }
    None
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = left.canonicalize().unwrap_or_else(|_| left.to_path_buf());
    let right = right.canonicalize().unwrap_or_else(|_| right.to_path_buf());
    left == right
}

fn create_probe_dir() -> Result<PathBuf, String> {
    let unique = format!(
        "cugraph-cmake-probe-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    );
    let probe_dir = env::temp_dir().join(unique);
    fs::create_dir_all(&probe_dir).map_err(|error| {
        format!(
            "Failed to create temporary directory for the cuGraph CMake metadata probe at '{}': {error}",
            probe_dir.display()
        )
    })?;
    Ok(probe_dir)
}

fn write_contract_probe_files(cmake_build_dir: &Path, probe_dir: &Path) -> Result<(), String> {
    let cmake_lists = format!(
        r#"cmake_minimum_required(VERSION 3.30)
project(cugraph_contract_probe LANGUAGES CXX CUDA)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(cugraph CONFIG REQUIRED PATHS "{build_dir}" NO_DEFAULT_PATH)
find_package(rmm CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
add_executable(contract_probe probe.cpp)
target_link_libraries(contract_probe PRIVATE cugraph::cugraph_c rmm::rmm raft::raft)
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/includes.txt" CONTENT "$<JOIN:$<TARGET_PROPERTY:contract_probe,INCLUDE_DIRECTORIES>,\n>")
"#,
        build_dir = cmake_build_dir.display(),
    );
    // Probe representative cugraph_c/RAFT/RMM headers. per_device_resource.hpp
    // exists in both RMM 26.04 and 26.06, unlike the removed polymorphic
    // device_memory_resource.hpp.
    let probe_cpp = r#"#include <cugraph_c/resource_handle.h>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <cuda_runtime_api.h>
int main() { return 0; }
"#;

    fs::write(probe_dir.join("CMakeLists.txt"), cmake_lists)
        .map_err(|error| format!("Failed to write temporary CMakeLists.txt for the cuGraph metadata probe: {error}"))?;
    fs::write(probe_dir.join("probe.cpp"), probe_cpp)
        .map_err(|error| format!("Failed to write temporary probe.cpp for the cuGraph metadata probe: {error}"))?;

    Ok(())
}

fn run_cmake_probe(probe_dir: &Path, package_prefixes: &[PathBuf]) -> Result<(), String> {
    let mut configure_command = Command::new("cmake");
    configure_command.arg("-S").arg(probe_dir).arg("-B").arg(probe_dir.join("build")).arg("-G").arg("Unix Makefiles");
    if !package_prefixes.is_empty() {
        configure_command.arg(format!("-DCMAKE_PREFIX_PATH={}", cmake_list(package_prefixes)));
    }

    let configure_output = configure_command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to run cmake while probing the cuGraph CMake export: {error}"))?;

    if !configure_output.status.success() {
        return Err(format!(
            "Failed to configure the cuGraph CMake metadata probe.\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&configure_output.stdout),
            String::from_utf8_lossy(&configure_output.stderr)
        ));
    }

    let build_output = Command::new("cmake")
        .arg("--build")
        .arg(probe_dir.join("build"))
        .arg("--target")
        .arg("contract_probe")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to build the cuGraph CMake metadata probe: {error}"))?;

    if !build_output.status.success() {
        return Err(format!(
            "The cuGraph CMake export did not provide a complete native closure for the probe target.\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&build_output.stdout),
            String::from_utf8_lossy(&build_output.stderr)
        ));
    }

    Ok(())
}

fn cmake_list(paths: &[PathBuf]) -> String {
    paths.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(";")
}

fn find_cmake_link_command(build_dir: &Path) -> Result<String, String> {
    let mut stack = vec![build_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|error| {
            format!("Failed to inspect generated CMake build directory '{}': {error}", dir.display())
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("Failed to inspect entry under '{}': {error}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().is_some_and(|name| name == "link.txt") {
                return fs::read_to_string(&path).map_err(|error| {
                    format!("Failed to read generated CMake link command from '{}': {error}", path.display())
                });
            }
        }
    }

    Err(format!("Failed to locate the generated CMake link command under '{}'.", build_dir.display()))
}

fn normalize_existing_dir(path: &Path) -> Result<PathBuf, String> {
    let canonical =
        path.canonicalize().map_err(|_| format!("Path '{}' does not exist or is not accessible.", path.display()))?;
    if !canonical.is_dir() {
        return Err(format!("Path '{}' is not a directory.", canonical.display()));
    }
    Ok(canonical)
}

fn normalize_existing_file(path: &Path) -> Result<PathBuf, String> {
    let canonical =
        path.canonicalize().map_err(|_| format!("Path '{}' does not exist or is not accessible.", path.display()))?;
    if !canonical.is_file() {
        return Err(format!("Path '{}' is not a file.", canonical.display()));
    }
    Ok(canonical)
}

fn looks_like_library_path(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    file_name.ends_with(".so")
        || file_name.contains(".so.")
        // Static archives (e.g. CUDA's `libcudart_static.a`) are linked by the
        // CMake export via absolute paths. Capture their directory so the
        // toolkit lib dir flows through the formal native-link metadata; the
        // same directory also holds the corresponding shared libraries that
        // the cugraph build script (`cugraph/build.rs`) links (e.g. `-lcudart`).
        || file_name.ends_with(".a")
}

fn has_library(dir: &Path, stem: &str) -> bool {
    if dir.join(format!("lib{stem}.so")).exists() {
        return true;
    }

    let versioned_prefix = format!("lib{stem}.so.");
    let Ok(entries) = dir.read_dir() else {
        return false;
    };

    entries.flatten().any(|entry| entry.file_name().to_str().is_some_and(|name| name.starts_with(&versioned_prefix)))
}

fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.push(candidate);
    }
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ")
}

fn format_command(command: &Command) -> String {
    let mut parts = vec![command.get_program().to_string_lossy().into_owned()];
    parts.extend(command.get_args().map(|arg| arg.to_string_lossy().into_owned()));
    parts.join(" ")
}

fn split_shell_words(input: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars();
    let mut quote = None;

    while let Some(ch) = chars.next() {
        match quote {
            Some(quote_char) if ch == quote_char => quote = None,
            Some(_) => current.push(ch),
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            },
            None if ch == '\\' => {
                let Some(next) = chars.next() else {
                    return Err("Trailing escape in shell-style command string.".to_string());
                };
                current.push(next);
            },
            None => current.push(ch),
        }
    }

    if quote.is_some() {
        return Err("Unterminated quote in shell-style command string.".to_string());
    }

    if !current.is_empty() {
        words.push(current);
    }

    Ok(words)
}

pub(crate) fn check_libcugraph_staleness(manifest_dir: &Path, native_link_dirs: &[PathBuf]) {
    if env::var_os("CUGRAPH_RS_SKIP_STALENESS_CHECK").is_some() {
        return;
    }

    let Some(libcugraph_so) = native_link_dirs.iter().map(|dir| dir.join("libcugraph.so")).find(|path| path.is_file())
    else {
        return;
    };
    let Ok(libcugraph_meta) = fs::metadata(&libcugraph_so) else { return };
    let Ok(libcugraph_mtime) = libcugraph_meta.modified() else { return };

    let cugraph_cpp = manifest_dir.join("../cugraph/cpp");
    let scan_dirs = [cugraph_cpp.join("src"), cugraph_cpp.join("include")];

    let mut newest: Option<(PathBuf, SystemTime)> = None;
    let mut total_newer = 0usize;
    for dir in &scan_dirs {
        if !dir.is_dir() {
            return;
        }
        let (dir_newest, count) = find_newest_cpp_source(dir, libcugraph_mtime);
        total_newer += count;
        if let Some(candidate) = dir_newest
            && newest.as_ref().is_none_or(|(_, time)| candidate.1 > *time)
        {
            newest = Some(candidate);
        }
    }

    if total_newer == 0 {
        return;
    }

    let display_path = newest.expect("a newer cugraph source requires a newest source path").0.display().to_string();
    println!(
        "cargo:warning=libcugraph appears stale: {total_newer} source file(s) under cugraph/cpp/{{src,include}} are newer than {}. Run `bash scripts/build/build_libcugraph.sh` to rebuild native cuGraph. (newest: {}) Set CUGRAPH_RS_SKIP_STALENESS_CHECK=1 to silence.",
        libcugraph_so.display(),
        display_path,
    );

    if env::var_os("CUGRAPH_RS_ENFORCE_STALENESS_CHECK").is_some()
        || env::var_os("CUGRAPH_ENFORCE_STALENESS_CHECK").is_some()
    {
        panic!(
            "\n{}\n[ERROR] libcugraph is stale: {total_newer} source file(s) under cugraph/cpp/{{src,include}} are newer than {}.\n\
             Run `bash scripts/build/build_libcugraph.sh` to rebuild native cuGraph, or set CUGRAPH_RS_SKIP_STALENESS_CHECK=1 to silence.\n{}\n",
            "=".repeat(80),
            libcugraph_so.display(),
            "=".repeat(80),
        );
    }
}

fn find_newest_cpp_source(dir: &Path, baseline: SystemTime) -> (Option<(PathBuf, SystemTime)>, usize) {
    let mut newest: Option<(PathBuf, SystemTime)> = None;
    let mut newer_than_baseline = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else { continue };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            let extension = path.extension().and_then(|extension| extension.to_str()).unwrap_or("");
            if !matches!(extension, "cpp" | "cu" | "hpp" | "cuh" | "h" | "c") {
                continue;
            }
            let Ok(metadata) = entry.metadata() else { continue };
            let Ok(mtime) = metadata.modified() else { continue };
            if mtime > baseline {
                newer_than_baseline += 1;
            }
            if newest.as_ref().is_none_or(|(_, time)| mtime > *time) {
                newest = Some((path, mtime));
            }
        }
    }
    (newest, newer_than_baseline)
}
