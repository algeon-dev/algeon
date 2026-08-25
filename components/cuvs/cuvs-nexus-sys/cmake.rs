use std::{
    collections::BTreeSet,
    env, fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result};
use cmake_package::{Error as CmakeError, Version, VersionError, find_package};

pub(crate) const NATIVE_CUVS_VERSION: &str = "26.8.1";
const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const CUVS_SUBMODULE_PATH: &str = "components/cuvs/cuvs";
const PYTHON_PRINT_LIBCUVS_PACKAGE_DIR: &str = r#"
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("libcuvs")
if spec is None or spec.submodule_search_locations is None:
    raise ModuleNotFoundError("libcuvs")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
"#;

pub(crate) struct CuvsMetadata {
    pub(crate) include_dirs: Vec<PathBuf>,
    pub(crate) native_link_dirs: Vec<PathBuf>,
    pub(crate) runtime_lib_dirs: Vec<PathBuf>,
    pub(crate) libcuvs_c: PathBuf,
    pub(crate) libcuvs: PathBuf,
    pub(crate) source_revision: String,
}

fn unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "CMake is unavailable or too old to inspect cuVS. Install CMake, then set CUVS_ROOT to the canonical cuVS 26.08 install prefix"
    )
}

fn package_error(error: CmakeError, required: &Version) -> anyhow::Error {
    match error {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => unavailable_error(),
        CmakeError::PackageNotFound => anyhow::anyhow!(
            "Could not find the cuVS {NATIVE_CUVS_VERSION} CMake c_api package. Build the pinned source with `scripts/build/build_libcuvs.sh`, set CUVS_ROOT to target/native/cuvs-install, or set CMAKE_PREFIX_PATH to a compatible installed package. cuGraph `_deps/cuvs-src` is not a supported fallback"
        ),
        CmakeError::Version(VersionError::VersionIncompatible(candidates)) => {
            let considered = candidates
                .iter()
                .map(|candidate| format!("- {} (version: {})", candidate.config, candidate.version))
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::anyhow!(
                "Found cuVS CMake packages, but none match required version {required} (same major/minor and not older):\n{considered}"
            )
        },
        CmakeError::Version(VersionError::InvalidVersion) => {
            anyhow::anyhow!("CMake reported an invalid cuVS package version")
        },
        CmakeError::IO(error) => anyhow::anyhow!("CMake package inspection failed: {error}"),
        CmakeError::Internal => anyhow::anyhow!("CMake package inspection failed internally"),
    }
}

fn canonical_existing_dir(path: impl AsRef<Path>, description: &str) -> Result<PathBuf> {
    let path = path.as_ref();
    anyhow::ensure!(path.is_dir(), "{description} does not exist or is not a directory: {}", path.display());
    fs::canonicalize(path).with_context(|| format!("failed to canonicalize {description} at {}", path.display()))
}

pub(crate) fn reject_private_dependency_path(path: &Path) -> Result<()> {
    let private = path.components().any(|component| matches!(component, Component::Normal(name) if name == "_deps"));
    anyhow::ensure!(
        !private,
        "cuVS resolved through a private `_deps` tree at '{}'. Build/install the pinned cuVS source and set CUVS_ROOT to that canonical prefix; cuGraph `_deps/cuvs-src` must never supply this crate",
        path.display()
    );
    Ok(())
}

pub(crate) fn require_explicit_root_selection(
    explicit_root: Option<&Path>,
    primary_include: &Path,
    libcuvs_c: &Path,
    libcuvs: &Path,
) -> Result<()> {
    let Some(explicit_root) = explicit_root else {
        return Ok(());
    };
    for (artifact, path) in [("cuVS headers", primary_include), ("libcuvs_c", libcuvs_c), ("libcuvs", libcuvs)] {
        anyhow::ensure!(
            path.starts_with(explicit_root),
            "CUVS_ROOT is '{}', but CMake selected {artifact} outside that explicit root: {}. Remove the conflicting default cuVS package or correct CUVS_ROOT",
            explicit_root.display(),
            path.display()
        );
    }
    Ok(())
}

fn resolve_shared_library(lib_dir: &Path, stem: &str) -> Result<PathBuf> {
    let base = format!("lib{stem}.so");
    let direct = lib_dir.join(&base);
    if direct.is_file() {
        return fs::canonicalize(&direct).with_context(|| format!("failed to canonicalize {}", direct.display()));
    }

    let mut candidates = BTreeSet::new();
    for entry in fs::read_dir(lib_dir).with_context(|| format!("failed to inspect {}", lib_dir.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(&format!("{base}.")) && entry.path().is_file() {
            candidates.insert(fs::canonicalize(entry.path())?);
        }
    }
    anyhow::ensure!(
        candidates.len() == 1,
        "expected exactly one canonical {base} implementation under {}, found {:?}",
        lib_dir.display(),
        candidates
    );
    Ok(candidates.into_iter().next().expect("one candidate"))
}

fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn cuvs_gitlink(workspace_root: &Path) -> Option<String> {
    for (args, sha_field) in
        [(["ls-files", "--stage", "--", CUVS_SUBMODULE_PATH], 1), (["ls-tree", "HEAD", "--", CUVS_SUBMODULE_PATH], 2)]
    {
        let Ok(output) = Command::new("git").arg("-C").arg(workspace_root).args(args).output() else {
            return None;
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let fields = stdout.split_whitespace().collect::<Vec<_>>();
        if fields.len() == 4 && fields[0] == "160000" && is_git_sha(fields[sha_field]) {
            return Some(fields[sha_field].to_owned());
        }
    }
    None
}

fn repository_root_from_manifest(manifest_dir: &Path) -> Option<PathBuf> {
    manifest_dir.parent()?.parent()?.parent().map(Path::to_path_buf)
}

/// The `cuvs` submodule gitlink recorded in the superproject is the source
/// pin. Build scripts must not hard-code a SHA, so the pin is resolved from
/// git at build time and falls back to the marker's recorded identity only
/// when the superproject is unavailable (e.g. a non-git source copy).
fn recorded_cuvs_gitlink() -> Option<String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")?;
    let workspace_root = repository_root_from_manifest(Path::new(&manifest_dir))?;
    cuvs_gitlink(&workspace_root)
}

pub(crate) fn recorded_cuvs_gitlink_input() -> Option<PathBuf> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")?;
    let workspace_root = repository_root_from_manifest(Path::new(&manifest_dir))?;
    let output = Command::new("git")
        .arg("-C")
        .arg(&workspace_root)
        .args(["rev-parse", "--path-format=absolute", "--git-path", "index"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    path.is_file().then_some(path)
}

fn source_revision(prefix: Option<&Path>, lib_dir: &Path) -> Result<String> {
    let inferred_prefix = lib_dir.parent().filter(|path| {
        matches!(lib_dir.file_name().and_then(|name| name.to_str()), Some("lib" | "lib64")) && path.is_dir()
    });
    let prefix = prefix.or(inferred_prefix);
    let marker = prefix.map(|path| path.join("share/cuvs-nexus/source-revision"));
    if let Some(marker) = marker.filter(|path| path.is_file()) {
        let revision = fs::read_to_string(&marker)
            .with_context(|| format!("failed to read source revision marker {}", marker.display()))?;
        let revision = revision.trim();
        anyhow::ensure!(
            is_git_sha(revision),
            "canonical cuVS prefix reports an invalid source revision {revision:?}: {}",
            marker.display()
        );
        if let Some(expected) = recorded_cuvs_gitlink() {
            anyhow::ensure!(
                revision == expected,
                "canonical cuVS prefix reports source revision {revision}, but the superproject records cuVS gitlink {expected}. Rebuild the canonical prefix with `scripts/build/build_libcuvs.sh` from the recorded submodule: {}",
                marker.display()
            );
        }
        return Ok(revision.to_owned());
    }

    if let Some(prefix) = prefix.filter(|_| std::env::var_os("CUVS_ROOT").is_some()) {
        anyhow::bail!(
            "CUVS_ROOT points to '{}', but share/cuvs-nexus/source-revision is missing. Build the canonical prefix with `scripts/build/build_libcuvs.sh`; do not substitute an untracked native build",
            prefix.display()
        );
    }
    Ok("external-compatible-package".to_owned())
}

fn canonical_runtime_dir(prefix: Option<&Path>, lib_dir: &Path) -> Result<Option<PathBuf>> {
    let inferred_prefix = lib_dir.parent().filter(|path| {
        matches!(lib_dir.file_name().and_then(|name| name.to_str()), Some("lib" | "lib64")) && path.is_dir()
    });
    let prefix = prefix.or(inferred_prefix);
    let marker = prefix.map(|path| path.join("share/cuvs-nexus/rmm-runtime-lib"));
    if let Some(marker) = marker.filter(|path| path.is_file()) {
        let runtime_lib = fs::read_to_string(&marker)
            .with_context(|| format!("failed to read RMM runtime marker {}", marker.display()))?;
        let runtime_lib = PathBuf::from(runtime_lib.trim());
        anyhow::ensure!(
            runtime_lib.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("librmm.so")),
            "canonical cuVS RMM marker does not identify librmm.so: {}",
            runtime_lib.display()
        );
        let runtime_lib = fs::canonicalize(&runtime_lib)
            .with_context(|| format!("canonical cuVS RMM marker is stale: {}", runtime_lib.display()))?;
        reject_private_dependency_path(&runtime_lib)?;
        return Ok(Some(runtime_lib.parent().expect("RMM library has a parent directory").to_owned()));
    }

    if let Some(prefix) = prefix.filter(|_| std::env::var_os("CUVS_ROOT").is_some()) {
        anyhow::bail!(
            "CUVS_ROOT points to '{}', but share/cuvs-nexus/rmm-runtime-lib is missing. Rebuild the canonical prefix with `scripts/build/build_libcuvs.sh` so its RMM runtime identity is explicit",
            prefix.display()
        );
    }
    Ok(None)
}

fn find_python_package_dir() -> Result<PathBuf> {
    let python = Path::new(if std::env::var_os("VIRTUAL_ENV").is_some() { "python" } else { "python3" });
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_PRINT_LIBCUVS_PACKAGE_DIR)
        .output()
        .with_context(|| format!("LIBCUVS_USE_PYTHON is set, but failed to run {python:?}"))?;
    anyhow::ensure!(
        output.status.success(),
        "LIBCUVS_USE_PYTHON is set, but {python:?} could not locate libcuvs: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let package_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    anyhow::ensure!(
        package_dir.join("lib64/cmake/cuvs").is_dir(),
        "Python libcuvs package at {} does not contain lib64/cmake/cuvs",
        package_dir.display()
    );
    Ok(package_dir)
}

pub(crate) fn locate_cuvs() -> Result<CuvsMetadata> {
    let required: Version = NATIVE_CUVS_VERSION.try_into().expect("valid native cuVS version");
    let explicit_root = std::env::var_os("CUVS_ROOT")
        .map(|path| canonical_existing_dir(PathBuf::from(path), "CUVS_ROOT"))
        .transpose()?;
    let python_root = std::env::var_os("LIBCUVS_USE_PYTHON").map(|_| find_python_package_dir()).transpose()?;
    anyhow::ensure!(
        !(explicit_root.is_some() && python_root.is_some()),
        "CUVS_ROOT and LIBCUVS_USE_PYTHON are mutually exclusive native package selectors"
    );

    let mut query = find_package("cuvs")
        .version(required)
        .components([CUVS_COMPONENT.to_owned()])
        .define("CMAKE_FIND_PACKAGE_PREFER_CONFIG", "TRUE");
    if let Some(prefix) = explicit_root.as_ref().or(python_root.as_ref()) {
        query = query.prefix_paths(vec![prefix.clone()]);
    }
    let package = query.find().map_err(|error| package_error(error, &required))?;
    let target = package.target(CUVS_C_API_TARGET).with_context(|| {
        format!("cuVS package was found, but it did not export required target {CUVS_C_API_TARGET}")
    })?;

    let mut include_dirs = target
        .include_directories
        .iter()
        .map(|path| canonical_existing_dir(path, "cuVS include directory"))
        .collect::<Result<Vec<_>>>()?;
    include_dirs.sort();
    include_dirs.dedup();
    let primary_include = include_dirs
        .iter()
        .find(|path| path.join("cuvs/core/c_api.h").is_file())
        .context("cuVS CMake target did not expose the C API include directory")?;
    reject_private_dependency_path(primary_include)?;

    let raw_lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .or_else(|| target.link_directories.first().map(Path::new))
        .context("cuVS CMake target did not expose a library location")?;
    let lib_dir = canonical_existing_dir(raw_lib_dir, "cuVS library directory")?;
    reject_private_dependency_path(&lib_dir)?;
    let libcuvs_c = resolve_shared_library(&lib_dir, "cuvs_c")?;
    let libcuvs = resolve_shared_library(&lib_dir, "cuvs")?;
    require_explicit_root_selection(explicit_root.as_deref(), primary_include, &libcuvs_c, &libcuvs)?;

    let mut native_link_dirs = target
        .link_directories
        .iter()
        .filter_map(|path| canonical_existing_dir(path, "cuVS link directory").ok())
        .collect::<Vec<_>>();
    native_link_dirs.push(lib_dir.clone());
    native_link_dirs.sort();
    native_link_dirs.dedup();
    for path in &native_link_dirs {
        reject_private_dependency_path(path)?;
    }

    let mut runtime_lib_dirs = native_link_dirs.clone();
    if let Some(runtime_dir) = canonical_runtime_dir(explicit_root.as_deref(), &lib_dir)? {
        runtime_lib_dirs.push(runtime_dir);
        runtime_lib_dirs.sort();
        runtime_lib_dirs.dedup();
    }
    let source_revision = source_revision(explicit_root.as_deref(), &lib_dir)?;
    target.link();

    Ok(CuvsMetadata { include_dirs, native_link_dirs, runtime_lib_dirs, libcuvs_c, libcuvs, source_revision })
}
