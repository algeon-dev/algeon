//! Root, sibling-repository, cuVS, and toolchain provenance capture.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::Serialize;

use super::{
    MAX_TOOL_VERSION_BYTES,
    util::{sha256_bytes, sha256_file},
};
use crate::ToolResult;

#[derive(Debug, Serialize)]
pub(super) struct Provenance {
    pub(super) repositories: RepositoryProvenanceSet,
    pub(super) cuvs_build_inputs: CuvsBuildInputs,
    pub(super) toolchains: ToolchainVersions,
    pub(super) runner: RunnerIdentity,
}

impl Provenance {
    pub(super) fn collect() -> ToolResult<Self> {
        let workspace_root = workspace_root()?;
        let sibling_root = workspace_root.parent().ok_or("workspace root has no parent for sibling provenance")?;
        let cudf_root = sibling_root.join("cudf-nexus");
        let cugraph_root = sibling_root.join("cugraph-nexus");
        Ok(Self {
            repositories: RepositoryProvenanceSet {
                root: RepositoryProvenance::collect("datafusion-nexus", &workspace_root)?,
                cudf_nexus: RepositoryProvenance::collect("cudf-nexus", &cudf_root)?,
                cugraph_nexus: RepositoryProvenance::collect("cugraph-nexus", &cugraph_root)?,
            },
            cuvs_build_inputs: CuvsBuildInputs::collect()?,
            toolchains: ToolchainVersions::collect()?,
            runner: RunnerIdentity::collect()?,
        })
    }
}

pub(super) fn workspace_root() -> ToolResult<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root =
        manifest_dir.parent().and_then(Path::parent).ok_or("cannot derive workspace root from nexus-tools manifest")?;
    fs::canonicalize(workspace_root).map_err(|_| "cannot canonicalize workspace root".into())
}

#[derive(Debug, Serialize)]
pub(super) struct RepositoryProvenanceSet {
    pub(super) root: RepositoryProvenance,
    pub(super) cudf_nexus: RepositoryProvenance,
    pub(super) cugraph_nexus: RepositoryProvenance,
}

#[derive(Debug, Serialize)]
pub(super) struct RepositoryProvenance {
    pub(super) name: &'static str,
    pub(super) path: String,
    pub(super) revision: String,
    pub(super) dirty_manifest: DirtyManifest,
}

impl RepositoryProvenance {
    pub(super) fn collect(name: &'static str, path: &Path) -> ToolResult<Self> {
        let canonical_path = fs::canonicalize(path).map_err(|_| format!("cannot canonicalize {name} repository"))?;
        let revision = git_output(&canonical_path, ["rev-parse", "HEAD"], name)?;
        let dirty_manifest = DirtyManifest::collect(&canonical_path, name)?;
        Ok(Self { name, path: canonical_path.display().to_string(), revision, dirty_manifest })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct DirtyManifest {
    pub(super) format: &'static str,
    pub(super) sha256: String,
    pub(super) entries: Vec<String>,
}

impl DirtyManifest {
    pub(super) fn collect(repo: &Path, name: &str) -> ToolResult<Self> {
        let output = git_output_bytes(
            repo,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
            name,
        )?;
        let entries = output
            .split(|byte| *byte == b'\0')
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                std::str::from_utf8(entry)
                    .map(str::to_string)
                    .map_err(|_| format!("{name} dirty manifest contains non-UTF-8 paths"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { format: "git-status-porcelain-v1-z", sha256: sha256_bytes(&output), entries })
    }
}

pub(super) fn git_output<const N: usize>(repo: &Path, args: [&str; N], name: &str) -> ToolResult<String> {
    let output = git_output_bytes(repo, args, name)?;
    String::from_utf8(output)
        .map(|value| value.trim().to_string())
        .map_err(|_| format!("{name} git output is not UTF-8").into())
}

pub(super) fn git_output_bytes<const N: usize>(repo: &Path, args: [&str; N], name: &str) -> ToolResult<Vec<u8>> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .map_err(|_| format!("cannot execute git for {name}"))?;
    if !output.status.success() {
        return Err(format!("git command failed for {name}").into());
    }
    Ok(output.stdout)
}

#[derive(Debug, Serialize)]
pub(super) struct CuvsBuildInputs {
    pub(super) install_dir: String,
    pub(super) cmake_config: FileIdentity,
    pub(super) libcuvs_selector: FileIdentity,
    pub(super) libcuvs_resolved: FileIdentity,
    pub(super) version: String,
}

impl CuvsBuildInputs {
    pub(super) fn collect() -> ToolResult<Self> {
        let install_dir = env::var_os("CUVS_INSTALL_DIR")
            .map(PathBuf::from)
            .ok_or("CUVS_INSTALL_DIR is required for production-validation provenance")?;
        let install_dir = fs::canonicalize(&install_dir).map_err(|_| "cannot canonicalize CUVS_INSTALL_DIR")?;
        let cmake_config_path = install_dir.join("lib/cmake/cuvs/cuvs-config.cmake");
        let cmake_config = FileIdentity::collect(&cmake_config_path, "cuvs-config.cmake")?;
        let selector_path = [install_dir.join("lib/libcuvs.so"), install_dir.join("lib64/libcuvs.so")]
            .into_iter()
            .find(|path| path.is_file())
            .ok_or("CUVS_INSTALL_DIR must contain lib/libcuvs.so or lib64/libcuvs.so")?;
        let libcuvs_selector = FileIdentity::collect(&selector_path, "libcuvs.so selector")?;
        let resolved_path = fs::canonicalize(&selector_path).map_err(|_| "cannot resolve selected libcuvs.so")?;
        if !resolved_path.starts_with(&install_dir) {
            return Err("selected libcuvs.so resolves outside CUVS_INSTALL_DIR".into());
        }
        let libcuvs_resolved = FileIdentity::collect(&resolved_path, "resolved libcuvs.so")?;
        let config_bytes = fs::read(&cmake_config_path).map_err(|_| "cannot read cuvs-config.cmake")?;
        let version = cmake_set_value(&config_bytes, "CUVS_VERSION").ok_or("cuvs-config.cmake has no CUVS_VERSION")?;
        Ok(Self {
            install_dir: install_dir.display().to_string(),
            cmake_config,
            libcuvs_selector,
            libcuvs_resolved,
            version,
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct FileIdentity {
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
}

impl FileIdentity {
    pub(super) fn collect(path: &Path, label: &str) -> ToolResult<Self> {
        let metadata = fs::metadata(path).map_err(|_| format!("cannot stat {label}"))?;
        if !metadata.is_file() {
            return Err(format!("{label} is not a regular file").into());
        }
        Ok(Self { path: path.display().to_string(), size_bytes: metadata.len(), sha256: sha256_file(path)? })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ToolchainVersions {
    pub(super) rustc: String,
    pub(super) cargo: String,
    pub(super) cuda_toolkit: String,
    pub(super) cuda_driver: String,
    pub(super) nsight_systems: String,
    pub(super) rapids: RapidsVersions,
    pub(super) build_profile: &'static str,
}

impl ToolchainVersions {
    pub(super) fn collect() -> ToolResult<Self> {
        Ok(Self {
            rustc: required_command_version("rustc", ["-Vv"])?,
            cargo: required_command_version("cargo", ["-V"])?,
            cuda_toolkit: required_command_version("nvcc", ["--version"])?,
            cuda_driver: required_command_version(
                "nvidia-smi",
                ["--query-gpu=driver_version", "--format=csv,noheader"],
            )?,
            nsight_systems: required_command_version("nsys", ["--version"])?,
            rapids: RapidsVersions::collect()?,
            build_profile: if cfg!(debug_assertions) { "debug" } else { "release" },
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct RapidsVersions {
    pub(super) cudf: CmakePackageVersion,
    pub(super) cugraph: SourcePackageVersion,
    pub(super) cuvs: CmakePackageVersion,
}

impl RapidsVersions {
    pub(super) fn collect() -> ToolResult<Self> {
        Ok(Self {
            cudf: CmakePackageVersion::collect("CUDF_INSTALL_DIR", "cudf", "CUDF_VERSION")?,
            cugraph: SourcePackageVersion::collect("cugraph-nexus", "cugraph/VERSION")?,
            cuvs: CmakePackageVersion::collect("CUVS_INSTALL_DIR", "cuvs", "CUVS_VERSION")?,
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct SourcePackageVersion {
    pub(super) source: FileIdentity,
    pub(super) version: String,
}

impl SourcePackageVersion {
    pub(super) fn collect(sibling: &str, version_file: &str) -> ToolResult<Self> {
        let workspace_root = workspace_root()?;
        let sibling_root =
            workspace_root.parent().ok_or("workspace root has no parent for RAPIDS provenance")?.join(sibling);
        let version_path = sibling_root.join(version_file);
        let source = FileIdentity::collect(&version_path, "cuGraph VERSION")?;
        let version = fs::read_to_string(&version_path).map_err(|_| "cannot read cuGraph VERSION")?.trim().to_string();
        if version.is_empty() {
            return Err("cuGraph VERSION is empty".into());
        }
        Ok(Self { source, version })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct CmakePackageVersion {
    pub(super) install_dir: String,
    pub(super) config: FileIdentity,
    pub(super) version: String,
}

impl CmakePackageVersion {
    pub(super) fn collect(environment: &str, package: &str, version_variable: &str) -> ToolResult<Self> {
        let install_dir = env::var_os(environment)
            .map(PathBuf::from)
            .ok_or_else(|| format!("{environment} is required for RAPIDS provenance"))?;
        let install_dir = fs::canonicalize(&install_dir).map_err(|_| format!("cannot canonicalize {environment}"))?;
        let config_path = install_dir.join("lib").join("cmake").join(package).join(format!("{package}-config.cmake"));
        let config = FileIdentity::collect(&config_path, &format!("{package}-config.cmake"))?;
        let bytes = fs::read(&config_path).map_err(|_| format!("cannot read {package}-config.cmake"))?;
        let version = cmake_set_value(&bytes, version_variable)
            .ok_or_else(|| format!("{package}-config.cmake has no {version_variable}"))?;
        Ok(Self { install_dir: install_dir.display().to_string(), config, version })
    }
}

pub(super) fn cmake_set_value(contents: &[u8], variable: &str) -> Option<String> {
    let text = std::str::from_utf8(contents).ok()?;
    let prefix = format!("set({variable} ");
    text.lines().find_map(|line| {
        let value = line.trim().strip_prefix(&prefix)?.trim();
        let value = value.strip_suffix(')')?.trim().trim_matches('"');
        (!value.is_empty()).then(|| value.to_string())
    })
}

pub(super) fn required_command_version<const N: usize>(program: &str, args: [&str; N]) -> ToolResult<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|_| format!("cannot execute required provenance tool {program}"))?;
    if !output.status.success() {
        return Err(format!("required provenance tool {program} failed").into());
    }
    if output.stdout.len() > MAX_TOOL_VERSION_BYTES || output.stderr.len() > MAX_TOOL_VERSION_BYTES {
        return Err(format!("required provenance tool {program} produced an oversized version response").into());
    }
    let value = if output.stdout.is_empty() { output.stderr } else { output.stdout };
    let value = String::from_utf8(value)
        .map_err(|_| format!("required provenance tool {program} returned non-UTF-8 output"))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(format!("required provenance tool {program} returned an empty version").into());
    }
    Ok(value.to_string())
}

#[derive(Debug, Serialize)]
pub(super) struct RunnerIdentity {
    pub(super) executable: FileIdentity,
}

impl RunnerIdentity {
    pub(super) fn collect() -> ToolResult<Self> {
        let executable = env::current_exe().map_err(|_| "cannot resolve validation runner executable")?;
        Ok(Self { executable: FileIdentity::collect(&executable, "validation runner executable")? })
    }
}
