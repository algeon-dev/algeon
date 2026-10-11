//! Root, component, cuVS, cuML, and toolchain provenance capture.

use std::{
    env, fs,
    path::{Component, Path, PathBuf},
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
    pub(super) process: ProcessIdentity,
    pub(super) repositories: RepositoryProvenanceSet,
    pub(super) native_libraries: NativeLibraryIdentities,
    pub(super) cuvs_build_inputs: CuvsBuildInputs,
    pub(super) cuml_build_inputs: CumlBuildInputs,
    pub(super) toolchains: ToolchainVersions,
    pub(super) runner: RunnerIdentity,
}

impl Provenance {
    pub(super) fn collect() -> ToolResult<Self> {
        let workspace_root = workspace_root()?;
        Ok(Self {
            process: ProcessIdentity::collect()?,
            repositories: RepositoryProvenanceSet {
                root: RepositoryProvenance::collect("algeon-datafusion", &workspace_root)?,
                cudf: RepositoryProvenance::collect(
                    "cudf-native-source",
                    &workspace_root.join("components/cudf/cudf"),
                )?,
                cugraph: RepositoryProvenance::collect(
                    "cugraph-native-source",
                    &workspace_root.join("components/cugraph/cugraph"),
                )?,
                cuvs: RepositoryProvenance::collect(
                    "cuvs-native-source",
                    &workspace_root.join("components/cuvs/cuvs"),
                )?,
                cuml: RepositoryProvenance::collect(
                    "cuml-native-source",
                    &workspace_root.join("components/cuml/cuml"),
                )?,
            },
            native_libraries: NativeLibraryIdentities::collect()?,
            cuvs_build_inputs: CuvsBuildInputs::collect()?,
            cuml_build_inputs: CumlBuildInputs::collect()?,
            toolchains: ToolchainVersions::collect()?,
            runner: RunnerIdentity::collect()?,
        })
    }
}

pub(super) fn workspace_root() -> ToolResult<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root =
        manifest_dir.parent().and_then(Path::parent).ok_or("cannot derive workspace root from tools manifest")?;
    fs::canonicalize(workspace_root).map_err(|_| "cannot canonicalize workspace root".into())
}

#[derive(Debug, Serialize)]
pub(super) struct RepositoryProvenanceSet {
    pub(super) root: RepositoryProvenance,
    pub(super) cudf: RepositoryProvenance,
    pub(super) cugraph: RepositoryProvenance,
    pub(super) cuvs: RepositoryProvenance,
    pub(super) cuml: RepositoryProvenance,
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
pub(super) struct NativeLibraryIdentities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) libcudf_resolved: Option<FileIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) libcudf_rust_resolved: Option<FileIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) libcugraph_resolved: Option<FileIdentity>,
}

impl NativeLibraryIdentities {
    fn collect() -> ToolResult<Self> {
        Ok(Self {
            libcudf_resolved: loaded_library_identity("libcudf.so")?,
            libcudf_rust_resolved: loaded_library_identity("libcudf_rust.so")?,
            libcugraph_resolved: loaded_library_identity("libcugraph.so")?,
        })
    }
}

fn loaded_library_identity(file_name: &str) -> ToolResult<Option<FileIdentity>> {
    let maps = fs::read_to_string("/proc/self/maps").map_err(|_| "cannot read /proc/self/maps")?;
    let mut paths = maps
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(Path::new)
        .filter(|path| {
            path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
                name == file_name || name.strip_prefix(file_name).is_some_and(|suffix| suffix.starts_with('.'))
            })
        })
        .map(fs::canonicalize)
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort_unstable();
    paths.dedup();
    match paths.as_slice() {
        [] => Ok(None),
        [path] => FileIdentity::collect(path, file_name).map(Some),
        _ => Err(format!("multiple loaded {file_name} identities are mapped").into()),
    }
}

#[derive(Debug, Serialize)]
pub(super) struct DirtyManifest {
    pub(super) format: &'static str,
    /// Digest of the complete direct dirty source state. A change to a dirty
    /// file's contents changes this value even when its status path stays the
    /// same.
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
        let tracked_diff = git_output_bytes(repo, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"], name)?;
        let untracked_paths = git_output_bytes(repo, ["ls-files", "--others", "--exclude-standard", "-z"], name)?;
        let sha256 = dirty_content_sha256(repo, name, &output, &tracked_diff, &untracked_paths)?;
        Ok(Self { format: "git-status-porcelain-v1-z+direct-worktree-content-sha256-v1", sha256, entries })
    }
}

fn dirty_content_sha256(
    repo: &Path,
    name: &str,
    status: &[u8],
    tracked_diff: &[u8],
    untracked_paths: &[u8],
) -> ToolResult<String> {
    // These bytes are never persisted; the artifact carries only their
    // digest. The revision, porcelain state, final tracked diff, every
    // untracked file together identify the source consumed by the release
    // binary without leaking a patch. Nested worktrees retain their normal Git
    // dirty entry, because their build outputs are not source inputs to this
    // release runner and can be arbitrarily large.
    let mut source_state = Vec::new();
    append_record(&mut source_state, b"format", b"direct-worktree-content-v1");
    append_record(&mut source_state, b"status", status);
    append_record(&mut source_state, b"tracked-diff", tracked_diff);
    for path in untracked_paths.split(|byte| *byte == b'\0').filter(|path| !path.is_empty()) {
        let path = std::str::from_utf8(path).map_err(|_| format!("{name} untracked path is not UTF-8"))?;
        let path = Path::new(path);
        if path.is_absolute() || path.components().any(|component| matches!(component, Component::ParentDir)) {
            return Err(format!("{name} untracked path escapes its repository").into());
        }
        let contents = read_untracked_contents(&repo.join(path), name)?;
        append_record(&mut source_state, b"untracked-path", path.as_os_str().as_encoded_bytes());
        append_record(&mut source_state, b"untracked-contents", &contents);
    }
    Ok(sha256_bytes(&source_state))
}

fn read_untracked_contents(path: &Path, name: &str) -> ToolResult<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| format!("cannot stat untracked {} in {name} for provenance", path.display()))?;
    if metadata.file_type().is_symlink() {
        let target = fs::read_link(path)
            .map_err(|_| format!("cannot read untracked symlink {} in {name} for provenance", path.display()))?;
        return Ok(target.as_os_str().as_encoded_bytes().to_vec());
    }
    if !metadata.is_file() {
        return Err(format!("untracked {} in {name} is not a regular file or symlink", path.display()).into());
    }
    fs::read(path).map_err(|_| format!("cannot read untracked {} in {name} for provenance", path.display()).into())
}

fn append_record(destination: &mut Vec<u8>, label: &[u8], value: &[u8]) {
    destination.extend_from_slice(&(label.len() as u64).to_be_bytes());
    destination.extend_from_slice(label);
    destination.extend_from_slice(&(value.len() as u64).to_be_bytes());
    destination.extend_from_slice(value);
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
pub(super) struct CumlBuildInputs {
    pub(super) install_dir: String,
    pub(super) cmake_config: FileIdentity,
    pub(super) libcuml_selector: FileIdentity,
    pub(super) libcuml_resolved: FileIdentity,
    pub(super) libcuml_rust_selector: FileIdentity,
    pub(super) libcuml_rust_resolved: FileIdentity,
    pub(super) version: String,
}

impl CumlBuildInputs {
    pub(super) fn collect() -> ToolResult<Self> {
        let install_dir = env::var_os("CUML_INSTALL_DIR")
            .map(PathBuf::from)
            .ok_or("CUML_INSTALL_DIR is required for production-validation provenance")?;
        let install_dir = fs::canonicalize(&install_dir).map_err(|_| "cannot canonicalize CUML_INSTALL_DIR")?;
        let cmake_config_path = install_dir.join("lib/cmake/cuml/cuml-config.cmake");
        let cmake_config = FileIdentity::collect(&cmake_config_path, "cuml-config.cmake")?;
        let (libcuml_selector, libcuml_resolved) = installed_library(&install_dir, "libcuml.so")?;
        let (libcuml_rust_selector, libcuml_rust_resolved) = installed_library(&install_dir, "libcuml_rust.so")?;
        let config_bytes = fs::read(&cmake_config_path).map_err(|_| "cannot read cuml-config.cmake")?;
        let version = cmake_set_value(&config_bytes, "CUML_VERSION").ok_or("cuml-config.cmake has no CUML_VERSION")?;
        Ok(Self {
            install_dir: install_dir.display().to_string(),
            cmake_config,
            libcuml_selector,
            libcuml_resolved,
            libcuml_rust_selector,
            libcuml_rust_resolved,
            version,
        })
    }
}

fn installed_library(install_dir: &Path, name: &str) -> ToolResult<(FileIdentity, FileIdentity)> {
    let selector_path = [install_dir.join("lib").join(name), install_dir.join("lib64").join(name)]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| format!("CUML_INSTALL_DIR must contain lib/{name} or lib64/{name}"))?;
    let selector = FileIdentity::collect(&selector_path, &format!("{name} selector"))?;
    let resolved_path = fs::canonicalize(&selector_path).map_err(|_| format!("cannot resolve selected {name}"))?;
    if !resolved_path.starts_with(install_dir) {
        return Err(format!("selected {name} resolves outside CUML_INSTALL_DIR").into());
    }
    let resolved = FileIdentity::collect(&resolved_path, &format!("resolved {name}"))?;
    Ok((selector, resolved))
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
    pub(super) cuml: CmakePackageVersion,
}

impl RapidsVersions {
    pub(super) fn collect() -> ToolResult<Self> {
        Ok(Self {
            cudf: CmakePackageVersion::collect("CUDF_INSTALL_DIR", "cudf", "CUDF_VERSION")?,
            cugraph: SourcePackageVersion::collect("components/cugraph/cugraph/VERSION")?,
            cuvs: CmakePackageVersion::collect("CUVS_INSTALL_DIR", "cuvs", "CUVS_VERSION")?,
            cuml: CmakePackageVersion::collect("CUML_INSTALL_DIR", "cuml", "CUML_VERSION")?,
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct SourcePackageVersion {
    pub(super) source: FileIdentity,
    pub(super) version: String,
}

impl SourcePackageVersion {
    pub(super) fn collect(version_file: &str) -> ToolResult<Self> {
        let workspace_root = workspace_root()?;
        let version_path = workspace_root.join(version_file);
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
        return Err(format!("required provenance tool {program} failed with status {}", output.status).into());
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

#[derive(Debug, Serialize)]
pub(super) struct ProcessIdentity {
    pub(super) boot_id: String,
    pub(super) pid: u32,
    pub(super) process_start_ticks: u64,
}

impl ProcessIdentity {
    pub(super) fn collect() -> ToolResult<Self> {
        let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map_err(|_| "cannot read process boot_id")?
            .trim()
            .to_string();
        if boot_id.is_empty() {
            return Err("process boot_id is empty".into());
        }
        let stat = fs::read_to_string("/proc/self/stat").map_err(|_| "cannot read /proc/self/stat")?;
        let fields = stat.rsplit_once(") ").ok_or("/proc/self/stat has no command boundary")?.1;
        let process_start_ticks = fields
            .split_whitespace()
            .nth(19)
            .ok_or("/proc/self/stat has no process start ticks")?
            .parse::<u64>()
            .map_err(|_| "/proc/self/stat process start ticks are invalid")?;
        Ok(Self { boot_id, pid: std::process::id(), process_start_ticks })
    }
}

impl RunnerIdentity {
    pub(super) fn collect() -> ToolResult<Self> {
        let executable = env::current_exe().map_err(|_| "cannot resolve validation runner executable")?;
        Ok(Self { executable: FileIdentity::collect(&executable, "validation runner executable")? })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::Path,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::dirty_content_sha256;

    #[test]
    fn dirty_content_digest_changes_when_a_tracked_diff_changes_at_the_same_path() {
        let status = b" M src/owner.rs\0";
        let first = dirty_content_sha256(Path::new("."), "test", status, b"diff --git a/src/owner.rs\n-old\n", b"")
            .expect("tracked diff digest");
        let second = dirty_content_sha256(Path::new("."), "test", status, b"diff --git a/src/owner.rs\n-new\n", b"")
            .expect("tracked diff digest");

        assert_ne!(first, second, "a porcelain-only digest would miss this source change");
    }

    #[test]
    fn dirty_content_digest_changes_when_an_untracked_file_changes_at_the_same_path() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("algeon-provenance-{nonce}"));
        fs::create_dir(&root).unwrap();
        let file = root.join("probe.rs");
        fs::write(&file, b"first").unwrap();
        let first = dirty_content_sha256(&root, "test", b"?? probe.rs\0", b"", b"probe.rs\0").unwrap();
        fs::write(&file, b"second").unwrap();
        let second = dirty_content_sha256(&root, "test", b"?? probe.rs\0", b"", b"probe.rs\0").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_ne!(first, second, "untracked source content belongs in the dirty manifest");
    }
}
