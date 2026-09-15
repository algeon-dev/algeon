//! CMake cache, libcudf discovery, link-manifest, and runtime-identity helpers.

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use crate::build_support::elf::{OptionalRuntimeDepLinkage, classify_optional_runtime_dep};

const CARGO_LINK_MANIFEST_RELATIVE_PATH: &str = "share/cudf_rust/cargo-link.txt";

pub(crate) fn cargo_link_manifest_path(install_prefix: &Path) -> PathBuf {
    install_prefix.join(CARGO_LINK_MANIFEST_RELATIVE_PATH)
}

pub(crate) fn read_cargo_link_directives(install_prefix: &Path) -> Result<Vec<String>, String> {
    let manifest_path = cargo_link_manifest_path(install_prefix);
    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("failed to read cargo link manifest {}: {error}", manifest_path.display()))?;
    parse_cargo_link_directives(&manifest).map_err(|error| format!("{error} ({})", manifest_path.display()))
}

pub(crate) fn parse_cargo_link_directives(manifest: &str) -> Result<Vec<String>, String> {
    let mut directives = Vec::new();

    for (index, raw_line) in manifest.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with("cargo:") {
            return Err(format!("invalid cargo directive at line {}", index + 1));
        }
        directives.push(line.to_owned());
    }

    if directives.is_empty() {
        return Err("cargo link manifest contained no directives".to_owned());
    }

    Ok(directives)
}

pub(crate) fn normalized_cmake_bool_from_env(var_name: &str) -> Result<Option<&'static str>, String> {
    match env::var(var_name) {
        Ok(value) => normalized_cmake_bool(var_name, Some(&value)),
        Err(env::VarError::NotPresent) => normalized_cmake_bool(var_name, None),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{var_name} contained non-Unicode data")),
    }
}

pub(crate) fn normalized_cmake_bool(var_name: &str, value: Option<&str>) -> Result<Option<&'static str>, String> {
    value.map(|value| parse_cmake_bool(var_name, value)).transpose()
}

pub(crate) fn parse_cmake_bool(var_name: &str, value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_uppercase().as_str() {
        "1" | "ON" | "TRUE" | "YES" => Ok("ON"),
        "0" | "OFF" | "FALSE" | "NO" => Ok("OFF"),
        _ => Err(format!("{var_name} must be one of 1, 0, ON, OFF, TRUE, FALSE, YES, or NO; got {value:?}")),
    }
}

pub(crate) fn cmake_cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    cache.lines().find_map(|line| {
        let (cache_name_and_type, value) = line.split_once('=')?;
        let (cache_name, _) = cache_name_and_type.split_once(':')?;
        (cache_name == key).then_some(value)
    })
}

pub(crate) fn cached_cudf_discovery_matches_env(cache: &str) -> bool {
    let install_dir = env_path("CUDF_INSTALL_DIR");
    let build_dir = env_path("CUDF_CPP_BUILD_DIR");
    cached_cudf_discovery_matches(cache, install_dir.as_deref(), build_dir.as_deref())
}

pub(crate) fn cached_cmake_compilers_match(cache: &str, c_compiler: &Path, cxx_compiler: &Path) -> bool {
    cached_cmake_compiler_matches(cache, "CMAKE_C_COMPILER", c_compiler)
        && cached_cmake_compiler_matches(cache, "CMAKE_CXX_COMPILER", cxx_compiler)
}

fn cached_cmake_compiler_matches(cache: &str, cache_key: &str, expected: &Path) -> bool {
    let Some(cached) = cmake_cache_value(cache, cache_key) else { return false };
    // CMake invalidates its cache when the compiler path's spelling changes,
    // even when both spellings resolve to the same executable through a symlink.
    Path::new(cached) == expected
}

pub(crate) fn cached_cudf_discovery_matches(cache: &str, install_dir: Option<&Path>, build_dir: Option<&Path>) -> bool {
    if let Some(install_dir) = install_dir {
        return cmake_cache_path_matches(cache, "CUDF_INSTALL_DIR", install_dir)
            && cmake_cache_value(cache, "CUDF_CPP_BUILD_DIR").is_none();
    }

    if let Some(build_dir) = build_dir {
        return cmake_cache_path_matches(cache, "CUDF_CPP_BUILD_DIR", build_dir)
            && cmake_cache_value(cache, "CUDF_INSTALL_DIR").is_none();
    }

    true
}

pub(crate) fn cmake_cache_path_matches(cache: &str, key: &str, expected: &Path) -> bool {
    let Some(cached) = cmake_cache_value(cache, key) else { return false };
    paths_equivalent(Path::new(cached), expected)
}

pub(crate) fn paths_equivalent(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

/// Resolve the exact libcudf shared library selected by the native discovery
/// contract. Installed prefixes use lib/lib64; a C++ build tree exposes
/// libcudf.so at its root.
pub(crate) fn locate_prebuilt_libcudf(install_dir: Option<&Path>, build_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(install_dir) = install_dir {
        for subdir in ["lib", "lib64"] {
            let candidate = install_dir.join(subdir).join("libcudf.so");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let candidate = build_dir?.join("libcudf.so");
    candidate.is_file().then_some(candidate)
}

/// Return the install prefix for a libcudf shared library in the conventional
/// `<prefix>/lib{,64}/libcudf.so` layout. A C++ build-tree library has no
/// install prefix and returns `None`.
#[cfg_attr(not(test), expect(dead_code, reason = "called by tests/build_link_contract.rs but unused in build.rs"))]
pub(crate) fn installed_prefix_for_libcudf(library_path: &Path) -> Option<&Path> {
    let library_dir = library_path.parent()?;
    matches!(library_dir.file_name().and_then(|name| name.to_str()), Some("lib" | "lib64"))
        .then(|| library_dir.parent())
        .flatten()
}

/// Wipe the CMake build directory when cached configure-time inputs no longer
/// match the current Cargo build. CMake otherwise keeps stale cache entries
/// across auto-reconfigures, including the libcudf discovery prefix, and may
/// silently ignore updated environment variables.
pub(crate) fn discard_stale_cmake_build_dir(c_compiler: &Path, cxx_compiler: &Path, test_hooks: &str) {
    let Ok(out_dir) = env::var("OUT_DIR") else { return };
    discard_stale_cmake_build_dir_at(Path::new(&out_dir), c_compiler, cxx_compiler, test_hooks);
}

/// Staleness check for a secondary CMake build prefix, e.g. the sibling
/// `production-out` shim produced when the active Cargo feature enables
/// test hooks. Each prefix keeps its own cache, so it is validated with the
/// same compiler/discovery/test-hook fingerprint as the primary build.
pub(crate) fn discard_stale_cmake_build_dir_at(
    out_dir: &Path,
    c_compiler: &Path,
    cxx_compiler: &Path,
    test_hooks: &str,
) {
    let build_dir = out_dir.join("build");
    let cache_path = build_dir.join("CMakeCache.txt");
    let Ok(cache) = fs::read_to_string(&cache_path) else { return };
    let expected = format!("CMAKE_INSTALL_PREFIX:PATH={}", out_dir.display());
    if cache.lines().any(|line| line == expected)
        && cached_cmake_compilers_match(&cache, c_compiler, cxx_compiler)
        && cached_cudf_discovery_matches_env(&cache)
        && cmake_cache_value(&cache, "CUDF_RUST_ENABLE_TEST_HOOKS") == Some(test_hooks)
    {
        return;
    }
    eprintln!(
        "cudf-sys build.rs: stale CMake cache detected in {}; \
         removing {} so cmake-rs can configure cleanly with OUT_DIR {} and current libcudf discovery env",
        cache_path.display(),
        build_dir.display(),
        out_dir.display(),
    );
    if let Err(error) = fs::remove_dir_all(&build_dir) {
        panic!("failed to remove stale cmake build directory at {}: {error}", build_dir.display());
    }
}

pub(crate) fn default_cudf_cmake_cache(manifest_dir: &Path) -> PathBuf {
    super::default_cudf_build_dir(manifest_dir)
        .expect("cudf-nexus-sys manifest must be inside the monorepo")
        .join("CMakeCache.txt")
}

/// `CMakeCache.txt` that participates in the active libcudf discovery
/// contract, if any. A build tree selected alongside an install prefix carries
/// capability metadata and must identify that same install prefix.
pub(crate) fn active_cudf_cmake_cache(
    manifest_dir: &Path,
    install_dir: Option<&Path>,
    build_dir: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(build_dir) = build_dir {
        return Some(build_dir.join("CMakeCache.txt"));
    }
    install_dir.is_none().then(|| default_cudf_cmake_cache(manifest_dir))
}

fn locate_runtime_dep_library(file_name: &str, build_dir_subpath: &str) -> Option<PathBuf> {
    if let Ok(install_dir) = env::var("CUDF_INSTALL_DIR") {
        for subdir in ["lib", "lib64"] {
            let candidate = PathBuf::from(&install_dir).join(subdir).join(file_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    if let Ok(build_dir) = env::var("CUDF_CPP_BUILD_DIR") {
        let candidate = PathBuf::from(build_dir).join(build_dir_subpath).join(file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn emit_runtime_dep_path_metadata(key: &str, path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo:{key}={}", path.display());
    println!("cargo:metadata={key}={}", path.display());
}

fn emit_runtime_dep_linkage_metadata(key: &str, linkage: &str) {
    println!("cargo:{key}={linkage}");
    println!("cargo:metadata={key}={linkage}");
}

fn emit_required_runtime_dep_library_metadata(
    key: &str,
    file_name: &str,
    build_dir_subpath: &str,
    discovery_env_present: bool,
) -> Option<PathBuf> {
    match locate_runtime_dep_library(file_name, build_dir_subpath) {
        Some(path) => {
            emit_runtime_dep_path_metadata(key, &path);
            Some(path)
        },
        None if discovery_env_present => {
            println!(
                "cargo:warning=required shared runtime dependency {file_name} was not found under the libcudf discovery prefix \
                 (CUDF_INSTALL_DIR/CUDF_CPP_BUILD_DIR); downstream builds that link a second RAPIDS \
                 consumer (e.g. cugraph) cannot verify the single-RMM invariant. Rebuild libcudf with \
                 CUDF_RMM_BUILD_SHARED=ON and re-run `cmake --install` if this is unexpected."
            );
            None
        },
        None => None,
    }
}

fn emit_optional_runtime_dep_library_metadata(
    key: &str,
    file_name: &str,
    build_dir_subpath: &str,
    linkage_key: &str,
    consumers: &[PathBuf],
) {
    let candidate = locate_runtime_dep_library(file_name, build_dir_subpath);
    let linkage = classify_optional_runtime_dep(candidate, consumers, file_name)
        .unwrap_or_else(|error| panic!("failed to determine {file_name} linkage: {error}"));
    match linkage {
        OptionalRuntimeDepLinkage::Shared(path) => {
            emit_runtime_dep_path_metadata(key, &path);
            emit_runtime_dep_linkage_metadata(linkage_key, "shared");
        },
        OptionalRuntimeDepLinkage::Absorbed => emit_runtime_dep_linkage_metadata(linkage_key, "absorbed"),
        OptionalRuntimeDepLinkage::StaleSharedArtifact(path) => {
            emit_runtime_dep_linkage_metadata(linkage_key, "absorbed");
            // Track the stale file so removing it reruns discovery and clears
            // the warning instead of replaying it from the cached output.
            println!("cargo:rerun-if-changed={}", path.display());
            println!(
                "cargo:warning={} exists but no shipped RAPIDS shared library links it (no DT_NEEDED entry in \
                 libcudf.so/librmm.so); it is a stale artifact from an earlier shared build and is reported as \
                 absorbed. Remove it and any stale rapids_logger CMake package (lib/cmake/rapids_logger under \
                 the same prefix): downstream single-runtime enforcement must observe real linkage, not \
                 leftovers `cmake --install` never removed.",
                path.display()
            );
        },
    }
}

/// Emit downstream runtime identity metadata while preserving the exact
/// install-prefix-over-build-tree discovery precedence used by CMake.
pub(crate) fn emit_runtime_dep_library_metadata() {
    let discovery_env_present =
        env::var_os("CUDF_INSTALL_DIR").is_some() || env::var_os("CUDF_CPP_BUILD_DIR").is_some();
    let install_dir = env::var_os("CUDF_INSTALL_DIR").map(PathBuf::from);
    let build_dir = env::var_os("CUDF_CPP_BUILD_DIR").map(PathBuf::from);
    let libcudf = match locate_prebuilt_libcudf(install_dir.as_deref(), build_dir.as_deref()) {
        Some(path) => {
            emit_runtime_dep_path_metadata("libcudf-runtime-lib", &path);
            Some(path)
        },
        None if discovery_env_present => {
            println!(
                "cargo:warning=required libcudf.so was not found under the configured libcudf discovery path \
                 (CUDF_INSTALL_DIR/CUDF_CPP_BUILD_DIR); downstream builds cannot pin the exact libcudf runtime identity"
            );
            None
        },
        None => None,
    };
    let rmm = emit_required_runtime_dep_library_metadata(
        "rmm-runtime-lib",
        "librmm.so",
        "_deps/rmm-build",
        discovery_env_present,
    );
    let consumers = [libcudf, rmm].into_iter().flatten().collect::<Vec<_>>();
    // Downstream binaries resolve libcudf and RMM through rpaths built from
    // these directories; tests inspecting the shim's loader resolution use the
    // same directories instead of the caller's LD_LIBRARY_PATH.
    let loader_path = env::join_paths(consumers.iter().filter_map(|library| library.parent()))
        .unwrap_or_else(|error| panic!("runtime library directories cannot form a loader search path: {error}"));
    println!("cargo:rustc-env=CUDF_RUST_RUNTIME_LIBRARY_PATH={}", loader_path.to_string_lossy());
    emit_optional_runtime_dep_library_metadata(
        "rapids-logger-runtime-lib",
        "librapids_logger.so",
        "_deps/rapids_logger-build",
        "rapids-logger-linkage",
        &consumers,
    );
}

/// Replay the CMake link manifest and make every native search path visible to
/// the dynamic loader and downstream build scripts.
pub(crate) fn replay_cargo_link_directives(directives: &[String]) {
    for directive in directives {
        println!("{directive}");
    }
    for path in directives.iter().filter_map(|directive| link_search_path(directive)) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
        println!("cargo:runtime-lib-dir={}", path.display());
        println!("cargo:metadata=runtime-lib-dir={}", path.display());
    }
}

fn link_search_path(directive: &str) -> Option<&Path> {
    directive.strip_prefix("cargo:rustc-link-search=native=").map(Path::new)
}

/// Re-assert the primary install prefix as the exported package root after
/// an auxiliary CMake build.
///
/// `cmake::Config::build` emits `cargo:root` for the prefix it configured
/// and Cargo keeps the last value for a repeated key, so the auxiliary
/// production prefix would otherwise become `DEP_CUDF_RUST_ROOT` while the
/// replayed link search, rpath, and `cudf-rust-runtime-lib` metadata keep
/// naming the primary prefix.
pub(crate) fn reassert_primary_root(primary_root: &Path) {
    println!("cargo:root={}", primary_root.display());
}

/// Verify the downstream identity contract for the primary install prefix:
/// the exported package root, the `cudf-rust-runtime-lib` path, and the
/// active linked shim derived from the replayed link manifest must all
/// describe the same prefix.
pub(crate) fn verify_primary_shim_identity(primary_root: &Path, runtime_lib: &Path) -> Result<(), String> {
    let directives = read_cargo_link_directives(primary_root)
        .map_err(|error| format!("primary root {primary_root:?} does not own a valid cargo link manifest: {error}"))?;
    let link_dir = directives.iter().find_map(|directive| link_search_path(directive)).ok_or_else(|| {
        format!("cargo link manifest under {} has no native link search path", primary_root.display())
    })?;
    let dylib = directives
        .iter()
        .find_map(|directive| directive.strip_prefix("cargo:rustc-link-lib=dylib="))
        .ok_or_else(|| format!("cargo link manifest under {} has no dylib link directive", primary_root.display()))?;
    let linked_shim = link_dir.join(format!("lib{dylib}.so"));
    let shim_prefix = linked_shim
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| format!("linked shim {} has no install prefix", linked_shim.display()))?;

    if !paths_equivalent(&linked_shim, runtime_lib) {
        return Err(format!(
            "active linked shim {} and runtime library {} must be the same file under the primary install prefix {}",
            linked_shim.display(),
            runtime_lib.display(),
            primary_root.display()
        ));
    }
    if !paths_equivalent(shim_prefix, primary_root) {
        return Err(format!(
            "exported root {}, runtime library {}, and active linked shim {} must describe the same primary install prefix",
            primary_root.display(),
            runtime_lib.display(),
            linked_shim.display()
        ));
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct StalenessReport {
    pub(crate) total_newer: usize,
    pub(crate) libcudf_so: PathBuf,
    pub(crate) newest_path: PathBuf,
}

/// Warn (via `cargo:warning=`) when libcudf C++ sources are newer than the
/// selected libcudf binary. `CUDF_RS_SKIP_STALENESS_CHECK=1` only suppresses
/// the warning; it remains a tracked native fingerprint input.
pub(crate) fn check_libcudf_staleness(manifest_dir: &Path) -> Option<StalenessReport> {
    if env::var_os("CUDF_RS_SKIP_STALENESS_CHECK").is_some() {
        return None;
    }

    let install_dir = env::var_os("CUDF_INSTALL_DIR").map(PathBuf::from);
    let build_dir = env::var_os("CUDF_CPP_BUILD_DIR").map(PathBuf::from);
    let libcudf_so = locate_prebuilt_libcudf(install_dir.as_deref(), build_dir.as_deref())?;
    let Ok(libcudf_meta) = fs::metadata(&libcudf_so) else { return None };
    let Ok(libcudf_mtime) = libcudf_meta.modified() else { return None };

    // cudf-sys lives at <repo>/cudf-nexus-sys, libcudf submodule at <repo>/cudf.
    let cudf_cpp = manifest_dir.join("../cudf/cpp");
    let scan_dirs = [cudf_cpp.join("src"), cudf_cpp.join("include")];

    let mut newest: Option<(PathBuf, SystemTime)> = None;
    let mut total_newer = 0usize;
    for dir in &scan_dirs {
        if !dir.is_dir() {
            return None;
        }
        let (dir_newest, count) = find_newest_source(dir, libcudf_mtime);
        total_newer += count;
        if let Some(candidate) = dir_newest
            && newest.as_ref().is_none_or(|(_, time)| candidate.1 > *time)
        {
            newest = Some(candidate);
        }
    }

    if total_newer == 0 {
        return None;
    }

    let newest_path = newest.expect("a newer libcudf source requires a newest source path").0;
    let display_path = newest_path.display().to_string();
    println!(
        "cargo:warning=libcudf appears stale: {total_newer} source file(s) under cudf/cpp/{{src,include}} are newer than {}. Run `bash scripts/build/build_libcudf.sh` to rebuild libcudf. (newest: {}) Set CUDF_RS_SKIP_STALENESS_CHECK=1 to silence.",
        libcudf_so.display(),
        display_path,
    );

    if env::var_os("CUDF_RS_ENFORCE_STALENESS_CHECK").is_some() || env::var_os("CUDF_ENFORCE_STALENESS_CHECK").is_some()
    {
        panic!(
            "\n{}\n[ERROR] libcudf is stale: {total_newer} source file(s) under cudf/cpp/{{src,include}} are newer than {}.\n\
             Run `bash scripts/build/build_libcudf.sh` to rebuild libcudf, or set CUDF_RS_SKIP_STALENESS_CHECK=1 to silence.\n{}\n",
            "=".repeat(80),
            libcudf_so.display(),
            "=".repeat(80),
        );
    }

    Some(StalenessReport { total_newer, libcudf_so, newest_path })
}

pub(crate) fn emit_stale_build_diagnostic(report: &StalenessReport) {
    eprintln!(
        "\n{}\n[ERROR] cudf-nexus-sys native build failed and native libcudf appears stale!\n\
         {count} source file(s) under cudf/cpp/{{src,include}} are newer than {lib}.\n\
         (Newest source: {newest})\n\
         👉 Fix: Run `bash scripts/build/build_libcudf.sh` from the repository root.\n{}\n",
        "=".repeat(80),
        "=".repeat(80),
        count = report.total_newer,
        lib = report.libcudf_so.display(),
        newest = report.newest_path.display(),
    );
}

/// Walk `dir` and return the newest C++ source/header and count newer than
/// `baseline` without reading source contents.
fn find_newest_source(dir: &Path, baseline: SystemTime) -> (Option<(PathBuf, SystemTime)>, usize) {
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
            if !matches!(extension, "cpp" | "cu" | "hpp" | "cuh" | "h") {
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

fn env_path(var_name: &str) -> Option<PathBuf> {
    env::var(var_name).ok().map(PathBuf::from)
}
