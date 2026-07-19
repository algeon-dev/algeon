mod build_support;

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn emit_runtime_search_path(path: &Path) {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}

fn link_search_path(directive: &str) -> Option<&Path> {
    directive.strip_prefix("cargo:rustc-link-search=native=").map(Path::new)
}

/// Recursively emit `cargo:rerun-if-changed` for every file under `dir`.
fn rerun_if_changed_recursive(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display())) {
        let entry = entry.unwrap_or_else(|e| panic!("failed to read entry in {}: {e}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            rerun_if_changed_recursive(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn ensure_file_exists(path: &Path, description: &str) {
    if path.is_file() {
        return;
    }

    panic!("required {description} was not found at {}", path.display());
}

fn run_clang_format_check(clang_format: &Path, config_path: &Path, lint_files: &[PathBuf]) {
    let mut command = Command::new(clang_format);
    command.arg("--dry-run").arg("--Werror");
    command.arg(format!("--style=file:{}", config_path.display()));
    for file in lint_files {
        command.arg(file);
    }

    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run clang-format from {}: {error}", clang_format.display()));
    if output.status.success() {
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    panic!(
        "clang-format check failed for cudf-nexus-sys/native using config {}.\nstdout:\n{}\nstderr:\n{}",
        config_path.display(),
        stdout,
        stderr
    );
}

/// Wipe the CMake build directory when cached configure-time inputs no longer
/// match the current Cargo build. CMake otherwise keeps stale cache entries
/// across auto-reconfigures, including the libcudf discovery prefix, and may
/// silently ignore updated environment variables.
fn discard_stale_cmake_build_dir(c_compiler: &Path, cxx_compiler: &Path) {
    let Ok(out_dir) = env::var("OUT_DIR") else { return };
    let out_dir = PathBuf::from(out_dir);
    let build_dir = out_dir.join("build");
    let cache_path = build_dir.join("CMakeCache.txt");
    let Ok(cache) = fs::read_to_string(&cache_path) else { return };
    let expected = format!("CMAKE_INSTALL_PREFIX:PATH={}", out_dir.display());
    if cache.lines().any(|line| line == expected)
        && build_support::cached_cmake_compilers_match(&cache, c_compiler, cxx_compiler)
        && build_support::cached_cudf_discovery_matches_env(&cache)
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
        panic!("failed to remove stale cmake build directory at {}: {error}", build_dir.display(),);
    }
}

/// Locate the pre-built `libcudf.so` based on the same env vars cudf-sys's
/// CMake uses to find libcudf: `CUDF_INSTALL_DIR` (preferred) or
/// `CUDF_CPP_BUILD_DIR`.
fn locate_prebuilt_libcudf() -> Option<PathBuf> {
    let install_dir = env::var_os("CUDF_INSTALL_DIR").map(PathBuf::from);
    let build_dir = env::var_os("CUDF_CPP_BUILD_DIR").map(PathBuf::from);
    build_support::locate_prebuilt_libcudf(install_dir.as_deref(), build_dir.as_deref())
}

/// Locate a shared runtime dependency library (e.g. `librmm.so`) shipped by
/// the libcudf build, using the same discovery env vars as
/// `locate_prebuilt_libcudf`: `CUDF_INSTALL_DIR` (preferred) or
/// `CUDF_CPP_BUILD_DIR`.
fn locate_runtime_dep_library(file_name: &str, build_dir_subpath: &str) -> Option<PathBuf> {
    if let Ok(install_dir) = env::var("CUDF_INSTALL_DIR") {
        for sub in ["lib", "lib64"] {
            let candidate = PathBuf::from(&install_dir).join(sub).join(file_name);
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
    let linkage = build_support::classify_optional_runtime_dep(candidate, consumers, file_name)
        .unwrap_or_else(|error| panic!("failed to determine {file_name} linkage: {error}"));
    match linkage {
        build_support::OptionalRuntimeDepLinkage::Shared(path) => {
            emit_runtime_dep_path_metadata(key, &path);
            emit_runtime_dep_linkage_metadata(linkage_key, "shared");
        },
        build_support::OptionalRuntimeDepLinkage::Absorbed => {
            emit_runtime_dep_linkage_metadata(linkage_key, "absorbed")
        },
        build_support::OptionalRuntimeDepLinkage::StaleSharedArtifact(path) => {
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

/// Expose the resolved shared RMM runtime library to downstream crates as
/// `DEP_CUDF_RUST_RMM_RUNTIME_LIB`, so integration builds that link a second
/// RAPIDS consumer (e.g. cugraph-sys) can assert both resolve the same single
/// librmm.so build. A process must never carry two RMM builds: the dynamic
/// loader binds every rmm symbol to one copy and any divergence in object
/// layout or per-device resource state corrupts execution.
///
/// rapids_logger is metadata-only here. cuDF's static-deps build can absorb
/// rapids_logger into librmm.so, so a missing librapids_logger.so is not itself
/// a single-RMM violation. A *present* librapids_logger.so is not proof of
/// shared linkage either: `cmake --install` never removes files, so stale
/// artifacts from earlier shared builds linger. Linkage is therefore
/// classified by the DT_NEEDED entries of the shipped libcudf.so/librmm.so,
/// and stale candidates are reported as absorbed with a removal warning.
fn emit_runtime_dep_library_metadata() {
    let discovery_env_present =
        env::var_os("CUDF_INSTALL_DIR").is_some() || env::var_os("CUDF_CPP_BUILD_DIR").is_some();
    let libcudf = match locate_prebuilt_libcudf() {
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
    emit_optional_runtime_dep_library_metadata(
        "rapids-logger-runtime-lib",
        "librapids_logger.so",
        "_deps/rapids_logger-build",
        "rapids-logger-linkage",
        &consumers,
    );
}

/// Walk `dir` and return the most recently modified C++ source/header file
/// (path and mtime) along with the count of files newer than `baseline`.
fn find_newest_source(
    dir: &Path,
    baseline: std::time::SystemTime,
) -> (Option<(PathBuf, std::time::SystemTime)>, usize) {
    let mut newest: Option<(PathBuf, std::time::SystemTime)> = None;
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
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "cpp" | "cu" | "hpp" | "cuh" | "h") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let Ok(mtime) = meta.modified() else { continue };
            if mtime > baseline {
                newer_than_baseline += 1;
            }
            if newest.as_ref().is_none_or(|(_, t)| mtime > *t) {
                newest = Some((path, mtime));
            }
        }
    }
    (newest, newer_than_baseline)
}

/// Warn (via `cargo:warning=`) when libcudf C++ sources are newer than the
/// installed `libcudf.so`, which usually means the user edited or pulled cudf
/// without rebuilding the libcudf install prefix. Set
/// `CUDF_RS_SKIP_STALENESS_CHECK=1` to suppress.
fn check_libcudf_staleness(manifest_dir: &Path) {
    println!("cargo:rerun-if-env-changed=CUDF_RS_SKIP_STALENESS_CHECK");
    if env::var_os("CUDF_RS_SKIP_STALENESS_CHECK").is_some() {
        return;
    }

    let Some(libcudf_so) = locate_prebuilt_libcudf() else { return };
    let Ok(libcudf_meta) = fs::metadata(&libcudf_so) else { return };
    let Ok(libcudf_mtime) = libcudf_meta.modified() else { return };

    // cudf-sys lives at <repo>/cudf-nexus-sys, libcudf submodule at <repo>/cudf.
    let cudf_cpp = manifest_dir.join("../cudf/cpp");
    let scan_dirs = [cudf_cpp.join("src"), cudf_cpp.join("include")];

    let mut newest: Option<(PathBuf, std::time::SystemTime)> = None;
    let mut total_newer = 0usize;
    for dir in &scan_dirs {
        if !dir.is_dir() {
            return;
        }
        let (dir_newest, count) = find_newest_source(dir, libcudf_mtime);
        total_newer += count;
        if let Some(candidate) = dir_newest
            && newest.as_ref().is_none_or(|(_, t)| candidate.1 > *t)
        {
            newest = Some(candidate);
        }
    }

    if total_newer == 0 {
        return;
    }

    let display_path = newest.as_ref().map(|(p, _)| p.display().to_string()).unwrap_or_default();
    let rebuild_command = match build_support::installed_prefix_for_libcudf(&libcudf_so) {
        Some(install_prefix) => format!("`INSTALL_PREFIX={} ./build.sh libcudf`", install_prefix.display()),
        None => "`./build.sh libcudf -n`".to_owned(),
    };
    println!(
        "cargo:warning=libcudf appears stale: {total_newer} source file(s) under cudf/cpp/{{src,include}} are newer than {}. In cudf/, run {rebuild_command} to rebuild the selected libcudf. (newest: {}) Set CUDF_RS_SKIP_STALENESS_CHECK=1 to silence.",
        libcudf_so.display(),
        display_path,
    );
}

fn main() {
    let target_os =
        env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|error| panic!("failed to read CARGO_CFG_TARGET_OS: {error}"));
    build_support::ensure_supported_target_os(&target_os).unwrap_or_else(|error| panic!("{error}"));

    // Resolve compilers once to absolute paths before any CMake cache checks so
    // discard and cmake-rs observe the same toolchain identity. Basename CC/CXX
    // values remain accepted for ergonomics, but PATH is no longer a Cargo
    // fingerprint input.
    let cc = build_support::resolve_compiler_from_env_or_name("CC", "cc").unwrap_or_else(|error| panic!("{error}"));
    // SAFETY: build scripts are single-threaded and cmake-rs reads CC before spawn.
    unsafe {
        env::set_var("CC", &cc);
    }
    let cxx = build_support::resolve_compiler_from_env_or_name("CXX", "c++").unwrap_or_else(|error| panic!("{error}"));
    unsafe {
        env::set_var("CXX", &cxx);
    }

    // Make the repository-owned CMake pin effective, not metadata-only. The
    // cmake crate reads the standard CMAKE variable when it constructs both its
    // configure and build commands.
    if env::var_os(build_support::CMAKE_EXE_ENV).is_some() {
        let cmake = build_support::resolve_tool_from_env_or_name(build_support::CMAKE_EXE_ENV, "cmake")
            .unwrap_or_else(|error| panic!("{error}"));
        // SAFETY: build scripts are single-threaded and cmake-rs reads CMAKE
        // only after this point.
        unsafe {
            env::set_var("CMAKE", cmake);
        }
    }

    discard_stale_cmake_build_dir(&cc, &cxx);

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    check_libcudf_staleness(&manifest_dir);
    let native_dir = build_support::native_dir(&manifest_dir);
    let clang_tidy_config = build_support::clang_tidy_config_path(&manifest_dir);
    let clang_format_config = build_support::clang_format_config_path(&manifest_dir);
    let clang_lint =
        build_support::normalized_cmake_bool_from_env("CUDF_RUST_CLANG_LINT").unwrap_or_else(|error| panic!("{error}"));

    let mut config = cmake::Config::new(&native_dir);
    config.define("CUDF_RUST_ENABLE_CLANG_TIDY", clang_lint.unwrap_or("OFF"));
    if clang_lint == Some("ON") {
        let clang_format =
            build_support::resolve_tool_from_env_or_name(build_support::CLANG_FORMAT_EXE_ENV, "clang-format")
                .unwrap_or_else(|error| panic!("{error}"));
        let clang_tidy = build_support::resolve_tool_from_env_or_name(build_support::CLANG_TIDY_EXE_ENV, "clang-tidy")
            .unwrap_or_else(|error| panic!("{error}"));

        ensure_file_exists(&clang_format_config, "clang-format config");
        ensure_file_exists(&clang_tidy_config, "clang-tidy config");
        let lint_files =
            build_support::collect_native_clang_lint_files(&manifest_dir).unwrap_or_else(|error| panic!("{error}"));
        run_clang_format_check(&clang_format, &clang_format_config, &lint_files);

        config.define("CUDF_RUST_CLANG_TIDY_EXE", &clang_tidy);
        config.define("CUDF_RUST_CLANG_TIDY_CONFIG_FILE", &clang_tidy_config);
    }

    if let Some(cuda_static_runtime) =
        build_support::normalized_cmake_bool_from_env("CUDA_STATIC_RUNTIME").unwrap_or_else(|error| panic!("{error}"))
    {
        config.define("CUDA_STATIC_RUNTIME", cuda_static_runtime);
    }

    // Build the C++ shim library using CMake, then replay the link directives it emitted.
    let dst = config.build();
    let directives = build_support::read_cargo_link_directives(&dst).unwrap_or_else(|error| panic!("{error}"));
    for directive in &directives {
        println!("{directive}");
    }
    for path in directives.iter().filter_map(|directive| link_search_path(directive)) {
        emit_runtime_search_path(path);
        println!("cargo:runtime-lib-dir={}", path.display());
        println!("cargo:metadata=runtime-lib-dir={}", path.display());
    }
    emit_runtime_dep_library_metadata();

    // Re-run if native sources or build inputs change.
    rerun_if_changed_recursive(&native_dir.join("src"));
    rerun_if_changed_recursive(&native_dir.join("include"));
    println!("cargo:rerun-if-changed={}", native_dir.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", native_dir.join("cmake/cudf-discovery.cmake").display());

    let install_dir = env::var_os("CUDF_INSTALL_DIR").map(PathBuf::from);
    let build_dir = env::var_os("CUDF_CPP_BUILD_DIR").map(PathBuf::from);
    if let Some(cmake_cache) =
        build_support::active_cudf_cmake_cache(&manifest_dir, install_dir.as_deref(), build_dir.as_deref())
    {
        println!("cargo:rerun-if-changed={}", cmake_cache.display());
    }

    println!("cargo:rerun-if-changed={}", native_dir.join(".clang-tidy").display());
    println!("cargo:rerun-if-changed={}", clang_format_config.display());
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=native/cmake/cargo-link.txt.in");
    // Explicit literals keep the rebuild contract reviewable and match the
    // static include_str! fingerprint matrix. Keep in sync with
    // build_support::tracked_native_build_env_vars().
    const TRACKED_ENV: &[&str] = &[
        "CC",
        "CXX",
        "CUDACXX",
        "CUDAHOSTCXX",
        "CUDF_INSTALL_DIR",
        "CUDF_CPP_BUILD_DIR",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
        "CUDA_STATIC_RUNTIME",
        "CMAKE",
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_PREFIX_PATH",
        "CMAKE_LIBRARY_PATH",
        "CMAKE_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "CUDF_RUST_CLANG_LINT",
        "CUDF_RUST_CLANG_FORMAT",
        "CUDF_RUST_CLANG_TIDY",
        "CUDF_RUST_CMAKE",
        "CUDF_RS_SKIP_STALENESS_CHECK",
    ];
    assert_eq!(
        build_support::tracked_native_build_env_vars(),
        TRACKED_ENV,
        "build.rs TRACKED_ENV drifted from build_support::tracked_native_build_env_vars()"
    );
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");
    println!("cargo:rerun-if-env-changed=CUDF_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUDF_CPP_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_STATIC_RUNTIME");
    println!("cargo:rerun-if-env-changed=CMAKE");
    println!("cargo:rerun-if-env-changed=CMAKE_GENERATOR");
    println!("cargo:rerun-if-env-changed=CMAKE_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDAARCHS");
    println!("cargo:rerun-if-env-changed=CUDF_RUST_CLANG_LINT");
    println!("cargo:rerun-if-env-changed=CUDF_RUST_CLANG_FORMAT");
    println!("cargo:rerun-if-env-changed=CUDF_RUST_CLANG_TIDY");
    println!("cargo:rerun-if-env-changed=CUDF_RUST_CMAKE");
    println!("cargo:rerun-if-env-changed=CUDF_RS_SKIP_STALENESS_CHECK");
}
