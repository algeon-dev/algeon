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
        "clang-format check failed for cudf-sys/native using config {}.\nstdout:\n{}\nstderr:\n{}",
        config_path.display(),
        stdout,
        stderr
    );
}

/// Wipe the CMake build directory when the cached `CMAKE_INSTALL_PREFIX` no
/// longer matches the cargo `OUT_DIR`. CMake otherwise sometimes drops the
/// `-DCMAKE_INSTALL_PREFIX=` flag during an auto-reconfigure (e.g. when the
/// host compiler changed between cargo invocations) and falls back to
/// `/usr/local`, which then fails the install step with EACCES on systems
/// where `/usr/local/lib` is not user-writable.
fn discard_stale_cmake_build_dir() {
    let Ok(out_dir) = env::var("OUT_DIR") else { return };
    let out_dir = PathBuf::from(out_dir);
    let build_dir = out_dir.join("build");
    let cache_path = build_dir.join("CMakeCache.txt");
    let Ok(cache) = fs::read_to_string(&cache_path) else { return };
    let expected = format!("CMAKE_INSTALL_PREFIX:PATH={}", out_dir.display());
    if cache.lines().any(|line| line == expected) && cached_compilers_match_env(&cache) {
        return;
    }
    eprintln!(
        "cudf-sys build.rs: stale CMake cache detected in {}; \
         removing {} so cmake-rs can configure cleanly with OUT_DIR {}",
        cache_path.display(),
        build_dir.display(),
        out_dir.display(),
    );
    if let Err(error) = fs::remove_dir_all(&build_dir) {
        panic!("failed to remove stale cmake build directory at {}: {error}", build_dir.display(),);
    }
}

fn cached_compilers_match_env(cache: &str) -> bool {
    cached_compiler_matches_env(cache, "CMAKE_C_COMPILER", "CC")
        && cached_compiler_matches_env(cache, "CMAKE_CXX_COMPILER", "CXX")
}

fn cached_compiler_matches_env(cache: &str, cache_key: &str, env_key: &str) -> bool {
    let Some(expected) = compiler_from_env(env_key) else { return true };
    let Some(cached) = cmake_cache_value(cache, cache_key) else { return false };
    paths_equivalent(Path::new(cached), &expected)
}

fn cmake_cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    cache.lines().find_map(|line| {
        let (cache_name_and_type, value) = line.split_once('=')?;
        let (cache_name, _) = cache_name_and_type.split_once(':')?;
        (cache_name == key).then_some(value)
    })
}

fn compiler_from_env(env_key: &str) -> Option<PathBuf> {
    let raw = env::var(env_key).ok()?;
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(char::is_whitespace) {
        return None;
    }

    let path = PathBuf::from(raw);
    if path.components().count() > 1 { Some(path) } else { build_support::find_required_executable(raw).ok() }
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn main() {
    let target_os =
        env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|error| panic!("failed to read CARGO_CFG_TARGET_OS: {error}"));
    build_support::ensure_supported_target_os(&target_os).unwrap_or_else(|error| panic!("{error}"));

    discard_stale_cmake_build_dir();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = build_support::native_dir(&manifest_dir);
    let clang_tidy_config = build_support::clang_tidy_config_path(&manifest_dir);
    let clang_format_config = build_support::clang_format_config_path(&manifest_dir);
    let clang_lint =
        build_support::normalized_cmake_bool_from_env("CUDF_RUST_CLANG_LINT").unwrap_or_else(|error| panic!("{error}"));

    let mut config = cmake::Config::new(&native_dir);
    config.define("CUDF_RUST_ENABLE_CLANG_TIDY", clang_lint.unwrap_or("OFF"));
    if clang_lint == Some("ON") {
        let clang_format =
            build_support::find_required_executable("clang-format").unwrap_or_else(|error| panic!("{error}"));
        let clang_tidy =
            build_support::find_required_executable("clang-tidy").unwrap_or_else(|error| panic!("{error}"));

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

    // Re-run if native sources or build inputs change.
    rerun_if_changed_recursive(&native_dir.join("src"));
    rerun_if_changed_recursive(&native_dir.join("include"));
    println!("cargo:rerun-if-changed={}", native_dir.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", native_dir.join(".clang-tidy").display());
    println!("cargo:rerun-if-changed={}", clang_format_config.display());
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=native/cmake/cargo-link.txt.in");
    println!("cargo:rerun-if-env-changed=CUDF_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUDF_CPP_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_STATIC_RUNTIME");
    println!("cargo:rerun-if-env-changed=CUDF_RUST_CLANG_LINT");
    println!("cargo:rerun-if-env-changed=PATH");
}
