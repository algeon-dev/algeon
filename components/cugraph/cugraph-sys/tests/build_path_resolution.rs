#[path = "../build_support.rs"]
mod build_support;

#[cfg(not(target_os = "windows"))]
use std::os::unix::fs::PermissionsExt;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let unique = format!(
            "cugraph-sys-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_required_library_files(dir: &Path) {
    #[cfg(target_os = "macos")]
    let libs = ["libcugraph.dylib", "libcugraph_c.dylib"];

    #[cfg(target_os = "windows")]
    let libs = ["cugraph.dll", "cugraph_c.dll"];

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let libs = ["libcugraph.so", "libcugraph_c.so"];

    for lib in libs {
        fs::write(dir.join(lib), []).unwrap_or_else(|_| panic!("create test library file {lib}"));
    }
}

#[test]
fn include_override_fails_when_directory_is_missing() {
    let root = TestDir::new();
    let missing = root.path().join("missing-include-dir");

    let err = build_support::resolve_include_dir(Some(missing.to_str().unwrap()), root.path()).unwrap_err();

    assert!(err.contains("CUGRAPH_INCLUDE_DIR"));
    assert!(err.contains("cugraph_c"));
    assert!(err.contains(&missing.display().to_string()));
}

#[test]
fn include_override_fails_when_cugraph_headers_are_missing() {
    let root = TestDir::new();
    let include_dir = root.path().join("include");
    fs::create_dir_all(&include_dir).expect("create include dir");

    let err = build_support::resolve_include_dir(Some(include_dir.to_str().unwrap()), root.path()).unwrap_err();

    assert!(err.contains("CUGRAPH_INCLUDE_DIR"));
    assert!(err.contains("cugraph_c"));
    assert!(err.contains(&include_dir.display().to_string()));
}

#[test]
fn include_resolution_uses_submodule_when_env_is_absent() {
    let root = TestDir::new();
    let include_dir = root.path().join("submodule-include");
    fs::create_dir_all(include_dir.join("cugraph_c")).expect("create cugraph_c dir");

    let resolved = build_support::resolve_include_dir(None, &include_dir).expect("resolve include fallback");

    assert_eq!(resolved, include_dir.canonicalize().expect("canonical include path"));
}

#[test]
fn library_override_fails_when_directory_is_missing() {
    let root = TestDir::new();
    let missing = root.path().join("missing-lib-dir");

    let err = build_support::resolve_library_dir(Some(missing.to_str().unwrap()), &[root.path().join("fallback")])
        .unwrap_err();

    assert!(err.contains("CUGRAPH_LIBRARY_DIR"));
    assert!(err.contains("libcugraph"));
    assert!(err.contains(&missing.display().to_string()));
}

#[test]
fn library_override_fails_when_required_libraries_are_missing() {
    let root = TestDir::new();
    let lib_dir = root.path().join("lib");
    fs::create_dir_all(&lib_dir).expect("create lib dir");

    let err = build_support::resolve_library_dir(Some(lib_dir.to_str().unwrap()), &[root.path().join("fallback")])
        .unwrap_err();

    assert!(err.contains("CUGRAPH_LIBRARY_DIR"));
    assert!(err.contains("libcugraph"));
    assert!(err.contains("libcugraph_c"));
    assert!(err.contains(&lib_dir.display().to_string()));
}

#[test]
fn library_resolution_uses_first_candidate_with_required_libraries_when_env_is_absent() {
    let root = TestDir::new();
    let first = root.path().join("first");
    let second = root.path().join("second");
    fs::create_dir_all(&first).expect("create first");
    fs::create_dir_all(&second).expect("create second");
    create_required_library_files(&second);

    let resolved =
        build_support::resolve_library_dir(None, &[first.clone(), second.clone()]).expect("resolve library fallback");

    assert_eq!(resolved, second.canonicalize().expect("canonical library path"));
}

#[test]
fn library_resolution_fails_when_no_fallback_candidate_contains_required_libraries() {
    let root = TestDir::new();
    let first = root.path().join("first");
    let second = root.path().join("second");
    fs::create_dir_all(&first).expect("create first");
    fs::create_dir_all(&second).expect("create second");

    let err = build_support::resolve_library_dir(None, &[first.clone(), second.clone()]).unwrap_err();

    assert!(err.contains("libcugraph"));
    assert!(err.contains("libcugraph_c"));
    assert!(err.contains(&first.display().to_string()));
    assert!(err.contains(&second.display().to_string()));
}

#[test]
fn compiler_probe_defaults_to_cc_when_env_is_absent() {
    let (compiler, args) = build_support::resolve_compiler_probe_command(None).expect("resolve default compiler");

    assert_eq!(compiler, "cc");
    assert!(args.is_empty());
}

#[test]
fn compiler_probe_uses_cc_env_command_and_flags() {
    let (compiler, args) = build_support::resolve_compiler_probe_command(Some("clang --target=aarch64-linux-gnu"))
        .expect("resolve compiler override");

    assert_eq!(compiler, "clang");
    assert_eq!(args, vec!["--target=aarch64-linux-gnu"]);
}

#[cfg(not(target_os = "windows"))]
#[test]
fn detect_system_include_dir_uses_selected_compiler_output() {
    let root = TestDir::new();
    let include_dir = root.path().join("fake-include");
    fs::create_dir_all(&include_dir).expect("create include dir");
    fs::write(include_dir.join("stddef.h"), []).expect("create stddef.h");

    let compiler = root.path().join("fake-cc");
    fs::write(
        &compiler,
        format!(
            "#!/bin/sh\ncat >/dev/null\ncat <<'EOF' 1>&2\n#include <...> search starts here:\n {}\nEnd of search list.\nEOF\n",
            include_dir.display()
        ),
    )
    .expect("write fake compiler");

    let mut permissions = fs::metadata(&compiler).expect("stat fake compiler").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&compiler, permissions).expect("chmod fake compiler");

    let detected = build_support::detect_system_include_dir(Some(compiler.to_str().unwrap()))
        .expect("detect include dir from fake compiler");

    assert_eq!(detected, include_dir);
}
