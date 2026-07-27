#[path = "../build_support.rs"]
mod build_support;

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
