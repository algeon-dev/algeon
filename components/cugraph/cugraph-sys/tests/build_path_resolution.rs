#[path = "../build_support.rs"]
mod build_support;

use std::{
    fs,
    io::Write,
    os::unix::fs::PermissionsExt,
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
    let libs = ["libcugraph.so", "libcugraph_c.so"];

    for lib in libs {
        fs::write(dir.join(lib), []).unwrap_or_else(|_| panic!("create test library file {lib}"));
    }
}

fn write_executable_script(path: &Path, contents: &str) {
    let staging = path.with_extension("tmp");
    let mut file = fs::File::create(&staging).expect("create staged script");
    file.write_all(contents.as_bytes()).expect("write staged script");
    file.sync_all().expect("sync staged script");
    drop(file);

    let mut permissions = fs::metadata(&staging).expect("stat staged script").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&staging, permissions).expect("chmod staged script");
    fs::rename(&staging, path).expect("install staged script");
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
fn include_resolution_prefers_pkg_config_before_submodule_fallback() {
    let root = TestDir::new();
    let include_dir = root.path().join("pkgconfig-include");
    fs::create_dir_all(include_dir.join("cugraph_c")).expect("create pkgconfig include");

    let submodule_include = root.path().join("submodule-include");
    fs::create_dir_all(submodule_include.join("cugraph_c")).expect("create submodule include");

    let pkg_config = root.path().join("fake-pkg-config");
    write_executable_script(&pkg_config, &format!("#!/bin/sh\nprintf '%s\\n' '-I{}'\n", include_dir.display()));

    let resolved = build_support::resolve_include_dir_with_pkg_config(
        None,
        Some(pkg_config.to_str().unwrap()),
        &submodule_include,
    )
    .expect("resolve include via pkg-config");

    assert_eq!(resolved, include_dir.canonicalize().expect("canonical pkg-config include"));
}

#[test]
fn include_resolution_fails_when_no_fallback_candidate_contains_headers() {
    let root = TestDir::new();
    let include_dir = root.path().join("missing-headers");
    fs::create_dir_all(&include_dir).expect("create include dir");

    let err = build_support::resolve_include_dir(None, &include_dir).unwrap_err();

    assert!(err.contains("cugraph_c"));
    assert!(err.contains("CUGRAPH_INCLUDE_DIR"));
    assert!(err.contains(&include_dir.display().to_string()));
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
fn library_resolution_prefers_pkg_config_before_submodule_fallback() {
    let root = TestDir::new();
    let pkg_lib = root.path().join("pkgconfig-lib");
    fs::create_dir_all(&pkg_lib).expect("create pkgconfig lib");
    create_required_library_files(&pkg_lib);

    let submodule_lib = root.path().join("submodule-lib");
    fs::create_dir_all(&submodule_lib).expect("create submodule lib");
    create_required_library_files(&submodule_lib);

    let pkg_config = root.path().join("fake-pkg-config");
    write_executable_script(&pkg_config, &format!("#!/bin/sh\nprintf '%s\\n' '-L{}'\n", pkg_lib.display()));

    let resolved =
        build_support::resolve_library_dir_with_pkg_config(None, Some(pkg_config.to_str().unwrap()), &[submodule_lib])
            .expect("resolve library via pkg-config");

    assert_eq!(resolved, pkg_lib.canonicalize().expect("canonical pkg-config lib"));
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
fn library_resolution_accepts_macos_style_library_names() {
    let root = TestDir::new();
    let lib_dir = root.path().join("macos-lib");
    fs::create_dir_all(&lib_dir).expect("create macos-lib");
    for lib in ["libcugraph.dylib", "libcugraph_c.dylib"] {
        fs::write(lib_dir.join(lib), []).unwrap_or_else(|_| panic!("create test library file {lib}"));
    }

    let resolved = build_support::resolve_library_dir(None, &[lib_dir.clone()]).expect("resolve dylib fallback");

    assert_eq!(resolved, lib_dir.canonicalize().expect("canonical dylib path"));
}

#[test]
fn conda_prefix_candidates_cover_unix_and_windows_layouts() {
    let prefix = Path::new("/tmp/fake-conda-prefix");
    let candidates = build_support::conda_prefix_library_candidates(prefix);

    assert_eq!(candidates, vec![prefix.to_path_buf(), prefix.join("lib"), prefix.join("Library").join("lib"),]);
}

#[test]
fn conda_include_candidates_cover_unix_and_windows_layouts() {
    let prefix = Path::new("/tmp/fake-conda-prefix");
    let candidates = build_support::conda_prefix_include_candidates(prefix);

    assert_eq!(candidates, vec![prefix.join("include"), prefix.join("Library").join("include"),]);
}

#[test]
fn pkg_config_probe_uses_selected_command_and_parses_paths() {
    let root = TestDir::new();
    let include_dir = root.path().join("pkg-inc");
    let lib_dir = root.path().join("pkg-lib");
    fs::create_dir_all(include_dir.join("cugraph_c")).expect("create include");
    fs::create_dir_all(&lib_dir).expect("create lib");

    let pkg_config = root.path().join("fake-pkg-config");
    write_executable_script(
        &pkg_config,
        &format!("#!/bin/sh\nprintf '%s\\n' '-I{} -L{}'\n", include_dir.display(), lib_dir.display()),
    );

    let probed = build_support::probe_pkg_config_paths(Some(pkg_config.to_str().unwrap()))
        .expect("probe pkg-config")
        .expect("pkg-config should succeed");

    assert_eq!(probed.include_paths, vec![include_dir.canonicalize().expect("canonical include")]);
    assert_eq!(probed.link_paths, vec![lib_dir.canonicalize().expect("canonical lib")]);
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

#[test]
fn detect_system_include_dir_uses_selected_compiler_output() {
    let root = TestDir::new();
    let include_dir = root.path().join("fake-include");
    fs::create_dir_all(&include_dir).expect("create include dir");
    fs::write(include_dir.join("stddef.h"), []).expect("create stddef.h");

    let compiler = root.path().join("fake-cc");
    write_executable_script(
        &compiler,
        &format!(
            "#!/bin/sh\ncat >/dev/null\ncat <<'EOF' 1>&2\n#include <...> search starts here:\n {}\nEnd of search list.\nEOF\n",
            include_dir.display()
        ),
    );

    let detected = build_support::detect_system_include_dir(Some(compiler.to_str().unwrap()))
        .expect("detect include dir from fake compiler");

    assert_eq!(detected, include_dir);
}
