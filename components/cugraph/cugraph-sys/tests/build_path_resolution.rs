#[path = "../build_support.rs"]
mod build_support;

use std::{
    env, fs,
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
        let path = env::temp_dir().join(unique);
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
    for lib in ["libcugraph.so", "libcugraph_c.so"] {
        fs::write(dir.join(lib), []).unwrap_or_else(|_| panic!("create test library file {lib}"));
    }
}

fn join_paths(paths: &[PathBuf]) -> std::ffi::OsString {
    env::join_paths(paths).expect("join test path list")
}

#[test]
fn native_build_contract_rejects_partial_env_contracts() {
    let root = TestDir::new();
    let include_dir = root.path().join("include");
    fs::create_dir_all(include_dir.join("cugraph_c")).expect("create include dir");

    let err =
        build_support::resolve_native_build_contract(root.path(), Some(join_paths(&[include_dir])), None, None, None)
            .unwrap_err();

    assert!(err.contains("CUGRAPH_NATIVE_INCLUDE_DIRS"));
    assert!(err.contains("CUGRAPH_NATIVE_LINK_DIRS"));
    assert!(err.contains("CUGRAPH_RUNTIME_LIB_DIRS"));
}

#[test]
fn native_include_dirs_are_required() {
    let err = build_support::resolve_native_include_dirs(None).unwrap_err();

    assert!(err.contains("CUGRAPH_NATIVE_INCLUDE_DIRS"));
    assert!(err.contains("required"));
}

#[test]
fn native_include_dirs_must_contain_cugraph_headers() {
    let root = TestDir::new();
    let include_dir = root.path().join("include");
    fs::create_dir_all(&include_dir).expect("create include dir");

    let err = build_support::resolve_native_include_dirs(Some(join_paths(std::slice::from_ref(&include_dir)))).unwrap_err();

    assert!(err.contains("CUGRAPH_NATIVE_INCLUDE_DIRS"));
    assert!(err.contains("cugraph_c"));
    assert!(err.contains(&include_dir.display().to_string()));
}

#[test]
fn native_link_dirs_are_required() {
    let err = build_support::resolve_native_link_dirs(None).unwrap_err();

    assert!(err.contains("CUGRAPH_NATIVE_LINK_DIRS"));
    assert!(err.contains("required"));
}

#[test]
fn native_link_dirs_must_contain_cugraph_libraries() {
    let root = TestDir::new();
    let link_dir = root.path().join("link");
    fs::create_dir_all(&link_dir).expect("create link dir");

    let err = build_support::resolve_native_link_dirs(Some(join_paths(std::slice::from_ref(&link_dir)))).unwrap_err();

    assert!(err.contains("CUGRAPH_NATIVE_LINK_DIRS"));
    assert!(err.contains("libcugraph"));
    assert!(err.contains("libcugraph_c"));
    assert!(err.contains(&link_dir.display().to_string()));
}

#[test]
fn runtime_lib_dirs_are_required() {
    let err = build_support::resolve_runtime_lib_dirs(None).unwrap_err();

    assert!(err.contains("CUGRAPH_RUNTIME_LIB_DIRS"));
    assert!(err.contains("required"));
}

#[test]
fn runtime_lib_dirs_must_contain_cugraph_libraries() {
    let root = TestDir::new();
    let runtime_dir = root.path().join("runtime");
    fs::create_dir_all(&runtime_dir).expect("create runtime dir");

    let err = build_support::resolve_runtime_lib_dirs(Some(join_paths(std::slice::from_ref(&runtime_dir)))).unwrap_err();

    assert!(err.contains("CUGRAPH_RUNTIME_LIB_DIRS"));
    assert!(err.contains("libcugraph"));
    assert!(err.contains("libcugraph_c"));
    assert!(err.contains(&runtime_dir.display().to_string()));
}

#[test]
fn native_build_contract_uses_explicit_path_lists_without_extending_build_tree() {
    let root = TestDir::new();
    let include_dir = root.path().join("explicit").join("include");
    let dep_include = root.path().join("explicit").join("raft").join("include");
    let link_dir = root.path().join("explicit").join("lib");
    let runtime_dir = root.path().join("explicit").join("runtime");
    let build_dir = root.path().join("explicit").join("build");
    let deps_dir = build_dir.join("_deps").join("cccl-src").join("thrust");

    fs::create_dir_all(include_dir.join("cugraph_c")).expect("create cugraph include");
    fs::create_dir_all(dep_include.join("raft")).expect("create dependency include");
    fs::create_dir_all(&link_dir).expect("create link dir");
    fs::create_dir_all(&runtime_dir).expect("create runtime dir");
    fs::create_dir_all(deps_dir.join("thrust")).expect("create fake _deps tree");
    create_required_library_files(&link_dir);
    create_required_library_files(&runtime_dir);

    let contract = build_support::resolve_native_build_contract(
        root.path(),
        Some(join_paths(&[include_dir.clone(), dep_include.clone()])),
        Some(join_paths(std::slice::from_ref(&link_dir))),
        Some(join_paths(std::slice::from_ref(&runtime_dir))),
        None,
    )
    .expect("resolve explicit contract");

    assert_eq!(
        contract.native_include_dirs,
        vec![
            include_dir.canonicalize().expect("canonical cugraph include"),
            dep_include.canonicalize().expect("canonical dependency include"),
        ]
    );
    assert_eq!(contract.native_link_dirs, vec![link_dir.canonicalize().expect("canonical link dir")]);
    assert_eq!(contract.runtime_lib_dirs, vec![runtime_dir.canonicalize().expect("canonical runtime dir")]);
}

#[test]
fn cmake_export_dir_uses_explicit_override_when_present() {
    let root = TestDir::new();
    let manifest_dir = root.path().join("manifest");
    let build_dir = root.path().join("native-build");

    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&build_dir).expect("create build dir");
    fs::write(build_dir.join("cugraph-config.cmake"), []).expect("write config file");

    let resolved = build_support::resolve_cmake_export_dir(&manifest_dir, Some(build_dir.as_os_str().to_os_string()))
        .expect("resolve explicit cmake build dir");

    assert_eq!(resolved, build_dir.canonicalize().expect("canonical build dir"));
}

#[test]
fn cmake_export_dir_uses_workspace_local_build_tree_when_available() {
    let root = TestDir::new();
    let manifest_dir = root.path().join("repo").join("cugraph-sys");
    let build_dir = root.path().join("repo").join("cugraph").join("cpp").join("build");

    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&build_dir).expect("create build dir");
    fs::write(build_dir.join("cugraph-config.cmake"), []).expect("write config file");

    let resolved = build_support::resolve_cmake_export_dir(&manifest_dir, None).expect("resolve local cmake build dir");

    assert_eq!(resolved, build_dir.canonicalize().expect("canonical build dir"));
}

#[test]
fn cmake_probe_parses_generated_include_and_link_outputs() {
    let root = TestDir::new();
    let include_a = root.path().join("include-a");
    let include_b = root.path().join("include-b");
    let link_a = root.path().join("lib-a");
    let link_b = root.path().join("lib-b");
    let runtime_a = root.path().join("runtime-a");
    let runtime_b = root.path().join("runtime-b");

    for dir in [&include_a, &include_b, &link_a, &link_b, &runtime_a, &runtime_b] {
        fs::create_dir_all(dir).expect("create probe dir");
    }

    let include_output = format!("{}\n{}\n{}\n", include_a.display(), include_b.display(), include_a.display());
    let includes = build_support::parse_cmake_generated_path_list(&include_output).expect("parse include output");
    assert_eq!(
        includes,
        vec![
            include_a.canonicalize().expect("canonical include a"),
            include_b.canonicalize().expect("canonical include b"),
        ]
    );

    let link_command = format!(
        "/usr/bin/c++ probe.o -o probe -Wl,-rpath,{}:{} {}/libcugraph_c.so -L{} {}/librmm.so",
        runtime_a.display(),
        runtime_b.display(),
        link_a.display(),
        link_b.display(),
        link_b.display()
    );
    let parsed = build_support::parse_cmake_link_command(&link_command).expect("parse link command");

    assert_eq!(
        parsed.link_dirs,
        vec![link_a.canonicalize().expect("canonical link a"), link_b.canonicalize().expect("canonical link b"),]
    );
    assert_eq!(
        parsed.runtime_dirs,
        vec![
            runtime_a.canonicalize().expect("canonical runtime a"),
            runtime_b.canonicalize().expect("canonical runtime b"),
        ]
    );
}

#[test]
fn primary_cugraph_include_dir_returns_the_path_with_cugraph_headers() {
    let root = TestDir::new();
    let dep_include = root.path().join("deps").join("raft").join("include");
    let cugraph_include = root.path().join("include");

    fs::create_dir_all(dep_include.join("raft")).expect("create dependency include");
    fs::create_dir_all(cugraph_include.join("cugraph_c")).expect("create cugraph include");

    let include_dirs = vec![
        dep_include.canonicalize().expect("canonical dep include"),
        cugraph_include.canonicalize().expect("canonical cugraph include"),
    ];

    let resolved = build_support::primary_cugraph_include_dir(&include_dirs).expect("primary include dir");

    assert_eq!(resolved, cugraph_include.canonicalize().expect("canonical cugraph include"));
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
    fs::write(
        &compiler,
        format!(
            "#!/bin/sh\ncat >/dev/null\ncat <<'EOF' 1>&2\n#include <...> search starts here:\n {}\nEnd of search list.\nEOF\n",
            include_dir.display()
        ),
    )
    .expect("write fake compiler");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(&compiler).expect("stat fake compiler").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&compiler, permissions).expect("chmod fake compiler");
    }

    let detected = build_support::detect_system_include_dir(Some(compiler.to_str().unwrap()))
        .expect("detect include dir from fake compiler");

    assert_eq!(detected, include_dir);
}
