#[path = "../build_support.rs"]
mod build_support;

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const TEST_CUVS_REVISION: &str = "2bd7cd71d31e39eb9ee00c2a250085a1afb84977";

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

fn create_cuvs_install(prefix: &Path) -> PathBuf {
    let config_dir = prefix.join("lib/cmake/cuvs");
    fs::create_dir_all(&config_dir).expect("create cuVS CMake package dir");
    fs::write(config_dir.join("cuvs-config.cmake"), []).expect("write cuVS CMake package config");
    fs::write(prefix.join("lib/libcuvs.so"), []).expect("write libcuvs.so");
    prefix.to_path_buf()
}

fn create_cudf_public_headers(prefix: &Path) {
    let include_dir = prefix.join("include/cudf_rust");
    fs::create_dir_all(&include_dir).expect("create cudf_rust public include dir");
    fs::write(include_dir.join("error_boundary.hpp"), "// test attributed exception header\n")
        .expect("write error boundary header");
    fs::write(include_dir.join("memory_resource.h"), "// test memory resource header\n")
        .expect("write memory resource header");
}

fn workspace_manifest_dir(root: &Path) -> PathBuf {
    root.join("components/cugraph/cugraph-nexus-sys")
}

fn create_root_cuvs_pin(workspace_root: &Path) {
    let source_parent = workspace_root.join("components/cuvs");
    fs::create_dir_all(&source_parent).expect("create root cuVS source parent");
    let init =
        Command::new("git").arg("-C").arg(workspace_root).args(["init", "--quiet"]).status().expect("run git init");
    assert!(init.success(), "initialize root repository");
    let cacheinfo = format!("160000,{TEST_CUVS_REVISION},components/cuvs/cuvs");
    let update = Command::new("git")
        .arg("-C")
        .arg(workspace_root)
        .args(["update-index", "--add", "--cacheinfo", &cacheinfo])
        .status()
        .expect("run git update-index");
    assert!(update.success(), "record root cuVS gitlink");
}

fn create_default_cuvs_install(workspace_root: &Path) -> PathBuf {
    create_root_cuvs_pin(workspace_root);
    let install = create_cuvs_install(&workspace_root.join("target/native/cuvs-install"));
    let marker = install.join("share/cuvs-nexus/source-revision");
    fs::create_dir_all(marker.parent().expect("source marker parent")).expect("create source marker directory");
    fs::write(marker, format!("{TEST_CUVS_REVISION}\n")).expect("write source marker");
    install
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
fn native_build_contract_rejects_multiple_contract_sources() {
    let root = TestDir::new();
    let include_dir = root.path().join("include");
    let link_dir = root.path().join("link");
    let runtime_dir = root.path().join("runtime");
    let build_dir = root.path().join("build");
    fs::create_dir_all(include_dir.join("cugraph_c")).expect("create include dir");
    fs::create_dir_all(&link_dir).expect("create link dir");
    fs::create_dir_all(&runtime_dir).expect("create runtime dir");
    fs::create_dir_all(&build_dir).expect("create build dir");
    create_required_library_files(&link_dir);
    create_required_library_files(&runtime_dir);
    fs::write(build_dir.join("cugraph-config.cmake"), []).expect("write config file");

    let err = build_support::resolve_native_build_contract(
        root.path(),
        Some(join_paths(&[include_dir])),
        Some(join_paths(&[link_dir])),
        Some(join_paths(&[runtime_dir])),
        Some(build_dir.into_os_string()),
    )
    .unwrap_err();

    assert!(err.contains("exactly one source"));
    assert!(err.contains("CUGRAPH_CMAKE_BUILD_DIR"));
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

    let err =
        build_support::resolve_native_include_dirs(Some(join_paths(std::slice::from_ref(&include_dir)))).unwrap_err();

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

    let err =
        build_support::resolve_runtime_lib_dirs(Some(join_paths(std::slice::from_ref(&runtime_dir)))).unwrap_err();

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
    let manifest_dir = workspace_manifest_dir(root.path());
    let build_dir = root.path().join("target/native/cugraph-build");

    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&build_dir).expect("create build dir");
    fs::write(build_dir.join("cugraph-config.cmake"), []).expect("write config file");

    let resolved = build_support::resolve_cmake_export_dir(&manifest_dir, None).expect("resolve local cmake build dir");

    assert_eq!(resolved, build_dir.canonicalize().expect("canonical build dir"));
}

#[test]
fn prepare_workspace_cmake_export_dir_reuses_existing_canonical_build() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cpp_dir = root.path().join("components/cugraph/cugraph/cpp");
    let build_dir = root.path().join("target/native/cugraph-build");
    let cudf_install = root.path().join("target/native/cudf-install");
    let cuvs_install = create_default_cuvs_install(root.path());
    let rmm_dir = cudf_install.join("lib").join("cmake").join("rmm");
    let logger_dir = cudf_install.join("lib").join("cmake").join("rapids_logger");
    let cuvs_dir = cuvs_install.join("lib/cmake/cuvs");

    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&cpp_dir).expect("create cpp dir");
    fs::create_dir_all(&build_dir).expect("create build dir");
    create_cudf_public_headers(&cudf_install);
    fs::create_dir_all(&rmm_dir).expect("create rmm cmake dir");
    fs::create_dir_all(&logger_dir).expect("create rapids_logger cmake dir");
    fs::write(cpp_dir.join("CMakeLists.txt"), []).expect("write cmake lists");
    fs::write(build_dir.join("cugraph-config.cmake"), []).expect("write cugraph config");
    fs::write(
        build_dir.join("CMakeCache.txt"),
        format!(
            "CMAKE_CUDA_ARCHITECTURES:STRING=NATIVE\nrmm_DIR:PATH={}\nrapids_logger_DIR:PATH={}\nCUGRAPH_CUDF_RUST_INCLUDE_DIR:PATH={}\ncuvs_DIR:PATH={}\n",
            rmm_dir.display(),
            logger_dir.display(),
            cudf_install.join("include").display(),
            cuvs_dir.display()
        ),
    )
    .expect("write cache");
    fs::write(rmm_dir.join("rmm-config.cmake"), []).expect("write rmm config");
    fs::write(logger_dir.join("rapids_logger-config.cmake"), []).expect("write logger config");

    let resolved = build_support::prepare_workspace_cmake_export_dir(&manifest_dir, None, None, None, None, None)
        .expect("reuse existing canonical native build");

    assert_eq!(resolved, build_dir.canonicalize().expect("canonical build dir"));
}

#[test]
fn prepare_workspace_cmake_export_dir_honors_auto_build_opt_out() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cpp_dir = root.path().join("components/cugraph/cugraph/cpp");
    let cudf_install = root.path().join("target/native/cudf-install");
    create_default_cuvs_install(root.path());
    let rmm_dir = cudf_install.join("lib").join("cmake").join("rmm");
    let logger_dir = cudf_install.join("lib").join("cmake").join("rapids_logger");

    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&cpp_dir).expect("create cpp dir");
    create_cudf_public_headers(&cudf_install);
    fs::create_dir_all(&rmm_dir).expect("create rmm cmake dir");
    fs::create_dir_all(&logger_dir).expect("create rapids_logger cmake dir");
    fs::write(cpp_dir.join("CMakeLists.txt"), []).expect("write cmake lists");
    fs::write(rmm_dir.join("rmm-config.cmake"), []).expect("write rmm config");
    fs::write(logger_dir.join("rapids_logger-config.cmake"), []).expect("write logger config");

    let err =
        build_support::prepare_workspace_cmake_export_dir(&manifest_dir, None, None, None, Some("0".into()), None)
            .unwrap_err();

    assert!(err.contains("CUGRAPH_SYS_AUTO_BUILD"));
    assert!(err.contains("disabled"));
}

#[test]
fn prepare_workspace_cmake_export_dir_rejects_stale_cuda_arch_when_auto_build_is_disabled() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cpp_dir = root.path().join("components/cugraph/cugraph/cpp");
    let build_dir = root.path().join("target/native/cugraph-build");
    let cudf_install = root.path().join("target/native/cudf-install");
    let cuvs_install = create_default_cuvs_install(root.path());
    let rmm_dir = cudf_install.join("lib").join("cmake").join("rmm");
    let logger_dir = cudf_install.join("lib").join("cmake").join("rapids_logger");
    let cuvs_dir = cuvs_install.join("lib/cmake/cuvs");

    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(&cpp_dir).expect("create cpp dir");
    fs::create_dir_all(&build_dir).expect("create build dir");
    create_cudf_public_headers(&cudf_install);
    fs::create_dir_all(&rmm_dir).expect("create rmm cmake dir");
    fs::create_dir_all(&logger_dir).expect("create rapids_logger cmake dir");
    fs::write(cpp_dir.join("CMakeLists.txt"), []).expect("write cmake lists");
    fs::write(build_dir.join("cugraph-config.cmake"), []).expect("write cugraph config");
    fs::write(
        build_dir.join("CMakeCache.txt"),
        format!(
            "CMAKE_CUDA_ARCHITECTURES:STRING=80\nrmm_DIR:PATH={}\nrapids_logger_DIR:PATH={}\nCUGRAPH_CUDF_RUST_INCLUDE_DIR:PATH={}\ncuvs_DIR:PATH={}\n",
            rmm_dir.display(),
            logger_dir.display(),
            cudf_install.join("include").display(),
            cuvs_dir.display()
        ),
    )
    .expect("write cache");
    fs::write(rmm_dir.join("rmm-config.cmake"), []).expect("write rmm config");
    fs::write(logger_dir.join("rapids_logger-config.cmake"), []).expect("write logger config");

    let err = build_support::prepare_workspace_cmake_export_dir(
        &manifest_dir,
        None,
        None,
        None,
        Some("0".into()),
        Some("90".into()),
    )
    .unwrap_err();

    assert!(err.contains("CMAKE_CUDA_ARCHITECTURES"));
    assert!(err.contains("CUGRAPH_SYS_AUTO_BUILD"));
}

#[test]
fn cudf_public_headers_must_be_regular_installed_files() {
    let root = TestDir::new();
    let cudf_install = root.path().join("cudf-install");
    create_cudf_public_headers(&cudf_install);

    build_support::validate_cudf_public_headers(&cudf_install).expect("validate installed cudf public headers");

    fs::remove_file(cudf_install.join("include/cudf_rust/error_boundary.hpp")).expect("remove header");
    let missing = build_support::validate_cudf_public_headers(&cudf_install).unwrap_err();
    assert!(missing.contains("error_boundary.hpp"));
    assert!(missing.contains("manual symlink"));

    create_cudf_public_headers(&cudf_install);
    let source = root.path().join("source-error-boundary.hpp");
    fs::write(&source, "// source header\n").expect("write source header");
    fs::remove_file(cudf_install.join("include/cudf_rust/error_boundary.hpp")).expect("remove regular header");
    std::os::unix::fs::symlink(&source, cudf_install.join("include/cudf_rust/error_boundary.hpp"))
        .expect("create target-local symlink");

    let symlinked = build_support::validate_cudf_public_headers(&cudf_install).unwrap_err();
    assert!(symlinked.contains("not a regular installed file"));
    assert!(symlinked.contains("manual symlinks"));
}

#[test]
fn cmake_package_prefixes_replay_native_build_roots() {
    let root = TestDir::new();
    let build_dir = root.path().join("build");
    let cudf_install = root.path().join("cudf-install");
    let raft_prefix = root.path().join("raft-prefix");
    let missing_prefix = root.path().join("missing-prefix");

    fs::create_dir_all(&build_dir).expect("create build dir");
    fs::create_dir_all(&cudf_install).expect("create cudf install prefix");
    fs::create_dir_all(&raft_prefix).expect("create raft prefix");
    fs::write(
        build_dir.join("CMakeCache.txt"),
        format!(
            r#"# comment
// another comment
rmm_ROOT:UNINITIALIZED={cudf}
nvtx3_ROOT:UNINITIALIZED={cudf}
raft_ROOT:PATH={raft}
CMAKE_PREFIX_PATH:STRING={cudf};{missing}
ignored_DIR:PATH={missing}
missing_ROOT:PATH={missing}-NOTFOUND
"#,
            cudf = cudf_install.display(),
            raft = raft_prefix.display(),
            missing = missing_prefix.display(),
        ),
    )
    .expect("write CMake cache");

    let prefixes = build_support::cmake_package_prefixes_from_cache(&build_dir).expect("read package prefixes");

    assert_eq!(
        prefixes,
        vec![
            cudf_install.canonicalize().expect("canonical cudf install"),
            raft_prefix.canonicalize().expect("canonical raft prefix"),
        ]
    );
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
fn canonical_cuvs_contract_resolves_cmake_package_and_library_identity() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_cuvs_install(&root.path().join("canonical-cuvs"));
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");

    let contract =
        build_support::resolve_canonical_cuvs_contract(&manifest_dir, Some(cuvs_install.as_os_str().to_os_string()))
            .expect("resolve canonical cuVS contract");

    assert_eq!(contract.install_dir, cuvs_install.canonicalize().expect("canonical cuVS prefix"));
    assert_eq!(contract.cmake_config_dir, cuvs_install.join("lib/cmake/cuvs"));
    assert_eq!(contract.runtime_lib_dir, cuvs_install.join("lib"));
    assert_eq!(contract.library_selector_path, cuvs_install.join("lib/libcuvs.so"));
    assert_eq!(contract.library_path, cuvs_install.join("lib/libcuvs.so"));
    assert!(contract.source_revision_inputs.is_empty());
}

#[test]
fn canonical_cuvs_contract_requires_matching_default_source_marker() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_default_cuvs_install(root.path());
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");

    let contract =
        build_support::resolve_canonical_cuvs_contract(&manifest_dir, None).expect("resolve default cuVS contract");

    assert_eq!(contract.install_dir, cuvs_install.canonicalize().expect("canonical default cuVS prefix"));
    assert_eq!(contract.source_revision_inputs.len(), 2);
    assert!(contract.source_revision_inputs.iter().any(|path| path.ends_with("source-revision")));
    assert!(contract.source_revision_inputs.iter().any(|path| path.ends_with("index")));
}

#[test]
fn canonical_cuvs_contract_rejects_missing_or_stale_default_source_marker() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_default_cuvs_install(root.path());
    let marker = cuvs_install.join("share/cuvs-nexus/source-revision");
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");

    fs::remove_file(&marker).expect("remove source marker");
    let missing = build_support::resolve_canonical_cuvs_contract(&manifest_dir, None).unwrap_err();
    assert!(missing.contains("missing its pinned source marker"));

    fs::write(&marker, "stale-revision\n").expect("write stale source marker");
    let stale = build_support::resolve_canonical_cuvs_contract(&manifest_dir, None).unwrap_err();
    assert!(stale.contains("stale-revision"));
    assert!(stale.contains(TEST_CUVS_REVISION));
}

#[test]
fn canonical_cuvs_contract_rejects_default_without_a_stage_zero_gitlink() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_cuvs_install(&root.path().join("target/native/cuvs-install"));
    let marker = cuvs_install.join("share/cuvs-nexus/source-revision");
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::create_dir_all(marker.parent().expect("source marker parent")).expect("create source marker directory");
    fs::write(marker, format!("{TEST_CUVS_REVISION}\n")).expect("write source marker");
    let init = Command::new("git").arg("-C").arg(root.path()).args(["init", "--quiet"]).status().expect("run git init");
    assert!(init.success(), "initialize root repository");

    let err = build_support::resolve_canonical_cuvs_contract(&manifest_dir, None).unwrap_err();

    assert!(err.contains("exactly one stage-0 gitlink"));
    assert!(err.contains("components/cuvs/cuvs"));
}

#[cfg(unix)]
#[test]
fn canonical_cuvs_contract_does_not_treat_a_default_prefix_symlink_as_external() {
    use std::os::unix::fs::symlink;

    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_default_cuvs_install(root.path());
    let alias = root.path().join("explicit-cuvs-alias");
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    symlink(&cuvs_install, &alias).expect("create explicit prefix alias");
    fs::write(cuvs_install.join("share/cuvs-nexus/source-revision"), "stale-revision\n")
        .expect("write stale source marker");

    let err = build_support::resolve_canonical_cuvs_contract(&manifest_dir, Some(alias.into_os_string())).unwrap_err();

    assert!(err.contains("stale-revision"));
    assert!(err.contains(TEST_CUVS_REVISION));
}

#[test]
fn canonical_cuvs_contract_rejects_missing_shared_library() {
    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_cuvs_install(&root.path().join("missing-lib"));
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    fs::remove_file(cuvs_install.join("lib/libcuvs.so")).expect("remove libcuvs fixture");

    let err =
        build_support::resolve_canonical_cuvs_contract(&manifest_dir, Some(cuvs_install.as_os_str().to_os_string()))
            .unwrap_err();

    assert!(err.contains("libcuvs.so"));
    assert!(err.contains("CUVS_INSTALL_DIR"));
}

#[cfg(unix)]
#[test]
fn canonical_cuvs_contract_preserves_selector_and_refreshes_symlink_identity() {
    use std::os::unix::fs::symlink;

    let root = TestDir::new();
    let manifest_dir = workspace_manifest_dir(root.path());
    let cuvs_install = create_cuvs_install(&root.path().join("external-cuvs"));
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    let selector = cuvs_install.join("lib/libcuvs.so");
    let version_one = cuvs_install.join("lib/libcuvs.so.one");
    let version_two = cuvs_install.join("lib/libcuvs.so.two");
    fs::remove_file(&selector).expect("replace fixture with symlink");
    fs::write(&version_one, []).expect("create first libcuvs identity");
    fs::write(&version_two, []).expect("create second libcuvs identity");
    symlink(&version_one, &selector).expect("link first libcuvs identity");

    let first =
        build_support::resolve_canonical_cuvs_contract(&manifest_dir, Some(cuvs_install.as_os_str().to_os_string()))
            .expect("resolve first identity");
    fs::remove_file(&selector).expect("remove first selector");
    symlink(&version_two, &selector).expect("link second libcuvs identity");
    let second =
        build_support::resolve_canonical_cuvs_contract(&manifest_dir, Some(cuvs_install.as_os_str().to_os_string()))
            .expect("resolve second identity");

    assert_eq!(first.library_selector_path, selector);
    assert_eq!(second.library_selector_path, selector);
    assert_eq!(first.library_path, version_one.canonicalize().unwrap());
    assert_eq!(second.library_path, version_two.canonicalize().unwrap());
}

#[test]
fn loader_contract_accepts_exact_canonical_cuvs_identity() {
    let root = TestDir::new();
    let canonical_lib = root.path().join("canonical/lib/libcuvs.so.26");
    fs::create_dir_all(canonical_lib.parent().unwrap()).expect("create canonical lib dir");
    fs::write(&canonical_lib, []).expect("create canonical libcuvs");
    let readelf = " 0x0000000000000001 (NEEDED) Shared library: [libcuvs.so]\n";
    let ldd = format!("libcuvs.so => {} (0x0000)\n", canonical_lib.display());

    let resolved = build_support::validate_cuvs_loader_outputs(readelf, &ldd, &canonical_lib)
        .expect("validate exact canonical loader identity");

    assert_eq!(resolved, canonical_lib.canonicalize().expect("canonical libcuvs"));
}

#[test]
fn loader_contract_rejects_private_cuvs_and_unresolved_dependencies() {
    let root = TestDir::new();
    let canonical_lib = root.path().join("canonical/lib/libcuvs.so");
    let private_lib = root.path().join("build/_deps/cuvs-build/libcuvs.so");
    fs::create_dir_all(canonical_lib.parent().unwrap()).expect("create canonical lib dir");
    fs::create_dir_all(private_lib.parent().unwrap()).expect("create private lib dir");
    fs::write(&canonical_lib, []).expect("create canonical libcuvs");
    fs::write(&private_lib, []).expect("create private libcuvs");

    let private_readelf = format!(
        " 0x1 (NEEDED) Shared library: [libcuvs.so]\n 0x1 (RUNPATH) Library runpath: [{}]\n",
        private_lib.parent().unwrap().display()
    );
    let private_ldd = format!("libcuvs.so => {} (0x0000)\n", private_lib.display());
    let err = build_support::validate_cuvs_loader_outputs(&private_readelf, &private_ldd, &canonical_lib).unwrap_err();
    assert!(err.contains("private _deps"));

    let clean_readelf = " 0x1 (NEEDED) Shared library: [libcuvs.so]\n";
    let unresolved_ldd = "libcuvs.so => not found\nlibraft.so => not found\n";
    let err = build_support::validate_cuvs_loader_outputs(clean_readelf, unresolved_ldd, &canonical_lib).unwrap_err();
    assert!(err.contains("unresolved loader dependencies"));
    assert!(err.contains("libraft.so"));
}

#[test]
fn loader_contract_allows_non_cuvs_deps_under_a_build_path_named_cuvs() {
    let root = TestDir::new();
    let canonical_lib = root.path().join("canonical/lib/libcuvs.so");
    fs::create_dir_all(canonical_lib.parent().unwrap()).expect("create canonical lib dir");
    fs::write(&canonical_lib, []).expect("create canonical libcuvs");
    let runpath = root.path().join("build-canonical-cuvs/_deps/raft-build");
    let readelf = format!(
        " 0x1 (NEEDED) Shared library: [libcuvs.so]\n 0x1 (RUNPATH) Library runpath: [{}]\n",
        runpath.display()
    );
    let ldd = format!("libcuvs.so => {} (0x0)\n", canonical_lib.display());

    let resolved = build_support::validate_cuvs_loader_outputs(&readelf, &ldd, &canonical_lib)
        .expect("RAFT _deps path must not be conflated with private cuVS");

    assert_eq!(resolved, canonical_lib.canonicalize().unwrap());
}

#[test]
fn loader_contract_rejects_noncanonical_resolution() {
    let root = TestDir::new();
    let canonical_lib = root.path().join("canonical/lib/libcuvs.so");
    let other_lib = root.path().join("other/lib/libcuvs.so");
    fs::create_dir_all(canonical_lib.parent().unwrap()).expect("create canonical lib dir");
    fs::create_dir_all(other_lib.parent().unwrap()).expect("create other lib dir");
    fs::write(&canonical_lib, []).expect("create canonical libcuvs");
    fs::write(&other_lib, []).expect("create other libcuvs");
    let readelf = " 0x1 (NEEDED) Shared library: [libcuvs.so]\n";
    let ldd = format!("libcuvs.so => {} (0x0000)\n", other_lib.display());

    let err = build_support::validate_cuvs_loader_outputs(readelf, &ldd, &canonical_lib).unwrap_err();

    assert!(err.contains("instead of canonical"));
    assert!(err.contains(&other_lib.display().to_string()));
}

#[test]
fn controlled_loader_search_path_prepends_declared_contract_and_preserves_ambient_paths() {
    let root = TestDir::new();
    let runtime_a = root.path().join("runtime-a");
    let runtime_b = root.path().join("runtime-b");
    let cuvs_runtime = root.path().join("cuvs-runtime");
    let ambient = root.path().join("ambient");
    let ambient_value = join_paths(&[ambient.clone(), runtime_a.clone()]);

    let encoded = build_support::controlled_loader_search_path(
        &[runtime_a.clone(), runtime_b.clone()],
        &cuvs_runtime,
        Some(ambient_value),
    )
    .expect("construct controlled loader path");

    assert_eq!(env::split_paths(&encoded).collect::<Vec<_>>(), vec![runtime_a, runtime_b, cuvs_runtime, ambient]);
}

#[test]
fn ambient_loader_check_allows_missing_cuvs_but_rejects_shadowing() {
    let root = TestDir::new();
    let canonical_lib = root.path().join("canonical/libcuvs.so");
    let shadow_lib = root.path().join("shadow/libcuvs.so");
    fs::create_dir_all(canonical_lib.parent().unwrap()).expect("create canonical dir");
    fs::create_dir_all(shadow_lib.parent().unwrap()).expect("create shadow dir");
    fs::write(&canonical_lib, []).expect("create canonical libcuvs");
    fs::write(&shadow_lib, []).expect("create shadow libcuvs");

    build_support::validate_ambient_cuvs_resolution("libcuvs.so => not found\n", &canonical_lib)
        .expect("ambient absence is allowed because controlled closure is authoritative");
    let shadow_output = format!("libcuvs.so => {} (0x0)\n", shadow_lib.display());
    let err = build_support::validate_ambient_cuvs_resolution(&shadow_output, &canonical_lib).unwrap_err();
    assert!(err.contains("shadows canonical libcuvs"));
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
