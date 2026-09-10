use std::{
    env, fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

struct ManualBuildFixture {
    root: PathBuf,
    script: PathBuf,
    workspace: PathBuf,
}

impl ManualBuildFixture {
    fn new(marker: Option<&str>) -> Self {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("system time").as_nanos();
        let root = env::temp_dir().join(format!("cugraph-manual-contract-{}-{unique}", std::process::id()));
        let workspace = root.join("workspace");
        let repo = workspace.join("components/cugraph/cugraph");
        let script = workspace.join("scripts/build/build_libcugraph.sh");
        let default_install = workspace.join("target/native/cuvs-install");
        fs::create_dir_all(script.parent().expect("script parent")).expect("create fixture scripts directory");
        fs::create_dir_all(&repo).expect("create fixture upstream directory");
        fs::create_dir_all(&default_install).expect("create fixture default cuVS prefix");
        fs::copy(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/build/build_libcugraph.sh"), &script)
            .expect("copy manual build script");
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/build/cuda_architecture_contract.sh"),
            workspace.join("scripts/build/cuda_architecture_contract.sh"),
        )
        .expect("copy CUDA architecture contract helper");

        let upstream = repo.join("build.sh");
        fs::write(&upstream, "#!/usr/bin/env bash\nexit 0\n").expect("write fixture upstream build script");
        fs::set_permissions(&upstream, fs::Permissions::from_mode(0o755))
            .expect("make fixture build script executable");

        if let Some(marker) = marker {
            let marker_path = default_install.join("share/cuvs-nexus/source-revision");
            fs::create_dir_all(marker_path.parent().expect("marker parent")).expect("create marker parent");
            fs::write(marker_path, format!("{marker}\n")).expect("write source marker");
        }

        Self { root, script, workspace }
    }

    fn default_install(&self) -> PathBuf {
        self.workspace.join("target/native/cuvs-install")
    }

    fn run(&self, cuvs_install: Option<&Path>) -> Output {
        let mut command = Command::new(&self.script);
        command.env_remove("CUVS_INSTALL_DIR").env_remove("CUDF_INSTALL_DIR").env_remove("CUGRAPH_CPP_BUILD_DIR");
        if let Some(cuvs_install) = cuvs_install {
            command.env("CUVS_INSTALL_DIR", cuvs_install);
        }
        command.output().expect("run manual build contract fixture")
    }
}

impl Drop for ManualBuildFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn assert_stderr_contains(output: &Output, expected: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(expected), "expected stderr to contain '{expected}', got:\n{stderr}");
}

#[test]
fn native_build_scripts_share_one_cuda_architecture_contract() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let helper = manifest_dir.join("../../../scripts/build/cuda_architecture_contract.sh");
    let cudf_script = fs::read_to_string(manifest_dir.join("../../../scripts/build/build_libcudf.sh"))
        .expect("read cuDF build script");
    let cuvs_script = fs::read_to_string(manifest_dir.join("../../../scripts/build/build_libcuvs.sh"))
        .expect("read cuVS build script");
    let cugraph_script = read_manual_build_script();

    for (component, script) in [("cuDF", cudf_script), ("cuVS", cuvs_script), ("cuGraph", cugraph_script)] {
        assert!(script.contains("cuda_architecture_contract.sh"), "{component} must load the shared contract helper");
        assert!(script.contains("nexus_cuda_architectures"), "{component} must use the canonical selector");
        assert!(script.contains("nexus_cuda_write_contract"), "{component} must record its real cubin set");
    }

    let output = Command::new("bash")
        .args([
            "-c",
            "source \"$1\"; CUDF_CMAKE_CUDA_ARCHITECTURES=89 nexus_cuda_architectures",
            "architecture-contract-test",
        ])
        .arg(&helper)
        .output()
        .expect("run architecture selector contract");
    assert!(!output.status.success());
    assert_stderr_contains(&output, "Set CMAKE_CUDA_ARCHITECTURES once");

    let output = Command::new("bash")
        .args([
            "-c",
            "source \"$1\"; CMAKE_CUDA_ARCHITECTURES=native nexus_cuda_architectures",
            "architecture-contract-test",
        ])
        .arg(&helper)
        .output()
        .expect("normalize native architecture selector");
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout), "NATIVE\n");
}

#[test]
fn cuda_architecture_contract_rejects_mismatched_real_cubins() {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("system time").as_nanos();
    let root = env::temp_dir().join(format!("cuda-architecture-contract-{}-{unique}", std::process::id()));
    let cudf = root.join("cudf");
    let cuvs = root.join("cuvs");
    for (prefix, real) in [(&cudf, "120a"), (&cuvs, "89")] {
        let marker = prefix.join("share/nexus-native/cuda-architectures.env");
        fs::create_dir_all(marker.parent().expect("marker parent")).expect("create marker directory");
        fs::write(marker, format!("requested=NATIVE\nreal={real}\n")).expect("write architecture marker");
    }
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/build/cuda_architecture_contract.sh");
    let output = Command::new("bash")
        .args([
            "-c",
            "source \"$1\"; nexus_cuda_require_matching_contracts cuDF \"$2\" cuVS \"$3\" \"rebuild stack\"",
            "architecture-contract-test",
        ])
        .arg(helper)
        .arg(&cudf)
        .arg(&cuvs)
        .output()
        .expect("run architecture mismatch contract");
    let _ = fs::remove_dir_all(&root);

    assert!(!output.status.success());
    assert_stderr_contains(&output, "cuDF contains full-kernel real CUDA cubins '120a'");
    assert_stderr_contains(&output, "cuVS contains '89'");
    assert_stderr_contains(&output, "rebuild stack");
}

#[test]
fn cuda_architecture_contract_keeps_suffixed_targets_and_ignores_stub_sets() {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).expect("system time").as_nanos();
    let root = env::temp_dir().join(format!("cuda-architecture-cubins-{}-{unique}", std::process::id()));
    let toolkit = root.join("cuda");
    let cuobjdump = toolkit.join("bin/cuobjdump");
    let library = root.join("libcudf.so");
    fs::create_dir_all(cuobjdump.parent().expect("cuobjdump parent")).expect("create fake CUDA toolkit");
    fs::write(
        &cuobjdump,
        concat!(
            "#!/usr/bin/env bash\n",
            "printf '%s\\n' \\\n",
            "  'ELF file 1: libcudf.1.sm_89.cubin' \\\n",
            "  'ELF file 2: libcudf.2.sm_100f.cubin' \\\n",
            "  'ELF file 3: libcudf.3.sm_100f.cubin' \\\n",
            "  'ELF file 4: libcudf.4.sm_100f.cubin' \\\n",
            "  'ELF file 5: libcudf.5.sm_100f.cubin' \\\n",
            "  'ELF file 6: libcudf.6.sm_120a.cubin' \\\n",
            "  'ELF file 7: libcudf.7.sm_120a.cubin' \\\n",
            "  'ELF file 8: libcudf.8.sm_120a.cubin' \\\n",
            "  'ELF file 9: libcudf.9.sm_120a.cubin'\n",
        ),
    )
    .expect("write fake cuobjdump");
    fs::set_permissions(&cuobjdump, fs::Permissions::from_mode(0o755)).expect("make fake cuobjdump executable");
    fs::write(&library, []).expect("write fake library");
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/build/cuda_architecture_contract.sh");
    let output = Command::new("bash")
        .args(["-c", "source \"$1\"; nexus_cuda_real_architectures \"$2\"", "architecture-contract-test"])
        .arg(helper)
        .arg(&library)
        .env("CUDAToolkit_ROOT", &toolkit)
        .output()
        .expect("inspect fixture cubins");
    let _ = fs::remove_dir_all(&root);

    assert!(output.status.success(), "fixture inspection failed: {}", String::from_utf8_lossy(&output.stderr));
    // The `f` family suffix must survive the shell parser exactly as it does in
    // build_support.rs; dropping it silently desynchronizes the two writers.
    assert_eq!(String::from_utf8_lossy(&output.stdout), "100f;120a\n");
}

#[test]
fn cudf_native_build_owns_shim_header_installation() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bootstrap = fs::read_to_string(manifest_dir.join("../../../scripts/build/bootstrap_cudf_install.sh"))
        .expect("read cuDF bootstrap script");
    let native_build = fs::read_to_string(manifest_dir.join("../../../scripts/build/build_libcudf.sh"))
        .expect("read cuDF native build script");

    assert!(
        !bootstrap.contains("include/cudf_rust"),
        "the dependency bootstrap must not duplicate the shim CMake install"
    );
    assert!(native_build.contains("cmake --install \"${shim_build_dir}\""));
    for installed in [
        "include/cudf_rust/cudf_rust.h",
        "lib/libcudf_rust.so",
        "lib/libcudf_rust_rmm_interop.so",
        "share/cudf_rust/cargo-link.txt",
    ] {
        assert!(native_build.contains(installed), "cuDF native build must verify {installed}");
    }
}

fn read_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("build.rs")).expect("failed to read build.rs")
}

fn read_build_support_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("build_support.rs")).expect("failed to read build_support.rs")
}

fn read_cargo_toml() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("failed to read Cargo.toml")
}

fn read_downstream_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("../cugraph-nexus/build.rs")).expect("failed to read downstream build.rs")
}

fn read_manual_build_script() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("../../../scripts/build/build_libcugraph.sh"))
        .expect("failed to read scripts/build/build_libcugraph.sh")
}

fn read_upstream_cugraph_file(path: &str) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("../cugraph/cpp").join(path))
        .unwrap_or_else(|_| panic!("failed to read upstream cuGraph file {path}"))
}

fn read_test_source(path: &str) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join(path)).unwrap_or_else(|_| panic!("failed to read {path}"))
}

#[test]
fn crate_uses_checked_in_bindings_without_build_time_bindgen() {
    let build_rs = read_build_rs();
    let cargo_toml = read_cargo_toml();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lib_rs = fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("failed to read src/lib.rs");
    let bindings = fs::read_to_string(manifest_dir.join("src/bindings.rs")).expect("failed to read bindings");

    assert!(!build_rs.contains("bindgen::"));
    assert!(!cargo_toml.contains("bindgen.workspace"));
    assert!(lib_rs.contains("include!(\"bindings.rs\");"));
    assert!(bindings.contains("Generated by scripts/dev/bindings.sh cugraph"));
}

#[test]
fn build_rs_tracks_explicit_native_contract_env_changes() {
    let build_rs = read_build_rs();

    for env_var in [
        "CUGRAPH_NATIVE_INCLUDE_DIRS",
        "CUGRAPH_NATIVE_LINK_DIRS",
        "CUGRAPH_RUNTIME_LIB_DIRS",
        "CUGRAPH_CMAKE_BUILD_DIR",
        "CUGRAPH_CPP_BUILD_DIR",
        "CUGRAPH_SYS_AUTO_BUILD",
        "CUDF_INSTALL_DIR",
        "CUVS_INSTALL_DIR",
        "LD_LIBRARY_PATH",
        "CMAKE_CUDA_ARCHITECTURES",
    ] {
        assert!(
            build_rs.contains(&format!("cargo:rerun-if-env-changed={env_var}")),
            "build.rs should rerun when {env_var} changes because the explicit native contract depends on it"
        );
    }
}

#[test]
fn build_rs_does_not_track_discovery_fallback_env_vars() {
    let build_rs = read_build_rs();

    for env_var in [
        "CUGRAPH_INCLUDE_DIR",
        "CUGRAPH_LIBRARY_DIR",
        "CONDA_PREFIX",
        "LIBRARY_PREFIX",
        "PREFIX",
        "PKG_CONFIG",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
    ] {
        assert!(
            !build_rs.contains(&format!("cargo:rerun-if-env-changed={env_var}")),
            "build.rs should not track {env_var} because fallback discovery is no longer part of the contract"
        );
    }
}

#[test]
fn build_rs_emits_runtime_library_metadata_for_downstream_consumers() {
    let build_rs = read_build_rs();

    assert!(
        build_rs.contains("cargo:metadata=runtime-lib-dirs="),
        "build.rs should emit the multi-path runtime library metadata key for downstream consumers"
    );
    assert!(
        build_rs.contains("cargo:metadata=native-include-dirs="),
        "build.rs should emit native include metadata for downstream shim compilation"
    );
    assert!(
        build_rs.contains("cargo:metadata=native-link-dirs="),
        "build.rs should emit native link metadata for downstream shim dependencies"
    );
    for key in ["cuvs-install-dir", "cuvs-library-path", "cuvs-runtime-lib-dir"] {
        assert!(
            build_rs.contains(&format!("cargo::metadata={key}=")),
            "build.rs should export canonical cuVS identity metadata key {key}"
        );
    }
}

#[test]
fn prebuilt_contract_carries_cudf_include_for_rmm_headers() {
    let build_rs = read_build_rs();

    assert!(
        build_rs.contains("cudf_install_dir.join(\"include\")"),
        "build.rs should append the cuDF prefix include dir to the prebuilt native contract: \
         cugraph-install/include does not vendor the RMM headers the downstream shim includes"
    );
    assert!(
        build_rs.contains("cudf_include_dir.join(\"rapids\")"),
        "build.rs should append the cuDF-vendored CCCL dir: RMM requires CCCL 3.3+, which the toolkit includes cannot satisfy"
    );
    assert!(
        build_rs.contains("contract.native_include_dirs.push(include_dir)"),
        "build.rs should carry the cuDF include dirs in native-include-dirs, mirroring the cuDF runtime dir handling"
    );
    assert!(
        build_rs.contains("rmm/aligned.hpp"),
        "build.rs should fail fast when the cuDF prefix lacks RMM headers instead of surfacing a bare cc error"
    );
}

#[test]
fn manual_and_cargo_builds_share_runtime_cuvs_contract() {
    let build_rs = read_build_rs();
    let build_support = read_build_support_rs();
    let manual_script = read_manual_build_script();

    assert!(build_rs.contains("env::var_os(\"CUVS_INSTALL_DIR\")"));
    assert!(build_support.contains("-Dcuvs_ROOT={}"));
    assert!(manual_script.contains("CUVS_INSTALL_DIR"));
    assert!(manual_script.contains("-Dcuvs_ROOT=${cuvs_install_dir}"));
    assert!(manual_script.contains("-DCMAKE_PREFIX_PATH=${cudf_install_dir};${cuvs_install_dir}"));
    assert!(manual_script.contains("configured_prefix_path=\"$(cmake_cache_value_any_type CMAKE_PREFIX_PATH)\""));
    assert!(build_support.contains("cuvs_DIR"));
    assert!(manual_script.contains("check_package_dir cuvs_DIR"));
    assert!(build_support.contains("libcuvs.so"));
    assert!(manual_script.contains("libcuvs.so"));
    assert!(build_support.contains("-DCUGRAPH_USE_CUVS_STATIC=OFF"));
    assert!(manual_script.contains("-DCUGRAPH_USE_CUVS_STATIC=OFF"));
    assert!(
        build_support.contains("share/cuvs-nexus/source-revision"),
        "the Cargo resolver must validate the default cuVS source marker"
    );
    assert!(
        manual_script.contains("share/cuvs-nexus/source-revision"),
        "the manual build must validate the default cuVS source marker"
    );
    assert!(
        build_support.contains("components/cuvs/cuvs"),
        "the Cargo resolver must compare the default marker with the root cuVS stage-0 gitlink"
    );
    assert!(
        build_support.contains("installed_revision.trim() != *expected_revision"),
        "the Cargo resolver must reject a default source marker that differs from the root cuVS gitlink"
    );
    for pinned_fragment in ["ls-files --stage -- cuvs", "expected_cuvs_revision", "does not match sibling gitlink"] {
        assert!(
            !manual_script.contains(pinned_fragment),
            "the shared manual build must not pin cuVS through '{pinned_fragment}'"
        );
    }
    assert!(
        manual_script.contains("if [[ \"${cuvs_install_dir}\" == \"${default_cuvs_install_dir}\" ]]; then"),
        "the manual build must validate an explicit or symlinked default cuVS prefix too"
    );
}

#[test]
fn manual_build_enforces_unpinned_source_marker_policy() {
    let missing = ManualBuildFixture::new(None);
    let output = missing.run(None);
    assert!(!output.status.success());
    assert_stderr_contains(&output, "missing 'share/cuvs-nexus/source-revision'");

    let marked = ManualBuildFixture::new(Some("active-checkout-revision"));
    let output = marked.run(None);
    assert!(!output.status.success());
    assert_stderr_contains(&output, "does not export cuVS's CMake package");

    let alias = marked.root.join("default-cuvs-alias");
    symlink(marked.default_install(), &alias).expect("create default prefix alias");
    let output = marked.run(Some(&alias));
    assert!(!output.status.success());
    assert_stderr_contains(&output, "does not export cuVS's CMake package");

    let external = missing.root.join("external-cuvs");
    fs::create_dir_all(&external).expect("create external cuVS prefix");
    let output = missing.run(Some(&external));
    assert!(!output.status.success());
    assert_stderr_contains(&output, "does not export cuVS's CMake package");
}

#[test]
fn build_rs_tracks_cuvs_selectors_and_native_identity_inputs() {
    let build_rs = read_build_rs();

    for tracked in [
        "cuvs.library_selector_path",
        "cuvs.library_path",
        "validated_loader.cugraph_library_selector_path",
        "validated_loader.cugraph_library_path",
    ] {
        assert!(
            build_rs.contains(&format!("cargo:rerun-if-changed={{}}\", {tracked}.display()")),
            "build.rs must track {tracked} so native identity metadata cannot become stale"
        );
    }

    assert!(
        build_rs.contains("for path in &cuvs.source_revision_inputs") && build_rs.contains("cargo:rerun-if-changed={}"),
        "build.rs must rerun when the root source marker or Git index changes"
    );
}

#[test]
fn loader_validation_uses_controlled_and_ambient_environments() {
    let build_support = read_build_support_rs();

    assert!(build_support.contains("controlled_loader_search_path("));
    assert!(build_support.contains(".env(\"LD_LIBRARY_PATH\", &loader_search_path)"));
    assert!(build_support.contains("inspect ambient libcugraph loader resolution"));
    assert!(build_support.contains("validate_ambient_cuvs_resolution"));
}

#[test]
fn upstream_cugraph_consumer_contract_links_the_shared_cuvs_cpp_target() {
    let get_cuvs = read_upstream_cugraph_file("cmake/thirdparty/get_cuvs.cmake");
    let cmake_lists = read_upstream_cugraph_file("CMakeLists.txt");

    assert!(get_cuvs.contains("GLOBAL_TARGETS      cuvs::cuvs"));
    assert!(get_cuvs.contains("rapids_cpm_find(cuvs ${PKG_VERSION}"));
    assert!(cmake_lists.contains("cuvs::cuvs"));
    assert!(cmake_lists.contains("$<IF:$<BOOL:${CUGRAPH_USE_CUVS_STATIC}>,cuvs::cuvs_static,cuvs::cuvs>"));
}

#[test]
fn upstream_cugraph_requires_installed_cudf_rust_error_boundary_header() {
    let cmake_lists = read_upstream_cugraph_file("CMakeLists.txt");

    for fragment in [
        "find_path(CUGRAPH_CUDF_RUST_INCLUDE_DIR",
        "cudf_rust/error_boundary.hpp",
        "HINTS \"${rmm_ROOT}\"",
        "IS_SYMLINK",
        "CUGRAPH_CUDF_RUST_INCLUDE_DIR",
    ] {
        assert!(cmake_lists.contains(fragment), "cuGraph CMake contract is missing '{fragment}'");
    }
}

#[test]
fn downstream_build_rs_consumes_runtime_library_metadata_without_native_guessing() {
    let build_rs = read_downstream_build_rs();

    assert!(
        build_rs.contains("DEP_CUGRAPH_RUNTIME_LIB_DIRS"),
        "cugraph-nexus/build.rs should consume the multi-path runtime metadata from cugraph-sys"
    );
    assert!(
        build_rs.contains("env::split_paths"),
        "cugraph-nexus/build.rs should parse DEP_CUGRAPH_RUNTIME_LIB_DIRS as an OS-native path list"
    );
    assert!(
        build_rs.contains("DEP_CUGRAPH_NATIVE_INCLUDE_DIRS"),
        "cugraph-nexus/build.rs should consume the native include metadata exported by cugraph-sys"
    );
    assert!(
        build_rs.contains("DEP_CUGRAPH_NATIVE_LINK_DIRS"),
        "cugraph-nexus/build.rs should consume the native link metadata exported by cugraph-sys"
    );
    assert!(
        !build_rs.contains("DEP_CUGRAPH_METADATA"),
        "cugraph-nexus/build.rs should consume the formal metadata keys directly instead of reparsing ad hoc metadata blobs"
    );
    for forbidden in ["CONDA_PREFIX", "LIBRARY_PREFIX", "PREFIX"] {
        assert!(
            !build_rs.contains(forbidden),
            "cugraph-nexus/build.rs should not compensate for missing native metadata via {forbidden}"
        );
    }
}

#[test]
fn build_rs_uses_cmake_export_fallback_instead_of_pkg_config_or_deps_scanning() {
    let build_rs = read_build_rs();
    let build_support = read_build_support_rs();

    assert!(
        build_rs.contains("CUGRAPH_CMAKE_BUILD_DIR"),
        "build.rs should support an explicit CMake export root for native metadata resolution"
    );
    assert!(
        build_support.contains("target/native/cugraph-build"),
        "build_support.rs should support the workspace-local CMake export root for repo-local builds"
    );
    assert!(
        build_support.contains("cmake_build_dir.join(\"CMakeCache.txt\")"),
        "build_support.rs should track CMakeCache.txt because package roots and CUDA architectures are cache inputs"
    );
    assert!(
        build_rs.contains("prepare_workspace_cmake_export_dir"),
        "build.rs should prepare the workspace-local cuGraph CMake export automatically before falling back to docs"
    );
    assert!(
        build_support.contains("Command::new(\"cmake\")") && build_support.contains("build_workspace_cugraph_cpp"),
        "build_support.rs should make the workspace-local native build a first-class cargo build path"
    );
    assert!(!build_support.contains("pkg-config"), "build_support.rs should not fall back to pkg-config discovery");
    assert!(!build_rs.contains("_deps"), "build.rs should not scan private dependency layout details directly");
}

#[test]
fn cmake_probe_pins_makefiles_generator_for_link_txt_contract() {
    let build_support = read_build_support_rs();

    assert!(
        build_support.contains("\"-G\"") && build_support.contains("\"Unix Makefiles\""),
        "the CMake metadata probe should pin a Makefiles generator because link.txt is part of that generator contract"
    );
}

#[test]
fn include_sensitive_tests_reuse_build_support_include_resolution() {
    let source = read_test_source("tests/support/mod.rs");
    assert!(
        source.contains("resolve_native_build_contract"),
        "tests/support/mod.rs should reuse build_support::resolve_native_build_contract so the test include path follows the formal native contract"
    );
}

#[test]
fn cargo_toml_declares_links_metadata_for_native_provider() {
    let cargo_toml = read_cargo_toml();

    assert!(
        cargo_toml.contains("links = \"cugraph\""),
        "cugraph-nexus-sys/Cargo.toml should declare links = \"cugraph\" so Cargo can coordinate the native library provider"
    );
}
