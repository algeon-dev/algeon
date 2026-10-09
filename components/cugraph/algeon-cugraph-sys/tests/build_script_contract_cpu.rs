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
            let marker_path = default_install.join("share/algeon-cuvs/source-revision");
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
        assert!(script.contains("algeon_cuda_architectures"), "{component} must use the canonical selector");
        assert!(script.contains("algeon_cuda_write_contract"), "{component} must record its real cubin set");
    }

    let output = Command::new("bash")
        .args([
            "-c",
            "source \"$1\"; CUDF_CMAKE_CUDA_ARCHITECTURES=89 algeon_cuda_architectures",
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
            "source \"$1\"; CMAKE_CUDA_ARCHITECTURES=native algeon_cuda_architectures",
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
        let marker = prefix.join("share/algeon-native/cuda-architectures.env");
        fs::create_dir_all(marker.parent().expect("marker parent")).expect("create marker directory");
        fs::write(marker, format!("requested=NATIVE\nreal={real}\n")).expect("write architecture marker");
    }
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/build/cuda_architecture_contract.sh");
    let output = Command::new("bash")
        .args([
            "-c",
            "source \"$1\"; algeon_cuda_require_matching_contracts cuDF \"$2\" cuVS \"$3\" \"rebuild stack\"",
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
        .args(["-c", "source \"$1\"; algeon_cuda_real_architectures \"$2\"", "architecture-contract-test"])
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

fn read_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("build.rs")).expect("failed to read build.rs")
}

fn read_downstream_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("../algeon-cugraph/build.rs")).expect("failed to read downstream build.rs")
}

fn read_manual_build_script() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("../../../scripts/build/build_libcugraph.sh"))
        .expect("failed to read scripts/build/build_libcugraph.sh")
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
fn manual_build_enforces_unpinned_source_marker_policy() {
    let missing = ManualBuildFixture::new(None);
    let output = missing.run(None);
    assert!(!output.status.success());
    assert_stderr_contains(&output, "missing 'share/algeon-cuvs/source-revision'");

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
fn downstream_build_rs_consumes_runtime_library_metadata_without_native_guessing() {
    let build_rs = read_downstream_build_rs();

    assert!(
        build_rs.contains("DEP_CUGRAPH_RUNTIME_LIB_DIRS"),
        "cugraph/build.rs should consume the multi-path runtime metadata from cugraph-sys"
    );
    assert!(
        build_rs.contains("env::split_paths"),
        "cugraph/build.rs should parse DEP_CUGRAPH_RUNTIME_LIB_DIRS as an OS-native path list"
    );
    assert!(
        build_rs.contains("DEP_CUGRAPH_NATIVE_INCLUDE_DIRS"),
        "cugraph/build.rs should consume the native include metadata exported by cugraph-sys"
    );
    assert!(
        build_rs.contains("DEP_CUGRAPH_NATIVE_LINK_DIRS"),
        "cugraph/build.rs should consume the native link metadata exported by cugraph-sys"
    );
    assert!(
        !build_rs.contains("DEP_CUGRAPH_METADATA"),
        "cugraph/build.rs should consume the formal metadata keys directly instead of reparsing ad hoc metadata blobs"
    );
    for forbidden in ["CONDA_PREFIX", "LIBRARY_PREFIX", "PREFIX"] {
        assert!(
            !build_rs.contains(forbidden),
            "cugraph/build.rs should not compensate for missing native metadata via {forbidden}"
        );
    }
}
