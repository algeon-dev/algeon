#[path = "../build_support.rs"]
mod build_support;

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use build_support::{
    CompilerFamily, external_warning_control_args, prioritize_rapids_cccl_include_dirs, resolve_shim_compile_layout,
    system_include_args,
};

fn read_build_rs() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(manifest_dir.join("build.rs")).expect("failed to read build.rs")
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let unique = format!(
            "cugraph-rs-build-test-{}-{}",
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

#[test]
fn local_shim_dir_stays_project_owned_include_root() {
    let manifest_dir = Path::new("/tmp/cugraph-rs");
    let native_include_dirs = vec![PathBuf::from("/opt/cugraph/include")];

    let layout = resolve_shim_compile_layout(manifest_dir, native_include_dirs).expect("compile layout");

    assert_eq!(layout.source_file, manifest_dir.join("shim").join("runtime_resource_handle.cpp"));
    assert_eq!(layout.local_include_dir, manifest_dir.join("shim"));
    assert_eq!(layout.native_include_dirs, vec![PathBuf::from("/opt/cugraph/include")]);
}

#[test]
fn rapids_cccl_include_roots_precede_cuda_toolkit_include_roots() {
    let root = TestDir::new();
    let cugraph_include = root.path().join("cugraph").join("include");
    let cuda_include = root.path().join("cuda-13.2").join("targets").join("x86_64-linux").join("include");
    let cuda_cccl_include = cuda_include.join("cccl");
    let rmm_include = root.path().join("rmm").join("include");
    let rapids_libcudacxx = root.path().join("rapids-cccl").join("libcudacxx").join("include");
    let rapids_cub = root.path().join("rapids-cccl").join("cub");
    let rapids_thrust = root.path().join("rapids-cccl").join("thrust");

    for dir in [
        &cugraph_include,
        &cuda_include,
        &cuda_cccl_include,
        &rmm_include,
        &rapids_libcudacxx.join("cuda").join("std"),
        &rapids_cub.join("cub"),
        &rapids_thrust.join("thrust"),
    ] {
        fs::create_dir_all(dir).expect("create include dir");
    }
    fs::write(rapids_libcudacxx.join("cuda").join("std").join("version"), []).expect("version");
    fs::write(rapids_cub.join("cub").join("version.cuh"), []).expect("cub version");
    fs::write(rapids_thrust.join("thrust").join("version.h"), []).expect("thrust version");

    let reordered = prioritize_rapids_cccl_include_dirs(vec![
        cugraph_include.clone(),
        rmm_include.clone(),
        cuda_include.clone(),
        cuda_cccl_include.clone(),
        rapids_thrust.clone(),
        rapids_libcudacxx.clone(),
        rapids_cub.clone(),
    ]);

    assert_eq!(
        reordered,
        vec![
            rapids_thrust,
            rapids_libcudacxx,
            rapids_cub,
            cugraph_include,
            rmm_include,
            cuda_include,
            cuda_cccl_include
        ]
    );
}

#[test]
fn build_rs_tracks_project_owned_shim_sources() {
    let build_rs = read_build_rs();

    assert!(build_rs.contains(r#"join("shim").join("runtime_resource_handle.cpp")"#));
    assert!(build_rs.contains(r#"join("shim").join("runtime_resource_handle.h")"#));
    assert!(
        !build_rs.contains(r#"join("cpp").join("runtime_resource_handle"#),
        "runtime resource shim sources should stay under cugraph-rs/shim, not cugraph-rs/cpp"
    );
}

#[test]
fn compile_layout_rejects_missing_native_include_dirs() {
    let manifest_dir = Path::new("/tmp/cugraph-rs");

    let err = resolve_shim_compile_layout(manifest_dir, Vec::new()).expect_err("missing native include dirs");

    assert!(err.contains("DEP_CUGRAPH_NATIVE_INCLUDE_DIRS"));
}

#[test]
fn gnu_like_compilers_use_isystem_for_native_include_roots() {
    let args = system_include_args(CompilerFamily::GnuLike, Path::new("/opt/cugraph/include"));
    let rendered = args.iter().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>();

    assert_eq!(rendered, vec!["-isystem".to_string(), "/opt/cugraph/include".to_string()]);
}

#[test]
fn msvc_compilers_use_external_include_mode_for_native_include_roots() {
    let args = system_include_args(CompilerFamily::Msvc, Path::new(r"C:\rapids\include"));
    let rendered = args.iter().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>();

    assert_eq!(rendered, vec![r"/external:IC:\rapids\include".to_string()]);
}

#[test]
fn msvc_external_warning_control_is_global_not_per_include_dir() {
    let args = external_warning_control_args(CompilerFamily::Msvc);
    let rendered = args.iter().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>();

    assert_eq!(rendered, vec!["/external:W0".to_string()]);
}

#[test]
fn gnu_like_compilers_do_not_need_global_external_warning_control() {
    assert!(external_warning_control_args(CompilerFamily::GnuLike).is_empty());
}
