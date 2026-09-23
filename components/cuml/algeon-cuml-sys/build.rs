mod build_support;

use std::{env, path::PathBuf};

const NATIVE_COMPONENT: rapids_build_support::NativeComponent = rapids_build_support::NativeComponent {
    crate_name: "algeon-cuml-sys",
    install_dir_name: "cuml-install",
    required_library: "lib/libcuml_rust.so",
    override_env: Some("CUML_INSTALL_DIR"),
    // A source shim is needed for test hooks; normal builds replay the
    // canonical install's link manifest instead of recompiling native code.
    source_required_env: Some("CARGO_FEATURE_TEST_HOOKS"),
    source_tree_sentinel: Some("../cuml/cmake/rapids_config.cmake"),
    build_script: "scripts/build/build_libcuml.sh",
};

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=ALGEON_NATIVE_DIR");
    println!("cargo:rerun-if-env-changed=CUML_INSTALL_DIR");
    println!("cargo:rerun-if-changed=native");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    match rapids_build_support::resolve_native_linking(&NATIVE_COMPONENT, &manifest, &|key| env::var_os(key)) {
        rapids_build_support::NativeLinking::Skip => (),
        rapids_build_support::NativeLinking::Missing(error) => panic!("{error}"),
        rapids_build_support::NativeLinking::Prebuilt(prefix) => use_prefix(&prefix, &prefix),
        rapids_build_support::NativeLinking::SourceTree => build_from_source(&manifest),
    }
}

fn use_prefix(prefix: &std::path::Path, cuml_install: &std::path::Path) {
    let link_file = prefix.join("share/cuml_rust/cargo-link.txt");
    let directives = build_support::read_cargo_link_manifest(prefix).unwrap_or_else(|error| panic!("{error}"));
    for directive in directives {
        println!("cargo:{directive}");
    }
    println!("cargo:rerun-if-changed={}", link_file.display());
    println!("cargo:cuml-rust-runtime-lib={}", prefix.join("lib/libcuml_rust.so").display());
    println!("cargo:metadata=cuml-rust-root={}", prefix.display());
    let cudf = env::var_os("CUDF_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/native/cudf-install"));
    let cuvs = env::var_os("CUVS_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/native/cuvs-install"));
    let cuda = env::var_os("CUDA_HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    // Test binaries need the shim and its canonical RAPIDS dependencies at
    // runtime; rlib link directives alone do not carry a loader search path.
    let dirs = [prefix.join("lib"), cuml_install.join("lib"), cudf.join("lib"), cuvs.join("lib"), cuda.join("lib64")];
    let encoded = dirs
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    println!("cargo::metadata=runtime-lib-dirs={encoded}");
    println!("cargo:metadata=cuda-runtime={}", env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into()));
}

fn build_from_source(manifest: &std::path::Path) {
    let install = env::var_os("CUML_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../target/native/cuml-install"));
    let cudf = env::var_os("CUDF_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../target/native/cudf-install"));
    let cuvs = env::var_os("CUVS_INSTALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../target/native/cuvs-install"));
    let shim_install = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("shim-install");
    let mut config = cmake::Config::new(manifest.join("native"));
    config.define("CMAKE_INSTALL_PREFIX", &shim_install);
    config.define("CUML_INSTALL_DIR", &install);
    config.define("CUDF_INSTALL_DIR", &cudf);
    config.define("CUVS_INSTALL_DIR", &cuvs);
    config.define(
        "CUML_RUST_ENABLE_TEST_HOOKS",
        if env::var_os("CARGO_FEATURE_TEST_HOOKS").is_some() { "ON" } else { "OFF" },
    );
    config.build();
    use_prefix(&shim_install, &install);
}
