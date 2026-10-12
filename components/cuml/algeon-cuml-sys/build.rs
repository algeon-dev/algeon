mod build_support;

use std::{
    env,
    path::{Path, PathBuf},
};

use rapids_build_support::{NativeComponent, NativeLinking, resolve_companion_prefix, verify_prebuilt_crate_version};

const NATIVE_COMPONENT: NativeComponent = NativeComponent {
    crate_name: "algeon-cuml-sys",
    install_dir_name: "cuml-install",
    required_library: "lib/libcuml_rust.so",
    override_env: Some("CUML_INSTALL_DIR"),
    // A source shim is needed for test hooks; normal builds replay the
    // canonical install's link manifest and skip recompiling native code.
    source_required_env: Some("CARGO_FEATURE_TEST_HOOKS"),
    source_tree_sentinel: Some("../cuml/cmake/rapids_config.cmake"),
    build_script: "scripts/build/build_libcuml.sh",
};

const CUDF_COMPONENT: NativeComponent = NativeComponent {
    crate_name: "algeon-cudf-sys",
    install_dir_name: "cudf-install",
    required_library: "lib/librmm.so",
    override_env: Some("CUDF_INSTALL_DIR"),
    source_required_env: None,
    source_tree_sentinel: None,
    build_script: "scripts/build/build_libcudf.sh",
};

const CUVS_COMPONENT: NativeComponent = NativeComponent {
    crate_name: "algeon-cuvs-sys",
    install_dir_name: "cuvs-install",
    required_library: "lib/libcuvs.so",
    override_env: Some("CUVS_INSTALL_DIR"),
    source_required_env: None,
    source_tree_sentinel: None,
    build_script: "scripts/build/build_libcuvs.sh",
};

fn main() {
    for name in [
        "DOCS_RS",
        "ALGEON_NATIVE_DIR",
        "CUML_INSTALL_DIR",
        "CUDF_INSTALL_DIR",
        "CUVS_INSTALL_DIR",
        "CUDAToolkit_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-changed=native");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    match rapids_build_support::resolve_native_linking(&NATIVE_COMPONENT, &manifest, &|key| env::var_os(key)) {
        NativeLinking::Skip => (),
        NativeLinking::Missing(error) => panic!("{error}"),
        NativeLinking::Prebuilt(prefix) => {
            let marker = "share/algeon-cuml/crate-version";
            println!("cargo:rerun-if-changed={}", prefix.join(marker).display());
            verify_prebuilt_crate_version(&NATIVE_COMPONENT, &prefix, marker, env!("CARGO_PKG_VERSION"))
                .unwrap_or_else(|error| panic!("{error}"));
            use_prefix(&manifest, &prefix, &prefix);
        },
        NativeLinking::SourceTree => build_from_source(&manifest),
    }
}

fn companion_prefix(component: &NativeComponent, manifest: &Path) -> PathBuf {
    resolve_companion_prefix(component, manifest, &|name| env::var_os(name)).unwrap_or_else(|error| panic!("{error}"))
}

fn canonical_library(prefix: &Path, relative: &str) -> PathBuf {
    let library = prefix.join(relative);
    println!("cargo:rerun-if-changed={}", library.display());
    library.canonicalize().unwrap_or_else(|error| {
        panic!(
            "algeon-cuml-sys: cannot use native library {} under prefix {}: {error}. Check the selected install prefix and rebuild its component from Algeon v{}. See https://algeon.dev/build/from-source",
            library.display(),
            prefix.display(),
            env!("CARGO_PKG_VERSION")
        )
    })
}

fn use_prefix(manifest: &Path, prefix: &Path, cuml_install: &Path) {
    let cudf = companion_prefix(&CUDF_COMPONENT, manifest);
    let cuvs = companion_prefix(&CUVS_COMPONENT, manifest);
    let cuvs_library = canonical_library(&cuvs, CUVS_COMPONENT.required_library);
    let _rmm = canonical_library(&cudf, CUDF_COMPONENT.required_library);
    let shim = canonical_library(prefix, "lib/libcuml_rust.so");
    let cuml = canonical_library(cuml_install, "lib/libcuml.so");
    let treelite = canonical_library(cuml_install, "lib/libtreelite.so");
    let nvforest = canonical_library(cuml_install, "lib/libnvforest.so");
    let link_file = prefix.join("share/cuml_rust/cargo-link.txt");
    let directives = build_support::read_cargo_link_manifest(prefix).unwrap_or_else(|error| panic!("{error}"));
    for directive in directives {
        println!("cargo:{directive}");
    }
    println!("cargo:rerun-if-changed={}", link_file.display());
    println!("cargo:cuml-rust-runtime-lib={}", shim.display());
    println!("cargo::metadata=cuml-rust-runtime-lib={}", shim.display());
    println!("cargo:metadata=cuml-rust-root={}", prefix.display());
    let cuda = ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"]
        .into_iter()
        .find_map(env::var_os)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    // Test binaries need the shim and its canonical RAPIDS dependencies at
    // runtime; rlib link directives alone do not carry a loader search path.
    let dirs = [prefix.join("lib"), cuml_install.join("lib"), cudf.join("lib"), cuvs.join("lib"), cuda.join("lib64")];
    let encoded = dirs
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    println!("cargo:runtime-lib-dirs={encoded}");
    println!("cargo::metadata=runtime-lib-dirs={encoded}");
    println!("cargo:libcuml-runtime-lib={}", cuml.display());
    println!("cargo::metadata=libcuml-runtime-lib={}", cuml.display());
    println!("cargo:treelite-runtime-lib={}", treelite.display());
    println!("cargo::metadata=treelite-runtime-lib={}", treelite.display());
    println!("cargo:nvforest-runtime-lib={}", nvforest.display());
    println!("cargo::metadata=nvforest-runtime-lib={}", nvforest.display());
    println!("cargo:cuvs-library-path={}", cuvs_library.display());
    println!("cargo::metadata=cuvs-library-path={}", cuvs_library.display());
    println!(
        "cargo::metadata=metadata=runtime-lib-dirs={encoded};cuml-rust-runtime-lib={};libcuml-runtime-lib={};treelite-runtime-lib={};nvforest-runtime-lib={};cuvs-library-path={}",
        shim.display(),
        cuml.display(),
        treelite.display(),
        nvforest.display(),
        cuvs_library.display()
    );
    println!("cargo:metadata=cuda-runtime={}", cuda.display());
}

fn build_from_source(manifest: &Path) {
    let install =
        companion_prefix(&NativeComponent { required_library: "lib/libcuml.so", ..NATIVE_COMPONENT }, manifest);
    let cudf = companion_prefix(&CUDF_COMPONENT, manifest);
    let cuvs = companion_prefix(&CUVS_COMPONENT, manifest);
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
    use_prefix(manifest, &shim_install, &install);
}
