use std::{fs, path::PathBuf};

#[test]
fn native_shim_targets_pinned_cuml_and_c_abi_contract() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cmake = fs::read_to_string(root.join("native/CMakeLists.txt")).unwrap();
    let build_script = fs::read_to_string(root.join("../../../scripts/build/build_libcuml.sh")).unwrap();
    let shim = fs::read_to_string(root.join("native/src/shim_common.hpp")).unwrap();
    let rust = fs::read_to_string(root.join("build.rs")).unwrap();
    let version_script = fs::read_to_string(root.join("native/cmake/cuml-rust.map")).unwrap();
    assert!(cmake.contains("cuml::cuml"));
    assert!(cmake.contains("share/algeon-cuml/openmp-disabled"));
    assert!(build_script.contains("-DDISABLE_OPENMP=ON"));
    assert!(build_script.contains("-DCMAKE_CUDA_FLAGS=-Xcompiler=-Wno-unknown-pragmas"));
    assert!(build_script.contains("nm -D --undefined-only"));
    assert!(build_script.contains("shim_build_dir=\"${repo_root}/target/native/cuml-rust-build\""));
    assert!(build_script.contains("-DCUML_INSTALL_DIR=${install_dir}"));
    assert!(build_script.contains("-DCUDF_INSTALL_DIR=${cudf_install_dir}"));
    assert!(build_script.contains("-DCUVS_INSTALL_DIR=${cuvs_install_dir}"));
    assert!(build_script.contains("-DCUML_RUST_ENABLE_TEST_HOOKS=OFF"));
    assert!(build_script.contains("cmake --install \"${shim_build_dir}\" --prefix \"${install_dir}\""));
    assert!(build_script.contains("cuml_test_"));
    assert!(build_script.contains("share/cuml_rust/cargo-link.txt"));
    assert!(build_script.contains("check_resolved_dependency libcuml.so"));
    assert!(build_script.contains("check_resolved_dependency libcudf_rust_rmm_interop.so"));
    assert!(build_script.contains("check_resolved_dependency libnvforest.so"));
    assert!(build_script.contains("check_resolved_dependency libtreelite.so"));
    assert!(build_script.contains("CMAKE_PROJECT_treelite_INCLUDE"));
    assert!(build_script.contains("-DCPM_DOWNLOAD_Treelite=ON"));
    // Header-instantiated shim code must share RMM's per-device resource map.
    assert!(version_script.contains("_ZN3rmm*; _ZZN3rmm*; _ZGVZN3rmm*;"));
    assert!(build_script.contains("does not export RMM's per-device resource map"));
    assert!(build_script.contains("CMAKE_PROJECT_NVForest_INCLUDE"));
    assert!(build_script.contains("nodes_{std::move(nodes)}"));
    assert!(cmake.contains("--no-undefined"));
    assert!(cmake.contains("src/test_support.cpp"));
    for source in [
        "src/pca.cpp",
        "src/tsvd.cpp",
        "src/glm.cpp",
        "src/hdbscan.cpp",
        "src/random_forest.cpp",
        "src/umap.cpp",
        "src/forest_inference.cpp",
    ] {
        assert!(cmake.contains(source), "missing native algorithm source {source}");
    }
    assert!(shim.contains("cuml_translate_exception"));
    assert!(rust.contains("libcuml_rust.so"));
    assert!(rust.contains("CARGO_FEATURE_TEST_HOOKS"));
}
