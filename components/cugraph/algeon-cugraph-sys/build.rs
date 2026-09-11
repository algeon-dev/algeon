use std::{env, path::PathBuf};

#[path = "build_support.rs"]
mod build_support;

const NATIVE_COMPONENT: rapids_build_support::NativeComponent = rapids_build_support::NativeComponent {
    crate_name: "algeon-cugraph-sys",
    install_dir_name: "cugraph-install",
    required_library: "lib/libcugraph.so",
    override_env: None,
    source_required_env: None,
    source_tree_sentinel: Some("../cugraph/cpp/CMakeLists.txt"),
    build_script: "scripts/build/build_libcugraph.sh",
};

fn emit_runtime_search_path(path: &std::path::Path) {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}

fn emit_runtime_metadata(paths: &[PathBuf]) {
    let joined = env::join_paths(paths).expect("runtime library search paths must be valid OS strings");
    println!("cargo:runtime-lib-dirs={}", joined.to_string_lossy());
    println!("cargo:metadata=runtime-lib-dirs={}", joined.to_string_lossy());
}

fn emit_native_include_metadata(paths: &[PathBuf]) {
    let joined = env::join_paths(paths).expect("native include search paths must be valid OS strings");
    println!("cargo:native-include-dirs={}", joined.to_string_lossy());
    println!("cargo:metadata=native-include-dirs={}", joined.to_string_lossy());
}

fn emit_native_link_metadata(paths: &[PathBuf]) {
    let joined = env::join_paths(paths).expect("native link search paths must be valid OS strings");
    println!("cargo:native-link-dirs={}", joined.to_string_lossy());
    println!("cargo:metadata=native-link-dirs={}", joined.to_string_lossy());
}

fn emit_cuvs_metadata(cuvs: &build_support::CanonicalCuvsContract) {
    println!("cargo::metadata=cuvs-install-dir={}", cuvs.install_dir.display());
    println!("cargo::metadata=cuvs-library-path={}", cuvs.library_path.display());
    println!("cargo::metadata=cuvs-runtime-lib-dir={}", cuvs.runtime_lib_dir.display());
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../../..");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=ALGEON_NATIVE_DIR");
    let linking =
        rapids_build_support::resolve_native_linking(&NATIVE_COMPONENT, &manifest_dir, &|name| env::var_os(name));
    if matches!(linking, rapids_build_support::NativeLinking::Skip) {
        return;
    }
    let explicit_contract_count = build_support::explicit_native_contract_count(
        env::var_os("CUGRAPH_NATIVE_INCLUDE_DIRS").as_ref(),
        env::var_os("CUGRAPH_NATIVE_LINK_DIRS").as_ref(),
        env::var_os("CUGRAPH_RUNTIME_LIB_DIRS").as_ref(),
    );
    let prebuilt_prefix = if explicit_contract_count == 0 {
        match linking {
            rapids_build_support::NativeLinking::Prebuilt(prefix) => {
                verify_prebuilt_version(&prefix);
                Some(prefix)
            },
            rapids_build_support::NativeLinking::SourceTree => None,
            rapids_build_support::NativeLinking::Missing(missing) => panic!("{missing}"),
            rapids_build_support::NativeLinking::Skip => unreachable!(),
        }
    } else {
        None
    };
    if let Some(prefix) = &prebuilt_prefix {
        let include = prefix.join("include");
        let lib = prefix.join("lib");
        // SAFETY: Cargo build scripts are single-threaded, and native contract
        // resolution reads these selectors only after this assignment.
        unsafe {
            env::set_var("CUGRAPH_NATIVE_INCLUDE_DIRS", &include);
            env::set_var("CUGRAPH_NATIVE_LINK_DIRS", &lib);
            env::set_var("CUGRAPH_RUNTIME_LIB_DIRS", &lib);
        }
        println!("cargo:rerun-if-changed={}", prefix.join(NATIVE_COMPONENT.required_library).display());
        println!("cargo:rerun-if-changed={}", prefix.join("share/algeon-cugraph/crate-version").display());
    }
    let explicit_contract_count = if prebuilt_prefix.is_some() { 3 } else { explicit_contract_count };
    let cudf_install_dir = env::var_os("CUDF_INSTALL_DIR")
        .map(PathBuf::from)
        .or_else(|| env::var_os("ALGEON_NATIVE_DIR").map(|root| PathBuf::from(root).join("cudf-install")))
        .unwrap_or_else(|| repo_root.join("target/native/cudf-install"));
    let cmake_build_dir_env = if prebuilt_prefix.is_some() { None } else { env::var_os("CUGRAPH_CMAKE_BUILD_DIR") };
    let effective_cmake_build_dir_env = if explicit_contract_count == 0 && cmake_build_dir_env.is_none() {
        Some(
            build_support::prepare_workspace_cmake_export_dir(
                &manifest_dir,
                env::var_os("CUGRAPH_CPP_BUILD_DIR"),
                env::var_os("CUDF_INSTALL_DIR"),
                env::var_os("CUVS_INSTALL_DIR"),
                env::var_os("CUGRAPH_SYS_AUTO_BUILD"),
                env::var_os("CMAKE_CUDA_ARCHITECTURES"),
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .into_os_string(),
        )
    } else {
        cmake_build_dir_env.clone()
    };

    let mut contract = build_support::resolve_native_build_contract(
        &manifest_dir,
        env::var_os("CUGRAPH_NATIVE_INCLUDE_DIRS"),
        env::var_os("CUGRAPH_NATIVE_LINK_DIRS"),
        env::var_os("CUGRAPH_RUNTIME_LIB_DIRS"),
        effective_cmake_build_dir_env.clone(),
    )
    .unwrap_or_else(|err| panic!("{err}"));
    if prebuilt_prefix.is_some() {
        let cudf_runtime_dir = [cudf_install_dir.join("lib"), cudf_install_dir.join("lib64")]
            .into_iter()
            .find(|dir| dir.join("librmm.so").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "ALGEON_NATIVE_DIR selected cuGraph, but the matching cuDF prefix {} does not contain lib/librmm.so or lib64/librmm.so",
                    cudf_install_dir.display()
                )
            });
        if !contract.runtime_lib_dirs.iter().any(|path| path == &cudf_runtime_dir) {
            contract.runtime_lib_dirs.push(cudf_runtime_dir);
        }
        // The downstream cuGraph shim includes RMM headers that
        // `cugraph-install/include` does not vendor; they ship with the cuDF
        // prefix, so the prebuilt contract must carry its include dir just
        // like it carries the cuDF runtime dir above.
        let cudf_include_dir = cudf_install_dir.join("include");
        if !cudf_include_dir.join("rmm/aligned.hpp").is_file() {
            panic!(
                "ALGEON_NATIVE_DIR selected cuGraph, but the matching cuDF prefix {} does not contain include/rmm/aligned.hpp",
                cudf_install_dir.display()
            );
        }
        // RMM itself requires CCCL 3.3+, which the CUDA toolkit include dirs
        // below cannot satisfy; the cuDF prefix vendors the matching CCCL
        // under `include/rapids`, and the downstream shim build prioritizes
        // that layout ahead of the toolkit.
        let cudf_rapids_include_dir = cudf_include_dir.join("rapids");
        if !cudf_rapids_include_dir.join("cuda/std/version").is_file() {
            panic!(
                "ALGEON_NATIVE_DIR selected cuGraph, but the matching cuDF prefix {} does not contain include/rapids/cuda/std/version",
                cudf_install_dir.display()
            );
        }
        for include_dir in [cudf_include_dir, cudf_rapids_include_dir] {
            if !contract.native_include_dirs.iter().any(|path| path == &include_dir) {
                contract.native_include_dirs.push(include_dir);
            }
        }
    }
    let cuvs_install_dir = env::var_os("CUVS_INSTALL_DIR").or_else(|| {
        env::var_os("ALGEON_NATIVE_DIR").map(|root| PathBuf::from(root).join("cuvs-install").into_os_string())
    });
    let cuvs = build_support::resolve_canonical_cuvs_contract(&manifest_dir, cuvs_install_dir)
        .unwrap_or_else(|err| panic!("{err}"));
    if explicit_contract_count == 0 {
        let cmake_export_dir = PathBuf::from(
            effective_cmake_build_dir_env
                .as_deref()
                .expect("CMake contract resolution must retain its selected export directory"),
        );
        let components = build_support::cuda_architecture_components(
            &repo_root,
            &cudf_install_dir,
            &cuvs.install_dir,
            &cmake_export_dir,
        );
        let requested = build_support::resolved_cuda_architectures(env::var_os("CMAKE_CUDA_ARCHITECTURES").as_deref());
        for path in build_support::validate_cuda_architecture_contracts(&components, &requested.to_string_lossy())
            .unwrap_or_else(|err| panic!("{err}"))
        {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    let validated_loader =
        build_support::validate_cugraph_cuvs_loader(&contract.native_link_dirs, &contract.runtime_lib_dirs, &cuvs)
            .unwrap_or_else(|err| panic!("{err}"));
    build_support::check_libcugraph_staleness(&manifest_dir, &contract.native_link_dirs);
    if !contract.runtime_lib_dirs.iter().any(|path| path == &cuvs.runtime_lib_dir) {
        contract.runtime_lib_dirs.push(cuvs.runtime_lib_dir.clone());
    }
    if explicit_contract_count == 0 {
        let cmake_export_dir = PathBuf::from(
            effective_cmake_build_dir_env
                .as_deref()
                .expect("CMake contract resolution must retain its selected export directory"),
        );
        for path in build_support::tracked_cmake_export_files(&cmake_export_dir) {
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    for library_dir in &contract.native_link_dirs {
        println!("cargo:rustc-link-search=native={}", library_dir.display());
    }
    for runtime_dir in &contract.runtime_lib_dirs {
        emit_runtime_search_path(runtime_dir);
    }
    println!("cargo:rustc-link-lib=dylib=cugraph");
    println!("cargo:rustc-link-lib=dylib=cugraph_common");
    println!("cargo:rustc-link-lib=dylib=cugraph_c");
    emit_runtime_metadata(&contract.runtime_lib_dirs);
    emit_native_include_metadata(&contract.native_include_dirs);
    emit_native_link_metadata(&contract.native_link_dirs);
    emit_cuvs_metadata(&cuvs);

    println!("cargo:rerun-if-env-changed=CUGRAPH_NATIVE_INCLUDE_DIRS");
    println!("cargo:rerun-if-env-changed=CUGRAPH_NATIVE_LINK_DIRS");
    println!("cargo:rerun-if-env-changed=CUGRAPH_RUNTIME_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=CUGRAPH_CMAKE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_CPP_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_SYS_AUTO_BUILD");
    println!("cargo:rerun-if-env-changed=CUDF_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUVS_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_RS_SKIP_STALENESS_CHECK");
    println!("cargo:rerun-if-env-changed=CUGRAPH_RS_ENFORCE_STALENESS_CHECK");
    println!("cargo:rerun-if-env-changed=CUGRAPH_ENFORCE_STALENESS_CHECK");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed={}", manifest_dir.join("../cugraph/cpp/CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", cuvs.cmake_config_dir.join("cuvs-config.cmake").display());
    println!("cargo:rerun-if-changed={}", cuvs.library_selector_path.display());
    println!("cargo:rerun-if-changed={}", cuvs.library_path.display());
    for path in &cuvs.source_revision_inputs {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed={}", validated_loader.cugraph_library_selector_path.display());
    println!("cargo:rerun-if-changed={}", validated_loader.cugraph_library_path.display());
}

fn verify_prebuilt_version(prefix: &std::path::Path) {
    rapids_build_support::verify_prebuilt_crate_version(
        &NATIVE_COMPONENT,
        prefix,
        "share/algeon-cugraph/crate-version",
        env!("CARGO_PKG_VERSION"),
    )
    .unwrap_or_else(|error| panic!("{error}"));
}
