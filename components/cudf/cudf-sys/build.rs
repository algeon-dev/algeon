mod build_support;

use std::{env, path::PathBuf};

fn main() {
    let native_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("native");
    let mut config = cmake::Config::new(&native_dir);

    if let Some(cuda_static_runtime) =
        build_support::normalized_cmake_bool_from_env("CUDA_STATIC_RUNTIME").unwrap_or_else(|error| panic!("{error}"))
    {
        config.define("CUDA_STATIC_RUNTIME", cuda_static_runtime);
    }

    // Build the C++ shim library using CMake, then replay the link directives it emitted.
    let dst = config.build();
    for directive in build_support::read_cargo_link_directives(&dst).unwrap_or_else(|error| panic!("{error}")) {
        println!("{directive}");
    }

    // Re-run if native sources or build inputs change.
    println!("cargo:rerun-if-changed=native/");
    println!("cargo:rerun-if-env-changed=CUDF_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUDF_CPP_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_STATIC_RUNTIME");
}
