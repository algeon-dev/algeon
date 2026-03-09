use std::env;
use std::path::PathBuf;

fn main() {
    let native_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("native");

    // Build the C++ shim library using CMake.
    let dst = cmake::Config::new(&native_dir)
        .define("BUILD_SHARED_LIBS", "ON")
        .build();

    // Tell cargo to link against the built library.
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=cudf_rust");

    // Also need libcudf itself and CUDA runtime on the link path.
    // These are typically found via the environment or CMake.
    if let Ok(cudf_install) = env::var("CUDF_INSTALL_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", cudf_install);
        println!("cargo:rustc-link-search=native={}/lib64", cudf_install);
    }
    if let Ok(cudf_build) = env::var("CUDF_CPP_BUILD_DIR") {
        println!("cargo:rustc-link-search=native={}", cudf_build);
        println!("cargo:rustc-link-search=native={}/lib", cudf_build);
    }
    if let Ok(cuda_path) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }

    println!("cargo:rustc-link-lib=dylib=cudf");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Re-run if native sources change.
    println!("cargo:rerun-if-changed=native/");
    println!("cargo:rerun-if-env-changed=CUDF_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=CUDF_CPP_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
