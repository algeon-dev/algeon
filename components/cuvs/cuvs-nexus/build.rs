use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUVS_C_RUNTIME_LIB_DIRS");

    let Some(encoded) = env::var_os("DEP_CUVS_C_RUNTIME_LIB_DIRS") else {
        return;
    };
    for path in env::split_paths(&encoded) {
        if !path.is_dir() {
            panic!("DEP_CUVS_C_RUNTIME_LIB_DIRS contains a directory that does not exist: {}", path.display());
        }
        // `rustc-link-arg-tests` does not apply to the unit-test binary that
        // Cargo builds for this library. Keep the canonical cuVS runtime
        // closure available to this package's unit tests, integration tests,
        // examples, and binaries. Cargo does not propagate rustc link args
        // through an rlib to downstream executable packages; consumers must
        // establish their own loader contract as documented in the README.
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
}
