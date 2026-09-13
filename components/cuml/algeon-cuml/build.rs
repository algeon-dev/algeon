use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUML_RUST_RUNTIME_LIB_DIRS");
    let Some(encoded) = env::var_os("DEP_CUML_RUST_RUNTIME_LIB_DIRS") else {
        return;
    };
    for path in env::split_paths(&encoded) {
        if !path.is_dir() {
            panic!("cuML runtime library directory does not exist: {}", path.display());
        }
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
}
