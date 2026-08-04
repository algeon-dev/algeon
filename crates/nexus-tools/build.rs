#[path = "../../scripts/build/rapids.rs"]
mod rapids;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Only the GPU-linking tool features need the RAPIDS rpath bake, and the
    // CPU-only features must still build without a RAPIDS install. Optional
    // dependencies enabled through `dep:` do not set `CARGO_FEATURE_*`, so
    // cudf-sys's `links` metadata is the signal that this build links cuDF.
    if std::env::var_os("DEP_CUDF_RUST_RUNTIME_LIB_DIR").is_none() {
        return;
    }
    rapids::configure(rapids::Config {
        support_source: "../../scripts/build/rapids.rs",
        // `all` rather than per-target: the bin carries `test = true`, so its
        // unit-test harness needs the same rpath, and this crate has no
        // separate test or bench targets for the narrower flags to apply to.
        link_arg_targets: rapids::LinkArgTargets::new(true, false, false),
        consumers: rapids::RapidsConsumers::new(
            std::env::var_os("DEP_CUGRAPH_RUNTIME_LIB_DIRS").is_some(),
            std::env::var_os("DEP_CUVS_C_METADATA").is_some(),
        ),
        nvtx3interop: false,
    });
}
