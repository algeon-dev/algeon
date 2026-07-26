#[path = "../../scripts/build/rapids.rs"]
mod rapids;

fn main() {
    rapids::configure(rapids::Config {
        support_source: "../../scripts/build/rapids.rs",
        link_arg_targets: rapids::LinkArgTargets::new(true, false, true),
        consumers: rapids::RapidsConsumers::new(
            std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some(),
            std::env::var_os("CARGO_FEATURE_CUVS").is_some(),
        ),
    });

    // Nsight Systems consumes the owning execution ranges through NVTX. The
    // CUDA toolchain provides this interop library independently of cuDF and
    // cuGraph, so link it only for the graph-enabled engine.
    if std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some() {
        println!("cargo:rustc-link-lib=dylib=nvtx3interop");
    }
}
