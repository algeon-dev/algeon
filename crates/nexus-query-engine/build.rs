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
        nvtx3interop: true,
    });
}
