#[path = "../../scripts/build/rapids.rs"]
mod rapids;

fn main() {
    rapids::configure(rapids::Config {
        support_source: "../../scripts/build/rapids.rs",
        link_arg_targets: rapids::LinkArgTargets::new(true, false, false),
        consumers: rapids::RapidsConsumers::new(true, std::env::var_os("CARGO_FEATURE_CUVS").is_some()),
        nvtx3interop: false,
    });
}
