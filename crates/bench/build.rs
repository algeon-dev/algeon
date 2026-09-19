fn main() {
    rapids_build_support::configure(rapids_build_support::Config {
        link_arg_targets: rapids_build_support::LinkArgTargets::new(true, true, true),
        consumers: rapids_build_support::RapidsConsumers::new(
            std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some(),
            std::env::var_os("CARGO_FEATURE_CUVS").is_some(),
        ),
        nvtx3interop: false,
    });
}
