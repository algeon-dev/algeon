fn main() {
    rapids_build_support::configure(rapids_build_support::Config {
        link_arg_targets: rapids_build_support::LinkArgTargets::new(true, false, false),
        consumers: rapids_build_support::RapidsConsumers::new(
            true,
            std::env::var_os("CARGO_FEATURE_CUVS").is_some(),
            std::env::var_os("CARGO_FEATURE_CUML").is_some(),
        ),
        nvtx3interop: false,
    });
}
