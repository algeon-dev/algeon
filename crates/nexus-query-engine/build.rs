fn main() {
    nexus_build_support::configure(nexus_build_support::Config {
        link_arg_targets: nexus_build_support::LinkArgTargets::new(true, false, true),
        consumers: nexus_build_support::RapidsConsumers::new(
            std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some(),
            std::env::var_os("CARGO_FEATURE_CUVS").is_some(),
        ),
        nvtx3interop: true,
    });
}
