fn main() {
    rapids_build_support::configure(rapids_build_support::Config {
        link_arg_targets: rapids_build_support::LinkArgTargets::new(true, false, false),
        consumers: rapids_build_support::RapidsConsumers::new(false, false, false),
        nvtx3interop: false,
    });
}
