#[path = "../../build-support/rapids.rs"]
mod rapids;

fn main() {
    rapids::configure(rapids::Config {
        support_source: "../../build-support/rapids.rs",
        link_arg_targets: rapids::LinkArgTargets::new(false, true, false),
        cugraph_runtime: rapids::CugraphRuntime::EmitRpathAndRequireCudfIdentity,
    });
}
