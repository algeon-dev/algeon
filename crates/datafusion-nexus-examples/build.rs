#[path = "../../scripts/build/rapids.rs"]
mod rapids;

fn main() {
    rapids::configure(rapids::Config {
        support_source: "../../scripts/build/rapids.rs",
        link_arg_targets: rapids::LinkArgTargets::new(false, true, false),
        cugraph_runtime: rapids::CugraphRuntime::EmitRpathAndRequireCudfIdentity,
    });
}
