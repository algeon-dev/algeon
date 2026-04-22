#[path = "../../scripts/build/rapids.rs"]
mod rapids;

fn main() {
    let cugraph_runtime = if std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some() {
        rapids::CugraphRuntime::EmitRpathAndRequireCudfIdentity
    } else {
        rapids::CugraphRuntime::Disabled
    };

    rapids::configure(rapids::Config {
        support_source: "../../scripts/build/rapids.rs",
        link_arg_targets: rapids::LinkArgTargets::new(true, false, true),
        cugraph_runtime,
    });
}
