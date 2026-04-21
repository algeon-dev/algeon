#[path = "../../build-support/rapids.rs"]
mod rapids;

fn main() {
    let cugraph_runtime = if std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some() {
        rapids::CugraphRuntime::EmitRpathAndRequireCudfIdentity
    } else {
        rapids::CugraphRuntime::Disabled
    };

    rapids::configure(rapids::Config {
        support_source: "../../build-support/rapids.rs",
        link_arg_targets: rapids::LinkArgTargets::new(true, true, true),
        cugraph_runtime,
    });
}
