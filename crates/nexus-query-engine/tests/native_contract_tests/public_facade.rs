use std::path::Path;

use nexus_query_engine::{
    capability::AggregateStrategyExecutionOptions, expr::Expr, metrics::QueryMetrics,
    mode::NativePipelineExecutionMode, optimizer::NativeOptimizerOptions, pipeline::PipelineTaskKind,
    runtime::ExecutionContext, triage::TriageSummary,
};

use super::public_surface::{assert_no_crate_root_reexports, assert_no_removed_public_surface};

#[test]
fn canonical_engine_paths_expose_intentional_contract_groups() {
    let _mode = NativePipelineExecutionMode::StrictBounded;
    let _expr = Expr::column("c0");
    let _metrics = QueryMetrics::default();
    let _runtime = ExecutionContext::builder();
    let _optimizer = NativeOptimizerOptions::default();
    let _pipeline_kind = PipelineTaskKind::SourceRead;
    let _aggregate = AggregateStrategyExecutionOptions::default();
    let _triage = TriageSummary::healthy();
}

#[test]
fn removed_engine_facades_and_glob_reexports_do_not_reappear() {
    let root = workspace_root();
    assert_no_crate_root_reexports(&root.join("crates/nexus-query-engine/src/lib.rs"));
    assert_no_removed_public_surface(&root.join("crates/nexus-query-engine/src"));
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("engine crate must remain under the workspace root")
        .to_path_buf()
}
