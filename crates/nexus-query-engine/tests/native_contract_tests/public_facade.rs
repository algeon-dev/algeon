use std::path::Path;

use nexus_query_engine::{
    admission::{
        AdaptivePipelineAdmission, StrictBoundedPipelineAdmission, adaptive_bounded_pipeline_admission,
        strict_bounded_pipeline_admission,
    },
    capability::AggregateStrategyExecutionOptions,
    error::Result,
    expr::Expr,
    metrics::QueryMetrics,
    mode::NativePipelineExecutionMode,
    optimizer::NativeOptimizerOptions,
    pipeline::PipelineTaskKind,
    plan::QueryPlan,
    runtime::{
        ExecutionContext,
        cache::{
            hash_join::GpuHashJoinBuildCache,
            local_parquet::{GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey},
        },
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{LocalParquetSource, validate_local_parquet_source},
    triage::TriageSummary,
};

use super::public_surface::{assert_no_crate_root_reexports, assert_no_removed_public_surface};

#[test]
fn canonical_engine_paths_expose_intentional_contract_groups() {
    let _mode = NativePipelineExecutionMode::StrictBounded;
    let _expr = Expr::column("c0");
    let _metrics = QueryMetrics::default();
    let _runtime = ExecutionContext::builder();
    let _hash_join_cache = GpuHashJoinBuildCache::new();
    let _local_parquet_cache = GpuLocalParquetSourceCache::new(0);
    let _local_parquet_key: fn(&LocalParquetSource) -> GpuLocalParquetSourceCacheKey =
        GpuLocalParquetSourceCacheKey::from_source;
    let _resident_input_map = GpuResidentInputMap::new();
    let _gpu_frame_new: fn(cudf::DataFrame) -> GpuDataFrame = GpuDataFrame::new;
    let _gpu_handle_new: fn(&ExecutionContext, GpuDataFrame, arrow::datatypes::SchemaRef) -> Result<GpuResultHandle> =
        GpuResultHandle::new;
    let _optimizer = NativeOptimizerOptions::default();
    let _pipeline_kind = PipelineTaskKind::SourceRead;
    let _aggregate = AggregateStrategyExecutionOptions::default();
    let _triage = TriageSummary::healthy();
    let _strict_admission: fn(&QueryPlan) -> Result<Option<StrictBoundedPipelineAdmission>> =
        strict_bounded_pipeline_admission;
    let _adaptive_admission: fn(&QueryPlan) -> Result<Option<AdaptivePipelineAdmission>> =
        adaptive_bounded_pipeline_admission;
    let _source_validation: fn(&LocalParquetSource) -> Result<()> = validate_local_parquet_source;
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
