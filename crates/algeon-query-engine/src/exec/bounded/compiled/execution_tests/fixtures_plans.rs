use fixture_local_parquet_source as LocalParquetSource;
pub(super) use join_limit_projection::shared_filtered_dimension_fan_out_plan;
pub(super) use join_roots::{
    aggregate_join_build_input_data_plan, keyless_root_join_data_plan, residual_one_shot_frame_stream_multi_join_plan,
    residual_one_shot_leaf_stream_join_plan, residual_one_shot_leaf_stream_left_build_join_plan, root_join_data_plan,
    root_join_gpu_sink_data_plan, root_join_sort_limit_filter_data_plan, root_join_sort_limit_offset_data_plan,
    root_join_with_unary_inputs_data_plan, shared_source_join_build_and_aggregate_data_plan,
    shared_source_self_join_data_plan, simple_multi_join_plan, single_join_post_root_aggregate_data_plan,
    single_join_post_root_aggregate_sort_limit_data_plan, string_group_join_post_root_aggregate_data_plan,
};
pub(super) use root_topk::aggregate_sort_limit_data_plan;

use super::{projection_outputs, *};
use crate::rapids_backend::rapids_analysis_context;

mod join_limit_projection;
mod join_roots;
mod pipelines;
mod projection_scalar_setop;
mod rollup_rank_topk;
mod root_full_sort;
mod root_topk;

mod fixture_local_parquet_source {
    use super::*;

    pub(super) fn new(
        path: impl Into<PathBuf>,
        schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        let path = path.into();
        if path.is_file() {
            NativeLocalParquetSource::from_files(vec![fixture_local_parquet_file(&path)], schema, read_columns)
        } else {
            NativeLocalParquetSource::new(path, schema, read_columns)
        }
    }

    pub(super) fn from_files(
        files: Vec<LocalParquetFile>,
        schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        NativeLocalParquetSource::from_files(
            files.into_iter().map(enrich_fixture_local_parquet_file).collect(),
            schema,
            read_columns,
        )
    }
}

pub(super) fn incremental_partitioned_merge_builder(
    dir: &Path,
    row_groups: usize,
    rows_per_row_group: usize,
) -> (QueryPlanBuilder, NodeId, Arc<Schema>) {
    incremental_merge_fixture::incremental_partitioned_merge_builder(
        dir,
        row_groups,
        rows_per_row_group,
        LocalParquetSource::from_files,
    )
}

// Shared with the black-box budgeting cases in `tests/`.
#[path = "../../../../../tests/support/incremental_merge_fixture.rs"]
mod incremental_merge_fixture;
