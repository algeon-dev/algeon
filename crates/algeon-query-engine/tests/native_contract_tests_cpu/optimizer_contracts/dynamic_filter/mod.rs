use fixtures::{
    SharedPartsuppDynamicFilterPlanOptions, build_subtree_source_reuse_dynamic_filter_cycle_plan,
    q02_like_shared_partsupp_dynamic_filter_plan, repeated_consumer_file_set_dynamic_filter_plan,
    shared_filtered_prefix_branch_local_dynamic_filter_plan, shared_prefix_branch_local_dynamic_filter_plan,
};

use super::*;

mod aggregate;
mod fixtures;
mod payload_metrics;
mod residual;
mod shared_sources;
mod source_routes;
mod union;
