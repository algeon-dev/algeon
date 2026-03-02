use super::*;
use crate::projection_outputs;

// Plan fixtures are grouped by admitted shape and included at this module
// level so existing `use fixtures_plans::*` consumers keep working.
include!("fixtures_plans/join_roots.rs");
include!("fixtures_plans/root_full_sort.rs");
include!("fixtures_plans/rollup_rank_topk.rs");
include!("fixtures_plans/pipelines.rs");
include!("fixtures_plans/root_topk.rs");
include!("fixtures_plans/projection_global_limit.rs");
include!("fixtures_plans/projection_scalar_setop.rs");
include!("fixtures_plans/adaptive.rs");
