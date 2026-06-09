use super::*;
// Admission tests are split by strict/adaptive envelope while staying in this
// module via include! so test discovery paths remain stable.
include!("admission/compiled_support.rs");
include!("admission/projection_global_limit.rs");
include!("admission/projection_scalar_setop.rs");
include!("admission/root_full_sort.rs");
include!("admission/root_topk.rs");
include!("admission/rollup_rank_topk.rs");
include!("admission/contract_report.rs");
include!("admission/sink_candidates.rs");
