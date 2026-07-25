use super::*;

pub(crate) fn candidate_shape_family(candidate: &GpuCandidate) -> String {
    let shape = candidate.shape.as_str();
    if shape == "UnsupportedNativePhysicalPlan" {
        return NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN.to_string();
    }

    let root = if shape.starts_with("SortPreservingMergeExec(fetch)") {
        "sort_preserving_merge_fetch"
    } else if shape.starts_with("SortPreservingMergeExec") {
        "sort_preserving_merge"
    } else if shape.starts_with("LocalLimitExec") {
        "local_limit_sort"
    } else if shape.starts_with("SortExec(fetch)") {
        "sort_fetch"
    } else if shape.starts_with("SortExec") {
        "sort"
    } else if shape.starts_with("ProjectionExec(AggregateExec")
        || shape.starts_with("ProjectionExec(FinalAggregateExec")
    {
        "projected_global"
    } else if shape.starts_with("AggregateExec") {
        "global"
    } else {
        "other"
    };
    let input = match (shape.contains("HashJoinExec"), shape.contains("NestedLoopJoinExec")) {
        (true, true) => "mixed_hash_nested_loop_join",
        (true, false) => "hash_join",
        (false, true) => "nested_loop_join",
        (false, false) => "local_parquet",
    };
    let distribution = if root_is_aggregate_shape(shape)
        && (shape.contains("FinalAggregateExec") || shape.contains("PartialAggregateExec"))
    {
        "distributed_aggregate"
    } else if root_is_aggregate_shape(shape) {
        "aggregate"
    } else {
        "input"
    };

    match (root, input, distribution) {
        ("sort", "local_parquet", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_LOCAL_PARQUET_AGGREGATE.to_string()
        },
        ("sort_fetch", "local_parquet", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_LOCAL_PARQUET_AGGREGATE.to_string()
        },
        ("sort_preserving_merge_fetch", "local_parquet", "distributed_aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_PRESERVING_MERGE_FETCH_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE.to_string()
        },
        ("projected_global", "local_parquet", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_AGGREGATE.to_string()
        },
        ("projected_global", "local_parquet", "distributed_aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE.to_string()
        },
        ("sort_fetch", "hash_join", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_AGGREGATE.to_string()
        },
        ("sort", "hash_join", "aggregate") => NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_AGGREGATE.to_string(),
        ("projected_global", "hash_join", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_HASH_JOIN_AGGREGATE.to_string()
        },
        ("sort", "mixed_hash_nested_loop_join", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_AGGREGATE.to_string()
        },
        ("sort_fetch", "hash_join", "input") => NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_INPUT.to_string(),
        ("sort", "hash_join", "input") => NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_INPUT.to_string(),
        ("sort", "mixed_hash_nested_loop_join", "input") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_INPUT.to_string()
        },
        _ => format!("{root}_{input}_{distribution}"),
    }
}

fn root_is_aggregate_shape(shape: &str) -> bool {
    shape.starts_with("AggregateExec")
        || shape.starts_with("FinalAggregateExec")
        || shape.starts_with("ProjectionExec(AggregateExec")
        || shape.starts_with("ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortExec(AggregateExec")
        || shape.starts_with("SortExec(FinalAggregateExec")
        || shape.starts_with("SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("SortExec(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortExec(fetch)(AggregateExec")
        || shape.starts_with("SortExec(fetch)(FinalAggregateExec")
        || shape.starts_with("SortExec(fetch)(ProjectionExec(AggregateExec")
        || shape.starts_with("SortExec(fetch)(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(AggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(FinalAggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(ProjectionExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(SortExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(SortExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(SortExec(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(ProjectionExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_preserving_merge_shapes_keep_aggregate_family() {
        for shape in [
            "SortPreservingMergeExec(AggregateExec(LocalParquetScan))",
            "SortPreservingMergeExec(SortExec(AggregateExec(LocalParquetScan)))",
        ] {
            let candidate = GpuCandidate::with_operator_count(shape, 2, 3);

            assert_eq!(candidate_shape_family(&candidate), "sort_preserving_merge_local_parquet_aggregate", "{shape}");
        }
    }

    #[test]
    fn sort_preserving_merge_fetch_without_child_sort_keeps_aggregate_family() {
        let candidate =
            GpuCandidate::with_operator_count("SortPreservingMergeExec(fetch)(AggregateExec(LocalParquetScan))", 2, 2);

        assert_eq!(candidate_shape_family(&candidate), "sort_preserving_merge_fetch_local_parquet_aggregate");
    }
}
