use super::{CandidateShape, GpuCandidate, NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN};
use crate::report::UNSUPPORTED_NATIVE_PLAN_OPERATOR;

/// Family of a candidate shape, `{root}_{input}_{body}`.
///
/// `body` is the first operator below the projections, sorts and limits at
/// the top of the shape: `aggregate`, `distributed_aggregate`, or `input`.
/// `root` is the outermost sort or limit above the body; without one, an
/// aggregate body is `global`, or `projected_global` under a projection, and
/// any other body is `other`. `input` names the joins the shape covers, or
/// else a union, or else the kind of its leaf sources.
pub(crate) fn candidate_shape_family(candidate: &GpuCandidate) -> String {
    let shape = candidate.shape.without_relation_input_label();
    if shape.operator() == UNSUPPORTED_NATIVE_PLAN_OPERATOR {
        return NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN.to_string();
    }

    let mut root = None;
    let mut projected = false;
    let mut body = Some(shape);
    while let Some(node) = body {
        match node.operator() {
            "ProjectionExec" => projected = true,
            operator => match ordering_or_limit_root(operator) {
                Some(kind) => {
                    root.get_or_insert(kind);
                },
                None => break,
            },
        }
        body = match node.inputs() {
            [input] => Some(input),
            _ => None,
        };
    }
    let body = match body.map(CandidateShape::operator) {
        Some("AggregateExec") => "aggregate",
        Some("FinalAggregateExec") => "distributed_aggregate",
        _ => "input",
    };
    let root = root.unwrap_or(match (body, projected) {
        ("input", _) => "other",
        (_, true) => "projected_global",
        (_, false) => "global",
    });
    format!("{root}_{}_{body}", input_family(shape))
}

fn ordering_or_limit_root(operator: &str) -> Option<&'static str> {
    match operator {
        "SortPreservingMergeExec(fetch)" => Some("sort_preserving_merge_fetch"),
        "SortPreservingMergeExec" => Some("sort_preserving_merge"),
        "SortExec(fetch)" => Some("sort_fetch"),
        "SortExec" => Some("sort"),
        "GlobalLimitExec" | "LocalLimitExec" => Some("limit"),
        _ => None,
    }
}

fn input_family(shape: &CandidateShape) -> &'static str {
    let hash = shape.contains_operator("HashJoinExec") || shape.contains_operator("HashJoinExec(fetch)");
    let nested_loop = shape.contains_operator("NestedLoopJoinExec");
    let cross = shape.contains_operator("CrossJoinExec");
    match (hash, nested_loop, cross) {
        (true, true, false) => return "mixed_hash_nested_loop_join",
        (true, false, false) => return "hash_join",
        (false, true, false) => return "nested_loop_join",
        (false, false, true) => return "cross_join",
        (false, false, false) => {},
        _ => return "mixed_join",
    }
    if shape.contains_operator("UnionExec") {
        return "union";
    }
    let mut sources = Vec::new();
    collect_leaf_source_kinds(shape, &mut sources);
    sources.dedup();
    match sources.as_slice() {
        [kind] => kind,
        _ => "mixed_source",
    }
}

fn collect_leaf_source_kinds(shape: &CandidateShape, sources: &mut Vec<&'static str>) {
    if shape.inputs().is_empty() {
        let kind = match shape.operator() {
            "LocalParquetScan" => "local_parquet",
            "MemorySource" | "PlaceholderRowExec" | "EmptyExec" => "literal",
            "GpuResidentSource" => "gpu_resident",
            _ => "other",
        };
        if !sources.contains(&kind) {
            sources.push(kind);
        }
    }
    for input in shape.inputs() {
        collect_leaf_source_kinds(input, sources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unary(operator: &str, input: CandidateShape) -> CandidateShape {
        CandidateShape::new(operator, vec![input])
    }

    fn family(shape: CandidateShape) -> String {
        candidate_shape_family(&GpuCandidate::new(shape, 1))
    }

    #[test]
    fn root_is_the_outermost_sort_above_projections() {
        let aggregate = unary("AggregateExec", CandidateShape::leaf("LocalParquetScan"));
        assert_eq!(
            family(unary("ProjectionExec", unary("SortExec", aggregate.clone()))),
            "sort_local_parquet_aggregate"
        );
        assert_eq!(
            family(unary("SortPreservingMergeExec", unary("SortExec", aggregate.clone()))),
            "sort_preserving_merge_local_parquet_aggregate"
        );
        assert_eq!(family(unary("ProjectionExec", aggregate)), "projected_global_local_parquet_aggregate");
    }

    #[test]
    fn input_names_the_sources_the_shape_reads() {
        assert_eq!(family(unary("ProjectionExec", CandidateShape::leaf("MemorySource"))), "other_literal_input");
        assert_eq!(
            family(unary(
                "SortExec",
                CandidateShape::new(
                    "UnionExec",
                    vec![CandidateShape::leaf("LocalParquetScan"), CandidateShape::leaf("MemorySource")]
                )
            )),
            "sort_union_input"
        );
        assert_eq!(
            family(unary(
                "AggregateExec",
                CandidateShape::new(
                    "CrossJoinExec",
                    vec![CandidateShape::leaf("LocalParquetScan"), CandidateShape::leaf("LocalParquetScan")]
                )
            )),
            "global_cross_join_aggregate"
        );
    }

    #[test]
    fn relation_input_label_is_not_an_operator() {
        let inner = unary("ProjectionExec", unary("AggregateExec", CandidateShape::leaf("LocalParquetScan")));
        let labelled = CandidateShape::relation_input("cugraph_pagerank.edges", inner.clone());

        assert_eq!(family(labelled.clone()), family(inner.clone()));
        assert_eq!(
            family(CandidateShape::relation_input(
                "cugraph_pagerank.edges",
                CandidateShape::leaf(UNSUPPORTED_NATIVE_PLAN_OPERATOR)
            )),
            NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN
        );
        assert_eq!(labelled.to_string(), "cugraph_pagerank.edges(ProjectionExec(AggregateExec(LocalParquetScan)))");
        assert_eq!(labelled.operator_count(), inner.operator_count());
    }
}
