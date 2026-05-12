use super::*;

mod expressions;
mod shape_family;

use expressions::expression_category;
pub(super) use shape_family::{candidate_shape_family, physical_plan_shape};

pub(super) fn post_native_execution_path(plan: &dyn ExecutionPlan) -> NativeDataFusionPostNativeExecutionPath {
    if plan.as_any().downcast_ref::<NativeCudfExec>().is_some()
        || plan.as_any().downcast_ref::<NativeCudfComposedExec>().is_some()
    {
        return NativeDataFusionPostNativeExecutionPath::Native;
    }
    if contains_native_exec(plan) {
        return NativeDataFusionPostNativeExecutionPath::PartialNative;
    }
    NativeDataFusionPostNativeExecutionPath::Cpu
}

fn contains_native_exec(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<NativeCudfExec>().is_some()
        || plan.as_any().downcast_ref::<NativeCudfComposedExec>().is_some()
        || plan.as_any().downcast_ref::<NativeCudfGpuProducerExec>().is_some()
        || plan.children().iter().any(|child| contains_native_exec(child.as_ref()))
}

pub(super) fn join_diagnostics(plan: &dyn ExecutionPlan) -> Vec<NativeJoinDiagnosticSummary> {
    let mut diagnostics = Vec::new();
    collect_join_diagnostics(plan, &mut diagnostics);
    diagnostics
}

fn collect_join_diagnostics(plan: &dyn ExecutionPlan, diagnostics: &mut Vec<NativeJoinDiagnosticSummary>) {
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        diagnostics.push(NativeJoinDiagnosticSummary {
            operator: "HashJoinExec".to_string(),
            join_type: stable_debug_value(hash_join.join_type()),
            key_count: hash_join.on().len(),
            key_shape: hash_join
                .on()
                .iter()
                .map(|(left, right)| format!("{left}={right}"))
                .collect::<Vec<_>>()
                .join(","),
            residual_predicate: hash_join.filter().is_some(),
            null_equality: stable_debug_value(&hash_join.null_equality()),
            build_side: "left".to_string(),
            probe_side: "right".to_string(),
        });
    } else if let Some(nested_loop_join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
        diagnostics.push(NativeJoinDiagnosticSummary {
            operator: "NestedLoopJoinExec".to_string(),
            join_type: stable_debug_value(nested_loop_join.join_type()),
            key_count: 0,
            key_shape: "none".to_string(),
            residual_predicate: nested_loop_join.filter().is_some(),
            null_equality: "not_applicable".to_string(),
            build_side: "left".to_string(),
            probe_side: "right".to_string(),
        });
    }

    for child in plan.children() {
        collect_join_diagnostics(child.as_ref(), diagnostics);
    }
}

pub(super) fn summarize_source_diagnostics(diagnostics: &NativeSourceDiagnostics) -> NativeSourceDiagnosticSummary {
    NativeSourceDiagnosticSummary {
        format: format!("{:?}", diagnostics.format),
        direct_device_read: diagnostics.direct_device_read,
        file_count: diagnostics.file_count,
        ranged_file_count: diagnostics.ranged_file_count,
        known_bytes: diagnostics.known_bytes,
        selected_row_groups: diagnostics.selected_row_groups,
        predicate_disposition: diagnostics.predicate_disposition.as_str().to_string(),
        byte_range_disposition: diagnostics.byte_range_disposition.as_str().to_string(),
        partition_column_disposition: diagnostics.partition_column_disposition.as_str().to_string(),
        zero_column_count_disposition: diagnostics.zero_column_count_disposition.as_str().to_string(),
        scan_limit_disposition: diagnostics.scan_limit_disposition.as_str().to_string(),
        schema_mapping_disposition: diagnostics.schema_mapping_disposition.as_str().to_string(),
        read_column_count: diagnostics.read_column_count,
        projected_output_column_count: diagnostics.projected_output_column_count,
        pure_filter_column_count: diagnostics.pure_filter_column_count,
        rejected_features: diagnostics.rejected_features.iter().map(|feature| feature.as_str().to_string()).collect(),
    }
}

fn stable_debug_value(value: &impl std::fmt::Debug) -> String {
    format!("{value:?}").to_ascii_lowercase()
}

pub(super) fn unsupported_category(
    plan: &dyn ExecutionPlan,
    candidate: &GpuCandidate,
    reason: FallbackReason,
) -> Option<String> {
    match reason {
        FallbackReason::UnsupportedExpression => first_unsupported_expression_category(plan)
            .or_else(|| {
                first_aggregate_category(plan).filter(|category| {
                    matches!(
                        category.as_str(),
                        "aggregate_filter" | "aggregate_distinct" | "aggregate_order_by" | "aggregate_argument_count"
                    )
                })
            })
            .or_else(|| Some("expression".to_string())),
        FallbackReason::UnsupportedOperator => first_aggregate_category(plan)
            .or_else(|| first_unsupported_operator_category(plan, candidate.shape.as_str()))
            .or_else(|| first_global_aggregate_category(plan))
            .or_else(|| Some("operator_shape".to_string())),
        FallbackReason::UnsupportedType => {
            first_unsupported_expression_category(plan).or_else(|| Some("type".to_string()))
        },
        FallbackReason::NativeDistributedAggregateUnsupported => {
            first_aggregate_category(plan).or_else(|| Some("aggregate_distributed".to_string()))
        },
        FallbackReason::NativeSourceNonLocalObjectStore
        | FallbackReason::NativeSourcePartitionColumns
        | FallbackReason::NativeSourceExprAdapter
        | FallbackReason::NativeSourceParquetFileReaderFactory
        | FallbackReason::NativeSourceScanLimit
        | FallbackReason::NativeSourcePreserveOrder
        | FallbackReason::NativeSourceOutputOrdering
        | FallbackReason::NativeSourcePushedDownParquetPredicate
        | FallbackReason::NativeSourceEmpty
        | FallbackReason::NativeSourcePartitionValues
        | FallbackReason::NativeSourceByteRange
        | FallbackReason::NativeSourceFileOrdering
        | FallbackReason::NativeSourceRowSelection
        | FallbackReason::NativeSourceSchemaMapping => Some("source".to_string()),
        FallbackReason::NativeSortPreservingMergeUnsupported
        | FallbackReason::NativePartitionPreservingSortUnsupported => Some("operator_sort".to_string()),
        FallbackReason::NativeBoundedJoinUnsupported | FallbackReason::NativeAdaptivePipelineUnsupported => {
            first_join_operator_category(plan).or_else(|| Some("operator_join".to_string()))
        },
        FallbackReason::ExplicitCpuPolicy => None,
    }
}

fn first_unsupported_operator_category(plan: &dyn ExecutionPlan, candidate_shape: &str) -> Option<String> {
    let name = plan.name();
    if !native_report_operator_is_supported(name, candidate_shape) {
        return Some(format!("operator_{}", stable_operator_identifier(name)));
    }
    for child in plan.children() {
        if let Some(category) = first_unsupported_operator_category(child.as_ref(), candidate_shape) {
            return Some(category);
        }
    }
    None
}

fn first_join_operator_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if plan.as_any().downcast_ref::<HashJoinExec>().is_some() {
        return Some("operator_hashjoin".to_string());
    }
    if plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some() {
        return Some("operator_nestedloopjoin".to_string());
    }
    for child in plan.children() {
        if let Some(category) = first_join_operator_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn native_report_operator_is_supported(name: &str, candidate_shape: &str) -> bool {
    matches!(
        name,
        "DataSourceExec"
            | "FilterExec"
            | "ProjectionExec"
            | "AggregateExec"
            | "SortExec"
            | "LocalLimitExec"
            | "GlobalLimitExec"
            | "CoalescePartitionsExec"
    ) || (name == "RepartitionExec" && candidate_shape.contains("RepartitionExec"))
}

fn stable_operator_identifier(name: &str) -> String {
    let stable = stable_identifier(name);
    if stable.starts_with("sortexec_topk") {
        return "sort_topk".to_string();
    }
    stable.strip_suffix("exec").map_or(stable.clone(), ToString::to_string)
}

fn first_aggregate_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>()
        && let Some(category) = aggregate_semantic_modifier_category(aggregate)
    {
        return Some(category);
    }

    for child in plan.children() {
        if let Some(category) = first_aggregate_category(child.as_ref())
            && matches!(category.as_str(), "aggregate_filter" | "aggregate_distinct" | "aggregate_order_by")
        {
            return Some(category);
        }
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        for expr in aggregate.aggr_expr() {
            let function = stable_identifier(expr.fun().name());
            if !matches!(function.as_str(), "sum" | "min" | "max" | "count" | "avg") {
                return Some(format!("aggregate_{function}"));
            }
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Some("aggregate_argument_count".to_string());
            };
            if let Some(category) = expression_category(arg) {
                return Some(category);
            }
        }
    }

    for child in plan.children() {
        if let Some(category) = first_aggregate_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn first_global_aggregate_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>()
        && aggregate.group_expr().expr().is_empty()
    {
        return Some("aggregate_global".to_string());
    }
    for child in plan.children() {
        if let Some(category) = first_global_aggregate_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn aggregate_semantic_modifier_category(aggregate: &AggregateExec) -> Option<String> {
    for filter in aggregate.filter_expr() {
        if filter.is_some() {
            return Some("aggregate_filter".to_string());
        }
    }
    for expr in aggregate.aggr_expr() {
        if expr.is_distinct() {
            return Some("aggregate_distinct".to_string());
        }
        if !expr.order_bys().is_empty() {
            return Some("aggregate_order_by".to_string());
        }
    }
    None
}

fn first_unsupported_expression_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>()
        && let Some(category) = expression_category(filter.predicate())
    {
        return Some(category);
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        for expr in projection.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        for expr in sort.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(merge) = plan.as_any().downcast_ref::<SortPreservingMergeExec>() {
        for expr in merge.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(category) = first_aggregate_category(plan) {
        return Some(category);
    }

    for child in plan.children() {
        if let Some(category) = first_unsupported_expression_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn stable_identifier(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { '_' }).collect()
}
