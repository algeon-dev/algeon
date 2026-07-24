//! Planner-side plan inspection helpers for native report rows.
//!
//! Walks DataFusion physical plans to derive native fragment coverage, join
//! and source diagnostics summaries, and stable unsupported-category labels
//! used by the planning/execution/fallback-pressure surfaces.

use super::*;

mod expressions;
mod shape_family;

use expressions::{
    expression_category, expression_category_with_schema, join_residual_expression_category,
    stable_data_type_identifier,
};
pub(super) use shape_family::candidate_shape_family;

pub(super) use crate::physical_plan::physical_plan_shape;

pub(crate) fn post_native_plan_coverage(plan: &dyn ExecutionPlan) -> NativeDataFusionPlanCoverage {
    let native_fragment_count = count_native_fragments(plan);
    let host_datafusion_boundary_count = count_host_datafusion_boundaries(plan);
    let execution_path = if native_plan_root_is_whole_native(plan) {
        NativeDataFusionPostNativeExecutionPath::Native
    } else if native_fragment_count > 0 {
        NativeDataFusionPostNativeExecutionPath::PartialNative
    } else {
        NativeDataFusionPostNativeExecutionPath::Cpu
    };
    NativeDataFusionPlanCoverage { execution_path, native_fragment_count, host_datafusion_boundary_count }
}

fn contains_native_exec(plan: &dyn ExecutionPlan) -> bool {
    count_native_fragments(plan) > 0
}

fn native_plan_root_is_whole_native(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<NexusNativeExec>().is_some() || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
}

fn is_native_fragment(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<NexusNativeExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        || plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some()
}

fn count_native_fragments(plan: &dyn ExecutionPlan) -> usize {
    usize::from(is_native_fragment(plan))
        + plan.children().iter().map(|child| count_native_fragments(child.as_ref())).sum::<usize>()
}

fn count_host_datafusion_boundaries(plan: &dyn ExecutionPlan) -> usize {
    let children = plan.children();
    let direct_native_child_boundary =
        !is_native_fragment(plan) && children.iter().any(|child| contains_native_exec(child.as_ref()));
    usize::from(direct_native_child_boundary)
        + children.iter().map(|child| count_host_datafusion_boundaries(child.as_ref())).sum::<usize>()
}

pub(crate) fn join_diagnostics(plan: &dyn ExecutionPlan) -> Vec<NativeJoinDiagnosticSummary> {
    let mut diagnostics = Vec::new();
    collect_join_diagnostics(plan, &mut diagnostics);
    diagnostics
}

fn collect_join_diagnostics(plan: &dyn ExecutionPlan, diagnostics: &mut Vec<NativeJoinDiagnosticSummary>) {
    if let Some(hash_join) = plan.downcast_ref::<HashJoinExec>() {
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
    } else if let Some(nested_loop_join) = plan.downcast_ref::<NestedLoopJoinExec>() {
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
    } else if plan.downcast_ref::<CrossJoinExec>().is_some() {
        diagnostics.push(NativeJoinDiagnosticSummary {
            operator: "CrossJoinExec".to_string(),
            join_type: "cross".to_string(),
            key_count: 0,
            key_shape: "none".to_string(),
            residual_predicate: false,
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
    let stats = diagnostics.selected_row_group_stats.as_ref();
    NativeSourceDiagnosticSummary {
        format: format!("{:?}", diagnostics.format),
        source_read_strategy: diagnostics.source_read_strategy.clone(),
        credential_mode: diagnostics.credential_mode.clone(),
        lakehouse_cache_status: diagnostics.lakehouse_cache_status.clone(),
        lakehouse_cache_read_mode: diagnostics.lakehouse_cache_read_mode.clone(),
        lakehouse_cache_fill_mode: diagnostics.lakehouse_cache_fill_mode.clone(),
        direct_device_read: diagnostics.direct_device_read,
        file_count: diagnostics.file_count,
        ranged_file_count: diagnostics.ranged_file_count,
        known_bytes: diagnostics.known_bytes,
        selected_row_groups: diagnostics.selected_row_groups,
        total_row_groups: stats.map(|stats| stats.total_row_groups),
        selected_source_bytes: stats.and_then(|stats| stats.selected_source_bytes()),
        total_source_bytes: stats.and_then(|stats| stats.total_source_bytes()),
        selected_rows: stats.and_then(|stats| stats.selected_rows),
        total_rows: stats.and_then(|stats| stats.total_rows),
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
            .or_else(|| first_window_category(plan))
            .or_else(|| first_unsupported_operator_category(plan, candidate.shape.as_str()))
            .or_else(|| first_global_aggregate_category(plan))
            .or_else(|| Some("operator_shape".to_string())),
        FallbackReason::UnsupportedType => first_unsupported_expression_category(plan)
            .or_else(|| first_literal_source_type_category(plan))
            .or_else(|| Some("type".to_string())),
        FallbackReason::NativeDistributedAggregateUnsupported => {
            first_aggregate_category(plan).or_else(|| Some("aggregate_distributed".to_string()))
        },
        FallbackReason::NativeSourceNonLocalObjectStore
        | FallbackReason::NativeSourcePartitionColumns
        | FallbackReason::NativeSourceExprAdapter
        | FallbackReason::NativeSourceParquetFileReaderFactory
        | FallbackReason::NativeSourcePreserveOrder
        | FallbackReason::NativeSourceOutputOrdering
        | FallbackReason::NativeSourcePushedDownParquetPredicate
        | FallbackReason::NativeSourceEmpty
        | FallbackReason::NativeSourcePartitionValues
        | FallbackReason::NativeSourceByteRange
        | FallbackReason::NativeSourceFileOrdering
        | FallbackReason::NativeSourceRowSelection
        | FallbackReason::NativeSourceSchemaMapping => Some("source".to_string()),
        FallbackReason::NativeSourceScanLimit => Some("limit_contract".to_string()),
        FallbackReason::TableFormatUnsupported
        | FallbackReason::TableFormatPlanningFailed
        | FallbackReason::IcebergNonParquetDataFile
        | FallbackReason::IcebergDeleteFilesUnsupported
        | FallbackReason::IcebergFieldIdMappingUnsupported
        | FallbackReason::IcebergNameMappingUnsupported
        | FallbackReason::IcebergNestedProjectionUnsupported
        | FallbackReason::IcebergPartitionTransformUnsupported
        | FallbackReason::IcebergResidualPredicateUnsupported
        | FallbackReason::IcebergRowSelectionUnsupported
        | FallbackReason::IcebergObjectStoreReadUnsupported
        | FallbackReason::IcebergSnapshotUnsupported => Some("table_format".to_string()),
        FallbackReason::NativeSortPreservingMergeUnsupported
        | FallbackReason::NativePartitionPreservingSortUnsupported => Some("operator_sort".to_string()),
        FallbackReason::NativeOrderedOutputUnsupported => Some("operator_ordering".to_string()),
        FallbackReason::NativeBoundedJoinUnsupported | FallbackReason::NativeAdaptivePipelineUnsupported => {
            first_join_operator_category(plan).or_else(|| Some("operator_join".to_string()))
        },
        FallbackReason::NativeSmallSelectiveTopkCpuPreferred
        | FallbackReason::NativeSmallSelectiveResidualJoinCpuPreferred
        | FallbackReason::NativeSmallSelectiveFullSortCpuPreferred => None,
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
    if plan.downcast_ref::<HashJoinExec>().is_some() {
        return Some("operator_hashjoin".to_string());
    }
    if plan.downcast_ref::<NestedLoopJoinExec>().is_some() {
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
    if name == "SortExec" || name.starts_with("SortExec(") {
        return true;
    }
    matches!(
        name,
        "DataSourceExec"
            | "FilterExec"
            | "ProjectionExec"
            | "AggregateExec"
            | "LocalLimitExec"
            | "GlobalLimitExec"
            | "CoalescePartitionsExec"
            | "NexusNativeExec"
            | "NexusNativeComposedExec"
            | "NexusNativeGpuProducerExec"
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
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
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

    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        for (expr, _alias) in aggregate.group_expr().expr() {
            if let Some(category) = expression_category_with_schema(expr, Some(aggregate.input().schema().as_ref())) {
                return Some(category);
            }
        }
        for expr in aggregate.aggr_expr() {
            let function = stable_identifier(expr.fun().name());
            if !matches!(
                function.as_str(),
                "sum"
                    | "min"
                    | "max"
                    | "count"
                    | "avg"
                    | "stddev"
                    | "stddev_samp"
                    | "stddev_pop"
                    | "var_samp"
                    | "var_sample"
                    | "var"
                    | "variance"
                    | "var_pop"
                    | "var_population"
            ) {
                return Some(format!("aggregate_{function}"));
            }
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Some("aggregate_argument_count".to_string());
            };
            if let Some(category) = expression_category_with_schema(arg, Some(aggregate.input().schema().as_ref())) {
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
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
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
    if let Some(filter) = aggregate.filter_expr().iter().flatten().next() {
        if let Some(category) = expression_category_with_schema(filter, Some(aggregate.input().schema().as_ref())) {
            return Some(format!("aggregate_filter_{category}"));
        }
        return Some("aggregate_filter".to_string());
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

fn first_window_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(window) = plan.downcast_ref::<WindowAggExec>()
        && let Some(category) = window_expr_category(window.window_expr())
    {
        return Some(category);
    }
    if let Some(window) = plan.downcast_ref::<BoundedWindowAggExec>()
        && let Some(category) = window_expr_category(window.window_expr())
    {
        return Some(category);
    }
    for child in plan.children() {
        if let Some(category) = first_window_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn window_expr_category(window_exprs: &[Arc<dyn WindowExpr>]) -> Option<String> {
    for window_expr in window_exprs {
        if let Some(category) = single_window_expr_category(window_expr) {
            return Some(category);
        }
    }
    None
}

fn single_window_expr_category(window_expr: &Arc<dyn WindowExpr>) -> Option<String> {
    if let Some(plain) = window_expr.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
        return aggregate_window_category(plain.get_aggregate_expr().fun().name(), plain.get_window_frame());
    }
    if let Some(sliding) = window_expr.as_any().downcast_ref::<SlidingAggregateWindowExpr>() {
        return aggregate_window_category(sliding.get_aggregate_expr().fun().name(), sliding.get_window_frame());
    }
    if let Some(standard) = window_expr.as_any().downcast_ref::<StandardWindowExpr>() {
        let function = standard_window_function_identifier(standard);
        if !matches!(function.as_str(), "rank" | "row_number") {
            return Some(format!("window_function_{function}"));
        }
        if function == "rank" && standard.order_by().is_empty() {
            return Some("window_function_rank_without_order".to_string());
        }
        return unsupported_window_frame_category(standard.get_window_frame(), true);
    }
    Some(format!("window_function_{}", stable_identifier(window_expr.name())))
}

fn aggregate_window_category(function: &str, frame: &WindowFrame) -> Option<String> {
    let function = stable_identifier(function);
    if !matches!(
        function.as_str(),
        "sum"
            | "min"
            | "max"
            | "count"
            | "avg"
            | "stddev"
            | "stddev_samp"
            | "stddev_pop"
            | "var_samp"
            | "var_sample"
            | "var"
            | "variance"
            | "var_pop"
            | "var_population"
    ) {
        return Some(format!("aggregate_{function}"));
    }
    unsupported_window_frame_category(frame, false)
}

fn standard_window_function_identifier(standard: &StandardWindowExpr) -> String {
    let function = standard.get_standard_func_expr();
    stable_identifier(
        function
            .as_any()
            .downcast_ref::<WindowUDFExpr>()
            .map_or_else(|| function.name().to_string(), |expr| expr.fun().name().to_string())
            .as_str(),
    )
}

fn unsupported_window_frame_category(frame: &WindowFrame, standard_function: bool) -> Option<String> {
    let whole_partition = is_whole_partition_window_frame(frame);
    let cumulative = is_rows_unbounded_preceding_to_current_window_frame(frame);
    let supported = if standard_function { true } else { whole_partition || cumulative };
    if supported {
        return None;
    }
    Some(format!("window_frame_{}", window_frame_identifier(frame)))
}

fn is_whole_partition_window_frame(frame: &WindowFrame) -> bool {
    frame.units == WindowFrameUnits::Rows
        && matches!(&frame.start_bound, WindowFrameBound::Preceding(value) if value.is_null())
        && matches!(&frame.end_bound, WindowFrameBound::Following(value) if value.is_null())
}

fn is_rows_unbounded_preceding_to_current_window_frame(frame: &WindowFrame) -> bool {
    frame.units == WindowFrameUnits::Rows
        && matches!(&frame.start_bound, WindowFrameBound::Preceding(value) if value.is_null())
        && matches!(frame.end_bound, WindowFrameBound::CurrentRow)
}

fn window_frame_identifier(frame: &WindowFrame) -> String {
    format!(
        "{}_{}_to_{}",
        stable_identifier(format!("{:?}", frame.units).as_str()),
        window_frame_bound_identifier(&frame.start_bound),
        window_frame_bound_identifier(&frame.end_bound)
    )
}

fn window_frame_bound_identifier(bound: &WindowFrameBound) -> String {
    match bound {
        WindowFrameBound::CurrentRow => "current_row".to_string(),
        WindowFrameBound::Preceding(value) if value.is_null() => "unbounded_preceding".to_string(),
        WindowFrameBound::Following(value) if value.is_null() => "unbounded_following".to_string(),
        WindowFrameBound::Preceding(value) => format!("{}_preceding", scalar_window_bound_identifier(value)),
        WindowFrameBound::Following(value) => format!("{}_following", scalar_window_bound_identifier(value)),
    }
}

fn scalar_window_bound_identifier(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Int8(Some(value)) => value.to_string(),
        ScalarValue::Int16(Some(value)) => value.to_string(),
        ScalarValue::Int32(Some(value)) => value.to_string(),
        ScalarValue::Int64(Some(value)) => value.to_string(),
        ScalarValue::UInt8(Some(value)) => value.to_string(),
        ScalarValue::UInt16(Some(value)) => value.to_string(),
        ScalarValue::UInt32(Some(value)) => value.to_string(),
        ScalarValue::UInt64(Some(value)) => value.to_string(),
        _ => stable_identifier(format!("{value:?}").as_str()).trim_matches('_').to_string(),
    }
}

fn first_literal_source_type_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if literal_source_plan(plan) {
        for field in plan.schema().fields() {
            if !is_supported_arrow_type(field.data_type()) {
                return Some(format!("literal_source_type_{}", stable_data_type_identifier(field.data_type())));
            }
        }
    }
    for child in plan.children() {
        if let Some(category) = first_literal_source_type_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn literal_source_plan(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<PlaceholderRowExec>().is_some()
        || plan.downcast_ref::<EmptyExec>().is_some()
        || plan
            .downcast_ref::<DataSourceExec>()
            .is_some_and(|source| source.data_source().downcast_ref::<MemorySourceConfig>().is_some())
}

fn first_unsupported_expression_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(filter) = plan.downcast_ref::<FilterExec>()
        && let Some(category) =
            expression_category_with_schema(filter.predicate(), Some(filter.input().schema().as_ref()))
    {
        return Some(category);
    }

    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        for expr in projection.expr() {
            if let Some(category) =
                expression_category_with_schema(&expr.expr, Some(projection.input().schema().as_ref()))
            {
                return Some(category);
            }
        }
    }

    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        for expr in sort.expr() {
            if let Some(category) = expression_category_with_schema(&expr.expr, Some(sort.input().schema().as_ref())) {
                return Some(category);
            }
        }
    }

    if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
        for expr in merge.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(category) = first_aggregate_category(plan) {
        return Some(category);
    }

    if let Some(source) = plan.downcast_ref::<DataSourceExec>()
        && let Some((file_scan, parquet_source)) = source.downcast_to_file_source::<ParquetSource>()
    {
        // DataFusion projection pushdown can absorb scalar expressions such as
        // TRY_CAST into the FileSource projection rather than leaving them in a
        // ProjectionExec above the scan. Walk those projection expressions
        // against the unprojected table schema so type-aware labels still
        // resolve here.
        let projection_input_schema = file_scan.file_source().table_schema().table_schema().as_ref();
        if let Some(projection) = file_scan.file_source().projection() {
            for projection_expr in projection.iter() {
                if let Some(category) =
                    expression_category_with_schema(&projection_expr.expr, Some(projection_input_schema))
                {
                    return Some(category);
                }
            }
        }
        if let Some(predicate) = parquet_source.filter()
            && let Some(category) = expression_category_with_schema(&predicate, Some(projection_input_schema))
        {
            return Some(category);
        }
    }

    if let Some(hash_join) = plan.downcast_ref::<HashJoinExec>() {
        for (left, right) in hash_join.on() {
            if let Some(category) = expression_category(left).or_else(|| expression_category(right)) {
                return Some(category);
            }
        }
        if let Some(filter) = hash_join.filter()
            && let Some(category) = join_residual_expression_category(filter.expression(), filter.schema())
        {
            return Some(category);
        }
    }

    if let Some(nested_loop_join) = plan.downcast_ref::<NestedLoopJoinExec>()
        && let Some(filter) = nested_loop_join.filter()
        && let Some(category) = join_residual_expression_category(filter.expression(), filter.schema())
    {
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
