use std::{collections::BTreeSet, path::PathBuf, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use cudf_query_engine::{
    AggregateExpr as NativeAggregateExpr, AggregateFunction as NativeAggregateFunction,
    AggregateSpec as NativeAggregateSpec, BinaryOp as NativeBinaryOp, Expr as NativeExpr, LimitSpec as NativeLimitSpec,
    LiteralValue as NativeLiteralValue, LocalParquetFile as NativeLocalParquetFile,
    LocalParquetSource as NativeLocalParquetSource, PredicateDisposition as NativePredicateDisposition,
    QueryPlan as NativeQueryPlan, QueryPlanBuilder as NativeQueryPlanBuilder,
    RejectedSourceFeature as NativeRejectedSourceFeature, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec,
    SourceDiagnostics as NativeSourceDiagnostics, SourceNode as NativeSourceNode,
};
use datafusion::{
    common::ScalarValue,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileScanConfig, FileSource, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess},
        },
        source::DataSourceExec,
    },
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, PhysicalSortExpr,
        expressions::{BinaryExpr, Column, Literal},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use crate::{
    policy::FallbackReason,
    report::{GpuCandidate, PlanningReport},
};

const LOCAL_LIMIT_SHAPE: &str = "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))";
const SORT_FETCH_SHAPE: &str = "SortExec(fetch)(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan))))";
const SORT_PRESERVING_MERGE_SHAPE: &str = "SortPreservingMergeExec";
const UNSUPPORTED_NATIVE_PLAN_SHAPE: &str = "UnsupportedNativePhysicalPlan";

#[derive(Debug, Clone)]
pub struct NativeLoweringResult {
    pub plans: Vec<NativeQueryPlan>,
    pub report: PlanningReport,
    pub source_diagnostics: Vec<NativeSourceDiagnostics>,
}

impl NativeLoweringResult {
    pub fn candidate_count(&self) -> usize {
        self.report.outcomes().len()
    }

    pub fn rewritten_count(&self) -> usize {
        self.report.rewritten_count()
    }

    pub fn rejected_count(&self) -> usize {
        self.report.rejected_count()
    }
}

pub fn lower_physical_plan_to_native(plan: Arc<dyn ExecutionPlan>) -> NativeLoweringResult {
    match normalize_native_plan(plan.as_ref()) {
        Ok(normalized) => lower_normalized_plan(&normalized).map_or_else(
            |reason| {
                let mut report = PlanningReport::new();
                report.push_rejected(normalized.candidate, reason);
                NativeLoweringResult {
                    plans: Vec::new(),
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
            |native_plan| {
                let mut report = PlanningReport::new();
                report.push_rewritten(normalized.candidate);
                NativeLoweringResult {
                    plans: vec![native_plan],
                    report,
                    source_diagnostics: source_diagnostics_for_normalized_plan(&normalized),
                }
            },
        ),
        Err(rejection) => {
            let mut report = PlanningReport::new();
            report.push_rejected(rejection.candidate, rejection.reason);
            NativeLoweringResult { plans: Vec::new(), report, source_diagnostics: Vec::new() }
        },
    }
}

#[derive(Debug, Clone, Copy)]
struct NormalizedNativePlan<'a> {
    candidate: GpuCandidate,
    source_exec: &'a DataSourceExec,
    filter: &'a FilterExec,
    projection: &'a ProjectionExec,
    aggregate: &'a AggregateExec,
    sort: &'a SortExec,
    limit_fetch: usize,
}

#[derive(Debug, Clone, Copy)]
struct NativePlanRejection {
    candidate: GpuCandidate,
    reason: FallbackReason,
}

fn normalize_native_plan(plan: &dyn ExecutionPlan) -> Result<NormalizedNativePlan<'_>, NativePlanRejection> {
    if plan.as_any().downcast_ref::<SortPreservingMergeExec>().is_some() {
        return Err(NativePlanRejection {
            candidate: GpuCandidate::with_operator_count(SORT_PRESERVING_MERGE_SHAPE, plan.schema().fields().len(), 1),
            reason: FallbackReason::NativeSortPreservingMergeUnsupported,
        });
    }

    if let Some(limit) = plan.as_any().downcast_ref::<LocalLimitExec>() {
        let candidate = GpuCandidate::with_operator_count(LOCAL_LIMIT_SHAPE, plan.schema().fields().len(), 5);
        let sort = required_child::<SortExec>(limit.input().as_ref(), candidate)?;
        let fetch = sort.fetch().map_or(limit.fetch(), |sort_fetch| sort_fetch.min(limit.fetch()));
        return normalize_sort_stack(candidate, sort, Some(fetch));
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        let candidate = GpuCandidate::with_operator_count(SORT_FETCH_SHAPE, plan.schema().fields().len(), 5);
        let Some(fetch) = sort.fetch() else {
            return Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator });
        };
        return normalize_sort_stack(candidate, sort, Some(fetch));
    }

    Err(NativePlanRejection {
        candidate: GpuCandidate::with_operator_count(UNSUPPORTED_NATIVE_PLAN_SHAPE, plan.schema().fields().len(), 1),
        reason: FallbackReason::UnsupportedOperator,
    })
}

fn normalize_sort_stack<'a>(
    candidate: GpuCandidate,
    sort: &'a SortExec,
    limit_fetch: Option<usize>,
) -> Result<NormalizedNativePlan<'a>, NativePlanRejection> {
    if sort.preserve_partitioning() {
        return Err(NativePlanRejection {
            candidate,
            reason: FallbackReason::NativePartitionPreservingSortUnsupported,
        });
    }
    let Some(limit_fetch) = limit_fetch else {
        return Err(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator });
    };
    let aggregate = required_child::<AggregateExec>(sort.input().as_ref(), candidate)?;
    let projection = required_child::<ProjectionExec>(aggregate.input().as_ref(), candidate)?;
    let filter = required_child::<FilterExec>(projection.input().as_ref(), candidate)?;
    let source_exec = required_child::<DataSourceExec>(filter.input().as_ref(), candidate)?;

    Ok(NormalizedNativePlan { candidate, source_exec, filter, projection, aggregate, sort, limit_fetch })
}

fn required_child<T: 'static>(plan: &dyn ExecutionPlan, candidate: GpuCandidate) -> Result<&T, NativePlanRejection> {
    plan.as_any()
        .downcast_ref::<T>()
        .ok_or(NativePlanRejection { candidate, reason: FallbackReason::UnsupportedOperator })
}

fn lower_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    lower_slice1_plan(normalized)
}

fn source_diagnostics_for_normalized_plan(normalized: &NormalizedNativePlan<'_>) -> Vec<NativeSourceDiagnostics> {
    let Some((file_scan, parquet_source)) = normalized.source_exec.downcast_to_file_source::<ParquetSource>() else {
        return Vec::new();
    };
    vec![local_parquet_source_diagnostics(file_scan, parquet_source)]
}

fn lower_slice1_plan(normalized: &NormalizedNativePlan<'_>) -> Result<NativeQueryPlan, FallbackReason> {
    let source_exec = normalized.source_exec;
    let filter = normalized.filter;
    let projection = normalized.projection;
    let aggregate = normalized.aggregate;
    let sort = normalized.sort;
    let (file_scan, parquet_source) =
        source_exec.downcast_to_file_source::<ParquetSource>().ok_or(FallbackReason::UnsupportedOperator)?;

    validate_schema(source_exec.schema())?;
    validate_schema(filter.schema())?;
    validate_schema(projection.schema())?;
    validate_schema(aggregate.schema())?;
    validate_schema(sort.schema())?;

    validate_slice1_parquet_scan(file_scan, parquet_source)?;
    validate_aggregate(aggregate)?;
    validate_sort(sort)?;

    let read_columns = required_parquet_read_columns(file_scan, filter, projection, aggregate)?;
    let native_files = local_parquet_files(file_scan)?;
    let mut source = NativeLocalParquetSource::from_files(native_files, source_exec.schema(), read_columns);
    source.diagnostics.predicate_disposition = if parquet_source.filter().is_some() {
        NativePredicateDisposition::Rejected
    } else {
        NativePredicateDisposition::Absent
    };

    let mut builder = NativeQueryPlanBuilder::new();
    let source_node = builder.add_source(NativeSourceNode::LocalParquet(source));
    let filter_node = builder.add_filter(
        source_node,
        lower_physical_expr(filter.predicate(), &source_exec.schema())?,
        filter.schema(),
    );
    let projection_node = builder.add_projection(filter_node, lower_projection_exprs(projection)?, projection.schema());
    let aggregate_node = builder.add_aggregate(
        projection_node,
        lower_aggregate_spec(aggregate, &projection.schema())?,
        aggregate.schema(),
    );
    let sort_node = builder.add_sort(aggregate_node, lower_sort_spec(sort)?, sort.schema());
    let limit_node = builder.add_limit(sort_node, NativeLimitSpec::fetch(normalized.limit_fetch), sort.schema());

    Ok(builder.finish_with_host_sink(limit_node))
}

fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
) -> Result<(), FallbackReason> {
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        return Err(FallbackReason::NativeSourceNonLocalObjectStore);
    }
    if !file_scan.table_partition_cols().is_empty() {
        return Err(FallbackReason::NativeSourcePartitionColumns);
    }
    if file_scan.expr_adapter_factory.is_some() {
        return Err(FallbackReason::NativeSourceExprAdapter);
    }
    if file_scan.limit.is_some() {
        return Err(FallbackReason::NativeSourceScanLimit);
    }
    if file_scan.preserve_order {
        return Err(FallbackReason::NativeSourcePreserveOrder);
    }
    if !file_scan.output_ordering.is_empty() {
        return Err(FallbackReason::NativeSourceOutputOrdering);
    }
    if parquet_source.filter().is_some() {
        return Err(FallbackReason::NativeSourcePushedDownParquetPredicate);
    }
    local_parquet_files(file_scan)?;
    Ok(())
}

fn local_parquet_source_diagnostics(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
) -> NativeSourceDiagnostics {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    let known_bytes =
        files.iter().map(|file| file.object_meta.size).try_fold(0u64, |total, bytes| Some(total.saturating_add(bytes)));
    let ranged_file_count = files.iter().filter(|file| file.range.is_some()).count();
    let selected_row_groups = selected_row_groups_from_access_plans(&files);

    let mut rejected_features = Vec::new();
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        rejected_features.push(NativeRejectedSourceFeature::NonLocalObjectStore);
    }
    if !file_scan.table_partition_cols().is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::PartitionColumns);
    }
    if file_scan.expr_adapter_factory.is_some() {
        rejected_features.push(NativeRejectedSourceFeature::ExprAdapter);
    }
    if file_scan.limit.is_some() {
        rejected_features.push(NativeRejectedSourceFeature::ScanLimit);
    }
    if file_scan.preserve_order {
        rejected_features.push(NativeRejectedSourceFeature::PreserveOrder);
    }
    if !file_scan.output_ordering.is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::OutputOrdering);
    }
    if parquet_source.filter().is_some() {
        rejected_features.push(NativeRejectedSourceFeature::PushedDownParquetPredicate);
    }
    if files.is_empty() {
        rejected_features.push(NativeRejectedSourceFeature::EmptySource);
    }
    for file in &files {
        if !file.partition_values.is_empty() {
            rejected_features.push(NativeRejectedSourceFeature::PartitionValues);
        }
        if file.range.is_some() {
            rejected_features.push(NativeRejectedSourceFeature::ByteRange);
        }
        if file.ordering.is_some() {
            rejected_features.push(NativeRejectedSourceFeature::FileOrdering);
        }
        if has_row_level_selection(file) {
            rejected_features.push(NativeRejectedSourceFeature::RowSelection);
        }
    }
    rejected_features.sort_by_key(|feature| feature.as_str());
    rejected_features.dedup();

    NativeSourceDiagnostics {
        format: cudf_query_engine::SourceFormat::Parquet,
        direct_device_read: rejected_features.is_empty(),
        file_count: files.len(),
        ranged_file_count,
        known_bytes,
        selected_row_groups,
        predicate_disposition: if parquet_source.filter().is_some() {
            NativePredicateDisposition::Rejected
        } else {
            NativePredicateDisposition::Absent
        },
        rejected_features,
    }
}

fn local_parquet_files(file_scan: &FileScanConfig) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if files.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    let mut native_files = Vec::with_capacity(files.len());
    for file in files {
        if !file.partition_values.is_empty() {
            return Err(FallbackReason::NativeSourcePartitionValues);
        }
        if file.range.is_some() {
            return Err(FallbackReason::NativeSourceByteRange);
        }
        if file.ordering.is_some() {
            return Err(FallbackReason::NativeSourceFileOrdering);
        }
        let mut native_file =
            NativeLocalParquetFile::new(local_object_path_to_filesystem(file.object_meta.location.as_ref()))
                .with_file_size_bytes(file.object_meta.size);
        if let Some(row_groups) = explicit_row_group_selection(file)? {
            native_file = native_file.with_row_groups(row_groups);
        }
        native_files.push(native_file);
    }
    Ok(native_files)
}

fn explicit_row_group_selection(file: &PartitionedFile) -> Result<Option<Vec<usize>>, FallbackReason> {
    let Some(extension) = &file.extensions else {
        return Ok(None);
    };
    let Some(access_plan) = extension.downcast_ref::<ParquetAccessPlan>() else {
        return Ok(None);
    };

    let mut row_groups = Vec::new();
    for (index, access) in access_plan.inner().iter().enumerate() {
        match access {
            RowGroupAccess::Skip => {},
            RowGroupAccess::Scan => row_groups.push(index),
            RowGroupAccess::Selection(_) => return Err(FallbackReason::NativeSourceRowSelection),
        }
    }
    Ok(Some(row_groups))
}

fn selected_row_groups_from_access_plans(files: &[&PartitionedFile]) -> Option<usize> {
    if files.is_empty() {
        return None;
    }
    files.iter().try_fold(0usize, |total, file| {
        explicit_row_group_selection_count(file).map(|count| total.saturating_add(count))
    })
}

fn explicit_row_group_selection_count(file: &PartitionedFile) -> Option<usize> {
    let extension = file.extensions.as_ref()?;
    let access_plan = extension.downcast_ref::<ParquetAccessPlan>()?;
    Some(access_plan.inner().iter().filter(|access| !matches!(access, RowGroupAccess::Skip)).count())
}

fn has_row_level_selection(file: &PartitionedFile) -> bool {
    let Some(extension) = &file.extensions else {
        return false;
    };
    let Some(access_plan) = extension.downcast_ref::<ParquetAccessPlan>() else {
        return false;
    };
    access_plan.inner().iter().any(|access| matches!(access, RowGroupAccess::Selection(_)))
}

fn validate_schema(schema: SchemaRef) -> Result<(), FallbackReason> {
    for field in schema.fields() {
        if !is_supported_native_type(field.data_type()) {
            return Err(FallbackReason::UnsupportedType);
        }
    }
    Ok(())
}

fn is_supported_native_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn validate_aggregate(aggregate: &AggregateExec) -> Result<(), FallbackReason> {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::SinglePartitioned) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    let group_by = aggregate.group_expr();
    if !group_by.is_single() || group_by.expr().is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if group_by.groups().len() != 1 || group_by.groups()[0].iter().any(|is_null| *is_null) {
        return Err(FallbackReason::UnsupportedOperator);
    }
    if aggregate.filter_expr().iter().any(Option::is_some) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    for aggregate_expr in aggregate.aggr_expr() {
        if aggregate_expr.is_distinct() || !aggregate_expr.order_bys().is_empty() {
            return Err(FallbackReason::UnsupportedExpression);
        }
        if !aggregate_expr.fun().name().eq_ignore_ascii_case("sum") {
            return Err(FallbackReason::UnsupportedOperator);
        }
        let expressions = aggregate_expr.expressions();
        let [arg] = expressions.as_slice() else {
            return Err(FallbackReason::UnsupportedExpression);
        };
        if arg.as_any().downcast_ref::<Column>().is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }
    for (group_expr, _alias) in group_by.expr() {
        if group_expr.as_any().downcast_ref::<Column>().is_none() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }
    Ok(())
}

fn validate_sort(sort: &SortExec) -> Result<(), FallbackReason> {
    if sort.expr().is_empty() {
        return Err(FallbackReason::UnsupportedExpression);
    }
    if sort.expr().iter().any(|sort_expr| sort_expr.expr.as_any().downcast_ref::<Column>().is_none()) {
        return Err(FallbackReason::UnsupportedExpression);
    }
    Ok(())
}

fn lower_projection_exprs(projection: &ProjectionExec) -> Result<Vec<NativeExpr>, FallbackReason> {
    projection
        .expr()
        .iter()
        .map(|projection_expr| lower_projection_expr(projection_expr, &projection.input().schema()))
        .collect()
}

fn lower_projection_expr(
    projection_expr: &ProjectionExpr,
    input_schema: &SchemaRef,
) -> Result<NativeExpr, FallbackReason> {
    let expr = lower_physical_expr(&projection_expr.expr, input_schema)?;
    Ok(alias_if_needed(expr, projection_expr.alias.as_str()))
}

fn lower_aggregate_spec(
    aggregate: &AggregateExec,
    input_schema: &SchemaRef,
) -> Result<NativeAggregateSpec, FallbackReason> {
    let group_exprs = aggregate
        .group_expr()
        .expr()
        .iter()
        .map(|(expr, alias)| {
            let native_expr = lower_physical_expr(expr, input_schema)?;
            Ok(alias_if_needed(native_expr, alias))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let aggr_exprs = aggregate
        .aggr_expr()
        .iter()
        .map(|expr| {
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Err(FallbackReason::UnsupportedExpression);
            };
            Ok(NativeAggregateExpr::new(
                NativeAggregateFunction::Sum,
                lower_physical_expr(arg, input_schema)?,
                expr.name(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(NativeAggregateSpec::new(group_exprs, aggr_exprs))
}

fn lower_sort_spec(sort: &SortExec) -> Result<NativeSortSpec, FallbackReason> {
    let expressions = sort.expr().iter().map(lower_sort_expr).collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

fn lower_sort_expr(sort_expr: &PhysicalSortExpr) -> Result<NativeSortExpr, FallbackReason> {
    let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(NativeSortExpr::new(
        NativeExpr::column(column.name()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}

fn lower_physical_expr(expr: &Arc<dyn PhysicalExpr>, input_schema: &SchemaRef) -> Result<NativeExpr, FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        validate_column_type(column, input_schema)?;
        return Ok(NativeExpr::column(column.name()));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_physical_expr(binary.left(), input_schema)?;
        let right = lower_physical_expr(binary.right(), input_schema)?;
        let op = lower_binary_operator(binary.op())?;
        return Ok(NativeExpr::binary(left, op, right));
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn validate_column_type(column: &Column, input_schema: &SchemaRef) -> Result<(), FallbackReason> {
    let field = input_schema.field_with_name(column.name()).map_err(|_error| FallbackReason::UnsupportedExpression)?;
    if is_supported_native_type(field.data_type()) { Ok(()) } else { Err(FallbackReason::UnsupportedType) }
}

fn lower_literal(value: &ScalarValue) -> Result<NativeExpr, FallbackReason> {
    let literal = match value {
        ScalarValue::Boolean(Some(value)) => NativeLiteralValue::Boolean(*value),
        ScalarValue::Int8(Some(value)) => NativeLiteralValue::Int8(*value),
        ScalarValue::Int16(Some(value)) => NativeLiteralValue::Int16(*value),
        ScalarValue::Int32(Some(value)) => NativeLiteralValue::Int32(*value),
        ScalarValue::Int64(Some(value)) => NativeLiteralValue::Int64(*value),
        ScalarValue::UInt8(Some(value)) => NativeLiteralValue::UInt8(*value),
        ScalarValue::UInt16(Some(value)) => NativeLiteralValue::UInt16(*value),
        ScalarValue::UInt32(Some(value)) => NativeLiteralValue::UInt32(*value),
        ScalarValue::UInt64(Some(value)) => NativeLiteralValue::UInt64(*value),
        ScalarValue::Float32(Some(value)) => NativeLiteralValue::Float32(*value),
        ScalarValue::Float64(Some(value)) => NativeLiteralValue::Float64(*value),
        _ => return Err(FallbackReason::UnsupportedExpression),
    };
    Ok(NativeExpr::literal(literal))
}

fn lower_binary_operator(op: &Operator) -> Result<NativeBinaryOp, FallbackReason> {
    match op {
        Operator::Plus => Ok(NativeBinaryOp::Add),
        Operator::Minus => Ok(NativeBinaryOp::Subtract),
        Operator::Multiply => Ok(NativeBinaryOp::Multiply),
        Operator::Divide => Ok(NativeBinaryOp::Divide),
        Operator::Eq => Ok(NativeBinaryOp::Eq),
        Operator::NotEq => Ok(NativeBinaryOp::NotEq),
        Operator::Lt => Ok(NativeBinaryOp::Lt),
        Operator::LtEq => Ok(NativeBinaryOp::LtEq),
        Operator::Gt => Ok(NativeBinaryOp::Gt),
        Operator::GtEq => Ok(NativeBinaryOp::GtEq),
        Operator::And => Ok(NativeBinaryOp::And),
        Operator::Or => Ok(NativeBinaryOp::Or),
        _ => Err(FallbackReason::UnsupportedExpression),
    }
}

fn alias_if_needed(expr: NativeExpr, alias: &str) -> NativeExpr {
    match &expr {
        NativeExpr::Column(name) if name == alias => expr,
        _ => expr.alias(alias),
    }
}

fn required_parquet_read_columns(
    file_scan: &FileScanConfig,
    filter: &FilterExec,
    projection: &ProjectionExec,
    aggregate: &AggregateExec,
) -> Result<Vec<String>, FallbackReason> {
    let file_schema = file_scan.file_schema();
    let mut required = BTreeSet::new();

    if let Some(projection) = file_scan.file_source().projection() {
        for index in projection.column_indices() {
            if index >= file_schema.fields().len() {
                return Err(FallbackReason::NativeSourcePartitionColumns);
            }
            required.insert(file_schema.field(index).name().clone());
        }
    }

    collect_required_columns(filter.predicate(), &mut required)?;
    for projection_expr in projection.expr() {
        collect_required_columns(&projection_expr.expr, &mut required)?;
    }
    for (group_expr, _alias) in aggregate.group_expr().expr() {
        collect_required_columns(group_expr, &mut required)?;
    }

    for column in &required {
        if file_schema.index_of(column).is_err() {
            return Err(FallbackReason::UnsupportedExpression);
        }
    }

    Ok(required.into_iter().collect())
}

fn collect_required_columns(
    expr: &Arc<dyn PhysicalExpr>,
    required: &mut BTreeSet<String>,
) -> Result<(), FallbackReason> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        required.insert(column.name().to_string());
        return Ok(());
    }
    if expr.as_any().downcast_ref::<Literal>().is_some() {
        return Ok(());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        collect_required_columns(binary.left(), required)?;
        collect_required_columns(binary.right(), required)?;
        return Ok(());
    }
    Err(FallbackReason::UnsupportedExpression)
}

fn local_object_path_to_filesystem(location: &str) -> PathBuf {
    PathBuf::from("/").join(location)
}
