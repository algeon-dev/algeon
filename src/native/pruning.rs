use std::{path::PathBuf, sync::Arc};

use arrow::datatypes::SchemaRef;
use cudf_query_engine::PredicateDisposition as NativePredicateDisposition;
use datafusion::{
    common::ScalarValue,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileScanConfig, FileSource, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess},
        },
    },
    logical_expr::Operator,
    parquet::file::{
        reader::{FileReader, SerializedFileReader},
        statistics::Statistics,
    },
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, IsNotNullExpr, IsNullExpr,
            LikeExpr, Literal, NotExpr,
        },
    },
    physical_plan::filter::FilterExec,
};

use super::validate::validate_physical_expr;
use crate::policy::FallbackReason;

#[derive(Debug, Clone, Copy)]
pub(super) struct CoalescedLocalParquetFile<'a> {
    pub(super) file: &'a PartitionedFile,
}

#[derive(Debug)]
struct FileCoverage<'a> {
    location: String,
    size: u64,
    partition_values: &'a [ScalarValue],
    representative: &'a PartitionedFile,
    entry_count: usize,
    saw_unranged: bool,
    ranges: Vec<(i64, i64)>,
}

pub(super) fn coalesced_local_parquet_scan_files<'a>(
    files: &[&'a PartitionedFile],
) -> Result<Vec<CoalescedLocalParquetFile<'a>>, FallbackReason> {
    let mut coverages = Vec::<FileCoverage<'a>>::new();
    for file in files {
        if file.ordering.is_some() {
            return Err(FallbackReason::NativeSourceFileOrdering);
        }
        if let Some(range) = &file.range {
            if file.extensions.is_some()
                || range.start < 0
                || range.end < range.start
                || u64::try_from(range.end).map_err(|_error| FallbackReason::NativeSourceByteRange)?
                    > file.object_meta.size
            {
                return Err(FallbackReason::NativeSourceByteRange);
            }
        }

        let location = file.object_meta.location.as_ref().to_string();
        let coverage_index = if let Some(index) = coverages.iter().position(|coverage| coverage.location == location) {
            index
        } else {
            coverages.push(FileCoverage {
                location: location.clone(),
                size: file.object_meta.size,
                partition_values: &file.partition_values,
                representative: file,
                entry_count: 0,
                saw_unranged: false,
                ranges: Vec::new(),
            });
            coverages.len() - 1
        };
        let coverage = &mut coverages[coverage_index];
        if coverage.size != file.object_meta.size {
            return Err(FallbackReason::NativeSourceByteRange);
        }
        if coverage.partition_values != file.partition_values.as_slice() {
            return Err(FallbackReason::NativeSourcePartitionValues);
        }
        coverage.entry_count += 1;
        if let Some(range) = &file.range {
            coverage.ranges.push((range.start, range.end));
        } else {
            coverage.saw_unranged = true;
        }
    }

    for coverage in &mut coverages {
        if coverage.saw_unranged {
            if coverage.entry_count != 1 || !coverage.ranges.is_empty() {
                return Err(FallbackReason::NativeSourceByteRange);
            }
            continue;
        }

        let size = i64::try_from(coverage.size).map_err(|_error| FallbackReason::NativeSourceByteRange)?;
        coverage.ranges.sort_unstable();
        let mut next_start = 0i64;
        for (start, end) in &coverage.ranges {
            if *start != next_start || *end <= *start {
                return Err(FallbackReason::NativeSourceByteRange);
            }
            next_start = *end;
        }
        if next_start != size {
            return Err(FallbackReason::NativeSourceByteRange);
        }
    }

    Ok(coverages.into_iter().map(|coverage| CoalescedLocalParquetFile { file: coverage.representative }).collect())
}

pub(super) fn parquet_predicate_disposition(
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: &SchemaRef,
) -> NativePredicateDisposition {
    if parquet_source.filter().is_none() {
        return NativePredicateDisposition::Absent;
    }
    if validate_pruning_only_parquet_predicate(parquet_source, filter, source_schema).is_ok() {
        NativePredicateDisposition::PruningOnly
    } else {
        NativePredicateDisposition::Rejected
    }
}

pub(super) fn validate_pruning_only_parquet_predicate(
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: &SchemaRef,
) -> Result<(), FallbackReason> {
    let Some(scan_predicate) = parquet_source.filter() else {
        return Ok(());
    };
    let Some(filter) = filter else {
        // Default SQL can leave dynamic pruning predicates on the scan. They are optimizer hints,
        // not row-level semantics, so native execution can read the source fully.
        return if scan_predicate_is_dynamic_filter_hint(scan_predicate.as_ref()) {
            Ok(())
        } else {
            Err(FallbackReason::NativeSourcePushedDownParquetPredicate)
        };
    };
    validate_physical_expr(filter.predicate(), source_schema)?;
    if scan_predicate_matches_filter(scan_predicate.as_ref(), filter.predicate().as_ref()) {
        Ok(())
    } else {
        Err(FallbackReason::NativeSourcePushedDownParquetPredicate)
    }
}

fn scan_predicate_is_dynamic_filter_hint(expr: &dyn PhysicalExpr) -> bool {
    if expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>().is_some() {
        return true;
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        return matches!(binary.op(), Operator::And | Operator::Or)
            && scan_predicate_is_dynamic_filter_hint(binary.left().as_ref())
            && scan_predicate_is_dynamic_filter_hint(binary.right().as_ref());
    }
    false
}

fn scan_predicate_matches_filter(scan_predicate: &dyn PhysicalExpr, filter_predicate: &dyn PhysicalExpr) -> bool {
    if physical_expr_eq(scan_predicate, filter_predicate) {
        return true;
    }
    let Some(scan_binary) = scan_predicate.as_any().downcast_ref::<BinaryExpr>() else {
        return false;
    };
    if !matches!(scan_binary.op(), Operator::And) {
        return false;
    }
    if scan_predicate_is_dynamic_filter_hint(scan_binary.left().as_ref()) {
        return scan_predicate_matches_filter(scan_binary.right().as_ref(), filter_predicate);
    }
    if scan_predicate_is_dynamic_filter_hint(scan_binary.right().as_ref()) {
        return scan_predicate_matches_filter(scan_binary.left().as_ref(), filter_predicate);
    }
    if let Some(filter_binary) = filter_predicate.as_any().downcast_ref::<BinaryExpr>()
        && matches!(filter_binary.op(), Operator::And)
    {
        return scan_predicate_matches_filter(scan_binary.left().as_ref(), filter_binary.left().as_ref())
            && scan_predicate_matches_filter(scan_binary.right().as_ref(), filter_binary.right().as_ref());
    }
    false
}

fn physical_expr_eq(left: &dyn PhysicalExpr, right: &dyn PhysicalExpr) -> bool {
    if let (Some(left), Some(right)) = (left.as_any().downcast_ref::<Column>(), right.as_any().downcast_ref::<Column>())
    {
        return left.name() == right.name();
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<Literal>(), right.as_any().downcast_ref::<Literal>())
    {
        return left.value() == right.value();
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<BinaryExpr>(), right.as_any().downcast_ref::<BinaryExpr>())
    {
        return left.op() == right.op()
            && physical_expr_eq(left.left().as_ref(), right.left().as_ref())
            && physical_expr_eq(left.right().as_ref(), right.right().as_ref());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<NotExpr>(), right.as_any().downcast_ref::<NotExpr>())
    {
        return physical_expr_eq(left.arg().as_ref(), right.arg().as_ref());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<IsNullExpr>(), right.as_any().downcast_ref::<IsNullExpr>())
    {
        return physical_expr_eq(left.arg().as_ref(), right.arg().as_ref());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<IsNotNullExpr>(), right.as_any().downcast_ref::<IsNotNullExpr>())
    {
        return physical_expr_eq(left.arg().as_ref(), right.arg().as_ref());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<CastExpr>(), right.as_any().downcast_ref::<CastExpr>())
    {
        return left.cast_type() == right.cast_type() && physical_expr_eq(left.expr().as_ref(), right.expr().as_ref());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<LikeExpr>(), right.as_any().downcast_ref::<LikeExpr>())
    {
        return left.negated() == right.negated()
            && left.case_insensitive() == right.case_insensitive()
            && physical_expr_eq(left.expr().as_ref(), right.expr().as_ref())
            && physical_expr_eq(left.pattern().as_ref(), right.pattern().as_ref());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<InListExpr>(), right.as_any().downcast_ref::<InListExpr>())
    {
        return left.negated() == right.negated()
            && physical_expr_eq(left.expr().as_ref(), right.expr().as_ref())
            && physical_expr_slices_eq(left.list(), right.list());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<CaseExpr>(), right.as_any().downcast_ref::<CaseExpr>())
    {
        return optional_physical_expr_eq(left.expr(), right.expr())
            && left.when_then_expr().len() == right.when_then_expr().len()
            && left.when_then_expr().iter().zip(right.when_then_expr()).all(|(left, right)| {
                physical_expr_eq(left.0.as_ref(), right.0.as_ref())
                    && physical_expr_eq(left.1.as_ref(), right.1.as_ref())
            })
            && optional_physical_expr_eq(left.else_expr(), right.else_expr());
    }
    if let (Some(left), Some(right)) =
        (left.as_any().downcast_ref::<ScalarFunctionExpr>(), right.as_any().downcast_ref::<ScalarFunctionExpr>())
    {
        return left.name().eq_ignore_ascii_case(right.name()) && physical_expr_slices_eq(left.args(), right.args());
    }
    false
}

fn optional_physical_expr_eq(left: Option<&Arc<dyn PhysicalExpr>>, right: Option<&Arc<dyn PhysicalExpr>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => physical_expr_eq(left.as_ref(), right.as_ref()),
        (None, None) => true,
        _ => false,
    }
}

fn physical_expr_slices_eq(left: &[Arc<dyn PhysicalExpr>], right: &[Arc<dyn PhysicalExpr>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| physical_expr_eq(left.as_ref(), right.as_ref()))
}

pub(super) fn metadata_row_group_selections(
    files: &[&PartitionedFile],
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: &SchemaRef,
) -> Option<Vec<Vec<usize>>> {
    let coalesced_files = coalesced_local_parquet_scan_files(files).ok()?;
    let files = coalesced_files.iter().map(|file| file.file).collect::<Vec<_>>();
    if files.is_empty()
        || !file_scan.object_store_url.as_str().starts_with("file://")
        || file_scan.expr_adapter_factory.is_some()
        || file_scan.limit.is_some()
        || file_scan.preserve_order
        || !file_scan.output_ordering.is_empty()
        || files.iter().any(|file| file.extensions.is_some() || file.ordering.is_some())
    {
        return None;
    }
    if parquet_source.filter().is_some()
        && validate_pruning_only_parquet_predicate(parquet_source, filter, source_schema).is_err()
    {
        return None;
    }
    let predicate = row_group_pruning_predicate(filter?.predicate(), file_scan.file_schema())?;
    let selections =
        files.iter().map(|file| metadata_row_group_selection(file, predicate)).collect::<Option<Vec<_>>>()?;
    let selected_row_groups = selections.iter().map(|selection| selection.row_groups.len()).sum::<usize>();
    let total_row_groups = selections.iter().map(|selection| selection.total_row_groups).sum::<usize>();
    if selected_row_groups > 0 && selected_row_groups < total_row_groups {
        Some(selections.into_iter().map(|selection| selection.row_groups).collect())
    } else {
        None
    }
}

#[derive(Debug, Clone)]
struct MetadataRowGroupSelection {
    row_groups: Vec<usize>,
    total_row_groups: usize,
}

fn metadata_row_group_selection(
    file: &PartitionedFile,
    predicate: RowGroupPruningPredicate,
) -> Option<MetadataRowGroupSelection> {
    let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
    let reader = SerializedFileReader::try_from(path.as_path()).ok()?;
    let metadata = reader.metadata();
    let total_row_groups = metadata.num_row_groups();
    if total_row_groups == 0 {
        return None;
    }
    let row_groups = metadata
        .row_groups()
        .iter()
        .enumerate()
        .map(|(index, row_group)| {
            let statistics = row_group.columns().get(predicate.column_index)?.statistics()?;
            row_group_may_satisfy_predicate(statistics, predicate).map(|may_satisfy| may_satisfy.then_some(index))
        })
        .collect::<Option<Vec<_>>>()
        .map(|row_groups| row_groups.into_iter().flatten().collect())?;
    Some(MetadataRowGroupSelection { row_groups, total_row_groups })
}

#[derive(Debug, Clone, Copy)]
struct RowGroupPruningPredicate {
    column_index: usize,
    op: Operator,
    literal: NumericLiteral,
}

#[derive(Debug, Clone, Copy)]
enum NumericLiteral {
    Int64(i64),
    Float64(f64),
}

fn row_group_pruning_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    file_schema: &SchemaRef,
) -> Option<RowGroupPruningPredicate> {
    let binary = expr.as_any().downcast_ref::<BinaryExpr>()?;
    if let Some((column, literal, op)) = column_literal_predicate(binary.left(), binary.op(), binary.right()) {
        return Some(RowGroupPruningPredicate {
            column_index: file_schema.index_of(column.name()).ok()?,
            op,
            literal: numeric_literal(literal.value())?,
        });
    }
    let inverted_op = invert_comparison_operator(binary.op())?;
    let (column, literal, op) = column_literal_predicate(binary.right(), &inverted_op, binary.left())?;
    Some(RowGroupPruningPredicate {
        column_index: file_schema.index_of(column.name()).ok()?,
        op,
        literal: numeric_literal(literal.value())?,
    })
}

fn column_literal_predicate<'a>(
    left: &'a Arc<dyn PhysicalExpr>,
    op: &Operator,
    right: &'a Arc<dyn PhysicalExpr>,
) -> Option<(&'a Column, &'a Literal, Operator)> {
    let column = left.as_any().downcast_ref::<Column>()?;
    let literal = right.as_any().downcast_ref::<Literal>()?;
    if matches!(op, Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq) {
        Some((column, literal, *op))
    } else {
        None
    }
}

fn invert_comparison_operator(op: &Operator) -> Option<Operator> {
    match op {
        Operator::Eq => Some(Operator::Eq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

fn numeric_literal(value: &ScalarValue) -> Option<NumericLiteral> {
    match value {
        ScalarValue::Int8(Some(value)) => Some(NumericLiteral::Int64(i64::from(*value))),
        ScalarValue::Int16(Some(value)) => Some(NumericLiteral::Int64(i64::from(*value))),
        ScalarValue::Int32(Some(value)) => Some(NumericLiteral::Int64(i64::from(*value))),
        ScalarValue::Int64(Some(value)) => Some(NumericLiteral::Int64(*value)),
        ScalarValue::Float32(Some(value)) => Some(NumericLiteral::Float64(f64::from(*value))),
        ScalarValue::Float64(Some(value)) => Some(NumericLiteral::Float64(*value)),
        _ => None,
    }
}

fn row_group_may_satisfy_predicate(statistics: &Statistics, predicate: RowGroupPruningPredicate) -> Option<bool> {
    match statistics {
        Statistics::Int32(stats) => {
            let min = i64::from(*stats.min_opt()?);
            let max = i64::from(*stats.max_opt()?);
            let NumericLiteral::Int64(literal) = predicate.literal else {
                return None;
            };
            Some(int_range_may_satisfy(min, max, predicate.op, literal))
        },
        Statistics::Int64(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let NumericLiteral::Int64(literal) = predicate.literal else {
                return None;
            };
            Some(int_range_may_satisfy(min, max, predicate.op, literal))
        },
        Statistics::Float(stats) => {
            let min = f64::from(*stats.min_opt()?);
            let max = f64::from(*stats.max_opt()?);
            let NumericLiteral::Float64(literal) = predicate.literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, predicate.op, literal))
        },
        Statistics::Double(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let NumericLiteral::Float64(literal) = predicate.literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, predicate.op, literal))
        },
        _ => None,
    }
}

fn int_range_may_satisfy(min: i64, max: i64, op: Operator, literal: i64) -> bool {
    match op {
        Operator::Eq => min <= literal && literal <= max,
        Operator::Lt => min < literal,
        Operator::LtEq => min <= literal,
        Operator::Gt => max > literal,
        Operator::GtEq => max >= literal,
        _ => true,
    }
}

fn float_range_may_satisfy(min: f64, max: f64, op: Operator, literal: f64) -> bool {
    if min.is_nan() || max.is_nan() || literal.is_nan() {
        return true;
    }
    match op {
        Operator::Eq => min <= literal && literal <= max,
        Operator::Lt => min < literal,
        Operator::LtEq => min <= literal,
        Operator::Gt => max > literal,
        Operator::GtEq => max >= literal,
        _ => true,
    }
}

pub(super) fn explicit_row_group_selection(file: &PartitionedFile) -> Result<Option<Vec<usize>>, FallbackReason> {
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

pub(super) fn selected_row_groups_from_access_plans(files: &[&PartitionedFile]) -> Option<usize> {
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

pub(super) fn has_row_level_selection(file: &PartitionedFile) -> bool {
    let Some(extension) = &file.extensions else {
        return false;
    };
    let Some(access_plan) = extension.downcast_ref::<ParquetAccessPlan>() else {
        return false;
    };
    access_plan.inner().iter().any(|access| matches!(access, RowGroupAccess::Selection(_)))
}

pub(super) fn local_object_path_to_filesystem(location: &str) -> PathBuf {
    PathBuf::from("/").join(location)
}
