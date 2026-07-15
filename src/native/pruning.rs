use std::{path::PathBuf, sync::Arc};

use arrow::datatypes::SchemaRef;
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
        metadata::{ColumnChunkMetaData, RowGroupMetaData},
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
use nexus_query_engine::source::PredicateDisposition as NativePredicateDisposition;

use super::expression_contract::validate_physical_expr;
use crate::policy::FallbackReason;

#[path = "pruning/row_groups.rs"]
mod row_groups;

pub(super) use row_groups::*;

/// Partitioned file chosen for native local Parquet scan after range coalescing.
#[derive(Debug, Clone)]
pub(super) struct CoalescedLocalParquetFile<'a> {
    pub(super) file: &'a PartitionedFile,
    pub(super) byte_range_representation: LocalParquetByteRangeRepresentation,
    pub(super) byte_range_row_groups: Option<Vec<usize>>,
}

/// Meaning assigned to DataFusion byte ranges after pruning analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalParquetByteRangeRepresentation {
    WholeFile,
    RowGroupSelection,
}

#[derive(Debug)]
struct FileCoverage<'a> {
    location: String,
    size: u64,
    partition_values: &'a [ScalarValue],
    representative: &'a PartitionedFile,
    entry_count: usize,
    saw_unranged: bool,
    saw_ranged_extension: bool,
    ranges: Vec<RangeCoverage<'a>>,
    byte_range_representation: LocalParquetByteRangeRepresentation,
    byte_range_row_groups: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
struct RangeCoverage<'a> {
    start: i64,
    end: i64,
    file: &'a PartitionedFile,
}

pub(super) fn coalesced_local_parquet_scan_files<'a>(
    files: &[&'a PartitionedFile],
) -> Result<Vec<CoalescedLocalParquetFile<'a>>, FallbackReason> {
    let mut coverages = Vec::<FileCoverage<'a>>::new();
    for file in files {
        if let Some(range) = &file.range
            && (range.start < 0
                || range.end <= range.start
                || u64::try_from(range.end).map_err(|_error| FallbackReason::NativeSourceByteRange)?
                    > file.object_meta.size
                || !range_extensions_are_supported(file)?)
        {
            return Err(FallbackReason::NativeSourceByteRange);
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
                saw_ranged_extension: false,
                ranges: Vec::new(),
                byte_range_representation: LocalParquetByteRangeRepresentation::WholeFile,
                byte_range_row_groups: None,
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
            coverage.ranges.push(RangeCoverage { start: range.start, end: range.end, file });
            if !file.extensions.is_empty() {
                coverage.saw_ranged_extension = true;
            }
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

        if ranges_cover_full_file(coverage)? && !coverage.saw_ranged_extension {
            continue;
        }

        let row_groups = row_groups_for_byte_ranges(coverage)?;
        coverage.byte_range_representation = LocalParquetByteRangeRepresentation::RowGroupSelection;
        coverage.byte_range_row_groups = Some(row_groups);
    }

    Ok(coverages
        .into_iter()
        .map(|coverage| CoalescedLocalParquetFile {
            file: coverage.representative,
            byte_range_representation: coverage.byte_range_representation,
            byte_range_row_groups: coverage.byte_range_row_groups,
        })
        .collect())
}

fn ranges_cover_full_file(coverage: &mut FileCoverage<'_>) -> Result<bool, FallbackReason> {
    let size = i64::try_from(coverage.size).map_err(|_error| FallbackReason::NativeSourceByteRange)?;
    coverage.ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut next_start = 0i64;
    for range in &coverage.ranges {
        if range.start != next_start || range.end <= range.start {
            return Ok(false);
        }
        next_start = range.end;
    }
    Ok(next_start == size)
}

fn row_groups_for_byte_ranges(coverage: &mut FileCoverage<'_>) -> Result<Vec<usize>, FallbackReason> {
    if !ranges_are_non_overlapping(coverage) {
        return Err(FallbackReason::NativeSourceByteRange);
    }
    let path = local_object_path_to_filesystem(coverage.location.as_str());
    let reader =
        SerializedFileReader::try_from(path.as_path()).map_err(|_error| FallbackReason::NativeSourceByteRange)?;
    let metadata = reader.metadata();
    let row_group_count = metadata.num_row_groups();
    let ranges = coverage
        .ranges
        .iter()
        .map(|range| {
            u64::try_from(range.start).map_err(|_error| FallbackReason::NativeSourceByteRange)?;
            u64::try_from(range.end).map_err(|_error| FallbackReason::NativeSourceByteRange)?;
            if let Some(access_plan) = range.file.extensions.get::<ParquetAccessPlan>()
                && access_plan.len() != row_group_count
            {
                return Err(FallbackReason::NativeSourceByteRange);
            }
            Ok(range)
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let mut row_groups = Vec::new();
    for (index, row_group) in metadata.row_groups().iter().enumerate() {
        let Some(start) = row_group_range_pruning_offset(row_group) else {
            return Err(FallbackReason::NativeSourceByteRange);
        };
        let mut selected = false;
        for range in &ranges {
            if range.start <= start && start < range.end && range_access_plan_selects_row_group(range, index)? {
                selected = true;
                break;
            }
        }
        if selected {
            row_groups.push(index);
        }
    }
    Ok(row_groups)
}

fn ranges_are_non_overlapping(coverage: &mut FileCoverage<'_>) -> bool {
    coverage.ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut previous_end = None;
    for range in &coverage.ranges {
        if previous_end.is_some_and(|end| range.start < end) {
            return false;
        }
        previous_end = Some(range.end);
    }
    true
}

fn range_access_plan_selects_row_group(range: &RangeCoverage<'_>, index: usize) -> Result<bool, FallbackReason> {
    let Some(access_plan) = range.file.extensions.get::<ParquetAccessPlan>() else {
        return Ok(true);
    };
    let Some(access) = access_plan.inner().get(index) else {
        return Err(FallbackReason::NativeSourceByteRange);
    };
    match access {
        RowGroupAccess::Skip => Ok(false),
        RowGroupAccess::Scan => Ok(true),
        RowGroupAccess::Selection(_) => Err(FallbackReason::NativeSourceRowSelection),
    }
}

fn row_group_range_pruning_offset(row_group: &RowGroupMetaData) -> Option<i64> {
    let column = row_group.columns().first()?;
    Some(first_dictionary_or_data_page_offset(column))
}

fn first_dictionary_or_data_page_offset(column: &ColumnChunkMetaData) -> i64 {
    column.dictionary_page_offset().unwrap_or_else(|| column.data_page_offset())
}

fn range_extensions_are_supported(file: &PartitionedFile) -> Result<bool, FallbackReason> {
    if file.extensions.is_empty() {
        return Ok(true);
    }
    Ok(file.extensions.len() == 1 && file.extensions.get::<ParquetAccessPlan>().is_some())
}

pub(super) fn parquet_predicate_disposition(
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: &SchemaRef,
    reader_pushdown: bool,
) -> NativePredicateDisposition {
    if reader_pushdown {
        return NativePredicateDisposition::ReaderPushdown;
    }
    if parquet_source.filter().is_none() {
        return if filter.is_some() {
            NativePredicateDisposition::NativePostScanOnly
        } else {
            NativePredicateDisposition::Absent
        };
    }
    if validate_pruning_only_parquet_predicate(parquet_source, filter, source_schema).is_ok() {
        NativePredicateDisposition::PruningOnly
    } else {
        NativePredicateDisposition::NativePostScanOnly
    }
}

pub(super) fn validate_pruning_only_parquet_predicate(
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    _source_schema: &SchemaRef,
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
    let filter_schema = filter.input().schema();
    validate_physical_expr(filter.predicate(), &filter_schema)?;
    if scan_predicate_matches_filter(scan_predicate.as_ref(), filter.predicate().as_ref()) {
        Ok(())
    } else {
        Err(FallbackReason::NativeSourcePushedDownParquetPredicate)
    }
}

pub(super) fn scan_predicate_exactly_matches_filter(
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
) -> bool {
    let (Some(scan_predicate), Some(filter)) = (parquet_source.filter(), filter) else {
        return false;
    };
    physical_expr_eq(scan_predicate.as_ref(), filter.predicate().as_ref())
}

pub(in crate::native) fn scan_predicate_is_dynamic_filter_hint(expr: &dyn PhysicalExpr) -> bool {
    if expr.downcast_ref::<DynamicFilterPhysicalExpr>().is_some() {
        return true;
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return matches!(binary.op(), Operator::And | Operator::Or)
            && scan_predicate_is_dynamic_filter_hint(binary.left().as_ref())
            && scan_predicate_is_dynamic_filter_hint(binary.right().as_ref());
    }
    false
}

pub(in crate::native) fn scan_predicate_without_dynamic_filter_hints(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<Arc<dyn PhysicalExpr>> {
    if scan_predicate_is_dynamic_filter_hint(expr.as_ref()) {
        return None;
    }
    let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
        return Some(Arc::clone(expr));
    };
    if !matches!(binary.op(), Operator::And) {
        return Some(Arc::clone(expr));
    }
    match (
        scan_predicate_without_dynamic_filter_hints(binary.left()),
        scan_predicate_without_dynamic_filter_hints(binary.right()),
    ) {
        (Some(left), Some(right)) => Some(Arc::new(BinaryExpr::new(left, Operator::And, right))),
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (None, None) => None,
    }
}

fn scan_predicate_matches_filter(scan_predicate: &dyn PhysicalExpr, filter_predicate: &dyn PhysicalExpr) -> bool {
    if scan_predicate_is_dynamic_filter_hint(scan_predicate) {
        return true;
    }
    if physical_expr_eq(scan_predicate, filter_predicate) {
        return true;
    }
    let Some(scan_binary) = scan_predicate.downcast_ref::<BinaryExpr>() else {
        return filter_conjunction_contains(filter_predicate, scan_predicate);
    };
    if !matches!(scan_binary.op(), Operator::And) {
        return filter_conjunction_contains(filter_predicate, scan_predicate);
    }
    if scan_predicate_is_dynamic_filter_hint(scan_binary.left().as_ref()) {
        return scan_predicate_matches_filter(scan_binary.right().as_ref(), filter_predicate);
    }
    if scan_predicate_is_dynamic_filter_hint(scan_binary.right().as_ref()) {
        return scan_predicate_matches_filter(scan_binary.left().as_ref(), filter_predicate);
    }
    scan_predicate_matches_filter(scan_binary.left().as_ref(), filter_predicate)
        && scan_predicate_matches_filter(scan_binary.right().as_ref(), filter_predicate)
}

fn filter_conjunction_contains(filter_predicate: &dyn PhysicalExpr, scan_predicate: &dyn PhysicalExpr) -> bool {
    if physical_expr_eq(filter_predicate, scan_predicate) {
        return true;
    }
    let Some(filter_binary) = filter_predicate.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    matches!(filter_binary.op(), Operator::And)
        && (filter_conjunction_contains(filter_binary.left().as_ref(), scan_predicate)
            || filter_conjunction_contains(filter_binary.right().as_ref(), scan_predicate))
}

fn physical_expr_eq(left: &dyn PhysicalExpr, right: &dyn PhysicalExpr) -> bool {
    if let (Some(left), Some(right)) = (left.downcast_ref::<Column>(), right.downcast_ref::<Column>()) {
        return left.name() == right.name();
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<Literal>(), right.downcast_ref::<Literal>()) {
        return left.value() == right.value();
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<BinaryExpr>(), right.downcast_ref::<BinaryExpr>()) {
        return left.op() == right.op()
            && physical_expr_eq(left.left().as_ref(), right.left().as_ref())
            && physical_expr_eq(left.right().as_ref(), right.right().as_ref());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<NotExpr>(), right.downcast_ref::<NotExpr>()) {
        return physical_expr_eq(left.arg().as_ref(), right.arg().as_ref());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<IsNullExpr>(), right.downcast_ref::<IsNullExpr>()) {
        return physical_expr_eq(left.arg().as_ref(), right.arg().as_ref());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<IsNotNullExpr>(), right.downcast_ref::<IsNotNullExpr>()) {
        return physical_expr_eq(left.arg().as_ref(), right.arg().as_ref());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<CastExpr>(), right.downcast_ref::<CastExpr>()) {
        return left.cast_type() == right.cast_type() && physical_expr_eq(left.expr().as_ref(), right.expr().as_ref());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<LikeExpr>(), right.downcast_ref::<LikeExpr>()) {
        return left.negated() == right.negated()
            && left.case_insensitive() == right.case_insensitive()
            && physical_expr_eq(left.expr().as_ref(), right.expr().as_ref())
            && physical_expr_eq(left.pattern().as_ref(), right.pattern().as_ref());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<InListExpr>(), right.downcast_ref::<InListExpr>()) {
        return left.negated() == right.negated()
            && physical_expr_eq(left.expr().as_ref(), right.expr().as_ref())
            && physical_expr_slices_eq(left.list(), right.list());
    }
    if let (Some(left), Some(right)) = (left.downcast_ref::<CaseExpr>(), right.downcast_ref::<CaseExpr>()) {
        return optional_physical_expr_eq(left.expr(), right.expr())
            && left.when_then_expr().len() == right.when_then_expr().len()
            && left.when_then_expr().iter().zip(right.when_then_expr()).all(|(left, right)| {
                physical_expr_eq(left.0.as_ref(), right.0.as_ref())
                    && physical_expr_eq(left.1.as_ref(), right.1.as_ref())
            })
            && optional_physical_expr_eq(left.else_expr(), right.else_expr());
    }
    if let (Some(left), Some(right)) =
        (left.downcast_ref::<ScalarFunctionExpr>(), right.downcast_ref::<ScalarFunctionExpr>())
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

pub(super) fn explicit_row_group_selection(file: &PartitionedFile) -> Result<Option<Vec<usize>>, FallbackReason> {
    let Some(access_plan) = file.extensions.get::<ParquetAccessPlan>() else {
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
    let access_plan = file.extensions.get::<ParquetAccessPlan>()?;
    Some(access_plan.inner().iter().filter(|access| !matches!(access, RowGroupAccess::Skip)).count())
}

pub(super) fn has_row_level_selection(file: &PartitionedFile) -> bool {
    let Some(access_plan) = file.extensions.get::<ParquetAccessPlan>() else {
        return false;
    };
    access_plan.inner().iter().any(|access| matches!(access, RowGroupAccess::Selection(_)))
}

pub(super) fn local_object_path_to_filesystem(location: &str) -> PathBuf {
    PathBuf::from("/").join(location)
}
