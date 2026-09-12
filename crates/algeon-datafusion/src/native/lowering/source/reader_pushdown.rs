use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::datasource::physical_plan::{FileSource, ParquetSource};
use query_engine::{
    expr::{BinaryOp as NativeBinaryOp, Expr as NativeExpr},
    source::LocalParquetSource as NativeLocalParquetSource,
};

use crate::{
    capability::CapabilityReason,
    native::{
        lowering::{expr::lower_physical_expr_with_bindings, source::source_bindings_for_schema},
        pruning::{
            scan_predicate_exactly_matches_filter, scan_predicate_is_dynamic_filter_hint,
            scan_predicate_without_dynamic_filter_hints, validate_pruning_only_parquet_predicate,
        },
    },
    source_cardinality::reader_predicate_proves_source_at_most_one_row,
};

#[derive(Debug, Clone)]
pub(in crate::native::lowering) struct SourceReaderPushdownPredicate {
    pub(in crate::native::lowering) expr: NativeExpr,
    pub(in crate::native::lowering) covers_filter: bool,
}

pub(in crate::native::lowering) fn reader_predicate_proves_local_parquet_at_most_one_row(
    table_schema: &Schema,
    file_schema: &Schema,
    predicate: &NativeExpr,
) -> bool {
    reader_predicate_proves_source_at_most_one_row(&[table_schema, file_schema], predicate)
}

pub(in crate::native::lowering) fn source_reader_pushdown_predicate(
    parquet_source: &ParquetSource,
    filter: Option<&datafusion::physical_plan::filter::FilterExec>,
    scan_limit: Option<usize>,
    _scan_projection_requires_native: bool,
    source_output_schema: &SchemaRef,
    frontend_schema: SchemaRef,
    native_source: &NativeLocalParquetSource,
) -> Result<Option<SourceReaderPushdownPredicate>, CapabilityReason> {
    let Some(scan_predicate) = parquet_source.filter() else {
        return Ok(None);
    };
    if filter.is_none() && scan_predicate_is_dynamic_filter_hint(scan_predicate.as_ref()) {
        return Ok(None);
    }
    if filter.is_some() && !parquet_source.table_parquet_options().global.pushdown_filters {
        return Ok(None);
    }
    let reader_scan_predicate = scan_predicate_without_dynamic_filter_hints(&scan_predicate);
    if scan_limit.is_some() {
        return if filter.is_some() { Ok(None) } else { Err(CapabilityReason::NativeSourcePushedDownParquetPredicate) };
    }
    if let Some(filter) = filter
        && validate_pruning_only_parquet_predicate(parquet_source, Some(filter), source_output_schema).is_err()
    {
        return Ok(None);
    }
    let scan_predicate_covers_filter = scan_predicate_exactly_matches_filter(parquet_source, filter);
    let predicate = if let Some(filter) = filter
        && scan_predicate_covers_filter
    {
        let bindings = source_bindings_for_schema(source_output_schema, frontend_schema);
        lower_physical_expr_with_bindings(filter.predicate(), &filter.input().schema(), Some(&bindings))
    } else {
        let Some(reader_scan_predicate) = reader_scan_predicate.as_ref() else {
            return Ok(None);
        };
        let bindings = source_bindings_for_schema(source_output_schema, Arc::clone(&frontend_schema));
        lower_physical_expr_with_bindings(reader_scan_predicate, &frontend_schema, Some(&bindings))
    };
    let predicate = match predicate {
        Ok(predicate) => predicate,
        Err(_error) if filter.is_some() => return Ok(None),
        Err(_error) if filter.is_none() => return Ok(None),
        Err(error) => return Err(error),
    };
    if let Some(predicate) = supported_reader_pushdown_predicate(&predicate, native_source, |_| true) {
        let covers_filter = if filter.is_some() {
            scan_predicate_covers_filter && predicate.covers_predicate
        } else {
            predicate.covers_predicate
        };
        Ok(Some(SourceReaderPushdownPredicate { covers_filter, expr: predicate.expr }))
    } else {
        Ok(None)
    }
}

/// Reader-supported predicate plus whether it fully covers the original filter.
#[derive(Debug)]
pub(super) struct SupportedReaderPushdownPredicate {
    pub(super) expr: NativeExpr,
    pub(super) covers_predicate: bool,
}

pub(super) fn supported_reader_pushdown_predicate(
    predicate: &NativeExpr,
    native_source: &NativeLocalParquetSource,
    allow_expr: impl Fn(&NativeExpr) -> bool,
) -> Option<SupportedReaderPushdownPredicate> {
    if let Some(expr) = native_source
        .reader_predicate_supported_expression(predicate, query_engine::rapids_backend::rapids_analysis_context())
        && allow_expr(&expr)
    {
        return Some(SupportedReaderPushdownPredicate { expr, covers_predicate: true });
    }

    let mut conjuncts = Vec::new();
    collect_supported_reader_pushdown_conjuncts(predicate, native_source, &allow_expr, &mut conjuncts);
    combine_reader_pushdown_conjuncts(conjuncts)
        .map(|expr| SupportedReaderPushdownPredicate { expr, covers_predicate: false })
}

fn collect_supported_reader_pushdown_conjuncts(
    predicate: &NativeExpr,
    native_source: &NativeLocalParquetSource,
    allow_expr: &impl Fn(&NativeExpr) -> bool,
    conjuncts: &mut Vec<NativeExpr>,
) {
    if let Some(expr) = native_source
        .reader_predicate_supported_expression(predicate, query_engine::rapids_backend::rapids_analysis_context())
        && allow_expr(&expr)
    {
        conjuncts.push(expr);
        return;
    }

    if let NativeExpr::Binary { left, op: NativeBinaryOp::And, right, .. } = predicate {
        collect_supported_reader_pushdown_conjuncts(left, native_source, allow_expr, conjuncts);
        collect_supported_reader_pushdown_conjuncts(right, native_source, allow_expr, conjuncts);
    }
}

fn combine_reader_pushdown_conjuncts(conjuncts: Vec<NativeExpr>) -> Option<NativeExpr> {
    let mut conjuncts = conjuncts.into_iter();
    let first = conjuncts.next()?;
    Some(conjuncts.fold(first, |left, right| NativeExpr::binary(left, NativeBinaryOp::And, right)))
}
