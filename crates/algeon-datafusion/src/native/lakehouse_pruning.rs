//! Object-store footer pruning for native plans produced from table formats.

mod object_store;
#[cfg(test)]
mod product_path_tests;

use std::{cmp::Ordering, mem, ops::Range, sync::Arc};

use arrow::datatypes::DataType;
use bytes::Bytes;
use datafusion::{
    execution::runtime_env::RuntimeEnv,
    object_store::{Error as ObjectStoreError, ObjectStore, ObjectStoreExt, path::Path as ObjectStorePath},
    parquet::{
        arrow::{
            ParquetRecordBatchStreamBuilder, ProjectionMask, arrow_reader::ArrowReaderOptions,
            async_reader::AsyncFileReader,
        },
        errors::{ParquetError, Result as ParquetResult},
        file::{
            metadata::{ColumnChunkMetaData, ParquetMetaData, ParquetMetaDataReader, RowGroupMetaData},
            statistics::Statistics,
        },
        schema::types::SchemaDescriptor,
    },
};
use futures::{FutureExt, TryStreamExt, future::BoxFuture};
use query_engine::rapids_backend::rapids_analysis_context;
use query_planning::{
    expr::{BinaryOp, Expr, LiteralValue},
    optimizer::{
        DimensionKeyAccumulator, DimensionKeyFilter, KeyRange, LakehouseFooterPruningEvent,
        LakehouseFooterPruningOptions, LakehouseFooterPruningResolver, LocalParquetRowGroupSelection,
        optimize_lakehouse_footer_pruning,
    },
    parquet_stats::{
        integer_statistics_range, leaf_column_chunk, leaf_column_statistics, row_group_i64_range_for_root_column,
    },
    plan::{NodeId, QueryPlan},
    policy::PlanningPolicy,
    source::{LocalParquetSource, SourceColumnRef},
};

use self::object_store::{
    FooterObjectStoreRegistry, foreign_error_diagnostic, object_error_context, object_store_file,
    object_store_pruning_error, stable_log_field,
};
use super::pruning::{
    and_may_satisfy, float_range_may_satisfy, in_list_may_satisfy, is_not_null_may_satisfy, is_null_may_satisfy,
    or_may_satisfy, ordered_range_may_satisfy, range_comparison_from_binary_op,
};
use crate::{
    error::{Error, Result},
    native::table_format_pruning_policy::TableFormatPruningPolicy,
};

/// Resolves footer-backed pruning for one already-lowered native plan.
///
/// The physical optimizer marks native wrappers for this async execution-time
/// continuation. Passing the resolver with the plan lets every native wrapper
/// use the same product path without DataFusion plan downcasts or a benchmark
/// side channel.
pub(crate) async fn prune_native_query_plan(
    plan: QueryPlan,
    runtime_env: &Arc<RuntimeEnv>,
    planning_policy: PlanningPolicy,
) -> Result<QueryPlan> {
    let policy = TableFormatPruningPolicy::default();
    let mut resolver = FooterPruningResolver { object_stores: FooterObjectStoreRegistry::new(runtime_env) };
    let result = optimize_lakehouse_footer_pruning(
        plan,
        LakehouseFooterPruningOptions::new(policy.max_target_files_across_requests()),
        &mut resolver,
        rapids_analysis_context(planning_policy),
    )
    .await
    .map_err(|error| {
        Error::from_native(
            "algeon_datafusion.native.lakehouse_footer_pruning",
            "lakehouse footer pruning produced a malformed plan",
            error,
        )
    })?;

    for event in result.events {
        match event {
            LakehouseFooterPruningEvent::StaticReaderPredicateApplied { before, after } => {
                tracing::info!(
                    event = "static_reader_predicate_applied",
                    before_row_groups = before,
                    after_row_groups = after,
                    "lakehouse footer pruning"
                );
            },
            LakehouseFooterPruningEvent::Skipped { reason } => {
                tracing::debug!(event = "skip", reason = stable_log_field(&reason), "lakehouse footer pruning");
            },
            LakehouseFooterPruningEvent::DimensionDerivedApplied { target_source_node_id, before, after } => {
                tracing::info!(
                    event = "applied",
                    target_source_node_id = target_source_node_id.index(),
                    before_row_groups = before,
                    after_row_groups = after,
                    "lakehouse footer pruning"
                );
            },
            LakehouseFooterPruningEvent::DimensionDerivedSkipped { target_source_node_id, reason } => {
                tracing::debug!(
                    event = "skip",
                    target_source_node_id = target_source_node_id.index(),
                    reason,
                    "lakehouse footer pruning"
                );
            },
            LakehouseFooterPruningEvent::DimensionDerivedRequestError { error } => {
                tracing::warn!(
                    event = "skip",
                    reason = "request_error",
                    error = stable_log_field(&error),
                    "lakehouse footer pruning"
                );
            },
        }
    }

    Ok(result.plan)
}

struct FooterPruningResolver<'a> {
    object_stores: FooterObjectStoreRegistry<'a>,
}

impl LakehouseFooterPruningResolver for FooterPruningResolver<'_> {
    async fn static_row_group_selections(
        &mut self,
        source_node_id: NodeId,
        source: &LocalParquetSource,
        predicate: &Expr,
    ) -> std::result::Result<Option<Vec<LocalParquetRowGroupSelection>>, String> {
        let Some(pruning_expr) = static_row_group_pruning_expr(source, predicate) else {
            return Ok(None);
        };
        static_reader_predicate_row_group_selections(&mut self.object_stores, source_node_id, source, &pruning_expr)
            .await
            .map(Some)
    }

    async fn dimension_key_filter(
        &mut self,
        source_node_id: NodeId,
        source: &LocalParquetSource,
        key_column: &str,
        predicate: &Expr,
        projection_indices: &[usize],
    ) -> std::result::Result<Option<DimensionKeyFilter>, String> {
        derive_dimension_key_filter(
            &mut self.object_stores,
            source_node_id,
            source,
            key_column,
            predicate,
            projection_indices,
        )
        .await
    }

    async fn target_row_group_key_ranges(
        &mut self,
        source_node_id: NodeId,
        source: &LocalParquetSource,
        key_column_index: usize,
    ) -> std::result::Result<Vec<Vec<Option<KeyRange>>>, String> {
        let mut per_file_row_group_key_ranges = Vec::with_capacity(source.files.len());
        for file in &source.files {
            let object = object_store_file(&file.location)?;
            let store = self.object_stores.store_for_file(&object, source_node_id)?;
            let metadata = parquet_metadata(store, object.path, file.file_size_bytes, source_node_id).await?;
            let selected_row_groups = file.selected_or_all_row_groups(metadata.num_row_groups());
            per_file_row_group_key_ranges.push(row_group_key_ranges(
                metadata.as_ref(),
                key_column_index,
                &selected_row_groups,
            ));
        }
        Ok(per_file_row_group_key_ranges)
    }
}

async fn static_reader_predicate_row_group_selections(
    object_stores: &mut FooterObjectStoreRegistry<'_>,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    predicate: &StaticRowGroupPruningExpr,
) -> std::result::Result<Vec<LocalParquetRowGroupSelection>, String> {
    let mut selections = Vec::with_capacity(source.files.len());
    for file in &source.files {
        let object = object_store_file(&file.location).map_err(|error| {
            format!(
                "static_reader_predicate_invalid_object_uri source_node_id={} reason={}",
                source_node_id.index(),
                stable_log_field(&error)
            )
        })?;
        let store = object_stores.store_for_file(&object, source_node_id)?;
        let metadata = parquet_metadata(store, object.path, file.file_size_bytes, source_node_id)
            .await
            .map_err(|error| format!("static_reader_predicate_footer_unavailable {error}"))?;
        let total_row_groups = metadata.num_row_groups();
        let row_group_rows = metadata
            .row_groups()
            .iter()
            .map(|row_group| {
                u64::try_from(row_group.num_rows()).map_err(|_| {
                    format!(
                        "static_reader_predicate_row_group_rows_invalid source_node_id={} row_group_count={}",
                        source_node_id.index(),
                        total_row_groups,
                    )
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let candidates = file.selected_or_all_row_groups(total_row_groups);
        let mut row_groups = Vec::with_capacity(candidates.len());
        for row_group_index in candidates {
            let Some(row_group) = metadata.row_groups().get(row_group_index) else {
                return Err("static_reader_predicate_row_group_out_of_range".to_string());
            };
            if static_row_group_may_satisfy_expr(row_group, metadata.file_metadata().schema_descr(), predicate)
                .unwrap_or(true)
            {
                row_groups.push(row_group_index);
            }
        }
        selections.push(LocalParquetRowGroupSelection { total_row_groups, row_groups, row_group_rows });
    }
    Ok(selections)
}

#[derive(Debug, Clone)]
enum StaticRowGroupPruningExpr {
    Atom(StaticRowGroupPruningAtom),
    And(Box<StaticRowGroupPruningExpr>, Box<StaticRowGroupPruningExpr>),
    Or(Box<StaticRowGroupPruningExpr>, Box<StaticRowGroupPruningExpr>),
}

#[derive(Debug, Clone)]
enum StaticRowGroupPruningAtom {
    Comparison { column_index: usize, op: BinaryOp, literal: StaticPruningLiteral },
    InList { column_index: usize, literals: Vec<StaticPruningLiteral> },
    IsNull { column_index: usize },
    IsNotNull { column_index: usize },
}

#[derive(Debug, Clone)]
enum StaticPruningLiteral {
    Int64(i64),
    Decimal128(i128),
    Float64(f64),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
struct StaticPruningColumn {
    column_index: usize,
    data_type: DataType,
}

fn static_row_group_pruning_expr(source: &LocalParquetSource, expr: &Expr) -> Option<StaticRowGroupPruningExpr> {
    match expr {
        Expr::IsNull(expr) => {
            let column = static_pruning_column(source, expr)?;
            Some(StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::IsNull {
                column_index: column.column_index,
            }))
        },
        Expr::IsNotNull(expr) => {
            let column = static_pruning_column(source, expr)?;
            Some(StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::IsNotNull {
                column_index: column.column_index,
            }))
        },
        Expr::InList { expr, list, negated: false } => {
            let column = static_pruning_column(source, expr)?;
            let literals = list
                .iter()
                .map(|expr| match expr {
                    Expr::Literal(value) => static_pruning_literal(value, &column.data_type),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::InList {
                column_index: column.column_index,
                literals,
            }))
        },
        Expr::InList { negated: true, .. } => None,
        Expr::Binary { left, op: BinaryOp::And, right, .. } => static_pruning_conjunction(source, left, right),
        Expr::Binary { left, op: BinaryOp::Or, right, .. } => {
            let left = static_row_group_pruning_expr(source, left)?;
            let right = static_row_group_pruning_expr(source, right)?;
            Some(StaticRowGroupPruningExpr::Or(Box::new(left), Box::new(right)))
        },
        Expr::Binary { left, op, right, .. } => {
            if let Some((column, literal, op)) = static_column_literal_predicate(source, left, *op, right) {
                return Some(StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::Comparison {
                    column_index: column.column_index,
                    op,
                    literal: static_pruning_literal(literal, &column.data_type)?,
                }));
            }
            let inverted_op = invert_static_comparison_operator(*op)?;
            let (column, literal, op) = static_column_literal_predicate(source, right, inverted_op, left)?;
            Some(StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::Comparison {
                column_index: column.column_index,
                op,
                literal: static_pruning_literal(literal, &column.data_type)?,
            }))
        },
        Expr::Column(_)
        | Expr::Literal(_)
        | Expr::Not(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::UnaryMath { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. }
        | Expr::Alias { .. } => None,
    }
}

fn static_pruning_conjunction(
    source: &LocalParquetSource,
    left: &Expr,
    right: &Expr,
) -> Option<StaticRowGroupPruningExpr> {
    match (static_row_group_pruning_expr(source, left), static_row_group_pruning_expr(source, right)) {
        (Some(left), Some(right)) => Some(StaticRowGroupPruningExpr::And(Box::new(left), Box::new(right))),
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (None, None) => None,
    }
}

fn static_column_literal_predicate<'a>(
    source: &LocalParquetSource,
    left: &Expr,
    op: BinaryOp,
    right: &'a Expr,
) -> Option<(StaticPruningColumn, &'a LiteralValue, BinaryOp)> {
    let column = static_pruning_column(source, left)?;
    let Expr::Literal(literal) = right else {
        return None;
    };
    matches!(op, BinaryOp::Eq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq)
        .then_some((column, literal, op))
}

fn static_pruning_column(source: &LocalParquetSource, expr: &Expr) -> Option<StaticPruningColumn> {
    if let Expr::Cast { expr, target_type } = expr {
        let column = static_pruning_column(source, expr)?;
        if static_order_preserving_pruning_cast(&column.data_type, target_type) {
            return Some(StaticPruningColumn { data_type: target_type.clone(), ..column });
        }
        return None;
    }
    let Expr::Column(column) = expr else {
        return None;
    };
    let SourceColumnRef::File(column_index) = source.source_column_ref(column.name())? else {
        return None;
    };
    let data_type = source.schema.field_with_name(column.name()).ok()?.data_type().clone();
    Some(StaticPruningColumn { column_index, data_type })
}

fn static_order_preserving_pruning_cast(source_type: &DataType, target_type: &DataType) -> bool {
    source_type == target_type
        || static_signed_integer_rank(source_type)
            .zip(static_signed_integer_rank(target_type))
            .is_some_and(|(source, target)| source <= target)
}

fn static_signed_integer_rank(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Int8 => Some(0),
        DataType::Int16 => Some(1),
        DataType::Int32 => Some(2),
        DataType::Int64 => Some(3),
        _ => None,
    }
}

fn invert_static_comparison_operator(op: BinaryOp) -> Option<BinaryOp> {
    match op {
        BinaryOp::Eq => Some(BinaryOp::Eq),
        BinaryOp::Lt => Some(BinaryOp::Gt),
        BinaryOp::LtEq => Some(BinaryOp::GtEq),
        BinaryOp::Gt => Some(BinaryOp::Lt),
        BinaryOp::GtEq => Some(BinaryOp::LtEq),
        _ => None,
    }
}

fn static_pruning_literal(value: &LiteralValue, column_type: &DataType) -> Option<StaticPruningLiteral> {
    match value {
        LiteralValue::Int8(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::Int16(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::Int32(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::Int64(value) => Some(StaticPruningLiteral::Int64(*value)),
        LiteralValue::UInt8(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::UInt16(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::UInt32(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::UInt64(value) => i64::try_from(*value).ok().map(StaticPruningLiteral::Int64),
        LiteralValue::Date32(value) => Some(StaticPruningLiteral::Int64(i64::from(*value))),
        LiteralValue::Date64(value) => Some(StaticPruningLiteral::Int64(*value)),
        LiteralValue::Timestamp { raw, .. } => Some(StaticPruningLiteral::Int64(*raw)),
        LiteralValue::Float32(value) => Some(StaticPruningLiteral::Float64(f64::from(*value))),
        LiteralValue::Float64(value) => Some(StaticPruningLiteral::Float64(*value)),
        LiteralValue::Decimal128 { raw, scale, .. } => match column_type {
            DataType::Decimal128(_, column_scale) => {
                rescale_decimal_raw(*raw, *scale, *column_scale).map(StaticPruningLiteral::Decimal128)
            },
            _ => None,
        },
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value)
            if matches!(column_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) =>
        {
            Some(StaticPruningLiteral::Bytes(value.as_bytes().to_vec()))
        },
        LiteralValue::Boolean(_)
        | LiteralValue::Utf8(_)
        | LiteralValue::LargeUtf8(_)
        | LiteralValue::Time32 { .. }
        | LiteralValue::Time64 { .. }
        | LiteralValue::Duration { .. }
        | LiteralValue::IntervalMonthDayNano { .. }
        | LiteralValue::Null(_) => None,
    }
}

fn rescale_decimal_raw(raw: i128, source_scale: i8, target_scale: i8) -> Option<i128> {
    match target_scale.cmp(&source_scale) {
        Ordering::Equal => Some(raw),
        Ordering::Greater => {
            let factor = decimal_scale_factor(target_scale.checked_sub(source_scale)?)?;
            raw.checked_mul(factor)
        },
        Ordering::Less => {
            let factor = decimal_scale_factor(source_scale.checked_sub(target_scale)?)?;
            (raw % factor == 0).then_some(raw / factor)
        },
    }
}

fn decimal_scale_factor(scale_delta: i8) -> Option<i128> {
    let scale_delta = u32::try_from(scale_delta).ok()?;
    10_i128.checked_pow(scale_delta)
}

fn static_row_group_may_satisfy_expr(
    row_group: &RowGroupMetaData,
    schema: &SchemaDescriptor,
    expr: &StaticRowGroupPruningExpr,
) -> Option<bool> {
    match expr {
        StaticRowGroupPruningExpr::Atom(atom) => static_row_group_may_satisfy_atom(row_group, schema, atom),
        StaticRowGroupPruningExpr::And(left, right) => {
            let left = static_row_group_may_satisfy_expr(row_group, schema, left);
            let right = static_row_group_may_satisfy_expr(row_group, schema, right);
            Some(and_may_satisfy(left, right))
        },
        StaticRowGroupPruningExpr::Or(left, right) => {
            let left = static_row_group_may_satisfy_expr(row_group, schema, left);
            let right = static_row_group_may_satisfy_expr(row_group, schema, right);
            Some(or_may_satisfy(left, right))
        },
    }
}

fn static_row_group_may_satisfy_atom(
    row_group: &RowGroupMetaData,
    schema: &SchemaDescriptor,
    atom: &StaticRowGroupPruningAtom,
) -> Option<bool> {
    match atom {
        StaticRowGroupPruningAtom::Comparison { column_index, op, literal } => {
            let column = leaf_column_chunk(row_group, schema, *column_index)?;
            static_row_group_may_satisfy_comparison(column, *op, literal)
        },
        StaticRowGroupPruningAtom::InList { column_index, literals } => {
            let column = leaf_column_chunk(row_group, schema, *column_index)?;
            static_row_group_may_satisfy_in_list(column, literals)
        },
        StaticRowGroupPruningAtom::IsNull { column_index } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            is_null_may_satisfy(statistics)
        },
        StaticRowGroupPruningAtom::IsNotNull { column_index } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            let row_count = u64::try_from(row_group.num_rows()).ok()?;
            is_not_null_may_satisfy(statistics, row_count)
        },
    }
}

fn static_row_group_may_satisfy_in_list(
    column: &ColumnChunkMetaData,
    literals: &[StaticPruningLiteral],
) -> Option<bool> {
    in_list_may_satisfy(literals, |literal| static_row_group_may_satisfy_comparison(column, BinaryOp::Eq, literal))
}

fn static_row_group_may_satisfy_comparison(
    column: &ColumnChunkMetaData,
    op: BinaryOp,
    literal: &StaticPruningLiteral,
) -> Option<bool> {
    let statistics = column.statistics()?;
    let comparison = range_comparison_from_binary_op(op);
    match statistics {
        Statistics::Int32(_) | Statistics::Int64(_) => {
            let range = integer_statistics_range(column)?;
            static_integer_or_decimal_literal(literal)
                .map(|literal| ordered_range_may_satisfy(range.min, range.max, comparison, literal))
        },
        Statistics::Float(stats) => {
            let min = f64::from(*stats.min_opt()?);
            let max = f64::from(*stats.max_opt()?);
            let StaticPruningLiteral::Float64(literal) = literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, comparison, *literal))
        },
        Statistics::Double(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let StaticPruningLiteral::Float64(literal) = literal else {
                return None;
            };
            Some(float_range_may_satisfy(min, max, comparison, *literal))
        },
        Statistics::ByteArray(stats) => {
            let min = stats.min_opt()?.data();
            let max = stats.max_opt()?.data();
            match literal {
                StaticPruningLiteral::Bytes(literal) => {
                    Some(ordered_range_may_satisfy(min, max, comparison, literal.as_slice()))
                },
                StaticPruningLiteral::Decimal128(literal) => {
                    let min = decimal_i128_from_be_bytes(min)?;
                    let max = decimal_i128_from_be_bytes(max)?;
                    Some(ordered_range_may_satisfy(min, max, comparison, *literal))
                },
                StaticPruningLiteral::Int64(_) | StaticPruningLiteral::Float64(_) => None,
            }
        },
        Statistics::FixedLenByteArray(stats) => {
            let min = stats.min_opt()?.data();
            let max = stats.max_opt()?.data();
            let StaticPruningLiteral::Decimal128(literal) = literal else {
                return None;
            };
            let min = decimal_i128_from_be_bytes(min)?;
            let max = decimal_i128_from_be_bytes(max)?;
            Some(ordered_range_may_satisfy(min, max, comparison, *literal))
        },
        _ => None,
    }
}

fn static_integer_or_decimal_literal(literal: &StaticPruningLiteral) -> Option<i128> {
    match literal {
        StaticPruningLiteral::Int64(value) => Some(i128::from(*value)),
        StaticPruningLiteral::Decimal128(value) => Some(*value),
        StaticPruningLiteral::Float64(_) | StaticPruningLiteral::Bytes(_) => None,
    }
}

fn decimal_i128_from_be_bytes(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() || bytes.len() > mem::size_of::<i128>() {
        return None;
    }
    let sign_extension = if bytes[0] & 0x80 == 0 { 0x00 } else { 0xff };
    let mut value = [sign_extension; mem::size_of::<i128>()];
    value[mem::size_of::<i128>() - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(value))
}

async fn derive_dimension_key_filter(
    object_stores: &mut FooterObjectStoreRegistry<'_>,
    source_node_id: NodeId,
    source: &LocalParquetSource,
    key_column: &str,
    predicate: &Expr,
    projection_indices: &[usize],
) -> std::result::Result<Option<DimensionKeyFilter>, String> {
    // Fold object-store dimension batches through the same accumulator the
    // in-engine local pass uses, so both derive identical set-aware filters.
    let policy = TableFormatPruningPolicy::default();
    let mut accumulator = DimensionKeyAccumulator::default();
    let mut total_rows = 0usize;
    for file in &source.files {
        let object = object_store_file(&file.location)?;
        let object_context = object_error_context(source_node_id, &object.path);
        let store = object_stores.store_for_file(&object, source_node_id)?;
        let object_meta = store
            .head(&object.path)
            .await
            .map_err(|error| object_store_pruning_error(error, &object_context, &object.path))?;
        let reader = ObjectStoreParquetReader::new(store, object.path.clone(), object_meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(|error| {
            foreign_error_diagnostic("parquet_record_batch_stream_builder", &error, &object_context, Some(&object.path))
        })?;
        let row_group_count = builder.metadata().num_row_groups();
        let row_groups = file.selected_or_all_row_groups(row_group_count);
        let projection = ProjectionMask::roots(builder.parquet_schema(), projection_indices.iter().copied());
        let mut stream = builder.with_projection(projection).with_row_groups(row_groups).build().map_err(|error| {
            foreign_error_diagnostic("parquet_record_batch_stream_build", &error, &object_context, Some(&object.path))
        })?;
        while let Some(batch) = stream.try_next().await.map_err(|error| {
            foreign_error_diagnostic("parquet_record_batch_stream", &error, &object_context, Some(&object.path))
        })? {
            total_rows = total_rows.saturating_add(batch.num_rows());
            if policy.dimension_row_guard_exceeded(total_rows) {
                return Err(format!(
                    "dimension_row_guard_exceeded total_rows={} max_rows={}",
                    total_rows,
                    policy.max_dimension_rows()
                ));
            }
            accumulator
                .observe_batch(&batch, key_column, predicate)
                .ok_or_else(|| "unsupported_dimension_predicate_or_key".to_string())?;
        }
    }
    Ok(accumulator.finish())
}

async fn parquet_metadata(
    store: Arc<dyn ObjectStore>,
    path: ObjectStorePath,
    file_size_bytes: Option<u64>,
    source_node_id: NodeId,
) -> std::result::Result<Arc<ParquetMetaData>, String> {
    let object_context = object_error_context(source_node_id, &path);
    let object_meta =
        store.head(&path).await.map_err(|error| object_store_pruning_error(error, &object_context, &path))?;
    let reader = ObjectStoreParquetReader::new(store, path.clone(), file_size_bytes.unwrap_or(object_meta.size));
    let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(|error| {
        foreign_error_diagnostic("parquet_record_batch_stream_builder", &error, &object_context, Some(&path))
    })?;
    Ok(Arc::clone(builder.metadata()))
}

/// Direct [`AsyncFileReader`] over one object-store location. The file size
/// is known from the preceding `head`, so the footer loads without suffix
/// probing.
struct ObjectStoreParquetReader {
    store: Arc<dyn ObjectStore>,
    path: ObjectStorePath,
    file_size: u64,
}

impl ObjectStoreParquetReader {
    fn new(store: Arc<dyn ObjectStore>, path: ObjectStorePath, file_size: u64) -> Self {
        Self { store, path, file_size }
    }
}

fn object_store_parquet_error(error: ObjectStoreError) -> ParquetError {
    ParquetError::External(Box::new(error))
}

impl AsyncFileReader for ObjectStoreParquetReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        async move { self.store.get_range(&self.path, range).await.map_err(object_store_parquet_error) }.boxed()
    }

    fn get_byte_ranges(&mut self, ranges: Vec<Range<u64>>) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        async move { self.store.get_ranges(&self.path, &ranges).await.map_err(object_store_parquet_error) }.boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        let file_size = self.file_size;
        async move {
            let metadata = ParquetMetaDataReader::new()
                .with_arrow_reader_options(options)
                .load_and_finish(self, file_size)
                .await?;
            Ok(Arc::new(metadata))
        }
        .boxed()
    }
}

fn row_group_key_ranges(
    metadata: &ParquetMetaData,
    column_index: usize,
    selected_row_groups: &[usize],
) -> Vec<Option<KeyRange>> {
    let schema = metadata.file_metadata().schema_descr();
    selected_row_groups
        .iter()
        .map(|row_group_index| {
            let row_group = metadata.row_groups().get(*row_group_index)?;
            row_group_i64_range_for_root_column(row_group, schema, column_index)
                .map(|range| KeyRange { min: range.min, max: range.max })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error as StdError,
        fmt,
        fs::{self, File},
        path::{Path, PathBuf},
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arrow::{
        array::{ArrayRef, Int64Array, StructArray, UInt32Array},
        datatypes::{Field, Fields, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::{
        object_store::memory::InMemory,
        parquet::{
            arrow::ArrowWriter,
            data_type::FixedLenByteArray,
            file::{
                properties::WriterProperties,
                reader::{FileReader, SerializedFileReader},
            },
        },
    };
    use parquet::{file::metadata::ColumnChunkMetaData, schema::parser::parse_message_type};

    use super::{object_store::stable_hash_hex, *};

    fn column_chunk_with_statistics(schema_body: &str, statistics: Statistics) -> ColumnChunkMetaData {
        let parquet_type = parse_message_type(schema_body).expect("test Parquet schema should parse");
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(parquet_type)));
        ColumnChunkMetaData::builder(Arc::clone(&schema.columns()[0]))
            .set_statistics(statistics)
            .build()
            .expect("test column metadata should build")
    }

    #[derive(Debug)]
    struct TestSourceError {
        message: &'static str,
        source: Option<Box<dyn StdError + Send + Sync>>,
    }

    impl TestSourceError {
        fn leaf(message: &'static str) -> Self {
            Self { message, source: None }
        }

        fn with_source(message: &'static str, source: impl StdError + Send + Sync + 'static) -> Self {
            Self { message, source: Some(Box::new(source)) }
        }
    }

    impl fmt::Display for TestSourceError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.message)
        }
    }

    impl StdError for TestSourceError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
        }
    }

    #[test]
    fn foreign_error_diagnostic_preserves_source_chain_and_resource_context() {
        let error = TestSourceError::with_source("async wrapper", TestSourceError::leaf("expired token"));

        let diagnostic = foreign_error_diagnostic(
            "parquet_record_batch_stream",
            &error,
            &[
                ("source_node_id", "n7".to_string()),
                ("object_path_hash", stable_hash_hex("warehouse/table/data.parquet")),
                ("object_path_leaf", "data.parquet".to_string()),
            ],
            None,
        );

        assert!(diagnostic.contains("parquet_record_batch_stream=async wrapper: expired token"), "{diagnostic}");
        assert!(diagnostic.contains("error_root_cause=expired token"), "{diagnostic}");
        assert!(diagnostic.contains("source_node_id=n7"), "{diagnostic}");
        assert!(diagnostic.contains("object_path_leaf=data.parquet"), "{diagnostic}");
        assert!(diagnostic.contains("object_path_hash="), "{diagnostic}");
        assert!(!diagnostic.contains("warehouse/table"), "{diagnostic}");
    }

    #[test]
    fn foreign_error_diagnostic_redacts_object_path_from_source_chain() {
        let path =
            ObjectStorePath::parse("warehouse/private/table/data.parquet").expect("test object path should parse");
        let error = TestSourceError::with_source(
            "parquet external error for warehouse/private/table/data.parquet",
            TestSourceError::leaf("read failed for warehouse/private/table/data.parquet"),
        );

        let diagnostic = foreign_error_diagnostic(
            "parquet_record_batch_stream_builder",
            &error,
            &[
                ("source_node_id", "n9".to_string()),
                ("object_path_hash", stable_hash_hex(path.as_ref())),
                ("object_path_leaf", "data.parquet".to_string()),
            ],
            Some(&path),
        );

        assert!(diagnostic.contains("<redacted-object-path>"), "{diagnostic}");
        assert!(diagnostic.contains("object_path_leaf=data.parquet"), "{diagnostic}");
        assert!(!diagnostic.contains("warehouse/private/table"), "{diagnostic}");
    }

    #[test]
    fn object_store_pruning_error_redacts_object_path_from_source_chain() {
        let path =
            ObjectStorePath::parse("warehouse/private/table/data.parquet").expect("test object path should parse");
        let error =
            ObjectStoreError::NotFound { path: path.to_string(), source: Box::new(TestSourceError::leaf("not found")) };
        let diagnostic = object_store_pruning_error(
            error,
            &[("object_path_leaf", "data.parquet".to_string()), ("object_path_hash", stable_hash_hex(path.as_ref()))],
            &path,
        );

        assert!(diagnostic.contains("error_source_chain="), "{diagnostic}");
        assert!(diagnostic.contains("<redacted-object-path>"), "{diagnostic}");
        assert!(diagnostic.contains("object_path_leaf=data.parquet"), "{diagnostic}");
        assert!(!diagnostic.contains("warehouse/private/table"), "{diagnostic}");
    }

    #[test]
    fn static_pruning_literal_rescales_decimal128_to_column_scale() {
        let literal = LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 };

        let pruning_literal = static_pruning_literal(&literal, &DataType::Decimal128(15, 3))
            .expect("decimal literal should be rescaled for finer column scale");

        assert!(matches!(pruning_literal, StaticPruningLiteral::Decimal128(50)));
        assert!(matches!(
            static_pruning_literal(&literal, &DataType::Decimal128(15, 2)),
            Some(StaticPruningLiteral::Decimal128(5))
        ));
        assert!(static_pruning_literal(&literal, &DataType::Decimal128(15, 1)).is_none());
    }

    #[test]
    fn static_row_group_comparison_uses_decimal128_literals_against_int64_stats() {
        let column = column_chunk_with_statistics(
            "message schema { required int64 k; }",
            Statistics::int64(Some(5), Some(7), None, Some(0), false),
        );

        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::Eq, &StaticPruningLiteral::Decimal128(6)),
            Some(true)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::Eq, &StaticPruningLiteral::Decimal128(8)),
            Some(false)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::Lt, &StaticPruningLiteral::Decimal128(5)),
            Some(false)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::GtEq, &StaticPruningLiteral::Decimal128(7)),
            Some(true)
        );
    }

    #[test]
    fn static_row_group_comparison_uses_decimal128_literals_against_fixed_len_byte_array_stats() {
        let column = column_chunk_with_statistics(
            "message schema { required fixed_len_byte_array(16) k; }",
            Statistics::fixed_len_byte_array(
                Some(FixedLenByteArray::from((-10_i128).to_be_bytes().to_vec())),
                Some(FixedLenByteArray::from(10_i128.to_be_bytes().to_vec())),
                None,
                Some(0),
                false,
            ),
        );

        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::Eq, &StaticPruningLiteral::Decimal128(0)),
            Some(true)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::Lt, &StaticPruningLiteral::Decimal128(-10)),
            Some(false)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&column, BinaryOp::Gt, &StaticPruningLiteral::Decimal128(10)),
            Some(false)
        );
    }

    #[test]
    fn static_row_group_comparison_decodes_unsigned_uint32_statistics_as_unsigned() {
        let dir = unique_test_dir("lakehouse-static-uint32");
        fs::create_dir_all(&dir).expect("temporary lakehouse pruning directory should be created");
        let path = dir.join("uint32.parquet");
        write_uint32_key_parquet(&path);
        let reader = SerializedFileReader::try_from(path.as_path()).expect("test parquet file should be readable");
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 1);
        let schema = metadata.file_metadata().schema_descr();

        let predicate = StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::Comparison {
            column_index: 0,
            op: BinaryOp::Eq,
            literal: StaticPruningLiteral::Int64(3_000_000_000),
        });
        assert_eq!(static_row_group_may_satisfy_expr(metadata.row_group(0), schema, &predicate), Some(true));

        assert_eq!(row_group_key_ranges(metadata, 0, &[0]), vec![Some(KeyRange { min: 100, max: 3_000_000_000 })]);

        fs::remove_dir_all(dir).expect("temporary lakehouse pruning directory should be removable");
    }

    #[test]
    fn static_reader_predicate_pruning_uses_leaf_stats_for_root_column() {
        let dir = unique_test_dir("lakehouse-static-nested");
        fs::create_dir_all(&dir).expect("temporary lakehouse pruning directory should be created");
        let path = dir.join("nested.parquet");
        write_nested_key_parquet(&path);
        let reader = SerializedFileReader::try_from(path.as_path()).expect("test parquet file should be readable");
        let metadata = reader.metadata();
        assert_eq!(metadata.num_row_groups(), 2);

        let predicate = StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::Comparison {
            column_index: 1,
            op: BinaryOp::Eq,
            literal: StaticPruningLiteral::Int64(5),
        });
        let schema = metadata.file_metadata().schema_descr();

        assert_eq!(static_row_group_may_satisfy_expr(metadata.row_group(0), schema, &predicate), Some(false));
        assert_eq!(static_row_group_may_satisfy_expr(metadata.row_group(1), schema, &predicate), Some(true));

        fs::remove_dir_all(dir).expect("temporary lakehouse pruning directory should be removable");
    }

    #[test]
    fn dimension_key_ranges_use_leaf_stats_for_root_column() {
        let dir = unique_test_dir("lakehouse-dimension-nested");
        fs::create_dir_all(&dir).expect("temporary lakehouse pruning directory should be created");
        let path = dir.join("nested.parquet");
        write_nested_key_parquet(&path);
        let reader = SerializedFileReader::try_from(path.as_path()).expect("test parquet file should be readable");
        let metadata = reader.metadata();

        let ranges = row_group_key_ranges(metadata, 1, &[0, 1]);

        assert_eq!(ranges, vec![Some(KeyRange { min: 100, max: 100 }), Some(KeyRange { min: 5, max: 5 })]);

        fs::remove_dir_all(dir).expect("temporary lakehouse pruning directory should be removable");
    }

    #[test]
    fn nested_root_column_stats_are_not_used_for_pruning() {
        let dir = unique_test_dir("lakehouse-nested-root");
        fs::create_dir_all(&dir).expect("temporary lakehouse pruning directory should be created");
        let path = dir.join("nested.parquet");
        write_nested_key_parquet(&path);
        let reader = SerializedFileReader::try_from(path.as_path()).expect("test parquet file should be readable");
        let metadata = reader.metadata();
        let predicate = StaticRowGroupPruningExpr::Atom(StaticRowGroupPruningAtom::Comparison {
            column_index: 0,
            op: BinaryOp::Eq,
            literal: StaticPruningLiteral::Int64(5),
        });
        let schema = metadata.file_metadata().schema_descr();

        assert_eq!(static_row_group_may_satisfy_expr(metadata.row_group(0), schema, &predicate), None);
        assert_eq!(row_group_key_ranges(metadata, 0, &[0, 1]), vec![None, None]);

        fs::remove_dir_all(dir).expect("temporary lakehouse pruning directory should be removable");
    }

    #[tokio::test]
    async fn object_store_parquet_reader_loads_metadata_and_reads_byte_ranges() {
        let dir = unique_test_dir("lakehouse-reader-test");
        fs::create_dir_all(&dir).expect("temporary directory should be created");
        let path = dir.join("test.parquet");
        write_nested_key_parquet(&path);

        let bytes = fs::read(&path).expect("test parquet file bytes should be readable");
        let file_size = bytes.len() as u64;

        let store = Arc::new(InMemory::new());
        let object_path = ObjectStorePath::parse("warehouse/test.parquet").expect("valid path");
        store.put(&object_path, bytes.clone().into()).await.expect("put should succeed");

        let mut reader = ObjectStoreParquetReader::new(Arc::clone(&store) as _, object_path.clone(), file_size);

        let metadata = reader.get_metadata(None).await.expect("metadata should load");
        assert_eq!(metadata.file_metadata().num_rows(), 2);
        assert_eq!(metadata.num_row_groups(), 2);

        let slice = reader.get_bytes(0..16).await.expect("range read should succeed");
        assert_eq!(slice.as_ref(), &bytes[0..16]);

        let slices = reader.get_byte_ranges(vec![0..8, 8..24]).await.expect("multi range read should succeed");
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].as_ref(), &bytes[0..8]);
        assert_eq!(slices[1].as_ref(), &bytes[8..24]);

        fs::remove_dir_all(dir).expect("cleanup should succeed");
    }

    #[tokio::test]
    async fn object_store_parquet_reader_maps_errors_to_parquet_external() {
        let store = Arc::new(InMemory::new());
        let missing_path = ObjectStorePath::parse("warehouse/missing.parquet").expect("valid path");
        let mut reader = ObjectStoreParquetReader::new(store, missing_path, 100);

        let result = reader.get_bytes(0..10).await;
        let err = result.expect_err("reading missing object must fail");
        assert!(matches!(err, ParquetError::External(_)), "error should be External ParquetError: {err}");
    }

    fn write_uint32_key_parquet(path: &Path) {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::UInt32, false)]));
        let key = Arc::new(UInt32Array::from(vec![100u32, 3_000_000_000u32])) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![key]).expect("test batch should be valid");
        let file = File::create(path).expect("test parquet file should be creatable");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("test parquet writer should be valid");
        writer.write(&batch).expect("test parquet batch should be writable");
        writer.close().expect("test parquet writer should close");
    }

    fn write_nested_key_parquet(path: &Path) {
        let payload_a = Arc::new(Field::new("a", DataType::Int64, false));
        let payload_b = Arc::new(Field::new("b", DataType::Int64, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "payload",
                DataType::Struct(Fields::from(vec![Arc::clone(&payload_a), Arc::clone(&payload_b)])),
                false,
            ),
            Field::new("key", DataType::Int64, false),
        ]));
        let payload_a_values = Arc::new(Int64Array::from(vec![0, 0])) as ArrayRef;
        let payload_b_values = Arc::new(Int64Array::from(vec![5, 100])) as ArrayRef;
        let payload =
            Arc::new(StructArray::from(vec![(payload_a, payload_a_values), (payload_b, payload_b_values)])) as ArrayRef;
        let key = Arc::new(Int64Array::from(vec![100, 5])) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![payload, key]).expect("test batch should be valid");
        let file = File::create(path).expect("test parquet file should be creatable");
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("test parquet writer should be valid");
        writer.write(&batch).expect("test parquet batch should be writable");
        writer.close().expect("test parquet writer should close");
    }

    fn unique_test_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be valid").as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }
}
