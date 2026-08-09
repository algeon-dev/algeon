//! Object-store footer pruning for native plans produced from table formats.

mod object_store;
#[cfg(test)]
mod product_path_tests;

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::{
    execution::runtime_env::RuntimeEnv,
    object_store::{ObjectStoreExt, path::Path as ObjectStorePath},
    parquet::{
        arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask, async_reader::ParquetObjectReader},
        file::{metadata::ParquetMetaData, statistics::Statistics},
        schema::types::SchemaDescriptor,
    },
};
use futures::TryStreamExt;
use nexus_query_engine::{
    expr::{BinaryOp, Expr, LiteralValue},
    optimizer::{
        DimensionKeyAccumulator, DimensionKeyFilter, KeyRange, LakehouseFooterPruningEvent,
        LakehouseFooterPruningOptions, LakehouseFooterPruningResolver, LocalParquetRowGroupSelection,
        optimize_lakehouse_footer_pruning,
    },
    parquet_stats::{leaf_column_statistics, row_group_i64_range_for_root_column},
    plan::{NodeId, QueryPlan},
    runtime::SourceStorageProfile,
    source::{LocalParquetSource, SourceColumnRef},
};

#[cfg(test)]
use self::object_store::stable_hash_hex;
use self::object_store::{
    FooterObjectStoreRegistry, foreign_error_diagnostic, object_error_context, object_store_file,
    object_store_pruning_error, stable_log_field,
};
use crate::native::table_format_pruning_policy::TableFormatPruningPolicy;

pub(crate) fn enabled_from_env() -> bool {
    std::env::var("NEXUS_ICEBERG_FOOTER_PRUNING")
        .ok()
        .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true"))
}

/// Resolves footer-backed pruning for one already-lowered native plan.
///
/// The physical optimizer marks native wrappers for this async execution-time
/// continuation. Keeping the resolver on `QueryPlan` lets every native wrapper
/// use the same product path without DataFusion plan downcasts or a benchmark
/// side channel.
pub(crate) async fn prune_native_query_plan(
    plan: QueryPlan,
    runtime_env: &Arc<RuntimeEnv>,
    source_storage_profiles: &[SourceStorageProfile],
) -> QueryPlan {
    let policy = TableFormatPruningPolicy::default();
    let mut resolver =
        FooterPruningResolver { object_stores: FooterObjectStoreRegistry::new(runtime_env, source_storage_profiles) };
    let result = optimize_lakehouse_footer_pruning(
        plan,
        LakehouseFooterPruningOptions::new(policy.max_target_files_across_requests()),
        &mut resolver,
    )
    .await;

    for event in result.events {
        match event {
            LakehouseFooterPruningEvent::StaticReaderPredicateApplied { before, after } => {
                eprintln!(
                    "[lakehouse_pruning] event=static_reader_predicate_applied before_row_groups={} after_row_groups={}",
                    before, after,
                );
            },
            LakehouseFooterPruningEvent::Skipped { reason } => {
                eprintln!("[lakehouse_pruning] event=skip reason={}", stable_log_field(&reason));
            },
            LakehouseFooterPruningEvent::DimensionDerivedApplied { target_source_node_id, before, after } => {
                eprintln!(
                    "[lakehouse_pruning] event=applied target_source_node_id={} before_row_groups={} after_row_groups={}",
                    target_source_node_id.index(),
                    before,
                    after,
                );
            },
            LakehouseFooterPruningEvent::DimensionDerivedSkipped { target_source_node_id, reason } => {
                eprintln!(
                    "[lakehouse_pruning] event=skip target_source_node_id={} reason={}",
                    target_source_node_id.index(),
                    reason,
                );
            },
            LakehouseFooterPruningEvent::DimensionDerivedRequestError { error } => {
                eprintln!("[lakehouse_pruning] event=skip reason=request_error error={}", stable_log_field(&error));
            },
        }
    }

    result.plan
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
            let store = self.object_stores.store_for_file(file, &object, source_node_id)?;
            let metadata = parquet_metadata(store, object.path, file.file_size_bytes, source_node_id).await?;
            let selected_row_groups = selected_or_all_row_groups(file.row_groups.as_deref(), metadata.num_row_groups());
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
        let store = object_stores.store_for_file(file, &object, source_node_id)?;
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
            .collect::<Result<Vec<_>, _>>()?;
        let candidates = selected_or_all_row_groups(file.row_groups.as_deref(), total_row_groups);
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
        | Expr::Sqrt { .. }
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
        std::cmp::Ordering::Equal => Some(raw),
        std::cmp::Ordering::Greater => {
            let factor = decimal_scale_factor(target_scale.checked_sub(source_scale)?)?;
            raw.checked_mul(factor)
        },
        std::cmp::Ordering::Less => {
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
    row_group: &parquet::file::metadata::RowGroupMetaData,
    schema: &SchemaDescriptor,
    expr: &StaticRowGroupPruningExpr,
) -> Option<bool> {
    match expr {
        StaticRowGroupPruningExpr::Atom(atom) => static_row_group_may_satisfy_atom(row_group, schema, atom),
        StaticRowGroupPruningExpr::And(left, right) => {
            let left = static_row_group_may_satisfy_expr(row_group, schema, left);
            let right = static_row_group_may_satisfy_expr(row_group, schema, right);
            Some(!matches!(left, Some(false)) && !matches!(right, Some(false)))
        },
        StaticRowGroupPruningExpr::Or(left, right) => {
            let left = static_row_group_may_satisfy_expr(row_group, schema, left);
            let right = static_row_group_may_satisfy_expr(row_group, schema, right);
            Some(!matches!((left, right), (Some(false), Some(false))))
        },
    }
}

fn static_row_group_may_satisfy_atom(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    schema: &SchemaDescriptor,
    atom: &StaticRowGroupPruningAtom,
) -> Option<bool> {
    match atom {
        StaticRowGroupPruningAtom::Comparison { column_index, op, literal } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            static_row_group_may_satisfy_comparison(statistics, *op, literal)
        },
        StaticRowGroupPruningAtom::InList { column_index, literals } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            static_row_group_may_satisfy_in_list(statistics, literals)
        },
        StaticRowGroupPruningAtom::IsNull { column_index } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            Some(statistics.null_count_opt()? > 0)
        },
        StaticRowGroupPruningAtom::IsNotNull { column_index } => {
            let statistics = leaf_column_statistics(row_group, schema, *column_index)?;
            let row_count = u64::try_from(row_group.num_rows()).ok()?;
            Some(statistics.null_count_opt()? < row_count)
        },
    }
}

fn static_row_group_may_satisfy_in_list(statistics: &Statistics, literals: &[StaticPruningLiteral]) -> Option<bool> {
    if literals.is_empty() {
        return Some(false);
    }
    let mut saw_unknown = false;
    for literal in literals {
        match static_row_group_may_satisfy_comparison(statistics, BinaryOp::Eq, literal) {
            Some(true) => return Some(true),
            Some(false) => {},
            None => saw_unknown = true,
        }
    }
    Some(saw_unknown)
}

fn static_row_group_may_satisfy_comparison(
    statistics: &Statistics,
    op: BinaryOp,
    literal: &StaticPruningLiteral,
) -> Option<bool> {
    match statistics {
        Statistics::Int32(stats) => {
            let min = i128::from(*stats.min_opt()?);
            let max = i128::from(*stats.max_opt()?);
            static_integer_or_decimal_literal(literal)
                .map(|literal| static_i128_range_may_satisfy(min, max, op, literal))
        },
        Statistics::Int64(stats) => {
            let min = i128::from(*stats.min_opt()?);
            let max = i128::from(*stats.max_opt()?);
            static_integer_or_decimal_literal(literal)
                .map(|literal| static_i128_range_may_satisfy(min, max, op, literal))
        },
        Statistics::Float(stats) => {
            let min = f64::from(*stats.min_opt()?);
            let max = f64::from(*stats.max_opt()?);
            let StaticPruningLiteral::Float64(literal) = literal else {
                return None;
            };
            Some(static_float_range_may_satisfy(min, max, op, *literal))
        },
        Statistics::Double(stats) => {
            let min = *stats.min_opt()?;
            let max = *stats.max_opt()?;
            let StaticPruningLiteral::Float64(literal) = literal else {
                return None;
            };
            Some(static_float_range_may_satisfy(min, max, op, *literal))
        },
        Statistics::ByteArray(stats) => {
            let min = stats.min_opt()?.data();
            let max = stats.max_opt()?.data();
            match literal {
                StaticPruningLiteral::Bytes(literal) => Some(static_bytes_range_may_satisfy(min, max, op, literal)),
                StaticPruningLiteral::Decimal128(literal) => {
                    let min = decimal_i128_from_be_bytes(min)?;
                    let max = decimal_i128_from_be_bytes(max)?;
                    Some(static_i128_range_may_satisfy(min, max, op, *literal))
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
            Some(static_i128_range_may_satisfy(min, max, op, *literal))
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
    if bytes.is_empty() || bytes.len() > std::mem::size_of::<i128>() {
        return None;
    }
    let sign_extension = if bytes[0] & 0x80 == 0 { 0x00 } else { 0xff };
    let mut value = [sign_extension; std::mem::size_of::<i128>()];
    value[std::mem::size_of::<i128>() - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(value))
}

fn static_i128_range_may_satisfy(min: i128, max: i128, op: BinaryOp, literal: i128) -> bool {
    match op {
        BinaryOp::Eq => min <= literal && literal <= max,
        BinaryOp::Lt => min < literal,
        BinaryOp::LtEq => min <= literal,
        BinaryOp::Gt => max > literal,
        BinaryOp::GtEq => max >= literal,
        _ => true,
    }
}

fn static_float_range_may_satisfy(min: f64, max: f64, op: BinaryOp, literal: f64) -> bool {
    if min.is_nan() || max.is_nan() || literal.is_nan() {
        return true;
    }
    match op {
        BinaryOp::Eq => min <= literal && literal <= max,
        BinaryOp::Lt => min < literal,
        BinaryOp::LtEq => min <= literal,
        BinaryOp::Gt => max > literal,
        BinaryOp::GtEq => max >= literal,
        _ => true,
    }
}

fn static_bytes_range_may_satisfy(min: &[u8], max: &[u8], op: BinaryOp, literal: &[u8]) -> bool {
    match op {
        BinaryOp::Eq => min <= literal && literal <= max,
        BinaryOp::Lt => min < literal,
        BinaryOp::LtEq => min <= literal,
        BinaryOp::Gt => max > literal,
        BinaryOp::GtEq => max >= literal,
        _ => true,
    }
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
        let store = object_stores.store_for_file(file, &object, source_node_id)?;
        let object_meta = store
            .head(&object.path)
            .await
            .map_err(|error| object_store_pruning_error(error, &object_context, &object.path))?;
        let reader = ParquetObjectReader::new(store, object.path.clone()).with_file_size(object_meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(|error| {
            foreign_error_diagnostic("parquet_record_batch_stream_builder", &error, &object_context, Some(&object.path))
        })?;
        let row_group_count = builder.metadata().num_row_groups();
        let row_groups = selected_or_all_row_groups(file.row_groups.as_deref(), row_group_count);
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
    store: Arc<dyn datafusion::object_store::ObjectStore>,
    path: ObjectStorePath,
    file_size_bytes: Option<u64>,
    source_node_id: NodeId,
) -> std::result::Result<Arc<ParquetMetaData>, String> {
    let object_context = object_error_context(source_node_id, &path);
    let object_meta =
        store.head(&path).await.map_err(|error| object_store_pruning_error(error, &object_context, &path))?;
    let reader =
        ParquetObjectReader::new(store, path.clone()).with_file_size(file_size_bytes.unwrap_or(object_meta.size));
    let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(|error| {
        foreign_error_diagnostic("parquet_record_batch_stream_builder", &error, &object_context, Some(&path))
    })?;
    Ok(Arc::clone(builder.metadata()))
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

fn selected_or_all_row_groups(selected: Option<&[usize]>, row_group_count: usize) -> Vec<usize> {
    selected.map_or_else(|| (0..row_group_count).collect(), |row_groups| row_groups.to_vec())
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
        array::{ArrayRef, Int64Array, StructArray},
        datatypes::{Field, Fields, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::parquet::{
        arrow::ArrowWriter,
        data_type::FixedLenByteArray,
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
    };

    use super::*;

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
        let error = datafusion::object_store::Error::NotFound {
            path: path.to_string(),
            source: Box::new(TestSourceError::leaf("not found")),
        };
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
    fn static_pruning_literal_accepts_decimal128_at_column_scale() {
        let literal = LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 };

        let pruning_literal = static_pruning_literal(&literal, &DataType::Decimal128(15, 2))
            .expect("decimal literal should be supported for decimal column pruning");

        assert!(matches!(pruning_literal, StaticPruningLiteral::Decimal128(5)));
    }

    #[test]
    fn static_pruning_literal_rescales_decimal128_to_column_scale() {
        let literal = LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 };

        let pruning_literal = static_pruning_literal(&literal, &DataType::Decimal128(15, 3))
            .expect("decimal literal should be rescaled for finer column scale");

        assert!(matches!(pruning_literal, StaticPruningLiteral::Decimal128(50)));
        assert!(static_pruning_literal(&literal, &DataType::Decimal128(15, 1)).is_none());
    }

    #[test]
    fn static_row_group_comparison_uses_decimal128_literals_against_int64_stats() {
        let statistics = Statistics::int64(Some(5), Some(7), None, Some(0), false);

        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::Eq, &StaticPruningLiteral::Decimal128(6)),
            Some(true)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::Eq, &StaticPruningLiteral::Decimal128(8)),
            Some(false)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::Lt, &StaticPruningLiteral::Decimal128(5)),
            Some(false)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::GtEq, &StaticPruningLiteral::Decimal128(7)),
            Some(true)
        );
    }

    #[test]
    fn static_row_group_comparison_uses_decimal128_literals_against_fixed_len_byte_array_stats() {
        let statistics = Statistics::fixed_len_byte_array(
            Some(FixedLenByteArray::from((-10_i128).to_be_bytes().to_vec())),
            Some(FixedLenByteArray::from(10_i128.to_be_bytes().to_vec())),
            None,
            Some(0),
            false,
        );

        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::Eq, &StaticPruningLiteral::Decimal128(0)),
            Some(true)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::Lt, &StaticPruningLiteral::Decimal128(-10)),
            Some(false)
        );
        assert_eq!(
            static_row_group_may_satisfy_comparison(&statistics, BinaryOp::Gt, &StaticPruningLiteral::Decimal128(10)),
            Some(false)
        );
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

    fn write_nested_key_parquet(path: &Path) {
        let payload_a = Arc::new(Field::new("a", DataType::Int64, false));
        let payload_b = Arc::new(Field::new("b", DataType::Int64, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new("payload", DataType::Struct(Fields::from(vec![payload_a.clone(), payload_b.clone()])), false),
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
