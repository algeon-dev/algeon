//! Host-side evaluation of dimension predicates and key expressions over
//! decoded Arrow batches.
//!
//! Resolvers fold the rows they read through [`I64ExprRangeAccumulator`] and
//! `DimensionKeyAccumulator`, so the in-crate local pass and the adapter's
//! object-store pass derive the same range from the same rows. The scalar
//! evaluator decides which rows contribute keys and values. A row it rejects
//! that the executor accepts drops a key or narrows a range, which prunes a
//! row group the join still needs or filters rows the join matches, so every
//! value it computes must equal the executor's. It returns `None` for every
//! expression form, type, or operator whose result it does not reproduce, and
//! callers treat `None` as "no facts" and skip the candidate.
//!
//! `AND`, `OR` and `NOT` follow SQL three-valued logic, and only the final
//! predicate result maps NULL to a rejected row. Integers keep the Arrow
//! type they were read or cast as. `+`, `-` and `*` take operands of one
//! signedness and wrap at the wider operand's width, as native integer
//! arithmetic does. Mixed signedness returns `None`.

use std::cmp::Ordering;

use super::{
    AggregateNode, Array, BinaryOp, BooleanArray, DataType, Date32Array, DimensionKeyFilter, Expr, Float32Array,
    Float64Array, Int32Array, Int64Array, KeyRange, LargeStringArray, LiteralValue, LocalParquetPruningResolver,
    LocalParquetSource, NodeId, PlanNode, ProjectionNode, QueryPlan, RecordBatch, SourceNode, StringArray,
    StringViewArray, UInt32Array, UInt64Array, aggregate_group_output_name, reader_projection_indices,
    reader_projection_indices_for_exprs, remap_unqualified_columns,
};

/// The dimension key filter for `source`'s reader predicate. `None` means the
/// source has no predicate, the projection could not be resolved, or the
/// resolver could not read it; `Some(None)` is a proven-empty key set.
pub(super) fn derive_dimension_key_filter(
    source: &LocalParquetSource,
    key_column: &str,
    resolver: &dyn LocalParquetPruningResolver,
) -> Option<Option<DimensionKeyFilter>> {
    let predicate = source.reader_predicate.as_ref()?;
    let projection_indices = reader_projection_indices(source, key_column, predicate)?.into_iter().collect::<Vec<_>>();
    resolver.dimension_key_filter(source, key_column, predicate, &projection_indices)
}

/// The inclusive `i64` range of `expr` over the rows a subtree produces, when
/// the subtree is filters, renaming projections, grouping-only aggregates and
/// sorts over one local Parquet source.
///
/// Filter predicates met on the way down are conjoined with the source's
/// reader predicate and evaluated on the host, and `expr` is rewritten through
/// each projection and aggregate so it names source columns at the leaf. A
/// grouping-only aggregate keeps the value set of its group expressions, so
/// the range below it is the range above it. `None` means the range cannot be
/// derived; `Some(None)` means no row qualified. Join-filter reassociation
/// reads this to bound a scalar it moves across a join.
pub(crate) fn derive_i64_expr_range_from_filtered_subtree(
    plan: &QueryPlan,
    node_id: NodeId,
    expr: &Expr,
    max_rows: u64,
    resolver: &dyn LocalParquetPruningResolver,
) -> Option<Option<KeyRange>> {
    derive_i64_expr_range_from_filtered_subtree_inner(plan, node_id, expr.clone(), None, max_rows, resolver)
}

fn derive_i64_expr_range_from_filtered_subtree_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    expr: Expr,
    predicate: Option<Expr>,
    max_rows: u64,
    resolver: &dyn LocalParquetPruningResolver,
) -> Option<Option<KeyRange>> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => {
            let SourceNode::LocalParquet(source) = source.as_ref() else {
                return None;
            };
            if !resolver.dimension_scan_admitted(source, max_rows) {
                return None;
            }
            let predicate = combine_predicates(source.reader_predicate.clone(), predicate);
            derive_i64_expr_range_from_local_parquet_source(source, &expr, predicate.as_ref(), resolver)
        },
        PlanNode::Filter(filter) => derive_i64_expr_range_from_filtered_subtree_inner(
            plan,
            filter.input,
            expr,
            combine_predicates(predicate, Some(filter.predicate.clone())),
            max_rows,
            resolver,
        ),
        PlanNode::Projection(projection) => {
            let expr = remap_projection_general_expr(&expr, projection)?;
            let predicate = remap_optional_projection_general_expr(predicate, projection)?;
            derive_i64_expr_range_from_filtered_subtree_inner(
                plan,
                projection.input,
                expr,
                predicate,
                max_rows,
                resolver,
            )
        },
        PlanNode::Aggregate(aggregate) => {
            if aggregate.spec.grouping_sets.is_some() || !aggregate.spec.aggr_exprs.is_empty() {
                return None;
            }
            let expr = remap_aggregate_group_general_expr(&expr, aggregate)?;
            let predicate = remap_optional_aggregate_group_general_expr(predicate, aggregate)?;
            derive_i64_expr_range_from_filtered_subtree_inner(
                plan,
                aggregate.input,
                expr,
                predicate,
                max_rows,
                resolver,
            )
        },
        PlanNode::Sort(sort) => {
            derive_i64_expr_range_from_filtered_subtree_inner(plan, sort.input, expr, predicate, max_rows, resolver)
        },
        PlanNode::Join(_)
        | PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

fn remap_optional_projection_general_expr(expr: Option<Expr>, projection: &ProjectionNode) -> Option<Option<Expr>> {
    match expr {
        Some(expr) => remap_projection_general_expr(&expr, projection).map(Some),
        None => Some(None),
    }
}

fn remap_optional_aggregate_group_general_expr(expr: Option<Expr>, aggregate: &AggregateNode) -> Option<Option<Expr>> {
    match expr {
        Some(expr) => remap_aggregate_group_general_expr(&expr, aggregate).map(Some),
        None => Some(None),
    }
}

fn remap_projection_general_expr(expr: &Expr, projection: &ProjectionNode) -> Option<Expr> {
    remap_unqualified_columns(expr, &mut |column| {
        let output = projection.outputs.iter().find(|output| output.field.name() == column.name())?;
        Some(strip_alias(output.expr.clone()))
    })
}

fn remap_aggregate_group_general_expr(expr: &Expr, aggregate: &AggregateNode) -> Option<Expr> {
    remap_unqualified_columns(expr, &mut |column| {
        aggregate
            .spec
            .group_exprs
            .iter()
            .find(|expr| aggregate_group_output_name(expr) == Some(column.name()))
            .cloned()
            .map(strip_alias)
    })
}

fn strip_alias(expr: Expr) -> Expr {
    match expr {
        Expr::Alias { expr, .. } => strip_alias(*expr),
        _ => expr,
    }
}

fn combine_predicates(left: Option<Expr>, right: Option<Expr>) -> Option<Expr> {
    match (left, right) {
        (Some(left), Some(right)) => Some(Expr::binary(left, BinaryOp::And, right)),
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (None, None) => None,
    }
}

fn derive_i64_expr_range_from_local_parquet_source(
    source: &LocalParquetSource,
    expr: &Expr,
    predicate: Option<&Expr>,
    resolver: &dyn LocalParquetPruningResolver,
) -> Option<Option<KeyRange>> {
    let projection_indices =
        reader_projection_indices_for_exprs(source, expr, predicate)?.into_iter().collect::<Vec<_>>();
    resolver.filtered_i64_expr_range(source, expr, predicate, &projection_indices)
}

/// Accumulates the inclusive `i64` range of one expression across decoded
/// batches whose rows satisfy an optional predicate.
///
/// Planning decodes nothing itself. A resolver answering
/// [`LocalParquetPruningResolver::filtered_i64_expr_range`] reads the rows and
/// folds them through this accumulator, so every resolver derives the same
/// range from the same rows.
#[derive(Debug, Clone, Copy, Default)]
pub struct I64ExprRangeAccumulator {
    range: Option<KeyRange>,
}

impl I64ExprRangeAccumulator {
    /// Folds one decoded batch's qualifying `expr` values into the range, or
    /// returns `None` when predicate or expression evaluation fails.
    pub fn observe_batch(&mut self, batch: &RecordBatch, expr: &Expr, predicate: Option<&Expr>) -> Option<()> {
        observe_i64_expr_range_batch(&mut self.range, batch, expr, predicate)
    }

    /// Returns the accumulated range, or `None` when no row qualified.
    pub fn finish(&self) -> Option<KeyRange> {
        self.range
    }
}

fn observe_i64_expr_range_batch(
    range: &mut Option<KeyRange>,
    batch: &RecordBatch,
    expr: &Expr,
    predicate: Option<&Expr>,
) -> Option<()> {
    for row in 0..batch.num_rows() {
        if let Some(predicate) = predicate
            && !eval_bool(predicate, batch, row)?
        {
            continue;
        }
        let value = match eval_scalar(expr, batch, row)? {
            Scalar::Null => continue,
            value => scalar_to_i64(&value)?,
        };
        *range = Some(match *range {
            Some(existing) => KeyRange { min: existing.min.min(value), max: existing.max.max(value) },
            None => KeyRange { min: value, max: value },
        });
    }
    Some(())
}

#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Boolean(bool),
    /// An integer and the type of the column, literal or cast that produced
    /// it. `i128` holds every supported integer type exactly.
    Integer {
        value: i128,
        data_type: IntegerType,
    },
    Float64(f64),
    String(String),
    Null,
}

/// The Arrow type of a [`Scalar::Integer`]. Dates are day or millisecond
/// counts; they compare as integers and take no arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Date32,
    Date64,
}

impl IntegerType {
    fn from_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int8 => Some(Self::Int8),
            DataType::Int16 => Some(Self::Int16),
            DataType::Int32 => Some(Self::Int32),
            DataType::Int64 => Some(Self::Int64),
            DataType::UInt8 => Some(Self::UInt8),
            DataType::UInt16 => Some(Self::UInt16),
            DataType::UInt32 => Some(Self::UInt32),
            DataType::UInt64 => Some(Self::UInt64),
            DataType::Date32 => Some(Self::Date32),
            DataType::Date64 => Some(Self::Date64),
            _ => None,
        }
    }

    fn contains(self, value: i128) -> bool {
        match self {
            Self::Int8 => i8::try_from(value).is_ok(),
            Self::Int16 => i16::try_from(value).is_ok(),
            Self::Int32 | Self::Date32 => i32::try_from(value).is_ok(),
            Self::Int64 | Self::Date64 => i64::try_from(value).is_ok(),
            Self::UInt8 => u8::try_from(value).is_ok(),
            Self::UInt16 => u16::try_from(value).is_ok(),
            Self::UInt32 => u32::try_from(value).is_ok(),
            Self::UInt64 => u64::try_from(value).is_ok(),
        }
    }

    /// Signedness and width rank of an arithmetic integer type; `None` for
    /// dates.
    fn arithmetic_rank(self) -> Option<(bool, u8)> {
        match self {
            Self::Int8 => Some((true, 0)),
            Self::Int16 => Some((true, 1)),
            Self::Int32 => Some((true, 2)),
            Self::Int64 => Some((true, 3)),
            Self::UInt8 => Some((false, 0)),
            Self::UInt16 => Some((false, 1)),
            Self::UInt32 => Some((false, 2)),
            Self::UInt64 => Some((false, 3)),
            Self::Date32 | Self::Date64 => None,
        }
    }

    /// Result type of integer `+`, `-` or `*` over `self` and `other`: the
    /// wider type when both have the same signedness, as the backend infers.
    /// `None` for mixed signedness, whose kernel computes in a type other than
    /// the inferred one, and for dates.
    fn arithmetic_result(self, other: Self) -> Option<Self> {
        let (signed, rank) = self.arithmetic_rank()?;
        let (other_signed, other_rank) = other.arithmetic_rank()?;
        (signed == other_signed).then_some(if rank >= other_rank { self } else { other })
    }

    /// Reduces `value` modulo the type's width into its range, the result of
    /// wrapping arithmetic in this type. `None` for dates.
    fn wrap(self, value: i128) -> Option<i128> {
        match self {
            Self::Int8 => Some(i128::from(value as i8)),
            Self::Int16 => Some(i128::from(value as i16)),
            Self::Int32 => Some(i128::from(value as i32)),
            Self::Int64 => Some(i128::from(value as i64)),
            Self::UInt8 => Some(i128::from(value as u8)),
            Self::UInt16 => Some(i128::from(value as u16)),
            Self::UInt32 => Some(i128::from(value as u32)),
            Self::UInt64 => Some(i128::from(value as u64)),
            Self::Date32 | Self::Date64 => None,
        }
    }
}

fn integer(value: impl Into<i128>, data_type: IntegerType) -> Scalar {
    Scalar::Integer { value: value.into(), data_type }
}

/// Evaluates a predicate for one row; a null result rejects the row.
pub(super) fn eval_bool(expr: &Expr, batch: &RecordBatch, row: usize) -> Option<bool> {
    match eval_scalar(expr, batch, row)? {
        Scalar::Boolean(value) => Some(value),
        Scalar::Null => Some(false),
        _ => None,
    }
}

fn eval_scalar(expr: &Expr, batch: &RecordBatch, row: usize) -> Option<Scalar> {
    match expr {
        Expr::Column(column) => array_scalar(batch.column(batch.schema().index_of(column.name()).ok()?).as_ref(), row),
        Expr::Literal(value) => literal_scalar(value),
        Expr::Alias { expr, .. } => eval_scalar(expr, batch, row),
        Expr::Cast { expr, target_type } => cast_scalar(eval_scalar(expr, batch, row)?, target_type),
        Expr::Binary { left, op, right, output_type } => {
            eval_binary(eval_scalar(left, batch, row)?, *op, eval_scalar(right, batch, row)?, output_type.as_ref())
        },
        Expr::Not(inner) => match eval_scalar(inner, batch, row)? {
            Scalar::Boolean(value) => Some(Scalar::Boolean(!value)),
            Scalar::Null => Some(Scalar::Null),
            _ => None,
        },
        Expr::IsNull(inner) => Some(Scalar::Boolean(matches!(eval_scalar(inner, batch, row)?, Scalar::Null))),
        Expr::IsNotNull(inner) => Some(Scalar::Boolean(!matches!(eval_scalar(inner, batch, row)?, Scalar::Null))),
        Expr::InList { expr, list, negated } => {
            let value = eval_scalar(expr, batch, row)?;
            if matches!(value, Scalar::Null) {
                return Some(Scalar::Null);
            }
            // Runnable plans hold only non-null literals in the list, and
            // `scalar_eq` returns `None` for a NULL item.
            let mut matched = false;
            for item in list {
                if scalar_eq(&value, &eval_scalar(item, batch, row)?)? {
                    matched = true;
                    break;
                }
            }
            Some(Scalar::Boolean(if *negated { !matched } else { matched }))
        },
        Expr::Like { expr, pattern, escape, negated } => match eval_scalar(expr, batch, row)? {
            Scalar::String(value) => {
                let matched = like_matches(&value, pattern, escape.as_deref())?;
                Some(Scalar::Boolean(if *negated { !matched } else { matched }))
            },
            Scalar::Null => Some(Scalar::Null),
            _ => None,
        },
        Expr::TryCast { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
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
        | Expr::DateTrunc { .. } => None,
    }
}

fn array_scalar(array: &dyn Array, row: usize) -> Option<Scalar> {
    if array.is_null(row) {
        return Some(Scalar::Null);
    }
    match array.data_type() {
        DataType::Boolean => Some(Scalar::Boolean(array.as_any().downcast_ref::<BooleanArray>()?.value(row))),
        DataType::Int32 => Some(integer(array.as_any().downcast_ref::<Int32Array>()?.value(row), IntegerType::Int32)),
        DataType::Date32 => {
            Some(integer(array.as_any().downcast_ref::<Date32Array>()?.value(row), IntegerType::Date32))
        },
        DataType::Int64 => Some(integer(array.as_any().downcast_ref::<Int64Array>()?.value(row), IntegerType::Int64)),
        DataType::UInt32 => {
            Some(integer(array.as_any().downcast_ref::<UInt32Array>()?.value(row), IntegerType::UInt32))
        },
        DataType::UInt64 => {
            Some(integer(array.as_any().downcast_ref::<UInt64Array>()?.value(row), IntegerType::UInt64))
        },
        DataType::Float32 => {
            Some(Scalar::Float64(f64::from(array.as_any().downcast_ref::<Float32Array>()?.value(row))))
        },
        DataType::Float64 => Some(Scalar::Float64(array.as_any().downcast_ref::<Float64Array>()?.value(row))),
        DataType::Utf8 => Some(Scalar::String(array.as_any().downcast_ref::<StringArray>()?.value(row).to_string())),
        DataType::LargeUtf8 => {
            Some(Scalar::String(array.as_any().downcast_ref::<LargeStringArray>()?.value(row).to_string()))
        },
        DataType::Utf8View => {
            Some(Scalar::String(array.as_any().downcast_ref::<StringViewArray>()?.value(row).to_string()))
        },
        _ => None,
    }
}

pub(super) fn array_i64_value(array: &dyn Array, row: usize) -> Option<Option<i64>> {
    if array.is_null(row) {
        return Some(None);
    }
    match array.data_type() {
        DataType::Int32 => Some(Some(i64::from(array.as_any().downcast_ref::<Int32Array>()?.value(row)))),
        DataType::Date32 => Some(Some(i64::from(array.as_any().downcast_ref::<Date32Array>()?.value(row)))),
        DataType::Int64 => Some(Some(array.as_any().downcast_ref::<Int64Array>()?.value(row))),
        DataType::UInt32 => Some(Some(i64::from(array.as_any().downcast_ref::<UInt32Array>()?.value(row)))),
        DataType::UInt64 => i64::try_from(array.as_any().downcast_ref::<UInt64Array>()?.value(row)).ok().map(Some),
        _ => None,
    }
}

fn literal_scalar(value: &LiteralValue) -> Option<Scalar> {
    match value {
        LiteralValue::Boolean(value) => Some(Scalar::Boolean(*value)),
        LiteralValue::Int8(value) => Some(integer(*value, IntegerType::Int8)),
        LiteralValue::Int16(value) => Some(integer(*value, IntegerType::Int16)),
        LiteralValue::Int32(value) => Some(integer(*value, IntegerType::Int32)),
        LiteralValue::Date32(value) => Some(integer(*value, IntegerType::Date32)),
        LiteralValue::Date64(value) => Some(integer(*value, IntegerType::Date64)),
        LiteralValue::Int64(value) => Some(integer(*value, IntegerType::Int64)),
        LiteralValue::UInt8(value) => Some(integer(*value, IntegerType::UInt8)),
        LiteralValue::UInt16(value) => Some(integer(*value, IntegerType::UInt16)),
        LiteralValue::UInt32(value) => Some(integer(*value, IntegerType::UInt32)),
        LiteralValue::UInt64(value) => Some(integer(*value, IntegerType::UInt64)),
        LiteralValue::Float32(value) => Some(Scalar::Float64(f64::from(*value))),
        LiteralValue::Float64(value) => Some(Scalar::Float64(*value)),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => Some(Scalar::String(value.clone())),
        LiteralValue::Null(_) => Some(Scalar::Null),
        LiteralValue::Decimal128 { .. }
        | LiteralValue::Time32 { .. }
        | LiteralValue::Time64 { .. }
        | LiteralValue::Timestamp { .. }
        | LiteralValue::Duration { .. }
        | LiteralValue::IntervalMonthDayNano { .. } => None,
    }
}

/// Casts that keep the value exact. An integer target takes only values in
/// its range and floats with no fractional part, because the executor fails
/// or truncates on the others.
fn cast_scalar(value: Scalar, target_type: &DataType) -> Option<Scalar> {
    if matches!(value, Scalar::Null) {
        return Some(Scalar::Null);
    }
    if let Some(data_type) = IntegerType::from_data_type(target_type) {
        let value = match value {
            Scalar::Integer { value, .. } => value,
            Scalar::Float64(value) => exact_i128_from_f64(value)?,
            Scalar::Boolean(_) | Scalar::String(_) | Scalar::Null => return None,
        };
        return data_type.contains(value).then_some(Scalar::Integer { value, data_type });
    }
    match target_type {
        DataType::Float64 => scalar_to_f64(&value).map(Scalar::Float64),
        // An integer rounds to f32 once; going through f64 would round twice.
        DataType::Float32 => match value {
            Scalar::Integer { value, .. } => Some(Scalar::Float64(f64::from(value as f32))),
            Scalar::Float64(value) => Some(Scalar::Float64(f64::from(value as f32))),
            Scalar::Boolean(_) | Scalar::String(_) | Scalar::Null => None,
        },
        DataType::Boolean => match value {
            Scalar::Boolean(value) => Some(Scalar::Boolean(value)),
            _ => None,
        },
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => match value {
            Scalar::String(value) => Some(Scalar::String(value)),
            _ => None,
        },
        _ => None,
    }
}

fn eval_binary(left: Scalar, op: BinaryOp, right: Scalar, output_type: Option<&DataType>) -> Option<Scalar> {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return eval_boolean_binary(left, op, right);
    }
    if matches!(left, Scalar::Null) || matches!(right, Scalar::Null) {
        return Some(Scalar::Null);
    }
    match op {
        BinaryOp::Eq => scalar_eq(&left, &right).map(Scalar::Boolean),
        BinaryOp::NotEq => scalar_eq(&left, &right).map(|value| Scalar::Boolean(!value)),
        BinaryOp::Lt => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering.is_lt())),
        BinaryOp::LtEq => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering.is_le())),
        BinaryOp::Gt => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering.is_gt())),
        BinaryOp::GtEq => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering.is_ge())),
        BinaryOp::Add => eval_wrapping_integer_binary(left, right, output_type, i128::wrapping_add),
        BinaryOp::Subtract => eval_wrapping_integer_binary(left, right, output_type, i128::wrapping_sub),
        BinaryOp::Multiply => eval_wrapping_integer_binary(left, right, output_type, i128::wrapping_mul),
        BinaryOp::Divide
        | BinaryOp::Modulo
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor
        | BinaryOp::ShiftLeft
        | BinaryOp::ShiftRight
        | BinaryOp::And
        | BinaryOp::Or => None,
    }
}

/// Integer `+`, `-` or `*`, wrapped at the width of the result type. The
/// kernel computes in at least that width before narrowing, so the low bits
/// are the exact result's. `i128` wrapping keeps the low 128 bits, which
/// reduces to the same value at 64 bits or less.
fn eval_wrapping_integer_binary(
    left: Scalar,
    right: Scalar,
    output_type: Option<&DataType>,
    op: fn(i128, i128) -> i128,
) -> Option<Scalar> {
    let (
        Scalar::Integer { value: left, data_type: left_type },
        Scalar::Integer { value: right, data_type: right_type },
    ) = (left, right)
    else {
        return None;
    };
    let data_type = left_type.arithmetic_result(right_type)?;
    if output_type.is_some_and(|output_type| IntegerType::from_data_type(output_type) != Some(data_type)) {
        return None;
    }
    let value = data_type.wrap(op(left, right))?;
    Some(Scalar::Integer { value, data_type })
}

/// SQL `AND`/`OR`: a NULL operand yields NULL unless the other operand
/// decides the result (`FALSE AND NULL` is FALSE, `TRUE OR NULL` is TRUE).
fn eval_boolean_binary(left: Scalar, op: BinaryOp, right: Scalar) -> Option<Scalar> {
    let sql_boolean = |value: Scalar| match value {
        Scalar::Boolean(value) => Some(Some(value)),
        Scalar::Null => Some(None),
        _ => None,
    };
    let (left, right) = (sql_boolean(left)?, sql_boolean(right)?);
    let result = match op {
        BinaryOp::And => match (left, right) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        },
        BinaryOp::Or => match (left, right) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
        _ => return None,
    };
    Some(result.map_or(Scalar::Null, Scalar::Boolean))
}

/// Equality of two non-null scalars; `None` when either is NULL.
fn scalar_eq(left: &Scalar, right: &Scalar) -> Option<bool> {
    if let (Scalar::String(left), Scalar::String(right)) = (left, right) {
        return Some(left == right);
    }
    Some(scalar_cmp(left, right)?.is_eq())
}

/// Order of two non-null numeric scalars. Integers compare exactly; an
/// integer against a float compares as `f64`, the type the comparison
/// coerces to. `None` for NaN, NULL and non-numeric operands.
fn scalar_cmp(left: &Scalar, right: &Scalar) -> Option<Ordering> {
    if let (Scalar::Integer { value: left, .. }, Scalar::Integer { value: right, .. }) = (left, right) {
        return Some(left.cmp(right));
    }
    scalar_to_f64(left)?.partial_cmp(&scalar_to_f64(right)?)
}

fn scalar_to_i64(value: &Scalar) -> Option<i64> {
    match value {
        Scalar::Integer { value, .. } => i64::try_from(*value).ok(),
        Scalar::Float64(value) => exact_i128_from_f64(*value).and_then(|value| i64::try_from(value).ok()),
        Scalar::Boolean(_) | Scalar::String(_) | Scalar::Null => None,
    }
}

/// Converts only when the float is an integer that round-trips. A rounded
/// conversion could place a key one step inside or outside a row-group range.
fn exact_i128_from_f64(value: f64) -> Option<i128> {
    // 2^64; every integer type this evaluator holds lies below it.
    const LIMIT: f64 = 18_446_744_073_709_551_616.0;
    if !value.is_finite() || value.fract() != 0.0 || value.abs() >= LIMIT {
        return None;
    }
    let converted = value as i128;
    ((converted as f64) == value).then_some(converted)
}

fn scalar_to_f64(value: &Scalar) -> Option<f64> {
    match value {
        Scalar::Integer { value, .. } => Some(*value as f64),
        Scalar::Float64(value) => Some(*value),
        Scalar::Boolean(_) | Scalar::String(_) | Scalar::Null => None,
    }
}

// Tokenized LIKE patterns let escaped literals and collapsed '%' runs share one matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LikeToken {
    Literal(char),
    AnyOne,
    AnyMany,
}

fn like_matches(value: &str, pattern: &str, escape: Option<&str>) -> Option<bool> {
    let escape = match escape {
        Some(escape) => {
            let mut chars = escape.chars();
            let escape = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Some(escape)
        },
        None => None,
    };
    let tokens = like_tokens(pattern, escape)?;
    let value = value.chars().collect::<Vec<_>>();
    let mut current = vec![false; value.len() + 1];
    current[0] = true;

    for token in tokens {
        let mut next = vec![false; value.len() + 1];
        match token {
            LikeToken::Literal(expected) => {
                for index in 0..value.len() {
                    if current[index] && value[index] == expected {
                        next[index + 1] = true;
                    }
                }
            },
            LikeToken::AnyOne => {
                for index in 0..value.len() {
                    if current[index] {
                        next[index + 1] = true;
                    }
                }
            },
            LikeToken::AnyMany => {
                let mut reachable = false;
                for index in 0..=value.len() {
                    reachable |= current[index];
                    if reachable {
                        next[index] = true;
                    }
                }
            },
        }
        current = next;
    }

    Some(current[value.len()])
}

fn like_tokens(pattern: &str, escape: Option<char>) -> Option<Vec<LikeToken>> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if Some(ch) == escape {
            let escaped = chars.next()?;
            if !matches!(escaped, '%' | '_') && Some(escaped) != escape {
                return None;
            }
            tokens.push(LikeToken::Literal(escaped));
            continue;
        }
        match ch {
            '%' if !matches!(tokens.last(), Some(LikeToken::AnyMany)) => tokens.push(LikeToken::AnyMany),
            '%' => {},
            '_' => tokens.push(LikeToken::AnyOne),
            _ => tokens.push(LikeToken::Literal(ch)),
        }
    }
    Some(tokens)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, BooleanArray, Int32Array, UInt32Array, UInt64Array},
        datatypes::{Field, Schema},
        record_batch::RecordBatch,
    };

    use super::{BinaryOp, Expr, LiteralValue, eval_bool, eval_scalar};

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let schema = Schema::new(
            columns.iter().map(|(name, array)| Field::new(*name, array.data_type().clone(), true)).collect::<Vec<_>>(),
        );
        RecordBatch::try_new(Arc::new(schema), columns.into_iter().map(|(_, array)| array).collect()).unwrap()
    }

    fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::binary(left, op, right)
    }

    #[test]
    fn integer_arithmetic_wraps_at_the_operand_type_width() {
        let batch = batch(vec![
            ("u1", Arc::new(UInt32Array::from(vec![3])) as ArrayRef),
            ("u2", Arc::new(UInt32Array::from(vec![5])) as ArrayRef),
            ("i", Arc::new(Int32Array::from(vec![i32::MAX])) as ArrayRef),
        ]);
        // 3 - 5 in UInt32 is 4294967294, so the row passes `> 10`.
        let unsigned = binary(
            binary(Expr::column("u1"), BinaryOp::Subtract, Expr::column("u2")),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::UInt32(10)),
        );
        assert_eq!(eval_bool(&unsigned, &batch, 0), Some(true));
        let signed = binary(
            binary(Expr::column("i"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(1))),
            BinaryOp::Lt,
            Expr::literal(LiteralValue::Int32(0)),
        );
        assert_eq!(eval_bool(&signed, &batch, 0), Some(true));
        // Int32 + Int64 is Int64 arithmetic, so it does not wrap at 32 bits.
        let widened = binary(
            binary(Expr::column("i"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(1))),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::Int64(0)),
        );
        assert_eq!(eval_bool(&widened, &batch, 0), Some(true));
        // The kernel computes UInt32 - Int32 in a type other than the inferred Int64.
        let mixed_sign = binary(Expr::column("u1"), BinaryOp::Subtract, Expr::column("i"));
        assert_eq!(eval_scalar(&mixed_sign, &batch, 0), None);
    }

    #[test]
    fn boolean_operators_keep_null_below_the_predicate_root() {
        let batch = batch(vec![
            ("p", Arc::new(BooleanArray::from(vec![None, Some(false)])) as ArrayRef),
            ("q", Arc::new(BooleanArray::from(vec![Some(true), None])) as ArrayRef),
        ]);
        let and_is_null = Expr::IsNull(Box::new(binary(Expr::column("p"), BinaryOp::And, Expr::column("q"))));
        let or_is_null = Expr::IsNull(Box::new(binary(Expr::column("p"), BinaryOp::Or, Expr::column("q"))));
        // NULL AND TRUE is NULL; FALSE AND NULL is FALSE.
        assert_eq!(eval_bool(&and_is_null, &batch, 0), Some(true));
        assert_eq!(eval_bool(&and_is_null, &batch, 1), Some(false));
        // NULL OR TRUE is TRUE; FALSE OR NULL is NULL.
        assert_eq!(eval_bool(&or_is_null, &batch, 0), Some(false));
        assert_eq!(eval_bool(&or_is_null, &batch, 1), Some(true));
        let not_and = Expr::logical_not(binary(Expr::column("p"), BinaryOp::And, Expr::column("q")));
        assert_eq!(eval_bool(&not_and, &batch, 0), Some(false), "NOT NULL rejects the row");
    }

    #[test]
    fn uint64_values_above_i64_max_compare_exactly() {
        let batch = batch(vec![("x", Arc::new(UInt64Array::from(vec![u64::MAX])) as ArrayRef)]);
        let predicate = binary(Expr::column("x"), BinaryOp::Gt, Expr::literal(LiteralValue::UInt64(u64::MAX - 1)));
        assert_eq!(eval_bool(&predicate, &batch, 0), Some(true));
    }
}
