//! Selection-predicate factoring for copies of one aggregate that differ only
//! in what their sources select.
//!
//! A member's selection predicate on one local Parquet source is a list of
//! conjuncts. [`factor_selections`] splits the members' lists into common
//! conjuncts, which every member holds or implies, and each member's
//! differing conjuncts. The widened predicate is the common conjuncts AND the
//! OR of the members' differing sets; the OR is omitted when a member has no
//! differing conjunct, since that member needs every row the common part
//! admits. Each member's rows are the widened rows that satisfy its differing
//! conjuncts. [`map_residual`] restates those conjuncts over aggregate group
//! outputs: filtering groups by a predicate over group keys keeps the groups
//! that filtering the input rows first would produce, for DISTINCT and for
//! grouped aggregation alike.
//!
//! Implication covers one column compared with a literal of the column's own
//! type (`< <= > >= = !=`). Floating-point columns are excluded because NaN
//! makes `NOT (x >= t)` differ from `x < t`. A NULL column value makes a
//! conjunct and its restatement NULL, so the row or group is dropped on both
//! sides.

use std::collections::BTreeSet;

use arrow::datatypes::{DataType, Schema};

use super::lineage::substitute_unqualified_columns;
use crate::expr::{BinaryOp, Expr, LiteralValue};

/// `p0 AND (p1 AND (... AND pn))`, or `None` for no predicates. The shape is
/// fixed because predicate text takes part in signatures.
pub(super) fn conjunction(mut predicates: Vec<Expr>) -> Option<Expr> {
    let first = predicates.pop()?;
    Some(predicates.into_iter().fold(first, |right, left| Expr::binary(left, BinaryOp::And, right)))
}

/// `p0 OR (p1 OR (... OR pn))`, or `None` for no predicates.
pub(super) fn disjunction(mut predicates: Vec<Expr>) -> Option<Expr> {
    let first = predicates.pop()?;
    Some(predicates.into_iter().fold(first, |right, left| Expr::binary(left, BinaryOp::Or, right)))
}

/// The conjuncts of an `AND` tree without a declared output type, in order.
pub(super) fn split_conjuncts(predicate: Expr) -> Vec<Expr> {
    match predicate {
        Expr::Binary { left, op: BinaryOp::And, right, output_type: None } => {
            let mut conjuncts = split_conjuncts(*left);
            conjuncts.extend(split_conjuncts(*right));
            conjuncts
        },
        other => vec![other],
    }
}

/// Members' selections on one source after factoring.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FactoredSelections {
    /// Predicate the shared scan applies, or `None` when no member selects.
    pub(super) widened: Option<Expr>,
    /// Each member's conjuncts outside the common part, in member order.
    pub(super) differing: Vec<Vec<Expr>>,
}

/// Factors the members' conjunct lists on one source. `schema` types the
/// source columns the conjuncts read.
///
/// A conjunct is common when every member holds it or holds a comparison
/// that implies it. Common conjuncts keep their first occurrence in member
/// order, so members that agree up to text factor the same as before
/// implication existed.
pub(super) fn factor_selections(members: &[Vec<Expr>], schema: &Schema) -> FactoredSelections {
    let member_keys = members
        .iter()
        .map(|member| member.iter().map(ToString::to_string).collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();
    let is_common = |conjunct: &Expr| {
        let key = conjunct.to_string();
        members
            .iter()
            .zip(&member_keys)
            .all(|(member, keys)| keys.contains(&key) || implied_by(conjunct, member, schema))
    };
    let mut common = Vec::new();
    let mut common_keys = BTreeSet::new();
    for conjunct in members.iter().flatten() {
        if !common_keys.contains(&conjunct.to_string()) && is_common(conjunct) {
            common_keys.insert(conjunct.to_string());
            common.push(conjunct.clone());
        }
    }
    let differing = members
        .iter()
        .map(|member| {
            member.iter().filter(|conjunct| !common_keys.contains(&conjunct.to_string())).cloned().collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut widened = common;
    if differing.iter().all(|conjuncts| !conjuncts.is_empty())
        && let Some(union) =
            disjunction(differing.iter().filter_map(|conjuncts| conjunction(conjuncts.clone())).collect())
    {
        widened.push(union);
    }
    FactoredSelections { widened: conjunction(widened), differing }
}

/// How one aggregate group output is computed from the columns of the source
/// whose selection is factored.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum GroupKeyDefinition {
    /// The output is this source column unchanged.
    Column(String),
    /// The output is `CAST(column)` to a wider integer type that represents
    /// every value of the column's type.
    WideningCast { column: String, target: DataType },
    /// The output is this boolean expression over source columns.
    Boolean(Expr),
}

/// One group output of the aggregate and its definition.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct GroupKeyOutput {
    pub(super) output: String,
    pub(super) definition: GroupKeyDefinition,
}

/// Restates `conjuncts` over group outputs, or `None` when one of them has
/// no restatement. An empty list restates to `Some(None)`: the member reads
/// the shared result unfiltered.
///
/// A conjunct whose columns are all identity group outputs is re-applied
/// with the output names. A comparison of a column with a literal is
/// re-applied on a widening integer cast of that column with the widened
/// literal, or becomes a boolean group output `k` defined as that comparison
/// (`k`) or its negation (`NOT k`).
pub(super) fn map_residual(conjuncts: &[Expr], outputs: &[GroupKeyOutput], schema: &Schema) -> Option<Option<Expr>> {
    conjuncts
        .iter()
        .map(|conjunct| map_residual_conjunct(conjunct, outputs, schema))
        .collect::<Option<Vec<_>>>()
        .map(conjunction)
}

fn map_residual_conjunct(conjunct: &Expr, outputs: &[GroupKeyOutput], schema: &Schema) -> Option<Expr> {
    let identity_output = |name: &str| {
        outputs.iter().find_map(|output| match &output.definition {
            GroupKeyDefinition::Column(column) if column == name => Some(output.output.as_str()),
            _ => None,
        })
    };
    if conjunct.columns().all(|column| column.qualifier().is_none() && identity_output(column.name()).is_some()) {
        return Some(substitute_unqualified_columns(conjunct.clone(), &|name| {
            identity_output(name).filter(|output| *output != name).map(Expr::column)
        }));
    }
    let (negated, comparison) = match conjunct {
        Expr::Not(inner) => (true, Comparison::of(inner, schema)?),
        other => (false, Comparison::of(other, schema)?),
    };
    for output in outputs {
        match &output.definition {
            GroupKeyDefinition::WideningCast { column, target } if !negated && *column == comparison.column => {
                if let Some(literal) = widened_integer_literal(comparison.literal, target) {
                    return Some(Expr::binary(Expr::column(&output.output), comparison.op, Expr::literal(literal)));
                }
            },
            GroupKeyDefinition::Boolean(definition) => {
                let Some(definition) = Comparison::of(definition, schema) else { continue };
                if definition.column != comparison.column || definition.literal != comparison.literal {
                    continue;
                }
                let op = if negated { negated_comparison(comparison.op) } else { Some(comparison.op) };
                if op == Some(definition.op) {
                    return Some(Expr::column(&output.output));
                }
                if op.and_then(negated_comparison) == Some(definition.op) {
                    return Some(Expr::Not(Box::new(Expr::column(&output.output))));
                }
            },
            _ => {},
        }
    }
    None
}

/// Whether `CAST(value AS to)` maps every value of integer type `from` to a
/// distinct value of integer type `to` in the same order.
pub(super) fn is_lossless_integer_widening(from: &DataType, to: &DataType) -> bool {
    match (integer_width(from), integer_width(to)) {
        (Some((from_signed, from_bits)), Some((to_signed, to_bits))) => {
            to_bits > from_bits && (from_signed == to_signed || to_signed)
        },
        _ => false,
    }
}

fn integer_width(data_type: &DataType) -> Option<(bool, u32)> {
    Some(match data_type {
        DataType::Int8 => (true, 8),
        DataType::Int16 => (true, 16),
        DataType::Int32 => (true, 32),
        DataType::Int64 => (true, 64),
        DataType::UInt8 => (false, 8),
        DataType::UInt16 => (false, 16),
        DataType::UInt32 => (false, 32),
        DataType::UInt64 => (false, 64),
        _ => return None,
    })
}

fn widened_integer_literal(literal: &LiteralValue, target: &DataType) -> Option<LiteralValue> {
    if !is_lossless_integer_widening(&literal.data_type(), target) {
        return None;
    }
    let value = match literal {
        LiteralValue::Int8(value) => i128::from(*value),
        LiteralValue::Int16(value) => i128::from(*value),
        LiteralValue::Int32(value) => i128::from(*value),
        LiteralValue::UInt8(value) => i128::from(*value),
        LiteralValue::UInt16(value) => i128::from(*value),
        LiteralValue::UInt32(value) => i128::from(*value),
        _ => return None,
    };
    Some(match target {
        DataType::Int16 => LiteralValue::Int16(i16::try_from(value).ok()?),
        DataType::Int32 => LiteralValue::Int32(i32::try_from(value).ok()?),
        DataType::Int64 => LiteralValue::Int64(i64::try_from(value).ok()?),
        DataType::UInt16 => LiteralValue::UInt16(u16::try_from(value).ok()?),
        DataType::UInt32 => LiteralValue::UInt32(u32::try_from(value).ok()?),
        DataType::UInt64 => LiteralValue::UInt64(u64::try_from(value).ok()?),
        _ => return None,
    })
}

/// `column op literal` over an unqualified column whose type is the literal's
/// type and has an [`OrderKey`].
struct Comparison<'a> {
    column: &'a str,
    op: BinaryOp,
    literal: &'a LiteralValue,
}

impl<'a> Comparison<'a> {
    fn of(expr: &'a Expr, schema: &Schema) -> Option<Self> {
        let Expr::Binary { left, op, right, .. } = expr else { return None };
        let (column, op, literal) = match (left.as_ref(), right.as_ref()) {
            (Expr::Column(column), Expr::Literal(literal)) => (column, *op, literal),
            (Expr::Literal(literal), Expr::Column(column)) => (column, flipped_comparison(*op)?, literal),
            _ => return None,
        };
        negated_comparison(op)?;
        let field = schema.field_with_name(column.name()).ok()?;
        if column.qualifier().is_some() || *field.data_type() != literal.data_type() || OrderKey::of(literal).is_none()
        {
            return None;
        }
        Some(Self { column: column.name(), op, literal })
    }
}

fn flipped_comparison(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::GtEq => BinaryOp::LtEq,
        BinaryOp::Eq => BinaryOp::Eq,
        BinaryOp::NotEq => BinaryOp::NotEq,
        _ => return None,
    })
}

fn negated_comparison(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Lt => BinaryOp::GtEq,
        BinaryOp::LtEq => BinaryOp::Gt,
        BinaryOp::Gt => BinaryOp::LtEq,
        BinaryOp::GtEq => BinaryOp::Lt,
        BinaryOp::Eq => BinaryOp::NotEq,
        BinaryOp::NotEq => BinaryOp::Eq,
        _ => return None,
    })
}

/// Literal values of one type in their SQL order. Floating-point literals have
/// no key: NaN is unordered, so `NOT (x >= t)` and `x < t` admit different
/// rows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum OrderKey {
    Integer(i128),
    Text(String),
}

impl OrderKey {
    fn of(literal: &LiteralValue) -> Option<Self> {
        Some(match literal {
            LiteralValue::Int8(value) => Self::Integer(i128::from(*value)),
            LiteralValue::Int16(value) => Self::Integer(i128::from(*value)),
            LiteralValue::Int32(value) | LiteralValue::Date32(value) => Self::Integer(i128::from(*value)),
            LiteralValue::Int64(value) | LiteralValue::Date64(value) => Self::Integer(i128::from(*value)),
            LiteralValue::UInt8(value) => Self::Integer(i128::from(*value)),
            LiteralValue::UInt16(value) => Self::Integer(i128::from(*value)),
            LiteralValue::UInt32(value) => Self::Integer(i128::from(*value)),
            LiteralValue::UInt64(value) => Self::Integer(i128::from(*value)),
            LiteralValue::Timestamp { raw, .. } | LiteralValue::Duration { raw, .. } => Self::Integer(i128::from(*raw)),
            LiteralValue::Time32 { raw, .. } => Self::Integer(i128::from(*raw)),
            LiteralValue::Time64 { raw, .. } => Self::Integer(i128::from(*raw)),
            LiteralValue::Decimal128 { raw, .. } => Self::Integer(*raw),
            LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => Self::Text(value.clone()),
            LiteralValue::Float32(_) | LiteralValue::Float64(_) => return None,
            _ => return None,
        })
    }
}

/// One end of the value interval a comparison admits; `inclusive` admits
/// the end value itself.
struct Bound {
    value: OrderKey,
    inclusive: bool,
}

/// Lower and upper ends of the interval a comparison other than `!=` admits;
/// `None` is unbounded.
fn interval(comparison: &Comparison<'_>) -> (Option<Bound>, Option<Bound>) {
    let value = OrderKey::of(comparison.literal).expect("a comparison literal has an order key");
    let bound = |inclusive| Some(Bound { value: value.clone(), inclusive });
    match comparison.op {
        BinaryOp::Lt => (None, bound(false)),
        BinaryOp::LtEq => (None, bound(true)),
        BinaryOp::Gt => (bound(false), None),
        BinaryOp::GtEq => (bound(true), None),
        _ => (bound(true), bound(true)),
    }
}

/// Whether every value `inner` admits is admitted by `outer`, for two
/// comparisons on one column.
fn admits_subset(inner: &Comparison<'_>, outer: &Comparison<'_>) -> bool {
    let inner_value = OrderKey::of(inner.literal).expect("a comparison literal has an order key");
    let outer_value = OrderKey::of(outer.literal).expect("a comparison literal has an order key");
    if outer.op == BinaryOp::NotEq {
        // `inner` must exclude the one value `outer` rejects.
        return match inner.op {
            BinaryOp::Eq => inner_value != outer_value,
            BinaryOp::NotEq => inner_value == outer_value,
            BinaryOp::Lt => outer_value >= inner_value,
            BinaryOp::LtEq => outer_value > inner_value,
            BinaryOp::Gt => outer_value <= inner_value,
            BinaryOp::GtEq => outer_value < inner_value,
            _ => false,
        };
    }
    if inner.op == BinaryOp::NotEq {
        return false;
    }
    let (inner_lower, inner_upper) = interval(inner);
    let (outer_lower, outer_upper) = interval(outer);
    let within = |inner: Option<Bound>, outer: Option<Bound>, lower: bool| match (inner, outer) {
        (_, None) => true,
        (None, Some(_)) => false,
        (Some(inner), Some(outer)) => {
            let tighter = if lower { inner.value > outer.value } else { inner.value < outer.value };
            tighter || (inner.value == outer.value && (outer.inclusive || !inner.inclusive))
        },
    };
    within(inner_lower, outer_lower, true) && within(inner_upper, outer_upper, false)
}

/// Whether some comparison in `set` admits only values `conjunct` admits.
fn implied_by(conjunct: &Expr, set: &[Expr], schema: &Schema) -> bool {
    let Some(conjunct) = Comparison::of(conjunct, schema) else {
        return false;
    };
    set.iter().any(|expr| {
        Comparison::of(expr, schema)
            .is_some_and(|comparison| comparison.column == conjunct.column && admits_subset(&comparison, &conjunct))
    })
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{Field, TimeUnit};

    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("t", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("x", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
        ])
    }

    fn compare(column: &str, op: BinaryOp, literal: LiteralValue) -> Expr {
        Expr::binary(Expr::column(column), op, Expr::literal(literal))
    }

    fn k(op: BinaryOp, value: i32) -> Expr {
        compare("k", op, LiteralValue::Int32(value))
    }

    fn t(op: BinaryOp, raw: i64) -> Expr {
        compare("t", op, LiteralValue::Timestamp { raw, unit: TimeUnit::Millisecond })
    }

    #[test]
    fn a_tighter_range_implies_the_wider_bound() {
        let schema = schema();
        assert!(implied_by(&t(BinaryOp::GtEq, 100), &[t(BinaryOp::GtEq, 200)], &schema));
        assert!(implied_by(&t(BinaryOp::GtEq, 100), &[t(BinaryOp::Gt, 100)], &schema));
        assert!(!implied_by(&t(BinaryOp::Gt, 100), &[t(BinaryOp::GtEq, 100)], &schema));
        assert!(!implied_by(&t(BinaryOp::GtEq, 100), &[t(BinaryOp::Lt, 200)], &schema));
        assert!(implied_by(&k(BinaryOp::NotEq, 0), &[k(BinaryOp::Gt, 0)], &schema));
        assert!(!implied_by(&k(BinaryOp::NotEq, 5), &[k(BinaryOp::Gt, 0)], &schema));
        assert!(implied_by(&k(BinaryOp::Lt, 10), &[k(BinaryOp::Eq, 3)], &schema));
        // A literal of another type and a predicate over another column imply nothing.
        assert!(!implied_by(&k(BinaryOp::GtEq, 0), &[compare("k", BinaryOp::Gt, LiteralValue::Int64(5))], &schema));
        assert!(!implied_by(&k(BinaryOp::GtEq, 0), &[t(BinaryOp::Gt, 5)], &schema));
    }

    #[test]
    fn floating_columns_take_no_part_in_implication_or_negation() {
        let schema = schema();
        let x = |op, value| compare("x", op, LiteralValue::Float64(value));
        assert!(!implied_by(&x(BinaryOp::GtEq, 1.0), &[x(BinaryOp::GtEq, 2.0)], &schema));
        let late = GroupKeyOutput {
            output: "late".to_string(),
            definition: GroupKeyDefinition::Boolean(x(BinaryOp::GtEq, 1.0)),
        };
        assert_eq!(map_residual(&[x(BinaryOp::Lt, 1.0)], &[late], &schema), None);
    }

    #[test]
    fn period_conjuncts_factor_into_one_range_and_restate_over_the_period_key() {
        let schema = schema();
        // The after copy lost `t >= 100` because `t >= 200` implies it.
        let before = vec![t(BinaryOp::GtEq, 100), t(BinaryOp::Lt, 300), t(BinaryOp::Lt, 200)];
        let after = vec![t(BinaryOp::GtEq, 200), t(BinaryOp::Lt, 300)];
        let all = vec![t(BinaryOp::GtEq, 100), t(BinaryOp::Lt, 300)];

        let factored = factor_selections(&[before, after, all], &schema);

        assert_eq!(factored.widened, conjunction(vec![t(BinaryOp::GtEq, 100), t(BinaryOp::Lt, 300)]));
        assert_eq!(factored.differing, [vec![t(BinaryOp::Lt, 200)], vec![t(BinaryOp::GtEq, 200)], vec![]]);
        let late = [GroupKeyOutput {
            output: "late".to_string(),
            definition: GroupKeyDefinition::Boolean(t(BinaryOp::GtEq, 200)),
        }];
        assert_eq!(
            map_residual(&factored.differing[0], &late, &schema),
            Some(Some(Expr::Not(Box::new(Expr::column("late")))))
        );
        assert_eq!(map_residual(&factored.differing[1], &late, &schema), Some(Some(Expr::column("late"))));
        assert_eq!(map_residual(&factored.differing[2], &late, &schema), Some(None));
        let negated = Expr::Not(Box::new(t(BinaryOp::GtEq, 200)));
        assert_eq!(map_residual(&[negated], &late, &schema), Some(Some(Expr::Not(Box::new(Expr::column("late"))))));
    }

    #[test]
    fn widening_casts_restate_integer_comparisons_and_reject_narrowing() {
        let schema = schema();
        let dst = |target| {
            [GroupKeyOutput {
                output: "dst".to_string(),
                definition: GroupKeyDefinition::WideningCast { column: "k".to_string(), target },
            }]
        };
        assert_eq!(
            map_residual(&[k(BinaryOp::Gt, 0), k(BinaryOp::NotEq, 7)], &dst(DataType::Int64), &schema),
            Some(conjunction(vec![
                compare("dst", BinaryOp::Gt, LiteralValue::Int64(0)),
                compare("dst", BinaryOp::NotEq, LiteralValue::Int64(7)),
            ]))
        );
        assert!(!is_lossless_integer_widening(&DataType::Int32, &DataType::UInt64));
        assert!(!is_lossless_integer_widening(&DataType::Int32, &DataType::Int32));
        assert_eq!(map_residual(&[k(BinaryOp::Gt, 0)], &dst(DataType::Int16), &schema), None);
    }

    #[test]
    fn identity_outputs_restate_any_predicate_and_other_columns_do_not() {
        let schema = schema();
        let outputs =
            [GroupKeyOutput { output: "key".to_string(), definition: GroupKeyDefinition::Column("s".to_string()) }];
        let like =
            Expr::Like { expr: Box::new(Expr::column("s")), pattern: "a%".to_string(), escape: None, negated: false };
        let renamed =
            Expr::Like { expr: Box::new(Expr::column("key")), pattern: "a%".to_string(), escape: None, negated: false };
        assert_eq!(map_residual(&[like], &outputs, &schema), Some(Some(renamed)));
        assert_eq!(map_residual(&[k(BinaryOp::Gt, 0)], &outputs, &schema), None);
    }

    #[test]
    fn members_that_agree_up_to_text_factor_without_implication() {
        let schema = schema();
        let shared = compare("s", BinaryOp::Eq, LiteralValue::Utf8("a".to_string()));
        let first = vec![shared.clone(), k(BinaryOp::Eq, 1)];
        let second = vec![k(BinaryOp::Eq, 2), shared.clone()];

        let factored = factor_selections(&[first, second], &schema);

        assert_eq!(
            factored.widened,
            conjunction(vec![shared, disjunction(vec![k(BinaryOp::Eq, 1), k(BinaryOp::Eq, 2)]).unwrap()])
        );
        assert_eq!(factored.differing, [vec![k(BinaryOp::Eq, 1)], vec![k(BinaryOp::Eq, 2)]]);
    }
}
