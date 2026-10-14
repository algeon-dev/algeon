use datafusion::sql::sqlparser::{
    ast::{
        Expr as SqlExpr, Ident, OrderByKind, Query as SqlQuery, Select, SelectItem, SetExpr, Statement, TableFactor,
        Value, ValueWithSpan,
    },
    dialect::GenericDialect,
    parser::Parser,
};

use super::{
    CanonicalField, DataType, ParityComparisonMode, ParityFloatTolerance, ParityStatus, RecordBatch, Schema,
    collect_canonical_rows, compare_canonical_rows, fingerprint_rows, fingerprint_schema_only,
};

/// Result of comparing CPU and native Arrow outputs.
#[derive(Debug, Clone)]
pub struct ParityComparisonResult {
    /// Match, mismatch, or skip status.
    pub status: ParityStatus,
    /// Row-order and float-tolerance mode used for comparison.
    pub mode: ParityComparisonMode,
    /// Mismatch or skip detail when available.
    pub detail: Option<String>,
    /// Stable diagnostic hash of canonical CPU output.
    pub cpu_fingerprint: String,
    /// Stable diagnostic hash of canonical native output.
    pub native_fingerprint: String,
}

/// Compares CPU and native Arrow output using TPC-DS parity rules.
///
/// Schema differences are reported before row comparison. When `ordered` is
/// false, canonical rows are sorted before comparison. Float tolerance applies
/// only to float columns and is recorded in the returned comparison mode.
pub fn compare_record_batches(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordered: bool,
    tolerance: ParityFloatTolerance,
) -> ParityComparisonResult {
    compare_record_batches_with_options(
        cpu_schema,
        cpu_batches,
        native_schema,
        native_batches,
        &if ordered { ParityRowOrdering::Ordered } else { ParityRowOrdering::Unordered },
        tolerance,
        ParityCompareOptions::strict_native(),
    )
}

/// Schema comparison switches for native-vs-native and cross-engine oracles.
#[derive(Debug, Clone, Copy)]
pub(super) struct ParityCompareOptions {
    pub(super) allow_decimal_precision_mismatch: bool,
}

impl ParityCompareOptions {
    pub(super) const fn strict_native() -> Self {
        Self { allow_decimal_precision_mismatch: false }
    }

    pub(super) const fn cross_engine_oracle() -> Self {
        Self { allow_decimal_precision_mismatch: true }
    }
}

pub(super) enum ParityRowOrdering {
    Unordered,
    Ordered,
    OrderBy(Vec<ParitySortKey>),
}

pub(super) struct ParitySortKey {
    column: usize,
    descending: bool,
}

pub(super) fn sql_row_ordering(sql: &str, schema: Option<&Schema>) -> ParityRowOrdering {
    let Some(query) = parse_query(sql) else {
        return ParityRowOrdering::Ordered;
    };
    if !query.pipe_operators.is_empty() {
        return ParityRowOrdering::Ordered;
    }
    if query.order_by.is_none() {
        return ParityRowOrdering::Unordered;
    }
    schema
        .and_then(|schema| order_by_keys(&query, schema))
        .map_or(ParityRowOrdering::Ordered, ParityRowOrdering::OrderBy)
}

pub(super) fn compare_record_batches_with_options(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordering: &ParityRowOrdering,
    tolerance: ParityFloatTolerance,
    options: ParityCompareOptions,
) -> ParityComparisonResult {
    let schema_diff = compare_schemas(cpu_schema, native_schema, options);
    let float_tolerant = tolerance.is_enabled() && schema_has_float_column(cpu_schema);
    let mode = ParityComparisonMode::select(!matches!(ordering, ParityRowOrdering::Unordered), float_tolerant);

    if let Some(detail) = schema_diff {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(detail),
            cpu_fingerprint: fingerprint_schema_only(cpu_schema),
            native_fingerprint: fingerprint_schema_only(native_schema),
        };
    }

    let cpu_rows = match collect_canonical_rows(cpu_schema, cpu_batches) {
        Ok(rows) => rows,
        Err(detail) => {
            return ParityComparisonResult {
                status: ParityStatus::NotCompared,
                mode,
                detail: Some(format!("cpu canonicalization failed: {detail}")),
                cpu_fingerprint: fingerprint_schema_only(cpu_schema),
                native_fingerprint: fingerprint_schema_only(native_schema),
            };
        },
    };
    let mut native_rows = match collect_canonical_rows(native_schema, native_batches) {
        Ok(rows) => rows,
        Err(detail) => {
            return ParityComparisonResult {
                status: ParityStatus::NotCompared,
                mode,
                detail: Some(format!("native canonicalization failed: {detail}")),
                cpu_fingerprint: fingerprint_rows(cpu_schema, &cpu_rows),
                native_fingerprint: fingerprint_schema_only(native_schema),
            };
        },
    };
    let mut cpu_rows = cpu_rows;
    match ordering {
        ParityRowOrdering::Unordered => {
            cpu_rows.sort();
            native_rows.sort();
        },
        ParityRowOrdering::Ordered => {},
        ParityRowOrdering::OrderBy(keys) => {
            sort_tied_rows(&mut cpu_rows, keys);
            sort_tied_rows(&mut native_rows, keys);
        },
    }

    let cpu_fingerprint = fingerprint_rows(cpu_schema, &cpu_rows);
    let native_fingerprint = fingerprint_rows(cpu_schema, &native_rows);

    if cpu_rows.len() != native_rows.len() {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(format!("row count differs: cpu={} native={}", cpu_rows.len(), native_rows.len())),
            cpu_fingerprint,
            native_fingerprint,
        };
    }

    for (index, (cpu_row, native_row)) in cpu_rows.iter().zip(native_rows.iter()).enumerate() {
        if let Some(detail) =
            compare_canonical_rows(cpu_schema, cpu_row, native_row, tolerance, float_tolerant, options)
        {
            return ParityComparisonResult {
                status: ParityStatus::Mismatched,
                mode,
                detail: Some(format!("row {index}: {detail}")),
                cpu_fingerprint,
                native_fingerprint,
            };
        }
    }

    if float_tolerant
        && let ParityRowOrdering::OrderBy(keys) = ordering
        && let Some((row, column)) = float_key_inversion(&native_rows, keys)
    {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(format!(
                "row {row}: ORDER BY column {column} ({}) is out of order",
                cpu_schema.field(column).name()
            )),
            cpu_fingerprint,
            native_fingerprint,
        };
    }

    ParityComparisonResult { status: ParityStatus::Matched, mode, detail: None, cpu_fingerprint, native_fingerprint }
}

fn sort_tied_rows(rows: &mut [Vec<CanonicalField>], keys: &[ParitySortKey]) {
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        // Approximate float equality is not transitive and cannot define a tie group.
        while end < rows.len()
            && keys.iter().all(|key| {
                !matches!(rows[start][key.column], CanonicalField::Unsupported(_))
                    && rows[start][key.column] == rows[end][key.column]
            })
        {
            end += 1;
        }
        rows[start..end].sort();
        start = end;
    }
}

fn float_key_inversion(rows: &[Vec<CanonicalField>], keys: &[ParitySortKey]) -> Option<(usize, usize)> {
    // Value comparison fixes non-float key sequences exactly. Float tolerance can still admit an inversion.
    for (index, pair) in rows.windows(2).enumerate() {
        for key in keys {
            let before = &pair[0][key.column];
            let after = &pair[1][key.column];
            if matches!(before, CanonicalField::Unsupported(_)) {
                break;
            }
            if before == after {
                continue;
            }
            if let (CanonicalField::Float { width, bits: before }, CanonicalField::Float { bits: after, .. }) =
                (before, after)
            {
                let order = match width {
                    32 => f32::from_bits(*before as u32).total_cmp(&f32::from_bits(*after as u32)),
                    _ => f64::from_bits(*before).total_cmp(&f64::from_bits(*after)),
                };
                if (key.descending && order.is_lt()) || (!key.descending && order.is_gt()) {
                    return Some((index + 1, key.column));
                }
            }
            break;
        }
    }
    None
}

pub(super) fn compare_schemas(cpu: &Schema, native: &Schema, options: ParityCompareOptions) -> Option<String> {
    if cpu.fields().len() != native.fields().len() {
        return Some(format!(
            "schema field count differs: cpu={} native={}",
            cpu.fields().len(),
            native.fields().len()
        ));
    }
    for (index, (cpu_field, native_field)) in cpu.fields().iter().zip(native.fields().iter()).enumerate() {
        if !logical_data_types_compatible(cpu_field.data_type(), native_field.data_type(), options) {
            return Some(format!(
                "column {} ({}/{}) type differs: cpu={:?} native={:?}",
                index,
                cpu_field.name(),
                native_field.name(),
                cpu_field.data_type(),
                native_field.data_type(),
            ));
        }
    }
    None
}

pub(super) fn logical_data_types_compatible(cpu: &DataType, native: &DataType, options: ParityCompareOptions) -> bool {
    if cpu == native {
        return true;
    }
    if options.allow_decimal_precision_mismatch {
        match (cpu, native) {
            (DataType::Decimal128(_, cpu_scale), DataType::Decimal128(_, native_scale))
            | (DataType::Decimal256(_, cpu_scale), DataType::Decimal256(_, native_scale))
                if cpu_scale == native_scale =>
            {
                return true;
            },
            _ => {},
        }
    }
    matches!(
        (cpu, native),
        (DataType::Utf8, DataType::Utf8View)
            | (DataType::Utf8View, DataType::Utf8)
            | (DataType::LargeUtf8, DataType::Utf8)
            | (DataType::Utf8, DataType::LargeUtf8)
            | (DataType::LargeUtf8, DataType::Utf8View)
            | (DataType::Utf8View, DataType::LargeUtf8)
            | (DataType::Binary, DataType::BinaryView)
            | (DataType::BinaryView, DataType::Binary)
            | (DataType::LargeBinary, DataType::Binary)
            | (DataType::Binary, DataType::LargeBinary)
            | (DataType::LargeBinary, DataType::BinaryView)
            | (DataType::BinaryView, DataType::LargeBinary)
    )
}

pub(super) fn schema_has_float_column(schema: &Schema) -> bool {
    schema.fields().iter().any(|field| matches!(field.data_type(), DataType::Float32 | DataType::Float64))
}

/// Detect whether the given SQL has a top-level `ORDER BY` clause.
///
/// Unparseable SQL is conservatively treated as ordered. Ordering inside a
/// derived table, window, string or comment does not order the outer result.
pub fn sql_has_top_level_order_by(sql: &str) -> bool {
    parse_query(sql).is_none_or(|query| query.order_by.is_some() || !query.pipe_operators.is_empty())
}

/// Returns true when every output column is a top-level `ORDER BY` key.
///
/// Keys may refer to output names, projected expressions or 1-based positions.
/// Unresolved keys return false. Partial lists may still determine a total
/// order through uniqueness constraints, which are not available here.
pub fn sql_top_level_order_by_covers_output_columns(sql: &str, schema: &Schema) -> bool {
    parse_query(sql)
        .and_then(|query| order_by_keys(&query, schema))
        .is_some_and(|keys| (0..schema.fields().len()).all(|index| keys.iter().any(|key| key.column == index)))
}

fn parse_query(sql: &str) -> Option<Box<SqlQuery>> {
    let mut statements = Parser::parse_sql(&GenericDialect, sql).ok()?;
    if statements.len() != 1 {
        return None;
    }
    let Statement::Query(mut query) = statements.pop()? else {
        return None;
    };
    // Parentheses around the whole query preserve its result ordering.
    while query.order_by.is_none() && query.pipe_operators.is_empty() {
        let SetExpr::Query(inner) = query.body.as_ref() else {
            break;
        };
        query = inner.clone();
    }
    Some(query)
}

fn order_by_keys(query: &SqlQuery, schema: &Schema) -> Option<Vec<ParitySortKey>> {
    let order_by = query.order_by.as_ref()?;
    if order_by.interpolate.is_some() || !query.pipe_operators.is_empty() {
        return None;
    }
    let OrderByKind::Expressions(keys) = &order_by.kind else {
        return None;
    };
    if keys.is_empty() {
        return None;
    }
    keys.iter()
        .map(|key| {
            if key.with_fill.is_some() {
                return None;
            }
            resolve_order_by_column(&key.expr, query, schema)
                .map(|column| ParitySortKey { column, descending: !key.options.asc.unwrap_or(true) })
        })
        .collect()
}

fn resolve_order_by_column(expr: &SqlExpr, query: &SqlQuery, schema: &Schema) -> Option<usize> {
    match expr {
        SqlExpr::Value(ValueWithSpan { value: Value::Number(position, _), .. }) => {
            return position.parse::<usize>().ok()?.checked_sub(1).filter(|&index| index < schema.fields().len());
        },
        SqlExpr::Identifier(ident) => {
            let mut matches =
                schema.fields().iter().enumerate().filter(|(_, field)| identifier_matches(ident, field.name()));
            if let Some((index, _)) = matches.next() {
                return matches.next().is_none().then_some(index);
            }
        },
        _ => {},
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    // Wildcard expansion changes projection positions, so they cannot index the schema.
    if select.projection.len() != schema.fields().len()
        || select
            .projection
            .iter()
            .any(|item| !matches!(item, SelectItem::UnnamedExpr(_) | SelectItem::ExprWithAlias { .. }))
    {
        return None;
    }
    let unqualified = match expr {
        SqlExpr::CompoundIdentifier(parts) if parts.len() == 2 && is_single_table_qualifier(&parts[0], select) => {
            Some(SqlExpr::Identifier(parts[1].clone()))
        },
        _ => None,
    };
    select.projection.iter().position(|item| {
        let projected = match item {
            SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => expr,
            _ => return false,
        };
        projected == expr || unqualified.as_ref().is_some_and(|expr| projected == expr)
    })
}

fn identifier_matches(ident: &Ident, name: &str) -> bool {
    if ident.quote_style.is_some() { ident.value == name } else { ident.value.to_ascii_lowercase() == name }
}

fn is_single_table_qualifier(qualifier: &Ident, select: &Select) -> bool {
    let [table] = select.from.as_slice() else {
        return false;
    };
    if !table.joins.is_empty() {
        return false;
    }
    let TableFactor::Table { name, alias, .. } = &table.relation else {
        return false;
    };
    if let Some(alias) = alias {
        return qualifier == &alias.name;
    }
    name.0.last().and_then(|part| part.as_ident()).is_some_and(|name| name == qualifier)
}
